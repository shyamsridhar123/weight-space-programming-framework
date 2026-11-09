"""Sequence obligations, authenticated replay, and the evaluation boundary."""
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from rlcd.weight_space import cumulative_research as cumulative
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.features import ReadoutHead, certify_feature_retention
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.program import difference


def rehash(record):
    record["sha256"] = digest({key: value for key, value in record.items() if key != "sha256"})
    return record


def repeated_insertions():
    base = TensorArtifact("repeated-affine/1", {"w": [[0.], [.3]], "b": [2., 0.]}, {"feature_id": "repeat/1"})
    head = ReadoutHead("w", "b", "repeat/1")
    arrays, artifacts = {}, {base.sha256: base}
    baseline = cumulative._add_array(arrays, np.array([[0.]], dtype=np.float32))
    labels = cumulative._add_array(arrays, np.array([0], dtype=np.int64))
    target = cumulative._add_array(arrays, np.array([[8.]], dtype=np.float32))
    order = [dict(subject="same", case_id=f"request-{step}", target_label=1, desired_code=1,
                  desired_queue=1, features=target) for step in range(3)]
    # Designed trust-boundary case, not the trained study's minimum-gain policy.
    policy = cumulative._policy(.5) | {"minimum_new_target_gain": 0.}
    def proposer(parent, _):
        weight = parent.tensors["w"].copy()
        weight[1, 0] += .05
        candidate = parent.replace(w=weight)
        return {"artifact": candidate, "program": difference(parent, candidate, "Repeated insertion falsifier")}
    sequence = cumulative._sequence(base, order, head, [0, 1], baseline, labels, policy,
                                    "cumulative", artifacts, arrays, proposer)
    return sequence, artifacts, arrays


def test_locally_passing_chain_exhausts_original_budget_and_erases_an_old_target():
    sequences, artifacts, arrays = cumulative.boundary_sequences()
    assert [row["accepted"] for row in sequences["local"]["records"]] == [True, True]
    assert [row["accepted"] for row in sequences["cumulative"]["records"]] == [True, False]
    last = sequences["cumulative"]["records"][-1]
    gates = {row["name"]: row for row in last["gates"]}
    assert gates["local_score_budget"]["status"] == gates["local_protected_behavior"]["status"] == "pass"
    assert gates["original_score_budget"]["status"] == "observed_violation"
    assert Fraction(gates["original_score_budget"]["evidence"]["receipt"]["maximum_exact"]) == Fraction(3, 4)
    assert any(row["name"].startswith("retained/") and row["status"] == "observed_violation" for row in last["gates"])
    assert not sequences["conflict"]["records"][-1]["accepted"]
    assert any(row["status"] == "conflict_rejection" for row in sequences["conflict"]["records"][-1]["gates"])
    for sequence in sequences.values():
        assert cumulative.verify_sequence(sequence, artifacts, arrays, sequence["contract_sha256"])
        assert sequence["rollback"]["restored"] and not sequence["rollback"]["external_state_restored"]
    request = last["request"]
    base, candidate = (artifacts[request[key]] for key in ("original_sha256", "candidate_sha256"))
    head = ReadoutHead(**request["head"])
    exact = certify_feature_retention(base, candidate, head, arrays[request["baseline_features"]])
    assert exact["maximum_residuals_exact"] == gates["original_score_budget"]["evidence"]["receipt"]["maximum_residuals_exact"]


def test_repeated_subject_keeps_all_insertions_and_cannot_reset_an_older_budget():
    sequence, artifacts, arrays = repeated_insertions()
    assert [row["accepted"] for row in sequence["records"]] == [True, True, False]
    last = sequence["records"][-1]
    assert len(last["request"]["retained"]) == 2
    gates = [row for row in last["gates"] if row["name"].startswith("retained/")]
    assert [row["evidence"]["memory_age_steps"] for row in gates] == [2, 1]
    assert [row["status"] for row in gates] == ["observed_violation", "pass"]
    altered = deepcopy(sequence)
    raw = altered["records"][-1]["request"]
    raw["retained"] = raw["retained"][1:]
    repaired = cumulative.admit_step(cumulative.CumulativeRequest.capture(raw), artifacts, arrays)
    assert repaired["accepted"]  # Removing the old obligation really changes admission.
    altered["records"][-1] = repaired
    assert not cumulative.verify_sequence(altered, artifacts, arrays, sequence["contract_sha256"])


def test_captured_request_and_artifact_pools_remain_independently_bound():
    sequences, artifacts, arrays = cumulative.boundary_sequences()
    record = sequences["cumulative"]["records"][0]
    raw = deepcopy(record["request"])
    frozen = cumulative.CumulativeRequest.capture(raw)
    raw["target"]["subject"] = "changed"
    raw["policy"]["score_budget"] = 99.
    assert frozen.sha256 == record["request_sha256"]
    with pytest.raises(AttributeError):
        frozen._bytes = b"{}"
    detached = frozen.to_dict()
    detached["target"]["desired_queue"] = 0
    changed = cumulative.CumulativeRequest.capture(detached)
    changed_record = cumulative.admit_step(changed, artifacts, arrays)
    assert not changed_record["accepted"]
    assert not cumulative.verify_step(changed_record, artifacts, arrays, frozen.sha256)
    assert cumulative.verify_step(changed_record, artifacts, arrays, changed.sha256)
    key = frozen.to_dict()["baseline_features"]
    bad_arrays = dict(arrays)
    bad_arrays[key] = arrays[key] + 1
    assert not cumulative.admit_step(frozen, artifacts, bad_arrays)["accepted"]
    bad_artifacts = dict(artifacts)
    candidate_key = frozen.to_dict()["candidate_sha256"]
    bad_artifacts[candidate_key] = artifacts[candidate_key].replace(b=[2., .5])
    assert not cumulative.admit_step(frozen, bad_artifacts, arrays)["accepted"]


@pytest.mark.parametrize("field,value", [("case_id", ""), ("subject", " "), ("target_label", True),
    ("target_label", 2), ("desired_code", False), ("desired_queue", "1")])
def test_invalid_goal_types_and_mapping_are_rejected(field, value):
    sequences, artifacts, arrays = cumulative.boundary_sequences()
    raw = deepcopy(sequences["cumulative"]["records"][0]["request"])
    raw["target"][field] = value
    assert not cumulative.admit_step(cumulative.CumulativeRequest.capture(raw), artifacts, arrays)["accepted"]


def test_policy_gate_tampering_and_native_scope_changes_fail_replay(monkeypatch):
    sequences, artifacts, arrays = cumulative.boundary_sequences()
    record = sequences["cumulative"]["records"][0]
    for mutation in ("rate", "operator", "head", "program", "gate"):
        changed = deepcopy(record)
        if mutation == "gate":
            changed["gates"][0]["required"] = False
            rehash(changed)
        else:
            raw = changed["request"]
            if mutation == "rate": raw["policy"]["maximum_old_target_loss"] = 1.01
            elif mutation == "operator": raw["execution"]["operator"] = "attention"
            elif mutation == "head": raw["head"]["feature_id"] = "wrong-features"
            else: raw["program"]["base_sha256"] = "0"*64
            changed = cumulative.admit_step(cumulative.CumulativeRequest.capture(raw), artifacts, arrays)
            assert not changed["accepted"]
        assert not cumulative.verify_step(changed, artifacts, arrays, record["request_sha256"])
    actual = cumulative.numerical_environment
    monkeypatch.setattr(cumulative, "numerical_environment", lambda backend: actual(backend) | {"kernel": "different"})
    mismatched = cumulative.admit_step(cumulative.CumulativeRequest.capture(record["request"]), artifacts, arrays)
    assert not mismatched["accepted"]
    assert any(row["status"] == "missing_correspondence" for row in mismatched["gates"])


def test_frozen_features_reject_encoder_mutation_even_when_projected_head_is_unchanged():
    sequences, _, arrays = cumulative.boundary_sequences()
    raw = deepcopy(sequences["cumulative"]["records"][0]["request"])
    base = TensorArtifact("captured-encoder/1", {"w": [[0.], [0.]], "b": [2., 0.], "encoder": [1.]},
                          {"feature_id": "designed/1"})
    candidate = base.replace(encoder=[2.])
    raw.update(original_sha256=base.sha256, parent_sha256=base.sha256, candidate_sha256=candidate.sha256,
               program=difference(base, candidate, "Forbidden encoder mutation").to_dict())
    artifacts = {base.sha256: base, candidate.sha256: candidate}
    record = cumulative.admit_step(cumulative.CumulativeRequest.capture(raw), artifacts, arrays)
    assert not record["accepted"]
    with pytest.raises(ValueError, match="outside"):
        cumulative.exact_sample_drift(base, candidate, ReadoutHead(**raw["head"]), arrays[raw["baseline_features"]])


def test_failed_and_invalid_proposals_keep_all_attempts_and_last_accepted_parent():
    sequence, artifacts, arrays = repeated_insertions()
    contract = sequence["contract"]
    def failed(parent, target):
        if target["case_id"] == "request-0": raise RuntimeError("Designed fitting failure")
        return {"artifact": None, "program": None}
    failed_sequence = cumulative._sequence(artifacts[contract["original_sha256"]], contract["order"],
        ReadoutHead(**contract["head"]), contract["queue_map"], contract["baseline_features"], contract["baseline_labels"],
        contract["policy"], "cumulative", artifacts, arrays, failed)
    assert len(failed_sequence["records"]) == len(contract["order"])
    assert all(not row["accepted"] and row["no_op"] for row in failed_sequence["records"])
    assert failed_sequence["endpoint_sha256"] == contract["original_sha256"]
    assert cumulative.verify_sequence(failed_sequence, artifacts, arrays, failed_sequence["contract_sha256"])


def test_utility_matrix_keeps_each_insertion_and_reports_only_named_rollback_state():
    sequence, artifacts, arrays = repeated_insertions()
    bundle = {"test_features": np.array([[0.], [8.]], dtype=np.float32), "test_labels": np.array([0, 1]),
              "test_novel_mask": np.array([True, True])}
    result = cumulative._evaluate_sequence(sequence, artifacts, bundle, True)
    memories = result["matrix"][-1]["target_matrix"][0]["retained_insertions"]
    assert [memory["accepted_step"] for memory in memories] == [1, 2]
    assert [memory["memory_age_steps"] for memory in memories] == [2, 1]
    assert all(memory["loss_from_insertion"] == 0 for memory in memories)
    assert cumulative.verify_sequence(sequence, artifacts, arrays, sequence["contract_sha256"])
    changed = deepcopy(sequence)
    changed["rollback"]["external_state_restored"] = True
    assert not cumulative.verify_sequence(changed, artifacts, arrays, sequence["contract_sha256"])


def test_missing_evaluation_files_remain_in_the_assigned_denominator(tmp_path, monkeypatch):
    sequences, artifacts, _ = cumulative.boundary_sequences()
    assigned = []
    for mode in ("local", "cumulative"):
        assigned.append(sequences[mode] | {"lineage_id": "missing-main-full-1", "domain": "missing",
                                          "order_name": "forward", "lineage_directory": "missing"})
    monkeypatch.setattr(cumulative, "ROOT", tmp_path)
    result = cumulative._evaluate(assigned, artifacts, {"domains": {"missing": {}}, "controllers": ["local", "cumulative"]})
    assert len(result["outcomes"]) == len(assigned)
    assert all(row["assigned"] and row["primary"]["missing"] for row in result["outcomes"])
    assert all(row["missing_sequences"] == 1 for row in result["summaries"])
    pair = result["paired_contrasts"][0]["lineages"][0]
    assert pair["target_gain_missing_bounds"] == [-2., 2.]
    assert pair["useful_missing_bounds"] == [-1., 1.]


def test_candidate_seal_precedes_any_evaluation_and_is_not_overwritten(tmp_path, monkeypatch):
    from rlcd.weight_space.research_tasks import ARCHITECTURE, HEAD_WEIGHT, HEAD_BIAS
    base = TensorArtifact(ARCHITECTURE, {HEAD_WEIGHT: [[0.], [.25]], HEAD_BIAS: [2., 0.]},
        {"feature_id": "tiny/1", "encoder_sha256": "e"*64, "preprocessing_sha256": "d"*64})
    bundle = {"queue_map": np.array([0, 1]), "calibration_ids": np.array(["c0", "c1"]),
        "calibration_labels": np.array([0, 1]), "calibration_features": np.array([[0.], [8.]], dtype=np.float32),
        "target_labels": np.array([1]), "target_features": np.array([[8.]], dtype=np.float32),
        "protected_ids": np.array(["p0"]), "protected_labels": np.array([0]),
        "protected_features": np.array([[0.]], dtype=np.float32)}
    protocol = {"domains": {"tiny": {"study": "tiny", "main_seeds": [1],
        "cases": [dict(case_id="tiny-repair", target_label=1, desired_code=1, desired_queue=1)]}},
        "orders": ["forward", "reverse"], "controllers": ["local", "cumulative"], "input_hashes": {}}
    monkeypatch.setattr(cumulative, "_protocol", lambda: deepcopy(protocol))
    monkeypatch.setattr(cumulative, "_safe_lineage", lambda path: {"id": "tiny-main-full-1", "base": base, "features": bundle})
    def compile_one(parent, *args, **kwargs):
        candidate = parent.replace(**{HEAD_WEIGHT: [[0.], [.375]]})
        return {"artifact": candidate, "program": difference(parent, candidate, "Tiny seal check")}
    monkeypatch.setattr(cumulative, "compile_arm", compile_one)
    out = tmp_path / "frozen"
    calls = []
    def evaluate_after_seal(sequences, artifacts, frozen_protocol):
        assert (out / "global-seal.json").is_file() and (out / "manifest.json").is_file()
        assert len(sequences) == 4 and frozen_protocol == protocol
        calls.append(True)
        return {"outcomes": [], "designation": "Designed staging test only"}
    monkeypatch.setattr(cumulative, "_evaluate", evaluate_after_seal)
    cumulative.seal(out)
    assert not calls and not (out / "results.json").exists()
    manifest = (out / "manifest.json").read_bytes()
    assert cumulative.verify_seal(out) and not calls
    cumulative.evaluate(out)
    assert calls and (out / "manifest.json").read_bytes() == manifest
    assert cumulative.verify(out)
    with pytest.raises(FileExistsError): cumulative.seal(out)
    with pytest.raises(FileExistsError): cumulative.evaluate(out)
