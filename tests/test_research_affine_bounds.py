"""Exact extrema and independent rejection checks for the narrow affine subset."""
from copy import deepcopy
from fractions import Fraction
from itertools import product
import json

import numpy as np
import pytest

from rlcd.weight_space import bound_research as bounds
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.model import InputBox, digest
from rlcd.weight_space.research_tasks import HEAD_BIAS, HEAD_WEIGHT, make_head


def endpoints():
    trained = {"head": {"weight": np.zeros((2, 3), dtype=np.float32),
                        "bias": np.zeros(2, dtype=np.float32)},
               "encoder": {"weight": np.eye(3, dtype=np.float32), "bias": np.zeros(3, dtype=np.float32)},
               "ancestry": "Independent test-only affine fixtures; no task impact evidence"}
    base = make_head(trained, digest({"test": "fixed preprocessing"}), ["a", "b"], 1, "test")
    candidate = base.replace(**{HEAD_WEIGHT: [[1., -1., 0.], [0., .5, -.25], [0., 0., 0.]],
                               HEAD_BIAS: [2., -1., -64.]})
    return base, candidate, InputBox([2., 3., -1.], [4., 5., 2.])


def test_extrema_equal_exhaustive_corners_and_tighten_triangle():
    base, candidate, box = endpoints()
    receipt = bounds.certify_affine_box(base, candidate, box)
    assert bounds.verify_affine_box(base, candidate, box, receipt)
    corners = list(product(*zip(box.lower, box.upper)))
    for output, interval in enumerate(receipt["intervals"]):
        q = lambda value: Fraction.from_float(float(value))
        changes = []
        for corner in corners:
            changes.append(q(candidate.tensors[HEAD_BIAS][output])-q(base.tensors[HEAD_BIAS][output])
                           +sum(((q(candidate.tensors[HEAD_WEIGHT][output, j])-q(base.tensors[HEAD_WEIGHT][output, j]))
                                 *q(value) for j, value in enumerate(corner)), Fraction(0)))
        assert Fraction(interval["lower_exact"]) == min(changes)
        assert Fraction(interval["upper_exact"]) == max(changes)
    triangle = bounds.triangle_certificate(base, candidate, box)
    assert max(map(Fraction, triangle["output_bounds"])) == 11
    assert Fraction(receipt["maximum_exact"]) == 3
    assert bounds.check_affine_budget(base, candidate, box, receipt, 3.)
    assert not bounds.check_affine_budget(base, candidate, box, receipt, 2.99)
    assert not bounds.check_affine_budget(base, candidate, box, receipt, -1.)


def test_checker_does_not_call_generator(monkeypatch):
    base, candidate, box = endpoints()
    receipt = bounds.certify_affine_box(base, candidate, box)
    monkeypatch.setattr(bounds, "certify_affine_box", lambda *args: (_ for _ in ()).throw(AssertionError("generator used")))
    assert bounds.verify_affine_box(base, candidate, box, receipt)


@pytest.mark.parametrize("mutation", ["bound", "corner", "identity", "flag", "unknown", "noncanonical", "interval_omission"])
def test_rehashed_certificate_mutations_fail(mutation):
    base, candidate, box = endpoints()
    receipt = deepcopy(bounds.certify_affine_box(base, candidate, box))
    if mutation == "bound":
        receipt["intervals"][0]["absolute_bound_exact"] = "2"
        receipt["maximum_exact"] = "2"
    elif mutation == "corner":
        receipt["intervals"][0]["lower_corner_bits"] = "110"
    elif mutation == "identity":
        receipt["base_artifact_sha256"] = "0"*64
    elif mutation == "flag":
        receipt["floating_execution_verified"] = 0
    elif mutation == "unknown":
        receipt["attention_certified"] = True
    elif mutation == "noncanonical":
        receipt["intervals"][0]["lower_exact"] = "-2/2"
    else:
        receipt["intervals"].pop()
    receipt["sha256"] = digest({key: value for key, value in receipt.items() if key != "sha256"})
    assert not bounds.verify_affine_box(base, candidate, box, receipt)


@pytest.mark.parametrize("mutation", ["attention", "normalization", "stochastic", "feature_identity", "native_dtype", "native_rounding", "feature_order"])
def test_unsupported_semantics_do_not_fall_back_to_affine_shape(mutation):
    base, candidate, box = endpoints()
    metadata, tensors, architecture = base.metadata, dict(base.tensors), base.architecture
    if mutation == "attention":
        architecture = "transformer-attention/1"
    elif mutation == "normalization":
        tensors["normalization.running_mean"] = np.zeros(3)
    elif mutation == "stochastic":
        metadata["head_semantics"] = "stochastic affine readout with dropout"
    elif mutation == "feature_identity":
        metadata["feature_id"] = "0"*64
    elif mutation == "native_dtype":
        metadata["native_dtype"] = "float16"
    elif mutation == "native_rounding":
        tensors[HEAD_WEIGHT] = [[.1, 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    else:
        metadata["feature_names"].reverse()
    unsupported = TensorArtifact(architecture, tensors, metadata)
    with pytest.raises(ValueError):
        bounds.certify_affine_box(unsupported, unsupported, box)
    assert not bounds.verify_affine_box(unsupported, candidate, box, {})


def test_exact_endpoint_subtraction_and_noop_do_not_make_float_claims():
    base, _, box = endpoints()
    no_op = bounds.certify_affine_box(base, base, box)
    assert no_op["maximum_exact"] == "0"
    assert bounds.verify_affine_box(base, base, box, no_op)
    maximum = float(np.finfo(np.float32).max)
    before = base.replace(**{HEAD_WEIGHT: [[-maximum, 0., 0.], [0., 0., 0.], [0., 0., 0.]]})
    after = base.replace(**{HEAD_WEIGHT: [[maximum, 0., 0.], [0., 0., 0.], [0., 0., 0.]]})
    narrow = InputBox([.5, 0., 0.], [.5, 0., 0.])
    receipt = bounds.certify_affine_box(before, after, narrow)
    assert Fraction(receipt["maximum_exact"]) == Fraction.from_float(maximum)
    assert bounds.verify_affine_box(before, after, narrow, receipt)
    assert receipt["floating_execution_verified"] is False
    assert receipt["implementation_machine_checked"] is False
    assert receipt["feature_extractor_verified"] is False
    assert receipt["core_native_admission_integrated"] is False
    registry = bounds.affine_registry()
    assert registry.verify_certificate(before, after, narrow, registry.certify(before, after, narrow))
    with pytest.raises(ValueError):
        bounds.certify_affine_box(base, base, InputBox([0.], [1.]))


def test_independent_integer_observation_matches_direct_rational_endpoints(monkeypatch):
    base, candidate, _ = endpoints()
    features = np.array([[2., 3., -1.], [4., 5., 2.], [.5, -.25, .125]], dtype=np.float32)
    monkeypatch.setattr(bounds, "_endpoint_delta", lambda *args: (_ for _ in ()).throw(AssertionError("shared delta used")))
    monkeypatch.setattr(bounds, "certify_affine_box", lambda *args: (_ for _ in ()).throw(AssertionError("generator used")))
    residuals, denominator = bounds._real_residuals(base, candidate, features)
    q = lambda value: Fraction.from_float(float(value))
    for row, values in zip(residuals, features):
        for output, observed in enumerate(row):
            expected = q(candidate.tensors[HEAD_BIAS][output])-q(base.tensors[HEAD_BIAS][output])
            expected += sum(((q(candidate.tensors[HEAD_WEIGHT][output, column])-q(base.tensors[HEAD_WEIGHT][output, column]))
                             *q(value) for column, value in enumerate(values)), Fraction(0))
            assert Fraction(int(observed), denominator) == expected


def test_native_outside_box_exceedance_has_no_real_certificate_premise():
    base, candidate, box = endpoints()
    pair = bounds._make_pair(base, candidate, box)
    bundle = {"test_features": np.array([[2., 5., -1.], [4., 3., 2.], [2., 99., -1.]], dtype=np.float32),
        "test_labels": np.array([1, 0, 1]), "test_novel_mask": np.array([True, True, False]),
        "test_ids": np.array(["a", "b", "c"]), "test_group_ids": np.array(["a", "b", "c"])}
    observation = bounds._pair_observation(base, candidate, box, pair, bundle, Fraction(3))
    primary, secondary = (observation["observations"][name] for name in ("primary_novel", "secondary_official"))
    assert primary["in_box_rows"] == 2 and primary["out_of_box_rows"] == 0
    assert primary["in_box"]["exact_real"]["signed_bound_violation_rows"] == 0
    assert secondary["out_of_box_rows"] == 1
    outside = secondary["out_of_box"]
    assert outside["native"]["signed_bound_exceedance_rows"] == 1
    assert outside["native"]["certificate_violation_claim"] is False
    assert outside["real_certificate_premise"] is False and outside["real_certificate_violation_claim"] is False


def test_generator_categories_are_counted_without_exhaustion():
    assert bounds._counts((name for name in ("unsupported", "pass", "unsupported")), ("pass", "unsupported")) == {"pass": 1, "unsupported": 2}


@pytest.mark.parametrize("measurement", ["negative_time", "integer_time", "boolean_peak", "unknown_scope"])
def test_paired_measurement_replay_rejects_changed_scope_or_types(measurement):
    base, candidate, box = endpoints()
    pair = bounds._make_pair(base, candidate, box)
    if measurement == "unknown_scope":
        pair["measurements"]["kernel_rss"] = {"seconds": .1, "python_peak_bytes": 1}
    else:
        key = "python_peak_bytes" if measurement == "boolean_peak" else "seconds"
        pair["measurements"]["signed_generation"][key] = {"negative_time": -.1, "integer_time": 0, "boolean_peak": True}[measurement]
    with pytest.raises(ValueError):
        bounds._check_pair(base, candidate, box, pair)


def signed(value):
    return value | {"sha256": digest(value)}


def replace_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def release_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(bounds, "ROOT", tmp_path)
    assignments, accounts = [], []
    statuses = sorted(bounds.TERMINAL_STATUSES)
    for index in range(bounds.GLOBAL_ASSIGNED_TOTAL):
        identity = f"{index:024x}"
        selected = signed({"schema": "wsp-live-selection/1", "assignment_id": identity,
            "terminal_status": statuses[index % len(statuses)], "evaluation_started": False})
        bounds.write_json(tmp_path/"artifacts/research/live-studies/selections"/identity/"selection.json", selected)
        assignments.append({"assignment_id": identity})
        accounts.append({"assignment_id": identity, "selection_sha256": selected["sha256"], "terminal_status": selected["terminal_status"]})
    lock = signed({"schema": "wsp-live-preparation-lock/1", "assignments": assignments,
        "assigned_total": bounds.GLOBAL_ASSIGNED_TOTAL, "selection_only": True, "evaluation_started": False,
        "provider_calls_started": False, "protocol_sha256": {"test.json": "a"*64}})
    lock_path = tmp_path/bounds.GLOBAL_LOCK_PATH
    bounds.write_json(lock_path, lock)
    gate = signed({"schema": "wsp-live-global-selection-seal/1", "phase": "selection_complete_before_evaluation",
        "selection_frozen": True, "all_assigned_accounted": True, "studies": ["blind-diagnosis", "weight-access"],
        "prepared_lock_path": bounds.GLOBAL_LOCK_PATH, "prepared_lock_sha256": bounds.file_hash(lock_path),
        "protocol_sha256": lock["protocol_sha256"], "assigned_total": bounds.GLOBAL_ASSIGNED_TOTAL,
        "terminal_total": len(accounts), "all_terminal": True, "evaluation_started": False,
        "accounting": accounts, "sealed_utc": "Synthetic test fixture only"})
    path = tmp_path/bounds.GLOBAL_GATE_PATH
    bounds.write_json(path, gate)
    return path, gate


@pytest.mark.parametrize("mutation", ["unreleased", "wrong_external_hash", "duplicate", "missing_account", "selection_rehash", "lock_rehash", "alternate_path"])
def test_global_release_requires_external_identity_and_all_terminal_accounts(tmp_path, monkeypatch, mutation):
    path, gate = release_fixture(tmp_path, monkeypatch)
    assert bounds._global_release(path, gate["sha256"])["canonical_sha256"] == gate["sha256"]
    expected = gate["sha256"]
    if mutation == "wrong_external_hash":
        expected = "0"*64
    elif mutation == "alternate_path":
        path = tmp_path/"alternate.json"
        bounds.write_json(path, gate)
    elif mutation in ("selection_rehash", "lock_rehash"):
        changed_path = (tmp_path/bounds.GLOBAL_LOCK_PATH if mutation == "lock_rehash"
            else tmp_path/"artifacts/research/live-studies/selections"/gate["accounting"][0]["assignment_id"]/"selection.json")
        changed = bounds.read_json(changed_path)
        changed.pop("sha256")
        changed["extra"] = "Changed but internally rehashed"
        replace_json(changed_path, signed(changed))
    else:
        changed = deepcopy(gate)
        changed.pop("sha256")
        if mutation == "unreleased":
            changed["selection_frozen"] = False
        elif mutation == "duplicate":
            changed["accounting"][-1] = changed["accounting"][0]
        else:
            changed["accounting"].pop()
        changed = signed(changed)
        replace_json(path, changed)
        expected = changed["sha256"]
    with pytest.raises(ValueError):
        bounds._global_release(path, expected)


def staged_fixture(tmp_path, monkeypatch):
    base, candidate, _ = endpoints()
    features = np.array([[2., 3., -1.], [4., 5., 2.]], dtype=np.float32)
    target = {"case_id": "tiny-target", "target_label": 1, "desired_queue": 1}
    policy = {"score_budget": 3., "minimum_new_target_gain": .10,
              "maximum_local_net_harm": .01, "maximum_local_decision_flip": .01}
    records = []
    for step, endpoint in enumerate((candidate, candidate, base), start=1):
        request = {"parent_sha256": base.sha256, "candidate_sha256": endpoint.sha256,
            "baseline_features": "calibration", "mode": "local", "step": step, "target": target,
            "policy": policy, "proposal_error": "Designed fitting failure" if step == 3 else None}
        records.append({"request": request, "request_sha256": digest({"synthetic_step": step}),
            "no_op": step == 3, "accepted": False, "deployed_sha256": base.sha256,
            "gates": [{"name": "local_score_budget", "evidence": {"receipt": {"maximum_exact": "0" if step == 3 else "1"}}},
                {"name": "new_target_gain", "status": "observed_violation" if step == 3 else "pass"}]})
    sequence = {"lineage_id": "test-full-1", "domain": "test", "order_name": "forward", "records": records,
        "lineage_directory": "evaluation-lineage", "contract": {"head": bounds._trained_head(base).to_dict(),
            "queue_map": [0, 1, 2], "order": [target]}}
    artifacts = {base.sha256: base, candidate.sha256: candidate}
    monkeypatch.setattr(bounds, "_inputs", lambda: ([sequence], artifacts, {"calibration": features}))
    monkeypatch.setattr(bounds, "ROOT", tmp_path)
    monkeypatch.setattr(bounds, "SOURCE_FILES", ("bound_research.py",))
    monkeypatch.setattr(bounds.cumulative, "SOURCE_FILES", ())
    monkeypatch.setattr(bounds, "CUMULATIVE_ROOT", tmp_path/"cumulative")
    bounds.write_json(bounds.CUMULATIVE_ROOT/"sources.json", {})
    (tmp_path/"bound_research.py").write_text("# Synthetic source closure only\n", encoding="utf-8")
    bounds.write_json(tmp_path/"docs/weight-space/v2/research-sources-2026-11.json",
        {"sources": [{"id": key, "sha256": value} for key, value in bounds.REFERENCE_HASHES.items()]})
    profile = tmp_path/"profile"
    bounds.profile(profile)
    monkeypatch.setattr(bounds, "PROFILE_ROOT", profile)
    monkeypatch.setattr(bounds, "PROFILE_MANIFEST", bounds.file_hash(profile/"manifest.json"))
    out = tmp_path/"sealed"
    result = bounds.seal(out)
    return out, result["manifest_sha256"], sequence, base


def test_staged_comparison_keeps_duplicates_failures_and_external_source_identity(tmp_path, monkeypatch):
    out, manifest, _, _ = staged_fixture(tmp_path, monkeypatch)
    assert bounds.verify_seal(out, manifest)
    comparison = bounds.read_json(out/"comparison.json")
    assert [row["timing_reused_for_duplicate_pair"] for row in comparison["rows"]] == [False, True, False]
    summary = comparison["summary"]["domains"]["test"]
    assert summary["baseline_registry"]["unsupported"] == 3
    assert summary["signed_registry"]["pass"] == 3
    assert summary["no_op_records"] == summary["proposal_failure_records"] == 1
    assert comparison["heldout_feature_arrays_read"] is False and not (out/"results.json").exists()
    with pytest.raises(FileExistsError): bounds.seal(out)
    with pytest.raises(ValueError): bounds.verify_seal(out, "0"*64)
    (tmp_path/"bound_research.py").write_text("# Changed source\n", encoding="utf-8")
    with pytest.raises(ValueError): bounds.verify_seal(out, manifest)


def test_failed_release_precedes_any_source_or_evaluation_read(tmp_path, monkeypatch):
    def rejected(*args):
        raise ValueError("Designed release rejection")
    def forbidden(*args):
        raise AssertionError("A pre-release dependency was read")
    monkeypatch.setattr(bounds, "_global_release", rejected)
    monkeypatch.setattr(bounds, "_verify_seal", forbidden)
    monkeypatch.setattr(bounds, "_evaluate", forbidden)
    for action in (bounds.evaluate, bounds.verify):
        with pytest.raises(ValueError):
            action(tmp_path, "a"*64, bounds.GLOBAL_GATE_PATH, "b"*64)
    assert not (tmp_path/"results.json").exists()


def test_released_toy_evaluation_and_replay_preserve_candidate_manifest(tmp_path, monkeypatch):
    out, manifest, sequence, base = staged_fixture(tmp_path, monkeypatch)
    path, gate = release_fixture(tmp_path, monkeypatch)
    bundle = {"test_features": np.array([[2., 5., -1.], [4., 3., 2.], [2., 99., -1.]], dtype=np.float32),
        "test_labels": np.array([1, 0, 1]), "test_novel_mask": np.array([True, True, False]),
        "test_ids": np.array(["a", "b", "c"]), "test_group_ids": np.array(["a", "b", "c"])}
    directory = tmp_path/sequence["lineage_directory"]
    directory.mkdir()
    np.savez_compressed(directory/"evaluation-features.npz", **bundle)
    bounds.write_json(directory/"evaluation-features.json", {"feature_id": bounds._trained_head(base).feature_id,
        "sha256": bounds.file_hash(directory/"evaluation-features.npz"), "bytes": (directory/"evaluation-features.npz").stat().st_size})
    original = (out/"manifest.json").read_bytes()
    result = bounds.evaluate(out, manifest, path, gate["sha256"])
    assert len(result["outcomes"]) == 3
    assert result["outcomes"][0]["primary_novel"]["utility_pass"]
    assert result["outcomes"][0]["primary_novel"]["signed_available_and_utility_pass"]
    assert not result["outcomes"][0]["primary_novel"]["triangle_available_and_utility_pass"]
    assert not result["outcomes"][2]["primary_novel"]["utility_pass"]
    assert bounds.verify(out, manifest, path, gate["sha256"])
    assert (out/"manifest.json").read_bytes() == original
    with pytest.raises(FileExistsError): bounds.evaluate(out, manifest, path, gate["sha256"])
    changed = bounds.read_json(out/"results.json")
    changed["outcomes"][0]["primary_novel"]["target_gain"] = .5
    replace_json(out/"results.json", changed)
    evaluation_manifest = bounds.read_json(out/"evaluation-manifest.json")
    evaluation_manifest["files"]["results.json"] = bounds.file_hash(out/"results.json")
    replace_json(out/"evaluation-manifest.json", evaluation_manifest)
    with pytest.raises(ValueError, match="failed replay"):
        bounds.verify(out, manifest, path, gate["sha256"])


def test_missing_toy_heldout_remains_in_assigned_lineage_sensitivity(tmp_path, monkeypatch):
    out, manifest, _, _ = staged_fixture(tmp_path, monkeypatch)
    path, gate = release_fixture(tmp_path, monkeypatch)
    result = bounds.evaluate(out, manifest, path, gate["sha256"])
    assert len(result["outcomes"]) == 3 and all(row["primary_novel"]["missing"] for row in result["outcomes"])
    lineage = result["summaries"][0]["lineages"][0]
    assert lineage["assigned_records"] == lineage["missing_records"] == 3
    assert lineage["gain_missing_bounds"] == lineage["harm_missing_bounds"] == [-1., 1.]
    assert lineage["useful_missing_bounds"] == [0., 1.]
    assert bounds.verify(out, manifest, path, gate["sha256"])


@pytest.mark.parametrize("arguments", [["evaluate", "unused"], ["verify", "unused", "--manifest-sha256", "a"*64], ["run", "unused"]])
def test_cli_rejects_missing_external_release_or_ungated_run(arguments):
    with pytest.raises(SystemExit) as failure:
        bounds.main(arguments)
    assert failure.value.code == 2
