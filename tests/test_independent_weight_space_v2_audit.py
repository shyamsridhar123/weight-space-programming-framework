"""Adversarial fixtures for the independent auditor; never generate study worlds."""
from __future__ import annotations

import copy
from fractions import Fraction
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("independent_weight_space_v2_audit", ROOT / "scripts/audit_weight_space_v2.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@pytest.mark.parametrize("value", [True, 1.0, -1, "1", np.int64(1)])
def test_counts_reject_noninteger_or_negative_values(value):
    with pytest.raises(ValueError):
        audit.count(value, "counter")


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1", 2**53 + 1])
def test_numeric_inputs_reject_ambiguous_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        audit.number(value)


@pytest.mark.parametrize("text", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', '{"x": 1e1000}'])
def test_json_rejects_duplicate_keys_and_nonfinite_literals(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        audit.read_json(path)


@pytest.mark.parametrize("path", ["../escape.json", "nested/../escape.json", "./a.json", "a//b.json", "C:/a.json", "a\\b.json"])
def test_evidence_paths_are_canonical_and_contained(tmp_path, path):
    with pytest.raises(ValueError):
        audit.contained(tmp_path, path)


def model():
    return {"schema": "wsp-tensors/1", "dtype": "float64", "architecture": audit.SEQUENTIAL,
        "metadata": {"layers": [{"weight": "w1", "bias": "b1", "activation": "relu"},
                                 {"weight": "w2", "bias": "b2", "activation": "identity"}],
                     "input_names": ["x"], "output_names": ["y"]},
        "tensors": {"w1": [[1.], [-1.]], "b1": [0., 0.], "w2": [[2., 3.]], "b2": [.5]}}


def edited():
    result = model()
    result["tensors"]["w2"] = [[3., 2.5]]
    result["tensors"]["b2"] = [.75]
    return result


def literal():
    return {"schema": "wsp-program/1", "base_sha256": audit.digest(model()), "intent": "Fixture edit",
        "edits": [{"tensor": "w2", "index": [0, 0], "before": 2., "after": 3.},
                  {"tensor": "w2", "index": [0, 1], "before": 3., "after": 2.5},
                  {"tensor": "b2", "index": [0], "before": .5, "after": .75}]}


def test_literal_replay_has_exact_endpoint_and_preserves_input():
    base = model()
    assert audit.literal_endpoint(base, literal()) == edited()
    assert base == model()
    assert audit.literal_difference(base, edited(), "Fixture edit") == literal()


@pytest.mark.parametrize("mutation", ["parent", "prevalue", "duplicate", "bool_coordinate", "unchanged", "out_of_bounds"])
def test_literal_rejects_stale_malformed_or_ambiguous_edits(mutation):
    program = literal()
    if mutation == "parent":
        program["base_sha256"] = "0" * 64
    elif mutation == "prevalue":
        program["edits"][0]["before"] = 1.
    elif mutation == "duplicate":
        program["edits"].append(copy.deepcopy(program["edits"][0]))
    elif mutation == "bool_coordinate":
        program["edits"][0]["index"] = [False, 0]
    elif mutation == "unchanged":
        program["edits"][0]["after"] = 2.
    else:
        program["edits"][0]["index"] = [1, 0]
    with pytest.raises(ValueError):
        audit.literal_endpoint(model(), program)


def test_exact_retention_receipt_and_frozen_extractor_boundary():
    receipt = audit.retention_receipt(model(), edited(), np.array([[1., 0.], [0., 2.]]))
    assert receipt["absolute_residuals_exact"] == [["5/4"], ["3/4"]]
    assert receipt["maximum_residuals_exact"] == ["5/4"]
    assert receipt["exactly_zero_on_supplied_features"] is False
    assert receipt["feature_extractor_verified"] is False
    assert receipt["floating_execution_verified"] is False
    changed = edited()
    changed["tensors"]["w1"][0][0] = 2.
    with pytest.raises(ValueError, match="frozen extractor"):
        audit.retention_receipt(model(), changed, np.array([[1., 0.]]))


def test_exact_sequential_recurrence_and_activation_scope():
    receipt = audit.sequential_receipt(model(), edited(), [-2.], [2.])
    assert receipt["layers"][0]["base_radius"] == ["2", "2"]
    assert receipt["layers"][0]["drift"] == ["0", "0"]
    assert receipt["layers"][1]["base_radius"] == ["21/2"]
    assert receipt["output_bounds"] == ["13/4"]
    base = model()
    base["metadata"]["layers"][0]["activation"] = "tanh"
    with pytest.raises(ValueError, match="Uncertified activation"):
        audit.sequential_receipt(base, base, [-2.], [2.])


def test_admission_checks_every_constraint_without_claiming_behavior():
    policy = {"writable": ["w2", "b2"], "max_edits": 3, "max_parameter_abs": 3., "max_change_abs": 1.}
    admitted = audit.admission_receipt(model(), literal(), policy)
    assert admitted["structural_accepted"] is True
    assert admitted["behavioral_status"] == "not_checked"
    assert admitted["formal_certificate"] is None
    rejected = audit.admission_receipt(model(), literal(), {
        "writable": ["b2"], "max_edits": 1, "max_parameter_abs": 2., "max_change_abs": .1})
    assert rejected["issues"] == ["write-set", "edit-count", "parameter-bound", "change-bound"]


def lowering_fixture():
    base = {"schema": "wsp-tensors/1", "dtype": "float64", "architecture": "fixture/1", "metadata": {},
            "tensors": {"weight": [[1., 2.], [3., 4.]], "bias": [0., 0.]}}
    proposal = {"schema": "wsp-structured-program/1", "base_sha256": audit.digest(base), "intent": "Structured fixture",
        "commands": [
            {"op": "scale", "tensor": "weight", "block": [[0, 1], [0, 2]], "factor": 2.},
            {"op": "add", "tensor": "weight", "values": [[0., -.5], [1., 0.]]},
            {"op": "low_rank", "tensor": "weight", "left": [[1.], [2.]], "right": [[.5, .25]], "alpha": 2.},
            {"op": "replace", "tensor": "bias", "values": [1.0001, 0.]}]}
    requested = copy.deepcopy(base)
    requested["tensors"] = {"weight": [[3., 4.], [6., 5.]], "bias": [1.0001, 0.]}
    endpoint = copy.deepcopy(requested)
    endpoint["tensors"]["bias"] = [1., 0.]
    edits = [{"tensor": "weight", "index": index, "before": before, "after": after}
             for index, before, after in [([0, 0], 1., 3.), ([0, 1], 2., 4.), ([1, 0], 3., 6.), ([1, 1], 4., 5.)]]
    edits.append({"tensor": "bias", "index": [0], "before": 0., "after": 1.})
    program = {"schema": "wsp-program/1", "base_sha256": audit.digest(base), "intent": "Structured fixture", "edits": edits}
    receipt = audit.signed({"schema": "wsp-lowering/1", "proposal_sha256": audit.digest(proposal),
        "base_sha256": audit.digest(base), "requested_candidate_sha256": audit.digest(requested),
        "candidate_sha256": audit.digest(endpoint), "program_sha256": audit.digest(program),
        "command_count": 4, "cast": "round", "native_dtypes": {"weight": "float16", "bias": "float16"},
        "projections": {
            "weight": {"dtype": "float16", "touched_cells": 4, "requested_changed_cells": 4,
                "effective_changed_cells": 4, "rounded_cells": 0, "maximum_rounding_error_exact": "0"},
            "bias": {"dtype": "float16", "touched_cells": 2, "requested_changed_cells": 1,
                "effective_changed_cells": 1, "rounded_cells": 1,
                "maximum_rounding_error_exact": str(Fraction.from_float(1.0001)-1)}},
        "effective_edits": 5, "behavioral_status": "not_checked",
        "numerics": "NumPy binary64 operations; explicit native endpoint projection", "numpy_version": np.__version__})
    return base, proposal, receipt, program, endpoint


def test_structured_lowering_replays_ordered_blocks_and_native_projection():
    audit.lowering(*lowering_fixture())


@pytest.mark.parametrize("mutation", ["broadcast", "boolean_block", "reject_rounding", "forged_projection", "wrong_endpoint"])
def test_structured_replay_rejects_rehashed_semantic_mutations(mutation):
    base, proposal, receipt, program, endpoint = lowering_fixture()
    if mutation == "broadcast":
        proposal["commands"][1]["values"] = [[1., 1.]]
    elif mutation == "boolean_block":
        proposal["commands"][0]["block"][0][0] = False
    elif mutation == "reject_rounding":
        receipt["cast"] = "reject"
    elif mutation == "forged_projection":
        receipt["projections"]["bias"]["rounded_cells"] = 0
    else:
        endpoint["tensors"]["weight"][0][0] = 3.5
        receipt["candidate_sha256"] = audit.digest(endpoint)
    receipt["proposal_sha256"] = audit.digest(proposal)
    receipt = audit.signed({key: value for key, value in receipt.items() if key != "sha256"})
    with pytest.raises(ValueError):
        audit.lowering(base, proposal, receipt, program, endpoint)


def session_fixture():
    base = {"schema": "wsp-tensors/1", "dtype": "float64", "architecture": "fixture/1", "metadata": {},
            "tensors": {"weight": [[2.]]}}
    task = {"schema": "wsp-task/1", "task_id": "fixture-read", "mode": "read", "objective": "Read a coefficient",
            "architecture": "fixture/1", "base_sha256": audit.digest(base), "constraints": {}}
    limits = dict.fromkeys(audit.COUNTERS, 0) | {"tool_calls": 2, "weight_values": 1, "proposals": 1}
    config = audit.session_config(task, audit.access_policy(metadata=False), limits,
        [{"name": "test", "sample_ids": ["fixture-0"], "lineage_ids": []}],
        {"feedback": False, "forward": False, "activations": False})
    task_id, session_id, parent = audit.digest(task), audit.digest(config), audit.digest(base)
    candidate = audit.signed({"schema": "wsp-candidate/1", "session_sha256": session_id, "task_sha256": task_id,
        "parent_sha256": parent, "proposal_index": 1, "kind": "answer", "program_sha256": None,
        "artifact_sha256": None, "artifact": None, "answer": {"coefficient": 2.}, "no_op": False})
    selected, result = candidate["sha256"], {"read_correct": True}
    operations = [
        ("session_created", config, {}, {}),
        ("weights", {"tensor": "weight", "weight_decimals": None, "indices": None}, [[2.]], {"tool_calls": 1, "weight_values": 1}),
        ("submit_answer", {}, {"candidate_sha256": selected, "artifact_sha256": None, "no_op": False}, {"tool_calls": 1, "proposals": 1}),
        ("seal", {"candidate_sha256": selected}, {"selection": "candidate"}, {}),
        ("heldout_evaluation", {"candidate_sha256": selected, "evaluator_id": "fixture/1", "split_name": "test"}, result, {}),
    ]
    usage, trace, previous = dict.fromkeys(audit.COUNTERS, 0), [], None
    for index, (action, request, outcome, charges) in enumerate(operations):
        for key, value in charges.items():
            usage[key] += value
        event = audit.signed({"schema": "wsp-trace-event/1", "sequence": index, "previous_sha256": previous,
            "session_sha256": session_id, "task_sha256": task_id, "action": action, "status": "succeeded",
            "request": request, "result": outcome, "charges": charges, "requested_charges": charges.copy(),
            "usage": usage.copy(), "error": None})
        trace.append(event)
        previous = event["sha256"]
    receipt = audit.signed({"schema": "wsp-evaluation-receipt/1", "status": "succeeded", "session_sha256": session_id,
        "task_sha256": task_id, "parent_sha256": parent, "configuration": config, "candidate_sha256": selected,
        "candidate": candidate, "candidates": [candidate], "usage": usage, "evaluator_id": "fixture/1", "split_name": "test",
        "result": result, "error": None, "trace": trace, "trace_head_sha256": previous,
        "heldout_input_handling": "caller_owned_closure_not_passed_to_proposer_tools", "evaluator_output_leakage_checked": False,
        "heldout_accounting": {"evaluator_calls": 1, "evaluated_examples": None, "scope": "outside_proposer_budget"},
        "evaluator_attested": False, "behavioral_certificate": None})
    return receipt, task, base, config, "fixture/1", "test", [op[0] for op in operations]


def rehash_trace(receipt):
    previous = None
    for index, event in enumerate(receipt["trace"]):
        event["sequence"], event["previous_sha256"] = index, previous
        event["sha256"] = audit.digest({key: value for key, value in event.items() if key != "sha256"})
        previous = event["sha256"]
    receipt["trace_head_sha256"] = previous
    receipt["sha256"] = audit.digest({key: value for key, value in receipt.items() if key != "sha256"})


def test_complete_workflow_replays_budget_and_sealed_evaluation():
    checked = audit.verify_session(*session_fixture())
    assert checked["usage"]["tool_calls"] == 2
    assert checked["usage"]["weight_values"] == 1
    assert checked["usage"]["proposals"] == 1
    assert checked["usage"]["execution_examples"] == 0


@pytest.mark.parametrize("mutation", ["undercount", "bool_usage", "before_seal", "wrong_weight", "forged_attestation", "wrong_parent"])
def test_trace_rejects_rehashed_order_accounting_and_binding_mutations(mutation):
    receipt, *arguments = session_fixture()
    if mutation == "undercount":
        receipt["trace"][1]["charges"]["weight_values"] = 0
        receipt["trace"][1]["requested_charges"]["weight_values"] = 0
        for event in receipt["trace"][1:]:
            event["usage"]["weight_values"] = 0
        receipt["usage"]["weight_values"] = 0
    elif mutation == "bool_usage":
        receipt["trace"][1]["usage"]["weight_values"] = True
    elif mutation == "before_seal":
        receipt["trace"][-2:] = list(reversed(receipt["trace"][-2:]))
    elif mutation == "wrong_weight":
        receipt["trace"][1]["result"] = [[3.]]
    elif mutation == "forged_attestation":
        receipt["evaluator_attested"] = True
    else:
        receipt["parent_sha256"] = "0" * 64
    rehash_trace(receipt)
    with pytest.raises(ValueError):
        audit.verify_session(receipt, *arguments)


def test_cluster_unit_is_world_and_ablation_failure_preserves_main_results():
    protocol = {"seeds": [900001, 900002], "families": ["relu_mlp", "tanh_mlp"],
                "bootstrap": {"draws": 100, "seed": 900099, "incomplete_world_policy": "fixture"}}
    cases = []
    for seed, family, effect, status in [(900001, "relu_mlp", 2., "completed"), (900001, "tanh_mlp", 10., "partial"),
                                         (900002, "relu_mlp", 100., "completed"), (900002, "tanh_mlp", None, "failed")]:
        arms = {arm: {"target_nrmse": 1., "retain_nrmse": 1., "target_gain_fraction": 0., "delta_l2": 1.} for arm in audit.ARMS}
        if effect is not None:
            arms["unconstrained"]["retain_nrmse"] += effect
        if seed == 900002:
            arms.pop("gradient")
        cases.append({"seed": seed, "family": family, "status": status,
            "main": None if effect is None else {"arms": arms, "selected_arm": "no_op", "failures": []}})
    summary = audit.summary_values(cases, protocol)
    assert summary["complete_world_seeds"] == [900001]
    assert summary["omitted_incomplete_worlds"] == [900002]
    assert summary["completed_checkpoints"] == 3
    assert summary["fully_completed_checkpoints"] == 2
    contrast = summary["contrasts"]["soft_collateral_reduction"]
    assert contrast["mean"] == 6.
    assert contrast["per_world_values"] == [6.]
    assert contrast["ci95_percentile"] == [6., 6.]
    assert summary["arm_denominators"]["gradient"] == 2
    assert summary["arm_denominators"]["soft_retention"] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        audit.summary_values(cases + [cases[0]], protocol)


def provenance_fixture(tmp_path):
    study, freeze = tmp_path / "study", tmp_path / "freeze"
    protocol = audit.read_json(ROOT / "docs/weight-space/v2/protocol.json")
    protocol["seeds"] = [900001]
    protocol_path = "docs/weight-space/v2/protocol.json"
    required = {f"rlcd/weight_space/{name}.py" for name in audit.SOURCE_MODULES} | {
        "rlcd/__init__.py", "requirements.lock.txt", "requirements.txt", "pytest.ini", protocol["source_review"],
        "artifacts/weight-space/source-review-v2/provenance.json", "artifacts/weight-space/source-review-v2/review-evidence.json",
        protocol_path} | {"tests/" + name for name in audit.SOURCE_TESTS}
    for name in required:
        path = freeze / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source, not executed\n", encoding="utf-8")
    write(freeze / "sources" / protocol_path, protocol)
    environment = {"packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch")}}
    frozen = audit.signed({"schema": "wsp-generic-freeze/2", "created_utc": "2026-01-01T00:00:00+00:00",
        "protocol_path": protocol_path, "protocol_sha256": audit.digest(protocol), "files": audit.inventory(freeze / "sources"),
        "environment": environment, "external_preregistration": False,
        "scope": "local pre-result snapshot of generic code, tests, protocol, and source review"})
    write(freeze / "receipt.json", frozen)
    write(study / "protocol.json", protocol)
    write(study / "results.json", {"started_utc": "2026-01-01T01:00:00+00:00", "finished_utc": "2026-01-01T02:00:00+00:00"})
    write(study / "execution-start.json", {"started_utc": "2026-01-01T01:00:00+00:00", "freeze_sha256": frozen["sha256"],
        "environment": environment, "torch_cpu_threads": 1, "provider_calls": 0, "provider_tokens": 0})
    write(study / "manifest.json", audit.signed({"schema": "wsp-generic-study-files/2", "freeze_sha256": frozen["sha256"],
                                               "files": audit.inventory(study)}))
    return study, freeze, frozen


def test_provenance_verifies_trusted_hashes_and_exact_file_inventory(tmp_path):
    study, freeze, frozen = provenance_fixture(tmp_path)
    runner = audit.Audit(study, freeze / "receipt.json")
    runner.provenance(frozen["sha256"], frozen["protocol_sha256"])
    assert runner.counts["frozen_files"] == 31
    runner.coverage()
    with pytest.raises(ValueError, match="trusted freeze"):
        audit.Audit(study, freeze / "receipt.json").provenance("0" * 64, None)


def test_rehashed_freeze_cannot_omit_a_required_source(tmp_path):
    study, freeze, frozen = provenance_fixture(tmp_path)
    removed = "rlcd/weight_space/workflow.py"
    (freeze / "sources" / removed).unlink()
    del frozen["files"][removed]
    frozen["sha256"] = audit.digest({key: value for key, value in frozen.items() if key != "sha256"})
    write(freeze / "receipt.json", frozen)
    with pytest.raises(ValueError, match="required file membership"):
        audit.Audit(study, freeze / "receipt.json").provenance(None, None)


def test_rehashed_manifest_cannot_hide_unlisted_or_unvisited_evidence(tmp_path):
    study, freeze, frozen = provenance_fixture(tmp_path)
    write(study / "unreviewed.json", {"invented": True})
    with pytest.raises(ValueError, match="study tree inventory"):
        audit.Audit(study, freeze / "receipt.json").provenance(None, None)
    files = audit.inventory(study)
    files.pop("manifest.json")
    write(study / "manifest.json", audit.signed({"schema": "wsp-generic-study-files/2", "freeze_sha256": frozen["sha256"], "files": files}))
    runner = audit.Audit(study, freeze / "receipt.json")
    runner.provenance(None, None)
    with pytest.raises(ValueError, match="unreviewed.json"):
        runner.coverage()


def test_protocol_rejects_relabelled_execution_allowances():
    protocol = audit.read_json(ROOT / "docs/weight-space/v2/protocol.json")
    protocol["seeds"] = [900001]
    audit.protocol_check(protocol)
    protocol["accounting"]["allowed"] = ["no fitting"]
    with pytest.raises(ValueError, match="accounting declarations"):
        audit.protocol_check(protocol)


def test_cli_reports_incomplete_fixture_without_mutating_inputs_or_overwriting(tmp_path):
    study, freeze, frozen = provenance_fixture(tmp_path)
    before = {"study": audit.inventory(study), "freeze": audit.inventory(freeze)}
    output = tmp_path / "independent-audit.json"
    command = [sys.executable, str(ROOT / "scripts/audit_weight_space_v2.py"), "--study-root", str(study),
               "--freeze", str(freeze), "--out", str(output), "--expected-freeze", frozen["sha256"]]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1, result.stdout + result.stderr
    report = audit.read_json(output)
    audit.signed_record(report, "audit report")
    assert report["status"] == "not_passed"
    assert report["complete"] is False
    assert report["input_inventories"]["before"] == before
    assert report["input_inventories"]["after"] == before
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert audit.read_json(output) == report
    with pytest.raises(ValueError, match="outside"):
        audit.output_path(study / "audit.json", study, freeze)
    with pytest.raises(ValueError, match="outside"):
        audit.output_path(freeze / "audit.json", study, freeze)
