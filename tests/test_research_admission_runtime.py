"""Substantive trust and decision boundaries, rather than row-count targets."""
from fractions import Fraction
import json

import pytest

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.contracts import (
    AdmissionRecord, DecisionRule, admit, replay_admission, verify_admission,
)
from rlcd.weight_space import evidence
from rlcd.weight_space.features import ReadoutHead, certify_feature_retention
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.runtime_research import (
    SUBSTITUTIONS, affine, bind_case, boundary_protocol, run, substitute_request, verify,
)
from rlcd.weight_space.semantics import SemanticsAdapter, SemanticsRegistry, default_registry
from rlcd.weight_space.sequential import sequential_forward
from rlcd.weight_space.structured import ExecutionSpec, native_execution_receipt, verify_native_execution


def request(name="valid-edit"):
    return bind_case(next(case for case in boundary_protocol()["admission_cases"] if case["case"] == name))


@pytest.mark.parametrize("name,accepted", [
    ("valid-edit", True), ("no-op", True), ("stale-program", False),
    ("witness-budget-violation", False), ("conservative-bound-rejection", False),
    ("stochastic-decision-unsupported", False), ("unregistered-semantics", False),
    ("tie-reject", False), ("tie-first_index", True), ("tie-last_index", True),
    ("real-pass-native-fail", False),
])
def test_frozen_admissions_replay_every_disposition(name, accepted):
    frozen = request(name)
    record = admit(frozen)
    assert record.accepted is accepted
    assert verify_admission(record, frozen.sha256)
    assert canonical_bytes(replay_admission(record, expected_binding_sha256=frozen.sha256).to_dict()) == canonical_bytes(record.to_dict())
    assert all(gate["binding_sha256"] == frozen.sha256 for gate in record.to_dict()["gates"].values())
    detached_request, detached_record = frozen.to_dict(), record.to_dict()
    detached_request["domain"]["upper"][0] = 99.
    detached_record["accepted"] = not accepted
    assert frozen.sha256 == record.binding_sha256 and record.accepted is accepted


@pytest.mark.parametrize("name", SUBSTITUTIONS)
def test_fully_rehashed_substitutions_do_not_cross_the_trusted_request(name):
    original = request()
    changed = substitute_request(original, name)
    record = admit(changed)
    assert changed.sha256 != original.sha256
    assert not verify_admission(record, original.sha256)
    assert verify_admission(record, changed.sha256)
    with pytest.raises(ValueError, match="trusted"):
        replay_admission(record, expected_binding_sha256=original.sha256)


@pytest.mark.parametrize("change", ["required", "evidence", "accepted", "claim", "gate_status"])
def test_rehashed_gate_or_claim_tampering_fails_independent_recomputation(change):
    frozen = request()
    record = admit(frozen).to_dict()
    if change == "required":
        record["gates"]["structural"]["required"] = False
    elif change == "evidence":
        record["gates"]["binding"]["evidence"]["no_op"] = True
    elif change == "accepted":
        record["accepted"] = False
    elif change == "claim":
        record["claims"]["native_execution_formally_verified"] = True
    else:
        record["gates"]["real_box_drift"]["status"] = "unsupported"
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    AdmissionRecord.from_dict(record)  # Valid shape and hashes are insufficient.
    assert not verify_admission(record, frozen.sha256)


def test_conservative_coverage_loss_and_witnessed_violation_are_distinct():
    loose = admit(request("conservative-bound-rejection")).to_dict()
    assert loose["gates"]["real_box_drift"]["status"] == "conservative_rejection"
    assert loose["gates"]["real_witness_scores"]["status"] == "pass"
    assert loose["gates"]["feature_retention"]["evidence"]["exactly_zero_on_supplied_features"]
    violating = admit(request("witness-budget-violation")).to_dict()
    assert violating["gates"]["real_witness_scores"]["status"] == "observed_violation"
    assert not loose["accepted"] and not violating["accepted"]


def test_adapter_closure_and_backend_environment_changes_lose_correspondence(monkeypatch):
    frozen = request()
    old = default_registry().adapter(TensorArtifact.from_dict(frozen.to_dict()["parent"]))
    configuration = {"scale": 1.}
    def evaluate(artifact, inputs):
        return sequential_forward(artifact, inputs) * configuration["scale"]
    registry = SemanticsRegistry()
    registry.register(SemanticsAdapter(old.architecture, evaluate, old.certify))
    case = next(c for c in boundary_protocol()["admission_cases"] if c["case"] == "valid-edit")
    # bind_case uses the default registry; explicitly freeze this reviewed callback.
    from rlcd.weight_space.contracts import AdmissionRequest
    raw = frozen.to_dict()
    raw["adapter"] = registry.adapter(TensorArtifact.from_dict(case["parent"])).descriptor()
    raw["sha256"] = digest({k: v for k, v in raw.items() if k != "sha256"})
    custom = AdmissionRequest.from_dict(raw)
    assert admit(custom, registry).accepted
    configuration["scale"] = 2.
    changed = admit(custom, registry)
    assert changed.to_dict()["gates"]["semantics"]["status"] == "missing_correspondence"
    assert not changed.accepted
    actual_environment = evidence.numerical_environment
    monkeypatch.setattr(evidence, "numerical_environment",
        lambda backend, device="cpu": actual_environment(backend, device) | {"numpy_version": "changed"})
    changed_backend = admit(frozen)
    assert changed_backend.to_dict()["gates"]["native_execution"]["status"] == "missing_correspondence"
    assert not changed_backend.accepted


def test_declared_ties_scores_only_stochastic_and_nonfinite_rules():
    assert DecisionRule().classify([1., 1.])["status"] == "conservative_rejection"
    assert DecisionRule("argmax", "first_index").classify([1., 1.])["winner"] == 0
    assert DecisionRule("argmax", "last_index").classify([1., 1.])["winner"] == 1
    assert DecisionRule("scores_only").classify([1., 1.])["winner"] is None
    assert DecisionRule("stochastic").classify([1., 0.])["status"] == "unsupported"
    for nonfinite in (float("nan"), float("inf"), -float("inf")):
        assert DecisionRule("scores_only").classify([1., nonfinite])["status"] == "observed_violation"


@pytest.mark.parametrize("backend,dtype,large", [
    ("numpy", "float64", 2.0**54), ("numpy", "float32", 2.0**25), ("numpy", "float16", 2.0**12),
    ("torch", "float64", 2.0**54), ("torch", "float32", 2.0**25),
    ("torch", "float16", 2.0**12), ("torch", "bfloat16", 2.0**9),
])
def test_serial_cancellation_has_zero_real_anchor_residual_and_native_order_effects(backend, dtype, large):
    if backend == "torch":
        pytest.importorskip("torch")
    base = affine([[large, -large, 1.], [0., 0., 0.]], [0., .5])
    candidate = base.replace(**{"layers.0.weight": [[large, 1., -large], [0., 0., 0.]]})
    anchors = [[1., 1., 1.]]
    retention = certify_feature_retention(base, candidate, ReadoutHead("layers.0.weight", "layers.0.bias"), anchors)
    assert retention["exactly_zero_on_supplied_features"]
    left = native_execution_receipt(base, candidate, anchors,
        ExecutionSpec(backend=backend, dtype=dtype, accumulation="left_to_right"))
    right = native_execution_receipt(base, candidate, anchors,
        ExecutionSpec(backend=backend, dtype=dtype, accumulation="right_to_left"))
    assert left["observations"]["parent"]["stored_exact_outputs"] == [["1", "1/2"]]
    assert left["observations"]["candidate"]["stored_exact_outputs"] == [["1", "1/2"]]
    assert left["observations"]["parent"]["native_outputs"] == [[1., .5]]
    assert left["observations"]["candidate"]["native_outputs"] == [[0., .5]]
    assert left["status"] == "observed_violation" and right["status"] == "pass"
    for endpoint in ("parent", "candidate"):
        assert right["observations"][endpoint]["native_outputs"] == [[0., .5]]
        assert right["observations"][endpoint]["decision_effects"]["stored_real_to_native"][0]["status"] == "observed_violation"
        for error in left["observations"][endpoint]["per_coordinate_errors"][0]:
            assert Fraction(error["endpoint_error_exact"]) == Fraction(error["input_error_exact"]) == 0
    assert verify_native_execution(base, candidate, left)
    assert verify_native_execution(base, candidate, right)


def test_near_tie_native_rejection_and_paired_error_margin_budget():
    record = admit(request("real-pass-native-fail")).to_dict()
    gates = record["gates"]
    assert all(gates[name]["status"] == "pass" for name in (
        "binding", "semantics", "structural", "real_box_drift", "real_witness_scores", "real_witness_decision", "feature_retention"))
    assert gates["native_witness_decision"]["status"] == "observed_violation"
    assert not record["accepted"] and not record["claims"]["native_witness_decision_preserved"]
    native = gates["native_execution"]["evidence"]
    parent, candidate = native["observations"]["parent"], native["observations"]["candidate"]
    assert parent["stored_exact_decisions"][0]["winner"] == candidate["stored_exact_decisions"][0]["winner"] == 1
    assert parent["native_decisions"][0]["winner"] == 0 and candidate["native_decisions"][0]["winner"] == 1
    assert native["parent_projection_changed"] and native["candidate_projection_changed"]
    assert native["projected_parent_sha256"] != native["base_sha256"]
    assert parent["decision_effects"]["endpoint_quantization"][0]["status"] == "observed_violation"
    for endpoint in (parent, candidate):
        for error in endpoint["per_coordinate_errors"][0]:
            assert Fraction(error["input_error_exact"]) == Fraction(error["intermediate_error_exact"]) == 0
    comparison = gates["native_witness_margin"]["evidence"][0]["comparisons"][0]
    bound = Fraction(gates["real_box_drift"]["evidence"]["certificate"]["output_bounds"][1])
    total = bound + sum(abs(Fraction(lane["per_coordinate_errors"][0][i]["total_error_exact"]))
                       for lane in (parent, candidate) for i in (0, 1))
    assert Fraction(comparison["total_budget_exact"]) == total
    assert comparison["status"] == "conservative_rejection"


def test_native_noop_can_preserve_native_decisions_while_disagreeing_with_real_parent():
    case = next(c for c in boundary_protocol()["native_cases"]
                if c["case"] == "no-op-native-preserved-real-correspondence-fails")
    base = TensorArtifact.from_dict(case["parent"])
    receipt = native_execution_receipt(base, base, case["witnesses"],
        ExecutionSpec.from_dict(case["execution"]), decision=DecisionRule.from_dict(case["decision"]))
    assert receipt["no_op"] and receipt["status"] == "pass"
    assert receipt["native_decision_preserved"][0]["status"] == "pass"
    assert receipt["observations"]["parent"]["decision_effects"]["stored_real_to_native"][0]["status"] == "observed_violation"


def test_input_projection_and_intermediate_overflow_are_separate_and_json_safe():
    base = affine([[1.], [0.]], [0., 0.])
    input_receipt = native_execution_receipt(base, base, [[1 + 2.0**-25]], ExecutionSpec(dtype="float32"))
    error = input_receipt["observations"]["parent"]["per_coordinate_errors"][0][0]
    assert input_receipt["input_projection_changed"] and not input_receipt["parent_projection_changed"]
    assert Fraction(error["endpoint_error_exact"]) == Fraction(error["intermediate_error_exact"]) == 0
    assert Fraction(error["input_error_exact"]) == -Fraction(1, 2**25)
    overflow = affine([[65504., 65504.], [0., 0.]], [0., 0.])
    receipt = native_execution_receipt(overflow, overflow, [[2., 2.]],
        ExecutionSpec(dtype="float16", accumulation="left_to_right"))
    assert receipt["no_op"] and receipt["status"] == "observed_violation"
    assert not receipt["parent_projection_changed"]
    parent = receipt["observations"]["parent"]
    assert parent["stored_exact_outputs"] == [["262016", "0"]]
    assert parent["native_outputs"] == [["+Infinity", 0.]]
    assert parent["per_coordinate_errors"][0][0]["total_error_exact"] is None
    assert parent["native_decisions"][0]["status"] == "observed_violation"
    json.dumps(receipt, allow_nan=False)
    assert verify_native_execution(overflow, overflow, receipt)


def test_unsupported_operator_formal_backend_and_forbidden_baseline_scopes_are_visible():
    cases = {c["case"]: c for c in boundary_protocol()["native_cases"]}
    for name in ("tanh-native-only", "formal-native-lane-unsupported", "unknown-operator-model",
                 "forbidden-endpoint-projection"):
        case = cases[name]
        base, candidate = TensorArtifact.from_dict(case["parent"]), TensorArtifact.from_dict(case["candidate"])
        receipt = native_execution_receipt(base, candidate, case["witnesses"],
            ExecutionSpec.from_dict(case["execution"]), decision=DecisionRule.from_dict(case["decision"]))
        assert receipt["no_op"]
        assert receipt["status"] == ("invalid" if name == "forbidden-endpoint-projection" else "unsupported")
        assert not receipt["formal_execution_verified"]
        assert verify_native_execution(base, candidate, receipt)
        if name == "tanh-native-only":
            assert "native_outputs" in receipt["observations"]["parent"]
            assert not receipt["exact_real_supported"]
    torch = pytest.importorskip("torch")
    base = affine([[1.], [0.]], [0., 0.])
    assert native_execution_receipt(base, base, [[1.]], ExecutionSpec(dtype="bfloat16"))["status"] == "unsupported"
    if not torch.cuda.is_available():
        assert native_execution_receipt(base, base, [[1.]],
            ExecutionSpec(backend="torch", device="cuda"))["status"] == "unsupported"


def test_frozen_study_measures_matmul_replays_and_refuses_evidence_overwrite(tmp_path):
    directory = tmp_path / "boundaries"
    results = run(directory)
    assert verify(directory)
    assert all(row["replay_verified"] for row in results["admissions"])
    assert all(not row["verifies_under_original_trust_anchor"] and row["new_claim_replay_verified"]
               for row in results["substitution_attacks"])
    # Matmul is measured, with no portable assumption about its reduction order.
    matmul = [row for row in results["native_cases"] if row["case"].endswith("/matmul")]
    assert matmul and all("environment" in row["receipt"] for row in matmul)
    with pytest.raises(FileExistsError):
        run(directory)
    stored = evidence.read_json(directory / "results.json")
    stored["claim"] = "altered"
    (directory / "results.json").write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValueError, match="altered"):
        verify(directory)
