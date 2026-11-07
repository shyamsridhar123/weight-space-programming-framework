"""Independent endpoint, precision, and semantic-boundary checks for V2."""
import copy
from fractions import Fraction
from itertools import product
import json

import numpy as np
import pytest

from rlcd.weight_space import (
    InputBox, ReadoutHead, StructuredProgram, TensorArtifact, apply_program,
    certify_feature_retention, certify_sequential_drift, compile_feature_edit,
    default_registry, lower_structured, native_round, sequential_artifact,
    sequential_forward, verify_feature_retention, verify_lowering,
    verify_sequential_certificate,
)
from rlcd.weight_space.__main__ import main
from rlcd.weight_space.evidence import read_json, write_json


def tensors():
    return TensorArtifact("caller-conv-and-readout/1", {
        "kernel": np.arange(16.).reshape(2, 2, 2, 2),
        "head.weight": np.zeros((2, 3)), "head.bias": [0., 0.], "scalar": np.array(1.),
    }, {"feature_contract": "caller-owned"})


def test_structured_blocks_scalars_low_rank_and_sequential_lowering():
    base = tensors()
    commands = [
        {"op": "add", "tensor": "kernel", "block": [[0, 1], [1, 2], [0, 2], [0, 2]],
         "values": [[[[1., 2.], [3., 4.]]]]},
        {"op": "low_rank", "tensor": "head.weight", "left": [[1.], [2.]],
         "right": [[.5, 1., -1.]], "alpha": .5},
        {"op": "scale", "tensor": "scalar", "factor": 2.},
        {"op": "add", "tensor": "scalar", "block": [], "values": .25},
        {"op": "replace", "tensor": "head.bias", "values": [1., -1.]},
    ]
    proposal = StructuredProgram(base.sha256, commands, "Block, low-rank and scalar update")
    commands[0]["values"][0][0][0][0] = 100.
    lowered = lower_structured(base, proposal)
    candidate = apply_program(base, lowered.program)
    expected = base.tensors["kernel"].copy()
    expected[0:1, 1:2, :, :] += [[[[1., 2.], [3., 4.]]]]
    np.testing.assert_array_equal(candidate.tensors["kernel"], expected)
    np.testing.assert_array_equal(candidate.tensors["head.weight"], [[.25, .5, -.5], [.5, 1., -1.]])
    assert float(candidate.tensors["scalar"]) == 2.25
    assert float(base.tensors["scalar"]) == 1.
    assert lowered.receipt["projections"]["scalar"]["touched_cells"] == 1
    assert lowered.receipt["candidate_sha256"] == candidate.sha256
    assert verify_lowering(base, proposal, lowered.receipt)
    assert StructuredProgram.from_dict(proposal.to_dict()).sha256 == proposal.sha256
    with pytest.raises(AttributeError):
        proposal._bytes = b"{}"


@pytest.mark.parametrize("command", [
    {"op": "call", "tensor": "scalar", "module": "anything"},
    {"op": [], "tensor": "scalar"},
    {"op": "scale", "tensor": "scalar", "factor": True},
    {"op": "scale", "tensor": "scalar", "factor": float("inf")},
    {"op": "scale", "tensor": "scalar", "factor": 2., "extra": 1},
    {"op": "add", "tensor": "head.bias", "values": [1., 2.], "block": [[False, 1]]},
    {"op": "add", "tensor": "head.bias", "values": [1., 2.], "block": [[1, 1]]},
])
def test_structured_rejects_malformed_data_only_instructions(command):
    with pytest.raises(ValueError):
        StructuredProgram(tensors().sha256, [command], "Invalid instruction")


@pytest.mark.parametrize("command", [
    {"op": "add", "tensor": "head.bias", "values": 1.},
    {"op": "add", "tensor": "head.bias", "values": [1., 2.], "block": [[1, 3]]},
    {"op": "replace", "tensor": "missing", "values": 0.},
    {"op": "low_rank", "tensor": "head.weight", "left": [[1., 2.]], "right": [[1., 2.]], "alpha": 1.},
])
def test_failed_lowering_preserves_the_original(command):
    base = tensors()
    original = base.sha256
    proposal = StructuredProgram(base.sha256, [
        {"op": "scale", "tensor": "scalar", "factor": 5.}, command,
    ], "A failing atomic lowering")
    with pytest.raises(ValueError):
        lower_structured(base, proposal)
    assert base.sha256 == original and float(base.tensors["scalar"]) == 1.
    wrong_parent = tensors().replace(scalar=np.array(2.))
    with pytest.raises(ValueError, match="parent"):
        lower_structured(wrong_parent, proposal)


@pytest.mark.parametrize("dtype,small", [("float32", 2.**-26), ("float16", 2.**-13), ("bfloat16", 2.**-10)])
def test_native_projection_counts_requested_but_ineffective_updates(dtype, small):
    if dtype == "bfloat16":
        pytest.importorskip("torch")
    base = TensorArtifact("one-parameter/1", {"w": [1.]})
    proposal = StructuredProgram(base.sha256, [{"op": "add", "tensor": "w", "values": [small]}], "Small update")
    with pytest.raises(ValueError, match="round"):
        lower_structured(base, proposal, native_dtypes={"w": dtype})
    lowered = lower_structured(base, proposal, native_dtypes={"w": dtype}, cast="round")
    assert not lowered.program.edits
    record = lowered.receipt["projections"]["w"]
    assert record["touched_cells"] == record["requested_changed_cells"] == record["rounded_cells"] == 1
    assert record["effective_changed_cells"] == 0
    assert Fraction(record["maximum_rounding_error_exact"]) == Fraction.from_float(small)
    assert verify_lowering(base, proposal, lowered.receipt)
    altered = lowered.receipt
    altered["effective_edits"] = False
    assert not verify_lowering(base, proposal, altered)


def test_native_projection_rejects_unrepresentable_base_and_overflow():
    base = TensorArtifact("one-parameter/1", {"w": [.1]})
    no_op = StructuredProgram(base.sha256, [], "No operation")
    with pytest.raises(ValueError, match="Base tensor"):
        lower_structured(base, no_op, native_dtypes={"w": "float32"}, cast="round")
    with pytest.raises(ValueError, match="cover"):
        lower_structured(base, no_op, native_dtypes={})
    with pytest.raises(ValueError, match="overflow"):
        native_round([1e10], "float16")


def test_generic_nullspace_exact_protection_and_full_rank_infeasibility():
    base, head = tensors(), ReadoutHead("head.weight", "head.bias", "frozen-conv-features/1")
    target, desired = [[0., 0., 1.]], [[1., -.5]]
    compiled = compile_feature_edit(base, head, target, desired, [[1., 0., 0.], [0., 1., 0.]],
                                    retention="nullspace", ridge=.01)
    after = apply_program(base, compiled.program)
    assert np.linalg.norm(after.tensors["head.weight"]) > .5
    assert compiled.diagnostics["nullspace_dimension"] == 1
    assert compiled.retention_certificate["exactly_zero_on_supplied_features"] is True
    blocked = compile_feature_edit(base, head, target, desired, np.eye(3), retention="nullspace")
    assert blocked.diagnostics["no_op"] is True
    assert blocked.diagnostics["nullspace_dimension"] == 0
    assert blocked.diagnostics["target_rmse_after_projection"] > .5
    assert blocked.program.edits == ()


def test_augmented_bias_span_and_residual_amplification_are_exact():
    base = TensorArtifact("arbitrary-frozen-features/1", {"head": [[0., 0.]], "bias": [0.]})
    head = ReadoutHead("head", "bias", "frozen-nonlinear/1")
    after = base.replace(head=[[.125, -.25]], bias=[.5])
    anchors = [[1., 0.], [0., 1.]]
    receipt = certify_feature_retention(base, after, head, anchors)
    assert receipt["absolute_residuals_exact"] == [["5/8"], ["1/4"]]
    # c=(2,-1) combines augmented anchors: (2,-1,1), not (2,-1,2).
    actual = abs(Fraction(1, 8)*2 + Fraction(-1, 4)*-1 + Fraction(1, 2))
    bound = 2*Fraction(receipt["absolute_residuals_exact"][0][0]) + Fraction(receipt["absolute_residuals_exact"][1][0])
    assert actual <= bound
    assert verify_feature_retention(base, after, head, anchors, receipt)
    receipt["feature_extractor_verified"] = 0
    assert not verify_feature_retention(base, after, head, anchors, receipt)
    blocked = compile_feature_edit(base, head, [[2., -1.]], [[1.]], [[0., 0.], *anchors],
                                   edit_bias=True, retention="nullspace")
    assert blocked.diagnostics["no_op"] is True


def test_retention_rejects_extractor_changes_and_wrong_feature_identity():
    base, head = tensors(), ReadoutHead("head.weight", "head.bias", "extractor-A")
    after = base.replace(**{"head.bias": [1., 0.]})
    anchors = [[0., 0., 0.]]
    receipt = certify_feature_retention(base, after, head, anchors)
    assert not verify_feature_retention(base, after, ReadoutHead("head.weight", "head.bias", "extractor-B"), anchors, receipt)
    changed = after.replace(kernel=base.tensors["kernel"]+1.)
    with pytest.raises(ValueError, match="outside"):
        certify_feature_retention(base, changed, head, anchors)


def test_soft_retention_reduces_anchor_drift_but_can_sacrifice_requested_change():
    base = TensorArtifact("caller-features/1", {"head": [[0., 0.]]})
    head, target, desired, anchors = ReadoutHead("head"), [[1., 1.]], [[1.]], [[1., 1.]]
    free = compile_feature_edit(base, head, target, desired, anchors, retention="none")
    soft = compile_feature_edit(base, head, target, desired, anchors, retention="soft", retention_weight=8.)
    assert soft.diagnostics["retention_rmse_after_projection"] < free.diagnostics["retention_rmse_after_projection"]
    assert soft.diagnostics["target_rmse_after_projection"] > free.diagnostics["target_rmse_after_projection"]


def rational_forward(artifact, x):
    """Independent scalar Fraction evaluator; never calls the runtime forward."""
    q = lambda value: Fraction.from_float(float(value))
    values = list(map(q, x))
    for layer in artifact.metadata["layers"]:
        weight, bias = artifact.tensors[layer["weight"]], artifact.tensors[layer["bias"]]
        output = [q(bias[i])+sum((q(w)*v for w, v in zip(row, values)), Fraction())
                  for i, row in enumerate(weight)]
        if layer["activation"] == "relu":
            output = [max(v, Fraction()) for v in output]
        elif layer["activation"] == "abs":
            output = [abs(v) for v in output]
        values = output
    return values


def test_arbitrary_depth_bounds_cover_exact_independent_execution():
    rng = np.random.default_rng(5011)
    sizes, activations = [2, 3, 4, 2, 1], ["relu", "abs", "identity", "relu"]
    weights = [rng.integers(-4, 5, (b, a))/4 for a, b in zip(sizes, sizes[1:])]
    biases = [rng.integers(-4, 5, b)/4 for b in sizes[1:]]
    base = sequential_artifact(weights, biases, activations)
    after = base.replace(**{name: array+rng.integers(-1, 2, array.shape)/8 for name, array in base.tensors.items()})
    box = InputBox([-1., -.5], [1., 1.5])
    certificate = certify_sequential_drift(base, after, box)
    bound = list(map(Fraction, certificate["output_bounds"]))
    for x in product(np.linspace(-1, 1, 9), np.linspace(-.5, 1.5, 9)):
        before_y, after_y = rational_forward(base, x), rational_forward(after, x)
        assert all(abs(a-b) <= limit for a, b, limit in zip(after_y, before_y, bound))
        np.testing.assert_array_equal(sequential_forward(base, x), list(map(float, before_y)))
    assert verify_sequential_certificate(base, after, certificate)
    envelope = default_registry().certify(base, after, box)
    assert envelope["certificate"] == certificate
    certificate["floating_execution_verified"] = 0
    assert not verify_sequential_certificate(base, after, certificate)


def test_execution_does_not_imply_proof_or_compatible_semantics():
    base = sequential_artifact([[[1., 2.]], [[2.]]], [[0.], [.5]], ["tanh", "identity"])
    np.testing.assert_allclose(sequential_forward(base, [[.5, -.25]]), [[.5]])
    np.testing.assert_array_equal(sequential_forward(base, [[.5, -.25]], through_layer=0), [[.5, -.25]])
    with pytest.raises(ValueError, match="No certificate"):
        certify_sequential_drift(base, base, InputBox([-1., -1.], [1., 1.]))
    changed_metadata = base.metadata
    changed_metadata["layers"][0]["activation"] = "relu"
    other = TensorArtifact(base.architecture, base.tensors, changed_metadata)
    with pytest.raises(ValueError, match="identical"):
        certify_sequential_drift(base, other, InputBox([-1., -1.], [1., 1.]))


def test_cli_structured_feature_and_sequential_receipts(tmp_path, capsys):
    base = sequential_artifact([[[1., 0.], [0., 1.]], [[0., 0.]]], [[0., 0.], [0.]], ["relu", "identity"])
    base_path = str(tmp_path / "base.json")
    write_json(base_path, base.to_dict())
    proposal = StructuredProgram(base.sha256, [{"op": "add", "tensor": "layers.1.bias", "values": [.25]}], "Shift output")
    proposal_path = str(tmp_path / "proposal.json")
    write_json(proposal_path, proposal.to_dict())
    out = tmp_path / "lowered"
    assert main(["lower", base_path, proposal_path, "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["verify-lowering", base_path, proposal_path, str(out / "lowering.json")]) == 0
    capsys.readouterr()
    candidate = apply_program(base, lower_structured(base, proposal).program)
    candidate_path = str(tmp_path / "candidate.json")
    write_json(candidate_path, candidate.to_dict())
    box_path, certificate_path = str(tmp_path / "box.json"), str(tmp_path / "certificate.json")
    write_json(box_path, {"lower": [-1., -1.], "upper": [1., 1.]})
    assert main(["certify", base_path, candidate_path, "--box", box_path, "--out", certificate_path]) == 0
    capsys.readouterr()
    assert main(["verify", base_path, candidate_path, certificate_path]) == 0
    capsys.readouterr()
    specification = {"head": {"weight": "layers.1.weight", "bias": "layers.1.bias", "feature_id": "relu-layer-0"},
                     "target_features": [[1., 0.]], "desired_delta": [[1.]],
                     "retain_features": [[0., 1.]], "retention": "nullspace"}
    spec_path, compiled = str(tmp_path / "spec.json"), tmp_path / "compiled"
    write_json(spec_path, specification)
    assert main(["compile-features", base_path, spec_path, "--out", str(compiled)]) == 0
    diagnostics = json.loads(capsys.readouterr().out)
    assert diagnostics["nullspace_dimension"] == 1
    assert read_json(compiled / "retention.json")["exactly_zero_on_supplied_features"] is True
    altered = read_json(certificate_path)
    altered["certificate"]["reward_verified"] = 0
    bad_path = str(tmp_path / "bad.json")
    write_json(bad_path, altered)
    with pytest.raises(SystemExit) as failure:
        main(["verify", base_path, candidate_path, bad_path])
    assert failure.value.code == 2
