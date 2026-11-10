"""Handcrafted restoration checks; no historical artifacts or scientific runners."""
import hashlib
import json

import numpy as np
import pytest

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.sequential import sequential_artifact
from rlcd.weight_space.sequential_runtime import restore_sequential_runtime
from rlcd.weight_space.structured import ExecutionSpec, _native_execution


SCOPE = {
    "backend": "numpy", "dtype": "float64", "device": "cpu",
    "accumulation": "left_to_right", "endpoint_cast": "reject", "error_model": "empirical",
}


def _endpoint():
    return sequential_artifact(
        [[[1., -2.], [-1., 1.]], [[1., 1.], [-1., 0.]], [[-2., .5]]],
        [[1., 0.], [-1., 3.], [.25]], ["identity", "relu", "abs"],
        input_names=("load", "capacity"), output_names=("pressure",),
    )


def test_restores_owned_snapshot_labels_receipt_and_hand_computed_multilayer_outputs():
    endpoint = _endpoint()
    runtime = restore_sequential_runtime(endpoint, expected_endpoint_sha256=endpoint.sha256,
                                         execution=ExecutionSpec.from_dict(SCOPE))
    # [1,-1] -> [4,-2] -> [1,0] -> abs(-1.75); [-1,1] -> [-2,2] -> [0,5] -> 2.75.
    np.testing.assert_array_equal(runtime.forward([1., -1.]), [1.75])
    np.testing.assert_array_equal(runtime.forward([[1., -1.], [-1., 1.]]), [[1.75], [2.75]])
    assert runtime.input_names == ("load", "capacity") and runtime.output_names == ("pressure",)
    snapshot, receipt = runtime.snapshot(), runtime.receipt
    assert snapshot is not endpoint and snapshot.sha256 == endpoint.sha256
    assert receipt["endpoint_sha256"] == receipt["restored_snapshot_sha256"] == endpoint.sha256
    assert receipt["schema"] == "wsp-sequential-runtime-restoration/1"
    assert receipt["status"] == "restored" and receipt["execution"] == SCOPE
    assert receipt["layers"] == endpoint.metadata["layers"]
    assert receipt["input_names"] == ["load", "capacity"] and receipt["output_names"] == ["pressure"]
    assert [tensor["name"] for tensor in receipt["tensors"]] == [
        "layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias",
        "layers.2.weight", "layers.2.bias",
    ]
    assert [tensor["shape"] for tensor in receipt["tensors"]] == [[2, 2], [2], [2, 2], [2], [1, 2], [1]]
    assert all(tensor["dtype"] == "float64" and tensor["device"] == "cpu" for tensor in receipt["tensors"])
    assert receipt["alias_groups"] == [] and "empirical" in receipt["scope"]
    assert receipt["environment"]["backend"] == "numpy" and receipt["environment"]["device"] == "cpu"
    assert "runtime_restoration_verified" not in receipt
    body = {key: value for key, value in receipt.items() if key != "sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert receipt["sha256"] == hashlib.sha256(encoded).hexdigest()


def test_left_to_right_accumulation_and_bias_after_dot_product():
    large = 2.**54
    endpoint = sequential_artifact(
        [[[large, -large, 1.], [large, 1., -large]]], [[.5, .5]], ["identity"],
    )
    runtime = restore_sequential_runtime(endpoint, expected_endpoint_sha256=endpoint.sha256, execution=SCOPE)
    np.testing.assert_array_equal(runtime.forward([1., 1., 1.]), [1.5, .5])


def test_original_arrays_snapshots_receipts_and_separate_restores_are_isolated():
    weights, bias = np.array([[1., -1.]]), np.array([.5])
    metadata = {"layers": [{"weight": "w", "bias": "b", "activation": "identity"}],
                "input_names": ["left", "right"], "output_names": ["result"]}
    endpoint = TensorArtifact("sequential-affine-activation/1", {"w": weights, "b": bias}, metadata)
    expected = endpoint.sha256
    runtime = restore_sequential_runtime(endpoint, expected_endpoint_sha256=expected, execution=SCOPE)
    other = restore_sequential_runtime(endpoint, expected_endpoint_sha256=expected, execution=SCOPE)
    snapshot = runtime.snapshot()
    other_snapshot = other.snapshot()
    for name in ("w", "b"):
        assert not np.shares_memory(snapshot.tensors[name], endpoint.tensors[name])
        assert not np.shares_memory(snapshot.tensors[name], other_snapshot.tensors[name])
    weights[:] = 99.
    bias[:] = 99.
    metadata["layers"][0]["activation"] = "abs"
    metadata["input_names"].reverse()
    header = snapshot.tensors["w"]
    header.shape = (2, 1)
    header.dtype = np.uint8
    header.base.shape = (1, 2)
    with pytest.raises(ValueError):
        snapshot.tensors["w"][0, 0] = 99.
    with pytest.raises(ValueError):
        snapshot.tensors["w"].setflags(write=True)
    detached = snapshot.to_dict()
    detached["tensors"]["w"][0][0] = 99.
    detached["metadata"]["layers"][0]["activation"] = "abs"
    receipt = runtime.receipt
    original_receipt = runtime.receipt
    receipt["execution"]["dtype"] = "float32"
    receipt["layers"][0]["activation"] = "abs"
    receipt["alias_groups"].append(["w", "b"])
    result = runtime.forward([2., -1.])
    result[:] = 99.
    with pytest.raises(AttributeError):
        runtime._artifact = endpoint
    assert runtime.receipt == original_receipt
    assert endpoint.sha256 == snapshot.sha256 == runtime.snapshot().sha256 == other.snapshot().sha256 == expected
    assert runtime.input_names == ("left", "right")
    np.testing.assert_array_equal(runtime.forward([2., -1.]), [3.5])
    np.testing.assert_array_equal(other.forward([2., -1.]), [3.5])


def test_tensor_mapping_order_is_ignored_but_endpoint_layer_and_label_substitutions_are_rejected():
    endpoint = _endpoint()
    retained = endpoint.sha256
    mapping_reordered = TensorArtifact(endpoint.architecture, dict(reversed(list(endpoint.tensors.items()))), endpoint.metadata)
    assert mapping_reordered.sha256 == retained
    runtime = restore_sequential_runtime(mapping_reordered, expected_endpoint_sha256=retained, execution=SCOPE)
    np.testing.assert_array_equal(runtime.forward([1., -1.]), [1.75])
    metadata = endpoint.metadata
    metadata["layers"][0], metadata["layers"][1] = metadata["layers"][1], metadata["layers"][0]
    layer_reordered = TensorArtifact(endpoint.architecture, endpoint.tensors, metadata)
    renamed = endpoint.metadata
    renamed["input_names"].reverse()
    for substituted in (
        endpoint.replace(**{"layers.2.bias": [.5]}), layer_reordered,
        TensorArtifact(endpoint.architecture, endpoint.tensors, renamed),
    ):
        assert substituted.sha256 != retained
        with pytest.raises(ValueError, match="endpoint"):
            restore_sequential_runtime(substituted, expected_endpoint_sha256=retained, execution=SCOPE)
    reordered_runtime = restore_sequential_runtime(layer_reordered, expected_endpoint_sha256=layer_reordered.sha256,
                                                   execution=SCOPE)
    np.testing.assert_array_equal(reordered_runtime.forward([1., -1.]), [7.25])
    assert reordered_runtime.receipt["layers"] == metadata["layers"]


@pytest.mark.parametrize("field,value", [
    ("backend", "torch"), ("dtype", "float16"), ("dtype", "float32"), ("dtype", "bfloat16"),
    ("device", "cuda"), ("device", "cuda:3"), ("accumulation", "matmul"),
    ("accumulation", "right_to_left"), ("endpoint_cast", "round"), ("error_model", "formal"),
])
def test_rejects_every_unsupported_execution_dimension(field, value):
    endpoint = _endpoint()
    with pytest.raises(ValueError, match="scope"):
        restore_sequential_runtime(endpoint, expected_endpoint_sha256=endpoint.sha256,
                                   execution=SCOPE | {field: value})


def test_rejects_float32_string_subclass_spoofing_supported_dtype_equality():
    class SpoofedDtype(str):
        def __eq__(self, other):
            return str.__eq__(self, other) or other == "float64"

        __hash__ = str.__hash__

    endpoint = _endpoint()
    spoofed = SpoofedDtype("float32")
    assert str(spoofed) == "float32" and spoofed == "float64"
    declaration = SCOPE | {"dtype": spoofed}
    spec = ExecutionSpec.from_dict(declaration)
    assert type(spec.dtype) is SpoofedDtype
    for execution in (declaration, spec):
        with pytest.raises(ValueError, match="scope"):
            restore_sequential_runtime(endpoint, expected_endpoint_sha256=endpoint.sha256,
                                       execution=execution)


@pytest.mark.parametrize("field", tuple(SCOPE))
def test_rejects_string_subclass_execution_keys_and_values(field):
    class DerivedString(str):
        pass

    endpoint = _endpoint()
    declaration = SCOPE | {field: DerivedString(SCOPE[field])}
    nonplain_key = {DerivedString(key) if key == field else key: value for key, value in SCOPE.items()}
    for execution in (declaration, ExecutionSpec.from_dict(declaration), nonplain_key):
        with pytest.raises(ValueError, match="scope"):
            restore_sequential_runtime(endpoint, expected_endpoint_sha256=endpoint.sha256,
                                       execution=execution)


@pytest.mark.parametrize("execution", [None, {}, SCOPE | {"threads": 1}, ExecutionSpec()])
def test_rejects_missing_extra_or_implicit_execution_declarations(execution):
    endpoint = _endpoint()
    with pytest.raises(ValueError):
        restore_sequential_runtime(endpoint, expected_endpoint_sha256=endpoint.sha256, execution=execution)


def test_rejects_nonplain_artifacts_unknown_architectures_and_wrong_hashes():
    endpoint = _endpoint()

    class DerivedArtifact(TensorArtifact):
        pass

    for value in (endpoint.to_dict(), DerivedArtifact(endpoint.architecture, endpoint.tensors, endpoint.metadata)):
        with pytest.raises(ValueError, match="plain TensorArtifact"):
            restore_sequential_runtime(value, expected_endpoint_sha256=endpoint.sha256, execution=SCOPE)
    unknown = TensorArtifact("caller-owned-arbitrary-module/1", endpoint.tensors, endpoint.metadata)
    with pytest.raises(ValueError, match="sequential"):
        restore_sequential_runtime(unknown, expected_endpoint_sha256=unknown.sha256, execution=SCOPE)
    for expected in ("0"*64, "invalid", None):
        with pytest.raises(ValueError, match="endpoint"):
            restore_sequential_runtime(endpoint, expected_endpoint_sha256=expected, execution=SCOPE)


def test_rejects_architecture_string_subclass_spoofing_sequential_semantics():
    class SpoofedArchitecture(str):
        def __eq__(self, other):
            return str.__eq__(self, other) or other == "sequential-affine-activation/1"

        def __ne__(self, other):
            return not self == other

        __hash__ = str.__hash__

    endpoint = _endpoint()
    architecture = SpoofedArchitecture("caller-owned-arbitrary-module/1")
    assert str(architecture) == "caller-owned-arbitrary-module/1"
    assert architecture == endpoint.architecture and not (architecture != endpoint.architecture)
    unsupported = TensorArtifact(architecture, endpoint.tensors, endpoint.metadata)
    assert unsupported.architecture is architecture
    with pytest.raises(ValueError, match="plain architecture"):
        restore_sequential_runtime(unsupported, expected_endpoint_sha256=unsupported.sha256, execution=SCOPE)


def test_rejects_tensor_name_string_subclass_spoofing_metadata_lookup():
    class SpoofedTensorName(str):
        def __eq__(self, other):
            return str.__eq__(self, other) or other == "layers.0.weight"

        def __hash__(self):
            return hash("layers.0.weight")

    endpoint = _endpoint()
    spoofed = SpoofedTensorName("undeclared.weight")
    tensors = {spoofed if name == "layers.0.weight" else name: value
               for name, value in endpoint.tensors.items()}
    unsupported = TensorArtifact(endpoint.architecture, tensors, endpoint.metadata)
    assert str(spoofed) == "undeclared.weight" and spoofed == "layers.0.weight"
    assert any(name is spoofed for name in unsupported.tensors)
    assert "layers.0.weight" in unsupported.tensors
    assert unsupported.sha256 != endpoint.sha256
    with pytest.raises(ValueError, match="plain tensor name"):
        restore_sequential_runtime(unsupported, expected_endpoint_sha256=unsupported.sha256, execution=SCOPE)


@pytest.mark.parametrize("extra", [
    {"torch_bridge": {"state": {"w": {"dtype": "torch.float32", "device": "mps"}}}},
    {"alias_groups": []}, {"alias_groups": [["layers.0.weight", "layers.1.weight"]]},
    {"native_dtypes": {"layers.0.weight": "float32"}}, {"device": "cuda:0"}, {"unknown": True},
])
def test_rejects_uninterpreted_capture_alias_and_execution_metadata(extra):
    endpoint = _endpoint()
    unsupported = TensorArtifact(endpoint.architecture, endpoint.tensors, endpoint.metadata | extra)
    with pytest.raises(ValueError, match="metadata"):
        restore_sequential_runtime(unsupported, expected_endpoint_sha256=unsupported.sha256, execution=SCOPE)


@pytest.mark.parametrize("layer_patch", [
    {"activation": "tanh"}, {"activation": "sigmoid"}, {"weight": "layers.1.weight"},
    {"bias": "layers.0.weight"}, {"capture": "undeclared"},
])
def test_rejects_unsupported_activations_reused_tensor_names_and_unknown_layer_fields(layer_patch):
    endpoint = _endpoint()
    metadata = endpoint.metadata
    metadata["layers"][0].update(layer_patch)
    unsupported = TensorArtifact(endpoint.architecture, endpoint.tensors, metadata)
    with pytest.raises(ValueError):
        restore_sequential_runtime(unsupported, expected_endpoint_sha256=unsupported.sha256, execution=SCOPE)


@pytest.mark.parametrize("tensor_patch,metadata_patch", [
    ({"layers.0.weight": [1., 2.]}, {}), ({"layers.0.bias": [0.]}, {}),
    ({"layers.1.weight": [[1., 0., 0.], [-1., 0., 0.]]}, {}), ({"uninterpreted": [0.]}, {}),
    ({}, {"layers": []}), ({}, {"input_names": ["same", "same"]}),
    ({}, {"input_names": ["left"]}), ({}, {"output_names": [""]}), ({}, {"output_names": []}),
])
def test_rejects_bad_shapes_extra_tensors_and_invalid_interfaces(tensor_patch, metadata_patch):
    endpoint = _endpoint()
    unsupported = TensorArtifact(endpoint.architecture, dict(endpoint.tensors) | tensor_patch,
                                 endpoint.metadata | metadata_patch)
    with pytest.raises(ValueError):
        restore_sequential_runtime(unsupported, expected_endpoint_sha256=unsupported.sha256, execution=SCOPE)


@pytest.mark.parametrize("inputs", [
    [], [1.], [[1., 2., 3.]], 1., [[[1., 2.]]], [[1.], [2., 3.]],
    [float("nan"), 0.], [float("inf"), 0.], [True, 0.], ["1", "2"], [2**53 + 1, 0.],
])
def test_rejects_invalid_nonfinite_or_inexact_binary64_inputs(inputs):
    endpoint = _endpoint()
    runtime = restore_sequential_runtime(endpoint, expected_endpoint_sha256=endpoint.sha256, execution=SCOPE)
    with pytest.raises(ValueError):
        runtime.forward(inputs)
    assert runtime.snapshot().sha256 == endpoint.sha256
    np.testing.assert_array_equal(runtime.forward([1., -1.]), [1.75])


@pytest.mark.parametrize("weights,bias,activation,inputs", [
    ([[-1e308]], [0.], "relu", [2.]),  # Product overflow would otherwise be hidden by ReLU.
    ([[1e308, 1e308]], [0.], "identity", [1., 1.]),  # Finite products, overflowing sum.
    ([[1e308]], [1e308], "abs", [1.]),  # Finite dot product, overflowing bias addition.
])
def test_rejects_nonfinite_products_sums_and_bias_before_activation(weights, bias, activation, inputs):
    endpoint = sequential_artifact([weights], [bias], [activation])
    runtime = restore_sequential_runtime(endpoint, expected_endpoint_sha256=endpoint.sha256, execution=SCOPE)
    with pytest.raises(ValueError, match="nonfinite"):
        runtime.forward(inputs)
    assert runtime.snapshot().sha256 == endpoint.sha256


def test_existing_native_execution_default_still_retains_nonfinite_observations():
    spec, inputs = ExecutionSpec.from_dict(SCOPE), np.array([[2.]])
    overflow = [(np.array([[1e308]]), np.array([0.]), "identity")]
    assert np.isposinf(_native_execution(overflow, inputs, spec)).all()
    masked = [(np.array([[-1e308]]), np.array([0.]), "relu")]
    np.testing.assert_array_equal(_native_execution(masked, inputs, spec), [[0.]])
