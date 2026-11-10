"""Literal native-count checks; no scientific data, training, or registry work."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.model import digest
from rlcd.weight_space.structured import ExecutionSpec
from scripts import staged_scientific_evaluator as evaluator


EXECUTION = ExecutionSpec("torch", "float32", "cpu", "matmul", "round", "empirical")
PATTERNS = ([64., 0.], [0., 64.], [64., 64.], [0., 0.],
            [-64., -64.], [32., 0.], [0., 32.], [64., 0.])
COUNTS = {"correct_by_class": [512, 512, 0, 0, 0, 0, 0, 0],
          "denominator_by_class": [512] * 8, "total_correct": 1024,
          "total_denominator": 4096, "reserved_output_predictions": 1024}


def _case():
    preprocessing = {"schema": "wsp-staged-identity-preprocessing/1", "input_width": 2}
    encoder = {"weight": np.zeros((32, 2), dtype=np.float32),
               "bias": np.zeros(32, dtype=np.float32)}
    encoder["weight"][0, 0] = encoder["weight"][1, 1] = 1.
    encoder_sha = digest({name: value.tolist() for name, value in encoder.items()})
    preprocessing_sha = digest(preprocessing)
    feature_id = digest({"encoder_sha256": encoder_sha, "preprocessing_sha256": preprocessing_sha,
                         "native": "Torch CPU float32 affine then ReLU", "feature_order": list(range(32))})
    weight = np.zeros((9, 32), dtype=np.float32)
    weight[0, 0] = weight[1, 1] = 2.
    endpoint = TensorArtifact("trained-frozen-readout/1",
                              {"head.weight": weight, "head.bias": [-128.] * 8 + [-64.]},
                              {"encoder_sha256": encoder_sha, "preprocessing_sha256": preprocessing_sha,
                               "feature_id": feature_id, "feature_names": [f"h{i:03d}" for i in range(32)]})
    rows = [{"inputs": list(pattern), "label": label}
            for label, pattern in enumerate(PATTERNS) for _ in range(512)]
    binding = {"endpoint_sha256": endpoint.sha256, "encoder_sha256": encoder_sha,
               "preprocessing_sha256": preprocessing_sha, "feature_id": feature_id, "rows_sha256": digest(rows)}
    return {"endpoint": endpoint, "encoder": encoder, "preprocessing": preprocessing,
            "ordered_rows": rows, "expected_binding": binding, "execution": EXECUTION}


def test_native_counts_include_relu_lowest_index_ties_and_reserved_output(monkeypatch):
    case = _case()
    original_encode, original_head = evaluator.encode_inputs, evaluator.evaluate_head
    observed_inputs, observed_features, predictions = [], [], []

    def encode(encoder, inputs):
        assert inputs.shape == (256, 2) and inputs.dtype == np.float32
        observed_inputs.extend(inputs.tolist())
        return original_encode(encoder, inputs)

    def head(endpoint, features, *, native=False):
        assert native is True and features.shape == (256, 32) and features.dtype == np.float32
        observed_features.extend(features[:, :2].tolist())
        assert np.count_nonzero(features[:, 2:]) == 0
        scores = original_head(endpoint, features, native=native)
        predictions.extend(np.argmax(scores, axis=1).tolist())
        return scores

    monkeypatch.setattr(evaluator, "encode_inputs", encode)
    monkeypatch.setattr(evaluator, "evaluate_head", head)
    result = evaluator.evaluate_complete_slot(**case)
    # Scores 0/1 are 2*relu(x/y)-128; the reserved score is always -64.
    # Class 2 ties ordinary scores; classes 5/6 tie an ordinary score with 8.
    assert predictions == [prediction for prediction in (0, 1, 0, 8, 8, 0, 1, 0) for _ in range(512)]
    assert observed_inputs == [row["inputs"] for row in case["ordered_rows"]]
    hidden_patterns = ([64., 0.], [0., 64.], [64., 64.], [0., 0.],
                       [0., 0.], [32., 0.], [0., 32.], [64., 0.])
    assert observed_features == [pattern for pattern in hidden_patterns for _ in range(512)]
    assert result["schema"] == "wsp-staged-native-slot-counts/1"
    assert result["binding"] == case["expected_binding"]
    assert result["execution"] == EXECUTION.to_dict()
    assert result["chunk_size"] == 256 and result["chunks_completed"] == 16
    assert result["counts"] == COUNTS
    assert all(type(count) is int for count in result["counts"]["correct_by_class"])
    assert isinstance(result["runtime_correspondence"], dict) and result["runtime_correspondence"]


def test_identity_shape_and_native_input_errors_reject_before_encoding(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid declarations must be rejected before native execution")

    monkeypatch.setattr(evaluator, "encode_inputs", forbidden)
    monkeypatch.setattr(evaluator, "evaluate_head", forbidden)

    def reject(case):
        with pytest.raises(ValueError):
            evaluator.evaluate_complete_slot(**case)

    for field in _case()["expected_binding"]:
        case = _case()
        case["expected_binding"][field] = "0" * 64
        reject(case)

    for mutation in ("reorder", "signed_zero"):
        case = _case()
        rows = case["ordered_rows"]
        if mutation == "reorder":
            rows[0], rows[512] = rows[512], rows[0]
        else:
            rows[0]["inputs"][1] = -0.0
        assert [sum(row["label"] == label for row in rows) for label in range(8)] == [512] * 8
        assert digest(rows) != case["expected_binding"]["rows_sha256"]
        reject(case)

    class Schema(str):
        pass

    base = _case()
    for field, value in (
        ("expected_binding", base["expected_binding"] | {"untrusted_extra": "0" * 64}),
        ("preprocessing", base["preprocessing"] | {"input_width": True}),
        ("preprocessing", base["preprocessing"] | {"schema": Schema(base["preprocessing"]["schema"])}),
        ("ordered_rows", base["ordered_rows"][:-1]),
        ("encoder", base["encoder"] | {"weight": base["encoder"]["weight"].astype(np.float64)}),
        ("encoder", base["encoder"] | {"weight": np.zeros((31, 2), dtype=np.float32)}),
        ("encoder", base["encoder"] | {"bias": np.full(32, np.inf, dtype=np.float32)}),
        ("execution", EXECUTION.to_dict()),
        ("execution", ExecutionSpec("torch", "float64", "cpu", "matmul", "round", "empirical")),
    ):
        case = _case()
        case[field] = value
        reject(case)

    for field, value in (("inputs", [0.1, 0.]), ("inputs", [True, 0.]),
                         ("inputs", [float("inf"), 0.]), ("inputs", [0.]),
                         ("label", True), ("label", 8), ("label", 1)):
        case = _case()
        case["ordered_rows"][0][field] = value
        reject(case)

    endpoint = base["endpoint"]
    bad_heads = (
        TensorArtifact(endpoint.architecture, {"head.weight": np.zeros((8, 32)), "head.bias": [0.] * 8},
                       endpoint.metadata),
        endpoint.replace(**{"head.bias": [0.1] + [-128.] * 7 + [-64.]}),
        endpoint.replace(**{"head.bias": [-128.] * 8 + [-63.]}),
        TensorArtifact(endpoint.architecture, endpoint.tensors,
                       endpoint.metadata | {"feature_names": list(reversed(endpoint.metadata["feature_names"]))}),
        TensorArtifact(endpoint.architecture, endpoint.tensors, endpoint.metadata | {"native_dtype": "float64"}),
        TensorArtifact(endpoint.architecture, endpoint.tensors, endpoint.metadata | {"encoder_sha256": "0" * 64}),
    )
    for endpoint in bad_heads:
        case = _case()
        case["endpoint"] = endpoint
        case["expected_binding"]["endpoint_sha256"] = endpoint.sha256
        reject(case)


def test_mutating_caller_inputs_during_execution_cannot_change_captured_slot(monkeypatch):
    case = _case()
    retained_binding = case["expected_binding"].copy()
    retained_rows = deepcopy(case["ordered_rows"])
    retained_encoder = {name: value.copy() for name, value in case["encoder"].items()}
    original_encode, observed_inputs = evaluator.encode_inputs, []

    def encode(encoder, inputs):
        assert encoder is not case["encoder"]
        for name, value in encoder.items():
            assert not np.shares_memory(value, case["encoder"][name])
            np.testing.assert_array_equal(value, retained_encoder[name])
        if not observed_inputs:
            for value in case["encoder"].values():
                value.fill(99.)
            for row in case["ordered_rows"]:
                row["inputs"][:] = [1., 1.]
                row["label"] = 0
            case["preprocessing"].clear()
            case["expected_binding"].clear()
        observed_inputs.extend(inputs.tolist())
        return original_encode(encoder, inputs)

    monkeypatch.setattr(evaluator, "encode_inputs", encode)
    result = evaluator.evaluate_complete_slot(**case)
    assert result["binding"] == retained_binding and result["counts"] == COUNTS
    assert observed_inputs == [row["inputs"] for row in retained_rows]
    assert result["chunks_completed"] == 16


def test_late_chunk_failures_and_invalid_native_outputs_produce_no_completed_result(monkeypatch):
    original_encode, original_head = evaluator.encode_inputs, evaluator.evaluate_head
    for failure in ("exception", "interrupt", "short_features", "float64_features",
                    "nonfinite_features", "short_scores", "float64_scores", "nonfinite_scores"):
        calls = {"encode": 0, "head": 0}
        completed = []
        error = KeyboardInterrupt("literal interruption") if failure == "interrupt" else RuntimeError("literal chunk failure")

        def encode(encoder, inputs):
            calls["encode"] += 1
            features = original_encode(encoder, inputs)
            if calls["encode"] == 3:
                if failure == "short_features":
                    return features[:-1]
                if failure == "float64_features":
                    return features.astype(np.float64)
                if failure == "nonfinite_features":
                    features[0, 0] = np.nan
            return features

        def head(endpoint, features, *, native=False):
            calls["head"] += 1
            if calls["head"] == 3 and failure in ("exception", "interrupt"):
                raise error
            scores = original_head(endpoint, features, native=native)
            if calls["head"] == 3:
                if failure == "short_scores":
                    return scores[:, :8]
                if failure == "float64_scores":
                    return scores.astype(np.float64)
                if failure == "nonfinite_scores":
                    scores[0, 0] = np.inf
            return scores

        with monkeypatch.context() as patch:
            patch.setattr(evaluator, "encode_inputs", encode)
            patch.setattr(evaluator, "evaluate_head", head)
            with pytest.raises(type(error) if failure in ("exception", "interrupt") else ValueError) as caught:
                completed.append(evaluator.evaluate_complete_slot(**_case()))
            if failure in ("exception", "interrupt"):
                assert caught.value is error
        assert completed == [] and calls["encode"] == 3
        assert calls["head"] == (2 if failure.endswith("features") else 3)


def test_runtime_contexts_restore_on_return_and_unwinding_without_global_policy_setters(monkeypatch):
    original_head = evaluator.evaluate_head

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not change global numerical policy")

    for name in ("set_num_threads", "set_num_interop_threads", "set_float32_matmul_precision", "set_flush_denormal"):
        monkeypatch.setattr(torch, name, forbidden)
    ambient = (str(torch.empty(0).device), torch.get_default_dtype(), torch.is_autocast_enabled("cpu"))

    for fail in (False, True):
        def head(endpoint, features, *, native=False):
            assert str(torch.empty(0).device) == "cpu"
            assert torch.is_autocast_enabled("cpu") is False
            if fail:
                raise RuntimeError("literal context unwind")
            return original_head(endpoint, features, native=native)

        with monkeypatch.context() as patch:
            patch.setattr(evaluator, "evaluate_head", head)
            with torch.device("meta"), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                case = _case()
                # The optional metadata declaration is accepted when it agrees.
                endpoint = case["endpoint"]
                case["endpoint"] = TensorArtifact(endpoint.architecture, endpoint.tensors,
                                                   endpoint.metadata | {"native_dtype": "float32"})
                case["expected_binding"]["endpoint_sha256"] = case["endpoint"].sha256
                if fail:
                    with pytest.raises(RuntimeError, match="literal context unwind"):
                        evaluator.evaluate_complete_slot(**case)
                else:
                    result = evaluator.evaluate_complete_slot(**case)
                    assert result["counts"] == COUNTS
                    runtime = result["runtime_correspondence"]
                    assert runtime["default_device"] == "cpu" and runtime["cpu_autocast_enabled"] is False
                    assert runtime["matmul_precision"] == "highest"
                assert str(torch.empty(0).device) == "meta" and torch.is_autocast_enabled("cpu") is True
        assert (str(torch.empty(0).device), torch.get_default_dtype(), torch.is_autocast_enabled("cpu")) == ambient

    thread_count = torch.get_num_threads()
    calls = {"head": 0}

    def head_before_drift(endpoint, features, *, native=False):
        scores = original_head(endpoint, features, native=native)
        calls["head"] += 1
        return scores

    with monkeypatch.context() as patch:
        patch.setattr(evaluator, "evaluate_head", head_before_drift)
        patch.setattr(torch, "get_num_threads", lambda: thread_count + int(calls["head"] == 16))
        with pytest.raises(ValueError, match="Observable runtime configuration changed during the slot"):
            evaluator.evaluate_complete_slot(**_case())
    assert calls["head"] == 16
    assert (str(torch.empty(0).device), torch.get_default_dtype(), torch.is_autocast_enabled("cpu")) == ambient

    for owner, name, value in ((torch, "__version__", "2.7.0+cpu"), (torch.version, "cuda", "12.8"),
                               (torch.version, "hip", "6.4"), (torch, "get_float32_matmul_precision", lambda: "high")):
        with monkeypatch.context() as patch:
            patch.setattr(owner, name, value)
            patch.setattr(evaluator, "encode_inputs", forbidden)
            with pytest.raises(ValueError):
                evaluator.evaluate_complete_slot(**_case())
