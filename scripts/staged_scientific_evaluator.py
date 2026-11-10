"""Complete-slot native counts for the proposed staged study, without admission.

This in-memory primitive owns no registry, persistence, retry or scientific
selection. Expected identities must be retained independently by its caller.
"""
from __future__ import annotations

import hashlib
import re
import struct
from typing import Any

import numpy as np

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.model import digest, finite_binary64
from rlcd.weight_space.research_tasks import encode_inputs, evaluate_head
from rlcd.weight_space.structured import ExecutionSpec


_EXECUTION = ExecutionSpec("torch", "float32", "cpu", "matmul", "round", "empirical")
_BINDING_KEYS = frozenset({
    "endpoint_sha256", "encoder_sha256", "preprocessing_sha256", "feature_id", "rows_sha256",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _float32(value: Any) -> float:
    _require(type(value) in (int, float), "Inputs require plain numeric scalars")
    finite_binary64(value)
    number = float(value)  # Preserve raw signed zero; finite_binary64 normalizes it.
    try:
        native = struct.unpack("=f", struct.pack("=f", number))[0]
    except (OverflowError, struct.error):
        raise ValueError("Values must be exactly representable as finite float32") from None
    _require(native == number, "Values must be exactly representable as finite float32")
    return number


def _runtime(torch: Any) -> dict[str, Any]:
    """Observable correspondence only; no complete floating-point-state proof."""
    return {
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": torch.version.hip,
        "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode("utf-8")).hexdigest(),
        "numpy_version": str(np.__version__),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "default_device": str(torch.get_default_device()),
        "default_dtype": str(torch.get_default_dtype()),
        "cpu_autocast_enabled": torch.is_autocast_enabled("cpu"),
        "cpu_autocast_dtype": str(torch.get_autocast_dtype("cpu")),
    }


def evaluate_complete_slot(
    endpoint: TensorArtifact,
    encoder: dict[str, np.ndarray],
    preprocessing: dict[str, Any],
    ordered_rows: list[dict[str, Any]],
    *,
    expected_binding: dict[str, str],
    execution: ExecutionSpec,
) -> dict[str, Any]:
    """Return full counts only after all sixteen contiguous native chunks succeed.

    Raw rows are plain {"inputs": [...], "label": 0..7} dictionaries.
    The exact identity-preprocessing schema supports finite binary32 numeric
    inputs, not text, normalization, or precomputed learned features.
    """
    _require(type(execution) is ExecutionSpec and execution == _EXECUTION,
             "Unsupported native execution declaration")
    _require(type(expected_binding) is dict and set(expected_binding) == _BINDING_KEYS,
             "Supply all five independently retained expected identities")
    binding = expected_binding.copy()
    _require(all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
                 for value in binding.values()), "Expected identities must be lowercase SHA-256")

    _require(type(preprocessing) is dict and set(preprocessing) == {"schema", "input_width"},
             "Unsupported identity preprocessing fields")
    preprocessing = preprocessing.copy()
    width = preprocessing["input_width"]
    _require(type(preprocessing["schema"]) is str
             and preprocessing["schema"] == "wsp-staged-identity-preprocessing/1"
             and type(width) is int and width > 0, "Unsupported identity preprocessing")

    _require(type(encoder) is dict and set(encoder) == {"weight", "bias"}
             and all(type(value) is np.ndarray for value in encoder.values()),
             "Supply only numeric encoder weight and bias arrays")
    encoder = {name: value.copy() for name, value in encoder.items()}
    _require(encoder["weight"].shape == (32, width) and encoder["bias"].shape == (32,)
             and all(value.dtype == np.dtype(np.float32) and value.dtype.isnative
                     and np.isfinite(value).all() for value in encoder.values()),
             "Encoder requires aligned finite native float32 arrays")

    _require(type(endpoint) is TensorArtifact
             and endpoint.architecture == "trained-frozen-readout/1",
             "Unsupported trained-head interface")
    tensors, metadata = endpoint.tensors, endpoint.metadata
    _require(set(tensors) == {"head.weight", "head.bias"}
             and tensors["head.weight"].shape == (9, 32)
             and tensors["head.bias"].shape == (9,), "Unsupported trained-head shapes")
    for tensor in tensors.values():
        for value in tensor.flat:
            _float32(float(value))
    _require(np.all(tensors["head.weight"][8] == 0) and tensors["head.bias"][8] == -64,
             "Reserved output row changed")
    _require(metadata.get("feature_names") == [f"h{index:03d}" for index in range(32)],
             "Head must declare canonical feature order")
    _require("native_dtype" not in metadata or metadata["native_dtype"] == "float32",
             "Head declares a different native dtype")

    _require(type(ordered_rows) is list and len(ordered_rows) == 4096,
             "A complete slot requires exactly 4096 ordered rows")
    captured_rows = []
    inputs, labels = [], []
    denominators = [0] * 8
    for row in ordered_rows:
        _require(type(row) is dict and set(row) == {"inputs", "label"}, "Unsupported raw row")
        values, label = row["inputs"], row["label"]
        _require(type(values) is list and len(values) == width
                 and type(label) is int and 0 <= label < 8, "Invalid raw input width or label")
        values = values.copy()
        inputs.append([_float32(value) for value in values])
        labels.append(label)
        denominators[label] += 1
        captured_rows.append({"inputs": values, "label": label})
    _require(denominators == [512] * 8, "Every ordinary class requires exactly 512 rows")

    encoder_sha = digest({name: value.tolist() for name, value in encoder.items()})
    preprocessing_sha = digest(preprocessing)
    feature_id = digest({
        "encoder_sha256": encoder_sha, "preprocessing_sha256": preprocessing_sha,
        "native": "Torch CPU float32 affine then ReLU", "feature_order": list(range(32)),
    })
    actual_binding = {
        "endpoint_sha256": endpoint.sha256, "encoder_sha256": encoder_sha,
        "preprocessing_sha256": preprocessing_sha, "feature_id": feature_id,
        "rows_sha256": digest(captured_rows),
    }
    _require(actual_binding == binding, "Expected binding does not match captured slot")
    _require(all(metadata.get(key) == actual_binding[key]
                 for key in ("encoder_sha256", "preprocessing_sha256", "feature_id")),
             "Head feature identities do not match captured encoder and preprocessing")
    inputs = np.array(inputs, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    import torch

    _require(str(torch.__version__).split("+", 1)[0] == "2.8.0"
             and torch.version.cuda is None and torch.version.hip is None,
             "Supported runtime is the CPU build of Torch 2.8.0")
    correct_by_class = [0] * 8
    reserved_predictions = 0
    with torch.device("cpu"), torch.autocast(device_type="cpu", enabled=False):
        before = _runtime(torch)
        _require(before["default_device"] == "cpu" and not before["cpu_autocast_enabled"],
                 "Native CPU execution context was not established")
        _require(before["matmul_precision"] == "highest",
                 "Native evaluation requires highest float32 matmul precision")
        for start in range(0, 4096, 256):
            features = encode_inputs(encoder, inputs[start:start + 256])
            _require(type(features) is np.ndarray and features.shape == (256, 32)
                     and features.dtype == np.float32 and np.isfinite(features).all(),
                     "Encoder must return finite float32 features for the complete chunk")
            scores = evaluate_head(endpoint, features, native=True)
            _require(type(scores) is np.ndarray and scores.shape == (256, 9)
                     and scores.dtype == np.float32 and np.isfinite(scores).all(),
                     "Head must return all nine finite float32 scores for the complete chunk")
            predictions = np.argmax(scores, axis=1)
            truth = labels[start:start + 256]
            for label in range(8):
                correct_by_class[label] += int(np.count_nonzero((truth == label) & (predictions == label)))
            reserved_predictions += int(np.count_nonzero(predictions == 8))
        after = _runtime(torch)
        _require(before == after, "Observable runtime configuration changed during the slot")

    _require(denominators == [512] * 8
             and all(0 <= correct <= denominator
                     for correct, denominator in zip(correct_by_class, denominators)),
             "Incomplete whole-slot counts")
    return {
        "schema": "wsp-staged-native-slot-counts/1",
        "binding": actual_binding,
        "execution": execution.to_dict(),
        "chunk_size": 256,
        "chunks_completed": 16,
        "counts": {
            "correct_by_class": correct_by_class, "denominator_by_class": denominators,
            "total_correct": sum(correct_by_class), "total_denominator": 4096,
            "reserved_output_predictions": reserved_predictions,
        },
        "runtime_correspondence": before,
    }
