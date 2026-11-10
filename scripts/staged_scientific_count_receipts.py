"""Create-once byte receipts for complete staged native count claims.

Validation binds claimed counts to retained identities; it does not attest
execution, provenance, independent storage, or power-loss durability.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from rlcd.weight_space.model import canonical_bytes, digest


_RESULT_KEYS = {
    "schema", "binding", "execution", "chunk_size", "chunks_completed", "counts",
    "runtime_correspondence",
}
_EXECUTION = {
    "backend": "torch", "dtype": "float32", "device": "cpu", "accumulation": "matmul",
    "endpoint_cast": "round", "error_model": "empirical",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fields(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict, label + " requires a plain dictionary")
    captured = value.copy()
    _require(all(type(key) is str for key in captured) and set(captured) == keys,
             label + " has unexpected fields")
    return captured


def _hash(value: Any) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "Identities must be plain lowercase SHA-256 strings")


def _binding(value: Any) -> dict[str, str]:
    captured = _fields(value, {
        "endpoint_sha256", "encoder_sha256", "preprocessing_sha256", "feature_id", "rows_sha256",
    }, "Binding")
    for identity in captured.values():
        _hash(identity)
    return captured


def _capture_body(result: Any, expected_binding: dict[str, str]) -> dict[str, Any]:
    body = _fields(result, _RESULT_KEYS, "Complete slot result")
    _require(type(body["schema"]) is str and body["schema"] == "wsp-staged-native-slot-counts/1",
             "Unsupported complete-slot count schema")
    body["binding"] = _binding(body["binding"])
    _require(body["binding"] == expected_binding, "Count binding differs from retained expectations")
    body["execution"] = _fields(body["execution"], set(_EXECUTION), "Execution")
    _require(all(type(value) is str for value in body["execution"].values())
             and body["execution"] == _EXECUTION, "Unsupported native execution declaration")
    _require(type(body["chunk_size"]) is int and body["chunk_size"] == 256
             and type(body["chunks_completed"]) is int and body["chunks_completed"] == 16,
             "A complete slot requires sixteen chunks of 256 rows")

    counts = _fields(body["counts"], {
        "correct_by_class", "denominator_by_class", "total_correct", "total_denominator",
        "reserved_output_predictions",
    }, "Counts")
    for name in ("correct_by_class", "denominator_by_class"):
        _require(type(counts[name]) is list, "Class counts require plain lists")
        counts[name] = counts[name].copy()
        _require(len(counts[name]) == 8
                 and all(type(value) is int and 0 <= value <= 512 for value in counts[name]),
                 "Class counts require eight plain integers in [0, 512]")
    _require(counts["denominator_by_class"] == [512] * 8,
             "Each ordinary class requires a denominator of 512")
    _require(type(counts["total_correct"]) is int
             and counts["total_correct"] == sum(counts["correct_by_class"])
             and type(counts["total_denominator"]) is int and counts["total_denominator"] == 4096,
             "Whole-slot count totals are inconsistent")
    _require(type(counts["reserved_output_predictions"]) is int
             and 0 <= counts["reserved_output_predictions"] <= 4096 - counts["total_correct"],
             "Reserved predictions exceed the incorrect predictions")
    body["counts"] = counts

    runtime = _fields(body["runtime_correspondence"], {
        "torch_version", "torch_git_version", "torch_cuda_version", "torch_hip_version",
        "torch_config_sha256", "numpy_version", "threads", "interop_threads",
        "deterministic_algorithms", "deterministic_warn_only", "mkldnn_enabled",
        "matmul_precision", "default_device", "default_dtype", "cpu_autocast_enabled",
        "cpu_autocast_dtype",
    }, "Runtime correspondence")
    _require(all(type(runtime[name]) is str for name in (
        "torch_version", "numpy_version", "matmul_precision", "default_device", "default_dtype",
        "cpu_autocast_dtype",
    )), "Runtime version, device and dtype observations require plain strings")
    _require(runtime["torch_git_version"] is None or type(runtime["torch_git_version"]) is str,
             "Torch Git metadata requires a plain string or null")
    _hash(runtime["torch_config_sha256"])
    _require(all(type(runtime[name]) is int and runtime[name] > 0
                 for name in ("threads", "interop_threads")),
             "Runtime thread observations require positive plain integers")
    _require(all(type(runtime[name]) is bool for name in (
        "deterministic_algorithms", "deterministic_warn_only", "mkldnn_enabled", "cpu_autocast_enabled",
    )), "Runtime policy observations require actual booleans")
    _require(runtime["torch_version"].split("+", 1)[0] == "2.8.0"
             and runtime["torch_cuda_version"] is None and runtime["torch_hip_version"] is None
             and runtime["matmul_precision"] == "highest" and runtime["default_device"] == "cpu"
             and runtime["cpu_autocast_enabled"] is False,
             "Unsupported native runtime observations")
    body["runtime_correspondence"] = runtime
    return body


def _plain(path: Path, *, directory: bool = False) -> None:
    info = path.lstat()
    _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 1024,
             "Links and reparse points are not admissible receipt paths")
    _require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
             "Expected an ordinary receipt directory or file")


def read_complete_slot_counts(
    path: str | os.PathLike[str], *, expected_binding: dict[str, str], expected_receipt_sha256: str,
) -> dict[str, Any]:
    """Read canonical claims against independently retained binding and body digest."""
    binding = _binding(expected_binding)
    _hash(expected_receipt_sha256)
    path = Path(path).absolute()
    for parent in path.parents:
        _plain(parent, directory=True)
    _plain(path)
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        document = {}
        for key, value in items:
            _require(key not in document, "Duplicate JSON key")
            document[key] = value
        return document

    def constant(value: str) -> None:
        raise ValueError("Nonfinite JSON constant: " + value)

    receipt = _fields(json.loads(raw, object_pairs_hook=pairs, parse_constant=constant),
                      _RESULT_KEYS | {"sha256"}, "Count receipt")
    _hash(receipt["sha256"])
    body = _capture_body({key: value for key, value in receipt.items() if key != "sha256"}, binding)
    _require(raw == canonical_bytes(receipt) + b"\n", "Count receipt bytes are not canonical")
    _require(digest(body) == receipt["sha256"] == expected_receipt_sha256,
             "Count receipt differs from its retained body digest")
    return body | {"sha256": receipt["sha256"]}


def write_complete_slot_counts_once(
    path: str | os.PathLike[str], result: dict[str, Any], *, expected_binding: dict[str, str],
) -> dict[str, Any]:
    """Capture, validate, create exclusively, flush, fsync and strictly reread once.

    An existing destination or any failed write remains a failure. No partial
    file is removed, overwritten or retried by this helper.
    """
    binding = _binding(expected_binding)
    body = _capture_body(result, binding)
    receipt = body | {"sha256": digest(body)}
    raw = canonical_bytes(receipt) + b"\n"
    path = Path(path).absolute()
    for parent in path.parents:
        _plain(parent, directory=True)
    try:
        _plain(path)
    except FileNotFoundError:
        pass
    with path.open("xb") as stream:
        _require(stream.write(raw) == len(raw), "Incomplete count receipt write")
        stream.flush()
        os.fsync(stream.fileno())
    reread = read_complete_slot_counts(path, expected_binding=binding,
                                      expected_receipt_sha256=receipt["sha256"])
    # The reader already checked that the stored bytes equal this exact encoding.
    _require(canonical_bytes(reread) + b"\n" == raw, "Count receipt bytes changed during reread")
    return reread
