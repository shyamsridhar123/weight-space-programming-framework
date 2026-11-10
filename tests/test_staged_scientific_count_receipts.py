"""Literal count-persistence checks; no evaluator, registry or scientific data."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import staged_scientific_count_receipts as receipts


def _result():
    return {
        "schema": "wsp-staged-native-slot-counts/1",
        "binding": {
            "endpoint_sha256": "a" * 64, "encoder_sha256": "b" * 64,
            "preprocessing_sha256": "c" * 64, "feature_id": "d" * 64,
            "rows_sha256": "e" * 64,
        },
        "execution": {
            "backend": "torch", "dtype": "float32", "device": "cpu",
            "accumulation": "matmul", "endpoint_cast": "round", "error_model": "empirical",
        },
        "chunk_size": 256, "chunks_completed": 16,
        "counts": {
            "correct_by_class": [512, 512, 0, 0, 0, 0, 0, 0],
            "denominator_by_class": [512] * 8, "total_correct": 1024,
            "total_denominator": 4096, "reserved_output_predictions": 1024,
        },
        "runtime_correspondence": {
            "torch_version": "2.8.0+cpu", "torch_git_version": None,
            "torch_cuda_version": None, "torch_hip_version": None,
            "torch_config_sha256": "f" * 64, "numpy_version": "recorded-numpy",
            "threads": 3, "interop_threads": 2,
            "deterministic_algorithms": False, "deterministic_warn_only": True,
            "mkldnn_enabled": False, "matmul_precision": "highest",
            "default_device": "cpu", "default_dtype": "torch.float64",
            "cpu_autocast_enabled": False, "cpu_autocast_dtype": "torch.float32",
        },
    }


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _receipt(body):
    return {**deepcopy(body), "sha256": hashlib.sha256(_encode(body)).hexdigest()}


def test_canonical_roundtrip_captures_inputs_and_preserves_recorded_runtime(tmp_path, monkeypatch):
    body = _result()
    expected = deepcopy(body["binding"])
    retained_binding = expected.copy()
    retained = _receipt(body)
    raw = _encode(retained) + b"\n"
    path = tmp_path / "counts.json"
    original_open = Path.open

    def mutate_after_capture(destination, mode="r", *args, **kwargs):
        if destination == path and mode == "xb":
            body["counts"]["correct_by_class"][0] = 0
            body["runtime_correspondence"]["numpy_version"] = "changed"
            expected["endpoint_sha256"] = "0" * 64
        return original_open(destination, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", mutate_after_capture)
    written = receipts.write_complete_slot_counts_once(path, body, expected_binding=expected)
    assert written == retained and path.read_bytes() == raw
    written["counts"]["correct_by_class"][0] = 0
    read = receipts.read_complete_slot_counts(
        path, expected_binding=retained_binding, expected_receipt_sha256=retained["sha256"])
    assert read == retained
    read["runtime_correspondence"]["threads"] = 999
    assert path.read_bytes() == raw

    # All wrong outputs need not be reserved-output predictions.
    for correct, reserved in (([512] * 8, 0), ([0] * 8, 4096), ([0] * 8, 0)):
        other = _result()
        other["counts"].update(correct_by_class=correct, total_correct=sum(correct),
                               reserved_output_predictions=reserved)
        result = receipts.write_complete_slot_counts_once(
            tmp_path / f"boundary-{sum(correct)}-{reserved}.json", other,
            expected_binding=other["binding"])
        assert result == _receipt(other)


def test_invalid_payloads_and_independent_bindings_reject_before_creation(tmp_path):
    class Dict(dict):
        pass

    class List(list):
        pass

    class String(str):
        pass

    class Integer(int):
        pass

    original = _result()
    mutations = [
        (("schema",), String(original["schema"])),
        (("schema",), "wsp-staged-native-slot-counts/2"),
        (("binding",), Dict(original["binding"])),
        (("counts",), Dict(original["counts"])),
        (("chunk_size",), 256.0), (("chunks_completed",), 15),
        (("counts", "correct_by_class"), List([512, 512, 0, 0, 0, 0, 0, 0])),
        (("counts", "correct_by_class"), [512] * 7),
        (("counts", "correct_by_class", 0), True),
        (("counts", "correct_by_class", 0), Integer(512)),
        (("counts", "correct_by_class", 0), 513),
        (("counts", "correct_by_class", 0), -1),
        (("counts", "denominator_by_class", 0), 511),
        (("counts", "denominator_by_class", 0), 512.0),
        (("counts", "total_correct"), 1023),
        (("counts", "total_denominator"), 4095),
        (("counts", "reserved_output_predictions"), True),
        (("counts", "reserved_output_predictions"), -1),
        (("counts", "reserved_output_predictions"), 3073),
        (("runtime_correspondence", "torch_version"), "2.8.1+cpu"),
        (("runtime_correspondence", "torch_cuda_version"), "12.0"),
        (("runtime_correspondence", "torch_hip_version"), "6.0"),
        (("runtime_correspondence", "torch_git_version"), 123),
        (("runtime_correspondence", "torch_config_sha256"), "F" * 64),
        (("runtime_correspondence", "threads"), True),
        (("runtime_correspondence", "interop_threads"), 0),
        (("runtime_correspondence", "matmul_precision"), "high"),
        (("runtime_correspondence", "default_device"), "cuda"),
        (("runtime_correspondence", "cpu_autocast_enabled"), True),
    ]
    for name in ("deterministic_algorithms", "deterministic_warn_only", "mkldnn_enabled"):
        mutations.append((("runtime_correspondence", name), 0))
    for name in ("numpy_version", "default_dtype", "cpu_autocast_dtype"):
        mutations.append((("runtime_correspondence", name), String("recorded")))
    for name in original["execution"]:
        mutations.append((("execution", name), "unsupported"))
    for name in original["binding"]:
        mutations.append((("binding", name), "0" * 64))
    for index, (keys, value) in enumerate(mutations):
        body = _result()
        target = body
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        path = tmp_path / f"invalid-{index}.json"
        with pytest.raises(ValueError):
            receipts.write_complete_slot_counts_once(path, body, expected_binding=original["binding"])
        assert not path.exists()

    for keys in ((), ("binding",), ("execution",), ("counts",), ("runtime_correspondence",)):
        reference = original
        for key in keys:
            reference = reference[key]
        for field in [*reference, "unexpected"]:
            body = _result()
            target = body
            for key in keys:
                target = target[key]
            if field == "unexpected":
                target[field] = None
            else:
                del target[field]
            path = tmp_path / "keys.json"
            with pytest.raises(ValueError):
                receipts.write_complete_slot_counts_once(path, body, expected_binding=original["binding"])
            assert not path.exists()

    for expected in (None, Dict(original["binding"]), {**original["binding"], "feature_id": "D" * 64}):
        path = tmp_path / "expectation.json"
        with pytest.raises(ValueError):
            receipts.write_complete_slot_counts_once(path, original, expected_binding=expected)
        assert not path.exists()


def test_strict_reread_rejects_corruption_and_coherently_rehashed_substitution(tmp_path):
    body = _result()
    retained = _receipt(body)
    raw = _encode(retained) + b"\n"
    changed = deepcopy(body)
    changed["counts"]["correct_by_class"][0] -= 1
    changed["counts"]["total_correct"] -= 1
    replacements = [
        raw[:-3], raw[:-1], raw + b"\n", b"\xef\xbb\xbf" + raw,
        b" " + raw, raw.replace(b'"chunk_size":256', b'"chunk_size":256.0'),
        raw.replace(b'"chunk_size":256', b'"chunk_size":256,"chunk_size":256'),
        raw.replace(b'"total_correct":1024', b'"total_correct":NaN'),
        raw.replace(b'"total_correct":1024', b'"total_correct":1e9999'),
        raw.replace(b'"total_correct":1024', b'"total_correct":Infinity'),
        raw.replace(b'"total_correct":1024', b'"total_correct":1023'),
        _encode(_receipt(changed)) + b"\n",
        _encode({**retained, "sha256": "0" * 64}) + b"\n",
    ]
    path = tmp_path / "reread.json"
    for replacement in replacements:
        path.write_bytes(replacement)
        with pytest.raises(ValueError):
            receipts.read_complete_slot_counts(
                path, expected_binding=body["binding"], expected_receipt_sha256=retained["sha256"])
        assert path.read_bytes() == replacement
    path.write_bytes(raw)
    for name in body["binding"]:
        wrong = {**body["binding"], name: "0" * 64}
        with pytest.raises(ValueError):
            receipts.read_complete_slot_counts(
                path, expected_binding=wrong, expected_receipt_sha256=retained["sha256"])
    for expected in (None, "0" * 64, retained["sha256"].upper(), 123):
        with pytest.raises(ValueError):
            receipts.read_complete_slot_counts(
                path, expected_binding=body["binding"], expected_receipt_sha256=expected)


def test_exclusive_creation_and_io_failures_never_recover_or_overwrite(tmp_path, monkeypatch):
    body = _result()
    path = tmp_path / "existing.json"
    path.write_bytes(b"existing bytes")
    with pytest.raises((FileExistsError, ValueError)):
        receipts.write_complete_slot_counts_once(path, body, expected_binding=body["binding"])
    assert path.read_bytes() == b"existing bytes"
    original_open = Path.open

    for phase in ("short_write", "write", "flush", "fsync", "reread"):
        path = tmp_path / f"failed-{phase}.json"
        creations = []

        class Stream:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def write(self, raw):
                if phase in ("short_write", "write"):
                    self.stream.write(raw[:11])
                    if phase == "write":
                        raise OSError("injected write failure")
                    return 11
                return self.stream.write(raw)

            def flush(self):
                if phase == "flush":
                    raise OSError("injected flush failure")
                return self.stream.flush()

            def fileno(self):
                return self.stream.fileno()

        def opened(destination, mode="r", *args, **kwargs):
            stream = original_open(destination, mode, *args, **kwargs)
            if destination == path and mode == "xb":
                creations.append(destination)
                return Stream(stream)
            return stream

        def fail(*args, **kwargs):
            raise OSError(f"injected {phase} failure")

        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", opened)
            if phase == "fsync":
                patch.setattr(receipts.os, "fsync", fail)
            if phase == "reread":
                patch.setattr(receipts, "read_complete_slot_counts", fail)
            with pytest.raises((OSError, ValueError)):
                receipts.write_complete_slot_counts_once(path, body, expected_binding=body["binding"])
            assert path.exists() and creations == [path]
            failed_bytes = path.read_bytes()
            with pytest.raises((FileExistsError, ValueError)):
                receipts.write_complete_slot_counts_once(path, body, expected_binding=body["binding"])
            assert path.read_bytes() == failed_bytes and creations == [path]


def test_ordinary_file_and_parent_checks_reject_links(tmp_path):
    body = _result()
    retained = _receipt(body)
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises((OSError, ValueError)):
        receipts.read_complete_slot_counts(
            directory, expected_binding=body["binding"], expected_receipt_sha256=retained["sha256"])
    with pytest.raises((OSError, ValueError)):
        receipts.write_complete_slot_counts_once(directory / "missing" / "counts.json", body,
                                                 expected_binding=body["binding"])
    target = tmp_path / "target.json"
    target.write_bytes(_encode(retained) + b"\n")
    linked_file = tmp_path / "file-link.json"
    linked_directory = tmp_path / "directory-link"
    try:
        os.symlink(target, linked_file)
        os.symlink(directory, linked_directory, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Local symlink creation unavailable: {error}")
    for path in (linked_file, linked_directory / "counts.json"):
        with pytest.raises(ValueError):
            receipts.write_complete_slot_counts_once(path, body, expected_binding=body["binding"])
    with pytest.raises(ValueError):
        receipts.read_complete_slot_counts(
            linked_file, expected_binding=body["binding"], expected_receipt_sha256=retained["sha256"])
    assert target.read_bytes() == _encode(retained) + b"\n"
