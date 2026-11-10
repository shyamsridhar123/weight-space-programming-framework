"""Pinned raw-text serving in fresh worker processes; no study inputs are loaded.

The package contains manifest.json, the four original text-parent files,
deployed-head.json, and policy.json. The caller supplies raw manifest and batch
SHA-256 pins. Only unchanged and edited-readout policies are supported. Admission
is a supplied declaration, not rerun or authenticated here; a rejected candidate
identity is declared only. Source hashes are correspondence records, not execution
attestation. Every call starts from raw text, including warmup and measured calls.
"""
from __future__ import annotations

import time

_IMPORT_STARTED = time.perf_counter()

import argparse
from collections import Counter
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.research_tasks import (
    HEAD_BIAS, PARENT_FILES, _json, _number, _sha, encode_inputs, evaluate_head,
    parent_from_bytes,
)

_IMPORT_SECONDS = time.perf_counter() - _IMPORT_STARTED
PACKAGE_FILES = (*PARENT_FILES, "deployed-head.json", "policy.json")
STAGES = ("preprocessing", "encoder", "head_and_policy", "raw_call")
FLAGS = {"provenance_authenticated": False, "admission_independently_verified": False,
         "scientific_freshness_verified": False, "evaluator_execution_attested": False}
TIMING_SCOPE = {
    "imports": "Worker module imports after the initial time import; interpreter startup is excluded.",
    "native_setup": "Torch import and deterministic CPU/thread setup, after module imports.",
    "load": "Source hashes and captured package/batch read, hash, parse, restore, and validation.",
    "exclusive_stages": "Preprocessing; encoder including copies/validation; native head, argmax, and queue mapping.",
    "raw_call": "Independent wall around all three stages, including intervening durable progress writes; output hashing is excluded.",
    "process": "Parent wall from before launch until exit or timeout kill/reap; includes imports, load, all calls and child output writes.",
    "cache": "Fresh process per job, with no claim of cold operating-system file caches.",
    "failures": "An interrupted stage has null duration; failed jobs never contribute success latency samples.",
    "memory": "Not measured; memory usage is unknown.",
    "energy": "Not measured; energy use is unknown.",
}


def _hash(value: Any) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("sha256_required")
    return value


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _snapshot(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_identity() -> dict:
    paths = [Path(__file__).resolve(), *sorted((ROOT / "rlcd/weight_space").glob("*.py")),
             ROOT / "pyproject.toml", ROOT / "requirements.lock.txt"]
    return {path.relative_to(ROOT).as_posix(): _sha(path.read_bytes()) for path in paths}


def _native_environment() -> dict:
    import torch
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {"python": platform.python_version(), "platform": sys.platform,
            "machine": platform.machine(), "numpy": np.__version__, "torch": torch.__version__,
            "numeric_lane": "Torch CPU float32", "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device": str(torch.get_default_device()), "argmax": "lowest output index",
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def _package_bytes(directory: Path, name: str) -> bytes:
    path = (directory / name).resolve(strict=True)
    if path.parent != directory:
        raise ValueError("package_path_escape")
    return path.read_bytes()


def _policy_evidence(policy: Any, parent: str, candidate: str | None, deployed: str) -> str:
    if (type(policy) is not dict or set(policy) != {
            "schema", "admitted", "requested_arm", "effective_arm", "configuration", "queue_names", "queue_map"}
            or policy["schema"] != "wsp-raw-text-serving-policy/1"
            or type(policy["admitted"]) is not bool or policy["configuration"] != {}):
        raise ValueError("invalid_serving_policy")
    if any(policy[key] not in ("unchanged", "edited-readout") for key in ("requested_arm", "effective_arm")):
        raise ValueError("unsupported_serving_policy")
    if policy["requested_arm"] == "unchanged":
        if candidate is not None or policy["effective_arm"] != "unchanged":
            raise ValueError("invalid_deployment_selection")
        selected, evidence = parent, "not_applicable"
    else:
        if candidate is None:
            raise ValueError("candidate_identity_required")
        effective = "edited-readout" if policy["admitted"] else "unchanged"
        if policy["effective_arm"] != effective:
            raise ValueError("invalid_deployment_selection")
        selected = candidate if policy["admitted"] else parent
        evidence = "verified_deployed_bytes" if policy["admitted"] else "declared_only"
    if deployed != selected:
        raise ValueError("invalid_deployment_selection")
    names, mapping = policy["queue_names"], policy["queue_map"]
    if (type(names) is not list or not names
            or any(type(name) is not str or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("invalid_queue_names")
    if (type(mapping) is not list or not mapping
            or any(type(code) is not int or not 0 <= code < len(names) for code in mapping)):
        raise ValueError("invalid_queue_map")
    return evidence


def load_package(package: Path, manifest_sha256: str, batch: Path, batch_sha256: str) -> dict:
    """Restore only pinned bytes; never follow lineage's historical file entries."""
    _hash(manifest_sha256)
    _hash(batch_sha256)
    directory = Path(package).resolve(strict=True)
    manifest_bytes = _package_bytes(directory, "manifest.json")
    if _sha(manifest_bytes) != manifest_sha256:
        raise ValueError("manifest_identity_mismatch")
    manifest = _json(manifest_bytes)
    if (type(manifest) is not dict or set(manifest) != {
            "schema", "files", "parent_sha256", "candidate_sha256", "deployed_sha256"}
            or manifest["schema"] != "wsp-raw-text-serving-package/1"
            or type(manifest["files"]) is not dict or set(manifest["files"]) != set(PACKAGE_FILES)):
        raise ValueError("invalid_package_manifest")
    for key in ("parent_sha256", "deployed_sha256"):
        _hash(manifest[key])
    if manifest["candidate_sha256"] is not None:
        _hash(manifest["candidate_sha256"])
    captured = {}
    for name in PACKAGE_FILES:
        record = manifest["files"][name]
        if (type(record) is not dict or set(record) != {"sha256", "bytes"}
                or type(record["bytes"]) is not int or record["bytes"] < 0):
            raise ValueError("invalid_package_file_record")
        _hash(record["sha256"])
        data = _package_bytes(directory, name)
        if len(data) != record["bytes"] or _sha(data) != record["sha256"]:
            raise ValueError("package_file_identity_mismatch")
        captured[name] = data
    parent = parent_from_bytes({name: captured[name] for name in PARENT_FILES})
    deployed = TensorArtifact.from_dict(_json(captured["deployed-head.json"]))
    if (parent["head"].sha256 != manifest["parent_sha256"]
            or deployed.sha256 != manifest["deployed_sha256"]
            or not deployed.compatible(parent["head"])):
        raise ValueError("deployed_head_identity_mismatch")
    # Artifacts store binary64; edited values need not round-trip through float32.
    with np.errstate(over="raise", invalid="raise"):
        if any(not np.isfinite(value.astype(np.float32)).all() for value in deployed.tensors.values()):
            raise ValueError("deployed_head_native_dtype_mismatch")
    policy = _json(captured["policy.json"])
    candidate = manifest["candidate_sha256"]
    candidate_evidence = _policy_evidence(policy, manifest["parent_sha256"], candidate, deployed.sha256)
    names, mapping = policy["queue_names"], policy["queue_map"]
    if len(mapping) != deployed.tensors[HEAD_BIAS].shape[0]:
        raise ValueError("invalid_queue_map")
    batch_bytes = Path(batch).read_bytes()
    if _sha(batch_bytes) != batch_sha256:
        raise ValueError("batch_identity_mismatch")
    envelope = _json(batch_bytes)
    if (type(envelope) is not dict or set(envelope) != {"schema", "rows"}
            or envelope["schema"] != "wsp-raw-text-batch/1"
            or type(envelope["rows"]) is not list or not envelope["rows"]):
        raise ValueError("invalid_raw_text_batch")
    rows = envelope["rows"]
    if any(type(row) is not dict or set(row) != {"request_id", "text"}
           or type(row["request_id"]) is not str or not row["request_id"].strip()
           or type(row["text"]) is not str for row in rows):
        raise ValueError("invalid_raw_text_row")
    request_ids = [row["request_id"] for row in rows]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("duplicate_request_id")
    for array in parent["encoder"].values():
        array.setflags(write=False)
    identities = {"manifest_sha256": manifest_sha256, "batch_sha256": batch_sha256,
                  "ordered_rows_sha256": digest(rows), "request_ids": request_ids, "row_count": len(rows),
                  "files": manifest["files"], "parent_sha256": manifest["parent_sha256"],
                  "candidate_sha256": candidate, "candidate_identity_evidence": candidate_evidence,
                  "deployed_sha256": deployed.sha256, "policy_sha256": digest(policy), "policy": policy,
                  **{key: value for key, value in parent["identities"].items() if key != "head_sha256"}}
    return {"head": deployed, "encoder": parent["encoder"], "preprocessing": parent["preprocessing"],
            "rows": tuple((row["request_id"], row["text"]) for row in rows),
            "queue_names": tuple(names), "queue_map": tuple(mapping), "identities": identities}


def _array(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("invalid_native_stage_output")
    return values


def _counts(warmup: int, repeats: int) -> list[str]:
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("invalid_call_counts")
    return ["first", *(["warmup"] * warmup), *(["measured"] * repeats)]


def worker(package: Path, manifest_sha256: str, batch: Path, batch_sha256: str,
           output: Path, *, job_id: str, warmup: int = 1, repeats: int = 3) -> dict:
    phases = _counts(warmup, repeats)
    _hash(manifest_sha256)
    _hash(batch_sha256)
    if type(job_id) is not str or not job_id.strip():
        raise ValueError("job_identity_required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema": "wsp-raw-text-serving-job/1", "job_id": job_id, "status": "running",
              "binding": {"manifest_sha256": manifest_sha256, "batch_sha256": batch_sha256,
                          "warmup": warmup, "repeats": repeats},
              "sources": None, "environment": None, "identities": None,
              "initialization_seconds": {"imports": _IMPORT_SECONDS, "native_setup": None},
              "load_seconds": {"load": None}, "active_stage": None, "active_call": None,
              "calls": [{"index": i, "phase": phase, "status": "pending",
                         "seconds": dict.fromkeys(STAGES), "outputs": None}
                        for i, phase in enumerate(phases)],
              "failure": None, "timing_scope": TIMING_SCOPE, **FLAGS}
    _snapshot(output / "progress.json", record)

    def measured(name, seconds, operation):
        record["active_stage"] = name
        _snapshot(output / "progress.json", record)
        started = time.perf_counter()
        value = operation()
        seconds[name] = time.perf_counter() - started
        record["active_stage"] = None
        _snapshot(output / "progress.json", record)
        return value

    def load():
        record["sources"] = _source_identity()
        return load_package(package, manifest_sha256, batch, batch_sha256)

    try:
        record["environment"] = measured("native_setup", record["initialization_seconds"], _native_environment)
        loaded = measured("load", record["load_seconds"], load)
        record["identities"] = loaded["identities"]
        rows, encoder = loaded["rows"], loaded["encoder"]
        expected_digest = None
        for call in record["calls"]:
            record["active_call"] = call["index"]
            call["status"] = "running"
            started = time.perf_counter()
            inputs = measured("preprocessing", call["seconds"], lambda: _array(
                loaded["preprocessing"].transform([text for _, text in rows]), (len(rows), encoder["weight"].shape[1])))
            features = measured("encoder", call["seconds"], lambda: _array(
                encode_inputs(encoder, inputs), (len(rows), encoder["weight"].shape[0])))

            def route():
                scores = _array(evaluate_head(loaded["head"], features, native=True),
                                (len(rows), len(loaded["queue_map"])))
                codes = scores.argmax(axis=1).tolist()
                return codes, [loaded["queue_map"][code] for code in codes]

            codes, queues = measured("head_and_policy", call["seconds"], route)
            call["seconds"]["raw_call"] = time.perf_counter() - started
            record["active_stage"] = "output_validation"
            _snapshot(output / "progress.json", record)
            payload = {"ordered_rows_sha256": loaded["identities"]["ordered_rows_sha256"],
                       "deployed_sha256": loaded["identities"]["deployed_sha256"],
                       "policy_sha256": loaded["identities"]["policy_sha256"],
                       "rows": [{"request_id": row[0], "native_code": code, "effective_code": code,
                                 "queue": queue, "queue_name": loaded["queue_names"][queue]}
                                for row, code, queue in zip(rows, codes, queues, strict=True)]}
            call["outputs"] = payload | {"sha256": digest(payload)}
            if expected_digest is not None and call["outputs"]["sha256"] != expected_digest:
                raise ValueError("repeated_output_mismatch")
            expected_digest = call["outputs"]["sha256"]
            call["status"] = "succeeded"
            record["active_stage"], record["active_call"] = None, None
            _snapshot(output / "progress.json", record)
        record["status"] = "succeeded"
    except Exception as exc:
        record["status"] = "failed"
        record["failure"] = {"type": type(exc).__name__}
        if record["active_call"] is not None:
            record["calls"][record["active_call"]]["status"] = "failed"
    _snapshot(output / "progress.json", record)
    _write(output / "receipt.json", canonical_bytes(record) + b"\n")
    return record


def _validate_record(value: Any, job_id: str, binding: dict, sources: dict, *, complete: bool) -> str | None:
    """Check receipt correspondence and shape, without re-executing the child."""
    try:
        if (type(value) is not dict or set(value) != {
                "schema", "job_id", "status", "binding", "sources", "environment", "identities",
                "initialization_seconds", "load_seconds", "active_stage", "active_call", "calls",
                "failure", "timing_scope", *FLAGS} or value["schema"] != "wsp-raw-text-serving-job/1"):
            return "malformed_receipt"
        if value["job_id"] != job_id or canonical_bytes(value["binding"]) != canonical_bytes(binding):
            return "identity_mismatch"
        if value["sources"] is not None and value["sources"] != sources:
            return "identity_mismatch"
        if value["status"] not in ("running", "failed", "succeeded"):
            return "malformed_receipt"
        finished = value["status"] == "succeeded"
        if complete and not finished:
            return "malformed_receipt"
        if finished and (value["sources"] is None or value["environment"] is None
                         or value["active_stage"] is not None or value["active_call"] is not None):
            return "malformed_receipt"
        if value["status"] == "failed":
            failure = value["failure"]
            if (type(failure) is not dict or set(failure) != {"type"}
                    or type(failure["type"]) is not str or not failure["type"].strip()):
                return "malformed_receipt"
        elif value["failure"] is not None:
            return "malformed_receipt"
        if any(value[key] is not expected for key, expected in FLAGS.items()) or value["timing_scope"] != TIMING_SCOPE:
            return "malformed_receipt"
        phases = _counts(binding["warmup"], binding["repeats"])
        if type(value["calls"]) is not list or len(value["calls"]) != len(phases):
            return "malformed_receipt"
        active = value["active_call"]
        stage = value["active_stage"]
        if ((active is not None and (type(active) is not int or not 0 <= active < len(phases)))
                or stage not in (None, "native_setup", "load", "preprocessing", "encoder", "head_and_policy", "output_validation")
                or (stage in ("native_setup", "load") and active is not None)
                or (stage in ("preprocessing", "encoder", "head_and_policy", "output_validation") and active is None)):
            return "malformed_receipt"
        environment = value["environment"]
        if environment is not None:
            if (type(environment) is not dict or set(environment) != {
                    "python", "platform", "machine", "numpy", "torch", "numeric_lane", "threads", "interop_threads",
                    "deterministic_algorithms", "device", "argmax", "thread_environment"}
                    or any(type(environment[key]) is not str for key in ("python", "platform", "machine", "numpy", "torch"))
                    or environment["numeric_lane"] != "Torch CPU float32" or environment["device"] != "cpu"
                    or environment["argmax"] != "lowest output index" or environment["deterministic_algorithms"] is not True
                    or type(environment["threads"]) is not int or environment["threads"] != 1
                    or type(environment["interop_threads"]) is not int or environment["interop_threads"] < 1):
                return "malformed_receipt"
            thread_environment = environment["thread_environment"]
            if (type(thread_environment) is not dict
                    or set(thread_environment) != {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
                    or any(item is not None and type(item) is not str for item in thread_environment.values())):
                return "malformed_receipt"
        timings = [(value["initialization_seconds"], {"imports", "native_setup"}, finished),
                   (value["load_seconds"], {"load"}, finished)]
        identities = value["identities"]
        if identities is not None:
            if type(identities) is not dict or set(identities) != {
                    "manifest_sha256", "batch_sha256", "ordered_rows_sha256", "request_ids", "row_count", "files",
                    "parent_sha256", "candidate_sha256", "candidate_identity_evidence", "deployed_sha256", "policy_sha256",
                    "policy", "encoder_sha256", "preprocessing_sha256", "feature_id"}:
                return "malformed_receipt"
            if any(identities[key] != binding[key] for key in ("manifest_sha256", "batch_sha256")):
                return "identity_mismatch"
            if (type(identities["row_count"]) is not int or identities["row_count"] < 1
                    or type(identities["request_ids"]) is not list
                    or len(identities["request_ids"]) != identities["row_count"]
                    or any(type(item) is not str or not item.strip() for item in identities["request_ids"])
                    or len(set(identities["request_ids"])) != identities["row_count"]
                    or digest(identities["policy"]) != identities["policy_sha256"]):
                return "malformed_receipt"
            for key in ("ordered_rows_sha256", "parent_sha256", "deployed_sha256", "policy_sha256",
                        "encoder_sha256", "preprocessing_sha256", "feature_id"):
                _hash(identities[key])
            if identities["candidate_sha256"] is not None:
                _hash(identities["candidate_sha256"])
            if identities["candidate_identity_evidence"] != _policy_evidence(
                    identities["policy"], identities["parent_sha256"], identities["candidate_sha256"], identities["deployed_sha256"]):
                return "malformed_receipt"
            if type(identities["files"]) is not dict or set(identities["files"]) != set(PACKAGE_FILES):
                return "malformed_receipt"
            for file_record in identities["files"].values():
                if (type(file_record) is not dict or set(file_record) != {"sha256", "bytes"}
                        or type(file_record["bytes"]) is not int or file_record["bytes"] < 0):
                    return "malformed_receipt"
                _hash(file_record["sha256"])
        elif finished:
            return "malformed_receipt"
        output_digests = set()
        unfinished = False
        for i, (call, phase) in enumerate(zip(value["calls"], phases, strict=True)):
            if (type(call) is not dict or set(call) != {"index", "phase", "status", "seconds", "outputs"}
                    or type(call["index"]) is not int or call["index"] != i or call["phase"] != phase
                    or call["status"] not in ("pending", "running", "failed", "succeeded")
                    or (finished and call["status"] != "succeeded")
                    or (unfinished and call["status"] != "pending")):
                return "malformed_receipt"
            unfinished = unfinished or call["status"] != "succeeded"
            seconds = call["seconds"]
            if type(seconds) is not dict or set(seconds) != set(STAGES):
                return "malformed_receipt"
            timings.append((seconds, set(STAGES), call["status"] == "succeeded"))
            outputs = call["outputs"]
            if call["status"] == "pending" and (outputs is not None or any(item is not None for item in seconds.values())):
                return "malformed_receipt"
            if (call["status"] in ("running", "failed") and
                    (active != i or value["status"] != call["status"])):
                return "malformed_receipt"
            if active == i and call["status"] not in ("running", "failed"):
                return "malformed_receipt"
            if call["status"] != "pending" and identities is None:
                return "malformed_receipt"
            missing_stage = False
            for name in STAGES:
                if seconds[name] is None:
                    missing_stage = True
                elif missing_stage:
                    return "malformed_receipt"
            if seconds["raw_call"] is not None and seconds["raw_call"] + 1e-12 < sum(seconds[name] for name in STAGES[:-1]):
                return "malformed_receipt"
            if outputs is None:
                if call["status"] == "succeeded":
                    return "malformed_receipt"
                continue
            if (identities is None or missing_stage or type(outputs) is not dict
                    or set(outputs) != {"ordered_rows_sha256", "deployed_sha256", "policy_sha256", "rows", "sha256"}):
                return "malformed_receipt"
            if any(outputs[key] != identities[key] for key in ("ordered_rows_sha256", "deployed_sha256", "policy_sha256")):
                return "identity_mismatch"
            if outputs["sha256"] != digest({key: item for key, item in outputs.items() if key != "sha256"}):
                return "malformed_receipt"
            output_digests.add(outputs["sha256"])
            if type(outputs["rows"]) is not list or len(outputs["rows"]) != identities["row_count"]:
                return "malformed_receipt"
            policy = identities["policy"]
            for row, request_id in zip(outputs["rows"], identities["request_ids"], strict=True):
                if (type(row) is not dict or set(row) != {"request_id", "native_code", "effective_code", "queue", "queue_name"}
                        or row["request_id"] != request_id
                        or any(type(row[key]) is not int for key in ("native_code", "effective_code", "queue"))
                        or not 0 <= row["native_code"] < len(policy["queue_map"])
                        or row["effective_code"] != row["native_code"]
                        or row["queue"] != policy["queue_map"][row["effective_code"]]
                        or not 0 <= row["queue"] < len(policy["queue_names"])
                        or row["queue_name"] != policy["queue_names"][row["queue"]]):
                    return "malformed_receipt"
        if finished and len(output_digests) != 1:
            return "malformed_receipt"
        for seconds, keys, required in timings:
            if type(seconds) is not dict or set(seconds) != keys:
                return "malformed_receipt"
            for duration in seconds.values():
                if duration is None and not required:
                    continue
                if _number(duration) < 0:
                    return "malformed_receipt"
        if (value["initialization_seconds"]["imports"] is None
                or (environment is not None and value["initialization_seconds"]["native_setup"] is None)
                or (value["sources"] is not None and environment is None)
                or (value["load_seconds"]["load"] is not None and value["initialization_seconds"]["native_setup"] is None)
                or (identities is not None and (environment is None or value["sources"] is None or value["load_seconds"]["load"] is None))):
            return "malformed_receipt"
        return None
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, RecursionError):
        return "malformed_receipt"


def _read_record(path: Path, job_id: str, binding: dict, sources: dict, *, complete: bool) -> tuple:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None, "missing_receipt", None
    except OSError:
        return None, "unreadable_receipt", None
    try:
        value = _json(data)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None, "malformed_receipt", _sha(data)
    problem = _validate_record(value, job_id, binding, sources, complete=complete)
    return (None if problem else value), problem, _sha(data)


def run(package: Path, manifest_sha256: str, batch: Path, batch_sha256: str,
        output: Path, *, timeout_seconds: float, jobs: int = 1, warmup: int = 1, repeats: int = 3) -> dict:
    _counts(warmup, repeats)
    _hash(manifest_sha256)
    _hash(batch_sha256)
    if type(jobs) is not int or jobs < 1 or _number(timeout_seconds) <= 0:
        raise ValueError("invalid_job_limits")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    binding = {"manifest_sha256": manifest_sha256, "batch_sha256": batch_sha256,
               "warmup": warmup, "repeats": repeats}
    assigned = [{"job_id": f"job-{i:06d}", "status": "pending", "process_seconds": None,
                 "returncode": None, "receipt": None, "receipt_sha256": None,
                 "receipt_validation": None, "progress": None, "progress_validation": None,
                 "failure": None} for i in range(jobs)]
    record = {"schema": "wsp-raw-text-serving-run/1", "binding": binding,
              "timeout_seconds": timeout_seconds, "jobs": assigned, "sources": None,
              "summary": None, "timing_scope": TIMING_SCOPE, **FLAGS}
    # Every job is durable before source reads or the first child is launched.
    _snapshot(output / "run.json", record)
    record["sources"] = _source_identity()
    _snapshot(output / "run.json", record)
    environment = os.environ | {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for job in assigned:
        job_output = output / job["job_id"]
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--package", str(Path(package).resolve()),
                   "--manifest-sha256", manifest_sha256, "--batch", str(Path(batch).resolve()),
                   "--batch-sha256", batch_sha256, "--output", str(job_output), "--job-id", job["job_id"],
                   "--warmup", str(warmup), "--repeats", str(repeats)]
        job["status"] = "running"
        _snapshot(output / "run.json", record)
        stdout, stderr = b"", b""
        started = time.perf_counter()
        try:
            child = subprocess.run(command, cwd=ROOT, env=environment, timeout=timeout_seconds,
                                   capture_output=True, check=False)
            job["process_seconds"] = time.perf_counter() - started
            job["returncode"] = child.returncode
            stdout, stderr = child.stdout, child.stderr
            job["status"] = "nonzero_exit" if child.returncode else "succeeded"
        except subprocess.TimeoutExpired as exc:
            job["process_seconds"] = time.perf_counter() - started
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
            job["status"] = "timeout"
        except OSError as exc:
            job["process_seconds"] = time.perf_counter() - started
            job["status"], job["failure"] = "launch_failed", {"type": type(exc).__name__}
        _write(output / (job["job_id"] + ".stdout.log"), stdout)
        _write(output / (job["job_id"] + ".stderr.log"), stderr)
        job["receipt"], problem, job["receipt_sha256"] = _read_record(
            job_output / "receipt.json", job["job_id"], binding, record["sources"], complete=job["status"] == "succeeded")
        job["receipt_validation"] = problem or "accepted"
        if job["status"] == "succeeded" and problem:
            job["status"] = problem
        if job["status"] != "succeeded":
            job["progress"], problem, _ = _read_record(
                job_output / "progress.json", job["job_id"], binding, record["sources"], complete=False)
            job["progress_validation"] = problem or "accepted_partial"
        _snapshot(output / "run.json", record)
    successful = [job for job in assigned if job["status"] == "succeeded"]
    samples = [{"job_id": job["job_id"], "call_index": call["index"], "seconds": call["seconds"]}
               for job in successful for call in job["receipt"]["calls"] if call["phase"] == "measured"]
    record["summary"] = {"assigned_jobs": jobs, "successful_jobs": len(successful),
                         "unsuccessful_jobs": jobs - len(successful),
                         "dispositions": dict(Counter(job["status"] for job in assigned)),
                         "measured_calls_assigned": jobs * repeats,
                         "measured_calls_excluded_from_success_samples": jobs * repeats - len(samples),
                         "conditional_success_samples": samples}
    _snapshot(output / "run.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--package", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
        command.add_argument("--batch", type=Path, required=True)
        command.add_argument("--batch-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--warmup", type=int, default=1)
        command.add_argument("--repeats", type=int, default=3)
        if name == "run":
            command.add_argument("--jobs", type=int, default=1)
            command.add_argument("--timeout-seconds", type=float, required=True)
        else:
            command.add_argument("--job-id", required=True)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    result = run(**arguments) if command == "run" else worker(**arguments)
    print(canonical_bytes(result["summary"] if command == "run" else {
        "job_id": result["job_id"], "status": result["status"]}).decode())
    return int(result["summary"]["unsuccessful_jobs"] > 0) if command == "run" else int(result["status"] != "succeeded")


if __name__ == "__main__":
    raise SystemExit(main())
