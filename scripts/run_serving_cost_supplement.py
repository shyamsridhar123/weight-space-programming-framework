"""Separately frozen, cached-development head serving and package I/O supplement.

This does not train, select, score heldout labels, or change either trained study.
Run freeze once, self-check, then run with the same script and locked protocol.
"""
from __future__ import annotations

import time
_PYTHON_ENTRY_NS = time.perf_counter_ns()

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/research/serving-cost-supplement"
ARTIFACTS = ROOT / "artifacts/research/serving-cost-supplement"
SELECTIONS = (("banking77", "trained-support", 7001, "ownership/card_arrival"),
              ("optical-digits", "trained-nontext", 8001, "ownership/8"))
GATE = "artifacts/research/live-studies/global-selection-seal.json"
GATE_RAW = "26e67134db350133a2f3ee2a324c39d79d53fd6915ee8513bdae37cc3fcbdc31"
GATE_CANONICAL = "601897dc212592ae76cb2eed4b56e120de6ed9d66fa292b7e42ba5b924a13d0f"
RELEASE = "artifacts/research/live-studies/evidence-release.json"
RELEASE_RAW = "76d7a7ef8bd49c090341342846ff0bb68a29b3f82c99fbda670794d61289f097"
HANDOFF_SHA = "8ca4f924e5280ae2f38678290e666921c647fbb9b93e5a19020de8aa6159ae77"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(path, expected):
    if identity(path) != expected:
        raise ValueError(f"File identity mismatch: {path.name}")


def now():
    return datetime.now(timezone.utc).isoformat()


def runner(study):
    path = ROOT / "experiments/research" / study / "source-freeze/scripts/run_trained_research.py"
    spec = importlib.util.spec_from_file_location("cost_frozen_" + study.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locked_protocol():
    lock = read(EXPERIMENT / "protocol-lock.json")
    verify(EXPERIMENT / "protocol.json", lock["protocol"])
    verify(Path(__file__), lock["script"])
    protocol = read(EXPERIMENT / "protocol.json")
    for relative, expected in protocol["fixed_input_files"].items():
        verify(ROOT / relative, expected)
    return protocol


def selected_rows(np, bundle, indices):
    source = bundle["target_row_indices"]
    positions = [np.flatnonzero(source == row) for row in indices]
    if any(len(position) != 1 for position in positions):
        raise ValueError("Target anchor identity is missing or ambiguous")
    return bundle["target_features"][[int(position[0]) for position in positions]].copy()


def batch_identity(r, features, ids, rows):
    return r.digest({"features": features.tolist(), "row_ids": ids.tolist(), "row_indices": rows.tolist()})


def deployed_policy(attempt, queue_map):
    admitted = bool(attempt["admitted"])
    return {"case": attempt["case"], "requested_arm": attempt["arm"],
            "effective_arm": attempt["arm"] if admitted else "unchanged",
            "admitted": admitted, "abstention_reasons": attempt["abstention_reasons"],
            "configuration": attempt["configuration"] if admitted else {},
            "queue_map": queue_map.tolist()}


def freeze():
    if (EXPERIMENT / "protocol-lock.json").exists():
        raise ValueError("Existing supplement lock is immutable; use its run phase")
    if identity(ROOT / GATE)["sha256"] != GATE_RAW or identity(ROOT / RELEASE)["sha256"] != RELEASE_RAW:
        raise ValueError("The independently authenticated release anchor changed")
    files = {relative: identity(ROOT / relative) for relative in (GATE, RELEASE)}
    studies = []
    for domain, study, seed, case in SELECTIONS:
        exp = ROOT / "experiments/research" / study
        protocol = read(exp / "protocol.json")
        seal = read(exp / "global-seal.json")
        replay = read(exp / "independent-replay.json")
        for relative in ("protocol.json", "global-seal.json", "source-freeze/source-manifest.json", "independent-replay.json"):
            path = exp / relative
            files[path.relative_to(ROOT).as_posix()] = identity(path)
        for relative, expected in protocol["source_files"].items():
            path = exp / "source-freeze" / relative
            verify(path, expected)
            files[path.relative_to(ROOT).as_posix()] = expected
        if (not replay["all_replayable_trials_verified"] or replay["problems"]
                or replay["protocol_sha256"] != identity(exp / "protocol.json")["sha256"]
                or replay["global_seal_sha256"] != identity(exp / "global-seal.json")["sha256"]):
            raise ValueError("Matching successful trained replay is required")
        r = runner(study)
        lineage_id = f"{domain}-main-full-{seed}"
        directory = ROOT / "artifacts/research" / study / "main" / lineage_id
        item = {"domain": domain, "study": study, "seed": seed, "case_id": case,
                "lineage_id": lineage_id, "lineage_path": directory.relative_to(ROOT).as_posix()}
        if not (directory / "lineage.json").is_file():
            item.update(status="preparation_missing", arms=[])
            studies.append(item)
            continue
        lineage = read(directory / "lineage.json")
        files[(directory / "lineage.json").relative_to(ROOT).as_posix()] = identity(directory / "lineage.json")
        for relative, expected in lineage["files"].items():
            verify(directory / relative, expected)
            files[(directory / relative).relative_to(ROOT).as_posix()] = expected
        base = r.TensorArtifact.from_dict(read(directory / "head.json"))
        if base.sha256 != lineage["head_sha256"]:
            raise ValueError("Parent artifact identity changed")
        with r.np.load(directory / "features.npz", allow_pickle=False) as saved:
            bundle = {key: saved[key].copy() for key in saved.files if "labels" not in key}
        batches = []
        for size in (1, 32):
            x, ids, rows = (bundle[key][:size] for key in
                            ("development_features", "development_ids", "development_row_indices"))
            if len(x) != size or x.dtype != r.np.float32 or not r.np.isfinite(x).all():
                raise ValueError("Fixed development input batch is unavailable")
            batches.append({"size": size, "row_ids": ids.tolist(), "row_indices": rows.tolist(),
                            "dtype": x.dtype.str, "shape": list(x.shape),
                            "sha256": batch_identity(r, x, ids, rows)})
        arms = []
        for arm in r.ARMS:
            trial_id = f"{lineage_id}/{case}/{arm}"
            matches = [entry for entry in seal["attempts"] if entry["trial_id"] == trial_id]
            if len(matches) != 1:
                arms.append({"arm": arm, "trial_id": trial_id, "status": "preparation_missing"})
                continue
            entry = matches[0]
            attempt_directory = ROOT / entry["relative_path"]
            attempt = read(attempt_directory / "attempt.json")
            if identity(attempt_directory / "attempt.json")["sha256"] != entry["attempt_sha256"]:
                raise ValueError("Sealed attempt changed")
            candidate = r.TensorArtifact.from_dict(read(attempt_directory / "candidate.json"))
            program = r.Program.from_dict(read(attempt_directory / "program.json"))
            if (candidate.sha256 != entry["candidate_artifact_sha256"]
                    or r.apply_program(base, program).sha256 != candidate.sha256):
                raise ValueError("Sealed candidate/application identity changed")
            for name in ("attempt.json", "candidate.json", "program.json", "rollback.json", "task.json", "retention.json"):
                path = attempt_directory / name
                if path.is_file():
                    files[path.relative_to(ROOT).as_posix()] = identity(path)
            policy = deployed_policy(attempt, bundle["queue_map"])
            anchors = selected_rows(r.np, bundle, attempt["target_rows"]) if policy["effective_arm"] == "context" else None
            if anchors is not None and r.digest(anchors.tolist()) != read(attempt_directory / "task.json")["constraints"]["target_feature_sha256"]:
                raise ValueError("Context anchors differ from sealed target features")
            head = attempt_directory / "candidate.json" if attempt["admitted"] else directory / "head.json"
            arms.append({"arm": arm, "trial_id": trial_id, "status": "prepared", "attempt_path": entry["relative_path"],
                         "admitted": attempt["admitted"], "policy": policy,
                         "candidate_weight_no_op": attempt["no_op_artifact"],
                         "deployed_weight_no_op": (candidate.sha256 == base.sha256) if attempt["admitted"] else True,
                         "parent_artifact_sha256": base.sha256, "candidate_artifact_sha256": candidate.sha256,
                         "deployed_artifact_sha256": candidate.sha256 if attempt["admitted"] else base.sha256,
                         "deployed_head_path": head.relative_to(ROOT).as_posix(),
                         "anchor_sha256": r.digest(anchors.tolist()) if anchors is not None else None,
                         "anchor_shape": list(anchors.shape) if anchors is not None else None})
        item.update(status="prepared", batches=batches, arms=arms, feature_id=lineage["feature_id"],
                    encoder_sha256=lineage["encoder_sha256"], preprocessing_sha256=lineage["preprocessing_sha256"])
        studies.append(item)
    protocol = {"schema": "wsp-serving-cost-supplement-protocol/1", "frozen_at_utc": now(),
                "global_gate_canonical_sha256": GATE_CANONICAL, "global_gate_raw_sha256": GATE_RAW,
                "pre_release_measurement_handoff_sha256": HANDOFF_SHA,
                "selection_scope": "First declared seed and ownership case, all eight arms; declared before outcome access",
                "studies": studies, "fixed_input_files": files, "batch_sizes": [1, 32],
                "fresh_process_repetitions": 3, "warmup_calls": 10, "measured_warm_calls": 30,
                "torch_threads": 4, "copy_repetitions": 3,
                "order": "Domain order above; declared arm order; batch 1 then 32; repetition 1 through 3; package I/O after serving",
                "serving_boundary": "One frozen external_codes native head call, actual admitted overlay/context, then queue_map[codes]",
                "startup_boundary": "Parent process launch to child first Python perf_counter marker; includes interpreter initialization/scheduling",
                "import_boundary": "Child marker through stdlib, protocol/source verification, serving dependency imports and Torch setup; verification is nested",
                "load_boundary": "Verify/read serving package and input bytes, parse deployed head/policy/anchors, validate immutable batch and endpoint identities",
                "export_boundary": "Read/derive/write minimal policy, optional anchors, cached batches and package copies; final manifest hashing/export excluded",
                "package_allowlist": "encoder.npz, preprocessing.json, deployed-head.json, policy.json, context-anchors.npz only if deployed context; audit/program.json, rollback.json and retention.json when present",
                "copy_boundary": "Inventory/stat allowed files; separately shutil.copy2 all serving/audit files; separately SHA256 copied bytes; directory setup and final validation outside timers",
                "cache_conditions": "OS cache uncontrolled; fresh-process first call is not cold disk; hashing follows copying and uses warm cache; no workspace performance isolation",
                "claim_limits": ["Head-only cached features; no preprocessing or encoder execution cost", "Engineering repetitions of one lineage per domain, not independent training trials", "No business cost/savings, peak memory, energy, transformer scale, full-model retraining, or complete amortization frontier", "Do not add nested or supplemental counters to historical broad timers"]}
    write(EXPERIMENT / "protocol.json", protocol)
    write(EXPERIMENT / "protocol-lock.json", {"schema": "wsp-serving-cost-supplement-lock/1", "locked_at_utc": now(),
                                               "protocol": identity(EXPERIMENT / "protocol.json"), "script": identity(Path(__file__))})
    print("supplement script and protocol frozen; no serving or I/O measurements run", flush=True)


def prepare(protocol):
    if (ARTIFACTS / "package-manifest.json").exists():
        raise ValueError("Existing supplement evidence is immutable; do not overwrite a run")
    packages, batches, exports = [], {}, []
    for study in protocol["studies"]:
        if study["status"] != "prepared":
            continue
        r = runner(study["study"])
        started = time.perf_counter_ns()
        directory = ROOT / study["lineage_path"]
        with r.np.load(directory / "features.npz", allow_pickle=False) as saved:
            bundle = {key: saved[key].copy() for key in saved.files if "labels" not in key}
        for batch in study["batches"]:
            size = batch["size"]
            path = ARTIFACTS / "cached-development" / study["domain"] / f"batch-{size}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            r.np.savez_compressed(path, features=bundle["development_features"][:size],
                                 row_ids=bundle["development_ids"][:size], row_indices=bundle["development_row_indices"][:size])
            batches[f"{study['domain']}/{size}"] = {"path": path.relative_to(ROOT).as_posix(), "identity": identity(path), **batch}
        exports.append({"domain": study["domain"], "scope": "Shared cached-development feature package read/assembly/export and batch hashing", "seconds": (time.perf_counter_ns()-started)/1e9})
        for arm in study["arms"]:
            if arm["status"] != "prepared":
                continue
            started = time.perf_counter_ns()
            package = ARTIFACTS / "packages" / study["domain"] / arm["arm"]
            package.mkdir(parents=True, exist_ok=True)
            serving = ["encoder.npz", "preprocessing.json", "deployed-head.json", "policy.json"]
            shutil.copy2(directory / "encoder.npz", package / "encoder.npz")
            shutil.copy2(directory / "preprocessing.json", package / "preprocessing.json")
            shutil.copy2(ROOT / arm["deployed_head_path"], package / "deployed-head.json")
            write(package / "policy.json", arm["policy"])
            if arm["anchor_sha256"] is not None:
                attempt = read(ROOT / arm["attempt_path"] / "attempt.json")
                anchors = selected_rows(r.np, bundle, attempt["target_rows"])
                if r.digest(anchors.tolist()) != arm["anchor_sha256"]:
                    raise ValueError("Exported anchor input changed")
                r.np.savez_compressed(package / "context-anchors.npz", features=anchors)
                serving.append("context-anchors.npz")
            audit = []
            for name in ("program.json", "rollback.json", "retention.json"):
                source = ROOT / arm["attempt_path"] / name
                if source.is_file():
                    target = package / "audit" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    audit.append("audit/" + name)
            exports.append({"domain": study["domain"], "arm": arm["arm"], "scope": protocol["export_boundary"], "seconds": (time.perf_counter_ns()-started)/1e9})
            included = {name: identity(package / name) for name in serving + audit}
            packages.append({"domain": study["domain"], "arm": arm["arm"], "path": package.relative_to(ROOT).as_posix(),
                             "serving_files": serving, "additional_audit_files": audit, "files": included,
                             "serving_bytes": sum(included[name]["bytes"] for name in serving),
                             "additional_audit_bytes": sum(included[name]["bytes"] for name in audit),
                             "policy": arm["policy"], "deployed_artifact_sha256": arm["deployed_artifact_sha256"]})
    manifest = {"schema": "wsp-serving-cost-package-manifest/1", "created_at_utc": now(),
                "protocol_sha256": identity(EXPERIMENT / "protocol.json")["sha256"],
                "script_sha256": identity(Path(__file__))["sha256"], "packages": packages, "batches": batches, "preparation_exports": exports}
    write(ARTIFACTS / "package-manifest.json", manifest)
    return manifest


def serve(r, head, features, policy, anchors, queue_map):
    return queue_map[r.external_codes(head, features, policy["case"], policy["effective_arm"], policy["configuration"], anchors)]


def worker(args):
    verification_started = time.perf_counter_ns()
    protocol = locked_protocol()
    verify(ARTIFACTS / "package-manifest.json", read(ARTIFACTS / "execution-lock.json")["package_manifest"])
    source_verification_seconds = (time.perf_counter_ns()-verification_started)/1e9
    study = next(item for item in protocol["studies"] if item["domain"] == args.domain)
    arm = next(item for item in study["arms"] if item["arm"] == args.arm)
    r = runner(study["study"])
    np = r.np
    import torch
    torch.set_num_threads(protocol["torch_threads"])
    torch.use_deterministic_algorithms(True)
    dependencies_ready = time.perf_counter_ns()
    started = time.perf_counter_ns()
    manifest = read(ARTIFACTS / "package-manifest.json")
    package = next(item for item in manifest["packages"] if item["domain"] == args.domain and item["arm"] == args.arm)
    directory = ROOT / package["path"]
    for name in package["serving_files"]:
        verify(directory / name, package["files"][name])
    batch = manifest["batches"][f"{args.domain}/{args.batch}"]
    verify(ROOT / batch["path"], batch["identity"])
    head = r.TensorArtifact.from_dict(read(directory / "deployed-head.json"))
    policy = read(directory / "policy.json")
    if policy != arm["policy"] or head.sha256 != arm["deployed_artifact_sha256"]:
        raise ValueError("Deployed endpoint/policy identity mismatch")
    with np.load(ROOT / batch["path"], allow_pickle=False) as saved:
        features, ids, rows = (saved[name].copy() for name in ("features", "row_ids", "row_indices"))
    if (features.dtype != np.float32 or list(features.shape) != batch["shape"]
            or not np.isfinite(features).all() or batch_identity(r, features, ids, rows) != batch["sha256"]):
        raise ValueError("Immutable cached input identity mismatch")
    anchors = np.empty((0, features.shape[1]), dtype=np.float32)
    if arm["anchor_sha256"] is not None:
        with np.load(directory / "context-anchors.npz", allow_pickle=False) as saved:
            anchors = saved["features"].copy()
        if (anchors.dtype != np.float32 or not np.isfinite(anchors).all()
                or r.digest(anchors.tolist()) != arm["anchor_sha256"]):
            raise ValueError("Deployed context anchor identity mismatch")
    queue_map = np.asarray(policy["queue_map"], dtype=np.int64)
    load_seconds = (time.perf_counter_ns()-started)/1e9
    started = time.perf_counter_ns()
    first_output = serve(r, head, features, policy, anchors, queue_map)
    first_seconds = (time.perf_counter_ns()-started)/1e9
    output_sha = r.digest(first_output.tolist())
    for _ in range(protocol["warmup_calls"]):
        output = serve(r, head, features, policy, anchors, queue_map)
        if r.digest(output.tolist()) != output_sha:
            raise ValueError("Warmup output changed")
    durations = []
    for _ in range(protocol["measured_warm_calls"]):
        started = time.perf_counter_ns()
        output = serve(r, head, features, policy, anchors, queue_map)
        durations.append((time.perf_counter_ns()-started)/1e9)
        if r.digest(output.tolist()) != output_sha:
            raise ValueError("Measured serving output changed")
    numpy_config = io.StringIO()
    with redirect_stdout(numpy_config):
        np.show_config()
    clock = time.get_clock_info("perf_counter")
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
               "os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
               "logical_cpu_count": os.cpu_count(), "device": "CPU", "torch_threads": torch.get_num_threads(),
               "torch_interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "numpy_build_config": numpy_config.getvalue(), "torch_build_config": torch.__config__.show(),
               "thread_environment": {key: os.environ.get(key) for key in
                                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS")},
               "perf_counter": {"resolution_seconds": clock.resolution, "monotonic": clock.monotonic,
                                "adjustable": clock.adjustable, "implementation": clock.implementation}}
    write(Path(args.output), {"schema": "wsp-serving-cost-process/1", "status": "success", "created_at_utc": now(),
                              "domain": args.domain, "arm": args.arm, "batch_size": args.batch, "repetition": args.repetition,
                              "policy": policy, "deployed_artifact_sha256": head.sha256, "input_sha256": batch["sha256"],
                              "input_file_identity": batch["identity"], "anchor_sha256": arm["anchor_sha256"],
                              "package_files": package["files"], "runtime": runtime, "outputs_equal": True, "output_sha256": output_sha,
                              "python_entry_ns": _PYTHON_ENTRY_NS, "module_import_and_setup_seconds": (dependencies_ready-_PYTHON_ENTRY_NS)/1e9,
                              "nested_protocol_source_verification_seconds": source_verification_seconds,
                              "model_input_read_verification_seconds": load_seconds, "fresh_process_first_call_seconds": first_seconds,
                              "warmup_calls": protocol["warmup_calls"], "warm_call_seconds": durations})


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"observations": 0, "median": None, "minimum": None, "maximum": None, "p95": None}
    position = .95 * (len(ordered)-1)
    lower = int(position)
    p95 = ordered[lower] if lower == len(ordered)-1 else ordered[lower] + (position-lower)*(ordered[lower+1]-ordered[lower])
    return {"observations": len(ordered), "median": statistics.median(ordered), "minimum": ordered[0], "maximum": ordered[-1], "p95": p95}


def package_io(protocol, manifest):
    records = []
    for package in manifest["packages"]:
        source = ROOT / package["path"]
        for repetition in range(1, protocol["copy_repetitions"]+1):
            destination = ARTIFACTS / "copy-repetitions" / package["domain"] / package["arm"] / str(repetition)
            destination.mkdir(parents=True, exist_ok=False)
            for name in package["files"]:
                (destination / name).parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter_ns()
            inventory = {name: (source / name).stat().st_size for name in package["files"]}
            inventory_seconds = (time.perf_counter_ns()-started)/1e9
            if inventory != {name: expected["bytes"] for name, expected in package["files"].items()}:
                raise ValueError("Package inventory differs from its fixed allowlist")
            started = time.perf_counter_ns()
            for name in package["files"]:
                shutil.copy2(source / name, destination / name)
            copy_seconds = (time.perf_counter_ns()-started)/1e9
            started = time.perf_counter_ns()
            hashes = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in package["files"]}
            hash_seconds = (time.perf_counter_ns()-started)/1e9
            for name, expected in package["files"].items():
                verify(destination / name, expected)
            if hashes != {name: expected["sha256"] for name, expected in package["files"].items()}:
                raise ValueError("Timed copied hashes differ from source identities")
            records.append({"domain": package["domain"], "arm": package["arm"], "repetition": repetition,
                            "filesystem_destination": destination.relative_to(ROOT).as_posix(), "path_drive": destination.drive,
                            "serving_bytes": package["serving_bytes"], "additional_audit_bytes": package["additional_audit_bytes"],
                            "inventory_stat_seconds": inventory_seconds, "copy2_seconds": copy_seconds, "hash_after_copy_seconds": hash_seconds,
                            "inventory_bytes": inventory, "copied_sha256": hashes, "verified": True})
            write(ARTIFACTS / "io-measurements.json", records)
    return records


def run():
    protocol = locked_protocol()
    manifest = prepare(protocol)
    write(ARTIFACTS / "execution-lock.json", {"schema": "wsp-serving-cost-execution-lock/1", "locked_at_utc": now(),
                                             "package_manifest": identity(ARTIFACTS / "package-manifest.json"),
                                             "protocol": identity(EXPERIMENT / "protocol.json"), "script": identity(Path(__file__))})
    records = []
    for study in protocol["studies"]:
        for arm in study["arms"]:
            for batch in protocol["batch_sizes"]:
                for repetition in range(1, protocol["fresh_process_repetitions"]+1):
                    item = {"domain": study["domain"], "arm": arm["arm"], "batch_size": batch, "repetition": repetition}
                    if arm["status"] != "prepared":
                        records.append({**item, "status": "preparation_missing"})
                        continue
                    output = ARTIFACTS / "process-measurements" / study["domain"] / arm["arm"] / f"batch-{batch}-repeat-{repetition}.json"
                    command = [sys.executable, "-B", "-u", str(Path(__file__)), "--phase", "worker", "--domain", study["domain"],
                               "--arm", arm["arm"], "--batch", str(batch), "--repetition", str(repetition), "--output", str(output)]
                    parent_started_ns = time.perf_counter_ns()
                    process = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
                    parent_elapsed_seconds = (time.perf_counter_ns()-parent_started_ns)/1e9
                    if process.returncode == 0 and output.is_file():
                        result = read(output)
                        if result["python_entry_ns"] < parent_started_ns or not result["outputs_equal"]:
                            raise ValueError("Child startup clock/output identity mismatch")
                        result.update(process_elapsed_seconds=parent_elapsed_seconds,
                                      process_startup_to_python_marker_seconds=(result["python_entry_ns"]-parent_started_ns)/1e9,
                                      parent_launch_ns=parent_started_ns)
                        write(output, result)
                        records.append({**item, "status": "success", "path": output.relative_to(ROOT).as_posix(), "identity": identity(output)})
                    else:
                        failure = {**item, "status": "failed", "exit_code": process.returncode,
                                   "process_elapsed_seconds": parent_elapsed_seconds, "stderr": process.stderr, "stdout": process.stdout}
                        write(output, failure)
                        records.append({**item, "status": "failed", "path": output.relative_to(ROOT).as_posix(), "identity": identity(output)})
                    write(ARTIFACTS / "process-index.json", records)
            print("completed serving settings", study["domain"], arm["arm"], flush=True)
    io_records = package_io(protocol, manifest)
    summaries = []
    for study in protocol["studies"]:
        for arm in study["arms"]:
            for batch in protocol["batch_sizes"]:
                assigned = [record for record in records if record["domain"] == study["domain"] and record["arm"] == arm["arm"] and record["batch_size"] == batch]
                available = [read(ROOT / record["path"]) for record in assigned if record["status"] == "success"]
                warm = [duration for record in available for duration in record["warm_call_seconds"]]
                summaries.append({"domain": study["domain"], "arm": arm["arm"], "batch_size": batch,
                                  "assigned_processes": len(assigned), "successful_processes": len(available), "failed_or_missing_processes": len(assigned)-len(available),
                                  "admitted": arm.get("admitted"), "candidate_weight_no_op": arm.get("candidate_weight_no_op"),
                                  "deployed_weight_no_op": arm.get("deployed_weight_no_op"), "effective_arm": arm.get("policy", {}).get("effective_arm"),
                                  "warm_per_batch_seconds": distribution(warm), "first_per_batch_seconds": distribution([record["fresh_process_first_call_seconds"] for record in available]),
                                  "warm_process_median_seconds": [statistics.median(record["warm_call_seconds"]) for record in available],
                                  "startup_to_python_marker_seconds": distribution([record["process_startup_to_python_marker_seconds"] for record in available]),
                                  "module_import_and_setup_seconds": distribution([record["module_import_and_setup_seconds"] for record in available]),
                                  "model_input_read_verification_seconds": distribution([record["model_input_read_verification_seconds"] for record in available]),
                                  "warm_median_rows_per_second": batch / statistics.median(warm) if warm else None})
    write(ARTIFACTS / "summary.json", {"schema": "wsp-serving-cost-supplement-summary/1", "created_at_utc": now(),
                                     "protocol_sha256": identity(EXPERIMENT / "protocol.json")["sha256"], "script_sha256": identity(Path(__file__))["sha256"],
                                     "assigned_processes": len(records), "successful_processes": sum(record["status"] == "success" for record in records),
                                     "failed_or_missing_processes": sum(record["status"] != "success" for record in records),
                                     "measurement_summaries": summaries, "io_assigned": len(manifest["packages"])*protocol["copy_repetitions"],
                                     "io_verified": sum(record["verified"] for record in io_records), "claim_limits": protocol["claim_limits"],
                                     "no_workspace_performance_isolation": True})
    print("supplement completed; results remain head-only, local and partial", flush=True)


def self_check():
    r = runner("trained-support")
    np = r.np
    base = r.TensorArtifact(r.ARCHITECTURE, {"head.weight": [[2., 0.], [0., 2.], [0., 0.]], "head.bias": [0., 0., -64.]})
    features = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    case = {"type": "ownership", "target_label": 0, "desired_code": 2}
    queue_map = np.array([4, 5, 6], dtype=np.int64)
    attempt = {"case": case, "arm": "overlay", "configuration": {}, "admitted": True, "abstention_reasons": []}
    policy = deployed_policy(attempt, queue_map)
    anchors = features[:1]
    assert serve(r, base, features, policy, anchors, queue_map).tolist() == [6, 5]
    fallback = deployed_policy(attempt | {"admitted": False, "abstention_reasons": ["calibration-protected-harm"]}, queue_map)
    assert fallback["effective_arm"] == "unchanged"
    assert serve(r, base, features, fallback, anchors, queue_map).tolist() == [4, 5]
    context = deployed_policy(attempt | {"arm": "context", "configuration": {"similarity_threshold": .95}}, queue_map)
    assert serve(r, base, features, context, anchors, queue_map).tolist() == [6, 5]
    assert batch_identity(r, features, np.array(["a", "b"]), np.array([1, 2])) != batch_identity(r, features, np.array(["a", "b"]), np.array([2, 1]))
    print("serving policy, rejection fallback, context, queue mapping and input identity self-check passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("freeze", "self-check", "run", "worker"))
    parser.add_argument("--domain", choices=("banking77", "optical-digits"))
    parser.add_argument("--arm")
    parser.add_argument("--batch", type=int, choices=(1, 32))
    parser.add_argument("--repetition", type=int, choices=(1, 2, 3))
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.phase == "worker":
        if None in (args.domain, args.arm, args.batch, args.repetition, args.output):
            parser.error("worker requires domain, arm, batch, repetition and output")
        worker(args)
    else:
        {"freeze": freeze, "self-check": self_check, "run": run}[args.phase]()


if __name__ == "__main__":
    main()
