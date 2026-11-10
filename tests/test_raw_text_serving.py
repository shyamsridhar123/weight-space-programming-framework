"""Authored engineering fixtures; no study data, timing benchmark, or freshness evidence."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest

from rlcd.weight_space import research_tasks
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.research_tasks import HEAD_WEIGHT, TextPreprocessing, make_head
from scripts import run_raw_text_serving as runner


def _bytes(value):
    return canonical_bytes(value) + b"\n"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_record(data):
    return {"sha256": _sha(data), "bytes": len(data)}


@pytest.fixture
def serving_case(tmp_path, monkeypatch):
    preprocessing = TextPreprocessing(("invoice", "password"), (1., 1.))
    encoder = {"weight": np.array([[0, 2], [3, 0]], dtype=np.float32),
               "bias": np.array([-1, -1], dtype=np.float32)}
    parent = make_head({"encoder": encoder,
                        "head": {"weight": np.array([[0, 2], [2, 0]], dtype=np.float32),
                                 "bias": np.zeros(2, dtype=np.float32)},
                        "ancestry": "Hand-authored fixture; no training or acquired data"},
                       digest(preprocessing.to_dict()), ["billing", "access"], 0, "authored-serving")
    edited = parent.tensors[HEAD_WEIGHT].copy()
    edited[2] = [0., 35.1]
    candidate = parent.replace(**{HEAD_WEIGHT: edited})
    archive = io.BytesIO()
    np.savez(archive, **encoder)
    files = {"head.json": _bytes(parent.to_dict()), "encoder.npz": archive.getvalue(),
             "preprocessing.json": _bytes(preprocessing.to_dict())}
    poison = b"AUTHORED POISON: historical feature files must never be opened"
    historical = ("features.npz", "evaluation-features.npz")
    lineage = {"schema": "wsp-trained-lineage/1", "head_sha256": parent.sha256,
               **{key: parent.metadata[key] for key in ("encoder_sha256", "preprocessing_sha256", "feature_id")},
               "files": {**{name: _file_record(data) for name, data in files.items()},
                         **{name: _file_record(poison) for name in historical}}}
    files["lineage.json"] = _bytes(lineage)
    packages = {}
    for selection in ("baseline", "admitted", "rejected"):
        directory = tmp_path / selection
        directory.mkdir()
        deployed = candidate if selection == "admitted" else parent
        policy = {"schema": "wsp-raw-text-serving-policy/1", "admitted": selection != "rejected",
                  "requested_arm": "unchanged" if selection == "baseline" else "edited-readout",
                  "effective_arm": "edited-readout" if selection == "admitted" else "unchanged",
                  "configuration": {}, "queue_names": [f"queue-{i}" for i in range(7)],
                  "queue_map": [4, 5, 6]}
        payloads = files | {"deployed-head.json": _bytes(deployed.to_dict()), "policy.json": _bytes(policy)}
        manifest = {"schema": "wsp-raw-text-serving-package/1",
                    "files": {name: _file_record(data) for name, data in payloads.items()},
                    "parent_sha256": parent.sha256,
                    "candidate_sha256": None if selection == "baseline" else candidate.sha256,
                    "deployed_sha256": deployed.sha256}
        payloads["manifest.json"] = _bytes(manifest)
        for name, data in payloads.items():
            (directory / name).write_bytes(data)
        for name in historical:
            (directory / name).write_bytes(poison)
        packages[selection] = {"path": directory, "pin": _sha(payloads["manifest.json"]),
                               "manifest": manifest, "policy": policy, "bytes": payloads}
    rows = [{"request_id": "raw/Invoice!!", "text": "Invoice!!!"},
            {"request_id": "RAW/password?", "text": "PASSWORD reset"},
            {"request_id": "empty-id", "text": ""},
            {"request_id": "oov-id", "text": "galactic nebula"}]
    envelope = {"schema": "wsp-raw-text-batch/1", "rows": rows}
    batch = tmp_path / "batch.json"
    batch.write_bytes(_bytes(envelope))
    real_open, real_load = Path.open, np.load

    def guarded_open(path, *args, **kwargs):
        assert path.name not in historical, "historical feature file opened"
        return real_open(path, *args, **kwargs)

    def guarded_load(file, *args, **kwargs):
        assert isinstance(file, io.BytesIO), "encoder must be restored from captured bytes"
        assert kwargs.get("allow_pickle") is False
        return real_load(file, *args, **kwargs)

    def forbidden_lineage(*args, **kwargs):
        raise AssertionError("load_lineage is forbidden for raw-text serving")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(np, "load", guarded_load)
    monkeypatch.setattr(research_tasks, "load_lineage", forbidden_lineage)
    return {"root": tmp_path, "packages": packages, "parent": parent, "candidate": candidate,
            "batch": batch, "batch_pin": _sha(_bytes(envelope)), "envelope": envelope, "rows": rows}


def test_native_raw_text_calls_bind_original_rows_and_deployed_selection(serving_case, monkeypatch):
    case = serving_case
    expected_inputs = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float32)
    expected_features = np.array([[0, 2], [1, 0], [0, 0], [0, 0]], dtype=np.float32)
    transforms, encoded, heads = [], [], []
    real_transform, real_encode, real_head = TextPreprocessing.transform, runner.encode_inputs, runner.evaluate_head
    real_validate, real_snapshot, real_clock = runner._validate_record, runner._snapshot, runner.time.perf_counter
    first_validations, first_output_spans, validation_clock_positions = [], {}, {}
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return real_clock()

    def validate(value, job_id, binding, sources, *, complete):
        first_output = value["active_stage"] == "output_validation" and value["active_call"] == 0
        if first_output:
            assert complete is False and value["load_to_first_output_seconds"] is None
            assert type(value["environment"]["torch"]) is str
            first = value["calls"][0]
            assert first["status"] == "running"
            assert [row["request_id"] for row in first["outputs"]["rows"]] == [
                row["request_id"] for row in case["rows"]]
        problem = real_validate(value, job_id, binding, sources, complete=complete)
        if first_output:
            assert problem is None
            first_validations.append(job_id)
            validation_clock_positions[job_id] = clock_calls
        return problem

    def snapshot(path, value):
        if value["schema"] == "wsp-raw-text-serving-job/2" and value["calls"][0]["status"] == "succeeded":
            span = value["load_to_first_output_seconds"]
            assert math.isfinite(span) and span >= 0
            if value["job_id"] not in first_output_spans:
                assert all(call["status"] == "pending" for call in value["calls"][1:])
                assert clock_calls > validation_clock_positions[value["job_id"]]
            assert span == first_output_spans.setdefault(value["job_id"], span)
        return real_snapshot(path, value)

    def transform(preprocessing, texts):
        assert texts == [row["text"] for row in case["rows"]]
        values = real_transform(preprocessing, texts)
        np.testing.assert_array_equal(values, expected_inputs)
        transforms.append(values)
        return values

    def encode(encoder, inputs):
        assert inputs is transforms[-1]
        assert all(not array.flags.writeable for array in encoder.values())
        values = real_encode(encoder, inputs)
        np.testing.assert_array_equal(values, expected_features)
        encoded.append(values)
        return values

    def head(artifact, features, *, native=False):
        assert native is True and features is encoded[-1]
        np.testing.assert_array_equal(features, expected_features)
        heads.append(artifact.sha256)
        return real_head(artifact, features, native=native)

    monkeypatch.setattr(TextPreprocessing, "transform", transform)
    monkeypatch.setattr(runner, "encode_inputs", encode)
    monkeypatch.setattr(runner, "evaluate_head", head)
    monkeypatch.setattr(runner, "_validate_record", validate)
    monkeypatch.setattr(runner, "_snapshot", snapshot)
    monkeypatch.setattr(runner.time, "perf_counter", clock)
    assert case["candidate"].tensors[HEAD_WEIGHT][2, 1] == 35.1
    assert float(np.float32(35.1)) != 35.1
    normalized_rows = [{**row, "text": research_tasks.normalized_text(row["text"])} for row in case["rows"]]
    for selection, expected_codes, evidence in (
            ("baseline", [0, 1, 0, 0], "not_applicable"),
            ("admitted", [2, 1, 0, 0], "verified_deployed_bytes"),
            ("rejected", [0, 1, 0, 0], "declared_only")):
        package = case["packages"][selection]
        record = runner.worker(package["path"], package["pin"], case["batch"], case["batch_pin"],
                               case["root"] / (selection + "-job"), job_id=selection, warmup=1, repeats=2)
        assert record["status"] == "succeeded", record
        assert record["schema"] == "wsp-raw-text-serving-job/2"
        span = record["load_to_first_output_seconds"]
        assert span == first_output_spans[selection]
        assert span + 1e-12 >= record["load_seconds"]["load"] + record["calls"][0]["seconds"]["raw_call"]
        identities = record["identities"]
        assert identities["ordered_rows_sha256"] == digest(case["rows"])
        assert identities["ordered_rows_sha256"] != digest(normalized_rows)
        assert identities["request_ids"] == [row["request_id"] for row in case["rows"]]
        assert identities["candidate_identity_evidence"] == evidence
        assert identities["candidate_sha256"] == (None if selection == "baseline" else case["candidate"].sha256)
        deployed = case["candidate"] if selection == "admitted" else case["parent"]
        assert identities["deployed_sha256"] == deployed.sha256
        assert heads[-4:] == [deployed.sha256] * 4
        assert record["environment"]["device"] == "cpu"
        assert [call["phase"] for call in record["calls"]] == ["first", "warmup", "measured", "measured"]
        assert len({call["outputs"]["sha256"] for call in record["calls"]}) == 1
        assert len({id(call["outputs"]) for call in record["calls"]}) == 4
        for call in record["calls"]:
            assert call["outputs"]["rows"] == [
                {"request_id": row["request_id"], "native_code": code, "effective_code": code,
                 "queue": code + 4, "queue_name": f"queue-{code + 4}"}
                for row, code in zip(case["rows"], expected_codes, strict=True)]
            assert all(value is not None and value >= 0 for value in call["seconds"].values())
        assert all(record[key] is False for key in runner.FLAGS)
    assert len(transforms) == len(encoded) == len(heads) == 12
    assert len({id(values) for values in transforms}) == len({id(values) for values in encoded}) == 12
    assert first_validations == ["baseline", "admitted", "rejected"]

    # One authored fresh process exercises the CLI boundary, without a cache or benchmark claim.
    baseline = case["packages"]["baseline"]
    report = runner.run(baseline["path"], baseline["pin"], case["batch"], case["batch_pin"],
                        case["root"] / "native-child", timeout_seconds=30., warmup=0, repeats=1)
    assert report["schema"] == "wsp-raw-text-serving-run/2"
    job = report["jobs"][0]
    assert job["status"] == "succeeded", job
    assert job["returncode"] == 0 and job["receipt_validation"] == "accepted"
    receipt = job["receipt"]
    assert receipt["schema"] == "wsp-raw-text-serving-job/2"
    span = receipt["load_to_first_output_seconds"]
    assert math.isfinite(span) and span >= 0
    assert span + 1e-12 >= receipt["load_seconds"]["load"] + receipt["calls"][0]["seconds"]["raw_call"]
    assert span <= job["process_seconds"] + 1e-12
    assert receipt["environment"]["numeric_lane"] == "Torch CPU float32"
    assert receipt["environment"]["device"] == "cpu" and receipt["environment"]["threads"] == 1
    assert receipt["identities"]["manifest_sha256"] == baseline["pin"]
    assert receipt["identities"]["batch_sha256"] == case["batch_pin"]
    assert receipt["identities"]["ordered_rows_sha256"] == digest(case["rows"])
    assert receipt["identities"]["request_ids"] == [row["request_id"] for row in case["rows"]]
    assert receipt["identities"]["deployed_sha256"] == case["parent"].sha256
    assert [call["phase"] for call in receipt["calls"]] == ["first", "measured"]
    for call in receipt["calls"]:
        assert call["outputs"]["rows"] == [
            {"request_id": row["request_id"], "native_code": code, "effective_code": code,
             "queue": code + 4, "queue_name": f"queue-{code + 4}"}
            for row, code in zip(case["rows"], [0, 1, 0, 0], strict=True)]
    component_seconds = (sum(receipt["initialization_seconds"].values()) + receipt["load_seconds"]["load"]
                         + sum(call["seconds"]["raw_call"] for call in receipt["calls"]))
    assert job["process_seconds"] + 1e-12 >= component_seconds > 0
    assert report["summary"]["successful_jobs"] == 1
    assert report["summary"]["measured_calls_excluded_from_success_samples"] == 0
    assert len(report["summary"]["conditional_success_samples"]) == 1
    assert report["summary"]["load_to_first_output_samples_assigned"] == 1
    assert report["summary"]["load_to_first_output_samples_excluded_from_success_samples"] == 0
    assert report["summary"]["conditional_success_load_to_first_output_samples"] == [
        {"job_id": job["job_id"], "seconds": span}]
    assert len(transforms) == len(encoded) == len(heads) == 12


def test_package_and_batch_trust_boundary_rejects_before_forward(serving_case, monkeypatch):
    case, package = serving_case, serving_case["packages"]["baseline"]

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid package or batch reached a forward stage")

    monkeypatch.setattr(TextPreprocessing, "transform", forbidden)
    monkeypatch.setattr(runner, "encode_inputs", forbidden)
    monkeypatch.setattr(runner, "evaluate_head", forbidden)

    def reject(error, *, policy=None, manifest=None, envelope=None, manifest_pin=None, batch_pin=None, tamper=False):
        payloads = dict(package["bytes"])
        manifest = copy.deepcopy(package["manifest"] if manifest is None else manifest)
        if policy is not None:
            payloads["policy.json"] = _bytes(policy)
            manifest["files"]["policy.json"] = _file_record(payloads["policy.json"])
        payloads["manifest.json"] = _bytes(manifest)
        for name, data in payloads.items():
            (package["path"] / name).write_bytes(data)
        batch_bytes = _bytes(case["envelope"] if envelope is None else envelope)
        case["batch"].write_bytes(batch_bytes)
        if tamper:
            (package["path"] / "head.json").write_bytes(payloads["head.json"] + b" ")
        with pytest.raises(ValueError, match=error):
            runner.load_package(package["path"], manifest_pin or _sha(payloads["manifest.json"]),
                                case["batch"], batch_pin or _sha(batch_bytes))

    reject("manifest_identity_mismatch", manifest_pin="0" * 64)
    reject("package_file_identity_mismatch", tamper=True)
    reject("invalid_package_manifest", manifest=package["manifest"] | {
        "files": package["manifest"]["files"] | {"../features.npz": _file_record(b"poison")}})
    reject("deployed_head_identity_mismatch", manifest=package["manifest"] | {"deployed_sha256": "0" * 64})
    reject("invalid_deployment_selection", manifest=package["manifest"] | {"candidate_sha256": case["candidate"].sha256})
    for arm in ("frozen-overlay", "context-conditioning"):
        reject("unsupported_serving_policy", policy=package["policy"] | {"requested_arm": arm})
    reject("invalid_serving_policy", policy=package["policy"] | {"configuration": {"threshold": 1}})
    reject("invalid_serving_policy", policy=package["policy"] | {"admitted": 1})
    for mapping in ([True, 5, 6], [4., 5, 6], [-1, 5, 6], [7, 5, 6], [4, 5], []):
        reject("invalid_queue_map", policy=package["policy"] | {"queue_map": mapping})
    reject("invalid_queue_names", policy=package["policy"] | {"queue_names": ["duplicate", "duplicate"]})
    reject("invalid_deployment_selection", policy=package["policy"] | {
        "requested_arm": "edited-readout", "effective_arm": "edited-readout"},
        manifest=package["manifest"] | {"candidate_sha256": case["candidate"].sha256})
    for rows, error in (([], "invalid_raw_text_batch"),
                        ([case["rows"][0], case["rows"][0]], "duplicate_request_id"),
                        ([{"request_id": "", "text": "invoice"}], "invalid_raw_text_row"),
                        ([{"request_id": "a", "text": 1}], "invalid_raw_text_row"),
                        ([case["rows"][0] | {"label": 0}], "invalid_raw_text_row")):
        reject(error, envelope=case["envelope"] | {"rows": rows})
    reversed_envelope = case["envelope"] | {"rows": list(reversed(case["rows"]))}
    reject("batch_identity_mismatch", envelope=reversed_envelope, batch_pin=case["batch_pin"])
    loaded = runner.load_package(package["path"], package["pin"], case["batch"], _sha(_bytes(reversed_envelope)))
    assert loaded["identities"]["ordered_rows_sha256"] == digest(reversed_envelope["rows"])
    assert loaded["rows"] == tuple((row["request_id"], row["text"]) for row in reversed_envelope["rows"])


def test_loader_uses_captured_bytes_without_historical_reads(serving_case, monkeypatch):
    case, package = serving_case, serving_case["packages"]["admitted"]
    expected = {package["path"] / name for name in package["bytes"]} | {case["batch"]}
    reads = []
    real_read = Path.read_bytes

    def capture_then_change(path):
        data = real_read(path)
        if path in expected:
            reads.append(path)
            path.write_bytes(b"CHANGED AFTER CAPTURE")
        return data

    monkeypatch.setattr(Path, "read_bytes", capture_then_change)
    loaded = runner.load_package(package["path"], package["pin"], case["batch"], case["batch_pin"])
    assert set(reads) == expected and len(reads) == len(expected)
    assert loaded["head"].sha256 == case["candidate"].sha256
    assert loaded["identities"]["ordered_rows_sha256"] == digest(case["rows"])
    assert all(not values.flags.writeable for values in loaded["encoder"].values())


@pytest.mark.parametrize("failure_call", [0, 1])
def test_worker_failure_keeps_partial_stage_evidence_and_refuses_reuse(serving_case, monkeypatch, failure_call):
    case, package = serving_case, serving_case["packages"]["baseline"]
    real_encode, real_snapshot = runner.encode_inputs, runner._snapshot
    encoder_calls, first_output_spans = 0, []

    def failed_encoder(*args, **kwargs):
        nonlocal encoder_calls
        index = encoder_calls
        encoder_calls += 1
        if index == failure_call:
            raise RuntimeError("private failure detail must not be copied into receipts")
        return real_encode(*args, **kwargs)

    def snapshot(path, value):
        if value["calls"][0]["status"] == "succeeded":
            first_output_spans.append(value["load_to_first_output_seconds"])
        return real_snapshot(path, value)

    monkeypatch.setattr(runner, "encode_inputs", failed_encoder)
    monkeypatch.setattr(runner, "_snapshot", snapshot)
    output = case["root"] / "failed-worker"
    record = runner.worker(package["path"], package["pin"], case["batch"], case["batch_pin"],
                           output, job_id="failed-worker", warmup=1, repeats=2)
    assert record["status"] == "failed" and record["failure"] == {"type": "RuntimeError"}
    assert record["active_stage"] == "encoder" and record["active_call"] == failure_call
    failed = record["calls"][failure_call]
    assert failed["status"] == "failed" and failed["outputs"] is None
    assert failed["seconds"]["preprocessing"] >= 0
    assert all(failed["seconds"][key] is None for key in ("encoder", "head_and_policy", "raw_call"))
    assert all(call["status"] == "succeeded" for call in record["calls"][:failure_call])
    assert all(call["status"] == "pending" for call in record["calls"][failure_call + 1:])
    span = record["load_to_first_output_seconds"]
    if failure_call == 0:
        assert span is None and not first_output_spans
    else:
        assert math.isfinite(span) and span >= 0
        assert first_output_spans and all(value == span for value in first_output_spans)
    receipt = (output / "receipt.json").read_bytes()
    assert b"private failure detail" not in receipt
    serialized = json.loads(receipt)
    assert serialized["load_to_first_output_seconds"] == span
    assert serialized == json.loads((output / "progress.json").read_bytes())
    assert runner._validate_record(serialized, "failed-worker", record["binding"],
                                   record["sources"], complete=False) is None
    with pytest.raises(FileExistsError):
        runner.worker(package["path"], package["pin"], case["batch"], case["batch_pin"],
                      output, job_id="failed-worker", warmup=1, repeats=2)
    assert (output / "receipt.json").read_bytes() == receipt


def test_supervisor_accounts_for_every_assignment_and_isolates_bad_receipts(serving_case, monkeypatch):
    case, package = serving_case, serving_case["packages"]["baseline"]
    template_output = case["root"] / "receipt-template"
    runner.worker(package["path"], package["pin"], case["batch"], case["batch_pin"],
                  template_output, job_id="template", warmup=0, repeats=2)
    template = json.loads((template_output / "receipt.json").read_bytes())
    assert template["status"] == "succeeded"
    cases = [("success", "succeeded"), ("nonzero", "nonzero_exit"), ("timeout", "timeout"),
             ("launch", "launch_failed"), ("missing", "missing_receipt"), ("unreadable", "unreadable_receipt"),
             ("json", "malformed_receipt"), ("deep_json", "malformed_receipt"),
             ("schema", "malformed_receipt"), ("binding", "identity_mismatch"),
             ("source", "identity_mismatch"), ("environment", "malformed_receipt"),
             ("files", "malformed_receipt"), ("candidate", "malformed_receipt"),
             ("evidence", "malformed_receipt"), ("request_ids", "malformed_receipt"),
             ("policy", "malformed_receipt"), ("timing", "malformed_receipt"),
             ("null_timing", "malformed_receipt"), ("output", "malformed_receipt"),
             ("missing_first_output", "malformed_receipt"), ("null_first_output", "malformed_receipt"),
             ("non_enclosing_first_output", "malformed_receipt"),
             ("nonzero_success_receipt", "nonzero_exit"),
             ("later_success", "succeeded")]
    output, launches = case["root"] / "supervised", []

    def controlled_child(command, *, cwd, env, timeout, capture_output, check):
        index = len(launches)
        kind = cases[index][0]
        job_id = command[command.index("--job-id") + 1]
        child_output = Path(command[command.index("--output") + 1])
        assigned = json.loads((output / "run.json").read_bytes())["jobs"]
        assert [job["job_id"] for job in assigned] == [f"job-{i:06d}" for i in range(len(cases))]
        assert assigned[index]["status"] == "running"
        assert all(job["status"] == "pending" for job in assigned[index + 1:])
        assert job_id == f"job-{index:06d}" and child_output == output / job_id
        assert cwd == runner.ROOT and capture_output is True and check is False and timeout == .5
        assert all(env[key] == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
        launches.append(kind)
        if kind == "launch":
            raise OSError("controlled launch failure")
        child_output.mkdir()
        record = copy.deepcopy(template)
        record["job_id"] = job_id
        if kind in ("nonzero", "timeout"):
            record.update(status="failed" if kind == "nonzero" else "running", active_stage="encoder", active_call=2,
                          failure={"type": "RuntimeError"} if kind == "nonzero" else None)
            call = record["calls"][2]
            call.update(status="failed" if kind == "nonzero" else "running", outputs=None)
            call["seconds"].update(encoder=None, head_and_policy=None, raw_call=None)
            (child_output / "progress.json").write_bytes(_bytes(record))
            if kind == "timeout":
                raise subprocess.TimeoutExpired(command, timeout, output=b"started", stderr=b"pending")
        elif kind == "missing":
            return subprocess.CompletedProcess(command, 0, b"", b"")
        elif kind == "unreadable":
            (child_output / "receipt.json").mkdir()
            return subprocess.CompletedProcess(command, 0, b"", b"")
        elif kind in ("json", "deep_json"):
            data = b"{" if kind == "json" else b"[" * 5000 + b"0" + b"]" * 5000
            (child_output / "receipt.json").write_bytes(data)
            return subprocess.CompletedProcess(command, 0, b"", b"")
        elif kind == "schema":
            record["schema"] = "unknown"
        elif kind == "binding":
            record["binding"]["batch_sha256"] = "0" * 64
        elif kind == "source":
            record["sources"][next(iter(record["sources"]))] = "0" * 64
        elif kind == "environment":
            record["environment"] = []
        elif kind == "files":
            del record["identities"]["files"]
        elif kind == "candidate":
            del record["identities"]["candidate_sha256"]
        elif kind == "evidence":
            record["identities"]["candidate_identity_evidence"] = "declared_only"
        elif kind == "request_ids":
            record["identities"]["request_ids"] = "abcd"
        elif kind == "policy":
            policy = record["identities"]["policy"]
            policy["effective_arm"] = "context-conditioning"
            record["identities"]["policy_sha256"] = digest(policy)
        elif kind == "timing":
            record["calls"][0]["seconds"]["raw_call"] = 0.
        elif kind == "null_timing":
            record["calls"][0]["seconds"]["encoder"] = None
        elif kind == "missing_first_output":
            del record["load_to_first_output_seconds"]
        elif kind == "null_first_output":
            record["load_to_first_output_seconds"] = None
        elif kind == "non_enclosing_first_output":
            record["load_seconds"]["load"] = 1.
            record["load_to_first_output_seconds"] = .5
        elif kind == "output":
            outputs = record["calls"][0]["outputs"]
            outputs["rows"][0]["queue"] = 5
            outputs["sha256"] = digest({key: value for key, value in outputs.items() if key != "sha256"})
        (child_output / "receipt.json").write_bytes(_bytes(record))
        return subprocess.CompletedProcess(command, 7 if kind in ("nonzero", "nonzero_success_receipt") else 0,
                                           b"child stdout", b"child stderr")

    monkeypatch.setattr(runner.subprocess, "run", controlled_child)
    report = runner.run(package["path"], package["pin"], case["batch"], case["batch_pin"], output,
                        timeout_seconds=.5, jobs=len(cases), warmup=0, repeats=2)
    assert launches == [kind for kind, _ in cases]
    assert [job["status"] for job in report["jobs"]] == [status for _, status in cases]
    assert all(job["process_seconds"] is not None and job["process_seconds"] >= 0 for job in report["jobs"])
    assert report["jobs"][1]["progress_validation"] == report["jobs"][2]["progress_validation"] == "accepted_partial"
    assert report["jobs"][1]["receipt_validation"] == "accepted"
    assert report["jobs"][1]["receipt"]["calls"][1]["status"] == "succeeded"
    assert report["jobs"][1]["receipt"]["load_to_first_output_seconds"] == template["load_to_first_output_seconds"]
    assert all(report["jobs"][index]["progress"]["load_to_first_output_seconds"] == template["load_to_first_output_seconds"]
               for index in (1, 2))
    assert report["jobs"][-2]["returncode"] == 7
    assert report["jobs"][-2]["receipt_validation"] == "accepted"
    assert report["jobs"][-2]["receipt"]["status"] == "succeeded"
    assert (output / "job-000002.stdout.log").read_bytes() == b"started"
    summary = report["summary"]
    assert summary["assigned_jobs"] == len(cases) and summary["successful_jobs"] == 2
    assert summary["unsuccessful_jobs"] == len(cases) - 2
    assert sum(summary["dispositions"].values()) == len(cases)
    assert summary["measured_calls_assigned"] == len(cases) * 2
    assert summary["measured_calls_excluded_from_success_samples"] == (len(cases) - 2) * 2
    assert len(summary["conditional_success_samples"]) == 4
    assert {sample["job_id"] for sample in summary["conditional_success_samples"]} == {
        "job-000000", f"job-{len(cases) - 1:06d}"}
    assert summary["load_to_first_output_samples_assigned"] == len(cases)
    assert summary["load_to_first_output_samples_excluded_from_success_samples"] == len(cases) - 2
    assert summary["conditional_success_load_to_first_output_samples"] == [
        {"job_id": job["job_id"], "seconds": job["receipt"]["load_to_first_output_seconds"]}
        for job in (report["jobs"][0], report["jobs"][-1])]
    assert json.loads((output / "run.json").read_bytes()) == report
    pending = copy.deepcopy(template)
    pending["status"] = "running"
    pending["calls"][-1].update(status="pending", outputs=None, seconds=dict.fromkeys(runner.STAGES))
    assert runner._validate_record(pending, "template", template["binding"], template["sources"], complete=False) is None
    pending["calls"][-1]["seconds"]["preprocessing"] = 1.
    assert runner._validate_record(pending, "template", template["binding"], template["sources"], complete=False) == "malformed_receipt"
    with pytest.raises(FileExistsError):
        runner.run(package["path"], package["pin"], case["batch"], case["batch_pin"], output,
                   timeout_seconds=.5, jobs=len(cases), warmup=0, repeats=2)
    assert len(launches) == len(cases)
