"""Synthetic reservation/count joins; no evaluator or runtime observations."""
from copy import deepcopy
from pathlib import Path

import pytest

from rlcd.weight_space.data_roles import DataRoleError, DataRoleRegistry
from rlcd.weight_space.model import canonical_bytes, digest
from scripts import staged_scientific_count_receipts as receipts


_FLAGS = (
    "provenance_authenticated", "semantic_independence_verified",
    "scientific_freshness_verified", "evaluator_execution_attested",
)


def _prepared(tmp_path, monkeypatch, *, status="succeeded", result=None):
    # These locally authored rows and count/runtime declarations are synthetic.
    slot = [{"sample": f"synthetic-slot-{index}", "label": index % 8,
             "features": [index, index % 19]} for index in range(4096)]
    rows = ([{"surrounding": f"before-{index}"} for index in range(3)]
            + slot + [{"surrounding": f"after-{index}"} for index in range(2)])
    start, stop = 3, 4099
    registry = DataRoleRegistry(tmp_path / "synthetic-registry.sqlite3")
    batch = registry.register_batch(rows, source_id="synthetic-aggregate",
                                    family_id="synthetic-correspondence-family")
    binding = {name: digest({"synthetic_component": name, "revision": "join-test-v1"})
               for name in ("endpoint_sha256", "encoder_sha256", "preprocessing_sha256", "feature_id")}
    binding["rows_sha256"] = digest(slot)
    correct = [470 - 17 * index for index in range(8)]
    declaration = {
        "schema": "wsp-staged-native-slot-counts/1", "binding": binding,
        "execution": {
            "backend": "torch", "dtype": "float32", "device": "cpu",
            "accumulation": "matmul", "endpoint_cast": "round", "error_model": "empirical",
        },
        "chunk_size": 256, "chunks_completed": 16,
        "counts": {
            "correct_by_class": correct, "denominator_by_class": [512] * 8,
            "total_correct": sum(correct), "total_denominator": 4096,
            "reserved_output_predictions": 29,
        },
        "runtime_correspondence": {
            "torch_version": "2.8.0", "torch_git_version": None,
            "torch_cuda_version": None, "torch_hip_version": None,
            "torch_config_sha256": digest({"synthetic_configuration": "not-observed"}),
            "numpy_version": "synthetic-not-an-observed-runtime",
            "threads": 1, "interop_threads": 1,
            "deterministic_algorithms": True, "deterministic_warn_only": False,
            "mkldnn_enabled": False, "matmul_precision": "highest",
            "default_device": "cpu", "default_dtype": "torch.float32",
            "cpu_autocast_enabled": False, "cpu_autocast_dtype": "torch.bfloat16",
        },
    }
    frozen_binding = {
        "protocol": {"name": "synthetic-correspondence", "revision": 1},
        "policy": {"enabled": True, "coefficient": 1.0},
        "selection": ["middle-slice", start, stop],
    }
    path = tmp_path / "synthetic-counts.json"
    captured, callbacks = {}, []
    original_reserve = registry._reserve

    def retain_reservation(connection, attempt):
        original_reserve(connection, attempt)
        captured["expected_attempt"] = {
            key: deepcopy(attempt[key]) for key in (
                "registry_id", "attempt_id", "source_id", "manifest_sha256",
                "content_sha256", "binding", "reservation_sha256",
            )
        }
        raw_slice = rows[start:stop]
        row_hash_slice = attempt["row_sha256"][start:stop]
        assert row_hash_slice == [digest(row) for row in raw_slice]
        assert attempt["content_sha256"] == digest(rows)
        captured["expected_slice"] = {
            "start": start, "stop": stop, "rows_sha256": digest(raw_slice),
            "row_sha256_list_sha256": digest(row_hash_slice),
        }
        captured["slot_row_sha256"] = row_hash_slice.copy()

    callback_failure = RuntimeError("synthetic callback failed after count persistence")
    finalization_failure = RuntimeError("synthetic terminal persistence unavailable")

    def scorer(supplied_rows):
        assert captured["expected_attempt"]["attempt_id"] == "synthetic-once"
        assert supplied_rows == rows
        callbacks.append("entered")
        captured["receipt"] = receipts.write_complete_slot_counts_once(
            path, declaration, expected_binding=binding)
        if status == "failed":
            raise callback_failure
        return deepcopy(result)

    def unavailable_finalization(attempt, decision, result_bytes):
        raise finalization_failure

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_reserve", retain_reservation)
        if status == "authorized_outcome_unknown":
            patch.setattr(registry, "_finalize", unavailable_finalization)
            with pytest.raises(RuntimeError) as caught:
                registry.score(batch, scorer, attempt_id="synthetic-once", binding=frozen_binding)
            assert caught.value is finalization_failure
        elif status == "failed":
            with pytest.raises(DataRoleError) as caught:
                registry.score(batch, scorer, attempt_id="synthetic-once", binding=frozen_binding)
            assert caught.value.decision["status"] == "failed"
            assert caught.value.decision["reason"] == "scorer_failed"
        else:
            returned, decision = registry.score(
                batch, scorer, attempt_id="synthetic-once", binding=frozen_binding)
            assert canonical_bytes(returned) == canonical_bytes(result)
            assert decision["status"] == "succeeded"
    assert callbacks == ["entered"]
    assert path.read_bytes() == canonical_bytes(captured["receipt"]) + b"\n"
    return {
        "registry": registry, "batch": batch, "path": path,
        "receipt": captured["receipt"], "slot_row_sha256": captured["slot_row_sha256"],
        "expectations": {
            "expected_binding": deepcopy(binding),
            "expected_receipt_sha256": captured["receipt"]["sha256"],
            "expected_attempt": captured["expected_attempt"],
            "expected_slice": captured["expected_slice"],
        },
    }


def _read(case, **overrides):
    return receipts.read_count_reservation_correspondence(
        case["path"], registry=case["registry"], **(case["expectations"] | overrides))


@pytest.mark.parametrize(("status", "result"), [
    ("succeeded", None),
    ("succeeded", {"private_scorer_result": [None, True, 7, 1.25, {"note": "arbitrary JSON"}]}),
    ("failed", None),
    ("authorized_outcome_unknown", {"uncommitted_result": ["count file survived"]}),
], ids=["success-null", "success-arbitrary-json", "failed-after-counts", "unfinalized-after-counts"])
def test_correspondence_preserves_lifecycle_and_never_grants_retry(tmp_path, monkeypatch, status, result):
    case = _prepared(tmp_path, monkeypatch, status=status, result=result)
    registry = case["registry"]
    inspect = registry.get_attempt
    inspections, snapshots = [], []

    def inspected(attempt_id):
        inspections.append(attempt_id)
        snapshot = inspect(attempt_id)
        snapshots.append(deepcopy(snapshot))
        return snapshot

    monkeypatch.setattr(registry, "get_attempt", inspected)
    report = _read(case)
    attempt = case["expectations"]["expected_attempt"]
    snapshot, = snapshots
    assert inspections == [attempt["attempt_id"]]
    assert snapshot["status"] == status
    assert snapshot["reason"] == {
        "succeeded": "observed_boundary_passed", "failed": "scorer_failed",
        "authorized_outcome_unknown": "evaluation_authorization_committed",
    }[status]
    if status == "authorized_outcome_unknown":
        assert snapshot["callback_invocations"] is None
    else:
        assert type(snapshot["callback_invocations"]) is int and snapshot["callback_invocations"] == 1
    assert snapshot["result_sha256"] == (digest(result) if status == "succeeded" else None)
    body = {
        "schema": "wsp-staged-count-reservation-correspondence/1",
        "count_receipt": case["receipt"],
        "reservation": {key: attempt[key] for key in (
            "registry_id", "attempt_id", "source_id", "manifest_sha256", "content_sha256",
            "reservation_sha256",
        )} | {"binding_sha256": digest(attempt["binding"])},
        "slice": case["expectations"]["expected_slice"],
        "lifecycle": {key: snapshot[key] for key in (
            "status", "reason", "callback_invocations", "lifecycle_sha256", "result_sha256", *_FLAGS,
        )},
    }
    # Exact shape excludes raw rows, the row-hash vector, and full scorer output.
    assert report == body | {"sha256": digest(body)}
    for flag in _FLAGS:
        assert report["lifecycle"][flag] is False
    retry_callbacks = []
    for attempt_id, reason in (("synthetic-once", "attempt_already_authorized"),
                               ("synthetic-retry", "observed_development_consumption")):
        with pytest.raises(DataRoleError) as rejected:
            registry.score(case["batch"], lambda rows: retry_callbacks.append(rows), attempt_id=attempt_id)
        assert rejected.value.decision["reason"] == reason
    assert retry_callbacks == []
    assert inspections == [attempt["attempt_id"]]


def test_nested_expectations_are_captured_before_receipt_io(tmp_path, monkeypatch):
    case = _prepared(tmp_path, monkeypatch)
    retained = deepcopy(case["expectations"])
    original_lstat = Path.lstat
    inspect = case["registry"].get_attempt
    first_io, inspections = [], []

    def mutate_on_first_io(path, *args, **kwargs):
        if not first_io:
            first_io.append(path)
            current = case["expectations"]
            current["expected_binding"]["encoder_sha256"] = "0" * 64
            current["expected_attempt"]["attempt_id"] = "caller-mutated"
            current["expected_attempt"]["reservation_sha256"] = "0" * 64
            current["expected_attempt"]["binding"]["policy"]["enabled"] = 1
            current["expected_attempt"]["binding"]["policy"]["coefficient"] = 1
            current["expected_slice"].update(start=0, stop=4096, rows_sha256="0" * 64,
                                             row_sha256_list_sha256="0" * 64)
        return original_lstat(path, *args, **kwargs)

    def inspected(attempt_id):
        inspections.append(attempt_id)
        return inspect(attempt_id)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", mutate_on_first_io)
        patch.setattr(case["registry"], "get_attempt", inspected)
        report = _read(case)
    assert first_io and inspections == [retained["expected_attempt"]["attempt_id"]]
    assert report["count_receipt"]["binding"] == retained["expected_binding"]
    assert report["reservation"]["reservation_sha256"] == retained["expected_attempt"]["reservation_sha256"]
    assert report["reservation"]["binding_sha256"] == digest(retained["expected_attempt"]["binding"])
    assert report["slice"] == retained["expected_slice"]
    assert report["sha256"] == digest({key: value for key, value in report.items() if key != "sha256"})


def test_invalid_expectations_and_receipt_failures_never_inspect(tmp_path, monkeypatch):
    case = _prepared(tmp_path, monkeypatch)
    inspections, reads = [], []

    class Dict(dict):
        pass

    class String(str):
        pass

    class Integer(int):
        pass

    def unexpected_inspection(attempt_id):
        inspections.append(attempt_id)
        raise AssertionError("Invalid expectations or unreadable counts must not inspect")

    def unexpected_read(*args, **kwargs):
        reads.append(args)
        raise AssertionError("Invalid expectations must reject before any receipt read")

    monkeypatch.setattr(case["registry"], "get_attempt", unexpected_inspection)
    expected = case["expectations"]
    mutations = [
        (("expected_binding",), Dict(expected["expected_binding"])),
        (("expected_binding", "feature_id"), "A" * 64),
        (("expected_receipt_sha256",), None),
        (("expected_receipt_sha256",), String(expected["expected_receipt_sha256"])),
        (("expected_attempt",), Dict(expected["expected_attempt"])),
        (("expected_attempt", "registry_id"), "another-registry"),
        (("expected_attempt", "attempt_id"), "  "),
        (("expected_attempt", "source_id"), String("synthetic-aggregate")),
        (("expected_attempt", "manifest_sha256"), "F" * 64),
        (("expected_attempt", "content_sha256"), True),
        (("expected_attempt", "reservation_sha256"), "0" * 63),
        (("expected_attempt", "binding"), []),
        (("expected_attempt", "binding", "policy"), Dict(enabled=True)),
        (("expected_attempt", "binding", "policy", "enabled"), Integer(1)),
        (("expected_attempt", "binding", "selection"), ("middle-slice", 3, 4099)),
        (("expected_attempt", "binding", "policy", "coefficient"), float("nan")),
        (("expected_attempt", "binding", "policy", "coefficient"), float("inf")),
        (("expected_slice",), Dict(expected["expected_slice"])),
        (("expected_slice", "start"), True),
        (("expected_slice", "start"), 3.0),
        (("expected_slice", "start"), Integer(3)),
        (("expected_slice", "start"), -1),
        (("expected_slice", "stop"), 4098),
        (("expected_slice", "stop"), 4099.0),
        (("expected_slice", "row_sha256_list_sha256"), String("a" * 64)),
        (("expected_slice", "rows_sha256"), "0" * 64),
    ]
    with monkeypatch.context() as patch:
        patch.setattr(receipts, "read_complete_slot_counts", unexpected_read)
        for keys, value in mutations:
            arguments = deepcopy(expected)
            target = arguments
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with pytest.raises(ValueError):
                _read(case, **arguments)
        for name in ("expected_attempt", "expected_slice"):
            for field in [*expected[name], "extra"]:
                arguments = deepcopy(expected)
                if field == "extra":
                    arguments[name][field] = None
                else:
                    del arguments[name][field]
                with pytest.raises(ValueError):
                    _read(case, **arguments)
    assert reads == [] and inspections == []
    case["path"].write_bytes(b'{"incomplete":true}\n')
    with pytest.raises(ValueError):
        _read(case)
    assert inspections == []


def test_substitutions_missing_and_exceptional_inspections_reject(tmp_path, monkeypatch):
    case = _prepared(tmp_path, monkeypatch)
    inspect = case["registry"].get_attempt
    inspections = []

    def inspected(attempt_id):
        inspections.append(attempt_id)
        return inspect(attempt_id)

    monkeypatch.setattr(case["registry"], "get_attempt", inspected)
    mutations = [
        (("expected_attempt", "attempt_id"), "another-attempt"),
        (("expected_attempt", "source_id"), "another-source"),
        (("expected_attempt", "manifest_sha256"), digest("another-manifest")),
        (("expected_attempt", "content_sha256"), digest("another-content")),
        (("expected_attempt", "reservation_sha256"), digest("another-reservation")),
        (("expected_attempt", "binding", "protocol", "revision"), True),
        (("expected_attempt", "binding", "policy", "enabled"), 1),
        (("expected_attempt", "binding", "policy", "coefficient"), 1),
        (("expected_slice", "row_sha256_list_sha256"), digest(case["slot_row_sha256"][::-1])),
    ]
    for keys, value in mutations:
        arguments = deepcopy(case["expectations"])
        target = arguments
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        inspections.clear()
        with pytest.raises(ValueError):
            _read(case, **arguments)
        assert inspections == [arguments["expected_attempt"]["attempt_id"]]
    shifted = deepcopy(case["expectations"]["expected_slice"])
    shifted.update(start=4, stop=4100)
    inspections.clear()
    with pytest.raises(ValueError):
        _read(case, expected_slice=shifted)
    assert inspections == ["synthetic-once"]

    def missing(attempt_id):
        inspections.append(attempt_id)
        return None

    monkeypatch.setattr(case["registry"], "get_attempt", missing)
    inspections.clear()
    with pytest.raises(ValueError):
        _read(case)
    assert inspections == ["synthetic-once"]
    for failure in (ValueError("synthetic invalid inspection"), OSError("synthetic unavailable inspection")):
        def unavailable(attempt_id):
            inspections.append(attempt_id)
            raise failure

        monkeypatch.setattr(case["registry"], "get_attempt", unavailable)
        inspections.clear()
        with pytest.raises(type(failure)) as caught:
            _read(case)
        assert caught.value is failure
        assert inspections == ["synthetic-once"]


def test_inspected_lifecycle_requires_plain_terminal_counts_and_false_flags(tmp_path, monkeypatch):
    case = _prepared(tmp_path, monkeypatch)
    original = case["registry"].get_attempt("synthetic-once")
    inspections = []

    class Integer(int):
        pass

    mutations = [("callback_invocations", True), ("callback_invocations", 1.0),
                 ("callback_invocations", Integer(1)), *[(flag, 0) for flag in _FLAGS]]
    for key, value in mutations:
        snapshot = deepcopy(original)
        snapshot[key] = value
        snapshot["lifecycle_sha256"] = digest({
            name: item for name, item in snapshot.items() if name != "lifecycle_sha256"
        })

        def inspected(attempt_id):
            inspections.append(attempt_id)
            return deepcopy(snapshot)

        monkeypatch.setattr(case["registry"], "get_attempt", inspected)
        inspections.clear()
        with pytest.raises(ValueError):
            _read(case)
        assert inspections == ["synthetic-once"]
