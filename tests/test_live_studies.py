"""Synthetic trust-boundary and recovery checks; no acquired data or live calls."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_live_studies as runner


class SimulatedCrash(BaseException):
    """Leave only state made durable before the injected interruption."""


def _resign(value):
    return runner.signed({key: item for key, item in value.items() if key != "sha256"})


@contextmanager
def _changed(path, value):
    """Restore a shared synthetic fixture even when a rejection assertion fails."""
    original = path.read_bytes() if path.exists() else None
    if value is None:
        path.unlink()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    try:
        yield
    finally:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)


def _encoded(value):
    return runner.canonical_bytes(value) + b"\n"


def _synthetic_model():
    preprocessing = {"schema": "synthetic-preprocessing/1", "features": ["left", "right"]}
    encoder = {"weight": np.eye(2, dtype=np.float32)}
    weight = np.zeros((78, 2), dtype=np.float32)
    weight[0, 0] = weight[1, 1] = 1.
    bias = np.zeros(78, dtype=np.float32)
    bias[2], bias[77] = -83.125, -.5
    parent = runner.TensorArtifact(
        runner.ARCHITECTURE, {runner.HEAD_WEIGHT: weight, runner.HEAD_BIAS: bias},
        {"feature_id": "synthetic-two-features",
         "feature_names": ["left", "right"],
         "output_names": [*runner.TARGETS, *[f"synthetic-intent-{i}" for i in range(2, 77)], "specialist"],
         "encoder_sha256": runner.digest({key: value.tolist() for key, value in encoder.items()}),
         "preprocessing_sha256": runner.digest(preprocessing),
         "private_note": "SYNTHETIC-EVALUATOR-SECRET"},
    )
    queue_map = np.arange(78, dtype=np.int64) % 5
    queue_map[77] = 5
    bundle = {"queue_map": queue_map}
    for role in ("target", "protected", "calibration"):
        bundle.update({
            role + "_features": np.array([[1., 0.], [.75, .25], [0., 1.], [.25, .75]], dtype=np.float32),
            role + "_labels": np.array([0, 0, 1, 1], dtype=np.int64),
            role + "_ids": np.array([f"synthetic/{role}/{i}" for i in range(4)]),
        })
    return parent, bundle, encoder, preprocessing


def _write_training(root, lineage):
    parent, bundle, encoder, preprocessing = _synthetic_model()
    directory = root / "artifacts/research/trained-support/main" / lineage
    directory.mkdir(parents=True)
    runner.write_json(directory / "head.json", parent.to_dict())
    runner.write_json(directory / "preprocessing.json", preprocessing)
    np.savez_compressed(directory / "encoder.npz", **encoder)
    np.savez_compressed(directory / "features.npz", **bundle)
    runner.write_json(directory / "lineage.json", {
        "lineage_id": lineage, "head_sha256": parent.sha256,
        "feature_id": parent.metadata["feature_id"],
        "encoder_sha256": parent.metadata["encoder_sha256"],
        "preprocessing_sha256": parent.metadata["preprocessing_sha256"],
        "files": {name: {"sha256": runner.sha(directory / name), "bytes": (directory / name).stat().st_size}
                  for name in runner.TRAINING_FILES if name != "lineage.json"},
    })


def _evaluation_bundle():
    return {
        "test_features": np.array([[1., 0.], [.75, .25], [0., 1.], [.25, .75]], dtype=np.float32),
        "test_labels": np.array([0, 0, 1, 1], dtype=np.int64),
        "test_ids": np.array([f"synthetic/test/{i}" for i in range(4)]),
        "test_group_ids": np.array(["target-a", "target-b", "protected-a", "protected-b"]),
        "test_novel_mask": np.ones(4, dtype=bool),
    }


def _write_evaluation_bundle(directory):
    np.savez_compressed(directory / "evaluation-features.npz", **_evaluation_bundle())
    path = directory / "evaluation-features.npz"
    runner.write_json(directory / "evaluation-features.json", {
        "sha256": runner.sha(path), "bytes": path.stat().st_size,
        "feature_id": "synthetic-two-features",
    })


def _npz_bytes(bundle):
    stream = io.BytesIO()
    np.savez_compressed(stream, **bundle)
    return stream.getvalue()


def _missing_outcome(assignment, selected, gate, error_type="SyntheticEvaluation"):
    deterministic = {
        "assignment_id": assignment["assignment_id"], "selection_sha256": selected["sha256"],
        "global_seal_sha256": gate["sha256"], "evaluation_missing": True,
        "joint_useful_success": False, "error_type": error_type,
    }
    return runner.signed({
        "schema": "wsp-live-outcome/1", **{key: assignment[key] for key in runner.assigned_specs()[0]},
        "assigned": True, "terminal_status": selected["terminal_status"], "no_op": selected["no_op"],
        "deterministic": deterministic, "deterministic_sha256": runner.digest(deterministic),
        "selection_sha256": selected["sha256"], "global_seal_sha256": gate["sha256"],
        "missingness_sensitivity": {"gain": [-1., 1.], "harm": [-1., 1.], "useful": [0., 1.]},
    })


def _synthetic_evaluate(root, assignment, selected, gate, lineage, primary, destination):
    """Keep the complete evaluation boundary, replacing only per-assignment arithmetic."""
    destination.mkdir(parents=True, exist_ok=False)
    outcome = _missing_outcome(assignment, selected, gate)
    runner.write_json(destination / "outcome.json", outcome)
    return outcome


@pytest.fixture(autouse=True)
def forbid_live_calls(monkeypatch, tmp_path_factory):
    def forbidden(*args, **kwargs):
        raise AssertionError("A synthetic test attempted a real provider call")

    original_load = runner.load_lineage
    temporary_root = tmp_path_factory.getbasetemp().resolve()

    def synthetic_only(path):
        assert Path(path).resolve().is_relative_to(temporary_root), "Attempt to load published real data"
        return original_load(path)

    monkeypatch.setattr(runner.CodexProvider, "call", forbidden)
    monkeypatch.setattr(runner, "load_lineage", synthetic_only)


@pytest.fixture(scope="module")
def prepared_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-live-prepared")
    protocols = {}
    for name in runner.PROTOCOL_HASHES:
        path = root / runner.PROTOCOL_DIR / name
        runner.write_json(path, {"schema": "synthetic-protocol/1", "name": name})
        protocols[name] = runner.sha(path)
    runner.write_json(root / runner.PROTOCOL_DIR / "label-amendment.json", {"scope": "synthetic"})
    for lineage in runner.LINEAGES:
        _write_training(root, lineage)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "PROTOCOL_HASHES", protocols)
        lock = runner.prepare(root)
        assert runner.verify_preparation(root) == lock
        yield SimpleNamespace(root=root, lock=lock)


def _checkpoint(root, assignment, lock):
    locked = next(item for item in lock["assignments"] if item["assignment_id"] == assignment["assignment_id"])
    checkpoint = runner.signed({
        "schema": "wsp-live-invocation-checkpoint/1",
        "assignment_id": assignment["assignment_id"], "assignment_sha256": locked["assignment_sha256"],
        "prepared_lock_sha256": lock["sha256"], "prompt_sha256": assignment["prompt_sha256"],
        "schema_sha256": runner.digest(runner.read(root / assignment["schema_path"])),
        "provider_configuration": lock["provider"], "phase": "call_may_start",
    })
    path = runner.selection_path(root, assignment["assignment_id"]).parent / "invocation-checkpoint.json"
    runner.write_json(path, checkpoint)
    return checkpoint


@pytest.fixture(scope="module")
def sealed_template(prepared_template):
    root, lock = prepared_template.root, prepared_template.lock
    for locked in lock["assignments"]:
        assignment = runner.read(root / locked["assignment_path"])
        checkpoint = _checkpoint(root, assignment, lock)
        runner.finish_selection(root, assignment, None, "interrupted", checkpoint)
    gate = runner.seal(root, lock)
    return SimpleNamespace(root=root, lock=lock, gate=gate)


def _copy_evaluation_study(study, root):
    shutil.copytree(study.root, root)
    for lineage in runner.LINEAGES:
        _write_evaluation_bundle(root / "artifacts/research/trained-support/main" / lineage)
    return SimpleNamespace(root=root, lock=study.lock, gate=study.gate)


@pytest.fixture(scope="module")
def evaluation_template(sealed_template, tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-live-evaluator") / "study"
    return _copy_evaluation_study(sealed_template, root)


@pytest.fixture(scope="module")
def completed_evaluation(evaluation_template):
    study = evaluation_template
    calls = []

    def evaluate(*args):
        calls.append(args[1]["assignment_id"])
        return _synthetic_evaluate(*args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "evaluate_assignment", evaluate)
        manifest = runner.run_evaluation(study.root, study.lock, study.gate)
    assert calls == [item["assignment_id"] for item in study.lock["assignments"]]
    return SimpleNamespace(root=study.root, lock=study.lock, gate=study.gate, manifest=manifest)


def _case(root, study="weight-access", view="none", regime="primary", mechanism="bias_shift"):
    root.mkdir()
    clean, bundle, _, _ = _synthetic_model()
    spec = runner.specification(study, runner.LINEAGES[0], runner.TARGETS[0], view, regime,
                                mechanism=mechanism if study == "blind-diagnosis" else None)
    parent, ledger = clean, None
    if study == "blind-diagnosis":
        parent, ledger = runner.plant(clean, spec, bundle["target_features"][:2], 0, bundle["queue_map"])
    directory = root / runner.ARTIFACT_DIR / "prepared/assignments" / spec["assignment_id"]
    directory.mkdir(parents=True)
    assignment = spec | {
        "objective": "Route synthetic intent zero while preserving intent one.",
        "context": {"feature_names": ["left", "right"],
                    "target_role_examples": runner.example_rows(bundle, "target", [0, 1]),
                    "protected_role_examples": runner.example_rows(bundle, "protected", [2, 3])},
        "target_label": 0, "desired_code": 77 if study == "weight-access" else 0,
        "desired_queue": 5 if study == "weight-access" else 0,
        "target_role_indices": [0, 1], "protected_role_indices": [2, 3],
        "parent_path": (directory / "parent.json").relative_to(root).as_posix(),
    }
    runner.write_json(directory / "parent.json", parent.to_dict())
    if ledger is not None:
        runner.write_json(directory / "evaluator-only-injection.json", runner.signed(ledger))
    broker = runner.make_broker(assignment, parent)
    public = runner.weight_prelude(broker, parent, view)
    supplemental, offers = None, []
    if regime != "primary":
        supplemental, offers, _ = runner.supplement(
            broker, parent, assignment, bundle, np.array([0, 1]), np.array([2, 3]))
    prompt = directory / "prompt.txt"
    prompt.write_text(runner.literal_prompt(assignment, public, supplemental), encoding="utf-8", newline="\n")
    schema = directory / "response-schema.json"
    runner.write_json(schema, runner.response_schema(study))
    assignment.update(
        fitted_offers=offers, preparation_broker_trace=broker.public_trace, parameter_values_exposed=0,
        prompt_path=prompt.relative_to(root).as_posix(), prompt_sha256=runner.sha(prompt),
        prompt_bytes=prompt.stat().st_size, schema_path=schema.relative_to(root).as_posix())
    assignment = runner.signed(assignment)
    assignment_path = directory / "assignment.json"
    runner.write_json(assignment_path, assignment)
    locked = spec | {"assignment_path": assignment_path.relative_to(root).as_posix(),
                     "assignment_sha256": runner.sha(assignment_path),
                     "prompt_path": assignment["prompt_path"], "prompt_sha256": assignment["prompt_sha256"],
                     "prompt_bytes": assignment["prompt_bytes"]}
    lock = runner.signed({"assignments": [locked], "provider": runner.CodexProvider().configuration()})
    runner.write_json(root / runner.LOCK_PATH, lock)
    return SimpleNamespace(root=root, raw=root.parent / (root.name + "-private"), assignment=assignment,
                           locked=locked, lock=lock, parent=parent, ledger=ledger,
                           lineage={"base": clean, "features": bundle | _evaluation_bundle()})


@pytest.fixture
def case_factory(tmp_path):
    cases = []

    def create(**kwargs):
        case = _case(tmp_path / f"case-{len(cases)}", **kwargs)
        cases.append(case)
        return case

    return create


def _response(case, commands=None):
    response = {
        "abstain": False,
        "commands_json": json.dumps(commands if commands is not None else [
            {"op": "add", "tensor": runner.HEAD_BIAS, "block": [[77, 78]], "values": [2.]}]),
        "selected_fitted_handle": None, "reason": "Synthetic committed response",
    }
    if case.assignment["study"] == "weight-access":
        response.update(predicted_argmax_codes=[0, 0], prediction_confidences=[1., 1.],
                        strongest_competing_codes=[1, 1], competing_confidences=[1., 1.])
    else:
        response.update(location_tensor=case.ledger["tensor"], output_row=case.ledger["output_row"],
                        mechanism_class=case.ledger["mechanism"], confidence=1.,
                        predicted_target_gain=.5, predicted_protected_harm=0.)
    return response


def _write_attempt(case, directory, response, status="completed"):
    directory.mkdir(parents=True, exist_ok=True)
    prompt = (case.root / case.assignment["prompt_path"]).read_text(encoding="utf-8")
    schema = runner.read(case.root / case.assignment["schema_path"])
    runner.write_json(directory / "request.json", {"prompt": prompt, "configuration": case.lock["provider"]})
    runner.write_json(directory / "response-schema.json", schema)
    events = _encoded({"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(response)}})
    events += _encoded({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}})
    (directory / "events.jsonl").write_bytes(events)
    receipt = runner.signed({
        "schema": "wsp-live-provider-attempt/1", "status": status,
        "error": None if status == "completed" else "synthetic_failure",
        "returncode": 0 if status == "completed" else 1, "elapsed_seconds": .01,
        "invocation": {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                       "schema_sha256": runner.digest(schema), "configuration": case.lock["provider"]},
        "response": response, "usage": {"input_tokens": 10, "output_tokens": 4},
        "unapproved_events": [], "event_errors": [], "events_sha256": hashlib.sha256(events).hexdigest(),
    })
    runner.write_json(directory / "receipt.json", receipt)
    return receipt


def _select(case, monkeypatch, response):
    calls = []

    def call(provider, prompt, *, directory, cwd, schema):
        checkpoint = runner.read(runner.selection_path(case.root, case.assignment["assignment_id"]).parent /
                                 "invocation-checkpoint.json")
        runner.check_checkpoint(case.root, case.assignment, case.lock, checkpoint)
        assert prompt == (case.root / case.assignment["prompt_path"]).read_text(encoding="utf-8")
        assert schema == runner.read(case.root / case.assignment["schema_path"])
        calls.append(directory)
        return _write_attempt(case, directory, response)

    monkeypatch.setattr(runner.CodexProvider, "call", call)
    selected = runner.select_one(case.root, case.raw, case.locked, case.lock)
    assert len(calls) == 1
    return selected


@pytest.mark.parametrize("kind", ["protocol", "lock", "source", "prompt", "schema", "assignment", "training"])
@pytest.mark.parametrize("missing", [False, True], ids=["mutated", "missing"])
def test_preparation_rejects_changed_or_missing_frozen_inputs(prepared_template, kind, missing):
    root, lock = prepared_template.root, prepared_template.lock
    assignment = runner.read(root / lock["assignments"][0]["assignment_path"])
    paths = {
        "protocol": root / runner.PROTOCOL_DIR / next(iter(runner.PROTOCOL_HASHES)),
        "lock": root / runner.LOCK_PATH,
        "source": root / lock["source_files"][0]["path"],
        "prompt": root / assignment["prompt_path"],
        "schema": root / assignment["schema_path"],
        "assignment": root / lock["assignments"][0]["assignment_path"],
        "training": root / assignment["training_path"] / "features.npz",
    }
    path = paths[kind]
    changed = None if missing else path.read_bytes() + b"\nchanged"
    with _changed(path, changed), pytest.raises((ValueError, FileNotFoundError)):
        runner.verify_preparation(root)


@pytest.mark.parametrize("mutation", [
    "workers", "provider", "runtime", "selection_phase", "missing_assignment", "duplicate_assignment",
    "reordered_assignments", "amendment",
])
def test_resigned_lock_cannot_change_the_frozen_contract(prepared_template, mutation):
    lock = deepcopy(prepared_template.lock)
    if mutation == "workers":
        lock["workers"] += 1
    elif mutation == "provider":
        lock["provider"]["requested_model"] = "unassigned-provider"
    elif mutation == "runtime":
        lock["runtime"]["numpy"] = "different-runtime"
    elif mutation == "selection_phase":
        lock["evaluation_started"] = True
    elif mutation == "missing_assignment":
        lock["assignments"].pop()
    elif mutation == "duplicate_assignment":
        lock["assignments"][-1] = lock["assignments"][0]
    elif mutation == "reordered_assignments":
        lock["assignments"].reverse()
    else:
        lock["amendment_sha256"]["implementation-amendment.json"] = "0" * 64
    with _changed(prepared_template.root / runner.LOCK_PATH, _encoded(_resign(lock))):
        with pytest.raises(ValueError):
            runner.verify_preparation(prepared_template.root)


@pytest.mark.parametrize("directory", ["prepared", "source-snapshot"])
def test_preparation_inventory_rejects_unlocked_files(prepared_template, directory):
    root = prepared_template.root
    path = root / (runner.ARTIFACT_DIR / directory if directory == "prepared" else runner.PROTOCOL_DIR / directory)
    with _changed(path / "unlocked.txt", b"unlocked"), pytest.raises(ValueError, match="inventory"):
        runner.verify_preparation(root)


@pytest.mark.parametrize("mutation", ["script", "unlocked_import"])
def test_runtime_snapshot_rejects_changed_or_unlocked_code(tmp_path, monkeypatch, mutation):
    source = tmp_path / "source"
    script = source / "scripts/run_live_studies.py"
    script.parent.mkdir(parents=True)
    script.write_text("# synthetic executable\n")
    lock = {"source_files": [{"source_path": "scripts/run_live_studies.py", "sha256": runner.sha(script)}]}
    monkeypatch.setattr(runner, "SOURCE_ROOT", source)
    monkeypatch.setattr(runner, "__file__", str(script))
    runner.require_snapshot(tmp_path, lock)
    if mutation == "script":
        script.write_text("# changed executable\n")
    else:
        helper = source / "unlocked.py"
        helper.write_text("# unrecorded import\n")
        monkeypatch.setitem(sys.modules, "_synthetic_live_unlocked", SimpleNamespace(__file__=str(helper)))
    with pytest.raises(ValueError):
        runner.require_snapshot(tmp_path, lock)


def test_runtime_snapshot_ignores_a_dynamic_module_filename_without_a_source_file(tmp_path, monkeypatch):
    script = tmp_path / "scripts/run_live_studies.py"
    script.parent.mkdir()
    script.write_text("# synthetic executable\n")
    lock = {"source_files": [{"source_path": "scripts/run_live_studies.py", "sha256": runner.sha(script)}]}
    monkeypatch.setattr(runner, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(runner, "__file__", str(script))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "_synthetic_dynamic_module", SimpleNamespace(__file__="_classes.py"))
    runner.require_snapshot(tmp_path, lock)


def test_evidence_paths_and_duplicate_file_records_are_rejected(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    outside.write_bytes(b"synthetic")
    try:
        with pytest.raises(ValueError, match="escapes"):
            runner.checked_path(tmp_path, "../" + outside.name)
        inside = tmp_path / "evidence.txt"
        inside.write_bytes(b"synthetic")
        record = runner.file_record(tmp_path, inside)
        with pytest.raises(ValueError, match="Duplicate"):
            runner.check_files(tmp_path, [record, record])
    finally:
        outside.unlink()


def test_seal_accounts_for_every_assignment_before_evaluation(sealed_template):
    study = sealed_template
    gate = runner.validate_gate(study.root, study.gate["sha256"])
    assert gate["assigned_total"] == gate["terminal_total"] == len(gate["accounting"]) == 224
    assert {row["assignment_id"] for row in gate["accounting"]} == {
        row["assignment_id"] for row in study.lock["assignments"]}
    assert {row["terminal_status"] for row in gate["accounting"]} == {"interrupted"}
    assert not (study.root / runner.ARTIFACT_DIR / "evaluation").exists()


@pytest.mark.parametrize("mutation", [
    "phase", "accounting_attestation", "missing_account", "duplicate_account",
    "wrong_status", "wrong_selection", "wrong_lock", "wrong_protocol",
])
def test_resigned_gate_cannot_replace_accounting_or_selection_evidence(sealed_template, mutation):
    study = sealed_template
    gate = deepcopy(study.gate)
    if mutation == "phase":
        gate["evaluation_started"] = True
    elif mutation == "accounting_attestation":
        gate["all_assigned_accounted"] = False
    elif mutation == "missing_account":
        gate["accounting"].pop()
    elif mutation == "duplicate_account":
        gate["accounting"][-1] = gate["accounting"][0]
    elif mutation == "wrong_status":
        gate["accounting"][0]["terminal_status"] = "sealed"
    elif mutation == "wrong_selection":
        gate["accounting"][0]["selection_sha256"] = "0" * 64
    elif mutation == "wrong_lock":
        gate["prepared_lock_sha256"] = "0" * 64
    else:
        gate["protocol_sha256"] = {}
    gate = _resign(gate)
    with _changed(study.root / runner.GATE_PATH, _encoded(gate)), pytest.raises(ValueError):
        runner.validate_gate(study.root, gate["sha256"])


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "removed_from_inventory"])
def test_selection_evidence_is_complete_and_immutable(sealed_template, mutation):
    study = sealed_template
    assignment = runner.read(study.root / study.lock["assignments"][0]["assignment_path"])
    path = runner.selection_path(study.root, assignment["assignment_id"])
    selected = runner.read(path)
    evidence = path.parent / "invocation-checkpoint.json"
    if mutation == "removed_from_inventory":
        selected["evidence_files"] = []
        changed_path, changed = path, _encoded(_resign(selected))
    elif mutation == "extra":
        changed_path, changed = path.parent / "unlocked.txt", b"not locked"
    else:
        changed_path, changed = evidence, None if mutation == "missing" else evidence.read_bytes() + b"\n"
    with _changed(changed_path, changed), pytest.raises((ValueError, FileNotFoundError)):
        runner.validate_selection(study.root, assignment, study.lock)


@pytest.mark.parametrize("field,value", [
    ("terminal_status", "sealed"), ("answer_valid", True), ("no_op", True),
    ("candidate_sha256", "0" * 64), ("provider_status", "completed"),
])
def test_resigned_selection_must_replay_its_actual_decision(sealed_template, field, value):
    study = sealed_template
    assignment = runner.read(study.root / study.lock["assignments"][0]["assignment_path"])
    selected = runner.read(runner.selection_path(study.root, assignment["assignment_id"]))
    selected[field] = value
    with pytest.raises(ValueError):
        runner.validate_selection(study.root, assignment, study.lock, _resign(selected))


@pytest.mark.parametrize("field", [
    "assignment_id", "assignment_sha256", "prepared_lock_sha256", "prompt_sha256",
    "schema_sha256", "provider_configuration", "phase",
])
def test_invocation_checkpoint_is_bound_to_assignment_and_lock(case_factory, field):
    case = case_factory()
    checkpoint = _checkpoint(case.root, case.assignment, case.lock)
    checkpoint[field] = {} if field == "provider_configuration" else "different"
    with pytest.raises(ValueError, match="checkpoint"):
        runner.check_checkpoint(case.root, case.assignment, case.lock, _resign(checkpoint))


@pytest.mark.parametrize("mutation", [
    "prompt", "schema", "configuration", "status", "error", "returncode", "tools", "events", "usage",
])
def test_receipt_requires_exact_inputs_and_successful_restricted_transport(case_factory, mutation):
    case = case_factory()
    checkpoint = _checkpoint(case.root, case.assignment, case.lock)
    receipt = _write_attempt(case, case.raw / "provider", _response(case))
    assert runner.check_receipt(case.root, case.assignment, case.lock, checkpoint, receipt) == receipt
    if mutation in {"prompt", "schema"}:
        receipt["invocation"][mutation + "_sha256"] = "0" * 64
    elif mutation == "configuration":
        receipt["invocation"]["configuration"] = {}
    else:
        field, value = {
            "status": ("status", "unknown"), "error": ("error", "transport_failed"),
            "returncode": ("returncode", 1), "tools": ("unapproved_events", ["command_execution"]),
            "events": ("event_errors", ["turn.failed"]), "usage": ("usage", {}),
        }[mutation]
        receipt[field] = value
    with pytest.raises(ValueError):
        runner.check_receipt(case.root, case.assignment, case.lock, checkpoint, _resign(receipt))


def test_completed_receipt_is_recovered_without_another_provider_call(case_factory, monkeypatch):
    case = case_factory()
    calls = []

    def call(provider, prompt, *, directory, cwd, schema):
        calls.append(directory)
        _write_attempt(case, directory, _response(case))
        raise SimulatedCrash()

    monkeypatch.setattr(runner.CodexProvider, "call", call)
    with pytest.raises(SimulatedCrash):
        runner.select_one(case.root, case.raw, case.locked, case.lock)
    selected = runner.select_one(case.root, case.raw, case.locked, case.lock)
    assert selected["terminal_status"] == "sealed" and not selected["no_op"]
    assert runner.select_one(case.root, case.raw, case.locked, case.lock) == selected
    assert len(calls) == 1


@pytest.mark.parametrize("crash", [SimulatedCrash, OSError])
def test_uncertain_provider_call_is_accounted_without_retry(case_factory, monkeypatch, crash):
    case = case_factory()
    calls = []

    def call(provider, prompt, *, directory, cwd, schema):
        calls.append(directory)
        directory.mkdir(parents=True)
        (directory / "events.jsonl").write_text('{"type":"turn.started"}\n')
        raise crash("synthetic interrupted transport")

    monkeypatch.setattr(runner.CodexProvider, "call", call)
    if crash is SimulatedCrash:
        with pytest.raises(SimulatedCrash):
            runner.select_one(case.root, case.raw, case.locked, case.lock)
    else:
        runner.select_one(case.root, case.raw, case.locked, case.lock)
    selected = runner.select_one(case.root, case.raw, case.locked, case.lock)
    assert selected["terminal_status"] == selected["provider_status"] == "interrupted"
    assert selected["provider_receipt_sha256"] is None and selected["response"] is None
    assert selected["candidate_sha256"] == case.parent.sha256
    assert selected["interrupted_transport_termination_attested"] is False
    assert len(calls) == 1


def test_partial_provider_publication_is_repaired_from_the_existing_receipt(case_factory, monkeypatch):
    case = case_factory()
    public = runner.selection_path(case.root, case.assignment["assignment_id"]).parent / "provider"
    calls = []

    def call(provider, prompt, *, directory, cwd, schema):
        calls.append(directory)
        return _write_attempt(case, directory, _response(case))

    original_replace = runner.os.replace

    def interrupted_replace(source, destination):
        if Path(destination) == public / "events.jsonl":
            raise SimulatedCrash()
        return original_replace(source, destination)

    monkeypatch.setattr(runner.CodexProvider, "call", call)
    with monkeypatch.context() as patch:
        patch.setattr(runner.os, "replace", interrupted_replace)
        with pytest.raises(SimulatedCrash):
            runner.select_one(case.root, case.raw, case.locked, case.lock)
    assert (public / "receipt.json").exists() and not (public / "events.jsonl").exists()
    selected = runner.select_one(case.root, case.raw, case.locked, case.lock)
    assert selected["terminal_status"] == "sealed"
    assert len(calls) == 1
    runner.validate_selection(case.root, case.assignment, case.lock)
    assert {path.name for path in public.iterdir()} == {
        "request.json", "response-schema.json", "receipt.json", "events.jsonl", "event-publication.json"}


@pytest.mark.parametrize("mutation", ["request", "schema", "events", "raw_identity", "public_identity", "raw_bytes"])
def test_public_provider_derivative_stays_bound_to_the_raw_receipt(case_factory, monkeypatch, mutation):
    case = case_factory()
    _select(case, monkeypatch, _response(case))
    public = runner.selection_path(case.root, case.assignment["assignment_id"]).parent / "provider"
    checkpoint = runner.read(public.parent / "invocation-checkpoint.json")
    if mutation == "events":
        path, value = public / "events.jsonl", b"changed\n"
    elif mutation in {"request", "schema"}:
        path = public / ("request.json" if mutation == "request" else "response-schema.json")
        value = _encoded({})
    else:
        path = public / "event-publication.json"
        derivative = runner.read(path)
        field = {"raw_identity": "raw_events_sha256", "public_identity": "public_events_sha256",
                 "raw_bytes": "raw_bytes_published"}[mutation]
        derivative[field] = True if mutation == "raw_bytes" else "0" * 64
        value = _encoded(derivative)
    with _changed(path, value), pytest.raises(ValueError):
        runner.check_published_provider(case.root, case.assignment, case.lock, checkpoint, public)


@pytest.mark.parametrize("study", ["weight-access", "blind-diagnosis"])
@pytest.mark.parametrize("invalid", ["malformed_commands", "unsupported_operation"])
def test_valid_reading_or_diagnosis_is_scored_even_when_the_edit_is_invalid(
        case_factory, monkeypatch, tmp_path, study, invalid):
    case = case_factory(study=study)
    response = _response(case, [{"op": "not-an-edit"}])
    if invalid == "malformed_commands":
        response["commands_json"] = "{"
    selected = _select(case, monkeypatch, response)
    assert selected["terminal_status"] == "invalid" and selected["answer_valid"]
    assert selected["candidate_sha256"] == case.parent.sha256 and selected["program_sha256"] is None
    result = runner.evaluate_assignment(case.root, case.assignment, selected, {"sha256": "synthetic-seal"},
                                        case.lineage, np.arange(4), tmp_path / "evaluation")
    rubric = result["deterministic"]["rubric"]
    if study == "weight-access":
        assert rubric["read"]["all_assigned_accuracy"] == rubric["read"]["prediction_coverage"] == 1.
    else:
        assert rubric["joint_diagnosis_correct"] and rubric["diagnosis_coverage"]
        assert rubric["conditional_binary_brier"] == 0.
    assert not result["deterministic"]["joint_useful_success"]


def test_none_view_and_analysis_prelude_do_not_disclose_evaluator_state(case_factory):
    case = case_factory(study="blind-diagnosis", regime="brokered_analysis_only")
    prompt = (case.root / case.assignment["prompt_path"]).read_text(encoding="utf-8")
    for private in (case.parent.sha256, case.lineage["base"].sha256, "SYNTHETIC-EVALUATOR-SECRET",
                    "-83.125", "clean_parent_sha256", "planted_parent_sha256", "evaluator-only-injection"):
        assert private not in prompt
    assert "target_feature_norms" in prompt
    assert "exposed_tensor_norms" not in prompt
    assert '"parameters":[]' in prompt


def test_fitted_handles_replay_without_exposing_candidate_bytes(case_factory, monkeypatch):
    case = case_factory(regime="brokered_fitting_permitted")
    prompt = (case.root / case.assignment["prompt_path"]).read_text(encoding="utf-8")
    supplementary = json.loads(prompt.split("Supplementary permitted training prelude:\n", 1)[1])
    assert len(supplementary["candidates"]) == len(runner.FIT_GRID)
    assert all(set(item) == {"candidate_handle", "configuration"} for item in supplementary["candidates"])
    assert "values" not in supplementary and "endpoint_scores" not in supplementary
    response = _response(case, [])
    response["selected_fitted_handle"] = case.assignment["fitted_offers"][0]["handle"]
    selected = _select(case, monkeypatch, response)
    assert selected["terminal_status"] == "sealed"
    runner.validate_selection(case.root, case.assignment, case.lock)
    parent, candidate, broker = runner.materialize_selection(case.root, case.assignment, selected)
    request = case.assignment["fitted_offers"][0]["request"]
    lowered = runner.lower_structured(parent, runner.StructuredProgram(parent.sha256, request["commands"], request["intent"]),
                                      native_dtypes=runner.DTYPES, cast="round")
    assert candidate.sha256 == runner.apply_program(parent, lowered.program).sha256
    assert runner.trace_semantics(broker.public_trace) == runner.trace_semantics(selected["broker_trace"])
    assert candidate.sha256 not in prompt


def test_literal_noop_is_preserved_and_cannot_be_a_useful_success(case_factory, monkeypatch, tmp_path):
    case = case_factory()
    selected = _select(case, monkeypatch, _response(case, []))
    assert selected["terminal_status"] == "sealed" and selected["no_op"]
    parent, candidate, _ = runner.materialize_selection(case.root, case.assignment, selected)
    assert candidate.sha256 == parent.sha256
    result = runner.evaluate_assignment(case.root, case.assignment, selected, {"sha256": "synthetic-seal"},
                                        case.lineage, np.arange(4), tmp_path / "evaluation")
    assert not result["deterministic"]["joint_useful_success"]
    missing = result | {"deterministic": result["deterministic"] | {"evaluation_missing": True}}
    assert runner.useful_value(missing) == 0.


@pytest.mark.parametrize("mechanism", ["bias_shift", "feature_row_shift"])
def test_planted_inverse_restores_the_exact_parent(case_factory, mechanism):
    case = case_factory(study="blind-diagnosis", mechanism=mechanism)
    assert case.parent.sha256 != case.lineage["base"].sha256
    inverse = runner.Program.from_dict(case.ledger["inverse"])
    assert runner.apply_program(case.parent, inverse).sha256 == case.lineage["base"].sha256
    assert runner.verify_lowering(case.lineage["base"], runner.StructuredProgram.from_dict(case.ledger["proposal"]),
                                 case.ledger["lowering"])


def test_bad_inverse_is_accounted_as_missing_without_a_success_claim(case_factory, monkeypatch, tmp_path):
    case = case_factory(study="blind-diagnosis")
    selected = _select(case, monkeypatch, _response(case))
    path = case.root / Path(case.assignment["parent_path"]).parent / "evaluator-only-injection.json"
    ledger = runner.read(path)
    ledger["clean_parent_sha256"] = "0" * 64
    with _changed(path, _encoded(_resign(ledger))):
        outcome = runner.safe_evaluation(case.root, case.assignment, selected, {"sha256": "synthetic-seal"},
                                         case.lineage, np.arange(4), tmp_path / "evaluation")
    assert outcome["deterministic"]["evaluation_missing"]
    assert outcome["deterministic"]["error_type"] == "ValueError"
    assert not outcome["deterministic"]["joint_useful_success"]
    assert outcome["missingness_sensitivity"] == {"gain": [-1., 1.], "harm": [-1., 1.], "useful": [0., 1.]}


def test_preparation_promotion_recovers_after_a_partial_move(prepared_template, monkeypatch, tmp_path):
    source = prepared_template.root
    root = tmp_path / "promotion"
    for name in (*runner.PROTOCOL_HASHES, "label-amendment.json"):
        destination = root / runner.PROTOCOL_DIR / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / runner.PROTOCOL_DIR / name, destination)
    calls = []
    targets = (runner.ARTIFACT_DIR / "prepared", runner.PROTOCOL_DIR / "source-snapshot",
               runner.PROTOCOL_DIR / "implementation-amendment.json", runner.LOCK_PATH)

    def synthetic_prepare(stage, origin):
        calls.append(stage)
        assert origin == root.resolve()
        for relative in targets:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (source / relative).is_dir():
                shutil.copytree(source / relative, destination)
            else:
                shutil.copyfile(source / relative, destination)
        return prepared_template.lock

    original_replace = runner.os.replace

    def interrupted_replace(origin, destination):
        if Path(destination) == root / runner.PROTOCOL_DIR / "source-snapshot":
            raise OSError("Synthetic interruption after the first promoted directory")
        return original_replace(origin, destination)

    monkeypatch.setattr(runner, "_prepare", synthetic_prepare)
    with monkeypatch.context() as patch:
        patch.setattr(runner.os, "replace", interrupted_replace)
        with pytest.raises(OSError):
            runner.prepare(root)
    assert (root / runner.ARTIFACT_DIR / "prepared").is_dir()
    journal = runner.read(root / runner.PREPARATION_JOURNAL)
    stage = root / journal["stage_path"]
    assert stage.exists() and not (stage / targets[0]).exists()
    recovered = runner.prepare(root)
    assert recovered == prepared_template.lock and len(calls) == 1
    assert not stage.exists() and not (root / runner.PREPARATION_JOURNAL).exists()
    assert runner.verify_preparation(root) == recovered


@pytest.mark.parametrize("bounds", [(-1., 1.), (-2., 2.), (0., 1.)])
def test_missingness_bounds_follow_the_quantity_and_preserve_every_lineage(bounds):
    records = [{"lineage_id": lineage, "value": value}
               for lineage in runner.LINEAGES for value in (.25, None)]
    result = runner.lineage_summary(records, lambda row: row["value"], missing_bounds=bounds)
    expected = [(.25 + bound) / 2 for bound in bounds]
    assert result["all_assigned_mean_bounds"] == pytest.approx(expected)
    assert all(pair == pytest.approx(expected) for pair in result["lineage_bounds"])
    assert result["estimate"] is None and result["sign_flip"] is None
    with pytest.raises(ValueError, match="lineage"):
        runner.lineage_summary(records[2:], lambda row: row["value"], missing_bounds=bounds)


def test_unanswered_predictions_keep_coverage_and_brier_missingness_explicit():
    unavailable = runner.binary_rubric([-1, -1], [0., 0.], [0, 1])
    assert unavailable["all_assigned_accuracy"] == unavailable["prediction_coverage"] == 0.
    assert unavailable["conditional_accuracy"] is unavailable["conditional_binary_brier"] is None
    assert unavailable["binary_brier_all_assigned_bounds"] == [0., 1.]
    partial = runner.binary_rubric([-1, 1], [0., .8], [0, 1])
    assert partial["prediction_coverage"] == .5
    assert partial["conditional_accuracy"] == 1.
    assert partial["binary_brier_all_assigned_bounds"] == pytest.approx([.02, .52])


@pytest.mark.parametrize("mutation", ["missing_gate", "gate_phase", "missing_account", "missing_evidence"])
def test_evaluation_rejects_an_invalid_gate_before_opening_evaluator_inputs(
        sealed_template, monkeypatch, mutation):
    study = sealed_template

    def forbidden(*args):
        raise AssertionError("Evaluator inputs were opened before the gate was accepted")

    monkeypatch.setattr(runner, "evaluation_inputs", forbidden)
    gate = deepcopy(study.gate)
    path = study.root / runner.GATE_PATH
    if mutation == "missing_gate":
        changed = None
    elif mutation == "gate_phase":
        gate["evaluation_started"] = True
        gate = _resign(gate)
        changed = _encoded(gate)
    elif mutation == "missing_account":
        gate["accounting"].pop()
        gate = _resign(gate)
        changed = _encoded(gate)
    else:
        path = runner.selection_path(study.root, study.lock["assignments"][0]["assignment_id"]).parent
        path /= "invocation-checkpoint.json"
        changed = None
    with _changed(path, changed), pytest.raises((ValueError, FileNotFoundError)):
        runner.run_evaluation(study.root, study.lock, gate)
    assert not (study.root / runner.ARTIFACT_DIR / "evaluation").exists()


def test_evaluator_input_inventory_binds_original_training_and_test_bundles(evaluation_template):
    study = evaluation_template
    runner.validate_gate(study.root, study.gate["sha256"])
    inputs, lineages = runner.evaluation_inputs(study.root, study.lock)
    paths = [record["path"] for record in inputs]
    assert paths == sorted(set(paths))
    expected = {
        (Path("artifacts/research/trained-support/main") / lineage / name).as_posix()
        for lineage in runner.LINEAGES
        for name in (*runner.TRAINING_FILES, "evaluation-features.npz", "evaluation-features.json")
    }
    assert expected <= set(paths)
    assert {item["path"] for item in study.lock["source_files"]} <= set(paths)
    assert {item["path"] for item in study.lock["permitted_input_files"]} <= set(paths)
    assert {runner.GATE_PATH.as_posix(), runner.LOCK_PATH.as_posix()} <= set(paths)
    for record in inputs:
        path = study.root / record["path"]
        assert record["sha256"] == runner.sha(path) and record["bytes"] == path.stat().st_size
    assert set(lineages) == set(runner.LINEAGES)
    assert all(len(value["features"]["test_features"]) == 4 for value in lineages.values())


def test_updated_original_provenance_cannot_replace_the_prepared_training(evaluation_template):
    study = evaluation_template
    directory = study.root / "artifacts/research/trained-support/main" / runner.LINEAGES[0]
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        features = {key: archive[key].copy() for key in archive.files}
    features["target_features"][0, 0] += .125
    changed = _npz_bytes(features)
    provenance = runner.read(directory / "lineage.json")
    provenance["files"]["features.npz"] = {"sha256": hashlib.sha256(changed).hexdigest(), "bytes": len(changed)}
    with _changed(directory / "features.npz", changed), _changed(directory / "lineage.json", _encoded(provenance)):
        assert runner.load_lineage(directory)["features"]["target_features"][0, 0] == 1.125
        with pytest.raises(ValueError, match="training bytes differ"):
            runner.evaluation_inputs(study.root, study.lock)


@pytest.mark.parametrize("mutation", ["missing_bundle", "missing_manifest", "feature_identity", "missing_novelty", "training_overlap"])
def test_evaluator_bundle_is_required_and_cannot_override_training(evaluation_template, mutation):
    study = evaluation_template
    directory = study.root / "artifacts/research/trained-support/main" / runner.LINEAGES[0]
    path, manifest_path = directory / "evaluation-features.npz", directory / "evaluation-features.json"
    manifest = runner.read(manifest_path)
    bundle = _evaluation_bundle()
    if mutation in {"missing_bundle", "missing_manifest"}:
        missing_path = path if mutation == "missing_bundle" else manifest_path
        with _changed(missing_path, None), pytest.raises((ValueError, FileNotFoundError)):
            runner.evaluation_inputs(study.root, study.lock)
        return
    if mutation == "feature_identity":
        manifest["feature_id"] = "different-features"
    else:
        if mutation == "missing_novelty":
            bundle.pop("test_novel_mask")
        else:
            bundle["target_features"] = bundle["test_features"]
        changed = _npz_bytes(bundle)
        manifest.update(sha256=hashlib.sha256(changed).hexdigest(), bytes=len(changed))
    with _changed(path, _npz_bytes(bundle)), _changed(manifest_path, _encoded(manifest)):
        with pytest.raises(ValueError):
            runner.evaluation_inputs(study.root, study.lock)


def test_evaluation_and_replay_bind_the_same_inputs_and_every_assignment(completed_evaluation, monkeypatch):
    study = completed_evaluation
    destination = study.root / runner.ARTIFACT_DIR / "evaluation"
    started = runner.check_signed(runner.read(destination / "evaluation-started.json"))
    assert started["schema"] == "wsp-live-evaluation-started/1"
    assert started["selection_frozen"] is True and started["replay"] is False
    assert started["prepared_lock_sha256"] == study.lock["sha256"]
    assert started["global_selection_seal_sha256"] == study.gate["sha256"]
    assert started["input_files_sha256"] == runner.digest(started["input_files"])
    assert study.manifest["evaluation_started_sha256"] == started["sha256"]
    assert study.manifest["evaluator_input_files_sha256"] == started["input_files_sha256"]
    assert study.manifest["assigned_total"] == study.manifest["evaluation_missing_total"] == 224
    assert len(study.manifest["outcome_files"]) == 224
    runner.check_files(study.root, study.manifest["outcome_files"])
    monkeypatch.setattr(runner, "evaluate_assignment", _synthetic_evaluate)
    replay = runner.run_evaluation(study.root, study.lock, study.gate, replay=True)
    assert replay["all_deterministic_outcomes_match"]
    assert replay["assigned_total"] == replay["evaluation_missing_total"] == len(replay["records"]) == 224
    assert all(record["matches"] and record["evaluation_missing"] for record in replay["records"])
    assert replay["evaluator_input_files_sha256"] == study.manifest["evaluator_input_files_sha256"]
    assert {record["assignment_id"] for record in replay["records"]} == {
        item["assignment_id"] for item in study.lock["assignments"]}
    for replay_mode in (False, True):
        with pytest.raises(ValueError, match="immutable"):
            runner.run_evaluation(study.root, study.lock, study.gate, replay=replay_mode)


def test_replay_rejects_a_self_consistent_replacement_evaluator_bundle(completed_evaluation, monkeypatch):
    study = completed_evaluation
    directory = study.root / "artifacts/research/trained-support/main" / runner.LINEAGES[0]
    bundle = _evaluation_bundle()
    bundle["test_features"][0, 0] += .25
    changed = _npz_bytes(bundle)
    manifest = runner.read(directory / "evaluation-features.json")
    manifest.update(sha256=hashlib.sha256(changed).hexdigest(), bytes=len(changed))

    def forbidden(*args):
        raise AssertionError("Replay ran an assignment after its evaluator inputs changed")

    monkeypatch.setattr(runner, "evaluate_assignment", forbidden)
    with _changed(directory / "evaluation-features.npz", changed), \
            _changed(directory / "evaluation-features.json", _encoded(manifest)):
        assert runner.load_lineage(directory)["features"]["test_features"][0, 0] == 1.25
        with pytest.raises(ValueError, match="Replay evaluator inputs differ"):
            runner.run_evaluation(study.root, study.lock, study.gate, replay=True)


@pytest.mark.parametrize("mutation", ["start_inputs", "manifest_inputs", "manifest_inventory", "outcome_binding"])
def test_replay_rejects_resigned_start_manifest_and_outcome_mutations(completed_evaluation, monkeypatch, mutation):
    study = completed_evaluation
    directory = study.root / runner.ARTIFACT_DIR / "evaluation"
    manifest_path = directory / "manifest.json"
    manifest = runner.read(manifest_path)
    changed_path = manifest_path
    if mutation == "start_inputs":
        changed_path = directory / "evaluation-started.json"
        changed = runner.read(changed_path)
        changed["input_files"].pop()
        changed["input_files_sha256"] = runner.digest(changed["input_files"])
        changed = _resign(changed)
        manifest["evaluation_started_sha256"] = changed["sha256"]
        manifest["evaluator_input_files_sha256"] = changed["input_files_sha256"]
    elif mutation == "manifest_inputs":
        manifest["evaluator_input_files_sha256"] = "0" * 64
        changed = _resign(manifest)
    elif mutation == "manifest_inventory":
        manifest["outcome_files"].pop()
        changed = _resign(manifest)
    else:
        changed_path = directory / study.lock["assignments"][0]["assignment_id"] / "outcome.json"
        changed = runner.read(changed_path)
        changed["deterministic"]["selection_sha256"] = "0" * 64
        changed["deterministic_sha256"] = runner.digest(changed["deterministic"])
        changed = _resign(changed)
        relative = changed_path.relative_to(study.root).as_posix()
        entry = next(item for item in manifest["outcome_files"] if item["path"] == relative)
        encoded = _encoded(changed)
        entry.update(sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded))

    def forbidden(*args):
        raise AssertionError("Replay computed outcomes before validating the original evidence")

    monkeypatch.setattr(runner, "evaluate_assignment", forbidden)
    with _changed(changed_path, _encoded(changed)):
        if changed_path == manifest_path:
            with pytest.raises(ValueError):
                runner.run_evaluation(study.root, study.lock, study.gate, replay=True)
        else:
            with _changed(manifest_path, _encoded(_resign(manifest))), pytest.raises(ValueError):
                runner.run_evaluation(study.root, study.lock, study.gate, replay=True)


@pytest.mark.parametrize("field", [
    "schema", "assigned", "assignment_id", "lineage_id", "terminal_status", "no_op",
    "selection_sha256", "global_seal_sha256", "deterministic_sha256",
    "inner_assignment_id", "inner_selection_sha256", "inner_global_seal_sha256",
])
def test_outcome_cannot_be_resigned_for_a_different_assignment_or_decision(case_factory, field):
    case = case_factory()
    checkpoint = _checkpoint(case.root, case.assignment, case.lock)
    selected = runner.finish_selection(case.root, case.assignment, None, "interrupted", checkpoint)
    gate = {"sha256": "synthetic-seal"}
    outcome = _missing_outcome(case.assignment, selected, gate)
    assert runner.check_outcome(outcome, case.assignment, selected, gate) == outcome
    if field.startswith("inner_"):
        outcome["deterministic"][field.removeprefix("inner_")] = "different-binding"
        outcome["deterministic_sha256"] = runner.digest(outcome["deterministic"])
    else:
        outcome[field] = {"assigned": False, "no_op": True, "terminal_status": "sealed"}.get(field, "different-binding")
    with pytest.raises(ValueError, match="outcome binding"):
        runner.check_outcome(_resign(outcome), case.assignment, selected, gate)


def test_evaluator_makes_the_assignment_durable_before_an_interruption(case_factory, monkeypatch, tmp_path):
    case = case_factory()
    checkpoint = _checkpoint(case.root, case.assignment, case.lock)
    selected = runner.finish_selection(case.root, case.assignment, None, "interrupted", checkpoint)
    destination = tmp_path / "interrupted-evaluation"

    def interrupt(*args):
        assert destination.is_dir()
        raise SimulatedCrash()

    monkeypatch.setattr(runner, "materialize_selection", interrupt)
    with pytest.raises(SimulatedCrash):
        runner.evaluate_assignment(case.root, case.assignment, selected, {"sha256": "synthetic-seal"},
                                   case.lineage, np.arange(4), destination)
    assert destination.is_dir() and not (destination / "outcome.json").exists()


def test_interrupted_evaluation_accounts_the_partial_assignment_without_repeating_it(
        sealed_template, monkeypatch, tmp_path):
    study = _copy_evaluation_study(sealed_template, tmp_path / "interrupted-study")
    calls = []

    def interrupt(*args):
        calls.append(args[1]["assignment_id"])
        args[-1].mkdir(parents=True, exist_ok=False)
        (args[-1] / "partial-evidence.bin").write_bytes(b"preserved synthetic evidence")
        raise SimulatedCrash()

    monkeypatch.setattr(runner, "evaluate_assignment", interrupt)
    with pytest.raises(SimulatedCrash):
        runner.run_evaluation(study.root, study.lock, study.gate)
    first = study.lock["assignments"][0]["assignment_id"]
    assert calls == [first]

    def resume(*args):
        calls.append(args[1]["assignment_id"])
        return _synthetic_evaluate(*args)

    monkeypatch.setattr(runner, "evaluate_assignment", resume)
    result = runner.run_evaluation(study.root, study.lock, study.gate)
    assert calls == [item["assignment_id"] for item in study.lock["assignments"]]
    first_directory = study.root / runner.ARTIFACT_DIR / "evaluation" / first
    first_outcome = runner.read(first_directory / "outcome.json")
    assert first_outcome["deterministic"]["error_type"] == "InterruptedEvaluation"
    assert (first_directory / "partial-evidence.bin").read_bytes() == b"preserved synthetic evidence"
    assert result["assigned_total"] == result["evaluation_missing_total"] == 224


def test_summary_uses_quantity_specific_bounds_for_all_frozen_assignments(sealed_template):
    study = sealed_template
    outcomes = [spec | {
        "terminal_status": "sealed", "no_op": False,
        "deterministic": {"evaluation_missing": True, "joint_useful_success": False},
    } for spec in runner.assigned_specs()]
    summary = runner.summarize(study.root, study.lock, study.gate, outcomes)

    def expect_bounds(result, bounds):
        assert result["missing_quantity_bounds"] == result["all_assigned_mean_bounds"] == list(bounds)
        assert result["lineage_ids"] == list(runner.LINEAGES)
        assert result["estimate"] is result["sign_flip"] is None

    assert summary["assigned_total"] == summary["evaluation_missing_total"] == 224
    for study_name in ("weight_access", "blind_diagnosis"):
        result = summary[study_name]
        expect_bounds(result["primary_contrast"], (-2., 2.))
        for values in result["views"].values():
            for name in ("target_gain", "protected_harm"):
                expect_bounds(values[name], (-1., 1.))
            for name in ("joint_useful_coverage", "read_accuracy", "prediction_coverage", "competing_accuracy",
                         "location_accuracy", "mechanism_accuracy", "diagnosis_coverage"):
                if name in values:
                    expect_bounds(values[name], (0., 1.))
            for name in ("read_confidence", "competing_confidence", "confidence"):
                if name in values:
                    confidence = values[name]
                    assert confidence["conditional_estimate"] is None
                    assert all(record["observed_predictions"] == 0 for record in confidence["per_lineage"])
                    expect_bounds(confidence["prediction_coverage"], (0., 1.))
                    expect_bounds(confidence["all_assigned_brier_sensitivity"], (0., 1.))
    expect_bounds(summary["blind_diagnosis"]["joint_useful_coverage"], (0., 1.))
    with pytest.raises(ValueError, match="lineage"):
        runner.confidence_summary([record for record in outcomes if record["lineage_id"] != runner.LINEAGES[0]])
