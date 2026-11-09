"""Literal selection-only recovery; no acquired inputs, fitting or live calls."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from rlcd.weight_space.data_roles import DataRoleRegistry
from scripts import run_live_studies as runner
from scripts import run_trained_research


def test_publication_failure_is_durable_without_retry_or_scoring(tmp_path, monkeypatch):
    class ForbiddenWork(BaseException):
        pass

    class SimulatedCrash(BaseException):
        pass

    def forbidden(*args, **kwargs):
        raise ForbiddenWork("Selection-only regression crossed its execution boundary")

    for name in (
            "prepare", "verify_preparation", "require_snapshot", "verify_protocols", "configure_runtime",
            "load_lineage", "evaluation_inputs", "run_evaluation", "evaluate_assignment", "evaluate_head",
            "group_primary", "compile_arm", "plant", "supplement", "lower_structured", "apply_program",
            "verify_lowering", "validate_answer", "validate_response"):
        monkeypatch.setattr(runner, name, forbidden)
    for name in ("request", "evaluate", "evaluate_registered"):
        monkeypatch.setattr(runner.ProposerBroker, name, forbidden)
    monkeypatch.setattr(DataRoleRegistry, "__init__", forbidden)
    monkeypatch.setattr(DataRoleRegistry, "register_batch", forbidden)
    monkeypatch.setattr(run_trained_research, "verify_frozen_source", forbidden)

    root, raw = tmp_path / "public", tmp_path / "private"
    inputs = root / "literal-inputs"
    parent = runner.TensorArtifact(runner.ARCHITECTURE, {
        runner.HEAD_WEIGHT: np.zeros((2, 1), dtype=np.float32),
        runner.HEAD_BIAS: np.zeros(2, dtype=np.float32),
    })
    runner.write_json(inputs / "parent.json", parent.to_dict())
    prompt = "Literal selection-only provider attempt. No data or evaluation is available.\n"
    (inputs / "prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")
    schema = {"type": "object"}
    runner.write_json(inputs / "response-schema.json", schema)
    spec = runner.specification("weight-access", "literal-fixture-lineage", "literal-fixture-target", "none")
    assignment = runner.signed(spec | {
        "objective": "Retain a literal terminal failure without executing a proposed edit.", "context": {},
        "parent_path": "literal-inputs/parent.json", "preparation_broker_trace": [], "fitted_offers": [],
        "parameter_values_exposed": 0, "prompt_path": "literal-inputs/prompt.txt",
        "prompt_sha256": runner.sha(inputs / "prompt.txt"), "prompt_bytes": len(prompt.encode("utf-8")),
        "schema_path": "literal-inputs/response-schema.json",
    })
    runner.write_json(inputs / "assignment.json", assignment)
    locked = spec | {"assignment_path": "literal-inputs/assignment.json",
                     "assignment_sha256": runner.sha(inputs / "assignment.json")}
    lock = runner.signed({"assignments": [locked],
                          "provider": {"schema": "literal-provider-config/1", "provider": "synthetic"}})
    runner.write_json(root / runner.LOCK_PATH, lock)
    selected_path = runner.selection_path(root, assignment["assignment_id"])
    public = selected_path.parent
    response = {
        "abstain": False,
        "commands_json": json.dumps([{"op": "add", "tensor": runner.HEAD_BIAS, "values": [1., 0.]}]),
        "selected_fitted_handle": None, "reason": "Literal edit that must never be interpreted",
        "predicted_argmax_codes": [0, 0], "prediction_confidences": [.5, .5],
        "strongest_competing_codes": [1, 1], "competing_confidences": [.5, .5],
    }
    calls, publications = [], []

    def call(provider, received_prompt, *, directory, cwd, schema):
        assert received_prompt == prompt and schema == {"type": "object"}
        assert directory == raw / assignment["assignment_id"] / "provider"
        assert cwd == raw / assignment["assignment_id"] / "cwd"
        runner.check_checkpoint(root, assignment, lock, runner.read(public / "invocation-checkpoint.json"))
        calls.append(directory)
        directory.mkdir(parents=True, exist_ok=False)
        runner.write_json(directory / "request.json", {"prompt": prompt, "configuration": lock["provider"]})
        runner.write_json(directory / "response-schema.json", schema)
        events = b"\n".join(runner.canonical_bytes(event) for event in (
            {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(response)}},
            {"type": "turn.completed", "usage": {"input_tokens": 17, "output_tokens": 7}},
        )) + b"\n"
        (directory / "events.jsonl").write_bytes(events)
        receipt = runner.signed({
            "schema": "wsp-live-provider-attempt/1", "status": "completed", "error": None,
            "returncode": 0, "elapsed_seconds": .25,
            "invocation": {"prompt_sha256": assignment["prompt_sha256"], "schema_sha256": runner.digest(schema),
                           "configuration": lock["provider"]},
            "response": response, "usage": {"input_tokens": 17, "output_tokens": 7},
            "unapproved_events": [], "event_errors": [], "events_sha256": hashlib.sha256(events).hexdigest(),
        })
        runner.write_json(directory / "receipt.json", receipt)
        return receipt

    def fail_publication(received_root, received_assignment, received_lock, checkpoint, source, destination):
        assert (received_root, received_assignment, received_lock) == (root, assignment, lock)
        assert runner.read(public / "provider-receipt.json") == runner.read(source / "receipt.json")
        publications.append(source)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "events.jsonl").write_bytes(b'{"type":"literal.partial-publication"}\n')
        raise OSError("literal-private-publication-error-detail")

    def crash_after_decision(*args, **kwargs):
        assert (public / "publication-failure.json").exists() and not selected_path.exists()
        raise SimulatedCrash("Crash after the durable publication decision")

    finish_selection = runner.finish_selection
    monkeypatch.setattr(runner.CodexProvider, "call", call)
    monkeypatch.setattr(runner, "publish_provider", fail_publication)
    monkeypatch.setattr(runner, "finish_selection", crash_after_decision)
    with pytest.raises(SimulatedCrash):
        runner.select_one(root, raw, locked, lock)
    assert len(calls) == len(publications) == 1
    preserved = {path: path.read_bytes() for path in (
        public / "invocation-checkpoint.json", public / "provider-receipt.json",
        public / "publication-failure.json", public / "provider/events.jsonl",
    )}
    checkpoint = runner.read(public / "invocation-checkpoint.json")
    retained = runner.read(public / "provider-receipt.json")
    failure = runner.read(public / "publication-failure.json")
    assert runner.check_publication_failure(root, assignment, lock, checkpoint, failure) == (failure, retained)
    assert {key: failure[key] for key in (
        "assignment_id", "assignment_sha256", "prepared_lock_sha256", "invocation_checkpoint_sha256",
        "provider_receipt_sha256", "phase", "error_type", "evidence_scope", "retry_performed",
    )} == {
        "assignment_id": assignment["assignment_id"], "assignment_sha256": locked["assignment_sha256"],
        "prepared_lock_sha256": lock["sha256"], "invocation_checkpoint_sha256": checkpoint["sha256"],
        "provider_receipt_sha256": retained["sha256"], "phase": "provider_publication", "error_type": "OSError",
        "evidence_scope": "receipt_only", "retry_performed": False,
    }
    assert b"literal-private-publication-error-detail" not in preserved[public / "publication-failure.json"]
    # Recovery must use the retained receipt even if the private copy is no longer available.
    (raw / assignment["assignment_id"] / "provider/receipt.json").unlink()
    monkeypatch.setattr(runner, "finish_selection", finish_selection)
    monkeypatch.setattr(runner.CodexProvider, "call", forbidden)
    monkeypatch.setattr(runner, "publish_provider", forbidden)
    selected = runner.select_one(root, raw, locked, lock)
    selected_bytes = selected_path.read_bytes()
    assert runner.select_one(root, raw, locked, lock) == selected
    assert selected_path.read_bytes() == selected_bytes
    assert len(calls) == len(publications) == 1
    assert all(path.read_bytes() == content for path, content in preserved.items())
    assert selected["provider_status"] == retained["status"] == "completed"
    assert selected["terminal_status"] == "interrupted"
    assert selected["response"] == response
    assert selected["provider_receipt_sha256"] == retained["sha256"]
    assert selected["publication_failure_sha256"] == failure["sha256"]
    assert selected["parent_sha256"] == selected["candidate_sha256"] == parent.sha256
    assert selected["program_sha256"] is None and selected["no_op"] is False
    assert selected["evaluation_started"] is False and selected["answer_valid"] is False
    assert selected["validation_error_type"] is None and selected["answer_validation_error_type"] is None
    assert [event["request"] for event in selected["broker_trace"]] == [{"op": "harness_deadline_seal"}]
    assert {record["path"] for record in selected["evidence_files"]} == {
        path.relative_to(root).as_posix() for path in preserved}
    assert not any((public / name).exists() for name in (
        "candidate.json", "proposal.json", "program.json", "lowering.json", "rollback.json"))
    assert runner.materialize_selection(root, assignment, selected)[2] is None
    omitted = runner.signed({key: value for key, value in selected.items()
                             if key not in ("sha256", "publication_failure_sha256")})
    with pytest.raises(ValueError, match="omits its durable publication failure"):
        runner.validate_selection(root, assignment, lock, omitted)

    gate = runner.signed({"schema": "literal-global-seal/1", "selection_sha256": selected["sha256"]})
    destination = tmp_path / "missing-outcome"
    outcome = runner.safe_evaluation(root, assignment, selected, gate, None, None, destination)
    assert runner.check_outcome(outcome, assignment, selected, gate) == outcome
    assert runner.read(destination / "outcome.json") == outcome
    assert outcome["assigned"] is True and outcome["assignment_id"] == assignment["assignment_id"]
    assert outcome["terminal_status"] == "interrupted" and outcome["no_op"] is False
    assert outcome["selection_sha256"] == selected["sha256"]
    assert outcome["global_seal_sha256"] == gate["sha256"]
    assert outcome["deterministic"] == {
        "assignment_id": assignment["assignment_id"], "selection_sha256": selected["sha256"],
        "global_seal_sha256": gate["sha256"], "evaluation_missing": True, "joint_useful_success": False,
        "error_type": "ProviderPublicationFailure", "publication_failure_sha256": failure["sha256"],
    }
    assert outcome["missingness_sensitivity"] == {"gain": [-1., 1.], "harm": [-1., 1.], "useful": [0., 1.]}
    assert runner.metric(outcome, "proposed", "target_gain") is None
    assert runner.metric(outcome, "proposed", "protected_harm") is None
    assert runner.useful_value(outcome) is None
    for field, value in (("publication_failure_sha256", None), ("evaluation_missing", False)):
        mutated = deepcopy(outcome)
        if value is None:
            del mutated["deterministic"][field]
        else:
            mutated["deterministic"][field] = value
        mutated["deterministic_sha256"] = runner.digest(mutated["deterministic"])
        mutated = runner.signed({key: item for key, item in mutated.items() if key != "sha256"})
        with pytest.raises(ValueError, match="Evaluation outcome binding differs"):
            runner.check_outcome(mutated, assignment, selected, gate)

    for receipts, latency, totals, known in (
            ([retained], [.25], (17, 7, None), 1),
            ([retained, None], [.25, None], (17, 7, None), 1),
            ([None], [None], (None, None, None), 0)):
        cost = runner.provider_usage_accounting(receipts)
        usage = cost["provider_usage"]
        assert cost["provider_latency_seconds"] == latency
        assert (cost["provider_input_tokens"], cost["provider_output_tokens"]) == totals[:2]
        assert cost["provider_financial_cost_usd"] is None
        assert (usage["assigned_attempts"], usage["known_receipts"], usage["unknown_receipts"]) == (
            len(receipts), known, len(receipts) - known)
        assert usage["units"] == "tokens" and usage["missing_usage_is_zero"] is False
        assert usage["cached_input_added_to_input"] is False
        assert "lower_bounds_when_coverage_is_incomplete" in usage["totals_scope"]
        for key, total in zip(("input_tokens", "output_tokens", "cached_input_tokens"), totals):
            observed = 0 if key == "cached_input_tokens" else known
            assert usage[key] == {"observed_sum": total, "known_attempts": observed,
                                  "unknown_attempts": len(receipts) - observed,
                                  "complete_for_attempts": observed == len(receipts)}

    # Patch only summary grouping; assignment construction above keeps its frozen constants.
    monkeypatch.setattr(runner, "LINEAGES", (assignment["lineage_id"],))
    useful = runner.lineage_summary([outcome], runner.useful_value, (0., 1.))
    assert useful["estimate"] is None and useful["sign_flip"] is None
    assert useful["lineage_ids"] == [assignment["lineage_id"]]
    assert useful["all_assigned_mean_bounds"] == [0., 1.]
    for key in ("target_gain", "protected_harm"):
        summary = runner.lineage_summary([outcome], lambda record: runner.metric(record, "proposed", key))
        assert summary["estimate"] is None and summary["all_assigned_mean_bounds"] == [-1., 1.]
