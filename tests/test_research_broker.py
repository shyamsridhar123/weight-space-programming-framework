"""The returned JSON surface must not be an endpoint/hidden-evaluation oracle."""
import json
import sys

import pytest

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.data_roles import DataRoleRegistry, RegisteredBatch
from rlcd.weight_space.broker import ProposerBroker, WeightView
from rlcd.weight_space.workflow import BudgetLimits, TaskSpec


def test_json_checkpoint_preserves_previous_receipt_when_replace_fails(tmp_path, monkeypatch):
    from rlcd.weight_space.live_provider import write_json
    path = tmp_path / "receipt.json"
    write_json(path, {"status": "pending"})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert path.read_bytes() == original

    def interrupted(source, destination):
        assert json.loads(source.read_text()) == {"status": "completed"}
        assert destination == path
        raise OSError("Interrupted before replacement")

    monkeypatch.setattr("rlcd.weight_space.live_provider.os.replace", interrupted)
    with pytest.raises(OSError):
        write_json(path, {"status": "completed"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_opaque_submission_and_all_denied_routes_are_recorded():
    base = TensorArtifact("affine/1", {"W": [[0.123456789]], "b": [0.]},
                          {"private_note": "must stay with evaluator"})
    task = TaskSpec("opaque", "edit", "Change one score", base.architecture, base.sha256,
                    {"secret_test": "never disclose"})
    broker = ProposerBroker(task, base=base, view=WeightView("rounded", 2))
    public = [broker.request({"op": "metadata"}), broker.request({"op": "weights", "tensor": "W"})]
    assert public[1]["values"] == [[0.12]]
    submissions = [{"op": "submit", "commands": [], "intent": "No-op"},
                   {"op": "submit", "commands": [{"op": "add", "tensor": "W", "values": [[1e308]]}], "intent": "overflow"},
                   {"op": "submit", "commands": [{"op": "add", "tensor": "W", "values": [[1.]]}], "intent": "edit"}]
    handles = []
    for request in submissions:
        response = broker.request(request)
        assert set(response) == {"status", "candidate_handle"}
        assert response["status"] == "submitted"
        handles.append(response["candidate_handle"])
        public.append(response)
    assert len(set(handles)) == 3
    for op in ("forward", "activations", "evaluate", "feedback", "filesystem", "network", "exec", "provider_usage", "audit", "candidate"):
        response = broker.request({"op": op, "candidate_handle": handles[-1]})
        assert response == {"status": "denied"}
        public.append(response)
    exposed = json.dumps(public)
    for secret in (base.sha256, "0.123456789", "private_note", "secret_test", "artifact_sha256", "no_op"):
        assert secret not in exposed
    assert broker.request({"op": "seal", "candidate_handle": handles[-1]}) == {"status": "sealed"}
    assert broker.request({"op": "weights", "tensor": "W"}) == {"status": "denied"}
    assert broker.request({"op": "metadata"}) == {"status": "denied"}
    audit = broker.evaluate(lambda candidate: {"observed": float(candidate.artifact.tensors["W"][0, 0])}, evaluator_id="isolated")
    assert audit["status"] == "succeeded"
    assert audit["evaluation"]["result"]["observed"] == 1.123456789
    assert len(audit["attempts"]) == 3
    assert len(audit["public_trace"]) == len(public) + 3
    with pytest.raises(ValueError):
        broker.evaluate(lambda c: {}, evaluator_id="again")


def test_recursive_request_is_denied_and_counted_before_trusted_sealing():
    base = TensorArtifact("affine/1", {"W": [[1.]]})
    task = TaskSpec("recursive-ingress", "edit", "Check ingress accounting", base.architecture, base.sha256)
    broker = ProposerBroker(task, base=base, max_requests=1)
    usage, handles = broker._session.usage, broker._handles.copy()
    nested = None
    for _ in range(sys.getrecursionlimit()):
        nested = [nested]

    assert broker.request({"op": "submit", "commands": nested, "intent": "nested"}) == {"status": "denied"}
    trace = broker.public_trace
    assert len(trace) == 1
    assert trace[0]["request"] == {"op": "malformed"}
    assert trace[0]["response"] == {"status": "denied"}
    assert broker._session.usage == usage and broker._handles == handles

    ordinary = {"op": "submit", "commands": [], "intent": "Request cap must deny this"}
    assert broker.request(ordinary) == {"status": "denied"}
    assert broker._session.usage == usage and broker._handles == handles
    assert not broker.is_sealed

    broker.abstain()
    assert broker.is_sealed and broker._session.is_sealed
    trace = broker.public_trace
    assert [event["request"] for event in trace] == [
        {"op": "malformed"}, ordinary, {"op": "harness_deadline_seal"}]
    assert trace[-1]["response"] == {"status": "sealed"}
    assert broker._session.usage == usage and broker._handles == handles
    broker.abstain()
    assert broker.public_trace == trace


def test_subset_and_invalid_selection_do_not_release_endpoint_state():
    base = TensorArtifact("affine/1", {"W": [[1., 2.]]})
    task = TaskSpec("subset", "edit", "Subset control", base.architecture, base.sha256)
    broker = ProposerBroker(task, base=base, view=WeightView("subset", coordinates=(("W", (0, 0)),)),
                            limits=BudgetLimits(proposals=1))
    assert broker.request({"op": "weights", "tensor": "W"}) == {"status": "denied"}
    assert broker.request({"op": "weights", "tensor": "W", "indices": [[0, 1]]}) == {"status": "denied"}
    assert broker.request({"op": "weights", "tensor": "W", "indices": [[0, 0]]})["values"] == [1.]
    submitted = broker.request({"op": "submit", "commands": [{"op": "exec", "tensor": "W"}], "intent": "bad"})
    assert submitted["status"] == "submitted"
    assert set(submitted) == {"status", "candidate_handle"}
    for op in ("candidate", "feedback"):
        assert broker.request({"op": op, "candidate_handle": submitted["candidate_handle"]}) == {"status": "denied"}
    assert broker.request({"op": "submit", "commands": [], "intent": "no budget"}) == {"status": "denied"}
    broker.request({"op": "seal", "candidate_handle": submitted["candidate_handle"]})
    invoked = []
    audit = broker.evaluate(lambda c: invoked.append(c), evaluator_id="not_invoked")
    assert audit["status"] == "invalid"
    assert invoked == []
    assert audit["attempts"][0]["status"] == "invalid"


def test_provider_failure_accounting_and_started_tool_detection(tmp_path, monkeypatch):
    from rlcd.weight_space.live_provider import CodexProvider, parse_events
    events = [
        {"type": "item.completed", "item": {"type": "error", "message": "Configuration warning"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Thinking about the request."}},
        {"type": "item.started", "item": {"type": "command_execution", "command": "blocked"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": '{"actions": [], "reason": "done"}'}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 3}},
    ]
    response, usage, tools, errors, diagnostics = parse_events("\n".join(json.dumps(e) for e in events))
    assert response == {"actions": [], "reason": "done"} and not errors
    assert usage["input_tokens"] == 10 and tools == ["item.started:command_execution"]
    assert diagnostics[0]["item"]["message"] == "Configuration warning"
    monkeypatch.setattr("rlcd.weight_space.live_provider.shutil.which", lambda _: None)
    receipt = CodexProvider().call("Return JSON", directory=tmp_path / "attempt", cwd=tmp_path / "cwd", schema={"type": "object"})
    assert receipt["status"] == "failed" and receipt["error"].startswith("transport_startup_failed")
    assert json.loads((tmp_path / "attempt" / "receipt.json").read_text())["sha256"] == receipt["sha256"]


def test_provider_version_decode_failure_retains_failed_attempt_and_trial(tmp_path, monkeypatch):
    import hashlib
    from unittest.mock import Mock
    from rlcd.weight_space.live_provider import CodexProvider, run_brokered

    cli = tmp_path / "fake-codex"
    cli.write_bytes(b"offline transport placeholder; never executed")
    monkeypatch.setattr("rlcd.weight_space.live_provider.shutil.which", lambda _: str(cli))
    probe = Mock(side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"))
    worker = Mock(side_effect=AssertionError("Worker must not start"))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.run", probe)
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.Popen", worker)
    base = TensorArtifact("affine/1", {"W": [[1.]], "b": [0.]})
    task = TaskSpec("version-decode", "edit", "Retain a failed offline version probe",
                    base.architecture, base.sha256)
    broker = ProposerBroker(task, base=base)
    charge = Mock(wraps=broker.record_usage)
    monkeypatch.setattr(broker, "record_usage", charge)
    directory = tmp_path / "trial"
    trial = run_brokered(CodexProvider(), broker, prompt="Return JSON actions",
                         directory=directory, max_rounds=3)

    probe.assert_called_once_with([str(cli), "--version"], capture_output=True, text=True,
                                  encoding="utf-8", timeout=15, check=True)
    worker.assert_not_called()
    assert trial["assigned_trials"] == 1 and len(trial["attempts"]) == 1
    attempt = trial["attempts"][0]
    assert trial["status"] == attempt["status"] == "failed"
    assert attempt["error"] == "transport_startup_failed:UnicodeDecodeError"
    assert attempt["usage"] == {} and attempt["cost"]["amount"] is None
    assert attempt["response"] is None and attempt["returncode"] is None
    assert attempt["invocation"]["cli_version"] is None
    assert attempt["invocation"]["executable_sha256"] == hashlib.sha256(cli.read_bytes()).hexdigest()
    charge.assert_not_called()
    assert not [event for event in broker._session.trace if event.to_dict()["action"] == "provider_usage"]
    assert broker._session.usage.input_tokens == broker._session.usage.output_tokens == 0
    assert broker._session.usage.proposals == 0
    assert broker.is_sealed and broker._session.is_sealed
    assert [event["request"]["op"] for event in broker.public_trace] == ["metadata", "harness_deadline_seal"]
    folder = directory / "provider-0"
    assert json.loads((folder / "receipt.json").read_text(encoding="utf-8")) == attempt
    assert json.loads((directory / "trial.json").read_text(encoding="utf-8")) == trial
    assert (folder / "events.jsonl").read_bytes() == b""
    assert (folder / "transport.stderr.private").read_bytes() == b""
    assert attempt["events_sha256"] == hashlib.sha256(b"").hexdigest()


def test_provider_completion_lifecycle_is_unambiguous_before_broker_actions(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from unittest.mock import Mock
    from rlcd.weight_space.live_provider import CodexProvider, run_brokered

    cli = tmp_path / "fake-codex"
    cli.write_bytes(b"offline transport placeholder; never executed")
    monkeypatch.setattr("rlcd.weight_space.live_provider.shutil.which", lambda _: str(cli))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="offline-test\n"))
    transport = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(returncode=0,
        communicate=lambda *args, **kwargs: (raw, "")))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.Popen", transport)
    base = TensorArtifact("affine/1", {"W": [[1.]], "b": [0.]})
    task = TaskSpec("completion-lifecycle", "edit", "Retain ambiguous offline endings",
                    base.architecture, base.sha256)
    response = {"actions": [json.dumps({"op": "submit", "commands": [], "intent": "Freeze baseline"}),
                            json.dumps({"op": "seal", "candidate_handle": None})],
                "reason": "offline fixture"}
    usage = {"input_tokens": 17, "output_tokens": 7}
    later_response = dict(response, reason="Response after completion")
    prefix = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Thinking about the request."}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(response)}},
    ]
    completed = {"type": "turn.completed", "usage": usage}
    cases = [
        ("commentary", prefix + [completed], [], response, usage),
        ("duplicate", prefix + [completed, {"type": "turn.completed",
                                           "usage": {"input_tokens": 1, "output_tokens": 0}}],
         ["duplicate_turn_completed"], response, {}),
        ("later_message", prefix + [completed, {"type": "item.completed",
            "item": {"type": "agent_message", "text": json.dumps(later_response)}}],
         ["agent_message_after_turn_completed"], later_response, {}),
    ]

    for name, events, errors, expected_response, expected_usage in cases:
        raw = "\n".join(json.dumps(event) for event in events) + "\n"
        transport.reset_mock()
        broker = ProposerBroker(task, base=base)
        charge = Mock(wraps=broker.record_usage)
        monkeypatch.setattr(broker, "record_usage", charge)
        directory = tmp_path / name
        trial = run_brokered(CodexProvider(), broker, prompt="Return the fixed offline actions",
                             directory=directory, max_rounds=3)
        valid = not errors
        assert transport.call_count == 1 and len(trial["attempts"]) == 1
        assert trial["assigned_trials"] == 1
        attempt = trial["attempts"][0]
        assert attempt["status"] == ("completed" if valid else "failed")
        assert attempt["error"] == (None if valid else "transport_or_response_failed")
        assert attempt["event_errors"] == errors
        assert attempt["response"] == expected_response and attempt["usage"] == expected_usage
        assert attempt["cost"]["amount"] is None
        folder = directory / "provider-0"
        assert json.loads((folder / "receipt.json").read_text(encoding="utf-8")) == attempt
        assert json.loads((directory / "trial.json").read_text(encoding="utf-8")) == trial
        assert (folder / "events.jsonl").read_bytes() == raw.encode("utf-8")
        assert attempt["events_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert trial["status"] == ("sealed" if valid else "failed") and broker.is_sealed
        assert [event["request"]["op"] for event in broker.public_trace] == (
            ["metadata", "submit", "seal"] if valid else ["metadata", "harness_deadline_seal"])
        if valid:
            charge.assert_called_once_with(17, 7)
        else:
            charge.assert_not_called()


def test_malformed_provider_event_type_retains_failed_attempt_and_trial(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from unittest.mock import Mock
    from rlcd.weight_space.live_provider import CodexProvider, run_brokered

    cli = tmp_path / "fake-codex"
    cli.write_bytes(b"offline transport placeholder; never executed")
    monkeypatch.setattr("rlcd.weight_space.live_provider.shutil.which", lambda _: str(cli))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="offline-test\n"))
    transport = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(returncode=0,
        communicate=lambda *args, **kwargs: (raw, "")))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.Popen", transport)
    base = TensorArtifact("affine/1", {"W": [[1.]], "b": [0.]})
    task = TaskSpec("event-type", "edit", "Retain a malformed offline event", base.architecture, base.sha256)
    response = {"actions": [json.dumps({"op": "submit", "commands": [], "intent": "Freeze baseline"}),
                            json.dumps({"op": "seal", "candidate_handle": None})],
                "reason": "offline fixture"}
    usage = {"input_tokens": 12, "output_tokens": 3}
    reasoning_item = {"type": "reasoning", "text": "offline fixture"}
    # All malformed representatives are falsey, so a truthiness guard is insufficient.
    malformed_types = (None, 0, False, [], {})
    cases = [({"type": event_type, "item": reasoning_item}, False)
             for event_type in malformed_types]
    # Rejection belongs to the event boundary, including events without an item.
    cases.extend(({"type": event_type}, False) for event_type in malformed_types)
    cases.extend(({"type": event_type, "item": reasoning_item}, True)
                 for event_type in ("item.started", "provider.extension", ""))
    cases.append(({"item": reasoning_item}, True))

    for index, (leading_event, valid) in enumerate(cases):
        raw = "\n".join(json.dumps(event) for event in [
            leading_event,
            {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(response)}},
            {"type": "turn.completed", "usage": usage},
        ]) + "\n"
        transport.reset_mock()
        broker = ProposerBroker(task, base=base)
        directory = tmp_path / f"case-{index}"
        trial = run_brokered(CodexProvider(), broker, prompt="Return the fixed offline actions",
                             directory=directory, max_rounds=3)
        assert transport.call_count == 1 and len(trial["attempts"]) == 1
        attempt = trial["attempts"][0]
        assert trial["status"] == ("sealed" if valid else "failed")
        assert attempt["status"] == ("completed" if valid else "failed")
        assert attempt["error"] == (None if valid else "transport_or_response_failed")
        assert attempt["event_errors"] == ([] if valid else ["invalid_event"])
        assert attempt["response"] == response and attempt["usage"] == usage
        assert attempt["unapproved_events"] == [] and attempt["diagnostic_items"] == 0
        assert attempt["cost"]["amount"] is None
        folder = directory / "provider-0"
        assert json.loads((folder / "receipt.json").read_text(encoding="utf-8")) == attempt
        assert json.loads((directory / "trial.json").read_text(encoding="utf-8")) == trial
        assert (folder / "events.jsonl").read_bytes() == raw.encode("utf-8")
        assert attempt["events_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert broker.is_sealed
        assert [event["request"]["op"] for event in broker.public_trace] == (
            ["metadata", "submit", "seal"] if valid else ["metadata", "harness_deadline_seal"])
        charges = [event.to_dict() for event in broker._session.trace
                   if event.to_dict()["action"] == "provider_usage"]
        assert len(charges) == 1
        assert {key: getattr(broker._session.usage, key) for key in usage} == usage
        assert broker._session.usage.proposals == int(valid)
        assert charges[0]["status"] == "succeeded" and charges[0]["charges"] == usage


def test_provider_usage_is_complete_before_broker_actions(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from rlcd.weight_space.live_provider import CodexProvider, run_brokered

    cli = tmp_path / "fake-codex"
    cli.write_bytes(b"offline transport placeholder; never executed")
    monkeypatch.setattr("rlcd.weight_space.live_provider.shutil.which", lambda _: str(cli))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="offline-test\n"))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.Popen",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0,
                            communicate=lambda *args, **kwargs: (raw, "")))
    base = TensorArtifact("affine/1", {"W": [[1.]], "b": [0.]})
    task = TaskSpec("usage-admission", "edit", "Freeze the synthetic baseline", base.architecture, base.sha256)
    response = {"actions": [json.dumps({"op": "submit", "commands": [], "intent": "Freeze baseline"}),
                            json.dumps({"op": "seal", "candidate_handle": None})],
                "reason": "NaN Infinity -Infinity 1e400 are ordinary strings"}
    cases = [
        ({"cached_input_tokens": 12}, False),
        ({}, False),
        ({"input_tokens": 12}, False),
        ({"output_tokens": 3}, False),
        (None, False),
        ([], False),
        ({"input_tokens": "12", "output_tokens": 3}, False),
        ({"input_tokens": 12, "output_tokens": 3.5}, False),
        ({"input_tokens": True, "output_tokens": 3}, False),
        ({"input_tokens": 12, "output_tokens": False}, False),
        ({"input_tokens": -1, "output_tokens": 3}, False),
        ({"input_tokens": 12, "output_tokens": -1}, False),
        ({"input_tokens": 12, "output_tokens": 3, "cached_input_tokens": 9}, True),
        ({"input_tokens": 0, "output_tokens": 0}, True),
    ]
    for index, (usage, valid) in enumerate(cases):
        raw = "\n".join(json.dumps(event) for event in [
            {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(response)}},
            {"type": "turn.completed", "usage": usage},
        ]) + "\n"
        # A synthetic completed receipt independently exercises the common guard.
        for route in ("transport", "completed-receipt"):
            provider = CodexProvider() if route == "transport" else SimpleNamespace(tool_free=True,
                call=lambda *args, **kwargs: {"status": "completed", "usage": usage, "response": response})
            broker = ProposerBroker(task, base=base)
            directory = tmp_path / f"{index}-{route}"
            trial = run_brokered(provider, broker, prompt="Return the fixed offline actions",
                                 directory=directory, max_rounds=1)
            attempt = trial["attempts"][0]
            assert json.loads((directory / "trial.json").read_text()) == trial
            assert broker.is_sealed and attempt["usage"] == usage
            assert trial["status"] == ("sealed" if valid else "failed" if route == "transport" else "invalid")
            assert [event["request"]["op"] for event in broker.public_trace] == (
                ["metadata", "submit", "seal"] if valid else ["metadata", "harness_deadline_seal"])
            charges = [event.to_dict() for event in broker._session.trace
                       if event.to_dict()["action"] == "provider_usage"]
            assert len(charges) == int(valid)
            expected = {key: usage[key] if valid else 0 for key in ("input_tokens", "output_tokens")}
            assert {key: getattr(broker._session.usage, key) for key in expected} == expected
            assert broker._session.usage.proposals == int(valid)
            if valid:
                assert charges[0]["status"] == "succeeded" and charges[0]["charges"] == expected
            if route == "transport":
                folder = directory / "provider-0"
                assert json.loads((folder / "receipt.json").read_text()) == attempt
                assert (folder / "events.jsonl").read_bytes() == raw.encode("utf-8")
                assert attempt["events_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
                assert attempt["response"] == response and attempt["cost"]["amount"] is None
                assert attempt["error"] == (None if valid else "invalid_provider_usage")


def test_nonfinite_provider_json_retains_failed_attempt_and_trial(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from unittest.mock import Mock
    from rlcd.weight_space.live_provider import CodexProvider, run_brokered

    cli = tmp_path / "fake-codex"
    cli.write_bytes(b"offline transport placeholder; never executed")
    monkeypatch.setattr("rlcd.weight_space.live_provider.shutil.which", lambda _: str(cli))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="offline-test\n"))
    transport = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(returncode=0,
        communicate=lambda *args, **kwargs: (raw, "")))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.Popen", transport)
    base = TensorArtifact("affine/1", {"W": [[1.]], "b": [0.]})
    task = TaskSpec("finite-json", "edit", "Retain a malformed offline attempt", base.architecture, base.sha256)
    response = {"actions": [json.dumps({"op": "submit", "commands": [], "intent": "Freeze baseline"})],
                "reason": "offline fixture"}
    valid_usage = {"input_tokens": 12, "output_tokens": 3}
    for token in ("NaN", "Infinity", "-Infinity", "1e400", "-1e400"):
        for field in ("usage", "final"):
            final = json.dumps(response)
            usage = json.dumps(valid_usage)
            if field == "usage":
                usage = '{"input_tokens": ' + token + ', "output_tokens": 3}'
            else:
                final = final[:-1] + ', "nested": [{"value": ' + token + '}]}'
            raw = (json.dumps({"type": "item.completed", "item": {
                "type": "agent_message", "text": final}}) + "\n"
                + '{"type": "turn.completed", "usage": ' + usage + '}\n')
            transport.reset_mock()
            broker = ProposerBroker(task, base=base)
            directory = tmp_path / f"{token}-{field}"
            trial = run_brokered(CodexProvider(), broker, prompt="Return the fixed offline actions",
                                 directory=directory, max_rounds=3)
            assert transport.call_count == 1 and len(trial["attempts"]) == 1
            attempt = trial["attempts"][0]
            assert trial["status"] == attempt["status"] == "failed"
            assert attempt["error"] == "transport_or_response_failed"
            assert attempt["event_errors"] == ["invalid_event" if field == "usage" else "invalid_final_response"]
            assert attempt["usage"] == ({} if field == "usage" else valid_usage)
            assert attempt["response"] == (response if field == "usage" else None)
            assert attempt["cost"]["amount"] is None
            folder = directory / "provider-0"
            assert json.loads((folder / "receipt.json").read_text()) == attempt
            assert json.loads((directory / "trial.json").read_text()) == trial
            assert (folder / "events.jsonl").read_bytes() == raw.encode("utf-8")
            assert attempt["events_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
            assert broker.is_sealed and [event["request"]["op"] for event in broker.public_trace] == [
                "metadata", "harness_deadline_seal"]
            charges = [event.to_dict() for event in broker._session.trace
                       if event.to_dict()["action"] == "provider_usage"]
            assert len(charges) == int(field == "final")
            expected = {key: valid_usage[key] if field == "final" else 0 for key in valid_usage}
            assert {key: getattr(broker._session.usage, key) for key in expected} == expected
            assert broker._session.usage.proposals == 0
            if field == "final":
                assert charges[0]["status"] == "succeeded" and charges[0]["charges"] == valid_usage


def test_failed_provider_usage_over_budget_retains_attempt_and_abstains(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from rlcd.weight_space.live_provider import run_brokered

    base = TensorArtifact("affine/1", {"W": [[1.]], "b": [0.]})
    task = TaskSpec("failed-usage-budget", "edit", "Retain an over-budget failed attempt",
                    base.architecture, base.sha256)
    usage = {"input_tokens": 12, "output_tokens": 3}
    attempt = {"status": "failed", "usage": usage, "response": None}
    provider = SimpleNamespace(tool_free=True, call=Mock(return_value=attempt))
    broker = ProposerBroker(task, base=base, limits=BudgetLimits(input_tokens=11))
    directory = tmp_path / "trial"
    trial = run_brokered(provider, broker, prompt="Return the fixed offline actions",
                         directory=directory, max_rounds=3)

    provider.call.assert_called_once()
    assert trial["status"] == attempt["status"] == "failed"
    assert trial["attempts"] == [attempt] and trial["assigned_trials"] == 1
    assert json.loads((directory / "trial.json").read_text(encoding="utf-8")) == trial
    assert broker.is_sealed and [event["request"]["op"] for event in broker.public_trace] == [
        "metadata", "harness_deadline_seal"]
    charges = [event.to_dict() for event in broker._session.trace
               if event.to_dict()["action"] == "provider_usage"]
    assert len(charges) == 1 and charges[0]["status"] == "denied"
    assert charges[0]["requested_charges"] == usage and charges[0]["charges"] == {}
    assert charges[0]["error"]["type"] == "BudgetExceeded"
    assert broker._session.usage.input_tokens == broker._session.usage.output_tokens == 0
    assert broker._session.usage.proposals == 0


def test_deeply_nested_provider_json_retains_failed_attempt_and_trial(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from unittest.mock import Mock
    from rlcd.weight_space.live_provider import CodexProvider, run_brokered

    cli = tmp_path / "fake-codex"
    cli.write_bytes(b"offline transport placeholder; never executed")
    monkeypatch.setattr("rlcd.weight_space.live_provider.shutil.which", lambda _: str(cli))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="offline-test\n"))
    transport = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(returncode=0,
        communicate=lambda *args, **kwargs: (raw, "")))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.Popen", transport)
    base = TensorArtifact("affine/1", {"W": [[1.]], "b": [0.]})
    task = TaskSpec("json-depth", "edit", "Retain a malformed offline attempt", base.architecture, base.sha256)
    response = {"actions": [json.dumps({"op": "submit", "commands": [], "intent": "Freeze baseline"}),
                            json.dumps({"op": "seal", "candidate_handle": None})],
                "reason": "offline fixture"}
    usage = {"input_tokens": 12, "output_tokens": 3}
    # CPython's JSON decoder can have a separate C recursion guard; use the
    # deep string from its recursion tests without recursively building objects.
    depth = 100_000
    nested = "[" * depth + "0" + "]" * depth
    with pytest.raises(RecursionError):
        json.loads(nested)
    for field in ("envelope", "final", "action", "shallow"):
        returned_response = response | {"actions": [nested]} if field == "action" else response
        final = nested if field == "final" else json.dumps(returned_response)
        payload = nested if field == "envelope" else "[[0]]"
        raw = ('{"type":"provider.extension","payload":' + payload + '}\n'
               + json.dumps({"type": "item.completed", "item": {
                   "type": "agent_message", "text": final}}) + "\n"
               + json.dumps({"type": "turn.completed", "usage": usage}) + "\n")
        transport.reset_mock()
        broker = ProposerBroker(task, base=base)
        directory = tmp_path / field
        trial = run_brokered(CodexProvider(), broker, prompt="Return the fixed offline actions",
                             directory=directory, max_rounds=3)
        valid = field == "shallow"
        completed = field in ("action", "shallow")
        assert transport.call_count == 1 and len(trial["attempts"]) == trial["assigned_trials"] == 1
        attempt = trial["attempts"][0]
        assert trial["status"] == ("invalid" if field == "action" else "sealed" if valid else "failed")
        assert attempt["status"] == ("completed" if completed else "failed")
        assert attempt["error"] == (None if completed else "transport_or_response_failed")
        assert attempt["event_errors"] == ([] if completed else [
            "invalid_event" if field == "envelope" else "invalid_final_response"])
        assert attempt["response"] == (None if field == "final" else returned_response)
        assert attempt["usage"] == usage
        assert attempt["unapproved_events"] == [] and attempt["diagnostic_items"] == 0
        assert attempt["cost"]["amount"] is None
        folder = directory / "provider-0"
        assert json.loads((folder / "receipt.json").read_text(encoding="utf-8")) == attempt
        assert json.loads((directory / "trial.json").read_text(encoding="utf-8")) == trial
        assert (folder / "events.jsonl").read_bytes() == raw.encode("utf-8")
        assert attempt["events_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert broker.is_sealed
        assert [event["request"]["op"] for event in broker.public_trace] == (
            ["metadata", "submit", "seal"] if valid else ["metadata", "harness_deadline_seal"])
        charges = [event.to_dict() for event in broker._session.trace
                   if event.to_dict()["action"] == "provider_usage"]
        assert len(charges) == 1
        assert {key: getattr(broker._session.usage, key) for key in usage} == usage
        assert broker._session.usage.proposals == int(valid)
        assert charges[0]["status"] == "succeeded" and charges[0]["charges"] == usage


def test_duplicate_provider_json_keys_preserve_recovery_accounting_and_public_identity(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from unittest.mock import Mock
    from rlcd.weight_space.live_provider import CodexProvider, parse_events, publish_attempt, run_brokered

    cli = tmp_path / "fake-codex"
    cli.write_bytes(b"offline transport placeholder; never executed")
    monkeypatch.setattr("rlcd.weight_space.live_provider.shutil.which", lambda _: str(cli))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="offline-test\n"))
    transport = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(returncode=0,
        communicate=lambda *args, **kwargs: (raw, "")))
    monkeypatch.setattr("rlcd.weight_space.live_provider.subprocess.Popen", transport)
    base = TensorArtifact("affine/1", {"W": [[1.]], "b": [0.]})
    task = TaskSpec("duplicate-json", "edit", "Reject ambiguous offline JSON",
                    base.architecture, base.sha256)
    response = {"actions": [json.dumps({"op": "submit", "commands": [], "intent": "Freeze baseline"}),
                            json.dumps({"op": "seal", "candidate_handle": None})],
                "reason": "offline fixture"}
    usage = {"input_tokens": 17, "output_tokens": 7}
    prefix = '{"type":"provider.extension","message":"/home/duplicate-fixture/log"}'
    completion = '{"type":"turn.completed","usage":{"input_tokens":17,"output_tokens":7}}'
    duplicate_action = '{"op":"metadata","op":"submit","commands":[],"intent":"Freeze baseline"}'
    action_response = response | {"actions": [duplicate_action, response["actions"][1]]}
    escaped_posix = r"\u002fhome\u002fduplicate-posix\u002fsecret"
    escaped_windows = r"C:\u005cUsers\u005cduplicate-windows\u005csecret"
    cases = [
        ("control", prefix, json.dumps(response), completion, []),
        ("usage", prefix, json.dumps(response),
         '{"type":"turn.completed","usage":{"input_tokens":999,"input_tokens":17,"output_tokens":7}}',
         ["invalid_event"]),
        ("event_type", '{"type":"item.started","type":"provider.extension",'
         '"message":"' + escaped_posix + '","item":{"type":"command_execution"}}',
         json.dumps(response), completion, ["invalid_event"]),
        ("item_type", '{"type":"item.started","item":{"type":"command_execution",'
         '"type":"reasoning","text":"' + escaped_windows + '"}}',
         json.dumps(response), completion, ["invalid_event"]),
        ("final", prefix, '{"actions":' + json.dumps(response["actions"])
         + ',"reason":"first","reason":"offline fixture"}', completion, ["invalid_final_response"]),
        ("action", prefix, json.dumps(action_response), completion, []),
    ]

    for name, first, final, last, errors in cases:
        lines = [first, json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": final}}, separators=(",", ":")), last]
        raw = "\n".join(lines) + "\n"
        transport.reset_mock()
        broker = ProposerBroker(task, base=base)
        charge = Mock(wraps=broker.record_usage)
        monkeypatch.setattr(broker, "record_usage", charge)
        directory = tmp_path / name
        trial = run_brokered(CodexProvider(), broker, prompt="Return the fixed offline actions",
                             directory=directory, max_rounds=3)
        completed = name in ("control", "action")
        valid = name == "control"
        expected_usage = {} if name == "usage" else usage
        attempt = trial["attempts"][0]
        assert transport.call_count == len(trial["attempts"]) == trial["assigned_trials"] == 1
        assert attempt["status"] == ("completed" if completed else "failed")
        assert attempt["error"] == (None if completed else "transport_or_response_failed")
        assert attempt["event_errors"] == errors and attempt["usage"] == expected_usage
        assert attempt["response"] == (None if name == "final" else action_response if name == "action" else response)
        assert attempt["unapproved_events"] == [] and attempt["diagnostic_items"] == 0
        assert attempt["cost"]["amount"] is None
        assert trial["status"] == ("sealed" if valid else "invalid" if name == "action" else "failed")
        assert broker.is_sealed
        assert [event["request"]["op"] for event in broker.public_trace] == (
            ["metadata", "submit", "seal"] if valid else ["metadata", "harness_deadline_seal"])
        charges = [event.to_dict() for event in broker._session.trace
                   if event.to_dict()["action"] == "provider_usage"]
        if name == "usage":
            charge.assert_not_called()
            assert charges == []
            assert broker._session.usage.input_tokens == broker._session.usage.output_tokens == 0
        else:
            charge.assert_called_once_with(17, 7)
            assert len(charges) == 1 and charges[0]["status"] == "succeeded"
            assert charges[0]["requested_charges"] == charges[0]["charges"] == usage
            assert {key: getattr(broker._session.usage, key) for key in usage} == usage
        assert broker._session.usage.proposals == int(valid)
        folder, public_folder = directory / "provider-0", tmp_path / (name + "-public")
        assert json.loads((folder / "receipt.json").read_text(encoding="utf-8")) == attempt
        assert json.loads((directory / "trial.json").read_text(encoding="utf-8")) == trial
        raw_bytes = raw.encode("utf-8")
        assert (folder / "events.jsonl").read_bytes() == raw_bytes
        assert attempt["events_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
        record = publish_attempt(folder, public_folder)
        public = (public_folder / "events.jsonl").read_bytes()
        expected_lines = [
            (json.dumps("public event omitted: duplicate JSON object key")
             if (name in ("event_type", "item_type") and index == 0)
             or (name == "usage" and index == 2) else json.dumps(json.loads(line), ensure_ascii=False))
            .replace("/home/duplicate-fixture", "[USER_PROFILE]")
            for index, line in enumerate(lines)]
        assert public == ("\n".join(expected_lines) + "\n").encode("utf-8")
        assert parse_events(public.decode("utf-8")) == parse_events(raw)
        if name in ("event_type", "item_type"):
            escaped = escaped_posix if name == "event_type" else escaped_windows
            assert escaped.encode("utf-8") in raw_bytes
            assert escaped.encode("utf-8") not in public
            assert b"duplicate-posix" not in public and b"duplicate-windows" not in public
            assert json.loads(public.splitlines()[0]) == "public event omitted: duplicate JSON object key"
            assert len(public.splitlines()) == len(lines)
            assert json.loads(public.splitlines()[1])["item"]["text"] == final
        assert (folder / "events.jsonl").read_bytes() == raw_bytes
        assert record["raw_events_sha256"] == attempt["events_sha256"]
        assert record["public_events_sha256"] == hashlib.sha256(public).hexdigest()
        assert record["raw_events_sha256"] != record["public_events_sha256"]
        assert record["redactions"] == (0 if name in ("event_type", "item_type") else 1)
        assert set(record) == {"schema", "raw_events_sha256", "public_events_sha256", "redactions",
                               "transform", "raw_bytes_published", "scope"}
        assert record["transform"] == "JSON_reserialization_and_user_profile_prefix_redaction"
        assert record["raw_bytes_published"] is False
        assert json.loads((public_folder / "event-publication.json").read_text(encoding="utf-8")) == record
        for filename in ("request.json", "response-schema.json", "receipt.json"):
            assert (public_folder / filename).read_bytes() == (folder / filename).read_bytes()


def test_public_event_derivative_retains_raw_identity_without_private_path(tmp_path):
    import hashlib
    from rlcd.weight_space.live_provider import publish_attempt
    source = tmp_path / "private"
    source.mkdir()
    for name in ("request.json", "response-schema.json", "receipt.json"):
        (source / name).write_text("{}\n")
    raw = (json.dumps({"item": {"message": r"warning at C:\Users\private-person\.codex\config.toml"}}) + "\n").encode()
    (source / "events.jsonl").write_bytes(raw)
    record = publish_attempt(source, tmp_path / "public")
    public = (tmp_path / "public" / "events.jsonl").read_bytes()
    assert b"private-person" not in public and b"[USER_PROFILE]" in public
    assert record["raw_events_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["public_events_sha256"] == hashlib.sha256(public).hexdigest()
    assert record["redactions"] == 1 and not record["raw_bytes_published"]


def test_public_event_derivative_handles_lone_surrogate_without_losing_replay_identity(tmp_path):
    import hashlib
    from rlcd.weight_space import live_provider

    custom_home = r"C:\Users\surrogate-fixture"
    # Redact decoded profile spellings while keeping lone surrogates escaped.
    surrogate_event = {"type": "diagnostic", "message": custom_home + r"\log", "note": "\ud800"}
    surrogate_line = json.dumps(surrogate_event).replace("Users", r"U\u0073ers")
    unicode_event = {"type": "diagnostic", "message": "caf\u00e9 \u03bb \U0001f680"}
    response = {"actions": [], "reason": "offline fixture"}
    usage = {"input_tokens": 17, "output_tokens": 7}
    lines = [surrogate_line, json.dumps(unicode_event, ensure_ascii=False),
             json.dumps({"type": "item.completed", "item": {
                 "type": "agent_message", "text": json.dumps(response)}}),
             json.dumps({"type": "turn.completed", "usage": usage})]
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    source, destination = tmp_path / "private", tmp_path / "public"
    source.mkdir()
    metadata = {name: (json.dumps({"synthetic": name}) + "\n").encode("utf-8")
                for name in ("request.json", "response-schema.json", "receipt.json")}
    for name, content in metadata.items():
        (source / name).write_bytes(content)
    (source / "events.jsonl").write_bytes(raw)

    record = live_provider.publish_attempt(source, destination)

    public = (destination / "events.jsonl").read_bytes()
    public_lines = public.decode("utf-8").splitlines()
    assert (source / "events.jsonl").read_bytes() == raw
    assert r"U\u0073ers" in lines[0] and json.loads(lines[0]) == surrogate_event
    assert public_lines[0] == json.dumps(surrogate_event | {"message": r"[USER_PROFILE]\log"})
    assert r"\ud800" in public_lines[0] and "surrogate-fixture" not in public_lines[0]
    assert json.loads(public_lines[1]) == unicode_event
    assert public_lines[2:] == lines[2:]
    assert live_provider.parse_events(public.decode("utf-8")) == live_provider.parse_events(raw.decode("utf-8")) == (
        response, usage, [], ["invalid_event"], [])
    for name, content in metadata.items():
        assert (source / name).read_bytes() == (destination / name).read_bytes() == content
    assert record == {"schema": "wsp-public-event-derivative/1",
                      "raw_events_sha256": hashlib.sha256(raw).hexdigest(),
                      "public_events_sha256": hashlib.sha256(public).hexdigest(),
                      "redactions": 1,
                      "transform": "JSON_reserialization_and_user_profile_prefix_redaction",
                      "raw_bytes_published": False,
                      "scope": "public_derivative_replay;_raw_identity_disclosed_not_publicly_replayed"}
    assert record["raw_events_sha256"] != record["public_events_sha256"]
    assert json.loads((destination / "event-publication.json").read_text(encoding="utf-8")) == record


@pytest.mark.parametrize("custom_home", [r"D:\SyntheticProfiles\synthetic-local-home", r"C:\Users\user name"])
def test_public_event_derivative_handles_decode_and_sanitation_recursion(tmp_path, monkeypatch, custom_home):
    import hashlib
    from rlcd.weight_space import live_provider

    decoder_depth = 100_000
    decoder_profile = r"C:\Users\example\.codex\config.toml"
    overnested = "[" * decoder_depth + json.dumps(decoder_profile) + "]" * decoder_depth
    # This fixture requires actual decoder recursion on the running interpreter.
    with pytest.raises(RecursionError):
        json.loads(overnested)

    sanitation_depth = 2 * sys.getrecursionlimit()
    nested = "synthetic-leaf"
    for _ in range(sanitation_depth):
        nested = [nested]
    sanitizer_event = {"message": r"C:\Users\private-person",
                       "custom_home": custom_home + r"\config.toml",
                       "quoted_posix": 'home is "/home/example"',
                       "nested": "synthetic-traversal"}
    sanitizer_line = json.dumps(sanitizer_event)
    shallow_event = {"item": {"message": "/home/username", "ok": True}}
    raw = ("\n".join((overnested, sanitizer_line, json.dumps(shallow_event))) + "\n").encode("utf-8")
    source, destination = tmp_path / "private", tmp_path / "public"
    source.mkdir()
    metadata = {name: (json.dumps({"synthetic": name}) + "\n").encode("utf-8")
                for name in ("request.json", "response-schema.json", "receipt.json")}
    for name, content in metadata.items():
        (source / name).write_bytes(content)
    (source / "events.jsonl").write_bytes(raw)

    original_loads, injections = json.loads, []

    def decode_with_deep_sanitation(value, *args, **kwargs):
        if value == sanitizer_line:
            injections.append(value)
            # The earlier message is redacted before traversal reaches this list.
            return {**sanitizer_event, "nested": nested}
        return original_loads(value, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(live_provider.json, "loads", decode_with_deep_sanitation)
        scoped.setattr(live_provider.Path, "home", lambda: live_provider.Path(custom_home))
        record = live_provider.publish_attempt(source, destination)

    expected_public = ("\n".join((
        "[" * decoder_depth + json.dumps(r"[USER_PROFILE]\.codex\config.toml") + "]" * decoder_depth,
        json.dumps({"message": "[USER_PROFILE]", "custom_home": r"[USER_PROFILE]\config.toml",
                    "quoted_posix": 'home is "[USER_PROFILE]"',
                    "nested": "synthetic-traversal"}),
        json.dumps({"item": {"message": "[USER_PROFILE]", "ok": True}}),
    )) + "\n").encode("utf-8")
    public = (destination / "events.jsonl").read_bytes()
    assert injections == [sanitizer_line]
    assert (source / "events.jsonl").read_bytes() == raw
    assert public == expected_public
    assert b"example" not in public
    assert b"private-person" not in public
    assert custom_home.rsplit("\\", 1)[-1].encode() not in public
    assert b"username" not in public
    assert json.loads(public.splitlines()[1]) == {"message": "[USER_PROFILE]",
                                                 "custom_home": r"[USER_PROFILE]\config.toml",
                                                 "quoted_posix": 'home is "[USER_PROFILE]"',
                                                 "nested": "synthetic-traversal"}
    assert json.loads(public.splitlines()[2]) == {"item": {"message": "[USER_PROFILE]", "ok": True}}
    for name, content in metadata.items():
        assert (source / name).read_bytes() == (destination / name).read_bytes() == content
    assert record == {"schema": "wsp-public-event-derivative/1",
                      "raw_events_sha256": hashlib.sha256(raw).hexdigest(),
                      "public_events_sha256": hashlib.sha256(expected_public).hexdigest(),
                      "redactions": 5,
                      "transform": "JSON_reserialization_and_user_profile_prefix_redaction",
                      "raw_bytes_published": False,
                      "scope": "public_derivative_replay;_raw_identity_disclosed_not_publicly_replayed"}
    assert json.loads((destination / "event-publication.json").read_text()) == record


def test_broker_registered_scoring_uses_materialized_data_and_keeps_public_route_closed(tmp_path):
    registry = DataRoleRegistry(tmp_path / "data-roles.sqlite3")
    batch = registry.register_batch([[2.], [3.]], source_id="synthetic-tickets", family_id="evaluation-family")
    base = TensorArtifact("affine/1", {"W": [[2.]], "b": [1.]})
    task = TaskSpec("support-routing", "edit", "Check a support routing readout", base.architecture, base.sha256)
    broker = ProposerBroker(task, base=base, data_registry=registry)
    assert broker.request({"op": "evaluate_registered", "batch": batch.to_dict()}) == {"status": "denied"}
    handle = broker.request({"op": "submit", "commands": [], "intent": "Freeze the baseline"})["candidate_handle"]
    broker.request({"op": "seal", "candidate_handle": handle})
    public_before = broker.public_trace
    scoring_calls = []
    def native_score(candidate, rows):
        import numpy as np
        scoring_calls.append(rows)
        a = candidate.artifact
        return {"scores": np.asarray(rows) @ a.tensors["W"].T + a.tensors["b"]}
    audit = broker.evaluate_registered(batch, native_score, evaluator_id="synthetic-support/1")
    assert audit["status"] == "succeeded" and scoring_calls == [[[2.], [3.]]]
    assert audit["evaluation"]["result"] == {"scores": [[5.], [7.]]}
    assert audit["evaluation"]["freshness"]["status"] == "registered_cooperative"
    assert audit["evaluation"]["freshness"]["provenance_authenticated"] is False
    assert audit["evaluation"]["freshness"]["semantic_independence_verified"] is False
    assert audit["evaluation"]["freshness"]["scientific_freshness_verified"] is False
    assert audit["evaluation"]["heldout_accounting"]["evaluator_calls"] == 1
    assert audit["evaluation"]["heldout_accounting"]["evaluated_examples"] == 2
    assert broker.public_trace == public_before
    assert batch.materialize() == [[2.], [3.]]
    with pytest.raises(ValueError):
        broker.evaluate_registered(batch, native_score, evaluator_id="retry/1")


def test_broker_malformed_registered_handle_has_private_rejection_receipt(tmp_path):
    base = TensorArtifact("affine/1", {"W": [[2.]], "b": [1.]})
    task = TaskSpec("routing", "edit", "Freeze the baseline", base.architecture, base.sha256)
    broker = ProposerBroker(task, base=base, data_registry=DataRoleRegistry(tmp_path / "data-roles.sqlite3"))
    handle = broker.request({"op": "submit", "commands": [], "intent": "Freeze the baseline"})["candidate_handle"]
    broker.request({"op": "seal", "candidate_handle": handle})
    public_before = broker.public_trace
    calls = []
    audit = broker.evaluate_registered(RegisteredBatch(), lambda c, rows: calls.append(rows),
                                       evaluator_id="malformed-handle/1")
    assert audit["evaluation"]["status"] == "rejected" and calls == []
    assert audit["evaluation"]["freshness"]["decision"]["reason"] == "invalid_registration"
    assert broker.public_trace == public_before
    with pytest.raises(ValueError):
        broker.evaluate_registered(RegisteredBatch(), lambda c, rows: {}, evaluator_id="retry/1")


@pytest.mark.parametrize("case", ["sealed", "failed_known", "failed_unknown", "denied"])
def test_saved_attempt_replay_reconciles_usage_and_trial(tmp_path, monkeypatch, case):
    import hashlib
    from scripts import run_broker_trial
    from rlcd.weight_space import live_provider
    from rlcd.weight_space.model import digest
    from rlcd.weight_space.workflow import verify_trace

    def forbidden_live_provider(*args, **kwargs):
        raise AssertionError("Saved-attempt replay must not instantiate a live provider")

    monkeypatch.setattr(run_broker_trial, "CodexProvider", forbidden_live_provider)
    monkeypatch.setattr(live_provider, "CodexProvider", forbidden_live_provider)

    def resign(document):
        document["sha256"] = digest({key: value for key, value in document.items() if key != "sha256"})
        return document

    class AuthoredProvider:
        tool_free = True

        def __init__(self):
            self.consumed = 0

        def call(self, prompt, *, directory, cwd, schema):
            round_number = self.consumed
            self.consumed += 1
            usage = ({"input_tokens": 7, "output_tokens": 2} if round_number == 0 else
                     {"input_tokens": 11, "output_tokens": 3})
            status, response = "failed", None
            if case == "failed_unknown":
                usage = {}
            elif case == "denied":
                usage = {"input_tokens": 100001, "output_tokens": 3}
            elif case == "sealed":
                status = "completed"
                if round_number == 0:
                    actions = [[], {"op": "seal", "candidate_handle": "unknown-authored-handle"},
                               {"op": "seal", "candidate_handle": []},
                               {"op": "submit", "commands": [
                                   {"op": "add", "tensor": "W", "values": [[1.0]]}],
                                "intent": "Authored additive edit"}]
                else:
                    assert round_number == 1
                    transcript = json.loads(prompt.split("\n\nJSON transcript:\n", 1)[1])
                    handle = transcript[-1]["data"]["candidate_handle"]
                    actions = [{"op": "seal", "candidate_handle": handle}, {"op": "metadata"}]
                response = {"actions": [json.dumps(action) for action in actions],
                            "reason": "Authored offline regression"}
            raw = (json.dumps({"type": "authored.saved_attempt", "round": round_number,
                               "usage": usage}) + "\n").encode("utf-8")
            configuration = {"provider": "authored-offline", "tool_free": True}
            receipt = resign({"schema": "wsp-live-provider-attempt/1", "status": status,
                              "response": response, "usage": usage,
                              "invocation": {"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                                             "schema_sha256": digest(schema), "configuration": configuration},
                              "events_sha256": hashlib.sha256(raw).hexdigest(),
                              "cost": {"amount": None, "currency": "USD", "status": "unknown"},
                              "unapproved_events": []})
            directory.mkdir(parents=True)
            live_provider.write_json(directory / "request.json", {"prompt": prompt, "configuration": configuration})
            live_provider.write_json(directory / "response-schema.json", schema)
            live_provider.write_json(directory / "receipt.json", receipt)
            (directory / "events.jsonl").write_bytes(raw)
            return receipt

    _, _, broker = run_broker_trial.fixture()
    private, public = tmp_path / "private", tmp_path / "public"
    provider = AuthoredProvider()
    trial = live_provider.run_brokered(provider, broker, prompt=run_broker_trial.PROMPT,
                                      directory=private, max_rounds=3)
    audit = broker.evaluate(run_broker_trial.evaluate, evaluator_id="synthetic-boundary/1")
    public.mkdir()
    live_provider.write_json(public / "trial.json", trial)
    live_provider.write_json(public / "audit.json", audit)
    for i in range(len(trial["attempts"])):
        live_provider.publish_attempt(private / f"provider-{i}", public / f"provider-{i}")

    charges = [event for event in audit["evaluation"]["trace"] if event["action"] == "provider_usage"]
    assert provider.consumed == len(trial["attempts"]) == (2 if case == "sealed" else 1)
    assert all(attempt["cost"]["amount"] is None for attempt in trial["attempts"])
    assert trial["status"] == ("sealed" if case == "sealed" else "failed")
    assert audit["status"] == ("succeeded" if case == "sealed" else "abstained")
    assert len(charges) == (0 if case == "failed_unknown" else 2 if case == "sealed" else 1)
    expected_usage = ({"input_tokens": 18, "output_tokens": 5} if case == "sealed" else
                      {"input_tokens": 7, "output_tokens": 2} if case == "failed_known" else
                      {"input_tokens": 0, "output_tokens": 0})
    assert {key: audit["evaluation"]["usage"][key] for key in expected_usage} == expected_usage
    if case == "denied":
        assert charges[0]["status"] == "denied" and charges[0]["charges"] == {}
        assert charges[0]["requested_charges"] == {"input_tokens": 100001, "output_tokens": 3}
    elif charges:
        assert all(event["status"] == "succeeded" and event["charges"] == event["requested_charges"]
                   for event in charges)
    assert run_broker_trial.replay(public)["status"] == "pass"

    # Missing, extra and noncontiguous folders must not change attempt selection.
    first, moved = public / "provider-0", tmp_path / "moved-provider"
    first.rename(moved)
    try:
        with pytest.raises(AssertionError, match="Provider attempt directories"):
            run_broker_trial.replay(public)
    finally:
        moved.rename(first)
    extra = public / f"provider-{len(trial['attempts'])}"
    extra.mkdir()
    try:
        with pytest.raises(AssertionError, match="Provider attempt directories"):
            run_broker_trial.replay(public)
    finally:
        extra.rmdir()
    first.rename(extra)
    try:
        with pytest.raises(AssertionError, match="Provider attempt directories"):
            run_broker_trial.replay(public)
    finally:
        extra.rename(first)

    def resign_chain(events, *, accounting=False):
        previous = None
        usage = {key: 0 for key in audit["evaluation"]["usage"]}
        for i, event in enumerate(events):
            event.update(sequence=i, previous_sha256=previous)
            if accounting:
                usage = {key: value + event["charges"].get(key, 0) for key, value in usage.items()}
                event["usage"] = usage.copy()
            previous = resign(event)["sha256"]
        return previous

    if case == "sealed":
        for mutation in ("response", "missing_handle"):
            tampered = json.loads(json.dumps(audit))
            if mutation == "response":
                tampered["public_trace"][1]["response"]["status"] = "ok"
                message = "Broker response changed"
            else:
                submission = next(event for event in tampered["public_trace"]
                                  if event["request"]["op"] == "submit")
                submission["response"].pop("candidate_handle")
                message = "Malformed saved broker event"
            resign_chain(tampered["public_trace"])
            live_provider.write_json(public / "audit.json", resign(tampered))
            with pytest.raises(AssertionError, match=message):
                run_broker_trial.replay(public)
        live_provider.write_json(public / "audit.json", audit)

        tampered_trial = json.loads(json.dumps(trial))
        tampered_trial["status"] = "invalid"
        live_provider.write_json(public / "trial.json", resign(tampered_trial))
        with pytest.raises(AssertionError, match="Brokered trial changed"):
            run_broker_trial.replay(public)
        tampered_trial["max_rounds"] = 2
        live_provider.write_json(public / "trial.json", resign(tampered_trial))
        with pytest.raises(AssertionError, match="Fixture round budget changed"):
            run_broker_trial.replay(public)
        live_provider.write_json(public / "trial.json", trial)

        # Re-sign every affected receipt/trial identity; exact prompt comparison
        # must still reject a saved request that the shared driver never made.
        request_path, receipt_path = first / "request.json", first / "receipt.json"
        original_request, original_receipt = request_path.read_bytes(), receipt_path.read_bytes()
        request = json.loads(original_request)
        request["prompt"] += "\nAuthored prompt mutation"
        receipt = json.loads(original_receipt)
        receipt["invocation"]["prompt_sha256"] = hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest()
        resign(receipt)
        live_provider.write_json(request_path, request)
        live_provider.write_json(receipt_path, receipt)
        with pytest.raises(AssertionError, match="Trial receipt changed"):
            run_broker_trial.replay(public)
        tampered_trial = json.loads(json.dumps(trial))
        tampered_trial["attempts"][0] = receipt
        live_provider.write_json(public / "trial.json", resign(tampered_trial))
        with pytest.raises(AssertionError, match="Provider prompt changed"):
            run_broker_trial.replay(public)
        request_path.write_bytes(original_request)
        receipt_path.write_bytes(original_receipt)
        live_provider.write_json(public / "trial.json", trial)

        schema_path = first / "response-schema.json"
        original_schema = schema_path.read_bytes()
        schema = json.loads(original_schema)
        schema["description"] = "Authored schema mutation"
        live_provider.write_json(schema_path, schema)
        with pytest.raises(AssertionError, match="Provider schema identity mismatch"):
            run_broker_trial.replay(public)
        receipt = json.loads(original_receipt)
        receipt["invocation"]["schema_sha256"] = digest(schema)
        resign(receipt)
        live_provider.write_json(receipt_path, receipt)
        tampered_trial = json.loads(json.dumps(trial))
        tampered_trial["attempts"][0] = receipt
        live_provider.write_json(public / "trial.json", resign(tampered_trial))
        with pytest.raises(AssertionError, match="Provider response schema changed"):
            run_broker_trial.replay(public)
        schema_path.write_bytes(original_schema)
        receipt_path.write_bytes(original_receipt)
        live_provider.write_json(public / "trial.json", trial)

    if case == "denied":
        for mutation in ("requested", "order", "final_usage"):
            tampered = json.loads(json.dumps(audit))
            evaluation = tampered["evaluation"]
            trace = evaluation["trace"]
            usage_index = next(i for i, event in enumerate(trace) if event["action"] == "provider_usage")
            if mutation == "requested":
                trace[usage_index]["requested_charges"]["input_tokens"] += 1
            elif mutation == "order":
                metadata_index = next(i for i, event in enumerate(trace) if event["action"] == "metadata")
                trace[metadata_index], trace[usage_index] = trace[usage_index], trace[metadata_index]
            else:
                evaluation["usage"]["input_tokens"] = 1
            evaluation["trace_head_sha256"] = resign_chain(trace, accounting=True)
            resign(evaluation)
            live_provider.write_json(public / "audit.json", resign(tampered))
            assert verify_trace(trace, expected_head=evaluation["trace_head_sha256"])
            with pytest.raises(AssertionError, match="Evaluation or provider accounting changed"):
                run_broker_trial.replay(public)
        live_provider.write_json(public / "audit.json", audit)
