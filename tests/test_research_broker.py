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
        assert len(charges) == int(valid)
        expected = {key: usage[key] if valid else 0 for key in ("input_tokens", "output_tokens")}
        assert {key: getattr(broker._session.usage, key) for key in expected} == expected
        assert broker._session.usage.proposals == int(valid)
        if valid:
            assert charges[0]["status"] == "succeeded" and charges[0]["charges"] == expected


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
            assert not any(event.to_dict()["action"] == "provider_usage" for event in broker._session.trace)
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
        assert len(charges) == int(completed)
        expected = {key: usage[key] if completed else 0 for key in ("input_tokens", "output_tokens")}
        assert {key: getattr(broker._session.usage, key) for key in expected} == expected
        assert broker._session.usage.proposals == int(valid)
        if completed:
            assert charges[0]["status"] == "succeeded" and charges[0]["charges"] == expected


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
