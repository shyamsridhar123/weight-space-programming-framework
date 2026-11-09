"""The returned JSON surface must not be an endpoint/hidden-evaluation oracle."""
import json

import pytest

from rlcd.weight_space.artifact import TensorArtifact
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
