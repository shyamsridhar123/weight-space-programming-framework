"""Fresh synthetic governance checks; no research, provider or model execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rlcd.weight_space import governance
from rlcd.weight_space.artifact import StructuralPolicy, TensorArtifact
from rlcd.weight_space.model import Network, digest
from rlcd.weight_space.program import Edit, Program, apply_program


CONTEXT_FIELDS = ("base", "program", "candidate", "constraints", "scope")


@pytest.fixture
def case():
    base = TensorArtifact("synthetic-governance/1", {"head.weight": [[1.0]]})
    program = Program(
        base.sha256,
        (Edit("head.weight", (0, 0), 1.0, 2.0),),
        "Synthetic local pilot routing change",
    )
    return {
        "base": base.to_dict(),
        "program": program.to_dict(),
        "candidate": apply_program(base, program).to_dict(),
        "constraints": {"max_change_abs": 1.0},
        "scope": {
            "environment": "local-pilot",
            "use_case": "support-routing",
            "target": "synthetic",
        },
        "purpose": "Review a synthetic routing change for a local pilot",
        "evidence": [{
            "name": "synthetic-check",
            "sha256": digest({"fixture": "new synthetic record"}),
        }],
        "requester": "alice",
        # Listing the requester must never permit self-approval or self-rejection.
        "approvers": ["alice", "bob"],
    }


def _create(case, **overrides):
    return governance.create_request(**(case | overrides))


def _pins(bundle):
    # Test callers retain these before attempting a decision or a hostile rewrite.
    return {
        "expected_request_sha256": bundle["request_sha256"],
        "expected_head_sha256": bundle["head_sha256"],
    }


def _context(case):
    return {name: case[name] for name in CONTEXT_FIELDS}


def _check(bundle, case, *, pins=None, **overrides):
    return governance.check_eligibility(
        bundle,
        **(_pins(bundle) if pins is None else pins),
        **(_context(case) | overrides),
    )


def _decide(bundle, action="approve", actor="bob", *, pins=None):
    return governance.decide(
        bundle,
        **(_pins(bundle) if pins is None else pins),
        actor=actor,
        action=action,
        reason="Synthetic review decision",
    )


def _assert_check(result, bundle, status):
    assert result == {
        "schema": "wsp-governance-check/1",
        "request_sha256": bundle["request_sha256"],
        "head_sha256": bundle["head_sha256"],
        "status": status,
        "local_pilot_eligible": status == "approved",
        "production_authorized": False,
        "scientific_authorized": False,
    }
    assert result["local_pilot_eligible"] is (status == "approved")
    assert result["production_authorized"] is False
    assert result["scientific_authorized"] is False


def _reseal(bundle, *, relink=False):
    """Recompute hostile records so tests exercise more than stale digests."""
    request_sha256 = digest(bundle["request"])
    bundle["request_sha256"] = request_sha256
    head = digest({
        "schema": "wsp-governance-genesis/1",
        "request_sha256": request_sha256,
    })
    for position, event in enumerate(bundle["events"], 1):
        if relink:
            event.update(
                request_sha256=request_sha256,
                previous_head_sha256=head,
                position=position,
            )
        event["sha256"] = digest({key: value for key, value in event.items() if key != "sha256"})
        head = event["sha256"]
    bundle["head_sha256"] = head
    return bundle


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


@pytest.mark.parametrize("revoker", ["alice", "bob"])
def test_request_approval_and_revocation_bind_exact_change_without_mutating_inputs(case, revoker):
    original = copy.deepcopy(case)
    requested = _create(case)
    expected_request = {
        "schema": "wsp-governance-request/1",
        **{name + "_sha256": digest(case[name]) for name in ("base", "program", "candidate", "constraints")},
        **{name: case[name] for name in ("scope", "purpose", "evidence", "requester", "approvers")},
    }
    assert requested == {
        "schema": "wsp-governance-bundle/1",
        "request": expected_request,
        "request_sha256": digest(expected_request),
        "events": [],
        "head_sha256": digest({
            "schema": "wsp-governance-genesis/1",
            "request_sha256": digest(expected_request),
        }),
        "production_authorized": False,
        "scientific_authorized": False,
    }
    _assert_check(_check(requested, case), requested, "requested")

    approved = _decide(requested)
    event = approved["events"][0]
    assert set(event) == {
        "schema", "request_sha256", "previous_head_sha256", "position",
        "actor", "action", "reason", "sha256",
    }
    assert event["schema"] == "wsp-governance-event/1"
    assert event["request_sha256"] == requested["request_sha256"]
    assert event["previous_head_sha256"] == requested["head_sha256"]
    assert type(event["position"]) is int and event["position"] == 1
    assert event["actor"] == "bob" and event["action"] == "approve"
    assert event["sha256"] == digest({key: value for key, value in event.items() if key != "sha256"})
    assert approved["head_sha256"] == event["sha256"]
    _assert_check(_check(approved, case), approved, "approved")

    revoked = _decide(approved, "revoke", revoker)
    _assert_check(_check(revoked, case), revoked, "revoked")
    assert revoked["events"][:-1] == approved["events"]
    assert revoked["events"][-1]["previous_head_sha256"] == approved["head_sha256"]
    assert revoked["events"][-1]["position"] == 2
    assert requested["events"] == []
    assert len(approved["events"]) == 1
    assert case == original
    for bundle in (requested, approved, revoked):
        assert bundle["production_authorized"] is False
        assert bundle["scientific_authorized"] is False

    # Returned histories and request metadata must not alias earlier records.
    revoked["request"]["scope"]["target"] = "changed-after-return"
    revoked["events"][0]["reason"] = "changed-after-return"
    assert approved["request"]["scope"] == original["scope"]
    assert approved["events"][0]["reason"] == "Synthetic review decision"


def test_request_owns_caller_metadata_and_canonical_constraints_identity(case):
    requested = _create(case, constraints={"b": [1, False], "a": None})
    assert requested["request"]["constraints_sha256"] == digest({"a": None, "b": [1, False]})
    scope_before = copy.deepcopy(case["scope"])
    evidence_before = copy.deepcopy(case["evidence"])
    case["scope"]["target"] = "mutated"
    case["evidence"][0]["name"] = "mutated"
    case["approvers"].append("carol")
    assert requested["request"]["scope"] == scope_before
    assert requested["request"]["evidence"] == evidence_before
    assert requested["request"]["approvers"] == ["alice", "bob"]


@pytest.mark.parametrize("constraints", [None, False, 0, 1.5, "opaque note", [], [1, {"x": True}]])
def test_constraints_may_be_any_strict_finite_json_value(case, constraints):
    requested = _create(case, constraints=constraints)
    assert requested["request"]["constraints_sha256"] == digest(constraints)
    _assert_check(_check(requested, case, constraints=constraints), requested, "requested")


def test_constraint_commitment_does_not_run_admission_or_inference(case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Governance must not execute a policy or model")

    monkeypatch.setattr(StructuralPolicy, "check", forbidden)
    monkeypatch.setattr(Network, "__call__", forbidden)
    constraints = {"max_edits": 0, "max_change_abs": 0.0}
    approved = _decide(_create(case, constraints=constraints))
    _assert_check(_check(approved, case, constraints=constraints), approved, "approved")


@pytest.mark.parametrize("actor,action", [
    ("alice", "approve"), ("alice", "reject"),
    ("mallory", "approve"), ("mallory", "reject"),
    ("bob", "revoke"), ("bob", "deploy"),
])
def test_pending_decisions_require_another_configured_approver(case, actor, action):
    requested = _create(case)
    before = copy.deepcopy(requested)
    with pytest.raises(ValueError):
        _decide(requested, action, actor)
    assert requested == before


@pytest.mark.parametrize("terminal", ["rejected", "revoked"])
@pytest.mark.parametrize("action", ["approve", "reject", "revoke"])
def test_rejected_and_revoked_histories_are_terminal(case, terminal, action):
    requested = _create(case)
    bundle = (_decide(requested, "reject") if terminal == "rejected"
              else _decide(_decide(requested), "revoke", "alice"))
    _assert_check(_check(bundle, case), bundle, terminal)
    with pytest.raises(ValueError):
        _decide(bundle, action)


@pytest.mark.parametrize("actor,action", [("bob", "approve"), ("bob", "reject"), ("mallory", "revoke")])
def test_approved_state_allows_only_an_authorized_revocation(case, actor, action):
    approved = _decide(_create(case))
    with pytest.raises(ValueError):
        _decide(approved, action, actor)


@pytest.mark.parametrize("defect", ["stale-parent", "wrong-before", "outside-coordinate", "wrong-candidate"])
def test_creation_recomputes_structural_application(case, defect):
    altered = copy.deepcopy(case)
    if defect == "stale-parent":
        altered["program"]["base_sha256"] = "0" * 64
    elif defect == "wrong-before":
        altered["program"]["edits"][0]["before"] = -1.0
    elif defect == "outside-coordinate":
        altered["program"]["edits"][0]["index"] = [1, 0]
    else:
        altered["candidate"]["tensors"]["head.weight"] = [[3.0]]
    with pytest.raises(ValueError):
        _create(altered)


@pytest.mark.parametrize("substitution", ["program", "candidate", "constraints", "scope", "coherent-parent"])
def test_check_requires_the_exact_reviewed_context(case, substitution):
    approved = _decide(_create(case))
    changed = copy.deepcopy(case)
    if substitution == "program":
        changed["program"]["intent"] = "A different intent with the same resulting tensor"
    elif substitution == "candidate":
        changed["candidate"]["tensors"]["head.weight"] = [[3.0]]
    elif substitution == "constraints":
        changed["constraints"] = {"max_change_abs": 2.0}
    elif substitution == "scope":
        changed["scope"]["target"] = "another-synthetic-target"
    else:
        base = TensorArtifact("synthetic-governance/1", {"head.weight": [[3.0]]})
        program = Program(base.sha256, (Edit("head.weight", (0, 0), 3.0, 4.0),), "Another synthetic change")
        changed.update(base=base.to_dict(), program=program.to_dict(), candidate=apply_program(base, program).to_dict())
    with pytest.raises(ValueError):
        _check(approved, changed)


@pytest.mark.parametrize("pin", ["expected_request_sha256", "expected_head_sha256"])
def test_both_retained_pins_reject_substitution_and_stale_decisions(case, pin):
    requested = _create(case)
    pins = _pins(requested)
    pins[pin] = "0" * 64
    with pytest.raises(ValueError):
        _check(requested, case, pins=pins)
    with pytest.raises(ValueError):
        _decide(requested, pins=pins)

    approved = _decide(requested)
    with pytest.raises(ValueError):
        _decide(approved, "revoke", "alice", pins=_pins(requested))


def test_rehashed_request_rewrite_cannot_replace_independently_retained_request_pin(case):
    approved = _decide(_create(case))
    retained = _pins(approved)
    rewritten = copy.deepcopy(approved)
    rewritten["request"]["purpose"] = "An unreviewed rewritten purpose"
    _reseal(rewritten, relink=True)
    assert rewritten["request_sha256"] != retained["expected_request_sha256"]
    # Supply the attacker's internally coherent new head to isolate the request pin.
    pins = retained | {"expected_head_sha256": rewritten["head_sha256"]}
    with pytest.raises(ValueError):
        _check(rewritten, case, pins=pins)
    with pytest.raises(ValueError):
        _decide(rewritten, "revoke", "alice", pins=pins)


@pytest.mark.parametrize("remaining_events", [0, 1])
def test_removing_revocation_cannot_replace_independently_retained_current_head(case, remaining_events):
    requested = _create(case)
    revoked = _decide(_decide(requested), "revoke", "alice")
    retained = _pins(revoked)
    truncated = copy.deepcopy(revoked)
    truncated["events"] = truncated["events"][:remaining_events]
    _reseal(truncated, relink=True)
    assert truncated["request_sha256"] == retained["expected_request_sha256"]
    assert truncated["head_sha256"] != retained["expected_head_sha256"]
    with pytest.raises(ValueError):
        _check(truncated, case, pins=retained)
    with pytest.raises(ValueError):
        _decide(truncated, "approve", pins=retained)


@pytest.mark.parametrize("field,value", [
    ("schema", "wsp-governance-event/999"),
    ("request_sha256", "a" * 64),
    ("previous_head_sha256", "a" * 64),
    ("position", True), ("position", 1.0), ("position", 2),
    ("actor", "alice"), ("actor", "mallory"), ("actor", "Bob"),
    ("action", "revoke"), ("action", "deploy"),
    ("reason", ""), ("reason", " \n"),
    ("unexpected", "field"),
])
def test_complete_replay_rejects_resealed_event_defects(case, field, value):
    malformed = _decide(_create(case))
    malformed["events"][0][field] = value
    _reseal(malformed)
    with pytest.raises(ValueError):
        _check(malformed, case)
    with pytest.raises(ValueError):
        _decide(malformed, "revoke", "alice")


def test_event_content_is_rehashed_even_when_retained_head_field_is_unchanged(case):
    approved = _decide(_create(case))
    retained = _pins(approved)
    altered = copy.deepcopy(approved)
    altered["events"][0]["reason"] = "Changed without updating any digest"
    with pytest.raises(ValueError):
        _check(altered, case, pins=retained)


@pytest.mark.parametrize("attack", ["reorder", "duplicate", "after-rejection", "after-revocation"])
def test_transition_replay_rejects_relinked_reordered_duplicate_and_terminal_events(case, attack):
    requested = _create(case)
    approved = _decide(requested)
    if attack == "reorder":
        malformed = _decide(approved, "revoke", "alice")
        malformed["events"].reverse()
    elif attack == "duplicate":
        malformed = copy.deepcopy(approved)
        malformed["events"].append(copy.deepcopy(approved["events"][0]))
    elif attack == "after-rejection":
        malformed = _decide(requested, "reject")
        malformed["events"].append(copy.deepcopy(approved["events"][0]))
    else:
        malformed = _decide(approved, "revoke", "alice")
        malformed["events"].append(copy.deepcopy(approved["events"][0]))
    _reseal(malformed, relink=True)
    with pytest.raises(ValueError):
        _check(malformed, case)


@pytest.mark.parametrize("layer", ["bundle", "request", "event"])
@pytest.mark.parametrize("defect", ["unknown-field", "missing-field", "schema"])
def test_consuming_api_rejects_unknown_missing_or_wrong_envelope_fields(case, layer, defect):
    bundle = _decide(_create(case))
    target = bundle if layer == "bundle" else bundle["request"] if layer == "request" else bundle["events"][0]
    if defect == "unknown-field":
        target["unexpected"] = True
    elif defect == "missing-field":
        del target[{"bundle": "production_authorized", "request": "purpose", "event": "reason"}[layer]]
    else:
        target["schema"] = "unsupported/1"
    _reseal(bundle, relink=True)
    with pytest.raises(ValueError):
        _check(bundle, case)


@pytest.mark.parametrize("flag", ["production_authorized", "scientific_authorized"])
@pytest.mark.parametrize("value", [0, 1, True, None, "false"])
def test_authority_flags_must_remain_literal_false(case, flag, value):
    bundle = _create(case)
    bundle[flag] = value
    with pytest.raises(ValueError):
        _check(bundle, case)


@pytest.mark.parametrize("field,value", [
    ("purpose", ""), ("purpose", " \n"),
    ("requester", "Alice"), ("requester", " alice"), ("requester", "alice "),
    ("requester", "1alice"), ("requester", "álîce"), ("requester", "a" * 65),
    ("approvers", []), ("approvers", ["bob", "bob"]), ("approvers", ["Bob"]),
    ("scope", {"environment": "production", "use_case": "support", "target": "synthetic"}),
    ("scope", {"environment": "local-pilot", "use_case": " ", "target": "synthetic"}),
    ("scope", {"environment": "local-pilot", "use_case": "support", "target": ""}),
    ("scope", {"environment": "local-pilot", "use_case": "support"}),
    ("scope", {"environment": "local-pilot", "use_case": "support", "target": "synthetic", "extra": True}),
    ("evidence", []),
    ("evidence", [{"name": " ", "sha256": "a" * 64}]),
    ("evidence", [{"name": "check", "sha256": "A" * 64}]),
    ("evidence", [{"name": "check", "sha256": "a" * 63}]),
    ("evidence", [{"name": "check"}]),
    ("evidence", [{"name": "check", "sha256": "a" * 64, "extra": True}]),
    ("evidence", [{"name": "check", "sha256": "a" * 64}, {"name": "check", "sha256": "b" * 64}]),
])
def test_request_requires_explicit_local_scope_canonical_actors_and_named_evidence(case, field, value):
    with pytest.raises(ValueError):
        _create(case, **{field: value})


def _forbidden_hook(*args, **kwargs):
    raise AssertionError("A user-supplied conversion or container hook was invoked")


class _ProtocolTrap:
    to_dict = _forbidden_hook
    __array__ = _forbidden_hook
    __float__ = _forbidden_hook
    __index__ = _forbidden_hook
    __iter__ = _forbidden_hook
    __json__ = _forbidden_hook


class _DictSubclass(dict):
    items = _forbidden_hook
    __iter__ = _forbidden_hook


class _ListSubclass(list):
    __iter__ = _forbidden_hook


class _StringSubclass(str):
    __str__ = _forbidden_hook


class _IntegerSubclass(int):
    __int__ = _forbidden_hook


class _FloatSubclass(float):
    __float__ = _forbidden_hook


@pytest.mark.parametrize("value", [
    pytest.param(_ProtocolTrap(), id="protocol-object"),
    pytest.param(_DictSubclass(x=1), id="dict-subclass"),
    pytest.param(_ListSubclass([1]), id="list-subclass"),
    pytest.param(_StringSubclass("value"), id="str-subclass"),
    pytest.param(_IntegerSubclass(1), id="int-subclass"),
    pytest.param(_FloatSubclass(1.0), id="float-subclass"),
    pytest.param((1, 2), id="tuple"),
    pytest.param({1, 2}, id="set"),
    pytest.param(b"value", id="bytes"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param({1: "non-string key"}, id="non-string-key"),
    pytest.param({_StringSubclass("key"): 1}, id="str-subclass-key"),
])
def test_nested_json_boundary_rejects_nonbuiltin_values_without_conversion_hooks(case, value):
    with pytest.raises(ValueError):
        _create(case, constraints={"nested": [value]})


@pytest.mark.parametrize("container", ["list", "dict"])
def test_json_boundary_rejects_cycles(case, container):
    value = [] if container == "list" else {}
    if container == "list":
        value.append(value)
    else:
        value["cycle"] = value
    with pytest.raises(ValueError):
        _create(case, constraints=value)


@pytest.mark.parametrize("field", [*CONTEXT_FIELDS, "purpose", "evidence", "requester", "approvers"])
def test_creation_preflights_every_supplied_value_before_framework_deserialization(case, monkeypatch, field):
    def forbidden_decoder(*args, **kwargs):
        raise AssertionError("A framework deserializer ran before strict JSON preflight")

    monkeypatch.setattr(TensorArtifact, "from_dict", forbidden_decoder)
    monkeypatch.setattr(Network, "from_dict", forbidden_decoder)
    monkeypatch.setattr(Program, "from_dict", forbidden_decoder)
    with pytest.raises(ValueError):
        _create(case, **{field: _ProtocolTrap()})


@pytest.mark.parametrize("field", ["bundle", *CONTEXT_FIELDS])
def test_check_preflights_every_supplied_value_before_framework_deserialization(case, monkeypatch, field):
    approved = _decide(_create(case))
    retained = _pins(approved)

    def forbidden_decoder(*args, **kwargs):
        raise AssertionError("A framework deserializer ran before strict JSON preflight")

    monkeypatch.setattr(TensorArtifact, "from_dict", forbidden_decoder)
    monkeypatch.setattr(Network, "from_dict", forbidden_decoder)
    monkeypatch.setattr(Program, "from_dict", forbidden_decoder)
    params = _context(case)
    supplied_bundle = _ProtocolTrap() if field == "bundle" else approved
    if field != "bundle":
        params[field] = _ProtocolTrap()
    with pytest.raises(ValueError):
        governance.check_eligibility(supplied_bundle, **retained, **params)


@pytest.mark.parametrize("field", ["actor", "action", "reason", "expected_request_sha256", "expected_head_sha256"])
def test_decision_rejects_non_json_values_without_protocol_hooks(case, field):
    requested = _create(case)
    params = _pins(requested) | {"actor": "bob", "action": "approve", "reason": "Synthetic review"}
    params[field] = _ProtocolTrap()
    with pytest.raises(ValueError):
        governance.decide(requested, **params)


@pytest.mark.parametrize("text", [
    '{"x": 1, "x": 2}',
    '{"outer": [{"x": 1, "x": 2}]}',
    "NaN", "Infinity", "-Infinity", "1e999", '{"nested": [1e999]}',
])
def test_strict_file_reader_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path, text):
    path = tmp_path / "input.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        governance.read_json(path)


def test_strict_file_reader_is_generic_and_envelope_validation_belongs_to_consumer(case, tmp_path):
    path = tmp_path / "input.json"
    value = {"arbitrary": [None, True, 1, 1.5, "text", {}]}
    _write_json(path, value)
    assert governance.read_json(path) == value

    bundle = _create(case)
    bundle["unknown-envelope-field"] = True
    _write_json(path, bundle)
    loaded = governance.read_json(path)
    assert loaded == bundle
    with pytest.raises(ValueError):
        _check(loaded, case)


def test_export_preserves_original_formatted_bytes_and_creates_exclusively(case, tmp_path):
    bundle = _decide(_create(case))
    source = tmp_path / "source.json"
    content = (json.dumps(bundle, indent=3, ensure_ascii=False).replace("\n", "\r\n") + "\r\n").encode("utf-8")
    source.write_bytes(content)
    destination = tmp_path / "export" / "bundle.json"
    assert governance.export_bundle(source, destination, **_pins(bundle)) == bundle
    assert destination.read_bytes() == content
    destination.write_bytes(b"existing destination must survive")
    with pytest.raises(FileExistsError):
        governance.export_bundle(source, destination, **_pins(bundle))
    assert destination.read_bytes() == b"existing destination must survive"


@pytest.mark.parametrize("pin", ["expected_request_sha256", "expected_head_sha256"])
def test_export_invalid_pins_create_no_destination(case, tmp_path, pin):
    bundle = _create(case)
    source = tmp_path / "source.json"
    _write_json(source, bundle)
    pins = _pins(bundle) | {pin: "0" * 64}
    destination = tmp_path / "absent" / "bundle.json"
    with pytest.raises(ValueError):
        governance.export_bundle(source, destination, **pins)
    assert not destination.exists()


def test_export_replays_entire_history_before_creating_destination(case, tmp_path):
    malformed = _decide(_create(case))
    malformed["events"][0]["actor"] = "alice"
    _reseal(malformed, relink=True)
    source, destination = tmp_path / "source.json", tmp_path / "out.json"
    _write_json(source, malformed)
    with pytest.raises(ValueError):
        governance.export_bundle(source, destination, **_pins(malformed))
    assert not destination.exists()


@pytest.mark.parametrize("operation", ["decide", "check", "export"])
@pytest.mark.parametrize("missing", ["expected_request_sha256", "expected_head_sha256"])
def test_public_operations_require_both_explicit_pins(case, tmp_path, operation, missing):
    bundle = _create(case)
    pins = _pins(bundle)
    del pins[missing]
    source = tmp_path / "source.json"
    _write_json(source, bundle)
    with pytest.raises(TypeError):
        if operation == "decide":
            governance.decide(bundle, **pins, actor="bob", action="approve", reason="Synthetic review")
        elif operation == "check":
            governance.check_eligibility(bundle, **pins, **_context(case))
        else:
            governance.export_bundle(source, tmp_path / "out.json", **pins)


def _cli_json(capsys, argv):
    from rlcd.weight_space.__main__ import main

    assert main(argv) == 0
    output = capsys.readouterr()
    assert output.err == ""
    return json.loads(output.out)


def _cli_files(case, tmp_path):
    files = {name: tmp_path / (name + ".json") for name in (*CONTEXT_FIELDS, "evidence")}
    for name, path in files.items():
        _write_json(path, case[name])
    return files


def _cli_context(files):
    return [str(files[name]) for name in ("base", "program", "candidate")] + [
        "--constraints", str(files["constraints"]), "--scope", str(files["scope"]),
    ]


def _cli_pins(bundle):
    return [
        "--expected-request-sha256", bundle["request_sha256"],
        "--expected-head-sha256", bundle["head_sha256"],
    ]


def test_cli_request_check_decide_revoke_and_export_lifecycle(case, tmp_path, capsys):
    files = _cli_files(case, tmp_path)
    requested_path = tmp_path / "requested.json"
    requested = _cli_json(capsys, [
        "governance", "request", *_cli_context(files),
        "--purpose", case["purpose"], "--evidence", str(files["evidence"]),
        "--requester", "alice", "--approver", "alice", "--approver", "bob",
        "--out", str(requested_path),
    ])
    assert governance.read_json(requested_path) == requested
    _assert_check(_cli_json(capsys, [
        "governance", "check", str(requested_path), *_cli_context(files), *_cli_pins(requested),
    ]), requested, "requested")

    approved_path = tmp_path / "approved.json"
    approved = _cli_json(capsys, [
        "governance", "decide", str(requested_path), "--actor", "bob", "--action", "approve",
        "--reason", "Synthetic approval", "--out", str(approved_path), *_cli_pins(requested),
    ])
    _assert_check(_cli_json(capsys, [
        "governance", "check", str(approved_path), *_cli_context(files), *_cli_pins(approved),
    ]), approved, "approved")

    revoked_path = tmp_path / "revoked.json"
    revoked = _cli_json(capsys, [
        "governance", "decide", str(approved_path), "--actor", "alice", "--action", "revoke",
        "--reason", "Synthetic withdrawal", "--out", str(revoked_path), *_cli_pins(approved),
    ])
    _assert_check(_cli_json(capsys, [
        "governance", "check", str(revoked_path), *_cli_context(files), *_cli_pins(revoked),
    ]), revoked, "revoked")
    exported_path = tmp_path / "exported.json"
    assert _cli_json(capsys, [
        "governance", "export", str(revoked_path), "--out", str(exported_path), *_cli_pins(revoked),
    ]) == revoked
    assert exported_path.read_bytes() == revoked_path.read_bytes()


def test_cli_strict_json_failure_exits_two_without_creating_request(case, tmp_path, capsys):
    from rlcd.weight_space.__main__ import main

    files = _cli_files(case, tmp_path)
    files["scope"].write_text(
        '{"environment":"local-pilot","use_case":"support","target":"one","target":"two"}',
        encoding="utf-8",
    )
    destination = tmp_path / "request.json"
    with pytest.raises(SystemExit) as error:
        main([
            "governance", "request", *_cli_context(files),
            "--purpose", case["purpose"], "--evidence", str(files["evidence"]),
            "--requester", "alice", "--approver", "bob", "--out", str(destination),
        ])
    assert error.value.code == 2
    assert not destination.exists()
    assert "weight-space:" in capsys.readouterr().err


@pytest.mark.parametrize("operation", ["decide", "check", "export"])
def test_cli_stale_head_exits_two_without_creating_output(case, tmp_path, capsys, operation):
    from rlcd.weight_space.__main__ import main

    files = _cli_files(case, tmp_path)
    bundle = _decide(_create(case))
    source, destination = tmp_path / "bundle.json", tmp_path / "out.json"
    _write_json(source, bundle)
    pins = ["--expected-request-sha256", bundle["request_sha256"], "--expected-head-sha256", "0" * 64]
    argv = ["governance", operation, str(source), *pins]
    if operation == "decide":
        argv += ["--actor", "alice", "--action", "revoke", "--reason", "Synthetic withdrawal", "--out", str(destination)]
    elif operation == "check":
        argv += _cli_context(files)
    else:
        argv += ["--out", str(destination)]
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    assert not destination.exists()
    assert "weight-space:" in capsys.readouterr().err
