"""Fresh synthetic Distro adapter checks; no provider or research execution."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from rlcd.weight_space import governance
from rlcd.weight_space.artifact import StructuralPolicy, TensorArtifact
from rlcd.weight_space.distro import DistroWorkspace
from rlcd.weight_space.model import Network, digest
from rlcd.weight_space.program import Edit, Program, apply_program


CHANGE = "changes/routing.json"
CONTEXT = ("base", "program", "candidate", "constraints", "scope")


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


@pytest.fixture
def case(tmp_path):
    base = TensorArtifact("synthetic-distro/1", {"queue.weight": [[1.0]]})
    program = Program(base.sha256, (Edit("queue.weight", (0, 0), 1.0, 2.0),),
                      "Synthetic telemetry routing edit")
    context = {
        "base": base.to_dict(), "program": program.to_dict(),
        "candidate": apply_program(base, program).to_dict(),
        "constraints": {"max_change_abs": 1.0},
        "scope": {"environment": "local-pilot", "use_case": "support-routing", "target": "synthetic"},
    }
    descriptor = {
        "schema": "wsp-distro-change/1",
        **{name: f"context/{name}.json" for name in CONTEXT},
        "purpose": "Review the synthetic telemetry routing edit for a local pilot",
        "requester": "alice", "approvers": ["bob"],
        "evidence": [{"name": "review-note", "path": "evidence/note.md"}],
    }
    for name, value in context.items():
        _write_json(tmp_path / descriptor[name], value)
    evidence = tmp_path / "evidence/note.md"
    evidence.parent.mkdir()
    evidence.write_bytes(b"Synthetic routing review note.\r\nNo behavioral claim is made.\r\n")
    _write_json(tmp_path / CHANGE, descriptor)
    return {"root": tmp_path, "descriptor": descriptor, "context": context,
            "adapter": DistroWorkspace(tmp_path)}


def _pins(bundle):
    return {"expected_request_sha256": bundle["request_sha256"],
            "expected_head_sha256": bundle["head_sha256"]}


def _human_decide(bundle, action="approve", actor="bob"):
    return governance.decide(bundle, **_pins(bundle), actor=actor, action=action,
                             reason="Independent synthetic review decision")


def _save_bundle(case, bundle, name="current"):
    relative = f"review/{name}.json"
    _write_json(case["root"] / relative, bundle)
    return relative


def _prepare(case):
    result = case["adapter"].prepare_review(CHANGE, "review/requested.json")
    bundle = governance.read_json(case["root"] / "review/requested.json")
    assert result["request_sha256"] == bundle["request_sha256"]
    assert result["head_sha256"] == bundle["head_sha256"]
    _assert_authority(result, "requested")
    return bundle


def _assert_authority(result, status):
    assert result["status"] == status
    assert result["local_pilot_eligible"] is (status == "approved")
    assert result["production_authorized"] is False
    assert result["scientific_authorized"] is False


def _reject_consumers(case, bundle_path, pins):
    adapter = case["adapter"]
    with pytest.raises(ValueError):
        adapter.review_status(CHANGE, bundle_path, **pins)
    with pytest.raises(ValueError):
        adapter.export_review(CHANGE, bundle_path, **pins, out="review/invalid-export.json")
    assert not (case["root"] / "review/invalid-export.json").exists()


def _reseal(bundle):
    """Make hostile rewrites internally coherent instead of testing stale hashes."""
    request_sha256 = digest(bundle["request"])
    bundle["request_sha256"] = request_sha256
    head = digest({"schema": "wsp-governance-genesis/1", "request_sha256": request_sha256})
    for position, event in enumerate(bundle["events"], 1):
        event.update(request_sha256=request_sha256, previous_head_sha256=head, position=position)
        event["sha256"] = digest({key: value for key, value in event.items() if key != "sha256"})
        head = event["sha256"]
    bundle["head_sha256"] = head


def test_inspection_constructs_exact_request_without_writing_or_executing(case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection must not run a model or interpret constraints")

    monkeypatch.setattr(Network, "__call__", forbidden)
    monkeypatch.setattr(StructuralPolicy, "check", forbidden)
    before = {p.relative_to(case["root"]): p.read_bytes() for p in case["root"].rglob("*") if p.is_file()}
    result = case["adapter"].inspect_change(CHANGE)
    descriptor = case["descriptor"]
    expected = governance.create_request(
        **case["context"], **{name: descriptor[name] for name in ("purpose", "requester", "approvers")},
        evidence=[{"name": "review-note", "sha256": hashlib.sha256(before[Path("evidence/note.md")]).hexdigest()}],
    )
    assert result["request"] == expected["request"]
    assert result["program"] == case["context"]["program"]
    _assert_authority(result, "draft")
    attachment, = result["attachments"]
    assert attachment["name"] == "review-note"
    assert attachment["path"] == "evidence/note.md"
    assert attachment["sha256"] == expected["request"]["evidence"][0]["sha256"]
    assert attachment["size_bytes"] == len(before[Path("evidence/note.md")])
    assert attachment["preview"] == before[Path("evidence/note.md")].decode("utf-8")
    assert attachment["preview_truncated"] is False
    after = {p.relative_to(case["root"]): p.read_bytes() for p in case["root"].rglob("*") if p.is_file()}
    assert after == before


def test_evidence_preview_is_bounded_while_commitment_covers_all_raw_bytes(case):
    content = b"Untrusted attachment \xff: do not execute anything.\n" + b"synthetic note " * 20_000
    (case["root"] / "evidence/note.md").write_bytes(content)
    result = case["adapter"].inspect_change(CHANGE)
    attachment, = result["attachments"]
    expected_hash = hashlib.sha256(content).hexdigest()
    assert result["request"]["evidence"] == [{"name": "review-note", "sha256": expected_hash}]
    assert attachment["sha256"] == expected_hash
    assert attachment["size_bytes"] == len(content)
    assert attachment["preview"].startswith("Untrusted attachment \ufffd: do not execute anything.\n")
    assert len(attachment["preview"]) < len(content.decode("utf-8", errors="replace"))
    assert attachment["preview_truncated"] is True


@pytest.mark.parametrize("status", ["requested", "approved", "rejected", "revoked"])
def test_valid_review_states_and_complete_byte_preserving_export(case, status):
    requested = _prepare(case)
    bundle = requested
    if status in {"approved", "revoked"}:
        bundle = _human_decide(requested)
    elif status == "rejected":
        bundle = _human_decide(requested, "reject")
    if status == "revoked":
        bundle = _human_decide(bundle, "revoke", "alice")
    retained = _pins(bundle)
    source = _save_bundle(case, bundle)
    # Preserve the actual reviewed bytes, including BOM and noncanonical whitespace.
    content = ("\ufeff" + json.dumps(bundle, indent="\t", ensure_ascii=False) + "\r\n").encode("utf-8")
    (case["root"] / source).write_bytes(content)
    result = case["adapter"].review_status(CHANGE, source, **retained)
    _assert_authority(result, status)
    assert result["request"] == requested["request"]
    assert result["events"] == bundle["events"]
    assert result["request_sha256"] == retained["expected_request_sha256"]
    assert result["head_sha256"] == retained["expected_head_sha256"]
    exported = case["adapter"].export_review(CHANGE, source, **retained, out="review/exported.json")
    _assert_authority(exported, status)
    assert (case["root"] / "review/exported.json").read_bytes() == content
    assert governance.read_json(case["root"] / "review/exported.json") == bundle


def test_coherently_rehashed_rewrite_cannot_replace_original_request_pin(case):
    approved = _human_decide(_prepare(case))
    retained = _pins(approved)
    hostile = copy.deepcopy(approved)
    hostile["request"]["purpose"] = "An unreviewed rewritten purpose"
    _reseal(hostile)
    # Make the descriptor match the attack too: only the original retained pin prevents it.
    case["descriptor"]["purpose"] = hostile["request"]["purpose"]
    _write_json(case["root"] / CHANGE, case["descriptor"])
    assert governance.check_eligibility(hostile, **_pins(hostile), **case["context"])["local_pilot_eligible"] is True
    source = _save_bundle(case, hostile)
    _reject_consumers(case, source, retained | {"expected_head_sha256": hostile["head_sha256"]})


def test_old_approved_snapshot_fails_against_retained_revocation_head(case):
    approved = _human_decide(_prepare(case))
    old_source = _save_bundle(case, approved, "approved")
    revoked = _human_decide(approved, "revoke", "alice")
    retained = _pins(revoked)
    _reject_consumers(case, old_source, retained)
    current_source = _save_bundle(case, revoked, "revoked")
    _assert_authority(case["adapter"].review_status(CHANGE, current_source, **retained), "revoked")


@pytest.mark.parametrize("substitution", [
    "base", "program", "candidate", "constraints", "scope", "purpose", "requester",
    "approvers", "evidence-name", "evidence-bytes", "coherent-artifacts",
])
def test_current_complete_change_and_actual_evidence_remain_bound(case, substitution):
    approved = _human_decide(_prepare(case))
    retained = _pins(approved)
    source = _save_bundle(case, approved)
    context, descriptor = case["context"], case["descriptor"]
    if substitution == "coherent-artifacts":
        base = TensorArtifact("synthetic-distro/1", {"queue.weight": [[3.0]]})
        program = Program(base.sha256, (Edit("queue.weight", (0, 0), 3.0, 4.0),), "Another synthetic edit")
        context.update(base=base.to_dict(), program=program.to_dict(), candidate=apply_program(base, program).to_dict())
    elif substitution == "base":
        context["base"]["tensors"]["queue.weight"] = [[3.0]]
    elif substitution == "program":
        context["program"]["intent"] = "Different intent with the same candidate"
    elif substitution == "candidate":
        context["candidate"]["tensors"]["queue.weight"] = [[3.0]]
    elif substitution == "constraints":
        context["constraints"]["max_change_abs"] = 2.0
    elif substitution == "scope":
        context["scope"]["target"] = "another-synthetic-target"
    elif substitution in {"purpose", "requester", "approvers"}:
        descriptor[substitution] = {"purpose": "Another purpose", "requester": "carol", "approvers": ["carol"]}[substitution]
    elif substitution == "evidence-name":
        descriptor["evidence"][0]["name"] = "another-note"
    else:
        note = case["root"] / "evidence/note.md"
        note.write_bytes(note.read_bytes() + b"\n")
    for name, value in context.items():
        _write_json(case["root"] / descriptor[name], value)
    _write_json(case["root"] / CHANGE, descriptor)
    _reject_consumers(case, source, retained)


@pytest.mark.parametrize("defect", ["extra-field", "missing-field", "schema", "declared-evidence-hash", "empty-evidence"])
def test_invalid_descriptor_never_creates_a_review(case, defect):
    descriptor = case["descriptor"]
    if defect == "extra-field":
        descriptor["production_authorized"] = True
    elif defect == "missing-field":
        del descriptor["purpose"]
    elif defect == "schema":
        descriptor["schema"] = "wsp-distro-change/999"
    elif defect == "declared-evidence-hash":
        descriptor["evidence"][0]["sha256"] = "0" * 64
    else:
        descriptor["evidence"] = []
    _write_json(case["root"] / CHANGE, descriptor)
    with pytest.raises(ValueError):
        case["adapter"].inspect_change(CHANGE)
    with pytest.raises(ValueError):
        case["adapter"].prepare_review(CHANGE, "review/invalid.json")
    assert not (case["root"] / "review/invalid.json").exists()


@pytest.mark.parametrize("target", [CHANGE, "context/constraints.json"])
def test_duplicate_application_json_is_rejected_even_with_identical_values(case, target):
    path = case["root"] / target
    value = governance.read_json(path)
    key = next(iter(value))
    raw = json.dumps(value)
    path.write_text("{" + json.dumps(key) + ":" + json.dumps(value[key]) + "," + raw[1:], encoding="utf-8")
    with pytest.raises(ValueError):
        case["adapter"].prepare_review(CHANGE, "review/invalid.json")
    assert not (case["root"] / "review/invalid.json").exists()


@pytest.mark.parametrize("nonfinite", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_nonfinite_opaque_constraints_cannot_enter_a_request(case, nonfinite):
    (case["root"] / "context/constraints.json").write_text('{"threshold":' + nonfinite + "}", encoding="utf-8")
    with pytest.raises(ValueError):
        case["adapter"].prepare_review(CHANGE, "review/invalid.json")
    assert not (case["root"] / "review/invalid.json").exists()


def test_complete_history_is_replayed_instead_of_trusting_head_field(case):
    approved = _human_decide(_prepare(case))
    retained = _pins(approved)
    approved["events"][0]["reason"] = "Substituted without changing retained head"
    _reject_consumers(case, _save_bundle(case, approved), retained)


@pytest.mark.parametrize("flag", ["production_authorized", "scientific_authorized"])
def test_received_bundle_cannot_upgrade_authority(case, flag):
    bundle = _human_decide(_prepare(case))
    retained = _pins(bundle)
    bundle[flag] = True
    _reject_consumers(case, _save_bundle(case, bundle), retained)


def test_both_pins_are_mandatory_and_independently_checked(case):
    requested = _prepare(case)
    source = "review/requested.json"
    for field in ("expected_request_sha256", "expected_head_sha256"):
        _reject_consumers(case, source, _pins(requested) | {field: "0" * 64})
        with pytest.raises(TypeError):
            case["adapter"].review_status(CHANGE, source, **{k: v for k, v in _pins(requested).items() if k != field})


def test_exclusive_outputs_preserve_existing_bytes(case):
    requested = _prepare(case)
    source = case["root"] / "review/requested.json"
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        case["adapter"].prepare_review(CHANGE, "review/requested.json")
    assert source.read_bytes() == before
    destination = case["root"] / "review/exported.json"
    destination.write_bytes(b"An existing independent record\n")
    with pytest.raises(FileExistsError):
        case["adapter"].export_review(CHANGE, "review/requested.json", **_pins(requested), out="review/exported.json")
    assert destination.read_bytes() == b"An existing independent record\n"


@pytest.mark.parametrize("unsafe", [
    "", ".", "../outside.json", "./changes/routing.json", "changes/../changes/routing.json",
    "changes//routing.json", "/absolute.json", "C:/absolute.json", r"changes\routing.json",
    "changes/routing.json:stream", "CON", "context/AUX.json", "changes/routing.json.", "changes/routing.json ",
])
def test_descriptor_path_requires_normalized_portable_workspace_relative_syntax(case, unsafe):
    with pytest.raises(ValueError):
        case["adapter"].inspect_change(unsafe)


@pytest.mark.parametrize("device", ["NUL .txt", "CON .txt", "COM1 .json", "CONIN$", "CONOUT$"])
@pytest.mark.parametrize("layout", ["{}", "nested/{}", "nested/{}/review.json"])
def test_windows_device_names_are_rejected_for_inputs_and_outputs_before_io(case, monkeypatch, device, layout):
    requested = _prepare(case)
    retained = _pins(requested)
    unsafe = layout.format(device)

    def guard(operation):
        def guarded(path, *args, **kwargs):
            if any(part.upper() == device.upper() for part in path.parts):
                pytest.fail("Reserved Windows device paths must be rejected before filesystem access")
            return operation(path, *args, **kwargs)
        return guarded

    # Fail safely even if validation regresses: never probe, create or open a device.
    for name in ("lstat", "stat", "mkdir", "open"):
        monkeypatch.setattr(Path, name, guard(getattr(Path, name)))

    adapter = case["adapter"]
    with pytest.raises(ValueError):
        adapter.inspect_change(unsafe)
    _reject_consumers(case, unsafe, retained)
    with pytest.raises(ValueError):
        adapter.prepare_review(CHANGE, unsafe)
    with pytest.raises(ValueError):
        adapter.export_review(CHANGE, "review/requested.json", **retained, out=unsafe)

    original = copy.deepcopy(case["descriptor"])
    for field in ("base", "evidence"):
        descriptor = copy.deepcopy(original)
        if field == "evidence":
            descriptor["evidence"][0]["path"] = unsafe
        else:
            descriptor[field] = unsafe
        _write_json(case["root"] / CHANGE, descriptor)
        with pytest.raises(ValueError):
            adapter.inspect_change(CHANGE)
        with pytest.raises(ValueError):
            adapter.prepare_review(CHANGE, "review/device-input.json")
        assert not (case["root"] / "review/device-input.json").exists()


@pytest.mark.parametrize("field", ["base", "evidence"])
def test_nested_descriptor_cannot_read_outside_workspace_or_use_absolute_paths(case, field):
    outside = case["root"].parent / "outside.json"
    outside.write_bytes(b"external data must remain outside the adapter\n")
    for unsafe in ("../outside.json", outside.as_posix()):
        if field == "evidence":
            case["descriptor"]["evidence"][0]["path"] = unsafe
        else:
            case["descriptor"][field] = unsafe
        _write_json(case["root"] / CHANGE, case["descriptor"])
        with pytest.raises(ValueError):
            case["adapter"].prepare_review(CHANGE, "review/invalid.json")
        assert not (case["root"] / "review/invalid.json").exists()
    assert outside.read_bytes() == b"external data must remain outside the adapter\n"


def test_bundle_and_output_paths_are_confined_before_io(case):
    requested = _prepare(case)
    retained = _pins(requested)
    outside = case["root"].parent / "external-bundle.json"
    _write_json(outside, requested)
    _reject_consumers(case, "../external-bundle.json", retained)
    for out in ("../escaped-review.json", (case["root"] / "absolute-review.json").as_posix()):
        with pytest.raises(ValueError):
            case["adapter"].prepare_review(CHANGE, out)
        with pytest.raises(ValueError):
            case["adapter"].export_review(CHANGE, "review/requested.json", **retained, out=out)
    assert not (case["root"].parent / "escaped-review.json").exists()
    assert not (case["root"] / "absolute-review.json").exists()


def _symlink(link: Path, target: Path):
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as error:
        # Windows may disallow link creation without Developer Mode or the link privilege.
        if error.errno in {1, 13, 38, 95} or getattr(error, "winerror", None) == 1314:
            pytest.skip(f"Symbolic links unavailable in this test environment: {error}")
        raise


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_links_are_rejected_even_when_their_target_is_inside_workspace(case, kind):
    if kind == "file":
        _symlink(case["root"] / "linked-base.json", case["root"] / "context/base.json")
        case["descriptor"]["base"] = "linked-base.json"
    else:
        _symlink(case["root"] / "linked-evidence", case["root"] / "evidence")
        case["descriptor"]["evidence"][0]["path"] = "linked-evidence/note.md"
    _write_json(case["root"] / CHANGE, case["descriptor"])
    with pytest.raises(ValueError):
        case["adapter"].prepare_review(CHANGE, "review/invalid.json")
    assert not (case["root"] / "review/invalid.json").exists()


def test_output_directory_link_cannot_write_outside_workspace(case):
    requested = _prepare(case)
    outside = case["root"].parent / "outside-output"
    outside.mkdir()
    _symlink(case["root"] / "linked-output", outside)
    with pytest.raises(ValueError):
        case["adapter"].prepare_review(CHANGE, "linked-output/requested.json")
    with pytest.raises(ValueError):
        case["adapter"].export_review(CHANGE, "review/requested.json", **_pins(requested), out="linked-output/exported.json")
    assert list(outside.iterdir()) == []


def test_linked_descriptor_and_bundle_are_rejected(case):
    requested = _prepare(case)
    _symlink(case["root"] / "linked-change.json", case["root"] / CHANGE)
    with pytest.raises(ValueError):
        case["adapter"].inspect_change("linked-change.json")
    _symlink(case["root"] / "linked-bundle.json", case["root"] / "review/requested.json")
    _reject_consumers(case, "linked-bundle.json", _pins(requested))


def test_official_sdk_stdio_discovery_review_cycle_and_errors(case, caplog):
    mcp = pytest.importorskip("mcp")
    import anyio
    from mcp.client.stdio import stdio_client

    def data(result):
        assert result.is_error is False
        assert type(result.structured_content) is dict
        text, = result.content
        assert text.type == "text"
        assert json.loads(text.text) == result.structured_content
        return result.structured_content

    async def exercise():
        server = mcp.StdioServerParameters(
            command=sys.executable,
            args=["-m", "rlcd.weight_space.distro_mcp", "--workspace", str(case["root"])],
            cwd=Path(__file__).resolve().parents[1],
        )
        with anyio.fail_after(30):
            async with stdio_client(server) as (read, write):
                async with mcp.ClientSession(read, write, read_timeout_seconds=10.0) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    status_inputs = {"change", "bundle", "expected_request_sha256", "expected_head_sha256"}
                    expected = {
                        "wsp_inspect_change": {"change"},
                        "wsp_prepare_review": {"change", "out"},
                        "wsp_review_status": status_inputs,
                        "wsp_export_review": status_inputs | {"out"},
                    }
                    # In particular, the host receives no agent decision tool.
                    assert {tool.name for tool in tools} == set(expected)
                    for tool in tools:
                        read_only = tool.name in {"wsp_inspect_change", "wsp_review_status"}
                        assert tool.annotations.read_only_hint is read_only
                        assert tool.annotations.destructive_hint is False
                        assert tool.annotations.idempotent_hint is read_only
                        assert tool.annotations.open_world_hint is False
                        schema = tool.input_schema
                        assert schema["additionalProperties"] is False
                        assert set(schema["required"]) == expected[tool.name]
                        assert set(schema["properties"]) == expected[tool.name]
                        assert all(value["type"] == "string" for value in schema["properties"].values())

                    inspected = data(await session.call_tool("wsp_inspect_change", {"change": CHANGE}))
                    assert inspected["request"] == case["adapter"].inspect_change(CHANGE)["request"]
                    _assert_authority(inspected, "draft")
                    prepared = data(await session.call_tool("wsp_prepare_review", {
                        "change": CHANGE, "out": "review/mcp-requested.json",
                    }))
                    retained = _pins(prepared)
                    _assert_authority(prepared, "requested")
                    status_args = {"change": CHANGE, "bundle": "review/mcp-requested.json", **retained}
                    _assert_authority(data(await session.call_tool("wsp_review_status", status_args)), "requested")

                    # Only the independently operated governance API supplies a human decision.
                    requested = governance.read_json(case["root"] / "review/mcp-requested.json")
                    approved = _human_decide(requested)
                    approved_path = _save_bundle(case, approved, "mcp-approved")
                    status_args.update(bundle=approved_path, **_pins(approved))
                    _assert_authority(data(await session.call_tool("wsp_review_status", status_args)), "approved")
                    exported = data(await session.call_tool("wsp_export_review", {
                        **status_args, "out": "review/mcp-exported.json",
                    }))
                    _assert_authority(exported, "approved")
                    assert exported["events"] == approved["events"]
                    assert (case["root"] / "review/mcp-exported.json").read_bytes() == (case["root"] / approved_path).read_bytes()

                    revoked = _human_decide(approved, "revoke", "alice")
                    status_args.update(**_pins(revoked))
                    assert (await session.call_tool("wsp_review_status", status_args)).is_error is True
                    assert (await session.call_tool("wsp_export_review", {
                        **status_args, "out": "review/mcp-stale-export.json",
                    })).is_error is True
                    assert not (case["root"] / "review/mcp-stale-export.json").exists()
                    status_args.update(bundle=_save_bundle(case, revoked, "mcp-revoked"))
                    _assert_authority(data(await session.call_tool("wsp_review_status", status_args)), "revoked")

                    invalid_calls = [
                        ("wsp_prepare_review", {"change": CHANGE, "out": "review/mcp-invalid.json", "actor": "bob"}),
                        ("wsp_review_status", {"change": CHANGE, "bundle": approved_path}),
                        ("wsp_inspect_change", {"change": 1}),
                        ("wsp_approve_review", {"actor": "bob"}),
                    ]
                    for name, arguments in invalid_calls:
                        assert (await session.call_tool(name, arguments)).is_error is True
                    assert not (case["root"] / "review/mcp-invalid.json").exists()

    anyio.run(exercise)
    assert not [record for record in caplog.records
                if record.name.startswith("mcp.client.stdio") and record.levelno >= 40]
