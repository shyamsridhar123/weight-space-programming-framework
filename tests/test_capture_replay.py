"""Fresh synthetic publication/replay regression using AST-extracted functions.

No project module, study entry point, or historical capture is imported. The
test-local JSON writer substitutes for atomic persistence. Metadata-absent
compatibility concerns receipt interpretation with the same authored configuration;
this check does not establish compatibility of older frozen studies or defaults.
"""

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import pytest


@pytest.mark.parametrize("case", ("complete", "incomplete", "invalid_utf8", "legacy"))
def test_capture_publication_and_construction_replay(tmp_path, case):
    def write_json(path, value):
        # Deliberately substitute for production atomic persistence, not validation.
        Path(path).write_bytes((json.dumps(value, allow_nan=False) + "\n").encode("utf-8"))

    bindings = {
        "__builtins__": __builtins__, "Any": Any, "Path": Path,
        "hashlib": hashlib, "json": json, "math": math, "re": re,
        "shutil": shutil, "write_json": write_json,
    }
    root = Path(__file__).resolve().parents[1]
    groups = (
        ("rlcd/weight_space/model.py", ("canonical_bytes", "digest")),
        ("rlcd/weight_space/live_provider.py", (
            "_DuplicateJSONKey", "_RedactedJSONKeyCollision", "_unique_json_object",
            "_jsonl_lines", "parse_events", "publish_attempt",
        )),
        ("scripts/run_construction_steering_pilot.py", (
            "file_hash", "loads", "read", "signed", "verify_signed", "inventory",
            "verify_receipt", "verify_attempt",
        )),
    )
    for relative, names in groups:
        source = root / relative
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        selected = []
        for name in names:
            matches = [node for node in tree.body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
            assert len(matches) == 1, name
            selected.append(matches[0])
        extracted = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
        exec(compile(extracted, str(source), "exec"), bindings)

    prompt = "Return the authored answer."
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    configuration = {"schema": "authored-offline-provider/1"}
    response = {"answer": "Literal authored response"}
    usage = {"input_tokens": 11, "output_tokens": 7}
    events = [
        {"type": "item.completed", "item": {"type": "error", "message": "authored warning"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(response)}},
        {"type": "turn.completed", "usage": usage},
    ]
    raw = ("\n".join(json.dumps(event, separators=(",", ":")) for event in events) + "\n").encode("utf-8")
    if case == "invalid_utf8":
        raw = raw.replace(b"warning", b"\xff")
    incomplete = case in ("incomplete", "invalid_utf8")
    invocation = {
        "configuration": configuration,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": bindings["digest"](schema),
        "cwd_role": "isolated_proposer_directory",
        "local_invocation_sha256": hashlib.sha256(b"authored-local-invocation").hexdigest(),
        "executable_sha256": hashlib.sha256(b"authored-cli").hexdigest(),
        "cli_version": "authored-offline-fixture",
    }
    if case != "legacy":
        invocation["output_capture"] = {"worker": {"complete": not incomplete}}
        if case == "invalid_utf8":
            invocation["output_capture"]["worker"]["text_decoding_error"] = "UnicodeDecodeError"
    receipt = bindings["signed"]({
        "schema": "wsp-live-provider-attempt/1",
        "started_utc": "2026-11-10T00:00:00+00:00",
        "status": "failed" if incomplete else "completed",
        "error": ("invalid_event_utf8" if case == "invalid_utf8" else "provider_output_limit_exceeded")
                 if incomplete else None,
        "returncode": 0,
        "timeout_cleanup_complete": None,
        "elapsed_seconds": 0.1,
        "invocation": invocation,
        "response": None if incomplete else response,
        "usage": {} if incomplete else usage,
        "unapproved_events": [],
        "event_errors": [],
        "diagnostic_items": 1,
        "events_sha256": hashlib.sha256(raw).hexdigest(),
        "cost": None,
        "prompt_scope": "authored_synthetic_fixture_only",
        "restrictions_scope": "authored_synthetic_fixture_only",
    })
    private, public = tmp_path / "private", tmp_path / "public"
    private.mkdir()
    for name, value in (
        ("request.json", {"prompt": prompt, "configuration": configuration}),
        ("response-schema.json", schema),
        ("receipt.json", receipt),
    ):
        write_json(private / name, value)
    (private / "events.jsonl").write_bytes(raw)
    originals = {path.name: path.read_bytes() for path in private.iterdir()}

    publication = bindings["publish_attempt"](private, public)
    public_bytes = (public / "events.jsonl").read_bytes()
    public_text = public_bytes.decode("utf-8")  # Strict decoding must succeed.
    transform = "JSON_reserialization_and_user_profile_prefix_redaction"
    replacement_transform = "UTF8_replacement_then_" + transform
    assert publication == {
        "schema": "wsp-public-event-derivative/1",
        "raw_events_sha256": hashlib.sha256(raw).hexdigest(),
        "public_events_sha256": hashlib.sha256(public_bytes).hexdigest(),
        "redactions": 0,
        "transform": replacement_transform if case == "invalid_utf8" else transform,
        "raw_bytes_published": False,
        "scope": "public_derivative_replay;_raw_identity_disclosed_not_publicly_replayed",
    }
    assert public_bytes != raw
    for name, original in originals.items():
        assert (private / name).read_bytes() == original
        if name != "events.jsonl":
            assert (public / name).read_bytes() == original

    # Even incomplete event bytes contain a plausible final answer and usage.
    for text in (raw.decode("utf-8", errors="replace"), public_text):
        parsed = bindings["parse_events"](text)
        assert parsed[:4] == (response, usage, [], [])
        assert len(parsed[4]) == 1
    replayed = bindings["verify_attempt"](public, prompt, schema, configuration)
    assert replayed == receipt
    assert replayed["response"] == (None if incomplete else response)
    assert replayed["usage"] == ({} if incomplete else usage)

    if not incomplete:
        # Replacement decoding is not an allowed claim for complete/legacy receipts.
        write_json(public / "event-publication.json", publication | {"transform": replacement_transform})
        with pytest.raises(ValueError, match="Public event derivative binding differs"):
            bindings["verify_attempt"](public, prompt, schema, configuration)
