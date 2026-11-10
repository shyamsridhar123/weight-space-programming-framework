"""Scoped synthetic regression of AST-extracted public event publication.

The publisher and its JSON key and framing helpers are extracted from source.
The test-local write_json sink checks publication arguments and readback; it does
not exercise the production writer, providers, studies, or event replay semantics.
All inputs below are fresh synthetic literals. Copied metadata remains verbatim;
this regression does not establish general confidentiality or historical repair.
"""

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any


def test_publication_redacts_json_keys_and_omits_collisions(tmp_path):
    # Resolve the installed tests/ source without importing the project module.
    source_path = Path(__file__).resolve().parents[1] / "rlcd/weight_space/live_provider.py"
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    required = (
        "_DuplicateJSONKey",
        "_RedactedJSONKeyCollision",
        "_unique_json_object",
        "_jsonl_lines",
        "publish_attempt",
    )
    selected = []
    for name in required:
        matches = [
            node for node in source_tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
        ]
        assert len(matches) == 1, name
        selected.append(matches[0])

    sink_calls = []

    def write_json(path, value):
        # Deliberately disclosed substitute for the production atomic JSON writer.
        serialized = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2) + "\n"
        snapshot = json.loads(serialized)
        sink_calls.append((Path(path), snapshot))
        Path(path).write_bytes(serialized.encode("utf-8"))

    bindings = {
        "__builtins__": __builtins__,
        "Any": Any,
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "re": re,
        "shutil": shutil,
        "write_json": write_json,
    }
    extracted = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(extracted, str(source_path), "exec"), bindings)
    publish_attempt = bindings["publish_attempt"]

    marker = "public event omitted: JSON object key collision after profile redaction"
    before_raw = r'{"type":"synthetic.before","message":"/home/private-person/kept"}'
    after_raw = r'{"type":"synthetic.after","message":"/Users/private-person/kept"}'
    before_public = {"type": "synthetic.before", "message": "[USER_PROFILE]/kept"}
    after_public = {"type": "synthetic.after", "message": "[USER_PROFILE]/kept"}
    # Approved fictional profile labels remain distinct before redaction.
    synthetic_identities = (
        b"private-person",
        b"user",
        b"owner",
        b"example",
        b"username",
    )

    # Every case gets independent metadata/events. The first and last events each
    # contribute one retained redaction, so collision rollback must restore one.
    cases = (
        (
            "nested_posix_unicode_and_surrogates",
            r'{"type":"synthetic.posix","details":[{"outer":{"/home/user/private.txt":"from /Users/owner/payload.txt","café":"雪","\ud800":"\udfff","[USER_PROFILE]/literal":"[USER_PROFILE]/value"}}]}',
            {
                "type": "synthetic.posix",
                "details": [{"outer": {
                    "[USER_PROFILE]/private.txt": "from [USER_PROFILE]/payload.txt",
                    "café": "雪",
                    "\ud800": "\udfff",
                    "[USER_PROFILE]/literal": "[USER_PROFILE]/value",
                }}],
            },
            4,
        ),
        (
            "escaped_windows_nested_keys_and_value",
            r'{"type":"synthetic.windows","details":[{"\u0043:\u005cUsers\u005cuser\u005csecret.txt":{"D:\\Documents and Settings\\owner\\note":"E:\\Users\\example\\payload.txt"}}]}',
            {
                "type": "synthetic.windows",
                "details": [{"[USER_PROFILE]\\secret.txt": {
                    "[USER_PROFILE]\\note": "[USER_PROFILE]\\payload.txt",
                }}],
            },
            5,
        ),
        (
            "two_profiles_collide_after_earlier_value_redactions",
            r'{"type":"synthetic.collision","before":["/home/username/one",{"note":"C:\\Users\\username\\two"}],"details":{"/home/user/x":"first","/home/owner/x":"second"}}',
            marker,
            2,
        ),
        (
            "transformed_key_precedes_existing_marker",
            r'{"type":"synthetic.collision","before":["/home/username/one",{"note":"C:\\Users\\username\\two"}],"details":{"/home/user/x":"first","[USER_PROFILE]/x":"second"}}',
            marker,
            2,
        ),
        (
            "existing_marker_precedes_transformed_key",
            r'{"type":"synthetic.collision","before":["/home/username/one",{"note":"C:\\Users\\username\\two"}],"details":{"[USER_PROFILE]/x":"first","/home/user/x":"second"}}',
            marker,
            2,
        ),
        (
            "escaped_windows_two_profiles_collide",
            r'{"type":"synthetic.collision","before":["/home/username/one",{"note":"C:\\Users\\username\\two"}],"details":{"\u0043:\u005cUsers\u005cuser\u005cx":"first","D:\\Users\\owner\\x":"second"}}',
            marker,
            2,
        ),
        (
            "two_profiles_collide_in_reverse_insertion_order",
            r'{"type":"synthetic.collision","before":["/home/username/one",{"note":"C:\\Users\\username\\two"}],"details":{"/home/owner/x":"second","/home/user/x":"first"}}',
            marker,
            2,
        ),
        (
            "equal_redacted_keys_in_separate_objects_are_allowed",
            r'{"type":"synthetic.separate","details":[{"/home/user/x":"first"},{"/home/owner/x":"second"}]}',
            {
                "type": "synthetic.separate",
                "details": [{"[USER_PROFILE]/x": "first"}, {"[USER_PROFILE]/x": "second"}],
            },
            4,
        ),
    )

    for case_name, middle_raw, middle_public, expected_redactions in cases:
        case_root = tmp_path / case_name
        source = case_root / "raw"
        destination = case_root / "public"
        source.mkdir(parents=True, exist_ok=False)
        metadata = {
            "request.json": b'{"synthetic_request":"/home/private-person/request"}\n',
            "response-schema.json": b'{ "type": "object", "synthetic": true }\n',
            "receipt.json": br'{"synthetic_receipt":"C:\\Users\\private-person\\receipt"}' + b"\n",
        }
        raw = ("\n".join((before_raw, middle_raw, after_raw)) + "\n").encode("utf-8")
        originals = {**metadata, "events.jsonl": raw}
        for name, content in originals.items():
            (source / name).write_bytes(content)

        expected_objects = [before_public, middle_public, after_public]
        expected_public = (
            "\n".join(json.dumps(value, ensure_ascii=True, allow_nan=False) for value in expected_objects)
            + "\n"
        ).encode("utf-8")
        expected_record = {
            "schema": "wsp-public-event-derivative/1",
            "raw_events_sha256": hashlib.sha256(raw).hexdigest(),
            "public_events_sha256": hashlib.sha256(expected_public).hexdigest(),
            "redactions": expected_redactions,
            "transform": "JSON_reserialization_and_user_profile_prefix_redaction",
            "raw_bytes_published": False,
            "scope": "public_derivative_replay;_raw_identity_disclosed_not_publicly_replayed",
        }

        sink_calls.clear()
        record = publish_attempt(source, destination)
        public = (destination / "events.jsonl").read_bytes()
        assert public == expected_public, case_name
        assert public != raw, case_name
        # This assertion is limited to event bytes; metadata stays verbatim below.
        for identity in synthetic_identities:
            assert identity not in public, (case_name, identity)
        assert [json.loads(line) for line in public.decode("utf-8").splitlines()] == expected_objects, case_name
        assert record == expected_record, case_name
        assert len(record) == 7, case_name
        assert record["raw_events_sha256"] != record["public_events_sha256"], case_name
        assert record["raw_events_sha256"] == hashlib.sha256((source / "events.jsonl").read_bytes()).hexdigest()
        assert record["public_events_sha256"] == hashlib.sha256(public).hexdigest()
        assert sink_calls == [(destination / "event-publication.json", expected_record)], case_name
        expected_record_bytes = (
            json.dumps(expected_record, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        assert (destination / "event-publication.json").read_bytes() == expected_record_bytes, case_name
        assert json.loads((destination / "event-publication.json").read_bytes()) == expected_record, case_name
        assert {path.name for path in source.iterdir()} == set(originals), case_name
        assert {path.name for path in destination.iterdir()} == {
            *metadata, "events.jsonl", "event-publication.json",
        }, case_name
        for name, original in originals.items():
            assert (source / name).read_bytes() == original, (case_name, name)
        for name, original in metadata.items():
            assert (destination / name).read_bytes() == original, (case_name, name)

        if middle_public == marker:
            # JSON-string omission is intentional; no parser-equivalence claim.
            assert public.splitlines()[1] == json.dumps(marker).encode("utf-8"), case_name
            assert json.loads(public.splitlines()[2]) == after_public, case_name
        if case_name == "nested_posix_unicode_and_surrogates":
            nested = json.loads(public.splitlines()[1])["details"][0]["outer"]
            assert nested["café"] == "雪"
            assert nested["\ud800"] == "\udfff"
            assert b"\\ud800" in public and b"\\udfff" in public
