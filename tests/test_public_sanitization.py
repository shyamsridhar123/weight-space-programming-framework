"""Verify that public derivatives cannot silently stand in for original bytes."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_publication_sanitization", ROOT / "scripts/verify_publication.py")
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)

EXACT = "data/exact.txt"
LIVE = "data/settings.json"
PRESERVED = "provenance/original-files/data/settings.json"
OMITTED = "evidence/unpublished.log"


def identity(content: bytes) -> dict:
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def write(root: Path, relative: str, content: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_json(root: Path, relative: str, value: object) -> None:
    write(root, relative, (json.dumps(value, indent=2) + "\n").encode())


def refresh_manifest(root: Path) -> None:
    lines = [
        f"{hashlib.sha256((root / relative).read_bytes()).hexdigest()}  {relative}\n"
        for relative in publication.publication_files(root)
    ]
    write(root, publication.MANIFEST, "".join(lines).encode())


def save_ledger(root: Path, ledger: dict) -> None:
    write_json(root, publication.PUBLIC_SANITIZATION, ledger)
    refresh_manifest(root)


def disclosure(path: str, original: bytes, public: bytes | None) -> dict:
    return {
        "path": path,
        "original": identity(original),
        "public": None if public is None else identity(public),
        "reasons": ["Remove private build metadata from the public derivative."],
    }


@pytest.fixture
def snapshot(tmp_path):
    exact = b"Unchanged source.\n"
    original = b'{"endpoint":"private"}\n'
    redacted = b'{"endpoint":"removed"}\n'
    omitted = b"Private build trace.\n"
    # The current file and its preserved source are distinct historical bytes.
    live_original = b'{"current":"before-publication"}\n'
    live_public = b'{"current":"public"}\n'
    for path, content in [(EXACT, exact), (PRESERVED, redacted), (LIVE, live_public)]:
        write(tmp_path, path, content)
    write_json(tmp_path, publication.SOURCE_INDEX, {
        "schema": "wspf-preserved-source-files/1",
        "files": [
            {"path": EXACT, "preserved_path": EXACT, **identity(exact)},
            {"path": LIVE, "preserved_path": PRESERVED, **identity(original)},
            {"path": OMITTED, "preserved_path": OMITTED, **identity(omitted)},
        ],
    })
    ledger = {
        "schema": publication.PUBLIC_SANITIZATION_SCHEMA,
        "source_commit": "2424c592fd8ed6b57217bd7b55224f77d0094455",
        "files": [
            disclosure(PRESERVED, original, redacted),
            disclosure(OMITTED, omitted, None),
            disclosure(LIVE, live_original, live_public),
        ],
    }
    save_ledger(tmp_path, ledger)
    return tmp_path, ledger


def test_legacy_snapshot_without_ledger_still_requires_exact_originals(tmp_path):
    content = b"Exact source bytes.\r\n"
    write(tmp_path, EXACT, content)
    write_json(tmp_path, publication.SOURCE_INDEX, {
        "files": [{"path": EXACT, "preserved_path": EXACT, **identity(content)}],
    })
    refresh_manifest(tmp_path)
    assert publication.load_public_sanitization(tmp_path) == {}
    receipt = publication.verify(tmp_path)
    assert receipt["ok"], receipt["failures"]
    assert receipt["counts"]["source_files_verified"] == 1
    assert receipt["counts"]["source_bytes_verified"] == len(content)
    assert receipt["counts"]["source_files_redacted"] == 0
    assert receipt["counts"]["source_files_omitted"] == 0
    assert receipt["counts"]["sanitization_records"] == 0


def test_disclosed_derivatives_have_separate_counts_and_leave_index_unchanged(snapshot):
    root, _ = snapshot
    before = (root / publication.SOURCE_INDEX).read_bytes()
    receipt = publication.verify(root)
    assert receipt["ok"], receipt["failures"]
    counts = receipt["counts"]
    assert counts["source_records"] == 3
    assert counts["source_files_verified"] == 1
    assert counts["source_bytes_verified"] == len(b"Unchanged source.\n")
    assert counts["source_files_redacted"] == 1
    assert counts["source_files_omitted"] == 1
    assert counts["sanitization_records"] == 3
    assert counts["sanitization_public_files_verified"] == 2
    assert counts["sanitization_omissions_verified"] == 1
    assert (root / publication.SOURCE_INDEX).read_bytes() == before


def test_index_binding_uses_preserved_path_instead_of_current_source_alias(snapshot):
    root, ledger = snapshot
    entries = publication.load_public_sanitization(root)
    assert ledger["files"][0]["original"] != ledger["files"][2]["original"]
    assert entries[PRESERVED]["original"] == ledger["files"][0]["original"]
    assert entries[LIVE]["original"] == ledger["files"][2]["original"]


def test_historical_match_requires_the_recorded_path_hash_and_available_size(snapshot):
    root, _ = snapshot
    entries = publication.load_public_sanitization(root)
    original = entries[PRESERVED]["original"]
    match = publication.match_public_sanitization
    assert match(entries, PRESERVED, original["sha256"], original["bytes"]) == "redacted"
    assert match(entries, PRESERVED, original["sha256"].upper()) == "redacted"
    assert match(entries, OMITTED, entries[OMITTED]["original"]["sha256"]) == "omitted"
    assert match(entries, "unrelated.json", original["sha256"], original["bytes"]) is None
    assert match(entries, LIVE, original["sha256"], original["bytes"]) is None
    assert match(entries, PRESERVED, "0" * 64, original["bytes"]) is None
    assert match(entries, PRESERVED, original["sha256"], original["bytes"] + 1) is None


def test_removing_ledger_does_not_make_changed_or_missing_sources_valid(snapshot):
    root, _ = snapshot
    (root / publication.PUBLIC_SANITIZATION).unlink()
    refresh_manifest(root)
    receipt = publication.verify(root)
    assert not receipt["ok"]
    assert receipt["counts"]["source_files_verified"] == 1
    assert receipt["counts"]["source_files_redacted"] == 0
    assert receipt["counts"]["source_files_omitted"] == 0
    assert {failure["code"] for failure in receipt["failures"]} >= {"source_sha256", "source_entry"}


@pytest.mark.parametrize("field", ["sha256", "bytes"])
def test_indexed_original_binding_cannot_be_replaced(snapshot, field):
    root, ledger = snapshot
    ledger["files"][0]["original"][field] = "0" * 64 if field == "sha256" else 999
    save_ledger(root, ledger)
    with pytest.raises(ValueError, match="original binding does not match the source index"):
        publication.load_public_sanitization(root)
    receipt = publication.verify(root)
    assert not receipt["ok"]
    assert receipt["counts"]["source_files_redacted"] == 0


@pytest.mark.parametrize("mutation", ["hash", "bytes", "actual-content", "missing-file"])
def test_public_binding_must_match_actual_bytes(snapshot, mutation):
    root, ledger = snapshot
    if mutation == "hash":
        ledger["files"][0]["public"]["sha256"] = "0" * 64
    elif mutation == "bytes":
        ledger["files"][0]["public"]["bytes"] += 1
    elif mutation == "actual-content":
        write(root, PRESERVED, b"Unexpected replacement.\n")
    else:
        (root / PRESERVED).unlink()
    save_ledger(root, ledger)
    receipt = publication.verify(root)
    assert not receipt["ok"]
    assert receipt["counts"]["sanitization_records"] == 0
    assert receipt["counts"]["source_files_redacted"] == 0
    assert any(failure["code"] == "public_sanitization" for failure in receipt["failures"])


@pytest.mark.parametrize("field", ["original", "public"])
@pytest.mark.parametrize("value", [
    [], {}, "identity",
    {"bytes": True, "sha256": "0" * 64},
    {"bytes": -1, "sha256": "0" * 64},
    {"bytes": 1.0, "sha256": "0" * 64},
    {"bytes": "1", "sha256": "0" * 64},
    {"bytes": 1},
    {"sha256": "0" * 64},
    {"bytes": 1, "sha256": "g" * 64},
    {"bytes": 1, "sha256": "0" * 63},
    {"bytes": 1, "sha256": 0},
])
def test_malformed_identity_fields_are_rejected(snapshot, field, value):
    root, ledger = snapshot
    ledger["files"][0][field] = value
    save_ledger(root, ledger)
    with pytest.raises(ValueError):
        publication.load_public_sanitization(root)


def test_null_original_is_rejected(snapshot):
    root, ledger = snapshot
    ledger["files"][0]["original"] = None
    save_ledger(root, ledger)
    with pytest.raises(ValueError, match="original must be an object"):
        publication.load_public_sanitization(root)


@pytest.mark.parametrize("field", ["path", "original", "public", "reasons"])
def test_required_record_fields_cannot_be_omitted(snapshot, field):
    root, ledger = snapshot
    del ledger["files"][0][field]
    save_ledger(root, ledger)
    with pytest.raises(ValueError):
        publication.load_public_sanitization(root)


@pytest.mark.parametrize("value", [[], "", [" "], [None], ["valid reason", 1]])
def test_each_disclosure_requires_explanations(snapshot, value):
    root, ledger = snapshot
    ledger["files"][0]["reasons"] = value
    save_ledger(root, ledger)
    with pytest.raises(ValueError, match="reasons"):
        publication.load_public_sanitization(root)


@pytest.mark.parametrize(("field", "value"), [
    ("schema", "wspf-public-sanitization/999"),
    ("source_commit", None),
    ("source_commit", "2424c59"),
    ("source_commit", "g" * 40),
    ("files", None),
    ("files", {}),
    ("files", [None]),
])
def test_ledger_envelope_is_validated(snapshot, field, value):
    root, ledger = snapshot
    ledger[field] = value
    save_ledger(root, ledger)
    with pytest.raises(ValueError):
        publication.load_public_sanitization(root)


def test_duplicate_json_keys_are_rejected(snapshot):
    root, ledger = snapshot
    text = json.dumps(ledger).replace(
        '"source_commit":', '"source_commit": "0000000000000000000000000000000000000000", "source_commit":', 1,
    )
    write(root, publication.PUBLIC_SANITIZATION, text.encode())
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        publication.load_public_sanitization(root)


@pytest.mark.parametrize("path", [
    "", None, 1, "../escape.txt", "a/../escape.txt", "./a.txt", "a//b.txt",
    "/absolute.txt", "C:/absolute.txt", "a\\b.txt", "a.txt/", "NUL.txt",
    "dir/COM1", "dir/a.", "dir/a ", " dir/a", "a\x00b", "a\nb",
    "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b",
])
def test_ledger_paths_are_portable_and_cannot_escape_repository(snapshot, path):
    root, ledger = snapshot
    ledger["files"].append(disclosure(path, b"Original bytes.", None))
    save_ledger(root, ledger)
    with pytest.raises(ValueError):
        publication.load_public_sanitization(root)


@pytest.mark.parametrize("path", [
    ".git/config", ".runs/receipt.json", ".RUNS/receipt.json", "node_modules/a.js",
    "build/a.txt", "dist/a.txt", "package.egg-info/PKG-INFO", "__pycache__/a.pyc",
    "formal/weight-space/.lake/a", "MANIFEST.sha256", "manifest.sha256",
    publication.PUBLIC_SANITIZATION,
])
def test_ledger_cannot_exempt_excluded_files_or_itself(snapshot, path):
    root, ledger = snapshot
    ledger["files"].append(disclosure(path, b"Original bytes.", None))
    save_ledger(root, ledger)
    with pytest.raises(ValueError):
        publication.load_public_sanitization(root)


def test_archived_cache_files_remain_within_publication_coverage(snapshot):
    root, ledger = snapshot
    path = "artifacts/historical/__pycache__/old.pyc"
    ledger["files"].append(disclosure(path, b"Archived bytes.", None))
    save_ledger(root, ledger)
    assert publication.load_public_sanitization(root)[path]["public"] is None


@pytest.mark.parametrize("change_case", [False, True])
def test_duplicate_and_case_colliding_paths_are_rejected(snapshot, change_case):
    root, ledger = snapshot
    duplicate = copy.deepcopy(ledger["files"][0])
    if change_case:
        duplicate["path"] = duplicate["path"].upper()
    ledger["files"].append(duplicate)
    save_ledger(root, ledger)
    with pytest.raises(ValueError, match="duplicate or case-colliding"):
        publication.load_public_sanitization(root)


def test_preserved_path_case_cannot_bypass_original_binding(snapshot):
    root, ledger = snapshot
    ledger["files"][0]["path"] = PRESERVED.upper()
    ledger["files"][0]["original"]["sha256"] = "0" * 64
    save_ledger(root, ledger)
    with pytest.raises(ValueError, match="preserved_path spelling"):
        publication.load_public_sanitization(root)


def test_ledger_file_paths_cannot_also_name_ancestors(snapshot):
    root, ledger = snapshot
    ledger["files"].extend([
        disclosure("missing/path/file", b"Child bytes.", None),
        disclosure("missing/path", b"Parent bytes.", None),
    ])
    save_ledger(root, ledger)
    with pytest.raises(ValueError, match="ancestors"):
        publication.load_public_sanitization(root)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_an_omission_must_really_be_absent(snapshot, kind):
    root, ledger = snapshot
    if kind == "file":
        write(root, OMITTED, b"Still present.\n")
    else:
        (root / OMITTED).mkdir(parents=True)
    save_ledger(root, ledger)
    with pytest.raises(ValueError, match="omitted path is still present"):
        publication.load_public_sanitization(root)


def test_omitted_path_cannot_have_a_file_ancestor(snapshot):
    root, ledger = snapshot
    write(root, "blocked", b"A file, not a directory.\n")
    ledger["files"][1]["path"] = "blocked/missing.log"
    save_ledger(root, ledger)
    with pytest.raises(ValueError, match="non-directory ancestor"):
        publication.load_public_sanitization(root)


@pytest.mark.parametrize("kind", ["broken-omission", "redacted", "linked-ancestor", "ledger"])
def test_links_are_never_valid_redactions_or_omissions(snapshot, kind):
    root, ledger = snapshot
    if kind == "broken-omission":
        path, target = root / OMITTED, root / "missing-target"
    elif kind == "redacted":
        path, target = root / PRESERVED, root / EXACT
        path.unlink()
    elif kind == "linked-ancestor":
        path, target = root / "linked", root / "data"
        ledger["files"][1]["path"] = "linked/absent.log"
        save_ledger(root, ledger)
    else:
        path, target = root / publication.PUBLIC_SANITIZATION, root / "ledger-copy.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.symlink_to(target, target_is_directory=kind == "linked-ancestor")
    except (OSError, NotImplementedError):
        pytest.skip("This filesystem does not permit creating symbolic links.")
    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        publication.load_public_sanitization(root)


def test_unchanged_bytes_cannot_be_labelled_a_redaction(snapshot):
    root, ledger = snapshot
    ledger["files"].append(disclosure(EXACT, b"Unchanged source.\n", b"Unchanged source.\n"))
    save_ledger(root, ledger)
    with pytest.raises(ValueError, match="must differ"):
        publication.load_public_sanitization(root)


def test_one_invalid_record_invalidates_all_ledger_exceptions(snapshot):
    root, ledger = snapshot
    ledger["files"][-1]["reasons"] = []
    save_ledger(root, ledger)
    receipt = publication.verify(root)
    assert not receipt["ok"]
    assert receipt["counts"]["sanitization_records"] == 0
    assert receipt["counts"]["source_files_redacted"] == 0
    assert receipt["counts"]["source_files_omitted"] == 0
    assert any(failure["code"] == "source_sha256" and failure.get("path") == PRESERVED
               for failure in receipt["failures"])


def test_original_verifier_rechecks_public_binding_after_ledger_load(snapshot):
    root, _ = snapshot
    entries = publication.load_public_sanitization(root)
    write(root, PRESERVED, b"Replaced after ledger validation.\n")
    counts = dict.fromkeys([
        "source_records", "source_files_verified", "source_bytes_verified",
        "source_files_redacted", "source_files_omitted",
    ], 0)
    failures = []
    publication._verify_originals(root, {}, counts, failures, entries)
    assert counts["source_files_redacted"] == 0
    assert any(failure["code"] == "source_sha256" for failure in failures)


def test_historical_helper_can_check_unindexed_disclosure_without_claiming_an_original(tmp_path):
    path = "historical/build.log"
    public = b"Public log.\n"
    write(tmp_path, path, public)
    record = disclosure(path, b"Private log.\n", public)
    save_ledger(tmp_path, {
        "schema": publication.PUBLIC_SANITIZATION_SCHEMA,
        "source_commit": "1" * 40,
        "files": [record],
    })
    entries = publication.load_public_sanitization(tmp_path)
    assert publication.match_public_sanitization(
        entries, path, record["original"]["sha256"], record["original"]["bytes"],
    ) == "redacted"
    receipt = publication.verify(tmp_path)
    assert not receipt["ok"]
    assert any(failure["code"] == "source_index" for failure in receipt["failures"])
