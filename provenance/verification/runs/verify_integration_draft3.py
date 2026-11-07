"""Read-only boundary checks for the draft.3 document preservation changes."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASE = "93e8a9b7f2cfbbffab6c7d10f831b2cde4b69176"
OUTPUT = ROOT / ".runs/integration-draft3-verification.json"
OWNED_DOCUMENTS = {
    "docs/PROJECT_MEMORY.md",
    "docs/weight-space/README.md",
    "docs/weight-space/design.md",
    "docs/weight-space/requirements.md",
    "docs/weight-space/research.md",
    "docs/weight-space/results.md",
    "formal/weight-space/README.md",
}
EXPECTED_MAP = {
    "docs/weight-space/v2/formal.md": "provenance/original-documents/docs/weight-space/v2/formal.md.txt",
    "formal/weight-space/README.md": "provenance/original-documents/formal/weight-space/README.md.txt",
}


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_bytes(relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)


assert not OUTPUT.exists(), "Choose a new receipt path; existing receipts are immutable."
proof = load("publication_proof_checker", "scripts/check_proofs.py")
assert proof.PRESERVED_DOCUMENT_PATHS == EXPECTED_MAP
historical, expected = proof.historical_inputs()
with patch.object(proof, "sha256", wraps=proof.sha256) as observed_hash:
    verified = proof.verify_hashes(expected)
assert verified == expected
lookups = [call.args[0].relative_to(ROOT).as_posix() for call in observed_hash.call_args_list]
assert lookups == [EXPECTED_MAP.get(relative, relative) for relative in expected]
assert set(EXPECTED_MAP) <= set(expected)
assert proof.HISTORICAL_RECEIPT_SHA256 == "7d3435dac048bd2ae4c28b90e7e9a035c5134cd0edee75ae200b32c2d68a9eac"

original_sha = proof.sha256
tamper_rejections = []
ordinary_source = "formal/weight-space/WeightSpace/Generic.lean"
assert ordinary_source in expected
for relative in [*EXPECTED_MAP, ordinary_source]:
    checked = EXPECTED_MAP.get(relative, relative)
    target = ROOT / checked

    def altered_hash(path: Path, target: Path = target) -> str:
        return "0" * 64 if path == target else original_sha(path)

    with patch.object(proof, "sha256", side_effect=altered_hash):
        try:
            proof.verify_hashes(expected)
        except RuntimeError as error:
            assert str(error) == f"Preserved proof input differs: {relative} (checked {checked})"
            tamper_rejections.append({"historical_name": relative, "checked_path": checked,
                                      "rejected": True, "simulation": "mocked digest; input file unchanged"})
        else:
            raise AssertionError(f"Simulated digest change was accepted: {relative}")
assert proof.verify_hashes(expected) == expected
print("Proof hash lookups and simulated tamper rejection passed.", flush=True)

frozen_identity = {}
for relative in (proof.HISTORICAL_RECEIPT, "formal/weight-space/verification-v2/check.ps1"):
    content = (ROOT / relative).read_bytes()
    assert content == base_bytes(relative)
    frozen_identity[relative] = hashlib.sha256(content).hexdigest()

index_path = "provenance/source-files.json"
before = json.loads(base_bytes(index_path))
after = json.loads((ROOT / index_path).read_text(encoding="utf-8"))
assert {k: v for k, v in before.items() if k != "files"} == {
    k: v for k, v in after.items() if k != "files"
}
changed = {}
for old, new in zip(before["files"], after["files"], strict=True):
    assert {k: v for k, v in old.items() if k != "preserved_path"} == {
        k: v for k, v in new.items() if k != "preserved_path"
    }
    if old["preserved_path"] != new["preserved_path"]:
        changed[old["path"]] = new["preserved_path"]
preservation = json.loads((ROOT / ".runs/docs-preservation-draft3.json").read_text(encoding="utf-8"))
assert changed == {
    record["original_path"]: record["preserved_original"]["path"]
    for record in preservation["documents"] if record["source_index_remapped"]
}
for record in preservation["documents"]:
    for key in ("preserved_original", "previous_published_document"):
        item = record.get(key)
        if item is not None:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]

publication = load("publication_checker", "scripts/verify_publication.py")
original_failures = []
cache = {}
for record in after["files"]:
    if record["path"] in changed:
        actual = publication._digest(ROOT, record["preserved_path"], cache)
        assert actual == (record["bytes"], record["sha256"])
assert not original_failures, original_failures
print("Revised archive mappings preserve the original document identities.", flush=True)
files = publication.publication_files(ROOT)
markdown_failures = []
publication._verify_markdown(ROOT, files, defaultdict(int), markdown_failures)
owned_link_failures = [failure for failure in markdown_failures if failure.get("path") in OWNED_DOCUMENTS]
assert not owned_link_failures, owned_link_failures

receipt = {
    "schema": "wspf-document-integration-verification/1",
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_publication_commit": BASE,
    "passed": True,
    "proof_archive_lookup": {
        "historical_hashes_verified": True,
        "only_explicit_document_paths_redirected": True,
        "preserved_document_paths": EXPECTED_MAP,
        "simulated_tamper_rejections": tamper_rejections,
        "frozen_receipt_and_powershell_checker_unchanged": frozen_identity,
        "driver_sha256": sha(ROOT / "scripts/check_proofs.py"),
        "scope": "Read-only hash lookup and mocked digest rejection; no Lean rebuild performed.",
    },
    "preservation": {
        "source_index_sha256": sha(ROOT / index_path),
        "only_preserved_path_values_changed": True,
        "remapped_source_records_match_preserved_files": True,
        "all_document_archives_match_preservation_receipt": True,
        "source_index_remappings": changed,
    },
    "owned_documents": {relative: sha(ROOT / relative) for relative in sorted(OWNED_DOCUMENTS)},
    "owned_document_local_link_check": "passed",
    "other_document_link_failures_at_check_time": [
        failure for failure in markdown_failures if failure.get("path") not in OWNED_DOCUMENTS
    ],
    "limitations": [
        "The in-progress publication manifest was not checked or regenerated.",
        "The focused preservation check reads the remapped document inputs; final publication verification checks the full original-source index.",
        "The Markdown checker validates local destinations, not fragments or external URLs.",
        "The receipt records boundary checks and does not assess prose quality by totals.",
    ],
}
with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
    json.dump(receipt, stream, indent=2, ensure_ascii=False)
    stream.write("\n")
print(json.dumps({"receipt": OUTPUT.relative_to(ROOT).as_posix(), "passed": True,
                  "other_document_link_failures_at_check_time": receipt["other_document_link_failures_at_check_time"]}, indent=2))
