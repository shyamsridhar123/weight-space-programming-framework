"""Preserve indexed document bytes before the draft.3 narrative revision."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE = "93e8a9b7f2cfbbffab6c7d10f831b2cde4b69176"
INDEX = ROOT / "provenance/source-files.json"
RECEIPT = ROOT / ".runs/docs-preservation-draft3.json"
NAMES = [
    "docs/PROJECT_MEMORY.md",
    "docs/weight-space/README.md",
    "docs/weight-space/design.md",
    "docs/weight-space/requirements.md",
    "docs/weight-space/research.md",
    "docs/weight-space/results.md",
    *[f"docs/weight-space/v2/{name}.md" for name in (
        "formal", "limitations-and-next-steps", "methods", "requirements",
        "results", "source-review", "workflow",
    )],
    "formal/weight-space/README.md",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def preserve(relative: str, data: bytes) -> dict:
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite archived bytes: {relative}")
    else:
        target.write_bytes(data)
    return {"path": relative, "bytes": len(data), "sha256": sha(data)}


assert not RECEIPT.exists(), "Preservation receipt already exists"
assert subprocess.check_output(
    ["git", "rev-parse", "v0.2.0-draft.2^{}"], cwd=ROOT, text=True
).strip() == BASE
before = INDEX.read_bytes()
assert before == git_bytes("provenance/source-files.json"), "Unexpected source-index edit"
index = json.loads(before)
original = json.loads(before)
by_name = {r["path"]: r for r in index["files"]}
records = []
for name in NAMES:
    record = by_name[name]
    source = git_bytes(name)
    previous_path = record["preserved_path"]
    if sha(source) == record["sha256"] and len(source) == record["bytes"]:
        destination = "provenance/original-documents/" + name + ".txt"
        if name == "docs/PROJECT_MEMORY.md":
            destination = "provenance/original-documents/docs/PROJECT_MEMORY.md"
        archived = preserve(destination, source)
        record["preserved_path"] = destination
        records.append({"original_path": name, "previous_preserved_path": previous_path,
                        "preserved_original": archived, "source_index_remapped": True})
    else:
        assert name == "docs/weight-space/design.md", name
        assert previous_path != name
        original_bytes = (ROOT / previous_path).read_bytes()
        assert len(original_bytes) == record["bytes"] and sha(original_bytes) == record["sha256"]
        previous_edition = preserve(
            "provenance/previous-publications/v0.2.0-draft.2/" + name + ".txt", source
        )
        records.append({"original_path": name, "previous_preserved_path": previous_path,
                        "preserved_original": {"path": previous_path, "bytes": len(original_bytes),
                                               "sha256": sha(original_bytes)},
                        "source_index_remapped": False, "previous_published_document": previous_edition,
                        "reason": "The draft.2 document already replaced the original Mermaid fence with image links."})

for old, new in zip(original["files"], index["files"], strict=True):
    assert {k: v for k, v in old.items() if k != "preserved_path"} == {
        k: v for k, v in new.items() if k != "preserved_path"
    }
assert {k: v for k, v in original.items() if k != "files"} == {
    k: v for k, v in index.items() if k != "files"
}
INDEX.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
after = INDEX.read_bytes()
receipt = {
    "schema": "wspf-document-preservation/1",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_publication_commit": BASE,
    "source_publication_tag": "v0.2.0-draft.2",
    "source_index_before": {"path": "provenance/source-files.json", "bytes": len(before), "sha256": sha(before)},
    "source_index_after": {"path": "provenance/source-files.json", "bytes": len(after), "sha256": sha(after)},
    "source_index_change_scope": "Only preserved_path values changed; original paths, hashes, byte sizes, ordering, and metadata remain unchanged.",
    "documents": records,
    "scope": "Exact original documentation preservation and one prior publication rendering; no scientific implementation, proof source, frozen input, or original RLCD file changed.",
}
RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({"receipt": RECEIPT.relative_to(ROOT).as_posix(),
                  "all_original_document_hashes_verified": True,
                  "source_index_after_sha256": sha(after)}, indent=2))
