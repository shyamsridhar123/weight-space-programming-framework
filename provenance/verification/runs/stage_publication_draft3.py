"""Stage the draft.3 manifest paths and verify each raw Git blob before commit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TAG = "v0.2.0-draft.3"
VERSION = "0.2.0.dev3"
OUTPUT = ROOT / ".runs/staging-verification-draft3-release.json"
PATHLIST = ROOT / ".runs/publication-paths-draft3-release.nul"
FINAL_RECEIPT = ROOT / ".runs/publication-final-verification-draft3-release.json"


def binding(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          check=True, timeout=120).stdout


def main() -> None:
    for path in (OUTPUT, PATHLIST):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Refusing to overwrite staging evidence: {path.name}")
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT:
        raise RuntimeError("Refusing to stage files from a parent repository")
    if git("rev-parse", "--show-object-format").decode().strip() != "sha1":
        raise RuntimeError("This staging verifier requires the repository's SHA-1 object format")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    publication = json.loads((ROOT / "provenance/publication-manifest.json").read_text(encoding="utf-8"))
    if (project["project"]["version"] != VERSION
            or publication["version"]["draft_tag"] != TAG
            or publication["version"]["python_distribution"] != VERSION):
        raise RuntimeError("Publication metadata does not identify draft.3")
    final_binding = binding(FINAL_RECEIPT)
    final = json.loads(FINAL_RECEIPT.read_text(encoding="utf-8"))
    if (final.get("ok") is not True or final.get("failures") != []
            or Path(final.get("root", "")).resolve() != ROOT):
        raise RuntimeError("A successful final publication receipt for this repository is required")
    spec = importlib.util.spec_from_file_location("publication_verifier", ROOT / "scripts/verify_publication.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verification = verifier.verify(ROOT)
    if verification.get("ok") is not True:
        raise RuntimeError(json.dumps(verification["failures"][:10]))
    manifest_binding = binding(ROOT / "MANIFEST.sha256")
    files = sorted(verifier.publication_files(ROOT) + ["MANIFEST.sha256"])
    if len(files) != len(set(files)):
        raise RuntimeError("Publication path selection contains duplicates")
    with PATHLIST.open("xb") as stream:
        stream.write(b"\0".join(name.encode("utf-8") for name in files) + b"\0")
    git("--literal-pathspecs", "add", "-f", "--pathspec-from-file=" + str(PATHLIST), "--pathspec-file-nul")
    staged_tree = git("write-tree").decode().strip()
    staged = {}
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.decode().split()
        name = raw_name.decode("utf-8")
        if stage != "0" or mode != "100644" or name in staged:
            raise RuntimeError(f"Unexpected staged entry: {name}")
        staged[name] = object_id
    if set(staged) != set(files):
        raise RuntimeError(json.dumps({"missing": sorted(set(files) - set(staged)),
                                      "extra": sorted(set(staged) - set(files))}))
    for name in files:
        raw = (ROOT / name).read_bytes()
        expected = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if staged[name] != expected:
            raise RuntimeError(f"Staged Git blob differs from raw file bytes: {name}")
    if binding(ROOT / "MANIFEST.sha256") != manifest_binding or binding(FINAL_RECEIPT) != final_binding:
        raise RuntimeError("Final publication inputs changed during staging")
    if git("write-tree").decode().strip() != staged_tree:
        raise RuntimeError("The Git index changed during staging verification")
    receipt = {
        "schema": "wspf-git-staging-verification/2",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "tag": TAG,
        "python_distribution": VERSION,
        "git_object_format": "sha1",
        "staged_tree": staged_tree,
        "all_manifest_paths_staged": True,
        "all_staged_git_blobs_match_raw_bytes": True,
        "manifest_sha256": manifest_binding["sha256"],
        "manifest": manifest_binding,
        "publication_verification": final_binding,
        "publication_check": verification,
        "scope": "Exact publication membership, file bytes, and staged-tree identity; no scientific or test-total quality claim.",
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "publication_check"}, indent=2))


if __name__ == "__main__":
    main()
