"""Create a new source ZIP and bind its raw bytes to the committed draft.3 tree."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TAG = "v0.2.0-draft.3"
VERSION = "0.2.0.dev3"
WHEEL = ROOT / "dist/weight_space_programming_framework-0.2.0.dev3-py3-none-any.whl"
WHEEL_SHA256 = "9d018555b7b06a2de57e184227550e9f764160fa668bfe122a0fc3cceaedb89a"
ARCHIVE = ROOT / "dist" / f"weight-space-programming-framework-{TAG}-source.zip"
OUTPUT = ROOT / ".runs/source-archive-verification-draft3.json"
FINAL_RECEIPT = ROOT / ".runs/publication-final-verification-draft3-release.json"
STAGING_RECEIPT = ROOT / ".runs/staging-verification-draft3-release.json"
WHEEL_EVIDENCE = {
    "wheel_build_verification": ".runs/wheel-build-draft3.json",
    "wheel_build_log": ".runs/wheel-build-draft3.log",
    "wheel_install_smoke_verification": ".runs/wheel-install-smoke-draft3.json",
    "wheel_install_smoke_log": ".runs/wheel-install-smoke-draft3.log",
}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          check=True, timeout=120).stdout


def binding(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def main() -> None:
    for path in (ARCHIVE, OUTPUT):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Refusing to overwrite a release artifact: {path.name}")
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT:
        raise RuntimeError("Refusing to archive a parent repository")
    if git("rev-parse", "--show-object-format").decode().strip() != "sha1":
        raise RuntimeError("This archive verifier requires the repository's SHA-1 object format")
    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    if git("rev-parse", "refs/tags/" + TAG + "^{}").decode().strip() != head:
        raise RuntimeError("The draft.3 tag must identify local HEAD")
    entries = {}
    for entry in git("ls-tree", "-rz", "--full-tree", "HEAD").split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.decode().split()
        name = raw_name.decode("utf-8")
        if mode != "100644" or kind != "blob" or name in entries:
            raise RuntimeError(f"Unexpected committed entry: {name}")
        entries[name] = object_id
    manifest_raw = (ROOT / "MANIFEST.sha256").read_bytes()
    if git_blob(manifest_raw) != entries.get("MANIFEST.sha256"):
        raise RuntimeError("Local and committed manifests differ")
    manifest = {}
    folded = set()
    for line in manifest_raw.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None or match[2].casefold() in folded:
            raise RuntimeError("Manifest has an invalid or duplicate entry")
        manifest[match[2]] = match[1]
        folded.add(match[2].casefold())
    if set(entries) != set(manifest) | {"MANIFEST.sha256"}:
        raise RuntimeError("Committed tree and manifest membership differ")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    publication = read(ROOT / "provenance/publication-manifest.json")
    if (project["project"]["version"] != VERSION
            or publication["version"]["draft_tag"] != TAG
            or publication["version"]["python_distribution"] != VERSION):
        raise RuntimeError("Publication metadata does not identify draft.3")
    final = read(FINAL_RECEIPT)
    staging = read(STAGING_RECEIPT)
    if (final.get("ok") is not True or final.get("failures") != []
            or Path(final.get("root", "")).resolve() != ROOT):
        raise RuntimeError("A successful final publication receipt is required")
    if (staging.get("staged_tree") != tree or staging.get("tag") != TAG
            or staging.get("all_manifest_paths_staged") is not True
            or staging.get("all_staged_git_blobs_match_raw_bytes") is not True
            or staging.get("manifest") != binding(ROOT / "MANIFEST.sha256")
            or staging.get("publication_verification") != binding(FINAL_RECEIPT)):
        raise RuntimeError("Staging evidence does not bind this committed tree and final verification")
    wheel_binding = binding(WHEEL)
    if wheel_binding["sha256"] != WHEEL_SHA256:
        raise RuntimeError("Wheel bytes differ from the fresh draft.3 validation")
    wheel_evidence = {key: binding(ROOT / relative) for key, relative in WHEEL_EVIDENCE.items()}
    build = read(ROOT / WHEEL_EVIDENCE["wheel_build_verification"])
    smoke = read(ROOT / WHEEL_EVIDENCE["wheel_install_smoke_verification"])
    if (build.get("status") != "passed" or build.get("exit_code") != 0
            or build.get("build_isolation") is not True
            or build.get("wheel_python_members_match_sources") is not True
            or build.get("wheel_readme_matches_source_text") is not True
            or smoke.get("status") != "passed" or smoke.get("exit_code") != 0
            or smoke.get("all_commands_completed") is not True
            or smoke.get("fresh_virtual_environment") is not True
            or smoke.get("wheel_unchanged") is not True
            or smoke.get("build_receipt") != wheel_evidence["wheel_build_verification"]):
        raise RuntimeError("Successful fresh wheel build and external installation evidence is required")
    if len(build.get("wheels", [])) != 1 or build["wheels"][0] != smoke.get("wheel"):
        raise RuntimeError("Build and installation receipts identify different wheels")
    if ({key: smoke["wheel"].get(key) for key in wheel_binding} != wheel_binding
            or smoke["wheel"].get("version") != VERSION):
        raise RuntimeError("Wheel receipts do not identify the release wheel")
    for report, log_key in ((build, "wheel_build_log"), (smoke, "wheel_install_smoke_log")):
        if report.get("log_sha256") != wheel_evidence[log_key]["sha256"]:
            raise RuntimeError("Wheel log digest differs from its receipt")
        for group in ("source_inputs", "example_inputs"):
            if (report.get(group + "_unchanged") is not True
                    or report[group + "_before"] != report[group + "_after"]):
                raise RuntimeError("Wheel validation did not preserve its source inputs")
            for name, recorded in report[group + "_after"].items():
                actual = binding(ROOT / name)
                if {key: actual[key] for key in ("bytes", "sha256")} != recorded:
                    raise RuntimeError(f"Wheel validation input has changed: {name}")
    archived_wheel_evidence = {}
    for key, relative in WHEEL_EVIDENCE.items():
        archived = "provenance/verification/runs/" + Path(relative).name
        archived_wheel_evidence[key] = binding(ROOT / archived)
        if (archived_wheel_evidence[key]["sha256"] != wheel_evidence[key]["sha256"]
                or manifest.get(archived) != wheel_evidence[key]["sha256"]):
            raise RuntimeError("The committed source archive must preserve the wheel evidence bytes")
    input_bindings = {
        "manifest": binding(ROOT / "MANIFEST.sha256"),
        "publication_verification": binding(FINAL_RECEIPT),
        "git_staging_verification": binding(STAGING_RECEIPT),
        "wheel": wheel_binding,
        **wheel_evidence,
    }
    prefix = f"weight-space-programming-framework-{TAG}/"
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open("xb") as stream:
        subprocess.run(["git", "archive", "--format=zip", "--prefix=" + prefix, head],
                       cwd=ROOT, stdout=stream, stderr=subprocess.PIPE, check=True, timeout=300)
    with zipfile.ZipFile(ARCHIVE) as snapshot:
        members = [item.filename for item in snapshot.infolist() if not item.is_dir()]
        if (len(members) != len(set(members))
                or set(members) != {prefix + name for name in entries}):
            raise RuntimeError("Source ZIP membership differs from the committed tree")
        for name, object_id in entries.items():
            raw = snapshot.read(prefix + name)
            if git_blob(raw) != object_id or raw != (ROOT / name).read_bytes():
                raise RuntimeError(f"Source ZIP bytes differ from the commit or checkout: {name}")
            if name != "MANIFEST.sha256" and hashlib.sha256(raw).hexdigest() != manifest[name]:
                raise RuntimeError(f"Source ZIP bytes differ from the manifest: {name}")
    if (git("rev-parse", "HEAD").decode().strip() != head
            or git("rev-parse", "refs/tags/" + TAG + "^{}").decode().strip() != head):
        raise RuntimeError("Local HEAD or the draft tag changed during archive verification")
    for expected in input_bindings.values():
        if binding(ROOT / expected["path"]) != expected:
            raise RuntimeError("A release input changed during archive verification")
    receipt = {
        "schema": "wspf-source-release-verification/2",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": head, "tree": tree, "tag": TAG, "python_distribution": VERSION,
        "source_zip": binding(ARCHIVE),
        **input_bindings,
        "archived_wheel_evidence": archived_wheel_evidence,
        "exact_git_tree_membership": True,
        "all_archive_file_bytes_match_commit_and_manifest": True,
        "staged_tree_matches_committed_tree": True,
        "scope": "Exact committed source and release-asset identities, with fresh wheel evidence bindings. Scientific and runtime scopes are documented separately.",
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
