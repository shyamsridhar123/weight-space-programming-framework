#!/usr/bin/env python3
"""Recheck the preserved proof snapshot; put new evidence in .proof-runs only.

Preparation is documented in PROOFS.md. This runner does not install Lean,
update dependencies, regenerate Lean sources, or run empirical studies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "formal/weight-space"
HISTORICAL_RECEIPT = "formal/weight-space/verification-v2/actual/verification.json"
HISTORICAL_RECEIPT_SHA256 = "7d3435dac048bd2ae4c28b90e7e9a035c5134cd0edee75ae200b32c2d68a9eac"
# Historical receipts keep their original names and hashes. Only these document
# lookups use archive copies after the live guides were revised for publication.
PRESERVED_DOCUMENT_PATHS = {
    "docs/weight-space/v2/formal.md": "provenance/original-documents/docs/weight-space/v2/formal.md.txt",
    "formal/weight-space/README.md": "provenance/original-documents/formal/weight-space/README.md.txt",
}
DRIVER_FILES = (
    "PROOFS.md",
    "scripts/check_proofs.py",
    "scripts/check_proofs.ps1",
    "scripts/check_proofs.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def repository_file(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        raise RuntimeError(f"Reference escapes the repository: {relative}")
    return path


def verify_hashes(expected: dict[str, str]) -> dict[str, str]:
    actual = {}
    for relative, expected_hash in expected.items():
        preserved_relative = PRESERVED_DOCUMENT_PATHS.get(relative, relative)
        actual[relative] = sha256(repository_file(preserved_relative))
        if actual[relative] != expected_hash:
            raise RuntimeError(f"Preserved proof input differs: {relative} (checked {preserved_relative})")
    return actual


def historical_inputs() -> tuple[dict, dict[str, str]]:
    expected = {HISTORICAL_RECEIPT: HISTORICAL_RECEIPT_SHA256}
    verify_hashes(expected)
    receipt = read_json(ROOT / HISTORICAL_RECEIPT)
    expected.update(receipt["source_sha256"])
    verify_hashes(expected)
    snapshot = read_json(repository_file(receipt["preserved_v1"]["snapshot"]))
    for relative, digest in snapshot["sha256"].items():
        if relative in expected and expected[relative] != digest:
            raise RuntimeError(f"Conflicting saved identities: {relative}")
        expected[relative] = digest
    receipt_directory = Path(HISTORICAL_RECEIPT).parent
    for filename, digest in receipt["log_sha256"].items():
        expected[(receipt_directory / filename).as_posix()] = digest
    return receipt, expected


def capture(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments, cwd=PROJECT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {arguments!r}\n{detail}")
    return result.stdout.strip()


def check_dependencies(manifest: dict) -> dict[str, str]:
    revisions = {}
    for package in manifest["packages"]:
        directory = PROJECT / manifest["packagesDir"] / package["name"]
        if not (directory / ".git").exists():
            raise RuntimeError(
                f"Missing dependency {package['name']}; complete the preparation in PROOFS.md."
            )
        revision = capture(["git", "-C", str(directory), "rev-parse", "HEAD"])
        if revision != package["rev"]:
            raise RuntimeError(f"Dependency {package['name']} is not at its locked commit.")
        capture(["git", "-C", str(directory), "diff", "--quiet", "HEAD", "--"])
        revisions[package["name"]] = revision
    return revisions


def run_logged(arguments: list[str], log: Path) -> dict:
    print(f"Running: {' '.join(arguments)}", flush=True)
    with log.open("x", encoding="utf-8", newline="\n") as stream:
        with subprocess.Popen(
            arguments, cwd=PROJECT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        ) as process:
            if process.stdout is None:
                raise RuntimeError("The proof command did not expose its output stream.")
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
    if code:
        raise RuntimeError(f"Proof command failed ({code}); see {log}")
    return {"arguments": arguments, "exit_code": code, "log": log.name}


def read_audit(log: Path, historical: dict) -> dict:
    text = log.read_text(encoding="utf-8")
    counts = {}
    checks = (
        ("all", "V2 full transitive axiom audit passed", "audited_declarations", "audited_theorem_declarations"),
        ("new", "V2 new namespaces", "audited_new_declarations", "audited_new_theorem_declarations"),
    )
    for label, prefix, declaration_key, theorem_key in checks:
        match = re.search(re.escape(prefix) + r": (\d+) declarations; (\d+) theorem declarations\.", text)
        if match is None:
            raise RuntimeError(f"Required audit output is missing: {prefix}")
        actual = {"declarations": int(match[1]), "theorem_declarations": int(match[2])}
        expected = {"declarations": historical[declaration_key], "theorem_declarations": historical[theorem_key]}
        if actual != expected:
            raise RuntimeError(f"Audit count differs from the preserved snapshot: {label}")
        counts[label] = actual
    counts["namespaces"] = {}
    for namespace, expected in historical["audited_new_namespaces"].items():
        match = re.search(
            r"V2 namespace " + re.escape(namespace) + r": (\d+) declarations; (\d+) theorem declarations\.",
            text,
        )
        if match is None:
            raise RuntimeError(f"Required namespace audit output is missing: {namespace}")
        actual = {"declarations": int(match[1]), "theorem_declarations": int(match[2])}
        if actual != expected:
            raise RuntimeError(f"Audit count differs from the preserved snapshot: {namespace}")
        counts["namespaces"][namespace] = actual
    match = re.search(r"V2 observed axioms:\s*\[([^\]]*)\]\.", text)
    if match is None:
        raise RuntimeError("Required observed-axiom audit output is missing.")
    observed = {name.strip() for name in match[1].split(",") if name.strip()}
    if observed != set(historical["observed_axioms"]):
        raise RuntimeError("The observed axiom set differs from the preserved snapshot.")
    if not observed <= set(historical["allowed_axioms"]):
        raise RuntimeError("The audit reported a disallowed axiom.")
    counts["observed_axioms"] = sorted(observed)
    return counts


def main() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or later is required.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="A new directory under the repository's .proof-runs/.")
    arguments = parser.parse_args()
    requested = Path(arguments.output).expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise RuntimeError(f"Output already exists; choose a fresh directory: {requested}")
    output = requested.resolve()
    run_root = ROOT / ".proof-runs"
    if run_root.resolve() != run_root or run_root not in output.parents:
        raise RuntimeError("Output must be a new subdirectory of this repository's .proof-runs/.")
    for command in ("git", "lake", "lean"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Missing command on PATH: {command}; see PROOFS.md.")

    started_at = datetime.now(timezone.utc).isoformat()
    historical, expected = historical_inputs()
    source_hashes = verify_hashes(expected)
    driver_hashes = {path: sha256(ROOT / path) for path in DRIVER_FILES}
    toolchain = (PROJECT / "lean-toolchain").read_text(encoding="utf-8-sig").strip()
    if toolchain != "leanprover/lean4:v4.24.0":
        raise RuntimeError("The preserved Lean toolchain pin is required.")
    manifest = read_json(PROJECT / "lake-manifest.json")
    dependency_revisions = check_dependencies(manifest)
    lean_version = capture(["lake", "env", "lean", "--version"])
    if re.search(r"version 4\.24\.0,", lean_version) is None:
        raise RuntimeError(f"Wrong Lean version: {lean_version}")
    lean_prefix = Path(capture(["lake", "env", "lean", "--print-prefix"]))
    lean_binary = lean_prefix / "bin" / ("lean.exe" if sys.platform == "win32" else "lean")
    lean_binary_hash = sha256(lean_binary)

    output.mkdir(parents=True, exist_ok=False)
    commands = []
    commands.append(run_logged(["lake", "build", "WeightSpace.GenericAudit"], output / "build.log"))
    commands.append(run_logged(["lake", "env", "lean", "WeightSpace/GenericAudit.lean"], output / "axiom-audit.log"))
    audit = read_audit(output / "axiom-audit.log", historical)
    verify_hashes(source_hashes)
    verify_hashes(driver_hashes)
    if check_dependencies(manifest) != dependency_revisions:
        raise RuntimeError("Dependency revisions changed during verification.")
    if sha256(lean_binary) != lean_binary_hash:
        raise RuntimeError("The selected Lean binary changed during verification.")

    receipt = {
        "schema": "wsp-standalone-proof-verification/1",
        "started_at": started_at,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": 0,
        "verification_mode": "Pinned proof snapshot build and transitive axiom audit; build caches may be reused.",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "lean_toolchain": toolchain,
        "lean_version_output": lean_version,
        "lean_executable_sha256": lean_binary_hash,
        "dependency_revisions": dependency_revisions,
        "dependency_tracked_sources_unmodified_before_and_after": True,
        "commands": commands,
        "audit": audit,
        "historical_receipt": {"path": HISTORICAL_RECEIPT, "sha256": HISTORICAL_RECEIPT_SHA256},
        "preserved_document_paths": PRESERVED_DOCUMENT_PATHS,
        "preserved_proof_files_checked": len(source_hashes),
        "source_sha256": source_hashes,
        "verification_driver_sha256": driver_hashes,
        "all_recorded_sources_unchanged_during_run": True,
        "log_sha256": {name: sha256(output / name) for name in ("build.log", "axiom-audit.log")},
        "scope_limit": "Does not rerun empirical studies, regenerate RTS definitions, or recursively validate every external reference in older receipts.",
    }
    with (output / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Proof verification passed. New evidence: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Proof verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
