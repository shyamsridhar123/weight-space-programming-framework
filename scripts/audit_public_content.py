#!/usr/bin/env python3
"""Reject recognizable credentials and private user paths in publishable files.

This is a deterministic publication check, not a guarantee that arbitrary prose
contains no confidential information. Findings disclose locations, never values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

try:
    from scripts.verify_publication import publication_files
except ModuleNotFoundError:
    from verify_publication import publication_files


SECRETS = {
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "github_token": rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})",
    "openai_token": rb"\bsk-[A-Za-z0-9_-]{40,}",
    "aws_access_key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
}
PROFILE = re.compile(rb"(?:[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+|/(?:home|Users)/)([^\\/\s\"'<>]+)", re.I)
PLACEHOLDERS = {b"...", b"example", b"private-person", b"user", b"username", b"owner", b"your_user"}
PUBLIC_CI_LOGS = {
    "provenance/verification/runs/github-workflow-run-42336025906-job-128293033475.log",
    "provenance/verification/runs/hosted-ci-job-draft2.log",
}


def audit(root: Path) -> dict:
    root = Path(root).resolve()
    files = publication_files(root)
    findings = []
    for relative in files:
        path = root / relative
        if path.name.casefold() in {".env", "credentials.json", "auth.json", "id_rsa", "id_ed25519"}:
            findings.append({"path": relative, "kind": "credential_filename"})
        data = path.read_bytes()
        for kind, pattern in SECRETS.items():
            if re.search(pattern, data):
                findings.append({"path": relative, "kind": kind})
        if any(m[1].lower() not in PLACEHOLDERS and not (
                relative in PUBLIC_CI_LOGS and m[0] == b"/home/" + b"runner")
                for m in PROFILE.finditer(data)):
            findings.append({"path": relative, "kind": "private_user_path"})
    return {"schema": "wsp-public-content-audit/1", "ok": not findings,
            "files_scanned": len(files), "findings": findings,
            "scope": "recognizable_credential_formats_and_user_paths;_not_universal_confidentiality"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    receipt = audit(args.root)
    print(json.dumps(receipt))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
