#!/usr/bin/env python3
"""Publish disclosed cost-record derivatives while retaining every original privately.

Only absolute build paths in two runtime strings change. Measurements, package
identities and frozen execution inputs remain unchanged. No study is executed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INDEX = "artifacts/research/serving-cost-supplement/process-index.json"
ORIGINAL_INDEX_SHA256 = "8f1b2b4950b53ddc67496346442cd27751e155defc88a2c15aede59d4faccd77"
DERIVATIVE = "artifacts/research/serving-cost-supplement/publication-derivative.json"
LEDGER = "provenance/public-sanitization.json"


def identity(data: bytes) -> dict:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sanitize_config(value: str, *, numpy: bool) -> tuple[str, list[int]]:
    """Retain build metadata around each explicit absolute-path replacement."""
    lines = value.splitlines(keepends=True)
    changed = []
    for number, line in enumerate(lines):
        if numpy:
            match = re.fullmatch(r"(\s*(?:include directory|lib directory|pc file directory|path):\s*)([^\r\n]*)(\r?\n)?", line)
            if match and re.match(r"[\"']?(?:[A-Za-z]:[\\/]|/)", match[2]):
                lines[number] = match[1] + "<redacted-build-path>" + (match[3] or "")
        else:
            lines[number] = re.sub(r"(CXX_COMPILER=)[A-Za-z]:[\\/][^,\r\n]*(?=,|\r?$)",
                                   r"\1<redacted-compiler-path>", line)
        if lines[number] != line:
            changed.append(number + 1)
    return "".join(lines), changed


def self_check() -> None:
    source = "  version: 1.2\n  path: C:\\build\\compiler.exe\n  path: cl\n"
    public, rows = sanitize_config(source, numpy=True)
    assert public == "  version: 1.2\n  path: <redacted-build-path>\n  path: cl\n" and rows == [2]
    source = " - Build settings: BLAS=mkl, CXX_COMPILER=C:/build/compiler.exe, USE_CUDA=OFF\n"
    public, rows = sanitize_config(source, numpy=False)
    assert public == " - Build settings: BLAS=mkl, CXX_COMPILER=<redacted-compiler-path>, USE_CUDA=OFF\n" and rows == [1]
    assert sanitize_config(public, numpy=False) == (public, [])
    assert sanitize_config("  path: /build/compiler\n", numpy=True)[1] == [1]
    assert sanitize_config("  pc file directory: 'C:/build/pkgconfig'\r\n", numpy=True) == (
        "  pc file directory: <redacted-build-path>\r\n", [1])
    print("build-path replacement checks passed; no publication or study mutation")


def publish(private_originals: Path) -> None:
    private_originals = private_originals.resolve()
    if private_originals.is_relative_to(ROOT) or private_originals.exists():
        raise ValueError("private-originals must be a new directory outside the public checkout")
    raw_index = (ROOT / INDEX).read_bytes()
    if identity(raw_index)["sha256"] != ORIGINAL_INDEX_SHA256:
        raise ValueError("original process index identity differs; no automatic retry")
    records = json.loads(raw_index)
    if len(records) != 96 or len({item["path"] for item in records}) != 96:
        raise ValueError("expected 96 distinct recorded assignments")
    if (ROOT / DERIVATIVE).exists():
        raise FileExistsError(DERIVATIVE)
    prepared = []
    for entry in records:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT) or path.is_symlink():
            raise ValueError("record path leaves public checkout")
        raw = path.read_bytes()
        if identity(raw) != entry["identity"]:
            raise ValueError("record original identity differs: " + entry["path"])
        original = json.loads(raw)
        public = deepcopy(original)
        changes = []
        for field, numpy in [("numpy_build_config", True), ("torch_build_config", False)]:
            value, lines = sanitize_config(original["runtime"][field], numpy=numpy)
            if not lines:
                raise ValueError("expected build-path disclosure: " + field)
            public["runtime"][field] = value
            changes.append({"json_pointer": "/runtime/" + field, "lines": lines})
        stripped_original = deepcopy(original)
        stripped_public = deepcopy(public)
        for field in ["numpy_build_config", "torch_build_config"]:
            del stripped_original["runtime"][field]
            del stripped_public["runtime"][field]
        if stripped_original != stripped_public:
            raise ValueError("non-runtime measurement changed")
        prepared.append((entry, raw, encode(public), changes))
    ledger_raw = (ROOT / LEDGER).read_bytes()
    ledger = json.loads(ledger_raw)
    disclosed = {item["path"] for item in ledger["files"]}
    if disclosed.intersection([INDEX] + [entry["path"] for entry in records]):
        raise ValueError("derivative paths already occur in sanitation ledger")
    private_originals.mkdir(parents=True)
    for name, raw in [(INDEX, raw_index), (LEDGER, ledger_raw)] + [(entry["path"], raw) for entry, raw, _, _ in prepared]:
        target = private_originals / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        if target.read_bytes() != raw:
            raise ValueError("private original copy differs")
    new_index = []
    correspondence = []
    for entry, raw, public, changes in prepared:
        public_identity = identity(public)
        new_index.append({**entry, "original_identity": entry["identity"], "identity": public_identity,
                          "publication_derivative": True})
        correspondence.append({"path": entry["path"], "original": identity(raw), "public": public_identity,
                               "changes": changes, "non_build_config_payload_equal": True})
        (ROOT / entry["path"]).write_bytes(public)
        ledger["files"].append({"path": entry["path"], "original": identity(raw), "public": public_identity,
                                "reasons": ["absolute-build-path-redaction-only"]})
    public_index = encode(new_index)
    (ROOT / INDEX).write_bytes(public_index)
    ledger["files"].append({"path": INDEX, "original": identity(raw_index), "public": identity(public_index),
                            "reasons": ["disclosed-public-record-identities-with-original-bindings"]})
    ledger["research_originals_policy"] = "New research originals are retained privately by raw identity; source_commit describes the imported historical source inventory."
    (ROOT / LEDGER).write_bytes(encode(ledger))
    derivative = {"schema": "wsp-serving-cost-publication-derivative/1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "original_index": identity(raw_index), "public_index": identity(public_index), "index_path": INDEX,
                  "sanitizer": identity(Path(__file__).read_bytes()), "records": correspondence,
                  "preserved_inputs": ["execution-lock.json", "package-manifest.json", "summary.json", "io-measurements.json"],
                  "interpretation": "Public build-path derivatives; original record bytes are not relabeled as published bytes. No measurement rerun, numerical change, or execution-lock rewrite."}
    (ROOT / DERIVATIVE).write_bytes(encode(derivative))
    for entry in new_index:
        assert identity((ROOT / entry["path"]).read_bytes()) == entry["identity"]
    print(json.dumps({"records": len(prepared), "original_index": identity(raw_index),
                      "public_index": identity(public_index), "derivative": identity((ROOT / DERIVATIVE).read_bytes())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-originals", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.private_originals:
        publish(args.private_originals)
    else:
        parser.error("choose --self-check or --private-originals")
