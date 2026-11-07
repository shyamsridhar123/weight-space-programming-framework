"""Small, explicit provenance helpers for repeatable, non-overwriting studies."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

from .model import digest

ROOT = Path(__file__).resolve().parents[2]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, value: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # "x" is intentional: result files are evidence, never a mutable cache.
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return file_hash(target)


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(directory: str | Path, manifest: dict[str, str]) -> None:
    """Verify recorded files without permitting paths to escape an evidence root."""
    root = Path(directory).resolve()
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("An evidence manifest must name its recorded files")
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Evidence artifact is missing, altered, or outside its root: {relative}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "matplotlib", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages}


def execution_sources() -> list[Path]:
    paths = list((ROOT / "rlcd/weight_space").glob("*.py"))
    # Include all game semantics, not just the directly imported entry points.
    paths += list((ROOT / "game/src/empires").rglob("*.ts"))
    paths += list((ROOT / "game/src/empires/weights").rglob("*.json"))
    paths += [ROOT / name for name in (
        "scripts/weight-space-empires.ts", "scripts/run_weight_space_study.ps1",
        "package-lock.json", "requirements.lock.txt")]
    return sorted(set(paths))


def freeze(protocol_path: str | Path, out: str | Path) -> dict[str, Any]:
    """Capture code and protocol before generating any study outcomes."""
    out, protocol_path = Path(out), Path(protocol_path).resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = execution_sources()
    copies = {}
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        destination = out / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(path.read_bytes())
        copies[relative] = file_hash(path)
    with (out / "protocol.json").open("xb") as handle:
        handle.write(protocol_path.read_bytes())
    payload = {"schema": "wsp-freeze/1", "created_at": utc_now(),
               "protocol_sha256": file_hash(protocol_path), "source_sha256": copies,
               "environment": environment(),
               "scope": "local pre-result snapshot; not an external preregistration or tamper-proof timestamp"}
    receipt = payload | {"sha256": digest(payload)}
    write_json(out / "receipt.json", receipt)
    return receipt


def verify_freeze(protocol_path: str | Path, receipt_path: str | Path) -> dict[str, Any]:
    receipt_path = Path(receipt_path)
    receipt = read_json(receipt_path)
    payload = {k: v for k, v in receipt.items() if k != "sha256"}
    if receipt.get("schema") != "wsp-freeze/1" or receipt.get("sha256") != digest(payload):
        raise ValueError("Malformed or altered freeze receipt")
    if file_hash(protocol_path) != receipt["protocol_sha256"]:
        raise ValueError("Protocol differs from pre-result freeze")
    if file_hash(receipt_path.parent / "protocol.json") != receipt["protocol_sha256"]:
        raise ValueError("Frozen protocol snapshot is missing or altered")
    current = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in execution_sources()}
    if current != receipt["source_sha256"]:
        changed = sorted(k for k in set(current) | set(receipt["source_sha256"])
                         if current.get(k) != receipt["source_sha256"].get(k))
        raise ValueError(f"Execution sources differ from freeze: {changed}")
    for relative, expected in receipt["source_sha256"].items():
        if file_hash(receipt_path.parent / "sources" / relative) != expected:
            raise ValueError(f"Frozen source snapshot changed: {relative}")
    return receipt


def bootstrap_mean(values: list[float], *, seed: int, resamples: int) -> dict[str, Any]:
    """Percentile interval, resampling whole independent experimental units."""
    import numpy as np
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or not len(values_array) or not np.isfinite(values_array).all():
        raise ValueError("Bootstrap requires finite, nonempty experimental units")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("A positive integer resample count is required")
    rng = np.random.default_rng(seed)
    means = values_array[rng.integers(0, len(values_array), (resamples, len(values_array)))].mean(axis=1)
    return {"n": len(values), "mean": float(values_array.mean()),
            "ci95": list(map(float, np.quantile(means, [.025, .975]))),
            "positive_units": int((values_array > 0).sum()), "bootstrap_seed": seed,
            "bootstrap_resamples": resamples, "method": "percentile bootstrap of paired unit differences"}
