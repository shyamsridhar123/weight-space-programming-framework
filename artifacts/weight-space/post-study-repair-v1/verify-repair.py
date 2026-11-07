"""Record the receipt-only repair without modifying or rerunning the study."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

from rlcd.weight_space.evidence import execution_sources, file_hash, read_json, utc_now, write_json


def main() -> None:
    if (OUT / "receipt.json").exists():
        raise FileExistsError("Repair evidence already exists; preserve this record")
    before = read_json(OUT / "before.json")
    study = Path(before["study_root"])
    archive = study / "freeze/sources"
    current = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in execution_sources()}
    changed = [name for name in sorted(set(current) | set(before["runtime_source_sha256"]))
               if current.get(name) != before["runtime_source_sha256"].get(name)]
    if changed != ["rlcd/weight_space/__main__.py"]:
        raise ValueError(f"Unexpected post-study runtime differences: {changed}")
    diff = []
    after_files = {}
    for name, old_hash in before["files_sha256"].items():
        previous = OUT / "before" / name
        if file_hash(previous) != old_hash:
            raise ValueError(f"Before snapshot changed: {name}")
        after_files[name] = file_hash(ROOT / name)
        diff.extend(difflib.unified_diff(previous.read_text(encoding="utf-8").splitlines(True),
                                        (ROOT / name).read_text(encoding="utf-8").splitlines(True),
                                        fromfile=f"before/{name}", tofile=f"after/{name}"))
        target = OUT / "after" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write((ROOT / name).read_bytes())
    with (OUT / "repair.patch").open("x", encoding="utf-8", newline="\n") as handle:
        handle.writelines(diff)

    diagnostic = ROOT / "artifacts/weight-space/verification-v1/certificate-diagnostic"
    model_args = [str(diagnostic / "base.json"), str(diagnostic / "candidate.json")]
    verify_freeze = ["-m", "rlcd.weight_space", "verify-freeze", "--protocol",
                     str(study / "freeze/protocol.json"), "--freeze", str(study / "freeze/receipt.json")]
    steps = [
        ("focused-regression", ROOT, ["-m", "pytest", "-q", "tests/test_weight_space.py",
          "tests/test_weight_space_generic.py", "tests/test_weight_space_torch.py",
          "tests/test_weight_space_boundaries.py"], 0),
        ("current-valid-receipt", ROOT, ["-m", "rlcd.weight_space", "verify", *model_args,
          str(diagnostic / "valid.json")], 0),
        ("current-altered-receipt", ROOT, ["-m", "rlcd.weight_space", "verify", *model_args,
          str(diagnostic / "altered.json")], 2),
        ("archived-altered-receipt", archive, ["-m", "rlcd.weight_space", "verify", *model_args,
          str(diagnostic / "altered.json")], 0),
        ("archived-freeze", archive, verify_freeze, 0),
        ("current-freeze-mismatch", ROOT, verify_freeze, 2),
    ]
    results = []
    for label, cwd, arguments, expected in steps:
        command = [sys.executable, *arguments]
        start = time.monotonic()
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                                encoding="utf-8", errors="replace", timeout=180)
        for stream in ("stdout", "stderr"):
            with (OUT / f"{label}.{stream}.log").open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(getattr(result, stream))
        record = {"name": label, "command": command, "cwd": str(cwd),
                  "exit_code": result.returncode, "expected_exit_code": expected,
                  "passed": result.returncode == expected,
                  "elapsed_seconds": time.monotonic() - start}
        results.append(record)
        print(json.dumps(record), flush=True)
    passed = all(row["passed"] for row in results)
    write_json(OUT / "receipt.json", {
        "schema": "wsp-post-study-repair/1", "created_at": utc_now(), "passed": passed,
        "study_root": str(study), "freeze_sha256": before["freeze_sha256"],
        "protocol_sha256": before["protocol_sha256"], "changed_runtime_sources": changed,
        "before_files_sha256": before["files_sha256"], "after_files_sha256": after_files,
        "verification": results,
        "scope": "Only generic CLI certificate comparison and its regression test changed. The study retains the original runtime, including its demonstrated comparison defect. No training, selection, simulations, mathematical checker edits, or frozen-evidence rewrites occurred.",
        "study_verification": "Archived runtime passes its exact source freeze. Current runtime correctly rejects the old freeze because __main__.py changed; this failure is expected and retained.",
        "artifacts": {path.relative_to(OUT).as_posix(): file_hash(path)
                      for path in sorted(OUT.rglob("*")) if path.is_file()},
    })
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
