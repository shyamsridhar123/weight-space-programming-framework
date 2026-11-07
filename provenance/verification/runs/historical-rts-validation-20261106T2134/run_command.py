from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RUN = Path(__file__).resolve().parent
STAGING = RUN.parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def file_record(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    label = sys.argv[1]
    args = sys.argv[2:]
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in label):
        raise ValueError("Invalid run label")
    environment = os.environ.copy()
    changes = {
        "TEMP": str(RUN / "tmp"),
        "TMP": str(RUN / "tmp"),
        "TMPDIR": str(RUN / "tmp"),
        "npm_config_cache": str(RUN / "npm-cache"),
        "npm_config_update_notifier": "false",
        "npm_config_logs_max": "0",
        "npm_config_audit": "false",
        "npm_config_fund": "false",
        "NODE_PATH": "",
    }
    environment.update(changes)
    output = RUN / (label + ".stdout.log")
    errors = RUN / (label + ".stderr.log")
    receipt = {
        "label": label,
        "argv": args,
        "command_line_windows": subprocess.list2cmdline(args),
        "working_directory": str(STAGING),
        "environment_overrides": changes,
        "started_at": now(),
        "files_before": {name: file_record(STAGING / name) for name in ("package.json", "package-lock.json")},
    }
    started = time.monotonic()
    with output.open("wb") as stdout, errors.open("wb") as stderr:
        result = subprocess.run(args, cwd=STAGING, env=environment, stdout=stdout, stderr=stderr, timeout=600)
    receipt.update({
        "finished_at": now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "exit_code": result.returncode,
        "stdout": {"file": output.name, **file_record(output)},
        "stderr": {"file": errors.name, **file_record(errors)},
        "files_after": {name: file_record(STAGING / name) for name in ("package.json", "package-lock.json")},
    })
    (RUN / (label + ".json")).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    print(output.read_text(encoding="utf-8", errors="replace")[-12000:], flush=True)
    print(errors.read_text(encoding="utf-8", errors="replace")[-12000:], flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
