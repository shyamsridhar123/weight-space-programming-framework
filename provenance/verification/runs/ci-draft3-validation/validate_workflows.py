"""Check current workflow syntax without executing jobs or altering old receipts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ACTIONLINT = Path(r"C:/public-user\AppData\Local\Temp\wsp-ci-validation-f1702c9e71a94e6da68deebdd3c8d192\actionlint\actionlint.exe")
BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
WORKFLOWS = sorted((ROOT / ".github/workflows").glob("*.yml"))
HISTORICAL = [ROOT / ".runs/ci-draft2-validation/validate_workflows.py",
              ROOT / ".runs/ci-draft2-validation/validation.json",
              ROOT / ".runs/verify_wheel_draft2.py"]
LOGS = OUT / "logs"
EXTRACTED = OUT / "extracted"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    LOGS.mkdir(exist_ok=True)
    EXTRACTED.mkdir(exist_ok=True)
    receipt = {
        "schema": "local-ci-workflow-validation/3",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "validator_sha256": sha256(Path(__file__)),
        "remote_execution": False, "network_access_used": False,
        "workflow_commands_executed": False,
        "workflow_hashes_before": {relative(p): sha256(p) for p in WORKFLOWS},
        "historical_inputs_before": {relative(p): sha256(p) for p in HISTORICAL},
        "tools": {
            "python": {"path": sys.executable, "version": sys.version},
            "pyyaml": {"version": yaml.__version__},
            "actionlint": {"path": str(ACTIONLINT), "sha256": sha256(ACTIONLINT)},
            "bash": {"path": str(BASH), "sha256": sha256(BASH)},
        },
        "commands": [], "workflows": [], "failures": [],
        "limitations": [
            "GitHub-hosted jobs, dependency installation, examples, tests, wheel builds, and proofs were not executed by this validator.",
            "Bash scripts received syntax checks; Python heredocs were compiled without execution.",
            "ShellCheck and Pyflakes integrations were disabled.",
            "Action pins were compared with local draft.2 Git bytes; remote action metadata was not refreshed.",
        ],
    }

    def run(label: str, argv: list[object]) -> subprocess.CompletedProcess:
        args = [str(x) for x in argv]
        result = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=60)
        stdout, stderr = LOGS / f"{label}.stdout.log", LOGS / f"{label}.stderr.log"
        stdout.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        receipt["commands"].append({
            "label": label, "argv": args, "exit_code": result.returncode,
            "stdout": {"path": relative(stdout), "sha256": sha256(stdout)},
            "stderr": {"path": relative(stderr), "sha256": sha256(stderr)},
        })
        if result.returncode:
            receipt["failures"].append(f"{label}: exit {result.returncode}")
        return result

    run("actionlint-version", [ACTIONLINT, "-version"])
    run("bash-version", [BASH, "--noprofile", "--norc", "--version"])
    run("actionlint", [ACTIONLINT, "-no-color", "-oneline", "-shellcheck=", "-pyflakes=", *WORKFLOWS])
    for path in WORKFLOWS:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        old_bytes = subprocess.run(
            ["git", "show", f"v0.2.0-draft.2:{relative(path)}"],
            cwd=ROOT, capture_output=True, check=True, timeout=30,
        ).stdout
        old = yaml.safe_load(old_bytes)
        pins = [step["uses"] for job in data["jobs"].values() for step in job["steps"] if "uses" in step]
        old_pins = [step["uses"] for job in old["jobs"].values() for step in job["steps"] if "uses" in step]
        if pins != old_pins or any(not re.fullmatch(r"[^@]+@[0-9a-f]{40}", pin) for pin in pins):
            receipt["failures"].append(f"Changed or unpinned action: {relative(path)}")
        record = {"path": relative(path), "action_pins": pins,
                  "matches_draft2_action_pins": pins == old_pins, "bash_steps": [], "python_heredocs": []}
        receipt["workflows"].append(record)
        for job_id, job in data["jobs"].items():
            for step_index, step in enumerate(job["steps"]):
                if "run" not in step:
                    continue
                stem = f"{path.stem}-{job_id}-{step_index}"
                script = step["run"].rstrip("\n") + "\n"
                script_path = EXTRACTED / f"{stem}.sh"
                script_path.write_text(script, encoding="utf-8", newline="\n")
                result = run(stem, [BASH, "--noprofile", "--norc", "-n", script_path.as_posix()])
                record["bash_steps"].append({"name": step["name"], "path": relative(script_path),
                                             "sha256": sha256(script_path), "exit_code": result.returncode})
                lines = script.splitlines()
                for index, line in enumerate(lines):
                    match = re.search(r"<<(['\"])([A-Za-z_]\w*)\1", line)
                    if not match or "python" not in line:
                        continue
                    end = lines.index(match.group(2), index + 1)
                    source = "\n".join(lines[index + 1:end]) + "\n"
                    python_path = EXTRACTED / f"{stem}-heredoc-{index}.py"
                    python_path.write_text(source, encoding="utf-8", newline="\n")
                    compile(source, str(python_path), "exec")
                    record["python_heredocs"].append({"step": step["name"], "path": relative(python_path),
                                                      "sha256": sha256(python_path), "compiled": True, "executed": False})
    receipt["workflow_hashes_after"] = {relative(p): sha256(p) for p in WORKFLOWS}
    receipt["historical_inputs_after"] = {relative(p): sha256(p) for p in HISTORICAL}
    for key in ("workflow_hashes", "historical_inputs"):
        if receipt[f"{key}_before"] != receipt[f"{key}_after"]:
            receipt["failures"].append(f"{key} changed during validation")
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    receipt["status"] = "failed" if receipt["failures"] else "passed"
    output = OUT / "validation.json"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"receipt": relative(output), "status": receipt["status"], "failures": receipt["failures"]}))
    return bool(receipt["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
