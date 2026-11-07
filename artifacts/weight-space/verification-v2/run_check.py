"""Capture a fresh, read-only-source full Python integration verification."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/weight-space/v2/protocol.json"
FREEZE = ROOT / "artifacts/weight-space/study-v2-freeze/receipt.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(path):
    return {"sha256": sha(path), "bytes": path.stat().st_size}


def sources():
    paths = {p.resolve() for directory in ("rlcd", "scripts", "tests")
             for p in (ROOT / directory).rglob("*.py")}
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    paths.update(ROOT / name for name in frozen["files"])
    paths.update(ROOT / name for name in ("requirements.txt", "requirements.lock.txt", "pytest.ini"))
    return {p.relative_to(ROOT).as_posix(): record(p) for p in sorted(paths)}


def frozen_state():
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    changed = [name for name, expected in frozen["files"].items()
               if record(ROOT / name) != expected]
    return {"receipt_path": FREEZE.relative_to(ROOT).as_posix(),
            "receipt_file_sha256": sha(FREEZE), "declared_sha256": frozen["sha256"],
            "file_count": len(frozen["files"]), "changed_files": changed,
            "all_current_files_match": not changed}


def import_audit():
    sys.path.insert(0, str(ROOT))
    from rlcd.weight_space import study_v2
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frozen_paths = {p.resolve() for p in study_v2.source_paths(PROTOCOL, protocol)}
    stages = {}
    for stage in ("eager_study_import", "including_run_auxiliary_import"):
        if stage == "including_run_auxiliary_import":
            from rlcd.weight_space import study_tasks_v2  # noqa: F401
        modules = {}
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if not filename or name == "__main__":
                continue
            path = Path(filename).resolve()
            if (path.exists() and path.is_relative_to(ROOT)
                    and ".venv" not in path.relative_to(ROOT).parts and path.suffix == ".py"):
                modules[name] = path
        stages[stage] = {
            "local_modules": {name: p.relative_to(ROOT).as_posix() for name, p in sorted(modules.items())},
            "uncovered": {name: p.relative_to(ROOT).as_posix() for name, p in sorted(modules.items())
                          if p not in frozen_paths},
        }
    excluded = {"rts_study", "experiments", "demo"}
    references = []
    for path in sorted((ROOT / "rlcd/weight_space").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imported = [node.module] if isinstance(node, ast.ImportFrom) else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in imported:
                if name and name.split(".")[-1] in excluded:
                    references.append({"path": path.relative_to(ROOT).as_posix(),
                                       "line": node.lineno, "imported_module": name})
    reserved_literals = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and type(node.value) is int and 7300 <= node.value <= 7311:
                reserved_literals.append({"path": path.relative_to(ROOT).as_posix(),
                                          "line": node.lineno, "value": node.value})
    result = {"timestamp_utc": now(), "frozen_source_count": len(frozen_paths), "stages": stages,
              "excluded_module_import_references": references,
              "reserved_seed_integer_literals_in_tests": reserved_literals,
              "seed_review": "Study protocol fixture replaces seeds with 900001 and 900002 before use; explicit main smoke cases use 900001 through 900008; auxiliary fixtures use 900001 through 900004. No actual study world or model was executed by this audit.",
              "scope": "Import-only dynamic audit plus static import and seed review; no claim of exhaustive tracing of arbitrary dynamic imports."}
    write("import-coverage.json", result)
    if reserved_literals or any(stage["uncovered"] for stage in stages.values()):
        raise RuntimeError("Import coverage or reserved seed preflight failed")


def junit_result():
    path = OUT / "python-tests.junit.xml"
    if not path.exists():
        return {"available": False, "counts": None, "failures": []}
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed, errors, skipped, details = 0, 0, 0, []
    for case in cases:
        failure, error, skip = case.find("failure"), case.find("error"), case.find("skipped")
        failed += failure is not None
        errors += error is not None
        skipped += skip is not None
        for kind, node in (("failure", failure), ("error", error)):
            if node is not None:
                details.append({"kind": kind, "classname": case.get("classname"),
                                "name": case.get("name"), "message": node.get("message"),
                                "text": node.text})
    counts = {"total": len(cases), "passed": len(cases)-failed-errors-skipped,
              "failed": failed, "errors": errors, "skipped": skipped}
    suites = [{k: suite.get(k) for k in ("name", "tests", "failures", "errors", "skipped", "time", "timestamp")}
              for suite in root.iter("testsuite")]
    return {"available": True, "counts": counts, "suites": suites, "failures": details}


def main():
    os.chdir(ROOT)
    started = now()
    dependencies = sorted(
        [{"name": dist.metadata.get("Name", "unknown"), "version": dist.version}
         for dist in importlib.metadata.distributions()], key=lambda row: (row["name"].lower(), row["version"]))
    environment = {"captured_at_utc": started, "python_executable": sys.executable,
                   "python_version": sys.version, "platform": platform.platform(),
                   "machine": platform.machine(), "processor": platform.processor(),
                   "cwd": str(ROOT), "local_time": datetime.now().astimezone().isoformat(),
                   "installed_distributions": dependencies,
                   "inherited_numerical_environment": {key: os.environ.get(key) for key in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")}}
    write("environment.json", environment)
    before = sources()
    write("sources-before.json", before)
    freeze_before = frozen_state()
    write("freeze-before.json", freeze_before)
    if not freeze_before["all_current_files_match"]:
        raise RuntimeError("Current sources differ from the actual study freeze")
    import_audit()
    argv = [sys.executable, "-B", "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
            "--basetemp", str(OUT / "pytest-tmp"), "--junitxml", str(OUT / "python-tests.junit.xml")]
    command = {"argv": argv, "windows_command_line": subprocess.list2cmdline(argv),
               "cwd": str(ROOT), "environment_overrides": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
               "runner_argv": getattr(sys, "orig_argv", [sys.executable, *sys.argv]),
               "runner_sha256": sha(Path(__file__)), "scope": "Full tests directory, no deselection"}
    write("python-tests.command.json", command)
    run_started = now()
    tick = time.perf_counter()
    child_env = dict(os.environ, **command["environment_overrides"])
    print(json.dumps({"status": "running", "start_utc": run_started, "command": command["windows_command_line"]}), flush=True)
    with (OUT / "python-tests.stdout.log").open("wb") as stdout, (OUT / "python-tests.stderr.log").open("wb") as stderr:
        completed = subprocess.run(argv, cwd=ROOT, env=child_env, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.perf_counter()-tick
    run_ended = now()
    after = sources()
    write("sources-after.json", after)
    freeze_after = frozen_state()
    write("freeze-after.json", freeze_after)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    results = junit_result()
    write("python-tests.results.json", results)
    status = {"exit_code": completed.returncode, "started_at_utc": run_started,
              "ended_at_utc": run_ended, "elapsed_seconds": elapsed, "counts": results["counts"]}
    write("python-tests.status.json", status)
    evidence_names = ["run_check.py", "environment.json", "sources-before.json", "sources-after.json",
                      "freeze-before.json", "freeze-after.json", "import-coverage.json", "python-tests.command.json",
                      "python-tests.stdout.log", "python-tests.stderr.log", "python-tests.status.json", "python-tests.results.json"]
    if (OUT / "python-tests.junit.xml").exists():
        evidence_names.append("python-tests.junit.xml")
    receipt = {"schema": "wsp-python-integration-verification/2", "started_at_utc": started,
               "finished_at_utc": now(), "test_run": status, "command": command,
               "source_file_count": len(before), "changed_source_files": changed,
               "source_files_unchanged": not changed, "study_freeze_before": freeze_before,
               "study_freeze_after": freeze_after, "failure_count": len(results["failures"]),
               "evidence": {name: record(OUT / name) for name in evidence_names},
               "complete": completed.returncode == 0 and not changed and freeze_after["all_current_files_match"],
               "limits": ["Python suite only; this receipt does not claim a browser, TypeScript, or formal proof build.",
                          "Test temporary artifacts are retained under pytest-tmp; the receipt hashes logs, results, commands, environment, runner and source manifests, not every temporary test artifact.",
                          "Reserved study seeds 7300 through 7311 were not invoked by this verification; the parent owns the separately frozen actual study."]}
    write("receipt.json", receipt)
    print(json.dumps({"status": "complete", "receipt": str(OUT / "receipt.json"), **status,
                      "changed_source_files": changed, "study_freeze_matches": freeze_after["all_current_files_match"]}), flush=True)
    return 0 if receipt["complete"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        (OUT / "runner-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
