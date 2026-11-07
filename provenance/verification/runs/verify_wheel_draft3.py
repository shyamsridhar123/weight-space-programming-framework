"""Build draft.3 and execute unchanged examples from an isolated installation.

Adapted deliberately from verify_wheel_draft2.py: the current README routes to
checked-in scripts rather than embedding a Python fence. Each invocation writes
an exclusive attempt receipt/log; successful attempts also create the requested
canonical receipts. Failed attempts and their staged wheels remain available.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / ".runs"
BUILD_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
EXPECTED_VERSION = "0.2.0.dev3"
WHEEL_NAME = f"weight_space_programming_framework-{EXPECTED_VERSION}-py3-none-any.whl"
WHEEL = ROOT / "dist" / WHEEL_NAME
REGISTRY = "https://redacted.invalid/environment"
EXAMPLES = ("examples/edit_retention.py", "examples/session_edit.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_info(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": sha256(content)}


def snapshot(paths):
    return {path.relative_to(ROOT).as_posix(): file_info(path) for path in sorted(set(paths))}


def source_inputs():
    return snapshot([ROOT / "pyproject.toml", ROOT / "README.md", *sorted((ROOT / "rlcd").rglob("*.py"))])


def example_inputs():
    return snapshot(ROOT / name for name in EXAMPLES)


def preserved_artifacts():
    paths = [ROOT / ".runs" / "verify_wheel_draft2.py"]
    for version in ("0.2.0.dev1", "0.2.0.dev2"):
        paths.append(ROOT / "dist" / f"weight_space_programming_framework-{version}-py3-none-any.whl")
    for directory in (RUNS, ROOT / "provenance" / "verification" / "runs"):
        for pattern in ("wheel-*.json", "wheel-*.log"):
            paths.extend(p for p in directory.glob(pattern) if "draft3" not in p.name)
    return snapshot(paths)


def wheel_info(path):
    with zipfile.ZipFile(path) as archive:
        files = archive.namelist()
        metadata_paths = [name for name in files if name.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1, metadata_paths
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        python_members = {name: sha256(archive.read(name)) for name in files if name.endswith(".py")}
        readme = metadata.get_payload(decode=True).decode("utf-8")
    return {
        "path": path.relative_to(ROOT).as_posix(), **file_info(path),
        "distribution": metadata["Name"], "version": metadata["Version"],
        "requires_python": metadata["Requires-Python"],
        "requires_dist": metadata.get_all("Requires-Dist", []),
        "files": files, "python_member_sha256": python_members,
        "readme_text_sha256": sha256(readme.replace("\r\n", "\n").rstrip("\n").encode("utf-8")),
    }


def command_environment():
    env = os.environ.copy()
    temporary_directory = RUNS / "wheel-draft3-tmp"
    temporary_directory.mkdir(exist_ok=True)
    env["TMP"] = str(temporary_directory)
    env["TEMP"] = str(temporary_directory)
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_INDEX_URL"] = REGISTRY
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_CACHE_DIR"] = str(RUNS / "wheel-draft3-pip-cache")
    env["PYTHONUTF8"] = "1"
    for name in ("PIP_EXTRA_INDEX_URL", "PIP_NO_BUILD_ISOLATION", "PIP_NO_INDEX", "PIP_FIND_LINKS", "PIP_REQUIRE_VIRTUALENV", "PIP_TARGET", "PIP_PREFIX", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    return env


def run_command(command, cwd, log, receipt, label):
    command = [str(part) for part in command]
    print(f"Running {label}", flush=True)
    log.write(f"\nCOMMAND ({label})\n{json.dumps(command)}\nCWD {cwd}\n")
    log.flush()
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=cwd, env=command_environment(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=240)
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        log.write(partial + "\nCOMMAND TIMED OUT\n")
        log.flush()
        receipt.setdefault("commands", []).append({"label": label, "command": command, "cwd": str(cwd),
                                                   "timed_out": True, "seconds": round(time.monotonic() - started, 3)})
        raise
    log.write(result.stdout)
    log.write(f"\nEXIT CODE {result.returncode}\n")
    log.flush()
    receipt.setdefault("commands", []).append({"label": label, "command": command, "cwd": str(cwd),
                                               "exit_code": result.returncode, "seconds": round(time.monotonic() - started, 3)})
    assert result.returncode == 0, f"{label} exited {result.returncode}; see {log.name}"
    return result.stdout


def start(kind):
    canonical = RUNS / f"{kind}-draft3.json"
    canonical_log = canonical.with_suffix(".log")
    assert not canonical.exists() and not canonical_log.exists(), "Refusing to overwrite a completed draft.3 check"
    attempt = 1
    while any((RUNS / f"{kind}-draft3-attempt-{attempt}{suffix}").exists() for suffix in (".json", ".log", "-wheel")):
        attempt += 1
    path = RUNS / f"{kind}-draft3-attempt-{attempt}.json"
    log_path = path.with_suffix(".log")
    receipt = {
        "schema": f"wspf-standalone-{kind}/2", "started_at": now(),
        "status": "failed", "exit_code": 1,
        "attempt_receipt": path.relative_to(ROOT).as_posix(),
        "canonical_receipt": canonical.relative_to(ROOT).as_posix(),
        "prior_attempts": snapshot([*RUNS.glob(f"{kind}-draft3-attempt-*.json"), *RUNS.glob(f"{kind}-draft3-attempt-*.log")]),
        "verification_helper": {"path": Path(__file__).relative_to(ROOT).as_posix(), **file_info(Path(__file__))},
        "source_inputs_before": source_inputs(), "example_inputs_before": example_inputs(),
        "preserved_prior_artifacts_before": preserved_artifacts(),
        "registry": REGISTRY, "public_registry_validated": False,
        "temporary_directory": str(RUNS / "wheel-draft3-tmp"),
        "pip_cache_directory": str(RUNS / "wheel-draft3-pip-cache"),
    }
    return receipt, path, log_path, canonical, canonical_log, time.monotonic()


def confirm_unchanged(receipt):
    receipt["source_inputs_after"] = source_inputs()
    receipt["source_inputs_unchanged"] = receipt["source_inputs_before"] == receipt["source_inputs_after"]
    receipt["example_inputs_after"] = example_inputs()
    receipt["example_inputs_unchanged"] = receipt["example_inputs_before"] == receipt["example_inputs_after"]
    receipt["preserved_prior_artifacts_after"] = snapshot(ROOT / path for path in receipt["preserved_prior_artifacts_before"])
    receipt["preserved_prior_artifacts_unchanged"] = receipt["preserved_prior_artifacts_before"] == receipt["preserved_prior_artifacts_after"]
    assert receipt["source_inputs_unchanged"], "Package sources changed during verification"
    assert receipt["example_inputs_unchanged"], "Example sources changed during verification"
    assert receipt["preserved_prior_artifacts_unchanged"], "Historical wheel evidence changed during verification"


def complete(receipt, path, log_path, canonical, canonical_log, started):
    receipt["finished_at"] = now()
    receipt["seconds"] = round(time.monotonic() - started, 3)
    receipt["log_sha256"] = file_info(log_path)["sha256"]
    content = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(content)
    if receipt["status"] == "passed":
        with canonical.open("xb") as stream:
            stream.write(content)
        with canonical_log.open("xb") as stream:
            stream.write(log_path.read_bytes())
    print(json.dumps({"receipt": str(canonical if receipt["status"] == "passed" else path),
                      "status": receipt["status"], "seconds": receipt["seconds"],
                      "source_inputs_unchanged": receipt.get("source_inputs_unchanged"),
                      "example_inputs_unchanged": receipt.get("example_inputs_unchanged"),
                      "preserved_prior_artifacts_unchanged": receipt.get("preserved_prior_artifacts_unchanged")}, indent=2), flush=True)
    return receipt["exit_code"]


def build():
    assert not WHEEL.exists(), f"Refusing to overwrite {WHEEL}"
    receipt, path, log_path, canonical, canonical_log, started = start("wheel-build")
    output_directory = path.with_suffix("").with_name(path.stem + "-wheel")
    output_directory.mkdir()
    staged_wheel = output_directory / WHEEL_NAME
    command = [BUILD_PYTHON, "-I", "-m", "pip", "wheel", "--use-pep517", "--no-deps", "--index-url", REGISTRY,
               "--wheel-dir", output_directory, ROOT]
    receipt.update(command=[str(part) for part in command], build_isolation=True, cwd=str(ROOT),
                   staged_wheel=staged_wheel.relative_to(ROOT).as_posix())
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        try:
            project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
            assert project["project"]["version"] == EXPECTED_VERSION, project["project"]["version"]
            output = run_command(command, ROOT, log, receipt, "isolated PEP 517 wheel build")
            assert "Installing build dependencies" in output, "Build isolation installation was not reported"
            info = wheel_info(staged_wheel)
            receipt["staged_wheel_info"] = info
            assert info["version"] == EXPECTED_VERSION, info["version"]
            assert info["distribution"] == "weight-space-programming-framework", info["distribution"]
            expected_members = {name: details["sha256"] for name, details in receipt["source_inputs_before"].items() if name.startswith("rlcd/") and name.endswith(".py")}
            receipt["wheel_python_members_match_sources"] = info["python_member_sha256"] == expected_members
            assert receipt["wheel_python_members_match_sources"], "Wheel Python members differ from declared source files"
            readme_text = (ROOT / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
            receipt["wheel_readme_matches_source_text"] = info["readme_text_sha256"] == sha256(readme_text.encode("utf-8"))
            assert receipt["wheel_readme_matches_source_text"], "Wheel metadata README differs from the current guide"
            confirm_unchanged(receipt)
            with WHEEL.open("xb") as stream:
                stream.write(staged_wheel.read_bytes())
            final_info = wheel_info(WHEEL)
            assert final_info["sha256"] == info["sha256"], "Copied wheel differs from staged wheel"
            receipt["wheels"] = [final_info]
            receipt.update(status="passed", exit_code=0)
        except Exception:
            receipt["error"] = traceback.format_exc()
            log.write(receipt["error"])
            try:
                confirm_unchanged(receipt)
            except Exception:
                receipt["final_identity_error"] = traceback.format_exc()
                log.write(receipt["final_identity_error"])
    return complete(receipt, path, log_path, canonical, canonical_log, started)


def smoke():
    receipt, path, log_path, canonical, canonical_log, started = start("wheel-install-smoke")
    external_cwd = Path(tempfile.mkdtemp(prefix="wspf-wheel-smoke-draft3-")).resolve()
    environment = external_cwd / "venv"
    python = environment / "Scripts" / "python.exe"
    build_path = RUNS / "wheel-build-draft3.json"
    receipt.update(platform=platform.platform(), isolated_python_flag=True,
                   fresh_virtual_environment=True, system_site_packages=False,
                   environment=str(environment), cwd=str(external_cwd),
                   all_commands_completed=False, examples=[])
    initial_check = """import importlib.metadata, importlib.util, json, sys
from pathlib import Path
assert sys.flags.isolated == 1
assert sys.prefix != sys.base_prefix
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('rlcd') is None
assert importlib.util.find_spec('numpy') is None
print(json.dumps({'python': sys.executable, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
                  'isolated': bool(sys.flags.isolated), 'installed_distributions': sorted(
                      [{'name': d.metadata['Name'], 'version': d.version} for d in importlib.metadata.distributions()], key=lambda d: d['name'])}))
"""
    installed_check = f"""import importlib.metadata, importlib.util, json, sys, sysconfig
from pathlib import Path
import numpy
import rlcd.weight_space as wsp
environment = Path({str(environment)!r}).resolve()
source_root = Path({str(ROOT)!r}).resolve()
site_packages = Path(sysconfig.get_paths()['purelib']).resolve()
location = Path(wsp.__file__).resolve()
assert Path(sys.prefix).resolve() == environment
assert site_packages == environment / 'Lib' / 'site-packages'
assert location.is_relative_to(site_packages), str(location)
assert Path(numpy.__file__).resolve().is_relative_to(site_packages)
assert not Path.cwd().resolve().is_relative_to(source_root)
assert all(not Path(entry).resolve().is_relative_to(source_root) for entry in sys.path)
assert sys.flags.isolated == 1
assert sys.flags.optimize == 0
assert importlib.util.find_spec('torch') is None
assert importlib.metadata.version('weight-space-programming-framework') == {EXPECTED_VERSION!r}
print(json.dumps({{'status': 'passed', 'import_path': str(location), 'site_packages': str(site_packages),
                  'numpy_import_path': str(Path(numpy.__file__).resolve()), 'python': sys.executable, 'python_version': sys.version,
                  'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'cwd': str(Path.cwd()), 'sys_path': sys.path,
                  'distribution': importlib.metadata.version('weight-space-programming-framework'),
                  'numpy': importlib.metadata.version('numpy'), 'torch_available': False, 'isolated_python': bool(sys.flags.isolated),
                  'checks': ['fresh environment site-packages imports', 'isolated Python with assertions enabled', 'Torch absent',
                             'distribution version', 'working directory outside checkout', 'checkout absent from import search path'],
                  'installed_distributions': sorted([{{'name': d.metadata['Name'], 'version': d.version}} for d in importlib.metadata.distributions()], key=lambda d: d['name'])}}, indent=2))
"""
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        try:
            assert not external_cwd.is_relative_to(ROOT), external_cwd
            assert not environment.exists(), environment
            build_receipt = json.loads(build_path.read_text(encoding="utf-8"))
            assert build_receipt["status"] == "passed", "Build receipt must pass before installation"
            assert receipt["source_inputs_before"] == build_receipt["source_inputs_after"], "Package sources changed since build"
            assert receipt["example_inputs_before"] == build_receipt["example_inputs_after"], "Example sources changed since build"
            receipt["build_receipt"] = {"path": build_path.relative_to(ROOT).as_posix(), **file_info(build_path)}
            receipt["wheel"] = wheel_info(WHEEL)
            assert receipt["wheel"] == build_receipt["wheels"][0], "Wheel changed since build"
            for name in EXAMPLES:
                source = ROOT / name
                copied = external_cwd / source.name
                with copied.open("xb") as stream:
                    stream.write(source.read_bytes())
                assert file_info(copied) == receipt["example_inputs_before"][name], name
                receipt["examples"].append({"source": name, "copied_path": str(copied),
                                            "copied_verbatim": True, **file_info(copied)})
            run_command([BUILD_PYTHON, "-I", "-m", "venv", environment], external_cwd, log, receipt, "create fresh core-only virtual environment outside checkout")
            configuration = (environment / "pyvenv.cfg").read_text(encoding="utf-8")
            assert "include-system-site-packages = false" in configuration.lower(), configuration
            receipt["pyvenv_cfg"] = configuration
            output = run_command([python, "-I", "-c", initial_check], external_cwd, log, receipt, "verify initial environment isolation")
            receipt["initial_environment_check"] = json.loads(output)
            run_command([python, "-I", "-m", "pip", "install", "--index-url", REGISTRY, WHEEL], external_cwd, log, receipt, "install core wheel and dependencies")
            run_command([python, "-I", "-m", "pip", "check"], external_cwd, log, receipt, "pip check")
            output = run_command([python, "-I", "-c", installed_check], external_cwd, log, receipt, "verify installed-package isolation")
            receipt["installed_environment_check"] = json.loads(output)
            for example in receipt["examples"]:
                copied = Path(example["copied_path"])
                output = run_command([python, "-I", copied], external_cwd, log, receipt, f"execute byte-identical {example['source']}")
                example["output"] = json.loads(output)
                example["exit_code"] = 0
                expected = receipt["example_inputs_before"][example["source"]]
                example["copied_bytes_unchanged"] = file_info(copied) == expected
                example["source_bytes_unchanged"] = file_info(ROOT / example["source"]) == expected
                assert example["copied_bytes_unchanged"] and example["source_bytes_unchanged"], example["source"]
            run_command([python, "-I", "-m", "rlcd.weight_space", "--help"], external_cwd, log, receipt, "isolated module CLI help")
            run_command([environment / "Scripts" / "weight-space.exe", "--help"], external_cwd, log, receipt, "console entry-point help")
            receipt["wheel_after"] = wheel_info(WHEEL)
            receipt["wheel_unchanged"] = receipt["wheel"] == receipt["wheel_after"]
            assert receipt["wheel_unchanged"], "Wheel changed during smoke verification"
            confirm_unchanged(receipt)
            receipt.update(status="passed", exit_code=0, all_commands_completed=True)
        except Exception:
            receipt["error"] = traceback.format_exc()
            log.write(receipt["error"])
            try:
                confirm_unchanged(receipt)
            except Exception:
                receipt["final_identity_error"] = traceback.format_exc()
                log.write(receipt["final_identity_error"])
    return complete(receipt, path, log_path, canonical, canonical_log, started)


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in {"build", "smoke"}, "Choose build or smoke"
    raise SystemExit(build() if sys.argv[1] == "build" else smoke())
