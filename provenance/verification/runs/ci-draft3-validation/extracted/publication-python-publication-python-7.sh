python -m venv "$RUNNER_TEMP/wsp-wheel-venv"
wheel_python="$RUNNER_TEMP/wsp-wheel-venv/bin/python"
"$wheel_python" -I -m pip install "$RUN_ROOT/wheel"/*.whl
"$wheel_python" -I -m pip check
receipt_dir="$GITHUB_WORKSPACE/$RUN_ROOT"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import tomllib

root = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
destination = Path(os.environ["RUNNER_TEMP"]).resolve() / "wsp-wheel-smoke"
assert not destination.is_relative_to(root), destination
destination.mkdir()
project_bytes = (root / "pyproject.toml").read_bytes()
project = tomllib.loads(project_bytes.decode("utf-8"))
wheels = sorted((root / os.environ["RUN_ROOT"] / "wheel").glob("*.whl"))
assert len(wheels) == 1, wheels
inputs = {
    "source_root": str(root),
    "expected_distribution_version": project["project"]["version"],
    "pyproject_sha256": hashlib.sha256(project_bytes).hexdigest(),
    "wheel": {
        "path": wheels[0].relative_to(root).as_posix(),
        "sha256": hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
    },
    "examples": [],
}
for name in ("edit_retention.py", "session_edit.py"):
    source = root / "examples" / name
    content = source.read_bytes()
    copied = destination / name
    copied.write_bytes(content)
    assert copied.read_bytes() == content
    inputs["examples"].append({
        "source": source.relative_to(root).as_posix(),
        "copied_name": name,
        "sha256": hashlib.sha256(content).hexdigest(),
    })
(destination / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
PY
cd "$RUNNER_TEMP/wsp-wheel-smoke"
"$wheel_python" -I - "$receipt_dir" <<'PY' | tee "$receipt_dir/wheel-smoke.log"
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
import rlcd.weight_space as wsp

inputs = json.loads(Path("inputs.json").read_text(encoding="utf-8"))
source_root = Path(inputs["source_root"]).resolve()
location = Path(wsp.__file__).resolve()
assert sys.flags.isolated
assert not Path.cwd().resolve().is_relative_to(source_root)
assert location.is_relative_to(Path(sys.prefix).resolve()), location
assert find_spec("torch") is None
installed_version = version("weight-space-programming-framework")
assert installed_version == inputs["expected_distribution_version"], installed_version
print(f"Installed distribution: {installed_version}", flush=True)
print(f"Imported package: {location}", flush=True)
commands = []
for example in inputs["examples"]:
    copied = Path(example["copied_name"]).resolve()
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == example["sha256"]
    command = [sys.executable, "-I", str(copied)]
    print(f"Running byte-identical example: {example['source']}", flush=True)
    subprocess.run(command, check=True)
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == example["sha256"]
    assert hashlib.sha256((source_root / example["source"]).read_bytes()).hexdigest() == example["sha256"]
    commands.append({"command": command, "exit_code": 0})
receipt = {
    "schema": "wspf-wheel-examples/1", "status": "passed",
    "cwd": str(Path.cwd().resolve()), "isolated_python_flag": True,
    "installed_distribution_version": installed_version,
    "package_location": str(location), "torch_available": False,
    "inputs": inputs, "commands": commands,
}
(Path(sys.argv[1]) / "wheel-examples.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8",
)
print("Installed-wheel examples passed with unchanged source bytes.", flush=True)
PY
"$wheel_python" -I -m rlcd.weight_space --help >> "$receipt_dir/wheel-smoke.log"
"$RUNNER_TEMP/wsp-wheel-venv/bin/weight-space" --help >> "$receipt_dir/wheel-smoke.log"
