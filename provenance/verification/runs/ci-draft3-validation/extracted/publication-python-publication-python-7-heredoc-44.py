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
