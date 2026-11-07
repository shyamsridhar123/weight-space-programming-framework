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
