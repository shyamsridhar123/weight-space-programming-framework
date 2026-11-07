"""Run the worked guide's exact Python fences against the installed draft.3 wheel."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def binding(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    receipt_path = ROOT / ".runs/worked-example-excerpts-draft3.json"
    if receipt_path.exists():
        raise RuntimeError("Refusing to overwrite an excerpt execution receipt")
    smoke_path = ROOT / ".runs/wheel-install-smoke-draft3.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke["status"] != "passed" or smoke["wheel"]["sha256"] != "9d018555b7b06a2de57e184227550e9f764160fa668bfe122a0fc3cceaedb89a":
        raise RuntimeError("Expected the identified installed draft.3 wheel")
    environment = Path(smoke["environment"]).resolve()
    interpreter = environment / "Scripts/python.exe"
    document = ROOT / "docs/weight-space/v2/worked-example.md"
    before = binding(document)
    text = document.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^```python\s*\n(.*?)^```\s*$", text, re.M | re.S))
    if not matches:
        raise RuntimeError("No executable Python excerpts found")
    blocks = [m.group(1) for m in matches]
    code = "import json, sys\nfrom pathlib import Path\nimport rlcd.weight_space as wsp\n"
    code += f"expected_environment = Path({str(environment)!r}).resolve()\n"
    code += "assert Path(wsp.__file__).resolve().is_relative_to(expected_environment)\n"
    code += "\n".join(blocks)
    code += "\nprint(json.dumps({'module': str(Path(wsp.__file__).resolve()), 'prefix': sys.prefix, 'protected_residual': receipt['maximum_residuals_exact'], 'candidate_is_requested': candidate.sha256 == requested.sha256, 'compiler_exact_retention': compiled.retention_certificate['exactly_zero_on_supplied_features'], 'compiler_target_rmse': compiled.diagnostics['target_rmse_after_projection']}))\n"
    cwd = ROOT / ".runs/worked-example-excerpts-draft3"
    cwd.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    run = subprocess.run([str(interpreter), "-I", "-B", "-c", code], cwd=cwd,
                         capture_output=True, text=True, timeout=60)
    after = binding(document)
    observations = json.loads(run.stdout.strip()) if run.returncode == 0 else None
    result = {
        "schema": "wspf-documentation-excerpt-execution/1",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "passed" if run.returncode == 0 and before == after else "failed",
        "exit_code": run.returncode,
        "document_before": before,
        "document_after": after,
        "source_unchanged": before == after,
        "driver": binding(Path(__file__)),
        "installed_wheel_receipt": binding(smoke_path),
        "wheel": smoke["wheel"],
        "environment": str(environment),
        "isolated_python_flags": ["-I", "-B"],
        "cwd": str(cwd),
        "excerpts": [{"start_line": text[:m.start()].count("\n") + 1,
                      "sha256_utf8": hashlib.sha256(m.group(1).encode()).hexdigest()}
                     for m in matches],
        "observations": observations,
        "stdout": run.stdout,
        "stderr": run.stderr,
        "scope": "Exact Python fences from the worked guide, executed in document order against the previously installed identified wheel. Assertions cover endpoint identity, exact supplied-anchor residual, receipt verification, rollback, and positive-ridge target error.",
        "limitations": ["This reuses the already checked fresh-install environment; it is not another package installation.", "The model, target, protected feature and useful update are supplied. Execution is not an experiment or a live-provider trial.", "Other documentation code fragments are covered by static editorial review and the separately recorded example executions, not by this receipt."],
    }
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"result": result["result"], "observations": observations,
                      "receipt": str(receipt_path)}))
    return 0 if result["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
