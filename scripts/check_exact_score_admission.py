"""Standalone compiler/CLI check: python -m scripts.check_exact_score_admission."""
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rlcd.weight_space.__main__ import main as cli_main
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.program import Program, apply_program


def main() -> None:
    with TemporaryDirectory(prefix="wsp-exact-score-") as temporary:
        root = Path(temporary)
        base = TensorArtifact("synthetic-exact-score-admission/1", {"head.weight": [[0., 0.]]})
        base_path = root / "base.json"
        base_path.write_text(json.dumps(base.to_dict()), encoding="utf-8")

        def invoke(specification: dict, name: str) -> tuple[int, str, Path]:
            spec_path, output = root / (name + ".json"), root / name
            spec_path.write_text(json.dumps(specification), encoding="utf-8")
            errors = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(errors):
                try:
                    status = cli_main(["compile-features", str(base_path), str(spec_path), "--out", str(output)])
                except SystemExit as failure:
                    status = failure.code
            return status, errors.getvalue(), output

        specification = {"head": {"weight": "head.weight"}, "target_features": [[3., 0.]],
                         "desired_delta": [[1.]], "retain_features": [[0., 1.]],
                         "retention": "none", "ridge": 1e-30,
                         "native_dtypes": {"head.weight": "float32"}, "cast": "round"}
        status, errors, approximate = invoke(specification, "approximate")
        assert status == 0, errors
        diagnostics = json.loads((approximate / "compilation.json").read_text(encoding="utf-8"))
        assert diagnostics["feature_constraints"]["exact"]["consistency_status"] == "feasible"
        program = Program.from_dict(json.loads((approximate / "program.json").read_text(encoding="utf-8")))
        endpoint = apply_program(base, program).tensors["head.weight"][0, 0]
        residual = 3*Fraction.from_float(float(endpoint))-1
        assert residual == Fraction(1, 33554432)

        status, errors, rejected = invoke(specification | {"require_exact_scores": True}, "rejected")
        assert status == 2 and "does not exactly realize the requested target-score changes" in errors
        assert not rejected.exists()

        exact_specification = specification | {"target_features": [[1., 0.]], "retention": "nullspace",
                                               "require_exact_scores": True}
        status, errors, accepted = invoke(exact_specification, "accepted")
        assert status == 0, errors
        program = Program.from_dict(json.loads((accepted / "program.json").read_text(encoding="utf-8")))
        weights = apply_program(base, program).tensors["head.weight"][0]
        assert [Fraction.from_float(float(value)) for value in weights] == [Fraction(1), Fraction(0)]

        status, errors, invalid = invoke(specification | {"require_exact_scores": 1}, "invalid")
        assert status == 2 and "require_exact_scores must be a boolean" in errors
        assert not invalid.exists()
    print("Exact-score compiler/CLI admission check passed.")


if __name__ == "__main__":
    main()
