"""Run the real failure path with stdlib fakes, without importing study code."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def test_calibration_failure_retains_fallback_and_all_fitted_option_costs():
    source = Path(__file__).resolve().parents[1]/"scripts/run_trained_research.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "fit_case_arm")

    class Rows(list):
        def __getitem__(self, index):
            if isinstance(index, list):
                return Rows(list.__getitem__(self, row) for row in index)
            value = super().__getitem__(index)
            return Rows(value) if isinstance(index, slice) else value

        def tolist(self):
            return list(self)

    def artifact(identity):
        return SimpleNamespace(sha256=identity, parameter_count=2, to_dict=lambda: {})

    parent = artifact("synthetic-parent")
    program = artifact("synthetic-program")
    candidate = SimpleNamespace(sha256="synthetic-candidate", no_op=True)
    metrics = {"target_gain": .2, "protected_harm": 0.}
    costs = [.125, .25, .5]
    options = [{"artifact": artifact(f"synthetic-option-{index}"), "program": program,
                "config": {"option": index}, "diagnostics": {"synthetic": True},
                "retention_receipt": None, "fit_seconds": seconds}
               for index, seconds in enumerate(costs)]
    calibrated = []

    def external_codes(proposed, *args):
        calibrated.append(proposed.sha256)
        if len(calibrated) == 2:
            raise RuntimeError("synthetic calibration failure")
        return Rows([0, 1])

    writes, sealed = {}, []
    session = SimpleNamespace(
        sha256="synthetic-session", trace=[], metadata=lambda: None,
        propose_program=lambda proposed: candidate, seal=sealed.append)
    namespace = {
        "Path": Path, "time": SimpleNamespace(perf_counter=iter([10., 13.]).__next__),
        "subset": lambda data, case, role, **kwargs: Rows([0] if role == "target" else [1]),
        "evaluate_head": lambda *args, **kwargs: SimpleNamespace(argmax=lambda axis: Rows([0, 1])),
        "fit_options": lambda *args: options, "external_codes": external_codes,
        "behavior_metrics": lambda *args: metrics, "calibration_rank": lambda value: 0,
        "Program": lambda *args: program,
        "StructuralPolicy": lambda *args: SimpleNamespace(
            check=lambda *args: {"structural_accepted": True, "issues": []}, to_dict=lambda: {}),
        "MARGINS": {"minimum_target_gain": .1, "maximum_protected_harm": .01},
        "digest": lambda value: "synthetic-digest", "ARCHITECTURE": "synthetic",
        "TaskSpec": lambda *args: artifact("synthetic-task"), "SplitSpec": lambda *args: args,
        "TEST_COUNTS": {"synthetic": 0}, "ProgrammerSession": lambda *args, **kwargs: session,
        "AccessPolicy": lambda **kwargs: kwargs, "BudgetLimits": lambda **kwargs: kwargs,
        "write_json": lambda path, value: writes.__setitem__(path, value),
        "rollback": lambda *args: program, "now": lambda: "synthetic-time",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    lineage = {"id": "synthetic-lineage", "base": parent,
               "trained": {"features": Rows([[1.], [0.]])}, "record": {"domain": "synthetic"}}
    data = {"roles": {"calibration": Rows([0, 1])}, "ids": Rows(["row-0", "row-1"]),
            "labels": Rows([0, 1]), "queue_map": [0, 1]}
    case = {"case_id": "synthetic-case", "target_label": 0, "desired_queue": 0}
    with TemporaryDirectory() as directory:
        result = namespace["fit_case_arm"](lineage, data, case, "soft", Path(directory), "synthetic")
        record = writes[result["directory"]/"attempt.json"]

    assert calibrated == ["synthetic-option-0", "synthetic-option-1"]
    assert record["proposal_error"] == {"type": "RuntimeError", "message": "synthetic calibration failure"}
    assert record["selected_option"] is record["calibration"] is None
    assert result["candidate"] is parent and record["candidate_artifact_sha256"] == parent.sha256
    assert record["abstention_reasons"] == ["proposal-failed"]
    assert record["abstained"] and not record["admitted"] and sealed == [None]
    assert [option["calibration"] for option in record["options"]] == [metrics, None, None]
    assert [option["fit_seconds"] for option in record["options"]] == costs
    assert record["fit_seconds_all_options"] == sum(costs)
    assert record["fit_selection_materialization_seconds"] == 3.


if __name__ == "__main__":
    test_calibration_failure_retains_fallback_and_all_fitted_option_costs()
