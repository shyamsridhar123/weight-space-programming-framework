"""Check returned fit-cost provenance without running study setup or fitting."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_fit_failures_keep_cost_unknown_until_options_return():
    source = Path(__file__).resolve().parents[1]/"scripts/run_trained_research.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "fit_case_arm"]
    assert len(functions) == 1
    body = functions[0].body
    setup_names = {"options", "fit_options_returned", "error"}
    initializers = {
        node.targets[0].id: index for index, node in enumerate(body)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id in setup_names
    }
    assert set(initializers) == setup_names
    first, fitting = min(initializers.values()), max(initializers.values())+1
    assert fitting-first == len(setup_names)
    assert isinstance(body[fitting], ast.Try)
    options_init = body[initializers["options"]].value
    assert isinstance(options_init, ast.List) and not options_init.elts
    for name, value in (("fit_options_returned", False), ("error", None)):
        literal = body[initializers[name]].value
        assert isinstance(literal, ast.Constant) and literal.value is value
    boundary = compile(ast.Module(body=body[first:fitting+1], type_ignores=[]), str(source), "exec")
    record = next(node.value for node in body
                  if isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "record")
    assert isinstance(record, ast.Dict)
    fields = [value for key, value in zip(record.keys, record.values)
              if isinstance(key, ast.Constant) and key.value == "fit_seconds_all_options"]
    assert len(fields) == 1
    cost_expression = compile(ast.Expression(body=fields[0]), str(source), "eval")

    class Rows(list):
        def __getitem__(self, index):
            if isinstance(index, list):
                return Rows(list.__getitem__(self, row) for row in index)
            value = super().__getitem__(index)
            return Rows(value) if isinstance(index, slice) else value

    for mode in ("immediate", "partial", "calibration", "success", "no-fit"):
        base = SimpleNamespace(sha256="synthetic-parent")
        fallback_program = object()
        costs = (0.0,) if mode == "no-fit" else (.125, .25, .5)
        returned = [{"artifact": SimpleNamespace(index=index), "program": object(),
                     "config": {"option": index}, "diagnostics": {},
                     "retention_receipt": None, "fit_seconds": seconds}
                    for index, seconds in enumerate(costs)]
        metrics = [{"target_gain": gain, "protected_harm": 0.0} for gain in (.1, .2, .3)]
        fitting_calls, completed_before_failure, calibrated = [], [], []

        def fit_options(*args):
            fitting_calls.append(args[-1])
            if mode in ("immediate", "partial"):
                if mode == "partial":
                    completed_before_failure.append(returned[0]["fit_seconds"])
                raise RuntimeError(f"synthetic {mode} failure")
            return returned

        def external_codes(proposed, *args):
            calibrated.append(proposed.index)
            if mode == "calibration" and len(calibrated) == 2:
                raise RuntimeError("synthetic calibration failure")
            return proposed.index

        arm = "unchanged" if mode == "no-fit" else "soft"
        namespace = {
            "base": base, "lineage": {}, "features": Rows([[1.], [0.]]),
            "data": {"labels": Rows([0, 1]), "queue_map": [0, 1]},
            "case": {"target_label": 0, "desired_queue": 0}, "arm": arm,
            "calibration": Rows([0, 1]), "target_rows": Rows([0]), "parent_cal": Rows([0, 1]),
            "fit_options": fit_options, "external_codes": external_codes,
            "behavior_metrics": lambda parent, prediction, *args: metrics[prediction],
            "calibration_rank": lambda value: value["target_gain"],
            "Program": lambda *args: fallback_program,
        }
        exec(boundary, namespace)
        measured = eval(cost_expression, namespace)
        assert fitting_calls == [arm]
        if mode in ("immediate", "partial"):
            # A pre-return failure must not turn an empty caller list into observed zero.
            assert measured is None
            assert namespace["options"] == [] and calibrated == []
            assert completed_before_failure == ([.125] if mode == "partial" else [])
        else:
            assert namespace["options"] is returned
            assert completed_before_failure == []
            if mode == "calibration":
                # Returned durations remain observed even when calibration later fails.
                assert measured == .875 and calibrated == [0, 1]
                assert [option.get("calibration") for option in returned] == [metrics[0], None, None]
            else:
                assert measured == (0.0 if mode == "no-fit" else .875)
                assert measured is not None
                assert calibrated == list(range(len(returned)))
                assert [option["calibration"] for option in returned] == metrics[:len(returned)]
                assert namespace["error"] is None
                assert namespace["selected"] == len(returned)-1
                assert namespace["chosen"] is returned[-1]
        if mode in ("immediate", "partial", "calibration"):
            assert namespace["error"] == {
                "type": "RuntimeError", "message": f"synthetic {mode} failure",
            }
            assert namespace["selected"] is None
            assert namespace["chosen"]["artifact"] is base
            assert namespace["chosen"]["program"] is fallback_program
            assert namespace["chosen"]["calibration"] is None
            assert namespace["chosen"]["fit_seconds"] == 0.0


if __name__ == "__main__":
    test_fit_failures_keep_cost_unknown_until_options_return()
