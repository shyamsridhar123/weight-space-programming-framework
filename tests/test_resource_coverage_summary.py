"""Check literal scalar accounting without importing or running study code."""
import ast
import math
from pathlib import Path


def test_resource_coverage_preserves_unknown_cost_and_exact_integer_totals():
    root = Path(__file__).resolve().parents[1]
    namespace = {"math": math}
    for relative, name in (
        ("rlcd/weight_space/metrics.py", "observed_resource_summary"),
        ("scripts/run_trained_research.py", "summarize_resource_measurements"),
    ):
        source = root/relative
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    summarize = namespace["observed_resource_summary"]
    adapt = namespace["summarize_resource_measurements"]

    def rejected(function, value):
        try:
            function(value)
        except ValueError:
            return
        raise AssertionError(f"Accepted invalid resource input: {value!r}")

    assert summarize([None, 0, 2.5]) == {
        "assigned": 3, "observed": 2, "missing": 1, "observed_sum": 2.5,
        "complete_for_assigned": False, "total_if_complete": None,
    }
    assert summarize([None, None]) == {
        "assigned": 2, "observed": 0, "missing": 2, "observed_sum": None,
        "complete_for_assigned": False, "total_if_complete": None,
    }
    assert summarize([]) == {
        "assigned": 0, "observed": 0, "missing": 0, "observed_sum": None,
        "complete_for_assigned": True, "total_if_complete": None,
    }
    assert summarize([0])["total_if_complete"] == 0
    assert type(summarize([0])["observed_sum"]) is int
    assert summarize([-0.0])["observed"] == 1
    assert summarize(iter([None, 0]))["observed_sum"] == 0
    for values, expected in (([2**60, 1], 2**60+1), ([10**400, 1], 10**400+1)):
        result = summarize(values)
        assert result["observed_sum"] == result["total_if_complete"] == expected
        assert type(result["observed_sum"]) is int
    assert summarize([1e16, 1.0, 1.0])["observed_sum"] == 10000000000000002.0

    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    class NumericImpostorType(type):
        def __eq__(cls, other):
            return other is int

    class NumericImpostor(metaclass=NumericImpostorType):
        def __lt__(self, other):
            return False

        def __float__(self):
            return 0.0

    for value in (True, False, -1, -.01, math.nan, math.inf, -math.inf,
                  "0", {}, [], object(), IntSubclass(0), FloatSubclass(0), NumericImpostor()):
        rejected(summarize, [value])
    rejected(summarize, [10**400, 0.0])
    rejected(summarize, [1.7e308, 1.7e308])

    assigned = [
        {"evaluation_missing": True, "cost": {
            "fit_seconds_all_options": 0,
            "heldout_workflow_seconds": 10.0,
            "native_head_evaluation_seconds": 3.0,
            "storage_bytes_pre_outcome": 2**60,
        }},
        {"evaluation_missing": False, "cost": {
            "fit_seconds_all_options": 2.5,
            "heldout_workflow_seconds": 0.0,
            "native_head_evaluation_seconds": 0.0,
            "storage_bytes_pre_outcome": 1,
        }},
        {}, {"cost": None}, {"cost": {"fit_seconds_all_options": None}},
    ]
    before = repr(assigned)
    result = adapt(assigned)
    fields = (
        "fit_seconds_all_options", "fit_selection_materialization_seconds",
        "native_head_evaluation_seconds", "binary64_head_evaluation_seconds",
        "exact_application_rollback_verification_seconds", "heldout_export_seconds",
        "heldout_workflow_seconds", "storage_bytes_pre_outcome",
    )
    measurements = result["resource_measurements"]
    assert tuple(measurements) == fields
    assert measurements["fit_seconds_all_options"] == {
        "assigned": 5, "observed": 2, "missing": 3, "observed_sum": 2.5,
        "complete_for_assigned": False, "total_if_complete": None,
    }
    assert measurements["storage_bytes_pre_outcome"]["observed_sum"] == 2**60+1
    assert type(measurements["storage_bytes_pre_outcome"]["observed_sum"]) is int
    assert measurements["heldout_workflow_seconds"]["observed_sum"] == 10.0
    assert measurements["native_head_evaluation_seconds"]["observed_sum"] == 3.0
    assert all(result[key] is None for key in result if key.startswith("total_"))
    complete = adapt(assigned[:2])
    assert complete["total_fit_seconds_all_options"] == 2.5
    assert complete["total_heldout_workflow_seconds"] == 10.0
    assert complete["total_storage_bytes"] == 2**60+1
    assert complete["total_fit_selection_materialization_seconds"] is None
    assert complete["resource_measurements"]["native_head_evaluation_seconds"]["total_if_complete"] == 3.0
    assert measurements["fit_selection_materialization_seconds"] == {
        "assigned": 5, "observed": 0, "missing": 5, "observed_sum": None,
        "complete_for_assigned": False, "total_if_complete": None,
    }
    assert adapt([{"cost": {"fit_selection_materialization_seconds": 4.0}}])[
        "total_fit_selection_materialization_seconds"] == 4.0
    assert all(value == summarize([None, None, None]) for value in
               adapt([{}, {"cost": None}, {"cost": {}}])["resource_measurements"].values())
    assert all(value == summarize([]) for value in adapt([])["resource_measurements"].values())
    assert adapt([{"cost": {"future_seconds": -1}}]) == adapt([{}])
    assert repr(assigned) == before
    for cost in (False, 0, [], "", ()):
        rejected(adapt, [{"cost": cost}])
    for storage in (True, -1, 0.0, math.nan, "0"):
        rejected(adapt, [{"cost": {"storage_bytes_pre_outcome": storage}}])
    rejected(adapt, [{"cost": {"native_head_evaluation_seconds": -1}}])
    rejected(adapt, [{"cost": {"heldout_workflow_seconds": 1.7e308}}]*2)


if __name__ == "__main__":
    test_resource_coverage_preserves_unknown_cost_and_exact_integer_totals()
