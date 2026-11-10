"""Check scalar posthoc accounting without importing or running study code."""
import ast
import math
from pathlib import Path
from statistics import mean


def test_posthoc_resources_preserve_coverage_pairs_and_exact_totals():
    root = Path(__file__).resolve().parents[1]
    namespace = {"math": math, "mean": mean}
    for relative, name in (
        ("rlcd/weight_space/metrics.py", "observed_resource_summary"),
        ("scripts/run_trained_research.py", "summarize_resource_measurements"),
        ("scripts/run_trained_research.py", "resource_distribution"),
        ("scripts/run_trained_research.py", "summarize_posthoc_resources"),
    ):
        source = root/relative
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == name]
        assert len(functions) == 1, f"Expected one scalar helper named {name}"
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    summarize = namespace["observed_resource_summary"]
    adapt_measurements = namespace["summarize_resource_measurements"]
    distribution = namespace["resource_distribution"]
    adapt = namespace["summarize_posthoc_resources"]
    fields = (
        "fit_seconds_all_options", "fit_selection_materialization_seconds",
        "native_head_evaluation_seconds", "binary64_head_evaluation_seconds",
        "exact_application_rollback_verification_seconds", "heldout_export_seconds",
        "heldout_workflow_seconds", "storage_bytes_pre_outcome",
    )
    statistic_fields = ("mean", "minimum", "maximum", "p50", "p95")
    coverage_keys = {
        "assigned", "observed", "missing", "observed_sum",
        "complete_for_assigned", "total_if_complete",
    }

    def rejected(function, value):
        try:
            function(value)
        except ValueError:
            return
        raise AssertionError(f"Accepted invalid resource input: {value!r}")

    def check_shape(report):
        assert tuple(report["resource_measurements"]) == fields
        assert tuple(report["stages"]) == fields
        assert "timing_records" not in report
        assert "timing_missing" not in report
        assert set(report["paired_top_level_seconds"]) == coverage_keys
        assert report["observed_incremental_top_level_seconds"] == report[
            "paired_top_level_seconds"]["observed_sum"]
        for field in fields:
            coverage = report["resource_measurements"][field]
            stage = report["stages"][field]
            assert set(coverage) == coverage_keys
            assert set(stage) == {
                "observations", "mean", "minimum", "maximum", "p50", "p95",
                "sum_observed", "missing_assigned",
            }
            assert stage["observations"] == coverage["observed"]
            assert stage["sum_observed"] == coverage["observed_sum"]
            assert stage["missing_assigned"] == coverage["missing"]
            for key in statistic_fields:
                assert (stage[key] is None if coverage["observed"] == 0
                        else math.isfinite(stage[key]))

    assigned = [
        {"evaluation_missing": True, "cost": {
            "fit_seconds_all_options": 3,
            "fit_selection_materialization_seconds": 4.0,
            "native_head_evaluation_seconds": 1.0,
            "binary64_head_evaluation_seconds": 2.0,
            "exact_application_rollback_verification_seconds": .5,
            "heldout_export_seconds": .25,
            "heldout_workflow_seconds": 6.0,
            "storage_bytes_pre_outcome": 2**60,
        }},
        {"evaluation_missing": False, "cost": {
            **{field: 0 for field in fields}, "storage_bytes_pre_outcome": 1,
        }},
        {}, {"cost": None},
        {"cost": {"fit_selection_materialization_seconds": None,
                  "heldout_workflow_seconds": None}},
    ]
    before = repr(assigned)
    report = adapt(assigned)
    check_shape(report)
    assert report["resource_measurements"]["fit_selection_materialization_seconds"] == {
        "assigned": 5, "observed": 2, "missing": 3, "observed_sum": 4.0,
        "complete_for_assigned": False, "total_if_complete": None,
    }
    assert report["paired_top_level_seconds"] == {
        "assigned": 5, "observed": 2, "missing": 3, "observed_sum": 10.0,
        "complete_for_assigned": False, "total_if_complete": None,
    }
    assert report["observed_incremental_top_level_seconds"] == 10.0
    assert report["stages"]["native_head_evaluation_seconds"]["sum_observed"] == 1.0
    assert math.isclose(report["stages"]["fit_selection_materialization_seconds"]["p95"], 3.8)
    storage = report["resource_measurements"]["storage_bytes_pre_outcome"]
    assert storage["observed_sum"] == 2**60+1
    assert type(storage["observed_sum"]) is int
    assert storage["total_if_complete"] is None
    storage_stage = report["stages"]["storage_bytes_pre_outcome"]
    assert storage_stage["sum_observed"] == 2**60+1
    assert type(storage_stage["sum_observed"]) is int
    assert storage_stage["maximum"] == float(2**60)
    complete = adapt(assigned[:2])
    check_shape(complete)
    assert complete["paired_top_level_seconds"] == {
        "assigned": 2, "observed": 2, "missing": 0, "observed_sum": 10.0,
        "complete_for_assigned": True, "total_if_complete": 10.0,
    }
    assert complete["resource_measurements"]["storage_bytes_pre_outcome"][
        "total_if_complete"] == 2**60+1
    assert type(complete["resource_measurements"]["storage_bytes_pre_outcome"][
        "total_if_complete"]) is int
    assert repr(assigned) == before

    # Marginal observations from different assignments never form a pair.
    complementary = adapt([
        {"cost": {"fit_selection_materialization_seconds": 2}},
        {"cost": {"heldout_workflow_seconds": 3}},
    ])
    check_shape(complementary)
    assert complementary["resource_measurements"]["fit_selection_materialization_seconds"][
        "observed_sum"] == 2
    assert complementary["resource_measurements"]["heldout_workflow_seconds"][
        "observed_sum"] == 3
    assert complementary["paired_top_level_seconds"] == {
        "assigned": 2, "observed": 0, "missing": 2, "observed_sum": None,
        "complete_for_assigned": False, "total_if_complete": None,
    }
    assert complementary["observed_incremental_top_level_seconds"] is None
    assert complementary["stages"]["storage_bytes_pre_outcome"] == {
        "observations": 0, "mean": None, "minimum": None, "maximum": None,
        "p50": None, "p95": None, "sum_observed": None, "missing_assigned": 2,
    }

    mixed_pairs = adapt([
        {"cost": {"fit_selection_materialization_seconds": 2}},
        {"cost": {"heldout_workflow_seconds": 3}},
        {"cost": {"fit_selection_materialization_seconds": 4,
                  "heldout_workflow_seconds": 6}},
    ])
    check_shape(mixed_pairs)
    for field, total in (("fit_selection_materialization_seconds", 6),
                         ("heldout_workflow_seconds", 9)):
        assert mixed_pairs["resource_measurements"][field] == {
            "assigned": 3, "observed": 2, "missing": 1, "observed_sum": total,
            "complete_for_assigned": False, "total_if_complete": None,
        }
    assert mixed_pairs["paired_top_level_seconds"] == {
        "assigned": 3, "observed": 1, "missing": 2, "observed_sum": 10,
        "complete_for_assigned": False, "total_if_complete": None,
    }

    unobserved = adapt([{}, {"cost": None}, {"cost": {}}])
    check_shape(unobserved)
    missing_three = {
        "assigned": 3, "observed": 0, "missing": 3, "observed_sum": None,
        "complete_for_assigned": False, "total_if_complete": None,
    }
    assert all(value == missing_three for value in unobserved["resource_measurements"].values())
    assert unobserved["paired_top_level_seconds"] == missing_three
    assert unobserved["observed_incremental_top_level_seconds"] is None

    empty = adapt([])
    check_shape(empty)
    empty_coverage = {
        "assigned": 0, "observed": 0, "missing": 0, "observed_sum": None,
        "complete_for_assigned": True, "total_if_complete": None,
    }
    assert all(value == empty_coverage for value in empty["resource_measurements"].values())
    assert empty["paired_top_level_seconds"] == empty_coverage
    assert empty["observed_incremental_top_level_seconds"] is None

    zero = adapt([{"evaluation_missing": True, "cost": {field: 0 for field in fields}}])
    check_shape(zero)
    zero_coverage = {
        "assigned": 1, "observed": 1, "missing": 0, "observed_sum": 0,
        "complete_for_assigned": True, "total_if_complete": 0,
    }
    assert all(value == zero_coverage for value in zero["resource_measurements"].values())
    assert zero["paired_top_level_seconds"] == zero_coverage
    assert zero["observed_incremental_top_level_seconds"] == 0
    assert all(stage[key] == 0.0 for stage in zero["stages"].values()
               for key in statistic_fields)

    partial_zero = adapt([{"cost": {
        "fit_selection_materialization_seconds": 0, "heldout_workflow_seconds": None,
    }}])
    check_shape(partial_zero)
    assert partial_zero["resource_measurements"][
        "fit_selection_materialization_seconds"] == zero_coverage
    missing_one = {
        "assigned": 1, "observed": 0, "missing": 1, "observed_sum": None,
        "complete_for_assigned": False, "total_if_complete": None,
    }
    assert partial_zero["resource_measurements"]["heldout_workflow_seconds"] == missing_one
    assert partial_zero["paired_top_level_seconds"] == missing_one
    assert partial_zero["observed_incremental_top_level_seconds"] is None

    # Pair completeness depends only on the two top-level clocks.
    top_only = adapt([{"cost": {
        "fit_selection_materialization_seconds": 4, "heldout_workflow_seconds": 6,
    }}])
    check_shape(top_only)
    assert top_only["paired_top_level_seconds"] == {
        "assigned": 1, "observed": 1, "missing": 0, "observed_sum": 10,
        "complete_for_assigned": True, "total_if_complete": 10,
    }
    assert top_only["resource_measurements"]["fit_seconds_all_options"]["missing"] == 1
    nested_only = adapt([{"cost": {
        "fit_seconds_all_options": 3, "native_head_evaluation_seconds": 1,
        "binary64_head_evaluation_seconds": 2,
        "exact_application_rollback_verification_seconds": 1,
        "heldout_export_seconds": 1, "storage_bytes_pre_outcome": 7,
    }}])
    check_shape(nested_only)
    assert nested_only["paired_top_level_seconds"] == {
        "assigned": 1, "observed": 0, "missing": 1, "observed_sum": None,
        "complete_for_assigned": False, "total_if_complete": None,
    }

    precise = adapt([{"cost": {"fit_seconds_all_options": value}}
                     for value in (1e16, 1.0, 1.0)])
    check_shape(precise)
    assert precise["resource_measurements"]["fit_seconds_all_options"][
        "observed_sum"] == 10000000000000002.0
    assert precise["stages"]["fit_seconds_all_options"]["sum_observed"] == 10000000000000002.0

    # Rowwise binary64 pairing can round differently from summing marginals.
    rounded_pairs = adapt([
        {"cost": {"fit_selection_materialization_seconds": 1e16,
                  "heldout_workflow_seconds": 1.0}},
        {"cost": {"fit_selection_materialization_seconds": 0.0,
                  "heldout_workflow_seconds": 1.0}},
    ])
    check_shape(rounded_pairs)
    assert rounded_pairs["resource_measurements"]["heldout_workflow_seconds"]["observed_sum"] == 2.0
    assert rounded_pairs["paired_top_level_seconds"]["observed_sum"] == 1e16
    assert rounded_pairs["paired_top_level_seconds"]["total_if_complete"] == 1e16
    assert rounded_pairs["observed_incremental_top_level_seconds"] == 1e16

    assert distribution([]) == {
        "observations": 0, "mean": None, "minimum": None, "maximum": None,
        "p50": None, "p95": None,
    }
    assert distribution(iter([7])) == {
        "observations": 1, "mean": 7.0, "minimum": 7.0, "maximum": 7.0,
        "p50": 7.0, "p95": 7.0,
    }
    values = [10, 2, 6]
    values_before = list(values)
    stats = distribution(values)
    assert values == values_before
    assert stats["observations"] == 3
    assert stats["mean"] == stats["p50"] == 6.0
    assert stats["minimum"] == 2.0 and stats["maximum"] == 10.0
    assert math.isclose(stats["p95"], 9.6)
    two = distribution([2, 10])
    assert two["p50"] == 6.0 and math.isclose(two["p95"], 9.6)
    integer_ranks = distribution(range(21))
    assert integer_ranks["p50"] == 10.0 and integer_ranks["p95"] == 19.0
    # These means are finite even when naive intermediate arithmetic fails.
    large_stats = distribution([1e308, 1e308])
    assert large_stats["mean"] == 1e308
    assert all(math.isfinite(large_stats[key]) for key in statistic_fields)
    assert distribution([5e-324, 5e-324])["mean"] == 5e-324

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

    for cost in (False, 0, 0.0, [], "", (), object()):
        rejected(adapt, [{"cost": cost}])
    for value in (True, False, -1, -.01, math.nan, math.inf, -math.inf,
                  "0", {}, [], object(), IntSubclass(0), FloatSubclass(0), NumericImpostor()):
        for field in fields:
            rejected(adapt, [{"evaluation_missing": True, "cost": {field: value}}])
    for storage in (0.0, 1.0, 2.5):
        rejected(adapt, [{"cost": {"storage_bytes_pre_outcome": storage}}])
    ignored = [{"cost": {"future_seconds": -1, "future_scalar": NumericImpostor()}}]
    ignored_before = repr(ignored)
    assert adapt(ignored) == adapt([{}])
    assert repr(ignored) == ignored_before

    huge = 10**400
    exact_huge = summarize([huge, 1])
    assert exact_huge["observed_sum"] == exact_huge["total_if_complete"] == huge+1
    assert type(exact_huge["observed_sum"]) is int
    huge_measurement = adapt_measurements([{"cost": {"storage_bytes_pre_outcome": huge}}])
    assert huge_measurement["resource_measurements"]["storage_bytes_pre_outcome"][
        "total_if_complete"] == huge
    rejected(distribution, [huge])
    rejected(adapt, [{"cost": {"storage_bytes_pre_outcome": huge}}])
    rejected(adapt, [{"cost": {"fit_seconds_all_options": huge}}])
    for value in (math.nan, math.inf, -math.inf):
        rejected(distribution, [value])

    # Each marginal is finite; one complete pair overflows.
    rejected(adapt, [{"cost": {
        "fit_selection_materialization_seconds": 1e308,
        "heldout_workflow_seconds": 1e308,
    }}])
    # Each marginal and each pair is finite; aggregating pairs overflows.
    rejected(adapt, [
        {"cost": {"fit_selection_materialization_seconds": 1e308,
                  "heldout_workflow_seconds": 0.0}},
        {"cost": {"fit_selection_materialization_seconds": 0.0,
                  "heldout_workflow_seconds": 1e308}},
    ])
    rejected(adapt, [{"cost": {"native_head_evaluation_seconds": 1.7e308}}]*2)


if __name__ == "__main__":
    test_posthoc_resources_preserve_coverage_pairs_and_exact_totals()
