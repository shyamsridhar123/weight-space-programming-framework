"""Exercise only calibration summary blocks with stdlib synthetic records."""
import ast
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace


def test_analysis_retains_missing_calibration_and_all_assigned_denominators():
    source = Path(__file__).resolve().parents[1]/"scripts/run_trained_research.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "analyze_study")
    option_loop = next(node for node in ast.walk(function)
                       if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
                       and [name.id for name in node.target.elts] == ["i", "option"])
    case_loop = next(node for node in ast.walk(function)
                     if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                     and node.target.id == "case")
    frontier_start = next(index for index, node in enumerate(case_loop.body)
                          if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id == "options")
    frontier_block = case_loop.body[frontier_start:frontier_start+2]
    assert isinstance(frontier_block[1], ast.For) and frontier_block[1].target.id == "config"
    option_code = compile(ast.Module(body=[option_loop], type_ignores=[]), str(source), "exec")
    frontier_code = compile(ast.Module(body=frontier_block, type_ignores=[]), str(source), "exec")

    records = [
        {"trial_id": "complete/soft", "lineage_id": "complete", "arm": "soft", "selected_option": 1,
         "options": [
             {"configuration": {"retention_weight": 1}, "candidate_sha256": "complete-1", "fit_seconds": .125,
              "calibration": {"target_gain": .125, "protected_harm": 0.}},
             {"configuration": {"retention_weight": 8}, "candidate_sha256": "complete-8", "fit_seconds": .25,
              "calibration": {"target_gain": .25, "protected_harm": .005}},
         ]},
        {"trial_id": "interrupted/soft", "lineage_id": "interrupted", "arm": "soft", "selected_option": None,
         "options": [
             {"configuration": {"retention_weight": 1}, "candidate_sha256": "interrupted-1", "fit_seconds": .5,
              "calibration": {"target_gain": .375, "protected_harm": .02}},
             {"configuration": {"retention_weight": 8}, "candidate_sha256": "interrupted-8", "fit_seconds": 1.,
              "calibration": None},
             {"configuration": {"retention_weight": 64}, "candidate_sha256": "interrupted-64", "fit_seconds": 2.,
              "calibration": None},
         ]},
    ]
    namespace = {
        "MARGINS": {"minimum_target_gain": .1, "maximum_protected_harm": .01},
        "json": json, "np": SimpleNamespace(mean=mean), "calibration": [], "frontiers": [],
        "arm": "soft", "case": {"case_id": "synthetic-case"},
        "selected": [{"lineage_id": "complete"}, {"lineage_id": "interrupted"}, {"lineage_id": "missing"}],
    }
    for record in records:
        namespace.update(record=record, identity=record["lineage_id"])
        exec(option_code, namespace)
    exec(frontier_code, namespace)

    calibration = namespace["calibration"]
    assert [(row["trial_id"], row["option"], row["candidate_sha256"], row["fit_seconds"])
            for row in calibration] == [
        ("complete/soft", 0, "complete-1", .125),
        ("complete/soft", 1, "complete-8", .25),
        ("interrupted/soft", 0, "interrupted-1", .5),
        ("interrupted/soft", 1, "interrupted-8", 1.),
        ("interrupted/soft", 2, "interrupted-64", 2.),
    ]
    assert [row["selected_before_test"] for row in calibration] == [False, True, False, False, False]
    assert [(row["calibration_available"], row["target_gain"], row["protected_harm"],
             row["behavioral_calibration_pass"]) for row in calibration] == [
        (True, .125, 0., True), (True, .25, .005, True), (True, .375, .02, False),
        (False, None, None, None), (False, None, None, None),
    ]
    frontiers = {row["configuration"]["retention_weight"]: row for row in namespace["frontiers"]}
    assert set(frontiers) == {1, 8, 64}
    assert [(frontiers[weight]["available_lineages"], frontiers[weight]["assigned_lineages"],
             frontiers[weight]["missing_calibration_lineages"], frontiers[weight]["selected_lineages"])
            for weight in (1, 8, 64)] == [(2, 3, 1, 0), (1, 3, 2, 1), (0, 3, 3, 0)]
    assert [(frontiers[weight]["mean_calibration_target_gain"],
             frontiers[weight]["mean_calibration_protected_harm"],
             frontiers[weight]["behavioral_calibration_pass_fraction_all_assigned"])
            for weight in (1, 8, 64)] == [(.25, .01, 1/3), (.25, .005, 1/3), (None, None, 0.)]


if __name__ == "__main__":
    test_analysis_retains_missing_calibration_and_all_assigned_denominators()
