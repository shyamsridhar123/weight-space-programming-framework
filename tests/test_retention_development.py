"""One scalar/structural check; no generator, training, inference or study run."""
from fractions import Fraction

import numpy as np

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.program import Edit, Program, apply_program
from scripts import run_retention_development as study


def test_fixed_retention_rules_preserve_exactness_and_missing_assignments():
    metrics = {scope: {"target_gain": .5, "protected_harm": 0.,
                       "newly_harmful_protected_owner_flips": 0,
                       "target_rows": 16, "protected_rows": 32}
               for scope in ("fitting", "calibration")}
    observation = {"maximum_exact_protected_score_drift": str(Fraction(1, 2**120)),
                   "maximum_native_protected_score_drift": 0.,
                   "minimum_native_protected_margin": .2, "protected_owner_flips": 0}
    verdict = study._gate_verdict({"score_budget": 0., "native_margin": 0.},
                                  observation, metrics, True)
    assert verdict["passes"] is False and verdict["useful"] is False
    assert verdict["reasons"] == ["exact_score_budget_exceeded"]

    weight, bias = np.zeros((4, 8), np.float32), np.zeros(4, np.float32)
    weight[0, 0], bias[3] = 300., -64.
    base = TensorArtifact("trained-frozen-readout/1",
                          {study.HEAD_WEIGHT: weight, study.HEAD_BIAS: bias}, {})
    program = Program(base.sha256, (Edit(study.HEAD_BIAS, (3,), -64., -63.),),
                      "Scalar structural check")
    assert study._structural(base, program, apply_program(base, program))["structural_accepted"] is True

    manifest = study._initial_manifest()
    first = manifest["assignments"][0]["candidates"][0]
    first.update(status="failed", metrics={"fitting": metrics["fitting"]})
    partial = study.summarize_manifest(manifest)
    assert partial["assigned_seed_task_denominator"] == 6
    assert all(row["useful_successes"] == 0 for row in partial["gate_frontier"])
    bounds = partial["candidate_summaries"][0]["fitting_target_gain"]
    assert bounds["observed_assignments"] == 1 and bounds["missing_assignments"] == 5
    assert bounds["all_assigned_mean_lower_bound"] == (.5-5)/6
    assert bounds["all_assigned_mean_upper_bound"] == (.5+5)/6


if __name__ == "__main__":
    test_fixed_retention_rules_preserve_exactness_and_missing_assignments()
    print("fixed retention scalar/structural check passed; no study executed")
