"""Synthetic joint-constraint proofs and independent native-endpoint checks."""
from fractions import Fraction

import numpy as np
import pytest

import rlcd.weight_space.features as feature_module
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.features import ReadoutHead, _analyze_feature_constraints, compile_feature_edit
from rlcd.weight_space.model import canonical_bytes
from rlcd.weight_space.program import apply_program
from rlcd.weight_space.research_tasks import feature_geometry, modular_rank_witness


def analyze(target, retained, desired, **kwargs):
    return _analyze_feature_constraints(
        np.asarray(target, dtype=np.float64),
        None if retained is None else np.asarray(retained, dtype=np.float64),
        None if desired is None else np.asarray(desired, dtype=np.float64), **kwargs)


def head_fixture(width, outputs=1, *, bias=False):
    tensors = {"head.weight": np.zeros((outputs, width))}
    if bias:
        tensors["head.bias"] = np.zeros(outputs)
    return (TensorArtifact("synthetic-constraint-head/1", tensors),
            ReadoutHead("head.weight", "head.bias" if bias else None))


def check_witnesses(exact, target, retained, desired):
    """Check linear certificates directly, without calling the analyzer's helpers."""
    targets = np.asarray(target, dtype=np.float64)
    answers = np.asarray(desired, dtype=np.float64)
    anchors = np.empty((0, targets.shape[1])) if retained is None else np.asarray(retained, dtype=np.float64)
    rows = np.vstack((anchors, targets))
    responses = np.vstack((np.zeros((len(anchors), answers.shape[1])), answers))
    q = lambda value: Fraction.from_float(float(value))
    for witness in exact["conflict_witnesses"]:
        indices = witness["row_indices"]
        coefficients = list(map(Fraction, witness["coefficients_exact"]))
        assert indices == sorted(set(indices))
        assert coefficients and all(coefficients)
        for column in range(rows.shape[1]):
            assert sum((coefficient*q(rows[index, column])
                        for index, coefficient in zip(indices, coefficients)), Fraction(0)) == 0
        for output, recorded in zip(witness["output_indices"], witness["response_exact"]):
            response = sum((coefficient*q(responses[index, output])
                            for index, coefficient in zip(indices, coefficients)), Fraction(0))
            assert response != 0 and response == Fraction(recorded)
            assert exact["per_output_consistency"][output] == "infeasible"


@pytest.mark.parametrize("desired,verdict,norm", [([[1.], [3.]], "infeasible", None),
                                               ([[1.], [2.]], "feasible", "1")])
def test_affine_joint_consistency_has_independently_checkable_witness(desired, verdict, norm):
    retained, target = [[0., 1.]], [[1., 1.], [2., 1.]]
    basis, diagnostic = analyze(target, retained, desired)
    exact = diagnostic["exact"]
    assert basis.shape == (2, 1)
    assert diagnostic["protected"]["rank"] == exact["protected_rank_exact"] == 1
    assert exact["protected_nullity_exact"] == 1
    assert exact["consistency_status"] == verdict
    assert exact["minimum_squared_l2_norm_exact"] == [norm]
    assert exact["minimum_squared_frobenius_norm_exact"] == norm
    assert exact["constraint_rank_exact"] == 2
    assert exact["completed_constraint_rows"] == 3
    if verdict == "infeasible":
        witness, = exact["conflict_witnesses"]
        assert witness["row_indices"] == [0, 1, 2]
        assert witness["coefficients_exact"] == ["1", "-2", "1"]
        assert witness["response_exact"] == ["1"]
        assert diagnostic["unregularized"]["target_residual_rmse"] == pytest.approx(np.sqrt(.1))
        assert diagnostic["unregularized"]["edit_l2_norm_per_output"] == pytest.approx([1.4])
        check_witnesses(exact, target, retained, desired)
    else:
        assert exact["conflict_witnesses"] == []
        assert diagnostic["unregularized"]["target_residual_rmse"] == pytest.approx(0., abs=1e-14)


def test_multioutput_conflict_preserves_other_outputs_exact_minimum_norm():
    retained, target, desired = [[0., 1.]], [[1., 1.], [2., 1.]], [[1., 2.], [3., 4.]]
    _, diagnostic = analyze(target, retained, desired)
    exact = diagnostic["exact"]
    assert exact["consistency_status"] == "infeasible"
    assert exact["per_output_consistency"] == ["infeasible", "feasible"]
    assert exact["per_output_minimum_norm_status"] == ["not_applicable", "complete"]
    assert exact["minimum_squared_l2_norm_exact"] == [None, "4"]
    assert exact["minimum_squared_frobenius_norm_exact"] is None
    check_witnesses(exact, target, retained, desired)


def test_duplicate_target_conflict_does_not_invalidate_a_feasible_output():
    target, desired = [[2., 1.], [2., 1.]], [[1., 2.], [2., 2.]]
    _, diagnostic = analyze(target, None, desired)
    exact = diagnostic["exact"]
    assert exact["per_output_consistency"] == ["infeasible", "feasible"]
    assert exact["minimum_squared_l2_norm_exact"] == [None, "4/5"]
    check_witnesses(exact, target, None, desired)


def test_underdetermined_nonorthogonal_constraints_have_exact_minimum_norm():
    _, diagnostic = analyze([[1., 0., 1.]], [[1., 1., 0.]], [[1., 2.]])
    exact = diagnostic["exact"]
    # x+y=0, x+z=d minimizes at (d/3, -d/3, 2d/3).
    assert exact["per_output_consistency"] == ["feasible", "feasible"]
    assert exact["minimum_squared_l2_norm_exact"] == ["2/3", "8/3"]
    assert exact["minimum_squared_frobenius_norm_exact"] == "10/3"
    assert diagnostic["unregularized"]["edit_frobenius_norm"] == pytest.approx(np.sqrt(10/3))


def test_editable_bias_changes_the_constraint_problem_and_immutable_bias_stays_fixed():
    base, head = head_fixture(1, bias=True)
    fixed = compile_feature_edit(base, head, [[0.]], [[1.]], [[1.]], retention="nullspace")
    affine = compile_feature_edit(base, head, [[0.]], [[1.]], [[1.]], retention="nullspace", edit_bias=True)
    fixed_exact = fixed.diagnostics["feature_constraints"]["exact"]
    affine_exact = affine.diagnostics["feature_constraints"]["exact"]
    assert fixed.diagnostics["edit_bias"] is False and fixed.diagnostics["no_op"] is True
    assert fixed_exact["consistency_status"] == "infeasible"
    assert fixed.diagnostics["feature_constraints"]["editable_width"] == 1
    assert affine.diagnostics["edit_bias"] is True and affine.diagnostics["no_op"] is False
    assert affine.diagnostics["feature_constraints"]["editable_width"] == 2
    assert affine_exact["consistency_status"] == "feasible"
    assert affine_exact["minimum_squared_l2_norm_exact"] == ["2"]
    candidate = apply_program(base, affine.program)
    # SVD rounding can leave a nonzero protected residual even for this
    # feasible affine system. Check the stored endpoint without a tolerance.
    residual = abs(Fraction.from_float(float(candidate.tensors["head.weight"][0, 0])) +
                   Fraction.from_float(float(candidate.tensors["head.bias"][0])))
    certificate = affine.retention_certificate
    assert certificate["absolute_residuals_exact"] == [[str(residual)]]
    assert certificate["maximum_residuals_exact"] == [str(residual)]
    assert certificate["exactly_zero_on_supplied_features"] is (residual == 0)
    assert candidate.tensors["head.bias"][0] > .9


def test_wide_protection_preserves_missing_right_singular_vectors_and_allows_edit():
    retained, target, desired = [[1., 0., 0.], [0., 1., 0.]], [[0., 0., 2.]], [[3.]]
    basis, diagnostic = analyze(target, retained, desired)
    assert basis.shape == (3, 1)
    np.testing.assert_array_equal(np.asarray(retained) @ basis, np.zeros((2, 1)))
    assert diagnostic["protected"]["rank"] == 2
    assert diagnostic["projected_target"]["row_projected_energy"] == [4.]
    assert diagnostic["projected_target"]["row_projected_energy_fraction"] == [1.]
    assert diagnostic["exact"]["minimum_squared_l2_norm_exact"] == ["9/4"]
    base, head = head_fixture(3)
    compiled = compile_feature_edit(base, head, target, desired, retained, retention="nullspace")
    assert compiled.diagnostics["nullspace_dimension"] == 1 and not compiled.diagnostics["no_op"]
    assert compiled.retention_certificate["exactly_zero_on_supplied_features"] is True


def test_tall_protection_does_not_request_a_square_left_singular_matrix(monkeypatch):
    calls = []
    original = np.linalg.svd

    def record(matrix, *, full_matrices):
        calls.append((matrix.shape, full_matrices))
        return original(matrix, full_matrices=full_matrices)

    monkeypatch.setattr(feature_module.np.linalg, "svd", record)
    _, diagnostic = analyze([[1., 2., 3.]], np.vstack((np.eye(3), np.eye(3))), [[1.]])
    assert calls[0] == ((6, 3), False)
    assert diagnostic["protected"]["nullity"] == 0


def test_full_column_rank_protection_produces_a_noop_and_exact_infeasibility():
    base, head = head_fixture(2)
    compiled = compile_feature_edit(base, head, [[2., 3.]], [[1.]], np.eye(2), retention="nullspace")
    diagnostic = compiled.diagnostics
    assert compiled.program.edits == () and diagnostic["no_op"] is True
    assert diagnostic["nullspace_dimension"] == 0 and diagnostic["fit_rank"] == 0
    assert diagnostic["condition_number"] is None
    assert diagnostic["feature_constraints"]["exact"]["protected_only_zero_update"] is True
    assert diagnostic["feature_constraints"]["exact"]["consistency_status"] == "infeasible"
    assert diagnostic["feature_constraints"]["unregularized"]["target_residual_rmse"] == 1.
    assert diagnostic["target_rmse_after_projection"] == 1.
    assert diagnostic["realized_target_residuals_exact"] == [["-1"]]
    assert compiled.retention_certificate["exactly_zero_on_supplied_features"] is True


def test_tiny_nonzero_dyadic_is_exactly_protected_despite_numerical_truncation():
    retained, target, desired = [[1., 0.], [0., 2.**-50]], [[0., 1.]], [[1.]]
    base, head = head_fixture(2)
    compiled = compile_feature_edit(base, head, target, desired, retained, retention="nullspace")
    diagnostic = compiled.diagnostics["feature_constraints"]
    assert diagnostic["protected"]["rank"] == 1 and diagnostic["protected"]["nullity"] == 1
    assert diagnostic["exact"]["protected_rank_exact"] == 2
    assert diagnostic["exact"]["protected_only_zero_update"] is True
    assert diagnostic["exact"]["consistency_status"] == "infeasible"
    assert diagnostic["unregularized"]["target_residual_rmse"] == 0.
    assert compiled.retention_certificate["exactly_zero_on_supplied_features"] is False
    check_witnesses(diagnostic["exact"], target, retained, desired)


def test_nondyadic_real_solution_does_not_certify_the_rounded_endpoint():
    base, head = head_fixture(1)
    compiled = compile_feature_edit(base, head, [[3.]], [[1.]], retention="none", ridge=1e-30,
                                    native_dtypes={"head.weight": "float32"}, cast="round")
    diagnostic = compiled.diagnostics
    exact = diagnostic["feature_constraints"]["exact"]
    assert exact["consistency_status"] == "feasible"
    assert exact["minimum_squared_l2_norm_exact"] == ["1/9"]
    endpoint = apply_program(base, compiled.program).tensors["head.weight"][0, 0]
    residual = 3*Fraction.from_float(float(endpoint))-1
    assert residual == Fraction(1, 33554432)
    assert diagnostic["realized_target_residuals_exact"] == [[str(residual)]]
    assert diagnostic["realized_target_maximum_absolute_residuals_exact"] == [str(abs(residual))]
    assert diagnostic["realized_target_exactly_matches_requested_delta"] is False
    assert diagnostic["target_rmse_after_projection"] == float(abs(residual))


@pytest.mark.parametrize("retention,expected", [("none", 20/11), ("soft", 20/91)])
def test_constraint_diagnostics_do_not_change_none_or_soft_ridge_objectives(retention, expected):
    base, head = head_fixture(1)
    compiled = compile_feature_edit(base, head, [[1.], [2.]], [[2.], [4.]], [[1.], [3.]],
                                    retention=retention, ridge=.25, retention_weight=4.)
    diagnostic = compiled.diagnostics
    endpoint = apply_program(base, compiled.program).tensors["head.weight"][0, 0]
    assert endpoint == pytest.approx(expected)
    assert diagnostic["optimizer_uses_numerical_nullspace"] is False
    assert diagnostic["retention_rank"] is diagnostic["nullspace_dimension"] is diagnostic["singular_cutoff"] is None
    assert diagnostic["feature_constraints"]["exact"]["consistency_status"] == "infeasible"
    assert diagnostic["condition_number"] == 1.
    assert compiled.retention_certificate is not None
    assert diagnostic["retention_rmse_after_projection"] == pytest.approx(abs(endpoint)*np.sqrt(5))


def test_ridge_fit_and_realized_residual_are_separate_from_unregularized_feasibility():
    base, head = head_fixture(1)
    compiled = compile_feature_edit(base, head, [[1.], [2.]], [[2.], [4.]], retention="none", ridge=.25)
    diagnostic = compiled.diagnostics
    constraints = diagnostic["feature_constraints"]
    assert constraints["exact"]["minimum_squared_l2_norm_exact"] == ["4"]
    assert constraints["unregularized"]["edit_l2_norm_per_output"] == pytest.approx([2.])
    assert constraints["unregularized"]["target_residual_rmse"] == pytest.approx(0., abs=1e-14)
    assert diagnostic["target_rmse_after_projection"] > .2
    assert diagnostic["realized_target_exactly_matches_requested_delta"] is False


def test_zero_target_direction_and_zero_requested_deltas_are_feasible_without_edit():
    _, diagnostic = analyze([[0., 0.], [1., 0.]], [[1., 0.]], [[0., 0.], [0., 0.]])
    assert diagnostic["projected_target"]["rank"] == 0
    assert diagnostic["projected_target"]["row_projected_energy"] == [0., 0.]
    assert diagnostic["projected_target"]["row_projected_energy_fraction"] == [0., 0.]
    assert diagnostic["unregularized"]["target_residual_rmse"] == 0.
    assert diagnostic["unregularized"]["edit_frobenius_norm"] == 0.
    assert diagnostic["exact"]["minimum_squared_l2_norm_exact"] == ["0", "0"]
    assert diagnostic["exact"]["minimum_squared_frobenius_norm_exact"] == "0"


def test_exhausted_consistency_budget_does_not_invent_protected_rank_or_nullity():
    _, diagnostic = analyze([[0., 1.]], [[1., 0.]], [[1.]], exact_work_limit=0)
    exact = diagnostic["exact"]
    assert exact["consistency_status"] == "unresolved"
    assert exact["consistency_limit_reason"] == "work_limit"
    assert exact["protected_rank_exact"] is exact["protected_nullity_exact"] is None
    assert exact["protected_only_zero_update"] is exact["constraint_rank_exact"] is None
    assert exact["constraint_rank_lower_bound"] == exact["completed_constraint_rows"] == exact["work_used"] == 0
    assert exact["minimum_norm_status"] == "unresolved"


def test_found_conflict_survives_later_exhaustion_without_claiming_other_outputs_feasible():
    _, first = analyze([[0.]], None, [[1., 0.]])
    target, desired = [[0.], [1.]], [[1., 0.], [2., 3.]]
    _, diagnostic = analyze(target, None, desired,
                            exact_work_limit=first["exact"]["consistency_work_used"])
    exact = diagnostic["exact"]
    assert exact["consistency_status"] == "infeasible"
    assert exact["per_output_consistency"] == ["infeasible", "unresolved"]
    assert exact["consistency_limit_reason"] == "work_limit"
    assert exact["constraint_rank_exact"] is None
    check_witnesses(exact, target, None, desired)


def test_rational_growth_limit_reports_unresolved_instead_of_exact_rank():
    target = np.eye(5)*2.**-1000 + np.diag(np.ones(4), k=-1)
    _, diagnostic = analyze(target, None, [[1.], [0.], [0.], [0.], [0.]])
    exact = diagnostic["exact"]
    assert exact["consistency_status"] == "unresolved"
    assert exact["consistency_limit_reason"] == "rational_bit_limit"
    assert exact["rational_bit_limit"] == 4096
    assert exact["constraint_rank_exact"] is None
    assert exact["constraint_rank_lower_bound"] < 5
    assert exact["work_used"] < exact["work_limit"]


def test_minimum_norm_exhaustion_preserves_completed_feasibility_and_zero_norms():
    target, retained, desired = [[1., 0., 1.]], [[1., 1., 0.]], [[0., 1.]]
    _, complete = analyze(target, retained, desired)
    _, limited = analyze(target, retained, desired,
                         exact_work_limit=complete["exact"]["consistency_work_used"])
    exact = limited["exact"]
    assert exact["consistency_status"] == "feasible"
    assert exact["consistency_limit_reason"] is None
    assert exact["constraint_rank_exact"] == 2
    assert exact["minimum_norm_status"] == "unresolved"
    assert exact["minimum_norm_limit_reason"] == "work_limit"
    assert exact["per_output_minimum_norm_status"] == ["complete", "unresolved"]
    assert exact["minimum_squared_l2_norm_exact"] == ["0", None]
    assert exact["minimum_squared_frobenius_norm_exact"] is None


def test_later_norm_exhaustion_preserves_an_already_completed_nonzero_norm():
    target, retained, desired = [[1., 0., 1.]], [[1., 1., 0.]], [[1., 2.]]
    _, complete = analyze(target, retained, desired)
    lower, upper = complete["exact"]["consistency_work_used"], complete["exact"]["work_used"]
    # Locate a budget boundary from reported work rather than baking in operation counts.
    while upper-lower > 1:
        budget = (lower+upper)//2
        _, probe = analyze(target, retained, desired, exact_work_limit=budget)
        if probe["exact"]["per_output_minimum_norm_status"][0] == "complete":
            upper = budget
        else:
            lower = budget
    _, diagnostic = analyze(target, retained, desired, exact_work_limit=upper)
    exact = diagnostic["exact"]
    assert exact["consistency_status"] == "feasible"
    assert exact["per_output_minimum_norm_status"] == ["complete", "unresolved"]
    assert exact["minimum_squared_l2_norm_exact"] == ["2/3", None]


def test_geometry_without_desired_deltas_keeps_modular_lower_bound_semantics():
    retained, target = np.array([[0.]]), np.array([[1.], [2.]])
    geometry = feature_geometry(retained, target)
    assert geometry["augmented_width"] == 2 and geometry["numerical_nullity"] == 1
    assert geometry["exact_score_lane"] == "unresolved_by_modular_lower_bound"
    assert geometry["exact_rank_witness"] == modular_rank_witness(retained)
    assert geometry["feature_constraints"]["exact"]["consistency_status"] == "not_requested"
    assert geometry["feature_constraints"]["unregularized"] is None
    assert geometry["feature_constraints"]["exact"]["work_used"] == 0
    assert geometry["target_projected_energy_fraction"] == pytest.approx([1/2, 4/5])
    with_deltas = feature_geometry(retained, target, desired_delta=[[1.], [3.]])
    assert with_deltas["exact_rank_witness"] == geometry["exact_rank_witness"]
    assert with_deltas["exact_score_lane"] == geometry["exact_score_lane"]
    assert with_deltas["feature_constraints"]["exact"]["consistency_status"] == "infeasible"


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_finite_desired_deltas_keep_existing_scaled_rmse_behavior(scale):
    base, head = head_fixture(1)
    compiled = compile_feature_edit(base, head, [[1.]], [[scale]], retention="none")
    diagnostic = compiled.diagnostics
    assert diagnostic["target_rmse_after_projection"]/scale == pytest.approx(.01/1.01)
    assert diagnostic["feature_constraints"]["unregularized"]["edit_frobenius_norm"]/scale == pytest.approx(1.)
    assert diagnostic["feature_constraints"]["exact"]["consistency_status"] == "feasible"


def test_small_singular_value_and_small_desired_delta_have_finite_unregularized_norm():
    _, diagnostic = analyze([[1e-320]], None, [[1e-320]])
    assert diagnostic["unregularized"]["edit_l2_norm_per_output"] == pytest.approx([1.])
    assert diagnostic["unregularized"]["edit_frobenius_norm"] == pytest.approx(1.)
    assert diagnostic["exact"]["minimum_squared_l2_norm_exact"] == ["1"]


def test_large_singular_value_and_mixed_desired_scales_keep_a_nonzero_edit_norm():
    _, diagnostic = analyze([[1e308], [0.]], None, [[1e100], [1e308]])
    assert diagnostic["unregularized"]["edit_l2_norm_per_output"] == pytest.approx([1e-208], abs=0.)
    assert diagnostic["unregularized"]["edit_frobenius_norm"] == pytest.approx(1e-208, abs=0.)
    assert canonical_bytes(diagnostic)


def test_large_protected_coordinate_does_not_erase_a_small_projected_signal():
    _, diagnostic = analyze([[1e308, 1e-100]], [[1., 0.]], [[1e-100]])
    projected = diagnostic["projected_target"]
    assert projected["rank"] == 1
    assert projected["singular_values"] == pytest.approx([1e-100], abs=0.)
    assert projected["row_projected_energy"] == pytest.approx([1e-200], abs=0.)
    assert diagnostic["unregularized"]["edit_l2_norm_per_output"] == pytest.approx([1.])
    assert canonical_bytes(diagnostic)


def test_projected_row_energy_uses_its_own_scale():
    _, diagnostic = analyze([[1e200, 1.]], [[1., 0.]], [[1.]])
    assert diagnostic["projected_target"]["row_projected_energy"] == [1.]
    assert diagnostic["unregularized"]["edit_l2_norm_per_output"] == pytest.approx([1.])


@pytest.mark.parametrize("target", [[[1.], [0.]], [[1e-200], [0.]], [[0.], [1.]]])
def test_lossy_response_normalization_reports_unavailable_instead_of_false_zero(target):
    _, diagnostic = analyze(target, None, [[1e-100], [1e308]])
    unregularized = diagnostic["unregularized"]
    assert unregularized["target_residual_rmse"] is None
    assert unregularized["target_residual_l2_per_output"] == [None]
    assert unregularized["edit_l2_norm_per_output"] == [None]
    assert unregularized["edit_frobenius_norm"] is None
    assert unregularized["per_output_numerical_limit_reason"] == ["response_normalization_underflow"]
    assert diagnostic["exact"]["consistency_status"] == "infeasible"
    assert canonical_bytes(diagnostic)


def test_lossy_response_output_does_not_hide_an_unaffected_outputs_diagnostics():
    _, diagnostic = analyze([[1.], [0.]], None, [[1e-100, 2.], [1e308, 0.]])
    unregularized = diagnostic["unregularized"]
    assert unregularized["target_residual_l2_per_output"] == [None, 0.]
    assert unregularized["edit_l2_norm_per_output"] == [None, 2.]
    assert unregularized["per_output_numerical_limit_reason"] == ["response_normalization_underflow", None]
    assert diagnostic["exact"]["per_output_consistency"] == ["infeasible", "feasible"]


def test_response_projection_underflow_reports_unavailable_and_preserves_other_outputs():
    _, diagnostic = analyze([[1e-100], [1e-300], [0.]], None,
                            [[0., 1.], [1e100, 0.], [1e300, 0.]])
    unregularized = diagnostic["unregularized"]
    assert unregularized["edit_l2_norm_per_output"][0] is None
    assert unregularized["target_residual_l2_per_output"][0] is None
    assert unregularized["edit_l2_norm_per_output"][1] == pytest.approx(1e100)
    assert unregularized["target_residual_l2_per_output"][1] == pytest.approx(1e-200, abs=0.)
    assert unregularized["edit_frobenius_norm"] is None
    assert unregularized["target_residual_rmse"] is None
    assert unregularized["per_output_numerical_limit_reason"] == ["response_projection_underflow", None]
    assert diagnostic["exact"]["consistency_status"] == "infeasible"
    assert canonical_bytes(diagnostic)


def test_unrepresentable_projected_matrix_reports_a_numerical_limit():
    largest = np.finfo(float).max
    _, diagnostic = analyze([[largest, largest]], [[1., -1.]], [[0.]])
    assert diagnostic["projected_target"]["rank"] is None
    assert diagnostic["projected_target"]["numerical_limit_reason"] == "projected_matrix_out_of_range"
    assert diagnostic["unregularized"]["edit_frobenius_norm"] is None
    assert diagnostic["exact"]["consistency_status"] == "feasible"
    assert canonical_bytes(diagnostic)


def test_projection_scaling_loss_preserves_independent_physical_row_energy():
    _, diagnostic = analyze([[1e308, 0.], [0., 1e-100]], None, [[0.], [0.]])
    projected = diagnostic["projected_target"]
    assert projected["rank"] is None
    assert projected["numerical_limit_reason"] == "projected_matrix_scaling_underflow"
    assert projected["row_projected_energy"][0] is None
    assert projected["row_projected_energy"][1] == pytest.approx(1e-200, abs=0.)
    assert diagnostic["exact"]["consistency_status"] == "feasible"
    assert canonical_bytes(diagnostic)


@pytest.mark.parametrize("retention", ["none", "soft", "nullspace"])
def test_compiler_serializes_lossy_response_diagnostics_in_all_modes(retention):
    base, head = head_fixture(1)
    compiled = compile_feature_edit(base, head, [[1.], [0.]], [[1e-100], [1e308]], [[0.]],
                                    retention=retention)
    unregularized = compiled.diagnostics["feature_constraints"]["unregularized"]
    assert unregularized["edit_l2_norm_per_output"] == [None]
    assert unregularized["target_residual_rmse"] is None
    assert unregularized["per_output_numerical_limit_reason"] == ["response_normalization_underflow"]
    assert canonical_bytes(compiled.diagnostics)


def test_geometry_serializes_lossy_response_diagnostics():
    geometry = feature_geometry(np.array([[0.]]), np.array([[1.], [0.]]),
                                desired_delta=[[1e-100], [1e308]])
    unregularized = geometry["feature_constraints"]["unregularized"]
    assert unregularized["edit_l2_norm_per_output"] == [None]
    assert unregularized["target_residual_rmse"] is None
    assert unregularized["per_output_numerical_limit_reason"] == ["response_normalization_underflow"]
    assert canonical_bytes(geometry)


@pytest.mark.parametrize("retention", ["none", "soft", "nullspace"])
def test_unrepresentable_diagnostic_spectrum_does_not_break_a_finite_zero_endpoint(retention):
    base, head = head_fixture(1)
    anchors = None if retention == "none" else [[0.]]
    compiled = compile_feature_edit(base, head, [[2.**1023]]*4, [[0.]]*4, anchors, retention=retention)
    diagnostic = compiled.diagnostics
    assert diagnostic["no_op"] is True and diagnostic["target_rmse_after_projection"] == 0.
    projected = diagnostic["feature_constraints"]["projected_target"]
    assert projected["rank"] == 1
    assert projected["singular_values"] == [None]
    assert projected["scaled_singular_values"] == pytest.approx([2.])
    assert projected["singular_values_scale"] == 2.**1023
    assert diagnostic["feature_constraints"]["exact"]["consistency_status"] == "feasible"
    assert canonical_bytes(diagnostic)


@pytest.mark.parametrize("retention", ["none", "soft", "nullspace"])
def test_unrepresentable_retained_condition_number_is_explicitly_unavailable(retention):
    base, head = head_fixture(2)
    compiled = compile_feature_edit(base, head, [[0., 0.]], [[0.]], [[1., 0.], [0., 1e-310]],
                                    retention=retention, nullspace_rtol=1e-320)
    protected = compiled.diagnostics["feature_constraints"]["protected"]
    assert protected["rank"] == 2
    assert protected["condition_number"] is None
    assert protected["retained_condition_number"] is None
    assert canonical_bytes(compiled.diagnostics)


@pytest.mark.parametrize("large_protection", [False, True])
def test_geometry_extreme_finite_spectra_remain_canonical_json(large_protection):
    ordinary, large = np.array([[0.]]), np.array([[2.**1023]]*4)
    retained, target = (large, ordinary) if large_protection else (ordinary, large)
    geometry = feature_geometry(retained, target, desired_delta=np.zeros((len(target), 1)))
    constraints = geometry["feature_constraints"]
    spectrum = constraints["protected"] if large_protection else constraints["projected_target"]
    assert spectrum["singular_values"][0] is None
    assert spectrum["rank"] == 1
    assert constraints["exact"]["consistency_status"] == "feasible"
    assert canonical_bytes(geometry)
