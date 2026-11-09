"""Adversarial and mathematical checks for the weight-programming trust boundary."""
from fractions import Fraction
from itertools import permutations, product
import json

import numpy as np
import pytest

from rlcd.weight_space import (Network, InputBox, Program, Edit, Contract, apply_program,
    difference, compose, certify_drift, verify_certificate, check_contract, compile_readout,
    Gauge, transform, transport, align_hidden, align_hidden_checked)
from rlcd.weight_space.adapters import from_rts, to_rts
from rlcd.weight_space.contracts import exact_score, preserves_argmax
from rlcd.weight_space.program import rollback, interpolate
from rlcd.weight_space.symmetry import _assignment, transfer_edit


def network(seed=1):
    rng = np.random.default_rng(seed)
    return Network(rng.normal(size=(4, 3)), rng.normal(size=4),
                   rng.normal(size=(2, 4)), rng.normal(size=2))


def test_immutable_identity_and_strict_schemas():
    base = network()
    with pytest.raises(ValueError):
        base.w1[0, 0] = 2
    with pytest.raises(ValueError):
        base.w1.flags.writeable = True
    assert Network.from_dict(base.to_dict()).sha256 == base.sha256
    for bad in (True, "2", float("nan"), float("inf")):
        value = base.to_dict()
        value["b2"] = [bad, bad]
        with pytest.raises(ValueError):
            Network.from_dict(value)
    with pytest.raises(ValueError):
        Network.from_dict(base.to_dict() | {"activation": "tanh"})
    z = Network([[1]], [0], [[1]], [-0.0])
    assert z.sha256 == z.replace(b2=[0.0]).sha256


def test_program_atomicity_lineage_composition_and_rollback():
    base = network()
    candidate = base.replace(b2=base.b2+1)
    patch = difference(base, candidate, "Raise both outputs")
    assert apply_program(base, Program.from_dict(patch.to_dict())).sha256 == candidate.sha256
    assert apply_program(candidate, rollback(base, patch)).sha256 == base.sha256
    next_model = candidate.replace(w2=candidate.w2*2)
    next_patch = difference(candidate, next_model, "Scale the readout")
    assert apply_program(base, compose(base, [patch, next_patch])).sha256 == next_model.sha256
    for stale in (candidate, network(2)):
        with pytest.raises(ValueError, match="parent"):
            apply_program(stale, patch)
    with pytest.raises(ValueError, match="Duplicate"):
        Program(base.sha256, (patch.edits[0], patch.edits[0]), "Ambiguous")
    invalid = Program(base.sha256, patch.edits + (Edit("w1", (200, 0), 1., 2.),), "Bad trailing cell")
    original = base.sha256
    with pytest.raises(ValueError, match="outside"):
        apply_program(base, invalid)
    assert base.sha256 == original
    with pytest.raises(ValueError, match="precondition"):
        apply_program(base, Program(base.sha256, (Edit("b2", (0,), 987., 1.),), "Wrong cell"))
    assert apply_program(base, interpolate(base, candidate, 0)).sha256 == base.sha256
    assert apply_program(base, interpolate(base, candidate, 1)).sha256 == candidate.sha256
    other = Network(**base.tensors, input_names=("different", "schema", "order"))
    with pytest.raises(ValueError):
        difference(base, other, "Invalid schema transfer")


def test_known_gauge_preserves_exact_function_and_edit_semantics():
    base = Network([[1, 2], [-3, 1], [2, -1]], [0, 1, -2], [[2, -1, 3]], [1])
    gauge = Gauge((2, 0, 1), (2., .5, 4.))
    other = transform(base, gauge)
    edited = base.replace(w1=base.w1+1, w2=base.w2+np.array([[0., 1., 0.]]))
    _, transferred = transport(base, difference(base, edited, "Mixed-layer edit"), gauge)
    after = apply_program(other, transferred)
    for x in product((-1., 0., .5, 1.), repeat=2):
        assert exact_score(base, list(x)) == exact_score(other, list(x))
        assert exact_score(edited, list(x)) == exact_score(after, list(x))
    assert transform(other, gauge.inverse()).sha256 == base.sha256
    with pytest.raises(ValueError):
        Gauge((0, 0, 2), (1, 1, 1))
    with pytest.raises(ValueError):
        Gauge((0, 1, 2), (1, -1, 1))


def test_assignment_matches_brute_force_and_recovers_identifiable_gauge():
    rng = np.random.default_rng(5)
    for n in range(1, 6):
        costs = rng.uniform(size=(n, n))
        actual = _assignment(costs)
        optimum = min(sum(costs[i, p[i]] for i in range(n)) for p in permutations(range(n)))
        assert sum(costs[i, actual[i]] for i in range(n)) == pytest.approx(optimum)
    source = network(4)
    gauge = Gauge((2, 0, 3, 1), (.5, 2., 1., 4.))
    target = transform(source, gauge)
    inferred, diagnostics = align_hidden(source, target)
    assert inferred.permutation == gauge.permutation
    assert inferred.scales == pytest.approx(gauge.scales)
    assert diagnostics["relative_parameter_residual"] < 1e-14
    program = difference(source, source.replace(w2=source.w2+.3), "Transfer readout edit")
    mapped = apply_program(target, transfer_edit(source, program, target, inferred))
    x = rng.uniform(-1, 1, (20, 3))
    np.testing.assert_allclose(mapped(x), apply_program(source, program)(x), atol=1e-13)


def test_checked_alignment_literal_identifiability_and_abstention_boundaries():
    # Fixed before these literal outcomes; this is not a trained-study threshold.
    threshold = 2.0**-18
    source = Network([[1, 0], [0, 1], [-1, 0]], [0, 0, .5], [[1, 2, -1]], [.25])
    edited = source.replace(w1=[[1.25, .25], [0, 1], [-1, 0]],
                            b1=[.25, 0, .5], w2=[[1.5, 2, -1]], b2=[.5])
    known = Gauge((2, 0, 1), (2., .5, 4.))
    target = transform(source, known)
    checked = align_hidden_checked(source, target, min_assignment_gap=threshold)
    outcomes = [checked]
    assert checked.gauge == known and checked.reason == "separated_assignment"
    assert checked.gap_status == "finite" and checked.assignment_gap is not None
    assert checked.assignment_gap - threshold > checked.diagnostics["numerical_guard"]
    legacy, diagnostics = align_hidden(source, target)
    assert legacy == known
    assert set(diagnostics) == {"mean_matching_cost", "relative_parameter_residual"}
    program = difference(source, edited, "Literal mixed-layer edit")
    mapped_parent, mapped_program = transport(source, program, checked.gauge)
    assert mapped_parent.sha256 == target.sha256
    mapped = apply_program(target, mapped_program)
    for x, expected in (((-1., 0.), Fraction(-1)), ((0., 0.), Fraction(3, 8)),
                        ((1., .5), Fraction(63, 16)), ((2., -1.), Fraction(17, 4))):
        assert exact_score(edited, list(x)) == (expected,)
        assert exact_score(mapped, list(x)) == (expected,)

    renumbered = align_hidden_checked(
        transform(source, Gauge((2, 0, 1), (1., 1., 1.))),
        transform(target, Gauge((1, 2, 0), (1., 1., 1.))),
        min_assignment_gap=threshold)
    outcomes.append(renumbered)
    assert renumbered.gauge == Gauge((1, 2, 0), (.5, 4., 2.))
    assert renumbered.reason == checked.reason and renumbered.gap_status == checked.gap_status
    assert renumbered.assignment_gap == pytest.approx(
        checked.assignment_gap, rel=0, abs=max(checked.diagnostics["numerical_guard"],
                                            renumbered.diagnostics["numerical_guard"]))

    # Unit-norm rows give this literal squared-distance matrix independently
    # of the matcher. The cheapest alternative keeps selected edge (0, 0).
    oracle_cost = ((0, 4, 3), (4, 0, 1), (3, 1, 0))
    oracle = sorted((sum(oracle_cost[row][col] for row, col in enumerate(p)), p)
                    for p in permutations(range(3)))
    assert oracle[0] == (0, (0, 1, 2))
    assert oracle[1] == (2, (0, 2, 1))
    assert min(value for value, p in oracle if p[0] != 0) == 6
    oracle_gap = oracle[1][0] - oracle[0][0]
    global_source = Network([[-1, 0, 0, 0], [1, 0, 0, 0], [.5, .5, .5, .5]],
                            [0, 0, 0], [[1, 2, 3]], [0])
    identity = Gauge((0, 1, 2), (1., 1., 1.))
    # Literal thresholds: clear separation, inside the guard, and exact equality.
    for frozen_gap, accepted in ((1.5, True), (2.0 - 2.0**-44, False), (2.0, False)):
        result = align_hidden_checked(global_source, global_source,
                                      min_assignment_gap=frozen_gap)
        outcomes.append(result)
        assert result.gap_status == "finite" and result.assignment_gap == oracle_gap
        assert result.diagnostics["assignment_objective"] == oracle[0][0]
        assert result.diagnostics["best_alternative_objective"] == oracle[1][0]
        assert result.gauge == (identity if accepted else None)
        assert result.reason == ("separated_assignment" if accepted
                                 else "assignment_not_separated")

    # Equal parent functions do not identify which duplicated feature was edited.
    duplicate_source = Network([[1], [1]], [0, 0], [[1, 2]], [0])
    duplicate_target = Network([[1], [1]], [0, 0], [[2, 1]], [0])
    duplicate_edit = difference(duplicate_source, duplicate_source.replace(w1=[[2], [1]]),
                                "Edit the first duplicate")
    legacy_duplicate, _ = align_hidden(duplicate_source, duplicate_target)
    legacy_after = apply_program(duplicate_target, transfer_edit(
        duplicate_source, duplicate_edit, duplicate_target, legacy_duplicate))
    known_parent, known_edit = transport(duplicate_source, duplicate_edit, Gauge((1, 0), (1., 1.)))
    assert known_parent.sha256 == duplicate_target.sha256
    assert exact_score(duplicate_source, [1]) == exact_score(duplicate_target, [1]) == (Fraction(3),)
    assert exact_score(apply_program(duplicate_source, duplicate_edit), [1]) == (Fraction(4),)
    assert exact_score(legacy_after, [1]) == (Fraction(5),)
    assert exact_score(apply_program(known_parent, known_edit), [1]) == (Fraction(4),)

    tiny = 2.0**-20
    near_source = Network([[1], [0]], [0, 1], [[1, 1]], [0])
    near_target = Network([[1], [1]], [0, tiny], [[1, 1]], [0])
    # Both rows prefer the first source column, with local cost gaps > 1.
    # Bijection forces a different comparison: the global gap is below threshold.
    assert 2 * (1 - tiny) / np.sqrt(1 + tiny*tiny) > 1
    for left, right, duplicate in ((duplicate_source, duplicate_target, True),
                                    (near_source, near_target, False)):
        result = align_hidden_checked(left, right, min_assignment_gap=threshold)
        permuted = align_hidden_checked(
            transform(left, Gauge((1, 0), (1., 1.))),
            transform(right, Gauge((0, 1), (1., 1.))), min_assignment_gap=threshold)
        outcomes.extend((result, permuted))
        assert result.gauge is None and permuted.gauge is None
        assert result.reason == permuted.reason == "assignment_not_separated"
        assert result.gap_status == permuted.gap_status == "finite"
        assert result.assignment_gap is not None
        if duplicate:
            assert result.assignment_gap == 0
        else:
            assert 0 < result.assignment_gap < threshold
        assert permuted.assignment_gap == pytest.approx(
            result.assignment_gap, rel=0, abs=max(result.diagnostics["numerical_guard"],
                                                permuted.diagnostics["numerical_guard"]))
        # An abstention is retained here; it is never passed as a raw-transfer gauge.

    zero = Network([[0], [1]], [0, 0], [[1, 1]], [0])
    nonzero = Network([[1], [1]], [0, 0], [[1, 1]], [0])
    for left, right, source_rows, target_rows in ((zero, nonzero, [0], []),
                                                (nonzero, zero, [], [0]), (zero, zero, [0], [0])):
        result = align_hidden_checked(left, right, min_assignment_gap=threshold)
        outcomes.append(result)
        assert result.gauge is None and result.reason == "zero_incoming_affine_row"
        assert result.assignment_gap is None and result.gap_status == "unavailable"
        assert result.diagnostics["source_zero_affine_rows"] == source_rows
        assert result.diagnostics["target_zero_affine_rows"] == target_rows

    single = Network([[1]], [0], [[2]], [.5])
    singleton = align_hidden_checked(single, Network([[2]], [0], [[1]], [.5]),
                                     min_assignment_gap=threshold)
    outcomes.append(singleton)
    assert singleton.gauge == Gauge((0,), (2.,))
    assert singleton.reason == "single_hidden_unit"
    assert singleton.assignment_gap is None and singleton.gap_status == "no_alternative"
    for magnitude in (1e308, 2.0**-600):
        unresolved = Network([[magnitude]], [0], [[1]], [0])
        result = align_hidden_checked(unresolved, unresolved, min_assignment_gap=threshold)
        outcomes.append(result)
        assert result.gauge is None and result.reason == "numerical_failure"
        assert result.assignment_gap is None and result.gap_status == "unavailable"
        assert result.diagnostics["failure_stage"] == "incoming_norms"
        assert result.diagnostics["source_zero_affine_rows"] == []
        assert result.diagnostics["target_zero_affine_rows"] == []
    for bad in (True, -1., float("nan"), float("inf"), "0", 2**54 + 1):
        with pytest.raises(ValueError):
            align_hidden_checked(source, target, min_assignment_gap=bad)
    with pytest.raises(TypeError):
        align_hidden_checked(source, target)
    for result in outcomes:
        json.dumps(result.diagnostics, allow_nan=False)
        assert result.assignment_gap is None or np.isfinite(result.assignment_gap)


def test_all_layer_exact_box_certificate_covers_exhaustive_rational_grid_and_tampering():
    base = Network([[1, -2], [2, 1]], [-1, 0], [[2, -1], [-1, 3]], [1, -2])
    candidate = base.replace(w1=[[1.25, -1.5], [2.25, 1]], b1=[-.5, -.25],
                             w2=[[2.5, -.75], [-.5, 2.75]], b2=[1.25, -1.75])
    box = InputBox([-1, -.5], [1, 1.5])
    certificate = certify_drift(base, candidate, box)
    bounds = tuple(map(Fraction, certificate["output_bounds"]))
    for x in product(np.linspace(-1, 1, 9), np.linspace(-.5, 1.5, 9)):
        delta = [abs(a-b) for a, b in zip(exact_score(candidate, list(x)), exact_score(base, list(x)))]
        assert all(d <= b for d, b in zip(delta, bounds))
    assert verify_certificate(base, candidate, certificate)
    for key, value in (("output_bounds", ["0", "0"]), ("base_sha256", "0"*64),
                       ("domain", {"lower": [0, 0], "upper": [0, 0]})):
        assert not verify_certificate(base, candidate, certificate | {key: value})


def test_drift_contract_blocks_parameter_bounded_but_semantically_large_edit():
    base = Network([[64]], [0], [[1]], [0])
    patch = difference(base, base.replace(w2=[[2]]), "One bounded coefficient changes the score by64")
    receipt = check_contract(base, patch, Contract(InputBox([0], [1]), (1.,)))
    assert not receipt["accepted"] and receipt["issues"] == ["behavioral-drift"]
    assert receipt["certificate"]["output_bounds"] == ["64"]
    assert preserves_argmax([4., 1., 0.], ["1", "1", "1"])
    assert not preserves_argmax([3., 1., 0.], ["1", "1", "1"])
    assert not preserves_argmax([1., 1.], ["0", "0"])


def test_synthesis_and_retention_use_actual_behavior_not_coordinate_size():
    base = Network([[1], [-1]], [0, 0], [[1, 1]], [0])
    target = [[.25], [.5], [1.]]
    retain = [[-.25], [-.5], [-1.]]
    compilation = compile_readout(base, target, [[.25], [.5], [1.]], retain,
                                  ridge=1e-9, editable_hidden=[0])
    candidate = apply_program(base, compilation.program)
    np.testing.assert_allclose(candidate(target)-base(target), target, atol=1e-8)
    np.testing.assert_array_equal(candidate(retain), base(retain))
    with pytest.raises(ValueError):
        compile_readout(base, [], [], retain)
    with pytest.raises(ValueError):
        compile_readout(base, target, [[1.]], retain)


def test_actual_rts_abi_and_zero_drift_protected_region():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    template = json.loads((root/"game/src/empires/weights/commander-v1.json").read_text())
    base = from_rts(template)
    assert base.parameter_count == 1293 and base.shape == (32, 38, 1)
    h = template["hiddenNames"].index("army_train_bias")
    weights = base.w2.copy()
    weights[0, h] += 1
    candidate = base.replace(w2=weights)
    assert from_rts(to_rts(candidate, template, description="Candidate")).sha256 == candidate.sha256
    lower, upper = np.zeros(32), np.ones(32)
    upper[24] = 0
    assert certify_drift(base, candidate, InputBox(lower, upper))["output_bounds"] == ["0"]
    assert certify_drift(base, candidate, InputBox(np.zeros(32), np.ones(32)))["output_bounds"] == ["1"]
    bad = template | {"featureNames": list(reversed(template["featureNames"]))}
    with pytest.raises(ValueError):
        from_rts(bad)
