"""Literal engineering examples; no model, research loader or study data."""

import numpy as np
import pytest

from rlcd.weight_space.metrics import correctness_transition_counts


def test_authored_corrections_regressions_and_boolean_boundaries():
    expected = np.array([0, 0, 1, 1, 2, 2])
    parent = np.array([1, 0, 0, 1, 2, 0])
    candidate = np.array([0, 1, 1, 1, 2, 2])
    parent_correct, candidate_correct = parent == expected, candidate == expected
    target = np.array([True, False, True, False, False, False])
    protected = ~target
    target_counts = correctness_transition_counts(parent_correct, candidate_correct, target)
    protected_counts = correctness_transition_counts(parent_correct, candidate_correct, protected)
    assert target_counts == {
        "rows": 2, "stayed_correct": 0, "corrected": 2,
        "newly_incorrect": 0, "stayed_incorrect": 0,
    }
    assert protected_counts == {
        "rows": 4, "stayed_correct": 2, "corrected": 1,
        "newly_incorrect": 1, "stayed_incorrect": 0,
    }
    assert target_counts["corrected"] - target_counts["newly_incorrect"] == 2
    assert protected_counts["newly_incorrect"] - protected_counts["corrected"] == 0
    assert np.count_nonzero(parent[protected] != candidate[protected]) == 2
    for counts in (target_counts, protected_counts):
        assert all(type(value) is int for value in counts.values())
        assert sum(value for key, value in counts.items() if key != "rows") == counts["rows"]

    # Different wrong decisions are still incorrect; flips need decoded outputs.
    assert correctness_transition_counts([False], [False], [True]) == {
        "rows": 1, "stayed_correct": 0, "corrected": 0,
        "newly_incorrect": 0, "stayed_incorrect": 1,
    }
    zeroes = {"rows": 0, "stayed_correct": 0, "corrected": 0,
              "newly_incorrect": 0, "stayed_incorrect": 0}
    assert correctness_transition_counts(parent_correct, candidate_correct, target & protected) == zeroes
    empty = np.array([], dtype=bool)
    assert correctness_transition_counts(empty, empty, empty) == zeroes
    assert correctness_transition_counts(parent_correct.tolist(), candidate_correct.tolist(),
                                         target.tolist()) == target_counts
    assert correctness_transition_counts(tuple(parent_correct), tuple(candidate_correct),
                                         tuple(target)) == target_counts

    for invalid in (True, [[True] * 6], [True] * 5, [0] * 6,
                    [1.0] * 6, ["True"] * 6, np.array([True] * 6, dtype=object),
                    np.ma.array([True] * 6, mask=[True, False, False, False, False, False]),
                    [np.ma.array(True, mask=True)] * 6, [None] * 6):
        for position in range(3):
            arguments = [parent_correct, candidate_correct, target]
            arguments[position] = invalid
            with pytest.raises(ValueError, match="Boolean|matching shapes"):
                correctness_transition_counts(*arguments)
