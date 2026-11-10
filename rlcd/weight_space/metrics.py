"""Task-independent counts from caller-defined correctness and region masks."""

import numpy as np


def correctness_transition_counts(parent_correct, candidate_correct, rows) -> dict[str, int]:
    """Count four correctness transitions in the selected rows.

    Inputs must be aligned, one-dimensional Boolean arrays or Boolean
    lists/tuples. Masked arrays and missing values are rejected before
    conversion. Empty Boolean arrays or selections return zero counts.
    Callers own prediction decoding, correctness, region policy and rates.
    A change between two wrong decisions remains ``stayed_incorrect``;
    decision flips require the original decoded outputs.
    """
    arrays = []
    for name, value in zip(("parent_correct", "candidate_correct", "rows"),
                           (parent_correct, candidate_correct, rows)):
        if np.ma.isMaskedArray(value) or (
            isinstance(value, (list, tuple))
            and any(type(item) not in (bool, np.bool_) for item in value)
        ):
            raise ValueError(f"{name} must contain only known Boolean values")
        value = np.asarray(value)
        if value.ndim != 1 or value.dtype != np.bool_:
            raise ValueError(f"{name} must be a one-dimensional Boolean array")
        arrays.append(value)
    parent, candidate, selected = arrays
    if parent.shape != candidate.shape or parent.shape != selected.shape:
        raise ValueError("correctness and row masks must have matching shapes")
    parent, candidate = parent[selected], candidate[selected]
    return {
        "rows": int(parent.size),
        "stayed_correct": int(np.count_nonzero(parent & candidate)),
        "corrected": int(np.count_nonzero(~parent & candidate)),
        "newly_incorrect": int(np.count_nonzero(parent & ~candidate)),
        "stayed_incorrect": int(np.count_nonzero(~parent & ~candidate)),
    }
