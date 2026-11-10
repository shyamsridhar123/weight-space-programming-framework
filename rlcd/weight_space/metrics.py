"""Task-independent correctness counts and observed resource summaries."""

import math

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


def observed_resource_summary(values) -> dict:
    """Summarize one field/unit/scope across all assigned records.

    ``None`` is unknown; zero is observed. Only plain nonnegative Python
    integers and finite floats are accepted. Integer-only sums are exact;
    any float selects binary64 ``math.fsum`` and its precision limits.
    Overflow is rejected. Empty assignments are complete with unknown sums.
    Availability does not establish instrumentation or lifecycle coverage.
    """
    observed = []
    assigned = 0
    for value in values:
        assigned += 1
        if value is None:
            continue
        if ((type(value) is not int and type(value) is not float) or value < 0
                or (type(value) is float and not math.isfinite(value))):
            raise ValueError("Resource values must be nonnegative Python integers or finite floats, or None")
        observed.append(value)
    total = None
    if observed:
        try:
            total = (sum(observed) if all(type(value) is int for value in observed)
                     else math.fsum(observed))
        except (OverflowError, ValueError) as error:
            raise ValueError("Resource sum exceeds finite binary64 range") from error
        if type(total) is float and not math.isfinite(total):
            raise ValueError("Resource sum exceeds finite binary64 range")
    missing = assigned-len(observed)
    return {
        "assigned": assigned,
        "observed": len(observed),
        "missing": missing,
        "observed_sum": total,
        "complete_for_assigned": missing == 0,
        "total_if_complete": total if missing == 0 else None,
    }
