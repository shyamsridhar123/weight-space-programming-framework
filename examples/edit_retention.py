"""Illustrate target change, protected behavior, and the realized endpoint.

Run from the repository root after installing the package:
    python examples/edit_retention.py

This is a constructed linear-head example, not a study observation or an LLM
trial. The exact cases use explicit edits; the numerical compiler has its own
positive-ridge objective and therefore need not attain the requested change.
"""
from __future__ import annotations

from fractions import Fraction
import json

from rlcd.weight_space import (
    ReadoutHead, StructuralPolicy, TensorArtifact, apply_program,
    certify_feature_retention, compile_feature_edit, difference, rollback,
    verify_feature_retention,
)


def real_output(artifact: TensorArtifact, feature: tuple[int, int]) -> Fraction:
    """Interpret the stored coefficients as exact rationals for this example."""
    weights = artifact.tensors["head.weight"][0]
    return sum((Fraction.from_float(float(w)) * x
                for w, x in zip(weights, feature)), Fraction(0))


def main() -> None:
    parent = TensorArtifact(
        "illustrative-linear-head/1", {"head.weight": [[1.0, 1.0]]},
        {"feature_id": "two-declared-features/1"},
    )
    head = ReadoutHead("head.weight", feature_id="two-declared-features/1")
    target, protected = (1, 1), (1, 0)
    policy = StructuralPolicy(
        writable=("head.weight",), max_edits=2,
        max_parameter_abs=2.0, max_change_abs=1.0,
    )
    cases = (
        ("no-op", [[1.0, 1.0]], Fraction(2), Fraction(1)),
        ("target-only exact edit", [[1.5, 1.5]], Fraction(3), Fraction(3, 2)),
        ("protected exact edit", [[1.0, 2.0]], Fraction(3), Fraction(1)),
    )
    records = []
    for label, weights, expected_target, expected_protected in cases:
        requested = parent.replace(**{"head.weight": weights})
        program = difference(parent, requested, label)
        admission = policy.check(parent, program)
        assert admission["structural_accepted"]
        assert admission["behavioral_status"] == "not_checked"
        candidate = apply_program(parent, program)
        receipt = certify_feature_retention(parent, candidate, head, [protected])
        assert candidate.sha256 == requested.sha256
        assert real_output(candidate, target) == expected_target
        assert real_output(candidate, protected) == expected_protected
        assert verify_feature_retention(parent, candidate, head, [protected], receipt)
        assert apply_program(candidate, rollback(parent, program)).sha256 == parent.sha256
        records.append({
            "case": label,
            "target_output": str(real_output(candidate, target)),
            "protected_output": str(real_output(candidate, protected)),
            "target_change_is_one": real_output(candidate, target) - 2 == 1,
            "protected_exactly": receipt["exactly_zero_on_supplied_features"],
            "protected_residual_exact": receipt["maximum_residuals_exact"][0],
        })

    compiled = compile_feature_edit(
        parent, head, [target], [[1.0]], [protected],
        retention="nullspace", ridge=0.01,
    )
    realized = apply_program(parent, compiled.program)
    compiled_receipt = compiled.retention_certificate
    assert compiled_receipt is not None
    assert verify_feature_retention(parent, realized, head, [protected], compiled_receipt)
    assert compiled_receipt["exactly_zero_on_supplied_features"]
    # Positive ridge shrinks the update. A verified retention receipt alone
    # would not tell us whether the requested target change was achieved.
    compiled_change = real_output(realized, target) - real_output(parent, target)
    assert 0 < compiled_change < 1
    records.append({
        "case": "numerical nullspace fit with ridge=0.01",
        "target_change_rounded": round(float(compiled_change), 6),
        "target_change_exact": str(compiled_change),
        "protected_exactly": compiled_receipt["exactly_zero_on_supplied_features"],
        "target_rmse_after_projection": compiled.diagnostics["target_rmse_after_projection"],
    })

    blocked = compile_feature_edit(
        parent, head, [target], [[1.0]], [[1.0, 0.0], [0.0, 1.0]],
        retention="nullspace", ridge=0.01,
    )
    blocked_candidate = apply_program(parent, blocked.program)
    assert blocked.diagnostics["no_op"]
    assert blocked_candidate.sha256 == parent.sha256
    records.append({
        "case": "protect both independent feature directions",
        "target_change": "0", "protected_exactly": True,
        "requested_target_change_met": False,
    })
    print(json.dumps({
        "scope": "Illustrative linear head; exact cases use explicit edits, not fitting",
        "parent_outputs": {"target": "2", "protected": "1"},
        "cases": records,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
