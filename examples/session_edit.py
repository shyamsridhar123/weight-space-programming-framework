"""Select an explicit edit using public criteria, then evaluate after sealing.

This constructed linear example demonstrates the cooperative session API. It
does not measure an editor's ability or supply a fresh empirical result.
"""
from __future__ import annotations

import json

import numpy as np

from rlcd.weight_space import (
    AccessPolicy, BudgetLimits, ProgrammerSession, SplitSpec, StructuralPolicy,
    TaskSpec, TensorArtifact, difference, verify_trace,
)


def predict(artifact: TensorArtifact, points: object) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ artifact.tensors["head.weight"].T


def main() -> None:
    parent = TensorArtifact(
        "illustrative-linear-head/1", {"head.weight": [[1.0, 1.0]]},
        {"feature_id": "two-declared-features/1"},
    )
    task = TaskSpec(
        "illustrative-protected-edit", "edit",
        "Add x2 to the output; preserve outputs where x2 is zero",
        parent.architecture, parent.sha256,
        constraints={
            "privileged_hint": "The desired output change is exactly x2",
            "feature_order": ["x1", "x2"],
            "public_target": [1.0, 1.0],
            "public_protected": [1.0, 0.0],
        },
    )

    def public_feedback(candidate, points):
        outputs = predict(candidate.artifact, points).reshape(-1)
        return {
            "target_error": abs(float(outputs[0]) - 3.0),
            "protected_drift": abs(float(outputs[1]) - 1.0),
        }

    session = ProgrammerSession(
        task, base=parent,
        access=AccessPolicy(metadata=True, weights=True, feedback=True),
        limits=BudgetLimits(
            tool_calls=8, weight_values=8, proposals=3,
            feedback_calls=3, feedback_examples=6,
            execution_calls=0, execution_examples=0,
            input_tokens=0, output_tokens=0,
        ),
        public_feedback=public_feedback,
        splits=(
            SplitSpec("validation", ("public-target", "public-protected")),
            SplitSpec("test", ("target-a", "target-b", "retain-a", "retain-b")),
        ),
    )
    policy = StructuralPolicy(
        writable=("head.weight",), max_edits=2,
        max_parameter_abs=2.0, max_change_abs=1.0,
    )
    session.metadata()
    assert session.weights("head.weight", [(0, 0), (0, 1)]) == [1.0, 1.0]
    candidates = []
    for label, weights in (
        ("no-op", [[1.0, 1.0]]),
        ("target-only exact", [[1.5, 1.5]]),
        ("protected exact", [[1.0, 2.0]]),
    ):
        proposal = difference(parent, parent.replace(**{"head.weight": weights}), label)
        admission = policy.check(parent, proposal)
        assert admission["structural_accepted"]
        assert admission["behavioral_status"] == "not_checked"
        candidate = session.propose_program(proposal)
        score = session.feedback(candidate, [[1.0, 1.0], [1.0, 0.0]])
        candidates.append((label, candidate, score))

    # This rule is fixed before looking at the private examples. The no-op
    # remains eligible; protection alone would not rank it below a useful edit.
    eligible = [row for row in candidates if row[2]["protected_drift"] == 0.0]
    selected_label, selected, _ = min(eligible, key=lambda row: row[2]["target_error"])
    session.seal(selected)

    def evaluate(candidate):
        target_points = [[-2.0, 1.0], [3.0, 1.0]]
        protected_points = [[2.0, 0.0], [-3.0, 0.0]]
        target_change = predict(candidate.artifact, target_points) - predict(parent, target_points)
        protected_change = predict(candidate.artifact, protected_points) - predict(parent, protected_points)
        return {
            "max_target_error": float(np.max(np.abs(target_change - 1.0))),
            "max_protected_drift": float(np.max(np.abs(protected_change))),
        }

    receipt = session.evaluate_heldout(
        evaluate, evaluator_id="illustrative-linear-spec/1", split_name="test",
    )
    record = receipt.to_dict()
    assert selected_label == "protected exact"
    assert record["status"] == "succeeded"
    assert record["result"] == {"max_target_error": 0.0, "max_protected_drift": 0.0}
    assert record["usage"]["weight_values"] == 8
    assert verify_trace(record["trace"], expected_head=record["trace_head_sha256"])
    print(json.dumps({
        "scope": "Constructed API example with an explicit formula and explicit edits",
        "public_candidates": [{"proposal": label, **score} for label, _, score in candidates],
        "selected": selected_label,
        "evaluation_status": record["status"],
        "heldout_result": record["result"],
        "interface_usage": record["usage"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
