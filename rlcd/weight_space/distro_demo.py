"""Fresh synthetic support-routing workspace for Distro's review tools."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import StructuralPolicy, TensorArtifact
from .distro import DistroWorkspace
from .evidence import write_json
from .features import ReadoutHead, certify_feature_retention, verify_feature_retention
from .program import apply_program, difference


def create_demo(workspace: Path) -> dict[str, Any]:
    """Create new illustrative data, verify its supplied features, and stop."""
    root = Path(os.path.abspath(workspace))
    ancestor = root.parent
    while True:
        try:
            ancestor.lstat()
            break
        except FileNotFoundError:
            ancestor = ancestor.parent
    DistroWorkspace(ancestor)
    root.mkdir(parents=True, exist_ok=False)
    root = DistroWorkspace(root).root

    queues = ["billing", "security", "ordinary-support", "platform-operations"]
    base = TensorArtifact(
        "synthetic-support-readout/1",
        {"head.weight": [[2., 0., 0., 0.], [0., 2., 0., 0.],
                         [0., 0., 2., 0.], [0., 0., 0., 1.]]},
        {"feature_id": "declared-ticket-features/1", "queues": queues},
    )
    proposed = base.replace(**{"head.weight": [[2., 0., 0., 0.], [0., 2., 0., 0.],
                                               [0., 0., 2., 0.], [0., 0., 0., 3.]]})
    program = difference(base, proposed, "Route the declared telemetry-outage case to Platform Operations")
    candidate = apply_program(base, program)
    policy = StructuralPolicy(writable=("head.weight",), max_edits=1,
                              max_parameter_abs=3., max_change_abs=2.)
    admission = policy.check(base, program)
    if not admission["structural_accepted"]:
        raise ValueError("Synthetic example failed its structural policy")

    head = ReadoutHead("head.weight", feature_id="declared-ticket-features/1")
    protected = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]
    target = [0., 0., 1., 1.]
    receipt = certify_feature_retention(base, candidate, head, protected)
    verified = verify_feature_retention(base, candidate, head, protected, receipt)
    if not verified or not receipt["exactly_zero_on_supplied_features"]:
        raise ValueError("Synthetic example failed its supplied-feature retention check")

    before = (base.tensors["head.weight"] @ np.array(target)).tolist()
    after = (candidate.tensors["head.weight"] @ np.array(target)).tolist()
    before_queue, after_queue = queues[int(np.argmax(before))], queues[int(np.argmax(after))]
    if (before_queue, after_queue) != ("ordinary-support", "platform-operations"):
        raise ValueError("Synthetic target did not change to the requested queue")
    checks = {
        "schema": "wsp-distro-demo-checks/1", "fixture": "synthetic supplied-feature routing",
        "score_order": queues, "target_features": target,
        "before_scores": before, "after_scores": after,
        "before_queue": before_queue, "after_queue": after_queue,
        "protected_features": protected, "structural_admission": admission,
        "retention_receipt_verified": verified,
        "scope": "Declared fixed features only; no raw-text extractor, unseen tickets or native endpoint is checked",
        "production_authorized": False, "scientific_authorized": False,
    }
    descriptor = {
        "schema": "wsp-distro-change/1",
        **{name: f"change/{name}.json" for name in ("base", "program", "candidate", "constraints", "scope")},
        "purpose": "Review the telemetry outage routing change for a local pilot",
        "requester": "alice", "approvers": ["bob"],
        "evidence": [
            {"name": "review-note", "path": "evidence/review-note.md"},
            {"name": "supplied-feature-retention", "path": "evidence/retention.json"},
            {"name": "synthetic-routing-checks", "path": "evidence/checks.json"},
        ],
    }
    documents = {
        "change/base.json": base.to_dict(), "change/program.json": program.to_dict(),
        "change/candidate.json": candidate.to_dict(),
        "change/constraints.json": {"protected_queues": queues[:3], "protected_features": protected,
                                    "writable": ["head.weight"], "max_change_abs": 2.},
        "change/scope.json": {"environment": "local-pilot", "use_case": "synthetic-support-routing",
                              "target": "declared-ticket-readout"},
        "evidence/retention.json": receipt, "evidence/checks.json": checks,
        "change.json": descriptor,
    }
    for relative, value in documents.items():
        write_json(root / relative, value)
    with (root / "evidence/review-note.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "# Telemetry outage routing: synthetic local pilot\n\n"
            "The declared outage feature reaches Ordinary Support before the edit and Platform Operations after it. "
            "All scores on the supplied billing, security and ordinary-support feature directions remain unchanged.\n\n"
            "The generator checked exact application, structural policy and the supplied-feature retention receipt. "
            "These checks cover constructed fixed features. They do not establish raw-text recognition, "
            "unseen-ticket behavior, enterprise savings or a production-ready model.\n\n"
            "The operational opportunity is to avoid a team transfer for outage tickets. A pilot would still need "
            "a representative feature/extractor evaluation and separately authorized operational controls. "
            "Governance commits this evidence; its tools do not independently interpret or re-run the checks.\n"
        )
    return {"schema": "wsp-distro-demo/1", "workspace": str(root), "change": "change.json",
            "before_queue": before_queue, "after_queue": after_queue,
            "supplied_protected_scores_unchanged": True, "local_pilot_eligible": False,
            "production_authorized": False, "scientific_authorized": False}
