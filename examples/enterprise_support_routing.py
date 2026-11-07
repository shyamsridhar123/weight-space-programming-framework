"""Synthetic enterprise policy rollout using the generic framework.

This is a supplied structured-field extractor and hand-constructed linear model,
not a trained text classifier or evidence of a live language model's ability.
Run from the repository root after installing the project with ``pip install -e .``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlcd.weight_space import (
    AccessPolicy, BudgetLimits, ProgrammerSession, ReadoutHead, SemanticsAdapter,
    SemanticsRegistry, SplitSpec, StructuralPolicy, StructuredProgram, TaskSpec, TensorArtifact,
    apply_program, certify_feature_retention, compile_feature_edit, difference, rollback,
    verify_feature_retention, verify_lowering, verify_trace,
)
from rlcd.weight_space.model import digest
from rlcd.weight_space.program import Program

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/data/enterprise_support_tickets.json"
ARCHITECTURE = "enterprise-support-linear-head/1"
FEATURE_ID = "structured-support-fields/1"
QUEUES = ("general", "billing", "security", "platform")
FEATURES = ("general_topic", "billing_topic", "security_topic", "platform_topic",
            "enterprise_tier", "urgency", "telemetry_outage")
HEAD = ReadoutHead("head.weight", "head.bias", FEATURE_ID)
SCENARIOS = ("policy-change", "conflict", "coverage-gap")


def featurize(tickets: list[dict[str, Any]]) -> np.ndarray:
    """Use only structured fields; expected_queue and role are evaluator labels."""
    rows = []
    for ticket in tickets:
        topic, product, symptom, tier = (ticket[name] for name in
                                         ("topic", "product", "symptom", "tier"))
        if (topic not in QUEUES or product not in ("core", "telemetry") or
                symptom not in ("help", "payment", "access", "outage", "latency",
                                "intermittent_errors") or tier not in ("standard", "enterprise")):
            raise ValueError("Unsupported support field value; send for caller-owned review")
        urgency = ticket["urgency"]
        if type(urgency) not in (int, float) or not np.isfinite(urgency) or not 0 <= urgency <= 1:
            raise ValueError("Urgency must be a finite number in [0, 1]")
        incident = topic == "general" and product == "telemetry" and symptom == "outage"
        rows.append([float(topic == queue) for queue in QUEUES] +
                    [float(tier == "enterprise"), float(urgency), float(incident)])
    if not rows:
        raise ValueError("A nonempty ticket batch is required")
    return np.array(rows, dtype=np.float64)


def model_metadata() -> dict[str, Any]:
    return {"feature_id": FEATURE_ID, "feature_names": list(FEATURES),
            "queue_names": list(QUEUES), "score_rule": "features @ head.weight.T + head.bias",
            "extractor_source_sha256": hashlib.sha256(inspect.getsource(featurize).encode()).hexdigest(),
            "origin": "Hand-constructed synthetic baseline; no training or customer data"}


def baseline() -> TensorArtifact:
    return TensorArtifact(ARCHITECTURE, {
        "head.weight": [[2, 0, 0, 0, .25, .25, 0],
                        [0, 3, 0, 0, .25, 0, 0],
                        [0, 0, 4, 0, .25, .5, 0],
                        [0, 0, 0, 3, .25, .25, .25]],
        "head.bias": [0, 0, 0, 0],
    }, model_metadata())


def scores(artifact: TensorArtifact, tickets: list[dict[str, Any]]) -> np.ndarray:
    if (artifact.architecture != ARCHITECTURE or artifact.metadata != model_metadata() or
            set(artifact.tensors) != {"head.weight", "head.bias"} or
            artifact.tensors["head.weight"].shape != (4, 7) or
            artifact.tensors["head.bias"].shape != (4,)):
        raise ValueError("Routing adapter requires the exact declared feature and queue schema")
    return featurize(tickets) @ artifact.tensors["head.weight"].T + artifact.tensors["head.bias"]


REGISTRY = SemanticsRegistry()
REGISTRY.register(SemanticsAdapter(ARCHITECTURE, scores,
    certificate_scope="Use the separately bound frozen-feature retention receipt; no native proof"))


def contract(scenario: str) -> dict[str, Any]:
    return {"schema": "enterprise-routing-contract/1", "scenario": scenario,
            "objective": "Route telemetry outage tickets to Platform Operations",
            "desired_score_shift": [0, 0, 0, 3], "maximum_target_shift_error": .05,
            "minimum_route_margin": .5, "maximum_protected_runtime_drift": 1e-12,
            "require_exact_anchor_retention": True,
            "selection_rule": "Among publicly eligible edits, minimize changed cells then program SHA-256; otherwise abstain",
            "final_gate": "The sealed edit must also pass the same behavioral rubric on the declared final split",
            "execution": "NumPy float64 matmul and addition; sampled observations, no floating-point proof",
            "production_authorized": False}


def routes(values: np.ndarray, minimum_margin: float) -> tuple[list[str], np.ndarray]:
    ordered = np.sort(values, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    labels = [QUEUES[int(index)] if margin >= minimum_margin else "manual_review"
              for index, margin in zip(np.argmax(values, axis=1), margins)]
    return labels, margins


def measure(parent: TensorArtifact, candidate: TensorArtifact,
            tickets: list[dict[str, Any]], rule: dict[str, Any]) -> dict[str, Any]:
    before = REGISTRY.evaluate(parent, tickets)
    after = REGISTRY.evaluate(candidate, tickets)
    old_routes, _ = routes(before, rule["minimum_route_margin"])
    new_routes, margins = routes(after, rule["minimum_route_margin"])
    target = np.array([ticket["role"] == "target" for ticket in tickets])
    protected = np.array([ticket["role"] == "protected" for ticket in tickets])
    if not target.any() or not protected.any() or not (target | protected).all():
        raise ValueError("Evaluation requires explicit target and protected cohorts")
    if any(ticket["expected_queue"] not in QUEUES for ticket in tickets):
        raise ValueError("Evaluator labels must name a declared queue")
    changes = after - before
    details = [{"ticket_id": ticket["ticket_id"], "role": ticket["role"],
                "expected_queue": ticket["expected_queue"], "before_queue": old,
                "after_queue": new, "after_margin": float(margin),
                "before_scores": old_score.tolist(), "after_scores": new_score.tolist()}
               for ticket, old, new, margin, old_score, new_score in
               zip(tickets, old_routes, new_routes, margins, before, after)]
    return {"maximum_target_shift_error": float(np.max(np.abs(
                changes[target] - np.array(rule["desired_score_shift"])))),
            "maximum_protected_score_drift": float(np.max(np.abs(changes[protected]))),
            "target_routes_correct": all(row["after_queue"] == row["expected_queue"]
                                         for row in details if row["role"] == "target"),
            "protected_routes_unchanged": all(row["before_queue"] == row["after_queue"] == row["expected_queue"]
                                              for row in details if row["role"] == "protected"),
            "minimum_target_margin": float(np.min(margins[target])), "tickets": details}


def behavior_failures(metrics: dict[str, Any], rule: dict[str, Any]) -> list[str]:
    return [name for name, passed in {
        "target_score_shift": metrics["maximum_target_shift_error"] <= rule["maximum_target_shift_error"],
        "target_routing": metrics["target_routes_correct"] and
                          metrics["minimum_target_margin"] >= rule["minimum_route_margin"],
        "protected_runtime_scores": metrics["maximum_protected_score_drift"] <= rule["maximum_protected_runtime_drift"],
        "protected_routes": metrics["protected_routes_unchanged"],
    }.items() if not passed]


def admit(parent: TensorArtifact, program: Program, anchors: np.ndarray,
          certificate: dict[str, Any], metrics: dict[str, Any], rule: dict[str, Any],
          policy: StructuralPolicy) -> dict[str, Any]:
    candidate = apply_program(parent, program)
    structural = policy.check(parent, program)
    bound = verify_feature_retention(parent, candidate, HEAD, anchors, certificate)
    failures = behavior_failures(metrics, rule)
    if not structural["structural_accepted"]:
        failures.append("structural_policy")
    if not bound:
        failures.append("certificate_binding")
    if not certificate.get("exactly_zero_on_supplied_features", False):
        failures.append("protected_exact_retention")
    payload = {"schema": "enterprise-routing-public-admission/1",
               "parent_sha256": parent.sha256, "artifact_sha256": candidate.sha256,
               "program_sha256": program.sha256, "contract_sha256": digest(rule),
               "certificate_sha256": certificate["sha256"], "certificate_verified": bound,
               "structural": structural, "metrics": metrics,
               "eligible": not failures, "failures": failures}
    return payload | {"sha256": digest(payload)}


def scenario_inputs(scenario: str) -> dict[str, Any]:
    inputs = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if scenario == "coverage-gap":
        for ticket in inputs["heldout"]:
            if ticket["role"] == "target":
                ticket["symptom"] = "intermittent_errors"
    return inputs


def run_scenario(scenario: str) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    inputs, parent, rule = scenario_inputs(scenario), baseline(), contract(scenario)
    public = inputs["public"]
    target = featurize([ticket for ticket in public if ticket["role"] == "target"])
    anchors = featurize([ticket for ticket in public if ticket["role"] == "protected"])
    if scenario == "conflict":
        # The same target representation is now explicitly required to retain its scores.
        anchors = np.vstack((anchors, target[:1]))
    desired = np.tile(rule["desired_score_shift"], (len(target), 1))
    policy = StructuralPolicy((HEAD.weight,), 7, 4, 3)
    task = TaskSpec("enterprise-routing-" + scenario, "edit", rule["objective"],
                    parent.architecture, parent.sha256,
                    {"contract": rule, "feature_schema": model_metadata(),
                     "fitting": "Numerical ridge fitting on supplied public targets and anchors; outside session charges"})
    session = ProgrammerSession(task, base=parent,
        access=AccessPolicy(metadata=True, weights=True, feedback=True),
        limits=BudgetLimits(tool_calls=11, weight_values=5 * parent.parameter_count,
            proposals=4, feedback_calls=4, feedback_examples=4 * len(public),
            execution_calls=0, execution_examples=0, input_tokens=0, output_tokens=0),
        public_feedback=lambda candidate, tickets: measure(parent, candidate.artifact, tickets, rule),
        splits=tuple(SplitSpec(name, [row["ticket_id"] for row in inputs[key]],
                              [row["case_id"] for row in inputs[key]])
                     for name, key in (("validation", "public"), ("test", "heldout"))))
    session.metadata()
    session.weights(HEAD.weight)
    session.weights(HEAD.bias)
    # Compiler work is explicitly allowed and not claimed to be session-budgeted.
    compiled = {mode: compile_feature_edit(parent, HEAD, target, desired, anchors,
                retention=mode, ridge=.01, retention_weight=8, intent="Enterprise routing: " + mode)
                for mode in ("none", "soft", "nullspace")}
    no_op = difference(parent, parent, "Enterprise routing: keep parent")
    proposals = [("keep-parent", no_op, None,
                  certify_feature_retention(parent, parent, HEAD, anchors))]
    proposals += [(label, compiled[mode].program, compiled[mode], compiled[mode].retention_certificate)
                  for label, mode in (("target-only", "none"), ("soft-retention", "soft"),
                                      ("protected-edit", "nullspace"))]
    candidates, records = {}, []
    for label, program, compilation, certificate in proposals:
        candidate = session.propose_program(program)
        candidates[label] = candidate
        metrics = session.feedback(candidate, public)
        admission = admit(parent, program, anchors, certificate, metrics, rule, policy)
        records.append({"label": label, "candidate": candidate.to_dict(),
                        "program": program.to_dict(), "retention_certificate": certificate,
                        "compilation": None if compilation is None else {
                            "proposal": compilation.proposal.to_dict(), "diagnostics": compilation.diagnostics},
                        "admission": admission})
    eligible = [record for record in records if record["admission"]["eligible"]]
    selected = min(eligible, key=lambda record: (len(record["program"]["edits"]),
                                               record["admission"]["program_sha256"])) if eligible else None
    session.seal(None if selected is None else candidates[selected["label"]])
    # No final outputs are inspected until selection is sealed; this file is public synthetic data.
    receipt = session.evaluate_heldout(
        lambda candidate: measure(parent, candidate.artifact, inputs["heldout"], rule),
        evaluator_id="enterprise-routing-rubric/1", split_name="test").to_dict()
    final_failures = (["no_publicly_eligible_edit"] if selected is None else
                      ["heldout_evaluator_failed"] if receipt["status"] != "succeeded" else
                      behavior_failures(receipt["result"], rule))
    final = {"schema": "enterprise-routing-final-admission/1",
             "decision": "abstained" if selected is None else
                         "rejected_after_holdout" if final_failures else "admitted_for_synthetic_trial",
             "parent_sha256": parent.sha256, "task_sha256": task.sha256,
             "contract_sha256": digest(rule), "inputs_sha256": digest(inputs),
             "selected_label": None if selected is None else selected["label"],
             "candidate_sha256": receipt["candidate_sha256"],
             "artifact_sha256": None if selected is None else selected["admission"]["artifact_sha256"],
             "public_admission_sha256": None if selected is None else selected["admission"]["sha256"],
             "evaluation_receipt_sha256": receipt["sha256"], "trace_head_sha256": receipt["trace_head_sha256"],
             "heldout_failures": final_failures, "production_authorized": False,
             "scope": "Constructed synthetic integration example, not independent enterprise deployment evidence"}
    final["sha256"] = digest(final)
    undo = None
    if selected is not None:
        program = Program.from_dict(selected["program"])
        undo_program = rollback(parent, program)
        restored = apply_program(candidates[selected["label"]].artifact, undo_program)
        if restored.sha256 != parent.sha256:
            raise RuntimeError("Rollback failed to restore the exact parent")
        undo = {"program": undo_program.to_dict(), "restored_parent_sha256": restored.sha256,
                "scope": "Restores model tensors and metadata; does not undo already routed tickets"}
    return {"schema": "enterprise-routing-bundle/1", "scenario": scenario,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "contract": rule, "inputs": inputs, "task": task.to_dict(), "parent": parent.to_dict(),
            "anchors": anchors.tolist(), "candidates": records, "session": receipt,
            "final_admission": final, "rollback": undo}


def report_markdown(bundle: dict[str, Any]) -> str:
    final, rule = bundle["final_admission"], bundle["contract"]
    lines = ["# Enterprise support routing: " + bundle["scenario"], "",
             "**Decision: " + final["decision"] + ".** Production deployment is not authorized by this demo.", "",
             "New policy: telemetry outage tickets should reach Platform Operations; existing routing scores and queues are protected.", "",
             "The baseline and business records are synthetic. Scores are raw routing scores, not probabilities. A winner margin below 0.5 sends a ticket to manual review.", "",
             "## Public candidate review", "",
             "| Candidate | Target shift error (score units) | Protected score drift | Target routes correct | Protected routes unchanged | Exact anchor retention | Eligible |",
             "| --- | ---: | ---: | --- | --- | --- | --- |"]
    for record in bundle["candidates"]:
        a = record["admission"]
        m = a["metrics"]
        lines.append(f"| {record['label']} | {m['maximum_target_shift_error']:.6f} | {m['maximum_protected_score_drift']:.6f} | {m['target_routes_correct']} | {m['protected_routes_unchanged']} | {record['retention_certificate']['exactly_zero_on_supplied_features']} | {a['eligible']} |")
    lines += ["", "Acceptance requires target shift error <= 0.05, correct target queues with margin >= 0.5, exact zero residual on protection anchors, protected sampled score drift <= 1e-12, and unchanged correct protected queues. Structural admission and certificate binding must also pass.", ""]
    for record in bundle["candidates"]:
        if record["admission"]["failures"]:
            lines.append("- **" + record["label"] + ":** " + ", ".join(record["admission"]["failures"]) + ".")
    result = bundle["session"]["result"]
    lines += ["", "## Sealed final evaluation", "", "Selected: **" + str(final["selected_label"]) + "**.", ""]
    if result is None:
        lines += ["No candidate met the public contract. The session abstained and did not call the final evaluator.", ""]
    else:
        lines += [f"Target shift error: **{result['maximum_target_shift_error']:.6f} score units**. Protected score drift: **{result['maximum_protected_score_drift']:.6f}**.", "",
                  "| Ticket | Cohort | Expected queue | Parent queue | Sealed queue | Sealed winner margin |",
                  "| --- | --- | --- | --- | --- | ---: |"]
        for ticket in result["tickets"]:
            lines.append(f"| {ticket['ticket_id']} | {ticket['role']} | {ticket['expected_queue']} | {ticket['before_queue']} | {ticket['after_queue']} | {ticket['after_margin']:.6f} |")
        lines += ["", "Final gate failures: " + (", ".join(final["heldout_failures"]) or "none") + ".", ""]
    if bundle["scenario"] == "conflict":
        lines += ["The protected features now span every editable feature direction. Exact preservation forces a zero update; the requested score shift cannot be admitted.", ""]
    elif bundle["scenario"] == "coverage-gap":
        lines += ["Final outage tickets use `intermittent_errors`. The supplied extractor recognizes only `outage`, so their incident feature is zero. Public acceptance did not establish symptom coverage; final evaluation blocks this candidate without another selection round.", ""]
    lines += ["## Evidence and operational boundary", "",
              "The bundle binds the parent, contract, task, literal edits, compiler lowering, certificates, candidate review, sealed evaluation, and rollback. Budget usage describes session interfaces; external fitting and final evaluation are outside proposer charges. No external provider ran.", "",
              "Exact receipts concern the real linear head on supplied features and their span under the stated premises. This run checks NumPy float64 scores on the recorded rows; it does not prove a production runtime or text extractor. Declared split identifiers are disjoint, but constructed public fixtures are not independent confirmation.", "",
              "Rollback restores the exact model artifact. It does not reverse ticket assignments, external side effects, or other model revisions. A real release needs approved data, an independently owned untouched evaluation, feature and runtime correspondence, and an operational rollout decision.", "",
              "Contract SHA-256: `" + digest(rule) + "`. Final admission SHA-256: `" + final["sha256"] + "`.", ""]
    return "\n".join(lines)


def write_bundle(bundle: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    files = {"bundle.json": json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n",
             "report.md": report_markdown(bundle)}
    selected = next((row for row in bundle["candidates"]
                     if row["label"] == bundle["final_admission"]["selected_label"]), None)
    if selected is not None:
        for name, data in (("candidate.json", selected["candidate"]["artifact"]),
                           ("edit.json", selected["program"]),
                           ("rollback.json", bundle["rollback"]["program"])):
            files[name] = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    checksums = {}
    for name, content in files.items():
        payload = content.encode("utf-8")
        (output / name).write_bytes(payload)
        checksums[name] = hashlib.sha256(payload).hexdigest()
    (output / "checksums.json").write_bytes(
        (json.dumps(checksums, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def verify_bundle(output: Path) -> dict[str, Any]:
    """Recompute recorded endpoints and rubrics; no numerical fitting rerun needed.

    Checksums detect byte changes, not a malicious re-authoring of the whole bundle.
    This verifier trusts the local adapter and published synthetic specification.
    """
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    # Fixed names avoid trusting paths supplied by an edited checksums file.
    bundle = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    expected = {"bundle.json", "report.md"}
    if bundle["final_admission"]["selected_label"] is not None:
        expected.update(("candidate.json", "edit.json", "rollback.json"))
    if set(checksums) != expected:
        raise ValueError("Unexpected bundle file inventory")
    for name, expected_hash in checksums.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Bundle checksum mismatch: " + name)
    scenario = bundle["scenario"]
    if (bundle["schema"] != "enterprise-routing-bundle/1" or scenario not in SCENARIOS or
            bundle["contract"] != contract(scenario) or bundle["inputs"] != scenario_inputs(scenario) or
            bundle["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or
            bundle["fixture_sha256"] != hashlib.sha256(FIXTURE.read_bytes()).hexdigest()):
        raise ValueError("Bundle specification, source, or fixture binding changed")
    parent, rule = TensorArtifact.from_dict(bundle["parent"]), bundle["contract"]
    if parent.sha256 != baseline().sha256:
        raise ValueError("Baseline binding changed")
    anchors = featurize([row for row in bundle["inputs"]["public"] if row["role"] == "protected"])
    if scenario == "conflict":
        anchors = np.vstack((anchors, featurize([row for row in bundle["inputs"]["public"] if row["role"] == "target"])[:1]))
    if not np.array_equal(anchors, bundle["anchors"]):
        raise ValueError("Protection domain changed")
    policy = StructuralPolicy((HEAD.weight,), 7, 4, 3)
    records = bundle["candidates"]
    if [row["label"] for row in records] != ["keep-parent", "target-only", "soft-retention", "protected-edit"]:
        raise ValueError("Candidate history changed")
    session = bundle["session"]
    for index, record in enumerate(records, start=1):
        program = Program.from_dict(record["program"])
        candidate = apply_program(parent, program)
        if candidate.to_dict() != record["candidate"]["artifact"]:
            raise ValueError("Candidate does not realize its literal program")
        identity = record["candidate"]
        if (identity["sha256"] != digest({key: value for key, value in identity.items() if key != "sha256"}) or
                identity["artifact_sha256"] != candidate.sha256 or identity["program_sha256"] != program.sha256 or
                identity["parent_sha256"] != parent.sha256 or identity["task_sha256"] != digest(bundle["task"]) or
                identity["session_sha256"] != session["session_sha256"] or identity["proposal_index"] != index or
                identity["no_op"] != (candidate.sha256 == parent.sha256) or identity["kind"] != "program"):
            raise ValueError("Candidate identity or lineage changed")
        compilation = record["compilation"]
        if compilation is not None:
            proposal = StructuredProgram.from_dict(compilation["proposal"])
            lowering = compilation["diagnostics"]["lowering"]
            if not verify_lowering(parent, proposal, lowering) or lowering["program_sha256"] != program.sha256:
                raise ValueError("Compiler lowering does not bind the realized program")
        metrics = measure(parent, candidate, bundle["inputs"]["public"], rule)
        actual = admit(parent, program, anchors, record["retention_certificate"], metrics, rule, policy)
        if actual != record["admission"]:
            raise ValueError("Public admission does not recompute")
    if not verify_trace(session["trace"], expected_head=session["trace_head_sha256"]):
        raise ValueError("Session trace failed verification")
    if session["sha256"] != digest({key: value for key, value in session.items() if key != "sha256"}):
        raise ValueError("Evaluation receipt identity changed")
    if session["candidates"] != [row["candidate"] for row in records]:
        raise ValueError("Session candidate history differs from the reviewed history")
    if session["task_sha256"] != digest(bundle["task"]) or session["configuration"]["task_sha256"] != digest(bundle["task"]):
        raise ValueError("Task binding changed")
    if session["session_sha256"] != digest(session["configuration"]):
        raise ValueError("Session configuration identity changed")
    eligible = [row for row in records if row["admission"]["eligible"]]
    selected = min(eligible, key=lambda row: (len(row["program"]["edits"]), row["admission"]["program_sha256"])) if eligible else None
    final = bundle["final_admission"]
    label = None if selected is None else selected["label"]
    if final["selected_label"] != label or session["candidate"] != (None if selected is None else selected["candidate"]):
        raise ValueError("Sealed selection violates the declared rule")
    sealed_hash = None if selected is None else selected["candidate"]["sha256"]
    if (session["candidate_sha256"] != sealed_hash or session["trace"][-2]["action"] != "seal" or
            session["trace"][-2]["request"]["candidate_sha256"] != sealed_hash or
            session["trace"][-1]["action"] != "heldout_evaluation" or
            session["trace"][-1]["request"]["candidate_sha256"] != sealed_hash or
            session["trace"][-1]["result"] != session["result"]):
        raise ValueError("Trace does not bind seal and final evaluation to the same candidate")
    if selected is None:
        failures, decision = ["no_publicly_eligible_edit"], "abstained"
        if session["status"] != "abstained" or session["result"] is not None or bundle["rollback"] is not None:
            raise ValueError("Abstention evidence changed")
    else:
        candidate = apply_program(parent, Program.from_dict(selected["program"]))
        actual = measure(parent, candidate, bundle["inputs"]["heldout"], rule)
        if session["status"] != "succeeded" or actual != session["result"]:
            raise ValueError("Final evaluation does not recompute")
        failures = behavior_failures(actual, rule)
        decision = "rejected_after_holdout" if failures else "admitted_for_synthetic_trial"
        undo = Program.from_dict(bundle["rollback"]["program"])
        if apply_program(candidate, undo).sha256 != parent.sha256 or undo != rollback(parent, Program.from_dict(selected["program"])):
            raise ValueError("Rollback does not restore the exact parent")
        for name, data in (("candidate.json", candidate.to_dict()), ("edit.json", selected["program"]),
                           ("rollback.json", undo.to_dict())):
            if json.loads((output / name).read_text(encoding="utf-8")) != data:
                raise ValueError("Exported artifact differs from the bundle: " + name)
    expected_final = copy.deepcopy(final)
    expected_final.update(decision=decision, selected_label=label, heldout_failures=failures,
        parent_sha256=parent.sha256, task_sha256=digest(bundle["task"]), contract_sha256=digest(rule),
        inputs_sha256=digest(bundle["inputs"]), candidate_sha256=session["candidate_sha256"],
        artifact_sha256=None if selected is None else selected["admission"]["artifact_sha256"],
        public_admission_sha256=None if selected is None else selected["admission"]["sha256"],
        evaluation_receipt_sha256=session["sha256"], trace_head_sha256=session["trace_head_sha256"],
        production_authorized=False)
    expected_final["sha256"] = digest({key: value for key, value in expected_final.items() if key != "sha256"})
    if final != expected_final or (output / "report.md").read_text(encoding="utf-8") != report_markdown(bundle):
        raise ValueError("Final admission or report does not recompute")
    return {"verified": True, "decision": decision, "scope": "Local evidence consistency; no independent attestation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / ".runs/enterprise-support-routing")
    parser.add_argument("--verify", type=Path, help="Verify one recorded scenario directory without refitting")
    args = parser.parse_args()
    if args.verify is not None:
        print(json.dumps(verify_bundle(args.verify), indent=2))
        return
    for scenario in SCENARIOS if args.scenario == "all" else (args.scenario,):
        output = args.output / scenario
        bundle = run_scenario(scenario)
        write_bundle(bundle, output)
        verified = verify_bundle(output)
        print(json.dumps({"scenario": scenario, "selected": bundle["final_admission"]["selected_label"],
                          **verified}, indent=2))


if __name__ == "__main__":
    main()
