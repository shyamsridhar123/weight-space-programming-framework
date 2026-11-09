"""Cumulative sampled contracts and frozen trained edit sequences.

The reference model, earlier target insertions and execution scope are immutable
request inputs. Real residuals concern only the supplied learned features.
Native decisions and heldout utility have separate evidence; neither is a
certificate for the encoder, deployment state, or arbitrary future requests.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np

from .artifact import TensorArtifact
from .evidence import ROOT, file_hash, numerical_environment, read_json, utc_now, verify_manifest, write_json
from .features import ReadoutHead, certify_feature_retention
from .model import canonical_bytes, digest
from .program import Program, apply_program, compose, difference, rollback
from .research_tasks import compile_arm, evaluate_head, head_spec, load_lineage


SOURCE_FILES = tuple("rlcd/weight_space/" + name + ".py" for name in (
    "__init__", "cumulative_research", "research_tasks", "artifact", "model", "program", "features",
    "structured", "semantics", "sequential", "contracts", "workflow", "compiler", "symmetry",
    "broker", "live_provider", "evidence")) + ("rlcd/__init__.py", "scripts/run_trained_research.py")
INPUT_FILES = ("head.json", "encoder.npz", "preprocessing.json", "lineage.json", "features.npz")
DOMAINS = {"banking77": ("trained-support", tuple(range(7001, 7007))),
           "optical-digits": ("trained-nontext", tuple(range(8001, 8007)))}
REQUEST_FIELDS = {"schema", "original_sha256", "parent_sha256", "candidate_sha256", "program",
    "head", "queue_map", "baseline_features", "baseline_labels", "target", "retained", "policy",
    "mode", "step", "history_sha256", "sequence_contract_sha256", "proposal_error", "execution"}


def _is_hash(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def array_id(value: Any) -> str:
    array = np.asarray(value)
    return digest({"dtype": str(array.dtype), "shape": list(array.shape), "values": array.tolist()})


def _add_array(pool: dict[str, np.ndarray], value: Any) -> str:
    array = np.asarray(value).copy()
    key = array_id(array)
    pool[key] = array
    return key


@dataclass(frozen=True)
class CumulativeRequest:
    """Canonical bytes, rather than mutable dictionaries, are the trust anchor."""
    _bytes: bytes

    @classmethod
    def capture(cls, value: dict[str, Any]) -> "CumulativeRequest":
        if (not isinstance(value, dict) or set(value) != REQUEST_FIELDS
                or value["schema"] != "wsp-cumulative-request/1"
                or value["mode"] not in ("local", "cumulative")
                or type(value["step"]) is not int or value["step"] < 1
                or any(not _is_hash(value[key]) for key in
                       ("original_sha256", "parent_sha256", "candidate_sha256", "baseline_features",
                        "baseline_labels", "history_sha256", "sequence_contract_sha256"))
                or value["proposal_error"] is not None and
                    (not isinstance(value["proposal_error"], str) or not value["proposal_error"].strip())):
            raise ValueError("Invalid cumulative request schema, identity, mode or step")
        return cls(canonical_bytes(value))

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._bytes)


def _feature_matrix(features: Any, width: int) -> np.ndarray:
    values = np.asarray(features)
    if (values.ndim != 2 or not len(values) or values.shape[1] != width
            or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
        raise ValueError("Features must be nonempty, finite and aligned to the frozen head")
    return values


def _head_only(base: TensorArtifact, candidate: TensorArtifact, head: ReadoutHead) -> None:
    if not base.compatible(candidate):
        raise ValueError("Cumulative endpoints must share a semantic interface")
    head.dimensions(base)
    head.dimensions(candidate)
    before, after = base.tensors, candidate.tensors
    if any(not np.array_equal(before[name], after[name]) for name in before
           if name not in (head.weight, head.bias)):
        raise ValueError("A frozen-feature contract rejects mutation outside its declared readout head")


def _execution() -> dict[str, Any]:
    return {"backend": "torch", "device": "cpu", "dtype": "float32",
        "operator": "torch.nn.functional.linear", "layout": "C-order [outputs,features]",
        "bias": "native linear argument; fusion and reduction order are backend-controlled",
        "endpoint_cast": "round", "feature_cast": "round",
        "decision": "lowest-index argmax then frozen queue map",
        "environment": numerical_environment("torch")}


def _head_scores(artifact: TensorArtifact, head: ReadoutHead, features: np.ndarray) -> np.ndarray:
    import torch
    width, _ = head.dimensions(artifact)
    values = _feature_matrix(features, width)
    if artifact.metadata.get("feature_id", head.feature_id) != head.feature_id:
        raise ValueError("Features and their declared frozen head must correspond")
    weight = artifact.tensors[head.weight]
    bias = None if head.bias is None else artifact.tensors[head.bias]
    # Rounding is an explicit part of this native scope, not real arithmetic.
    with torch.no_grad():
        result = torch.nn.functional.linear(torch.tensor(values, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            None if bias is None else torch.tensor(bias, dtype=torch.float32)).numpy()
    if not np.isfinite(result).all():
        raise ValueError("Native affine execution produced nonfinite scores")
    return result


def exact_sample_drift(base: TensorArtifact, candidate: TensorArtifact, head: ReadoutHead,
                       features: np.ndarray) -> dict[str, Any]:
    """Reuse the exact engine on changed output rows; equal rows have zero drift."""
    _head_only(base, candidate, head)
    width, outputs = head.dimensions(base)
    features = _feature_matrix(features, width)
    before, after = base.tensors, candidate.tensors
    changed = np.any(before[head.weight] != after[head.weight], axis=1)
    if head.bias is not None:
        changed |= before[head.bias] != after[head.bias]
    selected = np.flatnonzero(changed).tolist()
    maxima = ["0"]*outputs
    subreceipt = None
    if selected:
        metadata = {"selected_outputs": selected, "feature_id": head.feature_id}
        left = {head.weight: before[head.weight][selected]}
        right = {head.weight: after[head.weight][selected]}
        if head.bias is not None:
            left[head.bias], right[head.bias] = before[head.bias][selected], after[head.bias][selected]
        a = TensorArtifact("selected-affine-outputs/1", left, metadata)
        b = TensorArtifact("selected-affine-outputs/1", right, metadata)
        subreceipt = certify_feature_retention(a, b, head, features)
        for output, maximum in zip(selected, subreceipt["maximum_residuals_exact"]):
            maxima[output] = maximum
    payload = {"schema": "wsp-cumulative-sample-drift/1", "base_sha256": base.sha256,
        "candidate_sha256": candidate.sha256, "head": head.to_dict(),
        "features_sha256": array_id(features), "changed_outputs": selected,
        "maximum_residuals_exact": maxima, "maximum_exact": str(max(map(Fraction, maxima))),
        "exactly_zero": all(Fraction(value) == 0 for value in maxima),
        "changed_output_certificate": subreceipt,
        "scope": "Exact real affine residuals on these supplied features only; equal stored rows contribute exactly zero",
        "feature_extractor_verified": False, "native_execution_verified": False,
        "implementation_machine_checked": False}
    return payload | {"sha256": digest(payload)}


def bind_step(*, original: TensorArtifact, parent: TensorArtifact, candidate: TensorArtifact,
              program: Program, head: ReadoutHead, queue_map: list[int], baseline_features: str,
              baseline_labels: str, target: dict[str, Any], retained: list[dict[str, Any]],
              policy: dict[str, Any], mode: str, step: int, history_sha256: str,
              sequence_contract_sha256: str, proposal_error: str | None = None) -> CumulativeRequest:
    return CumulativeRequest.capture({"schema": "wsp-cumulative-request/1",
        "original_sha256": original.sha256, "parent_sha256": parent.sha256,
        "candidate_sha256": candidate.sha256, "program": program.to_dict(), "head": head.to_dict(),
        "queue_map": queue_map, "baseline_features": baseline_features, "baseline_labels": baseline_labels,
        "target": target, "retained": retained, "policy": policy, "mode": mode, "step": step,
        "history_sha256": history_sha256, "sequence_contract_sha256": sequence_contract_sha256,
        "proposal_error": proposal_error,
        "execution": _execution()})


def admit_step(request: CumulativeRequest, artifacts: dict[str, TensorArtifact],
               arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    value = CumulativeRequest.capture(request.to_dict()).to_dict()
    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, evidence: Any, required: bool = True,
             failure: str = "observed_violation") -> None:
        gates.append({"name": name, "required": required, "status": "pass" if passed else failure,
                      "request_sha256": request.sha256, "evidence": evidence})

    def artifact(key: str) -> TensorArtifact:
        item = artifacts[key]
        if not isinstance(item, TensorArtifact) or item.sha256 != key:
            raise ValueError("An artifact does not match its immutable identity")
        return item

    def array(key: str) -> np.ndarray:
        item = arrays[key]
        if not isinstance(item, np.ndarray) or array_id(item) != key:
            raise ValueError("A feature/label bundle does not match its immutable identity")
        return item

    try:
        head = ReadoutHead(**value["head"])
        original, parent, candidate = (artifact(value[key]) for key in
                                      ("original_sha256", "parent_sha256", "candidate_sha256"))
        _head_only(original, parent, head)
        _head_only(parent, candidate, head)
        if apply_program(parent, Program.from_dict(value["program"])).sha256 != candidate.sha256:
            raise ValueError("Program does not produce its captured candidate")
        policy = value["policy"]
        expected_policy = {"score_budget", "maximum_original_decision_flip", "maximum_local_decision_flip",
            "maximum_original_net_harm", "maximum_local_net_harm", "maximum_old_target_loss",
            "maximum_old_target_decision_flip", "minimum_new_target_gain", "conflict", "abstention"}
        if (set(policy) != expected_policy or policy["conflict"] != "reject"
                or policy["abstention"] != "keep_last_accepted"
                or any(type(policy[key]) not in (float, int) or not np.isfinite(policy[key]) or policy[key] < 0
                       for key in expected_policy - {"conflict", "abstention"})
                or any(policy[key] > 1 for key in expected_policy - {"conflict", "abstention", "score_budget"})):
            raise ValueError("Unknown or invalid cumulative budget/policy")
        execution = value["execution"]
        if execution != _execution():
            raise ValueError("Native execution differs from the captured scope")
        queue = np.asarray(value["queue_map"])
        _, outputs = head.dimensions(original)
        if (not isinstance(value["queue_map"], list) or queue.shape != (outputs,)
                or any(type(item) is not int or item < 0 for item in value["queue_map"])):
            raise ValueError("Queue mapping and step index must be explicit valid integers")
        baseline, labels = array(value["baseline_features"]), array(value["baseline_labels"])
        if (labels.shape != (len(baseline),) or labels.dtype.kind not in "iu"
                or (labels < 0).any() or (labels >= outputs).any()):
            raise ValueError("Baseline labels do not match the protected sample")
        target = value["target"]
        expected_target = {"subject", "case_id", "target_label", "desired_code", "desired_queue", "features"}

        def validate_goal(goal: dict[str, Any], expected: set[str]) -> None:
            if (not isinstance(goal, dict) or set(goal) != expected
                    or any(not isinstance(goal[key], str) or not goal[key].strip() for key in ("subject", "case_id"))
                    or any(type(goal[key]) is not int or not 0 <= goal[key] < outputs
                           for key in ("target_label", "desired_code"))
                    or type(goal["desired_queue"]) is not int
                    or int(queue[goal["desired_code"]]) != goal["desired_queue"]
                    or not _is_hash(goal["features"])):
                raise ValueError("A target must identify one explicit valid goal and supplied feature sample")

        validate_goal(target, expected_target)
        if not isinstance(value["retained"], list):
            raise ValueError("Retained insertions must be an ordered list")
        seen_requests, previous_step = set(), 0
        for memory in value["retained"]:
            validate_goal(memory, expected_target | {"inserted_endpoint_sha256", "accepted_step", "request_sha256"})
            if (not _is_hash(memory["inserted_endpoint_sha256"]) or not _is_hash(memory["request_sha256"])
                    or memory["request_sha256"] in seen_requests or type(memory["accepted_step"]) is not int
                    or not previous_step < memory["accepted_step"] < value["step"]):
                raise ValueError("Retained targets must be distinct prior accepted requests in step order")
            previous_step = memory["accepted_step"]
            seen_requests.add(memory["request_sha256"])
        target_features = array(target["features"])
        cumulative = value["mode"] == "cumulative"
        gate("correspondence", True, {"original": original.sha256, "parent": parent.sha256,
             "candidate": candidate.sha256, "program": digest(value["program"]), "execution": execution})
        gate("proposal", value["proposal_error"] is None, value["proposal_error"], failure="invalid")
        conflicts = [old["subject"] for old in value["retained"]
                     if old["subject"] == target["subject"] and
                     (old["desired_queue"] != target["desired_queue"] or old["desired_code"] != target["desired_code"])]
        gate("conflict", not conflicts, {"conflicting_subjects": conflicts, "policy": policy["conflict"]},
             failure="conflict_rejection")
        old_scores, local_scores, new_scores = (_head_scores(item, head, baseline)
                                              for item in (original, parent, candidate))
        expected = queue[labels]
        before, local, after = (queue[scores.argmax(1)] for scores in (old_scores, local_scores, new_scores))
        local_harm = float(np.mean((after != expected).astype(float)-(local != expected)))
        local_flip = float(np.mean(after != local))
        original_harm = float(np.mean((after != expected).astype(float)-(before != expected)))
        original_flip = float(np.mean(after != before))
        gate("local_protected_behavior", local_harm <= policy["maximum_local_net_harm"]
             and local_flip <= policy["maximum_local_decision_flip"],
             {"net_harm": local_harm, "decision_flip": local_flip})
        gate("original_protected_behavior", original_harm <= policy["maximum_original_net_harm"]
             and original_flip <= policy["maximum_original_decision_flip"],
             {"net_harm": original_harm, "decision_flip": original_flip}, required=cumulative)
        local_residual = exact_sample_drift(parent, candidate, head, baseline)
        gate("local_score_budget", Fraction(local_residual["maximum_exact"]) <= Fraction.from_float(float(policy["score_budget"])),
             {"budget": policy["score_budget"], "receipt": local_residual})
        residual = exact_sample_drift(original, candidate, head, baseline)
        gate("original_score_budget", Fraction(residual["maximum_exact"]) <= Fraction.from_float(float(policy["score_budget"])),
             {"budget": policy["score_budget"], "receipt": residual}, required=cumulative)
        parent_target = queue[_head_scores(parent, head, target_features).argmax(1)]
        candidate_target = queue[_head_scores(candidate, head, target_features).argmax(1)]
        target_gain = float(np.mean((candidate_target == target["desired_queue"]).astype(float)
                                    -(parent_target == target["desired_queue"])))
        gate("new_target_gain", target_gain >= policy["minimum_new_target_gain"],
             {"gain_from_current_parent": target_gain,
              "parent_accuracy": float(np.mean(parent_target == target["desired_queue"])),
              "candidate_accuracy": float(np.mean(candidate_target == target["desired_queue"])),
              "minimum_gain": policy["minimum_new_target_gain"]})
        for memory in value["retained"]:
            inserted, features = artifact(memory["inserted_endpoint_sha256"]), array(memory["features"])
            _head_only(original, inserted, head)
            inserted_codes = queue[_head_scores(inserted, head, features).argmax(1)]
            after_codes = queue[_head_scores(candidate, head, features).argmax(1)]
            accuracy_loss = float(np.mean((inserted_codes == memory["desired_queue"]).astype(float)
                                          -(after_codes == memory["desired_queue"])))
            flip = float(np.mean(inserted_codes != after_codes))
            drift = exact_sample_drift(inserted, candidate, head, features)
            passed = (accuracy_loss <= policy["maximum_old_target_loss"]
                      and flip <= policy["maximum_old_target_decision_flip"]
                      and Fraction(drift["maximum_exact"]) <= Fraction.from_float(float(policy["score_budget"])))
            gate("retained/"+memory["subject"]+"/"+str(memory["accepted_step"])+"/"+memory["request_sha256"], passed,
                 {"insertion": memory["inserted_endpoint_sha256"], "memory_age_steps": value["step"]-memory["accepted_step"],
                  "accepted_request_sha256": memory["request_sha256"],
                  "accuracy_loss": accuracy_loss, "decision_flip": flip, "score_budget": policy["score_budget"],
                  "receipt": drift}, required=cumulative)
    except (KeyError, ValueError, TypeError, IndexError, OverflowError) as error:
        gate("correspondence_failure", False, {"type": type(error).__name__, "message": str(error)},
             failure="missing_correspondence")
    accepted = all(row["status"] == "pass" for row in gates if row["required"])
    payload = {"schema": "wsp-cumulative-record/1", "request_sha256": request.sha256,
        "request": value, "gates": gates, "accepted": accepted,
        "no_op": value["parent_sha256"] == value["candidate_sha256"],
        "deployed_sha256": value["candidate_sha256"] if accepted else value["parent_sha256"],
        "scope": "Sampled cumulative affine-head contract; native decisions measured on declared CPU; no heldout utility claim"}
    return payload | {"sha256": digest(payload)}


def verify_step(record: dict[str, Any], artifacts: dict[str, TensorArtifact], arrays: dict[str, np.ndarray],
                expected_request_sha256: str) -> bool:
    try:
        request = CumulativeRequest.capture(record["request"])
        return (request.sha256 == expected_request_sha256
                and canonical_bytes(admit_step(request, artifacts, arrays)) == canonical_bytes(record))
    except (KeyError, ValueError, TypeError):
        return False


def _remember(record: dict[str, Any]) -> dict[str, Any]:
    request = record["request"]
    return request["target"] | {"inserted_endpoint_sha256": record["deployed_sha256"],
        "accepted_step": request["step"], "request_sha256": record["request_sha256"]}


def verify_sequence(sequence: dict[str, Any], artifacts: dict[str, TensorArtifact], arrays: dict[str, np.ndarray],
                    expected_contract_sha256: str) -> bool:
    """Replay transitions so removing an older obligation cannot reset a budget."""
    try:
        contract = sequence["contract"]
        if (contract.get("schema") != "wsp-cumulative-sequence-contract/1"
                or digest(contract) != expected_contract_sha256
                or sequence["contract_sha256"] != expected_contract_sha256
                or not contract["order"] or len(sequence["records"]) != len(contract["order"])):
            return False
        current, retained, history = contract["original_sha256"], [], expected_contract_sha256
        accepted_programs = []
        for step, (record, target) in enumerate(zip(sequence["records"], contract["order"]), 1):
            request = record["request"]
            fixed = {"original_sha256", "head", "queue_map", "baseline_features", "baseline_labels", "policy", "mode"}
            if (any(request[key] != contract[key] for key in fixed) or request["target"] != target
                    or request["sequence_contract_sha256"] != expected_contract_sha256
                    or request["parent_sha256"] != current or request["retained"] != retained
                    or request["history_sha256"] != history or request["step"] != step
                    or not verify_step(record, artifacts, arrays, record["request_sha256"])):
                return False
            if record["accepted"]:
                accepted_programs.append(Program.from_dict(request["program"]))
                retained.append(_remember(record))
            current, history = record["deployed_sha256"], record["sha256"]
        original = artifacts[contract["original_sha256"]]
        merged = compose(original, accepted_programs)
        inverse = rollback(original, merged)
        restored = apply_program(artifacts[current], inverse)
        expected_restore = {"schema": "wsp-cumulative-rollback/1", "original_sha256": original.sha256,
            "endpoint_sha256": current, "composition_sha256": digest(merged.to_dict()),
            "inverse": inverse.to_dict(), "restored_sha256": restored.sha256,
            "restored": restored.sha256 == original.sha256,
            "named_state": ["captured readout artifact tensors and immutable interface metadata"],
            "external_state_restored": False,
            "excluded_state": ["encoder mutation", "caches", "RNG", "assignments", "deployment process", "business actions"]}
        return sequence["endpoint_sha256"] == current and sequence["rollback"] == expected_restore
    except (KeyError, ValueError, TypeError):
        return False


def _sequence(original: TensorArtifact, targets: list[dict[str, Any]], head: ReadoutHead, queue_map: list[int],
              baseline_features: str, baseline_labels: str, policy: dict[str, Any], mode: str,
              artifacts: dict[str, TensorArtifact], arrays: dict[str, np.ndarray], proposer: Any) -> dict[str, Any]:
    contract = {"schema": "wsp-cumulative-sequence-contract/1", "original_sha256": original.sha256,
        "head": head.to_dict(), "queue_map": queue_map, "baseline_features": baseline_features,
        "baseline_labels": baseline_labels, "policy": policy, "mode": mode, "order": targets}
    contract_sha = digest(contract)
    current, memories, history, programs, records, proposals = original, [], contract_sha, [], [], []
    artifacts[original.sha256] = original
    for step, target in enumerate(targets, 1):
        started = time.perf_counter()
        error = None
        try:
            proposed = proposer(current, target)
            if (not isinstance(proposed, dict) or not isinstance(proposed.get("artifact"), TensorArtifact)
                    or not isinstance(proposed.get("program"), Program)):
                raise ValueError("A proposer must return a captured candidate and atomic program")
            candidate, program = proposed["artifact"], proposed["program"]
            proposals.append({"step": step, "fit_seconds": proposed.get("fit_seconds", 0.),
                "shared_proposal": proposed.get("shared_proposal", False), "diagnostics": proposed.get("diagnostics")})
        except (ValueError, RuntimeError, np.linalg.LinAlgError, KeyError, TypeError, OSError, IndexError) as failure:
            error = type(failure).__name__ + ": " + str(failure)
            candidate, program = current, difference(current, current, "Failed cumulative proposal; keep current parent")
            proposals.append({"step": step, "fit_seconds": time.perf_counter()-started, "failure": error})
        artifacts[candidate.sha256] = candidate
        request = bind_step(original=original, parent=current, candidate=candidate, program=program,
            head=head, queue_map=queue_map, baseline_features=baseline_features, baseline_labels=baseline_labels,
            target=target, retained=memories, policy=policy, mode=mode, step=step, history_sha256=history,
            sequence_contract_sha256=contract_sha, proposal_error=error)
        record = admit_step(request, artifacts, arrays)
        records.append(record)
        if record["accepted"]:
            programs.append(program)
            current = candidate
            memories.append(_remember(record))
        history = record["sha256"]
    merged = compose(original, programs)
    inverse = rollback(original, merged)
    restored = apply_program(current, inverse)
    restoration = {"schema": "wsp-cumulative-rollback/1", "original_sha256": original.sha256,
        "endpoint_sha256": current.sha256, "composition_sha256": digest(merged.to_dict()),
        "inverse": inverse.to_dict(), "restored_sha256": restored.sha256, "restored": restored.sha256 == original.sha256,
        "named_state": ["captured readout artifact tensors and immutable interface metadata"],
        "external_state_restored": False,
        "excluded_state": ["encoder mutation", "caches", "RNG", "assignments", "deployment process", "business actions"]}
    sequence = {"contract": contract, "contract_sha256": contract_sha, "records": records,
        "endpoint_sha256": current.sha256, "rollback": restoration, "proposals": proposals}
    if not verify_sequence(sequence, artifacts, arrays, contract_sha):
        raise ValueError("A cumulative sequence failed independent transition replay")
    return sequence


def _policy(score_budget: float) -> dict[str, Any]:
    return {"score_budget": score_budget, "maximum_original_decision_flip": .01,
        "maximum_local_decision_flip": .01, "maximum_original_net_harm": .01,
        "maximum_local_net_harm": .01, "maximum_old_target_loss": .01,
        "maximum_old_target_decision_flip": .01, "minimum_new_target_gain": .10,
        "conflict": "reject", "abstention": "keep_last_accepted"}


def boundary_sequences() -> tuple[dict[str, Any], dict[str, TensorArtifact], dict[str, np.ndarray]]:
    """Two individually allowed score changes exceed one original score cap."""
    base = TensorArtifact("cumulative-designed-affine/1", {"w": [[0.], [0.]], "b": [2., 0.]},
                          {"feature_id": "designed/1"})
    head = ReadoutHead("w", "b", "designed/1")
    artifacts, arrays = {base.sha256: base}, {}
    baseline = _add_array(arrays, np.array([[0.]], dtype=np.float32))
    labels = _add_array(arrays, np.array([0], dtype=np.int64))
    t1 = _add_array(arrays, np.array([[8.]], dtype=np.float32))
    t2 = _add_array(arrays, np.array([[-8.]], dtype=np.float32))
    order = [{"subject": "positive", "case_id": "first", "target_label": 1,
              "desired_code": 1, "desired_queue": 1, "features": t1},
             {"subject": "negative", "case_id": "second", "target_label": 1,
              "desired_code": 1, "desired_queue": 1, "features": t2}]
    policy = _policy(.5)
    # Scores move .375 twice while protected decisions stay unchanged. The
    # second request also erases the first target, supplying two global failures.
    def propose(parent: TensorArtifact, target: dict[str, Any]) -> dict[str, Any]:
        sign = 1. if target["subject"] == "positive" else -1.
        tensors = parent.tensors
        weight, bias = tensors["w"].copy(), tensors["b"].copy()
        weight[1, 0] = sign*.25
        bias[1] += .375
        candidate = parent.replace(w=weight, b=bias)
        return {"artifact": candidate, "program": difference(parent, candidate, "Designed cumulative budget counterexample")}
    result = {mode: _sequence(base, order, head, [0, 1], baseline, labels, policy, mode,
                              artifacts, arrays, propose) for mode in ("local", "cumulative")}
    conflict = order[0] | {"case_id": "conflict", "desired_code": 0, "desired_queue": 0}
    result["conflict"] = _sequence(base, [order[0], conflict], head, [0, 1], baseline, labels, policy,
                                    "cumulative", artifacts, arrays, propose)
    return result, artifacts, arrays


def _safe_lineage(directory: Path) -> dict[str, Any]:
    # load_lineage merges an available evaluation bundle. Copy only frozen
    # training inputs so candidate construction cannot accidentally read it.
    with tempfile.TemporaryDirectory(prefix="wsp-cumulative-calibration-") as temporary:
        copied = Path(temporary)
        for name in INPUT_FILES:
            shutil.copyfile(directory/name, copied/name)
        lineage = load_lineage(copied)
    lineage["directory"] = directory
    return lineage


def _hashed_rows(ids: np.ndarray, mask: np.ndarray, maximum: int = 128) -> np.ndarray:
    rows = np.flatnonzero(mask).tolist()
    rows.sort(key=lambda index: digest({"sample": str(ids[index]), "selection": "cumulative/1"}))
    return np.array(rows[:maximum], dtype=np.int64)


def _protocol() -> dict[str, Any]:
    inputs, domains = {}, {}
    for domain, (study, seeds) in DOMAINS.items():
        protocol_path = ROOT/"experiments/research"/study/"protocol.json"
        source = read_json(protocol_path)
        domains[domain] = {"study": study, "main_seeds": list(seeds), "cases": source["cases"],
                           "trained_protocol_sha256": file_hash(protocol_path)}
        inputs[protocol_path.relative_to(ROOT).as_posix()] = file_hash(protocol_path)
        for seed in seeds:
            directory = ROOT/"artifacts/research"/study/"main"/f"{domain}-main-full-{seed}"
            for name in INPUT_FILES:
                path = directory/name
                inputs[path.relative_to(ROOT).as_posix()] = file_hash(path)
            # Hash manifests, never load feature arrays, before the global seal.
            path = directory/"evaluation-features.json"
            inputs[path.relative_to(ROOT).as_posix()] = file_hash(path)
    return {"schema": "wsp-cumulative-protocol/1", "review_date_local": "2026-11-08",
        "frozen_at_utc": utc_now(), "domains": domains, "input_hashes": inputs,
        "orders": ["forward", "reverse"], "controllers": ["local", "cumulative"],
        "fitting": {"arm": "soft", "ridge": .01, "retention_weight": 8., "nullspace_rtol": 1e-12,
                    "target_role": "all rows of the scheduled label", "protected_fit_rows": 128},
        "calibration": "Hash-sort supplied IDs, retain at most 128 non-scheduled protected labels; all calibration rows of each scheduled target",
        "score_budget_rule": "Half the original protected-calibration median winning score margin, fixed for the whole sequence; model-score feasibility cap, not an enterprise SLA",
        "policy": _policy(0.), "policy_score_budget_replaced_by_frozen_baseline_rule": True,
        "local_controller": "The same fixed score cap and native local behavior/gain gates; original-baseline and prior-insertion gates are recorded but not required",
        "conflict": "Reject contrary code/queue obligations on an already accepted subject; do not silently replace a retained request",
        "abstention": "Deploy the last accepted parent; record every failed, rejected and no-op proposal",
        "heldout_boundary": "No evaluation-feature array read until every sequence candidate/admission is sealed in a global ledger",
        "heldout_scope": "Primary novel official-test feature rows; complete official test is a secondary sensitivity endpoint",
        "utility": {"minimum_target_gain": .10, "maximum_original_net_harm": .01,
                    "maximum_retained_target_gain_loss": .01, "minimum_useful_coverage": .75},
        "inference": "Descriptive paired 95% Student t over six independent full-training-lineage means per domain; average orders/steps inside lineage; not a confirmatory population claim",
        "missingness": "Keep every assigned full sequence; missing gain/harm sensitivity [-1,1], joint utility [0,1]; rejected proposals retain prior endpoint",
        "claim_limit": "These lineages were trained in an earlier sealed study; heldout rows are reused for an adjacent, explicitly secondary sequence study. No new independent data or population-editor confirmation is claimed",
        "rollback": "Inverse composition restores only the named captured readout artifact; encoder/preprocessing identities are bound and remain frozen, external state is excluded",
        "raw_publication": "Only derived attributed dataset features/model arrays and IDs; no raw acquired query/image rows"}


def seal(directory: str | Path) -> dict[str, Any]:
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    protocol = _protocol()
    files = {"protocol.json": write_json(out/"protocol.json", protocol)}
    sources = {}
    for relative in SOURCE_FILES:
        target = out/"sources"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write((ROOT/relative).read_bytes())
        sources[relative] = file_hash(target)
        files["sources/"+relative] = sources[relative]
    files["sources.json"] = write_json(out/"sources.json", sources)
    boundary, _, _ = boundary_sequences()
    files["boundary.json"] = write_json(out/"boundary.json", boundary)
    artifacts, arrays, sequences = {}, {}, []
    for domain, specification in protocol["domains"].items():
        study = specification["study"]
        for seed in specification["main_seeds"]:
            lineage_dir = ROOT/"artifacts/research"/study/"main"/f"{domain}-main-full-{seed}"
            lineage = _safe_lineage(lineage_dir)
            base, bundle = lineage["base"], lineage["features"]
            cases = specification["cases"]
            scheduled = [case["target_label"] for case in cases]
            chosen = _hashed_rows(bundle["calibration_ids"], ~np.isin(bundle["calibration_labels"], scheduled))
            protected_values = bundle["calibration_features"][chosen]
            baseline_features = _add_array(arrays, protected_values)
            baseline_labels = _add_array(arrays, bundle["calibration_labels"][chosen])
            baseline_scores = evaluate_head(base, protected_values, native=True)
            sorted_scores = np.sort(baseline_scores.astype(np.float64), axis=1)
            score_budget = float(np.median(sorted_scores[:, -1]-sorted_scores[:, -2]))/2.
            policy = _policy(score_budget)
            targets = [dict(subject="label:"+str(case["target_label"]), case_id=case["case_id"],
                target_label=case["target_label"], desired_code=case["desired_code"], desired_queue=case["desired_queue"],
                features=_add_array(arrays, bundle["calibration_features"][bundle["calibration_labels"] == case["target_label"]]))
                       for case in cases]
            cache = {}
            def proposer(parent: TensorArtifact, target: dict[str, Any]) -> dict[str, Any]:
                key = (parent.sha256, target["subject"], target["desired_code"])
                if key in cache:
                    return cache[key] | {"shared_proposal": True, "fit_seconds": 0.}
                selected = _hashed_rows(bundle["protected_ids"], bundle["protected_labels"] != target["target_label"])
                proposal = compile_arm(parent, bundle["target_features"][bundle["target_labels"] == target["target_label"]],
                    bundle["protected_features"][selected], target["desired_code"], "soft",
                    ridge=.01, retention_weight=8., nullspace_rtol=1e-12)
                cache[key] = proposal
                return proposal
            for order in protocol["orders"]:
                ordered = targets if order == "forward" else list(reversed(targets))
                for mode in protocol["controllers"]:
                    sequence = _sequence(base, ordered, head_spec(base), bundle["queue_map"].tolist(),
                        baseline_features, baseline_labels, policy, mode, artifacts, arrays, proposer)
                    sequence.update({"lineage_id": lineage["id"], "domain": domain, "order_name": order,
                                     "lineage_directory": lineage_dir.relative_to(ROOT).as_posix(),
                                     "encoder_sha256": base.metadata["encoder_sha256"],
                                     "preprocessing_sha256": base.metadata["preprocessing_sha256"]})
                    sequences.append(sequence)
            print("sealed cumulative candidates for "+lineage["id"], flush=True)
    for key, artifact in artifacts.items():
        relative = "artifacts/"+key+".json"
        files[relative] = write_json(out/relative, artifact.to_dict())
    arrays_path = out/"arrays.npz"
    with arrays_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    files["arrays.npz"] = file_hash(arrays_path)
    files["sequences.json"] = write_json(out/"sequences.json", sequences)
    seal = {"schema": "wsp-cumulative-global-seal/1", "protocol_sha256": files["protocol.json"],
        "sequences_sha256": files["sequences.json"], "sources_sha256": files["sources.json"],
        "arrays_sha256": files["arrays.npz"], "artifact_files": {key: value for key, value in files.items() if key.startswith("artifacts/")},
        "sealed_at_utc": utc_now(), "all_assigned_sequences_sealed": True,
        "evaluation_feature_arrays_read_by_this_study": False}
    files["global-seal.json"] = write_json(out/"global-seal.json", seal)
    manifest = {"schema": "wsp-cumulative-manifest/1", "files": files,
        "scope": "Evidence identity and exact replay; keep manifest/sequence-contract hashes externally for authentication"}
    write_json(out/"manifest.json", manifest)
    return seal


def _load_evaluation(directory: Path, feature_id: str) -> dict[str, np.ndarray]:
    path = directory/"evaluation-features.npz"
    manifest = read_json(directory/"evaluation-features.json")
    if (manifest["feature_id"] != feature_id or file_hash(path) != manifest["sha256"]
            or path.stat().st_size != manifest["bytes"]):
        raise ValueError("Evaluation feature correspondence failed")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _evaluate_sequence(sequence: dict[str, Any], artifacts: dict[str, TensorArtifact], bundle: dict[str, np.ndarray],
                       novel_only: bool) -> dict[str, Any]:
    contract = sequence["contract"]
    base = artifacts[contract["original_sha256"]]
    head = ReadoutHead(**contract["head"])
    queue = np.asarray(contract["queue_map"])
    width, outputs = head.dimensions(base)
    labels, features = np.asarray(bundle["test_labels"]), _feature_matrix(bundle["test_features"], width)
    if (labels.shape != (len(features),) or labels.dtype.kind not in "iu"
            or (labels < 0).any() or (labels >= outputs).any()
            or bundle["test_novel_mask"].shape != labels.shape):
        raise ValueError("Heldout labels/mask do not correspond to their declared features")
    mask = bundle["test_novel_mask"].astype(bool) if novel_only else np.ones(len(labels), dtype=bool)
    labels, features = labels[mask], features[mask]
    baseline_codes = queue[_head_scores(base, head, features).argmax(1)]
    protected = ~np.isin(labels, [target["target_label"] for target in contract["order"]])
    expected = queue[labels]
    endpoints = [base] + [artifacts[record["deployed_sha256"]] for record in sequence["records"]]
    memory_insertions = []
    rows = []
    for index, endpoint in enumerate(endpoints):
        codes = queue[_head_scores(endpoint, head, features).argmax(1)]
        if index and sequence["records"][index-1]["accepted"]:
            record = sequence["records"][index-1]
            target = record["request"]["target"]
            chosen = labels == target["target_label"]
            memory_insertions.append(target | {"step": index, "request_sha256": record["request_sha256"],
                "accuracy": float(np.mean(codes[chosen] == target["desired_queue"])) if chosen.any() else None})
        matrix = []
        for target in contract["order"]:
            chosen = labels == target["target_label"]
            accuracy = float(np.mean(codes[chosen] == target["desired_queue"])) if chosen.any() else None
            original = float(np.mean(baseline_codes[chosen] == target["desired_queue"])) if chosen.any() else None
            insertions = [memory for memory in memory_insertions if all(memory[key] == target[key]
                          for key in ("subject", "target_label", "desired_code", "desired_queue"))]
            retained = [{"accepted_step": memory["step"], "request_sha256": memory["request_sha256"],
                "memory_age_steps": index-memory["step"], "accuracy_at_insertion": memory["accuracy"],
                "loss_from_insertion": None if memory["accuracy"] is None or accuracy is None else memory["accuracy"]-accuracy}
                        for memory in insertions]
            losses = [memory["loss_from_insertion"] for memory in retained if memory["loss_from_insertion"] is not None]
            matrix.append({"case_id": target["case_id"], "target_label": target["target_label"],
                "accuracy": accuracy, "gain_from_original": None if accuracy is None else accuracy-original,
                "memory_age_steps": None if not retained else max(memory["memory_age_steps"] for memory in retained),
                "loss_from_insertion": max(losses) if losses else None, "retained_insertions": retained,
                "rows": int(chosen.sum())})
        harm = float(np.mean((codes[protected] != expected[protected]).astype(float)
                              -(baseline_codes[protected] != expected[protected]))) if protected.any() else None
        flip = float(np.mean(codes[protected] != baseline_codes[protected])) if protected.any() else None
        rows.append({"edit_step": index, "endpoint_sha256": endpoint.sha256,
            "accepted_step": False if index == 0 else sequence["records"][index-1]["accepted"],
            "rejected_step": False if index == 0 else not sequence["records"][index-1]["accepted"],
            "original_protected_net_harm": harm, "original_protected_decision_flip": flip,
            "protected_rows": int(protected.sum()), "target_matrix": matrix})
    final = rows[-1]
    gains = [item["gain_from_original"] for item in final["target_matrix"]]
    old_losses = [memory["loss_from_insertion"] for row in rows for item in row["target_matrix"]
                  for memory in item["retained_insertions"] if memory["memory_age_steps"] > 0
                  and memory["loss_from_insertion"] is not None]
    missing = any(value is None for value in gains) or any(row["original_protected_net_harm"] is None for row in rows)
    policy = contract["policy"]
    useful = (not missing and all(value >= policy["minimum_new_target_gain"] for value in gains)
              and all(row["original_protected_net_harm"] <= policy["maximum_original_net_harm"] for row in rows)
              and all(value <= policy["maximum_old_target_loss"] for value in old_losses))
    return {"missing": missing, "matrix": rows, "final_average_target_gain": None if missing else float(np.mean(gains)),
        "final_original_protected_net_harm": final["original_protected_net_harm"],
        "maximum_target_loss_from_insertion": max(old_losses, default=0.),
        "whole_sequence_joint_useful": useful, "novel_only": novel_only,
        "inference_unit": "Complete training lineage; edit steps/target cells/orders are repeated observations"}


def _evaluate(sequences: list[dict[str, Any]], artifacts: dict[str, TensorArtifact], protocol: dict[str, Any]) -> dict[str, Any]:
    from scripts.run_trained_research import mean_interval
    outcomes, evaluation_hashes, cache = [], {}, {}
    for sequence in sequences:
        identity = sequence["lineage_id"]
        directory = ROOT/sequence["lineage_directory"]
        try:
            for name in ("evaluation-features.json", "evaluation-features.npz"):
                evaluation_hashes[(directory/name).relative_to(ROOT).as_posix()] = file_hash(directory/name)
            if identity not in cache:
                cache[identity] = _load_evaluation(directory, sequence["contract"]["head"]["feature_id"])
            primary = _evaluate_sequence(sequence, artifacts, cache[identity], True)
            secondary = _evaluate_sequence(sequence, artifacts, cache[identity], False)
            outcomes.append({"lineage_id": identity, "domain": sequence["domain"], "order": sequence["order_name"],
                "controller": sequence["contract"]["mode"], "assigned": True, "primary": primary, "secondary": secondary,
                "admitted_steps": sum(record["accepted"] for record in sequence["records"]),
                "rejected_steps": sum(not record["accepted"] for record in sequence["records"]),
                "no_op_candidates": sum(record["no_op"] for record in sequence["records"]),
                "rollback_verified": sequence["rollback"]["restored"]})
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, IndexError) as error:
            outcomes.append({"lineage_id": identity, "domain": sequence["domain"], "order": sequence["order_name"],
                "controller": sequence["contract"]["mode"], "assigned": True, "primary": {"missing": True},
                "failure": {"type": type(error).__name__, "message": str(error)},
                "sensitivity": {"target_gain": [-1, 1], "protected_harm": [-1, 1], "useful": [0, 1]}})
    summaries, contrasts = [], []
    for domain in protocol["domains"]:
        grouped = {}
        for mode in protocol["controllers"]:
            selected = [row for row in outcomes if row["domain"] == domain and row["controller"] == mode]
            lineages = []
            for identity in sorted(set(row["lineage_id"] for row in selected)):
                assigned = [row for row in selected if row["lineage_id"] == identity]
                observed = [row["primary"] for row in assigned if not row["primary"]["missing"]]
                missing = len(assigned)-len(observed)
                gain = sum(row["final_average_target_gain"] for row in observed)/len(assigned)
                harm = sum(row["final_original_protected_net_harm"] for row in observed)/len(assigned)
                useful = sum(row["whole_sequence_joint_useful"] for row in observed)/len(assigned)
                lineages.append({"lineage_id": identity, "assigned_sequences": len(assigned), "missing": missing,
                    "gain_zero_missing": gain, "harm_zero_missing": harm, "useful_all_assigned": useful,
                    "gain_missing_bounds": [gain-missing/len(assigned), gain+missing/len(assigned)],
                    "harm_missing_bounds": [harm-missing/len(assigned), harm+missing/len(assigned)],
                    "useful_missing_bounds": [useful, useful+missing/len(assigned)]})
            grouped[mode] = lineages
            summaries.append({"domain": domain, "controller": mode, "lineages": lineages,
                "target_gain": mean_interval([row["gain_zero_missing"] for row in lineages]),
                "original_protected_harm": mean_interval([row["harm_zero_missing"] for row in lineages]),
                "whole_sequence_useful_coverage": mean_interval([row["useful_all_assigned"] for row in lineages]),
                "missing_sequences": sum(row["primary"]["missing"] for row in selected)})
        paired = []
        for left, right in zip(grouped["cumulative"], grouped["local"]):
            if left["lineage_id"] != right["lineage_id"]:
                raise ValueError("Paired full-lineage order mismatch")
            paired.append({"lineage_id": left["lineage_id"], "target_gain_difference": left["gain_zero_missing"]-right["gain_zero_missing"],
                "original_harm_difference": left["harm_zero_missing"]-right["harm_zero_missing"],
                "useful_difference": left["useful_all_assigned"]-right["useful_all_assigned"],
                "target_gain_missing_bounds": [left["gain_missing_bounds"][0]-right["gain_missing_bounds"][1],
                                               left["gain_missing_bounds"][1]-right["gain_missing_bounds"][0]],
                "original_harm_missing_bounds": [left["harm_missing_bounds"][0]-right["harm_missing_bounds"][1],
                                                 left["harm_missing_bounds"][1]-right["harm_missing_bounds"][0]],
                "useful_missing_bounds": [left["useful_missing_bounds"][0]-right["useful_missing_bounds"][1],
                                          left["useful_missing_bounds"][1]-right["useful_missing_bounds"][0]]})
        contrasts.append({"domain": domain, "contrast": "cumulative minus local", "lineages": paired,
            "target_gain": mean_interval([row["target_gain_difference"] for row in paired]),
            "original_harm": mean_interval([row["original_harm_difference"] for row in paired]),
            "useful_coverage": mean_interval([row["useful_difference"] for row in paired])})
    return {"schema": "wsp-cumulative-results/1", "outcomes": outcomes, "summaries": summaries,
        "paired_contrasts": contrasts, "evaluation_input_hashes": evaluation_hashes,
        "designation": "Adjacent secondary sequence estimation on previously acquired public heldout rows; no confirmatory population claim",
        "interval_missingness_rule": "Intervals use zero-filled missing gains/harms and failure-filled useful coverage; per-lineage and paired missingness bounds remain explicit",
        "test_counts_are_impact": False}


def _verify_seal(directory: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, TensorArtifact]]:
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out = Path(directory)
    manifest = read_json(out/"manifest.json")
    if manifest["schema"] != "wsp-cumulative-manifest/1":
        raise ValueError("Unknown cumulative manifest")
    verify_manifest(out, manifest["files"])
    sources = read_json(out/"sources.json")
    if set(sources) != set(SOURCE_FILES) or any(file_hash(ROOT/key) != value for key, value in sources.items()):
        raise ValueError("Current cumulative execution differs from frozen sources")
    protocol = read_json(out/"protocol.json")
    for key, expected in protocol["input_hashes"].items():
        path = (ROOT/key).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path) != expected:
            raise ValueError("A frozen cumulative input changed")
    seal = read_json(out/"global-seal.json")
    for field, name in (("protocol_sha256", "protocol.json"), ("sequences_sha256", "sequences.json"),
                        ("sources_sha256", "sources.json"), ("arrays_sha256", "arrays.npz")):
        if seal[field] != file_hash(out/name):
            raise ValueError("Global sequence seal correspondence failed")
    expected_artifacts = {key: value for key, value in manifest["files"].items() if key.startswith("artifacts/")}
    if (seal["artifact_files"] != expected_artifacts or not seal["all_assigned_sequences_sealed"]
            or seal["evaluation_feature_arrays_read_by_this_study"] is not False
            or {"artifacts/"+path.name for path in (out/"artifacts").glob("*.json")} != set(expected_artifacts)):
        raise ValueError("The global seal does not identify the complete candidate artifact pool")
    artifacts = {path.stem: TensorArtifact.from_dict(read_json(path)) for path in (out/"artifacts").glob("*.json")}
    with np.load(out/"arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    sequences = read_json(out/"sequences.json")
    expected = {(domain, f"{domain}-main-full-{seed}", order, mode)
                for domain, specification in protocol["domains"].items() for seed in specification["main_seeds"]
                for order in protocol["orders"] for mode in protocol["controllers"]}
    actual = [(sequence["domain"], sequence["lineage_id"], sequence["order_name"], sequence["contract"]["mode"])
              for sequence in sequences]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("The seal omits or duplicates an assigned lineage/order/controller")
    for sequence in sequences:
        if not verify_sequence(sequence, artifacts, arrays, sequence["contract_sha256"]):
            raise ValueError("An endpoint failed cumulative contract and transition replay")
    boundary, _, _ = boundary_sequences()
    if canonical_bytes(boundary) != canonical_bytes(read_json(out/"boundary.json")):
        raise ValueError("Designed cumulative falsifiers failed replay")
    return protocol, sequences, artifacts


def verify_seal(directory: str | Path) -> bool:
    _verify_seal(directory)
    return True


def evaluate(directory: str | Path) -> dict[str, Any]:
    out = Path(directory)
    if (out/"results.json").exists() or (out/"evaluation-manifest.json").exists():
        raise FileExistsError("Cumulative evaluation evidence already exists")
    protocol, sequences, artifacts = _verify_seal(out)
    results = _evaluate(sequences, artifacts, protocol)
    results["global_seal_sha256"] = file_hash(out/"global-seal.json")
    result_hash = write_json(out/"results.json", results)
    write_json(out/"evaluation-manifest.json", {"schema": "wsp-cumulative-evaluation-manifest/1",
        "candidate_manifest_sha256": file_hash(out/"manifest.json"), "files": {"results.json": result_hash}})
    return results


def run(directory: str | Path) -> dict[str, Any]:
    seal(directory)
    return evaluate(directory)


def verify(directory: str | Path) -> bool:
    out = Path(directory)
    protocol, sequences, artifacts = _verify_seal(out)
    evaluation_manifest = read_json(out/"evaluation-manifest.json")
    if (evaluation_manifest["schema"] != "wsp-cumulative-evaluation-manifest/1"
            or evaluation_manifest["candidate_manifest_sha256"] != file_hash(out/"manifest.json")
            or set(evaluation_manifest["files"]) != {"results.json"}):
        raise ValueError("Evaluation no longer corresponds to the frozen candidate manifest")
    verify_manifest(out, evaluation_manifest["files"])
    actual = _evaluate(sequences, artifacts, protocol)
    actual["global_seal_sha256"] = file_hash(out/"global-seal.json")
    if canonical_bytes(actual) != canonical_bytes(read_json(out/"results.json")):
        raise ValueError("Heldout cumulative matrix/uncertainty failed independent replay")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "seal", "evaluate", "verify-seal", "verify"))
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    action = {"run": run, "seal": seal, "evaluate": evaluate, "verify-seal": verify_seal, "verify": verify}[arguments.action]
    result = action(arguments.directory)
    print(json.dumps({"action": arguments.action, "completed": bool(result)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
