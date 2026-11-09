"""Exact rational real-arithmetic bounds; never silently call floats a proof.

Every finite binary64 parameter is lifted to its exact rational value. This is a
sound bound for the corresponding mathematical ReLU network over a real box.
The independently checked Lean theorem justifies the formula; Python itself and
IEEE-754 execution are not thereby formally verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from typing import Any

import numpy as np

from .model import Network, InputBox, canonical_bytes, digest, finite_binary64
from .program import Program, apply_program


def rational(value: float) -> Fraction:
    return Fraction.from_float(finite_binary64(value))


def exact_score(model: Network, inputs: list[float]) -> tuple[Fraction, ...]:
    if len(inputs) != model.shape[0]:
        raise ValueError("Wrong exact-score input")
    x = list(map(rational, inputs))
    hidden = [max(Fraction(0), rational(b) + sum((rational(w)*v for w, v in zip(row, x)), Fraction(0)))
              for row, b in zip(model.w1, model.b1)]
    return tuple(rational(b) + sum((rational(w)*h for w, h in zip(row, hidden)), Fraction(0))
                 for row, b in zip(model.w2, model.b2))


def certify_drift(base: Network, candidate: Network, box: InputBox) -> dict[str, Any]:
    """A global componentwise score-drift bound for simultaneous all-layer edits."""
    if not base.compatible(candidate) or len(box.lower) != base.shape[0]:
        raise ValueError("Certificate schemas/dimensions do not agree")
    lower, upper = list(map(rational, box.lower)), list(map(rational, box.upper))
    radius = [max(abs(lo), abs(hi)) for lo, hi in zip(lower, upper)]
    hidden_upper, hidden_drift = [], []
    for h in range(base.shape[1]):
        old_row = list(map(rational, base.w1[h]))
        new_row = list(map(rational, candidate.w1[h]))
        hidden_upper.append(max(Fraction(0), rational(base.b1[h]) + sum(
            (max(w*lo, w*hi) for w, lo, hi in zip(old_row, lower, upper)), Fraction(0))))
        hidden_drift.append(abs(rational(candidate.b1[h])-rational(base.b1[h])) + sum(
            (abs(v-w)*r for v, w, r in zip(new_row, old_row, radius)), Fraction(0)))
    bounds = [abs(rational(candidate.b2[o])-rational(base.b2[o])) + sum(
        (abs(rational(candidate.w2[o,h])-rational(base.w2[o,h]))*hidden_upper[h]
         + abs(rational(candidate.w2[o,h]))*hidden_drift[h]
         for h in range(base.shape[1])), Fraction(0)) for o in range(base.shape[2])]
    payload = {"schema": "wsp-drift-certificate/1", "arithmetic": "exact-rational-lift-of-binary64",
               "base_sha256": base.sha256, "candidate_sha256": candidate.sha256,
               "domain": box.to_dict(), "hidden_upper": list(map(str, hidden_upper)),
               "hidden_drift": list(map(str, hidden_drift)), "output_bounds": list(map(str, bounds)),
               "theorem": "WeightSpace.score_drift_le", "implementation_verified": False,
               "scope": "real score drift on the stated box; excludes floating-point error and task reward"}
    return payload | {"sha256": digest(payload)}


def verify_certificate(base: Network, candidate: Network, certificate: dict[str, Any]) -> bool:
    """Independently recompute, including all claims and provenance in the receipt."""
    try:
        box = InputBox(**certificate["domain"])
        return canonical_bytes(certificate) == canonical_bytes(certify_drift(base, candidate, box))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class Contract:
    box: InputBox
    max_output_drift: tuple[float, ...]
    allowed_tensors: tuple[str, ...] = ("w1", "b1", "w2", "b2")
    max_parameter_abs: float = 64.0
    max_edits: int | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.box, InputBox) or isinstance(self.max_output_drift, (str, bytes, dict))
                or isinstance(self.allowed_tensors, (str, bytes, dict))):
            raise ValueError("Contracts need an input box and sequences of budgets and tensor names")
        object.__setattr__(self, "max_output_drift", tuple(map(finite_binary64, self.max_output_drift)))
        object.__setattr__(self, "allowed_tensors", tuple(self.allowed_tensors))
        object.__setattr__(self, "max_parameter_abs", finite_binary64(self.max_parameter_abs))
        if (not self.max_output_drift or any(x < 0 for x in self.max_output_drift)
                or self.max_parameter_abs <= 0):
            raise ValueError("Contract budgets must be finite and nonnegative")
        if set(self.allowed_tensors) - {"w1", "b1", "w2", "b2"}:
            raise ValueError("Unknown writable tensor")
        if self.max_edits is not None and (type(self.max_edits) is not int or self.max_edits < 0):
            raise ValueError("Invalid edit-count budget")

    def to_dict(self) -> dict[str, Any]:
        return {"box": self.box.to_dict(), "max_output_drift": list(self.max_output_drift),
                "allowed_tensors": list(self.allowed_tensors), "max_parameter_abs": self.max_parameter_abs,
                "max_edits": self.max_edits}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Contract:
        if (not isinstance(value, dict)
                or set(value) != {"box", "max_output_drift", "allowed_tensors", "max_parameter_abs", "max_edits"}):
            raise ValueError("Unknown contract schema")
        if (not isinstance(value["box"], dict) or set(value["box"]) != {"lower", "upper"}
                or not isinstance(value["box"]["lower"], list) or not isinstance(value["box"]["upper"], list)
                or not isinstance(value["max_output_drift"], list)
                or not isinstance(value["allowed_tensors"], list)):
            raise ValueError("Serialized contracts require a box object and JSON arrays")
        return cls(InputBox(**value["box"]), tuple(value["max_output_drift"]),
                   tuple(value["allowed_tensors"]), value["max_parameter_abs"], value["max_edits"])


def check_contract(base: Network, program: Program, contract: Contract) -> dict[str, Any]:
    candidate = apply_program(base, program)
    if len(contract.max_output_drift) != base.shape[2]:
        raise ValueError("One drift budget is required per output")
    issues = []
    if any(edit.tensor not in contract.allowed_tensors for edit in program.edits):
        issues.append("write-set")
    if any(np.any(np.abs(t) > contract.max_parameter_abs) for t in candidate.tensors.values()):
        issues.append("parameter-bound")
    if contract.max_edits is not None and len(program.edits) > contract.max_edits:
        issues.append("edit-count")
    certificate = certify_drift(base, candidate, contract.box)
    if any(Fraction(bound) > rational(limit)
           for bound, limit in zip(certificate["output_bounds"], contract.max_output_drift)):
        issues.append("behavioral-drift")
    payload = {"schema": "wsp-admission/1", "accepted": not issues, "issues": issues,
               "program_sha256": program.sha256, "contract": contract.to_dict(),
               "certificate": certificate}
    return payload | {"sha256": digest(payload)}


def preserves_argmax(scores: list[float], per_output_bounds: list[str]) -> bool:
    """Sufficient strict-margin condition; ties intentionally yield no guarantee."""
    if len(scores) != len(per_output_bounds) or len(scores) < 2:
        raise ValueError("An argmax certificate needs at least two scores/bounds")
    values = list(map(rational, scores))
    bounds = list(map(Fraction, per_output_bounds))
    if any(b < 0 for b in bounds):
        raise ValueError("Drift bounds are nonnegative")
    winner = max(range(len(values)), key=values.__getitem__)
    return all(values[winner]-values[j] > bounds[winner]+bounds[j]
               for j in range(len(values)) if j != winner)


GATE_STATUSES = ("invalid", "unsupported", "missing_correspondence", "conservative_rejection",
                 "observed_violation", "pass")
_STATUS_PRIORITY = ("invalid", "observed_violation", "missing_correspondence", "unsupported", "conservative_rejection", "pass")


@dataclass(frozen=True)
class DecisionRule:
    """Declared winner semantics, including ties and nonfinite scores."""

    kind: str = "unique_argmax"
    tie_rule: str = "reject"
    nonfinite: str = "reject"

    def __post_init__(self) -> None:
        if (self.kind not in ("unique_argmax", "argmax", "scores_only", "stochastic")
                or self.nonfinite != "reject"
                or (self.kind == "argmax" and self.tie_rule not in ("first_index", "last_index"))
                or (self.kind != "argmax" and self.tie_rule != "reject")):
            raise ValueError("Declare a supported decision kind, explicit argmax tie rule, and nonfinite rejection")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "tie_rule": self.tie_rule, "nonfinite": self.nonfinite}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DecisionRule:
        if not isinstance(value, dict) or set(value) != {"kind", "tie_rule", "nonfinite"}:
            raise ValueError("Unknown decision-rule fields")
        return cls(**value)

    def classify(self, scores: Any) -> dict[str, Any]:
        try:
            if isinstance(scores, (str, bytes, dict)):
                raise ValueError("Scores must be a nonempty vector")
            raw = list(scores)
            if not raw:
                raise ValueError("Scores must be a nonempty vector")
            if any(isinstance(value, (float, np.floating)) and not np.isfinite(value) for value in raw):
                return {"status": "observed_violation", "winner": None, "tied_indices": [],
                        "score_count": len(raw), "reason": "Nonfinite scores are rejected"}
            values = [value if isinstance(value, Fraction) else rational(value) for value in raw]
        except (TypeError, ValueError, OverflowError):
            return {"status": "invalid", "winner": None, "tied_indices": [],
                    "score_count": None, "reason": "Scores must be a nonempty numeric vector"}
        result = {"status": "pass", "winner": None, "tied_indices": [],
                  "score_count": len(values), "reason": None}
        if self.kind == "scores_only":
            return result | {"reason": "No winner claim requested"}
        if self.kind == "stochastic":
            return result | {"status": "unsupported", "reason": "No stochastic decision coupling is registered"}
        maximum = max(values)
        tied = [i for i, value in enumerate(values) if value == maximum]
        result["tied_indices"] = tied
        if self.kind == "unique_argmax" and len(tied) != 1:
            return result | {"status": "conservative_rejection", "reason": "A tie has no unique winner"}
        return result | {"winner": tied[-1] if self.tie_rule == "last_index" else tied[0]}

    def preservation(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        for status in _STATUS_PRIORITY[:-1]:
            if status in (before.get("status"), after.get("status")):
                return {"status": status, "before": before.get("winner"), "after": after.get("winner"),
                        "reason": "At least one endpoint has no usable declared decision"}
        if (before.get("status") != "pass" or after.get("status") != "pass"
                or before.get("score_count") != after.get("score_count")):
            return {"status": "invalid", "before": before.get("winner"), "after": after.get("winner"),
                    "reason": "Decision records have incompatible output coordinates"}
        preserved = self.kind == "scores_only" or before["winner"] == after["winner"]
        return {"status": "pass" if preserved else "observed_violation",
                "before": before["winner"], "after": after["winner"],
                "reason": "No winner claim requested" if self.kind == "scores_only" else
                          None if preserved else "The declared winner changed"}


def _artifact(value: dict[str, Any]) -> Any:
    from .artifact import TensorArtifact
    if isinstance(value, dict) and value.get("schema") == "wsp-network/1":
        return Network.from_dict(value)
    return TensorArtifact.from_dict(value)


def _task(value: Any) -> Any:
    from .workflow import TaskSpec
    if type(value) is TaskSpec:
        return value
    if (not isinstance(value, dict) or set(value) != {
            "schema", "task_id", "mode", "objective", "architecture", "base_sha256", "constraints"}
            or value["schema"] != "wsp-task/1"):
        raise ValueError("Admission requires a TaskSpec or its strict serialized form")
    return TaskSpec(**{key: item for key, item in value.items() if key != "schema"})


def _semantic_artifact(artifact: Any) -> Any:
    from .semantics import to_tensor_artifact
    return to_tensor_artifact(artifact) if type(artifact) is Network else artifact


def _interface(artifact: Any, width: int, outputs: int) -> dict[str, Any]:
    from .model import _labels
    from .semantics import RELU_ARCHITECTURE, as_relu_network
    from .sequential import SEQUENTIAL_ARCHITECTURE, validate_sequential
    known = False
    if type(artifact) is Network:
        inputs, names, known = artifact.input_names, artifact.output_names, True
    else:
        metadata = artifact.metadata
        inputs, names = metadata.get("input_names", ()), metadata.get("output_names", ())
        if artifact.architecture == RELU_ARCHITECTURE:
            model = as_relu_network(artifact)
            inputs, names, known = model.input_names, model.output_names, True
        elif artifact.architecture == SEQUENTIAL_ARCHITECTURE:
            validate_sequential(artifact)
            known = True
    return {"input_names": list(_labels(inputs, width, "x")),
            "output_names": list(_labels(names, outputs, "y")),
            "bias_convention": "explicit-per-layer-vector-after-dot/1" if known else "caller-declared/1",
            "tensor_coordinate_order": "C"}


def _registry(registry: Any) -> Any:
    from .semantics import SemanticsRegistry, default_registry
    if registry is None:
        return default_registry()
    if not isinstance(registry, SemanticsRegistry):
        raise ValueError("Admission needs a caller-owned SemanticsRegistry")
    return registry


def _execution_scope(spec: Any) -> dict[str, Any]:
    from .evidence import numerical_environment
    from .structured import ExecutionSpec
    if not isinstance(spec, ExecutionSpec):
        raise ValueError("Use an ExecutionSpec for the measured native lane")
    try:
        return {"spec": spec.to_dict(), "environment": numerical_environment(spec.backend, spec.device),
                "status": "pass", "reason": None}
    except (ImportError, ValueError, RuntimeError) as error:
        return {"spec": spec.to_dict(), "environment": None, "status": "unsupported", "reason": str(error)}


_REQUEST_FIELDS = {"schema", "parent", "candidate", "parent_sha256", "candidate_sha256", "program",
                   "program_sha256", "task", "task_sha256", "interface", "adapter", "domain",
                   "structural_policy", "max_output_drift", "decision_rule", "witness_inputs",
                   "witness_sha256", "execution", "feature_retention", "sha256"}


@dataclass(frozen=True, slots=True, init=False)
class AdmissionRequest:
    """Frozen bytes shared by every gate; a digest requires an external trust anchor."""

    _bytes: bytes

    def __init__(self, value: dict[str, Any]):
        from .artifact import StructuralPolicy
        from .model import _TensorData, _labels
        from .structured import ExecutionSpec
        from .workflow import _json, _hash
        if (not isinstance(value, dict) or set(value) != _REQUEST_FIELDS
                or value["schema"] != "wsp-admission-request/1"):
            raise ValueError("Unknown admission-request schema or fields")
        normalized = _json(value)
        if normalized["sha256"] != digest({k: v for k, v in normalized.items() if k != "sha256"}):
            raise ValueError("Admission-request digest differs from its immutable bytes")
        for key in ("sha256", "parent_sha256", "candidate_sha256", "program_sha256", "task_sha256", "witness_sha256"):
            _hash(normalized[key], key)
        _artifact(normalized["parent"])
        _artifact(normalized["candidate"])
        Program.from_dict(normalized["program"])
        _task(normalized["task"])
        StructuralPolicy.from_dict(normalized["structural_policy"])
        DecisionRule.from_dict(normalized["decision_rule"])
        domain = normalized["domain"]
        if (not isinstance(domain, dict) or set(domain) != {"lower", "upper"}
                or not isinstance(domain["lower"], list) or not isinstance(domain["upper"], list)):
            raise ValueError("Admission domain requires lower and upper JSON arrays")
        box = InputBox(**domain)
        budgets = normalized["max_output_drift"]
        if not isinstance(budgets, list) or not budgets or any(finite_binary64(x) < 0 for x in budgets):
            raise ValueError("Admission needs finite nonnegative per-output budgets")
        interface = normalized["interface"]
        if (not isinstance(interface, dict) or set(interface) != {
                "input_names", "output_names", "bias_convention", "tensor_coordinate_order"}
                or not isinstance(interface["input_names"], list) or not isinstance(interface["output_names"], list)
                or not interface["input_names"] or not interface["output_names"]
                or not isinstance(interface["bias_convention"], str)
                or not isinstance(interface["tensor_coordinate_order"], str)):
            raise ValueError("Admission requires an explicit ordered interface and bias convention")
        _labels(interface["input_names"], len(box.lower), "x")
        _labels(interface["output_names"], len(budgets), "y")
        witnesses = _TensorData.capture(normalized["witness_inputs"]).array()
        if witnesses.ndim != 2 or witnesses.shape[1] != len(box.lower):
            raise ValueError("Witnesses must be a nonempty matrix in the declared input order")
        adapter = normalized["adapter"]
        if adapter is not None:
            if (not isinstance(adapter, dict) or set(adapter) != {
                    "architecture", "version", "deterministic", "implementation_sha256"}
                    or not isinstance(adapter["architecture"], str) or not isinstance(adapter["version"], str)
                    or type(adapter["deterministic"]) is not bool):
                raise ValueError("Unknown adapter descriptor")
            if adapter["implementation_sha256"] is not None:
                _hash(adapter["implementation_sha256"], "Adapter implementation")
        execution = normalized["execution"]
        if execution is not None:
            if (not isinstance(execution, dict) or set(execution) != {"spec", "environment", "status", "reason"}
                    or execution["status"] not in ("pass", "unsupported")):
                raise ValueError("Unknown pinned native scope")
            ExecutionSpec.from_dict(execution["spec"])
        feature = normalized["feature_retention"]
        if feature is not None:
            from .features import ReadoutHead, _matrix
            if (not isinstance(feature, dict) or set(feature) != {"head", "feature_names", "anchors", "receipt"}
                    or not isinstance(feature["head"], dict)
                    or set(feature["head"]) != {"weight", "bias", "feature_id"}
                    or not isinstance(feature["feature_names"], list)):
                raise ValueError("Unknown feature-retention binding")
            head = ReadoutHead(**feature["head"])
            width, _ = head.dimensions(_artifact(normalized["parent"]))
            _labels(feature["feature_names"], width, "f")
            _matrix(feature["anchors"], width, "Retention features")
            if feature["receipt"] is not None and not isinstance(feature["receipt"], dict):
                raise ValueError("Feature evidence must be a receipt object or absent")
        object.__setattr__(self, "_bytes", canonical_bytes(normalized))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._bytes)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AdmissionRequest:
        return cls(value)

    @property
    def sha256(self) -> str:
        return self.to_dict()["sha256"]


@dataclass(frozen=True, slots=True, init=False)
class AdmissionRecord:
    _bytes: bytes

    def __init__(self, value: dict[str, Any]):
        from .workflow import _json
        if (not isinstance(value, dict) or set(value) != {
                "schema", "binding_sha256", "request", "gates", "accepted", "claims", "scope", "sha256"}
                or value["schema"] != "wsp-integrated-admission/1" or type(value["accepted"]) is not bool
                or not isinstance(value["gates"], dict) or not isinstance(value["claims"], dict)):
            raise ValueError("Unknown integrated-admission schema or fields")
        normalized = _json(value)
        request = AdmissionRequest.from_dict(normalized["request"])
        if (request.sha256 != normalized["binding_sha256"]
                or normalized["sha256"] != digest({k: v for k, v in normalized.items() if k != "sha256"})):
            raise ValueError("Admission record differs from its request or gate bytes")
        for gate in normalized["gates"].values():
            if (not isinstance(gate, dict) or set(gate) != {
                    "binding_sha256", "status", "required", "reason", "evidence"}
                    or gate["binding_sha256"] != request.sha256 or gate["status"] not in GATE_STATUSES
                    or type(gate["required"]) is not bool):
                raise ValueError("Every gate must identify the same immutable request")
        object.__setattr__(self, "_bytes", canonical_bytes(normalized))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._bytes)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AdmissionRecord:
        return cls(value)

    @property
    def sha256(self) -> str:
        return self.to_dict()["sha256"]

    @property
    def binding_sha256(self) -> str:
        return self.to_dict()["binding_sha256"]

    @property
    def accepted(self) -> bool:
        return self.to_dict()["accepted"]


def bind_admission(base: Any, candidate: Any, program: Program, *, task: Any, policy: Any,
                   box: InputBox, max_output_drift: Any, decision: DecisionRule,
                   witnesses: Any, registry: Any = None, execution: Any = None,
                   feature_retention: dict[str, Any] | None = None,
                   interface: dict[str, Any] | None = None) -> AdmissionRequest:
    """Capture the request before checking it, including stale/invalid proposals."""
    from .artifact import TensorArtifact, StructuralPolicy
    from .features import ReadoutHead, _matrix, certify_feature_retention
    from .model import _TensorData, _labels
    if (type(base) not in (Network, TensorArtifact) or type(candidate) not in (Network, TensorArtifact)
            or not isinstance(program, Program) or not isinstance(policy, StructuralPolicy)
            or not isinstance(box, InputBox) or not isinstance(decision, DecisionRule)):
        raise ValueError("Bind immutable artifacts, a Program, a StructuralPolicy, a box, and a DecisionRule")
    budgets = [finite_binary64(value) for value in max_output_drift]
    samples = _TensorData.capture(witnesses).array()
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    task = _task(task)
    registry = _registry(registry)
    try:
        adapter = registry.adapter(_semantic_artifact(base)).descriptor()
    except ValueError:
        adapter = None
    feature = None
    if feature_retention is not None:
        if (not isinstance(feature_retention, dict) or set(feature_retention) - {
                "head", "anchors", "receipt", "feature_names"} or not {"head", "anchors"} <= set(feature_retention)):
            raise ValueError("Feature retention binds head, ordered anchors, optional names and receipt")
        raw_head = feature_retention["head"]
        head = raw_head if isinstance(raw_head, ReadoutHead) else ReadoutHead(**raw_head)
        width, _ = head.dimensions(base)
        anchors = _matrix(feature_retention["anchors"], width, "Retention features").tolist()
        if "receipt" in feature_retention:
            receipt = feature_retention["receipt"]
        else:
            try:
                receipt = certify_feature_retention(base, candidate, head, anchors)
            except ValueError:
                receipt = None
        feature = {"head": head.to_dict(), "anchors": anchors, "receipt": receipt,
                   "feature_names": list(_labels(feature_retention.get("feature_names", ()), width, "f"))}
    payload = {
        "schema": "wsp-admission-request/1", "parent": base.to_dict(), "candidate": candidate.to_dict(),
        "parent_sha256": base.sha256, "candidate_sha256": candidate.sha256,
        "program": program.to_dict(), "program_sha256": program.sha256,
        "task": task.to_dict(), "task_sha256": task.sha256,
        "interface": _interface(base, len(box.lower), len(budgets)) if interface is None else interface,
        "adapter": adapter, "domain": box.to_dict(), "structural_policy": policy.to_dict(),
        "max_output_drift": budgets, "decision_rule": decision.to_dict(),
        "witness_inputs": samples.tolist(), "witness_sha256": digest(samples.tolist()),
        "execution": None if execution is None else _execution_scope(execution), "feature_retention": feature,
    }
    return AdmissionRequest(payload | {"sha256": digest(payload)})


def _result_status(observations: list[dict[str, Any]]) -> str:
    return next((status for status in _STATUS_PRIORITY if any(row["status"] == status for row in observations)), "pass")


def _margin_rows(parent: list[list[Fraction]], bounds: list[Fraction], rule: DecisionRule,
                 errors: tuple[list[Any], list[Any]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for index, scores in enumerate(parent):
        classification = rule.classify(scores)
        if rule.kind == "scores_only":
            rows.append({"status": "pass", "reason": "No winner claim requested", "comparisons": []})
            continue
        if classification["status"] != "pass":
            rows.append({"status": classification["status"], "reason": classification["reason"], "comparisons": []})
            continue
        winner, comparisons = classification["winner"], []
        for other in range(len(scores)):
            if other == winner:
                continue
            allowance = bounds[winner] + bounds[other]
            if errors is not None:
                try:
                    allowance += sum((abs(Fraction(lane[index][coordinate]["total_error_exact"]))
                                      for lane in errors for coordinate in (winner, other)), Fraction(0))
                except (TypeError, ValueError):
                    comparisons.append({"other": other, "status": "observed_violation",
                                        "reason": "Nonfinite native score has no finite error budget"})
                    continue
            margin = scores[winner] - scores[other]
            comparisons.append({"other": other, "margin_exact": str(margin), "total_budget_exact": str(allowance),
                                "status": "pass" if margin > allowance else "conservative_rejection"})
        rows.append({"status": _result_status(comparisons), "winner": winner, "comparisons": comparisons,
                     "reason": "Sufficient strict margin; measured native errors cover both endpoints" if errors else
                               "Sufficient strict margin from the real box bound"})
    return rows


def admit(request: AdmissionRequest, registry: Any = None) -> AdmissionRecord:
    """Recompute independent evidence lanes without upgrading their scope."""
    from .artifact import StructuralPolicy
    from .features import ReadoutHead, certify_feature_retention, verify_feature_retention
    from .structured import ExecutionSpec, _execution_layers, _exact_execution, native_execution_receipt
    if not isinstance(request, AdmissionRequest):
        raise ValueError("Admission requires a frozen AdmissionRequest")
    data, registry = request.to_dict(), _registry(registry)
    base, candidate = _artifact(data["parent"]), _artifact(data["candidate"])
    program, task = Program.from_dict(data["program"]), _task(data["task"])
    box, policy = InputBox(**data["domain"]), StructuralPolicy.from_dict(data["structural_policy"])
    rule, budgets = DecisionRule.from_dict(data["decision_rule"]), list(map(rational, data["max_output_drift"]))
    samples = np.asarray(data["witness_inputs"], dtype=np.float64)
    gates: dict[str, Any] = {}
    def gate(name: str, status: str, reason: str | None = None, evidence: Any = None, required: bool = True) -> None:
        gates[name] = {"binding_sha256": request.sha256, "status": status, "required": required,
                       "reason": reason, "evidence": evidence}
    problems = []
    for label, captured, actual in (("parent", data["parent_sha256"], base.sha256),
                                   ("candidate", data["candidate_sha256"], candidate.sha256),
                                   ("program", data["program_sha256"], program.sha256),
                                   ("task", data["task_sha256"], task.sha256),
                                   ("witness", data["witness_sha256"], digest(data["witness_inputs"]))):
        if captured != actual:
            problems.append(label + " identity differs")
    if (canonical_bytes(base.to_dict()) != canonical_bytes(data["parent"])
            or canonical_bytes(candidate.to_dict()) != canonical_bytes(data["candidate"])):
        problems.append("Endpoint bytes use a different numerical representation")
    if not base.compatible(candidate):
        problems.append("Candidate architecture/interface differs from parent")
    if task.architecture != _semantic_artifact(base).architecture or task.base_sha256 != base.sha256 or task.mode != "edit":
        problems.append("Task does not identify this edit parent and architecture")
    try:
        expected_interface = _interface(base, len(box.lower), len(budgets))
        if canonical_bytes(expected_interface) != canonical_bytes(data["interface"]):
            problems.append("Input/output order, tensor order or bias convention differs")
    except ValueError as error:
        problems.append(str(error))
    if not bool(box.contains(samples).all()):
        problems.append("A witness is outside the declared real input domain")
    try:
        applied = apply_program(base, program)
        if applied.sha256 != candidate.sha256 or canonical_bytes(applied.to_dict()) != canonical_bytes(candidate.to_dict()):
            problems.append("Atomic program does not produce the captured candidate bytes")
    except ValueError as error:
        problems.append(str(error))
    gate("binding", "invalid" if problems else "pass", evidence={"issues": problems,
         "no_op": base.sha256 == candidate.sha256, "trusted_expected_binding_required_for_verification": True})
    runnable = not problems
    try:
        adapter = registry.adapter(_semantic_artifact(base))
    except ValueError:
        adapter = None
    if adapter is None:
        gate("semantics", "unsupported", "No semantic adapter registered for this architecture")
    elif (data["adapter"] is None or adapter.descriptor() != data["adapter"]
          or data["adapter"]["implementation_sha256"] is None):
        gate("semantics", "missing_correspondence", "Adapter version, callback bytes or frozen configuration differ")
    elif not adapter.deterministic:
        gate("semantics", "unsupported", "Nondeterministic adapter cannot replay a deterministic admission claim")
    else:
        gate("semantics", "pass", evidence={"adapter": adapter.descriptor(), "implementation_verified": False,
             "scope": "Caller-owned callback correspondence; excludes mutable globals, sandboxing and semantic correctness"})
    if runnable:
        structural = policy.check(base, program)
        gate("structural", "pass" if structural["structural_accepted"] else "observed_violation", evidence=structural)
    else:
        gate("structural", "invalid", "Binding failed; candidate application is not admissible")
    semantic_runnable = runnable and gates["semantics"]["status"] == "pass"
    bounds = None
    if semantic_runnable:
        try:
            certificate = registry.certify(_semantic_artifact(base), _semantic_artifact(candidate), box)
            envelope = certificate
            if (set(envelope) != {"schema", "base_artifact_sha256", "candidate_artifact_sha256", "architecture", "certificate"}
                    or envelope["schema"] != "wsp-adapter-certificate/1"
                    or envelope["base_artifact_sha256"] != _semantic_artifact(base).sha256
                    or envelope["candidate_artifact_sha256"] != _semantic_artifact(candidate).sha256
                    or envelope["architecture"] != task.architecture
                    or not registry.verify_certificate(_semantic_artifact(base), _semantic_artifact(candidate), box, certificate)):
                gate("real_box_drift", "missing_correspondence", "Certificate envelope or independent recomputation differs", certificate)
            else:
                body = envelope["certificate"]
                if (body.get("schema") not in ("wsp-drift-certificate/1", "wsp-sequential-drift/1")
                        or body.get("domain") != box.to_dict()):
                    gate("real_box_drift", "unsupported", "No supported real-bound formula/domain correspondence", certificate)
                else:
                    bounds = [Fraction(x) for x in body["output_bounds"]]
                    if len(bounds) != len(budgets) or any(x < 0 for x in bounds):
                        raise ValueError("Certificate bounds differ from the declared output coordinates")
                    passed = all(bound <= limit for bound, limit in zip(bounds, budgets))
                    gate("real_box_drift", "pass" if passed else "conservative_rejection",
                         None if passed else "A sufficient upper bound exceeds the budget; this is coverage loss, not a counterexample",
                         certificate)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            gate("real_box_drift", "unsupported", str(error))
    else:
        gate("real_box_drift", "invalid" if not runnable else gates["semantics"]["status"], "Binding/semantic prerequisites failed")
    exact_parent, exact_candidate = None, None
    if semantic_runnable:
        try:
            exact_parent = _exact_execution(_execution_layers(base), samples)
            exact_candidate = _exact_execution(_execution_layers(candidate), samples)
            observed = [[str(abs(y-x)) for x, y in zip(a, b)] for a, b in zip(exact_parent, exact_candidate)]
            passed = all(Fraction(x) <= budget for row in observed for x, budget in zip(row, budgets))
            gate("real_witness_scores", "pass" if passed else "observed_violation", evidence={
                "parent_outputs_exact": [[str(x) for x in row] for row in exact_parent],
                "candidate_outputs_exact": [[str(x) for x in row] for row in exact_candidate],
                "absolute_drifts_exact": observed, "scope": "Exact real execution only on the declared witnesses"})
            decisions = [rule.preservation(rule.classify(a), rule.classify(b)) for a, b in zip(exact_parent, exact_candidate)]
            gate("real_witness_decision", _result_status(decisions), evidence={"decisions": decisions,
                 "scope": "Declared real decisions only on the stated witnesses; no all-input decision guarantee"})
        except (ValueError, TypeError, OverflowError) as error:
            gate("real_witness_scores", "unsupported", str(error))
            gate("real_witness_decision", "unsupported", "Exact real decision path is unsupported")
    else:
        status = "invalid" if not runnable else gates["semantics"]["status"]
        gate("real_witness_scores", status, "Binding/semantic prerequisites failed")
        gate("real_witness_decision", status, "Binding/semantic prerequisites failed")
    if exact_parent is not None and bounds is not None:
        margins = _margin_rows(exact_parent, bounds, rule)
        gate("real_witness_margin", _result_status(margins), evidence=margins, required=False)
    else:
        gate("real_witness_margin", "unsupported", "No matching exact witness scores and box bounds", required=False)
    feature = data["feature_retention"]
    if feature is None:
        gate("feature_retention", "unsupported", "No feature-anchor retention claim requested", required=False)
    elif not runnable:
        gate("feature_retention", "invalid", "Binding failed")
    else:
        try:
            head = ReadoutHead(**feature["head"])
            actual = certify_feature_retention(base, candidate, head, feature["anchors"])
            if feature["receipt"] is None or not verify_feature_retention(base, candidate, head, feature["anchors"], feature["receipt"]):
                gate("feature_retention", "missing_correspondence", "Supplied feature receipt does not replay on these endpoints/anchors", actual)
            else:
                residuals = list(map(Fraction, actual["maximum_residuals_exact"]))
                if len(residuals) != len(budgets):
                    raise ValueError("Readout outputs differ from the declared output interface")
                passed = all(x <= budget for x, budget in zip(residuals, budgets))
                gate("feature_retention", "pass" if passed else "observed_violation",
                     "Conditional real head residuals use the declared score budgets; extractor correspondence remains caller-owned", actual)
        except (ValueError, TypeError, KeyError) as error:
            gate("feature_retention", "invalid", str(error))
    execution = data["execution"]
    native = None
    if execution is None:
        gate("native_execution", "unsupported", "No native backend scope requested", required=False)
        gate("native_witness_decision", "unsupported", "No native backend scope requested", required=False)
    elif not runnable or gates["semantics"]["status"] != "pass":
        status = "invalid" if not runnable else gates["semantics"]["status"]
        gate("native_execution", status, "Binding/semantic prerequisites failed")
        gate("native_witness_decision", status, "Binding/semantic prerequisites failed")
    else:
        spec = ExecutionSpec.from_dict(execution["spec"])
        current_scope = _execution_scope(spec)
        if canonical_bytes(current_scope) != canonical_bytes(execution):
            gate("native_execution", "missing_correspondence", "Backend/device/version/arithmetic environment differs from frozen request", current_scope)
            gate("native_witness_decision", "missing_correspondence", "Pinned numerical environment differs")
        elif execution["status"] != "pass":
            gate("native_execution", "unsupported", execution["reason"], execution)
            gate("native_witness_decision", "unsupported", execution["reason"])
        else:
            native = native_execution_receipt(base, candidate, samples, spec, decision=rule, binding_sha256=request.sha256)
            gate("native_execution", native["status"], native["reason"], native)
            decisions = native.get("native_decision_preserved")
            gate("native_witness_decision", _result_status(decisions) if decisions else native["status"],
                 "Observed declared decisions on this backend/witness set", decisions)
    if (native is not None and native.get("exact_real_supported") and exact_parent is not None and bounds is not None
            and "observations" in native and all(not isinstance(value, str)
                for row in native["observations"]["parent"]["native_outputs"] for value in row)):
        observations = native["observations"]
        native_parent = [[rational(x) for x in row] for row in observations["parent"]["native_outputs"]]
        margins = _margin_rows(native_parent, bounds, rule,
                    (observations["parent"]["per_coordinate_errors"], observations["candidate"]["per_coordinate_errors"]))
        gate("native_witness_margin", _result_status(margins),
             "Witness-only sufficient condition; both endpoint arithmetic errors included, no universal kernel proof", margins, required=False)
    else:
        gate("native_witness_margin", "unsupported", "No matching real bounds and finite paired native error accounting", required=False)
    claims = {"structural_accepted": gates["structural"]["status"] == "pass",
              "real_box_score_budget_satisfied": gates["real_box_drift"]["status"] == "pass",
              "real_witness_decision_preserved": rule.kind != "scores_only" and gates["real_witness_decision"]["status"] == "pass",
              "native_witness_decision_preserved": rule.kind != "scores_only" and gates["native_witness_decision"]["status"] == "pass",
              "feature_anchor_budget_satisfied": feature is not None and gates["feature_retention"]["status"] == "pass",
              "all_input_decision_preservation_verified": False, "feature_extractor_verified": False,
              "native_execution_formally_verified": False, "useful_business_behavior_verified": False}
    payload = {"schema": "wsp-integrated-admission/1", "binding_sha256": request.sha256, "request": data,
               "gates": gates, "accepted": all(row["status"] == "pass" for row in gates.values() if row["required"]),
               "claims": claims,
               "scope": "All requested mandatory lanes passed on one frozen request. Real box score bounds, conditional feature residuals and witness/backend decisions have distinct scope; Python and the kernel are not formally verified."}
    return AdmissionRecord(payload | {"sha256": digest(payload)})


def replay_admission(record: AdmissionRecord | dict[str, Any], registry: Any = None,
                     expected_binding_sha256: str | None = None) -> AdmissionRecord:
    """Reconstruct strict request data and independently execute every gate."""
    frozen = record if isinstance(record, AdmissionRecord) else AdmissionRecord.from_dict(record)
    if expected_binding_sha256 is not None and frozen.binding_sha256 != expected_binding_sha256:
        raise ValueError("Admission binding differs from the trusted expected request")
    return admit(AdmissionRequest.from_dict(frozen.to_dict()["request"]), registry)


def verify_admission(record: AdmissionRecord | dict[str, Any], expected_binding_sha256: str,
                     registry: Any = None) -> bool:
    """A new self-consistent digest is not an authorization for a substituted claim."""
    from .workflow import _hash
    try:
        _hash(expected_binding_sha256, "Trusted expected admission binding")
        frozen = record if isinstance(record, AdmissionRecord) else AdmissionRecord.from_dict(record)
        actual = replay_admission(frozen, registry, expected_binding_sha256)
        return canonical_bytes(actual.to_dict()) == canonical_bytes(frozen.to_dict())
    except (ValueError, TypeError, KeyError, OverflowError):
        return False
