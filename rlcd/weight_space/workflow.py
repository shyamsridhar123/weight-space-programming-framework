"""Generic, auditable weight-space tasks with cooperative access boundaries.

Callbacks are caller-owned Python code. This module is an interface and accounting
boundary, not a sandbox, an evaluator attestation, or a behavioral certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .artifact import TensorArtifact
from .data_roles import DataRoleError, DataRoleRegistry, RegisteredBatch
from .model import Network, canonical_bytes, digest
from .program import Program, apply_program

Artifact = Network | TensorArtifact
JSON = Any


def _json(value: Any) -> JSON:
    """Normalize finite data, without executing arbitrary serialization hooks."""
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numbers must be finite")
        return number
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if type(value) in (list, tuple):
        return [_json(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json(item) for key, item in value.items()}
    raise ValueError(f"Expected finite JSON data, received {type(value).__name__}")


def _bytes(value: Any) -> bytes:
    return canonical_bytes(_json(value))


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be an exact lowercase SHA-256")
    return value


def _architecture(artifact: Artifact) -> str:
    if type(artifact) is Network:
        return "dense-relu-dense/1"
    if type(artifact) is TensorArtifact:
        return artifact.architecture
    raise ValueError("Use an immutable Network or TensorArtifact")


@dataclass(frozen=True, slots=True, init=False)
class TaskSpec:
    task_id: str
    mode: str
    objective: str
    architecture: str
    base_sha256: str | None
    _constraints_bytes: bytes

    def __init__(self, task_id: str, mode: str, objective: str, architecture: str,
                 base_sha256: str | None = None, constraints: dict[str, JSON] | None = None):
        if mode not in ("read", "diagnose", "edit", "create"):
            raise ValueError("Task mode must be read, diagnose, edit, or create")
        if mode == "create":
            if base_sha256 is not None:
                raise ValueError("A create task has no parent")
        else:
            _hash(base_sha256, "Task parent")
        if constraints is not None and type(constraints) is not dict:
            raise ValueError("Constraints must be a finite JSON object")
        for key, value in (("task_id", task_id), ("objective", objective),
                           ("architecture", architecture)):
            object.__setattr__(self, key, _name(value, key))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "base_sha256", base_sha256)
        object.__setattr__(self, "_constraints_bytes", _bytes(constraints or {}))

    @property
    def constraints(self) -> dict[str, JSON]:
        return json.loads(self._constraints_bytes)

    def to_dict(self) -> dict[str, JSON]:
        return {"schema": "wsp-task/1", "task_id": self.task_id, "mode": self.mode,
                "objective": self.objective, "architecture": self.architecture,
                "base_sha256": self.base_sha256, "constraints": self.constraints}

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    metadata: bool = True
    weights: bool = True
    feedback: bool = False
    forward: bool = False
    activations: bool = False
    weight_decimals: int | None = None

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not bool for name in
               ("metadata", "weights", "feedback", "forward", "activations")):
            raise ValueError("Access capabilities must be booleans")
        if self.weight_decimals is not None and (
                type(self.weight_decimals) is not int or not 0 <= self.weight_decimals <= 15):
            raise ValueError("weight_decimals must be None or an integer from 0 through 15")

    def to_dict(self) -> dict[str, JSON]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    tool_calls: int = 0
    weight_values: int = 0
    feedback_calls: int = 0
    feedback_examples: int = 0
    execution_calls: int = 0
    execution_examples: int = 0
    proposals: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if any(type(getattr(self, field.name)) is not int or getattr(self, field.name) < 0
               for field in fields(self)):
            raise ValueError("Budget counters must be nonnegative integers, not booleans")

    def to_dict(self) -> dict[str, int]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class BudgetLimits(BudgetUsage):
    tool_calls: int = 100
    weight_values: int = 10000
    feedback_calls: int = 10
    feedback_examples: int = 1000
    execution_calls: int = 10
    execution_examples: int = 1000
    proposals: int = 10
    input_tokens: int = 100000
    output_tokens: int = 100000


@dataclass(frozen=True, slots=True, init=False)
class SplitSpec:
    name: str
    sample_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]

    def __init__(self, name: str, sample_ids: Iterable[str], lineage_ids: Iterable[str] = ()):
        object.__setattr__(self, "name", _name(name, "Split name"))
        for field, values in (("sample_ids", sample_ids), ("lineage_ids", lineage_ids)):
            if isinstance(values, (str, bytes, dict)):
                raise ValueError(f"{field} must be a sequence of identifiers")
            try:
                identifiers = tuple(values)
            except TypeError:
                raise ValueError(f"{field} must be a sequence of identifiers") from None
            for identifier in identifiers:
                _name(identifier, field)
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"Duplicate {field} within split {name}")
            object.__setattr__(self, field, identifiers)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "sample_ids": list(self.sample_ids),
                "lineage_ids": list(self.lineage_ids)}


@dataclass(frozen=True, slots=True, init=False)
class _Record:
    _payload_bytes: bytes

    def __init__(self, payload: dict[str, JSON]):
        object.__setattr__(self, "_payload_bytes", _bytes(payload))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._payload_bytes).hexdigest()

    def to_dict(self) -> dict[str, JSON]:
        return json.loads(self._payload_bytes) | {"sha256": self.sha256}


class SplitValidation(_Record):
    __slots__ = ()


def validate_splits(splits: Sequence[SplitSpec]) -> SplitValidation:
    """Check declared identifiers; never infer or attest hidden ancestry."""
    names: set[str] = set()
    samples: set[str] = set()
    lineages: set[str] = set()
    summaries = []
    for split in splits:
        if type(split) is not SplitSpec:
            raise ValueError("Splits must contain SplitSpec values")
        if split.name in names:
            raise ValueError("Split names must be unique")
        if samples.intersection(split.sample_ids):
            raise ValueError("Sample identifiers overlap across splits")
        if lineages.intersection(split.lineage_ids):
            raise ValueError("Declared lineage identifiers overlap across splits")
        names.add(split.name)
        samples.update(split.sample_ids)
        lineages.update(split.lineage_ids)
        summaries.append({"name": split.name, "sample_count": len(split.sample_ids),
                          "lineage_count": len(split.lineage_ids),
                          "manifest_sha256": digest(split.to_dict())})
    return SplitValidation({"schema": "wsp-split-validation/1", "splits": summaries,
                            "sample_overlap": False, "declared_lineage_overlap": False,
                            "lineage_scope": "caller_declared_only",
                            "lineage_independence_verified": False})


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Candidate:
    """A session-bound proposal; sha256 is distinct from artifact.sha256."""
    _payload_bytes: bytes
    _artifact: Artifact | None

    def __init__(self, *, session_sha256: str, task_sha256: str, parent_sha256: str | None,
                 proposal_index: int, kind: str, artifact: Artifact | None = None,
                 answer: JSON = None, program_sha256: str | None = None):
        if artifact is not None:
            _architecture(artifact)
        payload = {"schema": "wsp-candidate/1", "session_sha256": session_sha256,
                   "task_sha256": task_sha256, "parent_sha256": parent_sha256,
                   "proposal_index": proposal_index, "kind": kind,
                   "program_sha256": program_sha256,
                   "artifact_sha256": artifact.sha256 if artifact is not None else None,
                   "artifact": artifact.to_dict() if artifact is not None else None,
                   "answer": answer,
                   "no_op": artifact is not None and artifact.sha256 == parent_sha256}
        object.__setattr__(self, "_payload_bytes", _bytes(payload))
        object.__setattr__(self, "_artifact", artifact)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._payload_bytes).hexdigest()

    @property
    def artifact(self) -> Artifact | None:
        return self._artifact

    @property
    def answer(self) -> JSON:
        return json.loads(self._payload_bytes)["answer"]

    @property
    def no_op(self) -> bool:
        return json.loads(self._payload_bytes)["no_op"]

    @property
    def task_sha256(self) -> str:
        return json.loads(self._payload_bytes)["task_sha256"]

    @property
    def parent_sha256(self) -> str | None:
        return json.loads(self._payload_bytes)["parent_sha256"]

    def to_dict(self) -> dict[str, JSON]:
        return json.loads(self._payload_bytes) | {"sha256": self.sha256}


class TraceEvent(_Record):
    __slots__ = ()


class EvaluationReceipt(_Record):
    __slots__ = ()

    @property
    def status(self) -> str:
        return json.loads(self._payload_bytes)["status"]


class WorkflowError(ValueError):
    """An invalid or unavailable workflow action."""


class AccessDenied(WorkflowError):
    pass


class BudgetExceeded(WorkflowError):
    pass


class SessionSealed(WorkflowError):
    pass


def verify_trace(events: Sequence[TraceEvent | Mapping[str, JSON]], *,
                 expected_head: str | None = None) -> bool:
    """Check links and accounting. A trusted head is needed to detect replacement."""
    previous = None
    usage = BudgetUsage().to_dict()
    identity = None
    try:
        for sequence, event in enumerate(events):
            payload = event.to_dict() if isinstance(event, TraceEvent) else dict(event)
            claimed = payload.pop("sha256")
            if (digest(_json(payload)) != claimed or payload["sequence"] != sequence
                    or payload["previous_sha256"] != previous):
                return False
            current_identity = (payload["session_sha256"], payload["task_sha256"])
            if identity is not None and current_identity != identity:
                return False
            if sequence == 0 and payload["action"] != "session_created":
                return False
            charges = payload["charges"]
            if (set(charges) - set(usage) or any(type(v) is not int or v < 0 for v in charges.values())):
                return False
            usage = {key: count + charges.get(key, 0) for key, count in usage.items()}
            if canonical_bytes(usage) != canonical_bytes(payload["usage"]):
                return False
            previous, identity = claimed, current_identity
        return previous is not None and (expected_head is None or previous == expected_head)
    except (ValueError, KeyError, TypeError):
        return False


class ProgrammerSession:
    """Own one immutable task/configuration and a monotonic proposal history.

    The proposer should receive this interface, not private attributes or the
    caller's evaluator closure. Same-process Python cannot enforce that isolation.
    """

    def __init__(self, task: TaskSpec, *, base: Artifact | None = None,
                 access: AccessPolicy | None = None, limits: BudgetLimits | None = None,
                 public_feedback: Callable[[Candidate, list[JSON]], JSON] | None = None,
                 forward: Callable[[Artifact, list[JSON]], JSON] | None = None,
                 activations: Callable[[Artifact, list[JSON]], JSON] | None = None,
                 splits: Sequence[SplitSpec] = (),
                 data_registry: DataRoleRegistry | None = None):
        if type(task) is not TaskSpec:
            raise ValueError("A typed TaskSpec is required")
        if task.mode == "create":
            if base is not None:
                raise ValueError("A create session has no parent")
        elif base is None or _architecture(base) != task.architecture or base.sha256 != task.base_sha256:
            raise ValueError("Session requires the task's exact parent and architecture")
        access = AccessPolicy() if access is None else access
        limits = BudgetLimits() if limits is None else limits
        if type(access) is not AccessPolicy or type(limits) is not BudgetLimits:
            raise ValueError("Use typed, immutable access and budget configuration")
        callbacks = {"feedback": public_feedback, "forward": forward, "activations": activations}
        if any(callback is not None and not callable(callback) for callback in callbacks.values()):
            raise ValueError("Execution adapters must be caller-owned callables")
        if data_registry is not None and type(data_registry) is not DataRoleRegistry:
            raise ValueError("Use a DataRoleRegistry owned by the trusted harness")
        self._task, self._base, self._access, self._limits = task, base, access, limits
        self._callbacks = callbacks
        self._data_registry = data_registry
        self._splits = tuple(splits)
        self._split_validation = validate_splits(self._splits)
        # Hidden constraints and split IDs are committed by hash, not disclosed
        # through public accounting. Their originals remain caller-owned.
        self._configuration_bytes = _bytes({
            "schema": "wsp-session-config/1", "task_sha256": task.sha256,
            "task_id": task.task_id, "mode": task.mode, "objective": task.objective,
            "architecture": task.architecture, "base_sha256": task.base_sha256,
            "constraints_sha256": hashlib.sha256(task._constraints_bytes).hexdigest(),
            "access": access.to_dict(), "limits": limits.to_dict(),
            "callbacks_registered": {name: callback is not None for name, callback in callbacks.items()},
            "splits": self._split_validation.to_dict(),
            "data_roles": {"registry_id": data_registry.registry_id if data_registry is not None else None,
                           "scope": "registered_cooperative" if data_registry is not None else "unverified",
                           "provenance_authenticated": False,
                           "semantic_independence_verified": False},
            "enforcement": "cooperative_same_process_interface",
            "declared_constraints_automatically_enforced": False,
            "execution_accounting": "interface_calls_and_submitted_examples; callback_internal_work_not_observed",
            "token_accounting": "provider_supplied_not_locally_measured"})
        self._usage = BudgetUsage()
        self._trace: list[TraceEvent] = []
        self._candidates: list[Candidate] = []
        self._sealed = False
        self._busy = False
        self._selected: Candidate | None = None
        self._evaluated = False
        self._receipt: EvaluationReceipt | None = None
        self._append("session_created", "succeeded", self.configuration, {}, {})

    @property
    def configuration(self) -> dict[str, JSON]:
        return json.loads(self._configuration_bytes)

    @property
    def sha256(self) -> str:
        """Configuration identity, not a globally unique execution identifier."""
        return hashlib.sha256(self._configuration_bytes).hexdigest()

    @property
    def access(self) -> AccessPolicy:
        return self._access

    @property
    def limits(self) -> BudgetLimits:
        return self._limits

    @property
    def usage(self) -> BudgetUsage:
        return self._usage

    @property
    def trace(self) -> tuple[TraceEvent, ...]:
        return tuple(self._trace)

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(self._candidates)

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    @property
    def receipt(self) -> EvaluationReceipt | None:
        return self._receipt

    def _open(self) -> None:
        if self._sealed:
            raise SessionSealed("The session is sealed; proposer actions are closed")
        if self._busy:
            raise WorkflowError("Session callbacks cannot reenter the session")

    def _append(self, action: str, status: str, request: JSON, result: JSON,
                charges: dict[str, int], *, error: JSON = None,
                requested_charges: dict[str, int] | None = None) -> None:
        self._trace.append(TraceEvent({
            "schema": "wsp-trace-event/1", "sequence": len(self._trace),
            "previous_sha256": self._trace[-1].sha256 if self._trace else None,
            "session_sha256": self.sha256, "task_sha256": self._task.sha256,
            "action": action, "status": status, "request": request, "result": result,
            "charges": charges, "requested_charges": requested_charges or charges,
            "usage": self._usage.to_dict(), "error": error}))

    def _reject(self, action: str, request: JSON, error: Exception,
                *, status: str = "rejected") -> None:
        self._append(action, status, request, None, {},
                     error={"type": type(error).__name__, "message": str(error)})
        raise error

    def _require(self, action: str, capabilities: Sequence[str]) -> None:
        self._open()
        denied = [name for name in capabilities if not getattr(self._access, name)]
        if denied:
            self._reject(action, {"required_capabilities": list(capabilities)},
                         AccessDenied("Access denied: " + ", ".join(denied)), status="denied")

    def _mode(self, action: str, modes: Sequence[str]) -> None:
        self._open()
        if self._task.mode not in modes:
            self._reject(action, {"task_mode": self._task.mode},
                         WorkflowError(f"{action} is unavailable for {self._task.mode} tasks"))

    def _known(self, candidate: Candidate, action: str) -> Candidate:
        if not any(candidate is known for known in self._candidates):
            self._reject(action, {"candidate_type": type(candidate).__name__},
                         WorkflowError("Select an exact candidate produced by this session"))
        return candidate

    def _perform(self, action: str, request: dict[str, JSON], charges: dict[str, int],
                 operation: Callable[[], Any], *, candidate_result: bool = False) -> Any:
        self._open()
        projected = {key: value + charges.get(key, 0) for key, value in self._usage.to_dict().items()}
        exceeded = [key for key, value in projected.items() if value > getattr(self._limits, key)]
        if exceeded:
            error = BudgetExceeded("Budget exceeded: " + ", ".join(exceeded))
            self._append(action, "denied", request, None, {}, requested_charges=charges,
                         error={"type": type(error).__name__, "message": str(error)})
            raise error
        self._usage = BudgetUsage(**projected)
        self._busy = True
        try:
            result = operation()
            if candidate_result:
                if type(result) is not Candidate:
                    raise WorkflowError("Proposal operations must produce a Candidate")
                recorded = {"candidate_sha256": result.sha256,
                            "artifact_sha256": result.artifact.sha256 if result.artifact is not None else None,
                            "no_op": result.no_op}
            else:
                recorded = _json(result)
            self._append(action, "succeeded", request, recorded, charges)
            return result if candidate_result else json.loads(_bytes(recorded))
        except BaseException as error:
            self._append(action, "failed", request, None, charges,
                         error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            self._busy = False

    def metadata(self) -> dict[str, JSON]:
        self._require("metadata", ("metadata",))
        def inspect() -> dict[str, JSON]:
            artifact = None
            if self._base is not None:
                artifact = {"sha256": self._base.sha256, "architecture": _architecture(self._base),
                            "tensors": {name: {"shape": list(values.shape), "dtype": str(values.dtype)}
                                        for name, values in self._base.tensors.items()}}
                if type(self._base) is TensorArtifact:
                    artifact["metadata"] = self._base.metadata
                else:
                    artifact["metadata"] = {"input_names": list(self._base.input_names),
                                            "output_names": list(self._base.output_names)}
            return {"task": self._task.to_dict(), "artifact": artifact,
                    "constraints_status": "declared_context_not_automatically_enforced"}
        return self._perform("metadata", {}, {"tool_calls": 1}, inspect)

    def weights(self, tensor: str, indices: Sequence[Sequence[int]] | None = None) -> JSON:
        self._require("weights", ("weights",))
        request = {"tensor": tensor if isinstance(tensor, str) else None,
                   "weight_decimals": self._access.weight_decimals}
        try:
            if self._base is None or not isinstance(tensor, str) or tensor not in self._base.tensors:
                raise ValueError("Weights require a declared parent tensor")
            values = self._base.tensors[tensor]
            coordinates = None
            if indices is not None:
                if not isinstance(indices, (list, tuple)):
                    raise ValueError("Indices must be an explicit sequence of coordinate tuples")
                coordinates = []
                for index in indices:
                    if not isinstance(index, (list, tuple)):
                        raise ValueError("Each index must be a coordinate tuple")
                    if (len(index) != values.ndim or any(type(i) is not int or i < 0 or i >= size
                                                       for i, size in zip(index, values.shape))):
                        raise ValueError("Weight coordinate is outside its tensor")
                    coordinates.append(tuple(index))
            request["indices"] = coordinates
            count = int(values.size) if coordinates is None else len(coordinates)
        except (ValueError, TypeError) as error:
            self._reject("weights", request, error)
        def inspect() -> JSON:
            result = values.tolist() if coordinates is None else [float(values[index]) for index in coordinates]
            def rounded(value: JSON) -> JSON:
                if isinstance(value, list):
                    return [rounded(item) for item in value]
                return round(value, self._access.weight_decimals) if self._access.weight_decimals is not None else value
            return rounded(result)
        return self._perform("weights", request, {"tool_calls": 1, "weight_values": count}, inspect)

    def _candidate(self, kind: str, *, artifact: Artifact | None = None, answer: JSON = None,
                   program_sha256: str | None = None) -> Candidate:
        candidate = Candidate(session_sha256=self.sha256, task_sha256=self._task.sha256,
                              parent_sha256=self._task.base_sha256, proposal_index=self._usage.proposals,
                              kind=kind, artifact=artifact, answer=answer, program_sha256=program_sha256)
        self._candidates.append(candidate)
        return candidate

    def propose_program(self, program: Program | dict[str, JSON]) -> Candidate:
        self._mode("propose_program", ("edit",))
        self._require("propose_program", ("weights", "metadata"))
        if self._access.weight_decimals is not None:
            self._reject("propose_program", {"weight_decimals": self._access.weight_decimals},
                         AccessDenied("A realized program exposes exact weights; rounded-only access cannot return it"),
                         status="denied")
        request: dict[str, JSON] = {"parser": "typed_program_or_wsp-program/1",
                                   "proposal_type": type(program).__name__}
        def propose() -> Candidate:
            parsed = program
            if type(program) is dict:
                submitted = json.loads(_bytes(program))
                request["submitted_data"] = submitted
                request["submitted_sha256"] = digest(submitted)
                parsed = Program.from_dict(submitted)
            if type(parsed) is not Program:
                raise ValueError("Supply a Program or its serialized data")
            request["program"] = parsed.to_dict()
            request["program_sha256"] = parsed.sha256
            realized = apply_program(self._base, parsed)
            return self._candidate("program", artifact=realized, program_sha256=parsed.sha256)
        return self._perform("propose_program", request,
                             {"tool_calls": 1, "proposals": 1,
                              "weight_values": self._base.parameter_count}, propose, candidate_result=True)

    def propose_artifact(self, artifact: Artifact | dict[str, JSON]) -> Candidate:
        self._mode("propose_artifact", ("edit", "create"))
        request: dict[str, JSON] = {"parser": "typed_artifact_or_declared_schema",
                                   "proposal_type": type(artifact).__name__}
        def propose() -> Candidate:
            parsed = artifact
            if type(parsed) is dict:
                data = json.loads(_bytes(parsed))
                request["submitted_data"] = data
                request["submitted_sha256"] = digest(data)
                parsed = (Network.from_dict(data) if data.get("schema") == "wsp-network/1"
                          else TensorArtifact.from_dict(data))
            architecture = _architecture(parsed)
            request["artifact_sha256"] = parsed.sha256
            if architecture != self._task.architecture:
                raise ValueError("Candidate architecture differs from the declared task")
            if self._base is not None and not self._base.compatible(parsed):
                raise ValueError("An edit must preserve the parent's architecture, tensor shapes, and semantic metadata")
            return self._candidate("artifact", artifact=parsed)
        return self._perform("propose_artifact", request, {"tool_calls": 1, "proposals": 1},
                             propose, candidate_result=True)

    def submit_answer(self, answer: JSON) -> Candidate:
        self._mode("submit_answer", ("read", "diagnose"))
        return self._perform("submit_answer", {}, {"tool_calls": 1, "proposals": 1},
                             lambda: self._candidate("answer", answer=_json(answer)), candidate_result=True)

    def _batch(self, action: str, examples: Any) -> int:
        try:
            if type(examples) is RegisteredBatch:
                return len(examples.materialize())
            if not isinstance(examples, (list, tuple, np.ndarray)) or len(examples) == 0:
                raise ValueError("Examples must be a nonempty batch list, tuple, or NumPy array")
            return len(examples)
        except (TypeError, ValueError, AttributeError):
            self._reject(action, {"examples_type": type(examples).__name__},
                         ValueError("Examples must be a nonempty batch list, tuple, or NumPy array"))

    def _development_examples(self, action: str, examples: Any, request: dict[str, JSON]) -> list[JSON]:
        registered = type(examples) is RegisteredBatch
        captured = json.loads(_bytes(examples.materialize() if registered else examples))
        if registered and self._data_registry is None:
            raise WorkflowError("Registered batches require a configured data registry")
        if self._data_registry is not None:
            try:
                request["data_role_decision"] = self._data_registry.consume(
                    examples if registered else captured, reason=action)
            except DataRoleError as failure:
                request["data_role_decision"] = failure.decision
                raise
        request["examples"] = captured
        return json.loads(_bytes(captured))

    def feedback(self, candidate: Candidate, examples: Any) -> JSON:
        self._require("feedback", ("feedback",))
        self._known(candidate, "feedback")
        count = self._batch("feedback", examples)
        request: dict[str, JSON] = {"candidate_sha256": candidate.sha256, "example_count": count}
        def execute() -> JSON:
            callback = self._callbacks["feedback"]
            if callback is None:
                raise WorkflowError("No public feedback callback is registered")
            return callback(candidate, self._development_examples("feedback", examples, request))
        return self._perform("feedback", request, {"tool_calls": 1, "feedback_calls": 1,
                                                   "feedback_examples": count}, execute)

    def _execute(self, action: str, examples: Any, candidate: Candidate | None) -> JSON:
        self._require(action, (action,))
        artifact = self._base if candidate is None else self._known(candidate, action).artifact
        if artifact is None:
            self._reject(action, {}, WorkflowError("Execution requires a parent or an artifact candidate"))
        count = self._batch(action, examples)
        request: dict[str, JSON] = {"candidate_sha256": candidate.sha256 if candidate is not None else None,
                                   "artifact_sha256": artifact.sha256, "example_count": count}
        def execute() -> JSON:
            callback = self._callbacks[action]
            if callback is None:
                raise WorkflowError(f"No {action} callback is registered")
            return callback(artifact, self._development_examples(action, examples, request))
        return self._perform(action, request, {"tool_calls": 1, "execution_calls": 1,
                                              "execution_examples": count}, execute)

    def forward(self, examples: Any, *, candidate: Candidate | None = None) -> JSON:
        return self._execute("forward", examples, candidate)

    def activations(self, examples: Any, *, candidate: Candidate | None = None) -> JSON:
        return self._execute("activations", examples, candidate)

    def record_provider_usage(self, *, input_tokens: int, output_tokens: int) -> JSON:
        self._open()
        request = {"accounting": "provider_supplied_not_locally_measured"}
        if any(type(value) is not int or value < 0 for value in (input_tokens, output_tokens)):
            self._reject("provider_usage", request, ValueError("Token counts must be nonnegative integers"))
        request.update(input_tokens=input_tokens, output_tokens=output_tokens)
        return self._perform("provider_usage", request,
                             {"input_tokens": input_tokens, "output_tokens": output_tokens}, lambda: request)

    def seal(self, candidate: Candidate | None = None) -> Candidate | None:
        self._open()
        if candidate is not None:
            self._known(candidate, "seal")
        self._selected = candidate
        self._sealed = True
        self._append("seal", "succeeded", {"candidate_sha256": candidate.sha256 if candidate is not None else None},
                     {"selection": "candidate" if candidate is not None else "abstained"}, {})
        return candidate

    def evaluate_heldout(self, evaluator: Callable[[Candidate], JSON], *, evaluator_id: str,
                         split_name: str = "test") -> EvaluationReceipt:
        """Evaluate an opaque callback once; its input freshness is unverified."""
        return self._evaluate(evaluator, evaluator_id=evaluator_id, split_name=split_name)

    def evaluate_registered(self, batch: RegisteredBatch,
                            evaluator: Callable[[Candidate, list[JSON]], JSON], *,
                            evaluator_id: str, split_name: str = "test") -> EvaluationReceipt:
        """Bind the frozen selection and durably authorize cooperative scoring."""
        if type(batch) is not RegisteredBatch or self._data_registry is None:
            raise ValueError("Registered evaluation requires a registered batch and configured data registry")
        return self._evaluate(evaluator, evaluator_id=evaluator_id, split_name=split_name, batch=batch)

    def _evaluate(self, evaluator: Callable[..., JSON], *, evaluator_id: str,
                  split_name: str, batch: RegisteredBatch | None = None) -> EvaluationReceipt:
        if not self._sealed:
            raise WorkflowError("Seal the selected candidate before heldout evaluation")
        if self._evaluated:
            raise SessionSealed("Heldout evaluation was already attempted")
        _name(evaluator_id, "Evaluator ID")
        _name(split_name, "Split name")
        if not callable(evaluator):
            raise ValueError("The heldout evaluator must be a caller-owned callable")
        if self._splits and split_name not in {split.name for split in self._splits}:
            raise ValueError("The heldout split is absent from the declared provenance")
        self._evaluated = True
        result, error = None, None
        request = {"evaluator_id": evaluator_id, "split_name": split_name,
                   "candidate_sha256": self._selected.sha256 if self._selected is not None else None}
        freshness = {"status": "unverified", "scope": "opaque_evaluator_inputs",
                     "registry_id": None, "manifest_sha256": None, "decision": None,
                     "provenance_authenticated": False, "semantic_independence_verified": False,
                     "scientific_freshness_verified": False}
        if batch is not None:
            freshness.update(scope="registered_cooperative", registry_id=self._data_registry.registry_id)
        status = "abstained" if self._selected is None else "succeeded"
        interrupted: BaseException | None = None
        evaluator_calls, evaluated_examples = 0, 0 if batch is not None or self._selected is None else None
        def score(materialized: list[JSON] | None = None) -> JSON:
            nonlocal evaluator_calls, evaluated_examples
            evaluator_calls += 1
            if batch is not None:
                evaluated_examples = len(materialized)
                return _json(evaluator(self._selected, materialized))
            return _json(evaluator(self._selected))
        if self._selected is not None:
            try:
                if batch is None:
                    result = score()
                else:
                    binding = {"schema": "wsp-session-evaluation-binding/1",
                               "protocol_sha256": self.sha256, "task_sha256": self._task.sha256,
                               "parent_sha256": self._task.base_sha256,
                               "candidate_sha256": self._selected.sha256,
                               "selection_sha256": digest({"selected": self._selected.to_dict(),
                                   "candidates": [candidate.to_dict() for candidate in self._candidates],
                                   "seal_event_sha256": self._trace[-1].sha256}),
                               "evaluator_id": evaluator_id, "split_name": split_name}
                    result, decision = self._data_registry.score(batch, score, binding=binding)
                    freshness.update(status=decision["verdict"], decision=decision,
                                     manifest_sha256=decision["manifest_sha256"])
            except BaseException as failure:
                decision = failure.decision if isinstance(failure, DataRoleError) else (
                    getattr(failure, "data_role_decision", None) if batch is not None else None)
                if decision is not None:
                    freshness.update(status=decision["verdict"], decision=decision,
                                     manifest_sha256=decision["manifest_sha256"])
                status = "rejected" if batch is not None and evaluator_calls == 0 else "failed"
                # Exception messages can contain private examples. They remain
                # with the evaluator owner instead of crossing this interface.
                error = {"type": type(failure).__name__, "message":
                         "Registered evaluation rejected before scoring" if status == "rejected" else
                         "Heldout evaluator failed; details remain caller-owned"}
                if not isinstance(failure, Exception):
                    interrupted = failure
        request["freshness"] = freshness
        self._append("heldout_evaluation", status, request, result, {}, error=error)
        self._receipt = EvaluationReceipt({
            "schema": "wsp-evaluation-receipt/1", "status": status,
            "session_sha256": self.sha256, "task_sha256": self._task.sha256,
            "parent_sha256": self._task.base_sha256,
            "configuration": self.configuration,
            "candidate_sha256": self._selected.sha256 if self._selected is not None else None,
            "candidate": self._selected.to_dict() if self._selected is not None else None,
            "candidates": [candidate.to_dict() for candidate in self._candidates],
            "usage": self._usage.to_dict(), "evaluator_id": evaluator_id,
            "split_name": split_name, "result": result, "error": error,
            "trace": [event.to_dict() for event in self._trace], "trace_head_sha256": self._trace[-1].sha256,
            "heldout_input_handling": "registered_batch_materialized_inside_cooperative_guard" if batch is not None else
                                     "caller_owned_closure_not_passed_to_proposer_tools",
            "freshness": freshness,
            "evaluator_output_leakage_checked": False,
            "heldout_accounting": {"evaluator_calls": evaluator_calls,
                                   "evaluated_examples": evaluated_examples, "scope": "outside_proposer_budget"},
            "evaluator_attested": False, "behavioral_certificate": None})
        if interrupted is not None:
            raise interrupted
        return self._receipt


@dataclass(frozen=True, slots=True, init=False)
class ReplayProvider:
    """A bounded data script, not a live model or evidence of model ability."""
    _actions_bytes: bytes

    def __init__(self, actions: Sequence[dict[str, JSON]]):
        if not isinstance(actions, (list, tuple)) or len(actions) > 10000:
            raise ValueError("Replay requires at most 10000 explicit actions")
        data = _json(actions)
        if any(type(action) is not dict or type(action.get("op")) is not str for action in data):
            raise ValueError("Every replay action must be a JSON object with an op")
        object.__setattr__(self, "_actions_bytes", _bytes(data))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._actions_bytes).hexdigest()

    def run(self, session: ProgrammerSession) -> tuple[dict[str, JSON], ...]:
        session._open()
        actions = json.loads(self._actions_bytes)
        session._append("replay_started", "succeeded",
                        {"provider_kind": "scripted_replay", "actions_sha256": self.sha256,
                         "action_count": len(actions), "model_ability_measured": False}, None, {})
        candidates: list[Candidate] = []
        outcomes = []
        for index, action in enumerate(actions):
            op = action["op"]
            try:
                if op == "metadata":
                    result = session.metadata()
                elif op == "weights":
                    result = session.weights(action["tensor"], action.get("indices"))
                elif op == "propose_program":
                    result = session.propose_program(action["program"])
                elif op == "propose_artifact":
                    result = session.propose_artifact(action["artifact"])
                elif op == "submit_answer":
                    result = session.submit_answer(action["answer"])
                elif op == "provider_usage":
                    result = session.record_provider_usage(input_tokens=action["input_tokens"],
                                                           output_tokens=action["output_tokens"])
                elif op in ("feedback", "forward", "activations", "seal"):
                    selection = action.get("candidate")
                    if selection is not None and (type(selection) is not int or selection < 0 or selection >= len(candidates)):
                        raise ValueError("Replay candidate must index an earlier successful proposal")
                    candidate = candidates[selection] if selection is not None else None
                    if op == "feedback":
                        result = session.feedback(candidate, action["examples"])
                    elif op == "forward":
                        result = session.forward(action["examples"], candidate=candidate)
                    elif op == "activations":
                        result = session.activations(action["examples"], candidate=candidate)
                    else:
                        result = session.seal(candidate)
                else:
                    raise ValueError("Unknown replay operation")
                if isinstance(result, Candidate):
                    if op != "seal":
                        candidates.append(result)
                    result = {"candidate_sha256": result.sha256}
                outcomes.append({"index": index, "op": op, "status": "succeeded", "result": result})
            except Exception as error:
                failure = {"index": index, "op": op, "status": "failed",
                           "error": {"type": type(error).__name__, "message": str(error)}}
                outcomes.append(failure)
                if not session.is_sealed:
                    session._append("replay_action_failed", "failed", {"index": index, "op": op},
                                    None, {}, error=failure["error"])
                else:
                    break
            if session.is_sealed:
                break
        return tuple(outcomes)
