"""Data-only proposer I/O; exact endpoints remain with the broker until sealing.

This is a protocol boundary. A separate provider process receives JSON, never this
object, an evaluator closure, or a Candidate. It is not an OS/network sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import secrets
from typing import Any, Callable

import numpy as np

from .artifact import TensorArtifact
from .data_roles import DataRoleRegistry, RegisteredBatch
from .model import Network, canonical_bytes, digest
from .program import apply_program
from .structured import StructuredProgram, lower_structured
from .workflow import (AccessPolicy, BudgetLimits, Candidate, ProgrammerSession,
                       SplitSpec, TaskSpec, _json)


@dataclass(frozen=True)
class WeightView:
    mode: str = "none"
    decimals: int = 3
    coordinates: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in ("none", "exact", "rounded", "subset", "summary"):
            raise ValueError("Unknown parameter-view mode")
        if type(self.decimals) is not int or not 0 <= self.decimals <= 15:
            raise ValueError("Parameter-view decimals must be 0..15")
        if not isinstance(self.coordinates, (tuple, list)):
            raise ValueError("Subset coordinates must be an explicit sequence")
        captured = []
        for tensor, index in self.coordinates:
            if (type(tensor) is not str or not isinstance(index, (tuple, list))
                    or any(type(i) is not int or i < 0 for i in index)):
                raise ValueError("Invalid subset coordinate")
            captured.append((tensor, tuple(index)))
        if len(set(captured)) != len(captured):
            raise ValueError("Duplicate subset coordinate")
        if captured and self.mode != "subset":
            raise ValueError("Coordinates apply only to the subset view")
        object.__setattr__(self, "coordinates", tuple(captured))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "wsp-parameter-view/1", "mode": self.mode,
                "decimals": self.decimals if self.mode in ("rounded", "summary") else None,
                "coordinates": [[t, list(i)] for t, i in self.coordinates]}


class ProposerBroker:
    """Restrict supported routes and retain all attempted requests.

    Submissions receive handles even when endpoint lowering fails. Reporting that
    failure, a no-op, a structural verdict, or an endpoint hash before selection
    would create an exact-parent oracle. This version exposes no candidate
    feedback and no execution route, including for exact-weight conditions.
    """

    def __init__(self, task: TaskSpec, *, base: Network | TensorArtifact | None,
                 view: WeightView = WeightView(), limits: BudgetLimits | None = None,
                 public_context: dict[str, Any] | None = None,
                 splits: tuple[SplitSpec, ...] = (),
                 native_dtypes: dict[str, str] | None = None,
                 max_requests: int = 100, max_request_bytes: int = 1_000_000,
                 data_registry: DataRoleRegistry | None = None):
        if type(view) is not WeightView:
            raise ValueError("Use an immutable WeightView")
        if any(type(n) is not int or n <= 0 for n in (max_requests, max_request_bytes)):
            raise ValueError("Broker request bounds must be positive integers")
        if base is not None:
            for tensor, index in view.coordinates:
                a = base.tensors.get(tensor)
                if a is None or len(index) != a.ndim or any(i >= s for i, s in zip(index, a.shape)):
                    raise ValueError("Declared subset is outside the parent")
        elif view.mode != "none":
            raise ValueError("Creation has no parent parameter view")
        self._base, self._view = base, view
        self._native_dtypes = dict(native_dtypes) if native_dtypes is not None else None
        # Validate baseline representability once; proposal success cannot expose it.
        if base is not None and native_dtypes is not None:
            lower_structured(base, StructuredProgram(base.sha256, [], "Validate baseline"),
                             native_dtypes=native_dtypes, cast="round")
        self._session = ProgrammerSession(task, base=base,
            access=AccessPolicy(weights=view.mode != "none",
                weight_decimals=view.decimals if view.mode == "rounded" else None),
            limits=limits, splits=splits, data_registry=data_registry)
        self._public_context = json.loads(canonical_bytes(_json(public_context or {})))
        self._task = {"task_id": task.task_id, "mode": task.mode,
                      "objective": task.objective, "architecture": task.architecture}
        self._max_requests, self._max_request_bytes = max_requests, max_request_bytes
        self._handles: dict[str, tuple[Candidate | None, dict[str, Any]]] = {}
        self._events: list[dict[str, Any]] = []
        self._sealed = False
        self._selected: str | None = None
        self._evaluated = False

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    @property
    def public_trace(self) -> list[dict[str, Any]]:
        return json.loads(canonical_bytes(self._events))

    def _record(self, request: Any, result: dict[str, Any]) -> dict[str, Any]:
        event = {"schema": "wsp-broker-event/1", "sequence": len(self._events),
                 "previous_sha256": self._events[-1]["sha256"] if self._events else None,
                 "request": request, "response": result}
        self._events.append(event | {"sha256": digest(event)})
        return json.loads(canonical_bytes(result))

    def _submit(self, request: dict[str, Any]) -> dict[str, Any]:
        # Rejected lowering still consumes an attempted proposal. This check
        # depends only on the public submission history, never on endpoint state.
        if len(self._handles) >= self._session.limits.proposals:
            return {"status": "denied"}
        handle = secrets.token_hex(24)
        candidate, private = None, {"submitted": request}
        try:
            if request["op"] == "submit":
                if self._base is None or set(request) != {"op", "commands", "intent"}:
                    raise ValueError("An edit requires commands and intent")
                proposal = StructuredProgram(self._base.sha256, request["commands"], request["intent"])
                lowered = lower_structured(self._base, proposal, native_dtypes=self._native_dtypes,
                                           cast="round")
                endpoint = apply_program(self._base, lowered.program)
                candidate = self._session.propose_artifact(endpoint)
                private.update(program=lowered.program.to_dict(), lowering=lowered.receipt)
            elif request["op"] == "create":
                if self._base is not None or set(request) != {"op", "artifact"}:
                    raise ValueError("Creation requires artifact data and no parent")
                candidate = self._session.propose_artifact(request["artifact"])
            else:
                if set(request) != {"op", "answer"}:
                    raise ValueError("Answers require data")
                candidate = self._session.submit_answer(request["answer"])
            private["status"] = "lowered"
        except (ValueError, KeyError, TypeError, FloatingPointError, OverflowError) as error:
            private.update(status="invalid", error_type=type(error).__name__, error=str(error))
        self._handles[handle] = candidate, private
        return {"status": "submitted", "candidate_handle": handle}

    def request(self, data: Any) -> dict[str, Any]:
        """The entire proposer surface. Unknown routes are denied and logged."""
        try:
            request = _json(data)
            if (type(request) is not dict or type(request.get("op")) is not str
                    or len(canonical_bytes(request)) > self._max_request_bytes):
                raise ValueError("Malformed or oversized request")
        except (ValueError, TypeError):
            return self._record({"op": "malformed"}, {"status": "denied"})
        if self._sealed or len(self._events) >= self._max_requests:
            return self._record(request, {"status": "denied"})
        op = request["op"]
        try:
            if op == "metadata" and set(request) == {"op"}:
                original = self._session.metadata()
                tensors = original["artifact"]["tensors"] if original["artifact"] else None
                result = {"status": "ok", "schema": "wsp-proposer-metadata/1",
                          "task": self._task, "tensors": tensors, "context": self._public_context,
                          "parameter_view": self._view.to_dict(),
                          "candidate_feedback": "none_before_or_after_seal",
                          "submission": "data_only_commands_with_opaque_handles",
                          "execution": "no_broker_forward_or_activation_route",
                          "computation_boundary": "internal_model_computation_unobserved"}
            elif op == "weights" and set(request) <= {"op", "tensor", "indices"}:
                if self._view.mode == "none":
                    raise ValueError("Parameter view disabled")
                tensor, indices = request["tensor"], request.get("indices")
                if self._view.mode == "subset":
                    if not isinstance(indices, list) or not indices or any(
                        not isinstance(i, list) or (tensor, tuple(i)) not in self._view.coordinates
                        for i in indices):
                        raise ValueError("Outside declared subset")
                if self._view.mode == "summary" and indices is not None:
                    raise ValueError("Summary has no coordinate oracle")
                values = self._session.weights(tensor, indices)
                if self._view.mode == "summary":
                    a = np.asarray(values, dtype=np.float64)
                    values = {k: round(float(v), self._view.decimals) for k, v in
                              {"min": a.min(), "max": a.max(), "mean": a.mean(),
                               "mean_abs": np.abs(a).mean()}.items()}
                result = {"status": "ok", "values": values}
            elif op in ("submit", "create", "answer"):
                result = self._submit(request)
            elif op == "seal" and set(request) == {"op", "candidate_handle"}:
                handle = request["candidate_handle"]
                if handle is not None and (type(handle) is not str or handle not in self._handles):
                    raise ValueError("Unknown selection")
                candidate = self._handles[handle][0] if handle is not None else None
                self._session.seal(candidate)
                self._selected, self._sealed = handle, True
                result = {"status": "sealed"}
            else:
                result = {"status": "denied"}
        except (ValueError, KeyError, TypeError, FloatingPointError):
            # No exception message crosses the boundary: it may depend on private state.
            result = {"status": "denied"}
        return self._record(request, result)

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Trusted provider accounting; unavailable as a proposer operation."""
        self._session.record_provider_usage(input_tokens=input_tokens, output_tokens=output_tokens)

    def abstain(self) -> None:
        """Trusted deadline handling; a request cap cannot leave evaluation open."""
        if not self._sealed:
            self._session.seal(None)
            self._sealed = True
            self._record({"op": "harness_deadline_seal"}, {"status": "sealed"})

    def evaluate(self, evaluator: Callable[[Candidate], Any], *, evaluator_id: str,
                 split_name: str = "test") -> dict[str, Any]:
        """Trusted opaque evaluation; callback input freshness is unverified."""
        return self._evaluate(evaluator, evaluator_id=evaluator_id, split_name=split_name)

    def evaluate_registered(self, batch: RegisteredBatch,
                            evaluator: Callable[[Candidate, list[Any]], Any], *,
                            evaluator_id: str, split_name: str = "test") -> dict[str, Any]:
        """Score registered data inside the session guard; keep the audit private."""
        if type(batch) is not RegisteredBatch:
            raise ValueError("Registered evaluation requires a RegisteredBatch")
        return self._evaluate(evaluator, evaluator_id=evaluator_id, split_name=split_name, batch=batch)

    def _evaluate(self, evaluator: Callable[..., Any], *, evaluator_id: str,
                  split_name: str, batch: RegisteredBatch | None = None) -> dict[str, Any]:
        if not self._sealed or self._evaluated:
            raise ValueError("Evaluation needs one sealed, unevaluated selection")
        self._evaluated = True
        if batch is None:
            receipt = self._session.evaluate_heldout(evaluator, evaluator_id=evaluator_id,
                                                     split_name=split_name).to_dict()
        else:
            receipt = self._session.evaluate_registered(batch, evaluator, evaluator_id=evaluator_id,
                                                        split_name=split_name).to_dict()
        if self._selected is not None and self._handles[self._selected][0] is None:
            status = "invalid"
        else:
            status = receipt["status"]
        payload = {"schema": "wsp-broker-audit/1", "status": status,
                   "selected_handle": self._selected, "parameter_view": self._view.to_dict(),
                   "attempted_submissions": len(self._handles),
                   "proposal_limit": self._session.limits.proposals,
                   "attempts": [{"candidate_handle": h, **p,
                                 "candidate": c.to_dict() if c is not None else None}
                                for h, (c, p) in self._handles.items()],
                   "public_trace": self.public_trace, "evaluation": receipt,
                   "enforcement": "JSON_route_allowlist_and_seal;_not_OS_hermeticity",
                   "semantic_no_inference_attested": False}
        return payload | {"sha256": digest(payload)}
