"""Prospective diagnosis-first selection over generic immutable tensor artifacts.

This lane supersedes the prospective rubric's joint diagnosis/repair submission
rule only for new runs using this module. Frozen protocols and outcomes remain
unchanged. A proposer receives JSON query access, never this trusted coordinator,
a broker object, an endpoint, registered evaluation data, or an evaluator.

The two callbacks select separately at the interface boundary. Their private
reasoning, provenance, storage hardware and causal usefulness are not attested.
Run ``python -m rlcd.weight_space.staged_diagnosis`` for the nontraining guard.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any, Callable
import uuid

from .broker import ProposerBroker, WeightView
from .data_roles import DataRoleRegistry, RegisteredBatch
from .model import canonical_bytes, digest, finite_binary64
from .workflow import (Artifact, BudgetLimits, Candidate, ProgrammerSession,
                       TaskSpec, WorkflowError, _architecture, _json, verify_trace)


class StagedDiagnosisRepair:
    """One diagnosis selection, one later repair selection, one evaluation attempt.

    A diagnosis has exactly ``location``, ``mechanism``, ``predicted_effect``,
    ``confidence``, ``uncertainty`` and ``abstain``. The three explanatory fields
    are nonblank text; confidence is a finite number in [0, 1], not a boolean.
    Location is ``{"tensor": name, "index_prefix": [integer, ...]}`` within the
    captured tensor, or null for an explicit abstention. An empty prefix denotes
    the whole tensor, including a scalar. No repair commands belong here.

    The repair callback returns exactly ``{"commands": [...], "intent": text}``
    for existing structured lowering, or None to abstain. An invalid endpoint
    still closes repair selection. Receipts are private, exclusive JSON files;
    the caller must keep their directory and the shared registry under its control.
    """

    def __init__(self, run_dir: str | Path, *, task_id: str, objective: str,
                 base: Artifact, registry: DataRoleRegistry,
                 view: WeightView = WeightView(), limits: BudgetLimits | None = None,
                 native_dtypes: dict[str, str] | None = None):
        if type(registry) is not DataRoleRegistry or type(view) is not WeightView:
            raise ValueError("Use a trusted DataRoleRegistry and immutable WeightView")
        limits = BudgetLimits(proposals=1) if limits is None else limits
        if type(limits) is not BudgetLimits:
            raise ValueError("Use immutable BudgetLimits")
        architecture = _architecture(base)
        TaskSpec(task_id, "diagnose", objective, architecture, base.sha256)
        self._base, self._registry, self._view, self._limits = base, registry, view, limits
        self._native_dtypes = _json(native_dtypes) if native_dtypes is not None else None
        self._directory = Path(run_dir).resolve()
        self._directory.mkdir(parents=True, exist_ok=False)
        # ponytail: one lock serializes this single run; parallelize separate runs.
        self._lock = threading.RLock()
        self._started = {"diagnosis": False, "repair": False, "evaluation": False}
        self._invalidated = False
        self._records: dict[str, bytes] = {}
        self._protocol = self._write("protocol", {
            "schema": "wsp-staged-diagnosis-protocol/1", "run_id": uuid.uuid4().hex,
            "task_id": task_id, "objective": objective, "architecture": architecture,
            "parent_sha256": base.sha256, "registry_id": registry.registry_id,
            "parameter_view": view.to_dict(), "limits": limits.to_dict(),
            "native_dtypes": self._native_dtypes,
            "diagnosis_fields": ["location", "mechanism", "predicted_effect", "confidence", "uncertainty", "abstain"],
            "repair_fields": ["commands", "intent"],
            "selection_rule": "sealed_durable_diagnosis_before_separate_repair_session",
            "intervention_gate": "both_exact_durable_terminal_selections_verify_before_registered_authorization",
            "supersedes_for_this_lane_only": "single_diagnosis_plus_intervention_submission",
            "frozen_joint_selection_protocols_unchanged": True,
            "enforcement": "cooperative_JSON_interface_and_exact_receipt_binding",
            "causal_usefulness_established": False})
        protocol_sha256 = json.loads(self._protocol)["sha256"]
        task = TaskSpec(task_id + "/diagnosis", "diagnose", objective, architecture,
                        base.sha256, {"staged_protocol_sha256": protocol_sha256,
                                      "selection_rule": "diagnosis_only"})
        self._tasks = {"diagnosis": task}
        self._brokers = {"diagnosis": self._broker(task, {"stage": "diagnosis"})}

    def _broker(self, task: TaskSpec, context: dict[str, Any]) -> ProposerBroker:
        return ProposerBroker(task, base=self._base, view=self._view, limits=self._limits,
                              public_context=context, native_dtypes=self._native_dtypes,
                              data_registry=self._registry)

    def _write(self, name: str, body: dict[str, Any]) -> bytes:
        raw = canonical_bytes(body | {"sha256": digest(body)}) + b"\n"
        path = self._directory / (name + ".json")
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if path.read_bytes() != raw:
            raise WorkflowError("Durable receipt verification failed")
        return raw

    def _query(self, broker: ProposerBroker, data: Any) -> dict[str, Any]:
        with self._lock:
            if broker.is_sealed:
                return {"status": "denied"}  # Keep the terminal trace immutable.
            if (type(data) is not dict or type(data.get("op")) is not str
                    or data["op"] not in ("metadata", "weights")):
                return broker.request({"op": "stage_denied"})
            return broker.request(data)

    def _diagnosis(self, value: Any) -> dict[str, Any]:
        data = _json(value)
        if type(data) is not dict or set(data) != {
                "location", "mechanism", "predicted_effect", "confidence", "uncertainty", "abstain"}:
            raise WorkflowError("Diagnosis requires the diagnosis-only schema")
        if any(type(data[key]) is not str or not data[key].strip()
               for key in ("mechanism", "predicted_effect", "uncertainty")):
            raise WorkflowError("Diagnosis explanations must be nonblank text")
        if type(data["abstain"]) is not bool or not 0 <= finite_binary64(data["confidence"]) <= 1:
            raise WorkflowError("Diagnosis requires finite confidence and explicit abstention")
        location = data["location"]
        if data["abstain"]:
            if location is not None:
                raise WorkflowError("An abstention must not assert a location")
        else:
            if (type(location) is not dict or set(location) != {"tensor", "index_prefix"}
                    or type(location["tensor"]) is not str
                    or location["tensor"] not in self._base.tensors
                    or type(location["index_prefix"]) is not list):
                raise WorkflowError("Diagnosis location must name a captured tensor")
            prefix, shape = location["index_prefix"], self._base.tensors[location["tensor"]].shape
            if len(prefix) > len(shape) or any(type(i) is not int or not 0 <= i < n
                                               for i, n in zip(prefix, shape)):
                raise WorkflowError("Diagnosis location is outside the tensor")
        return data

    def _snapshot(self, stage: str) -> dict[str, Any]:
        broker = self._brokers[stage]
        session: ProgrammerSession = broker._session
        selected = broker._handles.get(broker._selected)
        candidate = selected[0] if selected is not None else None
        if (not broker.is_sealed or not session.is_sealed
                or session._task is not self._tasks[stage] or session._selected is not candidate
                or candidate is not None and not any(c is candidate for c in session.candidates)):
            raise WorkflowError("The exact terminal session selection is required")
        # Existing broker internals own the private selection; no endpoint is
        # added to its public request surface to obtain this harness-only snapshot.
        return {"schema": "wsp-staged-selection/1", "stage": stage,
                "protocol_sha256": json.loads(self._protocol)["sha256"],
                "task": self._tasks[stage].to_dict(), "task_sha256": self._tasks[stage].sha256,
                "session_sha256": session.sha256, "configuration": session.configuration,
                "selected_handle": broker._selected,
                "candidate": candidate.to_dict() if candidate is not None else None,
                "submission": selected[1] if selected is not None else None,
                "candidates": [c.to_dict() for c in session.candidates],
                "trace": [e.to_dict() for e in session.trace],
                "trace_head_sha256": session.trace[-1].sha256, "broker_trace": broker.public_trace}

    def _verify(self, stage: str) -> dict[str, Any]:
        if self._invalidated or stage not in self._records:
            raise WorkflowError("Verified durable terminal stages are required")
        try:
            if ((self._directory / "protocol.json").read_bytes() != self._protocol
                    or (self._directory / (stage + ".json")).read_bytes() != self._records[stage]):
                raise ValueError("receipt_replaced")
            document = json.loads(self._records[stage])
            body = {key: value for key, value in document.items() if key != "sha256"}
            if (digest(body) != document["sha256"]
                    or canonical_bytes(body) != canonical_bytes(self._snapshot(stage))
                    or not verify_trace(body["trace"], expected_head=body["trace_head_sha256"])
                    or body["trace"][-1]["action"] != "seal"):
                raise ValueError("terminal_selection_mismatch")
            return document
        except (OSError, ValueError, KeyError, TypeError):
            self._invalidated = True
            raise WorkflowError("Terminal evidence changed; the run is invalidated") from None

    def _finish(self, stage: str, submission: dict[str, Any] | None) -> dict[str, Any]:
        broker = self._brokers[stage]
        handle = broker.request(submission).get("candidate_handle") if submission is not None else None
        broker.request({"op": "seal", "candidate_handle": handle})
        self._records[stage] = self._write(stage, self._snapshot(stage))
        return self._verify(stage)

    def select_diagnosis(self, selector: Callable[[Callable[..., Any]], Any]) -> dict[str, Any]:
        with self._lock:
            if self._started["diagnosis"] or not callable(selector):
                raise WorkflowError("Diagnosis selection is available once")
            self._started["diagnosis"] = True
            broker = self._brokers["diagnosis"]
            try:
                answer = self._diagnosis(selector(lambda data: self._query(broker, data)))
                return self._finish("diagnosis", {"op": "answer", "answer": answer})
            finally:
                broker.abstain()  # A failed selection cannot leave its proposer open.

    def select_repair(self, selector: Callable[[dict[str, Any], Callable[..., Any]], Any]) -> dict[str, Any]:
        with self._lock:
            if self._started["repair"] or not callable(selector):
                raise WorkflowError("Repair selection is available once")
            diagnosis = self._verify("diagnosis")
            candidate = diagnosis["candidate"]
            if candidate is None or self._diagnosis(candidate["answer"])["abstain"]:
                raise WorkflowError("A nonabstaining sealed diagnosis is required")
            self._started["repair"] = True
            binding = {"staged_protocol_sha256": diagnosis["protocol_sha256"],
                       "diagnosis_receipt_sha256": diagnosis["sha256"],
                       "diagnosis_candidate_sha256": candidate["sha256"],
                       "diagnosis_task_sha256": diagnosis["task_sha256"],
                       "diagnosis_session_sha256": diagnosis["session_sha256"],
                       "diagnosis_seal_sha256": diagnosis["trace_head_sha256"],
                       "selection_rule": "separate_repair_after_verified_durable_diagnosis"}
            task = TaskSpec(self._tasks["diagnosis"].task_id + "/repair", "edit",
                            self._tasks["diagnosis"].objective, _architecture(self._base),
                            self._base.sha256, binding)
            self._tasks["repair"] = task
            broker = self._broker(task, {"stage": "repair", "diagnosis": candidate["answer"],
                                         "diagnosis_receipt_sha256": diagnosis["sha256"]})
            self._brokers["repair"] = broker
            try:
                repair = _json(selector(candidate["answer"], lambda data: self._query(broker, data)))
                if repair is not None and (type(repair) is not dict or set(repair) != {"commands", "intent"}):
                    raise WorkflowError("Repair selection requires structured commands or abstention")
                return self._finish("repair", {"op": "submit", **repair} if repair is not None else None)
            finally:
                broker.abstain()

    def evaluate_registered(self, batch: RegisteredBatch,
                            evaluator: Callable[[dict[str, Any], Candidate, list[Any]], Any], *,
                            evaluator_id: str, split_name: str = "test") -> dict[str, Any]:
        with self._lock:
            if self._started["evaluation"]:
                raise WorkflowError("Evaluation was already attempted")
            diagnosis, repair = self._verify("diagnosis"), self._verify("repair")
            if (diagnosis["candidate"] is None or self._diagnosis(diagnosis["candidate"]["answer"])["abstain"]
                    or repair["candidate"] is None or repair["candidate"]["artifact"] is None
                    or repair["task"]["constraints"]["diagnosis_receipt_sha256"] != diagnosis["sha256"]):
                raise WorkflowError("Both stages must contain the exact bound nonabstaining selections")
            if type(batch) is not RegisteredBatch or not callable(evaluator):
                raise ValueError("Use registered data and a trusted evaluator")
            self._started["evaluation"] = True

            def score(candidate: Candidate, rows: list[Any]) -> Any:
                self._verify("diagnosis")
                self._verify("repair")
                return evaluator(json.loads(self._records["diagnosis"])["candidate"]["answer"], candidate, rows)

            audit = self._brokers["repair"].evaluate_registered(batch, score, evaluator_id=evaluator_id,
                                                                split_name=split_name)
            return json.loads(self._write("evaluation", {
                "schema": "wsp-staged-evaluation/1", "protocol_sha256": diagnosis["protocol_sha256"],
                "diagnosis_receipt_sha256": diagnosis["sha256"], "repair_receipt_sha256": repair["sha256"],
                "evaluation": audit, "causal_usefulness_established": False}))


def _guard_check() -> None:
    """Exercise ordering and rejection using literal synthetic data, not a study."""
    from tempfile import TemporaryDirectory
    from .artifact import TensorArtifact

    def rejected(action: Callable[[], Any]) -> None:
        try:
            action()
        except (ValueError, FileExistsError):
            return
        raise AssertionError("Forbidden staged action was accepted")

    base = TensorArtifact("literal-guard/1", {"layer.weight": [[1.0, 2.0]], "buffer": 0.0})
    answer = {"location": {"tensor": "layer.weight", "index_prefix": [0]},
              "mechanism": "Synthetic row offset", "predicted_effect": "Undo the literal offset",
              "confidence": 0.5, "uncertainty": "No causal evidence; guard-only fixture", "abstain": False}
    with TemporaryDirectory(prefix="wsp-staged-guard-") as directory:
        root = Path(directory)
        registry = DataRoleRegistry(root / "roles.sqlite")
        batch = registry.register_batch([{"guard": "fresh literal row"}], source_id="guard", family_id="guard")

        def lane(name: str) -> StagedDiagnosisRepair:
            return StagedDiagnosisRepair(root / name, task_id=name, objective="Check staged guards only",
                                         base=base, registry=registry, view=WeightView("exact"))

        for index, bad in enumerate((answer | {"commands": []}, answer | {"confidence": True},
                                     answer | {"confidence": float("nan")},
                                     answer | {"location": {"tensor": "layer.weight", "index_prefix": [1]}})):
            invalid = lane("invalid-" + str(index))
            rejected(lambda: invalid.select_diagnosis(lambda query: bad))
            rejected(lambda: invalid.select_repair(lambda diagnosis, query: None))
            assert "repair" not in invalid._brokers

        selected = lane("ordered")
        calls: list[str] = []
        rejected(lambda: selected.select_repair(lambda diagnosis, query: calls.append("premature")))
        rejected(lambda: selected.evaluate_registered(batch, lambda *args: calls.append("premature"), evaluator_id="guard"))
        retained_queries: list[Callable[..., Any]] = []

        def diagnose(query: Callable[..., Any]) -> dict[str, Any]:
            assert query({"op": "submit", "commands": [], "intent": "premature"})["status"] == "denied"
            assert query({"op": "metadata"})["task"]["mode"] == "diagnose"
            retained_queries.append(query)
            calls.append("diagnosis")
            return answer

        diagnosis = selected.select_diagnosis(diagnose)
        rejected(lambda: selected.select_diagnosis(diagnose))
        rejected(lambda: selected.evaluate_registered(batch, lambda *args: calls.append("premature"), evaluator_id="guard"))

        def repair(frozen: dict[str, Any], query: Callable[..., Any]) -> dict[str, Any]:
            assert (root / "ordered" / "diagnosis.json").read_bytes() == selected._records["diagnosis"]
            assert query({"op": "metadata"})["context"]["diagnosis_receipt_sha256"] == diagnosis["sha256"]
            assert frozen == answer
            frozen["mechanism"] = "Mutating this copy must not replace diagnosis"
            calls.append("repair")
            return {"commands": [{"op": "add", "tensor": "layer.weight", "values": [[-1.0, -1.0]]}],
                    "intent": "Literal guard repair"}

        repaired = selected.select_repair(repair)
        rejected(lambda: selected.select_repair(repair))
        assert retained_queries[0]({"op": "weights", "tensor": "layer.weight"})["status"] == "denied"

        def evaluate(frozen: dict[str, Any], candidate: Candidate, rows: list[Any]) -> dict[str, Any]:
            assert frozen == answer and rows == [{"guard": "fresh literal row"}]
            assert candidate.artifact.tensors["layer.weight"].tolist() == [[0.0, 1.0]]
            assert repaired["task"]["constraints"]["diagnosis_receipt_sha256"] == diagnosis["sha256"]
            calls.append("evaluation")
            return {"guard": "ordered intervention"}

        result = selected.evaluate_registered(batch, evaluate, evaluator_id="literal-staged-guard")
        assert result["evaluation"]["status"] == "succeeded" and calls == ["diagnosis", "repair", "evaluation"]
        decision = result["evaluation"]["evaluation"]["freshness"]["decision"]
        attempt = registry.get_attempt(decision["attempt_id"])
        assert attempt["status"] == "succeeded" and attempt["callback_invocations"] == 1
        assert attempt["binding"]["protocol_sha256"] == repaired["session_sha256"]
        rejected(lambda: selected.evaluate_registered(batch, evaluate, evaluator_id="duplicate"))
        rejected(lambda: lane("ordered"))

        replaced = lane("replaced")
        replacement = replaced.select_diagnosis(lambda query: answer)
        original = replaced._records["diagnosis"]
        replacement["candidate"]["answer"]["mechanism"] = "Replacement with recomputed receipt hash"
        replacement.pop("sha256")
        (root / "replaced" / "diagnosis.json").write_bytes(
            canonical_bytes(replacement | {"sha256": digest(replacement)}) + b"\n")
        rejected(lambda: replaced.select_repair(lambda diagnosis, query: calls.append("replacement")))
        (root / "replaced" / "diagnosis.json").write_bytes(original)
        rejected(lambda: replaced.select_repair(lambda diagnosis, query: calls.append("replacement")))
        assert "repair" not in replaced._brokers and calls == ["diagnosis", "repair", "evaluation"]

        repair_replaced = lane("repair-replaced")
        repair_replaced.select_diagnosis(lambda query: answer)
        repair_replaced.select_repair(lambda diagnosis, query: {"commands": [], "intent": "Literal no-op guard"})
        (root / "repair-replaced" / "repair.json").write_bytes(b"{}\n")
        rejected(lambda: repair_replaced.evaluate_registered(batch, evaluate, evaluator_id="replacement"))
        assert calls == ["diagnosis", "repair", "evaluation"]

        abstaining = lane("abstaining")
        abstaining.select_diagnosis(lambda query: answer | {"location": None, "abstain": True})
        rejected(lambda: abstaining.select_repair(lambda diagnosis, query: calls.append("abstention")))
        assert "repair" not in abstaining._brokers and calls == ["diagnosis", "repair", "evaluation"]
        print("PASS: strict diagnosis; durable diagnosis before separate repair; exact binding; replacement, premature and duplicate rejection; one durable literal evaluation authorization.")
        print("Engineering guard only: no training, provider, held-out study or causal-utility claim.")


if __name__ == "__main__":
    _guard_check()
