"""Frozen cohort preparation, diagnosis-before-repair selection and endpoint seals.

This trusted cooperative controller owns its lanes and retains failed preparation,
failed selection and abstention in the complete ordered roster. A proposer gets
only a member row and the lane's JSON query function. This is not a Python sandbox
or a study result. There is deliberately no cohort scoring/evaluation method;
public per-lane evaluation is denied pending a separate evaluation bridge.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np

from .artifact import TensorArtifact
from .broker import WeightView
from .data_roles import DataRoleRegistry
from .model import Network, canonical_bytes, digest
from .program import Program, apply_program
from .staged_diagnosis import StagedDiagnosisRepair
from .structured import ExecutionSpec, StructuredProgram, lower_structured
from .workflow import Artifact, BudgetLimits, Candidate, WorkflowError, _json


def _captured_interface(endpoint: Artifact) -> dict[str, Any] | None:
    """Validate saved bridge declarations without importing or executing a module."""
    metadata = endpoint.metadata if type(endpoint) is TensorArtifact else {}
    if "torch_bridge" not in metadata:
        return None
    bridge = metadata["torch_bridge"]
    fields = {"schema", "torch_version", "module_type", "module_repr", "forward_signature",
              "modules", "state", "alias_groups", "execution_scope"}
    if (type(bridge) is not dict or set(bridge) != fields
            or bridge["schema"] != "wsp-torch-bridge/1"
            or any(type(bridge[key]) is not str for key in (
                "torch_version", "module_type", "module_repr", "forward_signature", "execution_scope"))
            or not bridge["module_type"].strip()):
        raise ValueError("Malformed captured Torch interface")
    modules, state, tensors = bridge["modules"], bridge["state"], endpoint.tensors
    if (type(modules) is not dict or "" not in modules
            or any(type(name) is not str or type(module) is not dict
                   or set(module) != {"type", "training"}
                   or type(module["type"]) is not str or not module["type"].strip()
                   or type(module["training"]) is not bool for name, module in modules.items())
            or modules[""]["type"] != bridge["module_type"]
            or type(state) is not dict or set(state) != set(tensors)):
        raise ValueError("Captured modules/state do not match the endpoint interface")
    dtypes = {"torch." + name: name for name in ("float16", "bfloat16", "float32", "float64")}
    for name, record in state.items():
        if (type(record) is not dict
                or set(record) != {"dtype", "shape", "device", "role", "requires_grad"}
                or type(record["dtype"]) is not str or record["dtype"] not in dtypes
                or type(record["shape"]) is not list
                or any(type(size) is not int or size <= 0 for size in record["shape"])
                or record["shape"] != list(tensors[name].shape)
                or type(record["device"]) is not str or not record["device"].strip()
                or type(record["role"]) is not str or record["role"] not in ("parameter", "buffer", "state")
                or type(record["requires_grad"]) is not bool):
            raise ValueError("Malformed captured state declaration")
    groups, seen = bridge["alias_groups"], set()
    if type(groups) is not list:
        raise ValueError("Captured aliases must be an ordered list")
    for group in groups:
        if (type(group) is not list or len(group) < 2
                or any(type(name) is not str or name not in state for name in group)
                or len(set(group)) != len(group) or seen.intersection(group)):
            raise ValueError("Captured aliases must be disjoint groups of distinct state names")
        first = group[0]
        if any(any(state[name][key] != state[first][key] for key in ("shape", "dtype", "device"))
               or not np.array_equal(tensors[first], tensors[name]) for name in group[1:]):
            raise ValueError("Captured alias declarations disagree with endpoint data")
        seen.update(group)
    return {"native_dtypes": {name: dtypes[record["dtype"]] for name, record in state.items()},
            "devices": {name: record["device"] for name, record in state.items()},
            "alias_groups": groups}


def _validate_evaluation_plan(plan: Any, controls: dict[str, Any],
                              member_ids: list[str] | tuple[str, ...]) -> None:
    """Validate declarations only; endpoint construction and scoring stay separate."""
    def fields(value: Any, expected: set[str], label: str) -> None:
        if type(value) is not dict or set(value) != expected:
            raise ValueError(f"Unknown or missing {label} fields")

    def identifier(value: Any) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("Evaluation identities must be nonblank UTF-8 strings")
        value.encode("utf-8")
        return value

    fields(plan, {"schema", "result_schema", "aggregate", "holdouts", "members"},
           "evaluation-plan")
    if (plan["schema"] != "wsp-staged-cohort-evaluation-plan/1"
            or plan["result_schema"] != "wsp-staged-cohort-result/1"):
        raise ValueError("Unknown cohort evaluation-plan or result schema")
    fields(plan["aggregate"], {"source_id", "family_id"}, "aggregate")
    aggregate_source = identifier(plan["aggregate"]["source_id"])
    identifier(plan["aggregate"]["family_id"])
    if type(plan["holdouts"]) is not list or not plan["holdouts"]:
        raise ValueError("Declare a nonempty ordered holdout list")
    holdout_ids, source_ids = set(), {aggregate_source}
    for holdout in plan["holdouts"]:
        fields(holdout, {"holdout_id", "source_id", "family_id", "row_count"}, "holdout")
        holdout_id, source_id = identifier(holdout["holdout_id"]), identifier(holdout["source_id"])
        identifier(holdout["family_id"])
        if (holdout_id in holdout_ids or source_id in source_ids
                or type(holdout["row_count"]) is not int or holdout["row_count"] <= 0):
            raise ValueError("Use unique holdout/source IDs and strict positive integer row counts")
        holdout_ids.add(holdout_id)
        source_ids.add(source_id)

    if set(controls) != set(member_ids):
        raise ValueError("Evaluation controls must cover exactly the original member IDs")
    variants = {"parent": {"kind"}, "reference": {"kind", "artifact"},
                "structured": {"kind", "proposal", "lowering"}}
    schemas = {"artifact": ("wsp-network/1", "wsp-tensors/1"),
               "proposal": ("wsp-structured-program/1",), "lowering": ("wsp-lowering/1",)}
    for definitions in controls.values():
        if type(definitions) is not dict:
            raise ValueError("Each member's controls must be a JSON object")
        for control_id, definition in definitions.items():
            identifier(control_id)
            kind = definition.get("kind") if type(definition) is dict else None
            if type(kind) is not str or kind not in variants:
                raise ValueError("Unknown cohort control kind")
            fields(definition, variants[kind], "control")
            for key in variants[kind] - {"kind"}:
                payload = definition[key]
                if type(payload) is not dict or payload.get("schema") not in schemas[key]:
                    raise ValueError(f"Unknown {key} declaration schema")

    members = plan["members"]
    if type(members) is not list or len(members) != len(member_ids):
        raise ValueError("Evaluation members must match the complete ordered roster")
    for member_id, member in zip(member_ids, members):
        fields(member, {"member_id", "holdout_id", "slots"}, "evaluation-member")
        if (identifier(member["member_id"]) != member_id
                or identifier(member["holdout_id"]) not in holdout_ids):
            raise ValueError("Evaluation members must match the roster and declared holdouts")
        if type(member["slots"]) is not list or not member["slots"]:
            raise ValueError("Declare nonempty ordered slots for every member")
        slot_ids = set()
        for slot in member["slots"]:
            fields(slot, {"slot_id", "source", "execution", "result_keys"}, "evaluation-slot")
            slot_id = identifier(slot["slot_id"])
            if slot_id in slot_ids:
                raise ValueError("Slot IDs must be unique within each member")
            slot_ids.add(slot_id)
            source = slot["source"]
            kind = source.get("kind") if type(source) is dict else None
            if kind == "selected":
                fields(source, {"kind", "fallback_control_id"}, "selected source")
                control_id = source["fallback_control_id"]
                if control_id is not None:
                    definition = controls[member_id].get(identifier(control_id))
                    if definition is None or definition["kind"] != "parent":
                        raise ValueError("Selected fallback must name a same-member parent control")
            elif kind == "control":
                fields(source, {"kind", "control_id"}, "control source")
                if identifier(source["control_id"]) not in controls[member_id]:
                    raise ValueError("Slot must name a same-member control")
            else:
                raise ValueError("Unknown evaluation-slot source")
            execution = slot["execution"]
            if type(execution) is not dict or any(type(value) is not str for value in execution.values()):
                raise ValueError("Execution declarations must contain the six string fields")
            ExecutionSpec.from_dict(execution)
            keys = slot["result_keys"]
            if type(keys) is not list or not keys:
                raise ValueError("Declare nonempty unique result keys")
            if len({identifier(key) for key in keys}) != len(keys):
                raise ValueError("Result keys must be unique")


@dataclass(frozen=True)
class PreparedMember:
    """Trusted preparation result; the controller constructs the lane itself."""

    base: Artifact
    task_id: str
    objective: str
    preparation: dict[str, Any]
    view: WeightView = WeightView()
    limits: BudgetLimits | None = None
    native_dtypes: dict[str, str] | None = None


class StagedCohort:
    def __init__(self, run_dir: str | Path, *, member_ids: list[str] | tuple[str, ...],
                 protocol: dict[str, Any], controls: dict[str, Any], evaluator_id: str,
                 registry: DataRoleRegistry, attempt_id: str):
        if (type(member_ids) not in (list, tuple) or not member_ids
                or any(type(value) is not str or not value.strip() for value in member_ids)
                or len(set(member_ids)) != len(member_ids)):
            raise ValueError("Freeze a nonempty ordered unique member-ID roster")
        if (type(registry) is not DataRoleRegistry
                or any(type(value) is not str or not value.strip()
                       for value in (evaluator_id, attempt_id))):
            raise ValueError("Use a trusted registry and explicit evaluator/attempt identities")
        protocol, controls = _json(protocol), _json(controls)
        if type(protocol) is not dict or type(controls) is not dict:
            raise ValueError("Protocol and controls must be finite JSON objects")
        if "evaluation_plan" in protocol:
            _validate_evaluation_plan(protocol["evaluation_plan"], controls, member_ids)
        rows = [{"member_index": index, "member_id": identifier}
                for index, identifier in enumerate(member_ids)]
        self._rows_bytes = canonical_bytes(rows)
        self._registry, self._owner = registry, object()
        self._lanes: list[StagedDiagnosisRepair | None] = [None] * len(rows)
        self._records: dict[str, bytes] = {}
        self._invalidated, self._prepared = False, False
        self._started = {"preparation": False, "diagnosis": False, "repair": False, "endpoints": False}
        self._endpoints: dict[tuple[int, str, str | None], dict[str, Any]] = {}
        # ponytail: serialize one cohort; use separate controllers for parallel cohorts.
        self._lock = threading.RLock()
        body = {"schema": "wsp-staged-cohort-protocol/1", "attempt_id": attempt_id,
                "registry_id": registry.registry_id, "evaluator_id": evaluator_id,
                "members": rows, "roster_sha256": digest(rows), "protocol": protocol,
                "controls": controls, "controls_sha256": digest(controls),
                "selection_rule": "all_durable_terminal_diagnoses_before_any_repair_construction",
                "evaluation_available": False, "causal_usefulness_established": False}
        canonical_bytes(body)  # Validate encodability before creating the exclusive run.
        self._directory = Path(run_dir).resolve()
        self._directory.mkdir(parents=True, exist_ok=False)
        try:
            self._save("protocol", body)
        except BaseException:
            self._invalidate()
            raise

    def _rows(self) -> list[dict[str, Any]]:
        return json.loads(self._rows_bytes)

    @staticmethod
    def _name(stage: str, index: int) -> str:
        return f"{stage}-{index:06d}"

    def _invalidate(self) -> None:
        self._invalidated = True
        for lane in self._lanes:
            if lane is not None:
                lane._invalidated = True

    def _require(self, condition: bool) -> None:
        if not condition:
            self._invalidate()
            raise WorkflowError("Cohort evidence changed; the attempt is invalidated")

    def _write(self, name: str, body: dict[str, Any]) -> bytes:
        raw = canonical_bytes(body | {"sha256": digest(body)}) + b"\n"
        path = self._directory / (name + ".json")
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if path.read_bytes() != raw:
            raise WorkflowError("Durable cohort receipt verification failed")
        return raw

    def _save(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        self._records[name] = self._write(name, body)
        return self._read(name)

    def _read(self, name: str) -> dict[str, Any]:
        if self._invalidated:
            raise WorkflowError("Cohort attempt is invalidated")
        try:
            raw = self._records[name]
            self._require((self._directory / (name + ".json")).read_bytes() == raw)
            document = json.loads(raw)
            self._require(digest({key: value for key, value in document.items()
                                  if key != "sha256"}) == document["sha256"])
            return document
        except BaseException as error:
            self._invalidate()
            if isinstance(error, (OSError, ValueError, KeyError, TypeError)):
                raise WorkflowError("Cohort evidence is unavailable or changed") from None
            raise

    def _check(self) -> dict[str, Any]:
        protocol = self._read("protocol")
        self._require(protocol["members"] == self._rows()
                      and protocol["registry_id"] == self._registry.registry_id)
        return protocol

    def _body(self, stage: str, row: dict[str, Any], outcome: str,
              **fields: Any) -> dict[str, Any]:
        protocol = self._check()
        return {"schema": "wsp-staged-cohort-disposition/1", "stage": stage, **row,
                "cohort_protocol_sha256": protocol["sha256"],
                "roster_sha256": protocol["roster_sha256"], "outcome": outcome, **fields}

    def _member_binding(self, index: int) -> dict[str, Any]:
        protocol = self._check()
        return {"cohort_protocol_sha256": protocol["sha256"],
                "cohort_roster_sha256": protocol["roster_sha256"],
                "cohort_member_index": index, "cohort_member_id": self._rows()[index]["member_id"]}

    def _lane_snapshot(self, lane: StagedDiagnosisRepair, index: int) -> dict[str, Any]:
        self._require(lane._owner is self._owner and lane._registry is self._registry
                      and not lane._invalidated
                      and (lane._directory / "protocol.json").read_bytes() == lane._protocol)
        protocol = json.loads(lane._protocol)
        self._require(protocol["cohort_binding"] == self._member_binding(index))
        return {"lane_protocol_sha256": protocol["sha256"],
                "parent_sha256": lane._base.sha256,
                "diagnosis_task_sha256": lane._tasks["diagnosis"].sha256,
                "diagnosis_session_sha256": lane._brokers["diagnosis"]._session.sha256,
                "configuration": lane._brokers["diagnosis"]._session.configuration}

    def _preparation(self, index: int) -> dict[str, Any]:
        preparation = self._read(self._name("preparation", index))
        lane = self._lanes[index]
        self._require((preparation["outcome"] == "prepared") == (lane is not None))
        if lane is not None:
            self._require(preparation["lane"] == self._lane_snapshot(lane, index))
        return preparation

    def prepare(self, prepare_member: Callable[[dict[str, Any]], PreparedMember]) -> list[dict[str, Any]]:
        with self._lock:
            self._check()
            if self._started["preparation"] or not callable(prepare_member):
                raise WorkflowError("Cohort preparation is available once")
            self._started["preparation"] = True
            try:
                for row in self._rows():
                    self._check()
                    index = row["member_index"]
                    try:
                        prepared = prepare_member(dict(row))
                        if type(prepared) is not PreparedMember:
                            raise ValueError("Preparation must return PreparedMember")
                        output = _json(prepared.preparation)
                        if type(output) is not dict:
                            raise ValueError("Preparation output must be a finite JSON object")
                        StagedDiagnosisRepair._configuration(
                            prepared.task_id, prepared.objective, prepared.base, self._registry,
                            prepared.view, prepared.limits, prepared.native_dtypes)
                    except BaseException as error:
                        try:
                            self._save(self._name("preparation", index), self._body(
                                "preparation", row, "unavailable", lane=None,
                                exception_type=type(error).__name__))
                        except BaseException:
                            self._invalidate()
                            raise error
                        if not isinstance(error, Exception):
                            raise
                        continue
                    try:
                        lane = StagedDiagnosisRepair(self._directory / f"member-{index:06d}",
                            task_id=prepared.task_id, objective=prepared.objective,
                            base=prepared.base, registry=self._registry, view=prepared.view,
                            limits=prepared.limits, native_dtypes=prepared.native_dtypes,
                            _owner=self._owner, _cohort_binding=self._member_binding(index))
                        self._lanes[index] = lane
                    except BaseException as error:
                        try:
                            self._save(self._name("preparation", index), self._body(
                                "preparation", row, "unavailable", lane=None,
                                exception_type=type(error).__name__, failure_phase="lane_setup"))
                        except BaseException:
                            pass
                        raise error
                    self._save(self._name("preparation", index), self._body(
                        "preparation", row, "prepared", preparation=output,
                        lane=self._lane_snapshot(lane, index)))
                self._prepared = True
                return [self._preparation(row["member_index"]) for row in self._rows()]
            except BaseException:
                self._invalidate()
                raise

    def _select(self, stage: str, row: dict[str, Any], selector: Callable[..., Any],
                cohort_binding: dict[str, Any] | None = None) -> dict[str, Any]:
        index, error = row["member_index"], None
        preparation = self._preparation(index)
        lane = self._lanes[index]
        try:
            if stage == "diagnosis":
                lane._select_diagnosis(lambda query: selector(dict(row), query), self._owner)
            else:
                lane._select_repair(lambda diagnosis, query: selector(dict(row), diagnosis, query),
                                    self._owner, cohort_binding)
        except BaseException as caught:
            error = caught
        try:
            # The raised selector error alone cannot attest successful finalization.
            actual = lane._verify(stage)
            self._preparation(index)
            diagnosis_binding = {}
            if stage == "repair":
                barrier = self._verify_barrier()
                self._require(barrier["sha256"] == cohort_binding["cohort_diagnosis_barrier_sha256"])
                member = barrier["members"][index]
                diagnosis_binding = {key: member[key] for key in (
                    "diagnosis_disposition_sha256", "diagnosis_receipt_sha256")}
                diagnosis_binding.update(diagnosis_outcome=member["outcome"],
                                         diagnosis_barrier_sha256=barrier["sha256"])
            disposition = self._save(self._name(stage, index), self._body(
                stage, row, actual["outcome"], preparation_receipt_sha256=preparation["sha256"],
                **{stage + "_receipt_sha256": actual["sha256"]},
                **diagnosis_binding,
                exception_type=type(error).__name__ if error is not None else None))
        except BaseException:
            self._invalidate()
            if error is not None:
                raise error
            raise
        if error is not None and not isinstance(error, Exception):
            self._invalidate()
            raise error
        return disposition

    def _diagnoses(self) -> list[dict[str, Any]]:
        members = []
        for row in self._rows():
            index = row["member_index"]
            preparation = self._preparation(index)
            disposition = self._read(self._name("diagnosis", index))
            lane = self._lanes[index]
            actual = lane._verify("diagnosis") if lane is not None else None
            actual_hash = actual["sha256"] if actual is not None else None
            outcome = actual["outcome"] if actual is not None else "unavailable"
            self._require(disposition["outcome"] == outcome
                          and disposition["diagnosis_receipt_sha256"] == actual_hash
                          and disposition["preparation_receipt_sha256"] == preparation["sha256"])
            members.append({**row, "preparation_receipt_sha256": preparation["sha256"],
                            "diagnosis_disposition_sha256": disposition["sha256"],
                            "diagnosis_receipt_sha256": actual_hash, "outcome": outcome})
        return members

    def _barrier_body(self) -> dict[str, Any]:
        protocol = self._check()
        return {"schema": "wsp-staged-cohort-diagnosis-barrier/1",
                "cohort_protocol_sha256": protocol["sha256"],
                "roster_sha256": protocol["roster_sha256"],
                "attempt_id": protocol["attempt_id"], "registry_id": protocol["registry_id"],
                "evaluator_id": protocol["evaluator_id"], "members": self._diagnoses(),
                "evaluation_available": False}

    def _verify_barrier(self) -> dict[str, Any]:
        try:
            barrier = self._read("diagnosis-barrier")
            self._require(canonical_bytes({key: value for key, value in barrier.items()
                                           if key != "sha256"}) == canonical_bytes(self._barrier_body()))
            return barrier
        except BaseException:
            self._invalidate()
            raise

    def _repair_barrier_body(self) -> dict[str, Any]:
        protocol, barrier = self._check(), self._verify_barrier()
        members = []
        for row in self._rows():
            index = row["member_index"]
            diagnosis = barrier["members"][index]
            disposition = self._read(self._name("repair", index))
            lane, actual = self._lanes[index], None
            if diagnosis["outcome"] == "selected":
                self._require(lane is not None)
                actual = lane._verify("repair")
                broker = lane._brokers["repair"]
                session, parent = broker._session, lane._base
                candidate = session._selected
                self._require(broker._base is parent and session._base is parent
                              and actual["task"]["base_sha256"] == parent.sha256)
                binding = {**self._member_binding(index),
                           "cohort_diagnosis_barrier_sha256": barrier["sha256"],
                           "cohort_preparation_receipt_sha256": diagnosis["preparation_receipt_sha256"],
                           "cohort_diagnosis_disposition_sha256": diagnosis["diagnosis_disposition_sha256"],
                           "diagnosis_receipt_sha256": diagnosis["diagnosis_receipt_sha256"]}
                self._require(all(actual["task"]["constraints"][key] == value
                                  for key, value in binding.items()))
                if actual["outcome"] == "selected":
                    self._require(type(candidate) is Candidate and candidate.artifact is not None
                                  and candidate.to_dict() == actual["candidate"]
                                  and candidate.parent_sha256 == parent.sha256
                                  and candidate.task_sha256 == actual["task_sha256"]
                                  and candidate.to_dict()["session_sha256"] == actual["session_sha256"]
                                  and candidate.artifact.sha256 == actual["candidate"]["artifact_sha256"]
                                  and candidate.artifact.to_dict() == actual["candidate"]["artifact"])
                else:
                    self._require(candidate is None)
            else:
                self._require(lane is None or ("repair" not in lane._records
                                              and "repair" not in lane._brokers
                                              and "repair" not in lane._tasks))
            expected = {key: diagnosis[key] for key in (
                "preparation_receipt_sha256", "diagnosis_disposition_sha256", "diagnosis_receipt_sha256")}
            expected.update(diagnosis_outcome=diagnosis["outcome"],
                            diagnosis_barrier_sha256=barrier["sha256"],
                            repair_receipt_sha256=actual["sha256"] if actual is not None else None,
                            outcome=actual["outcome"] if actual is not None else "unavailable")
            self._require(all(disposition[key] == value for key, value in expected.items()))
            members.append({**row, **{key: value for key, value in expected.items()
                                     if key != "diagnosis_barrier_sha256"},
                            "repair_disposition_sha256": disposition["sha256"]})
        return {"schema": "wsp-staged-cohort-repair-barrier/1",
                "cohort_protocol_sha256": protocol["sha256"],
                "roster_sha256": protocol["roster_sha256"],
                "attempt_id": protocol["attempt_id"], "registry_id": protocol["registry_id"],
                "evaluator_id": protocol["evaluator_id"], "diagnosis_barrier_sha256": barrier["sha256"],
                "members": members}

    def _verify_repair_barrier(self) -> dict[str, Any]:
        try:
            barrier = self._read("repair-barrier")
            self._require(canonical_bytes({key: value for key, value in barrier.items()
                                           if key != "sha256"}) == canonical_bytes(self._repair_barrier_body()))
            return barrier
        except BaseException:
            self._invalidate()
            raise

    def _verify_endpoint_configuration(self, lane: StagedDiagnosisRepair) -> None:
        # Inspect the frozen/live declarations directly: configuration preflight lowers.
        frozen = json.loads(lane._protocol)
        self._require(lane._base.sha256 == frozen["parent_sha256"]
                      and type(lane._view) is WeightView and type(lane._limits) is BudgetLimits
                      and canonical_bytes(lane._native_dtypes) == canonical_bytes(frozen["native_dtypes"])
                      and canonical_bytes(lane._view.to_dict()) == canonical_bytes(frozen["parameter_view"])
                      and canonical_bytes(lane._limits.to_dict()) == canonical_bytes(frozen["limits"]))
        for broker in lane._brokers.values():
            session = broker._session
            self._require(broker._base is lane._base and session._base is lane._base
                          and session._data_registry is self._registry
                          and type(broker._view) is WeightView
                          and canonical_bytes(broker._native_dtypes) == canonical_bytes(frozen["native_dtypes"])
                          and canonical_bytes(broker._view.to_dict()) == canonical_bytes(frozen["parameter_view"])
                          and canonical_bytes(session.limits.to_dict()) == canonical_bytes(frozen["limits"])
                          and canonical_bytes(session.access.to_dict()) == canonical_bytes(session.configuration["access"]))

    @staticmethod
    def _endpoint_key(index: int, slot: dict[str, Any], outcome: str
                      ) -> tuple[tuple[int, str, str | None], str]:
        source = slot["source"]
        if source["kind"] == "control":
            return (index, "control", source["control_id"]), "control"
        if outcome != "selected" and source["fallback_control_id"] is not None:
            return (index, "control", source["fallback_control_id"]), "parent_fallback"
        return (index, "selected", None), "selected"

    @staticmethod
    def _origin(key: tuple[int, str, str | None]) -> dict[str, Any]:
        return {"kind": key[1], **({"control_id": key[2]} if key[1] == "control" else {})}

    def _materialize_endpoint(self, key: tuple[int, str, str | None],
                              protocol: dict[str, Any], repair: dict[str, Any]) -> None:
        if key in self._endpoints:
            return
        index, kind, control_id = key
        lane = self._lanes[index]
        parent = lane._base if lane is not None else None
        endpoint = candidate = proposal = program = lowering = construction = None
        if parent is not None:
            _captured_interface(parent)
            if kind == "selected" and repair["members"][index]["outcome"] == "selected":
                selection = lane._verify("repair")
                candidate = lane._brokers["repair"]._session._selected
                endpoint = candidate.artifact
                submission = selection["submission"]
                submitted = submission["submitted"]
                self._require(set(submitted) == {"op", "commands", "intent"}
                              and submitted["op"] == "submit" and submission["status"] == "lowered")
                proposal = StructuredProgram(parent.sha256, submitted["commands"], submitted["intent"])
                program = Program.from_dict(submission["program"])
                lowering = canonical_bytes(submission["lowering"])
                receipt = json.loads(lowering)
                native = lane._native_dtypes
                native = {name: "float64" for name in parent.tensors} if native is None else native
                self._require(receipt["schema"] == "wsp-lowering/1"
                              and receipt["sha256"] == digest({k: v for k, v in receipt.items() if k != "sha256"})
                              and receipt["proposal_sha256"] == proposal.sha256
                              and receipt["base_sha256"] == program.base_sha256 == parent.sha256
                              and receipt["program_sha256"] == program.sha256
                              and receipt["candidate_sha256"] == endpoint.sha256
                              and receipt["command_count"] == len(submitted["commands"])
                              and receipt["effective_edits"] == len(program.edits)
                              and receipt["cast"] == "round"
                              and canonical_bytes(receipt["native_dtypes"]) == canonical_bytes(native)
                              and canonical_bytes(program.to_dict()) == canonical_bytes(submission["program"]))
                reproduced = apply_program(parent, program)
                self._require(canonical_bytes(reproduced.to_dict()) == canonical_bytes(endpoint.to_dict()))
                construction = {"native_dtypes": receipt["native_dtypes"], "cast": receipt["cast"],
                                "verification": "saved_literal_replay_only; structured_lowering_is_broker_provenance"}
            elif kind == "control":
                definition = protocol["controls"][self._rows()[index]["member_id"]][control_id]
                if definition["kind"] == "parent":
                    endpoint = parent
                elif definition["kind"] == "reference":
                    data = definition["artifact"]
                    endpoint = (Network.from_dict(data) if data["schema"] == "wsp-network/1"
                                else TensorArtifact.from_dict(data))
                else:
                    proposal = StructuredProgram.from_dict(definition["proposal"])
                    receipt = definition["lowering"]
                    lowered = lower_structured(parent, proposal, native_dtypes=receipt["native_dtypes"],
                                               cast=receipt["cast"])
                    self._require(canonical_bytes(lowered.receipt) == canonical_bytes(receipt))
                    program, lowering = lowered.program, canonical_bytes(receipt)
                    endpoint = apply_program(parent, program)
                    construction = {"native_dtypes": receipt["native_dtypes"], "cast": receipt["cast"],
                                    "verification": "complete_lowering_receipt_reproduced_once"}
            if endpoint is not None:
                self._require(parent.compatible(endpoint))
                _captured_interface(endpoint)
        self._endpoints[key] = {"parent": parent, "endpoint": endpoint, "candidate": candidate,
                                "proposal": proposal, "program": program, "lowering": lowering,
                                "construction": canonical_bytes(construction)}

    def _endpoint_body(self, key: tuple[int, str, str | None]) -> dict[str, Any]:
        index = key[0]
        retained, lane = self._endpoints[key], self._lanes[index]
        parent, endpoint, candidate = (retained[name] for name in ("parent", "endpoint", "candidate"))
        self._require(parent is (lane._base if lane is not None else None))
        if candidate is not None:
            self._require(key[1] == "selected" and candidate is lane._brokers["repair"]._session._selected
                          and endpoint is candidate.artifact)
        if endpoint is not None:
            self._require(parent is not None and parent.compatible(endpoint))
        captured = _captured_interface(endpoint) if endpoint is not None else None
        construction = json.loads(retained["construction"])
        agreement = ("absent" if captured is None else "not_applicable" if construction is None else
                     "agree" if construction["native_dtypes"] == captured["native_dtypes"] else "mismatch")
        return {**self._rows()[index], "origin": self._origin(key),
                "status": "materialized" if endpoint is not None else "unavailable",
                "parent_sha256": parent.sha256 if parent is not None else None,
                "endpoint_sha256": endpoint.sha256 if endpoint is not None else None,
                "artifact": endpoint.to_dict() if endpoint is not None else None,
                "candidate_sha256": candidate.sha256 if candidate is not None else None,
                "proposal": retained["proposal"].to_dict() if retained["proposal"] is not None else None,
                "program": retained["program"].to_dict() if retained["program"] is not None else None,
                "lowering": json.loads(retained["lowering"]) if retained["lowering"] is not None else None,
                "construction": construction,
                "captured_native_dtypes": captured["native_dtypes"] if captured is not None else None,
                "captured_devices": captured["devices"] if captured is not None else None,
                "captured_alias_groups": captured["alias_groups"] if captured is not None else None,
                "construction_capture_agreement": agreement}

    def _endpoint_seal_body(self) -> dict[str, Any]:
        protocol, repair = self._check(), self._verify_repair_barrier()
        plan = protocol["protocol"]["evaluation_plan"]
        members, origins, unused = [], {}, []
        for row, declared, disposition in zip(self._rows(), plan["members"], repair["members"]):
            index, slots = row["member_index"], []
            lane = self._lanes[index]
            if lane is not None:
                self._verify_endpoint_configuration(lane)
            for slot_index, slot in enumerate(declared["slots"]):
                key, resolution = self._endpoint_key(index, slot, disposition["outcome"])
                if key not in origins:
                    origins[key] = self._endpoint_body(key)
                origin = origins[key]
                slots.append({"slot_index": slot_index, "slot_id": slot["slot_id"],
                              "source": slot["source"], "resolved_origin": self._origin(key),
                              "resolution": resolution if origin["status"] == "materialized" else "unavailable",
                              "endpoint_sha256": origin["endpoint_sha256"],
                              "candidate_sha256": origin["candidate_sha256"],
                              "execution": slot["execution"], "result_keys": slot["result_keys"]})
            members.append({**row, "holdout_id": declared["holdout_id"],
                            "repair_outcome": disposition["outcome"], "slots": slots})
            unused.append({**row, "control_ids": [control_id for control_id in protocol["controls"][row["member_id"]]
                           if (index, "control", control_id) not in origins]})
        self._require(set(self._endpoints) == set(origins))
        return {"schema": "wsp-staged-cohort-endpoint-seal/1",
                "cohort_protocol_sha256": protocol["sha256"], "roster_sha256": protocol["roster_sha256"],
                "evaluation_plan_sha256": digest(plan), "controls_sha256": protocol["controls_sha256"],
                "attempt_id": protocol["attempt_id"], "registry_id": protocol["registry_id"],
                "evaluator_id": protocol["evaluator_id"],
                "diagnosis_barrier_sha256": repair["diagnosis_barrier_sha256"],
                "repair_barrier_sha256": repair["sha256"], "origins": list(origins.values()), "members": members,
                "unused_controls": unused, "evaluation_available": False,
                "runtime_restoration_verified": False, "causal_usefulness_established": False}

    def _verify_endpoint_seal(self) -> dict[str, Any]:
        try:
            seal = self._read("endpoint-seal")
            self._require(canonical_bytes({key: value for key, value in seal.items() if key != "sha256"})
                          == canonical_bytes(self._endpoint_seal_body()))
            return seal
        except BaseException:
            self._invalidate()
            raise

    def seal_endpoints(self) -> dict[str, Any]:
        """Materialize referenced origins once; retain every declared comparison slot."""
        with self._lock:
            protocol = self._check()
            if ("evaluation_plan" not in protocol["protocol"] or self._started["endpoints"]
                    or "repair-barrier" not in self._records):
                raise WorkflowError("A declared evaluation plan and complete repair barrier are required once")
            self._started["endpoints"] = True
            try:
                repair = self._verify_repair_barrier()
                for index, member in enumerate(protocol["protocol"]["evaluation_plan"]["members"]):
                    lane = self._lanes[index]
                    if lane is not None:
                        self._verify_endpoint_configuration(lane)
                    for slot in member["slots"]:
                        key, _ = self._endpoint_key(index, slot, repair["members"][index]["outcome"])
                        self._materialize_endpoint(key, protocol, repair)
                self._save("endpoint-seal", self._endpoint_seal_body())
                return self._verify_endpoint_seal()
            except BaseException:
                self._invalidate()
                raise

    def select_diagnoses(self, selector: Callable[..., Any]) -> dict[str, Any]:
        with self._lock:
            self._check()
            if not self._prepared or self._started["diagnosis"] or not callable(selector):
                raise WorkflowError("All preparation must finish before the single diagnosis pass")
            self._started["diagnosis"] = True
            try:
                for row in self._rows():
                    index = row["member_index"]
                    preparation = self._preparation(index)
                    if self._lanes[index] is None:
                        self._save(self._name("diagnosis", index), self._body(
                            "diagnosis", row, "unavailable",
                            preparation_receipt_sha256=preparation["sha256"],
                            diagnosis_receipt_sha256=None))
                    else:
                        self._select("diagnosis", row, selector)
                self._save("diagnosis-barrier", self._barrier_body())
                return self._verify_barrier()
            except BaseException:
                self._invalidate()
                raise

    def select_repairs(self, selector: Callable[..., Any]) -> list[dict[str, Any]]:
        with self._lock:
            self._check()
            if (self._started["repair"] or "diagnosis-barrier" not in self._records
                    or not callable(selector)):
                raise WorkflowError("The complete durable diagnosis barrier is required once")
            self._started["repair"] = True
            try:
                results = []
                for row in self._rows():
                    barrier = self._verify_barrier()
                    index = row["member_index"]
                    member = barrier["members"][index]
                    if member["outcome"] != "selected":
                        results.append(self._save(self._name("repair", index), self._body(
                            "repair", row, "unavailable", **{key: member[key] for key in (
                                "preparation_receipt_sha256", "diagnosis_disposition_sha256",
                                "diagnosis_receipt_sha256")}, repair_receipt_sha256=None,
                            diagnosis_outcome=member["outcome"],
                            diagnosis_barrier_sha256=barrier["sha256"])))
                        continue
                    binding = {**self._member_binding(index),
                               "cohort_diagnosis_barrier_sha256": barrier["sha256"],
                               "cohort_preparation_receipt_sha256": member["preparation_receipt_sha256"],
                               "cohort_diagnosis_disposition_sha256": member["diagnosis_disposition_sha256"]}
                    results.append(self._select("repair", row, selector, binding))
                self._save("repair-barrier", self._repair_barrier_body())
                self._verify_repair_barrier()
                return results
            except BaseException:
                self._invalidate()
                raise
