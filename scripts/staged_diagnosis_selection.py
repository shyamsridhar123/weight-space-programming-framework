"""Selection-only, live admission for the prospective staged diagnosis policies.

This consumer uses public phase methods and their source-pinned receipt format.
It neither restores a run nor authorizes evaluation. Source pins describe reviewed
disk-format compatibility, not loaded-code identity or a Python sandbox.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat
import statistics
import struct
from typing import Any, Callable
from weakref import WeakSet

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.broker import WeightView
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.program import Program, apply_program
from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort
from rlcd.weight_space.structured import StructuredProgram, lower_structured, verify_lowering
from rlcd.weight_space.workflow import (AccessPolicy, BudgetLimits, BudgetUsage,
                                       Candidate, TaskSpec, validate_splits, verify_trace)


SOURCE_PINS = {
    "rlcd/weight_space/staged_cohort.py": "2501fc968878cd3894646745dd6ce5986f4ffc808e8e6bf5567ca5688cd36687",
    "rlcd/weight_space/staged_diagnosis.py": "ff3449c6392d8fca2af2114ff36c3760a207eff5b7bd667f0939bdaf1ef74dd5",
    "rlcd/weight_space/data_roles.py": "da2cffdd0774634a10877c5182f77f2fbb9cf402a0d6c495cde10722a8a6d851",
    "rlcd/weight_space/artifact.py": "127621e5bb523c7d0f8104b3e398921dae9cbd59c3140147976ccb377c996500",
    "rlcd/weight_space/model.py": "fbb742d5f4a83055ff88609bb6de82b3cf9334b089b2eb8b7a47aa846ae9d5af",
    "rlcd/weight_space/workflow.py": "6fb43a315c3c5068a796e6df909b0fdfa9c0f7166ececd3d6f390eb7974c1040",
    "rlcd/weight_space/structured.py": "36f12528efc396693f2a91d9e8be002dc0ef530c9613d9f9fa5b702af23ce4ab",
    "rlcd/weight_space/program.py": "cb42326989d3bcec9c52405e578f0cc58de9a3fa8835fd7d517321e230003bfd",
    "rlcd/weight_space/broker.py": "58a2bbb99b2dc3240e1cf7790528d85ab9b401bdee4c52eff73fdd2fe310c56c",
}
NATIVE_DTYPES = {"head.weight": "float32", "head.bias": "float32"}
WEIGHT_REQUEST = {"op": "weights", "tensor": "head.bias", "indices": [[j] for j in range(8)]}
_EVENT = "overall_correct_delta_ge_41_of_4096"
# Same-process live-cohort ownership; not durable identity or sandbox enforcement.
_CLAIMED_COHORTS: WeakSet[StagedCohort] = WeakSet()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_bytes(value: Any) -> bytes:
    def finite(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                finite(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                finite(child)
            return
        raise ValueError("Expected strict finite JSON")
    finite(value)
    return canonical_bytes(value)


def _parse(raw: bytes | str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError("Nonfinite JSON constant: " + value)

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    _json_bytes(value)
    return value


def _same(actual: Any, expected: Any) -> None:
    _require(_json_bytes(actual) == _json_bytes(expected), "Evidence differs from its frozen expectation")


def _hash(value: Any, size: int = 64) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(size) + "}", value) is not None,
             "Expected an exact lowercase hexadecimal identity")


def _plain(path: Path, *, directory: bool = False) -> None:
    info = path.lstat()
    _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 1024,
             "Links and reparse points are not admissible evidence")
    _require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
             "Expected an ordinary evidence directory or file")


def _float32(value: Any) -> float:
    _require(type(value) in (int, float), "A weight must be a number, not a boolean")
    number = float(value)
    _require(math.isfinite(number) and struct.unpack("!f", struct.pack("!f", number))[0] == number,
             "A weight must be finite and exactly float32-representable")
    return number


def _payload(policy: str, values: list[float], choice: dict[str, Any] | None) -> dict[str, Any]:
    if policy == "metadata-uniform":
        row, direction, confidence, code, abstain = choice["row"], choice["direction"], 0.05, "random_prior", False
    else:
        median = statistics.median(values)
        scale = max(1.4826 * statistics.median([abs(value - median) for value in values]), 0.25)
        scores = [abs(value - median) / scale for value in values]
        row = max(range(8), key=scores.__getitem__)
        abstain = scores[row] < 3
        direction = "bias_up" if values[row] > median else "bias_down"
        confidence = 0 if abstain else 0.5 if scores[row] < 6 else 0.9
        code = "below_threshold" if abstain else "outlier"
    return {"location": None if abstain else {"tensor": "head.bias", "index_prefix": [row]},
            "mechanism": "undetermined" if abstain else direction,
            "predicted_effect": _json_bytes({"event": _EVENT, "p": None if abstain else 0.5,
                                              "probe_id": "bias-unit-v1"}).decode("utf-8"),
            "confidence": confidence,
            "uncertainty": _json_bytes({"calibrated": False, "code": code}).decode("utf-8"),
            "abstain": abstain}


def _diagnosis(value: Any) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == {
        "location", "mechanism", "predicted_effect", "confidence", "uncertainty", "abstain"},
        "Unknown diagnosis schema")
    _require(type(value["abstain"]) is bool and type(value["confidence"]) in (int, float)
             and math.isfinite(value["confidence"]) and 0 <= value["confidence"] <= 1,
             "Invalid diagnosis confidence or abstention")
    for key in ("mechanism", "predicted_effect", "uncertainty"):
        _require(type(value[key]) is str and bool(value[key].strip()), "Missing diagnosis explanation")
    effect, uncertainty = _parse(value["predicted_effect"]), _parse(value["uncertainty"])
    _require(type(effect) is dict and set(effect) == {"event", "p", "probe_id"}
             and type(uncertainty) is dict and set(uncertainty) == {"calibrated", "code"},
             "Unknown nested diagnosis schema")
    _same(effect, {"event": _EVENT, "p": None if value["abstain"] else 0.5, "probe_id": "bias-unit-v1"})
    _require(uncertainty["calibrated"] is False and uncertainty["code"] in (
        "outlier", "random_prior", "below_threshold"), "Unknown uncertainty declaration")
    _same(value["predicted_effect"], _json_bytes(effect).decode("utf-8"))
    _same(value["uncertainty"], _json_bytes(uncertainty).decode("utf-8"))
    if value["abstain"]:
        _same(value["location"], None)
        _same(value["mechanism"], "undetermined")
        _same(value["confidence"], 0)
        _same(uncertainty["code"], "below_threshold")
    else:
        location = value["location"]
        _require(type(location) is dict and set(location) == {"tensor", "index_prefix"}
                 and location["tensor"] == "head.bias" and type(location["index_prefix"]) is list
                 and len(location["index_prefix"]) == 1 and type(location["index_prefix"][0]) is int
                 and 0 <= location["index_prefix"][0] < 8 and value["mechanism"] in ("bias_up", "bias_down"),
                 "Invalid ordinary bias location or mechanism")
    return _parse(_json_bytes(value))


def _repair(answer: dict[str, Any]) -> dict[str, Any]:
    row = answer["location"]["index_prefix"][0]
    return {"commands": [{"op": "add", "tensor": "head.bias", "block": [[row, row + 1]],
                           "values": [-2.0 if answer["mechanism"] == "bias_up" else 2.0]}],
            "intent": "common-repair-v1"}


def _repair_structure(parent: TensorArtifact, endpoint: TensorArtifact, row: Any, mechanism: Any) -> dict[str, Any]:
    """Numerical criteria for a prepared head; the caller establishes provenance."""
    result = {"status": "unusable", "reason": "invalid_diagnosis",
              "realized_native_delta": None, "realized_native_norm": None}
    if type(row) is not int or not 0 <= row < 8 or mechanism not in ("bias_up", "bias_down"):
        return result
    if not parent.compatible(endpoint):
        return result | {"reason": "interface_changed"}
    tensors = [artifact.to_dict()["tensors"] for artifact in (parent, endpoint)]
    try:
        before_weight, after_weight = [[_float32(value) for values in data["head.weight"] for value in values]
                                       for data in tensors]
        before_bias, after_bias = [[_float32(value) for value in data["head.bias"]] for data in tensors]
    except (ValueError, OverflowError):
        return result | {"reason": "not_native_float32"}
    delta = after_bias[row] - before_bias[row]  # Native values are promoted before subtraction.
    result["realized_native_delta"] = delta if delta != 0 else 0.0
    outside = list(zip(before_weight, after_weight)) + [
        (before, after) for index, (before, after) in enumerate(zip(before_bias, after_bias)) if index != row]
    if any(struct.pack("!f", before) != struct.pack("!f", after) for before, after in outside):
        return result | {"reason": "outside_intended_support"}
    result["realized_native_norm"] = abs(delta)
    if delta == 0:
        return result | {"reason": "no_op"}
    if (mechanism == "bias_up" and delta > 0) or (mechanism == "bias_down" and delta < 0):
        return result | {"reason": "wrong_direction"}
    return result | {"status": "usable", "reason": "common_repair_structure"}


def _broker_trace(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    events = []
    for request, response in pairs:
        body = {"schema": "wsp-broker-event/1", "sequence": len(events),
                "previous_sha256": events[-1]["sha256"] if events else None,
                "request": request, "response": response}
        events.append(body | {"sha256": digest(body)})
    return events


def _trace(task: TaskSpec, config: dict[str, Any], candidate: Candidate | None,
           *, values: list[float] | None = None, proposal: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    usage, events = BudgetUsage().to_dict(), []

    def add(action: str, request: Any, result: Any, charges: dict[str, int]) -> None:
        for key, count in charges.items():
            usage[key] += count
        body = {"schema": "wsp-trace-event/1", "sequence": len(events),
                "previous_sha256": events[-1]["sha256"] if events else None,
                "session_sha256": digest(config), "task_sha256": task.sha256,
                "action": action, "status": "succeeded", "request": request, "result": result,
                "charges": charges, "requested_charges": charges, "usage": dict(usage), "error": None}
        events.append(body | {"sha256": digest(body)})

    add("session_created", config, {}, {})
    if values is not None:
        add("weights", {"tensor": "head.bias", "weight_decimals": None, "indices": [[j] for j in range(8)]},
            values, {"tool_calls": 1, "weight_values": 8})
    if candidate is not None:
        add("submit_answer" if candidate.artifact is None else "propose_artifact", proposal or {},
            {"candidate_sha256": candidate.sha256,
             "artifact_sha256": candidate.artifact.sha256 if candidate.artifact is not None else None,
             "no_op": candidate.no_op}, {"tool_calls": 1, "proposals": 1})
    add("seal", {"candidate_sha256": candidate.sha256 if candidate is not None else None},
        {"selection": "candidate" if candidate is not None else "abstained"}, {})
    return events


class SelectionAdmission:
    """One fresh cohort, three normal public returns, and retained exact evidence.

    ``prepare_member`` returns a PreparedMember; this driver supplies the fixed
    policy view, budgets, and native dtypes. Supplied metadata choices are frozen
    inputs, not evidence that a randomization procedure was performed.
    """

    def __init__(self, cohort: StagedCohort, run_dir: str | Path, *, assignments: list[dict[str, Any]],
                 protocol: dict[str, Any], registry_id: str, evaluator_id: str, attempt_id: str,
                 driver_sha256: str, controls: dict[str, Any] | None = None):
        self._invalidated, self._started = False, False
        self._cohort, self._directory = cohort, Path(run_dir).absolute()
        self._saved: dict[str, bytes] = {}
        self._returns: dict[str, bytes] = {}
        self._bases: dict[int, TensorArtifact] = {}
        self._artifacts: dict[int, TensorArtifact] = {}
        self._artifact_bytes: dict[tuple[str, int], bytes] = {}
        self._recipes: dict[int, dict[str, Any]] = {}
        self._preparation_errors: dict[int, str] = {}
        self._states: dict[str, dict[int, dict[str, Any]]] = {"diagnosis": {}, "repair": {}}
        self._tasks: dict[int, TaskSpec] = {}
        self._lanes: dict[int, dict[str, Any]] = {}
        self._diagnoses: dict[int, dict[str, Any]] = {}
        self._evidence: bytes | None = None
        try:
            _require(type(cohort) is StagedCohort, "Use an exact public StagedCohort")
            _require(cohort not in _CLAIMED_COHORTS, "This live cohort already has an admission owner")
            _CLAIMED_COHORTS.add(cohort)
            self._assignments = _parse(_json_bytes(assignments))
            _require(type(self._assignments) is list and bool(self._assignments), "Freeze a nonempty assignment roster")
            for item in self._assignments:
                _require(type(item) is dict and set(item) == {"member_id", "policy", "metadata_choice"}
                         and type(item["member_id"]) is str and bool(item["member_id"].strip())
                         and item["policy"] in ("bias-outlier", "metadata-uniform"), "Invalid assignment")
                choice = item["metadata_choice"]
                if item["policy"] == "bias-outlier":
                    _same(choice, None)
                else:
                    _require(type(choice) is dict and set(choice) == {"row", "direction"}
                             and type(choice["row"]) is int and 0 <= choice["row"] < 8
                             and choice["direction"] in ("bias_up", "bias_down"), "Invalid frozen metadata choice")
            self._rows = [{"member_index": index, "member_id": item["member_id"]}
                          for index, item in enumerate(self._assignments)]
            _require(len({row["member_id"] for row in self._rows}) == len(self._rows), "Duplicate member assignment")
            controls = _parse(_json_bytes({} if controls is None else controls))
            _require(type(controls) is dict, "Control recipes must be a finite JSON object")
            if controls:
                _require(set(controls) == {"schema", "members"}
                         and controls["schema"] == "wsp-native-control-recipes/1"
                         and type(controls["members"]) is list
                         and len(controls["members"]) == len(self._rows), "Unknown or incomplete control recipes")
                for row, recipe in zip(self._rows, controls["members"]):
                    _require(type(recipe) is dict and set(recipe) == {
                        "member_index", "member_id", "random", "shuffles"}, "Unknown member control recipe")
                    _same({key: recipe[key] for key in row}, row)
                    choice = recipe["random"]
                    _require(type(choice) is dict and set(choice) == {"row", "direction"}
                             and type(choice["row"]) is int and 0 <= choice["row"] < 8
                             and choice["direction"] in ("bias_up", "bias_down"), "Invalid supplied random control")
                    _require(type(recipe["shuffles"]) is list and len(recipe["shuffles"]) == 3,
                             "Supply exactly three row derangements")
                    for mapping in recipe["shuffles"]:
                        _require(type(mapping) is list and len(mapping) == 8
                                 and all(type(value) is int for value in mapping)
                                 and set(mapping) == set(range(8))
                                 and all(source != target for source, target in enumerate(mapping)),
                                 "A shuffle must derange exactly the eight eligible rows")
            self._controls_bytes = _json_bytes(controls)
            supplied = _parse(_json_bytes(protocol))
            _require(type(supplied) is dict and "evaluation_plan" not in supplied, "This increment admits selection only")
            for identifier in (registry_id, evaluator_id, attempt_id):
                _require(type(identifier) is str and bool(identifier.strip()), "Missing expected run identity")
            _hash(driver_sha256)
            self._sources = dict(SOURCE_PINS) | {"scripts/staged_diagnosis_selection.py": driver_sha256}
            self._source_root = Path(__file__).absolute().parents[1]
            for parent in (self._directory, *self._directory.parents):
                _plain(parent, directory=True)
            self._protocol = self._receipt("protocol.json", {
                "schema": "wsp-staged-cohort-protocol/1", "attempt_id": attempt_id,
                "registry_id": registry_id, "evaluator_id": evaluator_id, "members": self._rows,
                "roster_sha256": digest(self._rows), "protocol": supplied, "controls": controls,
                "controls_sha256": digest(controls),
                "selection_rule": "all_durable_terminal_diagnoses_before_any_repair_construction",
                "evaluation_available": False, "causal_usefulness_established": False})
            self._check()
            self._layout()
        except BaseException:
            self._invalidated = True
            raise

    def _receipt(self, name: str, expected: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self._directory / name
        _plain(path)
        raw = path.read_bytes()
        if name in self._saved:
            _require(raw == self._saved[name], "Previously accepted receipt bytes changed")
        document = _parse(raw)
        _require(type(document) is dict and "sha256" in document, "Missing receipt body digest")
        _hash(document["sha256"])
        body = {key: value for key, value in document.items() if key != "sha256"}
        _require(raw == _json_bytes(document) + b"\n" and digest(body) == document["sha256"],
                 "Receipt is noncanonical or has an invalid body digest")
        if expected is not None:
            _same(body, expected)
        self._saved[name] = raw
        return document

    def _check(self) -> None:
        try:
            _require(not self._invalidated, "This live admission is irreversibly ineligible")
            for name, expected in self._sources.items():
                path = self._source_root / name
                _plain(path)
                _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Reviewed source bytes changed")
            for name, raw in self._saved.items():
                path = self._directory / name
                _plain(path)
                _require(path.read_bytes() == raw, "Previously accepted receipt bytes changed")
            for (kind, index), raw in self._artifact_bytes.items():
                artifact = self._bases[index] if kind == "parent" else self._artifacts[index]
                _require(_json_bytes(artifact.to_dict()) == raw and artifact.sha256 == digest(_parse(raw)),
                         "A driver-owned artifact changed")
        except BaseException:
            self._invalidated = True
            raise

    def _layout(self) -> None:
        expected = set(self._saved)
        directories = {str(Path(name).parent).replace("\\", "/") for name in expected if "/" in name}
        _plain(self._directory, directory=True)
        actual = set()
        for path in self._directory.iterdir():
            if path.name in directories:
                _plain(path, directory=True)
                for child in path.iterdir():
                    _plain(child)
                    actual.add(path.name + "/" + child.name)
            else:
                _plain(path)
                actual.add(path.name)
        _require(actual == expected, "Unexpected, missing, or prior-phase run evidence")

    def _common(self) -> dict[str, Any]:
        return {key: self._protocol[key] for key in (
            "attempt_id", "registry_id", "evaluator_id", "roster_sha256")} | {
            "cohort_protocol_sha256": self._protocol["sha256"]}

    def _disposition(self, stage: str, row: dict[str, Any], outcome: str, **fields: Any) -> dict[str, Any]:
        return {"schema": "wsp-staged-cohort-disposition/1", "stage": stage, **row,
                "cohort_protocol_sha256": self._protocol["sha256"],
                "roster_sha256": self._protocol["roster_sha256"], "outcome": outcome, **fields}

    def _policy(self, index: int) -> tuple[WeightView, BudgetLimits]:
        outlier = self._assignments[index]["policy"] == "bias-outlier"
        view = WeightView("subset", coordinates=tuple(("head.bias", (j,)) for j in range(8))) if outlier else WeightView()
        return view, BudgetLimits(tool_calls=2 if outlier else 1, weight_values=8 if outlier else 0,
                                  proposals=1, feedback_calls=0, feedback_examples=0, execution_calls=0,
                                  execution_examples=0, input_tokens=0, output_tokens=0)

    def _config(self, index: int, task: TaskSpec) -> dict[str, Any]:
        view, limits = self._policy(index)
        return {"schema": "wsp-session-config/1", "task_sha256": task.sha256, "task_id": task.task_id,
                "mode": task.mode, "objective": task.objective, "architecture": task.architecture,
                "base_sha256": task.base_sha256, "constraints_sha256": digest(task.constraints),
                "access": AccessPolicy(weights=view.mode != "none").to_dict(), "limits": limits.to_dict(),
                "callbacks_registered": {"feedback": False, "forward": False, "activations": False},
                "splits": validate_splits(()).to_dict(), "data_roles": {
                    "registry_id": self._protocol["registry_id"], "scope": "registered_cooperative",
                    "provenance_authenticated": False, "semantic_independence_verified": False},
                "enforcement": "cooperative_same_process_interface", "declared_constraints_automatically_enforced": False,
                "execution_accounting": "interface_calls_and_submitted_examples; callback_internal_work_not_observed",
                "token_accounting": "provider_supplied_not_locally_measured"}

    def _prepare(self, callback: Callable[..., Any], row: dict[str, Any]) -> PreparedMember:
        index = len(self._recipes) + len(self._preparation_errors)
        try:
            self._check()
            _same(row, self._rows[index])
            try:
                supplied = callback(_parse(_json_bytes(row)))
            except BaseException as error:
                self._preparation_errors[index] = type(error).__name__
                raise
            try:
                _require(type(supplied) is PreparedMember and type(supplied.base) is TensorArtifact,
                         "Preparation must return an exact tensor PreparedMember")
                data = _parse(_json_bytes(supplied.base.to_dict()))
                base = TensorArtifact.from_dict(data)
                _same(base.to_dict(), data)
                _require(base.architecture == "trained-frozen-readout/1"
                         and set(data["tensors"]) == {"head.weight", "head.bias"}, "Wrong trained-head interface")
                weight, bias = data["tensors"]["head.weight"], data["tensors"]["head.bias"]
                _require(type(weight) is list and len(weight) == 9 and all(type(r) is list and len(r) == 32 for r in weight)
                         and type(bias) is list and len(bias) == 9, "Wrong trained-head shapes")
                for value in [v for r in weight for v in r] + bias:
                    _float32(value)
                _require(all(v == 0 for v in weight[8]) and bias[8] == -64, "Reserved output row changed")
                for key in ("encoder_sha256", "preprocessing_sha256", "feature_id"):
                    _hash(data["metadata"].get(key))
                for text in (supplied.task_id, supplied.objective):
                    _require(type(text) is str and bool(text.strip()), "Missing prepared task identity")
                preparation = _parse(_json_bytes(supplied.preparation))
                _require(type(preparation) is dict, "Preparation evidence must be a finite JSON object")
                self._bases[index] = base
                self._artifact_bytes[("parent", index)] = _json_bytes(data)
                self._recipes[index] = {"task_id": supplied.task_id, "objective": supplied.objective,
                                        "preparation": preparation}
                view, limits = self._policy(index)
                return PreparedMember(base, supplied.task_id, supplied.objective,
                                      _parse(_json_bytes(preparation)), view, limits, dict(NATIVE_DTYPES))
            except BaseException:
                self._invalidated = True
                raise
        except BaseException:
            if index not in self._preparation_errors:
                self._invalidated = True
            raise

    def _preparations(self, returned: Any) -> list[dict[str, Any]]:
        _require(type(returned) is list and len(returned) == len(self._rows), "Incomplete preparation return")
        result = []
        for row in self._rows:
            index = row["member_index"]
            if index in self._preparation_errors:
                expected = self._disposition("preparation", row, "unavailable", lane=None,
                                             exception_type=self._preparation_errors[index])
            else:
                recipe, base = self._recipes[index], self._bases[index]
                lane = self._receipt(f"member-{index:06d}/protocol.json")
                _hash(lane.get("run_id"), 32)
                view, limits = self._policy(index)
                binding = {"cohort_protocol_sha256": self._protocol["sha256"],
                           "cohort_roster_sha256": self._protocol["roster_sha256"],
                           "cohort_member_index": index, "cohort_member_id": row["member_id"]}
                body = {"schema": "wsp-staged-diagnosis-protocol/1", "run_id": lane["run_id"],
                        "task_id": recipe["task_id"], "objective": recipe["objective"], "architecture": base.architecture,
                        "parent_sha256": base.sha256, "registry_id": self._protocol["registry_id"],
                        "parameter_view": view.to_dict(), "limits": limits.to_dict(), "native_dtypes": NATIVE_DTYPES,
                        "cohort_binding": binding,
                        "diagnosis_fields": ["location", "mechanism", "predicted_effect", "confidence", "uncertainty", "abstain"],
                        "repair_fields": ["commands", "intent"],
                        "selection_rule": "sealed_durable_diagnosis_before_separate_repair_session",
                        "intervention_gate": "both_exact_durable_terminal_selections_verify_before_registered_authorization",
                        "supersedes_for_this_lane_only": "single_diagnosis_plus_intervention_submission",
                        "frozen_joint_selection_protocols_unchanged": True,
                        "enforcement": "cooperative_JSON_interface_and_exact_receipt_binding", "causal_usefulness_established": False}
                _same(lane, body | {"sha256": digest(body)})
                self._lanes[index] = lane
                task = TaskSpec(recipe["task_id"] + "/diagnosis", "diagnose", recipe["objective"], base.architecture,
                                base.sha256, {"staged_protocol_sha256": lane["sha256"], "selection_rule": "diagnosis_only"})
                self._tasks[index] = task
                config = self._config(index, task)
                expected = self._disposition("preparation", row, "prepared", preparation=recipe["preparation"], lane={
                    "lane_protocol_sha256": lane["sha256"], "parent_sha256": base.sha256,
                    "diagnosis_task_sha256": task.sha256, "diagnosis_session_sha256": digest(config), "configuration": config})
            document = self._receipt(f"preparation-{index:06d}.json", expected)
            _same(document, returned[index])
            result.append(document)
        return result

    def _callback(self, stage: str, row: dict[str, Any], query: Callable[..., Any], answer: Any = None) -> Any:
        states = self._states[stage]
        eligible = list(self._bases) if stage == "diagnosis" else [i for i in self._bases if self._diagnoses[i]["outcome"] == "selected"]
        index = eligible[len(states)]
        state = {"response": None, "queried": False, "blocked": False, "payload": None, "exception_type": None}
        states[index] = state
        try:
            self._check()
            _same(row, self._rows[index])
            if stage == "repair":
                _same(_diagnosis(answer), self._diagnoses[index]["candidate"]["answer"])
                payload = _repair(answer)  # No repair queries, including metadata.
            else:
                assignment = self._assignments[index]
                values = []
                if assignment["policy"] == "bias-outlier":
                    state["queried"] = True
                    try:
                        response = query(_parse(_json_bytes(WEIGHT_REQUEST)))
                    except BaseException:
                        state["blocked"] = True
                        raise
                    state["response"] = _parse(_json_bytes(response))
                    if _json_bytes(response) == _json_bytes({"status": "denied"}):
                        state["blocked"] = True
                        raise ValueError("Policy read denied")
                    _require(type(response) is dict and set(response) == {"status", "values"}
                             and response["status"] == "ok" and type(response["values"]) is list
                             and len(response["values"]) == 8, "Malformed policy read")
                    values = [_float32(value) for value in response["values"]]
                payload = _diagnosis(_payload(assignment["policy"], values, assignment["metadata_choice"]))
            state["payload"] = _json_bytes(payload)
            return _parse(state["payload"])
        except BaseException as error:
            state["exception_type"] = type(error).__name__
            if not state["blocked"]:
                self._invalidated = True
            raise

    def _failure_traces(self, index: int, stage: str, document: dict[str, Any], task: TaskSpec,
                        config: dict[str, Any], state: dict[str, Any], submitted: Any,
                        attempted: Candidate | None) -> None:
        trace = document["trace"]
        _require(type(trace) is list and 2 <= len(trace) <= 4
                 and verify_trace(trace, expected_head=document["trace_head_sha256"]), "Invalid failed-stage trace")
        empty = _trace(task, config, None)
        _same(trace[0], empty[0])
        _same({key: trace[-1][key] for key in ("action", "status", "request", "result", "charges", "requested_charges", "error")},
              {key: empty[-1][key] for key in ("action", "status", "request", "result", "charges", "requested_charges", "error")})
        allowed = ["weights"] if state["queried"] else []
        if submitted is not None:
            allowed += ["submit_answer" if stage == "diagnosis" else "propose_artifact"]
        actions = [event["action"] for event in trace[1:-1]]
        _require(len(actions) == len(set(actions)) and actions == [a for a in allowed if a in actions], "Unexpected failed-stage operation")
        keys, proposal_succeeded = set(empty[0]), False
        for event in trace:
            _require(type(event) is dict and set(event) == keys and type(event["sequence"]) is int
                     and event["schema"] == "wsp-trace-event/1" and event["session_sha256"] == digest(config)
                     and event["task_sha256"] == task.sha256, "Wrong failed-stage identity or schema")
            for key, count in event["usage"].items():
                _require(type(count) is int and 0 <= count <= config["limits"][key], "Failed-stage budget exceeded")
            if event["action"] in ("session_created", "seal"):
                continue
            charges = {"tool_calls": 1, "weight_values": 8} if event["action"] == "weights" else {"tool_calls": 1, "proposals": 1}
            _same(event["requested_charges"], charges)
            _require(event["status"] in ("succeeded", "denied", "failed", "rejected"), "Unknown failed operation status")
            _same(event["charges"], {} if event["status"] in ("denied", "rejected") else charges)
            if event["action"] == "weights":
                _same(event["request"], {"tensor": "head.bias", "weight_decimals": None, "indices": [[j] for j in range(8)]})
            elif event["action"] == "submit_answer":
                _same(event["request"], {})
            else:
                request = event["request"]
                expected_request = {"parser": "typed_artifact_or_declared_schema", "proposal_type": "TensorArtifact"}
                if event["status"] == "succeeded":
                    _require(attempted is not None, "Missing authentic attempted artifact")
                    expected_request["artifact_sha256"] = attempted.artifact.sha256
                elif type(request) is dict and "artifact_sha256" in request:
                    expected_request["artifact_sha256"] = self._artifacts[index].sha256
                _same(request, expected_request)
            if event["status"] == "succeeded":
                _same(event["error"], None)
                if event["action"] == "weights":
                    _same(event["result"], self._bases[index].to_dict()["tensors"]["head.bias"][:8])
                else:
                    _require(attempted is not None, "A successful proposal must retain its exact attempted Candidate")
                    _same(event["result"], {"candidate_sha256": attempted.sha256,
                          "artifact_sha256": attempted.artifact.sha256 if attempted.artifact is not None else None,
                          "no_op": attempted.no_op})
                    proposal_succeeded = True
            else:
                error = event["error"]
                _require(type(error) is dict and set(error) == {"type", "message"}
                         and all(type(text) is str for text in error.values()), "Missing failed-operation error")
                _same(event["result"], None)
        _require(proposal_succeeded == (attempted is not None), "Attempted Candidate and session operation disagree")
        broker = document["broker_trace"]
        _require(type(broker) is list and 1 <= len(broker) <= 4, "Invalid failed broker trace")
        pairs = []
        read_seen, submission_seen, seal_seen = False, False, False
        attempted_handle = None
        handle = document["selected_handle"]
        _require((handle is None) == (document["submission"] is None), "Failed handle and submission disagree")
        if handle is not None:
            _hash(handle, 48)
        for event in broker[:-1]:
            _require(type(event) is dict and "request" in event and "response" in event, "Malformed broker event")
            request, response = event["request"], event["response"]
            _require(type(response) is dict, "Malformed broker response")
            if _json_bytes(request) == _json_bytes(WEIGHT_REQUEST):
                _require(state["queried"] and not pairs, "Unexpected broker weight read")
                read_seen = True
                _require(_json_bytes(response) == _json_bytes({"status": "denied"})
                         or response.get("status") == "ok", "Unknown broker read response")
                if response.get("status") == "ok":
                    _same(response, {"status": "ok", "values": self._bases[index].to_dict()["tensors"]["head.bias"][:8]})
                    _require(any(e["action"] == "weights" and e["status"] == "succeeded" for e in trace),
                             "Broker read lacks its successful session operation")
                else:
                    _require(not any(e["action"] == "weights" and e["status"] == "succeeded" for e in trace),
                             "Broker denial contradicts a successful session read")
                if state["response"] is not None:
                    _same(response, state["response"])
            elif _json_bytes(request) == _json_bytes(submitted):
                _same(request, submitted)
                _require(submitted is not None and not submission_seen and not seal_seen, "Repeated failed broker submission")
                submission_seen = True
                if response.get("status") == "submitted":
                    attempted_handle = response.get("candidate_handle")
                    _hash(attempted_handle, 48)
                    _same(response, {"status": "submitted", "candidate_handle": attempted_handle})
                    if handle is not None:
                        _same(attempted_handle, handle)
                else:
                    _same(response, {"status": "denied"})
                    _same(handle, None)
                    _require(attempted is None, "A denied broker proposal cannot create a Candidate")
            else:
                _require(submitted is not None and submission_seen and not seal_seen, "Unexpected failed broker operation")
                _same(request, {"op": "seal", "candidate_handle": attempted_handle})
                _same(response, {"status": "denied"})
                seal_seen = True
            pairs.append((request, response))
        if state["response"] is not None:
            _require(read_seen, "Missing observed broker response")
        terminal = broker[-1]["request"]
        if handle is None and _json_bytes(terminal) == _json_bytes({"op": "harness_deadline_seal"}):
            if attempted is not None:
                _require(attempted_handle is not None and submission_seen and seal_seen,
                         "An unselected successful proposal requires its exact denied seal")
                _same(document["submission"], None)
        else:
            _same(terminal, {"op": "seal", "candidate_handle": handle})
            _require(submitted is not None and submission_seen and not seal_seen, "Unexpected failed selection seal")
            _same(handle, attempted_handle)
            _require(attempted is None, "A failed selection cannot seal a successful Candidate")
        pairs.append((terminal, {"status": "sealed"}))
        _same(broker, _broker_trace(pairs))

    def _selection(self, index: int, stage: str, task: TaskSpec) -> dict[str, Any]:
        document = self._receipt(f"member-{index:06d}/{stage}.json")
        state, base, config = self._states[stage][index], self._bases[index], self._config(index, task)
        outcome = document["outcome"]
        _require(outcome in ("selected", "abstained", "selector_failed", "invalid_submission"), "Unknown selection outcome")
        _require(type(document["candidates"]) is list and len(document["candidates"]) <= 1,
                 "The fixed policy permits at most one attempted Candidate")
        payload = _parse(state["payload"]) if state["payload"] is not None else None
        candidate, attempted, endpoint, expected_submission = None, None, None, None
        submitted = None if payload is None else ({"op": "answer", "answer": payload} if stage == "diagnosis" else {"op": "submit", **payload})
        selected = outcome in ("selected", "abstained")
        if payload is not None:
            if stage == "diagnosis":
                assignment = self._assignments[index]
                _same(_diagnosis(payload), _payload(assignment["policy"], base.to_dict()["tensors"]["head.bias"][:8], assignment["metadata_choice"]))
            else:
                _same(payload, _repair(self._diagnoses[index]["candidate"]["answer"]))
                proposal = StructuredProgram(base.sha256, payload["commands"], payload["intent"])
                lowered = lower_structured(base, proposal, native_dtypes=NATIVE_DTYPES, cast="round")
                _require(verify_lowering(base, proposal, lowered.receipt), "Repair lowering does not verify")
                program = Program.from_dict(lowered.program.to_dict())
                _same(program.to_dict(), lowered.program.to_dict())
                endpoint = apply_program(base, program)
                _require(type(endpoint) is TensorArtifact, "Repair changed the artifact representation")
                parsed = TensorArtifact.from_dict(endpoint.to_dict())
                _same(parsed.to_dict(), endpoint.to_dict())
                self._artifacts[index] = parsed
                self._artifact_bytes[("repair", index)] = _json_bytes(parsed.to_dict())
        if selected or document["candidates"]:
            _require(state["exception_type"] is None and not state["blocked"] and payload is not None,
                     "A denied or failed callback cannot supply a successful Candidate")
            if stage == "diagnosis":
                if selected:
                    _same(outcome, "abstained" if payload["abstain"] else "selected")
                expected_submission = {"submitted": submitted, "status": "lowered"}
            else:
                if selected:
                    _same(outcome, "selected")
                expected_submission = {"submitted": submitted, "program": program.to_dict(),
                                       "lowering": lowered.receipt, "status": "lowered"}
            attempted = Candidate(session_sha256=digest(config), task_sha256=task.sha256,
                                  parent_sha256=base.sha256, proposal_index=1,
                                  kind="answer" if stage == "diagnosis" else "artifact",
                                  artifact=endpoint, answer=payload if stage == "diagnosis" else None)
        if selected:
            candidate = attempted
            handle = document["selected_handle"]
            _hash(handle, 48)
            values = base.to_dict()["tensors"]["head.bias"][:8] if state["queried"] else None
            if values is not None:
                _same(state["response"], {"status": "ok", "values": values})
            trace = _trace(task, config, candidate, values=values,
                           proposal=None if stage == "diagnosis" else {"parser": "typed_artifact_or_declared_schema",
                               "proposal_type": "TensorArtifact", "artifact_sha256": endpoint.sha256})
            pairs = [(WEIGHT_REQUEST, {"status": "ok", "values": values})] if values is not None else []
            pairs += [(submitted, {"status": "submitted", "candidate_handle": handle}),
                      ({"op": "seal", "candidate_handle": handle}, {"status": "sealed"})]
            broker = _broker_trace(pairs)
        else:
            if outcome == "selector_failed":
                _require(state["exception_type"] is not None and payload is None, "Unobserved selector failure")
                _same(document["selected_handle"], None)
                _same(document["submission"], None)
            else:
                _require(payload is not None and state["exception_type"] is None, "Unobserved invalid submission")
                private = document["submission"]
                if private is not None:
                    _require(type(private) is dict and set(private) == {"submitted", "status", "error_type", "error"}
                             and private["status"] == "invalid" and type(private["error_type"]) is str
                             and type(private["error"]) is str, "Invalid failure submission schema")
                    _same(private["submitted"], submitted)
                expected_submission = private
            self._failure_traces(index, stage, document, task, config, state, submitted, attempted)
            trace, broker = document["trace"], document["broker_trace"]
        _require(verify_trace(trace, expected_head=trace[-1]["sha256"]), "Selection trace does not verify")
        body = {"schema": "wsp-staged-selection/1", "stage": stage, "outcome": outcome,
                "protocol_sha256": self._lanes[index]["sha256"], "task": task.to_dict(), "task_sha256": task.sha256,
                "session_sha256": digest(config), "configuration": config, "selected_handle": document["selected_handle"],
                "candidate": candidate.to_dict() if candidate is not None else None, "submission": expected_submission,
                "candidates": [attempted.to_dict()] if attempted is not None else [], "trace": trace,
                "trace_head_sha256": trace[-1]["sha256"], "broker_trace": broker}
        _same(document, body | {"sha256": digest(body)})
        return document

    def _diagnosis_barrier(self, preparations: list[dict[str, Any]], returned: Any) -> dict[str, Any]:
        _same(list(self._states["diagnosis"]), list(self._bases))
        members = []
        for row, preparation in zip(self._rows, preparations):
            index = row["member_index"]
            actual = self._selection(index, "diagnosis", self._tasks[index]) if index in self._bases else None
            if actual is not None:
                self._diagnoses[index] = actual
            outcome = actual["outcome"] if actual is not None else "unavailable"
            fields = {"preparation_receipt_sha256": preparation["sha256"],
                      "diagnosis_receipt_sha256": actual["sha256"] if actual is not None else None}
            if actual is not None:
                fields["exception_type"] = self._states["diagnosis"][index]["exception_type"]
                if outcome == "invalid_submission":
                    fields["exception_type"] = self._receipt(f"diagnosis-{index:06d}.json").get("exception_type")
                    error = fields["exception_type"]
                    _require(error is None or (not actual["candidates"] and actual["submission"] is None
                             and type(error) is str and bool(error.strip())), "Invalid admission exception")
            disposition = self._receipt(f"diagnosis-{index:06d}.json", self._disposition("diagnosis", row, outcome, **fields))
            members.append({**row, **{key: fields[key] for key in ("preparation_receipt_sha256", "diagnosis_receipt_sha256")},
                            "diagnosis_disposition_sha256": disposition["sha256"], "outcome": outcome})
        barrier = self._receipt("diagnosis-barrier.json", {
            "schema": "wsp-staged-cohort-diagnosis-barrier/1", **self._common(), "members": members, "evaluation_available": False})
        _same(barrier, returned)
        return barrier

    def _repairs(self, barrier: dict[str, Any], returned: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        _require(type(returned) is list and len(returned) == len(self._rows), "Incomplete repair return")
        _same(list(self._states["repair"]), [i for i in self._bases if self._diagnoses[i]["outcome"] == "selected"])
        members, evidence = [], []
        for row, diagnosis, disposition in zip(self._rows, barrier["members"], returned):
            index = row["member_index"]
            fields = {key: diagnosis[key] for key in (
                "preparation_receipt_sha256", "diagnosis_disposition_sha256", "diagnosis_receipt_sha256")}
            fields.update(diagnosis_outcome=diagnosis["outcome"], diagnosis_barrier_sha256=barrier["sha256"])
            actual = None
            if diagnosis["outcome"] == "selected":
                selected = self._diagnoses[index]
                binding = {"staged_protocol_sha256": selected["protocol_sha256"], "diagnosis_receipt_sha256": selected["sha256"],
                           "diagnosis_candidate_sha256": selected["candidate"]["sha256"], "diagnosis_task_sha256": selected["task_sha256"],
                           "diagnosis_session_sha256": selected["session_sha256"], "diagnosis_seal_sha256": selected["trace_head_sha256"],
                           "selection_rule": "separate_repair_after_verified_durable_diagnosis", **self._lanes[index]["cohort_binding"],
                           "cohort_diagnosis_barrier_sha256": barrier["sha256"],
                           "cohort_preparation_receipt_sha256": diagnosis["preparation_receipt_sha256"],
                           "cohort_diagnosis_disposition_sha256": diagnosis["diagnosis_disposition_sha256"]}
                task = TaskSpec(self._tasks[index].task_id + "/repair", "edit", self._tasks[index].objective,
                                self._bases[index].architecture, self._bases[index].sha256, binding)
                actual = self._selection(index, "repair", task)
                error = self._states["repair"][index]["exception_type"]
                if actual["outcome"] == "invalid_submission":
                    error = disposition.get("exception_type")
                    _require(error is None or (not actual["candidates"] and actual["submission"] is None
                             and type(error) is str and bool(error.strip())), "Invalid repair admission exception")
                fields["exception_type"] = error
            outcome = actual["outcome"] if actual is not None else "unavailable"
            fields["repair_receipt_sha256"] = actual["sha256"] if actual is not None else None
            saved = self._receipt(f"repair-{index:06d}.json", self._disposition("repair", row, outcome, **fields))
            _same(saved, disposition)
            members.append({**row, **{key: value for key, value in fields.items()
                                     if key not in ("diagnosis_barrier_sha256", "exception_type")},
                            "outcome": outcome, "repair_disposition_sha256": saved["sha256"]})
            candidate = actual["candidate"] if actual is not None else None
            usable = candidate is not None and not candidate["no_op"]
            evidence.append({**row, "policy": self._assignments[index]["policy"], "diagnosis_outcome": diagnosis["outcome"],
                             "repair_outcome": outcome, "diagnosis_receipt_sha256": diagnosis["diagnosis_receipt_sha256"],
                             "repair_receipt_sha256": fields["repair_receipt_sha256"],
                             "parent": self._bases[index].to_dict() if index in self._bases else None,
                             "diagnosis_candidate": self._diagnoses[index]["candidate"] if index in self._diagnoses else None,
                             "repair_candidate": candidate, "repair_usable": usable,
                             "repair_usability_reason": "non_noop_common_repair" if usable else "no_op" if candidate is not None else outcome})
        repair_barrier = self._receipt("repair-barrier.json", {"schema": "wsp-staged-cohort-repair-barrier/1",
            **self._common(), "diagnosis_barrier_sha256": barrier["sha256"], "members": members})
        return repair_barrier, evidence

    def run(self, prepare_member: Callable[[dict[str, Any]], PreparedMember]) -> dict[str, Any]:
        try:
            _require(not self._started and not self._invalidated and callable(prepare_member), "Admission runs once")
            self._started = True
            self._check()
            returned = self._cohort.prepare(lambda row: self._prepare(prepare_member, row))
            self._returns["preparation"] = _json_bytes(returned)
            self._check()
            preparations = self._preparations(_parse(self._returns["preparation"]))
            self._layout()
            self._check()
            returned = self._cohort.select_diagnoses(lambda row, query: self._callback("diagnosis", row, query))
            self._returns["diagnosis"] = _json_bytes(returned)
            self._check()
            barrier = self._diagnosis_barrier(preparations, _parse(self._returns["diagnosis"]))
            self._layout()
            self._check()  # All diagnosis bytes are captured before any repair construction.
            returned = self._cohort.select_repairs(lambda row, answer, query: self._callback("repair", row, query, answer))
            self._returns["repair"] = _json_bytes(returned)
            self._check()
            repair_barrier, members = self._repairs(barrier, _parse(self._returns["repair"]))
            self._layout()
            self._evidence = _json_bytes({"schema": "wsp-staged-selection-admission/1", **self._common(),
                "assignments": self._assignments, "assignments_sha256": digest(self._assignments),
                "phase_return_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in self._returns.items()},
                "source_sha256": self._sources, "source_pin_scope": "reviewed_disk_format_compatibility_only",
                "diagnosis_barrier_sha256": barrier["sha256"], "repair_barrier_sha256": repair_barrier["sha256"],
                "receipt_files": [{"path": name, "body_sha256": _parse(raw)["sha256"],
                                   "file_sha256": hashlib.sha256(raw).hexdigest()} for name, raw in sorted(self._saved.items())],
                "members": members, "evaluation_available": False, "final_study_seal": False,
                "trained_ancestry_established": False, "causal_usefulness_established": False})
            return self.admit()
        except BaseException:
            self._invalidated = True
            raise

    def admit(self) -> dict[str, Any]:
        """Recheck retained evidence; changed bytes permanently close this object."""
        try:
            self._check()
            _require(set(self._returns) == {"preparation", "diagnosis", "repair"} and self._evidence is not None,
                     "All three normal public returns must survive in this originating live object")
            self._layout()
            return _parse(self._evidence)
        except BaseException:
            self._invalidated = True
            raise

    def check_repair_usability(self) -> dict[str, Any]:
        """Check selected common repairs structurally, without writing evidence.

        This supplies no fallback, control matching, final study seal, trained
        ancestry or behavioral result. Integrity failures still close admission.
        """
        try:
            before = self.admit()
            _same([{key: member[key] for key in ("member_index", "member_id")} for member in before["members"]], self._rows)
            members = []
            for member in before["members"]:
                index = member["member_index"]
                parent = self._bases.get(index)
                _require(parent is None or type(parent) is TensorArtifact, "A retained parent changed representation")
                _same(member["parent"], parent.to_dict() if parent is not None else None)
                candidate = member["repair_candidate"]
                _require((candidate is not None) == (member["repair_outcome"] == "selected"),
                         "Selected repair evidence disagrees with its terminal outcome")
                result = {key: member[key] for key in (
                    "member_index", "member_id", "policy", "diagnosis_outcome", "repair_outcome",
                    "diagnosis_receipt_sha256", "repair_receipt_sha256")}
                result.update(parent_sha256=parent.sha256 if parent is not None else None,
                              selected_repair_candidate_sha256=None, selected_repair_artifact_sha256=None,
                              intended_row=None, diagnosed_direction=None, prescribed_delta=None,
                              realized_native_delta=None, realized_native_norm=None,
                              status="unavailable", reason="no_selected_repair")
                if candidate is not None:
                    endpoint = self._artifacts.get(index)
                    _require(type(parent) is TensorArtifact and type(endpoint) is TensorArtifact
                             and member["diagnosis_outcome"] == "selected", "A selected repair lacks its owned endpoints or diagnosis")
                    diagnosis = member["diagnosis_candidate"]
                    _same(diagnosis, self._diagnoses[index]["candidate"])
                    answer = _diagnosis(diagnosis["answer"])
                    _require(not answer["abstain"], "A selected repair cannot follow abstention")
                    _same(candidate["parent_sha256"], parent.sha256)
                    _same(candidate["artifact_sha256"], endpoint.sha256)
                    _same(candidate["artifact"], endpoint.to_dict())
                    _same(candidate["sha256"], digest({key: value for key, value in candidate.items() if key != "sha256"}))
                    _require(candidate["kind"] == "artifact" and candidate["no_op"] is (parent.sha256 == endpoint.sha256),
                             "Selected Candidate identity or no-op binding changed")
                    row, mechanism = answer["location"]["index_prefix"][0], answer["mechanism"]
                    result.update(selected_repair_candidate_sha256=candidate["sha256"],
                                  selected_repair_artifact_sha256=endpoint.sha256,
                                  intended_row=row, diagnosed_direction=mechanism,
                                  prescribed_delta=-2.0 if mechanism == "bias_up" else 2.0)
                    result.update(_repair_structure(parent, endpoint, row, mechanism))
                members.append(result)
            body = {"schema": "wsp-common-repair-structure/1", "selection_admission_sha256": digest(before),
                    **{key: before[key] for key in ("attempt_id", "registry_id", "evaluator_id", "roster_sha256", "cohort_protocol_sha256")},
                    "members": members}
            report = body | {"sha256": digest(body)}
            _same(self.admit(), before)
            return _parse(_json_bytes(report))
        except BaseException:
            self._invalidated = True
            raise

    def check_native_controls(self) -> dict[str, Any]:
        """Materialize prebound comparison controls without sealing or scoring.

        Recipe version 1 fixes common-repair-v1, magnitude 2, float32 round
        lowering and the main-referenced sum tolerance. Supplied choices/maps
        establish no randomization independence or prospective registration.
        """
        try:
            before = self.admit()
            recipes = _parse(self._controls_bytes)
            _require(bool(recipes), "Native controls require recipes bound before run()")
            _same(recipes, _parse(self._saved["protocol.json"])["controls"])
            main_report = self.check_repair_usability()
            _same(main_report["selection_admission_sha256"], digest(before))
            members, owned = [], []
            for member, main, recipe in zip(before["members"], main_report["members"], recipes["members"]):
                index = member["member_index"]
                parent = self._bases.get(index)
                answer = None
                if member["diagnosis_outcome"] == "selected":
                    _require(type(parent) is TensorArtifact and member["diagnosis_candidate"] is not None,
                             "A selected diagnosis lacks its owned parent or Candidate")
                    _same(member["diagnosis_candidate"], self._diagnoses[index]["candidate"])
                    answer = _diagnosis(member["diagnosis_candidate"]["answer"])
                    _require(not answer["abstain"], "Diagnosis-dependent controls cannot follow abstention")
                reference = main["realized_native_norm"] if main["status"] == "usable" else None
                choices = [("random", recipe["random"]["row"], recipe["random"]["direction"])]
                for number, mapping in enumerate(recipe["shuffles"], 1):
                    choices.append((f"shuffled_{number}", mapping[answer["location"]["index_prefix"][0]] if answer else None,
                                    answer["mechanism"] if answer else None))
                choices.append(("opposite", answer["location"]["index_prefix"][0] if answer else None,
                                ("bias_down" if answer["mechanism"] == "bias_up" else "bias_up") if answer else None))
                controls = []
                for control_id, row, direction in choices:
                    control = {"control_id": control_id, "materialized": False,
                               "unavailable_reason": "no_parent" if parent is None else "no_selected_diagnosis",
                               "intended_row": row, "direction": direction,
                               "prescribed_delta": (-2.0 if direction == "bias_up" else 2.0) if direction else None,
                               "proposal_sha256": None, "program_sha256": None,
                               "lowering_sha256": None, "artifact_sha256": None,
                               "structure": {"status": "unavailable", "reason": "not_materialized",
                                             "realized_native_delta": None, "realized_native_norm": None},
                               "matching": {"status": "unavailable", "reason": "no_usable_main_reference",
                                            "norm_difference": None}}
                    if parent is not None and row is not None:
                        payload = _repair({"location": {"index_prefix": [row]}, "mechanism": direction})
                        proposal = StructuredProgram(parent.sha256, payload["commands"], payload["intent"])
                        lowered = lower_structured(parent, proposal, native_dtypes=NATIVE_DTYPES, cast="round")
                        receipt = lowered.receipt
                        _require(verify_lowering(parent, proposal, receipt), "Control lowering does not verify")
                        program = Program.from_dict(lowered.program.to_dict())
                        _same(program.to_dict(), lowered.program.to_dict())
                        endpoint = apply_program(parent, program)
                        _require(type(endpoint) is TensorArtifact, "Control changed the artifact representation")
                        artifact = TensorArtifact.from_dict(endpoint.to_dict())
                        _same(artifact.to_dict(), endpoint.to_dict())
                        _same(receipt["candidate_sha256"], artifact.sha256)
                        owned.append((artifact, _json_bytes(artifact.to_dict())))
                        control.update(materialized=True, unavailable_reason=None, proposal_sha256=proposal.sha256,
                                       program_sha256=program.sha256, lowering_sha256=receipt["sha256"],
                                       artifact_sha256=artifact.sha256,
                                       structure=_repair_structure(parent, artifact, row, direction))
                    if reference is not None:
                        matching = control["matching"]
                        matching["reason"] = "control_unavailable" if not control["materialized"] else "control_unusable"
                        if control["structure"]["status"] == "usable":
                            difference = abs(control["structure"]["realized_native_norm"] - reference)
                            matched = difference <= 1e-5 + 1e-5 * abs(reference)
                            matching.update(status="matched" if matched else "mismatched",
                                            reason="native_norm_match" if matched else "native_norm_mismatch",
                                            norm_difference=difference)
                    controls.append(control)
                members.append({"member_index": index, "member_id": member["member_id"],
                                "diagnosis_candidate_sha256": member["diagnosis_candidate"]["sha256"]
                                    if member["diagnosis_candidate"] is not None else None,
                                "main": main, "main_reference_norm": reference, "controls": controls})
            for artifact, raw in owned:
                _require(_json_bytes(artifact.to_dict()) == raw and artifact.sha256 == digest(_parse(raw)),
                         "A driver-owned control artifact changed during calculation")
            body = {"schema": "wsp-native-control-matching/1", "selection_admission_sha256": digest(before),
                    "repair_structure_sha256": main_report["sha256"], "controls_sha256": digest(recipes),
                    **{key: before[key] for key in ("attempt_id", "registry_id", "evaluator_id", "roster_sha256",
                        "cohort_protocol_sha256", "diagnosis_barrier_sha256", "repair_barrier_sha256",
                        "source_sha256", "source_pin_scope")},
                    "construction": {"recipe_schema": "wsp-native-control-recipes/1", "intent": "common-repair-v1",
                                     "prescribed_magnitude": 2.0, "native_dtypes": dict(NATIVE_DTYPES), "cast": "round"},
                    "norm_matching": {"reference": "usable_selected_main", "atol": 1e-5, "rtol": 1e-5,
                                      "rule": "abs(control-main)<=atol+rtol*abs(main)"},
                    "members": members, "evaluation_available": False, "final_study_seal": False,
                    "causal_usefulness_established": False}
            report = body | {"sha256": digest(body)}
            _same(self.admit(), before)
            return _parse(_json_bytes(report))
        except BaseException:
            self._invalidated = True
            raise
