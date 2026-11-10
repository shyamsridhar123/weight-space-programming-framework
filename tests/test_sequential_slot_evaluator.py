"""One authored integration check; no training, research loaders or study data."""
from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import pytest

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.data_roles import DataRoleError, DataRoleRegistry
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.sequential import sequential_artifact
from rlcd.weight_space.sequential_runtime import SequentialRuntime
from rlcd.weight_space.sequential_slot_evaluator import evaluate_sequential_slot
from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort
from rlcd.weight_space.workflow import WorkflowError


def test_authored_sequential_slots_and_fail_stop(tmp_path):
    execution = {"backend": "numpy", "dtype": "float64", "device": "cpu",
                 "accumulation": "left_to_right", "endpoint_cast": "reject",
                 "error_model": "empirical"}
    parent = sequential_artifact(
        weights=[[[1.0, 0.0], [0.0, 1.0]], [[2.0, -1.0]]],
        biases=[[0.0, 0.0], [3.0]], activations=["relu", "identity"],
        input_names=("left", "right"), output_names=("score",),
    )
    repaired = TensorArtifact(parent.architecture,
        parent.tensors | {"layers.1.bias": [0.0]}, parent.metadata)
    diagnosis = {"location": {"tensor": "layers.1.bias", "index_prefix": [0]},
                 "mechanism": "An authored additive +3 output offset",
                 "predicted_effect": "Removing the offset lowers every output by three",
                 "confidence": 1.0, "uncertainty": "Handcrafted engineering fixture only",
                 "abstain": False}
    literal_rows = [
        {"case": "first", "inputs": {"right": 2, "left": 1}},
        {"case": "second", "inputs": {"left": 3, "right": 1}},
        {"case": "negative", "inputs": {"right": 2, "left": -1}},
    ]
    ordered_inputs = [[1.0, 2.0], [3.0, 1.0], [-1.0, 2.0]]

    def read(path):
        document = json.loads(path.read_bytes())
        assert document["sha256"] == digest({key: value for key, value in document.items()
                                             if key != "sha256"})
        return document

    def completed(name, rows):
        directory = tmp_path / name
        registry = DataRoleRegistry(tmp_path / (name + "-roles.sqlite"))
        batch = registry.register_batch(rows, source_id="authored-rows", family_id="authored-family")
        slots = [{"slot_id": identifier, "source": source, "execution": deepcopy(execution),
                  "result_keys": ["sequential"]} for identifier, source in [
            ("repaired", {"kind": "selected", "fallback_control_id": None}),
            ("parent", {"kind": "control", "control_id": "parent"}),
        ]]
        plan = {"schema": "wsp-staged-cohort-evaluation-plan/1",
                "result_schema": "wsp-staged-cohort-result/1",
                "aggregate": {"source_id": "authored-aggregate", "family_id": "aggregate-family"},
                "holdouts": [{"holdout_id": "authored", "source_id": "authored-rows",
                              "family_id": "authored-family", "row_count": len(rows)}],
                "members": [{"member_id": "authored", "holdout_id": "authored", "slots": slots}]}
        cohort = StagedCohort(directory, member_ids=["authored"],
            protocol={"evaluation_plan": plan},
            controls={"authored": {"parent": {"kind": "parent"}}},
            evaluator_id="authored-sequential-evaluator", registry=registry, attempt_id=name)
        cohort.prepare(lambda row: PreparedMember(parent, name + "/authored",
            "Remove the authored output offset after a separate diagnosis", {}))
        events = []

        def authored_diagnosis(row, query):
            assert row["member_id"] == "authored" and events == []
            events.append("diagnosis")
            return deepcopy(diagnosis)

        barrier = cohort.select_diagnoses(authored_diagnosis)

        def authored_repair(row, frozen_diagnosis, query):
            assert events == ["diagnosis"] and frozen_diagnosis == diagnosis
            assert read(directory / "diagnosis-barrier.json") == barrier
            events.append("repair")
            return {"commands": [{"op": "add", "tensor": "layers.1.bias", "values": [-3.0]}],
                    "intent": "Remove the authored additive output offset"}

        selections = cohort.select_repairs(authored_repair)
        assert events == ["diagnosis", "repair"] and selections[0]["outcome"] == "selected"
        return cohort, registry, batch, cohort.seal_endpoints()

    cohort, registry, batch, seal = completed("success", literal_rows)
    with patch.object(SequentialRuntime, "forward", autospec=True,
                      side_effect=SequentialRuntime.forward) as forward:
        terminal = cohort.evaluate_registered(lambda: {"authored": batch}, evaluate_sequential_slot)
        assert forward.call_count == 2
        assert [call.args[1] for call in forward.call_args_list] == [ordered_inputs, ordered_inputs]
        assert all(type(value) is float for call in forward.call_args_list
                   for row in call.args[1] for value in row)
    assert terminal == read(tmp_path / "success" / "evaluation.json")
    assert terminal["outcome"] == "succeeded"
    result = terminal["result"]
    attempt = registry.get_attempt("success")
    assert terminal["registry_attempt"] == terminal["data_role_decision"]["attempt"] == attempt
    assert attempt["status"] == "succeeded" and attempt["callback_invocations"] == 1
    assert attempt["result"] == result and attempt["result_sha256"] == digest(result)
    assert result["endpoint_seal_sha256"] == seal["sha256"]
    expected_hashes = [repaired.sha256, parent.sha256]
    expected_predictions = [[[0.0], [5.0], [-2.0]], [[3.0], [8.0], [1.0]]]
    slots = result["members"][0]["slots"]
    for index, slot in enumerate(slots):
        sealed = seal["members"][0]["slots"][index]
        assert slot["endpoint_sha256"] == sealed["endpoint_sha256"] == expected_hashes[index]
        assert slot["outcome"] == "succeeded" and slot["evaluator_invocations"] == 1
        assert type(slot["result"]) is dict and set(slot["result"]) == {"sequential"}
        evaluation = slot["result"]["sequential"]
        assert evaluation["schema"] == "wsp-sequential-slot-evaluation/1"
        assert evaluation["rows_sha256"] == digest(literal_rows)
        assert evaluation["ordered_inputs_sha256"] == digest({
            "input_names": ["left", "right"], "inputs": ordered_inputs})
        assert evaluation["predictions"] == expected_predictions[index]
        assert type(evaluation["predictions"]) is list
        assert all(type(row) is list and all(type(value) is float for value in row)
                   for row in evaluation["predictions"])
        assert json.loads(canonical_bytes(slot["result"])) == slot["result"]
        restoration = evaluation["restoration"]
        assert restoration["execution"] == execution
        assert restoration["endpoint_sha256"] == restoration["restored_snapshot_sha256"] == expected_hashes[index]
        assert restoration["sha256"] == digest({key: value for key, value in restoration.items()
                                                if key != "sha256"})
        receipt = read(tmp_path / "success" / f"evaluation-000000-{index:06d}.json")
        assert receipt["sha256"] == slot["receipt_sha256"]
        assert receipt["endpoint_seal_sha256"] == seal["sha256"]
        assert receipt["result"] == slot["result"]
    assert [parent_row[0] - repaired_row[0] for parent_row, repaired_row in zip(
        slots[1]["result"]["sequential"]["predictions"],
        slots[0]["result"]["sequential"]["predictions"])] == [3.0, 3.0, 3.0]
    for document in (seal, terminal, result):
        assert document["runtime_restoration_verified"] is False
        assert document["causal_usefulness_established"] is False
    for flag in ("provenance_authenticated", "semantic_independence_verified",
                 "scientific_freshness_verified", "evaluator_execution_attested"):
        assert attempt[flag] is False

    context = {"slot": deepcopy(seal["members"][0]["slots"][1])}
    single = evaluate_sequential_slot(context, parent, literal_rows[:1])
    assert single["sequential"]["predictions"] == [[3.0]]

    overflowing = sequential_artifact(
        weights=[[[-float.fromhex("0x1.fffffffffffffp+1023")]]], biases=[[0.0]],
        activations=["relu"], input_names=("value",), output_names=("score",),
    )
    overflow_context = {"slot": {"endpoint_sha256": overflowing.sha256,
        "execution": deepcopy(execution), "result_keys": ["sequential"]}}
    with patch.object(SequentialRuntime, "forward", autospec=True,
                      side_effect=SequentialRuntime.forward) as forward:
        with pytest.raises(ValueError, match="nonfinite intermediates"):
            evaluate_sequential_slot(overflow_context, overflowing, [{"inputs": {"value": 2.0}}])
        assert forward.call_count == 1 and forward.call_args.args[1] == [[2.0]]

    class DerivedDict(dict):
        pass

    class DerivedString(str):
        pass

    invalid_contexts = [None, {}, {"slot": []}, DerivedDict(context),
        context | {"extra": (1, 2)}, context | {"extra": float("nan")},
        context | {"extra": np.float64(1.0)},
        {"slot": context["slot"] | {"endpoint_sha256": "0" * 64}},
        {"slot": context["slot"] | {"execution": execution | {"accumulation": "matmul"}}},
        *[{"slot": context["slot"] | {"result_keys": keys}}
          for keys in ([], ["other"], ["sequential", "extra"], ("sequential",))],
    ]
    invalid_rows = [[], tuple(literal_rows), [[]], [{}], [{"inputs": []}],
        [{"inputs": {"left": 1}}], [{"inputs": {"left": 1, "right": 2, "extra": 0}}],
        [{"inputs": DerivedDict(left=1, right=2)}],
        [{"inputs": {DerivedString("left"): 1, "right": 2}}],
        [{"inputs": {"left": 1, "right": 2}, "extra": np.array([1.0])}],
        *[[literal_rows[0], {"inputs": {"left": 1, "right": value}}]
          for value in (True, "2", float("nan"), float("inf"), 2**53 + 1, np.int64(2))],
    ]
    with patch.object(SequentialRuntime, "forward",
                      side_effect=AssertionError("Invalid rows or declarations reached forward")) as forward:
        for invalid in invalid_contexts:
            with pytest.raises(ValueError):
                evaluate_sequential_slot(invalid, parent, literal_rows)
        for invalid in invalid_rows:
            with pytest.raises(ValueError):
                evaluate_sequential_slot(context, parent, invalid)
        with pytest.raises(ValueError, match="endpoint"):
            evaluate_sequential_slot(context, repaired, literal_rows)
        assert forward.call_count == 0

    malformed = [literal_rows[0], {"inputs": {"left": 3, "right": "1"}}]
    cohort, registry, batch, seal = completed("failed-input", malformed)
    with patch.object(SequentialRuntime, "forward",
                      side_effect=AssertionError("A later invalid row must prevent all forwarding")) as forward:
        with pytest.raises(ValueError, match="finite numbers"):
            cohort.evaluate_registered(lambda: {"authored": batch}, evaluate_sequential_slot)
        assert forward.call_count == 0
    failed = read(tmp_path / "failed-input" / "evaluation.json")
    assert failed["outcome"] == "failed" and failed["failure_phase"] == "slot_evaluation"
    assert failed["exception_type"] == "ValueError"
    failed_slots = failed["result"]["members"][0]["slots"]
    assert [slot["outcome"] for slot in failed_slots] == ["failed", "not_run"]
    assert [slot["evaluator_invocations"] for slot in failed_slots] == [1, 0]
    assert all(slot["result"] is None for slot in failed_slots)
    attempt = registry.get_attempt("failed-input")
    assert failed["registry_attempt"] == attempt
    assert attempt["status"] == "failed" and attempt["callback_invocations"] == 1
    assert attempt["evidence_role"] == "evaluation_consumed_on_authorization"
    assert ["source", "authored-rows"] in attempt["consumed_scope"]
    with pytest.raises(WorkflowError):
        cohort.evaluate_registered(lambda: {"authored": batch}, evaluate_sequential_slot)
    with pytest.raises(DataRoleError) as rejection:
        registry.score(batch, lambda rows: pytest.fail("Spent rows were reused"), attempt_id="no-reopen")
    assert rejection.value.decision["reason"] == "observed_development_consumption"
    assert registry.get_attempt("no-reopen") is None
    for document in (failed, failed["result"]):
        assert document["runtime_restoration_verified"] is False
        assert document["causal_usefulness_established"] is False
