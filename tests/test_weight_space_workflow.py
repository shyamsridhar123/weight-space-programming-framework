"""Adversarial boundaries for the generic programmer/evaluator protocol."""
import copy
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from rlcd.weight_space import DataRoleRegistry, Edit, Network, Program, RegisteredBatch, TensorArtifact
from rlcd.weight_space.model import digest
from rlcd.weight_space.workflow import (
    AccessDenied, AccessPolicy, BudgetExceeded, BudgetLimits, ProgrammerSession,
    ReplayProvider, SessionSealed, SplitSpec, TaskSpec, WorkflowError,
    validate_splits, verify_trace,
)


def artifact():
    return TensorArtifact("caller-linear/1", {"weight": [[1.234567, 2.], [3., 4.]], "bias": [0., 0.]},
                          {"semantic_names": ["first", "second"]})


def session(mode="edit", *, base=None, constraints=None, **kwargs):
    base = artifact() if base is None else base
    task = TaskSpec("task-1", mode, "An explicitly scoped test objective", base.architecture,
                    base.sha256, constraints)
    return ProgrammerSession(task, base=base, **kwargs)


def test_task_config_and_snapshots_cannot_be_mutated_through_inputs_or_returns():
    constraints = {"privileged_hints": ["hidden-unit role"], "semantic_names": {"x": ["a", "b"]}}
    base = artifact()
    task = TaskSpec("read-1", "read", "Describe the model", base.architecture, base.sha256, constraints)
    task_hash = task.sha256
    constraints["semantic_names"]["x"].reverse()
    task.constraints["privileged_hints"].clear()
    assert task.sha256 == task_hash
    s = ProgrammerSession(task, base=base)
    snapshot, trace = s.usage, s.trace
    result = s.metadata()
    result["task"]["constraints"]["privileged_hints"].clear()
    assert task.constraints["privileged_hints"] == ["hidden-unit role"]
    assert s.trace[-1].to_dict()["result"]["task"]["constraints"]["privileged_hints"] == ["hidden-unit role"]
    assert snapshot.tool_calls == 0 and s.usage.tool_calls == 1 and len(trace) == 1
    with pytest.raises((FrozenInstanceError, AttributeError)):
        s.usage.tool_calls = 0
    with pytest.raises((FrozenInstanceError, AttributeError)):
        s.access.weights = False
    with pytest.raises(AttributeError):
        s.access = AccessPolicy()
    s.configuration["limits"]["tool_calls"] = 99999
    assert s.limits.tool_calls == 100


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("nan"), "1"])
def test_counters_and_precision_are_strict_data(bad):
    with pytest.raises(ValueError):
        BudgetLimits(tool_calls=bad)
    with pytest.raises(ValueError):
        AccessPolicy(weight_decimals=bad)


def test_invalid_task_configuration_is_rejected_before_a_run():
    base = artifact()
    for mode in ("read", "diagnose", "edit"):
        with pytest.raises(ValueError, match="parent"):
            TaskSpec("id", mode, "objective", base.architecture)
    with pytest.raises(ValueError, match="no parent"):
        TaskSpec("id", "create", "objective", base.architecture, base.sha256)
    with pytest.raises(ValueError, match="finite"):
        TaskSpec("id", "create", "objective", base.architecture, constraints={"bound": float("inf")})
    with pytest.raises(ValueError, match="keys"):
        TaskSpec("id", "create", "objective", base.architecture, constraints={1: "ambiguous"})
    with pytest.raises(ValueError):
        AccessPolicy(metadata=1)
    with pytest.raises(ValueError):
        AccessPolicy(weight_decimals=16)
    task = TaskSpec("id", "edit", "objective", base.architecture, base.sha256)
    with pytest.raises(ValueError, match="exact parent"):
        ProgrammerSession(task, base=base.replace(bias=[1., 0.]))
    with pytest.raises(ValueError, match="architecture"):
        ProgrammerSession(TaskSpec("id", "edit", "objective", "other", base.sha256), base=base)
    with pytest.raises(ValueError, match="callables"):
        ProgrammerSession(task, base=base, forward="package.module:function")


def test_hidden_metadata_does_not_escape_through_accounting_or_read_receipts():
    s = session("read", constraints={"privileged_hints": ["a very private hint"]},
                access=AccessPolicy(metadata=False, weights=False))
    with pytest.raises(AccessDenied):
        s.metadata()
    with pytest.raises(AccessDenied):
        s.weights("weight")
    assert s.usage.tool_calls == 0
    candidate = s.submit_answer({"abstain": True})
    s.seal(candidate)
    receipt = s.evaluate_heldout(lambda c: {"accepted": True}, evaluator_id="read-rubric/1")
    assert "a very private hint" not in json.dumps(receipt.to_dict())
    assert receipt.to_dict()["configuration"]["declared_constraints_automatically_enforced"] is False
    assert receipt.to_dict()["configuration"]["enforcement"] == "cooperative_same_process_interface"


def test_weight_access_counts_scalars_duplicates_and_rounding_without_changing_parent():
    base = artifact()
    s = session(base=base, access=AccessPolicy(weight_decimals=3), limits=BudgetLimits(weight_values=6))
    assert s.weights("weight") == [[1.235, 2.], [3., 4.]]
    assert s.weights("weight", [(0, 0), (0, 0)]) == [1.235, 1.235]
    assert s.usage.weight_values == 6
    assert float(base.tensors["weight"][0, 0]) == 1.234567
    before = s.usage
    with pytest.raises(BudgetExceeded):
        s.weights("bias")
    assert s.usage is before
    with pytest.raises(AccessDenied, match="rounded-only"):
        s.propose_program(Program(base.sha256, (), "No-op must not disclose exact values"))
    assert not s.candidates and s.usage.proposals == 0
    assert s.trace[1].to_dict()["request"]["weight_decimals"] == 3
    assert verify_trace(s.trace)


def test_scalar_and_invalid_coordinates_do_not_expose_extra_values():
    base = TensorArtifact("scalar/1", {"bias": np.array(2.)})
    s = session(base=base)
    assert s.weights("bias") == 2.
    assert s.weights("bias", [()]) == [2.]
    for indices in ([(0,)], [(True,)], "bias", [None]):
        with pytest.raises(ValueError):
            s.weights("bias", indices)
    assert s.usage.weight_values == 2 and s.usage.tool_calls == 2
    assert verify_trace(s.trace)


def test_program_return_is_not_a_free_weight_or_metadata_channel():
    base = artifact()
    program = Program(base.sha256, (), "Keep the baseline eligible")
    for access in (AccessPolicy(weights=False), AccessPolicy(metadata=False)):
        s = session(base=base, access=access)
        with pytest.raises(AccessDenied):
            s.propose_program(program)
        assert s.usage.proposals == 0
    s = session(base=base, limits=BudgetLimits(weight_values=base.parameter_count - 1))
    with pytest.raises(BudgetExceeded):
        s.propose_program(program)
    assert s.usage.proposals == 0 and s.usage.tool_calls == 0
    s = session(base=base)
    candidate = s.propose_program(program)
    assert candidate.no_op and candidate.artifact.sha256 == base.sha256
    assert s.usage.weight_values == base.parameter_count


def test_failed_proposals_consume_attempt_budget_and_preserve_parent_and_parser_evidence():
    base = artifact()
    s = session(base=base)
    stale = Program("0" * 64, (), "Wrong parent")
    with pytest.raises(ValueError, match="parent"):
        s.propose_program(stale)
    bad_cell = Program(base.sha256, (Edit("bias", (0,), 7., 8.),), "Wrong precondition")
    with pytest.raises(ValueError, match="precondition"):
        s.propose_program(bad_cell)
    malformed = {"schema": "unknown/1", "intent": "record even parse failures"}
    with pytest.raises(ValueError):
        s.propose_program(malformed)
    assert s.usage.proposals == 3 and not s.candidates
    assert s.usage.weight_values == 3 * base.parameter_count
    assert s.trace[-1].to_dict()["request"]["submitted_sha256"] == digest(malformed)
    s.seal()
    receipt = s.evaluate_heldout(lambda c: pytest.fail("Abstentions do not invoke an evaluator"), evaluator_id="test/1")
    assert receipt.status == "abstained"
    assert len([e for e in receipt.to_dict()["trace"] if e["status"] == "failed"]) == 3
    assert receipt.to_dict()["heldout_accounting"]["evaluator_calls"] == 0
    assert base.sha256 == artifact().sha256


def test_artifact_proposals_enforce_mode_and_interface_but_need_no_parent_disclosure():
    base = artifact()
    s = session(base=base, access=AccessPolicy(metadata=False, weights=False))
    candidate = s.propose_artifact(base)
    assert candidate.no_op and s.usage.weight_values == 0
    wrong = TensorArtifact(base.architecture, {"weight": [1.], "bias": [0.]}, base.metadata)
    with pytest.raises(ValueError, match="preserve"):
        s.propose_artifact(wrong)
    changed_metadata = TensorArtifact(base.architecture, base.tensors, {"semantic_names": ["other"]})
    with pytest.raises(ValueError, match="preserve"):
        s.propose_artifact(changed_metadata)
    with pytest.raises(ValueError, match="architecture"):
        s.propose_artifact(TensorArtifact("other/1", {"weight": [1.]}))
    assert s.usage.proposals == 4
    with pytest.raises(WorkflowError):
        s.submit_answer({"not": "an edit"})
    assert s.usage.proposals == 4
    created = ProgrammerSession(TaskSpec("new", "create", "Create a scalar", "scalar/1"))
    out = created.propose_artifact(TensorArtifact("scalar/1", {"weight": [2.]}))
    assert out.parent_sha256 is None and not out.no_op
    with pytest.raises(WorkflowError):
        created.propose_program(Program(base.sha256, (), "Invalid create edit"))


@pytest.mark.parametrize("mode", ["read", "diagnose"])
def test_answers_are_finite_immutable_and_bound_to_the_task(mode):
    s = session(mode)
    answer = {"label": "linear", "confidence": np.float64(.75), "observations": [True, 1]}
    candidate = s.submit_answer(answer)
    answer["observations"].clear()
    candidate.answer["observations"].reverse()
    assert candidate.answer["observations"] == [True, 1]
    assert candidate.artifact is None and not candidate.no_op
    with pytest.raises(ValueError, match="finite"):
        s.submit_answer({"score": np.nan})
    assert s.usage.proposals == 2 and len(s.candidates) == 1
    with pytest.raises(WorkflowError):
        s.propose_artifact(artifact())
    s.seal(candidate)
    receipt = s.evaluate_heldout(lambda c: {"correct": c.answer["label"] == "linear"}, evaluator_id="rubric/1")
    assert receipt.to_dict()["result"]["correct"] is True


def test_feedback_charges_examples_before_failure_and_budget_denial_is_atomic():
    calls = []
    def feedback(candidate, examples):
        calls.append(examples)
        raise RuntimeError("Intentional adapter failure")
    s = session(access=AccessPolicy(feedback=True), public_feedback=feedback,
                limits=BudgetLimits(feedback_examples=2))
    candidate = s.propose_artifact(artifact())
    before = s.usage
    with pytest.raises(RuntimeError):
        s.feedback(candidate, [[1.], [2.]])
    assert s.usage.feedback_examples == 2 and s.usage.feedback_calls == 1
    assert s.usage.tool_calls == before.tool_calls + 1 and len(calls) == 1
    before = s.usage
    with pytest.raises(BudgetExceeded):
        s.feedback(candidate, [[3.]])
    assert s.usage is before and len(calls) == 1
    event = s.trace[-1].to_dict()
    assert event["charges"] == {} and event["requested_charges"]["feedback_examples"] == 1
    assert verify_trace(s.trace)


def test_execution_capabilities_account_for_batch_size_and_snapshot_inputs_and_outputs():
    examples = [[1., 2.], [3., 4.]]
    output = np.array([[1.], [2.]])
    def forward(base, captured):
        captured[0][0] = 77.
        examples[0][0] = 88.
        return output
    s = session(access=AccessPolicy(forward=True), forward=forward,
                limits=BudgetLimits(execution_examples=2))
    result = s.forward(examples)
    result[0][0] = -1.
    output[0][0] = -2.
    event = s.trace[-1].to_dict()
    assert event["request"]["examples"] == [[1., 2.], [3., 4.]]
    assert event["result"] == [[1.], [2.]]
    assert s.usage.execution_examples == 2 and s.usage.execution_calls == 1
    with pytest.raises(AccessDenied):
        s.activations([[1., 2.]])
    with pytest.raises(BudgetExceeded):
        s.forward([[1., 2.]])
    assert s.usage.execution_calls == 1
    a = session(access=AccessPolicy(activations=True), activations=lambda model, batch: np.array(batch))
    assert a.activations(np.array([[1., 2.], [3., 4.]])) == [[1., 2.], [3., 4.]]
    assert a.usage.execution_examples == 2


def test_nonfinite_or_unavailable_execution_fails_after_charging_and_prevents_reentry():
    bad = session(access=AccessPolicy(forward=True), forward=lambda model, rows: {"value": np.inf})
    with pytest.raises(ValueError, match="finite"):
        bad.forward([[1.]])
    assert bad.usage.execution_examples == 1 and bad.trace[-1].to_dict()["status"] == "failed"
    missing = session(access=AccessPolicy(forward=True))
    with pytest.raises(WorkflowError, match="registered"):
        missing.forward([[1.]])
    assert missing.usage.execution_calls == 1
    s = session(access=AccessPolicy(forward=True), forward=lambda model, rows: s.seal())
    with pytest.raises(WorkflowError, match="reenter"):
        s.forward([[1.]])
    assert not s.is_sealed and verify_trace(s.trace)


@pytest.mark.parametrize("action", ["feedback", "forward", "activations"])
def test_callbacks_cannot_return_candidate_objects_in_place_of_finite_json(action):
    callback_name = "public_feedback" if action == "feedback" else action
    s = session(access=AccessPolicy(**{action: True}),
                **{callback_name: lambda model, rows: candidate})
    candidate = s.propose_artifact(artifact())
    before = s.usage
    with pytest.raises(ValueError, match="finite JSON data, received Candidate"):
        if action == "feedback":
            s.feedback(candidate, [[1.], [2.]])
        else:
            getattr(s, action)([[1.], [2.]], candidate=candidate)
    counter = "feedback" if action == "feedback" else "execution"
    assert getattr(s.usage, f"{counter}_calls") == 1
    assert getattr(s.usage, f"{counter}_examples") == 2
    assert s.usage.tool_calls == before.tool_calls + 1 and s.candidates == (candidate,)
    assert s.trace[-1].to_dict()["status"] == "failed"
    assert verify_trace(s.trace)


def test_candidate_identity_prevents_foreign_selection_even_for_identical_content():
    first, second = session(), session()
    candidate = first.propose_artifact(artifact())
    equivalent = second.propose_artifact(artifact())
    assert candidate.sha256 == equivalent.sha256
    forged = copy.copy(candidate)
    for invalid in (equivalent, forged):
        with pytest.raises(WorkflowError, match="exact candidate"):
            first.seal(invalid)
        assert not first.is_sealed
    first.seal(candidate)
    receipt = first.evaluate_heldout(lambda selected: {"hash": selected.artifact.sha256}, evaluator_id="hash/1")
    assert receipt.to_dict()["candidate_sha256"] == candidate.sha256
    assert receipt.to_dict()["parent_sha256"] == artifact().sha256


def test_sealing_closes_proposer_operations_and_opaque_evaluation_is_once_but_unverified():
    s = session(access=AccessPolicy(feedback=True, forward=True, activations=True))
    candidate = s.propose_artifact(artifact())
    calls = []
    heldout_examples = ["PRIVATE_HELDOUT_SAMPLE_913", "PRIVATE_HELDOUT_SAMPLE_927"]
    def evaluator(selected):
        calls.append(selected.sha256)
        assert selected is candidate
        return {"score": len(heldout_examples)}
    with pytest.raises(WorkflowError, match="Seal"):
        s.evaluate_heldout(evaluator, evaluator_id="heldout/1")
    assert not calls
    s.seal(candidate)
    trace = s.trace
    actions = [s.metadata, lambda: s.weights("weight"), lambda: s.propose_artifact(artifact()),
               lambda: s.propose_program(Program(artifact().sha256, (), "No-op")),
               lambda: s.submit_answer({}), lambda: s.feedback(candidate, [[1.]]),
               lambda: s.forward([[1.]]), lambda: s.activations([[1.]]),
               lambda: s.record_provider_usage(input_tokens=1, output_tokens=1), s.seal]
    for action in actions:
        with pytest.raises(SessionSealed):
            action()
    assert s.trace == trace
    receipt = s.evaluate_heldout(evaluator, evaluator_id="heldout/1")
    with pytest.raises(SessionSealed, match="already attempted"):
        s.evaluate_heldout(evaluator, evaluator_id="heldout/1")
    assert len(calls) == 1 and receipt.status == "succeeded"
    assert receipt.to_dict()["freshness"]["status"] == "unverified"
    assert receipt.to_dict()["freshness"]["scope"] == "opaque_evaluator_inputs"
    assert all(sample not in json.dumps(receipt.to_dict()) for sample in heldout_examples)
    assert verify_trace(s.trace, expected_head=receipt.to_dict()["trace_head_sha256"])
    exported = receipt.to_dict()
    exported["result"]["score"] = -1
    assert receipt.to_dict()["result"]["score"] == 2


@pytest.mark.parametrize("nonfinite", [False, True])
def test_heldout_failure_produces_a_receipt_and_cannot_retry_or_leak_error_examples(nonfinite):
    s = session()
    s.seal(s.propose_artifact(artifact()))
    calls = []
    def evaluator(candidate):
        calls.append(1)
        if nonfinite:
            return {"score": float("nan")}
        raise RuntimeError("Private heldout sample SECRET_123 caused a failure")
    receipt = s.evaluate_heldout(evaluator, evaluator_id="failed/1")
    assert receipt.status == "failed" and receipt.to_dict()["result"] is None
    assert "SECRET_123" not in json.dumps(receipt.to_dict())
    assert receipt.to_dict()["trace"][-1]["status"] == "failed"
    with pytest.raises(SessionSealed):
        s.evaluate_heldout(evaluator, evaluator_id="retry/1")
    assert len(calls) == 1 and s.receipt is receipt


def test_token_usage_is_reported_not_measured_and_budget_addition_is_atomic():
    s = session(limits=BudgetLimits(input_tokens=10, output_tokens=2))
    s.record_provider_usage(input_tokens=7, output_tokens=1)
    snapshot = s.usage
    with pytest.raises(BudgetExceeded):
        s.record_provider_usage(input_tokens=1, output_tokens=2)
    assert s.usage is snapshot and s.usage.tool_calls == 0
    with pytest.raises(ValueError):
        s.record_provider_usage(input_tokens=True, output_tokens=0)
    assert s.trace[1].to_dict()["request"]["accounting"] == "provider_supplied_not_locally_measured"
    assert verify_trace(s.trace)


def test_split_validation_rejects_declared_leakage_without_claiming_unobserved_independence():
    for specs in ([SplitSpec("train", ["a"]), SplitSpec("test", ["a"])],
                  [SplitSpec("train", ["a"], ["teacher"]), SplitSpec("test", ["b"], ["teacher"])],
                  [SplitSpec("same", ["a"]), SplitSpec("same", ["b"])]):
        with pytest.raises(ValueError):
            validate_splits(specs)
        with pytest.raises(ValueError):
            session(splits=specs)
    with pytest.raises(ValueError, match="Duplicate"):
        SplitSpec("train", ["a", "a"])
    ids = ["a"]
    split = SplitSpec("train", ids)
    ids.append("b")
    assert split.sample_ids == ("a",)
    valid = validate_splits([split, SplitSpec("test", ["b"])])
    assert valid.to_dict()["lineage_scope"] == "caller_declared_only"
    assert valid.to_dict()["lineage_independence_verified"] is False
    s = session(splits=[split, SplitSpec("test", ["b"])])
    s.seal(s.propose_artifact(artifact()))
    with pytest.raises(ValueError, match="absent"):
        s.evaluate_heldout(lambda c: {}, evaluator_id="wrong-split/1", split_name="unknown")


def test_trace_commitment_detects_mutation_and_verifies_usage_not_just_link_hashes():
    s = session()
    s.metadata()
    s.weights("bias")
    events = [event.to_dict() for event in s.trace]
    head = events[-1]["sha256"]
    assert verify_trace(events, expected_head=head)
    events[1]["result"]["task"]["objective"] = "Tampered"
    assert not verify_trace(events)
    events = [event.to_dict() for event in s.trace]
    events[-1]["usage"]["weight_values"] = 0
    events[-1]["sha256"] = digest({key: value for key, value in events[-1].items() if key != "sha256"})
    assert not verify_trace(events)
    assert not verify_trace(s.trace[:-1], expected_head=head)
    assert verify_trace(s.trace)


def test_replay_is_data_only_deterministic_and_keeps_failures_and_noop_outputs():
    base = artifact()
    actions = [{"op": "metadata"},
               {"op": "propose_program", "program": {"schema": "malformed"}},
               {"op": "propose_program", "program": Program(base.sha256, (), "Keep baseline").to_dict()},
               {"op": "seal", "candidate": 0}]
    replay = ReplayProvider(actions)
    actions[0]["op"] = "dynamic_import"
    first, second = session(), session()
    assert replay.run(first) == replay.run(second)
    assert first.is_sealed and first.usage.proposals == 2
    first_receipt = first.evaluate_heldout(lambda c: {"noop": c.no_op}, evaluator_id="replay-test/1")
    second_receipt = second.evaluate_heldout(lambda c: {"noop": c.no_op}, evaluator_id="replay-test/1")
    assert first_receipt.sha256 == second_receipt.sha256
    assert first_receipt.to_dict()["candidates"][0]["no_op"] is True
    assert first.trace[1].to_dict()["request"]["model_ability_measured"] is False
    assert verify_trace(first.trace)
    blocked = session()
    outcomes = ReplayProvider([{"op": "import", "module": "os"}]).run(blocked)
    assert outcomes[0]["status"] == "failed" and blocked.usage.tool_calls == 0
    with pytest.raises(ValueError):
        ReplayProvider([{"op": "callback", "fn": lambda: None}])


def test_network_backend_uses_the_same_session_protocol():
    base = Network([[1.]], [0.], [[1.]], [0.])
    task = TaskSpec("relu", "edit", "Double the readout", "dense-relu-dense/1", base.sha256)
    s = ProgrammerSession(task, base=base, access=AccessPolicy(forward=True), forward=lambda model, x: model(x))
    candidate = s.propose_program(Program(base.sha256, (Edit("w2", (0, 0), 1., 2.),), "Scale readout"))
    assert s.forward([[1.], [2.]], candidate=candidate) == [[2.], [4.]]
    s.seal(candidate)
    receipt = s.evaluate_heldout(lambda c: {"prediction": c.artifact([[3.]])}, evaluator_id="relu/1")
    assert receipt.to_dict()["result"] == {"prediction": [[6.]]}


@pytest.mark.parametrize("route", ["feedback", "forward", "activations"])
@pytest.mark.parametrize("callback_fails", [False, True])
def test_development_exposure_invalidates_copied_evaluation_across_sessions(tmp_path, route, callback_fails):
    registry_path = tmp_path / "data-roles.sqlite3"
    registry = DataRoleRegistry(registry_path)
    rows = [[1., 0.], [0., 1.]]
    exposed = []
    def callback(_, materialized):
        exposed.append(copy.deepcopy(materialized))
        if callback_fails:
            raise RuntimeError("Development callback failed after seeing the batch")
        return {"observed": len(materialized)}
    adapters = {"public_feedback" if route == "feedback" else route: callback}
    development = session(data_registry=registry, access=AccessPolicy(**{route: True}), **adapters)
    candidate = development.propose_artifact(artifact())
    def expose():
        return development.feedback(candidate, rows) if route == "feedback" else getattr(development, route)(rows)
    if callback_fails:
        with pytest.raises(RuntimeError):
            expose()
    else:
        assert expose() == {"observed": 2}
    assert exposed == [rows]
    assert development.trace[-1].to_dict()["request"]["data_role_decision"]["status"] == "succeeded"

    reopened = DataRoleRegistry(registry_path)
    copied = reopened.register_batch(copy.deepcopy(rows), source_id="renamed-evaluation", family_id="new-label")
    evaluation = session(data_registry=reopened)
    evaluation.seal(evaluation.propose_artifact(artifact()))
    scoring_calls = []
    receipt = evaluation.evaluate_registered(copied, lambda candidate, x: scoring_calls.append(x),
                                            evaluator_id="copied-data-check/1").to_dict()
    assert receipt["status"] == "rejected" and scoring_calls == []
    assert receipt["freshness"]["status"] == "invalidated"
    assert receipt["freshness"]["decision"]["scoring_started"] is False
    assert receipt["heldout_accounting"]["evaluator_calls"] == 0
    assert receipt["heldout_accounting"]["evaluated_examples"] == 0
    assert receipt["freshness"]["semantic_independence_verified"] is False
    assert verify_trace(development.trace) and verify_trace(evaluation.trace)
    with pytest.raises(SessionSealed):
        evaluation.evaluate_registered(copied, lambda c, x: {}, evaluator_id="retry/1")


def test_registered_selection_is_bound_and_new_sessions_cannot_repeat_scoring(tmp_path):
    path = tmp_path / "data-roles.sqlite3"
    registry = DataRoleRegistry(path)
    batch = registry.register_batch([[2., 3.]], source_id="evaluation", family_id="evaluation-family")
    original = session(data_registry=registry)
    selected = original.propose_artifact(artifact())
    original.seal(selected)
    seal_hash = original.trace[-1].sha256
    receipt = original.evaluate_registered(batch, lambda c, rows: {"actual": rows},
                                           evaluator_id="bound-evaluator/1").to_dict()
    decision = receipt["freshness"]["decision"]
    attempt = registry.get_attempt(decision["attempt_id"])
    assert decision["attempt"] == attempt and attempt["status"] == "succeeded"
    assert attempt["binding"] == {"schema": "wsp-session-evaluation-binding/1",
        "protocol_sha256": original.sha256, "task_sha256": selected.task_sha256,
        "parent_sha256": selected.parent_sha256, "candidate_sha256": selected.sha256,
        "selection_sha256": digest({"selected": selected.to_dict(),
            "candidates": [selected.to_dict()], "seal_event_sha256": seal_hash}),
        "evaluator_id": "bound-evaluator/1", "split_name": "test"}
    rebuilt = session(data_registry=DataRoleRegistry(path))
    rebuilt.seal(rebuilt.propose_artifact(artifact()))
    calls = []
    rejected = rebuilt.evaluate_registered(batch, lambda c, rows: calls.append(rows),
                                            evaluator_id="new-evaluator/1").to_dict()
    assert rejected["status"] == "rejected" and calls == []
    assert rejected["freshness"]["decision"]["reason"] == "observed_development_consumption"


def test_registered_scorer_failure_has_one_call_and_legacy_input_remains_unverified(tmp_path):
    registry = DataRoleRegistry(tmp_path / "data-roles.sqlite3")
    batch = registry.register_batch([[2., 3.]], source_id="evaluation", family_id="evaluation-family")
    registered = session(data_registry=registry)
    registered.seal(registered.propose_artifact(artifact()))
    def failing(candidate, rows):
        raise RuntimeError("private evaluation record")
    receipt = registered.evaluate_registered(batch, failing, evaluator_id="failure/1").to_dict()
    assert receipt["status"] == "failed"
    assert receipt["heldout_accounting"]["evaluator_calls"] == 1
    assert receipt["freshness"]["decision"]["scoring_started"] is True
    assert "private evaluation record" not in json.dumps(receipt)

    registry.consume(batch, reason="development")
    opaque = session(data_registry=registry)
    opaque.seal(opaque.propose_artifact(artifact()))
    old = opaque.evaluate_heldout(lambda c: {"precomputed": True}, evaluator_id="opaque/1").to_dict()
    assert old["status"] == "succeeded"
    assert old["freshness"]["status"] == "unverified"
    assert old["freshness"]["scope"] == "opaque_evaluator_inputs"
    assert old["freshness"]["decision"] is None
    assert old["freshness"]["scientific_freshness_verified"] is False


@pytest.mark.parametrize("failure", ["nonfinite", "nonjson", "interrupt", "exit"])
def test_registered_scorer_invalid_result_and_cancellation_keep_receipt(tmp_path, failure):
    registry = DataRoleRegistry(tmp_path / "data-roles.sqlite3")
    batch = registry.register_batch([[2., 3.]], source_id="evaluation", family_id="evaluation-family")
    evaluation = session(data_registry=registry)
    evaluation.seal(evaluation.propose_artifact(artifact()))
    calls = []
    def scorer(candidate, rows):
        calls.append(rows)
        if failure == "interrupt":
            raise KeyboardInterrupt("private scorer detail")
        if failure == "exit":
            raise SystemExit("private scorer detail")
        return {"score": float("nan")} if failure == "nonfinite" else object()
    if failure in ("interrupt", "exit"):
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else SystemExit):
            evaluation.evaluate_registered(batch, scorer, evaluator_id="failure-boundary/1")
    else:
        evaluation.evaluate_registered(batch, scorer, evaluator_id="failure-boundary/1")
    receipt = evaluation.receipt.to_dict()
    assert calls == [batch.materialize()] and receipt["status"] == "failed"
    assert receipt["heldout_accounting"] == {"evaluator_calls": 1, "evaluated_examples": 1,
                                             "scope": "outside_proposer_budget"}
    assert receipt["freshness"]["decision"]["status"] == "failed"
    assert receipt["freshness"]["decision"]["scoring_started"] is True
    assert receipt["freshness"]["manifest_sha256"] == batch.manifest_sha256
    assert "private scorer detail" not in json.dumps(receipt)
    assert verify_trace(evaluation.trace)
    with pytest.raises(SessionSealed):
        evaluation.evaluate_registered(batch, scorer, evaluator_id="retry/1")


@pytest.mark.parametrize("route", ["evaluation", "feedback", "forward", "activations"])
def test_uninitialized_registered_handle_rejects_before_callback_and_records_attempt(tmp_path, route):
    calls = []
    callback = lambda c, rows: calls.append(rows)
    evaluation = session(data_registry=DataRoleRegistry(tmp_path / "data-roles.sqlite3"),
                         access=AccessPolicy(feedback=True, forward=True, activations=True),
                         public_feedback=callback, forward=callback, activations=callback)
    candidate = evaluation.propose_artifact(artifact())
    if route == "evaluation":
        evaluation.seal(candidate)
        receipt = evaluation.evaluate_registered(RegisteredBatch(), callback,
                                                 evaluator_id="malformed-handle/1").to_dict()
        assert receipt["freshness"]["status"] == "unverified"
        assert receipt["freshness"]["decision"]["reason"] == "invalid_registration"
        assert receipt["heldout_accounting"]["evaluator_calls"] == 0
    else:
        with pytest.raises(ValueError):
            if route == "feedback":
                evaluation.feedback(candidate, RegisteredBatch())
            else:
                getattr(evaluation, route)(RegisteredBatch())
    assert evaluation.trace[-1].to_dict()["status"] == "rejected" and calls == []
    assert verify_trace(evaluation.trace)
