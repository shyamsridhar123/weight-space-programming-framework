"""One literal saved-evidence regression; no research loaders or scientific study."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rlcd.weight_space import broker, program, staged_cohort, structured
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.data_roles import DataRoleRegistry, RegisteredBatch
from rlcd.weight_space.model import digest
from rlcd.weight_space.staged_cohort import (
    PreparedMember, StagedCohort, verify_registered_cohort_evaluation,
)
from rlcd.weight_space.staged_diagnosis import StagedDiagnosisRepair
from rlcd.weight_space.workflow import ProgrammerSession


def test_saved_cohort_success_joins_trusted_commitments_and_durable_attempt():
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved verification must not execute or inspect a current snapshot")

    def rehash(record, key="sha256"):
        record[key] = digest({name: value for name, value in record.items() if name != key})

    execution = {"backend": "numpy", "dtype": "float64", "device": "cpu",
                 "accumulation": "matmul", "endpoint_cast": "reject", "error_model": "empirical"}
    plan = {"schema": "wsp-staged-cohort-evaluation-plan/1",
            "result_schema": "wsp-staged-cohort-result/1",
            "aggregate": {"source_id": "aggregate", "family_id": "aggregate-family"},
            "holdouts": [{"holdout_id": "main", "source_id": "holdout", "family_id": "holdout-family",
                          "row_count": 1}],
            "members": [{"member_id": "member", "holdout_id": "main", "slots": [
                {"slot_id": identifier, "source": {"kind": "selected", "fallback_control_id": fallback},
                 "execution": deepcopy(execution), "result_keys": ["literal_result"]}
                for identifier, fallback in (("fallback", "parent"), ("unavailable", None))]}]}
    answer = {"location": None, "mechanism": "Literal abstention", "predicted_effect": "No repair",
              "confidence": 0.5, "uncertainty": "Engineering fixture only", "abstain": True}
    base = TensorArtifact("literal-cohort-verifier/1", {"layer.weight": [[1.0, 2.0]]})
    attempt_id = "literal-saved-success"
    calls, retained = [], {}

    with TemporaryDirectory(prefix="wsp-cohort-verifier-") as directory:
        root = Path(directory)
        registry = DataRoleRegistry(root / "roles.sqlite")
        batch = registry.register_batch([{"value": 11}], source_id="holdout", family_id="holdout-family")
        cohort = StagedCohort(root / "cohort", member_ids=["member"], protocol={"evaluation_plan": plan},
            controls={"member": {"parent": {"kind": "parent"}}}, evaluator_id="literal-evaluator",
            registry=registry, attempt_id=attempt_id)
        cohort.prepare(lambda row: PreparedMember(base, "literal-verifier", "Check saved evidence", {}))
        cohort.select_diagnoses(lambda row, query: deepcopy(answer))
        cohort.select_repairs(forbidden)
        seal = cohort.seal_endpoints()

        def evaluate(context, endpoint, rows):
            assert context["slot"]["slot_id"] == "fallback"
            assert endpoint is base and rows == [{"value": 11}]
            calls.append(deepcopy(context))
            return {"literal_result": 1}

        score = registry.score

        def trusted_score(aggregate, scorer, **kwargs):
            # Retain authorization inputs at the live trusted boundary, before
            # saving the receipt that a later consumer will treat as untrusted.
            retained["binding"] = deepcopy(kwargs["binding"])
            retained["manifest_sha256"] = aggregate.manifest_sha256
            return score(aggregate, scorer, **kwargs)

        with patch.object(registry, "score", trusted_score):
            completed = cohort.evaluate_registered(lambda: {"main": batch}, evaluate)
        expected = {"registry": registry, "expected_binding": retained["binding"],
                    "expected_attempt_id": attempt_id, "expected_terminal_sha256": completed["sha256"],
                    "expected_result_sha256": digest(completed["result"])}
        saved = json.loads((root / "cohort" / "evaluation.json").read_bytes())
        assert saved == completed and len(calls) == 1
        assert retained["binding"]["endpoint_seal_sha256"] == seal["sha256"]
        assert retained["binding"]["evaluation_plan_sha256"] == digest(plan)
        assert retained["binding"]["aggregate_manifest_sha256"] == retained["manifest_sha256"]
        assert [slot["outcome"] for slot in saved["result"]["members"][0]["slots"]] == [
            "succeeded", "unavailable"]
        assert verify_registered_cohort_evaluation(saved, **expected)
        durable = registry.get_attempt(attempt_id)

        # A later unrelated registration AND development consumption change the
        # current state without revoking this historical successful authorization.
        later = registry.register_batch([{"value": 99}], source_id="later", family_id="later-family")
        later_decision = registry.consume(later, reason="literal later development")
        assert later_decision["registry_revision_after"] > durable["registry_revision_after"]

        with ExitStack() as poisoned:
            for owner, name in (
                    (DataRoleRegistry, "score"), (DataRoleRegistry, "register_batch"),
                    (DataRoleRegistry, "consume"), (DataRoleRegistry, "_snapshot"),
                    (RegisteredBatch, "materialize"), (ProgrammerSession, "evaluate_registered"),
                    (StagedCohort, "evaluate_registered"), (StagedCohort, "seal_endpoints"),
                    (StagedCohort, "_verify_endpoint_seal"), (StagedDiagnosisRepair, "evaluate_registered"),
                    (staged_cohort, "lower_structured"), (staged_cohort, "apply_program"),
                    (structured, "lower_structured"), (structured, "verify_lowering"),
                    (program, "apply_program"), (broker, "lower_structured"), (broker, "apply_program")):
                poisoned.enter_context(patch.object(owner, name, forbidden))
            lookup = poisoned.enter_context(patch.object(registry, "get_attempt", wraps=registry.get_attempt))

            def verify(record=saved, **changes):
                lookup.reset_mock()
                accepted = verify_registered_cohort_evaluation(record, **{**expected, **changes})
                assert lookup.call_count <= 1
                if lookup.called:
                    lookup.assert_called_once_with(changes.get("expected_attempt_id", attempt_id))
                return accepted

            assert verify()
            lookup.assert_called_once_with(attempt_id)
            assert len(calls) == 1

            for key, value in (("expected_attempt_id", "missing"),
                               ("expected_terminal_sha256", "0" * 64),
                               ("expected_result_sha256", "0" * 64)):
                assert not verify(**{key: value})
            for key in ("schema", "cohort_protocol_sha256", "roster_sha256", "evaluation_plan_sha256",
                        "evaluator_id", "registry_id", "endpoint_seal_sha256", "evaluation_start_sha256",
                        "aggregate_manifest_sha256", "holdouts"):
                binding = deepcopy(retained["binding"])
                binding[key] = [] if key == "holdouts" else "0" * 64
                assert not verify(expected_binding=binding), key

            # Rehashed saved substitutions cannot replace retained commitments.
            for path, value in (
                    (("outcome",), "failed"), (("outcome",), "cancelled"),
                    (("score_requested",), False), (("score_callback_entered",), False),
                    (("registry_attempt_inspection_exception_type",), "OSError"),
                    (("runtime_restoration_verified",), True),
                    (("result", "causal_usefulness_established"), True),
                    (("result", "members", 0, "slots", 0, "result"), {"literal_result": 2}),
                    (("result", "members", 0, "slots", 0, "endpoint_sha256"), "0" * 64),
                    (("result", "holdouts", 0, "row_sha256"), ["0" * 64]),
                    (("result", "aggregate", "family_id"), "substituted-family"),
                    (("registry_attempt", "status"), "authorized_outcome_unknown"),
                    (("data_role_decision", "operation"), "consume")):
                changed = deepcopy(saved)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                rehash(changed)
                assert not verify(changed), path

            def reseal(record, binding):
                # Forge mutually consistent local result/binding/attempt/decision
                # commitments. This still cannot rewrite the durable attempt.
                attempt = record["registry_attempt"]
                attempt.update(binding=deepcopy(binding), binding_sha256=digest(binding),
                    result=deepcopy(record["result"]), result_sha256=digest(record["result"]))
                reservation = {key: value for key, value in attempt.items()
                               if key not in ("reservation_sha256", "lifecycle_sha256")}
                reservation.update(status="authorized_outcome_unknown", reason="evaluation_authorization_committed",
                                   callback_invocations=None, result=None, result_sha256=None)
                attempt["reservation_sha256"] = digest(reservation)
                rehash(attempt, "lifecycle_sha256")
                decision = record["data_role_decision"]
                decision["attempt"] = deepcopy(attempt)
                for key in ("binding_sha256", "result_sha256", "reservation_sha256", "lifecycle_sha256"):
                    decision[key] = attempt[key]
                rehash(decision, "decision_sha256")
                rehash(record)
                return {"expected_binding": binding, "expected_terminal_sha256": record["sha256"],
                        "expected_result_sha256": digest(record["result"])}

            changed = deepcopy(saved)
            changed["result"]["members"][0]["slots"][0]["result"] = {"literal_result": 2}
            assert not verify(changed, **reseal(changed, retained["binding"]))
            lookup.assert_called_once_with(attempt_id)
            for key in ("cohort_protocol_sha256", "evaluation_plan_sha256", "evaluator_id", "endpoint_seal_sha256"):
                changed, binding = deepcopy(saved), deepcopy(retained["binding"])
                changed[key] = changed["result"][key] = binding[key] = "0" * 64
                assert not verify(changed, **reseal(changed, binding)), key
                lookup.assert_called_once_with(attempt_id)

            # A saved success cannot promote missing, failed or unresolved
            # durable evidence, even if its own nested copy still says success.
            for status in (None, "failed", "authorized_outcome_unknown"):
                current = None if status is None else {**deepcopy(durable), "status": status,
                    "callback_invocations": 1 if status == "failed" else None,
                    "result": None, "result_sha256": None}
                with patch.object(registry, "get_attempt", return_value=current) as missing:
                    assert not verify_registered_cohort_evaluation(saved, **expected)
                    missing.assert_called_once_with(attempt_id)
            assert verify()
            lookup.assert_called_once_with(attempt_id)
            assert len(calls) == 1
