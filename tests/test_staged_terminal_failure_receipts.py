"""One literal engineering regression; no provider, training or study inputs."""
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.data_roles import DataRoleRegistry
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.staged_diagnosis import StagedDiagnosisRepair
from rlcd.weight_space.workflow import BudgetLimits, ProgrammerSession, WorkflowError


def test_staged_terminal_failure_receipts():
    def raised(action, expected=WorkflowError):
        try:
            action()
        except BaseException as error:
            assert isinstance(error, expected), repr(error)
            return error
        raise AssertionError("Expected rejection")

    def throws(error):
        def callback(*args):
            raise error
        return callback

    base = TensorArtifact("literal-terminal-receipt/1", {"layer.weight": [[1.0, 2.0]]})
    answer = {"location": {"tensor": "layer.weight", "index_prefix": [0]},
              "mechanism": "Literal row offset", "predicted_effect": "Undo the literal offset",
              "confidence": 0.5, "uncertainty": "Synthetic engineering fixture only", "abstain": False}
    repair = {"commands": [{"op": "add", "tensor": "layer.weight", "values": [[-1.0, -1.0]]}],
              "intent": "Literal fixture repair"}

    with TemporaryDirectory(prefix="wsp-terminal-failures-") as directory:
        root = Path(directory)
        registry = DataRoleRegistry(root / "roles.sqlite")
        batch = registry.register_batch([{"literal": "fresh engineering row"}],
                                        source_id="literal", family_id="literal")

        def lane(stage, name, **kwargs):
            result = StagedDiagnosisRepair(root / name, task_id=name, objective="Check terminal receipts",
                                           base=base, registry=registry, **kwargs)
            if stage == "repair":
                result.select_diagnosis(lambda query: answer)
            return result

        def select(instance, stage, callback):
            return getattr(instance, "select_" + stage)(callback)

        def receipt(instance, stage, outcome):
            result = instance._verify(stage)
            assert result["outcome"] == outcome
            assert json.loads((instance._directory / (stage + ".json")).read_bytes()) == result
            assert instance._brokers[stage].is_sealed and instance._brokers[stage]._session.is_sealed
            assert result["trace"][-1]["action"] == "seal"
            return result

        def blocked(instance, stage):
            calls = []
            if stage == "diagnosis":
                raised(lambda: instance.select_repair(lambda *args: calls.append("repair")))
                assert "repair" not in instance._brokers
            raised(lambda: instance.evaluate_registered(batch, lambda *args: calls.append("evaluation"),
                                                         evaluator_id="forbidden-literal"))
            assert calls == []

        for stage in ("diagnosis", "repair"):
            for kind in (RuntimeError, KeyboardInterrupt, SystemExit):
                instance = lane(stage, stage + "-" + kind.__name__)
                failure = kind("private-selector-detail-must-not-be-recorded")
                queries = []

                def fail(*args):
                    queries.append(args[-1])
                    assert args[-1]({"op": "metadata"})["status"] == "ok"
                    raise failure

                assert raised(lambda: select(instance, stage, fail), kind) is failure
                result = receipt(instance, stage, "selector_failed")
                assert result["candidate"] is None and result["submission"] is None
                assert "private-selector-detail" not in instance._records[stage].decode()
                assert queries[0]({"op": "metadata"}) == {"status": "denied"}
                assert receipt(instance, stage, "selector_failed") == result
                raised(lambda: select(instance, stage, lambda *args: answer))
                blocked(instance, stage)

            instance = lane(stage, stage + "-schema")
            raised(lambda: select(instance, stage, lambda *args: {"private-invalid-return": "discarded"}))
            result = receipt(instance, stage, "invalid_submission")
            assert result["candidate"] is None and result["submission"] is None
            assert "private-invalid-return" not in instance._records[stage].decode()
            blocked(instance, stage)

            instance = lane(stage, stage + "-abstained")
            value = answer | {"location": None, "abstain": True} if stage == "diagnosis" else None
            select(instance, stage, lambda *args: value)
            receipt(instance, stage, "abstained")
            blocked(instance, stage)

        null_diagnosis = lane("diagnosis", "null-diagnosis")
        raised(lambda: null_diagnosis.select_diagnosis(lambda query: None))
        receipt(null_diagnosis, "diagnosis", "invalid_submission")

        capped = lane("diagnosis", "request-capped-failure")
        capped._brokers["diagnosis"]._max_requests = 1
        failure = RuntimeError("private-capped-selector")

        def exhaust_then_fail(query):
            assert query({"op": "metadata"})["status"] == "ok"
            raise failure

        assert raised(lambda: capped.select_diagnosis(exhaust_then_fail), RuntimeError) is failure
        receipt(capped, "diagnosis", "selector_failed")

        for stage in ("diagnosis", "repair"):
            value = answer if stage == "diagnosis" else repair
            admission_method = "submit_answer" if stage == "diagnosis" else "propose_artifact"
            for exhausted in (True, False):
                instance = lane(stage, stage + "-capped-return-" + str(exhausted))
                queries = []

                def return_at_cap(*args):
                    broker = instance._brokers[stage]
                    broker._max_requests = 1
                    patches.enter_context(patch.object(broker, "abstain", wraps=broker.abstain))
                    queries.append(args[-1])
                    if exhausted:
                        assert args[-1]({"op": "metadata"})["status"] == "ok"
                    return value

                with ExitStack() as patches:
                    select(instance, stage, return_at_cap)
                    assert instance._brokers[stage].abstain.call_count == 1
                result = receipt(instance, stage, "invalid_submission")
                assert result["candidate"] is None and result["selected_handle"] is None
                assert len(result["candidates"]) == (0 if exhausted else 1)
                assert queries[0]({"op": "metadata"}) == {"status": "denied"}
                assert receipt(instance, stage, "invalid_submission") == result
                blocked(instance, stage)

            instance = lane(stage, stage + "-capped-abstention")
            abstention = answer | {"location": None, "abstain": True} if stage == "diagnosis" else None

            def abstain_at_cap(*args):
                broker = instance._brokers[stage]
                broker._max_requests = 1
                patches.enter_context(patch.object(broker, "abstain", wraps=broker.abstain))
                assert args[-1]({"op": "metadata"})["status"] == "ok"
                return abstention

            with ExitStack() as patches:
                select(instance, stage, abstain_at_cap)
                assert instance._brokers[stage].abstain.call_count == 1
            receipt(instance, stage, "invalid_submission" if stage == "diagnosis" else "abstained")
            blocked(instance, stage)

            for kind in (RuntimeError, KeyboardInterrupt, SystemExit):
                instance = lane(stage, stage + "-admission-" + kind.__name__)
                failure = kind("private-admission-detail-must-not-be-recorded")

                def return_before_admission(*args):
                    broker = instance._brokers[stage]
                    patches.enter_context(patch.object(ProgrammerSession, admission_method, side_effect=failure))
                    patches.enter_context(patch.object(broker, "abstain", wraps=broker.abstain))
                    return value

                with ExitStack() as patches:
                    assert raised(lambda: select(instance, stage, return_before_admission), kind) is failure
                    assert instance._brokers[stage].abstain.call_count == 1
                result = receipt(instance, stage, "invalid_submission")
                assert result["candidate"] is None and result["submission"] is None
                assert "private-admission-detail" not in instance._records[stage].decode()
                blocked(instance, stage)

                for operation in ("abstain", "_write"):
                    instance = lane(stage, stage + "-admission-" + operation + "-" + kind.__name__)
                    failure = kind("private-original-admission")
                    queries = []

                    def return_before_admission(*args):
                        target = instance._brokers[stage] if operation == "abstain" else instance
                        patches.enter_context(patch.object(ProgrammerSession, admission_method, side_effect=failure))
                        patches.enter_context(patch.object(target, operation, side_effect=OSError("private-finalization")))
                        queries.append(args[-1])
                        return value

                    with ExitStack() as patches:
                        assert raised(lambda: select(instance, stage, return_before_admission), kind) is failure
                        target = instance._brokers[stage] if operation == "abstain" else instance
                        assert getattr(target, operation).call_count == 1
                    assert instance._invalidated and stage not in instance._records
                    assert instance._brokers[stage].is_sealed == (operation == "_write")
                    assert not (instance._directory / (stage + ".json")).exists()
                    assert queries[0]({"op": "metadata"}) == {"status": "denied"}
                    blocked(instance, stage)

        no_candidate = lane("diagnosis", "candidate-budget", limits=BudgetLimits(proposals=0))
        no_candidate.select_diagnosis(lambda query: answer)
        receipt(no_candidate, "diagnosis", "invalid_submission")
        blocked(no_candidate, "diagnosis")

        invalid_lowering = lane("repair", "invalid-lowering")
        invalid_lowering.select_repair(lambda *args: {
            "commands": [{"op": "add", "tensor": "missing", "values": [[1.0, 1.0]]}],
            "intent": "Literal invalid tensor"})
        result = receipt(invalid_lowering, "repair", "invalid_submission")
        assert result["candidate"] is None and result["submission"]["status"] == "invalid"
        blocked(invalid_lowering, "repair")

        candidate_failure = lane("repair", "candidate-failure")
        with patch.object(ProgrammerSession, "propose_artifact", side_effect=ValueError("literal candidate failure")):
            candidate_failure.select_repair(lambda *args: repair)
        receipt(candidate_failure, "repair", "invalid_submission")
        blocked(candidate_failure, "repair")

        tampered = lane("diagnosis", "tampered-failure")
        raised(lambda: tampered.select_diagnosis(throws(RuntimeError("private-tamper-detail"))), RuntimeError)
        receipt(tampered, "diagnosis", "selector_failed")
        original = tampered._records["diagnosis"]
        body = {key: value for key, value in json.loads(original).items() if key != "sha256"}
        body["outcome"] = "abstained"
        path = tampered._directory / "diagnosis.json"
        path.write_bytes(canonical_bytes(body | {"sha256": digest(body)}) + b"\n")
        raised(lambda: tampered._verify("diagnosis"))
        path.write_bytes(original)
        raised(lambda: tampered._verify("diagnosis"))
        assert tampered._invalidated
        blocked(tampered, "diagnosis")

        for index, failure in enumerate((None, RuntimeError("private-original"),
                                         KeyboardInterrupt("private-cancel"), SystemExit("private-exit"))):
            instance = lane("diagnosis", "write-failure-" + str(index))
            write_failure = OSError("private-write-failure")
            callback = throws(failure) if failure is not None else lambda query: answer
            expected = failure if failure is not None else write_failure
            with patch.object(instance, "_write", side_effect=write_failure) as writer:
                assert raised(lambda: instance.select_diagnosis(callback), type(expected)) is expected
                assert writer.call_count == 1
            assert instance._invalidated and instance._brokers["diagnosis"].is_sealed
            assert "diagnosis" not in instance._records
            assert not (instance._directory / "diagnosis.json").exists()
            raised(lambda: instance._verify("diagnosis"))

        existing = lane("diagnosis", "exclusive-receipt")
        path = existing._directory / "diagnosis.json"
        path.write_bytes(b"existing-partial-evidence\n")
        raised(lambda: existing.select_diagnosis(lambda query: answer), FileExistsError)
        assert existing._invalidated and path.read_bytes() == b"existing-partial-evidence\n"
        assert "diagnosis" not in existing._records

        seal_failure = lane("diagnosis", "seal-failure")
        failure = RuntimeError("private-original-selector")
        queries = []

        def fail_before_seal(query):
            queries.append(query)
            raise failure

        with patch.object(seal_failure._brokers["diagnosis"], "abstain", side_effect=OSError("private-seal")) as seal:
            assert raised(lambda: seal_failure.select_diagnosis(fail_before_seal), RuntimeError) is failure
            assert seal.call_count == 1
        assert seal_failure._invalidated and "diagnosis" not in seal_failure._records
        assert queries[0]({"op": "metadata"}) == {"status": "denied"}
        blocked(seal_failure, "diagnosis")

        setup_failure = lane("repair", "repair-setup-failure")
        failure = RuntimeError("private-repair-setup")
        calls = []
        with patch.object(setup_failure, "_broker", side_effect=failure):
            assert raised(lambda: setup_failure.select_repair(lambda *args: calls.append("repair")), RuntimeError) is failure
        assert setup_failure._invalidated and calls == []
        assert "repair" not in setup_failure._brokers and "repair" not in setup_failure._records

        selected = lane("diagnosis", "successful-selection")
        diagnosis = selected.select_diagnosis(lambda query: answer)
        repair_receipt = selected.select_repair(lambda *args: repair)
        receipt(selected, "diagnosis", "selected")
        receipt(selected, "repair", "selected")
        binding = repair_receipt["task"]["constraints"]
        assert binding["staged_protocol_sha256"] == diagnosis["protocol_sha256"]
        for name, value in {"receipt": diagnosis["sha256"], "candidate": diagnosis["candidate"]["sha256"],
                            "task": diagnosis["task_sha256"], "session": diagnosis["session_sha256"],
                            "seal": diagnosis["trace_head_sha256"]}.items():
            assert binding["diagnosis_" + name + "_sha256"] == value

        def evaluate(frozen, candidate, rows):
            assert frozen == answer and rows == [{"literal": "fresh engineering row"}]
            assert candidate.artifact.tensors["layer.weight"].tolist() == [[0.0, 1.0]]
            calls.append("literal-evaluation")
            return {"literal": "successful engineering check"}

        result = selected.evaluate_registered(batch, evaluate, evaluator_id="literal-terminal-receipt")
        assert result["evaluation"]["status"] == "succeeded" and calls == ["literal-evaluation"]
        raised(lambda: selected.evaluate_registered(batch, evaluate, evaluator_id="duplicate-literal"))


if __name__ == "__main__":
    test_staged_terminal_failure_receipts()
