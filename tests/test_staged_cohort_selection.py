"""One literal cohort-ordering regression; no evaluation rows or study execution."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rlcd.weight_space import staged_diagnosis
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.broker import WeightView
from rlcd.weight_space.data_roles import DataRoleRegistry
from rlcd.weight_space.model import digest
from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort
from rlcd.weight_space.workflow import WorkflowError


def test_cohort_barrier_and_dispositions():
    def raised(action, expected=WorkflowError):
        try:
            action()
        except BaseException as error:
            assert isinstance(error, expected), repr(error)
            return error
        raise AssertionError("Expected rejection")

    def read(path):
        document = json.loads(path.read_bytes())
        assert document["sha256"] == digest({key: value for key, value in document.items()
                                             if key != "sha256"})
        return document

    base = TensorArtifact("literal-cohort/1", {"layer.weight": [[1.0, 2.0]]})
    answer = {"location": {"tensor": "layer.weight", "index_prefix": [0]},
              "mechanism": "Literal row offset", "predicted_effect": "Undo the literal offset",
              "confidence": 0.5, "uncertainty": "Engineering fixture only", "abstain": False}
    repair = {"commands": [{"op": "add", "tensor": "layer.weight", "values": [[-1.0, -1.0]]}],
              "intent": "Literal fixture repair"}

    with TemporaryDirectory(prefix="wsp-cohort-selection-") as directory:
        root = Path(directory)
        # No batch is registered: this check cannot consume a scoring authorization.
        registry = DataRoleRegistry(root / "roles.sqlite")
        ids = ["good", "prep-failed", "config-invalid", "selector-failed", "abstained", "invalid"]
        expected_ids = list(ids)
        rows = [{"member_index": index, "member_id": identifier}
                for index, identifier in enumerate(expected_ids)]
        protocol, controls = {"thresholds": [1, 2]}, {"identity": "literal-control"}
        cohort = StagedCohort(root / "cohort", member_ids=ids, protocol=protocol,
                              controls=controls, evaluator_id="literal-evaluator",
                              registry=registry, attempt_id="literal-attempt")
        frozen = read(root / "cohort" / "protocol.json")
        assert frozen["members"] == rows and frozen["roster_sha256"] == digest(rows)

        duplicate = root / "duplicate"
        raised(lambda: StagedCohort(duplicate, member_ids=["same", "same"], protocol={}, controls={},
                                    evaluator_id="literal", registry=registry, attempt_id="duplicate"),
               ValueError)
        assert not duplicate.exists()

        def prepare(row):
            assert read(root / "cohort" / "protocol.json") == frozen
            identifier = row["member_id"]
            if row["member_index"] == 0:
                ids.append("caller-added")
                protocol["thresholds"].append(999)
                controls["identity"] = "caller-mutated"
                row["member_id"] = "mutated-callback-copy"
            if identifier == "prep-failed":
                raise RuntimeError("literal preparation failure")
            return PreparedMember(base, "" if identifier == "config-invalid" else identifier,
                                  "Check complete cohort ordering", {"fixture": identifier})

        repair_tasks, repair_brokers, repair_calls, diagnosis_calls = [], [], [], []
        repair_phase = False
        expected_outcomes = ["selected", "unavailable", "unavailable", "selector_failed",
                             "abstained", "invalid_submission"]

        def durable_barrier():
            barrier = read(root / "cohort" / "diagnosis-barrier.json")
            assert barrier["cohort_protocol_sha256"] == frozen["sha256"]
            assert barrier["roster_sha256"] == frozen["roster_sha256"]
            assert [{key: item[key] for key in ("member_index", "member_id")}
                    for item in barrier["members"]] == rows
            assert [item["outcome"] for item in barrier["members"]] == expected_outcomes
            for row, item in zip(rows, barrier["members"]):
                index = row["member_index"]
                prep = read(root / "cohort" / f"preparation-{index:06d}.json")
                diagnosis = read(root / "cohort" / f"diagnosis-{index:06d}.json")
                assert item["preparation_receipt_sha256"] == prep["sha256"]
                assert item["diagnosis_disposition_sha256"] == diagnosis["sha256"]
                lane_receipt = root / "cohort" / f"member-{index:06d}" / "diagnosis.json"
                if item["outcome"] == "unavailable":
                    assert prep["outcome"] == "unavailable"
                    assert item["diagnosis_receipt_sha256"] is None and not lane_receipt.exists()
                else:
                    actual = read(lane_receipt)
                    assert item["diagnosis_receipt_sha256"] == actual["sha256"]
                    assert item["outcome"] == actual["outcome"]
                    assert actual["trace"][-1]["action"] == "seal"
            return barrier

        original_task = staged_diagnosis.TaskSpec
        original_broker = staged_diagnosis.StagedDiagnosisRepair._broker

        def task_spy(*args, **kwargs):
            mode = kwargs.get("mode", args[1] if len(args) > 1 else None)
            if mode == "edit":
                repair_tasks.append((args, kwargs))
                assert repair_phase
                durable_barrier()
            return original_task(*args, **kwargs)

        def broker_spy(lane, task, context):
            if context["stage"] == "repair":
                repair_brokers.append(task.to_dict())
                assert repair_phase
                durable_barrier()
            return original_broker(lane, task, context)

        with patch.object(staged_diagnosis, "TaskSpec", task_spy), patch.object(
                staged_diagnosis.StagedDiagnosisRepair, "_broker", broker_spy):
            preparations = cohort.prepare(prepare)
            assert [item["member_id"] for item in preparations] == expected_ids
            assert [item["outcome"] for item in preparations] == [
                "prepared", "unavailable", "unavailable", "prepared", "prepared", "prepared"]
            assert read(root / "cohort" / "protocol.json") == frozen
            assert frozen["protocol"] == {"thresholds": [1, 2]}
            assert frozen["controls"] == {"identity": "literal-control"}
            good = cohort._lanes[0]
            expected_binding = {"cohort_protocol_sha256": frozen["sha256"],
                                "cohort_roster_sha256": frozen["roster_sha256"],
                                "cohort_member_index": 0, "cohort_member_id": "good"}
            assert read(root / "cohort" / "member-000000" / "protocol.json")[
                "cohort_binding"] == expected_binding
            raised(lambda: good.select_diagnosis(lambda query: diagnosis_calls.append("bypass")))
            raised(lambda: good.select_repair(lambda *args: repair_calls.append("bypass")))

            def diagnose(row, query):
                assert repair_tasks == [] and repair_brokers == []
                diagnosis_calls.append(row["member_id"])
                assert query({"op": "metadata"})["status"] == "ok"
                if row["member_id"] == "selector-failed":
                    raise RuntimeError("literal selector failure")
                if row["member_id"] == "abstained":
                    return answer | {"location": None, "abstain": True}
                if row["member_id"] == "invalid":
                    assert not (root / "cohort" / "diagnosis-barrier.json").exists()
                    raised(lambda: cohort.select_repairs(lambda *args: repair_calls.append("early")))
                    raised(lambda: good.select_repair(lambda *args: repair_calls.append("bypass")))
                    raised(lambda: good.evaluate_registered(None, lambda *args: repair_calls.append("score"),
                                                             evaluator_id="forbidden"))
                    assert repair_tasks == [] and repair_brokers == [] and repair_calls == []
                    return {"commands": []}
                return answer

            barrier = cohort.select_diagnoses(diagnose)
            assert barrier == durable_barrier()
            assert diagnosis_calls == ["good", "selector-failed", "abstained", "invalid"]
            assert repair_tasks == [] and repair_brokers == []
            raised(lambda: good.select_repair(lambda *args: repair_calls.append("bypass")))
            raised(lambda: good.evaluate_registered(None, lambda *args: repair_calls.append("score"),
                                                     evaluator_id="forbidden"))

            def choose_repair(row, diagnosis, query):
                assert row == rows[0] and diagnosis == answer
                assert query({"op": "metadata"})["status"] == "ok"
                repair_calls.append(row["member_id"])
                return repair

            repair_phase = True
            dispositions = cohort.select_repairs(choose_repair)
            repair_phase = False
            assert len(repair_tasks) == len(repair_brokers) == 1 and repair_calls == ["good"]
            assert [item["member_id"] for item in dispositions] == expected_ids
            assert [item["outcome"] for item in dispositions] == ["selected"] + ["unavailable"] * 5
            actual_diagnosis = read(root / "cohort" / "member-000000" / "diagnosis.json")
            constraints = repair_brokers[0]["constraints"]
            assert all(constraints[key] == value for key, value in expected_binding.items())
            assert constraints["cohort_diagnosis_barrier_sha256"] == barrier["sha256"]
            assert constraints["cohort_preparation_receipt_sha256"] == preparations[0]["sha256"]
            assert constraints["cohort_diagnosis_disposition_sha256"] == barrier["members"][0][
                "diagnosis_disposition_sha256"]
            for key, value in {
                "diagnosis_receipt_sha256": actual_diagnosis["sha256"],
                "diagnosis_candidate_sha256": actual_diagnosis["candidate"]["sha256"],
                "diagnosis_task_sha256": actual_diagnosis["task_sha256"],
                "diagnosis_session_sha256": actual_diagnosis["session_sha256"],
                "diagnosis_seal_sha256": actual_diagnosis["trace_head_sha256"],
            }.items():
                assert constraints[key] == value

            def fresh(name):
                instance = StagedCohort(root / name, member_ids=["only"], protocol={}, controls={},
                                        evaluator_id="literal", registry=registry, attempt_id=name)
                instance.prepare(lambda row: PreparedMember(base, name, "Check fail-stop", {}))
                return instance

            def blocked(instance):
                callbacks = []
                assert instance._invalidated
                raised(lambda: instance.select_diagnoses(lambda *args: callbacks.append("diagnosis")))
                raised(lambda: instance.select_repairs(lambda *args: callbacks.append("repair")))
                assert callbacks == [] and len(repair_tasks) == len(repair_brokers) == 1

            failed = fresh("selector-plus-write-failure")
            sentinel = RuntimeError("literal original selector error")
            retained_queries = []

            def fail_selector(row, query):
                retained_queries.append(query)
                raise sentinel

            with patch.object(failed._lanes[0], "_write", side_effect=OSError("literal receipt failure")):
                assert raised(lambda: failed.select_diagnoses(fail_selector), RuntimeError) is sentinel
            assert not (root / "selector-plus-write-failure" / "diagnosis-barrier.json").exists()
            assert retained_queries[0]({"op": "metadata"}) == {"status": "denied"}
            blocked(failed)

            failed = fresh("barrier-write-failure")
            write = failed._write
            write_error = OSError("literal barrier failure")

            def fail_barrier(name, body):
                if name == "diagnosis-barrier":
                    raise write_error
                return write(name, body)

            with patch.object(failed, "_write", fail_barrier):
                assert raised(lambda: failed.select_diagnoses(lambda row, query: answer), OSError) is write_error
            assert not (root / "barrier-write-failure" / "diagnosis-barrier.json").exists()
            blocked(failed)

            cancelled = fresh("cancelled")
            cancellation = KeyboardInterrupt("literal cancellation")
            cancellation_calls = []

            def cancel(row, query):
                cancellation_calls.append(row["member_id"])
                raise cancellation

            assert raised(lambda: cancelled.select_diagnoses(cancel), KeyboardInterrupt) is cancellation
            assert cancellation_calls == ["only"]
            assert read(root / "cancelled" / "member-000000" / "diagnosis.json")["outcome"] == "selector_failed"
            assert not (root / "cancelled" / "diagnosis-barrier.json").exists()
            blocked(cancelled)

            replaced = fresh("replaced")
            replaced.select_diagnoses(lambda row, query: answer)
            target = root / "replaced" / "diagnosis-000000.json"
            original = target.read_bytes()
            target.write_bytes(b"replaced evidence\n")
            raised(lambda: replaced.select_repairs(lambda *args: repair_calls.append("replaced")))
            target.write_bytes(original)
            blocked(replaced)

        invalid_config = StagedCohort(root / "invalid-config", member_ids=[
            "bad-view", "bad-dtypes", "unrepresentable", "prepared"], protocol={}, controls={},
            evaluator_id="literal", registry=registry, attempt_id="invalid-config")

        def configured(row):
            identifier = row["member_id"]
            parent = (TensorArtifact("literal-cohort/1", {"layer.weight": [[0.1, 2.0]]})
                      if identifier == "unrepresentable" else base)
            view = (WeightView("subset", coordinates=(("missing", (0,)),))
                    if identifier == "bad-view" else WeightView())
            dtypes = ({"missing": "float32"} if identifier == "bad-dtypes" else
                      {"layer.weight": "float32"} if identifier == "unrepresentable" else None)
            return PreparedMember(parent, identifier, "Check preparation validation", {},
                                  view=view, native_dtypes=dtypes)

        preparations = invalid_config.prepare(configured)
        assert [item["outcome"] for item in preparations] == ["unavailable"] * 3 + ["prepared"]
        assert all(not (root / "invalid-config" / f"member-{index:06d}").exists()
                   for index in range(3))
        assert [item["outcome"] for item in invalid_config.select_diagnoses(
            lambda row, query: answer)["members"]] == ["unavailable"] * 3 + ["selected"]

        changed_earlier = StagedCohort(root / "changed-earlier", member_ids=["first", "second"],
                                      protocol={}, controls={}, evaluator_id="literal",
                                      registry=registry, attempt_id="changed-earlier")
        changed_earlier.prepare(lambda row: PreparedMember(base, row["member_id"],
                                                           "Check complete final verification", {}))
        changed_earlier.select_diagnoses(lambda row, query: answer)
        earlier_receipt = root / "changed-earlier" / "member-000000" / "repair.json"
        earlier_bytes, repair_order, retained = [], [], []

        def mutate_earlier(row, diagnosis, query):
            repair_order.append(row["member_id"])
            retained.append(query)
            if row["member_index"] == 1:
                earlier_bytes.append(earlier_receipt.read_bytes())
                earlier_receipt.write_bytes(b"changed during a later repair\n")
            return repair

        raised(lambda: changed_earlier.select_repairs(mutate_earlier))
        assert repair_order == ["first", "second"] and changed_earlier._invalidated
        earlier_receipt.write_bytes(earlier_bytes[0])
        raised(lambda: changed_earlier.select_repairs(mutate_earlier))
        assert repair_order == ["first", "second"]
        assert all(query({"op": "metadata"}) == {"status": "denied"} for query in retained)
