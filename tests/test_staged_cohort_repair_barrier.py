"""One literal repair-barrier regression; no evaluation data or study execution."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.data_roles import DataRoleRegistry
from rlcd.weight_space.model import digest
from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort
from rlcd.weight_space.workflow import WorkflowError


def test_complete_repair_barrier_and_fail_stop():
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

    base = TensorArtifact("literal-repair-barrier/1", {"layer.weight": [[1.0, 2.0]]})
    answer = {"location": {"tensor": "layer.weight", "index_prefix": [0]},
              "mechanism": "Literal row offset", "predicted_effect": "Undo the literal offset",
              "confidence": 0.5, "uncertainty": "Engineering fixture only", "abstain": False}
    repair = {"commands": [{"op": "add", "tensor": "layer.weight", "values": [[-1.0, -1.0]]}],
              "intent": "Literal fixture repair"}

    with TemporaryDirectory(prefix="wsp-repair-barrier-") as directory:
        root = Path(directory)
        # No batches or scoring authorizations are created by this check.
        registry = DataRoleRegistry(root / "roles.sqlite")
        ids = ["selected", "prep-unavailable", "diagnosis-failed", "repair-failed",
               "repair-abstained", "repair-invalid", "diagnosis-abstained"]
        cohort = StagedCohort(root / "complete", member_ids=ids, protocol={}, controls={},
                              evaluator_id="literal-evaluator", registry=registry,
                              attempt_id="literal-repair-attempt")

        def prepare(row):
            if row["member_id"] == "prep-unavailable":
                raise RuntimeError("literal unavailable preparation")
            return PreparedMember(base, row["member_id"], "Check repair barrier", {})

        cohort.prepare(prepare)

        def diagnose(row, query):
            if row["member_id"] == "diagnosis-failed":
                raise RuntimeError("literal diagnosis failure")
            if row["member_id"] == "diagnosis-abstained":
                return answer | {"location": None, "abstain": True}
            return answer

        diagnosis_barrier = cohort.select_diagnoses(diagnose)
        repair_calls = []

        def choose(row, diagnosis, query):
            assert diagnosis == answer
            assert not (root / "complete" / "repair-barrier.json").exists()
            repair_calls.append(row["member_id"])
            if row["member_id"] == "repair-failed":
                raise RuntimeError("literal repair failure")
            if row["member_id"] == "repair-abstained":
                return None
            if row["member_id"] == "repair-invalid":
                return {"unexpected": True}
            return repair

        dispositions = cohort.select_repairs(choose)
        assert repair_calls == ["selected", "repair-failed", "repair-abstained", "repair-invalid"]
        barrier = read(root / "complete" / "repair-barrier.json")
        frozen = read(root / "complete" / "protocol.json")
        assert barrier == cohort._verify_repair_barrier()
        assert set(barrier) == {"schema", "cohort_protocol_sha256", "roster_sha256", "attempt_id",
                                "registry_id", "evaluator_id", "diagnosis_barrier_sha256", "members", "sha256"}
        assert barrier["schema"] == "wsp-staged-cohort-repair-barrier/1"
        assert barrier["diagnosis_barrier_sha256"] == diagnosis_barrier["sha256"]
        for key in ("roster_sha256", "attempt_id", "registry_id", "evaluator_id"):
            assert barrier[key] == frozen[key]
        assert barrier["cohort_protocol_sha256"] == frozen["sha256"]
        outcomes = ["selected", "unavailable", "unavailable", "selector_failed", "abstained",
                    "invalid_submission", "unavailable"]
        assert [item["outcome"] for item in dispositions] == outcomes
        assert [item["member_id"] for item in barrier["members"]] == ids
        assert [item["outcome"] for item in barrier["members"]] == outcomes
        for index, item in enumerate(barrier["members"]):
            diagnosis = diagnosis_barrier["members"][index]
            disposition = dispositions[index]
            assert set(item) == {"member_index", "member_id", "preparation_receipt_sha256",
                                  "diagnosis_disposition_sha256", "diagnosis_receipt_sha256",
                                  "diagnosis_outcome", "repair_disposition_sha256",
                                  "repair_receipt_sha256", "outcome"}
            assert item["member_index"] == index
            for key in ("preparation_receipt_sha256", "diagnosis_disposition_sha256",
                        "diagnosis_receipt_sha256"):
                assert item[key] == diagnosis[key] == disposition[key]
            assert item["diagnosis_outcome"] == diagnosis["outcome"] == disposition["diagnosis_outcome"]
            assert item["repair_disposition_sha256"] == disposition["sha256"]
            assert disposition == read(root / "complete" / f"repair-{index:06d}.json")
            lane_path = root / "complete" / f"member-{index:06d}" / "repair.json"
            if item["outcome"] == "unavailable":
                assert item["repair_receipt_sha256"] is None and not lane_path.exists()
            else:
                actual = read(lane_path)
                assert item["repair_receipt_sha256"] == actual["sha256"]
                assert item["outcome"] == actual["outcome"]
                assert actual["trace"][-1]["action"] == "seal"
                assert actual["task"]["constraints"]["cohort_diagnosis_barrier_sha256"] == diagnosis_barrier["sha256"]
        candidate = cohort._lanes[0]._brokers["repair"]._session._selected
        assert candidate.sha256 != candidate.artifact.sha256
        assert candidate.artifact.tensors["layer.weight"].tolist() == [[0.0, 1.0]]
        dispositions[0]["outcome"] = "caller-mutated"
        assert cohort._verify_repair_barrier() == barrier

        def fresh(name, members=None):
            instance = StagedCohort(root / name, member_ids=members or ["only"], protocol={}, controls={},
                                    evaluator_id="literal", registry=registry, attempt_id=name)
            instance.prepare(lambda row: PreparedMember(base, row["member_id"], "Check repair fail-stop", {}))
            instance.select_diagnoses(lambda row, query: answer)
            return instance

        def blocked(instance):
            calls = []
            assert instance._invalidated
            raised(instance._verify_repair_barrier)
            raised(lambda: instance.select_repairs(lambda *args: calls.append("retry")))
            assert calls == []

        # Changed aggregate or member evidence permanently invalidates the owner.
        for name, relative in (("changed-barrier", "repair-barrier.json"),
                               ("changed-repair", "member-000000/repair.json"),
                               ("changed-diagnosis", "diagnosis-000000.json")):
            instance = fresh(name)
            instance.select_repairs(lambda *args: repair)
            target = root / name / relative
            original = target.read_bytes()
            target.write_bytes(b"changed literal evidence\n")
            raised(instance._verify_repair_barrier)
            target.write_bytes(original)
            blocked(instance)

        # A cached Candidate record must not hide a changed live execution artifact.
        instance = fresh("changed-live-artifact")
        instance.select_repairs(lambda *args: repair)
        selected = instance._lanes[0]._brokers["repair"]._session._selected
        original_artifact = selected.artifact
        object.__setattr__(selected, "_artifact", base)
        raised(instance._verify_repair_barrier)
        object.__setattr__(selected, "_artifact", original_artifact)
        blocked(instance)

        instance = fresh("barrier-write-failed")
        write = instance._write
        write_error = OSError("literal repair-barrier persistence failure")

        def fail_barrier(name, body):
            if name == "repair-barrier":
                raise write_error
            return write(name, body)

        with patch.object(instance, "_write", fail_barrier):
            assert raised(lambda: instance.select_repairs(lambda *args: repair), OSError) is write_error
        assert not (root / "barrier-write-failed" / "repair-barrier.json").exists()
        assert read(root / "barrier-write-failed" / "repair-000000.json")["outcome"] == "selected"
        blocked(instance)

        # Cancellation retains its original identity even when disposition I/O also fails.
        instance = fresh("cancelled-repair", ["first", "never-reached"])
        cancellation = KeyboardInterrupt("literal repair cancellation")
        calls, queries = [], []

        def cancel(row, diagnosis, query):
            calls.append(row["member_id"])
            queries.append(query)
            raise cancellation

        write = instance._write

        def fail_disposition(name, body):
            if name == "repair-000000":
                raise OSError("literal secondary disposition failure")
            return write(name, body)

        with patch.object(instance, "_write", fail_disposition):
            assert raised(lambda: instance.select_repairs(cancel), KeyboardInterrupt) is cancellation
        assert calls == ["first"] and queries[0]({"op": "metadata"}) == {"status": "denied"}
        assert read(root / "cancelled-repair" / "member-000000" / "repair.json")["outcome"] == "selector_failed"
        assert not (root / "cancelled-repair" / "repair-barrier.json").exists()
        assert not (root / "cancelled-repair" / "member-000001" / "repair.json").exists()
        blocked(instance)
