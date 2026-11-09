"""One genuine-receipt engineering check; no data registration or study execution."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.broker import ProposerBroker
from rlcd.weight_space.data_roles import DataRoleRegistry
from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort
from scripts import staged_diagnosis_selection as selection


def test_staged_selection_admits_only_live_normal_phase_returns():
    def rejected(action, expected=ValueError):
        try:
            action()
        except BaseException as error:
            assert isinstance(error, expected), repr(error)
            return
        raise AssertionError("Expected admission rejection")

    assignments = [{"member_id": "outlier", "policy": "bias-outlier", "metadata_choice": None},
                   {"member_id": "metadata", "policy": "metadata-uniform",
                    "metadata_choice": {"row": 2, "direction": "bias_down"}},
                   {"member_id": "abstain", "policy": "bias-outlier", "metadata_choice": None},
                   {"member_id": "unavailable", "policy": "bias-outlier", "metadata_choice": None}]
    weight = [[0.0] * 32 for _ in range(9)]
    metadata = {"encoder_sha256": "1" * 64, "preprocessing_sha256": "2" * 64, "feature_id": "3" * 64}
    ordinary = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, -64.0]
    outlier = TensorArtifact("trained-frozen-readout/1", {"head.weight": weight, "head.bias": ordinary}, metadata)
    quiet = TensorArtifact("trained-frozen-readout/1", {"head.weight": weight, "head.bias": [0.0] * 8 + [-64.0]}, metadata)
    driver_hash = hashlib.sha256(Path(selection.__file__).read_bytes()).hexdigest()

    def prepare(row):
        if row["member_id"] == "unavailable":
            raise RuntimeError("Literal preparation failure")
        return PreparedMember(quiet if row["member_id"] == "abstain" else outlier,
                              row["member_id"], "Check selection admission with literal heads", {"fixture": row["member_id"]})

    with TemporaryDirectory(prefix="wsp-selection-admission-") as temporary:
        root = Path(temporary)
        registry = DataRoleRegistry(root / "roles.sqlite")  # Deliberately register no batches.
        protocol = {"purpose": "literal engineering fixture; no trained ancestry or scoring"}

        def fresh(name):
            directory = root / name
            cohort = StagedCohort(directory, member_ids=[item["member_id"] for item in assignments],
                                  protocol=protocol, controls={}, evaluator_id="literal-evaluator",
                                  registry=registry, attempt_id=name)
            arguments = dict(assignments=assignments, protocol=protocol, registry_id=registry.registry_id,
                             evaluator_id="literal-evaluator", attempt_id=name, driver_sha256=driver_hash)
            return directory, cohort, selection.SelectionAdmission(cohort, directory, **arguments), arguments

        directory, cohort, admission, arguments = fresh("accepted")
        rejected(lambda: selection.SelectionAdmission(cohort, directory, **arguments))
        evidence = admission.run(prepare)
        assert admission.admit() == evidence
        assert evidence["evaluation_available"] is False and evidence["final_study_seal"] is False
        assert [row["diagnosis_outcome"] for row in evidence["members"]] == ["selected", "selected", "abstained", "unavailable"]
        assert [row["repair_outcome"] for row in evidence["members"]] == ["selected", "selected", "unavailable", "unavailable"]
        first, second, abstained, unavailable = evidence["members"]
        assert first["diagnosis_candidate"]["answer"]["location"] == {"tensor": "head.bias", "index_prefix": [3]}
        assert first["repair_candidate"]["artifact"]["tensors"]["head.bias"] == ordinary[:3] + [2.0] + ordinary[4:]
        assert first["repair_candidate"]["kind"] == "artifact" and first["repair_candidate"]["program_sha256"] is None
        assert first["repair_candidate"]["sha256"] != first["repair_candidate"]["artifact_sha256"]
        assert second["repair_candidate"]["artifact"]["tensors"]["head.bias"][2] == 2.0
        assert json.loads(second["diagnosis_candidate"]["answer"]["predicted_effect"])["p"] == 0.5
        assert abstained["diagnosis_candidate"]["answer"]["confidence"] == 0
        assert abstained["diagnosis_candidate"]["answer"]["abstain"] is True
        assert unavailable["diagnosis_candidate"] is None and unavailable["parent"] is None
        assert set(evidence["phase_return_sha256"]) == {"preparation", "diagnosis", "repair"}
        assert not any("evaluation" in item["path"] for item in evidence["receipt_files"])
        rejected(lambda: selection.SelectionAdmission(cohort, directory, **arguments))
        path = directory / "member-000000" / "diagnosis.json"
        raw = path.read_bytes()
        path.write_bytes(raw + b" ")
        rejected(admission.admit)
        path.write_bytes(raw)
        rejected(admission.admit)  # Restoring bytes never restores live eligibility.

        broker_init = ProposerBroker.__init__
        for name, diagnosis_caps, repair_cap, failed in (
                ("diagnosis-submit-denied", (1, 100), None, [(0, False), (2, False)]),
                ("diagnosis-seal-denied", (2, 1), None, [(0, True), (1, True), (2, True)]),
                ("repair-seal-denied", None, 1, [(0, True), (1, True)])):
            def bounded_broker(owner, task, **kwargs):
                if task.mode == "diagnose" and diagnosis_caps is not None:
                    kwargs["max_requests"] = diagnosis_caps[0 if kwargs["view"].mode == "subset" else 1]
                elif task.mode == "edit" and repair_cap is not None:
                    kwargs["max_requests"] = repair_cap
                broker_init(owner, task, **kwargs)

            with patch.object(ProposerBroker, "__init__", bounded_broker):
                directory, _, capped, _ = fresh(name)
                capped_evidence = capped.run(prepare)
            assert capped.admit() == capped_evidence
            stage = "repair" if repair_cap is not None else "diagnosis"
            for index, created in failed:
                document = json.loads((directory / f"member-{index:06d}" / (stage + ".json")).read_bytes())
                assert document["outcome"] == "invalid_submission"
                assert document["selected_handle"] is None and document["candidate"] is None and document["submission"] is None
                assert len(document["candidates"]) == int(created)
                broker = document["broker_trace"]
                proposal = next(event for event in broker if event["request"]["op"] in ("answer", "submit"))
                handle = proposal["response"].get("candidate_handle")
                assert broker[-2]["request"] == {"op": "seal", "candidate_handle": handle}
                assert broker[-2]["response"] == {"status": "denied"}
                assert broker[-1]["request"] == {"op": "harness_deadline_seal"}
                assert broker[-1]["response"] == {"status": "sealed"}
                assert document["trace"][-1]["request"] == {"candidate_sha256": None}
                action = "submit_answer" if stage == "diagnosis" else "propose_artifact"
                proposals = [event for event in document["trace"] if event["action"] == action]
                if created:
                    attempted = document["candidates"][0]
                    assert proposal["response"] == {"status": "submitted", "candidate_handle": handle}
                    assert isinstance(handle, str) and len(handle) == 48
                    assert len(proposals) == 1 and proposals[0]["status"] == "succeeded"
                    assert proposals[0]["result"] == {"candidate_sha256": attempted["sha256"],
                        "artifact_sha256": attempted["artifact_sha256"], "no_op": attempted["no_op"]}
                    if stage == "diagnosis":
                        assert attempted["answer"] == evidence["members"][index]["diagnosis_candidate"]["answer"]
                    else:
                        assert attempted["artifact"] == evidence["members"][index]["repair_candidate"]["artifact"]
                        assert proposals[0]["request"]["artifact_sha256"] == attempted["artifact_sha256"]
                else:
                    assert proposal["response"] == {"status": "denied"} and not proposals
                member = capped_evidence["members"][index]
                assert member[stage + "_outcome"] == "invalid_submission" and member[stage + "_candidate"] is None
                assert member["repair_candidate"] is None and member["repair_usable"] is False
                if stage == "diagnosis":
                    assert member["repair_outcome"] == "unavailable"

        directory, _, compromised, _ = fresh("swallowed-integrity-failure")
        select_diagnoses = StagedCohort.select_diagnoses
        observed = []

        def tamper_during_selector(owner, selector):
            def restoring_selector(row, query):
                if row["member_index"] != 0:
                    return selector(row, query)
                path = directory / "preparation-000000.json"
                raw = path.read_bytes()
                path.write_bytes(raw + b" ")
                try:
                    return selector(row, query)
                except ValueError:
                    observed.append("integrity failure reached the cohort")
                    raise
                finally:
                    path.write_bytes(raw)
            return select_diagnoses(owner, restoring_selector)

        with patch.object(StagedCohort, "select_diagnoses", tamper_during_selector):
            rejected(lambda: compromised.run(prepare))
        assert observed and (directory / "diagnosis-barrier.json").is_file()
        rejected(compromised.admit)  # The cohort swallowed the error after bytes were restored.

        for method, receipt in (("prepare", "preparation-000000.json"),
                                ("select_diagnoses", "diagnosis-barrier.json"),
                                ("select_repairs", "repair-barrier.json")):
            for error_type in (RuntimeError, KeyboardInterrupt):
                directory, _, interrupted, _ = fresh(method + "-" + error_type.__name__)
                original = getattr(StagedCohort, method)

                def return_then_raise(owner, *args, **kwargs):
                    original(owner, *args, **kwargs)
                    raise error_type("After genuine durable public phase evidence")

                with patch.object(StagedCohort, method, return_then_raise):
                    rejected(lambda: interrupted.run(prepare), error_type)
                assert (directory / receipt).is_file()
                rejected(interrupted.admit)
