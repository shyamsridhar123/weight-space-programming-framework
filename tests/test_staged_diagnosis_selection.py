"""Genuine-receipt engineering checks; no data registration or study execution."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.broker import ProposerBroker
from rlcd.weight_space.data_roles import DataRoleRegistry
from rlcd.weight_space.model import digest
from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort
from scripts import staged_diagnosis_selection as selection


@pytest.fixture
def reviewed_engineering_source_pins():
    # Prospective literal fixtures only; production's frozen source gate stays unchanged.
    frozen_pins = selection.SOURCE_PINS
    frozen_snapshot = dict(frozen_pins)
    assert frozen_pins["rlcd/weight_space/broker.py"] == "58a2bbb99b2dc3240e1cf7790528d85ab9b401bdee4c52eff73fdd2fe310c56c"
    fixture_pins = frozen_pins | {
        "rlcd/weight_space/broker.py": "18affe0acfc1e8f6a434aaca3e462530d0b1a05369b05a3a25be1d0b1090359b"}
    with patch.object(selection, "SOURCE_PINS", fixture_pins):
        yield frozen_pins
    assert selection.SOURCE_PINS is frozen_pins and frozen_pins == frozen_snapshot


@pytest.fixture
def prospective_summary_source_pins(reviewed_engineering_source_pins):
    frozen_fixture_pins = selection.SOURCE_PINS
    frozen_snapshot = dict(frozen_fixture_pins)
    assert frozen_fixture_pins["rlcd/weight_space/broker.py"] == (
        "18affe0acfc1e8f6a434aaca3e462530d0b1a05369b05a3a25be1d0b1090359b")
    prospective_pins = frozen_fixture_pins | {
        "rlcd/weight_space/broker.py": "52251eec40605972c00779c7d890c74c4de3d8c67d7677b01239066aae1fcede"}
    with patch.object(selection, "SOURCE_PINS", prospective_pins):
        yield
    assert selection.SOURCE_PINS is frozen_fixture_pins
    assert frozen_fixture_pins == frozen_snapshot


@pytest.mark.usefixtures("prospective_summary_source_pins")
def test_staged_selection_admits_only_live_normal_phase_returns(reviewed_engineering_source_pins):
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

        with patch.object(selection, "SOURCE_PINS", reviewed_engineering_source_pins):
            with pytest.raises(ValueError, match="^Reviewed source bytes changed$"):
                fresh("frozen-default-rejected")

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
        assert evidence["source_sha256"] == selection.SOURCE_PINS | {
            "scripts/staged_diagnosis_selection.py": driver_hash}
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

        _, _, source_changed, _ = fresh("prospective-source-drift")
        source_changed.run(prepare)
        broker_path = Path(selection.__file__).absolute().parents[1] / "rlcd/weight_space/broker.py"
        broker_bytes, read_bytes = broker_path.read_bytes(), Path.read_bytes

        def changed_bytes(path):
            return broker_bytes + b" " if path == broker_path else read_bytes(path)

        with patch.object(Path, "read_bytes", changed_bytes):
            with pytest.raises(ValueError, match="^Reviewed source bytes changed$"):
                source_changed.admit()
        with pytest.raises(ValueError, match="^This live admission is irreversibly ineligible$"):
            source_changed.admit()


@pytest.mark.usefixtures("prospective_summary_source_pins")
def test_staged_repair_usability_native_structure_and_live_binding(reviewed_engineering_source_pins):
    def rejected(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError("Expected permanent admission rejection")

    metadata = {"encoder_sha256": "1" * 64, "preprocessing_sha256": "2" * 64, "feature_id": "3" * 64}

    def head(row=0, value=4.0):
        bias = [0.0] * 8 + [-64.0]
        bias[row] = value
        return TensorArtifact("trained-frozen-readout/1", {"head.weight": [[0.0] * 32 for _ in range(9)],
                                                          "head.bias": bias}, metadata)

    # Literal private-helper checks only: these altered endpoints are not admitted selections.
    parent, good = head(), head(value=2.0)
    usable = {"status": "usable", "reason": "common_repair_structure",
              "realized_native_delta": -2.0, "realized_native_norm": 2.0}
    assert selection._repair_structure(parent, good, 0, "bias_up") == usable
    for row in (True, -1, 8, 1.0):
        assert selection._repair_structure(parent, good, row, "bias_up") == {
            "status": "unusable", "reason": "invalid_diagnosis", "realized_native_delta": None, "realized_native_norm": None}
    assert selection._repair_structure(parent, good, 0, "undetermined")["reason"] == "invalid_diagnosis"
    assert selection._repair_structure(parent, parent, 0, "bias_up") == {
        "status": "unusable", "reason": "no_op", "realized_native_delta": 0.0, "realized_native_norm": 0.0}
    assert selection._repair_structure(parent, head(value=6.0), 0, "bias_up") == {
        "status": "unusable", "reason": "wrong_direction", "realized_native_delta": 2.0, "realized_native_norm": 2.0}
    for tensor, index, value in (("head.bias", 1, 1.0), ("head.bias", 8, -63.0),
                                 ("head.weight", 0, 1.0), ("head.weight", 8, 1.0)):
        data = good.to_dict()
        if tensor == "head.weight":
            data["tensors"][tensor][index][0] = value
        else:
            data["tensors"][tensor][index] = value
        assert selection._repair_structure(parent, TensorArtifact.from_dict(data), 0, "bias_up") == {
            "status": "unusable", "reason": "outside_intended_support", "realized_native_delta": -2.0, "realized_native_norm": None}
    # Artifact capture canonicalizes signed zeros before the helper receives the retained endpoint.
    for tensor, index in (("head.bias", 1), ("head.weight", 0), ("head.weight", 8)):
        data = good.to_dict()
        if tensor == "head.weight":
            data["tensors"][tensor][index][0] = -0.0
        else:
            data["tensors"][tensor][index] = -0.0
        assert digest(data) != good.sha256
        canonical = TensorArtifact.from_dict(data)
        assert canonical.sha256 == good.sha256
        assert selection._repair_structure(parent, canonical, 0, "bias_up") == usable
    for change in ("architecture", "metadata", "tensor_names", "tensor_shape"):
        data = good.to_dict()
        if change == "architecture":
            data["architecture"] = "other-interface/1"
        elif change == "metadata":
            data["metadata"]["encoder_sha256"] = "4" * 64
        elif change == "tensor_names":
            data["tensors"]["other.bias"] = data["tensors"].pop("head.bias")
        else:
            data["tensors"]["head.bias"].pop()
        assert selection._repair_structure(parent, TensorArtifact.from_dict(data), 0, "bias_up") == {
            "status": "unusable", "reason": "interface_changed", "realized_native_delta": None, "realized_native_norm": None}
    for value in (0.1, float(2 ** 128)):
        data = good.to_dict()
        data["tensors"]["head.bias"][1] = value
        assert selection._repair_structure(parent, TensorArtifact.from_dict(data), 0, "bias_up") == {
            "status": "unusable", "reason": "not_native_float32", "realized_native_delta": None, "realized_native_norm": None}

    # Genuine admitted endpoints below use the public cohort, receipts and native lowering.
    choices = [("lower-up", 0, "bias_up", 4.0), ("upper-down", 7, "bias_down", -4.0),
               ("rounded-increase", 0, "bias_down", 33554436.0), ("rounded-decrease", 7, "bias_up", 33554436.0),
               ("native-no-op", 0, "bias_down", 33554432.0), ("denied-repair", 2, "bias_up", 4.0)]
    assignments = [{"member_id": name, "policy": "metadata-uniform", "metadata_choice": {"row": row, "direction": direction}}
                   for name, row, direction, _ in choices] + [
        {"member_id": "abstained", "policy": "bias-outlier", "metadata_choice": None},
        {"member_id": "unavailable", "policy": "bias-outlier", "metadata_choice": None}]
    parents = {name: head(row, value) for name, row, _, value in choices} | {"abstained": head(value=0.0)}
    driver_hash = hashlib.sha256(Path(selection.__file__).read_bytes()).hexdigest()

    def prepare(row):
        name = row["member_id"]
        if name == "unavailable":
            raise RuntimeError("Literal unavailable member")
        return PreparedMember(parents[name], name, "Check native repair structure", {"fixture": name})

    broker_init = ProposerBroker.__init__

    def denied_seal(owner, task, **kwargs):
        if task.mode == "edit" and task.task_id == "denied-repair/diagnosis/repair":
            kwargs["max_requests"] = 1
        broker_init(owner, task, **kwargs)

    with TemporaryDirectory(prefix="wsp-repair-structure-") as temporary:
        root = Path(temporary)
        registry = DataRoleRegistry(root / "roles.sqlite")  # Deliberately register no batches.
        protocol = {"purpose": "literal structural engineering check; no ancestry or scoring"}

        def fresh(name):
            directory = root / name
            cohort = StagedCohort(directory, member_ids=[item["member_id"] for item in assignments], protocol=protocol,
                                  controls={}, evaluator_id="literal-evaluator", registry=registry, attempt_id=name)
            owner = selection.SelectionAdmission(cohort, directory, assignments=assignments, protocol=protocol,
                                                 registry_id=registry.registry_id, evaluator_id="literal-evaluator",
                                                 attempt_id=name, driver_sha256=driver_hash)
            with patch.object(ProposerBroker, "__init__", denied_seal):
                return directory, owner, owner.run(prepare)

        directory, admission, evidence = fresh("native-structure")
        files = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
        report = admission.check_repair_usability()
        assert set(report) == {"schema", "selection_admission_sha256", "attempt_id", "registry_id", "evaluator_id",
                               "roster_sha256", "cohort_protocol_sha256", "members", "sha256"}
        assert report["schema"] == "wsp-common-repair-structure/1"
        assert report["selection_admission_sha256"] == digest(evidence)
        assert report["sha256"] == digest({key: value for key, value in report.items() if key != "sha256"})
        for key in ("attempt_id", "registry_id", "evaluator_id", "roster_sha256", "cohort_protocol_sha256"):
            assert report[key] == evidence[key]
        assert [row["member_id"] for row in report["members"]] == [item["member_id"] for item in assignments]
        assert [row["member_index"] for row in report["members"]] == list(range(len(assignments)))
        assert [row["status"] for row in report["members"]] == ["usable"] * 4 + ["unusable"] + ["unavailable"] * 3
        shared = ("member_index", "member_id", "policy", "diagnosis_outcome", "repair_outcome",
                  "diagnosis_receipt_sha256", "repair_receipt_sha256")
        numerical = ("intended_row", "prescribed_delta", "realized_native_delta", "realized_native_norm")
        selected_hashes = ("selected_repair_candidate_sha256", "selected_repair_artifact_sha256")
        for index, (checked, member) in enumerate(zip(report["members"], evidence["members"])):
            assert set(checked) == set(shared + numerical + selected_hashes + (
                "parent_sha256", "diagnosed_direction", "status", "reason"))
            assert {key: checked[key] for key in shared} == {key: member[key] for key in shared}
            assert checked["parent_sha256"] == (digest(member["parent"]) if member["parent"] is not None else None)
            candidate = member["repair_candidate"]
            if candidate is None:
                assert checked["status"] == "unavailable" and checked["reason"] == "no_selected_repair"
                assert all(checked[key] is None for key in numerical + selected_hashes + ("diagnosed_direction",))
                continue
            assert checked["selected_repair_candidate_sha256"] == candidate["sha256"] != candidate["artifact_sha256"]
            assert checked["selected_repair_artifact_sha256"] == candidate["artifact_sha256"]
            assert checked["parent_sha256"] == candidate["parent_sha256"]
            assert checked["intended_row"] == choices[index][1] and checked["diagnosed_direction"] == choices[index][2]
            assert checked["prescribed_delta"] == (-2.0 if choices[index][2] == "bias_up" else 2.0)
            delta = (-2.0, 2.0, 4.0, -4.0, 0.0)[index]
            assert checked["realized_native_delta"] == delta and checked["realized_native_norm"] == abs(delta)
            assert checked["reason"] == ("no_op" if delta == 0 else "common_repair_structure")
            assert member["repair_usable"] is (delta != 0)  # Existing admission schema stays preliminary and unchanged.
        denied = report["members"][5]
        assert denied["repair_outcome"] == "invalid_submission" and 5 in admission._artifacts
        denied_receipt = json.loads((directory / "member-000005" / "repair.json").read_bytes())
        assert denied_receipt["candidate"] is None and len(denied_receipt["candidates"]) == 1
        assert report["members"][6]["diagnosis_outcome"] == "abstained"
        assert report["members"][7]["diagnosis_outcome"] == "unavailable"
        frozen_report = json.loads(json.dumps(report, allow_nan=False))
        report["members"][0]["status"] = "changed caller copy"
        assert admission.check_repair_usability() == frozen_report and admission.admit() == evidence
        assert {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()} == files

        original_artifact = admission._artifacts[0]
        admission._artifacts[0] = parent
        rejected(admission.check_repair_usability)
        admission._artifacts[0] = original_artifact
        rejected(admission.admit)
        rejected(admission.check_repair_usability)

        directory, receipt_owner, _ = fresh("changed-during-check")
        path = directory / "repair-barrier.json"
        raw = path.read_bytes()
        structure = selection._repair_structure

        def mutate_after_structure(*args):
            result = structure(*args)
            path.write_bytes(raw + b" ")
            return result

        with patch.object(selection, "_repair_structure", mutate_after_structure):
            rejected(receipt_owner.check_repair_usability)  # The final admit catches mutation during the calculation.
        path.write_bytes(raw)
        rejected(receipt_owner.admit)
        rejected(receipt_owner.check_repair_usability)


@pytest.mark.usefixtures("prospective_summary_source_pins")
def test_staged_native_controls_bind_recipes_and_separate_native_matching(reviewed_engineering_source_pins):
    def rejected(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError("Expected permanent admission rejection")

    assignments = [
        {"member_id": "rounded", "policy": "metadata-uniform", "metadata_choice": {"row": 0, "direction": "bias_down"}},
        {"member_id": "main-no-op", "policy": "metadata-uniform", "metadata_choice": {"row": 2, "direction": "bias_down"}},
        {"member_id": "abstained", "policy": "bias-outlier", "metadata_choice": None},
        {"member_id": "unavailable", "policy": "bias-outlier", "metadata_choice": None},
    ]
    shuffles = [[1, 2, 3, 4, 5, 6, 7, 0], [2, 3, 4, 5, 6, 7, 0, 1], [3, 4, 5, 6, 7, 0, 1, 2]]
    frozen_controls = {"schema": "wsp-native-control-recipes/1", "members": [
        {"member_index": index, "member_id": item["member_id"], "random": {"row": 1, "direction": "bias_down"},
         "shuffles": [list(mapping) for mapping in shuffles]} for index, item in enumerate(assignments)]}
    metadata = {"encoder_sha256": "1" * 64, "preprocessing_sha256": "2" * 64, "feature_id": "3" * 64}
    weights = [[0.0] * 32 for _ in range(9)]
    parent = TensorArtifact("trained-frozen-readout/1", {
        "head.weight": weights, "head.bias": [33554436.0, 0.0, 33554432.0, 33554436.0, 0.0, 0.0, 0.0, 0.0, -64.0]}, metadata)
    quiet = TensorArtifact("trained-frozen-readout/1", {"head.weight": weights, "head.bias": [0.0] * 8 + [-64.0]}, metadata)
    driver_hash = hashlib.sha256(Path(selection.__file__).read_bytes()).hexdigest()

    def prepare(row):
        if row["member_id"] == "unavailable":
            raise RuntimeError("Literal unavailable parent")
        return PreparedMember(quiet if row["member_id"] == "abstained" else parent, row["member_id"],
                              "Check prebound native controls without scoring", {"fixture": row["member_id"]})

    with TemporaryDirectory(prefix="wsp-native-controls-") as temporary:
        root = Path(temporary)
        registry = DataRoleRegistry(root / "roles.sqlite")  # Deliberately register no batches.
        protocol = {"purpose": "literal native-control engineering check; no randomness or registration claim"}

        def fresh(name, supplied_controls):
            directory = root / name
            cohort = StagedCohort(directory, member_ids=[item["member_id"] for item in assignments], protocol=protocol,
                                  controls=frozen_controls, evaluator_id="literal-evaluator", registry=registry, attempt_id=name)
            admission = selection.SelectionAdmission(cohort, directory, assignments=assignments, protocol=protocol,
                registry_id=registry.registry_id, evaluator_id="literal-evaluator", attempt_id=name,
                driver_sha256=driver_hash, controls=supplied_controls)
            return directory, admission

        mismatch = json.loads(json.dumps(frozen_controls))
        mismatch["members"][0]["random"]["row"] = 7
        rejected(lambda: fresh("different-prebound-recipe", mismatch))
        invalid = json.loads(json.dumps(frozen_controls))
        invalid["members"][0]["shuffles"][0] = list(range(8))
        rejected(lambda: fresh("not-a-derangement", invalid))
        rejected(lambda: fresh("default-cannot-admit-opt-in-protocol", None))

        supplied = json.loads(json.dumps(frozen_controls))
        directory, admission = fresh("native-controls", supplied)
        protocol_raw = (directory / "protocol.json").read_bytes()
        assert json.loads(protocol_raw)["controls"] == frozen_controls
        assert json.loads(protocol_raw)["controls_sha256"] == digest(frozen_controls)
        supplied["members"][0]["random"]["row"] = 7
        supplied["members"][0]["shuffles"][0][0] = 7  # Caller mutations cannot change the captured recipe.
        evidence = admission.run(prepare)
        structure = admission.check_repair_usability()
        files = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
        lower = selection.lower_structured
        observed = []

        def observe_lowering(base, proposal, **kwargs):
            lowered = lower(base, proposal, **kwargs)
            observed.append((proposal.to_dict(), lowered.program.to_dict(), lowered.receipt))
            return lowered

        with patch.object(selection, "lower_structured", observe_lowering):
            report = admission.check_native_controls()
        assert report["schema"] == "wsp-native-control-matching/1"
        assert report["sha256"] == digest({key: value for key, value in report.items() if key != "sha256"})
        assert report["selection_admission_sha256"] == digest(evidence)
        assert report["repair_structure_sha256"] == structure["sha256"]
        assert report["controls_sha256"] == digest(frozen_controls)
        for key in ("attempt_id", "registry_id", "evaluator_id", "roster_sha256", "cohort_protocol_sha256",
                    "diagnosis_barrier_sha256", "repair_barrier_sha256", "source_sha256", "source_pin_scope"):
            assert report[key] == evidence[key]
        assert report["construction"] == {"recipe_schema": "wsp-native-control-recipes/1", "intent": "common-repair-v1",
            "prescribed_magnitude": 2.0, "native_dtypes": {"head.weight": "float32", "head.bias": "float32"}, "cast": "round"}
        assert report["norm_matching"] == {"reference": "usable_selected_main", "atol": 1e-5, "rtol": 1e-5,
                                           "rule": "abs(control-main)<=atol+rtol*abs(main)"}
        assert report["evaluation_available"] is report["final_study_seal"] is report["causal_usefulness_established"] is False
        assert [member["member_id"] for member in report["members"]] == [item["member_id"] for item in assignments]
        assert [member["member_index"] for member in report["members"]] == list(range(len(assignments)))
        for checked, main, member in zip(report["members"], structure["members"], evidence["members"]):
            assert checked["main"] == main
            assert checked["diagnosis_candidate_sha256"] == (
                member["diagnosis_candidate"]["sha256"] if member["diagnosis_candidate"] is not None else None)
            assert [control["control_id"] for control in checked["controls"]] == [
                "random", "shuffled_1", "shuffled_2", "shuffled_3", "opposite"]
        materialized = [control for member in report["members"] for control in member["controls"] if control["materialized"]]
        assert len(materialized) == len(observed)
        for control, (proposal, program, receipt) in zip(materialized, observed):
            assert control["unavailable_reason"] is None and "candidate_sha256" not in control
            assert proposal["intent"] == "common-repair-v1" and proposal["commands"] == [{
                "op": "add", "tensor": "head.bias", "block": [[control["intended_row"], control["intended_row"] + 1]],
                "values": [control["prescribed_delta"]]}]
            assert abs(control["prescribed_delta"]) == 2.0  # No tuning after native rounding or mismatch.
            assert control["proposal_sha256"] == digest(proposal) == receipt["proposal_sha256"]
            assert control["program_sha256"] == digest(program) == receipt["program_sha256"]
            assert control["lowering_sha256"] == receipt["sha256"]
            assert control["artifact_sha256"] == receipt["candidate_sha256"]

        rounded, no_op, abstained, unavailable = report["members"]
        assert rounded["main"]["realized_native_delta"] == rounded["main_reference_norm"] == 4.0
        assert rounded["main"]["selected_repair_candidate_sha256"] != rounded["main"]["selected_repair_artifact_sha256"]
        assert [control["structure"]["realized_native_delta"] for control in rounded["controls"]] == [2.0, 2.0, 0.0, 4.0, -4.0]
        assert [control["structure"]["status"] for control in rounded["controls"]] == ["usable", "usable", "unusable", "usable", "usable"]
        assert [control["matching"]["status"] for control in rounded["controls"]] == [
            "mismatched", "mismatched", "unavailable", "matched", "matched"]
        assert [control["matching"]["norm_difference"] for control in rounded["controls"]] == [2.0, 2.0, None, 0.0, 0.0]
        assert rounded["controls"][2]["structure"]["reason"] == "no_op"
        assert rounded["controls"][2]["matching"]["reason"] == "control_unusable"
        assert rounded["controls"][0]["artifact_sha256"] == rounded["controls"][1]["artifact_sha256"]
        assert no_op["main"]["status"] == "unusable" and no_op["main"]["realized_native_norm"] == 0.0
        assert no_op["main_reference_norm"] is None and all(control["materialized"] for control in no_op["controls"])
        assert all(control["matching"] == {"status": "unavailable", "reason": "no_usable_main_reference",
                                           "norm_difference": None} for control in no_op["controls"])
        assert abstained["main"]["diagnosis_outcome"] == "abstained" and abstained["main_reference_norm"] is None
        assert abstained["controls"][0]["materialized"] is True
        assert abstained["controls"][0]["structure"]["status"] == "usable"
        assert abstained["controls"][0]["matching"]["reason"] == "no_usable_main_reference"
        for control in abstained["controls"][1:]:
            assert control["materialized"] is False and control["unavailable_reason"] == "no_selected_diagnosis"
            assert control["intended_row"] is control["direction"] is control["prescribed_delta"] is None
        assert unavailable["main"]["parent_sha256"] is None and unavailable["main_reference_norm"] is None
        for control in unavailable["controls"]:
            assert control["materialized"] is False and control["unavailable_reason"] == "no_parent"
            assert all(control[key] is None for key in ("proposal_sha256", "program_sha256", "lowering_sha256", "artifact_sha256"))
            assert control["structure"] == {"status": "unavailable", "reason": "not_materialized",
                                             "realized_native_delta": None, "realized_native_norm": None}
        frozen_report = json.loads(json.dumps(report, allow_nan=False))
        report["members"][0]["controls"][0]["structure"]["status"] = "changed caller copy"
        assert admission.check_native_controls() == frozen_report and admission.admit() == evidence
        assert (directory / "protocol.json").read_bytes() == protocol_raw
        assert {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()} == files

        path = directory / "repair-barrier.json"
        raw = path.read_bytes()

        def mutate_after_lowering(*args, **kwargs):
            lowered = lower(*args, **kwargs)
            path.write_bytes(raw + b" ")
            return lowered

        with patch.object(selection, "lower_structured", mutate_after_lowering):
            rejected(admission.check_native_controls)  # Final admission catches mutation during control construction.
        path.write_bytes(raw)
        rejected(admission.admit)
        rejected(admission.check_native_controls)


@pytest.mark.usefixtures("prospective_summary_source_pins")
def test_staged_study_endpoint_seal_retains_actual_slots_without_regeneration(reviewed_engineering_source_pins):
    def rejected(action, expected=ValueError):
        try:
            action()
        except BaseException as error:
            assert isinstance(error, expected), repr(error)
            return error
        raise AssertionError("Expected permanent admission rejection")

    assignments = [
        {"member_id": "rounded", "policy": "metadata-uniform", "metadata_choice": {"row": 0, "direction": "bias_down"}},
        {"member_id": "main-no-op", "policy": "metadata-uniform", "metadata_choice": {"row": 2, "direction": "bias_down"}},
        {"member_id": "abstained", "policy": "bias-outlier", "metadata_choice": None},
        {"member_id": "denied-repair", "policy": "metadata-uniform", "metadata_choice": {"row": 0, "direction": "bias_down"}},
        {"member_id": "unavailable", "policy": "bias-outlier", "metadata_choice": None},
    ]
    shuffles = [[1, 2, 3, 4, 5, 6, 7, 0], [2, 3, 4, 5, 6, 7, 0, 1], [3, 4, 5, 6, 7, 0, 1, 2]]
    controls = {"schema": "wsp-native-control-recipes/1", "members": [
        {"member_index": index, "member_id": item["member_id"], "random": {"row": 1, "direction": "bias_down"},
         "shuffles": shuffles} for index, item in enumerate(assignments)]}
    metadata = {"encoder_sha256": "1" * 64, "preprocessing_sha256": "2" * 64, "feature_id": "3" * 64}
    weights = [[0.0] * 32 for _ in range(9)]
    parent = TensorArtifact("trained-frozen-readout/1", {"head.weight": weights,
        "head.bias": [33554436.0, 0.0, 33554432.0, 33554436.0, 0.0, 0.0, 0.0, 0.0, -64.0]}, metadata)
    quiet = TensorArtifact("trained-frozen-readout/1", {"head.weight": weights, "head.bias": [0.0] * 8 + [-64.0]}, metadata)
    driver_hash = hashlib.sha256(Path(selection.__file__).read_bytes()).hexdigest()

    def prepare(row):
        if row["member_id"] == "unavailable":
            raise RuntimeError("Literal unavailable parent")
        return PreparedMember(quiet if row["member_id"] == "abstained" else parent, row["member_id"],
                              "Check endpoint accounting without scoring", {"fixture": row["member_id"]})

    broker_init = ProposerBroker.__init__

    def denied_seal(owner, task, **kwargs):
        if task.mode == "edit" and task.task_id == "denied-repair/diagnosis/repair":
            kwargs["max_requests"] = 1
        broker_init(owner, task, **kwargs)

    with TemporaryDirectory(prefix="wsp-study-endpoint-seal-") as temporary:
        root = Path(temporary)
        registry = DataRoleRegistry(root / "roles.sqlite")  # Deliberately register no batches.

        def fresh(name, fallback="parent", supplied_controls=controls):
            protocol = {"purpose": "literal endpoint accounting; no scientific registration",
                "study_plan": {"schema": "wsp-staged-study-endpoint-plan/1", "selected_fallback": fallback}}
            directory = root / name
            cohort = StagedCohort(directory, member_ids=[item["member_id"] for item in assignments], protocol=protocol,
                                  controls=supplied_controls, evaluator_id="literal-evaluator", registry=registry, attempt_id=name)
            owner = selection.SelectionAdmission(cohort, directory, assignments=assignments, protocol=protocol,
                registry_id=registry.registry_id, evaluator_id="literal-evaluator", attempt_id=name,
                driver_sha256=driver_hash, controls=supplied_controls)
            with patch.object(ProposerBroker, "__init__", denied_seal):
                return directory, owner, owner.run(prepare)

        rejected(lambda: fresh("missing-recipes", supplied_controls={}))
        rejected(lambda: fresh("unknown-fallback", fallback="repair"))
        directory, admission, evidence = fresh("sealed")
        structure = admission.check_repair_usability()
        original_files = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
        with patch.object(selection.os, "fsync", wraps=selection.os.fsync) as fsync:
            sealed = admission.seal_study_endpoints()
            assert fsync.call_count == 1
        path = directory / "study-endpoint-seal.json"
        assert path.read_bytes() == selection._json_bytes(sealed) + b"\n"
        assert sealed["sha256"] == digest({key: value for key, value in sealed.items() if key != "sha256"})
        assert sealed["schema"] == "wsp-staged-study-endpoint-seal/1"
        assert sealed["selection_admission_sha256"] == digest(evidence)
        assert admission.admit() == evidence and "study-endpoint-seal.json" not in [row["path"] for row in evidence["receipt_files"]]
        assert sealed["study_plan"] == json.loads((directory / "protocol.json").read_bytes())["protocol"]["study_plan"]
        assert sealed["study_plan_sha256"] == digest(sealed["study_plan"])
        assert sealed["controls_sha256"] == digest(controls)
        assert sealed["assignments"] == assignments and sealed["assignments_sha256"] == digest(assignments)
        for key in ("attempt_id", "registry_id", "evaluator_id", "roster_sha256", "cohort_protocol_sha256",
                    "diagnosis_barrier_sha256", "repair_barrier_sha256", "source_sha256", "source_pin_scope"):
            assert sealed[key] == evidence[key]
        for key in ("evaluation_available", "final_study_seal", "trained_ancestry_established", "causal_usefulness_established"):
            assert sealed[key] is False
        assert sealed["repair_structure"] == structure and sealed["repair_structure_sha256"] == structure["sha256"]
        native = sealed["native_controls"]
        assert sealed["native_controls_sha256"] == native["sha256"] == digest({key: value for key, value in native.items() if key != "sha256"})
        assert native["selection_admission_sha256"] == digest(evidence) and native["repair_structure_sha256"] == structure["sha256"]
        assert [row["member_id"] for row in sealed["members"]] == [row["member_id"] for row in assignments]
        for member, selected, report in zip(sealed["members"], evidence["members"], native["members"]):
            index, slots = member["member_index"], member["slots"]
            assert [slot["slot_index"] for slot in slots] == list(range(7))
            assert [slot["slot_id"] for slot in slots] == ["main", "parent", "random", "shuffled_1", "shuffled_2", "shuffled_3", "opposite"]
            assert member["diagnosis_candidate"] == selected["diagnosis_candidate"]
            assert slots[0]["selected_candidate"] == selected["repair_candidate"]
            if selected["repair_candidate"] is not None:
                assert slots[0]["resolution"] == "selected" and slots[0]["materialized"] is True
                assert slots[0]["selected_candidate_sha256"] == selected["repair_candidate"]["sha256"] != slots[0]["artifact_sha256"]
                assert slots[0]["construction"] == json.loads((directory / f"member-{index:06d}" / "repair.json").read_bytes())["submission"]
            for slot in slots:
                assert slot["artifact_sha256"] == (digest(slot["artifact"]) if slot["artifact"] is not None else None)
            for slot, control in zip(slots[2:], report["controls"]):
                assert slot["selected_candidate"] is slot["selected_candidate_sha256"] is None
                assert slot["structure"] == control["structure"] and slot["matching"] == control["matching"]
                assert slot["materialized"] is control["materialized"]
                assert slot["resolution"] == ("materialized" if control["materialized"] else "unavailable")
                assert slot["unavailable_reason"] == control["unavailable_reason"]
                if slot["materialized"]:
                    construction = slot["construction"]
                    for key in ("proposal", "program"):
                        assert digest(construction[key]) == control[key + "_sha256"] == construction["lowering"][key + "_sha256"]
                    lowering = construction["lowering"]
                    assert lowering["sha256"] == control["lowering_sha256"] == digest({key: value for key, value in lowering.items() if key != "sha256"})
                    assert slot["artifact_sha256"] == control["artifact_sha256"] == lowering["candidate_sha256"]
                else:
                    assert slot["artifact"] is slot["artifact_sha256"] is slot["construction"] is None
        rounded, no_op, abstained, denied, unavailable = sealed["members"]
        assert rounded["slots"][2]["artifact_sha256"] == rounded["slots"][3]["artifact_sha256"]
        assert [slot["structure"]["realized_native_delta"] for slot in rounded["slots"][2:]] == [2.0, 2.0, 0.0, 4.0, -4.0]
        assert rounded["slots"][2]["matching"]["status"] == "mismatched" and rounded["slots"][4]["structure"]["reason"] == "no_op"
        assert no_op["slots"][0]["resolution"] == "selected" and no_op["slots"][0]["structure"]["status"] == "unusable"
        assert no_op["slots"][0]["artifact_sha256"] == no_op["slots"][1]["artifact_sha256"]
        assert all(slot["matching"]["reason"] == "no_usable_main_reference" for slot in no_op["slots"][2:])
        for member in (abstained, denied):
            assert member["slots"][0]["resolution"] == "parent_fallback" and member["slots"][0]["selected_candidate"] is None
            assert member["slots"][0]["artifact_sha256"] == member["slots"][1]["artifact_sha256"]
        assert 3 in admission._artifacts and admission._artifacts[3].sha256 != denied["slots"][0]["artifact_sha256"]
        assert abstained["slots"][2]["materialized"] is True and all(not slot["materialized"] for slot in abstained["slots"][3:])
        assert all(slot["resolution"] == "unavailable" and slot["artifact"] is None for slot in unavailable["slots"])
        frozen = json.loads(json.dumps(sealed, allow_nan=False))
        sealed["members"][0]["slots"][2]["artifact"]["tensors"]["head.bias"][1] = 999.0
        sealed["members"][0]["slots"][2]["construction"]["proposal"]["intent"] = "changed caller copy"
        with patch.object(selection, "lower_structured", side_effect=AssertionError("No relowering")), \
             patch.object(selection, "verify_lowering", side_effect=AssertionError("No reconstruction")), \
             patch.object(selection, "apply_program", side_effect=AssertionError("No reapplication")):
            assert admission.verify_study_endpoints() == admission.verify_study_endpoints() == frozen
            assert admission.check_native_controls() == frozen["native_controls"]
            assert admission.admit() == evidence
        assert {name: (directory / name).read_bytes() for name in original_files} == original_files
        assert set(admission._saved) == {str(name).replace("\\", "/") for name in original_files} | {"study-endpoint-seal.json"}
        key = (0, "random")
        original = admission._study_controls[key]
        admission._study_controls[key] = (quiet, original[1])
        rejected(admission.verify_study_endpoints)
        admission._study_controls[key] = original
        rejected(admission.verify_study_endpoints)
        rejected(admission.admit)

        directory, owner, _ = fresh("unavailable-fallback", fallback="unavailable")
        sealed = owner.seal_study_endpoints()
        assert all(sealed["members"][index]["slots"][0]["resolution"] == "unavailable" for index in (2, 3, 4))
        assert all(sealed["members"][index]["slots"][1]["resolution"] == "parent" for index in (2, 3))
        raw = (directory / "study-endpoint-seal.json").read_bytes()
        rejected(owner.seal_study_endpoints)  # A second seal closes the successful owner without overwriting.
        assert (directory / "study-endpoint-seal.json").read_bytes() == raw
        rejected(owner.verify_study_endpoints)

        directory, owner, _ = fresh("interrupted-persistence")
        signal = KeyboardInterrupt("Literal fsync interruption")
        with patch.object(selection.os, "fsync", side_effect=signal):
            assert rejected(owner.seal_study_endpoints, KeyboardInterrupt) is signal
        raw = (directory / "study-endpoint-seal.json").read_bytes()
        assert owner._study_seal is None  # Durable-looking bytes cannot supply a missing successful live return.
        rejected(owner.verify_study_endpoints)
        rejected(owner.seal_study_endpoints)
        assert (directory / "study-endpoint-seal.json").read_bytes() == raw
