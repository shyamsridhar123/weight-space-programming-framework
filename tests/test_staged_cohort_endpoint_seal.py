"""One literal endpoint-seal regression; no evaluation rows or study execution."""
import builtins
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rlcd.weight_space import broker, program, staged_cohort, structured
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.data_roles import DataRoleRegistry
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort
from rlcd.weight_space.staged_diagnosis import StagedDiagnosisRepair
from rlcd.weight_space.structured import StructuredProgram, lower_structured
from rlcd.weight_space.workflow import WorkflowError


def test_endpoint_reuse_static_verification_and_fail_stop():
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

    execution = {"backend": "numpy", "dtype": "float64", "device": "cpu",
                 "accumulation": "matmul", "endpoint_cast": "reject",
                 "error_model": "empirical"}
    alternate = {"backend": "torch", "dtype": "bfloat16", "device": "cuda:23",
                 "accumulation": "right_to_left", "endpoint_cast": "round",
                 "error_model": "formal"}
    selected = {"kind": "selected", "fallback_control_id": "parent"}
    no_fallback = {"kind": "selected", "fallback_control_id": None}

    def slot(identifier, source, spec=execution):
        return {"slot_id": identifier, "source": deepcopy(source),
                "execution": deepcopy(spec), "result_keys": ["literal_result"]}

    def control(identifier):
        return {"kind": "control", "control_id": identifier}

    plain = TensorArtifact("literal-endpoint-seal/1", {"layer.weight": [[1.0, 2.0]]})
    metadata = {"torch_bridge": {
        "schema": "wsp-torch-bridge/1", "torch_version": "literal-fixture",
        "module_type": "fixture.Module", "module_repr": "LiteralModule()",
        "forward_signature": "(value)",
        "modules": {"": {"type": "fixture.Module", "training": False}},
        "state": {"layer.weight": {"dtype": "torch.float32", "shape": [1, 2],
                                    "device": "mps:0", "role": "buffer",
                                    "requires_grad": True}},
        "alias_groups": [], "execution_scope": "Literal metadata; no module execution",
    }}
    captured = TensorArtifact(plain.architecture, plain.tensors, metadata)
    reference = captured.replace(**{"layer.weight": [[4.0, 8.0]]})
    proposal = StructuredProgram(captured.sha256,
        [{"op": "scale", "tensor": "layer.weight", "factor": 2.0}], "Literal control")
    lowering = lower_structured(captured, proposal,
                                native_dtypes={"layer.weight": "float32"}, cast="reject")
    structured_control = {"kind": "structured", "proposal": proposal.to_dict(),
                          "lowering": lowering.receipt}
    answer = {"location": {"tensor": "layer.weight", "index_prefix": [0]},
              "mechanism": "Literal row offset", "predicted_effect": "Undo the literal offset",
              "confidence": 0.5, "uncertainty": "Engineering fixture only", "abstain": False}
    commands = [{"op": "add", "tensor": "layer.weight", "values": [[-1.0, -1.0]]}]

    with TemporaryDirectory(prefix="wsp-endpoint-seal-") as directory, ExitStack() as guard:
        root = Path(directory)
        registry = DataRoleRegistry(root / "roles.sqlite")
        # These fixtures cannot register rows, consume authorization, or execute witnesses.
        for owner, name in [(DataRoleRegistry, "register_batch"), (DataRoleRegistry, "score"),
                            (DataRoleRegistry, "consume"),
                            (structured, "native_execution_receipt"),
                            (structured, "verify_native_execution")]:
            guard.enter_context(patch.object(owner, name,
                side_effect=AssertionError("Unexpected scoring or witness execution")))

        def completed(name, bases, slots, controls):
            plan = {"schema": "wsp-staged-cohort-evaluation-plan/1",
                    "result_schema": "wsp-staged-cohort-result/1",
                    "aggregate": {"source_id": "aggregate", "family_id": "literal-family"},
                    "holdouts": [{"holdout_id": "holdout", "source_id": "child",
                                  "family_id": "literal-family", "row_count": 1}],
                    "members": [{"member_id": identifier, "holdout_id": "holdout",
                                 "slots": slots[identifier]} for identifier in bases]}
            cohort = StagedCohort(root / name, member_ids=list(bases),
                protocol={"evaluation_plan": plan}, controls=controls,
                evaluator_id="literal-evaluator", registry=registry, attempt_id=name)

            def prepare(row):
                base = bases[row["member_id"]]
                if base is None:
                    raise RuntimeError("Literal unavailable parent")
                return PreparedMember(base, name + "/" + row["member_id"],
                                      "Check sealed literal endpoints", {})

            def repair(row, diagnosis, query):
                if row["member_id"] == "abstained":
                    return None
                return {"commands": [] if row["member_id"] == "noop" else deepcopy(commands),
                        "intent": "Literal fixture repair"}

            cohort.prepare(prepare)
            cohort.select_diagnoses(lambda row, query: deepcopy(answer))
            cohort.select_repairs(repair)
            return cohort

        controls = {
            "selected": {"parent": {"kind": "parent"}, "fallback": {"kind": "parent"},
                         "reference": {"kind": "reference", "artifact": reference.to_dict()},
                         "structured": structured_control,
                         "unused-reference": {"kind": "reference",
                                              "artifact": {"schema": "wsp-tensors/1"}}},
            "noop": {"parent": {"kind": "parent"}},
            "abstained": {"parent": {"kind": "parent"},
                          "reference": {"kind": "reference", "artifact": plain.to_dict()}},
            "unprepared": {"parent": {"kind": "parent"},
                           "reference": {"kind": "reference",
                                         "artifact": {"schema": "wsp-tensors/1"}},
                           "structured": {"kind": "structured",
                                          "proposal": {"schema": "wsp-structured-program/1"},
                                          "lowering": {"schema": "wsp-lowering/1"}}},
        }
        slots = {
            "selected": [
                slot("selected", selected | {"fallback_control_id": "fallback"}),
                slot("selected-other-execution", selected | {"fallback_control_id": "fallback"},
                     alternate),
                slot("structured", control("structured")),
                slot("structured-other-execution", control("structured"), alternate),
                slot("reference", control("reference")), slot("parent", control("parent"))],
            "noop": [slot("selected-noop", selected)],
            "abstained": [slot("fallback", selected), slot("unavailable", no_fallback),
                          slot("reference", control("reference")),
                          slot("parent-again", control("parent"), alternate)],
            "unprepared": [slot("fallback-without-parent", selected),
                           slot("reference-without-parent", control("reference")),
                           slot("structured-without-parent", control("structured")),
                           slot("selected-without-parent", no_fallback)],
        }
        cohort = completed("main", {"selected": captured, "noop": plain,
                                    "abstained": plain, "unprepared": None}, slots, controls)
        original_import = builtins.__import__

        def without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Static sealing imported Torch")
            return original_import(name, *args, **kwargs)

        with patch.object(staged_cohort, "lower_structured", wraps=lower_structured) as lower, \
                patch.object(staged_cohort, "apply_program", wraps=program.apply_program) as apply, \
                patch.object(builtins, "__import__", side_effect=without_torch):
            seal = cohort.seal_endpoints()
        # One structured control lowering, two selected literal replays, one control application.
        assert lower.call_count == 1 and apply.call_count == 3
        assert lower.call_args.kwargs == {"native_dtypes": {"layer.weight": "float32"},
                                         "cast": "reject"}
        assert seal == read(root / "main" / "endpoint-seal.json")
        assert seal["schema"] == "wsp-staged-cohort-endpoint-seal/1"
        assert seal["repair_barrier_sha256"] == read(root / "main" / "repair-barrier.json")["sha256"]
        assert all(seal[key] is False for key in ("evaluation_available",
                   "runtime_restoration_verified", "causal_usefulness_established"))
        assert [(item["member_index"], item["origin"]) for item in seal["origins"]] == [
            (0, {"kind": "selected"}), (0, control("structured")),
            (0, control("reference")), (0, control("parent")), (1, {"kind": "selected"}),
            (2, control("parent")), (2, {"kind": "selected"}), (2, control("reference")),
            (3, control("parent")), (3, control("reference")),
            (3, control("structured")), (3, {"kind": "selected"})]
        origins = {(item["member_index"], item["origin"]["kind"],
                    item["origin"].get("control_id")): item for item in seal["origins"]}
        chosen = origins[(0, "selected", None)]
        actual = read(root / "main" / "member-000000" / "repair.json")
        assert chosen["artifact"] == actual["candidate"]["artifact"]
        assert chosen["candidate_sha256"] == actual["candidate"]["sha256"]
        assert chosen["program"] == actual["submission"]["program"]
        assert chosen["lowering"] == actual["submission"]["lowering"]
        assert chosen["proposal"]["commands"] == commands
        assert chosen["construction"]["native_dtypes"] == {"layer.weight": "float64"}
        assert chosen["captured_native_dtypes"] == {"layer.weight": "float32"}
        assert chosen["captured_devices"] == {"layer.weight": "mps:0"}
        assert chosen["captured_alias_groups"] == []
        assert chosen["artifact"]["metadata"]["torch_bridge"]["state"]["layer.weight"]["requires_grad"] is True
        assert chosen["construction_capture_agreement"] == "mismatch"
        structured_origin = origins[(0, "control", "structured")]
        assert structured_origin["lowering"] == lowering.receipt
        assert structured_origin["program"] == lowering.program.to_dict()
        assert structured_origin["construction_capture_agreement"] == "agree"
        assert origins[(0, "control", "reference")]["artifact"] == reference.to_dict()
        assert origins[(0, "control", "parent")]["construction_capture_agreement"] == "not_applicable"
        noop = origins[(1, "selected", None)]
        assert noop["endpoint_sha256"] == plain.sha256 and noop["candidate_sha256"] is not None
        assert noop["program"]["edits"] == [] and noop["construction_capture_agreement"] == "absent"
        assert [item["repair_outcome"] for item in seal["members"]] == [
            "selected", "selected", "abstained", "unavailable"]
        assert [item["resolution"] for item in seal["members"][1]["slots"]] == ["selected"]
        abstained = seal["members"][2]["slots"]
        assert [item["resolution"] for item in abstained] == [
            "parent_fallback", "unavailable", "control", "control"]
        assert abstained[0]["endpoint_sha256"] == abstained[3]["endpoint_sha256"] == plain.sha256
        assert abstained[0]["resolved_origin"] == control("parent")
        assert origins[(2, "selected", None)]["status"] == "unavailable"
        for item in seal["origins"]:
            if item["member_index"] == 3:
                assert item["status"] == "unavailable"
                assert all(item[key] is None for key in ("parent_sha256", "endpoint_sha256",
                    "candidate_sha256", "artifact", "proposal", "program", "lowering", "construction"))
        assert all(item["resolution"] == "unavailable" for item in seal["members"][3]["slots"])
        assert seal["unused_controls"] == [
            {"member_index": 0, "member_id": "selected", "control_ids": ["fallback", "unused-reference"]},
            {"member_index": 1, "member_id": "noop", "control_ids": ["parent"]},
            {"member_index": 2, "member_id": "abstained", "control_ids": []},
            {"member_index": 3, "member_id": "unprepared", "control_ids": []}]
        for member in seal["members"]:
            assert len(member["slots"]) == len(slots[member["member_id"]])
            for index, (saved, declared) in enumerate(zip(member["slots"], slots[member["member_id"]])):
                assert saved["slot_index"] == index
                assert all(saved[key] == declared[key] for key in (
                    "slot_id", "source", "execution", "result_keys"))
        assert seal["members"][0]["slots"][0]["endpoint_sha256"] == chosen["endpoint_sha256"]
        assert seal["members"][0]["slots"][1]["endpoint_sha256"] == chosen["endpoint_sha256"]
        assert seal["members"][0]["slots"][2]["endpoint_sha256"] == structured_origin["endpoint_sha256"]
        assert seal["members"][0]["slots"][3]["endpoint_sha256"] == structured_origin["endpoint_sha256"]

        expected = deepcopy(seal)
        seal["origins"][0]["artifact"]["tensors"]["layer.weight"][0][0] = 999.0
        seal["members"][0]["slots"][0]["execution"]["dtype"] = "caller-mutated"
        with ExitStack() as poisoned:
            for owner, name in [(staged_cohort, "lower_structured"), (staged_cohort, "apply_program"),
                                (structured, "lower_structured"), (structured, "verify_lowering"),
                                (structured, "native_round"), (program, "apply_program"),
                                (broker, "lower_structured"), (broker, "apply_program"),
                                (StagedDiagnosisRepair, "_configuration"),
                                (broker.ProposerBroker, "_validate_parent")]:
                poisoned.enter_context(patch.object(owner, name,
                    side_effect=AssertionError("Verification reconstructed an endpoint")))
            poisoned.enter_context(patch.object(builtins, "__import__", side_effect=without_torch))
            assert cohort._verify_endpoint_seal() == expected
            assert cohort._verify_endpoint_seal() == expected
        raised(cohort.seal_endpoints)
        assert not cohort._invalidated and cohort._verify_endpoint_seal() == expected

        def single(name, base=plain, source=no_fallback, definitions=None):
            return completed(name, {"only": base}, {"only": [slot("only", source)]},
                             {"only": {} if definitions is None else definitions})

        control_only = single("control-only", source=control("parent"),
                              definitions={"parent": {"kind": "parent"}})
        assert control_only.seal_endpoints()["origins"][0]["origin"] == control("parent")

        durable = single("durable-tamper")
        durable.seal_endpoints()
        receipt_path = root / "durable-tamper" / "endpoint-seal.json"
        original = receipt_path.read_bytes()
        replaced = json.loads(original)
        replaced["members"][0]["slots"][0]["result_keys"] = ["tampered"]
        replaced["sha256"] = digest({key: value for key, value in replaced.items() if key != "sha256"})
        receipt_path.write_bytes(canonical_bytes(replaced) + b"\n")
        raised(durable._verify_endpoint_seal)
        assert durable._invalidated and durable._lanes[0]._invalidated
        receipt_path.write_bytes(original)
        raised(durable._verify_endpoint_seal)

        live = single("live-tamper")
        live.seal_endpoints()
        retained = live._endpoints[(0, "selected", None)]
        original_endpoint = retained["endpoint"]
        retained["endpoint"] = plain
        raised(live._verify_endpoint_seal)
        assert live._invalidated and live._lanes[0]._invalidated
        retained["endpoint"] = original_endpoint
        raised(live._verify_endpoint_seal)

        bad_control = deepcopy(structured_control)
        bad_receipt = bad_control["lowering"]
        bad_receipt["projections"]["layer.weight"]["rounded_cells"] = 1
        bad_receipt["sha256"] = digest({key: value for key, value in bad_receipt.items() if key != "sha256"})
        malformed_control = single("changed-control-receipt", captured, control("structured"),
                                   {"structured": bad_control})
        raised(malformed_control.seal_endpoints)
        assert malformed_control._invalidated
        assert not (root / "changed-control-receipt" / "endpoint-seal.json").exists()

        malformed_metadata = deepcopy(metadata)
        # bool compares equal to 1 in Python; it still is not a captured shape dimension.
        malformed_metadata["torch_bridge"]["state"]["layer.weight"]["shape"] = [True, 2]
        malformed_base = TensorArtifact(plain.architecture, plain.tensors, malformed_metadata)
        malformed_bridge = single("malformed-bridge", malformed_base)
        raised(malformed_bridge.seal_endpoints, ValueError)
        assert malformed_bridge._invalidated
        assert not (root / "malformed-bridge" / "endpoint-seal.json").exists()
        raised(malformed_bridge.seal_endpoints)
