"""One literal cooperative-registry regression; no research loaders or studies."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rlcd.weight_space import broker, program, staged_cohort, structured
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.data_roles import DataRoleError, DataRoleRegistry, RegisteredBatch
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort
from rlcd.weight_space.staged_diagnosis import StagedDiagnosisRepair
from rlcd.weight_space.workflow import WorkflowError


def test_evaluation_bridge_authorization_and_fail_stop():
    class RaisingMetadataError(RuntimeError):
        def __getattr__(self, name):
            if name == "data_role_decision":
                raise SystemExit("Literal secondary metadata cancellation")
            raise AttributeError(name)

    class RaisingMetadataCancellation(KeyboardInterrupt):
        @property
        def data_role_decision(self):
            raise OSError("Literal secondary metadata failure")

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
    base = TensorArtifact("literal-cohort-evaluation/1", {"layer.weight": [[1.0, 2.0]]})
    answer = {"location": {"tensor": "layer.weight", "index_prefix": [0]},
              "mechanism": "Literal row offset", "predicted_effect": "Undo the literal offset",
              "confidence": 0.5, "uncertainty": "Engineering fixture only", "abstain": False}
    literal_rows = {"main": [{"value": 11}, {"value": 12}],
                    "other": [{"value": 21}], "unused": [{"value": 31}]}

    def slot(identifier, source=selected, spec=execution):
        return {"slot_id": identifier, "source": deepcopy(source),
                "execution": deepcopy(spec), "result_keys": ["literal_result"]}

    with TemporaryDirectory(prefix="wsp-cohort-evaluation-") as directory:
        root = Path(directory)

        def completed(name):
            registry = DataRoleRegistry(root / (name + "-roles.sqlite"))
            batches = {identifier: registry.register_batch(rows, source_id=identifier + "-source",
                        family_id=identifier + "-family") for identifier, rows in literal_rows.items()}
            members = ["selected", "abstained", "unprepared"]
            plan = {"schema": "wsp-staged-cohort-evaluation-plan/1",
                    "result_schema": "wsp-staged-cohort-result/1",
                    "aggregate": {"source_id": "aggregate", "family_id": "aggregate-family"},
                    "holdouts": [{"holdout_id": identifier, "source_id": identifier + "-source",
                                  "family_id": identifier + "-family", "row_count": len(rows)}
                                 for identifier, rows in literal_rows.items()],
                    "members": [
                        {"member_id": "selected", "holdout_id": "main", "slots": [
                            slot("first"), slot("second", spec=alternate), slot("later")]},
                        {"member_id": "abstained", "holdout_id": "other", "slots": [
                            slot("fallback"), slot("unavailable", {
                                "kind": "selected", "fallback_control_id": None})]},
                        {"member_id": "unprepared", "holdout_id": "main", "slots": [slot("missing")]},
                    ]}
            cohort = StagedCohort(root / name, member_ids=members,
                protocol={"evaluation_plan": plan},
                controls={identifier: {"parent": {"kind": "parent"}} for identifier in members},
                evaluator_id="literal-evaluator", registry=registry, attempt_id=name)

            def prepare(row):
                if row["member_id"] == "unprepared":
                    raise RuntimeError("Literal unavailable parent")
                return PreparedMember(base, name + "/" + row["member_id"],
                                      "Check literal cohort evaluation", {})

            cohort.prepare(prepare)
            cohort.select_diagnoses(lambda row, query: deepcopy(answer))
            cohort.select_repairs(lambda row, diagnosis, query: None
                if row["member_id"] == "abstained" else {"commands": [], "intent": "Literal repair"})
            seal = cohort.seal_endpoints()
            return cohort, registry, batches, seal

        def slots(terminal):
            return [slot for member in terminal["result"]["members"] for slot in member["slots"]]

        def failed(name, cohort, registry, expected, invocations):
            terminal = read(root / name / "evaluation.json")
            assert terminal["exception_type"] == type(expected).__name__
            assert terminal["outcome"] == ("failed" if isinstance(expected, Exception) else "cancelled")
            assert [slot["evaluator_invocations"] for slot in slots(terminal)] == invocations
            assert cohort._invalidated and cohort._started["evaluation"]
            raised(lambda: cohort.evaluate_registered(lambda: {}, lambda *args: {}))
            return terminal

        cohort, registry, batches, seal = completed("success")
        loads, calls = [], []

        def load():
            loads.append(read(root / "success" / "evaluation-start.json"))
            # Caller mapping order cannot reorder the declared aggregate.
            return dict(reversed(list(batches.items())))

        def evaluate(context, endpoint, rows):
            member_index, declared = context["member_index"], context["slot"]
            assert declared == seal["members"][member_index]["slots"][declared["slot_index"]]
            origin = declared["resolved_origin"]
            assert endpoint is cohort._endpoints[(member_index, origin["kind"],
                                                  origin.get("control_id"))]["endpoint"]
            assert rows == literal_rows[context["holdout_id"]]
            # Registry transactions stay outside its scoring callback.
            assert raised(lambda: registry.get_attempt("success"), DataRoleError).decision["reason"] == "callback_reentry"
            calls.append(deepcopy(context))
            rows[0]["value"] = -1
            context["slot"]["execution"]["dtype"] = "caller-mutated"
            return {"literal_result": len(calls)}

        with ExitStack() as poisoned:
            for owner, name in [(staged_cohort, "lower_structured"), (staged_cohort, "apply_program"),
                                (structured, "lower_structured"), (structured, "verify_lowering"),
                                (structured, "native_round"), (program, "apply_program"),
                                (broker, "lower_structured"), (broker, "apply_program"),
                                (StagedDiagnosisRepair, "_configuration"),
                                (broker.ProposerBroker, "_validate_parent")]:
                poisoned.enter_context(patch.object(owner, name,
                    side_effect=AssertionError("Evaluation reconstructed an endpoint")))
            registered = poisoned.enter_context(patch.object(registry, "register_batch", wraps=registry.register_batch))
            scored = poisoned.enter_context(patch.object(registry, "score", wraps=registry.score))
            terminal = cohort.evaluate_registered(load, evaluate)
            assert registered.call_count == scored.call_count == 1
            assert registered.call_args.args == ([row for rows in literal_rows.values() for row in rows],)
            assert registered.call_args.kwargs["parents"] == tuple(batches.values())
            assert scored.call_args.kwargs["attempt_id"] == "success"
            assert scored.call_args.kwargs["binding"]["endpoint_seal_sha256"] == seal["sha256"]
            raised(lambda: cohort.evaluate_registered(load, evaluate))
            assert registered.call_count == scored.call_count == 1
        assert len(loads) == 1 and len(calls) == 4
        assert [context["slot"]["execution"] for context in calls[:2]] == [execution, alternate]
        assert terminal == read(root / "success" / "evaluation.json")
        result = terminal["result"]
        assert terminal["outcome"] == "succeeded" and not cohort._invalidated
        assert [item["holdout_id"] for item in result["holdouts"]] == list(literal_rows)
        assert [(item["start"], item["stop"]) for item in result["holdouts"]] == [(0, 2), (2, 3), (3, 4)]
        assert result["aggregate"]["parent_manifest_sha256"] == {
            batch.to_dict()["source_id"]: batch.manifest_sha256 for batch in batches.values()}
        assert [item["outcome"] for item in slots(terminal)] == [
            "succeeded", "succeeded", "succeeded", "succeeded", "unavailable", "unavailable"]
        assert terminal["registry_attempt"] == registry.get_attempt("success") == terminal["data_role_decision"]["attempt"]
        assert terminal["registry_attempt"]["result"] == result
        assert terminal["registry_attempt"]["result_sha256"] == digest(result)
        for member in result["members"]:
            for item in member["slots"]:
                receipt = read(root / "success" /
                    f"evaluation-{member['member_index']:06d}-{item['slot_index']:06d}.json")
                assert receipt["sha256"] == item["receipt_sha256"]
                assert receipt["evaluation_start_sha256"] == loads[0]["sha256"]
                assert receipt["endpoint_seal_sha256"] == seal["sha256"]
                assert receipt["outcome"] == item["outcome"]
        raised(lambda: cohort._lanes[0].evaluate_registered(batches["main"], evaluate,
                                                           evaluator_id="literal-evaluator"))

        cohort, registry, batches, _ = completed("forged")
        forged_rows = [{"value": 111}, {"value": 112}]
        manifest = batches["main"].to_dict()
        manifest.pop("manifest_sha256")
        manifest.update(content_sha256=digest(forged_rows), row_sha256=[digest(row) for row in forged_rows])
        batches["main"] = RegisteredBatch._from_bytes(canonical_bytes(forged_rows), canonical_bytes(manifest))
        with patch.object(registry, "score", wraps=registry.score) as score:
            error = raised(lambda: cohort.evaluate_registered(lambda: batches,
                lambda *args: {"literal_result": 1}), DataRoleError)
            assert error.decision["reason"] == "registration_mismatch" and score.call_count == 0
        terminal = failed("forged", cohort, registry, error, [0, 0, 0, 0, 0, 0])
        assert terminal["failure_phase"] == "aggregate_registration"
        assert not terminal["score_requested"] and registry.get_attempt("forged") is None

        for name in ("consumed-input", "before-bridge-callback"):
            cohort, registry, batches, _ = completed(name)
            if name == "consumed-input":
                registry.consume(batches["unused"], reason="Literal prior development use")
            original_score, rejections, calls = registry.score, [], []

            def no_evaluator(*args):
                calls.append(args)
                raise AssertionError("Bridge evaluator must not run")

            def score_at_boundary(batch, score_all, **options):
                def before_bridge(rows):
                    raise RuntimeError("Literal registry callback failure before bridge entry")

                try:
                    return original_score(batch, before_bridge
                        if name == "before-bridge-callback" else score_all, **options)
                except DataRoleError as rejection:
                    rejections.append(rejection)
                    raise

            with patch.object(registry, "score", side_effect=score_at_boundary) as score:
                error = raised(lambda: cohort.evaluate_registered(lambda: batches, no_evaluator), DataRoleError)
                assert score.call_count == 1 and len(rejections) == 1 and error is rejections[0]
            terminal = failed(name, cohort, registry, error, [0, 0, 0, 0, 0, 0])
            assert calls == [] and terminal["score_requested"] and not terminal["score_callback_entered"]
            assert terminal["data_role_decision"] == error.decision
            assert [item["outcome"] for item in slots(terminal)] == ["not_run"] * 4 + ["unavailable"] * 2
            if name == "consumed-input":
                assert error.decision["reason"] == "observed_development_consumption"
                assert terminal["registry_attempt"] is None and registry.get_attempt(name) is None
            else:
                attempt = registry.get_attempt(name)
                assert terminal["registry_attempt"] == attempt == error.decision["attempt"]
                assert attempt["status"] == "failed" and attempt["callback_invocations"] == 1
                assert attempt["evidence_role"] == "evaluation_consumed_on_authorization"
                assert ["source", "unused-source"] in attempt["consumed_scope"]

        cohort, registry, _, _ = completed("loader-failure")
        original = RaisingMetadataError("Literal loader failure")

        def broken_loader():
            read(root / "loader-failure" / "evaluation-start.json")
            raise original

        assert raised(lambda: cohort.evaluate_registered(broken_loader, evaluate), RuntimeError) is original
        terminal = failed("loader-failure", cohort, registry, original, [0, 0, 0, 0, 0, 0])
        assert [item["outcome"] for item in slots(terminal)] == ["not_run"] * 4 + ["unavailable"] * 2
        assert not terminal["score_requested"] and registry.get_attempt("loader-failure") is None

        for index, original in enumerate((RuntimeError("Literal evaluator failure"),
                                          KeyboardInterrupt("Literal cancellation"), SystemExit(23),
                                          RaisingMetadataCancellation("Literal guarded cancellation"))):
            name = "evaluator-failure-" + str(index)
            cohort, registry, batches, _ = completed(name)
            calls = []

            def fail_second(context, endpoint, rows):
                calls.append(context["slot"]["slot_id"])
                if len(calls) == 2:
                    raise original
                return {"literal_result": 1}

            assert raised(lambda: cohort.evaluate_registered(lambda: batches, fail_second),
                          type(original)) is original
            terminal = failed(name, cohort, registry, original, [1, 1, 0, 0, 0, 0])
            assert calls == ["first", "second"]
            assert [item["outcome"] for item in slots(terminal)] == [
                "succeeded", "failed", "not_run", "not_run", "unavailable", "unavailable"]
            assert slots(terminal)[0]["receipt_sha256"] and slots(terminal)[1]["receipt_sha256"]
            assert terminal["registry_attempt"] == registry.get_attempt(name)
            if isinstance(original, RaisingMetadataCancellation):
                # Its read-only metadata prevents registry finalization; authorization stays spent.
                assert terminal["data_role_decision"] is None
                assert terminal["registry_attempt"]["status"] == "authorized_outcome_unknown"
                assert terminal["registry_attempt"]["callback_invocations"] is None
                assert terminal["registry_attempt"]["evidence_role"] == "evaluation_consumed_on_authorization"
            else:
                assert terminal["registry_attempt"]["status"] == "failed"
                assert terminal["registry_attempt"]["callback_invocations"] == 1

        cohort, registry, batches, _ = completed("secondary-failures")
        original = RuntimeError("Preserve this evaluator error")

        def fail(*args):
            raise original

        with patch.object(registry, "_finalize", side_effect=OSError("Literal finalizer failure")), \
                patch.object(registry, "get_attempt", side_effect=OSError("Literal inspection failure")):
            assert raised(lambda: cohort.evaluate_registered(lambda: batches, fail), RuntimeError) is original
        terminal = failed("secondary-failures", cohort, registry, original, [1, 0, 0, 0, 0, 0])
        assert terminal["registry_attempt"] is None
        assert terminal["registry_attempt_inspection_exception_type"] == "OSError"
        unknown = registry.get_attempt("secondary-failures")
        assert unknown["status"] == "authorized_outcome_unknown" and unknown["callback_invocations"] is None

        cohort, registry, batches, _ = completed("terminal-write-failure")
        original = KeyboardInterrupt("Preserve cancellation after failed terminal write")
        writer = cohort._write

        def no_terminal(name, body):
            if name == "evaluation":
                raise OSError("Literal terminal persistence failure")
            return writer(name, body)

        with patch.object(cohort, "_write", side_effect=no_terminal):
            assert raised(lambda: cohort.evaluate_registered(lambda: batches, fail), KeyboardInterrupt) is original
        assert cohort._invalidated and registry.get_attempt("terminal-write-failure")["status"] == "failed"
        assert not (root / "terminal-write-failure" / "evaluation.json").exists()
        raised(lambda: cohort.evaluate_registered(lambda: batches, fail))

        cohort, registry, batches, _ = completed("slot-write-failure")
        original = OSError("Literal second-slot persistence failure")
        writer, writes, calls = cohort._write, [], []

        def no_second(name, body):
            writes.append(name)
            if name == "evaluation-000000-000001":
                raise original
            return writer(name, body)

        def count(context, endpoint, rows):
            calls.append(context["slot"]["slot_id"])
            return {"literal_result": 1}

        with patch.object(cohort, "_write", side_effect=no_second):
            assert raised(lambda: cohort.evaluate_registered(lambda: batches, count), OSError) is original
        terminal = failed("slot-write-failure", cohort, registry, original, [1, 1, 0, 0, 0, 0])
        assert calls == ["first", "second"] and writes.count("evaluation-000000-000001") == 1
        assert slots(terminal)[1]["outcome"] == "failed" and slots(terminal)[1]["receipt_sha256"] is None
        assert terminal["failure_phase"] == "slot_persistence"

        for changed in ("evaluation-start.json", "endpoint-seal.json", "evaluation-000000-000000.json"):
            name = "tamper-" + changed.removesuffix(".json")
            cohort, registry, batches, _ = completed(name)
            calls = []

            def tamper(context, endpoint, rows):
                calls.append(context["slot"]["slot_id"])
                if len(calls) == 2:
                    path = root / name / changed
                    document = read(path)
                    document["tampered"] = True
                    document["sha256"] = digest({key: value for key, value in document.items() if key != "sha256"})
                    path.write_bytes(canonical_bytes(document) + b"\n")
                return {"literal_result": 1}

            error = raised(lambda: cohort.evaluate_registered(lambda: batches, tamper))
            terminal = failed(name, cohort, registry, error, [1, 1, 0, 0, 0, 0])
            assert calls == ["first", "second"] and terminal["failure_phase"] == "slot_verification"
            assert slots(terminal)[1]["outcome"] == "failed"

        for index, output in enumerate(({"literal_result": float("nan")}, {"wrong_key": 1})):
            name = "invalid-output-" + str(index)
            cohort, registry, batches, _ = completed(name)
            error = raised(lambda: cohort.evaluate_registered(lambda: batches, lambda *args: output), ValueError)
            terminal = failed(name, cohort, registry, error, [1, 0, 0, 0, 0, 0])
            assert terminal["failure_phase"] == "slot_result_validation"
            assert terminal["registry_attempt"]["status"] == "failed"
