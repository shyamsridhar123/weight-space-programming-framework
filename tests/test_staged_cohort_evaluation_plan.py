"""One declaration-only regression; no preparation, scoring data, or study execution."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rlcd.weight_space.data_roles import DataRoleRegistry
from rlcd.weight_space.model import digest
from rlcd.weight_space.staged_cohort import StagedCohort


def test_frozen_evaluation_plan_and_legacy_compatibility():
    first, second = " first ", "second"
    execution = {"backend": "numpy", "dtype": "float64", "device": "cpu",
                 "accumulation": "matmul", "endpoint_cast": "reject",
                 "error_model": "empirical"}
    # These are valid declarations even when their execution is unsupported.
    alternate = execution | {"dtype": "bfloat16", "device": "cuda:23",
                             "accumulation": "right_to_left", "endpoint_cast": "round",
                             "error_model": "formal"}
    torch_execution = execution | {"backend": "torch", "dtype": "float32",
                                   "accumulation": "left_to_right"}

    def slot(identifier, source, spec=execution):
        return {"slot_id": identifier, "source": deepcopy(source),
                "execution": deepcopy(spec), "result_keys": ["loss", "correct"]}

    selected = {"kind": "selected", "fallback_control_id": "parent"}
    template = {
        "member_ids": [first, second],
        "protocol": {
            "fixture": {"thresholds": [1, 2]},
            "evaluation_plan": {
                "schema": "wsp-staged-cohort-evaluation-plan/1",
                "result_schema": "wsp-staged-cohort-result/1",
                "aggregate": {"source_id": " aggregate source ", "family_id": "shared-family"},
                "holdouts": [
                    {"holdout_id": " holdout-α ", "source_id": "child-source",
                     "family_id": "shared-family", "row_count": 2},
                    {"holdout_id": "unused-holdout", "source_id": "unused-source",
                     "family_id": "shared-family", "row_count": 1},
                ],
                "members": [
                    {"member_id": first, "holdout_id": " holdout-α ", "slots": [
                        slot("selected", selected),
                        slot("selected-other-execution", selected, alternate),
                        slot("parent", {"kind": "control", "control_id": "parent"}),
                        slot("parent-other-execution", {"kind": "control", "control_id": "parent"},
                             torch_execution),
                        slot("network", {"kind": "control", "control_id": "reference-network"}),
                        slot("tensors", {"kind": "control", "control_id": "reference-tensors"}),
                        slot("structured", {"kind": "control", "control_id": "structured"}),
                    ]},
                    {"member_id": second, "holdout_id": " holdout-α ", "slots": [
                        slot("selected", {"kind": "selected", "fallback_control_id": None}),
                        slot("parent", {"kind": "control", "control_id": "second-only-parent"}),
                    ]},
                ],
            },
        },
        "controls": {
            first: {
                "parent": {"kind": "parent"},
                # Schema-only payloads deliberately establish no materialized endpoint.
                "reference-network": {"kind": "reference", "artifact": {
                    "schema": "wsp-network/1", "fixture": {"unverified": [True]}}},
                "reference-tensors": {"kind": "reference", "artifact": {
                    "schema": "wsp-tensors/1"}},
                "structured": {"kind": "structured",
                               "proposal": {"schema": "wsp-structured-program/1"},
                               "lowering": {"schema": "wsp-lowering/1"}},
                "unused-control": {"kind": "parent"},
            },
            second: {"second-only-parent": {"kind": "parent"}},
        },
    }

    def read(path):
        document = json.loads(path.read_bytes())
        assert document["sha256"] == digest({key: value for key, value in document.items()
                                             if key != "sha256"})
        return document

    with TemporaryDirectory(prefix="wsp-evaluation-plan-") as directory:
        root = Path(directory)
        # No batches or scoring authorizations are registered.
        registry = DataRoleRegistry(root / "roles.sqlite")

        def construct(name, inputs):
            return StagedCohort(root / name, **inputs, evaluator_id="literal-evaluator",
                                registry=registry, attempt_id=name)

        caller = deepcopy(template)
        cohort = construct("valid", caller)
        protocol_path = root / "valid" / "protocol.json"
        frozen_bytes = protocol_path.read_bytes()
        frozen = read(protocol_path)
        rows = [{"member_index": index, "member_id": identifier}
                for index, identifier in enumerate(template["member_ids"])]
        assert frozen["members"] == rows and frozen["roster_sha256"] == digest(rows)
        assert frozen["protocol"] == template["protocol"]
        assert frozen["controls"] == template["controls"]
        assert frozen["controls_sha256"] == digest(template["controls"])
        assert frozen["evaluation_available"] is False
        assert sorted(path.name for path in (root / "valid").iterdir()) == ["protocol.json"]

        caller["member_ids"].append("caller-added")
        caller["protocol"]["fixture"]["thresholds"].append(99)
        caller_plan = caller["protocol"]["evaluation_plan"]
        caller_plan["holdouts"][0]["row_count"] = 999
        caller_plan["members"][0]["slots"][0]["source"]["fallback_control_id"] = None
        caller_plan["members"][0]["slots"][0]["execution"]["dtype"] = "caller-mutated"
        caller_plan["members"][0]["slots"][0]["result_keys"].append("caller-added")
        caller["controls"][first]["parent"]["kind"] = "caller-mutated"
        caller["controls"][first]["reference-network"]["artifact"]["fixture"]["unverified"].append(False)
        caller["controls"][first]["structured"]["lowering"]["schema"] = "caller-mutated"
        assert cohort._check() == frozen and protocol_path.read_bytes() == frozen_bytes
        returned = cohort._check()
        returned["protocol"]["evaluation_plan"]["members"].clear()
        returned["controls"].clear()
        assert cohort._check() == frozen

        # Existing finite-JSON normalization precedes plan validation.
        normalized = deepcopy(template)
        normalized["member_ids"] = tuple(normalized["member_ids"])
        normalized_plan = normalized["protocol"]["evaluation_plan"]
        normalized_plan["holdouts"] = tuple(normalized_plan["holdouts"])
        normalized_plan["members"] = tuple(normalized_plan["members"])
        for member in normalized_plan["members"]:
            member["slots"] = tuple(member["slots"])
            for item in member["slots"]:
                item["result_keys"] = tuple(item["result_keys"])
        assert construct("normalized", normalized)._check()["protocol"] == template["protocol"]

        # Empty per-member control maps are valid when no slot or fallback uses one.
        selected_only = deepcopy(template)
        selected_only["controls"] = {first: {}, second: {}}
        for member in selected_only["protocol"]["evaluation_plan"]["members"]:
            member["slots"] = [slot("selected", {"kind": "selected", "fallback_control_id": None})]
        assert construct("selected-only", selected_only)._check()["controls"] == {first: {}, second: {}}

        legacy = {"member_ids": [first, second], "protocol": {"thresholds": (1, 2)},
                  "controls": {"identity": "literal-control", "arbitrary": [None, True, 3, 0.5,
                                                                            {"nested": ["frozen"]}]}}
        legacy_cohort = construct("legacy", legacy)
        legacy_frozen = read(root / "legacy" / "protocol.json")
        assert legacy_frozen["protocol"] == {"thresholds": [1, 2]}
        assert legacy_frozen["controls"] == legacy["controls"]
        assert legacy_frozen["evaluation_available"] is False
        legacy["controls"]["arbitrary"][-1]["nested"].append("caller-mutated")
        assert legacy_cohort._check() == legacy_frozen

        plan = ("protocol", "evaluation_plan")
        member = plan + ("members", 0)
        selected_slot = member + ("slots", 0)
        control_source = member + ("slots", 2, "source")
        missing = object()

        def at(value, path):
            for key in path:
                value = value[key]
            return value

        def reject(label, path, replacement):
            inputs = deepcopy(template)
            container = at(inputs, path[:-1])
            if replacement is missing:
                del container[path[-1]]
            else:
                container[path[-1]] = deepcopy(replacement)
            destination = root / ("invalid-" + label)
            with patch.object(Path, "mkdir", side_effect=AssertionError("Created a run before validation")) as mkdir:
                try:
                    construct(destination.name, inputs)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Expected rejection: " + label)
                mkdir.assert_not_called()
            assert not destination.exists(), label

        # Each fixed envelope rejects both omitted fields and undeclared fields.
        exact_objects = [
            ("plan", plan), ("aggregate", plan + ("aggregate",)),
            ("holdout", plan + ("holdouts", 0)), ("member", member),
            ("slot", selected_slot), ("selected-source", selected_slot + ("source",)),
            ("control-source", control_source), ("execution", selected_slot + ("execution",)),
            ("parent-control", ("controls", first, "parent")),
            ("reference-control", ("controls", first, "reference-network")),
            ("structured-control", ("controls", first, "structured")),
        ]
        for label, path in exact_objects:
            for key in at(template, path):
                reject(label + "-missing-" + key, path + (key,), missing)
            reject(label + "-extra", path + ("unexpected",), True)
            reject(label + "-not-object", path, [])

        cases = [
            ("protocol-not-object", ("protocol",), []),
            ("controls-not-object", ("controls",), []),
            ("null-plan", plan, None),
            ("plan-schema", plan + ("schema",), "wsp-staged-cohort-evaluation-plan/2"),
            ("result-schema", plan + ("result_schema",), "wsp-staged-cohort-result/2"),
            ("duplicated-controls", plan + ("controls",), template["controls"]),
            ("empty-holdouts", plan + ("holdouts",), []),
            ("holdouts-not-list", plan + ("holdouts",), {}),
            ("duplicate-holdout", plan + ("holdouts", 1, "holdout_id"), " holdout-α "),
            ("duplicate-source", plan + ("holdouts", 1, "source_id"), "child-source"),
            ("aggregate-source-collision", plan + ("aggregate", "source_id"), "child-source"),
            ("empty-members", plan + ("members",), []),
            ("members-not-list", plan + ("members",), {}),
            ("missing-member", plan + ("members",), template["protocol"]["evaluation_plan"]["members"][:1]),
            ("reordered-members", plan + ("members",), template["protocol"]["evaluation_plan"]["members"][::-1]),
            ("duplicate-member", plan + ("members", 1, "member_id"), first),
            ("unknown-member", member + ("member_id",), "unknown"),
            ("unknown-holdout", member + ("holdout_id",), "unknown"),
            ("trimmed-holdout", member + ("holdout_id",), "holdout-α"),
            ("empty-slots", member + ("slots",), []),
            ("slots-not-list", member + ("slots",), {}),
            ("duplicate-slot", member + ("slots", 1, "slot_id"), "selected"),
            ("slot-alias", selected_slot + ("alias_of",), "parent"),
            ("source-kind", selected_slot + ("source", "kind"), "parent"),
            ("unknown-control", control_source + ("control_id",), "unknown"),
            ("foreign-control", control_source + ("control_id",), "second-only-parent"),
            ("foreign-fallback", selected_slot + ("source", "fallback_control_id"), "second-only-parent"),
            ("nonparent-fallback", selected_slot + ("source", "fallback_control_id"), "reference-network"),
            ("missing-control-member", ("controls", second), missing),
            ("foreign-control-member", ("controls", "unknown"), {}),
            ("member-controls-not-object", ("controls", first), []),
            ("control-kind", ("controls", first, "unused-control", "kind"), "selected"),
            ("empty-result-keys", selected_slot + ("result_keys",), []),
            ("result-keys-not-list", selected_slot + ("result_keys",), "loss"),
            ("duplicate-result-key", selected_slot + ("result_keys",), ["loss", "loss"]),
            ("nonfinite-plan", ("protocol", "fixture", "thresholds", 0), float("nan")),
            ("nonfinite-control", ("controls", first, "reference-network", "artifact", "bad"), float("inf")),
            ("custom-control-value", ("controls", first, "reference-network", "artifact", "bad"), object()),
            ("nonstring-json-key", ("controls", first, "reference-network", "artifact", "bad"), {1: "invalid"}),
        ]
        for label, path, replacement in cases:
            reject(label, path, replacement)

        for index, count in enumerate([True, False, 0, -1, 2.0, "2", None]):
            reject("row-count-" + str(index), plan + ("holdouts", 0, "row_count"), count)

        identities = [
            plan + ("aggregate", "source_id"), plan + ("aggregate", "family_id"),
            plan + ("holdouts", 0, "holdout_id"), plan + ("holdouts", 0, "source_id"),
            plan + ("holdouts", 0, "family_id"), member + ("member_id",),
            member + ("holdout_id",), selected_slot + ("slot_id",),
            selected_slot + ("source", "fallback_control_id"), control_source + ("control_id",),
            selected_slot + ("result_keys", 0),
        ]
        for index, path in enumerate(identities):
            for suffix, value in [("blank", " \t"), ("type", 1), ("utf8", "\ud800")]:
                reject("identity-" + str(index) + "-" + suffix, path, value)
        for suffix, key in [("blank", " "), ("utf8", "\ud800")]:
            reject("control-id-" + suffix, ("controls", first, key), {"kind": "parent"})

        for index, path in enumerate([
                ("controls", first, "reference-network", "artifact"),
                ("controls", first, "reference-tensors", "artifact"),
                ("controls", first, "structured", "proposal"),
                ("controls", first, "structured", "lowering")]):
            reject("payload-type-" + str(index), path, [])
            reject("payload-schema-missing-" + str(index), path + ("schema",), missing)
            reject("payload-schema-unknown-" + str(index), path + ("schema",), "unknown/1")
            reject("payload-schema-type-" + str(index), path + ("schema",), True)

        bad_execution = {"backend": "unknown", "dtype": "float128", "device": "cpu:0",
                         "accumulation": "unknown", "endpoint_cast": "unknown", "error_model": "unknown"}
        for key, value in bad_execution.items():
            reject("execution-invalid-" + key, selected_slot + ("execution", key), value)
            reject("execution-type-" + key, selected_slot + ("execution", key), True)
