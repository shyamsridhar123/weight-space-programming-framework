"""Business outcomes, retention boundaries, and evidence integrity for the example."""
import copy
import itertools
import numpy as np
import pytest

from examples.enterprise_support_routing import (
    ARCHITECTURE, HEAD, QUEUES, ROOT, admit, baseline, contract, featurize,
    measure, model_metadata, routes, run_scenario, scores, verify_bundle, write_bundle,
)
from rlcd.weight_space import (
    StructuralPolicy, TensorArtifact, apply_program, certify_feature_retention,
    difference, verify_feature_retention,
)
from rlcd.weight_space.program import Program


@pytest.fixture(scope="module")
def bundles():
    return {name: run_scenario(name) for name in ("policy-change", "conflict", "coverage-gap")}


def chosen(bundle):
    return next(row for row in bundle["candidates"] if row["label"] == "protected-edit")


def test_rollout_changes_business_queue_and_preserves_existing_scores(bundles):
    bundle = bundles["policy-change"]
    assert bundle["final_admission"]["decision"] == "admitted_for_synthetic_trial"
    result = bundle["session"]["result"]
    assert result["maximum_target_shift_error"] == pytest.approx(3 / 101)
    assert result["minimum_target_margin"] == pytest.approx(.25 + 300 / 101 - 2)
    assert result["maximum_protected_score_drift"] == 0
    assert result["target_routes_correct"] and result["protected_routes_unchanged"]
    for ticket in result["tickets"]:
        if ticket["role"] == "target":
            assert (ticket["before_queue"], ticket["after_queue"]) == ("general", "platform")
    row = chosen(bundle)
    assert len(row["program"]["edits"]) == 1
    assert row["program"]["edits"][0]["index"] == [3, 6]
    assert row["retention_certificate"]["exactly_zero_on_supplied_features"]
    assert not row["retention_certificate"]["floating_execution_verified"]


def test_unconstrained_and_soft_edits_do_not_get_admitted_just_for_target_gain(bundles):
    rows = {row["label"]: row for row in bundles["policy-change"]["candidates"]}
    assert rows["keep-parent"]["admission"]["metrics"]["maximum_target_shift_error"] == 3
    assert "target_routing" in rows["keep-parent"]["admission"]["failures"]
    direct = rows["target-only"]["admission"]
    assert direct["structural"]["structural_accepted"]
    assert direct["metrics"]["target_routes_correct"]
    assert "protected_routes" in direct["failures"]
    assert any(row["after_queue"] == "manual_review" for row in direct["metrics"]["tickets"]
               if row["role"] == "protected")
    soft = rows["soft-retention"]["admission"]
    assert soft["metrics"]["protected_routes_unchanged"]
    assert "protected_exact_retention" in soft["failures"]
    assert not direct["eligible"] and not soft["eligible"]


def test_conflicting_protection_abstains_without_final_evaluation(bundles):
    bundle = bundles["conflict"]
    row = chosen(bundle)
    assert row["compilation"]["diagnostics"]["nullspace_dimension"] == 0
    assert row["program"]["edits"] == []
    assert bundle["final_admission"]["decision"] == "abstained"
    assert bundle["session"]["status"] == "abstained"
    assert bundle["session"]["result"] is None and bundle["rollback"] is None
    assert bundle["session"]["heldout_accounting"]["evaluator_calls"] == 0


def test_unseen_symptom_blocks_the_sealed_candidate_without_reselection(bundles):
    bundle = bundles["coverage-gap"]
    assert chosen(bundle)["admission"]["eligible"]
    assert bundle["final_admission"]["selected_label"] == "protected-edit"
    assert bundle["final_admission"]["decision"] == "rejected_after_holdout"
    assert bundle["final_admission"]["heldout_failures"] == ["target_score_shift", "target_routing"]
    assert bundle["session"]["result"]["maximum_target_shift_error"] == 3
    assert bundle["session"]["trace"][-2]["action"] == "seal"
    assert bundle["session"]["trace"][-1]["action"] == "heldout_evaluation"


def test_protected_feature_plane_retention_extends_to_valid_unobserved_combinations(bundles):
    bundle = bundles["policy-change"]
    parent = baseline()
    candidate = apply_program(parent, Program.from_dict(chosen(bundle)["program"]))
    tickets = [{"topic": topic, "product": product, "symptom": symptom, "tier": tier, "urgency": urgency}
               for topic, product, symptom, tier, urgency in itertools.product(
                   QUEUES, ("core", "telemetry"), ("outage", "access", "latency"),
                   ("standard", "enterprise"), (0, .125, .5, .875, 1))]
    protected = [row for row, feature in zip(tickets, featurize(tickets)) if feature[-1] == 0]
    assert np.array_equal(scores(parent, protected), scores(candidate, protected))
    assert verify_feature_retention(parent, candidate, HEAD, bundle["anchors"],
                                    chosen(bundle)["retention_certificate"])
    undo = Program.from_dict(bundle["rollback"]["program"])
    assert apply_program(candidate, undo).sha256 == parent.sha256
    with pytest.raises(ValueError, match="Stale"):
        apply_program(parent.replace(**{"head.bias": [1, 0, 0, 0]}), Program.from_dict(chosen(bundle)["program"]))


def test_schema_and_routing_margin_are_enforced(bundles):
    ticket = copy.deepcopy(bundles["policy-change"]["inputs"]["public"][0])
    changed = TensorArtifact(ARCHITECTURE, baseline().tensors,
                             model_metadata() | {"feature_names": list(reversed(model_metadata()["feature_names"]))})
    with pytest.raises(ValueError, match="schema"):
        scores(changed, [ticket])
    for field, value in (("product", "unknown"), ("urgency", float("nan")), ("urgency", True)):
        with pytest.raises(ValueError):
            featurize([ticket | {field: value}])
    assert routes(np.array([[1, 1, 0, 0], [1.5, 1, 0, 0], [1.49, 1, 0, 0]]), .5)[0] == [
        "manual_review", "general", "manual_review"]


def test_nonwritable_bias_change_cannot_pass_domain_admission(bundles):
    bundle, parent, rule = bundles["policy-change"], baseline(), contract("policy-change")
    candidate = parent.replace(**{"head.bias": [0, 0, 0, 3]})
    program = difference(parent, candidate, "Unauthorized bias change")
    certificate = certify_feature_retention(parent, candidate, HEAD, bundle["anchors"])
    result = admit(parent, program, np.array(bundle["anchors"]), certificate,
                   measure(parent, candidate, bundle["inputs"]["public"], rule), rule,
                   StructuralPolicy((HEAD.weight,), 7, 4, 3))
    assert "structural_policy" in result["failures"]
    assert result["structural"]["behavioral_status"] == "not_checked"


@pytest.mark.parametrize("name", ["policy-change", "conflict", "coverage-gap"])
def test_saved_evidence_recomputes_without_refitting(bundles, tmp_path, name):
    write_bundle(bundles[name], tmp_path)
    assert verify_bundle(tmp_path)["decision"] == bundles[name]["final_admission"]["decision"]


@pytest.mark.parametrize("tamper", ["certificate", "lowering", "trace", "production"])
def test_rehashed_bundle_still_must_satisfy_the_local_specification(bundles, tmp_path, tamper):
    bundle = copy.deepcopy(bundles["policy-change"])
    if tamper == "certificate":
        chosen(bundle)["retention_certificate"]["floating_execution_verified"] = True
    elif tamper == "lowering":
        chosen(bundle)["compilation"]["proposal"]["commands"][0]["values"][3][6] += 1
    elif tamper == "trace":
        bundle["session"]["trace"][-2]["request"]["candidate_sha256"] = "0" * 64
    else:
        bundle["final_admission"]["production_authorized"] = True
    write_bundle(bundle, tmp_path)
    with pytest.raises(ValueError):
        verify_bundle(tmp_path)


def test_changed_file_bytes_are_rejected(bundles, tmp_path):
    write_bundle(bundles["policy-change"], tmp_path)
    (tmp_path / "edit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)


def test_published_synthetic_bundles_are_consistent():
    published = ROOT / "artifacts/enterprise/support-routing"
    for name in ("policy-change", "conflict", "coverage-gap"):
        assert verify_bundle(published / name)["verified"]
