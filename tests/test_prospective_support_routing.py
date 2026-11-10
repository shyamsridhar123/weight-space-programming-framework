"""Synthetic integration checks; no enterprise outcome or freshness evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
import pytest

from rlcd.weight_space import research_tasks
from rlcd.weight_space.data_roles import DataRoleRegistry
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.program import difference
from rlcd.weight_space.research_tasks import TextPreprocessing, make_head
from scripts import run_prospective_support_routing as runner


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _envelope(name, texts):
    return {"source_id": "synthetic/" + name, "family_id": "synthetic-family/" + name,
            "parents": [], "rows": [
                {"request_id": f"synthetic-{name}-{index}", "text": text, "label": label}
                for index, (text, label) in enumerate(texts)]}


def _cohort(envelope, experiment_id):
    return {**{key: envelope[key] for key in ("source_id", "family_id", "parents")},
            "row_count": len(envelope["rows"]), "rows_sha256": digest(envelope["rows"]),
            "normalized_inputs_sha256": digest([
                {"normalized_request": research_tasks.normalized_text(row["text"])}
                for row in envelope["rows"]]),
            "declaration": {"collection_id": "synthetic-fixture",
                "labeling_method": "hand-authored test labels", "reserved_for": experiment_id,
                "no_development_use": True, "synthetic": True}}


@pytest.fixture
def support_case(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    preprocessing = TextPreprocessing.fit(["invoice overdue", "password reset"])
    width = len(preprocessing.vocabulary)
    encoder = {"weight": np.eye(width, dtype=np.float32), "bias": np.zeros(width, dtype=np.float32)}
    weight = np.zeros((2, width), dtype=np.float32)
    for index, term in enumerate(preprocessing.vocabulary):
        weight[0 if "invoice" in term or "overdue" in term else 1, index] = 1.
    head = make_head({"encoder": encoder, "head": {"weight": weight, "bias": np.zeros(2, dtype=np.float32)},
                     "ancestry": "Hand-defined synthetic test fixture; no training or enterprise data"},
                    digest(preprocessing.to_dict()), ["billing", "access"], 0, "synthetic-support")
    _write_json(parent / "head.json", head.to_dict())
    _write_json(parent / "preprocessing.json", preprocessing.to_dict())
    np.savez(parent / "encoder.npz", **encoder)
    lineage = {"schema": "wsp-trained-lineage/1", "head_sha256": head.sha256,
               **{key: head.metadata[key] for key in ("encoder_sha256", "preprocessing_sha256", "feature_id")},
               "files": {name: {"sha256": _sha(parent / name), "bytes": (parent / name).stat().st_size}
                         for name in runner.PARENT_FILES[:-1]}}
    _write_json(parent / "lineage.json", lineage)
    for name in ("features.npz", "evaluation-features.npz"):
        (parent / name).write_bytes(b"SYNTHETIC POISON: historical features must never be opened")

    development = {
        "target": _envelope("target", [("invoice overdue target", 0)]),
        "protected": _envelope("protected", [("password reset anchor", 1)]),
        "calibration": _envelope("calibration", [
            ("invoice overdue calibration", 0), ("password reset calibration", 1)])}
    holdout = _envelope("prospective", [("invoice overdue prospective", 0), ("password reset prospective", 1)])
    config = {"experiment_id": "synthetic-prospective", "label_names": ["billing", "access"],
              "queue_names": ["billing", "access", "specialist"], "queue_map": [0, 1, 2],
              "assignments": [{"assignment_id": "repair-billing", "target_label": 0,
                               "desired_code": 2, "mode": "none"}],
              "cohort": _cohort(holdout, "synthetic-prospective"),
              "admission": {"minimum_target_gain": -1., "maximum_protected_harm": 1.}}
    case = {"root": tmp_path, "parent": parent, "development": development, "holdout": holdout,
            "config": config, "development_path": tmp_path / "development.json",
            "holdout_path": tmp_path / "prospective.json", "registry": tmp_path / "roles.sqlite"}
    _write_json(case["development_path"], development)
    _write_json(case["holdout_path"], holdout)

    real_open, real_load = Path.open, np.load

    def guarded_open(path, *args, **kwargs):
        assert path.name not in ("features.npz", "evaluation-features.npz"), "historical features opened"
        return real_open(path, *args, **kwargs)

    def guarded_load(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            assert Path(file).name not in ("features.npz", "evaluation-features.npz"), "historical features loaded"
        return real_load(file, *args, **kwargs)

    def forbidden_lineage(*args, **kwargs):
        raise AssertionError("load_lineage is forbidden for prospective support routing")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(np, "load", guarded_load)
    monkeypatch.setattr(research_tasks, "load_lineage", forbidden_lineage)
    return case


def _prepare(case, name="run", *, config=None, experiment_id=None):
    configuration = copy.deepcopy(case["config"] if config is None else config)
    if experiment_id is not None:
        configuration["experiment_id"] = experiment_id
        configuration["cohort"]["declaration"]["reserved_for"] = experiment_id
    path = case["root"] / (name + "-configuration.json")
    _write_json(path, configuration)
    run = case["root"] / name
    return run, runner.prepare(case["parent"], case["development_path"], path, run, case["registry"])


def _evaluate(case, run):
    return runner.evaluate(run, case["holdout_path"], case["registry"])


def _query(path, sql, parameters=()):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return connection.execute(sql, parameters).fetchall()
    finally:
        connection.close()


def _attempt(path, attempt_id):
    row = _query(path, "SELECT attempt_json FROM attempts WHERE attempt_id = ?", (attempt_id,))
    return None if not row else json.loads(row[0][0])


def _forbid_prediction_replay(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("verification must not execute prospective predictions or metrics")
    monkeypatch.setattr(TextPreprocessing, "transform", forbidden)
    for name in ("encode_inputs", "evaluate_head", "behavior_metrics", "_score_rows"):
        monkeypatch.setattr(runner, name, forbidden)


def _sample_threshold_case():
    statuses = ["admitted", "admitted", "admitted", "abstained", "rejected", "failed"]
    protocol = {"configuration": {"admission": {
        "minimum_target_gain": .25, "maximum_protected_harm": 0.}},
        "assignments": [{"assignment_id": f"assigned-{index}", "proposal_status": status,
                         "reason": "synthetic-" + status} for index, status in enumerate(statuses)]}
    outcomes = runner._outcomes(protocol, "succeeded")
    for outcome, (gain, harm, passed) in zip(outcomes, [(.25, 0., True), (.5, .25, False),
                                                      (None, -.25, False)]):
        outcome.update(metrics={"target_gain": gain, "protected_harm": harm},
                       meets_frozen_sample_thresholds=passed)
    return protocol, outcomes


def _assert_terminal_accounting(run, protocol, report):
    accounting = report["sample_threshold_accounting"]
    reduced = runner._sample_threshold_accounting(protocol, report["assignments"], report["status"])
    assert canonical_bytes(accounting) == canonical_bytes(reduced)
    manifest = runner.read_json(run / "manifest.json")
    assert manifest["schema"] == "wsp-prospective-support-manifest/2"
    assert report["schema"] == "wsp-prospective-support-evaluation/2"
    assert canonical_bytes(manifest["sample_threshold_accounting"]) == canonical_bytes(accounting)
    if report["result"] is not None:
        assert report["result"]["schema"] == "wsp-prospective-support-result/2"
        assert report["result"]["evaluator_id"] == "prospective-support-native-readout/2"
        assert canonical_bytes(report["result"]["sample_threshold_accounting"]) == canonical_bytes(accounting)
    return accounting


def _assert_unobserved_accounting(accounting, assigned, admitted):
    assert accounting == {
        "schema": "wsp-prospective-support-sample-threshold-accounting/1",
        "assigned_count": assigned, "admitted_count": admitted, "scored_count": 0,
        "target_gain_observed_count": 0, "protected_harm_observed_count": 0,
        "complete_endpoint_count": 0, "missing_endpoint_count": assigned,
        "admitted_missing_endpoint_count": admitted, "sample_threshold_success_count": 0,
        "artifact_coverage": admitted / assigned, "all_assigned_sample_threshold_success_rate": 0.,
        "conditional_sample_threshold_success_rate": 0. if admitted else None,
        "conditional_target_gain_mean": None, "conditional_protected_harm_mean": None}


@pytest.mark.parametrize(("last_endpoints", "target_count", "harm_count", "target_mean", "harm_mean"), [
    ((None, -.25), 2, 3, None, 0.), ((-.25, None), 3, 2, 1 / 6, None)])
def test_sample_threshold_accounting_preserves_assignment_and_endpoint_denominators(
        last_endpoints, target_count, harm_count, target_mean, harm_mean):
    protocol, outcomes = _sample_threshold_case()
    outcomes[2]["metrics"] = dict(zip(("target_gain", "protected_harm"), last_endpoints))
    assert runner._sample_threshold_accounting(protocol, outcomes, "succeeded") == {
        "schema": "wsp-prospective-support-sample-threshold-accounting/1",
        "assigned_count": 6, "admitted_count": 3, "scored_count": 3,
        "target_gain_observed_count": target_count, "protected_harm_observed_count": harm_count,
        "complete_endpoint_count": 2, "missing_endpoint_count": 4,
        "admitted_missing_endpoint_count": 1, "sample_threshold_success_count": 1,
        "artifact_coverage": .5, "all_assigned_sample_threshold_success_rate": 1 / 6,
        "conditional_sample_threshold_success_rate": 1 / 3,
        "conditional_target_gain_mean": target_mean, "conditional_protected_harm_mean": harm_mean}


def test_sample_threshold_accounting_preserves_small_terms_in_conditional_means():
    protocol, outcomes = _sample_threshold_case()
    protocol["configuration"]["admission"]["minimum_target_gain"] = -1.
    for outcome, gain in zip(outcomes, [1., 1e-16, -1.]):
        outcome.update(metrics={"target_gain": gain, "protected_harm": 0.},
                       meets_frozen_sample_thresholds=True)
    accounting = runner._sample_threshold_accounting(protocol, outcomes, "succeeded")
    assert accounting["conditional_target_gain_mean"] == 1e-16 / 3
    assert accounting["conditional_protected_harm_mean"] == 0.


@pytest.mark.parametrize(("minimum", "maximum", "gain", "harm", "passed"), [
    (.25, 0., .25, 0., True), (0., 0., 0, 0, True),
    (-.25, -.25, -.25, -.25, True), (-.25, -.25, -.5, -.25, False),
    (-.25, -.25, -.25, 0., False), (-1., 1., -1., 1., True),
    (.25, 0., .125, 0., False), (.25, 0., .25, .125, False),
    (0., 0., None, 0., False), (0., 0., 0., None, False), (0., 0., None, None, False)])
def test_sample_threshold_accounting_keeps_inclusive_zero_and_negative_thresholds(
        minimum, maximum, gain, harm, passed):
    protocol, outcomes = _sample_threshold_case()
    protocol["assignments"] = protocol["assignments"][:1]
    protocol["configuration"]["admission"].update(minimum_target_gain=minimum, maximum_protected_harm=maximum)
    outcome = outcomes[0]
    outcome.update(metrics={"target_gain": gain, "protected_harm": harm},
                   meets_frozen_sample_thresholds=passed)
    accounting = runner._sample_threshold_accounting(protocol, [outcome], "succeeded")
    assert accounting["sample_threshold_success_count"] == int(passed)
    assert accounting["all_assigned_sample_threshold_success_rate"] == float(passed)
    assert accounting["conditional_sample_threshold_success_rate"] == float(passed)
    assert accounting["complete_endpoint_count"] == int(gain is not None and harm is not None)
    assert accounting["conditional_target_gain_mean"] == gain
    assert accounting["conditional_protected_harm_mean"] == harm


@pytest.mark.parametrize("change", ["missing", "extra", "reordered", "duplicate_frozen", "duplicate_recorded"])
def test_sample_threshold_accounting_requires_exact_frozen_assignment_sequence(change):
    protocol, outcomes = _sample_threshold_case()
    if change == "missing":
        outcomes.pop()
    elif change == "extra":
        outcomes.append(copy.deepcopy(outcomes[-1]))
    elif change == "reordered":
        outcomes[0], outcomes[1] = outcomes[1], outcomes[0]
    elif change == "duplicate_frozen":
        protocol["assignments"][1]["assignment_id"] = protocol["assignments"][0]["assignment_id"]
    else:
        outcomes[1]["assignment_id"] = outcomes[0]["assignment_id"]
    with pytest.raises(ValueError):
        runner._sample_threshold_accounting(protocol, outcomes, "succeeded")


@pytest.mark.parametrize(("path", "replacement"), [
    (("protocol", "assignments", 0, "proposal_status"), "pending"),
    (("outcomes", 0, "assignment_id"), "unfrozen-assignment"),
    (("outcomes", 0, "proposal_status"), "abstained"),
    (("outcomes", 0, "reason"), "different-frozen-reason"),
    (("outcomes", 0, "evaluation_status"), "not_scored"),
    (("outcomes", 3, "evaluation_status"), "succeeded"),
    (("outcomes", 3, "metrics"), {}),
    (("outcomes", 3, "meets_frozen_sample_thresholds"), True),
    (("outcomes", 0, "meets_frozen_sample_thresholds"), 1),
    (("outcomes", 0, "meets_frozen_sample_thresholds"), False),
    (("outcomes", 1, "meets_frozen_sample_thresholds"), True),
    (("outcomes", 2, "meets_frozen_sample_thresholds"), True)])
def test_sample_threshold_accounting_rejects_changed_or_unscored_assignment_fields(path, replacement):
    protocol, outcomes = _sample_threshold_case()
    target = {"protocol": protocol, "outcomes": outcomes}
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError):
        runner._sample_threshold_accounting(protocol, outcomes, "succeeded")


@pytest.mark.parametrize("field", ["assignment_id", "proposal_status", "evaluation_status", "reason",
                                  "metrics", "meets_frozen_sample_thresholds"])
def test_sample_threshold_accounting_requires_successful_outcome_fields(field):
    protocol, outcomes = _sample_threshold_case()
    del outcomes[0][field]
    with pytest.raises(ValueError):
        runner._sample_threshold_accounting(protocol, outcomes, "succeeded")


@pytest.mark.parametrize("endpoint", ["target_gain", "protected_harm"])
@pytest.mark.parametrize(("value", "target_passes", "harm_passes"), [
    (True, True, False), (False, False, True), (float("nan"), False, False),
    (float("inf"), True, False), (-float("inf"), False, True),
    (1.01, True, False), (-1.01, False, True), ("0", False, False)])
def test_sample_threshold_accounting_rejects_invalid_endpoints(endpoint, value, target_passes, harm_passes):
    protocol, outcomes = _sample_threshold_case()
    outcomes[0]["metrics"][endpoint] = value
    # Keep the flag consistent with comparisons wherever defined, so numeric
    # validation cannot pass solely by detecting a contradictory threshold flag.
    outcomes[0]["meets_frozen_sample_thresholds"] = target_passes if endpoint == "target_gain" else harm_passes
    with pytest.raises(ValueError):
        runner._sample_threshold_accounting(protocol, outcomes, "succeeded")


@pytest.mark.parametrize("endpoint", ["target_gain", "protected_harm"])
def test_sample_threshold_accounting_requires_both_endpoint_keys(endpoint):
    protocol, outcomes = _sample_threshold_case()
    del outcomes[0]["metrics"][endpoint]
    outcomes[0]["meets_frozen_sample_thresholds"] = False
    with pytest.raises(ValueError):
        runner._sample_threshold_accounting(protocol, outcomes, "succeeded")


@pytest.mark.parametrize("status", ["failed", "rejected", "abstained"])
def test_sample_threshold_accounting_retains_admissions_without_retained_scores(status):
    protocol, _ = _sample_threshold_case()
    if status == "abstained":
        protocol["assignments"] = protocol["assignments"][3:]
    accounting = runner._sample_threshold_accounting(protocol, runner._outcomes(protocol, status), status)
    _assert_unobserved_accounting(accounting, 3 if status == "abstained" else 6,
                                 0 if status == "abstained" else 3)
    if status == "abstained":
        with pytest.raises(ValueError):
            runner._sample_threshold_accounting(protocol, runner._outcomes(protocol, "succeeded"), "succeeded")


@pytest.mark.parametrize(("status", "admitted_status"), [
    ("succeeded", "failed"), ("failed", "rejected"), ("rejected", "failed"),
    ("abstained", "not_scored"), ("outcome_unknown", "outcome_unknown")])
def test_sample_threshold_accounting_rejects_terminal_state_contradictions(status, admitted_status):
    protocol, _ = _sample_threshold_case()
    with pytest.raises(ValueError):
        runner._sample_threshold_accounting(protocol, runner._outcomes(protocol, admitted_status), status)


@pytest.mark.parametrize("status", ["failed", "rejected"])
def test_sample_threshold_accounting_rejects_metrics_retained_after_failure(status):
    protocol, _ = _sample_threshold_case()
    outcomes = runner._outcomes(protocol, status)
    outcomes[0].update(metrics={"target_gain": None, "protected_harm": None},
                       meets_frozen_sample_thresholds=False)
    with pytest.raises(ValueError):
        runner._sample_threshold_accounting(protocol, outcomes, status)


def test_real_pipeline_commits_aliases_and_attempt_before_native_callback(support_case, monkeypatch):
    case = support_case
    config = copy.deepcopy(case["config"])
    config["assignments"].append({"assignment_id": "repair-soft", "target_label": 0,
                                  "desired_code": 2, "mode": "soft"})
    real_transform = TextPreprocessing.transform
    development_calls = []

    def development_transform(preprocessing, texts):
        consumed = {row[0] for row in _query(case["registry"], "SELECT key FROM consumed WHERE kind = 'source'")}
        for envelope in case["development"].values():
            source = envelope["source_id"]
            assert {source, source + "/normalized-inputs", source + "/request-identities"} <= consumed
        development_calls.append(texts)
        return real_transform(preprocessing, texts)

    with monkeypatch.context() as patch:
        patch.setattr(TextPreprocessing, "transform", development_transform)
        run, protocol = _prepare(case, config=config)
    assert len(development_calls) == 3
    assert protocol["schema"] == "wsp-prospective-support-protocol/1"
    assert protocol["evaluator"]["id"] == "prospective-support-native-readout/2"
    assert "sample_threshold_accounting" not in runner.read_json(run / "manifest.json")
    assert all(record["proposal_status"] == "admitted" for record in protocol["assignments"])
    assert protocol["configuration"]["assignments"][0]["configuration"] == runner.FIT
    assert all(_sha(run / name) == expected for name, expected in protocol["files"].items())
    assert all((run / "parent" / name).read_bytes() == (case["parent"] / name).read_bytes()
               for name in runner.PARENT_FILES)
    for source, payload in _query(case["registry"], "SELECT source_id, payload_json FROM sources"):
        if source.endswith("/normalized-inputs"):
            assert all(set(row) == {"normalized_request"} for row in json.loads(payload))
        if source.endswith("/request-identities"):
            assert all(set(row) == {"request_id"} for row in json.loads(payload))

    active, captured = [], []
    counts = {"score": 0, "transform": 0, "encode_inputs": 0, "evaluate_head": 0, "behavior_metrics": 0}
    real_score = DataRoleRegistry.score

    def observed_score(registry, batch, scorer, **kwargs):
        counts["score"] += 1

        def observed_callback(rows):
            pending = _attempt(case["registry"], protocol["attempt_id"])
            consumed = set(_query(case["registry"], "SELECT kind, key FROM consumed"))
            assert set(map(tuple, pending["consumed_scope"])) <= consumed
            manifest = runner.read_json(run / "manifest.json")
            assert manifest["phase"] == "evaluation_requested"
            assert "sample_threshold_accounting" not in manifest
            captured.append(pending)
            active.append(True)
            try:
                return scorer(rows)
            finally:
                active.pop()
        return real_score(registry, batch, observed_callback, **kwargs)

    def observed_transform(preprocessing, texts):
        assert active
        counts["transform"] += 1
        return real_transform(preprocessing, texts)

    monkeypatch.setattr(DataRoleRegistry, "score", observed_score)
    monkeypatch.setattr(TextPreprocessing, "transform", observed_transform)
    for name in ("encode_inputs", "evaluate_head", "behavior_metrics"):
        real = getattr(runner, name)

        def observed(*args, _name=name, _real=real, **kwargs):
            assert active
            counts[_name] += 1
            if _name == "evaluate_head":
                assert kwargs["native"] is True
            return _real(*args, **kwargs)
        monkeypatch.setattr(runner, name, observed)

    report = _evaluate(case, run)
    assert report["status"] == "succeeded", report
    assert counts == {"score": 1, "transform": 1, "encode_inputs": 1, "evaluate_head": 3, "behavior_metrics": 2}
    pending = captured[0]
    assert pending["status"] == "authorized_outcome_unknown"
    assert pending["callback_invocations"] is None and pending["result"] is None
    assert pending["binding"]["protocol_sha256"] == protocol["protocol_sha256"]
    assert pending["binding"]["frozen_files_sha256"] == digest(protocol["files"])
    assert {("source", case["holdout"]["source_id"] + suffix)
            for suffix in ("", "/normalized-inputs", "/request-identities")} <= set(map(tuple, pending["consumed_scope"]))
    terminal = DataRoleRegistry(case["registry"]).get_attempt(protocol["attempt_id"])
    assert terminal["status"] == "succeeded" and terminal["callback_invocations"] == 1
    assert terminal["reservation_sha256"] == pending["reservation_sha256"]
    assert terminal["lifecycle_sha256"] != pending["lifecycle_sha256"]
    assert terminal == report["evidence"]["attempt"]
    assert terminal["result"] == report["result"]
    assert terminal["binding"]["schema"] == "wsp-prospective-support-binding/1"
    assert report["result"]["parent_codes"] == [0, 1]
    assert all(row["metrics"]["target_rows"] == row["metrics"]["protected_rows"] == 1
               for row in report["assignments"])
    accounting = _assert_terminal_accounting(run, protocol, report)
    assert all(accounting[key] == 2 for key in (
        "assigned_count", "admitted_count", "scored_count", "target_gain_observed_count",
        "protected_harm_observed_count", "complete_endpoint_count", "sample_threshold_success_count"))
    assert accounting["missing_endpoint_count"] == accounting["admitted_missing_endpoint_count"] == 0
    assert accounting["artifact_coverage"] == accounting["all_assigned_sample_threshold_success_rate"] == 1.
    assert accounting["conditional_sample_threshold_success_rate"] == 1.
    documents = [protocol, report, report["result"], report["evidence"], terminal, *report["inputs"].values()]
    assert all(document[key] is False for document in documents for key in runner.FLAGS)
    assert runner.verify_result(run, case["registry"])
    assert counts == {"score": 1, "transform": 1, "encode_inputs": 1, "evaluate_head": 3, "behavior_metrics": 2}


@pytest.mark.parametrize("overlap", ["normalized_text", "request_identity", "family", "ancestry"])
def test_observed_development_aliases_reject_before_forward(support_case, monkeypatch, overlap):
    case = support_case
    target = case["development"]["target"]
    holdout = copy.deepcopy(case["holdout"])
    if overlap == "normalized_text":
        holdout["rows"][0].update(text="INVOICE, overdue target!!!", label=1,
                                    request_id="changed-id-and-label")
    elif overlap == "request_identity":
        holdout["rows"][0].update(request_id=target["rows"][0]["request_id"],
                                    text="entirely different prospective wording", label=1)
    elif overlap == "family":
        holdout["family_id"] = target["family_id"]
    else:
        holdout["parents"] = [target["source_id"]]
    _write_json(case["holdout_path"], holdout)
    config = copy.deepcopy(case["config"])
    config["cohort"] = _cohort(holdout, config["experiment_id"])
    run, protocol = _prepare(case, config=config)
    _forbid_prediction_replay(monkeypatch)
    report = _evaluate(case, run)
    assert report["status"] == "rejected"
    assert report["evidence"]["reason"] == "observed_development_consumption"
    assert report["evidence"]["scoring_started"] is False
    assert report["evidence"]["consumed_matches"]
    assert DataRoleRegistry(case["registry"]).get_attempt(protocol["attempt_id"]) is None
    assert report["assignments"][0]["evaluation_status"] == "rejected"
    _assert_unobserved_accounting(_assert_terminal_accounting(run, protocol, report), 1, 1)


def test_success_cannot_be_retried_or_reused_under_new_experiment(support_case, monkeypatch):
    case = support_case
    run, protocol = _prepare(case)
    first = _evaluate(case, run)
    assert first["status"] == "succeeded"
    report_bytes = (run / "evaluation.json").read_bytes()
    with pytest.raises(FileExistsError):
        _evaluate(case, run)
    assert (run / "evaluation.json").read_bytes() == report_bytes
    other, other_protocol = _prepare(case, "another-run", experiment_id="synthetic-other-experiment")
    _forbid_prediction_replay(monkeypatch)
    second = _evaluate(case, other)
    assert second["status"] == "rejected"
    assert second["evidence"]["reason"] == "observed_development_consumption"
    assert DataRoleRegistry(case["registry"]).get_attempt(other_protocol["attempt_id"]) is None
    assert DataRoleRegistry(case["registry"]).get_attempt(protocol["attempt_id"])["result"] == first["result"]
    assert runner.verify_result(run, case["registry"])


def test_every_assignment_survives_abstention_rejection_and_compiler_failure(support_case, monkeypatch):
    case = support_case
    config = copy.deepcopy(case["config"])
    config["admission"]["minimum_target_gain"] = .1
    specs = [("admit", "none", 2), ("decline", "abstain", 2), ("invalid", "unknown", 2),
             ("compiler-failure", "soft", 2), ("calibration-abstention", "none", 0),
             ("protected-edit", "nullspace", 2)]
    config["assignments"] = [{"assignment_id": name, "mode": mode, "desired_code": code, "target_label": 0}
                              for name, mode, code in specs]
    real_compile = runner.compile_arm

    def injected_compile(base, target, protected, desired_code, mode, **kwargs):
        if mode == "soft":
            raise RuntimeError("SYNTHETIC-COMPILER-PRIVATE-TEXT")
        return real_compile(base, target, protected, desired_code, mode, **kwargs)

    monkeypatch.setattr(runner, "compile_arm", injected_compile)
    run, protocol = _prepare(case, config=config)
    statuses = ["admitted", "abstained", "rejected", "failed", "abstained", "admitted"]
    assert [row["proposal_status"] for row in protocol["assignments"]] == statuses
    assert protocol["assignments"][4]["calibration"]["target_gain"] == 0.
    assert "artifact_directory" in protocol["assignments"][4]
    assert "SYNTHETIC-COMPILER-PRIVATE-TEXT" not in json.dumps(protocol)
    report = _evaluate(case, run)
    assert report["status"] == "succeeded"
    assert [row["assignment_id"] for row in report["assignments"]] == [row[0] for row in specs]
    assert [row["proposal_status"] for row in report["assignments"]] == statuses
    assert [row["evaluation_status"] for row in report["assignments"]] == [
        "succeeded", "not_scored", "not_scored", "not_scored", "not_scored", "succeeded"]
    for row in report["assignments"]:
        assert ("metrics" in row) is (row["proposal_status"] == "admitted")
    assert runner.read_json(run / "manifest.json")["assignments"] == report["assignments"]
    accounting = _assert_terminal_accounting(run, protocol, report)
    assert accounting["assigned_count"] == 6 and accounting["admitted_count"] == 2
    assert all(accounting[key] == 2 for key in (
        "scored_count", "target_gain_observed_count", "protected_harm_observed_count", "complete_endpoint_count"))
    assert accounting["missing_endpoint_count"] == 4 and accounting["admitted_missing_endpoint_count"] == 0
    assert accounting["artifact_coverage"] == 2 / 6
    assert accounting["all_assigned_sample_threshold_success_rate"] == accounting["sample_threshold_success_count"] / 6
    assert accounting["conditional_sample_threshold_success_rate"] == accounting["sample_threshold_success_count"] / 2
    assert runner.verify_result(run, case["registry"])


def test_all_abstentions_never_open_holdout_or_authorize_attempt(support_case, monkeypatch):
    case = support_case
    config = copy.deepcopy(case["config"])
    config["assignments"][0]["mode"] = "abstain"
    run, protocol = _prepare(case, config=config)
    _forbid_prediction_replay(monkeypatch)
    report = runner.evaluate(run, case["root"] / "nonexistent-holdout.json", case["registry"])
    assert report["status"] == "abstained"
    assert report["result"] is report["evidence"] is report["inputs"] is None
    assert report["assignments"][0]["evaluation_status"] == "not_scored"
    assert DataRoleRegistry(case["registry"]).get_attempt(protocol["attempt_id"]) is None
    _assert_unobserved_accounting(_assert_terminal_accounting(run, protocol, report), 1, 0)
    with pytest.raises(ValueError):
        runner.verify_result(run, case["registry"])


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_scorer_failure_or_cancellation_is_durable_and_sanitized(support_case, monkeypatch, failure_type):
    case = support_case
    run, protocol = _prepare(case)
    interruption = failure_type("SYNTHETIC-SCORER-PRIVATE-TEXT")

    def fail_transform(*args, **kwargs):
        raise interruption

    with monkeypatch.context() as patch:
        patch.setattr(TextPreprocessing, "transform", fail_transform)
        if failure_type is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt) as caught:
                _evaluate(case, run)
            assert caught.value is interruption
            report = runner.read_json(run / "evaluation.json")
        else:
            report = _evaluate(case, run)
    assert report["status"] == "failed"
    assert report["result"] is None
    assert report["evidence"]["scoring_started"] is True
    assert report["evidence"]["reason"] == "scorer_failed"
    assert all(row["evaluation_status"] == "failed" for row in report["assignments"])
    terminal = DataRoleRegistry(case["registry"]).get_attempt(protocol["attempt_id"])
    assert terminal["status"] == "failed" and terminal["callback_invocations"] == 1
    assert terminal["result"] is terminal["result_sha256"] is None
    assert "SYNTHETIC-SCORER-PRIVATE-TEXT" not in json.dumps([report, terminal])
    assert runner.read_json(run / "manifest.json")["assignments"] == report["assignments"]
    _assert_unobserved_accounting(_assert_terminal_accounting(run, protocol, report), 1, 1)
    with pytest.raises(ValueError):
        runner.verify_result(run, case["registry"])
    retry, retry_protocol = _prepare(case, "retry", experiment_id="synthetic-after-failure")
    _forbid_prediction_replay(monkeypatch)
    rejected = _evaluate(case, retry)
    assert rejected["status"] == "rejected" and rejected["evidence"]["scoring_started"] is False
    assert DataRoleRegistry(case["registry"]).get_attempt(retry_protocol["attempt_id"]) is None


def test_late_scorer_failure_discards_partial_endpoint_accounting(support_case, monkeypatch):
    case = support_case
    config = copy.deepcopy(case["config"])
    config["assignments"].append({"assignment_id": "repair-soft", "target_label": 0,
                                  "desired_code": 2, "mode": "soft"})
    run, protocol = _prepare(case, config=config)
    real_metrics, completed_metrics = runner.behavior_metrics, []

    def fail_after_one_metric(*args, **kwargs):
        if completed_metrics:
            raise RuntimeError("SYNTHETIC-LATE-SCORER-PRIVATE-TEXT")
        metrics = real_metrics(*args, **kwargs)
        completed_metrics.append(metrics)
        return metrics

    monkeypatch.setattr(runner, "behavior_metrics", fail_after_one_metric)
    report = _evaluate(case, run)
    assert len(completed_metrics) == 1 and report["status"] == "failed"
    assert report["result"] is None
    assert all(row["evaluation_status"] == "failed" and "metrics" not in row
               and "meets_frozen_sample_thresholds" not in row for row in report["assignments"])
    terminal = DataRoleRegistry(case["registry"]).get_attempt(protocol["attempt_id"])
    assert terminal["status"] == "failed" and terminal["callback_invocations"] == 1
    assert terminal["result"] is terminal["result_sha256"] is None
    assert "SYNTHETIC-LATE-SCORER-PRIVATE-TEXT" not in json.dumps([report, terminal])
    _assert_unobserved_accounting(_assert_terminal_accounting(run, protocol, report), 2, 2)


def test_verification_failure_reduces_the_final_failed_report_again(support_case, monkeypatch):
    case = support_case
    run, protocol = _prepare(case)

    def reject_report(*args, **kwargs):
        raise ValueError("synthetic_report_verification_failure")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_verify_report", reject_report)
        report = _evaluate(case, run)
    terminal = DataRoleRegistry(case["registry"]).get_attempt(protocol["attempt_id"])
    assert terminal["status"] == "succeeded" and terminal["callback_invocations"] == 1
    assert terminal["result"]["sample_threshold_accounting"]["scored_count"] == 1
    assert report["evidence"]["attempt"] == terminal
    assert report["status"] == "failed" and report["result"] is None
    assert report["assignments"][0]["evaluation_status"] == "failed"
    assert "metrics" not in report["assignments"][0]
    _assert_unobserved_accounting(_assert_terminal_accounting(run, protocol, report), 1, 1)
    _forbid_prediction_replay(monkeypatch)
    with pytest.raises(ValueError):
        runner.verify_result(run, case["registry"])


def _rehash(document, key):
    document[key] = digest({name: value for name, value in document.items() if name != key})


def _rehash_attempt(evidence):
    attempt = evidence["attempt"]
    attempt["binding_sha256"] = digest(attempt["binding"])
    attempt["result_sha256"] = digest(attempt["result"]) if attempt["result"] is not None else None
    reservation = {key: value for key, value in attempt.items() if key not in ("reservation_sha256", "lifecycle_sha256")}
    reservation.update(status="authorized_outcome_unknown", reason="evaluation_authorization_committed",
                       callback_invocations=None, result=None, result_sha256=None)
    attempt["reservation_sha256"] = digest(reservation)
    _rehash(attempt, "lifecycle_sha256")
    for key in ("binding_sha256", "reservation_sha256", "lifecycle_sha256", "result_sha256"):
        evidence[key] = attempt[key]


def test_consumer_rejects_missing_opaque_rehashed_and_contradictory_evidence(support_case, monkeypatch):
    case = support_case
    run, _ = _prepare(case)
    assert _evaluate(case, run)["status"] == "succeeded"
    # Parsing the persisted document makes its displayed assignments independent
    # of the result object, as they are for a downstream report consumer.
    valid = runner.read_json(run / "evaluation.json")
    _forbid_prediction_replay(monkeypatch)
    assert runner.verify_result(run, case["registry"])
    mutations = [
        (("evidence",), None), (("evidence",), {"status": "succeeded", "result_sha256": digest(valid["result"])}),
        (("schema",), "wsp-prospective-support-evaluation/1"), (("status",), "abstained"),
        (("sample_threshold_accounting",), None),
        (("sample_threshold_accounting", "assigned_count"), True),
        (("sample_threshold_accounting", "sample_threshold_success_count"), 0),
        (("sample_threshold_accounting", "conditional_target_gain_mean"), None),
        (("evidence", "attempt_id"), "another-attempt"),
        (("evidence", "schema"), "legacy-decision/1"), (("evidence", "operation"), "consume"),
        (("evidence", "verdict"), "unverified"), (("evidence", "reason"), "not_resolved"),
        (("evidence", "scoring_started"), False),
        (("evidence", "registry_revision_before"), -1), (("evidence", "registry_revision_after"), -1),
        (("evidence", "registry_state_sha256_before"), "0" * 64),
        (("evidence", "registry_state_sha256_after"), "0" * 64),
        (("evidence", "consumed_matches"), [{"kind": "family", "key": "contradiction"}]),
        (("evidence", "reservation_sha256"), "0" * 64), (("evidence", "lifecycle_sha256"), "0" * 64),
        (("result", "schema"), "wsp-prospective-support-result/1"), (("result", "parent_codes"), [1, 0]),
        (("result", "evaluator_id"), "prospective-support-native-readout/1"),
        (("result", "sample_threshold_accounting"), None),
        (("assignments", 0, "metrics", "target_gain"), -1.), (("assignments",), []),
        (("inputs", "raw", "parents"), []), (("inputs", "normalized", "source_id"), "false-alias"),
        (("inputs", "normalized", "family_id"), "false-family"),
        (("inputs", "normalized", "parents"), ["undeclared-parent"]),
    ]
    mutations.extend(((key,), True) for key in runner.FLAGS)
    mutations.extend(((document, "evaluator_execution_attested"), True)
                     for document in ("result", "evidence"))
    for path, replacement in mutations:
        altered = copy.deepcopy(valid)
        target = altered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        if path[0] == "inputs":
            _rehash(altered["inputs"][path[1]], "manifest_sha256")
        if isinstance(altered["evidence"], dict):
            _rehash(altered["evidence"], "decision_sha256")
        with pytest.raises(ValueError):
            runner.verify_result(run, case["registry"], altered)

    for location in (None, "result"):
        altered = copy.deepcopy(valid)
        document = altered if location is None else altered[location]
        del document["sample_threshold_accounting"]
        with pytest.raises(ValueError):
            runner.verify_result(run, case["registry"], altered)

    for change in ("attempt", "binding", "result", "accounting", "assurance"):
        altered = copy.deepcopy(valid)
        attempt = altered["evidence"]["attempt"]
        if change == "attempt":
            attempt["attempt_id"] = altered["evidence"]["attempt_id"] = "rehashed-alternative"
        elif change == "binding":
            attempt["binding"]["protocol_sha256"] = "0" * 64
        elif change == "result":
            altered["result"]["parent_codes"] = [1, 0]
            attempt["result"] = copy.deepcopy(altered["result"])
        elif change == "accounting":
            accounting = copy.deepcopy(altered["sample_threshold_accounting"])
            accounting["all_assigned_sample_threshold_success_rate"] = 0.
            altered["sample_threshold_accounting"] = accounting
            altered["result"]["sample_threshold_accounting"] = copy.deepcopy(accounting)
            attempt["result"] = copy.deepcopy(altered["result"])
        else:
            attempt["evaluator_execution_attested"] = True
        _rehash_attempt(altered["evidence"])
        _rehash(altered["evidence"], "decision_sha256")
        with pytest.raises(ValueError):
            runner.verify_result(run, case["registry"], altered)
    assert runner.verify_result(run, case["registry"])


def test_frozen_bytes_reject_before_forward_and_rehashed_bundles_fail_durable_gate(support_case, monkeypatch):
    case = support_case
    run, protocol = _prepare(case)
    candidate_path = run / protocol["assignments"][0]["artifact_directory"] / "candidate.json"
    frozen_candidate = candidate_path.read_bytes()
    candidate_path.write_bytes(b"{}\n")
    with monkeypatch.context() as patch:
        _forbid_prediction_replay(patch)
        with pytest.raises(ValueError, match="frozen_file_identity_mismatch"):
            _evaluate(case, run)
    assert not (run / "evaluation-request.json").exists()
    assert DataRoleRegistry(case["registry"]).get_attempt(protocol["attempt_id"]) is None
    candidate_path.write_bytes(frozen_candidate)
    assert _evaluate(case, run)["status"] == "succeeded"
    parent = runner.load_parent(run / "parent")["head"]
    _forbid_prediction_replay(monkeypatch)
    for change in ("file-list", "extra-path", "artifact-path", "config", "coherent-artifact", "evaluator"):
        altered_run = case["root"] / ("altered-" + change)
        shutil.copytree(run, altered_run)
        altered = copy.deepcopy(protocol)
        if change == "file-list":
            del altered["files"]["configuration.json"]
        elif change == "extra-path":
            altered["files"]["../outside.json"] = "0" * 64
        elif change == "artifact-path":
            altered["assignments"][0]["artifact_directory"] = "../outside"
        elif change == "config":
            altered["configuration"]["admission"]["maximum_protected_harm"] = .5
            altered["configuration_sha256"] = digest(altered["configuration"])
            _write_json(altered_run / "configuration.json", altered["configuration"])
            altered["files"]["configuration.json"] = _sha(altered_run / "configuration.json")
        elif change == "coherent-artifact":
            record = altered["assignments"][0]
            candidate = runner.TensorArtifact.from_dict(runner.read_json(altered_run / record["artifact_directory"] / "candidate.json"))
            bias = candidate.tensors[research_tasks.HEAD_BIAS].copy()
            bias[0] += .125
            candidate = candidate.replace(**{research_tasks.HEAD_BIAS: bias})
            program = difference(parent, candidate, "synthetic rehashed tamper")
            record.update(candidate_sha256=candidate.sha256, program_sha256=program.sha256)
            admission = altered["configuration"]["admission"]
            record["structural"] = runner.StructuralPolicy(
                (research_tasks.HEAD_WEIGHT, research_tasks.HEAD_BIAS), parent.parameter_count,
                admission["max_parameter_abs"], admission["max_change_abs"]).check(parent, program)
            for name, value in (("candidate.json", candidate.to_dict()), ("program.json", program.to_dict())):
                path = record["artifact_directory"] + "/" + name
                _write_json(altered_run / path, value)
                altered["files"][path] = _sha(altered_run / path)
        else:
            altered["evaluator"]["id"] = "unbound-evaluator"
        _rehash(altered, "protocol_sha256")
        _write_json(altered_run / "protocol.json", altered)
        if change in ("config", "coherent-artifact"):
            loaded, _, _ = runner._load_frozen(altered_run, DataRoleRegistry(case["registry"]))
            assert loaded["protocol_sha256"] == altered["protocol_sha256"]
        report = runner.read_json(altered_run / "evaluation.json")
        report["protocol_sha256"] = altered["protocol_sha256"]
        with pytest.raises(ValueError):
            runner.verify_result(altered_run, case["registry"], report)


def test_parent_identity_failure_retains_assignments_without_historical_reads(support_case):
    case = support_case
    lineage = runner.read_json(case["parent"] / "lineage.json")
    lineage["feature_id"] = "0" * 64
    _write_json(case["parent"] / "lineage.json", lineage)
    with pytest.raises(ValueError, match="parent_identity_mismatch"):
        _prepare(case)
    manifest = runner.read_json(case["root"] / "run" / "manifest.json")
    assert manifest["phase"] == "preparation_failed"
    assert [row["assignment_id"] for row in manifest["assignments"]] == ["repair-billing"]
    assert all(row["proposal_status"] == "failed" for row in manifest["assignments"])
    assert not (case["root"] / "run" / "protocol.json").exists()
