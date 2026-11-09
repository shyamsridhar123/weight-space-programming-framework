"""Synthetic checks for the durable, cooperative data-role boundary."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
import threading

import pytest

from rlcd.weight_space.data_roles import DataRoleError, DataRoleRegistry, RegisteredBatch


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _assert_binding(path, decision):
    digest = decision["decision_sha256"]
    body = {key: value for key, value in decision.items() if key != "decision_sha256"}
    assert digest == hashlib.sha256(_bytes(body)).hexdigest()
    with sqlite3.connect(path) as connection:
        captured = connection.execute("SELECT decision_json FROM decisions WHERE decision_sha256 = ?", (digest,)).fetchone()[0]
    assert json.loads(captured) == decision
    for key in ("provenance_authenticated", "semantic_independence_verified",
                "scientific_freshness_verified", "evaluator_execution_attested"):
        assert decision[key] is False


def _rejects_before_scoring(registry, batch, *, reason=None):
    called = []
    with pytest.raises(DataRoleError) as caught:
        registry.score(batch, lambda rows: called.append(rows) or {"metric": 1})
    assert called == []
    decision = caught.value.decision
    assert decision["status"] == "rejected"
    assert decision["scoring_started"] is False
    if reason is not None:
        assert decision["reason"] == reason
    assert decision["scientific_freshness_verified"] is False
    return decision


def test_disjoint_control_captures_data_and_binds_copied_manifests_and_decisions(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    original = [{"features": [1.0, 2.0], "label": "one"}, {"features": [3.0, 4.0], "label": "two"}]
    batch = registry.register_batch(original, source_id="control", family_id="control-family")
    manifest = batch.to_dict()
    original[0]["features"][0] = 999
    batch.materialize()[0]["features"][0] = 888
    manifest["row_sha256"].clear()
    manifest["family_id"] = "mutated"
    assert batch.materialize()[0]["features"] == [1.0, 2.0]
    assert len(batch.to_dict()["row_sha256"]) == 2
    assert batch.to_dict()["family_id"] == "control-family"
    assert batch.registry_id == registry.registry_id
    assert batch.manifest_sha256 == hashlib.sha256(batch._manifest_bytes).hexdigest()
    assert "features" not in repr(batch)

    def scorer(rows):
        rows[0]["features"][0] = 777
        return {"metric": 0.5, "examples": len(rows)}

    result, decision = registry.score(batch, scorer)
    assert result == {"metric": 0.5, "examples": 2}
    assert decision["verdict"] == "registered_cooperative"
    assert decision["scoring_started"] is True
    assert decision["consumed_matches"] == []
    assert decision["manifest_sha256"] == batch.manifest_sha256
    assert decision["result_sha256"] == hashlib.sha256(_bytes(result)).hexdigest()
    _assert_binding(path, decision)
    digest = decision["decision_sha256"]
    decision["verdict"] = "modified-by-caller"
    with sqlite3.connect(path) as connection:
        stored = json.loads(connection.execute("SELECT decision_json FROM decisions WHERE decision_sha256 = ?", (digest,)).fetchone()[0])
    assert stored["verdict"] == "registered_cooperative"
    assert batch.materialize()[0]["features"] == [1.0, 2.0]
    # The registry guards observed consumption; it does not reserve evaluations.
    _, repeated = registry.score(batch, lambda rows: {"metric": 0.4})
    assert repeated["verdict"] == "registered_cooperative"


def test_raw_consumption_invalidates_per_row_aliases_after_restart(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    feedback = registry.consume([[1.0, 2.0], [3.0, 4.0]], reason="feedback")
    assert feedback["verdict"] == "unverified"
    _assert_binding(path, feedback)
    alias = registry.register_batch([[9.0, 8.0], [3.0, 4.0]], source_id="new-id", family_id="new-family")
    original = registry.register_batch([[1.0, 2.0]], source_id="original", family_id="original-family")
    control = registry.register_batch([[6.0, 7.0]], source_id="control", family_id="unrelated")
    restarted = DataRoleRegistry(path)
    assert restarted.registry_id == registry.registry_id
    for batch in (alias, original):
        rejected = _rejects_before_scoring(restarted, batch, reason="observed_development_consumption")
        assert rejected["verdict"] == "invalidated"
        assert any(match["kind"] == "content" for match in rejected["consumed_matches"])
        _assert_binding(path, rejected)
    _, decision = restarted.score(control, lambda rows: {"metric": 1})
    assert decision["verdict"] == "registered_cooperative"


def test_observed_families_and_transitive_parents_are_monotonic(tmp_path):
    registry = DataRoleRegistry(tmp_path / "roles.sqlite")
    ancestor = registry.register_batch([[1]], source_id="ancestor", family_id="family")
    sibling = registry.register_batch([[2]], source_id="sibling", family_id="family")
    child = registry.register_batch([[3]], source_id="child", family_id="child-family", parents=(ancestor,))
    grandchild = registry.register_batch([[4]], source_id="grandchild", family_id="grandchild-family", parents=("child",))
    control = registry.register_batch([[5]], source_id="control", family_id="control-family")
    assert grandchild.to_dict()["ancestor_source_ids"] == ["ancestor", "child"]
    registry.consume(ancestor, reason="threshold_selection")
    for batch in (ancestor, sibling, child, grandchild):
        assert _rejects_before_scoring(registry, batch)["verdict"] == "invalidated"
    registry.score(control, lambda rows: {})
    late_descendant = registry.register_batch([[6]], source_id="later", family_id="later-family", parents=(grandchild,))
    _rejects_before_scoring(registry, late_descendant)


def test_consumed_descendant_conservatively_invalidates_observed_ancestors_and_siblings(tmp_path):
    registry = DataRoleRegistry(tmp_path / "roles.sqlite")
    ancestor = registry.register_batch([[1]], source_id="ancestor", family_id="root-family")
    child = registry.register_batch([[2]], source_id="child", family_id="child-family", parents=(ancestor,))
    sibling = registry.register_batch([[3]], source_id="sibling", family_id="sibling-family", parents=(ancestor,))
    registry.consume(child, reason="activation_inspection")
    for batch in (ancestor, child, sibling):
        _rejects_before_scoring(registry, batch)


def test_late_alias_registration_propagates_earlier_raw_consumption_into_family(tmp_path):
    registry = DataRoleRegistry(tmp_path / "roles.sqlite")
    disjoint_rows = registry.register_batch([[101]], source_id="other-family-rows", family_id="observed-family")
    registry.consume([[100]], reason="prompt_tuning")
    registry.score(disjoint_rows, lambda rows: {})
    registry.register_batch([[100]], source_id="late-alias", family_id="observed-family")
    assert _rejects_before_scoring(registry, disjoint_rows)["verdict"] == "invalidated"


@pytest.mark.parametrize("mutation,expected", [
    ("source", "unknown_registration"),
    ("family", "registration_mismatch"),
    ("payload", "registration_mismatch"),
    ("malformed", "invalid_registration"),
])
def test_forged_and_tampered_handles_never_authorize_scoring(tmp_path, mutation, expected):
    registry = DataRoleRegistry(tmp_path / "roles.sqlite")
    batch = registry.register_batch([[1]], source_id="known", family_id="known-family")
    forged = RegisteredBatch._from_bytes(batch._payload_bytes, batch._manifest_bytes)
    manifest = json.loads(forged._manifest_bytes)
    if mutation == "source":
        manifest["source_id"] = "unregistered"
    elif mutation == "family":
        manifest["family_id"] = "unregistered-family"
    elif mutation == "payload":
        object.__setattr__(forged, "_payload_bytes", _bytes([[999]]))
    else:
        object.__setattr__(forged, "_manifest_bytes", b"\xff")
    if mutation in ("source", "family"):
        object.__setattr__(forged, "_manifest_bytes", _bytes(manifest))
    _rejects_before_scoring(registry, forged, reason=expected)
    registry.score(batch, lambda rows: {})


def test_unknown_and_foreign_parents_and_conflicting_ids_fail_closed(tmp_path):
    registry = DataRoleRegistry(tmp_path / "roles.sqlite")
    known = registry.register_batch([[1]], source_id="known", family_id="family")
    other = DataRoleRegistry(tmp_path / "foreign.sqlite")
    foreign = other.register_batch([[2]], source_id="foreign", family_id="family")
    for parents, reason in ((["missing"], "unknown_parent"), ([foreign], "foreign_registry")):
        with pytest.raises(DataRoleError) as caught:
            registry.register_batch([[3]], source_id="child", family_id="family", parents=parents)
        assert caught.value.decision["reason"] == reason
        assert caught.value.decision["scoring_started"] is False
    _rejects_before_scoring(registry, foreign, reason="foreign_registry")
    _rejects_before_scoring(registry, [[1]], reason="registered_batch_required")
    with pytest.raises(DataRoleError) as caught:
        registry.register_batch([[9]], source_id="known", family_id="family")
    assert caught.value.decision["reason"] == "conflicting_registration"
    assert registry.register_batch([[1]], source_id="known", family_id="family").to_dict() == known.to_dict()
    registry.score(known, lambda rows: {})


def test_uninitialized_handle_is_rejected_before_scoring(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    decision = _rejects_before_scoring(registry, RegisteredBatch(), reason="invalid_registration")
    _assert_binding(path, decision)
    known = registry.register_batch([[1]], source_id="known", family_id="family")
    registry.score(known, lambda rows: {})


def test_consumption_commits_before_a_failed_development_exposure(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="used", family_id="family")
    decision = registry.consume(batch, reason="feedback")
    _assert_binding(path, decision)
    with pytest.raises(RuntimeError):
        # A session performs this exposure only after consume has returned.
        def expose(rows):
            assert rows == [[1]]
            raise RuntimeError("callback failed after seeing development data")
        expose(batch.materialize())
    _rejects_before_scoring(DataRoleRegistry(path), batch)


@pytest.mark.parametrize("result", [float("nan"), {"metric": float("inf")}, object()])
def test_nonfinite_or_nonjson_score_results_preserve_sanitized_failure_decision(tmp_path, result):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="control", family_id="control-family")
    with pytest.raises(DataRoleError) as caught:
        registry.score(batch, lambda rows: result)
    decision = caught.value.decision
    assert decision["status"] == "failed"
    assert decision["reason"] == "scorer_failed"
    assert decision["scoring_started"] is True
    assert decision["result_sha256"] is None
    _assert_binding(path, decision)
    registry.score(batch, lambda rows: {})


@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit])
def test_cancellation_preserves_original_exception_and_durable_decision(tmp_path, interruption_type):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="control", family_id="control-family")
    interruption = interruption_type("SYNTHETIC-SCORER-CANCELLATION")
    called = []

    def scorer(rows):
        called.append(rows)
        raise interruption

    with pytest.raises(interruption_type) as caught:
        registry.score(batch, scorer)
    assert caught.value is interruption
    assert called == [[[1]]]
    decision = caught.value.data_role_decision
    assert decision["status"] == "failed"
    assert decision["reason"] == "scorer_failed"
    assert decision["scoring_started"] is True
    assert decision["result_sha256"] is None
    assert "SYNTHETIC-SCORER-CANCELLATION" not in json.dumps(decision)
    _assert_binding(path, decision)
    registry.score(batch, lambda rows: {})


@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit])
def test_evidence_persistence_failure_does_not_replace_cancellation(tmp_path, monkeypatch, interruption_type):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="control", family_id="control-family")
    interruption = interruption_type("synthetic cancellation")
    interruption.data_role_decision = {"verdict": "untrusted-scorer-claim", "scientific_freshness_verified": True}
    called = []

    def scorer(rows):
        called.append(rows)
        raise interruption

    def unavailable_finish(connection, decision):
        raise sqlite3.OperationalError("synthetic unavailable evidence commit")

    monkeypatch.setattr(registry, "_finish", unavailable_finish)
    with pytest.raises(interruption_type) as caught:
        registry.score(batch, scorer)
    assert caught.value is interruption
    assert called == [[[1]]]
    assert caught.value.data_role_decision is None
    monkeypatch.undo()
    registry.score(batch, lambda rows: {})


def test_scorer_error_is_not_disclosed_and_reentry_releases_writer_lock(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    peer = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="control", family_id="control-family")

    def failed_scorer(rows):
        raise RuntimeError("CONFIDENTIAL-EXAMPLE-DO-NOT-PUBLISH")

    with pytest.raises(DataRoleError) as caught:
        registry.score(batch, failed_scorer)
    assert "CONFIDENTIAL" not in str(caught.value)
    assert "CONFIDENTIAL" not in json.dumps(caught.value.decision)
    decision = caught.value.decision
    _assert_binding(path, decision)
    decision["reason"] = "modified-by-caller"
    assert caught.value.decision["reason"] == "scorer_failed"

    def reentrant_scorer(rows):
        peer.consume(rows, reason="reentrant_feedback")
        return {}

    with pytest.raises(DataRoleError) as reentry:
        registry.score(batch, reentrant_scorer)
    assert reentry.value.decision["status"] == "failed"
    assert reentry.value.decision["scoring_started"] is True
    assert reentry.value.decision["registry_revision_before"] == reentry.value.decision["registry_revision_after"]
    _, decision = peer.score(batch, lambda rows: {})
    assert decision["verdict"] == "registered_cooperative"
    peer.consume(batch, reason="later_feedback")
    _rejects_before_scoring(registry, batch)


def test_consumption_committed_first_prevents_concurrent_scorer_entry(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="used", family_id="family")
    committed = threading.Event()

    def consume_first():
        decision = DataRoleRegistry(path).consume(batch, reason="feedback")
        committed.set()
        return decision

    def score_after():
        assert committed.wait(3)
        return _rejects_before_scoring(DataRoleRegistry(path), batch)

    with ThreadPoolExecutor(max_workers=2) as pool:
        consume = pool.submit(consume_first)
        score = pool.submit(score_after)
        assert consume.result(timeout=5)["status"] == "succeeded"
        assert score.result(timeout=5)["verdict"] == "invalidated"


def test_scoring_holds_writer_lock_until_result_and_decision_commit(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    peer = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="control", family_id="family")
    scorer_entered, release_scorer = threading.Event(), threading.Event()
    consumer_entered, consumption_committed = threading.Event(), threading.Event()

    def scorer(rows):
        scorer_entered.set()
        assert release_scorer.wait(3)
        return {"metric": 1}

    def consume():
        consumer_entered.set()
        decision = peer.consume(batch, reason="concurrent_feedback")
        consumption_committed.set()
        return decision

    with ThreadPoolExecutor(max_workers=2) as pool:
        score_future = pool.submit(registry.score, batch, scorer)
        try:
            assert scorer_entered.wait(3)
            # Deterministically probe the held SQLite writer lock without sleeps.
            with sqlite3.connect(path, timeout=0, isolation_level=None) as probe:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    probe.execute("BEGIN IMMEDIATE")
            consume_future = pool.submit(consume)
            assert consumer_entered.wait(3)
            assert not consumption_committed.is_set()
        finally:
            release_scorer.set()
        result, score_decision = score_future.result(timeout=5)
        consumed = consume_future.result(timeout=5)
    assert result == {"metric": 1}
    assert score_decision["verdict"] == "registered_cooperative"
    assert consumed["registry_revision_after"] > score_decision["registry_revision_after"]
    assert consumption_committed.is_set()
    _assert_binding(path, score_decision)
    _assert_binding(path, consumed)
    _rejects_before_scoring(registry, batch)


@pytest.mark.parametrize("examples", [[], [[float("nan")]], [[float("inf")]], [{1: "nonstring-key"}], [object()]])
def test_registration_rejects_nonfinite_and_opaque_json_without_poisoning_registry(tmp_path, examples):
    registry = DataRoleRegistry(tmp_path / "roles.sqlite")
    with pytest.raises(DataRoleError) as caught:
        registry.register_batch(examples, source_id="bad", family_id="bad-family")
    assert caught.value.decision["scoring_started"] is False
    assert caught.value.decision["verdict"] == "unverified"
    good = registry.register_batch([[1]], source_id="good", family_id="good-family")
    registry.score(good, lambda rows: {})
