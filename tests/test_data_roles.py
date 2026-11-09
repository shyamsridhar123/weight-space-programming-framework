"""Synthetic checks for the durable, cooperative data-role boundary."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
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


def _rejects_before_scoring(registry, batch, *, reason=None, attempt_id=None, binding=None):
    called = []
    with pytest.raises(DataRoleError) as caught:
        registry.score(batch, lambda rows: called.append(rows) or {"metric": 1},
                       attempt_id=attempt_id, binding=binding)
    assert called == []
    decision = caught.value.decision
    assert decision["status"] == "rejected"
    assert decision["scoring_started"] is False
    if reason is not None:
        assert decision["reason"] == reason
    assert decision["scientific_freshness_verified"] is False
    return decision


def _assert_attempt(registry, attempt_id, *, status, callback_invocations, binding, result=None):
    attempt = registry.get_attempt(attempt_id)
    assert type(attempt) is dict
    assert attempt["attempt_id"] == attempt_id
    assert attempt["registry_id"] == registry.registry_id
    assert attempt["status"] == status
    assert attempt["callback_invocations"] == callback_invocations
    assert attempt["binding"] == binding
    assert attempt["binding_sha256"] == hashlib.sha256(_bytes(binding)).hexdigest()
    lifecycle = {key: value for key, value in attempt.items() if key != "lifecycle_sha256"}
    assert attempt["lifecycle_sha256"] == hashlib.sha256(_bytes(lifecycle)).hexdigest()
    reservation = {key: value for key, value in lifecycle.items() if key != "reservation_sha256"}
    reservation.update(status="authorized_outcome_unknown", reason="evaluation_authorization_committed",
                       callback_invocations=None, result=None, result_sha256=None)
    assert attempt["reservation_sha256"] == hashlib.sha256(_bytes(reservation)).hexdigest()
    assert attempt["consumed_scope_sha256"] == hashlib.sha256(_bytes(attempt["consumed_scope"])).hexdigest()
    assert attempt["evidence_role"] == "evaluation_consumed_on_authorization"
    for key in ("provenance_authenticated", "semantic_independence_verified",
                "scientific_freshness_verified", "evaluator_execution_attested"):
        assert attempt[key] is False
    if status == "succeeded":
        assert attempt["result"] == result
        assert attempt["result_sha256"] == hashlib.sha256(_bytes(result)).hexdigest()
    else:
        assert attempt.get("result") is None
        assert attempt["result_sha256"] is None
    captured = json.loads(_bytes(attempt))
    attempt["status"] = "caller-mutated"
    attempt["binding"]["caller-mutated"] = True
    assert registry.get_attempt(attempt_id) == captured
    return captured


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
    attempt = _assert_attempt(DataRoleRegistry(path), stored["attempt_id"],
                              status="succeeded", callback_invocations=1,
                              binding={}, result=result)
    for key in ("binding_sha256", "reservation_sha256", "lifecycle_sha256"):
        assert stored[key] == attempt[key]
    assert stored["attempt"] == attempt
    _rejects_before_scoring(DataRoleRegistry(path), batch,
                           reason="observed_development_consumption")


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
    control = registry.register_batch([[102]], source_id="unrelated-control", family_id="unrelated-family")
    registry.score(control, lambda rows: {})
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
        registry.score(batch, lambda rows: result, attempt_id="invalid-result")
    decision = caught.value.decision
    assert decision["status"] == "failed"
    assert decision["reason"] == "scorer_failed"
    assert decision["scoring_started"] is True
    assert decision["result_sha256"] is None
    _assert_binding(path, decision)
    _assert_attempt(DataRoleRegistry(path), "invalid-result", status="failed",
                    callback_invocations=1, binding={})
    _rejects_before_scoring(DataRoleRegistry(path), batch,
                           reason="observed_development_consumption")


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
        registry.score(batch, scorer, attempt_id="cancelled")
    assert caught.value is interruption
    assert called == [[[1]]]
    decision = caught.value.data_role_decision
    assert decision["status"] == "failed"
    assert decision["reason"] == "scorer_failed"
    assert decision["scoring_started"] is True
    assert decision["result_sha256"] is None
    assert "SYNTHETIC-SCORER-CANCELLATION" not in json.dumps(decision)
    _assert_binding(path, decision)
    _assert_attempt(DataRoleRegistry(path), "cancelled", status="failed",
                    callback_invocations=1, binding={})
    _rejects_before_scoring(DataRoleRegistry(path), batch,
                           reason="observed_development_consumption")


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
        registry.score(batch, scorer, attempt_id="cancelled-unfinalized")
    assert caught.value is interruption
    assert called == [[[1]]]
    assert caught.value.data_role_decision is None
    monkeypatch.undo()
    restarted = DataRoleRegistry(path)
    _assert_attempt(restarted, "cancelled-unfinalized", status="authorized_outcome_unknown",
                    callback_invocations=None, binding={})
    _rejects_before_scoring(restarted, batch, attempt_id="cancelled-unfinalized",
                           reason="attempt_already_authorized")
    _rejects_before_scoring(restarted, batch, reason="observed_development_consumption")


def test_scorer_error_is_not_disclosed_and_reentry_releases_writer_lock(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    peer = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="control", family_id="control-family")

    def failed_scorer(rows):
        raise RuntimeError("CONFIDENTIAL-EXAMPLE-DO-NOT-PUBLISH")

    with pytest.raises(DataRoleError) as caught:
        registry.score(batch, failed_scorer, attempt_id="failed-scorer")
    assert "CONFIDENTIAL" not in str(caught.value)
    assert "CONFIDENTIAL" not in json.dumps(caught.value.decision)
    decision = caught.value.decision
    _assert_binding(path, decision)
    decision["reason"] = "modified-by-caller"
    assert caught.value.decision["reason"] == "scorer_failed"
    _assert_attempt(peer, "failed-scorer", status="failed", callback_invocations=1, binding={})
    _rejects_before_scoring(peer, batch, reason="observed_development_consumption")
    reentry_batch = registry.register_batch([[2]], source_id="reentry", family_id="reentry-family")

    def reentrant_scorer(rows):
        with pytest.raises(DataRoleError) as inner:
            peer.consume([[99]], reason="reentrant_feedback")
        assert inner.value.decision["reason"] == "callback_reentry"
        raise inner.value

    with pytest.raises(DataRoleError) as reentry:
        registry.score(reentry_batch, reentrant_scorer, attempt_id="reentrant-scorer")
    assert reentry.value.decision["status"] == "failed"
    assert reentry.value.decision["scoring_started"] is True
    _assert_attempt(peer, "reentrant-scorer", status="failed", callback_invocations=1, binding={})
    _rejects_before_scoring(peer, reentry_batch, reason="observed_development_consumption")
    control = peer.register_batch([[99]], source_id="reentry-control", family_id="control-after-reentry")
    _, decision = peer.score(control, lambda rows: {})
    assert decision["verdict"] == "registered_cooperative"


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


def test_authorization_commits_attempt_and_releases_writer_lock_before_scorer(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    peer = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="control", family_id="family")
    alias = registry.register_batch([[1]], source_id="alias", family_id="alias-family")
    disjoint = registry.register_batch([[2]], source_id="disjoint", family_id="disjoint-family")
    binding = {"protocol": {"revision": "synthetic-v1"}, "selection": {"candidate": "one"}}
    expected_binding = json.loads(_bytes(binding))
    scorer_entered, release_scorer = threading.Event(), threading.Event()

    def scorer(rows):
        assert rows == [[1]]
        scorer_entered.set()
        assert release_scorer.wait(10)
        return {"metric": 1}

    with ThreadPoolExecutor(max_workers=1) as pool:
        score_future = pool.submit(registry.score, batch, scorer,
                                   attempt_id="pending", binding=binding)
        try:
            assert scorer_entered.wait(5)
            binding["protocol"]["revision"] = "caller-mutated"
            pending = _assert_attempt(peer, "pending", status="authorized_outcome_unknown",
                                      callback_invocations=None, binding=expected_binding)
            assert pending["source_id"] == "control"
            assert pending["manifest_sha256"] == batch.manifest_sha256
            # A committed reservation is visible and holds no writer lock while
            # the callback is blocked; no scheduling-dependent sleep is needed.
            with sqlite3.connect(path, timeout=0, isolation_level=None) as probe:
                probe.execute("BEGIN IMMEDIATE")
                probe.rollback()
            _rejects_before_scoring(peer, alias, attempt_id="overlap",
                                   reason="observed_development_consumption")
            assert peer.get_attempt("overlap") is None
            disjoint_result, disjoint_decision = peer.score(
                disjoint, lambda rows: {"metric": len(rows)}, attempt_id="disjoint")
            assert disjoint_result == {"metric": 1}
            _assert_binding(path, disjoint_decision)
            assert peer.get_attempt("pending") == pending
        finally:
            release_scorer.set()
        result, score_decision = score_future.result(timeout=5)
    assert result == {"metric": 1}
    assert score_decision["verdict"] == "registered_cooperative"
    _assert_binding(path, score_decision)
    terminal = _assert_attempt(peer, "pending", status="succeeded", callback_invocations=1,
                               binding=expected_binding, result=result)
    assert terminal["reservation_sha256"] == pending["reservation_sha256"]
    assert terminal["lifecycle_sha256"] != pending["lifecycle_sha256"]
    assert score_decision["attempt"] == terminal
    _rejects_before_scoring(peer, batch, reason="observed_development_consumption")


def test_concurrent_overlapping_authorizations_enter_exactly_one_callback(tmp_path):
    path = tmp_path / "roles.sqlite"
    first, second = DataRoleRegistry(path), DataRoleRegistry(path)
    batches = [first.register_batch([[value]], source_id=f"source-{value}", family_id="shared-family")
               for value in (1, 2)]
    barrier = threading.Barrier(2)
    scorer_entered, rejection_finished, release_scorer = (threading.Event() for _ in range(3))
    callbacks = []
    callback_lock = threading.Lock()

    def scorer(rows):
        with callback_lock:
            callbacks.append(rows)
        scorer_entered.set()
        assert release_scorer.wait(10)
        return {"metric": 1}

    def contend(registry, batch, attempt_id):
        barrier.wait(timeout=5)
        try:
            return "accepted", registry.score(batch, scorer, attempt_id=attempt_id)
        except DataRoleError as failure:
            rejection_finished.set()
            return "rejected", failure.decision

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(contend, registry, batch, attempt_id)
                   for registry, batch, attempt_id in zip((first, second), batches, ("race-one", "race-two"))]
        try:
            assert scorer_entered.wait(5)
            assert rejection_finished.wait(5)
            assert len(callbacks) == 1
            attempts = [first.get_attempt(attempt_id) for attempt_id in ("race-one", "race-two")]
            pending = [attempt for attempt in attempts if attempt is not None]
            assert len(pending) == 1
            assert pending[0]["status"] == "authorized_outcome_unknown"
            assert pending[0]["callback_invocations"] is None
        finally:
            release_scorer.set()
        outcomes = [future.result(timeout=5) for future in futures]
    accepted = [outcome for status, outcome in outcomes if status == "accepted"]
    rejected = [outcome for status, outcome in outcomes if status == "rejected"]
    assert len(accepted) == len(rejected) == len(callbacks) == 1
    assert rejected[0]["reason"] == "observed_development_consumption"
    assert rejected[0]["scoring_started"] is False
    _assert_binding(path, rejected[0])
    result, decision = accepted[0]
    _assert_binding(path, decision)
    _assert_attempt(first, decision["attempt_id"], status="succeeded", callback_invocations=1,
                    binding={}, result=result)


@pytest.mark.parametrize("phase", ["before_callback", "during_callback"])
def test_process_crash_keeps_authorization_unknown_consumed_and_nonresumable(tmp_path, phase):
    path, marker = tmp_path / "roles.sqlite", tmp_path / "scorer-entered.txt"
    script = """
import os
import sys
from rlcd.weight_space.data_roles import DataRoleRegistry

path, phase, marker = sys.argv[1:]
registry = DataRoleRegistry(path)
batch = registry.register_batch([[1]], source_id="crashed-source", family_id="crashed-family")
if phase == "before_callback":
    reserve = registry._reserve
    def crash_after_reserve(*args, **kwargs):
        reserve(*args, **kwargs)
        os._exit(23)
    registry._reserve = crash_after_reserve

def scorer(rows):
    assert rows == [[1]]
    if phase == "during_callback":
        with open(marker, "w", encoding="utf-8") as stream:
            stream.write("synthetic scorer entered")
        os._exit(23)
    raise AssertionError("scorer unexpectedly entered")

registry.score(batch, scorer, attempt_id="crashed", binding={"protocol": "synthetic-v1"})
raise AssertionError("crash injection did not terminate")
"""
    child = subprocess.run([sys.executable, "-c", script, str(path), phase, str(marker)],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True,
                           text=True, timeout=15)
    assert child.returncode == 23, child.stderr
    assert marker.exists() is (phase == "during_callback")
    restarted = DataRoleRegistry(path)
    attempt = _assert_attempt(restarted, "crashed", status="authorized_outcome_unknown",
                              callback_invocations=None, binding={"protocol": "synthetic-v1"})
    assert attempt["reason"] == "evaluation_authorization_committed"
    batch = restarted.register_batch([[1]], source_id="crashed-source", family_id="crashed-family")
    _rejects_before_scoring(restarted, batch, attempt_id="crashed",
                           reason="attempt_already_authorized")
    fresh_id_rejection = _rejects_before_scoring(restarted, batch, attempt_id="cannot-resume",
                                                reason="observed_development_consumption")
    _assert_binding(path, fresh_id_rejection)
    assert restarted.get_attempt("cannot-resume") is None
    alias = restarted.register_batch([[1]], source_id="late-alias", family_id="late-alias-family")
    descendant = restarted.register_batch([[2]], source_id="late-child", family_id="late-child-family",
                                           parents=(batch,))
    sibling = restarted.register_batch([[3]], source_id="late-sibling", family_id="crashed-family")
    for related in (alias, descendant, sibling):
        _rejects_before_scoring(restarted, related, reason="observed_development_consumption")
    assert restarted.get_attempt("crashed") == attempt
    control = restarted.register_batch([[99]], source_id="disjoint", family_id="disjoint-family")
    result, decision = restarted.score(control, lambda rows: {"metric": len(rows)}, attempt_id="independent")
    assert result == {"metric": 1}
    _assert_binding(path, decision)


@pytest.mark.parametrize("scorer_fails", [False, True], ids=["successful-scorer", "failed-scorer"])
def test_finalization_failure_keeps_unknown_attempt_and_never_accepts_success(tmp_path, monkeypatch, scorer_fails):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="control", family_id="control-family")
    callbacks = []

    def scorer(rows):
        callbacks.append(rows)
        if scorer_fails:
            raise RuntimeError("synthetic scorer failure")
        return {"metric": 1}

    def unavailable_finish(connection, decision):
        raise sqlite3.OperationalError("synthetic unavailable evidence commit")

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_finish", unavailable_finish)
        with pytest.raises(DataRoleError) as caught:
            registry.score(batch, scorer, attempt_id="unfinalized", binding={"protocol": "synthetic-v1"})
    assert callbacks == [[[1]]]
    assert caught.value.decision["status"] == "failed"
    assert caught.value.decision["reason"] == "registry_unavailable"
    assert caught.value.decision["scoring_started"] is True
    assert caught.value.decision["result_sha256"] is None
    restarted = DataRoleRegistry(path)
    _assert_attempt(restarted, "unfinalized", status="authorized_outcome_unknown",
                    callback_invocations=None, binding={"protocol": "synthetic-v1"})
    with sqlite3.connect(path) as connection:
        decisions = [json.loads(row[0]) for row in connection.execute("SELECT decision_json FROM decisions")]
    assert not any(decision.get("attempt_id") == "unfinalized" and decision["status"] == "succeeded"
                   for decision in decisions)
    _rejects_before_scoring(restarted, batch, attempt_id="unfinalized", reason="attempt_already_authorized")
    _rejects_before_scoring(restarted, batch, attempt_id="cannot-retry",
                           reason="observed_development_consumption")


def test_attempt_id_collision_cannot_replace_binding_or_consume_unrelated_scope(tmp_path):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    first = registry.register_batch([[1]], source_id="first", family_id="first-family")
    unrelated = registry.register_batch([[2]], source_id="unrelated", family_id="unrelated-family")
    binding = {"protocol": {"revision": "synthetic-v1"}, "selection": {"candidate": "one"}}
    expected_binding = json.loads(_bytes(binding))
    raw_result = {"metric": 0.5, "details": {"examples": 1}}
    expected_result = json.loads(_bytes(raw_result))
    result, decision = registry.score(first, lambda rows: raw_result, attempt_id="shared-id", binding=binding)
    _assert_binding(path, decision)
    original_attempt = _assert_attempt(registry, "shared-id", status="succeeded", callback_invocations=1,
                                       binding=expected_binding, result=expected_result)
    binding["protocol"]["revision"] = "caller-mutated"
    raw_result["details"]["examples"] = 99
    result["details"]["examples"] = 88
    decision["attempt"]["binding"]["protocol"]["revision"] = "evidence-mutated"
    decision["attempt"]["result"]["details"]["examples"] = 77
    assert registry.get_attempt("shared-id") == original_attempt
    replacement_binding = {"protocol": {"revision": "synthetic-v2"}, "selection": {"candidate": "two"}}
    collision = _rejects_before_scoring(registry, unrelated, attempt_id="shared-id",
                                       binding=replacement_binding, reason="attempt_already_authorized")
    _assert_binding(path, collision)
    assert registry.get_attempt("shared-id") == original_attempt
    independent_result, independent_decision = registry.score(
        unrelated, lambda rows: {"metric": 1}, attempt_id="unique-id", binding=replacement_binding)
    _assert_binding(path, independent_decision)
    _assert_attempt(registry, "unique-id", status="succeeded", callback_invocations=1,
                    binding=replacement_binding, result=independent_result)


@pytest.mark.parametrize("failure", ["commit", "reservation_value_error"])
def test_reservation_failure_never_enters_scorer_or_consumes_scope(tmp_path, monkeypatch, failure):
    path = tmp_path / "roles.sqlite"
    registry = DataRoleRegistry(path)
    batch = registry.register_batch([[1]], source_id="control", family_id="control-family")
    with sqlite3.connect(path) as connection:
        revision_before = connection.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()[0]
    real_connect = sqlite3.connect
    callbacks = []

    class UncommittableConnection(sqlite3.Connection):
        def commit(self):
            raise sqlite3.OperationalError("synthetic reservation commit failure")

    def unavailable_connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=UncommittableConnection)

    def unavailable_reserve(connection, attempt):
        connection.execute("INSERT INTO attempts VALUES (?, ?)", (attempt["attempt_id"], _bytes(attempt)))
        raise ValueError("synthetic reservation failure before commit")

    with monkeypatch.context() as patch:
        if failure == "commit":
            patch.setattr(sqlite3, "connect", unavailable_connect)
        else:
            patch.setattr(registry, "_reserve", unavailable_reserve)
        with pytest.raises(DataRoleError) as caught:
            registry.score(batch, lambda rows: callbacks.append(rows) or {}, attempt_id="reservation-failure")
    assert callbacks == []
    decision = caught.value.decision
    assert decision["status"] == "rejected"
    assert decision["reason"] == "registry_unavailable"
    assert decision["scoring_started"] is False
    assert decision["registry_revision_before"] == int(revision_before)
    assert decision["registry_revision_after"] is None
    assert decision["registry_state_sha256_after"] is None
    assert registry.get_attempt("reservation-failure") is None
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM consumed").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
        assert connection.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()[0] == revision_before
    result, decision = registry.score(batch, lambda rows: callbacks.append(rows) or {"metric": 1},
                                      attempt_id="reservation-failure")
    assert callbacks == [[[1]]]
    _assert_binding(path, decision)
    _assert_attempt(registry, "reservation-failure", status="succeeded", callback_invocations=1,
                    binding={}, result=result)


@pytest.mark.parametrize("examples", [[], [[float("nan")]], [[float("inf")]], [{1: "nonstring-key"}], [object()]])
def test_registration_rejects_nonfinite_and_opaque_json_without_poisoning_registry(tmp_path, examples):
    registry = DataRoleRegistry(tmp_path / "roles.sqlite")
    with pytest.raises(DataRoleError) as caught:
        registry.register_batch(examples, source_id="bad", family_id="bad-family")
    assert caught.value.decision["scoring_started"] is False
    assert caught.value.decision["verdict"] == "unverified"
    good = registry.register_batch([[1]], source_id="good", family_id="good-family")
    registry.score(good, lambda rows: {})
