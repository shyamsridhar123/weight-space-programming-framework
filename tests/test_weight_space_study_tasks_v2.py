"""Smoke-only auxiliary checks; reserved study seeds are never executed here."""
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from rlcd.weight_space import Network, Program, TensorArtifact, apply_program
from rlcd.weight_space.evidence import read_json, verify_manifest
from rlcd.weight_space.study_tasks_v2 import (
    DESIGNATED_COEFFICIENT, analytic_network, affine_fixture, run_auxiliary_tasks,
)
from rlcd.weight_space.workflow import verify_trace


@pytest.mark.parametrize("spec", [
    {"kind": "scaled_absolute", "scale": "3/2", "center": "-1/4"},
    {"kind": "clip", "low": "-3/4", "high": "5/4"},
])
def test_analytic_construction_matches_an_independent_rational_formula(spec):
    network = analytic_network(spec)
    assert network.shape == (1, 2, 1)
    points = [Fraction(k, 8) for k in range(-48, 49)]
    if spec["kind"] == "scaled_absolute":
        target = [Fraction(spec["scale"]) * abs(x - Fraction(spec["center"])) for x in points]
    else:
        target = [min(max(x, Fraction(spec["low"])), Fraction(spec["high"])) for x in points]
    actual = network([[float(x)] for x in points])[:, 0]
    assert [Fraction.from_float(float(value)) for value in actual] == target


@pytest.mark.parametrize("spec", [
    {"kind": "scaled_absolute", "scale": "1/3", "center": "0"},
    {"kind": "scaled_absolute", "scale": "0", "center": "0"},
    {"kind": "scaled_absolute", "scale": True, "center": "0"},
    {"kind": "clip", "low": "1", "high": "1"},
    {"kind": "clip", "low": "0", "high": "1", "code": "not executable"},
    {"kind": "unknown"},
])
def test_analytic_constructors_reject_unsupported_or_inexact_specifications(spec):
    with pytest.raises(ValueError):
        analytic_network(spec)


def test_affine_pair_differs_at_exactly_one_small_coefficient():
    positive, negative = affine_fixture(900001, 1), affine_fixture(900001, -1)
    difference = positive.tensors["weight"] != negative.tensors["weight"]
    assert difference.sum() == 1 and difference[DESIGNATED_COEFFICIENT]
    assert positive.tensors["weight"][DESIGNATED_COEFFICIENT] == 0.0004
    assert negative.tensors["weight"][DESIGNATED_COEFFICIENT] == -0.0004
    np.testing.assert_array_equal(positive.tensors["bias"], negative.tensors["bias"])
    with pytest.raises(ValueError):
        affine_fixture(900001, True)


def test_auxiliary_sessions_preserve_access_boundaries_abstentions_and_replayable_evidence(tmp_path):
    out = tmp_path / "smoke-900001"
    summary = run_auxiliary_tasks(900001, out)
    assert summary == read_json(out / "summary.json")
    assert summary["cases"] == 6 and summary["errors"] == 0
    assert summary["correct"] == {"coefficient_sign": 2, "analytic_creation": 2}
    assert summary["abstentions"] == 2 and summary["incorrect_signs"] == 0
    assert summary["provider"] == {"kind": "deterministic_fixture", "llm_calls": 0,
                                   "input_tokens": 0, "output_tokens": 0, "model_ability_measured": False}
    assert summary["formal_certificate"] is None
    verify_manifest(out, read_json(out / "manifest.json")["files"])
    by_case = {case["case_id"]: case for case in summary["case_results"]}
    for label in ("a", "b"):
        assert by_case[f"affine-{label}-exact"]["base_sha256"] == by_case[f"affine-{label}-rounded3"]["base_sha256"]
        assert read_json(out / f"affine-{label}-exact" / "heldout.json")["inputs"] == read_json(
            out / f"affine-{label}-rounded3" / "heldout.json")["inputs"]
    for case in summary["case_results"]:
        directory = out / case["case_id"]
        receipt = read_json(directory / "receipt.json")
        assert verify_trace(read_json(directory / "trace.json"), expected_head=receipt["trace_head_sha256"])
        assert receipt["heldout_accounting"]["evaluator_calls"] == 1
        assert receipt["usage"]["feedback_calls"] == receipt["usage"]["execution_calls"] == 0
        assert receipt["usage"]["input_tokens"] == receipt["usage"]["output_tokens"] == 0
        assert receipt["configuration"]["access"]["metadata"] is False
        assert receipt["trace"][-2]["action"] == "seal"
        assert receipt["trace"][-1]["action"] == "heldout_evaluation"
        heldout = read_json(directory / "heldout.json")
        x = np.asarray(heldout["inputs"])
        assert len(x) == 128
        if case["family"] == "affine_sign":
            assert receipt["usage"]["weight_values"] == 51 and receipt["usage"]["proposals"] == 1
            base = TensorArtifact.from_dict(read_json(directory / "base.json"))
            reconstructed = TensorArtifact.from_dict(heldout["reconstructed_artifact"])
            expected = x @ base.tensors["weight"].T + base.tensors["bias"]
            actual = x @ reconstructed.tensors["weight"].T + reconstructed.tensors["bias"]
            np.testing.assert_array_equal(expected, heldout["reference_outputs"])
            np.testing.assert_array_equal(actual, heldout["reconstructed_outputs"])
            if case["access"] == "exact":
                assert case["metrics"]["sign_correct"] is True
                assert case["metrics"]["hidden_response"]["rmse"] == 0
            else:
                assert case["metrics"]["sign_correct"] is None and case["abstained"] is True
                assert case["metrics"]["reported_sign"] is None
                assert case["metrics"]["hidden_response"]["rmse"] > 0
                assert case["metrics"]["coefficient_absolute_error"] == 0.0004
        else:
            assert receipt["usage"]["proposals"] == 3 and receipt["usage"]["weight_values"] == 0
            assert case["metrics"]["selected_arm"] == "analytic"
            assert receipt["candidate_sha256"] == case["selected_candidate_id"]
            assert set(case["metrics"]["arms"]) == {"zero", "random", "analytic"}
            zero = Network.from_dict(read_json(directory / "zero.network.json"))
            for arm, metrics in case["metrics"]["arms"].items():
                network = Network.from_dict(read_json(directory / f"{arm}.network.json"))
                program = Program.from_dict(read_json(directory / f"{arm}.program.json"))
                assert apply_program(zero, program).sha256 == network.sha256 == metrics["artifact_sha256"]
                np.testing.assert_array_equal(network(x), heldout["arm_outputs"][arm])
            assert heldout["evaluator_accounting"]["evaluated_model_examples"] == 384


def test_repeated_smoke_is_deterministic_and_existing_evidence_is_never_overwritten(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert run_auxiliary_tasks(900002, first) == run_auxiliary_tasks(900002, second)
    assert read_json(first / "manifest.json") == read_json(second / "manifest.json")
    original = (first / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_auxiliary_tasks(900002, first)
    assert (first / "summary.json").read_bytes() == original
    for invalid in (True, -1, 1.5):
        with pytest.raises(ValueError):
            run_auxiliary_tasks(invalid, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_failed_hidden_evaluation_retains_every_attempt_and_declared_abstention(tmp_path, monkeypatch):
    from rlcd.weight_space import study_tasks_v2

    def fail_forward(artifact, examples):
        raise RuntimeError("Intentional hidden-adapter failure")

    monkeypatch.setattr(study_tasks_v2, "_affine_forward", fail_forward)
    out = tmp_path / "failed-900003"
    summary = run_auxiliary_tasks(900003, out)
    assert summary["cases"] == 6 and summary["errors"] == 4 and summary["abstentions"] == 2
    assert summary["correct"] == {"coefficient_sign": 0, "analytic_creation": 2}
    for case in summary["case_results"]:
        if case["family"] == "affine_sign":
            receipt = read_json(out / case["case_id"] / "receipt.json")
            assert receipt["status"] == "failed" and receipt["result"] is None
            assert len(receipt["candidates"]) == 1 and receipt["heldout_accounting"]["evaluator_calls"] == 1
            assert verify_trace(receipt["trace"], expected_head=receipt["trace_head_sha256"])
    verify_manifest(out, read_json(out / "manifest.json")["files"])


def test_failed_constructor_keeps_an_incomplete_case_without_forcing_a_score(tmp_path, monkeypatch):
    from rlcd.weight_space import study_tasks_v2

    original = study_tasks_v2.analytic_network
    def fail_one(spec):
        if spec["kind"] == "clip":
            raise ValueError("Intentional constructor failure")
        return original(spec)

    monkeypatch.setattr(study_tasks_v2, "analytic_network", fail_one)
    out = tmp_path / "incomplete-900004"
    summary = run_auxiliary_tasks(900004, out)
    assert summary["cases"] == 6 and summary["errors"] == 1
    case = next(case for case in summary["case_results"] if case["case_id"] == "create-clip")
    assert case["status"] == "failed" and case["metrics"] is None and case["candidate_ids"] == []
    assert case["partial_trace_present"] is True
    state = read_json(out / "create-clip" / "session.json")
    assert state["sealed"] is False and state["receipt_status"] is None
    assert not (out / "create-clip" / "receipt.json").exists()
