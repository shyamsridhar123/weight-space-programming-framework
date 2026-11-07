"""Independent numerical and end-to-end controls on non-study worlds."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.benchmarks import (FAMILIES, build_model, features, forward, gradient_program,
                                         head_for, make_world, metrics, native_forward, random_matched_program)
from rlcd.weight_space.evidence import ROOT, read_json, write_json
from rlcd.weight_space.features import ReadoutHead, compile_feature_edit
from rlcd.weight_space.model import digest
from rlcd.weight_space.program import apply_program
from rlcd.weight_space.sequential import sequential_artifact
from rlcd.weight_space.study_v2 import (choose_candidate, cluster_estimate, freeze, load_protocol,
                                       run, run_ablation_case, run_main_case, run_sequence_case,
                                       summarize, verify_freeze)
from rlcd.weight_space.workflow import verify_trace


@pytest.fixture
def protocol():
    value = load_protocol(ROOT / "docs/weight-space/v2/protocol.json")
    value["seeds"] = [900001, 900002]
    value["sizes"].update(train=96, validation=16, test=32, sequence=16)
    value["bootstrap"]["draws"] = 200
    return value


def test_worlds_have_distinct_teacher_and_sample_streams(protocol):
    first, second = [make_world(seed, protocol["sizes"]) for seed in protocol["seeds"]]
    assert first["teacher"] != second["teacher"]
    ids = [sample for record in first["splits"].values() for sample in record["sample_ids"]]
    assert len(ids) == len(set(ids))
    rows = [tuple(row) for record in first["splits"].values() for row in record["inputs"]]
    assert len(rows) == len(set(rows))
    assert np.asarray(first["splits"]["test_target"]["inputs"])[:, 0].min() > .35
    assert np.asarray(first["splits"]["test_retain"]["inputs"])[:, 0].max() < -.1
    assert first == make_world(900001, protocol["sizes"])


@pytest.mark.parametrize("family", FAMILIES)
def test_training_does_not_read_heldout_and_native_executor_agrees(family, protocol):
    world = make_world(900001, protocol["sizes"])
    base, _ = build_model(family, 900001, world)
    changed = copy.deepcopy(world)
    for name in changed["splits"]:
        if name != "train":
            changed["splits"][name] = {"inputs": "unavailable", "labels": "unavailable"}
    other, _ = build_model(family, 900001, changed)
    assert base.sha256 == other.sha256
    x = world["splits"]["test_target"]["inputs"][:8]
    np.testing.assert_allclose(native_forward(base, x, "float64"), forward(base, x), rtol=2e-13, atol=2e-13)


def test_convolution_features_against_scalar_reference(protocol):
    world = make_world(900003, protocol["sizes"])
    base, _ = build_model("conv_tanh", 900003, world)
    x = np.asarray(world["splits"]["public_target"]["inputs"][:2]).reshape(-1, 4, 4)
    expected = []
    for sample in x:
        row = []
        for channel in range(6):
            for i in range(3):
                for j in range(3):
                    value = float(base.tensors["conv.bias"][channel])
                    for a in range(2):
                        for b in range(2):
                            value += float(sample[i+a, j+b])*float(base.tensors["conv.weight"][channel, 0, a, b])
                    row.append(np.tanh(value))
        expected.append(row)
    np.testing.assert_allclose(features(base, x.reshape(-1, 16)), expected, atol=3e-16)


def test_metrics_obey_constant_shift_and_noop_controls(protocol):
    world = make_world(900004, protocol["sizes"])
    base, _ = build_model("relu_mlp", 900004, world)
    head = head_for(base)
    candidate = base.replace(**{head.bias: base.tensors[head.bias]+protocol["shift"]})
    target, retain = [world["splits"][name]["inputs"] for name in ("test_target", "test_retain")]
    unchanged = metrics(base, base, target, retain, protocol["shift"])
    shifted = metrics(base, candidate, target, retain, protocol["shift"])
    assert unchanged["target_nrmse"] == pytest.approx(1.)
    assert unchanged["retain_nrmse"] == 0.
    assert shifted["target_nrmse"] < 1e-14
    assert shifted["retain_nrmse"] == pytest.approx(1.)
    assert shifted["target_gain_fraction"] == pytest.approx(1.)


def test_gradient_and_random_baselines_have_declared_objective_and_norm(protocol):
    world = make_world(900005, protocol["sizes"])
    base, _ = build_model("relu_mlp", 900005, world)
    target, retain = [features(base, world["splits"][name]["inputs"]) for name in ("public_target", "public_retain")]
    desired = np.tile(protocol["shift"], (len(target), 1))
    head = head_for(base)
    optimal = compile_feature_edit(base, head, target, desired, retain, edit_bias=True)
    exact = apply_program(base, optimal.program)
    _, gradient, _ = gradient_program(base, target, desired, retain, ridge=.01, retention_weight=8., steps=200)
    approximate = apply_program(base, gradient.program)
    def objective(model):
        delta_w = model.tensors[head.weight]-base.tensors[head.weight]
        delta_b = model.tensors[head.bias]-base.tensors[head.bias]
        return (np.mean(np.sum((target@delta_w.T+delta_b-desired)**2, axis=1))
                +8*np.mean(np.sum((retain@delta_w.T+delta_b)**2, axis=1))
                +.01*(np.sum(delta_w**2)+np.sum(delta_b**2)))
    assert objective(exact) <= objective(approximate)+1e-12
    assert objective(approximate) < objective(base)
    _, random, _ = random_matched_program(base, exact, 900005)
    randomized = apply_program(base, random.program)
    original_delta = np.concatenate([(exact.tensors[name]-base.tensors[name]).ravel() for name in (head.weight, head.bias)])
    random_delta = np.concatenate([(randomized.tensors[name]-base.tensors[name]).ravel() for name in (head.weight, head.bias)])
    assert np.linalg.norm(random_delta) == pytest.approx(np.linalg.norm(original_delta), rel=1e-12)
    assert np.array_equal(random_delta != 0, original_delta != 0)
    assert head_for(randomized).feature_id == head.feature_id


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_finite_extreme_feature_errors_do_not_underflow_or_overflow(scale):
    base = TensorArtifact("linear-test/1", {"w": [[0.]]})
    result = compile_feature_edit(base, ReadoutHead("w"), [[1.]], [[scale]], retention="none")
    observed = result.diagnostics["target_rmse_after_projection"]
    assert np.isfinite(observed) and observed > 0
    assert observed/scale == pytest.approx(.01/1.01, rel=1e-12)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_finite_extreme_metrics_and_random_matching(scale):
    base = sequential_artifact([[[0.]]], [[0.]], ["identity"])
    head = head_for(base)
    shifted = base.replace(**{head.bias: [scale]})
    unchanged = metrics(base, base, [[0.]], [[0.]], [scale])
    changed = metrics(base, shifted, [[0.]], [[0.]], [scale])
    assert unchanged["target_nrmse"] == 1. and unchanged["target_gain_fraction"] == 0.
    assert changed["target_nrmse"] == 0. and changed["target_gain_fraction"] == 1.
    assert changed["retain_nrmse"] == 1. and changed["delta_l2"]/scale == pytest.approx(1.)
    _, lowered, details = random_matched_program(base, shifted, 900009)
    random = apply_program(base, lowered.program)
    assert abs(float(random.tensors[head.bias][0]))/scale == pytest.approx(1.)
    assert details["requested_l2"]/scale == pytest.approx(1.)
    with pytest.raises(ValueError, match="nonzero shift"):
        metrics(base, shifted, [[0.]], [[0.]], [0.])


def test_study_adapter_rejects_nonlinear_final_layer():
    nonlinear = sequential_artifact([[[1.]]], [[0.]], ["relu"])
    for action in (lambda: head_for(nonlinear), lambda: features(nonlinear, [[-1.]]),
                   lambda: forward(nonlinear, [[-1.]])):
        with pytest.raises(ValueError, match="identity final"):
            action()


def test_validation_selector_retains_noop_fallback():
    records = {"no_op": {"structural_accepted": True, "validation": {"target_nrmse": 1., "retain_nrmse": 0., "delta_l2": 0.}},
               "bad_retention": {"structural_accepted": True, "validation": {"target_nrmse": 0., "retain_nrmse": 1., "delta_l2": 2.}},
               "good": {"structural_accepted": False, "validation": {"target_nrmse": .1, "retain_nrmse": .1, "delta_l2": 2.}}}
    rule = {"max_retain_nrmse": .35, "max_target_nrmse": 1.}
    assert choose_candidate(records, rule) == "no_op"
    records["good"]["structural_accepted"] = True
    assert choose_candidate(records, rule) == "good"


def test_cluster_bootstrap_preserves_independent_unit():
    result = cluster_estimate([1., 1., 1.], 500, 7)
    assert result["independent_worlds"] == 3
    assert result["ci95_percentile"] == [1., 1.]
    with pytest.raises(ValueError):
        cluster_estimate([], 500, 7)


def test_study_roundtrip_seals_selection_and_audits_auxiliary_edits(tmp_path, protocol):
    import torch
    torch.set_num_threads(1)
    world = make_world(900006, protocol["sizes"])
    base, _ = build_model("relu_mlp", 900006, world)
    main = run_main_case(base, world, "relu_mlp", protocol, tmp_path / "main")
    assert not main["failures"]
    assert len(main["arms"]) == 6
    session = read_json(tmp_path / "main/session.json")
    assert verify_trace(session["trace"], expected_head=session["trace_head_sha256"])
    actions = [event["action"] for event in session["trace"]]
    assert actions.index("seal") < actions.index("heldout_evaluation")
    assert session["usage"]["proposals"] == 6
    assert session["usage"]["weight_values"] == 7*base.parameter_count
    assert main["arms"]["no_op"]["target_nrmse"] == pytest.approx(1.)
    assert main["test_accounting"]["evaluated_splits"] == {"test_target": 32, "test_retain": 32}
    ablations = run_ablation_case(base, world, protocol, tmp_path / "ablations")
    full = ablations["feasibility"][-1]
    assert full["nullspace_dimension"] == 0
    assert full["test"]["no_op"] and full["exactly_zero_on_anchors"]
    assert [row["dtype"] for row in ablations["precision"]] == protocol["precision"]
    sequence = run_sequence_case(base, world, protocol, tmp_path / "sequence")
    for result in sequence["methods"].values():
        assert result["composition_exact"] and result["rollback_exact"]
        assert len(result["stages"]) == 3
    with pytest.raises(FileExistsError):
        run_main_case(base, world, "relu_mlp", protocol, tmp_path / "main")


def test_failed_verification_never_becomes_a_candidate(tmp_path, protocol, monkeypatch):
    import rlcd.weight_space.study_v2 as study
    original = study.verify_feature_retention
    calls = 0
    def reject_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        return False if calls == 1 else original(*args, **kwargs)
    monkeypatch.setattr(study, "verify_feature_retention", reject_first)
    world = make_world(900007, protocol["sizes"])
    base, _ = build_model("relu_mlp", 900007, world)
    result = run_main_case(base, world, "relu_mlp", protocol, tmp_path / "main")
    assert [row["arm"] for row in result["failures"]] == ["unconstrained"]
    assert "unconstrained" not in result["arms"]
    selection = read_json(tmp_path / "main/selection.json")
    assert "unconstrained" not in selection["validation"]
    assert result["usage"]["proposals"] == 5
    assert result["validation_feedback_attempts"] == {"attempted": 5, "completed": 5}
    assert (tmp_path / "main/unconstrained/candidate.json").exists()


@pytest.mark.parametrize("section,key,bad", [
    (None, "training_ridge", True), (None, "training_ridge", -1),
    (None, "precision", []), (None, "feasibility", []),
    ("compiler", "ridge", float("inf")), ("compiler", "gradient_steps", True),
    ("selection", "fallback", "soft_retention"), ("selection", "max_target_nrmse", False),
    ("sequence", "regions", ["all", "all", "all"]), ("sequence", "target", "current plus shift"),
    ("bootstrap", "seed", -1), ("bootstrap", "unit", "individual examples"),
])
def test_protocol_rejects_unsupported_or_mistyped_semantics(tmp_path, protocol, section, key, bad):
    target = protocol if section is None else protocol[section]
    target[key] = bad
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError):
        load_protocol(path)


def test_freeze_binds_protocol_and_snapshot_and_rejects_overwrite(tmp_path, protocol):
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, protocol)
    snapshot = tmp_path / "freeze"
    frozen = freeze(protocol_path, snapshot)
    assert verify_freeze(protocol_path, snapshot / "receipt.json")["sha256"] == frozen["sha256"]
    with pytest.raises(FileExistsError):
        freeze(protocol_path, snapshot)
    source_name = next(name for name in frozen["files"] if name.endswith("features.py"))
    source = snapshot / "sources" / source_name
    source.write_text(source.read_text(encoding="utf-8")+"\n# altered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source"):
        verify_freeze(protocol_path, snapshot / "receipt.json")


@pytest.mark.parametrize("tamper", ["empty_manifest", "protocol_path", "float_bytes", "extra_file", "frozen_protocol"])
def test_freeze_rejects_self_consistent_incomplete_or_misbound_manifest(tmp_path, protocol, tamper):
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, protocol)
    snapshot = tmp_path / "freeze"
    receipt = freeze(protocol_path, snapshot)
    if tamper == "empty_manifest":
        receipt["files"] = {}
    elif tamper == "protocol_path":
        receipt["protocol_path"] = "requirements.lock.txt"
    elif tamper == "float_bytes":
        record = next(iter(receipt["files"].values()))
        record["bytes"] = float(record["bytes"])
    elif tamper == "extra_file":
        (snapshot / "sources/extra.py").write_text("# unexpected code", encoding="utf-8")
    else:
        name = receipt["protocol_path"]
        saved = snapshot / "sources" / name
        changed = copy.deepcopy(protocol)
        changed["training_ridge"] = .06
        saved.write_text(json.dumps(changed), encoding="utf-8")
        from rlcd.weight_space.evidence import file_hash
        receipt["files"][name] = {"sha256": file_hash(saved), "bytes": saved.stat().st_size}
    receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
    (snapshot / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_freeze(protocol_path, snapshot / "receipt.json", current_sources=False)


def test_component_and_auxiliary_failures_keep_main_outcomes_and_mark_incomplete(tmp_path, protocol, monkeypatch):
    import rlcd.weight_space.study_v2 as study
    import rlcd.weight_space.study_tasks_v2 as auxiliary
    protocol["seeds"], protocol["families"] = [900008], ["relu_mlp"]
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    snapshot = tmp_path / "freeze"
    freeze(path, snapshot)
    def fail_ablation(*args):
        raise ValueError("deliberate descriptive-ablation failure")
    monkeypatch.setattr(study, "run_ablation_case", fail_ablation)
    monkeypatch.setattr(auxiliary, "run_auxiliary_tasks", lambda seed, out: {
        "seed": seed, "errors": 1, "cases": 6, "case_results": [],
    })
    result = run(path, snapshot / "receipt.json", tmp_path / "run")
    assert result["complete"] is False
    assert result["completed_checkpoints"] == 1
    assert result["complete_worlds_for_cluster_inference"] == 1
    saved = read_json(tmp_path / "run/results.json")
    assert saved["cases"][0]["main"] is not None
    assert saved["cases"][0]["sequence"] is not None
    assert saved["cases"][0]["ablations"] is None
    assert saved["cases"][0]["status"] == "partial"
    assert saved["arm_denominators"] == dict.fromkeys(study.ARMS, 1)
    assert len(saved["failures"]) == 2
    duplicated = saved["cases"]*2
    protocol["families"] = ["relu_mlp", "tanh_mlp"]
    assert summarize(duplicated, protocol)["complete_worlds_for_cluster_inference"] == 0
