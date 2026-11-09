"""One end-to-end check for the trained task's split and numerical boundaries."""
import numpy as np
import pytest

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.features import verify_feature_retention
from rlcd.weight_space.program import apply_program, rollback
from rlcd.weight_space.research_tasks import (
    ARCHITECTURE, compile_arm, encode_inputs, evaluate_head, feature_geometry,
    grouped_roles, head_spec, make_head, modular_rank_witness, train_encoder,
)


def test_trained_readout_end_to_end_and_exact_feasibility():
    inputs = np.array([[1., 0.], [0., 1.]]*12, dtype=np.float32)
    labels = np.array([0, 1]*12, dtype=np.int64)
    roles = grouped_roles(labels, [f"g{i//2}" for i in range(24)], "check")
    for rows in roles.values():
        assert all((i ^ 1) in rows for i in rows)
    trained = train_encoder(inputs, labels, np.arange(16), np.arange(16, 20),
                            seed=71, hidden=4, epochs=18)
    assert trained["initialization_sha256"] != trained["selected_parameters_sha256"]
    assert trained["history"][-1]["training_cross_entropy"] < trained["history"][0]["training_cross_entropy"]
    assert np.array_equal(encode_inputs(trained["encoder"], inputs), trained["features"])
    head = make_head(trained, "a"*64, ["left", "right"], 71, "check")
    assert (evaluate_head(head, trained["features"], native=True).argmax(1) == labels).all()
    assert head.metadata["pretraining_ancestry"].startswith("Independent encoder")
    assert make_head(trained, "b"*64, ["left", "right"], 71, "check").sha256 != head.sha256

    base = TensorArtifact(ARCHITECTURE, {"head.weight": [[1., 0.], [0., 1.]],
                          "head.bias": [0., 0.]}, {"feature_id": "two-features"})
    anchors = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    assert modular_rank_witness(anchors)["full_rank_certified"]
    geometry = feature_geometry(anchors, np.array([[2., 2.]], dtype=np.float32))
    assert geometry["numerical_nullity"] == 0 and geometry["exact_score_lane"] == "only_zero_update"
    compiled = compile_arm(base, np.array([[2., 2.]], dtype=np.float32), anchors, 1, "nullspace")
    assert not compiled["program"].edits
    assert compiled["retention_receipt"]["exactly_zero_on_supplied_features"]
    assert verify_feature_retention(base, compiled["artifact"], head_spec(base), anchors,
                                    compiled["retention_receipt"])
    assert apply_program(compiled["artifact"], rollback(base, compiled["program"])).sha256 == base.sha256

    # An underdetermined anchor matrix must retain missing right singular vectors.
    under = feature_geometry(np.array([[0., 0.]], dtype=np.float32), np.array([[2., 0.]], dtype=np.float32))
    assert under["numerical_nullity"] == 2
    assert under["target_projected_energy_fraction"][0] > 0.
    with pytest.raises(ValueError):
        encode_inputs(trained["encoder"], inputs.astype(np.float64))
    with pytest.raises(ValueError):
        train_encoder(inputs, labels, np.arange(16), np.arange(12, 20), seed=71, epochs=1)


def test_runner_seals_before_heldout_and_retains_missing_assignments(tmp_path, monkeypatch):
    from scripts import run_trained_research as runner

    labels = np.array([0, 1]*50, dtype=np.int64)
    inputs = np.eye(2, dtype=np.float32)[labels]
    data = {"inputs": inputs, "labels": labels, "label_names": ["left", "right"],
            "roles": grouped_roles(labels, [f"row-{i}" for i in range(100)], "synthetic"),
            "ids": np.array([f"optical-digits/train/{i}" for i in range(100)]),
            "preprocessing_dict": {"scope": "synthetic fixture"}, "queue_map": np.arange(3)}
    case = {"case_id": "recognition/left", "type": "recognition", "target_label": 0,
            "target_name": "left", "desired_code": 0, "desired_queue": 0}
    protocol = {"cases": [case], "designation": "synthetic runner check",
                "primary_contrast": "soft minus overlay", "source_manifest_sha256": "f"*64}
    experiment, artifacts = runner.study_paths(tmp_path, "optical-digits")
    runner.write_json(experiment/"protocol.json", protocol)
    monkeypatch.setitem(runner.MAIN_SEEDS, "optical-digits", list(range(71, 77)))
    monkeypatch.setitem(runner.TEST_COUNTS, "optical-digits", 20)
    monkeypatch.setattr(runner, "ARMS", ("unchanged", "overlay", "soft"))
    monkeypatch.setattr(runner, "verify_frozen_source", lambda root, domain: protocol)
    monkeypatch.setattr(runner, "check_raw", lambda root, domain: {})
    monkeypatch.setattr(runner, "load_parent_data", lambda root, domain: data)
    real_train, real_fit = runner.train_encoder, runner.fit_case_arm

    def train(x, y, train_rows, dev_rows, **kwargs):
        if kwargs["seed"] == 76:
            raise RuntimeError("Synthetic training failure")
        return real_train(x, y, train_rows, dev_rows, seed=kwargs["seed"], hidden=4, epochs=5)

    def fit(lineage, source, task, arm, root, phase):
        if lineage["record"]["seed"] == 72 and arm == "soft":
            raise RuntimeError("Synthetic failure before candidate sealing")
        return real_fit(lineage, source, task, arm, root, phase)

    def heldout(root, domain, source):
        seal = runner.read_json(experiment/"global-seal.json")
        assert seal["all_main_selection_complete"] and not seal["official_test_labels_parsed"]
        assert seal["assigned_trials"] == 18 and len(seal["attempts"]) == 14
        assert len(seal["attempt_failures"]) == len(seal["lineage_failures"]) == 1
        # The missing target stratum must remain missing rather than become zero.
        return {"inputs": np.tile([[0., 1.]], (20, 1)).astype(np.float32),
                "labels": np.ones(20, dtype=np.int64), "ids": np.array([f"optical-digits/test/{i}" for i in range(20)]),
                "groups": np.array([f"heldout-{i}" for i in range(20)]), "novel": np.ones(20, dtype=bool),
                "oos_inputs": None, "audit": {"scope": "synthetic fixture; no acquired dataset"}}

    monkeypatch.setattr(runner, "train_encoder", train)
    monkeypatch.setattr(runner, "fit_case_arm", fit)
    monkeypatch.setattr(runner, "load_heldout", heldout)
    summary = runner.main_run(tmp_path, tmp_path/"unused-raw", "optical-digits")
    assert summary["assigned_attempts"] == summary["outcome_missingness"] == 18
    assert summary["primary"]["crossed_exact_group_bootstrap"]["status"] == "not_computed"
    outcomes = runner.read_json(experiment/"all-assigned-outcomes.json")
    assert all(row["sensitivity"]["target_gain"] == [-1, 1] for row in outcomes)
    loaded = runner.load_lineage(artifacts/"main/optical-digits-main-full-71")
    assert loaded["features"]["test_features"].shape == (20, 4)
    replay = runner.replay_study(tmp_path, "optical-digits")
    assert replay["verified_trials"] == 14 and replay["all_replayable_trials_verified"]
    historical = {name: runner.sha(experiment/name) for name in
                  ("protocol.json", "global-seal.json", "all-assigned-outcomes.json")}
    frozen = experiment/"source-freeze/check.txt"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("frozen executable fixture", encoding="utf-8")
    protocol["source_files"] = {"check.txt": {"sha256": runner.sha(frozen), "bytes": frozen.stat().st_size}}
    # The synthetic protocol lives in memory; do not mutate its sealed on-disk identity.
    read = runner.read_json
    monkeypatch.setattr(runner, "read_json", lambda path: protocol if path == experiment/"protocol.json" else read(path))
    analysis = runner.analyze_study(tmp_path, "optical-digits")
    assert analysis["assigned_trials"] == 18 and analysis["analyzable_sealed_trials"] == 14
    assert len(analysis["missing_trial_ids"]) == 4
    assert all(row["assigned"] == row["missing"] == 6 for row in analysis["case_results"])
    assert all(row["evaluation_missing"] for row in analysis["trials"])
    assert all(analysis["input_file_sha256"][name] == identity for name, identity in historical.items())
    assert historical == {name: runner.sha(experiment/name) for name in historical}
    assert all(row["exact_rank_lower_bound"] <= row["exact_rank_upper_bound"] for row in analysis["geometry"])
    frozen.write_text("tampered executable fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="Historical source changed"):
        runner.analyze_study(tmp_path, "optical-digits")
    feature_path = loaded["directory"]/"features.npz"
    feature_path.write_bytes(feature_path.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.load_lineage(loaded["directory"])
