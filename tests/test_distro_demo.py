"""Fresh enterprise illustration: routing effect, supplied-feature proof, and CLI."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rlcd.weight_space.__main__ import main
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.distro import DistroWorkspace
from rlcd.weight_space.distro_demo import create_demo
from rlcd.weight_space.features import ReadoutHead, verify_feature_retention
from rlcd.weight_space.governance import read_json
from rlcd.weight_space.program import Program, apply_program


def test_fresh_demo_routes_target_preserves_supplied_scores_and_binds_evidence(tmp_path):
    root = tmp_path / "support-pilot"
    result = create_demo(root)
    base = TensorArtifact.from_dict(read_json(root / "change/base.json"))
    candidate = TensorArtifact.from_dict(read_json(root / "change/candidate.json"))
    program = Program.from_dict(read_json(root / "change/program.json"))
    checks = read_json(root / "evidence/checks.json")
    receipt = read_json(root / "evidence/retention.json")
    assert apply_program(base, program).sha256 == candidate.sha256

    before = base.tensors["head.weight"]
    after = candidate.tensors["head.weight"]
    target = np.array(checks["target_features"])
    protected = np.array(checks["protected_features"])
    queues = checks["score_order"]
    assert queues[int(np.argmax(before @ target))] == "ordinary-support"
    assert queues[int(np.argmax(after @ target))] == "platform-operations"
    np.testing.assert_array_equal(before @ protected.T, after @ protected.T)
    assert verify_feature_retention(
        base, candidate, ReadoutHead("head.weight", feature_id="declared-ticket-features/1"),
        protected, receipt,
    )
    assert receipt["exactly_zero_on_supplied_features"] is True
    assert receipt["feature_extractor_verified"] is False
    assert receipt["floating_execution_verified"] is False
    altered = candidate.replace(**{"head.weight": (after + 1).tolist()})
    assert not verify_feature_retention(
        base, altered, ReadoutHead("head.weight", feature_id="declared-ticket-features/1"),
        protected, receipt,
    )

    draft = DistroWorkspace(root).inspect_change("change.json")
    assert draft["program"] == program.to_dict()
    assert {item["name"] for item in draft["attachments"]} == {
        "review-note", "supplied-feature-retention", "synthetic-routing-checks",
    }
    for data in (result, checks, draft):
        assert data["production_authorized"] is False
        assert data["scientific_authorized"] is False
    assert result["local_pilot_eligible"] is False
    assert draft["status"] == "draft"
    assert not (root / "review").exists()


def test_demo_exclusively_creates_workspace_and_preserves_existing_content(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    existing = root / "change.json"
    existing.write_bytes(b"existing customer change\n")
    with pytest.raises(FileExistsError):
        create_demo(root)
    assert existing.read_bytes() == b"existing customer change\n"
    assert list(root.iterdir()) == [existing]


def test_cli_demo_and_setup_emit_usable_native_plugin(tmp_path, capsys):
    root = tmp_path / "support pilot"
    assert main(["distro", "demo", "--workspace", str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["before_queue"] == "ordinary-support"
    assert result["after_queue"] == "platform-operations"
    plugin = result["plugin"]
    assert plugin["launch_args"]["claude"] == [
        "distro", "claude", "--plugin-dir", str(root.resolve() / ".runs/distro-plugin"),
    ]
    assert main(["distro", "setup", "--workspace", str(root)]) == 0
    assert json.loads(capsys.readouterr().out) == plugin
    with pytest.raises(SystemExit) as failure:
        main(["distro", "demo", "--workspace", str(root)])
    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "weight-space:" in captured.err
