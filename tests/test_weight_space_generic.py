"""Architecture and evidence boundaries that must hold beyond any environment."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rlcd.weight_space import (
    Edit, InputBox, Network, Program, SemanticsAdapter, SemanticsRegistry,
    StructuralPolicy, TensorArtifact, apply_program, compose, default_registry,
    difference, rollback, to_tensor_artifact,
)
from rlcd.weight_space.__main__ import main
from rlcd.weight_space import evidence
from rlcd.weight_space.rts_study import _verify_evaluator_sources, select_candidate, summarize_test


def generic():
    return TensorArtifact("caller-conv-tanh/1", {
        "0.weight": np.arange(16.).reshape(2, 2, 2, 2),
        "head.bias": np.array(.5), "attention/query": np.eye(2),
    }, {"channels": ["red", "green"], "activation": "tanh"})


def test_arbitrary_ranks_scalar_edits_atomicity_and_immutable_interface():
    base = generic()
    assert TensorArtifact.from_dict(base.to_dict()).sha256 == base.sha256
    assert base.parameter_count == 21
    metadata = base.metadata
    metadata["channels"].reverse()
    assert base.metadata["channels"] == ["red", "green"]
    for array in base.tensors.values():
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(TypeError):
        base.tensors["new"] = np.zeros(2)
    with pytest.raises(AttributeError):
        base._architecture = "changed"
    candidate = base.replace(**{"head.bias": np.array(.75)})
    patch = difference(base, candidate, "Raise the scalar bias")
    assert patch.edits[0].index == ()
    assert apply_program(base, patch).sha256 == candidate.sha256
    undo = rollback(base, patch)
    assert apply_program(base, compose(base, [patch, undo])).sha256 == base.sha256
    for bad in (Edit("head.bias", (0,), .5, 1.), Edit("0.weight", (0, 0), 0., 1.),
                Edit("unknown", (), .5, 1.)):
        invalid = Program(base.sha256, (bad,), "Bad rank or undeclared tensor")
        with pytest.raises(ValueError):
            apply_program(base, invalid)
    assert base.tensors["head.bias"] == .5
    with pytest.raises(ValueError, match="shapes"):
        base.replace(**{"head.bias": [.5]})
    changed_interface = TensorArtifact(base.architecture, base.tensors, {"channels": ["green", "red"], "activation": "tanh"})
    with pytest.raises(ValueError, match="schemas"):
        difference(base, changed_interface, "Same numbers, incompatible meaning")


def test_generic_structural_acceptance_never_promises_behavior():
    base = generic()
    patch = difference(base, base.replace(**{"head.bias": np.array(1.5)}), "Increase bias")
    accepted = StructuralPolicy(("head.bias",), 1, 16., 1.).check(base, patch)
    assert accepted["structural_accepted"]
    assert accepted["behavioral_status"] == "not_checked" and accepted["formal_certificate"] is None
    rejected = StructuralPolicy(("0.weight",), 0, 14., .5).check(base, patch)
    assert set(rejected["issues"]) == {"write-set", "edit-count", "parameter-bound", "change-bound"}
    # Binary64 subtraction rounds this difference to 1.0; exact arithmetic sees
    # it is larger. The structural change limit must not accept that rounding.
    tiny = TensorArtifact("scalar/1", {"x": np.array(-2.0**-54)})
    change = difference(tiny, tiny.replace(x=np.array(1.)), "Exact-change boundary")
    assert 1. - float(tiny.tensors["x"]) == 1.
    assert not StructuralPolicy(("x",), 1, 2., 1.).check(tiny, change)["structural_accepted"]


def test_registry_requires_explicit_semantics_and_proof_backend():
    base, registry = generic(), SemanticsRegistry()
    with pytest.raises(ValueError, match="No execution adapter"):
        registry.evaluate(base, [[1.]])
    registry.register(SemanticsAdapter(base.architecture, lambda a, x: np.asarray(x) + a.tensors["head.bias"]))
    np.testing.assert_array_equal(registry.evaluate(base, [[1.]]), [[1.5]])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(SemanticsAdapter(base.architecture, lambda a, x: x))
    with pytest.raises(ValueError, match="No behavioral proof backend"):
        registry.certify(base, base, InputBox([0.], [1.]))
    with pytest.raises(ValueError, match="nonfinite"):
        registry.evaluate(base, [[float("nan")]])
    relu = to_tensor_artifact(Network([[1.]], [0.], [[1.]], [0.]))
    after = relu.replace(w2=[[2.]])
    certificate = default_registry().certify(relu, after, InputBox([0.], [1.]))
    assert certificate["base_artifact_sha256"] == relu.sha256
    assert certificate["candidate_artifact_sha256"] == after.sha256
    assert certificate["certificate"]["output_bounds"] == ["1"]
    assert certificate["certificate"]["base_sha256"] != relu.sha256
    with pytest.raises(ValueError):
        default_registry().certify(base, base, InputBox([0.], [1.]))


def test_cli_generic_roundtrip_and_semantics_failure_is_explicit(tmp_path, capsys):
    base, candidate = generic(), generic().replace(**{"head.bias": np.array(.75)})
    evidence.write_json(tmp_path / "base.json", base.to_dict())
    evidence.write_json(tmp_path / "candidate.json", candidate.to_dict())
    assert main(["diff", str(tmp_path / "base.json"), str(tmp_path / "candidate.json"),
                 "--intent", "Change scalar bias", "--out", str(tmp_path / "program.json")]) == 0
    capsys.readouterr()
    assert main(["apply", str(tmp_path / "base.json"), str(tmp_path / "program.json"),
                 "--out", str(tmp_path / "applied.json")]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["behavioral_status"] == "not_checked"
    assert TensorArtifact.from_dict(evidence.read_json(tmp_path / "applied.json")).sha256 == candidate.sha256
    evidence.write_json(tmp_path / "box.json", {"lower": [0.], "upper": [1.]})
    with pytest.raises(SystemExit) as failure:
        main(["certify", str(tmp_path / "base.json"), str(tmp_path / "candidate.json"),
              "--box", str(tmp_path / "box.json"), "--out", str(tmp_path / "certificate.json")])
    assert failure.value.code == 2 and not (tmp_path / "certificate.json").exists()


def test_cli_generic_certificate_rejects_numeric_substitute_for_boolean(tmp_path, capsys):
    base = to_tensor_artifact(Network([[1.]], [0.], [[1.]], [0.]))
    candidate = base.replace(w2=[[2.]])
    evidence.write_json(tmp_path / "base.json", base.to_dict())
    evidence.write_json(tmp_path / "candidate.json", candidate.to_dict())
    evidence.write_json(tmp_path / "box.json", {"lower": [0.], "upper": [1.]})
    paths = [str(tmp_path / name) for name in ("base.json", "candidate.json")]
    assert main(["certify", *paths, "--box", str(tmp_path / "box.json"),
                 "--out", str(tmp_path / "valid.json")]) == 0
    capsys.readouterr()
    assert main(["verify", *paths, str(tmp_path / "valid.json")]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    # Python container equality conflates False and 0, but these JSON claims
    # have distinct meanings and identities. Leave the recorded digest intact.
    altered = evidence.read_json(tmp_path / "valid.json")
    assert altered["certificate"]["implementation_verified"] is False
    altered["certificate"]["implementation_verified"] = 0
    evidence.write_json(tmp_path / "altered.json", altered)
    with pytest.raises(SystemExit) as failure:
        main(["verify", *paths, str(tmp_path / "altered.json")])
    assert failure.value.code == 2
    assert "Certificate failed exact recomputation" in capsys.readouterr().err


def test_freeze_checks_protocol_current_code_and_archived_copies(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    source, protocol = root / "runtime.py", root / "protocol.json"
    source.write_text("x = 1\n")
    protocol.write_text('{"fixed":true}\n')
    monkeypatch.setattr(evidence, "ROOT", root)
    monkeypatch.setattr(evidence, "execution_sources", lambda: [source])
    receipt = evidence.freeze(protocol, root / "freeze")
    path = root / "freeze" / "receipt.json"
    assert evidence.verify_freeze(protocol, path)["sha256"] == receipt["sha256"]
    original = source.read_bytes()
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="Execution sources"):
        evidence.verify_freeze(protocol, path)
    source.write_bytes(original)
    (root / "freeze" / "sources" / "runtime.py").write_text("altered snapshot")
    with pytest.raises(ValueError, match="Frozen source"):
        evidence.verify_freeze(protocol, path)
    with pytest.raises(ValueError, match="outside"):
        evidence.verify_manifest(root / "freeze", {"../runtime.py": evidence.file_hash(source)})


def test_evaluator_source_match_is_exact_not_just_protocol_match():
    frozen = {"protocol_sha256": "protocol", "source_sha256": {
        "scripts/weight-space-empires.ts": "runner", "package-lock.json": "lock",
        "game/src/empires/policy.ts": "policy", "game/src/empires/new-semantic.ts": "new",
        "rlcd/weight_space/model.py": "python"}}
    actual = {"protocol_sha256": "protocol", "source_sha256": {
        key: value for key, value in frozen["source_sha256"].items() if not key.startswith("rlcd/")}}
    _verify_evaluator_sources(actual, frozen)
    for name in actual["source_sha256"]:
        changed = copy.deepcopy(actual)
        changed["source_sha256"][name] = "other"
        with pytest.raises(ValueError, match="frozen evaluator"):
            _verify_evaluator_sources(changed, frozen)


def test_validation_selection_keeps_baseline_eligible_and_test_is_bound_to_it(tmp_path, monkeypatch):
    import rlcd.weight_space.rts_study as study
    protocol = {"rts": {"validation_seeds": [910001, 910002], "test_seeds": [910003, 910004],
                        "candidate_recruitment_deltas": [1.], "selection_rule": "utility; ties smaller delta"},
                "statistical_policy": {"bootstrap_seed": 3, "bootstrap_resamples": 100}}
    protocol_path = tmp_path / "protocol.json"
    evidence.write_json(protocol_path, protocol)
    frozen = {"sha256": "frozen", "protocol_sha256": evidence.file_hash(protocol_path),
              "source_sha256": {"scripts/weight-space-empires.ts": "runner", "package-lock.json": "lock"}}
    monkeypatch.setattr(study, "verify_freeze", lambda p, f: frozen)
    programs_path = tmp_path / "programs.json"
    evidence.write_json(programs_path, {"candidates": [{"id": "recruit_1_0", "model": {}, "source": "toy"}]})
    prepared = {"schema": "wsp-rts-candidates/1", "freeze_sha256": frozen["sha256"],
                "protocol_sha256": frozen["protocol_sha256"], "programs_sha256": evidence.file_hash(programs_path),
                "artifacts": {"programs.json": evidence.file_hash(programs_path)}}
    from rlcd.weight_space.model import digest
    prepared["sha256"] = digest(prepared)
    evidence.write_json(tmp_path / "receipt.json", prepared)

    def save_results(split, bundle, include_candidate):
        directory = tmp_path / split
        directory.mkdir()
        manifest, runs = {}, []
        ids = ["control_noop", "control_b2_shift"] + (["recruit_1_0"] if include_candidate else [])
        for seed in protocol["rts"][f"{split}_seeds"]:
            for identifier in ids:
                raw_file = f"raw-{seed}-{identifier}.json"
                raw_hash = evidence.write_json(directory / raw_file, {"toy": True})
                manifest[raw_file] = raw_hash
                metric = {"utility": 1., "defeat": False, "victory": True,
                          "recruits_completed": 0, "elapsed_seconds": 100.}
                runs.append({"candidate_id": identifier, "seed": seed, "split": split, "team": 0,
                             "baseline": metric.copy(), "candidate": metric.copy(), "deltas": {"utility": 0.},
                             "behavior_equal": True, "baseline_checkpoint_sha256": str(seed),
                             "raw_result_file": raw_file, "raw_result_sha256": raw_hash})
        results = {"protocol_sha256": frozen["protocol_sha256"], "source_sha256": frozen["source_sha256"],
                   "programs_sha256": evidence.file_hash(bundle), "split": split, "runs": runs,
                   "failures": [], "controls_pass": True, "planned_runs": len(runs), "completed_runs": len(runs)}
        manifest["results.json"] = evidence.write_json(directory / "results.json", results)
        evidence.write_json(directory / "receipt.json", {"protocol_sha256": frozen["protocol_sha256"],
                            "source_sha256": frozen["source_sha256"], "artifacts": manifest})
        return directory / "results.json"

    validation = save_results("validation", programs_path, True)
    chosen = study.select_candidate(validation, programs_path, protocol_path, "unused", tmp_path / "selection")
    assert chosen["baseline_selected"] and chosen["selected_id"] == "control_noop"
    test = save_results("test", tmp_path / "selection" / "programs.json", False)
    summary = study.summarize_test(test, tmp_path / "selection" / "selection.json", protocol_path,
                                  "unused", tmp_path / "summary.json")
    assert summary["H3"]["status"] == "unsupported" and summary["H3"]["effect"]["mean"] == 0.
    with pytest.raises(ValueError, match="Incomplete"):
        study.select_candidate(test, programs_path, protocol_path, "unused", tmp_path / "leaked")
    raw = next((tmp_path / "test").glob("raw-*.json"))
    raw.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="altered"):
        study.summarize_test(test, tmp_path / "selection" / "selection.json", protocol_path,
                             "unused", tmp_path / "tampered-summary.json")
