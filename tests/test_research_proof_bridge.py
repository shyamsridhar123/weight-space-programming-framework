"""Trust-boundary checks for the closed runtime-to-Lean instance bridge."""
from fractions import Fraction
import json
from pathlib import Path
import shutil

import pytest

from rlcd.weight_space import proof_bridge
from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.features import ReadoutHead
from rlcd.weight_space.model import InputBox, digest
from rlcd.weight_space.sequential import sequential_artifact
from scripts import check_weight_space_bridge as runner


def bound_inputs():
    return runner.restore_input(runner.fixtures()["head-nonzero-bound"])


def declaration(source, name, old, new):
    lines = source.splitlines(keepends=True)
    altered = "".join(line.replace(old, new, 1) if line.startswith("def " + name + " :") else line
                      for line in lines)
    assert altered != source
    return altered


@pytest.mark.parametrize("name", list(runner.fixtures()))
def test_supported_fixture_regeneration_and_receipt_scope(name):
    base, candidate, box, options = runner.restore_input(runner.fixtures()[name])
    source = proof_bridge.generate_instance(base, candidate, box, **options)
    assert source == proof_bridge.generate_instance(base, candidate, box, **options)
    receipt = proof_bridge.validate_instance(source, base, candidate, box, **options)
    assert receipt["independent_decoding_passed"] and not receipt["lean_checked"]
    assert receipt["parent_sha256"] == base.sha256 and receipt["candidate_sha256"] == candidate.sha256
    assert not any(receipt[key] for key in ("feature_extractor_verified", "floating_execution_verified",
                                          "full_model_semantics_verified", "useful_business_behavior_verified"))


def test_independent_decoder_walks_original_nonsquare_rows_without_emitter_lift(monkeypatch):
    base = TensorArtifact("nonsquare-head/1", {"w": [[.1, -.2, .3], [2., .5, -1.25]], "b": [.1, -.3]})
    candidate = base.replace(w=[[.2, -.2, .3], [2., .5, -1.]], b=[.1, -.25])
    box = InputBox([-1., -2., -3.], [1., 2., 3.])
    options = {"head": ReadoutHead("w", "b", "features/1"),
               "feature_names": ["a", "b", "c"], "output_names": ["left", "right"]}
    source = proof_bridge.generate_instance(base, candidate, box, **options)
    decoded = proof_bridge.decode_instance(source)
    assert decoded["values"]["candidateWeights"] == [[str(Fraction(*float(candidate.tensors["w"][o, i]).as_integer_ratio()))
                                                    for o in range(2) for i in range(3)]]
    def forbidden(*args):
        raise AssertionError("Validator reused the emitter lifting path")
    monkeypatch.setattr(proof_bridge, "_lift_emitted", forbidden)
    assert proof_bridge.validate_instance(source, base, candidate, box, **options)["independent_decoding_passed"]


def test_substituted_value_decodes_but_cannot_bind_to_original_runtime():
    base, candidate, box, options = bound_inputs()
    source = proof_bridge.generate_instance(base, candidate, box, **options)
    altered = declaration(source, "candidateWeights", "(9 / 8 : ℝ)", "(17 / 16 : ℝ)")
    assert proof_bridge.decode_instance(altered)["values"]["candidateWeights"][0][0] == "17/16"
    with pytest.raises(ValueError, match="candidateWeights"):
        proof_bridge.validate_instance(altered, base, candidate, box, **options)


@pytest.mark.parametrize("name,old,new", [
    ("candidateWeights", "(9 / 8 : ℝ), (-1 / 2 : ℝ)", "(-1 / 2 : ℝ), (9 / 8 : ℝ)"),
    ("candidateBiases", "(-3 / 16 : ℝ)", "(-1 / 8 : ℝ)"),
    ("widths", "[2, 2]", "[1, 4]"),
    ("certifiedActivations", ".identity", ".tanh"),
    ("domainLower", "(-1 / 1 : ℝ)", "(-2 / 1 : ℝ)"),
    ("outputBudget", "(1 / 1 : ℝ)", "(2 / 1 : ℝ)"),
])
def test_theorem_input_mutations_are_rejected(name, old, new):
    base, candidate, box, options = bound_inputs()
    source = proof_bridge.generate_instance(base, candidate, box, **options)
    with pytest.raises(ValueError):
        proof_bridge.validate_instance(declaration(source, name, old, new), base, candidate, box, **options)


@pytest.mark.parametrize("literal", ["(18 / 16 : ℝ)", "(-0 / 1 : ℝ)", "(09 / 8 : ℝ)",
                                    "(9 / 0 : ℝ)", "(9 / -8 : ℝ)", "(1.125 : ℝ)",
                                    "by sorry", "[[[[]]]]", "[" * 10000])
def test_noncanonical_or_executable_literal_inputs_fail_closed(literal):
    base, candidate, box, options = bound_inputs()
    source = proof_bridge.generate_instance(base, candidate, box, **options)
    with pytest.raises(ValueError):
        proof_bridge.decode_instance(declaration(source, "candidateWeights", "(9 / 8 : ℝ)", literal))


def test_extra_axioms_changed_proofs_and_external_input_substitution_fail_closed():
    base, candidate, box, options = bound_inputs()
    source = proof_bridge.generate_instance(base, candidate, box, **options)
    for altered in (source + "\naxiom escaped : False\n", source.replace("  decide\n", "  sorry\n", 1)):
        with pytest.raises(ValueError, match="scaffold"):
            proof_bridge.decode_instance(altered)
    for changed in (options | {"feature_names": list(reversed(options["feature_names"]))},
                    options | {"head": ReadoutHead("head.weight", "head.bias", "different-features/1")},
                    options | {"max_output_drift": [2., 1.]}):
        with pytest.raises(ValueError):
            proof_bridge.validate_instance(source, base, candidate, box, **changed)
    with pytest.raises(ValueError):
        proof_bridge.validate_instance(source, base, candidate, InputBox([-2., -2.], [1., 2.]), **options)
    for changed in ("program", "task"):
        binding = json.loads(json.dumps(options["binding"]))
        if changed == "program":
            binding["program"]["base_sha256"] = "0" * 64
        else:
            binding["task"]["objective"] = "A self-consistent substituted task"
        binding[changed + "_sha256"] = digest(binding[changed])
        binding["sha256"] = digest({key: value for key, value in binding.items() if key != "sha256"})
        with pytest.raises(ValueError):
            proof_bridge.validate_instance(source, base, candidate, box, **(options | {"binding": binding}))


def test_unsupported_executable_activation_and_inferred_feature_domain_are_not_certified():
    unsupported = sequential_artifact([[[1.]]], [[0.]], ["tanh"])
    with pytest.raises(ValueError, match="only identity"):
        proof_bridge.generate_instance(unsupported, unsupported, InputBox([-1.], [1.]))
    base, candidate, _, options = bound_inputs()
    with pytest.raises(ValueError, match="width"):
        proof_bridge.generate_instance(base, candidate, InputBox([-1.] * 3, [1.] * 3), **options)


def test_manifest_requires_external_identity_and_rejects_tampering_or_unlisted_files(tmp_path):
    sources = tmp_path / "checker-sources"
    sources.mkdir()
    shutil.copyfile(Path(proof_bridge.__file__), sources / "proof_bridge.py")
    shutil.copyfile(Path(runner.__file__), sources / "check_weight_space_bridge.py")
    for path in runner.checker_source_closure():
        target = sources / path.relative_to(runner.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    framework = tmp_path / "framework"
    for name in runner.MODULES + (runner.IMPORT_HELPER,):
        target = framework / "WeightSpace" / (name + ".lean")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runner.ROOT / "formal/weight-space/WeightSpace" / (name + ".lean"), target)
    for name in ("lean-toolchain", "lake-manifest.json", "lakefile.toml"):
        shutil.copyfile(runner.ROOT / "formal/weight-space" / name, framework / name)
    manifest = runner.finalize(tmp_path, [{"real_theorem_checked": False, "independent_decoding_passed": False}], [])
    expected = manifest["sha256"]
    assert runner.check_manifest(tmp_path, expected)["sha256"] == expected
    with pytest.raises(ValueError, match="external expected"):
        runner.check_manifest(tmp_path, "0" * 64)
    (tmp_path / "unlisted.txt").write_text("extra")
    with pytest.raises(ValueError, match="unlisted"):
        runner.check_manifest(tmp_path, expected)
    (tmp_path / "unlisted.txt").unlink()
    (sources / "proof_bridge.py").write_text("substituted")
    with pytest.raises(ValueError, match="bytes differ"):
        runner.check_manifest(tmp_path, expected)
