"""Frozen boundary examples supplementing the existing v2 precision study.

These designed examples expose admission and arithmetic failure modes. They are
not a sample of enterprise traffic and have no population effect estimate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .artifact import StructuralPolicy, TensorArtifact
from .contracts import AdmissionRequest, DecisionRule, admit, bind_admission, verify_admission
from .evidence import ROOT, file_hash, read_json, verify_manifest, write_json
from .model import InputBox, canonical_bytes, digest
from .program import Program, difference
from .sequential import sequential_artifact
from .structured import ExecutionSpec, native_execution_receipt
from .workflow import TaskSpec


SOURCE_FILES = tuple("rlcd/weight_space/" + name + ".py" for name in (
    "runtime_research", "artifact", "model", "program", "contracts", "semantics",
    "sequential", "structured", "features", "evidence", "workflow"))
SUBSTITUTIONS = (
    "parent", "candidate", "program", "domain", "adapter_version", "adapter_configuration",
    "input_order", "output_order", "tensor_order", "bias_convention", "witnesses",
    "feature_names", "feature_anchors", "feature_head", "feature_bias", "decision_rule",
    "score_budgets", "structural_policy", "task", "backend_environment",
)


def affine(weights: Any, bias: Any, activation: str = "identity") -> TensorArtifact:
    return sequential_artifact([weights], [bias], [activation])


def _admission_case(name: str, base: TensorArtifact, candidate: TensorArtifact,
                    *, box: InputBox, budgets: list[float], witnesses: list[list[float]],
                    decision: DecisionRule = DecisionRule(), execution: ExecutionSpec | None = None,
                    feature: bool = True, stale: bool = False) -> dict[str, Any]:
    program = difference(base, candidate, name)
    if stale:
        program = Program("a"*64, program.edits, program.intent)
    task = TaskSpec(name, "edit", "Designed admission boundary example", base.architecture, base.sha256)
    return {"case": name, "parent": base.to_dict(), "candidate": candidate.to_dict(),
            "program": program.to_dict(), "task": task.to_dict(), "domain": box.to_dict(),
            "budgets": budgets, "witnesses": witnesses, "decision": decision.to_dict(),
            "policy": StructuralPolicy(tuple(base.tensors), 16, 2.0**56, 2.0**56).to_dict(),
            "execution": None if execution is None else execution.to_dict(),
            "feature": None if not feature else {
                "head": {"weight": "layers.0.weight", "bias": "layers.0.bias",
                         "feature_id": "supplied-affine-inputs/1"},
                "anchors": witnesses, "feature_names": list(base.metadata["input_names"])}}


def bind_case(case: dict[str, Any]) -> AdmissionRequest:
    return bind_admission(TensorArtifact.from_dict(case["parent"]),
        TensorArtifact.from_dict(case["candidate"]), Program.from_dict(case["program"]),
        task=case["task"], policy=StructuralPolicy.from_dict(case["policy"]),
        box=InputBox(**case["domain"]), max_output_drift=case["budgets"],
        decision=DecisionRule.from_dict(case["decision"]), witnesses=case["witnesses"],
        execution=None if case["execution"] is None else ExecutionSpec.from_dict(case["execution"]),
        feature_retention=case["feature"])


def _native_case(name: str, base: TensorArtifact, candidate: TensorArtifact, inputs: Any,
                 spec: ExecutionSpec, rule: DecisionRule = DecisionRule()) -> dict[str, Any]:
    return {"case": name, "parent": base.to_dict(), "candidate": candidate.to_dict(),
            "witnesses": inputs, "execution": spec.to_dict(), "decision": rule.to_dict()}


def boundary_protocol() -> dict[str, Any]:
    """Build input assignments only, before evaluating any admission/native lane."""
    base = affine([[1., 0.], [0., 1.]], [2., 0.])
    candidate = base.replace(**{"layers.0.bias": [2.125, 0.]})
    common = {"box": InputBox([-1., -1.], [1., 1.]), "witnesses": [[.5, .5]]}
    admission = [
        _admission_case("valid-edit", base, candidate, budgets=[.125, 0.],
                        execution=ExecutionSpec(), **common),
        _admission_case("no-op", base, base, budgets=[0., 0.], **common),
        _admission_case("stale-program", base, candidate, budgets=[.125, 0.], stale=True, **common),
        _admission_case("witness-budget-violation", base,
            base.replace(**{"layers.0.bias": [2.25, 0.]}), budgets=[.125, 0.], **common),
        _admission_case("stochastic-decision-unsupported", base, base, budgets=[0., 0.],
                        decision=DecisionRule("stochastic"), **common),
    ]
    loose = affine([[1., 1.], [0., 0.]], [2., 0.])
    admission.append(_admission_case("conservative-bound-rejection", loose,
        loose.replace(**{"layers.0.weight": [[2., 0.], [0., 0.]]}), budgets=[0., 0.], **common))
    tied = affine([[0.], [0.]], [0., 0.])
    for kind, tie in (("unique_argmax", "reject"), ("argmax", "first_index"), ("argmax", "last_index")):
        admission.append(_admission_case("tie-" + tie, tied, tied, box=InputBox([0.], [0.]),
            budgets=[0., 0.], witnesses=[[0.]], decision=DecisionRule(kind, tie)))
    unknown = TensorArtifact("unregistered-affine/1", tied.tensors)
    admission.append(_admission_case("unregistered-semantics", unknown, unknown,
        box=InputBox([0.], [0.]), budgets=[0., 0.], witnesses=[[0.]], feature=False))
    near = affine([[0.], [0.]], [1 + 2.0**-25, 1 + 2.0**-24])
    near_candidate = near.replace(**{"layers.0.bias": [1 + 2.0**-25, 1 + 2.0**-23]})
    tie_rule = DecisionRule("argmax", "first_index")
    near_spec = ExecutionSpec(dtype="float32", endpoint_cast="round")
    admission.append(_admission_case("real-pass-native-fail", near, near_candidate,
        box=InputBox([0.], [0.]), budgets=[0., 2.0**-24], witnesses=[[0.]],
        decision=tie_rule, execution=near_spec))

    native = []
    for dtype, large in (("float64", 2.0**54), ("float32", 2.0**25),
                         ("float16", 2.0**12), ("bfloat16", 2.0**9)):
        old = affine([[large, -large, 1.], [0., 0., 0.]], [0., .5])
        new = old.replace(**{"layers.0.weight": [[large, 1., -large], [0., 0., 0.]]})
        for backend in ("numpy", "torch"):
            for order in ("left_to_right", "right_to_left", "matmul"):
                native.append(_native_case(f"cancellation/{backend}/{dtype}/{order}", old, new,
                    [[1., 1., 1.]], ExecutionSpec(backend=backend, dtype=dtype, accumulation=order)))
    native.extend([
        _native_case("near-tie-endpoint-projection", near, near_candidate, [[0.]], near_spec, tie_rule),
        _native_case("no-op-native-preserved-real-correspondence-fails", near, near, [[0.]], near_spec, tie_rule),
        _native_case("forbidden-endpoint-projection", near, near, [[0.]],
                     ExecutionSpec(dtype="float32"), tie_rule),
    ])
    input_base = affine([[1.], [0.]], [0., 0.])
    overflow = affine([[65504., 65504.], [0., 0.]], [0., 0.])
    tanh = affine([[1.], [0.]], [0., 0.], "tanh")
    native.extend([
        _native_case("input-projection-only", input_base, input_base, [[1 + 2.0**-25]],
                     ExecutionSpec(dtype="float32")),
        _native_case("finite-endpoints-intermediate-overflow", overflow, overflow, [[2., 2.]],
                     ExecutionSpec(dtype="float16", accumulation="left_to_right")),
        _native_case("tanh-native-only", tanh, tanh, [[1.]], ExecutionSpec()),
        _native_case("formal-native-lane-unsupported", input_base, input_base, [[1.]],
                     ExecutionSpec(error_model="formal")),
        _native_case("unknown-operator-model", unknown, unknown, [[1.]], ExecutionSpec()),
        _native_case("optional-cuda-scope", input_base, input_base, [[1.]],
                     ExecutionSpec(backend="torch", dtype="float32", device="cuda")),
    ])
    return {"schema": "wsp-admission-runtime-protocol/1", "research_date": "2026-11-08",
            "pre_result_scope": "Local frozen designed inputs; not an external preregistration",
            "relation_to_v2": "Adjacent supplement to study_v2.run_ablation_case; historical results are unchanged",
            "falsifiers": ["A substituted record verifies against the original trusted request hash",
                           "An observed native decision violation is labeled preserved",
                           "An unsupported kernel/activation is reported as formally verified"],
            "admission_cases": admission, "native_cases": native,
            "substitutions": list(SUBSTITUTIONS),
            "inference_scope": "Designed boundary examples only; no traffic frequency, business impact or population inference"}


def substitute_request(request: AdmissionRequest, name: str) -> AdmissionRequest:
    """Fully rehash a changed claim; it must still fail the original trust anchor."""
    data = request.to_dict()
    if name in ("parent", "candidate"):
        data[name]["tensors"]["layers.0.bias"][0] += .25
    elif name == "program":
        data["program"]["edits"][0]["after"] += .125
    elif name == "domain":
        data["domain"]["lower"][0] = -.25
    elif name == "adapter_version":
        data["adapter"]["version"] = "substituted/2"
    elif name == "adapter_configuration":
        data["adapter"]["implementation_sha256"] = "0"*64
    elif name in ("input_order", "output_order"):
        data["interface"]["input_names" if name == "input_order" else "output_names"].reverse()
    elif name == "tensor_order":
        data["interface"]["tensor_coordinate_order"] = "F"
    elif name == "bias_convention":
        data["interface"]["bias_convention"] = "bias-before-dot/1"
    elif name == "witnesses":
        data["witness_inputs"][0][0] = .75
    elif name == "feature_names":
        data["feature_retention"]["feature_names"].reverse()
    elif name == "feature_anchors":
        data["feature_retention"]["anchors"][0][0] = .75
    elif name == "feature_head":
        data["feature_retention"]["head"]["feature_id"] = "different-extractor/1"
    elif name == "feature_bias":
        data["feature_retention"]["head"]["bias"] = None
    elif name == "decision_rule":
        data["decision_rule"] = DecisionRule("argmax", "first_index").to_dict()
    elif name == "score_budgets":
        data["max_output_drift"][0] += .25
    elif name == "structural_policy":
        data["structural_policy"]["max_edits"] = 0
    elif name == "task":
        data["task"]["task_id"] = "different-task"
    elif name == "backend_environment":
        data["execution"]["environment"]["numpy_version"] = "substituted"
    else:
        raise ValueError("Unknown deliberate admission substitution")
    for key in ("parent", "candidate", "program", "task"):
        data[key + "_sha256"] = digest(data[key])
    data["witness_sha256"] = digest(data["witness_inputs"])
    data["sha256"] = digest({k: v for k, v in data.items() if k != "sha256"})
    return AdmissionRequest.from_dict(data)


def _execute(protocol: dict[str, Any]) -> dict[str, Any]:
    admissions = []
    for case in protocol["admission_cases"]:
        request = bind_case(case)
        record = admit(request)
        admissions.append({"case": case["case"], "request_sha256": request.sha256,
                           "record": record.to_dict(),
                           "replay_verified": verify_admission(record, request.sha256)})
    original = bind_case(protocol["admission_cases"][0])
    attacks = []
    for name in protocol["substitutions"]:
        changed = substitute_request(original, name)
        record = admit(changed)
        attacks.append({"substitution": name, "original_binding_sha256": original.sha256,
                       "new_binding_sha256": changed.sha256, "record": record.to_dict(),
                       "verifies_under_original_trust_anchor": verify_admission(record, original.sha256),
                       "new_claim_replay_verified": verify_admission(record, changed.sha256)})
    native = []
    for case in protocol["native_cases"]:
        receipt = native_execution_receipt(TensorArtifact.from_dict(case["parent"]),
            TensorArtifact.from_dict(case["candidate"]), case["witnesses"],
            ExecutionSpec.from_dict(case["execution"]),
            decision=DecisionRule.from_dict(case["decision"]), binding_sha256=digest(case))
        native.append({"case": case["case"], "case_sha256": digest(case), "receipt": receipt})
    return {"schema": "wsp-admission-runtime-results/1", "protocol_sha256": digest(protocol),
            "admissions": admissions, "substitution_attacks": attacks, "native_cases": native,
            "claim": "Frozen request correspondence and measured witness decisions; no universal kernel, extractor or business proof"}


def run(out: str | Path) -> dict[str, Any]:
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=False)
    protocol = boundary_protocol()
    files = {"protocol.json": write_json(directory / "protocol.json", protocol)}
    sources = {}
    for relative in SOURCE_FILES:
        raw = (ROOT / relative).read_bytes()
        target = directory / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
        sources[relative] = file_hash(target)
        files["sources/" + relative] = sources[relative]
    files["sources.json"] = write_json(directory / "sources.json", sources)
    results = _execute(protocol)
    if not all(row["replay_verified"] for row in results["admissions"]):
        raise ValueError("An admission failed independent replay")
    if any(row["verifies_under_original_trust_anchor"] or not row["new_claim_replay_verified"]
           for row in results["substitution_attacks"]):
        raise ValueError("A substituted claim crossed the trusted request boundary or failed replay")
    files["results.json"] = write_json(directory / "results.json", results)
    manifest = {"schema": "wsp-admission-runtime-manifest/1", "files": files,
                "scope": "Local file identities; authentication requires retaining this manifest hash externally"}
    write_json(directory / "manifest.json", manifest)
    verify(directory)
    return results


def verify(directory: str | Path) -> bool:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest["schema"] != "wsp-admission-runtime-manifest/1":
        raise ValueError("Unknown admission/runtime manifest")
    verify_manifest(directory, manifest["files"])
    for relative, expected in read_json(directory / "sources.json").items():
        if relative not in SOURCE_FILES or file_hash(ROOT / relative) != expected:
            raise ValueError("Current execution sources differ from the frozen boundary study")
    protocol = read_json(directory / "protocol.json")
    if canonical_bytes(protocol) != canonical_bytes(boundary_protocol()):
        raise ValueError("Frozen boundary assignments differ from the declared source protocol")
    if canonical_bytes(_execute(protocol)) != canonical_bytes(read_json(directory / "results.json")):
        raise ValueError("Independent native/admission recomputation differs from recorded evidence")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "verify"))
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    result = run(arguments.directory) if arguments.action == "run" else verify(arguments.directory)
    print("Frozen admission/runtime evidence verified" if result else "Verification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
