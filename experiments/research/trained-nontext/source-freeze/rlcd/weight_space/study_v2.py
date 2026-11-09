"""Frozen, non-overwriting algorithmic controls for the generic framework.

This module measures deterministic compilers, not language-model programming.
It keeps shared-world repetitions clustered and records every fixed candidate.
The protocol is locally hash-frozen; it is not an external preregistration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path
import json
import math
import re
import shutil
from typing import Any

import numpy as np

from .artifact import StructuralPolicy, TensorArtifact
from .benchmarks import (FAMILIES, build_model, features, forward, gradient_program,
                         head_for, make_world, metrics, native_forward, project_model,
                         random_matched_program)
from .contracts import InputBox
from .evidence import ROOT, environment, file_hash, read_json, utc_now, write_json
from .features import _rms, compile_feature_edit, verify_feature_retention
from .model import canonical_bytes, digest
from .program import Program, apply_program, compose, difference, rollback
from .sequential import certify_sequential_drift, verify_sequential_certificate
from .structured import verify_lowering
from .workflow import AccessPolicy, BudgetLimits, ProgrammerSession, SplitSpec, TaskSpec, verify_trace

ARMS = ("no_op", "unconstrained", "soft_retention", "nullspace", "matched_random", "gradient")
SELECTION_ORDER = ["minimum validation target_nrmse", "minimum delta_l2", "lexicographic arm name"]
SEQUENCE_DECLARATIONS = {
    "methods": ["unconstrained", "soft_retention"],
    "regions": ["x0 < -0.5", "abs(x0) < 0.2", "x0 > 0.5"],
    "target": "original checkpoint outputs plus that region's shift",
    "retention": "current checkpoint outputs on the other two public regions",
    "selection": "none; all three edits and both methods fixed before any heldout evaluation",
}
BOOTSTRAP_DECLARATIONS = {
    "unit": "independent world; average paired family effects within each world",
    "interval": "exploratory percentile 95%; no multiplicity correction",
    "incomplete_world_policy": "omit worlds missing any main arm or family; retain available main outcomes descriptively; ablation failures do not remove main outcomes",
}
ACCOUNTING_DECLARATIONS = {
    "proposer": "deterministic algorithm; no language model, API calls, or provider tokens",
    "allowed": ["exact weights", "public feature activations", "least-squares fitting", "validation feedback for six fixed candidates"],
    "session_counters": ["tool_calls", "weight_values", "execution_calls", "execution_examples", "feedback_calls",
                         "feedback_examples", "proposals", "input_tokens", "output_tokens"],
    "heldout": "one post-seal evaluator callback for all predeclared arms; actual model-example executions reported separately",
    "limits": "metadata and every returned candidate reveal full captured weights; those scalar values are charged",
    "training": "random frozen feature extractors; only output heads fitted by ridge on the training split",
    "split_lineage": "independent random streams within world; family checkpoints share teacher and input splits; no claim that families are independent tasks",
    "external_preregistration": False,
}


def _fields(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"Unknown {label} fields")


def _number(value: Any, *, minimum: float = 0., positive: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > minimum if positive else value >= minimum))


def _shift(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 3
            and all(type(x) in (int, float) and math.isfinite(x) for x in value)
            and any(value) and _rms(np.asarray(value, dtype=np.float64)) > 0.)


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    expected = {"schema", "name", "study_kind", "seeds", "families", "sizes", "training_ridge",
                "shift", "compiler", "selection", "precision", "feasibility", "sequence", "auxiliary",
                "bootstrap", "accounting", "claims", "source_review"}
    if not isinstance(value, dict) or set(value) != expected or value["schema"] != "wsp-generic-study/2":
        raise ValueError("Unknown generic study protocol schema or fields")
    seeds = value["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Use unique nonnegative world seeds")
    if (not isinstance(value["families"], list) or not value["families"]
            or any(not isinstance(family, str) or family not in FAMILIES for family in value["families"])
            or len(set(value["families"])) != len(value["families"])):
        raise ValueError("Unknown or duplicate family")
    _fields(value["sizes"], {"train", "public", "validation", "test", "sequence"}, "split size")
    if any(type(n) is not int or n < 1 for n in value["sizes"].values()):
        raise ValueError("Every split size must be a positive integer")
    if not _shift(value["shift"]):
        raise ValueError("The study needs a finite nonzero three-score shift")
    if not _number(value["training_ridge"], positive=True):
        raise ValueError("Training ridge must be finite and positive")
    compiler = value["compiler"]
    _fields(compiler, {"ridge", "retention_weight", "nullspace_rtol", "nullspace_anchors", "gradient_steps"}, "compiler")
    if (not _number(compiler["ridge"], positive=True) or not _number(compiler["retention_weight"])
            or not _number(compiler["nullspace_rtol"], positive=True) or compiler["nullspace_rtol"] >= 1
            or type(compiler["gradient_steps"]) is not int or compiler["gradient_steps"] < 1
            or type(compiler["nullspace_anchors"]) is not int
            or not 1 <= compiler["nullspace_anchors"] <= value["sizes"]["public"]):
        raise ValueError("Invalid fixed compiler settings")
    if (not isinstance(value["precision"], list) or not value["precision"]
            or any(not isinstance(dtype, str) or dtype not in ("float64", "float32", "float16", "bfloat16") for dtype in value["precision"])
            or len(set(value["precision"])) != len(value["precision"])):
        raise ValueError("Unknown or duplicate native format")
    if (not isinstance(value["feasibility"], list) or not value["feasibility"]
            or any(type(n) is not int or not 1 <= n <= value["sizes"]["public"] for n in value["feasibility"])
            or len(set(value["feasibility"])) != len(value["feasibility"])):
        raise ValueError("Feasibility anchors must fit in the public split")
    if (value["auxiliary"] is not True or value["study_kind"] != "deterministic_algorithmic_controls"
            or not isinstance(value["name"], str) or not value["name"].strip()):
        raise ValueError("This runner supports the declared algorithmic controls only")
    selection = value["selection"]
    _fields(selection, {"max_retain_nrmse", "max_target_nrmse", "order", "fallback", "structural_policy"}, "selection")
    policy = selection["structural_policy"]
    _fields(policy, {"write_set", "max_edits", "max_parameter_abs", "max_change_abs"}, "structural policy")
    if (not _number(selection["max_retain_nrmse"]) or not _number(selection["max_target_nrmse"])
            or selection["order"] != SELECTION_ORDER or selection["fallback"] != "no_op"
            or policy["write_set"] != "declared head and bias" or policy["max_edits"] != "parent parameter count"
            or not _number(policy["max_parameter_abs"]) or not _number(policy["max_change_abs"])):
        raise ValueError("Unsupported selection or structural policy")
    sequence = value["sequence"]
    _fields(sequence, set(SEQUENCE_DECLARATIONS) | {"shifts"}, "sequence")
    if any(sequence[key] != declared for key, declared in SEQUENCE_DECLARATIONS.items()):
        raise ValueError("Unsupported repeated-edit semantics")
    if not isinstance(sequence["shifts"], list) or len(sequence["shifts"]) != 3 or not all(_shift(x) for x in sequence["shifts"]):
        raise ValueError("Repeated edits require three finite nonzero score shifts")
    bootstrap = value["bootstrap"]
    _fields(bootstrap, set(BOOTSTRAP_DECLARATIONS) | {"draws", "seed"}, "bootstrap")
    if (type(bootstrap["draws"]) is not int or bootstrap["draws"] < 100
            or type(bootstrap["seed"]) is not int or bootstrap["seed"] < 0
            or any(bootstrap[key] != declared for key, declared in BOOTSTRAP_DECLARATIONS.items())):
        raise ValueError("Invalid cluster bootstrap configuration")
    if canonical_bytes(value["accounting"]) != canonical_bytes(ACCOUNTING_DECLARATIONS):
        raise ValueError("Unsupported execution and accounting declarations")
    _fields(value["claims"], {"primary_descriptive_contrasts", "controls", "not_established"}, "claim annotations")
    if any(not isinstance(items, list) or not items or any(not isinstance(x, str) or not x.strip() for x in items)
           for items in value["claims"].values()):
        raise ValueError("Claim annotations must be nonempty lists of text")
    if value["source_review"] != "docs/weight-space/v2/source-review.md":
        raise ValueError("This protocol requires the versioned source review")
    # Canonical serialization rejects nonfinite JSON values in unused metadata too.
    canonical_bytes(value)
    return value


def source_paths(protocol_path: str | Path, protocol: dict[str, Any]) -> list[Path]:
    """Freeze this generic runner's code, not the RTS application or its assets."""
    modules = [p for p in (ROOT / "rlcd/weight_space").glob("*.py")
               if p.name not in {"rts_study.py", "experiments.py", "demo.py"}]
    paths = modules + [Path(protocol_path).resolve(), ROOT / "rlcd/__init__.py", ROOT / "requirements.lock.txt",
                       ROOT / "requirements.txt", ROOT / "pytest.ini", ROOT / protocol["source_review"],
                       ROOT / "artifacts/weight-space/source-review-v2/provenance.json",
                       ROOT / "artifacts/weight-space/source-review-v2/review-evidence.json"]
    paths += [ROOT / "tests" / name for name in (
        "test_weight_space_v2.py", "test_weight_space_workflow.py",
        "test_weight_space_study_v2.py", "test_weight_space_study_tasks_v2.py")]
    for path in paths:
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"A frozen input is absent or outside this repository: {path}")
    return sorted(set(paths))


def freeze(protocol_path: str | Path, out: str | Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    paths = source_paths(protocol_path, protocol)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    for source in paths:
        relative = source.relative_to(ROOT).as_posix()
        snapshot = target / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
        files[relative] = {"sha256": file_hash(snapshot), "bytes": snapshot.stat().st_size}
    payload = {"schema": "wsp-generic-freeze/2", "created_utc": utc_now(),
               "protocol_path": Path(protocol_path).resolve().relative_to(ROOT).as_posix(),
               "protocol_sha256": digest(protocol), "files": files,
               "environment": environment(), "external_preregistration": False,
               "scope": "local pre-result snapshot of generic code, tests, protocol, and source review"}
    receipt = payload | {"sha256": digest(payload)}
    write_json(target / "receipt.json", receipt)
    return receipt


def verify_freeze(protocol_path: str | Path, freeze_path: str | Path, *, current_sources: bool = True) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    receipt = read_json(freeze_path)
    _fields(receipt, {"schema", "created_utc", "protocol_path", "protocol_sha256", "files", "environment",
                      "external_preregistration", "scope", "sha256"}, "freeze receipt")
    payload = {key: value for key, value in receipt.items() if key != "sha256"}
    if receipt.get("schema") != "wsp-generic-freeze/2" or digest(payload) != receipt.get("sha256"):
        raise ValueError("Generic freeze receipt identity mismatch")
    if digest(protocol) != receipt["protocol_sha256"]:
        raise ValueError("Protocol changed after freezing")
    expected_names = {path.relative_to(ROOT).as_posix() for path in source_paths(protocol_path, protocol)}
    protocol_name = Path(protocol_path).resolve().relative_to(ROOT).as_posix()
    if (not isinstance(receipt["files"], dict) or set(receipt["files"]) != expected_names
            or receipt["protocol_path"] != protocol_name or receipt["external_preregistration"] is not False):
        raise ValueError("Frozen source set or protocol path differs from the required inputs")
    root = Path(freeze_path).resolve().parent / "sources"
    actual_names = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("Frozen source tree has missing or unrecorded files")
    for name, info in receipt["files"].items():
        _fields(info, {"sha256", "bytes"}, "frozen source record")
        if (not isinstance(info["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])
                or type(info["bytes"]) is not int or info["bytes"] < 0):
            raise ValueError("Malformed frozen source digest or byte count")
        snapshot = (root / name).resolve()
        if (not snapshot.is_relative_to(root) or not snapshot.is_file() or file_hash(snapshot) != info["sha256"]
                or snapshot.stat().st_size != info["bytes"]):
            raise ValueError(f"Frozen source missing or altered: {name}")
        if current_sources and (not (ROOT / name).resolve().is_relative_to(ROOT)
                                or file_hash(ROOT / name) != info["sha256"]):
            raise ValueError(f"Execution source changed after freezing: {name}")
    if digest(load_protocol(root / protocol_name)) != receipt["protocol_sha256"]:
        raise ValueError("Frozen protocol snapshot does not match its semantic digest")
    return receipt


def _saved_compilation(out: Path, base: TensorArtifact, compilation: Any) -> TensorArtifact:
    out.mkdir(parents=True, exist_ok=False)
    candidate = apply_program(base, compilation.program)
    write_json(out / "proposal.json", compilation.proposal.to_dict())
    write_json(out / "program.json", compilation.program.to_dict())
    write_json(out / "candidate.json", candidate.to_dict())
    write_json(out / "compilation.json", compilation.diagnostics)
    if compilation.retention_certificate is not None:
        write_json(out / "retention.json", compilation.retention_certificate)
    if not verify_lowering(base, compilation.proposal, compilation.diagnostics["lowering"]):
        raise ValueError("Compiler lowering failed independent recomputation")
    return candidate


def _settings(protocol: dict[str, Any]) -> dict[str, Any]:
    compiler = protocol["compiler"]
    return {key: compiler[key] for key in ("ridge", "retention_weight", "nullspace_rtol")} | {"edit_bias": True}


def _split_inputs(world: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(world["splits"][name]["inputs"], dtype=np.float64)


def choose_candidate(records: dict[str, dict[str, Any]], selection: dict[str, Any]) -> str:
    """A fixed validation-only rule; the no-op is an explicit fallback."""
    eligible = [name for name, record in records.items()
                if record["structural_accepted"]
                and record["validation"]["retain_nrmse"] <= selection["max_retain_nrmse"]
                and record["validation"]["target_nrmse"] <= selection["max_target_nrmse"]]
    if not eligible:
        return "no_op"
    return min(eligible, key=lambda name: (records[name]["validation"]["target_nrmse"],
                                           records[name]["validation"]["delta_l2"], name))


def run_main_case(base: TensorArtifact, world: dict[str, Any], family: str,
                  protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    seed, head = world["seed"], head_for(base)
    public_t, public_r = (_split_inputs(world, f"public_{region}") for region in ("target", "retain"))
    validation_t, validation_r = (_split_inputs(world, f"validation_{region}") for region in ("target", "retain"))
    task = TaskSpec(f"world-{seed}/{family}/conditional-edit", "edit",
                    "Add the declared score shift on x0>0.35 while retaining x0<-0.1 behavior",
                    base.architecture, base.sha256,
                    {"shift": protocol["shift"], "regime": "fitting_allowed_algorithmic_control",
                     "head": head.to_dict(), "fixed_arms": list(ARMS)})
    # Feedback sees only examples explicitly submitted through the interface.
    feedback_attempts = {"attempted": 0, "completed": 0}
    def feedback(candidate: Any, examples: list[Any]) -> dict[str, Any]:
        feedback_attempts["attempted"] += 1
        boundary = len(examples)//2
        result = metrics(base, candidate.artifact, examples[:boundary], examples[boundary:], protocol["shift"])
        feedback_attempts["completed"] += 1
        return result
    splits = [SplitSpec(name, record["sample_ids"]) for name, record in world["splits"].items()]
    session = ProgrammerSession(task, base=base,
                                access=AccessPolicy(feedback=True, activations=True),
                                limits=BudgetLimits(tool_calls=50, weight_values=base.parameter_count*8,
                                                    feedback_calls=6, feedback_examples=12*len(validation_t),
                                                    execution_calls=2, execution_examples=len(public_t)+len(public_r),
                                                    proposals=6, input_tokens=0, output_tokens=0),
                                public_feedback=feedback, activations=lambda model, x: features(model, x), splits=splits)
    session.metadata()
    for name in base.tensors:
        session.weights(name)
    target = np.asarray(session.activations(public_t), dtype=np.float64)
    retain = np.asarray(session.activations(public_r), dtype=np.float64)
    desired = np.tile(protocol["shift"], (len(target), 1))
    policy = protocol["selection"]["structural_policy"]
    structural = StructuralPolicy((head.weight, head.bias), base.parameter_count,
                                  policy["max_parameter_abs"], policy["max_change_abs"])
    candidates, records, failures = {}, {}, []
    for arm in ARMS:
        arm_dir = out / arm
        try:
            comp = None
            if arm == "no_op":
                program = difference(base, base, "Fixed unchanged baseline")
                arm_dir.mkdir()
                write_json(arm_dir / "program.json", program.to_dict())
                write_json(arm_dir / "candidate.json", base.to_dict())
            elif arm in ("unconstrained", "soft_retention", "nullspace"):
                mode = {"unconstrained": "none", "soft_retention": "soft", "nullspace": "nullspace"}[arm]
                anchors = retain[:protocol["compiler"]["nullspace_anchors"]] if mode == "nullspace" else retain
                comp = compile_feature_edit(base, head, target, desired, anchors, retention=mode, **_settings(protocol))
                _saved_compilation(arm_dir, base, comp)
                program = comp.program
            else:
                if arm == "matched_random":
                    proposal, lowered, details = random_matched_program(
                        base, candidates["soft_retention"].artifact, seed*100+FAMILIES.index(family))
                else:
                    proposal, lowered, details = gradient_program(
                        base, target, desired, retain, ridge=protocol["compiler"]["ridge"],
                        retention_weight=protocol["compiler"]["retention_weight"],
                        steps=protocol["compiler"]["gradient_steps"])
                arm_dir.mkdir()
                program = lowered.program
                write_json(arm_dir / "proposal.json", proposal.to_dict())
                write_json(arm_dir / "program.json", program.to_dict())
                write_json(arm_dir / "candidate.json", apply_program(base, program).to_dict())
                write_json(arm_dir / "compilation.json", details)
                if not verify_lowering(base, proposal, lowered.receipt):
                    raise ValueError("Baseline lowering failed independent recomputation")
            if comp is not None:
                anchors = retain[:protocol["compiler"]["nullspace_anchors"]] if arm == "nullspace" else retain
                if not verify_feature_retention(base, apply_program(base, program), head, anchors, comp.retention_certificate):
                    raise ValueError("Endpoint retention failed exact recomputation")
            admission = structural.check(base, program)
            write_json(arm_dir / "admission.json", admission)
            candidate = session.propose_program(program)
            validation = session.feedback(candidate, np.vstack((validation_t, validation_r)))
            # Registration is the final successful action. A retained failed
            # attempt must never become eligible or enter the heldout arms.
            candidates[arm] = candidate
            records[arm] = {"status": "completed", "candidate_sha256": candidate.sha256,
                            "artifact_sha256": candidate.artifact.sha256, "program_sha256": program.sha256,
                            "structural_accepted": admission["structural_accepted"], "validation": validation}
        except Exception as error:
            failure = {"arm": arm, "status": "failed", "type": type(error).__name__, "message": str(error)}
            failures.append(failure)
            write_json(out / f"{arm}-failure.json", failure)
    if "no_op" not in candidates:
        raise ValueError("The fixed baseline failed; cannot perform selection")
    selected_arm = choose_candidate(records, protocol["selection"])
    selection = {"schema": "wsp-generic-selection/2", "selected_arm": selected_arm,
                 "candidate_sha256": candidates[selected_arm].sha256, "rule": protocol["selection"],
                 "validation": records, "heldout_observed": False}
    write_json(out / "selection.json", selection)
    session.seal(candidates[selected_arm])
    # Only after selection is saved and sealed does the evaluator consume test inputs.
    def evaluate(selected: Any) -> dict[str, Any]:
        test_t, test_r = (_split_inputs(world, f"test_{region}") for region in ("target", "retain"))
        arms = {arm: metrics(base, candidate.artifact, test_t, test_r, protocol["shift"], teacher=world["teacher"])
                for arm, candidate in candidates.items()}
        return {"arms": arms, "selected_arm": selected_arm,
                "selected_candidate_sha256": selected.sha256, "selected": arms[selected_arm],
                "unique_test_examples": len(test_t)+len(test_r),
                "evaluated_splits": {"test_target": len(test_t), "test_retain": len(test_r)},
                "candidate_evaluations": len(arms), "model_example_executions": 2*len(arms)*(len(test_t)+len(test_r)),
                "accounting_note": "Each arm executes base and candidate on both regions; selected metrics alias its fixed arm"}
    receipt = session.evaluate_heldout(evaluate, evaluator_id="conditional-score-edit/2", split_name="test_target")
    receipt_data = receipt.to_dict()
    write_json(out / "task.json", task.to_dict())
    write_json(out / "session.json", receipt_data)
    if not verify_trace(session.trace) or receipt_data["status"] != "succeeded":
        raise ValueError("Session trace or final evaluation failed")
    bound = None
    if family == "relu_mlp" and "soft_retention" in candidates:
        certificate = certify_sequential_drift(base, candidates["soft_retention"].artifact, InputBox([-1.]*16, [1.]*16))
        if not verify_sequential_certificate(base, candidates["soft_retention"].artifact, certificate):
            raise ValueError("Sequential certificate failed recomputation")
        write_json(out / "soft-retention-domain-certificate.json", certificate)
        bound = {"output_bounds": [float(Fraction(value)) for value in certificate["output_bounds"]],
                 "scope": "real arithmetic on all [-1,1]^16; not an IEEE or task-loss proof"}
    result = {"seed": seed, "family": family, "base_sha256": base.sha256,
              "selected_arm": selected_arm, "arms": receipt_data["result"]["arms"],
              "usage": session.usage.to_dict(), "failures": failures, "real_domain_bound": bound,
              "validation_model_example_executions": 2*feedback_attempts["completed"]*(len(validation_t)+len(validation_r)),
              "validation_execution_count_complete": feedback_attempts["attempted"] == feedback_attempts["completed"],
              "validation_feedback_attempts": feedback_attempts,
              "test_accounting": {key: receipt_data["result"][key] for key in
                                  ("unique_test_examples", "evaluated_splits", "candidate_evaluations", "model_example_executions")}}
    write_json(out / "result.json", result)
    return result


def run_ablation_case(base: TensorArtifact, world: dict[str, Any], protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    public_t, public_r = (_split_inputs(world, f"public_{region}") for region in ("target", "retain"))
    target, retain = features(base, public_t), features(base, public_r)
    desired = np.tile(protocol["shift"], (len(target), 1))
    head, prepared = head_for(base), []
    for count in protocol["feasibility"]:
        compiled = compile_feature_edit(base, head, target, desired, retain[:count],
                                        retention="nullspace", **_settings(protocol))
        candidate = _saved_compilation(out / f"anchors-{count}", base, compiled)
        if not verify_feature_retention(base, candidate, head, retain[:count], compiled.retention_certificate):
            raise ValueError("Feasibility certificate mismatch")
        prepared.append((count, candidate, compiled))
    native_prepared = []
    for dtype in protocol["precision"]:
        parent = project_model(base, dtype)
        parent_head = head_for(parent)
        compiled = compile_feature_edit(parent, parent_head, features(parent, public_t), desired,
                                        features(parent, public_r), retention="soft", **_settings(protocol),
                                        native_dtypes={name: dtype for name in parent.tensors}, cast="round")
        candidate = _saved_compilation(out / dtype, parent, compiled)
        write_json(out / dtype / "parent.json", parent.to_dict())
        if not verify_feature_retention(parent, candidate, parent_head, features(parent, public_r), compiled.retention_certificate):
            raise ValueError("Precision endpoint retention mismatch")
        native_prepared.append((dtype, parent, candidate, compiled))
    # All ablation candidates are fixed before the heldout section below.
    write_json(out / "sealed.json", {"feasibility": {str(n): candidate.sha256 for n, candidate, _ in prepared},
                                     "precision": {dtype: candidate.sha256 for dtype, _, candidate, _ in native_prepared},
                                     "adaptive_selection": False})
    test_t, test_r = (_split_inputs(world, f"test_{region}") for region in ("target", "retain"))
    feasibility = []
    for count, candidate, compiled in prepared:
        certificate = compiled.retention_certificate
        feasibility.append({"anchors": count, "retention_rank": compiled.diagnostics["retention_rank"],
                            "nullspace_dimension": compiled.diagnostics["nullspace_dimension"],
                            "exactly_zero_on_anchors": certificate["exactly_zero_on_supplied_features"],
                            "maximum_anchor_residual": max(float(Fraction(v)) for v in certificate["maximum_residuals_exact"]),
                            "test": metrics(base, candidate, test_t, test_r, protocol["shift"])})
    precision = []
    all_test = np.vstack((test_t, test_r))
    for dtype, parent, candidate, compiled in native_prepared:
        lifted_base, lifted_candidate = forward(parent, all_test), forward(candidate, all_test)
        native_base, native_candidate = native_forward(parent, all_test, dtype), native_forward(candidate, all_test, dtype)
        lifted_drift, native_drift = lifted_candidate-lifted_base, native_candidate-native_base
        native_target = native_drift[:len(test_t)]
        native_retain = native_drift[len(test_t):]
        norm = _rms(np.asarray(protocol["shift"]))
        lowering = compiled.diagnostics["lowering"]
        precision.append({"dtype": dtype, "parent_sha256": parent.sha256, "candidate_sha256": candidate.sha256,
                          "lifted": metrics(parent, candidate, test_t, test_r, protocol["shift"]),
                          "native_target_nrmse": _rms(native_target-protocol["shift"])/norm,
                          "native_retain_nrmse": _rms(native_retain)/norm,
                          "native_base_vs_lifted_max_abs": float(np.max(np.abs(native_base-lifted_base))),
                          "native_candidate_vs_lifted_max_abs": float(np.max(np.abs(native_candidate-lifted_candidate))),
                          "native_drift_vs_lifted_max_abs": float(np.max(np.abs(native_drift-lifted_drift))),
                          "lowering": lowering, "floating_execution_verified": False,
                          "scope": "CPU PyTorch measurement; compiler features evaluated in lifted float64 semantics"})
    result = {"feasibility": feasibility, "precision": precision,
              "split_reuse": "same heldout inputs as main arms; descriptive ablations, not independent trials",
              "feasibility_model_example_executions": 2*len(prepared)*len(all_test),
              "precision_model_example_executions": 6*len(native_prepared)*len(all_test)}
    write_json(out / "result.json", result)
    return result


def run_sequence_case(base: TensorArtifact, world: dict[str, Any], protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    public = [_split_inputs(world, f"sequence_public_{i}") for i in range(3)]
    head = head_for(base)
    phi = [features(base, x) for x in public]
    baseline_outputs = [forward(base, x) for x in public]
    prepared = {}
    for method in protocol["sequence"]["methods"]:
        current, chain, stages = base, [], []
        method_dir = out / method
        method_dir.mkdir()
        for i, shift in enumerate(protocol["sequence"]["shifts"]):
            # Absolute desired behavior is defined against the original model;
            # already induced drift in this region must be included in the edit.
            desired = baseline_outputs[i]+shift-forward(current, public[i])
            retain = np.vstack([phi[j] for j in range(3) if j != i])
            compiled = compile_feature_edit(current, head, phi[i], desired, retain,
                                            retention="none" if method == "unconstrained" else "soft", **_settings(protocol))
            candidate = _saved_compilation(method_dir / f"step-{i+1}", current, compiled)
            if not verify_feature_retention(current, candidate, head, retain, compiled.retention_certificate):
                raise ValueError("Sequential endpoint retention mismatch")
            chain.append(compiled.program)
            stages.append(candidate)
            current = candidate
        combined = compose(base, chain, "Compose three fixed conditional edits")
        inverse = rollback(base, combined)
        write_json(method_dir / "composed.json", combined.to_dict())
        write_json(method_dir / "rollback.json", inverse.to_dict())
        if apply_program(base, combined).sha256 != current.sha256 or apply_program(current, inverse).sha256 != base.sha256:
            raise ValueError("Sequential edit composition or rollback failed")
        prepared[method] = stages
    write_json(out / "sealed.json", {"candidates": {method: [candidate.sha256 for candidate in stages]
                                                     for method, stages in prepared.items()},
                                     "adaptive_selection": False,
                                     "retention_objective": "preserve current outputs in the other two public regions",
                                     "all_chains_fixed_before_heldout": True})
    test = [_split_inputs(world, f"sequence_test_{i}") for i in range(3)]
    base_outputs = [forward(base, x) for x in test]
    methods = {}
    for method, stages in prepared.items():
        outputs = [[forward(candidate, x) for x in test] for candidate in stages]
        stage_metrics = []
        for stage, region_outputs in enumerate(outputs):
            regions = []
            for region, y in enumerate(region_outputs):
                desired_shift = np.asarray(protocol["sequence"]["shifts"][region]) if region <= stage else np.zeros(3)
                normalization = _rms(np.asarray(protocol["sequence"]["shifts"][region]))
                value = {"region": region, "edited": region <= stage,
                         "desired_behavior_nrmse": _rms(y-base_outputs[region]-desired_shift)/normalization}
                if region < stage:
                    value["drift_since_acquisition_nrmse"] = _rms(y-outputs[region][region])/normalization
                regions.append(value)
            stage_metrics.append({"step": stage+1, "regions": regions})
        methods[method] = {"stages": stage_metrics,
                           "final_mean_target_nrmse": float(np.mean([r["desired_behavior_nrmse"] for r in stage_metrics[-1]["regions"]])),
                           "final_prior_region_interference_nrmse": float(np.mean([r["drift_since_acquisition_nrmse"] for r in stage_metrics[-1]["regions"][:-1]])),
                           "composition_exact": True, "rollback_exact": True}
    result = {"methods": methods, "model_example_executions": (1+3*len(prepared))*sum(len(x) for x in test),
              "selection": "none; fixed chains, all heldout inputs consumed after sealing"}
    write_json(out / "result.json", result)
    return result


def cluster_estimate(values_by_world: list[float], draws: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values_by_world, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Cluster estimates need finite per-world observations")
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), (draws, len(array)))].mean(axis=1)
    lower, upper = np.quantile(means, [.025, .975])
    return {"mean": float(array.mean()), "ci95_percentile": [float(lower), float(upper)],
            "independent_worlds": len(array), "per_world_values": array.tolist(),
            "inference": "exploratory world-cluster bootstrap; families within a world averaged first; no multiplicity correction"}


def summarize(cases: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    main_cases = [case for case in cases if isinstance(case.get("main"), dict)]
    grouped = {seed: [case for case in main_cases if case["seed"] == seed] for seed in protocol["seeds"]}
    complete = [seed for seed, rows in grouped.items()
                if len(rows) == len(protocol["families"])
                and {row["family"] for row in rows} == set(protocol["families"])
                and all(set(row["main"]["arms"]) == set(ARMS) and not row["main"]["failures"] for row in rows)]
    contrasts = {
        "soft_collateral_reduction": lambda arms: arms["unconstrained"]["retain_nrmse"]-arms["soft_retention"]["retain_nrmse"],
        "soft_target_error_cost": lambda arms: arms["soft_retention"]["target_nrmse"]-arms["unconstrained"]["target_nrmse"],
        "soft_advantage_over_random_target": lambda arms: arms["matched_random"]["target_nrmse"]-arms["soft_retention"]["target_nrmse"],
        "gradient_minus_soft_target_error": lambda arms: arms["gradient"]["target_nrmse"]-arms["soft_retention"]["target_nrmse"],
    }
    estimates = {}
    for index, (name, fn) in enumerate(contrasts.items()):
        values = [float(np.mean([fn(case["main"]["arms"]) for case in grouped[seed]])) for seed in complete]
        estimates[name] = None if not values else cluster_estimate(values, protocol["bootstrap"]["draws"], protocol["bootstrap"]["seed"]+index)
    by_family = {}
    for family in protocol["families"]:
        rows = [case for case in main_cases if case["family"] == family]
        arm_means = {arm: {metric: float(np.mean([case["main"]["arms"][arm][metric] for case in rows
                                                 if arm in case["main"]["arms"]]))
                          for metric in ("target_nrmse", "retain_nrmse", "target_gain_fraction", "delta_l2")}
                     for arm in ARMS if any(arm in case["main"]["arms"] for case in rows)}
        by_family[family] = {"checkpoints": len(rows), "attempted_checkpoints": sum(case["family"] == family for case in cases),
                             "arm_denominators": {arm: sum(arm in case["main"]["arms"] for case in rows) for arm in ARMS},
                             "arm_means": arm_means,
                             "selected_arms": dict(Counter(case["main"]["selected_arm"] for case in rows))}
    return {"completed_checkpoints": len(main_cases), "attempted_checkpoints": len(cases),
            "fully_completed_checkpoints": sum(case.get("status") == "completed" for case in cases),
            "complete_worlds_for_cluster_inference": len(complete), "complete_world_seeds": complete,
            "omitted_incomplete_worlds": [seed for seed in grouped if seed not in complete],
            "incomplete_world_policy": protocol["bootstrap"]["incomplete_world_policy"],
            "contrasts": estimates, "by_family": by_family,
            "arm_denominators": {arm: sum(arm in case["main"]["arms"] for case in main_cases) for arm in ARMS},
            "selected_arms": dict(Counter(case["main"]["selected_arm"] for case in main_cases)),
            "model_ability_measured": False}


def _auxiliary_complete(result: dict[str, Any]) -> bool:
    return (result.get("errors") == 0 and result.get("cases") == 6
            and len(result.get("case_results", [])) == 6
            and len({case["case_id"] for case in result["case_results"]}) == 6
            and all(case.get("status") == "succeeded" for case in result["case_results"]))


def run(protocol_path: str | Path, freeze_path: str | Path, out: str | Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    frozen = verify_freeze(protocol_path, freeze_path)
    import torch
    from .study_tasks_v2 import run_auxiliary_tasks
    torch.set_num_threads(1)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    write_json(target / "protocol.json", protocol)
    write_json(target / "execution-start.json", {"started_utc": started, "freeze_sha256": frozen["sha256"],
                                                 "environment": environment(), "torch_cpu_threads": 1,
                                                 "provider_calls": 0, "provider_tokens": 0})
    cases, auxiliary, failures = [], [], []
    for seed in protocol["seeds"]:
        world_dir = target / f"world-{seed}"
        world_dir.mkdir()
        world = make_world(seed, protocol["sizes"])
        write_json(world_dir / "world.json", world)
        for family in protocol["families"]:
            case_dir = world_dir / family
            case_dir.mkdir()
            case = {"seed": seed, "family": family, "training": None, "main": None,
                    "ablations": None, "sequence": None, "status": "failed"}
            try:
                base, training = build_model(family, seed, world, ridge=protocol["training_ridge"])
                write_json(case_dir / "base.json", base.to_dict())
                write_json(case_dir / "training.json", training)
                case["training"] = training
            except Exception as error:
                failure = {"seed": seed, "family": family, "component": "training",
                           "type": type(error).__name__, "message": str(error)}
                failures.append(failure)
                write_json(case_dir / "failure.json", failure)
                print(json.dumps(failure), flush=True)
            else:
                components = {
                    "main": lambda: run_main_case(base, world, family, protocol, case_dir / "main"),
                    "ablations": lambda: run_ablation_case(base, world, protocol, case_dir / "ablations"),
                    "sequence": lambda: run_sequence_case(base, world, protocol, case_dir / "sequence"),
                }
                for component, execute in components.items():
                    try:
                        case[component] = execute()
                    except Exception as error:
                        failure = {"seed": seed, "family": family, "component": component,
                                   "type": type(error).__name__, "message": str(error)}
                        failures.append(failure)
                        write_json(case_dir / f"{component}-failure.json", failure)
                        print(json.dumps(failure), flush=True)
                case["status"] = ("completed" if all(case[name] is not None for name in components)
                                  and not case["main"]["failures"] else "partial")
            write_json(case_dir / "case.json", case)
            cases.append(case)
            print(json.dumps({"world": seed, "family": family, "status": case["status"],
                              "selected": None if case["main"] is None else case["main"]["selected_arm"]}), flush=True)
        try:
            aux = run_auxiliary_tasks(seed, world_dir / "auxiliary")
            auxiliary.append({"seed": seed, "result": aux})
            if not _auxiliary_complete(aux):
                raise ValueError("One or more auxiliary cases failed or are absent; see retained auxiliary results")
        except Exception as error:
            failure = {"seed": seed, "family": "auxiliary", "type": type(error).__name__, "message": str(error)}
            failures.append(failure)
            write_json(world_dir / "auxiliary-failure.json", failure)
    try:
        source_unchanged = verify_freeze(protocol_path, freeze_path)["sha256"] == frozen["sha256"]
    except Exception as error:
        source_unchanged = False
        failure = {"component": "source_verification", "type": type(error).__name__, "message": str(error)}
        failures.append(failure)
        write_json(target / "source-verification-failure.json", failure)
    summary = summarize(cases, protocol)
    summary.update({"schema": "wsp-generic-study-results/2", "started_utc": started, "finished_utc": utc_now(),
                    "freeze_sha256": frozen["sha256"], "protocol_sha256": digest(protocol),
                    "sources_unchanged": source_unchanged, "failures": failures,
                    "complete": source_unchanged and len(cases) == len(protocol["seeds"])*len(protocol["families"])
                                and len(auxiliary) == len(protocol["seeds"]) and not failures
                                and all(case["status"] == "completed" for case in cases)
                                and all(_auxiliary_complete(item["result"]) for item in auxiliary),
                    "auxiliary": auxiliary, "cases": cases,
                    "limitations": protocol["claims"]["not_established"]})
    write_json(target / "results.json", summary)
    manifest = {path.relative_to(target).as_posix(): {"sha256": file_hash(path), "bytes": path.stat().st_size}
                for path in sorted(target.rglob("*")) if path.is_file()}
    payload = {"schema": "wsp-generic-study-files/2", "freeze_sha256": frozen["sha256"], "files": manifest}
    write_json(target / "manifest.json", payload | {"sha256": digest(payload)})
    return {key: summary[key] for key in ("complete", "completed_checkpoints", "complete_worlds_for_cluster_inference",
                                         "selected_arms", "contrasts", "failures")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify-freeze", "run"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            if args.out is None:
                raise ValueError("--out is required")
            result = freeze(args.protocol, args.out)
        elif args.command == "verify-freeze":
            if args.freeze is None:
                raise ValueError("--freeze is required")
            result = {"valid": True, "sha256": verify_freeze(args.protocol, args.freeze)["sha256"]}
        else:
            if args.freeze is None or args.out is None:
                raise ValueError("--freeze and --out are required")
            result = run(args.protocol, args.freeze, args.out)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("complete", True) else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"generic-study: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
