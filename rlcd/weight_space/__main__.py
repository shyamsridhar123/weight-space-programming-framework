"""Data-only command line interface: python -m rlcd.weight_space --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifact import StructuralPolicy, TensorArtifact, WeightArtifact
from .compiler import compile_readout
from .contracts import Contract, check_contract, certify_drift, verify_certificate
from .evidence import freeze, read_json, verify_freeze, write_json
from .model import InputBox, Network, canonical_bytes
from .program import Program, apply_program, compose, difference, interpolate, rollback
from .semantics import as_relu_network, default_registry, to_tensor_artifact
from .structured import StructuredProgram, lower_structured, verify_lowering
from .features import ReadoutHead, compile_feature_edit, certify_feature_retention, verify_feature_retention


def load_artifact(path: str | Path) -> WeightArtifact:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("An artifact must be a JSON object")
    if value.get("schema") == "wsp-network/1":
        return Network.from_dict(value)
    if value.get("schema") == "wsp-tensors/1":
        return TensorArtifact.from_dict(value)
    raise ValueError("Unrecognized artifact schema; architecture adapters must be explicit")


def _network(artifact: WeightArtifact) -> Network:
    if isinstance(artifact, Network):
        return artifact
    if isinstance(artifact, TensorArtifact):
        return as_relu_network(artifact)
    raise ValueError("This command requires the supported two-layer ReLU semantic adapter")


def _behavioral_check(base: WeightArtifact, program: Program, contract: Contract) -> dict[str, Any]:
    candidate = apply_program(base, program)
    semantic_base, semantic_candidate = _network(base), _network(candidate)
    semantic_program = difference(semantic_base, semantic_candidate, program.intent)
    return {"base_artifact_sha256": base.sha256, "candidate_artifact_sha256": candidate.sha256,
            "artifact_program_sha256": program.sha256,
            "normalized_semantic_admission": check_contract(semantic_base, semantic_program, contract)}


def _apply(args: argparse.Namespace) -> dict[str, Any]:
    base, program = load_artifact(args.base), Program.from_dict(read_json(args.program))
    candidate = apply_program(base, program)
    result: dict[str, Any] = {
        "schema": "wsp-application/1", "base_sha256": base.sha256,
        "program_sha256": program.sha256, "candidate_sha256": candidate.sha256,
        "structural_status": "identity-and-coordinates-checked", "behavioral_status": "not_checked",
    }
    if args.policy:
        result["structural_admission"] = StructuralPolicy(**read_json(args.policy)).check(base, program)
        if not result["structural_admission"]["structural_accepted"]:
            raise ValueError(f"Structural policy rejected the program: {result['structural_admission']['issues']}")
        result["structural_status"] = "policy-accepted"
    if args.contract:
        result["behavioral_admission"] = _behavioral_check(base, program, Contract.from_dict(read_json(args.contract)))
        admission = result["behavioral_admission"]["normalized_semantic_admission"]
        if not admission["accepted"]:
            raise ValueError(f"Behavioral contract rejected the program: {admission['issues']}")
        result["behavioral_status"] = "real-score-drift-contract-accepted"
    output = Path(args.out)
    receipt = output.with_name(output.name + ".receipt.json")
    if output.exists() or receipt.exists():
        raise FileExistsError("Output or application receipt already exists")
    write_json(output, candidate.to_dict())
    write_json(receipt, result)
    return result


def _compile(args: argparse.Namespace) -> dict[str, Any]:
    base, spec = load_artifact(args.base), read_json(args.spec)
    allowed = {"target_inputs", "desired_delta", "retain_inputs", "retention_weight", "ridge",
               "editable_hidden", "edit_bias", "intent"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("Unknown compilation specification fields")
    compilation = compile_readout(_network(base), **spec)
    semantic_candidate = apply_program(_network(base), compilation.program)
    candidate = (semantic_candidate if isinstance(base, Network)
                 else to_tensor_artifact(semantic_candidate))
    program = difference(base, candidate, compilation.program.intent)
    result = {"program_sha256": program.sha256, "diagnostics": compilation.diagnostics,
              "behavioral_status": "not_checked", "admission": None}
    if args.contract:
        result["admission"] = _behavioral_check(base, program, Contract.from_dict(read_json(args.contract)))
        accepted = result["admission"]["normalized_semantic_admission"]["accepted"]
        result["behavioral_status"] = "real-score-drift-contract-accepted" if accepted else "rejected"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "program.json", program.to_dict())
    write_json(out / "compilation.json", result)
    # A rejected proposal is retained for review; applying it is a separate command.
    return result


def _certificate(args: argparse.Namespace, verify: bool = False) -> dict[str, Any]:
    base, candidate = load_artifact(args.base), load_artifact(args.candidate)
    if not base.compatible(candidate):
        raise ValueError("Certificate requires matching artifact architecture and interface")
    if verify:
        supplied = read_json(args.certificate)
        if isinstance(base, Network):
            valid = verify_certificate(base, _network(candidate), supplied)
        else:
            try:
                box = InputBox(**supplied["certificate"]["domain"])
                valid = canonical_bytes(supplied) == canonical_bytes(default_registry().certify(base, candidate, box))
            except (KeyError, ValueError, TypeError):
                valid = False
        if not valid:
            raise ValueError("Certificate failed exact recomputation or identity checking")
        return {"valid": True, "scope": "real arithmetic; Python checker and floating execution are not verified"}
    box = InputBox(**read_json(args.box))
    receipt = (certify_drift(base, _network(candidate), box) if isinstance(base, Network)
               else default_registry().certify(base, candidate, box))
    write_json(args.out, receipt)
    return receipt


def _lower(args: argparse.Namespace, verify: bool = False) -> dict[str, Any]:
    base = load_artifact(args.base)
    proposal = StructuredProgram.from_dict(read_json(args.proposal))
    if verify:
        if not verify_lowering(base, proposal, read_json(args.receipt)):
            raise ValueError("Structured lowering receipt failed exact recomputation")
        return {"valid": True, "behavioral_status": "not_checked"}
    dtypes = None if args.native_dtypes is None else read_json(args.native_dtypes)
    lowered = lower_structured(base, proposal, native_dtypes=dtypes, cast=args.cast)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "program.json", lowered.program.to_dict())
    write_json(out / "lowering.json", lowered.receipt)
    return lowered.receipt


def _features(args: argparse.Namespace) -> dict[str, Any]:
    base, spec = load_artifact(args.base), read_json(args.spec)
    if not isinstance(spec, dict) or "head" not in spec:
        raise ValueError("A feature specification must include a declared head")
    head = ReadoutHead(**spec["head"])
    if args.command == "compile-features":
        allowed = {"head", "target_features", "desired_delta", "retain_features", "retention",
                   "retention_weight", "ridge", "edit_bias", "nullspace_rtol", "require_exact_scores",
                   "native_dtypes", "cast", "intent"}
        if set(spec) - allowed:
            raise ValueError("Unknown frozen-feature compilation fields")
        result = compile_feature_edit(base, head, **{k: v for k, v in spec.items() if k != "head"})
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "program.json", result.program.to_dict())
        write_json(out / "proposal.json", result.proposal.to_dict())
        write_json(out / "compilation.json", result.diagnostics)
        if result.retention_certificate is not None:
            write_json(out / "retention.json", result.retention_certificate)
        return result.diagnostics
    if set(spec) != {"head", "anchors"}:
        raise ValueError("A retention certificate specification contains head and anchors")
    candidate = load_artifact(args.candidate)
    if args.command == "verify-retention":
        if not verify_feature_retention(base, candidate, head, spec["anchors"], read_json(args.receipt)):
            raise ValueError("Feature retention receipt failed exact recomputation")
        return {"valid": True, "scope": "real head on supplied features; extractor and float execution unverified"}
    receipt = certify_feature_retention(base, candidate, head, spec["anchors"])
    write_json(args.out, receipt)
    return receipt


def _governance(args: argparse.Namespace) -> dict[str, Any]:
    from . import governance

    operation = args.governance_command
    pins = ({} if operation == "request" else {
        "expected_request_sha256": args.expected_request_sha256,
        "expected_head_sha256": args.expected_head_sha256})
    if operation == "export":
        return governance.export_bundle(args.bundle, args.out, **pins)
    if operation in ("request", "check"):
        context = {name: governance.read_json(getattr(args, name))
                   for name in ("base", "program", "candidate", "constraints", "scope")}
    if operation == "request":
        result = governance.create_request(
            **context, purpose=args.purpose, evidence=governance.read_json(args.evidence),
            requester=args.requester, approvers=args.approver)
    elif operation == "decide":
        result = governance.decide(governance.read_json(args.bundle), **pins,
                                   actor=args.actor, action=args.action, reason=args.reason)
    else:
        return governance.check_eligibility(governance.read_json(args.bundle), **pins, **context)
    write_json(args.out, result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Generic tensor programs with explicit architecture-specific behavioral checks.")
    sub = root.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Inspect shapes, architecture, interface, and content identity")
    inspect.add_argument("artifact")
    diff = sub.add_parser("diff", help="Create an exact-parent atomic edit program")
    diff.add_argument("base"); diff.add_argument("candidate"); diff.add_argument("--intent", required=True)
    diff.add_argument("--out", required=True)
    apply = sub.add_parser("apply", help="Apply to a new artifact; unchecked behavior is explicitly reported")
    apply.add_argument("base"); apply.add_argument("program"); apply.add_argument("--out", required=True)
    apply.add_argument("--policy", help="Generic structural policy JSON")
    apply.add_argument("--contract", help="Supported ReLU behavioral contract JSON")
    chain = sub.add_parser("compose", help="Compose a sequence of programs with exact intermediate parents")
    chain.add_argument("base"); chain.add_argument("programs", nargs="+")
    chain.add_argument("--intent", default="Sequential composition"); chain.add_argument("--out", required=True)
    undo = sub.add_parser("rollback", help="Create the exact inverse program for an applied edit")
    undo.add_argument("base"); undo.add_argument("program"); undo.add_argument("--out", required=True)
    blend = sub.add_parser("interpolate", help="Coordinate interpolation; no behavioral claim")
    blend.add_argument("base"); blend.add_argument("candidate"); blend.add_argument("--alpha", type=float, required=True)
    blend.add_argument("--out", required=True)
    compile_cmd = sub.add_parser("compile", help="Compile examples into a frozen-readout ReLU edit")
    compile_cmd.add_argument("base"); compile_cmd.add_argument("spec"); compile_cmd.add_argument("--out", required=True)
    compile_cmd.add_argument("--contract")
    for name in ("lower", "verify-lowering"):
        command = sub.add_parser(name, help="Lower/check data-only tensor blocks and low-rank proposals")
        command.add_argument("base"); command.add_argument("proposal")
        if name == "lower":
            command.add_argument("--native-dtypes", help="Complete tensor-name to native-format JSON map")
            command.add_argument("--cast", choices=("reject", "round"), default="reject")
            command.add_argument("--out", required=True)
        else:
            command.add_argument("receipt")
    for name in ("compile-features", "certify-retention", "verify-retention"):
        command = sub.add_parser(name, help="Compile/check a readout over explicitly supplied frozen features")
        command.add_argument("base")
        if name != "compile-features":
            command.add_argument("candidate")
        command.add_argument("spec")
        if name == "verify-retention":
            command.add_argument("receipt")
        else:
            command.add_argument("--out", required=True)
    for name in ("certify", "verify"):
        command = sub.add_parser(name, help="Compute/check an exact rational certificate for a registered backend")
        command.add_argument("base"); command.add_argument("candidate")
        if name == "certify":
            command.add_argument("--box", required=True); command.add_argument("--out", required=True)
        else:
            command.add_argument("certificate")
    demo = sub.add_parser("demo", help="Run a generic convolution/Tanh example and a proved ReLU specialization")
    demo.add_argument("--out", required=True)
    governance = sub.add_parser("governance", help="Optional local review; no production or scientific authority")
    governance_sub = governance.add_subparsers(dest="governance_command", required=True)
    for name in ("request", "decide", "check", "export"):
        command = governance_sub.add_parser(name)
        if name != "request":
            command.add_argument("bundle")
            command.add_argument("--expected-request-sha256", required=True)
            command.add_argument("--expected-head-sha256", required=True)
        if name in ("request", "check"):
            for artifact in ("base", "program", "candidate"):
                command.add_argument(artifact)
            command.add_argument("--constraints", required=True)
            command.add_argument("--scope", required=True)
        if name == "request":
            command.add_argument("--purpose", required=True)
            command.add_argument("--evidence", required=True)
            command.add_argument("--requester", required=True)
            command.add_argument("--approver", action="append", required=True)
        if name == "decide":
            command.add_argument("--actor", required=True)
            command.add_argument("--action", choices=("approve", "reject", "revoke"), required=True)
            command.add_argument("--reason", required=True)
        if name != "check":
            command.add_argument("--out", required=True)
    for name in ("freeze", "verify-freeze", "synthetic", "rts-prepare", "rts-select", "rts-summarize"):
        command = sub.add_parser(name)
        command.add_argument("--protocol", required=True)
        if name != "freeze":
            command.add_argument("--freeze", required=True, help="Path to the pre-result freeze receipt.json")
        if name != "verify-freeze":
            command.add_argument("--out", required=True)
        if name == "rts-prepare": command.add_argument("--export", required=True)
        if name in ("rts-select", "rts-summarize"): command.add_argument("--results", required=True)
        if name == "rts-select": command.add_argument("--programs", required=True)
        if name == "rts-summarize": command.add_argument("--selection", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        if args.command == "inspect":
            artifact = load_artifact(args.artifact)
            result = {"schema": artifact.to_dict()["schema"], "sha256": artifact.sha256,
                      "architecture": getattr(artifact, "architecture", "dense-relu-dense/1"),
                      "parameter_count": artifact.parameter_count,
                      "shapes": {k: list(v.shape) for k, v in artifact.tensors.items()},
                      "behavioral_status": "not_checked"}
        elif args.command in ("diff", "compose", "rollback", "interpolate"):
            base = load_artifact(args.base)
            if args.command == "diff":
                program = difference(base, load_artifact(args.candidate), args.intent)
            elif args.command == "compose":
                program = compose(base, [Program.from_dict(read_json(p)) for p in args.programs], args.intent)
            elif args.command == "rollback":
                program = rollback(base, Program.from_dict(read_json(args.program)))
            else:
                program = interpolate(base, load_artifact(args.candidate), args.alpha)
            write_json(args.out, program.to_dict())
            result = {"program_sha256": program.sha256, "edits": len(program.edits), "behavioral_status": "not_checked"}
        elif args.command == "apply": result = _apply(args)
        elif args.command == "compile": result = _compile(args)
        elif args.command in ("lower", "verify-lowering"): result = _lower(args, args.command == "verify-lowering")
        elif args.command in ("compile-features", "certify-retention", "verify-retention"): result = _features(args)
        elif args.command in ("certify", "verify"): result = _certificate(args, args.command == "verify")
        elif args.command == "governance": result = _governance(args)
        elif args.command == "demo":
            from .demo import run_demo
            result = run_demo(args.out)
        elif args.command == "freeze": result = freeze(args.protocol, args.out)
        elif args.command == "verify-freeze":
            result = {"valid": True, "freeze_sha256": verify_freeze(args.protocol, args.freeze)["sha256"]}
        elif args.command == "synthetic":
            from .experiments import run_synthetic
            study = run_synthetic(args.protocol, args.freeze, args.out)
            result = {key: study[key] for key in ("completed_networks", "completed_pairs", "complete", "controls_pass", "H1", "H2")}
        elif args.command == "rts-prepare":
            from .rts_study import make_candidates
            result = make_candidates(args.export, args.protocol, args.freeze, args.out)
        elif args.command == "rts-select":
            from .rts_study import select_candidate
            result = select_candidate(args.results, args.programs, args.protocol, args.freeze, args.out)
        else:
            from .rts_study import summarize_test
            result = summarize_test(args.results, args.selection, args.protocol, args.freeze, args.out)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, FloatingPointError) as error:
        command_parser.exit(2, f"weight-space: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
