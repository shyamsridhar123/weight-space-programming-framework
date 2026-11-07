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
    for name in ("certify", "verify"):
        command = sub.add_parser(name, help="Compute/check an exact rational ReLU score-drift certificate")
        command.add_argument("base"); command.add_argument("candidate")
        if name == "certify":
            command.add_argument("--box", required=True); command.add_argument("--out", required=True)
        else:
            command.add_argument("certificate")
    demo = sub.add_parser("demo", help="Run a generic convolution/Tanh example and a proved ReLU specialization")
    demo.add_argument("--out", required=True)
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
        elif args.command in ("certify", "verify"): result = _certificate(args, args.command == "verify")
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
    except (OSError, ValueError, TypeError, KeyError) as error:
        command_parser.exit(2, f"weight-space: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
