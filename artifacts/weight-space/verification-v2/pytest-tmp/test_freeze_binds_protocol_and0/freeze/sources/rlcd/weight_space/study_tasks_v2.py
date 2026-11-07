"""Deterministic auxiliary tasks for the V2 protocol, with no live model calls.

These are controlled interface fixtures, not measurements of language-model
ability. Hidden sample metrics are empirical; no behavioral proof is issued.
"""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import TensorArtifact
from .evidence import file_hash, read_json, write_json
from .model import Network, digest
from .program import difference
from .workflow import AccessPolicy, BudgetLimits, ProgrammerSession, SplitSpec, TaskSpec


HIDDEN_EXAMPLES = 128
RESPONSE_TOLERANCE = 1e-12
DESIGNATED_COEFFICIENT = (1, 7)
CREATION_ARMS = ("zero", "random", "analytic")


def _seed(seed: int) -> int:
    if type(seed) is not int or seed < 0:
        raise ValueError("The world seed must be a nonnegative integer")
    return seed


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([_seed(seed), stream]))


def affine_fixture(seed: int, sign: int) -> TensorArtifact:
    """Paired 16-to-3 affine fixtures, differing only at the designated cell.

The small value is the binary64 literal 0.0004 with the requested sign. All
other values are multiples of 1/8, retained exactly by three-decimal rounding.
"""
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("The coefficient sign must be -1 or 1")
    rng = _rng(seed, 1100)
    weight = rng.integers(-16, 17, (3, 16)) / 8.0
    weight[DESIGNATED_COEFFICIENT] = sign * 0.0004
    bias = rng.integers(-8, 9, 3) / 8.0
    return TensorArtifact("affine/1", {"weight": weight, "bias": bias}, {
        "formula": "x @ weight.T + bias", "input_dimension": 16, "output_dimension": 3,
        "semantic_names": {"inputs": [f"x{i}" for i in range(16)],
                           "outputs": [f"y{i}" for i in range(3)]},
    })


def _dyadic(value: Any, label: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{label} must be an exact rational string")
    try:
        rational = Fraction(value)
        number = float(rational)
    except (ValueError, ZeroDivisionError, OverflowError):
        raise ValueError(f"{label} must be a finite binary64-representable dyadic") from None
    if not math.isfinite(number) or Fraction.from_float(number) != rational:
        raise ValueError(f"{label} must be a finite binary64-representable dyadic")
    return number


def analytic_network(spec: dict[str, Any]) -> Network:
    """Construct two ReLU units from an exact dyadic formula, without fitting."""
    if type(spec) is not dict:
        raise ValueError("An analytic specification must be a data object")
    kind = spec.get("kind")
    if kind == "scaled_absolute" and set(spec) == {"kind", "scale", "center"}:
        scale, center = _dyadic(spec["scale"], "scale"), _dyadic(spec["center"], "center")
        if scale <= 0:
            raise ValueError("The absolute-value scale must be positive")
        return Network([[1.], [-1.]], [-center, center], [[scale, scale]], [0.],
                       input_names=("x",), output_names=("y",))
    if kind == "clip" and set(spec) == {"kind", "low", "high"}:
        low, high = _dyadic(spec["low"], "low"), _dyadic(spec["high"], "high")
        if low >= high:
            raise ValueError("Clip limits must be strictly ordered")
        return Network([[1.], [1.]], [-low, -high], [[1., -1.]], [low],
                       input_names=("x",), output_names=("y",))
    raise ValueError("Unsupported analytic specification or fields")


def _creation_specs(seed: int) -> tuple[dict[str, Any], ...]:
    rng = _rng(seed, 2100)
    scale, center = int(rng.integers(1, 9)), int(rng.integers(-8, 9))
    low, width = int(rng.integers(-8, 1)), int(rng.integers(1, 9))
    return ({"kind": "scaled_absolute", "scale": str(Fraction(scale, 4)),
             "center": str(Fraction(center, 4))},
            {"kind": "clip", "low": str(Fraction(low, 4)), "high": str(Fraction(low + width, 4))})


def _metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("A response must be finite and match the reference shape")
    residual = actual - expected
    maximum = float(np.max(np.abs(residual)))
    return {"examples": len(expected), "output_dimension": expected.shape[1],
            "rmse": float(np.sqrt(np.mean(residual * residual))),
            "mean_absolute_error": float(np.mean(np.abs(residual))),
            "maximum_absolute_error": maximum,
            "exact_array_equality": bool(np.array_equal(expected, actual)),
            "within_tolerance": maximum <= RESPONSE_TOLERANCE,
            "absolute_tolerance": RESPONSE_TOLERANCE}


def _limits(*, proposals: int, weight_values: int = 0, reads: int = 0) -> BudgetLimits:
    return BudgetLimits(tool_calls=proposals + reads, proposals=proposals, weight_values=weight_values,
                        feedback_calls=0, feedback_examples=0, execution_calls=0,
                        execution_examples=0, input_tokens=0, output_tokens=0)


def _persist_session(out: Path, session: ProgrammerSession) -> None:
    write_json(out / "session.json", {"configuration": session.configuration,
               "usage": session.usage.to_dict(), "sealed": session.is_sealed,
               "receipt_status": session.receipt.status if session.receipt is not None else None})
    write_json(out / "candidates.json", [candidate.to_dict() for candidate in session.candidates])
    write_json(out / "trace.json", [event.to_dict() for event in session.trace])
    if session.receipt is not None:
        write_json(out / "receipt.json", session.receipt.to_dict())


def _read_answer(session: ProgrammerSession) -> dict[str, Any]:
    """This fixed reader receives only the declared session interface."""
    weight, bias = session.weights("weight"), session.weights("bias")
    row, column = DESIGNATED_COEFFICIENT
    observed = weight[row][column]
    # Zero includes signed zero: its formatting/sign bit is never used as a hint.
    abstain = observed == 0.0
    return {"schema": "wsp-affine-reading/1", "weight": weight, "bias": bias,
            "designated_coefficient": list(DESIGNATED_COEFFICIENT),
            "sign": None if abstain else (1 if observed > 0 else -1),
            "abstain": abstain,
            "decision_rule": "Abstain at displayed zero; otherwise report the displayed sign"}


def _affine_forward(artifact: TensorArtifact, examples: np.ndarray) -> np.ndarray:
    return examples @ artifact.tensors["weight"].T + artifact.tensors["bias"]


def _run_read_case(seed: int, sign: int, label: str, access: str, mode: str,
                   out: Path) -> dict[str, Any]:
    base = affine_fixture(seed, sign)
    case_id = out.name
    task = TaskSpec(f"auxiliary:{seed}:{case_id}", mode,
                    "Read the 16-to-3 affine response and the sign of weight[1,7]; "
                    "abstain on sign when its displayed value is zero",
                    base.architecture, base.sha256,
                    {"privileged_hints": [], "designated_coefficient": list(DESIGNATED_COEFFICIENT),
                     "semantic_names": "Opaque input/output indices only", "fitting_allowed": False})
    hidden_stream = 1200 if label == "a" else 1201
    session = ProgrammerSession(task, base=base,
        access=AccessPolicy(metadata=False, weight_decimals=None if access == "exact" else 3),
        limits=_limits(proposals=1, weight_values=base.parameter_count, reads=2),
        splits=[SplitSpec("test", [f"{case_id}:{i}" for i in range(HIDDEN_EXAMPLES)])])
    write_json(out / "task.json", task.to_dict())
    write_json(out / "base.json", base.to_dict())
    try:
        session.record_provider_usage(input_tokens=0, output_tokens=0)
        candidate = session.submit_answer(_read_answer(session))
        session.seal(candidate)
        hidden = _rng(seed, hidden_stream).uniform(-2., 2., (HIDDEN_EXAMPLES, 16))
        audit: dict[str, Any] = {
            "seed_entropy": [seed, hidden_stream], "rng": "numpy.PCG64",
            "distribution": "uniform(-2,2), shape=(128,16)", "inputs": hidden.tolist(),
            "input_visibility": "evaluator_only_until_seal", "paired_across_access_conditions": True,
        }

        def evaluate(selected):
            answer = selected.answer
            reconstructed = TensorArtifact(base.architecture,
                {"weight": answer["weight"], "bias": answer["bias"]}, base.metadata)
            expected = _affine_forward(base, hidden)
            actual = _affine_forward(reconstructed, hidden)
            audit.update(reference_outputs=expected.tolist(), reconstructed_outputs=actual.tolist(),
                         reconstructed_artifact=reconstructed.to_dict(),
                         evaluator_accounting={"affine_forward_batches": 2,
                                               "evaluated_model_examples": 2 * len(hidden)})
            truth = 1 if base.tensors["weight"][DESIGNATED_COEFFICIENT] > 0 else -1
            return {"exact_sign": truth, "reported_sign": answer["sign"],
                    "abstained": answer["abstain"],
                    "sign_correct": None if answer["abstain"] else answer["sign"] == truth,
                    "coefficient_absolute_error": abs(float(
                        reconstructed.tensors["weight"][DESIGNATED_COEFFICIENT]
                        - base.tensors["weight"][DESIGNATED_COEFFICIENT])),
                    "hidden_response": _metrics(expected, actual), "formal_certificate": None}

        receipt = session.evaluate_heldout(evaluate, evaluator_id="affine-sign-and-response/1")
        write_json(out / "heldout.json", audit)
        return {"case_id": case_id, "family": "affine_sign", "mode": mode, "access": access,
                "status": receipt.status, "candidate_ids": [candidate.sha256],
                "abstained": candidate.answer["abstain"],
                "base_sha256": base.sha256, "receipt_sha256": receipt.sha256,
                "metrics": receipt.to_dict()["result"], "error": receipt.to_dict()["error"]}
    finally:
        _persist_session(out, session)


def _target(spec: dict[str, Any], inputs: np.ndarray) -> np.ndarray:
    if spec["kind"] == "scaled_absolute":
        return _dyadic(spec["scale"], "scale") * np.abs(inputs - _dyadic(spec["center"], "center"))
    return np.clip(inputs, _dyadic(spec["low"], "low"), _dyadic(spec["high"], "high"))


def _run_creation_case(seed: int, spec: dict[str, Any], index: int, out: Path) -> dict[str, Any]:
    case_id = out.name
    task = TaskSpec(f"auxiliary:{seed}:{case_id}", "create",
                    "Construct the declared scalar formula using two ReLU units",
                    "dense-relu-dense/1", constraints={
                        "formula": spec, "selection": "analytic arm fixed before evaluation",
                        "training_allowed": False, "feedback_allowed": False,
                        "privileged_hints": ["Exact target formula and two-ReLU construction family"],
                        "semantic_names": {"input": "x", "output": "y"}})
    session = ProgrammerSession(task, access=AccessPolicy(metadata=False, weights=False),
        limits=_limits(proposals=3),
        splits=[SplitSpec("test", [f"{case_id}:{i}" for i in range(HIDDEN_EXAMPLES)])])
    write_json(out / "task.json", task.to_dict())
    write_json(out / "spec.json", spec)
    try:
        rng = _rng(seed, 2200 + index)
        zero = Network(np.zeros((2, 1)), np.zeros(2), np.zeros((1, 2)), np.zeros(1),
                       input_names=("x",), output_names=("y",))
        random = Network(rng.integers(-8, 9, (2, 1)) / 4., rng.integers(-8, 9, 2) / 4.,
                         rng.integers(-8, 9, (1, 2)) / 4., rng.integers(-8, 9, 1) / 4.,
                         input_names=("x",), output_names=("y",))
        artifacts = {"zero": zero, "random": random, "analytic": analytic_network(spec)}
        write_json(out / "construction.json", {
            "method": "fixed_formula_no_fitting", "predeclared_arms": list(CREATION_ARMS),
            "selected_arm": "analytic", "selection_uses_heldout": False,
            "random_seed_entropy": [seed, 2200 + index], "rng": "numpy.PCG64",
            "random_parameters": "uniform integers [-8,8] divided by 4, tensor order w1,b1,w2,b2",
            "replay_program_parent_sha256": zero.sha256,
            "replay_program_scope": "Reproduce each artifact from a zero reference; create task has no parent",
            "formal_certificate": None,
        })
        session.record_provider_usage(input_tokens=0, output_tokens=0)
        candidates = {}
        for arm in CREATION_ARMS:
            write_json(out / f"{arm}.network.json", artifacts[arm].to_dict())
            program = difference(zero, artifacts[arm], f"Replay the fixed {arm} construction from zero")
            write_json(out / f"{arm}.program.json", program.to_dict())
            candidates[arm] = session.propose_artifact(artifacts[arm])
        session.seal(candidates["analytic"])
        hidden_stream = 2300 + index
        hidden = _rng(seed, hidden_stream).uniform(-4., 4., (HIDDEN_EXAMPLES, 1))
        expected = _target(spec, hidden)
        audit: dict[str, Any] = {
            "seed_entropy": [seed, hidden_stream], "rng": "numpy.PCG64",
            "distribution": "uniform(-4,4), shape=(128,1)", "inputs": hidden.tolist(),
            "input_visibility": "evaluator_only_until_seal", "reference_outputs": expected.tolist(),
            "arm_outputs": {},
        }

        def evaluate(selected):
            if selected is not candidates["analytic"]:
                raise ValueError("The analytic arm must be selected before hidden evaluation")
            metrics = {}
            for arm in CREATION_ARMS:
                actual = candidates[arm].artifact(hidden)
                audit["arm_outputs"][arm] = actual.tolist()
                metrics[arm] = {"candidate_sha256": candidates[arm].sha256,
                                "artifact_sha256": candidates[arm].artifact.sha256,
                                "hidden_response": _metrics(expected, actual)}
            audit["evaluator_accounting"] = {"network_forward_batches": len(CREATION_ARMS),
                                            "evaluated_model_examples": len(CREATION_ARMS) * len(hidden)}
            return {"selected_arm": "analytic", "selection_predeclared": True,
                    "arms": metrics, "formal_certificate": None}

        receipt = session.evaluate_heldout(evaluate, evaluator_id="analytic-construction-response/1")
        write_json(out / "heldout.json", audit)
        return {"case_id": case_id, "family": "analytic_creation", "mode": "create",
                "formula": spec["kind"], "status": receipt.status,
                "candidate_ids": [candidates[arm].sha256 for arm in CREATION_ARMS],
                "selected_candidate_id": candidates["analytic"].sha256,
                "receipt_sha256": receipt.sha256, "metrics": receipt.to_dict()["result"],
                "error": receipt.to_dict()["error"]}
    finally:
        _persist_session(out, session)


def run_auxiliary_tasks(seed: int, out: Path) -> dict[str, Any]:
    """Write six controlled cases to a fresh directory, preserving failed cases.

The returned dictionary equals summary.json. Correct counts remain separated by
task family; they are not pooled into an overall task-accuracy claim. Caller-
owned evidence files are not a security boundary around a same-process reader.
"""
    seed, out = _seed(seed), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    specifications = _creation_specs(seed)
    cases = [
        {"case_id": f"affine-{label}-{access}", "family": "affine_sign", "sign": sign,
         "label": label, "access": access, "mode": mode}
        for sign, label, mode in ((1, "a", "read"), (-1, "b", "diagnose"))
        for access in ("exact", "rounded3")
    ] + [
        {"case_id": f"create-{spec['kind']}", "family": "analytic_creation", "spec": spec, "index": index}
        for index, spec in enumerate(specifications)
    ]
    write_json(out / "protocol.json", {
        "schema": "wsp-auxiliary-protocol/1", "seed": seed, "cases": cases,
        "hidden_examples_per_case": HIDDEN_EXAMPLES, "absolute_tolerance": RESPONSE_TOLERANCE,
        "creation_arms": list(CREATION_ARMS), "creation_selection": "analytic_a_priori",
        "programmer_kind": "deterministic_fixture", "model_ability_measured": False,
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
        "affine_streams": {"weights": 1100, "hidden_a": 1200, "hidden_b": 1201},
        "creation_streams": {"specification": 2100, "random_parameters": [2200, 2201],
                             "hidden_inputs": [2300, 2301]},
        "enforcement": "cooperative_same_process_interface", "formal_certificate": None,
    })
    results = []
    for case in cases:
        directory = out / case["case_id"]
        directory.mkdir()
        write_json(directory / "attempt.json", case | {"world_seed": seed})
        try:
            if case["family"] == "affine_sign":
                result = _run_read_case(seed, case["sign"], case["label"], case["access"], case["mode"], directory)
            else:
                result = _run_creation_case(seed, case["spec"], case["index"], directory)
        except Exception as error:
            candidates_path = directory / "candidates.json"
            candidates = read_json(candidates_path) if candidates_path.exists() else []
            result = {"case_id": case["case_id"], "family": case["family"], "status": "failed",
                      "metrics": None, "candidate_ids": [candidate["sha256"] for candidate in candidates],
                      "error": {"type": type(error).__name__, "message": str(error)},
                      "partial_trace_present": (directory / "trace.json").exists()}
            if case["family"] == "affine_sign":
                result["abstained"] = any(candidate["answer"]["abstain"] for candidate in candidates)
        write_json(directory / "result.json", result)
        results.append(result)
    readings = [case for case in results if case["family"] == "affine_sign" and case["metrics"] is not None]
    creations = [case for case in results if case["family"] == "analytic_creation" and case["metrics"] is not None]
    summary = {
        "schema": "wsp-auxiliary-study/1", "seed": seed, "cases": len(cases),
        "case_type_counts": {"read": 2, "diagnose": 2, "create": 2},
        "errors": sum(case["status"] == "failed" for case in results),
        "correct": {"coefficient_sign": sum(case["metrics"]["sign_correct"] is True for case in readings),
                    "analytic_creation": sum(case["metrics"]["arms"]["analytic"]["hidden_response"]["within_tolerance"]
                                             for case in creations)},
        "correct_scope": {"coefficient_sign": "Nonabstained exact-sign classifications, out of four cases",
                          "analytic_creation": "Selected analytic response within tolerance, out of two cases"},
        "abstentions": sum(case.get("abstained", False) for case in results),
        "incorrect_signs": sum(case["metrics"]["sign_correct"] is False for case in readings),
        "candidate_ids": {case["case_id"]: case["candidate_ids"] for case in results},
        "case_results": results,
        "provider": {"kind": "deterministic_fixture", "llm_calls": 0, "input_tokens": 0,
                     "output_tokens": 0, "model_ability_measured": False},
        "formal_certificate": None, "evidence_manifest": "manifest.json",
    }
    summary["summary_sha256"] = digest(summary)
    write_json(out / "summary.json", summary)
    manifest = {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*.json"))}
    write_json(out / "manifest.json", {"schema": "wsp-auxiliary-manifest/1", "files": manifest})
    return summary
