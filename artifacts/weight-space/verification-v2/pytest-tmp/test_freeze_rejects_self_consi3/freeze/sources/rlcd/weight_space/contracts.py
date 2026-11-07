"""Exact rational real-arithmetic bounds; never silently call floats a proof.

Every finite binary64 parameter is lifted to its exact rational value. This is a
sound bound for the corresponding mathematical ReLU network over a real box.
The independently checked Lean theorem justifies the formula; Python itself and
IEEE-754 execution are not thereby formally verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from .model import Network, InputBox, canonical_bytes, digest, finite_binary64
from .program import Program, apply_program


def rational(value: float) -> Fraction:
    return Fraction.from_float(finite_binary64(value))


def exact_score(model: Network, inputs: list[float]) -> tuple[Fraction, ...]:
    if len(inputs) != model.shape[0]:
        raise ValueError("Wrong exact-score input")
    x = list(map(rational, inputs))
    hidden = [max(Fraction(0), rational(b) + sum((rational(w)*v for w, v in zip(row, x)), Fraction(0)))
              for row, b in zip(model.w1, model.b1)]
    return tuple(rational(b) + sum((rational(w)*h for w, h in zip(row, hidden)), Fraction(0))
                 for row, b in zip(model.w2, model.b2))


def certify_drift(base: Network, candidate: Network, box: InputBox) -> dict[str, Any]:
    """A global componentwise score-drift bound for simultaneous all-layer edits."""
    if not base.compatible(candidate) or len(box.lower) != base.shape[0]:
        raise ValueError("Certificate schemas/dimensions do not agree")
    lower, upper = list(map(rational, box.lower)), list(map(rational, box.upper))
    radius = [max(abs(lo), abs(hi)) for lo, hi in zip(lower, upper)]
    hidden_upper, hidden_drift = [], []
    for h in range(base.shape[1]):
        old_row = list(map(rational, base.w1[h]))
        new_row = list(map(rational, candidate.w1[h]))
        hidden_upper.append(max(Fraction(0), rational(base.b1[h]) + sum(
            (max(w*lo, w*hi) for w, lo, hi in zip(old_row, lower, upper)), Fraction(0))))
        hidden_drift.append(abs(rational(candidate.b1[h])-rational(base.b1[h])) + sum(
            (abs(v-w)*r for v, w, r in zip(new_row, old_row, radius)), Fraction(0)))
    bounds = [abs(rational(candidate.b2[o])-rational(base.b2[o])) + sum(
        (abs(rational(candidate.w2[o,h])-rational(base.w2[o,h]))*hidden_upper[h]
         + abs(rational(candidate.w2[o,h]))*hidden_drift[h]
         for h in range(base.shape[1])), Fraction(0)) for o in range(base.shape[2])]
    payload = {"schema": "wsp-drift-certificate/1", "arithmetic": "exact-rational-lift-of-binary64",
               "base_sha256": base.sha256, "candidate_sha256": candidate.sha256,
               "domain": box.to_dict(), "hidden_upper": list(map(str, hidden_upper)),
               "hidden_drift": list(map(str, hidden_drift)), "output_bounds": list(map(str, bounds)),
               "theorem": "WeightSpace.score_drift_le", "implementation_verified": False,
               "scope": "real score drift on the stated box; excludes floating-point error and task reward"}
    return payload | {"sha256": digest(payload)}


def verify_certificate(base: Network, candidate: Network, certificate: dict[str, Any]) -> bool:
    """Independently recompute, including all claims and provenance in the receipt."""
    try:
        box = InputBox(**certificate["domain"])
        return canonical_bytes(certificate) == canonical_bytes(certify_drift(base, candidate, box))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class Contract:
    box: InputBox
    max_output_drift: tuple[float, ...]
    allowed_tensors: tuple[str, ...] = ("w1", "b1", "w2", "b2")
    max_parameter_abs: float = 64.0
    max_edits: int | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.box, InputBox) or isinstance(self.max_output_drift, (str, bytes, dict))
                or isinstance(self.allowed_tensors, (str, bytes, dict))):
            raise ValueError("Contracts need an input box and sequences of budgets and tensor names")
        object.__setattr__(self, "max_output_drift", tuple(map(finite_binary64, self.max_output_drift)))
        object.__setattr__(self, "allowed_tensors", tuple(self.allowed_tensors))
        object.__setattr__(self, "max_parameter_abs", finite_binary64(self.max_parameter_abs))
        if (not self.max_output_drift or any(x < 0 for x in self.max_output_drift)
                or self.max_parameter_abs <= 0):
            raise ValueError("Contract budgets must be finite and nonnegative")
        if set(self.allowed_tensors) - {"w1", "b1", "w2", "b2"}:
            raise ValueError("Unknown writable tensor")
        if self.max_edits is not None and (type(self.max_edits) is not int or self.max_edits < 0):
            raise ValueError("Invalid edit-count budget")

    def to_dict(self) -> dict[str, Any]:
        return {"box": self.box.to_dict(), "max_output_drift": list(self.max_output_drift),
                "allowed_tensors": list(self.allowed_tensors), "max_parameter_abs": self.max_parameter_abs,
                "max_edits": self.max_edits}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Contract:
        if (not isinstance(value, dict)
                or set(value) != {"box", "max_output_drift", "allowed_tensors", "max_parameter_abs", "max_edits"}):
            raise ValueError("Unknown contract schema")
        if (not isinstance(value["box"], dict) or set(value["box"]) != {"lower", "upper"}
                or not isinstance(value["box"]["lower"], list) or not isinstance(value["box"]["upper"], list)
                or not isinstance(value["max_output_drift"], list)
                or not isinstance(value["allowed_tensors"], list)):
            raise ValueError("Serialized contracts require a box object and JSON arrays")
        return cls(InputBox(**value["box"]), tuple(value["max_output_drift"]),
                   tuple(value["allowed_tensors"]), value["max_parameter_abs"], value["max_edits"])


def check_contract(base: Network, program: Program, contract: Contract) -> dict[str, Any]:
    candidate = apply_program(base, program)
    if len(contract.max_output_drift) != base.shape[2]:
        raise ValueError("One drift budget is required per output")
    issues = []
    if any(edit.tensor not in contract.allowed_tensors for edit in program.edits):
        issues.append("write-set")
    if any(np.any(np.abs(t) > contract.max_parameter_abs) for t in candidate.tensors.values()):
        issues.append("parameter-bound")
    if contract.max_edits is not None and len(program.edits) > contract.max_edits:
        issues.append("edit-count")
    certificate = certify_drift(base, candidate, contract.box)
    if any(Fraction(bound) > rational(limit)
           for bound, limit in zip(certificate["output_bounds"], contract.max_output_drift)):
        issues.append("behavioral-drift")
    payload = {"schema": "wsp-admission/1", "accepted": not issues, "issues": issues,
               "program_sha256": program.sha256, "contract": contract.to_dict(),
               "certificate": certificate}
    return payload | {"sha256": digest(payload)}


def preserves_argmax(scores: list[float], per_output_bounds: list[str]) -> bool:
    """Sufficient strict-margin condition; ties intentionally yield no guarantee."""
    if len(scores) != len(per_output_bounds) or len(scores) < 2:
        raise ValueError("An argmax certificate needs at least two scores/bounds")
    values = list(map(rational, scores))
    bounds = list(map(Fraction, per_output_bounds))
    if any(b < 0 for b in bounds):
        raise ValueError("Drift bounds are nonnegative")
    winner = max(range(len(values)), key=values.__getitem__)
    return all(values[winner]-values[j] > bounds[winner]+bounds[j]
               for j in range(len(values)) if j != winner)
