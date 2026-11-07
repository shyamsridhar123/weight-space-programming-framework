"""Behavioral examples -> literal readout edits with an auditable least-squares fit."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .model import Network
from .program import Program, difference


@dataclass(frozen=True)
class Compilation:
    program: Program
    diagnostics: dict[str, Any]


def compile_readout(base: Network, target_inputs: Any, desired_delta: Any,
                    retain_inputs: Any | None = None, *, retention_weight: float = 8.0,
                    ridge: float = 0.01, editable_hidden: Sequence[int] | None = None,
                    edit_bias: bool = False, intent: str = "Fit a behavioral output-change specification") -> Compilation:
    """Minimize mean target error + λ mean retention drift + ridge*||edit||².

    The feature extractor is frozen. This compiles examples, not unrestricted
    natural language; an LLM proposal can enter the same program/admission API.
    No optimizer ever reads a held-out evaluation set inside this function.
    """
    if (not math.isfinite(retention_weight) or retention_weight < 0
            or not math.isfinite(ridge) or ridge <= 0):
        raise ValueError("Retention weight must be nonnegative and ridge strictly positive")
    indices = list(range(base.shape[1])) if editable_hidden is None else list(editable_hidden)
    if (any(type(i) is not int or i < 0 or i >= base.shape[1] for i in indices)
            or len(set(indices)) != len(indices) or (not indices and not edit_bias)):
        raise ValueError("Invalid or empty editable readout basis")
    target = np.asarray(target_inputs, dtype=np.float64)
    desired = np.asarray(desired_delta, dtype=np.float64)
    if (target.ndim != 2 or not len(target) or desired.shape != (len(target), base.shape[2])
            or not np.isfinite(desired).all()):
        raise ValueError("Supply nonempty target examples and one desired delta vector per example")

    def design(inputs: np.ndarray) -> np.ndarray:
        matrix = base.hidden(inputs)[:, indices]
        return np.column_stack((matrix, np.ones(len(inputs)))) if edit_bias else matrix

    target_design = design(target)
    matrices = [target_design/np.sqrt(len(target))]
    responses = [desired/np.sqrt(len(target))]
    retain_design = None
    if retain_inputs is not None:
        retain = np.asarray(retain_inputs, dtype=np.float64)
        if retain.ndim != 2 or not len(retain):
            raise ValueError("Retention examples must be a nonempty matrix when provided")
        retain_design = design(retain)
        matrices.append(retain_design*np.sqrt(retention_weight/len(retain)))
        responses.append(np.zeros((len(retain), base.shape[2])))
    width = target_design.shape[1]
    matrices.append(np.sqrt(ridge)*np.eye(width))
    responses.append(np.zeros((width, base.shape[2])))
    edit, _, rank, singular = np.linalg.lstsq(np.vstack(matrices), np.vstack(responses), rcond=None)
    w2, b2 = base.w2.copy(), base.b2.copy()
    w2[:, indices] += edit[:len(indices)].T
    if edit_bias:
        b2 += edit[-1]
    candidate = base.replace(w2=w2, b2=b2)
    program = difference(base, candidate, intent)
    return Compilation(program, {
        "compiler": "frozen-feature-ridge-least-squares/1", "target_examples": len(target),
        "retention_examples": 0 if retain_design is None else len(retain_design),
        "retention_weight": retention_weight, "ridge": ridge, "rank": int(rank),
        "condition_number": float(singular[0]/singular[-1]), "edit_cells": len(program.edits),
        "target_rmse": float(np.sqrt(np.mean((target_design@edit-desired)**2))),
        "retention_rmse": None if retain_design is None else float(np.sqrt(np.mean((retain_design@edit)**2))),
        "edit_l2": float(np.linalg.norm(edit)), "semantics": "fit to supplied examples; admission and generalization unproven",
    })
