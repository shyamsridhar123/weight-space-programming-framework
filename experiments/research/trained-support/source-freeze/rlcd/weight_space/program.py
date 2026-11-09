"""Typed atomic edit programs, composition, rollback and arithmetic baselines."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, TypeVar

import numpy as np

from .model import digest, finite_binary64
from .artifact import WeightArtifact, valid_tensor_name

A = TypeVar("A", bound=WeightArtifact)


@dataclass(frozen=True)
class Edit:
    tensor: str
    index: tuple[int, ...]
    before: float
    after: float

    def __post_init__(self) -> None:
        if isinstance(self.index, (str, bytes, dict)):
            raise ValueError("Coordinates must be a sequence of nonnegative integers")
        try:
            object.__setattr__(self, "index", tuple(self.index))
        except TypeError:
            raise ValueError("Coordinates must be a sequence of nonnegative integers") from None
        if not valid_tensor_name(self.tensor):
            raise ValueError("Tensor names must be portable, nonempty identifiers")
        if any(type(i) is not int or i < 0 for i in self.index):
            raise ValueError("Coordinates must be nonnegative integers")
        object.__setattr__(self, "before", finite_binary64(self.before))
        object.__setattr__(self, "after", finite_binary64(self.after))
        if self.before == self.after:
            raise ValueError("Unchanged cells do not constitute an edit")

    def to_dict(self) -> dict[str, Any]:
        return {"tensor": self.tensor, "index": list(self.index),
                "before": float(self.before), "after": float(self.after)}


@dataclass(frozen=True)
class Program:
    base_sha256: str
    edits: tuple[Edit, ...]
    intent: str

    def __post_init__(self) -> None:
        if isinstance(self.edits, (str, bytes, dict)):
            raise ValueError("Programs contain a sequence of typed edits")
        try:
            object.__setattr__(self, "edits", tuple(self.edits))
        except TypeError:
            raise ValueError("Programs contain a sequence of typed edits") from None
        if (not isinstance(self.base_sha256, str) or len(self.base_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.base_sha256)):
            raise ValueError("An exact parent SHA-256 is required")
        if not isinstance(self.intent, str) or not self.intent.strip():
            raise ValueError("Programs need an explicit intent, which remains a hypothesis")
        if any(not isinstance(e, Edit) for e in self.edits):
            raise ValueError("Programs contain typed edits")
        if len({(e.tensor, e.index) for e in self.edits}) != len(self.edits):
            raise ValueError("Duplicate coordinates are ambiguous")

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "wsp-program/1", "base_sha256": self.base_sha256,
                "intent": self.intent, "edits": [e.to_dict() for e in self.edits]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Program:
        if (not isinstance(value, dict) or set(value) != {"schema", "base_sha256", "intent", "edits"}
                or value["schema"] != "wsp-program/1"):
            raise ValueError("Unknown program schema or fields")
        if not isinstance(value["edits"], list):
            raise ValueError("Serialized edits must be a JSON array")
        edits = []
        for e in value["edits"]:
            if not isinstance(e, dict) or set(e) != {"tensor", "index", "before", "after"}:
                raise ValueError("Unknown edit fields")
            if not isinstance(e["index"], list):
                raise ValueError("Serialized coordinates must be a JSON array")
            edits.append(Edit(e["tensor"], tuple(e["index"]), e["before"], e["after"]))
        return cls(value["base_sha256"], tuple(edits), value["intent"])


def apply_program(base: A, program: Program) -> A:
    """Validate all coordinates/preconditions before constructing a new artifact."""
    if base.sha256 != program.base_sha256:
        raise ValueError("Stale or wrong parent network")
    base_tensors = base.tensors
    for edit in program.edits:
        if edit.tensor not in base_tensors:
            raise ValueError("Edit names an undeclared tensor")
        tensor = base_tensors[edit.tensor]
        if len(edit.index) != tensor.ndim or any(i >= size for i, size in zip(edit.index, tensor.shape)):
            raise ValueError("Edit coordinate is outside its tensor")
        if float(tensor[edit.index]) != edit.before:
            raise ValueError("Edit precondition differs from the parent cell")
    tensors = {k: v.copy() for k, v in base_tensors.items()}
    for edit in program.edits:
        tensors[edit.tensor][edit.index] = edit.after
    return base.replace(**tensors)


def difference(base: A, candidate: A, intent: str) -> Program:
    if not base.compatible(candidate):
        raise ValueError("Arithmetic requires matching architecture and semantic schemas")
    candidate_tensors = candidate.tensors
    edits = tuple(Edit(name, tuple(int(i) for i in index), float(values[tuple(index)]),
                       float(candidate_tensors[name][tuple(index)]))
                  for name, values in base.tensors.items()
                  for index in np.argwhere(values != candidate_tensors[name]))
    return Program(base.sha256, edits, intent)


def compose(base: A, programs: list[Program], intent: str = "Sequential composition") -> Program:
    current = base
    for program in programs:
        current = apply_program(current, program)
    return difference(base, current, intent)


def rollback(base: A, program: Program) -> Program:
    candidate = apply_program(base, program)
    return difference(candidate, base, f"Rollback: {program.intent}")


def interpolate(base: A, other: A, alpha: float) -> Program:
    if not base.compatible(other) or not math.isfinite(alpha):
        raise ValueError("Interpolation requires compatible finite parameters")
    with np.errstate(over="raise", invalid="raise"):
        candidate = base.replace(**{k: (1-alpha)*v + alpha*other.tensors[k]
                                    for k, v in base.tensors.items()})
    return difference(base, candidate, f"Coordinate interpolation alpha={alpha}; behavior unverified")
