"""Architecture-independent named tensors and structural admission.

An artifact is data, never executable model code. A caller supplies its execution
adapter. Behavioral certificates require a separately supported semantic model.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Self, runtime_checkable

import numpy as np

from .model import _TensorData, canonical_bytes, digest, finite_binary64


def valid_tensor_name(name: str) -> bool:
    return isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,255}", name))


@runtime_checkable
class WeightArtifact(Protocol):
    """The edit algebra needs tensors and identity, not a particular architecture."""

    @property
    def tensors(self) -> Mapping[str, np.ndarray]: ...
    @property
    def sha256(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...
    def compatible(self, other: Any) -> bool: ...
    def replace(self, /, **tensors: Any) -> Self: ...


class TensorArtifact:
    """Immutable binary64 tensor data with an explicit architecture/interface ID.

    Arbitrary ranks include scalars, attention matrices, convolution kernels, and
    model buffers. Tensor shapes are structural facts, not a proof of semantics.
    Original native dtypes can be recorded in metadata by an execution adapter.
    """

    __slots__ = ("_architecture", "_tensors", "_metadata_bytes", "_sealed")

    def __init__(self, architecture: str, tensors: Mapping[str, Any], metadata: dict[str, Any] | None = None):
        if not isinstance(architecture, str) or not architecture.strip():
            raise ValueError("A nonblank architecture/interface identifier is required")
        if not isinstance(tensors, Mapping) or not tensors or any(not valid_tensor_name(name) for name in tensors):
            raise ValueError("Supply nonempty, portable named tensors")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")
        object.__setattr__(self, "_architecture", architecture)
        object.__setattr__(self, "_tensors", MappingProxyType({
            name: _TensorData.capture(value) for name, value in tensors.items()}))
        object.__setattr__(self, "_metadata_bytes", canonical_bytes(metadata or {}))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Tensor artifacts are immutable")
        object.__setattr__(self, name, value)

    @property
    def architecture(self) -> str:
        return self._architecture

    @property
    def tensors(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({name: tensor.array() for name, tensor in self._tensors.items()})

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self._metadata_bytes)

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    @property
    def parameter_count(self) -> int:
        """Scalar entries, including any explicitly captured buffers."""
        return sum(tensor.size for tensor in self.tensors.values())

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "wsp-tensors/1", "dtype": "float64", "architecture": self.architecture,
                "metadata": self.metadata, "tensors": {k: v.tolist() for k, v in self.tensors.items()}}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TensorArtifact:
        if (not isinstance(value, dict)
                or set(value) != {"schema", "dtype", "architecture", "metadata", "tensors"}
                or value["schema"] != "wsp-tensors/1" or value["dtype"] != "float64"):
            raise ValueError("Unknown named-tensor schema or numerical representation")
        if not isinstance(value["tensors"], dict) or not isinstance(value["metadata"], dict):
            raise ValueError("Named tensors and metadata must be JSON objects")
        return cls(value["architecture"], value["tensors"], value["metadata"])

    def compatible(self, other: Any) -> bool:
        return (isinstance(other, TensorArtifact) and self.architecture == other.architecture
                and self._metadata_bytes == other._metadata_bytes
                and self._tensors.keys() == other._tensors.keys()
                and all(t.shape == other._tensors[k].shape for k, t in self._tensors.items()))

    def replace(self, /, **tensors: Any) -> TensorArtifact:
        if set(tensors) - self._tensors.keys():
            raise ValueError("Cannot add undeclared tensors through an edit")
        result = TensorArtifact(self.architecture, dict(self.tensors) | tensors, self.metadata)
        if not self.compatible(result):
            raise ValueError("An edit cannot change tensor shapes or semantic metadata")
        return result


@dataclass(frozen=True)
class StructuralPolicy:
    """Portable admission requirements that make no behavioral guarantee."""

    writable: tuple[str, ...]
    max_edits: int
    max_parameter_abs: float
    max_change_abs: float

    def __post_init__(self) -> None:
        if isinstance(self.writable, (str, bytes, dict)):
            raise ValueError("Writable tensors must be a sequence of names")
        object.__setattr__(self, "writable", tuple(self.writable))
        object.__setattr__(self, "max_parameter_abs", finite_binary64(self.max_parameter_abs))
        object.__setattr__(self, "max_change_abs", finite_binary64(self.max_change_abs))
        if (not self.writable or any(not valid_tensor_name(x) for x in self.writable)
                or type(self.max_edits) is not int or self.max_edits < 0
                or self.max_parameter_abs < 0 or self.max_change_abs < 0):
            raise ValueError("Invalid structural admission policy")

    def to_dict(self) -> dict[str, Any]:
        return {"writable": list(self.writable), "max_edits": self.max_edits,
                "max_parameter_abs": self.max_parameter_abs, "max_change_abs": self.max_change_abs}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StructuralPolicy:
        if (not isinstance(value, dict) or set(value) != {
                "writable", "max_edits", "max_parameter_abs", "max_change_abs"}
                or not isinstance(value["writable"], list)):
            raise ValueError("Unknown structural-policy fields or representation")
        return cls(**value)

    def check(self, base: WeightArtifact, program: Any) -> dict[str, Any]:
        from fractions import Fraction
        from .program import apply_program
        candidate = apply_program(base, program)
        issues = []
        if any(edit.tensor not in self.writable for edit in program.edits):
            issues.append("write-set")
        if len(program.edits) > self.max_edits:
            issues.append("edit-count")
        if any(np.any(np.abs(t) > self.max_parameter_abs) for t in candidate.tensors.values()):
            issues.append("parameter-bound")
        if any(abs(Fraction.from_float(e.after)-Fraction.from_float(e.before))
               > Fraction.from_float(self.max_change_abs) for e in program.edits):
            issues.append("change-bound")
        payload = {"schema": "wsp-structural-admission/1", "structural_accepted": not issues,
                   "issues": issues, "program_sha256": program.sha256,
                   "candidate_sha256": candidate.sha256,
                   "policy": self.to_dict(),
                   "behavioral_status": "not_checked", "formal_certificate": None}
        return payload | {"sha256": digest(payload)}
