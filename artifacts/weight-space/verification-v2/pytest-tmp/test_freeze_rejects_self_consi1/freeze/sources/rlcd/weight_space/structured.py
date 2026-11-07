"""Data-only tensor programs and explicit projection to native floating formats.

Structured commands are lowered to the existing exact-parent literal Program.
They never name Python imports or execute provider-authored source code. Projection
is a proposal operation: the resulting endpoint still needs admission/evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from typing import Any, Mapping

import numpy as np

from .artifact import WeightArtifact, valid_tensor_name
from .model import _TensorData, canonical_bytes, digest, finite_binary64
from .program import Program, difference


NATIVE_DTYPES = ("float64", "float32", "float16", "bfloat16")


def _array(value: Any) -> np.ndarray:
    return _TensorData.capture(value).array()


def _validate_command(command: Any) -> None:
    if not isinstance(command, dict):
        raise ValueError("Commands must be JSON objects")
    kind = command.get("op")
    operands = {"replace": {"values"}, "add": {"values"}, "scale": {"factor"},
                "low_rank": {"left", "right", "alpha"}}
    if not isinstance(kind, str) or kind not in operands:
        raise ValueError("Unsupported structured operation")
    required = {"op", "tensor"} | operands[kind]
    if set(command) not in (required, required | {"block"}):
        raise ValueError("Unknown or missing command fields")
    if not valid_tensor_name(command["tensor"]):
        raise ValueError("Invalid tensor name")
    if "block" in command:
        block = command["block"]
        if not isinstance(block, list) or any(
            not isinstance(axis, list) or len(axis) != 2 or
            any(type(n) is not int for n in axis) or axis[0] < 0 or axis[1] <= axis[0]
            for axis in block
        ):
            raise ValueError("Blocks contain half-open nonempty [start, stop] axes")
    for key in operands[kind]:
        if key in ("factor", "alpha"):
            finite_binary64(command[key])
        else:
            _array(command[key])


class StructuredProgram:
    """Immutable JSON commands with sequential, atomic lowering semantics."""

    __slots__ = ("_bytes",)

    def __init__(self, base_sha256: str, commands: list[dict[str, Any]], intent: str):
        Program(base_sha256, (), intent)  # Reuse the literal identity/intent contract.
        if not isinstance(commands, list):
            raise ValueError("Commands must be a JSON array")
        for command in commands:
            _validate_command(command)
        object.__setattr__(self, "_bytes", canonical_bytes({
            "schema": "wsp-structured-program/1", "base_sha256": base_sha256,
            "intent": intent, "commands": commands,
        }))

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Structured programs are immutable")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._bytes)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StructuredProgram:
        if (not isinstance(value, dict) or
                set(value) != {"schema", "base_sha256", "intent", "commands"} or
                value["schema"] != "wsp-structured-program/1"):
            raise ValueError("Unknown structured-program schema or fields")
        return cls(value["base_sha256"], value["commands"], value["intent"])

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


def native_round(values: Any, dtype: str) -> np.ndarray:
    """Return the finite endpoint actually representable in a declared format.

    bfloat16 uses the installed PyTorch conversion, recorded as a trusted runtime
    dependency. This is measured conversion, not a proof of hardware rounding.
    """
    if dtype not in NATIVE_DTYPES:
        raise ValueError(f"Unsupported native floating format: {dtype}")
    source = _array(values)
    if dtype == "bfloat16":
        try:
            import torch
        except ImportError as error:
            raise ValueError("bfloat16 projection requires the optional PyTorch bridge") from error
        result = torch.from_numpy(source.copy()).to(torch.bfloat16).to(torch.float64).numpy()
    else:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            result = source.astype(dtype).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Native projection overflowed to a nonfinite value")
    return _array(result)


@dataclass(frozen=True)
class Lowering:
    program: Program
    _receipt_bytes: bytes

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_bytes)


def lower_structured(base: WeightArtifact, proposal: StructuredProgram, *,
                     native_dtypes: Mapping[str, str] | None = None,
                     cast: str = "reject") -> Lowering:
    """Apply commands to a temporary copy, project once, and bind literal edits.

    ``reject`` refuses any rounding. ``round`` explicitly records every rounded
    endpoint. The *base* must already be representable in its declared formats,
    so projection cannot silently change unrelated source state.
    """
    if cast not in ("reject", "round"):
        raise ValueError("Cast policy must be reject or round")
    data = proposal.to_dict()
    if data["base_sha256"] != base.sha256:
        raise ValueError("Structured proposal has a stale or wrong parent")
    original = base.tensors
    dtypes = ({name: "float64" for name in original} if native_dtypes is None
              else dict(native_dtypes))
    if set(dtypes) != set(original) or any(t not in NATIVE_DTYPES for t in dtypes.values()):
        raise ValueError("Native formats must cover every captured tensor exactly")
    for name, array in original.items():
        if not np.array_equal(native_round(array, dtypes[name]), array):
            raise ValueError(f"Base tensor {name} is not representable in its declared format")
    values = {name: array.copy() for name, array in original.items()}
    touched = {name: np.zeros(array.shape, dtype=bool) for name, array in original.items()}
    for command in data["commands"]:
        name = command["tensor"]
        if name not in values:
            raise ValueError("Command names an undeclared tensor")
        target = values[name]
        if "block" in command:
            block = command["block"]
            if len(block) != target.ndim or any(stop > size for (_, stop), size in zip(block, target.shape)):
                raise ValueError("Block is outside its tensor")
            index = tuple(slice(start, stop) for start, stop in block)
        else:
            index = tuple(slice(None) for _ in target.shape)
        current = target[index]
        kind = command["op"]
        with np.errstate(over="raise", invalid="raise"):
            if kind in ("replace", "add"):
                operand = _array(command["values"])
                if operand.shape != current.shape:
                    raise ValueError("Block operand shape must match exactly; broadcasting is forbidden")
                updated = operand if kind == "replace" else current + operand
            elif kind == "scale":
                updated = current * command["factor"]
            else:
                left, right = _array(command["left"]), _array(command["right"])
                if (current.ndim != 2 or left.ndim != 2 or right.ndim != 2 or
                        left.shape[1] != right.shape[0] or
                        (left.shape[0], right.shape[1]) != current.shape):
                    raise ValueError("Low-rank factors must multiply to the selected matrix shape")
                updated = current + command["alpha"] * (left @ right)
        target[index] = _array(updated)
        touched[name][index] = True
    requested = base.replace(**values)
    endpoints, projections = {}, {}
    for name, array in values.items():
        endpoint = native_round(array, dtypes[name])
        changed = array != endpoint
        if cast == "reject" and np.any(changed):
            raise ValueError(f"Native format would round tensor {name}; explicitly choose cast='round'")
        errors = [abs(Fraction.from_float(float(a))-Fraction.from_float(float(b)))
                  for a, b in zip(array[changed].flat, endpoint[changed].flat)]
        projections[name] = {
            "dtype": dtypes[name], "touched_cells": int(touched[name].sum()),
            "requested_changed_cells": int(np.count_nonzero(original[name] != array)),
            "effective_changed_cells": int(np.count_nonzero(original[name] != endpoint)),
            "rounded_cells": int(changed.sum()),
            "maximum_rounding_error_exact": str(max(errors, default=Fraction(0))),
        }
        endpoints[name] = endpoint
    candidate = base.replace(**endpoints)
    program = difference(base, candidate, data["intent"])
    payload = {
        "schema": "wsp-lowering/1", "proposal_sha256": proposal.sha256,
        "base_sha256": base.sha256, "requested_candidate_sha256": requested.sha256,
        "candidate_sha256": candidate.sha256, "program_sha256": program.sha256,
        "command_count": len(data["commands"]), "cast": cast,
        "native_dtypes": dtypes, "projections": projections,
        "effective_edits": len(program.edits), "behavioral_status": "not_checked",
        "numerics": "NumPy binary64 operations; explicit native endpoint projection",
        "numpy_version": np.__version__,
    }
    if "bfloat16" in dtypes.values():
        import torch
        payload["torch_version"] = torch.__version__
    return Lowering(program, canonical_bytes(payload | {"sha256": digest(payload)}))


def verify_lowering(base: WeightArtifact, proposal: StructuredProgram,
                    receipt: dict[str, Any]) -> bool:
    """Recompute the lowering; compare JSON types as well as numerical values."""
    try:
        actual = lower_structured(base, proposal, native_dtypes=receipt["native_dtypes"],
                                  cast=receipt["cast"]).receipt
        return canonical_bytes(actual) == canonical_bytes(receipt)
    except (ValueError, TypeError, KeyError, FloatingPointError):
        return False
