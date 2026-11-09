"""Explicit semantics and immutable, content-addressed neural artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import numpy as np


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def finite_binary64(value: Any) -> float:
    """Normalize a numeric scalar without changing its declared exact value."""
    message = "Values must be finite numbers exactly representable as binary64"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(message)
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(message) from None
    if not math.isfinite(result):
        raise ValueError(message)
    # NumPy integer/float comparisons can first round the integer. Compare as
    # Python integers or exact ratios, never through that implicit conversion.
    if isinstance(value, (int, np.integer)):
        exact = int(result) == int(value)
    else:
        exact = result.as_integer_ratio() == value.as_integer_ratio()
    if not exact:
        raise ValueError(message)
    return 0.0 if result == 0 else result


@dataclass(frozen=True, slots=True)
class _TensorData:
    """Only immutable storage and dimensions are retained, never array headers."""

    data: bytes
    shape: tuple[int, ...]

    @classmethod
    def capture(cls, value: Any, ndim: int | None = None) -> _TensorData:
        if isinstance(value, np.ndarray):
            original = np.asarray(value)
            if original.dtype.kind not in "iuf":
                raise ValueError("Tensors require numeric values, not booleans, strings or objects")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    data = np.array(original, dtype=np.float64, copy=True)
                    if original.dtype.kind in "iu" or original.dtype.itemsize > 8:
                        if not np.array_equal(data.astype(original.dtype), original):
                            raise ValueError("Tensor values must be exactly representable as binary64")
            except (FloatingPointError, OverflowError):
                raise ValueError("Tensor values must be finite and exactly representable as binary64") from None
        else:
            # An inferred NumPy dtype can already round mixed integer/float
            # lists or turn booleans into integers. Preserve each supplied
            # scalar until its exact binary64 conversion has been checked.
            try:
                original = np.asarray(value, dtype=object)
            except (TypeError, ValueError):
                raise ValueError("Tensors require rectangular numeric values") from None
            data = np.fromiter((finite_binary64(x) for x in original.flat),
                               dtype=np.float64, count=original.size).reshape(original.shape)
        if (ndim is not None and data.ndim != ndim) or any(size == 0 for size in data.shape):
            raise ValueError(f"Expected a nonempty {ndim if ndim is not None else 'arbitrary'}-dimensional tensor")
        if not np.isfinite(data).all():
            raise ValueError("Tensors must contain finite binary64 numbers")
        data[data == 0] = 0.0
        return cls(data.tobytes(), data.shape)

    def array(self) -> np.ndarray:
        # Every call makes fresh headers; even mutations of a returned view's
        # shape, dtype, or base header cannot alter this stored tensor.
        return np.frombuffer(self.data, dtype=np.float64).reshape(self.shape)


def frozen_array(value: Any, ndim: int) -> np.ndarray:
    return _TensorData.capture(value, ndim).array()


def _labels(value: tuple[str, ...], size: int, prefix: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, dict)):
        raise ValueError("Semantic labels must be a sequence of names")
    try:
        labels = tuple(value)
    except TypeError:
        raise ValueError("Semantic labels must be a sequence of names") from None
    labels = labels or tuple(f"{prefix}{i}" for i in range(size))
    if (len(labels) != size
            or any((type(x) is not str and type(x) is not np.str_) or not x.strip() for x in labels)
            or len(set(labels)) != size):
        raise ValueError("Semantic labels must be nonblank, unique and dimension-matched")
    return tuple(str(x) for x in labels)


@dataclass(frozen=True, eq=False, init=False)
class Network:
    """x -> W2 ReLU(W1 x + b1) + b2. W1[H,F], W2[O,H]."""

    _w1: _TensorData = field(repr=False)
    _b1: _TensorData = field(repr=False)
    _w2: _TensorData = field(repr=False)
    _b2: _TensorData = field(repr=False)
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()

    def __init__(self, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray,
                 input_names: tuple[str, ...] = (), output_names: tuple[str, ...] = ()) -> None:
        values = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        for name, ndim in (("w1", 2), ("b1", 1), ("w2", 2), ("b2", 1)):
            object.__setattr__(self, f"_{name}", _TensorData.capture(values[name], ndim))
        h, f = self.w1.shape
        o, h2 = self.w2.shape
        if self.b1.shape != (h,) or h2 != h or self.b2.shape != (o,):
            raise ValueError("Incompatible network tensor dimensions")
        object.__setattr__(self, "input_names", _labels(input_names, f, "x"))
        object.__setattr__(self, "output_names", _labels(output_names, o, "y"))

    @property
    def w1(self) -> np.ndarray:
        return self._w1.array()

    @property
    def b1(self) -> np.ndarray:
        return self._b1.array()

    @property
    def w2(self) -> np.ndarray:
        return self._w2.array()

    @property
    def b2(self) -> np.ndarray:
        return self._b2.array()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.w1.shape[1], self.w1.shape[0], self.w2.shape[0]

    @property
    def parameter_count(self) -> int:
        return sum(x.size for x in self.tensors.values())

    @property
    def tensors(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ("w1", "b1", "w2", "b2")}

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "wsp-network/1", "activation": "relu", "dtype": "float64",
                "input_names": list(self.input_names), "output_names": list(self.output_names),
                **{k: v.tolist() for k, v in self.tensors.items()}}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Network:
        expected = {"schema", "activation", "dtype", "input_names", "output_names",
                    "w1", "b1", "w2", "b2"}
        if not isinstance(value, dict) or set(value) != expected or value["schema"] != "wsp-network/1":
            raise ValueError("Unknown network schema or fields")
        if value["activation"] != "relu" or value["dtype"] != "float64":
            raise ValueError("Only explicit ReLU/binary64 semantics are supported")
        if not isinstance(value["input_names"], list) or not isinstance(value["output_names"], list):
            raise ValueError("Serialized semantic labels must be JSON arrays")
        return cls(**{k: value[k] for k in ("w1", "b1", "w2", "b2")},
                   input_names=tuple(value["input_names"]), output_names=tuple(value["output_names"]))

    def hidden(self, inputs: Any) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != self.shape[0] or not np.isfinite(x).all():
            raise ValueError("Inputs must be finite vectors with the declared feature order")
        with np.errstate(over="raise", invalid="raise"):
            return np.maximum(x @ self.w1.T + self.b1, 0)

    def __call__(self, inputs: Any) -> np.ndarray:
        with np.errstate(over="raise", invalid="raise"):
            return self.hidden(inputs) @ self.w2.T + self.b2

    def replace(self, /, **tensors: Any) -> Network:
        if set(tensors) - set(self.tensors):
            raise ValueError("Unknown tensor")
        return Network(**(self.tensors | tensors), input_names=self.input_names,
                       output_names=self.output_names)

    def compatible(self, other: Network) -> bool:
        return (isinstance(other, Network) and self.shape == other.shape and self.input_names == other.input_names
                and self.output_names == other.output_names)


@dataclass(frozen=True, eq=False, init=False)
class InputBox:
    _lower: _TensorData = field(repr=False)
    _upper: _TensorData = field(repr=False)

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        object.__setattr__(self, "_lower", _TensorData.capture(lower, 1))
        object.__setattr__(self, "_upper", _TensorData.capture(upper, 1))
        if self.lower.shape != self.upper.shape or np.any(self.lower > self.upper):
            raise ValueError("A box must have finite ordered bounds of the same dimension")

    @property
    def lower(self) -> np.ndarray:
        return self._lower.array()

    @property
    def upper(self) -> np.ndarray:
        return self._upper.array()

    def to_dict(self) -> dict[str, Any]:
        return {"lower": self.lower.tolist(), "upper": self.upper.tolist()}

    def contains(self, inputs: Any) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim == 0 or x.shape[-1] != len(self.lower):
            raise ValueError("Box and input dimension differ")
        return np.all(np.isfinite(x) & (x >= self.lower) & (x <= self.upper), axis=-1)
