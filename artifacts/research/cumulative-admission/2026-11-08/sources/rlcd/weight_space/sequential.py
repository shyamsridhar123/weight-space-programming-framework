"""Arbitrary-depth affine/activation execution and exact real drift certificates.

This is a second explicit semantic backend, not an interpreter for arbitrary
Python or general computation graphs. The certified activations are identity,
ReLU, and absolute value. Tanh is executable but has no registered certificate
in this backend, keeping the implementation/proof coverage boundary explicit.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Sequence

import numpy as np

from .artifact import TensorArtifact
from .model import InputBox, canonical_bytes, digest, _labels


SEQUENTIAL_ARCHITECTURE = "sequential-affine-activation/1"
EXECUTABLE_ACTIVATIONS = ("identity", "relu", "abs", "tanh")
CERTIFIED_ACTIVATIONS = ("identity", "relu", "abs")


def sequential_artifact(weights: Sequence[Any], biases: Sequence[Any], activations: Sequence[str], *,
                        input_names: Sequence[str] = (), output_names: Sequence[str] = ()) -> TensorArtifact:
    if not len(weights) or len(weights) != len(biases) or len(weights) != len(activations):
        raise ValueError("Every nonempty layer needs weights, bias, and an activation")
    tensors, layers = {}, []
    for i, (weight, bias, activation) in enumerate(zip(weights, biases, activations)):
        wn, bn = f"layers.{i}.weight", f"layers.{i}.bias"
        tensors[wn], tensors[bn] = weight, bias
        layers.append({"weight": wn, "bias": bn, "activation": activation})
    first, last = np.asarray(weights[0]), np.asarray(weights[-1])
    if first.ndim != 2 or last.ndim != 2:
        raise ValueError("Affine weights must be matrices")
    result = TensorArtifact(SEQUENTIAL_ARCHITECTURE, tensors, {
        "layers": layers, "input_names": list(_labels(tuple(input_names), first.shape[1], "x")),
        "output_names": list(_labels(tuple(output_names), last.shape[0], "y")),
    })
    validate_sequential(result)
    return result


def validate_sequential(artifact: TensorArtifact) -> list[dict[str, str]]:
    if not isinstance(artifact, TensorArtifact) or artifact.architecture != SEQUENTIAL_ARCHITECTURE:
        raise ValueError("Artifact does not declare sequential affine/activation semantics")
    metadata, tensors = artifact.metadata, artifact.tensors
    if set(metadata) != {"layers", "input_names", "output_names"}:
        raise ValueError("Unknown sequential metadata")
    layers = metadata["layers"]
    if not isinstance(layers, list) or not layers:
        raise ValueError("A sequential model needs a nonempty layer list")
    used, previous = set(), None
    for layer in layers:
        if not isinstance(layer, dict) or set(layer) != {"weight", "bias", "activation"}:
            raise ValueError("Unknown layer fields")
        w, b, activation = layer["weight"], layer["bias"], layer["activation"]
        if not isinstance(w, str) or not isinstance(b, str) or w == b:
            raise ValueError("Layer tensor names must be distinct strings")
        if w in used or b in used or w not in tensors or b not in tensors:
            raise ValueError("Every layer must name its own captured weight and bias")
        weight, bias = tensors[w], tensors[b]
        if (weight.ndim != 2 or bias.shape != (weight.shape[0],) or
                (previous is not None and weight.shape[1] != previous)):
            raise ValueError("Incompatible sequential layer dimensions")
        if activation not in EXECUTABLE_ACTIVATIONS:
            raise ValueError("Unsupported activation")
        previous = weight.shape[0]
        used.update((w, b))
    if used != set(tensors):
        raise ValueError("Uninterpreted tensors are not allowed in this semantic backend")
    for key, width, prefix in (("input_names", tensors[layers[0]["weight"]].shape[1], "x"),
                               ("output_names", previous, "y")):
        if not isinstance(metadata[key], list) or len(metadata[key]) != width:
            raise ValueError("Explicit input/output labels must match dimensions")
        _labels(tuple(metadata[key]), width, prefix)
    return layers


def sequential_forward(artifact: TensorArtifact, inputs: Any, *,
                        through_layer: int | None = None) -> np.ndarray:
    layers = validate_sequential(artifact)
    if through_layer is not None and (type(through_layer) is not int or not 0 <= through_layer <= len(layers)):
        raise ValueError("through_layer must count executed layers from zero to model depth")
    tensors = artifact.tensors
    values = np.asarray(inputs, dtype=np.float64)
    if (values.ndim not in (1, 2) or values.shape[-1] != tensors[layers[0]["weight"]].shape[1]
            or not np.isfinite(values).all() or values.size == 0):
        raise ValueError("Supply finite vectors with the declared input width")
    with np.errstate(over="raise", invalid="raise"):
        for layer in layers[:through_layer]:
            values = values @ tensors[layer["weight"]].T + tensors[layer["bias"]]
            activation = layer["activation"]
            if activation == "relu":
                values = np.maximum(0, values)
            elif activation == "abs":
                values = np.abs(values)
            elif activation == "tanh":
                values = np.tanh(values)
    if not np.isfinite(values).all():
        raise ValueError("Sequential execution produced nonfinite values")
    return values


def certify_sequential_drift(base: TensorArtifact, candidate: TensorArtifact,
                              box: InputBox) -> dict[str, Any]:
    """Sound componentwise recurrence in exact rational arithmetic.

    r'_i = |b_i| + sum_j |W_ij| r_j
    d'_i = |db_i| + sum_j (|dW_ij| r_j + |W'_ij| d_j)

    Both paths use the same zero-preserving, one-Lipschitz activation. Depth and
    width are arbitrary finite positive sizes; interval growth can be loose.
    """
    if not base.compatible(candidate):
        raise ValueError("Sequential certification requires identical architecture/interface metadata")
    layers = validate_sequential(base)
    validate_sequential(candidate)
    if any(layer["activation"] not in CERTIFIED_ACTIVATIONS for layer in layers):
        raise ValueError("No certificate registered for one or more activations")
    before, after = base.tensors, candidate.tensors
    width = before[layers[0]["weight"]].shape[1]
    if len(box.lower) != width:
        raise ValueError("Box width differs from the declared input width")
    q = lambda x: Fraction.from_float(float(x))
    radii = [max(abs(q(lo)), abs(q(hi))) for lo, hi in zip(box.lower, box.upper)]
    drift = [Fraction(0) for _ in radii]
    intermediate = []
    for layer in layers:
        w, b = before[layer["weight"]], before[layer["bias"]]
        wp, bp = after[layer["weight"]], after[layer["bias"]]
        next_radii, next_drift = [], []
        for i in range(len(b)):
            next_radii.append(abs(q(b[i])) + sum((abs(q(w[i, j]))*radii[j] for j in range(len(radii))), Fraction(0)))
            next_drift.append(abs(q(bp[i])-q(b[i])) + sum((
                abs(q(wp[i, j])-q(w[i, j]))*radii[j] + abs(q(wp[i, j]))*drift[j]
                for j in range(len(radii))), Fraction(0)))
        radii, drift = next_radii, next_drift
        intermediate.append({"layer": layer, "base_radius": [str(v) for v in radii],
                             "drift": [str(v) for v in drift]})
    payload = {
        "schema": "wsp-sequential-drift/1", "base_artifact_sha256": base.sha256,
        "candidate_artifact_sha256": candidate.sha256, "architecture": SEQUENTIAL_ARCHITECTURE,
        "domain": box.to_dict(), "layers": intermediate, "output_bounds": [str(v) for v in drift],
        "arithmetic": "Exact rationals lifted from stored binary64 endpoints",
        "scope": "All real inputs in the declared box; shared certified activation per layer",
        "implementation_verified": False, "floating_execution_verified": False,
        "reward_verified": False,
    }
    return payload | {"sha256": digest(payload)}


def verify_sequential_certificate(base: TensorArtifact, candidate: TensorArtifact,
                                  receipt: dict[str, Any]) -> bool:
    try:
        box = InputBox(**receipt["domain"])
        return canonical_bytes(certify_sequential_drift(base, candidate, box)) == canonical_bytes(receipt)
    except (ValueError, KeyError, TypeError):
        return False


def register_sequential(registry: Any) -> None:
    from .semantics import SemanticsAdapter
    def certificate(base: TensorArtifact, candidate: TensorArtifact, box: InputBox) -> dict[str, Any]:
        return {"schema": "wsp-adapter-certificate/1", "base_artifact_sha256": base.sha256,
                "candidate_artifact_sha256": candidate.sha256, "architecture": SEQUENTIAL_ARCHITECTURE,
                "certificate": certify_sequential_drift(base, candidate, box)}
    registry.register(SemanticsAdapter(
        SEQUENTIAL_ARCHITECTURE, sequential_forward, certificate,
        "Arbitrary finite affine/identity/ReLU/abs chains over reals; no float, task, or arbitrary-code guarantee",
    ))
