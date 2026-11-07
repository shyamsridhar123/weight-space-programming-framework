"""Explicit architecture adapters; artifact data never imports or executes code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .artifact import TensorArtifact
from .contracts import certify_drift
from .model import InputBox, Network

RELU_ARCHITECTURE = "dense-relu-dense/1"


def to_tensor_artifact(model: Network) -> TensorArtifact:
    return TensorArtifact(RELU_ARCHITECTURE, model.tensors,
                          {"input_names": list(model.input_names), "output_names": list(model.output_names)})


def as_relu_network(artifact: TensorArtifact) -> Network:
    if (artifact.architecture != RELU_ARCHITECTURE
            or set(artifact.tensors) != {"w1", "b1", "w2", "b2"}
            or set(artifact.metadata) != {"input_names", "output_names"}):
        raise ValueError("Artifact does not declare the supported two-layer ReLU semantics")
    return Network(**dict(artifact.tensors), **artifact.metadata)


@dataclass(frozen=True)
class SemanticsAdapter:
    architecture: str
    evaluate: Callable[[TensorArtifact, Any], np.ndarray]
    certify: Callable[[TensorArtifact, TensorArtifact, InputBox], dict[str, Any]] | None = None
    certificate_scope: str = "No formal behavioral certificate registered"


class SemanticsRegistry:
    """Caller-owned registration, with no fallback between incompatible semantics."""

    def __init__(self) -> None:
        self._adapters: dict[str, SemanticsAdapter] = {}

    def register(self, adapter: SemanticsAdapter) -> None:
        if adapter.architecture in self._adapters:
            raise ValueError("An architecture adapter is already registered")
        self._adapters[adapter.architecture] = adapter

    def adapter(self, artifact: TensorArtifact) -> SemanticsAdapter:
        try:
            return self._adapters[artifact.architecture]
        except KeyError as error:
            raise ValueError(f"No execution adapter registered for {artifact.architecture}") from error

    def evaluate(self, artifact: TensorArtifact, inputs: Any) -> np.ndarray:
        result = np.asarray(self.adapter(artifact).evaluate(artifact, inputs), dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Execution adapter returned nonfinite outputs")
        return result

    def certify(self, base: TensorArtifact, candidate: TensorArtifact, box: InputBox) -> dict[str, Any]:
        if not base.compatible(candidate):
            raise ValueError("Certificate requires compatible architecture and interface metadata")
        adapter = self.adapter(base)
        if adapter.certify is None:
            raise ValueError(f"No behavioral proof backend for {adapter.architecture}")
        return adapter.certify(base, candidate, box)


def default_registry() -> SemanticsRegistry:
    registry = SemanticsRegistry()
    def certificate(base: TensorArtifact, candidate: TensorArtifact, box: InputBox) -> dict[str, Any]:
        # Both identities are retained: the specialized receipt identifies the
        # normalized semantic network, while the envelope identifies its artifact.
        return {"schema": "wsp-adapter-certificate/1", "base_artifact_sha256": base.sha256,
                "candidate_artifact_sha256": candidate.sha256, "architecture": RELU_ARCHITECTURE,
                "certificate": certify_drift(as_relu_network(base), as_relu_network(candidate), box)}
    registry.register(SemanticsAdapter(
        RELU_ARCHITECTURE, lambda artifact, inputs: as_relu_network(artifact)(inputs), certificate,
        "Real arithmetic, bounded input box, finite two-layer ReLU network; no float/reward guarantee"))
    return registry
