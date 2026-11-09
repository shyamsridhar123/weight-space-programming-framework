"""Explicit architecture adapters; artifact data never imports or executes code."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from types import CodeType
from typing import Any, Callable

import numpy as np

from .artifact import TensorArtifact
from .contracts import certify_drift
from .model import InputBox, Network, canonical_bytes, digest

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
    version: str = "1"
    deterministic: bool = True

    def __post_init__(self) -> None:
        if (not isinstance(self.architecture, str) or not self.architecture.strip()
                or not isinstance(self.version, str) or not self.version.strip()
                or type(self.deterministic) is not bool or not callable(self.evaluate)
                or (self.certify is not None and not callable(self.certify))):
            raise ValueError("Adapters need explicit architecture/version and executable caller-owned callbacks")

    def descriptor(self) -> dict[str, Any]:
        """Bind reviewed callback bytes/configuration, without executing data hooks.

        This is correspondence accounting, not an attestation of Python globals,
        a sandbox, or a proof that the implementation realizes its semantics.
        """
        def callback(function: Any) -> Any:
            if function is None:
                return None
            def code(value: CodeType) -> dict[str, Any]:
                return {"bytecode": value.co_code.hex(), "names": list(value.co_names),
                        "constants": [code(x) if isinstance(x, CodeType) else
                                      x.hex() if isinstance(x, bytes) else x
                                      for x in value.co_consts]}
            if not inspect.isfunction(function):
                raise ValueError("Callback implementation/configuration cannot be frozen")
            source = inspect.getsourcefile(function)
            return {"module": function.__module__, "code": code(function.__code__),
                    "source_sha256": None if source is None or not Path(source).is_file() else
                                     hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                    "defaults": function.__defaults__, "keyword_defaults": function.__kwdefaults__,
                    "closure": [cell.cell_contents for cell in (function.__closure__ or ())]}
        try:
            implementation = digest({"evaluate": callback(self.evaluate), "certify": callback(self.certify)})
        except (TypeError, ValueError, OSError):
            implementation = None
        return {"architecture": self.architecture, "version": self.version,
                "deterministic": self.deterministic, "implementation_sha256": implementation}


class SemanticsRegistry:
    """Caller-owned registration, with no fallback between incompatible semantics."""

    def __init__(self) -> None:
        self._adapters: dict[str, SemanticsAdapter] = {}

    def register(self, adapter: SemanticsAdapter) -> None:
        if not isinstance(adapter, SemanticsAdapter):
            raise ValueError("Register an explicit SemanticsAdapter")
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

    def verify_certificate(self, base: TensorArtifact, candidate: TensorArtifact,
                           box: InputBox, receipt: dict[str, Any]) -> bool:
        try:
            return canonical_bytes(self.certify(base, candidate, box)) == canonical_bytes(receipt)
        except (TypeError, ValueError, KeyError, OverflowError):
            return False


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
    from .sequential import register_sequential
    register_sequential(registry)
    return registry
