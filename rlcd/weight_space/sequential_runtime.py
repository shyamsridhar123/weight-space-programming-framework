"""Owned restoration for one explicit, data-only sequential execution scope.

This restores the declared artifact, not an arbitrary original model. Execution
is empirical NumPy CPU binary64 with serial multiply/add and finite-intermediate
checks; it provides neither a floating-point theorem nor evidence of task utility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np

from .artifact import TensorArtifact
from .evidence import numerical_environment
from .model import _TensorData, canonical_bytes, digest
from .sequential import validate_sequential
from .structured import ExecutionSpec, _execution_layers, _native_execution


_EXECUTION = ExecutionSpec(
    backend="numpy", dtype="float64", device="cpu", accumulation="left_to_right",
    endpoint_cast="reject", error_model="empirical",
)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SequentialRuntime:
    """Immutable owned state, detached snapshots/receipts, and explicit forward."""

    _artifact: TensorArtifact = field(repr=False)
    _execution: ExecutionSpec
    _receipt_bytes: bytes = field(repr=False)

    def __init__(self, endpoint: TensorArtifact, *, expected_endpoint_sha256: str,
                 execution: ExecutionSpec | dict[str, Any]):
        if type(endpoint) is not TensorArtifact:
            raise ValueError("Restoration requires a plain TensorArtifact")
        if type(execution) is ExecutionSpec:
            execution = execution.to_dict()
        if type(execution) is not dict:
            raise ValueError("Supply an ExecutionSpec or an exact execution declaration")
        if (any(type(key) is not str or type(value) is not str for key, value in execution.items())
                or canonical_bytes(execution) != canonical_bytes(_EXECUTION.to_dict())):
            raise ValueError("Unsupported sequential restoration execution scope")
        spec = _EXECUTION
        if type(expected_endpoint_sha256) is not str or endpoint.sha256 != expected_endpoint_sha256:
            raise ValueError("Expected endpoint SHA-256 does not match the retained endpoint")

        owned = TensorArtifact(endpoint.architecture, endpoint.tensors, endpoint.metadata)
        layers = validate_sequential(owned)
        if any(layer["activation"] not in ("identity", "relu", "abs") for layer in layers):
            raise ValueError("Restoration supports only identity, relu, and abs activations")
        object.__setattr__(self, "_artifact", owned)
        object.__setattr__(self, "_execution", spec)
        restored = self.snapshot()
        if restored.sha256 != expected_endpoint_sha256:
            raise ValueError("Restored snapshot differs from the retained endpoint")
        metadata, tensors = restored.metadata, restored.tensors
        payload = {
            "schema": "wsp-sequential-runtime-restoration/1", "status": "restored",
            "endpoint_sha256": expected_endpoint_sha256,
            "restored_snapshot_sha256": restored.sha256,
            "architecture": restored.architecture, "execution": spec.to_dict(),
            "layers": metadata["layers"],
            "tensors": [{"name": name, "shape": list(tensors[name].shape),
                         "dtype": "float64", "device": "cpu"}
                        for layer in layers for name in (layer["weight"], layer["bias"])],
            "input_names": metadata["input_names"], "output_names": metadata["output_names"],
            "alias_groups": [],
            "operator_order": "left-to-right dot product, then explicit bias, then activation",
            "environment": numerical_environment(spec.backend, spec.device),
            "scope": "Owned declared sequential endpoint; empirical execution in the recorded environment; "
                     "original-module restoration, formal floating-point proofs, and task utility are unevaluated",
        }
        object.__setattr__(self, "_receipt_bytes", canonical_bytes(payload | {"sha256": digest(payload)}))

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(self._artifact.metadata["input_names"])

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self._artifact.metadata["output_names"])

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_bytes)

    def snapshot(self) -> TensorArtifact:
        """Capture the runtime's actual owned state into another independent artifact."""
        return TensorArtifact(self._artifact.architecture, self._artifact.tensors, self._artifact.metadata)

    def forward(self, inputs: Any) -> np.ndarray:
        """Execute a finite vector or batch; preserve its vector/batch output shape.

        Input conversion must be exact binary64. Each call rejects nonfinite
        intermediates before activation; underflow follows NumPy's finite result.
        """
        samples = _TensorData.capture(inputs).array()
        if samples.ndim not in (1, 2) or samples.shape[-1] != len(self.input_names):
            raise ValueError("Supply finite vectors with the declared input width")
        vector = samples.ndim == 1
        if vector:
            samples = samples.reshape(1, -1)
        try:
            result = _native_execution(_execution_layers(self._artifact), samples, self._execution,
                                       finite_intermediates=True)
        except FloatingPointError as error:
            raise ValueError("Sequential runtime produced nonfinite intermediates") from error
        return result[0] if vector else result


def restore_sequential_runtime(endpoint: TensorArtifact, *, expected_endpoint_sha256: str,
                               execution: ExecutionSpec | dict[str, Any]) -> SequentialRuntime:
    """Restore only the exact retained endpoint under the supported execution scope."""
    return SequentialRuntime(endpoint, expected_endpoint_sha256=expected_endpoint_sha256,
                             execution=execution)
