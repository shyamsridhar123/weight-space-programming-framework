"""Compile readout edits for any explicitly supplied frozen representation.

Features may come from an MLP, a CNN, a recurrent model, an embedding, or a
caller-owned feature extractor. The head equation is an adapter obligation.
Exact receipts below concern the supplied feature vectors and stored endpoints;
they do not certify an arbitrary extractor or a floating-point least-squares solve.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from typing import Any, Mapping

import numpy as np

from .artifact import WeightArtifact, valid_tensor_name
from .model import _TensorData, canonical_bytes, digest, finite_binary64
from .program import Program, apply_program
from .structured import StructuredProgram, lower_structured


@dataclass(frozen=True)
class ReadoutHead:
    weight: str
    bias: str | None = None
    feature_id: str = "caller-declared-frozen-features/1"

    def __post_init__(self) -> None:
        if (not valid_tensor_name(self.weight) or
                (self.bias is not None and not valid_tensor_name(self.bias)) or
                self.weight == self.bias or not isinstance(self.feature_id, str) or not self.feature_id.strip()):
            raise ValueError("A readout needs distinct tensor names and a feature identity")

    def to_dict(self) -> dict[str, Any]:
        return {"weight": self.weight, "bias": self.bias, "feature_id": self.feature_id}

    def dimensions(self, artifact: WeightArtifact) -> tuple[int, int]:
        tensors = artifact.tensors
        if self.weight not in tensors or tensors[self.weight].ndim != 2:
            raise ValueError("Readout weight must be a captured [outputs, features] matrix")
        outputs, features = tensors[self.weight].shape
        if self.bias is not None and (self.bias not in tensors or tensors[self.bias].shape != (outputs,)):
            raise ValueError("Readout bias must have one coordinate per output")
        return features, outputs


def _matrix(value: Any, width: int, name: str) -> np.ndarray:
    result = _TensorData.capture(value).array()
    if result.ndim != 2 or not len(result) or result.shape[1] != width:
        raise ValueError(f"{name} must be a nonempty matrix with width {width}")
    return result


def _rms(values: np.ndarray) -> float:
    """Avoid squaring finite extreme residuals before scaling them."""
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale):
        raise ValueError("Readout residuals exceed the numerical evaluator's finite range")
    return 0. if scale == 0. else scale*float(np.sqrt(np.mean((values/scale)**2)))


def _endpoint_delta(base: WeightArtifact, candidate: WeightArtifact, head: ReadoutHead) -> list[list[Fraction]]:
    if not base.compatible(candidate):
        raise ValueError("Readout endpoints must have compatible semantics")
    head.dimensions(base)
    allowed = {head.weight, head.bias}
    before, after = base.tensors, candidate.tensors
    if any(not np.array_equal(before[name], after[name]) for name in before if name not in allowed):
        raise ValueError("A frozen-feature receipt cannot accompany changes outside the declared head")
    delta = [[Fraction.from_float(float(b))-Fraction.from_float(float(a))
              for a, b in zip(row_a, row_b)]
             for row_a, row_b in zip(before[head.weight], after[head.weight])]
    for output, row in enumerate(delta):
        row.append(Fraction(0) if head.bias is None else
                   Fraction.from_float(float(after[head.bias][output]))-
                   Fraction.from_float(float(before[head.bias][output])))
    return delta


def certify_feature_retention(base: WeightArtifact, candidate: WeightArtifact,
                              head: ReadoutHead, anchors: Any) -> dict[str, Any]:
    """Compute exact dyadic residuals on anchors, including bias as a constant.

    For a feature vector in their *augmented* span with coefficients c, the real
    change is bounded by sum(abs(c[k])*residual[k][output]). A bound on coefficients
    is an additional premise; anchors do not automatically cover unseen inputs.
    """
    width, outputs = head.dimensions(base)
    features = _matrix(anchors, width, "Retention features")
    delta = _endpoint_delta(base, candidate, head)
    residuals = []
    for row in features:
        feature = [Fraction.from_float(float(x)) for x in row] + [Fraction(1)]
        residuals.append([abs(sum((w*x for w, x in zip(weights, feature)), Fraction(0))) for weights in delta])
    maximum = [max(row[output] for row in residuals) for output in range(outputs)]
    payload = {
        "schema": "wsp-feature-retention/1", "base_sha256": base.sha256,
        "candidate_sha256": candidate.sha256, "head": head.to_dict(),
        "feature_sha256": digest(features.tolist()), "anchor_count": len(features),
        "absolute_residuals_exact": [[str(value) for value in row] for row in residuals],
        "maximum_residuals_exact": [str(value) for value in maximum],
        "exactly_zero_on_supplied_features": all(value == 0 for value in maximum),
        "semantics": "Real linear head on supplied frozen features; bias is a constant augmented feature",
        "feature_extractor_verified": False, "floating_execution_verified": False,
        "span_extension": "At augmented feature sum(c[k]*anchor[k]), drift <= sum(abs(c[k])*anchor_residual[k])",
    }
    return payload | {"sha256": digest(payload)}


def verify_feature_retention(base: WeightArtifact, candidate: WeightArtifact, head: ReadoutHead,
                             anchors: Any, receipt: dict[str, Any]) -> bool:
    try:
        actual = certify_feature_retention(base, candidate, head, anchors)
        return canonical_bytes(actual) == canonical_bytes(receipt)
    except (ValueError, KeyError, TypeError):
        return False


@dataclass(frozen=True)
class FeatureCompilation:
    program: Program
    proposal: StructuredProgram
    _diagnostics_bytes: bytes
    _retention_bytes: bytes | None

    @property
    def diagnostics(self) -> dict[str, Any]:
        return json.loads(self._diagnostics_bytes)

    @property
    def retention_certificate(self) -> dict[str, Any] | None:
        return None if self._retention_bytes is None else json.loads(self._retention_bytes)


def compile_feature_edit(base: WeightArtifact, head: ReadoutHead, target_features: Any,
                         desired_delta: Any, retain_features: Any | None = None, *,
                         retention: str = "soft", retention_weight: float = 8.0,
                         ridge: float = 0.01, edit_bias: bool = False,
                         nullspace_rtol: float = 1e-12,
                         native_dtypes: Mapping[str, str] | None = None,
                         cast: str = "reject", intent: str = "Compile a frozen-feature output edit") -> FeatureCompilation:
    """Fit none/soft/nullspace retention; verify the *actual* rounded endpoint.

    A numerical nullspace is chosen by an explicit SVD threshold. Full rank yields
    a no-op, not an invented successful edit. Exact endpoint residuals remain in
    the receipt even when the solve reports a tiny floating-point residual.
    """
    width, outputs = head.dimensions(base)
    target = _matrix(target_features, width, "Target features")
    desired = _matrix(desired_delta, outputs, "Desired deltas")
    if len(target) != len(desired):
        raise ValueError("Target features and desired deltas must have equal row counts")
    if type(edit_bias) is not bool or (edit_bias and head.bias is None):
        raise ValueError("Editing bias requires an explicit bias tensor")
    ridge = finite_binary64(ridge)
    retention_weight = finite_binary64(retention_weight)
    nullspace_rtol = finite_binary64(nullspace_rtol)
    if (retention not in ("none", "soft", "nullspace") or ridge <= 0 or
            retention_weight < 0 or not 0 < nullspace_rtol < 1):
        raise ValueError("Invalid retention mode, regularizer, or nullspace tolerance")
    anchors = None if retain_features is None else _matrix(retain_features, width, "Retention features")
    if retention != "none" and anchors is None:
        raise ValueError("Retention mode requires explicit retention features")

    def augment(matrix: np.ndarray) -> np.ndarray:
        return np.column_stack((matrix, np.ones(len(matrix)))) if edit_bias else matrix

    target_design = augment(target)
    retain_design = None if anchors is None else augment(anchors)
    size = target_design.shape[1]
    basis = np.eye(size)
    retention_rank, cutoff = 0, None
    if retention == "nullspace":
        assert retain_design is not None
        _, singular, vh = np.linalg.svd(retain_design, full_matrices=True)
        cutoff = nullspace_rtol * (float(singular[0]) if len(singular) else 0.0)
        retention_rank = int(np.count_nonzero(singular > cutoff))
        basis = vh[retention_rank:].T
    if basis.shape[1] == 0:
        update = np.zeros((size, outputs))
        fit_rank, condition = 0, None
    else:
        matrices = [target_design @ basis / np.sqrt(len(target)), np.sqrt(ridge)*np.eye(basis.shape[1])]
        responses = [desired / np.sqrt(len(target)), np.zeros((basis.shape[1], outputs))]
        if retention == "soft":
            assert retain_design is not None
            matrices.append(retain_design @ basis * np.sqrt(retention_weight/len(retain_design)))
            responses.append(np.zeros((len(retain_design), outputs)))
        solution, _, fit_rank, singular = np.linalg.lstsq(np.vstack(matrices), np.vstack(responses), rcond=None)
        update = basis @ solution
        condition = float(singular[0]/singular[-1])
        if not np.isfinite(condition):
            condition = None
    commands = [{"op": "add", "tensor": head.weight, "values": update[:width].T.tolist()}]
    if edit_bias:
        commands.append({"op": "add", "tensor": head.bias, "values": update[-1].tolist()})
    proposal = StructuredProgram(base.sha256, commands, intent)
    lowered = lower_structured(base, proposal, native_dtypes=native_dtypes, cast=cast)
    candidate = apply_program(base, lowered.program)
    endpoint = _endpoint_delta(base, candidate, head)
    realized = np.array([[float(x) for x in row] for row in endpoint])

    def effect(features: np.ndarray) -> np.ndarray:
        return np.column_stack((features, np.ones(len(features)))) @ realized.T

    certificate = None if anchors is None else certify_feature_retention(base, candidate, head, anchors)
    diagnostics = {
        "compiler": "generic-frozen-feature-readout/1", "head": head.to_dict(),
        "retention": retention, "retention_weight": retention_weight, "ridge": ridge,
        "nullspace_rtol": nullspace_rtol, "singular_cutoff": cutoff,
        "retention_rank": retention_rank if retention == "nullspace" else None,
        "nullspace_dimension": int(basis.shape[1]) if retention == "nullspace" else None,
        "fit_rank": int(fit_rank), "condition_number": condition,
        "target_rows": len(target), "retention_rows": 0 if anchors is None else len(anchors),
        "target_feature_sha256": digest(target.tolist()), "desired_delta_sha256": digest(desired.tolist()),
        "retention_feature_sha256": None if anchors is None else digest(anchors.tolist()),
        "target_rmse_after_projection": _rms(effect(target)-desired),
        "retention_rmse_after_projection": None if anchors is None else _rms(effect(anchors)),
        "edit_cells": len(lowered.program.edits), "no_op": not lowered.program.edits,
        "lowering": lowered.receipt, "behavioral_status": "supplied_features_only",
        "optimizer_verified": False, "feature_extractor_verified": False,
    }
    return FeatureCompilation(lowered.program, proposal, canonical_bytes(diagnostics),
                              None if certificate is None else canonical_bytes(certificate))
