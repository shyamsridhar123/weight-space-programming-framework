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
from operator import add, mul, sub, truediv
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


_EXACT_WORK_LIMIT = 250_000
_EXACT_RATIONAL_BIT_LIMIT = 4096


def _editable_design(features: np.ndarray, edit_bias: bool) -> np.ndarray:
    """Include a constant coordinate only when the corresponding bias is writable."""
    return np.column_stack((features, np.ones(len(features)))) if edit_bias else features


def _exact_feature_constraints(target: np.ndarray, retain: np.ndarray, desired: np.ndarray | None,
                               work_limit: int) -> dict[str, Any]:
    """Bounded rational consistency and minimum norm for R X=0, T X=D.

    Each captured scalar and Fraction operation consumes work. Every stored
    rational has bounded numerator/denominator length; a binary operation on
    checked operands creates at most twice that bit length plus one before its
    result is checked. Sparse row combinations refer to original constraint rows,
    so a conflict can be checked without trusting this elimination procedure.
    """
    width, outputs = target.shape[1], 0 if desired is None else desired.shape[1]
    result: dict[str, Any] = {
        "consistency_status": "not_requested" if desired is None else "unresolved",
        "per_output_consistency": ["unresolved"]*outputs,
        "minimum_norm_status": "not_requested" if desired is None else "unresolved",
        "per_output_minimum_norm_status": ["unresolved"]*outputs,
        "minimum_squared_l2_norm_exact": [None]*outputs,
        "minimum_squared_frobenius_norm_exact": None,
        "protected_rank_exact": 0 if len(retain) == 0 else None,
        "protected_nullity_exact": width if len(retain) == 0 else None,
        "protected_only_zero_update": width == 0 if len(retain) == 0 else None,
        "constraint_rank_exact": None, "constraint_rank_lower_bound": 0,
        "independent_constraint_rows": [], "completed_constraint_rows": 0,
        "conflict_witnesses": [], "consistency_limit_reason": None,
        "minimum_norm_limit_reason": None, "work_limit": work_limit,
        "rational_bit_limit": _EXACT_RATIONAL_BIT_LIMIT,
        "consistency_work_used": 0, "work_used": 0,
        "row_order": "protected rows followed by target rows",
        "scope": "Unconstrained real/rational edits to writable head coordinates; not native representability, parameter budgets, edit counts, or alternative argmax goals",
    }
    if desired is None:
        return result

    class LimitReached(Exception):
        pass

    work = 0

    def charge() -> None:
        nonlocal work
        if work >= work_limit:
            raise LimitReached("work_limit")
        work += 1

    def checked(value: Fraction) -> Fraction:
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > _EXACT_RATIONAL_BIT_LIMIT:
            raise LimitReached("rational_bit_limit")
        return value

    def capture(value: float) -> Fraction:
        charge()
        return checked(Fraction.from_float(float(value)))

    def calculate(operation: Any, left: Fraction, right: Fraction) -> Fraction:
        charge()
        return checked(operation(left, right))

    pivots: list[int] = []
    rows: list[list[Fraction]] = []
    responses: list[list[Fraction]] = []
    combinations: list[dict[int, Fraction]] = []
    original_rows: list[list[Fraction]] = []
    original_responses: list[list[Fraction]] = []
    statuses = result["per_output_consistency"]
    complete = False
    try:
        for index in range(len(retain)+len(target)):
            protected = index < len(retain)
            row = [capture(value) for value in (retain[index] if protected else target[index-len(retain)])]
            response = [capture(0. if protected else value) for value in
                        (np.zeros(outputs) if protected else desired[index-len(retain)])]
            original_row, original_response = row.copy(), response.copy()
            combination = {index: Fraction(1)}
            for pivot, known, answer, coefficients in zip(pivots, rows, responses, combinations):
                factor = row[pivot]
                if factor == 0:
                    continue
                row = [calculate(sub, value, calculate(mul, factor, other)) for value, other in zip(row, known)]
                response = [calculate(sub, value, calculate(mul, factor, other))
                            for value, other in zip(response, answer)]
                for source, coefficient in coefficients.items():
                    value = calculate(sub, combination.get(source, Fraction(0)), calculate(mul, factor, coefficient))
                    if value:
                        combination[source] = value
                    else:
                        combination.pop(source, None)
            pivot = next((column for column, value in enumerate(row) if value), None)
            if pivot is None:
                conflicts = [output for output, value in enumerate(response)
                             if value and statuses[output] != "infeasible"]
                if conflicts:
                    sources = sorted(combination)
                    result["conflict_witnesses"].append({
                        "row_indices": sources,
                        "coefficients_exact": [str(combination[source]) for source in sources],
                        "output_indices": conflicts,
                        "response_exact": [str(response[output]) for output in conflicts],
                    })
                    for output in conflicts:
                        statuses[output] = "infeasible"
            else:
                divisor = row[pivot]
                normalized = [calculate(truediv, value, divisor) for value in row]
                answer = [calculate(truediv, value, divisor) for value in response]
                coefficients = {source: calculate(truediv, value, divisor) for source, value in combination.items()}
                pivots.append(pivot)
                rows.append(normalized)
                responses.append(answer)
                combinations.append(coefficients)
                original_rows.append(original_row)
                original_responses.append(original_response)
                result["independent_constraint_rows"].append(index)
            result["completed_constraint_rows"] = index+1
            if index+1 == len(retain):
                result["protected_rank_exact"] = len(rows)
                result["protected_nullity_exact"] = width-len(rows)
                result["protected_only_zero_update"] = len(rows) == width
        complete = True
        result["constraint_rank_exact"] = len(rows)
        statuses[:] = ["infeasible" if status == "infeasible" else "feasible" for status in statuses]
    except LimitReached as exc:
        result["consistency_limit_reason"] = str(exc)
    result["constraint_rank_lower_bound"] = len(rows)
    result["consistency_work_used"] = work
    result["consistency_status"] = ("infeasible" if "infeasible" in statuses else
                                    "feasible" if complete else "unresolved")

    norm_statuses = result["per_output_minimum_norm_status"]
    norms: list[Fraction | None] = [None]*outputs
    for output, status in enumerate(statuses):
        if status == "infeasible":
            norm_statuses[output] = "not_applicable"
        elif status == "feasible" and all(row[output] == 0 for row in original_responses):
            norms[output], norm_statuses[output] = Fraction(0), "complete"
    pending = [output for output, status in enumerate(statuses)
               if status == "feasible" and norms[output] is None]
    if complete:
        try:
            if pending:
                # Independent original rows M give the minimum-norm edit
                # M.T (M M.T)^-1 Y_M. Factor its positive definite Gram matrix
                # once, then retain each completed output norm independently.
                gram: list[list[Fraction]] = []
                for row in original_rows:
                    values = []
                    for other in original_rows:
                        value = Fraction(0)
                        for left, right in zip(row, other):
                            value = calculate(add, value, calculate(mul, left, right))
                        values.append(value)
                    gram.append(values)
                for pivot in range(len(gram)):
                    for row in range(pivot+1, len(gram)):
                        factor = calculate(truediv, gram[row][pivot], gram[pivot][pivot])
                        gram[row][pivot] = factor
                        for column in range(pivot+1, len(gram)):
                            gram[row][column] = calculate(sub, gram[row][column], calculate(mul, factor, gram[pivot][column]))
                for output in pending:
                    answer = [row[output] for row in original_responses]
                    for row in range(len(gram)):
                        for column in range(row):
                            answer[row] = calculate(sub, answer[row], calculate(mul, gram[row][column], answer[column]))
                    for row in reversed(range(len(gram))):
                        for column in range(row+1, len(gram)):
                            answer[row] = calculate(sub, answer[row], calculate(mul, gram[row][column], answer[column]))
                        answer[row] = calculate(truediv, answer[row], gram[row][row])
                    value = Fraction(0)
                    for original, solved in zip(original_responses, answer):
                        value = calculate(add, value, calculate(mul, original[output], solved))
                    norms[output], norm_statuses[output] = value, "complete"
            if all(status == "feasible" for status in statuses):
                total = Fraction(0)
                for value in norms:
                    assert value is not None
                    total = calculate(add, total, value)
                result["minimum_squared_frobenius_norm_exact"] = str(total)
        except LimitReached as exc:
            result["minimum_norm_limit_reason"] = str(exc)
    result["minimum_squared_l2_norm_exact"] = [None if value is None else str(value) for value in norms]
    result["minimum_norm_status"] = ("unresolved" if "unresolved" in norm_statuses or result["minimum_norm_limit_reason"] else
                                      "complete" if "complete" in norm_statuses else "not_applicable")
    result["work_used"] = work
    return result


def _analyze_feature_constraints(target_design: np.ndarray, retain_design: np.ndarray | None,
                                 desired_delta: np.ndarray | None = None, *,
                                 nullspace_rtol: float = 1e-12,
                                 exact_work_limit: int = _EXACT_WORK_LIMIT) -> tuple[np.ndarray, dict[str, Any]]:
    """Analyze supplied writable coordinates, independently of any ridge fit."""
    if (target_design.ndim != 2 or not len(target_design) or not np.isfinite(target_design).all()
            or (retain_design is not None and (retain_design.ndim != 2
                or retain_design.shape[1] != target_design.shape[1] or not np.isfinite(retain_design).all()))
            or (desired_delta is not None and (desired_delta.ndim != 2
                or desired_delta.shape[0] != len(target_design) or not desired_delta.shape[1]
                or not np.isfinite(desired_delta).all()))
            or not 0 < nullspace_rtol < 1 or type(exact_work_limit) is not int or exact_work_limit < 0):
        raise ValueError("Constraint analysis needs finite aligned designs, deltas, and valid deterministic limits")
    retained = np.empty((0, target_design.shape[1])) if retain_design is None else retain_design

    def analysis_scale(matrix: np.ndarray) -> float:
        # Leave ordinary spectra unchanged. Scale before projection/SVD when
        # a Frobenius upper bound approaches the binary64 range limit.
        largest = float(np.max(np.abs(matrix), initial=0.))
        bound = 2*max(1., float(np.sqrt(matrix.size)))
        return largest if largest > np.finfo(float).max/bound else 1.

    def finite(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    def norm(values: np.ndarray) -> float | None:
        scale = float(np.max(np.abs(values), initial=0.))
        if not np.isfinite(scale):
            return None
        value = 0. if scale == 0. else scale*float(np.linalg.norm(values/scale))
        return finite(value)

    def squared_magnitude(value: float | None) -> float | None:
        if value is None:
            return None
        squared = value*value
        return None if value != 0. and squared == 0. else finite(squared)

    def product_underflow(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # Conservatively reject affected outputs if any nonzero dot-product
        # term is smaller than binary64's smallest representable magnitude.
        minimum = np.min(np.where(left == 0., np.inf, np.abs(left)), axis=0, initial=np.inf)
        threshold = np.finfo(float).smallest_subnormal/minimum
        return np.any((right != 0.) & (np.abs(right) < threshold[:, None]), axis=0)

    def spectrum(values: np.ndarray, scale: float) -> dict[str, Any]:
        cutoff = nullspace_rtol*(float(values[0]) if len(values) else 0.)
        rank = int(np.count_nonzero(values > cutoff))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            physical = values*scale
            condition = None if not len(values) or values[-1] == 0 else finite(values[0]/values[-1])
            retained_condition = None if rank == 0 else finite(values[0]/values[rank-1])
        return {"singular_values": [finite(value) for value in physical],
                "scaled_singular_values": values.tolist(), "singular_values_scale": scale,
                "rank": rank, "absolute_cutoff": finite(cutoff*scale),
                "condition_number": condition, "retained_condition_number": retained_condition}

    retained_scale = analysis_scale(retained)
    _, singular, vh = np.linalg.svd(retained/retained_scale,
                                     full_matrices=(retained.shape[0] < retained.shape[1]))
    protected = spectrum(singular, retained_scale)
    protected["nullity"] = target_design.shape[1]-protected["rank"]
    basis = vh[protected["rank"]:].T
    # Project before range scaling: a large protected coordinate must not erase
    # a small writable coordinate. If this exceeds binary64, report a limit.
    with np.errstate(over="ignore", invalid="ignore"):
        physical_projected = target_design @ basis
    projected_limit = None
    if not np.isfinite(physical_projected).all():
        projected_limit = "projected_matrix_out_of_range"
    else:
        projected_scale = analysis_scale(physical_projected)
        projected = physical_projected/projected_scale
        if np.any((physical_projected != 0.) & (projected == 0.)):
            projected_limit = "projected_matrix_scaling_underflow"
    if projected_limit is None:
        u, singular, vh = np.linalg.svd(projected, full_matrices=False)
        target_spectrum = spectrum(singular, projected_scale)
    else:
        target_spectrum = {"singular_values": None, "scaled_singular_values": None,
                           "singular_values_scale": None, "rank": None, "absolute_cutoff": None,
                           "condition_number": None, "retained_condition_number": None}
    target_spectrum["numerical_limit_reason"] = projected_limit
    energies, fractions = [], []
    for row, projected_row in zip(target_design, physical_projected):
        scale = float(np.max(np.abs(row), initial=0.))
        if scale == 0.:
            energies.append(0.)
            fractions.append(0.)
            continue
        length = float(np.linalg.norm(row/scale))
        magnitude = norm(projected_row)
        energies.append(squared_magnitude(magnitude))
        floor = float(np.sqrt(np.finfo(float).tiny))
        fraction = (None if magnitude is None else magnitude/scale/length if scale >= floor/length
                    else magnitude/floor)
        fractions.append(None if magnitude != 0. and fraction == 0. else squared_magnitude(fraction))
    target_spectrum["row_projected_energy"] = energies
    target_spectrum["row_projected_energy_fraction"] = fractions
    unregularized = None
    if desired_delta is not None:
        outputs = desired_delta.shape[1]
        unregularized = {
            "target_residual_rmse": None, "target_residual_l2_per_output": [None]*outputs,
            "edit_l2_norm_per_output": [None]*outputs, "edit_frobenius_norm": None,
            "per_output_numerical_limit_reason": [projected_limit]*outputs,
            "semantics": "Unregularized truncated-SVD fit in the numerical protected nullspace; no endpoint rounding",
        }
        if projected_limit is None:
            rank = target_spectrum["rank"]
            scales = np.max(np.abs(desired_delta), axis=0)
            normalized = desired_delta/np.where(scales == 0., 1., scales)
            normalization_lost = np.any((desired_delta != 0.) & (normalized == 0.), axis=0)
            projection_lost = product_underflow(u[:, :rank].T, normalized)
            coordinates = u[:, :rank].T @ normalized
            projection_lost |= product_underflow(u[:, :rank], coordinates)
            lost = normalization_lost | projection_lost
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                residual = (u[:, :rank] @ coordinates-normalized)*scales
                # Combine factors by exponent so neither a large reciprocal nor
                # a tiny intermediate product destroys a representable quotient.
                coordinate_mantissa, coordinate_exponent = np.frexp(coordinates)
                response_mantissa, response_exponent = np.frexp(scales)
                singular_mantissa, singular_exponent = np.frexp(singular[:rank, None])
                projection_mantissa, projection_exponent = np.frexp(projected_scale)
                edit_coordinates = np.ldexp(
                    coordinate_mantissa*response_mantissa/(singular_mantissa*projection_mantissa),
                    coordinate_exponent+response_exponent-singular_exponent-projection_exponent)
            # Individually rounded-away coordinates may have a representable
            # combined norm. Do not report the norm of their rounded zeros.
            coordinate_lost = np.any((coordinates != 0.) & (edit_coordinates == 0.), axis=0)
            edit_lost = lost | coordinate_lost
            unregularized.update({
                "target_residual_rmse": _rms(residual) if not lost.any() and np.isfinite(residual).all() else None,
                "target_residual_l2_per_output": [None if missing else norm(row) for missing, row in zip(lost, residual.T)],
                "edit_l2_norm_per_output": [None if missing else norm(row) for missing, row in zip(edit_lost, edit_coordinates.T)],
                "edit_frobenius_norm": None if edit_lost.any() else norm(edit_coordinates),
                "per_output_numerical_limit_reason": ["response_normalization_underflow" if normalized_loss else
                    "response_projection_underflow" if projected_loss else "edit_coordinate_underflow" if coordinate_loss else None
                    for normalized_loss, projected_loss, coordinate_loss in zip(normalization_lost, projection_lost, coordinate_lost)],
            })
    return basis, {
        "schema": "wsp-feature-constraints/1", "editable_width": target_design.shape[1],
        "protected_rows": len(retained), "target_rows": len(target_design),
        "relative_cutoff": nullspace_rtol, "protected": protected,
        "projected_target": target_spectrum, "unregularized": unregularized,
        "numerical_magnitude_encoding": "Singular values equal scaled_singular_values times singular_values_scale; nonrepresentable magnitudes are null",
        "exact": _exact_feature_constraints(target_design, retained, desired_delta, exact_work_limit),
    }


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

    target_design = _editable_design(target, edit_bias)
    retain_design = None if anchors is None else _editable_design(anchors, edit_bias)
    size = target_design.shape[1]
    constrained_basis, constraints = _analyze_feature_constraints(
        target_design, retain_design, desired, nullspace_rtol=nullspace_rtol)
    basis = constrained_basis if retention == "nullspace" else np.eye(size)
    retention_rank = constraints["protected"]["rank"]
    cutoff = constraints["protected"]["absolute_cutoff"] if retention == "nullspace" else None
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
    target_residuals = []
    for row, requested in zip(target, desired):
        augmented = [Fraction.from_float(float(value)) for value in row] + [Fraction(1)]
        target_residuals.append([
            sum((weight*feature for weight, feature in zip(weights, augmented)), Fraction(0))-
            Fraction.from_float(float(change)) for weights, change in zip(endpoint, requested)])
    maximum_target_residuals = [max(abs(row[output]) for row in target_residuals) for output in range(outputs)]

    def effect(features: np.ndarray) -> np.ndarray:
        return np.column_stack((features, np.ones(len(features)))) @ realized.T

    certificate = None if anchors is None else certify_feature_retention(base, candidate, head, anchors)
    diagnostics = {
        "compiler": "generic-frozen-feature-readout/1", "head": head.to_dict(),
        "retention": retention, "retention_weight": retention_weight, "ridge": ridge,
        "edit_bias": edit_bias, "feature_constraints": constraints,
        "optimizer_uses_numerical_nullspace": retention == "nullspace",
        "nullspace_rtol": nullspace_rtol, "singular_cutoff": cutoff,
        "retention_rank": retention_rank if retention == "nullspace" else None,
        "nullspace_dimension": int(basis.shape[1]) if retention == "nullspace" else None,
        "fit_rank": int(fit_rank), "condition_number": condition,
        "target_rows": len(target), "retention_rows": 0 if anchors is None else len(anchors),
        "target_feature_sha256": digest(target.tolist()), "desired_delta_sha256": digest(desired.tolist()),
        "retention_feature_sha256": None if anchors is None else digest(anchors.tolist()),
        "target_rmse_after_projection": _rms(effect(target)-desired),
        "realized_target_residuals_exact": [[str(value) for value in row] for row in target_residuals],
        "realized_target_maximum_absolute_residuals_exact": [str(value) for value in maximum_target_residuals],
        "realized_target_exactly_matches_requested_delta": all(value == 0 for value in maximum_target_residuals),
        "retention_rmse_after_projection": None if anchors is None else _rms(effect(anchors)),
        "edit_cells": len(lowered.program.edits), "no_op": not lowered.program.edits,
        "lowering": lowered.receipt, "behavioral_status": "supplied_features_only",
        "optimizer_verified": False, "feature_extractor_verified": False,
    }
    return FeatureCompilation(lowered.program, proposal, canonical_bytes(diagnostics),
                              None if certificate is None else canonical_bytes(certificate))
