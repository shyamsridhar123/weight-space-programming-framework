"""ReLU gauge transformations and an explicitly heuristic matching baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import Network
from .program import Program, apply_program, difference


@dataclass(frozen=True)
class Gauge:
    """Target hidden row h reads source row permutation[h], then scales by s[h]."""

    permutation: tuple[int, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "permutation", tuple(self.permutation))
        object.__setattr__(self, "scales", tuple(self.scales))
        n = len(self.permutation)
        if (not n or any(type(i) is not int for i in self.permutation)
                or sorted(self.permutation) != list(range(n)) or len(self.scales) != n):
            raise ValueError("Gauge must contain a hidden-unit bijection and matching scales")
        if any(not np.isfinite(s) or s <= 0 for s in self.scales):
            raise ValueError("ReLU gauge scales must be finite and strictly positive")

    def inverse(self) -> Gauge:
        inverse = np.argsort(self.permutation)
        return Gauge(tuple(map(int, inverse)), tuple(1 / self.scales[i] for i in inverse))

    def to_dict(self) -> dict[str, Any]:
        return {"permutation": list(self.permutation), "scales": list(self.scales)}


def transform(model: Network, gauge: Gauge) -> Network:
    if len(gauge.permutation) != model.shape[1]:
        raise ValueError("Gauge width differs from network hidden width")
    p, s = np.array(gauge.permutation), np.array(gauge.scales)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return model.replace(w1=model.w1[p]*s[:, None], b1=model.b1[p]*s,
                             w2=model.w2[:, p]/s[None, :])


def transport(base: Network, program: Program, gauge: Gauge) -> tuple[Network, Program]:
    """Conjugate an actual edit through a known gauge; leaves interface labels fixed."""
    transformed_base = transform(base, gauge)
    transformed_candidate = transform(apply_program(base, program), gauge)
    return transformed_base, difference(transformed_base, transformed_candidate,
                                        f"Known-gauge transport: {program.intent}")


def _assignment(cost: np.ndarray) -> np.ndarray:
    """Square O(H³) Hungarian assignment; result maps each row to one column."""
    n = len(cost)
    if cost.shape != (n, n) or not np.isfinite(cost).all():
        raise ValueError("Assignment costs must be a finite square matrix")
    u, v = np.zeros(n+1), np.zeros(n+1)
    matching, previous = np.zeros(n+1, dtype=int), np.zeros(n+1, dtype=int)
    for i in range(1, n+1):
        matching[0] = i
        j0 = 0
        minimum, used = np.full(n+1, np.inf), np.zeros(n+1, dtype=bool)
        while True:
            used[j0] = True
            i0, step, j1 = matching[j0], np.inf, 0
            for j in range(1, n+1):
                if used[j]:
                    continue
                current = cost[i0-1, j-1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j], previous[j] = current, j0
                if minimum[j] < step:
                    step, j1 = minimum[j], j
            for j in range(n+1):
                if used[j]:
                    u[matching[j]] += step
                    v[j] -= step
                else:
                    minimum[j] -= step
            j0 = j1
            if matching[j0] == 0:
                break
        while True:
            j1 = previous[j0]
            matching[j0] = matching[j1]
            j0 = j1
            if j0 == 0:
                break
    result = np.empty(n, dtype=int)
    for j in range(1, n+1):
        result[matching[j]-1] = j-1
    return result


def align_hidden(source: Network, target: Network) -> tuple[Gauge, dict[str, float]]:
    """Match normalized incoming affine weights. Residual is NOT a certificate.

    This can recover a known permutation/scaling when rows are identifiable.
    Independently trained representations need not admit such a correspondence.
    """
    if not source.compatible(target):
        raise ValueError("Alignment requires matching semantic interfaces and shape")
    a = np.column_stack((source.w1, source.b1))
    b = np.column_stack((target.w1, target.b1))
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ua = a / np.where(na > 0, na, 1)[:, None]
    ub = b / np.where(nb > 0, nb, 1)[:, None]
    cost = np.sum((ub[:, None, :] - ua[None, :, :])**2, axis=2)
    permutation = _assignment(cost)
    scales = np.divide(nb, na[permutation], out=np.ones_like(nb),
                       where=(nb > 0) & (na[permutation] > 0))
    gauge = Gauge(tuple(map(int, permutation)), tuple(map(float, scales)))
    mapped = transform(source, gauge)
    residual = np.sqrt(sum(float(np.sum((mapped.tensors[k]-v)**2))
                           for k, v in target.tensors.items()))
    norm = np.sqrt(sum(float(np.sum(v*v)) for v in target.tensors.values()))
    return gauge, {"mean_matching_cost": float(cost[np.arange(len(cost)), permutation].mean()),
                   "relative_parameter_residual": residual/max(norm, np.finfo(float).eps)}


def transfer_edit(source: Network, program: Program, target: Network,
                  gauge: Gauge | None = None) -> Program:
    """Heuristic additive transfer to a different model; admission must be rerun."""
    if not source.compatible(target):
        raise ValueError("Transfer requires matching semantic schemas")
    edited = apply_program(source, program)
    if gauge is not None:
        source, edited = transform(source, gauge), transform(edited, gauge)
    with np.errstate(over="raise", invalid="raise"):
        candidate = target.replace(**{k: v + (edited.tensors[k]-source.tensors[k])
                                      for k, v in target.tensors.items()})
    return difference(target, candidate, f"Unverified additive transfer: {program.intent}")
