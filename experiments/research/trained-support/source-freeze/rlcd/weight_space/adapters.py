"""Explicit bridge to the actual Crown & Charter scalar candidate-scoring ABI."""
from __future__ import annotations

from typing import Any
import copy
import math

import numpy as np

from .model import Network


RTS_FEATURES = tuple("food_shortage wood_shortage gold_shortage stone_shortage idle_workers "
    "worker_deficit population_pressure army_readiness visible_base_threat production_deficit "
    "age_progress resource_surplus visible_army_advantage army_deficit candidate_coverage_gap "
    "candidate_order_disruption gather_food gather_wood gather_gold gather_stone train_worker "
    "build_house build_farm build_production train_army advance_age research defend attack scout "
    "build_defense continue_orders".split())


def from_rts(value: dict[str, Any]) -> Network:
    expected = {"name", "version", "provenance", "description", "featureNames", "hiddenNames",
                "w1", "b1", "w2", "b2", "temperature"}
    if set(value) != expected or tuple(value["featureNames"]) != RTS_FEATURES:
        raise ValueError("Input must use the exact RTS policy schema and executable feature order")
    if (not all(isinstance(value[k], str) and value[k].strip() for k in ("name", "version", "provenance", "description"))
            or len(value["hiddenNames"]) != 38 or len(set(value["hiddenNames"])) != 38
            or any(not isinstance(x, str) or not x.strip() for x in value["hiddenNames"])):
        raise ValueError("Invalid policy metadata or hidden labels")
    temperature = value["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not .05 <= temperature <= 20:
        raise ValueError("Invalid RTS temperature")
    model = Network(value["w1"], value["b1"], [value["w2"]], [value["b2"]], RTS_FEATURES, ("candidate_score",))
    if model.shape != (32, 38, 1) or any(np.any(np.abs(t) > 64) for t in model.tensors.values()):
        raise ValueError("RTS weights must match 32→38→1 and lie in [-64,64]")
    return model


def to_rts(model: Network, template: dict[str, Any], *, description: str) -> dict[str, Any]:
    original = from_rts(template)
    if not model.compatible(original):
        raise ValueError("Network cannot be represented by this RTS adapter")
    value = copy.deepcopy(template)
    value.update(w1=model.w1.tolist(), b1=model.b1.tolist(), w2=model.w2[0].tolist(),
                 b2=float(model.b2[0]), provenance="weight-space compiler; experimental",
                 description=description)
    from_rts(value)  # Enforce the actual numerical ABI before exporting.
    return value
