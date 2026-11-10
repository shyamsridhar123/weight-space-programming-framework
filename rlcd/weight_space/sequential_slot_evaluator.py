"""Trusted slot evaluation for the declared owned sequential runtime.

This helper binds empirical predictions to a sealed endpoint and supplied rows.
It does not attest evaluator execution or establish useful causal repair.
"""
from __future__ import annotations

from typing import Any

from .artifact import TensorArtifact
from .model import canonical_bytes, digest, finite_binary64
from .sequential_runtime import restore_sequential_runtime


def _require_plain_json(value: Any) -> None:
    if value is None or type(value) in (str, bool, int, float):
        return
    if type(value) is list:
        for item in value:
            _require_plain_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _require_plain_json(item)
        return
    raise ValueError("Slot evaluation requires plain JSON types and string object keys")


def evaluate_sequential_slot(context: dict[str, Any], endpoint: TensorArtifact,
                             rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Restore the independently sealed endpoint and forward one ordered batch.

    Use as ``cohort.evaluate_registered(load_holdouts, evaluate_sequential_slot)``
    with frozen result keys ``["sequential"]`` and the runtime's exact execution
    declaration. The caller must trust the cohort and its retained seal.
    """
    if type(context) is not dict:
        raise ValueError("Slot context must be a plain JSON object")
    _require_plain_json(context)
    canonical_bytes(context)
    slot = context.get("slot")
    if (type(slot) is not dict or type(slot.get("result_keys")) is not list
            or slot["result_keys"] != ["sequential"]):
        raise ValueError('Declare exactly the slot result keys ["sequential"]')
    if type(rows) is not list or not rows:
        raise ValueError("Supply a nonempty plain row list")
    _require_plain_json(rows)
    rows_sha256 = digest(rows)
    if any(type(row) is not dict or type(row.get("inputs")) is not dict for row in rows):
        raise ValueError("Every row must contain a plain inputs object")

    runtime = restore_sequential_runtime(
        endpoint, expected_endpoint_sha256=slot["endpoint_sha256"],
        execution=slot["execution"],
    )
    input_names = list(runtime.input_names)
    ordered_inputs = []
    for row in rows:
        inputs = row["inputs"]
        if set(inputs) != set(input_names):
            raise ValueError("Row input keys must exactly match the declared runtime inputs")
        ordered_inputs.append([finite_binary64(inputs[name]) for name in input_names])

    predictions = runtime.forward(ordered_inputs).tolist()
    result = {"sequential": {
        "schema": "wsp-sequential-slot-evaluation/1",
        "restoration": runtime.receipt,
        "rows_sha256": rows_sha256,
        "ordered_inputs_sha256": digest({"input_names": input_names, "inputs": ordered_inputs}),
        "predictions": predictions,
    }}
    _require_plain_json(result)
    canonical_bytes(result)
    return result
