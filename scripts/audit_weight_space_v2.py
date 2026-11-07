"""Independently replay a frozen generic study from its saved evidence.

No RLCD module is imported. This program does not fit, generate worlds, generate
proposals, or tune a selection rule. NumPy implements the declared saved model
equations; PyTorch is used only to replay the recorded native CPU executions and
bfloat16 endpoint casts. Exact hashes, integer types and rational receipts are
checked separately from numerical observations. Write the new audit outside both
the study and its source freeze; neither input tree is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import sys
from typing import Any, Callable

import numpy as np


ARMS = ("no_op", "unconstrained", "soft_retention", "nullspace", "matched_random", "gradient")
FAMILIES = ("relu_mlp", "tanh_mlp", "conv_tanh")
FORMATS = ("float64", "float32", "float16", "bfloat16")
SEQUENTIAL = "sequential-affine-activation/1"
CONV = "study-conv2d-tanh-flatten-linear/1"
COUNTERS = ("tool_calls", "weight_values", "feedback_calls", "feedback_examples", "execution_calls",
            "execution_examples", "proposals", "input_tokens", "output_tokens")
ATOL, RTOL = 1e-12, 1e-10
SOURCE_MODULES = ("__init__", "__main__", "adapters", "artifact", "benchmarks", "compiler", "contracts",
                  "evidence", "features", "model", "program", "semantics", "sequential", "structured",
                  "study_tasks_v2", "study_v2", "symmetry", "torch_bridge", "workflow")
SOURCE_TESTS = ("test_weight_space_v2.py", "test_weight_space_workflow.py",
                "test_weight_space_study_v2.py", "test_weight_space_study_tasks_v2.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fields(value: Any, expected: set[str], label: str) -> None:
    require(type(value) is dict and set(value) == expected, f"Unexpected fields: {label}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def same(actual: Any, expected: Any, label: str) -> None:
    # Python equality treats True, 1 and 1.0 alike. Evidence equality must not.
    require(canonical(actual) == canonical(expected), f"Exact value/type mismatch: {label}")


def sha(value: Any, label: str) -> str:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"Expected lowercase SHA-256: {label}")
    return value


def count(value: Any, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"Expected exact integer >= {minimum}: {label}")
    return value


def number(value: Any, label: str = "number") -> float:
    require(type(value) in (int, float), f"Expected a number, excluding booleans: {label}")
    result = float(value)
    require(math.isfinite(result), f"Nonfinite number: {label}")
    if type(value) is int:
        require(int(result) == value, f"Inexact integer-to-binary64 conversion: {label}")
    return result


def q(value: Any) -> Fraction:
    return Fraction.from_float(number(float(value) if isinstance(value, np.floating) else value))


def array(value: Any, label: str = "array") -> np.ndarray:
    original = np.asarray(value, dtype=object)
    require(original.size > 0, f"Empty numerical array: {label}")
    values = [number(v, label) for v in original.flat]
    result = np.asarray(values, dtype=np.float64).reshape(original.shape)
    result[result == 0] = 0.0
    return result


def close(actual: Any, expected: Any, label: str, *, atol: float = ATOL, rtol: float = RTOL) -> None:
    a, b = array(actual, label), array(expected, label)
    require(a.shape == b.shape, f"Shape mismatch: {label}")
    require(bool(np.allclose(a, b, atol=atol, rtol=rtol)),
            f"Numerical mismatch: {label}; max_abs={float(np.max(np.abs(a-b)))}")


def compare_record(actual: Any, expected: Any, label: str) -> None:
    """Only expected float leaves admit numerical tolerance; all types stay exact."""
    if type(expected) is float:
        require(type(actual) is float, f"Expected a JSON floating observation: {label}")
        close(actual, expected, label)
    elif type(expected) is dict:
        fields(actual, set(expected), label)
        for key in expected:
            compare_record(actual[key], expected[key], f"{label}.{key}")
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), f"List length/type mismatch: {label}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_record(a, b, f"{label}[{i}]")
    else:
        same(actual, expected, label)


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path: Path) -> Any:
    result = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs,
                        parse_constant=_constant)
    canonical(result)  # Reject overflowing exponent literals too.
    return result


def contained(root: Path, relative: str) -> Path:
    require(type(relative) is str and bool(relative) and "\\" not in relative and ":" not in relative,
            "Evidence paths must be relative POSIX paths")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and pure.as_posix() == relative and
            all(part not in ("", ".", "..") for part in pure.parts), f"Noncanonical path: {relative}")
    target = (root / relative).resolve()
    require(target.is_relative_to(root.resolve()) and target != root.resolve(), f"Path escapes root: {relative}")
    return target


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    require(root.is_dir(), f"Missing evidence directory: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            name = path.relative_to(root).as_posix()
            contained(root, name)
            result[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    return result


def signed_record(value: Any, label: str, key: str = "sha256") -> dict[str, Any]:
    require(type(value) is dict and key in value, f"Missing signed record: {label}")
    sha(value[key], label)
    payload = {name: item for name, item in value.items() if name != key}
    same(value[key], digest(payload), label + ".sha256")
    return payload


def model_arrays(model: dict[str, Any]) -> dict[str, np.ndarray]:
    require(type(model) is dict, "Model must be a data object")
    if model.get("schema") == "wsp-network/1":
        fields(model, {"schema", "activation", "dtype", "input_names", "output_names", "w1", "b1", "w2", "b2"}, "network")
        same([model["activation"], model["dtype"]], ["relu", "float64"], "network semantics")
        tensors = {name: array(model[name], name) for name in ("w1", "b1", "w2", "b2")}
        w, b, v, c = [tensors[name] for name in ("w1", "b1", "w2", "b2")]
        require(w.ndim == v.ndim == 2 and b.shape == (w.shape[0],) and v.shape[1] == w.shape[0]
                and c.shape == (v.shape[0],), "Incompatible explicit ReLU network")
        for name, width in (("input_names", w.shape[1]), ("output_names", v.shape[0])):
            _labels(model[name], width)
        return tensors
    fields(model, {"schema", "dtype", "architecture", "metadata", "tensors"}, "tensor artifact")
    require(model["schema"] == "wsp-tensors/1" and model["dtype"] == "float64", "Unsupported tensor representation")
    require(type(model["architecture"]) is str and bool(model["architecture"].strip()), "Missing architecture")
    require(type(model["metadata"]) is dict and type(model["tensors"]) is dict and bool(model["tensors"]), "Invalid captured tensors")
    require(all(type(name) is str and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,255}", name)
                for name in model["tensors"]), "Invalid tensor name")
    return {name: array(value, name) for name, value in model["tensors"].items()}


def _labels(value: Any, width: int) -> None:
    require(type(value) is list and len(value) == width and all(type(x) is str and x.strip() for x in value)
            and len(set(value)) == width, "Invalid semantic labels")


def replace(model: dict[str, Any], tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    original = model_arrays(model)
    require(set(original) == set(tensors), "Replayed tensor set differs")
    result = copy.deepcopy(model)
    destination = result["tensors"] if model["schema"] == "wsp-tensors/1" else result
    for name, value in tensors.items():
        converted = np.asarray(value, dtype=np.float64).copy()
        require(converted.shape == original[name].shape and np.isfinite(converted).all(), "Invalid replayed endpoint")
        converted[converted == 0] = 0.0
        destination[name] = converted.tolist()
    return result


def compatible(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    before, after = model_arrays(base), model_arrays(candidate)
    same(replace(base, before), base, "canonical base")
    same(replace(candidate, after), candidate, "canonical candidate")
    require(set(before) == set(after) and all(before[k].shape == after[k].shape for k in before), "Incompatible endpoints")
    zero = {name: np.zeros_like(values) for name, values in before.items()}
    same(replace(base, zero), replace(candidate, zero), "endpoint semantic metadata")


def layers(model: dict[str, Any]) -> list[dict[str, str]]:
    require(model.get("architecture") == SEQUENTIAL, "Expected explicit sequential semantics")
    meta, tensors = model["metadata"], model_arrays(model)
    fields(meta, {"layers", "input_names", "output_names"}, "sequential metadata")
    require(type(meta["layers"]) is list and bool(meta["layers"]), "Empty sequential model")
    used, width = set(), None
    for layer in meta["layers"]:
        fields(layer, {"weight", "bias", "activation"}, "layer")
        w, b = layer["weight"], layer["bias"]
        require(type(w) is str and type(b) is str and w != b and w in tensors and b in tensors
                and w not in used and b not in used, "Invalid layer tensor mapping")
        weight, bias = tensors[w], tensors[b]
        require(weight.ndim == 2 and bias.shape == (weight.shape[0],) and
                (width is None or weight.shape[1] == width), "Invalid layer dimensions")
        require(layer["activation"] in ("identity", "relu", "abs", "tanh"), "Unsupported activation")
        used.update((w, b))
        width = weight.shape[0]
    require(used == set(tensors), "Uninterpreted tensors in sequential artifact")
    _labels(meta["input_names"], tensors[meta["layers"][0]["weight"]].shape[1])
    _labels(meta["output_names"], width)
    return meta["layers"]


def readout(model: dict[str, Any]) -> dict[str, Any]:
    tensors = model_arrays(model)
    if model.get("architecture") == SEQUENTIAL:
        last = layers(model)[-1]
        require(last["activation"] == "identity", "Readout must have an identity final activation")
        w, b = last["weight"], last["bias"]
        identity = digest({"architecture": SEQUENTIAL, "metadata": model["metadata"],
                           "extractor_tensors": {k: v.tolist() for k, v in tensors.items() if k not in (w, b)}})
    else:
        require(model.get("architecture") == CONV, "Unsupported study feature semantics")
        w, b = "head.weight", "head.bias"
        same(model["metadata"], {"input_shape": [1, 4, 4], "stride": 1, "padding": 0, "activation": "tanh",
                                "flatten": "channel-row-column", "outputs": ["score0", "score1", "score2"]}, "CNN metadata")
        require(set(tensors) == {"conv.weight", "conv.bias", w, b} and tensors["conv.weight"].shape == (6, 1, 2, 2)
                and tensors["conv.bias"].shape == (6,) and tensors[w].shape == (3, 54) and tensors[b].shape == (3,),
                "Invalid CNN tensor ABI")
        identity = digest({"architecture": CONV, "metadata": model["metadata"],
                           "conv.weight": tensors["conv.weight"].tolist(), "conv.bias": tensors["conv.bias"].tolist()})
    return {"weight": w, "bias": b, "feature_id": "study-frozen-features/" + identity}


def feature_values(model: dict[str, Any], inputs: Any) -> np.ndarray:
    x, tensors = array(inputs), model_arrays(model)
    require(x.ndim == 2, "Feature inputs must be matrices")
    readout(model)
    if model["architecture"] == SEQUENTIAL:
        for layer in layers(model)[:-1]:
            x = x @ tensors[layer["weight"]].T + tensors[layer["bias"]]
            op = layer["activation"]
            x = np.maximum(x, 0.) if op == "relu" else np.abs(x) if op == "abs" else np.tanh(x) if op == "tanh" else x
    else:
        require(x.shape[1] == 16, "CNN input width differs")
        images = x.reshape(-1, 4, 4)
        patches = np.lib.stride_tricks.sliding_window_view(images, (2, 2), axis=(1, 2))
        x = np.tanh(np.einsum("nxyij,cij->ncxy", patches, tensors["conv.weight"][:, 0])
                    + tensors["conv.bias"][None, :, None, None]).reshape(len(x), -1)
    require(np.isfinite(x).all(), "Nonfinite replayed features")
    return x


def forward(model: dict[str, Any], inputs: Any) -> np.ndarray:
    tensors, x = model_arrays(model), array(inputs)
    if model["schema"] == "wsp-network/1":
        return np.maximum(x @ tensors["w1"].T + tensors["b1"], 0.) @ tensors["w2"].T + tensors["b2"]
    if model["architecture"] == "affine/1":
        require(set(tensors) == {"weight", "bias"} and tensors["weight"].shape == (3, 16)
                and tensors["bias"].shape == (3,), "Invalid affine ABI")
        return x @ tensors["weight"].T + tensors["bias"]
    head = readout(model)
    return feature_values(model, x) @ tensors[head["weight"]].T + tensors[head["bias"]]


def native_round(values: np.ndarray, dtype: str) -> np.ndarray:
    require(dtype in FORMATS, "Unsupported endpoint format")
    if dtype == "bfloat16":
        import torch
        result = torch.tensor(values.copy(), dtype=torch.float64).to(torch.bfloat16).to(torch.float64).numpy()
    else:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = values.astype(dtype).astype(np.float64)
    require(np.isfinite(result).all(), "Nonfinite endpoint projection")
    result[result == 0] = 0.0
    return result


def native_forward(model: dict[str, Any], inputs: Any, dtype: str) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    require(dtype in FORMATS, "Unsupported native execution format")
    dt = getattr(torch, dtype)
    tensors = {name: torch.tensor(value, dtype=dt) for name, value in model_arrays(model).items()}
    x = torch.tensor(array(inputs), dtype=dt)
    with torch.inference_mode():
        if model["architecture"] == SEQUENTIAL:
            for layer in layers(model):
                x = functional.linear(x, tensors[layer["weight"]], tensors[layer["bias"]])
                op = layer["activation"]
                x = torch.relu(x) if op == "relu" else torch.abs(x) if op == "abs" else torch.tanh(x) if op == "tanh" else x
        elif model["architecture"] == CONV:
            x = torch.tanh(functional.conv2d(x.reshape(-1, 1, 4, 4), tensors["conv.weight"], tensors["conv.bias"]))
            x = functional.linear(x.flatten(1), tensors["head.weight"], tensors["head.bias"])
        else:
            raise ValueError("Unsupported native replay semantics")
    result = x.to(torch.float64).numpy()
    require(np.isfinite(result).all(), "Nonfinite native forward replay")
    return result


def literal_endpoint(base: dict[str, Any], program: dict[str, Any]) -> dict[str, Any]:
    fields(program, {"schema", "base_sha256", "intent", "edits"}, "literal program")
    require(program["schema"] == "wsp-program/1", "Unsupported literal program")
    same(sha(program["base_sha256"], "program parent"), digest(base), "program parent")
    require(type(program["intent"]) is str and program["intent"].strip() and type(program["edits"]) is list,
            "Invalid literal intent/edits")
    original = model_arrays(base)
    tensors, seen = {name: value.copy() for name, value in original.items()}, set()
    for edit in program["edits"]:
        fields(edit, {"tensor", "index", "before", "after"}, "literal edit")
        name, index = edit["tensor"], edit["index"]
        require(type(name) is str and name in tensors and type(index) is list, "Unknown tensor/coordinate")
        require(all(type(i) is int and i >= 0 for i in index), "Coordinates must be exact nonnegative integers")
        coordinate, tensor = tuple(index), tensors[name]
        require(len(index) == tensor.ndim and all(i < n for i, n in zip(index, tensor.shape)), "Edit is out of bounds")
        require((name, coordinate) not in seen, "Duplicate literal coordinate")
        seen.add((name, coordinate))
        require(q(edit["before"]) == q(float(original[name][coordinate])) and q(edit["before"]) != q(edit["after"]),
                "Stale prevalue or unchanged literal edit")
        tensor[coordinate] = number(edit["after"])
    return replace(base, tensors)


def literal_difference(base: dict[str, Any], candidate: dict[str, Any], intent: str) -> dict[str, Any]:
    compatible(base, candidate)
    before, after = model_arrays(base), model_arrays(candidate)
    edits = []
    for name, values in before.items():
        for index in np.argwhere(values != after[name]):
            coordinate = tuple(int(i) for i in index)
            edits.append({"tensor": name, "index": list(coordinate), "before": float(values[coordinate]),
                          "after": float(after[name][coordinate])})
    return {"schema": "wsp-program/1", "base_sha256": digest(base), "intent": intent, "edits": edits}


def signed(payload: dict[str, Any]) -> dict[str, Any]:
    return payload | {"sha256": digest(payload)}


def lowering(base: dict[str, Any], proposal: dict[str, Any], receipt: dict[str, Any],
             program: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Replay only saved commands, never the solver that proposed them."""
    fields(proposal, {"schema", "base_sha256", "intent", "commands"}, "structured program")
    same(proposal["schema"], "wsp-structured-program/1", "structured schema")
    same(proposal["base_sha256"], digest(base), "structured parent")
    require(type(proposal["intent"]) is str and bool(proposal["intent"].strip()), "Missing structured intent")
    require(type(proposal["commands"]) is list, "Structured commands must be a list")
    signed_record(receipt, "lowering")
    original = model_arrays(base)
    dtypes, cast = receipt["native_dtypes"], receipt["cast"]
    require(type(dtypes) is dict and set(dtypes) == set(original) and
            all(type(t) is str and t in FORMATS for t in dtypes.values()), "Incomplete native dtype map")
    require(cast in ("reject", "round"), "Unknown cast policy")
    for name, value in original.items():
        require(np.array_equal(native_round(value, dtypes[name]), value), "Base was not already native-representable")
    values = {name: value.copy() for name, value in original.items()}
    touched = {name: np.zeros(value.shape, dtype=bool) for name, value in original.items()}
    operands = {"replace": {"values"}, "add": {"values"}, "scale": {"factor"},
                "low_rank": {"left", "right", "alpha"}}
    for command in proposal["commands"]:
        require(type(command) is dict and type(command.get("op")) is str and command["op"] in operands,
                "Unknown structured operation")
        op = command["op"]
        required = {"op", "tensor"} | operands[op]
        require(set(command) in (required, required | {"block"}), "Unexpected command fields")
        name = command["tensor"]
        require(type(name) is str and name in values, "Unknown structured tensor")
        target = values[name]
        if "block" in command:
            block = command["block"]
            require(type(block) is list and len(block) == target.ndim, "Invalid command block rank")
            for axis, size in zip(block, target.shape):
                require(type(axis) is list and len(axis) == 2 and all(type(v) is int for v in axis)
                        and 0 <= axis[0] < axis[1] <= size, "Invalid command block limits")
            index = tuple(slice(start, stop) for start, stop in block)
        else:
            index = tuple(slice(None) for _ in target.shape)
        current = target[index]
        with np.errstate(over="raise", invalid="raise"):
            if op in ("replace", "add"):
                operand = array(command["values"])
                require(operand.shape == current.shape, "Structured operand broadcasts")
                update = operand if op == "replace" else current + operand
            elif op == "scale":
                update = current * number(command["factor"])
            else:
                left, right = array(command["left"]), array(command["right"])
                require(current.ndim == left.ndim == right.ndim == 2 and left.shape[1] == right.shape[0]
                        and (left.shape[0], right.shape[1]) == current.shape, "Invalid low-rank operands")
                update = current + number(command["alpha"]) * (left @ right)
        target[index] = array(update)
        touched[name][index] = True
    requested = replace(base, values)
    endpoints, projections = {}, {}
    for name, value in values.items():
        endpoint = native_round(value, dtypes[name])
        rounded = value != endpoint
        require(cast != "reject" or not np.any(rounded), "Reject cast silently rounded")
        projections[name] = {"dtype": dtypes[name], "touched_cells": int(touched[name].sum()),
            "requested_changed_cells": int(np.count_nonzero(original[name] != value)),
            "effective_changed_cells": int(np.count_nonzero(original[name] != endpoint)),
            "rounded_cells": int(rounded.sum()),
            "maximum_rounding_error_exact": str(max((abs(q(a)-q(b)) for a, b in
                zip(value[rounded].flat, endpoint[rounded].flat)), default=Fraction(0)))}
        endpoints[name] = endpoint
    replayed = replace(base, endpoints)
    expected_program = literal_difference(base, replayed, proposal["intent"])
    same(candidate, replayed, "lowered candidate")
    same(program, expected_program, "lowered literal program")
    expected = {"schema": "wsp-lowering/1", "proposal_sha256": digest(proposal),
        "base_sha256": digest(base), "requested_candidate_sha256": digest(requested),
        "candidate_sha256": digest(replayed), "program_sha256": digest(expected_program),
        "command_count": len(proposal["commands"]), "cast": cast, "native_dtypes": dtypes,
        "projections": projections, "effective_edits": len(program["edits"]),
        "behavioral_status": "not_checked", "numerics": "NumPy binary64 operations; explicit native endpoint projection",
        "numpy_version": np.__version__}
    if "bfloat16" in dtypes.values():
        import torch
        expected["torch_version"] = str(torch.__version__)
    same(receipt, signed(expected), "lowering receipt")


def head_delta(base: dict[str, Any], candidate: dict[str, Any]) -> list[list[Fraction]]:
    compatible(base, candidate)
    head, before, after = readout(base), model_arrays(base), model_arrays(candidate)
    for name in before:
        if name not in (head["weight"], head["bias"]):
            same(before[name].tolist(), after[name].tolist(), "frozen extractor " + name)
    result = [[q(b)-q(a) for a, b in zip(row_a, row_b)]
              for row_a, row_b in zip(before[head["weight"]], after[head["weight"]])]
    for i, row in enumerate(result):
        row.append(q(after[head["bias"]][i])-q(before[head["bias"]][i]))
    return result


def retention_receipt(base: dict[str, Any], candidate: dict[str, Any], anchors: np.ndarray) -> dict[str, Any]:
    delta, head = head_delta(base, candidate), readout(base)
    require(anchors.ndim == 2 and anchors.shape[1] + 1 == len(delta[0]), "Retention width differs")
    residuals = [[abs(sum((weight*feature for weight, feature in zip(row,
                    [q(x) for x in anchor] + [Fraction(1)])), Fraction(0))) for row in delta]
                 for anchor in anchors]
    maximum = [max(row[i] for row in residuals) for i in range(len(delta))]
    return signed({"schema": "wsp-feature-retention/1", "base_sha256": digest(base),
        "candidate_sha256": digest(candidate), "head": head, "feature_sha256": digest(anchors.tolist()),
        "anchor_count": len(anchors), "absolute_residuals_exact": [[str(x) for x in row] for row in residuals],
        "maximum_residuals_exact": [str(x) for x in maximum],
        "exactly_zero_on_supplied_features": all(x == 0 for x in maximum),
        "semantics": "Real linear head on supplied frozen features; bias is a constant augmented feature",
        "feature_extractor_verified": False, "floating_execution_verified": False,
        "span_extension": "At augmented feature sum(c[k]*anchor[k]), drift <= sum(abs(c[k])*anchor_residual[k])"})


def sequential_receipt(base: dict[str, Any], candidate: dict[str, Any], lower: list[float], upper: list[float]) -> dict[str, Any]:
    compatible(base, candidate)
    before, after, spec = model_arrays(base), model_arrays(candidate), layers(base)
    require(len(lower) == len(upper) == before[spec[0]["weight"]].shape[1], "Domain width mismatch")
    require(all(number(lo) <= number(hi) for lo, hi in zip(lower, upper)), "Empty domain interval")
    radii = [max(abs(q(lo)), abs(q(hi))) for lo, hi in zip(lower, upper)]
    drift = [Fraction(0)] * len(radii)
    intermediate = []
    for layer in spec:
        require(layer["activation"] in ("identity", "relu", "abs"), "Uncertified activation")
        w, b = before[layer["weight"]], before[layer["bias"]]
        wp, bp = after[layer["weight"]], after[layer["bias"]]
        next_radii = [abs(q(b[i])) + sum((abs(q(w[i, j]))*radii[j] for j in range(len(radii))), Fraction(0))
                      for i in range(len(b))]
        drift = [abs(q(bp[i])-q(b[i])) + sum((abs(q(wp[i, j])-q(w[i, j]))*radii[j]
                  + abs(q(wp[i, j]))*drift[j] for j in range(len(radii))), Fraction(0)) for i in range(len(b))]
        radii = next_radii
        intermediate.append({"layer": layer, "base_radius": [str(x) for x in radii], "drift": [str(x) for x in drift]})
    return signed({"schema": "wsp-sequential-drift/1", "base_artifact_sha256": digest(base),
        "candidate_artifact_sha256": digest(candidate), "architecture": SEQUENTIAL,
        "domain": {"lower": lower, "upper": upper}, "layers": intermediate, "output_bounds": [str(x) for x in drift],
        "arithmetic": "Exact rationals lifted from stored binary64 endpoints",
        "scope": "All real inputs in the declared box; shared certified activation per layer",
        "implementation_verified": False, "floating_execution_verified": False, "reward_verified": False})


def admission_receipt(base: dict[str, Any], program: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    candidate, issues = literal_endpoint(base, program), []
    if any(edit["tensor"] not in policy["writable"] for edit in program["edits"]):
        issues.append("write-set")
    if len(program["edits"]) > count(policy["max_edits"], "max_edits"):
        issues.append("edit-count")
    if any(np.any(np.abs(value) > number(policy["max_parameter_abs"])) for value in model_arrays(candidate).values()):
        issues.append("parameter-bound")
    if any(abs(q(edit["after"])-q(edit["before"])) > q(policy["max_change_abs"]) for edit in program["edits"]):
        issues.append("change-bound")
    return signed({"schema": "wsp-structural-admission/1", "structural_accepted": not issues,
        "issues": issues, "program_sha256": digest(program), "candidate_sha256": digest(candidate), "policy": policy,
        "behavioral_status": "not_checked", "formal_certificate": None})


def rms(values: Any) -> float:
    values = array(values)
    scale = float(np.max(np.abs(values)))
    return 0.0 if scale == 0.0 else scale * float(np.sqrt(np.mean((values / scale) ** 2)))


def l2(values: list[np.ndarray]) -> float:
    vector = np.concatenate([value.ravel() for value in values])
    return rms(vector) * float(np.sqrt(vector.size))


def teacher_values(inputs: Any, teacher: dict[str, Any]) -> np.ndarray:
    x = array(inputs)
    fields(teacher, {"linear", "nonlinear"}, "teacher")
    linear, nonlinear = array(teacher["linear"]), array(teacher["nonlinear"])
    require(linear.shape == nonlinear.shape == (16, 3), "Teacher shape mismatch")
    return x @ linear + .2 * np.sin(x @ nonlinear)


def metrics(base: dict[str, Any], candidate: dict[str, Any], target: Any, retain: Any,
            shift: Any, teacher: dict[str, Any] | None = None) -> dict[str, Any]:
    compatible(base, candidate)
    s = array(shift)
    require(s.shape == (3,) and rms(s) > 0, "Invalid metric shift")
    bt, br, at, ar = forward(base, target), forward(base, retain), forward(candidate, target), forward(candidate, retain)
    dt, dr = at-bt, ar-br
    normalization = rms(s)
    target_error, retain_error = rms(dt-s), rms(dr)
    direction = s / float(np.max(np.abs(s)))
    gain = float(np.mean((dt / float(np.max(np.abs(s)))) @ direction) / np.dot(direction, direction))
    before, after = model_arrays(base), model_arrays(candidate)
    changes = [after[name]-value for name, value in before.items()]
    result = {"target_nrmse": target_error/normalization, "retain_nrmse": retain_error/normalization,
        "target_gain_fraction": gain, "target_rmse": target_error, "retain_rmse": retain_error,
        "effective_changed_cells": sum(int(np.count_nonzero(value)) for value in changes),
        "delta_l2": l2(changes), "no_op": digest(base) == digest(candidate)}
    if teacher is not None:
        labels_t, labels_r = teacher_values(target, teacher), teacher_values(retain, teacher)
        result.update(original_target_rmse=rms(bt-labels_t), edited_task_target_rmse=rms(at-labels_t-s),
                      original_retain_rmse=rms(br-labels_r), edited_task_retain_rmse=rms(ar-labels_r))
    canonical(result)
    return result


def response_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    require(expected.shape == actual.shape and actual.ndim == 2, "Auxiliary response shape mismatch")
    residual = actual-expected
    maximum = float(np.max(np.abs(residual)))
    return {"examples": len(expected), "output_dimension": expected.shape[1], "rmse": rms(residual),
        "mean_absolute_error": float(np.mean(np.abs(residual))), "maximum_absolute_error": maximum,
        "exact_array_equality": bool(np.array_equal(expected, actual)), "within_tolerance": maximum <= 1e-12,
        "absolute_tolerance": 1e-12}


def split_commitment(splits: list[dict[str, Any]]) -> dict[str, Any]:
    seen_samples, seen_lineages, names, records = set(), set(), set(), []
    for split in splits:
        fields(split, {"name", "sample_ids", "lineage_ids"}, "split")
        name, sample_ids, lineage_ids = split["name"], split["sample_ids"], split["lineage_ids"]
        require(type(name) is str and name and name not in names, "Duplicate or invalid split name")
        names.add(name)
        for values, seen in ((sample_ids, seen_samples), (lineage_ids, seen_lineages)):
            require(type(values) is list and all(type(v) is str and v.strip() for v in values)
                    and len(values) == len(set(values)) and not seen.intersection(values), "Repeated split identities")
            seen.update(values)
        records.append({"name": name, "sample_count": len(sample_ids), "lineage_count": len(lineage_ids),
                        "manifest_sha256": digest(split)})
    return signed({"schema": "wsp-split-validation/1", "splits": records, "sample_overlap": False,
        "declared_lineage_overlap": False, "lineage_scope": "caller_declared_only", "lineage_independence_verified": False})


def session_config(task: dict[str, Any], access: dict[str, Any], limits: dict[str, int],
                   splits: list[dict[str, Any]], callbacks: dict[str, bool]) -> dict[str, Any]:
    fields(task, {"schema", "task_id", "mode", "objective", "architecture", "base_sha256", "constraints"}, "task")
    same(task["schema"], "wsp-task/1", "task schema")
    require(task["mode"] in ("read", "diagnose", "edit", "create"), "Unknown task mode")
    for key in ("task_id", "objective", "architecture"):
        require(type(task[key]) is str and task[key].strip(), "Empty task field")
    require(type(task["constraints"]) is dict, "Task constraints must be an object")
    if task["mode"] == "create":
        same(task["base_sha256"], None, "Creation parent")
    else:
        sha(task["base_sha256"], "task parent")
    fields(limits, set(COUNTERS), "budget limits")
    for key, value in limits.items():
        count(value, key)
    fields(access, {"metadata", "weights", "feedback", "forward", "activations", "weight_decimals"}, "access policy")
    require(all(type(access[k]) is bool for k in access if k != "weight_decimals"), "Access capabilities are not booleans")
    require(access["weight_decimals"] is None or type(access["weight_decimals"]) is int and
            0 <= access["weight_decimals"] <= 15, "Invalid weight display precision")
    fields(callbacks, {"feedback", "forward", "activations"}, "callbacks")
    require(all(type(v) is bool for v in callbacks.values()), "Callback capabilities are not booleans")
    return {"schema": "wsp-session-config/1", "task_sha256": digest(task),
        **{key: task[key] for key in ("task_id", "mode", "objective", "architecture", "base_sha256")},
        "constraints_sha256": digest(task["constraints"]), "access": access, "limits": limits,
        "callbacks_registered": callbacks, "splits": split_commitment(splits),
        "enforcement": "cooperative_same_process_interface", "declared_constraints_automatically_enforced": False,
        "execution_accounting": "interface_calls_and_submitted_examples; callback_internal_work_not_observed",
        "token_accounting": "provider_supplied_not_locally_measured"}


def verify_session(receipt: dict[str, Any], task: dict[str, Any], base: dict[str, Any] | None,
                   configuration: dict[str, Any], evaluator_id: str, split_name: str,
                   expected_actions: list[str]) -> dict[str, Any]:
    """Strict complete-run trace replay; unsuccessful attempts are not silently accepted."""
    payload = signed_record(receipt, "evaluation receipt")
    require(receipt["status"] == "succeeded", "Only completed evaluation traces have replayed outcomes")
    same(receipt["configuration"], configuration, "session configuration")
    task_id, session_id = digest(task), digest(configuration)
    parent = None if base is None else digest(base)
    same(task["base_sha256"], parent, "task exact parent")
    if base is not None:
        model_arrays(base)
    candidates = receipt["candidates"]
    require(type(candidates) is list and candidates, "No session candidates")
    ids, by_id = [], {}
    for index, candidate in enumerate(candidates, 1):
        signed_record(candidate, "candidate")
        kind, artifact = candidate["kind"], candidate["artifact"]
        require(kind in ("program", "artifact", "answer"), "Unsupported candidate kind")
        if kind == "answer":
            same([artifact, candidate["artifact_sha256"], candidate["program_sha256"]], [None, None, None], "answer artifacts")
        else:
            model_arrays(artifact)
            if base is not None:
                compatible(base, artifact)
            same(candidate["artifact_sha256"], digest(artifact), "candidate artifact hash")
            same(candidate["answer"], None, "artifact answer")
            if kind == "program":
                sha(candidate["program_sha256"], "candidate program")
            else:
                same(candidate["program_sha256"], None, "artifact program")
        expected = {"schema": "wsp-candidate/1", "session_sha256": session_id, "task_sha256": task_id,
            "parent_sha256": parent, "proposal_index": index, "kind": kind,
            "program_sha256": candidate["program_sha256"], "artifact_sha256": None if artifact is None else digest(artifact),
            "artifact": artifact, "answer": candidate["answer"], "no_op": artifact is not None and digest(artifact) == parent}
        same(candidate, signed(expected), "candidate identity")
        require(candidate["sha256"] not in by_id, "Duplicate candidate identity")
        ids.append(candidate["sha256"])
        by_id[candidate["sha256"]] = candidate
    selected_id = receipt["candidate_sha256"]
    require(selected_id in by_id, "Selected candidate is absent")
    same(receipt["candidate"], by_id[selected_id], "selected candidate")
    trace = receipt["trace"]
    require(type(trace) is list and trace, "Missing trace")
    same([event["action"] for event in trace], expected_actions, "complete trace actions")
    usage = dict.fromkeys(COUNTERS, 0)
    previous, proposal_index, sealed, evaluated = None, 0, False, False
    events: dict[str, list[dict[str, Any]]] = {}
    for sequence, event in enumerate(trace):
        signed_record(event, "trace event")
        fields(event, {"schema", "sequence", "previous_sha256", "session_sha256", "task_sha256", "action", "status",
            "request", "result", "charges", "requested_charges", "usage", "error", "sha256"}, "trace event")
        same([event["schema"], event["sequence"], event["previous_sha256"], event["session_sha256"], event["task_sha256"],
              event["status"], event["error"]],
             ["wsp-trace-event/1", sequence, previous, session_id, task_id, "succeeded", None], "trace identity/status")
        action, request, result = event["action"], event["request"], event["result"]
        require(not evaluated and (not sealed or action == "heldout_evaluation"), "Action after sealing/evaluation")
        charges: dict[str, int] = {}
        if action == "session_created":
            require(sequence == 0, "Repeated session creation")
            same(request, configuration, "trace initial configuration")
            same(result, {}, "trace initial result")
        elif action == "metadata":
            require(configuration["access"]["metadata"], "Unauthorized metadata read")
            same(request, {}, "metadata request")
            artifact = None if base is None else {"sha256": parent, "architecture": task["architecture"],
                "tensors": {name: {"shape": list(value.shape), "dtype": "float64"} for name, value in model_arrays(base).items()},
                "metadata": base["metadata"] if base["schema"] == "wsp-tensors/1" else
                    {"input_names": base["input_names"], "output_names": base["output_names"]}}
            same(result, {"task": task, "artifact": artifact,
                 "constraints_status": "declared_context_not_automatically_enforced"}, "metadata result")
            charges = {"tool_calls": 1}
        elif action == "weights":
            require(base is not None and configuration["access"]["weights"], "Unauthorized weight read")
            fields(request, {"tensor", "weight_decimals", "indices"}, "weight request")
            require(request["tensor"] in model_arrays(base), "Unknown weight tensor")
            same(request["indices"], None, "Study only uses complete tensor reads")
            same(request["weight_decimals"], configuration["access"]["weight_decimals"], "Display precision")
            value = model_arrays(base)[request["tensor"]]
            decimals = request["weight_decimals"]
            rounded = value.tolist() if decimals is None else np.asarray(
                [round(float(x), decimals) for x in value.flat]).reshape(value.shape).tolist()
            same(result, rounded, "weight observation")
            charges = {"tool_calls": 1, "weight_values": int(value.size)}
        elif action in ("propose_program", "propose_artifact", "submit_answer"):
            require(proposal_index < len(candidates), "Unrecorded proposal")
            candidate = candidates[proposal_index]
            proposal_index += 1
            expected_kind = {"propose_program": "program", "propose_artifact": "artifact", "submit_answer": "answer"}[action]
            same(candidate["kind"], expected_kind, "proposal kind")
            charges = {"tool_calls": 1, "proposals": 1}
            if action == "propose_program":
                require(task["mode"] == "edit" and configuration["access"]["metadata"] and
                        configuration["access"]["weights"] and configuration["access"]["weight_decimals"] is None,
                        "Unauthorized literal proposal")
                fields(request, {"parser", "proposal_type", "program", "program_sha256"}, "literal proposal request")
                same([request["parser"], request["proposal_type"], request["program_sha256"]],
                     ["typed_program_or_wsp-program/1", "Program", digest(request["program"])], "literal request identity")
                same(candidate["program_sha256"], request["program_sha256"], "candidate literal binding")
                same(literal_endpoint(base, request["program"]), candidate["artifact"], "trace literal endpoint")
                charges["weight_values"] = sum(v.size for v in model_arrays(base).values())
            elif action == "propose_artifact":
                require(task["mode"] in ("create", "edit"), "Artifact proposal in wrong task mode")
                same(request, {"parser": "typed_artifact_or_declared_schema", "proposal_type": "Network" if
                    candidate["artifact"]["schema"] == "wsp-network/1" else "TensorArtifact",
                    "artifact_sha256": candidate["artifact_sha256"]}, "artifact proposal request")
            else:
                require(task["mode"] in ("read", "diagnose"), "Answer in wrong task mode")
                same(request, {}, "answer request")
            same(result, {"candidate_sha256": candidate["sha256"], "artifact_sha256": candidate["artifact_sha256"],
                          "no_op": candidate["no_op"]}, "proposal result")
        elif action in ("activations", "forward", "feedback"):
            require(configuration["access"][action] and configuration["callbacks_registered"][action], "Unauthorized callback")
            examples = array(request["examples"])
            count(request["example_count"], "example count", 1)
            same(request["example_count"], len(examples), "submitted example count")
            candidate_id = request["candidate_sha256"]
            require(candidate_id is None or candidate_id in ids[:proposal_index], "Execution references future candidate")
            if action == "feedback":
                fields(request, {"candidate_sha256", "example_count", "examples"}, "feedback request")
                require(candidate_id is not None, "Feedback without a candidate")
                charges = {"tool_calls": 1, "feedback_calls": 1, "feedback_examples": len(examples)}
            else:
                fields(request, {"candidate_sha256", "artifact_sha256", "example_count", "examples"}, "execution request")
                model = base if candidate_id is None else by_id[candidate_id]["artifact"]
                same(request["artifact_sha256"], digest(model), "execution artifact binding")
                close(result, feature_values(model, examples) if action == "activations" else forward(model, examples), "public execution")
                charges = {"tool_calls": 1, "execution_calls": 1, "execution_examples": len(examples)}
        elif action == "provider_usage":
            same(request, {"accounting": "provider_supplied_not_locally_measured", "input_tokens": 0, "output_tokens": 0}, "provider usage")
            same(result, request, "provider result")
            charges = {"input_tokens": 0, "output_tokens": 0}
        elif action == "seal":
            require(selected_id in ids[:proposal_index] and proposal_index == len(candidates), "Selection before candidate completion")
            same(request, {"candidate_sha256": selected_id}, "sealed selection")
            same(result, {"selection": "candidate"}, "seal result")
            sealed = True
        elif action == "heldout_evaluation":
            require(sealed and sequence == len(trace)-1, "Evaluation before sealing or not last")
            same(request, {"candidate_sha256": selected_id, "evaluator_id": evaluator_id, "split_name": split_name}, "heldout request")
            same(result, receipt["result"], "receipt/trace evaluation binding")
            evaluated = True
        else:
            raise ValueError(f"Unsupported trace action: {action}")
        same(event["charges"], charges, "action charges")
        same(event["requested_charges"], charges, "requested charges")
        fields(event["usage"], set(COUNTERS), "trace usage")
        for key in COUNTERS:
            usage[key] += count(charges.get(key, 0), "charge")
            count(event["usage"][key], "usage")
            require(usage[key] <= configuration["limits"][key], "Exceeded session limit")
        same(event["usage"], usage, "cumulative usage")
        previous = event["sha256"]
        events.setdefault(action, []).append(event)
    require(sealed and evaluated and proposal_index == len(candidates), "Incomplete workflow")
    expected = {"schema": "wsp-evaluation-receipt/1", "status": "succeeded", "session_sha256": session_id,
        "task_sha256": task_id, "parent_sha256": parent, "configuration": configuration,
        "candidate_sha256": selected_id, "candidate": by_id[selected_id], "candidates": candidates,
        "usage": usage, "evaluator_id": evaluator_id, "split_name": split_name, "result": receipt["result"], "error": None,
        "trace": trace, "trace_head_sha256": previous,
        "heldout_input_handling": "caller_owned_closure_not_passed_to_proposer_tools",
        "evaluator_output_leakage_checked": False,
        "heldout_accounting": {"evaluator_calls": 1, "evaluated_examples": None, "scope": "outside_proposer_budget"},
        "evaluator_attested": False, "behavioral_certificate": None}
    same(payload, expected, "evaluation envelope")
    return {"events": events, "candidates": by_id, "usage": usage}


def access_policy(**overrides: Any) -> dict[str, Any]:
    return {"metadata": True, "weights": True, "feedback": False, "forward": False,
            "activations": False, "weight_decimals": None} | overrides


def cluster_estimate(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    """Resample persisted paired observations only; never generate study worlds."""
    observations = array(values)
    require(observations.ndim == 1, "Bootstrap observations must be one-dimensional")
    count(draws, "bootstrap draws", 100)
    count(seed, "bootstrap seed")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    estimates = observations[indices].mean(axis=1)
    return {"mean": float(observations.mean()),
        "ci95_percentile": [float(v) for v in np.quantile(estimates, [.025, .975])],
        "independent_worlds": len(values), "per_world_values": observations.tolist(),
        "inference": "exploratory world-cluster bootstrap; families within a world averaged first; no multiplicity correction"}


def summary_values(cases: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    pairs = [(case["seed"], case["family"]) for case in cases]
    require(len(pairs) == len(set(pairs)), "Duplicate seed/family checkpoints")
    require(set(pairs) == {(seed, family) for seed in protocol["seeds"] for family in protocol["families"]},
            "Summary omits or adds scheduled checkpoints")
    main = [case for case in cases if type(case.get("main")) is dict]
    groups = {seed: [case for case in main if case["seed"] == seed] for seed in protocol["seeds"]}
    complete = [seed for seed, rows in groups.items() if len(rows) == len(protocol["families"])
        and {row["family"] for row in rows} == set(protocol["families"])
        and all(set(row["main"]["arms"]) == set(ARMS) and row["main"]["failures"] == [] for row in rows)]
    formulas = [
        ("soft_collateral_reduction", "unconstrained", "soft_retention", "retain_nrmse"),
        ("soft_target_error_cost", "soft_retention", "unconstrained", "target_nrmse"),
        ("soft_advantage_over_random_target", "matched_random", "soft_retention", "target_nrmse"),
        ("gradient_minus_soft_target_error", "gradient", "soft_retention", "target_nrmse")]
    contrasts = {}
    for index, (name, positive, negative, metric) in enumerate(formulas):
        values = [float(np.mean([case["main"]["arms"][positive][metric]-case["main"]["arms"][negative][metric]
                                 for case in groups[seed]])) for seed in complete]
        contrasts[name] = None if not values else cluster_estimate(values, protocol["bootstrap"]["draws"],
                                                                   protocol["bootstrap"]["seed"]+index)
    families = {}
    for family in protocol["families"]:
        rows = [case for case in main if case["family"] == family]
        means = {}
        denominators = {arm: sum(arm in case["main"]["arms"] for case in rows) for arm in ARMS}
        for arm in ARMS:
            if denominators[arm]:
                means[arm] = {metric: float(np.mean([case["main"]["arms"][arm][metric] for case in rows
                    if arm in case["main"]["arms"]])) for metric in ("target_nrmse", "retain_nrmse", "target_gain_fraction", "delta_l2")}
        families[family] = {"checkpoints": len(rows), "attempted_checkpoints": sum(case["family"] == family for case in cases),
            "arm_denominators": denominators, "arm_means": means,
            "selected_arms": dict(Counter(case["main"]["selected_arm"] for case in rows))}
    return {"completed_checkpoints": len(main), "attempted_checkpoints": len(cases),
        "fully_completed_checkpoints": sum(case.get("status") == "completed" for case in cases),
        "complete_worlds_for_cluster_inference": len(complete), "complete_world_seeds": complete,
        "omitted_incomplete_worlds": [seed for seed in protocol["seeds"] if seed not in complete],
        "incomplete_world_policy": protocol["bootstrap"]["incomplete_world_policy"], "contrasts": contrasts,
        "by_family": families, "arm_denominators": {arm: sum(arm in case["main"]["arms"] for case in main) for arm in ARMS},
        "selected_arms": dict(Counter(case["main"]["selected_arm"] for case in main)), "model_ability_measured": False}


def protocol_check(protocol: dict[str, Any]) -> None:
    fields(protocol, {"schema", "name", "study_kind", "seeds", "families", "sizes", "training_ridge", "shift", "compiler",
        "selection", "precision", "feasibility", "sequence", "auxiliary", "bootstrap", "accounting", "claims", "source_review"}, "protocol")
    same(protocol["schema"], "wsp-generic-study/2", "protocol schema")
    same(protocol["study_kind"], "deterministic_algorithmic_controls", "study kind")
    require(type(protocol["name"]) is str and protocol["name"].strip(), "Missing protocol name")
    for key in ("seeds", "families", "precision", "feasibility"):
        values = protocol[key]
        require(type(values) is list and values and len(set(values)) == len(values), "Invalid protocol list: " + key)
    for seed in protocol["seeds"]:
        count(seed, "world seed")
    require(all(type(f) is str and f in FAMILIES for f in protocol["families"]), "Unsupported family")
    require(all(type(f) is str and f in FORMATS for f in protocol["precision"]), "Unsupported precision")
    fields(protocol["sizes"], {"train", "public", "validation", "test", "sequence"}, "split sizes")
    for size in protocol["sizes"].values():
        count(size, "split size", 1)
    for n in protocol["feasibility"]:
        require(count(n, "anchor count", 1) <= protocol["sizes"]["public"], "Too many feasibility anchors")
    require(number(protocol["training_ridge"]) > 0, "Nonpositive training ridge")
    compiler = protocol["compiler"]
    fields(compiler, {"ridge", "retention_weight", "nullspace_rtol", "nullspace_anchors", "gradient_steps"}, "compiler settings")
    require(number(compiler["ridge"]) > 0 and number(compiler["retention_weight"]) >= 0
            and 0 < number(compiler["nullspace_rtol"]) < 1, "Invalid compiler settings")
    require(count(compiler["nullspace_anchors"], "nullspace anchors", 1) <= protocol["sizes"]["public"], "Too many anchors")
    count(compiler["gradient_steps"], "gradient iterations", 1)
    selection = protocol["selection"]
    fields(selection, {"max_retain_nrmse", "max_target_nrmse", "order", "fallback", "structural_policy"}, "selection rule")
    same(selection["order"], ["minimum validation target_nrmse", "minimum delta_l2", "lexicographic arm name"], "selection ordering")
    same(selection["fallback"], "no_op", "fallback")
    require(number(selection["max_retain_nrmse"]) >= 0 and number(selection["max_target_nrmse"]) >= 0, "Negative selection threshold")
    structural = selection["structural_policy"]
    fields(structural, {"write_set", "max_edits", "max_parameter_abs", "max_change_abs"}, "structural declaration")
    same([structural["write_set"], structural["max_edits"]], ["declared head and bias", "parent parameter count"], "structural scope")
    require(number(structural["max_parameter_abs"]) >= 0 and number(structural["max_change_abs"]) >= 0, "Negative structural limit")
    sequence = protocol["sequence"]
    expected_sequence = {"methods": ["unconstrained", "soft_retention"],
        "regions": ["x0 < -0.5", "abs(x0) < 0.2", "x0 > 0.5"],
        "target": "original checkpoint outputs plus that region's shift",
        "retention": "current checkpoint outputs on the other two public regions",
        "selection": "none; all three edits and both methods fixed before any heldout evaluation"}
    fields(sequence, set(expected_sequence) | {"shifts"}, "sequence")
    same({k: sequence[k] for k in expected_sequence}, expected_sequence, "sequence declarations")
    require(type(sequence["shifts"]) is list and len(sequence["shifts"]) == 3, "Wrong number of sequence shifts")
    for shift in [protocol["shift"]] + sequence["shifts"]:
        value = array(shift)
        require(type(shift) is list and value.shape == (3,) and rms(value) > 0, "Invalid score shift")
    bootstrap = protocol["bootstrap"]
    fields(bootstrap, {"draws", "seed", "unit", "interval", "incomplete_world_policy"}, "bootstrap settings")
    count(bootstrap["draws"], "bootstrap draws", 100)
    count(bootstrap["seed"], "bootstrap seed")
    same(bootstrap["unit"], "independent world; average paired family effects within each world", "inferential unit")
    same(bootstrap["interval"], "exploratory percentile 95%; no multiplicity correction", "interval definition")
    same(bootstrap["incomplete_world_policy"], "omit worlds missing any main arm or family; retain available main outcomes descriptively; ablation failures do not remove main outcomes", "incomplete-world policy")
    same(protocol["auxiliary"], True, "auxiliary declaration")
    same(protocol["source_review"], "docs/weight-space/v2/source-review.md", "source review")
    fields(protocol["claims"], {"primary_descriptive_contrasts", "controls", "not_established"}, "claim annotations")
    for value in protocol["claims"].values():
        require(type(value) is list and value and all(type(v) is str and v.strip() for v in value), "Invalid claim annotation")
    same(protocol["accounting"], {
        "proposer": "deterministic algorithm; no language model, API calls, or provider tokens",
        "allowed": ["exact weights", "public feature activations", "least-squares fitting", "validation feedback for six fixed candidates"],
        "session_counters": ["tool_calls", "weight_values", "execution_calls", "execution_examples", "feedback_calls",
                             "feedback_examples", "proposals", "input_tokens", "output_tokens"],
        "heldout": "one post-seal evaluator callback for all predeclared arms; actual model-example executions reported separately",
        "limits": "metadata and every returned candidate reveal full captured weights; those scalar values are charged",
        "training": "random frozen feature extractors; only output heads fitted by ridge on the training split",
        "split_lineage": "independent random streams within world; family checkpoints share teacher and input splits; no claim that families are independent tasks",
        "external_preregistration": False}, "execution and accounting declarations")


def dyadic(value: Any) -> float:
    require(type(value) is str, "Expected exact rational string")
    rational = Fraction(value)
    number = float(rational)
    require(math.isfinite(number) and Fraction.from_float(number) == rational, "Specification is not exactly binary64 representable")
    return number


def analytic_model(spec: dict[str, Any]) -> dict[str, Any]:
    require(type(spec) is dict, "Creation specification must be an object")
    if spec.get("kind") == "scaled_absolute":
        fields(spec, {"kind", "scale", "center"}, "absolute specification")
        scale, center = dyadic(spec["scale"]), dyadic(spec["center"])
        require(scale > 0, "Nonpositive absolute scale")
        tensors = {"w1": [[1.], [-1.]], "b1": [-center, center], "w2": [[scale, scale]], "b2": [0.]}
    elif spec.get("kind") == "clip":
        fields(spec, {"kind", "low", "high"}, "clip specification")
        low, high = dyadic(spec["low"]), dyadic(spec["high"])
        require(low < high, "Unordered clip limits")
        tensors = {"w1": [[1.], [1.]], "b1": [-low, -high], "w2": [[1., -1.]], "b2": [low]}
    else:
        raise ValueError("Unsupported analytic construction")
    result = {"schema": "wsp-network/1", "activation": "relu", "dtype": "float64",
              "input_names": ["x"], "output_names": ["y"], **tensors}
    return replace(result, model_arrays(result))


def formula_values(spec: dict[str, Any], inputs: np.ndarray) -> np.ndarray:
    analytic_model(spec)
    return (dyadic(spec["scale"]) * np.abs(inputs-dyadic(spec["center"])) if spec["kind"] == "scaled_absolute"
            else np.clip(inputs, dyadic(spec["low"]), dyadic(spec["high"])))


class Audit:
    def __init__(self, study: Path, frozen: Path, repo: Path | None = None):
        self.root, self.freeze_path, self.repo = study.resolve(), frozen.resolve(), repo
        self.freeze_root = self.freeze_path.parent
        require(not self.root.is_relative_to(self.freeze_root) and not self.freeze_root.is_relative_to(self.root),
                "Study and freeze trees must be distinct and non-nested")
        self.read_files: set[str] = set()
        self.checks: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.current_source_differences: list[str] = []
        self.protocol: dict[str, Any] = {}
        self.frozen: dict[str, Any] = {}
        self.results: dict[str, Any] = {}
        self.manifest_files: set[str] = set()

    def read(self, name: str) -> Any:
        self.read_files.add(name)
        return read_json(contained(self.root, name))

    def check(self, name: str, action: Callable[[], Any]) -> Any:
        try:
            result = action()
            self.checks.append({"check": name, "status": "passed"})
            return result
        except Exception as error:
            self.checks.append({"check": name, "status": "failed", "type": type(error).__name__, "message": str(error)})
            print(json.dumps({"audit_check": name, "status": "failed", "error": str(error)}), flush=True)
            return None

    def partial(self, name: str, explanation: str) -> None:
        self.checks.append({"check": name, "status": "not_verified", "reason": explanation})

    def provenance(self, expected_freeze: str | None, expected_protocol: str | None) -> None:
        self.frozen = read_json(self.freeze_path)
        fields(self.frozen, {"schema", "created_utc", "protocol_path", "protocol_sha256", "files", "environment",
                           "external_preregistration", "scope", "sha256"}, "freeze")
        signed_record(self.frozen, "freeze")
        same(self.frozen["schema"], "wsp-generic-freeze/2", "freeze schema")
        same(self.frozen["external_preregistration"], False, "freeze provenance")
        same(self.frozen["scope"], "local pre-result snapshot of generic code, tests, protocol, and source review", "freeze scope")
        if expected_freeze is not None:
            same(self.frozen["sha256"], sha(expected_freeze, "trusted freeze"), "trusted freeze identity")
        self.protocol = self.read("protocol.json")
        protocol_check(self.protocol)
        protocol_path = self.frozen["protocol_path"]
        frozen_protocol = read_json(contained(self.freeze_root / "sources", protocol_path))
        same(frozen_protocol, self.protocol, "frozen protocol bytes semantics")
        same(self.frozen["protocol_sha256"], digest(self.protocol), "protocol identity")
        if expected_protocol is not None:
            same(digest(self.protocol), sha(expected_protocol, "trusted protocol"), "trusted protocol identity")
        required = {f"rlcd/weight_space/{name}.py" for name in SOURCE_MODULES} | {
            "rlcd/__init__.py", "requirements.lock.txt", "requirements.txt", "pytest.ini", self.protocol["source_review"],
            "artifacts/weight-space/source-review-v2/provenance.json", "artifacts/weight-space/source-review-v2/review-evidence.json",
            protocol_path} | {"tests/" + name for name in SOURCE_TESTS}
        file_records = self.frozen["files"]
        require(type(file_records) is dict and set(file_records) == required, "Freeze required file membership differs")
        for name, info in file_records.items():
            fields(info, {"sha256", "bytes"}, "frozen file")
            sha(info["sha256"], "frozen file")
            count(info["bytes"], "frozen byte count")
            contained(self.freeze_root / "sources", name)
        same(inventory(self.freeze_root / "sources"), file_records, "frozen source tree inventory")
        expected_freeze_names = {"receipt.json"} | {"sources/" + name for name in required}
        same(sorted(inventory(self.freeze_root)), sorted(expected_freeze_names), "freeze tree membership")
        manifest = self.read("manifest.json")
        fields(manifest, {"schema", "freeze_sha256", "files", "sha256"}, "study manifest")
        signed_record(manifest, "study manifest")
        same([manifest["schema"], manifest["freeze_sha256"]], ["wsp-generic-study-files/2", self.frozen["sha256"]], "manifest provenance")
        require(type(manifest["files"]) is dict and manifest["files"], "Empty study manifest")
        for name, info in manifest["files"].items():
            contained(self.root, name)
            require(name.endswith(".json") and name != "manifest.json", "Unknown study file type or recursive manifest")
            fields(info, {"sha256", "bytes"}, "study file")
            sha(info["sha256"], "study file")
            count(info["bytes"], "study byte count")
        actual = inventory(self.root)
        actual.pop("manifest.json")
        same(actual, manifest["files"], "study tree inventory")
        self.manifest_files = set(manifest["files"]) | {"manifest.json"}
        self.results = self.read("results.json")
        start = self.read("execution-start.json")
        fields(start, {"started_utc", "freeze_sha256", "environment", "torch_cpu_threads", "provider_calls", "provider_tokens"}, "start record")
        same([start["freeze_sha256"], start["torch_cpu_threads"], start["provider_calls"], start["provider_tokens"]],
             [self.frozen["sha256"], 1, 0, 0], "execution provenance")
        same(start["started_utc"], self.results["started_utc"], "start timestamp")
        for environment in (self.frozen["environment"], start["environment"]):
            require(type(environment) is dict, "Missing runtime environment")
            for package in ("numpy", "torch"):
                same(environment["packages"][package], importlib.metadata.version(package), "numerical runtime " + package)
        created = datetime.fromisoformat(self.frozen["created_utc"])
        begun, finished = datetime.fromisoformat(start["started_utc"]), datetime.fromisoformat(self.results["finished_utc"])
        require(created.tzinfo is not None and begun.tzinfo is not None and finished.tzinfo is not None
                and created <= begun <= finished, "Declared evidence timestamps are inconsistent")
        if self.repo is not None:
            for name, info in file_records.items():
                current = contained(self.repo, name)
                if not current.is_file() or file_hash(current) != info["sha256"]:
                    self.current_source_differences.append(name)
        self.counts["frozen_files"] = len(required)
        self.counts["study_files"] = len(manifest["files"]) + 1
        self.counts["raw_file_hashes"] = len(required) + len(manifest["files"])

    def world(self, seed: int) -> dict[str, Any]:
        world = self.read(f"world-{seed}/world.json")
        fields(world, {"schema", "seed", "teacher", "input_domain", "target_region", "retention_region", "unmeasured_region", "splits"}, "world")
        same([world["schema"], world["seed"], world["input_domain"], world["target_region"], world["retention_region"], world["unmeasured_region"]],
             ["wsp-synthetic-world/2", seed, "16 independent uniform[-1,1] coordinates; CNN interprets a 1x4x4 field",
              "x0 > 0.35", "x0 < -0.1", "-0.1 <= x0 <= 0.35"], "world declarations")
        definitions = [("train", self.protocol["sizes"]["train"], "all")]
        definitions += [(f"{split}_{region}", self.protocol["sizes"][split], region)
                        for split in ("public", "validation", "test") for region in ("target", "retain")]
        definitions += [(f"sequence_{split}_{i}", self.protocol["sizes"]["sequence"], f"sequence_{i}")
                        for split in ("public", "test") for i in range(3)]
        require(set(world["splits"]) == {name for name, _, _ in definitions}, "Incomplete world splits")
        rows_seen = set()
        for name, size, region in definitions:
            split = world["splits"][name]
            fields(split, {"inputs", "labels", "sample_ids"}, "world split")
            x, y = array(split["inputs"]), array(split["labels"])
            require(x.shape == (size, 16) and y.shape == (size, 3) and np.all(np.abs(x) <= 1), "World sample dimensions/domain differ")
            if region == "target":
                require(np.all(x[:, 0] > .35), "Target region violated")
            elif region == "retain":
                require(np.all(x[:, 0] < -.1), "Retention region violated")
            elif region.startswith("sequence_"):
                region_id = int(region[-1])
                require(np.all((x[:, 0] < -.5, np.abs(x[:, 0]) < .2, x[:, 0] > .5)[region_id]), "Sequence region violated")
            close(y, teacher_values(x, world["teacher"]), "teacher labels " + name)
            same(split["sample_ids"], [f"world-{seed}/{name}/{i}" for i in range(size)], "sample identities")
            row_keys = {tuple(float(v) for v in row) for row in x}
            require(len(row_keys) == size and not rows_seen.intersection(row_keys), "Repeated exact input rows within/across splits")
            rows_seen.update(row_keys)
        self.counts["worlds"] += 1
        self.counts["world_splits"] += len(definitions)
        return world

    def training(self, path: str, family: str, world: dict[str, Any]) -> dict[str, Any]:
        base, record = self.read(path + "/base.json"), self.read(path + "/training.json")
        tensors, head = model_arrays(base), readout(base)
        if family in ("relu_mlp", "tanh_mlp"):
            spec = layers(base)
            widths = [16] + [tensors[layer["weight"]].shape[0] for layer in spec]
            same(widths, [16, 24, 3] if family == "relu_mlp" else [16, 20, 24, 3], "family widths")
            same([layer["activation"] for layer in spec], ["relu", "identity"] if family == "relu_mlp" else
                 ["tanh", "tanh", "identity"], "family activations")
        else:
            same(base["architecture"], CONV, "CNN family")
        phi = feature_values(base, world["splits"]["train"]["inputs"])
        expected_fields = {"training", "training_rows", "ridge", "design_rank", "condition_number", "train_rmse", "parameter_count",
                           "feature_width", "native_storage", "training_verified"}
        fields(record, expected_fields, "training record")
        same(record["training"], "random frozen extractor; ridge least-squares readout on training split only", "training declaration")
        same(record["training_rows"], len(phi), "training rows")
        same(record["ridge"], self.protocol["training_ridge"], "training ridge")
        same(record["parameter_count"], sum(v.size for v in tensors.values()), "parameter count")
        same(record["feature_width"], phi.shape[1], "feature width")
        same([record["native_storage"], record["training_verified"]], ["float64", False], "training status")
        require(count(record["design_rank"], "training rank") <= phi.shape[1]+1 and number(record["condition_number"]) >= 1, "Invalid training diagnostics")
        close(record["train_rmse"], rms(forward(base, world["splits"]["train"]["inputs"])-array(world["splits"]["train"]["labels"])), "training loss")
        # Training solver and initialization streams are deliberately not rerun.
        self.counts["trained_endpoints_replayed"] += 1
        return base

    def compilation(self, path: str, base: dict[str, Any], target: np.ndarray, desired: np.ndarray,
                    anchors: np.ndarray, retention: str, dtype: str = "float64", cast: str = "reject") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        proposal, program, candidate, record, certificate = (self.read(path + "/" + name + ".json")
            for name in ("proposal", "program", "candidate", "compilation", "retention"))
        same(literal_endpoint(base, program), candidate, "saved literal endpoint")
        lowering(base, proposal, record["lowering"], program, candidate)
        same(record["lowering"]["native_dtypes"], {name: dtype for name in model_arrays(base)}, "compilation native formats")
        same(record["lowering"]["cast"], cast, "compilation cast")
        target, desired, anchors = array(target), array(desired), array(anchors)
        expected_certificate = retention_receipt(base, candidate, anchors)
        same(certificate, expected_certificate, "exact endpoint retention receipt")
        head, settings = readout(base), self.protocol["compiler"]
        expected_fields = {"compiler", "head", "retention", "retention_weight", "ridge", "nullspace_rtol", "singular_cutoff",
            "retention_rank", "nullspace_dimension", "fit_rank", "condition_number", "target_rows", "retention_rows",
            "target_feature_sha256", "desired_delta_sha256", "retention_feature_sha256", "target_rmse_after_projection",
            "retention_rmse_after_projection", "edit_cells", "no_op", "lowering", "behavioral_status", "optimizer_verified",
            "feature_extractor_verified"}
        fields(record, expected_fields, "compilation diagnostics")
        same({key: record[key] for key in ("compiler", "head", "retention", "retention_weight", "ridge", "nullspace_rtol",
            "target_rows", "retention_rows", "target_feature_sha256", "desired_delta_sha256", "retention_feature_sha256",
            "edit_cells", "no_op", "behavioral_status", "optimizer_verified", "feature_extractor_verified")},
             {"compiler": "generic-frozen-feature-readout/1", "head": head, "retention": retention,
              "retention_weight": float(settings["retention_weight"]), "ridge": float(settings["ridge"]),
              "nullspace_rtol": float(settings["nullspace_rtol"]), "target_rows": len(target), "retention_rows": len(anchors),
              "target_feature_sha256": digest(target.tolist()), "desired_delta_sha256": digest(desired.tolist()),
              "retention_feature_sha256": digest(anchors.tolist()), "edit_cells": len(program["edits"]),
              "no_op": not program["edits"], "behavioral_status": "supplied_features_only", "optimizer_verified": False,
              "feature_extractor_verified": False}, "compilation bindings")
        require(len(proposal["commands"]) == 2, "Compiler must propose the head and its bias")
        for command, name in zip(proposal["commands"], (head["weight"], head["bias"])):
            fields(command, {"op", "tensor", "values"}, "compiler command")
            same([command["op"], command["tensor"]], ["add", name], "compiler write scope")
        dimensions = target.shape[1]+1
        if retention == "nullspace":
            # Singular values only: no fitted coefficients or new edit are produced.
            singular = np.linalg.svd(np.column_stack((anchors, np.ones(len(anchors)))), compute_uv=False)
            cutoff = settings["nullspace_rtol"] * float(singular[0])
            rank = int(np.count_nonzero(singular > cutoff))
            close(record["singular_cutoff"], cutoff, "nullspace threshold")
            same(record["retention_rank"], rank, "retention rank")
            same(record["nullspace_dimension"], dimensions-rank, "nullspace dimension")
            if rank == dimensions:
                same(candidate, base, "Full-rank retention must produce the unchanged endpoint")
                same([record["fit_rank"], record["condition_number"]], [0, None], "Infeasible fit diagnostics")
                self.counts["full_rank_noops"] += 1
        else:
            same([record["singular_cutoff"], record["retention_rank"], record["nullspace_dimension"]], [None, None, None], "non-nullspace diagnostics")
        require(count(record["fit_rank"], "fit rank") <= dimensions, "Invalid fitted rank")
        if record["condition_number"] is not None:
            require(number(record["condition_number"]) >= 1, "Invalid solver condition number")
        delta = np.asarray([[float(value) for value in row] for row in head_delta(base, candidate)])
        target_effect = np.column_stack((target, np.ones(len(target)))) @ delta.T
        retain_effect = np.column_stack((anchors, np.ones(len(anchors)))) @ delta.T
        close(record["target_rmse_after_projection"], rms(target_effect-desired), "compiler target residual")
        close(record["retention_rmse_after_projection"], rms(retain_effect), "compiler retention residual")
        self.counts.update(literal_programs=1, structured_lowerings=1, exact_retention_receipts=1)
        return candidate, record, certificate

    def main_case(self, path: str, base: dict[str, Any], world: dict[str, Any], family: str) -> dict[str, Any]:
        protocol, seed, head = self.protocol, world["seed"], readout(base)
        get = lambda name: array(world["splits"][name]["inputs"])
        target, retain = feature_values(base, get("public_target")), feature_values(base, get("public_retain"))
        desired = np.tile(protocol["shift"], (len(target), 1))
        nt, nr = get("validation_target"), get("validation_retain")
        test_t, test_r = get("test_target"), get("test_retain")
        parameters = sum(v.size for v in model_arrays(base).values())
        expected_task = {"schema": "wsp-task/1", "task_id": f"world-{seed}/{family}/conditional-edit", "mode": "edit",
            "objective": "Add the declared score shift on x0>0.35 while retaining x0<-0.1 behavior",
            "architecture": base["architecture"], "base_sha256": digest(base),
            "constraints": {"shift": protocol["shift"], "regime": "fitting_allowed_algorithmic_control", "head": head, "fixed_arms": list(ARMS)}}
        task = self.read(path + "/task.json")
        same(task, expected_task, "main task")
        limits = {"tool_calls": 50, "weight_values": 8*parameters, "feedback_calls": 6, "feedback_examples": 12*len(nt),
            "execution_calls": 2, "execution_examples": len(target)+len(retain), "proposals": 6, "input_tokens": 0, "output_tokens": 0}
        splits = [{"name": name, "sample_ids": split["sample_ids"], "lineage_ids": []} for name, split in world["splits"].items()]
        configuration = session_config(task, access_policy(feedback=True, activations=True), limits, splits,
                                       {"feedback": True, "forward": False, "activations": True})
        receipt = self.read(path + "/session.json")
        expected_actions = ["session_created", "metadata"] + ["weights"] * len(model_arrays(base)) + ["activations"] * 2
        expected_actions += [action for _ in ARMS for action in ("propose_program", "feedback")] + ["seal", "heldout_evaluation"]
        session = verify_session(receipt, task, base, configuration, "conditional-score-edit/2", "test_target", expected_actions)
        events = session["events"]
        same([event["request"]["tensor"] for event in events["weights"]], list(model_arrays(base)), "complete ordered weight access")
        for event, name in zip(events["activations"], ("public_target", "public_retain")):
            same(event["request"]["examples"], world["splits"][name]["inputs"], "public activation split")
            same(event["request"]["candidate_sha256"], None, "public parent activations")
        candidates, records, test_metrics = {}, {}, {}
        structural = protocol["selection"]["structural_policy"]
        policy = {"writable": [head["weight"], head["bias"]], "max_edits": parameters,
                  "max_parameter_abs": float(structural["max_parameter_abs"]), "max_change_abs": float(structural["max_change_abs"])}
        for index, arm in enumerate(ARMS):
            directory = path + "/" + arm
            if arm in ("unconstrained", "soft_retention", "nullspace"):
                anchors = retain[:protocol["compiler"]["nullspace_anchors"]] if arm == "nullspace" else retain
                candidate, _, _ = self.compilation(directory, base, target, desired, anchors,
                    {"unconstrained": "none", "soft_retention": "soft", "nullspace": "nullspace"}[arm])
                program = self.read(directory + "/program.json")
            else:
                candidate, program = self.read(directory + "/candidate.json"), self.read(directory + "/program.json")
                same(literal_endpoint(base, program), candidate, "main arm literal endpoint")
                self.counts["literal_programs"] += 1
                if arm == "no_op":
                    same(candidate, base, "no-op baseline")
                    same(program["edits"], [], "no-op literal")
                else:
                    proposal, details = self.read(directory + "/proposal.json"), self.read(directory + "/compilation.json")
                    lowering(base, proposal, details["lowering"], program, candidate)
                    same(details["lowering"]["native_dtypes"], {name: "float64" for name in model_arrays(base)}, "baseline precision")
                    same(details["lowering"]["cast"], "reject", "baseline cast")
                    head_delta(base, candidate)
                    self.counts["structured_lowerings"] += 1
                    if arm == "matched_random":
                        fields(details, {"reference_artifact_sha256", "random_seed", "requested_l2", "matching_scope", "lowering"}, "random diagnostics")
                        reference = candidates["soft_retention"]
                        same(details["reference_artifact_sha256"], digest(reference), "random matched reference")
                        same(details["random_seed"], seed*100+FAMILIES.index(family), "recorded random seed")
                        same(details["matching_scope"], "support and L2 before endpoint addition rounding; realized counts and norm reported separately", "random matching scope")
                        before, ref = model_arrays(base), model_arrays(reference)
                        offsets = [ref[name]-before[name] for name in (head["weight"], head["bias"])]
                        commands = proposal["commands"]
                        require(len(commands) == 2, "Random head command count")
                        requested = []
                        for command, name, delta in zip(commands, (head["weight"], head["bias"]), offsets):
                            fields(command, {"op", "tensor", "values"}, "random command")
                            same([command["op"], command["tensor"]], ["add", name], "random command scope")
                            value = array(command["values"])
                            require(value.shape == delta.shape and np.all(value[delta == 0] == 0), "Random requested support exceeds reference")
                            requested.append(value)
                        close(details["requested_l2"], l2(offsets), "random reference norm")
                        close(l2(requested), l2(offsets), "random requested norm matching")
                    else:
                        fields(details, {"method", "iterations", "step", "objective", "lowering"}, "gradient diagnostics")
                        same([details["method"], details["iterations"], details["objective"]],
                             ["gradient_descent", protocol["compiler"]["gradient_steps"],
                              "mean target square error + lambda mean anchor square drift + ridge norm"], "gradient declaration")
                        require(number(details["step"]) > 0, "Gradient step must be positive")
            admission = self.read(directory + "/admission.json")
            same(admission, admission_receipt(base, program, policy), "structural admission receipt")
            self.counts["structural_admissions"] += 1
            registered = receipt["candidates"][index]
            same(registered["artifact"], candidate, "registered main candidate")
            same(events["propose_program"][index]["request"]["program"], program, "saved/proposed program")
            validation = metrics(base, candidate, nt, nr, protocol["shift"])
            feedback = events["feedback"][index]
            same(feedback["request"]["candidate_sha256"], registered["sha256"], "feedback candidate")
            same(feedback["request"]["examples"], np.vstack((nt, nr)).tolist(), "validation-only feedback inputs")
            compare_record(feedback["result"], validation, "validation metrics")
            records[arm] = {"status": "completed", "candidate_sha256": registered["sha256"], "artifact_sha256": digest(candidate),
                "program_sha256": digest(program), "structural_accepted": admission["structural_accepted"], "validation": feedback["result"]}
            candidates[arm] = candidate
            test_metrics[arm] = metrics(base, candidate, test_t, test_r, protocol["shift"], world["teacher"])
        eligible = [arm for arm in ARMS if records[arm]["structural_accepted"] and
                    records[arm]["validation"]["retain_nrmse"] <= protocol["selection"]["max_retain_nrmse"] and
                    records[arm]["validation"]["target_nrmse"] <= protocol["selection"]["max_target_nrmse"]]
        selected = min(eligible, key=lambda arm: (records[arm]["validation"]["target_nrmse"],
            records[arm]["validation"]["delta_l2"], arm)) if eligible else "no_op"
        same(self.read(path + "/selection.json"), {"schema": "wsp-generic-selection/2", "selected_arm": selected,
            "candidate_sha256": records[selected]["candidate_sha256"], "rule": protocol["selection"], "validation": records,
            "heldout_observed": False}, "frozen validation selection")
        same(receipt["candidate_sha256"], records[selected]["candidate_sha256"], "selected session candidate")
        accounting = {"unique_test_examples": len(test_t)+len(test_r),
            "evaluated_splits": {"test_target": len(test_t), "test_retain": len(test_r)}, "candidate_evaluations": len(ARMS),
            "model_example_executions": 2*len(ARMS)*(len(test_t)+len(test_r))}
        compare_record(receipt["result"], {"arms": test_metrics, "selected_arm": selected,
            "selected_candidate_sha256": records[selected]["candidate_sha256"], "selected": test_metrics[selected], **accounting,
            "accounting_note": "Each arm executes base and candidate on both regions; selected metrics alias its fixed arm"}, "heldout outcomes")
        bound = None
        if family == "relu_mlp":
            certificate = self.read(path + "/soft-retention-domain-certificate.json")
            same(certificate, sequential_receipt(base, candidates["soft_retention"], [-1.]*16, [1.]*16), "sequential domain receipt")
            bound = {"output_bounds": [float(Fraction(value)) for value in certificate["output_bounds"]],
                     "scope": "real arithmetic on all [-1,1]^16; not an IEEE or task-loss proof"}
            self.counts["exact_domain_receipts"] += 1
        result = self.read(path + "/result.json")
        compare_record(result, {"seed": seed, "family": family, "base_sha256": digest(base), "selected_arm": selected,
            "arms": test_metrics, "usage": session["usage"], "failures": [], "real_domain_bound": bound,
            "validation_model_example_executions": 2*len(ARMS)*(len(nt)+len(nr)), "validation_execution_count_complete": True,
            "validation_feedback_attempts": {"attempted": len(ARMS), "completed": len(ARMS)}, "test_accounting": accounting}, "main result")
        self.counts.update(main_sessions=1, validated_candidates=len(ARMS), main_heldout_candidate_evaluations=len(ARMS))
        return result

    def ablations(self, path: str, base: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        get = lambda name: array(world["splits"][name]["inputs"])
        public_t, public_r, test_t, test_r = (get(name) for name in ("public_target", "public_retain", "test_target", "test_retain"))
        target, retain = feature_values(base, public_t), feature_values(base, public_r)
        desired = np.tile(self.protocol["shift"], (len(target), 1))
        sealed = {"feasibility": {}, "precision": {}, "adaptive_selection": False}
        feasibility, precision = [], []
        for n in self.protocol["feasibility"]:
            candidate, record, certificate = self.compilation(path + f"/anchors-{n}", base, target, desired, retain[:n], "nullspace")
            sealed["feasibility"][str(n)] = digest(candidate)
            feasibility.append({"anchors": n, "retention_rank": record["retention_rank"],
                "nullspace_dimension": record["nullspace_dimension"], "exactly_zero_on_anchors": certificate["exactly_zero_on_supplied_features"],
                "maximum_anchor_residual": max(float(Fraction(v)) for v in certificate["maximum_residuals_exact"]),
                "test": metrics(base, candidate, test_t, test_r, self.protocol["shift"])})
        all_test = np.vstack((test_t, test_r))
        for dtype in self.protocol["precision"]:
            directory = path + "/" + dtype
            parent = self.read(directory + "/parent.json")
            same(parent, replace(base, {name: native_round(value, dtype) for name, value in model_arrays(base).items()}), "precision parent")
            candidate, record, _ = self.compilation(directory, parent, feature_values(parent, public_t), desired,
                feature_values(parent, public_r), "soft", dtype, "round")
            sealed["precision"][dtype] = digest(candidate)
            lifted_base, lifted_candidate = forward(parent, all_test), forward(candidate, all_test)
            native_base, native_candidate = native_forward(parent, all_test, dtype), native_forward(candidate, all_test, dtype)
            lifted_drift, native_drift = lifted_candidate-lifted_base, native_candidate-native_base
            normalization = rms(self.protocol["shift"])
            precision.append({"dtype": dtype, "parent_sha256": digest(parent), "candidate_sha256": digest(candidate),
                "lifted": metrics(parent, candidate, test_t, test_r, self.protocol["shift"]),
                "native_target_nrmse": rms(native_drift[:len(test_t)]-array(self.protocol["shift"]))/normalization,
                "native_retain_nrmse": rms(native_drift[len(test_t):])/normalization,
                "native_base_vs_lifted_max_abs": float(np.max(np.abs(native_base-lifted_base))),
                "native_candidate_vs_lifted_max_abs": float(np.max(np.abs(native_candidate-lifted_candidate))),
                "native_drift_vs_lifted_max_abs": float(np.max(np.abs(native_drift-lifted_drift))),
                "lowering": record["lowering"], "floating_execution_verified": False,
                "scope": "CPU PyTorch measurement; compiler features evaluated in lifted float64 semantics"})
        same(self.read(path + "/sealed.json"), sealed, "ablation endpoint commitments")
        result = self.read(path + "/result.json")
        compare_record(result, {"feasibility": feasibility, "precision": precision,
            "split_reuse": "same heldout inputs as main arms; descriptive ablations, not independent trials",
            "feasibility_model_example_executions": 2*len(feasibility)*len(all_test),
            "precision_model_example_executions": 6*len(precision)*len(all_test)}, "ablation outcomes")
        self.counts.update(feasibility_conditions=len(feasibility), native_precision_conditions=len(precision))
        return result

    def sequence(self, path: str, base: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        get = lambda name: array(world["splits"][name]["inputs"])
        public, test = [get(f"sequence_public_{i}") for i in range(3)], [get(f"sequence_test_{i}") for i in range(3)]
        phi = [feature_values(base, x) for x in public]
        public_baseline, test_baseline = [forward(base, x) for x in public], [forward(base, x) for x in test]
        sealed = {"candidates": {}, "adaptive_selection": False,
            "retention_objective": "preserve current outputs in the other two public regions", "all_chains_fixed_before_heldout": True}
        methods = {}
        for method in self.protocol["sequence"]["methods"]:
            current, stages = base, []
            directory = path + "/" + method
            for step, shift in enumerate(self.protocol["sequence"]["shifts"]):
                desired = public_baseline[step]+array(shift)-forward(current, public[step])
                retain = np.vstack([phi[i] for i in range(3) if i != step])
                current, _, _ = self.compilation(directory + f"/step-{step+1}", current, phi[step], desired,
                    retain, "none" if method == "unconstrained" else "soft")
                stages.append(current)
            combined, inverse = self.read(directory + "/composed.json"), self.read(directory + "/rollback.json")
            same(combined, literal_difference(base, current, "Compose three fixed conditional edits"), "composed literal program")
            same(literal_endpoint(base, combined), current, "composed endpoint")
            same(inverse, literal_difference(current, base, "Rollback: Compose three fixed conditional edits"), "inverse literal program")
            same(literal_endpoint(current, inverse), base, "rollback endpoint")
            sealed["candidates"][method] = [digest(stage) for stage in stages]
            outputs = [[forward(stage, x) for x in test] for stage in stages]
            stage_metrics = []
            for step, region_outputs in enumerate(outputs):
                regions = []
                for region, y in enumerate(region_outputs):
                    shift = array(self.protocol["sequence"]["shifts"][region])
                    normalization = rms(shift)
                    desired = shift if region <= step else np.zeros(3)
                    record = {"region": region, "edited": region <= step,
                              "desired_behavior_nrmse": rms(y-test_baseline[region]-desired)/normalization}
                    if region < step:
                        record["drift_since_acquisition_nrmse"] = rms(y-outputs[region][region])/normalization
                    regions.append(record)
                stage_metrics.append({"step": step+1, "regions": regions})
            methods[method] = {"stages": stage_metrics,
                "final_mean_target_nrmse": float(np.mean([r["desired_behavior_nrmse"] for r in stage_metrics[-1]["regions"]])),
                "final_prior_region_interference_nrmse": float(np.mean([r["drift_since_acquisition_nrmse"] for r in stage_metrics[-1]["regions"][:-1]])),
                "composition_exact": True, "rollback_exact": True}
            self.counts.update(sequential_chains=1, literal_programs=2, exact_compositions=1, exact_rollbacks=1)
        same(self.read(path + "/sealed.json"), sealed, "sequence endpoint commitments")
        result = self.read(path + "/result.json")
        compare_record(result, {"methods": methods, "model_example_executions": (1+3*len(methods))*sum(len(x) for x in test),
            "selection": "none; fixed chains, all heldout inputs consumed after sealing"}, "sequence outcomes")
        return result

    def auxiliary_session(self, path: str, task: dict[str, Any], base: dict[str, Any] | None,
                          configuration: dict[str, Any], evaluator: str, actions: list[str]) -> dict[str, Any]:
        same(self.read(path + "/task.json"), task, "auxiliary task")
        receipt = self.read(path + "/receipt.json")
        verify_session(receipt, task, base, configuration, evaluator, "test", actions)
        same(self.read(path + "/session.json"), {"configuration": configuration, "usage": receipt["usage"],
            "sealed": True, "receipt_status": "succeeded"}, "auxiliary session state")
        same(self.read(path + "/candidates.json"), receipt["candidates"], "auxiliary candidate inventory")
        same(self.read(path + "/trace.json"), receipt["trace"], "auxiliary trace file")
        self.counts["auxiliary_sessions"] += 1
        return receipt

    def auxiliary_read(self, path: str, seed: int, definition: dict[str, Any]) -> dict[str, Any]:
        case_id, sign, label, access, mode = (definition[key] for key in ("case_id", "sign", "label", "access", "mode"))
        base = self.read(path + "/base.json")
        tensors = model_arrays(base)
        same(base["architecture"], "affine/1", "affine read architecture")
        same(base["metadata"], {"formula": "x @ weight.T + bias", "input_dimension": 16, "output_dimension": 3,
            "semantic_names": {"inputs": [f"x{i}" for i in range(16)], "outputs": [f"y{i}" for i in range(3)]}}, "affine interface")
        require(tensors["weight"].shape == (3, 16) and tensors["bias"].shape == (3,), "Affine fixture shape")
        same(float(tensors["weight"][1, 7]), sign*.0004, "designated coefficient")
        other = tensors["weight"].copy()
        other[1, 7] = 0.
        require(np.all(other*8 == np.round(other*8)) and np.all(np.abs(other) <= 2)
                and np.all(tensors["bias"]*8 == np.round(tensors["bias"]*8)) and np.all(np.abs(tensors["bias"]) <= 1), "Affine fixture non-designated values")
        task = {"schema": "wsp-task/1", "task_id": f"auxiliary:{seed}:{case_id}", "mode": mode,
            "objective": "Read the 16-to-3 affine response and the sign of weight[1,7]; abstain on sign when its displayed value is zero",
            "architecture": "affine/1", "base_sha256": digest(base), "constraints": {"privileged_hints": [],
                "designated_coefficient": [1, 7], "semantic_names": "Opaque input/output indices only", "fitting_allowed": False}}
        limits = dict.fromkeys(COUNTERS, 0) | {"tool_calls": 3, "weight_values": 51, "proposals": 1}
        split = [{"name": "test", "sample_ids": [f"{case_id}:{i}" for i in range(128)], "lineage_ids": []}]
        configuration = session_config(task, access_policy(metadata=False, weight_decimals=None if access == "exact" else 3),
            limits, split, {"feedback": False, "forward": False, "activations": False})
        receipt = self.auxiliary_session(path, task, base, configuration, "affine-sign-and-response/1",
            ["session_created", "provider_usage", "weights", "weights", "submit_answer", "seal", "heldout_evaluation"])
        reads = [event for event in receipt["trace"] if event["action"] == "weights"]
        same([event["request"]["tensor"] for event in reads], ["weight", "bias"], "affine read order")
        weight, bias = reads[0]["result"], reads[1]["result"]
        observed = weight[1][7]
        answer = {"schema": "wsp-affine-reading/1", "weight": weight, "bias": bias, "designated_coefficient": [1, 7],
            "sign": None if observed == 0 else (1 if observed > 0 else -1), "abstain": observed == 0,
            "decision_rule": "Abstain at displayed zero; otherwise report the displayed sign"}
        same(receipt["candidate"]["answer"], answer, "affine reading decision")
        hidden = self.read(path + "/heldout.json")
        fields(hidden, {"seed_entropy", "rng", "distribution", "inputs", "input_visibility", "paired_across_access_conditions",
            "reference_outputs", "reconstructed_outputs", "reconstructed_artifact", "evaluator_accounting"}, "affine heldout evidence")
        same([hidden["seed_entropy"], hidden["rng"], hidden["distribution"], hidden["input_visibility"], hidden["paired_across_access_conditions"]],
             [[seed, 1200 if label == "a" else 1201], "numpy.PCG64", "uniform(-2,2), shape=(128,16)", "evaluator_only_until_seal", True], "affine input provenance")
        x = array(hidden["inputs"])
        require(x.shape == (128, 16) and np.all(np.abs(x) <= 2), "Affine hidden domain/shape")
        reconstructed = replace(base, {"weight": array(weight), "bias": array(bias)})
        same(hidden["reconstructed_artifact"], reconstructed, "reconstructed affine endpoint")
        expected, actual = forward(base, x), forward(reconstructed, x)
        close(hidden["reference_outputs"], expected, "affine reference outputs")
        close(hidden["reconstructed_outputs"], actual, "affine reconstructed outputs")
        same(hidden["evaluator_accounting"], {"affine_forward_batches": 2, "evaluated_model_examples": 256}, "affine execution accounting")
        outcome = {"exact_sign": sign, "reported_sign": answer["sign"], "abstained": answer["abstain"],
            "sign_correct": None if answer["abstain"] else answer["sign"] == sign,
            "coefficient_absolute_error": abs(float(model_arrays(reconstructed)["weight"][1, 7]-tensors["weight"][1, 7])),
            "hidden_response": response_metrics(expected, actual), "formal_certificate": None}
        compare_record(receipt["result"], outcome, "affine evaluator result")
        result = self.read(path + "/result.json")
        compare_record(result, {"case_id": case_id, "family": "affine_sign", "mode": mode, "access": access, "status": "succeeded",
            "candidate_ids": [receipt["candidate_sha256"]], "abstained": answer["abstain"], "base_sha256": digest(base),
            "receipt_sha256": receipt["sha256"], "metrics": outcome, "error": None}, "affine case result")
        self.counts["auxiliary_read_cases"] += 1
        return result

    def auxiliary_create(self, path: str, seed: int, definition: dict[str, Any]) -> dict[str, Any]:
        case_id, spec, index = definition["case_id"], definition["spec"], definition["index"]
        same(self.read(path + "/spec.json"), spec, "analytic formula specification")
        analytic = analytic_model(spec)
        if spec["kind"] == "scaled_absolute":
            scale, center = dyadic(spec["scale"]), dyadic(spec["center"])
            require(1 <= scale*4 <= 8 and scale*4 == round(scale*4) and -8 <= center*4 <= 8
                    and center*4 == round(center*4), "Absolute fixture outside declared parameter support")
        else:
            low, high = dyadic(spec["low"]), dyadic(spec["high"])
            require(-8 <= low*4 <= 0 and low*4 == round(low*4) and 1 <= (high-low)*4 <= 8
                    and (high-low)*4 == round((high-low)*4), "Clip fixture outside declared parameter support")
        arms = ("zero", "random", "analytic")
        task = {"schema": "wsp-task/1", "task_id": f"auxiliary:{seed}:{case_id}", "mode": "create",
            "objective": "Construct the declared scalar formula using two ReLU units", "architecture": "dense-relu-dense/1",
            "base_sha256": None, "constraints": {"formula": spec, "selection": "analytic arm fixed before evaluation",
                "training_allowed": False, "feedback_allowed": False,
                "privileged_hints": ["Exact target formula and two-ReLU construction family"],
                "semantic_names": {"input": "x", "output": "y"}}}
        limits = dict.fromkeys(COUNTERS, 0) | {"tool_calls": 3, "proposals": 3}
        splits = [{"name": "test", "sample_ids": [f"{case_id}:{i}" for i in range(128)], "lineage_ids": []}]
        configuration = session_config(task, access_policy(metadata=False, weights=False), limits, splits,
            {"feedback": False, "forward": False, "activations": False})
        receipt = self.auxiliary_session(path, task, None, configuration, "analytic-construction-response/1",
            ["session_created", "provider_usage", "propose_artifact", "propose_artifact", "propose_artifact", "seal", "heldout_evaluation"])
        artifacts = {arm: self.read(path + f"/{arm}.network.json") for arm in arms}
        zero = replace(analytic, {name: np.zeros_like(value) for name, value in model_arrays(analytic).items()})
        same(artifacts["zero"], zero, "zero construction control")
        same(artifacts["analytic"], analytic, "analytic construction formula")
        compatible(zero, artifacts["random"])
        for value in model_arrays(artifacts["random"]).values():
            require(np.all(np.abs(value) <= 2) and np.all(value*4 == np.round(value*4)), "Random control outside recorded dyadic support")
        construction = self.read(path + "/construction.json")
        same(construction, {"method": "fixed_formula_no_fitting", "predeclared_arms": list(arms), "selected_arm": "analytic",
            "selection_uses_heldout": False, "random_seed_entropy": [seed, 2200+index], "rng": "numpy.PCG64",
            "random_parameters": "uniform integers [-8,8] divided by 4, tensor order w1,b1,w2,b2",
            "replay_program_parent_sha256": digest(zero),
            "replay_program_scope": "Reproduce each artifact from a zero reference; create task has no parent", "formal_certificate": None}, "construction declaration")
        candidates = {}
        for arm, candidate in zip(arms, receipt["candidates"]):
            same(candidate["artifact"], artifacts[arm], "creation candidate endpoint")
            program = self.read(path + f"/{arm}.program.json")
            same(program, literal_difference(zero, artifacts[arm], f"Replay the fixed {arm} construction from zero"), "creation replay literal")
            same(literal_endpoint(zero, program), artifacts[arm], "creation replay endpoint")
            candidates[arm] = candidate
            self.counts["literal_programs"] += 1
        same(receipt["candidate_sha256"], candidates["analytic"]["sha256"], "analytic predeclared selection")
        hidden = self.read(path + "/heldout.json")
        fields(hidden, {"seed_entropy", "rng", "distribution", "inputs", "input_visibility", "reference_outputs", "arm_outputs", "evaluator_accounting"}, "creation heldout evidence")
        same([hidden["seed_entropy"], hidden["rng"], hidden["distribution"], hidden["input_visibility"]],
            [[seed, 2300+index], "numpy.PCG64", "uniform(-4,4), shape=(128,1)", "evaluator_only_until_seal"], "creation input provenance")
        x = array(hidden["inputs"])
        require(x.shape == (128, 1) and np.all(np.abs(x) <= 4), "Creation hidden domain/shape")
        expected = formula_values(spec, x)
        close(hidden["reference_outputs"], expected, "formula outputs")
        require(set(hidden["arm_outputs"]) == set(arms), "Missing creation output arm")
        results = {}
        for arm in arms:
            actual = forward(artifacts[arm], x)
            close(hidden["arm_outputs"][arm], actual, "creation outputs " + arm)
            results[arm] = {"candidate_sha256": candidates[arm]["sha256"], "artifact_sha256": digest(artifacts[arm]),
                           "hidden_response": response_metrics(expected, actual)}
        same(hidden["evaluator_accounting"], {"network_forward_batches": 3, "evaluated_model_examples": 384}, "creation execution accounting")
        outcome = {"selected_arm": "analytic", "selection_predeclared": True, "arms": results, "formal_certificate": None}
        compare_record(receipt["result"], outcome, "creation evaluator result")
        result = self.read(path + "/result.json")
        compare_record(result, {"case_id": case_id, "family": "analytic_creation", "mode": "create", "formula": spec["kind"],
            "status": "succeeded", "candidate_ids": [candidates[arm]["sha256"] for arm in arms],
            "selected_candidate_id": candidates["analytic"]["sha256"], "receipt_sha256": receipt["sha256"],
            "metrics": outcome, "error": None}, "creation case result")
        self.counts["auxiliary_creation_cases"] += 1
        return result

    def auxiliary(self, seed: int) -> dict[str, Any]:
        root = f"world-{seed}/auxiliary"
        protocol, summary = self.read(root + "/protocol.json"), self.read(root + "/summary.json")
        signed_record(summary, "auxiliary summary", "summary_sha256")
        definitions = protocol["cases"]
        require(type(definitions) is list and len(definitions) == 6, "Missing auxiliary cases")
        readings = [{"case_id": f"affine-{label}-{access}", "family": "affine_sign", "sign": sign,
            "label": label, "access": access, "mode": mode}
            for sign, label, mode in ((1, "a", "read"), (-1, "b", "diagnose")) for access in ("exact", "rounded3")]
        same(definitions[:4], readings, "auxiliary read definitions")
        for index, (definition, kind) in enumerate(zip(definitions[4:], ("scaled_absolute", "clip"))):
            fields(definition, {"case_id", "family", "spec", "index"}, "creation definition")
            same([definition["case_id"], definition["family"], definition["index"], definition["spec"]["kind"]],
                 [f"create-{kind}", "analytic_creation", index, kind], "creation definition identity")
            analytic_model(definition["spec"])
        same(protocol, {"schema": "wsp-auxiliary-protocol/1", "seed": seed, "cases": definitions,
            "hidden_examples_per_case": 128, "absolute_tolerance": 1e-12, "creation_arms": ["zero", "random", "analytic"],
            "creation_selection": "analytic_a_priori", "programmer_kind": "deterministic_fixture", "model_ability_measured": False,
            "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "affine_streams": {"weights": 1100, "hidden_a": 1200, "hidden_b": 1201},
            "creation_streams": {"specification": 2100, "random_parameters": [2200, 2201], "hidden_inputs": [2300, 2301]},
            "enforcement": "cooperative_same_process_interface", "formal_certificate": None}, "auxiliary protocol")
        outcomes = []
        for definition in definitions:
            path = root + "/" + definition["case_id"]
            same(self.read(path + "/attempt.json"), definition | {"world_seed": seed}, "auxiliary attempt")
            outcomes.append(self.auxiliary_read(path, seed, definition) if definition["family"] == "affine_sign" else
                            self.auxiliary_create(path, seed, definition))
        for label in ("a", "b"):
            exact = root + f"/affine-{label}-exact"
            rounded = root + f"/affine-{label}-rounded3"
            same(self.read(exact + "/base.json"), self.read(rounded + "/base.json"), "paired access parents")
            same(self.read(exact + "/heldout.json")["inputs"], self.read(rounded + "/heldout.json")["inputs"], "paired access evaluation inputs")
        positive, negative = self.read(root + "/affine-a-exact/base.json"), self.read(root + "/affine-b-exact/base.json")
        positive_tensors = model_arrays(positive)
        positive_tensors["weight"][1, 7] *= -1
        same(replace(positive, positive_tensors), negative, "paired sign fixtures differ only in designated coefficient")
        expected = {"schema": "wsp-auxiliary-study/1", "seed": seed, "cases": 6,
            "case_type_counts": {"read": 2, "diagnose": 2, "create": 2}, "errors": 0,
            "correct": {"coefficient_sign": sum(case["metrics"]["sign_correct"] is True for case in outcomes[:4]),
                        "analytic_creation": sum(case["metrics"]["arms"]["analytic"]["hidden_response"]["within_tolerance"] for case in outcomes[4:])},
            "correct_scope": {"coefficient_sign": "Nonabstained exact-sign classifications, out of four cases",
                              "analytic_creation": "Selected analytic response within tolerance, out of two cases"},
            "abstentions": sum(case["abstained"] for case in outcomes[:4]),
            "incorrect_signs": sum(case["metrics"]["sign_correct"] is False for case in outcomes[:4]),
            "candidate_ids": {case["case_id"]: case["candidate_ids"] for case in outcomes}, "case_results": outcomes,
            "provider": {"kind": "deterministic_fixture", "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "model_ability_measured": False},
            "formal_certificate": None, "evidence_manifest": "manifest.json"}
        same(summary, expected | {"summary_sha256": digest(expected)}, "auxiliary aggregate")
        manifest = self.read(root + "/manifest.json")
        fields(manifest, {"schema", "files"}, "auxiliary manifest")
        same(manifest["schema"], "wsp-auxiliary-manifest/1", "auxiliary manifest schema")
        actual = inventory(contained(self.root, root))
        actual.pop("manifest.json")
        same(manifest["files"], {name: info["sha256"] for name, info in actual.items()}, "auxiliary manifest inventory")
        self.counts["auxiliary_worlds"] += 1
        return summary

    def case_record(self, path: str, seed: int, family: str) -> dict[str, Any]:
        case = self.read(path + "/case.json")
        fields(case, {"seed", "family", "training", "main", "ablations", "sequence", "status"}, "checkpoint record")
        same([case["seed"], case["family"]], [seed, family], "checkpoint identity")
        for component in ("training", "main", "ablations", "sequence"):
            value = case[component]
            require(value is None or type(value) is dict, "Malformed checkpoint component: " + component)
            if value is not None:
                name = "training.json" if component == "training" else component + "/result.json"
                same(value, self.read(path + "/" + name), "checkpoint component binding " + component)
        status = ("failed" if case["training"] is None else
                  "completed" if all(case[name] is not None for name in ("main", "ablations", "sequence"))
                  and case["main"]["failures"] == [] else "partial")
        same(case["status"], status, "checkpoint completion status")
        return case

    def aggregate(self, cases: list[dict[str, Any]], auxiliary: list[dict[str, Any]]) -> None:
        """Recompute reported statistics from the bound checkpoint outcomes."""
        same(self.results["cases"], cases, "root checkpoint copies")
        same(self.results["auxiliary"], auxiliary, "root auxiliary copies")
        expected = summary_values(cases, self.protocol) | {
            "schema": "wsp-generic-study-results/2", "started_utc": self.results["started_utc"],
            "finished_utc": self.results["finished_utc"], "freeze_sha256": self.frozen["sha256"],
            "protocol_sha256": digest(self.protocol), "sources_unchanged": True, "failures": [],
            "complete": True, "auxiliary": auxiliary, "cases": cases,
            "limitations": self.protocol["claims"]["not_established"]}
        compare_record(self.results, expected, "root results and clustered summary")
        require(all(case["status"] == "completed" for case in cases)
                and len(auxiliary) == len(self.protocol["seeds"])
                and all(item["result"]["errors"] == 0 and item["result"]["cases"] == 6 for item in auxiliary),
                "Root results claim a complete run with incomplete components")
        self.counts["paired_cluster_contrasts"] = len(expected["contrasts"])
        self.counts["bootstrap_draws_per_contrast"] = self.protocol["bootstrap"]["draws"]
        self.counts["complete_worlds_for_cluster_inference"] = expected["complete_worlds_for_cluster_inference"]

    def coverage(self) -> None:
        unvisited = sorted(self.manifest_files-self.read_files)
        unexpected = sorted(self.read_files-self.manifest_files)
        require(not unvisited and not unexpected,
                f"Evidence coverage differs; unvisited={unvisited}; unexpected_reads={unexpected}")
        self.counts["evidence_json_files_visited"] = len(self.read_files)

    def run(self, expected_freeze: str | None = None, expected_protocol: str | None = None) -> None:
        self.check("frozen_sources_protocol_and_file_manifest", lambda: self.provenance(expected_freeze, expected_protocol))
        if self.checks[-1]["status"] != "passed":
            self.partial("saved_evidence_replay", "The freeze/protocol/manifest prerequisites did not pass")
            return
        cases, auxiliary = [], []
        for seed in self.protocol["seeds"]:
            world = self.check(f"world-{seed}/world", lambda: self.world(seed))
            for family in self.protocol["families"]:
                path = f"world-{seed}/{family}"
                case = self.check(path + "/case_record", lambda: self.case_record(path, seed, family))
                if case is not None:
                    cases.append(case)
                if world is None:
                    self.partial(path + "/endpoint_and_outcomes", "World inputs and labels were not verified")
                    continue
                base = self.check(path + "/training_endpoint", lambda: self.training(path, family, world))
                if base is None:
                    self.partial(path + "/candidate_and_outcomes", "Saved parent endpoint was not verified")
                    continue
                self.check(path + "/main", lambda: self.main_case(path + "/main", base, world, family))
                self.check(path + "/ablations", lambda: self.ablations(path + "/ablations", base, world))
                self.check(path + "/sequence", lambda: self.sequence(path + "/sequence", base, world))
            aux = self.check(f"world-{seed}/auxiliary", lambda: self.auxiliary(seed))
            if aux is not None:
                auxiliary.append({"seed": seed, "result": aux})
            print(json.dumps({"audit_world": seed, "completed_checks": len(self.checks),
                              "failed_checks": sum(item["status"] == "failed" for item in self.checks)}), flush=True)
        self.check("root_results_and_world_cluster_statistics", lambda: self.aggregate(cases, auxiliary))
        self.check("all_persisted_evidence_visited", self.coverage)


AUDIT_LIMITATIONS = [
    "Saved evidence is replayed; no worlds, initial weights, proposals, or optimization paths are regenerated.",
    "Random stream declarations and distribution support are checked, but RNG seed-to-sample generation is not independently verified.",
    "Training and solver optimality are not proved. Recorded fit ranks/condition numbers and gradient histories are not fully recomputed; nullspace anchor rank is recomputed.",
    "Numerical output replay uses independent Python equations but shares NumPy and, for native CPU execution/casts, PyTorch with the study.",
    "Floating observations use atol=1e-12 and rtol=1e-10. Endpoint hashes, literal edits, types, counters, and rational certificate payloads use exact checks.",
    "Exact rational receipt replay does not verify the feature extractor, all-input floating execution, Python implementation, or an optimizer.",
    "The formal Lean build and axiom audit have separate receipts and are not rerun by this auditor.",
    "Freeze hashes and declared chronology are provenance checks, not trusted timestamps, security attestation, or proof that frozen source was executed.",
    "Current repository source differences are informational; frozen copies and persisted evidence are the audit inputs.",
    "Complete successful session traces are supported. Failure traces and incomplete studies cannot receive a fully passed audit.",
    "Bootstrap intervals describe saved synthetic worlds and do not establish population, real-data, large-model, or language-model programming claims.",
]


def output_path(path: Path, study: Path, frozen: Path) -> Path:
    target = path.resolve()
    require(target.suffix.lower() == ".json", "Audit output must be a new .json file")
    require(not target.exists(), "Audit output already exists; choose a new path")
    require(not target.is_relative_to(study.resolve()) and not target.is_relative_to(frozen.resolve()),
            "Audit output must be outside both immutable input trees")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path, help="Source-freeze receipt.json or its directory")
    parser.add_argument("--out", required=True, type=Path, help="New JSON report outside both input trees")
    parser.add_argument("--repo", type=Path, help="Optional current checkout for informational source comparison")
    parser.add_argument("--expected-freeze", help="Previously recorded canonical freeze receipt hash")
    parser.add_argument("--expected-protocol", help="Previously recorded canonical protocol hash")
    args = parser.parse_args(argv)
    try:
        frozen = args.freeze / "receipt.json" if args.freeze.is_dir() else args.freeze
        audit = Audit(args.study_root, frozen, args.repo)
        target = output_path(args.out, audit.root, audit.freeze_root)
        source = Path(__file__).resolve()
        source_before = file_hash(source)
        started = datetime.now(timezone.utc).isoformat()
        before = {"study": inventory(audit.root), "freeze": inventory(audit.freeze_root)}
        import torch
        torch.set_num_threads(1)
        environment = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                       "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch")},
                       "torch_cpu_threads": torch.get_num_threads()}
        audit.check("no_rlcd_module_imported_before_replay", lambda: require(
            not any(name == "rlcd" or name.startswith("rlcd.") for name in sys.modules), "RLCD module imported"))
        audit.run(args.expected_freeze, args.expected_protocol)
        after = {"study": inventory(audit.root), "freeze": inventory(audit.freeze_root)}
        source_after = file_hash(source)
        audit.check("immutable_study_and_freeze_trees", lambda: same(after, before, "before/after input inventory"))
        audit.check("unchanged_auditor_source", lambda: same(source_after, source_before, "auditor source"))
        audit.check("no_rlcd_module_imported_after_replay", lambda: require(
            not any(name == "rlcd" or name.startswith("rlcd.") for name in sys.modules), "RLCD module imported"))
        statuses = dict(Counter(item["status"] for item in audit.checks))
        passed = all(item["status"] == "passed" for item in audit.checks)
        payload = {
            "schema": "wsp-independent-saved-evidence-audit/2", "status": "passed" if passed else "not_passed",
            "complete": passed, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "study_root": str(audit.root), "freeze_receipt": str(audit.freeze_path),
            "auditor_source": {"path": str(source), "sha256_before": source_before, "sha256_after": source_after},
            "trusted_expected_hashes": {"freeze": args.expected_freeze, "protocol": args.expected_protocol},
            "frozen_receipt_sha256": audit.frozen.get("sha256"),
            "protocol_sha256": digest(audit.protocol) if audit.protocol else None,
            "environment": environment, "check_status_counts": statuses, "checks": audit.checks,
            "counts": dict(audit.counts), "current_source_differences": audit.current_source_differences,
            "input_inventories": {"before": before, "after": after},
            "input_inventory_sha256": {"before": digest(before), "after": digest(after)},
            "unvisited_evidence_files": sorted(audit.manifest_files-audit.read_files),
            "numerical_tolerances": {"absolute": ATOL, "relative": RTOL},
            "limitations": AUDIT_LIMITATIONS,
        }
        report = signed(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"audit_report": str(target), "status": report["status"], "sha256": report["sha256"],
                          "check_status_counts": statuses, "counts": dict(audit.counts)}, indent=2), flush=True)
        return 0 if passed else 1
    except Exception as error:
        print(json.dumps({"audit_status": "error", "type": type(error).__name__, "message": str(error)}), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
