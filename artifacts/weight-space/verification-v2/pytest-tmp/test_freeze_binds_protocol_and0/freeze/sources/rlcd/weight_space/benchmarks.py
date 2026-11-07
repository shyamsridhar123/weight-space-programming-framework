"""Small explicit task fixtures for the generic study, never a universal adapter.

The feature extractors are random and frozen; only their output heads are fitted
on the training split. This is an algorithmic control study, not an LLM benchmark
or a reproduction of the source report's no-regression Go arm.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .artifact import TensorArtifact
from .features import ReadoutHead, _rms
from .model import digest
from .sequential import (SEQUENTIAL_ARCHITECTURE, sequential_artifact, sequential_forward,
                         validate_sequential)
from .structured import StructuredProgram, lower_structured, native_round


FAMILIES = ("relu_mlp", "tanh_mlp", "conv_tanh")
CONV_ARCHITECTURE = "study-conv2d-tanh-flatten-linear/1"


def _l2(arrays: list[np.ndarray]) -> float:
    values = np.concatenate([array.ravel() for array in arrays])
    result = _rms(values)*np.sqrt(values.size)
    if not np.isfinite(result):
        raise ValueError("Parameter norm exceeds the finite numerical metric range")
    return float(result)


def oracle(inputs: Any, teacher: dict[str, Any]) -> np.ndarray:
    x = np.asarray(inputs, dtype=np.float64)
    return x @ np.asarray(teacher["linear"]) + .2*np.sin(x @ np.asarray(teacher["nonlinear"]))


def draw_region(rng: np.random.Generator, count: int, region: str) -> np.ndarray:
    accepted = []
    while sum(len(batch) for batch in accepted) < count:
        x = rng.uniform(-1., 1., (max(count*2, 64), 16))
        if region == "target":
            x = x[x[:, 0] > .35]
        elif region == "retain":
            x = x[x[:, 0] < -.1]
        elif region.startswith("sequence_"):
            index = int(region[-1])
            masks = (x[:, 0] < -.5, np.abs(x[:, 0]) < .2, x[:, 0] > .5)
            x = x[masks[index]]
        elif region != "all":
            raise ValueError("Unknown benchmark region")
        accepted.append(x)
    return np.concatenate(accepted)[:count]


def make_world(seed: int, sizes: dict[str, int]) -> dict[str, Any]:
    teacher_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    teacher = {"linear": (teacher_rng.normal(size=(16, 3))/4).tolist(),
               "nonlinear": (teacher_rng.normal(size=(16, 3))/4).tolist()}
    definitions = [("train", sizes["train"], "all")]
    for split in ("public", "validation", "test"):
        for region in ("target", "retain"):
            definitions.append((f"{split}_{region}", sizes[split], region))
    # Separate inputs for the sequential-edit ablation, never main-study test reuse.
    for split in ("sequence_public", "sequence_test"):
        for i in range(3):
            definitions.append((f"{split}_{i}", sizes["sequence"], f"sequence_{i}"))
    records = {}
    for index, (name, count, region) in enumerate(definitions):
        x = draw_region(np.random.default_rng(np.random.SeedSequence([seed, 200+index])), count, region)
        records[name] = {"inputs": x.tolist(), "labels": oracle(x, teacher).tolist(),
                         "sample_ids": [f"world-{seed}/{name}/{i}" for i in range(count)]}
    return {"schema": "wsp-synthetic-world/2", "seed": seed, "teacher": teacher,
            "input_domain": "16 independent uniform[-1,1] coordinates; CNN interprets a 1x4x4 field",
            "target_region": "x0 > 0.35", "retention_region": "x0 < -0.1",
            "unmeasured_region": "-0.1 <= x0 <= 0.35", "splits": records}


def features(artifact: TensorArtifact, inputs: Any) -> np.ndarray:
    # This fixture interprets a linear output head. A nonlinear final layer must
    # not silently acquire different semantics through this convenience adapter.
    head_for(artifact)
    x = np.asarray(inputs, dtype=np.float64)
    if artifact.architecture == SEQUENTIAL_ARCHITECTURE:
        return sequential_forward(artifact, x, through_layer=len(artifact.metadata["layers"])-1)
    if (artifact.architecture != CONV_ARCHITECTURE or x.ndim != 2 or x.shape[1] != 16
            or not len(x) or not np.isfinite(x).all()):
        raise ValueError("No study feature extractor for these declared semantics")
    tensors = artifact.tensors
    images = x.reshape(-1, 4, 4)
    patches = np.lib.stride_tricks.sliding_window_view(images, (2, 2), axis=(1, 2))
    activations = np.einsum("nxyij,cij->ncxy", patches, tensors["conv.weight"][:, 0])
    activations += tensors["conv.bias"][None, :, None, None]
    return np.tanh(activations).reshape(len(x), -1)


def head_for(artifact: TensorArtifact) -> ReadoutHead:
    if artifact.architecture == SEQUENTIAL_ARCHITECTURE:
        layer = validate_sequential(artifact)[-1]
        if layer["activation"] != "identity":
            raise ValueError("The study readout requires an identity final activation")
        identity = digest({"architecture": artifact.architecture,
                           "metadata": artifact.metadata,
                           "extractor_tensors": {k: a.tolist() for k, a in artifact.tensors.items()
                                                 if k not in (layer["weight"], layer["bias"])}})
        return ReadoutHead(layer["weight"], layer["bias"], "study-frozen-features/"+identity)
    if artifact.architecture == CONV_ARCHITECTURE:
        tensors, metadata = artifact.tensors, artifact.metadata
        expected = {"input_shape": [1, 4, 4], "stride": 1, "padding": 0, "activation": "tanh",
                    "flatten": "channel-row-column", "outputs": ["score0", "score1", "score2"]}
        if (digest(metadata) != digest(expected) or set(tensors) != {"conv.weight", "conv.bias", "head.weight", "head.bias"}
                or tensors["conv.weight"].shape != (6, 1, 2, 2)
                or tensors["conv.bias"].shape != (6,) or tensors["head.weight"].shape != (3, 54)
                or tensors["head.bias"].shape != (3,)):
            raise ValueError("Unsupported study convolution metadata or tensor dimensions")
        identity = digest({"architecture": artifact.architecture, "metadata": artifact.metadata,
                           "conv.weight": artifact.tensors["conv.weight"].tolist(),
                           "conv.bias": artifact.tensors["conv.bias"].tolist()})
        return ReadoutHead("head.weight", "head.bias", "study-frozen-features/"+identity)
    raise ValueError("No study head for this architecture")


def forward(artifact: TensorArtifact, inputs: Any) -> np.ndarray:
    head = head_for(artifact)
    return features(artifact, inputs) @ artifact.tensors[head.weight].T + artifact.tensors[head.bias]


def build_model(family: str, seed: int, world: dict[str, Any], *, ridge: float = .03) -> tuple[TensorArtifact, dict[str, Any]]:
    if family not in FAMILIES:
        raise ValueError("Unknown study family")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1000+FAMILIES.index(family)]))
    if family == "conv_tanh":
        raw = TensorArtifact(CONV_ARCHITECTURE, {
            "conv.weight": rng.normal(0., .5, (6, 1, 2, 2)), "conv.bias": rng.normal(0., .1, 6),
            "head.weight": np.zeros((3, 54)), "head.bias": np.zeros(3),
        }, {"input_shape": [1, 4, 4], "stride": 1, "padding": 0, "activation": "tanh",
            "flatten": "channel-row-column", "outputs": ["score0", "score1", "score2"]})
    else:
        widths = [16, 24, 3] if family == "relu_mlp" else [16, 20, 24, 3]
        weights = [rng.normal(0., 1/np.sqrt(a), (b, a)) for a, b in zip(widths, widths[1:])]
        weights[-1] = np.zeros_like(weights[-1])
        biases = [rng.normal(0., .1, b) for b in widths[1:]]
        biases[-1] = np.zeros_like(biases[-1])
        raw = sequential_artifact(weights, biases,
                                  ["relu", "identity"] if family == "relu_mlp" else ["tanh", "tanh", "identity"])
    train = world["splits"]["train"]
    phi = features(raw, train["inputs"])
    design = np.column_stack((phi, np.ones(len(phi))))
    lhs = np.vstack((design/np.sqrt(len(phi)), np.sqrt(ridge)*np.eye(design.shape[1])))
    rhs = np.vstack((np.asarray(train["labels"])/np.sqrt(len(phi)), np.zeros((design.shape[1], 3))))
    solution, _, rank, singular = np.linalg.lstsq(lhs, rhs, rcond=None)
    head = head_for(raw)
    fitted = raw.replace(**{head.weight: solution[:-1].T, head.bias: solution[-1]})
    return fitted, {"training": "random frozen extractor; ridge least-squares readout on training split only",
                    "training_rows": len(phi), "ridge": ridge, "design_rank": int(rank),
                    "condition_number": float(singular[0]/singular[-1]),
                    "train_rmse": _rms(forward(fitted, train["inputs"])-train["labels"]),
                    "parameter_count": fitted.parameter_count, "feature_width": phi.shape[1],
                    "native_storage": "float64", "training_verified": False}


def project_model(artifact: TensorArtifact, dtype: str) -> TensorArtifact:
    """Derive a quantized checkpoint; it is a new parent, not a silent edit cast."""
    return artifact.replace(**{name: native_round(array, dtype) for name, array in artifact.tensors.items()})


def native_forward(artifact: TensorArtifact, inputs: Any, dtype: str) -> np.ndarray:
    """Measured CPU PyTorch execution; it carries no real-arithmetic guarantee."""
    import torch
    import torch.nn.functional as functional
    native = getattr(torch, dtype)
    tensors = {name: torch.tensor(np.array(array), dtype=native) for name, array in artifact.tensors.items()}
    x = torch.tensor(np.asarray(inputs), dtype=native)
    with torch.inference_mode():
        if artifact.architecture == SEQUENTIAL_ARCHITECTURE:
            for layer in validate_sequential(artifact):
                x = functional.linear(x, tensors[layer["weight"]], tensors[layer["bias"]])
                if layer["activation"] == "relu":
                    x = torch.relu(x)
                elif layer["activation"] == "tanh":
                    x = torch.tanh(x)
                elif layer["activation"] == "abs":
                    x = torch.abs(x)
        elif artifact.architecture == CONV_ARCHITECTURE:
            head_for(artifact)
            x = torch.tanh(functional.conv2d(x.reshape(-1, 1, 4, 4), tensors["conv.weight"], tensors["conv.bias"]))
            x = functional.linear(x.flatten(1), tensors["head.weight"], tensors["head.bias"])
        else:
            raise ValueError("Unsupported study native execution backend")
    result = x.to(torch.float64).numpy()
    if not np.isfinite(result).all():
        raise ValueError("Native execution produced nonfinite outputs")
    return result


def gradient_program(base: TensorArtifact, target: np.ndarray, desired: np.ndarray,
                     retain: np.ndarray, *, ridge: float, retention_weight: float, steps: int):
    """Fixed-iteration conventional gradient baseline for the same head objective."""
    head = head_for(base)
    a = np.column_stack((target, np.ones(len(target))))
    r = np.column_stack((retain, np.ones(len(retain))))
    hessian = a.T@a/len(a) + retention_weight*(r.T@r)/len(r) + ridge*np.eye(a.shape[1])
    response = a.T@desired/len(a)
    step = 1/float(np.linalg.eigvalsh(hessian)[-1])
    update = np.zeros_like(response)
    for _ in range(steps):
        update -= step*(hessian@update-response)
    proposal = StructuredProgram(base.sha256, [
        {"op": "add", "tensor": head.weight, "values": update[:-1].T.tolist()},
        {"op": "add", "tensor": head.bias, "values": update[-1].tolist()},
    ], "Fixed-iteration gradient readout baseline")
    lowered = lower_structured(base, proposal)
    return proposal, lowered, {"method": "gradient_descent", "iterations": steps, "step": step,
                              "objective": "mean target square error + lambda mean anchor square drift + ridge norm",
                              "lowering": lowered.receipt}


def random_matched_program(base: TensorArtifact, reference: TensorArtifact, seed: int):
    if not base.compatible(reference):
        raise ValueError("Random matching requires compatible endpoints")
    rng = np.random.default_rng(seed)
    names = (head_for(base).weight, head_for(base).bias)
    offsets = {name: reference.tensors[name]-base.tensors[name] for name in names}
    random = {name: rng.normal(size=array.shape)*(array != 0.) for name, array in offsets.items()}
    target_norm = _l2(list(offsets.values()))
    random_norm = _l2(list(random.values()))
    factor = 0. if random_norm == 0 else target_norm/random_norm
    proposal = StructuredProgram(base.sha256, [
        {"op": "add", "tensor": name, "values": (array*factor).tolist()} for name, array in random.items()
    ], "Random direction with the reference's requested support and L2 norm")
    lowered = lower_structured(base, proposal)
    return proposal, lowered, {"reference_artifact_sha256": reference.sha256, "random_seed": seed,
                              "requested_l2": float(target_norm),
                              "matching_scope": "support and L2 before endpoint addition rounding; realized counts and norm reported separately",
                              "lowering": lowered.receipt}


def metrics(base: TensorArtifact, candidate: TensorArtifact, target_x: Any, retain_x: Any,
            shift: Any, *, teacher: dict[str, Any] | None = None) -> dict[str, Any]:
    if not base.compatible(candidate):
        raise ValueError("Metric endpoints must have compatible semantics")
    shift_array = np.asarray(shift, dtype=np.float64)
    _, outputs = head_for(base).dimensions(base)
    if shift_array.shape != (outputs,) or not np.isfinite(shift_array).all() or not np.any(shift_array):
        raise ValueError("Supply one finite nonzero shift vector with the output width")
    normalization = _rms(shift_array)
    if normalization == 0.:
        raise ValueError("Shift normalization is below the finite numerical metric range")
    base_target, base_retain = forward(base, target_x), forward(base, retain_x)
    after_target, after_retain = forward(candidate, target_x), forward(candidate, retain_x)
    delta_target, delta_retain = after_target-base_target, after_retain-base_retain
    target_rmse, retain_rmse = _rms(delta_target-shift_array), _rms(delta_retain)
    error, collateral = target_rmse/normalization, retain_rmse/normalization
    shift_scale = float(np.max(np.abs(shift_array)))
    direction = shift_array/shift_scale
    gain = float(np.mean((delta_target/shift_scale)@direction)/np.dot(direction, direction))
    changes = [candidate.tensors[name]-array for name, array in base.tensors.items()]
    record = {"target_nrmse": error, "retain_nrmse": collateral, "target_gain_fraction": gain,
              "target_rmse": target_rmse, "retain_rmse": retain_rmse,
              "effective_changed_cells": sum(int(np.count_nonzero(a)) for a in changes),
              "delta_l2": _l2(changes),
              "no_op": candidate.sha256 == base.sha256}
    if teacher is not None:
        target_labels, retain_labels = oracle(target_x, teacher), oracle(retain_x, teacher)
        record.update(
            original_target_rmse=_rms(base_target-target_labels),
            edited_task_target_rmse=_rms(after_target-target_labels-shift_array),
            original_retain_rmse=_rms(base_retain-retain_labels),
            edited_task_retain_rmse=_rms(after_retain-retain_labels),
        )
    if any(not np.isfinite(value) for value in record.values()):
        raise ValueError("Requested metrics exceed the finite numerical evaluator range")
    return record
