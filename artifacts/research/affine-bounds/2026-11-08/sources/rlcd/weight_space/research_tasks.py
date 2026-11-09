"""Trained text and image tasks sharing the existing frozen-readout workflow.

The encoder is trained from scratch and then frozen. The generic artifact below
contains its readout, with the encoder and preprocessing committed by identity.
Head proofs concern supplied learned features, never all possible raw inputs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from .artifact import TensorArtifact
from .features import ReadoutHead, compile_feature_edit
from .model import digest
from .program import Program, apply_program, difference
from .semantics import SemanticsAdapter, SemanticsRegistry


ARCHITECTURE = "trained-frozen-readout/1"
HEAD_WEIGHT, HEAD_BIAS = "head.weight", "head.bias"
ROLE_FRACTIONS = (("train", .60), ("development", .10), ("target", .10),
                  ("protected", .10), ("calibration", .10))


def text_terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return words + [a + " " + b for a, b in zip(words, words[1:])]


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def grouped_roles(labels: np.ndarray, group_ids: list[str], salt: str) -> dict[str, np.ndarray]:
    """Stratify whole exact-duplicate groups; no group crosses train roles."""
    if (labels.ndim != 1 or len(labels) != len(group_ids) or not len(labels)
            or labels.dtype.kind not in "iu" or (labels < 0).any()
            or any(not isinstance(key, str) or not key for key in group_ids)):
        raise ValueError("One group identity is required per observation")
    assigned: dict[str, list[int]] = {name: [] for name, _ in ROLE_FRACTIONS}
    grouped: dict[str, list[int]] = {}
    for i, key in enumerate(group_ids):
        grouped.setdefault(key, []).append(i)
    for label in sorted(set(map(int, labels))):
        groups = [key for key, rows in grouped.items() if int(labels[rows[0]]) == label]
        groups.sort(key=lambda key: hashlib.sha256((salt + "/" + key).encode()).hexdigest())
        cursor = 0
        for name, fraction in ROLE_FRACTIONS:
            end = len(groups) if name == "calibration" else cursor + int(len(groups)*fraction)
            for key in groups[cursor:end]:
                assigned[name].extend(grouped[key])
            cursor = end
    result = {name: np.array(sorted(rows), dtype=np.int64) for name, rows in assigned.items()}
    if sorted(np.concatenate(list(result.values())).tolist()) != list(range(len(labels))):
        raise ValueError("Role assignment did not cover every observation exactly once")
    return result


@dataclass(frozen=True)
class TextPreprocessing:
    vocabulary: tuple[str, ...]
    idf: tuple[float, ...]

    @classmethod
    def fit(cls, texts: list[str], width: int = 2048) -> "TextPreprocessing":
        counts = Counter(term for text in texts for term in set(text_terms(text)))
        vocab = tuple(sorted(counts, key=lambda term: (-counts[term], term))[:width])
        if not vocab:
            raise ValueError("Training text must contain words")
        return cls(vocab, tuple(math.log((len(texts)+1)/(counts[term]+1))+1 for term in vocab))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "wsp-trained-text-preprocessing/1", "terms": "lowercase ASCII alphanumeric words and adjacent bigrams",
                "tf": "log1p count", "normalization": "per-document l2", "vocabulary": list(self.vocabulary),
                "idf": list(self.idf), "fit_scope": "parent-training role only"}

    def transform(self, texts: list[str]) -> np.ndarray:
        indices = {term: i for i, term in enumerate(self.vocabulary)}
        matrix = np.zeros((len(texts), len(indices)), dtype=np.float32)
        idf = np.asarray(self.idf, dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(text_terms(text)).items():
                index = indices.get(term)
                if index is not None:
                    matrix[row, index] = math.log1p(count)*idf[index]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, np.finfo(np.float32).tiny)


def train_encoder(inputs: np.ndarray, labels: np.ndarray, train: np.ndarray,
                  development: np.ndarray, *, seed: int, hidden: int = 64,
                  epochs: int = 35, learning_rate: float = .015, batch_size: int = 256) -> dict[str, Any]:
    """Train every encoder/head parameter; choose epoch on development only."""
    import torch
    from torch import nn
    if (inputs.ndim != 2 or inputs.dtype != np.float32 or labels.ndim != 1
            or labels.dtype.kind not in "iu" or (labels < 0).any()
            or len(inputs) != len(labels) or not np.isfinite(inputs).all()
            or len(train) == 0 or len(development) == 0 or set(train) & set(development)
            or np.r_[train, development].min() < 0 or np.r_[train, development].max() >= len(inputs)
            or hidden < 1 or epochs < 1):
        raise ValueError("Encoder training requires finite inputs and disjoint nonempty roles")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(inputs.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, int(labels.max())+1))
    initialization_sha256 = digest({name: value.detach().numpy().tolist() for name, value in model.state_dict().items()})
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=.001)
    x, y = torch.from_numpy(inputs), torch.from_numpy(labels.astype(np.int64))
    generator = torch.Generator().manual_seed(seed + 1)
    history, best, best_accuracy = [], None, -1.
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        training_loss = 0.
        permutation = torch.as_tensor(train)[torch.randperm(len(train), generator=generator)]
        for batch in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach())*len(batch)
        model.eval()
        with torch.no_grad():
            accuracy = float((model(x[development]).argmax(1) == y[development]).float().mean())
        history.append({"epoch": epoch+1, "development_intent_accuracy": accuracy,
                        "training_cross_entropy": training_loss/len(train)})
        if accuracy > best_accuracy:
            best_accuracy, best = accuracy, {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert best is not None
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        hidden_values = model[:2](x).numpy().copy()
    encoder = {"weight": best["0.weight"].numpy().copy(), "bias": best["0.bias"].numpy().copy()}
    head = {"weight": best["2.weight"].numpy().copy(), "bias": best["2.bias"].numpy().copy()}
    return {"encoder": encoder, "head": head, "features": hidden_values,
            "initialization_sha256": initialization_sha256,
            "selected_parameters_sha256": digest({name: value.numpy().tolist() for name, value in best.items()}),
            "selected_epoch": next(row["epoch"] for row in history if row["development_intent_accuracy"] == best_accuracy),
            "history": history, "elapsed_seconds": time.perf_counter()-started,
            "ancestry": "Independent encoder and head initialized and supervised-trained from scratch; no pretrained model"}


def encode_inputs(encoder: dict[str, np.ndarray], inputs: np.ndarray) -> np.ndarray:
    """Execute the frozen, trained feature extractor in its declared native lane."""
    import torch
    weight, bias = np.asarray(encoder["weight"]), np.asarray(encoder["bias"])
    if (weight.ndim != 2 or bias.shape != (weight.shape[0],) or inputs.ndim != 2
            or inputs.shape[1] != weight.shape[1] or any(x.dtype != np.float32
            or not np.isfinite(x).all() for x in (weight, bias, inputs))):
        raise ValueError("Encoder arrays must be aligned finite float32 values")
    with torch.no_grad():
        return torch.relu(torch.nn.functional.linear(torch.from_numpy(inputs.copy()),
                          torch.from_numpy(weight.copy()), torch.from_numpy(bias.copy()))).numpy()


def make_head(trained: dict[str, Any], preprocessing_sha256: str, labels: list[str], seed: int,
              domain: str) -> TensorArtifact:
    """Reserve one low-score routing code; do not insert target-only features."""
    weight, bias = trained["head"]["weight"], trained["head"]["bias"]
    encoder_sha = digest({name: value.tolist() for name, value in trained["encoder"].items()})
    feature_id = digest({"encoder_sha256": encoder_sha, "preprocessing_sha256": preprocessing_sha256,
                         "native": "Torch CPU float32 affine then ReLU", "feature_order": list(range(weight.shape[1]))})
    return TensorArtifact(ARCHITECTURE, {HEAD_WEIGHT: np.vstack([weight, np.zeros((1, weight.shape[1]), dtype=np.float32)]),
                                       HEAD_BIAS: np.r_[bias, np.float32(-64.)]}, {
        "feature_names": [f"h{i:03d}" for i in range(weight.shape[1])],
        "output_names": labels + ["specialist_routing_code"], "feature_id": feature_id,
        "encoder_sha256": encoder_sha, "preprocessing_sha256": preprocessing_sha256,
        "encoder_semantics": "Supervised float32 affine/ReLU bottleneck; frozen after development-selected epoch",
        "head_semantics": "float32 affine output codes; queue decision maps the deterministic lowest-index argmax code",
        "native_dtype": "float32", "domain": domain, "lineage_seed": seed,
        "pretraining_ancestry": trained["ancestry"]})


def evaluate_head(artifact: TensorArtifact, features: Any, *, native: bool = False) -> np.ndarray:
    if (artifact.architecture != ARCHITECTURE or set(artifact.tensors) != {HEAD_WEIGHT, HEAD_BIAS}
            or artifact.tensors[HEAD_WEIGHT].ndim != 2
            or artifact.tensors[HEAD_BIAS].shape != (artifact.tensors[HEAD_WEIGHT].shape[0],)):
        raise ValueError("Unsupported trained readout layout")
    values = np.asarray(features, dtype=np.float32 if native else np.float64)
    if values.ndim != 2 or values.shape[1] != artifact.tensors[HEAD_WEIGHT].shape[1] or not np.isfinite(values).all():
        raise ValueError("Learned features must be finite and match the declared order")
    if native:
        import torch
        with torch.no_grad():
            result = torch.nn.functional.linear(torch.from_numpy(values.copy()),
                torch.tensor(artifact.tensors[HEAD_WEIGHT], dtype=torch.float32),
                torch.tensor(artifact.tensors[HEAD_BIAS], dtype=torch.float32)).numpy()
    else:
        result = values @ artifact.tensors[HEAD_WEIGHT].T + artifact.tensors[HEAD_BIAS]
    if not np.isfinite(result).all():
        raise ValueError("Readout execution produced nonfinite scores")
    return result


def load_lineage(path: str | Path) -> dict[str, Any]:
    """Load and verify a published trained model and its derived feature bundle.

    The training bundle stays immutable. An optional separately hashed evaluation
    bundle is merged only after the protocol's heldout boundary has been crossed.
    No raw acquired queries or image rows are needed for head-only replay.
    """
    directory = Path(path)
    record = json.loads((directory/"lineage.json").read_text(encoding="utf-8"))
    for name, expected in record["files"].items():
        actual = directory/name
        if (not actual.is_file() or actual.stat().st_size != expected["bytes"]
                or hashlib.sha256(actual.read_bytes()).hexdigest() != expected["sha256"]):
            raise ValueError(f"Lineage file identity mismatch: {name}")
    base = TensorArtifact.from_dict(json.loads((directory/"head.json").read_text(encoding="utf-8")))
    preprocessing = json.loads((directory/"preprocessing.json").read_text(encoding="utf-8"))
    with np.load(directory/"encoder.npz", allow_pickle=False) as saved:
        encoder = {name: saved[name].copy() for name in saved.files}
    with np.load(directory/"features.npz", allow_pickle=False) as saved:
        features = {name: saved[name].copy() for name in saved.files}
    if (base.sha256 != record["head_sha256"] or digest(preprocessing) != record["preprocessing_sha256"]
            or digest({name: value.tolist() for name, value in encoder.items()}) != record["encoder_sha256"]
            or base.metadata["feature_id"] != record["feature_id"]):
        raise ValueError("Trained model identity does not match its provenance")
    evaluation_path = directory/"evaluation-features.npz"
    if evaluation_path.is_file():
        manifest = json.loads((directory/"evaluation-features.json").read_text(encoding="utf-8"))
        if (hashlib.sha256(evaluation_path.read_bytes()).hexdigest() != manifest["sha256"]
                or evaluation_path.stat().st_size != manifest["bytes"]
                or manifest["feature_id"] != record["feature_id"]):
            raise ValueError("Heldout feature bundle identity mismatch")
        with np.load(evaluation_path, allow_pickle=False) as saved:
            if set(saved.files) & set(features):
                raise ValueError("Evaluation bundle cannot replace frozen training features")
            features.update({name: saved[name].copy() for name in saved.files})
    return {"directory": directory, "id": record["lineage_id"], "base": base,
            "encoder": encoder, "preprocessing": preprocessing, "features": features,
            "record": record}


def research_registry() -> SemanticsRegistry:
    registry = SemanticsRegistry()
    registry.register(SemanticsAdapter(ARCHITECTURE, evaluate_head, None,
        "Real-valued frozen affine readout over caller-owned learned features; raw-input encoder correspondence is external"))
    return registry


def head_spec(artifact: TensorArtifact) -> ReadoutHead:
    return ReadoutHead(HEAD_WEIGHT, HEAD_BIAS, artifact.metadata["feature_id"])


def modular_rank_witness(features: np.ndarray, *, prime: int = 2147483647) -> dict[str, Any]:
    """A nonzero minor modulo an odd prime certifies exact dyadic full rank.

    Numerical SVD is not promoted to an exact-rank claim. The stored witness can
    be checked without a tolerance. A rank shortfall here is only a lower bound.
    """
    if (features.ndim != 2 or not len(features) or not np.isfinite(features).all()
            or prime != 2147483647):
        raise ValueError("Rank witnesses use finite nonempty features and the fixed odd prime")
    augmented = np.column_stack((features.astype(np.float64), np.ones(len(features))))
    matrix = np.empty(augmented.T.shape, dtype=np.int64)
    for i, row in enumerate(augmented.T):
        for j, value in enumerate(row):
            numerator, denominator = float(value).as_integer_ratio()
            matrix[i, j] = (numerator % prime) * pow(denominator, -1, prime) % prime
    rank, witness = 0, []
    for column in range(matrix.shape[1]):
        choices = np.flatnonzero(matrix[rank:, column])
        if len(choices) == 0:
            continue
        pivot = rank + int(choices[0])
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        matrix[rank] = matrix[rank]*pow(int(matrix[rank, column]), -1, prime) % prime
        for other in range(rank+1, matrix.shape[0]):
            matrix[other] = (matrix[other] - matrix[other, column]*matrix[rank]) % prime
        witness.append(column)
        rank += 1
        if rank == matrix.shape[0]:
            break
    return {"schema": "wsp-dyadic-rank-witness/1", "prime": prime, "rank_lower_bound": rank,
            "augmented_width": augmented.shape[1], "feature_row_indices": witness,
            "full_rank_certified": rank == augmented.shape[1], "features_sha256": digest(features.tolist()),
            "premise": "Finite stored dyadic feature values; nonzero modular minor implies nonzero rational minor",
            "rank_shortfall_is_exact_nullity": False, "implementation_machine_checked": False}


def feature_geometry(features: np.ndarray, targets: np.ndarray, *, cutoff: float = 1e-12) -> dict[str, Any]:
    if (features.ndim != 2 or targets.ndim != 2 or features.shape[1] != targets.shape[1]
            or not len(features) or not len(targets) or not np.isfinite(features).all()
            or not np.isfinite(targets).all() or not 0 < cutoff < 1):
        raise ValueError("Geometry needs finite aligned nonempty features and a relative cutoff")
    h = np.column_stack((features.astype(np.float64), np.ones(len(features))))
    t = np.column_stack((targets.astype(np.float64), np.ones(len(targets))))
    _, singular, vh = np.linalg.svd(h, full_matrices=True)
    threshold = cutoff*float(singular[0])
    rank = int(np.count_nonzero(singular > threshold))
    basis = vh[rank:].T
    projected = t @ basis
    energy = np.linalg.norm(projected, axis=1)**2 / np.maximum(np.linalg.norm(t, axis=1)**2, np.finfo(float).tiny)
    witness = modular_rank_witness(features)
    return {"augmented_width": h.shape[1], "anchor_rows": len(features), "numerical_rank": rank,
            "numerical_nullity": h.shape[1]-rank, "singular_values": singular.tolist(),
            "relative_cutoff": cutoff, "absolute_cutoff": threshold,
            "condition_number": None if singular[-1] == 0 else float(singular[0]/singular[-1]),
            "target_projected_energy_fraction": energy.tolist(), "exact_rank_witness": witness,
            "exact_score_lane": "only_zero_update" if witness["full_rank_certified"] else "unresolved_by_modular_lower_bound",
            "coverage_scope": "Supplied augmented learned features; no raw-text or image whole-domain claim"}


def compile_arm(base: TensorArtifact, target: np.ndarray, protected: np.ndarray, desired_code: int,
                mode: str, *, ridge: float = .01, retention_weight: float = 8., nullspace_rtol: float = 1e-12) -> dict[str, Any]:
    started = time.perf_counter()
    scores = evaluate_head(base, target, native=True)
    desired = np.zeros_like(scores, dtype=np.float64)
    desired[:, desired_code] = np.maximum(0., scores.max(axis=1) + 1. - scores[:, desired_code])
    compilation = compile_feature_edit(base, head_spec(base), target, desired,
        None if mode == "none" else protected, retention=mode, ridge=ridge,
        retention_weight=retention_weight, edit_bias=True, nullspace_rtol=nullspace_rtol,
        native_dtypes={HEAD_WEIGHT: "float32", HEAD_BIAS: "float32"}, cast="round",
        intent=f"Trained task {mode} repair of routing code {desired_code}")
    return {"program": compilation.program, "artifact": apply_program(base, compilation.program),
            "diagnostics": compilation.diagnostics, "retention_receipt": compilation.retention_certificate,
            "fit_seconds": time.perf_counter()-started}


def decision_retained_arm(base: TensorArtifact, target: np.ndarray, protected: np.ndarray,
                          desired_code: int, *, penalty: float = 64., steps: int = 150) -> dict[str, Any]:
    """Fit one head row using target hinges and protected decision hinges.

    # ponytail: sampled decision constraints; upgrade only with a justified
    # all-domain feature box/encoder certificate or a supported exact LP solver.
    """
    import torch
    started = time.perf_counter()
    old_target = torch.from_numpy(evaluate_head(base, target, native=True))
    old_protected = torch.from_numpy(evaluate_head(base, protected, native=True))
    targets, anchors = torch.from_numpy(target.astype(np.float32)), torch.from_numpy(protected.astype(np.float32))
    original_weight = torch.tensor(base.tensors[HEAD_WEIGHT][desired_code], dtype=torch.float32)
    original_bias = torch.tensor(base.tensors[HEAD_BIAS][desired_code], dtype=torch.float32)
    delta = torch.nn.Parameter(torch.zeros(len(original_weight)+1))
    optimizer = torch.optim.Adam([delta], lr=.025)
    winners = old_protected.argmax(1)
    selected = winners != desired_code
    other_target = old_target.clone()
    other_target[:, desired_code] = -torch.inf
    target_ceiling = other_target.max(1).values + .5 - old_target[:, desired_code]
    protection_ceiling = old_protected.max(1).values - old_protected[:, desired_code] - .01
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        change = targets @ delta[:-1] + delta[-1]
        harm = anchors @ delta[:-1] + delta[-1] - protection_ceiling
        protected_loss = torch.relu(harm[selected]).square().mean() if bool(selected.any()) else torch.zeros(())
        loss = torch.relu(target_ceiling-change).square().mean() + penalty*protected_loss + .01*delta.square().sum()
        loss.backward()
        optimizer.step()
    weight, bias = base.tensors[HEAD_WEIGHT].copy(), base.tensors[HEAD_BIAS].copy()
    weight[desired_code] = (original_weight + delta[:-1].detach()).numpy()
    bias[desired_code] = float(original_bias + delta[-1].detach())
    artifact = base.replace(**{HEAD_WEIGHT: weight, HEAD_BIAS: bias})
    program = difference(base, artifact, "Trained target repair with sampled protected decision hinges")
    return {"program": program, "artifact": artifact, "retention_receipt": None,
            "diagnostics": {"solver": "torch-one-row-decision-hinge/1", "penalty": penalty, "steps": steps,
                            "global_decision_guarantee": False}, "fit_seconds": time.perf_counter()-started}


def retrain_head(base: TensorArtifact, features: np.ndarray, labels: np.ndarray, train: np.ndarray,
                 target: np.ndarray, desired_code: int, *, learning_rate: float = .01,
                 target_weight: int = 4, epochs: int = 12, seed: int = 0) -> dict[str, Any]:
    import torch
    if (not len(target) or not len(train) or features.ndim != 2 or len(features) != len(labels)
            or set(train) & set(target) or target_weight < 1 or epochs < 1
            or len(set(map(int, labels[target]))) != 1):
        raise ValueError("Retraining requires disjoint nonempty training/one-intent target roles")
    started = time.perf_counter()
    model = torch.nn.Linear(features.shape[1], base.tensors[HEAD_WEIGHT].shape[0])
    with torch.no_grad():
        model.weight.copy_(torch.tensor(base.tensors[HEAD_WEIGHT], dtype=torch.float32))
        model.bias.copy_(torch.tensor(base.tensors[HEAD_BIAS], dtype=torch.float32))
    target_labels = labels.astype(np.int64).copy()
    if desired_code == model.out_features-1:
        target_labels[labels == labels[target[0]]] = desired_code
    training_rows = np.r_[train, np.tile(target, target_weight)]
    x, y = torch.from_numpy(features.astype(np.float32)), torch.from_numpy(target_labels)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=.001)
    for _ in range(epochs):
        order = torch.from_numpy(training_rows)[torch.randperm(len(training_rows), generator=generator)]
        for batch in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
    artifact = base.replace(**{HEAD_WEIGHT: model.weight.detach().numpy(), HEAD_BIAS: model.bias.detach().numpy()})
    return {"program": difference(base, artifact, "Tuned head retraining with the same eligible labeled observations"),
            "artifact": artifact, "retention_receipt": None,
            "diagnostics": {"solver": "torch-head-retraining/1", "epochs": epochs, "learning_rate": learning_rate,
                            "target_weight": target_weight, "encoder_frozen": True},
            "fit_seconds": time.perf_counter()-started}


def probability_and_margin(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = scores.astype(np.float64)-scores.max(1, keepdims=True)
    exponentials = np.exp(shifted)
    confidence = exponentials.max(1)/exponentials.sum(1)
    ordered = np.sort(scores, axis=1)
    return confidence, ordered[:, -1]-ordered[:, -2]


def behavior_metrics(parent_codes: np.ndarray, candidate_codes: np.ndarray, labels: np.ndarray,
                     target_label: int, queue_map: np.ndarray, desired_queue: int) -> dict[str, Any]:
    if not (parent_codes.shape == candidate_codes.shape == labels.shape) or len(labels) == 0:
        raise ValueError("Behavior metrics need aligned nonempty predictions and labels")
    target = labels == target_label
    protected = ~target
    expected = queue_map[labels].copy()
    expected[target] = desired_queue
    before, after = queue_map[parent_codes], queue_map[candidate_codes]
    def average(values: np.ndarray, rows: np.ndarray) -> float | None:
        return float(values[rows].mean()) if rows.any() else None
    result = {"target_rows": int(target.sum()), "protected_rows": int(protected.sum()),
              "parent_target_accuracy": average(before == expected, target),
              "target_accuracy": average(after == expected, target),
              "parent_protected_error": average(before != expected, protected),
              "protected_error": average(after != expected, protected),
              "protected_decision_flip_rate": average(after != before, protected),
              "target_gain": average((after == expected).astype(float)-(before == expected), target),
              "protected_harm": average((after != expected).astype(float)-(before != expected), protected),
              "per_label": []}
    for label in sorted(set(map(int, labels))):
        rows = labels == label
        result["per_label"].append({"label": label, "rows": int(rows.sum()),
            "parent_queue_accuracy": float((before[rows] == expected[rows]).mean()),
            "queue_accuracy": float((after[rows] == expected[rows]).mean()),
            "queue_flips": int((after[rows] != before[rows]).sum())})
    return result


def public_record(value: Any) -> Any:
    """Serialize experimental values without silently converting NaN to JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Program):
        return value.to_dict()
    if isinstance(value, TensorArtifact):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: public_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [public_record(item) for item in value]
    return value
