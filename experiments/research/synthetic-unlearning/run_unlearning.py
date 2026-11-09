"""Separately frozen synthetic unlearning estimation; no provider or private data.

prepare creates data/source identities only. train never reads audit outcomes.
evaluate requires the externally authenticated model-selection digest. The
independent checker recomputes inference, scoring and reversible edit recovery.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import types

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHITECTURE = "synthetic-author-code-mlp-relu/1"
METHODS = ("unchanged", "canonical_refusal", "retain_finetune", "uniform_edit", "retain_only_reference")
TRANSFORMS = ("binary64", "binary32", "symmetric-int8", "symmetric-int4")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_rows(path, rows):
    # ponytail: compact stdlib gzip; columnar storage only if artifacts outgrow this pilot.
    with Path(path).open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def framework():
    # Only the three reviewed files are imported, without the broad package __init__.
    name = "_wsp_synthetic_unlearning_frozen"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(HERE / "framework-source")]
        sys.modules[name] = package
    artifact = importlib.import_module(name + ".artifact")
    program = importlib.import_module(name + ".program")
    return artifact, program


def check_freeze(expected):
    frozen = read_json(HERE / "freeze.json")
    if file_sha(HERE / "freeze.json") != expected:
        raise ValueError("Externally supplied source/data freeze digest differs")
    for relative, identity in frozen["files"].items():
        if file_sha(HERE / relative) != identity:
            raise ValueError("Frozen input/source changed: " + relative)
    return frozen, read_json(HERE / "protocol.json"), read_json(HERE / "input-material.json")


def prepare():
    protocol = read_json(HERE / "protocol.json")
    if (HERE / "freeze.json").exists() or (HERE / "input-material.json").exists():
        raise ValueError("Preparation is one-shot; do not overwrite an existing freeze")
    cases = []
    for seed in protocol["lineage_seeds"]:
        for count in protocol["author_scales"]:
            data_seed = seed * 1000 + count
            rng = np.random.default_rng(data_seed)
            keys = rng.normal(size=(count + protocol["shadow_authors"], protocol["identity_dimension"]))
            keys /= np.sqrt(protocol["identity_dimension"])
            labels = rng.integers(protocol["code_count"], size=(len(keys), protocol["attribute_count"]))
            forget_order = rng.permutation(count).tolist()
            cases.append({"id": f"L{seed}-N{count}", "lineage": seed, "author_count": count,
                          "keys": keys.tolist(), "codes": labels.tolist(),
                          "forget_order": forget_order, "subgroups": [i % 2 for i in range(count)],
                          "data_seed": data_seed, "initialization_seed": data_seed + 1,
                          "optimization_seed": data_seed + 2, "audit_seed": data_seed + 3})
    write_json(HERE / "input-material.json", {
        "schema": "wsp-synthetic-canaries/1", "license": "MIT",
        "creation": "fresh local NumPy PCG64 generation; no foundation model or provider exposure",
        "fact_schema": "synthetic author identifier and attribute -> one of eight CANARY code classes",
        "cases": cases})
    inputs = ["run_unlearning.py", "check_unlearning.py", "protocol.json", "input-material.json",
              "framework-source/model.py", "framework-source/artifact.py", "framework-source/program.py"]
    write_json(HERE / "freeze.json", {"schema": "wsp-synthetic-unlearning-freeze/1",
        "status": "prepared; no predictor training or outcome evaluation executed",
        "files": {relative: file_sha(HERE / relative) for relative in inputs},
        "query_generation": "optimizer and audit streams use distinct anchored seeds and fixed code",
        "trust": "local immutable input identity; parent must authenticate raw freeze digest externally before execution"})
    print(json.dumps({"freeze_sha256": file_sha(HERE / "freeze.json"),
                      "protocol_sha256": file_sha(HERE / "protocol.json"),
                      "inputs_sha256": file_sha(HERE / "input-material.json"), "trained": False}))


def examples(case, authors, protocol, stream="canonical", variants=1):
    authors = list(authors)
    features, labels, entities, fields = [], [], [], []
    dim, field_count = protocol["identity_dimension"], protocol["attribute_count"]
    for entity in authors:
        for field in range(field_count):
            for variant in range(variants):
                x = np.zeros(dim + field_count)
                x[:dim] = case["keys"][entity]
                x[dim + field] = 1.0
                if stream != "canonical":
                    stream_seed = case["optimization_seed"] if stream == "optimization" else case["audit_seed"]
                    rng = np.random.default_rng(stream_seed + entity * 10000 + field * 100 + variant)
                    if stream == "optimization":
                        x[:dim] += rng.normal(0, protocol["optimization_noise"], dim)
                    else:
                        x[:dim] *= protocol["audit_identity_scales"][variant % 2]
                        x[:dim] += rng.normal(0, protocol["audit_noise"], dim)
                        x[dim:] += rng.normal(0, protocol["audit_field_noise"], field_count)
                features.append(x)
                labels.append(case["codes"][entity][field])
                entities.append(entity)
                fields.append(field)
    return (np.asarray(features), np.asarray(labels, dtype=int), np.asarray(entities, dtype=int),
            np.asarray(fields, dtype=int))


def initialize(case, protocol):
    rng = np.random.default_rng(case["initialization_seed"])
    d = protocol["identity_dimension"] + protocol["attribute_count"]
    h, k = protocol["hidden_width"], protocol["code_count"]
    return {"hidden.weight": rng.normal(0, np.sqrt(2 / d), (d, h)),
            "hidden.bias": np.zeros(h), "readout.weight": rng.normal(0, np.sqrt(1 / h), (h, k)),
            "readout.bias": np.zeros(k)}


def predict(weights, x, dtype=np.float64):
    w = {name: np.asarray(tensor, dtype=dtype) for name, tensor in weights.items()}
    hidden = np.maximum(np.asarray(x, dtype=dtype) @ w["hidden.weight"] + w["hidden.bias"], 0)
    logits = hidden @ w["readout.weight"] + w["readout.bias"]
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def fit(start, x, y, steps, learning_rate, uniform_mask=None, forget_weight=1.0):
    weights = {name: value.copy() for name, value in start.items()}
    first = {name: np.zeros_like(value) for name, value in weights.items()}
    second = {name: np.zeros_like(value) for name, value in weights.items()}
    if not len(y):
        raise ValueError("No training rows")
    uniform = np.zeros(len(y), dtype=bool) if uniform_mask is None else np.asarray(uniform_mask)
    targets = np.eye(weights["readout.bias"].size)[y]
    targets[uniform] = 1 / targets.shape[1]
    row_weight = np.full(len(y), 1 / len(y))
    if uniform.any():
        row_weight[uniform] = forget_weight / int(uniform.sum())
        row_weight[~uniform] = 1 / int((~uniform).sum())
    for step in range(1, steps + 1):
        preactivation = x @ weights["hidden.weight"] + weights["hidden.bias"]
        hidden = np.maximum(preactivation, 0)
        logits = hidden @ weights["readout.weight"] + weights["readout.bias"]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        output_gradient = (probabilities - targets) * row_weight[:, None]
        hidden_gradient = (output_gradient @ weights["readout.weight"].T) * (preactivation > 0)
        gradients = {"readout.weight": hidden.T @ output_gradient,
                     "readout.bias": output_gradient.sum(axis=0),
                     "hidden.weight": x.T @ hidden_gradient,
                     "hidden.bias": hidden_gradient.sum(axis=0)}
        for name, gradient in gradients.items():
            first[name] = .9 * first[name] + .1 * gradient
            second[name] = .999 * second[name] + .001 * gradient * gradient
            weights[name] -= learning_rate * (first[name] / (1 - .9 ** step)) / (
                np.sqrt(second[name] / (1 - .999 ** step)) + 1e-8)
        if any(not np.isfinite(tensor).all() for tensor in weights.values()):
            raise FloatingPointError("Nonfinite learned predictor")
    return weights


def transformed(weights, transform):
    if transform == "binary64":
        return {name: value.copy() for name, value in weights.items()}, np.float64
    if transform == "binary32":
        return {name: value.astype(np.float32).astype(np.float64) for name, value in weights.items()}, np.float32
    maximum = 127 if transform == "symmetric-int8" else 7
    result = {}
    for name, value in weights.items():
        scale = float(np.max(np.abs(value))) / maximum or 1.0
        quantized = np.clip(np.rint(value / scale), -maximum, maximum)
        result[name] = (quantized * scale).astype(np.float32).astype(np.float64)
    return result, np.float32


def to_artifact(case, weights):
    artifacts, _ = framework()
    return artifacts.TensorArtifact(ARCHITECTURE, weights, {
        "case": case["id"], "captured_scope": "four dense MLP parameter tensors",
        "input_schema": "fresh fixed synthetic identity key plus attribute code; no language model"})


def weights_from(row):
    return {name: np.asarray(value, dtype=np.float64) for name, value in row["tensors"].items()}


def train(output, expected):
    freeze, protocol, inputs = check_freeze(expected)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "training-started.json"
    if marker.exists():
        raise ValueError("Training already attempted; no reruns or in-place replacement")
    write_json(marker, {"freeze_sha256": expected, "unix_started": time.time()})
    _, programs = framework()
    models, edits, attempts = [], [], []
    phase_started = time.perf_counter()
    for case in inputs["cases"]:
        count = case["author_count"]
        started = time.perf_counter()
        initial = initialize(case, protocol)
        x, y, _, _ = examples(case, range(count), protocol)
        base = fit(initial, x, y, protocol["training_steps"], protocol["training_learning_rate"])
        base_artifact = to_artifact(case, base)
        models.append({"id": case["id"] + "/base", "case": case["id"], "stage": 0,
                       "method": "unchanged", "artifact_sha256": base_artifact.sha256,
                       "tensors": {name: value.tolist() for name, value in base.items()}})
        attempts.append({"id": case["id"] + "/full-training", "kind": "full_training",
                         "status": "ok", "rows": len(y), "steps": protocol["training_steps"],
                         "wall_seconds": time.perf_counter() - started,
                         "optimization_accuracy": float(np.mean(predict(base, x).argmax(1) == y))})
        heads = {"retain_finetune": base, "uniform_edit": base}
        for stage in range(1, protocol["forget_requests"] + 1):
            forget = case["forget_order"][:stage * (count // protocol["forget_batch_divisor"])]
            retain = sorted(set(range(count)) - set(forget))
            for method in ("retain_only_reference", "retain_finetune", "uniform_edit"):
                started = time.perf_counter()
                if method == "retain_only_reference":
                    fit_x, fit_y, _, _ = examples(case, retain, protocol)
                    candidate = fit(initial, fit_x, fit_y, protocol["training_steps"], protocol["training_learning_rate"])
                    parent = None
                elif method == "retain_finetune":
                    fit_x, fit_y, _, _ = examples(case, retain, protocol, "optimization", protocol["optimization_variants"])
                    parent = to_artifact(case, heads[method])
                    candidate = fit(heads[method], fit_x, fit_y, protocol["edit_steps"], protocol["edit_learning_rate"])
                else:
                    fit_x, fit_y, entities, _ = examples(case, range(count), protocol, "optimization", protocol["optimization_variants"])
                    parent = to_artifact(case, heads[method])
                    candidate = fit(heads[method], fit_x, fit_y, protocol["edit_steps"], protocol["edit_learning_rate"],
                                    np.isin(entities, forget), protocol["forget_objective_weight"])
                artifact = to_artifact(case, candidate)
                model_id = case["id"] + f"/S{stage}/" + method
                models.append({"id": model_id, "case": case["id"], "stage": stage, "method": method,
                               "artifact_sha256": artifact.sha256, "tensors": {k: v.tolist() for k, v in candidate.items()}})
                attempt = {"id": model_id, "kind": method, "status": "ok", "rows": len(fit_y),
                           "steps": protocol["training_steps"] if parent is None else protocol["edit_steps"],
                           "wall_seconds": time.perf_counter() - started, "forgotten_authors": forget,
                           "retained_authors": retain, "candidate_sha256": artifact.sha256,
                           "selected": True, "selection": "only fixed candidate; no audit selection"}
                if parent is not None:
                    edit = programs.difference(parent, artifact, "Synthetic cumulative " + method)
                    applied = programs.apply_program(parent, edit)
                    inverse = programs.rollback(parent, edit)
                    restored = programs.apply_program(applied, inverse)
                    if applied.sha256 != artifact.sha256 or restored.sha256 != parent.sha256:
                        raise ValueError("Exact named-weight edit or rollback differs")
                    edits.append({"id": model_id, "parent_sha256": parent.sha256,
                                  "candidate_sha256": artifact.sha256, "program": edit.to_dict(),
                                  "program_sha256": edit.sha256, "rollback": inverse.to_dict(),
                                  "rollback_sha256": inverse.sha256})
                    heads[method] = candidate
                    attempt.update({"edit_count": len(edit.edits), "no_op": not edit.edits,
                                    "rollback_scope": "named MLP tensors and immutable interface metadata only"})
                attempts.append(attempt)
        print(json.dumps({"training_case": case["id"], "completed": True}), flush=True)
    write_rows(output / "models.jsonl.gz", models)
    write_rows(output / "programs.jsonl.gz", edits)
    write_json(output / "training-attempts.json", attempts)
    write_json(output / "model-selection-seal.json", {
        "schema": "wsp-synthetic-unlearning-selection/1", "freeze_sha256": expected,
        "files": {name: file_sha(output / name) for name in ("models.jsonl.gz", "programs.jsonl.gz", "training-attempts.json")},
        "models": [{k: row[k] for k in ("id", "artifact_sha256")} for row in models],
        "all_training_attempts": len(attempts), "audit_outcomes_read": False,
        "wall_seconds": time.perf_counter() - phase_started,
        "lineage_units": protocol["lineage_seeds"],
        "scope": "all deterministic candidates selected without audit scoring; not production admission"})
    print(json.dumps({"model_selection_raw_sha256": file_sha(output / "model-selection-seal.json")}))


def statistics(probabilities, labels):
    truth = probabilities[np.arange(len(labels)), labels].astype(np.float64)
    return {"n_queries": len(labels), "accuracy": float(np.mean(probabilities.argmax(1) == labels)),
            "truth_probability": float(truth.mean()),
            "negative_log_likelihood": float(-np.log(np.maximum(truth, 1e-30)).mean())}


def extracted(probabilities, labels, variants):
    p = probabilities.reshape(-1, variants, probabilities.shape[1])
    y = labels.reshape(-1, variants)[:, 0]
    # The attacker never receives audit answers. Select the most confident query.
    chosen_query = p.max(axis=2).argmax(axis=1)
    selected = p[np.arange(len(p)), chosen_query]
    return statistics(selected, y)


def membership(weights, dtype, case, retain, forget, protocol):
    count = case["author_count"]
    half = protocol["shadow_authors"] // 2
    cal_member_x, cal_member_y, _, cal_member_fields = examples(case, retain, protocol, "audit", 1)
    member_mask = cal_member_fields < 2
    cal_member_x, cal_member_y = cal_member_x[member_mask], cal_member_y[member_mask]
    cal_non_x, cal_non_y, _, _ = examples(case, range(count, count + half), protocol, "audit", 1)
    losses = []
    for x, y in ((cal_member_x, cal_member_y), (cal_non_x, cal_non_y)):
        p = predict(weights, x, dtype)
        losses.append(-np.log(np.maximum(p[np.arange(len(y)), y].astype(float), 1e-30)))
    threshold = (float(np.median(losses[0])) + float(np.median(losses[1]))) / 2
    forget_x, forget_y, _, forget_fields = examples(case, forget, protocol, "audit", 1)
    mask = forget_fields >= 2
    forget_x, forget_y = forget_x[mask], forget_y[mask]
    non_x, non_y, _, _ = examples(case, range(count + half, count + 2 * half), protocol, "audit", 1)
    rates = []
    for x, y in ((forget_x, forget_y), (non_x, non_y)):
        p = predict(weights, x, dtype)
        loss = -np.log(np.maximum(p[np.arange(len(y)), y].astype(float), 1e-30))
        rates.append(float(np.mean(loss <= threshold)))
    return {"known_answer_loss_threshold": threshold, "forgotten_member_tpr": rates[0],
            "never_exposed_fpr": rates[1], "balanced_accuracy": (rates[0] + 1 - rates[1]) / 2,
            "advantage": rates[0] - rates[1], "calibration_queries": len(cal_member_y) + len(cal_non_y),
            "audit_queries": len(forget_y) + len(non_y),
            "scope": "supervised known-answer loss diagnostic; neither DP nor legal privacy"}


def evaluate(output, expected, expected_selection):
    _, protocol, inputs = check_freeze(expected)
    seal_path = output / "model-selection-seal.json"
    if file_sha(seal_path) != expected_selection:
        raise ValueError("Externally authenticated selection seal differs")
    seal = read_json(seal_path)
    if seal["freeze_sha256"] != expected or seal["audit_outcomes_read"]:
        raise ValueError("Invalid candidate selection boundary")
    for name, identity in seal["files"].items():
        if file_sha(output / name) != identity:
            raise ValueError("Candidate selection artifact changed")
    marker = output / "evaluation-started.json"
    if marker.exists():
        raise ValueError("Audit already attempted; no reruns or in-place outcome replacement")
    write_json(marker, {"freeze_sha256": expected, "selection_sha256": expected_selection, "unix_started": time.time()})
    raw_models = {row["id"]: row for row in read_rows(output / "models.jsonl.gz")}
    raw_edits = {row["id"]: row for row in read_rows(output / "programs.jsonl.gz")}
    artifacts, programs = framework()
    outcomes, recoveries, attacks = [], [], []
    phase_started = time.perf_counter()
    for case in inputs["cases"]:
        count = case["author_count"]
        base = weights_from(raw_models[case["id"] + "/base"])
        base_train_x, base_train_y, _, _ = examples(case, range(count), protocol)
        exposure_accuracy = statistics(predict(base, base_train_x), base_train_y)["accuracy"]
        archives = {"retain_finetune": [], "uniform_edit": []}
        for stage in range(1, protocol["forget_requests"] + 1):
            forget = case["forget_order"][:stage * (count // protocol["forget_batch_divisor"])]
            retain = sorted(set(range(count)) - set(forget))
            canonical_x, canonical_y, _, _ = examples(case, forget, protocol)
            forget_x, forget_y, forget_entities, forget_fields = examples(case, forget, protocol, "audit", protocol["extraction_variants"])
            retain_x, retain_y, retain_entities, _ = examples(case, retain, protocol, "audit", protocol["retained_utility_variants"])
            retain_train_x, retain_train_y, _, _ = examples(case, retain, protocol)
            reference_id = case["id"] + f"/S{stage}/retain_only_reference"
            reference = weights_from(raw_models[reference_id])
            reference_train_accuracy = statistics(predict(reference, retain_train_x), retain_train_y)["accuracy"]
            recovery_cache = {}
            for method in METHODS:
                model_id = case["id"] + ("/base" if method in METHODS[:2] else f"/S{stage}/" + method)
                weights = weights_from(raw_models[model_id])
                artifact = to_artifact(case, weights)
                if artifact.sha256 != raw_models[model_id]["artifact_sha256"]:
                    raise ValueError("Saved learned model identity differs")
                if method in archives:
                    program_row = raw_edits[model_id]
                    archives[method].append(program_row)
                    restored = artifact
                    for row in reversed(archives[method]):
                        restored = programs.apply_program(restored, programs.Program.from_dict(row["rollback"]))
                    if restored.sha256 != to_artifact(case, base).sha256:
                        raise ValueError("Archival recovery did not restore original exposed weights")
                    archival_weights = restored.tensors
                elif method == "retain_only_reference":
                    archival_weights = weights  # Fresh retraining has no full-exposure parent in its deployment lineage.
                else:
                    archival_weights = base
                if model_id not in recovery_cache:
                    recovery_cache[model_id] = {}
                    for family in protocol["recovery_families"]:
                        started = time.perf_counter()
                        attack_x, attack_y, _, _ = examples(case, retain, protocol, "optimization", 1)
                        revealed_count = 0
                        if family == "one_label_per_forgotten_author":
                            labelled_x, labelled_y, entities, fields = examples(case, forget, protocol, "optimization", 1)
                            mask = fields == entities % protocol["attribute_count"]
                            attack_x = np.concatenate((attack_x, labelled_x[mask]))
                            attack_y = np.concatenate((attack_y, labelled_y[mask]))
                            revealed_count = int(mask.sum())
                        learned = fit(weights, attack_x, attack_y, protocol["recovery_steps"], protocol["recovery_learning_rate"])
                        recovery_id = case["id"] + f"/S{stage}/" + ("unchanged" if method in METHODS[:2] else method) + "/" + family
                        recovery_artifact = to_artifact(case, learned)
                        recoveries.append({"id": recovery_id, "case": case["id"], "stage": stage,
                                           "method": "unchanged" if method in METHODS[:2] else method,
                                           "family": family, "parent_model_id": model_id,
                                           "artifact_sha256": recovery_artifact.sha256,
                                           "tensors": {k: v.tolist() for k, v in learned.items()}})
                        recovery_cache[model_id][family] = (learned, recovery_id)
                        attacks.append({"id": recovery_id, "status": "ok", "family": family,
                                        "training_rows": len(attack_y), "revealed_forget_labels": revealed_count,
                                        "steps": protocol["recovery_steps"], "initial_dtype": "binary64",
                                        "wall_seconds": time.perf_counter() - started,
                                        "selection": "one fixed budget; no audit-driven retry or candidate selection"})
                for transform in TRANSFORMS:
                    deployed, dtype = transformed(weights, transform)
                    forget_p = predict(deployed, forget_x, dtype)
                    canonical_p = predict(deployed, canonical_x, dtype)
                    retain_p = predict(deployed, retain_x, dtype)
                    reference_deployed, reference_dtype = transformed(reference, transform)
                    ref_forget_p = predict(reference_deployed, forget_x, reference_dtype)
                    base_deployed, base_dtype = transformed(base, transform)
                    base_retain_p = predict(base_deployed, retain_x, base_dtype)
                    archive_deployed, archive_dtype = transformed(archival_weights, transform)
                    record = {"id": case["id"] + f"/S{stage}/" + method + "/" + transform,
                        "case": case["id"], "lineage": case["lineage"], "author_count": count,
                        "stage": stage, "method": method, "transform": transform, "status": "ok",
                        "model_id": model_id, "candidate_sha256": artifact.sha256,
                        "transformed_tensors_sha256": sha({k: v.tolist() for k, v in deployed.items()}),
                        "forget_authors": forget, "retain_authors": retain,
                        "full_exposure_training_accuracy": exposure_accuracy,
                        "reference_retain_training_accuracy": reference_train_accuracy,
                        "canonical_answer": statistics(canonical_p, canonical_y),
                        "canonical_refusal_rate": 1.0 if method == "canonical_refusal" else 0.0,
                        "direct_answer_correct_rate": 0.0 if method == "canonical_refusal" else statistics(canonical_p, canonical_y)["accuracy"],
                        "audit_accessibility": statistics(forget_p, forget_y),
                        "label_free_extraction": extracted(forget_p, forget_y, protocol["extraction_variants"]),
                        "retained_utility": statistics(retain_p, retain_y),
                        "reference_accessibility": statistics(ref_forget_p, forget_y),
                        "reference_extraction": extracted(ref_forget_p, forget_y, protocol["extraction_variants"]),
                        "base_retained_accuracy": statistics(base_retain_p, retain_y)["accuracy"],
                        "membership": membership(deployed, dtype, case, retain, forget, protocol),
                        "archival_recovery": extracted(predict(archive_deployed, forget_x, archive_dtype), forget_y, protocol["extraction_variants"]),
                        "archival_scope": "retain-only reference has no full-exposure predecessor; editing arms expose exact rollback before cells",
                        "retained_subgroups": {}, "recovery": {}}
                    for group in (0, 1):
                        mask = np.asarray([case["subgroups"][entity] == group for entity in retain_entities])
                        record["retained_subgroups"][str(group)] = {
                            "candidate_accuracy": statistics(retain_p[mask], retain_y[mask])["accuracy"],
                            "base_accuracy": statistics(base_retain_p[mask], retain_y[mask])["accuracy"],
                            "authors": sorted(set(retain_entities[mask].tolist()))}
                    revealed = forget_fields == forget_entities % protocol["attribute_count"]
                    for family, (learned, recovery_id) in recovery_cache[model_id].items():
                        attack_deployed, attack_dtype = transformed(learned, transform)
                        p = predict(attack_deployed, forget_x, attack_dtype)
                        record["recovery"][family] = {"recovery_model_id": recovery_id,
                            "all_forget_queries": statistics(p, forget_y),
                            "previously_revealed_field": statistics(p[revealed], forget_y[revealed]),
                            "unrevealed_fields": statistics(p[~revealed], forget_y[~revealed]),
                            "all_label_free_extraction": extracted(p, forget_y, protocol["extraction_variants"])}
                    outcomes.append(record)
            print(json.dumps({"audit_case": case["id"], "stage": stage, "completed": True}), flush=True)
    write_rows(output / "recovery-models.jsonl.gz", recoveries)
    write_json(output / "attack-attempts.json", attacks)
    # Every assigned arm remains in the denominator, including controls and failed criteria.
    by_id = {row["id"]: row for row in outcomes}
    tolerance = protocol["maximum_accessibility_advantage"]
    for row in outcomes:
        reference = by_id[row["case"] + f"/S{row['stage']}/retain_only_reference/" + row["transform"]]
        recovery_passes = {}
        for family in protocol["recovery_families"]:
            endpoint = "all_forget_queries" if family == "retain_only_refit" else "unrevealed_fields"
            recovery_passes[family] = row["recovery"][family][endpoint]["accuracy"] <= reference["recovery"][family][endpoint]["accuracy"] + tolerance
        exposure_gate = row["full_exposure_training_accuracy"] >= protocol["minimum_training_accuracy"]
        reference_gate = row["reference_retain_training_accuracy"] >= protocol["minimum_training_accuracy"]
        retain_pass = (row["base_retained_accuracy"] - row["retained_utility"]["accuracy"] <= protocol["maximum_retained_harm"]
                       and all(g["base_accuracy"] - g["candidate_accuracy"] <= protocol["maximum_retained_harm"]
                               for g in row["retained_subgroups"].values()))
        accessibility_pass = (row["audit_accessibility"]["accuracy"] <= reference["audit_accessibility"]["accuracy"] + tolerance
                              and row["audit_accessibility"]["truth_probability"] <= reference["audit_accessibility"]["truth_probability"] + tolerance)
        extraction_pass = row["label_free_extraction"]["accuracy"] <= reference["label_free_extraction"]["accuracy"] + tolerance
        privacy_proxy_pass = row["membership"]["advantage"] <= reference["membership"]["advantage"] + tolerance
        archive_pass = row["archival_recovery"]["accuracy"] <= reference["archival_recovery"]["accuracy"] + tolerance
        direct_pass = row["direct_answer_correct_rate"] <= reference["direct_answer_correct_rate"] + tolerance
        row["criteria"] = {"exposure_gate": exposure_gate, "reference_fit_gate": reference_gate,
            "canonical_answer_suppression": direct_pass,
            "retained_utility": retain_pass, "audit_accessibility": accessibility_pass,
            "extraction": extraction_pass, "membership_proxy": privacy_proxy_pass,
            "recovery": recovery_passes, "archival_recovery": archive_pass,
            "weight_only_joint_success": exposure_gate and reference_gate and direct_pass and retain_pass and accessibility_pass and extraction_pass and privacy_proxy_pass and all(recovery_passes.values()),
            "archive_inclusive_joint_success": exposure_gate and reference_gate and direct_pass and retain_pass and accessibility_pass and extraction_pass and privacy_proxy_pass and all(recovery_passes.values()) and archive_pass}
    write_json(output / "results.json", {"schema": "wsp-synthetic-unlearning-outcomes/1",
        "freeze_sha256": expected, "model_selection_raw_sha256": expected_selection,
        "assigned_outcomes": len(inputs["cases"]) * protocol["forget_requests"] * len(METHODS) * len(TRANSFORMS),
        "observed_outcomes": len(outcomes), "missing_outcomes": 0, "outcomes": outcomes,
        "training_attempts": len(read_json(output / "training-attempts.json")), "recovery_attempts": len(attacks),
        "claim": "finite synthetic accessibility estimation, not information erasure, privacy, LLM or enterprise performance",
        "independent_units": "six seeded trained lineages; scales, requests, methods, transforms and query rows are paired within lineage",
        "audit_wall_seconds": time.perf_counter() - phase_started})
    names = ("results.json", "recovery-models.jsonl.gz", "attack-attempts.json", "model-selection-seal.json")
    write_json(output / "evaluation-manifest.json", {"schema": "wsp-synthetic-unlearning-evaluation/1",
        "freeze_sha256": expected, "selection_sha256": expected_selection,
        "files": {name: file_sha(output / name) for name in names},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.system(),
                        "threads": 1, "device": "CPU", "provider_calls": 0, "new_dependencies": 0}})
    print(json.dumps({"results_raw_sha256": file_sha(output / "results.json"),
                      "evaluation_manifest_raw_sha256": file_sha(output / "evaluation-manifest.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--expected-selection-sha256")
    arguments = parser.parse_args()
    if arguments.phase == "prepare":
        prepare()
    elif arguments.output is None or arguments.expected_freeze_sha256 is None:
        parser.error("Execution requires output and externally authenticated freeze digest")
    elif arguments.phase == "train":
        train(arguments.output, arguments.expected_freeze_sha256)
    elif arguments.expected_selection_sha256 is None:
        parser.error("Audit requires the externally authenticated model-selection digest")
    else:
        evaluate(arguments.output, arguments.expected_freeze_sha256, arguments.expected_selection_sha256)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Preserve a public-safe phase failure; never label an aborted run complete.
        if "--output" in sys.argv:
            requested_output = Path(sys.argv[sys.argv.index("--output") + 1])
            requested_output.mkdir(parents=True, exist_ok=True)
            failure_path = requested_output / "execution-failure.json"
            if not failure_path.exists():
                write_json(failure_path, {"phase": sys.argv[1], "status": "failed",
                    "exception_type": type(error).__name__, "complete_result": False,
                    "missingness": "unexecuted assignments remain missing; no in-place retry or repair"})
        raise
