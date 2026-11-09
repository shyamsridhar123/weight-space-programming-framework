"""Independent inference/scoring checker for frozen synthetic unlearning results.

Reuses only the anchored query constructor, not its predictor, metric, transform,
membership, criterion or edit implementation. Does not retrain models or prove
the optimizer correct. The external result/selection/source identities are inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np

from run_unlearning import (HERE, ARCHITECTURE, METHODS, TRANSFORMS, canonical, sha,
                           file_sha, read_json, read_rows, write_json, examples)


def equal(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Dictionary fields differ: " + label)
        for key in expected:
            equal(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, (float, np.floating)):
        if not np.isclose(actual, expected, rtol=1e-8, atol=2e-7):
            raise ValueError("Numeric outcome differs: " + label)
    elif actual != expected:
        raise ValueError("Outcome differs: " + label)


def weights(row):
    return {name: np.asarray(value, dtype=np.float64) for name, value in row["tensors"].items()}


def artifact_hash(case, tensors):
    return sha({"schema": "wsp-tensors/1", "dtype": "float64", "architecture": ARCHITECTURE,
                "metadata": {"case": case, "captured_scope": "four dense MLP parameter tensors",
                    "input_schema": "fresh fixed synthetic identity key plus attribute code; no language model"},
                "tensors": {name: value.tolist() for name, value in tensors.items()}})


def deployment(tensors, version):
    dtype = np.float64 if version == "binary64" else np.float32
    converted = {}
    for name, tensor in tensors.items():
        tensor = np.array(tensor, copy=True)
        if version in ("symmetric-int8", "symmetric-int4"):
            limit = {"symmetric-int8": 127, "symmetric-int4": 7}[version]
            divisor = max(np.abs(tensor).flat) / limit
            if divisor == 0:
                divisor = 1.0
            tensor = np.minimum(limit, np.maximum(-limit, np.round(tensor / divisor))) * divisor
        converted[name] = tensor.astype(dtype).astype(np.float64)
    return converted, dtype


def posterior(tensors, queries, dtype):
    hidden = np.asarray(queries, dtype=dtype).dot(tensors["hidden.weight"].astype(dtype))
    hidden += tensors["hidden.bias"].astype(dtype)
    hidden[hidden < 0] = 0
    scores = hidden.dot(tensors["readout.weight"].astype(dtype)) + tensors["readout.bias"].astype(dtype)
    exp_scores = np.exp(scores - np.amax(scores, axis=1)[:, None])
    return exp_scores / exp_scores.sum(axis=1)[:, None]


def measures(p, answer):
    truth = p[np.arange(len(answer)), answer].astype(np.float64)
    return {"n_queries": int(len(answer)), "accuracy": float((np.argmax(p, axis=1) == answer).mean()),
            "truth_probability": float(truth.sum() / len(answer)),
            "negative_log_likelihood": float(np.mean(-np.log(np.clip(truth, 1e-30, None))))}


def extraction(p, answer, variants):
    chosen, labels = [], []
    for offset in range(0, len(answer), variants):
        subset = p[offset:offset + variants]
        index = int(np.argmax(np.max(subset, axis=1)))
        chosen.append(subset[index])
        labels.append(answer[offset])
    return measures(np.asarray(chosen), np.asarray(labels))


def privacy(tensors, dtype, case, retained, forgotten, protocol):
    count, half = case["author_count"], protocol["shadow_authors"] // 2
    member_x, member_y, _, fields = examples(case, retained, protocol, "audit", 1)
    member_x, member_y = member_x[fields < 2], member_y[fields < 2]
    non_x, non_y, _, _ = examples(case, range(count, count + half), protocol, "audit", 1)
    loss_vectors = []
    for x, y in ((member_x, member_y), (non_x, non_y)):
        p = posterior(tensors, x, dtype)
        loss_vectors.append(-np.log(np.clip(p[np.arange(len(y)), y].astype(float), 1e-30, None)))
    threshold = .5 * (np.median(loss_vectors[0]) + np.median(loss_vectors[1]))
    forgotten_x, forgotten_y, _, fields = examples(case, forgotten, protocol, "audit", 1)
    forgotten_x, forgotten_y = forgotten_x[fields >= 2], forgotten_y[fields >= 2]
    audit_non_x, audit_non_y, _, _ = examples(case, range(count + half, count + 2 * half), protocol, "audit", 1)
    rates = []
    for x, y in ((forgotten_x, forgotten_y), (audit_non_x, audit_non_y)):
        p = posterior(tensors, x, dtype)
        losses = -np.log(np.clip(p[np.arange(len(y)), y].astype(float), 1e-30, None))
        rates.append(float(np.count_nonzero(losses <= threshold) / len(y)))
    return {"known_answer_loss_threshold": float(threshold), "forgotten_member_tpr": rates[0],
            "never_exposed_fpr": rates[1], "balanced_accuracy": .5 * (rates[0] + 1 - rates[1]),
            "advantage": rates[0] - rates[1], "calibration_queries": len(member_y) + len(non_y),
            "audit_queries": len(forgotten_y) + len(audit_non_y),
            "scope": "supervised known-answer loss diagnostic; neither DP nor legal privacy"}


def manual_edit(case, parent, serialized):
    if artifact_hash(case, parent) != serialized["base_sha256"]:
        raise ValueError("Wrong parent in manual edit replay")
    result = {name: value.copy() for name, value in parent.items()}
    seen = set()
    for edit in serialized["edits"]:
        key = (edit["tensor"], tuple(edit["index"]))
        if key in seen or result[key[0]][key[1]] != edit["before"] or edit["before"] == edit["after"]:
            raise ValueError("Ambiguous, stale or empty cell edit")
        result[key[0]][key[1]] = edit["after"]
        seen.add(key)
    return result


def verify(arguments):
    output = arguments.output
    freeze = read_json(HERE / "freeze.json")
    if file_sha(HERE / "freeze.json") != arguments.expected_freeze_sha256:
        raise ValueError("Source freeze differs from external digest")
    for relative, identity in freeze["files"].items():
        if file_sha(HERE / relative) != identity:
            raise ValueError("Source/input file identity differs: " + relative)
    if file_sha(output / "model-selection-seal.json") != arguments.expected_selection_sha256:
        raise ValueError("Selection differs from external digest")
    if file_sha(output / "results.json") != arguments.expected_results_sha256:
        raise ValueError("Outcomes differ from external digest")
    seal = read_json(output / "model-selection-seal.json")
    for name, identity in seal["files"].items():
        if file_sha(output / name) != identity:
            raise ValueError("Selected model/program file changed")
    manifest = read_json(output / "evaluation-manifest.json")
    for name, identity in manifest["files"].items():
        if file_sha(output / name) != identity:
            raise ValueError("Evaluated artifact file changed")
    protocol, inputs = read_json(HERE / "protocol.json"), read_json(HERE / "input-material.json")
    results = read_json(output / "results.json")
    models = {row["id"]: row for row in read_rows(output / "models.jsonl.gz")}
    edits = {row["id"]: row for row in read_rows(output / "programs.jsonl.gz")}
    recovery_models = {row["id"]: row for row in read_rows(output / "recovery-models.jsonl.gz")}
    records = {row["id"]: row for row in results["outcomes"]}
    assigned = {case["id"] + f"/S{stage}/" + method + "/" + transform
                for case in inputs["cases"] for stage in range(1, protocol["forget_requests"] + 1)
                for method in METHODS for transform in TRANSFORMS}
    if set(records) != assigned or len(records) != len(results["outcomes"]):
        raise ValueError("Assignment denominator, duplicate or missing outcome differs")
    for row in list(models.values()) + list(recovery_models.values()):
        if artifact_hash(row["case"], weights(row)) != row["artifact_sha256"]:
            raise ValueError("Checkpoint tensor identity differs")
    recomputed, rollback_replays = {}, 0
    for case in inputs["cases"]:
        count = case["author_count"]
        base = weights(models[case["id"] + "/base"])
        full_x, full_y, _, _ = examples(case, range(count), protocol)
        full_accuracy = measures(posterior(base, full_x, np.float64), full_y)["accuracy"]
        parent = {method: base for method in ("retain_finetune", "uniform_edit")}
        inverse_lineages = {method: [] for method in parent}
        for stage in range(1, protocol["forget_requests"] + 1):
            forgotten = case["forget_order"][:stage * (count // protocol["forget_batch_divisor"])]
            retained = sorted(set(range(count)) - set(forgotten))
            canonical_x, canonical_y, _, _ = examples(case, forgotten, protocol)
            forget_x, forget_y, forget_e, forget_f = examples(case, forgotten, protocol, "audit", protocol["extraction_variants"])
            retain_x, retain_y, retain_e, _ = examples(case, retained, protocol, "audit", protocol["retained_utility_variants"])
            ref = weights(models[case["id"] + f"/S{stage}/retain_only_reference"])
            retain_train_x, retain_train_y, _, _ = examples(case, retained, protocol)
            ref_fit = measures(posterior(ref, retain_train_x, np.float64), retain_train_y)["accuracy"]
            for method in METHODS:
                model_id = case["id"] + ("/base" if method in METHODS[:2] else f"/S{stage}/" + method)
                candidate = weights(models[model_id])
                if method in parent:
                    edit = edits[model_id]
                    equal(edit["program_sha256"], sha(edit["program"]), "program identity")
                    equal(edit["rollback_sha256"], sha(edit["rollback"]), "rollback identity")
                    applied = manual_edit(case["id"], parent[method], edit["program"])
                    equal(artifact_hash(case["id"], applied), artifact_hash(case["id"], candidate), "exact edited checkpoint")
                    inverse_lineages[method].append(edit["rollback"])
                    archive = candidate
                    for inverse in reversed(inverse_lineages[method]):
                        archive = manual_edit(case["id"], archive, inverse)
                        rollback_replays += 1
                    equal(artifact_hash(case["id"], archive), artifact_hash(case["id"], base), "archival original restore")
                    parent[method] = candidate
                else:
                    archive = candidate if method == "retain_only_reference" else base
                for version in TRANSFORMS:
                    identity = case["id"] + f"/S{stage}/" + method + "/" + version
                    row = records[identity]
                    deployed, dtype = deployment(candidate, version)
                    deployed_ref, ref_dtype = deployment(ref, version)
                    deployed_base, base_dtype = deployment(base, version)
                    deployed_archive, archive_dtype = deployment(archive, version)
                    p = posterior(deployed, forget_x, dtype)
                    cp = posterior(deployed, canonical_x, dtype)
                    rp = posterior(deployed, retain_x, dtype)
                    bp = posterior(deployed_base, retain_x, base_dtype)
                    refp = posterior(deployed_ref, forget_x, ref_dtype)
                    expected = {"canonical_answer": measures(cp, canonical_y),
                        "direct_answer_correct_rate": 0.0 if method == "canonical_refusal" else measures(cp, canonical_y)["accuracy"],
                        "canonical_refusal_rate": 1.0 if method == "canonical_refusal" else 0.0,
                        "audit_accessibility": measures(p, forget_y),
                        "label_free_extraction": extraction(p, forget_y, protocol["extraction_variants"]),
                        "retained_utility": measures(rp, retain_y),
                        "base_retained_accuracy": measures(bp, retain_y)["accuracy"],
                        "reference_accessibility": measures(refp, forget_y),
                        "reference_extraction": extraction(refp, forget_y, protocol["extraction_variants"]),
                        "membership": privacy(deployed, dtype, case, retained, forgotten, protocol),
                        "archival_recovery": extraction(posterior(deployed_archive, forget_x, archive_dtype), forget_y, protocol["extraction_variants"]),
                        "full_exposure_training_accuracy": full_accuracy,
                        "reference_retain_training_accuracy": ref_fit,
                        "transformed_tensors_sha256": sha({name: tensor.tolist() for name, tensor in deployed.items()}),
                        "retained_subgroups": {}, "recovery": {}}
                    for group in (0, 1):
                        mask = np.asarray([case["subgroups"][entity] == group for entity in retain_e])
                        expected["retained_subgroups"][str(group)] = {
                            "candidate_accuracy": measures(rp[mask], retain_y[mask])["accuracy"],
                            "base_accuracy": measures(bp[mask], retain_y[mask])["accuracy"],
                            "authors": sorted(set(retain_e[mask].tolist()))}
                    revealed = forget_f == forget_e % protocol["attribute_count"]
                    for family in protocol["recovery_families"]:
                        recovery_id = case["id"] + f"/S{stage}/" + ("unchanged" if method in METHODS[:2] else method) + "/" + family
                        recovered, recovered_dtype = deployment(weights(recovery_models[recovery_id]), version)
                        recovered_p = posterior(recovered, forget_x, recovered_dtype)
                        expected["recovery"][family] = {"recovery_model_id": recovery_id,
                            "all_forget_queries": measures(recovered_p, forget_y),
                            "previously_revealed_field": measures(recovered_p[revealed], forget_y[revealed]),
                            "unrevealed_fields": measures(recovered_p[~revealed], forget_y[~revealed]),
                            "all_label_free_extraction": extraction(recovered_p, forget_y, protocol["extraction_variants"])}
                    for key, value in expected.items():
                        equal(row[key], value, identity + "." + key)
                    recomputed[identity] = expected
    tolerance = protocol["maximum_accessibility_advantage"]
    for identity, row in records.items():
        r = recomputed[identity]
        reference = recomputed[row["case"] + f"/S{row['stage']}/retain_only_reference/" + row["transform"]]
        recovery = {family: r["recovery"][family]["all_forget_queries" if family == "retain_only_refit" else "unrevealed_fields"]["accuracy"]
                    <= reference["recovery"][family]["all_forget_queries" if family == "retain_only_refit" else "unrevealed_fields"]["accuracy"] + tolerance
                    for family in protocol["recovery_families"]}
        c = {"exposure_gate": r["full_exposure_training_accuracy"] >= protocol["minimum_training_accuracy"],
            "reference_fit_gate": r["reference_retain_training_accuracy"] >= protocol["minimum_training_accuracy"],
            "canonical_answer_suppression": r["direct_answer_correct_rate"] <= reference["direct_answer_correct_rate"] + tolerance,
            "retained_utility": r["base_retained_accuracy"] - r["retained_utility"]["accuracy"] <= protocol["maximum_retained_harm"]
                and all(g["base_accuracy"] - g["candidate_accuracy"] <= protocol["maximum_retained_harm"] for g in r["retained_subgroups"].values()),
            "audit_accessibility": r["audit_accessibility"]["accuracy"] <= reference["audit_accessibility"]["accuracy"] + tolerance
                and r["audit_accessibility"]["truth_probability"] <= reference["audit_accessibility"]["truth_probability"] + tolerance,
            "extraction": r["label_free_extraction"]["accuracy"] <= reference["label_free_extraction"]["accuracy"] + tolerance,
            "membership_proxy": r["membership"]["advantage"] <= reference["membership"]["advantage"] + tolerance,
            "recovery": recovery,
            "archival_recovery": r["archival_recovery"]["accuracy"] <= reference["archival_recovery"]["accuracy"] + tolerance}
        joint = all(c[key] for key in ("exposure_gate", "reference_fit_gate", "canonical_answer_suppression", "retained_utility", "audit_accessibility", "extraction", "membership_proxy")) and all(recovery.values())
        c["weight_only_joint_success"] = joint
        c["archive_inclusive_joint_success"] = joint and c["archival_recovery"]
        equal(row["criteria"], c, identity + ".criteria")
    equal(results["assigned_outcomes"], len(assigned), "assigned denominator")
    equal(results["observed_outcomes"], len(records), "observed denominator")
    equal(results["missing_outcomes"], 0, "missing denominator")
    report = {"schema": "wsp-independent-unlearning-check/1", "verified": True,
        "externally_supplied": {"freeze_sha256": arguments.expected_freeze_sha256,
            "selection_sha256": arguments.expected_selection_sha256, "results_sha256": arguments.expected_results_sha256},
        "all_assigned_outcomes_recomputed": len(records), "all_checkpoint_identities_checked": len(models) + len(recovery_models),
        "manual_rollback_chain_steps_replayed": rollback_replays,
        "independence": "own transform, inference, scoring, threshold, criteria and cell replay; anchored query constructor reused",
        "unverified": ["optimizer correctness", "independent model training reproduction", "formal real-to-native refinement",
            "absence of information in untested internal probes", "production privacy", "untested recovery attacks"]}
    write_json(output / "independent-check.json", report)
    print(canonical(report).decode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-results-sha256", required=True)
    verify(parser.parse_args())
