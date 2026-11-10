"""Pilot, locally freeze, evaluate, and independently replay trained head edits.

Raw acquisitions stay outside the public repository. Official test labels are
first parsed only after all main candidates are trained, selected, and sealed.
The source snapshot is executable, so concurrent repository edits cannot change
the code used after the local freeze. This is an estimation study, not a live
LLM experiment, an external preregistration, or a deployment trial.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import shutil
from statistics import NormalDist
import sys
import time
import zipfile

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np

from rlcd.weight_space.artifact import StructuralPolicy, TensorArtifact
from rlcd.weight_space.features import certify_feature_retention, verify_feature_retention
from rlcd.weight_space.model import digest
from rlcd.weight_space.program import Program, apply_program, rollback
from rlcd.weight_space.research_tasks import (
    ARCHITECTURE, TextPreprocessing, behavior_metrics, compile_arm,
    decision_retained_arm, encode_inputs, evaluate_head, feature_geometry,
    grouped_roles, head_spec, load_lineage, make_head, normalized_text, probability_and_margin,
    modular_rank_witness, public_record, retrain_head, train_encoder,
)
from rlcd.weight_space.workflow import AccessPolicy, BudgetLimits, ProgrammerSession, SplitSpec, TaskSpec, verify_trace

ARMS = ("unchanged", "overlay", "context", "target_only", "soft", "nullspace", "decision", "retrain")
PILOT_SEEDS = {"banking77": [6901, 6902, 6903], "optical-digits": [7901, 7902, 7903]}
MAIN_SEEDS = {"banking77": [7001, 7002, 7003, 7004, 7005, 7006],
              "optical-digits": [8001, 8002, 8003, 8004, 8005, 8006]}
TEST_COUNTS = {"banking77": 3080, "optical-digits": 1797}
MARGINS = {"minimum_target_gain": .10, "maximum_protected_harm": .01, "minimum_useful_coverage": .75}
RAW_HASHES = {
    "banking77/banking_data/train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "banking77/banking_data/test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "banking77/banking_data/categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "banking77/LICENSE": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
    "optical-digits/optical-digits.zip": "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0",
    "clinc-oos/data/data_full.json": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    "clinc-oos/LICENSE": "e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public_record(payload), indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def study_paths(public_root: Path, domain: str) -> tuple[Path, Path]:
    name = "trained-support" if domain == "banking77" else "trained-nontext"
    return public_root/"experiments/research"/name, public_root/"artifacts/research"/name


def check_raw(raw_root: Path, domain: str) -> dict:
    included = [key for key in RAW_HASHES if key.startswith(domain+"/")
                or (domain == "banking77" and key.startswith("clinc-oos/"))]
    result = {}
    for key in included:
        path = raw_root/key
        if not path.is_file() or sha(path) != RAW_HASHES[key]:
            raise ValueError(f"Pinned acquisition mismatch: {key}")
        result[key] = {"sha256": RAW_HASHES[key], "bytes": path.stat().st_size}
    return result


def load_parent_data(raw_root: Path, domain: str) -> dict:
    started = time.perf_counter()
    if domain == "banking77":
        labels = read_json(raw_root/"banking77/banking_data/categories.json")
        with (raw_root/"banking77/banking_data/train.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        texts = [row["text"] for row in rows]
        y = np.array([labels.index(row["category"]) for row in rows], dtype=np.int64)
        groups = [hashlib.sha256(normalized_text(text).encode()).hexdigest() for text in texts]
        roles = grouped_roles(y, groups, "banking77-trained-role/1")
        preprocessing = TextPreprocessing.fit([texts[i] for i in roles["train"]])
        x = preprocessing.transform(texts)
        preprocessing_dict = preprocessing.to_dict()
        names = ["cards", "cash", "transfers", "account", "general", "specialist"]
        def queue(label):
            value = label.lower()
            if any(term in value for term in ("cash", "atm")): return 1
            if any(term in value for term in ("transfer", "beneficiary", "receiving_money", "direct_debit")): return 2
            if any(term in value for term in ("card", "pin", "contactless", "apple_pay")): return 0
            if any(term in value for term in ("account", "identity", "verify", "personal", "passcode", "age_limit", "phone")): return 3
            return 4
        queue_map = np.array([queue(label) for label in labels]+[5], dtype=np.int64)
        identity = "banking77"
    else:
        with zipfile.ZipFile(raw_root/"optical-digits/optical-digits.zip") as archive:
            original = np.loadtxt(io.BytesIO(archive.read("optdigits.tra")), delimiter=",", dtype=np.int64)
        x = (original[:, :-1]/16.).astype(np.float32)
        y = original[:, -1].astype(np.int64)
        labels = [str(i) for i in range(10)]
        groups = [hashlib.sha256(row.tobytes()).hexdigest() for row in original[:, :-1]]
        roles = grouped_roles(y, groups, "optical-digits-trained-role/1")
        texts, preprocessing = None, None
        preprocessing_dict = {"schema": "wsp-optical-digits-preprocessing/1", "order": "row-major 8x8 block on-pixel counts",
                              "scale": "fixed divide by 16", "fit_scope": "none; dataset-declared physical scale"}
        names = ["digit-"+str(i) for i in range(10)]+["manual-review"]
        queue_map = np.arange(11, dtype=np.int64)
        identity = "optical-digits"
    ids = np.array([identity+"/train/"+str(i) for i in range(len(y))])
    return {"inputs": x, "labels": y, "label_names": labels, "roles": roles, "ids": ids,
            "group_ids": groups, "texts": texts, "preprocessing": preprocessing,
            "preprocessing_dict": preprocessing_dict, "queue_map": queue_map, "queue_names": names,
            "load_encode_seconds": time.perf_counter()-started}


def split_audit(data: dict) -> dict:
    groups = defaultdict(list)
    for row, key in enumerate(data["group_ids"]): groups[key].append(row)
    role_of = {int(row): name for name, rows in data["roles"].items() for row in rows}
    conflict = [key for key, rows in groups.items() if len(set(map(int, data["labels"][rows]))) > 1]
    exact_cross = [key for key, rows in groups.items() if len({role_of[i] for i in rows}) > 1]
    token_cross = 0
    if data["texts"] is not None:
        near = defaultdict(set)
        for i, text in enumerate(data["texts"]):
            near[" ".join(sorted(set(normalized_text(text).split())))].add(role_of[i])
        token_cross = sum(len(values) > 1 for values in near.values())
    return {"schema": "wsp-trained-split-audit/1", "role_counts": {key: len(rows) for key, rows in data["roles"].items()},
            "role_ids_sha256": {key: digest(data["ids"][rows].tolist()) for key, rows in data["roles"].items()},
            "grouping": "normalized exact text groups; identical raw feature rows for digits",
            "exact_duplicate_groups": sum(len(rows) > 1 for rows in groups.values()),
            "cross_label_duplicate_groups": len(conflict), "cross_role_exact_groups": len(exact_cross),
            "cross_role_token_set_near_groups": token_cross,
            "semantic_near_duplicate_exclusion_verified": False,
            "near_duplicate_boundary": "token-set equality audit is incomplete for semantic paraphrases; fixed requests are crossed across independently trained lineages",
            "official_test_audited": False}


def train_lineage(data: dict, domain: str, seed: int, artifact_root: Path, phase: str) -> dict:
    lineage_id = f"{domain}-{phase}-full-{seed}"
    directory = artifact_root/phase/lineage_id
    trained = train_encoder(data["inputs"], data["labels"], data["roles"]["train"], data["roles"]["development"],
                            seed=seed, hidden=64 if domain == "banking77" else 32,
                            epochs=35 if domain == "banking77" else 40)
    preprocessing_sha = digest(data["preprocessing_dict"])
    head = make_head(trained, preprocessing_sha, data["label_names"], seed, domain)
    directory.mkdir(parents=True, exist_ok=True)
    export_started = time.perf_counter()
    write_json(directory/"head.json", head.to_dict())
    write_json(directory/"preprocessing.json", data["preprocessing_dict"])
    np.savez_compressed(directory/"encoder.npz", **trained["encoder"])
    bundle = {}
    for name, rows in data["roles"].items():
        bundle[name+"_features"] = trained["features"][rows]
        bundle[name+"_labels"] = data["labels"][rows]
        bundle[name+"_ids"] = data["ids"][rows]
        bundle[name+"_row_indices"] = rows
    bundle["queue_map"] = data["queue_map"]
    np.savez_compressed(directory/"features.npz", **bundle)
    record = {"schema": "wsp-trained-lineage/1", "lineage_id": lineage_id, "seed": seed,
              "phase": phase, "domain": domain, "encoder_ancestry": trained["ancestry"],
              "independent_parameters": "complete from-scratch encoder and original head; shared fixed dataset and preprocessing",
              "training_rows": len(data["roles"]["train"]), "development_rows": len(data["roles"]["development"]),
              "selected_epoch": trained["selected_epoch"], "history": trained["history"],
              "initialization_sha256": trained["initialization_sha256"], "selected_parameters_sha256": trained["selected_parameters_sha256"],
              "training_seconds": trained["elapsed_seconds"], "preparation_export_seconds": time.perf_counter()-export_started,
              "head_sha256": head.sha256, "feature_id": head.metadata["feature_id"],
              "encoder_sha256": head.metadata["encoder_sha256"], "preprocessing_sha256": preprocessing_sha,
              "files": {name: {"sha256": sha(directory/name), "bytes": (directory/name).stat().st_size}
                        for name in ("head.json", "encoder.npz", "features.npz", "preprocessing.json")}}
    write_json(directory/"lineage.json", record)
    print(f"trained {lineage_id} in {trained['elapsed_seconds']:.2f}s, development accuracy {max(row['development_intent_accuracy'] for row in trained['history']):.3f}", flush=True)
    return {"id": lineage_id, "directory": directory, "base": head, "trained": trained, "record": record}


def discover_cases(data: dict, lineages: list[dict], domain: str) -> tuple[list[dict], dict]:
    labels = data["labels"]
    dev = data["roles"]["development"]
    ownership_names = ["card_arrival", "pending_cash_withdrawal"] if domain == "banking77" else ["8"]
    cases = []
    for name in ownership_names:
        label = data["label_names"].index(name)
        cases.append({"case_id": "ownership/"+name, "type": "ownership", "target_label": label,
                      "target_name": name, "desired_code": len(data["label_names"]),
                      "desired_queue": int(data["queue_map"][-1]), "selection": "fixed before pilot; no official test opened"})
    confusions = Counter()
    error_rates = defaultdict(list)
    for lineage in lineages:
        prediction = evaluate_head(lineage["base"], lineage["trained"]["features"][dev], native=True).argmax(1)
        for label in range(len(data["label_names"])):
            selected = labels[dev] == label
            error_rates[label].append(float((data["queue_map"][prediction[selected]] != data["queue_map"][label]).mean()))
        for truth, pred in zip(labels[dev], prediction):
            if data["queue_map"][truth] != data["queue_map"][pred]: confusions[(int(truth), int(pred))] += 1
    picked = set()
    for (truth, wrong), count in sorted(confusions.items(), key=lambda item: (-item[1], item[0])):
        if truth in picked: continue
        if sum(rate > 0 for rate in error_rates[truth]) < 2: continue
        picked.add(truth)
        name = data["label_names"][truth]
        cases.append({"case_id": "recognition/"+name, "type": "recognition", "target_label": truth,
                      "target_name": name, "confused_with": data["label_names"][wrong],
                      "desired_code": truth, "desired_queue": int(data["queue_map"][truth]),
                      "selection": "largest aggregated cross-queue confusion over pilot development; distinct truth labels, >=2 pilot lineages err, label-order tie break",
                      "pilot_development_confusion_events": count, "pilot_development_error_rates": error_rates[truth]})
        if len(picked) == (2 if domain == "banking77" else 1): break
    if not picked: raise ValueError("No reproducible development-only recognition case; disclose and revise a new protocol")
    return cases, {"cross_queue_confusions": [{"truth": truth, "predicted": pred, "events": count}
                                             for (truth, pred), count in confusions.most_common()]}


def subset(data: dict, case: dict, role: str, *, target: bool) -> np.ndarray:
    rows = data["roles"][role]
    return rows[(data["labels"][rows] == case["target_label"]) == target]


def external_codes(base: TensorArtifact, features: np.ndarray, case: dict, arm: str, config: dict,
                   target_features: np.ndarray) -> np.ndarray:
    codes = evaluate_head(base, features, native=True).argmax(1)
    if arm == "overlay" and case["type"] == "ownership":
        codes[codes == case["target_label"]] = case["desired_code"]
    elif arm == "context":
        norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), np.finfo(np.float32).tiny)
        anchors = target_features/np.maximum(np.linalg.norm(target_features, axis=1, keepdims=True), np.finfo(np.float32).tiny)
        similarity = ((features/norms) @ anchors.T).max(1)
        codes[similarity >= config["similarity_threshold"]] = case["desired_code"]
    return codes


def calibration_rank(metrics: dict) -> tuple:
    gain, harm = metrics["target_gain"], metrics["protected_harm"]
    return (harm <= MARGINS["maximum_protected_harm"], gain-harm, -harm, gain)


def fit_options(lineage: dict, data: dict, case: dict, arm: str) -> list[dict]:
    base, features = lineage["base"], lineage["trained"]["features"]
    targets = subset(data, case, "target", target=True)
    protected_rows = subset(data, case, "protected", target=False)[:128]
    if not len(targets) or not len(protected_rows): raise ValueError("Case has no eligible target/protected examples")
    target, protected = features[targets], features[protected_rows]
    if arm in ("unchanged", "overlay", "context"):
        configs = [{"similarity_threshold": value} for value in (.65, .75, .85, .95)] if arm == "context" else [{}]
        return [{"program": Program(base.sha256, [], arm+" external intervention"), "artifact": base,
                 "config": config, "diagnostics": {"fitting_scope": "none; frozen target-feature retrieval" if arm == "context" else "none; downstream policy only"},
                 "retention_receipt": None, "fit_seconds": 0.} for config in configs]
    if arm in ("target_only", "soft", "nullspace"):
        mode = "none" if arm == "target_only" else arm
        options = ([{"ridge": .01}, {"ridge": .1}] if arm == "target_only" else
                   [{"retention_weight": value} for value in (1., 8., 64.)] if arm == "soft" else [{}])
        return [compile_arm(base, target, protected, case["desired_code"], mode, **config) | {"config": config} for config in options]
    if arm == "decision":
        return [decision_retained_arm(base, target, protected, case["desired_code"], penalty=value)
                | {"config": {"penalty": value}} for value in (8., 64.)]
    return [retrain_head(base, features, data["labels"], data["roles"]["train"], targets, case["desired_code"],
                         target_weight=value, seed=lineage["record"]["seed"]+100)
            | {"config": {"target_weight": value}} for value in (2, 8)]


def fit_case_arm(lineage: dict, data: dict, case: dict, arm: str, artifact_root: Path, phase: str) -> dict:
    started = time.perf_counter()
    base, features = lineage["base"], lineage["trained"]["features"]
    target_rows = subset(data, case, "target", target=True)
    anchor_rows = subset(data, case, "protected", target=False)[:128]
    calibration = data["roles"]["calibration"]
    parent_cal = evaluate_head(base, features[calibration], native=True).argmax(1)
    trial_id = lineage["id"]+"/"+case["case_id"]+"/"+arm
    directory = artifact_root/phase/lineage["id"]/"attempts"/case["case_id"].replace("/", "-")/arm
    directory.mkdir(parents=True, exist_ok=True)
    options = []
    error = None
    try:
        options = fit_options(lineage, data, case, arm)
        for option in options:
            prediction = external_codes(option["artifact"], features[calibration], case, arm, option["config"], features[target_rows])
            option["calibration"] = behavior_metrics(parent_cal, prediction, data["labels"][calibration],
                case["target_label"], data["queue_map"], case["desired_queue"])
        selected = max(range(len(options)), key=lambda index: calibration_rank(options[index]["calibration"]))
        chosen = options[selected]
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        selected, chosen = None, {"artifact": base, "program": Program(base.sha256, [], "failed proposal fallback"),
                                  "config": {}, "diagnostics": {}, "retention_receipt": None, "fit_seconds": 0., "calibration": None}
    structural_policy = StructuralPolicy(("head.weight", "head.bias"), base.parameter_count, 256., 256.)
    structural = structural_policy.check(base, chosen["program"])
    reasons = []
    if error: reasons.append("proposal-failed")
    if not structural["structural_accepted"]: reasons.extend(structural["issues"])
    if arm != "unchanged" and chosen["calibration"]:
        if chosen["calibration"]["target_gain"] < MARGINS["minimum_target_gain"]: reasons.append("calibration-target-gain")
        if chosen["calibration"]["protected_harm"] > MARGINS["maximum_protected_harm"]: reasons.append("calibration-protected-harm")
    if arm == "nullspace" and (chosen["retention_receipt"] is None
                              or not chosen["retention_receipt"]["exactly_zero_on_supplied_features"]):
        reasons.append("nonzero-exact-anchor-residual")
    admitted = not reasons
    constraints = {"case": case, "arm": arm, "configuration": chosen["config"], "margins": MARGINS,
                   "structural_policy": structural_policy.to_dict(), "numeric_lane": "native Torch CPU float32 readout",
                   "behavioral_admission": "frozen finite calibration samples; no whole-domain guarantee",
                   "nullspace_additional_contract": "exact real-score equality on stored anchors" if arm == "nullspace" else None,
                   "target_feature_sha256": digest(features[target_rows].tolist()),
                   "protected_feature_sha256": digest(features[anchor_rows].tolist())}
    if "protocol_sha256" in data:
        constraints["protocol_sha256"] = data["protocol_sha256"]
    task = TaskSpec(trial_id, "edit", "Change the derived routing decision for a frozen trained representation",
                    ARCHITECTURE, base.sha256, constraints)
    splits = [SplitSpec(name, data["ids"][rows].tolist()) for name, rows in data["roles"].items()]
    splits.append(SplitSpec("test", [lineage["record"]["domain"]+"/test/"+str(i)
                                   for i in range(TEST_COUNTS[lineage["record"]["domain"]])]))
    session = ProgrammerSession(task, base=base, access=AccessPolicy(metadata=True, weights=True),
        limits=BudgetLimits(tool_calls=4, weight_values=base.parameter_count*2, proposals=1), splits=splits)
    session.metadata()
    candidate = session.propose_program(chosen["program"])
    session.seal(candidate if admitted else None)
    write_json(directory/"program.json", chosen["program"].to_dict())
    write_json(directory/"candidate.json", chosen["artifact"].to_dict())
    write_json(directory/"rollback.json", rollback(base, chosen["program"]).to_dict())
    write_json(directory/"task.json", task.to_dict())
    write_json(directory/"pre-heldout-trace.json", [event.to_dict() for event in session.trace])
    if chosen["retention_receipt"] is not None: write_json(directory/"retention.json", chosen["retention_receipt"])
    record = {"schema": "wsp-trained-attempt/1", "trial_id": trial_id, "lineage_id": lineage["id"],
              "case": case, "arm": arm, "assigned": True, "admitted": admitted, "abstained": not admitted,
              "no_op_artifact": candidate.no_op, "external_intervention": arm in ("overlay", "context"),
              "abstention_reasons": reasons, "proposal_error": error,
              "selected_option": selected, "configuration": chosen["config"], "calibration": chosen["calibration"],
              "options": [{"configuration": option["config"], "calibration": option.get("calibration"),
                           "candidate_sha256": option["artifact"].sha256, "fit_seconds": option["fit_seconds"],
                           "diagnostics": option["diagnostics"]} for option in options],
              "structural_admission": structural, "fit_selection_materialization_seconds": time.perf_counter()-started,
              "fit_seconds_all_options": sum(option["fit_seconds"] for option in options),
              "fitting_execution_accounting": "Torch/NumPy work outside cooperative session tools, explicitly timed; not a no-fitting interpreter",
              "parent_sha256": base.sha256, "candidate_artifact_sha256": chosen["artifact"].sha256,
              "program_sha256": chosen["program"].sha256, "task_sha256": task.sha256,
              "session_sha256": session.sha256, "selected_candidate_sha256": candidate.sha256,
              "sealed_at_utc": now(), "target_rows": target_rows.tolist(), "protected_anchor_rows": anchor_rows.tolist(),
              "diagnostics": chosen["diagnostics"]}
    write_json(directory/"attempt.json", record)
    return {"record": record, "directory": directory, "session": session, "candidate": chosen["artifact"],
            "lineage": lineage, "target_features": features[target_rows]}


def restore_trained(path: Path, data: dict) -> dict:
    loaded = load_lineage(path)
    features = np.empty((len(data["labels"]), loaded["base"].tensors["head.weight"].shape[1]), dtype=np.float32)
    for name, rows in data["roles"].items():
        if not np.array_equal(loaded["features"][name+"_row_indices"], rows):
            raise ValueError("Frozen role assignment does not match the parent-data reproduction")
        features[rows] = loaded["features"][name+"_features"]
    return {"id": loaded["id"], "directory": path, "base": loaded["base"],
            "trained": {"features": features, "encoder": loaded["encoder"]}, "record": loaded["record"]}


def pilot_run(public_root: Path, raw_root: Path, domain: str) -> dict:
    """Use development/calibration only; never deserialize official test rows."""
    experiment_root, artifact_root = study_paths(public_root, domain)
    experiment_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw_manifest = check_raw(raw_root, domain)
    data = load_parent_data(raw_root, domain)
    write_json(experiment_root/"split-audit.json", split_audit(data))
    lineages = []
    for seed in PILOT_SEEDS[domain]:
        path = artifact_root/"pilot"/f"{domain}-pilot-full-{seed}"
        lineage = restore_trained(path, data) if (path/"lineage.json").is_file() else train_lineage(data, domain, seed, artifact_root, "pilot")
        lineages.append(lineage)
    cases, discovery = discover_cases(data, lineages, domain)
    write_json(experiment_root/"pilot-cases.json", {"cases": cases, "discovery": discovery, "official_test_opened": False})
    print("pilot cases: "+", ".join(case["case_id"] for case in cases), flush=True)
    records = []
    for lineage in lineages:
        for case in cases:
            target = subset(data, case, "target", target=True)
            anchors = subset(data, case, "protected", target=False)[:128]
            geometry = feature_geometry(lineage["trained"]["features"][anchors], lineage["trained"]["features"][target])
            geometry["threshold_sensitivity_development_only"] = [
                {"relative_cutoff": cutoff, "rank": int(np.count_nonzero(np.array(geometry["singular_values"]) > cutoff*geometry["singular_values"][0]))}
                for cutoff in (1e-6, 1e-9, 1e-12, 1e-15)]
            write_json(lineage["directory"]/"geometry"/(case["case_id"].replace("/", "-")+".json"), geometry)
            for arm in ARMS:
                attempt = fit_case_arm(lineage, data, case, arm, artifact_root, "pilot")
                development = data["roles"]["development"]
                parent = evaluate_head(lineage["base"], lineage["trained"]["features"][development], native=True).argmax(1)
                chosen_codes = external_codes(attempt["candidate"], lineage["trained"]["features"][development], case,
                    arm, attempt["record"]["configuration"], attempt["target_features"])
                metrics = behavior_metrics(parent, chosen_codes, data["labels"][development], case["target_label"], data["queue_map"], case["desired_queue"])
                record = attempt["record"] | {"pilot_development_diagnostic": metrics,
                    "pilot_scope": "Development was also used for training epoch and case discovery; exploratory variance planning only"}
                write_json(attempt["directory"]/"attempt.json", record)
                records.append(record)
                print(f"pilot {lineage['record']['seed']} {case['case_id']} {arm}: admitted={record['admitted']} gain={metrics['target_gain']:.3f} harm={metrics['protected_harm']:.4f}", flush=True)
    summary = {"schema": "wsp-trained-pilot/1", "domain": domain, "created_at_utc": now(),
               "pilot_seeds": PILOT_SEEDS[domain], "main_seeds_reserved": MAIN_SEEDS[domain],
               "cases": cases, "discovery": discovery, "raw_acquisition_hashes": raw_manifest,
               "split_audit": split_audit(data), "assigned_attempts": len(records), "attempts": records,
               "official_test_labels_opened": False, "phase_seconds": time.perf_counter()-started,
               "variance_scope": "Three independent full encoder/head training lineages; shared fixed requests, selection-reused development diagnostics"}
    write_json(experiment_root/"pilot.json", summary)
    return summary


def planning_from_pilot(pilot: dict) -> dict:
    """Preserve small-lineage uncertainty rather than converting rows to n."""
    indexed = {(row["lineage_id"], row["case"]["case_id"], row["arm"]): row for row in pilot["attempts"]}
    recognition = [case for case in pilot["cases"] if case["type"] == "recognition"]
    effects, harms = [], []
    for seed in pilot["pilot_seeds"]:
        lineage = f"{pilot['domain']}-pilot-full-{seed}"
        effect, harm = [], []
        for case in recognition:
            soft = indexed[(lineage, case["case_id"], "soft")]
            overlay = indexed[(lineage, case["case_id"], "overlay")]
            sm, om = soft["pilot_development_diagnostic"], overlay["pilot_development_diagnostic"]
            # Admission failure counts as the unchanged parent, never deletion.
            effect.append((sm["target_gain"] if soft["admitted"] else 0.)-(om["target_gain"] if overlay["admitted"] else 0.))
            harm.append((sm["protected_harm"] if soft["admitted"] else 0.)-(om["protected_harm"] if overlay["admitted"] else 0.))
        effects.append(float(np.mean(effect)))
        harms.append(float(np.mean(harm)))
    point_gain_sd, point_harm_sd = float(np.std(effects, ddof=1)), float(np.std(harms, ddof=1))
    # df=2 chi-square lower 5% quantile is -2*log(.95), exactly.
    upper_multiplier = math.sqrt(2/(-2*math.log(.95)))
    z = NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(.80)
    def required(sd, margin):
        return max(2, int(math.ceil((z*sd/margin)**2)))
    coverage_needed = math.ceil(math.log(.025)/math.log(MARGINS["minimum_useful_coverage"]))
    return {"schema": "wsp-trained-sample-planning/1", "primary_contrast": "soft retention minus unchanged-intent overlay on recognition repairs, averaged within full training lineage",
            "pilot_lineage_target_contrasts": effects, "pilot_lineage_protected_contrasts": harms,
            "pilot_independent_lineages": 3, "pilot_variance_scope": pilot["variance_scope"],
            "pilot_target_sd": point_gain_sd, "pilot_protected_sd": point_harm_sd,
            "margins": MARGINS, "nominal_power": .80, "two_sided_alpha": .05,
            "normal_approximation_formula": "ceil(((z_.975 + z_.80) * paired lineage SD / effect margin)^2); not a finite-n power guarantee",
            "required_lineages_point_sd_target": required(point_gain_sd, MARGINS["minimum_target_gain"]),
            "required_lineages_point_sd_protected": required(point_harm_sd, MARGINS["maximum_protected_harm"]),
            "upper_sd_multiplier_95_df2": upper_multiplier,
            "required_lineages_upper_sd_target": required(point_gain_sd*upper_multiplier, MARGINS["minimum_target_gain"]),
            "required_lineages_upper_sd_protected": required(point_harm_sd*upper_multiplier, MARGINS["maximum_protected_harm"]),
            "sd_sensitivity": [{"multiplier": multiplier,
                "target_lineages": required(max(point_gain_sd, .025)*multiplier, MARGINS["minimum_target_gain"]),
                "protected_lineages": required(max(point_harm_sd, .0025)*multiplier, MARGINS["maximum_protected_harm"])} for multiplier in (1., 2., 4.)],
            "zero_variance_boundary": "Three equal pilot outcomes cannot establish zero future variance; sensitivity floors are explicit .025 gain/.0025 harm, not observed SD",
            "coverage_minimum_even_if_all_lineages_succeed": coverage_needed,
            "coverage_bound_formula": "Two-sided exact binomial lower bound for n/n successes is .025**(1/n); n is independent full lineage, never request rows or cases",
            "maximum_lower_coverage_bound_with_6_lineages": .025**(1/6),
            "planned_main_lineages": 6, "designation": "resource-capped estimation; insufficient independent lineages for the coverage criterion even if every lineage succeeds",
            "pilot_reused_for_confirmation": False,
            "claim_limit": "Pilot selection reuses development/calibration and has only three independent lineages; main estimates do not satisfy a confirmatory success gate"}


def freeze_protocol(public_root: Path, raw_root: Path, domain: str) -> dict:
    experiment_root, _ = study_paths(public_root, domain)
    protocol_path = experiment_root/"protocol.json"
    if protocol_path.exists():
        raise ValueError("Existing local protocol cannot be silently overwritten")
    pilot = read_json(experiment_root/"pilot.json")
    if pilot["official_test_labels_opened"]:
        raise ValueError("Pilot crossed the official test boundary")
    planning = planning_from_pilot(pilot)
    write_json(experiment_root/"sample-planning.json", planning)
    snapshot = experiment_root/"source-freeze"
    if snapshot.exists():
        raise ValueError("Executable source snapshot already exists")
    selected = [SOURCE_ROOT/"rlcd/__init__.py", SOURCE_ROOT/"scripts/run_trained_research.py"]+sorted((SOURCE_ROOT/"rlcd/weight_space").glob("*.py"))
    files = {}
    for source in selected:
        relative = source.relative_to(SOURCE_ROOT)
        content = source.read_bytes()
        destination = snapshot/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        files[relative.as_posix()] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        if source.read_bytes() != content:
            raise ValueError("Source changed while freezing; preserve failed snapshot and make a new protocol")
    source_manifest = {"schema": "wsp-executable-source-freeze/1", "files": files, "frozen_at_utc": now(),
                       "source_scope": "Runner and complete Python weight_space package; all main imports execute from this immutable local snapshot"}
    write_json(snapshot/"source-manifest.json", source_manifest)
    protocol = {"schema": "wsp-trained-protocol/1", "domain": domain, "review_date_local": "2026-11-08",
        "frozen_at_utc": now(), "freeze_kind": "local executable source and benchmark protocol, not externally preregistered",
        "pilot_sha256": sha(experiment_root/"pilot.json"), "sample_planning_sha256": sha(experiment_root/"sample-planning.json"),
        "source_manifest_sha256": sha(snapshot/"source-manifest.json"), "source_files": files,
        "raw_acquisition_hashes": check_raw(raw_root, domain), "cases": pilot["cases"], "arms": list(ARMS),
        "pilot_seeds": PILOT_SEEDS[domain], "main_seeds": MAIN_SEEDS[domain],
        "margins": MARGINS, "margin_rationale": {"minimum_target_gain": "Project feasibility criterion: ten percentage points means at least one additional correct diversion per ten target requests; pilot target-role resolution and SD are reported. This is not an enterprise SLA or measured savings threshold.",
            "maximum_protected_harm": "Project feasibility cap: at most one percentage point net protected queue error increase, with decision flip and per-intent harms reported separately. Improvements cannot conceal those diagnostics.",
            "minimum_useful_coverage": "At least three quarters of all assigned cases should meet gain and harm jointly; independent-lineage confidence limits prevent a six-lineage confirmation claim."},
        "planned_sample_size": planning, "designation": "resource-capped estimation; main is heldout evaluation but not a sufficiently powered confirmation",
        "lineage_unit": "Complete encoder and head initialized/trained independently from scratch; preprocessing and request sets shared across lineages",
        "request_units": "Exact normalized-text or identical-image duplicate groups; shared groups are crossed observations, semantic independence is not proved",
        "role_assignment": "Hash-sort whole normalized exact-duplicate groups stratified by first label, .60/.10/.10/.10/remainder train/development/target/protected/calibration; cross-label groups audited",
        "encoder_training": {"hidden": 64 if domain == "banking77" else 32, "epochs": 35 if domain == "banking77" else 40,
            "learning_rate": .015, "weight_decay": .001, "batch_size": 256, "epoch_selection": "highest development original-intent accuracy, earliest tie"},
        "candidate_selection": "Frozen grid; calibration protected cap first, then target gain minus protected harm, then smaller harm and larger gain; first index tie",
        "fitting_grid": {"context_cosine_threshold": [.65, .75, .85, .95], "target_only_ridge": [.01, .1],
            "soft_retention_weight": [1., 8., 64.], "soft_ridge": .01, "nullspace_relative_cutoff": 1e-12,
            "decision_penalty": [8., 64.], "decision_steps": 150, "retrain_target_oversampling": [2, 8], "retrain_epochs": 12},
        "decision_rule": "Lowest-index argmax original-intent/specialist output followed by fixed derived queue mapping; overlay sees predicted code, never truth",
        "exact_contract": "Numerical nullspace is diagnostic; hard lane additionally requires exact dyadic zero anchor residual on the realized float32 endpoint",
        "all_assigned": "All lineage x case x arm assignments, including failures, rejected edits, no-op artifacts and abstentions; admission failure deploys unchanged parent with zero target gain/harm and joint success false",
        "evaluation_missingness": "Any missing target/protected evaluation or failed lineage is retained, joint success false; sensitivity imputes missing gain [-1,+1], harm [-1,+1], success [0,1]",
        "official_test_count": TEST_COUNTS[domain], "official_test_use": "No test parsing until every main candidate and task is sealed and a global seal ledger is hashed",
        "test_overlap_policy": "Primary novel-row analysis excludes normalized exact duplicate groups seen anywhere in official training roles; full official test secondary. Exclusion keys are fixed before labels/outcomes, exact and token-set overlaps audited after sealing; no semantic near-duplicate claim",
        "primary_contrast": planning["primary_contrast"], "inference": "Two-sided 95% paired Student t over six full lineage means, conditional on fixed requests; independently seeded exact-group crossed bootstrap is a sensitivity estimate. No population-editor inference; few-lineage uncertainty remains",
        "multiplicity": "One prespecified primary soft-vs-overlay recognition target contrast with protected-harm condition; every other arm/case/domain comparison is descriptive, with no familywise confirmatory discovery claim",
        "stress": "BANKING77 fixed development confusion neighbors and heldout near-intent metrics; CLINC oos_test only after seal as OOD dispatch/confidence stress with unrelated labels. No authored paraphrase, multi-intent, or blinded enterprise validation claim" if domain == "banking77" else "Per-digit confusion, official distinct-writer test partition, and native float32/binary64 decision/margin diagnostics; no authored image corruption claims",
        "budgets": {"full_main_lineages": 6, "cpu_torch_threads": 4, "protected_anchor_rows_per_case": 128,
            "maximum_training_epochs_per_lineage": 35 if domain == "banking77" else 40,
            "maximum_selected_candidates_per_case_arm": 1, "live_provider_calls": 0,
            "training_timeout_policy": "Failure retains assigned main seed; no substitute seed"},
        "cost_scope": "Acquisition/preparation, full parent encoder training, all proposal options and selection, exact verification, native/binary64 execution, storage/export and rollback are charged separately; wall times are CPU/environment estimates, not operational savings",
        "raw_publication": "No acquired raw query or image observations in repo; publish derived trained features/model arrays and source/license notices for attributed datasets"}
    write_json(protocol_path, protocol)
    write_json(experiment_root/"protocol-lock.json", {"protocol_sha256": sha(protocol_path), "source_manifest_sha256": sha(snapshot/"source-manifest.json"), "official_test_opened": False})
    print(f"locally frozen {domain}: {sha(protocol_path)}; designation={protocol['designation']}", flush=True)
    return protocol


def verify_frozen_source(public_root: Path, domain: str) -> dict:
    experiment_root, _ = study_paths(public_root, domain)
    protocol = read_json(experiment_root/"protocol.json")
    lock = read_json(experiment_root/"protocol-lock.json")
    snapshot = experiment_root/"source-freeze"
    if SOURCE_ROOT.resolve() != snapshot.resolve():
        raise ValueError("Main evaluation must execute the frozen snapshot's runner in a fresh process")
    if sha(experiment_root/"protocol.json") != lock["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    for name, expected in protocol["source_files"].items():
        if sha(SOURCE_ROOT/name) != expected["sha256"] or (SOURCE_ROOT/name).stat().st_size != expected["bytes"]:
            raise ValueError(f"Frozen executable changed: {name}")
    if sha(snapshot/"source-manifest.json") != lock["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    return protocol


def load_heldout(raw_root: Path, domain: str, data: dict) -> dict:
    """Only called after the complete assignment ledger has been sealed."""
    if domain == "banking77":
        with (raw_root/"banking77/banking_data/test.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        texts = [row["text"] for row in rows]
        labels = np.array([data["label_names"].index(row["category"]) for row in rows], dtype=np.int64)
        inputs = data["preprocessing"].transform(texts)
        groups = [hashlib.sha256(normalized_text(text).encode()).hexdigest() for text in texts]
        train_token_keys = {" ".join(sorted(set(normalized_text(text).split()))) for text in data["texts"]}
        near_seen = np.array([" ".join(sorted(set(normalized_text(text).split()))) in train_token_keys for text in texts])
        original_oos = read_json(raw_root/"clinc-oos/data/data_full.json")["oos_test"]
        oos_inputs = data["preprocessing"].transform([row[0] for row in original_oos])
        oos_ids = np.array(["clinc-oos/oos_test/"+str(i) for i in range(len(original_oos))])
    else:
        with zipfile.ZipFile(raw_root/"optical-digits/optical-digits.zip") as archive:
            original = np.loadtxt(io.BytesIO(archive.read("optdigits.tes")), delimiter=",", dtype=np.int64)
        inputs = (original[:, :-1]/16.).astype(np.float32)
        labels = original[:, -1].astype(np.int64)
        groups = [hashlib.sha256(row.tobytes()).hexdigest() for row in original[:, :-1]]
        near_seen = np.zeros(len(labels), dtype=bool)
        oos_inputs, oos_ids = None, None
    if len(labels) != TEST_COUNTS[domain]:
        raise ValueError("Published test count differs from prespecified split identities")
    known = set(data["group_ids"])
    seen = np.array([key in known for key in groups])
    ids = np.array([domain+"/test/"+str(i) for i in range(len(labels))])
    return {"inputs": inputs, "labels": labels, "ids": ids, "groups": np.array(groups),
            "novel": ~seen, "oos_inputs": oos_inputs, "oos_ids": oos_ids,
            "audit": {"official_test_rows": len(labels), "exact_training_overlap_rows": int(seen.sum()),
                "primary_novel_rows": int((~seen).sum()), "test_exact_duplicate_groups": len(groups)-len(set(groups)),
                "test_token_set_training_overlap_rows": int(near_seen.sum()) if domain == "banking77" else None,
                "semantic_near_duplicate_independence_verified": False,
                "test_labels_first_opened_at_utc": now(),
                "policy": "Frozen exact overlap exclusion primary; official test and token-set audit secondary. No outcome-dependent exclusion"}}


def ood_metrics(base: TensorArtifact, candidate: TensorArtifact, features: np.ndarray,
                case: dict, arm: str, config: dict, target_features: np.ndarray, queue_map: np.ndarray) -> dict:
    parent_scores = evaluate_head(base, features, native=True)
    candidate_scores = evaluate_head(candidate, features, native=True)
    parent = parent_scores.argmax(1)
    after = external_codes(candidate, features, case, arm, config, target_features)
    confidence, margin = probability_and_margin(candidate_scores)
    parent_confidence, _ = probability_and_margin(parent_scores)
    return {"rows": len(features), "parent_specialist_dispatch_rate": float((parent == len(queue_map)-1).mean()),
            "specialist_dispatch_rate": float((after == len(queue_map)-1).mean()),
            "desired_queue_dispatch_rate": float((queue_map[after] == case["desired_queue"]).mean()),
            "queue_flip_rate": float((queue_map[parent] != queue_map[after]).mean()),
            "parent_mean_head_confidence": float(parent_confidence.mean()), "mean_head_confidence": float(confidence.mean()),
            "head_confidence_above_.9": float((confidence > .9).mean()), "median_head_margin": float(np.median(margin)),
            "label_scope": "CLINC oos_test is unrelated to BANKING77 intent/derived queue labels; no OOD routing accuracy or correct specialist assignment claimed",
            "external_confidence_boundary": "External overlay/context changes codes without a probabilistic calibration model; head confidence describes the unchanged head for those arms"}


def evaluate_attempt(attempt: dict, data: dict, test: dict, bundle: dict, protocol_sha256: str,
                     global_seal_sha256: str) -> dict:
    started = time.perf_counter()
    record, case = attempt["record"], attempt["record"]["case"]
    base, candidate, features = attempt["lineage"]["base"], attempt["candidate"], bundle["test_features"]
    arm = record["arm"] if record["proposal_error"] is None else "unchanged"
    native_started = time.perf_counter()
    parent_scores = evaluate_head(base, features, native=True)
    candidate_scores = evaluate_head(candidate, features, native=True)
    parent = parent_scores.argmax(1)
    diagnostic_codes = external_codes(candidate, features, case, arm, record["configuration"], attempt["target_features"])
    deployed = diagnostic_codes.copy() if record["admitted"] else parent.copy()
    native_seconds = time.perf_counter()-native_started
    binary64_started = time.perf_counter()
    exact_lane_parent = evaluate_head(base, features)
    exact_lane_candidate = evaluate_head(candidate, features)
    binary64_seconds = time.perf_counter()-binary64_started
    native_confidence, native_margin = probability_and_margin(candidate_scores)
    def measure(codes, mask):
        return behavior_metrics(parent[mask], codes[mask], test["labels"][mask], case["target_label"], data["queue_map"], case["desired_queue"])
    all_rows = np.ones(len(parent), dtype=bool)
    novel = test["novel"]
    primary = measure(deployed, novel)
    secondary = measure(deployed, all_rows)
    candidate_primary = measure(diagnostic_codes, novel)
    candidate_secondary = measure(diagnostic_codes, all_rows)
    usable = primary["target_gain"] is not None and primary["protected_harm"] is not None
    joint = bool(record["admitted"] and usable and primary["target_gain"] >= MARGINS["minimum_target_gain"]
                 and primary["protected_harm"] <= MARGINS["maximum_protected_harm"])
    nearby = set([data["label_names"].index(case["confused_with"])]) if "confused_with" in case else {
        i for i, queue in enumerate(data["queue_map"][:-1]) if queue == data["queue_map"][case["target_label"]] and i != case["target_label"]}
    verification_started = time.perf_counter()
    applied = apply_program(base, Program.from_dict(read_json(attempt["directory"]/"program.json")))
    reversed_artifact = apply_program(candidate, Program.from_dict(read_json(attempt["directory"]/"rollback.json")))
    if applied.sha256 != candidate.sha256 or reversed_artifact.sha256 != base.sha256:
        raise ValueError("Candidate application/rollback identity failed before heldout receipt")
    verification = {"application_sha256_verified": True, "rollback_sha256_verified": True}
    if (attempt["directory"]/"retention.json").is_file():
        anchor_features = attempt["lineage"]["trained"]["features"][record["protected_anchor_rows"]]
        verification["exact_anchor_receipt_verified"] = verify_feature_retention(base, candidate, head_spec(base), anchor_features,
            read_json(attempt["directory"]/"retention.json"))
        if not verification["exact_anchor_receipt_verified"]: raise ValueError("Stored exact receipt could not be independently recomputed")
    verification_seconds = time.perf_counter()-verification_started
    result = {"primary_novel": primary, "secondary_official": secondary, "joint_useful_success": joint,
              "all_assigned_policy": "Rejected/failed/abstaining candidate deploys unchanged parent, joint success false",
              "candidate_diagnostic_only": {"primary_novel": candidate_primary, "secondary_official": candidate_secondary,
                  "admitted": record["admitted"], "scope": "Sealed candidate diagnostic; rejected outcomes cannot be used for reselection"},
              "nearby_protected_per_label": [row for row in primary["per_label"] if row["label"] in nearby],
              "native_diagnostics": {"lane": "Torch CPU float32 affine/ReLU encoder then affine head",
                  "binary64_lane": "NumPy binary64 affine head on stored float32 feature coordinates; not an exact real proof",
                  "parent_float32_binary64_argmax_disagreements": int((parent != exact_lane_parent.argmax(1)).sum()),
                  "candidate_float32_binary64_argmax_disagreements": int((candidate_scores.argmax(1) != exact_lane_candidate.argmax(1)).sum()),
                  "maximum_parent_score_difference": float(np.max(np.abs(parent_scores.astype(np.float64)-exact_lane_parent))),
                  "maximum_candidate_score_difference": float(np.max(np.abs(candidate_scores.astype(np.float64)-exact_lane_candidate))),
                  "candidate_margin_minimum": float(native_margin.min()), "candidate_margin_quantiles": np.quantile(native_margin, [0., .01, .05, .5, .95, 1.]).tolist(),
                  "candidate_margin_below_1e-4": int((native_margin < 1e-4).sum()), "candidate_mean_head_confidence": float(native_confidence.mean()),
                  "native_guarantee": False}, "verification": verification}
    if "oos_features" in bundle:
        result["ood_stress"] = ood_metrics(base, candidate if record["admitted"] else base, bundle["oos_features"], case,
            arm if record["admitted"] else "unchanged", record["configuration"], attempt["target_features"], data["queue_map"])
    def heldout(selected):
        if selected.artifact.sha256 != record["candidate_artifact_sha256"]:
            raise ValueError("Heldout candidate differs from immutable sealed artifact")
        return result
    receipt = attempt["session"].evaluate_heldout(heldout, evaluator_id="trained-derived-routing/1", split_name="test").to_dict()
    if receipt["status"] != ("succeeded" if record["admitted"] else "abstained"):
        raise ValueError("Heldout workflow did not produce the expected admitted/abstaining receipt")
    if not verify_trace(receipt["trace"], expected_head=receipt["trace_head_sha256"]):
        raise ValueError("Post-heldout session trace failed integrity verification")
    export_started = time.perf_counter()
    np.savez_compressed(attempt["directory"]/"predictions.npz", parent_codes=parent, candidate_codes=diagnostic_codes,
                        deployed_codes=deployed, parent_binary64_codes=exact_lane_parent.argmax(1),
                        candidate_binary64_codes=exact_lane_candidate.argmax(1), native_margins=native_margin,
                        native_confidence=native_confidence)
    write_json(attempt["directory"]/"evaluation-receipt.json", receipt)
    outcome = {"schema": "wsp-trained-outcome/1", "trial_id": record["trial_id"], "lineage_id": record["lineage_id"],
        "arm": record["arm"], "case": case, "assigned": True, "admitted": record["admitted"],
        "abstained": record["abstained"], "no_op_artifact": record["no_op_artifact"], "external_intervention": record["external_intervention"],
        "evaluation_missing": not usable, "abstention_reasons": record["abstention_reasons"],
        "protocol_sha256": protocol_sha256, "global_seal_sha256": global_seal_sha256,
        "candidate_artifact_sha256": candidate.sha256, "task_sha256": record["task_sha256"],
        "evaluation_receipt_status": receipt["status"], "evaluation_receipt_sha256": digest(receipt), "result": result,
        "cost": {"fit_seconds_all_options": record["fit_seconds_all_options"],
            "fit_selection_materialization_seconds": record["fit_selection_materialization_seconds"],
            "native_head_evaluation_seconds": native_seconds, "binary64_head_evaluation_seconds": binary64_seconds,
            "exact_application_rollback_verification_seconds": verification_seconds,
            "heldout_export_seconds": time.perf_counter()-export_started,
            "heldout_workflow_seconds": time.perf_counter()-started,
            "storage_bytes_pre_outcome": sum(path.stat().st_size for path in attempt["directory"].iterdir() if path.is_file())},
        "predictions_sha256": sha(attempt["directory"]/"predictions.npz")}
    if not usable:
        outcome["missing_reason"] = {"type": "MissingPrimaryStratum", "message": "Novel test rows contain no eligible target or protected stratum"}
        outcome["sensitivity"] = {"joint_success": [0, 1], "target_gain": [-1, 1], "protected_harm": [-1, 1]}
    write_json(attempt["directory"]/"outcome.json", outcome)
    print(f"heldout {record['lineage_id']} {case['case_id']} {record['arm']}: joint={joint} gain={primary['target_gain']} harm={primary['protected_harm']}", flush=True)
    return outcome


def mean_interval(values: list[float]) -> dict:
    array = np.array(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else None
    # The protocol permits six lineages; failures are handled with bounds below.
    critical = {3: 4.302652729911275, 6: 2.570581835636314}.get(len(array))
    interval = [mean-critical*sd/math.sqrt(len(array)), mean+critical*sd/math.sqrt(len(array))] if critical else None
    return {"independent_full_lineages": len(array), "lineage_values": values, "mean": mean, "lineage_sd": sd,
            "ci95_student_t_fixed_requests": interval,
            "scope": "Descriptive paired full-lineage means conditional on shared fixed request sample; small-lineage interval is not a confirmatory editor-population claim"}


def summarize_main(public_root: Path, domain: str, protocol: dict, outcomes: list[dict], lineages: list[dict], test: dict) -> dict:
    experiment_root, artifact_root = study_paths(public_root, domain)
    indexed = {(row["lineage_id"], row["case"]["case_id"], row["arm"]): row for row in outcomes}
    summaries = []
    for arm in ARMS:
        assigned = [row for row in outcomes if row["arm"] == arm]
        valid = [row for row in assigned if not row["evaluation_missing"]]
        successes = sum(row["result"]["joint_useful_success"] for row in valid)
        admitted = sum(row["admitted"] for row in assigned)
        by_lineage = []
        gains, harms, coverage = [], [], []
        for seed in MAIN_SEEDS[domain]:
            identity = f"{domain}-main-full-{seed}"
            selected = [indexed[(identity, case["case_id"], arm)] for case in protocol["cases"]]
            observed = [row for row in selected if not row["evaluation_missing"]]
            missing = len(selected)-len(observed)
            gain = sum(row["result"]["primary_novel"]["target_gain"] for row in observed)/len(selected)
            harm = sum(row["result"]["primary_novel"]["protected_harm"] for row in observed)/len(selected)
            success = sum(row["result"]["joint_useful_success"] for row in observed)/len(selected)
            gains.append(gain); harms.append(harm); coverage.append(success)
            by_lineage.append({"lineage_id": identity, "assigned": len(selected), "missing": missing,
                "target_gain_zero_missing": gain, "protected_harm_zero_missing": harm, "joint_useful_coverage": success,
                "gain_missing_bounds": [gain-missing/len(selected), gain+missing/len(selected)],
                "harm_missing_bounds": [harm-missing/len(selected), harm+missing/len(selected)],
                "success_missing_bounds": [success, success+missing/len(selected)]})
        summaries.append({"arm": arm, "assigned": len(assigned), "admitted": admitted, "abstained": len(assigned)-admitted,
            "no_op_artifacts": sum(row["no_op_artifact"] for row in assigned), "external_intervention": arm in ("overlay", "context"),
            "joint_useful_successes": successes, "joint_useful_success_all_assigned": successes/len(assigned),
            "admission_coverage": admitted/len(assigned), "conditional_success_if_admitted": successes/admitted if admitted else None,
            "missing": len(assigned)-len(valid), "lineages": by_lineage,
            "missingness_policy": "Means use zero for missing outcomes; lineage sensitivity bounds retain gain/harm [-1,1] and success [0,1]",
            "target_gain": mean_interval(gains), "protected_harm": mean_interval(harms), "useful_coverage": mean_interval(coverage),
            "total_fit_seconds_all_options": sum(row.get("cost", {}).get("fit_seconds_all_options", 0.) for row in assigned),
            "total_fit_selection_materialization_seconds": sum(row.get("cost", {}).get("fit_selection_materialization_seconds", 0.) for row in assigned),
            "total_heldout_workflow_seconds": sum(row.get("cost", {}).get("heldout_workflow_seconds", 0.) for row in assigned),
            "total_storage_bytes": sum(row.get("cost", {}).get("storage_bytes_pre_outcome", 0.) for row in assigned)})
    target_contrasts, harm_contrasts, primary_missing_bounds = [], [], []
    recognition = [case for case in protocol["cases"] if case["type"] == "recognition"]
    for seed in MAIN_SEEDS[domain]:
        identity = f"{domain}-main-full-{seed}"
        target, protected, missing = [], [], 0
        for case in recognition:
            soft, overlay = indexed[(identity, case["case_id"], "soft")], indexed[(identity, case["case_id"], "overlay")]
            if soft["evaluation_missing"] or overlay["evaluation_missing"]:
                missing += 1
                target.append(0.); protected.append(0.)
            else:
                target.append(soft["result"]["primary_novel"]["target_gain"]-overlay["result"]["primary_novel"]["target_gain"])
                protected.append(soft["result"]["primary_novel"]["protected_harm"]-overlay["result"]["primary_novel"]["protected_harm"])
        target_contrasts.append(float(np.mean(target))); harm_contrasts.append(float(np.mean(protected)))
        primary_missing_bounds.append({"lineage_id": identity, "missing_case_pairs": missing,
            "target_contrast_bounds": [target_contrasts[-1]-2*missing/len(recognition), target_contrasts[-1]+2*missing/len(recognition)],
            "protected_contrast_bounds": [harm_contrasts[-1]-2*missing/len(recognition), harm_contrasts[-1]+2*missing/len(recognition)]})
    primary = {"contrast": protocol["primary_contrast"], "target": mean_interval(target_contrasts),
               "protected": mean_interval(harm_contrasts), "designation": protocol["designation"],
               "missingness_bounds": primary_missing_bounds}
    complete_primary = len(lineages) == 6 and not any(row["missing_case_pairs"] for row in primary_missing_bounds)
    if complete_primary:
        primary["crossed_exact_group_bootstrap"] = crossed_bootstrap(public_root, domain, protocol, test)
    else:
        primary["crossed_exact_group_bootstrap"] = {"status": "not_computed", "reason": "At least one required lineage or primary paired evaluation is missing"}
    summary = {"schema": "wsp-trained-study-summary/1", "domain": domain, "created_at_utc": now(),
        "designation": protocol["designation"], "protocol_sha256": sha(experiment_root/"protocol.json"),
        "global_seal_sha256": sha(experiment_root/"global-seal.json"), "pilot_excluded": True,
        "independent_full_training_lineages": len(lineages), "assigned_full_training_lineages": 6,
        "assigned_attempts": len(outcomes), "outcome_missingness": sum(row["evaluation_missing"] for row in outcomes),
        "official_test_audit": test["audit"], "arms": summaries, "primary": primary,
        "lineage_costs": [row["record"] for row in lineages],
        "claim_status": "Heldout estimates over trained text/non-text models; no sufficiently powered confirmation, no live provider/editor ability, no production savings, no exact whole-domain raw-input or native guarantee"}
    write_json(experiment_root/"results.json", summary)
    write_json(experiment_root/"all-assigned-outcomes.json", outcomes)
    return summary


def crossed_bootstrap(public_root: Path, domain: str, protocol: dict, test: dict) -> dict:
    """Crossed full lineages and shared exact request groups, preserving cases."""
    _, artifact_root = study_paths(public_root, domain)
    recognition = [case for case in protocol["cases"] if case["type"] == "recognition"]
    novel = test["novel"]
    labels = test["labels"][novel]
    groups, inverse = np.unique(test["groups"][novel], return_inverse=True)
    target_mask = np.array([labels == case["target_label"] for case in recognition], dtype=float)
    protected_mask = 1-target_mask
    differences = []
    queue_map = load_lineage(artifact_root/"main"/f"{domain}-main-full-{MAIN_SEEDS[domain][0]}")["features"]["queue_map"]
    for seed in MAIN_SEEDS[domain]:
        line_diffs = []
        for case in recognition:
            root = artifact_root/"main"/f"{domain}-main-full-{seed}"/"attempts"/case["case_id"].replace("/", "-")
            with np.load(root/"soft/predictions.npz", allow_pickle=False) as saved:
                soft = saved["deployed_codes"][novel]
            with np.load(root/"overlay/predictions.npz", allow_pickle=False) as saved:
                overlay = saved["deployed_codes"][novel]
            expected = queue_map[labels].copy()
            expected[labels == case["target_label"]] = case["desired_queue"]
            line_diffs.append((queue_map[soft] == expected).astype(float)-(queue_map[overlay] == expected))
        differences.append(line_diffs)
    differences = np.asarray(differences)
    target_sums = (differences*target_mask[None]).reshape((-1, len(labels)))
    harm_sums = (-differences*protected_mask[None]).reshape((-1, len(labels)))
    rng = np.random.default_rng(98261 if domain == "banking77" else 98262)
    resamples = 2000
    request_counts = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=resamples)[:, inverse]
    lineage_counts = rng.multinomial(6, np.full(6, 1/6), size=resamples)
    target_denominator = request_counts @ target_mask.T
    protected_denominator = request_counts @ protected_mask.T
    valid = (target_denominator > 0).all(1) & (protected_denominator > 0).all(1)
    target = (request_counts @ target_sums.T).reshape((resamples, 6, len(recognition)))/target_denominator[:, None, :]
    harm = (request_counts @ harm_sums.T).reshape((resamples, 6, len(recognition)))/protected_denominator[:, None, :]
    target_values = (target.mean(2)*lineage_counts).sum(1)/6
    harm_values = (harm.mean(2)*lineage_counts).sum(1)/6
    return {"seed": 98261 if domain == "banking77" else 98262, "resamples": resamples, "valid_resamples": int(valid.sum()),
        "exact_request_groups": len(groups), "full_training_lineages": 6, "shared_case_membership_retained": True,
        "target_ci95_percentile": np.quantile(target_values[valid], [.025, .975]).tolist(),
        "protected_ci95_percentile": np.quantile(harm_values[valid], [.025, .975]).tolist(),
        "scope": "Descriptive crossed bootstrap of full independent training lineages and common exact request groups; semantic dependence and six-lineage small-sample limits remain"}


def main_run(public_root: Path, raw_root: Path, domain: str) -> dict:
    protocol = verify_frozen_source(public_root, domain)
    experiment_root, artifact_root = study_paths(public_root, domain)
    if (experiment_root/"global-seal.json").exists():
        raise ValueError("Main selection cannot be repeated after a global seal; use independent replay")
    check_raw(raw_root, domain)
    preparation_started = time.perf_counter()
    data = load_parent_data(raw_root, domain)
    data["protocol_sha256"] = sha(experiment_root/"protocol.json")
    preparation_seconds = time.perf_counter()-preparation_started
    attempts, lineages, failures, attempt_failures = [], [], [], []
    for seed in MAIN_SEEDS[domain]:
        try:
            lineage = train_lineage(data, domain, seed, artifact_root, "main")
            lineages.append(lineage)
        except Exception as failure:
            failures.append({"lineage_id": f"{domain}-main-full-{seed}", "type": type(failure).__name__, "message": str(failure)})
            continue
        for case in protocol["cases"]:
            target = subset(data, case, "target", target=True)
            anchors = subset(data, case, "protected", target=False)[:128]
            try:
                geometry = feature_geometry(lineage["trained"]["features"][anchors], lineage["trained"]["features"][target])
            except Exception as failure:
                geometry = {"status": "failed", "type": type(failure).__name__, "message": str(failure)}
            write_json(lineage["directory"]/"geometry"/(case["case_id"].replace("/", "-")+".json"), geometry)
            for arm in ARMS:
                try:
                    attempts.append(fit_case_arm(lineage, data, case, arm, artifact_root, "main"))
                except Exception as failure:
                    attempt_failures.append({"trial_id": lineage["id"]+"/"+case["case_id"]+"/"+arm,
                        "lineage_id": lineage["id"], "case": case, "arm": arm,
                        "type": type(failure).__name__, "message": str(failure),
                        "scope": "Assigned attempt failed before producing a sealed candidate; retained without a substitute"})
    seal = {"schema": "wsp-trained-global-seal/1", "sealed_at_utc": now(), "protocol_sha256": data["protocol_sha256"],
        "source_manifest_sha256": protocol["source_manifest_sha256"], "assigned_full_lineages": MAIN_SEEDS[domain],
        "assigned_trials": len(MAIN_SEEDS[domain])*len(protocol["cases"])*len(ARMS), "lineage_failures": failures,
        "attempt_failures": attempt_failures,
        "all_main_selection_complete": True, "official_test_labels_parsed": False,
        "attempts": [{"trial_id": row["record"]["trial_id"], "relative_path": row["directory"].relative_to(public_root).as_posix(),
            "attempt_sha256": sha(row["directory"]/"attempt.json"), "preheldout_trace_sha256": sha(row["directory"]/"pre-heldout-trace.json"),
            "candidate_artifact_sha256": row["record"]["candidate_artifact_sha256"], "task_sha256": row["record"]["task_sha256"],
            "sealed_at_utc": row["record"]["sealed_at_utc"], "admitted": row["record"]["admitted"]} for row in attempts]}
    verify_frozen_source(public_root, domain)
    write_json(experiment_root/"global-seal.json", seal)
    seal_sha = sha(experiment_root/"global-seal.json")
    print(f"globally sealed {domain}: {seal_sha}; opening official test now", flush=True)
    test_load_started = time.perf_counter()
    test = load_heldout(raw_root, domain, data)
    test_load_seconds = time.perf_counter()-test_load_started
    write_json(experiment_root/"official-test-audit.json", test["audit"] | {"global_seal_sha256": seal_sha})
    bundles = {}
    encoder_times = {}
    for lineage in lineages:
        encoder_started = time.perf_counter()
        bundle = {"test_features": encode_inputs(lineage["trained"]["encoder"], test["inputs"]),
            "test_labels": test["labels"], "test_ids": test["ids"], "test_group_ids": test["groups"], "test_novel_mask": test["novel"]}
        if test["oos_inputs"] is not None:
            bundle["oos_features"] = encode_inputs(lineage["trained"]["encoder"], test["oos_inputs"])
            bundle["oos_ids"] = test["oos_ids"]
        encoder_times[lineage["id"]] = time.perf_counter()-encoder_started
        np.savez_compressed(lineage["directory"]/"evaluation-features.npz", **bundle)
        write_json(lineage["directory"]/"evaluation-features.json", {"schema": "wsp-heldout-feature-bundle/1",
            "sha256": sha(lineage["directory"]/"evaluation-features.npz"), "bytes": (lineage["directory"]/"evaluation-features.npz").stat().st_size,
            "feature_id": lineage["base"].metadata["feature_id"], "protocol_sha256": data["protocol_sha256"],
            "global_seal_sha256": seal_sha, "native_encoder_seconds": encoder_times[lineage["id"]], "derived_from_attributed_dataset": True})
        bundles[lineage["id"]] = bundle
    outcomes = []
    for attempt in attempts:
        try:
            outcomes.append(evaluate_attempt(attempt, data, test, bundles[attempt["lineage"]["id"]], data["protocol_sha256"], seal_sha))
        except Exception as failure:
            record = attempt["record"]
            outcome = {"schema": "wsp-trained-outcome/1", "trial_id": record["trial_id"], "lineage_id": record["lineage_id"],
                "case": record["case"], "arm": record["arm"], "assigned": True, "admitted": record["admitted"],
                "no_op_artifact": record["no_op_artifact"], "evaluation_missing": True,
                "missing_reason": {"type": type(failure).__name__, "message": str(failure)},
                "sensitivity": {"joint_success": [0, 1], "target_gain": [-1, 1], "protected_harm": [-1, 1]}}
            write_json(attempt["directory"]/"outcome.json", outcome)
            outcomes.append(outcome)
    for failure in failures:
        for case in protocol["cases"]:
            for arm in ARMS:
                outcomes.append({"schema": "wsp-trained-outcome/1", "trial_id": failure["lineage_id"]+"/"+case["case_id"]+"/"+arm,
                    "lineage_id": failure["lineage_id"], "case": case, "arm": arm, "assigned": True, "admitted": False,
                    "no_op_artifact": True, "evaluation_missing": True, "missing_reason": failure,
                    "sensitivity": {"joint_success": [0, 1], "target_gain": [-1, 1], "protected_harm": [-1, 1]}})
    for failure in attempt_failures:
        outcomes.append({"schema": "wsp-trained-outcome/1", "trial_id": failure["trial_id"], "lineage_id": failure["lineage_id"],
            "case": failure["case"], "arm": failure["arm"], "assigned": True, "admitted": False,
            "no_op_artifact": True, "evaluation_missing": True, "missing_reason": failure,
            "sensitivity": {"joint_success": [0, 1], "target_gain": [-1, 1], "protected_harm": [-1, 1]}})
    summary = summarize_main(public_root, domain, protocol, outcomes, lineages, test)
    write_json(experiment_root/"shared-cost.json", {"parent_preparation_seconds": preparation_seconds,
        "test_loading_preprocessing_seconds": test_load_seconds, "native_encoder_seconds_by_lineage": encoder_times,
        "parent_encoder_training_seconds": sum(row["record"]["training_seconds"] for row in lineages),
        "charge_scope": "Shared full parent encoder preparation is separate from each arm's incremental costs; no repeat-count inflation or free retraining preparation"})
    return summary


def replay_study(public_root: Path, domain: str) -> dict:
    """Fresh-process replay using public derived features, never raw data."""
    protocol = verify_frozen_source(public_root, domain)
    experiment_root, artifact_root = study_paths(public_root, domain)
    seal = read_json(experiment_root/"global-seal.json")
    outcomes = read_json(experiment_root/"all-assigned-outcomes.json")
    indexed = {row["trial_id"]: row for row in outcomes}
    started = time.perf_counter()
    cache, verified, problems = {}, [], []
    for entry in seal["attempts"]:
        directory = public_root/entry["relative_path"]
        try:
            if sha(directory/"attempt.json") != entry["attempt_sha256"] or sha(directory/"pre-heldout-trace.json") != entry["preheldout_trace_sha256"]:
                raise ValueError("Pre-test sealed attempt or trace changed")
            record = read_json(directory/"attempt.json")
            if record["lineage_id"] not in cache:
                cache[record["lineage_id"]] = load_lineage(artifact_root/"main"/record["lineage_id"])
            lineage = cache[record["lineage_id"]]
            base = lineage["base"]
            candidate = TensorArtifact.from_dict(read_json(directory/"candidate.json"))
            program = Program.from_dict(read_json(directory/"program.json"))
            if candidate.sha256 != entry["candidate_artifact_sha256"] or apply_program(base, program).sha256 != candidate.sha256:
                raise ValueError("Candidate program identity replay failed")
            if apply_program(candidate, Program.from_dict(read_json(directory/"rollback.json"))).sha256 != base.sha256:
                raise ValueError("Exact rollback replay failed")
            if record["task_sha256"] != entry["task_sha256"] or read_json(directory/"task.json")["constraints"]["protocol_sha256"] != sha(experiment_root/"protocol.json"):
                raise ValueError("Frozen task/protocol identity changed")
            bundle = lineage["features"]
            features = bundle["test_features"]
            targets = bundle["target_features"][bundle["target_labels"] == record["case"]["target_label"]]
            parent = evaluate_head(base, features, native=True).argmax(1)
            arm = record["arm"] if record["proposal_error"] is None else "unchanged"
            diagnostic = external_codes(candidate, features, record["case"], arm, record["configuration"], targets)
            deployed = diagnostic if record["admitted"] else parent
            with np.load(directory/"predictions.npz", allow_pickle=False) as prediction:
                if not all(np.array_equal(prediction[key], value) for key, value in (("parent_codes", parent), ("candidate_codes", diagnostic), ("deployed_codes", deployed))):
                    raise ValueError("Published native predictions differ on independent replay")
            outcome = indexed[record["trial_id"]]
            for name, mask in (("primary_novel", bundle["test_novel_mask"]), ("secondary_official", np.ones(len(parent), dtype=bool))):
                metrics = behavior_metrics(parent[mask], deployed[mask], bundle["test_labels"][mask], record["case"]["target_label"], bundle["queue_map"], record["case"]["desired_queue"])
                if digest(metrics) != digest(outcome["result"][name]): raise ValueError("Heldout metric replay differs")
            receipt = read_json(directory/"evaluation-receipt.json")
            if digest(receipt) != outcome["evaluation_receipt_sha256"] or not verify_trace(receipt["trace"], expected_head=receipt["trace_head_sha256"]):
                raise ValueError("Evaluation receipt/trace integrity failed")
            if (directory/"retention.json").is_file():
                indices = bundle["protected_row_indices"]
                anchors = bundle["protected_features"][[int(np.flatnonzero(indices == index)[0]) for index in record["protected_anchor_rows"]]]
                if not verify_feature_retention(base, candidate, head_spec(base), anchors, read_json(directory/"retention.json")):
                    raise ValueError("Exact endpoint retention receipt replay differs")
            verified.append({"trial_id": record["trial_id"], "application": True, "exact_rollback": True, "native_predictions": True,
                "metrics": True, "sealed_identity": True, "receipt_trace": True, "exact_retention_receipt": (directory/"retention.json").is_file()})
        except Exception as failure:
            problems.append({"trial_id": entry["trial_id"], "type": type(failure).__name__, "message": str(failure)})
    result = {"schema": "wsp-trained-independent-replay/1", "domain": domain, "created_at_utc": now(),
        "protocol_sha256": sha(experiment_root/"protocol.json"), "global_seal_sha256": sha(experiment_root/"global-seal.json"),
        "assigned_trials": seal["assigned_trials"], "verified_trials": len(verified), "problems": problems,
        "training_failures": seal["lineage_failures"], "attempt_failures": seal.get("attempt_failures", []),
        "elapsed_seconds": time.perf_counter()-started,
        "scope": "Independent fresh-process replay of public derived-feature native predictions, metrics, immutable proposals, exact rollback and endpoint receipts; not an independent source acquisition, raw-input proof, machine-checked Lean theorem, or independently retrained replication",
        "verification": verified, "all_replayable_trials_verified": not problems and len(verified) == len(seal["attempts"])}
    write_json(experiment_root/"independent-replay.json", result)
    if problems: raise ValueError(f"{len(problems)} independent replay failures")
    print(f"verified {len(verified)} {domain} sealed trials in fresh process", flush=True)
    return result


def measured_distribution(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("A measured distribution cannot hide nonfinite values")
    return {"observations": len(values), "mean": float(values.mean()) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
            "p50": float(np.quantile(values, .5)) if len(values) else None,
            "p95": float(np.quantile(values, .95)) if len(values) else None}


def analyze_study(public_root: Path, domain: str) -> dict:
    """Describe sealed evidence after replay; never fit, reselect, or update it."""
    experiment_root, artifact_root = study_paths(public_root, domain)
    protocol = read_json(experiment_root/"protocol.json")
    seal = read_json(experiment_root/"global-seal.json")
    replay = read_json(experiment_root/"independent-replay.json")
    if (not replay["all_replayable_trials_verified"]
            or replay["protocol_sha256"] != sha(experiment_root/"protocol.json")
            or replay["global_seal_sha256"] != sha(experiment_root/"global-seal.json")
            or seal["protocol_sha256"] != replay["protocol_sha256"]):
        raise ValueError("Supplemental analysis requires a matching successful frozen replay")
    for name, expected in protocol.get("source_files", {}).items():
        frozen = experiment_root/"source-freeze"/name
        if sha(frozen) != expected["sha256"] or frozen.stat().st_size != expected["bytes"]:
            raise ValueError("Historical source changed before supplemental analysis")
    outcomes = read_json(experiment_root/"all-assigned-outcomes.json")
    if (len(outcomes) != seal["assigned_trials"]
            or len({row["trial_id"] for row in outcomes}) != len(outcomes)
            or not all(row["assigned"] for row in outcomes)):
        raise ValueError("Supplemental analysis must retain every unique assigned trial")
    indexed = {row["trial_id"]: row for row in outcomes}
    sealed_ids = [entry["trial_id"] for entry in seal["attempts"]]
    if len(set(sealed_ids)) != len(sealed_ids) or not set(sealed_ids) <= set(indexed):
        raise ValueError("Sealed proposals must match unique assigned trials")
    inputs = {name: sha(experiment_root/name) for name in
              ("protocol.json", "global-seal.json", "independent-replay.json",
               "all-assigned-outcomes.json", "results.json", "shared-cost.json")}
    cache, spans, trials, calibration = {}, {}, [], []
    for entry in seal["attempts"]:
        directory = public_root/entry["relative_path"]
        if sha(directory/"attempt.json") != entry["attempt_sha256"]:
            raise ValueError("Sealed proposal changed before supplemental analysis")
        record = read_json(directory/"attempt.json")
        identity, case = record["lineage_id"], record["case"]
        if identity not in cache:
            cache[identity] = load_lineage(artifact_root/"main"/identity)
        lineage = cache[identity]
        base, bundle = lineage["base"], lineage["features"]
        candidate = TensorArtifact.from_dict(read_json(directory/"candidate.json"))
        if candidate.sha256 != entry["candidate_artifact_sha256"]:
            raise ValueError("Sealed candidate changed before supplemental analysis")
        inputs[entry["relative_path"]+"/attempt.json"] = entry["attempt_sha256"]
        inputs[entry["relative_path"]+"/candidate.json"] = sha(directory/"candidate.json")
        key = identity+"/"+case["case_id"]
        if key not in spans:
            indices = bundle["protected_row_indices"]
            anchors = bundle["protected_features"][[int(np.flatnonzero(indices == i)[0])
                                                   for i in record["protected_anchor_rows"]]]
            targets = bundle["target_features"][bundle["target_labels"] == case["target_label"]]
            h = np.column_stack((anchors.astype(np.float64), np.ones(len(anchors))))
            t = np.column_stack((targets.astype(np.float64), np.ones(len(targets))))
            geometry_path = lineage["directory"]/"geometry"/(case["case_id"].replace("/", "-")+".json")
            geometry = read_json(geometry_path)
            witness = modular_rank_witness(anchors)
            if digest(witness) != digest(geometry["exact_rank_witness"]):
                raise ValueError("Geometry rank witness no longer matches the frozen anchors")
            zero = np.flatnonzero((h == 0).all(0))
            upper = h.shape[1]-len(zero)
            exact_rank = witness["rank_lower_bound"] == upper
            _, singular, vh = np.linalg.svd(h, full_matrices=True)
            rank = int(np.count_nonzero(singular > geometry["relative_cutoff"]*singular[0]))
            if rank != geometry["numerical_rank"]:
                raise ValueError("Frozen numerical rank differs in supplemental environment")
            basis = vh[rank:].T
            samples = {"fitted_target": targets,
                       "unseen_protected_test": bundle["test_features"][bundle["test_novel_mask"]
                                                        & (bundle["test_labels"] != case["target_label"])]}
            if "oos_features" in bundle:
                samples["clinc_oos_stress"] = bundle["oos_features"]
            coverage = {}
            for name, values in samples.items():
                augmented = np.column_stack((values.astype(np.float64), np.ones(len(values))))
                energy = np.linalg.norm(augmented @ basis, axis=1)**2 / np.maximum(
                    np.linalg.norm(augmented, axis=1)**2, np.finfo(float).tiny)
                outside = (augmented[:, zero] != 0).any(1)
                coverage[name] = {"rows": len(values), "numerical_projected_energy_fraction": measured_distribution(energy),
                                  "exact_rows_outside_anchor_span": int(outside.sum()) if exact_rank else None,
                                  "exact_span_membership_scope": "Stored finite dyadic features; modular rank lower bound equals nonzero-column upper bound" if exact_rank else "unresolved"}
            parent_target = evaluate_head(base, targets, native=True)
            required = np.maximum(0., parent_target.max(1)+1.-parent_target[:, case["desired_code"]])
            outside_norm = np.linalg.norm(t[:, zero], axis=1)
            nonzero = outside_norm > 0
            cap_incompatible = sum(256*sum(Fraction(float(value)) for value in abs(row[zero]))
                                   < Fraction(float(change)) for row, change in zip(t, required))
            spans[key] = {"lineage_id": identity, "case": case,
                "geometry_relative_path": geometry_path.relative_to(public_root).as_posix(),
                "geometry_sha256": sha(geometry_path), "numerical_rank": rank,
                "numerical_nullity": h.shape[1]-rank, "relative_cutoff": geometry["relative_cutoff"],
                "active_singular_condition_number": float(singular[0]/singular[rank-1]),
                "exact_rank_lower_bound": witness["rank_lower_bound"], "exact_rank_upper_bound": upper,
                "exact_rank_certified_by_matching_bounds": exact_rank,
                "exact_nullity_if_certified": h.shape[1]-upper if exact_rank else None,
                "exact_zero_anchor_columns": zero.tolist(), "coverage": coverage,
                "requested_delta_incompatible_target_rows": int(((~nonzero) & (required > 0)).sum()) if exact_rank else None,
                "requested_delta_rows_exceeding_256_cell_change_cap": cap_incompatible if exact_rank else None,
                "cell_cap_bound_scope": "Exact rational l1 comparison for the stored dyadic features and frozen requested deltas; conditional on matching exact rank bounds",
                "rowwise_edit_l2_lower_bounds_on_nonzero_components": measured_distribution(required[nonzero]/outside_norm[nonzero]) if exact_rank else None,
                "requested_delta_scope": "Frozen compiler construction raises desired output to parent maximum plus one; stricter than decision success. Norms are numeric diagnostics, not a Lean proof or a general multi-target solver",
                "implementation_machine_checked": False}
            inputs[geometry_path.relative_to(public_root).as_posix()] = sha(geometry_path)
        delta = np.column_stack((candidate.tensors["head.weight"]-base.tensors["head.weight"],
                                 candidate.tensors["head.bias"]-base.tensors["head.bias"]))
        unseen = bundle["test_features"][bundle["test_novel_mask"]
                                         & (bundle["test_labels"] != case["target_label"])]
        drift = np.column_stack((unseen.astype(np.float64), np.ones(len(unseen)))) @ delta.T
        retention = read_json(directory/"retention.json") if (directory/"retention.json").is_file() else None
        maximum = max(map(Fraction, retention["maximum_residuals_exact"])) if retention else None
        trials.append({"trial_id": record["trial_id"], "arm": record["arm"], "case": case,
            "admitted": record["admitted"], "abstention_reasons": record["abstention_reasons"],
            "evaluation_missing": indexed[record["trial_id"]]["evaluation_missing"],
            "missing_reason": indexed[record["trial_id"]].get("missing_reason"),
            "candidate_augmented_head_l2": float(np.linalg.norm(delta)),
            "candidate_changed_cells": int(np.count_nonzero(delta)),
            "candidate_unseen_protected_score_maximum_abs_binary64": float(np.abs(drift).max()) if len(unseen) else None,
            "candidate_unseen_protected_score_rmse_binary64": float(np.sqrt(np.mean(drift**2))) if len(unseen) else None,
            "anchor_maximum_residual_exact": str(maximum) if maximum is not None else None,
            "anchor_maximum_residual_numeric": float(maximum) if maximum is not None else None,
            "anchor_exactly_zero": retention["exactly_zero_on_supplied_features"] if retention else None,
            "candidate_diagnostic_only": True, "deployed": indexed[record["trial_id"]].get("result", {}).get("primary_novel")})
        for i, option in enumerate(record["options"]):
            metrics = option["calibration"]
            calibration.append({"trial_id": record["trial_id"], "lineage_id": identity, "arm": record["arm"],
                "case": case, "configuration": option["configuration"], "option": i,
                "selected_before_test": i == record["selected_option"], "candidate_sha256": option["candidate_sha256"],
                "target_gain": metrics["target_gain"], "protected_harm": metrics["protected_harm"],
                "behavioral_calibration_pass": metrics["target_gain"] >= MARGINS["minimum_target_gain"]
                                                   and metrics["protected_harm"] <= MARGINS["maximum_protected_harm"],
                "fit_seconds": option["fit_seconds"]})
    arm_costs, case_results, frontiers = [], [], []
    cost_keys = sorted({key for row in outcomes for key in row.get("cost", {})})
    for arm in protocol.get("arms", ARMS):
        assigned = [row for row in outcomes if row["arm"] == arm]
        observed = [row for row in assigned if "cost" in row]
        arm_costs.append({"arm": arm, "assigned": len(assigned), "timing_records": len(observed),
            "timing_missing": len(assigned)-len(observed),
            "stages": {key: measured_distribution(row["cost"][key] for row in observed if key in row["cost"])
                       | {"sum_observed": sum(row["cost"][key] for row in observed if key in row["cost"]),
                          "missing_assigned": sum(key not in row.get("cost", {}) for row in assigned)} for key in cost_keys},
            "observed_incremental_top_level_seconds": sum(row["cost"].get("fit_selection_materialization_seconds", 0.)
                                                          + row["cost"].get("heldout_workflow_seconds", 0.) for row in observed)})
        for case in protocol["cases"]:
            selected = [row for row in assigned if row["case"]["case_id"] == case["case_id"]]
            available = [row for row in selected if not row["evaluation_missing"]]
            case_results.append({"arm": arm, "case": case, "assigned": len(selected), "missing": len(selected)-len(available),
                "admitted": sum(row["admitted"] for row in selected),
                "joint_useful_successes": sum(row["result"]["joint_useful_success"] for row in available),
                "mean_target_gain_zero_missing": sum(row["result"]["primary_novel"]["target_gain"] for row in available)/len(selected),
                "mean_protected_harm_zero_missing": sum(row["result"]["primary_novel"]["protected_harm"] for row in available)/len(selected)})
            options = [row for row in calibration if row["arm"] == arm and row["case"]["case_id"] == case["case_id"]]
            for config in sorted({json.dumps(row["configuration"], sort_keys=True) for row in options}):
                rows = [row for row in options if json.dumps(row["configuration"], sort_keys=True) == config]
                frontiers.append({"arm": arm, "case": case, "configuration": json.loads(config),
                    "available_lineages": len(rows), "assigned_lineages": len(selected),
                    "selected_lineages": sum(row["selected_before_test"] for row in rows),
                    "mean_calibration_target_gain": float(np.mean([row["target_gain"] for row in rows])),
                    "mean_calibration_protected_harm": float(np.mean([row["protected_harm"] for row in rows])),
                    "behavioral_calibration_pass_fraction_all_assigned": sum(row["behavioral_calibration_pass"] for row in rows)/len(selected)})
    result = {"schema": "wsp-trained-posthoc-analysis/1", "domain": domain, "created_at_utc": now(),
        "source_sha256": {"runner": sha(Path(__file__)), "research_tasks": sha(SOURCE_ROOT/"rlcd/weight_space/research_tasks.py")},
        "input_file_sha256": inputs, "assigned_trials": len(outcomes), "analyzable_sealed_trials": len(trials),
        "missing_trial_ids": sorted(set(indexed)-{row["trial_id"] for row in trials}),
        "scope": "Supplemental description of frozen candidates and public derived features after independent replay; no fitting, test-based reselection, replacement lineages, or change to historical results",
        "geometry": list(spans.values()), "trials": trials, "case_results": case_results,
        "calibration_options": calibration, "calibration_gain_harm_frontier": frontiers,
        "frontier_scope": "Recorded pre-test option grid and all-assigned deployed outcomes; unselected calibration options never become heldout claims or new deployment policies",
        "abstention_reason_counts": dict(Counter(reason for row in outcomes for reason in row.get("abstention_reasons", []))),
        "costs": {"shared": read_json(experiment_root/"shared-cost.json"), "arms": arm_costs,
            "current_main_bundle_storage_bytes": sum(path.stat().st_size for path in (artifact_root/"main").rglob("*") if path.is_file()),
            "nesting": "all-option fitting is inside fit/selection/materialization; native, binary64, exact verification and export are inside heldout workflow. Sum only the two top-level incremental timings; shared preparation/training charged once per study",
            "unmeasured": ["network acquisition wall time", "isolated scans/copies/hashes time", "peak memory", "energy", "matched cold/warm repeated-use inference", "hardware/model-size crossover", "LoRA/learned-reader training and amortization"],
            "cost_claim": "Single-environment CPU wall times and observed bytes; no production savings or complete inference-cost frontier"}}
    write_json(experiment_root/"posthoc-analysis.json", result)
    print(f"described {len(trials)} sealed {domain} trials, retaining {len(outcomes)} assigned trials", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--domain", choices=tuple(PILOT_SEEDS), required=True)
    parser.add_argument("--phase", choices=("pilot", "freeze", "main", "replay", "analyze"), required=True)
    args = parser.parse_args()
    if args.phase not in ("replay", "analyze") and args.raw_root is None:
        parser.error("--raw-root is required for pilot, freeze, and main")
    if args.phase == "pilot":
        pilot_run(args.public_root.resolve(), args.raw_root.resolve(), args.domain)
    elif args.phase == "freeze":
        freeze_protocol(args.public_root.resolve(), args.raw_root.resolve(), args.domain)
    elif args.phase == "main":
        main_run(args.public_root.resolve(), args.raw_root.resolve(), args.domain)
    elif args.phase == "replay":
        replay_study(args.public_root.resolve(), args.domain)
    else:
        analyze_study(args.public_root.resolve(), args.domain)


if __name__ == "__main__":
    main()
