"""Compare released main-study artifacts using Python's standard library.

This program reads the saved evaluation and replay. It checks canonical
deterministic equality and the original source/input/outcome inventories.
It does not execute an evaluator or attest historical provider provenance.
The original comparison implementation's identity and the precise code
changes are recorded in main-comparison-provenance.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath


EXPECTED_GATE = "601897dc212592ae76cb2eed4b56e120de6ed9d66fa292b7e42ba5b924a13d0f"
LOCK_CANONICAL = "fb465aeac240f769c7d2a5765321e332adae6d4d2edc1a49697587c349c14bbf"
ORIGINAL_COMPARATOR_RAW_SHA256 = "3330e42110fa68ce3aa0527c08abb0947cd24047ab4f93f5609df08ef61d0a78"
ORIGINAL_REPLAY_EXIT_RAW_SHA256 = "cb1509c1991f231b7875b2dafae0df9843e886e7a16f4511e8ec1cadb73f8378"
REPLAY_EXIT_DERIVATIVE_RELATIVE = "artifacts/research/live-studies/verification/main-replay-exit.sanitized.json"
REPLAY_EXIT_DERIVATIVE_CANONICAL_SHA256 = "db2e4f50c00a45add5267f75964bfef4732a582a7bc49d3a632753622983dd30"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def no_nonfinite(value):
    raise ValueError("Nonfinite JSON number")


def read(path, signed=False):
    value = json.loads(path.read_bytes(), object_pairs_hook=unique_object, parse_constant=no_nonfinite)
    require(type(value) is dict, "Expected evidence object")
    if signed:
        require(value.get("sha256") == digest({key: item for key, item in value.items() if key != "sha256"}),
                "Canonical evidence identity mismatch")
    return value


def path_for(repo, relative):
    require(type(relative) is str and "\\" not in relative and ":" not in relative, "Noncanonical evidence path")
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and rel.as_posix() == relative and ".." not in rel.parts, "Unsafe evidence path")
    path = repo.joinpath(*rel.parts)
    require(path.resolve().is_relative_to(repo), "Evidence path escapes checkout")
    return path


def file_hash(path):
    before = path.stat()
    with path.open("rb") as handle:
        identity = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Evidence changed during hashing")
    return identity


def check_record(repo, record):
    path = path_for(repo, record["path"])
    require(path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"], "Raw evidence identity mismatch")


def compare(repo):
    """Return an artifact-comparison report; read evidence without writing files."""
    artifact = repo / "artifacts/research/live-studies"
    receipt = read(path_for(repo, REPLAY_EXIT_DERIVATIVE_RELATIVE), signed=True)
    require(receipt.get("schema") == "wsp-public-main-replay-exit-derivative/1"
            and receipt.get("sha256") == REPLAY_EXIT_DERIVATIVE_CANONICAL_SHA256,
            "Sanitized replay exit derivative differs from released authority")
    require(receipt.get("source", {}).get("raw_sha256") == ORIGINAL_REPLAY_EXIT_RAW_SHA256,
            "Sanitized replay exit source identity differs")
    replay_exit = receipt["recorded"]
    require(replay_exit.get("exit_code") == 0 and replay_exit.get("gate_canonical_sha256") == EXPECTED_GATE,
            "Replay lacks a successful matching terminal receipt")
    gate = read(artifact / "global-selection-seal.json", signed=True)
    require(gate["sha256"] == EXPECTED_GATE, "Gate differs from explicit authority")
    lock = read(repo / "experiments/research/live-studies/preparation-lock.json", signed=True)
    require(lock["sha256"] == LOCK_CANONICAL, "Preparation lock differs from recorded identity")
    assignments = lock["assignments"]
    ids = [row["assignment_id"] for row in assignments]
    require(len(ids) == len(set(ids)) == 224, "Frozen assignment inventory is incomplete")
    require([row["assignment_id"] for row in gate["accounting"]] == ids, "Gate accounting order differs")
    manifests = {phase: read(artifact / phase / "manifest.json", signed=True) for phase in ("evaluation", "replay")}
    starts = {phase: read(artifact / phase / "evaluation-started.json", signed=True) for phase in ("evaluation", "replay")}
    evaluation, replay = manifests["evaluation"], manifests["replay"]
    require(evaluation.get("schema") == "wsp-live-study-results/1" and replay.get("schema") == "wsp-live-replay/1",
            "Unexpected evaluation/replay schemas")
    require(evaluation.get("global_selection_seal_sha256") == replay.get("global_seal_sha256") == EXPECTED_GATE,
            "Manifest gate bindings differ")
    require(evaluation.get("prepared_lock_sha256") == LOCK_CANONICAL, "Evaluation lock binding differs")
    require(evaluation.get("assigned_total") == replay.get("assigned_total") == 224, "Manifest assignment denominator differs")
    require(evaluation.get("sha256") == replay_exit.get("evaluation_manifest_canonical_sha256"), "Evaluation changed after replay launch")
    require(replay_exit.get("replay_manifest_raw_sha256") == file_hash(artifact / "replay/manifest.json"),
            "Replay manifest differs from terminal receipt")
    bound_start_keys = ("schema", "global_selection_seal_sha256", "prepared_lock_sha256", "input_files",
                        "input_files_sha256", "runtime", "selection_frozen")
    require(all(starts["evaluation"].get(key) == starts["replay"].get(key) for key in bound_start_keys),
            "Replay input/runtime binding differs from evaluation")
    require(starts["evaluation"].get("replay") is False and starts["replay"].get("replay") is True,
            "Evaluation/replay start phase differs")
    require(starts["evaluation"].get("global_selection_seal_sha256") == EXPECTED_GATE
            and starts["evaluation"].get("prepared_lock_sha256") == LOCK_CANONICAL
            and starts["evaluation"].get("selection_frozen") is True, "Evaluator start anchors differ")
    inputs = starts["evaluation"]["input_files"]
    input_digest = digest(inputs)
    require(len({record["path"] for record in inputs}) == len(inputs), "Duplicate evaluator input file")
    for record in inputs:
        check_record(repo, record)
    for phase in ("evaluation", "replay"):
        require(starts[phase]["input_files_sha256"] == input_digest
                and manifests[phase]["evaluator_input_files_sha256"] == input_digest
                and manifests[phase]["evaluation_started_sha256"] == starts[phase]["sha256"],
                "Evaluator manifest/input binding differs")
        records = manifests[phase]["outcome_files"]
        expected_paths = {f"artifacts/research/live-studies/{phase}/{assignment_id}/outcome.json" for assignment_id in ids}
        require(len(records) == 224 and {row["path"] for row in records} == expected_paths,
                "Manifest outcome inventory differs from all assigned cases")
        require({path.relative_to(repo).as_posix() for path in (artifact / phase).glob("*/outcome.json")} == expected_paths,
                "Unexpected or missing on-disk outcome")
        for record in records:
            check_record(repo, record)
    comparisons, missing, statuses = [], Counter(), Counter()
    identity_keys = ("assignment_id", "study", "lineage_id", "target_name", "parameter_view", "regime", "repeat", "mechanism")
    for assignment, account in zip(assignments, gate["accounting"]):
        assignment_id = assignment["assignment_id"]
        original = read(artifact / "evaluation" / assignment_id / "outcome.json", signed=True)
        repeated = read(artifact / "replay" / assignment_id / "outcome.json", signed=True)
        for phase, outcome in (("evaluation", original), ("replay", repeated)):
            require(outcome.get("schema") == "wsp-live-outcome/1" and outcome.get("assigned") is True,
                    "Invalid outcome type or assignment flag")
            require(all(outcome.get(key) == assignment[key] for key in identity_keys), "Outcome assignment identity differs")
            require(outcome.get("terminal_status") == account["terminal_status"]
                    and outcome.get("selection_sha256") == account["selection_sha256"]
                    and outcome.get("global_seal_sha256") == EXPECTED_GATE, "Outcome selection/gate binding differs")
            deterministic = outcome["deterministic"]
            require(outcome["deterministic_sha256"] == digest(deterministic), "Deterministic outcome identity mismatch")
            require(deterministic.get("assignment_id") == assignment_id
                    and deterministic.get("selection_sha256") == account["selection_sha256"]
                    and deterministic.get("global_seal_sha256") == EXPECTED_GATE, "Deterministic binding differs")
            require(type(deterministic.get("evaluation_missing")) is bool, "Malformed evaluation missingness")
            missing[phase] += deterministic["evaluation_missing"]
        matches = canonical(original["deterministic"]) == canonical(repeated["deterministic"])
        comparisons.append({"assignment_id": assignment_id,
                            "original_deterministic_sha256": original["deterministic_sha256"],
                            "replay_deterministic_sha256": repeated["deterministic_sha256"],
                            "matches": matches, "evaluation_missing": repeated["deterministic"]["evaluation_missing"]})
        statuses[account["terminal_status"]] += 1
    require(replay.get("records") == comparisons, "Frozen replay comparison records differ from independent recomputation")
    matches_total = sum(row["matches"] for row in comparisons)
    require(matches_total == 224 and replay.get("all_deterministic_outcomes_match") is True,
            "At least one deterministic outcome differs")
    require(evaluation.get("terminal_statuses") == dict(statuses), "Evaluation terminal status denominator differs")
    require(evaluation.get("evaluation_missing_total") == missing["evaluation"]
            and replay.get("evaluation_missing_total") == missing["replay"], "Evaluation missingness accounting differs")
    for record in lock["source_files"]:
        check_record(repo, record)
        require(file_hash(path_for(repo, record["source_path"])) == record["sha256"], "Current executable source differs from frozen snapshot")
    require(read(artifact / "global-selection-seal.json", signed=True)["sha256"] == EXPECTED_GATE,
            "Global gate changed during comparison")
    report = {
        "schema": "wsp-portable-main-artifact-comparison/1", "ok": True,
        "checked_utc": datetime.now(timezone.utc).isoformat(), "gate_canonical_sha256": EXPECTED_GATE,
        "prepared_lock_canonical_sha256": LOCK_CANONICAL,
        "evaluation_manifest_canonical_sha256": evaluation["sha256"],
        "replay_manifest_canonical_sha256": replay["sha256"],
        "assigned_total": 224, "distinct_assignment_ids": 224, "deterministic_matches": matches_total,
        "deterministic_mismatches": 224 - matches_total, "all_assigned_retained": True,
        "terminal_statuses": dict(sorted(statuses.items())), "evaluation_missing_total": missing["evaluation"],
        "replay_missing_total": missing["replay"], "evaluator_input_files_checked": len(inputs),
        "evaluator_input_files_sha256": input_digest, "frozen_source_files_checked": len(lock["source_files"]),
        "outcome_files_checked": 448, "comparison_inventory_sha256": digest(comparisons),
        "matching_runtime": starts["evaluation"]["runtime"], "provider_calls_performed_by_comparator": 0,
        "scope": "Artifact comparison of all released assigned deterministic outputs, with bound input/source/outcome identities and accounting checked. This program does not execute the evaluator or establish provider provenance. Timing and opaque handles are outside deterministic equality.",
    }
    report["original_comparator_raw_sha256"] = ORIGINAL_COMPARATOR_RAW_SHA256
    report["replay_exit_derivative_canonical_sha256"] = receipt["sha256"]
    report["artifact_comparison_only"] = True
    report["evaluator_executed_by_comparator"] = False
    report["provider_provenance_established"] = False
    report["sha256"] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare released main-study evaluation/replay artifacts; no evaluator or provider execution.")
    parser.add_argument("--repo", required=True, type=Path, help="Root of the released repository")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    require(repo.is_dir(), "Repository root must be a directory")
    print(json.dumps(compare(repo), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
