"""Source-only checks and independent scoring for the frozen v2 amendment.

Reuses only the anchored query constructor, not its predictor, metric, transform,
membership, criterion or edit implementation. Does not retrain models or prove
the optimizer correct. --source-only checks query RNG identities and artificial
durable ledgers without predictor execution. External identities are inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import run_unlearning as runner

from run_unlearning import (HERE, ARCHITECTURE, METHODS, TRANSFORMS, canonical, sha,
                           file_sha, read_json, read_rows, write_json, examples)


def source_inputs(expected):
    freeze = read_json(HERE / "freeze.json")
    if file_sha(HERE / "freeze.json") != expected:
        raise ValueError("Source freeze differs from external digest")
    filenames = {"run_unlearning.py", "check_unlearning.py", "protocol.json", "input-material.json",
                 "framework-source/model.py", "framework-source/artifact.py", "framework-source/program.py", "amendment.json"}
    if freeze["schema"] != "wsp-synthetic-unlearning-freeze/2" or set(freeze["files"]) != filenames:
        raise ValueError("Amendment source inventory differs")
    for relative, identity in freeze["files"].items():
        if file_sha(HERE / relative) != identity:
            raise ValueError("Source/input file identity differs: " + relative)
    amendment = read_json(HERE / "amendment.json")
    if amendment["amended_files_before_freeze"] != {name: value for name, value in freeze["files"].items() if name != "amendment.json"}:
        raise ValueError("Amendment and freeze source inventories disagree")
    original = HERE.parent / "synthetic-unlearning"
    anchor = amendment["original_freeze_raw_sha256"]
    if freeze["original_freeze_raw_sha256"] != anchor or file_sha(original / "freeze.json") != anchor:
        raise ValueError("Original source freeze changed")
    protocol, inputs = read_json(HERE / "protocol.json"), read_json(HERE / "input-material.json")
    original_files = read_json(original / "freeze.json")["files"]
    if amendment["original_frozen_files"] != original_files or protocol["source_amendment"]["original_freeze_raw_sha256"] != anchor:
        raise ValueError("Amendment or protocol original-source inventory/anchor differs")
    for relative, identity in original_files.items():
        if file_sha(original / relative) != identity:
            raise ValueError("Original frozen file changed: " + relative)
    old_inputs = read_json(original / "input-material.json")
    old_inputs["schema"] = "wsp-synthetic-canaries/2"
    old_inputs["amendment"] = "original keys/codes/forget order/initialization preserved; only query RNG identities amended"
    for case in old_inputs["cases"]:
        del case["optimization_seed"], case["audit_seed"]
        case["query_stream_seed"] = case["data_seed"]
    if inputs != old_inputs:
        raise ValueError("Actual data, exposure or initialization changed beyond query stream metadata")
    old_protocol = read_json(original / "protocol.json")
    normalized = json.loads(canonical(protocol))
    changed_paths = ["schema", "status", "resource_caps.abort_policy", "data_and_exposure.identity",
                     "data_and_exposure.training_rows", "analysis_and_reporting.all_assignment_accounting",
                     "analysis_and_reporting.cost", "analysis_and_reporting.verification"]
    if amendment["protocol_changed_paths"] != changed_paths:
        raise ValueError("Undeclared protocol change")
    for path in changed_paths:
        amended_node, original_node = normalized, old_protocol
        parts = path.split(".")
        for part in parts[:-1]:
            amended_node, original_node = amended_node[part], original_node[part]
        amended_node[parts[-1]] = original_node[parts[-1]]
    added = ["query_streams", "durable_attempt_accounting", "source_amendment"]
    if amendment["added_protocol_metadata"] != added:
        raise ValueError("Undeclared added protocol metadata")
    for name in added:
        del normalized[name]
    if normalized != old_protocol:
        raise ValueError("Scientific parameters, methods, endpoints, tolerances or analysis changed")
    if METHODS != ("unchanged", "canonical_refusal", "retain_finetune", "uniform_edit", "retain_only_reference") or TRANSFORMS != ("binary64", "binary32", "symmetric-int8", "symmetric-int4") or ARCHITECTURE != "synthetic-author-code-mlp-relu/1":
        raise ValueError("Declared method, deployment or architecture interface changed")
    return freeze, protocol, inputs


def assigned_attempts(inputs, protocol):
    # Independent schedule derivation, without runner's assignment helpers.
    training, recovery = [], []
    for case in inputs["cases"]:
        count, prefix = case["author_count"], case["id"]
        training.append({"id": prefix + "/full-training", "kind": "full_training", "case": prefix,
            "stage": 0, "model_id": prefix + "/base", "rows": count * protocol["attribute_count"],
            "steps": protocol["training_steps"]})
        for stage in range(1, protocol["forget_requests"] + 1):
            forgotten = stage * (count // protocol["forget_batch_divisor"])
            for method in ("retain_only_reference", "retain_finetune", "uniform_edit"):
                rows = (count if method == "uniform_edit" else count - forgotten) * protocol["attribute_count"]
                if method != "retain_only_reference":
                    rows *= protocol["optimization_variants"]
                identifier = prefix + f"/S{stage}/" + method
                training.append({"id": identifier, "kind": method, "case": prefix, "stage": stage,
                    "model_id": identifier, "rows": rows,
                    "steps": protocol["training_steps"] if method == "retain_only_reference" else protocol["edit_steps"]})
            for method in ("unchanged", "retain_finetune", "uniform_edit", "retain_only_reference"):
                for family in protocol["recovery_families"]:
                    revealed = forgotten if family == "one_label_per_forgotten_author" else 0
                    recovery.append({"id": prefix + f"/S{stage}/" + method + "/" + family,
                        "kind": "recovery", "case": prefix, "stage": stage, "method": method, "family": family,
                        "parent_model_id": prefix + ("/base" if method == "unchanged" else f"/S{stage}/" + method),
                        "training_rows": (count - forgotten) * protocol["attribute_count"] + revealed,
                        "revealed_forget_labels": revealed, "steps": protocol["recovery_steps"]})
    for rows, cap in ((training, "training_attempts"), (recovery, "recovery_attempts")):
        if len(rows) != protocol["resource_caps"][cap] or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Fixed attempt inventory or denominator differs")
    return training, recovery


def checked_ledger(path, assignments, phase, expected, require_complete=False):
    raw = Path(path).read_bytes()
    lines = raw.splitlines(keepends=True)
    partial = bool(lines and not lines[-1].endswith(b"\n"))
    events = [json.loads(line) for line in (lines[:-1] if partial else lines)]
    if not events:
        raise ValueError("Ledger has no committed inventory")
    header = events[0]
    if set(header) != {"schema", "event", "phase", "freeze_sha256", "unix_created", "assignments"} or header["schema"] != "wsp-unlearning-attempt-events/2" or header["event"] != "assignments":
        raise ValueError("Invalid durable ledger inventory")
    if header["phase"] != phase or header["freeze_sha256"] != expected or header["assignments"] != assignments:
        raise ValueError("Durable inventory differs from independently assigned schedule or source anchor")
    inventory = {row["id"]: row for row in assignments}
    if not inventory or len(inventory) != len(assignments) or any(not isinstance(key, str) or not key for key in inventory):
        raise ValueError("Invalid assignment identities")
    def stamp(value, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and not value):
            raise ValueError("Nonfinite, negative or nonnumeric ledger time")
        return value
    created = stamp(header["unix_created"], positive=True)
    started, ended, completed, failed, active = {}, set(), [], [], None
    for event in events[1:]:
        identifier = event["id"]
        if identifier not in inventory:
            raise ValueError("Undeclared attempt event")
        if event["event"] == "start":
            if set(event) != {"event", "id", "unix_started"} or identifier in started or active is not None:
                raise ValueError("Duplicate or overlapping attempt start")
            started[identifier] = stamp(event["unix_started"], positive=True)
            if started[identifier] < created:
                raise ValueError("Start precedes durable inventory")
            active = identifier
            continue
        terminal_fields = {"event", "id", "status", "unix_finished", "wall_seconds", "details"}
        if event["event"] == "failure":
            terminal_fields.add("exception_type")
        if event["event"] not in ("end", "failure") or set(event) != terminal_fields or identifier != active or identifier in ended:
            raise ValueError("Unknown, duplicate or unstarted terminal event")
        status = "ok" if event["event"] == "end" else "failed"
        if event["status"] != status or not isinstance(event["details"], dict):
            raise ValueError("Invalid terminal status or details")
        if stamp(event["unix_finished"], positive=True) < started[identifier]:
            raise ValueError("Finish precedes start")
        stamp(event["wall_seconds"])
        for key, value in event["details"].items():
            if key in {"id", "status", "unix_started", "unix_finished", "wall_seconds", "exception_type"} or (key in inventory[identifier] and value != inventory[identifier][key]):
                raise ValueError("Terminal details alter fixed identity, schedule or status")
        row = {**inventory[identifier], **event["details"], "id": identifier, "status": status,
               "unix_started": started[identifier], "unix_finished": event["unix_finished"], "wall_seconds": event["wall_seconds"]}
        if status == "failed":
            if not isinstance(event["exception_type"], str) or not event["exception_type"]:
                raise ValueError("Failure has no exception type")
            row["exception_type"] = event["exception_type"]
            failed.append(row)
        else:
            completed.append(row)
        ended.add(identifier)
        active = None
    interrupted = sorted(set(started) - ended)
    unstarted = sorted(set(inventory) - set(started))
    if require_complete and (failed or interrupted or unstarted or partial):
        raise ValueError("Incomplete assignment accounting cannot authenticate a complete phase")
    return {"phase": phase, "freeze_sha256": expected, "assigned": len(assignments), "assignments": assignments,
            "completed": completed, "failed": failed, "interrupted": interrupted, "not_started": unstarted,
            "trailing_partial_event": partial, "ledger_sha256": file_sha(path)}


def query_source_check(inputs, protocol):
    settings = protocol["query_streams"]
    if runner.QUERY_NAMESPACE != settings["namespace_uint32"] or runner.QUERY_ROLES != settings["role_uint32"]:
        raise ValueError("Declared query namespace or roles differ from constructor")
    identities, states, by_role = set(), set(), {role: 0 for role in settings["role_uint32"]}
    state_digest = hashlib.sha256()
    original = read_json(HERE.parent / "synthetic-unlearning" / "input-material.json")
    for case in inputs["cases"]:
        for role, tag in settings["role_uint32"].items():
            variants = protocol["optimization_variants"] if role == "optimization" else max(protocol["extraction_variants"], protocol["retained_utility_variants"])
            for author in range(case["author_count"] + protocol["shadow_authors"]):
                for field in range(protocol["attribute_count"]):
                    for variant in range(variants):
                        identity = (settings["namespace_uint32"], tag, case["data_seed"], author, field, variant)
                        if identity != runner.query_identity(case, role, author, field, variant) or identity in identities:
                            raise ValueError("Query domain identity differs or collides")
                        actual = runner.query_rng(case, role, author, field, variant).bit_generator.state
                        separate = np.random.PCG64(np.random.SeedSequence(identity)).state
                        if actual != separate:
                            raise ValueError("Query constructor state differs from independently initialized declared tuple")
                        serialized = canonical(actual)
                        if serialized in states:
                            raise ValueError("Assigned query identities share an initialized PCG64 state")
                        identities.add(identity)
                        states.add(serialized)
                        by_role[role] += 1
                        state_digest.update(canonical({"identity": identity, "state": actual}) + b"\n")
        for role in by_role:
            coordinates = (case["author_count"] + protocol["shadow_authors"] - 1, protocol["attribute_count"] - 1, 0)
            first = runner.query_rng(case, role, *coordinates).normal(size=protocol["identity_dimension"] + protocol["attribute_count"])
            again = runner.query_rng(case, role, *coordinates).normal(size=len(first))
            if not np.array_equal(first, again):
                raise ValueError("Query generation is not repeatable")
        old_case = next(row for row in original["cases"] if row["id"] == case["id"])
        if old_case["optimization_seed"] + 1 != old_case["audit_seed"]:
            raise ValueError("Former role-collision witness differs")
        if runner.query_rng(case, "optimization", 0, 0, 1).bit_generator.state == runner.query_rng(case, "audit", 0, 0, 0).bit_generator.state:
            raise ValueError("Former optimization-v1/audit-v0 RNG collision persists")
    case = inputs["cases"][0]
    invalid = [("unknown", 0, 0, 0), ("optimization", -1, 0, 0), ("audit", True, 0, 0),
               ("audit", 0, 0, 2 ** 32), ("optimization", 0, 0.5, 0)]
    for coordinates in invalid:
        try:
            runner.query_identity(case, *coordinates)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid query identity accepted")
    return {"domain": "all case authors plus all shadow authors, four fields, both optimization variants and every audit variant; safe finite superset of scheduled calls",
            "by_role": by_role, "distinct_identities": len(identities), "distinct_initialized_pcg64_states": len(states),
            "ordered_identity_state_sha256": state_digest.hexdigest(), "old_collision_witnesses_separated": len(inputs["cases"]),
            "repeatability_checked": True, "invalid_identity_coordinates_rejected": len(invalid),
            "scope": "observed state disjointness for this finite assigned domain; neither a mathematical nor statistical independence proof"}


def ledger_source_check(expected):
    assignments = [{"id": "fixture/" + name, "kind": "source_only_fixture"} for name in ("A", "B", "C")]
    reports = {}
    with tempfile.TemporaryDirectory(prefix="wsp-unlearning-source-") as directory:
        directory = Path(directory)
        successful = directory / "successful.jsonl"
        runner.start_ledger(successful, assignments, "fixture", expected)
        for row in assignments:
            with runner.recorded_attempt(successful, row["id"]) as details:
                details["source_only"] = True
        complete = checked_ledger(successful, assignments, "fixture", expected, require_complete=True)
        assert runner.read_attempt_ledger(successful) == complete
        assert runner.complete_attempts(successful) == complete["completed"]
        try:
            runner.start_ledger(successful, assignments, "fixture", expected)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing assignment inventory was overwritten")
        reports["successful"] = {"assigned": 3, "completed": len(complete["completed"]), "no_overwrite": True}

        failed = directory / "failed.jsonl"
        runner.start_ledger(failed, assignments, "fixture", expected)
        try:
            with runner.recorded_attempt(failed, "fixture/A"):
                raise ValueError("source-only deliberate fixture failure")
        except ValueError:
            pass
        failure = checked_ledger(failed, assignments, "fixture", expected)
        assert runner.read_attempt_ledger(failed) == failure
        assert [row["id"] for row in failure["failed"]] == ["fixture/A"]
        assert failure["not_started"] == ["fixture/B", "fixture/C"]
        reports["exception_failure"] = {"failed": [row["id"] for row in failure["failed"]],
                                         "not_started": failure["not_started"], "exception_type": "ValueError"}

        abrupt = directory / "abrupt.jsonl"
        code = '''import json, os, sys
from pathlib import Path
import run_unlearning as runner
def forbidden(*args, **kwargs):
    raise AssertionError("Predictor execution forbidden in source-only fixture")
for name in ("initialize", "fit", "predict", "to_artifact", "framework", "train", "evaluate"):
    setattr(runner, name, forbidden)
runner.start_ledger(Path(sys.argv[1]), json.loads(sys.argv[2]), "fixture", sys.argv[3])
with runner.recorded_attempt(Path(sys.argv[1]), "fixture/A"):
    os._exit(73)
'''
        child = subprocess.run([sys.executable, "-c", code, str(abrupt), canonical(assignments).decode(), expected],
                               cwd=HERE, timeout=15, capture_output=True)
        assert child.returncode == 73
        interrupted = checked_ledger(abrupt, assignments, "fixture", expected)
        assert runner.read_attempt_ledger(abrupt) == interrupted
        assert interrupted["interrupted"] == ["fixture/A"] and interrupted["not_started"] == ["fixture/B", "fixture/C"]
        reports["process_exit_after_fsynced_start"] = {"child_returncode": child.returncode,
            "producer_exited_before_read": True, "interrupted": interrupted["interrupted"], "not_started": interrupted["not_started"]}

        torn = directory / "torn.jsonl"
        runner.start_ledger(torn, assignments, "fixture", expected)
        with runner.recorded_attempt(torn, "fixture/A"):
            pass
        runner.append_event(torn, {"event": "start", "id": "fixture/B", "unix_started": time.time()})
        with torn.open("ab") as handle:
            handle.write(b'{"event":"end","id":"fixture/B"')
            handle.flush()
            os.fsync(handle.fileno())
        partial = checked_ledger(torn, assignments, "fixture", expected)
        assert runner.read_attempt_ledger(torn) == partial
        assert [row["id"] for row in partial["completed"]] == ["fixture/A"]
        assert partial["interrupted"] == ["fixture/B"] and partial["not_started"] == ["fixture/C"] and partial["trailing_partial_event"]
        reports["torn_final_event"] = {"completed_preserved": [row["id"] for row in partial["completed"]],
            "interrupted": partial["interrupted"], "not_started": partial["not_started"], "partial_event_flagged": True}
        for path in (failed, abrupt, torn):
            for verify in (runner.complete_attempts, lambda item: checked_ledger(item, assignments, "fixture", expected, require_complete=True)):
                try:
                    verify(path)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Incomplete attempt ledger accepted as complete")

        events = [json.loads(line) for line in successful.read_bytes().splitlines()]
        malformed = [events + [events[1]], json.loads(canonical(events)), json.loads(canonical(events)), json.loads(canonical(events))]
        malformed[1][1]["unix_started"] = True
        malformed[2][2]["status"] = "failed"
        malformed[3][2]["details"]["id"] = "other/identity"
        for index, fixture in enumerate(malformed):
            path = directory / f"invalid-{index}.jsonl"
            path.write_bytes(b"".join(canonical(event) + b"\n" for event in fixture))
            for verify in (runner.read_attempt_ledger, lambda item: checked_ledger(item, assignments, "fixture", expected)):
                try:
                    verify(path)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Malformed or duplicate ledger event accepted")
        reports["invalid_records_rejected"] = {"duplicate_start": True, "boolean_timestamp": True,
            "inconsistent_terminal_status": True, "identity_overwrite": True, "complete_seal_refused_for_incomplete_ledgers": True}
    return reports


def source_only(arguments):
    freeze, protocol, inputs = source_inputs(arguments.expected_freeze_sha256)
    training, recovery = assigned_attempts(inputs, protocol)
    assert training == runner.training_assignments(inputs, protocol)
    assert recovery == runner.recovery_assignments(inputs, protocol)
    guarded_names = ("initialize", "fit", "predict", "to_artifact", "framework", "train", "evaluate")
    saved = {name: getattr(runner, name) for name in guarded_names}
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("forbidden_predictor_api")
        raise AssertionError("Predictor execution forbidden during source-only checks")
    for name in guarded_names:
        setattr(runner, name, forbidden)
    try:
        query_report = query_source_check(inputs, protocol)
        ledger_report = ledger_source_check(arguments.expected_freeze_sha256)
    finally:
        for name, method in saved.items():
            setattr(runner, name, method)
    assert not calls
    assigned_audit = [case["id"] + f"/S{stage}/" + method + "/" + transform
        for case in inputs["cases"] for stage in range(1, protocol["forget_requests"] + 1)
        for method in METHODS for transform in TRANSFORMS]
    assert len(assigned_audit) == len(set(assigned_audit)) == protocol["resource_caps"]["assigned_method_stage_transform_outcomes"]
    report = {"schema": "wsp-synthetic-unlearning-source-check/2", "mode": "source_only", "verified": True,
        "freeze_raw_sha256": arguments.expected_freeze_sha256, "original_freeze_raw_sha256": freeze["original_freeze_raw_sha256"],
        "frozen_files": freeze["files"], "original_frozen_files_preserved": True,
        "data_initialization_scientific_parameters_and_decisions_preserved": True,
        "query_streams": query_report, "durable_attempt_fixtures": ledger_report,
        "assignment_inventories": {"training": {"assigned": len(training), "sha256": sha(training)},
            "recovery": {"assigned": len(recovery), "sha256": sha(recovery)},
            "audit": {"assigned": len(assigned_audit), "sha256": sha(assigned_audit)},
            "scope": "source bookkeeping inventories only; no scientific effect or independent-unit claim"},
        "predictor_api_calls": len(calls), "provider_calls": 0, "trained": False, "outcomes_evaluated": False,
        "scope": "constructor state checks and artificial durable-ledger fixtures only; no predictor initialized, fitted, inferred or recovered",
        "durability_ceiling": "single local writer with flush/fsync; source fixtures do not prove storage hardware power-loss durability. Buffered checkpoints/outcomes may be absent after phase abort"}
    arguments.output.mkdir(parents=True, exist_ok=True)
    receipt = arguments.output / "source-only-check.json"
    if receipt.exists():
        raise ValueError("Source-check receipt already exists; do not replace it")
    write_json(receipt, report)
    print(canonical({"verified": True, "mode": "source_only", "receipt_raw_sha256": file_sha(receipt),
                     "freeze_raw_sha256": arguments.expected_freeze_sha256, "trained": False}).decode())


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
    _, protocol, inputs = source_inputs(arguments.expected_freeze_sha256)
    if file_sha(output / "model-selection-seal.json") != arguments.expected_selection_sha256:
        raise ValueError("Selection differs from external digest")
    if file_sha(output / "results.json") != arguments.expected_results_sha256:
        raise ValueError("Outcomes differ from external digest")
    seal = read_json(output / "model-selection-seal.json")
    if seal["schema"] != "wsp-synthetic-unlearning-selection/2" or seal["freeze_sha256"] != arguments.expected_freeze_sha256 or seal["audit_outcomes_read"] is not False:
        raise ValueError("Selection boundary or schema differs")
    if set(seal["files"]) != {"models.jsonl.gz", "programs.jsonl.gz", "training-attempts.json", "training-attempt-events.jsonl"}:
        raise ValueError("Selection does not bind the complete durable training ledger")
    for name, identity in seal["files"].items():
        if file_sha(output / name) != identity:
            raise ValueError("Selected model/program file changed")
    manifest = read_json(output / "evaluation-manifest.json")
    if manifest["schema"] != "wsp-synthetic-unlearning-evaluation/2" or manifest["freeze_sha256"] != arguments.expected_freeze_sha256 or manifest["selection_sha256"] != arguments.expected_selection_sha256:
        raise ValueError("Evaluation manifest boundary differs")
    if set(manifest["files"]) != {"results.json", "recovery-models.jsonl.gz", "attack-attempts.json", "recovery-attempt-events.jsonl", "model-selection-seal.json"}:
        raise ValueError("Evaluation does not bind complete durable recovery accounting")
    for name, identity in manifest["files"].items():
        if file_sha(output / name) != identity:
            raise ValueError("Evaluated artifact file changed")
    results = read_json(output / "results.json")
    if results["schema"] != "wsp-synthetic-unlearning-outcomes/2" or results["freeze_sha256"] != arguments.expected_freeze_sha256 or results["model_selection_raw_sha256"] != arguments.expected_selection_sha256:
        raise ValueError("Outcome source/selection boundary differs")
    training_assignment, recovery_assignment = assigned_attempts(inputs, protocol)
    training_ledger = checked_ledger(output / "training-attempt-events.jsonl", training_assignment, "train", arguments.expected_freeze_sha256, require_complete=True)
    recovery_ledger = checked_ledger(output / "recovery-attempt-events.jsonl", recovery_assignment, "evaluate", arguments.expected_freeze_sha256, require_complete=True)
    if read_json(output / "training-attempts.json") != training_ledger["completed"] or read_json(output / "attack-attempts.json") != recovery_ledger["completed"]:
        raise ValueError("Legacy attempt summaries disagree with durable event records")
    raw_models = read_rows(output / "models.jsonl.gz")
    models = {row["id"]: row for row in raw_models}
    raw_edits = read_rows(output / "programs.jsonl.gz")
    edits = {row["id"]: row for row in raw_edits}
    raw_recovery = read_rows(output / "recovery-models.jsonl.gz")
    recovery_models = {row["id"]: row for row in raw_recovery}
    if len(models) != len(raw_models) or set(models) != {row["model_id"] for row in training_assignment}:
        raise ValueError("Training checkpoint inventory differs or has duplicates")
    if len(recovery_models) != len(raw_recovery) or set(recovery_models) != {row["id"] for row in recovery_assignment}:
        raise ValueError("Recovery checkpoint inventory differs or has duplicates")
    if len(edits) != len(raw_edits) or set(edits) != {row["model_id"] for row in training_assignment if row["kind"] in ("retain_finetune", "uniform_edit")}:
        raise ValueError("Exact edit inventory differs or has duplicates")
    for assignment in training_assignment:
        model = models[assignment["model_id"]]
        expected = {"id": assignment["model_id"], "case": assignment["case"], "stage": assignment["stage"],
            "method": "unchanged" if assignment["kind"] == "full_training" else assignment["kind"]}
        for key, value in expected.items():
            equal(model[key], value, assignment["id"] + ".checkpoint." + key)
    for assignment in recovery_assignment:
        for key in ("id", "case", "stage", "method", "family", "parent_model_id"):
            equal(recovery_models[assignment["id"]][key], assignment[key], assignment["id"] + ".checkpoint." + key)
    if seal["models"] != [{"id": row["id"], "artifact_sha256": row["artifact_sha256"]} for row in raw_models]:
        raise ValueError("Model selection inventory differs from saved checkpoints")
    for row in training_ledger["completed"]:
        if row["candidate_sha256"] != models[row["model_id"]]["artifact_sha256"]:
            raise ValueError("Training attempt candidate differs from checkpoint")
    for row in recovery_ledger["completed"]:
        if row["candidate_sha256"] != recovery_models[row["id"]]["artifact_sha256"]:
            raise ValueError("Recovery attempt candidate differs from checkpoint")
    equal(seal["all_training_attempts"], len(training_assignment), "selection training denominator")
    equal(seal["lineage_units"], protocol["lineage_seeds"], "selection lineage units")
    equal(results["training_attempts"], len(training_assignment), "result training denominator")
    equal(results["recovery_attempts"], len(recovery_assignment), "result recovery denominator")
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
                    equal(edit["parent_sha256"], artifact_hash(case["id"], parent[method]), "edit parent checkpoint identity")
                    equal(edit["candidate_sha256"], artifact_hash(case["id"], candidate), "edit candidate checkpoint identity")
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
                    expected_identity = {"id": identity, "case": case["id"], "lineage": case["lineage"],
                        "author_count": count, "stage": stage, "method": method, "transform": version,
                        "status": "ok", "model_id": model_id,
                        "candidate_sha256": models[model_id]["artifact_sha256"],
                        "forget_authors": forgotten, "retain_authors": retained}
                    for key, value in expected_identity.items():
                        equal(row[key], value, identity + "." + key)
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
    report = {"schema": "wsp-independent-unlearning-check/2", "verified": True,
        "externally_supplied": {"freeze_sha256": arguments.expected_freeze_sha256,
            "selection_sha256": arguments.expected_selection_sha256, "results_sha256": arguments.expected_results_sha256},
        "all_assigned_outcomes_recomputed": len(records), "all_checkpoint_identities_checked": len(models) + len(recovery_models),
        "manual_rollback_chain_steps_replayed": rollback_replays,
        "durable_accounting": {"training_assigned_and_terminal_ok": len(training_assignment),
            "recovery_assigned_and_terminal_ok": len(recovery_assignment),
            "training_ledger_sha256": training_ledger["ledger_sha256"], "recovery_ledger_sha256": recovery_ledger["ledger_sha256"],
            "inventory_transitions_timestamps_statuses_and_summaries_independently_checked": True},
        "independence": "own schedule derivation, ledger parsing, transform, inference, scoring, threshold, criteria and cell replay; anchored query constructor reused",
        "unverified": ["optimizer correctness", "independent model training reproduction", "formal real-to-native refinement",
            "absence of information in untested internal probes", "production privacy", "untested recovery attacks"]}
    write_json(output / "independent-check.json", report)
    print(canonical(report).decode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--expected-selection-sha256")
    parser.add_argument("--expected-results-sha256")
    arguments = parser.parse_args()
    if not __debug__:
        parser.error("Checks require assertions enabled; do not use Python -O")
    if arguments.source_only:
        if arguments.expected_selection_sha256 or arguments.expected_results_sha256:
            parser.error("Source-only mode accepts no model-selection or outcome identities")
        source_only(arguments)
    elif not arguments.expected_selection_sha256 or not arguments.expected_results_sha256:
        parser.error("Outcome verification requires external selection and result identities")
    else:
        verify(arguments)
