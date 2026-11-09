"""Authenticate the 2026-11-08 live-selection gate using only the standard library.

This is a private, read-only identity/accounting check. It imports no project code,
runs no providers or evaluators, and never opens training arrays, injection ledgers,
heldout bundles, or outcome files. Frozen validate-gate supplies semantic replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import sys


LOCK_CANONICAL = "fb465aeac240f769c7d2a5765321e332adae6d4d2edc1a49697587c349c14bbf"
LOCK_RAW = "357637e9bb363ce0e27ef4dba584a3a89586cf56707ed29f52b674d7e4e1f6c8"
PROTOCOL_HASHES = {
    "weight-access-protocol.json": "d39e207131da8e751817ad657c29d09267c679fbe87c6b8cbdf5072c395d6937",
    "blind-diagnosis-protocol.json": "d2750eddc91a37597f26e450cf96fcf952a487cfeb2394fd9b1b3166ab42e938",
}
PROTOCOL = "experiments/research/live-studies"
ARTIFACT = "artifacts/research/live-studies"
LOCK = PROTOCOL + "/preparation-lock.json"
GATE = ARTIFACT + "/global-selection-seal.json"
PREPARED = ARTIFACT + "/prepared"
SELECTIONS = ARTIFACT + "/selections"
TERMINAL = {"sealed", "abstained", "invalid", "failed", "timeout", "interrupted"}
TRAINING_NAMES = {"lineage.json", "head.json", "encoder.npz", "preprocessing.json", "features.npz"}
PROGRAM_NAMES = {"program.json", "rollback.json", "proposal.json", "lowering.json", "candidate.json"}
PROVIDER_NAMES = {"request.json", "response-schema.json", "receipt.json", "events.jsonl", "event-publication.json"}
IDENTITY_KEYS = ("study", "lineage_id", "target_name", "parameter_view", "regime", "repeat", "mechanism")
GATE_KEYS = {
    "schema", "phase", "selection_frozen", "all_assigned_accounted", "studies", "prepared_lock_path",
    "prepared_lock_sha256", "protocol_sha256", "assigned_total", "terminal_total", "all_terminal",
    "evaluation_started", "accounting", "sealed_utc", "sha256",
}
SELECTION_KEYS = {
    "schema", "assignment_id", "terminal_status", "provider_status", "selection_utc",
    "invocation_checkpoint_sha256", "provider_receipt_sha256", "response", "validation_error_type",
    "no_op", "answer_valid", "answer_validation_error_type", "interrupted_transport_termination_attested",
    "parent_sha256", "candidate_sha256", "program_sha256", "broker_trace", "evaluation_started",
    "evidence_files", "selection_rule", "sha256",
}


class Rejected(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Rejected(message)


def is_hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def distinct_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "JSON contains a duplicate object key")
        value[key] = item
    return value


def invalid_constant(_):
    raise Rejected("JSON contains a nonfinite numeric constant")


def signed(value, schema=None):
    require(type(value) is dict and is_hash(value.get("sha256")), "Missing canonical record identity")
    require(value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"}),
            "Canonical record identity mismatch")
    if schema is not None:
        require(value.get("schema") == schema, "Unexpected signed-record schema")
    return value


def safe_path(base, relative, prefix=None):
    require(type(relative) is str and relative and "\\" not in relative and ":" not in relative,
            "Noncanonical evidence path")
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and rel.as_posix() == relative and ".." not in rel.parts,
            "Unsafe evidence path")
    if prefix is not None:
        require(relative.startswith(prefix + "/"), "Evidence path is outside its assigned scope")
    path = base.joinpath(*rel.parts)
    cursor = path
    while cursor != base:
        require(not cursor.is_symlink() and not getattr(cursor, "is_junction", lambda: False)(),
                "Linked evidence path is not permitted")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(base), "Evidence path escapes its declared root")
    return path


def tree_files(base, relative):
    directory = safe_path(base, relative)
    require(directory.is_dir(), "Missing required evidence directory: " + relative)
    files = set()
    for current, dirs, names in os.walk(directory, followlinks=False):
        for name in dirs + names:
            path = Path(current) / name
            rel = path.relative_to(base).as_posix()
            safe_path(base, rel)
            require(path.is_file() or path.is_dir(), "Nonregular evidence entry")
        files.update((Path(current) / name).relative_to(base).as_posix() for name in names)
    return files


def file_hash(path):
    before = path.stat()
    require(path.is_file(), "Expected a regular evidence file")
    with path.open("rb") as handle:
        value = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Evidence changed while being hashed")
    return value, after.st_size


class Audit:
    def __init__(self, root, raw):
        self.root, self.raw = root, raw
        self.observed = {}

    def observe(self, path, expected=None, size=None):
        identity = file_hash(path)
        require(expected is None or identity[0] == expected, "Raw file identity mismatch: " + path.name)
        require(size is None or identity[1] == size, "File byte count mismatch: " + path.name)
        require(path not in self.observed or self.observed[path] == identity,
                "Evidence changed between reads: " + path.name)
        self.observed[path] = identity
        return identity[0]

    def public(self, relative, expected=None, size=None):
        path = safe_path(self.root, relative)
        self.observe(path, expected, size)
        return path

    def load(self, relative, expected=None, size=None, private=False):
        path = safe_path(self.raw if private else self.root, relative)
        self.observe(path, expected, size)
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == self.observed[path][0],
                "Evidence changed while being decoded: " + path.name)
        value = json.loads(data.decode("utf-8"), object_pairs_hook=distinct_object,
                           parse_constant=invalid_constant)
        require(type(value) is dict, "Expected a JSON object: " + path.name)
        return value

    def finish(self):
        for path, expected in self.observed.items():
            require(file_hash(path) == expected, "Evidence changed during authentication: " + path.name)
        records = []
        for path, (sha256, size) in self.observed.items():
            scope, base = ("public", self.root) if path.is_relative_to(self.root) else ("private", self.raw)
            records.append({"scope": scope, "path": path.relative_to(base).as_posix(),
                            "sha256": sha256, "bytes": size})
        return sorted(records, key=lambda x: (x["scope"], x["path"]))


def records_table(records, prefix):
    require(type(records) is list, "File inventory must be an array")
    table, folded = {}, set()
    for record in records:
        require(type(record) is dict and {"path", "sha256", "bytes"} <= set(record),
                "Malformed file identity record")
        path = record["path"]
        require(type(path) is str and path.startswith(prefix + "/") and path.casefold() not in folded,
                "Duplicate or out-of-scope file identity")
        require(is_hash(record["sha256"]) and type(record["bytes"]) is int and record["bytes"] >= 0,
                "Malformed file hash or byte count")
        table[path] = record
        folded.add(path.casefold())
    return table


def expected_assignments():
    lineages = tuple(f"banking77-main-full-{seed}" for seed in range(7001, 7007))
    targets = ("pending_transfer", "card_payment_wrong_exchange_rate")
    views = ("none", "summary", "rounded", "subset", "exact")
    regimes = ("brokered_analysis_only", "brokered_calibration_forward", "brokered_fitting_permitted")
    result = []

    def add(study, lineage, target, view, regime="primary", repeat=0, mechanism=None):
        spec = dict(zip(IDENTITY_KEYS, (study, lineage, target, view, regime, repeat, mechanism)))
        result.append(spec | {"assignment_id": digest(spec)[:24]})

    for lineage, target, view in itertools.product(lineages, targets, views):
        for repeat in range(2 if lineage in lineages[:2] else 1):
            add("weight-access", lineage, target, view, repeat=repeat)
    for lineage, target, view, regime in itertools.product(lineages[:2], targets, ("none", "exact"), regimes):
        add("weight-access", lineage, target, view, regime=regime)
    for lineage, target, mechanism, view in itertools.product(lineages, targets, ("bias_shift", "feature_row_shift"), views):
        add("blind-diagnosis", lineage, target, view, mechanism=mechanism)
    return result


def assert_evaluation_unopened(root):
    for relative in (ARTIFACT + "/evaluation", ARTIFACT + "/replay",
                     ARTIFACT + "/evaluation-started.json", ARTIFACT + "/replay-started.json"):
        require(not os.path.lexists(root / relative), "Evaluation/replay path already exists: " + relative)


def check_preparation(audit, lock):
    require(lock.get("assigned_total") == 224 and type(lock.get("assigned_total")) is int,
            "Frozen assignment total must be 224")
    require(lock.get("selection_only") is True and lock.get("evaluation_started") is False
            and lock.get("provider_calls_started") is False and lock.get("workers") == 8,
            "Preparation phase or worker contract changed")
    require(lock.get("protocol_sha256") == PROTOCOL_HASHES, "Preparation protocol anchors differ")
    specs = expected_assignments()
    assignments = lock.get("assignments")
    require(type(assignments) is list and len(assignments) == 224, "Incorrect assignment array")
    require([{k: item.get(k) for k in (*IDENTITY_KEYS, "assignment_id")} for item in assignments] == specs,
            "Assignments differ from the frozen independent construction")
    for name, identity in PROTOCOL_HASHES.items():
        audit.public(PROTOCOL + "/" + name, identity)
    amendments = lock.get("amendment_sha256", {})
    require(set(amendments) == {"label-amendment.json", "implementation-amendment.json"},
            "Unexpected amendment inventory")
    for name, identity in amendments.items():
        require(is_hash(identity), "Invalid amendment hash")
        audit.public(PROTOCOL + "/" + name, identity)
    sources = records_table(lock.get("source_files"), PROTOCOL + "/source-snapshot")
    require(tree_files(audit.root, PROTOCOL + "/source-snapshot") == set(sources),
            "Source snapshot inventory differs")
    source_names = set()
    for relative, record in sources.items():
        original = record.get("source_path")
        require(type(original) is str and original.endswith(".py") and original not in source_names
                and relative == PROTOCOL + "/source-snapshot/" + original, "Invalid source closure record")
        source_names.add(original)
        audit.public(relative, record["sha256"], record["bytes"])
        audit.public(original, record["sha256"], record["bytes"])
    require("scripts/run_live_studies.py" in source_names, "Frozen executable is not in source closure")
    prepared = records_table(lock.get("permitted_input_files"), PREPARED)
    expected_files = set()
    for spec in specs:
        prefix = PREPARED + "/assignments/" + spec["assignment_id"] + "/"
        expected_files.update(prefix + name for name in ("assignment.json", "parent.json", "prompt.txt", "response-schema.json"))
        if spec["study"] == "blind-diagnosis":
            expected_files.add(prefix + "evaluator-only-injection.json")
    for lineage in {item["lineage_id"] for item in specs}:
        expected_files.update(PREPARED + "/training/" + lineage + "/" + name for name in TRAINING_NAMES)
    require(set(prepared) == expected_files and tree_files(audit.root, PREPARED) == expected_files,
            "Prepared inventory differs from the frozen assignment/training allowlist")
    for relative, record in prepared.items():
        path = safe_path(audit.root, relative, PREPARED)
        require(path.is_file() and path.stat().st_size == record["bytes"], "Prepared file is missing or resized")
    full_assignments = {}
    for item in assignments:
        assignment_id = item["assignment_id"]
        prefix = PREPARED + "/assignments/" + assignment_id + "/"
        require(item.get("assignment_path") == prefix + "assignment.json"
                and item.get("prompt_path") == prefix + "prompt.txt", "Frozen assignment paths differ")
        record = prepared[item["assignment_path"]]
        require(record["sha256"] == item.get("assignment_sha256"), "Assignment raw identities disagree")
        assignment = signed(audit.load(item["assignment_path"], record["sha256"], record["bytes"]))
        require(all(assignment.get(key) == item[key] for key in (*IDENTITY_KEYS, "assignment_id")),
                "Assignment identity differs from lock")
        require(assignment.get("parent_path") == prefix + "parent.json"
                and assignment.get("schema_path") == prefix + "response-schema.json"
                and assignment.get("training_path") == PREPARED + "/training/" + item["lineage_id"],
                "Assignment input paths differ")
        for key in ("prompt_path", "prompt_sha256", "prompt_bytes"):
            require(assignment.get(key) == item.get(key), "Assignment prompt identity differs from lock")
        prompt_record = prepared[item["prompt_path"]]
        require(prompt_record["sha256"] == item["prompt_sha256"] and prompt_record["bytes"] == item["prompt_bytes"],
                "Prompt identity records disagree")
        audit.public(item["prompt_path"], item["prompt_sha256"], item["prompt_bytes"])
        schema_record = prepared[assignment["schema_path"]]
        schema = audit.load(assignment["schema_path"], schema_record["sha256"], schema_record["bytes"])
        parent_record = prepared[assignment["parent_path"]]
        parent = audit.load(assignment["parent_path"], parent_record["sha256"], parent_record["bytes"])
        require(parent.get("schema") == "wsp-tensors/1" and parent.get("dtype") == "float64", "Unexpected parent format")
        full_assignments[assignment_id] = (item, assignment, schema, digest(parent))
    return full_assignments, prepared, sources


def check_provider(audit, prefix, assignment, schema, checkpoint, selected, evidence):
    assignment_id = assignment["assignment_id"]
    public_prefix, private_prefix = prefix + "/provider/", assignment_id + "/provider/"
    if selected["provider_receipt_sha256"] is None:
        require(selected["provider_status"] == "interrupted" and selected["response"] is None,
                "Unreceipted selection claims a provider result")
        require(public_prefix + "receipt.json" not in evidence
                and not os.path.lexists(safe_path(audit.raw, private_prefix + "receipt.json")),
                "An interrupted selection leaves a recoverable receipt unbound")
        return
    require({public_prefix + name for name in PROVIDER_NAMES} <= set(evidence), "Incomplete provider publication")
    receipt = signed(audit.load(public_prefix + "receipt.json"), "wsp-live-provider-attempt/1")
    require(receipt["sha256"] == selected["provider_receipt_sha256"]
            and receipt.get("status") == selected["provider_status"]
            and receipt.get("status") in {"completed", "failed", "timeout"}, "Provider receipt status/identity differs")
    require(canonical(receipt.get("response")) == canonical(selected["response"]), "Committed provider response differs")
    invocation = receipt.get("invocation", {})
    require(invocation.get("prompt_sha256") == assignment["prompt_sha256"]
            and invocation.get("schema_sha256") == checkpoint["schema_sha256"]
            and invocation.get("configuration") == checkpoint["provider_configuration"]
            and invocation.get("cwd_role") == "isolated_proposer_directory", "Provider invocation binding differs")
    require(is_hash(receipt.get("events_sha256")), "Missing raw event identity")
    if receipt["status"] == "completed":
        require(receipt.get("error") is None and type(receipt.get("returncode")) is int and receipt["returncode"] == 0
                and receipt.get("unapproved_events") == [] and receipt.get("event_errors") == []
                and type(receipt.get("usage")) is dict and bool(receipt["usage"]), "Completed receipt has transport/access failures")
    request = audit.load(public_prefix + "request.json")
    require(set(request) == {"prompt", "configuration"} and type(request.get("prompt")) is str
            and request.get("configuration") == checkpoint["provider_configuration"], "Published request differs")
    prompt_bytes = request["prompt"].encode("utf-8")
    require(hashlib.sha256(prompt_bytes).hexdigest() == assignment["prompt_sha256"]
            and len(prompt_bytes) == assignment["prompt_bytes"], "Published prompt differs from frozen bytes")
    require(canonical(audit.load(public_prefix + "response-schema.json")) == canonical(schema), "Published response schema differs")
    publication = audit.load(public_prefix + "event-publication.json")
    require(publication.get("schema") == "wsp-public-event-derivative/1"
            and publication.get("raw_events_sha256") == receipt["events_sha256"]
            and publication.get("public_events_sha256") == evidence[public_prefix + "events.jsonl"]["sha256"]
            and publication.get("raw_bytes_published") is False
            and publication.get("transform") == "JSON_reserialization_and_user_profile_prefix_redaction"
            and type(publication.get("redactions")) is int and publication["redactions"] >= 0,
            "Published event derivative binding differs")
    for name in ("request.json", "response-schema.json", "receipt.json"):
        record = evidence[public_prefix + name]
        audit.observe(safe_path(audit.raw, private_prefix + name), record["sha256"], record["bytes"])
    audit.observe(safe_path(audit.raw, private_prefix + "events.jsonl"), receipt["events_sha256"])


def check_selection(audit, item, assignment, schema, parent_hash, account, provider):
    assignment_id = assignment["assignment_id"]
    prefix = SELECTIONS + "/" + assignment_id
    selected = signed(audit.load(prefix + "/selection.json"), "wsp-live-selection/1")
    require(set(selected) == SELECTION_KEYS and selected["assignment_id"] == assignment_id
            and selected["sha256"] == account["selection_sha256"]
            and selected["terminal_status"] == account["terminal_status"]
            and selected["evaluation_started"] is False, "Gate/selection identity or phase differs")
    terminal, status = selected["terminal_status"], selected["provider_status"]
    require(terminal in TERMINAL and ((status == "completed" and terminal in {"sealed", "abstained", "invalid"})
            or (status in {"failed", "timeout", "interrupted"} and terminal == status)), "Illegal terminal accounting transition")
    require(type(selected["no_op"]) is bool and type(selected["answer_valid"]) is bool
            and selected["parent_sha256"] == parent_hash and is_hash(selected["candidate_sha256"]),
            "Selection parent/flag identity differs")
    evidence = records_table(selected["evidence_files"], prefix)
    actual = tree_files(audit.root, prefix) - {prefix + "/selection.json"}
    require(actual == set(evidence), "Selection evidence inventory is incomplete")
    allowed = {prefix + "/invocation-checkpoint.json", prefix + "/transport-interruption.json"}
    allowed.update(prefix + "/" + name for name in PROGRAM_NAMES)
    allowed.update(prefix + "/provider/" + name for name in PROVIDER_NAMES)
    require(set(evidence) <= allowed, "Unexpected/publicly unsafe selection evidence file")
    for relative, record in evidence.items():
        audit.public(relative, record["sha256"], record["bytes"])
    require(prefix + "/invocation-checkpoint.json" in evidence, "Invocation checkpoint is absent from evidence")
    checkpoint = signed(audit.load(prefix + "/invocation-checkpoint.json"), "wsp-live-invocation-checkpoint/1")
    require(checkpoint["sha256"] == selected["invocation_checkpoint_sha256"]
            and checkpoint.get("assignment_id") == assignment_id
            and checkpoint.get("assignment_sha256") == item["assignment_sha256"]
            and checkpoint.get("prepared_lock_sha256") == LOCK_CANONICAL
            and checkpoint.get("prompt_sha256") == assignment["prompt_sha256"]
            and checkpoint.get("schema_sha256") == digest(schema)
            and checkpoint.get("provider_configuration") == provider
            and checkpoint.get("phase") == "call_may_start", "Invocation checkpoint binding differs")
    check_provider(audit, prefix, assignment, schema, checkpoint, selected, evidence)
    if prefix + "/transport-interruption.json" in evidence:
        interruption = signed(audit.load(prefix + "/transport-interruption.json"))
        require(interruption.get("phase") == "transport_or_publication_interrupted"
                and interruption.get("retry_performed") is False and type(interruption.get("receipt_exists")) is bool,
                "Transport interruption record differs")
    if terminal == "sealed":
        require({prefix + "/" + name for name in PROGRAM_NAMES} <= set(evidence), "Sealed program evidence is incomplete")
        program = audit.load(prefix + "/program.json")
        candidate = audit.load(prefix + "/candidate.json")
        rollback = audit.load(prefix + "/rollback.json")
        proposal = audit.load(prefix + "/proposal.json")
        lowering = signed(audit.load(prefix + "/lowering.json"), "wsp-lowering/1")
        require(program.get("schema") == "wsp-program/1" and program.get("base_sha256") == parent_hash
                and digest(program) == selected["program_sha256"] and digest(candidate) == selected["candidate_sha256"]
                and candidate.get("schema") == "wsp-tensors/1" and candidate.get("dtype") == "float64"
                and rollback.get("schema") == "wsp-program/1" and rollback.get("base_sha256") == selected["candidate_sha256"]
                and proposal.get("schema") == "wsp-structured-program/1" and proposal.get("base_sha256") == parent_hash,
                "Selected artifact/program identities differ")
        require(lowering.get("proposal_sha256") == digest(proposal) and lowering.get("base_sha256") == parent_hash
                and lowering.get("candidate_sha256") == selected["candidate_sha256"]
                and lowering.get("program_sha256") == selected["program_sha256"]
                and lowering.get("behavioral_status") == "not_checked", "Lowering evidence identities differ")
        require(selected["no_op"] is (selected["candidate_sha256"] == parent_hash), "No-op identity flag differs")
    else:
        require(selected["program_sha256"] is None and selected["candidate_sha256"] == parent_hash
                and selected["no_op"] is False and not ({prefix + "/" + name for name in PROGRAM_NAMES} & set(evidence)),
                "Nonsealed assignment does not preserve its unchanged parent")
    if terminal in {"sealed", "abstained"}:
        require(type(selected["response"]) is dict and selected["response"].get("abstain") is (terminal == "abstained")
                and selected["answer_valid"] is True and selected["answer_validation_error_type"] is None
                and selected["validation_error_type"] is None, "Successful selection response accounting differs")
    elif terminal == "invalid":
        require(type(selected["validation_error_type"]) is str and bool(selected["validation_error_type"]),
                "Invalid selection lacks a validation failure")
    else:
        require(selected["answer_valid"] is False and selected["answer_validation_error_type"] is None
                and selected["validation_error_type"] is None, "Failed transport claims a validated answer")
    require(selected["interrupted_transport_termination_attested"] is (False if status == "interrupted" else None),
            "Interrupted transport attestation differs from frozen semantics")
    return terminal, set(evidence) | {prefix + "/selection.json"}


def authenticate(root, raw, expected_gate=None):
    root, raw = Path(root).resolve(), Path(raw).resolve()
    require(root.is_dir() and raw.is_dir() and not raw.is_relative_to(root) and not root.is_relative_to(raw),
            "Public checkout and private transport must be separate existing directories")
    assert_evaluation_unopened(root)
    audit = Audit(root, raw)
    lock = signed(audit.load(LOCK, LOCK_RAW), "wsp-live-preparation-lock/1")
    require(lock["sha256"] == LOCK_CANONICAL, "Canonical preparation lock differs from supplied anchor")
    gate = signed(audit.load(GATE), "wsp-live-global-selection-seal/1")
    require(expected_gate is None or (is_hash(expected_gate) and gate["sha256"] == expected_gate),
            "Gate differs from separately supplied canonical anchor")
    require(set(gate) == GATE_KEYS and gate["phase"] == "selection_complete_before_evaluation"
            and gate["selection_frozen"] is True and gate["all_assigned_accounted"] is True
            and gate["all_terminal"] is True and gate["evaluation_started"] is False
            and gate["studies"] == ["blind-diagnosis", "weight-access"]
            and gate["prepared_lock_path"] == LOCK and gate["prepared_lock_sha256"] == LOCK_RAW
            and gate["protocol_sha256"] == PROTOCOL_HASHES
            and type(gate["assigned_total"]) is int and gate["assigned_total"] == 224
            and type(gate["terminal_total"]) is int and gate["terminal_total"] == 224,
            "Global gate contract differs from frozen anchors")
    full_assignments, prepared, sources = check_preparation(audit, lock)
    accounting = gate["accounting"]
    require(type(accounting) is list and len(accounting) == 224, "Gate must account for all 224 assignments")
    require(all(type(row) is dict and set(row) == {"assignment_id", "selection_sha256", "terminal_status"}
                and is_hash(row["selection_sha256"]) and row["terminal_status"] in TERMINAL for row in accounting),
            "Malformed or nonterminal gate accounting")
    require([row["assignment_id"] for row in accounting] == list(full_assignments)
            and len({row["assignment_id"] for row in accounting}) == 224, "Gate IDs are missing, duplicated, reordered, or unassigned")
    statuses, selection_files = Counter(), set()
    for account in accounting:
        item, assignment, schema, parent_hash = full_assignments[account["assignment_id"]]
        terminal, files = check_selection(audit, item, assignment, schema, parent_hash, account, lock["provider"])
        statuses[terminal] += 1
        selection_files.update(files)
    require(tree_files(root, SELECTIONS) == selection_files, "Selection tree contains unassigned files")
    records = audit.finish()
    require(tree_files(root, PREPARED) == set(prepared)
            and tree_files(root, PROTOCOL + "/source-snapshot") == set(sources)
            and tree_files(root, SELECTIONS) == selection_files, "Evidence inventory changed during authentication")
    assert_evaluation_unopened(root)
    unopened = sum(safe_path(root, relative) not in audit.observed for relative in prepared)
    require(unopened == 150, "Unexpected unopened prepared-input inventory")
    return {
        "schema": "wsp-independent-live-gate-authentication/1", "ok": True,
        "authenticated_utc": datetime.now(timezone.utc).isoformat(),
        "validator_sha256": file_hash(Path(__file__).resolve())[0],
        "prepared_lock_canonical_sha256": LOCK_CANONICAL, "prepared_lock_raw_sha256": LOCK_RAW,
        "gate_canonical_sha256": gate["sha256"], "gate_raw_sha256": audit.observed[safe_path(root, GATE)][0],
        "protocol_sha256": PROTOCOL_HASHES, "assigned_total": 224, "terminal_total": 224,
        "distinct_assigned_ids": 224, "distinct_terminal_ids": 224, "terminal_counts": dict(sorted(statuses.items())),
        "source_files_checked": len(sources), "selection_files_checked": len(selection_files),
        "hashed_file_count": len(records), "hashed_file_inventory_sha256": digest(records),
        "private_transport_files_checked": sum(item["scope"] == "private" for item in records),
        "prepared_files_membership_and_sizes_checked": len(prepared), "unopened_prepared_files": unopened,
        "unopened_prepared_scope": "30 training files and 120 evaluator-only injection ledgers; frozen validate-gate checks their content hashes",
        "evaluation_started": False, "evaluation_directory_absent": True, "replay_directory_absent": True,
        "scope": "Independent identities, input bindings and terminal accounting; no semantic replay, utility scoring, or evaluator data access",
        "interrupted_transport_scope": "An interrupted selection does not attest that an uncertain external transport terminated",
    }


def main():
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=workspace / "exports/weight-space-programming-framework-public")
    parser.add_argument("--raw-root", type=Path, default=workspace / ".runs/wsp-live-2026-11-08")
    parser.add_argument("--expected-gate-sha256", help="Optional independently recorded canonical gate identity")
    args = parser.parse_args()
    try:
        report = authenticate(args.root, args.raw_root, args.expected_gate_sha256)
    except (Rejected, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        # Never echo provider payloads, private event contents, or raw transport errors.
        message = str(error) if isinstance(error, Rejected) else "Evidence is missing, malformed, or unreadable (" + type(error).__name__ + ")"
        print(json.dumps({"ok": False, "error": message}, ensure_ascii=False))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
