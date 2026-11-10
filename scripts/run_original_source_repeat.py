#!/usr/bin/env python3
"""Repeat the released exp1 reading task with separate reproduction tiers.

Historical scorer replay and arithmetic correspondence are offline. Fresh model
calls require the main study's externally identified selection seal. No retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.live_provider import CodexProvider, parse_events, publish_attempt, write_json
from rlcd.weight_space.model import digest
from rlcd.weight_space.sequential import sequential_artifact, sequential_forward
from scripts.check_weight_space_bridge import file_hash, python_runtime_identity

EXPERIMENT = ROOT / "experiments/research/original-source-repeat"
ARCHIVE = ROOT / "artifacts/research/original-source-repeat"
SOURCE = ARCHIVE / "source"
ORIGINAL = SOURCE / "exp1_which_game"
LIVE = ARCHIVE / "live-20261108-001"
ACQUISITION_SHA256 = "7b9fb07e48225e40246fde2e3cd79869a8320ef3ca63341935feb08e4d05d606"
COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
POLICIES = (("a9f535cc", "breakout"), ("dfc6cb65", "pong"),
            ("bdf99279", "pacman"), ("0bbcc3ea", "freeway"))
LAYERS = (("dense1", (500, 32)), ("dense2", (32, 32)), ("logits", (32, 5)))
SECTIONS = (
    (2, 3, 503, "W1 (layer 1 weights), shape 500 x 32: one line per row (row i = input i), 32 values per line", (500, 32)),
    (504, 505, 506, "b1 (layer 1 biases), shape 32:", (32,)),
    (507, 508, 540, "W2 (layer 2 weights), shape 32 x 32: one line per row", (32, 32)),
    (541, 542, 543, "b2 (layer 2 biases), shape 32:", (32,)),
    (544, 545, 577, "W3 (output layer weights), shape 32 x 5: one line per row", (32, 5)),
    (578, 579, 580, "b3 (output layer biases), shape 5:", (5,)),
)
ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
                 "required": ["answer"], "additionalProperties": False}


def read_json(path: Path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def sealed(value: dict[str, Any]) -> dict[str, Any]:
    return value | {"sha256": digest(value)}


def verify_seal(value: Any, expected: str, schema: str) -> None:
    if (type(value) is not dict or value.get("schema") != schema or value.get("sha256") != expected
            or not re.fullmatch(r"[a-f0-9]{64}", expected)
            or digest({k: v for k, v in value.items() if k != "sha256"}) != expected):
        raise ValueError("Receipt differs from its externally expected identity: " + schema)


def inventory(directory: Path, *, exclude: str | None = None) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): file_hash(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.relative_to(directory).as_posix() != exclude}


def verify_inventory(directory: Path, expected: dict[str, str], *, exclude: str | None = None) -> None:
    for name in expected:
        if not (directory / name).resolve().is_relative_to(directory.resolve()):
            raise ValueError("Inventory path escapes its archive")
    if inventory(directory, exclude=exclude) != expected:
        raise ValueError("Archive contains missing, changed or unlisted files")


def check_acquisition() -> dict[str, Any]:
    acquisition = read_json(ARCHIVE / "acquisition.json")
    verify_seal(acquisition, ACQUISITION_SHA256, "wsp-original-source-acquisition/1")
    rows = acquisition["assets"]
    expected = {r["path"]: r["sha256"] for r in rows}
    if (acquisition["repository"] != "RikiyaT/lm_understand-logs" or acquisition["commit"] != COMMIT
            or acquisition["tree_truncated"] is not False or len(rows) != 44 or len(expected) != 44
            or any(r["acquired"] is not True for r in rows)):
        raise ValueError("The complete pinned exp1 acquisition is required")
    verify_inventory(SOURCE, expected)
    for row in rows:
        raw = (SOURCE / row["path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if len(raw) != row["bytes"] or blob != row["git_blob_sha1"]:
            raise ValueError("Source bytes differ from the pinned Git blob")
    return acquisition


def actor_arrays(text: str, source_arrays: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    """Validate every decimal string against its original float32 coordinate."""
    lines = text.splitlines()
    if len(lines) != 580 or any(lines[i] for i in (1, 503, 506, 540, 543, 577)):
        raise ValueError("Unexpected actor text layout")
    arrays, signed_zero, normalized_zero = [], 0, 0
    for (heading, start, stop, expected_heading, shape), raw in zip(SECTIONS, source_arrays, strict=True):
        if lines[heading] != expected_heading or raw.shape != shape or raw.dtype != np.dtype("float32"):
            raise ValueError("Unexpected actor tensor shape, dtype or heading")
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite actor coordinates")
        rows = [line.split() for line in lines[start:stop]]
        width = shape[-1]
        if len(rows) != (shape[0] if len(shape) == 2 else 1) or any(len(row) != width for row in rows):
            raise ValueError("Actor decimal rows do not match their declared shape")
        tokens = [token for row in rows for token in row]
        for token, original in zip(tokens, raw.flat, strict=True):
            expected = format(float(original), ".3f")
            normalized_zero += expected == "-0.000"
            if expected == "-0.000":
                expected = "0.000"
            if not re.fullmatch(r"-?(?:0|[1-9]\d*)\.\d{3}", token) or token != expected:
                raise ValueError("Rounded actor text differs from its original coordinate")
            signed_zero += token == "-0.000"
        arrays.append(np.asarray([float(t) for t in tokens], dtype=np.float64).reshape(shape))
    return arrays, signed_zero, normalized_zero


def reconstruct(policy_id: str) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    with np.load(ORIGINAL / f"networks/{policy_id}.npz", allow_pickle=False) as original:
        names = [f"actor/{name}/{part}" for name, _ in LAYERS for part in ("kernel", "bias")]
        if {key for key in original.files if key.startswith("actor/")} != set(names):
            raise ValueError("Unexpected captured actor tensors")
        full = [np.array(original[name], copy=True) for name in names]
    text = (ORIGINAL / f"networks/{policy_id}_actor.txt").read_text(encoding="utf-8")
    rounded, signed_zero, normalized_zero = actor_arrays(text, full)
    coordinate_count = sum(a.size for a in full)
    if coordinate_count != 17253:
        raise ValueError("Actor coordinate inventory is incomplete")
    return full, rounded, {"policy_id": policy_id, "actor_coordinates_checked": coordinate_count,
                          "critic_supplied_to_reader": False, "signed_zero_decimal_tokens": signed_zero,
                          "negative_rounded_zero_coordinates_normalized_in_source": normalized_zero,
                          "rounding": "Every source float32 coordinate matched by Python format(value, '.3f'), with released -0.000 normalized to 0.000",
                          "max_abs_rounding_difference": max(float(np.max(np.abs(a.astype(np.float64) - b)))
                                                              for a, b in zip(full, rounded, strict=True))}


def converted(arrays: list[np.ndarray]) -> TensorArtifact:
    artifact = sequential_artifact([arrays[i].T for i in (0, 2, 4)],
        [arrays[i] for i in (1, 3, 5)], ["tanh", "tanh", "identity"],
        input_names=[f"row.{r}.column.{c}.channel.{k}" for r in range(10) for c in range(10) for k in range(5)],
        output_names=["no-op", "up", "down", "left", "right"])
    # Validate the captured output/input coordinates without using the conversion transpose.
    tensors = artifact.tensors
    for layer in range(3):
        original, bias = arrays[2 * layer], arrays[2 * layer + 1]
        for out in range(original.shape[1]):
            if float(tensors[f"layers.{layer}.bias"][out]) != float(bias[out]):
                raise ValueError("Captured actor bias differs")
            for inp in range(original.shape[0]):
                if float(tensors[f"layers.{layer}.weight"][out, inp]) != float(original[inp, out]):
                    raise ValueError("Captured actor orientation or coefficient differs")
    return artifact


def synthetic_inputs() -> list[list[float]]:
    # ponytail: six fixed binary grids check arithmetic only; environment replay needs original game code.
    return [[float(predicate(r, c, k)) for r in range(10) for c in range(10) for k in range(5)]
            for predicate in (lambda r, c, k: False, lambda r, c, k: True,
                lambda r, c, k: k == 3 and (r in (0, 9) or c in (0, 9)),
                lambda r, c, k: r == 8 and c == 4 and k == 0,
                lambda r, c, k: k == 1 and r in (2, 4, 6) and c % 3 == 1,
                lambda r, c, k: ((r * 10 + c) * 5 + k) % 2 == 0)]


def arithmetic(arrays: list[np.ndarray], artifact: TensorArtifact) -> dict[str, Any]:
    inputs = synthetic_inputs()
    expected = []
    for vector in inputs:
        values = vector
        for layer in range(3):
            weight, bias = arrays[2 * layer], arrays[2 * layer + 1]
            values = [math.fsum([float(bias[o])] + [values[i] * float(weight[i, o])
                      for i in range(weight.shape[0])]) for o in range(weight.shape[1])]
            if layer < 2:
                values = [math.tanh(v) for v in values]
        expected.append(values)
    observed = sequential_forward(artifact, inputs)
    error = float(np.max(np.abs(observed - np.asarray(expected))))
    if error > 1e-10:
        raise ValueError("Independent actor arithmetic differs beyond the fixed tolerance")
    return {"artifact_sha256": artifact.sha256, "inputs": inputs, "source_orientation_logits": expected,
            "framework_logits": observed.tolist(), "max_abs_error": error, "absolute_tolerance": 1e-10,
            "reference": "Source input/output orientation; Python math.fsum and math.tanh",
            "scope": "Six synthetic binary grids, not original trajectories, reward reproduction or a float theorem"}


def answer_field(text: str, label: str) -> str:
    found = []
    lines = [re.sub(r"^\s*#{1,6}\s*", "", line).replace("**", "").replace("__", "").replace("`", "").strip()
             for line in text.splitlines()]
    for i, line in enumerate(lines):
        match = re.fullmatch(label + r"(?:\s*:\s*(.*))?", line, re.IGNORECASE)
        if match:
            value = (match.group(1) or "").strip()
            if not value:
                value = next((item for item in lines[i + 1:] if item), "")
            found.append(value)
    if len(found) != 1 or not found[0] or re.match(r"^(?:Game|Confidence|Description|Reasoning|Observations)\s*:?$", found[0], re.I):
        raise ValueError("Missing or ambiguous " + label + " field")
    return found[0]


def score(text: str, truth: str) -> dict[str, Any]:
    warnings, game_line, confidence = [], None, None
    try:
        game_line = answer_field(text, "Game")
        keywords = {"breakout": r"\bbreakout\b", "pong": r"\bpong\b",
                    "pacman": r"\bpac[ -]?man\b", "freeway": r"\b(?:freeway|frogger)\b"}
        matches = [game for game, pattern in keywords.items() if re.search(pattern, game_line, re.I)]
        guess = matches[0] if len(matches) == 1 else None
        if guess is None:
            warnings.append("unmapped_or_ambiguous_game")
    except ValueError:
        guess = None
        warnings.append("missing_or_ambiguous_game_field")
    try:
        value = answer_field(text, "Confidence")
        if not re.fullmatch(r"(?:100|\d{1,2})\s*%", value):
            raise ValueError("Invalid confidence")
        confidence = int(value.replace("%", "").strip())
    except ValueError:
        warnings.append("missing_or_invalid_confidence")
    return {"true_game": truth, "guess": guess, "game_field": game_line, "correct": guess == truth,
            "confidence_percent": confidence, "warnings": warnings,
            "scorer": "Reconstructed keyword rule on Game field only; Frogger maps to Freeway; ambiguity abstains"}


def historical() -> dict[str, Any]:
    key = read_json(ORIGINAL / "eval/answer_key.json")
    recorded = {item["id"]: item for item in read_json(ORIGINAL / "eval/scores.json")}
    rows, question = [], None
    description = (ORIGINAL / "networks/description.txt").read_bytes()
    for policy_id, truth in POLICIES:
        folder = ORIGINAL / "runs" / truth
        prompt, actor = (folder / "prompt.txt").read_bytes(), (ORIGINAL / f"networks/{policy_id}_actor.txt").read_bytes()
        if not prompt.startswith(description) or not prompt.endswith(actor):
            raise ValueError("Original prompt differs from released description/actor text")
        current_question = prompt[len(description):-len(actor)]
        if question is None:
            question = current_question
        if current_question != question or (folder / "instructions.md").read_text(encoding="utf-8").strip() != "Answer the user's request.":
            raise ValueError("Original prompt question or instructions differ")
        answer = (folder / "answer.md").read_text(encoding="utf-8")
        replayed = score(answer, truth)
        run = read_json(folder / "run.json")
        event_rows = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        messages = [row["item"]["text"] for row in event_rows if row.get("type") == "item.completed"
                    and row.get("item", {}).get("type") == "agent_message"]
        tools = [row for row in event_rows if row.get("type", "").startswith("item.")
                 and row.get("item", {}).get("type") not in ("reasoning", "agent_message", "error")]
        usages = [row["usage"] for row in event_rows if row.get("type") == "turn.completed"]
        source_score = recorded[policy_id]
        if (key[policy_id]["game"] != truth or replayed["guess"] != source_score["guess"]
                or replayed["correct"] != source_score["correct"]
                or replayed["confidence_percent"] != source_score["confidence"] or tools
                or len(messages) != 1 or messages[0].strip() != answer.strip()
                or len(usages) != 1 or usages[0] != source_score["usage"] or usages[0] != run["audit"]["usage"]):
            raise ValueError("Historical scorer/event replay differs from released records")
        rows.append({"policy_id": policy_id, "seed": key[policy_id]["seed"], "score": replayed,
                     "prompt_sha256": hashlib.sha256(prompt).hexdigest(), "prompt_bytes": len(prompt),
                     "answer_sha256": file_hash(folder / "answer.md"), "recorded_usage": usages[0],
                     "recorded_wall_seconds": run["wall_s"], "tool_events_observed": len(tools),
                     "event_final_matches_answer_ignoring_outer_whitespace": True})
    return {"schema": "wsp-original-historical-replay/1", "rows": rows,
            "correct": sum(r["score"]["correct"] for r in rows), "assigned": len(rows),
            "scope": "Historical released replies only, not fresh model performance or independently audited training",
            "original_executable_scorer_available": False, "fresh_provider_calls": 0}


def project_closure() -> dict[str, str]:
    files = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT):
                files.add(path)
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(files)}


def runtime() -> dict[str, Any]:
    return python_runtime_identity() | {"source_package": "source-freeze/rlcd"}


def prepare() -> dict[str, Any]:
    check_acquisition()
    historical_result = historical()
    EXPERIMENT.mkdir(parents=True, exist_ok=False)
    conversions, numerical = [], []
    for policy_id, _ in POLICIES:
        full, rounded, audit = reconstruct(policy_id)
        for kind, arrays in (("full-precision", full), ("rounded", rounded)):
            artifact = converted(arrays)
            write_json(EXPERIMENT / f"converted/{policy_id}/{kind}-artifact.json", artifact.to_dict())
            numerical.append({"policy_id": policy_id, "representation": kind, "result": arithmetic(arrays, artifact)})
            audit[kind + "_artifact_sha256"] = artifact.sha256
        conversions.append(audit)
    protocol = {"schema": "wsp-original-source-repeat-protocol/1", "source_repository": "RikiyaT/lm_understand-logs",
        "source_commit": COMMIT, "acquisition_sha256": ACQUISITION_SHA256,
        "policies": [{"policy_id": policy_id, "truth": truth} for policy_id, truth in POLICIES],
        "selection_basis": "All four exp1 checkpoints with complete released assets; original checkpoints are best of three training seeds",
        "fresh_provider": CodexProvider().configuration(), "fresh_calls": 4, "retry_rule": "none",
        "fresh_bundle_path": LIVE.relative_to(ROOT).as_posix(), "response_schema": ANSWER_SCHEMA,
        "prompt_rule": "Exact original prompt bytes decoded as UTF-8; no labels, historical answers or answer keys added",
        "fresh_adaptations": ["Requested model gpt-6.1-sol, low effort instead of historical gpt-6-astra, high effort",
                              "Original Markdown reply is carried inside a required JSON answer string",
                              "Current CLI system scaffolding is not exported or asserted identical to the original instructions"],
        "scoring_rule": "Game-field keyword reconstruction; ambiguity abstains; confidence is diagnostic, not calibrated",
        "success_rule": "Completed transport and one unambiguous correct Game field; failures remain in denominator of four",
        "arithmetic_absolute_tolerance": 1e-10, "synthetic_grid_count": 6,
        "framework_conversion": "Actor only; source input/output matrices explicitly transposed; exact float32 values lifted to binary64 and rounded decimal values separately captured; signed zero canonicalized",
        "certificate_scope": "Tanh execution only; framework real-drift and Lean bridge do not certify this actor",
        "known_limits": ["Public-source/model familiarity cannot be ruled out", "Four selected networks do not estimate general weight-reading ability",
                         "No retraining or original game trajectory/reward repetition", "No mechanistic truth test of the prose explanations"]}
    write_json(EXPERIMENT / "protocol.json", sealed(protocol))
    write_json(EXPERIMENT / "conversion-audit.json", {"schema": "wsp-original-actor-conversion/1", "rows": conversions})
    write_json(EXPERIMENT / "arithmetic-correspondence.json", {"schema": "wsp-original-arithmetic-correspondence/1", "rows": numerical})
    write_json(EXPERIMENT / "historical-replay.json", historical_result)
    write_json(EXPERIMENT / "runtime.json", runtime())
    closure = project_closure()
    for name in closure:
        target = EXPERIMENT / "source-freeze" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    lock = sealed({"schema": "wsp-original-source-repeat-lock/1", "protocol_sha256": digest(protocol),
                   "acquisition_sha256": ACQUISITION_SHA256, "acquisition_file_sha256": file_hash(ARCHIVE / "acquisition.json"),
                   "project_source_files": closure, "files_sha256": inventory(EXPERIMENT),
                   "scope": "Protocol, complete converted actors, historical replay, independent synthetic arithmetic, actual imported project source closure and runtime identities frozen before fresh calls"})
    write_json(EXPERIMENT / "protocol-lock.json", lock)
    return {"status": "prepared", "lock_sha256": lock["sha256"], "protocol_sha256": lock["protocol_sha256"],
            "historical_correct": historical_result["correct"], "fresh_provider_calls": 0,
            "project_sources": len(closure), "lock_path": "experiments/research/original-source-repeat/protocol-lock.json"}


def validate_lock(expected: str) -> dict[str, Any]:
    lock = read_json(EXPERIMENT / "protocol-lock.json")
    verify_seal(lock, expected, "wsp-original-source-repeat-lock/1")
    verify_inventory(EXPERIMENT, lock["files_sha256"], exclude="protocol-lock.json")
    check_acquisition()
    if file_hash(ARCHIVE / "acquisition.json") != lock["acquisition_file_sha256"] or project_closure() != lock["project_source_files"]:
        raise ValueError("Current source acquisition or project runtime closure differs from the lock")
    if runtime() != read_json(EXPERIMENT / "runtime.json"):
        raise ValueError("Python interpreter or NumPy runtime identity differs from the lock")
    return lock


def replay_historical(expected: str) -> dict[str, Any]:
    lock = validate_lock(expected)
    if historical() != read_json(EXPERIMENT / "historical-replay.json"):
        raise ValueError("Historical replay differs from the locked result")
    audits, numerical = [], []
    for policy_id, _ in POLICIES:
        full, rounded, audit = reconstruct(policy_id)
        for kind, arrays in (("full-precision", full), ("rounded", rounded)):
            artifact = converted(arrays)
            if artifact.to_dict() != read_json(EXPERIMENT / f"converted/{policy_id}/{kind}-artifact.json"):
                raise ValueError("Regenerated actor differs from the locked artifact")
            audit[kind + "_artifact_sha256"] = artifact.sha256
            numerical.append({"policy_id": policy_id, "representation": kind, "result": arithmetic(arrays, artifact)})
        audits.append(audit)
    if (audits != read_json(EXPERIMENT / "conversion-audit.json")["rows"]
            or numerical != read_json(EXPERIMENT / "arithmetic-correspondence.json")["rows"]):
        raise ValueError("Conversion/arithmetic replay differs from the locked results")
    return {"schema": "wsp-original-offline-replay/1", "status": "pass", "lock_sha256": lock["sha256"],
            "historical_correct": 4, "historical_assigned": 4, "complete_actor_coordinates_per_policy": 17253,
            "fresh_provider_calls": 0, "original_environment_reward_repeated": False, "lean_checked": False}


def global_gate(path: Path, expected: str) -> dict[str, Any]:
    gate = read_json(path)
    verify_seal(gate, expected, "wsp-live-global-selection-seal/1")
    if (gate.get("phase") != "selection_complete_before_evaluation" or gate.get("assigned_total") != 224
            or gate.get("terminal_total") != 224 or gate.get("all_terminal") is not True
            or gate.get("evaluation_started") is not False or gate.get("selection_frozen") is not True
            or gate.get("all_assigned_accounted") is not True
            or gate.get("studies") != ["blind-diagnosis", "weight-access"]
            or gate.get("prepared_lock_path") != "experiments/research/live-studies/preparation-lock.json"):
        raise ValueError("Main-study selections are not completely sealed")
    prep_path = (ROOT / gate["prepared_lock_path"]).resolve()
    if not prep_path.is_relative_to(ROOT.resolve()) or file_hash(prep_path) != gate["prepared_lock_sha256"]:
        raise ValueError("Main preparation lock bytes differ from the selection seal")
    prep = read_json(prep_path)
    verify_seal(prep, prep["sha256"], "wsp-live-preparation-lock/1")
    assignment_ids = [row["assignment_id"] for row in prep["assignments"]]
    accounting = gate["accounting"]
    ids = [row["assignment_id"] for row in accounting]
    if (len(assignment_ids) != 224 or len(set(assignment_ids)) != 224 or len(ids) != 224
            or len(set(ids)) != 224 or set(ids) != set(assignment_ids)
            or gate["protocol_sha256"] != prep["protocol_sha256"]
            or any(type(row) is not dict or set(row) != {"assignment_id", "selection_sha256", "terminal_status"}
                   or row["terminal_status"] not in {"sealed", "abstained", "invalid", "failed", "timeout", "interrupted"}
                   or not re.fullmatch(r"[a-f0-9]{64}", row["selection_sha256"]) for row in accounting)):
        raise ValueError("Main assignment identities or terminal accounting differ")
    return {"gate_schema": gate["schema"], "gate_sha256": expected,
            "prepared_lock_sha256": gate["prepared_lock_sha256"], "main_evaluation_outcomes_inspected": False}


def live(expected: str, gate_path: Path, gate_sha256: str, private: Path) -> dict[str, Any]:
    lock = validate_lock(expected)
    binding = global_gate(gate_path, gate_sha256)
    private = private.resolve()
    if private.is_relative_to(ROOT):
        raise ValueError("Raw provider transport must stay outside the public repository")
    if LIVE.exists() or private.exists():
        raise ValueError("Both fresh-run destination directories must be new")
    LIVE.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    # Every assignment exists on disk before the first call, including a stopped run's remainder.
    rows = [{"policy_id": policy_id, "attempted": False, "provider_status": "unattempted",
             "valid_response": False, "attempt_sha256": None, "harness_error": None,
             "score": score("", truth)} for policy_id, truth in POLICIES]
    def persist() -> None:
        write_json(LIVE / "outcomes.json", {"schema": "wsp-original-live-outcomes/1", "rows": rows,
            "assigned": 4, "accounted": len(rows), "fresh_calls": sum(row["attempted"] for row in rows),
            "retry_rule": "none", "fresh_calls_scope": "provider.call invocations attempted, not independently attested model executions"})
    persist()
    provider = CodexProvider()
    for index, (policy_id, truth) in enumerate(POLICIES):
        receipt, phase = None, "prompt"
        try:
            prompt = (ORIGINAL / f"runs/{truth}/prompt.txt").read_bytes().decode("utf-8")
            phase = "provider"
            rows[index].update(attempted=True, provider_status="interrupted",
                               harness_error={"exception_type": "IncompleteAssignment", "phase": phase})
            persist()
            receipt = provider.call(prompt, directory=private / policy_id, cwd=private / (policy_id + "-cwd"), schema=ANSWER_SCHEMA)
            phase = "publication"
            publish_attempt(private / policy_id, LIVE / policy_id)
            phase = "scoring"
            response = receipt["response"]
            valid = (receipt["status"] == "completed" and type(response) is dict and set(response) == {"answer"}
                     and type(response["answer"]) is str)
            result = score(response["answer"] if valid else "", truth)
            result["correct"] = valid and result["correct"]
            rows[index] = {"policy_id": policy_id, "attempted": True, "provider_status": receipt["status"],
                           "valid_response": valid, "attempt_sha256": receipt["sha256"],
                           "harness_error": None, "score": result}
        except BaseException as error:
            # ponytail: stop after an unexpected harness failure; retain partial bytes, never retry.
            rows[index].update(provider_status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                               valid_response=False, attempt_sha256=receipt["sha256"] if receipt else None,
                               harness_error={"exception_type": type(error).__name__, "phase": phase},
                               score=score("", truth))
            if receipt is not None:
                write_json(LIVE / policy_id / "receipt.json", receipt)
            persist()
            break
        persist()
        print(json.dumps({"policy_id": policy_id, "provider_status": receipt["status"]}), flush=True)
    result = sealed({"schema": "wsp-original-live-repeat-bundle/1", "protocol_lock_sha256": lock["sha256"],
                     "global_selection_gate": binding, "assigned": 4, "accounted": 4,
                     "fresh_calls": sum(row["attempted"] for row in rows),
                     "correct": sum(r["score"]["correct"] for r in rows), "files_sha256": inventory(LIVE),
                     "scope": "Fresh adapted model task repetition, publicly replayable derivative receipts, no independent provider provenance or original training/environment reproduction"})
    write_json(LIVE / "manifest.json", result)
    return {"status": "stopped_all_four_accounted" if any(row["harness_error"] for row in rows) else "all_four_accounted",
            "bundle_sha256": result["sha256"], "correct": result["correct"], "assigned": 4,
            "fresh_calls": result["fresh_calls"]}


def replay_live(expected_lock: str, expected_bundle: str) -> dict[str, Any]:
    lock = validate_lock(expected_lock)
    bundle = read_json(LIVE / "manifest.json")
    verify_seal(bundle, expected_bundle, "wsp-original-live-repeat-bundle/1")
    verify_inventory(LIVE, bundle["files_sha256"], exclude="manifest.json")
    if bundle["protocol_lock_sha256"] != lock["sha256"]:
        raise ValueError("Fresh outcomes belong to a different protocol lock")
    outcomes = read_json(LIVE / "outcomes.json")
    if outcomes["assigned"] != 4 or outcomes["accounted"] != 4 or [r["policy_id"] for r in outcomes["rows"]] != [p for p, _ in POLICIES]:
        raise ValueError("Fresh assignment accounting is incomplete or changed")
    stopped = False
    for (policy_id, truth), row in zip(POLICIES, outcomes["rows"], strict=True):
        folder = LIVE / policy_id
        if type(row["attempted"]) is not bool:
            raise ValueError("Fresh attempt accounting must be explicit")
        if row["harness_error"] is not None or not row["attempted"]:
            if (row["score"] != score("", truth) or row["valid_response"] is not False
                    or (row["harness_error"] is None and not row["attempted"] and (not stopped or row["provider_status"] != "unattempted"
                         or row["attempt_sha256"] is not None or row["harness_error"] is not None or folder.exists()))
                    or (row["harness_error"] is not None and (stopped or row["provider_status"] not in {"failed", "interrupted"}
                         or type(row["harness_error"]) is not dict
                         or set(row["harness_error"]) != {"exception_type", "phase"}
                         or row["harness_error"]["phase"] not in {"prompt", "provider", "publication", "scoring"}
                         or row["attempted"] != (row["harness_error"]["phase"] != "prompt")
                         or type(row["harness_error"]["exception_type"]) is not str))):
                raise ValueError("Interrupted or unattempted assignment accounting differs")
            if row["attempt_sha256"] is not None:
                verify_seal(read_json(folder / "receipt.json"), row["attempt_sha256"], "wsp-live-provider-attempt/1")
            stopped = True
            continue
        if stopped:
            raise ValueError("A stopped run cannot contain later attempted assignments")
        receipt = read_json(folder / "receipt.json")
        verify_seal(receipt, row["attempt_sha256"], "wsp-live-provider-attempt/1")
        capture = receipt["invocation"].get("output_capture")
        worker_capture = capture.get("worker") if type(capture) is dict else None
        incomplete_capture = "output_capture" in receipt["invocation"] and (
            type(worker_capture) is not dict or worker_capture.get("complete") is not True)
        publication = read_json(folder / "event-publication.json")
        if (file_hash(folder / "events.jsonl") != publication["public_events_sha256"]
                or receipt["events_sha256"] != publication["raw_events_sha256"]
                or not (publication["transform"] == "JSON_reserialization_and_user_profile_prefix_redaction"
                        or incomplete_capture and publication["transform"] ==
                        "UTF8_replacement_then_JSON_reserialization_and_user_profile_prefix_redaction")):
            raise ValueError("Public event derivative or disclosed raw identity differs")
        response, usage, tools, errors, diagnostics = parse_events((folder / "events.jsonl").read_text(encoding="utf-8"))
        if incomplete_capture:
            response, usage = None, {}
        request = read_json(folder / "request.json")
        prompt = (ORIGINAL / f"runs/{truth}/prompt.txt").read_bytes()
        if (request["prompt"].encode("utf-8") != prompt or request["configuration"] != CodexProvider().configuration()
                or hashlib.sha256(prompt).hexdigest() != receipt["invocation"]["prompt_sha256"]
                or read_json(folder / "response-schema.json") != ANSWER_SCHEMA
                or receipt["invocation"]["schema_sha256"] != digest(ANSWER_SCHEMA)
                or response != receipt["response"] or usage != receipt["usage"] or tools != receipt["unapproved_events"]
                or errors != receipt["event_errors"] or len(diagnostics) != receipt["diagnostic_items"]):
            raise ValueError("Public invocation/response/event replay differs")
        valid = receipt["status"] == "completed" and type(response) is dict and set(response) == {"answer"} and type(response["answer"]) is str
        result = score(response["answer"] if valid else "", truth)
        result["correct"] = valid and result["correct"]
        if row != {"policy_id": policy_id, "attempted": True, "provider_status": receipt["status"], "valid_response": valid,
                   "attempt_sha256": receipt["sha256"], "harness_error": None, "score": result}:
            raise ValueError("Fresh scoring replay differs")
    if (bundle["correct"] != sum(row["score"]["correct"] for row in outcomes["rows"])
            or bundle["fresh_calls"] != outcomes["fresh_calls"]
            or outcomes["fresh_calls"] != sum(row["attempted"] for row in outcomes["rows"])):
        raise ValueError("Fresh result accounting differs")
    return {"status": "pass", "bundle_sha256": bundle["sha256"], "correct": bundle["correct"],
            "assigned": 4, "fresh_provider_calls": 0, "scope": "Public derivative replay; raw provider event bytes not published"}


def self_check() -> dict[str, Any]:
    from unittest.mock import patch
    assert score("Reasoning: Pong and Pac-Man.\n## Game\n**Breakout**\n## Confidence\n99%", "breakout")["correct"]
    assert score("Game: **Frogger**\nConfidence: 97%", "freeway")["correct"]
    assert not score("Game: Pong or Breakout\nConfidence: 99%", "pong")["correct"]
    assert not score("Game: Pong\nGame: Pong\nConfidence: 99%", "pong")["correct"]
    assert not score("Reasoning: Pong\nConfidence: 99%", "pong")["correct"]
    assert score("Game: Pac-Man\nConfidence: 101%", "pacman")["confidence_percent"] is None
    full, _, _ = reconstruct("a9f535cc")
    original = (ORIGINAL / "networks/a9f535cc_actor.txt").read_text(encoding="utf-8")
    try:
        actor_arrays(original.replace("0.054 -0.026", "0.055 -0.026", 1), full)
    except ValueError:
        pass
    else:
        raise AssertionError("Altered actor coordinate was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        (directory / "locked.txt").write_text("original", encoding="utf-8")
        expected = inventory(directory)
        (directory / "unlisted.txt").write_text("extra", encoding="utf-8")
        try:
            verify_inventory(directory, expected)
        except ValueError:
            pass
        else:
            raise AssertionError("Unlisted archive file was accepted")
        (directory / "unlisted.txt").unlink()
        (directory / "protocol-lock.json").write_text("excluded root receipt", encoding="utf-8")
        expected = inventory(directory, exclude="protocol-lock.json")
        (directory / "nested").mkdir()
        (directory / "nested/protocol-lock.json").write_text("unlisted nested receipt", encoding="utf-8")
        try:
            verify_inventory(directory, expected, exclude="protocol-lock.json")
        except ValueError:
            pass
        else:
            raise AssertionError("Unlisted nested file sharing the excluded receipt's name was accepted")
        module = sys.modules[__name__]
        private = directory / "existing-private"
        private.mkdir()
        fresh = directory / "collision-live"
        with (patch.object(module, "LIVE", fresh),
              patch.object(module, "validate_lock", return_value={"sha256": "a" * 64}),
              patch.object(module, "global_gate", return_value={"gate_sha256": "b" * 64})):
            try:
                live("a" * 64, directory / "unused-gate.json", "b" * 64, private)
            except ValueError:
                pass
            else:
                raise AssertionError("Existing private destination was accepted")
            assert not fresh.exists(), "Destination guard produced an orphan public directory"
        fresh = directory / "interrupted-live"
        with (patch.object(module, "LIVE", fresh),
              patch.object(module, "validate_lock", return_value={"sha256": "a" * 64}),
              patch.object(module, "global_gate", return_value={"gate_sha256": "b" * 64}),
              patch.object(module, "CodexProvider") as fake):
            fake.return_value.call.side_effect = KeyboardInterrupt
            result = live("a" * 64, directory / "unused-gate.json", "b" * 64, directory / "new-private")
            rows = read_json(fresh / "outcomes.json")["rows"]
            assert fake.return_value.call.call_count == 1
            assert [row["provider_status"] for row in rows] == ["interrupted", "unattempted", "unattempted", "unattempted"]
            assert result["assigned"] == 4 and result["fresh_calls"] == 1 and result["correct"] == 0
            assert replay_live("a" * 64, result["bundle_sha256"])["status"] == "pass"
        gate_root = directory / "gate-fixture"
        prep_name = "experiments/research/live-studies/preparation-lock.json"
        prep_path = gate_root / prep_name
        assignments = [{"assignment_id": f"fixture-{i}"} for i in range(224)]
        protocol_ids = {"blind-diagnosis": "c" * 64, "weight-access": "d" * 64}
        write_json(prep_path, sealed({"schema": "wsp-live-preparation-lock/1",
                                     "assignments": assignments, "protocol_sha256": protocol_ids}))
        gate_payload = {"schema": "wsp-live-global-selection-seal/1", "phase": "selection_complete_before_evaluation",
                       "assigned_total": 224, "terminal_total": 224, "all_terminal": True,
                       "evaluation_started": False, "selection_frozen": True, "all_assigned_accounted": True,
                       "studies": ["blind-diagnosis", "weight-access"], "protocol_sha256": protocol_ids,
                       "prepared_lock_path": prep_name, "prepared_lock_sha256": file_hash(prep_path),
                       "accounting": [{"assignment_id": row["assignment_id"], "selection_sha256": digest(row),
                                       "terminal_status": "sealed"} for row in assignments], "sealed_utc": "fixture"}
        gate_path = gate_root / "selection-seal.json"
        with patch.object(module, "ROOT", gate_root):
            gate = sealed(gate_payload)
            write_json(gate_path, gate)
            assert global_gate(gate_path, gate["sha256"])["main_evaluation_outcomes_inspected"] is False
            for mutation in ({"evaluation_started": True}, {"terminal_total": 223},
                             {"accounting": gate_payload["accounting"][:-1] + [gate_payload["accounting"][0]]}):
                changed = sealed(gate_payload | mutation)
                write_json(gate_path, changed)
                try:
                    global_gate(gate_path, changed["sha256"])
                except ValueError:
                    pass
                else:
                    raise AssertionError("Unreleased or duplicate-assignment main gate was accepted")
            write_json(gate_path, gate)
            prep_path.write_bytes(prep_path.read_bytes() + b"\n")
            try:
                global_gate(gate_path, gate["sha256"])
            except ValueError:
                pass
            else:
                raise AssertionError("Changed preparation-lock bytes were accepted")
    return {"status": "pass", "checks": "Game-field extraction, ambiguity/alternative rejection, confidence validation, actor coefficient mutation, closed inventory including nested excluded-name mutation, side-effect-free destination guards, interrupted four-assignment replay and externally identified global-gate release/accounting/byte mutations", "fresh_provider_calls": 0,
            "mocked_provider_invocations": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "replay-historical", "live", "replay-live", "self-check"))
    parser.add_argument("--expected-lock-sha256")
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--global-gate", type=Path)
    parser.add_argument("--expected-global-gate-sha256")
    parser.add_argument("--private-root", type=Path)
    parser.add_argument("--output", type=Path, help="Fresh JSON receipt path for offline replay only")
    args = parser.parse_args()
    if args.operation in ("replay-historical", "live", "replay-live") and not args.expected_lock_sha256:
        parser.error("An external expected protocol-lock identity is required")
    if args.operation == "live" and not all((args.global_gate, args.expected_global_gate_sha256, args.private_root)):
        parser.error("Fresh calls require an externally identified main selection gate and private transport directory")
    if args.operation == "replay-live" and not args.expected_bundle_sha256:
        parser.error("Fresh receipt replay requires an external expected bundle identity")
    if args.output and (args.operation not in ("replay-historical", "self-check") or args.output.exists()
            or any(args.output.resolve().is_relative_to(path.resolve()) for path in (EXPERIMENT, SOURCE, LIVE))):
        parser.error("Offline receipt output must be new and outside the locked preparation and source/live archives")
    try:
        if args.operation == "prepare":
            result = prepare()
        elif args.operation == "replay-historical":
            result = replay_historical(args.expected_lock_sha256)
        elif args.operation == "live":
            result = live(args.expected_lock_sha256, args.global_gate, args.expected_global_gate_sha256, args.private_root)
        elif args.operation == "replay-live":
            result = replay_live(args.expected_lock_sha256, args.expected_bundle_sha256)
        else:
            result = self_check()
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print("Original-source repeat failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
