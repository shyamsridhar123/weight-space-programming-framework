"""Freeze, select, seal, evaluate and replay matched live weight-space studies.

Only verified training files enter preparation. Native test execution and planted
interventions require the immutable global seal. Provider failures never retry.
The CLI restrictions are a data protocol, not a hermetic operating-system jail.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.broker import ProposerBroker, WeightView
from rlcd.weight_space.live_provider import CodexProvider, publish_attempt, write_json
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.program import Program, apply_program, difference, rollback
from rlcd.weight_space.research_tasks import (
    ARCHITECTURE, HEAD_BIAS, HEAD_WEIGHT, behavior_metrics, compile_arm,
    evaluate_head, load_lineage, public_record,
)
from rlcd.weight_space.structured import StructuredProgram, lower_structured, verify_lowering
from rlcd.weight_space.workflow import BudgetLimits, TaskSpec, verify_trace
from scripts.run_trained_research import mean_interval

PROTOCOL_DIR = Path("experiments/research/live-studies")
ARTIFACT_DIR = Path("artifacts/research/live-studies")
LOCK_PATH = PROTOCOL_DIR / "preparation-lock.json"
GATE_PATH = ARTIFACT_DIR / "global-selection-seal.json"
PREPARATION_JOURNAL = Path(".runs/live-preparation-promotion.json")
PROTOCOL_HASHES = {
    "weight-access-protocol.json": "d39e207131da8e751817ad657c29d09267c679fbe87c6b8cbdf5072c395d6937",
    "blind-diagnosis-protocol.json": "d2750eddc91a37597f26e450cf96fcf952a487cfeb2394fd9b1b3166ab42e938",
}
TRAINING_FILES = ("lineage.json", "head.json", "encoder.npz", "preprocessing.json", "features.npz")
VIEWS = ("none", "summary", "rounded", "subset", "exact")
REGIMES = ("brokered_analysis_only", "brokered_calibration_forward", "brokered_fitting_permitted")
LINEAGES = tuple(f"banking77-main-full-{seed}" for seed in range(7001, 7007))
TARGETS = ("pending_transfer", "card_payment_wrong_exchange_rate")
TERMINAL = frozenset(("sealed", "abstained", "invalid", "failed", "timeout", "interrupted"))
DTYPES = {HEAD_WEIGHT: "float32", HEAD_BIAS: "float32"}
WORKERS = 8
MATCH_ATOL, MATCH_RTOL = 1e-5, 1e-5
FIT_GRID = tuple({"ridge": ridge, "retention_weight": retention}
                 for ridge in (.01, .1) for retention in (1., 8., 64.))
MARGINS = {"target_gain": .10, "protected_harm": .01, "useful_coverage": .75}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(payload):
    payload = public_record(payload)
    return payload | {"sha256": digest(payload)}


def check_signed(value):
    if not isinstance(value, dict) or value.get("sha256") != digest(
            {k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Canonical record identity mismatch")
    return value


def file_record(root, path):
    path = Path(path)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path),
            "bytes": path.stat().st_size}


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes the repository")
    return path


def check_files(root, records):
    seen = set()
    for record in records:
        if record["path"] in seen:
            raise ValueError("Duplicate file identity")
        seen.add(record["path"])
        path = checked_path(root, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Frozen file identity mismatch: " + record["path"])


def verify_protocols(root):
    for name, expected in PROTOCOL_HASHES.items():
        if sha(root / PROTOCOL_DIR / name) != expected:
            raise ValueError("Original live protocol was changed")


def configure_runtime():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "device": "cpu", "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True}


def specification(study, lineage, target, view, regime="primary", repeat=0, mechanism=None):
    identity = {"study": study, "lineage_id": lineage, "target_name": target,
                "parameter_view": view, "regime": regime, "repeat": repeat, "mechanism": mechanism}
    return identity | {"assignment_id": digest(identity)[:24]}


def assigned_specs():
    assignments = []
    for lineage, target, view in itertools.product(LINEAGES, TARGETS, VIEWS):
        for repeat in range(2 if lineage in LINEAGES[:2] else 1):
            assignments.append(specification("weight-access", lineage, target, view, repeat=repeat))
    for lineage, target, view, regime in itertools.product(LINEAGES[:2], TARGETS, ("none", "exact"), REGIMES):
        assignments.append(specification("weight-access", lineage, target, view, regime=regime))
    for lineage, target, mechanism, view in itertools.product(
            LINEAGES, TARGETS, ("bias_shift", "feature_row_shift"), VIEWS):
        assignments.append(specification("blind-diagnosis", lineage, target, view, mechanism=mechanism))
    if len(assignments) != 224 or len({x["assignment_id"] for x in assignments}) != 224:
        raise ValueError("Frozen assignment construction changed")
    return assignments


def ordered_rows(bundle, role, label, target, maximum):
    labels, ids = bundle[role + "_labels"], bundle[role + "_ids"]
    rows = np.flatnonzero((labels == label) == target)
    rows = sorted(map(int, rows), key=lambda i: (str(ids[i]), i))[:maximum]
    if not rows:
        raise ValueError("No eligible training-role observations")
    return np.asarray(rows, dtype=np.int64)


def example_rows(bundle, role, rows):
    return [{"id": str(bundle[role + "_ids"][i]),
             "label_code": int(bundle[role + "_labels"][i]),
             "features": bundle[role + "_features"][i].tolist()} for i in rows]


def make_view(parent, mode):
    coordinates = ()
    if mode == "subset":
        rows, width = parent.tensors[HEAD_WEIGHT].shape
        coordinates = tuple((HEAD_WEIGHT, (row, column)) for row in range(rows) for column in range(0, width, 8))
        coordinates += tuple((HEAD_BIAS, (row,)) for row in range(rows))
    return WeightView(mode, decimals=3, coordinates=coordinates)


def make_broker(assignment, parent):
    return ProposerBroker(
        TaskSpec(assignment["assignment_id"], "edit", assignment["objective"],
                 ARCHITECTURE, parent.sha256),
        base=parent, view=make_view(parent, assignment["parameter_view"]),
        limits=BudgetLimits(proposals=8, weight_values=20000, tool_calls=100,
                            input_tokens=2_000_000, output_tokens=200_000),
        public_context=assignment["context"], native_dtypes=DTYPES,
        max_request_bytes=1_000_000,
    )


def response_schema(study):
    confidence = {"type": "number", "minimum": 0, "maximum": 1}
    code = {"type": "integer", "minimum": -1, "maximum": 77}
    properties = {
        "abstain": {"type": "boolean"},
        "commands_json": {"type": "string", "maxLength": 500000},
        "selected_fitted_handle": {"type": ["string", "null"]},
        "reason": {"type": "string", "maxLength": 4000},
    }
    if study == "weight-access":
        properties.update({
            "predicted_argmax_codes": {"type": "array", "items": code, "minItems": 2, "maxItems": 2},
            "prediction_confidences": {"type": "array", "items": confidence, "minItems": 2, "maxItems": 2},
            "strongest_competing_codes": {"type": "array", "items": code, "minItems": 2, "maxItems": 2},
            "competing_confidences": {"type": "array", "items": confidence, "minItems": 2, "maxItems": 2},
        })
    else:
        properties.update({
            "location_tensor": {"type": "string", "enum": [HEAD_WEIGHT, HEAD_BIAS, "uncertain"]},
            "output_row": code,
            "mechanism_class": {"type": "string", "enum": ["bias_shift", "feature_row_shift", "uncertain"]},
            "confidence": confidence,
            "predicted_target_gain": {"type": "number", "minimum": -1, "maximum": 1},
            "predicted_protected_harm": {"type": "number", "minimum": -1, "maximum": 1},
        })
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def validate_fields(response, properties):
    if type(response) is not dict or not set(properties) <= set(response):
        raise ValueError("Response is missing declared fields")
    for key, rule in properties.items():
        value = response[key]
        kind = rule["type"]
        if kind == ["string", "null"]:
            if value is not None and (type(value) is not str or len(value) > 128):
                raise ValueError("Invalid fitted handle")
        elif kind == "boolean":
            if type(value) is not bool:
                raise ValueError("Invalid abstention flag")
        elif kind == "string":
            if type(value) is not str or len(value) > rule.get("maxLength", 500000):
                raise ValueError("Invalid text field")
            if "enum" in rule and value not in rule["enum"]:
                raise ValueError("Unknown response class")
        elif kind == "array":
            if type(value) is not list or len(value) != 2:
                raise ValueError("Prediction arrays must contain two entries")
            for item in value:
                number_rule = rule["items"]
                if (number_rule["type"] == "integer" and type(item) is not int
                        or number_rule["type"] == "number" and type(item) not in (float, int)
                        or not math.isfinite(item)
                        or not number_rule["minimum"] <= item <= number_rule["maximum"]):
                    raise ValueError("Invalid prediction or confidence")
        elif (kind == "integer" and type(value) is not int
              or kind == "number" and type(value) not in (float, int)
              or not math.isfinite(value)
              or not rule["minimum"] <= value <= rule["maximum"]):
            raise ValueError("Invalid numeric response field")


def validate_answer(response, study):
    edit_fields = {"abstain", "commands_json", "selected_fitted_handle", "reason"}
    validate_fields(response, {k: v for k, v in response_schema(study)["properties"].items() if k not in edit_fields})


def validate_response(response, study):
    properties = response_schema(study)["properties"]
    if type(response) is not dict or set(response) != set(properties):
        raise ValueError("Response does not match the declared fields")
    validate_fields(response, properties)
    commands = json.loads(response["commands_json"])
    if type(commands) is not list or len(commands) > 8:
        raise ValueError("A literal program has at most eight commands")
    if response["abstain"] and (commands or response["selected_fitted_handle"] is not None):
        raise ValueError("Abstention cannot also select an edit")
    if response["selected_fitted_handle"] is not None and commands:
        raise ValueError("Select a literal program or a fitted handle")
    return commands


def plant(parent, spec, target_features, target_label, queue_map):
    outputs = parent.tensors[HEAD_BIAS].shape[0]
    choices = [i for i in range(outputs - 1) if queue_map[i] != queue_map[target_label]]
    seed = int(digest({k: spec[k] for k in ("lineage_id", "target_name", "mechanism")}), 16)
    row = choices[seed % len(choices)]
    if spec["mechanism"] == "bias_shift":
        commands = [{"op": "add", "tensor": HEAD_BIAS, "block": [[row, row + 1]], "values": [6.]}]
    else:
        direction = target_features.astype(np.float64).mean(0)
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            raise ValueError("Permitted target feature mean has zero norm")
        delta = 8. * direction / norm
        commands = [{"op": "add", "tensor": HEAD_WEIGHT, "block": [[row, row + 1], [0, len(delta)]],
                     "values": [delta.tolist()]}]
    proposal = StructuredProgram(parent.sha256, commands, "Evaluator-only planted study parent")
    lowered = lower_structured(parent, proposal, native_dtypes=DTYPES, cast="round")
    planted = apply_program(parent, lowered.program)
    return planted, {"mechanism": spec["mechanism"], "output_row": row,
                     "tensor": HEAD_BIAS if spec["mechanism"] == "bias_shift" else HEAD_WEIGHT,
                     "clean_parent_sha256": parent.sha256, "planted_parent_sha256": planted.sha256,
                     "proposal": proposal.to_dict(), "lowering": lowered.receipt,
                     "inverse": rollback(parent, lowered.program).to_dict()}


def weight_prelude(broker, parent, mode):
    public = {"metadata": broker.request({"op": "metadata"}), "parameters": []}
    if mode != "none":
        view = make_view(parent, mode)
        for tensor in (HEAD_WEIGHT, HEAD_BIAS):
            request = {"op": "weights", "tensor": tensor}
            if mode == "subset":
                request["indices"] = [list(index) for name, index in view.coordinates if name == tensor]
            result = broker.request(request)
            if result["status"] != "ok":
                raise ValueError("Declared parameter view was denied")
            public["parameters"].append({"request": request, "response": result})
    return public


def supplement(broker, parent, assignment, bundle, target_rows, protected_rows):
    regime = assignment["regime"]
    result, offers = {"regime": regime, "interaction": "fixed_training_prelude_then_one_response"}, []
    tick = time.perf_counter()
    target = bundle["target_features"][target_rows]
    protected = bundle["protected_features"][protected_rows]
    if regime == "brokered_analysis_only":
        result.update(operation="norm_arithmetic_on_already_exposed_values",
                      target_feature_norms=np.linalg.norm(target.astype(np.float64), axis=1).tolist(),
                      protected_feature_norms=np.linalg.norm(protected.astype(np.float64), axis=1).tolist())
        if assignment["parameter_view"] == "exact":
            result["exposed_tensor_norms"] = {name: float(np.linalg.norm(value)) for name, value in parent.tensors.items()}
        result["target_execution_performed"] = False
    elif regime == "brokered_calibration_forward":
        cal_target = ordered_rows(bundle, "calibration", assignment["target_label"], True, 4)
        cal_protected = ordered_rows(bundle, "calibration", assignment["target_label"], False, 8)
        rows = np.r_[cal_target, cal_protected]
        result.update(operation="trusted_fixed_calibration_forward",
                      allowlist=example_rows(bundle, "calibration", rows),
                      original_native_scores=evaluate_head(parent, bundle["calibration_features"][rows], native=True).tolist(),
                      native_lane="Torch CPU float32 affine; lowest-index argmax",
                      adaptive_execution=False, endpoint_feedback=False)
    elif regime == "brokered_fitting_permitted":
        result.update(operation="trusted_fixed_soft_retention_grid", candidates=[],
                      candidate_feedback="opaque_handle_and_configuration_only; no endpoint_scores_or_delta_bytes")
        for config in FIT_GRID:
            fit_tick, error = time.perf_counter(), None
            try:
                fitted = compile_arm(parent, target, protected, assignment["desired_code"], "soft", **config)
                commands = endpoint_commands(fitted["artifact"])
            except Exception as failure:
                commands, error = [{"op": "unavailable"}], type(failure).__name__
            request = {"op": "submit", "commands": commands, "intent": "Fixed fitting-permitted grid candidate"}
            offered = broker.request(request)
            if offered["status"] != "submitted":
                raise ValueError("Fitted candidate was not assigned an opaque handle")
            offers.append({"handle": offered["candidate_handle"], "configuration": config,
                           "request": request, "fit_error_type": error,
                           "fit_seconds": time.perf_counter() - fit_tick})
            result["candidates"].append({"candidate_handle": offered["candidate_handle"], "configuration": config})
    else:
        raise ValueError("Unknown supplementary regime")
    return public_record(result), offers, time.perf_counter() - tick


def endpoint_commands(endpoint):
    return [{"op": "replace", "tensor": name, "values": endpoint.tensors[name].tolist()}
            for name in (HEAD_WEIGHT, HEAD_BIAS)]


def literal_prompt(assignment, public, supplementary):
    if assignment["study"] == "weight-access":
        task = (
            "Read the original head on the first two target-role feature rows. Predict each original argmax code, "
            "its confidence, and the strongest competing code excluding the argmax (lowest index breaks ties). "
            "Then propose one literal head edit for the stated ownership policy, or abstain. "
            "Use -1 for an uncertain prediction. You receive no execution or candidate feedback."
        )
    else:
        task = (
            "Inspect this customer-support routing head against the stated desired policy. "
            "Diagnose a possible bias-shift or feature-row-shift error: name a tensor and output row, "
            "a mechanism class or uncertain, and your confidence. Predict signed target gain and protected "
            "net harm relative to this supplied parent, then propose one literal repair or abstain. "
            "No clean reference, error table, injection location hint, execution, or intervention result is provided."
        )
    rules = (
        "Return only the response-schema JSON object. commands_json is a JSON-encoded list of up to eight commands. "
        "Commands support replace/add/scale/low_rank with tensors head.weight or head.bias. "
        "Blocks are half-open [start,stop] axes; array operands must match the block shape exactly, without broadcasting. "
        "Example bias command: {\"op\":\"add\",\"tensor\":\"head.bias\",\"block\":[[77,78]],\"values\":[1.0]}. "
        "Lowering uses binary64 arithmetic followed by explicit float32 endpoint projection. "
        "Use commands_json=\"[]\" for abstention. selected_fitted_handle must be null except in the disclosed "
        "fitting-permitted prelude; choose exactly one offered handle or your literal commands, never both. "
        "Every proposal is committed before trusted lowering. No retry, correction, or endpoint feedback follows your response. "
        "Native tools are disabled. Your unobserved internal arithmetic is not an attested no-inference regime."
    )
    return task + "\n" + rules + "\n\nMatched context and declared parameter view:\n" + json.dumps(
        public, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ) + ("\n\nSupplementary permitted training prelude:\n" + json.dumps(
        supplementary, separators=(",", ":"), allow_nan=False) if supplementary is not None else "")


def implementation_amendment():
    return signed({
        "schema": "wsp-live-precall-implementation-amendment/1",
        "phase": "before_executable_lock_and_before_any_live_call",
        "recorded_utc": now(), "protocol_sha256": PROTOCOL_HASHES,
        "original_protocols_modified": False, "heldout_inputs_opened": False,
        "prior_main_utility_outcomes_inspected_by_root": False,
        "choices": {
            "ownership": "Weight access routes the chosen intent to specialist output 77/queue 5; diagnosis repairs its original intent queue.",
            "examples": "Sort eligible training-role rows by their fixed sample ID then row index; at most four target and eight protected rows.",
            "competing_code": "Highest original native score excluding the argmax; lowest-index ties.",
            "confidence": "Binary Brier for correctness of the stated argmax/competitor or joint diagnosis; prediction coverage and conditional Brier reported separately. Missing Brier has [0,1] sensitivity.",
            "answer_and_edit": "Independently valid reading/diagnosis fields remain scored when literal commands or edit fields are invalid. Transport failure or prohibited tools invalidates both. Edit validity and answer validity are separate recorded fields.",
            "supplements": "Fixed training-only analysis/forward/fitting preludes and one independent response; no adaptive multi-round interaction is claimed.",
            "analysis": "Norm arithmetic on already exposed feature rows and, for exact view only, exposed tensors; none receives no hidden weight statistics.",
            "forward": "Fixed calibration allowlist of at most four target and eight protected rows; native original-parent scores only.",
            "fitting": {"grid": FIT_GRID, "selection": "Opaque handles and configuration labels only; fitted deltas and endpoint feedback are absent from the prompt."},
            "control_fitting": "Same grid on the same four/eight context observations; deterministic ranking on the fixed training calibration role: harm-cap pass, gain minus harm, lower harm, gain, then grid order.",
            "analytic_control": "Copy target intent weight row to specialist output 77 and set its bias to target bias +0.5; an explicitly weight-dependent formula, not anonymous understanding.",
            "cosine_control": "Route to desired code when maximum cosine similarity to identical target examples is at least 0.85; no tuned threshold.",
            "diagnosis_primary": "Equal weighting of all five prespecified views, both target families and both mechanisms within each independent lineage; exact-only and other view slices are descriptive.",
            "matched_controls": {"absolute_norm_tolerance": MATCH_ATOL, "relative_norm_tolerance": MATCH_RTOL,
                                 "support": "Exact changed-coordinate count per tensor; native projection recorded separately. Mismatch retains diagnostic results and primary missing-effect sensitivity."},
            "random_location": "Assignment-hashed nonzero cyclic row shift; all outputs move, preserving requested tensor support and norm.",
            "shuffled_location": "Assignment-seeded permutation cycle, a derangement fixed independently of outcomes; native projection separately checked.",
            "primary_requests": "Novel official-test exact-request groups, one representative selected by sample ID then row index; full official test secondary.",
            "parallel_workers": WORKERS, "provider_rounds_per_assignment": 1, "provider_retries": 0,
            "failure": "An invocation checkpoint precedes every call. Recovery uses an existing complete receipt or accounts interrupted without retry. Failed/invalid/abstained/no-op candidates are not useful successes.",
            "inference": "Task/repeat/view observations average within lineage. Paired Student-t over six full-training lineages and exact sign-flip sensitivity; estimation only, no multiplicity-adjusted confirmation.",
            "missing_quantity_bounds": "Gain/harm [-1,1], paired gain differences [-2,2], accuracy/coverage/Brier [0,1]. Conditional Brier reports its observed prediction coverage; all-assigned Brier retains missing-answer bounds.",
            "evaluation_identity": "After the independently validated global seal, verify original training files against their preparation copies, bind model/encoder/preprocessing/features and evaluator-only file identities before execution, and require identical bytes for replay.",
        },
        "scope": "Generic frozen readout framework on a public trained customer-support router; no transformer-scale, zero-data, natural-backdoor or business-savings claim.",
    })


def prepare(root):
    root = root.resolve()
    targets = (ARTIFACT_DIR / "prepared", PROTOCOL_DIR / "source-snapshot",
               PROTOCOL_DIR / "implementation-amendment.json", LOCK_PATH)
    journal_path = root / PREPARATION_JOURNAL
    if journal_path.exists():
        return promote_preparation(root, check_signed(read(journal_path)))
    if any((root / path).exists() for path in targets):
        raise ValueError("Preparation is immutable; do not overwrite an existing study")
    verify_protocols(root)
    staging_parent = root / ".runs"
    staging_parent.mkdir(exist_ok=True)
    # The journal lets a process restart finish promotion without rebuilding inputs.
    stage = Path(tempfile.mkdtemp(dir=staging_parent, prefix="live-preparation-"))
    try:
        (stage / PROTOCOL_DIR).mkdir(parents=True)
        for name in (*PROTOCOL_HASHES, "label-amendment.json"):
            shutil.copyfile(root / PROTOCOL_DIR / name, stage / PROTOCOL_DIR / name)
        lock = _prepare(stage, root)
        verify_preparation(stage)
        journal = signed({"schema": "wsp-live-preparation-promotion/1",
            "stage_path": stage.relative_to(root).as_posix(),
            "targets": [path.as_posix() for path in targets], "lock_sha256": lock["sha256"],
            "files": [file_record(stage, path) for relative in targets
                      for path in ([stage / relative] if (stage / relative).is_file()
                                   else sorted((stage / relative).rglob("*"))) if path.is_file()]})
        write_json(journal_path, journal)
    except BaseException:
        # stage is a verified direct child created by mkdtemp in this repository.
        if stage.resolve().parent != staging_parent.resolve():
            raise ValueError("Preparation cleanup target escaped its staging directory")
        if not journal_path.exists():
            shutil.rmtree(stage)
        raise
    return promote_preparation(root, journal)


def promote_preparation(root, journal):
    check_signed(journal)
    targets = [path.as_posix() for path in (ARTIFACT_DIR / "prepared", PROTOCOL_DIR / "source-snapshot",
               PROTOCOL_DIR / "implementation-amendment.json", LOCK_PATH)]
    stage = checked_path(root, journal["stage_path"])
    if (journal.get("schema") != "wsp-live-preparation-promotion/1" or journal["targets"] != targets
            or stage.parent != (root / ".runs").resolve() or not stage.name.startswith("live-preparation-")):
        raise ValueError("Preparation promotion journal contract mismatch")
    files = journal["files"]
    if any(not any(record["path"] == target or record["path"].startswith(target + "/")
                   for target in targets) for record in files):
        raise ValueError("Preparation promotion inventory escaped its targets")
    for relative in targets:
        origin, destination = stage / relative, root / relative
        if origin.exists() == destination.exists():
            raise ValueError("Preparation promotion has missing or conflicting artifacts")
        location = origin if origin.exists() else destination
        base = stage if origin.exists() else root
        records = [record for record in files if record["path"] == relative or record["path"].startswith(relative + "/")]
        actual = {path.relative_to(base).as_posix() for path in
                  ([location] if location.is_file() else location.rglob("*")) if path.is_file()}
        if not records or actual != {record["path"] for record in records}:
            raise ValueError("Preparation promotion inventory is incomplete")
        check_files(base, records)
    for relative in targets:
        origin, destination = stage / relative, root / relative
        if origin.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(origin, destination)
    lock = verify_preparation(root)
    if lock["sha256"] != journal["lock_sha256"]:
        raise ValueError("Promoted preparation lock differs from the journal")
    (root / PREPARATION_JOURNAL).unlink()
    if stage.exists():
        shutil.rmtree(stage)
    return lock


def _prepare(root, origin_root):
    root = root.resolve()
    runtime = configure_runtime()
    verify_protocols(root)
    if (root / LOCK_PATH).exists() or (root / ARTIFACT_DIR / "prepared").exists():
        raise ValueError("Preparation is immutable; do not overwrite an existing study")
    amendment_path = root / PROTOCOL_DIR / "implementation-amendment.json"
    if amendment_path.exists():
        raise ValueError("Pre-call amendment already exists")
    write_json(amendment_path, implementation_amendment())
    prepared = root / ARTIFACT_DIR / "prepared"
    lineages = {}
    for lineage in LINEAGES:
        origin = origin_root / "artifacts/research/trained-support/main" / lineage
        destination = prepared / "training" / lineage
        destination.mkdir(parents=True)
        record = read(origin / "lineage.json")
        if set(record["files"]) != set(TRAINING_FILES) - {"lineage.json"}:
            raise ValueError("Training allowlist changed")
        for name in TRAINING_FILES:
            if name != "lineage.json":
                expected = record["files"][name]
                if sha(origin / name) != expected["sha256"] or (origin / name).stat().st_size != expected["bytes"]:
                    raise ValueError("Original training file failed identity verification")
            shutil.copyfile(origin / name, destination / name)
        lineages[lineage] = load_lineage(destination)
        if any(key.startswith("test_") for key in lineages[lineage]["features"]):
            raise ValueError("Evaluation data entered selection preparation")
    assignments = []
    for spec in assigned_specs():
        directory = prepared / "assignments" / spec["assignment_id"]
        directory.mkdir(parents=True)
        lineage = lineages[spec["lineage_id"]]
        clean, bundle = lineage["base"], lineage["features"]
        target_label = clean.metadata["output_names"].index(spec["target_name"])
        target_rows = ordered_rows(bundle, "target", target_label, True, 4)
        protected_rows = ordered_rows(bundle, "protected", target_label, False, 8)
        queue_map = bundle["queue_map"]
        parent, ledger = clean, None
        if spec["study"] == "blind-diagnosis":
            parent, ledger = plant(clean, spec, bundle["target_features"][target_rows], target_label, queue_map)
            write_json(directory / "evaluator-only-injection.json", signed(ledger))
        desired_code = 77 if spec["study"] == "weight-access" else target_label
        desired_queue = 5 if spec["study"] == "weight-access" else int(queue_map[target_label])
        objective = ("Route " + spec["target_name"] + " to " + ("specialist queue 5" if desired_queue == 5 else f"its intended queue {desired_queue}") +
                     " while preserving routes for all other intent families.")
        context = {
            "architecture": ARCHITECTURE, "feature_names": parent.metadata["feature_names"],
            "output_names": parent.metadata["output_names"], "queue_map": queue_map.tolist(),
            "encoder_sha256": parent.metadata["encoder_sha256"],
            "preprocessing_sha256": parent.metadata["preprocessing_sha256"],
            "feature_id": parent.metadata["feature_id"],
            "native_semantics": "Torch CPU float32 affine on supplied learned features; lowest-index argmax; map output code to queue.",
            "target_policy": {"target_name": spec["target_name"], "target_label_code": target_label,
                              "desired_output_code": desired_code, "desired_queue": desired_queue,
                              "protected_policy": "All other true intent codes retain their original queue mapping."},
            "target_role_examples": example_rows(bundle, "target", target_rows),
            "protected_role_examples": example_rows(bundle, "protected", protected_rows),
            "feature_scope": "Learned ReLU64 features from a from-scratch supervised encoder; raw-input correspondence is external.",
        }
        assignment = spec | {"objective": objective, "context": context, "target_label": target_label,
                             "desired_code": desired_code, "desired_queue": desired_queue,
                             "target_role_indices": target_rows.tolist(), "protected_role_indices": protected_rows.tolist(),
                             "training_path": (prepared / "training" / spec["lineage_id"]).relative_to(root).as_posix(),
                             "parent_path": (directory / "parent.json").relative_to(root).as_posix()}
        # Each path remains opaque to the proposer. Evaluator-only ledger never enters the prompt.
        write_json(directory / "parent.json", parent.to_dict())
        broker = make_broker(assignment, parent)
        public = weight_prelude(broker, parent, spec["parameter_view"])
        supplementary, offers, prelude_seconds = None, [], 0.
        if spec["regime"] != "primary":
            supplementary, offers, prelude_seconds = supplement(
                broker, parent, assignment, bundle, target_rows, protected_rows)
        assignment.update(fitted_offers=offers, preparation_broker_trace=broker.public_trace,
                          prelude_seconds=prelude_seconds,
                          parameter_values_exposed=sum(
                              np.asarray(item["response"]["values"]).size
                              if type(item["response"]["values"]) is not dict else len(item["response"]["values"])
                              for item in public["parameters"]))
        assignment["context_bytes"] = len(canonical_bytes(context))
        prompt_path = directory / "prompt.txt"
        prompt_path.write_text(literal_prompt(assignment, public, supplementary), encoding="utf-8", newline="\n")
        schema_path = directory / "response-schema.json"
        write_json(schema_path, response_schema(spec["study"]))
        assignment.update(prompt_path=prompt_path.relative_to(root).as_posix(),
                          schema_path=schema_path.relative_to(root).as_posix(),
                          prompt_sha256=sha(prompt_path), prompt_bytes=prompt_path.stat().st_size)
        assignment_path = directory / "assignment.json"
        write_json(assignment_path, signed(assignment))
        assignments.append(spec | {"prompt_path": assignment["prompt_path"],
                                  "prompt_sha256": assignment["prompt_sha256"], "prompt_bytes": assignment["prompt_bytes"],
                                  "assignment_path": assignment_path.relative_to(root).as_posix(),
                                  "assignment_sha256": sha(assignment_path)})
    sources = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_file() and path.is_relative_to(SOURCE_ROOT):
                sources[path.relative_to(SOURCE_ROOT).as_posix()] = path
    sources[Path(__file__).resolve().relative_to(SOURCE_ROOT).as_posix()] = Path(__file__).resolve()
    snapshot = root / PROTOCOL_DIR / "source-snapshot"
    if snapshot.exists():
        raise ValueError("A source snapshot cannot be overwritten")
    source_files = []
    for relative, origin in sorted(sources.items()):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
        source_files.append(file_record(root, destination) | {"source_path": relative})
    lock = signed({
        "schema": "wsp-live-preparation-lock/1", "prepared_utc": now(),
        "protocol_sha256": PROTOCOL_HASHES,
        "amendment_sha256": {"label-amendment.json": sha(root / PROTOCOL_DIR / "label-amendment.json"),
                             "implementation-amendment.json": sha(amendment_path)},
        "assignments": assignments, "assigned_total": len(assignments), "source_files": source_files,
        "permitted_input_files": [file_record(root, path) for path in sorted(prepared.rglob("*")) if path.is_file()],
        "provider": CodexProvider().configuration(), "workers": WORKERS,
        "runtime": runtime,
        "source_closure": "Actual imported project files including package initialization side effects; third-party dependencies recorded by version.",
        "selection_only": True, "evaluation_started": False, "provider_calls_started": False,
    })
    write_json(root / LOCK_PATH, lock)
    print(json.dumps({"prepared_lock_path": LOCK_PATH.as_posix(), "canonical_sha256": lock["sha256"],
                      "file_sha256": sha(root / LOCK_PATH), "assigned_workload": len(assignments)}), flush=True)
    return lock


def verify_preparation(root):
    root = root.resolve()
    verify_protocols(root)
    lock = check_signed(read(root / LOCK_PATH))
    if (lock["schema"] != "wsp-live-preparation-lock/1" or lock["protocol_sha256"] != PROTOCOL_HASHES
            or lock["selection_only"] is not True or lock["evaluation_started"] is not False
            or lock["provider_calls_started"] is not False
            or lock["provider"] != CodexProvider().configuration() or lock["workers"] != WORKERS
            or lock["runtime"] != configure_runtime()
            or set(lock["amendment_sha256"]) != {"label-amendment.json", "implementation-amendment.json"}
            or lock["assigned_total"] != 224 or len(lock["assignments"]) != 224
            or len({a["assignment_id"] for a in lock["assignments"]}) != 224
            or [{k: a[k] for k in assigned_specs()[0]} for a in lock["assignments"]] != assigned_specs()):
        raise ValueError("Preparation contract mismatch")
    for name, expected in lock["amendment_sha256"].items():
        if sha(root / PROTOCOL_DIR / name) != expected:
            raise ValueError("Pre-call amendment changed")
    check_files(root, lock["source_files"])
    check_files(root, lock["permitted_input_files"])
    for relative, records in ((PROTOCOL_DIR / "source-snapshot", lock["source_files"]),
                              (ARTIFACT_DIR / "prepared", lock["permitted_input_files"])):
        actual = {path.relative_to(root).as_posix() for path in (root / relative).rglob("*") if path.is_file()}
        if actual != {record["path"] for record in records}:
            raise ValueError("Preparation file inventory is incomplete")
    for assignment in lock["assignments"]:
        if sha(checked_path(root, assignment["assignment_path"])) != assignment["assignment_sha256"]:
            raise ValueError("Assignment changed after preparation")
        prompt = checked_path(root, assignment["prompt_path"])
        if sha(prompt) != assignment["prompt_sha256"] or prompt.stat().st_size != assignment["prompt_bytes"]:
            raise ValueError("Literal prompt changed after preparation")
    return lock


def require_snapshot(root, lock):
    expected = {record["source_path"]: record["sha256"] for record in lock["source_files"]}
    script_key = "scripts/run_live_studies.py"
    if sha(Path(__file__)) != expected[script_key]:
        raise ValueError("Run the frozen script, not a modified live working file")
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_file() and path.is_relative_to(SOURCE_ROOT):
                relative = path.relative_to(SOURCE_ROOT).as_posix()
                if relative not in expected or sha(path) != expected[relative]:
                    raise ValueError("Runtime project import is outside the executable source lock")


def replay_broker(assignment, parent):
    broker = make_broker(assignment, parent)
    handles = {}
    for event in assignment["preparation_broker_trace"]:
        result = broker.request(event["request"])
        original = event["response"]
        if original.get("candidate_handle"):
            handles[original["candidate_handle"]] = result.get("candidate_handle")
            result = result | {"candidate_handle": original["candidate_handle"]}
        if canonical_bytes(result) != canonical_bytes(original):
            raise ValueError("Prepared broker request replay differs")
    return broker, handles


def selection_path(root, assignment_id):
    return root / ARTIFACT_DIR / "selections" / assignment_id / "selection.json"


def check_checkpoint(root, assignment, lock, checkpoint):
    check_signed(checkpoint)
    locked = next(x for x in lock["assignments"] if x["assignment_id"] == assignment["assignment_id"])
    if (checkpoint.get("schema") != "wsp-live-invocation-checkpoint/1"
            or checkpoint.get("assignment_id") != assignment["assignment_id"]
            or checkpoint.get("assignment_sha256") != locked["assignment_sha256"]
            or checkpoint.get("prepared_lock_sha256") != lock["sha256"]
            or checkpoint.get("prompt_sha256") != assignment["prompt_sha256"]
            or checkpoint.get("schema_sha256") != digest(read(root / assignment["schema_path"]))
            or checkpoint.get("provider_configuration") != lock["provider"]
            or checkpoint.get("phase") != "call_may_start"):
        raise ValueError("Invocation checkpoint differs from the frozen assignment")
    return checkpoint


def check_receipt(root, assignment, lock, checkpoint, receipt):
    check_checkpoint(root, assignment, lock, checkpoint)
    check_signed(receipt)
    invocation = receipt.get("invocation", {})
    if (receipt.get("schema") != "wsp-live-provider-attempt/1"
            or receipt.get("status") not in {"completed", "failed", "timeout"}
            or invocation.get("prompt_sha256") != assignment["prompt_sha256"]
            or invocation.get("schema_sha256") != checkpoint["schema_sha256"]
            or invocation.get("configuration") != lock["provider"]):
        raise ValueError("Provider receipt is not bound to the exact prompt, schema and configuration")
    if receipt["status"] == "completed" and (receipt.get("error") is not None or receipt.get("returncode") != 0
            or receipt.get("unapproved_events") or receipt.get("event_errors") or not receipt.get("usage")):
        raise ValueError("Completed receipt contains a transport or access failure")
    return receipt


def check_published_provider(root, assignment, lock, checkpoint, directory):
    receipt = check_receipt(root, assignment, lock, checkpoint, read(directory / "receipt.json"))
    if (read(directory / "request.json") != {
            "prompt": (root / assignment["prompt_path"]).read_text(encoding="utf-8"),
            "configuration": lock["provider"]}
            or read(directory / "response-schema.json") != read(root / assignment["schema_path"])):
        raise ValueError("Published invocation inputs differ from the frozen prompt")
    derivative = read(directory / "event-publication.json")
    if (derivative.get("schema") != "wsp-public-event-derivative/1"
            or derivative.get("raw_events_sha256") != receipt["events_sha256"]
            or derivative.get("public_events_sha256") != sha(directory / "events.jsonl")
            or derivative.get("raw_bytes_published") is not False):
        raise ValueError("Published event derivative identity mismatch")
    return receipt


def publish_provider(root, assignment, lock, checkpoint, source, destination):
    # A publication retry copies an existing receipt; it never calls the model.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".provider-publication-") as stage:
        staged = Path(stage) / "provider"
        publish_attempt(source, staged)
        check_published_provider(root, assignment, lock, checkpoint, staged)
        destination.mkdir(exist_ok=True)
        for name in ("request.json", "response-schema.json", "receipt.json", "events.jsonl", "event-publication.json"):
            temporary = destination / ("." + name + ".tmp")
            try:
                with temporary.open("wb") as handle:
                    handle.write((staged / name).read_bytes())
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination / name)
            finally:
                temporary.unlink(missing_ok=True)
    return check_published_provider(root, assignment, lock, checkpoint, destination)


def validate_selection(root, assignment, lock, selected=None):
    directory = selection_path(root, assignment["assignment_id"]).parent
    selected = check_signed(selected if selected is not None else read(directory / "selection.json"))
    if (selected.get("schema") != "wsp-live-selection/1" or selected.get("assignment_id") != assignment["assignment_id"]
            or selected.get("terminal_status") not in TERMINAL or selected.get("evaluation_started") is not False):
        raise ValueError("Inconsistent terminal selection")
    check_files(root, selected["evidence_files"])
    actual_files = {path.relative_to(root).as_posix() for path in directory.rglob("*")
                    if path.is_file() and path.name != "selection.json"}
    if actual_files != {record["path"] for record in selected["evidence_files"]}:
        raise ValueError("Selection evidence inventory is incomplete")
    checkpoint = check_checkpoint(root, assignment, lock, read(directory / "invocation-checkpoint.json"))
    if selected["invocation_checkpoint_sha256"] != checkpoint["sha256"]:
        raise ValueError("Selection checkpoint identity mismatch")
    receipt = None
    if selected["provider_receipt_sha256"] is not None:
        receipt = check_published_provider(root, assignment, lock, checkpoint, directory / "provider")
        if (receipt["sha256"] != selected["provider_receipt_sha256"]
                or receipt["status"] != selected["provider_status"]
                or canonical_bytes(receipt["response"]) != canonical_bytes(selected["response"])):
            raise ValueError("Selection does not preserve its committed provider response")
    elif selected["provider_status"] != "interrupted" or selected["response"] is not None:
        raise ValueError("Missing receipt cannot contain a completed response")
    parent = TensorArtifact.from_dict(read(root / assignment["parent_path"]))
    state = selection_state(assignment, parent, receipt, selected["provider_status"])
    for key in ("terminal_status", "validation_error_type", "no_op", "answer_valid", "answer_validation_error_type"):
        if selected[key] != state[key]:
            raise ValueError("Selection decision differs from the committed response")
    if (selected["parent_sha256"] != parent.sha256 or selected["candidate_sha256"] != state["candidate"].sha256
            or selected["program_sha256"] != (state["program"].sha256 if state["program"] else None)
            or trace_semantics(selected["broker_trace"]) != trace_semantics(state["broker"].public_trace)):
        raise ValueError("Selected endpoints or broker history differ from the committed response")
    materialize_selection(root, assignment, selected)
    return selected


def trace_semantics(trace):
    """Compare replayed events while preserving each opaque handle's references."""
    handles, normalized, previous = {}, [], None
    for sequence, event in enumerate(trace):
        check_signed(event)
        if (event["schema"] != "wsp-broker-event/1" or event["sequence"] != sequence
                or event["previous_sha256"] != previous):
            raise ValueError("Selection broker trace was changed")
        previous = event["sha256"]
        request, response = dict(event["request"]), dict(event["response"])
        if response.get("candidate_handle"):
            handle = response["candidate_handle"]
            if handle in handles:
                raise ValueError("Opaque handle was issued twice")
            handles[handle] = len(handles)
            response["candidate_handle"] = handles[handle]
        if request.get("candidate_handle") is not None:
            handle = request["candidate_handle"]
            if handle not in handles:
                raise ValueError("Opaque handle was used before submission")
            request["candidate_handle"] = handles[handle]
        normalized.append({"request": request, "response": response})
    return normalized


def selection_state(assignment, parent, receipt, status):
    broker, handles = replay_broker(assignment, parent)
    candidate, program, lowering, proposal, no_op, error = parent, None, None, None, False, None
    response = receipt.get("response") if receipt else None
    provider_status = status
    answer_valid, answer_error = False, None
    if status == "completed":
        try:
            validate_answer(response, assignment["study"])
            answer_valid = True
        except (ValueError, KeyError, TypeError, OverflowError) as failure:
            answer_error = type(failure).__name__
    if status == "completed":
        try:
            commands = validate_response(response, assignment["study"])
            usage = receipt["usage"]
            broker.record_usage(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))
            if response["abstain"]:
                broker.abstain()
                status = "abstained"
            else:
                offered = response["selected_fitted_handle"]
                if offered is not None:
                    if assignment["regime"] != "brokered_fitting_permitted" or offered not in handles:
                        raise ValueError("Unavailable fitted candidate handle")
                    request = next(x["request"] for x in assignment["fitted_offers"] if x["handle"] == offered)
                    selected_handle = handles[offered]
                else:
                    request = {"op": "submit", "commands": commands,
                               "intent": response["reason"].strip() or "Selected live literal proposal"}
                    submitted = broker.request(request)
                    if submitted["status"] != "submitted":
                        raise ValueError("Opaque submission denied")
                    selected_handle = submitted["candidate_handle"]
                proposal = StructuredProgram(parent.sha256, request["commands"], request["intent"])
                lowered = lower_structured(parent, proposal, native_dtypes=DTYPES, cast="round")
                program, lowering = lowered.program, lowered.receipt
                candidate = apply_program(parent, program)
                if broker.request({"op": "seal", "candidate_handle": selected_handle})["status"] != "sealed":
                    raise ValueError("Opaque selection could not be sealed")
                no_op, status = candidate.sha256 == parent.sha256, "sealed"
        except Exception as failure:
            error, status = type(failure).__name__, "invalid"
            broker.abstain()
            candidate, program, lowering, proposal = parent, None, None, None
    else:
        broker.abstain()
    if status not in TERMINAL:
        raise ValueError("Missing terminal selection status")
    return {"terminal_status": status, "provider_status": provider_status, "response": response,
            "candidate": candidate, "program": program, "lowering": lowering, "proposal": proposal,
            "broker": broker, "no_op": no_op, "validation_error_type": error,
            "answer_valid": answer_valid, "answer_validation_error_type": answer_error}


def finish_selection(root, assignment, receipt, status, invocation):
    directory = selection_path(root, assignment["assignment_id"]).parent
    parent = TensorArtifact.from_dict(read(root / assignment["parent_path"]))
    state = selection_state(assignment, parent, receipt, status)
    candidate, program, lowering, proposal = (state[key] for key in ("candidate", "program", "lowering", "proposal"))
    if program is not None:
        write_json(directory / "program.json", program.to_dict())
        write_json(directory / "rollback.json", rollback(parent, program).to_dict())
        write_json(directory / "proposal.json", proposal.to_dict())
        write_json(directory / "lowering.json", lowering)
        write_json(directory / "candidate.json", candidate.to_dict())
    selection = signed({
        "schema": "wsp-live-selection/1", "assignment_id": assignment["assignment_id"],
        "terminal_status": state["terminal_status"], "provider_status": state["provider_status"], "selection_utc": now(),
        "invocation_checkpoint_sha256": invocation["sha256"],
        "provider_receipt_sha256": receipt["sha256"] if receipt else None,
        **{key: state[key] for key in ("response", "validation_error_type", "no_op", "answer_valid", "answer_validation_error_type")},
        "interrupted_transport_termination_attested": False if state["provider_status"] == "interrupted" else None,
        "parent_sha256": parent.sha256, "candidate_sha256": candidate.sha256,
        "program_sha256": program.sha256 if program else None,
        "broker_trace": state["broker"].public_trace, "evaluation_started": False,
        "evidence_files": [file_record(root, path) for path in sorted(directory.rglob("*"))
                           if path.is_file() and path.name != "selection.json"],
        "selection_rule": "One committed response; one literal or pre-offered opaque handle; no corrections or retries.",
    })
    write_json(directory / "selection.json", selection)
    return selection


def select_one(root, raw, locked_assignment, lock):
    assignment = check_signed(read(root / locked_assignment["assignment_path"]))
    existing = selection_path(root, assignment["assignment_id"])
    if existing.exists():
        return validate_selection(root, assignment, lock)
    public_dir = existing.parent
    public_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir = raw / assignment["assignment_id"] / "provider"
    checkpoint_path = public_dir / "invocation-checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = check_checkpoint(root, assignment, lock, read(checkpoint_path))
        if (attempt_dir / "receipt.json").exists():
            receipt = publish_provider(root, assignment, lock, checkpoint, attempt_dir, public_dir / "provider")
            return finish_selection(root, assignment, receipt, receipt["status"], checkpoint)
        return finish_selection(root, assignment, None, "interrupted", checkpoint)
    checkpoint = signed({
        "schema": "wsp-live-invocation-checkpoint/1", "assignment_id": assignment["assignment_id"],
        "assignment_sha256": locked_assignment["assignment_sha256"],
        "prepared_lock_sha256": lock["sha256"], "prompt_sha256": assignment["prompt_sha256"],
        "schema_sha256": digest(read(root / assignment["schema_path"])),
        "invocation_utc": now(), "provider_configuration": lock["provider"], "phase": "call_may_start",
        "retry_rule": "No retry after uncertain invocation; recover receipt or account interrupted.",
    })
    write_json(checkpoint_path, checkpoint)
    receipt, status = None, "interrupted"
    try:
        receipt = CodexProvider().call(
            (root / assignment["prompt_path"]).read_text(encoding="utf-8"),
            directory=attempt_dir, cwd=raw / assignment["assignment_id"] / "cwd",
            schema=read(root / assignment["schema_path"]),
        )
        receipt = publish_provider(root, assignment, lock, checkpoint, attempt_dir, public_dir / "provider")
        status = receipt["status"]
    except Exception as failure:
        write_json(public_dir / "transport-interruption.json", signed({
            "phase": "transport_or_publication_interrupted", "error_type": type(failure).__name__,
            "receipt_exists": (attempt_dir / "receipt.json").exists(), "retry_performed": False,
        }))
        if (attempt_dir / "receipt.json").exists():
            receipt = publish_provider(root, assignment, lock, checkpoint, attempt_dir, public_dir / "provider")
            status = receipt["status"]
    selection = finish_selection(root, assignment, receipt, status, checkpoint)
    print(json.dumps({"assignment_id": assignment["assignment_id"],
                      "terminal_status": selection["terminal_status"],
                      "provider_seconds": receipt["elapsed_seconds"] if receipt else None}), flush=True)
    return selection


def run_selection(root, raw, lock):
    verified = verify_preparation(root)
    if canonical_bytes(lock) != canonical_bytes(verified):
        raise ValueError("Selection received a different preparation lock")
    require_snapshot(root, lock)
    if (root / GATE_PATH).exists():
        raise ValueError("Global selection is already sealed; no more provider invocations")
    if raw.resolve().is_relative_to(root.resolve()):
        raise ValueError("Raw provider transport must stay outside the public repository")
    # ponytail: fixed eight independent workers; change throughput only in a new pre-call amendment.
    with ThreadPoolExecutor(max_workers=lock["workers"]) as pool:
        jobs = {pool.submit(select_one, root, raw, assignment, lock): assignment for assignment in lock["assignments"]}
        for job in as_completed(jobs):
            job.result()
    return seal(root, lock)


def seal(root, lock):
    verified = verify_preparation(root)
    if canonical_bytes(lock) != canonical_bytes(verified):
        raise ValueError("Selection seal received a different preparation lock")
    require_snapshot(root, lock)
    gate_file = root / GATE_PATH
    if gate_file.exists():
        return validate_gate(root, expected=check_signed(read(gate_file))["sha256"])
    if (root / ARTIFACT_DIR / "evaluation").exists():
        raise ValueError("Evaluation started before the global selection seal")
    accounting = []
    for assignment in lock["assignments"]:
        full = check_signed(read(root / assignment["assignment_path"]))
        selected = validate_selection(root, full, lock)
        if (selected["assignment_id"] != assignment["assignment_id"]
                or selected["terminal_status"] not in TERMINAL or selected["evaluation_started"] is not False):
            raise ValueError("Incomplete or inconsistent terminal selection")
        accounting.append({"assignment_id": assignment["assignment_id"],
                           "selection_sha256": selected["sha256"], "terminal_status": selected["terminal_status"]})
    gate = signed({
        "schema": "wsp-live-global-selection-seal/1", "phase": "selection_complete_before_evaluation",
        "selection_frozen": True, "all_assigned_accounted": True,
        "studies": ["blind-diagnosis", "weight-access"],
        "prepared_lock_path": LOCK_PATH.as_posix(), "prepared_lock_sha256": sha(root / LOCK_PATH),
        "protocol_sha256": PROTOCOL_HASHES,
        "assigned_total": len(lock["assignments"]), "terminal_total": len(accounting),
        "all_terminal": True, "evaluation_started": False, "accounting": accounting, "sealed_utc": now(),
    })
    write_json(gate_file, gate)
    return validate_gate(root, expected=gate["sha256"])


def validate_gate(root, expected):
    lock = verify_preparation(root)
    require_snapshot(root, lock)
    gate = check_signed(read(root / GATE_PATH))
    if (gate["schema"] != "wsp-live-global-selection-seal/1" or gate["sha256"] != expected
            or gate["phase"] != "selection_complete_before_evaluation"
            or gate["selection_frozen"] is not True or gate["all_assigned_accounted"] is not True
            or gate["all_terminal"] is not True or gate["evaluation_started"] is not False
            or gate["studies"] != ["blind-diagnosis", "weight-access"]
            or gate["prepared_lock_path"] != LOCK_PATH.as_posix()
            or gate["prepared_lock_sha256"] != sha(root / LOCK_PATH)
            or gate["protocol_sha256"] != PROTOCOL_HASHES
            or gate["assigned_total"] != 224 or gate["terminal_total"] != 224
            or len(gate["accounting"]) != 224):
        raise ValueError("Global selection gate contract mismatch")
    accounts = {x["assignment_id"]: x for x in gate["accounting"]}
    if len(accounts) != 224 or set(accounts) != {a["assignment_id"] for a in lock["assignments"]}:
        raise ValueError("Global accounting IDs differ from frozen assignments")
    for assignment_id, account in accounts.items():
        assignment = next(x for x in lock["assignments"] if x["assignment_id"] == assignment_id)
        full = check_signed(read(root / assignment["assignment_path"]))
        selected = validate_selection(root, full, lock)
        if (selected["assignment_id"] != assignment_id or selected["sha256"] != account["selection_sha256"]
                or selected["terminal_status"] != account["terminal_status"]
                or account["terminal_status"] not in TERMINAL or selected["evaluation_started"] is not False):
            raise ValueError("Global gate selection identity mismatch")
    return gate


def materialize_selection(root, assignment, selected):
    parent = TensorArtifact.from_dict(read(root / assignment["parent_path"]))
    if parent.sha256 != selected["parent_sha256"]:
        raise ValueError("Selected parent identity mismatch")
    directory = selection_path(root, assignment["assignment_id"]).parent
    if selected["terminal_status"] != "sealed":
        if selected["candidate_sha256"] != parent.sha256:
            raise ValueError("Failed selection must deploy its unchanged parent")
        return parent, parent, None
    proposal = StructuredProgram.from_dict(read(directory / "proposal.json"))
    lowering = read(directory / "lowering.json")
    if not verify_lowering(parent, proposal, lowering):
        raise ValueError("Independent selected lowering verification failed")
    program = Program.from_dict(read(directory / "program.json"))
    candidate = apply_program(parent, program)
    inverse = Program.from_dict(read(directory / "rollback.json"))
    if (candidate.sha256 != selected["candidate_sha256"] or program.sha256 != selected["program_sha256"]
            or program.sha256 != lowering["program_sha256"] or candidate.sha256 != lowering["candidate_sha256"]
            or candidate.sha256 != TensorArtifact.from_dict(read(directory / "candidate.json")).sha256
            or apply_program(candidate, inverse).sha256 != parent.sha256):
        raise ValueError("Exact selected endpoint or rollback identity failed")
    broker, handles = replay_broker(assignment, parent)
    if selected["response"]:
        usage = read(directory / "provider" / "receipt.json")["usage"]
        broker.record_usage(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))
        response = selected["response"]
        offered = response["selected_fitted_handle"]
        if offered is not None:
            if assignment["regime"] != "brokered_fitting_permitted" or offered not in handles:
                raise ValueError("Committed fitted handle is unavailable")
            request = next(x["request"] for x in assignment["fitted_offers"] if x["handle"] == offered)
            handle = handles[offered]
        else:
            request = {"op": "submit", "commands": validate_response(response, assignment["study"]),
                       "intent": response["reason"].strip() or "Selected live literal proposal"}
            handle = broker.request(request)["candidate_handle"]
        if canonical_bytes(proposal.to_dict()) != canonical_bytes(
                StructuredProgram(parent.sha256, request["commands"], request["intent"]).to_dict()):
            raise ValueError("Selected proposal differs from the committed response")
        broker.request({"op": "seal", "candidate_handle": handle})
    return parent, candidate, broker


def group_primary(bundle):
    groups, ids = bundle["test_group_ids"], bundle["test_ids"]
    novel = bundle["test_novel_mask"].astype(bool)
    selected = {}
    for row in sorted(range(len(ids)), key=lambda i: (str(ids[i]), i)):
        if novel[row]:
            selected.setdefault(str(groups[row]), row)
    primary = np.zeros(len(ids), dtype=bool)
    primary[list(selected.values())] = True
    if not primary.any():
        raise ValueError("No novel exact-request group representatives")
    return primary


def mapped_intervention(parent, candidate, assignment_id, kind):
    rows = parent.tensors[HEAD_BIAS].shape[0]
    seed = int(digest({"assignment_id": assignment_id, "control": kind})[:16], 16)
    if kind == "random":
        mapping = np.roll(np.arange(rows), 1 + seed % (rows - 1))
    else:
        order = np.random.default_rng(seed).permutation(rows)
        mapping = np.empty(rows, dtype=int)
        mapping[order] = np.roll(order, 1)
    requested, realized, support, errors = {}, {}, {}, []
    for name in DTYPES:
        original_delta = candidate.tensors[name] - parent.tensors[name]
        proposed_delta = np.zeros_like(original_delta)
        proposed_delta[mapping] = original_delta
        requested[name] = proposed_delta
        realized[name] = (parent.tensors[name] + proposed_delta).astype(np.float32).astype(np.float64)
        actual_delta = realized[name] - parent.tensors[name]
        norm, actual_norm = float(np.linalg.norm(original_delta)), float(np.linalg.norm(actual_delta))
        exact_support = int(np.count_nonzero(original_delta)) == int(np.count_nonzero(actual_delta))
        matched = exact_support and math.isclose(norm, actual_norm, abs_tol=MATCH_ATOL, rel_tol=MATCH_RTOL)
        support[name] = {"proposed_support": int(np.count_nonzero(original_delta)),
                         "requested_control_support": int(np.count_nonzero(proposed_delta)),
                         "realized_control_support": int(np.count_nonzero(actual_delta)),
                         "proposed_norm": norm, "requested_control_norm": float(np.linalg.norm(proposed_delta)),
                         "realized_control_norm": actual_norm, "matches_frozen_tolerance": matched}
        errors.append(matched)
    endpoint = parent.replace(**realized)
    return endpoint, {"kind": kind, "row_mapping": mapping.tolist(), "derangement": bool(np.all(mapping != np.arange(rows))),
                      "tensors": support, "matches_frozen_tolerance": all(errors),
                      "absolute_tolerance": MATCH_ATOL, "relative_tolerance": MATCH_RTOL,
                      "scope": "Requested permutation preserves support/norm; native float32 projection may change realized support/norm."}


def cosine_codes(scores, features, target_features, desired_code):
    codes = scores.argmax(1).copy()
    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), np.finfo(np.float32).tiny)
    anchors = target_features / np.maximum(np.linalg.norm(target_features, axis=1, keepdims=True), np.finfo(np.float32).tiny)
    similarity = ((features / norms) @ anchors.T).max(1)
    codes[similarity >= .85] = desired_code
    return codes


def fitted_control(parent, assignment, bundle):
    target = bundle["target_features"][assignment["target_role_indices"]]
    protected = bundle["protected_features"][assignment["protected_role_indices"]]
    cal = bundle["calibration_features"]
    labels, queue_map = bundle["calibration_labels"], bundle["queue_map"]
    before = evaluate_head(parent, cal, native=True).argmax(1)
    options, started = [], time.perf_counter()
    for index, config in enumerate(FIT_GRID):
        option = compile_arm(parent, target, protected, assignment["desired_code"], "soft", **config)
        codes = evaluate_head(option["artifact"], cal, native=True).argmax(1)
        metric = behavior_metrics(before, codes, labels, assignment["target_label"], queue_map, assignment["desired_queue"])
        gain, harm = metric["target_gain"], metric["protected_harm"]
        rank = (harm <= MARGINS["protected_harm"], gain - harm, -harm, gain, -index)
        options.append((rank, option["artifact"], config, metric))
    chosen = max(options, key=lambda item: item[0])
    return chosen[1], {"selected_configuration": chosen[2], "calibration": chosen[3],
                      "options": [{"configuration": x[2], "calibration": x[3], "candidate_sha256": x[1].sha256} for x in options],
                      "fit_selection_seconds": time.perf_counter() - started,
                      "fitting_scope": "Same permitted context plus training calibration selection; separately fitting-permitted control."}


def binary_rubric(predictions, confidences, truth):
    available = np.asarray(predictions) >= 0
    correct = np.asarray(predictions) == np.asarray(truth)
    brier = (np.asarray(confidences, dtype=float) - correct.astype(float)) ** 2
    return {"truth_codes": list(map(int, truth)), "predicted_codes": predictions,
            "all_assigned_accuracy": float(correct.mean()), "prediction_coverage": float(available.mean()),
            "conditional_accuracy": float(correct[available].mean()) if available.any() else None,
            "conditional_binary_brier": float(brier[available].mean()) if available.any() else None,
            "binary_brier_all_assigned_bounds": [float(brier[available].sum() / len(truth)),
                                                float((brier[available].sum() + (~available).sum()) / len(truth))]}


def evaluate_assignment(root, assignment, selected, gate, lineage, primary, destination):
    started = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=False)
    parent, candidate, broker = materialize_selection(root, assignment, selected)
    bundle, clean = lineage["features"], lineage["base"]
    features, labels, queue_map = bundle["test_features"], bundle["test_labels"], bundle["queue_map"]
    native_tick = time.perf_counter()
    parent_scores = evaluate_head(parent, features, native=True)
    clean_scores = evaluate_head(clean, features, native=True)
    candidate_scores = evaluate_head(candidate, features, native=True)
    codes = {"parent": parent_scores.argmax(1), "proposed": candidate_scores.argmax(1), "clean": clean_scores.argmax(1)}
    controls, matching, control_records = {}, {}, {}
    target_features = bundle["target_features"][assignment["target_role_indices"]]
    if assignment["study"] == "weight-access":
        overlay = codes["parent"].copy()
        overlay[overlay == assignment["target_label"]] = assignment["desired_code"]
        codes["overlay"] = overlay
        codes["cosine"] = cosine_codes(parent_scores, features, target_features, assignment["desired_code"])
        weight, bias = parent.tensors[HEAD_WEIGHT].copy(), parent.tensors[HEAD_BIAS].copy()
        weight[77], bias[77] = weight[assignment["target_label"]], np.float32(bias[assignment["target_label"]] + .5)
        controls["analytic"] = parent.replace(**{HEAD_WEIGHT: weight, HEAD_BIAS: bias})
        controls["fitting"], control_records["fitting"] = fitted_control(parent, assignment, bundle)
    else:
        controls["sham"] = parent
        ledger = check_signed(read(root / Path(assignment["parent_path"]).parent / "evaluator-only-injection.json"))
        inverse = Program.from_dict(ledger["inverse"])
        oracle = apply_program(parent, inverse)
        if (ledger["clean_parent_sha256"] != clean.sha256 or ledger["planted_parent_sha256"] != parent.sha256
                or oracle.sha256 != clean.sha256
                or not verify_lowering(clean, StructuredProgram.from_dict(ledger["proposal"]), ledger["lowering"])):
            raise ValueError("Planted inverse oracle does not restore the exact original parent")
        controls["exact_inverse_oracle"] = oracle
        control_records["exact_inverse_oracle"] = {"inverse_sha256": inverse.sha256,
            "planted_parent_sha256": parent.sha256, "restored_sha256": oracle.sha256,
            "clean_parent_sha256": clean.sha256, "exact_restoration_verified": True}
        for kind in ("random", "shuffled"):
            controls[kind], matching[kind] = mapped_intervention(parent, candidate, assignment["assignment_id"], kind)
        spec = specification("blind-diagnosis", assignment["lineage_id"], assignment["target_name"],
                             "none", mechanism=assignment["mechanism"])
        context_assignment = check_signed(read(root / next(
            x["assignment_path"] for x in read(root / LOCK_PATH)["assignments"] if x["assignment_id"] == spec["assignment_id"])))
        context_selected = check_signed(read(selection_path(root, spec["assignment_id"])))
        controls["context_only"] = materialize_selection(root, context_assignment, context_selected)[1]
    for name, endpoint in controls.items():
        codes[name] = evaluate_head(endpoint, features, native=True).argmax(1)
    native_seconds = time.perf_counter() - native_tick
    metrics = {}
    for name, prediction in codes.items():
        if name == "parent":
            continue
        metrics[name] = {
            "primary_novel_groups": behavior_metrics(codes["parent"][primary], prediction[primary], labels[primary],
                                                     assignment["target_label"], queue_map, assignment["desired_queue"]),
            "secondary_official": behavior_metrics(codes["parent"], prediction, labels, assignment["target_label"],
                                                   queue_map, assignment["desired_queue"]),
        }
        if assignment["study"] == "blind-diagnosis":
            metrics[name]["clean_relative_primary"] = behavior_metrics(
                codes["clean"][primary], prediction[primary], labels[primary],
                assignment["target_label"], queue_map, assignment["desired_queue"])
    primary_metric = metrics["proposed"]["primary_novel_groups"]
    usable = primary_metric["target_gain"] is not None and primary_metric["protected_harm"] is not None
    useful = bool(selected["terminal_status"] == "sealed" and not selected["no_op"] and usable
                  and primary_metric["target_gain"] >= MARGINS["target_gain"]
                  and primary_metric["protected_harm"] <= MARGINS["protected_harm"])
    response = selected["response"] if selected["provider_status"] == "completed" else None
    rubric = {}
    if assignment["study"] == "weight-access":
        original = evaluate_head(parent, target_features[:2], native=True)
        winners = original.argmax(1)
        competing_scores = original.copy()
        competing_scores[np.arange(len(winners)), winners] = -np.inf
        competitors = competing_scores.argmax(1)
        if response is None or not selected["answer_valid"]:
            argmax, competitor, confidence, competing_confidence = [-1, -1], [-1, -1], [0., 0.], [0., 0.]
        else:
            argmax, competitor = response["predicted_argmax_codes"], response["strongest_competing_codes"]
            confidence, competing_confidence = response["prediction_confidences"], response["competing_confidences"]
        rubric = {"read": binary_rubric(argmax, confidence, winners),
                  "competing_code": binary_rubric(competitor, competing_confidence, competitors),
                  "competing_code_scope": "Native score ranking, not causal localization."}
    else:
        available = response is not None and selected["answer_valid"]
        location = bool(available and response["location_tensor"] == ledger["tensor"] and response["output_row"] == ledger["output_row"])
        mechanism = bool(available and response["mechanism_class"] == ledger["mechanism"])
        confident = bool(available and response["location_tensor"] != "uncertain" and response["output_row"] >= 0
                         and response["mechanism_class"] != "uncertain")
        joint = location and mechanism
        rubric = {"location_correct": location, "mechanism_correct": mechanism, "joint_diagnosis_correct": joint,
                  "diagnosis_coverage": confident,
                  "conditional_binary_brier": (response["confidence"] - int(joint)) ** 2 if confident else None,
                  "binary_brier_bounds": [(response["confidence"] - int(joint)) ** 2] * 2 if confident else [0., 1.],
                  "predicted_target_gain": response["predicted_target_gain"] if available else None,
                  "predicted_protected_harm": response["predicted_protected_harm"] if available else None,
                  "observed_target_gain": primary_metric["target_gain"],
                  "observed_protected_harm": primary_metric["protected_harm"],
                  "mechanism_population": "Only the two controlled planted families; no natural-backdoor claim."}
    exact_scores = evaluate_head(candidate, features)
    numerics = {
        "native_lane": "Torch CPU float32 affine on fixed learned float32 features",
        "binary64_lane": "NumPy binary64 affine; not an exact real proof",
        "candidate_argmax_disagreements": int((candidate_scores.argmax(1) != exact_scores.argmax(1)).sum()),
        "maximum_score_difference": float(np.max(np.abs(candidate_scores.astype(np.float64) - exact_scores))),
        "whole_encoder_or_native_kernel_certificate": False,
    }
    prediction_identity = digest({name: value.tolist() for name, value in sorted(codes.items())})
    deterministic = public_record({
        "assignment_id": assignment["assignment_id"], "selection_sha256": selected["sha256"],
        "global_seal_sha256": gate["sha256"], "metrics": metrics, "rubric": rubric, "numerics": numerics,
        "control_matching": matching, "prediction_identity": prediction_identity,
        "control_endpoint_sha256": {name: endpoint.sha256 for name, endpoint in controls.items()},
        "joint_useful_success": useful, "evaluation_missing": not usable,
        "verification": {"lowering_verified": selected["terminal_status"] == "sealed",
                         "application_and_rollback_verified": selected["terminal_status"] == "sealed",
                         "exact_inverse_oracle_applied": assignment["study"] == "blind-diagnosis"},
    })
    np.savez_compressed(destination / "predictions.npz", **{name + "_codes": value for name, value in codes.items()})
    audit = None
    if broker is not None:
        audit = broker.evaluate(lambda chosen: {"candidate_sha256": chosen.artifact.sha256,
                                                "deterministic_outcome_sha256": digest(deterministic)},
                                evaluator_id="sealed-live-native-readout/1")
        if (audit["status"] != "succeeded"
                or audit["evaluation"]["result"]["candidate_sha256"] != candidate.sha256
                or not verify_trace(audit["evaluation"]["trace"],
                                    expected_head=audit["evaluation"]["trace_head_sha256"])):
            raise ValueError("Sealed broker evaluation replay failed")
        write_json(destination / "broker-audit.json", audit)
    outcome = signed({
        "schema": "wsp-live-outcome/1", **{k: assignment[k] for k in assigned_specs()[0]},
        "assigned": True, "terminal_status": selected["terminal_status"], "no_op": selected["no_op"],
        "deterministic": deterministic, "deterministic_sha256": digest(deterministic),
        "selection_sha256": selected["sha256"], "global_seal_sha256": gate["sha256"],
        "control_records": control_records,
        "missingness_sensitivity": {"gain": [-1., 1.], "harm": [-1., 1.], "useful": [0., 1.]} if not usable else None,
        "cost": {"native_evaluation_seconds": native_seconds, "evaluation_seconds": time.perf_counter() - started,
                 "evidence_bytes_pre_outcome": sum(path.stat().st_size for path in destination.iterdir() if path.is_file())},
        "predictions_file_sha256": sha(destination / "predictions.npz"),
    })
    write_json(destination / "outcome.json", outcome)
    return outcome


def safe_evaluation(root, assignment, selected, gate, lineage, primary, destination):
    try:
        return evaluate_assignment(root, assignment, selected, gate, lineage, primary, destination)
    except Exception as failure:
        deterministic = {"assignment_id": assignment["assignment_id"], "selection_sha256": selected["sha256"],
                         "global_seal_sha256": gate["sha256"], "evaluation_missing": True,
                         "joint_useful_success": False, "error_type": type(failure).__name__}
        outcome = signed({"schema": "wsp-live-outcome/1", **{k: assignment[k] for k in assigned_specs()[0]},
                          "assigned": True, "terminal_status": selected["terminal_status"], "no_op": selected["no_op"],
                          "deterministic": deterministic, "deterministic_sha256": digest(deterministic),
                          "selection_sha256": selected["sha256"], "global_seal_sha256": gate["sha256"],
                          "missingness_sensitivity": {"gain": [-1., 1.], "harm": [-1., 1.], "useful": [0., 1.]}})
        write_json(destination / "outcome.json", outcome)
        return outcome


def evaluation_inputs(root, lock):
    """Called only after gate validation; bind every file used by the evaluator."""
    records = {item["path"]: item for item in (*lock["source_files"], *lock["permitted_input_files"])}
    for path in (LOCK_PATH, GATE_PATH, *(PROTOCOL_DIR / name for name in PROTOCOL_HASHES),
                 *(PROTOCOL_DIR / name for name in lock["amendment_sha256"])):
        record = file_record(root, root / path)
        records[record["path"]] = record
    for assignment in lock["assignments"]:
        path = selection_path(root, assignment["assignment_id"])
        selected = check_signed(read(path))
        for item in (file_record(root, path), *selected["evidence_files"]):
            records[item["path"]] = item
    lineages = {}
    for lineage in LINEAGES:
        original = root / "artifacts/research/trained-support/main" / lineage
        prepared = root / ARTIFACT_DIR / "prepared/training" / lineage
        for name in TRAINING_FILES:
            if (original / name).read_bytes() != (prepared / name).read_bytes():
                raise ValueError("Evaluator training bytes differ from preparation")
        for name in (*TRAINING_FILES, "evaluation-features.npz", "evaluation-features.json"):
            record = file_record(root, original / name)
            records[record["path"]] = record
        loaded = load_lineage(original)
        if not all(key in loaded["features"] for key in
                   ("test_features", "test_labels", "test_ids", "test_group_ids", "test_novel_mask")):
            raise ValueError("Evaluator-only published test bundle is missing")
        base, provenance = loaded["base"], loaded["record"]
        if (loaded["id"] != lineage or base.metadata["encoder_sha256"] != provenance["encoder_sha256"]
                or base.metadata["preprocessing_sha256"] != provenance["preprocessing_sha256"]
                or base.metadata["feature_id"] != provenance["feature_id"]):
            raise ValueError("Evaluator model, encoder, preprocessing or feature identity differs")
        lineages[lineage] = loaded
    records = [records[key] for key in sorted(records)]
    check_files(root, records)
    return records, lineages


def check_outcome(outcome, assignment, selected, gate):
    check_signed(outcome)
    if (outcome.get("schema") != "wsp-live-outcome/1" or outcome.get("assigned") is not True
            or any(outcome.get(key) != assignment[key] for key in assigned_specs()[0])
            or outcome.get("terminal_status") != selected["terminal_status"]
            or outcome.get("no_op") != selected["no_op"]
            or outcome.get("selection_sha256") != selected["sha256"]
            or outcome.get("global_seal_sha256") != gate["sha256"]
            or outcome.get("deterministic_sha256") != digest(outcome["deterministic"])
            or outcome["deterministic"].get("assignment_id") != assignment["assignment_id"]
            or outcome["deterministic"].get("selection_sha256") != selected["sha256"]
            or outcome["deterministic"].get("global_seal_sha256") != gate["sha256"]):
        raise ValueError("Evaluation outcome binding differs from its frozen assignment")
    return outcome


def run_evaluation(root, lock, gate, replay=False):
    root = root.resolve()
    verified = verify_preparation(root)
    if canonical_bytes(lock) != canonical_bytes(verified):
        raise ValueError("Evaluation received a different preparation lock")
    require_snapshot(root, lock)
    authenticated = validate_gate(root, check_signed(gate)["sha256"])
    if canonical_bytes(gate) != canonical_bytes(authenticated):
        raise ValueError("Evaluation received a different global gate")
    inputs, lineages = evaluation_inputs(root, lock)
    binding = {"schema": "wsp-live-evaluation-started/1",
               "global_selection_seal_sha256": gate["sha256"], "prepared_lock_sha256": lock["sha256"],
               "input_files": inputs, "input_files_sha256": digest(inputs),
               "runtime": configure_runtime(), "selection_frozen": True}
    destination = root / ARTIFACT_DIR / ("replay" if replay else "evaluation")
    if replay:
        original_start = check_signed(read(root / ARTIFACT_DIR / "evaluation/evaluation-started.json"))
        if any(original_start.get(key) != value for key, value in binding.items()) or original_start.get("replay") is not False:
            raise ValueError("Replay evaluator inputs differ from the original evaluation")
        original_manifest = check_signed(read(root / ARTIFACT_DIR / "evaluation/manifest.json"))
        original_files = original_manifest.get("outcome_files", [])
        expected_paths = {(ARTIFACT_DIR / "evaluation" / item["assignment_id"] / "outcome.json").as_posix()
                          for item in lock["assignments"]}
        if (original_manifest.get("schema") != "wsp-live-study-results/1"
                or original_manifest.get("global_selection_seal_sha256") != gate["sha256"]
                or original_manifest.get("prepared_lock_sha256") != lock["sha256"]
                or original_manifest.get("protocol_sha256") != PROTOCOL_HASHES
                or original_manifest.get("assigned_total") != len(lock["assignments"])
                or original_manifest.get("evaluator_input_files_sha256") != binding["input_files_sha256"]
                or original_manifest.get("evaluation_started_sha256") != original_start["sha256"]
                or len(original_files) != len(expected_paths)
                or {item["path"] for item in original_files} != expected_paths):
            raise ValueError("Original evaluation manifest does not account for the frozen study")
        check_files(root, original_files)
        for locked in lock["assignments"]:
            assignment = check_signed(read(root / locked["assignment_path"]))
            selected = check_signed(read(selection_path(root, assignment["assignment_id"])))
            check_outcome(read(root / ARTIFACT_DIR / "evaluation" / assignment["assignment_id"] / "outcome.json"),
                          assignment, selected, gate)
    start_path = destination / "evaluation-started.json"
    if destination.exists():
        previous = check_signed(read(start_path))
        if any(previous.get(key) != value for key, value in binding.items()) or previous.get("replay") is not replay:
            raise ValueError("Interrupted evaluation has different frozen inputs")
        if (destination / "manifest.json").exists():
            raise ValueError("Completed evaluation and replay outputs are immutable")
    else:
        destination.mkdir(parents=True)
        write_json(start_path, signed(binding | {"started_utc": now(), "replay": replay}))
    outcomes, replay_records = [], []
    for locked in lock["assignments"]:
        require_snapshot(root, lock)
        assignment = check_signed(read(root / locked["assignment_path"]))
        selected = check_signed(read(selection_path(root, assignment["assignment_id"])))
        loaded = lineages[assignment["lineage_id"]]
        primary = group_primary(loaded["features"])
        directory = destination / assignment["assignment_id"]
        if (directory / "outcome.json").exists():
            outcome = check_outcome(read(directory / "outcome.json"), assignment, selected, gate)
        elif directory.exists():
            # Preserve partial artifacts and account failure without rerunning the intervention.
            deterministic = {"assignment_id": assignment["assignment_id"], "selection_sha256": selected["sha256"],
                             "global_seal_sha256": gate["sha256"], "evaluation_missing": True,
                             "joint_useful_success": False, "error_type": "InterruptedEvaluation"}
            outcome = signed({"schema": "wsp-live-outcome/1", **{k: assignment[k] for k in assigned_specs()[0]},
                              "assigned": True, "terminal_status": selected["terminal_status"], "no_op": selected["no_op"],
                              "deterministic": deterministic, "deterministic_sha256": digest(deterministic),
                              "selection_sha256": selected["sha256"], "global_seal_sha256": gate["sha256"],
                              "missingness_sensitivity": {"gain": [-1., 1.], "harm": [-1., 1.], "useful": [0., 1.]}})
            write_json(directory / "outcome.json", outcome)
        else:
            outcome = safe_evaluation(root, assignment, selected, gate, loaded, primary, directory)
        outcomes.append(outcome)
        if replay:
            original = check_signed(read(root / ARTIFACT_DIR / "evaluation" / assignment["assignment_id"] / "outcome.json"))
            replay_records.append({"assignment_id": assignment["assignment_id"],
                                   "original_deterministic_sha256": original["deterministic_sha256"],
                                   "replay_deterministic_sha256": outcome["deterministic_sha256"],
                                   "matches": original["deterministic_sha256"] == outcome["deterministic_sha256"],
                                   "evaluation_missing": outcome["deterministic"]["evaluation_missing"]})
        print(json.dumps({"phase": "replay" if replay else "evaluation", "assignment_id": assignment["assignment_id"],
                          "evaluation_missing": outcome["deterministic"]["evaluation_missing"]}), flush=True)
    if replay:
        manifest = signed({"schema": "wsp-live-replay/1", "global_seal_sha256": gate["sha256"],
                           "assigned_total": len(outcomes), "records": replay_records,
                           "all_deterministic_outcomes_match": all(x["matches"] for x in replay_records),
                           "evaluation_missing_total": sum(x["evaluation_missing"] for x in replay_records),
                           "scope": "Frozen public bytes and computations; no regenerated live provider response. Timing and random opaque handles are excluded from outcome equality."})
    else:
        manifest = summarize(root, lock, gate, outcomes)
    require_snapshot(root, lock)
    check_files(root, inputs)
    manifest = signed({key: value for key, value in manifest.items() if key != "sha256"} | {
        "evaluator_input_files_sha256": binding["input_files_sha256"],
        "evaluation_started_sha256": check_signed(read(start_path))["sha256"],
        "outcome_files": [file_record(root, destination / item["assignment_id"] / "outcome.json") for item in outcomes],
    })
    write_json(destination / "manifest.json", manifest)
    return manifest


def sign_flip(values):
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    flipped = [float((values * signs).mean()) for signs in itertools.product((-1, 1), repeat=len(values))]
    return {"observed_mean": observed, "two_sided_exact_sign_flip_sensitivity":
            float(np.mean(np.abs(flipped) >= abs(observed) - 1e-15)),
            "scope": "Paired-lineage exchangeability sensitivity, not a confirmatory population test."}


def lineage_summary(records, value, missing_bounds=(-1., 1.)):
    means, lower, upper = [], [], []
    for lineage in LINEAGES:
        selected = [record for record in records if record["lineage_id"] == lineage]
        measured, bounds = [], []
        for record in selected:
            result = value(record)
            if result is None:
                bounds.append(missing_bounds)
            elif isinstance(result, (tuple, list)):
                if len(result) != 2 or not missing_bounds[0] <= result[0] <= result[1] <= missing_bounds[1]:
                    raise ValueError("Metric interval is outside its declared bounds")
                bounds.append(tuple(result))
                if result[0] == result[1]:
                    measured.append(result[0])
            else:
                if not math.isfinite(result) or not missing_bounds[0] <= result <= missing_bounds[1]:
                    raise ValueError("Metric is outside its declared bounds")
                measured.append(result)
                bounds.append((result, result))
        if not selected:
            raise ValueError("Comparison is missing a frozen full lineage")
        means.append(float(np.mean(measured)) if len(measured) == len(selected) else None)
        lower.append(float(np.mean([x[0] for x in bounds])))
        upper.append(float(np.mean([x[1] for x in bounds])))
    complete = all(value is not None for value in means)
    return {"estimate": mean_interval(means) if complete else None,
            "missing_quantity_bounds": list(missing_bounds), "lineage_ids": list(LINEAGES),
            "lineage_bounds": list(zip(lower, upper)),
            "all_assigned_mean_bounds": [float(np.mean(lower)), float(np.mean(upper))],
            "sign_flip": sign_flip(means) if complete else None}


def confidence_summary(records, rubric_key=None):
    diagnosis = rubric_key is None
    coverage_key = "diagnosis_coverage" if diagnosis else "prediction_coverage"
    bounds_key = "binary_brier_bounds" if diagnosis else "binary_brier_all_assigned_bounds"

    def rubric(record):
        if record["deterministic"]["evaluation_missing"]:
            return None
        values = record["deterministic"]["rubric"]
        return values if diagnosis else values[rubric_key]

    per_lineage = []
    for lineage in LINEAGES:
        selected = [record for record in records if record["lineage_id"] == lineage]
        if not selected:
            raise ValueError("Confidence comparison is missing a frozen full lineage")
        observed, loss = 0., 0.
        for record in selected:
            values = rubric(record)
            if values is not None and values[coverage_key]:
                count = float(values[coverage_key]) * (1 if diagnosis else 2)
                observed += count
                loss += count * values["conditional_binary_brier"]
        assigned = len(selected) * (1 if diagnosis else 2)
        per_lineage.append({"lineage_id": lineage, "assigned_predictions": assigned,
                            "observed_predictions": int(observed), "observed_coverage": observed / assigned,
                            "conditional_binary_brier": loss / observed if observed else None})
    conditional = [item["conditional_binary_brier"] for item in per_lineage]
    return {"conditional_estimate": mean_interval(conditional) if all(x is not None for x in conditional) else None,
            "per_lineage": per_lineage,
            "prediction_coverage": lineage_summary(records, lambda x: None if rubric(x) is None else float(rubric(x)[coverage_key]), (0., 1.)),
            "all_assigned_brier_sensitivity": lineage_summary(records, lambda x: None if rubric(x) is None else rubric(x)[bounds_key], (0., 1.)),
            "conditioning": "Conditional Brier averages only stated predictions within each lineage; it is not all-assigned performance. Missing predictions/evaluations retain [0,1] bounds."}


def metric(record, control, key):
    deterministic = record["deterministic"]
    if deterministic["evaluation_missing"]:
        return None
    return deterministic["metrics"][control]["primary_novel_groups"][key]


def useful_value(record):
    if record["no_op"]:
        return 0.
    if record["deterministic"]["evaluation_missing"] and record["terminal_status"] == "sealed":
        return None
    return float(record["deterministic"]["joint_useful_success"])


def summarize(root, lock, gate, outcomes):
    primary_weight = [x for x in outcomes if x["study"] == "weight-access" and x["regime"] == "primary"]
    primary_diagnosis = [x for x in outcomes if x["study"] == "blind-diagnosis"]
    weight_views = {}
    for view in VIEWS:
        records = [x for x in primary_weight if x["parameter_view"] == view]
        weight_views[view] = {
            "target_gain": lineage_summary(records, lambda x: metric(x, "proposed", "target_gain")),
            "protected_harm": lineage_summary(records, lambda x: metric(x, "proposed", "protected_harm")),
            "joint_useful_coverage": lineage_summary(records, useful_value, (0., 1.)),
            "read_accuracy": lineage_summary(records, lambda x: None if x["deterministic"]["evaluation_missing"] else x["deterministic"]["rubric"]["read"]["all_assigned_accuracy"], (0., 1.)),
            "prediction_coverage": lineage_summary(records, lambda x: None if x["deterministic"]["evaluation_missing"] else x["deterministic"]["rubric"]["read"]["prediction_coverage"], (0., 1.)),
            "competing_accuracy": lineage_summary(records, lambda x: None if x["deterministic"]["evaluation_missing"] else x["deterministic"]["rubric"]["competing_code"]["all_assigned_accuracy"], (0., 1.)),
            "read_confidence": confidence_summary(records, "read"),
            "competing_confidence": confidence_summary(records, "competing_code"),
        }
    contrasts = []
    for exact in [x for x in primary_weight if x["parameter_view"] == "exact"]:
        other = next(x for x in primary_weight if x["parameter_view"] == "none"
                     and (x["lineage_id"], x["target_name"], x["repeat"]) ==
                         (exact["lineage_id"], exact["target_name"], exact["repeat"]))
        gain, comparison = metric(exact, "proposed", "target_gain"), metric(other, "proposed", "target_gain")
        contrasts.append(exact | {"paired_value": None if gain is None or comparison is None else gain - comparison})
    weight_contrast = lineage_summary(contrasts, lambda x: x["paired_value"], (-2., 2.))
    diagnosis_contrast = lineage_summary(primary_diagnosis, lambda x:
        None if x["deterministic"]["evaluation_missing"] or not x["deterministic"]["control_matching"]["shuffled"]["matches_frozen_tolerance"]
        else metric(x, "proposed", "target_gain") - metric(x, "shuffled", "target_gain"), (-2., 2.))
    diagnosis_views = {}
    for view in VIEWS:
        records = [x for x in primary_diagnosis if x["parameter_view"] == view]
        diagnosis_views[view] = {
            "target_gain": lineage_summary(records, lambda x: metric(x, "proposed", "target_gain")),
            "protected_harm": lineage_summary(records, lambda x: metric(x, "proposed", "protected_harm")),
            "joint_useful_coverage": lineage_summary(records, useful_value, (0., 1.)),
            "location_accuracy": lineage_summary(records, lambda x: None if x["deterministic"]["evaluation_missing"] else float(x["deterministic"]["rubric"]["location_correct"]), (0., 1.)),
            "mechanism_accuracy": lineage_summary(records, lambda x: None if x["deterministic"]["evaluation_missing"] else float(x["deterministic"]["rubric"]["mechanism_correct"]), (0., 1.)),
            "diagnosis_coverage": lineage_summary(records, lambda x: None if x["deterministic"]["evaluation_missing"] else float(x["deterministic"]["rubric"]["diagnosis_coverage"]), (0., 1.)),
            "confidence": confidence_summary(records),
        }
    receipts = []
    for assignment in lock["assignments"]:
        path = selection_path(root, assignment["assignment_id"]).parent / "provider" / "receipt.json"
        if path.exists():
            receipts.append(read(path))
    statuses = Counter(x["terminal_status"] for x in outcomes)
    supplemental = {}
    for regime, view in itertools.product(REGIMES, ("none", "exact")):
        records = [x for x in outcomes if x["regime"] == regime and x["parameter_view"] == view]
        supplemental[regime + "/" + view] = {
            "full_lineages": list(LINEAGES[:2]), "assigned_workload": len(records),
            "target_gain_all_assigned": [metric(x, "proposed", "target_gain") for x in records],
            "protected_harm_all_assigned": [metric(x, "proposed", "protected_harm") for x in records],
            "useful_success_all_assigned": [useful_value(x) for x in records],
            "scope": "Fixed prelude, one response, only first two training lineages; descriptive, not adaptive-tool evidence."}
    return signed({
        "schema": "wsp-live-study-results/1", "prepared_lock_sha256": lock["sha256"],
        "global_selection_seal_sha256": gate["sha256"], "protocol_sha256": PROTOCOL_HASHES,
        "assigned_total": len(outcomes), "terminal_statuses": dict(statuses),
        "evaluation_missing_total": sum(x["deterministic"]["evaluation_missing"] for x in outcomes),
        "weight_access": {"declared_primary": "Exact minus none target gain, lineage paired; retention/coverage assessed separately.",
                          "primary_contrast": weight_contrast, "views": weight_views},
        "blind_diagnosis": {"declared_primary": "Proposed minus support/norm-matched shuffled location gain; equal weighting of all five views within lineage.",
                           "primary_contrast": diagnosis_contrast, "views": diagnosis_views,
                           "joint_useful_coverage": lineage_summary(primary_diagnosis, useful_value, (0., 1.)),
                           "matching_failed_total": sum(
                               not x["deterministic"]["control_matching"]["shuffled"]["matches_frozen_tolerance"]
                               for x in primary_diagnosis if not x["deterministic"]["evaluation_missing"])},
        "supplementary": supplemental,
        "cost": {"provider_latency_seconds": [x["elapsed_seconds"] for x in receipts],
                 "provider_input_tokens": sum(x["usage"].get("input_tokens", 0) for x in receipts),
                 "provider_output_tokens": sum(x["usage"].get("output_tokens", 0) for x in receipts),
                 "provider_financial_cost_usd": None, "financial_unknown_reason": "Authenticated CLI subscription; no request charge exposed.",
                 "parameter_values_exposed": sum(check_signed(read(root / x["assignment_path"]))["parameter_values_exposed"] for x in lock["assignments"]),
                 "attempts_are_workload_not_independent_units_or_impact": True},
        "inference_scope": "Six independently initialized and trained complete encoders/heads, shared fixed public requests and one provider configuration. Small-lineage estimation only.",
        "failure_policy": "All assignments retained. Invalid, failed, interrupted and abstained candidates use unchanged study parent; no-op has useful success false.",
        "claim_boundaries": ["No external preregistration or provider-population confirmation.",
                             "No anonymous/zero-data understanding, natural backdoor detection, transformer-scale capability or production advantage.",
                             "Configured JSON/tool limits are not an OS/network hermeticity attestation.",
                             "Training lineage, editor attempt, request group and engineering check are distinct units."],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "select", "seal", "validate-gate", "evaluate", "replay"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--expected-gate-sha256")
    args = parser.parse_args()
    root = args.repo.resolve()
    if args.phase == "prepare":
        prepare(root)
        return
    lock = verify_preparation(root)
    require_snapshot(root, lock)
    if args.phase == "verify":
        result = {"prepared_lock_canonical_sha256": lock["sha256"], "prepared_lock_file_sha256": sha(root / LOCK_PATH),
                  "source_files": len(lock["source_files"]), "assigned_workload": len(lock["assignments"])}
    elif args.phase == "select":
        if args.raw is None:
            parser.error("--raw is required for private provider transport")
        result = run_selection(root, args.raw.resolve(), lock)
    elif args.phase == "seal":
        result = seal(root, lock)
    else:
        if args.expected_gate_sha256 is None:
            parser.error("--expected-gate-sha256 is required; do not infer authority from a mutable local file")
        gate = validate_gate(root, args.expected_gate_sha256)
        result = gate if args.phase == "validate-gate" else run_evaluation(root, lock, gate, replay=args.phase == "replay")
    print(json.dumps({key: value for key, value in result.items() if key not in
                      ("accounting", "assignments", "records", "cost", "weight_access", "blind_diagnosis", "supplementary")}), flush=True)


if __name__ == "__main__":
    main()
