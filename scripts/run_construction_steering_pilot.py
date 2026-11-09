"""Separately frozen, bounded formula-guided construction and steering pilot.

Preparation and self-check never call a live provider. Live selection, evaluation,
and replay require the frozen runner, explicit source-lock identity, and exclusive
destinations. No other study is imported or read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time
from typing import Any

sys.dont_write_bytecode = True
IMPORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPORT_ROOT))

import numpy as np

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.broker import ProposerBroker, WeightView
from rlcd.weight_space.live_provider import CodexProvider, parse_events, publish_attempt, write_json
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.program import apply_program
from rlcd.weight_space.sequential import (SEQUENTIAL_ARCHITECTURE, sequential_artifact,
                                         sequential_forward, validate_sequential)
from rlcd.weight_space.structured import StructuredProgram, lower_structured
from rlcd.weight_space.workflow import BudgetLimits, TaskSpec
from scripts.check_weight_space_bridge import python_runtime_identity

EXPERIMENT = Path("experiments/research/construction-steering-pilot")
ARTIFACTS = Path("artifacts/research/construction-steering-pilot")
SCRIPT = Path("scripts/run_construction_steering_pilot.py")
PREFIX = "wsp-construction-steering-"
ASSIGNED = 12
TOLERANCE = 1e-10
MAX_PARAMETER = 64.0
WEIGHT = "layers.0.weight"
BIAS = "layers.0.bias"
TERMINALS = {"sealed", "abstained", "invalid", "failed", "timeout", "interrupted",
             "harness_failed", "unattempted"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loads(text: str) -> Any:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON field")
            value[key] = item
        return value
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def read(path: Path) -> Any:
    return loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value) + b"\n")


def signed(payload: dict[str, Any]) -> dict[str, Any]:
    if "sha256" in payload:
        raise ValueError("Cannot sign an already signed payload")
    return payload | {"sha256": digest(payload)}


def verify_signed(value: Any, schema: str, expected: str | None = None) -> dict[str, Any]:
    if (type(value) is not dict or value.get("schema") != schema
            or type(value.get("sha256")) is not str
            or not re.fullmatch(r"[a-f0-9]{64}", value["sha256"])
            or digest({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]
            or (expected is not None and value["sha256"] != expected)):
        raise ValueError("Signed identity differs: " + schema)
    return value


def inventory(root: Path, exclude: str | None = None) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Inventory must contain local ordinary files")
        if path.is_file() and path.relative_to(root).as_posix() != exclude:
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def verify_inventory(root: Path, expected: dict[str, str], exclude: str | None = None) -> None:
    if (type(expected) is not dict or any(not (root / name).resolve().is_relative_to(root.resolve())
                                        for name in expected)
            or inventory(root, exclude) != expected):
        raise ValueError("Inventory has missing, changed, or additional files")


def project_closure() -> dict[str, str]:
    files = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(IMPORT_ROOT):
                files.add(path)
    return {path.relative_to(IMPORT_ROOT).as_posix(): file_hash(path) for path in sorted(files)}


def runtime() -> dict[str, Any]:
    return python_runtime_identity() | {"source_package": "source-freeze/rlcd",
        "solver": "numpy.linalg.lstsq(rcond=None), one call per unique task; no tuning"}


def template(kind: str) -> TensorArtifact:
    if kind == "creation":
        return sequential_artifact([[[0.0], [0.0]], [[0.0, 0.0]]], [[0.0, 0.0], [0.0]],
                                   ["relu", "identity"], input_names=["x"], output_names=["target"])
    return sequential_artifact([[[0.0] * 3, [0.0] * 3]], [[0.0, 0.0]], ["identity"],
                               input_names=["x0", "x1", "x2"], output_names=["target", "protected"])


def artifact_contract(kind: str) -> dict[str, Any]:
    value = template(kind)
    return {"schema": "wsp-tensors/1", "dtype": "float64", "architecture": value.architecture,
            "metadata": value.metadata,
            "tensor_shapes": {key: list(a.shape) for key, a in value.tensors.items()},
            "max_parameter_abs": MAX_PARAMETER}


def response_schema(kind: str) -> dict[str, Any]:
    field = "artifact_json" if kind == "creation" else "commands_json"
    properties = {"abstain": {"type": "boolean"}, field: {"type": "string", "maxLength": 20000},
                  "reason": {"type": "string", "maxLength": 2000}}
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def numeric_tree(value: Any) -> None:
    if type(value) is list:
        for item in value:
            numeric_tree(item)
    elif type(value) not in (int, float) or not math.isfinite(value) or abs(value) > MAX_PARAMETER:
        raise ValueError("Only bounded finite numeric coordinates are allowed")


def validate_candidate(candidate: TensorArtifact, task: dict[str, Any],
                       parent: TensorArtifact | None) -> None:
    validate_sequential(candidate)
    reference = template(task["kind"]) if parent is None else parent
    if not reference.compatible(candidate):
        raise ValueError("Candidate architecture, labels, or tensor shapes differ")
    for values in candidate.tensors.values():
        numeric_tree(values.tolist())
    if parent is not None and (not np.array_equal(candidate.tensors[BIAS], parent.tensors[BIAS])
                              or not np.array_equal(candidate.tensors[WEIGHT][1], parent.tensors[WEIGHT][1])):
        raise ValueError("Protected output and original target bias must remain exact")


def decode_response(response: Any, task: dict[str, Any]) -> Any:
    kind = task["kind"]
    field = "artifact_json" if kind == "creation" else "commands_json"
    if (type(response) is not dict or set(response) != {"abstain", field, "reason"}
            or type(response["abstain"]) is not bool
            or type(response[field]) is not str or len(response[field]) > 20000
            or type(response["reason"]) is not str or len(response["reason"]) > 2000):
        raise ValueError("Response differs from strict schema")
    value = loads(response[field])
    if response["abstain"]:
        if value != (None if kind == "creation" else []):
            raise ValueError("Abstention must have no proposal")
        return None
    if kind == "creation":
        if type(value) is not dict or type(value.get("tensors")) is not dict:
            raise ValueError("Creation requires complete artifact data")
        for coordinates in value["tensors"].values():
            numeric_tree(coordinates)
        artifact = TensorArtifact.from_dict(value)
        validate_candidate(artifact, task, None)
        return artifact
    if (type(value) is not list or len(value) != 1 or type(value[0]) is not dict
            or set(value[0]) != {"op", "tensor", "block", "values"}
            or value[0]["op"] != "replace" or value[0]["tensor"] != WEIGHT
            or canonical_bytes(value[0]["block"]) != canonical_bytes([[0, 1], [0, 3]])):
        raise ValueError("Steering permits exactly one literal target-row replacement")
    numeric_tree(value[0]["values"])
    if np.asarray(value[0]["values"]).shape != (1, 3):
        raise ValueError("Replacement has wrong shape")
    return value


def analytic_control(task: dict[str, Any], parent: TensorArtifact | None) -> TensorArtifact:
    """Algebra only: no examples, target execution, regression, or optimization."""
    p = task["parameters"]
    if task["kind"] == "creation":
        if task["family"] == "scaled_absolute":
            weights = [[[1.0], [-1.0]], [[p["a"], p["a"]]]]
            biases = [[-p["c"], p["c"]], [p["b"]]]
        else:
            weights = [[[1.0], [1.0]], [[p["a"], -p["a"]]]]
            biases = [[-p["l"], -p["u"]], [p["a"] * p["l"] + p["b"]]]
        return sequential_artifact(weights, biases, ["relu", "identity"],
                                   input_names=["x"], output_names=["target"])
    row = parent.tensors[WEIGHT][0].tolist()
    scale = sum(abs(v) for v in row) / sum(abs(v) for v in p["direction"])
    values = parent.tensors[WEIGHT].copy()
    values[0] = [scale * v for v in p["direction"]]
    return parent.replace(**{WEIGHT: values})


def target_values(task: dict[str, Any], parent: TensorArtifact | None,
                  inputs: list[list[float]]) -> list[list[float]]:
    """Declared mathematical labels; this does not execute an artifact."""
    p = task["parameters"]
    if task["kind"] == "creation":
        return [[p["a"] * (abs(x[0] - p["c"]) if task["family"] == "scaled_absolute"
                          else min(max(x[0], p["l"]), p["u"])) + p["b"]] for x in inputs]
    weights, biases = parent.tensors[WEIGHT].tolist(), parent.tensors[BIAS].tolist()
    scale = sum(abs(v) for v in weights[0]) / sum(abs(v) for v in p["direction"])
    return [[math.fsum([scale * d * v for d, v in zip(p["direction"], x)] + [biases[0]]),
             math.fsum([w * v for w, v in zip(weights[1], x)] + [biases[1]])] for x in inputs]


def fitted_control(task: dict[str, Any], parent: TensorArtifact | None,
                   training: dict[str, Any]) -> tuple[TensorArtifact, dict[str, Any]]:
    """Exactly one fixed training-only least-squares solve; no evaluator argument."""
    x, y = np.asarray(training["inputs"]), np.asarray(training["targets"])
    p = task["parameters"]
    if task["kind"] == "creation":
        if task["family"] == "scaled_absolute":
            w, b = [[1.0], [-1.0]], [-p["c"], p["c"]]
        else:
            w, b = [[1.0], [1.0]], [-p["l"], -p["u"]]
        features = np.maximum(0.0, x @ np.asarray(w).T + np.asarray(b))
        design = np.column_stack([features, np.ones(len(x))])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y[:, 0], rcond=None)
        endpoint = sequential_artifact([w, [coefficients[:2].tolist()]], [b, [float(coefficients[2])]],
                                       ["relu", "identity"], input_names=["x"], output_names=["target"])
    else:
        design = x
        coefficients, _, rank, _ = np.linalg.lstsq(design, y[:, 0] - parent.tensors[BIAS][0], rcond=None)
        weights = parent.tensors[WEIGHT].copy()
        weights[0] = coefficients
        endpoint = parent.replace(**{WEIGHT: weights})
    return endpoint, {"solver": "numpy.linalg.lstsq", "rcond": None, "solve_calls": 1,
                      "training_rows": len(x), "design_columns": 3, "rank": int(rank),
                      "training_sha256": digest(training), "heldout_rows_used": 0,
                      "hyperparameter_searches": 0}


def make_dataset(task: dict[str, Any], parent: TensorArtifact | None, role: str) -> dict[str, Any]:
    rng = np.random.default_rng(task["seeds"][role])
    if task["kind"] == "creation":
        p = task["parameters"]
        if role == "train":
            regions = ([(-4.0, p["c"], 16), (p["c"], 4.0, 16)]
                       if task["family"] == "scaled_absolute" else
                       [(-4.0, p["l"], 10), (p["l"], p["u"], 12), (p["u"], 4.0, 10)])
            inputs = [[float(v)] for lo, hi, count in regions for v in rng.uniform(lo, hi, count)]
        else:
            knots = [p["c"]] if task["family"] == "scaled_absolute" else [p["l"], p["u"]]
            values = list(rng.uniform(-4.0, 4.0, 64)) + [-4.0, 4.0]
            values += [k + delta for k in knots for delta in (-2.0 ** -10, 0.0, 2.0 ** -10)]
            inputs = [[float(v)] for v in values]
    else:
        inputs = rng.uniform(-2.0, 2.0, (24 if role == "train" else 64, 3)).tolist()
    return {"schema": PREFIX + "dataset/1", "task_id": task["task_id"], "role": role,
            "seed": task["seeds"][role], "sample_ids": [f'{task["task_id"]}:{role}:{i:03d}' for i in range(len(inputs))],
            "inputs": inputs, "targets": target_values(task, parent, inputs),
            "proposer_visibility": "never", "label_source": "frozen_declared_formula_not_artifact_execution"}


def task_factory(development: bool = False) -> list[tuple[dict[str, Any], TensorArtifact | None]]:
    offset = 30000 if development else 0
    result = []
    for index in range(9):
        task_seed = 17001 + index + offset
        rng = np.random.default_rng(task_seed)
        kind = "creation" if index < 6 else "steering"
        family = ("scaled_absolute" if index < 3 else "clipping") if index < 6 else "retained_l1_direction"
        task_id = (f"create-{family}-{index % 3 + 1}" if index < 6 else f"steer-{index - 5}")
        seeds = {"task": task_seed, "parent": None if index < 6 else 17101 + index - 6 + offset,
                 "train": 17201 + index + offset, "eval": 17301 + index + offset}
        if index < 6:
            p = {"a": float(rng.choice([-2.5, -1.75, -1.25, -0.75, 0.5, 1.5, 2.25, 2.75])),
                 "b": int(rng.integers(-12, 13)) / 8.0}
            if family == "scaled_absolute":
                p["c"] = int(rng.integers(-10, 11)) / 8.0
                objective = "Construct f(x)=a*abs(x-c)+b from the supplied formula."
            else:
                p.update(l=int(rng.integers(-12, 0)) / 8.0, u=int(rng.integers(1, 13)) / 8.0)
                objective = "Construct f(x)=a*clip(x,l,u)+b, clip(x,l,u)=min(max(x,l),u)."
            parent = None
        else:
            p = {"direction": [float(rng.choice([-3, -2, -1, 1, 2, 3])) for _ in range(3)]}
            parent_rng = np.random.default_rng(seeds["parent"])
            weight = parent_rng.choice(np.asarray([i / 8.0 for i in range(-16, 17) if i]), (2, 3)).tolist()
            bias = (parent_rng.integers(-8, 9, 2) / 8.0).tolist()
            parent = sequential_artifact([weight], [bias], ["identity"],
                                        input_names=["x0", "x1", "x2"], output_names=["target", "protected"])
            objective = ("Replace output0 coefficient row w with d*sum(abs(w))/sum(abs(d)); retain output0 bias "
                         "and all output1 coefficients and bias exactly. The task is supplied, not inferred.")
        task = {"schema": PREFIX + "task/1", "task_id": task_id, "kind": kind, "family": family,
                "objective": objective, "parameters": p, "seeds": seeds,
                "parent_sha256": parent.sha256 if parent else None, "artifact_contract": artifact_contract(kind)}
        result.append((task, parent))
    all_seeds = [v for task, _ in result for v in task["seeds"].values() if v is not None]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Task, parent, train, and evaluator seeds must be disjoint")
    return result


def make_broker(task: dict[str, Any], parent: TensorArtifact | None, view: str):
    context = {"parameters": task["parameters"], "artifact_contract": task["artifact_contract"],
               "labeled_examples": 0, "candidate_feedback": "none", "target_execution": "not_permitted",
               "fitting": "not_permitted_in_this_live_lane", "task_semantics": "explicitly_supplied"}
    broker = ProposerBroker(TaskSpec(task["task_id"], "create" if parent is None else "edit",
        task["objective"], SEQUENTIAL_ARCHITECTURE, parent.sha256 if parent else None),
        base=parent, view=WeightView(view), public_context=context,
        limits=BudgetLimits(proposals=1, tool_calls=8, weight_values=8, feedback_calls=0,
                            feedback_examples=0, execution_calls=0, execution_examples=0,
                            input_tokens=200000, output_tokens=20000),
        max_requests=8, max_request_bytes=25000)
    public = {"metadata": broker.request({"op": "metadata"}), "weights": None}
    if view == "exact":
        public["weights"] = {key: broker.request({"op": "weights", "tensor": key})["values"]
                             for key in sorted(parent.tensors)}
    return broker, public


def prompt_text(task: dict[str, Any], public: dict[str, Any]) -> str:
    common = ("Return only the strict response-schema JSON object. You have one response and at most one proposal. "
              "No retry, correction, tools, code execution, target forward pass, fitting, labeled examples, "
              "candidate feedback, or evaluator information is available. Internal model arithmetic is unobserved, "
              "so this is not an attestation of semantic no-inference. The harness submits your data to an opaque "
              "ProposerBroker handle and seals that handle without returning endpoint feedback. "
              "The formula, architecture, and labels are intentionally supplied; do not infer an anonymous task. ")
    if task["kind"] == "creation":
        instructions = ("This is true creation: there is no parent or inherited parameter tensor. artifact_json must "
            "encode a complete wsp-tensors/1 artifact with exactly the declared metadata and tensor shapes. "
            "All tensor entries must be finite JSON numbers with absolute value at most 64.0. "
            "Implement the supplied formula using the fixed 1-to-2 ReLU-to-1 identity architecture. "
            "For abstention set abstain=true and artifact_json to the string null. Otherwise abstain=false. ")
    else:
        instructions = ("This is steering of an existing 3-input, 2-output affine head. The desired direction d is supplied. "
            "Set the target row to d multiplied by the original target row's L1 coefficient norm divided by sum(abs(d)). "
            "The original norm is not separately disclosed. Preserve the original target bias and entire protected output. "
            'commands_json must encode exactly one command with op=replace, tensor=layers.0.weight, '
            'block=[[0,1],[0,3]], and values=[[v0,v1,v2]]. Only literal numbers are accepted; no expressions. '
            "No other operation or block is accepted. For abstention set abstain=true and commands_json to the string []. "
            "Otherwise abstain=false. ")
    return common + instructions + "\nFrozen semantic context and parameter view:\n" + canonical_bytes(public).decode("utf-8")


def protocol(development: bool) -> dict[str, Any]:
    return {"schema": PREFIX + "protocol/1", "development_fixture": development,
        "assigned_provider_calls": ASSIGNED, "maximum_live_calls": ASSIGNED, "retry_rule": "none",
        "provider": CodexProvider().configuration(), "execution": "one sequential call per assignment",
        "conditions": {"formula_guided_creation": 6, "zero_labeled_example_steering_none": 3,
                       "zero_labeled_example_steering_exact": 3},
        "estimands": {"creation": "Formula-to-weight construction without any parent; no weight-view comparison",
                      "steering": "Within-parent paired none/exact weight-access comparison under the same supplied rule"},
        "generalization_scope": "Held-out parameter variants and independent synthetic parent seeds within supplied families; not unseen rule families or trained-model lineages",
        "success": {"creation": "Structural validity and max absolute target error <=1e-10 on frozen evaluator points",
                    "steering": "Structural validity, target max absolute error <=1e-10, protected error <=1e-10, original target bias and protected row/bias exactly equal",
                    "absolute_tolerance": TOLERANCE, "missing_candidate_success": False},
        "controls": {"analytic_no_fitting": "Direct two-hinge algebra or sums/absolute values/ratios on parent row; no examples, target execution, solver, gradients, or tuning",
                     "fitting_permitted": "One numpy.linalg.lstsq(rcond=None) solve on frozen train-only labels per unique task; creation uses two fixed hinge features and an output bias, steering fits three target coefficients while retaining all original biases and protected coefficients",
                     "fitting_budget": {"unique_tasks": 9, "solve_calls_per_task": 1, "creation_training_rows": 32,
                                        "steering_training_rows": 24, "design_columns": 3, "searches": 0},
                     "unchanged_parent": "Steering-only separately reported baseline; no fallback candidate credited for missing live proposals"},
        "broker_budget": {"proposals": 1, "max_requests": 8, "weight_values": 8,
                          "input_tokens": 200000, "output_tokens": 20000,
                          "token_limit_scope": "Post-response broker accounting ceilings, not transport token limits"},
        "freeze": "Prompts, response schemas, tasks, parent arrays, train/evaluator arrays and identities, controls algorithm, actual imported project source closure, and interpreter/NumPy runtime before any live call",
        "evaluation_gate": "All twelve terminal selections, including failed, invalid, interrupted, and unattempted, are sealed before any assigned endpoint scoring or control solve",
        "accounting": "All twelve assignments remain in denominator; absent usage is unknown, reported token totals are lower bounds; cached input is not added to input",
        "interruption": "SIGINT defers to the bounded active call; stop before the next call. Unexpected BaseException retains current interrupted/failed row and unattempted remainder. No resume/retry.",
        "stop_rule": "Unexpected harness failure, interruption request, or timeout without attested cleanup stops later calls",
        "disclosures": ["Human-authored formulas, architecture, labels, and harness are supplied before trial; no midtrial help",
                        "Tool-free CLI configuration and observed events are recorded; no OS hermeticity or hidden computation attestation",
                        "Requested model and effort are recorded; deployment revision and CLI system scaffolding are not exported",
                        "Raw provider events and stderr remain private; public events are publish_attempt derivatives",
                        "Hash-bound replay establishes retained-data consistency, not independent provider provenance",
                        "Small synthetic pilot does not establish general model capability, theorem-level error bounds, or end-to-end learned construction"]}


def prepare(repo: Path, *, development: bool = False) -> dict[str, Any]:
    experiment = repo / EXPERIMENT
    if experiment.exists() or (repo / ARTIFACTS).exists():
        raise ValueError("Preparation requires new experiment and artifact paths")
    closure = project_closure()
    if any(not (repo / name).is_file() or file_hash(repo / name) != expected for name, expected in closure.items()):
        raise ValueError("Source mirrors do not match the actual imported project closure")
    runtime_identity = runtime()
    tasks = task_factory(development)
    prepared = []
    assignments = []
    for task, parent in tasks:
        datasets = {role: make_dataset(task, parent, role) for role in ("train", "eval")}
        if {tuple(x) for x in datasets["train"]["inputs"]} & {tuple(x) for x in datasets["eval"]["inputs"]}:
            raise ValueError("Training and evaluator inputs overlap")
        prepared.append((task, parent, datasets))
        views = ["none"] if task["kind"] == "creation" else (["none", "exact"] if task["task_id"] != "steer-2" else ["exact", "none"])
        pair_public = []
        for view in views:
            _, public = make_broker(task, parent, view)
            assignment_id = task["task_id"] if task["kind"] == "creation" else task["task_id"] + "-" + view
            assignment = {"assignment_id": assignment_id, "task_id": task["task_id"], "kind": task["kind"],
                          "view": view, "order": len(assignments), "prompt_path": f"prompts/{assignment_id}.txt",
                          "public_context_path": f"prompts/{assignment_id}.json",
                          "response_schema_path": f'schemas/{task["kind"]}.json'}
            assignments.append(assignment | {"prompt": prompt_text(task, public), "public": public})
            pair_public.append({"metadata": {k: v for k, v in public["metadata"].items() if k != "parameter_view"}})
        if len(pair_public) == 2 and pair_public[0] != pair_public[1]:
            raise ValueError("Steering semantic context differs across views")
    if len(assignments) != ASSIGNED:
        raise ValueError("Assignment denominator differs")
    experiment.mkdir(parents=True, exist_ok=False)
    for task, parent, datasets in prepared:
        folder = experiment / "tasks" / task["task_id"]
        write_new(folder / "task.json", task)
        if parent is not None:
            write_new(folder / "parent.json", parent.to_dict())
        for role, value in datasets.items():
            write_new(folder / (role + ".json"), value)
    for assignment in assignments:
        path = experiment / assignment["prompt_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(assignment["prompt"].encode("utf-8"))
        write_new(experiment / assignment["public_context_path"], assignment["public"])
    for kind in ("creation", "steering"):
        write_new(experiment / "schemas" / (kind + ".json"), response_schema(kind))
    frozen_protocol = signed(protocol(development))
    write_new(experiment / "protocol.json", frozen_protocol)
    write_new(experiment / "assignments.json", [{k: v for k, v in row.items() if k not in {"prompt", "public"}} for row in assignments])
    write_new(experiment / "runtime.json", runtime_identity)
    for name in closure:
        target = experiment / "source-freeze" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((IMPORT_ROOT / name).read_bytes())
    if project_closure() != closure:
        raise ValueError("Source changed during preparation")
    lock = signed({"schema": PREFIX + "lock/1", "protocol_sha256": frozen_protocol["sha256"],
                   "created_utc": now(), "assigned": ASSIGNED, "source_files_sha256": closure,
                   "files_sha256": inventory(experiment), "provider_configuration": CodexProvider().configuration(),
                   "preparation_endpoint_scores": 0, "preparation_control_solves": 0,
                   "fresh_provider_calls": 0})
    write_new(experiment / "protocol-lock.json", lock)
    return {"status": "prepared", "lock_sha256": lock["sha256"], "protocol_sha256": frozen_protocol["sha256"],
            "lock_file_sha256": file_hash(experiment / "protocol-lock.json"), "source_files": len(closure),
            "assigned": ASSIGNED, "fresh_provider_calls": 0, "assigned_endpoint_scores": 0,
            "lock_path": str(EXPERIMENT / "protocol-lock.json")}


def validate_lock(repo: Path, expected: str) -> dict[str, Any]:
    experiment = repo / EXPERIMENT
    lock = verify_signed(read(experiment / "protocol-lock.json"), PREFIX + "lock/1", expected)
    verify_inventory(experiment, lock["files_sha256"], "protocol-lock.json")
    frozen_protocol = verify_signed(read(experiment / "protocol.json"), PREFIX + "protocol/1", lock["protocol_sha256"])
    closure = project_closure()
    if closure != lock["source_files_sha256"]:
        raise ValueError("Actual imported project source closure differs")
    for name, identity in closure.items():
        if (file_hash(repo / name) != identity
                or file_hash(experiment / "source-freeze" / name) != identity):
            raise ValueError("Current or frozen source mirror differs")
    if runtime() != read(experiment / "runtime.json"):
        raise ValueError("Interpreter or NumPy runtime differs")
    if (lock["provider_configuration"] != CodexProvider().configuration()
            or frozen_protocol["provider"] != lock["provider_configuration"]
            or lock["assigned"] != ASSIGNED
            or frozen_protocol != signed(protocol(frozen_protocol["development_fixture"]))):
        raise ValueError("Frozen protocol or provider configuration differs")
    assignments = read(experiment / "assignments.json")
    if (len(assignments) != ASSIGNED or len({a["assignment_id"] for a in assignments}) != ASSIGNED
            or [a["order"] for a in assignments] != list(range(ASSIGNED))):
        raise ValueError("Assignment order or denominator differs")
    return lock


def require_frozen_runner(repo: Path) -> None:
    if Path(__file__).resolve() != (repo / EXPERIMENT / "source-freeze" / SCRIPT).resolve():
        raise ValueError("Live, evaluation, and replay must run the frozen source snapshot")


def load_task(repo: Path, task_id: str):
    folder = repo / EXPERIMENT / "tasks" / task_id
    task = read(folder / "task.json")
    parent = TensorArtifact.from_dict(read(folder / "parent.json")) if (folder / "parent.json").exists() else None
    if task["parent_sha256"] != (parent.sha256 if parent else None):
        raise ValueError("Parent identity differs")
    return task, parent


def normalized_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    handles = {}
    result = []
    for index, event in enumerate(trace):
        verify_signed(event, "wsp-broker-event/1")
        if event["sequence"] != index or event["previous_sha256"] != (trace[index - 1]["sha256"] if index else None):
            raise ValueError("Broker trace chain differs")
        request, response = loads(canonical_bytes(event["request"]).decode()), loads(canonical_bytes(event["response"]).decode())
        for value in (request, response):
            handle = value.get("candidate_handle")
            if handle is not None:
                if type(handle) is not str or not re.fullmatch(r"[a-f0-9]{48}", handle):
                    raise ValueError("Candidate handle is not opaque")
                if handle not in handles:
                    if value is not response or response.get("status") != "submitted":
                        raise ValueError("Unissued candidate handle")
                    handles[handle] = f"proposal-{len(handles)}"
                value["candidate_handle"] = handles[handle]
        result.append({"request": request, "response": response})
    return result


def selection_state(task: dict[str, Any], parent: TensorArtifact | None, view: str,
                    receipt: dict[str, Any] | None, attempted: bool,
                    harness_error: dict[str, Any] | None = None):
    broker, _ = make_broker(task, parent, view)
    provider_status = receipt["status"] if receipt else ("unknown" if attempted else "unattempted")
    candidate, error_type = None, None
    if harness_error is not None:
        terminal = "interrupted" if harness_error["exception_type"] in {"KeyboardInterrupt", "SystemExit"} else "harness_failed"
    elif not attempted:
        terminal = "unattempted"
    elif receipt is None:
        raise ValueError("An attempted assignment needs a receipt or a harness error")
    elif provider_status != "completed":
        terminal = provider_status
    else:
        try:
            decoded = decode_response(receipt["response"], task)
            usage = receipt.get("usage", {})
            if all(type(usage.get(key)) is int and usage[key] >= 0 for key in ("input_tokens", "output_tokens")):
                broker.record_usage(usage["input_tokens"], usage["output_tokens"])
            if decoded is None:
                terminal = "abstained"
            else:
                if parent is None:
                    request = {"op": "create", "artifact": decoded.to_dict()}
                    candidate = decoded
                else:
                    intent = receipt["response"]["reason"].strip() or "Frozen literal target-row steering"
                    request = {"op": "submit", "commands": decoded, "intent": intent}
                    proposal = StructuredProgram(parent.sha256, decoded, intent)
                    candidate = apply_program(parent, lower_structured(parent, proposal).program)
                    validate_candidate(candidate, task, parent)
                offered = broker.request(request)
                if offered.get("status") != "submitted":
                    raise ValueError("Proposal was denied")
                if broker.request({"op": "seal", "candidate_handle": offered["candidate_handle"]}).get("status") != "sealed":
                    raise ValueError("Selection seal was denied")
                terminal = "sealed"
        except (ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as error:
            terminal, error_type, candidate = "invalid", type(error).__name__, None
    broker.abstain()
    if terminal not in TERMINALS:
        raise ValueError("Unknown terminal selection")
    trace = broker.public_trace
    stable = {"terminal_status": terminal, "provider_status": provider_status,
              "attempted": attempted, "harness_error": harness_error, "validation_error_type": error_type,
              "candidate": candidate.to_dict() if candidate else None,
              "candidate_sha256": candidate.sha256 if candidate else None,
              "no_op": candidate.sha256 == parent.sha256 if candidate and parent else None,
              "normalized_broker_trace": normalized_trace(trace)}
    return stable, trace


def verify_receipt(receipt: Any, prompt: str, schema: dict[str, Any],
                   configuration: dict[str, Any]) -> dict[str, Any]:
    verify_signed(receipt, "wsp-live-provider-attempt/1")
    expected_fields = {"schema", "started_utc", "status", "error", "returncode", "timeout_cleanup_complete",
        "elapsed_seconds", "invocation", "response", "usage", "unapproved_events", "event_errors",
        "diagnostic_items", "events_sha256", "cost", "prompt_scope", "restrictions_scope", "sha256"}
    invocation = receipt.get("invocation", {})
    if (set(receipt) != expected_fields or receipt["status"] not in {"completed", "failed", "timeout"}
            or invocation.get("configuration") != configuration
            or invocation.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or invocation.get("schema_sha256") != digest(schema)
            or invocation.get("cwd_role") != "isolated_proposer_directory"
            or not re.fullmatch(r"[a-f0-9]{64}", invocation.get("local_invocation_sha256", ""))
            or type(receipt["usage"]) is not dict
            or type(receipt["unapproved_events"]) is not list or type(receipt["event_errors"]) is not list
            or type(receipt["diagnostic_items"]) is not int or receipt["diagnostic_items"] < 0
            or type(receipt["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(receipt["elapsed_seconds"]) or receipt["elapsed_seconds"] < 0
            or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("events_sha256", ""))):
        raise ValueError("Provider receipt does not bind the frozen invocation")
    for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
        if key in receipt["usage"] and (type(receipt["usage"][key]) is not int or receipt["usage"][key] < 0):
            raise ValueError("Invalid provider token accounting")
    if receipt["status"] == "completed" and (
            receipt["returncode"] != 0 or receipt["error"] is not None or not receipt["usage"]
            or receipt["response"] is None or receipt["unapproved_events"] or receipt["event_errors"]
            or not re.fullmatch(r"[a-f0-9]{64}", invocation.get("executable_sha256") or "")
            or type(invocation.get("cli_version")) is not str or not invocation["cli_version"].strip()):
        raise ValueError("Completed provider receipt contradicts transport evidence")
    if receipt["status"] == "timeout" and receipt["timeout_cleanup_complete"] not in (None, True, False):
        raise ValueError("Unknown timeout cleanup attestation")
    return receipt


def verify_attempt(folder: Path, prompt: str, schema: dict[str, Any],
                   configuration: dict[str, Any]) -> dict[str, Any]:
    if set(inventory(folder)) != {"request.json", "response-schema.json", "receipt.json",
                                  "events.jsonl", "event-publication.json"}:
        raise ValueError("Public provider attempt has missing or additional files")
    if (read(folder / "request.json") != {"prompt": prompt, "configuration": configuration}
            or read(folder / "response-schema.json") != schema):
        raise ValueError("Published request or response schema differs from the freeze")
    receipt = verify_receipt(read(folder / "receipt.json"), prompt, schema, configuration)
    publication = read(folder / "event-publication.json")
    if (set(publication) != {"schema", "raw_events_sha256", "public_events_sha256", "redactions",
                             "transform", "raw_bytes_published", "scope"}
            or publication["schema"] != "wsp-public-event-derivative/1"
            or publication["raw_events_sha256"] != receipt["events_sha256"]
            or publication["public_events_sha256"] != file_hash(folder / "events.jsonl")
            or type(publication["redactions"]) is not int or publication["redactions"] < 0
            or publication["raw_bytes_published"] is not False
            or publication["transform"] != "JSON_reserialization_and_user_profile_prefix_redaction"
            or publication["scope"] != "public_derivative_replay;_raw_identity_disclosed_not_publicly_replayed"):
        raise ValueError("Public event derivative binding differs")
    event_text = (folder / "events.jsonl").read_text(encoding="utf-8")
    response, usage, unapproved, errors, diagnostics = parse_events(event_text)
    if (response != receipt["response"] or usage != receipt["usage"]
            or unapproved != receipt["unapproved_events"] or errors != receipt["event_errors"]
            or len(diagnostics) != receipt["diagnostic_items"]):
        raise ValueError("Public derivative does not reproduce retained receipt fields")
    # A completed final response also has to survive duplicate/nonfinite-rejecting JSON parsing.
    if receipt["status"] == "completed":
        final = None
        for line in event_text.splitlines():
            event = loads(line)
            item = event.get("item")
            if event.get("type") == "item.completed" and type(item) is dict and item.get("type") == "agent_message":
                final = item.get("text")
        if type(final) is not str or loads(final) != response:
            raise ValueError("Completed response encoding differs")
    return receipt


@contextmanager
def deferred_sigint():
    state = {"requested": False}
    previous = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame):
        state["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, previous)


def runner_guard(repo: Path, *, fixture: bool) -> None:
    development = read(repo / EXPERIMENT / "protocol.json")["development_fixture"]
    if fixture:
        if development is not True:
            raise ValueError("Internal offline mode requires disjoint development fixtures")
    else:
        require_frozen_runner(repo)
        if development is not False:
            raise ValueError("Public execution refuses development fixtures")


def terminal_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["state"]["terminal_status"] for row in rows)
    return {key: counts[key] for key in sorted(TERMINALS)}


def select(repo: Path, expected_lock: str, private_root: Path, *, provider=None,
           fixture: bool = False) -> dict[str, Any]:
    """One fixed sequence, no retries or resumption; do not score any endpoints."""
    lock = validate_lock(repo, expected_lock)
    runner_guard(repo, fixture=fixture)
    experiment, artifact_root = repo / EXPERIMENT, repo / ARTIFACTS
    private_root = private_root.resolve()
    if (artifact_root.exists() or private_root.exists() or private_root.is_symlink()
            or private_root.is_relative_to(repo.resolve())):
        raise ValueError("Selection requires new public/private destinations with private storage outside the repository")
    provider = CodexProvider() if provider is None else provider
    if provider.configuration() != lock["provider_configuration"]:
        raise ValueError("Provider configuration differs from the freeze")
    assignments = read(experiment / "assignments.json")
    artifact_root.mkdir(parents=True, exist_ok=False)
    private_root.mkdir(parents=True, exist_ok=False)
    selection = artifact_root / "selection"
    selection.mkdir()
    start = signed({"schema": PREFIX + "selection-start/1", "started_utc": now(),
        "lock_sha256": lock["sha256"], "lock_file_sha256": file_hash(experiment / "protocol-lock.json"),
        "protocol_sha256": lock["protocol_sha256"], "assignment_ids": [a["assignment_id"] for a in assignments],
        "provider_configuration": lock["provider_configuration"], "mode": "offline_fixture" if fixture else "live",
        "provider_invocation_limit": ASSIGNED, "retries": 0, "resume_allowed": False})
    write_new(selection / "start.json", start)
    ledger = {"schema": PREFIX + "selection-ledger/1", "start_sha256": start["sha256"],
        "complete": False, "stop": None, "rows": [
            {"assignment_id": a["assignment_id"], "phase": "not_started", "attempted": False,
             "terminal_status": "unattempted", "row_sha256": None} for a in assignments]}
    write_new(selection / "ledger.json", signed(ledger))
    rows, stop = [], None
    with deferred_sigint() as interruption:
        for assignment in assignments:
            index, assignment_id = assignment["order"], assignment["assignment_id"]
            folder = selection / "assignments" / assignment_id
            folder.mkdir(parents=True, exist_ok=False)
            task, parent = load_task(repo, assignment["task_id"])
            prompt = (experiment / assignment["prompt_path"]).read_text(encoding="utf-8")
            schema = read(experiment / assignment["response_schema_path"])
            if interruption["requested"] and stop is None:
                stop = {"after_order": index - 1, "reason": "sigint_requested"}
            attempted, receipt, harness_error, scope = stop is None, None, None, "absent"
            stage = "before_invocation"
            if attempted:
                checkpoint = signed({"schema": PREFIX + "invocation-checkpoint/1", "created_utc": now(),
                    "assignment_id": assignment_id, "order": index, "lock_sha256": lock["sha256"],
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "response_schema_sha256": digest(schema), "provider_configuration": lock["provider_configuration"],
                    "attempt_number": 1, "scope": "about_to_enter_provider_call;_provider_acceptance_unobserved"})
                write_new(folder / "invocation.json", checkpoint)
                ledger["rows"][index].update(phase="in_flight", attempted=True, terminal_status=None)
                write_json(selection / "ledger.json", signed(ledger))
                private_attempt = private_root / "attempts" / assignment_id
                try:
                    stage = "provider_call"
                    returned = provider.call(prompt, directory=private_attempt,
                        cwd=private_root / "proposer-cwd" / assignment_id, schema=schema)
                    stage = "receipt_validation"
                    retained = verify_receipt(returned, prompt, schema, lock["provider_configuration"])
                    if (read(private_attempt / "receipt.json") != retained
                            or read(private_attempt / "request.json") != {"prompt": prompt, "configuration": lock["provider_configuration"]}
                            or read(private_attempt / "response-schema.json") != schema
                            or file_hash(private_attempt / "events.jsonl") != retained["events_sha256"]):
                        raise ValueError("Private receipt or frozen invocation differs")
                    receipt = retained
                    write_new(folder / "receipt.json", receipt)
                    scope = "receipt_only"
                    stage = "event_publication"
                    publish_attempt(private_attempt, folder / "provider")
                    if verify_attempt(folder / "provider", prompt, schema, lock["provider_configuration"]) != receipt:
                        raise ValueError("Published receipt differs")
                    scope = "verified_public_derivative"
                    stage = "selection_seal"
                    state, trace = selection_state(task, parent, assignment["view"], receipt, True)
                except BaseException as error:
                    harness_error = {"stage": stage, "exception_type": type(error).__name__,
                                     "worker_cleanup_attested": False}
                    state, trace = selection_state(task, parent, assignment["view"], receipt, True, harness_error)
                    stop = {"after_order": index,
                            "reason": "interrupted" if state["terminal_status"] == "interrupted" else "harness_failure"}
                if stop is None and receipt is not None and receipt["status"] == "timeout" and receipt["timeout_cleanup_complete"] is not True:
                    stop = {"after_order": index, "reason": "timeout_cleanup_not_attested"}
                if stop is None and interruption["requested"]:
                    stop = {"after_order": index, "reason": "sigint_requested"}
            else:
                state, trace = selection_state(task, parent, assignment["view"], None, False)
            row = signed({"schema": PREFIX + "selection/1", "assignment": assignment,
                "lock_sha256": lock["sha256"], "sealed_utc": now(), "state": state,
                "receipt_sha256": receipt["sha256"] if receipt is not None else None,
                "evidence_scope": scope, "broker_trace": trace})
            write_new(folder / "selection.json", row)
            rows.append(row)
            ledger["rows"][index].update(phase="terminal", attempted=attempted,
                terminal_status=state["terminal_status"], row_sha256=row["sha256"])
            ledger["stop"] = stop
            write_json(selection / "ledger.json", signed(ledger))
    ledger["complete"] = True
    final_ledger = signed(ledger)
    write_json(selection / "ledger.json", final_ledger)
    validate_lock(repo, expected_lock)
    gate = signed({"schema": PREFIX + "selection-gate/1", "sealed_utc": now(),
        "lock_sha256": lock["sha256"], "lock_file_sha256": file_hash(experiment / "protocol-lock.json"),
        "protocol_sha256": lock["protocol_sha256"], "start_sha256": start["sha256"],
        "ledger_sha256": final_ledger["sha256"], "assigned": ASSIGNED,
        "assignment_ids": [a["assignment_id"] for a in assignments],
        "selection_rows_sha256": {row["assignment"]["assignment_id"]: row["sha256"] for row in rows},
        "terminal_counts": terminal_counts(rows), "attempted_invocations": sum(row["state"]["attempted"] for row in rows),
        "invocation_scope": "provider_method_entered_after_checkpoint;_provider_acceptance_unobserved",
        "mode": start["mode"], "stop": stop, "endpoint_scores_before_gate": 0,
        "control_solves_before_gate": 0, "selection_files_sha256": inventory(selection)})
    write_new(artifact_root / "selection-gate.json", gate)
    validate_gate(repo, expected_lock, gate["sha256"], lock=lock)
    return {"status": "selections_sealed", "gate_sha256": gate["sha256"], "assigned": ASSIGNED,
            "attempted_invocations": gate["attempted_invocations"], "terminal_counts": gate["terminal_counts"],
            "fresh_provider_calls": 0 if fixture else gate["attempted_invocations"],
            "fresh_provider_call_scope": "method_invocations;_accepted_or_billed_requests_unobserved",
            "endpoint_scores": 0, "control_solves": 0, "stop": stop}


def validate_gate(repo: Path, expected_lock: str, expected_gate: str, *, lock=None):
    lock = validate_lock(repo, expected_lock) if lock is None else lock
    experiment, selection = repo / EXPERIMENT, repo / ARTIFACTS / "selection"
    gate = verify_signed(read(repo / ARTIFACTS / "selection-gate.json"), PREFIX + "selection-gate/1", expected_gate)
    assignments = read(experiment / "assignments.json")
    ids = [a["assignment_id"] for a in assignments]
    if (gate["lock_sha256"] != expected_lock or gate["lock_file_sha256"] != file_hash(experiment / "protocol-lock.json")
            or gate["protocol_sha256"] != lock["protocol_sha256"] or gate["assigned"] != ASSIGNED
            or gate["assignment_ids"] != ids or set(gate["selection_rows_sha256"]) != set(ids)
            or gate["endpoint_scores_before_gate"] != 0 or gate["control_solves_before_gate"] != 0):
        raise ValueError("Gate does not bind all frozen assignments")
    verify_inventory(selection, gate["selection_files_sha256"])
    start = verify_signed(read(selection / "start.json"), PREFIX + "selection-start/1", gate["start_sha256"])
    ledger = verify_signed(read(selection / "ledger.json"), PREFIX + "selection-ledger/1", gate["ledger_sha256"])
    expected_mode = "offline_fixture" if read(experiment / "protocol.json")["development_fixture"] else "live"
    if (start["lock_sha256"] != expected_lock or start["lock_file_sha256"] != gate["lock_file_sha256"]
            or start["protocol_sha256"] != lock["protocol_sha256"] or start["assignment_ids"] != ids
            or start["provider_configuration"] != lock["provider_configuration"]
            or start["provider_invocation_limit"] != ASSIGNED or start["retries"] != 0
            or start["resume_allowed"] is not False or start["mode"] != expected_mode
            or gate["mode"] != expected_mode or ledger["start_sha256"] != start["sha256"]
            or ledger["complete"] is not True or ledger["stop"] != gate["stop"] or len(ledger["rows"]) != ASSIGNED):
        raise ValueError("Selection start or complete ledger differs")
    rows, receipts = [], []
    for assignment in assignments:
        assignment_id, index = assignment["assignment_id"], assignment["order"]
        folder = selection / "assignments" / assignment_id
        row = verify_signed(read(folder / "selection.json"), PREFIX + "selection/1", gate["selection_rows_sha256"][assignment_id])
        if row["assignment"] != assignment or row["lock_sha256"] != expected_lock:
            raise ValueError("Selection assignment differs")
        state = row["state"]
        if type(state["attempted"]) is not bool:
            raise ValueError("Attempt accounting must be boolean")
        prompt = (experiment / assignment["prompt_path"]).read_text(encoding="utf-8")
        schema = read(experiment / assignment["response_schema_path"])
        receipt = None
        if state["attempted"]:
            checkpoint = verify_signed(read(folder / "invocation.json"), PREFIX + "invocation-checkpoint/1")
            if (checkpoint["assignment_id"] != assignment_id or checkpoint["order"] != index
                    or checkpoint["lock_sha256"] != expected_lock or checkpoint["attempt_number"] != 1
                    or checkpoint["prompt_sha256"] != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    or checkpoint["response_schema_sha256"] != digest(schema)
                    or checkpoint["provider_configuration"] != lock["provider_configuration"]):
                raise ValueError("Invocation checkpoint differs")
        elif (folder / "invocation.json").exists() or state["harness_error"] is not None:
            raise ValueError("Unattempted assignment has invocation evidence")
        if row["receipt_sha256"] is not None:
            receipt = verify_receipt(read(folder / "receipt.json"), prompt, schema, lock["provider_configuration"])
            if not state["attempted"] or receipt["sha256"] != row["receipt_sha256"]:
                raise ValueError("Retained receipt binding differs")
        elif (folder / "receipt.json").exists():
            raise ValueError("Unbound retained receipt")
        scope = row["evidence_scope"]
        if scope == "verified_public_derivative":
            if verify_attempt(folder / "provider", prompt, schema, lock["provider_configuration"]) != receipt:
                raise ValueError("Public provider derivative differs from retained receipt")
        elif scope in {"receipt_only", "absent"}:
            if ((scope == "receipt_only") != (receipt is not None)
                    or (state["attempted"] and state["harness_error"] is None)
                    or (not state["attempted"] and (folder / "provider").exists())):
                raise ValueError("Incomplete publication needs a stopped harness failure")
        else:
            raise ValueError("Unknown evidence scope")
        task, parent = load_task(repo, assignment["task_id"])
        rebuilt, _ = selection_state(task, parent, assignment["view"], receipt,
                                     state["attempted"], state["harness_error"])
        if (rebuilt != state or normalized_trace(row["broker_trace"]) != rebuilt["normalized_broker_trace"]
                or ledger["rows"][index] != {"assignment_id": assignment_id, "phase": "terminal",
                    "attempted": state["attempted"], "terminal_status": state["terminal_status"], "row_sha256": row["sha256"]}):
            raise ValueError("Selection or broker trace does not reconstruct")
        rows.append(row)
        receipts.append(receipt)
    attempted = sum(row["state"]["attempted"] for row in rows)
    if ([row["state"]["attempted"] for row in rows] != [True] * attempted + [False] * (ASSIGNED - attempted)
            or gate["attempted_invocations"] != attempted or gate["terminal_counts"] != terminal_counts(rows)):
        raise ValueError("Attempt order, count, or terminal denominator differs")
    stop = gate["stop"]
    if stop is None:
        if attempted != ASSIGNED:
            raise ValueError("Unattempted suffix lacks a stop record")
    elif (set(stop) != {"after_order", "reason"} or stop["after_order"] != attempted - 1
          or stop["reason"] not in {"sigint_requested", "interrupted", "harness_failure", "timeout_cleanup_not_attested"}):
        raise ValueError("Stop record differs from invocation prefix")
    for index, (row, receipt) in enumerate(zip(rows, receipts)):
        mandatory_stop = ("interrupted" if row["state"]["terminal_status"] == "interrupted" else
                          "harness_failure" if row["state"]["terminal_status"] == "harness_failed" else
                          "timeout_cleanup_not_attested" if receipt is not None and receipt["status"] == "timeout"
                          and receipt["timeout_cleanup_complete"] is not True else None)
        if mandatory_stop is not None and stop != {"after_order": index, "reason": mandatory_stop}:
            raise ValueError("Calls continued after a mandatory stop")
    if stop is not None and stop["reason"] != "sigint_requested":
        last = rows[attempted - 1]["state"] if attempted else None
        if (last is None or (stop["reason"] == "interrupted" and last["terminal_status"] != "interrupted")
                or (stop["reason"] == "harness_failure" and last["terminal_status"] != "harness_failed")
                or (stop["reason"] == "timeout_cleanup_not_attested" and last["terminal_status"] != "timeout")):
            raise ValueError("Stop cause lacks corresponding terminal evidence")
    return gate, rows, receipts


def score_endpoint(candidate: TensorArtifact | None, task: dict[str, Any],
                   parent: TensorArtifact | None, evaluation: dict[str, Any]) -> dict[str, Any]:
    result = {"status": "missing_candidate", "structural_valid": None, "success": False,
              "target_max_abs_error": None, "protected_max_abs_error": None,
              "exact_target_bias_preserved": None, "exact_protected_row_and_bias_preserved": None,
              "evaluator_rows": len(evaluation["inputs"]), "evaluation_error_type": None}
    if candidate is None:
        return result
    if parent is not None:
        result["exact_target_bias_preserved"] = bool(candidate.tensors[BIAS][0] == parent.tensors[BIAS][0])
        result["exact_protected_row_and_bias_preserved"] = bool(
            np.array_equal(candidate.tensors[WEIGHT][1], parent.tensors[WEIGHT][1])
            and candidate.tensors[BIAS][1] == parent.tensors[BIAS][1])
    try:
        validate_candidate(candidate, task, parent)
    except (ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as error:
        return result | {"status": "invalid_candidate", "structural_valid": False,
                         "evaluation_error_type": type(error).__name__}
    result["structural_valid"] = True
    try:
        actual = sequential_forward(candidate, evaluation["inputs"])
        expected = np.asarray(evaluation["targets"], dtype=np.float64)
        if actual.shape != expected.shape:
            raise ValueError("Evaluator output shape differs")
        errors = np.max(np.abs(actual - expected), axis=0)
        if not np.isfinite(errors).all():
            raise ValueError("Evaluator errors must be finite")
        result.update(status="scored", target_max_abs_error=float(errors[0]))
        if parent is not None:
            result["protected_max_abs_error"] = float(errors[1])
        result["success"] = bool(errors[0] <= TOLERANCE and (parent is None or (
            errors[1] <= TOLERANCE and result["exact_target_bias_preserved"]
            and result["exact_protected_row_and_bias_preserved"])))
    except (ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as error:
        result.update(status="evaluation_failed", evaluation_error_type=type(error).__name__)
    return result


def usage_accounting(rows: list[dict[str, Any]], receipts: list[dict[str, Any] | None]) -> dict[str, Any]:
    attempted = sum(row["state"]["attempted"] for row in rows)
    result = {"assigned": ASSIGNED, "attempted_invocations": attempted,
              "not_attempted": ASSIGNED - attempted, "units": "tokens",
              "missing_usage_is_zero": False, "cached_input_added_to_input": False,
              "totals_scope": "sums_of_observed_fields;_lower_bounds_when_coverage_is_incomplete",
              "monetary_cost": None, "monetary_cost_scope": "not_exposed_by_authenticated_CLI"}
    for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
        known = [receipt["usage"][key] for row, receipt in zip(rows, receipts)
                 if row["state"]["attempted"] and receipt is not None and key in receipt["usage"]]
        result[key] = {"observed_sum": sum(known) if known else None, "known_attempts": len(known),
                       "unknown_attempts": attempted - len(known), "complete_for_attempts": len(known) == attempted}
    return result


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    successes = sum(record["metrics"]["success"] for record in records)
    return {"denominator": total, "successes": successes, "success_rate": successes / total if total else None,
            "scored": sum(record["metrics"]["status"] == "scored" for record in records),
            "missing_candidate": sum(record["metrics"]["status"] == "missing_candidate" for record in records),
            "evaluation_failures": sum(record["metrics"]["status"] == "evaluation_failed" for record in records),
            "invalid_candidates": sum(record["metrics"]["status"] == "invalid_candidate" for record in records)}


def compute_outcomes(repo: Path, lock: dict[str, Any], gate: dict[str, Any],
                     rows: list[dict[str, Any]], receipts: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Only called after complete gate validation. Controls see training labels only."""
    experiment = repo / EXPERIMENT
    task_ids = list(dict.fromkeys(row["assignment"]["task_id"] for row in rows))
    if len(task_ids) != 9 or len(rows) != ASSIGNED or sum(gate["terminal_counts"].values()) != ASSIGNED:
        raise ValueError("Complete twelve-assignment gate is required for evaluation")
    task_data, controls, live_records = {}, [], []
    for task_id in task_ids:
        task, parent = load_task(repo, task_id)
        folder = experiment / "tasks" / task_id
        training, evaluation = read(folder / "train.json"), read(folder / "eval.json")
        task_data[task_id] = (task, parent, evaluation)
        analytic = analytic_control(task, parent)
        fitted, fitting = fitted_control(task, parent, training)
        if fitting["solve_calls"] != 1 or fitting["heldout_rows_used"] != 0:
            raise ValueError("Fitting control exceeded the frozen training-only budget")
        endpoints = {"analytic_no_fitting": analytic, "fitting_permitted": fitted}
        if parent is not None:
            endpoints["unchanged_parent"] = parent
        entries = {}
        for lane, endpoint in endpoints.items():
            entries[lane] = {"candidate": endpoint.to_dict(), "candidate_sha256": endpoint.sha256,
                             "metrics": score_endpoint(endpoint, task, parent, evaluation),
                             "no_op": endpoint.sha256 == parent.sha256 if parent else None}
        controls.append(signed({"schema": PREFIX + "control-outcome/1", "task_id": task_id,
            "kind": task["kind"], "family": task["family"], "task_sha256": digest(task),
            "parent_sha256": task["parent_sha256"], "training_sha256": digest(training),
            "evaluator_sha256": digest(evaluation), "fitting": fitting, "controls": entries,
            "selection_gate_sha256": gate["sha256"], "fresh_provider_calls": 0}))
    for row, receipt in zip(rows, receipts):
        assignment, state = row["assignment"], row["state"]
        task, parent, evaluation = task_data[assignment["task_id"]]
        candidate = TensorArtifact.from_dict(state["candidate"]) if state["candidate"] is not None else None
        live_records.append(signed({"schema": PREFIX + "assignment-outcome/1", "assignment": assignment,
            "selection_sha256": row["sha256"], "selection_gate_sha256": gate["sha256"],
            "attempted": state["attempted"], "terminal_status": state["terminal_status"],
            "provider_status": state["provider_status"], "evidence_scope": row["evidence_scope"],
            "candidate_sha256": state["candidate_sha256"], "no_op": state["no_op"],
            "usage": receipt["usage"] if receipt is not None else None,
            "evaluator_sha256": digest(evaluation), "metrics": score_endpoint(candidate, task, parent, evaluation),
            "missing_provider_evidence": receipt is None, "missing_live_candidate": candidate is None,
            "parent_fallback_used": False}))
    creation = [row for row in live_records if row["assignment"]["kind"] == "creation"]
    none = [row for row in live_records if row["assignment"]["kind"] == "steering" and row["assignment"]["view"] == "none"]
    exact = [row for row in live_records if row["assignment"]["kind"] == "steering" and row["assignment"]["view"] == "exact"]
    paired = []
    for task_id in task_ids[6:]:
        pair = {row["assignment"]["view"]: row for row in live_records if row["assignment"]["task_id"] == task_id}
        a, b = pair["none"]["metrics"], pair["exact"]["metrics"]
        paired.append({"task_id": task_id, "none_success": a["success"], "exact_success": b["success"],
                       "exact_minus_none_success": int(b["success"]) - int(a["success"]),
                       "target_error_reduction": a["target_max_abs_error"] - b["target_max_abs_error"]
                       if a["target_max_abs_error"] is not None and b["target_max_abs_error"] is not None else None})
    control_summary = {}
    for kind in ("creation", "steering"):
        control_summary[kind] = {}
        for lane in ("analytic_no_fitting", "fitting_permitted", "unchanged_parent"):
            values = [control["controls"][lane] for control in controls if control["kind"] == kind and lane in control["controls"]]
            if values:
                control_summary[kind][lane] = metric_summary(values)
    return signed({"schema": PREFIX + "deterministic-outcomes/1", "lock_sha256": lock["sha256"],
        "lock_file_sha256": file_hash(experiment / "protocol-lock.json"), "protocol_sha256": lock["protocol_sha256"],
        "selection_gate_sha256": gate["sha256"], "selection_gate_file_sha256": file_hash(repo / ARTIFACTS / "selection-gate.json"),
        "source_closure_sha256": digest(lock["source_files_sha256"]), "runtime_sha256": digest(read(experiment / "runtime.json")),
        "assignments": live_records, "controls": controls, "terminal_counts": terminal_counts(rows),
        "live_summary": {"creation_formula_to_weights": metric_summary(creation),
                         "steering_none": metric_summary(none), "steering_exact": metric_summary(exact),
                         "all_assigned": metric_summary(live_records)},
        "control_summary": control_summary, "steering_pairs": paired, "usage": usage_accounting(rows, receipts),
        "accounting": {"assigned": ASSIGNED, "unique_tasks": 9, "control_solve_calls": 9,
                       "analytic_control_scores": 9, "fitted_control_scores": 9, "unchanged_parent_scores": 3,
                       "live_candidate_scores": sum(row["state"]["candidate"] is not None for row in rows),
                       "fresh_provider_calls_during_evaluation": 0, "heldout_rows_used_for_fitting": 0},
        "interpretation": {"creation": "supplied-formula construction;_not_a_weight_view_comparison",
                           "steering": "paired_weight_access_comparison_under_identical_supplied_rule",
                           "generalization": "heldout_parameter_variants_and_independent_synthetic_parent_seeds_within_supplied_families",
                           "unseen_rule_families_tested": False, "trained_model_lineages_tested": False,
                           "independent_provider_provenance_established": False}})


def write_outcomes(folder: Path, outcomes: dict[str, Any]) -> None:
    write_new(folder / "outcomes.json", outcomes)
    for row in outcomes["assignments"]:
        write_new(folder / "assignments" / (row["assignment"]["assignment_id"] + ".json"), row)
    for control in outcomes["controls"]:
        write_new(folder / "controls" / (control["task_id"] + ".json"), control)


def evaluate(repo: Path, expected_lock: str, expected_gate: str, *, fixture: bool = False) -> dict[str, Any]:
    lock = validate_lock(repo, expected_lock)
    runner_guard(repo, fixture=fixture)
    gate, rows, receipts = validate_gate(repo, expected_lock, expected_gate, lock=lock)
    folder = repo / ARTIFACTS / "evaluation"
    if folder.exists():
        raise ValueError("Evaluation requires a new exclusive destination")
    folder.mkdir()
    started, tick = now(), time.perf_counter()
    outcomes = compute_outcomes(repo, lock, gate, rows, receipts)
    write_outcomes(folder, outcomes)
    validate_lock(repo, expected_lock)
    validate_gate(repo, expected_lock, expected_gate, lock=lock)
    manifest = signed({"schema": PREFIX + "evaluation-manifest/1", "started_utc": started,
        "finished_utc": now(), "elapsed_seconds": time.perf_counter() - tick,
        "lock_sha256": expected_lock, "selection_gate_sha256": expected_gate,
        "outcomes_sha256": outcomes["sha256"], "outcomes_file_sha256": file_hash(folder / "outcomes.json"),
        "files_sha256": inventory(folder), "fresh_provider_calls": 0})
    write_new(folder / "manifest.json", manifest)
    return {"status": "evaluated", "evaluation_sha256": manifest["sha256"], "outcomes_sha256": outcomes["sha256"],
            "assigned": ASSIGNED, "fresh_provider_calls": 0, "control_solve_calls": 9,
            "live_summary": outcomes["live_summary"]}


def validate_evaluation(repo: Path, expected_lock: str, expected_gate: str, expected_evaluation: str):
    folder = repo / ARTIFACTS / "evaluation"
    manifest = verify_signed(read(folder / "manifest.json"), PREFIX + "evaluation-manifest/1", expected_evaluation)
    if (manifest["lock_sha256"] != expected_lock or manifest["selection_gate_sha256"] != expected_gate
            or manifest["fresh_provider_calls"] != 0):
        raise ValueError("Evaluation manifest input bindings differ")
    verify_inventory(folder, manifest["files_sha256"], "manifest.json")
    outcomes = verify_signed(read(folder / "outcomes.json"), PREFIX + "deterministic-outcomes/1", manifest["outcomes_sha256"])
    if (file_hash(folder / "outcomes.json") != manifest["outcomes_file_sha256"]
            or (folder / "outcomes.json").read_bytes() != canonical_bytes(outcomes) + b"\n"
            or outcomes["lock_sha256"] != expected_lock or outcomes["selection_gate_sha256"] != expected_gate
            or len(outcomes["assignments"]) != ASSIGNED or len(outcomes["controls"]) != 9):
        raise ValueError("Deterministic outcome identity or denominator differs")
    expected_files = {"outcomes.json"}
    for row in outcomes["assignments"]:
        verify_signed(row, PREFIX + "assignment-outcome/1")
        path = "assignments/" + row["assignment"]["assignment_id"] + ".json"
        expected_files.add(path)
        if (folder / path).read_bytes() != canonical_bytes(row) + b"\n":
            raise ValueError("Assignment outcome file differs from complete payload")
    for control in outcomes["controls"]:
        verify_signed(control, PREFIX + "control-outcome/1")
        path = "controls/" + control["task_id"] + ".json"
        expected_files.add(path)
        if (folder / path).read_bytes() != canonical_bytes(control) + b"\n":
            raise ValueError("Control outcome file differs from complete payload")
    if set(manifest["files_sha256"]) != expected_files:
        raise ValueError("Evaluation inventory differs from the declared result files")
    return manifest, outcomes


def replay(repo: Path, expected_lock: str, expected_gate: str, expected_evaluation: str,
           *, fixture: bool = False) -> dict[str, Any]:
    lock = validate_lock(repo, expected_lock)
    runner_guard(repo, fixture=fixture)
    gate, rows, receipts = validate_gate(repo, expected_lock, expected_gate, lock=lock)
    manifest, original = validate_evaluation(repo, expected_lock, expected_gate, expected_evaluation)
    folder = repo / ARTIFACTS / "replay"
    if folder.exists():
        raise ValueError("Replay requires a new exclusive destination")
    folder.mkdir()
    started, tick = now(), time.perf_counter()
    rebuilt = compute_outcomes(repo, lock, gate, rows, receipts)
    write_outcomes(folder, rebuilt)
    matches = {path: file_hash(folder / path) == identity for path, identity in manifest["files_sha256"].items()}
    complete_match = canonical_bytes(original) == canonical_bytes(rebuilt)
    validate_lock(repo, expected_lock)
    validate_gate(repo, expected_lock, expected_gate, lock=lock)
    validate_evaluation(repo, expected_lock, expected_gate, expected_evaluation)
    report = signed({"schema": PREFIX + "replay-report/1", "status": "matched" if complete_match and all(matches.values()) else "mismatch",
        "started_utc": started, "finished_utc": now(), "elapsed_seconds": time.perf_counter() - tick,
        "lock_sha256": expected_lock, "selection_gate_sha256": expected_gate,
        "evaluation_manifest_sha256": expected_evaluation, "original_outcomes_sha256": original["sha256"],
        "reconstructed_outcomes_sha256": rebuilt["sha256"], "complete_canonical_payload_match": complete_match,
        "result_file_matches": matches, "files_sha256": inventory(folder), "assigned": ASSIGNED,
        "fresh_provider_calls": 0, "replayed_training_solve_calls": 9,
        "scope": "retained_public_derivative_reconstruction;_not_independent_provider_provenance"})
    write_new(folder / "report.json", report)
    return {"status": report["status"], "replay_sha256": report["sha256"], "assigned": ASSIGNED,
            "complete_canonical_payload_match": complete_match, "result_files_checked": len(matches),
            "fresh_provider_calls": 0}


@contextmanager
def replaced(target, name, value):
    original = target[name] if type(target) is dict else getattr(target, name)
    if type(target) is dict:
        target[name] = value
    else:
        setattr(target, name, value)
    try:
        yield original
    finally:
        if type(target) is dict:
            target[name] = original
        else:
            setattr(target, name, original)


@contextmanager
def changed_bytes(path: Path, content: bytes):
    original = path.read_bytes()
    path.write_bytes(content)
    try:
        yield
    finally:
        path.write_bytes(original)


class OfflineProvider:
    """Private disjoint-seed self-check transport. Never instantiated by live CLI."""

    def __init__(self, repo: Path, behaviors: list[str]):
        if read(repo / EXPERIMENT / "protocol.json")["development_fixture"] is not True:
            raise ValueError("Offline transport requires development fixtures")
        self.repo, self.behaviors, self.calls = repo, behaviors, 0

    def configuration(self):
        return CodexProvider().configuration()

    def call(self, prompt, *, directory, cwd, schema):
        index = self.calls
        self.calls += 1
        behavior = self.behaviors[index]
        if behavior == "interrupt":
            raise KeyboardInterrupt()
        if behavior == "harness_error":
            raise RuntimeError("Private synthetic harness error")
        assignment = read(self.repo / EXPERIMENT / "assignments.json")[index]
        task, parent = load_task(self.repo, assignment["task_id"])
        if task["seeds"]["task"] < 47001:
            raise ValueError("Offline transport refuses assigned seeds")
        response, usage, events = None, {}, []
        status, error, returncode, cleanup = "completed", None, 0, None
        if behavior in {"valid", "invalid", "abstain", "partial_usage", "signal"}:
            endpoint = analytic_control(task, parent)
            field = "artifact_json" if task["kind"] == "creation" else "commands_json"
            proposal = endpoint.to_dict() if parent is None else [{"op": "replace", "tensor": WEIGHT,
                "block": [[0, 1], [0, 3]], "values": [endpoint.tensors[WEIGHT][0].tolist()]}]
            response = {"abstain": False, field: canonical_bytes(proposal).decode("utf-8"), "reason": "Offline synthetic fixture"}
            if behavior == "invalid":
                response["extra"] = "forbidden"
            if behavior == "abstain":
                response.update(abstain=True)
                response[field] = "null" if parent is None else "[]"
            usage = {"input_tokens": 100 + index, "output_tokens": 10 + index, "cached_input_tokens": index}
            if behavior == "partial_usage":
                usage = {"input_tokens": 100 + index}
            events = [{"type": "item.completed", "item": {"type": "agent_message", "text": canonical_bytes(response).decode("utf-8")}},
                      {"type": "turn.completed", "usage": usage}]
        elif behavior == "failed":
            status, error, returncode = "failed", "transport_or_response_failed", 1
            events = [{"type": "turn.failed", "error": {"message": "Offline synthetic transport failure"}}]
        elif behavior in {"timeout", "unclean_timeout"}:
            status, error, returncode, cleanup = "timeout", "provider_deadline", -9, behavior == "timeout"
        else:
            raise ValueError("Unknown offline fixture behavior")
        directory.mkdir(parents=True, exist_ok=False)
        cwd.mkdir(parents=True, exist_ok=False)
        write_new(directory / "request.json", {"prompt": prompt, "configuration": self.configuration()})
        write_new(directory / "response-schema.json", schema)
        event_bytes = b"".join(canonical_bytes(event) + b"\n" for event in events)
        with (directory / "events.jsonl").open("xb") as stream:
            stream.write(event_bytes)
        with (directory / "transport.stderr.private").open("xb") as stream:
            stream.write(b"offline-fixture\n")
        _, _, unapproved, event_errors, diagnostics = parse_events(event_bytes.decode("utf-8"))
        receipt = signed({"schema": "wsp-live-provider-attempt/1", "started_utc": now(),
            "status": status, "error": error, "returncode": returncode, "timeout_cleanup_complete": cleanup,
            "elapsed_seconds": 0.0, "invocation": {"executable_sha256": digest("offline-fixture"),
                "cli_version": "offline-fixture", "configuration": self.configuration(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "schema_sha256": digest(schema),
                "cwd_role": "isolated_proposer_directory", "local_invocation_sha256": digest({"offline_fixture": True, "order": index})},
            "response": response, "usage": usage, "unapproved_events": unapproved, "event_errors": event_errors,
            "diagnostic_items": len(diagnostics), "events_sha256": hashlib.sha256(event_bytes).hexdigest(),
            "cost": {"currency": "USD", "amount": None, "reason": "offline_fixture_no_provider_cost"},
            "prompt_scope": "exact_harness_prompt;_CLI_system_scaffolding_not_exported",
            "restrictions_scope": "configured_CLI_tools_and_observed_events;_not_OS_hermeticity"})
        write_new(directory / "receipt.json", receipt)
        if behavior == "signal":
            signal.raise_signal(signal.SIGINT)
        return receipt


def self_check(repo: Path, private_root: Path) -> dict[str, Any]:
    """Exercise only +30000 fixtures; forbid the real provider throughout."""
    private_root = private_root.resolve()
    if private_root.is_relative_to(repo.resolve()):
        raise ValueError("Self-check storage must be outside the public repository")
    private_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(dir=private_root, prefix="construction-selfcheck-"))
    checks, real_calls = [], {"blocked": 0}

    def forbid(*_args, **_kwargs):
        raise AssertionError("Forbidden execution in algebra/selection self-check")

    def no_provider(*_args, **_kwargs):
        real_calls["blocked"] += 1
        raise AssertionError("Real provider is forbidden during self-check")

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejected(name, operation):
        try:
            operation()
        except (ValueError, KeyError, TypeError, FileNotFoundError, FileExistsError, OverflowError):
            checks.append(name)
        else:
            raise AssertionError(name)

    def fixture_repo(name):
        target = root / name
        target.mkdir()
        for filename in project_closure():
            path = target / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write((IMPORT_ROOT / filename).read_bytes())
        with replaced(globals(), "sequential_forward", forbid), replaced(np.linalg, "lstsq", forbid), replaced(globals(), "analytic_control", forbid):
            prepared = prepare(target, development=True)
        check(prepared["fresh_provider_calls"] == 0 and prepared["assigned_endpoint_scores"] == 0,
              name + ":preparation_no_calls_no_endpoint_execution_no_control_construction_or_solve")
        return target, prepared

    started = now()
    with replaced(CodexProvider, "call", no_provider):
        development_tasks = task_factory(True)
        check(all(value >= 47001 for task, _ in development_tasks for value in task["seeds"].values() if value is not None),
              "development_seed_namespace_is_disjoint")
        analytic_endpoints = []
        with replaced(globals(), "sequential_forward", forbid), replaced(globals(), "target_values", forbid), replaced(np.linalg, "lstsq", forbid):
            for task, parent in development_tasks:
                analytic_endpoints.append(analytic_control(task, parent))
        check(len(analytic_endpoints) == 9, "analytic_controls_do_not_execute_targets_or_use_labels_or_solver")
        for (task, parent), endpoint in zip(development_tasks, analytic_endpoints):
            validate_candidate(endpoint, task, parent)
            score = score_endpoint(endpoint, task, parent, make_dataset(task, parent, "eval"))
            check(score["success"], task["task_id"] + ":analytic_development_correct")
            training = make_dataset(task, parent, "train")
            solves = {"count": 0}
            original_solver = np.linalg.lstsq

            def counted_solver(*args, **kwargs):
                solves["count"] += 1
                if kwargs != {"rcond": None} or len(args) != 2 or args[0].shape[0] != len(training["inputs"]):
                    raise AssertionError("Training-only solver budget differs")
                return original_solver(*args, **kwargs)

            with replaced(np.linalg, "lstsq", counted_solver), replaced(globals(), "target_values", forbid), replaced(globals(), "sequential_forward", forbid):
                fitted, budget = fitted_control(task, parent, training)
            check(solves["count"] == 1 and budget["solve_calls"] == 1 and budget["heldout_rows_used"] == 0,
                  task["task_id"] + ":one_training_only_solve")
            check(score_endpoint(fitted, task, parent, make_dataset(task, parent, "eval"))["success"],
                  task["task_id"] + ":fitted_development_correct")
        creation_task, _ = development_tasks[0]
        steering_task, steering_parent = development_tasks[6]
        good_creation = {"abstain": False, "artifact_json": canonical_bytes(analytic_endpoints[0].to_dict()).decode(), "reason": "fixture"}
        good_steering = {"abstain": False, "commands_json": canonical_bytes([{"op": "replace", "tensor": WEIGHT,
            "block": [[0, 1], [0, 3]], "values": [analytic_endpoints[6].tensors[WEIGHT][0].tolist()]}]).decode(), "reason": "fixture"}
        check(isinstance(decode_response(good_creation, creation_task), TensorArtifact), "strict_creation_accepts_valid_full_artifact")
        check(type(decode_response(good_steering, steering_task)) is list, "strict_steering_accepts_only_literal_target_row")
        rejected("extra_response_fields_rejected", lambda: decode_response(good_creation | {"extra": 1}, creation_task))
        rejected("boolean_numeric_coordinate_rejected", lambda: numeric_tree([True]))
        rejected("nonfinite_json_rejected", lambda: loads('{"value":NaN}'))
        rejected("duplicate_json_fields_rejected", lambda: loads('{"value":1,"value":2}'))
        bad_artifact = analytic_endpoints[0].to_dict()
        bad_artifact["tensors"]["layers.0.weight"] = [[1.0, 1.0]]
        rejected("wrong_creation_shape_rejected", lambda: decode_response(good_creation | {"artifact_json": canonical_bytes(bad_artifact).decode()}, creation_task))
        bad_command = loads(good_steering["commands_json"])
        bad_command[0]["block"] = [[1, 2], [0, 3]]
        rejected("protected_row_edit_rejected", lambda: decode_response(good_steering | {"commands_json": canonical_bytes(bad_command).decode()}, steering_task))
        bad_command[0]["block"] = [[0, True], [0, 3]]
        rejected("boolean_block_index_rejected", lambda: decode_response(good_steering | {"commands_json": canonical_bytes(bad_command).decode()}, steering_task))
        bad_command[0]["block"] = [[0, 1], [0, 3]]
        bad_command[0]["values"] = [[0.0, 1.0]]
        rejected("wrong_steering_value_shape_rejected", lambda: decode_response(good_steering | {"commands_json": canonical_bytes(bad_command).decode()}, steering_task))
        for op in ("scale", "normalize", "copy"):
            command = [{"op": op, "tensor": WEIGHT, "block": [[0, 1], [0, 3]], "values": [[1, 2, 3]]}]
            rejected("symbolic_" + op + "_rejected", lambda command=command: decode_response(good_steering | {"commands_json": canonical_bytes(command).decode()}, steering_task))
        modified_bias = steering_parent.tensors[BIAS].copy()
        modified_bias[0] += 1.0
        rejected("target_bias_mutation_rejected", lambda: validate_candidate(steering_parent.replace(**{BIAS: modified_bias}), steering_task, steering_parent))
        broker, context = make_broker(creation_task, None, "none")
        check(context["metadata"]["task"]["mode"] == "create" and context["weights"] is None,
              "creation_has_no_parent_or_weight_view_comparison")
        rejected("creation_cannot_receive_exact_parent_view", lambda: make_broker(creation_task, None, "exact"))

        fixture, prepared = fixture_repo("mixed-status")
        lock_id = prepared["lock_sha256"]
        rejected("prepare_is_exclusive", lambda: prepare(fixture, development=True))
        with replaced(globals(), "sequential_forward", forbid), replaced(np.linalg, "lstsq", forbid):
            rejected("evaluation_before_gate_rejected", lambda: evaluate(fixture, lock_id, "0" * 64, fixture=True))
        check(not (fixture / ARTIFACTS).exists(), "missing_gate_rejected_before_artifact_creation")
        rejected("wrong_lock_anchor_rejected", lambda: select(fixture, "0" * 64, root / "wrong-lock-private", fixture=True))
        check(not (root / "wrong-lock-private").exists() and not (fixture / ARTIFACTS).exists(),
              "wrong_lock_rejected_before_outputs_or_provider")
        rejected("public_live_rejects_nonfrozen_runner_and_development", lambda: select(fixture, lock_id, root / "wrong-runner-private"))
        check(not (root / "wrong-runner-private").exists() and not (fixture / ARTIFACTS).exists(),
              "public_live_guard_rejected_before_outputs_or_provider")
        tamper_targets = {
            "source_mirror": fixture / SCRIPT,
            "frozen_source": fixture / EXPERIMENT / "source-freeze" / SCRIPT,
            "prompt": fixture / EXPERIMENT / "prompts" / "create-scaled_absolute-1.txt",
            "task": fixture / EXPERIMENT / "tasks" / "create-scaled_absolute-1" / "task.json",
            "training": fixture / EXPERIMENT / "tasks" / "create-scaled_absolute-1" / "train.json",
            "evaluator": fixture / EXPERIMENT / "tasks" / "create-scaled_absolute-1" / "eval.json",
            "runtime": fixture / EXPERIMENT / "runtime.json"}
        for name, path in tamper_targets.items():
            with changed_bytes(path, path.read_bytes() + b"\n"):
                rejected(name + "_tamper_rejected", lambda: validate_lock(fixture, lock_id))
        validate_lock(fixture, lock_id)
        behaviors = ["valid", "invalid", "failed", "timeout", "abstain", "valid", "valid", "valid", "valid", "interrupt", "valid", "valid"]
        provider = OfflineProvider(fixture, behaviors)
        with replaced(globals(), "sequential_forward", forbid), replaced(np.linalg, "lstsq", forbid), replaced(globals(), "fitted_control", forbid):
            selection_result = select(fixture, lock_id, root / "mixed-status-private", provider=provider, fixture=True)
        gate_id = selection_result["gate_sha256"]
        gate, rows, receipts = validate_gate(fixture, lock_id, gate_id)
        expected_counts = {key: 0 for key in TERMINALS}
        expected_counts.update(sealed=5, invalid=1, failed=1, timeout=1, abstained=1, interrupted=1, unattempted=2)
        check(provider.calls == 10 and gate["attempted_invocations"] == 10 and gate["terminal_counts"] == expected_counts,
              "all_twelve_accounted_with_ten_attempts_no_retry_and_interrupted_suffix")
        check(all(row["state"]["candidate"] is None for row in rows if row["state"]["terminal_status"] != "sealed"),
              "failed_invalid_abstained_interrupted_unattempted_have_no_fallback_candidate")
        accounting = usage_accounting(rows, receipts)
        check(accounting["input_tokens"]["known_attempts"] == 7 and accounting["input_tokens"]["unknown_attempts"] == 3
              and accounting["cached_input_added_to_input"] is False, "unknown_usage_retained_and_cached_input_not_double_counted")
        rejected("selection_cannot_resume_or_retry", lambda: select(fixture, lock_id, root / "retry-private", provider=provider, fixture=True))
        check(provider.calls == 10 and not (root / "retry-private").exists(), "rerun_rejected_before_provider_or_private_output")
        gate_path = fixture / ARTIFACTS / "selection-gate.json"
        bad_gate = {k: v for k, v in gate.items() if k != "sha256"}
        bad_gate["assignment_ids"] = bad_gate["assignment_ids"][:-1]
        bad_gate = signed(bad_gate)
        with changed_bytes(gate_path, canonical_bytes(bad_gate) + b"\n"):
            rejected("self_signed_gate_missing_assignment_rejected", lambda: evaluate(fixture, lock_id, bad_gate["sha256"], fixture=True))
        check(not (fixture / ARTIFACTS / "evaluation").exists(), "incomplete_gate_rejected_before_evaluation_output")
        rejected("wrong_gate_anchor_rejected", lambda: evaluate(fixture, lock_id, "0" * 64, fixture=True))
        evaluation_result = evaluate(fixture, lock_id, gate_id, fixture=True)
        evaluation_id = evaluation_result["evaluation_sha256"]
        manifest, outcomes = validate_evaluation(fixture, lock_id, gate_id, evaluation_id)
        check(outcomes["live_summary"]["all_assigned"]["denominator"] == ASSIGNED
              and outcomes["live_summary"]["all_assigned"]["successes"] == 5
              and outcomes["accounting"]["control_solve_calls"] == 9, "evaluation_denominator_and_control_budget_preserved")
        check(all(not row["metrics"]["success"] and row["metrics"]["target_max_abs_error"] is None
                  for row in outcomes["assignments"] if row["terminal_status"] != "sealed"),
              "missing_live_proposals_score_false_with_null_error")
        rejected("evaluation_is_exclusive", lambda: evaluate(fixture, lock_id, gate_id, fixture=True))
        outcome_path = fixture / ARTIFACTS / "evaluation" / "outcomes.json"
        with changed_bytes(outcome_path, outcome_path.read_bytes() + b"\n"):
            rejected("changed_outcome_rejected_before_replay", lambda: replay(fixture, lock_id, gate_id, evaluation_id, fixture=True))
        check(not (fixture / ARTIFACTS / "replay").exists(), "changed_outcome_rejected_before_replay_output")
        rejected("wrong_evaluation_anchor_rejected", lambda: replay(fixture, lock_id, gate_id, "0" * 64, fixture=True))
        replay_result = replay(fixture, lock_id, gate_id, evaluation_id, fixture=True)
        check(replay_result["status"] == "matched" and replay_result["complete_canonical_payload_match"]
              and replay_result["result_files_checked"] == 22, "provider_free_replay_matches_complete_payload_and_all_twenty_one_records")
        rejected("replay_is_exclusive", lambda: replay(fixture, lock_id, gate_id, evaluation_id, fixture=True))

        for name, behaviors, expected_calls, expected_terminal in (
                ("unclean-timeout", ["unclean_timeout"] + ["valid"] * 11, 1, "timeout"),
                ("harness-failure", ["harness_error"] + ["valid"] * 11, 1, "harness_failed"),
                ("deferred-sigint", ["signal"] + ["valid"] * 11, 1, "sealed"),
                ("all-completed", ["partial_usage"] + ["valid"] * 11, 12, "sealed")):
            test_repo, test_prepared = fixture_repo(name)
            test_provider = OfflineProvider(test_repo, behaviors)
            with replaced(globals(), "sequential_forward", forbid), replaced(np.linalg, "lstsq", forbid), replaced(globals(), "fitted_control", forbid):
                result = select(test_repo, test_prepared["lock_sha256"], root / (name + "-private"), provider=test_provider, fixture=True)
            test_gate, test_rows, test_receipts = validate_gate(test_repo, test_prepared["lock_sha256"], result["gate_sha256"])
            check(test_provider.calls == expected_calls and test_gate["attempted_invocations"] == expected_calls
                  and test_rows[0]["state"]["terminal_status"] == expected_terminal,
                  name + ":fixed_attempt_count_and_stop_behavior")
            if name == "all-completed":
                partial = usage_accounting(test_rows, test_receipts)
                check(partial["input_tokens"]["known_attempts"] == 12 and partial["output_tokens"]["unknown_attempts"] == 1,
                      "partial_usage_remains_unknown_per_field")
                check(test_gate["stop"] is None and test_gate["terminal_counts"]["sealed"] == 12,
                      "complete_twelve_call_fixture_has_no_stop_or_missing_selections")
        check(real_calls["blocked"] == 0, "zero_real_provider_method_entries")
    report = signed({"schema": PREFIX + "self-check/1", "started_utc": started, "finished_utc": now(),
        "status": "passed", "checks": checks, "check_count": len(checks), "fresh_provider_calls": 0,
        "assigned_task_endpoint_scores": 0, "assigned_task_control_solves": 0,
        "fixture_task_seed_range": [47001, 47009], "fixture_parent_seed_range": [47101, 47103],
        "fixture_train_seed_range": [47201, 47209], "fixture_evaluator_seed_range": [47301, 47309],
        "real_provider_method_blocked_entries": real_calls["blocked"],
        "source_files_sha256": project_closure(), "runtime_sha256": digest(runtime()),
        "primary_fixture": {"lock_sha256": lock_id, "gate_sha256": gate_id,
                            "evaluation_sha256": evaluation_id, "replay_sha256": replay_result["replay_sha256"]},
        "scope": "offline_development_fixtures_only;_no_assigned_task_outcomes_or_live_provider_evidence"})
    write_new(root / "self-check-report.json", report)
    return {"status": "passed", "check_count": len(checks), "fresh_provider_calls": 0,
            "assigned_task_endpoint_scores": 0, "self_check_sha256": report["sha256"],
            "report_path": str(root / "self-check-report.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("self-check", "prepare", "validate", "live", "evaluate", "replay"):
        child = subparsers.add_parser(command)
        child.add_argument("--repo", required=True, type=Path)
        if command in {"self-check", "live"}:
            child.add_argument("--private-root", required=True, type=Path)
        if command in {"validate", "live", "evaluate", "replay"}:
            child.add_argument("--expected-lock-sha256", required=True)
        if command in {"evaluate", "replay"}:
            child.add_argument("--expected-gate-sha256", required=True)
        if command == "replay":
            child.add_argument("--expected-evaluation-sha256", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        if args.command == "self-check":
            result = self_check(repo, args.private_root)
        elif args.command == "prepare":
            result = prepare(repo)
        elif args.command == "validate":
            lock = validate_lock(repo, args.expected_lock_sha256)
            result = {"status": "valid", "lock_sha256": lock["sha256"],
                      "protocol_sha256": lock["protocol_sha256"], "source_files": len(lock["source_files_sha256"]),
                      "assigned": ASSIGNED, "fresh_provider_calls": 0, "endpoint_scores": 0, "control_solves": 0}
        elif args.command == "live":
            result = select(repo, args.expected_lock_sha256, args.private_root)
        elif args.command == "evaluate":
            result = evaluate(repo, args.expected_lock_sha256, args.expected_gate_sha256)
        else:
            result = replay(repo, args.expected_lock_sha256, args.expected_gate_sha256, args.expected_evaluation_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 1 if result.get("status") == "mismatch" else 0
    except Exception as error:
        # Provider response, transport diagnostics, and private path text are never echoed.
        print(json.dumps({"status": "rejected_or_incomplete", "exception_type": type(error).__name__}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
