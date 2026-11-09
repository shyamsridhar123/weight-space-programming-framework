"""Prepare, evaluate, and verify a separately supplied prospective support cohort.

Run with ``python scripts/run_prospective_support_routing.py --help``. Preparation
reads only development rows and four explicitly named parent files. Evaluation
uses the externally supplied cohort committed by the configuration; it never
loads historical feature bundles. One durable attempt covers the parent and
every admitted proposal on that cohort. There is no efficacy or freshness claim.

Configuration fields: experiment_id, label_names, queue_names, queue_map,
assignments, cohort, and optional admission. Each assignment has assignment_id,
target_label, desired_code, mode (none/soft/nullspace/abstain), and optional
configuration (ridge, retention_weight, nullspace_rtol). Admission defaults are
minimum_target_gain=0, maximum_protected_harm=0, max_parameter_abs=256, and
max_change_abs=256. All effective defaults are copied into the frozen config.

Development JSON maps target/protected/calibration to envelopes with source_id,
family_id, parents (registered source IDs), and rows. Rows have request_id, text,
and integer label. Holdout JSON has the same envelope shape. The config's cohort
is that envelope without rows, plus row_count, rows_sha256,
normalized_inputs_sha256, and declaration. The declaration has collection_id,
labeling_method, reserved_for (experiment_id), no_development_use=true, and a
synthetic boolean. Hashes use canonical finite JSON; normalized input rows are
{"normalized_request": normalized_text(text)} in the original row order. These
are caller declarations, not authenticated provenance or semantic independence.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlcd.weight_space.artifact import StructuralPolicy, TensorArtifact
from rlcd.weight_space.data_roles import DataRoleError, DataRoleRegistry
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.program import Program, apply_program
from rlcd.weight_space.research_tasks import (
    ARCHITECTURE, HEAD_BIAS, HEAD_WEIGHT, TextPreprocessing, behavior_metrics,
    compile_arm, encode_inputs, evaluate_head, normalized_text,
)

EVALUATOR_ID = "prospective-support-native-readout/1"
PARENT_FILES = ("head.json", "encoder.npz", "preprocessing.json", "lineage.json")
ROLES = ("target", "protected", "calibration")
FLAGS = {"provenance_authenticated": False, "semantic_independence_verified": False,
         "scientific_freshness_verified": False, "evaluator_execution_attested": False}
ADMISSION = {"minimum_target_gain": 0., "maximum_protected_harm": 0.,
             "max_parameter_abs": 256., "max_change_abs": 256.}
FIT = {"ridge": .01, "retention_weight": 8., "nullspace_rtol": 1e-12}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_field")
        result[key] = value
    return result


def _json(data: bytes) -> Any:
    value = json.loads(data, object_pairs_hook=_object)
    canonical_bytes(value)  # Reject NaN and infinity, including overflowed numbers.
    return value


def read_json(path: Path) -> Any:
    return _json(Path(path).read_bytes())


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write(path, canonical_bytes(value) + b"\n")


def _manifest(path: Path, value: dict) -> None:
    # Keep every assignment visible even if a process dies between phases.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _name(value: Any) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("nonempty_identity_required")
    return value


def _hash(value: Any) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("sha256_required")
    return value


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite_number_required")
    return float(value)


def _native_environment() -> dict:
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {"python": platform.python_version(), "platform": sys.platform,
            "machine": platform.machine(), "numpy": np.__version__, "torch": version("torch"),
            "numeric_lane": "Torch CPU float32", "threads": 1,
            "deterministic_algorithms": True, "argmax": "lowest output index"}


def _evaluator_identity() -> dict:
    paths = [Path(__file__).resolve(), *sorted((ROOT / "rlcd/weight_space").glob("*.py")),
             ROOT / "pyproject.toml", ROOT / "requirements.lock.txt"]
    return {"id": EVALUATOR_ID, "source_files": {
        path.relative_to(ROOT).as_posix(): _sha(path.read_bytes()) for path in paths},
        "environment": _native_environment(), **FLAGS}


def load_parent(directory: Path) -> dict:
    """Validate captured parent bytes without opening any features file."""
    captured = {name: (Path(directory) / name).read_bytes() for name in PARENT_FILES}
    return _parent_from_bytes(captured)


def _parent_from_bytes(captured: dict[str, bytes]) -> dict:
    head = TensorArtifact.from_dict(_json(captured["head.json"]))
    preprocessing = _json(captured["preprocessing.json"])
    lineage = _json(captured["lineage.json"])
    with np.load(io.BytesIO(captured["encoder.npz"]), allow_pickle=False) as archive:
        if set(archive.files) != {"weight", "bias"}:
            raise ValueError("unsupported_encoder")
        encoder = {name: archive[name].copy() for name in archive.files}
    text = TextPreprocessing(tuple(preprocessing["vocabulary"]), tuple(preprocessing["idf"]))
    weight, bias = encoder["weight"], encoder["bias"]
    if (text.to_dict() != preprocessing or not text.vocabulary
            or len(set(text.vocabulary)) != len(text.vocabulary)
            or any(type(term) is not str or not term for term in text.vocabulary)
            or len(text.idf) != len(text.vocabulary)
            or any(_number(value) <= 0 for value in text.idf)
            or weight.ndim != 2 or weight.shape[1] != len(text.vocabulary)
            or weight.shape[0] < 1 or bias.shape != (weight.shape[0],)
            or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in encoder.values())
            or head.architecture != ARCHITECTURE or set(head.tensors) != {HEAD_WEIGHT, HEAD_BIAS}
            or head.tensors[HEAD_WEIGHT].ndim != 2
            or head.tensors[HEAD_WEIGHT].shape[1] != weight.shape[0]
            or head.tensors[HEAD_BIAS].shape != (head.tensors[HEAD_WEIGHT].shape[0],)
            or head.metadata.get("native_dtype") != "float32"):
        raise ValueError("parent_semantics_mismatch")
    with np.errstate(over="raise", invalid="raise"):
        if any(not np.array_equal(value, value.astype(np.float32).astype(np.float64))
               for value in head.tensors.values()):
            raise ValueError("parent_native_dtype_mismatch")
    encoder_sha = digest({name: value.tolist() for name, value in encoder.items()})
    preprocessing_sha = digest(preprocessing)
    feature_id = digest({"encoder_sha256": encoder_sha, "preprocessing_sha256": preprocessing_sha,
                         "native": "Torch CPU float32 affine then ReLU",
                         "feature_order": list(range(weight.shape[0]))})
    identities = {"head_sha256": head.sha256, "encoder_sha256": encoder_sha,
                  "preprocessing_sha256": preprocessing_sha, "feature_id": feature_id}
    if (lineage.get("schema") != "wsp-trained-lineage/1"
            or any(lineage.get(key) != value for key, value in identities.items())
            or any(head.metadata.get(key) != value for key, value in identities.items() if key != "head_sha256")
            or len(head.metadata.get("output_names", [])) != head.tensors[HEAD_BIAS].shape[0]
            or head.metadata.get("feature_names") != [f"h{i:03d}" for i in range(weight.shape[0])]):
        raise ValueError("parent_identity_mismatch")
    for name in PARENT_FILES[:-1]:
        record = lineage["files"][name]
        if record["sha256"] != _sha(captured[name]) or record["bytes"] != len(captured[name]):
            raise ValueError("parent_file_identity_mismatch")
    return {"head": head, "encoder": encoder, "preprocessing": text,
            "identities": identities, "captured": captured}


def _config(value: Any) -> dict:
    config = _json(canonical_bytes(value))
    required = {"experiment_id", "label_names", "queue_names", "queue_map", "assignments", "cohort"}
    if type(config) is not dict or not required <= set(config) or set(config) - required - {"admission"}:
        raise ValueError("invalid_configuration_fields")
    _name(config["experiment_id"])
    for field in ("label_names", "queue_names"):
        names = config[field]
        if type(names) is not list or not names or len(set(names)) != len(names):
            raise ValueError("distinct_names_required")
        for name in names:
            _name(name)
    if (type(config["queue_map"]) is not list
            or len(config["queue_map"]) != len(config["label_names"]) + 1
            or any(type(code) is not int or not 0 <= code < len(config["queue_names"]) for code in config["queue_map"])):
        raise ValueError("invalid_queue_map")
    assignments = config["assignments"]
    if type(assignments) is not list or not assignments or any(type(row) is not dict for row in assignments):
        raise ValueError("assigned_proposals_required")
    ids = [_name(row.get("assignment_id")) for row in assignments]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_assignment")
    admission = config.get("admission", {})
    if type(admission) is not dict or set(admission) - set(ADMISSION):
        raise ValueError("invalid_admission")
    config["admission"] = {key: _number(value) for key, value in (ADMISSION | admission).items()}
    if (any(not -1 <= config["admission"][key] <= 1 for key in ("minimum_target_gain", "maximum_protected_harm"))
            or any(config["admission"][key] < 0 for key in ("max_parameter_abs", "max_change_abs"))):
        raise ValueError("invalid_admission_bounds")
    cohort = config["cohort"]
    if type(cohort) is not dict or set(cohort) != {
            "source_id", "family_id", "parents", "row_count", "rows_sha256", "normalized_inputs_sha256", "declaration"}:
        raise ValueError("invalid_cohort_manifest")
    _envelope_identity(cohort)
    _hash(cohort["rows_sha256"])
    _hash(cohort["normalized_inputs_sha256"])
    if type(cohort["row_count"]) is not int or cohort["row_count"] < 1:
        raise ValueError("nonempty_cohort_required")
    declaration = cohort["declaration"]
    if (type(declaration) is not dict or set(declaration) != {
            "collection_id", "labeling_method", "reserved_for", "no_development_use", "synthetic"}
            or declaration["reserved_for"] != config["experiment_id"]
            or declaration["no_development_use"] is not True or type(declaration["synthetic"]) is not bool):
        raise ValueError("cohort_declaration_required")
    _name(declaration["collection_id"])
    _name(declaration["labeling_method"])
    # Keep malformed proposals in their assigned position; valid proposals carry
    # their effective fitting defaults in the copied configuration as well.
    for index, assigned in enumerate(assignments):
        try:
            assignments[index] = _assignment(assigned, config)
        except (ValueError, TypeError):
            pass
    return config


def _envelope_identity(envelope: dict) -> None:
    _name(envelope["source_id"])
    _name(envelope["family_id"])
    if type(envelope["parents"]) is not list or len(set(envelope["parents"])) != len(envelope["parents"]):
        raise ValueError("invalid_parents")
    for parent in envelope["parents"]:
        _name(parent)


def _rows(envelope: Any, label_count: int) -> list[dict]:
    if type(envelope) is not dict or set(envelope) != {"source_id", "family_id", "parents", "rows"}:
        raise ValueError("invalid_input_envelope")
    _envelope_identity(envelope)
    rows = envelope["rows"]
    if type(rows) is not list or not rows:
        raise ValueError("nonempty_rows_required")
    ids = []
    for row in rows:
        if type(row) is not dict or set(row) != {"request_id", "text", "label"}:
            raise ValueError("invalid_support_row")
        ids.append(_name(row["request_id"]))
        if (type(row["text"]) is not str or not normalized_text(row["text"])
                or type(row["label"]) is not int or not 0 <= row["label"] < label_count):
            raise ValueError("invalid_support_row")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_request_id")
    return rows


def _normalized(rows: list[dict]) -> list[dict]:
    return [{"normalized_request": normalized_text(row["text"])} for row in rows]


def _register(registry: DataRoleRegistry, envelope: dict, label_count: int) -> tuple[Any, dict]:
    rows = _rows(envelope, label_count)
    source, family = envelope["source_id"], envelope["family_id"]
    # Input-only aliases are registered before rows reach fitting or scoring.
    normalized = registry.register_batch(_normalized(rows), source_id=source + "/normalized-inputs",
        family_id=family, parents=tuple(envelope["parents"]))
    identities = registry.register_batch([{"request_id": row["request_id"]} for row in rows],
        source_id=source + "/request-identities", family_id=family, parents=tuple(envelope["parents"]))
    raw = registry.register_batch(rows, source_id=source, family_id=family, parents=(normalized, identities))
    return raw, {"raw": raw.to_dict(), "normalized": normalized.to_dict(), "identities": identities.to_dict()}


def _assignment(value: dict, config: dict) -> dict:
    if (set(value) - {"assignment_id", "target_label", "desired_code", "mode", "configuration"}
            or not {"assignment_id", "target_label", "desired_code", "mode"} <= set(value)
            or value["mode"] not in ("none", "soft", "nullspace", "abstain")
            or type(value["target_label"]) is not int or not 0 <= value["target_label"] < len(config["label_names"])
            or type(value["desired_code"]) is not int or not 0 <= value["desired_code"] < len(config["queue_map"])):
        raise ValueError("invalid_assignment")
    options = value.get("configuration", {})
    if type(options) is not dict or set(options) - set(FIT):
        raise ValueError("invalid_fit_configuration")
    options = {key: _number(item) for key, item in (FIT | options).items()}
    if options["ridge"] <= 0 or options["retention_weight"] < 0 or options["nullspace_rtol"] < 0:
        raise ValueError("invalid_fit_configuration")
    return value | {"configuration": options}


def _passes(metrics: dict, admission: dict) -> bool:
    return (metrics["target_gain"] is not None and metrics["protected_harm"] is not None
            and metrics["target_gain"] >= admission["minimum_target_gain"]
            and metrics["protected_harm"] <= admission["maximum_protected_harm"])


def prepare(parent_directory: Path, development_path: Path, configuration_path: Path,
            run_directory: Path, registry_path: Path) -> dict:
    config = _config(read_json(configuration_path))
    run = Path(run_directory)
    run.mkdir(parents=True, exist_ok=False)
    registry = DataRoleRegistry(registry_path)
    records = [{"assignment_id": item["assignment_id"], "assignment_sha256": digest(item),
                "proposal_status": "pending", "reason": None} for item in config["assignments"]]
    manifest = {"schema": "wsp-prospective-support-manifest/1", "experiment_id": config["experiment_id"],
                "phase": "preparing", "assignments": records, **FLAGS}
    _manifest(run / "manifest.json", manifest)
    try:
        parent = load_parent(parent_directory)
        base = parent["head"]
        if base.metadata["output_names"] != config["label_names"] + ["specialist_routing_code"]:
            raise ValueError("label_ontology_mismatch")
        for name, data in parent["captured"].items():
            _write(run / "parent" / name, data)
        evaluator = _evaluator_identity()
        development = read_json(development_path)
        if type(development) is not dict or set(development) != set(ROLES):
            raise ValueError("development_roles_required")
        bundles, development_records = {}, {}
        # Consume all raw development roles before any preprocessing or fitting.
        for role in ROLES:
            batch, inputs = _register(registry, development[role], len(config["label_names"]))
            consumption = registry.consume(batch, reason=f"support-development/{config['experiment_id']}/{role}")
            bundles[role] = batch
            development_records[role] = {"inputs": inputs, "consumption": consumption}
        features, labels = {}, {}
        for role, batch in bundles.items():
            rows = batch.materialize()
            features[role] = encode_inputs(parent["encoder"], parent["preprocessing"].transform([row["text"] for row in rows]))
            labels[role] = np.asarray([row["label"] for row in rows], dtype=np.int64)
            feature_rows = [{"request_id": row["request_id"], "features": values.tolist(),
                             "feature_id": base.metadata["feature_id"], "dtype": "float32"}
                            for row, values in zip(rows, features[role])]
            derived = registry.register_batch(feature_rows,
                source_id=development[role]["source_id"] + "/features/" + base.metadata["feature_id"],
                family_id=development[role]["family_id"], parents=(batch,))
            development_records[role]["features"] = derived.to_dict()
            development_records[role]["feature_consumption"] = registry.consume(derived, reason="support-derived-development")
        policy = StructuralPolicy((HEAD_WEIGHT, HEAD_BIAS), base.parameter_count,
            config["admission"]["max_parameter_abs"], config["admission"]["max_change_abs"])
        parent_codes = evaluate_head(base, features["calibration"], native=True).argmax(1)
        for index, (assigned, record) in enumerate(zip(config["assignments"], records)):
            try:
                spec = _assignment(assigned, config)
                record["configuration"] = spec
                if spec["mode"] == "abstain":
                    record.update(proposal_status="abstained", reason="declared_abstention")
                    continue
                target = features["target"][labels["target"] == spec["target_label"]]
                protected = features["protected"][labels["protected"] != spec["target_label"]]
                if not len(target) or not len(protected):
                    raise ValueError("missing_development_stratum")
                fitted = compile_arm(base, target, protected, spec["desired_code"], spec["mode"], **spec["configuration"])
                candidate, program = fitted["artifact"], fitted["program"]
                structural = policy.check(base, program)
                codes = evaluate_head(candidate, features["calibration"], native=True).argmax(1)
                metrics = behavior_metrics(parent_codes, codes, labels["calibration"], spec["target_label"],
                    np.asarray(config["queue_map"]), config["queue_map"][spec["desired_code"]])
                folder = f"assignments/{index:04d}"
                _write_json(run / folder / "candidate.json", candidate.to_dict())
                _write_json(run / folder / "program.json", program.to_dict())
                accepted = structural["structural_accepted"] and _passes(metrics, config["admission"])
                record.update(proposal_status="admitted" if accepted else "abstained",
                    reason="development_admission_passed" if accepted else "development_admission_failed",
                    candidate_sha256=candidate.sha256, program_sha256=program.sha256, artifact_directory=folder,
                    calibration=metrics, structural=structural)
            except Exception as failure:
                record.update(proposal_status="rejected" if isinstance(failure, ValueError) else "failed",
                              reason=type(failure).__name__)
            finally:
                _manifest(run / "manifest.json", manifest)
        _write_json(run / "configuration.json", config)
        frozen_names = ["configuration.json", *("parent/" + name for name in PARENT_FILES)]
        for record in records:
            if "artifact_directory" in record:
                frozen_names.extend(record["artifact_directory"] + "/" + name for name in ("candidate.json", "program.json"))
        body = {"schema": "wsp-prospective-support-protocol/1", "experiment_id": config["experiment_id"],
                "attempt_id": config["experiment_id"] + "/support-routing-evaluation", "registry_id": registry.registry_id,
                "configuration_sha256": digest(config), "configuration": config, "parent": parent["identities"],
                "assignments": records, "development": development_records, "evaluator": evaluator,
                "files": {name: _sha((run / name).read_bytes()) for name in frozen_names}, **FLAGS}
        protocol = body | {"protocol_sha256": digest(body)}
        _write_json(run / "protocol.json", protocol)
        manifest.update(phase="sealed", protocol_sha256=protocol["protocol_sha256"], attempt_id=protocol["attempt_id"])
        _manifest(run / "manifest.json", manifest)
        return protocol
    except BaseException as failure:
        for record in records:
            if record["proposal_status"] == "pending":
                record.update(proposal_status="failed", reason=type(failure).__name__)
        manifest.update(phase="preparation_failed", reason=type(failure).__name__)
        _manifest(run / "manifest.json", manifest)
        raise


def _load_frozen(run: Path, registry: DataRoleRegistry) -> tuple[dict, dict, dict]:
    protocol = read_json(run / "protocol.json")
    if (type(protocol) is not dict or protocol.get("schema") != "wsp-prospective-support-protocol/1"
            or protocol.get("protocol_sha256") != digest({key: value for key, value in protocol.items() if key != "protocol_sha256"})
            or protocol["registry_id"] != registry.registry_id
            or protocol["evaluator"] != _evaluator_identity()
            or any(protocol.get(key) is not False for key in FLAGS)):
        raise ValueError("frozen_protocol_identity_mismatch")
    config = _config(protocol["configuration"])
    if config != protocol["configuration"] or digest(config) != protocol["configuration_sha256"]:
        raise ValueError("frozen_configuration_mismatch")
    if (protocol["experiment_id"] != config["experiment_id"]
            or protocol["attempt_id"] != config["experiment_id"] + "/support-routing-evaluation"
            or type(protocol["assignments"]) is not list
            or len(protocol["assignments"]) != len(config["assignments"])
            or [row["assignment_id"] for row in protocol["assignments"]] != [row["assignment_id"] for row in config["assignments"]]):
        raise ValueError("assigned_proposal_missing")
    required = {"configuration.json", *("parent/" + name for name in PARENT_FILES)}
    artifact_fields = {"candidate_sha256", "program_sha256", "artifact_directory", "calibration", "structural"}
    for index, (assigned, record) in enumerate(zip(config["assignments"], protocol["assignments"])):
        if (record["assignment_sha256"] != digest(assigned)
                or record["proposal_status"] not in ("admitted", "abstained", "rejected", "failed")):
            raise ValueError("frozen_assignment_mismatch")
        _name(record["reason"])
        if "configuration" in record and record["configuration"] != _assignment(assigned, config):
            raise ValueError("frozen_selection_mismatch")
        if artifact_fields & set(record):
            if (not artifact_fields <= set(record) or record["artifact_directory"] != f"assignments/{index:04d}"
                    or record["proposal_status"] not in ("admitted", "abstained")
                    or record["configuration"]["mode"] == "abstain"):
                raise ValueError("frozen_selection_mismatch")
            required.update(record["artifact_directory"] + "/" + name for name in ("candidate.json", "program.json"))
        elif record["proposal_status"] == "admitted" or (record["proposal_status"] == "abstained"
                and (record["configuration"]["mode"] != "abstain" or record["reason"] != "declared_abstention")):
            raise ValueError("frozen_selection_mismatch")
    if type(protocol["files"]) is not dict or set(protocol["files"]) != required:
        raise ValueError("frozen_file_set_mismatch")
    captured = {}
    for name in sorted(required):
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()):
            raise ValueError("frozen_file_identity_mismatch")
        captured[name] = path.read_bytes()
        if _sha(captured[name]) != protocol["files"][name]:
            raise ValueError("frozen_file_identity_mismatch")
    if _json(captured["configuration.json"]) != config:
        raise ValueError("frozen_configuration_mismatch")
    parent = _parent_from_bytes({name: captured["parent/" + name] for name in PARENT_FILES})
    if (parent["identities"] != protocol["parent"]
            or parent["head"].metadata["output_names"] != config["label_names"] + ["specialist_routing_code"]):
        raise ValueError("frozen_parent_mismatch")
    policy = StructuralPolicy((HEAD_WEIGHT, HEAD_BIAS), parent["head"].parameter_count,
        config["admission"]["max_parameter_abs"], config["admission"]["max_change_abs"])
    candidates = {}
    for record in protocol["assignments"]:
        if "artifact_directory" in record:
            folder = record["artifact_directory"]
            candidate = TensorArtifact.from_dict(_json(captured[folder + "/candidate.json"]))
            program = Program.from_dict(_json(captured[folder + "/program.json"]))
            if (candidate.sha256 != record["candidate_sha256"] or program.sha256 != record["program_sha256"]
                    or apply_program(parent["head"], program).sha256 != candidate.sha256):
                raise ValueError("frozen_candidate_mismatch")
            structural = policy.check(parent["head"], program)
            accepted = structural["structural_accepted"] and _passes(record["calibration"], config["admission"])
            if (structural != record["structural"] or accepted != (record["proposal_status"] == "admitted")
                    or record["reason"] != ("development_admission_passed" if accepted else "development_admission_failed")):
                raise ValueError("frozen_selection_mismatch")
            if accepted:
                candidates[record["assignment_id"]] = candidate
    return protocol, parent, candidates


def _binding(protocol: dict, inputs: dict) -> dict:
    cohort = protocol["configuration"]["cohort"]
    if type(inputs) is not dict or set(inputs) != {"raw", "normalized", "identities"}:
        raise ValueError("input_manifest_mismatch")
    for manifest in inputs.values():
        if (manifest["schema"] != "wsp-data-batch/1" or manifest["registry_id"] != protocol["registry_id"]
                or manifest["family_id"] != cohort["family_id"]
                or manifest["manifest_sha256"] != digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
                or any(manifest.get(key) is not False for key in FLAGS)):
            raise ValueError("input_manifest_mismatch")
    for key, suffix in (("normalized", "/normalized-inputs"), ("identities", "/request-identities")):
        if (inputs[key]["source_id"] != cohort["source_id"] + suffix
                or inputs[key]["parents"] != sorted(cohort["parents"])):
            raise ValueError("input_manifest_mismatch")
    raw = inputs["raw"]
    if (raw["source_id"] != cohort["source_id"] or raw["family_id"] != cohort["family_id"]
            or raw["content_sha256"] != cohort["rows_sha256"] or len(raw["row_sha256"]) != cohort["row_count"]
            or raw["parents"] != sorted(inputs[key]["source_id"] for key in ("normalized", "identities"))
            or raw["parent_manifest_sha256"] != {inputs[key]["source_id"]: inputs[key]["manifest_sha256"]
                                                 for key in ("normalized", "identities")}
            or inputs["normalized"]["content_sha256"] != cohort["normalized_inputs_sha256"]):
        raise ValueError("cohort_identity_mismatch")
    return {"schema": "wsp-prospective-support-binding/1", "experiment_id": protocol["experiment_id"],
            "protocol_sha256": protocol["protocol_sha256"], "configuration_sha256": protocol["configuration_sha256"],
            "parent": protocol["parent"], "selection_sha256": digest(protocol["assignments"]),
            "frozen_files_sha256": digest(protocol["files"]), "evaluator": protocol["evaluator"],
            "cohort_sha256": digest(cohort), "inputs": inputs}


def _outcomes(protocol: dict, status: str) -> list[dict]:
    return [{"assignment_id": record["assignment_id"], "proposal_status": record["proposal_status"],
             "evaluation_status": status if record["proposal_status"] == "admitted" else "not_scored",
             "reason": record["reason"]} for record in protocol["assignments"]]


def _score_rows(rows: list[dict], protocol: dict, parent: dict, candidates: dict) -> dict:
    """The only prospective computation path; called by registry.score."""
    config = protocol["configuration"]
    if digest(rows) != config["cohort"]["rows_sha256"]:
        raise ValueError("callback_rows_mismatch")
    inputs = parent["preprocessing"].transform([row["text"] for row in rows])
    features = encode_inputs(parent["encoder"], inputs)
    parent_codes = evaluate_head(parent["head"], features, native=True).argmax(1)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    outcomes = _outcomes(protocol, "succeeded")
    for record, outcome in zip(protocol["assignments"], outcomes):
        if record["proposal_status"] == "admitted":
            spec = record["configuration"]
            codes = evaluate_head(candidates[record["assignment_id"]], features, native=True).argmax(1)
            metrics = behavior_metrics(parent_codes, codes, labels, spec["target_label"],
                np.asarray(config["queue_map"]), config["queue_map"][spec["desired_code"]])
            outcome.update(candidate_sha256=record["candidate_sha256"], codes=codes.tolist(), metrics=metrics,
                           meets_frozen_sample_thresholds=_passes(metrics, config["admission"]))
    return {"schema": "wsp-prospective-support-result/1", "protocol_sha256": protocol["protocol_sha256"],
            "evaluator_id": EVALUATOR_ID, "input_content_sha256": digest(rows), "row_count": len(rows),
            "feature_sha256": digest(features.tolist()), "parent_codes": parent_codes.tolist(),
            "assignments": outcomes, **FLAGS}


def _verify_report(protocol: dict, registry: DataRoleRegistry, report: dict) -> None:
    """Require exact durable success; legacy hashes/receipts alone cannot pass."""
    try:
        evidence, result = report["evidence"], report["result"]
        binding = _binding(protocol, report["inputs"])
        current = registry.get_attempt(protocol["attempt_id"])
        raw = report["inputs"]["raw"]
        expected = {"attempt_id": protocol["attempt_id"], "registry_id": registry.registry_id,
                    "manifest_sha256": raw["manifest_sha256"], "content_sha256": raw["content_sha256"],
                    "row_sha256": raw["row_sha256"], "result_sha256": digest(result), "binding_sha256": digest(binding)}
        if (report["schema"] != "wsp-prospective-support-evaluation/1" or report["status"] != "succeeded"
                or report["protocol_sha256"] != protocol["protocol_sha256"]
                or evidence["schema"] != "wsp-data-role-decision/1" or evidence["operation"] != "score"
                or evidence["verdict"] != "registered_cooperative" or evidence["reason"] != "observed_boundary_passed"
                or evidence["status"] != "succeeded" or evidence["scoring_started"] is not True
                or not current or canonical_bytes(evidence["attempt"]) != canonical_bytes(current)
                or current["status"] != "succeeded" or current["callback_invocations"] != 1
                or canonical_bytes(current["binding"]) != canonical_bytes(binding)
                or canonical_bytes(current["result"]) != canonical_bytes(result)
                or any(current.get(key) != value or evidence.get(key) != value for key, value in expected.items())
                or evidence["consumed_matches"] != []
                or any(evidence.get(key) != current.get(key) for key in (
                    "registry_revision_before", "registry_revision_after",
                    "registry_state_sha256_before", "registry_state_sha256_after"))
                or any(evidence.get(key) != current.get(key) or not current.get(key)
                       for key in ("reservation_sha256", "lifecycle_sha256"))
                or evidence["decision_sha256"] != digest({key: value for key, value in evidence.items() if key != "decision_sha256"})
                or any(document.get(key) is not False for document in (report, result, evidence, current) for key in FLAGS)
                or result["schema"] != "wsp-prospective-support-result/1" or result["row_count"] != len(raw["row_sha256"])
                or result["protocol_sha256"] != protocol["protocol_sha256"]
                or result["evaluator_id"] != EVALUATOR_ID or result["input_content_sha256"] != raw["content_sha256"]
                or canonical_bytes(report["assignments"]) != canonical_bytes(result["assignments"])
                or [row["assignment_id"] for row in result["assignments"]] != [row["assignment_id"] for row in protocol["assignments"]]):
            raise ValueError("prospective_evidence_mismatch")
    except (KeyError, TypeError, AttributeError):
        raise ValueError("prospective_evidence_mismatch") from None


def evaluate(run_directory: Path, holdout_path: Path, registry_path: Path) -> dict:
    run, registry = Path(run_directory), DataRoleRegistry(registry_path)
    protocol, parent, candidates = _load_frozen(run, registry)
    # This exclusive local marker also prevents races overwriting a successful report.
    _write_json(run / "evaluation-request.json", {"protocol_sha256": protocol["protocol_sha256"],
                "attempt_id": protocol["attempt_id"]})
    manifest = read_json(run / "manifest.json")
    manifest.update(phase="evaluation_requested", assignments=_outcomes(protocol, "outcome_unknown"))
    _manifest(run / "manifest.json", manifest)
    report = {"schema": "wsp-prospective-support-evaluation/1", "protocol_sha256": protocol["protocol_sha256"],
              "status": "abstained", "result": None, "evidence": None, "inputs": None,
              "assignments": _outcomes(protocol, "not_scored"), **FLAGS}
    interrupted = None
    try:
        if candidates:
            envelope = read_json(holdout_path)
            rows = _rows(envelope, len(protocol["configuration"]["label_names"]))
            cohort = protocol["configuration"]["cohort"]
            if (any(envelope[key] != cohort[key] for key in ("source_id", "family_id", "parents"))
                    or len(rows) != cohort["row_count"] or digest(rows) != cohort["rows_sha256"]
                    or digest(_normalized(rows)) != cohort["normalized_inputs_sha256"]):
                raise ValueError("cohort_identity_mismatch")
            batch, inputs = _register(registry, envelope, len(protocol["configuration"]["label_names"]))
            binding = _binding(protocol, inputs)
            report["inputs"] = inputs
            result, evidence = registry.score(batch, lambda captured: _score_rows(captured, protocol, parent, candidates),
                                              attempt_id=protocol["attempt_id"], binding=binding)
            report.update(status="succeeded", result=result, evidence=evidence, assignments=result["assignments"])
            _verify_report(protocol, registry, report)
    except BaseException as failure:
        evidence = (failure.decision if isinstance(failure, DataRoleError)
                    else getattr(failure, "data_role_decision", None)) or report["evidence"]
        status = "failed" if evidence and evidence.get("scoring_started") else "rejected"
        report.update(status=status, result=None, evidence=evidence,
                      reason=type(failure).__name__, assignments=_outcomes(protocol, status))
        if not isinstance(failure, Exception):
            interrupted = failure
    _write_json(run / "evaluation.json", report)
    manifest.update(phase=report["status"], assignments=report["assignments"],
                    evaluation_sha256=digest(report))
    _manifest(run / "manifest.json", manifest)
    if interrupted is not None:
        raise interrupted
    return report


def verify_result(run_directory: Path, registry_path: Path, report: dict | None = None) -> bool:
    """Verification-only prospective report gate; never recomputes predictions."""
    run, registry = Path(run_directory), DataRoleRegistry(registry_path)
    protocol, _, _ = _load_frozen(run, registry)
    _verify_report(protocol, registry, read_json(run / "evaluation.json") if report is None else report)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--registry", type=Path, required=True, help="The same shared durable SQLite database for every phase")
        if name == "prepare":
            command.add_argument("--parent", type=Path, required=True)
            command.add_argument("--development", type=Path, required=True)
            command.add_argument("--config", type=Path, required=True)
        elif name == "evaluate":
            command.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        protocol = prepare(args.parent, args.development, args.config, args.run_dir, args.registry)
        output = {"status": "sealed", "protocol_sha256": protocol["protocol_sha256"]}
    elif args.command == "evaluate":
        report = evaluate(args.run_dir, args.holdout, args.registry)
        output = {"status": report["status"], "protocol_sha256": report["protocol_sha256"]}
    else:
        output = {"verified": verify_result(args.run_dir, args.registry)}
    print(json.dumps(output, sort_keys=True))
    return int(output.get("status") in ("failed", "rejected"))


if __name__ == "__main__":
    raise SystemExit(main())
