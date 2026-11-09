"""Bottleneck-first, secondary affine-head domain research.

The first profile reads only the already sealed cumulative calibration pool.
It neither opens heldout features nor changes a candidate or score budget.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
from statistics import median
import time
import tracemalloc
from typing import Any

import numpy as np

from . import cumulative_research as cumulative
from .artifact import TensorArtifact
from .evidence import ROOT, environment, file_hash, read_json, utc_now, verify_manifest, write_json
from .features import ReadoutHead, _endpoint_delta, certify_feature_retention
from .model import InputBox, canonical_bytes, digest, finite_binary64
from .research_tasks import ARCHITECTURE, HEAD_BIAS, HEAD_WEIGHT, evaluate_head, research_registry
from .semantics import SemanticsAdapter, SemanticsRegistry
from .sequential import certify_sequential_drift, sequential_artifact


SOURCE_FILES = cumulative.SOURCE_FILES + ("rlcd/weight_space/bound_research.py",)
CUMULATIVE_ROOT = ROOT/"artifacts/research/cumulative-admission/2026-11-08"
CUMULATIVE_MANIFEST = "208802510de6675f1ae09efcac54932c4dae9651df5d9d08f395e1b364b1309d"
PROFILE_ROOT = ROOT/"artifacts/research/affine-bounds/2026-11-08-profile"
PROFILE_MANIFEST = "45212f1ee628622530ed36245b246ced6fc7d7bd23436721dd9fd48c8e77548f"
ENCODER_SEMANTICS = "Supervised float32 affine/ReLU bottleneck; frozen after development-selected epoch"
HEAD_SEMANTICS = "float32 affine output codes; queue decision maps the deterministic lowest-index argmax code"
CERTIFICATE_SCOPE = "Exact real affine score differences on the declared learned-feature box; no raw-input, native-kernel, or task guarantee"
GLOBAL_GATE_PATH = "artifacts/research/live-studies/global-selection-seal.json"
GLOBAL_LOCK_PATH = "experiments/research/live-studies/preparation-lock.json"
GLOBAL_ASSIGNED_TOTAL = 224
TERMINAL_STATUSES = frozenset(("sealed", "abstained", "invalid", "failed", "timeout", "interrupted"))
REFERENCE_HASHES = {"torchlean": "dd8b7ce74b70d6a5bd928bf32f5caaeaf9354b56902d8f9a47f6f14f77c0c4d3",
    "crown": "7b57198bb3693c7a32c1000f2585b7772b5e76c6954614d8d3392c7a1a32ad32",
    "provable-repair": "948b259ed6e359ad7926182ebce4f62a5a3c0c3c431f660687460abbdf8d2302"}


def _trained_head(artifact: TensorArtifact) -> ReadoutHead:
    """An explicit head-only subset, never a fallback for a head-shaped graph."""
    fields = {"feature_names", "output_names", "feature_id", "encoder_sha256", "preprocessing_sha256",
              "encoder_semantics", "head_semantics", "native_dtype", "domain", "lineage_seed", "pretraining_ancestry"}
    if (not isinstance(artifact, TensorArtifact) or artifact.architecture != ARCHITECTURE
            or set(artifact.tensors) != {HEAD_WEIGHT, HEAD_BIAS} or set(artifact.metadata) != fields):
        raise ValueError("Only the explicitly declared trained affine readout layout is supported")
    metadata = artifact.metadata
    if any(not isinstance(metadata[key], str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[key])
           for key in ("feature_id", "encoder_sha256", "preprocessing_sha256")):
        raise ValueError("The trained subset requires explicit feature, encoder and preprocessing identities")
    head = ReadoutHead(HEAD_WEIGHT, HEAD_BIAS, metadata["feature_id"])
    width, outputs = head.dimensions(artifact)
    expected_feature_id = digest({"encoder_sha256": metadata["encoder_sha256"],
        "preprocessing_sha256": metadata["preprocessing_sha256"],
        "native": "Torch CPU float32 affine then ReLU", "feature_order": list(range(width))})
    names = metadata["output_names"]
    if (metadata["feature_names"] != [f"h{i:03d}" for i in range(width)]
            or not isinstance(names, list) or len(names) != outputs
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names) or metadata["feature_id"] != expected_feature_id
            or metadata["encoder_semantics"] != ENCODER_SEMANTICS
            or metadata["head_semantics"] != HEAD_SEMANTICS or metadata["native_dtype"] != "float32"
            or type(metadata["lineage_seed"]) is not int
            or any(not isinstance(metadata[key], str) or not metadata[key].strip()
                   for key in ("domain", "pretraining_ancestry"))):
        raise ValueError("The affine feature order, identities, activation/state declarations or output layout are unsupported")
    with np.errstate(over="ignore", under="ignore"):
        if any(not np.array_equal(value, value.astype(np.float32).astype(np.float64))
               for value in artifact.tensors.values()):
            raise ValueError("Stored trained endpoints must be exactly representable in their declared float32 lane")
    return head


def _context(base: TensorArtifact, candidate: TensorArtifact, box: InputBox) -> dict[str, Any]:
    head = _trained_head(base)
    _trained_head(candidate)
    if not base.compatible(candidate) or len(box.lower) != head.dimensions(base)[0]:
        raise ValueError("Affine endpoints, feature identity and box dimensions must correspond")
    return {"schema": "wsp-trained-affine-box-drift/1", "base_artifact_sha256": base.sha256,
        "candidate_artifact_sha256": candidate.sha256, "architecture": ARCHITECTURE,
        "head": head.to_dict(), "domain": box.to_dict(), "domain_sha256": digest(box.to_dict()),
        "arithmetic": "Exact rational differences of stored binary endpoints; real affine equation",
        "scope": CERTIFICATE_SCOPE, "implementation_machine_checked": False,
        "feature_extractor_verified": False, "floating_execution_verified": False,
        "core_native_admission_integrated": False}


def certify_affine_box(base: TensorArtifact, candidate: TensorArtifact, box: InputBox) -> dict[str, Any]:
    """Exact signed extrema; each output has its own attaining box corners."""
    payload = _context(base, candidate, box)
    head = ReadoutHead(**payload["head"])
    delta = _endpoint_delta(base, candidate, head)
    lower = [Fraction.from_float(float(value)) for value in box.lower]
    upper = [Fraction.from_float(float(value)) for value in box.upper]
    intervals = []
    for row in delta:
        terms = [(weight*lo, weight*hi) for weight, lo, hi in zip(row[:-1], lower, upper)]
        lo = row[-1] + sum((min(a, b) for a, b in terms), Fraction(0))
        hi = row[-1] + sum((max(a, b) for a, b in terms), Fraction(0))
        intervals.append({"lower_exact": str(lo), "upper_exact": str(hi),
            "absolute_bound_exact": str(max(abs(lo), abs(hi))),
            "lower_corner_bits": "".join("0" if weight >= 0 else "1" for weight in row[:-1]),
            "upper_corner_bits": "".join("1" if weight >= 0 else "0" for weight in row[:-1])})
    payload |= {"intervals": intervals,
                "maximum_exact": str(max(Fraction(row["absolute_bound_exact"]) for row in intervals))}
    return payload | {"sha256": digest(payload)}


def verify_affine_box(base: TensorArtifact, candidate: TensorArtifact, box: InputBox,
                      receipt: dict[str, Any]) -> bool:
    """Check independent corner evaluations/sign conditions, not the generator."""
    try:
        context = _context(base, candidate, box)
        if (not isinstance(receipt, dict) or set(receipt) != set(context) | {"intervals", "maximum_exact", "sha256"}
                or canonical_bytes({key: receipt[key] for key in context}) != canonical_bytes(context)
                or digest({key: value for key, value in receipt.items() if key != "sha256"}) != receipt["sha256"]):
            return False
        before, after = base.tensors, candidate.tensors
        lower, upper = box.lower, box.upper
        width, outputs = before[HEAD_WEIGHT].shape[1], before[HEAD_WEIGHT].shape[0]
        if not isinstance(receipt["intervals"], list) or len(receipt["intervals"]) != outputs:
            return False
        bounds = []
        for output, interval in enumerate(receipt["intervals"]):
            if (not isinstance(interval, dict) or set(interval) != {"lower_exact", "upper_exact", "absolute_bound_exact",
                                                                  "lower_corner_bits", "upper_corner_bits"}):
                return False
            exact = []
            for key in ("lower_exact", "upper_exact", "absolute_bound_exact"):
                if type(interval[key]) is not str:
                    return False
                value = Fraction(interval[key])
                if str(value) != interval[key]:
                    return False
                exact.append(value)
            bias = Fraction.from_float(float(after[HEAD_BIAS][output]))-Fraction.from_float(float(before[HEAD_BIAS][output]))
            for corner_key, expected in (("lower_corner_bits", exact[0]), ("upper_corner_bits", exact[1])):
                bits = interval[corner_key]
                if not isinstance(bits, str) or len(bits) != width or any(bit not in "01" for bit in bits):
                    return False
                evaluated = bias
                for column, bit in enumerate(bits):
                    change = (Fraction.from_float(float(after[HEAD_WEIGHT][output, column]))
                              -Fraction.from_float(float(before[HEAD_WEIGHT][output, column])))
                    # Sign checks establish extremality; corner evaluation establishes attainment.
                    minimizing = corner_key == "lower_corner_bits"
                    required = "0" if (change > 0) == minimizing else "1"
                    if change and bit != required:
                        return False
                    coordinate = lower[column] if bit == "0" else upper[column]
                    evaluated += change*Fraction.from_float(float(coordinate))
                if evaluated != expected:
                    return False
            if exact[0] > exact[1] or exact[2] != max(abs(exact[0]), abs(exact[1])):
                return False
            bounds.append(exact[2])
        return type(receipt["maximum_exact"]) is str and receipt["maximum_exact"] == str(max(bounds))
    except (ValueError, KeyError, TypeError, OverflowError, ZeroDivisionError, AttributeError):
        return False


def _affine_forward(artifact: TensorArtifact, features: Any) -> np.ndarray:
    _trained_head(artifact)
    return evaluate_head(artifact, features)


def affine_registry() -> SemanticsRegistry:
    """Opt-in research registry; frozen default/core admission is unchanged."""
    registry = SemanticsRegistry()
    registry.register(SemanticsAdapter(ARCHITECTURE, _affine_forward, certify_affine_box,
        CERTIFICATE_SCOPE, version="signed-affine-box/1"))
    return registry


def check_affine_budget(base: TensorArtifact, candidate: TensorArtifact, box: InputBox,
                        receipt: dict[str, Any], budget: float) -> bool:
    budget = finite_binary64(budget)
    return budget >= 0 and verify_affine_box(base, candidate, box, receipt) and Fraction(receipt["maximum_exact"]) <= Fraction.from_float(budget)


def _inputs() -> tuple[list[dict[str, Any]], dict[str, TensorArtifact], dict[str, np.ndarray]]:
    if file_hash(CUMULATIVE_ROOT/"manifest.json") != CUMULATIVE_MANIFEST:
        raise ValueError("The trusted cumulative candidate manifest changed")
    _, sequences, artifacts = cumulative._verify_seal(CUMULATIVE_ROOT)
    with np.load(CUMULATIVE_ROOT/"arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    return sequences, artifacts, arrays


def _pairs(sequences: list[dict[str, Any]], arrays: dict[str, np.ndarray]):
    for sequence in sequences:
        for record in sequence["records"]:
            request = record["request"]
            features = arrays[request["baseline_features"]]
            box = InputBox(features.min(0), features.max(0))
            yield sequence, record, box


def triangle_certificate(base: TensorArtifact, candidate: TensorArtifact, box: InputBox) -> dict[str, Any]:
    """Reuse the existing identity-chain bound on identical affine endpoint data."""
    def normalize(artifact: TensorArtifact) -> TensorArtifact:
        return sequential_artifact([artifact.tensors[HEAD_WEIGHT]], [artifact.tensors[HEAD_BIAS]], ["identity"],
            input_names=artifact.metadata["feature_names"], output_names=artifact.metadata["output_names"])
    return certify_sequential_drift(normalize(base), normalize(candidate), box)


def profile(directory: str | Path) -> dict[str, Any]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    sequences, artifacts, arrays = _inputs()
    files, sources = {}, {}
    for relative in SOURCE_FILES:
        target = out/"sources"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        sources[relative] = file_hash(target)
        files["sources/"+relative] = sources[relative]
    files["sources.json"] = write_json(out/"sources.json", sources)
    registry, cache, rows = research_registry(), {}, []
    for sequence, record, box in _pairs(sequences, arrays):
        request = record["request"]
        base, candidate = (artifacts[request[key]] for key in ("parent_sha256", "candidate_sha256"))
        key = digest([base.sha256, candidate.sha256, box.to_dict()])
        if key not in cache:
            start = time.perf_counter()
            tracemalloc.start()
            triangle = triangle_certificate(base, candidate, box)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed = time.perf_counter()-start
            try:
                registry.certify(base, candidate, box)
                support = {"status": "pass", "reason": None}
            except ValueError as error:
                support = {"status": "unsupported", "reason": str(error)}
            cache[key] = {"triangle": triangle, "seconds": elapsed, "python_peak_bytes": peak, "registry": support}
        item = cache[key]
        budget = Fraction.from_float(float(request["policy"]["score_budget"]))
        bound = max(map(Fraction, item["triangle"]["output_bounds"]))
        gates = {gate["name"]: gate for gate in record["gates"]}
        observed = Fraction(gates["local_score_budget"]["evidence"]["receipt"]["maximum_exact"])
        status = "pass" if bound <= budget else "observed_violation" if observed > budget else "conservative_rejection"
        rows.append({"request_sha256": record["request_sha256"], "lineage_id": sequence["lineage_id"],
            "domain": sequence["domain"], "order": sequence["order_name"], "controller": request["mode"],
            "step": request["step"], "case_id": request["target"]["case_id"], "pair_sha256": key,
            "parent_sha256": base.sha256, "candidate_sha256": candidate.sha256,
            "calibration_features_sha256": request["baseline_features"], "feature_domain": box.to_dict(),
            "domain_sha256": digest(box.to_dict()), "score_budget_exact": str(budget),
            "triangle_maximum_exact": str(bound), "calibration_maximum_exact": str(observed),
            "registry": item["registry"], "triangle_status": status,
            "calibration_utility_status": gates["new_target_gain"]["status"],
            "triangle_seconds": item["seconds"], "triangle_python_peak_bytes": item["python_peak_bytes"],
            "timing_reused_for_duplicate_pair": sum(row["pair_sha256"] == key for row in rows) > 0,
            "no_op": record["no_op"]})
    payload = {"schema": "wsp-trained-bound-profile/1", "recorded_at_utc": utc_now(),
        "cumulative_manifest_sha256": CUMULATIVE_MANIFEST, "rows": rows,
        "selection": "Before implementing an extension: all assigned cumulative soft proposals, unchanged parents/candidates/caps; no heldout utility opened",
        "domain": "Coordinate min/max box from the frozen protected calibration rows; off-manifold corners are included, raw-input encoder coverage is unproved",
        "categories": "Registry unsupported is separate from a loose sufficient rejection, exact observed calibration violation, and calibration target-gain failure",
        "timing": "Single-run unique pair time with tracemalloc enabled, Python allocation peak only; duplicates reuse it, process/kernel RSS is not measured",
        "heldout_feature_arrays_read": False, "independent_population_evidence": False}
    files["profile.json"] = write_json(out/"profile.json", payload)
    write_json(out/"manifest.json", {"schema": "wsp-bound-profile-manifest/1", "files": files})
    return payload


def _rules() -> dict[str, Any]:
    return {"schema": "wsp-affine-bound-rules/1", "review_date_local": "2026-11-08",
        "profile_manifest_sha256": PROFILE_MANIFEST, "cumulative_manifest_sha256": CUMULATIVE_MANIFEST,
        "selection": "All pre-extension profile records; actual frozen parent-to-proposed-candidate endpoints, feature boxes and binary64 score caps are unchanged",
        "extension": "Exact signed affine feature-box extrema with independently checked per-output attaining corners",
        "comparison": "Existing identity-chain triangle envelope and signed interval on identical endpoints and boxes; no relaxation solver or candidate refitting",
        "timing": "One unique-pair run in fixed triangle-generator, signed-generator, signed-checker order; tracemalloc enabled for each timed operation; Python allocation peak, not process/kernel RSS",
        "duplicate_accounting": "Preserve every assigned step and controller/order; repeated endpoint/domain pairs share certificates and timing",
        "evaluation": "Frozen proposed candidate versus its actual parent; no revised sequence deployment or new admission policy",
        "heldout": "Novel official-test feature rows primary, complete official test secondary; reused earlier-study rows, not independent population confirmation",
        "feature_domain": "Unchanged protected-calibration coordinate box including off-manifold corners; separate in-box and out-of-box heldout observations",
        "violation": "Exact real affine residual exceeding its per-output signed bound on an in-box row; out-of-box rows have no certificate premise",
        "native": "Torch CPU float32 affine scores, separate observed float64 differences of native outputs; no kernel refinement theorem",
        "utility": "Native queue target gain versus current parent at least 0.10 and protected local queue flip/net harm at most 0.01; certificate availability is separate",
        "inference": "Descriptive Student-t intervals over six full-training-lineage means per domain; average repeated steps/orders/controllers inside lineage",
        "missingness": "Every assigned record retained; missing target gain/net harm sensitivity [-1,1], usefulness [0,1]",
        "release": {"path": GLOBAL_GATE_PATH, "schema": "wsp-live-global-selection-seal/1",
                    "prepared_lock_path": GLOBAL_LOCK_PATH, "assigned_total": GLOBAL_ASSIGNED_TOTAL,
                    "canonical_sha256": "Required external argument, never inferred from the gate itself",
                    "check_scope": "Authenticate gate, preparation lock and terminal selection accounting without importing or auditing the live harness"},
        "reference_pdf_sha256": REFERENCE_HASHES,
        "trust": {"implementation_machine_checked": False, "feature_extractor_verified": False,
                  "floating_execution_verified": False, "core_native_admission_integrated": False}}


def _identity(sequence: dict[str, Any], record: dict[str, Any], box: InputBox) -> dict[str, Any]:
    request = record["request"]
    gates = {gate["name"]: gate for gate in record["gates"]}
    return {"request_sha256": record["request_sha256"], "lineage_id": sequence["lineage_id"],
        "domain": sequence["domain"], "order": sequence["order_name"], "controller": request["mode"],
        "step": request["step"], "case_id": request["target"]["case_id"],
        "pair_sha256": digest([request["parent_sha256"], request["candidate_sha256"], box.to_dict()]),
        "parent_sha256": request["parent_sha256"], "candidate_sha256": request["candidate_sha256"],
        "calibration_features_sha256": request["baseline_features"], "feature_domain": box.to_dict(),
        "domain_sha256": digest(box.to_dict()),
        "score_budget_exact": str(Fraction.from_float(float(request["policy"]["score_budget"]))),
        "calibration_maximum_exact": gates["local_score_budget"]["evidence"]["receipt"]["maximum_exact"],
        "calibration_utility_status": gates["new_target_gain"]["status"], "no_op": record["no_op"]}


def _check_profile(assigned: list[tuple[Any, Any, InputBox]]) -> list[dict[str, Any]]:
    if file_hash(PROFILE_ROOT/"manifest.json") != PROFILE_MANIFEST:
        raise ValueError("The externally retained pre-extension profile manifest changed")
    manifest = read_json(PROFILE_ROOT/"manifest.json")
    sources = read_json(PROFILE_ROOT/"sources.json")
    if (manifest["schema"] != "wsp-bound-profile-manifest/1" or set(sources) != set(SOURCE_FILES)
            or set(manifest["files"]) != {"sources/"+key for key in sources} | {"sources.json", "profile.json"}
            or any(manifest["files"]["sources/"+key] != value for key, value in sources.items())):
        raise ValueError("The pre-extension profile source closure is incomplete")
    verify_manifest(PROFILE_ROOT, manifest["files"])
    if {key: sources[key] for key in cumulative.SOURCE_FILES} != read_json(CUMULATIVE_ROOT/"sources.json"):
        raise ValueError("The profile and frozen cumulative execution closures differ")
    profile = read_json(PROFILE_ROOT/"profile.json")
    if (profile["schema"] != "wsp-trained-bound-profile/1"
            or profile["cumulative_manifest_sha256"] != CUMULATIVE_MANIFEST
            or profile["heldout_feature_arrays_read"] is not False
            or profile["independent_population_evidence"] is not False
            or len(profile["rows"]) != len(assigned)):
        raise ValueError("The pre-extension profile boundary or assigned denominator changed")
    seen = set()
    for old, (sequence, record, box) in zip(profile["rows"], assigned):
        identity = _identity(sequence, record, box)
        if (canonical_bytes({key: old[key] for key in identity}) != canonical_bytes(identity)
                or old["timing_reused_for_duplicate_pair"] is not (identity["pair_sha256"] in seen)):
            raise ValueError("The comparison changed an assigned endpoint, box, cap, order or failed/no-op outcome")
        seen.add(identity["pair_sha256"])
    return profile["rows"]


def _timed(function) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    tracemalloc.start()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, {"seconds": time.perf_counter()-started, "python_peak_bytes": peak}


def _status(bound: Fraction, budget: Fraction, observed: Fraction) -> str:
    return "pass" if bound <= budget else "observed_violation" if observed > budget else "conservative_rejection"


def _make_pair(base: TensorArtifact, candidate: TensorArtifact, box: InputBox) -> dict[str, Any]:
    triangle, triangle_time = _timed(lambda: triangle_certificate(base, candidate, box))
    def attempt():
        try:
            return {"status": "pass", "reason": None}, certify_affine_box(base, candidate, box)
        except ValueError as failure:
            return {"status": "unsupported", "reason": str(failure)}, None
    (support, signed), signed_time = _timed(attempt)
    checker_time = None
    if signed is not None:
        valid, checker_time = _timed(lambda: verify_affine_box(base, candidate, box, signed))
        if not valid:
            raise ValueError("The generated signed certificate failed its independent checker")
    return {"schema": "wsp-affine-bound-pair/1",
        "pair_sha256": digest([base.sha256, candidate.sha256, box.to_dict()]),
        "parent_sha256": base.sha256, "candidate_sha256": candidate.sha256,
        "feature_domain": box.to_dict(), "domain_sha256": digest(box.to_dict()),
        "triangle_certificate": triangle, "signed_certificate": signed, "signed_support": support,
        "measurements": {"triangle_generation": triangle_time, "signed_generation": signed_time,
                         "signed_independent_check": checker_time}}


def _check_pair(base: TensorArtifact, candidate: TensorArtifact, box: InputBox, pair: dict[str, Any]) -> None:
    expected = {"schema": "wsp-affine-bound-pair/1",
        "pair_sha256": digest([base.sha256, candidate.sha256, box.to_dict()]),
        "parent_sha256": base.sha256, "candidate_sha256": candidate.sha256,
        "feature_domain": box.to_dict(), "domain_sha256": digest(box.to_dict())}
    if (set(pair) != set(expected) | {"triangle_certificate", "signed_certificate", "signed_support", "measurements"}
            or canonical_bytes({key: pair[key] for key in expected}) != canonical_bytes(expected)
            or canonical_bytes(pair["triangle_certificate"]) != canonical_bytes(triangle_certificate(base, candidate, box))):
        raise ValueError("A paired certificate changed its endpoints/domain or failed triangle replay")
    receipt = pair["signed_certificate"]
    if receipt is not None:
        if pair["signed_support"] != {"status": "pass", "reason": None} or not verify_affine_box(base, candidate, box, receipt):
            raise ValueError("A signed certificate failed independent corner/identity checking")
        triangle = list(map(Fraction, pair["triangle_certificate"]["output_bounds"]))
        if any(Fraction(row["absolute_bound_exact"]) > bound for row, bound in zip(receipt["intervals"], triangle)):
            raise ValueError("The signed bound exceeds the unchanged triangle envelope")
    else:
        try:
            _context(base, candidate, box)
        except ValueError as failure:
            if pair["signed_support"] != {"status": "unsupported", "reason": str(failure)}:
                raise ValueError("Unsupported subset accounting changed") from failure
        else:
            raise ValueError("A supported frozen pair was omitted")
    measurements = pair["measurements"]
    if set(measurements) != {"triangle_generation", "signed_generation", "signed_independent_check"}:
        raise ValueError("Unknown timing measurement scope")
    for name, value in measurements.items():
        if value is None and name == "signed_independent_check" and receipt is None:
            continue
        if (not isinstance(value, dict) or set(value) != {"seconds", "python_peak_bytes"}
                or type(value["seconds"]) is not float or not np.isfinite(value["seconds"]) or value["seconds"] < 0
                or type(value["python_peak_bytes"]) is not int or value["python_peak_bytes"] < 0):
            raise ValueError("Invalid descriptive timing/allocation measurement")


def _comparison_row(sequence, record, box, pair, old, duplicate) -> dict[str, Any]:
    identity = _identity(sequence, record, box)
    triangle = list(map(Fraction, pair["triangle_certificate"]["output_bounds"]))
    maximum = max(triangle)
    budget, observed = Fraction(identity["score_budget_exact"]), Fraction(identity["calibration_maximum_exact"])
    if old["triangle_maximum_exact"] != str(maximum) or old["triangle_status"] != _status(maximum, budget, observed):
        raise ValueError("The unchanged baseline bound/category failed pre-extension profile replay")
    receipt = pair["signed_certificate"]
    signed_maximum = None if receipt is None else Fraction(receipt["maximum_exact"])
    if signed_maximum is not None and observed > signed_maximum:
        raise ValueError("An exact calibration residual exceeds the signed feature-box certificate")
    triangle_widths = [2*value for value in triangle]
    signed_widths = None if receipt is None else [Fraction(row["upper_exact"])-Fraction(row["lower_exact"]) for row in receipt["intervals"]]
    return identity | {"proposal_failed": record["request"]["proposal_error"] is not None,
        "prior_admission_accepted": record["accepted"], "prior_deployed_sha256": record["deployed_sha256"],
        "baseline_registry": old["registry"], "signed_registry": pair["signed_support"],
        "triangle_output_bounds_exact": list(map(str, triangle)), "triangle_maximum_exact": str(maximum),
        "triangle_interval_widths_exact": list(map(str, triangle_widths)),
        "signed_output_bounds_exact": None if receipt is None else [row["absolute_bound_exact"] for row in receipt["intervals"]],
        "signed_maximum_exact": None if signed_maximum is None else str(signed_maximum),
        "signed_interval_widths_exact": None if signed_widths is None else list(map(str, signed_widths)),
        "maximum_bound_ratio_exact": None if signed_maximum is None or maximum == 0 else str(signed_maximum/maximum),
        "total_interval_width_ratio_exact": None if signed_widths is None or sum(triangle_widths) == 0 else str(sum(signed_widths)/sum(triangle_widths)),
        "triangle_status": old["triangle_status"],
        "signed_status": "unsupported" if signed_maximum is None else _status(signed_maximum, budget, observed),
        "triangle_certificate_sha256": pair["triangle_certificate"]["sha256"],
        "signed_certificate_sha256": None if receipt is None else receipt["sha256"],
        "measurements": pair["measurements"], "timing_reused_for_duplicate_pair": duplicate,
        "pre_extension_profile_measurements": {"seconds": old["triangle_seconds"], "python_peak_bytes": old["triangle_python_peak_bytes"]}}


def _counts(values, categories):
    values = list(values)
    return {name: sum(value == name for value in values) for name in categories}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains = {}
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in rows if row["domain"] == domain]
        unique = list({row["pair_sha256"]: row for row in selected}.values())
        ratios = [float(Fraction(row["maximum_bound_ratio_exact"])) for row in unique if row["maximum_bound_ratio_exact"] is not None]
        widths = [float(Fraction(row["total_interval_width_ratio_exact"])) for row in unique if row["total_interval_width_ratio_exact"] is not None]
        measurements = {}
        for operation in ("triangle_generation", "signed_generation", "signed_independent_check"):
            observed = [row["measurements"][operation] for row in unique if row["measurements"][operation] is not None]
            measurements[operation] = {"median_seconds": median(value["seconds"] for value in observed) if observed else None,
                "median_python_peak_bytes": median(value["python_peak_bytes"] for value in observed) if observed else None}
        categories = ("pass", "conservative_rejection", "observed_violation", "unsupported")
        domains[domain] = {"assigned_step_records": len(selected), "distinct_endpoint_domain_pairs": len(unique),
            "no_op_records": sum(row["no_op"] for row in selected), "proposal_failure_records": sum(row["proposal_failed"] for row in selected),
            "baseline_registry": _counts((row["baseline_registry"]["status"] for row in selected), ("pass", "unsupported")),
            "signed_registry": _counts((row["signed_registry"]["status"] for row in selected), ("pass", "unsupported")),
            "triangle_categories": _counts([row["triangle_status"] for row in selected], categories),
            "signed_categories": _counts([row["signed_status"] for row in selected], categories),
            "calibration_target_gain": _counts([row["calibration_utility_status"] for row in selected], ("pass", "observed_violation")),
            "median_unique_pair_maximum_bound_ratio": median(ratios) if ratios else None,
            "median_unique_pair_total_interval_width_ratio": median(widths) if widths else None,
            "measurements": measurements}
    return {"domains": domains, "denominators": "Assigned records and distinct pairs are coverage accounting, never task-impact claims",
            "timing": "Recorded single-run measurements validated for scope/type and retained by identity; elapsed time/allocation peaks are not exactly reproducible"}


def seal(directory: str | Path) -> dict[str, Any]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    sequences, artifacts, arrays = _inputs()
    assigned = list(_pairs(sequences, arrays))
    profile_rows = _check_profile(assigned)
    sources, files = {}, {}
    for relative in SOURCE_FILES:
        target = out/"sources"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write((ROOT/relative).read_bytes())
        sources[relative] = file_hash(target)
        files["sources/"+relative] = sources[relative]
    files["sources.json"] = write_json(out/"sources.json", sources)
    files["protocol.json"] = write_json(out/"protocol.json", {"schema": "wsp-affine-bound-protocol/1",
        "frozen_at_utc": utc_now(), "rules": _rules(), "environment": environment()})
    register = read_json(ROOT/"docs/weight-space/v2/research-sources-2026-11.json")
    references = [row for row in register["sources"] if row["id"] in REFERENCE_HASHES]
    if {row["id"]: row["sha256"] for row in references} != REFERENCE_HASHES:
        raise ValueError("The selected reviewed paper versions changed")
    files["reviewed-sources.json"] = write_json(out/"reviewed-sources.json", references)
    pairs, rows, seen = {}, [], set()
    for (sequence, record, box), old in zip(assigned, profile_rows):
        request = record["request"]
        base, candidate = (artifacts[request[key]] for key in ("parent_sha256", "candidate_sha256"))
        key = old["pair_sha256"]
        if key not in pairs:
            pairs[key] = _make_pair(base, candidate, box)
            _check_pair(base, candidate, box, pairs[key])
        rows.append(_comparison_row(sequence, record, box, pairs[key], old, key in seen))
        seen.add(key)
    files["pairs.json"] = write_json(out/"pairs.json", pairs)
    files["comparison.json"] = write_json(out/"comparison.json", {"schema": "wsp-affine-bound-comparison/1",
        "rows": rows, "summary": _summary(rows), "heldout_feature_arrays_read": False,
        "independent_population_evidence": False})
    files["selection-seal.json"] = write_json(out/"selection-seal.json", {
        "schema": "wsp-affine-bound-selection-seal/1", "sealed_at_utc": utc_now(),
        "files": dict(files), "cumulative_manifest_sha256": CUMULATIVE_MANIFEST,
        "profile_manifest_sha256": PROFILE_MANIFEST, "all_assigned_records_accounted": True,
        "evaluation_feature_arrays_read": False})
    manifest_hash = write_json(out/"manifest.json", {"schema": "wsp-affine-bound-manifest/1", "files": files,
        "scope": "Local source/calibration comparison seal; external manifest identity required for authentication"})
    return {"manifest_sha256": manifest_hash, "summary": _summary(rows)}


def _verify_seal(directory: str | Path, expected_manifest_sha256: str):
    out = Path(directory)
    if (not cumulative._is_hash(expected_manifest_sha256)
            or file_hash(out/"manifest.json") != expected_manifest_sha256):
        raise ValueError("The affine comparison manifest differs from its externally supplied identity")
    manifest = read_json(out/"manifest.json")
    expected_files = {"sources/"+key for key in SOURCE_FILES} | {
        "sources.json", "protocol.json", "reviewed-sources.json", "pairs.json", "comparison.json", "selection-seal.json"}
    if manifest["schema"] != "wsp-affine-bound-manifest/1" or set(manifest["files"]) != expected_files:
        raise ValueError("The affine comparison omits part of its source or evidence closure")
    verify_manifest(out, manifest["files"])
    sources = read_json(out/"sources.json")
    if (set(sources) != set(SOURCE_FILES)
            or any(file_hash(ROOT/key) != value or manifest["files"]["sources/"+key] != value for key, value in sources.items())
            or read_json(out/"protocol.json")["rules"] != _rules()):
        raise ValueError("The affine execution closure or pre-evaluation protocol changed")
    selection = read_json(out/"selection-seal.json")
    if (selection["schema"] != "wsp-affine-bound-selection-seal/1"
            or selection["files"] != {key: value for key, value in manifest["files"].items() if key != "selection-seal.json"}
            or selection["cumulative_manifest_sha256"] != CUMULATIVE_MANIFEST
            or selection["profile_manifest_sha256"] != PROFILE_MANIFEST
            or selection["all_assigned_records_accounted"] is not True
            or selection["evaluation_feature_arrays_read"] is not False
            or {row["id"]: row["sha256"] for row in read_json(out/"reviewed-sources.json")} != REFERENCE_HASHES):
        raise ValueError("The affine selection/source-review boundary changed")
    sequences, artifacts, arrays = _inputs()
    assigned = list(_pairs(sequences, arrays))
    profile_rows = _check_profile(assigned)
    pairs, rows, checked = read_json(out/"pairs.json"), [], set()
    if set(pairs) != {row["pair_sha256"] for row in profile_rows}:
        raise ValueError("The affine comparison omitted or added a frozen endpoint/domain pair")
    for (sequence, record, box), old in zip(assigned, profile_rows):
        key = old["pair_sha256"]
        if key not in checked:
            _check_pair(artifacts[old["parent_sha256"]], artifacts[old["candidate_sha256"]], box, pairs[key])
        rows.append(_comparison_row(sequence, record, box, pairs[key], old, key in checked))
        checked.add(key)
    expected = {"schema": "wsp-affine-bound-comparison/1", "rows": rows, "summary": _summary(rows),
                "heldout_feature_arrays_read": False, "independent_population_evidence": False}
    if canonical_bytes(read_json(out/"comparison.json")) != canonical_bytes(expected):
        raise ValueError("The affine calibration categories/widths or assigned records failed replay")
    return sequences, artifacts, pairs, rows


def verify_seal(directory: str | Path, expected_manifest_sha256: str) -> bool:
    _verify_seal(directory, expected_manifest_sha256)
    return True


def _authenticated_json(path: Path, schema: str) -> dict[str, Any]:
    value = read_json(path)
    if (not isinstance(value, dict) or value.get("schema") != schema
            or not cumulative._is_hash(value.get("sha256"))
            or digest({key: item for key, item in value.items() if key != "sha256"}) != value["sha256"]):
        raise ValueError("A release document failed canonical identity checking")
    return value


def _global_release(gate_path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Authenticate root's release binding, without importing its evolving harness."""
    path = Path(gate_path)
    path = (path if path.is_absolute() else ROOT/path).resolve()
    if (not cumulative._is_hash(expected_sha256) or not path.is_relative_to(ROOT.resolve())
            or path != (ROOT/GLOBAL_GATE_PATH).resolve()):
        raise ValueError("An explicit external release hash and the fixed repository gate path are required")
    gate = _authenticated_json(path, "wsp-live-global-selection-seal/1")
    fields = {"schema", "phase", "selection_frozen", "all_assigned_accounted", "studies",
        "prepared_lock_path", "prepared_lock_sha256", "protocol_sha256", "assigned_total", "terminal_total",
        "all_terminal", "evaluation_started", "accounting", "sealed_utc", "sha256"}
    lock_path = (ROOT/GLOBAL_LOCK_PATH).resolve()
    if (set(gate) != fields or gate["sha256"] != expected_sha256
            or gate["phase"] != "selection_complete_before_evaluation"
            or gate["selection_frozen"] is not True or gate["all_assigned_accounted"] is not True
            or gate["all_terminal"] is not True or gate["evaluation_started"] is not False
            or gate["studies"] != ["blind-diagnosis", "weight-access"]
            or gate["prepared_lock_path"] != GLOBAL_LOCK_PATH or not lock_path.is_relative_to(ROOT.resolve())
            or gate["prepared_lock_sha256"] != file_hash(lock_path)
            or type(gate["assigned_total"]) is not int or gate["assigned_total"] != GLOBAL_ASSIGNED_TOTAL
            or type(gate["terminal_total"]) is not int or gate["terminal_total"] != GLOBAL_ASSIGNED_TOTAL
            or not isinstance(gate["accounting"], list) or len(gate["accounting"]) != GLOBAL_ASSIGNED_TOTAL):
        raise ValueError("The authenticated global selection gate is incomplete or changed")
    lock = _authenticated_json(lock_path, "wsp-live-preparation-lock/1")
    if (lock.get("selection_only") is not True or lock.get("evaluation_started") is not False
            or lock.get("provider_calls_started") is not False
            or type(lock.get("assigned_total")) is not int or lock["assigned_total"] != GLOBAL_ASSIGNED_TOTAL
            or not isinstance(lock.get("assignments"), list) or len(lock["assignments"]) != GLOBAL_ASSIGNED_TOTAL
            or lock.get("protocol_sha256") != gate["protocol_sha256"]
            or not isinstance(gate["protocol_sha256"], dict) or not gate["protocol_sha256"]
            or any(not cumulative._is_hash(value) for value in gate["protocol_sha256"].values())):
        raise ValueError("The release lock does not bind the complete pre-evaluation selection")
    locked_ids = [row.get("assignment_id") for row in lock["assignments"] if isinstance(row, dict)]
    if (len(locked_ids) != GLOBAL_ASSIGNED_TOTAL
            or any(not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{24}", identity) for identity in locked_ids)
            or len(set(locked_ids)) != GLOBAL_ASSIGNED_TOTAL):
        raise ValueError("The release lock contains missing, duplicate or unsafe assignment IDs")
    accounts = {}
    for row in gate["accounting"]:
        if (not isinstance(row, dict) or set(row) != {"assignment_id", "selection_sha256", "terminal_status"}
                or not isinstance(row["assignment_id"], str) or row["assignment_id"] not in locked_ids
                or row["assignment_id"] in accounts or not cumulative._is_hash(row["selection_sha256"])
                or not isinstance(row["terminal_status"], str) or row["terminal_status"] not in TERMINAL_STATUSES):
            raise ValueError("The release accounting omits or changes a terminal assignment")
        accounts[row["assignment_id"]] = row
    if set(accounts) != set(locked_ids):
        raise ValueError("The release accounting differs from the locked assignments")
    for identity, account in accounts.items():
        selection_path = (ROOT/"artifacts/research/live-studies/selections"/identity/"selection.json").resolve()
        if not selection_path.is_relative_to(ROOT.resolve()):
            raise ValueError("A released selection path escapes the repository")
        selected = _authenticated_json(selection_path, "wsp-live-selection/1")
        if (selected.get("assignment_id") != identity or selected["sha256"] != account["selection_sha256"]
                or selected.get("terminal_status") != account["terminal_status"]
                or selected.get("evaluation_started") is not False):
            raise ValueError("A released terminal selection differs from the authenticated accounting")
    return {"path": GLOBAL_GATE_PATH, "canonical_sha256": expected_sha256, "file_sha256": file_hash(path),
        "prepared_lock_path": GLOBAL_LOCK_PATH, "prepared_lock_file_sha256": gate["prepared_lock_sha256"],
        "prepared_lock_canonical_sha256": lock["sha256"], "assigned_total": GLOBAL_ASSIGNED_TOTAL,
        "check_scope": _rules()["release"]["check_scope"]}


def _integer_lift(values: np.ndarray) -> tuple[np.ndarray, int]:
    """A common power-of-two denominator preserves each stored binary value."""
    ratios = [float(value).as_integer_ratio() for value in values.flat]
    denominator = max((item[1] for item in ratios), default=1)
    lifted = np.array([numerator*(denominator//divisor) for numerator, divisor in ratios], dtype=object)
    return lifted.reshape(values.shape), denominator


def _real_residuals(base: TensorArtifact, candidate: TensorArtifact, features: np.ndarray) -> tuple[np.ndarray, int]:
    """Independent endpoint subtraction and exact integer affine execution."""
    _trained_head(base)
    _trained_head(candidate)
    if not base.compatible(candidate):
        raise ValueError("Exact observation endpoints differ in their declared trained interface")
    before, after = base.tensors, candidate.tensors
    first, first_den = _integer_lift(np.column_stack((before[HEAD_WEIGHT], before[HEAD_BIAS])))
    second, second_den = _integer_lift(np.column_stack((after[HEAD_WEIGHT], after[HEAD_BIAS])))
    denominator = max(first_den, second_den)
    delta = second*(denominator//second_den)-first*(denominator//first_den)
    lifted, feature_den = _integer_lift(features)
    residuals = lifted @ delta[:, :-1].T + delta[:, -1]*feature_den
    return residuals, denominator*feature_den


def _real_summary(residuals: np.ndarray, denominator: int, selected: np.ndarray,
                  signed_bounds: list[Fraction] | None) -> dict[str, Any]:
    values = residuals[selected]
    if not len(values):
        return {"rows": 0, "output_maxima_exact": None, "maximum_exact": None,
                "signed_bound_violation_rows": None if signed_bounds is None else 0}
    maxima = [max(abs(int(value)) for value in values[:, output]) for output in range(values.shape[1])]
    violations = None
    if signed_bounds is not None:
        violated = np.zeros(len(values), dtype=bool)
        for output, bound in enumerate(signed_bounds):
            violated |= np.asarray(np.abs(values[:, output])*bound.denominator > bound.numerator*denominator, dtype=bool)
        violations = int(violated.sum())
    return {"rows": len(values), "output_maxima_exact": [str(Fraction(value, denominator)) for value in maxima],
        "maximum_exact": str(Fraction(max(maxima), denominator)), "signed_bound_violation_rows": violations}


def _native_summary(drift: np.ndarray, selected: np.ndarray, signed_bounds: list[Fraction] | None,
                    triangle_bounds: list[Fraction], budget: Fraction, in_box: bool) -> dict[str, Any]:
    values = drift[selected]
    def exceedances(bounds):
        if bounds is None:
            return None
        return int(np.any(values > np.array([float(value) for value in bounds]), axis=1).sum())
    return {"rows": len(values), "maximum_absolute_score_change": float(values.max()) if len(values) else None,
        "signed_bound_exceedance_rows": exceedances(signed_bounds),
        "triangle_bound_exceedance_rows": exceedances(triangle_bounds),
        "score_cap_exceedance_rows": int(np.any(values > float(budget), axis=1).sum()),
        "feature_box_premise": in_box, "certificate_violation_claim": False,
        "comparison": "Binary64 absolute differences of native float32 outputs against binary64 projections of real bounds and cap; no floating-execution refinement"}


def _validate_evaluation(bundle: dict[str, np.ndarray], base: TensorArtifact) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, outputs = _trained_head(base).dimensions(base)
    values = cumulative._feature_matrix(bundle["test_features"], width)
    labels, novel = np.asarray(bundle["test_labels"]), np.asarray(bundle["test_novel_mask"])
    if (labels.shape != (len(values),) or labels.dtype.kind not in "iu" or (labels < 0).any() or (labels >= outputs).any()
            or novel.shape != labels.shape or novel.dtype.kind != "b"
            or any(np.asarray(bundle[key]).shape != labels.shape for key in ("test_ids", "test_group_ids"))
            or not np.array_equal(values, values.astype(np.float32).astype(values.dtype))):
        raise ValueError("The fixed test features, labels, identity arrays or float32 input correspondence differ")
    return values, labels, novel


def _pair_observation(base: TensorArtifact, candidate: TensorArtifact, box: InputBox,
                      pair: dict[str, Any], bundle: dict[str, np.ndarray], budget: Fraction) -> dict[str, Any]:
    values, _, novel = _validate_evaluation(bundle, base)
    before, after = evaluate_head(base, values, native=True), evaluate_head(candidate, values, native=True)
    drift = np.abs(after.astype(np.float64)-before.astype(np.float64))
    in_box = np.all((values >= box.lower) & (values <= box.upper), axis=1)
    signed = pair["signed_certificate"]
    signed_bounds = None if signed is None else [Fraction(row["absolute_bound_exact"]) for row in signed["intervals"]]
    triangle_bounds = list(map(Fraction, pair["triangle_certificate"]["output_bounds"]))
    residuals, denominator = _real_residuals(base, candidate, values[in_box])
    observations = {}
    for name, selected in (("primary_novel", novel), ("secondary_official", np.ones(len(values), dtype=bool))):
        inside, outside = selected & in_box, selected & ~in_box
        observations[name] = {"rows": int(selected.sum()), "in_box_rows": int(inside.sum()),
            "out_of_box_rows": int(outside.sum()),
            "in_box": {"exact_real": _real_summary(residuals, denominator, selected[in_box], signed_bounds),
                "native": _native_summary(drift, inside, signed_bounds, triangle_bounds, budget, True)},
            "out_of_box": {"native": _native_summary(drift, outside, signed_bounds, triangle_bounds, budget, False),
                "real_certificate_premise": False, "real_certificate_violation_claim": False}}
    return {"parent_codes": before.argmax(1), "candidate_codes": after.argmax(1), "observations": observations}


def _utility(sequence, record, comparison, bundle, observation, novel_only):
    labels = np.asarray(bundle["test_labels"])
    selected = np.asarray(bundle["test_novel_mask"]) if novel_only else np.ones(len(labels), dtype=bool)
    queue = np.asarray(sequence["contract"]["queue_map"])
    before, after = queue[observation["parent_codes"]], queue[observation["candidate_codes"]]
    target, policy = record["request"]["target"], record["request"]["policy"]
    chosen = selected & (labels == target["target_label"])
    protected = selected & ~np.isin(labels, [item["target_label"] for item in sequence["contract"]["order"]])
    original = float(np.mean(before[chosen] == target["desired_queue"])) if chosen.any() else None
    candidate = float(np.mean(after[chosen] == target["desired_queue"])) if chosen.any() else None
    gain = None if original is None else candidate-original
    expected = queue[labels]
    harm = (float(np.mean((after[protected] != expected[protected]).astype(np.float64)
                         -(before[protected] != expected[protected]).astype(np.float64))) if protected.any() else None)
    flip = float(np.mean(before[protected] != after[protected])) if protected.any() else None
    missing = gain is None or harm is None or flip is None
    useful = (not missing and gain >= policy["minimum_new_target_gain"]
        and harm <= policy["maximum_local_net_harm"] and flip <= policy["maximum_local_decision_flip"])
    return {"missing": missing, "target_rows": int(chosen.sum()), "protected_rows": int(protected.sum()),
        "parent_target_accuracy": original, "candidate_target_accuracy": candidate, "target_gain": gain,
        "protected_local_net_harm": harm, "protected_local_decision_flip": flip, "utility_pass": bool(useful),
        "signed_available_and_utility_pass": bool(useful and comparison["signed_status"] == "pass"),
        "triangle_available_and_utility_pass": bool(useful and comparison["triangle_status"] == "pass"),
        "observations": observation["observations"]["primary_novel" if novel_only else "secondary_official"],
        "sensitivity": {"target_gain": [-1, 1], "protected_net_harm": [-1, 1], "protected_flip": [0, 1], "useful": [0, 1]} if missing else None}


def _missing_outcome() -> dict[str, Any]:
    return {"missing": True, "target_rows": None, "protected_rows": None,
        "parent_target_accuracy": None, "candidate_target_accuracy": None, "target_gain": None,
        "protected_local_net_harm": None, "protected_local_decision_flip": None, "utility_pass": False,
        "signed_available_and_utility_pass": False, "triangle_available_and_utility_pass": False, "observations": None,
        "sensitivity": {"target_gain": [-1, 1], "protected_net_harm": [-1, 1], "protected_flip": [0, 1], "useful": [0, 1]}}


def _evaluation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scripts.run_trained_research import mean_interval
    summaries = []
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in rows if row["domain"] == domain]
        for population in ("primary_novel", "secondary_official"):
            lineages = []
            for identity in sorted({row["lineage_id"] for row in selected}):
                assigned = [row for row in selected if row["lineage_id"] == identity]
                measured = [row[population] for row in assigned]
                count = len(assigned)
                gain = sum(row["target_gain"] or 0. for row in measured)/count
                harm = sum(row["protected_local_net_harm"] or 0. for row in measured)/count
                flip = sum(row["protected_local_decision_flip"] or 0. for row in measured)/count
                missing_gain = sum(row["target_gain"] is None for row in measured)/count
                missing_harm = sum(row["protected_local_net_harm"] is None for row in measured)/count
                missing_flip = sum(row["protected_local_decision_flip"] is None for row in measured)/count
                missing_useful = sum(row["missing"] for row in measured)/count
                usefulness = sum(row["utility_pass"] for row in measured)/count
                signed = sum(row["signed_available_and_utility_pass"] for row in measured)/count
                triangle = sum(row["triangle_available_and_utility_pass"] for row in measured)/count
                lineages.append({"lineage_id": identity, "assigned_records": count,
                    "missing_records": sum(row["missing"] for row in measured),
                    "target_gain_zero_missing": gain, "protected_net_harm_zero_missing": harm,
                    "protected_flip_zero_missing": flip, "utility_pass_coverage": usefulness,
                    "signed_available_and_useful_coverage": signed, "triangle_available_and_useful_coverage": triangle,
                    "signed_minus_triangle_useful_coverage": signed-triangle,
                    "gain_missing_bounds": [gain-missing_gain, gain+missing_gain],
                    "harm_missing_bounds": [harm-missing_harm, harm+missing_harm],
                    "flip_missing_bounds": [flip, flip+missing_flip],
                    "useful_missing_bounds": [usefulness, usefulness+missing_useful],
                    "signed_useful_missing_bounds": [signed, signed+sum(row[population]["missing"] and row["signed_status"] == "pass" for row in assigned)/count],
                    "triangle_useful_missing_bounds": [triangle, triangle+sum(row[population]["missing"] and row["triangle_status"] == "pass" for row in assigned)/count]})
            summaries.append({"domain": domain, "population": population, "lineages": lineages,
                "target_gain": mean_interval([row["target_gain_zero_missing"] for row in lineages]),
                "protected_local_net_harm": mean_interval([row["protected_net_harm_zero_missing"] for row in lineages]),
                "protected_local_decision_flip": mean_interval([row["protected_flip_zero_missing"] for row in lineages]),
                "utility_pass_coverage": mean_interval([row["utility_pass_coverage"] for row in lineages]),
                "signed_available_and_useful_coverage": mean_interval([row["signed_available_and_useful_coverage"] for row in lineages]),
                "triangle_available_and_useful_coverage": mean_interval([row["triangle_available_and_useful_coverage"] for row in lineages]),
                "signed_minus_triangle_useful_coverage": mean_interval([row["signed_minus_triangle_useful_coverage"] for row in lineages]),
                "inference_unit": "Complete trained lineage; repeated assigned steps, orders, controllers and pair observations are averaged within lineage"})
    return summaries


def _evaluate(sequences, artifacts, pairs, comparison_rows) -> dict[str, Any]:
    outcomes, evaluation_hashes, bundles, observations = [], {}, {}, {}
    index = {row["request_sha256"]: row for row in comparison_rows}
    if len(index) != len(comparison_rows):
        raise ValueError("Assigned comparison request identities are not unique")
    for sequence in sequences:
        identity = sequence["lineage_id"]
        directory = (ROOT/sequence["lineage_directory"]).resolve()
        failure = None
        try:
            if not directory.is_relative_to(ROOT.resolve()):
                raise ValueError("A frozen evaluation lineage path escapes the repository")
            for name in ("evaluation-features.json", "evaluation-features.npz"):
                path = directory/name
                evaluation_hashes[path.relative_to(ROOT).as_posix()] = file_hash(path)
            if identity not in bundles:
                bundles[identity] = cumulative._load_evaluation(directory, sequence["contract"]["head"]["feature_id"])
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, IndexError) as error:
            failure = type(error).__name__
        for record in sequence["records"]:
            comparison = index[record["request_sha256"]]
            outcome = {key: comparison[key] for key in ("request_sha256", "lineage_id", "domain", "order", "controller",
                "step", "case_id", "pair_sha256", "parent_sha256", "candidate_sha256", "score_budget_exact",
                "no_op", "proposal_failed", "prior_admission_accepted", "triangle_status", "signed_status",
                "triangle_certificate_sha256", "signed_certificate_sha256")}
            outcome["assigned"] = True
            try:
                if failure is not None:
                    raise ValueError("The fixed evaluation bundle is unavailable")
                key = digest([comparison["pair_sha256"], comparison["score_budget_exact"]])
                if key not in observations:
                    pair = pairs[comparison["pair_sha256"]]
                    observations[key] = _pair_observation(artifacts[comparison["parent_sha256"]],
                        artifacts[comparison["candidate_sha256"]], InputBox(**comparison["feature_domain"]),
                        pair, bundles[identity], Fraction(comparison["score_budget_exact"]))
                outcome["primary_novel"] = _utility(sequence, record, comparison, bundles[identity], observations[key], True)
                outcome["secondary_official"] = _utility(sequence, record, comparison, bundles[identity], observations[key], False)
            except (ValueError, KeyError, OSError, RuntimeError, TypeError, IndexError, OverflowError) as error:
                outcome["primary_novel"], outcome["secondary_official"] = _missing_outcome(), _missing_outcome()
                outcome["failure"] = {"type": failure or type(error).__name__,
                    "message": "Fixed evaluation input or native/real pair observation unavailable; assigned record retained"}
            outcomes.append(outcome)
    if len(outcomes) != len(comparison_rows) or {row["request_sha256"] for row in outcomes} != set(index):
        raise ValueError("Heldout accounting differs from the pre-evaluation assigned comparison")
    return {"schema": "wsp-affine-bound-results/1", "outcomes": outcomes, "summaries": _evaluation_summary(outcomes),
        "evaluation_input_hashes": evaluation_hashes, "test_counts_are_impact": False,
        "designation": "Secondary fixed-proposal comparisons on previously acquired public test rows; no new selection, sequence deployment or independent population confirmation",
        "native_admission_policy_changed": False, "trust": _rules()["trust"],
        "missingness_rule": "Zero-filled descriptive gain/harm/flip estimates and failure-filled usefulness; explicit per-lineage sensitivity bounds retain every assigned record"}


def evaluate(directory: str | Path, expected_manifest_sha256: str, gate_path: str | Path,
             expected_gate_sha256: str) -> dict[str, Any]:
    out = Path(directory)
    if (out/"results.json").exists() or (out/"evaluation-manifest.json").exists():
        raise FileExistsError("Affine evaluation evidence already exists")
    release = _global_release(gate_path, expected_gate_sha256)
    sequences, artifacts, pairs, rows = _verify_seal(out, expected_manifest_sha256)
    results = _evaluate(sequences, artifacts, pairs, rows)
    results |= {"candidate_manifest_sha256": expected_manifest_sha256,
                "selection_seal_sha256": file_hash(out/"selection-seal.json"), "global_release": release}
    result_hash = write_json(out/"results.json", results)
    write_json(out/"evaluation-manifest.json", {"schema": "wsp-affine-bound-evaluation-manifest/1",
        "candidate_manifest_sha256": expected_manifest_sha256, "global_gate_canonical_sha256": expected_gate_sha256,
        "files": {"results.json": result_hash}})
    return results


def verify(directory: str | Path, expected_manifest_sha256: str, gate_path: str | Path,
           expected_gate_sha256: str) -> bool:
    out = Path(directory)
    release = _global_release(gate_path, expected_gate_sha256)
    sequences, artifacts, pairs, rows = _verify_seal(out, expected_manifest_sha256)
    manifest = read_json(out/"evaluation-manifest.json")
    if (set(manifest) != {"schema", "candidate_manifest_sha256", "global_gate_canonical_sha256", "files"}
            or manifest["schema"] != "wsp-affine-bound-evaluation-manifest/1"
            or manifest["candidate_manifest_sha256"] != expected_manifest_sha256
            or manifest["global_gate_canonical_sha256"] != expected_gate_sha256
            or set(manifest["files"]) != {"results.json"}):
        raise ValueError("Affine evaluation evidence differs from its candidate or release identity")
    verify_manifest(out, manifest["files"])
    actual = _evaluate(sequences, artifacts, pairs, rows)
    actual |= {"candidate_manifest_sha256": expected_manifest_sha256,
               "selection_seal_sha256": file_hash(out/"selection-seal.json"), "global_release": release}
    if canonical_bytes(actual) != canonical_bytes(read_json(out/"results.json")):
        raise ValueError("Affine real/native observations, utility or assigned-lineage uncertainty failed replay")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile", "seal", "verify-seal", "evaluate", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--global-gate", type=Path)
    parser.add_argument("--global-gate-sha256")
    args = parser.parse_args(argv)
    if args.action in ("verify-seal", "evaluate", "verify") and not cumulative._is_hash(args.manifest_sha256):
        parser.error("--manifest-sha256 is required as an externally retained identity")
    if args.action in ("evaluate", "verify") and (args.global_gate is None or not cumulative._is_hash(args.global_gate_sha256)):
        parser.error("--global-gate and --global-gate-sha256 require the explicit coordinated release")
    if args.action in ("profile", "seal"):
        result = {"profile": profile, "seal": seal}[args.action](args.directory)
    elif args.action == "verify-seal":
        result = verify_seal(args.directory, args.manifest_sha256)
    else:
        result = {"evaluate": evaluate, "verify": verify}[args.action](args.directory, args.manifest_sha256,
                                                                   args.global_gate, args.global_gate_sha256)
    response = {"action": args.action, "completed": bool(result)}
    if args.action == "seal":
        response |= result
    print(json.dumps(response, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
