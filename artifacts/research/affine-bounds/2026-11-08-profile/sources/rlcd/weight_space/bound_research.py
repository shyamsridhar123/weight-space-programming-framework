"""Bottleneck-first, secondary affine-head domain research.

The first profile reads only the already sealed cumulative calibration pool.
It neither opens heldout features nor changes a candidate or score budget.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import shutil
import time
import tracemalloc
from typing import Any

import numpy as np

from . import cumulative_research as cumulative
from .artifact import TensorArtifact
from .evidence import ROOT, file_hash, read_json, utc_now, verify_manifest, write_json
from .model import InputBox, digest
from .research_tasks import HEAD_BIAS, HEAD_WEIGHT, research_registry
from .sequential import certify_sequential_drift, sequential_artifact


SOURCE_FILES = cumulative.SOURCE_FILES + ("rlcd/weight_space/bound_research.py",)
CUMULATIVE_ROOT = ROOT/"artifacts/research/cumulative-admission/2026-11-08"
CUMULATIVE_MANIFEST = "208802510de6675f1ae09efcac54932c4dae9651df5d9d08f395e1b364b1309d"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile",))
    parser.add_argument("directory")
    args = parser.parse_args(argv)
    profile(args.directory)
    print('{"action":"profile","completed":true}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
