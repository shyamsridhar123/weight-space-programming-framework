"""Independently audit an existing frozen weight-space study without executing it.

Uses only the standard library and NumPy. In particular, it does not import the
study's model, endpoint, certificate, alignment, selection, or bootstrap code.
Saved tensors are evaluated here; no training, data generation, or simulation
occurs. Write the audit outside the study directory to keep evidence immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any, Callable

import numpy as np


TENSORS = ("w1", "b1", "w2", "b2")
CONTROLS = ("control_noop", "control_b2_shift")
RESOURCES = ("food", "wood", "gold", "stone")
RTS_FEATURES = """food_shortage wood_shortage gold_shortage stone_shortage idle_workers
worker_deficit population_pressure army_readiness visible_base_threat production_deficit
age_progress resource_surplus visible_army_advantage army_deficit candidate_coverage_gap
candidate_order_disruption gather_food gather_wood gather_gold gather_stone train_worker
build_house build_farm build_production train_army advance_age research defend attack scout
build_defense continue_orders""".split()
ATOL, RTOL = 1e-12, 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"),
                      object_pairs_hook=json_pairs, parse_constant=invalid_constant)


def contained(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "Empty evidence path")
    target = (root / relative).resolve()
    require(target.is_relative_to(root.resolve()), f"Path escapes evidence root: {relative}")
    return target


def q(value: Any) -> Fraction:
    require(type(value) in (int, float) or isinstance(value, np.floating), "Expected numeric scalar")
    number = float(value)
    require(math.isfinite(number), "Expected finite binary64 scalar")
    if type(value) is int:
        require(int(number) == value, "Integer is not exactly representable in binary64")
    return Fraction.from_float(number)


def arrays(model: dict[str, Any]) -> dict[str, np.ndarray]:
    require(set(model) == {"schema", "activation", "dtype", "input_names", "output_names", *TENSORS},
            "Unexpected network fields")
    require(model["schema"] == "wsp-network/1" and model["activation"] == "relu"
            and model["dtype"] == "float64", "Unsupported network semantics")
    result = {}
    for name in TENSORS:
        values = np.asarray(model[name], dtype=object)
        require(all(type(v) in (int, float) and math.isfinite(float(v)) for v in values.flat),
                f"Invalid tensor values: {name}")
        result[name] = np.asarray(values, dtype=np.float64)
        require(result[name].size > 0, f"Empty tensor: {name}")
    w, b, v, c = (result[k] for k in TENSORS)
    require(w.ndim == 2 and b.shape == (w.shape[0],) and v.ndim == 2
            and v.shape[1] == w.shape[0] and c.shape == (v.shape[0],), "Incompatible tensor shapes")
    for name, count in (("input_names", w.shape[1]), ("output_names", v.shape[0])):
        labels = model[name]
        require(isinstance(labels, list) and len(labels) == count and len(set(labels)) == count
                and all(isinstance(s, str) and bool(s.strip()) for s in labels), "Invalid semantic labels")
    return result


def replace_tensors(model: dict[str, Any], tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    result = copy.deepcopy(model)
    for name, value in tensors.items():
        value = np.asarray(value, dtype=np.float64).copy()
        require(np.isfinite(value).all(), "Nonfinite reconstructed tensor")
        value[value == 0] = 0.0
        result[name] = value.tolist()
    return result


def rts_network(model: dict[str, Any]) -> dict[str, Any]:
    require(set(model) == {"name", "version", "provenance", "description", "featureNames",
                          "hiddenNames", *TENSORS, "temperature"}, "Unexpected RTS policy fields")
    require(model["featureNames"] == RTS_FEATURES and len(model["hiddenNames"]) == 38
            and len(set(model["hiddenNames"])) == 38
            and all(isinstance(s, str) and s.strip() for s in model["hiddenNames"]), "Invalid RTS labels")
    require(all(isinstance(model[k], str) and model[k].strip()
                for k in ("name", "version", "provenance", "description")), "Invalid RTS metadata")
    require(type(model["temperature"]) in (int, float) and .05 <= model["temperature"] <= 20,
            "Invalid RTS temperature")
    result = {"schema": "wsp-network/1", "activation": "relu", "dtype": "float64",
              "input_names": model["featureNames"], "output_names": ["candidate_score"],
              "w1": model["w1"], "b1": model["b1"], "w2": [model["w2"]], "b2": [model["b2"]]}
    # Normalize integer-valued JSON numbers to the generic binary64 artifact ABI.
    tensors = arrays(result)
    require(tensors["w1"].shape == (38, 32) and tensors["w2"].shape == (1, 38)
            and all(np.abs(t).max() <= 64 for t in tensors.values()), "Invalid RTS tensor ABI")
    return replace_tensors(result, tensors)


def forward(model: dict[str, Any], inputs: np.ndarray) -> np.ndarray:
    t = arrays(model)
    return np.maximum(inputs @ t["w1"].T + t["b1"], 0.0) @ t["w2"].T + t["b2"]


def apply_literal(base: dict[str, Any], program: dict[str, Any]) -> dict[str, Any]:
    require(set(program) == {"schema", "base_sha256", "intent", "edits"}
            and program["schema"] == "wsp-program/1", "Unknown program schema")
    require(program["base_sha256"] == digest(base), "Program parent hash differs")
    require(isinstance(program["intent"], str) and program["intent"].strip(), "Missing program intent")
    tensors = {name: value.copy() for name, value in arrays(base).items()}
    seen = set()
    for edit in program["edits"]:
        require(set(edit) == {"tensor", "index", "before", "after"}, "Unexpected edit fields")
        name, index = edit["tensor"], tuple(edit["index"])
        require(name in tensors and all(type(i) is int and i >= 0 for i in index), "Invalid coordinate")
        tensor = tensors[name]
        require(len(index) == tensor.ndim and all(i < n for i, n in zip(index, tensor.shape)), "Out of bounds edit")
        require((name, index) not in seen, "Duplicate edit coordinate")
        seen.add((name, index))
        require(q(float(tensor[index])) == q(edit["before"]), "Exact edit prevalue differs")
        require(q(edit["before"]) != q(edit["after"]), "No-op cell is recorded as an edit")
        tensor[index] = float(edit["after"])
    return replace_tensors(base, tensors)


def transform(model: dict[str, Any], gauge: dict[str, Any]) -> dict[str, Any]:
    t = arrays(model)
    n = len(t["b1"])
    permutation, scales = gauge["permutation"], np.asarray(gauge["scales"], dtype=np.float64)
    require(all(type(i) is int for i in permutation) and sorted(permutation) == list(range(n)), "Invalid gauge permutation")
    require(scales.shape == (n,) and np.isfinite(scales).all() and (scales > 0).all(), "Invalid gauge scales")
    return replace_tensors(model, {"w1": t["w1"][permutation] * scales[:, None],
        "b1": t["b1"][permutation] * scales, "w2": t["w2"][:, permutation] / scales[None, :], "b2": t["b2"]})


def transfer(source: dict[str, Any], edited: dict[str, Any], target: dict[str, Any],
             gauge: dict[str, Any] | None = None) -> dict[str, Any]:
    if gauge is not None:
        source, edited = transform(source, gauge), transform(edited, gauge)
    old, new, to = arrays(source), arrays(edited), arrays(target)
    return replace_tensors(target, {name: to[name] + (new[name] - old[name]) for name in TENSORS})


def exact_score(model: dict[str, Any], x: np.ndarray) -> tuple[Fraction, ...]:
    t = arrays(model)
    inputs = [q(float(v)) for v in x]
    hidden = [max(Fraction(0), q(float(b)) + sum((q(float(w)) * v for w, v in zip(row, inputs)), Fraction(0)))
              for row, b in zip(t["w1"], t["b1"])]
    return tuple(q(float(b)) + sum((q(float(w)) * h for w, h in zip(row, hidden)), Fraction(0))
                 for row, b in zip(t["w2"], t["b2"]))


def drift_bounds(base: dict[str, Any], candidate: dict[str, Any], box: dict[str, Any]) -> dict[str, list[str]]:
    old, new = arrays(base), arrays(candidate)
    lower, upper = [q(x) for x in box["lower"]], [q(x) for x in box["upper"]]
    require(len(lower) == len(upper) == old["w1"].shape[1]
            and all(a <= b for a, b in zip(lower, upper)), "Invalid certificate box")
    radius = [max(abs(a), abs(b)) for a, b in zip(lower, upper)]
    activation, perturbation = [], []
    for j in range(len(old["b1"])):
        activation.append(max(Fraction(0), q(float(old["b1"][j])) + sum(
            (max(q(float(w)) * a, q(float(w)) * b) for w, a, b in zip(old["w1"][j], lower, upper)), Fraction(0))))
        perturbation.append(abs(q(float(new["b1"][j])) - q(float(old["b1"][j]))) + sum(
            (abs(q(float(v)) - q(float(w))) * r for v, w, r in zip(new["w1"][j], old["w1"][j], radius)), Fraction(0)))
    output = []
    for o in range(len(old["b2"])):
        bound = abs(q(float(new["b2"][o])) - q(float(old["b2"][o])))
        for j in range(len(activation)):
            bound += (abs(q(float(new["w2"][o, j])) - q(float(old["w2"][o, j]))) * activation[j]
                      + abs(q(float(new["w2"][o, j]))) * perturbation[j])
        output.append(bound)
    return {"hidden_upper": list(map(str, activation)), "hidden_drift": list(map(str, perturbation)),
            "output_bounds": list(map(str, output))}


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def edit_metrics(base: dict[str, Any], candidate: dict[str, Any], x: np.ndarray,
                 threshold: float, shift: float) -> dict[str, float]:
    mask = x[:, 0] > threshold
    require(mask.any() and (~mask).any(), "Empty target or retention partition")
    delta = forward(candidate, x) - forward(base, x)
    return {"target_gain_fraction": float(delta[mask, 0].mean() / shift),
            "target_error_rmse": rmse(delta[mask, 0] - shift),
            "retention_rmse": rmse(delta[~mask]), "maximum_observed_drift": float(np.abs(delta).max())}


def bootstrap(values: list[float], config: dict[str, Any]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    require(data.ndim == 1 and len(data) > 0 and np.isfinite(data).all(), "Invalid bootstrap units")
    n, count, seed = len(data), config["bootstrap_resamples"], config["bootstrap_seed"]
    indices = np.random.default_rng(seed).integers(low=0, high=n, size=(count, n))
    sampled_means = np.mean(data[indices], axis=1)
    return {"n": n, "mean": float(np.mean(data)), "ci95": [float(v) for v in np.quantile(sampled_means, [0.025, 0.975])],
            "positive_units": int(np.count_nonzero(data > 0)), "bootstrap_seed": seed,
            "bootstrap_resamples": count, "method": "percentile bootstrap of paired unit differences"}


def compact_json(text: str) -> str:
    """Remove only insignificant JSON whitespace, retaining JS number spelling."""
    output, quoted, escaped = [], False, False
    for char in text:
        if quoted:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            output.append(char)
        elif not char.isspace():
            output.append(char)
    return "".join(output)


def lexical_field(text: str, name: str) -> str:
    text = compact_json(text)
    decoder, position = json.JSONDecoder(), 1
    require(text.startswith("{"), "Expected JSON object")
    while position < len(text) and text[position] != "}":
        key, end = decoder.raw_decode(text, position)
        require(text[end] == ":", "Malformed JSON object separator")
        start = end + 1
        _, end = decoder.raw_decode(text, start)
        if key == name:
            return text[start:end]
        position = end + 1
    raise ValueError(f"Missing lexical JSON field: {name}")


def fingerprint_model(path: Path) -> str:
    value = compact_json(path.read_text(encoding="utf-8-sig")).encode("utf-16-le")
    hash_value = 0xCBF29CE484222325
    for byte in value:
        hash_value = ((hash_value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return f"fnv1a64_{hash_value:016x}"


class Audit:
    def __init__(self, root: Path, repo: Path):
        self.root, self.repo = root, repo
        self.checks: list[dict[str, Any]] = []
        self.summaries: dict[str, Any] = {}
        self.notes: list[str] = []
        self.freeze: dict[str, Any] = {}
        self.protocol: dict[str, Any] = {}
        self.rts_runs: dict[str, list[dict[str, Any]]] = {}

    def check(self, name: str, passed: bool, mode: str = "exact", details: Any = None,
              required: bool = True) -> bool:
        item = {"id": name, "passed": bool(passed), "comparison": mode, "required": required}
        if details is not None:
            item["details"] = details
        self.checks.append(item)
        return bool(passed)

    def same(self, name: str, actual: Any, expected: Any) -> bool:
        return self.check(name, actual == expected, details=None if actual == expected else
                          {"actual": actual, "expected": expected})

    def close(self, name: str, actual: Any, expected: Any, atol: float = ATOL, rtol: float = RTOL) -> bool:
        try:
            a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            valid = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            error = float(np.max(np.abs(a - b))) if valid and a.size else 0.0
            passed = valid and bool(np.allclose(a, b, atol=atol, rtol=rtol))
            details = {"max_absolute_error": error, "absolute_tolerance": atol, "relative_tolerance": rtol}
        except (TypeError, ValueError):
            passed, details = False, {"reason": "Non-numeric or mismatched values"}
        return self.check(name, passed, "numerical", details)

    def metrics(self, name: str, actual: dict[str, Any], expected: dict[str, Any]) -> None:
        for key, value in expected.items():
            if type(value) is float:
                self.close(f"{name}.{key}", actual.get(key), value)
            elif isinstance(value, list) and value and type(value[0]) is float:
                self.close(f"{name}.{key}", actual.get(key), value)
            else:
                self.same(f"{name}.{key}", actual.get(key), value)

    def load(self, relative: str) -> Any:
        return read_json(contained(self.root, relative))

    def hash(self, relative: str) -> str:
        return file_sha(contained(self.root, relative))

    def receipt_digest(self, name: str, value: dict[str, Any]) -> None:
        self.same(name + ".digest", value.get("sha256"), digest({k: v for k, v in value.items() if k != "sha256"}))

    def manifest(self, directory: str, manifest: dict[str, str]) -> None:
        folder = contained(self.root, directory)
        errors = []
        for relative, expected in manifest.items():
            try:
                path = contained(folder, relative)
                if not path.is_file() or file_sha(path) != expected:
                    errors.append(relative)
            except ValueError:
                errors.append(relative)
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
                  and p.relative_to(folder).as_posix() != "receipt.json"}
        self.check(directory + ".manifest", bool(manifest) and not errors and actual == set(manifest),
                   details={"recorded_files": len(manifest), "mismatched": errors,
                            "unrecorded": sorted(actual - set(manifest)), "missing": sorted(set(manifest) - actual)})

    def source_link(self, name: str, value: dict[str, Any]) -> None:
        expected = {key: sha for key, sha in self.freeze["source_sha256"].items()
                    if key in ("scripts/weight-space-empires.ts", "package-lock.json")
                    or key.startswith("game/src/empires/") and
                    (key.endswith(".ts") or key.startswith("game/src/empires/weights/") and key.endswith(".json"))}
        self.same(name + ".sources", value.get("source_sha256"), expected)
        self.same(name + ".protocol", value.get("protocol_sha256"), self.freeze["protocol_sha256"])

    def stage(self, name: str, function: Callable[[], None]) -> None:
        try:
            function()
            self.check(name + ".completed", True, "structural")
        except Exception as error:
            self.check(name + ".completed", False, "structural", {"error": f"{type(error).__name__}: {error}"})

    def admission(self, name: str, base: dict[str, Any], program: dict[str, Any],
                  candidate: dict[str, Any], receipt: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
        self.receipt_digest(name, receipt)
        self.same(name + ".schema", receipt.get("schema"), "wsp-admission/1")
        self.same(name + ".program", receipt["program_sha256"], digest(program))
        self.same(name + ".contract", receipt["contract"], contract)
        bounds = drift_bounds(base, candidate, contract["box"])
        certificate = receipt["certificate"]
        self.receipt_digest(name + ".certificate", certificate)
        expected = {"schema": "wsp-drift-certificate/1", "arithmetic": "exact-rational-lift-of-binary64",
                    "base_sha256": digest(base), "candidate_sha256": digest(candidate), "domain": contract["box"],
                    **bounds, "theorem": "WeightSpace.score_drift_le", "implementation_verified": False}
        self.check(name + ".exact_rational_certificate", all(certificate.get(k) == v for k, v in expected.items()),
                   "exact rational", {"output_bounds": bounds["output_bounds"]})
        issues = []
        if any(edit["tensor"] not in contract["allowed_tensors"] for edit in program["edits"]):
            issues.append("write-set")
        if any(np.any(np.abs(t) > contract["max_parameter_abs"]) for t in arrays(candidate).values()):
            issues.append("parameter-bound")
        if contract["max_edits"] is not None and len(program["edits"]) > contract["max_edits"]:
            issues.append("edit-count")
        if any(Fraction(bound) > q(limit) for bound, limit in zip(bounds["output_bounds"], contract["max_output_drift"])):
            issues.append("behavioral-drift")
        self.same(name + ".issues", receipt["issues"], issues)
        self.same(name + ".accepted", receipt["accepted"], not issues)
        return {"accepted": not issues, "issues": issues, "bounds": bounds["output_bounds"]}

    def provenance(self, expected_protocol: str | None) -> None:
        self.freeze = self.load("freeze/receipt.json")
        self.protocol = self.load("freeze/protocol.json")
        self.same("freeze.schema", self.freeze["schema"], "wsp-freeze/1")
        self.receipt_digest("freeze", self.freeze)
        self.same("freeze.protocol_bytes", self.hash("freeze/protocol.json"), self.freeze["protocol_sha256"])
        if expected_protocol is not None:
            self.same("freeze.expected_protocol", self.freeze["protocol_sha256"], expected_protocol)
        snapshots = contained(self.root, "freeze/sources")
        names = {p.relative_to(snapshots).as_posix() for p in (snapshots / "rlcd/weight_space").glob("*.py")}
        names |= {p.relative_to(snapshots).as_posix() for p in (snapshots / "game/src/empires").rglob("*.ts")}
        names |= {p.relative_to(snapshots).as_posix() for p in (snapshots / "game/src/empires/weights").rglob("*.json")}
        names |= {"scripts/weight-space-empires.ts", "scripts/run_weight_space_study.ps1", "package-lock.json", "requirements.lock.txt"}
        recorded = self.freeze["source_sha256"]
        self.same("freeze.source_set", sorted(recorded), sorted(names))
        altered, current_differences = [], []
        for name, sha in recorded.items():
            snapshot = contained(snapshots, name)
            if not snapshot.is_file() or file_sha(snapshot) != sha:
                altered.append(name)
            current = contained(self.repo, name)
            if not current.is_file() or file_sha(current) != sha:
                current_differences.append(name)
        self.check("freeze.source_snapshots", not altered, details={"files": len(recorded), "altered": altered})
        self.check("freeze.current_source_observation", not current_differences,
                   details={"differences": current_differences,
                            "scope": "Historical copied sources are authoritative; later workspace changes do not invalidate an old study."},
                   required=False)
        seeds = self.protocol["networks"]["training_seeds"]
        rts = self.protocol["rts"]
        rts_seeds = [s for split in ("development", "validation", "test") for s in rts[f"{split}_seeds"]]
        self.check("protocol.independent_planned_units", len(seeds) % 2 == 0 and len(set(seeds)) == len(seeds)
                   and len(set(rts_seeds)) == len(rts_seeds) and not set(seeds).intersection(rts_seeds),
                   details={"synthetic_pairs": len(seeds) // 2, "source_networks": len(seeds) // 2,
                            "rts_development_worlds": len(rts["development_seeds"]),
                            "rts_validation_worlds": len(rts["validation_seeds"]), "rts_test_worlds": len(rts["test_seeds"])})
        self.summaries["provenance"] = {"freeze_sha256": self.freeze["sha256"],
            "protocol_sha256": self.freeze["protocol_sha256"], "frozen_source_files": len(recorded),
            "current_source_differences": current_differences,
            "timestamp_scope": "Local self-recorded sequence; no external preregistration or tamper-proof timestamp."}

    def logs(self) -> None:
        stages = [("01-freeze", "freeze", "freeze"), ("02-synthetic", "synthetic", "synthetic"),
                  ("03-rts-export", "export", "rts-export"), ("04-rts-prepare", "rts-prepare", "rts-candidates"),
                  ("05-rts-development", "evaluate", "rts-development"),
                  ("06-rts-validation", "evaluate", "rts-validation"), ("07-rts-select", "rts-select", "rts-selection"),
                  ("08-rts-test", "evaluate", "rts-test"), ("09-rts-summarize", "rts-summarize", "rts-summary.json")]
        previous_end = None
        timeline = []
        for stage, action, output in stages:
            command = self.load(f"logs/{stage}.command.json")
            status = self.load(f"logs/{stage}.exit.json")
            arguments = command["arguments"]
            def option(name: str) -> str:
                return arguments[arguments.index(name) + 1]
            start = datetime.fromisoformat(status["started_at"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(status["finished_at"].replace("Z", "+00:00"))
            checks = [command["stage"] == status["stage"] == stage, status["exit_code"] == 0,
                      status["launch_or_capture_error"] is None, action in arguments,
                      option("--out").replace("\\", "/").rstrip("/").split("/")[-1] == output,
                      end >= start, previous_end is None or start >= previous_end]
            if stage in ("05-rts-development", "06-rts-validation", "08-rts-test"):
                split = stage.split("rts-", 1)[1]
                bundle = "rts-selection" if split == "test" else "rts-candidates"
                checks += [option("--split") == split,
                           option("--programs").replace("\\", "/").endswith(f"{bundle}/programs.json")]
            self.check(f"logs.{stage}", all(checks), "structural")
            self.check(f"logs.{stage}.captures", all((self.root / f"logs/{stage}.{stream}.log").is_file()
                       for stream in ("stdout", "stderr")), "structural")
            previous_end = end
            timeline.append({"stage": stage, "started_at": status["started_at"], "finished_at": status["finished_at"]})
        self.summaries["execution_sequence"] = timeline

    def synthetic(self) -> None:
        config, stats = self.protocol["networks"], self.protocol["statistical_policy"]
        results, receipt = self.load("synthetic/results.json"), self.load("synthetic/receipt.json")
        self.manifest("synthetic", receipt["artifacts"])
        self.same("synthetic.schema", results["schema"], "wsp-synthetic-results/1")
        self.same("synthetic.receipt.freeze", receipt["freeze_sha256"], self.freeze["sha256"])
        for name, value in (("results", results), ("started", self.load("synthetic/started.json"))):
            self.same(f"synthetic.{name}.freeze", value["freeze_sha256"], self.freeze["sha256"])
            self.same(f"synthetic.{name}.protocol", value["protocol_sha256"], self.freeze["protocol_sha256"])
        self.same("synthetic.data_hash", results["data_sha256"], self.hash("synthetic/data.npz"))
        with np.load(self.root / "synthetic/data.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        expected_keys = {f"{split}_{kind}" for split in ("train", "compile", "validation", "test") for kind in ("x", "y")}
        self.same("synthetic.data_keys", sorted(data), sorted(expected_keys))
        rows, partitions = [], {}
        for split in ("train", "compile", "validation", "test"):
            x, y = data[f"{split}_x"], data[f"{split}_y"]
            self.check(f"synthetic.data.{split}", x.shape == (config[f"{split}_samples"], config["input_features"])
                       and y.shape == (len(x), config["output_actions"]) and x.dtype == np.float64 and y.dtype == np.float64
                       and np.isfinite(x).all() and np.isfinite(y).all() and np.abs(x).max() <= 1,
                       "structural")
            rows.extend(row.tobytes() for row in x)
            target_count = int(np.sum(x[:, 0] > config["target_threshold"]))
            partitions[split] = {"samples": len(x), "target": target_count, "retention": len(x) - target_count}
            # Check the saved analytic labels, without regenerating the seeded inputs.
            teacher = np.column_stack((
                1.1 * np.sin(1.5 * x[:, 0]) + .55 * x[:, 1] * x[:, 2] - .35 * x[:, 3] + .25 * np.maximum(x[:, 4] + x[:, 5] - .2, 0),
                .8 * np.cos(1.2 * x[:, 1]) - .65 * x[:, 0] * x[:, 4] + .45 * x[:, 2] - .3 * np.maximum(x[:, 3] - x[:, 5], 0),
                .7 * np.sin(x[:, 2] + 1.1 * x[:, 5]) + .4 * x[:, 0] * x[:, 3] - .2 * x[:, 1] + .3 * np.maximum(x[:, 4] - .1, 0)))
            self.close(f"synthetic.data.{split}.teacher", y, teacher)
        self.same("synthetic.data.no_duplicate_inputs", len(set(rows)), len(rows))
        seeds = config["training_seeds"]
        self.same("synthetic.model_files", sorted(p.stem for p in (self.root / "synthetic/models").glob("*.json")), sorted(map(str, seeds)))
        self.same("synthetic.training_files", sorted(p.stem for p in (self.root / "synthetic/training").glob("*.json")), sorted(map(str, seeds)))
        models, training = {}, []
        for seed in seeds:
            model, log = self.load(f"synthetic/models/{seed}.json"), self.load(f"synthetic/training/{seed}.json")
            tensors = arrays(model)
            self.check(f"synthetic.model.{seed}", tensors["w1"].shape == (config["hidden_units"], config["input_features"])
                       and tensors["w2"].shape[0] == config["output_actions"] and log["model_sha256"] == digest(model)
                       and log["seed"] == seed and log["epochs"] == config["training_epochs"]
                       and len(log["loss_trace"]) == config["training_epochs"] and np.isfinite(log["loss_trace"]).all(), "structural")
            independent = {f"{split}_mse": float(np.mean((forward(model, data[f"{split}_x"]) - data[f"{split}_y"]) ** 2))
                           for split in ("train", "validation", "test")}
            self.metrics(f"synthetic.training.{seed}", log, independent)
            training.append({k: v for k, v in log.items() if k != "loss_trace"})
            models[seed] = model
        self.check("synthetic.training_summary_link", results["training_summary"] == training)
        pair_ids = list(zip(seeds[::2], seeds[1::2]))
        self.same("synthetic.pair_files", sorted(p.name for p in (self.root / "synthetic").glob("pair-*")),
                  sorted(f"pair-{a}-{b}" for a, b in pair_ids))
        self.same("synthetic.reported_pair_units", [(p["source_seed"], p["target_seed"]) for p in results["pairs"]], pair_ids)
        contract = {"box": {"lower": [-1.0] * config["input_features"], "upper": [1.0] * config["input_features"]},
                    "max_output_drift": [config["drift_budget"]] * config["output_actions"],
                    "allowed_tensors": ["w2"], "max_parameter_abs": 64.0, "max_edits": None}
        independent_pairs, admissions = [], []
        shift, threshold = config["requested_shift"], config["target_threshold"]
        test_inputs = data["test_x"][data["test_x"][:, 0] > threshold]
        for source_seed, target_seed in pair_ids:
            directory = f"synthetic/pair-{source_seed}-{target_seed}"
            name = f"synthetic.pair.{source_seed}.{target_seed}"
            reported = self.load(directory + "/result.json")
            source, target = models[source_seed], models[target_seed]
            self.check(name + ".parents", reported["source_sha256"] == digest(source) and reported["target_sha256"] == digest(target)
                       and reported["source_seed"] == source_seed and reported["target_seed"] == target_seed)
            matching = [p for p in results["pairs"] if p["source_seed"] == source_seed and p["target_seed"] == target_seed]
            self.check(name + ".aggregate_link", matching == [reported])
            edits, measurements = {}, {}
            for arm in ("target_only", "retention"):
                program = self.load(f"{directory}/{arm}-program.json")
                candidate = apply_literal(source, program)
                self.same(f"{name}.{arm}.literal_model", digest(candidate), digest(self.load(f"{directory}/{arm}-model.json")))
                admission = self.admission(f"{name}.{arm}", source, program, candidate,
                                           self.load(f"{directory}/{arm}-admission.json"), contract)
                self.same(f"{name}.{arm}.admission_link", reported["admission"][arm], admission)
                measurements[arm] = {}
                for split in ("validation", "test"):
                    values = edit_metrics(source, candidate, data[f"{split}_x"], threshold, shift)
                    self.metrics(f"{name}.{arm}.{split}", reported["metrics"][arm][split], values)
                    measurements[arm][split] = values
                admissions.append({"source_seed": source_seed, "target_seed": target_seed, "arm": arm,
                    **admission, "test_target_gain_fraction": measurements[arm]["test"]["target_gain_fraction"]})
                edits[arm] = candidate
            reference = (forward(edits["retention"], test_inputs) - forward(source, test_inputs))[:, 0]
            known_gauge = self.load(directory + "/known-gauge.json")
            self.check(name + ".known_scales", all(scale in config["gauge_scales"] for scale in known_gauge["scales"]))
            known = self.load(directory + "/known-base.json")
            self.same(name + ".known_base", digest(known), digest(transform(source, known_gauge)))
            transported = apply_literal(known, self.load(directory + "/known-transport-program.json"))
            raw_known = apply_literal(known, self.load(directory + "/known-raw-program.json"))
            self.same(name + ".known_transport", digest(transported), digest(transform(edits["retention"], known_gauge)))
            self.same(name + ".known_raw_transfer", digest(raw_known), digest(transfer(source, edits["retention"], known)))
            inferred = transfer(source, edits["retention"], known, reported["controls"]["inferred_gauge"])
            point_checks = all(exact_score(source, x) == exact_score(known, x)
                               and exact_score(edits["retention"], x) == exact_score(transported, x) for x in test_inputs[:5])
            control_values = {"exact_rational_point_checks": point_checks, "exact_point_count": min(5, len(test_inputs))}
            for key, candidate in (("known_transport_error", transported), ("inferred_known_gauge_error", inferred), ("raw_permuted_error", raw_known)):
                control_values[key] = rmse((forward(candidate, test_inputs) - forward(known, test_inputs))[:, 0] - reference) / shift
            self.metrics(name + ".controls", reported["controls"], control_values)
            self.check(name + ".exact_control_points", point_checks, "exact rational")
            gauge = self.load(directory + "/inferred-gauge.json")
            transfer_values = {}
            for arm in ("raw", "aligned"):
                program = self.load(f"{directory}/transfer-{arm}-program.json")
                candidate = apply_literal(target, program)
                self.same(f"{name}.transfer_{arm}.literal_model", digest(candidate), digest(self.load(f"{directory}/transfer-{arm}-model.json")))
                self.same(f"{name}.transfer_{arm}.transformation", digest(candidate),
                          digest(transfer(source, edits["retention"], target, gauge if arm == "aligned" else None)))
                admission = self.admission(f"{name}.transfer_{arm}", target, program, candidate,
                                           self.load(f"{directory}/transfer-{arm}-admission.json"), contract)
                error = rmse((forward(candidate, test_inputs) - forward(target, test_inputs))[:, 0] - reference) / shift
                self.metrics(f"{name}.transfer_{arm}.report", reported["transfer"][arm],
                    {"normalized_target_delta_rmse": error, "admitted": admission["accepted"],
                     "admission_issues": admission["issues"], "bounds": admission["bounds"]})
                transfer_values[arm] = error
                admissions.append({"source_seed": source_seed, "target_seed": target_seed, "arm": f"transfer_{arm}", **admission})
            endpoints = {"H1_raw_minus_aligned": transfer_values["raw"] - transfer_values["aligned"],
                "H2_retention_rmse_reduction": measurements["target_only"]["test"]["retention_rmse"] - measurements["retention"]["test"]["retention_rmse"],
                "H2_target_gain_fraction": measurements["retention"]["test"]["target_gain_fraction"]}
            self.metrics(name + ".endpoints", reported, endpoints)
            independent_pairs.append({"source_seed": source_seed, "target_seed": target_seed, **endpoints,
                "transfer": transfer_values, "metrics": measurements, "controls": control_values})
        controls_pass = all(p["controls"]["exact_rational_point_checks"] and p["controls"]["known_transport_error"] < 1e-12
                            and p["controls"]["inferred_known_gauge_error"] < 1e-10 for p in independent_pairs)
        self.metrics("synthetic.completeness", results, {"planned_networks": len(seeds), "completed_networks": len(models),
                     "planned_pairs": len(pair_ids), "completed_pairs": len(independent_pairs), "failures": [], "complete": True,
                     "controls_pass": controls_pass})
        self.check("synthetic.controls_succeeded", controls_pass, "numerical")
        h1 = bootstrap([p["H1_raw_minus_aligned"] for p in independent_pairs], stats)
        h2 = bootstrap([p["H2_retention_rmse_reduction"] for p in independent_pairs], stats)
        gain = bootstrap([p["H2_target_gain_fraction"] for p in independent_pairs], stats)
        h1_status = "inconclusive" if not controls_pass else "supported" if h1["ci95"][0] > 0 else "unsupported"
        h2_status = "inconclusive" if not controls_pass else "supported" if h2["ci95"][0] > 0 and gain["mean"] >= .5 else "unsupported"
        self.metrics("synthetic.H1.effect", results["H1"]["effect"], h1)
        self.metrics("synthetic.H2.effect", results["H2"]["effect"], h2)
        self.metrics("synthetic.H2.gain", results["H2"]["target_gain_fraction"], gain)
        self.same("synthetic.H1.status", results["H1"]["status"], h1_status)
        self.same("synthetic.H2.status", results["H2"]["status"], h2_status)
        admission_summary = {}
        for arm in ("target_only", "retention", "transfer_raw", "transfer_aligned"):
            subset = [row for row in admissions if row["arm"] == arm]
            entry = {"proposals": len(subset), "admitted": sum(row["accepted"] for row in subset),
                     "rejected": sum(not row["accepted"] for row in subset),
                     "rejection_reasons": dict(Counter(reason for row in subset for reason in row["issues"]))}
            if arm in ("target_only", "retention"):
                useful = sum(row["accepted"] and row["test_target_gain_fraction"] >= .5 for row in subset)
                entry.update(useful_and_admitted=useful, unconditional_useful_and_admitted_fraction=useful / len(pair_ids))
            admission_summary[arm] = entry
        self.same("synthetic.admission_count", len(admissions), 4 * len(pair_ids))
        self.summaries["synthetic"] = {"data_partitions": partitions, "network_count": len(models), "pair_count": len(independent_pairs),
            "H1": {"effect": h1, "status": h1_status}, "H2": {"effect": h2, "target_gain_fraction": gain, "status": h2_status},
            "controls_pass": controls_pass, "admissions": admission_summary,
            "useful_and_admitted_scope": "Descriptive: per-program test gain >=0.5 of requested shift and exact admission; denominator includes every planned source program. H2 predeclared a mean gain rule, not this per-program endpoint.",
            "negative_control_scope": "Raw-permuted error is reported per unit; its magnitude is not part of the frozen aggregate controls_pass rule.",
            "inference_scope": "Disjoint initialization units, conditional on one saved shared data partition; exploratory unadjusted intervals.",
            "units": independent_pairs, "admission_records": admissions}

    def rts_export(self) -> None:
        receipt, fixtures = self.load("rts-export/receipt.json"), self.load("rts-export/fixtures.json")
        self.manifest("rts-export", receipt["artifacts"])
        self.source_link("rts.export", receipt)
        self.same("rts.export.protocol_bytes", self.hash("rts-export/protocol.json"), self.freeze["protocol_sha256"])
        self.rts_template = self.load("rts-export/model.json")
        self.rts_base = rts_network(self.rts_template)
        authored = self.load("freeze/sources/game/src/empires/weights/commander-v1.json")
        self.same("rts.export.authored_policy", self.rts_template, authored)
        self.metrics("rts.export.receipt", receipt, {"schema_version": 1, "operation": "export",
            "fixture_count": len(fixtures["fixtures"]), "exported_splits": ["development", "validation"],
            "test_fixtures_exported": False})
        self.metrics("rts.export.fixture_header", fixtures, {"schema_version": 1,
            "protocol_sha256": self.freeze["protocol_sha256"], "model_sha256": self.hash("rts-export/model.json"),
            "model_fingerprint": fingerprint_model(self.root / "rts-export/model.json"),
            "feature_names": RTS_FEATURES})
        self.same("rts.export.layout", fixtures["layout"], {"order": list(TENSORS), "flattening": "row-major",
            "parameters": 1293, "tensors": {"w1": {"shape": [38, 32], "offset": 0, "length": 1216},
            "b1": {"shape": [38], "offset": 1216, "length": 38},
            "w2": {"shape": [38], "offset": 1254, "length": 38},
            "b2": {"shape": [], "offset": 1292, "length": 1}},
            "temperature": "separate; unused by rankPlayPolicy"})

        # This source inspection corroborates the denominator; it is not a JS
        # semantic proof and cannot recover the unsaved initial checkpoint body.
        source_root = self.root / "freeze/sources/game/src/empires"
        types_source = (source_root / "types.ts").read_text(encoding="utf-8")
        mission_source = (source_root / "mission.ts").read_text(encoding="utf-8")
        world_source = (source_root / "simulation.ts").read_text(encoding="utf-8")
        match = re.search(r"towncenter:\s*\{[^\n]*?\bhp:\s*([0-9.]+)", types_source)
        require(match is not None, "Missing initial town health source definition")
        self.initial_town_hp = float(match.group(1))
        initialization = world_source.split("private createFirstVictory()", 1)[1].split("\n  entity(", 1)[0]
        source_guard = ("return new World(seed, { scenario: 'first-victory' });" in mission_source
            and "if (options.scenario === 'first-victory')" in world_source
            and "this.createFirstVictory();" in world_source
            and "for (const team of [0, 1] as const)" in initialization
            and initialization.count("this.spawn(team, 'towncenter', x, z);") == 1
            and "construction = 1" in world_source
            and "hp: def.hp * (0.1 + construction * 0.9), maxHp: def.hp" in world_source)
        self.check("rts.initial_town_hp_source_contract", source_guard and self.initial_town_hp == 1800,
            "source inspection", {"initial_town_max_hp_per_team": self.initial_town_hp,
            "scope": "Inspected archived mission initialization spawns one completed town center per team; no checkpoint regeneration or formal TS proof."})

        snapshots, snapshot_counts, snapshot_errors = {}, Counter(), []
        town_observations = []
        with (self.root / "rts-export/snapshots.jsonl").open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line, object_pairs_hook=json_pairs, parse_constant=invalid_constant)
                checkpoint = value["checkpoint"]
                actual_hash = hashlib.sha256(lexical_field(line, "checkpoint").encode("utf-8")).hexdigest()
                if actual_hash != value["state_id"]:
                    snapshot_errors.append(line_number)
                snapshot_counts[value["state_id"]] += 1
                snapshots[value["state_id"]] = checkpoint
                town_observations.extend((entity["owner"], entity["maxHp"])
                    for entity in checkpoint["entities"] if entity["kind"] == "towncenter")
        self.check("rts.export.snapshot_content_hashes", not snapshot_errors and bool(snapshots),
                   details={"records": sum(snapshot_counts.values()), "invalid_lines": snapshot_errors})
        self.check("rts.export.snapshot_town_hp", bool(town_observations)
            and {team for team, _ in town_observations} == {0, 1}
            and all(hp == self.initial_town_hp for _, hp in town_observations),
            details={"observations": len(town_observations)})
        self.same("rts.export.fixture_snapshot_multiset", dict(snapshot_counts),
                  dict(Counter(f["state_id"] for f in fixtures["fixtures"])))
        expected_units = {(split, seed) for split in ("development", "validation")
                          for seed in self.protocol["rts"][f"{split}_seeds"]}
        seen_ids, seen_units, ticks_by_unit = [], set(), {}
        sequential_error, numpy_error, examples = 0.0, 0.0, 0
        sequential_mismatches, numpy_mismatches = [], []
        t = arrays(self.rts_base)
        for fixture in fixtures["fixtures"]:
            split, seed, tick, team = (fixture[k] for k in ("split", "seed", "tick", "team"))
            identifier = f"{split}/seed-{seed}/tick-{tick}/team-{team}"
            name = "rts.fixture." + identifier.replace("/", ".")
            checkpoint = snapshots[fixture["state_id"]]
            self.check(name + ".identity", fixture["id"] == identifier and (split, seed) in expected_units
                and team in (0, 1) and type(tick) is int and tick > 0
                and tick % int(self.protocol["rts"]["fixture_interval_seconds"] / .1) == 0
                and tick <= int(self.protocol["rts"]["fixture_seconds"] / .1)
                and checkpoint["seed"] == seed and checkpoint["tick"] == tick
                and checkpoint["time"] == fixture["time"] and checkpoint["options"]["scenario"] == "first-victory",
                "structural")
            self.close(name + ".time", fixture["time"], tick * .1)
            seen_ids.append(identifier)
            seen_units.add((split, seed))
            ticks_by_unit.setdefault((split, seed), []).append((tick, team))
            entries = fixture["candidates"]
            if not entries:
                continue
            inputs = np.asarray([entry["input"] for entry in entries], dtype=np.float64)
            self.check(name + ".feature_domain", inputs.shape == (len(entries), 32)
                       and np.isfinite(inputs).all() and (inputs >= 0).all() and (inputs <= 1).all(), "structural")
            ids = [entry["id"] for entry in entries]
            self.check(name + ".candidate_ids", len(set(ids)) == len(ids), "structural")
            sequential = []
            for x, entry in zip(inputs, entries):
                score = float(t["b2"][0])
                for hidden in range(38):
                    activation = float(t["b1"][hidden])
                    for feature in range(32):
                        activation += float(t["w1"][hidden, feature]) * float(x[feature])
                    score += float(t["w2"][0, hidden]) * max(0.0, activation)
                sequential.append(score)
                sequential_error = max(sequential_error, abs(score - entry["score"]))
            vector = forward(self.rts_base, inputs)[:, 0]
            numpy_error = max(numpy_error, float(np.max(np.abs(vector - [e["score"] for e in entries]))))
            order = sorted(range(len(entries)), key=lambda i: (-sequential[i], ids[i]))
            self.same(name + ".ranked_sequence", order, list(range(len(entries))))
            if order[0] != 0:
                sequential_mismatches.append(identifier)
            if min(range(len(entries)), key=lambda i: (-vector[i], ids[i])) != 0:
                numpy_mismatches.append(identifier)
            examples += len(entries)
        self.same("rts.export.planned_fixture_units", sorted(seen_units), sorted(expected_units))
        self.same("rts.export.unique_fixture_ids", len(set(seen_ids)), len(seen_ids))
        interval = int(self.protocol["rts"]["fixture_interval_seconds"] / .1)
        self.check("rts.export.fixture_tick_pairs", all(sorted(values) ==
            [(tick, team) for tick in range(interval, max(tick for tick, _ in values) + 1, interval) for team in (0, 1)]
            for values in ticks_by_unit.values()), "structural")
        agreement = {"fixture_count": len(fixtures["fixtures"]), "scored_examples": examples,
            "sequential_max_absolute_error": sequential_error, "numpy_max_absolute_error": numpy_error,
            "sequential_argmax_mismatches": sequential_mismatches, "numpy_argmax_mismatches": numpy_mismatches,
            "passed": examples > 0 and sequential_error == 0 and not sequential_mismatches and numpy_error < 1e-10,
            "scope": "pass requires exact sequential scores/choices and close NumPy scores; NumPy argmax mismatches are reported separately; no IEEE proof"}
        self.metrics("rts.export.fixture_agreement", self.load("rts-candidates/fixture-agreement.json"), agreement)
        self.check("rts.export.exact_sequential_scores", sequential_error == 0, "exact binary64 operations")
        self.check("rts.export.fixture_agreement_passed", agreement["passed"], "numerical")
        self.fixture_agreement = agreement
        self.summaries["rts_export"] = {**agreement, "snapshot_records": sum(snapshot_counts.values()),
            "town_max_hp_denominator": self.initial_town_hp,
            "feature_scope": "Re-evaluates saved vectors and recorded ranks; no simulation, legality or state-to-feature extraction replay."}

    def rts_candidates(self) -> None:
        directory = "rts-candidates"
        receipt = self.load(directory + "/receipt.json")
        self.manifest(directory, receipt["artifacts"])
        self.receipt_digest("rts.candidates", receipt)
        self.metrics("rts.candidates.receipt", receipt, {"schema": "wsp-rts-candidates/1",
            "freeze_sha256": self.freeze["sha256"], "protocol_sha256": self.freeze["protocol_sha256"],
            "base_sha256": digest(self.rts_base), "programs_sha256": self.hash(directory + "/programs.json")})
        self.metrics("rts.candidates.fixture_link", receipt["fixture_agreement"], self.fixture_agreement)
        self.same("rts.candidates.base_network", self.load(directory + "/base-network.json"), self.rts_base)
        self.rts_bundle = self.load(directory + "/programs.json")
        deltas = self.protocol["rts"]["candidate_recruitment_deltas"]
        ids = [f"recruit_{str(float(delta)).replace('.', '_')}" for delta in deltas]
        self.same("rts.candidates.bundle_fields", sorted(self.rts_bundle), ["candidates"])
        self.same("rts.candidates.predeclared_set", [p["id"] for p in self.rts_bundle["candidates"]], ids)
        hidden = self.rts_template["hiddenNames"].index("army_train_bias")
        feature = RTS_FEATURES.index("train_army")
        t = arrays(self.rts_base)
        selector = np.zeros(32)
        selector[feature] = 1
        self.check("rts.candidates.exact_recruitment_selector", np.array_equal(t["w1"][hidden], selector)
                   and t["b1"][hidden] == 0, "exact rational")
        common = {"allowed_tensors": ["w2"], "max_parameter_abs": 64.0, "max_edits": 1}
        global_contract = {**common, "box": {"lower": [0.0] * 32, "upper": [1.0] * 32},
                           "max_output_drift": [self.protocol["rts"]["maximum_certified_score_drift"]]}
        protected_upper = [1.0] * 32
        protected_upper[feature] = 0.0
        protected_contract = {**common, "box": {"lower": [0.0] * 32, "upper": protected_upper},
                              "max_output_drift": [self.protocol["rts"]["protected_region_budget"]]}
        self.same("rts.candidates.global_contract", self.load(directory + "/global-contract.json"), global_contract)
        self.same("rts.candidates.protected_contract", self.load(directory + "/protected-contract.json"), protected_contract)
        exact = {"schema": "wsp-rational-policy/1", "base_sha256": digest(self.rts_base),
            "source_rts_file_sha256": self.hash("rts-export/model.json"),
            "arithmetic": "Fraction.from_float: exact binary64 rational value",
            "w1": [[str(q(float(v))) for v in row] for row in t["w1"]],
            "b1": [str(q(float(v))) for v in t["b1"]], "w2": [str(q(float(v))) for v in t["w2"][0]],
            "b2": str(q(float(t["b2"][0]))), "hidden_index": hidden, "feature_index": feature, "edits": []}
        candidate_evidence = []
        for identifier, delta, supplied in zip(ids, deltas, self.rts_bundle["candidates"]):
            name = "rts.candidate." + identifier
            program = self.load(f"{directory}/{identifier}-program.json")
            candidate = apply_literal(self.rts_base, program)
            after = float(t["w2"][0, hidden]) + delta
            self.same(name + ".literal_single_coordinate", program["edits"], [{"tensor": "w2", "index": [0, hidden],
                "before": float(t["w2"][0, hidden]), "after": after}])
            self.same(name + ".adapter_tensors", rts_network(supplied["model"]), candidate)
            expected_policy = copy.deepcopy(self.rts_template)
            expected_policy.update(w1=candidate["w1"], b1=candidate["b1"], w2=candidate["w2"][0], b2=candidate["b2"][0],
                                   provenance="weight-space compiler; experimental", description=program["intent"])
            self.same(name + ".adapter_metadata", supplied["model"], expected_policy)
            self.same(name + ".program_link", supplied["source"],
                      f"Literal readout edit; program {digest(program)}; isolated H3 candidate")
            saved = self.load(f"{directory}/{identifier}-admissions.json")
            general = self.admission(name + ".global", self.rts_base, program, candidate, saved["global"], global_contract)
            protected = self.admission(name + ".protected", self.rts_base, program, candidate, saved["protected"], protected_contract)
            self.check(name + ".both_admitted", general["accepted"] and protected["accepted"], "exact rational")
            candidate_evidence.append({"id": identifier, "requested_delta": delta, "network_sha256": digest(candidate),
                "program_sha256": digest(program), "global_bound": general["bounds"][0], "protected_bound": protected["bounds"][0]})
            exact["edits"].append({"id": identifier, "requested_delta": delta,
                "actual_delta": str(q(after) - q(float(t["w2"][0, hidden]))), "after_w2": str(q(after)),
                "candidate_sha256": digest(candidate)})
        self.same("rts.candidates.evidence", receipt["candidates"], candidate_evidence)
        self.same("rts.candidates.exact_policy", self.load(directory + "/exact-policy.json"), exact)
        self.summaries["rts_candidates"] = {"proposals": len(candidate_evidence), "candidates": candidate_evidence,
            "scope": "Exact rational score-drift admission on declared feature boxes; no claim that admission implies better strategy or utility."}

    def rts_metrics(self, name: str, value: dict[str, Any]) -> None:
        counts = ("recruitsCompleted", "buildingsCompleted", "units", "army", "villagers", "buildings",
                  "population", "capacity", "queued", "kills", "age")
        resources_ok = all(set(value[k]) == set(RESOURCES)
            and all(type(v) in (int, float) and math.isfinite(v) and v >= -ATOL for v in value[k].values())
            for k in ("gathered", "stockpile"))
        production_ok = all(isinstance(value[k], dict)
            and all(type(v) is int and v >= 0 for v in value[k].values()) for k in ("trained", "constructed"))
        self.check(name + ".structure", resources_ok and production_ok
            and all(type(value[k]) is int and value[k] >= 0 for k in counts)
            and type(value["townHealth"]) in (int, float) and math.isfinite(value["townHealth"])
            and value["townHealth"] >= 0 and value["status"] in ("playing", "won", "lost", "draw"), "structural")
        self.check(name + ".internal_counts",
            value["recruitsCompleted"] == sum(count for kind, count in value["trained"].items() if kind != "villager")
            and value["buildingsCompleted"] == sum(value["constructed"].values())
            and value["units"] == value["army"] + value["villagers"]
            and value["population"] == value["units"] + value["queued"], "exact integer")

    def rts_branch(self, name: str, raw: dict[str, Any], branch: dict[str, Any], directory: str) -> dict[str, Any]:
        metrics, other = branch["metrics"], branch["otherMetrics"]
        self.rts_metrics(name + ".own", metrics)
        self.rts_metrics(name + ".opponent", other)
        self.same(name + ".shared_status", metrics["status"], other["status"])
        initial, initial_other = raw["initialMetrics"], raw["initialOtherMetrics"]
        elapsed, orders = branch["elapsedSeconds"], branch["orders"]
        schedule = [(order["tick"], order["team"]) for order in orders]
        self.check(name + ".duration", math.isfinite(elapsed) and 0 < elapsed <= raw["requestedSeconds"] + 1e-8
            and (metrics["status"] != "playing" or abs(elapsed - raw["requestedSeconds"]) < 1e-8), "numerical")
        self.check(name + ".order_schedule", schedule == sorted(schedule) and len(set(schedule)) == len(schedule)
            and all(type(order["tick"]) is int and order["tick"] > raw["startTick"] and order["tick"] % 20 == 0
                and order["team"] in (0, 1) and order["executed"] is True and order["command"] is not None
                and isinstance(order["id"], str) and math.isfinite(order["score"])
                and 0 < order["elapsedSeconds"] <= elapsed
                and abs(order["time"] - order["tick"] * .1) < 1e-8
                and order["elapsedSeconds"] == order["time"] - raw["startTime"] for order in orders), "structural")
        own_orders = sum(order["team"] == raw["team"] for order in orders)
        other_orders = len(orders) - own_orders
        self.check(name + ".decision_counts", branch["executedOrders"] == own_orders
            and branch["otherExecutedOrders"] == other_orders
            and all(type(branch[k]) is int and branch[k] >= 0 for k in ("decisions", "waitDecisions"))
            and own_orders + branch["waitDecisions"] <= branch["decisions"]
            and branch["decisions"] <= math.floor((elapsed + 1e-8) / 2), "exact integer")
        self.check(name + ".final_checkpoint_digest_format", bool(re.fullmatch(r"[0-9a-f]{64}", branch["final_checkpoint_sha256"])), "structural")
        if branch["checkpoint_file"] is not None:
            checkpoint_path = contained(contained(self.root, directory), branch["checkpoint_file"])
            checkpoint_json = checkpoint_path.read_text(encoding="utf-8-sig")
            read_json(checkpoint_path)
            self.same(name + ".saved_checkpoint_body", hashlib.sha256(compact_json(checkpoint_json).encode("utf-8")).hexdigest(),
                      branch["final_checkpoint_sha256"])
        own_loss = initial["townHealth"] - metrics["townHealth"]
        opponent_loss = initial_other["townHealth"] - other["townHealth"]
        recruits = metrics["recruitsCompleted"] - initial["recruitsCompleted"]
        winner = 0 if metrics["status"] == "won" else 1 if metrics["status"] == "lost" else None
        victory, defeat = winner == raw["team"], winner == 1 - raw["team"]
        return {"outcome": "win" if victory else "loss" if defeat else "draw" if metrics["status"] == "draw" else "unfinished",
            "victory": victory, "defeat": defeat, "status": metrics["status"], "elapsed_seconds": elapsed,
            "utility": opponent_loss / self.initial_town_hp - own_loss / self.initial_town_hp + .02 * recruits,
            "recruits_completed": recruits, "trained_units": metrics["trained"],
            "initial_town_hp": initial["townHealth"], "town_max_hp": self.initial_town_hp, "final_town_hp": metrics["townHealth"],
            "initial_opponent_town_hp": initial_other["townHealth"], "opponent_town_max_hp": self.initial_town_hp,
            "final_opponent_town_hp": other["townHealth"], "own_town_hp_lost": own_loss, "opponent_town_hp_lost": opponent_loss,
            "gathered": {kind: metrics["gathered"][kind] - initial["gathered"][kind] for kind in RESOURCES},
            "stockpile": metrics["stockpile"], "kills": metrics["kills"] - initial["kills"],
            "decisions": branch["decisions"], "executed_orders": own_orders, "wait_decisions": branch["waitDecisions"],
            "metrics": metrics, "other_metrics": other}

    def rts_evaluation(self, split: str) -> None:
        directory = "rts-" + split
        results, receipt = self.load(directory + "/results.json"), self.load(directory + "/receipt.json")
        self.manifest(directory, receipt["artifacts"])
        for label, item in (("results", results), ("receipt", receipt)):
            self.source_link(f"rts.{split}.{label}", item)
            self.metrics(f"rts.{split}.{label}.identity", item, {"schema_version": 1, "operation": "evaluate"})
        self.same(f"rts.{split}.protocol_bytes", self.hash(directory + "/protocol.json"), self.freeze["protocol_sha256"])
        bundle_source = "rts-selection" if split == "test" else "rts-candidates"
        bundle_hash = self.hash(bundle_source + "/programs.json")
        supplied = self.load(bundle_source + "/programs.json")["candidates"]
        self.same(f"rts.{split}.captured_bundle", self.hash(directory + "/programs.json"), bundle_hash)
        self.same(f"rts.{split}.receipt_bundle", receipt["programs_sha256"], bundle_hash)
        self.same(f"rts.{split}.results_bundle", results["programs_sha256"], bundle_hash)
        ids = [*CONTROLS, *[p["id"] for p in supplied]]
        seeds = self.protocol["rts"][f"{split}_seeds"]
        expected_units = {(seed, identifier) for seed in seeds for identifier in ids}
        actual_units = [(run["seed"], run["candidate_id"]) for run in results["runs"]]
        self.check(f"rts.{split}.planned_units", len(actual_units) == len(set(actual_units))
                   and set(actual_units) == expected_units and results["split"] == split, "structural",
                   {"planned": len(expected_units), "recorded": len(actual_units)})
        self.same(f"rts.{split}.program_order", [p["id"] for p in results["programs"]], ids)
        self.same(f"rts.{split}.saved_model_set", sorted(p.stem for p in (self.root / directory / "models").glob("*.json")), sorted(ids))
        shifted = copy.deepcopy(self.rts_template)
        shifted["b2"] += 1 if shifted["b2"] <= 63 else -1
        programs = [{"id": CONTROLS[0], "model": self.rts_template,
            "source": "Exact authored commander-v1 baseline; no parameter edits."},
            {"id": CONTROLS[1], "model": shifted,
            "source": "Only shared output bias shifted by one; behavioral negative control."}, *supplied]
        metadata = {}
        for program, reported in zip(programs, results["programs"]):
            identifier = program["id"]
            model_path = f"{directory}/models/{identifier}.json"
            actual_model = self.load(model_path)
            rts_network(actual_model)
            self.same(f"rts.{split}.{identifier}.policy", actual_model, program["model"])
            value = {"id": identifier, "source": program["source"], "control": identifier in CONTROLS,
                     "model_sha256": self.hash(model_path), "model_fingerprint": fingerprint_model(self.root / model_path)}
            self.same(f"rts.{split}.{identifier}.metadata", reported, value)
            metadata[identifier] = value
        methodology = {"scenario": "first-victory", "candidate_team": 0,
            "opponent": "authored commander-v1 under the mission rival mask", "paired_same_checkpoint": True,
            "horizon_seconds": self.protocol["rts"]["horizon_seconds"], "dt_seconds": .1,
            "policy_interval_seconds": 2.0, "policy_order": [0, 1], "rival_recruitment_begins_seconds": 270,
            "rival_raid_eligibility_seconds": 330}
        self.metrics(f"rts.{split}.methodology", results["methodology"], methodology)
        independent, saved_checkpoint_bodies = [], 0
        for run in results["runs"]:
            seed, identifier = run["seed"], run["candidate_id"]
            name = f"rts.{split}.{seed}.{identifier}"
            relative = f"runs/{split}/seed-{seed}-{identifier}.json"
            self.same(name + ".raw_file", run["raw_result_file"], relative)
            self.same(name + ".raw_sha256", run["raw_result_sha256"], self.hash(directory + "/" + relative))
            raw = self.load(directory + "/" + relative)
            self.metrics(name + ".raw_identity", raw, {"schema_version": 1,
                "protocol_sha256": self.freeze["protocol_sha256"], "programs_sha256": bundle_hash,
                "split": split, "seed": seed, "program": metadata[identifier], "team": 0,
                "otherControl": "policy", "requestedSeconds": self.protocol["rts"]["horizon_seconds"],
                "startTick": 0, "startTime": 0})
            self.metrics(name + ".summary_identity", run, {"candidate_id": identifier, "split": split,
                "seed": seed, "team": 0, **{k: v for k, v in metadata[identifier].items() if k != "id"}})
            for label in ("initialMetrics", "initialOtherMetrics"):
                self.rts_metrics(name + "." + label, raw[label])
                self.metrics(name + "." + label + ".initial_state", raw[label],
                             {"townHealth": self.initial_town_hp, "status": "playing", "recruitsCompleted": 0})
            branches = {}
            for label in ("baseline", "candidate"):
                branches[label] = self.rts_branch(name + "." + label, raw, raw[label], directory)
                self.same(name + "." + label + ".summary_fields", sorted(run[label]), sorted(branches[label]))
                self.metrics(name + "." + label + ".summary", run[label], branches[label])
                self.same(name + "." + label + ".checkpoint_link", run[label + "_checkpoint_sha256"], raw[label]["final_checkpoint_sha256"])
                saved_checkpoint_bodies += raw[label]["checkpoint_file"] is not None
            baseline, candidate = branches["baseline"], branches["candidate"]
            delta_keys = ("utility", "recruits_completed", "own_town_hp_lost", "opponent_town_hp_lost", "kills", "victory", "defeat")
            deltas = {key: candidate[key] - baseline[key] for key in delta_keys}
            deltas.update({key: {kind: candidate[key][kind] - baseline[key][kind] for kind in RESOURCES}
                           for key in ("gathered", "stockpile")})
            self.same(name + ".delta_fields", sorted(run["deltas"]), sorted(deltas))
            self.metrics(name + ".deltas", run["deltas"], deltas)
            behavior_equal = raw["firstChangedChoice"] is None and raw["baseline"]["final_checkpoint_sha256"] == raw["candidate"]["final_checkpoint_sha256"]
            self.same(name + ".behavior_equality", run["behavior_equal"], behavior_equal)
            self.same(name + ".first_changed_choice_link", run["first_changed_choice"], raw["firstChangedChoice"])
            self.check(name + ".initial_checkpoint_digest_format", bool(re.fullmatch(r"[0-9a-f]{64}", run["initial_checkpoint_sha256"])), "structural")
            if behavior_equal:
                without_scores = lambda branch: [{k: v for k, v in order.items() if k != "score"} for order in branch["orders"]]
                self.check(name + ".equal_behavior_records", baseline == candidate
                           and without_scores(raw["baseline"]) == without_scores(raw["candidate"]), "exact")
            changed = raw["firstChangedChoice"]
            if changed is not None:
                left, right = changed["baseline"], changed["candidate"]
                different = (left is None) != (right is None) or (left is not None and right is not None
                            and (left["id"] != right["id"] or left["command"] != right["command"]))
                self.check(name + ".changed_choice_structure", different and type(changed["tick"]) is int
                    and changed["tick"] % 20 == 0 and 0 < changed["elapsedSeconds"] <= max(baseline["elapsed_seconds"], candidate["elapsed_seconds"])
                    and changed["time"] - raw["startTime"] == changed["elapsedSeconds"], "structural")
                prefix = lambda branch: [{k: v for k, v in order.items() if k != "score"}
                    for order in branch["orders"] if order["tick"] < changed["tick"]]
                self.same(name + ".unchanged_order_prefix", prefix(raw["baseline"]), prefix(raw["candidate"]))
            self.check(name + ".wall_time", type(run["wall_seconds"]) in (int, float)
                       and math.isfinite(run["wall_seconds"]) and run["wall_seconds"] >= 0, "structural")
            independent.append({"candidate_id": identifier, "seed": seed, "control": identifier in CONTROLS,
                "baseline": baseline, "candidate": candidate, "deltas": deltas, "behavior_equal": behavior_equal,
                "initial_checkpoint_sha256": run["initial_checkpoint_sha256"],
                "baseline_checkpoint_sha256": raw["baseline"]["final_checkpoint_sha256"],
                "baseline_record_sha256": digest({k: v for k, v in raw["baseline"].items() if k != "checkpoint_file"}),
                "initial_metrics_sha256": digest([raw["initialMetrics"], raw["initialOtherMetrics"], raw["startTick"], raw["startTime"]])})
        for seed in seeds:
            selected = [run for run in independent if run["seed"] == seed]
            self.check(f"rts.{split}.{seed}.paired_baseline_identity", len(selected) == len(ids)
                and all(len({run[key] for run in selected}) == 1 for key in ("initial_checkpoint_sha256",
                    "baseline_checkpoint_sha256", "baseline_record_sha256", "initial_metrics_sha256")), "exact")
        controls = [run for run in independent if run["control"]]
        controls_pass = len(controls) == len(seeds) * len(CONTROLS) and all(run["behavior_equal"] and run["deltas"]["utility"] == 0 for run in controls)
        for label, item in (("receipt", receipt), ("results", results)):
            self.metrics(f"rts.{split}.{label}.completion", item, {"planned_runs": len(expected_units),
                "completed_runs": len(independent), "controls_pass": controls_pass,
                "failures": 0 if label == "receipt" else []})
        self.same(f"rts.{split}.receipt.complete", receipt["complete"], True)
        self.check(f"rts.{split}.controls_succeeded", controls_pass, "exact")
        aggregates = []
        self.same(f"rts.{split}.aggregate_ids", [row["candidate_id"] for row in results["summary"]], ids)
        for identifier, reported in zip(ids, results["summary"]):
            selected = [run for run in independent if run["candidate_id"] == identifier]
            mean = lambda get: float(np.mean([get(run) for run in selected]))
            row = {"split": split, "candidate_id": identifier, "control": identifier in CONTROLS,
                "completed_seeds": len(selected), "mean_baseline_utility": mean(lambda r: r["baseline"]["utility"]),
                "mean_candidate_utility": mean(lambda r: r["candidate"]["utility"]), "mean_utility_delta": mean(lambda r: r["deltas"]["utility"]),
                "mean_recruits_delta": mean(lambda r: r["deltas"]["recruits_completed"]),
                "baseline_loss_rate": mean(lambda r: r["baseline"]["defeat"]), "candidate_loss_rate": mean(lambda r: r["candidate"]["defeat"]),
                "baseline_win_rate": mean(lambda r: r["baseline"]["victory"]), "candidate_win_rate": mean(lambda r: r["candidate"]["victory"]),
                "changed_futures": sum(not r["behavior_equal"] for r in selected)}
            self.metrics(f"rts.{split}.{identifier}.aggregate", reported, row)
            aggregates.append(row)
        self.rts_runs[split] = independent
        self.summaries["rts_" + split] = {"planned_runs": len(expected_units), "audited_runs": len(independent),
            "seed_count": len(seeds), "control_pairs": len(controls), "controls_pass": controls_pass,
            "aggregate": aggregates, "saved_final_checkpoint_bodies": saved_checkpoint_bodies,
            "checkpoint_scope": "Raw records preserve final checkpoint digests, but initial checkpoint bodies and default final checkpoint bodies were not saved. Hash link consistency is checked; absent bodies cannot be rehashed or independently replayed."}

    def rts_selection(self) -> None:
        selection = self.load("rts-selection/selection.json")
        self.receipt_digest("rts.selection", selection)
        self.metrics("rts.selection.identity", selection, {"schema": "wsp-selection/1",
            "protocol_sha256": self.freeze["protocol_sha256"], "freeze_sha256": self.freeze["sha256"],
            "validation_results_sha256": self.hash("rts-validation/results.json"),
            "candidate_bundle_sha256": self.hash("rts-candidates/programs.json"),
            "test_programs_sha256": self.hash("rts-selection/programs.json"),
            "rule": self.protocol["rts"]["selection_rule"], "test_used_for_selection": False})
        delta_by_id = {f"recruit_{str(float(delta)).replace('.', '_')}": delta
                      for delta in self.protocol["rts"]["candidate_recruitment_deltas"]}
        rows = []
        for identifier in [CONTROLS[0], *[p["id"] for p in self.rts_bundle["candidates"]]]:
            runs = [r for r in self.rts_runs["validation"] if r["candidate_id"] == identifier]
            losses, base_losses = sum(r["candidate"]["defeat"] for r in runs), sum(r["baseline"]["defeat"] for r in runs)
            mean_utility = sum((q(float(r["candidate"]["utility"])) for r in runs), Fraction(0)) / len(runs)
            rows.append({"id": identifier, "delta": delta_by_id.get(identifier, 0), "mean_utility_exact": str(mean_utility),
                "mean_utility": float(mean_utility), "losses": losses, "baseline_losses": base_losses,
                "eligible": losses <= base_losses, "n": len(runs)})
        winner = min((row for row in rows if row["eligible"]), key=lambda row: (-Fraction(row["mean_utility_exact"]), row["delta"]))
        selected = [p for p in self.rts_bundle["candidates"] if p["id"] == winner["id"]]
        self.same("rts.selection.exact_validation_rows", selection["validation_rows"], rows)
        self.metrics("rts.selection.winner", selection, {"selected_id": winner["id"], "selected_delta": winner["delta"],
                                                       "baseline_selected": not selected})
        self.same("rts.selection.exact_test_bundle", self.load("rts-selection/programs.json"), {"candidates": selected})
        self.same("rts.selection.files", sorted(p.name for p in (self.root / "rts-selection").iterdir()),
                  ["programs.json", "selection.json"])
        start = self.load("logs/07-rts-select.exit.json")
        stamp = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        self.check("rts.selection.timestamp", stamp(start["started_at"]) <= stamp(selection["created_at"]) <= stamp(start["finished_at"]),
                   "self-recorded timestamp")
        self.selection = selection
        self.summaries["rts_selection"] = {"selected_id": winner["id"], "selected_delta": winner["delta"],
            "baseline_selected": not selected, "validation_rows": rows,
            "selection_scope": "Reconstructed solely from saved validation raw branch metrics with exact rational means of binary64 utility values; timestamps and linkage are local evidence, not proof of no unseen earlier runs."}

    def rts_hypothesis(self) -> None:
        report = self.load("rts-summary.json")
        identifier = self.selection["selected_id"]
        runs = [run for run in self.rts_runs["test"] if run["candidate_id"] == identifier]
        self.same("rts.H3.planned_units", [r["seed"] for r in runs], self.protocol["rts"]["test_seeds"])
        self.metrics("rts.H3.identity", report, {"schema": "wsp-rts-hypothesis/1",
            "protocol_sha256": self.freeze["protocol_sha256"], "freeze_sha256": self.freeze["sha256"],
            "selection_sha256": self.selection["sha256"], "test_results_sha256": self.hash("rts-test/results.json"),
            "selected_id": identifier})
        effect = bootstrap([run["deltas"]["utility"] for run in runs], self.protocol["statistical_policy"])
        losses, base_losses = sum(r["candidate"]["defeat"] for r in runs), sum(r["baseline"]["defeat"] for r in runs)
        h3 = {"effect": effect, "candidate_losses": losses, "baseline_losses": base_losses,
            "candidate_wins": sum(r["candidate"]["victory"] for r in runs), "baseline_wins": sum(r["baseline"]["victory"] for r in runs),
            "status": "supported" if effect["ci95"][0] > 0 and losses <= base_losses else "unsupported"}
        self.metrics("rts.H3.effect", report["H3"]["effect"], effect)
        self.metrics("rts.H3.result", report["H3"], {k: v for k, v in h3.items() if k != "effect"})
        other = {"candidate_mean_recruits": float(np.mean([r["candidate"]["recruits_completed"] for r in runs])),
            "baseline_mean_recruits": float(np.mean([r["baseline"]["recruits_completed"] for r in runs])),
            "candidate_finished_before_270": sum(r["candidate"]["elapsed_seconds"] < 270 for r in runs),
            "baseline_finished_before_270": sum(r["baseline"]["elapsed_seconds"] < 270 for r in runs),
            "controls_pass": self.summaries["rts_test"]["controls_pass"]}
        self.metrics("rts.H3.additional", report, other)
        decomposition = {}
        for label in ("baseline", "candidate"):
            recruitment = float(np.mean([.02 * r[label]["recruits_completed"] for r in runs]))
            town_damage = float(np.mean([(r[label]["opponent_town_hp_lost"] - r[label]["own_town_hp_lost"]) / self.initial_town_hp for r in runs]))
            utility = float(np.mean([r[label]["utility"] for r in runs]))
            self.close(f"rts.H3.{label}.utility_decomposition", utility, recruitment + town_damage)
            decomposition[label] = {"mean_utility": utility, "mean_recruitment_contribution": recruitment,
                                    "mean_net_town_hp_contribution": town_damage}
        self.summaries["rts_hypothesis"] = {"selected_id": identifier, "H3": h3, **other,
            "utility_decomposition": decomposition,
            "units": [{"seed": r["seed"], "utility_difference": r["deltas"]["utility"],
                "baseline_utility": r["baseline"]["utility"], "candidate_utility": r["candidate"]["utility"],
                "baseline_elapsed_seconds": r["baseline"]["elapsed_seconds"], "candidate_elapsed_seconds": r["candidate"]["elapsed_seconds"],
                "baseline_recruits_completed": r["baseline"]["recruits_completed"], "candidate_recruits_completed": r["candidate"]["recruits_completed"],
                "baseline_outcome": r["baseline"]["outcome"], "candidate_outcome": r["candidate"]["outcome"]} for r in runs],
            "scope": "Single mission, one frozen opponent, net town-HP loss and completed-recruitment utility proxy, finite horizon. Equal observed losses are not statistical noninferiority.",
            "baseline_selection_scope": "When baseline wins validation, test evaluates that selected unchanged policy and both controls; it provides no test effect estimate for an edited policy."}


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            require(path.resolve().is_relative_to(root), f"Evidence symlink escapes study root: {relative}")
            result[relative] = file_sha(path)
    require(bool(result), "Empty study evidence inventory")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True, help="Existing frozen study directory; opened read-only")
    parser.add_argument("--out", type=Path, required=True, help="New JSON report outside the study directory; never overwritten")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Current workspace for informational source differences only")
    parser.add_argument("--expected-protocol-sha256", help="Optional externally supplied raw protocol SHA-256")
    options = parser.parse_args(argv)
    root, output, repo = options.study_root.resolve(), options.out.resolve(), options.repo.resolve()
    if not root.is_dir():
        parser.error("--study-root must be an existing directory")
    if output.is_relative_to(root):
        parser.error("--out must be outside the study directory")
    if output.exists():
        parser.error("--out already exists; choose a fresh report path")
    if not output.parent.is_dir():
        parser.error("--out parent directory must already exist")
    if options.expected_protocol_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", options.expected_protocol_sha256):
        parser.error("--expected-protocol-sha256 must be 64 lowercase hexadecimal characters")
    audit = Audit(root, repo)
    started_at = datetime.now(timezone.utc).isoformat()
    auditor_sha = file_sha(Path(__file__))
    before: dict[str, str] = {}
    try:
        before = inventory(root)
        audit.check("audit.initial_inventory", True, "raw file SHA-256", {"files": len(before)})
    except Exception as error:
        audit.check("audit.initial_inventory", False, "structural", {"error": f"{type(error).__name__}: {error}"})
    stages = [
        ("provenance", lambda: audit.provenance(options.expected_protocol_sha256)),
        ("execution_logs", audit.logs),
        ("synthetic", audit.synthetic),
        ("rts_export", audit.rts_export),
        ("rts_candidates", audit.rts_candidates),
        ("rts_development", lambda: audit.rts_evaluation("development")),
        ("rts_validation", lambda: audit.rts_evaluation("validation")),
        ("rts_selection", audit.rts_selection),
        ("rts_test", lambda: audit.rts_evaluation("test")),
        ("rts_hypothesis", audit.rts_hypothesis),
    ]
    for name, function in stages:
        audit.stage(name, function)
    try:
        after = inventory(root)
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        audit.check("audit.evidence_unchanged", before == after, "raw file SHA-256",
                    {"files_before": len(before), "files_after": len(after), "changed": changed})
    except Exception as error:
        audit.check("audit.evidence_unchanged", False, "structural", {"error": f"{type(error).__name__}: {error}"})
    audit.same("audit.auditor_unchanged", file_sha(Path(__file__)), auditor_sha)
    failed = [check["id"] for check in audit.checks if check["required"] and not check["passed"]]
    required_checks = [check for check in audit.checks if check["required"]]
    report = {
        "schema": "wsp-independent-audit/1", "passed": not failed, "study_root": str(root),
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "auditor": {"path": str(Path(__file__).resolve()), "sha256": auditor_sha,
                    "imports_study_runtime": False, "executes_training_data_generation_or_simulation": False},
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "source_files_sha256": before, "source_inventory_sha256": digest(before),
        "source_inventory_scope": "Every file beneath study_root, keyed by relative POSIX path and hashed as raw bytes before audit; the final inventory must match.",
        "comparison_policy": {
            "exact": "Raw SHA-256 identities, literal tensor edits, rational certificate arithmetic, integer counts, receipt/link schemas and validation selection fractions.",
            "numerical": {"absolute_tolerance": ATOL, "relative_tolerance": RTOL,
                          "scope": "Independent NumPy forward, metrics and bootstrap comparisons; deterministic TS sequential fixture scores separately require zero error."},
            "bootstrapping": "Only resampling of saved independent endpoint units; no new experimental units or reserved experiment seed executions.",
            "historical_sources": "Authoritative freeze/sources copies and hashes are mandatory; current workspace differences are informational."
        },
        "check_totals": {"required": len(required_checks), "required_passed": len(required_checks) - len(failed),
                         "required_failed": len(failed), "informational": len(audit.checks) - len(required_checks)},
        "failed_check_ids": failed, "checks": audit.checks, "summaries": audit.summaries,
        "limitations": [
            "Internal hash consistency and local timestamps do not authenticate authorship or establish external preregistration or absence of unrecorded experiments.",
            "No training, seeded dataset/gauge generation, compilation optimization, alignment optimization or game simulation is rerun. Saved edits and gauge applications are reconstructed; reported losses and endpoints are recomputed.",
            "Exact rational certificates are independently recalculated for saved binary64 parameters; this script does not invoke Lean or establish IEEE floating-point semantic equivalence.",
            "Synthetic inference uses disjoint initialization units conditional on the single shared saved data partition; the three exploratory intervals are unadjusted for multiplicity.",
            "Source useful-and-admitted fractions are descriptive per-program endpoints; the predeclared H2 useful-gain condition concerns the mean gain.",
            "RTS fixtures establish scores and ranks on saved feature vectors, without re-extracting features or replaying legality from snapshots.",
            "RTS initial checkpoint bodies and default final checkpoint bodies are absent. Their recorded digests can be linked but not recomputed; raw production metrics cannot be independently recovered from absent states.",
            "The 1800-HP utility denominator is corroborated by inspected frozen mission initialization and exported snapshots, not by regenerating unsaved initial worlds or formally proving TypeScript.",
            "Successful order histories are saved, but complete per-decision histories including waits/rejected commands are not. First-change records are structurally checked and linked, not fully replayed.",
            "RTS effects apply only to the selected policy, one mission/ruleset, frozen opponent and declared utility proxy. Equal observed losses are not statistical noninferiority; baseline selection provides no held-out effect estimate for an edited policy."
        ],
    }
    # Exclusive creation is intentional: later auditor fixes must produce a new
    # report rather than silently replacing any earlier failure evidence.
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"schema": report["schema"], "passed": report["passed"], "out": str(output),
        "report_sha256": file_sha(output), "source_files": len(before),
        "check_totals": report["check_totals"], "failed_check_ids": failed}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
