"""Deterministically embed an exact RTS policy and sparse edits in Lean.

This program is an exporter, not part of Lean's trusted kernel.  Its output
contains every base parameter, explicit edited endpoints, and proofs about
real arithmetic.  Hashes are provenance labels; independently regenerating
and inspecting the output checks the external-data mapping.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re


def rational(value: object, location: str) -> Fraction:
    if not isinstance(value, str) or not re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", value):
        raise ValueError(f"{location}: expected an exact rational string")
    result = Fraction(value)
    if str(result) != value:
        raise ValueError(f"{location}: rational must use canonical reduced form")
    return result


def hash_label(value: object, location: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{location}: expected a lowercase SHA256 digest")
    return value


def binary64_rational(value: object, location: str) -> Fraction:
    result = rational(value, location)
    try:
        runtime_value = float(result)
    except OverflowError as error:
        raise ValueError(f"{location}: coefficient exceeds finite binary64 range") from error
    if not math.isfinite(runtime_value) or Fraction.from_float(runtime_value) != result:
        raise ValueError(f"{location}: coefficient is not exactly representable in binary64")
    return result


def lean_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator) if value >= 0 else f"({value.numerator})"
    return f"({value.numerator} / {value.denominator} : ℝ)"


def verify_source(payload: dict, source_path: Path) -> None:
    """Independently compare all exported coefficients and content identities."""
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != payload["source_rts_file_sha256"]:
        raise ValueError("source model file SHA256 differs from the exact-policy label")
    source = json.loads(source_bytes)
    feature_names, hidden_names = source["featureNames"], source["hiddenNames"]
    for names, count, label in ((feature_names, 32, "featureNames"), (hidden_names, 38, "hiddenNames")):
        if (not isinstance(names, list) or len(names) != count
                or not all(isinstance(name, str) and name.strip() for name in names)
                or len(set(names)) != count):
            raise ValueError(f"source {label} has invalid dimensions or labels")
    if (feature_names[payload["feature_index"]] != "train_army"
            or hidden_names[payload["hidden_index"]] != "army_train_bias"):
        raise ValueError("source feature or hidden labels differ from the selected semantics")

    def compare(exact: object, runtime: object, location: str) -> object:
        if isinstance(exact, list):
            if not isinstance(runtime, list) or len(runtime) != len(exact):
                raise ValueError(f"source shape differs at {location}")
            return [compare(a, b, f"{location}[{i}]") for i, (a, b) in enumerate(zip(exact, runtime))]
        if isinstance(runtime, bool) or not isinstance(runtime, (int, float)):
            raise ValueError(f"source coefficient is not numeric at {location}")
        represented = binary64_rational(exact, location)
        if isinstance(runtime, int):
            source_value = Fraction(runtime)
        else:
            if not math.isfinite(runtime):
                raise ValueError(f"source coefficient is not finite at {location}")
            source_value = Fraction.from_float(runtime)
        if source_value != represented:
            raise ValueError(f"source coefficient differs at {location}")
        return float(represented)  # Match the network format's canonical unsigned zero.

    tensors = {field: compare(payload[field], source[field], field) for field in ("w1", "b1", "w2", "b2")}
    network = {"schema": "wsp-network/1", "activation": "relu", "dtype": "float64",
               "input_names": feature_names, "output_names": ["candidate_score"],
               "w1": tensors["w1"], "b1": tensors["b1"], "w2": [tensors["w2"]], "b2": [tensors["b2"]]}

    def digest(value: dict) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    if digest(network) != payload["base_sha256"]:
        raise ValueError("base network SHA256 does not match independently reconstructed content")
    for edit in payload["edits"]:
        readout = list(tensors["w2"])
        readout[payload["hidden_index"]] = float(Fraction(edit["after_w2"]))
        if digest(network | {"w2": [readout]}) != edit["candidate_sha256"]:
            raise ValueError(f"candidate {edit['id']} SHA256 does not match independently reconstructed content")


def emit_policy(payload: dict, exact_file_sha256: str) -> str:
    if payload.get("schema") != "wsp-rational-policy/1":
        raise ValueError("unsupported exact-policy schema")
    if payload.get("arithmetic") != "Fraction.from_float: exact binary64 rational value":
        raise ValueError("exact-policy arithmetic provenance is missing or unsupported")
    base_hash = hash_label(payload.get("base_sha256"), "base_sha256")
    source_hash = hash_label(payload.get("source_rts_file_sha256"), "source_rts_file_sha256")
    raw_w1 = payload.get("w1")
    if not isinstance(raw_w1, list) or not raw_w1 or not isinstance(raw_w1[0], list) or not raw_w1[0]:
        raise ValueError("w1 must be a nonempty hidden-by-feature matrix")
    hidden_count, feature_count = len(raw_w1), len(raw_w1[0])
    if (feature_count, hidden_count) != (32, 38):
        raise ValueError("this RTS certificate expects exactly 32 features and 38 hidden units")
    if any(not isinstance(row, list) or len(row) != feature_count for row in raw_w1):
        raise ValueError("w1 rows have inconsistent dimensions")
    w1 = [[binary64_rational(v, f"w1[{h}][{f}]") for f, v in enumerate(row)]
          for h, row in enumerate(raw_w1)]
    vectors = {}
    for field in ("b1", "w2"):
        raw = payload.get(field)
        if not isinstance(raw, list) or len(raw) != hidden_count:
            raise ValueError(f"{field} must have {hidden_count} entries")
        vectors[field] = [binary64_rational(v, f"{field}[{h}]") for h, v in enumerate(raw)]
    b1, w2 = vectors["b1"], vectors["w2"]
    b2 = binary64_rational(payload.get("b2"), "b2")
    selected, feature = payload.get("hidden_index"), payload.get("feature_index")
    if type(selected) is not int or not 0 <= selected < hidden_count:
        raise ValueError("hidden_index is out of range")
    if type(feature) is not int or not 0 <= feature < feature_count:
        raise ValueError("feature_index is out of range")
    if (selected, feature) != (19, 24):
        raise ValueError("this integration expects army_train_bias=19 and train_army=24")
    if b1[selected] != 0 or any(v != int(f == feature) for f, v in enumerate(w1[selected])):
        raise ValueError("the selected hidden unit is not exactly ReLU(train_army)")
    raw_edits = payload.get("edits")
    if not isinstance(raw_edits, list) or not raw_edits:
        raise ValueError("at least one explicit readout edit is required")
    edits, ids = [], set()
    for index, raw_edit in enumerate(raw_edits):
        if not isinstance(raw_edit, dict):
            raise ValueError(f"edits[{index}] must be an object")
        identifier = raw_edit.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"recruit_[0-9]+_[0-9]+", identifier):
            raise ValueError(f"edits[{index}].id is not a recruitment candidate identifier")
        if identifier in ids:
            raise ValueError(f"duplicate edit identifier: {identifier}")
        ids.add(identifier)
        delta = rational(raw_edit.get("actual_delta"), f"{identifier}.actual_delta")
        after = binary64_rational(raw_edit.get("after_w2"), f"{identifier}.after_w2")
        if after - w2[selected] != delta:
            raise ValueError(f"{identifier}: actual_delta disagrees with exact endpoint subtraction")
        if delta == 0:
            raise ValueError(f"{identifier}: expected a nonzero edit")
        candidate_hash = hash_label(raw_edit.get("candidate_sha256"), f"{identifier}.candidate_sha256")
        edits.append((identifier, delta, after, candidate_hash))

    lines = [
        "import WeightSpace.Contracts",
        "",
        "/-!",
        "# Exact real-arithmetic certificates for the frozen RTS integration",
        "",
        "Generated by generate_rts_certificate.py; do not edit by hand.",
        f"Exact policy file SHA256: {exact_file_sha256}",
        f"Base network SHA256: {base_hash}",
        f"Source RTS model file SHA256: {source_hash}",
        "",
        "Every serialized binary64 parameter is embedded as its exact rational value.",
        "The score and all proofs use real arithmetic, without an IEEE rounding bound.",
        "The source and candidate hashes are provenance labels, not byte-binding proofs.",
        "Feature and hidden indices preserve the exported ordering; the selected unit",
        "is army_train_bias (19), whose input is train_army (24).",
        "-/",
        "",
        "noncomputable section",
        "",
        "namespace WeightSpace.RTS",
        "",
        "def baseW1 (h : Fin 38) (f : Fin 32) : ℝ :=",
        "  match h.val with",
    ]
    for h, row in enumerate(w1):
        if any(row):
            lines.extend([f"  | {h} => match f.val with"])
            lines.extend(f"    | {f} => {lean_number(v)}" for f, v in enumerate(row) if v)
            lines.append("    | _ => 0")
    lines.extend(["  | _ => 0", ""])
    for name, vector in (("baseB1", b1), ("baseW2", w2)):
        lines.extend([f"def {name} (h : Fin 38) : ℝ :=", "  match h.val with"])
        lines.extend(f"  | {h} => {lean_number(v)}" for h, v in enumerate(vector) if v)
        lines.extend(["  | _ => 0", ""])
    lines.extend([
        "def base : Params (Fin 32) (Fin 38) where",
        "  w1 := baseW1",
        "  b1 := baseB1",
        "  w2 := baseW2",
        f"  b2 := {lean_number(b2)}",
        "",
        "def globalBox : InputBox (Fin 32) where",
        "  lower _ := 0",
        "  upper _ := 1",
        "",
        "def protectedBox : InputBox (Fin 32) where",
        "  lower _ := 0",
        "  upper f := if f = 24 then 0 else 1",
        "",
        "theorem globalBox_valid : globalBox.Valid := by",
        "  intro f",
        "  norm_num [globalBox]",
        "",
        "theorem protectedBox_valid : protectedBox.Valid := by",
        "  intro f",
        "  simp only [protectedBox]",
        "  split <;> norm_num",
        "",
        "theorem selected_hidden (x : Fin 32 → ℝ) :",
        "    hidden base x 19 = max 0 (x 24) := by",
        "  norm_num [hidden, preactivation, base, baseW1, baseB1, Fin.sum_univ_succ, Fin.succ] <;> rfl",
        "",
        "theorem selected_global_upper : hiddenUpper base globalBox 19 = 1 := by",
        "  norm_num [hiddenUpper, base, baseW1, baseB1, globalBox, Fin.sum_univ_succ, Fin.succ]",
        "",
        "theorem selected_protected_upper : hiddenUpper base protectedBox 19 = 0 := by",
        "  norm_num [hiddenUpper, base, baseW1, baseB1, protectedBox, Fin.sum_univ_succ,",
        "    Fin.succ, Fin.ext_iff]",
        "",
    ])
    for identifier, delta, after, candidate_hash in edits:
        d, a, budget = lean_number(delta), lean_number(after), lean_number(abs(delta))
        lines.extend([
            f"/- Candidate network SHA256: {candidate_hash} -/",
            f"def {identifier} : Params (Fin 32) (Fin 38) :=",
            f"  {{ base with w2 := fun h => if h = 19 then {a} else base.w2 h }}",
            "",
            f"theorem {identifier}_eq :",
            f"    {identifier} = editReadoutAt base 19 {d} := by",
            f"  unfold {identifier} editReadoutAt editReadout",
            "  simp only [add_zero]",
            "  congr 1",
            "  funext h",
            "  by_cases hh : h = 19",
            "  · subst h",
            "    norm_num [base, baseW2]",
            "  · simp [hh]",
            "",
            f"theorem {identifier}_exact (x : Fin 32 → ℝ) :",
            f"    score {identifier} x - score base x = {d} * max 0 (x 24) := by",
            f"  rw [{identifier}_eq, readout_single_exact, selected_hidden]",
            "",
            f"theorem {identifier}_bound (x : Fin 32 → ℝ) (hx : InBox globalBox x) :",
            f"    |score {identifier} x - score base x| ≤ {budget} := by",
            f"  rw [{identifier}_eq]",
            f"  apply readout_single_drift_le_of_budget base 19 {d} globalBox {budget} _ x hx",
            "  rw [selected_global_upper]",
            "  norm_num",
            "",
            f"theorem {identifier}_protected (x : Fin 32 → ℝ) (hx : InBox protectedBox x) :",
            f"    score {identifier} x = score base x := by",
            f"  rw [{identifier}_eq]",
            f"  have hd := readout_single_drift_le base 19 {d} protectedBox x hx",
            "  rw [selected_protected_upper, mul_zero] at hd",
            "  exact sub_eq_zero.mp (abs_nonpos_iff.mp hd)",
            "",
        ])
    lines.extend(["end WeightSpace.RTS", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exact_policy", type=Path, help="frozen exact-policy.json exported by the RTS study")
    parser.add_argument("output", type=Path, help="destination Lean module, normally WeightSpace/RTS.lean")
    parser.add_argument("--check", action="store_true", help="compare output byte-for-byte; never modify it")
    parser.add_argument("--source-model", type=Path,
                        help="also independently check source coefficients, selected labels, and network identities")
    args = parser.parse_args()
    source = args.exact_policy.read_bytes()
    payload = json.loads(source)
    if not isinstance(payload, dict):
        raise ValueError("exact-policy JSON must be an object")
    rendered = emit_policy(payload, hashlib.sha256(source).hexdigest()).encode("utf-8")
    if args.source_model is not None:
        verify_source(payload, args.source_model)
        print("Source parameters, selected labels, and network SHA256 identities match.")
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != rendered:
            raise SystemExit(f"certificate differs from deterministic regeneration: {args.output}")
        print(f"Certificate regeneration matches: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rendered)
        print(f"Wrote exact real-arithmetic certificate: {args.output}")


if __name__ == "__main__":
    main()
