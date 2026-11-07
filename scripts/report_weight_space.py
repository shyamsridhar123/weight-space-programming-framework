"""Build a source-backed Markdown report and standalone research figures.

This analysis never trains a model, selects a candidate, or runs a simulation.
It requires a completed frozen study and a passing independent evidence audit.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlcd.weight_space.evidence import (
    bootstrap_mean, environment, file_hash, read_json, utc_now,
    execution_sources, verify_manifest, write_json,
)
from rlcd.weight_space.model import Network, digest


COLORS = {"ink": "#203247", "muted": "#697784", "teal": "#137D91",
          "gold": "#B87425", "gray": "#D5DCE1", "paper": "#FFFFFF"}
ARMS = ("target_only", "retention", "raw", "aligned")
LABELS = {"target_only": "Target only", "retention": "With retention",
          "raw": "Raw transfer", "aligned": "Aligned transfer"}


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"Recomputed {label} differs: {actual} != {expected}")


def stat(values: list[float], protocol: dict[str, Any]) -> dict[str, Any]:
    config = protocol["statistical_policy"]
    return bootstrap_mean(values, seed=config["bootstrap_seed"],
                          resamples=config["bootstrap_resamples"])


def compare_stat(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    if actual["n"] != expected["n"]:
        raise ValueError(f"{label} sample sizes differ")
    for key in ("mean",):
        close(actual[key], expected[key], f"{label}.{key}")
    for i in range(2):
        close(actual["ci95"][i], expected["ci95"][i], f"{label}.ci95[{i}]")


def csv_file(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No chart rows for {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path, origin: Path) -> str:
    return Path(os.path.relpath(path, origin)).as_posix()


def interval(value: dict[str, Any], digits: int = 4) -> str:
    lo, hi = value["ci95"]
    return f"{value['mean']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def frozen_snapshot(study: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate historical evidence independently of later workspace repairs.

    Experiment execution still uses the stricter runtime verify_freeze, which
    requires the live source tree to match. Reporting must retain the original
    source identity and disclose later differences instead of rewriting history.
    """
    receipt = read_json(study / "freeze/receipt.json")
    payload = {key: value for key, value in receipt.items() if key != "sha256"}
    if receipt.get("schema") != "wsp-freeze/1" or receipt.get("sha256") != digest(payload):
        raise ValueError("Malformed or altered freeze receipt")
    if file_hash(study / "freeze/protocol.json") != receipt["protocol_sha256"]:
        raise ValueError("Frozen protocol differs from its receipt")
    verify_manifest(study / "freeze/sources", receipt["source_sha256"])
    current = {path.relative_to(ROOT).as_posix(): file_hash(path)
               for path in execution_sources()}
    differences = [{"path": name, "frozen_sha256": receipt["source_sha256"].get(name),
                    "workspace_sha256": current.get(name)}
                   for name in sorted(set(current) | set(receipt["source_sha256"]))
                   if current.get(name) != receipt["source_sha256"].get(name)]
    return receipt, differences


def load_evidence(study: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(study / "freeze/protocol.json")
    freeze, workspace_differences = frozen_snapshot(study)
    sources: dict[str, str] = {}
    for directory in ("synthetic", "rts-export", "rts-candidates", "rts-development",
                      "rts-validation", "rts-test"):
        receipt_path = study / directory / "receipt.json"
        receipt = read_json(receipt_path)
        verify_manifest(receipt_path.parent, receipt["artifacts"])
        sources[f"{directory}/receipt.json"] = file_hash(receipt_path)
    for name in ("freeze/receipt.json", "freeze/protocol.json", "synthetic/results.json",
                 "rts-selection/selection.json", "rts-selection/programs.json", "rts-summary.json"):
        sources[name] = file_hash(study / name)
    audit = read_json(audit_path)
    if audit.get("passed") is not True:
        raise ValueError("Report requires a passing independent evidence audit")
    if Path(audit["study_root"]).resolve() != study:
        raise ValueError("Independent audit identifies a different study")
    for name, expected in audit["source_files_sha256"].items():
        path = (study / name).resolve()
        if not path.is_relative_to(study) or file_hash(path) != expected:
            raise ValueError(f"Evidence differs from independent audit: {name}")
    synthetic = read_json(study / "synthetic/results.json")
    if not synthetic["complete"] or not synthetic["controls_pass"] or synthetic["failures"]:
        raise ValueError("This report requires all planned synthetic units and passing controls")
    pairs = synthetic["pairs"]
    planned = list(zip(protocol["networks"]["training_seeds"][::2],
                       protocol["networks"]["training_seeds"][1::2]))
    if [(p["source_seed"], p["target_seed"]) for p in pairs] != planned:
        raise ValueError("Synthetic pair identities/order differ from protocol")

    with np.load(study / "synthetic/data.npz") as data:
        inputs = data["test_x"].copy()
        partitions = {key: int(len(data[f"{key}_x"])) for key in
                      ("train", "compile", "validation", "test")}
    shift = protocol["networks"]["requested_shift"]
    mask = inputs[:, 0] > protocol["networks"]["target_threshold"]
    rows, arms = [], []
    for index, pair in enumerate(pairs, 1):
        directory = study / "synthetic" / f"pair-{pair['source_seed']}-{pair['target_seed']}"
        source = Network.from_dict(read_json(study / "synthetic/models" / f"{pair['source_seed']}.json"))
        target = Network.from_dict(read_json(study / "synthetic/models" / f"{pair['target_seed']}.json"))
        if source.sha256 != pair["source_sha256"] or target.sha256 != pair["target_sha256"]:
            raise ValueError("Model identity differs from pair result")
        arm_rows = {}
        reference = None
        for arm in ARMS:
            prefix = arm if arm in ("target_only", "retention") else f"transfer-{arm}"
            candidate = Network.from_dict(read_json(directory / f"{prefix}-model.json"))
            admission_path = directory / f"{prefix}-admission.json"
            admission = read_json(admission_path)
            base = source if arm in ("target_only", "retention") else target
            change = candidate(inputs) - base(inputs)
            gain = float(change[mask, 0].mean()/shift)
            retention = float(np.sqrt(np.mean(change[~mask]**2)))
            if arm == "retention":
                reference = change[mask, 0].copy()
            transfer_error = (float(np.sqrt(np.mean((change[mask, 0]-reference)**2))/shift)
                              if arm in ("raw", "aligned") else None)
            if transfer_error is not None:
                close(transfer_error, pair["transfer"][arm]["normalized_target_delta_rmse"], arm)
            else:
                for key, value in (("target_gain_fraction", gain), ("retention_rmse", retention)):
                    close(value, pair["metrics"][arm]["test"][key], f"{arm}.{key}")
            row = {"pair": index, "source_seed": pair["source_seed"], "target_seed": pair["target_seed"],
                   "arm": arm, "admitted": bool(admission["accepted"]),
                   "target_gain_fraction": gain, "retention_rmse": retention,
                   "normalized_transfer_error": transfer_error,
                   "maximum_observed_drift": float(np.abs(change).max()),
                   "admission_file": admission_path.relative_to(study).as_posix(),
                   "admission_file_sha256": file_hash(admission_path),
                   "certificate_sha256": admission["certificate"]["sha256"],
                   "certified_output0_bound_exact": admission["certificate"]["output_bounds"][0],
                   "certified_output0_bound": float(Fraction(admission["certificate"]["output_bounds"][0])),
                   "useful_descriptive": gain >= .5,
                   "useful_and_admitted": gain >= .5 and bool(admission["accepted"])}
            arms.append(row)
            arm_rows[arm] = row
        row = {"pair": index, "source_seed": pair["source_seed"], "target_seed": pair["target_seed"],
               "raw_error": arm_rows["raw"]["normalized_transfer_error"],
               "aligned_error": arm_rows["aligned"]["normalized_transfer_error"],
               "h1_improvement": arm_rows["raw"]["normalized_transfer_error"]-arm_rows["aligned"]["normalized_transfer_error"],
               "target_only_retention_rmse": arm_rows["target_only"]["retention_rmse"],
               "retention_rmse": arm_rows["retention"]["retention_rmse"],
               "h2_reduction": arm_rows["target_only"]["retention_rmse"]-arm_rows["retention"]["retention_rmse"],
               "target_only_gain": arm_rows["target_only"]["target_gain_fraction"],
               "retention_gain": arm_rows["retention"]["target_gain_fraction"],
               "known_transport_error": pair["controls"]["known_transport_error"],
               "inferred_known_error": pair["controls"]["inferred_known_gauge_error"],
               "raw_permuted_error": pair["controls"]["raw_permuted_error"]}
        close(row["h1_improvement"], pair["H1_raw_minus_aligned"], "H1 pair effect")
        close(row["h2_reduction"], pair["H2_retention_rmse_reduction"], "H2 pair effect")
        rows.append(row)
    h1 = stat([r["h1_improvement"] for r in rows], protocol)
    h2 = stat([r["h2_reduction"] for r in rows], protocol)
    gain = stat([r["retention_gain"] for r in rows], protocol)
    compare_stat(h1, synthetic["H1"]["effect"], "H1")
    compare_stat(h2, synthetic["H2"]["effect"], "H2")
    compare_stat(gain, synthetic["H2"]["target_gain_fraction"], "H2 gain")
    admission_rows = []
    for arm in ARMS:
        values = [r for r in arms if r["arm"] == arm]
        admitted = sum(r["admitted"] for r in values)
        useful_admitted = sum(r["useful_and_admitted"] for r in values)
        admission_rows.append({"arm": arm, "n": len(values), "admitted": admitted,
                               "useful_descriptive": sum(r["useful_descriptive"] for r in values),
                               "useful_and_admitted": useful_admitted,
                               "rate_all_proposals": useful_admitted/len(values),
                               "rate_among_admitted": useful_admitted/admitted if admitted else None,
                               "mean_gain": float(np.mean([r["target_gain_fraction"] for r in values])),
                               "mean_retention_rmse": float(np.mean([r["retention_rmse"] for r in values])),
                               "mean_output0_bound": float(np.mean([r["certified_output0_bound"] for r in values]))})

    selection = read_json(study / "rts-selection/selection.json")
    rts_summary = read_json(study / "rts-summary.json")
    split_results = {name: read_json(study / f"rts-{name}/results.json")
                     for name in ("development", "validation", "test")}
    selected = [r for r in split_results["test"]["runs"] if r["candidate_id"] == selection["selected_id"]]
    if [r["seed"] for r in selected] != protocol["rts"]["test_seeds"]:
        raise ValueError("Selected test units differ from protocol")
    rts_rows = []
    for run in selected:
        row = {"seed": run["seed"], "utility_delta": run["deltas"]["utility"],
               "recruits_delta": run["deltas"]["recruits_completed"],
               "baseline_loss": run["baseline"]["defeat"], "candidate_loss": run["candidate"]["defeat"]}
        for branch in ("baseline", "candidate"):
            value = run[branch]
            town = value["opponent_town_hp_lost"]/value["opponent_town_max_hp"] - value["own_town_hp_lost"]/value["town_max_hp"]
            recruitment = .02*value["recruits_completed"]
            close(town+recruitment, value["utility"], f"{branch} utility")
            row.update({f"{branch}_utility": value["utility"], f"{branch}_town_component": town,
                        f"{branch}_recruitment_component": recruitment,
                        f"{branch}_recruits": value["recruits_completed"],
                        f"{branch}_elapsed": value["elapsed_seconds"],
                        f"{branch}_win": value["victory"]})
        row["town_component_delta"] = row["candidate_town_component"]-row["baseline_town_component"]
        row["recruitment_component_delta"] = row["candidate_recruitment_component"]-row["baseline_recruitment_component"]
        close(row["town_component_delta"]+row["recruitment_component_delta"], row["utility_delta"], "utility decomposition")
        rts_rows.append(row)
    h3 = stat([r["utility_delta"] for r in rts_rows], protocol)
    compare_stat(h3, rts_summary["H3"]["effect"], "H3")
    split_rows = [{"split": name, "world_seeds": len(protocol["rts"][f"{name}_seeds"]),
                   "arms": len({r["candidate_id"] for r in value["runs"]}),
                   "paired_records": value["completed_runs"], "planned_records": value["planned_runs"],
                   "controls": sum(r["control"] for r in value["runs"]),
                   "failures": len(value["failures"]), "controls_pass": value["controls_pass"]}
                  for name, value in split_results.items()]
    training_mse = [t["test_mse"] for t in synthetic["training_summary"]]
    result = {"schema": "wsp-report-analysis/1", "created_at": utc_now(),
              "study_root": str(study), "freeze_sha256": freeze["sha256"],
              "protocol_sha256": freeze["protocol_sha256"], "frozen_at": freeze["created_at"],
              "source_files_sha256": sources, "audit_sha256": file_hash(audit_path),
              "workspace_source_differences": workspace_differences,
              "H1": {"status": synthetic["H1"]["status"], "effect": h1},
              "H2": {"status": synthetic["H2"]["status"], "effect": h2, "target_gain_fraction": gain},
              "H3": {"status": rts_summary["H3"]["status"], "effect": h3},
              "synthetic_partitions": partitions, "test_target_examples": int(mask.sum()),
              "test_retention_examples": int((~mask).sum()), "pairs": rows, "arms": arms,
              "admission": admission_rows, "training_test_mse": {
                  "mean": float(np.mean(training_mse)), "median": float(np.median(training_mse)),
                  "min": min(training_mse), "max": max(training_mse)},
              "selection": selection, "rts_test": rts_rows, "rts_splits": split_rows,
              "rts_summary": rts_summary,
              "fixture_agreement": read_json(study / "rts-candidates/fixture-agreement.json"),
              "exact_policy": read_json(study / "rts-candidates/exact-policy.json"),
              "descriptive_useful_threshold": .5,
              "descriptive_note": "Per-program gain >= 0.5 of the requested shift is a descriptive report criterion, not a predeclared hypothesis or a proof of usefulness."}
    return result, protocol


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.labelsize": 11, "text.color": COLORS["ink"],
                         "axes.labelcolor": COLORS["ink"], "xtick.color": COLORS["ink"],
                         "ytick.color": COLORS["ink"], "axes.edgecolor": COLORS["gray"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": COLORS["paper"], "axes.facecolor": COLORS["paper"],
                         "grid.color": "#E8ECEF", "grid.linewidth": .7,
                         "svg.hashsalt": "weight-space-foundations-v1"})


def finish_figure(fig: Any, out: Path, name: str, note: str) -> None:
    fig.text(.055, .025, note, ha="left", va="bottom", fontsize=9, color=COLORS["muted"])
    fig.savefig(out / f"{name}.png", dpi=180, facecolor=COLORS["paper"])
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def figures(data: dict[str, Any], out: Path) -> None:
    style()
    pairs = data["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(left=.075, right=.97, bottom=.20, top=.86, wspace=.34)
    fig.suptitle("Generic network edits: held-out effects across 20 independent pairs", x=.055,
                 ha="left", fontsize=16, fontweight="bold")
    raw = np.array([r["raw_error"] for r in pairs])
    aligned = np.array([r["aligned_error"] for r in pairs])
    maximum = max(raw.max(), aligned.max())*1.08
    axes[0].plot([0, maximum], [0, maximum], "--", color=COLORS["muted"], lw=1, label="Equal error")
    axes[0].scatter(raw, aligned, s=48, c=COLORS["teal"], edgecolors="white", linewidths=.7, zorder=3)
    axes[0].set(xlabel="Raw transfer error / requested shift", ylabel="Aligned transfer error / requested shift",
                xlim=(0, maximum), ylim=(0, maximum), title="H1 · Transferring the realized edit")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    for row in pairs:
        axes[1].plot([row["target_only_gain"], row["retention_gain"]],
                     [row["target_only_retention_rmse"], row["retention_rmse"]],
                     color=COLORS["gray"], lw=.9, zorder=1)
    axes[1].scatter([r["target_only_gain"] for r in pairs], [r["target_only_retention_rmse"] for r in pairs],
                    color=COLORS["gold"], marker="s", s=40, label="Target only", zorder=3)
    axes[1].scatter([r["retention_gain"] for r in pairs], [r["retention_rmse"] for r in pairs],
                    color=COLORS["teal"], marker="o", s=42, label="With retention", zorder=3)
    axes[1].axvline(.5, color=COLORS["muted"], ls="--", lw=1)
    axes[1].set(xlabel="Target gain / requested shift", ylabel="Collateral RMSE across all 3 outputs",
                title="H2 · Retention and target gain", xlim=(0, max(1.05, max(r["target_only_gain"] for r in pairs)*1.06)), ylim=(0, None))
    axes[1].legend(frameon=False, loc="upper left", fontsize=10)
    for ax in axes:
        ax.grid(axis="both", zorder=0)
        ax.set_axisbelow(True)
    finish_figure(fig, out, "generic-effects", "Each point is one planned pair/source. Rejected proposals remain in this numerical study.\nH1 compares transferred changes to the source edit; H2 collateral error uses non-target test inputs. Dashed gain reference: 0.5.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    fig.subplots_adjust(left=.07, right=.98, bottom=.26, top=.78, wspace=.45)
    fig.suptitle("Predeclared estimates and paired bootstrap uncertainty", x=.055,
                 ha="left", fontsize=16, fontweight="bold")
    specs = [(data["H1"]["effect"], "H1 · Transfer error reduction", "Raw minus aligned, normalized", 0),
             (data["H2"]["effect"], "H2 · Collateral reduction", "Output RMSE reduction", 0),
             (data["H2"]["target_gain_fraction"], "H2 · Mean target gain", "Fraction of requested shift", .5)]
    for ax, (value, title, xlabel, threshold) in zip(axes, specs):
        mean, (lo, hi) = value["mean"], value["ci95"]
        ax.axvline(threshold, color=COLORS["muted"], ls="--", lw=1)
        ax.errorbar(mean, 0, xerr=[[mean-lo], [hi-mean]], fmt="o", color=COLORS["teal"], capsize=6, lw=2, ms=7)
        ax.set(title=title, xlabel=xlabel, yticks=[], ylim=(-.6, .6))
        left, right = min(0, lo, threshold), max(hi, threshold)
        span = max(right-left, .05)
        ax.set_xlim(left-.08*span, right+.14*span)
        ax.text(.5, .82, interval(value, 3), transform=ax.transAxes, ha="center", fontsize=11)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
    finish_figure(fig, out, "generic-estimates", "95% percentile bootstrap intervals; 10,000 resamples of 20 paired units. Unadjusted exploratory inference.\nH2's predeclared gain gate uses the mean (>= 0.5), not the lower confidence endpoint. Conditional on one fixed dataset.")

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    fig.subplots_adjust(left=.20, right=.97, bottom=.24, top=.78)
    fig.suptitle("Behavioral admission and descriptive useful edits", x=.055, ha="left", fontsize=16, fontweight="bold")
    records = data["admission"]
    y = np.arange(len(records))
    useful = np.array([r["useful_and_admitted"] for r in records])
    other = np.array([r["admitted"]-r["useful_and_admitted"] for r in records])
    rejected = np.array([r["n"]-r["admitted"] for r in records])
    ax.barh(y, useful, color=COLORS["teal"], label="Admitted + gain >= 0.5", height=.65)
    ax.barh(y, other, left=useful, color=COLORS["gold"], hatch="//", label="Other admitted", height=.65)
    ax.barh(y, rejected, left=useful+other, color=COLORS["gray"], label="Rejected", height=.65)
    for i, record in enumerate(records):
        ax.text(record["n"]+.22, i, f"{record['admitted']}/{record['n']} admitted", va="center", fontsize=10)
    ax.set(yticks=y, yticklabels=[LABELS[r["arm"]] for r in records],
           xlim=(0, len(pairs)+4.8), xlabel="Programs (20 proposals per arm)", xticks=np.arange(0, len(pairs)+1, 5))
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(.43, 1.02), ncol=3, frameon=False, fontsize=10)
    finish_figure(fig, out, "admission", "Admission: exact rational score-drift bound <= 4 on [-1,1]^6 plus structural checks.\nThe per-program gain threshold is descriptive; rejected bounds can be conservative and are not counterexamples.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(left=.12, right=.97, bottom=.22, top=.80, wspace=.36)
    fig.suptitle("RTS adapter study: validation selection and held-out proxy utility", x=.055,
                 ha="left", fontsize=16, fontweight="bold")
    validation = data["selection"]["validation_rows"]
    labels = ["Baseline" if r["delta"] == 0 else f"Readout +{r['delta']:g}" for r in validation]
    for i, row in enumerate(validation):
        selected = row["id"] == data["selection"]["selected_id"]
        axes[0].scatter(row["mean_utility"], i, s=80 if selected else 45,
                        marker="D" if selected else "o", color=COLORS["teal"] if selected else COLORS["muted"],
                        facecolors="none" if not row["eligible"] else None, zorder=3)
        axes[0].annotate(f"{row['losses']}/{row['n']} losses", (row["mean_utility"], i),
                         xytext=(8, -4), textcoords="offset points", fontsize=9)
    axes[0].set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel="Mean validation proxy utility",
                title="Validation only · 8 world seeds")
    axes[0].invert_yaxis()
    means = [r["mean_utility"] for r in validation]
    spread = max(max(means)-min(means), .1)
    axes[0].set_xlim(min(means)-.15*spread, max(means)+.65*spread)
    axes[0].grid(axis="x")
    rows = data["rts_test"]
    x = np.arange(1, len(rows)+1)
    axes[1].axhline(0, color=COLORS["muted"], lw=1)
    effects = [r["utility_delta"] for r in rows]
    axes[1].scatter(x, effects, color=COLORS["teal"], s=40)
    h3 = data["H3"]["effect"]
    axes[1].axhline(h3["mean"], color=COLORS["gold"], lw=1.4, label="Mean paired effect")
    axes[1].axhspan(*h3["ci95"], color=COLORS["gold"], alpha=.15, label="95% bootstrap interval for mean")
    axes[1].set(xlabel="Test world index (seeds 63001–63020)", ylabel="Candidate minus baseline utility",
                title=f"Frozen choice · {data['selection']['selected_id']}", xticks=[1, 5, 10, 15, 20])
    if all(effect == 0 for effect in effects) and h3["ci95"] == [0.0, 0.0]:
        axes[1].set_ylim(-.05, .05)
        axes[1].set_yticks([-.04, -.02, 0, .02, .04])
        axes[1].text(.5, .82, f"All {len(effects)} paired effects = 0\nMean 0; 95% interval [0, 0]",
                     transform=axes[1].transAxes, ha="center", va="center", fontsize=10)
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, fontsize=9, loc="best")
    finish_figure(fig, out, "rts-study", "Diamond = validation-selected policy; open marker = failed observed-loss guard. No test-based reselection.\nUtility includes a 0.02 bonus per recruit. Games may finish before the maximum 420 seconds; one mission and frozen opponent.")


def markdown(data: dict[str, Any], protocol: dict[str, Any], study: Path, out: Path, audit: Path) -> str:
    pairs, admissions, test = data["pairs"], data["admission"], data["rts_test"]
    h1, h2, h3 = data["H1"], data["H2"], data["H3"]
    n = len(pairs)
    mean = lambda key: float(np.mean([r[key] for r in pairs]))
    test_mean = lambda key: float(np.mean([r[key] for r in test]))
    branch_text = []
    for branch in ("baseline", "candidate"):
        times = [r[f"{branch}_elapsed"] for r in test]
        branch_text.append(f"| {branch.title()} | {test_mean(f'{branch}_utility'):.4f} | {test_mean(f'{branch}_town_component'):.4f} | {test_mean(f'{branch}_recruitment_component'):.4f} | {test_mean(f'{branch}_recruits'):.2f} | {sum(t < 270 for t in times)}/{len(test)} | {min(times):.1f}–{max(times):.1f} s |")
    admission_table = []
    for row in admissions:
        among = "undefined (none admitted)" if row["rate_among_admitted"] is None else f"{100*row['rate_among_admitted']:.0f}%"
        admission_table.append(f"| {LABELS[row['arm']]} | {row['n']} | {row['admitted']} | {row['useful_descriptive']} | {row['useful_and_admitted']} | {100*row['rate_all_proposals']:.0f}% | {among} |")
    validation_table = "\n".join(f"| {r['id']} | {r['delta']:g} | {r['mean_utility']:.4f} | {r['losses']}/{r['n']} | {'yes' if r['eligible'] else 'no'} |" for r in data["selection"]["validation_rows"])
    split_table = "\n".join(f"| {r['split']} | {r['world_seeds']} | {r['arms']} | {r['paired_records']}/{r['planned_records']} | {r['controls']} | {r['failures']} |" for r in data["rts_splits"])
    exact_table = "\n".join(f"| {r['id']} | {r['requested_delta']:g} | `{r['actual_delta']}` |" for r in data["exact_policy"]["edits"])
    fixture = data["fixture_agreement"]
    training = data["training_test_mse"]
    summary = data["rts_summary"]["H3"]
    selection_note = (
        "The fixed selection rule retained the unchanged baseline. The test effect of zero is therefore "
        "a baseline consistency result; it is not an independent trial of a "
        "nonzero recruitment edit. H3 is unsupported. Because the selected "
        "policy is also the no-op control, test contains two distinct arms, "
        "with no duplicate no-op runs."
        if data["selection"]["baseline_selected"] else
        "The chosen nonzero edit was evaluated without reselection on the test worlds."
    )
    workspace_note = (
        "The workspace differs from the frozen execution tree in "
        + ", ".join(f"`{row['path']}`" for row in data["workspace_source_differences"])
        + ". Exact old/new hashes are recorded in `analysis.json`. These later "
        "changes are not attributed to the completed study; its archived source "
        "bytes remain authoritative."
        if data["workspace_source_differences"] else
        "The workspace execution sources also match this frozen snapshot."
    )
    return f"""# Weight-space programming: frozen study results

The framework's generic tensor programs, explicit adapters, and proved ReLU semantics
support a reproducible separation between proposing an edit, admitting its contract,
and measuring its effects. This study completed all **40 independently initialized
regression networks, {n} disjoint network pairs, and {len(test)} held-out RTS worlds**.
No failed planned unit was excluded. The independent evidence audit passed.

| Hypothesis and predeclared endpoint | Mean effect [95% paired bootstrap interval] | Outcome under the fixed rule |
| --- | --- | --- |
| H1: raw-minus-aligned transfer error, normalized by requested shift | {interval(h1['effect'])} | {h1['status']} |
| H2: target-only minus retention-aware collateral RMSE | {interval(h2['effect'])} | {h2['status']} |
| H2 target-gain gate: retention-aware mean target gain / requested shift | {interval(h2['target_gain_fraction'])} | Mean must be at least 0.5 |
| H3: frozen RTS choice minus baseline proxy utility | {interval(h3['effect'])} | {h3['status']} |

These are **three exploratory hypotheses with unadjusted intervals**. Each interval
uses 10,000 percentile bootstrap resamples with seed 736021, resampling whole paired
units. H1/H2 inference is conditional on one fixed shared dataset. H3 concerns one
mission and authored opponent. A supported rule is local evidence, not a universal
property of weight-space programming. No provider calls were made.

## The general network study

Forty 6→16→3 ReLU regression networks were trained independently for 350 full-batch
Adam epochs on a fixed nonlinear analytic teacher. All optimization used CPU float64,
one Torch thread, and the frozen learning rate 0.015. Disjoint adjacent seeds
71001–71040 form {n} source/target pairs; a network appears in only one pair.
The shared Uniform[-1,1]^6 partition contains 2,048 training, 256 compilation,
512 validation, and 2,048 test examples. The test target region contains
{data['test_target_examples']} examples; its complement contains {data['test_retention_examples']}.
Mean base-network test MSE was {training['mean']:.5f}
(range {training['min']:.5f}–{training['max']:.5f}).

The explicit specification requested an output-0 increase of 0.75 where x0 > 0.35,
with retention outside that region. A frozen-hidden readout least-squares compiler
used ridge 0.01, with retention weight either zero or eight. The same settings were
used for every network. Validation/test examples were never optimization inputs;
there was no synthetic model or hyperparameter selection.

![All planned generic pairs and the retention tradeoff](generic-effects.png)

**H1: alignment.** Raw and incoming-weight-aligned transfers were applied to the
independently trained target network. Their errors compare the transferred output-0
change with the source retention-aware edit's realized change, not with perfect
fulfillment of the requested 0.75. Mean normalized error was {mean('raw_error'):.4f}
for raw transfer and {mean('aligned_error'):.4f} for aligned transfer.
Alignment reduced error in {h1['effect']['positive_units']}/{n} pairs.
Its correspondence is heuristic: independently trained hidden representations
need not be related by a function-preserving symmetry. Every transfer is a new
proposal and requires its own admission.

**H2: retention.** Mean collateral RMSE over all three output coordinates fell
from {mean('target_only_retention_rmse'):.4f} to {mean('retention_rmse'):.4f}.
Mean target gain changed from {mean('target_only_gain'):.4f} to
{mean('retention_gain'):.4f} of the request. The fixed rule tests a positive collateral
reduction interval and a mean gain of at least 0.5; it does not require the gain
interval's lower endpoint to exceed 0.5. Retention is a soft penalty and does not
establish zero change on all unobserved inputs.

![Predeclared effect estimates with uncertainty](generic-estimates.png)

**Controls.** Known permutation/positive power-of-two gauges preserved the realized
edit to maximum normalized floating RMSE {max(r['known_transport_error'] for r in pairs):.3g}.
Inferred recovery of those known gauges had maximum error
{max(r['inferred_known_error'] for r in pairs):.3g}. The raw-coordinate permuted
negative control had mean error {mean('raw_permuted_error'):.4f}.
Exact rational evaluations agreed at five saved test points per pair; those finite
checks complement the general Lean symmetry theorem and are not themselves a
universal proof. All positive controls passed.

## Admission is a separate outcome

Every proposal was checked against a score-drift budget of four per output over the
entire box [-1,1]^6, with readout-only writes and the structural restrictions in the
saved contract. H1/H2 retain all planned proposals, including rejected proposals,
in isolated numerical evaluation. A rejection can reflect a loose sufficient bound;
it is not evidence that the actual drift exceeds four.

For the following **descriptive** table, a useful edit means its held-out mean
output-0 target gain is at least half the request. This per-program threshold is
an analysis aid, not a new preregistered endpoint or an overall task-success metric.
Transfer usefulness here is measured on its target network.

| Arm | Proposed | Admitted | Gain ≥ 0.5 | Both | Both / all proposals | Both / admitted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(admission_table)}

![Admission counts over all proposed programs](admission.png)

The complete per-program values, exact certificate references, and descriptive
flags are retained in [synthetic-arms.csv](synthetic-arms.csv) and
[analysis.json](analysis.json). These denominators prevent successful numerical
edits from being silently presented as deployed, certified edits.

## RTS as one downstream adapter

This adapter imports the actual authored, untrained 32→38→1 scorer with 1,293
parameters. The recruitment candidates edit only W2[0,19], attached to the hidden
feature ReLU(x24). The four requested increments were 0.5, 1, 2, and 4.
All four passed the global [0,1]^32 score bound and a zero-drift contract on x24=0.
The real-valued increments use the exact difference of serialized binary64 endpoints:

| Candidate | Requested increment | Actual rational increment |
| --- | ---: | --- |
{exact_table}

The import check evaluated {fixture['scored_examples']} scored examples in
{fixture['fixture_count']} development/validation fixtures. Sequential Python
accumulation matched JavaScript scores exactly, with no choice mismatch. NumPy's
maximum score discrepancy was {fixture['numpy_max_absolute_error']:.3g}, with
{len(fixture['numpy_argmax_mismatches'])} separately recorded argmax mismatches.
This tests the specific adapter's numerical agreement; it is not an IEEE proof.

| Split | World seeds | Evaluated arms including controls | Completed / planned paired records | Control records | Failed records |
| --- | ---: | ---: | ---: | ---: | ---: |
{split_table}

Repeated baseline branches across candidate arms share the same world seed;
they do not increase the independent sample size. Recorded no-op and
shared-output-bias comparisons reported no first changed choice, matching terminal
state hashes, and equal utility in every split.
The bias control tests this scorer's argmax policy; the separate living-world
stochastic softmax policy has different semantics.

Validation chose **{data['selection']['selected_id']}** using the highest mean utility
among candidates with no more observed losses than baseline; ties prefer the
smallest increment, and baseline was eligible. The selected bundle was written
before test execution. Test evaluated only that choice and the two controls.

{selection_note}

| Validation arm | Requested increment | Mean utility | Losses | Eligible |
| --- | ---: | ---: | ---: | --- |
{validation_table}

![RTS validation selection and selected test effects](rts-study.png)

On {len(test)} held-out world seeds, selected utility changed by {interval(h3['effect'])}.
The selected policy had {summary['candidate_losses']} losses and {summary['candidate_wins']} wins;
baseline had {summary['baseline_losses']} losses and {summary['baseline_wins']} wins.
The observed-loss guard is not a statistical noninferiority test.

Utility is net opponent Town Center damage fraction minus net own Town Center
damage fraction, plus 0.02 per completed recruited soldier. The decomposition is:

| Branch | Mean utility | Net town-damage component | Recruitment bonus | Mean recruits | Finished before 270 s | Elapsed range |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(branch_text)}

Mean paired town-damage contribution changed by {test_mean('town_component_delta'):.4f};
the recruitment contribution changed by {test_mean('recruitment_component_delta'):.4f}.
Games stop naturally when terminal; 420 seconds is a maximum, not a guaranteed
observation window. Rival recruitment begins at 270 seconds and raid eligibility
at 330 seconds, so early terminal games do not test those later behaviors.
This declared proxy and mission cannot establish general RTS strategic improvement.

The independent auditor reconstructed utility from the retained raw HP, recruitment,
and outcome fields and checked the 1,800-point Town Center maximum against frozen
initialization sources and available export snapshots. Raw trial records retain
checkpoint hashes but omit the initial checkpoint bodies and most final checkpoint
bodies. Those omitted states and their hashes cannot be independently regenerated
from the retained raw records alone. This is a provenance limit, distinct from the
checked arithmetic, program identities, and recorded control comparisons.
Successful order histories are retained, but complete decision histories including
waits and rejected commands are not. The auditor checks and links first-change
records; it does not independently replay those missing decisions or recover raw
production metrics from absent checkpoint bodies.

## Formal guarantees and remaining limits

The [Lean project]({relative(ROOT / 'formal/weight-space/README.md', out)}) proves
hidden permutation and positive-rescaling invariance, transported edits, the exact
readout effect, a box bound for edits to every layer, finite drift composition, and
strict argmax preservation under sufficient margins. Its actual-policy specialization
embeds every exported coefficient and checks the four recruitment changes, their
global score bounds, and exact preservation on the protected region.

These are theorems about real-valued networks with exact rational lifts of stored
weights. The Python checker, exporter, PyTorch execution, JavaScript execution,
hardware rounding, and task reward are outside the proved boundary. Caller-supplied
architectures can use generic tensor programs and execution adapters; arbitrary
architectures do not inherit the ReLU compiler or proof backend. JSON copying,
dense alignment, and rational arithmetic are reference-scale implementations.
No universal natural-language-to-weights compiler, LLM editing benchmark, or new
Astra proposal-quality result is claimed.

## Reproduce and inspect

The protocol and executable study sources were frozen at `{data['frozen_at']}`.
This is a local pre-result snapshot, not an external preregistration or a trusted
timestamp service. The source/protocol verifier still matches the archived bytes.
Reporting and independent audit scripts are separate analysis artifacts; their
hashes are retained with their results and they do not train, select, or simulate.

{workspace_note}

- Protocol SHA-256: `{data['protocol_sha256']}`
- Freeze receipt identity: `{data['freeze_sha256']}`
- [Frozen protocol and sources]({relative(study / 'freeze/receipt.json', out)})
- [Synthetic raw results]({relative(study / 'synthetic/results.json', out)}) and [manifest]({relative(study / 'synthetic/receipt.json', out)})
- [Validation selection]({relative(study / 'rts-selection/selection.json', out)}) and [held-out RTS results]({relative(study / 'rts-test/results.json', out)})
- [Independent audit]({relative(audit, out)})
- [Report data and calculation checks](analysis.json), [artifact receipt](receipt.json), and [stage logs]({relative(study / 'logs', out)})

From the repository root, choose unused output paths:

```powershell
pwsh -NoLogo -NoProfile -File scripts/run_weight_space_study.ps1 -Out artifacts/weight-space/my-study
.venv/Scripts/python.exe scripts/audit_weight_space_study.py --study artifacts/weight-space/my-study --out artifacts/weight-space/my-study-audit.json
.venv/Scripts/python.exe scripts/report_weight_space.py --study artifacts/weight-space/my-study --audit artifacts/weight-space/my-study-audit.json --out artifacts/weight-space/my-report
pwsh -NoLogo -NoProfile -File scripts/check_weight_space_proofs.ps1 -SkipCache
```

The [formal instructions]({relative(ROOT / 'formal/weight-space/README.md', out)})
include regenerating the actual-policy instance from the selected study's exported
model before a new proof check. Each run retains all planned arms, control outcomes,
raw models/programs, admissions, failures, exact command arguments, and exit logs.
The plots are available as PNG and SVG; CSV files retain every displayed unit.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    study, audit, out = args.study.resolve(), args.audit.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Report output already exists: {out}")
    data, protocol = load_evidence(study, audit)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "analysis.json", data)
    csv_file(out / "synthetic-pairs.csv", data["pairs"])
    csv_file(out / "synthetic-arms.csv", data["arms"])
    csv_file(out / "admission.csv", data["admission"])
    csv_file(out / "rts-validation.csv", data["selection"]["validation_rows"])
    csv_file(out / "rts-test.csv", data["rts_test"])
    figures(data, out)
    with (out / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown(data, protocol, study, out, audit))
    write_json(out / "receipt.json", {
        "schema": "wsp-report-receipt/1", "created_at": utc_now(), "environment": environment(),
        "script_sha256": file_hash(__file__), "audit_script_sha256": file_hash(ROOT / "scripts/audit_weight_space_study.py"),
        "audit_sha256": file_hash(audit), "study_root": str(study),
        "freeze_sha256": data["freeze_sha256"], "source_files_sha256": data["source_files_sha256"],
        "workspace_source_differences": data["workspace_source_differences"],
        "analysis_runtime_sources_sha256": {
            path.relative_to(ROOT).as_posix(): file_hash(path)
            for path in sorted((ROOT / "rlcd/weight_space").glob("*.py"))},
        "artifacts": {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "scope": "Post-freeze reporting of retained evidence; no new model training, selection, or simulations.",
    })
    print(json.dumps({"report": str(out / "report.md"), "H1": data["H1"], "H2": data["H2"],
                      "H3": data["H3"], "admission": data["admission"]}, indent=2))


if __name__ == "__main__":
    main()
