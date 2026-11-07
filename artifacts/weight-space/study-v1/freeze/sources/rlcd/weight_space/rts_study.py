"""One downstream case study; the generic edit engine has no dependency on this."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import from_rts, to_rts
from .contracts import Contract, check_contract
from .evidence import (bootstrap_mean, file_hash, read_json, utc_now,
                       verify_freeze, verify_manifest, write_json)
from .model import InputBox, Network, digest
from .program import difference


def _verify_evaluator_sources(receipt: dict[str, Any], frozen: dict[str, Any]) -> None:
    expected = {name: sha for name, sha in frozen["source_sha256"].items()
                if name in ("scripts/weight-space-empires.ts", "package-lock.json")
                or name.startswith("game/src/empires/") and
                (name.endswith(".ts") or name.startswith("game/src/empires/weights/") and name.endswith(".json"))}
    if receipt.get("source_sha256") != expected or receipt.get("protocol_sha256") != frozen["protocol_sha256"]:
        raise ValueError("RTS evidence was not generated with the frozen evaluator sources and protocol")


def _verify_results_artifacts(results_path: str | Path, results: dict[str, Any], frozen: dict[str, Any]) -> None:
    path = Path(results_path)
    receipt = read_json(path.parent / "receipt.json")
    _verify_evaluator_sources(results, frozen)
    _verify_evaluator_sources(receipt, frozen)
    verify_manifest(path.parent, receipt["artifacts"])
    if receipt["artifacts"].get(path.name) != file_hash(path):
        raise ValueError("Results do not match the recorded evaluation manifest")
    for run in results["runs"]:
        if receipt["artifacts"].get(run["raw_result_file"]) != run["raw_result_sha256"]:
            raise ValueError("A paired trial differs from its recorded artifact hash")


def _verify_candidate_bundle(programs_path: str | Path, frozen: dict[str, Any]) -> None:
    path = Path(programs_path)
    receipt = read_json(path.parent / "receipt.json")
    if (receipt.get("schema") != "wsp-rts-candidates/1"
            or receipt.get("sha256") != digest({k: v for k, v in receipt.items() if k != "sha256"})
            or receipt.get("freeze_sha256") != frozen["sha256"]
            or receipt.get("protocol_sha256") != frozen["protocol_sha256"]
            or receipt.get("programs_sha256") != file_hash(path)):
        raise ValueError("Evaluated candidate bundle differs from its frozen admission evidence")
    verify_manifest(path.parent, receipt["artifacts"])
    if receipt["artifacts"].get(path.name) != file_hash(path):
        raise ValueError("Candidate bundle is absent from the admission manifest")


def fixture_agreement(model: Network, fixtures: dict[str, Any]) -> dict[str, Any]:
    sequential_error, vector_error, examples = 0.0, 0.0, 0
    sequential_mismatches, vector_mismatches = [], []
    for fixture in fixtures["fixtures"]:
        if fixture["split"] not in ("development", "validation"):
            raise ValueError("Test fixtures must not be exported before selection")
        entries = fixture["candidates"]
        if not entries:
            continue
        inputs = np.array([entry["input"] for entry in entries])
        if not InputBox(np.zeros(32), np.ones(32)).contains(inputs).all():
            raise ValueError("Actual feature values exceed the certified domain")
        vector = model(inputs)[:, 0]
        sequential = []
        # Preserve the actual JS forwardPolicy accumulation order explicitly.
        for x, entry in zip(inputs, entries):
            score = float(model.b2[0])
            for h in range(model.shape[1]):
                activation = float(model.b1[h])
                for f in range(model.shape[0]):
                    activation += float(model.w1[h, f])*float(x[f])
                score += float(model.w2[0, h])*max(0.0, activation)
            sequential.append(score)
            sequential_error = max(sequential_error, abs(score-entry["score"]))
        vector_error = max(vector_error, float(np.max(np.abs(vector-np.array([e["score"] for e in entries])))))
        sequential_winner = min(range(len(entries)), key=lambda i: (-sequential[i], entries[i]["id"]))
        vector_winner = min(range(len(entries)), key=lambda i: (-vector[i], entries[i]["id"]))
        if sequential_winner != 0:
            sequential_mismatches.append(fixture["id"])
        if vector_winner != 0:
            vector_mismatches.append(fixture["id"])
        examples += len(entries)
    return {"fixture_count": len(fixtures["fixtures"]), "scored_examples": examples,
            "sequential_max_absolute_error": sequential_error, "numpy_max_absolute_error": vector_error,
            "sequential_argmax_mismatches": sequential_mismatches, "numpy_argmax_mismatches": vector_mismatches,
            "passed": examples > 0 and sequential_error == 0 and not sequential_mismatches and vector_error < 1e-10,
            "scope": "pass requires exact sequential scores/choices and close NumPy scores; NumPy argmax mismatches are reported separately; no IEEE proof"}


def make_candidates(export_dir: str | Path, protocol_path: str | Path,
                    freeze_path: str | Path, out: str | Path) -> dict[str, Any]:
    freeze = verify_freeze(protocol_path, freeze_path)
    protocol = read_json(protocol_path)
    export_dir, out = Path(export_dir), Path(out)
    template, fixtures = read_json(export_dir / "model.json"), read_json(export_dir / "fixtures.json")
    export_receipt = read_json(export_dir / "receipt.json")
    _verify_evaluator_sources(export_receipt, freeze)
    verify_manifest(export_dir, export_receipt["artifacts"])
    if (fixtures["protocol_sha256"] != freeze["protocol_sha256"]
            or fixtures["model_sha256"] != file_hash(export_dir / "model.json")):
        raise ValueError("Export identities differ from frozen protocol or actual model")
    model = from_rts(template)
    agreement = fixture_agreement(model, fixtures)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "fixture-agreement.json", agreement)
    if not agreement["passed"]:
        raise ValueError("Actual TS fixture numerical agreement failed; evidence retained")
    h, feature = template["hiddenNames"].index("army_train_bias"), template["featureNames"].index("train_army")
    expected = np.zeros(32)
    expected[feature] = 1
    if model.b1[h] != 0 or not np.array_equal(model.w1[h], expected):
        raise ValueError("The planned recruitment edit no longer has its declared feature semantics")
    global_box = InputBox(np.zeros(32), np.ones(32))
    protected_upper = np.ones(32)
    protected_upper[feature] = 0
    global_contract = Contract(global_box, (protocol["rts"]["maximum_certified_score_drift"],),
                               ("w2",), max_edits=1)
    protected_contract = Contract(InputBox(np.zeros(32), protected_upper), (0.,), ("w2",), max_edits=1)
    candidates, evidence = [], []
    exact = {"schema": "wsp-rational-policy/1", "base_sha256": model.sha256,
             "source_rts_file_sha256": file_hash(export_dir / "model.json"),
             "arithmetic": "Fraction.from_float: exact binary64 rational value",
             "w1": [[str(Fraction.from_float(float(v))) for v in row] for row in model.w1],
             "b1": [str(Fraction.from_float(float(v))) for v in model.b1],
             "w2": [str(Fraction.from_float(float(v))) for v in model.w2[0]],
             "b2": str(Fraction.from_float(float(model.b2[0]))),
             "hidden_index": h, "feature_index": feature, "edits": []}
    for delta in protocol["rts"]["candidate_recruitment_deltas"]:
        candidate_id = f"recruit_{str(float(delta)).replace('.', '_')}"
        w2 = model.w2.copy()
        w2[0, h] += delta
        candidate = model.replace(w2=w2)
        program = difference(model, candidate, f"Recruitment preference +{delta}; utility remains a hypothesis")
        general = check_contract(model, program, global_contract)
        protected = check_contract(model, program, protected_contract)
        if not general["accepted"] or not protected["accepted"]:
            write_json(out / f"{candidate_id}-rejection.json", {"global": general, "protected": protected})
            raise ValueError(f"Predeclared {candidate_id} failed exact admission; do not substitute or tune it")
        policy = to_rts(candidate, template, description=program.intent)
        candidates.append({"id": candidate_id, "model": policy,
                           "source": f"Literal readout edit; program {program.sha256}; isolated H3 candidate"})
        evidence.append({"id": candidate_id, "requested_delta": delta, "network_sha256": candidate.sha256,
                         "program_sha256": program.sha256, "global_bound": general["certificate"]["output_bounds"][0],
                         "protected_bound": protected["certificate"]["output_bounds"][0]})
        exact["edits"].append({"id": candidate_id, "requested_delta": delta,
                              "actual_delta": str(Fraction.from_float(float(w2[0, h]))
                                                  - Fraction.from_float(float(model.w2[0, h]))),
                              "after_w2": str(Fraction.from_float(float(w2[0, h]))),
                              "candidate_sha256": candidate.sha256})
        write_json(out / f"{candidate_id}-program.json", program.to_dict())
        write_json(out / f"{candidate_id}-admissions.json", {"global": general, "protected": protected})
    write_json(out / "base-network.json", model.to_dict())
    write_json(out / "global-contract.json", global_contract.to_dict())
    write_json(out / "protected-contract.json", protected_contract.to_dict())
    write_json(out / "programs.json", {"candidates": candidates})
    write_json(out / "exact-policy.json", exact)
    result = {"schema": "wsp-rts-candidates/1", "created_at": utc_now(),
              "freeze_sha256": freeze["sha256"], "protocol_sha256": freeze["protocol_sha256"],
              "base_sha256": model.sha256, "candidates": evidence,
              "programs_sha256": file_hash(out / "programs.json"), "fixture_agreement": agreement,
              "artifacts": {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    result["sha256"] = digest(result)
    write_json(out / "receipt.json", result)
    return result


def _validate_runs(results: dict[str, Any], protocol: dict[str, Any], split: str,
                   supplied_ids: list[str]) -> None:
    expected = {(identifier, seed) for identifier in ["control_noop", "control_b2_shift"]+supplied_ids
                for seed in protocol["rts"][f"{split}_seeds"]}
    actual = [(r["candidate_id"], r["seed"]) for r in results["runs"]]
    if (results["split"] != split or results["failures"] or not results["controls_pass"]
            or len(actual) != len(set(actual)) or set(actual) != expected
            or results["completed_runs"] != len(expected) or results["planned_runs"] != len(expected)
            or any(r["split"] != split or r["team"] != 0 for r in results["runs"])):
        raise ValueError("Incomplete, duplicated, leaking, or control-failing RTS evidence")
    for run in results["runs"]:
        if run["candidate_id"] in ("control_noop", "control_b2_shift"):
            if not run["behavior_equal"] or run["deltas"]["utility"] != 0:
                raise ValueError("A claimed negative control does not preserve behavior")
        expected_delta = run["candidate"]["utility"]-run["baseline"]["utility"]
        if run["deltas"]["utility"] != expected_delta:
            raise ValueError("RTS paired difference is inconsistent with raw branch metrics")
    # A candidate must not silently change the paired baseline or seed.
    for seed in protocol["rts"][f"{split}_seeds"]:
        rows = [r for r in results["runs"] if r["seed"] == seed]
        if any(r["baseline_checkpoint_sha256"] != rows[0]["baseline_checkpoint_sha256"] for r in rows):
            raise ValueError("Paired baselines differ across candidate arms")


def select_candidate(results_path: str | Path, programs_path: str | Path, protocol_path: str | Path,
                     freeze_path: str | Path, out: str | Path) -> dict[str, Any]:
    freeze = verify_freeze(protocol_path, freeze_path)
    results, programs, protocol = read_json(results_path), read_json(programs_path), read_json(protocol_path)
    _verify_results_artifacts(results_path, results, freeze)
    _verify_candidate_bundle(programs_path, freeze)
    ids = [p["id"] for p in programs["candidates"]]
    _validate_runs(results, protocol, "validation", ids)
    if (results["protocol_sha256"] != freeze["protocol_sha256"]
            or results["programs_sha256"] != file_hash(programs_path)):
        raise ValueError("Validation does not identify the exact frozen candidate bundle")
    delta_by_id = {f"recruit_{str(float(d)).replace('.', '_')}": d
                   for d in protocol["rts"]["candidate_recruitment_deltas"]}
    if set(ids) != set(delta_by_id):
        raise ValueError("Validation candidate set differs from the predeclared candidates")
    rows = []
    for identifier in ["control_noop"]+ids:
        runs = [r for r in results["runs"] if r["candidate_id"] == identifier]
        loss_count = sum(r["candidate"]["defeat"] for r in runs)
        base_loss_count = sum(r["baseline"]["defeat"] for r in runs)
        utility = sum((Fraction.from_float(float(r["candidate"]["utility"])) for r in runs), Fraction(0))/len(runs)
        rows.append({"id": identifier, "delta": delta_by_id.get(identifier, 0),
                     "mean_utility_exact": str(utility), "mean_utility": float(utility),
                     "losses": loss_count, "baseline_losses": base_loss_count,
                     "eligible": loss_count <= base_loss_count, "n": len(runs)})
    winner = min((r for r in rows if r["eligible"]),
                 key=lambda r: (-Fraction(r["mean_utility_exact"]), r["delta"]))
    selected = [p for p in programs["candidates"] if p["id"] == winner["id"]]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    bundle_hash = write_json(out / "programs.json", {"candidates": selected})
    payload = {"schema": "wsp-selection/1", "created_at": utc_now(),
               "protocol_sha256": freeze["protocol_sha256"], "freeze_sha256": freeze["sha256"],
               "validation_results_sha256": file_hash(results_path),
               "candidate_bundle_sha256": file_hash(programs_path), "test_programs_sha256": bundle_hash,
               "selected_id": winner["id"], "selected_delta": winner["delta"], "validation_rows": rows,
               "rule": protocol["rts"]["selection_rule"],
               "test_used_for_selection": False, "baseline_selected": not selected}
    receipt = payload | {"sha256": digest(payload)}
    write_json(out / "selection.json", receipt)
    return receipt


def summarize_test(results_path: str | Path, selection_path: str | Path,
                   protocol_path: str | Path, freeze_path: str | Path, out: str | Path) -> dict[str, Any]:
    freeze = verify_freeze(protocol_path, freeze_path)
    results, selection, protocol = read_json(results_path), read_json(selection_path), read_json(protocol_path)
    _verify_results_artifacts(results_path, results, freeze)
    if selection.get("sha256") != digest({k: v for k, v in selection.items() if k != "sha256"}):
        raise ValueError("Selection receipt is altered")
    supplied_ids = [] if selection["baseline_selected"] else [selection["selected_id"]]
    _validate_runs(results, protocol, "test", supplied_ids)
    if (results["programs_sha256"] != selection["test_programs_sha256"]
            or results["protocol_sha256"] != selection["protocol_sha256"]
            or results["protocol_sha256"] != file_hash(protocol_path)
            or selection["freeze_sha256"] != freeze["sha256"]
            or file_hash(Path(selection_path).parent / "programs.json") != selection["test_programs_sha256"]):
        raise ValueError("Test did not evaluate the frozen selection under its exact protocol")
    runs = [r for r in results["runs"] if r["candidate_id"] == selection["selected_id"]]
    config = protocol["statistical_policy"]
    effect = bootstrap_mean([r["deltas"]["utility"] for r in runs],
                            seed=config["bootstrap_seed"], resamples=config["bootstrap_resamples"])
    losses = sum(r["candidate"]["defeat"] for r in runs)
    base_losses = sum(r["baseline"]["defeat"] for r in runs)
    payload = {"schema": "wsp-rts-hypothesis/1", "created_at": utc_now(),
               "protocol_sha256": file_hash(protocol_path), "freeze_sha256": freeze["sha256"],
               "selection_sha256": selection["sha256"],
               "test_results_sha256": file_hash(results_path), "selected_id": selection["selected_id"],
               "H3": {"effect": effect, "candidate_losses": losses, "baseline_losses": base_losses,
                      "candidate_wins": sum(r["candidate"]["victory"] for r in runs),
                      "baseline_wins": sum(r["baseline"]["victory"] for r in runs),
                      "status": "supported" if effect["ci95"][0] > 0 and losses <= base_losses else "unsupported"},
               "candidate_mean_recruits": float(np.mean([r["candidate"]["recruits_completed"] for r in runs])),
               "baseline_mean_recruits": float(np.mean([r["baseline"]["recruits_completed"] for r in runs])),
               "candidate_finished_before_270": sum(r["candidate"]["elapsed_seconds"] < 270 for r in runs),
               "baseline_finished_before_270": sum(r["baseline"]["elapsed_seconds"] < 270 for r in runs),
               "controls_pass": results["controls_pass"],
               "scope": "single mission, one frozen opponent, declared recruitment-reward proxy, finite horizon"}
    write_json(out, payload)
    return payload
