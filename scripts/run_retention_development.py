"""Fixed, synthetic support-ownership development study; no heldout evaluation.

Preparation freezes the generator, roles, complete assignment schedule, actual
sources, runtime and report definitions before training. Run once under an
external 600-second process cap. Summarize reads only finite JSON records and
keeps all six assigned seed/task units, including partial or missing outcomes.
"""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()
PROCESS_CPU_STARTED = time.process_time()

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import io
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlcd.weight_space import features as feature_tools
from rlcd.weight_space.artifact import StructuralPolicy, TensorArtifact
from rlcd.weight_space.data_roles import DataRoleError, DataRoleRegistry, RegisteredBatch
from rlcd.weight_space.model import canonical_bytes, digest
from rlcd.weight_space.program import Program, apply_program
from rlcd.weight_space.research_tasks import (
    HEAD_BIAS, HEAD_WEIGHT, TextPreprocessing, behavior_metrics,
    decision_retained_arm, encode_inputs, evaluate_head, head_spec, make_head,
    probability_and_margin, train_encoder,
)
from scripts import run_prospective_support_routing as support

STUDY_ID = "wsp-support-ownership-cutoff-budget-development-20261109-v1"
GENERATOR_ID = "synthetic-support-ownership-families/1"
DRIVER_ID = "support-ownership-retention-development/1"
SEEDS = (11401, 11402, 11403)
LABELS = ("invoice_dispute", "account_access", "subscription_cancel")
OWNERS = ("Billing support", "Account support", "Subscription support")
TASKS = (
    {"task_id": "invoice-dispute-transfer", "target_label": 0,
     "specialist_owner": "Billing dispute specialist"},
    {"task_id": "account-access-transfer", "target_label": 1,
     "specialist_owner": "Account recovery specialist"},
)
PREFIXES = ("", "Please help. ", "I need assistance. ", "Support team. ")
SUFFIXES = ("", " Please review this.", " I need an update.", " Can you help me?")
CORES = (
    (
        "I dispute a charge on my invoice.",
        "The amount on my invoice is incorrect.",
        "My invoice lists a service I did not receive.",
        "There is an unexpected fee on my invoice.",
        "My invoice includes a charge that should be refunded.",
        "I was billed twice on the same invoice.",
        "The invoice total does not match the agreed price.",
        "I need someone to review a disputed invoice charge.",
        "The invoice contains a fee that I did not authorize.",
        "I dispute the balance on this invoice and need it checked.",
    ),
    (
        "I cannot sign in to my account.",
        "My account password does not let me log in.",
        "I am locked out of my account.",
        "My account login is failing.",
        "I need to reset the password for my account.",
        "The sign-in code for my account is not arriving.",
        "My account sign-in link has expired.",
        "I need help recovering access to my account.",
        "The account login screen rejects my password.",
        "I cannot access my account after a password reset.",
    ),
    (
        "I want to cancel my subscription.",
        "Please stop the renewal of my subscription.",
        "End my subscription at the next billing date.",
        "I do not want my subscription renewed.",
        "Cancel the recurring subscription plan.",
        "I want to turn off automatic subscription renewal.",
        "My subscription should end before it renews.",
        "Please close my subscription and stop future renewals.",
        "I am requesting cancellation of my subscription.",
        "Help me end my subscription before the next renewal.",
    ),
)
ROLE_FAMILIES = {"train": list(range(1, 7)), "development": [7],
                 "target": [8], "protected": [9], "calibration": [10]}
CUTOFFS = (1e-12, 1e-8, 1e-4, 1e-2, 1e-1)
PENALTIES = (1., 8., 64.)
SCORE_BUDGETS = tuple(64. * value for value in (0., 1e-4, 1e-3, 1e-2, 1e-1, 1.))
MARGINS = (0., .01, .1)
EXTERNAL_CAP_SECONDS = 600.
FINALIZATION_RESERVE_SECONDS = 10.


class StudyError(ValueError):
    """Controlled, non-data-bearing reason for a rejected protocol operation."""


class StudyBudgetExceeded(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure(caught: BaseException) -> dict:
    if isinstance(caught, DataRoleError):
        return {"type": type(caught).__name__, "reason": caught.decision["reason"],
                "data_role_decision": caught.decision}
    return {"type": type(caught).__name__,
            "reason": str(caught) if isinstance(caught, StudyError) else
                      "cooperative_wall_budget_exhausted" if isinstance(caught, StudyBudgetExceeded) else
                      "operation_failed_no_exception_payload_published"}


def configuration() -> dict:
    candidates = [{"candidate_id": "parent", "mode": "parent"},
                  {"candidate_id": "unrestricted", "mode": "none", "ridge": .01}]
    candidates += [{"candidate_id": f"hard-rtol-{cutoff:g}", "mode": "nullspace",
                    "ridge": .01, "nullspace_rtol": cutoff} for cutoff in CUTOFFS]
    candidates += [{"candidate_id": f"soft-penalty-{penalty:g}", "mode": "soft",
                    "ridge": .01, "retention_weight": penalty} for penalty in PENALTIES]
    candidates += [{"candidate_id": f"decision-penalty-{penalty:g}", "mode": "decision",
                    "penalty": penalty, "steps": 150} for penalty in PENALTIES]
    return {
        "schema": "wsp-retention-development-configuration/1", "study_id": STUDY_ID,
        "generator": {"id": GENERATOR_ID, "cores": [list(row) for row in CORES],
                      "prefixes": list(PREFIXES), "suffixes": list(SUFFIXES),
                      "order": "label, family, prefix index, suffix index",
                      "request_id": "study_id/label-{label}/f{family:02d}/p{prefix}/s{suffix}"},
        "label_names": list(LABELS), "original_owners": list(OWNERS),
        "tasks": list(TASKS), "seeds": list(SEEDS),
        "roles": ROLE_FAMILIES, "split_salt": None,
        "split_reason": "explicit_frozen_family_memberships",
        "training": {"vocabulary_width": 128, "hidden": 8, "epochs": 35,
                     "learning_rate": .015, "batch_size": 256, "threads": 4,
                     "optimizer": "Adam", "weight_decay": .001,
                     "epoch_selection": "earliest maximum development intent accuracy",
                     "parent_quality_exclusions": False, "replacement_seeds": False},
        "execution": {"native_lane": "Torch CPU float32", "threads": 1,
                      "deterministic_algorithms": True, "argmax": "lowest output index",
                      "external_whole_process_cap_seconds": EXTERNAL_CAP_SECONDS,
                      "cooperative_finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
                      "external_cap_enforced_by": "launching harness; not attested by this driver",
                      "separate_run_subtracts_observed_preparation_time": True,
                      "retry": False},
        "target_goal": {"output": 3, "margin": .5,
                        "equation": "D[:,3]=max(0,max(native_parent_scores[:,:3])+.5-native_parent_scores[:,3]); other outputs zero",
                        "arithmetic": "capture native float32, lift to float64 before goal arithmetic",
                        "tasks_edit_unchanged_parent_independently": True},
        "candidates": candidates,
        "diagnostic": {"equation": "R X = 0; T X = D[:,3:4]", "editable_width": 9,
                       "edit_bias": True, "cutoffs": list(CUTOFFS),
                       "exact_work_limit": 250000, "exact_rational_bit_limit": 4096,
                       "repeated_exact_analysis_is_charged": True,
                       "exact_edits_returned": False},
        "decision_fit": {"learning_rate": .025, "regularizer": .01,
                         "target_margin": .5, "protected_hinge_margin": .01,
                         "arithmetic": "existing helper computes target and protected hinges in native float32; common score-system diagnostic D is lifted float64",
                         "consumes_common_score_system_D": False},
        "structural": {"writable": [HEAD_WEIGHT, HEAD_BIAS], "editable_output": 3,
                       "max_edit_cells": 9, "max_specialist_parameter_abs": 256.,
                       "max_exact_change_abs": 256., "cast": "round", "native_dtype": "float32",
                       "clip_rejected_proposals": False,
                       "frozen_rows_parameter_cap": None},
        "gates": {"score_budgets": list(SCORE_BUDGETS), "native_margins": list(MARGINS),
                  "score_budget_interpretation": "exact value of stored binary64 threshold",
                  "protected_rows": "32 fitting anchors plus 32 non-target calibration rows",
                  "score_outputs": "all four outputs",
                  "native_score_drift_arithmetic": "lift captured float32 scores to float64 before subtraction",
                  "native_margin_arithmetic": "probability_and_margin: float32 top-two subtraction, then lift",
                  "owner_condition": "every protected owner equals its parent owner",
                  "strict_zero": "exact dyadic and native maxima both equal zero",
                  "refit_for_gates": False},
        "useful_success": {"minimum_target_gain": .10, "maximum_protected_error_increase": 0.,
                           "maximum_newly_harmful_protected_owner_flips": 0,
                           "scopes_required": ["fitting", "calibration"],
                           "structural_and_gate_pass_required": True,
                           "business_requirement_validated": False},
        "report": {"id": "support-ownership-retention-development-report/1",
                   "assignment_denominator": 6, "independent_training_seeds": 3,
                   "all_candidate_gate_policies_reported": True, "winner_selection": False,
                   "missing_success_credit": 0,
                   "missing_gain_harm_bounds": [-1., 1.],
                   "bounds_are_sampling_intervals": False,
                   "parent_fallback_is_separate": True,
                   "cost": "current-process wall and CPU stages, including repeated analysis; ending record I/O may be censored",
                   "memory_and_energy_measured": False},
        "evidence_role": "synthetic development and descriptive calibration",
        "heldout_evaluation": False, **support.FLAGS,
    }


def _design() -> list[dict]:
    config = configuration()
    grid = [{"gate_id": f"budget-{b}/margin-{m}", "score_budget": budget, "native_margin": margin}
            for b, budget in enumerate(SCORE_BUDGETS) for m, margin in enumerate(MARGINS)]
    return [{"assignment_id": f"seed-{seed}/{task['task_id']}", "seed": seed, "task": task,
             "candidates": [{"configuration": candidate,
                             "configuration_sha256": digest(candidate),
                             "gates": [dict(gate) for gate in grid]}
                            for candidate in config["candidates"]]}
            for seed in SEEDS for task in TASKS]


def _initial_manifest() -> dict:
    assignments = _design()
    for assignment in assignments:
        assignment.update(status="assigned", reason=None, parent_sha256=None,
                          common_diagnostics=[], parent_metrics=None)
        for candidate in assignment["candidates"]:
            candidate.update(status="assigned", reason=None, structural_eligible=None,
                             no_op=None, metrics=None, gate_observation=None)
            for gate in candidate["gates"]:
                gate.update(status="unreached", passes=None, useful=False, reasons=[])
    return {"schema": "wsp-retention-development-manifest/1", "study_id": STUDY_ID,
            "phase": "preparing", "created_utc": _now(), "configuration_sha256": digest(configuration()),
            "assigned_schedule_sha256": digest(_design()), "protocol_sha256": None,
            "assignments": assignments, "costs": [], "execution_requests": [],
            "preparation_observed_wall_seconds": None, "run_observed_wall_seconds": None,
            "reason": None, **support.FLAGS}


def _check_schedule(manifest: dict) -> None:
    try:
        actual = [{"assignment_id": assignment["assignment_id"], "seed": assignment["seed"],
                   "task": assignment["task"],
                   "candidates": [{"configuration": candidate["configuration"],
                                   "configuration_sha256": candidate["configuration_sha256"],
                                   "gates": [{key: gate[key] for key in
                                              ("gate_id", "score_budget", "native_margin")}
                                             for gate in candidate["gates"]]}
                                  for candidate in assignment["candidates"]]}
                  for assignment in manifest["assignments"]]
        if (manifest["schema"] != "wsp-retention-development-manifest/1"
                or manifest["study_id"] != STUDY_ID or canonical_bytes(actual) != canonical_bytes(_design())
                or manifest["assigned_schedule_sha256"] != digest(_design())
                or manifest["configuration_sha256"] != digest(configuration())):
            raise StudyError("frozen_schedule_or_configuration_changed")
    except (KeyError, TypeError):
        raise StudyError("frozen_schedule_or_configuration_missing") from None


@contextmanager
def _stage(run: Path, manifest: dict, name: str, *, deadline: float | None = None):
    if deadline is not None and time.perf_counter() >= deadline:
        raise StudyBudgetExceeded()
    wall, cpu = time.perf_counter(), time.process_time()
    record = {"stage_id": len(manifest["costs"]), "name": name, "status": "running",
              "started_utc": _now(), "wall_seconds": None, "cpu_seconds": None, "reason": None}
    manifest["costs"].append(record)
    support._manifest(run / "manifest.json", manifest)
    try:
        yield record
        record["status"] = "succeeded"
    except BaseException as caught:
        record.update(status="failed", reason=_failure(caught))
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                      ended_utc=_now())
        support._manifest(run / "manifest.json", manifest)


def _source_identity() -> dict:
    result = support._evaluator_identity()
    result["id"] = DRIVER_ID
    result["source_files"][Path(__file__).relative_to(ROOT).as_posix()] = support._sha(Path(__file__).read_bytes())
    result["training_runtime"] = {"threads": 4, "deterministic_algorithms": True,
                                  "seed_method": "torch.manual_seed(seed); minibatches seed+1"}
    result["environment"]["logical_cpu_count"] = __import__("os").cpu_count()
    return result


def _require_analyzer() -> None:
    if (not callable(getattr(feature_tools, "_analyze_feature_constraints", None))
            or not callable(getattr(feature_tools, "_editable_design", None))
            or getattr(feature_tools, "_EXACT_WORK_LIMIT", None) != 250000
            or getattr(feature_tools, "_EXACT_RATIONAL_BIT_LIMIT", None) != 4096):
        raise StudyError("bounded_feature_analyzer_source_required")


def _file_record(path: Path) -> dict:
    captured = path.read_bytes()
    return {"sha256": support._sha(captured), "bytes": len(captured)}


def _generate() -> tuple[dict, dict]:
    families, roles = {}, {role: [] for role in ROLE_FAMILIES}
    for label, cores in enumerate(CORES):
        for family, core in enumerate(cores, 1):
            family_id = f"{STUDY_ID}/label-{label}/f{family:02d}"
            rows = [{"request_id": f"{family_id}/p{p}/s{s}", "text": prefix + core + suffix, "label": label}
                    for p, prefix in enumerate(PREFIXES) for s, suffix in enumerate(SUFFIXES)]
            families[family_id] = {"label": label, "family": family, "authored_core": core, "rows": rows}
            for role, members in ROLE_FAMILIES.items():
                if family in members:
                    roles[role].extend(rows)
    return families, roles


def prepare(run_directory: Path, registry_path: Path, *, deadline: float | None = None) -> dict:
    run = Path(run_directory)
    run.mkdir(parents=True, exist_ok=False)
    manifest = _initial_manifest()
    support._manifest(run / "manifest.json", manifest)  # Complete schedule precedes generation.
    deadline = PROCESS_STARTED + EXTERNAL_CAP_SECONDS - FINALIZATION_RESERVE_SECONDS if deadline is None else deadline
    started = time.perf_counter()
    try:
        with _stage(run, manifest, "prepare/source-and-runtime", deadline=deadline):
            _require_analyzer()
            evaluator = _source_identity()
            config = configuration()
            registry = DataRoleRegistry(registry_path)
        with _stage(run, manifest, "prepare/generate-fixed-families", deadline=deadline):
            families, roles = _generate()
        records = {}
        with _stage(run, manifest, "prepare/register-all-development-roles", deadline=deadline):
            generator = registry.register_batch([config["generator"]], source_id=STUDY_ID + "/generator",
                                                family_id=STUDY_ID + "/shared-generator")
            family_records = {}
            for source, family in families.items():
                family_records[source] = registry.register_batch(
                    [{key: family[key] for key in ("label", "family", "authored_core")}],
                    source_id=source, family_id=source, parents=(generator,))
            batches = {}
            for role, rows in roles.items():
                parents = [source for source, family in families.items() if family["family"] in ROLE_FAMILIES[role]]
                envelope = {"source_id": STUDY_ID + "/raw/" + role,
                            "family_id": STUDY_ID + "/role/" + role, "parents": parents, "rows": rows}
                # Only canonical registry capture precedes every role's durable
                # development consumption. Caller normalization happens below.
                batches[role] = registry.register_batch(
                    rows, source_id=envelope["source_id"], family_id=envelope["family_id"],
                    parents=tuple(parents))
                path = run / "development" / (role + ".json")
                support._write(path, canonical_bytes(rows))
                records[role] = {"registration": batches[role].to_dict(),
                                 "file": _file_record(path), "row_count": len(rows)}
            manifest["development_roles"] = records
        with _stage(run, manifest, "prepare/consume-all-development-roles", deadline=deadline):
            for role, batch in batches.items():
                records[role]["consumption"] = registry.consume(batch, reason=f"{STUDY_ID}/development/{role}")
                support._manifest(run / "manifest.json", manifest)
            manifest["all_raw_roles_consumed"] = True
        with _stage(run, manifest, "prepare/validate-and-register-aliases-after-all-role-commits", deadline=deadline):
            for role, batch in batches.items():
                envelope = {"source_id": STUDY_ID + "/validated/" + role,
                            "family_id": STUDY_ID + "/role/" + role,
                            "parents": [batch.to_dict()["source_id"]], "rows": batch.materialize()}
                validated, aliases = support._register(registry, envelope, len(LABELS))
                records[role]["aliases"] = aliases
                records[role]["alias_consumption"] = registry.consume(
                    validated, reason=f"{STUDY_ID}/development-aliases/{role}")
                support._manifest(run / "manifest.json", manifest)
        with _stage(run, manifest, "prepare/freeze-protocol", deadline=deadline):
            protocol = {"schema": "wsp-retention-development-protocol/1", "configuration": config,
                        "assigned_schedule": _design(), "evaluator": evaluator,
                        "registry_id": registry.registry_id, "development_roles": records,
                        "generator_registration": generator.to_dict(),
                        "family_registrations": {source: batch.to_dict() for source, batch in family_records.items()},
                        "boundary": "Only registry canonical integrity capture/decoding precedes all raw-role development commits; caller validation, normalization, alias derivation and model exposures follow all commits",
                        "external_preregistration": False, **support.FLAGS}
            support._write_json(run / "protocol.json", protocol)
            manifest.update(phase="prepared", protocol_sha256=digest(protocol),
                            protocol_file=_file_record(run / "protocol.json"))
    except BaseException as caught:
        manifest.update(phase="preparation_failed", reason=_failure(caught))
        raise
    finally:
        manifest["preparation_observed_wall_seconds"] = time.perf_counter()-started
        manifest["preparation_process_observed_wall_seconds"] = time.perf_counter()-PROCESS_STARTED
        support._manifest(run / "manifest.json", manifest)
    return manifest


def _structural(base: TensorArtifact, program: Program, candidate: TensorArtifact) -> dict:
    # ponytail: reuse the generic policy; its all-tensor cap is deliberately
    # permissive because this study caps only the specialist endpoints.
    generic = StructuralPolicy((HEAD_WEIGHT, HEAD_BIAS), 9, np.finfo(np.float64).max, 256.).check(base, program)
    issues = list(generic["issues"])
    if apply_program(base, program).sha256 != candidate.sha256:
        issues.append("candidate-program-identity")
    if not base.compatible(candidate):
        issues.append("metadata-shape-or-encoder-identity")
    if any(not ((edit.tensor == HEAD_WEIGHT and len(edit.index) == 2 and edit.index[0] == 3)
                or (edit.tensor == HEAD_BIAS and edit.index == (3,))) for edit in program.edits):
        issues.append("specialist-only-write-set")
    tensors = candidate.tensors
    if (np.any(np.abs(tensors[HEAD_WEIGHT][3]) > 256.) or abs(float(tensors[HEAD_BIAS][3])) > 256.):
        issues.append("specialist-parameter-bound")
    with np.errstate(over="ignore", invalid="ignore"):
        if any(not np.array_equal(values, values.astype(np.float32).astype(np.float64))
               for values in tensors.values()):
            issues.append("native-float32-representation")
    payload = {"schema": "wsp-specialist-structural-admission/1",
               "structural_accepted": not issues, "issues": issues, "generic_receipt": generic,
               "program_sha256": program.sha256, "candidate_sha256": candidate.sha256,
               "specialist_parameter_cap": 256., "editable_output": 3,
               "encoder_and_preprocessing_sha256": {
                   key: base.metadata.get(key) for key in ("encoder_sha256", "preprocessing_sha256", "feature_id")},
               "behavioral_guarantee": False}
    return payload | {"sha256": digest(payload)}


def _ownership_metrics(parent: np.ndarray, candidate: np.ndarray, labels: np.ndarray, task: dict) -> dict:
    queue_map = np.arange(4, dtype=np.int64)
    before, after = parent.argmax(1), candidate.argmax(1)
    result = behavior_metrics(before, after, labels, task["target_label"], queue_map, 3)
    protected = labels != task["target_label"]
    result["newly_harmful_protected_owner_flips"] = int(
        np.count_nonzero(protected & (before == labels) & (after != labels)))
    result["protected_owner_flips"] = int(np.count_nonzero(protected & (after != before)))
    result["owners_by_output_code"] = list(OWNERS) + [task["specialist_owner"]]
    result["semantic_labels_unchanged"] = True
    return result


def _useful(metrics: dict | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    gain, harm = metrics.get("target_gain"), metrics.get("protected_harm")
    return (type(gain) in (int, float) and math.isfinite(gain) and gain >= .10
            and type(harm) in (int, float) and math.isfinite(harm) and harm <= 0.
            and type(metrics.get("newly_harmful_protected_owner_flips")) is int
            and metrics["newly_harmful_protected_owner_flips"] == 0
            and all(type(metrics.get(key)) is int and metrics[key] > 0
                    for key in ("target_rows", "protected_rows")))


def _gate_verdict(gate: dict, observation: dict, metrics: dict, structural_eligible: bool) -> dict:
    exact = Fraction(observation["maximum_exact_protected_score_drift"])
    native = observation["maximum_native_protected_score_drift"]
    margin = observation["minimum_native_protected_margin"]
    budget = gate["score_budget"]
    if (exact < 0 or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                         for value in (native, margin, budget, gate["native_margin"]))
            or type(observation["protected_owner_flips"]) is not int
            or observation["protected_owner_flips"] < 0):
        raise StudyError("invalid_gate_scalar_observation")
    reasons = []
    if not structural_eligible:
        reasons.append("structural_ineligible")
    # Exact Fraction comparison has no absolute or relative slack, including at
    # budget zero when native float32 arithmetic masks a nonzero exact residual.
    if exact > Fraction.from_float(float(budget)):
        reasons.append("exact_score_budget_exceeded")
    if native > budget:
        reasons.append("native_score_budget_exceeded")
    if margin < gate["native_margin"]:
        reasons.append("native_protected_margin_below_threshold")
    if observation["protected_owner_flips"] != 0:
        reasons.append("protected_owner_changed")
    passes = not reasons
    return {"status": "evaluated", "passes": passes,
            "useful": passes and _useful(metrics.get("fitting")) and _useful(metrics.get("calibration")),
            "reasons": reasons}


def _gate_observation(base: TensorArtifact, candidate: TensorArtifact, protected_features: np.ndarray,
                      parent_scores: np.ndarray, candidate_scores: np.ndarray) -> tuple[dict, dict]:
    receipt = feature_tools.certify_feature_retention(base, candidate, head_spec(base), protected_features)
    maximum = max(Fraction(value) for value in receipt["maximum_residuals_exact"])
    drift = np.abs(candidate_scores.astype(np.float64) - parent_scores.astype(np.float64))
    native_margins = probability_and_margin(candidate_scores)[1].astype(np.float64)
    observation = {
        "protected_rows": len(protected_features), "outputs_checked": 4,
        "maximum_exact_protected_score_drift": str(maximum),
        "maximum_native_protected_score_drift": float(drift.max()),
        "minimum_native_protected_margin": float(native_margins.min()),
        "protected_owner_flips": int(np.count_nonzero(parent_scores.argmax(1) != candidate_scores.argmax(1))),
        "exact_receipt_sha256": receipt["sha256"],
        "native_score_drift_arithmetic": "float32 captures lifted to float64 before difference",
        "native_margin_arithmetic": "float32 top-two subtraction, then lift to float64",
        "scope": "all outputs on fitting anchors and protected calibration descendants only",
    }
    return observation, receipt


def _metric_bounds(candidates: list[dict], scope: str, field: str) -> dict:
    observed = []
    for candidate in candidates:
        value = (candidate.get("metrics") or {}).get(scope, {}).get(field)
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
                raise StudyError("invalid_recorded_behavior_metric")
            observed.append(float(value))
    missing = 6-len(observed)
    return {"assigned_denominator": 6, "observed_assignments": len(observed), "missing_assignments": missing,
            "observed_conditional_mean": sum(observed)/len(observed) if observed else None,
            "all_assigned_mean_lower_bound": (sum(observed)-missing)/6,
            "all_assigned_mean_upper_bound": (sum(observed)+missing)/6,
            "bound_premise": "each missing gain/harm is in [-1,1]; no imputation or sampling interval"}


def summarize_manifest(manifest: dict) -> dict:
    """Array-free accounting; assigned denominators never depend on completion."""
    _check_schedule(manifest)
    candidate_summaries, frontier = [], []
    for index, config in enumerate(configuration()["candidates"]):
        candidates = [assignment["candidates"][index] for assignment in manifest["assignments"]]
        candidate_summaries.append({
            "configuration": config, "assigned_denominator": 6,
            "statuses": dict(Counter(candidate["status"] for candidate in candidates)),
            "structurally_eligible": sum(candidate["structural_eligible"] is True for candidate in candidates),
            "structurally_ineligible": sum(candidate["structural_eligible"] is False for candidate in candidates),
            "structural_outcome_missing": sum(candidate["structural_eligible"] is None for candidate in candidates),
            "realized_no_ops": sum(candidate["no_op"] is True for candidate in candidates),
            "fitting_target_gain": _metric_bounds(candidates, "fitting", "target_gain"),
            "calibration_target_gain": _metric_bounds(candidates, "calibration", "target_gain"),
            "fitting_protected_error_increase": _metric_bounds(candidates, "fitting", "protected_harm"),
            "calibration_protected_error_increase": _metric_bounds(candidates, "calibration", "protected_harm"),
        })
        for gate_index, gate_design in enumerate(_design()[0]["candidates"][index]["gates"]):
            computed, evaluated, passed, useful, ineligible = 0, 0, 0, 0, 0
            statuses = Counter()
            for candidate in candidates:
                gate = candidate["gates"][gate_index]
                statuses[gate["status"]] += 1
                if candidate["structural_eligible"] is False:
                    ineligible += 1
                if (candidate["status"] == "completed" and gate["status"] == "evaluated"
                        and candidate.get("gate_observation") is not None and candidate.get("metrics") is not None):
                    actual = _gate_verdict(gate_design, candidate["gate_observation"], candidate["metrics"],
                                           candidate["structural_eligible"] is True)
                    if any(gate.get(key) != actual[key] for key in ("passes", "useful", "reasons")):
                        raise StudyError("recorded_gate_contradicts_frozen_scalar_rule")
                    if candidate["structural_eligible"] is not True:
                        raise StudyError("completed_gate_without_structural_admission")
                    computed += 1
                    if manifest.get("source_unchanged_at_completion") is True:
                        evaluated += 1
                        passed += actual["passes"]
                        useful += actual["useful"]
            frontier.append({"configuration_id": config["candidate_id"], **gate_design,
                             "assigned_denominator": 6, "evaluated_assignments": evaluated,
                             "computed_gate_assignments": computed,
                             "structurally_ineligible_assignments": ineligible,
                             "source_unverified_computed_assignments": (
                                 computed if manifest.get("source_unchanged_at_completion") is None else 0),
                             "source_invalidated_computed_assignments": (
                                 computed if manifest.get("source_unchanged_at_completion") is False else 0),
                             "unknown_or_unfinished_gate_assignments": 6-computed-ineligible,
                             "gate_statuses": dict(statuses), "gate_passes": passed,
                             "gate_pass_fraction_all_assigned": passed/6,
                             "useful_successes": useful, "useful_fraction_all_assigned": useful/6,
                             "missing_outcome_useful_credit": 0,
                             "parent_fallback": {"proposed_for_nonpassing_or_missing": 6-passed,
                                                 "available_trained_parents": sum(
                                                     a.get("parent_sha256") is not None for a in manifest["assignments"]),
                                                 "change_relative_to_same_parent": 0.,
                                                 "ownership_transfer_gain": 0.,
                                                 "proposal_useful_credit": 0,
                                                 "actually_deployed": False}})
    exact_by_cutoff = []
    exact_work = 0
    for cutoff in CUTOFFS:
        consistency, minimum_norm = Counter(), Counter()
        for assignment in manifest["assignments"]:
            records = [item for item in assignment["common_diagnostics"] if item["nullspace_rtol"] == cutoff]
            exact = records[0]["analysis"]["exact"] if records else {}
            consistency[exact.get("consistency_status", "missing")] += 1
            minimum_norm[exact.get("minimum_norm_status", "missing")] += 1
            exact_work += exact.get("work_used", 0)
        exact_by_cutoff.append({"nullspace_rtol": cutoff, "assigned_denominator": 6,
                                "exact_consistency_statuses": dict(consistency),
                                "exact_minimum_norm_statuses": dict(minimum_norm)})
    for assignment in manifest["assignments"]:
        for candidate in assignment["candidates"]:
            exact_work += (candidate.get("fit_diagnostics") or {}).get("feature_constraints", {}).get("exact", {}).get("work_used", 0)
    costs = manifest["costs"]
    observed_costs = [record for record in costs if record["wall_seconds"] is not None]
    return {"schema": "wsp-retention-development-summary/1", "study_id": STUDY_ID,
            "manifest_content_sha256": digest(manifest), "protocol_sha256": manifest["protocol_sha256"],
            "phase": manifest["phase"], "assigned_seed_task_denominator": 6,
            "independent_training_seeds": 3, "paired_tasks_share_parent": True,
            "assignment_statuses": dict(Counter(a["status"] for a in manifest["assignments"])),
            "candidate_summaries": candidate_summaries, "gate_frontier": frontier,
            "common_exact_diagnostics_by_cutoff": exact_by_cutoff,
            "exact_work_observed_in_completed_diagnostics": exact_work,
            "cost": {"stage_statuses": dict(Counter(record["status"] for record in costs)),
                     "observed_stage_wall_seconds": sum(record["wall_seconds"] for record in observed_costs),
                     "observed_stage_cpu_seconds": sum(record["cpu_seconds"] for record in observed_costs),
                     "unfinished_stage_times_unknown": sum(record["wall_seconds"] is None for record in costs),
                     "preparation_observed_wall_seconds": manifest["preparation_observed_wall_seconds"],
                     "run_observed_wall_seconds": manifest["run_observed_wall_seconds"],
                     "includes_repeated_diagnostics_and_candidate_verification": True,
                     "last_persistence_and_abrupt_termination_may_be_censored": True,
                     "external_cap_execution_attested": False, "memory_measured": False, "energy_measured": False},
            "winner_selected": False, "business_impact_measured": False,
            "behavior_metrics_source_identity_status": (
                "verified" if manifest.get("source_unchanged_at_completion") is True else
                "invalidated" if manifest.get("source_unchanged_at_completion") is False else "unverified"),
            "frozen_evaluator_identity_verified_at_end": manifest.get("source_unchanged_at_completion") is True,
            "claims": "fixed synthetic development frontier; calibration is descriptive; exact score infeasibility is not decision infeasibility",
            **support.FLAGS}


def _consume_derived(registry: DataRoleRegistry, manifest: dict, key: str, rows: list[dict],
                     parents: tuple[RegisteredBatch, ...]) -> RegisteredBatch:
    batch = registry.register_batch(rows, source_id=STUDY_ID + "/derived/" + key,
                                    family_id=STUDY_ID + "/derived/" + key, parents=parents)
    consumed = registry.consume(batch, reason=STUDY_ID + "/derived-development/" + key)
    manifest.setdefault("derived_development", {})[key] = {
        "registration": batch.to_dict(), "consumption": consumed}
    return batch


def _reopen_development(run: Path, registry: DataRoleRegistry, protocol: dict, manifest: dict) -> tuple[dict, dict]:
    batches = {}
    if registry.registry_id != protocol["registry_id"]:
        raise StudyError("prepared_registry_identity_changed")
    # Reconstruct immutable captures without caller-level JSON materialization.
    # consume validates their complete bytes against the original shared registry.
    for role, record in protocol["development_roles"].items():
        path = run / "development" / (role + ".json")
        if _file_record(path) != record["file"]:
            raise StudyError("frozen_raw_role_file_changed")
        raw_manifest = dict(record["registration"])
        expected = raw_manifest.pop("manifest_sha256")
        batch = RegisteredBatch._from_bytes(path.read_bytes(), canonical_bytes(raw_manifest))
        if batch.manifest_sha256 != expected:
            raise StudyError("frozen_raw_role_registration_changed")
        batches[role] = batch
    manifest["execution_role_consumptions"] = {}
    for role, batch in batches.items():
        manifest["execution_role_consumptions"][role] = registry.consume(
            batch, reason=STUDY_ID + "/execution-development/" + role)
    support._manifest(run / "manifest.json", manifest)
    return {role: batch.materialize() for role, batch in batches.items()}, batches


def _export_parent(run: Path, seed: int, trained: dict, preprocessing: TextPreprocessing,
                   base: TensorArtifact) -> dict:
    directory = run / "parents" / f"seed-{seed}"
    support._write_json(directory / "head.json", base.to_dict())
    support._write_json(directory / "preprocessing.json", preprocessing.to_dict())
    stream = io.BytesIO()
    np.savez(stream, **trained["encoder"])
    support._write(directory / "encoder.npz", stream.getvalue())
    lineage = {
        "schema": "wsp-trained-lineage/1", "lineage_id": f"{STUDY_ID}/seed-{seed}",
        "head_sha256": base.sha256,
        **{key: base.metadata[key] for key in ("encoder_sha256", "preprocessing_sha256", "feature_id")},
        "files": {name: _file_record(directory / name) for name in support.PARENT_FILES[:-1]},
        "training": {key: trained[key] for key in (
            "initialization_sha256", "selected_parameters_sha256", "selected_epoch", "history", "ancestry")},
        "training_threads": 4, "execution_threads": 1, "synthetic": True,
    }
    support._write_json(directory / "lineage.json", lineage)
    parent = support.load_parent(directory)  # Only these four newly written safe files.
    if parent["head"].sha256 != base.sha256:
        raise StudyError("new_parent_roundtrip_identity_failed")
    return parent


def _train_parent(run: Path, manifest: dict, registry: DataRoleRegistry, rows: dict,
                  raw_batches: dict, seed: int, deadline: float) -> tuple[dict, dict]:
    key = f"seed-{seed}"
    with _stage(run, manifest, key + "/fit-train-only-preprocessing", deadline=deadline):
        preprocessing = TextPreprocessing.fit([row["text"] for row in rows["train"]], width=128)
        preprocessing_batch = _consume_derived(registry, manifest, key + "/preprocessing",
                                               [preprocessing.to_dict()], (raw_batches["train"],))
    with _stage(run, manifest, key + "/transform-and-consume-inputs", deadline=deadline):
        inputs, input_batches = {}, {}
        for role, role_rows in rows.items():
            inputs[role] = preprocessing.transform([row["text"] for row in role_rows])
            input_batches[role] = _consume_derived(
                registry, manifest, key + "/preprocessed/" + role,
                [{"request_id": row["request_id"], "values": values.tolist(), "dtype": "float32",
                  "preprocessing_sha256": digest(preprocessing.to_dict())}
                 for row, values in zip(role_rows, inputs[role])],
                (raw_batches[role], preprocessing_batch))
    with _stage(run, manifest, key + "/train-parent", deadline=deadline):
        parent_inputs = np.vstack((inputs["train"], inputs["development"]))
        labels = np.asarray([row["label"] for role in ("train", "development") for row in rows[role]], dtype=np.int64)
        n = len(rows["train"])
        trained = train_encoder(parent_inputs, labels, np.arange(n), np.arange(n, len(labels)),
                                seed=seed, hidden=8, epochs=35, learning_rate=.015, batch_size=256)
        base = make_head(trained, digest(preprocessing.to_dict()), list(LABELS), seed, "synthetic-support-ownership")
        model_batch = _consume_derived(
            registry, manifest, key + "/model",
            [{"head_sha256": base.sha256,
              **{name: base.metadata[name] for name in ("encoder_sha256", "preprocessing_sha256", "feature_id")},
              "selected_epoch": trained["selected_epoch"], "seed": seed}],
            (input_batches["train"], input_batches["development"], preprocessing_batch))
    with _stage(run, manifest, key + "/export-and-verify-new-parent", deadline=deadline):
        support._native_environment()  # train_encoder sets four; frozen execution uses one.
        parent = _export_parent(run, seed, trained, preprocessing, base)
        manifest.setdefault("parents", {})[key] = {
            "head_sha256": base.sha256, "identities": parent["identities"],
            "files": {name: _file_record(run / "parents" / key / name) for name in support.PARENT_FILES},
            "training": {name: trained[name] for name in (
                "selected_epoch", "history", "initialization_sha256", "selected_parameters_sha256")},
            "training_threads": 4, "execution_threads": 1}
    with _stage(run, manifest, key + "/encode-and-consume-development-features", deadline=deadline):
        features = {}
        for role, role_rows in rows.items():
            features[role] = encode_inputs(parent["encoder"], inputs[role])
            _consume_derived(
                registry, manifest, key + "/features/" + role,
                [{"request_id": row["request_id"], "features": values.tolist(),
                  "feature_id": base.metadata["feature_id"], "dtype": "float32"}
                 for row, values in zip(role_rows, features[role])],
                (input_batches[role], model_batch))
    return parent, features


def _fit(base: TensorArtifact, target: np.ndarray, protected: np.ndarray,
         desired: np.ndarray, config: dict) -> dict:
    if config["mode"] == "parent":
        return {"program": Program(base.sha256, (), "Unchanged parent development baseline"), "artifact": base,
                "diagnostics": {"solver": "unchanged-parent/1", "no_fit": True}, "retention_receipt": None}
    if config["mode"] == "decision":
        return decision_retained_arm(base, target, protected, 3,
                                     penalty=config["penalty"], steps=config["steps"])
    compiled = feature_tools.compile_feature_edit(
        base, head_spec(base), target, desired,
        None if config["mode"] == "none" else protected,
        retention=config["mode"], ridge=.01, edit_bias=True,
        retention_weight=config.get("retention_weight", 8.),
        nullspace_rtol=config.get("nullspace_rtol", 1e-12),
        native_dtypes={HEAD_WEIGHT: "float32", HEAD_BIAS: "float32"}, cast="round",
        intent="Fixed synthetic ownership transfer to output 3")
    return {"program": compiled.program, "artifact": apply_program(base, compiled.program),
            "diagnostics": compiled.diagnostics, "retention_receipt": compiled.retention_certificate,
            "structured_proposal": compiled.proposal.to_dict()}


def _export_candidate(directory: Path, base: TensorArtifact, proposal: dict, structural: dict) -> dict:
    values = {"program.json": proposal["program"].to_dict(), "head.json": proposal["artifact"].to_dict(),
              "fit.json": {"diagnostics": proposal["diagnostics"],
                           "fitting_retention_receipt": proposal.get("retention_receipt"),
                           "structured_proposal": proposal.get("structured_proposal")},
              "structural.json": structural}
    for name, value in values.items():
        support._write_json(directory / name, value)
    restored_program = Program.from_dict(support.read_json(directory / "program.json"))
    restored_head = TensorArtifact.from_dict(support.read_json(directory / "head.json"))
    if (apply_program(base, restored_program).sha256 != restored_head.sha256
            or restored_head.sha256 != proposal["artifact"].sha256):
        raise StudyError("candidate_roundtrip_identity_failed")
    return {name: _file_record(directory / name) for name in values}


def _run_assignment(run: Path, manifest: dict, assignment: dict, parent: dict, features: dict,
                    rows: dict, deadline: float) -> None:
    task, base = assignment["task"], parent["head"]
    target_label = task["target_label"]
    labels = {role: np.asarray([row["label"] for row in role_rows], dtype=np.int64)
              for role, role_rows in rows.items()}
    target = features["target"][labels["target"] == target_label]
    protected = features["protected"][labels["protected"] != target_label]
    fitting = np.vstack((target, protected))
    fitting_labels = np.r_[labels["target"][labels["target"] == target_label],
                           labels["protected"][labels["protected"] != target_label]]
    calibration = features["calibration"]
    calibration_labels = labels["calibration"]
    protected_features = np.vstack((protected, calibration[calibration_labels != target_label]))
    directory = run / "assignments" / assignment["assignment_id"]
    assignment.update(status="running", parent_sha256=base.sha256,
                      target_feature_sha256=digest(target.tolist()),
                      protected_feature_sha256=digest(protected.tolist()),
                      calibration_feature_sha256=digest(calibration.tolist()),
                      protected_gate_feature_sha256=digest(protected_features.tolist()))
    with _stage(run, manifest, assignment["assignment_id"] + "/native-parent-scores", deadline=deadline):
        parent_scores = {"fitting": evaluate_head(base, fitting, native=True),
                         "calibration": evaluate_head(base, calibration, native=True)}
        assignment["parent_metrics"] = {
            scope: _ownership_metrics(scores, scores, scope_labels, task)
            for scope, scores, scope_labels in (
                ("fitting", parent_scores["fitting"], fitting_labels),
                ("calibration", parent_scores["calibration"], calibration_labels))}
        native_target = parent_scores["fitting"][:len(target)].astype(np.float64)
        desired = np.zeros((len(target), 4), dtype=np.float64)
        desired[:, 3] = np.maximum(0., native_target[:, :3].max(1) + .5 - native_target[:, 3])
        assignment["desired_delta_sha256"] = digest(desired.tolist())
    for cutoff in CUTOFFS:
        with _stage(run, manifest, assignment["assignment_id"] + f"/common-diagnostic/{cutoff:g}", deadline=deadline):
            _, diagnostic = feature_tools._analyze_feature_constraints(
                feature_tools._editable_design(target, True),
                feature_tools._editable_design(protected, True), desired[:, 3:4],
                nullspace_rtol=cutoff, exact_work_limit=250000)
            assignment["common_diagnostics"].append({"nullspace_rtol": cutoff, "analysis": diagnostic})
    with _stage(run, manifest, assignment["assignment_id"] + "/export-and-verify-common-diagnostics", deadline=deadline):
        path = directory / "common-diagnostics.json"
        support._write_json(path, assignment["common_diagnostics"])
        if support.read_json(path) != assignment["common_diagnostics"]:
            raise StudyError("common_diagnostics_roundtrip_failed")
        assignment["common_diagnostics_file"] = _file_record(path)
    parent_protected_scores = np.vstack((parent_scores["fitting"][fitting_labels != target_label],
                                        parent_scores["calibration"][calibration_labels != target_label]))
    for record in assignment["candidates"]:
        config = record["configuration"]
        candidate_directory = directory / config["candidate_id"]
        stage_name = assignment["assignment_id"] + "/" + config["candidate_id"]
        try:
            record["status"] = "fitting"
            with _stage(run, manifest, stage_name + "/fit", deadline=deadline):
                proposal = _fit(base, target, protected, desired, config)
                program, candidate = proposal["program"], proposal["artifact"]
                record.update(no_op=not program.edits, fit_diagnostics=proposal["diagnostics"],
                              program_sha256=program.sha256, candidate_sha256=candidate.sha256)
            with _stage(run, manifest, stage_name + "/structural-admission", deadline=deadline):
                structural = _structural(base, program, candidate)
                record["structural_eligible"] = structural["structural_accepted"]
                record["structural"] = structural
            with _stage(run, manifest, stage_name + "/export-and-verify-candidate", deadline=deadline):
                record["files"] = _export_candidate(candidate_directory, base, proposal, structural)
            if not record["structural_eligible"]:
                record["status"] = "structurally_rejected"
                for gate in record["gates"]:
                    gate.update(status="ineligible", passes=False, useful=False, reasons=["structural_ineligible"])
                support._manifest(run / "manifest.json", manifest)
                continue
            record["status"], record["metrics"] = "native_scoring", {}
            with _stage(run, manifest, stage_name + "/native-scores-and-ownership", deadline=deadline):
                native_scores = {}
                for scope, values, scope_labels in (
                        ("fitting", fitting, fitting_labels), ("calibration", calibration, calibration_labels)):
                    native_scores[scope] = evaluate_head(candidate, values, native=True)
                    record["metrics"][scope] = _ownership_metrics(
                        parent_scores[scope], native_scores[scope], scope_labels, task)
                    support._manifest(run / "manifest.json", manifest)
                candidate_protected_scores = np.vstack((
                    native_scores["fitting"][fitting_labels != target_label],
                    native_scores["calibration"][calibration_labels != target_label]))
            with _stage(run, manifest, stage_name + "/exact-receipt-and-observation", deadline=deadline):
                observation, receipt = _gate_observation(
                    base, candidate, protected_features, parent_protected_scores, candidate_protected_scores)
                record["gate_observation"] = observation
                support._write_json(candidate_directory / "gate-retention.json", receipt)
                record["files"]["gate-retention.json"] = _file_record(candidate_directory / "gate-retention.json")
            with _stage(run, manifest, stage_name + "/native-score-export-and-verification", deadline=deadline):
                captured = {
                    "schema": "wsp-development-native-score-capture/1", "dtype": "float32",
                    "base_sha256": base.sha256, "candidate_sha256": candidate.sha256,
                    "fitting_labels": fitting_labels.tolist(), "calibration_labels": calibration_labels.tolist(),
                    "parent": {scope: scores.tolist() for scope, scores in parent_scores.items()},
                    "candidate": {scope: scores.tolist() for scope, scores in native_scores.items()},
                    "ownership_task": task}
                path = candidate_directory / "native-scores.json"
                support._write_json(path, captured)
                if support.read_json(path) != captured:
                    raise StudyError("native_score_capture_roundtrip_failed")
                record["files"]["native-scores.json"] = _file_record(path)
            with _stage(run, manifest, stage_name + "/all-fixed-gates-no-refit", deadline=deadline):
                for gate in record["gates"]:
                    gate.update(_gate_verdict(gate, observation, record["metrics"], True))
                record["status"] = "completed"
        except BaseException as caught:
            record.update(status="failed", reason=_failure(caught))
            support._manifest(run / "manifest.json", manifest)
            if isinstance(caught, StudyBudgetExceeded) or not isinstance(caught, Exception):
                raise
    assignment["status"] = ("schedule_finished_with_failures"
                            if any(record["status"] == "failed" for record in assignment["candidates"])
                            else "schedule_finished")
    support._manifest(run / "manifest.json", manifest)


def _verify_prepared(run: Path, manifest: dict) -> dict:
    _check_schedule(manifest)
    protocol = support.read_json(run / "protocol.json")
    if (protocol.get("schema") != "wsp-retention-development-protocol/1"
            or digest(protocol) != manifest["protocol_sha256"]
            or _file_record(run / "protocol.json") != manifest["protocol_file"]
            or canonical_bytes(protocol["configuration"]) != canonical_bytes(configuration())
            or canonical_bytes(protocol["assigned_schedule"]) != canonical_bytes(_design())
            or protocol["evaluator"] != _source_identity()
            or manifest.get("all_raw_roles_consumed") is not True
            or canonical_bytes(protocol["development_roles"]) != canonical_bytes(manifest["development_roles"])
            or set(protocol["development_roles"]) != set(ROLE_FAMILIES)):
        raise StudyError("prepared_protocol_source_runtime_or_consumption_changed")
    for role, record in protocol["development_roles"].items():
        raw = record["registration"]
        raw_body = {key: value for key, value in raw.items() if key != "manifest_sha256"}
        parents = sorted(f"{STUDY_ID}/label-{label}/f{family:02d}"
                         for label in range(3) for family in ROLE_FAMILIES[role])
        if (raw.get("schema") != "wsp-data-batch/1" or digest(raw_body) != raw["manifest_sha256"]
                or raw["registry_id"] != protocol["registry_id"]
                or raw["source_id"] != STUDY_ID + "/raw/" + role
                or raw["family_id"] != STUDY_ID + "/role/" + role
                or raw["parents"] != parents
                or raw["content_sha256"] != record["file"]["sha256"]
                or record["row_count"] != 48*len(ROLE_FAMILIES[role])
                or len(raw["row_sha256"]) != record["row_count"]
                or any(raw.get(key) is not False for key in support.FLAGS)):
            raise StudyError("frozen_development_role_binding_invalid")
        for field, registration in (("consumption", raw), ("alias_consumption", record["aliases"]["raw"])):
            decision = record[field]
            body = {key: value for key, value in decision.items() if key != "decision_sha256"}
            if (digest(body) != decision["decision_sha256"]
                    or decision["schema"] != "wsp-data-role-decision/1"
                    or decision["operation"] != "consume" or decision["status"] != "succeeded"
                    or decision["reason"] != "development_consumption_committed"
                    or decision["verdict"] != "registered_cooperative"
                    or decision["registry_id"] != protocol["registry_id"]
                    or decision["manifest_sha256"] != registration["manifest_sha256"]
                    or decision["content_sha256"] != registration["content_sha256"]
                    or decision["row_sha256"] != registration["row_sha256"]
                    or decision["scoring_started"] is not False
                    or any(decision.get(key) is not False for key in support.FLAGS)):
                raise StudyError("frozen_development_consumption_binding_invalid")
        aliases = record["aliases"]
        if (set(aliases) != {"raw", "normalized", "identities"}
                or any(raw["source_id"] not in alias["ancestor_source_ids"] for alias in aliases.values())):
            raise StudyError("frozen_development_alias_ancestry_invalid")
    _require_analyzer()
    return protocol


def run_study(run_directory: Path, registry_path: Path) -> dict:
    run = Path(run_directory)
    fresh = not run.exists()
    deadline = PROCESS_STARTED + EXTERNAL_CAP_SECONDS - FINALIZATION_RESERVE_SECONDS
    if fresh:
        prepare(run, registry_path, deadline=deadline)
    manifest = support.read_json(run / "manifest.json")
    prior_requests = len(manifest["execution_requests"])
    request = {"request_index": prior_requests, "requested_utc": _now(),
               "status": "requested", "reason": None}
    manifest["execution_requests"].append(request)
    support._manifest(run / "manifest.json", manifest)
    if prior_requests:
        request.update(status="rejected", reason={"type": "StudyError", "reason": "development_execution_request_already_recorded"})
        support._manifest(run / "manifest.json", manifest)
        support._manifest(run / "summary.json", summarize_manifest(manifest))
        return manifest
    started, started_cpu = time.perf_counter(), time.process_time()
    protocol = None
    try:
        if manifest["phase"] != "prepared":
            raise StudyError("only_a_prepared_unexecuted_study_can_run")
        if not fresh:
            preparation_seconds = manifest["preparation_process_observed_wall_seconds"]
            if (type(preparation_seconds) not in (int, float) or not math.isfinite(preparation_seconds)
                    or preparation_seconds < 0):
                raise StudyError("observed_preparation_process_cost_required")
            deadline -= preparation_seconds
        manifest["cooperative_work_budget_remaining_at_request_seconds"] = max(0., deadline-time.perf_counter())
        # Claim before source/runtime verification: even a rejected first
        # request cannot silently become a new study attempt in another folder.
        with _stage(run, manifest, "run/claim-development-attempt"):
            registry = DataRoleRegistry(registry_path)
            binding = {"study_id": STUDY_ID, "protocol_sha256": manifest["protocol_sha256"],
                       "registry_id": registry.registry_id, "requested_utc": request["requested_utc"],
                       "retry_allowed": False, "heldout_authorization": False}
            registry_file = Path(registry_path).resolve()
            shared_marker = registry_file.with_name(
                registry_file.name + "." + digest(STUDY_ID)[:16] + ".development-attempt.json")
            try:
                support._write_json(shared_marker, binding)
            except FileExistsError:
                raise StudyError("shared_registry_development_attempt_already_claimed") from None
            support._write_json(run / "execution-attempt.json", binding)
            manifest.update(phase="execution_claimed", source_unchanged_at_completion=None)
            request["status"] = "claimed"
        with _stage(run, manifest, "run/verify-frozen-design-and-runtime", deadline=deadline):
            protocol = _verify_prepared(run, manifest)
            if registry.registry_id != protocol["registry_id"]:
                raise StudyError("prepared_registry_identity_changed")
            manifest["phase"] = "running"
        with _stage(run, manifest, "run/reconsume-all-raw-roles-before-caller-materialization", deadline=deadline):
            rows, raw_batches = _reopen_development(run, registry, protocol, manifest)
        for seed in SEEDS:
            assigned = [a for a in manifest["assignments"] if a["seed"] == seed]
            try:
                parent, features = _train_parent(run, manifest, registry, rows, raw_batches, seed, deadline)
                for assignment in assigned:
                    assignment["parent_sha256"] = parent["head"].sha256
                    try:
                        _run_assignment(run, manifest, assignment, parent, features, rows, deadline)
                    except BaseException as caught:
                        assignment.update(status="failed", reason=_failure(caught))
                        support._manifest(run / "manifest.json", manifest)
                        if isinstance(caught, StudyBudgetExceeded) or not isinstance(caught, Exception):
                            raise
            except BaseException as caught:
                for assignment in assigned:
                    if assignment["status"] in ("assigned", "running"):
                        assignment.update(status="failed", reason=_failure(caught))
                support._manifest(run / "manifest.json", manifest)
                if isinstance(caught, StudyBudgetExceeded) or not isinstance(caught, Exception):
                    raise
        manifest["phase"] = ("schedule_finished_with_failures"
                             if any(a["status"] != "schedule_finished" for a in manifest["assignments"])
                             else "schedule_finished")
        request["status"] = "schedule_finished"
    except BaseException as caught:
        claimed = manifest["phase"] in ("execution_claimed", "running")
        request.update(status="interrupted" if claimed else "rejected", reason=_failure(caught))
        if claimed:
            manifest.update(phase="interrupted", reason=_failure(caught))
        elif not prior_requests and manifest["phase"] == "prepared":
            manifest.update(phase="execution_rejected", reason=_failure(caught))
        # No retry: a failed/rejected execution request remains visible even if no
        # training was reached. Preparation itself never authorizes outcomes.
        if isinstance(caught, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if protocol is not None and request["status"] != "rejected":
            try:
                with _stage(run, manifest, "finalize/source-identity-check"):
                    manifest["source_unchanged_at_completion"] = protocol["evaluator"] == _source_identity()
                    if not manifest["source_unchanged_at_completion"]:
                        manifest["phase"] = "source_invalidated"
                        manifest["source_identity_failure"] = {"reason": "frozen_evaluator_changed_during_run"}
            except Exception as caught:
                manifest["source_unchanged_at_completion"] = None
                manifest.setdefault("finalization_failures", []).append({"stage": "source-identity", **_failure(caught)})
        # Finalization does not check the work deadline. An external cap can
        # interrupt it; the durable schedule and unfinished stages survive.
        try:
            with _stage(run, manifest, "finalize/summary-initial-export-and-verification"):
                initial_summary = summarize_manifest(manifest)
                support._manifest(run / "summary.json", initial_summary)
                if support.read_json(run / "summary.json") != initial_summary:
                    raise StudyError("summary_roundtrip_failed")
        except Exception as caught:
            manifest.setdefault("finalization_failures", []).append({"stage": "summary-export", **_failure(caught)})
        manifest.update(run_observed_wall_seconds=time.perf_counter()-started,
                        run_observed_cpu_seconds=time.process_time()-started_cpu,
                        process_observed_wall_seconds=time.perf_counter()-PROCESS_STARTED,
                        process_observed_cpu_seconds=time.process_time()-PROCESS_CPU_STARTED,
                        timing_snapshot_scope="through_initial_summary_verification; final manifest/summary persistence is censored")
        # Both best-effort writes are independent; source or summary failures
        # must not conceal the original execution reason or prevent the manifest.
        try:
            support._manifest(run / "manifest.json", manifest)
        except Exception as caught:
            manifest.setdefault("finalization_failures", []).append({"stage": "final-manifest", **_failure(caught)})
        try:
            support._manifest(run / "summary.json", summarize_manifest(manifest))
        except Exception as caught:
            manifest.setdefault("finalization_failures", []).append({"stage": "final-summary", **_failure(caught)})
            try:
                support._manifest(run / "manifest.json", manifest)
            except Exception:
                pass  # Filesystem failure: caller still receives the in-memory failure record.
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path, help="Summary output; defaults to partial-summary.json.")
    args = parser.parse_args()
    if args.command == "summarize":
        manifest = support.read_json(args.run_directory / "manifest.json")
        output = args.output or args.run_directory / "partial-summary.json"
        support._manifest(output, summarize_manifest(manifest))
    else:
        if args.registry is None:
            parser.error("--registry must name the shared development registry")
        manifest = (prepare(args.run_directory, args.registry) if args.command == "prepare"
                    else run_study(args.run_directory, args.registry))
        output = args.run_directory / ("protocol.json" if args.command == "prepare" else "summary.json")
    print({"phase": manifest["phase"], "output": str(output), "assigned_seed_task_denominator": 6})


if __name__ == "__main__":
    main()
