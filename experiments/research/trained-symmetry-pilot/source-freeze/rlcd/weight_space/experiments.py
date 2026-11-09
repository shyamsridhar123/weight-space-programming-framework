"""Frozen, falsifiable experiments over independently trained small networks.

No optimizer or model-selection procedure consumes held-out examples. All
predeclared units, rejected proposals, controls, and failures remain in evidence.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from .compiler import compile_readout
from .contracts import Contract, check_contract, exact_score
from .evidence import (bootstrap_mean, environment, file_hash, read_json, utc_now,
                       verify_freeze, write_json)
from .model import InputBox, Network
from .program import apply_program
from .symmetry import Gauge, align_hidden, transfer_edit, transform, transport


def teacher(x: np.ndarray) -> np.ndarray:
    """Fixed analytic teacher; equations are frozen with the executable source."""
    return np.column_stack((
        1.1*np.sin(1.5*x[:, 0]) + .55*x[:, 1]*x[:, 2] - .35*x[:, 3]
        + .25*np.maximum(x[:, 4]+x[:, 5]-.2, 0),
        .8*np.cos(1.2*x[:, 1]) - .65*x[:, 0]*x[:, 4] + .45*x[:, 2]
        - .3*np.maximum(x[:, 3]-x[:, 5], 0),
        .7*np.sin(x[:, 2]+1.1*x[:, 5]) + .4*x[:, 0]*x[:, 3] - .2*x[:, 1]
        + .3*np.maximum(x[:, 4]-.1, 0),
    ))


def make_data(config: dict[str, Any]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config["data_seed"])
    data = {}
    for split in ("train", "compile", "validation", "test"):
        x = rng.uniform(-1, 1, (config[f"{split}_samples"], config["input_features"]))
        data[f"{split}_x"], data[f"{split}_y"] = x, teacher(x)
    return data


def train_network(seed: int, config: dict[str, Any], data: dict[str, np.ndarray]) -> tuple[Network, dict[str, Any]]:
    import torch
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(config["input_features"], config["hidden_units"], dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(config["hidden_units"], config["output_actions"], dtype=torch.float64),
    )
    inputs, labels = torch.from_numpy(data["train_x"]), torch.from_numpy(data["train_y"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    losses = []
    started = time.perf_counter()
    for _ in range(config["training_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(inputs)-labels)**2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    network = Network(model[0].weight.detach().numpy(), model[0].bias.detach().numpy(),
                      model[2].weight.detach().numpy(), model[2].bias.detach().numpy(),
                      tuple(f"state_{i}" for i in range(config["input_features"])),
                      tuple(f"action_{i}" for i in range(config["output_actions"])))
    metrics = {f"{split}_mse": float(np.mean((network(data[f"{split}_x"])-data[f"{split}_y"])**2))
               for split in ("train", "validation", "test")}
    return network, {"seed": seed, "epochs": config["training_epochs"], "loss_trace": losses,
                     "wall_seconds": time.perf_counter()-started, **metrics}


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _edit_metrics(base: Network, candidate: Network, inputs: np.ndarray,
                  threshold: float, shift: float) -> dict[str, float]:
    target = inputs[:, 0] > threshold
    actual = candidate(inputs)-base(inputs)
    return {"target_gain_fraction": float(actual[target, 0].mean()/shift),
            "target_error_rmse": _rmse(actual[target, 0]-shift),
            "retention_rmse": _rmse(actual[~target]),
            "maximum_observed_drift": float(np.max(np.abs(actual)))}


def _pair(source: Network, target: Network, seed: int, config: dict[str, Any],
          data: dict[str, np.ndarray], out: Path) -> dict[str, Any]:
    out.mkdir()
    shift, threshold = config["requested_shift"], config["target_threshold"]
    compile_inputs = data["compile_x"]
    mask = compile_inputs[:, 0] > threshold
    target_inputs, retain_inputs = compile_inputs[mask], compile_inputs[~mask]
    desired = np.zeros((len(target_inputs), source.shape[2]))
    desired[:, 0] = shift
    box = InputBox(-np.ones(source.shape[0]), np.ones(source.shape[0]))
    contract = Contract(box, (config["drift_budget"],)*source.shape[2], allowed_tensors=("w2",))
    programs, candidates, admissions = {}, {}, {}
    diagnostics, metrics = {}, {}
    for arm, retention in (("target_only", 0.0), ("retention", config["retention_weight"])):
        compilation = compile_readout(
            source, target_inputs, desired, retain_inputs if retention else None,
            retention_weight=retention, ridge=config["ridge"],
            intent=f"{arm}: action0 +{shift} for x0>{threshold}; retention outside target")
        programs[arm] = compilation.program
        candidates[arm] = apply_program(source, compilation.program)
        admissions[arm] = check_contract(source, compilation.program, contract)
        diagnostics[arm] = compilation.diagnostics
        metrics[arm] = {split: _edit_metrics(source, candidates[arm], data[f"{split}_x"], threshold, shift)
                        for split in ("validation", "test")}
        write_json(out / f"{arm}-program.json", compilation.program.to_dict())
        write_json(out / f"{arm}-model.json", candidates[arm].to_dict())
        write_json(out / f"{arm}-admission.json", admissions[arm])
    # H1 tests the predeclared retention program, independently of its admission.
    # Rejected proposals are studied only in the isolated numerical experiment.
    program = programs["retention"]
    inputs = data["test_x"][data["test_x"][:, 0] > threshold]
    reference = (candidates["retention"](inputs)-source(inputs))[:, 0]

    rng = np.random.default_rng(seed+config["gauge_seed_offset"])
    known = Gauge(tuple(map(int, rng.permutation(source.shape[1]))),
                  tuple(map(float, rng.choice(config["gauge_scales"], source.shape[1]))))
    equivalent, transported = transport(source, program, known)
    transported_candidate = apply_program(equivalent, transported)
    inferred_known, known_diagnostic = align_hidden(source, equivalent)
    inferred_program = transfer_edit(source, program, equivalent, inferred_known)
    raw_known_program = transfer_edit(source, program, equivalent)
    exact_positive = all(
        exact_score(source, list(x)) == exact_score(equivalent, list(x))
        and exact_score(candidates["retention"], list(x)) == exact_score(transported_candidate, list(x))
        for x in inputs[:5])
    controls = {
        "exact_rational_point_checks": exact_positive, "exact_point_count": min(5, len(inputs)),
        "known_transport_error": _rmse((transported_candidate(inputs)-equivalent(inputs))[:, 0]-reference)/shift,
        "inferred_known_gauge_error": _rmse((apply_program(equivalent, inferred_program)(inputs)-equivalent(inputs))[:, 0]-reference)/shift,
        "raw_permuted_error": _rmse((apply_program(equivalent, raw_known_program)(inputs)-equivalent(inputs))[:, 0]-reference)/shift,
        "inferred_gauge": inferred_known.to_dict(), "inference_diagnostics": known_diagnostic,
    }
    write_json(out / "known-gauge.json", known.to_dict())
    write_json(out / "known-base.json", equivalent.to_dict())
    write_json(out / "known-transport-program.json", transported.to_dict())
    write_json(out / "known-raw-program.json", raw_known_program.to_dict())

    gauge, alignment = align_hidden(source, target)
    transfer_metrics = {}
    for name, mapping in (("raw", None), ("aligned", gauge)):
        transferred = transfer_edit(source, program, target, mapping)
        candidate = apply_program(target, transferred)
        admission = check_contract(target, transferred, contract)
        transfer_metrics[name] = {
            "normalized_target_delta_rmse": _rmse((candidate(inputs)-target(inputs))[:, 0]-reference)/shift,
            "admitted": admission["accepted"], "admission_issues": admission["issues"],
            "bounds": admission["certificate"]["output_bounds"],
        }
        write_json(out / f"transfer-{name}-program.json", transferred.to_dict())
        write_json(out / f"transfer-{name}-model.json", candidate.to_dict())
        write_json(out / f"transfer-{name}-admission.json", admission)
    write_json(out / "inferred-gauge.json", gauge.to_dict())
    return {"source_seed": seed, "source_sha256": source.sha256, "target_sha256": target.sha256,
            "compilation": diagnostics, "metrics": metrics, "controls": controls,
            "admission": {arm: {"accepted": a["accepted"], "issues": a["issues"],
                                "bounds": a["certificate"]["output_bounds"]} for arm, a in admissions.items()},
            "alignment": alignment, "transfer": transfer_metrics,
            "H1_raw_minus_aligned": transfer_metrics["raw"]["normalized_target_delta_rmse"]
            - transfer_metrics["aligned"]["normalized_target_delta_rmse"],
            "H2_retention_rmse_reduction": metrics["target_only"]["test"]["retention_rmse"]
            - metrics["retention"]["test"]["retention_rmse"],
            "H2_target_gain_fraction": metrics["retention"]["test"]["target_gain_fraction"]}


def run_synthetic(protocol_path: str | Path, freeze_path: str | Path, out: str | Path) -> dict[str, Any]:
    freeze = verify_freeze(protocol_path, freeze_path)
    protocol = read_json(protocol_path)
    config = protocol["networks"]
    seeds = config["training_seeds"]
    if len(seeds) % 2 or len(set(seeds)) != len(seeds):
        raise ValueError("The fixed study requires disjoint adjacent seed pairs")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(out / "started.json", {"created_at": utc_now(), "freeze_sha256": freeze["sha256"],
                                    "protocol_sha256": freeze["protocol_sha256"], "environment": environment()})
    data = make_data(config)
    with (out / "data.npz").open("xb") as handle:
        np.savez_compressed(handle, **data)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    models, training, failures, pairs = {}, [], [], []
    for seed in seeds:
        try:
            model, log = train_network(seed, config, data)
            models[seed] = model
            log["model_sha256"] = model.sha256
            write_json(out / "models" / f"{seed}.json", model.to_dict())
            write_json(out / "training" / f"{seed}.json", log)
            training.append(log)
            print(f"trained {seed}: test MSE={log['test_mse']:.6f}", flush=True)
        except Exception:
            failure = {"stage": "training", "seed": seed, "reason": traceback.format_exc()}
            failures.append(failure)
            write_json(out / "failures" / f"train-{seed}.json", failure)
    for source_seed, target_seed in zip(seeds[::2], seeds[1::2]):
        try:
            pair = _pair(models[source_seed], models[target_seed], source_seed, config,
                         data, out / f"pair-{source_seed}-{target_seed}")
            pair["target_seed"] = target_seed
            write_json(out / f"pair-{source_seed}-{target_seed}" / "result.json", pair)
            pairs.append(pair)
            print(f"pair {source_seed}/{target_seed}: H1={pair['H1_raw_minus_aligned']:.5f} "
                  f"H2 collateral reduction={pair['H2_retention_rmse_reduction']:.5f}", flush=True)
        except Exception:
            failure = {"stage": "pair", "source_seed": source_seed, "target_seed": target_seed,
                       "reason": traceback.format_exc()}
            failures.append(failure)
            write_json(out / "failures" / f"pair-{source_seed}-{target_seed}.json", failure)
    stats_config = protocol["statistical_policy"]
    def stat(key: str) -> dict[str, Any] | None:
        return bootstrap_mean([p[key] for p in pairs], seed=stats_config["bootstrap_seed"],
                              resamples=stats_config["bootstrap_resamples"]) if pairs else None
    h1, h2, gain = stat("H1_raw_minus_aligned"), stat("H2_retention_rmse_reduction"), stat("H2_target_gain_fraction")
    complete = not failures and len(pairs) == len(seeds)//2
    controls_pass = complete and all(p["controls"]["exact_rational_point_checks"]
        and p["controls"]["known_transport_error"] < 1e-12
        and p["controls"]["inferred_known_gauge_error"] < 1e-10 for p in pairs)
    results = {"schema": "wsp-synthetic-results/1", "created_at": utc_now(),
               "protocol_sha256": freeze["protocol_sha256"], "freeze_sha256": freeze["sha256"],
               "data_sha256": file_hash(out / "data.npz"), "planned_networks": len(seeds),
               "completed_networks": len(models), "planned_pairs": len(seeds)//2,
               "completed_pairs": len(pairs), "failures": failures, "complete": complete,
               "controls_pass": controls_pass, "wall_seconds": time.perf_counter()-started,
               "inference_scope": "initialization variation conditional on the shared fixed data partition",
               "H1": {"effect": h1, "status": "inconclusive" if not controls_pass else
                       ("supported" if h1["ci95"][0] > 0 else "unsupported")},
               "H2": {"effect": h2, "target_gain_fraction": gain,
                       "status": "inconclusive" if not controls_pass else
                       ("supported" if h2["ci95"][0] > 0 and gain["mean"] >= .5 else "unsupported")},
               "training_summary": [{k: v for k, v in t.items() if k != "loss_trace"} for t in training],
               "pairs": pairs}
    write_json(out / "results.json", results)
    hashes = {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "receipt.json", {"created_at": utc_now(), "freeze_sha256": freeze["sha256"],
                                    "artifacts": hashes})
    return results
