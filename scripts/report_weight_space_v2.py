"""Render saved generic-study evidence; never train, edit, or regenerate worlds."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("no_op", "unconstrained", "soft_retention", "nullspace", "matched_random", "gradient")
LABELS = ("No-op", "Target only", "Soft retention", "Nullspace (12)", "Matched random", "Gradient (200)")
FAMILY_LABELS = {"relu_mlp": "ReLU MLP", "tanh_mlp": "Tanh MLP", "conv_tanh": "Convolution + tanh"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def mean(values):
    return None if not values else float(np.mean(values))


def summarize(results, protocol, study: Path):
    cases = results["cases"]
    mains = [case for case in cases if case.get("main")]
    families = protocol["families"]
    metrics = ("target_nrmse", "retain_nrmse", "target_gain_fraction", "delta_l2",
               "original_target_rmse", "edited_task_target_rmse", "original_retain_rmse", "edited_task_retain_rmse")
    overall = {arm: {"n": sum(arm in c["main"]["arms"] for c in mains),
                     **{key: mean([c["main"]["arms"][arm][key] for c in mains if arm in c["main"]["arms"]])
                        for key in metrics}} for arm in ARMS}
    selected, admission = [], Counter()
    for case in mains:
        main = case["main"]
        selection = read(study / f"world-{case['seed']}" / case["family"] / "main/selection.json")
        for arm, record in selection["validation"].items():
            admission[arm] += int(record["structural_accepted"])
        values = main["arms"][main["selected_arm"]]
        selected.append({"seed":case["seed"], "family":case["family"], "arm":main["selected_arm"],
                         "test_target_nrmse":values["target_nrmse"], "test_retain_nrmse":values["retain_nrmse"],
                         "within_validation_thresholds_on_test": values["target_nrmse"] <= protocol["selection"]["max_target_nrmse"]
                             and values["retain_nrmse"] <= protocol["selection"]["max_retain_nrmse"]})
    feasibility = []
    for family in families:
        for anchors in protocol["feasibility"]:
            rows = [r for c in cases if c["family"] == family and c.get("ablations")
                    for r in c["ablations"]["feasibility"] if r["anchors"] == anchors]
            if not rows:
                continue
            feasibility.append({"family":family, "anchors":anchors, "n":len(rows),
                "rank_range":[min(r["retention_rank"] for r in rows),max(r["retention_rank"] for r in rows)],
                "nullity_range":[min(r["nullspace_dimension"] for r in rows),max(r["nullspace_dimension"] for r in rows)],
                "no_ops":sum(r["test"]["no_op"] for r in rows),
                "exact_anchor_zero":sum(r["exactly_zero_on_anchors"] for r in rows),
                "maximum_anchor_residual":max(r["maximum_anchor_residual"] for r in rows),
                "target_nrmse":mean([r["test"]["target_nrmse"] for r in rows]),
                "retain_nrmse":mean([r["test"]["retain_nrmse"] for r in rows])})
    precision = []
    for dtype in protocol["precision"]:
        rows = [r for c in cases if c.get("ablations") for r in c["ablations"]["precision"] if r["dtype"] == dtype]
        if not rows:
            continue
        projections = [p for r in rows for p in r["lowering"]["projections"].values()]
        precision.append({"dtype":dtype,"n":len(rows),
            "native_target_nrmse":mean([r["native_target_nrmse"] for r in rows]),
            "native_retain_nrmse":mean([r["native_retain_nrmse"] for r in rows]),
            "lifted_target_nrmse":mean([r["lifted"]["target_nrmse"] for r in rows]),
            "lifted_retain_nrmse":mean([r["lifted"]["retain_nrmse"] for r in rows]),
            "maximum_native_drift_discrepancy":max(r["native_drift_vs_lifted_max_abs"] for r in rows),
            "maximum_native_base_discrepancy":max(r["native_base_vs_lifted_max_abs"] for r in rows),
            "maximum_native_candidate_discrepancy":max(r["native_candidate_vs_lifted_max_abs"] for r in rows),
            "requested_changed_cells":sum(p["requested_changed_cells"] for p in projections),
            "effective_changed_cells":sum(p["effective_changed_cells"] for p in projections),
            "rounded_cells":sum(p["rounded_cells"] for p in projections),
            "maximum_endpoint_rounding_error":max(float(Fraction(p["maximum_rounding_error_exact"])) for p in projections)})
    sequence = []
    for family in families:
        for method in protocol["sequence"]["methods"]:
            rows = [c["sequence"]["methods"][method] for c in cases if c["family"] == family and c.get("sequence")]
            if not rows:
                continue
            sequence.append({"family":family,"method":method,"n":len(rows),
                "final_mean_target_nrmse":mean([r["final_mean_target_nrmse"] for r in rows]),
                "final_prior_region_interference_nrmse":mean([r["final_prior_region_interference_nrmse"] for r in rows]),
                "composition_exact_count":sum(r["composition_exact"] for r in rows),
                "rollback_exact_count":sum(r["rollback_exact"] for r in rows),
                "stage_region_target_nrmse":[[mean([r["stages"][stage]["regions"][region]["desired_behavior_nrmse"] for r in rows])
                                               for region in range(3)] for stage in range(3)]})
    auxiliary = [c for world in results["auxiliary"] for c in world["result"]["case_results"]]
    readings = []
    for access in ("exact", "rounded3"):
        rows = [c for c in auxiliary if c["family"] == "affine_sign" and c.get("access") == access and c.get("metrics")]
        readings.append({"access":access,"n":len(rows),
            "correct":sum(c["metrics"]["sign_correct"] is True for c in rows),
            "incorrect":sum(c["metrics"]["sign_correct"] is False for c in rows),
            "abstentions":sum(c["abstained"] for c in rows),
            "response_exact_count":sum(c["metrics"]["hidden_response"]["exact_array_equality"] for c in rows),
            "response_within_tolerance_count":sum(c["metrics"]["hidden_response"]["within_tolerance"] for c in rows),
            "maximum_response_error":max((c["metrics"]["hidden_response"]["maximum_absolute_error"] for c in rows),default=None)})
    creation = []
    for formula in ("scaled_absolute","clip"):
        for arm in ("zero","random","analytic"):
            rows=[c["metrics"]["arms"][arm]["hidden_response"] for c in auxiliary
                  if c["family"] == "analytic_creation" and c.get("formula") == formula and c.get("metrics")]
            creation.append({"formula":formula,"arm":arm,"n":len(rows),
                             "exact_count":sum(r["exact_array_equality"] for r in rows),
                             "within_tolerance_count":sum(r["within_tolerance"] for r in rows),
                             "mean_rmse":mean([r["rmse"] for r in rows]),
                             "maximum_response_error":max((r["maximum_absolute_error"] for r in rows),default=None)})
    bounds = [c["main"]["real_domain_bound"]["output_bounds"] for c in mains if c["main"]["real_domain_bound"]]
    return {"schema":"wsp-generic-derived-report/2","complete":results["complete"],"overall_arms":overall,
        "structurally_admitted":dict(admission),"selected_cases":selected,
        "selected_within_validation_thresholds_on_test":sum(r["within_validation_thresholds_on_test"] for r in selected),
        "selected_target_nrmse":mean([r["test_target_nrmse"] for r in selected]),
        "selected_retain_nrmse":mean([r["test_retain_nrmse"] for r in selected]),
        "feasibility":feasibility,"precision":precision,"sequence":sequence,
        "auxiliary":{"attempted_cases":len(auxiliary),"readings":readings,"creation":creation},
        "relu_domain_bounds":{"count":len(bounds),"minimum_by_output":np.min(bounds,axis=0).tolist() if bounds else None,
                              "maximum_by_output":np.max(bounds,axis=0).tolist() if bounds else None},
        "accounting":{"main_usage":{key:sum(c["main"]["usage"][key] for c in mains) for key in mains[0]["main"]["usage"]} if mains else {},
                      "validation_model_example_executions":sum(c["main"]["validation_model_example_executions"] for c in mains),
                      "main_test_model_example_executions":sum(c["main"]["test_accounting"]["model_example_executions"] for c in mains),
                      "feasibility_model_example_executions":sum(c["ablations"]["feasibility_model_example_executions"] for c in cases if c.get("ablations")),
                      "precision_model_example_executions":sum(c["ablations"]["precision_model_example_executions"] for c in cases if c.get("ablations")),
                      "sequence_model_example_executions":sum(c["sequence"]["model_example_executions"] for c in cases if c.get("sequence"))}}


def table(headers, rows):
    lines=["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def figures(results, protocol, derived, out):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
    def save(fig,name):
        for ext in ("png","svg","pdf"):
            fig.savefig(out/f"{name}.{ext}",bbox_inches="tight")
        plt.close(fig)
    families=protocol["families"]
    fig,axes=plt.subplots(1,len(families),figsize=(15,4.7),sharey=True,squeeze=False)
    x=np.arange(len(ARMS))
    for ax,family in zip(axes[0],families):
        rows=results["by_family"][family]["arm_means"]
        ax.bar(x-.19,[rows.get(a,{}).get("target_nrmse",np.nan) for a in ARMS],.38,color="#2b6cb0",label="Target error")
        ax.bar(x+.19,[rows.get(a,{}).get("retain_nrmse",np.nan) for a in ARMS],.38,color="#c77724",label="Collateral drift")
        ax.set_xticks(x,LABELS,rotation=35,ha="right")
        ax.set_title(FAMILY_LABELS[family]);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    axes[0,0].set_ylabel("Mean normalized RMS error (lower is better)")
    axes[0,-1].legend(frameon=False)
    fig.suptitle("Fixed arms on held-out inputs · 12 paired worlds per family",y=1.02)
    fig.tight_layout();save(fig,"main-arms")

    names=list(results["contrasts"])
    labels=["Collateral reduction\n(target-only − soft)","Target error cost\n(soft − target-only)",
            "Target advantage\n(random − soft)","Target advantage\n(gradient − soft)"]
    fig,ax=plt.subplots(figsize=(9,4.4))
    for i,name in enumerate(names):
        row=results["contrasts"][name]
        if row is None:
            continue
        lo,hi=row["ci95_percentile"];m=row["mean"]
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt="o",capsize=4,color="#2b6cb0")
        ax.text(hi+.02,i,f"{m:.3f} [{lo:.3f}, {hi:.3f}]",va="center",fontsize=9)
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.axvline(0,color="#777777",lw=1)
    ax.set_xlim(-.025,1.3);ax.set_xlabel("Difference in normalized RMS error")
    ax.set_title("Exploratory 95% world-cluster bootstrap intervals\nFamilies averaged within each world; no multiplicity correction")
    fig.tight_layout();save(fig,"paired-contrasts")

    fig,axes=plt.subplots(1,len(families),figsize=(13,4),sharey=True,squeeze=False)
    for ax,family in zip(axes[0],families):
        rows=[r for r in derived["feasibility"] if r["family"]==family]
        for key,label,color in (("target_nrmse","Target error","#2b6cb0"),("retain_nrmse","Collateral drift","#c77724")):
            ax.plot([r["anchors"] for r in rows],[r[key] for r in rows],"o-",label=label,color=color)
        ax.set_xticks(protocol["feasibility"]);ax.set_xlabel("Protected public anchors")
        ax.set_title(FAMILY_LABELS[family]);ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Mean held-out normalized RMS error");axes[0,-1].legend(frameon=False)
    fig.suptitle("More anchor constraints can remove all editable directions",y=1.02)
    fig.tight_layout();save(fig,"anchor-feasibility")

    fig,ax=plt.subplots(figsize=(8,4))
    rows=derived["precision"]
    values=[r["maximum_native_drift_discrepancy"] for r in rows]
    ax.plot([r["dtype"] for r in rows],values,"o-",color="#6b46a4")
    ax.set_yscale("log");ax.set_ylabel("Maximum |native edit drift − lifted edit drift|")
    ax.set_title("CPU arithmetic discrepancy on saved test inputs\nMaximum across 36 checkpoints; each format has its own projected parent")
    ax.grid(axis="y",alpha=.2)
    for i,value in enumerate(values):
        ax.annotate(f"{value:.2e}",(i,value),xytext=(0,8),textcoords="offset points",ha="center")
    ax.margins(y=.15);fig.tight_layout();save(fig,"native-precision")

    fig,axes=plt.subplots(2,len(families),figsize=(12,7),squeeze=False,layout="constrained")
    vmax=max(float(np.max(r["stage_region_target_nrmse"])) for r in derived["sequence"])
    for row_index,method in enumerate(protocol["sequence"]["methods"]):
        for col,family in enumerate(families):
            row=next(r for r in derived["sequence"] if r["family"]==family and r["method"]==method)
            values=np.asarray(row["stage_region_target_nrmse"])
            ax=axes[row_index,col];im=ax.imshow(values,vmin=0,vmax=vmax,cmap="YlOrRd",aspect="auto")
            for i in range(3):
                for j in range(3):
                    ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",color="white" if values[i,j]>.6*vmax else "black")
            ax.set_xticks(range(3),["Region 1","Region 2","Region 3"])
            ax.set_yticks(range(3),["After edit 1","After edit 2","After edit 3"])
            ax.set_title(FAMILY_LABELS[family]+"\n"+method.replace("_"," "))
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Mean desired-behavior NRMSE (lower is better)",shrink=.8)
    fig.suptitle("Three fixed edits: desired behavior before acquisition is the unchanged baseline")
    save(fig,"sequential-interference")


def report(results,protocol,derived):
    lines=["# Generic weight-space framework: saved study results", "",
           "Generated from immutable saved evidence. This report evaluates deterministic fitting-allowed controls, not LLM programming. All uncertainty intervals are exploratory; no multiplicity correction is applied.","",
           f"Completion: **{results['complete']}**. Main checkpoints: **{results['completed_checkpoints']}/{results['attempted_checkpoints']}**. Complete independent worlds: **{results['complete_worlds_for_cluster_inference']}**. Failures: **{len(results['failures'])}**.","",
           f"Source freeze: `{results['freeze_sha256']}`. Protocol: `{results['protocol_sha256']}`.","",
           "Interpretation, equations, and boundaries are in [methods](../../../docs/weight-space/v2/methods.md), [results discussion](../../../docs/weight-space/v2/results.md), and [limitations](../../../docs/weight-space/v2/limitations-and-next-steps.md). Exact derived values are in [summary.json](summary.json).", "",
           "## Fixed arms, averaged over available checkpoints", "",
           table(["Arm","n","Target NRMSE","Retain NRMSE","Directional gain","Delta L2"],
                 [[a,derived["overall_arms"][a]["n"],*[f"{derived['overall_arms'][a][k]:.6f}" for k in ("target_nrmse","retain_nrmse","target_gain_fraction","delta_l2")]] for a in ARMS]),"",
           "![Fixed arm errors by architecture](main-arms.png)","",
           "Each family has twelve paired checkpoints. Error averages do not turn those 36 checkpoints into 36 independent task units.","",
           "## Predeclared paired contrasts", "",
           table(["Contrast","Worlds","Mean","Exploratory 95% interval"],
                 [[name,row["independent_worlds"],f"{row['mean']:.6f}",f"[{row['ci95_percentile'][0]:.6f}, {row['ci95_percentile'][1]:.6f}]"]
                  for name,row in results["contrasts"].items() if row is not None]),"",
           "![Paired contrasts](paired-contrasts.png)","",
           "## Validation-selected candidates", "",
           f"Selections: `{json.dumps(results['selected_arms'],sort_keys=True)}`. Mean held-out target NRMSE: **{derived['selected_target_nrmse']:.6f}**; retention NRMSE: **{derived['selected_retain_nrmse']:.6f}**. **{derived['selected_within_validation_thresholds_on_test']}/{len(derived['selected_cases'])}** also meet the validation error thresholds on test inputs. This is a descriptive threshold check, not a population guarantee or a newly predeclared success endpoint.","",
           f"Structurally admitted fixed candidates: `{json.dumps(derived['structurally_admitted'],sort_keys=True)}`. The selected candidate aliases an existing arm.","",
           "## Anchor feasibility", "",
           table(["Family","Anchors","n","Rank range","Nullity range","No-op","Exact anchor zero","Max anchor residual","Target","Retain"],
                 [[r["family"],r["anchors"],r["n"],r["rank_range"],r["nullity_range"],r["no_ops"],r["exact_anchor_zero"],f"{r['maximum_anchor_residual']:.3e}",f"{r['target_nrmse']:.6f}",f"{r['retain_nrmse']:.6f}"] for r in derived["feasibility"]]),"",
           "![Anchor feasibility](anchor-feasibility.png)","",
           "A nonzero exact endpoint residual remains nonzero even when it is tiny. Full protected rank can force a no-op; preserving sampled anchors does not establish unseen retention.","",
           "## Native precision", "",
           table(["Format","n","Native target","Native retain","Max drift discrepancy","Rounded cells","Requested/effective changed cells"],
                 [[r["dtype"],r["n"],f"{r['native_target_nrmse']:.6f}",f"{r['native_retain_nrmse']:.6f}",f"{r['maximum_native_drift_discrepancy']:.3e}",r["rounded_cells"],f"{r['requested_changed_cells']}/{r['effective_changed_cells']}"] for r in derived["precision"]]),"",
           "![Native arithmetic discrepancy](native-precision.png)","",
           "These are measured CPU PyTorch results. Each format projects a separate parent; compiler features use lifted binary64 semantics. No IEEE or GPU theorem is claimed.","",
           "## Repeated edits", "",
           table(["Family","Method","Chains","Final target NRMSE","Prior-region interference","Exact compositions","Exact rollbacks"],
                 [[r["family"],r["method"],r["n"],f"{r['final_mean_target_nrmse']:.6f}",f"{r['final_prior_region_interference_nrmse']:.6f}",r["composition_exact_count"],r["rollback_exact_count"]] for r in derived["sequence"]]),"",
           "![Sequential interference](sequential-interference.png)","",
           "Requested target behavior refers to the original checkpoint; retention fits current outputs in other regions. Exact rollback restores parameters, not earlier deployed trajectories. Three edits are not a long-horizon guarantee.","",
           "## Auxiliary task controls", "",
           table(["Read/diagnose access","n","Correct sign","Incorrect","Abstain","Exact response","Max response error"],
                 [[r["access"],r["n"],r["correct"],r["incorrect"],r["abstentions"],r["response_exact_count"],f"{r['maximum_response_error']:.3e}"] for r in derived["auxiliary"]["readings"]]),"",
           table(["Creation formula","Arm","n","Within tolerance","Exact response","Mean RMSE","Max error"],
                 [[r["formula"],r["arm"],r["n"],r["within_tolerance_count"],r["exact_count"],f"{r['mean_rmse']:.6f}",f"{r['maximum_response_error']:.3e}"] for r in derived["auxiliary"]["creation"]]),"",
           "The analytic creation arm was selected a priori. Read/diagnose sign checks and formula creation are different endpoints and are not pooled into one accuracy. They measure deterministic fixtures, not mechanism discovery or autonomous synthesis.","",
           "## Accounting and boundaries", "", "```json", json.dumps(derived["accounting"],indent=2), "```", "",
           "These counts distinguish callbacks, submitted examples, and actual model-example executions. They are not a complete FLOP or energy accounting. Provider calls and model tokens are zero.","",
           "The three families share a synthetic teacher and split design within a world. Only output heads were trained. The main-region gap is unmeasured, generic contract admission is not fully integrated, native execution and the compiler are unproved, and the new runtime candidates are not bound to generated Lean instances. See the roadmap before extending these results to live LLMs, real data, larger architectures, or deployment.","",
           "All figures are also available as SVG and PDF files beside their PNG previews. The renderer receipt records input, script, dependency, and generated-file identities.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();study=args.study.resolve();out=args.out.resolve()
    manifest=read(study/"manifest.json")
    before={p.relative_to(study).as_posix():sha(p) for p in sorted(study.rglob("*")) if p.is_file()}
    if set(before) != set(manifest["files"])|{"manifest.json"}:
        raise ValueError("Study file membership differs from manifest")
    for name,record in manifest["files"].items():
        if before[name]!=record["sha256"] or (study/name).stat().st_size!=record["bytes"]:
            raise ValueError(f"Study input identity mismatch: {name}")
    results=read(study/"results.json");protocol=read(study/"protocol.json")
    derived=summarize(results,protocol,study)
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/"summary.json",derived)
    figures(results,protocol,derived,out)
    with (out/"report.md").open("x",encoding="utf-8") as handle:
        handle.write(report(results,protocol,derived))
    after={p.relative_to(study).as_posix():sha(p) for p in sorted(study.rglob("*")) if p.is_file()}
    if before!=after:
        raise ValueError("Study inputs changed during rendering")
    outputs={p.relative_to(out).as_posix():{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file()}
    receipt={"schema":"wsp-generic-report-receipt/2","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
             "study":str(study),"source_script_sha256":sha(Path(__file__)),
             "study_manifest_sha256":before["manifest.json"],"study_results_sha256":before["results.json"],
             "input_files_unchanged":True,"input_files_checked":len(before),
             "python":platform.python_version(),"numpy":np.__version__,"matplotlib":matplotlib.__version__,"files":outputs,
             "scope":"post-result descriptive rendering; no new generation, training, fitting, or proposal selection"}
    save_json(out/"receipt.json",receipt)
    print(json.dumps({"out":str(out),"files":len(outputs)+1,"input_files_unchanged":True,"input_files_checked":len(before)},indent=2))


if __name__=="__main__":
    main()
