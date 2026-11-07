"""Render enterprise workflow and measured score tradeoff from the frozen bundle.

Requires the project's optional matplotlib dependency. SVG and PNG are original
project diagrams; no Mermaid, browser renderer, or generative labels are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/enterprise/support-routing"
INK, TEAL, RED, PAPER = "#172c36", "#176e68", "#9d4735", "#faf8f3"


def save(figure, name: str) -> None:
    for extension in ("svg", "png"):
        metadata = {"Date": None} if extension == "svg" else {"Software": "Matplotlib"}
        path = OUTPUT / f"{name}.{extension}"
        figure.savefig(path, dpi=180, facecolor=PAPER, metadata=metadata)
        if extension == "svg":
            # Preserve stable UTF-8 bytes across platforms in the publication manifest.
            content = "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n"
            path.write_bytes(content.encode("utf-8"))
    plt.close(figure)


def workflow() -> None:
    figure, axis = plt.subplots(figsize=(12, 9))
    axis.set(xlim=(0, 12), ylim=(0, 10))
    axis.axis("off")
    figure.text(.06, .95, "A routing change needs utility and protection", fontsize=21, color=INK, weight="bold")
    figure.text(.06, .91, "Synthetic enterprise workflow · selection is sealed before final evaluation", fontsize=12, color=INK)

    def card(x, y, title, detail, color=INK, width=3.1, height=1.4):
        axis.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                       boxstyle="round,pad=0.08,rounding_size=0.1", facecolor="white",
                       edgecolor=color, linewidth=1.4))
        axis.text(x, y+.25, title, ha="center", va="center", fontsize=12, color=color, weight="bold")
        axis.text(x, y-.24, detail, ha="center", va="center", fontsize=10, color=INK, linespacing=1.45)

    def arrow(start, end, label="", offset=(0, .2)):
        axis.annotate("", xy=end, xytext=start,
                      arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.3})
        if label:
            axis.text((start[0]+end[0])/2+offset[0], (start[1]+end[1])/2+offset[1],
                      label, ha="center", va="center", fontsize=10, color=INK)

    card(2, 8, "Define the policy", "Outages → Platform\nExisting queues protected")
    card(6, 8, "Capture the parent", "Exact weights + feature schema\nFixed acceptance rubric")
    card(10, 8, "Compile candidates", "Target-only · soft · protected\nInclude unchanged parent")
    arrow((3.65, 8), (4.35, 8))
    arrow((7.65, 8), (8.35, 8))
    arrow((10, 7.2), (10, 6.3))
    axis.add_patch(Polygon([(10, 6.3), (11.7, 5.35), (10, 4.4), (8.3, 5.35)],
                           closed=True, facecolor="white", edgecolor=TEAL, linewidth=1.4))
    axis.text(10, 5.55, "Any eligible edit?", ha="center", fontsize=12, weight="bold", color=TEAL)
    axis.text(10, 5.1, "Utility + retention\nStructure + receipt binding", ha="center", fontsize=9.5, color=INK)
    card(6, 5.35, "Seal one candidate", "Use only public evidence\nFreeze the selected artifact", TEAL)
    card(2, 5.35, "Final evaluation", "Same routing and drift rubric\nNo further candidate selection")
    arrow((8.3, 5.35), (7.65, 5.35), "yes", (0, .35))
    arrow((4.35, 5.35), (3.65, 5.35))
    card(10, 2.65, "Abstain", "No eligible proposal\nFinal evaluator is not called", RED)
    arrow((10, 4.4), (10, 3.45), "no", (.35, 0))
    card(2, 2.65, "Apply the final gate", "Pass → synthetic trial\nFail → reject the candidate", TEAL)
    arrow((2, 4.55), (2, 3.45))
    card(6, .65, "Retain the complete evidence", "Parent · contract · programs · certificates · session · final receipt · exact rollback",
         width=11, height=.9)
    arrow((2, 1.85), (2, 1.18))
    arrow((10, 1.85), (10, 1.18))
    figure.text(.06, .015, "Synthetic trial admission does not authorize production. Real-valued retention and sampled runtime behavior have separate scopes.",
                 fontsize=9, color=INK)
    save(figure, "routing-workflow")


def tradeoff(bundle: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    figure.subplots_adjust(top=.77, bottom=.23, left=.09, right=.97, wspace=.28)
    figure.text(.06, .94, "A target improvement can disturb protected queues", fontsize=21, color=INK, weight="bold")
    figure.text(.06, .875, "Public synthetic fixtures · raw score units · lower is better on both axes", fontsize=12, color=INK)
    labels = ["Keep\nparent", "Target\nonly", "Soft\nretention", "Protected\nedit"]
    for axis, key, title, threshold in zip(axes,
            ("maximum_target_shift_error", "maximum_protected_score_drift"),
            ("Target score-shift error", "Protected score drift"), (.05, 1e-12)):
        values = [row["admission"]["metrics"][key] for row in bundle["candidates"]]
        axis.set_facecolor(PAPER)
        bars = axis.bar(labels, values, width=.6, color=["#9babae", RED, "#b89759", TEAL])
        axis.set_title(title, loc="left", fontsize=14, color=INK, pad=12)
        axis.set_ylabel("Maximum absolute difference (score units)", color=INK, fontsize=10)
        axis.set_ylim(0, max(values)*1.25)
        axis.axhline(threshold, color=INK, linewidth=1, linestyle="--")
        axis.text(.99, .96, "Acceptance limit: " + ("0.05" if key.startswith("maximum_target") else "1e−12"),
                  transform=axis.transAxes, ha="right", fontsize=10, color=INK)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x()+bar.get_width()/2, value+max(values)*.035,
                      f"{value:.6f}", ha="center", va="bottom", fontsize=10, color=INK)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#b6c1c2")
        axis.tick_params(colors=INK)
    figure.text(.06, .06, "Only the protected edit meets the full public contract. Zero drift alone would also admit the unchanged parent unless target utility is required.",
                 fontsize=10, color=INK)
    save(figure, "score-tradeoff")


def main() -> None:
    path = OUTPUT / "policy-change/bundle.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "enterprise-routing/1"})
    workflow()
    tradeoff(bundle)
    source = {"schema": "enterprise-routing-visual-source/1", "source": "policy-change/bundle.json",
              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "renderer": "scripts/render_enterprise_routing.py", "matplotlib_version": matplotlib.__version__,
              "scope": "Original workflow diagram and measured public synthetic score differences",
              "metric_fields": ["maximum_target_shift_error", "maximum_protected_score_drift"],
              "mermaid_used": False}
    (OUTPUT / "visual-source.json").write_bytes(
        (json.dumps(source, indent=2, sort_keys=True) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
