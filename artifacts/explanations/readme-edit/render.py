"""Render the README's exact linear illustration from its adjacent source.json.

Run with the existing documentation environment (Matplotlib required):
    python artifacts/explanations/readme-edit/render.py

The SVG retains editable text. This draws an illustration; it runs no study.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
PALETTE = {
    "paper": "#F8F7F3",
    "ink": "#23312F",
    "muted": "#5A6662",
    "grid": "#E3E5DD",
    "target": "#A45739",
    "protect": "#22776C",
    "axis": "#A1AAA2",
}


def main() -> None:
    source = json.loads((ROOT / "source.json").read_text(encoding="utf-8"))
    model, updates = source["model"], source["updates"]
    # This figure is a view of the supplied two-coordinate model, not a solver.
    if model != {
        "equation": "y = w_1*x_1 + w_2*x_2",
        "parent_weights": [1, 1],
        "target_input": [1, 1],
        "target_change": 1,
        "protected_input": [1, 0],
        "target_condition": "delta_1 + delta_2 = 1",
        "protection_condition": "delta_1 = 0",
    }:
        raise ValueError("Revise the geometry and labels before changing this model.")
    if updates != {"parent": [0, 0], "target_only": [0.5, 0.5], "protected": [0, 1]}:
        raise ValueError("Revise the labels before changing the supplied updates.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "wspf-readme-edit-geometry-v1",
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "figure.facecolor": PALETTE["paper"],
        "savefig.facecolor": PALETTE["paper"],
    })
    fig = plt.figure(figsize=(10, 5.3), dpi=200)
    fig.text(0.055, 0.91, source["title"], fontsize=25, weight="bold")
    fig.text(0.055, 0.855, "A small linear model, drawn in weight-update space.",
             fontsize=12.8, color=PALETTE["muted"])

    fig.text(0.055, 0.70, "CHANGE THE TARGET", fontsize=10.4,
             weight="bold", color=PALETTE["target"])
    fig.text(0.055, 0.645, "Raise the output at (1, 1) by 1.", fontsize=13.2)
    fig.text(0.055, 0.575, "Δw₁ + Δw₂ = 1", fontsize=21, color=PALETTE["target"])

    fig.text(0.055, 0.43, "KEEP THE PROTECTED OUTPUT", fontsize=10.4,
             weight="bold", color=PALETTE["protect"])
    fig.text(0.055, 0.375, "Leave the output at (1, 0) unchanged.", fontsize=13.2)
    fig.text(0.055, 0.305, "Δw₁ = 0", fontsize=21, color=PALETTE["protect"])

    fig.text(0.055, 0.17, "The intersection satisfies both conditions.",
             fontsize=12, weight="normal")

    ax = fig.add_axes((0.52, 0.21, 0.40, 0.62), facecolor=PALETTE["paper"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(-0.2, 1.3)
    ax.set_xticks([0, 0.5, 1], labels=["0", "0.5", "1"])
    ax.set_yticks([0.5, 1], labels=["0.5", "1"])
    ax.tick_params(axis="both", length=0, pad=8, labelsize=10.5,
                   colors=PALETTE["muted"])
    ax.grid(color=PALETTE["grid"], linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_position("zero")
        ax.spines[side].set_color(PALETTE["axis"])
        ax.spines[side].set_linewidth(0.9)
    ax.set_xlabel("Δw₁", fontsize=15, labelpad=0)
    ax.xaxis.set_label_coords(1.02, 0.13)
    ax.set_ylabel("Δw₂", fontsize=15, rotation=0, labelpad=0)
    ax.yaxis.set_label_coords(0.13, 1.015)

    ax.plot([-0.16, 1.16], [1.16, -0.16], color=PALETTE["target"],
            linewidth=1.8, zorder=2)
    ax.plot([0, 0], [-0.16, 1.22], color=PALETTE["protect"],
            linewidth=1.8, linestyle=(0, (3, 3)), zorder=2)

    for key, color in (("target_only", PALETTE["target"]),
                       ("protected", PALETTE["protect"])):
        ax.annotate("", xy=updates[key], xytext=updates["parent"],
                    arrowprops={"arrowstyle": "-|>", "color": color,
                                "linewidth": 2.0, "mutation_scale": 12,
                                "shrinkA": 7, "shrinkB": 7}, zorder=3)

    for key, marker, color in (("parent", "o", PALETTE["ink"]),
                               ("target_only", "s", PALETTE["target"]),
                               ("protected", "D", PALETTE["protect"])):
        ax.scatter(*updates[key], s=62, marker=marker, facecolor=color,
                   edgecolor=PALETTE["paper"], linewidth=1.4, zorder=4)

    ax.annotate("No edit", xy=updates["parent"], xytext=(12, -23),
                textcoords="offset points", fontsize=11, color=PALETTE["muted"])
    ax.annotate("Target only\n(0.5, 0.5)", xy=updates["target_only"], xytext=(15, 5),
                textcoords="offset points", fontsize=11.5, linespacing=1.5,
                color=PALETTE["target"])
    ax.annotate("Both conditions\n(0, 1)", xy=updates["protected"], xytext=(14, 8),
                textcoords="offset points", fontsize=11.5, linespacing=1.5,
                color=PALETTE["protect"])

    fig.add_artist(Line2D([0.055, 0.945], [0.105, 0.105],
                          transform=fig.transFigure, color=PALETTE["grid"], linewidth=1))
    fig.text(0.055, 0.052, "ILLUSTRATION", fontsize=9.4, weight="bold",
             color=PALETTE["muted"])
    fig.text(0.22, 0.052, "Fixed features  ·  No bias change  ·  Exact arithmetic",
             fontsize=10.5, color=PALETTE["muted"])

    png, svg = ROOT / "edit-geometry.png", ROOT / "edit-geometry.svg"
    fig.savefig(png, dpi=200, metadata={"Title": source["title"]})
    fig.savefig(svg, metadata={"Date": None, "Title": source["title"],
                               "Description": source["evidence_type"]})
    plt.close(fig)

    # Include a readable description in the standalone SVG as well as README alt text.
    svg_text = svg.read_text(encoding="utf-8")
    start = svg_text.index(">", svg_text.index("<svg")) + 1
    description = (
        '<title id="figure-title">One edit, two obligations</title>'
        '<desc id="figure-description">Exact linear illustration with fixed features '
        'and no bias change. In weight-update coordinates, the target condition is '
        'delta w1 plus delta w2 equals 1. The protection condition is delta w1 equals 0. '
        'The target-only edit is (0.5, 0.5); the intersection (0, 1) meets both conditions. '
        'The starting point is no edit (0, 0). Both axes use the same scale.</desc>'
    )
    svg_text = svg_text[:start] + description + svg_text[start:]
    svg_text = svg_text.replace('<svg ', '<svg role="img" aria-labelledby="figure-title figure-description" ', 1)
    svg_text = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
    ET.fromstring(svg_text)
    svg.write_text(svg_text, encoding="utf-8", newline="\n")
    print("Rendered the README illustration as PNG and SVG.")


if __name__ == "__main__":
    main()
