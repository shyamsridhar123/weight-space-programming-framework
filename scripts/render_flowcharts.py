#!/usr/bin/env python3
"""Render the three generic framework flowcharts without a Mermaid dependency.

Run from the standalone repository root::

    python scripts/render_flowcharts.py

Requires Matplotlib. The graph definitions, labels, geometry, and source-diagram
hashes are all editable below. SVG labels remain text. Fixed SVG IDs and omitted
timestamps make exports byte-stable with the same renderer and dependencies.
Only docs/images outputs (or an explicit --output-dir) are written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath


PALETTE = {
    "ink": "#182C3B",
    "muted": "#526575",
    "edge": "#63798A",
    "rule": "#D8E1E8",
    "standard": ("#F1F5F9", "#62788B", "#182C3B"),
    "proposal": ("#EEF3FB", "#5577A6", "#203E64"),
    "caller": ("#EAF5F0", "#397D68", "#205443"),
    "receipt": ("#173E50", "#173E50", "#FFFFFF"),
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "svg.fonttype": "none",
        "svg.hashsalt": "weight-space-programming-framework-flowcharts-v1",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "path.simplify": False,
    }
)


@dataclass(frozen=True)
class Node:
    key: str
    text: str
    x: float
    y: float
    width: float
    height: float = 0.9
    kind: str = "standard"

    @property
    def label(self) -> str:
        return " ".join(self.text.split())

    def port(self, side: str) -> tuple[float, float]:
        return {
            "top": (self.x, self.y + self.height / 2),
            "bottom": (self.x, self.y - self.height / 2),
            "left": (self.x - self.width / 2, self.y),
            "right": (self.x + self.width / 2, self.y),
        }[side]


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    start: str = "bottom"
    end: str = "top"
    via: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Diagram:
    stem: str
    title: str
    subtitle: str
    height: float
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    footer: tuple[str, ...]
    source_document: str
    source_fence_line: int
    source_diagram_sha256: str
    alt_text: str


DIAGRAMS = (
    Diagram(
        stem="task-lifecycle",
        title="A generic task lifecycle",
        subtitle="Read, diagnose, edit, and create under declared access and budgets.",
        height=16.5,
        nodes=(
            Node("T", "Task, semantics, access,\nbudgets, splits", 7, 14.35, 5.8, 0.95),
            Node("S", "Programmer session", 7, 12.95, 4.0, 0.8),
            Node("A", "Read or diagnose:\nanswer candidate", 2.5, 11.3, 3.8, 1.0, "proposal"),
            Node("P", "Edit: structured or\nliteral program", 7, 11.3, 3.8, 1.0, "proposal"),
            Node("C", "Create:\ncomplete artifact", 11.5, 11.3, 3.8, 1.0, "proposal"),
            Node("L", "Lower, project,\nretain actual endpoint", 7, 9.65, 4.2, 1.0, "proposal"),
            Node("V", "Caller checks structure\nand required behavior", 7, 7.95, 5.8, 1.0, "caller"),
            Node("D", "Validation, declared selection,\nor abstention", 7, 6.3, 5.8, 1.0, "caller"),
            Node("F", "Seal selected identity", 7, 4.8, 4.6, 0.8),
            Node("E", "Held-out evaluation", 7, 3.4, 4.6, 0.8, "caller"),
            Node("R", "Receipt, complete history,\nindependent replay", 7, 1.85, 6.0, 1.0, "receipt"),
        ),
        edges=(
            Edge("T", "S"),
            Edge("S", "A", via=((7, 12.1), (2.5, 12.1))),
            Edge("S", "P"),
            Edge("S", "C", via=((7, 12.1), (11.5, 12.1))),
            Edge("P", "L"),
            Edge("L", "V"),
            Edge("A", "V", end="left", via=((2.5, 7.95),)),
            Edge("C", "V", end="right", via=((11.5, 7.95),)),
            Edge("V", "D"),
            Edge("D", "F"),
            Edge("F", "E"),
            Edge("E", "R"),
        ),
        footer=(
            "Caller responsibility: structural and behavioral checks, selection, and held-out evaluation.",
            "Retain all attempts, observations, refusals, failures, and decisions.",
        ),
        source_document="docs/weight-space/v2/README.md",
        source_fence_line=32,
        source_diagram_sha256="eae3711068d95ba8110bef94b4ac84773365504968ade3f77482272356ef820f",
        alt_text=(
            "Generic task lifecycle. A declared task enters a programmer session and branches "
            "to an answer candidate, an edit program, or a complete artifact. Edit programs "
            "are lowered and projected while retaining the actual endpoint. All three routes "
            "join caller checks of structure and required behavior, then validation and declared "
            "selection or abstention, sealing the selected identity, held-out evaluation, and "
            "a receipt with complete history and independent replay."
        ),
    ),
    Diagram(
        stem="architecture",
        title="Framework architecture",
        subtitle="Explicit proposal lowering, session history, and caller-owned evaluation.",
        height=13.65,
        nodes=(
            Node("P", "StructuredProgram", 2.7, 11.4, 3.7, 0.85, "proposal"),
            Node("L", "Explicit lowering", 7, 11.4, 3.6, 0.85, "proposal"),
            Node("I", "Literal Program", 11.3, 11.4, 3.7, 0.85, "proposal"),
            Node("T", "TaskSpec, access,\nbudgets", 2.7, 8.95, 3.7, 1.0),
            Node("S", "ProgrammerSession", 7, 8.95, 3.8, 0.85),
            Node("A", "Answer or\ncomplete artifact", 11.3, 8.95, 3.7, 1.0, "proposal"),
            Node("C", "Candidate history", 7, 7.55, 4.4, 0.8),
            Node("V", "Caller checks\nand selection", 7, 6.1, 5.5, 1.0, "caller"),
            Node("F", "Seal candidate or abstain", 7, 4.65, 5.5, 0.8),
            Node("E", "Caller-owned\nheld-out evaluator", 7, 3.2, 5.5, 1.0, "caller"),
            Node("R", "EvaluationReceipt", 7, 1.8, 5.0, 0.85, "receipt"),
        ),
        edges=(
            Edge("T", "S", start="right", end="left"),
            Edge("P", "L", start="right", end="left"),
            Edge("L", "I", start="right", end="left"),
            Edge("I", "S", via=((11.3, 9.95), (7, 9.95))),
            Edge("A", "S", start="left", end="right"),
            Edge("S", "C"),
            Edge("C", "V"),
            Edge("V", "F"),
            Edge("F", "E"),
            Edge("E", "R"),
        ),
        footer=(
            "Caller responsibility: run required checks and retain structured proposals and lowering receipts.",
            "Submitting a literal program does not automatically retain those files or certify a candidate.",
        ),
        source_document="docs/weight-space/v2/architecture.md",
        source_fence_line=20,
        source_diagram_sha256="164355cd33cadc2fbf4fb01295493e89f27f105dd2b35e28fa263fab50d20fc5",
        alt_text=(
            "Generic framework architecture. A StructuredProgram passes through explicit lowering "
            "to a literal Program. The literal Program, TaskSpec with access and budgets, and an "
            "answer or complete artifact feed ProgrammerSession. Candidate history proceeds to "
            "caller checks and selection, sealing a candidate or abstaining, a caller-owned "
            "held-out evaluator, and EvaluationReceipt. Callers must run checks and retain "
            "structured proposals and lowering receipts."
        ),
    ),
    Diagram(
        stem="evidence-branches",
        title="Three evidence branches",
        subtitle="Structural admission, semantic certification, and empirical evaluation.",
        height=12.1,
        nodes=(
            Node("S", "Examples, rules, optimizer,\nor LLM proposal", 3.65, 9.75, 5.7, 1.0, "proposal"),
            Node("A", "Named tensor artifact\nand interface metadata", 10.35, 9.75, 5.7, 1.0),
            Node("P", "Program with exact parent\nand cell preconditions", 7, 8.0, 6.4, 1.0, "proposal"),
            Node("C", "Candidate artifact", 7, 6.45, 4.4, 0.85),
            Node("T", "Structural policy", 2.5, 4.65, 3.8, 0.9),
            Node("M", "Explicit\narchitecture adapter", 7, 4.65, 3.8, 1.0, "proposal"),
            Node("B", "Supported\nsemantic contract", 7, 3.0, 3.8, 1.0, "proposal"),
            Node("E", "Caller evaluator\nand held-out\ntask trials", 11.5, 4.65, 3.8, 1.25, "caller"),
            Node("R", "Recorded identities\nand receipts", 7, 1.4, 6.4, 1.0, "receipt"),
        ),
        edges=(
            Edge("S", "P", via=((3.65, 8.85), (7, 8.85))),
            Edge("A", "P", via=((10.35, 8.85), (7, 8.85))),
            Edge("P", "C"),
            Edge("C", "T", via=((7, 5.65), (2.5, 5.65))),
            Edge("C", "M"),
            Edge("M", "B"),
            Edge("C", "E", via=((7, 5.65), (11.5, 5.65))),
            Edge("T", "R", end="left", via=((2.5, 1.4),)),
            Edge("B", "R"),
            Edge("E", "R", end="right", via=((11.5, 1.4),)),
        ),
        footer=("Each branch establishes a different claim; none substitutes for another.",),
        source_document="docs/weight-space/design.md",
        source_fence_line=15,
        source_diagram_sha256="03657c937ea85a581ef7462dc1de514049c4142a14da3fe708d75cdd8e405f24",
        alt_text=(
            "Three distinct evidence branches. Examples, rules, an optimizer, or an LLM proposal "
            "and a named tensor artifact with interface metadata feed a program with exact-parent "
            "and cell preconditions. Its candidate artifact branches to a structural policy, an "
            "explicit architecture adapter followed by a supported semantic contract, and a caller "
            "evaluator with held-out task trials. All branches feed recorded identities and "
            "receipts. Evidence from one branch does not substitute for another."
        ),
    ),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def draw_diagram(diagram: Diagram, dpi: int) -> tuple[dict[str, bytes], list[dict]]:
    width = 14.0
    figure_width = 12.0
    figure, axes = plt.subplots(figsize=(figure_width, figure_width * diagram.height / width))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axes.set(xlim=(0, width), ylim=(0, diagram.height), aspect="equal")
    axes.axis("off")

    axes.text(0.7, diagram.height - 0.42, "GENERIC WEIGHT-SPACE PROGRAMMING", color=PALETTE["muted"], fontsize=9.5, weight="bold", ha="left", va="top")
    axes.text(0.7, diagram.height - 0.82, diagram.title, color=PALETTE["ink"], fontsize=25, weight="bold", ha="left", va="top")
    axes.text(0.7, diagram.height - 1.42, diagram.subtitle, color=PALETTE["muted"], fontsize=11.5, ha="left", va="top")

    nodes = {node.key: node for node in diagram.nodes}
    if len(nodes) != len(diagram.nodes):
        raise ValueError(f"Duplicate node ID in {diagram.stem}")
    rendered_edges = []
    for edge in diagram.edges:
        points = [nodes[edge.source].port(edge.start), *edge.via, nodes[edge.target].port(edge.end)]
        arrow = FancyArrowPatch(
            path=PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1)),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
            color=PALETTE["edge"], capstyle="round", joinstyle="round", zorder=2,
        )
        arrow.set_gid(f"edge-{edge.source}-{edge.target}")
        axes.add_patch(arrow)
        rendered_edges.append({"source": edge.source, "target": edge.target, "points": points})

    labels = []
    for node in diagram.nodes:
        fill, border, ink = PALETTE[node.kind]
        patch = FancyBboxPatch(
            (node.x - node.width / 2, node.y - node.height / 2),
            node.width, node.height, boxstyle="round,pad=0,rounding_size=0.11",
            linewidth=1.25, facecolor=fill, edgecolor=border, zorder=3,
        )
        patch.set_gid(f"node-{node.key}")
        axes.add_patch(patch)
        label = axes.text(node.x, node.y, node.text, color=ink, fontsize=14,
                          weight="normal", ha="center", va="center", linespacing=1.28, zorder=4)
        label.set_gid(f"label-{node.key}")
        labels.append((node, label))

    footer_top = 0.78 if len(diagram.footer) > 1 else 0.62
    axes.plot((0.7, 13.3), (footer_top, footer_top), color=PALETTE["rule"], linewidth=0.8)
    for index, line in enumerate(diagram.footer):
        axes.text(0.7, footer_top - 0.17 - index * 0.26, line, color=PALETTE["muted"],
                  fontsize=10.4, ha="left", va="top")

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for node, label in labels:
        extent = label.get_window_extent(renderer).transformed(axes.transData.inverted())
        if (extent.x0 < node.x - node.width / 2 + 0.12
                or extent.x1 > node.x + node.width / 2 - 0.12
                or extent.y0 < node.y - node.height / 2 + 0.08
                or extent.y1 > node.y + node.height / 2 - 0.08):
            raise ValueError(f"Label does not fit: {diagram.stem}/{node.key}")

    outputs = {}
    for extension in ("png", "svg"):
        buffer = io.BytesIO()
        metadata = {"Software": "Weight-space programming framework flowchart renderer"} if extension == "png" else {"Date": None, "Creator": "Weight-space programming framework flowchart renderer", "Title": diagram.title, "Description": diagram.alt_text}
        figure.savefig(buffer, format=extension, dpi=dpi, metadata=metadata)
        outputs[extension] = buffer.getvalue()
    plt.close(figure)
    return outputs, rendered_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "images")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution; default: 220")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")

    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "wsp-flowchart-images/1",
        "purpose": "Generic framework diagrams; no experiments, audits, tests, or proof builds are run.",
        "renderer": {
            "path": "scripts/render_flowcharts.py",
            "sha256": sha256(Path(__file__).read_bytes()),
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "font": "DejaVu Sans",
            "png_dpi": args.dpi,
            "svg_labels": "editable text",
            "determinism": "Fixed coordinates, font, SVG hash salt, and metadata; no timestamps. Byte identity assumes matching renderer and dependencies.",
        },
        "source_hash_convention": "SHA-256 of the original fenced diagram body, excluding fences, with LF newlines and its trailing newline retained.",
        "diagrams": [],
    }
    for diagram in DIAGRAMS:
        outputs, edge_geometry = draw_diagram(diagram, args.dpi)
        record = {
            "id": diagram.stem,
            "title": diagram.title,
            "original_source_document": diagram.source_document,
            "original_source_fence_start_line": diagram.source_fence_line,
            "original_source_diagram_sha256": diagram.source_diagram_sha256,
            "alt_text": diagram.alt_text,
            "caller_and_scope_notes": list(diagram.footer),
            "nodes": [{"id": node.key, "label": node.label} for node in diagram.nodes],
            "edges": edge_geometry,
            "node_count": len(diagram.nodes),
            "edge_count": len(diagram.edges),
            "outputs": [],
        }
        for extension, data in outputs.items():
            path = destination / f"{diagram.stem}.{extension}"
            path.write_bytes(data)
            output = {"file": path.name, "bytes": len(data), "sha256": sha256(data)}
            if extension == "png":
                pixel_width, pixel_height = struct.unpack(">II", data[16:24])
                output.update({"width_px": pixel_width, "height_px": pixel_height})
            record["outputs"].append(output)
        manifest["diagrams"].append(record)
    manifest_path = destination / "flowchart-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output_directory": str(destination), "diagrams": [{"id": item["id"], "nodes": item["node_count"], "edges": item["edge_count"]} for item in manifest["diagrams"]], "manifest": manifest_path.name}, indent=2))


if __name__ == "__main__":
    main()
