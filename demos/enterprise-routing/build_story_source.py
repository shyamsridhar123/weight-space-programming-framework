"""Bind the simple visual story to the published synthetic scenario results.

This checks the selected candidate and its actual inverse. It generates the
portable source register and subtitle files; it does not generate video frames.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rlcd.weight_space import TensorArtifact, apply_program
from rlcd.weight_space.program import Program

OUT = ROOT / "artifacts/explanations/enterprise-story"
EVIDENCE = ROOT / "artifacts/enterprise/support-routing"

def write(name: str, value: str):
    (OUT / name).write_bytes((value.rstrip() + "\n").encode("utf-8"))

def stamp(seconds: float, separator="."):
    millis = round(seconds * 1000)
    return f"{millis//3600000:02}:{millis//60000%60:02}:{millis//1000%60:02}{separator}{millis%1000:03}"

def main():
    bundles = {scenario: json.loads((EVIDENCE / scenario / "bundle.json").read_text())
               for scenario in ("policy-change", "conflict", "coverage-gap")}
    b = bundles["policy-change"]
    candidate = lambda label: next(c for c in b["candidates"] if c["label"] == label)
    protected, naive = candidate("protected-edit"), candidate("target-only")
    restored = apply_program(TensorArtifact.from_dict(protected["candidate"]["artifact"]),
                             Program.from_dict(b["rollback"]["program"]))
    assert restored.sha256 == b["final_admission"]["parent_sha256"]
    assert b["final_admission"]["decision"] == "admitted_for_synthetic_trial"
    assert bundles["conflict"]["final_admission"]["decision"] == "abstained"
    assert bundles["coverage-gap"]["final_admission"]["decision"] == "rejected_after_holdout"
    assert b["session"]["result"]["maximum_protected_score_drift"] == 0
    assert all(t["after_queue"] == "platform" for t in b["session"]["result"]["tickets"] if t["role"] == "target")
    weights = protected["candidate"]["artifact"]["tensors"]["head.weight"]
    base = b["parent"]["tensors"]["head.weight"]
    naive_weights = naive["candidate"]["artifact"]["tensors"]["head.weight"]
    ticket = lambda record, identifier: next(t for t in record["admission"]["metrics"]["tickets"] if t["ticket_id"] == identifier)
    model = {"target": ticket(protected, "public-outage-standard"),
             "ordinaryNaive": ticket(naive, "public-general"),
             "weightBefore": base[3][6], "weightAfter": weights[3][6],
             "naiveShared": naive_weights[3][0] - base[3][0],
             "naiveOutage": naive_weights[3][6] - base[3][6],
             "protectedDrift": b["session"]["result"]["maximum_protected_score_drift"],
             "decision": b["final_admission"]["decision"],
             "conflictDecision": bundles["conflict"]["final_admission"]["decision"],
             "gapDecision": bundles["coverage-gap"]["final_admission"]["decision"],
             "restored": True, "parent": restored.sha256,
             "finalReceipt": b["final_admission"]["sha256"]}
    source_paths = ["examples/enterprise_support_routing.py", "examples/data/enterprise_support_tickets.json"] + [
        f"artifacts/enterprise/support-routing/{s}/bundle.json" for s in bundles]
    sources = [{"id": f"S{i+1}", "path": path,
                "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
               for i, path in enumerate(source_paths)]
    claims = [
        {"id":"C1", "kind":"observation", "text":"The public standard outage routes to General Support before the edit and Platform afterwards.", "sources":["S3"], "locator":"candidates[label=protected-edit].admission.metrics.tickets[ticket_id=public-outage-standard]", "beats":[1,2,6]},
        {"id":"C2", "kind":"observation", "text":"The recognized outage feature is zero on supplied protection anchors.", "sources":["S1","S2","S3"], "locator":"featurize; anchors; public records with role=protected", "beats":[3,5]},
        {"id":"C3", "kind":"observation", "text":"Target-only fitting changes the ordinary-support Platform score, while its route stays General Support.", "sources":["S3"], "locator":"candidates[label=target-only].admission.metrics.tickets[ticket_id=public-general]", "beats":[4]},
        {"id":"C4", "kind":"inference", "text":"The 2D weight slice shows a permitted zero-shared-coefficient direction; the complete compiler enforces additional feature and bias constraints.", "sources":["S1","S3"], "locator":"compile_feature_edit retention=nullspace; protected-edit artifact and retention_certificate", "beats":[5]},
        {"id":"C5", "kind":"observation", "text":"The selected program changes only head.weight [3,6], from 0.25 to 3.22029702970297.", "sources":["S3"], "locator":"candidates[label=protected-edit].program.edits", "beats":[6]},
        {"id":"C6", "kind":"observation", "text":"Recognized final targets route to Platform; maximum sampled protected drift is zero.", "sources":["S3"], "locator":"session.result; final_admission", "beats":[7]},
        {"id":"C7", "kind":"proposal", "text":"A direct route could avoid a transfer. Handling time and savings are not measured.", "sources":["S1","S2"], "locator":"contract.objective; synthetic scope", "beats":[1,7]},
        {"id":"C8", "kind":"observation", "text":"The conflict scenario abstains and the coverage-gap scenario rejects after final evaluation.", "sources":["S4","S5"], "locator":"final_admission.decision; inputs; featurize", "beats":[8]},
        {"id":"C9", "kind":"observation", "text":"Applying the bound inverse restores the exact parent artifact; external ticket assignments are outside scope.", "sources":["S3"], "locator":"rollback.program; rollback.restored_parent_sha256; rollback.scope", "beats":[9]},
        {"id":"C10", "kind":"observation", "text":"The example uses synthetic structured fields and a hand-constructed linear head, with exact retention scoped to fixed-feature real-valued semantics.", "sources":["S1","S2","S3"], "locator":"module docstring; model_metadata; retention_certificate; contract.execution", "beats":[1,2,3,4,5,6,7,8,9]}
    ]
    source = {"schema":"enterprise-story-source/1", "audience":"A non-specialist evaluating the framework's purpose",
              "question":"How can a weight edit route outages to the right team without disturbing protected support behavior?",
              "sources":sources, "claims":claims, "model":model,
              "visual_mapping":{"bar_length":"queue score in raw units, common scale 0 to 4", "dots":"presentation of a contribution or an edit, not measured latency", "weight_plane":"schematic two-coefficient slice, not the entire feasible space", "transfer_arrow":"possible handoff that the proposed direct route could avoid"},
              "style_reference":{"repository":"shyamsridhar123/STE-Pro-Max", "commit":"a7da74e83e5fcf8ea23dce075e64b463e5e24584", "skills":["ste-promax","ste-storytelling","ste-line-studio"], "method":"Host-authored original SVG/HTML; optional STE engine validators were not run"},
              "scope":"Synthetic structured data and supplied linear head. No customer deployment, measured savings, trained text understanding, floating-point proof, process isolation proof or independent attestation."}
    write("source.json", json.dumps(source, indent=2, ensure_ascii=False))
    write("story-data.js", "'use strict';\n// Generated from the source register by build_story_source.py.\nglobalThis.STORY_DATA = " + json.dumps(model, indent=2).replace("<", "\\u003c") + ";")
    js = (OUT / "story.js").read_text(encoding="utf-8")
    cues = ast.literal_eval(re.search(r"const cues = (\[.*?\]);", js, re.S).group(1))
    assert cues[0][0] == 0 and cues[-1][1] == 90
    assert all(left[1] == right[0] for left, right in zip(cues, cues[1:]))
    vtt, srt = ["WEBVTT", ""], []
    for i, (start, end, subtitle) in enumerate(cues, 1):
        assert end > start
        vtt += [f"{stamp(start)} --> {stamp(end)}", subtitle, ""]
        srt += [str(i), f"{stamp(start, ',')} --> {stamp(end, ',')}", subtitle, ""]
    write("subtitles.vtt", "\n".join(vtt))
    write("subtitles.srt", "\n".join(srt))
    prose = ["# An outage. The right team.", "", "A 90-second visual explanation of the Weight Space Programming Framework through enterprise support routing.", "", "The subtitles below are captured in the browser. There is no spoken narrator. The music is a separate quiet instrumental track.", ""]
    for start, end, subtitle in cues:
        prose += [f"**{stamp(start)[3:8]}–{stamp(end)[3:8]}** {subtitle}", ""]
    prose += ["## Evidence and limits", "", "All scores and edits come from the scenario bundles linked in [the source register](source.json). The bar chart follows one public standard outage record. The final result summarizes the declared synthetic final split. A possible avoided handoff is an operational opportunity, not a measured business outcome.", "", "The two-dimensional weight picture is a schematic slice. The compiler handles the remaining feature and bias constraints. Its numerical fit uses ridge regularization, so the target shift is approximate. The exact retention certificate covers the supplied fixed-feature anchors and their span under real-valued linear semantics; it does not prove floating-point execution or text understanding.", "", "The interactive page starts paused. Use Play, chapter arrows, the time slider, and Reset. The local server recomputes all scenarios and applies the inverse before playback. A portable static page uses the recorded, verified source model. Readable prose and the understanding check remain available without JavaScript.", "", "The explanation follows STE-Pro-Max's concrete-case-first storytelling and visual guidance at commit `a7da74e83e5fcf8ea23dce075e64b463e5e24584`. Its figures and code are original. No Mermaid is used."]
    write("story.md", "\n".join(prose))
    print("Bound story scores and inverse verified; continuous 90-second VTT/SRT generated")

if __name__ == "__main__":
    main()
