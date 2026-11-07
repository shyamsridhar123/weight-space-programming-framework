"""Index retained source bytes and selected reported values; never run release code.

This standard-library script reads archived text/JSON and local documentation.
It writes only review-evidence.json beside itself. It does not retrieve sources,
import the released harness, reconstruct checkpoints, or run an experiment.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[2]
REVIEW = ROOT / "docs/weight-space/v2/source-review.md"
OUTPUT = ARCHIVE / "review-evidence.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


manifest = read_json(ARCHIVE / "provenance.json")
records = manifest["records"]
retrieved = [r for r in records if r["status"] == "retrieved"]
failed = [r for r in records if r["status"] != "retrieved"]
assert len(records) == 84 and len(retrieved) == 77 and len(failed) == 7
assert sum(r["local_path"].startswith("release/") for r in retrieved) == 75
for r in retrieved:
    path = ARCHIVE / r["local_path"]
    assert path.stat().st_size == r["bytes"], r["local_path"]
    assert sha(path) == r["sha256"], r["local_path"]

references = read_json(ROOT / "docs/weight-space/references.json")
old_sources = {r["id"]: r for r in references["sources"]}
comparisons = []
for r in old_sources["SparksLogs"]["reviewed_artifacts"]:
    path = "release/" + r["path"]
    actual = sha(ARCHIVE / path)
    assert actual == r["sha256"], path
    comparisons.append({"path": path, "reference_id": r["id"],
                        "sha256": actual, "matches_prior_hash": True})
assert len(comparisons) == 13
article_sha = sha(ARCHIVE / "article.html")
assert article_sha == old_sources["Sparks"]["reviewed_artifact"]["sha256"]

tree_route = None
for body in re.findall(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>',
                       (ARCHIVE / "repository-tree.html").read_text(encoding="utf-8"), re.S):
    try:
        candidate = json.loads(body).get("payload", {}).get("codeViewTreeRoute")
    except (ValueError, AttributeError):
        continue
    if candidate is not None:
        tree_route = candidate
        break
assert tree_route is not None
assert tree_route["refInfo"]["currentOid"] == manifest["release_commit"]
assert tree_route["tree"]["totalCount"] == 11

observations = []
inspected_json = set()


def release_json(path):
    inspected_json.add("release/" + path)
    return read_json(ARCHIVE / "release" / path)


def observe(path, locator, value, note=None):
    record = {"source": "release/" + path, "locator": locator, "value": value}
    if note:
        record["interpretation_boundary"] = note
    observations.append(record)


for experiment, count in [("exp1_which_game", 4), ("exp2_diffusion", 3)]:
    path = experiment + "/eval/scores.json"
    rows = release_json(path)
    assert len(rows) == count
    for i, row in enumerate(rows):
        assert row["correct"] and row["tool_free"]
        for field in ["id", "true", "guess", "correct", "tool_free"]:
            observe(path, f"/{i}/{field}", row[field])
    if count == 4:
        assert "frogger" in rows[0]["answer_line"]
        observe(path, "/0/answer_line", "contains frogger",
                "Released scoring accepts Frogger for Freeway; see experiment README lines 21 and 33.")
    else:
        assert all(any(term in row["answer_line"] for term in ["hollow", "outline"]) for row in rows)
        observe(path, "/*/answer_line", "all three describe hollow/outline shapes",
                "Filled targets are stated in README lines 8 and 32, not an explicit answer-key field.")

path = "exp2_diffusion/eval/answer_key.json"
for key, row in release_json(path).items():
    observe(path, f"/{key}/shape", row["shape"])
    observe(path, f"/{key}/quality/intended", row["quality"]["intended"])

path = "exp3_language/eval/scores.json"
rows = release_json(path)
assert len(rows) == 6 and len({r["net"] for r in rows}) == 3
for i, row in enumerate(rows):
    assert row["corpus_correct"] and row["order"] == row["order_guess"] == "as is"
    for field in ["net", "corpus_correct", "order", "order_guess", "true_copy_heads", "copy_named"]:
        observe(path, f"/{i}/{field}", row[field])
heads_path = "exp3_language/eval/heads.json"
heads = release_json(heads_path)
for corpus, names in {"python": ["L2H1", "L3H1", "L3H5"],
                      "romaji": ["L3H3"], "english": ["L3H1"]}.items():
    net = corpus + "_L4_H6_d384_s0"
    for head in names:
        observe(heads_path, f"/{net}/{head}/induct", heads[net][head]["induct"])
for row in rows:
    assert sorted(row["true_copy_heads"]) == sorted(
        h for h, data in heads[row["net"]].items() if data["induct"] > 0.3)

path = "exp4_fix_car/eval/oneshot_result.json"
d = release_json(path)
for group, fields in {"edit_report": ["statements", "applied", "entries_written", "errors", "cut_off"],
                      "changes": ["changed", "entries", "frac_changed"],
                      "test": ["success", "episodes"], "base": ["success"]}.items():
    for field in fields:
        observe(path, f"/{group}/{field}", d[group][field])
assert d["changes"]["changed"] == 6073 and d["test"]["episodes"] == 1024
assert d["test"]["success"] == 1.0 and d["base"]["success"] == 0.0

path = "exp5_write_from_scratch/eval/minatar_eval.json"
for game, rows in release_json(path).items():
    best = max((key for key in rows if key.startswith("ppo500_")), key=lambda key: rows[key]["mean"])
    for key in ["hand", best]:
        for field in ["mean", "episodes"]:
            observe(path, f"/{game}/{key}/{field}", rows[key][field],
                    "PPO comparator is the best retained 500M-step seed by mean reward.")

path = "exp6_go_from_scratch/eval/budget_0/eval_submission.json"
d = release_json(path)
for key in ["ppo_s0", "ppo_s2"]:
    for field in ["games", "score", "wins", "losses"]:
        observe(path, f"/{key}/{field}", d[key][field])
path = "exp6_go_from_scratch/eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json"
for key, d in release_json(path).items():
    observe(path, f"/{key}/score", d["score"])

margin_values = []
for run in range(1, 4):
    path = f"exp7_sycophancy/eval/weights_only_{run}/patch_score.json"
    d = release_json(path)
    assert len(d) == 3
    for pair, row in d.items():
        assert row["astra"]["gap_closed_h2s"] == 0.0
        for field in ["gap_closed_h2s", "margin_closed_h2s"]:
            observe(path, f"/{pair}/astra/{field}", row["astra"][field],
                    "Classification-gap closure and continuous margin closure are distinct metrics.")
        margin_values.append(row["astra"]["margin_closed_h2s"])
assert len(margin_values) == 9 and all(value != 0.0 for value in margin_values)

for run in ["unedited", "p50c"]:
    path = f"exp8_math/eval/{run}/merged.json"
    for benchmark, row in release_json(path).items():
        for field in ["acc", "complete", "problems", "cut_share", "mean_tokens"]:
            if field in row:
                observe(path, f"/{benchmark}/{field}", row[field])

path = "exp8_math/runs/p50c/state.json"
d = release_json(path)
for field, expected in {"runs": 80, "requests": 69, "passes": 50, "handoffs": 3}.items():
    assert d[field] == expected
    observe(path, "/" + field, d[field])

for run, count, entries in [("p50c", 2, 130188), ("p10c", 30, 72986982)]:
    path = f"exp8_math/runs/{run}/edit/manifest.json"
    d = release_json(path)
    assert d["changed_tensors"] == len(d["tensors"]) == count
    assert d["changed_entries_total"] == sum(t["changed_entries"] for t in d["tensors"]) == entries
    for field in ["changed_tensors", "changed_entries_total", "note"]:
        observe(path, "/" + field, d[field])
    for i, tensor in enumerate(d["tensors"]):
        for field in ["name", "shape", "dtype", "changed_entries", "changed_columns", "form", "rank", "fixups"]:
            if field in tensor:
                observe(path, f"/tensors/{i}/{field}", tensor[field])

for run, expected in {"unedited": 0.068, "g05b": 0.0705, "g10b": 0.076,
                      "g25b": 0.077, "g50b": 0.07}.items():
    path = f"exp8_math/gemma_followup/eval/{run}/merged.json"
    d = release_json(path)["math500"]
    assert d["acc_lenient"] == expected and d["problems"] == 500 and d["complete"]
    for field in ["acc_lenient", "acc", "problems", "complete", "cut_share"]:
        if field in d:
            observe(path, "/math500/" + field, d[field],
                    "The report uses acc_lenient; raw g05b is 7.05%, displayed as 7.0% in its README.")

local_sections = {
    "docs/weight-space/README.md": [("Understand the guarantees", 179)],
    "docs/weight-space/design.md": [("Readout compilation and its limits", 339),
                                    ("Caller-supplied PyTorch modules", 398),
                                    ("Proof boundary and evaluation boundary", 438),
                                    ("Use and scale", 467)],
    "docs/weight-space/research.md": [("Follow-up hypotheses are", 231),
                                      ("Project-specific interpretation and remaining gaps", 240)],
    "docs/weight-space/results.md": [("Raw RTS records omit initial checkpoint bodies", 119), ("Reproduce or extend", 125)],
    "docs/PROJECT_MEMORY.md": [("Follow-up research and existing applications", 109)],
    "formal/weight-space/README.md": [("Semantic scope and trust boundary", 181)],
}
local_inventory = []
for path, sections in local_sections.items():
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    locators = []
    for label, recorded_line in sections:
        matches = [i for i, line in enumerate(lines, 1) if label.casefold() in line.casefold()]
        line = matches[0] if matches else recorded_line
        locators.append({"section_or_paragraph": label, "start_line": line})
    local_inventory.append({"path": path, "sha256_at_review": sha(ROOT / path),
                            "review_scope": "selected sections", "locators": locators})

review_text = REVIEW.read_text(encoding="utf-8")
cited_paths = set()
links = []
for target in re.findall(r"^\[[^\]]+\]:\s+(\S+)", review_text, re.M):
    links.append(target)
for target in re.findall(r"\]\(([^)]+)\)", review_text):
    links.append(target)
for target in links:
    if re.match(r"^[a-zA-Z]+://", target):
        continue
    path = (REVIEW.parent / target.split("#", 1)[0]).resolve()
    assert path.is_file() or path == OUTPUT, target
    if path.is_relative_to(ARCHIVE):
        cited_paths.add(path.relative_to(ARCHIVE).as_posix())

inventory = []
for r in retrieved:
    path = r["local_path"]
    if path == "article.html":
        scope, note = "text_read", "Report text and What Comes Next? discussion read; no claim to audit every figure asset."
    elif path == "repository-tree.html":
        scope, note = "structured_metadata", "Only pinned object identity and root entries inspected; not a recursive tree audit."
    elif path.endswith("README.md"):
        scope, note = "full_text_read", "Setup, results, limitations/omissions and reconstruction sections read."
    elif path == "release/LICENSE":
        scope, note = "full_text_read", "License allocation for logs/data, author scripts, embedded MinAtar and third-party data read."
    elif path in inspected_json:
        scope, note = "selected_structured_values", "Values and JSON locators in observations checked; this is not execution or a whole-log audit."
    elif path in cited_paths:
        scope, note = "selected_sections", "Protocol, output, or implementation sections relevant to cited claims inspected; code was not executed."
    else:
        scope, note = "archived_context", "Retained and hash-checked; no claim to full content review or execution."
    inventory.append({"path": path, "sha256": r["sha256"], "bytes": r["bytes"],
                      "review_scope": scope, "note": note})

scan_roots = ["docs/weight-space", "formal/weight-space"]
excluded_scan_components = {".lake", ".git", "__pycache__", ".venv"}
scanned_files = sorted(p.relative_to(ROOT).as_posix() for root in scan_roots
                       for p in (ROOT / root).rglob("*")
                       if p.is_file() and not excluded_scan_components.intersection(p.parts))
standalone = [path for path in scanned_files
              if re.search(r"limitations|next[_ -]?steps|roadmap", Path(path).stem, re.I)]

evidence = {
    "schema": "wsp-source-review-evidence/2",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "review_local_date": manifest["review_local_date"],
    "local_timezone": manifest["local_timezone"],
    "release_commit": manifest["release_commit"],
    "source_manifest_sha256": sha(ARCHIVE / "provenance.json"),
    "source_review_sha256": sha(REVIEW),
    "evidence_builder_sha256": sha(Path(__file__)),
    "execution_boundary": {"released_code_executed": False, "models_reconstructed": False,
                           "provider_calls": 0, "training_runs": 0, "simulation_runs": 0,
                           "independent_scientific_reproductions": 0,
                           "performed": "Source retrieval, text reading, JSON parsing, hashing, local-link and arithmetic checks only.",
                           "third_party_replication": "The article credits a third-party Qwen rebuild/evaluation; it was not independently inspected here."},
    "integrity": {"acquisition_records": len(records), "successful_retrievals": len(retrieved),
                  "release_files": 75, "failed_acquisitions": len(failed),
                  "all_successful_byte_counts_and_hashes_match": True,
                  "prior_article_hash_matches": True, "article_sha256": article_sha,
                  "prior_release_hash_comparisons": comparisons,
                  "review_local_link_targets_checked": len(links), "missing_link_targets": []},
    "tree_verification": {"source": "repository-tree.html", "locator": "payload.codeViewTreeRoute",
                          "current_oid": tree_route["refInfo"]["currentOid"],
                          "root_entries": tree_route["tree"]["items"],
                          "recursive_audit": False, "commit_timestamp_freshly_verified": False},
    "acquisition_failures": failed,
    "article_reading_aid": {"path": "article-text.txt", "sha256": sha(ARCHIVE / "article-text.txt"),
                            "derived_from": "article.html", "source_sha256": article_sha,
                            "method": "Built-in HTML parser used during review; exact extraction implementation was not archived.",
                            "deterministic_reconstruction_claimed": False,
                            "authority": "Raw article.html is the primary retained source; text is a derived reading aid."},
    "source_inventory": inventory,
    "review_scope_counts": dict(Counter(item["review_scope"] for item in inventory)),
    "observations": observations,
    "textual_cross_checks": [
        {"source": "release/exp2_diffusion/README.md", "lines": [8, 32],
         "finding": "Training targets are filled; all three descriptions erroneously depict outlines."},
        {"source": "release/exp3_language/README.md", "lines": [57, 58, 59],
         "finding": "The authors disclose post threshold >0.2 versus scores.json true_copy_heads threshold >0.3."},
        {"source": "release/exp4_fix_car/README.md", "lines": [13, 47, 57, 58],
         "finding": "Base evaluated on 256 episodes; changes counted from rounded displayed network."},
        {"source": "release/exp8_math/README.md", "lines": [81, 82, 83, 134, 138],
         "finding": "Final GPQA baseline replaces an interim 149/198 subset; Qwen base pinned; touched and changed bf16 counts distinguished."},
        {"source": "release/exp8_math/gemma_followup/runs/g50b/final_answer.md", "lines": [10, 12],
         "finding": "Author-reported HOLDOUT: 128 problems, 16384 answers/model, +0.421 percentage points, cluster SE 0.365, approximate 95% CI [-0.300, 1.143]; cutoffs 7.31% to 10.50%."},
        {"source": "release/exp8_math/code/apply_edit.py", "lines": [26, 39, 50, 51, 53, 54],
         "finding": "Reconstruction verifies base and edited tensor hashes; sparse and low-rank reconstruction read, not executed."},
        {"source": "release/exp8_math/code/build_ckpt.py", "lines": [37],
         "finding": "Submitted tensors cast to original dtype before saving."},
        {"source": "release/exp8_math/code/st_numpy.py", "lines": [53, 55, 56],
         "finding": "bf16 conversion implements nearest-even rounding through low-bit adjustment."},
        {"source": "release/harness/sandbox.py", "lines": [33, 46, 50, 68],
         "finding": "Saved-array shape/finiteness and execution resource controls are distinct from semantic restrictions; saved tool output is truncated to 6000 characters."}
    ],
    "model_versions": {"Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
                       "Gemma_followup_base": "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
                       "source": "release/exp8_math/README.md:134; release/exp8_math/gemma_followup/README.md:15"},
    "local_limitations_sources": local_inventory,
    "bounded_document_search": {"roots": scan_roots, "excluded_path_components": sorted(excluded_scan_components),
                                "files_observed": scanned_files,
                                "standalone_limitation_next_steps_roadmap_matches": standalone,
                                "original_release_root_matches": [],
                                "original_actual_discussion": "article What Comes Next? and experiment README setup/results/omitted-artifact sections",
                                "boundary": "Only these local roots and the archived release root were scanned; no claim of global absence or complete recursive release audit."},
    "licensing": {"release_source": "release/LICENSE", "logs_data_networks": "CC BY 4.0",
                  "author_python_shell_scripts": "MIT", "embedded_minatar_source": "GPL-3.0; release/exp5_write_from_scratch/LICENSE-MinAtar",
                  "third_party_data": "Retains original licenses", "article_html": "Separate source; no release-license assignment made."},
}
OUTPUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
assert OUTPUT.is_file()
print(json.dumps({"sources_checked": len(retrieved), "prior_release_matches": len(comparisons),
                  "observations": len(observations), "local_links": len(links),
                  "standalone_documents_found": standalone,
                  "output": OUTPUT.relative_to(ROOT).as_posix()}))
