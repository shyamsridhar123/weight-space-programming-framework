# Retained sources for the v2 source review

This archive supports [the critical source review](../../../docs/weight-space/v2/source-review.md). The review date is November 6, 2026 in America/Chicago; source retrieval timestamps fall on November 7 in UTC.

The original source is the November 2, 2026 author report **Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights**, by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. Its separate accompanying release is `RikiyaT/lm_understand-logs`, pinned to commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Exact source URLs and HTTP acquisition metadata are in [provenance.json](provenance.json).

There are 77 successful acquisitions: the raw report HTML, the pinned repository-root HTML, and 75 selected release files. Seven failed acquisition attempts remain recorded: five initial incorrect path guesses returned 404, and recursive-tree and commit-metadata API requests returned 504. The successful root HTML verifies the pinned object ID and its eleven root entries. This is a selected-artifact archive, not a complete repository mirror or recursive audit.

[review-evidence.json](review-evidence.json) records per-file review scope, byte counts and hashes, comparison with the prior review's thirteen release hashes and article hash, selected numerical observations with JSON locators, local limitation-section identities, and the bounded document search. All experiment READMEs and the report text were read. Other files range from selected-section inspection to archival context; retaining a source does not imply auditing its entire contents.

`article.html` preserves retrieved bytes. `article-text.txt` is a derived reading aid produced during review with a built-in HTML parser; its exact extraction implementation was not archived. The evidence index records both hashes and does not claim deterministic regeneration of that reading aid. The raw HTML is authoritative.

`fetch_sources.py` and `selected-release-paths.txt` record the acquisition mechanism and selection. Re-running acquisition can produce different HTTP metadata or report bytes. The read-only checks and evidence index can be regenerated from retained files with:

```powershell
python artifacts/weight-space/source-review-v2/build_review_evidence.py
```

That script uses the Python standard library to read text and JSON, calculate hashes, check local link targets, and index selected reported values. It writes only the evidence index. It does not import or execute the archived experiment code. Local documentation hashes are a snapshot of the files when the index is generated.

No editor or provider call, training run, simulation, model reconstruction, or independent scientific reproduction was performed for this source review. The report credits a third-party Qwen rebuild/evaluation; that work was not independently inspected here.

No standalone limitations, next-steps, or roadmap file was found in the inspected release root or the bounded local subsystem scan. The report's actual discussion is **What Comes Next?**; experiment READMEs supply setup, results, and omitted-artifact sections. The review also names the actual local design, research, results, project-memory, and formal-boundary sections it read.

Licensing follows [the release's own LICENSE](release/LICENSE): its logs, data, and networks use CC BY 4.0; the authors' Python and shell scripts use MIT; embedded MinAtar source uses GPL 3, with [its license retained](release/exp5_write_from_scratch/LICENSE-MinAtar). Third-party datasets retain their original licenses. These release terms are not assigned to the separate report HTML.
