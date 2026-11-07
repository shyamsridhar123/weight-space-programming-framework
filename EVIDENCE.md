# Evidence and provenance

[PUBLICATION.md](PUBLICATION.md) explains the draft.4 public derivatives and
omissions. Original hashes below identify the historical records; a redacted
public copy is bound separately by the sanitation ledger. Historical passing
receipts do not establish a fresh exact replay of this sanitized snapshot.

The evidence answers a limited research question: can a structured weight edit change a model's outputs in one region while limiting change elsewhere? In the generic synthetic study, soft retention reduced mean held-out collateral error from **0.888351 to 0.199321 NRMSE**, while target error rose from **0.038695 to 0.360218**. The comparison covers **12 independent worlds**, each with three paired architectures whose output heads alone were trained.

That result supports a measured target/retention tradeoff. It does not establish a live LLM's editing ability, useful changes on real tasks, or large-model performance. The historical RTS study selected the **no-op and observed zero held-out effect**. Those conclusions remain separate from successful reconstruction, numerical replay, and real-valued proofs.

This guide connects each claim to the record that can support it. Historical experiments, original-source material, proof runs, and publication checks retain separate identities. Keeping those identities intact makes a result reviewable; it does not make the result stronger.

## Where to start

| Question | Entry point and its purpose |
| --- | --- |
| What is the research question and what changed? | [Research guide](docs/weight-space/v2/README.md), [requirements](docs/weight-space/v2/requirements.md), and [architecture](docs/weight-space/v2/architecture.md). |
| What does the original report show? | [Source review](docs/weight-space/v2/source-review.md) and [retrieval archive](artifacts/weight-space/source-review-v2/README.md), with attributed results and review limits. |
| How did the local experiment work? | [Methods](docs/weight-space/v2/methods.md), [protocol](docs/weight-space/v2/protocol.json), and [freeze receipt](artifacts/weight-space/study-v2-freeze/receipt.json). |
| What are the observed effects and their uncertainty? | [Results discussion](docs/weight-space/v2/results.md) and [completed generated report](artifacts/weight-space/study-v2-report-complete/report.md). |
| Can the saved outcomes be recomputed? | [Independent auditor](scripts/audit_weight_space_v2.py), [saved audit](artifacts/weight-space/study-v2-audit.json), and [reproduction instructions](REPRODUCING.md). |
| What is proved, and under which premises? | [Proof guide](PROOFS.md), [formal discussion](docs/weight-space/v2/formal.md), and [saved formal verification](formal/weight-space/verification-v2/actual/verification.json). |
| Which claims remain open? | [Limitations and proposed experiments](docs/weight-space/v2/limitations-and-next-steps.md) and [roadmap](ROADMAP.md). |
| Which publication changes are new? | [Changelog](CHANGELOG.md), [publication record](provenance/publication-manifest.json), and [verification record](VERIFICATION.md). |

## Preserved evidence branches

The branches answer different questions:

| Evidence branch | What it establishes | Boundary that remains |
| --- | --- | --- |
| [Original-source review](docs/weight-space/v2/source-review.md) | What the inspected author report, protocols, prompts, and released scores say. | The review did not rerun an editor, reconstruct a large checkpoint, or independently repeat an original experiment. |
| [First study](docs/weight-space/results.md) | Conditional regression results and the unchanged RTS selection under that study's protocol. | Regression pairs share a fixed dataset. Initial and most final RTS checkpoint bodies are missing. |
| [Generic second study](docs/weight-space/v2/results.md) | Target/retention errors, anchor feasibility, native precision effects, and interference in explicit synthetic fixtures. | The population is small random-feature models under a deterministic, fitting-permitted procedure. |
| [Formal evidence](PROOFS.md) | Conditional algebra and drift results for the stated real-valued models and supplied witnesses. | Native execution, compiler correctness, and automatic runtime-to-Lean correspondence are outside those proofs. |
| [Saved-data replay](artifacts/weight-space/study-v2-audit.json) | The historical run found agreement among reconstructed endpoints, metrics, and its exact/numerical checks. | That result concerns its original inputs. It does not validate sanitized bytes, recreate training history, prove hidden-access restrictions, or independently attest execution. |
| [Publication verification](VERIFICATION.md) | A completed current check can establish agreement with the public manifest, original-source mappings, and disclosed derivatives. | Read the actual candidate outcome. Packaging cannot establish exact historical replay or broaden a scientific result. |

The principal generic evidence is in `artifacts/weight-space/study-v2/`. It retains all 12 worlds, 36 checkpoints, candidate and selection records, ablations, sequences, auxiliary tasks, and aggregate outcomes. Its manifest closes the recorded tree. It lists member files and separately records a canonical identity for its own JSON; that identity differs from a raw hash of the manifest's bytes. The freeze is in `study-v2-freeze/`, execution records are in `study-v2-execution/`, and the saved independent audit is `study-v2-audit.json`.

The generic study's main paired collateral reduction is **0.689030 NRMSE**, with exploratory 95% interval **[0.674819, 0.703416]**. Its paired target-error cost is **0.321522**, interval **[0.299410, 0.344362]**. Both intervals resample 12 world-level averages; they do not treat all checkpoints or model executions as independent tasks. They are unadjusted exploratory intervals. The [results](docs/weight-space/v2/results.md#four-predeclared-paired-contrasts) give all controls and family results.

The limiting outcomes are preserved alongside those means. Full numerical anchor rank made every proposal a no-op. Every non-no-op feasibility candidate had a small but nonzero exact anchor residual. Lower-precision execution differed from lifted evaluation. Later edits damaged earlier targets despite exact parameter composition and rollback. Rounded sign fixtures retain `-0.0` and use a prescribed abstention policy; they do not show learned infeasibility detection. Only output heads were trained, and no provider was called.

Report directory names require care:

| Path | Meaning |
| --- | --- |
| `artifacts/weight-space/report-v2/` | A later chart rendering of the **first** study. |
| `artifacts/weight-space/study-v2-report/` | An incomplete rendering attempt for the generic second study. |
| `artifacts/weight-space/study-v2-report-complete/` | The authoritative completed generic second-study report. |

The source-review archive is a bounded selection. It retains failed acquisition attempts as well as downloaded material; its historical manifest and evidence companion distinguish retrieval from review. The raw GitHub UI capture is replaced by a disclosed minimal source reference, and environment locators are redacted where the sanitation ledger records a derivative. The review preserves both positive examples and contrary findings, including the absence of successful behavioral repair in inspected sycophancy patches and no established gain in the Gemma follow-up. Upstream ownership and supplied licenses remain in force; see [RIGHTS.md](RIGHTS.md).

The [source inventory](provenance/source-files.json) retains each recorded original's path, size, hash, and preservation mapping. The [sanitation ledger](provenance/public-sanitization.json) separately identifies changed public copies and omitted files, including ten bytecode caches. Directory aliases and local build dependencies are not independent evidence. Historical failures, incomplete reports, and corrected renderings remain identifiable instead of being replaced by the final presentation.

## Important recorded identities

A **raw SHA-256** hashes file bytes. A **canonical identity** hashes normalized JSON as defined by its writer. These are different objects. The identities below describe the original historical records, not necessarily the bytes of their sanitized public copies. Use the sanitation ledger for each disclosed public identity. These values distinguish intended records from older attempts; they do not grade scientific quality.

| Record | Identity convention | Value |
| --- | --- | --- |
| `study-v2-freeze/receipt.json` | Canonical receipt identity | `0aef8bc624bc5ea501558a807c3c90afa46b3b8ab5d27ea1423943cbb986349e` |
| `docs/weight-space/v2/protocol.json` | Canonical protocol identity | `c809fc30b69d9fec2fd720c2e80e0d35a08d49a1dd7f6e2f352a6b78e2cbb3de` |
| `study-v2-audit.json` | Raw SHA-256 | `07cc2034a6ca1ee7f730a455523269b627287c473810a8df91a8da64cb84e620` |
| `final-integrity-v2.json` | Raw SHA-256 | `aeb6b3aaffd079ded26da21884bee5dc1976ec5d13a700ed9ab2274937ec76f7` |
| `final-documentation-review-v2.json` | Raw SHA-256 | `db9ddd2195851912f3136a03074a24b99f4a9bdb592f77c0b253e21e1e0ceda7` |
| `final-review-v2.json` | Raw SHA-256 | `4edf4ad38f5a41be30beba746d9e94d61f01f516759db4d71f8815810f9f52cd` |
| `final-review-v2.json` | Canonical consolidation identity | `1a6511679d73f3fb65e214ca278b96a08f1c9db661a807af81014983bbd52072` |
| `formal/weight-space/verification-v2/actual/verification.json` | Raw SHA-256 | `7d3435dac048bd2ae4c28b90e7e9a035c5134cd0edee75ae200b32c2d68a9eac` |

Unqualified evidence filenames in this table are relative to `artifacts/weight-space/`. The source inventory retains original raw hashes. The independent auditor accepts explicit expected canonical freeze and protocol identities so the intended historical inputs can be checked directly; a sanitation entry does not waive those checks.

## Historical verification chronology

The original integration run verified its source state before the independent auditor's final revision. Comparing today's auditor with that older input list therefore exposes a declared difference. The auditor's earlier raw hash was `3cf6ecb39d644e8a91beca50348943d663c8bd0710ea906c12428778ff39b4db`; its final hash is `b7ad808a592539dfb2097fdfb4d53589d11c48e9b77a8ea02b9c494411addad6`. The later saved-data replay and [focused auditor checks](artifacts/weight-space/study-v2-audit-execution/focused-tests.log) cover the final revision. The earlier integration receipt must not be presented as a fresh run of that later code.

The [final integrity review](artifacts/weight-space/final-integrity-v2.json) checked the original preserved bytes and closed directories. The [historical documentation review](artifacts/weight-space/final-documentation-review-v2.json) checked its then-current links, fragments, and command examples. The [consolidation receipt](artifacts/weight-space/final-review-v2.json) binds those inputs and documents. Their detailed inventories remain historical records. They describe their own source state, not the rewritten guides or sanitized artifacts in this publication.

One distinction affects reproduction directly. The generic study froze `docs/weight-space/v2/source-review.md`. Its original bytes remain in the frozen sources and in [the archived publication copy](provenance/original-documents/docs/weight-space/v2/source-review.md.txt). The saved-data auditor treats a live-checkout difference as informational while replaying the historical archive. Strict `study_v2.verify_freeze(current_sources=True)` and `run()` reject the rewritten live file against that old freeze. Use the historical freeze for saved-evidence replay and create a fresh freeze for fresh execution. Reusing the same seeds still gives reproduction, not fresh confirmation.

Public sanitation adds a separate boundary. The first study's frozen lock is
changed, and redacted logs and omitted caches no longer match its original
audit inventory. Its strict freeze, audit, and report checks are retained and
must reject those original-byte claims. A historical `passed: true` remains
a result about the private original inputs. A successful current publication
check can validate the disclosed derivative, but cannot supply a fresh exact
replay of that freeze.

[VERIFICATION.md](VERIFICATION.md) records current standalone checks separately. Test, file, and theorem totals describe the scope of particular checks; they are not measures of research quality or additional experimental observations.

## Proof evidence

The v2 formal receipt records Lean **4.24.0**, mathlib revision `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`, the explicit `WeightSpace.GenericAudit` target, the closed proof/receipt inputs, and the combined axiom audit. The allowed observed axioms are `propext`, `Classical.choice`, and `Quot.sound`. The [proof guide](PROOFS.md) maps the statements, source files, premises, and historical external inputs.

The proofs establish conditional statements over real-valued models and supplied witnesses. They explain, for example, what exact feature-span protection would imply and how coordinatewise drift bounds propagate. They do not prove IEEE execution, the Python compiler or checkers, a feature extractor, or automatic correspondence between a runtime candidate and a Lean instance. A small observed residual cannot satisfy an exact-zero premise merely because a preservation theorem exists.

Authored theorem statements and generated environment declarations have different counting rules. Their inventories remain in the formal receipts. New proof runs belong in `.proof-runs/` and receive distinct receipts; they must not replace the historical result.

## Publication transformations

The publication distinguishes original scientific identities, public derivatives, and revised explanatory material. Original live-guide mappings are under `provenance/original-documents/`. Most use `.md.txt` suffixes; `docs/PROJECT_MEMORY.md` retains `.md` within that archive. The [source inventory](provenance/source-files.json), [current publication record](provenance/publication-manifest.json), and [sanitation ledger](provenance/public-sanitization.json) identify the original records, their delivered representations, and the revised live documents. The [draft.3 publication manifest](provenance/historical-publication/draft3-publication-manifest.json) retains the prior publication's scope. An archived document is evidence of prior wording, not a current instruction to follow.

The earlier diagram conversion affected `docs/weight-space/v2/README.md`, `docs/weight-space/v2/architecture.md`, and `docs/weight-space/design.md`. Their live guides embed images, with PNG and editable SVG versions. The diagram manifest records graph identities, alt text, renderer identity, and output hashes. Later prose revisions explain the research question, mechanisms, comparisons, and limits; their mappings are recorded as publication changes rather than changes to historical evidence.

The Python implementation, Lean sources, experiment payloads, and retained tests keep their scientific scope. Original hashes and historical receipt identities are not recomputed to conceal sanitation; the ledger discloses changed receipt, log, lock, and environment bytes. Package metadata, root guides, wrappers, CI, diagrams, and verification records have their own publication history. The optional Node package carries the historical runner's dependency closure, with public environment-locator changes disclosed separately.

`MANIFEST.sha256` binds publishable files except itself. The publication checker verifies declared coverage, original-source mappings, and the disclosed public representations, rejects unsafe paths, and checks links and image-only diagrams in live guides. Its own code is included in the checksums. This is a consistency check, not an independent root of trust. A final Git commit and release record must identify a delivered package; the draft.4 release remains unpublished.

## What preservation does and does not establish

Preservation lets a reader inspect the same candidate, data, and claimed outcome. Saved-data replay adds a separate implementation of the equations, while still sharing numerical libraries. A local freeze records what was fixed locally. None of these establishes hostile-code isolation, execution attestation, a live provider result, or fresh confirmation on untouched tasks.

The strongest empirical conclusion remains bounded: retention-aware head fitting changes the target/retention compromise in the tested generator. Evidence on new task lineages could narrow or overturn its generality. Cumulative usefulness would require earlier targets to survive later edits; the recorded sequence already shows why exact parameter composition is insufficient. A specific causal explanation would need matched intervention controls beyond a useful final endpoint.

Every published study seed is observed. The [roadmap](ROADMAP.md) states proposed comparisons and acceptance or falsification conditions for new evidence. New results, changed sources, and failed attempts need separate identities so readers can distinguish a new experiment from a new explanation of the old one.
