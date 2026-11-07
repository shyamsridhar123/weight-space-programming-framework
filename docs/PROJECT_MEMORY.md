# Project memory

> **Historical decision record.** The decisions and evidence below describe the
> original RLCD work. Start with the [current framework guide](weight-space/v2/README.md)
> for the current document structure and [reproduction guide](../REPRODUCING.md) for
> standalone commands. The [original record](../provenance/original-documents/docs/PROJECT_MEMORY.md)
> remains available with its exact source bytes.

Updated November 6, 2026 (America/Chicago). Evidence timestamps use UTC and cross
into November 7. Start here for durable decisions; use the linked artifacts for
exact measurements and source identities.

## Weight-space programming is a generic framework

The user's explicit scope is a **generic framework, not an RTS-only system**.
The implementation lives in [`rlcd/weight_space`](../rlcd/weight_space/) and is
documented in the [framework guide](weight-space/README.md),
[comprehensive current overview](weight-space/v2/README.md), and
[source review](weight-space/v2/source-review.md). The earlier
[design](weight-space/design.md) and [research review](weight-space/research.md)
remain historical foundations.

- The core handles immutable named tensors, atomic coordinate programs, exact
  parent identities and old-value checks, composition, rollback, and structural
  policies. Tensor names and ranks do not depend on a game or network architecture.
- Execution semantics, synthesis, behavioral certificates, and environment
  evaluation are separate adapters. Serialized artifacts cannot choose executable
  Python imports. An architecture needs an explicitly registered backend.
- The reference format stores finite binary64 values. The PyTorch bridge preserves
  supported native state on a copy and rejects restoration that would round the
  requested endpoint. Arbitrary tensor editing does not verify a module's forward
  function, non-state configuration, or behavior.
- Finite two-layer ReLU networks were the first proved semantic family. The
  extension adds arbitrary frozen feature maps and arbitrary finite-depth,
  variable-width sequential drift. Execution supports identity, ReLU, absolute
  value, and tanh; the sequential certificate supports the first three. Feature
  extractors, fitting, Python, and native execution remain outside the proof.
- Structured replace/add/scale/factored proposals lower to literal endpoint
  edits. Float64/float32/float16/bfloat16 projection records requested, effective,
  touched, and rounded cells. Explicit rounding in proposal realization does not
  relax the PyTorch bridge's exact restoration checks.
- The frozen-feature compiler fits linear heads with optional bias using target
  fitting, soft retention, or numerical-nullspace constraints. Exact rational
  residual receipts concern actual stored endpoints and supplied feature rows;
  a numerical nullspace does not automatically give exact-zero preservation.
- Read, diagnose, edit, and create sessions have separate candidate routes,
  access policies, budgets, immutable identities, hash-linked traces, and a
  seal-before-evaluation lifecycle. They are cooperative Python interfaces,
  without hostile-code isolation or a live-provider adapter. The sign and
  analytic-creation fixtures test routes and controls, not autonomous discovery.
- A proposal, a contract admission, and an empirical outcome are different records.
  Transferred edits need fresh admission. Rejected proposals are retained for
  research and evaluated only in isolated numerical experiments.

The [generic demo](../artifacts/weight-space/demo-v1/) includes arbitrary tensors
with Tanh semantics and a separately certified ReLU example. Both research
versions have generic regression studies; RTS is one downstream adapter.
Dense snapshots/JSON, cubic hidden matching, and exact rational arithmetic make
this a small-model research reference rather than a transformer-scale system.

## Source-grounded expansion and current evidence

The user requested a comprehensive expansion grounded in the original work,
limitations, and next steps. The [v2 source review](weight-space/v2/source-review.md)
inspects the November 2, 2026 author report, **Sparks of Weight-Space Programming**,
and the release pinned to `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
This is an author report, not a peer-reviewed paper. The bounded archive contains
77 successful retrievals and seven acquisition failures. No standalone original
limitations/next-steps document was found in the inspected root/local scan;
the review uses the actual **What Comes Next?** section, experiment caveats and
omissions, and the repository's existing limitation sections.

Preserve the review's distinctions: supplied semantics and hints are inputs;
identification, diagnosis, causal repair, and synthesis need different evidence;
episodes or sampled answers from one edit are not independent editor trials;
changed-parameter sparsity does not establish retention. The inspected sycophancy
patches have zero classification-gap closure despite continuous margin effects,
and the Gemma follow-up is inconclusive. The original Go no-fitting protocol
forbids least squares; this framework's fitting-permitted controls are a different
regime. Missing bodies, prompts, graders, and intermediate artifacts limit
reconstruction. Source hypotheses about auditing, unlearning, recursive
improvement, and grafting remain open.

The [requirements](weight-space/v2/requirements.md) give 60 traceable obligations:
**23 implemented within stated scope, 27 partial, and 10 open**. The
[architecture](weight-space/v2/architecture.md),
[workflow](weight-space/v2/workflow.md),
[methods](weight-space/v2/methods.md), and
[results](weight-space/v2/results.md) distinguish implementation, mathematical
claims, and observations.

The completed [second study](../artifacts/weight-space/study-v2/) has 12
independent synthetic worlds (seeds 7300–7311), with three paired architectures
per world: ReLU MLP, tanh MLP, and convolution-plus-tanh. Only output heads over
random frozen features were trained. All 36 checkpoints completed; no failures
were recorded. The 36 checkpoints are not 36 independent task populations.

- Protocol semantic SHA-256:
  `c809fc30b69d9fec2fd720c2e80e0d35a08d49a1dd7f6e2f352a6b78e2cbb3de`.
- Freeze identity:
  `0aef8bc624bc5ea501558a807c3c90afa46b3b8ab5d27ea1423943cbb986349e`;
  31 files including the protocol; sources unchanged during execution.
- Six fixed arms: no-op, target-only ridge, soft retention, 12-anchor numerical
  nullspace, one matched random direction, and a fixed 200-step gradient control.
- Mean soft target/retention NRMSE: **0.360218 / 0.199321**; target-only:
  **0.038695 / 0.888351**. Soft reduces collateral NRMSE by **0.689030**,
  exploratory 95% world-cluster interval **[0.674819, 0.703416]**, at a target-error
  cost of **0.321522 [0.299410, 0.344362]**. This is a tradeoff, not universal
  dominance over optimization.
- Selection chose 19 soft, 15 nullspace, and two gradient candidates before test
  access. All 36 also meet the validation thresholds on test; that threshold
  count is post-result descriptive, not a new predeclared endpoint.
- At 64 anchors all 36 nullspace proposals are no-ops. All 72 non-no-op
  feasibility candidates have small but nonzero exact anchor residuals.
- Native CPU discrepancies grow with reduced precision; eight requested
  bfloat16 changes disappear. Each format has its own projected parent.
- All 72 three-edit chains compose and roll back exactly. Soft retention reduces
  interference but the middle region's final desired-behavior error exceeds one
  in every family; this does not establish complete acquisition or no forgetting.
- The 72 auxiliary cases separate exact/rounded read-diagnose controls and
  analytic creation. Three-decimal access preserves signed zero; the prescribed
  reader abstains at either signed zero without using its sign bit. This is a
  policy control, not a demonstration that rounding hides all sign information.
  No live language model or provider was evaluated.

The [independent audit](../artifacts/weight-space/study-v2-audit.json) passed
**211/211 checks**, replaying all 5,020 evidence files, 216 main candidate outcomes,
108 feasibility conditions, 144 precision conditions, 72 chains, 72 auxiliary
cases, exact receipts, and four world-cluster contrasts. Its numerical replay
shares NumPy/PyTorch; it does not regenerate training/optimization or attest
execution history. The auditor's adversarial suite passed 49 tests separately.
The [integration verification](../artifacts/weight-space/verification-v2/README.md)
passed 341 Python tests, with zero failures/errors/skips and two dependency
deprecation warnings. Test counts are distinct runs, not additive evidence units.

The authoritative current report is
[`study-v2-report-complete`](../artifacts/weight-space/study-v2-report-complete/report.md),
with machine-readable summaries and five figures in PNG/SVG/PDF. Its initial
`study-v2-report` attempt timed out before the final receipt and remains incomplete.
The older `report-v2` directory is only a chart revision of study-v1.

## Historical version-one evidence and conclusions

The [version-one results guide](weight-space/results.md) links its generated report, independent
audit, commands, raw records, proofs, and tests. The original study is
[`artifacts/weight-space/study-v1`](../artifacts/weight-space/study-v1/).

- Protocol SHA-256:
  `ae17fc858130ee06d436eb77218712b9a757ed514183b6bc99cb962ce7231998`.
- Freeze identity:
  `8618af22249c963a70f07966e39dd7db82f53064485707797485dc12d916ac6e`.
- Forty independently initialized 6→16→3 networks form twenty disjoint pairs.
  Every planned unit completed and all positive controls passed.
- H1 alignment: normalized transfer-error reduction **0.20661**, 95% paired
  bootstrap interval **[0.14866, 0.26882]**; supported by the fixed rule.
- H2 retention: collateral-RMSE reduction **0.32812**, interval
  **[0.31304, 0.34320]**, with mean target gain **0.63345** of the request;
  supported by the fixed rule. These are exploratory, unadjusted intervals,
  conditional on one fixed shared dataset.
- Admission counts over all twenty proposals per arm: target-only 20, retention
  8, raw transfer 7, aligned transfer 7. Numerical improvement does not imply
  contract admission.
- H3 RTS: validation selected `control_noop` under the fixed tie rule. All four
  recruitment candidates tied baseline utility and loss count. Test effect is
  **0 [0, 0]**; H3 is unsupported. Test contains twenty unique selected no-op
  pairs and twenty bias-control pairs, forty records total. No nonzero recruitment
  edit received held-out evaluation in this study.
- Every selected/baseline test world ended before 270 seconds. The 420-second
  maximum does not imply exposure to rival recruitment at 270 seconds or raids
  eligible at 330 seconds. The observed-loss guard is not a noninferiority test.

The saved initial checkpoint bodies and most final checkpoint bodies are absent
from raw trial records. Their recorded hashes cannot be independently regenerated
from those records alone. Frozen initialization sources and available export
snapshots support the utility denominator; retained HP/recruitment fields support
its arithmetic. Preserve this distinction in later reports.

## Formal guarantees and implementation boundary

The [Lean project](../formal/weight-space/README.md) pins Lean 4.24.0 and Mathlib.
The original actual-model build and separate axiom audit passed: 59 explicit theorem
statements, 236 audited declarations including 121 theorem declarations. Generated
declarations are included in environment counts. Only `propext`,
`Classical.choice`, and `Quot.sound` occur.

The proofs cover positive ReLU rescaling/permutation, transported edits, exact
readout effects, all-layer box drift, sequential drift composition, and strict
argmax preservation under sufficient margins. All 1,293 exported RTS coefficients
and four endpoint contracts are instantiated. Mathematical proofs concern real
arithmetic with exact rational lifts of stored endpoints. Python/JavaScript
execution, the exporter and checker implementation, hardware rounding, optimizer
behavior, and task reward are outside the proved boundary. A structural receipt
for another architecture does not inherit these theorems.

The [v2 formal extension](weight-space/v2/formal.md) adds **53 explicit authored
theorems**: 17 generic-feature statements, 17 sequential statements, and 19
concrete witnesses. These prove exact endpoint effects, exact/approximate
feature-span retention, residual amplification, off-span terms, infeasibility,
factored protection, and arbitrary-depth variable-width drift. Witnesses include
nonlinear features and mixed ReLU/absolute-value/identity networks without game
assumptions. Its completed combined audit covers 376 declarations including 215
theorem declarations, including generated declarations; only `propext`,
`Classical.choice`, and `Quot.sound` occur. All 21 historical formal files are
preserved. These counts describe the new combined audit, not a replacement for
the historical receipt or a sum of independent proofs.

Use the extension's explicit `WeightSpace.GenericAudit` target and recorded
commands; a plain default build still targets the historical library. New
runtime candidates are not generated as checked Lean instances. Generic
behavioral admission is also incomplete: the original `Contract` remains
specialized to `w1/b1/w2/b2`. A feature residual or generic structural receipt
does not silently inherit the original RTS candidate-specific theorem.

## Preserve the original study and the later receipt repair

The original study retains its exact archived runtime. A subsequent audit found
that the generic CLI used Python dictionary equality when comparing certificate
receipts, accepting a boolean-to-numeric substitution such as `false` to `0`.
The current CLI compares canonical serialized bytes, and a regression checks that
the altered receipt is rejected. This repair changes receipt validation only.

[`post-study-repair-v1`](../artifacts/weight-space/post-study-repair-v1/receipt.json)
retains before/after source copies, the patch, commands, logs, and passing checks.
Immediately after that repair, only `rlcd/weight_space/__main__.py` differed from
the original frozen runtime. That is a historical statement: the subsequent
generic expansion changes additional runtime files. The original archived
runtime remains the source for reproducing its study; the new study has its own
freeze. Strict verification must report current-source differences rather than
rewriting an older freeze to make it pass.

The broad pre-repair checks passed 239 Python tests, 386 TypeScript tests, the
TypeScript compiler, and the production build. The repaired runtime passed all 76
focused weight-space tests. These are distinct runs, not additive test counts.

## Follow-up research and existing applications

The original supplied research index is pinned at
`929ecc819301d2ad0b3ccea99e976529998abb81`; the review records 28 sources,
including 24 primary papers, and distinguishes inspected releases from
author-reported findings.
No novelty priority, broad operator superiority, or Astra proposal-quality result
is established. Neither generic study made provider calls.

The [current 11-part research program](weight-space/v2/limitations-and-next-steps.md)
defines dependencies, success criteria, and falsifying outcomes. First priorities
are generic behavioral admission, actual-candidate formal instance generation,
numerical correspondence, and an auditable restricted/live-provider harness.
Independent task/access studies, feature coverage, causal repair, and cumulative
edit contracts follow; additional operators, scale/cost, and selected source
reproductions have their own acceptance obligations. Do not mark the entire
framework or those source hypotheses solved because the current study and tests
pass.

All study seeds have now been observed. Re-executing the frozen protocol is a
reproduction, not a fresh holdout. New confirmatory work needs a new frozen
protocol and untouched seeds; broader generality needs new datasets, architectures,
and tasks. An RTS difficulty/horizon redesign also requires fresh validation and
test worlds, with the old negative result preserved.

The existing Laya/Astra applications, benchmarks, and media are separate project
components. First Conquest's authored 32→38→1 network scores legal candidates
with deterministic argmax and lexical ties. Shared output bias and temperature
are decision-invariant under that policy in real arithmetic. The living-world
application uses different features and stochastic softmax sampling; its
temperature affects behavior. Do not transfer a semantic claim between them
without an explicit adapter and checked assumptions. Existing provider code may
propose edits; the generic framework and scientific study do not require it.
