# Generic weight-space programming: framework, evidence, and research program

This is the comprehensive second version of the repository's weight-space programming framework. It supports **reading, diagnosis, editing, and creation** through shared model identities, proposal formats, access records, and evaluation workflows. Model execution, behavioral contracts, proof backends, and task evaluators have explicit interfaces. RTS remains one historical application; none of the new core interfaces, mathematical extensions, or study tasks depends on a game.

The expansion begins with a fresh [original-source review](source-review.md), turns the observed limitations into [60 traceable requirements](requirements.md), implements a broader [architecture](architecture.md) and [workflow](workflow.md), adds [generic mathematical results](formal.md), and evaluates the implementation under a [frozen protocol](methods.md). The [results discussion](results.md) includes negative and limiting outcomes. The [research program](limitations-and-next-steps.md) states the remaining work with acceptance and falsification criteria.

The evidence supports a small-model research framework and specific conditional theorems. It does not establish a universal neural-network compiler, autonomous mechanism discovery, successful unlearning, or large-model improvement. Those claims have concrete remaining obligations below.

## What changed after rereading the sources

The original November 2, 2026 work is an author report, with a released experiment repository. The review uses its actual future-directions discussion, experiment caveats and omitted-artifact sections, and this repository's existing design, results, and formal limitations. No standalone original limitations or next-steps file was found in the bounded inspection. The source archive retains 77 successful retrievals, seven acquisition failures, hashes, and per-file review scope; it is not a complete recursive audit. See the [source boundary](source-review.md#sources-and-review-boundary).

The resulting design decisions are substantive:

| Source or prior-framework problem | Response in this version | Remaining distinction |
| --- | --- | --- |
| Identification, diagnosis, repair, and synthesis have different evidence | Separate task kinds, answer/artifact/program candidates, and mode-specific evaluation | Correct classification does not establish a causal mechanism or successful repair |
| Supplied semantics, hints, tools, and feedback affect the task | Explicit access policies, budgets, split records, callback accounting, and retained traces | The cooperative interface does not enforce isolation from arbitrary Python or outside information |
| Compact proposals and native storage can produce different edits | Structured commands lower to an exact-parent literal program, with requested and realized endpoints and projection receipts | A short program, a low-rank factorization, and a sparse stored change are different properties |
| Sparse parameter changes do not guarantee retention | Exact feature-row residuals, feature-span theorems, real-valued drift bounds, held-out retention, and repeated-edit measurements | Each guarantee requires its own domain and premises |
| The earlier proof family was a two-layer ReLU network | Arbitrary frozen feature maps and arbitrary finite-depth, variable-width sequential drift | Supported operators and the runtime-to-proof correspondence remain explicit |
| Repeated samples and released final patches have limited inferential scope | Independent-world clustering, full candidate/failure ledgers, closed manifests, and independent replay | Reproduction, new editor trials, and confirmation on new tasks are separate activities |

This framework intentionally permits numerical fitting in its algorithmic controls. The original Go experiment prohibits regression and least squares. The new experiments therefore test this framework's mechanisms and evidence handling; they are not a replication of that access-restricted experiment or a measurement of an LLM's ability.

## A generic task and its lifecycle

A task declares the input model or creation interface, output meanings, requested behavior or answer, retained behavior, permitted information and computation, budget, candidate rules, evaluator, and independent evaluation units. These declarations are meaningful only when the corresponding implementation or experimental design supplies them. A label such as an architecture name or evaluator ID is not a proof of semantics or ownership.

The same lifecycle accommodates an answer about a checkpoint, a proposed diagnosis, a patch to existing weights, and a complete newly constructed artifact:

```mermaid
flowchart TD
    T[Task, semantics, access, budgets, splits] --> S[Programmer session]
    S --> A[Read or diagnose: answer candidate]
    S --> P[Edit: structured or literal program]
    S --> C[Create: complete artifact]
    P --> L[Lower, project, retain actual endpoint]
    L --> V[Caller checks structure and required behavior]
    A --> V
    C --> V
    V --> D[Validation, declared selection, or abstention]
    D --> F[Seal selected identity]
    F --> E[Held-out evaluation]
    E --> R[Receipt, complete history, independent replay]
```

Structural and behavioral checks remain caller responsibilities. A session does not automatically certify a submitted candidate. All candidate attempts, validation observations, refusals, failed calls, and selection decisions belong in the evidence. The unchanged model is a legitimate edit fallback; an answer can abstain rather than manufacture a conclusion from inadequate information.

The [architecture guide](architecture.md) contains executable Python examples and CLI commands. The [workflow guide](workflow.md) explains what is charged, when a session is sealed, what an evaluation receipt binds, and what the cooperative boundary cannot enforce.

## Implemented components and supported scope

| Component | Available capability | Scope boundary |
| --- | --- | --- |
| Model and program core | Immutable named finite tensors, canonical identities, literal coordinate edits, exact-parent checks, atomic application, composition, rollback | Canonical binary64 artifacts; no automatic semantics for an arbitrary forward program |
| Structured proposal language | Replacement, addition, scaling, matrix factors, explicit blocks, deterministic lowering to literal endpoints | Explicit shape rules; no broadcasting or arbitrary executable proposal code |
| Native projection | Float64, float32, float16, and bfloat16 conversion; reject or record rounding; requested/touched/effective/rounded counts | Endpoint realization and measured execution are separate; hardware arithmetic is unproved |
| Frozen-feature compiler | Linear heads with optional bias; target-only, soft-retention, and numerical-nullspace fitting | Fixed extractor and positive ridge; fitting is permitted and the solver is trusted |
| Feature certificates | Exact rational residuals from the actual endpoints on supplied feature rows | Domain-wide retention additionally needs feature correspondence and span/residual witnesses |
| Sequential semantics | Finite compatible affine/activation chains of variable widths; identity, ReLU, absolute value, and tanh execution | The current all-layer certificate supports identity, ReLU, and absolute value |
| Generic workflow | Read, diagnose, edit, create; access and budgets; immutable candidates; trace verification; sealing and one held-out attempt | Same-process cooperation, without a hostile-code sandbox, live-provider integration, or evaluator attestation |
| PyTorch bridge | Supported floating persistent state captured and edited on a copy, with native identity and alias checks | Unsupported state, quantization, arbitrary hooks and forwards need separate support or rejection |
| Evidence tools | Frozen required inputs, retained failures, closed artifact manifests, independent saved-evidence replay, reproducible figures | Hash consistency is local integrity evidence, not an external timestamp or independent signed attestation |

The older behavioral `Contract` is still specialized to the two-layer `Network`. Generic structural policies, sequential bounds, and feature receipts are implemented, but a unified generic behavioral-admission record is an open integration task. The [requirements matrix](requirements.md) marks **23 requirements implemented within their stated scope, 27 partial, and 10 open**. These are engineering status categories, not empirical success rates.

## Mathematical contribution and its practical meaning

For a fixed feature map \(\phi\), a linear readout's actual change has the exact real-valued form

\[
h_Q(x)-h_W(x)=(Q-W)\phi(x).
\]

If the actual endpoint difference vanishes on protected anchors, it vanishes on inputs with a valid representation in their feature span. With nonzero anchor residuals, the guarantee scales with the absolute representation coefficients; inputs outside the span contribute an additional representation-error term. A target inside an exactly protected span cannot receive a nonzero change. These results explain both useful selective edits and legitimate infeasibility.

A separate induction proves componentwise drift bounds for arbitrary finite-depth, variable-width sequential networks under explicit activation assumptions. It covers simultaneous layer changes, and supplies checked instances for identity, ReLU, and absolute value. Factored-edit identities, residual amplification, and composition are also proved. The concrete witnesses use nonlinear features and mixed activations without RTS assumptions.

The extension contains **53 explicit new theorem statements**. The build and combined axiom audit passed, examining 376 declarations including 215 theorem declarations; generated declarations are included in those environment counts. Only `propext`, `Classical.choice`, and `Quot.sound` occur. The [formal guide and receipts](formal.md) distinguish source-level statements, environment counts, dependencies, and the trusted toolchain.

These are real-arithmetic theorems. The new study artifacts are not automatically instantiated as generated Lean definitions, and the compiler, feature extractor, Python checker, and native execution are not verified by those proofs. Small exact residuals remain nonzero. A valid but loose bound can fail to establish a useful contract.

## Completed generic study

The study contains **12 independently generated synthetic worlds and 36 paired checkpoints** across ReLU MLP, tanh MLP, and convolution-plus-tanh feature extractors. Only their output heads were trained. Six fixed editing arms share the same public and validation information. Selection is fixed before held-out evaluation. All 36 checkpoints completed, with no recorded failures.

| Main fixed arm | Mean target NRMSE | Mean collateral NRMSE |
| --- | ---: | ---: |
| No-op | 1.000000 | 0.000000 |
| Target-only ridge edit | 0.038695 | 0.888351 |
| Soft retention | 0.360218 | 0.199321 |
| Numerical nullspace, 12 anchors | 0.343031 | 0.338563 |
| Matched random direction | 1.210108 | 0.682883 |
| Gradient, 200 steps | 0.383597 | 0.185988 |

Soft retention reduced collateral NRMSE by **0.689030**, with an exploratory 95% world-cluster bootstrap interval of **[0.674819, 0.703416]**, while increasing target NRMSE by **0.321522 [0.299410, 0.344362]** against the target-only arm. The tradeoff is the central finding. The fixed gradient control has slightly higher target error and lower collateral drift; this is not a compute-matched claim that one method dominates optimization.

The study also exposes important boundaries:

- With 64 retention anchors, every protected design matrix had full numerical column rank and every nullspace proposal was a no-op. With 4 or 12 anchors, all realized anchor residuals were small but nonzero.
- Native CPU execution differs from lifted binary64 reference evaluation. The maximum measured edit-drift discrepancy grows from about \(1.6\times10^{-15}\) in float64 to \(1.0\times10^{-2}\) in bfloat16, and eight requested bfloat16 cell changes disappear after projection.
- Across 72 three-edit chains, composition and rollback restore the declared parameter identities exactly. Soft retention reduces interference relative to target-only editing but does not reliably acquire and preserve every requested regional behavior.
- Exact-access sign fixtures answer correctly. Three-decimal access displays zero, and the prescribed reader abstains at either signed zero without using its sign bit. The interface preserves signed zero, so this is a policy control rather than an information-hiding result. Analytic creation fixtures pass their supplied formulas. These validate routes and controls, not live inference or autonomous synthesis.

The [full results](results.md), [generated numerical report](../../../artifacts/weight-space/study-v2-report-complete/report.md), [raw study](../../../artifacts/weight-space/study-v2/), and [methods](methods.md) provide the denominators, equations, family-specific values, selection results, precision and sequence tables, uncertainty rules, and audit boundary. Figures are available as PNG, SVG, and PDF.

The standalone [saved-evidence audit](../../../artifacts/weight-space/study-v2-audit.json) passed **211 of 211 checks**, covering all 5,020 study files and 31 frozen inputs. It independently replays the saved candidates, outputs, precision conditions, edit chains, exact rational receipts, and clustered estimates without importing the framework or refitting models. Numerical replay shares NumPy/PyTorch; it does not attest execution history or prove the numerical libraries. The [results audit section](results.md#independent-audit-and-verification) records its complete scope. Separately, the [integration run](../../../artifacts/weight-space/verification-v2/README.md) passed 341 Python tests, and the auditor's adversarial suite passed 49 tests. These are separate runs and counts.

## What remains and the order to tackle it

The immediate engineering priorities are to integrate generic behavioral admission, generate checked formal instances from actual candidates, and connect real bounds to a specified numerical execution model. Each closes a different missing link between a valid edit, a mathematical statement, and deployed computation.

Next, an auditable restricted/provider harness is needed before testing live programmers under controlled access. Independent read, diagnose, repair, and create tasks should compare identical supplied context with and without weights, use untouched lineages and tasks, and retain every editor attempt. Mechanistic explanations require frozen intervention predictions and appropriate controls. More labels or rollout samples from one final patch do not supply those missing independent units.

Feature coverage and feasibility, cumulative contracts, additional operators and tighter bounds, scale and total cost, and selected source reproductions form separate work packages. The [11-package roadmap](limitations-and-next-steps.md#priority-and-dependencies) gives dependencies, concrete designs, success criteria, and falsifying outcomes. Its final table converts source proposals such as auditing, unlearning, recursive improvement, and cross-architecture grafting into distinct tests. None is marked solved by this framework.

## Reading and reproduction map

| Purpose | Document or evidence |
| --- | --- |
| Understand the original claims, actual caveats, and missing artifacts | [Source review](source-review.md) and [retained archive](../../../artifacts/weight-space/source-review-v2/) |
| Trace every requirement to a source and an acceptance obligation | [60-requirement matrix](requirements.md) |
| Use the APIs and commands | [Architecture and examples](architecture.md), [session workflow](workflow.md) |
| Inspect assumptions, theorems, and proof reproduction | [Formal guarantees](formal.md) |
| Reproduce the study design without treating old seeds as fresh | [Methods](methods.md), [protocol](protocol.json), [source freeze](../../../artifacts/weight-space/study-v2-freeze/receipt.json) |
| Assess observed effects, failures, and limits | [Results discussion](results.md), [generated report](../../../artifacts/weight-space/study-v2-report-complete/report.md) |
| Inspect independent replay and software verification | [Audit scope](results.md#independent-audit-and-verification), [audit receipt](../../../artifacts/weight-space/study-v2-audit.json), [integration verification](../../../artifacts/weight-space/verification-v2/README.md) |
| Plan the next implementation or experiment | [Limitations and prioritized next steps](limitations-and-next-steps.md) |
| Understand earlier results and their historical scope | [Version-one results](../results.md), [project memory](../../PROJECT_MEMORY.md) |

All published study seeds have now been observed. Rerunning them is reproduction. Fresh confirmation requires a new frozen protocol and untouched units. Evidence commands require unused destinations; audits and reports must remain outside closed study directories.

The earlier regression study, its receipt repair, and the negative RTS result remain preserved with their own identities. The old `artifacts/weight-space/report-v2` directory is a chart revision of the first study, not this second generic study. The current report is `artifacts/weight-space/study-v2-report-complete`. This version adds evidence and scope without rewriting the earlier record.
