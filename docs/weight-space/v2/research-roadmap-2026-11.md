# Where the framework goes next

Research review and proposed execution plan, **November 8, 2026**.

The next milestone should be a **trained customer-support routing experiment**:
change ownership of one class of requests, preserve the rest, and determine
whether editing the model is preferable to a rule, a routing overlay, or
retraining. Build candidate/contract binding and an auditable experiment harness
alongside it. This gives the framework a concrete decision to support and a
credible opportunity to fail.

The framework remains generic. Text routing is the first trained application,
followed by a trained non-text task using the same artifact, proposal, contract,
admission, and evaluation interfaces. The present synthetic outage demo stays a
teaching example. It is not the evidence for an enterprise performance claim.

This plan builds on the original author report's **What Comes Next?**, the
[source review](source-review.md), the [existing limitations](limitations-and-next-steps.md),
the [requirements ledger](requirements.md), and implementation at revision
`ab4ca365c3c145f55e49c5fa2d52e897a44ac76d`. It adds a primary-literature review,
a specific trained-use-case protocol, decision gates, and
[published work issues](research-issues-2026-11.md). **All new experiments and
engineering deliverables below are proposed, not completed.**

## What the current evidence leaves unresolved

The repository already records many useful distinctions. Its remaining problem
is connecting them to an independently useful outcome, rather than producing
another favorable fixture.

| Present evidence | Remaining question | Consequence for the next milestone |
| --- | --- | --- |
| Synthetic structured support records and a hand-constructed output head in [the enterprise example](../../../examples/enterprise_support_routing.py) | Does a trained representation contain a useful direction that avoids protected behavior? | Replace the dedicated outage flag with learned text features; measure feasibility before promising an edit |
| Trained heads over random frozen extractors in [V2](results.md) | Does retention generalize beyond the supplied feature rows and random worlds? | Use trained text encoders/heads, independent task cases and training lineages, untouched evaluation, and explicit OOD cases |
| Soft retention improves collateral error while worsening target error | Is the target gain worth the retained-behavior harm relative to simpler alternatives? | Freeze minimum useful gain, maximum harm, coverage, and resource constraints |
| Numerical nullspace endpoints with exact residual receipts | Is numerical near-preservation enough for deployed decisions? | Publish residuals, decision changes, arithmetic conditions, and margins separately; nonzero residuals remain nonzero |
| CPU precision ablation and native-state round trips | How do operation order, cancellation, boundary margins, and other supported backends change decisions? | Extend the existing precision study rather than duplicate it |
| Cooperative session access checks in [workflow.py](../../../rlcd/weight_space/workflow.py) | Does a live editor benefit from weights, and which restrictions are actually enforced? | Use brokered access, opaque submissions, complete attempts, and actual provider provenance; retain a cooperative lane with honest labels |
| Real-valued [Lean results](formal.md) | Does the theorem refer to this candidate, this domain, and the specified execution? | Bind immutable endpoints and semantics, generate checked instances, and state the remaining implementation/backend assumptions |
| Three-edit sequence study against changing checkpoints | Are old target gains and original-baseline retention maintained across a chain? | Evaluate whole lineages with a cumulative contract and explicit conflicts |
| Retained original logs and source hashes | Can the source task be independently repeated under its actual access rules? | Separate inspection, reconstruction, reevaluation, replay, and independent repetition; missing assets stay visible |

All recorded V2 worlds, including `7300–7311`, have already been observed.
Rerunning them is reproduction, not fresh confirmation. There is currently no
live-provider study. The fitted compiler is not evidence that a language model
can infer a repair without examples or fitting. See the corresponding boundaries
in [limitations and next steps](limitations-and-next-steps.md).

## What the literature changes about the plan

### Selective editing is an identifiable geometry problem

AlphaEdit [S2] motivates projecting updates away from retained feature keys.
It also exposes the critical dependency: protection is defined by a particular
linear representation and practical eigenvalue threshold. The method should
inform this framework's experiments, rather than supply a universal preservation
claim for a different task.

For a frozen encoder, let the augmented feature include the bias coordinate:
`h(x) = [encoder(x); 1]`, with head `f(x) = W h(x)`. Let `H_P` contain protected
feature columns. Preserving every protected score in exact arithmetic requires
`Delta W H_P = 0`. If `H_P` has full row rank, the only such head edit is
`Delta W = 0`. More anchor rows can therefore eliminate every useful direction.
Their count alone says nothing about domain coverage.

For one target feature, let `q` be its orthogonal component outside the protected
span and let `g` be the desired vector of output changes. When `q != 0`, a
single-target exact protected update is `Delta W = g q^T / ||q||^2`, whose
Frobenius norm is `||g|| / ||q||`. As `q` approaches zero, the required edit grows.
At `q = 0`, a nonzero target change is infeasible. Multiple targets can also
conflict with each other even when a nontrivial nullspace exists. These are
exact-arithmetic consequences of the frozen-head model, not guarantees about
native solver results or an arbitrary encoder. The constructive update assumes
all augmented head coefficients are editable; structural restrictions add
further constraints. Changing few stored coefficients is also insufficient for
preservation: shared features can propagate a sparse change to many outputs.
Edit sparsity is a structural measure, not retained-behavior evidence.

The next study must record protected rank/nullity, singular values, cutoff,
conditioning, target projected energy, required edit norm, realized endpoint
residuals, and success away from anchors. Compare hard score retention with
explicitly approximate score retention and decision-margin contracts. Approximate
protection changes the claim and must identify its tolerance. An abstaining
system that never edits can preserve everything while supplying no useful
coverage; selective-classification work [S23] makes that tradeoff explicit.

The feature map is another boundary. For `h(x) = H_P a + r`, exact anchor
preservation implies `Delta W h(x) = Delta W r`, so drift depends on both edit
size and the uncovered component. A sampled estimate of `r` is an empirical
coverage diagnostic. A whole-domain theorem needs a justified bound over the
declared domain, including the encoder. The existing head proof cannot supply
that missing premise by itself.

### Editing needs a non-editing baseline and realistic evaluation

ROME [S3] and MEMIT [S4] establish useful methods and factual-association
settings, but do not make every policy change a knowledge-editing problem.
The Mirage/WILD study [S5] demonstrates how teacher forcing, target-length
truncation, narrow prompts, and permissive answer matching can inflate apparent
success. Benchmarking and Rethinking Knowledge Editing [S6] includes a
contextual alternative and separates several generalization outcomes.

For routing, use complete model predictions on independently held-out text,
with an explicit queue policy and abstention rule. For a later generated-answer
task, freeze autoregressive decoding, natural termination, context variation,
scoring, and judge validation. Do not transplant language-generation metrics
into a classifier or use a successful single rewrite as enterprise impact.

A queue ownership change can often be solved by mapping an existing predicted
intent to a new queue. That overlay is a serious baseline. If it works as well
with less cost, the result should favor the overlay. Weight editing is more
plausibly useful for a learned misclassification that a simple downstream map
cannot fix, or when a justified deployment constraint favors a self-contained
artifact. Both hypotheses need evidence; neither is assumed.

### A diagnosis needs a falsifiable intervention

Mechanistic explanation is useful only if it predicts something a controlled
intervention can test. Following the intervention logic of ROME [S3], freeze an
editor's suspected coordinates, mechanism, and predicted effect before repair.
Compare the repair with sham edits, random locations, and norm/support-matched
controls. Evaluate the primary effect and collateral behavior separately.

Start with blinded planted mechanisms whose causal origin is known to the
evaluator, but hidden from the proposer. Localization alone is not causal repair;
weight access that includes the planted answer in tensor names is not blind
auditing. Access to task descriptions, metadata, weight values, activations,
forward execution, feedback, and fitting must be separate experimental factors.

### Proofs must name the artifact and arithmetic

TorchLean [S12] offers a relevant shared-graph approach and a particularly useful
scope table distinguishing proved, checked, and external components. Its reviewed
v2 has 55 pages and a restricted finite-execution Float32 refinement; the
hardware/kernel link remains external. This review did not build or audit its
implementation. Treat it as a candidate dependency to evaluate, not an inherited
proof of our runtime.

CROWN [S13] and Provable Repair [S14] suggest ways to improve supported bound
coverage and repair tractability. Their assumptions matter: a conservative bound
may reject a safe candidate; the decoupled repair architecture's LP result does
not make arbitrary ordinary neural repair easy. Begin with an explicit small
semantic family, immutable serialization, independent decoding checks, and
candidate-specific Lean instances. Extend operators only when a trained task or
measured bound bottleneck justifies the work.

A decision-preservation claim needs more than a small score residual. If the
original unique winner has margin `m` and every score can drift by at most `b`,
then `m > 2b` suffices to preserve argmax. Per-coordinate budgets give a tighter
winner-versus-competitor check. Tie breaking, nonfinite values, preprocessing,
candidate classes, calibration, and numerical execution remain part of the
declared decision semantics. A real-valued theorem does not automatically cover
CPU, GPU, quantization, compiler rewrites, or accumulated roundoff.

### Long sequences, symmetry, and cost need their own units

ForgetBench [S7] evaluates forgetting across edit time and memory age. The
framework should retain an edit-by-target matrix and original-baseline harm,
including old targets and conflicts, instead of reporting only the newest gain.
The independent unit is a whole lineage and assigned edit order. Artifact
rollback restores weights; it does not undo ticket assignments, caches, random
state, or deployment effects unless those are explicitly restored too.

DeepAlign [S15], ScaleGMN [S16], NFN [S17], and DWS [S18] show why raw coordinate
transfer is fragile: compatible models can implement the same function through
different neuron permutations, and valid scaling symmetries depend on the
architecture/activation. Positive controls should use known function-preserving
transformations before testing independent trained lineages. Alignment is not a
causal circuit correspondence across unrelated architectures.

SANE [S19] motivates scalable representations of trained model collections.
LoRA [S20] supplies a relevant adaptation baseline where applicable. A sparse
patch can still require dense loading, copying, hashing, contextualization,
proposal, verification, and evaluation. Measure all these costs at matched
quality and retention. Charge learned-reader training, model-zoo construction,
human selection, failed proposals, and amortization assumptions explicitly.

### Unlearning is a later, separate threat-model study

TOFU [S10] provides synthetic forget data and a retain-only reference. MUSE [S9]
separates memorization, knowledge, privacy, utility, scale, and repetition.
Quantization-induced recovery [S11] and the recovery-forecasting preprint [S8]
reinforce the gap between suppressing an answer and making information
inaccessible under later operations. These are scoped findings, not universal
erasure tests.

Run a synthetic unlearning study only after the access, evaluation, and sequence
infrastructure exists. Declare attacker knowledge, allowed queries, extraction
scoring, transform family, recovery data, recovery compute, and retain metrics.
Hold audit attacks apart from optimization/selection. Passing finite attacks
supports a bounded empirical robustness statement, never universal erasure or
a production privacy claim.

## The first trained enterprise study

### The business question and data boundary

Start with a banking-support intent classifier. Derive a policy in which a
selected service-request intent moves from a general queue to a specialist queue,
while other intent ownership stays fixed. The framework must either produce a
useful admitted artifact or explain why the change cannot be made under the
retention contract.

BANKING77 [S22] is a reasonable first public text source. CLINC [S21] is a
subsequent cross-dataset/out-of-scope extension, not interchangeable labeled
examples for the same model. The author repositories were checked: BANKING77's
repository displays CC BY 4.0; CLINC's displays CC BY 3.0. Pin repository revisions,
data hashes, dataset-specific notices, encoder/model revision and license before
execution. The project's MIT license does not replace those notices.

These datasets supply English intent queries. Our queue mapping and ownership
changes are derived tasks. They do not supply actual enterprise queues, telemetry
outages, observed transfers, resolution times, or operational savings. A text
case resembling the current outage story requires a separately disclosed public
or synthetic dataset. Do not silently relabel banking queries as outage tickets.

### Training, policy cases, and splits

Use a pinned text encoder with a trained head as the first controllable model.
Define explicitly whether the editable head outputs intents or queue scores.
Evaluate the final queue decision in either case. An overlay acts on the unchanged
intent predictions with the same eligible policy information; it must not receive
oracle test labels. A separately trained baseline receives the same eligible
training/development data, and its preparation is charged. Do not equate an
argmax over grouped queue scores with mapping the winning intent unless that
equivalence is actually established for the chosen decision rule.
Record all pretraining ancestry that is known; a frozen pretrained encoder with
new head seeds supplies independent head-training lineages, not independently
pretrained foundations. Add independent encoder fine-tunes if the desired claim
requires representation-level variation, and charge their training cost.

Define two case types before evaluation:

1. **Ownership change:** unchanged intent recognition with a new queue policy.
   This tests whether a weight edit has value over a downstream intent-to-queue
   overlay. The overlay must be allowed to use the same policy information.
2. **Recognition repair:** a reproducible held-out confusion within a frozen
   policy, chosen from development data by a prespecified rule. This tests whether
   representation/head changes correct unseen errors beyond a mapping baseline.
   Discovering the failure on the confirmatory test set invalidates that split.

Separate encoder/head training, editor development, target examples, protected
anchors, calibration/selection, and sealed evaluation. Audit exact and near
duplicates and lineage ancestry; split identifiers alone do not prove independence.
Preserve any published dataset test split for the frozen evaluation stage rather
than using it to design edit cases. A fresh protocol can be frozen locally, but
do not claim that widely available benchmarks were absent from pretraining or
that a local freeze is an external preregistration.

Include semantically nearby retained intents, ambiguous/OOD requests and
paraphrase/context stress cases. Newly authored stress examples need a separate
provenance ledger, blinded validation, and no selection on their final scores.
Single-intent public data cannot justify a multi-intent deployment claim.

### Comparisons and access

| Arm | What it answers | Fairness requirement |
| --- | --- | --- |
| Unchanged parent | How much improvement was needed? | Same preprocessing, policy and evaluated requests |
| Rules / unchanged intent-to-queue overlay | Can policy configuration solve the task more cheaply? | Same explicit task/policy information and eligible development data |
| Context/retrieval method where task-compatible | Can supplied evidence solve the task without changing weights? | Same evidence; charge added inference latency and maintenance |
| Target-only fitted head edit | What collateral harm results without protection? | Same target data, development tuning, edit scope and fitting budget |
| Soft retention | What gain/harm tradeoff is available? | Tolerance/hyperparameters selected before sealed evaluation |
| Hard nullspace retention | Does selective geometry permit a useful realized edit? | Record feasibility and exact residuals; do not discard no-ops/abstentions |
| Head retraining and appropriate low-rank/full-training controls | Is editing preferable to a tuned adaptation method? | Publish head-only and larger-training scopes separately; compare cost frontiers when identical budgets cannot be matched |
| Live proposer with access ablations | What does a live model gain from weights? | Actual provider trials, identical semantic context, explicit exposed information and permitted computation |

A fitted compiler may be one arm, but it cannot be described as a no-fitting
weight interpreter. A proposer denied forward calls may still reconstruct a
forward pass from exact weights if allowed general computation. State the
enforceable policy; broker logs alone cannot prove that prohibited outside
computation never occurred. Opaque candidate handles should prevent an editing
submission from revealing exact hidden parent values in a rounded-weight arm.

### Outcomes that count as impact

The primary outcomes are paired target queue accuracy/gain, protected queue
error increase, all-assigned-trial joint useful success, coverage/abstention,
and full workflow cost. Also report per-intent harms, nearby-confusion harms,
decision flips and margins, calibration/OOD behavior, and deployed numerical
differences. Score residuals remain mathematical/numerical diagnostics alongside
these behavioral metrics; neither patch size nor test-row count is business value.

Freeze `Delta_min` (minimum worthwhile target gain), `epsilon` (maximum protected
harm), `c_min` (minimum useful coverage), and resource limits from a separate
pilot or an explicit stakeholder constraint before inspecting confirmatory
outcomes. The roadmap deliberately does not invent universal numerical margins.
The benchmark issue requires concrete values and a selection rule before runs.

Use a primary paired contrast and confidence bounds suited to tasks, training
lineages, editor trials, and reused request sets. Do not count retries, multiple
outputs of one proposal, or rows from one edit as independent editors. Preserve
crossed factors where present; report inadequate independent units descriptively.
Plan sample size using pilot variability and the chosen effect margins, not a
large arbitrary test count. Keep abstentions, failures, timeouts, rejected edits,
and missing outcomes in the assigned population; publish reason-specific
missingness and prespecified sensitivity bounds, rather than a favorable
complete-case subset.

Reduced misrouting on benchmark queries can establish routing benefit in the
derived task. It does not establish fewer transfers or faster resolution in
production. An eventual authorized shadow deployment needs event-level queue
ground truth, delayed outcomes, a declared counterfactual design, and protected
operational state. Any scenario calculation of saved time must label its transfer
and handling-time assumptions; measured savings require actual operational data.

### Stop and go decisions

| Finding | Decision |
| --- | --- |
| Exact protected span is full rank or targets conflict | Abstain in the exact-score lane; evaluate a separately declared approximate/decision contract or conclude the request is infeasible |
| Target-only gain disappears on sealed text | Reject anchor-overfit claims and investigate representation coverage; do not tune on that holdout |
| Rule/overlay achieves equivalent utility and harm at lower cost | Recommend the overlay for this policy case; retain weight editing as a research result without an enterprise advantage claim |
| Live raw-weight access adds no worthwhile gain over identical context/summaries | Do not claim general-purpose weight-reading benefit; examine task cues and compression before scaling |
| A real bound passes but native decisions violate the deployed rule | Block the native-guarantee claim and identify arithmetic/correspondence assumptions; keep empirical/proof lanes separate |
| Bounds are sound but too loose to admit useful cases | Report reduced certificate coverage; compare tighter supported bounds without treating rejection as a behavioral counterexample |
| Gain, harm, coverage and total cost meet frozen criteria on untouched units | Advance to a second trained domain and bounded sequential cases; claim only the evaluated task/access/model scope |

## Execution order and evidence gates

Priorities are dependencies of claims, not a demand to finish every proof before
running an honestly labeled empirical study. The issue index supplies actual
GitHub dependencies. No calendar estimate or implementation completion is implied.

| Gate | Work to complete | Evidence needed to advance |
| --- | --- | --- |
| A: credible experiment | Immutable candidate/contract binding; declared numeric lane; frozen benchmark and attempts ledger; trained routing pilot | No artifact substitution; reproducible training/data mapping; concrete effect/harm/coverage margins; no confirmatory selection leakage |
| B: useful trained editing | Routing comparisons; coverage/feasibility; sealed outcomes; complete cost accounting | A publishable gain/harm/cost frontier, including negative results and simpler alternatives |
| C: live understanding and sustained edits | Brokered access/provider trials; blind causal audit; cumulative sequences | Added value of weights at matched context/access; intervention effects; retention across whole independent lineages |
| D: broader framework claims | Second trained non-text task; justified operator extension; checked runtime proof instances; supported scaling experiments | Cross-task evidence through shared interfaces and explicit semantic/backend boundaries |
| E: exploratory creation/transfer/unlearning | Restricted no-fitting construction; symmetry-aware transfer; bounded synthetic unlearning | Separate protocols with admissible controls and explicit falsification; no universal capability or erasure claim |

Formal instance work can begin in Gate A on small supported cases and continue
through D. A candidate-specific machine-checked claim waits for that bridge.
An empirical routing claim can proceed earlier with its proof/backend gaps
explicitly disclosed. Infrastructure changes should be driven by demonstrated
admission failures, restrictions, or profiling results.

## Coverage of the original future directions

The report [S1] contains five listed possibilities plus a benchmark/harness
commitment. Splitting steering from unlearning produces seven research threads.

| Original direction | Concrete next test | Gate / issue key |
| --- | --- | --- |
| Auditing without evaluation | Blind weight-based diagnosis; evaluator-only interventions after the prediction is frozen | C / causal-audit, live-access |
| Lower optimization cost | Matched utility/retention frontiers including full preparation, reader, verification and evaluation costs | B–D / trained-routing, total-cost |
| Zero-data steering | Declare supplied rules, labels, hints and computation; compare analytic and live no-fitting construction separately | E / no-fitting |
| Unlearning | Synthetic forget/retain reference, extraction/privacy/utility endpoints, transform/recovery attacks | E / unlearning |
| Recursive self-improvement | Defer until live benefit, causal repair and cumulative retention survive independent evaluation | No immediate build issue; reconsider after C and E |
| Architectural merging / circuit grafting | Known symmetry positive controls, aligned transfer, then held-out lineages; causal correspondence required before cross-architecture claims | E / transfer |
| Benchmark and harness | Distinct read/diagnose/edit/create rubrics, audited access, untouched units, full attempt accounting | A–C / benchmark, broker, live-access |

Recursive improvement is deliberately a decision gate rather than a vague
implementation issue. A later self-edit study must define which editor/model
identity changes, prevent evaluation leakage, account for all adaptive selection,
and compare with externally edited controls. Repeated improvement of one fixture
would not establish an open-ended self-improvement process.

## Coverage of existing limitations

| Existing work package | New executable issue keys |
| --- | --- |
| 1. Candidate/contract binding | binding |
| 2. Formal instances | proof-bridge |
| 3. Numerical execution | numerics |
| 4. Restricted live harness | broker, live-access |
| 5. Independent task/access tests | benchmark, trained-routing, live-access, second-domain, no-fitting |
| 6. Coverage/feasibility | feasibility |
| 7. Causal diagnosis/repair | causal-audit |
| 8. Cumulative sequences | sequences |
| 9. Operators/tighter bounds | semantics |
| 10. Scale/total cost | total-cost, transfer |
| 11. Source reconstruction/repetition | source-repetition |

The [requirements ledger](requirements.md) retains its present status. A
roadmap entry or opened issue is not evidence that a requirement is implemented.
Each work issue states its prerequisites, deliverables, acceptance/falsification
criteria, and the exact stronger claim it could unlock.

## Review method and limits

This is a targeted deep review of primary papers, the author's original report,
dataset repositories, and current framework code/documents, not an exhaustive
systematic review. Search focused on nullspace editing, realistic knowledge-edit
evaluation, sustained editing, causal localization, weight symmetries, numerical
formalization, repair certificates, intent routing/OOD, adaptation cost, and
unlearning recovery. Relevant papers through the local review date were included;
we do not claim the search found every recent publication.

Downloaded PDF identities were checked against their title pages. Two misleading
search leads were excluded when the linked documents proved unrelated; see the
source register. Selected methods, tables and limitations were inspected. Visual
inspection included AlphaEdit page 5, Mirage page 4, and TorchLean page 19.
The register records reviewed sections and file-page indices, version URLs,
retrieval timestamps, and byte hashes. A hash identifies acquired bytes; it does
not validate a paper's claims. Unversioned acquisition URLs are accompanied by
the version printed in the acquired PDF where available.

Only original analysis and source metadata are published. Full PDFs/extractions
remain private working material. We did not rerun external algorithms, benchmark
new models, audit external implementations, or measure production outcomes as
part of this documentation task. The 2026 preprints remain research leads whose
claims need independent testing. Their limitations are retained below.

## Primary-source register

The [machine-readable source review](research-sources-2026-11.json) contains
acquisition identity, reviewed scope, findings, limits and framework implications.
PDF pages below are file pages, which can differ from printed proceedings pages.

### S1. Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights

[Author report/blog, November 2026](https://rikiyat.github.io/blog/weight-space-programming/). Reviewed: What Comes Next?; access protocols and released-source boundaries through the repository source review; file pages not a PDF.

The report motivates direct reading, writing, auditing, creation, cost reduction, unlearning, recursive improvement, and circuit grafting. **Limit:** Selected demonstrations and author hypotheses do not establish population-level reliability, causal interpretability, universal preservation, or efficiency at matched outcomes. **Next test:** Use its future directions as hypotheses with separate access, intervention, retention, and cost tests; preserve source reproduction tiers.


### S2. AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models

[ICLR 2025](https://arxiv.org/pdf/2410.02355v4). Reviewed: Nullspace projection, sequential-edit objective, eigenvalue cutoff, limitations; file pages 2, 4, 5, 10.

Projecting an update against retained linear keys reduces interference under the specified representation. The practical method uses an eigenvalue threshold of 1e-2. **Limit:** A numerical near-nullspace over sampled keys is not exact preservation of every input or every downstream decision. Transfer to reasoning and multimodal models remains outside its experiments. **Next test:** Measure rank, conditioning, threshold sensitivity, exact endpoint residuals, and held-out behavior instead of calling a tolerance pass an exact guarantee.


### S3. Locating and Editing Factual Associations in GPT

[Primary research paper, 2022](https://arxiv.org/pdf/2202.05262v5). Reviewed: Causal tracing and intervention method; file pages 1, 2, 3, 4, 5.

Localization is tested through interventions and subsequent factual edits, providing a model for testing a diagnosis rather than accepting a plausible explanation. **Limit:** Evidence for a particular factual-association setting does not establish a universal causal circuit or cross-architecture transfer. **Next test:** Freeze predicted repair locations and effects, then compare with sham, random, and matched interventions on independent cases.


### S4. Mass-Editing Memory in a Transformer

[ICLR 2023](https://arxiv.org/pdf/2210.07229v2). Reviewed: Batch editing method and evaluation setting; file pages 1, 2, 3.

Multiple factual associations can be edited jointly in the studied transformer models. **Limit:** Batch insertion performance is different from preservation across a long adaptive sequence; knowledge editing is different from enterprise queue routing. **Next test:** Use as a relevant editing control where architecture/task fit; retain all previously inserted targets in sequential evaluation.


### S5. The Mirage of Model Editing: Revisiting Evaluation in the Wild

[ACL 2025](https://aclanthology.org/2025.acl-long.745.pdf). Reviewed: WILD evaluation modules, limitations; file pages 2, 4, 9, 10.

Teacher forcing, target-length truncation, context-free prompts, and permissive matching can inflate reported editing success; natural decoding exposes failures. **Limit:** The study examines representative methods and models; its findings do not establish universal failure of every future editor. **Next test:** For generated-output tasks freeze autoregressive decoding, natural termination, prompt/context variation, and explicit scoring. For routing, evaluate complete predictions on independent text.


### S6. Benchmarking and Rethinking Knowledge Editing for Large Language Models

[2025 preprint version reviewed](https://arxiv.org/pdf/2505.18690v1). Reviewed: Task/access comparison and contextual reasoning alternative; file pages 1, 2, 3.

Evaluation spans fact, event, and more general tasks, with reliability, generalization, locality, and portability; a contextual reasoning baseline tests whether weight changes are needed. **Limit:** Reported comparisons depend on task, prompt, compute, and model choices; a context baseline is not proven best for every application. **Next test:** Compare weight edits with rules, a routing overlay, and context/retrieval methods when task-compatible, using the same allowed evidence and full costs.


### S7. ForgetBench: Benchmarking Forgetting Dynamics of Long-Term Parametric Memory in Language Models

[July 2026 preprint](https://arxiv.org/pdf/2607.26455v2). Reviewed: Time-indexed retention evaluation and limitations; file pages 1, 5, 9, 13, 14.

An accuracy matrix indexed by edit time and memory age separates insertion success from later decay and variability. **Limit:** Synthetic and semi-synthetic knowledge-edit streams do not establish production retention or arbitrary continual learning performance. **Next test:** Evaluate whole edit lineages, original-baseline retention, old targets, conflicts, and task order; report retention curves with sequence-level uncertainty.


### S8. Measure, Don’t Optimize: Forecasting Recovery in LLM Unlearning

[August 2026 preprint](https://arxiv.org/pdf/2608.11408v1). Reviewed: Recovery forecasting rationale and audit/optimization distinction; file pages 1, 2.

Internal accessibility measurements can predict recovery in the studied settings; optimizing an audit signal can obscure the robustness it was intended to measure. **Limit:** A scoped empirical audit is not a universal test of information erasure or an implementation we have reproduced. **Next test:** Keep blinded recovery tests independent of the edit objective and disclose exactly which recovery attacks and budgets were evaluated.


### S9. MUSE: Machine Unlearning Six-Way Evaluation for Language Models

[2024 primary paper version reviewed](https://arxiv.org/pdf/2407.06460v2). Reviewed: Six evaluation endpoints; file pages 1, 2, 3.

Verbatim memorization, knowledge memorization, privacy leakage, utility, scalability, and repeated requests are separate unlearning outcomes. **Limit:** Passing one endpoint or benchmark is insufficient for general erasure or privacy guarantees. **Next test:** Use distinct forget/retain/extraction/privacy/cost/repetition endpoints and a retain-only reference in a gated synthetic unlearning study.


### S10. TOFU: A Task of Fictitious Unlearning for LLMs

[2024 primary paper version reviewed](https://arxiv.org/pdf/2401.06121v1). Reviewed: Synthetic identities, retain-only reference, truth-ratio evaluation; file pages 3, 7, 8.

Fictitious data permits a controlled reference trained without forget examples and distributional comparison of forgetting behavior. **Limit:** Synthetic author identities and benchmark behavior do not demonstrate deletion of real confidential data from arbitrary pretrained models. **Next test:** Begin with synthetic forget data; distinguish refusal or suppression from residual accessibility and preserve retain utility.


### S11. Catastrophic Failure of LLM Unlearning via Quantization

[ICLR 2025](https://arxiv.org/pdf/2410.16454v3). Reviewed: Quantization recovery experiments; file pages 1, 6.

Deployment quantization can recover knowledge suppressed by the evaluated unlearning methods. **Limit:** The result is scoped to studied models, methods, and transforms; it is not a universal mathematical impossibility result. **Next test:** Evaluate forgetting after deployed numerical transforms and bounded recovery attempts; do not infer erasure from the original full-precision output.


### S12. TorchLean: Formalizing Neural Networks in Lean

[May 2026 v2 preprint](https://arxiv.org/pdf/2602.22631v2). Reviewed: Shared graph semantics; Table 3 scope; finite Float32 refinement and external boundaries; file pages 6, 8, 19, 54, 55.

Shared SSA/DAG semantics connect execution and analysis. The scope table distinguishes proved, checked, and external components; finite-path IEEE32Exec refinement is explicitly restricted. **Limit:** The paper does not discharge the target CPU/CUDA kernel and hardware link universally. Repository code was not built or audited in this review. **Next test:** Evaluate reuse of explicit semantic/proof subsets; keep serializer, compiler, backend, nonfinite values, and unsupported operators visible in the trust boundary.


### S13. Efficient Neural Network Robustness Certification with General Activation Functions

[2018 primary paper version reviewed](https://arxiv.org/pdf/1811.00866v1). Reviewed: Activation relaxations and certified bounds; file pages 1, 3.

Activation envelopes yield conservative bounds for a specified input domain and supported network. **Limit:** Loose bounds can reject safe candidates; absence of a passing bound is different from an observed violation. The paper is not a proof of this repository runtime. **Next test:** Compare current sequential drift bounds with tighter validated relaxations using coverage, width, time, and memory under identical domains.


### S14. Provable Repair of Deep Neural Networks

[2021 primary paper version reviewed](https://arxiv.org/pdf/2104.04413v2). Reviewed: Decoupled architecture and one-layer linear-program repair; file pages 2, 4.

Separating activation and value paths permits a one-layer repair problem to be expressed as an LP in the specified setting. **Limit:** This tractability does not extend automatically to arbitrary ordinary networks, unknown activation regions, or a general editor. **Next test:** Explore a narrowly supported repair adapter and benchmark certificate coverage before promising arbitrary nonlinear repair.


### S15. Equivariant Deep Weight Space Alignment

[2024 v4 primary paper version reviewed](https://arxiv.org/pdf/2310.13397v4). Reviewed: Permutation-equivariant matching and scope; file pages 3, 13.

Symmetry-aware alignment helps compare or merge compatible parameterizations. **Limit:** Alignment is not proof of a causally transferable circuit between unrelated architectures. **Next test:** Test known same-function permutations first, then independent trained lineages; compare raw, aligned, and matched controls before circuit-grafting claims.


### S16. Scale Equivariant Graph Metanetworks

[2024 primary paper version reviewed](https://arxiv.org/pdf/2406.10685v2). Reviewed: Scaling and permutation symmetry assumptions; file pages 1, 2.

Weight processing can incorporate valid scaling and permutation symmetries for supported model families. **Limit:** Scaling symmetry depends on activation and graph structure; positive rescaling is not an unrestricted equivalence for tanh or arbitrary models. **Next test:** Declare symmetry groups per adapter and include function-preservation positive controls and unsupported-family rejection.


### S17. Permutation Equivariant Neural Functionals

[2023 primary paper version reviewed](https://arxiv.org/pdf/2302.14040v3). Reviewed: Weight/gradient processing architecture and tasks; file pages 1, 2.

Permutation-equivariant processing provides a learned alternative to raw coordinate interpretation for weights and gradients. **Limit:** Success on the evaluated neural-functional tasks is not a general-purpose language model understanding arbitrary weights. **Next test:** Compare symmetry-aware summaries/learned readers with raw weights and context-only readers on held-out training lineages.


### S18. Equivariant Architectures for Learning in Deep Weight Spaces

[2023 primary paper version reviewed](https://arxiv.org/pdf/2301.12780v2). Reviewed: MLP symmetry and implementation/training limits; file pages 1, 2, 9.

Architectures that respect MLP parameter symmetries support functions over a collection of trained networks. **Limit:** Architecture specificity, initialization, training difficulty, and implementation complexity constrain generalization. **Next test:** Hold out model lineages and architecture families; charge reader training and model-zoo construction in cost claims.


### S19. Towards Scalable and Versatile Weight Space Learning

[2024 primary paper version reviewed](https://arxiv.org/pdf/2406.09997v1). Reviewed: Token/subsequence processing of trained-model collections; file pages 1, 3, 4.

Subsequence processing and learned representations can make model-zoo weight analysis scalable. **Limit:** Compression, conditioning, data collection, reader training, and reconstruction create costs and information losses. **Next test:** Profile dense weight access first; evaluate summaries/chunks at matched downstream performance and include amortization assumptions.


### S20. LoRA: Low-Rank Adaptation of Large Language Models

[2021 v2 primary paper version reviewed](https://arxiv.org/pdf/2106.09685v2). Reviewed: Low-rank adaptation baseline; file pages 1, 2.

Low-rank fitting is an established adaptation route and a useful comparator when architecture and task permit. **Limit:** Low parameter count or rank does not establish retained behavior, cheap total workflow, or no-fitting construction. **Next test:** Include head retraining and appropriate low-rank/full-training controls with tuned development budgets and retention/cost measurements.


### S21. An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction

[2019 primary paper](https://arxiv.org/pdf/1909.02027v1). Reviewed: Dataset task and out-of-scope evaluation; author repository README/LICENSE; file pages 1, 2, 3, 4, 5, 6.

A public English single-intent dataset includes explicit in-scope and out-of-scope queries for text routing experiments. **Limit:** Crowdsourced single-intent queries are not multi-intent tickets, live customer telemetry, or our existing outage policy. **Next test:** Use as a cross-dataset/OOD routing extension; freeze a derived queue mapping and retain its synthetic-policy provenance.


### S22. Efficient Intent Detection with Dual Sentence Encoders

[2020 primary paper](https://arxiv.org/pdf/2003.04807v1). Reviewed: Fine-grained banking intent task; author repository README/LICENSE; file pages 1, 3.

Public fine-grained banking service intents provide a tractable trained text-classification starting point. **Limit:** The dataset supplies intents, not enterprise operational queues or a telemetry-outage policy. Queue assignment is a task derived by this project. **Next test:** Train a real text head, derive and freeze an ownership-change policy, and compare with an overlay operating on the unchanged intent predictions.


### S23. Selective Classification for Deep Neural Networks

[2017 primary paper version reviewed](https://arxiv.org/pdf/1705.08500v2). Reviewed: Risk/coverage formulation and assumptions; file pages 1, 2.

Abstention trades coverage for conditional prediction risk under the studied evaluation setting. **Limit:** Calibration and in-distribution bounds do not establish arbitrary OOD correctness, and low risk can conceal near-zero coverage. **Next test:** Report all-trial useful success, admission/coverage, conditional harm, and abstention reasons together; freeze thresholds on development data.
