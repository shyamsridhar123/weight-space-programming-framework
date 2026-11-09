# Prospective rubrics for reading, diagnosis, editing, creation and transfer

Defined **November 9, 2026, America/Chicago**. These are project protocol
definitions for five distinct claims. They complete the rubric fields without
assigning an experimental population, choosing numerical margins or reporting
new outcomes. A definition is not a frozen runnable study. Before a study starts,
its actual configuration must resolve every required field below, identify its
assigned units and permitted data, and be frozen before proposer calls,
candidate selection or confirmatory scoring. Unresolved fields block a
confirmatory claim; they cannot be filled from the resulting scores.

The [research roadmap](../weight-space/v2/research-roadmap-2026-11.md) and
[primary-source review](primary-paper-review.md) motivate these distinctions.
The [data-role account](evaluation-data-roles.md) records the historical c7
failure and the separate implementation scopes. These rubrics neither revise
that evidence nor close Issue #2 criteria c2 or c7. Existing reproductions and
synthetic pilots retain their original protocols and population descriptions.

## Required freeze and common accounting

Each study must freeze an access manifest, source/ancestry ledger, assignment
table, candidate-selection rule, comparison algorithms and tuning budgets,
scoring code and runtime, endpoint definitions, numerical margins, resource
ceilings, missingness rules and analysis plan. The freeze must identify exact
source/configuration bytes and the candidate artifacts used by every arm. A
local content-bound freeze is recorded as such; it is not described as external
preregistration or independent execution attestation.

The access manifest must enumerate architecture and forward-code descriptions,
input/output meanings, task rules, prompts, examples and labels, tensor values
and precision, summaries, gradients, activations and every permitted tool. It
must say whether target computation reconstructed from exposed weights is
allowed. A restriction on a named forward tool is recorded separately from a
restriction on equivalent computation. It must record preparation, training,
fitting, search, human assistance and all inference computation for readers,
editors, compilers, aligners and baselines. A method with a fitted preparation
stage belongs in a fitted arm even if its final submission uses no gradient
updates. Claims such as zero labeled examples or no target execution must apply
to the whole declared stage and access scope.

Separate model training, method development, target examples, protected anchors,
calibration/selection and sealed evaluation. Audit exact aliases, declared
ancestry, templates, semantic near duplicates and task families before splitting.
All checkpoints, edits, exports, permutations, rescalings, re-encodings and
derived examples from a source retain their parent identifiers. Exact hashes
enforce recorded identity; they do not supply a semantic independence audit.
Unknown provenance is retained as unknown. Evaluation results used to choose a
prompt, threshold, explanation, patch, alignment or baseline invalidate those
units for confirmation and enter development accounting.

An assignment is a predeclared attempt by a method under an access arm on a
task/model unit. Preserve every assignment, including an unattempted remainder
after a stop. Retain all submitted candidates, no-ops, abstentions, rejects,
failures, timeouts and reasons; never replace the assigned denominator with
successful submissions. Freeze either one terminal submission per assignment
or an explicit bounded selection rule using development feedback only. Log all
intermediate proposals and costs. Freeze retry and stop rules before execution;
infrastructure failures do not silently create new independent attempts. The
durable registered scoring boundary additionally forbids retry of the same
reserved observed evaluation scope after success, failure or uncertainty.

For behavioral rubrics, let `Delta` be the paired improvement on the target
endpoint, `H` the increase in protected harm and `C` the fraction of assigned
attempts returning an eligible terminal artifact. Freeze numerical
`Delta_min > 0`, `epsilon >= 0`, `c_min` and resource limits using independent
development/pilot data or an explicit stakeholder requirement, with its source
recorded. Tasks that use a loss must freeze its orientation and scale. Reading
and diagnosis additionally require explicit accuracy/calibration margins;
creation requires an absolute task-quality floor. A structural check, a small
patch or a proof residual does not substitute for a behavioral useful-effect
margin. Freeze the primary contrast and joint rule, including any uncertainty
bound used to accept it. Designate all other contrasts as secondary and specify
multiplicity handling before outcomes.

Report all-assigned joint usefulness, artifact coverage and conditional quality
among eligible artifacts together. A no-op is an eligible artifact only if it
meets the frozen validity rule; it receives its measured unchanged behavior and
no automatic useful-success credit. An abstention, reject, failure, timeout or
missing primary endpoint receives joint useful success `false`. Its hypothetical
behavior remains missing, rather than being imputed as a measured failure. If a
deployed fallback is evaluated, report that system outcome separately and name
the fallback rule; do not credit fallback performance to a missing proposal.

Freeze reason-specific sensitivity analyses. For bounded endpoints, publish
worst/best case bounds for all missing assigned outcomes using the declared
endpoint support; for unbounded costs/losses, define justified bounds or retain
the inference as unidentified. Include failures, unknown usage and timeout
ceilings in cost summaries. Report stricter duplicate/ancestry exclusions,
uncertain-exposure exclusions and leave-one-family/lineage-out results without
redefining the primary population. Use paired, hierarchical or crossed analysis
for the actual repeated units, with the specified confidence procedure and
sample-size rationale frozen from independent information. With inadequate
independent units, report descriptive uncertainty and an insufficient-evidence
decision. Rows, rollouts, repeated checkpoints and retries are nested
measurements, not extra independent readers, editors or training lineages.

## Reading rubric

**Claim and scored output.** The reader predicts a predeclared property of an
unseen model or its behavior. Freeze an answer schema that separates identity
or category, detailed behavioral properties, confidence and any mechanistic
account. Define evaluator-only truth for each property and a scoring rule for
wrong, incomplete and overconfident answers. A correct category cannot make an
incorrect property description correct. Explanations receive a separate score;
their plausibility alone cannot satisfy a behavioral prediction endpoint.

**Allowed context, weights, examples and computation.** All arms receive the
same declared architecture, input/output semantics, candidate descriptions and
task context. Freeze weight access as separate no-weight, summary,
rounded/subset and exact-value arms where applicable. Specify tensor selection,
precision, ordering and token limits before outcomes. Declare whether readers
may execute the target, reconstruct its computation, inspect activations or
gradients, fit a decoder or consult examples; each such condition is a separate
access arm with its own cost. An anonymous-weight claim requires an explicitly
anonymous arm. A supplied rule or labeled target example remains disclosed
information, even if no new target forward is exposed.

**Independent unit, ancestry and split policy.** The primary model unit is an
independently trained source lineage, paired across access arms. Underlying
task/source identities are an additional grouping factor when multiple models
encode the same task or example. Keep all checkpoints and symmetry variants
of a lineage in one split, and all copies of the same underlying source example
in one source group. Hold out the model lineages required for the claim; a
new initialization of one known task tests initialization variation, not a new
task family. A claim about unseen architectures requires a separately frozen
architecture split. Reader training and property-label construction remain
development/preparation, with no final-unit labels exposed during selection.

**Candidate selection.** Freeze the prompt, reader checkpoint and output schema
using development models. Take one terminal answer per assigned unit/arm, or
freeze a bounded development-only ensemble/selection rule. No answer rewriting,
confidence recalibration or choice among descriptions after evaluator truth is
revealed. Record abstention as an explicit answer option.

**Comparisons.** The primary contrast is the chosen weight-access reader minus
the same-context no-weight reader on the paired property score. Include a
development-derived prior/simple reader and, where task-compatible, a permitted
behavior-probe reader. Charge probe calls and fitting. A trained symmetry-aware
reader and a pointwise/statistical reader must use the same eligible training
models and labels. Task clues and context are held constant in the access
contrast; changes in those inputs are separate contrasts.

**Useful-effect margins.** Freeze minimum absolute property accuracy/quality,
minimum paired gain `Delta_read_min`, maximum calibration error or a minimum
improvement in a specified proper score, minimum coverage `c_read_min` and
resource limits. Report the confidence criterion for these margins. Detailed
property accuracy and all-assigned usefulness are primary components; generic
fluent explanation scores are secondary. Mechanistic correctness is required
only when independently testable ground truth/interventions are frozen, and is
reported separately from ordinary reading success.

**Abstention, missingness and sensitivity.** Abstentions count in assigned
coverage and receive no useful-reading credit. Invalid schemas, missing truth,
timeouts and unknown exposure keep their reasons. Apply bounded-score
sensitivity to the assigned population and report both conditional accuracy and
all-assigned usefulness. Prespecify sensitivity to context-only clues, stricter
task-family/source grouping, weight precision and symmetry variants; these
analyses cannot become post hoc primary winners.

## Diagnosis rubric

**Claim and scored output.** Before intervention feedback, the diagnostician
identifies a fault/mechanism and its tensor or functional location, predicts
the direction and magnitude of specified intervention effects, states
confidence and submits a literal intervention or abstention. Freeze the
mechanism/location answer key and causal endpoints separately. Localization,
mechanism identification, calibrated predictions and successful retained repair
are distinct scores. Natural faults and controlled planted faults have separate
population claims.

**Allowed context, weights, examples and computation.** Freeze architecture,
task rules, target examples and any observable failure evidence equally across
arms. Declare weight views, allowed forward/activation/gradient calls, diagnostic
probes, fitting and proposal budgets. For a blind planted study, the clean
reference, injection seed/ledger and exact inverse remain evaluator-only; neither
the prompt nor exposed file names may reveal assignment truth. Diagnostic
feedback comes only from declared development/target examples. The causal test
set and evaluator interventions stay sealed until all diagnoses and repairs in
the declared comparison set are terminal and frozen.

**Independent unit, ancestry and split policy.** Group by independently trained
parent lineage and fault-generating source/mechanism family, with crossed
assignments where actually present. Different locations or strengths on one
parent remain dependent derivatives. Keep clean parents, injections,
checkpoints and all proposed interventions in their lineage group. Select
faults and target cases from development or a predeclared generator before final
outcomes. An unseen-location split does not imply an unseen mechanism, and
controlled injections do not supply a population of naturally occurring faults.

**Candidate selection.** Freeze a single diagnosis-plus-intervention submission
per assignment or a bounded development-feedback rule. Diagnose and select the
repair before evaluator-only interventions. Freeze how to handle uncertainty,
multiple candidate locations and absent interventions; never choose the most
effective submitted intervention using causal test outcomes. Parser repairs or
human amendments follow a predeclared rule and remain in the attempt record.

**Comparisons.** Compare the submitted intervention with the faulty unchanged
parent, a sham intervention, a random-location intervention and a prespecified
support/norm-matched location-shuffled intervention where those matches are
meaningful. Include the same-context no-weight diagnostician and an eligible
ordinary diagnostic method. Clean reference/exact inverse is an evaluator
oracle ceiling, never a competing non-oracle editor. Freeze randomization,
matching and seeds before outcomes. The primary causal contrast is repair
gain beyond the declared matched control, with target gain relative to the
faulty parent and protected harm also required.

**Useful-effect margins.** Freeze location/mechanism accuracy floors,
calibration/predicted-effect tolerances, `Delta_diag_min` beyond the primary
matched control, target-repair gain floor, protected harm cap `epsilon_diag`,
coverage `c_diag_min` and total-cost ceiling. A causal-success rule requires the
predicted signed effect and a useful observed effect on untouched units under
the frozen retention rule. Correct localization without that result is reported
as localization success with causal repair unestablished.

**Abstention, missingness and sensitivity.** Uncertain diagnoses and absent
repairs remain assigned, with no joint causal-success credit. Retain negative
effects, sham failures and invalid matched controls; do not replace a difficult
injection or failed parent after outcomes. Apply bounded gain/harm and
calibration sensitivity for missing endpoints. Prespecify analyses by mechanism,
lineage, injection strength and alternative valid matching rules, along with
the stricter unseen-mechanism claim boundary. If a control cannot be validly
matched, that causal contrast remains missing/unidentified.

## Editing rubric

**Claim and scored output.** Starting from a frozen parent, produce a realized
candidate that improves a declared target behavior while respecting a protected
behavior contract at a worthwhile total cost. Freeze target and protected
endpoint definitions, the exact editable tensors, candidate realization and
native runtime. Report mathematical constraints, numerical residuals and native
behavior as distinct evidence. Target-policy changes and recognition/behavior
repairs are separately assigned task types.

**Allowed context, weights, examples and computation.** Freeze common task and
policy context, permitted weight views, target examples, protected anchors,
feedback and development data. Declare gradients, solver calls, retraining,
low-rank adaptation, rules/context tools and human assistance for each arm.
Every arm receives the same eligible semantic information; equal pass counts
also require recording examples, batch sizes, tokens, duration and hardware.
No-fitting and no-forward conditions state their full computation policy.
Evaluator inputs, labels and protected final regions are unavailable during
selection.

**Independent unit, ancestry and split policy.** The editing process is repeated
on independently trained parent lineages and independently assigned task cases,
with editor attempts nested/crossed as actually assigned. All patches, retries,
checkpoints and architecture exports retain the parent lineage. Target and
protected final examples must be split by source/family and ancestry from
training, anchors and selection examples. A new head seed on a shared encoder
is recorded as head-training variation with common encoder ancestry. Recognition
failures must be chosen from development by a frozen rule, not discovered on
the confirmatory split.

**Candidate selection.** Freeze one terminal patch or a bounded sequence using
development feedback and a deterministic selection rule. Seal all terminal
candidate identities before final scoring. Record infeasibility, no-ops,
rejected constraints, parser changes and stop reasons. Feasibility thresholds
and soft-retention hyperparameters cannot be adjusted using final outcomes.
If sequential editing is the task, freeze edit order and separate each stage's
unexposed evaluation scope; repeated scoring of a consumed scope is not a new
freshness claim.

**Comparisons.** Include the unchanged parent, task-compatible rules/overlay,
context/retrieval, target-only fitted edit, soft-retention edit,
hard-retention edit and appropriate head/low-rank/full-training controls.
Freeze the primary competing method and its tuning rule. Compare under equal
eligible information and preparation budgets; if budgets cannot be matched,
report frozen quality/harm/cost frontiers rather than selecting a favorable
single point. An ownership-policy task must allow the unchanged prediction plus
policy overlay to use the same policy information.

**Useful-effect margins.** Require target gain `Delta_edit_min`, protected harm
at most `epsilon_edit`, coverage at least `c_edit_min` and declared resource
limits, including preparation, failed proposals, realization and scoring. Freeze
minimum advantage over the primary competitor in utility or cost at the same
retention/quality requirements. A business-advantage claim must specify its
operational endpoint and evidence separately; benchmark accuracy or score
preservation alone cannot satisfy that claim.

**Abstention, missingness and sensitivity.** Full-rank protection, conflicting
targets and failed realization can cause explicit abstention/rejection; retain
them in joint usefulness and coverage. Count no-ops with measured parent
behavior. Keep all failures/timeouts and missing costs, and report a named
fallback's system outcome separately. Freeze sensitivity to protected-anchor
coverage, semantic near duplicates, stricter lineage grouping, numerical
runtime and missing gain/harm/cost bounds. Report per-family harms and tail
failures alongside aggregate retention; do not remove harmed groups post hoc.

## Creation rubric

**Claim and scored output.** Produce a complete executable model from a frozen
task specification without an edited parent. Freeze architecture constraints,
input/output meanings, allowed artifact schema, native realization and task
quality endpoint. Formula-to-weight construction with supplied formulas is its
own task; a supplied formula is not inferred behavior. Steering an existing
parent remains editing even when it uses no labeled examples.

**Allowed context, weights, examples and computation.** Declare task rules,
architecture/forward descriptions, output semantics and any exemplars or labels.
There is no hidden parent to read; library weights, pretrained components and
retrieved templates are disclosed preparation/context. Freeze whether creation
is analytic, fitted, gradient-trained or search-based. A no-fitting arm forbids
regression/solver fitting and data-driven tuning in its declared stage, while
allowing only the explicitly declared algebra/code generation. Any target
execution, feedback, teacher, training examples or formula hints is recorded
and matched in comparisons. Final inputs remain evaluator-only.

**Independent unit, ancestry and split policy.** The independent task unit is
the source rule/problem family required by the claim, with creation attempts as
assigned repetitions. Parameter variants, generated points and multiple seeds
within one formula are nested units. Split shared templates, source problems
and their derivatives together. A held-out parameter or initialization split
tests that variation; unseen rule-family claims require an actual family split.
Training/fitting points, method-development problems and selection examples
remain separate from final problem units and evaluator points.

**Candidate selection.** Freeze the construction method/prompt and take one
terminal complete artifact per assignment, or a bounded development-only
selection rule. Record all generations, compilation attempts, deterministic
parser/realization repairs and their costs. No choosing the best artifact on
final evaluator points. If compilation fails or no complete model is supplied,
retain the assigned failure; an analytic baseline is not substituted as a
successful live-created candidate.

**Comparisons.** Include a prespecified analytic/hand-programmed construction
where available, an eligible simple architecture/template and a fitted or
trained method with declared data and budgets. A supplied formula must also be
available to the analytic comparison. Specify whether pretrained methods are
allowed and charge their preparation. Freeze the primary comparison; report
analytic exactness and learned approximation under their own declared task
contracts rather than implying that either is an unchanged-parent edit.

**Useful-effect margins.** Freeze an absolute quality floor or maximum task
error `tau_create`, structural/native validity, coverage `c_create_min` and
resource ceilings. Also freeze a worthwhile quality or total-cost advantage
over the primary competitor at a common quality requirement. For exact
construction, define the supported input domain, arithmetic and numerical
tolerance before execution; passing finitely many evaluator points is reported
as that empirical result unless a separate validated proof covers the domain.

**Abstention, missingness and sensitivity.** Missing, invalid, nonfinite and
unrealizable artifacts receive no joint creation-success credit. Retain unknown
usage, failed compilation and unattempted assignments after a stop. Apply
bounded quality and cost sensitivity or report unidentifiability. Freeze
sensitivity to rule-family grouping, template leakage, evaluator-domain
coverage and runtime precision. Formula-guided outcomes cannot be relabeled
as discovery of unseen formulas or general learned model construction.

## Transfer rubric

**Claim and scored output.** Transfer a frozen donor capability or intervention
to a recipient, with a specified alignment/adapter and realized recipient
artifact. Freeze the donor behavior, recipient target and protected endpoints,
editable scope and proposed mechanism. Coordinate alignment, interpolation,
feature correspondence and causal capability transfer are separate endpoints.
Compatible-architecture/symmetry transfer and cross-architecture transfer are
separate populations and contracts.

**Allowed context, weights, examples and computation.** Declare donor and
recipient architectures, input/output correspondence, exact/rounded/summary
weight views, donor behavior descriptions, permitted examples, aligner training,
recipient feedback, fitting and native forward/gradient calls. Match semantic
information across transfer and recipient-only arms. Freeze any adapter's
training population and cost; a learned adapter is fitted preparation. State
activation/architecture conditions for a proposed symmetry and validate native
realization separately. Final donor/recipient target and protected examples
remain sealed through donor, adapter and candidate selection.

**Independent unit, ancestry and split policy.** Retain independent donor and
recipient training lineages plus the underlying capability/task-family groups.
Repeated pairs sharing a donor or recipient are crossed dependent measurements,
not independent transfers. Keep every donor derivative, transformed checkpoint,
recipient edit and export in its ancestry group. Freeze lineage-disjoint
development/evaluation splits for the desired donor and recipient claims; an
unseen-pair split alone is insufficient when it reuses the same trained models.
Record shared pretraining explicitly. A cross-architecture claim requires
predeclared independently represented architecture pairs and task correspondence,
not only permutations or rescalings of one architecture.

**Candidate selection.** Choose donors, recipients, alignment, transferred
support and adapter hyperparameters using development information by a frozen
rule. Seal one terminal transferred artifact per assignment or a bounded
development-feedback selection rule before final scoring. Preserve all failed
alignments, incompatible pairs, no-ops and rejected realizations. Selecting a
donor or transfer location by recipient test gain consumes that evaluation
scope and cannot supply confirmatory transfer evidence.

**Comparisons.** Include the unchanged recipient, recipient-only adaptation
under the same examples/budgets, raw-coordinate transfer, a valid
symmetry/alignment-aware transfer where supported, sham transfer and a
prespecified random/unrelated-donor or support/norm-matched intervention.
Include task-compatible context/rule transfer when it uses the same capability
description. Freeze matching and negative-control selection before outcomes.
An evaluator oracle correspondence is a ceiling, not an ordinary transfer arm.
The primary causal-transfer contrast is target benefit beyond the chosen
matched/recipient-only control with protected retention also required.

**Useful-effect margins.** Freeze recipient target gain `Delta_transfer_min`,
minimum advantage over the primary control, protected harm cap
`epsilon_transfer`, coverage `c_transfer_min` and total preparation/alignment/
adaptation/runtime costs. Add a quantitative donor-behavior correspondence rule
if the claim concerns transferring that particular capability. Alignment
similarity or weight distance alone cannot satisfy this rule. A causal-circuit
claim additionally requires a frozen intervention prediction and matched
causal evidence; successful compatible-coordinate adaptation does not receive
that label automatically.

**Abstention, missingness and sensitivity.** Unsupported architectures,
ambiguous correspondence, failed alignment, adverse transfers and rejected
native realizations remain assigned. Give no-op and unchanged recipients their
measured behavior without useful-transfer credit unless the frozen gain rule
actually holds. Report bounded missing gain/harm sensitivity and unknown-cost
limits. Prespecify stricter donor/recipient/task disjointness, leave-one-donor
and leave-one-recipient analyses, valid symmetry/precision stress and negative
control sensitivity. If the observed units do not support independent
architecture/family transfer, retain a narrower descriptive claim.

## Readiness and evidence boundary

A future runnable protocol must supply actual values and sources for its
margins, real assigned identifiers/counts, split audits and ancestry limits,
access/budget details, exact baseline implementations, candidate-selection rule,
attempt/stop policy, endpoint support and complete missingness/analysis settings.
The five rubrics are now explicit definitions. No experimental population or
numerical pass threshold is invented here, and no new study is claimed to meet
them. The separate synthetic prospective runner demonstrates its own declared
cooperative execution path only. It cannot establish semantic independence,
complete historical-attempt capture, scientific freshness or production impact.
