# Limitations and a prioritized research program

A useful weight-space edit has several obligations. It must refer to the
intended model, survive realization into stored parameters, obey the declared
access rules, and produce the requested behavior while retaining what the
task requires. A mathematical preservation claim adds another obligation:
the actual artifact and domain must satisfy the theorem's premises.

The framework implements parts of this path. Its main research gaps are the
connections between those parts. This plan names the failures those gaps can
cause, the mechanisms behind them, and the evidence needed to resolve them.
It applies to reading, diagnosis, repair, creation, and transfer across
domains. Games are possible applications, alongside other uses of neural
models.

The plan builds on the original report's **What Comes Next?** discussion,
released experiment records, and earlier local limitations identified in the
[source review](source-review.md) and its
[prior-boundary table](source-review.md#prior-documented-boundaries-in-this-repository).
The [requirements](requirements.md), [workflow](workflow.md),
[formal guide](formal.md), [protocol](protocol.json), and
[study runner](../../../rlcd/weight_space/study_v2.py) specify the present
implementation and evidence. Proposed work below is not marked completed.

There is **no live language-model/provider study** in the current evidence.
All published study seeds, including worlds `7300–7311`, have been observed.
Running them again can reproduce the recorded study; fresh confirmation needs
new units and a protocol frozen before their outcomes are seen.

## What the present work can and cannot establish

The [small head-edit example](formal.md#a-small-edit-that-exposes-the-problem)
shows why the distinctions matter. Both proposed edits raise the target score
by one. Only the edit with zero dot product against the protected feature
preserves that input. If target and protected features coincide, exact
protection prevents the requested gain. The identity of the candidate, its
protection, and its utility are separate questions even in this tiny case.

| Area | Present capability or study design | Consequential remaining limit |
| --- | --- | --- |
| Representation | Named tensors, exact-parent literal edits, structured lowering, and caller-supplied semantics | Tensor acceptance does not define the forward computation, cover every operator, preserve hidden state, or certify arbitrary programs |
| Task routes | Distinct outputs for reading, diagnosis, editing, and creation; retained candidates, failures, budgets, selection, and evaluations | Each mode still needs a rubric; the deterministic sign fixture exercises answer routing without discovering a causal mechanism |
| Formal guarantees | Checked real-valued endpoint/span results and finite-depth affine/activation drift | New runtime candidates are not automatically bound to Lean instances; extractor correspondence and native execution remain unproved |
| Realization | Supported native projection, exact endpoint residuals, and supported PyTorch state round trips | Receipts do not model intermediate rounding, kernels, accumulation order, overflow, arbitrary storage layouts, or execution equivalence |
| Access and budgets | Capability checks and charges for work crossing session methods | Cooperative code in one process does not enforce the absence of outside fitting, forward computation, network access, private-data access, or selection |
| Algorithmic comparison | Fixed head-editing controls, public features, fitting, and selection before private evaluation | No live proposer; the gradient arm is a fixed head optimizer, without a fully tuned or compute-matched full-network training comparison |
| Architecture variants | ReLU MLP, tanh MLP, and convolution-plus-tanh extractors with linear heads in shared synthetic worlds | Shared teachers and splits make paired variants; random frozen features with trained heads do not establish end-to-end learning or transformer repair |
| Retention | Target/collateral metrics and exact residuals on supplied rows | The intervening input region is absent from the main metrics; anchors do not cover a whole domain by themselves, and real scores do not automatically measure reward |
| Precision and feasibility | Native-format and anchor-count ablations | They reuse the main holdout and are descriptive; precision conditions also project the parent, changing the baseline |
| Sequences | Recorded chains, intermediate behavior, endpoint composition, and rollback checks | Other-region retention targets refer to the current checkpoint while requested shifts refer to the original; no original-baseline cumulative retention contract is imposed |
| Uncertainty | Paired family effects aggregated within worlds, then world-level resampling | Intervals are exploratory and unadjusted; complete-world restriction can be biased by informative failures, and additional rows are not additional worlds |
| Source reproduction | Retained sources and hashes support inspection and some reconstruction prerequisites | Missing checkpoint bodies, observations, graders, work directories, and provider state still block stronger reproduction tiers |

Two older limits have partly changed. The previous
[design](../design.md) described a ReLU-only default registry and proposed
exact nullspace protection as future work. V2 registers sequential semantics
and supplies numerical nullspace proposals with exact realized-endpoint
residual receipts. The unresolved questions now concern operator coverage,
domain coverage, implementation correspondence, and useful behavior. The
[historical results](../results.md) keep their original interpretation.

## Priority and dependencies

Priorities indicate what evidence a claim needs next. They do not require
finishing every proof before running an honestly labeled empirical study. A
live access experiment can measure behavior while explicitly reporting that
implementation and numerical proofs are absent.

| Priority | Work package | Main requirement IDs | Decision it enables |
| --- | --- | --- | --- |
| P0 | [1. Bind candidate and contract](#1-bind-admission-to-one-exact-candidate-and-contract) | R05, R07, R27, R29, R32, R40 | Establish which artifact and claim each admission result concerns |
| P0 | [2. Generate formal instances](#2-generate-and-check-new-formal-instances) | R31, R34, R37–R39 | Attach a checked theorem instance to a new runtime candidate |
| P0 | [3. Account for numerical execution](#3-bound-or-explicitly-measure-numerical-execution) | R24–R28, R39 | State how a real bound relates to a specified native backend |
| P0 | [4. Build the restricted live harness](#4-build-an-auditable-restricted-and-live-provider-harness) | R06, R09–R20, R46 | Audit the information, work, attempts, and restrictions of an actual editor trial |
| P1 | [5. Test tasks and access](#5-run-independent-task-and-access-experiments) | R03, R04, R08, R41–R54 | Estimate what weights add beyond the same supplied context |
| P1 | [6. Test feature coverage and feasibility](#6-establish-feature-coverage-and-feasibility-limits) | R27, R33–R35, R39, R47 | Separate selective edits from anchor overfitting and conflicting requests |
| P1 | [7. Test causal repair](#7-test-causal-diagnosis-and-repair) | R04, R45, R53, R54 | Connect a frozen explanation to intervention effects and retained behavior |
| P1 | [8. Test cumulative sequences](#8-admit-and-evaluate-cumulative-edit-sequences) | R32, R55–R57 | Evaluate progress and retention across an explicit lineage contract |
| P2 | [9. Extend operators and bounds](#9-extend-operators-and-tighten-bounds) | R29–R31, R34, R40 | Expand a specified semantic family and measure useful certificate coverage |
| P2 | [10. Measure scale and total cost](#10-measure-and-improve-scale-and-total-cost) | R15, R23–R26, R42, R59 | Test efficiency at matched quality and retention |
| P2 | [11. Repeat selected source tasks](#11-reconstruct-and-independently-repeat-selected-source-tasks) | R01, R02, R18, R46, R54, R58 | State which reproduction tier has actually been achieved |

Packages 1–4 can proceed in parallel where their artifacts and interfaces are
independent. Packages 5–8 need frozen evaluation rules and untouched units.
New operators and storage formats should meet the same identity and evidence
requirements as existing ones.

## Common design for the next experiments

Before confirmation, specify the claim, population, allowed information,
allowed computation, comparison, independent unit, selection rule, and minimum
worthwhile effect. Include failures and abstentions in the planned population.
A local protocol freeze records when those choices were fixed. An external
preregistration is a separate action and evidence claim.

For a repair study, declare a target-utility metric `U` and retention-harm
metric `H`. Choose a minimum useful gain `Δmin`, maximum acceptable harm `ε`,
and resource limit before inspecting confirmatory outcomes. One possible
primary rule requires both a lower confidence bound on paired target gain
above `Δmin` and an upper confidence bound on harm below `ε`. Calibrate that
rule for the declared comparison family. Report raw paired effects and
uncertainty even when the binary rule fails. The current fixture's normalized
error thresholds do not provide universal margins for other tasks.

A preservation theorem has a different acceptance condition: its premises and
bound must hold for the declared artifact and domain. Finite sampled agreement
does not establish an all-input premise. An empirically useful candidate may
also fail a conservative certificate. Record the measured outcome and the
certificate result separately.

### Information, computation, and task axes

Labels such as “weights only” or “zero data” leave too much unspecified. Keep
the following factors separate when describing a trial or making a comparison.

| Axis | Conditions to distinguish | Consequence for the next design |
| --- | --- | --- |
| Semantic context | Architecture/interface; named inputs/outputs; task rules; training/corpus descriptions; explicit mechanism hints | Hold context fixed while adding weights, then ablate hints separately |
| Parameter view | No weights; summaries; exact values; rounded values; allowed coordinate subsets | Prevent submission feedback from revealing hidden exact state; account for information that remains visible after rounding |
| Labels and examples | None; unlabeled inputs; labeled public examples; worked solutions; evaluator specification | Disclose examples and fitting in the compiler; disclose supplied formulas and construction families in analytic creation |
| Target observations | None; scalar feedback; logits/scores; outputs; internal activations | Record actual exposed information and work; equal callback counts need not be equal access |
| Allowed computation | No tools; arithmetic weight analysis; target execution; analytic construction; regression/least squares; gradient fitting | Give each regime its own admissible methods and audit |
| Proposer | Fixed analytic program; fitted algorithm; scripted replay; live language model; human-assisted process | Only actual provider trials measure a live model's proposal ability; retain human involvement |
| Task | Identify a property; diagnose a mechanism; repair behavior; create a network; transfer an edit | Use distinct rubrics and outcomes rather than a common edit-success flag |
| Distribution | Familiar construction; unseen seed; unseen training lineage; unseen task variant; unseen dataset/operator/model family | Match the strength of the generalization claim to the units actually held out |

Two access details need care. First, the sign fixture mandates abstention at
either signed zero while preserving its sign bit. Rounding therefore does not
erase every visible sign cue. Second, rounded-weight editing can leak exact
parent values if the editor receives the realized candidate state. The current
read-only fixtures return answers; a future editing interface needs a scoped
submission route, such as opaque candidate handles.

The source Go protocol explicitly forbids regression and least squares; see
the [access analysis](source-review.md#no-training-tool-free-and-few-passes-need-operational-definitions).
The fitting-permitted compiler, a no-fitting constructor, and a tool-free LLM
belong in separately specified arms. Reimplementing the target forward pass
in a different tool still counts as target execution under a no-forward rule.
An analytic constructor given a formula can test a construction route. Its
success does not establish that a language model inferred that formula from
anonymous weights.

### Independent units, uncertainty, and missingness

More observations of the same candidate can reduce measurement noise without
adding independent editor trials or task families. The analysis must preserve
the unit at which its claim varies.

| Claim | Unit or structure to preserve | Repeated observations that do not replace it |
| --- | --- | --- |
| Current synthetic algorithm comparison | World, with paired arms and architecture variants inside it | Shared-teacher families, the selected-arm alias, and repeated test rows |
| Checkpoint generalization | Independent training lineage, with common pretrained ancestors recorded | Descendants, exports, or correlated fine-tunes counted as unrelated families |
| Editor variability | Independent editor trial with fixed provider/configuration, paired or nested within checkpoint/task | Retries, handoffs, candidates, and adaptive turns |
| Language-task accuracy | Problem, with answers nested within it and checkpoint/editor structure retained | Sampled answers treated as independent problems |
| Rollout reward | Episode randomness conditional on an editor/checkpoint; additional lineages/editors for broader claims | Episodes of one edit treated as independently proposed edits |
| Cross-task generalization | Held-out task family or dataset, with model and editor sampling specified | Rows or architecture variants from one generator |
| Sequence robustness | Whole lineage and assigned task order, retaining the time series | Edit steps treated as unrelated trials |

Use paired contrasts and resampling or hierarchical models that reflect actual
nesting and crossed factors. For repeated language answers, keep all answers
to a problem together and retain common problems across methods. Preserve
shared ancestry when analyzing descendants. If tasks, lineages, and editors
are crossed, do not force them into a single independent-sample analysis.
Report cluster counts and model assumptions. Keep estimates descriptive when
independent units are inadequate for the proposed inference.

Plan sample sizes from a separate pilot or a conservative variance assumption
and the minimum useful effect. A pilot used to choose prompts, budgets, or
hyperparameters is development data. Evaluating existing candidates more often
cannot remedy a lack of independent editors or task families.

Retain attempted, completed, admitted, selected, abstained, failed, and missing
counts. Report joint useful success over all assigned trials, admission rate,
and useful success conditional on admission. For missing behavioral outcomes,
retain reason-specific counts and prespecified conservative bounds or justified
sensitivity analyses. Do not silently impute scores or report only successful
complete cases. The current summary's complete-world restriction and
unadjusted exploratory intervals must remain visible.

Freeze a primary contrast or state the multiplicity method for a confirmatory
family. The current precision and feasibility ablations share a holdout and
remain descriptive. Transfer claims need entire held-out task or model
lineages, with problem separation within them as appropriate. Split-ID checks
are useful bookkeeping; they do not prove content independence, absence of
shared ancestry, or absence of pretraining exposure.

## 1. Bind admission to one exact candidate and contract

**Failure to prevent.** A valid certificate for one candidate is attached to
another candidate with the same shape. Its mathematics may be correct while
the admission decision concerns the wrong endpoint. Substituting a domain,
feature order, or semantic adapter creates the same problem.

The existing structural policy, older two-layer `Contract`, sequential
certificates, feature receipts, and workflow evaluation are separate
mechanisms. The [design](../design.md) and R05/R32 identify the integration
work. Create one contract record binding the parent and candidate identities,
interface, semantic adapter/version, domain, output coordinates, allowed
structural changes, semantic budgets, and any decision rule or feature
witnesses.

Retain each check's result and scope. Admission must distinguish invalid
input, unsupported semantics, missing correspondence, a conservative bound
above budget, an observed violation, and a passing bound. For strict argmax
preservation, retain the actual candidate set, winner, margins, and coordinate
budgets. Ties and stochastic sampling need their own semantics. Behavioral
acceptance remains a separately evaluated condition.

**Completion and falsification.** Independent recomputation must reproduce
the admission decision on the frozen record. Deliberate endpoint, domain,
adapter, coordinate-order, and witness substitutions must be rejected. An
accepted mismatch or silently certified unsupported backend falsifies the
binding claim. A sound but loose rejection reduces coverage; it is not a
semantic counterexample. Completion does not establish numerical execution
or task utility.

## 2. Generate and check new formal instances

**Failure to prevent.** A theorem is checked for hand-written coefficients,
while a runtime candidate contains different values or interprets them in a
different order. The proof does not yet concern that candidate.

The [formal guide](formal.md#what-the-proof-does-not-connect-yet) makes this
boundary explicit. Connect a deliberately supported subset of runtime
artifacts to the frozen-feature and sequential theorems, preserving R37–R39.
Generate exact rational definitions from immutable endpoints, the input
domain, activations, feature order, and bias convention. Independently
recompute artifact identities and check that generated values decode to the
same endpoints.

Each instance should include the numeric premises and the precise theorem
application. Pin the generator, generated source, Lean target, dependencies,
and logs. Preserve historical source identities and use the explicit generic
audit target. The parser/generator stays within the trust boundary until
checked or independently validated.

**Completion and falsification.** Require deterministic regeneration,
successful theorem application, and adversarial mutation rejection. Exercise
zero edits, simultaneous layer edits, and unsupported activations as well as
useful cases. Endpoint, shape, ordering, feature-ID, and domain mismatches must
invalidate the binding. One incorrectly bound candidate defeats the binding
claim. A successful instance still proves the stated real semantics; numerical
execution and behavior need their own evidence.

## 3. Bound or explicitly measure numerical execution

**Failure to prevent.** The stored endpoint has an exact real-valued residual,
but its native forward pass exceeds the claimed drift because intermediate
operations round, accumulate differently, or become nonfinite. Endpoint
conversion and execution using that endpoint are separate operations.

Start with a small supported backend whose dtype, operation order,
accumulation, and activation behavior can be specified. Compare requested
arithmetic, projected stored parameters, real lifted execution, and actual
native execution. Include cancellation, very small and large finite values,
zeros, rounding boundaries, ties, bias, and nonrepresentable proposed edits.
The [realization review](source-review.md#exact-candidate-realization-matters)
and [PyTorch boundary](../design.md) motivate R24–R28.

Treat bfloat16 conversion through installed PyTorch as a named trusted
operation until independently characterized. Preserve alias/state round-trip
checks and reject unsupported layouts. Record backend and library versions,
plus repeated-run variability where relevant; a dtype label alone does not
specify execution.

For a deployed guarantee, derive numerical error terms for both the base and
candidate executions and combine them with the real drift bound under the
declared domain assumptions. If only measurements are available, report them
as measurements on that input set. In precision comparisons, separately
record the projected baseline: changing it can alter outcomes before the edit
is even applied.

**Completion and falsification.** A verified extension must cover every case
within its stated assumptions and fail closed on unsupported or potentially
nonfinite execution. One native output outside the combined claimed bound
falsifies that guarantee. Zero sampled violations do not prove it. Attributing
all precision-arm differences to edit rounding while the parent also changed
invalidates that interpretation, even if conversion itself was correct.

## 4. Build an auditable restricted and live-provider harness

**Failure to prevent.** A trial is labeled no-forward or no-fitting because
none of the session callbacks recorded the prohibited work. The editor or
another component performed it outside those callbacks. Cooperative accounting
cannot establish the claimed restriction.

Implement a live-provider adapter and, for claims that need enforcement, an
isolated execution boundary with an explicit allowed-computation policy.
The [workflow](workflow.md) and R09–R20 distinguish these obligations. Retain
exact visible prompts and observations, provider/model identifiers and
settings, requests, responses, errors, retries, token telemetry, candidate
attempts, parser revisions, manual interventions, and selection.

Keep the evaluator in a separate ownership boundary. Bind scorer code,
configuration, and data manifests, and specify which results can return to
the proposer. If tamper detection beyond internal self-consistency is claimed,
retain an external commitment to the completed evidence manifest. Record
target calls, examples, tokens, activations, bytes, wall time, memory, and work
inside adapters.

Define prohibited operations by semantics. Arbitrary Python cannot be declared
fitting-free merely because no gradient API appears in a log. A narrower
audited construction language, restricted capabilities, or a separately
reviewed execution trace can support a specifically bounded compliance claim.
Rounded-weight editing also needs a submission route that prevents candidate
return values from exposing exact parent state.

**Completion and falsification.** Predeclared adversarial probes must fail to
reach forbidden data/tools or leak evaluator content. Independent accounting
must reconcile with provider and target records, including abandoned attempts
and retries. A successful bypass invalidates the corresponding enforced-access
claim; undisclosed usage invalidates a cost-matched comparison. Passing these
checks establishes only the documented harness boundary. Actual provider
trials are still needed to measure a language model's task ability.

## 5. Run independent task and access experiments

**Failure to test.** A model answers correctly after receiving weights and
rich task hints, but the same hints would have supported the answer without
weights. The observed answer cannot identify what weight access contributed.

The [source context analysis](source-review.md#supplied-semantics-and-hints-are-part-of-the-task)
motivates a primary contrast: identical metadata alone versus weights plus
that metadata. Build separate reading, diagnosis, repair, and creation task
families with independent ground truth and appropriate rubrics. Separately
vary precision, labels, feedback, activations, and hints. Include unfamiliar
but well-defined variants, no-op/abstention, fixed analytic algorithms, and
fitting-permitted baselines where allowed. A no-fitting condition needs its
own admissible baselines and the audit in package 4.

Repeat editor trials across independently trained lineages and untouched task
families. Retain all candidates and failures. Randomize access assignment or
use paired trials with controlled context isolation. Prevent candidate return
values and feedback from contaminating the contrast. Prespecify budgets for
actual information and work as well as calls.

Permutation or decoy controls must remain valid tasks with correctly
recomputed ground truth. Changing weights while keeping an obsolete behavior
label is not a valid information-ablation control. Report descriptive
properties and confidence calibration separately from category labels.

**Support and falsification.** The added-information hypothesis needs the
prespecified improvement over the context-matched baseline, with uncertainty
at the lineage/task/editor unit and appropriate retention for edit tasks. If
metadata explains performance or gains disappear on new lineages, the stronger
weight-understanding claim fails on the tested population. Analytic
construction from a supplied formula validates that route without establishing
live inference. A null result applies to its tasks, models, access, and
budgets; it does not resolve every possible weight-space method.

## 6. Establish feature coverage and feasibility limits

**Failure to test.** An edit has tiny residuals on many anchors but large drift
at an intended protected input. Its features may lie outside the span, or its
span coefficients may amplify the anchor errors. Alternatively, exact
protection may forbid the desired target change altogether.

The [span and residual results](formal.md#approximate-protection-and-conditioning)
show how to distinguish these mechanisms. Begin with extractors whose
mathematical features and domain are known. Supply explicit anchor
representations or bounded off-span residuals, including bias augmentation.
Vary anchor rank, conditioning, target distance from the protected span,
requested shift, and native realization under a frozen design.

Include provably infeasible targets inside the protected span, useful feasible
constructions, and numerical near-degeneracy. Compare unconstrained, soft,
nullspace, and no-op behavior. Measure realized target effect, residuals,
coefficient norms, off-span terms, and edit magnitude. A no-op on a conflicting
request can be correct while providing no useful target gain.

Then use independently trained extractors and untouched lineages. Evaluate the
intended retention domain, including the current fixture's unmeasured
intervening region when making a domain-wide claim. Separate representation,
solver, and execution errors. Record whether `feature_id` is a label, a hash
of extraction code/state, or a checked semantic correspondence.

**Support and falsification.** Exact protection requires zero actual endpoint
residuals and valid representation premises. Approximate protection requires
the full coefficient-weighted and off-span bound. Small anchor residuals with
uncontrolled coefficients do not pass. Claimed target gain under a provably
conflicting exact contract calls for locating a violated premise or
implementation defect. A generalization claim additionally needs held-out
gain and retention on independent lineages under prespecified margins, for
that representation and domain.

## 7. Test causal diagnosis and repair

**Failure to test.** An explanation identifies a component and a patch changes
a diagnostic score, but unrelated patches have the same effect or the useful
behavior does not improve. Identification, causal prediction, and repair have
not all been established.

The [source repair analysis](source-review.md#identification-mechanistic-diagnosis-and-causal-repair-are-separate-claims)
motivates an intervention design. Before interventions or hidden evaluation,
freeze the diagnosis, target components, predicted sign, expected behavioral
effect, and observations that would contradict it. Compare the proposal with
no-op, support/magnitude-matched random edits, largest-change or simple
heuristic baselines where appropriate, opposite-direction edits, and
restoration to the actual original endpoint.

Use multiple random-control draws when estimating their distribution. Match
realized support and norm after native projection or report the remaining
mismatch. Keep checkpoint-pair classification, continuous diagnostic scores,
mechanism localization, target repair, and retention as distinct outcomes.
For language generation, separately measure correctness, termination,
length/cutoff effects, and format compliance. Repeat on untouched problems
and lineages; cluster answers by problem and retain editor variability.

**Support and falsification.** The frozen intervention prediction must
outperform its specified controls with the expected directional/restoration
behavior and a meaningful effect under the declared rubric. Similar effects
from unrelated edits, an opposite directional effect, or failed restoration
undermine the proposed explanation. Repair also needs the joint utility and
retention criterion. A continuous score shift without useful behavioral repair
remains a measured effect.

## 8. Admit and evaluate cumulative edit sequences

**Failure to test.** Every step passes a local retention threshold, yet the
final model drifts too far from the original or loses the gains of earlier
edits. A threshold against the current checkpoint does not by itself enforce
a cumulative contract.

The [composition results](formal.md#composition-and-decisions) add conditional
drift budgets; R55–R57 require lineage-level behavior as well. Bind every
intermediate parent/candidate and declare whether retention refers to the
original checkpoint, current checkpoint, or accumulated task set. Maintain a
global budget on a fixed domain and coordinate interpretation. Recheck it
after each step rather than reusing a per-step threshold.

Freeze task order, selection, stopping, and abstention policies. Compare the
current-checkpoint retention strategy, an original-baseline strategy, and a
cumulative-budget controller on the same assigned sequences. Use independent
training lineages and task orders, revisits to earlier tasks, a range of
sequence lengths, and untouched per-step and final evaluation units.

Report utility at each stage, erosion of earlier gains, cumulative collateral
drift, budget use, abstentions, and rejected edits. Check artifact rollback by
identity. External deployment state and random trajectories require separate
handling; restoring parameters alone does not restore them.

**Support and falsification.** Every admitted intermediate candidate must
satisfy the cumulative contract. One admitted violation falsifies that
controller guarantee. Useful progress must also pass the global retention
criterion across independent sequences. Excess total drift or systematic loss
of earlier gains despite local acceptance refutes the proposed accumulation
rule. An always-no-op controller can be sound while failing a nonzero-progress
objective. A bounded experiment does not establish unlimited editing or the
absence of all future forgetting.

## 9. Extend operators and tighten bounds

**Failure to address.** A useful candidate cannot be certified because its
operator lacks a proof, or a supported deep network receives a bound too loose
for the intended budget. These are different failure mechanisms.

Choose extensions from concrete tasks. Specify operator/activation semantics,
state and shape constraints, input domain, and adapter correspondence. Add
checked lemmas and independent executable references. Attention,
normalization, residual paths, and convolutional graphs have distinct
obligations. A frozen final-head abstraction does not certify edits inside
those operators.

For supported deeper chains, compare the present absolute-value recurrence
with signed intervals, activation information, or other sound relaxations on
a frozen task set. The current recurrence loses correlations and cancellation;
tighter estimates require a sound argument for retaining that information.

**Completion and falsification.** Report certified coverage at fixed budgets,
tightness against measured or exactly solved small cases, runtime, and failure
or unsupported rates. Any violated claimed bound is a soundness failure. A
tighter empirical estimate without a proof does not improve a certificate.
Universal refusal or bounds too loose for the declared use are practical
failures even when sound. An extension establishes its supported family; it
does not supply a general architecture-transfer theorem.

## 10. Measure and improve scale and total cost

**Failure to test.** A small edit program appears cheap, but creating it or
checking it requires dense snapshots, copies, full scans, feature extraction,
large context, or many target evaluations. Cost has moved rather than fallen.

The report's context/harness concerns and the [design's scale limits](../design.md)
motivate a complete profile. Vary artifact size, feature width, anchor count,
factor rank, and sequence length. Include model preparation, representation
extraction, serialization, reads, copies, compilation, native projection,
certificate computation, provider context/latency, target calls, and final
evaluation. Keep peak memory, persistent bytes, wall time, and work units
distinct. Retain failures and out-of-resource cases.

Where profiling supports it, prototype chunked/content-addressed storage,
explicit sparse deltas, or compact execution. Preserve deterministic lowering,
native dtype, alias rules, parent checks, and exact endpoint verification.
Run differential reconstruction against the existing dense reference before
measuring throughput.

An optimization-savings comparison needs equally informed baselines with
matched quality, retention, tuning, and training-versus-target-time
accounting. One fixed gradient configuration cannot establish broad optimizer
superiority.

**Support and falsification.** Require identical declared endpoints and a
prespecified resource reduction at the target scale, with no hidden relocation
of work into preprocessing or callbacks. Reconstruction mismatches invalidate
the storage change. Savings that vanish after full accounting, or require
sacrificing target/retention quality, fail the efficiency hypothesis. Sparse
changed entries or a small factor program alone do not establish lower runtime
or context cost.

## 11. Reconstruct and independently repeat selected source tasks

**Failure to prevent.** Reconstructing a final patch is reported as reproducing
the editing process, or reevaluating one retained endpoint is counted as a
fresh editor trial. These procedures answer different questions.

The [reproduction review](source-review.md#reproducibility-is-a-set-of-distinct-obligations)
separates inspection, reconstruction, reevaluation, replay, and independent
replication. Choose a task and name the tier being attempted. Inventory model
bodies, base revisions, native dtypes, prompts, parser versions, intermediate
observations, harness code, evaluators, data, environment, provider access, and
stochastic settings. Record unavailable inputs explicitly.

For candidate reconstruction, verify parent and resulting tensor identities
against retained manifests. For reevaluation, preserve metric and decoding
rules and separately identify intentional changes. For replay, disclose
missing visible observations and hidden infrastructure work that prevent
equivalence.

An independent repeat needs fresh editor trials and appropriate new
checkpoints or tasks under a frozen protocol, with no-op eligibility and the
complete attempt population. A protocol-replication claim must preserve its
access and fitting restrictions. A constrained least-squares control can be
informative alongside a no-fitting task, but cannot stand in for that
condition. Historical seeds and retained final patches are reproduction
inputs, not new independent confirmation.

**Completion and falsification.** Reconstruction passes only when the exact
declared endpoint checks pass. Reevaluation agreement needs a prespecified
tolerance appropriate to numerical/stochastic variability; retain disagreement
and audit its inputs and protocol. Independent replication needs its own
effect and uncertainty criterion. An unavailable checkpoint or grader blocks
a reproduction tier; it is not a negative scientific outcome. A reconstructed
patch alone does not establish a reproduced editing process.

## Source future directions and their unresolved tests

The original report proposes the following directions in **What Comes Next?**;
the [source review](source-review.md#actual-limitations-and-next-steps-in-the-original-report)
records their status. Each needs its own claim and test after the relevant
measurement and access boundaries are in place.

| Direction | Minimum credible next experiment | Outcome that limits or falsifies the claim |
| --- | --- | --- |
| Auditing without evaluation | Blindly identify a predefined property or planted mechanism on untouched lineages from allowed weights/context; use metadata-only and simple statistical baselines; keep ground truth with an independent evaluator | Report sensitivity, specificity, calibration, abstention, and domain shift. Failure to beat context-matched controls refutes the tested added-information claim. A few labels do not establish broad auditing or backdoor detection |
| Zero-data steering | Define which labels, examples, rules, hints, and training information are absent; compare analytic and live proposals under the audited regime | Require useful behavior and retention on independent tasks. Supplied target formulas or rules remain privileged inputs; no labeled examples does not imply no privileged information |
| Unlearning | Predefine the requested removal, retained capabilities, threat/evaluation model, and held-out extraction or membership probes; compare appropriate retraining/editing baselines under matched budgets | Residual recoverability under the declared probes or excessive collateral loss fails the objective. One suppressed prompt response is insufficient, and finite tests cannot establish universal erasure |
| Reducing optimization cost | Compare preparation, training, editing, inference, tuning, and evaluation at matched quality/retention across independent tasks | Require a prespecified cost/quality improvement with uncertainty. Work moved into prompts, features, hidden fitting, or evaluation still belongs in total cost |
| Recursive self-improvement | Record proposer/model versions and exact lineage; freeze independent evaluations; maintain cumulative contracts; compare repeated nonrecursive editing under matched resources | Require sustained gain beyond selection noise with retention. Reused-evaluator fitting or lost earlier capabilities defeats the tested claim. A short sequence does not establish open-ended improvement |
| Circuit grafting across architectures | Specify source/target computation, a validated correspondence, admissible transformations, native realization, and causal controls on untouched architecture pairs | Require the predicted effect and retention beyond controls. Shape compatibility or feature alignment alone is insufficient; no general transfer theorem is supplied here |
| Benchmark and harness development | Release mode-specific rubrics, access regimes, complete populations, task/lineage splits, scorer identities, retained failures, and an independent reproduction path | Preserve abstention, negative outcomes, and restrictions rather than collapse them into one success rate. An API by itself does not demonstrate the ability being measured |

## Reviewable completion criteria

Each package completes a declared scope. Engineering work should leave
versioned code, focused tests or checked proof obligations, exact artifact and
receipt identities, and an explicit unsupported boundary. Experimental work
should leave a frozen protocol, raw per-unit records, a full failure and
selection ledger, reproducible analysis, and conclusions matched to the
independent units. Both should retain negative results and make the stated
falsification checks reproducible.

When a claim remains unsupported, record the remaining obligation precisely:
a missing artifact, unproved correspondence, unenforced restriction, uncovered
domain, unmeasured cost, or inadequate independent units. That record identifies
the next implementation change or experiment. The [requirements ledger](requirements.md)
tracks status; this research plan does not convert proposed work into
implemented capability.
