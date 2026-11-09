# Research directions after the November 2026 follow-ups

**Reviewed November 8, 2026, America/Chicago. This is a research plan, not a report of new framework results.**

The next useful step is to finish the evidence for a trained use case and ask
whether editing weights improves the outcome enough to justify its cost and
retention risk. For the support-routing application, that means comparing a
trained model edit with an ordinary queue-policy overlay, a task-compatible
context method, analytic head updates, and tuned head retraining. A cheaper
overlay that meets the same requirements is a successful research finding.

Two recent manuscripts sharpen that plan. **WeightLoop** supplies a measured,
offline example of a fixed editor consuming new feedback and saving changes
into a predictor. The image-restoration manuscript supplies testable proposals
for visual diagnosis and repair, but reports no restoration experiments. They
support different experiments and different access rules; neither establishes
general weight understanding, broad preservation, or autonomous improvement of
the editing agent. [2](#reference-2), [3](#reference-3)

This document extends the
[limitations and next steps](../weight-space/v2/limitations-and-next-steps.md)
and [existing roadmap](../weight-space/v2/research-roadmap-2026-11.md).
The [companion review](primary-paper-review.md) covers the original author
report and related primary research in more detail. The proposals here do not
change a frozen protocol, reopen a held-out partition, authorize a provider
trial, or select a new winner using previously unseen outcomes. They describe
future work that must receive its own protocol and evidence record.

## What was reviewed

The original report is an author-published experimental report dated
November 2, 2026. The two follow-ups below are arXiv manuscripts. The reviewed
versions were available by the review date; their submission dates are not
dates on which this framework completed an experiment. [1](#reference-1),
[2](#reference-2), [3](#reference-3)

| Source | Exact reviewed version | Evidence in that source | Main implication for this project |
| --- | --- | --- | --- |
| Original weight-space programming report | Author report dated November 2, 2026 | Exploratory reading, diagnosis, editing, and construction trials under several access conditions | Preserve separate claims about task success, allowed computation, and what weights add to supplied context |
| WeightLoop | arXiv:2611.07319v1; submitted November 5, 2026, 02:06:51 UTC | Offline replay on scientific prediction and acquisition tasks; a fixed generated editor with measured feedback | Test reusable feedback-driven editors against simple fitting and policy controls |
| Image-restoration conjectures | arXiv:2611.04268v1; submitted November 3, 2026, 03:56:44 UTC | Proposed defect taxonomy, conjectures, and illustrative sandbox code; no measured restoration results | Build a blinded diagnosis-and-repair experiment, with explicit stochastic execution and budget rules |

The WeightLoop PDF was read in full, including both figures, all three tables,
methods, and references. The restoration PDF was read in full, including its
conjectures, two code listings, limitations, and references. Numerical and
pseudocode passages were also checked against rendered pages. Official
abstract pages, versioned PDFs, and arXiv HTML were retained privately with
retrieval receipts and byte identities. This is a review of these sources,
not a claim of an exhaustive search of all work published through November 8.

## Keep the claim boundaries explicit

An editor can be useful without satisfying every restriction explored in the
original report. The important requirement is to name what it actually does.
The following distinctions should be fields in an experiment record, rather
than assumptions hidden inside the term "weight-space programming."

| Question | Conditions to distinguish | Why it changes the interpretation |
| --- | --- | --- |
| What information reaches the proposer? | Architecture and task context; summaries; exact weights; examples; labels; outputs; internal activations | A task hint, activation cache, or labeled feedback can explain success without establishing a weight-only ability |
| What computation is permitted? | Arithmetic weight analysis; target execution; finite search; regression; least squares; gradient fitting; tools | Absence of gradients does not imply absence of fitting, supervision, model execution, or external tools |
| What changes over time? | Feedback set; predictor; fixed editor; newly called editor; editor's own parameters | Updating a target model is a different experiment from improving the agent that updates it |
| What is the reference model? | Original checkpoint at every round; previous accepted checkpoint | Growing feedback and cumulative parameter edits require different lineage and retention contracts |
| What task is being scored? | Property identification; causal diagnosis; repair; construction; transfer | Correct classification of a model property does not by itself validate an explanation or an intervention |
| What is preserved? | Tensor/interface structure; an auxiliary branch; supplied anchors; sampled decisions; a declared input domain | An unchanged encoder or reloadable checkpoint does not establish broad behavioral retention |

WeightLoop deliberately uses labels, target predictions, validation, and
analytic fitting in some arms. The restoration proposal uses paired feedback
images, activation summaries, ablations, and repeated denoising evaluations.
Those are legitimate feedback-permitted regimes. They would be inadmissible
in a no-forward or no-fitting arm without a separately declared exception.
[2](#reference-2), [3](#reference-3)

The framework should therefore publish the access regime beside the result.
A regression program generated by an LLM tests whether a generated program
implements a useful supervised update. It does not establish that the LLM
inferred an unknown computation from anonymous weights without examples.
Similarly, an analytic constructor supplied with the desired function belongs
in a different arm from an editor asked to discover that function.

## What WeightLoop contributes, and what still needs testing

### A reusable feedback editor is a concrete hypothesis

WeightLoop's LLM inspects the task specification and model implementation once,
then produces an executable editor. Numerical execution uses that fixed
program without further LLM calls. Its update rule is
`theta_t = theta_0 + E(theta_0, S_t)`, where `S_t` contains the measurements
available by that point. Every patch is reconstructed from the original
checkpoint. **Feedback accumulates; patches do not compound.** A round that
does not pass its validation rule returns the original model.
[2](#reference-2), section 2.1

That design is relevant to an enterprise workflow in which approved feedback
arrives repeatedly. It gives a specific engineering question: can a fixed,
bounded editor turn new evidence into useful updates, without repeatedly
paying for editor construction? The scientific use case in the paper is an
offline replay of experiment selection. It does not provide evidence of a
live laboratory deployment, business savings, or an editor learning to make
better edits across independent tasks.
[2](#reference-2), abstract and sections 3–4

The neural arm uses an existing representation with a 128-dimensional feature
vector and permits changes only to its functional readout coefficients.
Diagnosis attenuates individual coefficients and checks a label-dependent
ranking loss on cached activations. Three identity-based support folds select
among six combinations of edit size and attenuation, subject to a ranking
utility guard. The encoder, bias, scaler, auxiliary RNA head, and source
adapters remain fixed. The ridge arm solves regularized least squares; the
tree arm adjusts supported leaf values using measured residuals. These are
different update operators, and the latter two are supervised fitting.
[2](#reference-2), sections 2.2–2.3

The promising lesson is to expose a small, interpretable update interface and
retain its selection and persistence record. The lesson is not that a small
fraction of all stored parameters certifies a small functional change. A
readout coefficient can materially alter many predictions while every
encoder parameter remains unchanged.

### The measured results are encouraging and mixed

In the single-round neural ranking study, the edited mean exceeds the
permuted-edit mean in all eight endpoint–budget settings, and the original
mean in seven. The exception is HL60 migration at budget 50: the original
normalized utility is `0.0103` and the edited utility is `0.0087`. The
100-label full-fit reference is stronger than the edited model in six of the
eight settings. Both the initial and full-reference fits use the same stated
update count; editing consumes additional measured feedback.
[2](#reference-2), section 3.1 and table 1

The sequential replay reports the following final discovery utilities.

| Endpoint | Original | Edited | Edited minus original | Positive paired gene folds |
| --- | ---: | ---: | ---: | ---: |
| LC3 flux | 0.1757 | 0.2091 | 0.0334 | 4 of 5 |
| NBR1 flux | 0.1671 | 0.1878 | 0.0207 | 4 of 5 |
| HuH7 LDL decrease | 0.0038 | 0.0185 | 0.0147 | 3 of 5 |

These are the paper's table 2 values. Zero denotes the random expectation for
the same candidate pool and one denotes the oracle; negative utilities are
possible. A delta of `0.0334` is a change on that normalized utility scale,
not a percentage improvement in discovery, customer experience, or cost.
Gene partitions are repeated views of the same studies, rather than
independent biological task families. No confidence intervals are reported
for these comparisons. [2](#reference-2), section 3.2, figure 2, and table 2

In the coefficient/leaf experiments, an ordinary feedback-fitted affine
calibration is stronger than the edited XGBoost model for Jurkat on the
primary MSE metric: `0.3183` versus `0.3258`, compared with `0.3346` for the
original model. That is a useful warning against treating random edits as the
only serious comparison. [2](#reference-2), section 3.3 and table 3

The paper also verifies a native save/reload example: a readout edit within
a larger predictor reloads with identical scores on the checked queries.
That supports persistence for the specified checkpoint and inference route.
It does not establish equivalence across backends or retention over an
unbounded input domain. The reviewed manuscript does not provide broad harm
intervals, an end-to-end cost comparison, or independent LLM-construction
variability. An editor-code repository and provider transcript were not
located in the reviewed PDF/HTML references; that is a limitation of the
reviewed evidence, not a claim that no such asset exists anywhere.
[2](#reference-2), sections 3–4 and references

### Separate update quality from acquisition policy

For a fixed support set, permuting a diagnostic edit while matching edit size,
norm, and validation opportunity is a useful localization control. In the
sequential study, policies can acquire different later support. Its end-to-end
contrast therefore combines model updating with the consequences of changed
experiment selection. The last update changes permanent-set predictions but
cannot improve discoveries already acquired. [2](#reference-2), sections
2.2 and 3.2

A framework follow-up should declare two estimands:

1. **Update quality with shared feedback.** Give every method the same
   time-indexed measurements, fit/select under matched rules, and compare
   predictions on untouched evaluation inputs. This isolates the update
   operator from the acquisition policy.
2. **End-to-end decision quality.** Let each frozen policy choose its own next
   inputs before their outcomes are revealed. Compare complete trajectories
   under the same acquisition and resource limits. This tests the whole
   feedback loop; it cannot isolate the editor alone.

The second design must not silently use future labels to choose an earlier
action. The first must not be presented as evidence that the method chooses
better experiments. Both should retain no-ops, rejected candidates, incomplete
trajectories, and the original checkpoint as an explicit reference.

## What the restoration proposal contributes, and what must be repaired

### Controlled visual defects can test causal diagnosis

The restoration manuscript proposes localized output-bias, upsampling, and
skip-path defects, alongside naturally occurring diffuse quality failures.
It separates layer diagnosis, a sparse repair, and repeated repair rounds.
Its abstract and arXiv comments explicitly describe the work as conjectural.
No measured restoration gain, pinned trained restoration model, experimental
provider run, or validated sandbox implementation is reported.
[3](#reference-3), abstract and sections 4, 6, and 9.3

This suggests a useful second trained domain. An evaluator can introduce a
known defect, hide its location from the proposer, and test whether a frozen
diagnosis predicts which intervention will help. Repair and retention can
then be scored independently. An exact inverse of the injected defect is an
oracle ceiling, while a task-compatible output correction tests whether
modifying weights is necessary at all.

The experiment must also contain natural errors or defects outside the
proposal's repair family. Otherwise, finding the inverse of an evaluator's
known construction could be mistaken for discovering a naturally occurring
mechanism. The manuscript's proposed classification of diffuse failures as
not sparsely repairable is a hypothesis to test. Failure under one edit
family and budget does not establish impossibility for all sparse edits.

### The numerical and implementation rules are not ready to adopt

| Reviewed passage | Unresolved issue | Requirement before an experiment |
| --- | --- | --- |
| C1 uses 200 forward passes in its goal, 30 in its expected result, and 50 in its falsification rule; the deployment outline uses 30–50 and the code defaults to 40 | The resource limit changes between sections and does not define a common work unit | Freeze one budget ledger and distinguish a candidate evaluation, an image/seed denoising chain, and an individual UNet call |
| C2 proposes at least 0.2 dB PSNR gain, while sandbox acceptance uses 0.1 dB | Search acceptance and final success have different thresholds, without an analysis of selection noise | Declare separate development selection and confirmatory effect rules before outcomes are seen |
| The prose caps cumulative changed parameters; Listing 1 stores `max_ratio` without checking it | The illustrative implementation does not enforce its claimed edit limit | Check the chosen cumulative support/norm rule before evaluating or accepting a candidate |
| Acceptance is described as improving PSNR without new artifacts; Listing 1 checks only PSNR | The displayed code has no implemented artifact or retention criterion | Declare measurable protected quality conditions, include their checks, and report subjective review separately |
| Baseline initialization calls the evaluator before the displayed counter advances | The counter omits work needed to establish the baseline | Charge initialization, rejected candidates, selection, repeated measurements, and all actual denoising work |
| The outer loop repeatedly calls the LLM and revisits a small feedback set | Provider cost and adaptive selection pressure are absent from the acceptance logic | Retain provider calls, full attempts, data access, and a separate untouched confirmation set |

These observations concern the reviewed proposal, not a demonstrated bug in
this framework. The manuscript's `0.2 dB`, `0.1%`, and multi-round gain ranges
are proposed conditions or forecasts. They are not measured findings and are
not adopted framework margins. [3](#reference-3), sections 4 and 6, listings
1–2

Although the proposal discusses avoiding paired training data, its pipeline
uses 32–64 paired feedback images and PSNR/LPIPS supervision. Activation norms
and ablations also require target execution. The correct experimental label
is paired-feedback access with permitted execution, rather than no paired
supervision or no-forward evidence. [3](#reference-3), sections 4.1 and 6.1

### Diffusion behavior needs an execution contract

The manuscript recognizes stochastic sampling, perceptual trade-offs, and
dependence on the number of denoising steps. Its illustrative acceptance loop
does not resolve them. A change that helps a reused feedback set can harm
other images or another sampling configuration, and monotonically improving
feedback PSNR does not establish untouched retention.
[3](#reference-3), sections 7–8

A future adapter must pin whether the model predicts noise, a clean image,
a residual, or another parameterization, along with the sampler, timestep
schedule, seed handling, preprocessing, and output transform. A constant
UNet output-bias delta need not become a constant final RGB correction.
That proposed intervention must be tested under the actual computation.
Likewise, a channel permutation is a valid equivalence control only when all
affected paths are transported consistently through the graph.

Use paired seeds for method comparisons, retaining image-level and
checkpoint/editor-level clustering. Report complete denoising chains and
underlying UNet evaluations. Validation at one frozen schedule supports only
that schedule; a step-count or sampler extension needs its own prespecified
stress evaluation. Measure target quality and protected quality separately,
including a perception-oriented metric and a defined artifact rubric where
the use case calls for them. More seeds improve measurement precision but do
not create new independently constructed editors or model lineages.

The manuscript's timing and parameter comparisons with LoRA/full fine-tuning
are unmeasured assertions. Its stronger statements about what requires AGI
or what sparse edits cannot do are extrapolations. The original report's
failure on one Gemma construction/improvement task supports failure under
the tested model and budget, not a universal lower bound on capability or
compute. [1](#reference-1), [3](#reference-3), sections 5 and 9.2

## Proposed experiments and existing issue ownership

The work packages below sharpen existing issues. They do not require a new
issue for every paper observation. Issue numbers identify the repository's
published roadmap; state and completion must be checked at execution time.

| Work package | Main question | Existing issues | Evidence that changes the next decision |
| --- | --- | --- | --- |
| Trained routing and ordinary controls | When does a weight edit add value beyond changing a queue policy or fitting the head? | [#7](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/7), [#6](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/6), [#14](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/14) | A complete paired comparison of gain, protected harm, coverage, abstention, and total cost |
| Fixed feedback editor | Can one generated editor reuse new evidence effectively, and is the advantage in updating or decision selection? | [#7](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/7), [#10](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/10), [#14](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/14) | Separate shared-feedback and complete-trajectory results with a declared original/current-parent rule |
| Actual contribution of weight access | Do exact weights improve proposals beyond identical task context and summaries? | [#2](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/2), [#5](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/5), [#8](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/8) | Audited provider trials and paired access ablations, including independent editor construction |
| Visual diagnosis and repair | Does a diagnosis predict a beneficial intervention on a trained non-text model? | [#9](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/9), [#11](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/11) | Blinded injected-defect and natural-error results, with repair and retention scored separately |
| Artifact, runtime, and proof scope | Does the admitted claim concern the exact stored candidate and declared computation? | [#3](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/3), [#4](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/4), [#12](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/12), [#13](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/13) | Candidate-specific correspondence and clearly scoped native measurements or checked premises |
| Transfer and construction boundaries | Does an edit survive a compatible representation change, and what is possible when fitting is forbidden? | [#16](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/16), [#17](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/17) | Held-out lineage tests, verified symmetry controls, and independently audited computation regimes |
| Source repetition and bounded unlearning | Which source claims can be repeated, and what limited removal claim can be defended? | [#15](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/15), [#18](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/18) | Declared reproduction tier; separate licensed synthetic identity study with retraining and bounded attack controls |

The umbrella remains
[#1](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/1).
The following designs explain how these packages connect to the use case and
where a negative result should stop expansion.

### 1. Establish value in the trained support-routing use case

Use the existing trained-use-case design and its controller. The proposed
task has two distinct kinds of change:

1. **Ownership change:** a recognized intent should now go to a different
   queue. An intent-to-queue policy overlay is the necessary control.
2. **Recognition repair:** the trained representation/head incorrectly
   recognizes a class of requests identified using development data. This
   can test whether changing the learned decision rule helps beyond a policy
   remapping.

For each case, compare the unchanged parent, ordinary policy/context controls,
target-only fitting, retained head edits, and tuned head retraining. Keep the
encoder and adaptation scope explicit. If broader model adaptation is added,
give it a separately declared training and cost scope.

The principal evidence is target queue performance, per-intent and protected
queue harm, performance on confusing nearby intents, relevant OOD behavior,
feasibility, and abstention. Exact residuals on protected feature rows help
explain a candidate, but they are not substitutes for queue outcomes on
untouched requests. The derived queue policy is a project specification;
dataset intent labels do not supply real transfers, customer waiting times,
or resolution costs.

Report each paired method contrast against its prespecified useful-gain and
harm margins. Publish the complete attempt denominator, failed fitting and
admission cases, and uncertainty at the appropriate independent unit. Do not
pick whichever ownership change makes the edit look useful after seeing its
confirmation result. If a cheap overlay reaches the same quality/retention
frontier, recommend it for that case and use recognition repair to identify
the remaining research question. This sharpens #7 rather than replacing its
frozen design.

### 2. Test a fixed editor with accumulating feedback

After the trained-task evidence permits a new study, introduce a small,
explicitly feedback-fitting editor as a separate regime. Compare a generated
fixed editor with hand-written analytic updates, regularized residual fitting,
affine calibration where compatible, retained fitting, and head retraining.
Use matched development and selection opportunities and charge editor
construction as part of cost.

For the shared-feedback experiment, freeze a feedback sequence whose labels
arrive only at the declared time. For a decision-policy experiment, freeze
the candidate pool and acquisition rules, reveal outcomes only after an
action, and analyze whole assigned trajectories. Training-history labels,
new feedback, selection data, protected data, and final evaluation must have
explicit roles; they should not silently migrate between them.

Declare one of two update semantics, or compare them in planned arms:

- `theta_t = theta_0 + E(theta_0, S_t)`: every candidate is a replacement
  edit of the original checkpoint, following WeightLoop's design.
- `theta_t = theta_(t-1) + E(theta_(t-1), S_t)`: each edit modifies the
  previous accepted endpoint, creating cumulative parameter drift.

For either regime, check the original-baseline retention contract at each
endpoint and preserve earlier accepted target requirements when the task
requires them. A no-op in the first regime means the original model; a no-op
in the second means the previous model. Those outcomes must not be conflated.
Record the full edit-by-target matrix, rejection reasons, elapsed feedback,
and final predictions. A changing predictor with a fixed editor supports
feedback-driven adaptation of that predictor. Improving editor capability
would require a separate transfer study on new tasks with independently
constructed editors.

### 3. Measure what weight access adds to a live proposer

The generated-editor claim needs actual provider evidence. Freeze the same
task/interface context and compare no parameter view, declared summaries,
and exact permitted weights. Ablate mechanism hints separately. Match the
candidate-validation and target-execution opportunities; a summary arm that
gets activations is not equivalent to an exact-weight arm forbidden to run
the model.

Use the access broker to retain prompts, provider/model configuration,
responses, tool use, permitted tensors, costs, failures, retries, human
involvement, and selection. Independent construction trials are the unit for
claims about editor reliability. Candidate attempts inside one adaptive
session remain part of that session, not new independent editors.

Freeze the generated program before applying it to untouched feedback and
evaluation lineages. A hand-selected successful editor can demonstrate an
existence case; it cannot estimate the provider's ordinary success rate.
Missing or unauditable provider records prevent an access-restriction claim
even if the eventual candidate has useful measured behavior. #5 and #8 own
that distinction; #2 owns the comparison and complete-attempt protocol.

### 4. Add one trained visual task with blinded interventions

The first non-text pilot should use one public, pinned trained model and a
repair route the framework can actually represent. A deterministic trained
restorer or a small vision model can test the diagnosis/repair workflow
before adding a stochastic diffusion chain. The dataset, architecture,
weights, license, preprocessing, and selected defect family must be fixed
before confirmation. A diffusion experiment is a further execution regime,
not an interchangeable export of that first pilot.

The prospective [cohort selection API](staged-diagnosis.md#cohort-preparation-and-selection)
records the full assignment roster before preparation and requires all terminal
diagnosis dispositions before any repair construction. Preparation failures,
including ordinary configuration-preflight failures, remain in the denominator,
as do abstentions and selection failures. A separate [engineering receipt](../../provenance/verification/staged-cohort-selection-20261109.json)
records the focused passing selection regression at its captured source snapshot.
That receipt covers the selection boundary at its captured source snapshot; it
does not cover the later registered bridge or supply new study evidence. Issue
#9 c1 remains partial with the issue open.

The subsequent [complete repair barrier](staged-diagnosis.md#complete-repair-barrier)
durably binds every assigned member's terminal disposition and lane evidence,
including the actual selected candidate and live artifact. Its separate
[literal regression receipt](../../provenance/verification/staged-cohort-repair-barrier-20261109.json)
adds engineering evidence only. At that receipt's source snapshot, the endpoint
seal and single-score bridge remained pending. The repair barrier alone enables no
scoring or evaluation, and #9 c1 remains partial with the issue open.

An opt-in [evaluation-plan declaration](staged-diagnosis.md#optional-evaluation-plan-declaration)
now freezes roster order, holdout/control references, execution declarations and
result keys. Its [focused regression receipt](../../provenance/verification/staged-cohort-evaluation-plan-20261109.json)
covers declaration validation and immutable capture, without establishing runtime
availability. That declaration-only receipt does not cover the later endpoint
seal or registered bridge. Declaration validation alone enables no cohort scoring
or evaluation; #9 c1 is partial and the issue remains open.

The explicit [endpoint seal](staged-diagnosis.md#endpoint-seal) now materializes
referenced origins once after the complete repair barrier, preserving all logical
slots, unavailable members and declared parent fallbacks. Selected endpoints
verify saved literal replay; structured controls reproduce the complete supplied
lowering receipt. Saved capture dtype/device/alias declarations remain distinct
from construction native maps and establish no runtime restoration. Unused controls
remain declarations, and selected no-ops do not become fallbacks. Its separate
[engineering receipt](../../provenance/verification/staged-cohort-endpoint-seal-20261109.json)
records one focused regression: exit 0, `1 passed in 4.30s`, with no retry or
timeout and unchanged before/after source inventory and tracked diff. This covers
its captured source snapshot, not a completed-tree gate or scientific result.
The seal's evaluation, runtime-restoration verification and causal-usefulness
flags remain false. The separate registered bridge below evaluates sealed slots;
#9 c1 remains partial and the issue remains open.

The [registered cohort bridge](staged-diagnosis.md#registered-cohort-evaluation)
now supplies one aggregate registry score authorization over every declared holdout
in plan order, including unused holdouts, after the lazy loader runs once and
sealed endpoints verify. It evaluates each available logical slot reached against
its retained endpoint, with copied member context and holdout rows, and requires
exactly the declared finite-JSON result keys. Failure stops later slots and
preserves unavailable, `not_run` and unresolved registry outcomes without retry or
releasing authorized scope. Its separate [engineering receipt](../../provenance/verification/staged-cohort-evaluation-bridge-20261109.json)
records one focused regression: exit 0, `1 passed in 18.47s`, with no retry or
timeout and unchanged before/after source inventory and tracked diff. This covers
its captured source snapshot, not a completed-tree gate or scientific result.
This generic bridge still needs the trusted study evaluator and its runtime
integration; it supplies no real held-out study. Runtime-restoration and
causal-usefulness flags remain false; #9 c1 remains partial and the issue remains
open, with earlier receipt scopes and historical results unchanged.

The evaluator should sample defect location and strength using a private,
prespecified intervention generator. The proposer should receive the declared
context/access arm without the injection record. Freeze the diagnosis before
revealing repair feedback. Compare the named intervention with matched wrong
locations, norm/support-matched random edits, simple output processing,
ordinary parameter fitting, and the known inverse as an oracle reference.

Score localization, diagnostic abstention, target restoration gain, and
protected quality separately. Use a natural-error set or a separately trained
failure family to test whether the injection-trained repair strategy
generalizes. Do not define success as "fails on diffuse errors": successful
repair of a supposedly diffuse case would be evidence against that boundary,
and a failed attempt may reflect access, budget, or the edit family.

For diffusion, the work ledger should aggregate each actual UNet evaluation
over candidate, feedback image, seed, and denoising step. Include baseline
initialization and repeated measurements. Keep confirmation costs visible
but unavailable for adaptive selection. Specify a seed policy and quality
rubric before judging gains near measurement noise. #9 owns the causal
diagnosis protocol; #11 owns the distinct trained-domain claim.

### 5. Bind cumulative and stochastic candidates to scoped evidence

Every accepted candidate needs one immutable record binding its parent and
endpoint, tensor order, semantic adapter, computation configuration, domain,
target, protected requirements, and resource policy. A sequence additionally
needs an original reference, prior target obligations, and the scope of
rollback. Restoring model bytes does not undo actions already taken by a
decision policy.

Keep three results separate: mathematical admission under stated premises,
native execution agreement for the measured backend, and behavioral utility
on the task. Reloaded scores do not connect arbitrary runtime artifacts to
Lean theorem instances. A theorem about a finite deterministic network does
not automatically cover a sampled denoising procedure, an unsupported
operator, an output transform, or a different storage backend.

Start the proof bridge with a deliberately supported candidate family. Bind
generated exact values and premises to the actual endpoint, independently
check identity, and exercise wrong-parent, wrong-domain, reordered-coordinate,
and altered-adapter substitutions. Expand semantics only when required by a
declared task. If a certificate is too conservative, report its coverage and
measured utility separately. Tightening a bound without increasing useful
coverage on the supported domain is a reason to reassess that work, rather
than repeatedly run the same experiment. #3, #4, #12, and #13 keep these
obligations distinct.

### 6. Extend transfer, construction, and unlearning only after their premises exist

For compatible-family transfer, begin with verified function-preserving
representation changes. Transport all connected tensor and state paths,
verify the parent behavior, and compare direct coordinate copying with
aligned transport. Then use independently trained held-out lineages to test
whether alignment provides a useful edit transfer. A permutation check is an
equivariance control; it is not cross-architecture circuit grafting. #16
should retain that progression.

For restricted construction, freeze whether analytic formulas, examples,
target execution, regression, and gradients are allowed. A no-fitting arm
must actually prohibit residual regression and least squares, including those
operations inside generated code. The feedback-fitting editors above belong
in other arms. #17 should publish both successful constructions and the
scope of failures without turning one budget failure into an impossibility
claim.

For unlearning, use a separate licensed synthetic-identity task, a declared
forget/retain split, reference retraining where feasible, and a bounded
attack suite. Changed outputs or reduced recall under one prompt do not
establish erasure from the weights. #18 should specify exactly which removal
and retention claims its evidence can support. It should not borrow the
routing holdout, infer private-data removal from an unrelated repair, or
expand an attack suite after observing which attacks succeed.

Broader recursive improvement remains a later hypothesis. A target predictor
improving during one task does not show the editor becomes more capable, and
repeated selection on one feedback set does not show reliable gains on new
tasks. A stronger experiment needs independently constructed editors,
held-out task/model lineages, a fixed selection rule, and a contrast against
ordinary optimization at matched quality, retention, and total cost.

## Advancement and stop criteria

These are proposed decision rules for new protocols. They must not be added
retroactively to a completed or frozen study. Task-specific useful-gain,
harm, coverage, and cost margins should come from the intended decision and
a separate development/pilot process, rather than from paper headline
thresholds or favorable confirmation outcomes.

| Decision | Proposed evidence rule | Consequence |
| --- | --- | --- |
| Advance a trained use case | The edit clears the prespecified target/retention criteria relative to serious ordinary controls, with all assigned attempts and uncertainty reported | Extend that scoped application or access regime |
| Prefer a simpler control | A policy/output correction or ordinary fitting reaches the same requirements at lower total cost | Recommend the simpler intervention for that case; retain the negative edit-value finding |
| Abstain or reject | Target and protected requirements conflict, supported-feature coverage is inadequate, or an actual harm limit is violated | Retain the reason and attempt in the denominator; avoid an unsafe or meaningless success claim |
| Stop the adaptive loop | The declared budget is exhausted, no qualifying candidate is found, or cumulative retention/earlier targets fail | End the sequence according to its frozen policy and report the final admissible endpoint |
| Withhold an access claim | Provider/tool records are missing, ambiguous, or show prohibited computation | Report any independently valid candidate behavior with its narrower scope |
| Stop a semantic extension | The adapter cannot faithfully describe the declared execution, or candidate/proof identity cannot be bound | Mark unsupported or missing correspondence; design a separate supported regime |
| Reassess proof work | Repeated bound tightening provides no useful certificate coverage for the chosen domain | Revisit the contract, domain, or method before investing in broader operators |
| Hold a broad capability claim | Evidence is limited to one editor, one task lineage, injected defects, or reused feedback | Keep the conclusion descriptive and scoped; design new independent confirmation units |

Cost should include data preparation and labeling, feature extraction,
training history needed for the comparison, editor construction, provider
tokens/calls, analytic or gradient fitting, target executions, rejected
attempts, admission/proof work, and persistence/deployment work within the
declared boundary. Show recurring cost separately from construction cost and
state any amortization assumption. Report quality/retention versus cost,
including the uncertainty in the quality comparison. Small edit support,
few candidates, or a short test run are resource descriptions; none is an
impact metric by itself.

The immediate deliverable is a defensible answer to the trained use case:
which changes are best handled by policy, which benefit from editing learned
weights, which requests conflict with retention, and what evidence supports
each decision. A positive result can motivate the next domain. A negative
result can make the framework more useful by specifying when to abstain and
which ordinary method to choose.

## References and provenance

<a id="reference-1"></a>
**[1] T. Takehi / RikiyaT.** *Sparks of Weight-Space Programming: LLMs Can
Natively Read & Write Neural Network Weights*. Author report dated
November 2, 2026. [Original report](https://rikiyat.github.io/blog/weight-space-programming/).
The [companion review](primary-paper-review.md) records its experimental
scope and source-reproduction limits.

<a id="reference-2"></a>
**[2] Mengran Li, Bo Li, Weijing Zhao, Chengyang Zhang, Boyu Zhang, Jinfeng Xu,
Zelin Zang, Victor C. M. Leung, and Stan Z. Li.** *WeightLoop: Weight-Space
Programming with Experimental Feedback*. arXiv:2611.07319v1, submitted
November 5, 2026, 02:06:51 UTC.
[Official record](https://arxiv.org/abs/2611.07319),
[reviewed PDF version](https://arxiv.org/pdf/2611.07319v1).
Reviewed PDF SHA-256:
`ac046172b4c25a7602784f7c781b522f133f348a0015ff90bc6ded2d294e4a1b`.

<a id="reference-3"></a>
**[3] Zhuoran Zheng and Doubao.** *Weight-Space Programming for Improving
Image-Restoration Diffusion Models: Technical Conjectures and a Recursive
Improvement Loop*. arXiv:2611.04268v1, submitted
November 3, 2026, 03:56:44 UTC.
[Official record](https://arxiv.org/abs/2611.04268),
[reviewed PDF version](https://arxiv.org/pdf/2611.04268v1).
Reviewed PDF SHA-256:
`6d03ed8c88db51dc51bb3f09065c18e639532b2fff0d99923e4e768876f0798b`.

The follow-up downloads occurred November 9, 2026, about 03:19–03:20 UTC,
which was November 8, 2026, about 21:19–21:20 in America/Chicago. The
restoration manuscript lists access to the original report on November 1,
while the reviewed original report is dated November 2. This is a provenance
inconsistency in the reviewed sources; no explanation is inferred.

Source byte identities identify reviewed artifacts. They do not establish
independent replication, source completeness, statistical certainty, or
framework task success. The public document contains attributed analysis
and proposed experiments; downloaded papers, extracted text, and reading
receipts remain outside the public documentation.
