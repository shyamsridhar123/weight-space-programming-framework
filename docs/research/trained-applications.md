# Trained applications: useful changes, protected behavior, and cost

**Review date: November 8, 2026, America/Chicago.** The experiments, replays,
and supplementary reviews have UTC receipts dated November 9; those events
occurred on November 8 locally. This report covers the frozen trained support
and optical-digit studies, independent artifact replay, and a separate serving
cost supplement. It builds on the [primary-paper review](primary-paper-review.md)
and the [research roadmap](../weight-space/v2/research-roadmap-2026-11.md).

The clearest useful result is an ownership change: keep a trained classifier,
then send its recognized target class to a new destination. A routing overlay
achieved useful heldout changes for both support ownership cases and the digit
review case. Soft head edits also achieved useful ownership changes, with
smaller target gains. No method achieved a useful recognition repair under the
frozen gain and harm criteria. Every hard protected-score candidate abstained.

These outcomes identify a practical role for a generic framework: determine
whether a requested change belongs in policy configuration, whether the learned
representation permits a selective head edit, and what evidence covers the
realized deployment. They do not establish general weight understanding,
whole-domain safety, production savings, or a reason to edit weights whenever
a policy changes. Text routing and image classification supply different task
semantics to the same artifact, selection, retention, and evaluation interface.

## The enterprise use case in ordinary language

A support team changes who owns requests about a card's arrival or a pending
cash withdrawal. Before the change, the classifier identifies an intent and a
fixed mapping selects a queue. After the change, requests with that intent
should reach a specialist queue. Other requests should keep their intended
destinations.

There are two different problems. If the classifier already recognizes the
intent, changing the intent-to-queue mapping can implement the new ownership
policy. If it confuses a bank-transfer charge with a card charge, changing
ownership alone does not fix recognition. The second problem may require an
edit, ordinary fitting, a better representation, or abstention.

The support study authors a six-queue demonstration policy over BANKING77
intents. The two ownership cases append output code `77`, mapped to specialist
queue `5`. The recognition cases seek the original intended queues: bank
transfer charge versus card charge, and ATM support versus Visa/Mastercard
support. Queue meanings, ownership changes, and the specialist policy are this
project's choices. They are not observed BANKING77 business processes.

The non-text study uses genuinely trained optical-digit classifiers. Its
ownership case sends recognized digit `8` to a reserved review destination,
code `10`; its recognition case seeks to repair digit `4` confused with `6`.
This is an image-classification analogue of selective review routing. It is not
an evaluated document-processing system or a deployment at an enterprise.

For a real support operation, the measured analogue of impact would be fewer
target requests sent to the wrong queue, together with the misrouting introduced
elsewhere and the cost of making and serving the change. Customer time saved,
staff cost, incident avoidance, and financial benefit require authorized
operational data. The public benchmark supplies none of those business values.

## What was frozen and what was trained

The [support protocol](../../experiments/research/trained-support/protocol.json)
and [digit protocol](../../experiments/research/trained-nontext/protocol.json)
freeze task selection, training lineages, grids, margins, selection order,
decision rules, missingness, and source identities. This is a local executable
freeze, not external preregistration. The official test partitions were opened
after candidate and task sealing and the global selection gate.

Each domain has six complete encoder/head lineages trained independently from
scratch. Support uses seeds `7001`–`7006`, a ReLU hidden layer of width `64`, and
at most `35` epochs. Digits uses seeds `8001`–`8006`, a ReLU hidden layer of
width `32`, and at most `40` epochs. Both use Adam with learning rate `.015`,
weight decay `.001`, and batches of `256`. Within each lineage the retained
parent is the earliest epoch with highest development original-label accuracy.
Pilot lineages choose the recognition cases and inform sample planning; they
are excluded from the heldout main estimates.

The six parents are independent training runs conditional on a shared corpus,
preprocessing, and case definition. The same requests evaluated across parents
are crossed observations. Requests, cases, candidate grid options, repeated
forwards, and timing repetitions do not become independent trained lineages.

Support preprocessing learns a training-only vocabulary of up to `2,048` word
and bigram terms, with TF-IDF, `log1p`, and L2 normalization. Digits uses `64`
pixel-count features divided by `16`. The encoder is then learned from the
training role. Adaptation changes the readout head or the downstream policy;
the learned encoder is frozen during these interventions.

Whole normalized exact-duplicate groups receive one of five stratified roles:
training, development, target fitting, protected fitting, or calibration.
Target fitting and protected fitting are adaptation data, not the official
heldout outcome sample. The protocol uses at most `128` protected anchor rows
per case.

| Domain | Training | Development | Target fitting | Protected fitting | Calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Support | 5,971 | 965 | 968 | 968 | 1,131 |
| Optical digits | 2,291 | 377 | 377 | 377 | 401 |

The support official test has `3,080` rows. The frozen exact-overlap rule
excludes `25` rows seen in the training roles, leaving `3,055` primary novel
rows. The audit also records `60` token-set overlaps and four exact duplicate
groups within the test. The primary crossed bootstrap consequently has
`3,051` exact request groups. These audits do not establish semantic
independence or eliminate paraphrase overlap. Digits has `1,797` official test
rows, with no audited exact overlap or exact test duplication. Its official
train and test partitions use different writers; writer identifiers were not
available for separating the internal training roles.

## The comparators and their actual access

The frozen [runner source](../../experiments/research/trained-support/source-freeze/scripts/run_trained_research.py)
records the realized comparators. Names alone can obscure what each method
does, so the following definitions are part of the result.

| Arm | Intervention | Information and limits |
| --- | --- | --- |
| `unchanged` | Deploy the parent and its original mapping | Baseline for gain and harm; no requested improvement |
| `overlay` | Redirect a predicted original target code to the authored specialist code | Ownership cases only; sees the prediction, never the heldout truth; weight bytes remain unchanged |
| `context` | Redirect when maximum cosine similarity to a target-role feature row passes the selected threshold | Uses frozen target examples and encoder features; thresholds `.65`, `.75`, `.85`, `.95`; this is retrieval against individual rows, not a centroid |
| `target_only` | Fit a head edit to requested target scores | Target ridge grid `.01`, `.1`; no fitting retention penalty, but the common calibration harm gate still applies |
| `soft` | Fit a head edit with a protected-score penalty | Retention weights `1`, `8`, `64`; ridge `.01`; approximate anchor retention |
| `nullspace` | Restrict a head edit to a numerical protected-feature nullspace | Fixed relative cutoff `1e-12`; admission additionally requires exact zero dyadic anchor residual for the realized endpoint |
| `decision` | Optimize target decisions with a protected decision penalty | Penalties `8`, `64`; `150` optimization steps; distinct from exact score equality |
| `retrain` | Fit a replacement readout head with target oversampling | Oversampling `2`, `8`, at most `12` epochs; encoder remains frozen; this is not full-model retraining |

The parent training and these adaptation grids are fitted operations. Context
retrieval and a predicted-code overlay do not fit new weights, but they still
use a declared representation, task semantics, and selection/evaluation
information. This study has no live provider calls and no learned
weight-reading proposer. It therefore cannot establish an LLM's advantage from
weights over matched metadata, summaries, or a learned reader. The separately
scoped access studies address that question.

Candidate selection first favors satisfying the finite calibration protected
harm cap, then maximizes target gain minus protected harm, then prefers smaller
harm and larger gain. A first-index tie break makes the choice deterministic.
At most one selected candidate per lineage, case, and arm proceeds to sealing.
The candidate grids are not expanded after heldout outcomes.

## Admission, deployment, and useful success are different

In these trained studies, `admitted` means that the selected candidate passes
the structural policy and the frozen finite calibration gain/harm gate. The
structural policy permits edits to `head.weight` and `head.bias` and caps each
realized cell change at `256`. Hard `nullspace` candidates additionally require
an exactly zero anchor residual under dyadic arithmetic.

The structural receipt explicitly reports `behavioral_status: not_checked`
and `formal_certificate: null`. The runner binds its finite calibration rules
to the sealed task. It does not call an integrated whole-domain behavioral
admission rule or attach a Lean certificate to these trained deployments. A
successful structural receipt alone is not proof of behavioral retention.

A rejected candidate deploys the unchanged parent. Its **deployed** target
gain and protected harm are consequently zero, and useful success is false.
Saved rejected-candidate predictions remain diagnostics; they are not the
deployed outcome and cannot be used for reselection. An unchanged tensor
artifact can still have a meaningful external policy intervention: overlay
and context preserve weight bytes while changing downstream queue decisions.
A weight no-op, a decision no-op, a failure, and a policy change must remain
distinguishable.

Heldout useful success requires the admitted deployment to achieve at least
`10` percentage points of target queue-accuracy gain and at most `1` percentage
point of net protected queue-error increase. The project coverage criterion is
`.75` across all assigned cases. Ten percentage points means one additional
correct destination per ten target requests; it is a project feasibility
criterion, not an enterprise SLA or a measured savings threshold.

Target gain and protected harm are paired against the same parent on the same
rows. Positive harm means net error increased; negative harm means net error
decreased. Net harm can hide offsetting improvements and mistakes, so the
saved outcomes also retain decision changes, per-intent or per-digit behavior,
confusion neighbors, and numerical margins. A small mean net harm does not
establish that every protected class or every request was retained.

## Heldout outcomes, with every assigned case retained

The [support results](../../experiments/research/trained-support/results.json)
and [digit results](../../experiments/research/trained-nontext/results.json)
retain every assignment, including abstentions and parent fallback. Missing
outcomes are zero in both studies. The tables average gains and harms across
all assigned lineage/case deployments, including zero-gain fallback. They do
not condition on admission or discard a failed method.

The support denominator is six parents times four cases; the digit denominator
is six parents times two cases. Admission and useful counts explain coverage
and evidence completeness. They are not an impact score or a count of
independent population trials. Gain and harm below are **percentage points**.

| Support arm | Admitted / 24 | Useful / 24 | Mean target gain | Mean protected harm |
| --- | ---: | ---: | ---: | ---: |
| `unchanged` | 24 | 0 | 0 | 0 |
| `overlay` | 12 | 12 | 44.0625 | 0.0995 |
| `context` | 12 | 6 | 41.9792 | 0.4657 |
| `target_only` | 17 | 5 | 42.1928 | 0.5735 |
| `soft` | 13 | 12 | 22.7110 | 0.0815 |
| `nullspace` | 0 | 0 | 0 | 0 |
| `decision` | 3 | 0 | 0.5262 | 0.0649 |
| `retrain` | 12 | 10 | 40.9428 | 0.0221 |

| Optical-digit arm | Admitted / 12 | Useful / 12 | Mean target gain | Mean protected harm |
| --- | ---: | ---: | ---: | ---: |
| `unchanged` | 12 | 0 | 0 | 0 |
| `overlay` | 6 | 6 | 43.4387 | 0 |
| `context` | 0 | 0 | 0 | 0 |
| `target_only` | 0 | 0 | 0 | 0 |
| `soft` | 6 | 6 | 14.7031 | 0.0103 |
| `nullspace` | 0 | 0 | 0 | 0 |
| `decision` | 0 | 0 | 0 | 0 |
| `retrain` | 0 | 0 | 0 | 0 |

Aggregate means mix ownership and recognition. The useful ownership result is
clearer when stated in the case's own terms:

| Ownership case, overlay | Useful lineages | Mean target gain, pp | Mean protected harm, pp |
| --- | ---: | ---: | ---: |
| Card arrival → specialist queue | 6 / 6 | 89.5833 | 0.3870 |
| Pending cash withdrawal → specialist queue | 6 / 6 | 86.6667 | 0.0111 |
| Digit 8 → review destination | 6 / 6 | 86.8774 | 0 |

The parent has no specialist output in its original routing policy, so its
target accuracy for an authored ownership destination is zero. The ownership
gain measures implementing that new policy, not repairing the parent's
original label recognition. Overlay success depends on its ability to identify
the original target code. False target predictions can still misroute
protected requests, as the card-arrival harm illustrates.

Soft editing succeeds usefully for all twelve support ownership assignments
and all six digit ownership assignments. The same finite data favor the
overlay's larger target gains for the ownership task. Context routing passes
calibration in the support card-arrival case but exceeds the useful heldout
protected-harm cap in every lineage. This is a concrete demonstration that
finite calibration admission does not guarantee useful heldout retention.

There is **no useful recognition success**. All digit recognition candidates
are rejected. In support, a few candidates pass calibration and yield small
heldout changes, but none reaches the joint useful criterion. A negative
recognition finding remains part of the result; ownership success cannot
substitute for it.

## Recognition uncertainty and the attainable-gain ceiling

The prespecified primary contrast is soft retention minus overlay on the
recognition cases, averaged within each full training lineage. For support,
the mean gain contrast is `0.213675` pp. Five lineage contrasts are zero; the
sixth is `1.282051` pp. The paired Student-t 95% interval conditional on the
shared fixed requests is `[-0.335594, 0.762945]` pp. The mean protected-harm
contrast is `0.013815` pp, with interval `[-0.021698, 0.049328]` pp.

The independently seeded crossed-bootstrap sensitivity analysis resamples full
lineages and common exact request groups, retaining shared case membership.
Its `2,000` resamples give support gain interval `[0, 1.219512]` pp and harm
interval `[0, 0.049801]` pp. These are descriptive small-lineage estimates;
they do not establish a population advantage, semantic independence, or a
family of confirmatory discoveries. All other arm/case/domain comparisons are
descriptive under the frozen multiplicity declaration.

The digit recognition contrasts and their stored intervals are zero because
both methods deploy the identical parent after rejection. That identity is
an exact explanation of these finite outcomes. It does not establish zero
future variance or certainty about a new population of editable models.

A separate [attainable-gain arithmetic review](../../experiments/research/trained-applications/attainable-gain-ceiling-review.json)
exposes a limitation of the chosen margin. Even a perfect repair cannot
increase target accuracy by more than the fraction of target requests the
parent gets wrong. If a parent is already correct on `38` of `39` target rows,
the best possible gain is `1/39`, about `2.56` pp; a `10` pp gain is impossible.

The calculation authenticates unchanged saved predictions, separately hashed
heldout labels, and the queue mapping from the immutable training feature
bundle against lineage manifests and seals. It performs integer-count
arithmetic only. It runs no inference, fits no candidate, changes no selection,
and leaves all historical thresholds intact. Seeds below are in ascending
main-lineage order.

| Case | Heldout target rows | Parent target errors by lineage | Lineages capable of a 10 pp gain, even with perfect repair |
| --- | ---: | --- | --- |
| Support card ownership | 40 | 40, 40, 40, 40, 40, 40 | All six |
| Support withdrawal ownership | 40 | 40, 40, 40, 40, 40, 40 | All six |
| ATM recognition | 39 | 2, 1, 1, 1, 0, 2 | None |
| Transfer-charge recognition | 40 | 3, 5, 2, 3, 4, 5 | `7002`, `7005`, `7006` |
| Digit 8 ownership | 174 | 174, 174, 174, 174, 174, 174 | All six |
| Digit 4 recognition | 181 | 3, 6, 4, 5, 5, 5 | None |

The boundary `4/40 = .10` qualifies exactly. Integer fractions avoid treating
that case as a failure through rounded arithmetic. Ignoring every harm,
admission, geometry, and implementability constraint, the optimistic useful
coverage ceiling is therefore `15/24 = .625` for support and `6/12 = .50`
for digits. Both are below the frozen `.75` all-case criterion. These are
deterministic bounds for the finite assigned sample, not binomial population
estimates.

This diagnostic prevents attributing every recognition failure to constrained
geometry. The heldout parents also leave too little room to meet the selected
absolute improvement margin in several cases. The correct next step is a
separately frozen study with a development-justified achievable margin or an
error-reduction objective, together with a protected-harm requirement. Selecting
only heldout mistakes or lowering the historical margin after seeing results
would change the question and invalidate the original comparison.

Six independent all-success units would have a two-sided exact 95% lower
coverage bound of only `.540742`. Thirteen perfect independent units would be
needed for that bound alone to exceed `.75`; this is not full power planning.
The frozen planning documents explicitly designate the main studies as
resource-capped estimation. A zero-width empirical interval over equal
lineage outcomes does not overcome that limitation. A careful negative study
can still complete a scoped research deliverable without establishing positive
population success.

## Why exact protected-score editing abstained

For a frozen encoder, write an augmented feature row as `x`, including the
bias coordinate, and a head change as `ΔW`. If protected rows form matrix `X`,
exact preservation of their real-valued scores requires:

```text
X ΔWᵀ = 0.
```

If a target feature lies in the row span of `X`, it also satisfies
`x ΔWᵀ = 0`. An exact score-retaining head edit cannot change that target's
score. A full augmented protected span forces every admissible head change to
zero. Even when rank is smaller than width, nullity supplied only by inactive
feature columns need not provide a useful target direction.

The [support posthoc analysis](../../experiments/research/trained-support/posthoc-analysis.json)
and [digit posthoc analysis](../../experiments/research/trained-nontext/posthoc-analysis.json)
are independently checked against frozen source, features, candidate bytes,
and saved outcomes. Exact rank uses a modular
lower-bound calculation matched to the nonzero-column upper bound. The
relative numerical cutoff remains the frozen `1e-12`; no threshold sweep or
test-based adjustment is performed.

| Domain | Augmented width | Exact protected-anchor ranks, in seed order | Relevant finite geometry |
| --- | ---: | --- | --- |
| Support | 65 | 63, 64, 62, 64, 63, 64 | Every fitted target, analyzed unseen protected test row, and CLINC OOS row lies in the corresponding anchor span |
| Digits | 33 | 29, 29, 29, 31, 30, 30 | Some unseen protected test rows lie outside anchor span; almost every fitted target lies inside |

The sole fitted-target exception is digit seed `8002`, recognition `4`: two
of `38` fitted target rows lie outside the anchor span. Their necessary
rowwise L2 norms for the requested score changes are approximately `500,049`
and `808,185`, far beyond what the `256` per-cell change cap permits. Thus
every frozen requested target-score change is incompatible with exact anchor
retention or exceeds the declared cap.

The fitted objective raises the desired score to the parent's maximum score
plus `1`. This is stronger than merely obtaining the desired argmax decision.
Failure of that particular score objective is not a theorem that every
decision-retaining construction fails. The `decision` comparator addresses a
different relaxation, but it has no useful heldout result in these studies.
A general solver for multiple incompatible target requirements, controlled
cutoff/budget sweeps, and an alternative decision-feasibility analysis remain
research work.

Twelve of the twenty-four support hard candidates and six of the twelve digit
hard candidates have exactly-zero finite anchor residual receipts. None
passes the complete frozen admission conditions. Exact finite retention can
coexist with no useful target change. Soft candidates have nonzero anchor
residuals throughout: approximately `0.188754`–`40.3959` for support and
`0.580992`–`69.7583` for digits. Their useful decisions do not turn those
residuals into exact preservation.

Dyadic arithmetic here evaluates the realized finite float values as exact
real numbers on supplied stored features. It is not a Lean theorem, proof
that every raw input produces those features, proof of native floating-point
equality, or whole-domain behavioral admission. Unseen digit rows outside the
anchor span also prevent replacing a finite anchor claim with universal
retention. The candidate-specific proof work in the roadmap remains separate.

The later [bounded feature-feasibility engineering supplement](feature-feasibility.md),
dated November 9, adds joint exact real-score consistency and minimum-norm
diagnostics with explicit unresolved limits. Numerical compilation and stored
endpoint residuals remain separate. This source/engineering extension does
not rerun the trained studies or alter the historical gap statement above;
it does not itself supply a development-only trained cutoff/budget sweep or
a general alternative decision-feasibility analysis. No generic exact-target
admission/useful-success gate is added by that extension.

The separate later [synthetic trained-development retention study](retention-development.md),
also dated November 9, records the frozen comparison of exact score-equation
feasibility, five numerical hard-projection cutoffs, three soft retention
penalties and three decision-hinge penalties. It retains all thirteen
configurations across eighteen score-budget/margin policies for six paired
seed/task assignments from three trained synthetic parents. Fitting and
calibration are reported separately, but every role was designated
development before the attempt and calibration participates in the gates.
This supplies Issue #6 c1 at that bounded synthetic trained-development scope;
it does not retune the historical fixed-cutoff studies, provide heldout
confirmation, select a winning policy or establish population retention or
business usefulness. The generic exact-target admission/useful-success gate
and general alternative decision-feasibility conditions remain incomplete;
c4 stays partial and Issue #6 stays open.

## Native numerical behavior and out-of-scope stress

The independent replay verifies stored native float32 predictions and paired
metrics. On the finite stored observations, parent and candidate native
float32 versus binary64 argmax disagreement counts are zero. Maximum candidate
score differences are approximately `4.33529e-5` for support and `2.46513e-5`
for digits. Score differences remain visible even when argmax decisions agree.

These comparisons bind the tested native execution to the stored artifacts
and features. They do not establish bitwise agreement on another device,
compiler, batching schedule, backend, or unseen low-margin input. Score
retention in exact-real arithmetic and native decision conformance answer
different questions.

Support also evaluates the `1,000` CLINC `oos_test` queries after sealing.
Their labels are unrelated to the authored BANKING77 queue policy. Dispatch
changes, specialist activation, and confidence are out-of-scope stress
diagnostics, not routing correctness. Drift in a rejected candidate remains
candidate drift; deployed fallback is evaluated as the unchanged parent.
Neither study adds authored paraphrases, multi-intent requests, image
corruptions, blinded enterprise review, or live customer evaluation.

## Cost: measured stages and their limits

The original trained records charge parent preparation/training and adaptation
stages separately. A supplementary [cost protocol and execution lock](../../artifacts/research/serving-cost-supplement/execution-lock.json)
adds fresh-process head-serving measurements and package I/O without rerunning
training or changing heldout selection. Its [summary](../../artifacts/research/serving-cost-supplement/summary.json)
and [process index](../../artifacts/research/serving-cost-supplement/process-index.json)
retain assigned and failed/missing measurements. There are no supplement
failures.

The supplement uses the first declared main lineage in each domain, `7001`
and `8001`, and its ownership case. It measures eight arms at batch sizes
`1` and `32`, with three fresh processes per configuration, four Torch CPU
threads, ten warmups, and thirty timed warm calls per process. It executes the
native readout and the actual deployed overlay/context/mapping on cached
development features. Rejected arms measure parent fallback. Independent
checks reproduce the deployed-policy outputs and verify package manifests and
copy/hash records; timing repeats remain engineering measurements of one
trained lineage per domain.

The following values are pooled warm medians in **microseconds per batch**,
not per input for batch `32`. An asterisk identifies rejected-policy parent
fallback.

| Arm | Support batch 1 | Support batch 32 | Digits batch 1 | Digits batch 32 |
| --- | ---: | ---: | ---: | ---: |
| `unchanged` | 148.05 | 347.60 | 101.00 | 181.90 |
| `overlay` | 172.30 | 193.40 | 154.70 | 234.05 |
| `context` | 262.95 | 236.15 | 149.50* | 168.15* |
| `target_only` | 187.65 | 149.45 | 128.50* | 174.15* |
| `soft` | 99.40 | 237.65 | 94.40 | 138.40 |
| `nullspace` | 115.20* | 215.60* | 95.35* | 121.05* |
| `decision` | 109.15* | 311.65* | 98.15* | 141.10* |
| `retrain` | 127.10* | 222.50* | 93.65* | 146.55* |

Identical fallback policies receive different timings in different processes.
This demonstrates why the table cannot supply a causal method ranking or a
speedup claim. Execution order is fixed, and the concurrently used workspace
has no performance isolation. Per-process medians, first-call measurements,
and warm tails remain in the records instead of being replaced by one headline
number.

Across the supplement's `96` processes, median process startup is
`0.19529435` seconds, imports/setup `5.3693133` seconds, and input read/verify
`0.03447585` seconds. Source verification's `0.2492438` seconds is nested
inside imports/setup, so it must not be added again. The `48` package-copy
repetitions have median inventory time `0.0009542` seconds, copy time
`0.02458655` seconds, and post-copy hashing `0.07153445` seconds. These are
the timed operations on this machine, not a controlled cold-disk experiment.

| Domain and ownership arm | Serving package bytes | Separate audit-package bytes |
| --- | ---: | ---: |
| Support overlay | 730,904 | 376 |
| Support soft edit | 731,921 | 148,353 |
| Digits overlay | 21,294 | 376 |
| Digits soft edit | 21,834 | 35,048 |

The package manifests distinguish serving content from audit content. These
byte sizes do not establish process memory, loading latency for another
environment, secrecy, or an amortized operational benefit.

For context, the historical main-run sums below combine all-options
fit/selection/materialization with the heldout workflow for each arm. They are
**partial historical incremental workflow seconds**, not serving latency,
complete project cost, or contemporaneous controlled comparisons.

| Arm | Support seconds | Digit seconds |
| --- | ---: | ---: |
| `unchanged` | 6.262865 | 0.953801 |
| `overlay` | 5.546188 | 0.870393 |
| `context` | 6.648982 | 0.920094 |
| `target_only` | 10.538548 | 1.395286 |
| `soft` | 111.936081 | 6.728660 |
| `nullspace` | 55.672479 | 4.064643 |
| `decision` | 12.521297 | 4.622952 |
| `retrain` | 41.764764 | 4.544497 |

Shared parent training is separately charged at `16.6899731` seconds for
support and `2.5907619` seconds for digits. Historical diagnostic timers
include several native and binary64 calls, receipt construction, and workflow
operations. They are neither one-pass serving measurements nor authenticated
historical machine-performance results. The independent fresh replay's
elapsed time also measures verification, not serving.

The head-only supplement excludes text/image preprocessing and encoder
execution from warm serving timing. Full workflow acquisition, failed-work
recovery, controlled cold loading, peak memory, energy, model/device/scale
variation, full-model retraining, LoRA preparation, and complete
preparation/amortization/crossover comparisons remain incomplete. Neither a
frozen cost-improvement margin nor enterprise ROI is established. Nested
timers and supplementary counters must not be added to broad historical
totals.

## Prospective raw-text serving engineering

The separate [raw-text runner](../../scripts/run_raw_text_serving.py) adds a
prospective path from original text to queue routing. It does not extend or rerun
the historical head-only supplement. The caller supplies independent SHA-256
pins for a package manifest and a raw JSON batch. The package captures exactly
`head.json`, `encoder.npz`, `preprocessing.json`, `lineage.json`,
`deployed-head.json` and `policy.json`; its manifest pins each file's size and raw
bytes. The batch contains ordered `request_id` and `text` rows. Both the raw batch
hash and an ordered-row digest retain original punctuation, text, IDs and order.

The shared captured-parent validator restores preprocessing, encoder and original
head independently of the deployed head. It does not follow the historical
feature-file references in lineage metadata. Only unchanged and edited-readout
policies are supported. An admitted edit must match the candidate identity;
rejection deploys the exact parent. The rejected candidate identity remains a
declaration because those candidate bytes are not supplied. Admission is also a
declaration here: this runner does not independently establish admission,
scientific freshness, source authentication or evaluator execution attestation.

Each worker call starts again with raw text, including first, warmup and measured
calls: `TextPreprocessing.transform`, native encoder execution, native head
execution, argmax and the declared queue map. The numeric lane is deterministic
Torch CPU float32 with one compute thread. Artifacts retain binary64 storage;
finite edited values are cast to float32 for native inference. Parent, candidate,
deployed endpoint and policy identities remain distinct in receipts.

The supervisor durably records all job assignments before launching a fresh
process for each job, imposes a timeout, and retains launch, timeout, nonzero-exit
and receipt failures. Interrupted stages retain null durations alongside already
completed stages. Success latency samples condition on successful jobs; failed
jobs, including their completed calls, remain visible in assignment totals and
the excluded-sample count. Existing output directories cannot be reused.

Instrumentation reports imports, native setup and package/batch load separately
from first/warmup/measured calls and enclosing process wall. Stage timers cover
preprocessing, encoding, and native head plus policy. Raw-call wall covers the
whole batch and intervening durable progress writes, excluding output hashing.
It therefore includes instrumentation overhead. Nested timers are not additive.
There is no direct load-through-first-output metric, per-request percentile or
controlled cold-cache claim; fresh processes do not imply cold operating-system
file caches. Memory and energy remain unknown.

The [November 9 authored engineering receipt](../../provenance/verification/raw-text-serving-20261109.json)
records one passing bounded capture. It checks original/empty/OOV text,
unchanged/admitted/rejected deployment, captured-byte isolation, failure evidence
and complete assignment accounting. One actual baseline child exercises the CLI
boundary with first and measured calls; admitted and rejected selection are
checked in parent-process workers. The other supervisor outcomes are controlled
fixtures, and the checks are not exhaustive. Two existing prospective-loader
nodes check the shared extraction without historical features or lineage loads.

This establishes an engineering path for a future separately sealed serving
study. It supplies no trained-lineage evidence, historical cost recovery,
production latency, speedup, throughput, matched crossover or enterprise ROI.
Queue policy is authored, with no measured waits, transfers or resolution costs.
The historical exclusions above retain their scope, and Issue #7 c3 and
#14 c1/c5 remain partial and open.

## Independent replay and publication provenance

Before reading heldout comparisons, independent review authenticated the global
selection gate, domain protocols, source manifests, lineage identities, and
sealed attempts. Successful fresh-process replay subsequently reproduced all
`192` support and `96` digit replayable assignments: saved native predictions,
metrics, immutable proposal application, receipt traces, exact rollback, and
applicable endpoint-retention receipts. These numbers describe expected
artifact coverage. They are not a success-rate claim about trained editing.

The current public [support replay](../../experiments/research/trained-support/independent-replay.json)
and [digit replay](../../experiments/research/trained-nontext/independent-replay.json)
contain the successful fresh replay bytes. A preceding missing-NumPy launch
failed before useful replay; copied historical records labeled independent
replay were not a new independent execution. Those historical bytes are
preserved separately. Historical posthoc input hashes refer to the preserved
historical receipts, so their identities must not be silently interpreted as
the replacement fresh receipts. Independent review explicitly checks that
correspondence.

Replay used Python `3.12.10`, NumPy `2.5.3`, and Torch `2.8.0+cpu`. It verifies
the public derived-feature experiment and its realized artifacts. It does not
independently reacquire the raw corpus, reconstruct training from new raw
inputs, independently retrain a second set of models, prove encoder feature
semantics, or validate historical runtime metadata. Those are separate
reproduction obligations.

The attainable-gain receipt is a byte-identical public copy of the completed
posthoc arithmetic review, with repository-relative input identities. Its
source hash records the original analyzer. This receipt records arithmetic
over authenticated prior evidence; it does not change a frozen study or imply
that every private review dependency is distributed as a runnable public
analysis package.

Serving-process records contain build-configuration path strings incidental
to scientific measurements. Their public derivatives redact only those paths.
The [publication correspondence](../../artifacts/research/serving-cost-supplement/publication-derivative.json)
binds original and public hashes, rather than relabeling original records as
published bytes. Independent comparison of all `96` preserved originals and
public records verifies the exact NumPy build-path replacements and Torch
compiler-path replacement, with every other parsed value unchanged. The
execution lock, package manifest, measurement summary, and I/O record retain
their prior identities. No measurement is rerun by this publication step.
The public correspondence makes the transformation inspectable; the preserved
raw build-path records remain outside the public checkout.

| Evidence object | SHA-256 |
| --- | --- |
| Global selection gate, raw file | `26e67134db350133a2f3ee2a324c39d79d53fd6915ee8513bdae37cc3fcbdc31` |
| Shared trained source manifest | `bf9a8a0a6c56fc489a3380891f08db80271e7c40b2de68efa2afb17249c01ee1` |
| Support successful fresh replay | `07645b24b5517027391049a20522e2bb3a99a7df2b335530f9622a7e14f928be` |
| Digit successful fresh replay | `05ab8ea3edf0935f0c4337000f111fe461c33fe98eaf600f60a3e5c5fb8fc8dc` |
| Public attainable-gain receipt | `53b95f110d79d6a7154fb0fc25e77519ec8550363aa0f0878fbe2fb104978854` |
| Original attainable-gain analyzer source | `745ad2e1ec5efc5bda014911b92b29ee03553c6934da33ae7dab85f8713ab594` |
| Serving supplement source | `6df4578019d4822bdbaba84a34fb3fe92c7cb23b592d4bdeb242afcb0d98ca0a` |
| Serving measurement summary | `f3a125e614cb5332c2ee16c6749f4b795e7de7da8bf5d269cb967dedc54455cf` |
| Serving public process index | `59e37ae6ccc232a09904c065d97541ea138650fafdd6862ae6b23ee01272c6dd` |
| Serving publication correspondence | `11bb0f3cf333fa928c16c510a0a4e9b22e2b85cbdc0f5a5bc2140e735651a432` |
| Independent cost-publication comparison receipt | `800a48f002e724a8cd3deedefbd74827b25838c552bd85524a08c9f4109aa81d` |

The last receipt is retained with independent review records outside published
measurement files. Hashes identify reviewed bytes and transformations. Their
length or quantity does not measure scientific confidence or product impact.

## Dataset rights and publication boundaries

[Support dataset notices](../../experiments/research/trained-support/DATASET-NOTICE.md)
and [digit dataset notices](../../experiments/research/trained-nontext/DATASET-NOTICE.md)
bind acquisition revisions, byte identities, attribution, and license text.
BANKING77 is pinned to PolyAI-LDN revision
`57ec275d8078af65b7731c2a98be812d844a6d6b`, under CC BY 4.0. CLINC OOS is
pinned to revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`, under CC BY 3.0.
UCI Optical Recognition of Handwritten Digits, DOI `10.24432/C50P49`, is used
under CC BY 4.0. Their authors and original publications are attributed in
those notices.

The framework's MIT license covers its code; it does not relicense these
datasets. Published derived models, features, role IDs, labels, predictions,
and evaluation records retain the stated dataset attribution boundaries. Raw
acquisition copies and incidental contact metadata are not redistributed.
Derived features are not proof of anonymization or irreversible removal of
source information.

## Where these results take the framework

The immediate enterprise demonstration should explain one ownership change:
recognize the request, redirect it to the specialist, check the other queues,
and show the evidence and rollback. The same demonstration can compare the
overlay with a soft edit and explain why the exact-retention method abstains.
Its measured impact is target routing gain, protected harm, useful coverage,
and scoped cost. Recognition and ownership should have separate results.

The next recognition study should choose tasks and feasible effect criteria
from development information, declare its independent trained units and
uncertainty plan, then seal candidates before opening heldout outcomes. It
should preserve unchanged, predicted-code policy, context, target-only, soft,
hard, decision, and appropriate retraining comparisons. A negative result
remains admissible evidence; an editor should not need to win against a simple
policy configuration to make the framework useful.

The November 9 [prospective recognition-repair design](recognition-repair-next-study.md)
develops this next question with explicit data-exposure, attainable-gain,
matched-comparison, uncertainty and complete lifecycle-cost gates. It remains
a source-only proposal; the historical criteria and results above keep their
original scope. Issue #7 c3 remains partial until the missing lifecycle costs
are measured, and the broader Issue #14 and #11 scopes remain unresolved.

| Research issue | Evidence now delivered | Remaining work within the issue's scope |
| --- | --- | --- |
| #6: retention geometry | Trained exact rank, residual, inactive-column, unseen-row, out-of-scope, and abstention analysis; later [bounded supplied-feature score-consistency diagnostics](feature-feasibility.md) with conflict witnesses and exact minimum norms; separate [synthetic trained-development cutoff/budget/margin sweep](retention-development.md) | Generic exact-target admission policy; general alternative decision-feasibility conditions; unfinished exact judgments remain unresolved; no heldout or population-retention claim from the development sweep |
| #7: trained enterprise comparison | Independently trained support parents, ownership/recognition separation, fitted and non-editing baselines, heldout negative outcomes, all-assigned replay | A separately justified feasible recognition objective; fuller cost comparison and declared cost margin; operational validation before business-impact claims |
| #8: access and resource accounting | These fitted studies declare task metadata, data roles, candidate grids, and partial resource stages | Live provider and matched context/summary/weight/reader access evidence belongs to the separate access study; these results do not establish weight-reading advantage |
| #11: trained non-text use | Independent optical-digit training, selective review task, recognition failure, native/real-score diagnostics, and replay | Broader task and representation scope; complete comparable workflow cost; uncertainty appropriate to the claim sought |
| #14: cost and amortization | Head serving, startup/setup, package bytes, and copy/hash supplement with preserved measurement provenance | Cold/full workflow, failed-work recovery, memory/energy, model/device variation, full-model or LoRA comparators, preparation/amortization and crossover evidence |

These entries identify delivered evidence and remaining claim limits. They do
not redefine an issue's original acceptance criteria or demand positive
powered confirmation where a scoped negative result is a valid deliverable.
The [work index](../weight-space/v2/research-issues-2026-11.md) retains the
broader proof, source-repetition, symmetry-transfer, restricted-construction,
and synthetic-unlearning dependencies described in the primary review.

The experiment's useful contribution is inspectable choice: a policy overlay
can win an ownership task, exact geometry can justify abstention, and a frozen
comparison can report failure without disguising it as general success.
Wider claims should grow from separately declared evidence, with the same
task-specific semantics and artifact-level accountability.
