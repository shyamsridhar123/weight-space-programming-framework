# Support ownership: fixed retention development study

**Methodology and result date: November 9, 2026, America/Chicago.** The single
frozen attempt finished all six assignments within its 600-second ceiling.
All thirteen configurations completed, but nonzero edits passed no fixed
protection gate at score budgets through 6.4. Some assignments met the fixed
development useful-success criteria at budget 64 and margins 0 or .01; no
configuration or policy was selected as a winner. This report describes
[run_retention_development.py](../../scripts/run_retention_development.py),
study `wsp-support-ownership-cutoff-budget-development-20261109-v1`.
The driver is repository-only; it is not included in the wheel or exposed as
a CLI command.
The recorded behavior is synthetic development evidence, with calibration
participating in the gates. It supplies no heldout efficacy or business-impact
claim.

A business wants invoice disputes to reach a Billing dispute specialist, or
account-access requests to reach an Account recovery specialist. The meaning
of each request stays the same. Output 3 represents the new owner, while the
three semantic intents remain invoice dispute, account access and subscription
cancellation. Other requests should keep their parent-model owners.

This is one adapter for the generic weight-space framework: freeze the
representation, constrain a proposed head edit, and inspect its realized
retention and behavior. The framework is not specific to support routing.
The exercise explores score and margin budgets for this adapter; it does not
establish that a head edit is preferable to a policy mapping or reproduce the
[earlier trained studies](trained-applications.md).

## Fixed development design

Seeds **11401, 11402 and 11403** each train one parent. Both ownership transfers
edit that unchanged parent independently. The six seed/task assignments stay
in every policy denominator. There are three independent training seeds; the
two tasks sharing each parent are paired. There are no quality exclusions,
replacement seeds, adaptive retries or winner selection.

All inputs are authored synthetic development data. Ten fixed lexical
families per intent each produce sixteen prefix/suffix variants. Families
01–06 supply 288 training rows; family 07 supplies 48 epoch-selection rows;
families 08, 09 and 10 supply 48 target, protected and calibration rows each.
Each transfer fits sixteen target-intent rows from family 08 and protects
32 non-target rows from family 09. Calibration uses all 48 family-10 rows,
including 32 non-target protection rows. Target-intent rows are deliberately
excluded from protection. Family separation does not establish independent
semantic generalization. Calibration participates in the gates and is not
heldout evidence.

The design commits all roles as development before caller exposure and freezes
the schedule, source/runtime identities and reporting rules before training.
Vocabulary/IDF fitting uses training rows only, with width 128. Each parent
uses eight hidden units and 35 Adam epochs; the earliest maximum development
intent accuracy selects its epoch. Native scoring uses deterministic Torch
CPU float32 and lowest-output-index argmax ties.

| Configuration | Fixed settings |
| --- | --- |
| Unchanged parent | No fitted edit |
| Unrestricted | Ridge .01, no retention penalty |
| Hard numerical projection ×5 | Ridge .01; relative cutoff `1e-12`, `1e-8`, `1e-4`, `1e-2` or `1e-1` |
| Soft retention ×3 | Ridge .01; penalty 1, 8 or 64 |
| Decision hinge ×3 | Penalty 1, 8 or 64; 150 fixed optimization steps |

Only the specialist output's eight weights and bias may change. The nine-cell
write limit, absolute specialist endpoint cap 256 and exact change cap 256
apply before behavior gates. Proposals are rounded to float32, never clipped
or refitted after rejection. Other output rows, encoder and preprocessing
remain fixed. Every reached proposal, including structural rejections, is
exported and checked against its program.

## Exact equations, numerical proposals and realized protection

The common score goal raises the specialist score to the parent's largest
original-owner score plus .5 when needed, with zero requested changes to other
outputs. Native parent captures are lifted to float64 before goal arithmetic.
On the nine editable augmented coordinates, the supplied-feature system is
`R X = 0; T X = D[:,3:4]`, using 32 fitting anchors and sixteen target rows.

The bounded rational analyzer reports consistency, ranks/nullity, witnesses
and exact minimum squared norms where completed. Its limits are 250,000 work
units and 4,096 rational bits. Unfinished judgments remain unresolved on
exhaustion; completed results remain distinct. The same exact system is
analyzed at all five numerical cutoffs. Repeated analysis, including compiler
analysis, is charged work and supplies no independent observations.

Numerical SVD projection and ridge fitting can differ from exact consistency.
The decision helper uses its existing float32 hinges rather than the common
float64 score target. The analyzer supplies no executable rational edit.
Real-score feasibility does not guarantee a float32 endpoint; score-equation
infeasibility does not rule out a useful ownership decision. Exact dyadic
receipts for stored endpoint changes and native float32 scoring are checked
separately on the supplied protected features.

## Eighteen fixed gates per configuration

Every structurally admitted candidate is evaluated without refitting at all
six score budgets `64 × [0, .0001, .001, .01, .1, 1]` and three native margin
thresholds `0, .01, .1`. Budgets mean their stored binary64 values, with exact
rational comparison and no slack. Across all four outputs on the 64 protected
fitting/calibration rows, both exact and native maximum score drift must fit
the budget, every owner must equal its parent owner, and the minimum native
margin must meet the threshold. Strict zero requires both maxima exactly zero.
Native score captures are lifted before drift subtraction; top-two margin
subtraction remains float32.

Useful success also requires target ownership accuracy gain at least .10,
protected ownership error increase at most zero, and zero newly harmful
protected owner flips, in **both fitting and calibration**. These are fixed
engineering criteria without stakeholder validation as business requirements.
A no-op has zero transfer gain and receives no useful-success credit.

## Single attempt and source identity

One attempt ran under a **600-second external ceiling including
preparation**, imports, training, fitting, diagnostics, scoring, exports and
finalization. Cooperative checks reserve ten seconds for finalization; only
the outer watchdog can interrupt a stuck native operation. A rejected or
failed first execution request remains the attempt.

The outer receipt records invocation at **06:53:47.116035 CST** and completion
at **06:55:09.797854 CST** on November 9, 2026, `returncode=0`,
`child_lifecycle_completed`, and exit observed within the ceiling. Observed
child wall time was **82.331 seconds**; outer wall time was **82.682 seconds**.
The receipt lists no outer failures, no retry or replacement attempt, zero
provider invocations by the outer process, and unchanged immutable input
identities. Process completion alone is not scientific success.

The outer's finite manifest observations contain one matching record for each
assignment; every record has status `schedule_finished`:

| Seed | Transfer | Recorded disposition |
| --- | --- | --- |
| 11401 | invoice-dispute-transfer | schedule_finished |
| 11401 | account-access-transfer | schedule_finished |
| 11402 | invoice-dispute-transfer | schedule_finished |
| 11402 | account-access-transfer | schedule_finished |
| 11403 | invoice-dispute-transfer | schedule_finished |
| 11403 | account-access-transfer | schedule_finished |

The child summary reports behavior-metric source identity `verified` and the
frozen evaluator identity verified at completion; the outer observes source
unchanged at completion. All 78 candidate records completed and were
structurally eligible. There were **zero structural rejections, missing
structural outcomes, failed, unattempted or unfinished candidate records**.
All 234 configuration/policy aggregates contain six computed and six
source-verified evaluated assignments: 1,404 policy/assignment evaluations of
the same 78 candidates, not 1,404 independent edits. There were zero computed
source-unverified, source-invalidated or unknown/unfinished gate assignments.

These are source-bound records, not independent authentication of provenance,
scientific freshness, semantic independence or evaluator execution. The child
summary leaves those four attestations false. The outer did not recompute the
gates or open model arrays and granted no useful-success credit itself. Its
runtime identity covers interpreter bytes and distribution-version metadata,
not an exhaustive binary attestation.

## Recorded exact and numerical outcomes

For **each** cutoff `1e-12`, `1e-8`, `1e-4`, `1e-2` and `1e-1`, the common
exact score system was **infeasible in 6/6 assignments**. Exact minimum norm
was `not_applicable` in 6/6 at each cutoff, with no unresolved consistency
outcomes. These thirty entries repeat the same six systems; changing an SVD
cutoff does not change exact consistency. Completed common and compiler
diagnostics record **1,080,541 exact-work units** in aggregate; this is charged
repeated work, not a per-analysis limit, count of independent systems or
measure of proposal quality.

The three tight hard cutoffs (`1e-12`, `1e-8`, `1e-4`) each realized **6/6
parameter no-ops**, with zero target gain. The parent contributes six more
no-ops. Both larger hard cutoffs and all soft and decision configurations
changed parameters in 6/6 assignments. Thus there were 24 no-op candidate
records, including the six unchanged parents, and 54 nonzero edits.
The decision configurations changed parameters but produced zero target
ownership gain in both fitting and calibration.

Exact consistency concerns the requested score equations. The gate results
below concern stored dyadic endpoint changes **and** realized native float32
scores, parent-owner retention and native margins. A failed combined gate
does not identify which scalar failed. All 54 nonzero edits failed every
fixed budget through 6.4; strict-zero passing candidates were only parents
and no-ops and earned zero useful-success credit. No configuration passed
margin .1 at any budget, including the unchanged parent. These outcomes do
not convert score-equation infeasibility into decision infeasibility.

## All configurations: fitting and development calibration

Each value below is the observed conditional mean over **all six assigned
seed/task pairs**, in ownership-accuracy fractions, rounded to six decimal
places. Target gain is relative to the unchanged parent. Positive protected
error increase is harm. Fitting and calibration remain separate; no heldout
measure is present.

| Configuration | No-ops /6 | Fitting target gain | Calibration target gain | Fitting protected error increase | Calibration protected error increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent | 6 | 0 | 0 | 0 | 0 |
| Unrestricted | 0 | .572917 | .822917 | .083333 | .031250 |
| Hard `1e-12` | 6 | 0 | 0 | 0 | 0 |
| Hard `1e-8` | 6 | 0 | 0 | 0 | 0 |
| Hard `1e-4` | 6 | 0 | 0 | 0 | 0 |
| Hard `1e-2` | 0 | .041667 | .552083 | 0 | 0 |
| Hard `1e-1` | 0 | .187500 | .822917 | 0 | 0 |
| Soft 1 | 0 | .145833 | .822917 | 0 | 0 |
| Soft 8 | 0 | .104167 | .802083 | 0 | 0 |
| Soft 64 | 0 | .052083 | .656250 | 0 | 0 |
| Decision 1 | 0 | 0 | 0 | 0 | 0 |
| Decision 8 | 0 | 0 | 0 | 0 | 0 |
| Decision 64 | 0 | 0 | 0 | 0 | 0 |

All four metrics have six observed assignments and zero missing assignments
for every configuration. Their all-assigned lower and upper mean bounds
therefore equal their observed means (before display rounding). The fixed
missing-outcome rule would use `(sum(observed) ± (6 − n))/6` for differences
bounded by `[-1,1]`, with zero useful credit for missing outcomes. Those are
missing-outcome bounds, not sampling confidence intervals. Aggregate zero
protected error increase alone is insufficient for the gate: owner changes,
native margins, exact/native drift and per-assignment useful criteria still
apply.

## All eighteen gate policies per configuration

Each cell lists three **pass:useful counts**, in margin order **0, .01, .1**.
Every count retains denominator **six**: `3:3` means gate passes **3/6** and
useful successes **3/6**; `2:0` means **2/6** and **0/6**. Each budget column
contains all three margins, so the table reports all eighteen fixed policies
for each of the thirteen configurations. Displayed budgets abbreviate their
stored binary64 values; exact comparisons use those stored values.

| Configuration | Budget 0 | .0064 | .064 | .64 | 6.4 | 64 |
| --- | --- | --- | --- | --- | --- | --- |
| Parent | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 |
| Unrestricted | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 3:3 / 1:1 / 0:0 |
| Hard `1e-12` | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 |
| Hard `1e-8` | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 |
| Hard `1e-4` | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 | 6:0 / 2:0 / 0:0 |
| Hard `1e-2` | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 4:0 / 1:0 / 0:0 |
| Hard `1e-1` | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 3:3 / 1:1 / 0:0 |
| Soft 1 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 5:3 / 2:1 / 0:0 |
| Soft 8 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 4:3 / 1:1 / 0:0 |
| Soft 64 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 4:1 / 1:1 / 0:0 |
| Decision 1 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 6:0 / 2:0 / 0:0 |
| Decision 8 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 6:0 / 2:0 / 0:0 |
| Decision 64 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 0:0 / 0:0 / 0:0 | 6:0 / 2:0 / 0:0 |

Only budget 64 yielded any nonzero-edit protection passes. At margin 0 the
recorded useful fractions were 3/6 for unrestricted, hard `1e-1`, soft 1 and
soft 8, and 1/6 for soft 64. At margin .01 those same five configurations
each recorded 1/6. All other policy/configuration useful fractions were 0/6.
These paired, small development observations select no winner and provide
no independent policy comparison or population success-rate estimate.

For each policy, the summary separately proposes unchanged-parent fallback
for the `6 − pass` nonpassing assignments. Fallback has zero change, zero
transfer gain and zero proposal useful credit. The three trained parents were
available for all six seed/task assignments; **no fallback was actually
deployed**. This proposal is distinct
from both the parent baseline and useful-success outcomes.

## Recorded costs and evidence limits

| Measured item | Recorded result |
| --- | ---: |
| Outer wall time | 82.681680 s |
| Child wall time to observed exit | 82.330998 s |
| Preparation observed wall time | 6.704229 s |
| Run observed wall time | 74.036531 s |
| Sum of recorded stage wall times | 65.972227 s |
| Sum of recorded stage CPU times | 55.375000 s |
| Stage dispositions | 614 succeeded |
| Unfinished stages with unknown timing | 0 |
| Exact work in completed common and compiler diagnostics | 1,080,541 units |

The timing scopes differ and are not additive: stage sums omit uninstrumented
work, while the outer observes the complete child lifecycle. Costs include
repeated diagnostics and reached candidate verification. The child's
`external_cap_execution_attested` field is false; the separate outer receipt
records the 600-second ceiling and observed exit within it. Ending persistence
may remain outside child stage timings even on completion. No memory or energy
measurements were taken.

This report transcribes finite receipts and aggregate JSON summaries without
opening model arrays, cohort rows, SQLite or raw logs, or recomputing study
results. The source evidence consists of `OUTER_RESULT.json`,
its `child_json_observations`, and `study-output/summary.json`. The observed
summary file has SHA256
`36b4a46d9b331a452bd36eff7eef7ae570dfad911ffce7de2cfaf9a8d7b2fceb`
(196,752 bytes); the outer receipt has SHA256
`6cd8dacc460c5028e421aaf6814901433a955a7473fb26f7d7b128979b112070`
(22,290 bytes). The outer records a finished manifest identity of SHA256
`6dfeab38a7dfae4c93ec427d6c2b1631dabccaca0e8d28592c1c1fe42eaccba4`
(2,140,639 bytes); this report uses only its finite outer observations.

The public evidence comprises the
[finite summary](../../reports/research/retention-development-20261109/summary.json)
and [verification receipt](../../provenance/verification/retention-development-20261109.json).
These are path-sanitized finite projections of the recorded summary and outer
receipt. Preparing these projections does not rerun the study or authenticate
its provenance, freshness, semantic independence or evaluator execution.

The earlier trained-study negatives remain unchanged. This synthetic
development exercise supplies no untouched evaluation, population retention
guarantee, production-ticket result, operational savings, enterprise
effectiveness or business-impact evidence.
