# A completed bounded synthetic unlearning pilot

Protocol and source-amendment date: **November 8, 2026**. Final report review:
**November 9, 2026, America/Chicago**. This is a separate experiment for
[issue #18](https://github.com/shyamsridhar123/weight-space-programming-framework/issues/18).
The framework remains generic. Changing a support route is a behavior change;
this experiment instead asks whether a newly trained predictor stops revealing
designated synthetic facts while preserving other facts.

The completed v2 pilot gives a negative forgetting result for both editing
methods. Canonical refusal suppresses the displayed answer while leaving
alternate-query access intact. Retained fine-tuning leaves most forgotten codes
accessible. Uniform-target editing lowers ordinary audit and extraction
accuracy, but neither editing method passes the frozen weight-only or
archive-inclusive joint criterion under any of the four transforms. The fresh
retain-only reference passes its relative-control gates; that does not establish
absolute erasure or the absence of information in every representation.

The original [prepared experiment package](../../experiments/research/synthetic-unlearning/)
remains a source-only historical record. No predictor training or outcome
evaluation had occurred when it was prepared. The separately frozen
[v2 protocol](../../experiments/research/synthetic-unlearning-v2/protocol.json)
was reviewed and authenticated before the completed training, selection, audit
and independent-check sequence. The immutable preparation-status strings in
both freezes describe their creation times, rather than the current state of
the v2 study. Previously frozen routing, non-text, cumulative-admission and
bounds studies remain separate.

**Source amendment before execution.** Independent review found that the
original arithmetic seed formula gave optimization variant 1 and audit variant 0
the same initialized RNG state for an author/attribute. It also found that
training and recovery attempt records buffered until phase completion. The
original prepared package is retained byte for byte as the review record; its
description of independent audit seeding below must be read with this correction.
The separately versioned [v2 amendment](../../experiments/research/synthetic-unlearning-v2/amendment.json)
preserves the actual data, parameter initializations, scientific parameters,
methods and decisions. It changes optimization/audit query draw identities and
adds durable attempt accounting before any predictor execution. The amended
source required its own independent review and external freeze authentication.
The [original source review](../../experiments/research/synthetic-unlearning/source-review-original-freeze.json)
records the collision across 1,728 author/attribute identities and recommends
holding execution under that original freeze. These are source defects found
before execution, not failed predictor attempts or outcomes that were dropped.
The v2 source review also retained the original launcher and reviewed a corrected
launcher's failure receipts, exact ledger identities, cleanup, timeout handling
and refusal of a second invocation. Artificial process-exit and receipt fixtures
were source/accounting exercises; they did not train, infer or recover a
predictor. The later completed study had no failed or retried phase.

## Completed results and their independent units

The [saved results](../../artifacts/research/synthetic-unlearning-v2/2026-11-08/results.json)
contain 720 observed assignments and zero missing outcomes. The scientific
units are the six independently initialized and trained lineage seeds
**18401–18406**. Each lineage pairs its two author scales and three sequential
requests; methods, transforms, attributes and alternate queries are repeated
measurements within that lineage. The twelve scale cases are not twelve
independent lineage units.

The tables below copy the already saved final readback's equal-lineage means,
with displayed values rounded to six decimal places. Each lineage receives
equal weight after its paired scale/request cells are summarized. They are
descriptive summaries of these saved outcomes, not a new inference run or
population estimate. The public
[checker-run receipt](../../artifacts/research/synthetic-unlearning-v2/2026-11-08/verification/checker-run.sanitized.json)
publishes an authenticated projection of the saved scientific summary: exact
equal-lineage means and paired-cell ranges for every metric and criterion,
plus the binary64 uniform-edit lineage means. It binds the full saved summary
by its canonical hash and includes public source/artifact identities and
hash-only private process/readback provenance. These copied values require
no additional aggregation or inference.

| Binary64 arm | Direct correct answer | Audit accuracy | Audit true-code probability | Label-free extraction accuracy | Membership proxy advantage | Retained accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged | 1.000000 | 1.000000 | 0.998536 | 1.000000 | 0.727865 | 1.000000 |
| Canonical refusal | 0.000000 | 1.000000 | 0.998536 | 1.000000 | 0.727865 | 1.000000 |
| Retain fine-tuning | 0.930941 | 0.898791 | 0.887915 | 0.920910 | 0.710021 | 1.000000 |
| Uniform-target editing | 0.191551 | 0.161587 | 0.136090 | 0.171296 | 0.501302 | 0.999743 |
| Retain-only reference | 0.112269 | 0.113780 | 0.114161 | 0.114390 | −0.033613 | 1.000000 |

Full-exposure and reference fit gates pass throughout the saved summaries, so
these negatives are not explained by an unlearned full-exposure control or a
failed reference fit. The unchanged predictor learned the canonical codes and
kept them accessible. The canonical-refusal control uses those same weights;
its direct correct-answer rate is assigned zero by the authored deterministic
refusal rule. This is a simulated control, not a measured production serving
wrapper. Its underlying canonical probability remains recorded, and its
binary64 and binary32 alternate-query audit and extraction accuracy are one
in every paired cell. Answer suppression therefore supplies no evidence of
erasure in this control.

Retain fine-tuning preserves retained audit accuracy but leaves binary64 audit
accuracy at 0.898791 and extraction at 0.920910. Every saved paired cell fails
canonical suppression, audit accessibility, extraction and membership gates.
Uniform-target editing has much lower audit scores, but a lower score alone is
insufficient. Its binary64 descriptive pass fraction is 0.666667 for canonical
suppression, 0.888889 for audit accessibility and 0.750000 for extraction. Its
membership proxy gate fails in every paired cell. These fractions average
within the six lineages; they are not independent binomial trial counts.

| Binary64 arm | Retain-only refit: forgotten accuracy | One-label recovery: undisclosed-field accuracy | Archival extraction accuracy | Weight-only joint pass fraction | Archive-inclusive joint pass fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged | 0.958478 | 0.950939 | 1.000000 | 0.000000 | 0.000000 |
| Canonical refusal | 0.958478 | 0.950939 | 1.000000 | 0.000000 | 0.000000 |
| Retain fine-tuning | 0.863860 | 0.842507 | 1.000000 | 0.000000 | 0.000000 |
| Uniform-target editing | 0.355131 | 0.205097 | 1.000000 | 0.000000 | 0.000000 |
| Retain-only reference | 0.110420 | 0.110575 | 0.114390 | 1.000000 | 1.000000 |

The retain-only refit receives no forgotten answer labels. After its fixed
64-update budget, uniform-target editing's saved forgotten accuracy is
0.355131, compared with 0.110420 for the equivalently recovered reference.
That family exposes a recovery boundary even after ordinary audit accuracy
fell to 0.161587. One-label recovery deliberately receives one forgotten
attribute per author; its supplied answer is reintroduced information. The
table reports only the other three, undisclosed attributes for that family.
The raw results separately preserve all-forgotten, supplied-field and
undisclosed-field scores, along with post-recovery highest-confidence
extraction. The frozen joint recovery gates use the specified mean accuracies;
post-recovery extraction remains a separate diagnostic, rather than being
silently substituted into the decision.

The reference's joint pass fraction is one for both gates in every transform,
but it is a relative control. Its weights never had a full-exposure ancestor
in that reference's own lineage. Comparing the reference with itself allows
nonzero code accuracy and nonzero true-code probabilities; it is not an
absolute-zero accessibility test. Its binary64 extraction accuracy is
0.114390, and the vocabulary has only eight codes. Neither these scores nor
the reference pass prove that all information has disappeared.

All edited and full-exposure arms have archival extraction accuracy one under
every transform. For the editing methods, the exact inverse-program chains
reconstruct the original exposed tensors before extraction. The original
weights are directly available in unchanged/refusal controls. The reference
has no comparable full-exposure archive. Archive access is therefore a
different threat model: maintaining exact reversibility retains a route back
to the exposed predictor, and a weight-only edit cannot erase that archive.
The weight-only negative already holds without granting this extra access.

## Transform, request and retained-utility results

All four saved transform summaries give zero weight-only and zero
archive-inclusive joint pass fractions for retain fine-tuning and
uniform-target editing. The transform-specific results below preserve the
different accessibility, membership and utility observations.

| Editing arm | Transform | Audit accuracy | Extraction accuracy | Membership advantage | Retained accuracy | Retained harm against transformed base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Retain fine-tuning | Binary64 | 0.898791 | 0.920910 | 0.710021 | 1.000000 | 0.000000 |
| Retain fine-tuning | Binary32 | 0.898791 | 0.920910 | 0.710021 | 1.000000 | 0.000000 |
| Retain fine-tuning | Symmetric int8 simulation | 0.900350 | 0.922454 | 0.709828 | 1.000000 | 0.000000 |
| Retain fine-tuning | Symmetric int4 simulation | 0.871833 | 0.885610 | 0.704958 | 0.999633 | −0.000715 |
| Uniform-target editing | Binary64 | 0.161587 | 0.171296 | 0.501302 | 0.999743 | 0.000257 |
| Uniform-target editing | Binary32 | 0.161587 | 0.171296 | 0.501302 | 0.999743 | 0.000257 |
| Uniform-target editing | Symmetric int8 simulation | 0.162793 | 0.168981 | 0.485002 | 0.999688 | 0.000312 |
| Uniform-target editing | Symmetric int4 simulation | 0.136671 | 0.146219 | 0.331645 | 0.984570 | 0.014349 |

Negative harm means an accuracy improvement over the equivalently transformed
base in these saved cells. In particular, int4 retained harm is not computed
against the binary64 base. Binary64 and binary32 displayed accuracies agree in
these summaries, while their saved probabilities differ in low-order digits.
That observation does not establish equality of every numerical operation or
a real-to-native refinement theorem.

The int4 simulation lowers uniform editing's mean audit and membership scores,
but the membership gate's pass fraction is only 0.083333; its retained-utility
pass fraction is 0.888889. Overall mean retained harm of 0.014349 is below
0.05, yet the frozen gate applies to each assigned case/request and each
predetermined parity subgroup as well as the overall retained set. A favorable
overall average cannot remove failed utility assignments. Binary64,
binary32 and int8 uniform-edit retained-utility pass fractions are one. No
demographic or organizational fairness conclusion follows from numeric author
parity subgroups.

The transformed full-exposure/refusal control still has extraction accuracy
one in every transform. Its int4 audit mean is 0.999791 and retained accuracy
is 0.998919. Quantization therefore changes some finite audit/utility scores
without making those learned codes inaccessible. The int4 reference's
extraction mean is 0.107253 and retained accuracy is 0.999821; its relative
joint gates still pass. This does not make four-bit storage an erasure method.

The following binary64 lineage values preserve visible variation in the
uniform-edit result. Each row is already saved as the lineage's mean across
its two scales and three cumulative requests. No new lineage aggregation is
performed here.

| Lineage | Audit accuracy | Extraction accuracy | Membership advantage | Retain-only refit forgotten accuracy | Retained accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 18401 | 0.162905 | 0.128472 | 0.494502 | 0.328318 | 1.000000 |
| 18402 | 0.163387 | 0.170139 | 0.471644 | 0.323206 | 1.000000 |
| 18403 | 0.160494 | 0.157407 | 0.469039 | 0.331404 | 0.999476 |
| 18404 | 0.149691 | 0.199074 | 0.567419 | 0.350502 | 0.999669 |
| 18405 | 0.180459 | 0.208333 | 0.479745 | 0.370081 | 0.999807 |
| 18406 | 0.152585 | 0.164352 | 0.525463 | 0.427276 | 0.999504 |

All six lineages have zero edited-arm joint pass fractions under every
transform. The paired scales and repeated requests supply within-lineage
stress conditions, rather than an independently estimated causal effect of
model size or request number. The saved results retain each scale, stage,
method and transform cell; the descriptive summaries above do not establish
a monotone request trend, a scaling law or generalization to a new task family.

## What the source research motivates

The [primary-paper review](primary-paper-review.md) separates controlled exposure
from claims about erased information. TOFU's synthetic identities and retain-only
reference motivate a comparison with a model that never received the forget
data. MUSE motivates separate memorization, knowledge, privacy, utility, scale
and repeated-request lanes. Research on quantization recovery motivates measuring
access after declared deployment transformations. These design principles do
not make this a reproduction of those language-model benchmarks.

This pilot uses newly generated numeric author keys and four categorical
attributes per author. Each answer is one of eight arbitrary `CANARY` codes.
It does not implement TOFU's generative answer likelihood normalization,
truth-ratio or KS statistic, MUSE's source-defined text/privacy metrics, a
foundation-model edit, natural-language paraphrases or real private information.
Its low-entropy code vocabulary is a deliberate ceiling. An interpretation must
stay with these finite learned predictors and queries.

The practical question is relevant to an employee-directory deletion request:
can a predictor stop returning a requested profile while still answering other
profiles? Here the profiles, labels and keys are entirely synthetic, and no
enterprise integration or business effect is measured. A refused answer and an
inaccessible learned fact are different observations.

## Fresh exposure, references and repeated requests

Six predetermined lineage seeds each produce a 24-author and a 48-author case.
The two scales use different scale-specific synthetic inputs and initializations
but share the six lineage labels. Within a case, the predictor is a dense
two-layer ReLU MLP with 20 input coordinates, 96 hidden units and eight output
classes. The identity features are fixed random vectors; both dense layers and
both biases are learned. This is not an author-index lookup table with an
obvious isolated slot for every secret.

The full predictor is trained from scratch on one canonical query for every
author/attribute. The full and retain-only models use identical architecture,
initialization, optimizer, step count and learning rate. At each request the
retain-only reference restarts from that initialization and trains on only the
retained authors. It has no full-exposure training ancestor in its own reference
lineage. All newly created data and exposure identities are pinned; there is no
pretrained checkpoint, external provider or inherited public fictitious-author
dataset.

Three disjoint predetermined author batches each contain one eighth of the
case's authors. The cumulative requested fractions are 12.5%, 25% and 37.5%.
All four attributes of each requested author belong to the forget set. Editing
arms continue their own previous candidate at the next request. References
restart rather than inheriting forgotten training.

Thirty-two additional synthetic authors are never exposed during full or
reference training. Their identity vectors come from the same generator. Half
support the fixed known-answer attack calibration; a disjoint half supports its
nonmember audit. This makes a finite membership diagnostic possible, but does
not construct an optimal attacker or a differential-privacy guarantee.

## Controls and edit scope

| Arm | Learned weights and behavior | What the comparison diagnoses |
| --- | --- | --- |
| Unchanged | Original full-exposure predictor after every request | Whether the data were learned and remain accessible |
| Canonical refusal | Same original weights; a deterministic refusal on the canonical known forget query | Whether answer suppression can hide unchanged learned information |
| Retain fine-tuning | Continue original/current weights on retained noisy queries for a fixed budget | Whether ordinary fitting alone changes forget accessibility |
| Uniform-target editing | Fixed retained cross-entropy plus uniform-target forgotten cross-entropy; continue current weights | Whether the declared forgetting objective trades access against retained utility |
| Retain-only reference | Retrain from case initialization without any cumulative forget author | Behavior when these facts were absent from this controlled training lineage |

Both editing arms expose all four parameter tensors as writable. They use the
existing `TensorArtifact`, `difference`, `apply_program` and `rollback` helpers
from a separately copied, pinned three-file import closure. The exact before
and after cells, parent and candidate identities, and inverse program are
recorded. These operations check reversible tensor identity. They do not
certify unlearning, semantic utility or native execution.

The rollback scope is the four captured MLP tensors and their immutable
interface metadata. It excludes the data already disclosed, caches, random
generator state, public logs, external actions and an organization's archives.

## Audit separation and the declared attackers

Training uses canonical queries. The fit/edit objective uses a separately
seeded small-noise optimization stream. The v2 audit stream uses a distinct
seed role for identity scaling, noise and attribute noise. These are numeric query
perturbations of the same facts, not a new population of independent entities.
The seed and query constructor are public so another process can replay them.
Blinding here means the editor does not inspect audit outcomes or select a
candidate using them; it does not mean a physical host-isolation or access
control system hides the public benchmark data.

Every fixed candidate was saved before the complete selection seal was
externally authenticated. The audit required the exact authenticated digest.
There is one candidate per editing method/request and no audit-driven prompt,
threshold or hyperparameter search. The following attack family is declared
before any predictor training, and the completed audit used this fixed family:

| Endpoint or attack | Knowledge and budget | Interpretation |
| --- | --- | --- |
| Canonical direct answering | One canonical query per requested attribute; wrapper refusal recorded separately from underlying probability | Measures the displayed answer only |
| Label-free extraction | Architecture, author keys, vocabulary and deployed weights; 12 fixed alternate queries per requested attribute; select the highest-confidence query, tie by first index | Finite alternate-query accessibility without audit answer labels |
| Retain-only recovery fitting | Original binary64 candidate plus retained labeled data; 64 full-batch Adam updates | Tests whether fitting without forget answers restores access |
| One-label-per-author recovery fitting | Same retained data plus one disclosed forget attribute per author; 64 updates | Reports reintroduced known answers separately from all three undisclosed attributes |
| Archival rollback recovery | Editing arm's exact before-value inverse-program chain; restore original exposed tensors, then run the fixed extraction family | Tests the information retained by reversible archives |
| Known-answer membership diagnostic | A threshold calibrated on retained members and disjoint nonmembers; audit other forget attributes and independent nonmembers | A supervised loss diagnostic with explicit false-positive rate, not general privacy |

The labeled recovery attribute is fixed as author index modulo four. Successful
relearning of that supplied answer is not evidence that an undisclosed secret
survived. Accordingly the robustness criterion uses undisclosed attributes for
that family, while preserving both values in the raw results. The retain-only
recovery family receives no forgotten answer labels at all.

The recovery optimizer starts from each original binary64 candidate, and each
result is then evaluated under every transform. An attacker who first
quantizes and then refits, searches many optimizers, uses much more compute,
queries arbitrary representations or trains unrestricted internal probes is
outside this experiment. Those boundaries are part of the result, even if the
declared finite family fails to recover a code.

The archival attack has additional access compared with weight-only attacks.
The fresh retain-only reference has no full-exposure predecessor in its own
lineage. These knowledge differences are explicit. A public synthetic benchmark
also publishes enough data for external retraining; it cannot claim that a code
is private from someone who already has the original data.

## Transform and numerical scope

All learned parameters are stored as binary64 values. Evaluation includes:

1. Binary64 storage and NumPy execution.
2. Casting parameters and inputs to binary32 and executing binary32 operations.
3. Per-tensor symmetric eight-bit max-absolute quantization, ties-to-even rounding,
   clipping, dequantization and binary32 execution.
4. The same simulation with signed four-bit range `[-7, 7]`.

The exact scale, tie and zero-tensor rules are in the frozen protocol. This is
quantization/dequantization simulation followed by CPU NumPy matrix products;
it is not a vendor integer kernel. Native floating-point measurements remain
diagnostics, and no real-to-kernel refinement proof is claimed. Retained harms
are compared with the equivalently transformed original base; transform losses
against binary64 are also available by comparing the preserved transform cells.

## Frozen criteria and uncertainty

The full-exposure predictor and each stage's retain-only reference must reach
at least 90% canonical training accuracy. A failed fit gate stays in the
assigned denominator and prevents joint success. It cannot be filtered away
to make unlearning appear favorable.

The toy pilot tolerates at most 0.1 additional accessible code accuracy,
true-code probability or known-answer membership advantage relative to the
equivalently transformed retain-only reference. The retained audit accuracy
harm must stay at or below 0.05 overall and within each predetermined parity
subgroup relative to the equivalently transformed full-exposure base. These are
frozen toy falsification conventions: roughly one additional code per ten
queries and five percentage points of retained harm. They are not
stakeholder-approved enterprise limits, legal thresholds or a privacy standard.

Weight-only joint success requires both fit gates, canonical suppression,
retained utility, audit accuracy and true-code probability, label-free
extraction, the membership proxy and both recovery criteria to pass. A separate
archive-inclusive criterion also requires the archival recovery check. A
single passed suppression metric must never be labeled erasure. A failed
declared attack or utility check rejects its corresponding robustness claim.

All 720 method/request/transform audit assignments, 120 training assignments
and 288 fixed recovery attempts are bookkeeping denominators, not scientific
impact. Author/attribute/query rows, exported transforms, repeated requests and
replays do not enlarge the number of independently trained lineage units.
All controls, no-ops, gate failures, aborts and missing assignments are retained.
There are no outcome-driven retries.

Cell estimates and paired differences are descriptive. The frozen protocol
specifies 10,000 lineage-cluster bootstrap draws with seed 91818, sampling the
six lineage means within a cell rather than individual queries. The completed
result bundle and saved final scientific summary used here do not supply those
intervals; this report does not run a bootstrap or invent an interval. No
confirmatory multiplicity-adjusted win or equivalence conclusion is made.
The frozen protocol requires disclosure if a future reported interval is
degenerate because outcomes are identical. Its pre-execution illustration
states that even six perfect binary units would have only about 0.541 as the
two-sided exact 95% lower bound for success probability, below 0.75; that is
not an empirical interval for this report. A small resource-capped descriptive
estimate does not establish population confirmation.

## Verification and cost boundaries

The completed independent checker reuses the anchored query constructor. It independently
implements quantization, MLP inference, endpoint scoring, threshold calibration,
decision criteria and manual cell/rollback-chain replay. It checks externally
supplied source, selection and result digests. This is a stronger inference
and accounting check than simply reading the result file again. It does not
independently train the models, verify the Adam optimizer, prove privacy or
prove the absence of information in every internal representation.

The v2 query constructor initializes NumPy PCG64 from a `SeedSequence` tuple of
namespace, role, case data seed, author, attribute and variant. Optimization and
audit have different role coordinates. Source-only checks enumerate a safe
superset of the assigned query coordinates, compare the constructor with a
separate initialization, test repeatability and reject any repeated actual
initialized state. This supports observed state disjointness on that finite
domain; it does not prove mathematical or statistical independence.

V2 writes the complete fixed training/recovery assignment inventory once, then
flushes and fsyncs each attempt's start and terminal success or failure event.
After a producer exits, a start without a terminal event is interrupted; other
untouched assignments remain not started. An unterminated final event is ignored
conservatively and flagged while preserving earlier committed events. Artificial
source-only fixtures cover successful records, exception failure, subprocess
exit after a committed start and a torn final record without initializing or
running a predictor. Full verification independently checks assignment identities,
timestamps, event transitions, terminal statuses, checkpoint identities and
agreement with the terminal summaries bound by selection/evaluation manifests.
The completed ledgers record all 120 training and 288 recovery assignments as
terminal `ok`, with zero failed, interrupted or not-started assignments and no
torn final event. Their old terminal snapshot files agree with the independently
reconstructed durable records. No observed failure was omitted from a
complete-case analysis; the source-only artificial failure fixtures remain
history, not scientific failed attacks.

Training and audit have separately capped CPU budgets, one numeric-library
thread and zero provider calls. The launcher retains start, exit and failure
receipts, while the result bundle records per-training and recovery wall time.
This is not a full RSS, energy, GPU, throughput or enterprise cost benchmark.
If execution aborts, unexecuted assignments remain missing and no complete
result is issued; a new source amendment cannot overwrite the original attempt.

Durable successful attempt events report completed computation. Model/program,
recovery checkpoint and audit outcome files still buffer until their phase
finishes and may be absent after an abort. The ledger does not claim a checkpoint
survived merely because its computation completed. The local single-writer
flush/fsync fixtures also do not prove storage hardware power-loss durability.

## Saved verification, binding and measured cost

The [selection seal](../../artifacts/research/synthetic-unlearning-v2/2026-11-08/model-selection-seal.json)
records all 120 model identities, the model/program files and the training
ledger/snapshot identities, with `audit_outcomes_read: false`. It selects the
fixed candidates for experimental audit, rather than admitting them as useful
production edits. The
[evaluation manifest](../../artifacts/research/synthetic-unlearning-v2/2026-11-08/evaluation-manifest.json)
binds the selection, results, recovery checkpoints, durable recovery events and
terminal attack snapshot. The separate
[independent check](../../artifacts/research/synthetic-unlearning-v2/2026-11-08/independent-check.json)
records `verified: true` for its declared recomputation and accounting scope.

| Public evidence object | Raw SHA-256 |
| --- | --- |
| Original preparation freeze | `8cb44549858912813049f4df5ab0e57a4601cf0266aee1d48aeea08b9b42b099` |
| V2 preparation freeze | `aebfb931fe110f7e986f05039ba9686373d84eebdfe1ca6a57507aa94044eec8` |
| Model-selection seal | `d5de5ab12391d1e71dd02256bb76b51e80cfe346aa3fb8f7b47c979434998b0c` |
| Evaluation manifest | `bd9e8a6de462086e0132e1892a4c8e148af3496fe0a0ba05a17765b3a9da0aff` |
| Results | `8e8705e03d834806796131ad8b76194f714662e1d9edb3a01f7e2ce8ee5df4fb` |
| Independent check | `69e5c2124bfc07f60d9e361d8f1ef94528ea9d4b6de234960e89748c88080f42` |
| Sanitized checker-run receipt | `e7893e8502b23a22311d8d31a49644e62c01224d38f861d251fd9d0dce2a7acb` |

The independent-check whole-object canonical SHA-256 is
`282b608403a57872b0784b0510b594b105c06a46c1328e81cb9dceeae99bcac0`.
The sanitized receipt's whole-object canonical SHA-256 is
`e9598d8efe53d4ab8a688f2f2df9463492a674f5cabd2ceaf09c5cd5ed398c24`;
the full saved scientific summary it projects is canonically bound as
`b9e48526301b70e0a4efd1dfa4ebb616ce3d599975e0d52c56f10fa070b779cd`.
Raw hashes identify original file bytes. Canonical hashes identify the UTF-8
serialization of the entire parsed object with sorted keys, compact separators,
`ensure_ascii=False` and nonfinite numbers prohibited. These different byte
identities are not interchangeable, and neither digest by itself demonstrates
successful execution. The sanitized receipt is a published derivative of the
authenticated saved checker process/readback evidence. It records successful
host and wrapper exits, stdout/report correspondence, independently checked
accounting and preserved inventories, while exposing only hashes for private
provenance. It includes no private commands, absolute paths, environment dumps
or raw captures. A hash-only binding does not make a private receipt publicly
re-executable or by itself independently prove its execution.

The checker independently derives the schedule, parses durable ledgers,
implements transforms, inference, scoring, membership thresholds and criteria,
and manually replays tensor cells and inverse chains. It records **720**
recomputed outcomes, **408** checked checkpoint identities and **144** manual
rollback-chain steps. The 408 checkpoint inventory comprises saved training
and recovery checkpoints; the 144 replay steps repeat parts of the accumulated
editing chains. These are verification/bookkeeping counts, not 720 effect
replications, 408 independent models for population inference or 144 successes
at deleting information.

The [training ledger](../../artifacts/research/synthetic-unlearning-v2/2026-11-08/training-attempt-events.jsonl)
and [training snapshot](../../artifacts/research/synthetic-unlearning-v2/2026-11-08/training-attempts.json)
retain all 120 assignments. The
[recovery ledger](../../artifacts/research/synthetic-unlearning-v2/2026-11-08/recovery-attempt-events.jsonl)
and [attack snapshot](../../artifacts/research/synthetic-unlearning-v2/2026-11-08/attack-attempts.json)
retain all 288 fixed fits. Refusal and unchanged share recovery computations
because they have the same underlying weights; repeated control outcomes do
not become new fitting attempts. Ledgers, checkpoints and outcomes serve
different roles and their counts must not be added into a scientific sample
size.

| Saved phase measurement | Wall seconds | Frozen cap seconds |
| --- | ---: | ---: |
| One training phase, launcher measurement | 32.6078058 | 1800 |
| One evaluation/recovery phase, launcher measurement | 35.4687203 | 1800 |
| One checker phase, wrapper measurement | 26.5377355000237 | 300 |
| Checker host orchestration measurement | 26.74435930000618 | 300 |

The inner selection seal separately reports 31.07250909996219 seconds, and
the result's inner audit measurement is 32.56244750000769 seconds. Their
measurement boundaries differ from the launcher wall times, which include
additional process and publication work. The checker wrapper and host times
overlap and must not be added as independent compute costs. Each producer
phase ran once; the saved process/readback evidence records successful exits,
zero failures, zero timeouts, zero retries, zero provider calls and zero new
dependencies. Zero provider spend is specific to this local pilot and does
not establish zero total cost. Memory, energy, GPU operation, throughput,
acquisition, integration and enterprise lifecycle cost remain unmeasured.

This is limited checker independence. It reuses the anchored query constructor
and checks the already trained candidates; it does not independently retrain
the models or verify the Adam implementation. Schedule, byte, metric and exact
rollback checks do not prove privacy, unrestricted attack resistance, absence
of information in untested representations or real-to-native refinement.
Successful exact rollback supports the four-tensor structural claim while
also demonstrating the archival recovery route. There is no formal erasure
certificate for these learned runtime candidates.

## Literal issue #18 assessment

The original eight criteria are preserved in the
[acceptance map](issue-acceptance-status.md#issue-18). The report's following
assessment names each literal criterion and its present bounded evidence;
this documentation update changes no checkbox or issue state. Negative robustness
reporting can satisfy a criterion that explicitly permits rejection, while
missing deployment or source-specific evidence remains a gap.

| Original criterion | Completed evidence and assessment | Material limits |
| --- | --- | --- |
| **#18 c0:** Use TOFU-style synthetic identities/forget data and a retain-only trained reference; freeze MUSE-inspired memorization, knowledge, privacy, utility, scale and repeated-request endpoints. | **Partial beyond the finite pilot.** Fresh synthetic facts, six full-exposure lineages and matched fresh retain-only references were trained. Separate named access, membership-proxy, retained utility, paired scale and repeated-request endpoints were frozen and evaluated. | Numeric eight-code classification does not supply TOFU generative normalized likelihood, truth-ratio/KS or MUSE source-defined text/privacy metrics. It remains an inspired synthetic dense-MLP pilot; full source benchmark or language-model coverage is open. |
| **#18 c1:** Declare attacker knowledge, prompts/queries, extraction scoring, transform/quantization versions, relearning/recovery data and compute budgets. | **Supported in declared scope.** Frozen architecture/keys/weights, label restrictions, numeric query construction, confidence selection, four transform algorithms and fixed 64-update recovery families are explicit and exercised. | Finite alternate-query and fitting budgets do not cover unrestricted attackers, natural-language extraction, quantize-then-refit attacks, larger compute searches or stronger internal probes. |
| **#18 c2:** Keep blinded recovery/extraction attacks apart from edit optimization and selection; compare suppression/refusal, applicable fitting/editing controls and retain-only reference. | **Supported for logical selection separation.** V2 separates query roles; the complete fixed selection seal precedes audit, with no audit-outcome selection. All five controls/arms are evaluated and recovery labels are disclosed separately. | Original RNG overlap is retained as a pre-execution defect. Finite state disjointness does not prove stochastic independence. Public seeds/data are not hidden by host access controls, so logical blindness is not physical isolation. |
| **#18 c3:** Evaluate full-precision and deployed transforms, repeated edits and retained utility; retain attack failures, missingness and all attempts. | **Partial for deployment; completed within simulation scope.** Binary64/binary32 execution, declared int8/int4 simulation, three cumulative requests, overall/parity utility, 720 assigned outcomes and complete durable 120/288 attempt records are present with zero missing/failed/interrupted assignments. | Quantization/dequantization followed by CPU NumPy operations does not validate a vendor integer deployment kernel. Actual deployed-kernel behavior, broader hardware and resource measurements remain open. Planned bootstrap intervals are not supplied by the saved final summaries. |
| **#18 c4:** A passed answer-suppression metric cannot be reported as erasure; residual accessibility, utility and privacy endpoints remain separate. | **Supported.** Simulated refusal has direct accuracy zero while audit/extraction stay one; uniform editing's lower scores remain separate from recovery, membership, utility and archival failure. | No lower answer score or reference-relative pass is called absolute erasure. Membership is a supervised finite proxy, not a general privacy guarantee. |
| **#18 c5:** Forgetting survives the declared bounded attack/transform family at frozen criteria, or the robustness claim is rejected. | **Supported by a negative result.** Both edited methods have zero weight-only and archive-inclusive joint pass fractions in every transform. Their declared robustness claim is rejected rather than rescued by favorable component averages. | This rejects those fixed methods/budgets on the six synthetic lineages. It is not an impossibility theorem for all unlearning methods or a confirmed population effect. |
| **#18 c6:** Audit signals used in optimization are disclosed and independently stress-tested; a lower audit score alone is not robust forgetting. | **Supported within the tested finite family.** The original query-state defect is disclosed, the v2 change precedes execution, no audit outcomes select edits, and saved sealed candidates face fixed extraction/recovery/transform checks with independent metric/accounting recomputation. | The checker reuses query construction and does not reproduce training or validate its optimizer. Stronger learned probes, unrestricted recovery and independent task-family replication remain untested. |
| **#18 c7:** All public data are synthetic/public with pinned licenses; no live confidential-data experiment or universal erasure claim is implied. | **Supported.** The original/v2 input and source identities pin locally generated synthetic keys, facts, subgroups and codes with the declared MIT license. No imported confidential identities or live private-data experiment is used. | Public synthetic labels are available for external retraining. Future data additions need their own provenance/notices; no enterprise or universal deletion conclusion is supported. |

Issue #18 retains material open scope beyond this pilot, especially complete
source-defined language-model evaluation and actual deployed quantization
kernels. The completed negative result should replace the obsolete
source-only outcome status where applicable; it should not convert those
remaining requirements into completed evidence.

## What remains outside this pilot

This experiment does not promise deletion of arbitrary pretrained information,
real personal data or a production archive. It does not reproduce a generative
language-model benchmark or solve host isolation. It does not validate
unrestricted recovery, natural-language extraction, a stronger membership
attacker, high-entropy long secrets or resistance at production scale.

The observed bottlenecks now guide concrete follow-up work. First, evaluate a
separately frozen method against the same retain-only refit and known-answer
membership proxy, because uniform editing's low ordinary scores did not pass
those checks. Second, make archive deletion or archive access an explicit
system assumption and audit it separately: an exact inverse chain cannot be
both retained for rollback and treated as erased exposure. Third, evaluate
actual deployed kernels and subgroup utility before treating int4 simulation
as deployment evidence. Fourth, use independent trained lineages and a
separately frozen higher-entropy or generative source-defined task when
extending beyond categorical accessibility. These are future experiments,
not actions executed by this documentation update.

The present evidence supports complete bounded execution/accounting and an
honest negative forgetting report. Broader pretrained-data deletion, privacy,
production benefit, universal attack resistance and formal erasure remain
unsupported, and the corresponding broader issue requirements remain open.
