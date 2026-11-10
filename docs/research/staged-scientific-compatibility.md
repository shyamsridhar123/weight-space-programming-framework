# Prospective scientific compatibility for staged diagnosis

This proposes a scientific companion to the [seven-slot engineering endpoint contract](staged-diagnosis.md#study-specific-endpoint-accounting). It changes no engineering schema, endpoint identity or API. It is a prospective design only: no protocol freeze, execution, scientific admission or Issue #9 acceptance promotion follows from this document. The numerical choices below belong to this proposed benchmark; the weight-space programming framework remains general.

## Design and endpoint mapping

The proposed benchmark pairs bias-outlier and metadata-uniform diagnosis policies on eight independently seeded trained parents. Each parent supplies clean, -2, -4, +2 and +4 head-bias conditions. Both policies share the exact condition artifact and lineage holdout. There are 80 assigned policy-condition records: 32 fault and eight clean records per policy. These assignments and their endpoint slots remain paired within parent; the eight parents are the independent units.

Let F be an assignment's exact condition-parent artifact, with its frozen encoder and preprocessing. For a valid nonabstaining diagnosis at ordinary row j, let s=+1 for `bias_up` and s=-1 for `bias_down`. The common command adds -2s to `head.bias[j]`; the unit probe adds -s. Magnitudes are fixed independently of inspected values, confidence, planted magnitude or outcomes.

Preserve engineering order `main`, `parent`, `random`, `shuffled_1`, `shuffled_2`, `shuffled_3`, `opposite`. A separate scientific table has these eleven entries, in this order:

| Scientific entry | Required engineering mapping or additional evidence |
| --- | --- |
| `unchanged` | Engineering `parent`, authenticated as exact F. |
| `sham` | Separate genuine persist/reload of F with equal artifact identity. |
| `probe` | Separate magnitude-1 proposal from the frozen valid diagnosis, with lowering and native evidence. |
| `common_repair` | Actual selected `main` as attempted repair, plus separate scientific usability and operational resolution. |
| `random_diagnosis` | `random` only after equivalence to the prescribed independent uniform row/direction recipe. |
| `shuffled_1` | `shuffled_1`, using the first precommitted row derangement and matching evidence. |
| `shuffled_2` | `shuffled_2`, using the second derangement and matching evidence. |
| `shuffled_3` | `shuffled_3`, using the third derangement and matching evidence. |
| `opposite_direction` | `opposite`, with diagnosed row fixed, direction reversed and native matching verified. |
| `clean_reference` | Separate authenticated unmodified trained-clean parent; it may exist when F does not. |
| `inverse_oracle` | Separate evaluator-only inverse of the planted offset; on clean explicitly alias `clean_reference`. Float32 inversion need not recover clean bytes. |

For each lineage/condition, precommit one uniform ordinary row/direction random control independent of the metadata-policy guess, and three independently sampled uniform eight-row derangements; share them between policies. Repeated mapped rows are permitted and disclosed. Shuffles preserve diagnosed direction. No match failure permits redraw or retuning.

Active endpoints require genuine exact-parent construction/lowering evidence, both head tensors float32, `cast="round"`, exactly one eligible nonzero changed bias coordinate of prescribed sign, finite realized delta, unchanged native bits elsewhere and unchanged encoder/preprocessing. Compare each common-magnitude control against the actual usable main using `abs(norm_control-norm_main) <= 1e-5 + 1e-5*abs(norm_main)`. The probe is magnitude-1 and exempt from that norm comparison. References are not common-magnitude controls.

## Selection identity and operational fallback

Keep selected Candidate and artifact hashes, engineering slot, scientific usability, operational resolution/artifact and evidence references separate. Selected fields are null when no Candidate was selected. A denied seal's proposal remains attempted evidence. Equal hashes establish neither serialization nor randomization, ancestry or selection provenance.

The engineering `main` continues to retain an actual selection, including an unusable or native no-op selection. Scientific `common_repair` deploys that selection only when scientifically usable. Established ordinary scientific unusability or absent selection resolves operationally to exact F when F exists, preserving attempted/selected identities and failure reasons. Without F no fallback or numerical effect is invented.

Missing evaluation never triggers fallback. Unverified or lost provenance blocks admission. A usable main stays deployed if a required shuffle fails; that failure makes joint success zero and the matched numerical margin undefined. Random, opposite, probe, oracle and reference failures do not silently add joint-success criteria.

## Truth, usefulness and denominators

Each future holdout has 4,096 rows: 512 from each ordinary class 0 through 7. Fault masks come from sealed injection truth, never diagnosis. A negative planted bias at row j targets true class j (512 rows) and protects the other classes (3,584); a positive bias reverses these sets. They are disjoint complements. Clean cases have no fault target or matched margin.

Predictions use frozen native float32 encoder/head execution, fixed contiguous 256-row chunks and lowest-index argmax over all nine raw scores. Reserved output row 8 remains ineligible for intervention and unchanged; predicting it is incorrect for every ordinary label. Correctness is exactly predicted class equals label; a routing/queue metric cannot substitute.

For intact fault records define:

| Symbol | Definition |
| --- | --- |
| D | Indicator of a valid nonabstaining diagnosis with exact planted row and signed mechanism. Invalid/abstained diagnoses receive no correctness credit. Clean signed-fault diagnoses have D=0. |
| G | Target accuracy of the operational common endpoint R minus target accuracy of F. |
| H | Protected accuracy of F minus protected accuracy of R; positive values mean harm. |
| M | Main target gain minus the equally weighted mean target gain of all three valid matched shuffles. |
| U | Joint-success indicator requiring D=1, an active scientifically usable common repair, all three valid matched shuffles and every integer criterion below. |
| E | Parent lineages whose clean artifact and all four fault-condition artifacts exist with valid construction identities, committed before diagnosis. |

Let c_R,c_F be target correct counts; p_R,p_F protected correct counts; n_T,n_P their full denominators; and s1,s2,s3 the shuffled target correct counts. The fixed descriptive cutpoints G>=0.10, H<=0.01 and M>=0.10 are evaluated exactly:

```text
10*(c_R-c_F) >= n_T
100*(p_F-p_R) <= n_P
10*(3*c_R-s1-s2-s3) >= 3*n_T
```

Operational F fallback gives G=H=0 by identity, not fabricated counts or active-repair credit. On a correctly diagnosed negative fault (D=1), the common repair raises the true target row, while every derangement raises a different competing row. Under valid fixed-affine native execution, unchanged features/other coordinates, positive realized control deltas and deterministic ties, each shuffled target gain is at most zero. Thus M>=G, and G>=0.10 already implies M>=0.10. Retain the cutpoint for continuity, disclose its algebraic dependence, and treat any violation under a claimed valid tuple as a scoring/execution inconsistency. The positive complementary target has no such redundancy. Report signs and per-class changes separately; the seven-class positive target can conceal class-specific harm.

All 32 fault assignments per policy remain in U's denominator, including failed parents and terminal blocked/not-started descendants. Established training/construction failure, invalid/abstained diagnosis, D=0, failed common repair or failed required shuffle gives U=0. Otherwise missing evaluation can leave U unknown unless an observed criterion already excludes success. With K established successes and R_unresolved possible successes, report K/32 and `[K/32,(K+R_unresolved)/32]`. Lost expected truth/terminal evidence is an integrity failure, not an ordinary zero. Keep eight clean assignments per policy separately.

For policy-success coverages with completion intervals [L_bias,U_bias] and [L_metadata,U_metadata], report their contrast with at least the conservative interval `[L_bias-U_metadata,U_bias-L_metadata]`, tightening it only through authenticated shared constraints. The proposed 0.75 coverage benchmark is descriptive, not a powered or confirmatory success rule. Uniform metadata diagnosis has fault D=1 probability 1/16, so expected U is at most 1/16; this is not a hard ceiling on a realized 32-assignment sample.

Numerical policy effects use exactly E for both policies. For each eligible parent and policy, average the four fault-specific G values with weight 1/4. The primary numerical contrast averages the paired difference between those two policy means with equal parent weights. E empty means unavailable. Preserve all-assigned reporting for excluded parents; label the numerical estimand conditional on complete construction when E omits a parent. Do not average surviving conditions, silently renormalize an unresolved condition, or substitute a diagnostic control margin for this contrast. Report corresponding H, overall/per-class changes and individual parent records.

A nonexistent F has no defined numerical effect; an unmatched tuple has no M. These are undefined quantities, not missing effects to fill with arbitrary intervals. Publish each defined assignment-level M and its eligibility against all fault assignments. Form a four-condition parent M summary only when all four tuples for that parent/policy are eligible; an available-condition subset is not an unqualified parent summary.

For well-defined endpoints with unobserved outcomes, preserve full-denominator completion bounds and shared-baseline identities. With a known baseline target accuracy a_T, G lies in [-a_T,1-a_T]; otherwise its range is [-1,1]. With baseline protected accuracy a_P, H lies in [a_P-1,a_P]. For a shared F, the paired policy G contrast equals target accuracy of the bias endpoint minus target accuracy of the metadata endpoint, so it lies in [-1,1]. Likewise M equals main target accuracy minus mean shuffled target accuracy and lies in [-1,1]. The common F cancels in each expression; independently bounding its repeated occurrence would incorrectly inflate these ranges to [-2,2]. Authenticated whole-slot counts and presealed identity aliases may tighten the bounds. Incomplete chunks supply no completed observation or shorter holdout.

## Parent summaries and uncertainty

The eight independently seeded parent assignments are the independent units. Both policies, all conditions, signs, strengths, controls, references and holdout rows remain nested and paired within parent. Neither assignments, slots nor rows supply extra independent parent replicates. Report all eight parent records, all assignment/slot statuses and full denominators, including failed parents; retain clean intervention, abstention, fallback and overall/per-class harm summaries separately. This narrow synthetic signed-bias comparison of two specified policies and a common operator does not isolate a general causal effect of information access or establish natural-fault repair/general mechanism discovery. Correct signed-location diagnosis already supplies a favorable inverse direction; reducing a planted offset guarantees neither useful accuracy gain nor acceptable protected harm.

For an estimand whose eight prescribed parent metrics are all observed or mathematically identified, permit only the fixed descriptive 95% parent-bootstrap rule: 10,000 draws from its independent precommitted seed; sample eight parents with replacement; carry both policies and all within-parent observations together; take 2.5th/97.5th percentiles with the linear quantile convention. With an excluded parent or unresolved required parent metric, omit this interval and report records and bounds. Do not bootstrap rows or endpoint slots. With eight parents, this interval is a fragile descriptive summary; no confirmatory p-value, power claim, post-hoc subgroup success declaration or selected-threshold conclusion follows. Oracle/reference outcomes cannot establish a policy's primary success.

## Forecasts

Commit diagnosis-confidence and probe-forecast eligibility separately at the diagnosis barrier, independently of later endpoint/evaluation success. Diagnosis confidence q predicts D=1 with Brier loss `(q-D)^2`; later repair/evaluation failure does not erase a score whose truth is known. For a genuine forecast with well-defined but unobserved binary truth, sharp per-forecast Brier bounds are `min(q^2,(1-q)^2)` and `max(q^2,(1-q)^2)`. Missing truth/provenance remains an integrity failure rather than an invented event. Report genuine-forecast coverage and fault/clean composition separately from scoring; a selected nonabstaining subset is not the original assignment mixture.

The unit-probe event is exactly `Z=1[overall_correct(probe)-overall_correct(F) >= 41]` on the full 4,096 rows, a finite-sample overall gain of at least 0.01. Every genuine probe forecast is p=0.5, giving constant Brier loss 0.25 for an existing valid probe even when Z remains unobserved. It measures no predictive discrimination or population significance. An unmaterialized probe has no endpoint event or manufactured score. Report forecast, valid-endpoint and evaluation coverage, event rates, defined-event bounds and the single-bin reliability difference `event_rate-0.5`; do not fit a calibration curve or claim predictive skill from the constant Brier loss.

## Scientific seal and future observation

Before diagnosis, commit protocol/source identities, complete assignments, recipes, independent seed commitments and E. Before any repair, probe, diagnosis control or oracle construction, verify the cohort-wide diagnosis barrier with every assigned terminal and forecast eligibility. Ordinary failed/unavailable/not-started entries need truthful protocol-permitted terminal evidence; missing implementation alone cannot complete an entry.

The study-owned final scientific seal is separate from both the framework repair-selection barrier and `wsp-staged-study-endpoint-seal/1`. It must bind the complete ordered eleven-entry table; attempted, selected, operational, parent/reference and encoder/preprocessing snapshots; construction/matching/fallback states; all training/condition/stage terminals; genuine receipts and successful live-return snapshots; genuine framework diagnosis/repair barriers and seven-slot engineering evidence; truth and eligibility commitments; aliases; actual source/schema/protocol identities; and holdout domains/order, generator/seed, evaluator, analysis and resource rules. Additional evidence uses separately owned sibling storage, outside the engineering owner's exclusive evidence directory.

Verify the whole seal before holdout generation, registration or any holdout forward. Later actual row hashes attach to its frozen domains without changing endpoints or aliases. Missing/mutated required terminal, receipt or truth evidence blocks whole-study admission; a surviving subset cannot replace it. Surviving files do not reconstruct successful live returns or authorize reload/resume.

Prospective scoring uses one direct shared [DataRoleRegistry](../../rlcd/weight_space/data_roles.py) attempt for the complete aggregate of all eight lineage holdouts, including failed-parent lineages. Incomplete registration admits no subset. No holdout forward precedes registered callback entry; no retry is permitted. Full-denominator whole-slot count receipts must be persisted, reread and joined to the durable reservation after `score` returns or unwinds and all evaluation or evidence-mutating workers have stopped, before analysis; incomplete chunks are not completed observations. Failure/outcome-unknown lifecycle remains disclosed, even when completed slots survive.

All 880 proposed logical slots remain explicit, including failures and equal-hash entries. Presealed acyclic aliases save forwards only for identical endpoint, encoder/preprocessing, exact registered rows/order and prediction semantics. Each recipient needs its own completed-slot attribution and full denominator. Different truth masks require recipient counts derived from full per-class donor counts; a partial donor supplies no alias result. Aliases create no independent replicates.

## Isolated native whole-slot count contract

The isolated [`evaluate_complete_slot(..., *, expected_binding, execution)`](../../scripts/staged_scientific_evaluator.py) computes whole-slot counts under five independent expected SHA-256 bindings: endpoint, encoder, preprocessing, feature identity and rows. Before any forward, it captures and hashes the actual ordered bodies, including input values, labels and row order, preserving raw signed-zero representation. Identity preprocessing is exactly `{"schema":"wsp-staged-identity-preprocessing/1","input_width":N}`. The encoder has shape 32 by N and the head 9 by 32; reserved head row 8 has zero weights and bias -64.

Each slot contains exactly 4,096 plain row dictionaries with exactly the keys `inputs` and `label`. `inputs` is a list of N built-in Python `int` or `float` values exactly representable as binary32; `label` is a built-in Python `int` from 0 through 7, with exactly 512 rows per class. Sixteen contiguous 256-row chunks use native `encode_inputs` and `evaluate_head(native=True)`. Lowest-index argmax includes all nine raw scores; output 8 is incorrect for every permitted label. Full per-class and total denominators are returned only after the entire slot completes; an incomplete slot supplies no completed observation.

The helper requires the existing exact `ExecutionSpec` declaration `torch` / `float32` / `cpu` / `matmul` / `round` / `empirical`, a CPU-build Torch 2.8.0 runtime, actual matmul precision `highest`, explicit CPU execution and disabled autocast. It compares observable runtime state before and after execution. These checks supply no complete floating-point-state, flush-to-zero or bit-replay proof.

A separate isolated [receipt helper](../../scripts/staged_scientific_count_receipts.py) validates, exclusively creates, flushes and rereads the complete native-count payload using canonical finite JSON. `write_complete_slot_counts_once(path, result, *, expected_binding)` returns a fresh receipt containing the seven result fields and `sha256`, the canonical result-body digest. `read_complete_slot_counts(path, *, expected_binding, expected_receipt_sha256)` checks the exact stored bytes against all five independently retained bindings and the independently retained body digest, while enforcing the fixed execution declaration. The stored form is the canonical receipt plus one newline. Runtime fields remain recorded observations; rereading does not query or change the current numerical runtime. Missing, malformed, mutated or incomplete receipts fail without overwrite, recovery or reevaluation. These persistence functions establish local count persistence and reread correspondence only; they supply no execution attestation, scientific admission, registered observation, durable reservation join or power-loss durability guarantee.

A complete canonical file can remain after a flush, `fsync` or reread failure; a later valid read does not prove that the original writer returned successfully. Link/reparse checks precede opening and do not protect against concurrent filesystem replacement or establish independent storage.

The generic framework helper `read_count_reservation_correspondence(...)` strictly rereads the count receipt against independently retained bindings and body digest, then inspects one real registry with exactly one `get_attempt` call using the trusted attempt identity. It compares the exact reservation and canonical finite-JSON binding; this distinguishes boolean and integer values. A missing attempt is rejected; a registry inspection exception remains an exception. Neither permits retry. Independently retained raw-row and row-hash-list commitments must come from the same trusted producer for the same ordered rows. The raw-row digest binds the count receipt, while the hash-list digest binds the exact 4,096-row reservation slice; the helper cannot derive one commitment from the other.

Call this reader only after `score` returns or unwinds and all evaluation or evidence-mutating workers have stopped. That is a trusted harness precondition, not a fact established by registry inspection. Its fresh `wsp-staged-count-reservation-correspondence/1` result contains the validated count receipt, reservation identities, slice bounds and row commitments, and observed lifecycle status, reason, callback count and lifecycle/result digests. It preserves failed and outcome-unknown lifecycle with all four authority flags false; count claims are not equated with the callback result. The result's `sha256` commits its own body. It supplies no execution attestation, durable storage, scientific admission or proof of the original writer's success.

The [scoped verification](../../provenance/verification/staged-count-correspondence-20261110.json) records synthetic temporary-registry checks of this reader. That invocation ran neither the native evaluator nor the proposed scientific study.

The native evaluator owns no scientific admission, persistence, retry, registry or alias handling. Isolated persistence and read correspondence helpers exist; scientific ownership/API, the full scientific seal, aggregate registry integration and durable scientific observation joins remain unresolved. The draft's scientific identities remain `null`, the study remains unselected and its execution state remains `unexecuted`. Issue #9 c1 remains partial/open.

## Remaining prerequisites

A complete reviewed public study protocol must settle actual training/generator/policy/scorer sources and runtime authentication, independent seed commitments, genuine extra-endpoint/equivalence evidence, scientific schema/owner/API, scientific count-persistence integration and registry-observation joins, and monitored resource/stopping settings. The applicable historical-loading restriction in [the prospective protocol rubric](prospective-protocol-rubrics.md) remains binding; this decision supplies no bypass. Engineering receipt scopes and Issue #9 acceptance status remain unchanged.

## Draft admission package

The next step is to specify the evidence needed to admit this proposed study. [Issue #9 c1](issue-acceptance-status.md#9-c1) still lacks prospective evidence from separate diagnosis and repair selections; more engineering checks cannot change the historical joint selection. This outline remains unselected and unexecuted. Its field labels are proposed documentation fields, not an implemented schema. `null` means unresolved, not a study failure or an admission receipt.

| Proposed field | Required artifact or decision; current draft state |
| --- | --- |
| `state` | `study_selected=false`, `execution_state=unexecuted`; protocol freeze, scientific seal and registry-attempt identities are `null`. No scientific claim is established. |
| `source_pins` | Actual source/configuration identities for training, condition generation, both policies, common repair, native evaluator, analysis and scientific owner: `null`. Earlier compatibility pins and captured receipts do not pin a future execution. |
| `access_and_lineage` | Access manifest, permitted tools/data/computation, ancestry and split audit, evaluator-only truth boundaries and declared baseline comparisons: `null`. Unknown provenance remains unknown. |
| `execution_contract` | The proposed native float32, 256-row chunk and nine-score tie semantics above; actual runtime/evaluator identities and authentication basis: `null`. |
| `roster_and_seeds` | Proposed eight parents, five conditions and two policies. Actual ordered assignments, construction eligibility E, truth commitments and independent training/policy/control/bootstrap seed commitments: `null`. |
| `selection_evidence` | All preparation/diagnosis/repair terminals, genuine successful live-return snapshots, receipts and complete diagnosis/repair barriers: `null`. Retain attempted and selected identities separately. |
| `scientific_entries` | The ordered eleven-entry table above, all 880 proposed logical slots, operational/parent/reference and encoder/preprocessing identities, usability/matching/fallback evidence and presealed aliases: `null`. Additional endpoints need genuine construction evidence. |
| `observation` | Frozen holdout domains, generator/seeds/order, source/family identities, shared registry binding and full-denominator whole-slot count contract: `null`. Actual row hashes and durable attempt/count receipts do not yet exist. |
| `analysis_and_limits` | Actual scoring/analysis sources, margin justifications, missingness and conditional-E accounting, parent uncertainty, resources and stop rules: `null`. The proposed descriptive cutpoints above supply no powered success rule. |

Reuse the [selection admission](../../scripts/staged_diagnosis_selection.py), [cohort](../../rlcd/weight_space/staged_cohort.py) and [registry](../../rlcd/weight_space/data_roles.py) contracts within their existing scopes. `SelectionAdmission` rejects `protocol.evaluation_plan`; its seven-slot study seal is not already wired to the generic cohort evaluator. The [owned sequential runtime](../weight-space/sequential-runtime.md) supports its exact NumPy/float64/CPU, left-to-right, reject-cast, empirical declaration. It cannot silently replace this proposal's native float32 encoder/head execution. The [isolated native evaluator](#isolated-native-whole-slot-count-contract) supplies the count kernel, with separate persistence and read correspondence helpers; scientific ownership/API, the full scientific seal, aggregate registry integration and durable scientific observation joins remain unresolved.

The future admission decisions must occur in this order:

1. **Resolve and freeze the actual protocol.** Settle the table's sources, access, comparisons, native execution, scientific owner/API, margins and resource/stop rules before proposer calls. A content-bound local freeze is not external preregistration or execution attestation. Unresolved fields cannot be filled from study outcomes.
2. **Commit every assignment before diagnosis.** Bind the full roster, independent seeds, control recipes, evaluator-only truth, truthful preparation/construction terminals and E. Failed parents remain assigned; missing implementation or lost evidence cannot complete an unavailable entry.
3. **Verify the complete diagnosis barrier before interventions.** Freeze location, signed mechanism, predicted effect, uncertainty, abstention and genuine forecast eligibility before constructing any repair, probe, diagnosis control or oracle. Surviving files cannot recreate successful live returns or authorize reload/resume.
4. **Verify both endpoint evidence scopes before holdout generation.** After the repair barrier, retain the seven-slot engineering seal and separately bind the full eleven-entry scientific table, extra construction evidence, terminals, truth, aliases and operational resolutions. Keep extra evidence in sibling storage outside the engineering owner's exclusive directory. Missing/mutated required evidence blocks the whole study; missing evaluation never activates fallback.
5. **Authorize one complete registered observation.** Register the aggregate of all eight lineage holdouts, including failed-parent lineages, under one direct shared registry attempt. No subset admission, holdout forward before callback authorization or retry after consumed authorization is permitted.
6. **Verify persisted counts against the durable attempt.** Persist and reread whole-slot, full-denominator counts; join them to the durable reservation after `score` returns or unwinds and all evaluation or evidence-mutating workers have stopped, before analysis. Partial chunks and incomplete alias donors supply no completed observation. Preserve failure/outcome-unknown lifecycle and each logical slot's attribution.
7. **Analyze the frozen population, then review the evidence.** Keep all-assigned usefulness denominators, separate clean/fallback outcomes, E-conditional numerical effects, undefined quantities, unresolved-outcome bounds and parent-level uncertainty as specified above. Neither endpoint availability nor an oracle result supplies useful-success credit. Review new prospective evidence against the original criterion; this outline does not close c1.

Where an exact-score contract is requested, specify automatic rejection of infeasible/unresolved equations and nonexact stored residuals, with rejected attempts retained in usefulness accounting. The opt-in [`require_exact_scores=True`](../../rlcd/weight_space/features.py) guard covers supplied features only; it is not a general scientific admission gate. [Issue #6 c4](issue-acceptance-status.md#6-c4) remains partial. Do not silently replace this proposal's protected-harm allowance with exact-score retention or infer that exact-score infeasibility excludes every argmax-useful edit.

No protocol is frozen or study admitted here. Scientific selection, execution and evidence identities remain unset; original acceptance states, historical outcomes and earlier receipt scopes remain unchanged.
