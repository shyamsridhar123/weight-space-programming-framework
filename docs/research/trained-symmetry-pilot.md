# Trained symmetry and edit-transfer estimation pilot (#16)

The frozen estimation pilot completed all 12 pairs, 24 fresh trained lineages, 72 independent arms, and 48 known controls. It produced **0/72 useful admitted candidates** and **0/12 useful sampled heuristic transfers**. Only five Banking77 recipient fits met the frozen sampled-usefulness definition; every edited arm was rejected by integrated admission. These are negative results for useful admitted transfer under this protocol, with complete denominators and no assignment replacements.

The original fresh-process native replay failed before its first completed admission because the frozen NPZ loader retained NumPy string interface names. A separately authorized interface-name normalization derivative passed its identity assertions but failed the first admission-receipt equality comparison. Neither replay produced a success receipt. A separate literal-control reconstruction succeeded for all 48 controls, and a stdlib authentication/accounting audit passed its limited scope. Independent native admission and metric replay therefore remains an explicit gap.

Selection was sealed at `2026-11-09T04:39:58.957477+00:00`, before the first official test parse. Evaluation was recorded at `2026-11-09T04:43:33.653897+00:00`. These UTC timestamps fall on the evening of **November 8, 2026 in America/Chicago**. The [original evaluation](../../artifacts/research/trained-symmetry-pilot/evaluation.json) and [saved-artifact summary](../../artifacts/research/trained-symmetry-pilot/saved-artifact-summary.json) retain all assigned outcomes and exact stored intervals.

## Frozen scope and provenance

This is a resource-capped, 12-pair estimation pilot, with six disjoint donor–recipient initialization pairs per domain and 24 independently trained full lineages. It does not replace the proposed 24-pair confirmation study. All pairs share the public corpus and preprocessing within their domain; initialization pairs are the independent unit for the descriptive analysis, not independent datasets or task families.

The Banking77 variants are `Refund_not_showing_up`, `activate_my_card`, and `age_limit`, with two pairs per variant. The first name sorts first under the frozen case-sensitive Python alphabetic rule. `card_arrival` and `pending_cash_withdrawal` were excluded from variant selection only; the prepared original Banking77 corpus and original head retain all 77 classes. Optical Digits variants are classes 0, 1, and 2, with two pairs each. The task routes the target original class to an appended specialist output while retaining other original classes.

All models are dense affine/ReLU/affine classifiers with hidden width 64, trained from fresh initialization for 35 epochs using Adam, learning rate 0.015, batch size 256, weight decay 0.001, deterministic Torch algorithms, and four CPU threads. The earliest epoch achieving the highest development accuracy supplies the frozen encoder and original head. Consecutive donor/recipient seeds are 160101–160112 for Banking77 and 160201–160212 for Optical Digits.

Whole exact-duplicate groups receive fixed stratified roles: training 60%, development 10%, target 10%, protected 10%, and calibration remainder, using fresh salt `trained-symmetry-pilot-role/1`. Banking77 uses a 2,048-term vocabulary fitted on the training role only, log-count TF/IDF, and per-document L2 normalization. Digits divide the 64 pixel-block counts by 16. No official test CSV or ZIP test member is parsed during preparation, training, fitting, alignment, or selection. Authentication hashes the pinned raw bytes without interpreting their test observations.

The full `Network` assembly retains the original encoder, head, lineage seed, initialization and selected-parameter hashes, preprocessing binding, feature identity, semantic input ordering, and output ontology. Original native encoder/readout scores are checked against both full native Torch float32 execution and stored binary64 execution before any pair proceeds. Reparameterized native execution recomputes the changed encoder's hidden values.

The public raw acquisitions remain outside the export. The freeze pins their relative acquisition names, byte counts, and hashes. Banking77 is attributed to PolyAI's task-specific datasets under CC BY 4.0. Optical Recognition of Handwritten Digits is attributed to E. Alpaydin and C. Kaynak, UCI dataset 80, DOI `10.24432/C50P49`; the saved source page's CC BY 4.0 notice and metadata are pinned as well as the archive. The archive does not contain a separate license text.

| Binding | SHA-256 |
|---|---|
| Frozen study script | `118c741586369582e8bab39325c95f1d2fd7251544b8551ba30f4a9fb8d71564` |
| Protocol file | `f3187be850bb9d413fbce8951b8215865f9c10a982fe054da0b8c86afcd716d8` |
| Freeze manifest | `f5e067f5f61d2c59830e6c895d697b02106b0945e224635fc3cda39129cfe4c7` |
| Selection seal | `4641b45692eeb61f4564c6f1de5458e8bf408ec0aa07b9a2c2541dcb38276bae` |
| Evaluation file | `8a452a4a8ce3d61bd1203096d1985be8e7664c561f4359c8a58f93713cd3acf0` |
| Canonical evaluation digest | `a218166e568e691340312b50020adc62c83e241931fc1d146c1f7e2e67fba347` |
| Authenticated prior public selection gate, canonical digest | `601897dc212592ae76cb2eed4b56e120de6ed9d66fa292b7e42ba5b924a13d0f` |

The [protocol](../../experiments/research/trained-symmetry-pilot/protocol.json) and [freeze manifest](../../experiments/research/trained-symmetry-pilot/freeze.json) were externally anchored before training. Every subsequent phase executes the separately copied source snapshot. Original live/trained studies and their selections remain unchanged. Synthetic checks covered native gauge transport and raw sensitivity, stale-parent rejection, atomic rollback, realized random support/L2 matching, a full ReLU artifact, integrated admission, missing-seal rejection, and sealed-evidence tampering. The final frozen script also passed those synthetic checks in a separate process.

## Arms, controls, and gates

One fixed donor edit uses the existing `compile_arm` soft fit, ridge 0.01, retention weight 8, at most 32 target-role examples, and at most 128 non-target protected-role examples. Recipient fitting uses the identical fixed recipe on the recipient's recomputed features. There is no fitting grid, held-out feedback, method selection, failed-pair replacement, or retry of an original frozen study phase. Separate post-hoc verification attempts are accounted for below, including a timed-out literal-control invocation and its authorized second invocation.

| Assigned arm | Permitted construction | Interpretation |
|---|---|---|
| `none` | Unchanged recipient | Paired baseline; a no-op cannot be useful |
| `raw` | Add donor literal delta in recipient coordinates | Weight-only transfer without correspondence |
| `heuristic` | Existing normalized incoming-row Hungarian alignment, then additive transfer | Identified heuristic correspondence; parameter residual is diagnostic |
| `wrong` | Fixed one-position rotation of the computed assignment, with its scales | Wrong-correspondence control retained even if insensitive |
| `matched_random` | Assignment-seeded random support with per-tensor changed-coordinate count/L2 matched to the realized heuristic delta | Float32 support and norm mismatches are recorded |
| `fixed_target_fit` | One calibrated recipient fit on the same capped target/protected examples | More informed, fitting-permitted comparator |

Each pair also has four known same-function controls: identity, permutation, positive dyadic scaling in `{0.5, 1, 2}`, and combined permutation/scaling. Each action transforms incoming rows and bias, and inversely transforms connected outgoing columns. The known donor program is transported to its transformed parent; alignment recovery, raw coordinate copying, and wrong correspondence are evaluated separately. These are same-function controls, not an oracle correspondence between independently trained models. No finite intervention substructure was frozen, so these are edit-transfer tests and cannot supply a circuit or reasoning-module claim.

Native equality requires score agreement at absolute tolerance `1e-5` plus relative tolerance `1e-5`, identical lowest-index argmax decisions, and zero nonfinite values. Every maximum discrepancy, decision mismatch, and near-tie count is retained. Zero and duplicate incoming affine rows are diagnosed; the helper's deterministic assignment and unit-scale fallback are retained without reselection. Known positive ReLU actions have an algebraic interpretation on lifted exact parameters; the native checks cover only declared finite observations.

Every independent candidate receives a fresh integrated admission request bound to its recipient, literal program, ordered full-model interface, semantic adapter, task, real box `[0,1]^F`, one all-zero witness, per-output score-drift budget 1, deterministic argmax rule, and empirical Torch CPU float32 execution with endpoint casts. Structural limits permit only `w2` and `b2`, at most 1,024 edits, absolute parameters at most 256, and absolute changes at most 256.

The admission record separates binding, structural validity, exact box score bounds, exact witness score/decision observations, native execution/witness decisions, and sufficient witness margins. An upper bound exceeding the budget is a conservative rejection. A witness actually exceeding a score budget or changing the protected witness decision is an observed violation. The optional feature-retention lane is unsupported in this full-model request; sampled retained behavior is evaluated separately. A passed box score bound plus finite witness checks does not establish a whole-domain decision guarantee.

Two usefulness indicators are deliberately distinct. `useful_sampled` requires literal/native validity, a non-no-op candidate, target accuracy gain at least 0.05, protected accuracy loss at most 0.02, and finite held-out native scores. `useful_admitted` additionally requires accepted integrated admission. Neither indicator upgrades finite official-test retention to global retention.

## Complete outcomes and uncertainty

All 12 assigned pairs were sealed and evaluated: 24 fresh lineages, 72 independent arms, and 48 known controls. There were zero assignment failures, missing outcomes, replacements, or retries of the original frozen study phases. All-assigned worst/best mean bounds therefore collapse to the observed means. The saved evaluation contains every arm, including invalid candidates, no-ops, and rejected edits.

Accuracy changes below are fractions: 0.05 means five percentage points. Brackets contain the **original stored descriptive two-sided 95% percentile bootstrap intervals**, with 10,000 pair-level resamples and seed 161616. The tables round for display; the evaluation and saved summary retain the original numbers verbatim. No intervals were recomputed during documentation. Six pairs contribute to each domain/arm estimate; twelve contribute to each pooled arm estimate. Pooled results average initialization pairs on these two shared corpora and do not establish new-dataset generalization.

### Banking77

| Arm | Target gain [95% interval] | Protected accuracy loss [95% interval] | Literal/native valid | Useful sampled fraction [95% interval] | Useful admitted fraction [95% interval] |
|---|---|---|---|---|---|
| none | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 6/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| raw | 0.000000 [0.000000, 0.000000] | 0.007072 [0.001700, 0.014200] | 6/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| heuristic | 0.000000 [0.000000, 0.000000] | 0.000493 [0.000110, 0.000932] | 3/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| wrong | 0.000000 [0.000000, 0.000000] | 0.005592 [0.000439, 0.013816] | 2/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| matched_random | 0.000000 [0.000000, 0.000000] | 0.815132 [0.802357, 0.827906] | 4/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| fixed_target_fit | 0.337500 [0.225000, 0.450000] | 0.001645 [0.000274, 0.003893] | 5/6 | 0.833333 [0.500000, 1.000000] | 0.000000 [0.000000, 0.000000] |

### Optical Digits

| Arm | Target gain [95% interval] | Protected accuracy loss [95% interval] | Literal/native valid | Useful sampled fraction [95% interval] | Useful admitted fraction [95% interval] |
|---|---|---|---|---|---|
| none | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 6/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| raw | 0.357665 [0.085687, 0.666750] | 0.234193 [0.120908, 0.327987] | 1/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| heuristic | 0.245327 [0.015939, 0.540995] | 0.101738 [0.019398, 0.211742] | 0/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| wrong | 0.200068 [0.016566, 0.471048] | 0.144041 [0.046639, 0.244084] | 0/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| matched_random | 0.000000 [0.000000, 0.000000] | 0.823053 [0.757834, 0.880817] | 0/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| fixed_target_fit | 0.109453 [0.065093, 0.158160] | 0.000000 [0.000000, 0.000000] | 0/6 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |

### Pooled

| Arm | Target gain [95% interval] | Protected accuracy loss [95% interval] | Literal/native valid | Useful sampled fraction [95% interval] | Useful admitted fraction [95% interval] |
|---|---|---|---|---|---|
| none | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 12/12 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| raw | 0.178833 [0.020716, 0.377835] | 0.120633 [0.043358, 0.207140] | 7/12 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| heuristic | 0.122663 [0.003766, 0.304247] | 0.051116 [0.001416, 0.115309] | 3/12 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| wrong | 0.100034 [0.003745, 0.245746] | 0.074817 [0.017838, 0.143530] | 2/12 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| matched_random | 0.000000 [0.000000, 0.000000] | 0.819092 [0.786495, 0.849526] | 4/12 | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| fixed_target_fit | 0.223477 [0.140401, 0.319408] | 0.000822 [0.000055, 0.002029] | 5/12 | 0.416667 [0.166667, 0.666667] | 0.000000 [0.000000, 0.000000] |

Only the 12 unchanged recipients were admitted. All 60 edited candidates were rejected, giving 0/72 useful admitted outcomes. Frozen literal/native validity was 33/72. The five useful sampled outcomes were Banking77 recipient fits (5/6); useful sampled heuristic transfer was 0/12. The frozen sampled definition combines structural validity, development known-control gates, development candidate native/binary64 agreement, rollback, and held-out candidate native/binary64 agreement. **It does not additionally gate held-out known edited-control agreement** (`evaluate`, frozen script lines 506–508). Those control failures remain separate limitations; they are not retroactively removed or added to the frozen usefulness definition.

### Paired target-gain contrasts

| Contrast | Banking77 mean [95% interval] | Optical Digits mean [95% interval] | Pooled mean [95% interval] |
|---|---|---|---|
| heuristic_minus_raw | 0.000000 [0.000000, 0.000000] | -0.112339 [-0.489124, 0.211647] | -0.056169 [-0.253037, 0.101478] |
| heuristic_minus_none | 0.000000 [0.000000, 0.000000] | 0.245327 [0.015939, 0.538446] | 0.122663 [0.003766, 0.304247] |
| heuristic_minus_matched_random | 0.000000 [0.000000, 0.000000] | 0.245327 [0.015939, 0.540995] | 0.122663 [0.003766, 0.301318] |
| heuristic_minus_wrong | 0.000000 [0.000000, 0.000000] | 0.045259 [-0.395234, 0.466006] | 0.022629 [-0.193805, 0.237233] |
| heuristic_minus_fixed_target_fit | -0.337500 [-0.450000, -0.225000] | 0.135873 [-0.109512, 0.437054] | -0.100813 [-0.284795, 0.119949] |

The heuristic did not show a positive mean target-gain advantage over raw copying in either domain: Banking77 had zero gain throughout; Optical Digits had mean heuristic-minus-raw gain -0.112339 with a wide descriptive interval. The Optical Digits heuristic had positive average gain over no transfer and matched random edits, but that gain did not jointly satisfy protected retention and frozen validity. No heuristic pair cleared both metric thresholds. A positive target-gain interval alone cannot establish useful retained transfer.

The original confirmation proposal remains unchanged: 24 pairs/48 lineages, four variants per domain and three pairs per variant, one-sided aligned-minus-raw lower bound >0.02, aligned-minus-none lower bound >0.05, protected-loss upper bound <=0.02, observed subgroup loss <=0.05, useful coverage >=0.80 observed with lower bound >=0.60, and positive target advantage over random/wrong controls. This 12-pair pilot uses three variants per domain and two pairs per variant. Its descriptive two-sided intervals are not those one-sided decision tests. Together with failed held-out edited controls and incomplete replay, these results do not validate the proposed transfer claim or supply a confirmation-level rejection of all compatible transfer methods.

### Variant slices

Each row averages its two assigned pairs. These are post-hoc descriptive slices with no additional resampling or confirmation test. Every row has two observed and zero missing outcomes; all useful admitted counts are zero.

| Domain / variant | Arm | Mean target gain | Mean protected loss | Maximum subgroup loss | Literal/native valid | Useful sampled |
|---|---|---|---|---|---|---|
| banking77 / Refund_not_showing_up | none | 0.000000 | 0.000000 | 0.000000 | 2/2 | 0/2 |
| banking77 / Refund_not_showing_up | raw | 0.000000 | 0.000000 | 0.000000 | 2/2 | 0/2 |
| banking77 / Refund_not_showing_up | heuristic | 0.000000 | 0.000000 | 0.000000 | 2/2 | 0/2 |
| banking77 / Refund_not_showing_up | wrong | 0.000000 | 0.000000 | 0.000000 | 1/2 | 0/2 |
| banking77 / Refund_not_showing_up | matched_random | 0.000000 | 0.806743 | 1.000000 | 2/2 | 0/2 |
| banking77 / Refund_not_showing_up | fixed_target_fit | 0.275000 | 0.001316 | 0.100000 | 2/2 | 2/2 |
| banking77 / activate_my_card | none | 0.000000 | 0.000000 | 0.000000 | 2/2 | 0/2 |
| banking77 / activate_my_card | raw | 0.000000 | 0.005099 | 0.375000 | 2/2 | 0/2 |
| banking77 / activate_my_card | heuristic | 0.000000 | 0.000822 | 0.100000 | 0/2 | 0/2 |
| banking77 / activate_my_card | wrong | 0.000000 | 0.001316 | 0.125000 | 0/2 | 0/2 |
| banking77 / activate_my_card | matched_random | 0.000000 | 0.834375 | 1.000000 | 0/2 | 0/2 |
| banking77 / activate_my_card | fixed_target_fit | 0.200000 | 0.000000 | 0.000000 | 2/2 | 2/2 |
| banking77 / age_limit | none | 0.000000 | 0.000000 | 0.000000 | 2/2 | 0/2 |
| banking77 / age_limit | raw | 0.000000 | 0.016118 | 0.475000 | 2/2 | 0/2 |
| banking77 / age_limit | heuristic | 0.000000 | 0.000658 | 0.050000 | 1/2 | 0/2 |
| banking77 / age_limit | wrong | 0.000000 | 0.015461 | 0.950000 | 1/2 | 0/2 |
| banking77 / age_limit | matched_random | 0.000000 | 0.804276 | 1.000000 | 2/2 | 0/2 |
| banking77 / age_limit | fixed_target_fit | 0.537500 | 0.003618 | 0.400000 | 1/2 | 1/2 |
| optical-digits / 0 | none | 0.000000 | 0.000000 | 0.000000 | 2/2 | 0/2 |
| optical-digits / 0 | raw | 0.500000 | 0.247375 | 0.896175 | 0/2 | 0/2 |
| optical-digits / 0 | heuristic | 0.036517 | 0.174799 | 0.906077 | 0/2 | 0/2 |
| optical-digits / 0 | wrong | 0.011236 | 0.054663 | 0.858757 | 0/2 | 0/2 |
| optical-digits / 0 | matched_random | 0.000000 | 0.873070 | 0.989011 | 0/2 | 0/2 |
| optical-digits / 0 | fixed_target_fit | 0.174157 | 0.000000 | 0.000000 | 0/2 | 0/2 |
| optical-digits / 1 | none | 0.000000 | 0.000000 | 0.000000 | 2/2 | 0/2 |
| optical-digits / 1 | raw | 0.315934 | 0.186068 | 0.893855 | 1/2 | 0/2 |
| optical-digits / 1 | heuristic | 0.239011 | 0.054489 | 0.683333 | 0/2 | 0/2 |
| optical-digits / 1 | wrong | 0.450549 | 0.297214 | 0.893855 | 0/2 | 0/2 |
| optical-digits / 1 | matched_random | 0.000000 | 0.832817 | 1.000000 | 0/2 | 0/2 |
| optical-digits / 1 | fixed_target_fit | 0.041209 | 0.000000 | 0.000000 | 0/2 | 0/2 |
| optical-digits / 2 | none | 0.000000 | 0.000000 | 0.000000 | 2/2 | 0/2 |
| optical-digits / 2 | raw | 0.257062 | 0.269136 | 0.901099 | 0/2 | 0/2 |
| optical-digits / 2 | heuristic | 0.460452 | 0.075926 | 0.620879 | 0/2 | 0/2 |
| optical-digits / 2 | wrong | 0.138418 | 0.080247 | 0.670330 | 0/2 | 0/2 |
| optical-digits / 2 | matched_random | 0.000000 | 0.763272 | 0.989011 | 0/2 | 0/2 |
| optical-digits / 2 | fixed_target_fit | 0.112994 | 0.000000 | 0.000000 | 0/2 | 0/2 |

### Threshold margins and subgroup harm

Target-gain margin is gain minus 0.05; protected-loss headroom is 0.02 minus loss. A nonnegative value satisfies the corresponding metric threshold. Ranges below include every assigned pair, regardless of admission or validity. The [saved-artifact summary](../../artifacts/research/trained-symmetry-pilot/saved-artifact-summary.json), `all_assigned_usefulness_margins`, retains all 72 individual gains, harms, margins, validity flags, and sampled/admitted outcomes. The frozen evaluation retains every protected-label subgroup and row count.

| Domain | Arm | Target margin range | Protected headroom range | Both metric thresholds | Maximum subgroup loss |
|---|---|---|---|---|---|
| banking77 | none | [-0.050000, -0.050000] | [0.020000, 0.020000] | 0/6 | 0.000000 |
| banking77 | raw | [-0.050000, -0.050000] | [-0.001382, 0.020000] | 0/6 | 0.475000 |
| banking77 | heuristic | [-0.050000, -0.050000] | [0.018684, 0.020000] | 0/6 | 0.100000 |
| banking77 | wrong | [-0.050000, -0.050000] | [-0.005658, 0.020000] | 0/6 | 0.950000 |
| banking77 | matched_random | [-0.050000, -0.050000] | [-0.815526, -0.770789] | 0/6 | 1.000000 |
| banking77 | fixed_target_fit | [0.150000, 0.500000] | [0.013092, 0.020000] | 6/6 | 0.400000 |
| optical-digits | none | [-0.050000, -0.050000] | [0.020000, 0.020000] | 0/6 | 0.000000 |
| optical-digits | raw | [-0.050000, 0.950000] | [-0.349040, 0.016904] | 0/6 | 0.901099 |
| optical-digits | heuristic | [-0.050000, 0.848305] | [-0.328981, 0.020000] | 0/6 | 0.906077 |
| optical-digits | wrong | [-0.050000, 0.774176] | [-0.311269, 0.020000] | 0/6 | 0.893855 |
| optical-digits | matched_random | [-0.050000, -0.050000] | [-0.905880, -0.667037] | 0/6 | 1.000000 |
| optical-digits | fixed_target_fit | [-0.017033, 0.169101] | [0.020000, 0.020000] | 4/6 | 0.000000 |

All six Banking77 fits met the metric thresholds; one failed frozen native validity, leaving five useful sampled outcomes. Four of six Optical Digits fits met the metric thresholds, but all six failed held-out native/binary64 score agreement, leaving no useful sampled outcomes. This separates insufficient certification from observed gain/harm failure.

Pooled protected loss can hide label-specific harm. Banking77's worst heuristic subgroup is pair 03, original label 5, 40 rows, loss 0.10; its worst fit subgroup is pair 06, label 13, 40 rows, loss 0.40. Optical Digits' worst heuristic subgroup is pair 01, label 4, 181 rows, loss 0.9060773480662984. All exceed the proposed confirmation subgroup cap 0.05. The summary retains every maximum occurrence; no subgroup was dropped after selection.

## Native controls, correspondence, and admission

All 24 original full-model native bridges passed with maximum absolute score discrepancy zero. All 24 binary64 bridges passed, with maximum discrepancy 3.009126476882784e-6. Each aggregate bridge lane contains **one near-tie count**, zero decision mismatches, and zero nonfinite values. Passing these finite original bridges does not imply every edited native score will agree with the stored lane.

Development known parent, edited, and recovered comparisons passed 48/48 each, with zero decision mismatches, nonfinite values, and near ties. Development raw copying passed only the 12 identity controls and produced 215 Banking77 and 558 Optical Digits decision mismatches; wrong correspondence passed 0/48 and produced 1,104 and 2,246 mismatches respectively. These retained negative controls expose parameterization sensitivity. They do not identify a causal correspondence between independently trained networks.

### Official-test known controls

Every gauge/domain row contains six controls. Parent and recovered comparisons passed 48/48 each overall; recovered score discrepancy was zero. Edited controls passed only 36/48 overall. All essential parent/edited/recovered comparisons had zero decision mismatches, nonfinite values, and near ties, but score-tolerance failures still count as failed controls.

| Domain | Gauge | Parent pass | Edited pass | Edited maximum absolute score discrepancy | Recovered pass | Raw edited pass | Raw decision mismatches | Wrong edited pass | Wrong decision mismatches |
|---|---|---|---|---|---|---|---|---|---|
| banking77 | combined | 6/6 | 6/6 | 3.81469726562e-05 | 6/6 | 0/6 | 2266 | 0/6 | 8036 |
| banking77 | dyadic | 6/6 | 6/6 | 0 | 6/6 | 0/6 | 2639 | 0/6 | 6116 |
| banking77 | identity | 6/6 | 6/6 | 0 | 6/6 | 6/6 | 0 | 0/6 | 6126 |
| banking77 | permutation | 6/6 | 4/6 | 5.34057617188e-05 | 6/6 | 0/6 | 206 | 0/6 | 6274 |
| optical-digits | combined | 6/6 | 1/6 | 4.19616699219e-05 | 6/6 | 0/6 | 5031 | 0/6 | 8996 |
| optical-digits | dyadic | 6/6 | 6/6 | 0 | 6/6 | 0/6 | 765 | 0/6 | 7823 |
| optical-digits | identity | 6/6 | 6/6 | 0 | 6/6 | 6/6 | 0 | 0/6 | 7646 |
| optical-digits | permutation | 6/6 | 1/6 | 4.57763671875e-05 | 6/6 | 0/6 | 1694 | 0/6 | 7448 |

Held-out raw copying passed 12/48, the identity gauges only; wrong correspondence passed 0/48. The edited permutation controls failed 2/6 Banking77 and 5/6 Optical Digits score comparisons, and the Optical Digits combined controls failed 5/6. Maximum absolute discrepancy alone is not the whole criterion: the native gate uses elementwise absolute 1e-5 plus relative 1e-5, identical lowest-index argmax, and finite scores. These finite failures leave known edited-function native equivalence unresolved on the official test, despite passing decisions.

Independent candidate held-out native/binary64 agreement passed 41/72 (Banking77 30/36, Optical Digits 11/36). All 72 comparisons had zero decision mismatches, nonfinite values, and near ties. The domain/arm maximum discrepancies and pass counts are retained in the summary. Agreement of decisions does not erase failing score comparisons or confer global retention.

### Alignment and realized random matching

| Domain | Mean matching-cost range | Relative parameter-residual range | Maximum relative per-tensor random L2 mismatch |
|---|---|---|---|
| banking77 | [1.16014005549, 1.20584143092] | [0.649298941196, 3620963.37913] | 5.58290932998e-08 (banking77-pair-05, b2) |
| optical-digits | [0.646156337501, 0.736598014759] | [0.161160990659, 476.082821918] | 5.43097596482e-08 (optical-digits-pair-04, b2) |

All realized random per-tensor changed-coordinate counts matched the heuristic delta. The small recorded relative L2 mismatches remain in evidence after float32 casts. Across 24 lineages, exact duplicate and zero incoming affine-row counts were both zero; approximate ambiguity or unique causal correspondence was not certified. The very large Banking77 residual range reinforces that matching cost and parameter residual are diagnostics, not equivalence certificates.

### Integrated admission gates

Each row counts all 72 independent requests. A conservative rejection means a sufficient bound or margin did not establish the property; an observed violation means the checked candidate actually violated its declared condition. These statuses do not all represent sampled target or subgroup harm.

| Gate | Pass | Observed violation | Conservative rejection | Unsupported |
|---|---|---|---|---|
| binding | 72 | 0 | 0 | 0 |
| feature_retention | 0 | 0 | 0 | 72 |
| native_execution | 61 | 11 | 0 | 0 |
| native_witness_decision | 61 | 11 | 0 | 0 |
| native_witness_margin | 12 | 0 | 60 | 0 |
| real_box_drift | 12 | 0 | 60 | 0 |
| real_witness_decision | 61 | 11 | 0 | 0 |
| real_witness_margin | 12 | 0 | 60 | 0 |
| real_witness_scores | 13 | 59 | 0 | 0 |
| semantics | 72 | 0 | 0 | 0 |
| structural | 52 | 20 | 0 | 0 |

Real witness scores passed only 13/72 requests: the 12 no-ops and one edit. Thus 59/60 edits had an observed score-budget violation at the fixed all-zero witness. Box drift and both sufficient witness-margin gates conservatively rejected all 60 edits. Structural validity passed 52/72. Feature retention was explicitly unsupported in all 72 full-model requests; finite protected accuracy was reported separately. No whole-domain decision guarantee follows from these observations.

## Data roles, overlap, and work accounting

Whole exact groups were assigned before selection. Cross-role exact shared-group count was zero in both domains. The original Banking77 corpus has 10,003 training and 3,080 official-test rows; 25 exact groups/25 test rows are shared with the original training corpus. Optical Digits has 3,823 training and 1,797 official-test rows with no exact shared group. Full official tests were retained without post-seal exclusions. Near duplicates were not audited.

| Domain | Train role | Development | Target | Protected | Calibration |
|---|---|---|---|---|---|
| banking77 | 5966 | 971 | 968 | 968 | 1130 |
| optical-digits | 2291 | 377 | 377 | 377 | 401 |

The frozen environment is Python 3.12.10, NumPy 2.5.3, Torch 2.8.0+cpu, Windows 11, AMD64, 16 logical CPU cores, with four Torch threads. All donor and recipient training and both fixed fits are charged. Summed lineage training elapsed time was 74.63726010010578 seconds, with 13,860 training and 13,860 gradient/backward batches, 3,467,940 training forward rows, 566,160 development forward rows, and 165,912 selected-feature forward rows. Epoch histories and earliest-best development checkpoint selection remain in the sealed lineage records.

| Phase | Elapsed seconds | Full native calls / rows | Hidden native calls / rows | Binary64 calls / rows |
|---|---|---|---|---|
| Selection | 1299.7845393 | 576 / 73728 | 48 / 3624 | 96 / 12288 |
| Evaluation | 44.1077337 | 984 / 2399484 | 0 / 0 | 72 / 175572 |

The 542 sealed selected files occupy 416,169,359 bytes. `admission_seconds` times `admit` itself. Binding, literal realization, alignment, persistence, and gzip serialization contribute to selection elapsed time but are not all separately timed. Two endpoint witness rows per arm are declared for native admission; its internal exact/native arithmetic is not fully instrumented by the outer ledger. Artifact size measures stored bytes, not every tensor value or byte read.

Acquisition network time, peak memory, energy, hardware price, complete tensor-read/activation counts, human setup/editor effort, and provider/learned training work are unmeasured or absent. DeepAlign, NFN, DWS, and ScaleGMN did not run, so their costs are missing rather than zero. No efficiency, amortization, matched-quality superiority, or learned-method superiority claim is supported.

## Independent verification attempts and scope

The original frozen `replay` was invoked once with zero retries. It failed before the first completed admission with `ValueError: Expected finite JSON data, received str_`. The frozen NPZ loader (`run_trained_symmetry_pilot.py`, lines 81/84) preserves NumPy `str_` interface values; admission binding reaches the strict JSON normalizer (`contracts.py`, lines 305/476; `workflow.py`, lines 25/27/46), which accepts exact Python `str`. The [original failure receipt](../../artifacts/research/trained-symmetry-pilot/independent-replay-failure.json) retains the attempt, zero completed admissions/pairs, and external launch/stdout/stderr identities. `independent-replay.json` is absent. Frozen source bytes were not repaired or rerun.

A separately authorized [normalized-name wrapper](../../scripts/replay_trained_symmetry_normalized_names.py) was invoked once. Its initially authorized 300-second cap was raised to 1,800 seconds before launch; there was no 300-second invocation. Hidden supervisor PID 43408 launched at `2026-11-09T04:56:23.7623703Z`. The worker failed with exit 1 at `2026-11-09T04:56:50.910715+00:00`. Four loads passed assertions for identical tensor bytes/dtype/shape, endpoint hash, and ordered textual interface names while replacing only NumPy string values with Python strings. One derivative admission was recomputed, but the first canonical comparison, `banking77-pair-01/none`, failed with `ValueError: Fresh-process admission differs` at frozen line 503. No pair replay completed, and `posthoc-normalized-interface-replay.json` is absent. The retained [verification receipt](../../artifacts/research/trained-symmetry-pilot/posthoc-normalized-interface-verification.json), [progress](../../artifacts/research/trained-symmetry-pilot/posthoc-normalized-interface-progress.json), wrapper, launch record, and supervisor/worker logs preserve this failed derivative. **The recomputed admission fields were not retained, so the exact mismatch cause is unknown.** This documentation does not infer one, retry, or label the derivative a successful native replay.

The November 9 publication review removed only the two private local `File` paths from the normalized progress traceback. The public progress retains the failure message, line numbers, and every other JSON value. The [sanitation ledger](../../provenance/public-sanitization.json) binds the original 2,323-byte record (`22018b52d4812c9990e91b7b7c34f99ba9916c3a97e0fab685ef8894ea376638`) and the disclosed public derivative; the original bytes remain privately retained. Historical verification identities refer to that original, rather than claiming the derivative is byte-identical.

The separate unfrozen [literal-control reconstruction](../../scripts/replay_trained_symmetry_controls.py) had two explicitly retained attempts: the first timed out after 10 seconds (exit 124), produced no receipt, and left no worker; the authorized second 300-second invocation passed all 48 controls (reported wall time 85.2 seconds; verifier time 82.58543810003903 seconds). Its [success receipt](../../artifacts/research/trained-symmetry-pilot/supplemental-control-replay.json) and [attempt ledger](../../artifacts/research/trained-symmetry-pilot/supplemental-control-replay-attempts.json) verify saved literal program transport and reconstruction. They do not parse the raw test observations or recompute native admissions/metrics.

The original [stdlib auditor](../../scripts/audit_trained_symmetry_pilot.py) passed once and is preserved unchanged with its [receipt](../../artifacts/research/trained-symmetry-pilot/stdlib-audit.json). It authenticated the frozen sources, selected files, external anchors, canonical saved program/admission digests, assignments, complete denominators, saved arithmetic, original replay failure/logs, successful literal reconstruction and both attempts, plus a synthetic tamper fixture. Its explicit scope is authentication/accounting, with `native_replay_verified:false` and `completed_native_admission_replays:0`. It predates the normalized derivative and does not authenticate that later evidence or establish replay equality. A separate [supplemental documentation authentication](../../scripts/authenticate_trained_symmetry_documentation.py) and [receipt](../../artifacts/research/trained-symmetry-pilot/supplemental-documentation-authentication.json) authenticate the later evidence and final report/criterion map without replacing or rerunning the original audit.

## Assessment of original issue #16

The [portable criterion map](trained-symmetry-pilot-criterion-map.md) and its [JSON evidence map](../../artifacts/research/trained-symmetry-pilot/criterion-assessment.json) copy every original issue deliverable/acceptance criterion and every #16/shared design-checklist requirement with their original source hashes. They distinguish implemented pilot scope, negative results, missing work, and enforced claim boundaries. The historical design/checklist remain unchanged proposals; the pilot is not substituted for their confirmation cohort or gates.

Issue #16 remains **incomplete**. Fresh compatible ReLU lineages, all six baseline arms, known-gauge controls, recipient admissions, complete negative outcomes, and saved accounting were produced. Held-out known edited controls and both independent replay routes have unresolved native validation gaps. Learned representation/correspondence lanes are missing; unsupported activation/graph-family negative fixtures were not part of this pilot's frozen synthetic selfcheck. No finite connected substructure, removal/reinstatement/rescue test, or cross-architecture protocol was frozen. The report uses edit-transfer/correspondence vocabulary, makes no arbitrary tanh-scaling equivalence claim, and withholds circuit, reasoning-module, global/native-equivalence, and cross-architecture claims. Dependencies #2, #3, #9, and #11 require separate final-claim checks.
