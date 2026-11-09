# Exact signed bounds for trained affine heads

Research review: November 8, 2026. Roadmap issue: #13.

The signed certificate tightens the median maximum score bound to about half the existing triangle bound in both trained domains. It still accepts no frozen proposal under the original score caps. That is the measured calibration result: the bound improves, while this workload's admission bottleneck remains. Evaluation and separate-process replay now passed after the authenticated release; [the released result report](cumulative-and-bounds.md) separates fixed-proposal heldout utility from the unchanged zero score-cap admission coverage.

This work adds a narrow research semantic subset to the generic framework. Its statement concerns a declared affine readout and a declared learned-feature box. It does not establish a raw-input guarantee, a verified float32 kernel, a new native admission policy, or enterprise business impact.

## Why this extension was selected

The first profile was sealed before implementing the extension. It uses the complete frozen candidate pool from the [cumulative admission study](cumulative-admission.md), including each actual current parent, proposed candidate, order, controller, step, feature box, score cap, and calibration outcome. It never opens the study's heldout feature arrays.

That profile separates four questions:

1. Does the default trained semantic registry provide a certificate? The `trained-frozen-readout/1` adapter has no registered certificate callback.
2. Does an existing bound fit the fixed score cap? The existing sequential certificate, applied to the identical affine endpoints through a single identity layer, accepts none.
3. Does an exact residual on an observed calibration row already exceed the cap? Such a violation cannot be repaired by a tighter sound bound over a domain containing that row.
4. Does the proposal satisfy the separately measured target-gain requirement? Preserving scores does not itself create useful new behavior.

The selected extension is an exact signed feature-box score-drift certificate. Affine extrema on a coordinate box require no activation relaxation solver. This makes the proposed improvement small enough to check independently and compare against the unchanged baseline. A general encoder, attention, normalization, or nonlinear graph backend was not implemented by this study.

The pre-extension profile has 144 assigned step records and 36 distinct endpoint/domain pairs. These are audit denominators. Orders and controllers reuse lineages and sometimes endpoint pairs; neither a step count nor a pair count measures task impact or creates an independent sample.

## The fixed comparison

BANKING77 uses six complete training lineages, seeds 7001–7006, with 64 learned features and 78 output codes, including the reserved specialist code. Optical digits uses seeds 8001–8006, 32 learned features, and 11 output codes. The encoder in each lineage was trained before this adjacent study and is frozen. This is a study of edits to its readout, not an experiment with a pretrained foundation language model.

Forward and reverse request orders and local and cumulative controllers are retained exactly as sealed by the cumulative study. The proposer remains the existing soft editor with ridge 0.01, retention weight 8, and nullspace cutoff 1e-12. Failed and no-op proposals are part of the protocol and remain assigned, although none occurred in this particular trained comparison.

Each pair is **the actual parent to its proposed candidate**. Substituting the original ancestor for that parent would ask a different question and is rejected by replay. This comparison does not refit proposals or simulate a new sequence whose parents change after signed-bound admission.

For each recorded protected calibration feature matrix, the domain is its coordinate-wise minimum/maximum box. Both certificates use that same box and the same cap. The cap is half the original model's median winning score margin on the original protected calibration rows. It is a model-score feasibility limit, not an enterprise SLA or a cost metric, and it was not enlarged after rejection.

The box contains the protected calibration rows, but it also permits combinations of feature coordinates that may never be produced together by the encoder. This off-manifold coverage is part of the chosen mathematical domain. The study does not prove that all future inputs map into the box or that all box corners are reachable raw inputs.

## Certificate and elementary proof

For output `i` and a feature vector `h`, define the endpoint differences before any arithmetic rounding:

```text
delta_w_ij = candidate_weight_ij - parent_weight_ij
delta_b_i  = candidate_bias_i   - parent_bias_i

delta_y_i(h) = delta_b_i + sum_j(delta_w_ij * h_j)
```

Every finite stored endpoint and every declared box coordinate is lifted to its exact binary value, represented as a rational number. In the supported trained subset, the endpoints must be exactly representable as float32 values. The equation above is a real affine equation on those stored values; it is not a description of a float32 reduction's intermediate rounding.

For a box with `lo_j <= h_j <= hi_j`, the signed extrema are:

```text
L_i = delta_b_i + sum_j min(delta_w_ij * lo_j, delta_w_ij * hi_j)
U_i = delta_b_i + sum_j max(delta_w_ij * lo_j, delta_w_ij * hi_j)
B_i = max(abs(L_i), abs(U_i))
B   = max_i B_i
```

The proof follows coordinate by coordinate. If a coefficient is positive, its term is smallest at the lower coordinate and largest at the upper coordinate. A negative coefficient reverses those choices; a zero coefficient contributes nothing. Adding the term inequalities and the unchanged bias difference gives `L_i <= delta_y_i(h) <= U_i` for every `h` in the declared box.

The receipt supplies two corner bitstrings for each output. Their signs select minimizing and maximizing coordinates. Evaluating the endpoint difference at those corners attains `L_i` and `U_i`, so this bound is exact for the real affine equation over this box. Different outputs can use different corners. No single shared corner is required: the output whose bound is largest provides a corner attaining the maximum absolute output-coordinate drift.

The unchanged triangle envelope is:

```text
T_i = abs(delta_b_i)
    + sum_j abs(delta_w_ij) * max(abs(lo_j), abs(hi_j))
```

Every signed term lies inside the corresponding triangle term interval, so `[L_i, U_i]` lies inside `[-T_i, T_i]` and `B_i <= T_i`. Replay checks this relation for every supported output. Tightening this real score envelope says nothing by itself about target accuracy or business outcomes.

A constructed cancellation example makes the difference concrete. If a score changes by `h_1 - h_2 + 2`, with `h_1` in `[2,4]` and `h_2` in `[3,5]`, its signed interval is `[-1,3]` and its maximum absolute change is 3. The triangle envelope is `2 + 4 + 5 = 11`. These are score bounds on an artificial mathematical example, not task success statistics.

## Supported semantics and independent checking

The implementation is [bound_research.py](../../rlcd/weight_space/bound_research.py). The subset requires the known `trained-frozen-readout/1` architecture, exactly the declared head weight and bias tensors, and the complete expected metadata layout. It checks tensor dimensions, output names, feature order, the declared float32 lane, feature/encoder/preprocessing SHA256s, recomputed feature identity, and the fixed encoder and head semantic declarations.

Unknown graph operators, captured normalization state, stochastic semantic declarations, undeclared tensors, a changed feature order or identity, a different native dtype, and endpoints that would change when cast into the declared float32 lane are unsupported. A matrix shaped like an affine head is insufficient to authorize this subset. The metadata identifies the assumed encoder and preprocessing; it does not prove their implementation or their relationship to arbitrary raw inputs.

The generator constructs the signed intervals from endpoint differences. The independent checker does not call that generator or its endpoint-delta helper. It checks the exact receipt schema and canonical hash, endpoint and box identities, canonical rational encodings, corner bitstrings and sign conditions, direct rational evaluation of each attaining corner, interval ordering, absolute bounds, and the reported maximum.

The certificate schema is `wsp-trained-affine-box-drift/1`. The budget checker first verifies the receipt and then compares its exact maximum against the exact binary64 value of a finite, nonnegative supplied cap. Rehashed mutations to a bound, corner, identity, field set, trust flag, rational encoding, or omitted interval fail checking.

This is an independent Python checker, not a machine-checked proof of the Python implementation. The existing Lean bridge establishes a different symmetric/l1 inequality; it does not verify this new signed certificate, its generator, this checker, or native execution. The new receipt explicitly sets all four claims to false:

```text
implementation_machine_checked = false
feature_extractor_verified      = false
floating_execution_verified    = false
core_native_admission_integrated = false
```

The opt-in research registry uses the signed certificate under version `signed-affine-box/1`. The default/core trained registry and native admission controller are unchanged. Registry availability and the separately invoked independent checker are distinct interfaces.

## Measured calibration comparison

All baseline unsupported outcomes become supported in the explicitly declared research affine subset. This means checked real certificates are available for those pairs. It does not mean that those certificates pass their score caps or that a core native controller has begun accepting them.

| Calibration accounting | BANKING77 | Optical digits |
|---|---:|---:|
| Assigned step records | 96 | 48 |
| Distinct endpoint/domain pairs | 24 | 12 |
| Default registry unsupported records | 96 | 48 |
| Research signed subset supported records | 96 | 48 |
| Triangle score-cap passes | 0 | 0 |
| Signed score-cap passes | 0 | 0 |
| Observed exact calibration score violations | 72 | 24 |
| Rejections with calibration residual within cap | 24 | 24 |
| Calibration target-gain passes | 52 | 12 |
| Calibration target-gain failures | 44 | 36 |

The categories and gains in this table refer to frozen calibration records. They are not heldout utility, enterprise traffic, or independent population success rates. Every row from the original profile is accounted for, including reused pair measurements.

The median signed/triangle maximum-bound ratio across distinct pairs is **0.5014** for BANKING77 and **0.5045** for digits. The corresponding median ratio of total signed interval widths to total triangle interval widths is **0.4952** and **0.4943**. Ratios exclude zero denominators; the exact rational per-output bounds and ratios remain in `comparison.json`.

No score-cap category changes. For the signed certificate, the label `conservative_rejection` is relative to the finite observed calibration sample. It does **not** mean the signed affine box extrema are loose. Since those extrema are attained, a signed maximum above cap means some declared box corner exceeds the cap. A different scalar bound calculation on that same affine equation and box cannot remove that violation.

The timing comparison measures one run per distinct pair, in the fixed order triangle generation, signed generation, and independent signed checking. Each timed operation enables `tracemalloc`; duplicates reuse the recorded measurements.

| Median time per distinct pair, seconds | BANKING77 | Optical digits |
|---|---:|---:|
| Triangle generation | 0.3200 | 0.02510 |
| Signed generation | 0.2026 | 0.01753 |
| Independent signed check | 0.4081 | 0.03228 |

| Median traced Python allocation peak, KiB | BANKING77 | Optical digits |
|---|---:|---:|
| Triangle generation | 483.1 | 42.8 |
| Signed generation | 398.4 | 48.9 |
| Independent signed check | 372.3 | 31.8 |

The signed generator is cheaper in these recorded runs; generating and independently checking its receipt adds a separate cost. This is not a repeated performance benchmark or a claim about total deployment latency. Allocation peaks cover traced Python allocations, not process RSS, native library or kernel memory, GPU memory, or peak resource use of an enterprise service. Replay checks the measurement scope and types and retains the original observations; elapsed times and allocation peaks are not expected to replay exactly.

## Heldout design and release boundary

The study opened no heldout feature array or utility outcome before sealing its execution source. The evaluation code was implemented and exercised only on synthetic fixtures before that boundary. The authenticated release now permits the completed secondary evaluation documented in [the combined results](cumulative-and-bounds.md); the frozen design below remains unchanged.

After an explicit coordinated release, evaluation will compare each frozen proposed candidate with its actual parent. It uses the previously acquired novel official-test rows as the primary population and the complete official test as secondary sensitivity evidence. These rows were already used by the earlier study, so this is an adjacent secondary analysis rather than fresh population confirmation. There is no outcome-based candidate reselection, domain shrinking, cap adjustment, or new sequence deployment.

Each pair separates features inside and outside its unchanged calibration box. For in-box rows, an independent exact observer lifts stored features and endpoints into integers with a common power-of-two denominator, subtracts the endpoints directly, and evaluates the affine residual without the signed generator. It records exact per-output maxima and rows exceeding a signed per-output bound. An out-of-box row lacks the premise; its observed behavior is never called a real certificate violation.

Native execution separately uses Torch CPU float32 affine scores, deterministic lowest-index argmax, and the frozen output-code-to-queue map. Its reported score changes are binary64 differences of native outputs. Native exceedance counts compare those values against binary64 projections of the real bounds and cap. They remain diagnostics even for in-box rows; the real certificate does not establish the native reduction's rounding behavior. Inputs must correspond exactly to the declared stored float32 feature coordinates.

Task utility compares candidate and parent on the requested target and protected scheduled-task complement. It requires target queue gain at least 0.10, protected local queue flips at most 0.01, and protected local net harm at most 0.01. The evaluator separately reports utility pass, signed-bound availability together with utility pass, and triangle-bound availability together with utility pass. These intersections describe frozen proposals; they are not new admitted deployments.

Missing files, invalid evaluation correspondence, execution failures, absent target/protected strata, failures, and no-ops retain their assigned records. Descriptive means average repeated steps, orders, and controllers inside each complete training lineage. Student-t intervals use the six lineage means per domain, conditional on the shared fixed requests. Missing gain and net harm use zero-filled descriptive estimates with sensitivity `[-1,1]`; missing flip uses `[0,1]`; missing usefulness is failure-filled with sensitivity `[0,1]`. Per-lineage bounds expose missingness rather than dropping unsuccessful attempts.

Both `evaluate` and full `verify` require an externally supplied candidate-manifest hash and the externally released canonical global-gate hash. Before opening evaluation inputs, the checker authenticates the fixed global gate, its preparation lock by byte and canonical identities, 224 unique locked assignments, complete terminal accounting, and every bound terminal selection's canonical hash and status. It checks the selection-complete, frozen, pre-evaluation flags. Terminal failure, abstention, invalid, timeout, and interrupted records remain legitimate accounted outcomes.

This checker authenticates the root study's release binding without importing its evolving live harness. It does not independently audit that harness, prove the absence of every possible side channel, or substitute for its full preparation/selection validation. Internal JSON hashes are correspondence checks; an externally retained identity is needed to authenticate a ledger against substitution. The CLI has no ungated `run` action.

## Reviewed primary sources and implications

The exact reviewed PDFs and their limits are recorded in [the source register](../weight-space/v2/research-sources-2026-11.json) and copied into the immutable `reviewed-sources.json`. The selected versions were read before the extension. Their implementations were not built or audited by this review.

| Primary source | Reviewed version and PDF pages | Implication for this implementation |
|---|---|---|
| *TorchLean: Formalizing Neural Networks in Lean* | `2602.22631v2`; pages 6, 8, 19, 54, 55 | Shared graph semantics and restricted finite Float32 refinement motivate explicit execution/proof subsets. Serializer, compiler, target backend, hardware, nonfinite values, and unsupported operators remain visible boundaries. This repository has not adopted that implementation or obtained its runtime guarantees. |
| *Efficient Neural Network Robustness Certification with General Activation Functions* (CROWN) | `1811.00866v1`; pages 1, 3 | Supported activation envelopes can give conservative domain bounds for nonlinear networks. Loose sufficient rejection differs from an observed violation. This affine-head box has exact extrema and needs no relaxation solver. |
| *Provable Repair of Deep Neural Networks* | `2104.04413v2`; pages 2, 4 | One-layer LP repair tractability depends on the paper's decoupled activation/value architecture. It does not establish arbitrary ordinary-network repair or certify the present editor's task usefulness. |

| Reviewed PDF | SHA256 |
|---|---|
| TorchLean | `dd8b7ce74b70d6a5bd928bf32f5caaeaf9354b56902d8f9a47f6f14f77c0c4d3` |
| CROWN | `7b57198bb3693c7a32c1000f2585b7772b5e76c6954614d8d3392c7a1a32ad32` |
| Provable Repair | `948b259ed6e359ad7926182ebce4f62a5a3c0c3c431f660687460abbdf8d2302` |

The mathematical signed-box proof above is elementary and specific to the affine subset. Reading these papers does not convert it into a machine-checked implementation or a deployment guarantee.

## Evidence and reproducible commands

The pre-extension profile is in `artifacts/research/affine-bounds/2026-11-08-profile`. The signed comparison is in `artifacts/research/affine-bounds/2026-11-08`.

| Frozen object | SHA256 |
|---|---|
| Pre-extension profile manifest | `45212f1ee628622530ed36245b246ced6fc7d7bd23436721dd9fd48c8e77548f` |
| Cumulative candidate manifest | `208802510de6675f1ae09efcac54932c4dae9651df5d9d08f395e1b364b1309d` |
| Signed comparison manifest | `a125163265d1ab8afd4592d06c2d30b5391a65655cff091e1eee5bfc486a284c` |
| Signed protocol | `4cef0362cae7a521666d332e895df1e2d4daf9020e53848263615bc2f0919dd9` |
| Signed selection seal | `24c217ef03ed505d29d89de5b5c8f71169727d64619d70e893f059906bcf1d1e` |
| Signed calibration comparison | `5395243bcc6ae000b43504c4866b438501d233d6903b83c003f9e02fe7881d36` |

The signed manifest identifies the actual project source closure, source inventory, frozen protocol, reviewed sources, distinct paired receipts, complete calibration comparison, and selection seal. It includes the unchanged cumulative import closure and the new research module. Source and protocol snapshots are written before generating the calibration comparison. Changing the active execution source, frozen cumulative inputs, a proposed pair, assigned outcome, box, cap, category, certificate, or deterministic comparison field fails replay.

```powershell
python -m rlcd.weight_space.bound_research verify-seal artifacts/research/affine-bounds/2026-11-08 --manifest-sha256 a125163265d1ab8afd4592d06c2d30b5391a65655cff091e1eee5bfc486a284c
python -m pytest tests/test_research_affine_bounds.py -q
```

`verify-seal` passed on this actual calibration evidence and does not load evaluation features. The affine, cumulative, and admission-runtime regressions also passed. The checks include exhaustive artificial corners, exact cap boundaries, cancellation, independent generation-disabled replay, rehashed receipt mutations, semantic subset rejection, endpoint subtraction, measurement scope, complete release accounting, duplicate retention, missingness, out-of-box interpretation, exclusive staging, and rehashed outcome falsifiers on synthetic data. Their passing status is engineering evidence, not an estimate of trained task impact.

After the root study explicitly releases its canonical gate identity, the evaluation commands are:

```powershell
$releaseHash = "601897dc212592ae76cb2eed4b56e120de6ed9d66fa292b7e42ba5b924a13d0f"
python -m rlcd.weight_space.bound_research evaluate artifacts/research/affine-bounds/2026-11-08 --manifest-sha256 a125163265d1ab8afd4592d06c2d30b5391a65655cff091e1eee5bfc486a284c --global-gate artifacts/research/live-studies/global-selection-seal.json --global-gate-sha256 $releaseHash
python -m rlcd.weight_space.bound_research verify artifacts/research/affine-bounds/2026-11-08 --manifest-sha256 a125163265d1ab8afd4592d06c2d30b5391a65655cff091e1eee5bfc486a284c --global-gate artifacts/research/live-studies/global-selection-seal.json --global-gate-sha256 $releaseHash
```

The global-gate argument is its canonical identity supplied outside the gate itself; the candidate-manifest argument is its recorded file hash. `evaluate` appends exclusive `results.json` and `evaluation-manifest.json` after checking both boundaries. It never rewrites the candidate manifest. `verify` checks the result identity and independently replays the real/native observations, utility, complete assigned accounting, and lineage summaries. Both commands completed successfully on the actual heldout features after the authenticated release. The replay uses a new process with the same reviewed implementation; it is not an independent formal implementation.

## Decisions for the next study

The exact box extension should be retained as a supported real semantic subset and an independent receipt checker. It removes an interface gap and measures a sizeable envelope reduction, while providing no new score-cap admission in this frozen workload.

The observed calibration violations require a smaller or better constrained proposal if those same calibration rows and caps remain obligations. The other rejected signed pairs require either a changed proposal or a separately justified domain that excludes the offending box combinations while still covering the intended feature population. A correlated feature domain or raw-input-to-feature analysis is a new contract with new proof and coverage obligations; shrinking the current box after seeing results would invalidate this comparison.

A later native adapter must specify its supported operator, rounding assumptions, input correspondence, and a sound error allowance or refinement theorem before it can use this real bound as a kernel guarantee. Machine-checking the signed extrema theorem and connecting it to the serialized checker would address a different remaining boundary. Neither improvement automatically verifies the feature extractor or proves task usefulness.

The coordinated heldout comparison should answer whether the fixed proposals help their target queues, harm protected queues, or leave the declared feature box. That evidence must be reported separately from mathematical tightness and registry availability. Any richer-domain backend, drift-aware proposer, or deployment study should receive its own frozen design and evaluation release rather than revise these sealed candidates.
