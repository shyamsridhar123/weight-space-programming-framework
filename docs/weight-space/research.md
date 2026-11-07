# Weight-space programming: evidence, hypotheses, and proof obligations

> **Earlier literature review.** This review records the sources and hypotheses
> considered on November 6, 2026. The [current source review](v2/source-review.md)
> explains how the original report informs the extended framework; the
> [current methods](v2/methods.md) explain the study that was actually run.
> The [original review](../../provenance/original-documents/docs/weight-space/research.md.txt)
> preserves the earlier record and its qualifications.

Weight-space programming can be made into a testable engineering discipline by
separating a proposed parameter change, its mathematical contract, and its measured
task effect. The reviewed literature supports exact results for specified
symmetries and restricted edits, useful empirical methods for generating and
aligning weights, and substantial reasons to test transfer and preservation rather
than assume them. This document develops that position for the RLCD framework; it
does not report results from the new local experiments.

The recorded review date is **November 6, 2026, America/Chicago**; retrieval crossed
into **November 7 UTC**. The newest author report reviewed was published on
**November 2, 2026**. The review covers selected methods, theoretical claims, and
evaluation sections of **24 primary papers**, one survey used for discovery, the
supplied index, and a recent author report with **13 inspected released artifacts**.
It is neither exhaustive nor an every-page review. No external experiment was
independently reproduced, and no external repository received a complete code audit.

[references.json](references.json) records exact titles, authors, original arXiv
dates, reviewed version dates, versioned PDF URLs, physical PDF page locations,
SHA-256 digests, and evidence limitations. Page numbers below count the PDF title
page as page 1. Version dates identify the artifact reviewed; they are not claims
about later journal publication dates or that every selected version is the latest.
Browser retrieval returned empty results, so directly retrieved arXiv PDFs, author
HTML, and commit-pinned GitHub files supplied the inspected bytes.

The supplied [Awesome Weight Space Learning index][Index] is pinned at commit
929ecc819301d2ad0b3ccea99e976529998abb81, dated May 12, 2026. Its taxonomy and the
[March 10, 2026 survey][Survey] guided coverage of understanding, representation,
and generation. Listing a method in either source is not evidence of its efficacy.
The selection adds editing, repair, verification, and negative findings because
those topics determine whether a parameter program has an interpretable contract.

**Representation and symmetry.** A network has many parameter descriptions of the
same function. An editor that depends arbitrarily on neuron numbering can fail
after a harmless relabeling. This problem motivates equivariant representations,
but an equivariant editor can still produce an ineffective or harmful edit.

| Primary source and reviewed location | What the source establishes or demonstrates | Assumptions, limits, and consequence for this framework |
| --- | --- | --- |
| [DWS, pp. 2–4, 6][DWS] | Constructs equivariant layers for processing deep network weights, with prediction and editing experiments. | Hidden permutations must act consistently on incoming weights, outgoing weights, and biases. Learned task behavior remains empirical. |
| [NFN, pp. 3, 5][NFN] | Develops permutation-equivariant neural functionals. | Its full NP assumption includes input and output permutations. Fixed input features and output labels are not interchangeable merely because hidden units are. |
| [UNF, pp. 3–4][UNF] | Uses computational structure to construct functionals for different parameter organizations. | It explicitly distinguishes unwanted spatial-filter permutations from function-preserving symmetries. Universal representation does not mean universal behavioral correctness. |
| [GMN, pp. 3–4][GMN] | Represents diverse architectures as parameter graphs and processes them with metanetworks. | Representing a graph does not supply verified semantics for every operator it contains. Architecture adapters need separate obligations. |
| [ScaleGMN, pp. 2–4, 6][ScaleGMN] | Incorporates activation-dependent permutation, positive-scale, and sign symmetries. | ReLU permits positive scaling with a compensating outgoing inverse scale; tanh/sine sign symmetries have different conditions. Scale the bias too. |
| [DeepAlign, pp. 3–4, 7][DeepAlign] | Learns equivariant weight alignment and states recovery results under assumptions. | Equivariance of an optimal alignment requires uniqueness or a set-valued treatment; perfect recovery requires activation and genericity conditions. |
| [Git Re-Basin, pp. 3–4, 7][ReBasin] | Aligns permutations before interpolation or merging and evaluates independently trained models. | Multilayer coordinate descent is heuristic. Applying a valid permutation preserves function exactly; favorable interpolation is empirical. The paper does not establish a zero-barrier ImageNet result. |

The derived design requirement is to record the architecture, fixed feature and
label meanings, supported group action, and every symmetry transform. A discovered
alignment between independently trained networks is a correspondence hypothesis,
not an identity proof. Quotienting by supported symmetries removes known
redundancy; it does not characterize every possible functional equivalence.

**Generation and learned dynamics.** Producing a parameter tensor is a useful
interface, but its provenance includes the model corpus, training procedure, and
any inference-time feedback.

| Primary source and reviewed location | What the source establishes or demonstrates | Assumptions, limits, and consequence for this framework |
| --- | --- | --- |
| [HyperNetworks, pp. 3–4][HyperNetworks] | Trains one network to produce another network's weights. | The generator uses gradient-based training. Generated weights have no automatic behavioral certificate. |
| [HyperDiffusion, pp. 2–3][HyperDiffusion] | Learns diffusion over weights of fitted implicit neural fields. | The field corpus and diffusion model both have training costs; evidence concerns the demonstrated distribution. |
| [SANE, pp. 3–4][SANE] | Uses sequence representations and autoencoding for scalable weight-space learning. | Scalability does not eliminate reconstruction controls, lineage leakage, or the cost of the training corpus. |
| [Generative Modeling of Weights, pp. 3–5][Memorization] | Investigates whether generated checkpoints generalize or largely reproduce training examples. | Its empirical findings motivate nearest-neighbor and noise controls; they are not a universal impossibility theorem. |
| [Drag-and-Drop LLMs, pp. 3–4][DnD] | Learns a prompt-to-adaptation-weight mapping and evaluates zero-shot target-time generation. | The generator is trained on prompt/LoRA checkpoint pairs. Zero-shot inference does not mean absence of gradient training or task-related provenance. |
| [WaRP, pp. 3–4][WaRP] | Learns recurrent weight dynamics of the form θₜ = Aθₜ₋₁ + BΔxₜ. | This supports a dynamic modeling approach, not a guarantee for arbitrary static edits. The reviewed version 3 was revised March 2, 2026. |

A useful evaluation therefore records both construction cost and deployment cost.
Entire training lineages, tasks, and initialization seeds should be separated
where the claimed generalization requires them. Neighbor checks should account
for supported symmetries: raw Euclidean separation alone can mistake a permuted
checkpoint for a novel model. Reconstruction, retrieved-neighbor, random-edit,
and matched-budget optimization baselines test different alternative explanations
for an apparent generator advantage. These are proposed controls, not results
attributed to every paper in the table.

**Arithmetic and localized editing.** Low-rank structure, small parameter norm,
and successful average task arithmetic are useful observations. None alone is a
contract about all outputs on a stated domain.

| Primary source and reviewed location | What the source establishes or demonstrates | Assumptions, limits, and consequence for this framework |
| --- | --- | --- |
| [Model soups, pp. 3, 7][Soups] | Evaluates averaging fine-tuned weights without ensemble inference cost. | Compatibility, common pretraining, and validation/selection choices matter. |
| [Task arithmetic, pp. 2–3][TaskArithmetic] | Uses fine-tuning differences for task addition, negation, and analogies. | Vector addition does not prove independent or compositional semantic effects. |
| [TIES-Merging, pp. 3–4][TIES] | Addresses interference by trimming, choosing signs, and merging compatible entries. | Empirical interference reduction is not a proof that each task contract survives. |
| [Tangent-space arithmetic, pp. 3–4][TangentArithmetic] | Analyzes linearized models and task-support/disentanglement conditions. | Ordinary nonlinear fine-tuning need not satisfy the linearized regime or those conditions. |
| [LoRA, pp. 3–4][LoRA] | Optimizes low-rank updates with a frozen base model. | Rank bounds the update representation; it does not bound output drift, task locality, or transfer error. |
| [ROME, pp. 4–5][ROME] | Derives a constrained associative-memory update and evaluates factual editing. | Estimated keys/covariance and end-to-end locality remain empirical; a local linear equation does not protect all knowledge. |
| [AlphaEdit, pp. 3–5][AlphaEdit] | Restricts updates using preserved-key nullspaces. | Exact protection is ΔK₀ = 0 for specified keys at a linear map. The practical eigenvalue cutoff of 10⁻² yields an approximate nullspace; Wikipedia-estimated keys do not represent all knowledge. |

The derived implementation rule is to attach a behavioral specification to each
edit and check the resulting artifact. If a preservation basis is approximate,
report a residual bound rather than exact preservation. Rechecking after each
edit matters because later edits can alter the features or preconditions used by
an earlier certificate.

**Verification and repair.** A proof about real-valued network semantics, a sound
certificate checker, and a guarantee about deployed floating-point execution are
different claims.

| Primary source and reviewed location | What the source establishes or demonstrates | Assumptions, limits, and consequence for this framework |
| --- | --- | --- |
| [Reluplex, pp. 8–9, 11–12][Reluplex] | Gives an abstract decision procedure for the stated piecewise-linear verification problem. | Page 12 explicitly says that floating-point tableau restoration in the implementation does not guarantee soundness. Abstract soundness/completeness must not be attributed automatically to executable numerics. |
| [CROWN, pp. 3–5][CROWN] | Derives affine output bounds from valid local activation envelopes on a declared input domain. | A failed certificate may reflect loose bounds; it is not necessarily a counterexample. Envelope and domain validity are premises. |
| [Provable Repair, pp. 4–5, 7–9][ProvableRepair] | Changes the architecture to decoupled activation/value networks, making a single value-layer repair linear. | Polytope repair enumerates activation-region vertices. Polynomial-time statements are conditional on the representation; region count can be exponential. This is not an unrestricted repair theorem for an ordinary network. |
| [TorchLean, pp. 1–4, 7–9, 19, 42–43][TorchLean] | Supplies shared SSA/DAG semantics, differentiation, IBP/CROWN analysis, and certificate checking in Lean. | Version 2 was revised May 24, 2026. Table 3 limits proved operator/compiler coverage; finite-path IEEE32Exec refinement is internal, not proof of deployed CPU/CUDA behavior. Its repository was not independently built or audited here. |

TorchLean and Provable Repair establish a clear novelty boundary. This project
should describe its contribution as a contract-guided edit algebra, integration
with explicit symmetry actions, executable provenance, and reproducible
evaluation for a concrete policy. It should not claim to be the first verified
neural framework, the first provable neural repair method, or a universal
compiler from natural-language intent to correct weights.

**Recent direct programming evidence.** The [Sparks author report][Sparks] is dated
November 2, 2026. It supplies interesting positive cases and material negative
cases. The [released repository][SparksLogs] is pinned to commit
4d09ba6d6a7e891dd8063974dfe81b3d78e3494f, dated October 29, 2026. The distinctions
below use the report together with selected README, result-JSON, and manifest
checks; they are not independent replications.

| Task | Inspected evidence | What can and cannot be inferred |
| --- | --- | --- |
| Kinetix policy repair | The [experiment README][SparksKinetix] and [result JSON][SparksKinetixResult] report 1,024 successful episodes out of 1,024, after 6,073 of 171,664 weights changed (3.54%). | One reply and no tool calls is a meaningful constraint. The prompt still supplied scene, observation and architecture metadata plus a failing rollout; edits were applied to rounded displayed weights. This is one informed repair case, not weight-only inference without context. |
| MinAtar synthesis | The [experiment README][SparksMinAtar] describes one fresh reply per game without evaluation feedback and provides exact game source/rules in the prompt. | The PPO comparison displays the best seed and the release omits those PPO checkpoints. Interpret it as the reported comparison, not an independently verified average advantage over PPO. |
| Go synthesis | The [experiment README][SparksGo] reports zero-feedback win rates of 54.1/54.9% against smaller hidden PPO opponents, but 42.6/26.3/36.3% against tuned same-size PPO. Fifty feedback passes yield 74.2/70.0/78.5% against same-size PPO. | The zero-feedback and feedback-assisted arms answer different questions. One run per budget does not estimate variation across repeated editor trials. |
| Sycophancy identification | The [experiment README][SparksSycophancy] says rounds 1–3 output probability 0.5 and forced label A, giving one of three correct by chance. Round 4 stopped around 29–30 calls per pair; only one pair finished. An attention-tool round identified three of three. | Describing this as four completed zero-accuracy trials is incorrect. The evidence includes abstention-like outputs, an incomplete trial, and a positive result with additional instrumentation. |
| Qwen math editing | [Unedited][SparksMathBase] and [p50c edited][SparksMathEdited] JSON records give MATH-500 7.9→48.1%, GSM8K 13.8→67.1%, and IFEval 21.0→62.3%. The [manifest][SparksMathManifest] distinguishes 131,072 touched entries from 130,188 entries different after bfloat16 rounding. | The [experiment README][SparksMath] supports an interpretation involving removal of reasoning loops, not established acquisition of absent mathematics. Fifty passes are batched calls: generation can process up to 512 practice problems and teacher-forced reads up to 1,024 texts per pass. Final unedited GPQA is 22.7%; 24.5% in a blog figure was interim. |
| Gemma follow-up | The [README][SparksGemma], [base JSON][SparksGemmaBase], and [g25b JSON][SparksGemmaEdited] give format-agnostic MATH-500 base 6.8% and edited results 7.0/7.6/7.7/7.0%. | These results do not establish a gain. Grading and sampling differ from the Qwen harness, so their scores should not be compared as a shared benchmark experiment. |

The [release description][SparksRoot] and experiment READMEs identify missing
earlier prompt versions/trial runs and some intermediate or harness assets.
Consequently, this review cannot establish the total unreported selection budget
or independently verify every evaluation path. The positive demonstrations
motivate replicated tests with retained failures; they do not settle a claim that
an LLM can read or program arbitrary networks reliably.

**Mathematical obligations derived for this framework.** The statements here are
proposed proof targets and design consequences of the review. Whether a local
implementation proves each one must be determined from its checked Lean build
and linked certificate artifacts.

Let a supported two-layer network have real semantics

\[
f_\theta(x)=V\,\operatorname{ReLU}(Wx+b)+c.
\]

The artifact identity must include dimensions, feature/output meanings, activation,
numeric interpretation, preprocessing, parameters, and serialization. An edit
program produces a new artifact together with explicit preconditions, a domain,
and a requested postcondition. A content hash identifies the inspected bytes; it
does not by itself establish that a parser or deployed evaluator implements the
proved semantics.

1. **Supported symmetry preserves the function.** For a hidden permutation matrix
   P and positive diagonal D, set W′ = DPW, b′ = DPb, V′ = VP⁻¹D⁻¹, and c′ = c.
   ReLU(DPz) = DP ReLU(z), so fθ′ = fθ. Input coordinates and output labels stay
   fixed. The orbit relation implies functional equality, but the reverse need
   not hold. A transported edit must transform consistently with these weights.

2. **Readout edits have an exact output equation.** When W and b are frozen and
   h(x) = ReLU(Wx+b), an edit (ΔV, Δc) changes the output by
   Δf(x) = ΔVh(x)+Δc. This is a useful compilation target because it exposes the
   realized change without linearizing the entire network.

3. **Anchor preservation is explicitly scoped.** With augmented features
   h̃(x) = (h(x), 1), the constraint ΔṼ h̃(xₖ) = 0 protects the listed anchors.
   It protects their feature span by linearity. It does not protect all inputs
   unless an additional argument covers their features. Approximate nullspace
   computations yield residual bounds, not an exact equation.

4. **Domain bounds imply limited output drift.** If 0 ≤ hⱼ(x) ≤ Hⱼ throughout
   a domain, then each readout coordinate obeys
   |Δfᵢ(x)| ≤ Σⱼ |ΔVᵢⱼ|Hⱼ + |Δcᵢ|.
   Tighter sign-aware interval or affine bounds may admit more edits. For edits to
   both layers, induced infinity norms and ReLU's Lipschitz property give, when
   ∥x∥∞ ≤ R,
   ∥Δf∥∞ ≤ ∥ΔV∥∞∥h∥∞ + ∥V′∥∞(∥ΔW∥∞R + ∥Δb∥∞) + ∥Δc∥∞.
   These are sufficient bounds; failure to meet them is not proof of actual
   violation.

5. **A score margin can imply an unchanged decision.** For the same legal
   candidate set, coordinate drift bounded by ε preserves a unique argmax if its
   original lead over every competitor exceeds 2ε. Ties need the actual tie
   rule. A common additive score shift or positive common rescaling preserves
   order in real arithmetic. A stochastic softmax policy needs a distributional
   argument; score bounds alone do not identify sampled actions.

6. **Composition requires compatible postconditions and identity.** A poststate
   proved for one edit must be the prestate of the next. On a common domain,
   per-step drift bounds add by the triangle inequality. Independent target
   gains do not add automatically. Changing the feature map, domain, action
   selector, or numeric interpretation can invalidate an earlier premise.

These obligations support a small proof-producing core and a broader proposal
interface. An LLM, optimizer, learned functional, or handwritten rule can propose
weights; admission then depends on a checked contract for the supported semantics.
This architecture allows failed proposals to remain visible and meaningful.

**Falsifiable hypotheses and evaluation.** The current
[protocol](../../experiments/weight-space/protocol.json) fixes three exploratory
hypotheses before the new experiment results. Its evidence units are deliberately
limited: H1 uses **20 disjoint source/target seed pairs, 40 trained models in
total**, and H2 uses the **20 source models**. One fixed, disjoint data partition is
shared across these models, so the inference is conditional on that partition
and the chosen teacher/distribution. Initialization seeds do not establish
generalization over datasets or task families.

| Hypothesis and null | Primary measurement and controls | Falsification and scope |
| --- | --- | --- |
| H1: incoming-weight alignment reduces edit-transfer error between independently trained networks. Null: mean paired raw-minus-aligned error is nonpositive. | Compare normalized RMSE of transferred output changes against the source model's realized changes on held-out target-region inputs. Use 20 disjoint pairs. Known same-function permutation/scaling is a positive control; raw coordinate transfer after permutation is a negative control. | The paired bootstrap 95% lower endpoint must exceed zero for support. The control's exact symmetry does not establish semantic equivalence for independent networks. Fixed data and one architecture limit inference. |
| H2: retaining non-target behavior during readout compilation lowers collateral drift while retaining useful target gain. Null: collateral reduction is nonpositive, or mean target gain fails the declared threshold. | Compare held-out retention RMSE within each of 20 source models using the same compile data, parameterization, and fixed regularization choices except for the retention term. The protocol requires mean target gain at least 50% of the requested 0.75 shift. | Support requires the collateral-reduction interval's lower endpoint above zero and the target-gain criterion. Report both endpoints, all failures, and admission/rejection; preservation achieved only by a zero edit is insufficient evidence of useful editing. |
| H3: a development/validation-selected recruitment edit improves held-out RTS utility without a higher observed loss rate. Null: mean paired utility gain is nonpositive or observed loss rate increases. | Select using the declared development/validation split and freeze before 20 held-out world seeds. Compare to baseline with no-op and shared output-bias controls. Utility is opponent town-center damage fraction minus own damage fraction plus 0.02 times completed recruited soldiers. | Support requires a positive paired-bootstrap lower endpoint and no higher observed loss rate over 420 seconds. Equal observed loss rates are not a statistical noninferiority proof. The declared utility is a proxy tied to one mission and opponent. |

The protocol uses 10,000 percentile paired-bootstrap resamples and unadjusted 95%
intervals for three exploratory hypotheses. These intervals are conditional
uncertainty summaries, not a multiplicity-adjusted confirmatory finding. Training
failures, infeasible synthesis, rejected certificates, and failed simulations
belong in the trial record; incomplete planned units leave the corresponding
hypothesis inconclusive under the stated policy.

The design should report useful-edit success over all proposed trials, along with
success among admitted trials and the admission fraction. Otherwise a conservative
checker can appear effective merely by rejecting almost everything. Distinguish
an actual counterexample, a rejected conservative bound, invalid input, and an
execution failure. Budget reports should separate training compute, target-time
search, generated tokens, feedback calls, examples per call, parameter changes,
and measured latency where each is relevant.

Follow-up hypotheses are **not** part of the fixed three-hypothesis experiment:
whether equivariant learned proposals outperform matched-budget non-equivariant
ones across withheld lineages; whether certificate coverage remains useful as
domains widen; and whether a sequence of individually accepted edits retains
useful gains under cumulative contracts. Such work requires new pre-result
protocols, whole-lineage/task splits, per-step rechecking, and both unconditional
joint success and coverage measurements. It must not be presented as already
tested by the current small experiment.

**Project-specific interpretation and remaining gaps.** The
[current-state audit](requirements.md) identifies the RTS scorer as **32→38→1
with 1,293 parameters**. Its output scores a candidate; it is not a success
probability. First Conquest selects the highest score among legal candidates and
breaks ties by candidate ID. The separate living-world policy has different
observations and stochastic softmax selection. Certificates, demonstrations, and
rollouts must name the relevant adapter.

For the RTS experiments, a score contract on the declared feature box and an
exactly protected non-recruitment region can justify claims about scores on those
inputs. It does not establish better strategy, longer-horizon outcomes, or
robustness to changing feature semantics. Prior project simulation seeds are
previously used evidence, not untouched holdouts. New rollouts require frozen
artifacts, disjoint selection/evaluation seeds, consistent legal-action filtering,
and the recorded opponent and horizon.

The remaining research questions are measurable: does alignment improve transfer
beyond same-function controls; does retention help without erasing target gain;
how often do useful edits admit certificates; and do accepted policy edits improve
the declared rollout endpoint? The mathematical work can establish conditional
properties even if any of those empirical hypotheses fails. Conversely, a
positive rollout result does not fill a gap between real arithmetic and the
deployed numerical implementation.

[DWS]: https://arxiv.org/pdf/2301.12780v2
[NFN]: https://arxiv.org/pdf/2302.14040v3
[UNF]: https://arxiv.org/pdf/2402.05232v1
[GMN]: https://arxiv.org/pdf/2312.04501v2
[ScaleGMN]: https://arxiv.org/pdf/2406.10685v2
[DeepAlign]: https://arxiv.org/pdf/2310.13397v4
[ReBasin]: https://arxiv.org/pdf/2209.04836v6
[HyperNetworks]: https://arxiv.org/pdf/1609.09106v4
[HyperDiffusion]: https://arxiv.org/pdf/2303.17015v1
[SANE]: https://arxiv.org/pdf/2406.09997v1
[Memorization]: https://arxiv.org/pdf/2506.07998v2
[DnD]: https://arxiv.org/pdf/2506.16406v1
[WaRP]: https://arxiv.org/pdf/2506.01153v3
[Soups]: https://arxiv.org/pdf/2203.05482v3
[TaskArithmetic]: https://arxiv.org/pdf/2212.04089v3
[TIES]: https://arxiv.org/pdf/2306.01708v2
[TangentArithmetic]: https://arxiv.org/pdf/2305.12827v3
[LoRA]: https://arxiv.org/pdf/2106.09685v2
[ROME]: https://arxiv.org/pdf/2202.05262v5
[AlphaEdit]: https://arxiv.org/pdf/2410.02355v4
[Reluplex]: https://arxiv.org/pdf/1702.01135v2
[CROWN]: https://arxiv.org/pdf/1811.00866v1
[ProvableRepair]: https://arxiv.org/pdf/2104.04413v2
[TorchLean]: https://arxiv.org/pdf/2602.22631v2
[Survey]: https://arxiv.org/pdf/2603.10090v1
[Index]: https://raw.githubusercontent.com/Zehong-Wang/Awesome-Weight-Space-Learning/929ecc819301d2ad0b3ccea99e976529998abb81/README.md
[Sparks]: https://rikiyat.github.io/blog/weight-space-programming/
[SparksLogs]: https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
[SparksRoot]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/README.md
[SparksKinetix]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp4_fix_car/README.md
[SparksMinAtar]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch/README.md
[SparksGo]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp6_go_from_scratch/README.md
[SparksSycophancy]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp7_sycophancy/README.md
[SparksMath]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md
[SparksGemma]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/gemma_followup/README.md
[SparksKinetixResult]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp4_fix_car/eval/oneshot_result.json
[SparksMathBase]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/eval/unedited/merged.json
[SparksMathEdited]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/eval/p50c/merged.json
[SparksMathManifest]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json
[SparksGemmaBase]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/gemma_followup/eval/unedited/merged.json
[SparksGemmaEdited]: https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/gemma_followup/eval/g25b/merged.json
