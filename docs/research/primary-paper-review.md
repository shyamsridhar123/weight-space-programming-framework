# Primary-source review: what weight-space programming can establish next

**Review date: November 8, 2026, America/Chicago.** This is a full-text review of
the original author report and six primary papers, including their appendices.
It extends the earlier [selected original-release review](../weight-space/v2/source-review.md)
and the [limitations and next steps](../weight-space/v2/limitations-and-next-steps.md).
The [research roadmap](../weight-space/v2/research-roadmap-2026-11.md) supplies the
proposed enterprise study and issue dependencies.

The strongest next contribution is a framework that makes an edit's information,
artifact, mathematical scope, numerical behavior, utility, retention, and cost
inspectable together. These papers supply useful components for that framework.
They do not jointly establish a general editor that understands any network,
transfers arbitrary circuits, preserves all behavior, or removes all access to
specified information.

Text routing provides a concrete first trained application: a support request
that previously went to a general queue must now go to a specialist, while other
requests keep their intended routes. The literature makes two questions
essential. Does the representation permit that selective change? Does changing
the model improve on changing the routing policy after an unchanged prediction?
Those questions apply to classifiers, controllers, image models, and other
supported tasks; the framework is not specific to real-time strategy or games.

## Scope, acquisition, and the meaning of this review

All pages of the six pinned PDFs were read, rather than only abstracts or the
pages recorded in the earlier source register. The full original report was
read. The earlier selected-release audit was used for its documented observations
of prompts, evaluations, patches, and reproduction limits. This review did not
repeat that release audit exhaustively, execute any external implementation,
build TorchLean, reconstruct a large checkpoint, train a reader, run an editor,
or inspect the framework's withheld trained-study outcomes.

The ordinary web reader returned empty displays when opening primary pages;
those responses did not verify their contents. Read-only HTTPS retrieval of the
official author page and arXiv pages/PDFs succeeded. Fresh retrievals occurred at
approximately 03:03 UTC on **November 9, 2026**, which was **November 8, 2026** in
America/Chicago. The retrieved original-report HTML and all six pinned PDFs
matched the previously cached byte hashes. Official arXiv version histories
were checked separately from PDF identities. This is source verification, not
independent verification of the experiments described by their authors.

PDF locators below use **physical PDF page numbers**, including appendices and
front matter. An author's printed page number can differ. Original-report
locators use experiment or discussion names because the source is HTML. Links
point to primary sources or to this repository's explicitly bounded prior audit.
Public document hashes identify the reviewed bytes; they are not measures of
scientific confidence.

| Short name | Primary source and reviewed version | Version date | Full reading |
| --- | --- | --- | --- |
| Sparks | [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights][SP]; author report/blog | November 2, 2026 | Entire report, including “What Comes Next?” |
| DeepAlign | [Equivariant Deep Weight Space Alignment][DA], `2310.13397v4` | November 9, 2024 | PDF pp. 1–20 |
| DWS | [Equivariant Architectures for Learning in Deep Weight Spaces][DWS], `2301.12780v2`; ICML 2023 | May 31, 2023 | PDF pp. 1–27 |
| NFN | [Permutation Equivariant Neural Functionals][NFN], `2302.14040v3`; NeurIPS 2023 | September 26, 2023 | PDF pp. 1–25 |
| ScaleGMN | [Scale Equivariant Graph Metanetworks][SG], `2406.10685v2`; NeurIPS 2024 | October 30, 2024 | PDF pp. 1–34 |
| TorchLean | [TorchLean: Formalizing Neural Networks in Lean][TL], `2602.22631v2`; preprint | May 24, 2026 | PDF pp. 1–55 |
| TOFU | [TOFU: A Task of Fictitious Unlearning for LLMs][TOFU], `2401.06121v1` | January 11, 2024 | PDF pp. 1–24 |

The [research source register](../weight-space/v2/research-sources-2026-11.json)
covers a wider roadmap corpus. This review expands the selective readings from
the first roadmap pass for these seven sources. Its source table, page locators,
and hash register identify that expanded scope. Papers outside this table are
not silently included in the full-text claim.

Throughout this document:

- **Source finding** means a theorem, experiment, or limitation described in the
  pinned source. An experimental finding remains the authors' reported result.
- **Author interpretation** means an explanation or future direction offered
  by those authors, without turning it into a demonstrated mechanism.
- **Review inference** means a consequence drawn here from stated assumptions,
  metrics, or experimental design.
- **Framework proposal** means work this repository should perform. It is not
  an outcome already obtained.

## The original report: preserve the demonstrations and their boundaries

Sparks is the report by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab,
published November 2, 2026. It is an author report/blog, rather than a
peer-reviewed paper. Its demonstrations span policy identification, image-output
identification, language/attention analysis, policy synthesis, behavior repair,
and sycophancy analysis. Their access conditions differ materially. [SP]
(experiments 1–8); [AUDIT] (“Sources and review boundary”).

### What the report supports

Small-model identification combines tensors with supplied architecture and
semantics. The game-policy editor receives observation channels and action
meanings; the diffusion editor receives pixel semantics, noise schedule and
training objective; language identification includes forward code, training
procedure and candidate corpus descriptions. Correct answers under these
conditions are meaningful observations about a contextualized task. They do not
isolate what the weights add over the same context without weights. [SP] (experiments 1–3; AUDIT, “Supplied semantics and hints are part of the task.”)

The diffusion demonstrations expose a scoring boundary particularly clearly:
the shape categories were correct, while descriptions called filled targets
hollow or outlines. A category-only answer key can therefore reward an answer
that misstates output structure. The future benchmark needs separate rubrics
for identity, visual properties, confidence, and mechanistic explanation. [SP] (experiment 2; AUDIT, “Identification, mechanistic diagnosis, and causal repair
are separate claims.”)

Kinetix evaluates one proposed actor edit across repeated episodes. The reported
episodes measure the behavior of that candidate under the rollout distribution;
they do not provide independent repetitions of the editing process. MinAtar's
written policy has a strong Breakout comparison, while the Seaquest and Asterix
comparisons differ. The selected PPO training budget and game must remain in the
claim. Go's comparisons also depend on opponent size, training/tuning and
feedback allowance. [SP] (experiments 4–6; AUDIT, task tables and “Baseline
selection and omitted history limit generality.”)

The Qwen math experiment is adaptive repair with labels, tools, feedback and
diagnostic guidance. Its reported improvements belong to the retained edited
candidate and its evaluation protocol. The authors' loop-direction account is
a mechanistic interpretation that needs interventions against matched controls.
The Gemma follow-up did not establish the same improvement; an obligatory final
submission is not evidence that the submission helps. [SP] (experiment 8;
AUDIT, Qwen/Gemma comparisons and “Identification, mechanistic diagnosis, and
causal repair are separate claims.”)

In sycophancy, the inspected weight-only rounds did not identify the pair with
informative confidence. An attention-assisted arm classified the pairs, while
the inspected interventions did not establish successful repair. Classification,
localization and causal behavioral change must remain different endpoints.
[SP] (experiment 7; AUDIT, inspected weight-only and attention interventions.)

### Access and cost are part of the result

“No tools,” “no target forward pass,” “no feedback,” “no labeled data,” and
“no fitting” constrain different operations. The original Go no-training
construction disallows regression and least-squares fitting as well as gradient
training. A framework that solves a least-squares readout problem may still be
useful, but that experiment belongs in a fitted arm. Source-provided architecture,
action meanings, candidate descriptions, labeled practice, diagnosis hints and
scoring details must appear in the access manifest. [SP] (experiments 3, 5, 6
and 8; AUDIT, “‘No training,’ ‘tool free,’ and ‘few passes’ need operational
definitions.”)

The report's charged-pass units differ. The prior audit documents game/position
batches, generated-answer batches and teacher-forced text batches with different
limits; duration and batch size are not necessarily charged. Equal pass counts
therefore do not imply equal compute, examples, token exposure, or information.
Budget experiments need both the task's original accounting and a common
resource ledger. Infrastructure retries and failed attempts must remain
visible even if they do not consume a charged pass. [AUDIT] (same section,
original notices and sandbox observations.)

Candidate realization needs its own evidence. Rounded input weights, parser
repairs, a proposed patch, a reconstructed checkpoint, and native dtype
conversion are distinct objects. A sparse stored patch neither proves that the
desired candidate was realized nor that unrelated outputs were retained.
[AUDIT] (“Exact candidate realization matters” and “Reproducibility is a set of
distinct obligations.”)

### What remains a hypothesis

The report's “What Comes Next?” proposes auditing without ordinary evaluation,
cheaper optimization, zero-data steering/unlearning, recursive improvement and
circuit transfer across architectures; it also identifies context/harness
limits and a benchmark under development. These are useful research questions.
The reported edits do not demonstrate general backdoor detection, robust
unlearning, absence of catastrophic forgetting, recursive improvement, or
cross-architecture circuit grafting. [SP] (“What Comes Next?”; AUDIT, “Actual
limitations and next steps in the original report.”)

**Framework consequence.** Keep the original task distinctions: read,
diagnose, repair, and construct. Reproduce a selected task with the original
access rules before claiming a faithful repetition. Then change one rule at a
time. Independent training lineages and editor runs are the units for claims
about repeatability; repeated evaluation of one candidate remains nested within
that candidate. These are evidence obligations for [issue #15][I15], rather
than a reason to average the demonstrations into a single success rate.

## DeepAlign: alignment of compatible coordinates precedes transfer claims

### The contribution and theorem

DeepAlign learns to align compatible networks while respecting the permutation
structure of the alignment problem. Its inference produces assignment scores
and uses **Hungarian assignment** to obtain permutations. Differentiable
projection used in training is not the same operation as the final discrete
assignment. The learned reader and the model collection used to train it are
part of the method. [DA] (architecture/projection, PDF pp. 4–6; experimental
details, Appendix E, pp. 16–17.)

Proposition 5.2 concerns perfect, permutation-related inputs. Under nonconstant
analytic activations and at least two feature channels, the claimed recovery is
an almost-everywhere result in the stated input/parameter setting. It does not
say the method finds a globally optimal alignment for arbitrary independently
trained models. The distinction is explicit in the discussion of “tame” inputs
preceding the proposition. [DA] (Proposition 5.2 and “Exactness,” PDF p. 7;
proof, Appendix C.)

The paper evaluates weight alignment and model merging using interpolation loss
barriers, area under the interpolation curve and related distances. A small
barrier can make merging useful in that setting. Weight-space distance and loss
interpolation are different endpoints; neither measures whether a selected
causal circuit has transferred or whether protected examples retain their
original decisions. [DA] (evaluation metrics, PDF p. 7; experiments, pp. 8–9.)

### What “no labels” and “fast inference” leave to account for

The absence of externally labeled alignment pairs does not remove preparation.
Model collections, synthetically permuted examples, reader optimization, and
task data/loss in relevant loss-based configurations still enter the workflow.
Source-model training and learned-reader training should be charged separately
from aligning one held-out pair. Tests on held-out weights also need a split by
the source training run, rather than a split that scatters related checkpoints
across training and evaluation. [DA] (training objectives, PDF pp. 5–6;
Appendix E, pp. 16–17.)

The time-comparison description averages alignment over 1,000 random pairs on a
single NVIDIA A100 for the stated CNN/VGG collections. That is inference timing
for the alignment operation, rather than complete model preparation, reader
training or downstream validation. **The timing unit is unresolved in this
review:** Figure 5's axis is labeled “Runtime,” and Appendix F Table 6's header
also omits a unit. No numerical seconds or speedup calculation is inferred from
those entries here. [DA] (Figure 5, PDF p. 7; time-comparison description,
Appendix E, p. 17; Table 6, p. 19.)

### What the framework should take from it

For [issue #16][I16], start with known function-preserving permutations of one
supported model. Transport an edit through the known permutation and verify
that the target/protected contracts agree after transport. This is a controlled
test of coordinate handling. Next, align independently trained models of the
same supported architecture and test actual transfer against an unaligned edit,
a matched random/sham edit and an independently fit target edit.

A successful same-function permutation test establishes transport under a
known symmetry. A successful independent-model experiment establishes only the
tested transfer distribution. Cross-architecture grafting would need an
additional representation and causal argument; the alignment theorem does not
provide that bridge.

## DWS: an equivariant learned reader with explicit model-family assumptions

### What is characterized

DWS constructs networks that operate on MLP weights and biases using pooling,
broadcasting and shared affine operations. Its affine equivariant/invariant
layer characterization is for the stated hidden-neuron permutation action and
MLP structure. It does not characterize every graph architecture, activation
rescaling, or all nonlinear maps over arbitrary neural programs. [DWS] (architecture and characterization, §§3–5, PDF pp. 3–8; appendices B–E.)

Proposition 6.1 approximates the forward computation of a fixed ReLU MLP on
specified compact input/weight domains. Proposition 6.2 adds compactness and
regularity assumptions, including uniform Lipschitz control and regularity of
the relevant output-volume maps, to approximate the stated function-space
mapping. These conditions matter: the propositions are not a blanket
universality result for any equivariant editing problem. A theorem that a reader
can express an input network's computation also does not establish compliance
with a benchmark that forbids computing that network. [DWS] (§6, PDF p. 9;
Appendix F, pp. 20–21; Appendix G, pp. 21–22.)

The experiments include INR classification, classifier generalization-related
tasks and domain adaptation. In the domain-adaptation setup, preparation uses
source/target examples and differentiates through an edited classifier. “No
target examples at inference” therefore describes an inference boundary, not
the absence of target-data supervision across the method. [DWS] (experiments,
PDF pp. 9–10; Appendix J, pp. 23–25.)

### Preparation, scaling and unsuccessful cases

The simplified equal-width analysis compares DWS parameter scaling
`O((M+d)^2)` with flattened fully connected scaling `O((Md^2)^2)`, where the
paper's layer/width definitions apply. This is a parameter-count analysis;
tensor operation cost, activations, model-collection storage, batch sizes and
reader optimization remain additional costs. [DWS] (Appendix I, pp. 22–23.)

Preparation includes fitting per-image INRs, training classifier collections
and retaining trajectory checkpoints. Appendix J describes, among other
settings, up to 1,000 Adam steps for MNIST/Fashion image INRs with a PSNR stopping
rule, a collection of CIFAR classifiers trained for two epochs, and three reader
seeds for INR classification. Domain-adaptation preparation uses a roughly
four-million-parameter reader, 10,000 optimization steps and batches of models
and images. These setup details support a cost ledger; they are not independent
evidence that an enterprise edit is fast or robust. [DWS] (Appendix J,
pp. 23–25.)

One baseline's preparation is sensitive to convergence settings: Appendix J
reports that MergeMany did not complete an iteration at its initial tolerance
after more than 24 hours, followed by a changed tolerance and a reduced model
subset. Such changes need to appear in a reproduction, rather than presenting
all comparator costs as identically configured. [DWS] (Appendix J, PDF p. 24.)

Appendix K.5 reports unsuccessful or problematic applications. Pruning could
concentrate removals in some layers, and CIFAR INR classification was
unsatisfactory. Suggested initialization or outside-domain-noise explanations
are author hypotheses, not independently confirmed remedies. This is useful
negative evidence against treating an equivariant architecture as sufficient
for every downstream task. [DWS] (Appendix K.5, PDF pp. 26–27.)

### Implications for the generic framework

A DWS-based reader is a plausible comparator for supported MLPs. Its model-family
signature should declare layer shapes, hidden-neuron permutation rules,
input/output semantics, activations and any preprocessing. Same-image INRs must
stay grouped across splits; checkpoints from one training trajectory must stay
grouped by training lineage. Otherwise a reader can exploit repeated views of
the same object or training run while evaluation appears to contain many
independent models.

The important enterprise comparison is whether a prepared reader improves
useful target edits or diagnosis beyond a context-only reader, and at what
amortized preparation cost. A learned reader trained with labeled target
examples belongs to a declared fitted preparation arm for [issue #17][I17]. It
does not qualify as no-fitting construction simply because deployment uses a
single forward call through the reader.

## NFN: linear symmetry structure helps, but task evidence is separate

### The symmetry and architecture contribution

NFN studies neural functionals over weights or gradients. It characterizes
**linear** equivariant NF-layers under two permutation choices. Hidden-neuron
permutation, HNP, preserves fixed input/output meaning; the stronger neuron
permutation choice, NP, treats a larger set of coordinates as permutable, with
positional encodings modifying that symmetry where fixed identities are needed.
The characterization is not a universal guarantee for arbitrary nonlinear
editing or arbitrary language-model reasoning about tensors. [NFN] (§§2–3,
PDF pp. 3–9; appendices B–C, pp. 15–20.)

Parameter scaling differs: the paper gives HNP scaling proportional to
`c_i c_o (L+n_0+n_L)^2` and NP scaling proportional to `c_i c_o L^2` under its
notation. The CNN extension folds spatial kernel dimensions into feature
channels. Complicated convolution-to-fully-connected flattening and
ResNet/Transformer architecture support remain boundaries/future work in this
paper. Activation memory is an explicit limitation. [NFN] (layer construction,
PDF pp. 4–6; limitations, §5, p. 9.)

The distinction between HNP and NP is operationally important for support
routing. Input feature meanings and output queue identities cannot be changed
silently by a symmetry transformation. Hidden-neuron relabeling may preserve
the represented function; relabeling the customer-visible outputs changes a
decision unless the downstream semantics are transformed too.

### What the experiments demonstrate, including an informative ablation

The paper reports learned generalization prediction, winning-ticket mask
prediction, INR classification and supervised INR editing for selected MLP/CNN
families. These are prepared supervised neural-functional tasks. INR editing
optimizes a loss through the edited representation using target image pixels;
it is not an instruction-only edit with no target evaluations or labels.
[NFN] (§§3.1–3.4, PDF pp. 6–9; Appendix D, pp. 20–22.)

The winning-ticket ablation prevents an overly broad interpretation. In Table
5, the pointwise model matches or exceeds the NP model on the reported
95%-sparse CIFAR and MNIST ticket outcomes. The task therefore does not establish
that modeling weight interactions is always necessary or better. The outcomes
are obtained after ticket training; they also depend on IMP/model preparation,
rather than only the mask reader's runtime. [NFN] (§3.3 and Table 5,
PDF pp. 7–8; Appendix D.2, pp. 20–21.)

For image editing, Table 6 evaluates MNIST dilation and CIFAR contrast using
reconstructed-image error. This demonstrates the studied transformations and
loss, not broad natural-language edit transfer, causal circuit identification,
or preservation of unrelated tasks. A pixel MSE improvement should remain a
pixel MSE improvement in the framework's evidence record. [NFN] (§3.4 and
Table 6, PDF pp. 8–9.)

### Costs and independent units

Appendix D reports approximately one hour of Titan RTX training for the
generalization reader, five hours for ticket prediction, four hours for INR
classification and one hour for INR editing in the stated settings. These are
reader-training reports, not the total cost of training classifier collections,
performing iterative magnitude pruning, fitting all INRs and validating edited
outputs. Hardware and schedules must travel with any quoted cost. [NFN] (Appendix D.1–D.4, PDF pp. 20–22.)

The INR collections include independently initialized copies of the same
underlying image. Generalization collections include several checkpoints from
each source training run. The source's run-level split is a valuable design
detail. A new experiment should retain this grouping and make image/model/run
identity explicit in uncertainty estimates. A copy of an image representation
is a different initialization; it is not a new independent image. A checkpoint
is a different time in a trajectory; it is not a new independent training run.
[NFN] (Appendix D.3, pp. 21–22; Appendix E.2, pp. 23–25.)

Reported uncertainty also differs by table: extrema across two runs, standard
errors across three runs, and initialization variation are not interchangeable.
Pooling them into a common reliability number would erase what varies in each
study. [NFN] (experimental tables and their captions, PDF pp. 6–9.)

**Framework consequence.** Include pointwise and simpler readers alongside
symmetry-aware models. Evaluate target gain, protected behavior and independence
at the task's natural unit. Record whether the reader observes weights,
gradients or activations and whether it was trained through target-model
forwards. These choices support fair tests for [#16][I16] and [#17][I17]; their
labels do not determine the result in advance.

## ScaleGMN: activation-specific symmetry and an ideal simulation theorem

### Scaling is not one universal group

ScaleGMN extends graph metanetworks to respect supported scaling as well as
permutation symmetries. Positive ReLU rescaling and tanh/sine sign transformations
have different validity conditions. Fixed input/output meanings remain part of
the graph construction. The activation hypotheses include the stated
invertibility/regularity conditions; the sine construction also has restrictions
on its frequency parameter. [SG] (§4, PDF pp. 5–6; Appendix A.7.1,
pp. 27–28.)

For a ReLU hidden unit, real-arithmetic preservation can multiply incoming
weights and bias by a positive scalar and divide corresponding outgoing weights
by that scalar. Applying arbitrary positive rescaling to a tanh unit does not
give the same identity. Even a valid real-arithmetic symmetry needs separate
finite-precision checking: extreme scaling can change rounding, underflow,
overflow, and decision margins. This latter observation is a review inference
about the framework's numerical bridge, not a reported enterprise experiment.

### The expressivity result and its assumptions

Theorem 5.2 constructs a sufficiently expressive ScaleGMN that simulates forward
and backward computations of an input feedforward network. It assumes message
and update functions capable of expressing the relevant activations/derivatives,
and supplies input values and output loss gradients through the construction's
positional information. Forward simulation takes the stated layer traversal;
bidirectional simulation handles the backward traversal. This is a constructive
expressivity theorem, not evidence that a practical module learns the intended
algorithm or that a no-evaluation interface prevents its computation. [SG] (§5, PDF pp. 7–8; Appendix A.7.2, pp. 30–34.)

The proof uses inverted edge weights and reciprocal quantities in parts of the
backward construction. Zeros, reciprocal domains and finite precision require
explicit treatment before a numerical implementation inherits the claim.
The appendices themselves discuss instability from inversion and unusual
reciprocal distributions, and practical modules avoid some inversions for
stability. The ideal construction and the implemented module must not be
collapsed into one theorem. [SG] (Appendix A.1.3, p. 18; Appendix A.2,
pp. 19–20; Appendix A.7.2, pp. 30–34.)

The practical Hadamard-product construction is narrower than the full
outer-product construction discussed for expressivity. Skip connections,
softmax and normalization **in the input networks** are excluded from the
paper's supported scope. Full expressivity of the practical scaling modules
and full graph-metanetwork expressivity remain open questions in the source.
[SG] (§5, p. 7; limitations, §6, PDF p. 10.)

### Empirical scope and reproduction details

Experiments concern stated weight collections and tasks, including
generalization/INR classification and supervised image editing. The MNIST
dilation result is informative: the forward variant does not beat the stated
probe baseline, while the bidirectional variant does. This supports comparing
specific variants, rather than replacing the table with a blanket claim that
scaling-aware processing dominates every baseline. Some comparisons use new
reruns and others reported results; preparation and optimization schedules
should remain attached to the relevant row. [SG] (Tables 1–3, PDF pp. 8–9;
Appendix A.4, pp. 21–24.)

The paper describes RTX 4090 tuning grids, graph/feature widths, batch sizes,
and task-specific schedules. Its reuse of existing weight collections saves
work for that experiment but does not make those collections free for a new
framework. Forward graph traversal scales as `O(V+E)` and bidirectional traversal
as `O(V+2E)` under the paper's abstraction; feature width, message depth,
training batches and edge storage still determine actual resource use. [SG] (Appendix A.4, pp. 21–24; Appendix A.6, p. 26.)

There is a concrete reproduction ambiguity: Appendix A.4.3 gives a gamma
initialization of `0.01` in the equation on PDF p. 22 and `0.001` in text on
p. 23. A reproduction should resolve the pinned release configuration or
request clarification, record the choice, and avoid silently rewriting the
paper. Sine also has additional periodic/phase symmetries; the chosen bounded
bias setup removes some cases from the experiments. The tested symmetry set is
therefore not every possible equivalence of every sine parameterization.
[SG] (Appendix A.4.3–A.4.4, PDF pp. 22–24.)

### Implications for transfer and restricted access

For [#16][I16], an adapter should declare the exact valid symmetry group for
its activation and architecture. Validate known transformations first, including
numerical realization, output semantics and protected decisions. Compare
permutation-only, scale-aware and raw-coordinate transfer at matched task
information. Differences between these arms are evidence about representation,
rather than a proof that the transferred feature is a causal circuit.

For [#17][I17], record whether the reader's preparation uses input values,
target-model forwards, gradients or supervision. Algebraic simulation of the
target computation still produces target information. Interface rules need to
state whether equivalent reconstructed computations are forbidden; a tool name
alone does not enforce that boundary.

## TorchLean: useful semantic machinery with a visible proof/runtime boundary

### What connects execution and analysis

TorchLean uses an operator-tagged SSA/DAG intermediate representation as a shared
semantic target. Its API, eager execution, graph lowering and verification
machinery are connected through explicit representations and semantics. This
is directly relevant to a framework whose proposed candidate, mathematical
certificate and executed native model could otherwise diverge. [TL] (§§2–3,
PDF pp. 3–12; scope inventory, Appendix A, Table 3, p. 19.)

The source explicitly distinguishes **proved**, **checked**, **external**, and
**future-scope** components in Table 3. Executable examples across many model
families are not all full end-to-end correctness theorems. This review adopts
that distinction rather than interpreting the abstract's broad component list
as a universal verification claim. [TL] (Table 3, PDF p. 19.)

The lowering results in Appendix B.1–B.2 cover a first-order forward fragment.
The full compiler-correctness bridge for arbitrary higher-order/tagless user
programs is future work. Eager correspondence concerns a fixed completed
execution. Dynamic control flow, data-dependent shape changes, general
gather/scatter, arbitrary variable-length sequences and hidden mutable state
fall outside the stronger fixed-graph verifier fragment. [TL] (Appendix B.1–B.2,
PDF p. 22; graph/verifier scope, Table 3 and Appendix B.)

### Derivatives and floating point require premises

Classical derivative correctness uses local operator hypotheses. ReLU at zero,
log outside its domain, inverse/division at zero, and max/min ties need
pointwise conditions or an explicitly different mathematical object. A runtime
kink convention is not automatically a theorem about the classical derivative
at a nondifferentiable point. The same discipline is needed for trained
candidate bounds. [TL] (operator contract inventory, Table 4, PDF p. 24;
Appendix B.6–B.7, pp. 27–28.)

Appendix D.1 relates explicit IEEE32 execution to internal FP32/normal-form
semantics for stated core operations and **finite execution**, excluding
NaN/Inf/overflow cases. It does not universally validate every transcendental
implementation or every hardware instruction sequence. The link to a deployed
CPU/CUDA target still depends on rounding, subnormal/FTZ/DAZ behavior, fused
multiply-add and reassociation, reduction order, library/compiler versions and
nonfinite policy. These premises must be attached to the backend under test.
[TL] (Appendix D.1, PDF p. 43; explicit native/FFI boundaries in Table 3,
p. 19, and associated appendices.)

Float/Float32 primitives, CUDA/cuBLAS/cuFFT and FFI, ONNX export, Arb/FLINT and
external simulators have stated boundary roles. Formal attention/FlashAttention
equalities concern denotational specifications; they are not a verification of
the deployed CUDA FlashAttention kernel. Similarly, replay of interval/affine or
CROWN-style certificates does not make the whole external optimization tool
trusted proof code. Shape, enclosure, phase, coverage and property obligations
belong to the checker; external search and richer unverified solver outputs
remain outside that boundary. [TL] (Table 3, PDF p. 19; attention discussion,
pp. 33–34; certificate-checking and external-boundary appendices.)

Appendix F.2 gives an exact-hull specification, rather than a general algorithmic
theorem that exact hulls are attained for every case. Appendix F.3–F.4 separates
approximation, parameter and execution errors under explicit hypotheses. The
framework can use this style of decomposition without asserting that its own
serialization, operators or native kernels have already been proved correct.
[TL] (Appendix F.2–F.4, PDF pp. 54–55.)

### Costs cannot be reduced to certificate replay latency

The paper's tiny-controller Stage 2 example reports approximately `0.015 s` for
a PyTorch execution and `1,212 s` for explicit IEEE32 execution, alongside
different positive-candidate counts. This is a sharply scoped illustration of
the cost/semantic tradeoff, not an enterprise throughput benchmark or an
equal-candidate speed comparison. The framework should count graph preparation,
prover/search time, compilation, replay, model training, native execution and
evaluation separately. A fast final checker does not erase preparation cost.
[TL] (§3, PDF p. 11 and associated experiment description.)

This review has not built or audited the external TorchLean repository. Exact
dependency versions, proof elaboration, generated artifacts and backend
conformance would be deliverables of a separate implementation evaluation.

### The proof bridge required for issue #12

[Issue #12][I12] should produce a candidate-specific evidence chain, with each
link naming the object and theorem it covers:

1. **Supported program.** Enumerate the actual framework operators, tensor
   shapes, forward semantics, preprocessing, mutable-state exclusions and
   decision rule. Reject unsupported constructs before generating a proof claim.
2. **Lowered artifact.** Specify the lowering/serialization map and bind the
   exact parent, candidate bytes, contract and input domain. A proof about a
   fixture does not prove a newly proposed runtime candidate.
3. **Mathematical instance.** Generate the theorem/certificate for the actual
   weights and declared property, including activation/domain premises,
   tolerances, protected span or bound assumptions, and input quantification.
4. **Independent replay.** Check the candidate-specific instance without
   trusting the proposer or an unchecked report. Retain failed elaboration,
   unsupported operators and unresolved numeric assumptions as explicit states.
5. **Native bridge.** Separately establish or test conformance for the target
   dtype/backend and operation order, including finiteness and decision margins.
   If this link is empirical, label it empirical rather than extending the
   formal theorem to the hardware by assertion.

A mathematical score-preservation theorem still has to be connected to the
deployed decision: grouped queue-score argmax, intent argmax followed by a queue
mapping, tie-breaking and abstention are different functions. Even a fully
checked bounded candidate theorem would not establish that the task distribution
is representative of customers or that the edit has business value. Those
remain statistical and operational evidence.

## TOFU: controlled entity forgetting and the limits of its metrics

### The controlled reference is the valuable design element

TOFU creates synthetic author profiles and introduces them through controlled
fine-tuning. It includes a retain-only model as the reference for behavior when
the designated forget entities were not included in that fine-tuning. The
original baseline experiments study Llama-2-7B and Phi-1.5 in the stated entity
forgetting setup. Exact forget and retain sets are provided; discovering all
instances of sensitive data in uncontrolled pretraining is not solved by this
benchmark. [TOFU] (task construction, §2, PDF pp. 3–8; methods, §3,
pp. 8–10; appendices.)

The dataset's 200 authors with 20 QA pairs each describe a construction. Those
QA pairs are dependent observations of the same entity, and they are evaluated
through models with shared training lineages. They are not 4,000 independent
unlearning experiments. Forget-set size, training run, method and audit query
all affect the independent unit in a new study. [TOFU] (dataset construction,
PDF pp. 4–6.)

The original setup studies entity-level forgetting. Instance, behavior and
in-context forgetting are different tasks. Supplying known forget entities also
does not establish that the synthetic facts were absent from every possible
foundation checkpoint. In particular, public 2024 fictitious data cannot by
itself justify an absence-from-pretraining claim for a 2026 model. A new
controlled-exposure study should use fresh synthetic canaries or secrets and
record their generation, introduction and exposure lineage. This is a framework
proposal, not a new claim about the TOFU authors' tested 2024 checkpoints.

### Truth ratio, utility and the statistical interpretation

In Formula (1), the truth ratio divides the **mean length-normalized likelihood
of perturbed false answers** by the **length-normalized likelihood of a
paraphrased true answer**. The direction matters: reversing the ratio changes
the metric. Evaluation should preserve the exact source definition and all
normalization choices, rather than merely reusing its name. [TOFU] (§2.2 and
Formula (1), PDF p. 7.)

“Forget quality” uses a two-sample Kolmogorov–Smirnov test on truth-ratio
distributions against the retain-only reference. A high p-value is a failure to
reject a difference under that test and sample. It is not a proof of equality,
distributional equivalence, privacy or erased internal information. Entity/run
clustering and statistical power matter if this test is used in a new study.
[TOFU] (§2.2.3, PDF pp. 7–8; Appendix A, p. 19.)

TOFU aggregates model utility as the harmonic mean of nine source-defined
metrics on retain, real-author and world-fact sets. A framework report should
retain those component values and subgroup harms, because aggregation can hide
different failure patterns. Section 5.2 explicitly notes that gibberish/random
output can look favorable on forget quality; useful retained behavior is
therefore indispensable. Matching a forget statistic while losing ordinary
capability is not a successful enterprise outcome. [TOFU] (utility definition,
PDF pp. 6–8; §5.2, p. 14.)

The paper's baseline failure and tradeoff conclusions apply to the methods,
models and settings studied. They motivate stronger evaluation without proving
that all future unlearning must fail. Conversely, an improved benchmark score
would not establish deletion of real confidential information from arbitrary
pretrained weights. [TOFU] (results/discussion, PDF pp. 10–16.)

### Reproduction discrepancies that should remain visible

Two ambiguities were checked against rendered PDF pages, rather than relying
on text extraction alone:

- The probability discussion on PDF p. 6 uses multiple-choice normalization
  for Real Authors and World Facts, while Table 1 on p. 7 prints a
  length-normalized conditional-probability expression for those rows. A
  reproduction must pin the code and actual normalization; this review does
  not invent a correction.
- Appendix E's expectation discussion on p. 22 says random/pretrained
  comparisons on forget sets should have high p-values. The displayed Table 4
  does not uniformly support that prose: its random comparison values are
  small and its pretrained values are `0.0031`, `0.0297` and `0.1650` for the
  displayed forget settings. The observed discrepancy should be recorded
  before interpreting these checks as a universal metric validation.

These are source-version reproduction questions, not claims that the whole
benchmark is invalid. They reinforce the need to preserve executable metric
definitions and reference-model lineage. [TOFU] (PDF pp. 6–7 and Appendix E,
p. 22.)

### Preparation and the unlearning study the framework can defend

The paper describes 1%, 5% and 10% forgetting settings, five epochs for the
relevant fine-tuning/unlearning settings, AdamW, an effective batch size of 32
and two A6000 GPUs. This does not provide a complete wall-clock or cost ledger
for data creation, full/retain-only model training, hyperparameter search and
all audit endpoints. A new study needs those costs as well as the final edit's
runtime. [TOFU] (§3 and training/evaluation appendices, PDF pp. 8–10 and 19–24.)

For [issue #18][I18], define a bounded threat model before optimization:

1. Introduce fresh synthetic forget material with controlled exposure and
   compare against an identically specified retain-only training reference.
2. Separate the optimization queries from the independent audit queries. Freeze
   attack access, query counts, tokens, prompts, sampling, compute and selection
   rules before inspecting audit outcomes.
3. Measure direct answering, paraphrase/generalization, extraction and bounded
   recovery attempts as different endpoints. Where deployment permits model
   transformation, include the declared transform and attacker access rather
   than assuming the original output behavior persists.
4. Measure retain utility, unrelated facts/behaviors, subgroup harm and cost.
   Include the unchanged model, output-suppression/refusal controls and the
   retain-only reference so that indiscriminate damage is visible.
5. Report uncertainty at entity and training-lineage levels. A failed finite
   attack supports a bounded empirical accessibility claim; it does not prove
   that the information has been universally erased.

Support routing itself is a behavior-change use case. Moving a request to a
different queue is not an unlearning result. The unlearning lane needs its own
data, references, adversaries and claims rather than borrowing the routing
study's retention metric.

## Cross-paper synthesis: the assumptions that should become framework fields

### Different contributions answer different questions

| Component | What the reviewed work supplies | Evidence still required for a framework claim |
| --- | --- | --- |
| Contextualized weight reading/writing | Original demonstrations with different access regimes | Independent repeated runs, metadata/weights ablations, complete attempt accounting and exact realization |
| Permutation alignment | DeepAlign's compatible-coordinate matching and scoped recovery theorem | Edit transport, causal target benefit, protected behavior and independent-model transfer |
| Learned weight readers/editors | DWS/NFN's symmetry-based constructions and prepared tasks | Fair simpler baselines, lineage-held-out performance, declared supervision and full preparation cost |
| Activation-specific graph processing | ScaleGMN's supported scaling structure and ideal expressivity construction | Practical-module/numerical bridge, unsupported-family rejection and measured task benefit |
| Shared semantic verification | TorchLean's explicit IR, scoped theorems, checkers and external boundaries | Candidate-specific binding/replay and the target framework/native implementation bridge |
| Controlled forgetting | TOFU's synthetic exposure, retain-only reference and multi-part evaluation | Independent extraction/recovery audits, utility and bounded claims under a declared threat model |

This table combines roles, not effect sizes. Sources with different tasks,
data, seeds, supervision, hardware and endpoint definitions cannot be treated as
a shared intervention meta-analysis.

### Preparation, adaptation and evaluation need separate budgets

For an end-to-end experiment, record at least:

`C_total = C_data + C_source_models + C_reader + C_proposal + C_realization + C_verification + C_evaluation + C_integration`.

Each term should include elapsed time, compute/hardware, memory/storage where
relevant, and the consumed information: examples, labels, target-model forwards,
gradients, activations, generated/teacher-forced tokens and feedback responses.
Human task/metadata preparation should be identified when it changes the method.
The same information can be exposed repeatedly, so unique items and total
exposure counts should be distinguished.

One-time reader/model preparation may be amortized, but the assumed number of
**usable admitted edits** and the update/retraining schedule must be stated.
Amortizing over every attempted edit hides the cost of failures; amortizing
over an unobserved future workload creates a projection. Report both measured
one-time preparation and marginal editing cost before projecting a deployment
scenario. These are framework accounting proposals, motivated by the source
training and timing boundaries reviewed above.

| Source | Preparation that a new experiment must budget | Reported marginal/training boundary |
| --- | --- | --- |
| Sparks | Semantic metadata, original models, labeled practice where supplied, sandbox/harness and all attempts | Charged passes are task-specific; batches and duration need separate accounting [SP] (AUDIT) |
| DeepAlign | Weight collections, synthetic permutation supervision, reader training and relevant task data/loss | Inference pair timing on one A100; displayed runtime units unresolved [DA] (Appendix E/F) |
| DWS | Per-image INRs, classifier/trajectory collections, reader supervision and adaptation optimization | Parameter scaling and task-specific schedules, not an enterprise total [DWS] (appendices I–K) |
| NFN | INRs, classifier trajectories, IMP masks and trained neural-functional readers | Titan RTX reader-training estimates exclude full collection/IMP preparation [NFN] (Appendix D/E) |
| ScaleGMN | Prepared/reused weight collections, tuning grids and supervised readers/editors | Graph traversal asymptotics exclude feature/depth/batch/storage factors [SG] (Appendix A.4/A.6) |
| TorchLean | Supported lowering, graph/proof/search construction, compilation and backend setup | Final replay/native timing is different from full proving/training workflow [TL] (§3; appendices) |
| TOFU | Synthetic entities, full and retain-only training, unlearning search and independent audits | Hardware/epochs are reported; complete wall-clock/cost accounting is not [TOFU] (methods/appendices) |

### Independence must match the claim

Training-run identity, source-object identity and edit-lineage identity should
be immutable metadata. A model zoo can contain several checkpoints of one
training run; an INR zoo can contain several parameterizations of one image;
an unlearning set can contain several questions about one entity; a policy
evaluation can contain thousands of episodes for one edit. Each is useful
measurement, but its repetitions are nested rather than universally independent.

Split and resample at the relevant cluster. Hold out lineages for reader
generalization, images for image-task generalization, entities for forgetting
queries and independent editing runs for editor repeatability. Preserve the
paired before/after comparison within a candidate and the variation between
candidates. Development-only selection and untouched evaluation are additional
requirements; a good cluster split does not cure adaptive holdout selection.
These are review-derived protocol requirements, not a new estimate of any
paper's sampling error.

### Symmetry preserves a represented function only under a declared action

For a supported linear action `g` on the parameter tensors, transporting an
update requires the corresponding action on both base weights and update:
`g(w + Δ) = g(w) + g(Δ)`. A reader/editor that is claimed equivariant should be
tested against its declared action. This identity is a coordinate condition;
target benefit and protected behavior still require their own evaluation.

Valid hidden permutation does not permit arbitrary changes to input/output
meaning. Valid ReLU scaling does not authorize arbitrary tanh scaling. Known
same-function transformations differ from inferred alignments between
independently trained functions. Real-arithmetic equality differs from a finite
native computation. These distinctions should live in adapter contracts,
rather than only in explanatory prose. See [DWS], [NFN] and [SG] for the
symmetry constructions, [DA] for perfect-alignment scope, and [TL] for numerical
boundaries.

### Retention has several incompatible meanings

A small weight update, a sparse patch, equality on sampled feature rows, low
interpolation loss, low reconstructed-image MSE, unchanged benchmark accuracy,
and a verified bound over a specified domain are different statements. None
should be used as a shorthand for all retained behavior.

For a fixed readout and protected features `H_P`, exact score preservation can
use `ΔW H_P = 0`. If those features span the full editable feature space, the
exact-score lane may permit only a no-op. If they do not, held-out features may
still leave their span. This is the framework's stated geometry boundary in
the [roadmap](../weight-space/v2/research-roadmap-2026-11.md), not a new
experimental result or a theorem supplied by DeepAlign/DWS/NFN. Decision-level
retention can use a different declared margin contract, but the change of
contract must be explicit.

Report target gain, protected-score residuals, protected decisions, nearby
confusions, admission/coverage and abstentions separately. A conservative
certificate that rejects a safe edit is a coverage limitation; it is not an
observed behavioral counterexample. A passing algebraic constraint is a
mathematical result under its inputs; it does not establish distribution-wide
customer retention.

## A concrete generic-framework study: trained support routing

The [roadmap's first trained study](../weight-space/v2/research-roadmap-2026-11.md#the-first-trained-enterprise-study)
uses public intent queries with an explicitly derived queue policy. Dataset
intents are not actual enterprise queues or customer telemetry. An ownership
change, a selective representation repair and an out-of-scope routing policy
should be separate cases, rather than one convenient aggregate.

Consider a query recognized as a card-service intent. A policy changes the queue
that owns that intent. If the original intent prediction is already right, an
unchanged intent-to-queue overlay may solve the request exactly and cheaply.
If the model instead confuses that intent with a nearby one, the problem is
recognition/representation, and a head edit or retraining may have different
value. The framework's role is to make that decision assessable; it should not
require model editing to win the ownership-change case.

The study should freeze the model/data lineage, eligible task metadata, queue
mapping, protected cases, argmax/tie/abstention rules, useful-effect margins and
selection rules before confirmatory evaluation. Every comparator should see
its declared eligible development information. Count training and adaptation
cost separately where identical budgets are infeasible, and compare useful
gain/harm/cost frontiers rather than a single cost-free score.

| Comparison | Question it answers | Relevant lesson from the review |
| --- | --- | --- |
| Unchanged model | What actually needs improvement? | Baseline behavior and model lineage must be explicit |
| Queue overlay/rule | Is a policy configuration sufficient? | A non-editing solution is a valid outcome |
| Target-only fitted edit | How much target improvement is available without the retention constraint? | Gain and retention are separate endpoints |
| Soft-retention edit | What tradeoff is available under a declared tolerance? | Near-preservation is not exact preservation |
| Hard protected-span edit | Does geometry admit a useful realized candidate? | Feasibility/no-op/abstention cannot be discarded |
| Tuned head/appropriate larger retraining | Is editing preferable to ordinary fitting? | Reader/model preparation and tuned comparator cost count |
| Context-only versus weights/summary/reader proposer | What information do weights add? | Original source metadata and hints are actual access conditions |
| Candidate-specific proof/native checks | What property covers the deployed artifact? | Semantic, mathematical and hardware links are distinct |

A routing result should report paired target queue gain, protected/nearby
misrouting harm, coverage, no-ops, abstentions, uncertainty across independent
lineages/cases and complete workflow cost. Customer time saved, incident
avoidance or financial impact needs authorized operational data and a separate
study. A public benchmark routing benefit does not supply those business
quantities.

No sealed trained outcomes are asserted in this review. This section describes
the study and the literature-derived interpretation rules. The same artifact,
access, comparator, retention and cost interface can later support a trained
non-text task while that task supplies its own semantics and evaluation.

## Evidence obligations for the five research issues

| Issue | Source-backed starting point | Proposed deliverable | What could falsify the intended claim | Stronger claim still excluded |
| --- | --- | --- | --- | --- |
| [#12: candidate-specific Lean instances][I12] | TorchLean's shared IR and explicit proof/checked/external split | Exact candidate/contract binding; supported lowering; independent instance replay; separate native conformance | Proof applies to different bytes/domain; unsupported operator; unresolved derivative/numeric premise; native decision mismatch | Universal verification of arbitrary user programs, backends or all customer inputs |
| [#15: selected original-task repetition][I15] | Sparks demonstrations and the prior selected-release audit | Source-matched task/access definition; exact realization; original metric reproduction; independent repeated runs; declared deviations | Candidate cannot reconstruct; metric differs; allowed access differs; effect fails under the stated comparator | General repeatability across all reported tasks or a pooled success rate |
| [#16: symmetry-aware transfer][I16] | DeepAlign matching; DWS/NFN permutations; ScaleGMN activation-specific transformations | Known-symmetry positive controls; independent-model transfer; unaligned/random/sham/independently fit controls; target and retained outcomes | Function-preserving control fails; alignment helps distance but not target behavior; retained harm defeats utility; advantage disappears under matched controls | Causal circuit equivalence or cross-architecture grafting |
| [#17: no-fitting/zero-data construction][I17] | Original restricted Go arm; prepared DWS/NFN/ScaleGMN readers | Separate restricted/fitted arms; operation-level access rules; common budget ledger; no-feedback construction versus fitted controls | Undeclared labels/forwards/fitting; metadata-only control matches benefit; construction fails or adds no useful gain | Zero-data steering of arbitrary pretrained models or equivalence between no gradient descent and no fitting |
| [#18: synthetic unlearning][I18] | TOFU controlled exposure/reference; source-defined metrics and their limits | Fresh synthetic exposure; retain-only reference; independent extraction/recovery audits; threat/budget manifest; utility and transformed-deployment checks where applicable | Ordinary or attack queries recover targets; apparent forgetting is indiscriminate damage; reference/metric lineage is invalid | Universal erasure, a privacy theorem, or deletion of real pretraining data |

These deliverables are research gates, not a checklist whose length measures
capability. A careful negative result, explicit unsupported scope, or demonstrated
superiority of a simpler policy overlay is a useful outcome. Issue implementation
and the final empirical/proof claim can have different dependency timing, as
the [work index](../weight-space/v2/research-issues-2026-11.md) explains.

## Reproduction questions that remain open

This full-text review leaves the following questions unresolved rather than
assigning guessed answers:

- **DeepAlign timing:** no time unit was found in the visually inspected
  Figure 5/Table 6 labels. Confirm the timed release code or author clarification
  before quoting precise seconds or speedups.
- **ScaleGMN initialization:** resolve `0.01` versus `0.001` against a pinned
  implementation/configuration and preserve the version-specific discrepancy.
- **TOFU probability normalization:** resolve the prose/Table 1 difference with
  the pinned evaluator and reference checkpoint lineage.
- **TOFU metric expectation:** retain the Appendix E prose/Table 4 discrepancy;
  distinguish the actual p-values from an expectation stated in prose.
- **TorchLean reuse:** a separate build/proof/native-conformance evaluation is
  needed before claiming that its external implementation validates this
  framework's operators or runtime candidates.
- **Original reproduction:** the prior source archive is a selected-artifact
  audit with documented acquisition/coverage limits, rather than a new exhaustive
  audit or an independent rerun.

Resolving these questions improves reproducibility. It does not replace the
independent experiments required to establish useful selective edits.

## Byte identities and primary references

Hashes below are SHA-256 of the reviewed raw original HTML or pinned PDF, not
the extracted reading text. Fresh official retrievals matched these identities.
The paper files and private acquisition receipts are not embedded in this
document.

| Source | Reviewed bytes | SHA-256 |
| --- | ---: | --- |
| Sparks author HTML | 85,874 | `a326bc5b34b3152a93bf38013f412bb140c9e0a299c72491d27fb17b1b7d9cbb` |
| DeepAlign `2310.13397v4` PDF | 5,753,705 | `9882f534572d8cc420c363c68be7bf5109ec0a4b3c2c1b91d660a5ca7124c889` |
| DWS `2301.12780v2` PDF | 5,291,953 | `a685648be137c17bf20d0cd71317f08f24240d40ace17bfbb85992473f684f6e` |
| NFN `2302.14040v3` PDF | 5,183,219 | `068640b6b12e90ad82729aa82759131cca400e1bbb2d4839c7c895be33e55a2a` |
| ScaleGMN `2406.10685v2` PDF | 1,343,101 | `4425b55bc760977e91c0de9e7ee301bc334cea8e8fa4b9f0b413ab66b0b11957` |
| TorchLean `2602.22631v2` PDF | 7,762,633 | `dd8b7ce74b70d6a5bd928bf32f5caaeaf9354b56902d8f9a47f6f14f77c0c4d3` |
| TOFU `2401.06121v1` PDF | 1,693,823 | `b747069131e44f1f07b8984dc2762ecc0c2e84ddf25af029a0ca3c1476b61a4b` |

For the original released-task artifacts, the prior audit pins
`RikiyaT/lm_understand-logs` at commit
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` and records its acquisition and
coverage limits. This review inherits only those stated observations. It does
not claim that an unreviewed downloaded artifact, external code path, or omitted
trial has now been verified. [AUDIT] (“Sources and review boundary.”)

[SP]: https://rikiyat.github.io/blog/weight-space-programming/
[DA]: https://arxiv.org/pdf/2310.13397v4
[DWS]: https://arxiv.org/pdf/2301.12780v2
[NFN]: https://arxiv.org/pdf/2302.14040v3
[SG]: https://arxiv.org/pdf/2406.10685v2
[TL]: https://arxiv.org/pdf/2602.22631v2
[TOFU]: https://arxiv.org/pdf/2401.06121v1
[AUDIT]: ../weight-space/v2/source-review.md
[I12]: https://github.com/shyamsridhar123/weight-space-programming-framework/issues/12
[I15]: https://github.com/shyamsridhar123/weight-space-programming-framework/issues/15
[I16]: https://github.com/shyamsridhar123/weight-space-programming-framework/issues/16
[I17]: https://github.com/shyamsridhar123/weight-space-programming-framework/issues/17
[I18]: https://github.com/shyamsridhar123/weight-space-programming-framework/issues/18
