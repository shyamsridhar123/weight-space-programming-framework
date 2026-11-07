# Methods and frozen study protocol

The second study asks a narrow editing question: **can an update to a model's output head make a requested change in one input region while limiting change in another?** It then tests three reasons that protection can fail: incomplete anchor coverage, native rounding, and interference from later edits.

The mechanism is explicit. Each model maps an input to frozen nonlinear features, then uses a trainable linear head to produce three scores. The editor changes that head. Target-only fitting rewards the requested output change. Soft retention adds a penalty for changing other outputs. Numerical-nullspace fitting instead restricts the update to directions that approximately preserve specified feature rows.

The study compares deterministic algorithms with access to exact weights, public activations, and validation feedback. Least-squares fitting is allowed. No live LLM or provider is involved. It therefore tests the editing mechanism and its evidence records, not an LLM's ability to infer mechanisms or write useful weights. This access regime also differs from the original report's Go experiment, which forbids regression and least squares.

The executable [protocol](protocol.json), [runner](../../../rlcd/weight_space/study_v2.py), and [fixtures](../../../rlcd/weight_space/benchmarks.py) define the procedure. The [results](results.md) report its outcomes. The [source review](source-review.md), [requirements](requirements.md), and [limitations and next steps](limitations-and-next-steps.md) connect the procedure to the broader research questions.

## Questions, scope, and evidence levels

The study keeps each question attached to evidence that can answer it:

| Question | Evidence collected | Interpretation limit |
| --- | --- | --- |
| Did the submitted program produce the intended stored candidate? | Exact parent identity, proposal, literal program, lowering receipt, and endpoint hashes. | Reconstruction does not establish useful behavior or native execution equivalence. |
| Did the candidate obey parameter restrictions? | Allowed tensors, shape, changed cells, endpoint magnitude, and per-cell change. | Structural admission does not establish a behavioral contract. |
| What changed on protected feature rows? | Exact rational endpoint residuals and numerical errors. | Supplied rows need not cover unseen features; the extractor remains a separate dependency. |
| What can be bounded on the full input box? | A real-valued sequential ReLU drift certificate for the supported fixture. | A valid bound can be loose. It does not prove floating execution, classification utility, or reward. |
| Does the edit work on new examples from this generator? | Held-out target and retention errors, plus paired contrasts across worlds. | These worlds do not represent real tasks, large pretrained models, or arbitrary architectures. |
| Can the workflow retain different task outputs and access histories? | Read, diagnose, edit, and create records; candidate seals and budget traces. | A cooperative trace is neither hostile-code isolation nor execution attestation. |

A no-op is a necessary control: it preserves every output but makes none of the requested change. A nonzero edit can pass structural admission and fail retention. A residual checker can correctly report a small nonzero value rather than exact preservation. Each remains a valid recorded outcome.

## Freeze, execution, and provenance

Before the reserved seeds were consumed, the study copied its required protocol, implementation, tests, and review inputs into a local source snapshot. The [freeze receipt](../../../artifacts/weight-space/study-v2-freeze/receipt.json) binds the required path set, bytes, environment, and snapshot identities. The runner verifies both the snapshots and execution sources before and after the run. It rejects a manifest that is internally consistent but omits a required input.

Development used different seeds, starting at 900001. The published study uses **7300–7311**. The freeze provides local integrity evidence, not external preregistration or third-party attestation. All published seeds are now observed. A run with those seeds reproduces the procedure; it cannot supply untouched confirmation data.

The [execution record](../../../artifacts/weight-space/study-v2-execution/run-command.json) retains the command, interpreter, elapsed time, exit status, and log identities. Saved evidence includes the worlds, trained endpoints, proposed and realized candidates, selection records, auxiliary cases, and a closed file manifest. Failures remain in the record. Changing source or protocol requires a new identity.

The historical receipts bind the source state used for that run. The source-review document was one frozen input. Strict execution checks reject its rewritten live version against the historical freeze. Its [original publication copy](../../../provenance/original-documents/docs/weight-space/v2/source-review.md.txt) and frozen snapshot retain the old bytes. Saved-data replay can report this current-source difference without changing the historical result; a fresh execution requires a fresh freeze. The [evidence guide](../../../EVIDENCE.md) and [reproduction guide](../../../REPRODUCING.md) identify the preserved records and current procedures.

To repeat the procedure from the repository root, create a new freeze and use unused output paths:

```powershell
.venv/Scripts/python.exe -m rlcd.weight_space.study_v2 freeze --protocol docs/weight-space/v2/protocol.json --out artifacts/weight-space/my-v2-freeze
.venv/Scripts/python.exe -m rlcd.weight_space.study_v2 verify-freeze --protocol docs/weight-space/v2/protocol.json --freeze artifacts/weight-space/my-v2-freeze/receipt.json
.venv/Scripts/python.exe -m rlcd.weight_space.study_v2 run --protocol docs/weight-space/v2/protocol.json --freeze artifacts/weight-space/my-v2-freeze/receipt.json --out artifacts/weight-space/my-v2-study
```

Evidence writers refuse overwrite. Keep rendering and auditing outputs outside the study directory. A confirmation study needs new seeds or task lineages and an analysis fixed before their outcomes are observed.

## Independent units and data construction

There are **12 independently generated worlds**. Each world has its own teacher, input streams, and initialization streams. It supplies three architecture variants, giving **36 checkpoints**. The variants share their world's teacher and split inputs, so they are paired comparisons within a world. Uncertainty is computed across worlds.

Inputs have 16 independent coordinates, uniform on \([-1,1]\). The convolution fixture reads the same vector as a single-channel \(4\times4\) field. The noiseless teacher produces three continuous scores:

\[
y(x)=xL+0.2\sin(xN).
\]

Both \(L\) and \(N\) are \(16\times3\) matrices with independent normal entries divided by four. Each split uses a separate seed stream and the same teacher within its world.

| Split | Rows per world | Use |
| --- | ---: | --- |
| Training, whole box | 256 | Fit the original output head. |
| Public target / public retention | 64 each | Fit candidate edits and define retention anchors. |
| Validation target / validation retention | 64 each | Compare the six fixed arms and select once. |
| Test target / test retention | 128 each | Evaluate all fixed arms after selection and sealing. |
| Sequence public, three regions | 32 per region | Construct both complete three-edit chains. |
| Sequence test, three regions | 32 per region | Evaluate all stages after both chains are sealed. |

These splits contain 960 declared input rows per world and 11,520 overall. Auxiliary read/create tasks have separate inputs. The main test rows are reused across architectures and ablations; the sequence test rows use separate streams. Repeated evaluation of a row does not create another independent observation.

The main target region is \(x_0>0.35\); the retention region is \(x_0<-0.1\). Rejection sampling gives conditional uniform inputs in each region. The gap \(-0.1\leq x_0\leq0.35\) is unmeasured by the main task. The full-box ReLU bound is a separate statement that also covers this gap.

Sample IDs encode world, split, and row. The workflow rejects overlaps in declared IDs or lineage records. It does not detect duplicates by content or prove independence. Here, separate streams and saved arrays support the split description. External datasets would require their own provenance checks.

## Architectures and initial training

| Family | Computation | Frozen feature width | Total parameters | Editable head and bias |
| --- | --- | ---: | ---: | ---: |
| `relu_mlp` | 16 → 24 ReLU → 3 linear | 24 | 483 | 75 |
| `tanh_mlp` | 16 → 20 tanh → 24 tanh → 3 linear | 24 | 919 | 75 |
| `conv_tanh` | Six valid \(2\times2\) filters, tanh, flatten \(6\times3\times3\), linear head | 54 | 195 | 165 |

Hidden weights and biases are randomly initialized, then frozen. Only the output head is trained. These models are readouts over random features; they do not have end-to-end trained representations.

Write the frozen feature map as \(\phi(x)\in\mathbb R^d\), augment it with a bias coordinate \(z(x)=(\phi(x),1)\), and collect the head and bias in \(B\in\mathbb R^{(d+1)\times3}\). Initial training solves

\[
\min_B\;\frac{1}{n}\|ZB-Y\|_F^2+0.03\|B\|_F^2.
\]

The bias is regularized along with the other head coordinates. Saved diagnostics include the design rank, condition estimate, and fit error. The solver and feature computation are trusted numerical components; neither is formally verified. The saved-data audit checks the retained endpoints and their outputs, not the optimization history that produced them.

The fixtures have explicit executable semantics. Sequential fixtures require an identity final activation. The convolution fixture checks shapes, activation, flattening order, and metadata. These checks define the supported models; they do not infer the meaning of arbitrary network code.

## Requested change and six fixed arms

On target inputs, the requested score shift is

\[
s=(0.6,-0.3,0).
\]

On retention inputs, the request is zero change. Both requests are relative to the original checkpoint's output. They do not ask the editor to remove its existing approximation error against the teacher. Teacher-relative errors are reported separately.

Let \(A\) contain public target rows \(z(x)\), \(R\) contain public retention rows, \(D\) repeat the target shift, and \(\Delta\) be the augmented head update. The six arms use public fitting inputs only:

| Arm | Construction | Comparison it enables |
| --- | --- | --- |
| No-op | Empty literal program. | Exact preservation and no target progress; legal selection fallback. |
| Unconstrained target fit | Ridge target fit with retention weight zero. | Target emphasis under the fixed ridge objective. |
| Soft retention | Same fit with retention weight 8. | Target cost and retention benefit of the added penalty. |
| Numerical nullspace | Fit within the numerical kernel of the first 12 retained design rows. | Protection at anchors versus behavior on unseen retention inputs. |
| Matched random | One normal direction per checkpoint, with requested support and L2 matched to the realized soft update. | Whether the fitted direction matters at comparable requested size. |
| Gradient | 200 fixed steps on the soft objective. | A conventional iterative solver for the same head problem. |

The soft objective is

\[
J(\Delta)=\frac{1}{|A|}\|A\Delta-D\|_F^2
+8\frac{1}{|R|}\|R\Delta\|_F^2
+0.01\|\Delta\|_F^2.
\]

The first term fits the target shift. The second penalizes retention drift. The third limits the update through ridge regularization, including bias. The target-only arm removes the second term.

The nullspace arm uses SVD with threshold \(10^{-12}\sigma_{\max}\). Singular values strictly above that threshold count as nonzero. The remaining right-singular directions form \(Q\), and the solver fits the target-plus-ridge objective over \(\Delta=QU\). If no direction remains, it returns a no-op. Floating solving and endpoint realization can still leave nonzero anchor residuals. The exact rational checker measures those residuals from the actual stored endpoints and supplied feature rows.

For the gradient arm, define

\[
H=A^\top A/|A|+8R^\top R/|R|+0.01I,
\qquad C=A^\top D/|A|.
\]

Starting from zero, the runner takes 200 steps of \(\Delta\leftarrow\Delta-\eta(H\Delta-C)\), with \(\eta=1/\lambda_{\max}(H)\). This is a fixed control, not a tuned optimum or full-network fine-tune. Direct fitting and iterative optimization have different costs; proposal counts do not make them compute matched.

The random control samples on the soft candidate's changed head/bias coordinates. It scales the requested update to the L2 norm of the realized soft candidate-minus-parent delta. Addition to the parent can round entries, so realized support and norm are measured separately. One random direction per checkpoint supplies a control, not an estimate of every plausible random edit.

Every proposal becomes a literal program bound to the exact parent. The runner verifies lowering and feature-retention receipts before registering a successful candidate. Failed attempts remain in the record but cannot enter selection or held-out arms. These checks establish what was realized; they do not certify solver optimality.

## Structural checks, selection, and held-out evaluation

Structural admission permits only the declared head and bias. It caps changed cells at the parent parameter count, endpoint magnitude at 100, and per-cell change at 10. Shape and semantic compatibility are also required. These restrictions do not bound behavior. The pre-existing `Contract` API still specializes a two-layer ReLU family; generic feature and sequential certificates are separate facilities.

All six arms receive validation feedback. To be eligible, a candidate must pass structural admission, have validation retention NRMSE ≤0.35, and have validation target NRMSE ≤1. Selection minimizes target error, then delta L2, then the lexical arm name. The no-op remains eligible as the fallback.

The runner saves selection with `heldout_observed: false` and seals the chosen candidate. It then evaluates every successfully registered fixed arm on both test regions. The selected outcome aliases one of those arms; it is not a seventh independent candidate. Validation eligibility does not guarantee test eligibility, so any test violations must remain visible.

Sealing enforces this order through the cooperative API. The runner still owns the world and evaluator in one Python process. The procedure does not provide a secure data firewall. A live access-controlled editor experiment needs a separate evaluator and a stronger enforcement boundary.

## Metrics and denominators

Let \(\delta(x)=f_{\mathrm{edited}}(x)-f_{\mathrm{base}}(x)\) and

\[
q=\sqrt{(0.6^2+(-0.3)^2+0^2)/3}=\sqrt{0.15}.
\]

RMS averages over examples **and all three output coordinates**. The two main errors are

\[
E_T=\frac{\mathrm{RMS}(\delta(x)-s)}q
\quad\text{on target inputs},\qquad
E_R=\frac{\mathrm{RMS}(\delta(x))}q
\quad\text{on retention inputs}.
\]

Both are dimensionless and lower is better. A no-op gives \(E_T=1,E_R=0\). A perfect shift applied everywhere gives \(E_T=0,E_R=1\). These reference cases explain why target accuracy and preservation need separate metrics.

Directional gain is

\[
g=\frac{\mathbb E_x[\delta(x)^\top s]}{\|s\|_2^2}.
\]

Gain 1 means the mean projection equals the request. It does not imply low target error, pointwise success, or absence of change in other directions. Gain can be negative or exceed 1.

The report also records changed cells, realized parameter L2, identity-based no-op status, and raw RMSE against the teacher or shifted teacher. Scaled RMS and L2 calculations avoid avoidable overflow and underflow; quantities outside the finite evaluator range are rejected. Attempted, registered, admitted, and evaluated denominators remain separate. Sign decisions and formula creation are different outcomes and are not pooled with edit metrics.

## Access and computational accounting

The interface records weight access, public activation calls, proposals, validation feedback, and tokens. It charges metadata and returned candidates for captured weight values as well as individual tensor reads. Candidate identity includes task and proposal context, so two workflow candidates can share artifact bytes without being the same candidate.

For one complete main checkpoint, two public activation requests cover 128 examples, six feedback requests submit 768 examples, and six proposals are registered. Actual model/example executions are a different denominator. Validation evaluates the base and candidate on both regions for every arm: 1,536 model/example executions per checkpoint. Held-out evaluation uses 3,072. The selected alias requires no extra execution.

Counters for submitted examples and executed model/examples overlap in meaning and must not be added as disjoint compute. Feature extraction, initial training, fitting, memory, and ablations need separate accounting. A single held-out callback can perform substantial work.

Validation records retain attempted and completed callbacks and whether execution counts are complete. If a callback fails internally, it may perform work that the returned counter cannot recover. This is cooperative accounting, not automatic instrumentation of everything the process does.

Provider calls and input/output tokens are zero. This describes the deterministic arms. It says nothing about the token cost or performance of a live LLM trial.

## Feasibility and anchor-span ablation

The nullspace compiler uses 4, 12, or 64 retained rows. Bias augmentation gives 25 coordinates for either MLP and 55 for the convolution fixture. Each condition saves numerical rank/nullity, the realized endpoint, exact maximum anchor residuals, an exact-zero flag, and held-out target/retention errors.

This ablation asks how much edit freedom remains as anchor rank grows. Full numerical rank removes every kernel direction and returns a no-op. A smaller anchor set can permit the target change while leaving unseen retention drift. Neither result alone is a compiler failure; the first limits feasibility and the second limits coverage.

The real-valued theorems state preservation on an exact feature span under their premises. They also bound approximate residuals and off-span cases. This experiment provides neither domain-wide span witnesses nor a symbolic rank certificate. Numerical SVD rank and a small realized residual cannot be substituted for those premises.

## Native precision ablation

For each format—float64, float32, float16, and bfloat16—the runner first projects **all parent tensors** to that format. It lifts the stored values into the reference artifact, giving each format its own parent identity. The soft compiler computes features in lifted binary64 semantics and lowers its update with explicit native rounding.

The evidence records requested, touched, effective, and rounded cells; exact endpoint rounding error; endpoint retention residuals; and lifted/native target and retention errors. PyTorch executes the native parent and candidate on the same saved test inputs, on CPU with one thread. The runner also measures maximum native-versus-lifted discrepancies for base outputs, candidate outputs, and edit drift.

This is a measurement of one implementation on sampled inputs. The format rows are not the same-parent intervention with only an output dtype changed. They provide no worst-case IEEE bound, GPU claim, or proof of correspondence to the real-valued theorem.

## Repeated edits, composition, and interference

The sequence requests three changes in disjoint regions:

| Region | Requested shift |
| --- | --- |
| \(x_0<-0.5\) | \((0.6,-0.3,0)\) |
| \(|x_0|<0.2\) | \((0,0.45,-0.3)\) |
| \(x_0>0.5\) | \((-0.3,0,0.6)\) |

Both target-only and soft retention construct their complete chains before sequence holdout access. At step \(i\), the desired output in the new region remains the **original model plus that region's shift**. The incremental fit subtracts the current output to account for previous drift. Soft retention penalizes changes to the **current** output in the other two public regions; target-only fitting omits that penalty. This is not a cumulative preservation contract against the original model.

Every stage is saved. Literal composition must reproduce the final stored endpoint, and rollback must reproduce the original artifact hash. Those checks concern parameter identities. They cannot reverse deployed actions, external state, or stochastic trajectories.

Desired-behavior error is measured for all regions at every stage. Before a region's own edit, its desired change is zero; afterward, it is the region's prescribed shift. For a previously edited region, interference measures drift from its outputs immediately after its own edit. Each error uses that region's prescribed-shift RMS as denominator, including before acquisition.

Final summaries average target errors across three regions and interference across the first two. Low interference can preserve a poorly acquired behavior. Target error must therefore accompany it. Three edits in one fixed order test bounded interference, not indefinite accumulation or absence of forgetting.

## Read, diagnose, and create fixtures

Each world also contains four affine-sign decisions and two formula-creation cases. The sign cases cross read/diagnose routes with exact/three-decimal access. The designated coefficient is \(+0.0004\) or \(-0.0004\). Exact access reveals its sign; rounding displays `+0.0` or `-0.0` and preserves the sign bit.

The task instructs the fixed reader to abstain at either displayed zero. The reader follows that rule without using the sign bit. The fixture therefore tests prescribed abstention and response reconstruction, not information-theoretic loss of the sign or learned detection of infeasibility. Correct sign, incorrect sign, and abstention remain distinct outcomes. A coefficient-sign task also does not establish general mechanistic diagnosis.

Creation uses analytic small ReLU networks for supplied absolute-value and clipping formulas. Each case saves zero, seeded-random, and analytic candidates; the analytic choice is fixed before hidden evaluation. These dyadic fixtures provide transparent positive controls for the creation route. They use no fitting, feedback, provider, or language model, and they make no claim of autonomous synthesis from unrestricted language.

## Statistical analysis and failure handling

The protocol fixes four paired contrasts:

1. Target-only minus soft-retention collateral NRMSE.
2. Soft-retention minus target-only target NRMSE.
3. Matched-random minus soft-retention target NRMSE.
4. Gradient minus soft-retention target NRMSE.

For each contrast, the three family differences are averaged **within each world first**. The resulting 12 world values are resampled with replacement 10,000 times. Bootstrap seeds are 11981 plus contrast index. Interval endpoints are the 2.5th and 97.5th percentiles.

Positive contrasts respectively mean less collateral drift with soft retention, a target-error cost for that retention, better target accuracy than the random control, and better target accuracy than the fixed gradient control. The first two test a tradeoff and must be interpreted together. The fourth does not test retention, total objective value, or compute efficiency.

These are exploratory percentile intervals, without multiplicity correction, from 12 independent worlds. They quantify variation under this generator. They are not confirmatory population claims about neural architectures, real tasks, or LLM editors. Precision and feasibility reuse main test inputs and remain descriptive; sequences use distinct inputs from the same world lineages.

Training, main arms, ablations, and sequence components retain separate failures. An ablation failure does not erase a main outcome. Paired inference requires the exact declared family set and all six main arms without main-arm failures. Other available outcomes remain in descriptive tables. Overall completion also requires the auxiliary cases. Missing outcomes are neither successes nor completed negatives, and complete-case analysis does not remove bias from informative failures.

## Mathematical and empirical boundaries

The [formal extension](formal.md) proves frozen-feature algebra and finite-depth coordinatewise drift for supported real-valued operations. Its direct examples are not generated from these study models. Runtime checkers use exact rationals on lifted stored coefficients and feature rows. The extractor, numerical optimizer, native kernels, protocol code, and runtime-to-Lean binding remain separate dependencies.

Only the ReLU soft-retention arm receives a full-box sequential runtime certificate here. Tanh and convolution execute in the study, but lack corresponding all-layer runtime certificate support. Feature-retention checks apply to supplied rows from all three extractors under the declared frozen-head decomposition. Storage and edit support therefore extend beyond the formal execution boundary.

The main comparison can support a retention tradeoff on these worlds if lower collateral error accompanies a measured target cost. It cannot establish useful editing on new task populations. That claim would require untouched tasks and lineages; failure there would narrow or overturn generality. Cumulative preservation would require all earlier conditions to remain satisfied after later edits; exact composition cannot substitute for that outcome. A claim of native numerical correctness would require a correspondence argument or bound, not a small discrepancy on sampled inputs.

The [next-step program](limitations-and-next-steps.md) specifies that further work: integrated behavioral admission, runtime proof binding, extractor and native-arithmetic correspondence, enforced access controls, live editor trials, longer sequences, and scalable storage and solvers.
