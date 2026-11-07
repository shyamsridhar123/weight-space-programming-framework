# Formal guarantees for generic weight-space programs

The formal library answers two useful questions. When an edit changes only a
linear head, which inputs can it leave unchanged? When an edit changes several
layers, how much can the output move? The answers are conditional equalities
and bounds over real numbers. Their assumptions identify the evidence an
application must supply.

Neither question depends on a game. Inputs can be observations, images, text,
feature vectors, or any other mathematical type. Outputs can be scores or other
real-valued coordinates. An application must still define their meaning. The
historical RTS scorer is one instance; it does not define the framework's
scope.

This guide explains the checked statements and their limits. The theorem
signatures in the linked Lean files are authoritative. [PROOFS.md](../../../PROOFS.md)
describes the pinned build, historical evidence, archive mapping, and portable
reproduction procedure. The [research program](limitations-and-next-steps.md)
explains how to close the remaining artifact, execution, and empirical gaps.

## A small edit that exposes the problem

Consider the scalar head `y = w · φ`, with parent weights `w = (1, 1)`.
The target has feature vector `t = (1, 1)`, and its requested output change is
`+1`. The protected input has feature vector `p = (1, 0)`.

| Candidate | Weight change `δ` | Target output | Protected output |
| --- | --- | ---: | ---: |
| Parent | `(0, 0)` | `2` | `1` |
| Target-only edit | `(0.5, 0.5)` | `3` | `1.5` |
| Protected edit | `(0, 1)` | `3` | `1` |

Both edits achieve the requested target change. Only the second has
`δ · p = 0`. Its protection comes from that equation. Counting changed
parameters would not establish it.

This is an illustrative calculation, not a new Lean witness or an empirical
study result. The general theorems below explain the same mechanism. They also
explain why the request becomes impossible if the protected feature equals
the target feature, and why approximate protection needs more than a small
anchor error.

## Choose the semantic family before applying a theorem

| Question | Supported mathematical family | Main source |
| --- | --- | --- |
| What does a head edit change? | A real-valued linear head over any fixed feature map with finitely many coordinates | [Generic.lean](../../../formal/weight-space/WeightSpace/Generic.lean) |
| How does parameter drift propagate through several layers? | A finite affine/activation chain with shared coordinates and activations satisfying the stated bounds | [Sequential.lean](../../../formal/weight-space/WeightSpace/Sequential.lean) |
| Can several drift contracts compose, or preserve a winner? | Functions on a common domain; declared output budgets and strict margins | [Contracts.lean](../../../formal/weight-space/WeightSpace/Contracts.lean) |
| What do the guarantees look like with actual constants? | Hand-written rational examples in the two families above | [GenericExamples.lean](../../../formal/weight-space/WeightSpace/GenericExamples.lean) |

The frozen feature map may be nonlinear. A fixed neural backbone followed by a
linear head can fit this abstraction if the feature correspondence is
established. It does not follow that edits inside that backbone are covered.
Similarly, arbitrary finite depth in the sequential theorem does not supply
semantics for every neural operator.

The [source review](source-review.md) explains why these distinctions matter.
The original report discusses sparse editing, deeper architectures, diagnosis,
and repair. The library addresses specific mathematical obligations arising
from those ambitions. It does not replicate the source experiments. In
particular, the executable constrained least-squares proposer uses a
fitting-permitted regime; the source Go protocol explicitly forbids regression
and least squares.

## From a request to an actual endpoint

Let `X` be any input type, `F` a finite feature-coordinate type, and `O` an
output-coordinate type. Define a fixed feature map and a head:

```text
φ : X → (F → ℝ)
W : O → F → ℝ

h_W(x)[o] = Σ_f W[o,f] φ(x)[f]
```

The same `φ` must be used before and after the edit. Bias can be represented by
a constant feature equal to one. Its coordinate, ordering, and correspondence
with the stored bias must be declared. Having a bias tensor in a checkpoint
does not establish that correspondence.

There are two ways to describe a change:

```text
edit(W,D) = W + D
endpointDelta(W,Q) = Q − W
```

`D` is the proposed real addition. `W` and `Q` are the two realized endpoints.
The library proves:

```text
edit_exact:
  h_(W+D)(x)[o] − h_W(x)[o] = Σ_f D[o,f] φ(x)[f]

endpoint_exact:
  h_Q(x)[o] − h_W(x)[o] = Σ_f (Q[o,f] − W[o,f]) φ(x)[f]
```

For the small example, the realized protected edit has `Q = (1, 2)` and
`Q − W = (0, 1)`. The target changes by `(0, 1) · (1, 1) = 1`, while the
protected input changes by `(0, 1) · (1, 0) = 0`.

If native storage rounds `W + D` to a different `Q`, use `Q − W` in the second
identity. A claim about the requested `D` does not automatically describe that
endpoint. This distinction also applies after factor multiplication, clipping,
projection, or another realization step.

The related identifiers `effect_add`, `edit_compose`, and `edit_rollback` prove
the real-arithmetic edit algebra. In particular, adding an edit and its
negative restores the real head. They do not prove that floating addition
followed by floating subtraction restores checkpoint bytes. Artifact rollback
requires checking the restored artifact's identity.

## Exact protection through a feature span

Protecting a finite set of anchors can protect more inputs, but only when
their feature relationship is known. Choose anchors `x_a` indexed by a finite
type `A`. The proposition `InFeatureSpan φ anchor c x` supplies coefficients
`c[a]` such that:

```text
for every feature coordinate f,
  φ(x)[f] = Σ_a c[a] φ(x_a)[f]
```

This is an explicit representation of the feature vector. It is not a
numerical rank estimate, a similarity score, or a claim that the input itself
is a mixture of anchor inputs. The feature map may be nonlinear.

`feature_span_effect` states that the edit effect respects this representation:

```text
Σ_f D[o,f] φ(x)[f] = Σ_a c[a] (Σ_f D[o,f] φ(x_a)[f])
```

If every anchor effect is zero, the right side is zero. `protected_span`
therefore proves `h_Q(x)[o] = h_W(x)[o]` from two premises: a valid feature-span
witness and equality of the actual endpoint readouts at every anchor.
`protected_span_edit` gives the version stated directly for an edit `D`.

In the small illustration, `δ = (0, 1)` preserves every feature vector of the
form `(a, 0)`, because it equals `a p`. The claim covers all such feature
vectors, not just the anchor `p`. Applying it to a domain of real inputs still
requires showing that each input's `φ(x)` has this form.

The checked nonlinear example makes that last step explicit. It uses
`φ(x,y) = (1,x,y²)` and anchors `(0,0)` and `(1,0)`. For every real `t`:

```text
φ(t,0) = (1−t) φ(0,0) + t φ(1,0)
```

This proves protection on an infinite set of inputs. It does not establish
protection away from that set. Nor does preserving a declared score coordinate
automatically preserve a decoded string, a sampled action, or downstream
reward.

## Approximate protection and conditioning

A numerical solver may leave small anchor residuals. Exact zero cannot be
inferred from its tolerance. Suppose the actual endpoint residual at anchor
`a`, for a chosen output `o`, is bounded by `ε[a]`. Given the same feature-span
witness, `endpoint_span_bound` proves:

```text
|h_Q(x)[o] − h_W(x)[o]| ≤ Σ_a |c[a]| ε[a]
```

`residual_span_bound` states the edit version. With a uniform anchor bound
`ε`, `residual_span_l1_bound` gives:

```text
|Δh(x)[o]| ≤ ||c||₁ ε
where ||c||₁ = Σ_a |c[a]|
```

The coefficients determine how errors can grow. In the nonlinear example,
the input `(3,0)` uses coefficients `(-2,3)`. Anchor residuals of magnitude at
most `0.01` therefore give the bound `2 × 0.01 + 3 × 0.01 = 0.05`.
`extrapolation_l1_bound_is_attained` checks an edit that reaches this bound:
its anchor effects are `−1/100` and `1/100`, and its effect at `(3,0)` is
`1/20`. A claim of `0.01` protection everywhere in this span would be false.

The theorem accepts a valid coefficient witness. It does not choose the
best-conditioned representation or prove that a numerical routine found one.
A domain-wide approximate guarantee needs coefficient control throughout that
domain.

Inputs may also lie outside the exact span. Suppose the representation error
in each feature coordinate is bounded by `ρ[f]`:

```text
|φ(x)[f] − Σ_a c[a] φ(x_a)[f]| ≤ ρ[f]
```

`approximate_span_bound` retains the additional contribution:

```text
|Δh(x)[o]| ≤ Σ_a |c[a]| ε[a] + Σ_f |D[o,f]| ρ[f]
```

The first term carries anchor error through the coefficients. The second
measures how the edit acts on features that the anchors do not represent.
Neither a large anchor count nor a small fitting residual removes that second
term. For an endpoint guarantee, instantiate `D = endpointDelta(W,Q)` and use
`endpoint_exact`.

A valid application must therefore establish the anchor residuals, coefficient
witnesses, and any feature-error bounds for its declared domain. No sampling or
distribution-coverage theorem supplies these premises here.

## When a target request is impossible

Return to the requested target change `+1`. If the protected feature becomes
`p = t = (1,1)`, protection demands `δ · t = 0` while the target request demands
`δ · t = 1`. They cannot both hold.

`nonzero_target_in_protected_span_impossible` states the more general result.
If a target's feature vector is in the exactly protected span, its output
change cannot equal any specified nonzero value. The target need not be an
anchor. In the nonlinear witness, exact preservation of both anchors prevents
a change of `1` in output coordinate zero at `(3,0)`.

This is a necessary infeasibility condition. It is not a general solver, and
it does not assert that every target outside the span is feasible under rank,
support, magnitude, realization, or other constraints. If protected features
span the entire head feature space, exact protection leaves only a zero head
change. A numerical full-rank estimate must still be distinguished from a
proved spanning premise.

Correct outcomes include rejection, abstention, or a no-op with unmet target
gain. Zero retention residual alone does not establish a useful edit.

## Factored edits

A factor program writes a head change as a product. For a finite factor index
type `R`, `factoredEdit` defines:

```text
D[o,f] = Σ_r U[o,r] V[r,f]
```

`factored_effect` makes the effect interpretable in two steps:

```text
Δh(x)[o] = Σ_r U[o,r] (Σ_f V[r,f] φ(x)[f])
```

First `V` reads a set of feature directions. Then `U` maps those readings to
output changes. If `V φ(x) = 0`, there is no signal for `U` to amplify.
`factored_preservation` proves preservation under this sufficient condition
for every left factor. `factored_span_preservation` extends a right-factor
kernel condition on anchors to represented inputs.

In the scalar illustration, take `U = (1)` and `V = (0,1)`. Then
`V p = 0` and `V t = 1`, giving the protected edit `(0,1)`. This calculation
illustrates the sufficient condition. It does not make factor count or sparse
storage a preservation guarantee.

With approximate annihilation, `factored_residual_bound` gives:

```text
|(V φ(x))[r]| ≤ η[r] for every r
  implies
|Δh(x)[o]| ≤ Σ_r |U[o,r]| η[r]
```

Large left-factor coefficients can amplify small right-factor residuals. The
module does not formalize `Matrix.rank`, certify an SVD, prove minimum rank or
solver optimality, or establish that a compact program changes few stored
values. Its equations concern the real factor product. The
[structured lowering](../../../rlcd/weight_space/structured.py) and
[feature receipts](../../../rlcd/weight_space/features.py) expose realization
as separate runtime stages; their Python implementations are outside the Lean
proof. After realization, check the actual endpoint residuals.

## Drift through a sequential network

Frozen-feature guarantees cease to apply if the edited computation changes
the feature map. The second family directly compares two affine/activation
chains. A finite coordinate type `D_k` supplies the width at each depth:

```text
z₀ = x
z_(k+1)[i] = σ_(k,i)(b_k[i] + Σ_j W_k[i,j] z_k[j])
```

Old and edited networks share their coordinate types and componentwise
activation functions. Every activation must satisfy, for all real `u,v`:

```text
MagnitudeBound: |σ(u)| ≤ |u|
OneLipschitz:   |σ(u) − σ(v)| ≤ |u − v|
```

The first assumption bounds baseline activation magnitudes. The second bounds
how much an activation can enlarge a difference. The library supplies proofs
for identity, ReLU, and absolute value. Other activations need their own
proofs and a validated runtime mapping. There is no supplied theorem for tanh,
sigmoid, GELU, softmax, attention, or normalization. The executable adapter's
activation set and its certified activation set are deliberately distinct.

### Why the recurrence has two kinds of terms

Let `r_k[j]` bound the magnitude of the baseline coordinate and `d_k[j]` bound
the difference at that coordinate. For one weight/input pair:

```text
W' z' − W z = (W' − W) z + W' (z' − z)
```

The changed weight acts on the old activation. The edited weight also carries
forward the activation difference already accumulated. Applying absolute-value
bounds gives the recurrence:

```text
r_(k+1)[i] = |b_k[i]| + Σ_j |W_k[i,j]| r_k[j]

d_(k+1)[i] = |b'_k[i] − b_k[i]|
           + Σ_j (|W'_k[i,j] − W_k[i,j]| r_k[j]
                  + |W'_k[i,j]| d_k[j])
```

The use of baseline radii and edited weights is intentional.
`layer_radius_le` and `layer_drift_le` establish a single layer.
`chain_radius_le` and `chain_drift_le` propagate the bounds through every
finite depth by induction. All layers' affine weights and biases may change;
the proof does not require an unchanged ReLU activation pattern.

Initially, `r₀` must bound the baseline input magnitudes, and `d₀` must bound
any difference between the two inputs. `chain_same_input_drift` sets `d₀ = 0`
and concludes:

```text
|z'_n[i] − z_n[i]| ≤ d_n[i]
```

For a box domain, `r₀[j] = max(|lower[j]|, |upper[j]|)` provides the magnitude
premise. The runtime [sequential certificate](../../../rlcd/weight_space/sequential.py)
computes this recurrence using exact rational lifts of stored binary64 values
and box endpoints. Its current receipt compares the same input in both
networks. The formal theorem also allows a separately bounded input change.
Matching formulas do not constitute a proved refinement of the Python code.

### A checked three-layer calculation

The rational witness uses ReLU, then absolute value, then identity. Its input
satisfies `|x| ≤ 1`. Every layer changes both weight and bias:

| Layer | Old `(weight, bias)` | Edited `(weight, bias)` | Baseline radius after layer | Drift bound after layer |
| --- | --- | --- | --- | --- |
| ReLU | `(1, 0)` | `(5/4, 1/8)` | `1` | `3/8` |
| Absolute value | `(2, 1/2)` | `(7/4, 1/4)` | `5/2` | `37/32` |
| Identity | `(−1, 0)` | `(−3/2, 1/10)` | `5/2` | `987/320` |

For example, the second drift is
`1/4 + (1/4) × 1 + (7/4) × (3/8) = 37/32`.
`three_layer_uniform_certificate` bounds the final drift by `987/320` for
every input in the interval. `three_layer_endpoint_difference` separately
checks the signed difference `−443/320` at `x = 1`.

The uniform bound is conservative and is not asserted sharp. Absolute-value
propagation discards correlations and cancellation; width and depth can make
that loss substantial. If the bound exceeds an admission budget, this method
has failed to certify the candidate. That alone is not an observed output
violation. The theorem supplies neither exact reachability nor a probability
distribution over inputs.

## Composition and decisions

`coordinate_drift_compose` adds two output-coordinate budgets by the triangle
inequality. The older `DriftContract.trans` and `DriftContract.chain` express
the same idea over a declared input domain and a finite sequence of edits.
The intermediate endpoint must be the same in both comparisons. Domains,
coordinate meanings, and applicable feature maps must also agree.

Thus per-step bounds can supply a cumulative bound against the original
endpoint. They do not say that useful target changes add, that a later edit
retains an earlier improvement, or that a per-step threshold is a global
threshold. A sequence controller must track its cumulative contract and
recheck endpoint identities.

Output bounds can preserve a decision if the decision has enough margin.
For `argmax_preserved`, a baseline winner `w` must satisfy, for every competitor
`c ≠ w`:

```text
base[w] − base[c] > ε[w] + ε[c]
```

Here `ε` bounds the actual coordinate changes. For illustration, scores `3`
and `2`, each moving by at most `1/4`, retain their strict ordering: the gap is
`1`, larger than the combined budget `1/2`. With budgets `1/2` each, the two
scores could meet at `2.5`. This is why the theorem requires a strict
inequality. `argmax_preserved_uniform` is the uniform-budget condition
`margin > 2ε`.

The candidate set, winner, margins, rule, and bounds must be supplied. The
theorem does not establish a margin or resolve ties. It gives no direct claim
about sampling, decoded language, long-horizon reward, language accuracy, or
unlearning.

## Checked mathematical examples

The introductory two-feature example is explanatory. The following witnesses
are actually defined and proved in
[GenericExamples.lean](../../../formal/weight-space/WeightSpace/GenericExamples.lean).
They use explicit real numbers and rational constants.

| Witness | Construction and conclusion | Useful identifier |
| --- | --- | --- |
| Infinite protected input set | `φ(x,y)=(1,x,y²)`; coefficients `(1−t,t)` represent every `(t,0)` using two anchors | `line_in_feature_span` |
| Selective two-output edit | Quadratic-feature coefficients `1/2` and `−1/4` preserve both outputs on `(t,0)` | `nonlinear_feature_protected_line` |
| Target change | The same edit changes the outputs at `(2,2)` by `2` and `−1` | `target_output_zero_change`, `target_output_one_change` |
| Attained residual bound | Scalar edit `(-1/100,1/50,1/2)` has anchor magnitudes at most `1/100` and drift `1/20` at `(3,0)` | `extrapolation_l1_certificate`, `extrapolation_l1_bound_is_attained` |
| Factored protection | `U=(1/2,−1/4)` and `V=(0,0,1)` produce the selective edit and annihilate protected features | `factored_edit_eq`, `right_factor_protected_kernel`, `factored_nonlinear_protection` |
| Infeasible request | Exact preservation of the anchors prevents a change of `1` in output coordinate zero at `(3,0)` | `protected_target_one_infeasible` |
| Multilayer drift | The mixed-activation chain has the all-input budget and endpoint difference described above | `three_layer_uniform_certificate`, `three_layer_endpoint_difference` |

These are hand-written mathematical instances. They are not generated from
the Python study or formally bound to its JSON candidates. The general
theorems, these witnesses, runtime receipts, and measured outcomes serve
different evidential roles.

## What the proof does not connect yet

An application needs a chain of justified correspondences. A theorem about a
real function is useful only to the extent that its definitions and premises
describe the artifact and claim at hand.

| Connection | Evidence currently available | Remaining obligation |
| --- | --- | --- |
| Serialized model to mathematical parameters | Exact artifact identities and declared interpretations; a historical RTS-specific generator | Bind new runtime endpoints, feature order, bias convention, domain, and operator semantics to new formal instances |
| Runtime extractor to fixed `φ` | Caller-supplied feature identity and supported fixture code | Prove equality or a bounded correspondence on the claimed domain |
| Proposal to realized endpoint | Separate requested edits, structured lowering, and endpoint residual receipts | Establish the relevant realized premises; theorems do not certify SVD exactness, solver termination, optimality, or minimum rank |
| Real arithmetic to native execution | Real/rational bounds and supported native realization | Bound intermediate rounding, accumulation, overflow, kernels, serialization, and any hidden state |
| Runtime certificate to Lean statement | Corresponding formulas and explicit supported cases | Prove or independently validate the Python/NumPy/PyTorch implementation correspondence |
| Protected scores to useful behavior | Equality, coordinate bounds, and conditional strict argmax preservation | Establish decision semantics, causal repair, task retention, utility, and appropriate distribution coverage |
| One architecture to another | A language for separate semantic instances | Supply a validated circuit correspondence or transfer relation; compatible tensor shapes are insufficient |

Sparse edits can coexist with regressions, and a plausible diagnosis can
coexist with failed repair. A valid mathematical result may hold while a
behavioral hypothesis fails. A useful measured result may also lack a formal
certificate. Reports should identify which connection has been established.

## Reproduce the check

The explicit target is `WeightSpace.GenericAudit`. From
`formal/weight-space`, after installing the pinned toolchain and locked
dependencies, run:

```text
lake build WeightSpace.GenericAudit
lake env lean WeightSpace/GenericAudit.lean
```

Both commands must succeed. Plain `lake build` remains the preserved
historical target and does not check the generic extension. The generic audit
loads both generations of the library, follows the transitive dependencies of
every loaded `WeightSpace` declaration with `Lean.collectAxioms`, and permits
only `propext`, `Classical.choice`, and `Quot.sound`. Representative axiom
printouts are not the coverage boundary. Additional axioms and proof
placeholders fail the full audit; ordinary `decide` in finite examples produces
kernel-checkable proofs.

The saved [verification receipt](../../../formal/weight-space/verification-v2/actual/verification.json),
[build log](../../../formal/weight-space/verification-v2/actual/build.log), and
[audit log](../../../formal/weight-space/verification-v2/actual/axiom-audit.log)
record the historical run with Lean `4.24.0` and Mathlib commit
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. The
[preservation snapshot](../../../formal/weight-space/verification-v2/preserved-v1-inputs.json)
retains the earlier proof evidence. Those receipts have their original scope;
rewriting this explanatory guide does not create a new verification run.

For a fresh portable run and a new receipt, follow the prerequisites and
commands in [PROOFS.md](../../../PROOFS.md#fresh-clone-procedure). It also
explains how the original, hash-bound documentation is preserved separately
from the live guide. The old
[Windows checker](../../../formal/weight-space/verification-v2/check.ps1)
documents the original local environment and is not the portable entry point.

The trusted base includes Lean's kernel and toolchain, the allowed axioms,
imported libraries and compiled artifacts, and the execution environment. A
cached build is not a from-source rebuild of Lean or all of Mathlib. Source
pins and hashes make evidence inspectable; they are not independent signed
attestations.

## Next formal obligations

The next work should make a specific application claim possible. Each item
below has a concrete completion condition.

| Obligation | Evidence needed before claiming completion |
| --- | --- |
| Bind runtime candidates to formal instances | Deterministic generation from exact endpoints and domain; independently checked identities; numeric premises; rejection of changed endpoints, shapes, order, feature identities, and domains |
| Establish the feature correspondence | A declared extractor model and a proved equality or error bound over the intended domain, with residual terms propagated into the guarantee |
| Extend the bound to numerical execution | A specified dtype, operation order, and supported backend; rounding and overflow assumptions; numerical error terms that compose with the real drift bound |
| Tighten deeper-network bounds | Checked inequalities for signed intervals, activation information, or other sound relaxations; measured useful certification coverage and tightness on frozen tasks |
| Add operators or transfer relations | Operator-specific semantics and proofs, validated adapters, and explicit correspondence obligations for any architecture transfer |
| Support decisions and repeated edits | Candidate-set, margin, domain, and composition witnesses, checked against each endpoint in a recorded lineage |

The [limitations and next-steps plan](limitations-and-next-steps.md) connects
these obligations to access controls, task experiments, causal interventions,
independent units, and success or falsification criteria. Formal completion
for one declared family does not settle those empirical questions.
