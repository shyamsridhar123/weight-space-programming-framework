# Weight-space proof kernel

This project formalizes a finite, scalar-output, two-layer ReLU network in Lean
4.24.0. Mathlib is pinned to commit
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7` (`v4.24.0`), with the complete dependency
lock in `lake-manifest.json`.

The parameters are real numbers. A vector-output network instantiates the scalar
scorer once per output coordinate while sharing its first layer. The definitions
include both biases:

```text
h_p(x)[j] = max(0, b1_p[j] + sum_i w1_p[j,i] * x[i])
score_p(x) = b2_p + sum_j w2_p[j] * h_p(x)[j]
```

## Checked interfaces

`WeightSpace/Core.lean` defines the network and edit semantics:

- `score_permute`: bijectively relabeling hidden units, with every incident
  parameter transported, preserves the score.
- `transported_edit_score`: transporting both a base network and its additive
  edit preserves the edited score.
- `readout_exact`: a frozen-hidden-layer edit has exactly its stated linear
  effect on the score.
- `readoutEffect_add` and `editReadout_add`: readout effects compose additively
  in a shared hidden representation.

`WeightSpace/Gauge.lean` additionally proves exact positive hidden-unit
rescaling. Incoming weights and the hidden bias are multiplied by a strictly
positive factor, and the outgoing weight is divided by that factor.
`score_rescale` proves score invariance; `rescale_applyEdit` and
`rescaled_edit_score` prove correct joint transport of a base network and its
additive parameter edit.

`WeightSpace/Bounds.lean` derives the all-layer certificate. For a feature box
`lower[i] <= x[i] <= upper[i]`, it defines

```text
r[i] = max(abs(lower[i]), abs(upper[i]))
U_p[j] = max(0, b1_p[j] + sum_i max(w1_p[j,i]*lower[i], w1_p[j,i]*upper[i]))
D_pq[j] = abs(b1_q[j]-b1_p[j]) + sum_i abs(w1_q[j,i]-w1_p[j,i])*r[i]
B_pq = abs(b2_q-b2_p)
       + sum_j (abs(w2_q[j]-w2_p[j])*U_p[j] + abs(w2_q[j])*D_pq[j])
```

The theorem `score_drift_le` proves `abs(score_q(x)-score_p(x)) <= B_pq` for
every real input in that box. `score_drift_le_of_budget` converts a proved
numeric inequality `B_pq <= epsilon` into a uniform semantic guarantee. The
proof uses signed interval bounds, the triangle inequality, and ReLU's global
one-Lipschitz property. It does not assume unchanged activation patterns.

For readout-only programs, `driftBudget_readout` proves that the general formula
reduces to the readout budget. For a single edited hidden readout coordinate,
`readout_single_drift_le_of_budget` requires only
`abs(delta) * hiddenUpper p box selected <= epsilon`. Consequently the numeric
certificate need evaluate only the selected hidden row, even when `p` contains
a complete larger network. `readout_single_exact` also supplies the exact
score difference `delta * hidden p x selected`.

`WeightSpace/Contracts.lean` proves finite sequential composition of drift
contracts and strict argmax preservation. The coordinate-specific margin
condition is `epsilon[winner] + epsilon[c] < base[winner] - base[c]` for every
competitor. `argmax_preserved_uniform` specializes this to a strict margin
greater than `2*epsilon`. These theorems preserve an already established strict
winner; they do not assert that such a margin exists.

`WeightSpace/Examples.lean` contains small rational examples, including an edit
of every parameter block with budget `49/20`. These examples are illustrative
and are not the RTS model.

## Build

From the repository root on Windows:

```powershell
pwsh -File scripts/check_weight_space_proofs.ps1 -Bootstrap
```

Once the pinned toolchain and Mathlib cache exist, omit `-Bootstrap`; add
`-SkipCache` to avoid downloading already available dependency artifacts:

```powershell
pwsh -File scripts/check_weight_space_proofs.ps1 -SkipCache
```

The bootstrap verifies the official Windows archive's SHA256, checks out each
dependency at the exact locked commit, retrieves the Mathlib artifacts, builds
the project, and runs the axiom audit. It does not silently skip verification
when the toolchain or a dependency is missing. The first cache retrieval may
take several minutes because this Mathlib version hashes its full source tree.

With Lean 4.24.0 already on `PATH`, the equivalent project commands are:

```powershell
Set-Location formal/weight-space
lake exe cache get Mathlib.Data.Real.Basic Mathlib.Algebra.BigOperators.Fin Mathlib.Algebra.Order.BigOperators.Group.Finset Mathlib.Tactic.Ring Mathlib.Tactic.Linarith Mathlib.Tactic.NormNum
lake build
lake env lean WeightSpace/AxiomAudit.lean
```

`WeightSpace/AxiomAudit.lean` prints the transitive axioms of representative
theorems and audits every declaration in the `WeightSpace` namespace. It fails
elaboration if any declaration uses anything beyond `propext`,
`Classical.choice`, and `Quot.sound`, the ordinary Lean/Mathlib foundations used
here. In particular, proof placeholders, added axioms, and native-evaluation
proof shortcuts are not accepted. The preserved generic build and audit logs
in `verification/` record the checked sources and explicitly distinguish the
generic library from the later RTS specialization.

## Generate an RTS certificate

`generate_rts_certificate.py` embeds a frozen `wsp-rational-policy/1` artifact
as a complete 32-feature, 38-hidden-unit network. All 1,293 base coefficients
are represented, with zero defaults for sparse matrix entries. Each candidate
uses its explicit final readout value; Lean proves that its change equals the
exact rational difference of its two binary64 endpoints.

From the repository root, using the frozen study bundle:

```powershell
python formal/weight-space/generate_rts_certificate.py artifacts/weight-space/study-v1/rts-candidates/exact-policy.json formal/weight-space/WeightSpace/RTS.lean --source-model artifacts/weight-space/study-v1/rts-export/model.json
python formal/weight-space/generate_rts_certificate.py artifacts/weight-space/study-v1/rts-candidates/exact-policy.json formal/weight-space/WeightSpace/RTS.lean --source-model artifacts/weight-space/study-v1/rts-export/model.json --check
pwsh -File scripts/check_weight_space_proofs.ps1 -SkipCache
```

The optional `--source-model` check independently compares every exported
coefficient, the source file hash, feature and hidden labels, and reconstructed
canonical base/candidate network hashes. The generator uses only the Python
standard library and does not import the runtime exporter. `--check` requires
byte-for-byte deterministic regeneration and never modifies the Lean file.

The RTS specialization selects hidden unit 19 (`army_train_bias`), whose
hidden activation is exactly `ReLU(x[24])`, for feature 24 (`train_army`). It proves
the exact score effect for every real input, the absolute score-drift bound
`abs(actual_delta)` on `[0,1]^32`, and exact score preservation on that box
with `x[24] = 0`. Both concrete boxes have checked validity proofs. These are
parameter and score guarantees; candidate utility remains an empirical question.

The selected base readout is exactly
`3602879701896397/4503599627370496`. For each candidate, the `_eq` theorem
connects its explicit endpoint to the readout edit, `_exact` gives its score
difference as `actual_delta * max(0, x[24])`, `_bound` bounds the absolute
difference on the global box, and `_protected` proves equal scores on the
protected box:

| Candidate | Exact delta and global bound | Protected bound |
|---|---|---|
| `recruit_0_5` | `1/2` | `0` |
| `recruit_1_0` | `1` | `0` |
| `recruit_2_0` | `9007199254740991/4503599627370496` (`2 - 2^-52`) | `0` |
| `recruit_4_0` | `18014398509481983/4503599627370496` (`4 - 2^-52`) | `0` |

These are four independently checked endpoint contracts. They remain valid
regardless of which candidate an empirical selection procedure chooses, and
they make no gameplay-effectiveness claim. In particular, the last exact
bound must not be replaced by its rounded binary64 display `4.0`.

The successful frozen-model run is preserved in
[actual-verification.json](verification/actual-verification.json), with
[deterministic regeneration](verification/actual-regeneration.log),
[complete build](verification/actual-build.log), and
[standalone axiom audit](verification/actual-axiom-audit.log) logs. The checker
returned exit code 0; the build completed 987 jobs. There are **59 explicit
`theorem` statements** in the source, including **21 RTS statements**. The
namespace audit covers **236 declarations**, of which **121 are theorem
declarations**; its RTS subset contains **98 declarations / 50 theorem
declarations**. The environment counts include compiler-generated declarations
and are not counts of written theorem statements. All audited declarations use
only the three permitted foundational axioms.

The earlier generic-only evidence remains unchanged in
[generic-verification.json](verification/generic-verification.json),
[generic-build.log](verification/generic-build.log), and
[generic-axiom-audit.log](verification/generic-axiom-audit.log). Its 137 audited
declarations describe that earlier stage. The actual receipt records hashes
of all proof sources, build configuration, generator, checker, frozen inputs,
and saved logs, plus the hashes of these unchanged generic records.

## Semantic scope and trust boundary

The theorems are about explicitly defined finite real-valued networks. Decimal
runtime weights must be embedded as the exact rational represented by each
serialized binary64 number, not as a visually similar short decimal. For
example, binary64 `0.8` is not exactly the rational `4/5`. An edit delta must be
the exact rational difference of the serialized candidate and base endpoints;
binary64 addition may not equal addition over the reals.

Lean checks the proof after those rational values and the box have been
embedded. A generated certificate must identify the source model, feature and
output order, source and candidate hashes, and numerical budget. A hash label
by itself is not a theorem connecting external bytes to a Lean definition;
that connection depends on inspecting or independently checking the exporter.

The trusted boundary includes Lean's kernel and standard foundational axioms,
the pinned libraries and build artifacts, and the mapping from runtime data to
the explicit formal definitions. These theorems do not bound binary64 rounding
in JavaScript or Python execution, prove simulator reachability of a box, prove
gameplay improvement, or prove equivalence to a different network architecture.
A runtime system requiring such conclusions needs corresponding additional
proofs or separately reported empirical validation. Hidden permutations are
also exact symmetries of the real scorer; changed floating-point summation
order is outside this claim.

`InBox` states the pointwise domain assumption. `InputBox.Valid` is available
for explicit nonempty-box checking. The universal bound remains mathematically
valid on an empty box, so application code should reject inconsistent endpoints
and generated concrete boxes should prove their validity.
