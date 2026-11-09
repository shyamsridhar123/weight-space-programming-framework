# Bounded frozen-feature score feasibility

**Engineering supplement: November 9, 2026, America/Chicago.** The readout
compiler now reports bounded exact consistency and minimum-norm diagnostics
for supplied feature-score constraints, alongside numerical projection and
stored endpoint residuals. This extends the diagnostic implementation. It does
not rerun the November 8 trained studies, change their frozen `1e-12` cutoff,
or add a trained development-only threshold sweep. Their negative outcomes and
all-assigned denominators remain in [the trained report](trained-applications.md).

The separate later [synthetic trained-development retention study](retention-development.md),
also dated November 9, supplies a frozen cutoff/budget/margin sweep. Its
recorded outcomes are separate from this engineering extension and do not
retune the earlier trained evaluations.

The implementation is in [features.py](../../rlcd/weight_space/features.py),
with the affine research adapter in
[research_tasks.py](../../rlcd/weight_space/research_tasks.py).
[The verification receipt](../../provenance/verification/bounded-feature-feasibility-20261109.json)
records the execution results and captured source identities for this
engineering extension at its then-captured source: 47 focused checks passed,
followed by 397 configured engineering checks passed with three frozen-study
deselections. Those inputs remained unchanged, with no provider calls or
scientific study runs. The separate
[affine endpoint correction receipt](../../provenance/verification/affine-endpoint-certificate-correction-20261109.json)
preserves the later PR #20 hosted failure: its affine fixture incorrectly
required an SVD-derived stored endpoint to have an exactly zero protected
residual. The corrected assertion independently computes that residual with
exact fractions and checks the certificate fields and zero flag. Its focused
local check passed without adding a tolerance or changing the implementation.
The original receipt is unchanged; corrected hosted results are tracked
separately at the PR head.
A subsequent hosted job at affine-corrected head `809a40e0` completed
publication verification, the engineering suite, wheel build and a fresh
wheel-installation check successfully. Its step summaries and the later
[edit-coordinate underflow correction](../../provenance/verification/edit-coordinate-underflow-correction-20261109.json)
are recorded separately; that hosted success predates the underflow change.
The new focused underflow check passed at its captured source. Final-head
hosted validation is tracked separately on PR #20.
[The focused checks](../../tests/test_feature_constraints.py)
include direct rational checks of conflict witnesses, known minimum norms,
resource exhaustion, and numerical/endpoint disagreements. They cover
synthetic supplied matrices and an [in-memory, from-scratch trained readout
fixture](../../tests/test_trained_research.py). This is engineering verification;
it supplies no new trained efficacy, freshness, decision-feasibility or
whole-domain evidence.

For a frozen feature map, let `R` contain protected feature rows and `T` contain
target feature rows. Each column is an editable head coordinate. Let `X` be the
coordinate-by-output edit matrix and `D` the requested target-score changes.
The exact real-score system is:

```text
R X = 0,   T X = D.
A = [R; T],   B = [0; D],   A X = B.
```

The compiler adds a constant design column only when `edit_bias=True` and an
explicit bias tensor exists. With a fixed bias, its change is zero and its
coordinate is excluded from this system. `feature_geometry` analyzes the
affine augmented design, including that constant coordinate. A supplied
feature identity and head equation remain adapter premises.

Full **column** rank of row-oriented `R` forces `ker(R) = {0}`, so the only
exact score-retaining head edit is zero. Full row rank alone does not imply
this for a short, wide matrix. For example, `R = [[1, 0, 0], [0, 1, 0]]` has
full row rank and leaves the third coordinate editable. The original Issue #6
c4 checkbox is preserved verbatim; its assessment states the correct
orientation of the intended full protected-span condition.

`compile_feature_edit` places the analysis in
`FeatureCompilation.diagnostics["feature_constraints"]`, schema
`wsp-feature-constraints/1`. The internal `_analyze_feature_constraints` helper
also serves `feature_geometry(features, targets, desired_delta=...)`. Omitting
`desired_delta` from `feature_geometry` leaves exact consistency
`not_requested` and keeps its existing modular-rank lower-bound semantics;
it does not establish exact nullity from a modular rank shortfall.

| Diagnostic | Meaning and scope |
| --- | --- |
| `protected` | Numerical singular values, cutoff, rank/nullity, and full/retained-spectrum conditioning of the editable protection design |
| `projected_target` | Numerical spectrum and per-row energy/fraction after projection into the SVD protected nullspace |
| `unregularized` | Truncated-SVD target residuals and minimum edit-norm estimates before ridge fitting or endpoint rounding; numerical limits are explicit |
| `exact` | Bounded rational consistency, protected/constraint rank where completed, conflict witnesses, and exact minimum squared norms for the specified real-score equations |
| Compiler endpoint diagnostics and retention certificate | Exact-dyadic residuals of the actual stored head relative to the requested target changes and supplied protected rows |

If final edit-coordinate conversion rounds a nonzero coordinate to zero,
`unregularized` reports the affected per-output edit norm as `null` with reason
`edit_coordinate_underflow`; the joint Frobenius norm is also unavailable.
Individually lost coordinates can have a representable combined norm, so zero
would be misleading. Residual fields retain their independent loss checks.

The exact procedure lifts the captured finite binary floats to rational
numbers. Its `consistency_status` is `feasible`, `infeasible`, `unresolved`, or
`not_requested`, with `per_output_consistency` retaining individual output
judgments. A conflict witness records a sparse combination of original rows of
`A` whose left side is zero and whose response in a named output is nonzero.
The focused checker verifies those combinations directly without invoking the
analyzer's elimination helpers. A conflict in one output does not invalidate
another output's completed feasible judgment or norm.

For example, protected row `[0, 1]`, target rows `[1, 1]` and `[2, 1]`, and
requested changes `1` and `3` conflict jointly. The combination
`protected - 2*target_1 + target_2` has zero feature row and response `1`.
The same rows with requested changes `1` and `2` are consistent and have
minimum squared edit norm `1`. Checking target rows separately would miss the
first joint conflict.

The default exact work limit is `250,000` captured scalars/Fraction operations;
the maximum numerator/denominator length is `4,096` bits. The internal helper
accepts an explicit nonnegative `exact_work_limit`; the compiler and research
adapter use the default. The result records limits, consumed work, completed
rows and `consistency_limit_reason`/`minimum_norm_limit_reason`. Exact limits
are separate from numerical magnitude limits and from study wall-time or
memory budgets.

When exact work or rational growth stops unfinished analysis, the unfinished
judgments remain `unresolved`. An independently established conflict witness
is retained: the overall system can be `infeasible` while other output
judgments remain unresolved after a later limit. Protected rank/nullity and
full constraint rank are exact only when their relevant rows are completed;
otherwise recorded lower bounds do not become exact-nullity claims. Completed
consistency can also remain `feasible` when minimum-norm work subsequently
exhausts its limit. Completed per-output norms are preserved and unfinished
norms remain unavailable.

Exact norms are rational strings in `minimum_squared_l2_norm_exact`, one per
output. `minimum_squared_frobenius_norm_exact` is available only when the full
system is feasible and the aggregate computation completes. These are
unconstrained real/rational minimum squared norms on the declared writable
coordinates. They do not incorporate parameter-cell caps, edit-count limits,
native representability or alternative argmax goals.

The exact diagnostic does not select an exact rational executable proposal.
`compile_feature_edit` still fits its declared `none`, `soft` or `nullspace`
numerical ridge objective and lowers the resulting proposal. With the default
`require_exact_scores=False`, it can emit an approximate proposal even when
the specified exact score equations
are infeasible. SVD may truncate a tiny nonzero protected direction; exact
analysis can establish a zero-only edit space while the numerical compiler
still proposes a change. `optimizer_uses_numerical_nullspace` identifies that
choice. An exact conflict is a failure of the specified score equations,
rather than proof that every alternative argmax-useful edit is impossible.

The actual stored endpoint is checked separately. The compiler records
`realized_target_residuals_exact`,
`realized_target_maximum_absolute_residuals_exact`, and
`realized_target_exactly_matches_requested_delta`.
`certify_feature_retention` records exact protected residuals and
`exactly_zero_on_supplied_features`. For example, the real equation `3*x = 1`
is consistent with minimum squared norm `1/9`, but the focused float32
realization has exact target residual `1/33554432`. A feasible real system or
passing numerical tolerance does not make that residual zero.

`compile_feature_edit(..., require_exact_scores=True)` opts into a strict
supplied-feature score contract. The same boolean field is accepted in a
`compile-features` JSON specification. Infeasible or unresolved exact equations
are rejected before fitting. After lowering, the stored endpoint must exactly
realize every requested target change and have exact zero protected residuals
when protected rows were supplied, regardless of the numerical retention mode.
Failures have distinct feasibility, target-realization and protection errors;
the CLI rejects before creating its output directory. Successful opted-in
diagnostics record `require_exact_scores: true`. The default remains false and
preserves the existing numerical objectives and diagnostics.

The [source-bound standalone check](../../provenance/verification/exact-score-admission-20261109.json)
passed once on November 9, 2026: it covered default approximate compilation,
rounded target-residual rejection, exact acceptance and strict boolean validation.
It did not exercise infeasible, unresolved or protection-only rejection. Its
sanitized receipt identifies the unchanged check-time source snapshot, which
predates this verification paragraph and later integration; it adds no native
execution, use-case, proof or replay guarantee, and c4 remains partial.

This guard does not construct an exact rational executable proposal or add
target bindings to general `AdmissionRequest` replay. A feasible real system
can still be rejected because ridge fitting or endpoint rounding leaves a
nonzero residual. A zero requested change can yield an exact no-op; `no_op`
remains explicit and does not establish useful success. Trained and prospective
callers retain their existing declared behavior gates. Application usefulness
and replayable general admission remain separate requirements, so
[Issue #6 c4 remains partial](issue-acceptance-status.md#issue-6).

The supplied-feature scope remains explicit: no proof of a raw-input encoder,
native floating execution, unseen/domain coverage, business benefit or useful
population coverage follows from these diagnostics. Issue #6 c0 is supported
for this bounded diagnostic deliverable. The separate November 9
[synthetic trained-development sweep](retention-development.md) supports c1
only for its declared comparison and fixed thresholds on development data.
Its calibration participates in the gates and supplies no heldout evidence,
population-retention claim or independently authenticated evaluator run.
Issue #6 c4 remains partial for the generic conflict-admission requirement,
and Issue #6 remains open.
