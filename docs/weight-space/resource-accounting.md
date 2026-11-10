# Observed resource coverage

[`observed_resource_summary`](../../rlcd/weight_space/metrics.py) summarizes one
resource field across caller-supplied assignments. The caller must keep the
quantity, unit and measurement scope consistent. `None` means unknown; a known
zero remains an observation. A partial sum never supplies a complete total.

| Returned field | Meaning |
| --- | --- |
| `assigned` | Number of supplied values, including unknown values. |
| `observed` | Number of non-`None` values. |
| `missing` | `assigned - observed`. |
| `observed_sum` | Sum of observed values, or `None` when none are observed. |
| `complete_for_assigned` | Whether every supplied value is observed. |
| `total_if_complete` | `observed_sum` when complete; otherwise `None`. |

For example, the authored input `[0, None, 2]` has three assignments, two
observations, one missing value and an observed sum of `2`. Its completeness flag
is false and its complete total is `None`. `[0]` has an observed and complete
total of `0`. Empty input has all three counts equal to zero, completeness true
and both sums `None`; it does not assert a measured zero cost. These examples
describe the contract and are not measured research costs.

Observed values must have exactly the native Python type `int` or `float` and
be nonnegative; floats must also be finite. Booleans, subclasses, numeric
strings, NumPy scalars, other numeric types, negative values, infinities and
NaNs raise `ValueError`. Integer-only sums use
Python integer arithmetic and remain exact. Any float selects `math.fsum` and
binary64 precision: decimal fractions need not be exact, and large integers in
a mixed sum can lose precision. Conversion or sum overflow is rejected with
`ValueError`; the helper does not return an infinite total.

Completeness concerns the supplied values only. It does not show that an
instrument covers every operation, that a reported zero reflects no work, or
that the caller supplied every intended assignment. The helper adds no timer,
unit conversion, missing-value imputation or lifecycle cost estimate.

## Fresh trained-study summaries

[`summarize_resource_measurements`](../../scripts/run_trained_research.py)
applies this contract to every assigned outcome in an arm, including outcomes
marked `evaluation_missing` when they retain known costs. Behavioral evaluation
availability and cost availability are separate. A missing or `None` cost
dictionary, absent field or `None` field value contributes an unknown value for
that field. A non-dictionary cost container is rejected. Malformed values in
the eight recognized fields are rejected rather than silently discarded;
`storage_bytes_pre_outcome` additionally requires native integers, so a float
byte count is invalid even when numerically integral. Extra cost keys are not
included in this fixed adapter.

Each arm's `resource_measurements` contains a separate six-field summary for
each quantity below. The boundaries describe the current producer; this
summary change does not alter its timers.

| Cost field | Unit and current scope |
| --- | --- |
| `fit_seconds_all_options` | Seconds summed from option fit records returned to `fit_case_arm`; excludes selection and other surrounding work. |
| `fit_selection_materialization_seconds` | Seconds from entry to `fit_case_arm` through its timer sample during attempt-record construction, including calibration, fitting, selection, structural/session work and earlier artifact writes; excludes final `attempt.json` persistence. |
| `native_head_evaluation_seconds` | Seconds around parent/candidate native head evaluation, diagnostic code generation and deployed-code copying on retained test features. |
| `binary64_head_evaluation_seconds` | Seconds around the parent and candidate NumPy head calls on retained test features. |
| `exact_application_rollback_verification_seconds` | Seconds for program/rollback reads, application/rollback identity checks and optional anchor-receipt verification. |
| `heldout_export_seconds` | Seconds from before the prediction archive and evaluation-receipt writes through its sample during cost construction; excludes the later storage scan, prediction hash and final `outcome.json` write. |
| `heldout_workflow_seconds` | Seconds from entry to `evaluate_attempt` through its sample during cost construction; includes native/binary64 evaluation, metrics, verification, heldout session work and export, but excludes the later storage scan, prediction hash and final outcome write. |
| `storage_bytes_pre_outcome` | Integer bytes of immediate files in the attempt directory at the scan before the current outcome write; not a recursive or lifecycle storage measure. |

The four existing total aliases now read the corresponding
`total_if_complete`. A missing value for any assigned outcome makes that alias
`None`; known costs remain visible through `observed_sum` and the counts.

| Existing alias | Source field |
| --- | --- |
| `total_fit_seconds_all_options` | `fit_seconds_all_options` |
| `total_fit_selection_materialization_seconds` | `fit_selection_materialization_seconds` |
| `total_heldout_workflow_seconds` | `heldout_workflow_seconds` |
| `total_storage_bytes` | `storage_bytes_pre_outcome` |

Fresh `summarize_main` output uses `wsp-trained-study-summary/2` for this
changed total contract. Historical summaries retain their recorded schema and
values. This change performs no historical re-summary, failed-work recovery or
scientific execution, and leaves `analyze_study` deferred.

## Fresh posthoc resource analysis

The preceding `analyze_study` deferral records the scope of the earlier
summary-only patch. This later consumer update adds
[`resource_distribution` and `summarize_posthoc_resources`](../../scripts/run_trained_research.py)
for fresh posthoc analysis source. Each arm retains `resource_measurements`
coverage for the same fixed eight fields above, across every supplied assignment,
including known costs on outcomes marked `evaluation_missing`. The numeric and
cost-container validation rules remain those of the trained-resource adapter.

Each field also has a `stages` entry. `observations` counts its known values;
`mean`, `minimum`, `maximum`, `p50` and `p95` describe those observations in
approximate binary64, with linear interpolation at ranks `(n - 1) * p` for the
quantiles. Empty observations leave these statistics `None`. `sum_observed`
copies the coverage summary's observed sum, and `missing_assigned` copies its
missing count. Integer-only accounting sums stay exact: for example, byte
observations `2**60` and `1` sum to the integer `2**60 + 1`, independently of the
rounded float statistics. Statistics require finite binary64 conversion and
results; an integer accepted by the exact coverage helper can still be too large
for a posthoc distribution, causing the posthoc helper to raise `ValueError`.
These statistics do not make a partial sum a complete total.

`paired_top_level_seconds` is a six-field coverage summary, not a distribution.
For each assignment it observes the sum of
`fit_selection_materialization_seconds` and `heldout_workflow_seconds` only when
both are present. Either missing value makes that assignment's pair unknown.
`observed_incremental_top_level_seconds` aliases this summary's `observed_sum`;
it is `None` when no complete pair is observed and is not a lifecycle total.
For authored rows `(2, None)` and `(None, 3)`, the separate field sums are `2`
and `3`, but paired coverage is assigned `2`, observed `0`, missing `2`,
`observed_sum=None`, `complete_for_assigned=False` and `total_if_complete=None`.
The alias is therefore `None`, never `5`. A known `(0, 0)` is an observed zero
pair. Pair completeness concerns only the two enclosing clocks, not all eight
fields or full instrumentation. Binary64 pairing happens within each row before
aggregation and can round differently from separately summing field totals.
Nested option/evaluation/export timings are not added to these enclosing clocks.

Fresh `analyze_study` source emits `wsp-trained-posthoc-analysis/2`, adds the
current metrics source hash at `source_sha256.metrics`, and replaces arm-level
`timing_records` and `timing_missing` with the field-specific coverage above.
The current metrics source hash is
`3cf3061c0a6f1676066ef15649ac5877ba8161d11a5a3d24c4b0f06ecaea77ad`.
Geometry continues to use the unchanged `measured_distribution` helper.
Historical `/1` analyses retain their recorded schema and values. This change
performs no historical re-analysis, failed-work recovery or scientific execution.
The producer deferral recorded in this posthoc receipt describes its earlier
source snapshot. The producer follow-up below changes pre-return cost
availability; #14 c0/c5 remain partial and open.

The [posthoc engineering receipt](../../provenance/verification/posthoc-resource-coverage-20261109.json)
records one passing authored scalar check of four AST-extracted functions:
`observed_resource_summary`, `summarize_resource_measurements`,
`resource_distribution` and `summarize_posthoc_resources`. It captures source
commit `1f78ffc8efe4f14190435dead249b4a7ec5d7207` and integrated main
`cc8190cf92c66820033c130929fbba5fd76ab2b1`. The check uses literal costs and
stdlib bindings without project imports. It does not execute `analyze_study`,
its schema/provenance wiring, geometry, study loading, JSON persistence or replay,
and does not verify later documentation/publication edits. Captured runtime
strings and installed version metadata were unchanged; executable/package-content
identity and agreement with the lockfile were not established. Fixture values
are not measured research costs. The earlier summary receipt below retains its
original snapshot and scope.

## Timer nesting and remaining gaps

Option fit times are nested within fit/selection/materialization elapsed.
Native evaluation, binary64 evaluation, verification and export timers are
nested within heldout workflow elapsed. Adding these nested quantities to
their enclosing timers counts some work twice. The named verification timer
does not establish an exact raw-input or whole-pipeline numerical guarantee.

In the source snapshots covered by the earlier receipts, eager `fit_options`
could perform fitting and then raise before returning its list. The caller's
local `options` remained empty, allowing `fit_seconds_all_options` to report
zero despite attempted fitting. Those receipts retain their original source
snapshots and execution scopes.

The producer follow-up tracks whether eager `fit_options` returned, setting
`fit_options_returned` immediately after its return. Any failure before return
now records `fit_seconds_all_options` as unknown (`None` / JSON `null`),
including a failure after partial internal fitting. It does not recover
unreturned option timings. Once options have returned, a later calibration
failure retains their existing summed cost; returned explicit no-fit options
retain observed zero.

Existing enclosing timer boundaries and nesting are unchanged. The broad
fit/selection/materialization timer retains enclosing elapsed time when the
fallback completes, but cannot recover missing option timings or isolate
fitting. Availability counts cannot recover those timings, and complete fields
remain subject to their instrumentation limits.

The [fit-failure engineering receipt](../../provenance/verification/fit-failure-cost-20261109.json)
records one passing regression using AST-extracted initializers, the
proposal/calibration `try`/`except` block and the `fit_seconds_all_options`
record-field expression, with stdlib fakes and literal costs. It captures source
commit `5a07d74a4464bdd57e81bab110419b0d1ed8a7ae` and integrated main
`24790faf7f705b7563c00390829055f10e5848f6`. It does not execute full
`fit_case_arm`, perform fitting, measure timers, run scientific studies or
recover historical costs. Captured runtime strings and installed version
metadata were unchanged; executable/package-content identity and agreement with
the lockfile were not established. The recorded test does not verify later
documentation/publication edits. Fixture values are not measured research costs;
#14 c0/c5 remain partial and open.

The [historical cost report](../research/trained-applications.md#cost-measured-stages-and-their-limits)
retains its own measurement boundaries. Complete cold acquisition,
preparation/encoding, memory/energy, model/device coverage and inference
maintenance remain incomplete. This API establishes no lifecycle total,
amortization crossover or cost advantage, and #14 c0/c5 remain partial and open.

The [engineering receipt](../../provenance/verification/resource-coverage-summary-20261109.json)
records one passing authored scalar seam check of the AST-extracted helper and
trained-resource adapter. It captures source commit
`78d6eeef59bfe97bc3202fd1660e357b67051d00` and integrated main
`ba2bf994a8f0688e61d27d3b438360e44ebf1e24`. Runtime identity was captured and
unchanged across the check; agreement with the lockfile was not enforced.
The check uses literal costs without project imports, scientific execution or
study replay. It does not execute `summarize_main`, persist study summaries or
verify later documentation/publication edits. Fixture values are not measured
research costs; #14 c0/c5 remain partial and open.
