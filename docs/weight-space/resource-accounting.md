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
scientific execution, and does not change `analyze_study`.

## Timer nesting and remaining gaps

Option fit times are nested within fit/selection/materialization elapsed.
Native evaluation, binary64 evaluation, verification and export timers are
nested within heldout workflow elapsed. Adding these nested quantities to
their enclosing timers counts some work twice. The named verification timer
does not establish an exact raw-input or whole-pipeline numerical guarantee.

There is a known producer limit before resource aggregation: `fit_options`
may perform fitting before returning its list. If it raises before return,
the caller's local `options` remains empty and `fit_seconds_all_options` can
report zero despite attempted fitting. The broad fit/selection/materialization
timer retains enclosing elapsed time when the fallback completes, but cannot
recover the missing option timings or isolate fitting. A later calibration
failure can retain the already returned option records. Availability counts
cannot detect or repair this underreporting; even complete fields remain
subject to their instrumentation limits.

The [historical cost report](../research/trained-applications.md#cost-measured-stages-and-their-limits)
retains its own measurement boundaries. Complete cold acquisition,
preparation/encoding, memory/energy, model/device coverage and inference
maintenance remain incomplete. This API establishes no lifecycle total,
amortization crossover or cost advantage, and #14 c0/c5 remain partial and open.

The planned engineering receipt is
`provenance/verification/resource-coverage-summary-20261109.json`. Capture is
pending; this document claims no passing verification run.
