# Evaluation data roles after feedback consumption

Reviewed **November 9, 2026**. Issue #2 criterion c7 remains **partial**:
“A holdout used to tune prompts, thresholds or edit selection is invalidated and
becomes development data.” A frozen exercise completed once on November 9
shows that the frozen callback boundary did not enforce this transition
persistently. All 13 predeclared contaminated evaluation routes executed their
caller-owned callbacks. Two controls also executed. All six predeclared existing
guards rejected before any native evaluator invocation. These are engineering
observations about execution and rejection, with no evidence of fresh-holdout
validity or useful model behavior.

The negative finding is preserved here together with the guards that worked.
The exercise authenticated the saved report, source/protocol freeze, protocol,
results and 54 output/seal objects. Authentication establishes the identities
and bindings of that record; it does not turn the route counts into statistical
confidence or a measured capability gain. The data-role registry described
below is a proposed repair. It was not implemented, tested or used to admit a
model in this exercise.

An enterprise support-routing update provides a generic reason this boundary
matters. A team might inspect routing scores while selecting a policy prompt,
an escalation threshold or a candidate edit, then report improvement on those
same tickets as heldout performance. Inspection can influence selection even
when no weights change. Renaming the tickets, changing a split label or creating
a new programming session cannot supply new evidence. Reusing inspected data
can exaggerate an apparent gain; an execution receipt alone cannot establish
that the new routing policy improves service outcomes.

The frozen exercise used a private snapshot of four already-open historical
development rows from a previously trained routing benchmark. Salted hashing
selected original identifiers before the exercise's model outcomes. This
prospective selection prevented outcome-based selection within the exercise;
it did not make the original data prospective or untouched. All four rows
remained historical development data.

An actual `ProgrammerSession.feedback` callback exposed native Torch CPU
float32 outputs for the first two selected rows. The frozen tuning rule selected
the minimum top-one minus top-two score margin, saving the threshold
`1.7220568656921387`. The candidate was a no-op: no weights changed, no fitting
occurred and no retraining occurred. Feedback inspection and threshold
selection still consumed the data for development. A seal bound that
consumption, the selected threshold, the unchanged candidate and an external
data-role ledger before the evaluation-case predictions.

The ledger marked the consumed originals, identical-content aliases and
registered descendants as development. It was an observation ledger only. It
provided no wrapper, persistent guard or enforcement decision to the APIs
under examination. The exercise therefore asked whether the existing boundary
would refuse callback execution despite this known consumption, rather than
testing an implemented registry.

The seven session variants and six corresponding broker variants account for
all 13 contaminated routes. Every route returned `succeeded` and invoked its
native evaluator exactly once. A dash means no corresponding broker route was
predeclared for that variant.

| Contamination or metadata variant | Session route | Broker route |
| --- | --- | --- |
| Exact consumed data, same session instance | Succeeded; 1 native call | — |
| Exact consumed data, new session instance | Succeeded; 1 native call | Succeeded; 1 native call |
| Renamed split | Succeeded; 1 native call | Succeeded; 1 native call |
| Renamed sample identifiers, identical content and common ancestry | Succeeded; 1 native call | Succeeded; 1 native call |
| Registered changed-content descendants of consumed originals | Succeeded; 1 native call | Succeeded; 1 native call |
| Explicit development split name | Succeeded; 1 native call | Succeeded; 1 native call |
| No split declaration | Succeeded; 1 native call | Succeeded; 1 native call |

The descendant variant reversed the batch and moved feature coordinate zero
one float32 ULP toward positive infinity while retaining the original ancestry.
That is a small derivative of saved features. It is not a new source sample,
an independent semantic example or a newly evaluated encoder transformation.
Identical bytes can establish an exact alias; different bytes cannot establish
independence from a consumed parent.

The session and broker disjoint controls each returned `succeeded`, with one
native evaluator invocation apiece. They used the other two selected rows,
which were not used to tune this exercise's threshold. They show that callback
execution was available for those predeclared controls. Because those rows were
also historical development data, the controls provide no independent
confirmation of holdout freshness, model quality or enterprise usefulness.

The working guards provide a separate, bounded result. Each recorded rejection
matched its predeclared guard, and each occurred with zero native evaluator
invocations.

| Existing guard | Recorded exception | Native evaluator calls |
| --- | --- | ---: |
| Second evaluation on the source session | `SessionSealed` | 0 |
| Second evaluation through the broker | `ValueError` | 0 |
| Explicit sample overlap | `ValueError` | 0 |
| Explicit declared-ancestry overlap | `ValueError` | 0 |
| Unsealed session | `WorkflowError` | 0 |
| Supplied split name absent from declarations | `ValueError` | 0 |

These guards enforce some declared workflow conditions. Their success does not
close c7. In particular, rejecting a supplied split name absent from the
declarations did not prevent the tested routes from omitting split metadata.
Rejecting overlap among identifiers supplied together did not establish a
persistent record of prior consumption across aliases or restarted sessions.

The saved boundary analysis explains this difference. In the frozen source,
`ProgrammerSession.feedback` executed a caller-submitted feedback callback
without persistently invalidating the affected data roles or ancestry.
`ProgrammerSession.evaluate_heldout` checked sealing, one evaluation per session
and the declared split name. Its overlap checks compared caller-submitted
identifiers supplied together. Receipt provenance explicitly recorded
`caller_declared_only` and
`lineage_independence_verified: false`.

Even the successful same-session receipt retained feedback identifiers, content
hashes and ancestry while allowing the later heldout callback. The observed
failure therefore includes exact reuse with visible prior consumption; it is
not limited to discovering aliases with an unknown encoder. Broker delegation
reached the same frozen session boundary. A fresh session object reset local
workflow state without establishing fresh evidence.

The meaning of `succeeded` is consequently narrow: the requested caller-owned
callback ran and a receipt was produced. It does not certify an untouched
holdout, independent lineage, a beneficial edit or admission of a useful model.
Likewise, a receipt that accurately describes caller declarations does not
verify those declarations. An explicit development evaluation can be useful
for diagnostics, but its receipt must retain that role and cannot certify a
fresh holdout.

The proposed repair is a durable shared data-role registry and a common decision
before scoring. The registry would make consumption a property of evidence
identity across the workflow, rather than a property of a session name. The
following design requirements are proposed; none is implemented evidence from
this frozen result.

1. **Resolve the actual evidence identity.** The boundary would resolve a
   trusted source manifest and the actual native payload content, then bind
   known source or semantic-family membership and verified transitive ancestry.
   Stable original identities, exact-content aliases and verified parent chains
   would refer to the same relevant consumption state. Caller-supplied role
   names, content strings, ancestry lists or freshness booleans would not confer
   freshness. Hashes can identify exact content; they cannot prove semantic
   independence. Missing or unresolved provenance would remain unverified.
2. **Commit consumption before exposing feedback.** Feedback, prompt tuning,
   threshold tuning and edit selection would commit their affected sample,
   content, family and ancestry roots before exposed outputs return. This
   invalidation would be monotonic and shared by new sessions and renamed
   splits or identifiers. A changed-byte descendant with a verified consumed
   parent would inherit the parent's development status. Operations would also
   need to retain exposure records so later selection cannot silently treat
   previously inspected evaluation results as untouched evidence. Merely
   observing consumption in a separate ledger is insufficient. A failed or
   uncertain consumption commit would prevent output exposure. Concurrent
   sessions would need a defined ordering of exposure commits and scoring
   authorizations against shared state.
3. **Decide before every scorer.** A common boundary would combine the actual
   evaluation manifest with the current registry to return an explicit
   `allow`, `reject` or `unverified` decision. A request for fresh-holdout
   evidence would require verified identity and no invalidating consumption
   under the declared policy. Known consumed data would be rejected for that
   role; unresolved provenance would remain unverified. The decision would
   occur before native prediction or callback execution, so a rejected request
   would expose no scoring output. A permitted development diagnostic would
   carry its development role throughout the workflow.
4. **Bind the decision into the receipt.** A receipt would bind the decision,
   candidate and selection identity, actual evaluation identity, registry
   version and hash, and evaluator identity. The persisted record would explain
   the decision using the resolved evidence and consumption history. It would
   distinguish permission to run diagnostics from verified eligibility for
   fresh-holdout evidence. A stale registry snapshot or a caller assertion would
   not substitute for the decision at the scoring boundary.
5. **Enforce the same decision at evidence consumers.** Direct session routes
   and broker delegation need the common boundary. Custom scoring consumers
   need it before prediction, including consumers in the frozen implementation
   that scored before
   obtaining a broker receipt. Final evidence and admission consumers need the
   same role binding before relying on a result. Partial integration would
   leave alternate scoring or admission paths outside the claimed boundary.

This would be a boundary for covered, cooperative APIs and evidence consumers.
It would not prevent arbitrary Python callbacks from reading other data or
establish OS isolation. A caller-owned callback that cannot be bound to the
resolved manifest would remain unverified for fresh-holdout evidence.

This proposal separates the closure obligation from the preserved diagnosis.
The frozen admission contract bound structural, real, feature, witness and
backend evidence, but had no persistent data-role ancestry binding. It also
explicitly recorded `useful_business_behavior_verified: false`. Structural
evidence and a
successful backend callback do not fill that gap. Closing c7 would require
implemented enforcement and a separately frozen exercise showing the relevant
decisions before all covered scorers and final consumers. No such fix,
follow-up run or closure evidence exists in the preserved result.

## Implemented boundary and synthetic verification

**Earlier implementation account, November 9, 2026.** This section describes
the optional registry version and source hashes bound by the guard receipt
linked below. Its transaction-spanning scorer and repeatable direct scoring
limitations are preserved as that version's scope. The separately dated
[durable lifecycle supplement](#durable-lifecycle-supplement-november-9-2026)
describes the subsequent source version; earlier successful checks do not verify
that later version.

The source now includes an optional registry integration added after the frozen
exercise. The focused registry, session and broker checks and the configured
engineering suite passed on November 9, 2026. The
[verification receipt](../../provenance/verification/registered-data-role-guard-20261109.json)
binds their commands, times, output commitments and exact source hashes before
and after execution. These are synthetic engineering fixtures. They do not
revise the authenticated historical result, establish that all proposed
requirements are implemented, or change c7's partial status.

The source-preservation index now points to exact archived originals for
[`workflow.py`](../../artifacts/research/source-preservation/registered-data-role-guard-20261109/rlcd/weight_space/workflow.py)
and its
[`workflow tests`](../../artifacts/research/source-preservation/registered-data-role-guard-20261109/tests/test_weight_space_workflow.py).
Their original sizes and hashes remain unchanged. The separate guard receipt
identifies the later implementation and tests exercised by the checks above.

`DataRoleRegistry(path)` stores captured finite JSON batches, observed source
and family declarations, registered parent chains, monotonic consumption and
decision records in one shared SQLite database. `register_batch(examples,
source_id=..., family_id=..., parents=...)` returns a `RegisteredBatch`.
Its `materialize()` and `to_dict()` methods return copies of the captured JSON
and manifest. Source identifiers cannot be reused for different bytes or
lineage. A registered handle is checked against the stored manifest and payload;
the presence of a handle alone does not authorize scoring.

Consumption follows exact per-row content aliases, observed family membership
and registered transitive ancestry. Propagation is conservative across connected
observed scopes, including ancestors and siblings. Registering an alias later
propagates any earlier raw-content consumption into its newly observed family
and ancestry before returning the handle. These rules can invalidate more
evidence than exact-row overlap alone. They still cannot discover an undeclared
semantic relationship or authenticate a declared parent.

`consume(examples_or_batch, reason=...)` commits development consumption before
the cooperating exposure callback. A callback failure does not undo that
commit. `score(batch, scorer)` resolves the current consumption state before
calling `scorer(materialized_json)`, normalizes its result to finite JSON, and
persists the decision. Known development consumption rejects the scoring
request before scorer entry. The earlier guard implementation held SQLite's
`BEGIN IMMEDIATE` transaction across scorer execution and result/decision commit.
The current [durable lifecycle](#durable-lifecycle-supplement-november-9-2026)
instead commits reservation and observed consumption in a short transaction,
closes that transaction before calling the scorer, and finalizes its outcome in
a separate transaction. Writer transactions use a two-second busy timeout;
same-thread scorer reentry remains rejected.

`ProgrammerSession` and `ProposerBroker` accept optional `data_registry`
configuration supplied by the trusted harness. The broker passes that registry
to its session. Registered evaluation requires a `RegisteredBatch` and a
configured registry. Both interfaces expose `evaluate_registered(batch,
evaluator, evaluator_id=..., split_name=...)`; the evaluator receives
`evaluator(candidate, materialized_json)`. The session delegates the registered
batch to the registry's scoring boundary before calling that evaluator. Its
receipt records the registry decision, registry identity, batch manifest hash
and evaluator callback invocation accounting under the `registered_cooperative`
scope.

For configured sessions, the feedback, forward and activation tool paths call
the registry's consumption operation before invoking their caller-owned
callbacks. These paths supply either the registered batch or a captured copy of
the submitted JSON examples to that operation, then pass a materialized copy
to the callback. A consumption error prevents that callback invocation. This
orders consumption before the proposer-facing tool callback can expose its
output; it does not observe arbitrary callback-internal reads or work.

The legacy `ProgrammerSession.evaluate_heldout` and `ProposerBroker.evaluate`
routes remain available with their sealing and single-evaluation behavior.
Their evaluator accepts only the candidate and owns its inputs through an
opaque closure. Those inputs remain freshness-unverified, including when a
registry is configured. Older research callers using these legacy routes remain
legacy and unverified; the optional integration does not migrate their saved
results or retrospectively establish freshness.

Passing precomputed predictions to a scorer cannot establish that prediction
occurred after the registry decision or used the registered materialization.
The registered interface supplies materialized JSON to the evaluator so a
cooperative scorer can perform prediction there. A callback that ignores that
input, returns previously computed scores or reads unrelated data falls outside
what this interface can verify. Its execution cannot establish fresh evidence
merely because a registered batch was also submitted.

The successful focused checks exercised copied materializations, exact aliases
after registry restart, observed families, descendants, conservative sibling
invalidation, late aliases, disjoint controls and both concurrent commit orders.
Forged, tampered, foreign and uninitialized handles rejected before the scorer.
The session checks exercised feedback, forward and activation consumption
before both successful and failed exposure callbacks, then rejected copied data
in a new session. Broker checks exercised the same registered boundary.

Receipts distinguish a rejection with zero evaluator callback invocations from
a failure after one invocation. Invalid results and scorer exception messages
remain sanitized. The cancellation checks retained the original
`KeyboardInterrupt` or `SystemExit`, cleared a preattached decision claim and
attached a copied decision only after successful persistence. A failed evidence
commit did not replace cancellation. These observations account for callback
entry, not the number of native model forward passes inside a callback.

The focused selection ran independently of the engineering configuration. The
registry and broker test files were then added to `pytest-engineering.ini`
before the engineering run. The receipt records each configuration hash
separately, its bookkeeping counts and the three declared engineering
deselections. Source bytes captured for each run were unchanged afterward.
Verification counts are not experimental units, scientific confidence or
business impact. No new scientific study, live provider call or genuine holdout
evaluation is claimed by these fixture runs.

The implementation explicitly retains false provenance-authentication,
semantic-independence, scientific-freshness and evaluator-attestation fields.
Its scope is a cooperative interface in the same process and supplies no OS
isolation. Scoring does not consume or reserve the batch, so direct registry
scoring can be repeated; there is no durable global once-only guarantee.
External observations, arbitrary scorer reads, undeclared transformations and
copied databases remain outside the boundary. When evaluation results influence
later selection, the affected data must be explicitly marked consumed; scoring
alone does not perform that transition. The registry is not an authenticated
source manifest service.

Production research callers have not migrated to this API. Some currently
compute native predictions before obtaining an opaque receipt; wrapping those
precomputed results would be too late. Migration must place every actual
prediction and every input region inside the registered boundary, then bind
the data role into final evidence and admission consumers. Issue #2 criteria
c0, c2 and c7 remain partial.

The frozen `scripts/audit_weight_space_v2.py` compares historical configuration
and receipt shapes exactly. It rejects the new shapes. That is an explicit
compatibility limit, not grounds to weaken the frozen auditor or reinterpret
old evidence. A future protocol and verifier need a separate version that
preserves historical replay. The original c7 negative exercise has not been
rerun and remains distinct from these synthetic checks.

There is also a distinct statistical question. Dwork, Feldman, Hardt, Pitassi,
Reingold and Roth's primary paper,
[Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629v2)
(version 2, September 25, 2015), explains why repeated adaptive testing can
overfit a holdout and develops statistical protection for adaptive reuse.
Its algorithm and validity frameworks go beyond keeping an exposure ledger.
The proposed registry tracks operational consumption and prevents unsupported
freshness claims. It does not implement the paper's protection for adaptively
reused holdouts, supply a generalization bound or solve adaptive statistical
reuse. A system seeking deliberate reusable-holdout validation would need a
separate, justified statistical mechanism and protocol.

The preserved record is bound by the raw identities below. The freeze was a
local prospective source/protocol freeze before exercise predictions, with no
model outcomes generated at freeze time. It was not an external preregistration
or a tamper-proof timestamp. Raw SHA-256 values identify the exact saved bytes;
they do not establish the truth of a scientific or deployment claim.

| Preserved object | Bytes | Raw SHA-256 |
| --- | ---: | --- |
| Research report | 8,903 | `9968014fccc6b5f021a6c3441dcaa9c5bd93b0020f03f878d519eb533c99908c` |
| Source/protocol freeze | 17,276 | `574dfd5a1e88724be02a75d48df78946e97c27d3991b55475ef1ab49b7f52efb` |
| Frozen protocol | 7,555 | `347ef13135c61c33abf636b615ddc70820286bcc357cf29b38a6558ade68b0e1` |
| Saved results | 21,399 | `87874e72e7ca7fcc3b4106990f714ae4e5dc6bea2f2c1857b2afecc04f642c35` |
| Session boundary source snapshot | 39,208 | `72f82f90cdad7c7dae559da331515506bd072fc8bb5450155cf5ab221668c520` |
| Broker boundary source snapshot | 13,310 | `c3ba7be6005b12bf7a42aae76b41948726e05fb854744e7d3d835ae6d3e87373` |
| Admission contract source snapshot | 43,906 | `6167b89038690d6d3389d0ae2314d09ebabbf94a1a1eac71a3d35444b9210019` |

The evidence is limited to one private native execution on historical
development rows. The saved result records no genuine evaluation-bundle access,
provider calls, retraining or public implementation modification. Python
file-open and module-origin observations are bounded provenance observations;
they are not OS or process attestation and do not establish sandboxing. No raw
holdout, feature arrays, model tensors, provider data or workstation receipts
are published in this note.

The exercise establishes an authenticated saved negative finding about these
predeclared callback routes and working guards. It does not establish semantic
independence, fresh prospective validation, production usefulness, a general
admission guarantee or the solution to adaptive statistical reuse. The c7
status therefore remains partial. Any later repair must preserve this original
failure record and report its own implementation, coverage and evidence limits.

## Durable lifecycle supplement: November 9, 2026

**Later implementation account, November 9, 2026, America/Chicago.** The
subsequent source version changes registered scoring to commit an attempt
reservation and conservative observed consumption before scorer entry, release
the SQLite writer lock, and finalize the outcome in a separate transaction.
This supplement describes that source contract. It does not use the earlier
guard receipt as verification of this revision, rerun the original c7 exercise
or turn the synthetic fixtures into fresh scientific evaluation.

`DataRoleRegistry.score(batch, scorer, *, attempt_id=None, binding=None)` now
resolves the captured batch and current shared consumption under a short
`BEGIN IMMEDIATE` transaction. It checks the attempt identity and observed
scope before committing both the durable reservation and monotonic consumption
in that same transaction. The reservation binds the registry, batch manifest,
content and row hashes, copied finite-JSON binding, conservative consumed scope,
registry revisions and state hashes. The `registry_*_before/after` fields
describe snapshots of the reservation transaction, not the current registry
state at terminal finalization. Concurrent disjoint callbacks may advance
registry state while this callback runs. A missing `attempt_id` generates an
identity; it does not bypass consumption. Every committed authorization uses
this lifecycle, including a call with no optional binding.

The evidence role is `evaluation_consumed_on_authorization`. Consuming at
authorization is conservative: a crash between the commit and callback entry
can leave no scorer output while still invalidating the observed scope for
another evaluation. The boundary does not roll that scope back to freshness.
Exact-row aliases, observed families, registered ancestors, conservative
connected siblings and late aliases inherit the existing propagation rule.
New sessions, changed split/source names, new attempt identifiers and registered
descendants cannot release the consumed scope. A new identifier is not a retry
authorization. Existing development consumption still rejects before scorer
entry.

After the reservation transaction closes, the scorer receives a copied
materialization. No SQLite writer transaction spans scorer execution; another
thread/process using the same shared database can inspect or mutate disjoint
registered state while the callback runs. Short writer transactions retain the
two-second busy timeout, and same-thread callback reentry remains rejected.
Finite-JSON normalization and terminal evidence persistence occur after
callback entry. The separate finalization transaction binds a successful result
or a sanitized failure to the original reservation; it does not grant another
attempt.

`get_attempt(attempt_id)` returns a copied durable record, or `None` if no
reservation with that identity exists. Its status meanings are:

| Durable status | What is recorded | Later scoring eligibility |
| --- | --- | --- |
| `authorized_outcome_unknown` | Reservation and consumption committed; callback entry, outcome and result are not durably established; `callback_invocations` is `None` | The reserved observed scope remains consumed |
| `succeeded` | One cooperating scorer callback invocation, finite-JSON result and result hash finalized against the reservation | The reserved observed scope remains consumed |
| `failed` | One cooperating scorer callback invocation and sanitized terminal failure finalized, with no successful result | The reserved observed scope remains consumed |

Cancellation preserves the original `KeyboardInterrupt` or `SystemExit` and
attempts to persist a failed terminal record. If that finalization cannot be
committed, the previously committed record remains
`authorized_outcome_unknown`; evidence failure does not replace cancellation.
An abrupt process termination or unsuccessful post-callback persistence can
also retain that unknown outcome. The API provides inspection rather than a
resume token. Success, failure, cancellation and crash never undo committed
consumption, and an unknown record is not proof that the scorer ran zero times.
Invocation accounting concerns the cooperating callback, not native forward
passes inside it.

The optional `binding` must be a finite-JSON object copied from the caller's
actual frozen prospective protocol, selected candidate and evaluator identity.
Its content hash records the submitted binding; it does not authenticate its
truth. The default empty object has no protocol/selection identity claim while
retaining the same scope-consumption rule. In the configured session path,
`evaluate_registered` constructs a binding from the session configuration hash,
task and parent hashes, actual selected candidate, all submitted candidates and
seal event, evaluator identity and split name. A session configuration identity
is not automatically a complete prospective scientific protocol; a runner must
also bind its actual frozen preprocessing, assignments, margins, comparison and
analysis configuration.

A separately frozen prospective runner must place actual preprocessing and
prediction on the supplied materialization inside the registered callback.
Capturing inputs but computing predictions before authorization is too late.
Opaque legacy evaluation, arbitrary callback reads, ignored materializations,
precomputed returned predictions and copied databases remain outside the
claimed boundary. The new attempt ledger does not retroactively capture every
historical attempt made through the earlier API or an uncovered consumer.
Opening an earlier registry preserves its existing batches, decision bytes and
monotonic consumption; that compatibility operation cannot manufacture missing
historical reservations. V2 once-only authorization is prospective: a batch
scored under V1 but never consumed has no historical reservation and can receive
V2 authorization. Migration cannot reconstruct that missing history.

The new
[`run_prospective_support_routing.py`](../../scripts/run_prospective_support_routing.py)
implements separate `prepare`, `evaluate` and `verify` stages. It reads an
explicit four-file parent (`head.json`, `encoder.npz`, `preprocessing.json` and
`lineage.json`) and caller-supplied development envelopes/configuration.
Preparation records effective configuration and terminal assignments before
evaluation. Input normalization and request identifiers are registered as
observed aliases before exposure; raw evaluation rows are linked to those
registered identities. The scoring callback performs feature preprocessing,
encoding, native parent and candidate forwards and metrics under one durable
reservation for the declared cohort and all admitted candidates. Its result
consumer requires the exact durable successful attempt, binding and result
commitments rather than accepting a legacy receipt or rerunning predictions.

Terminal reports include `sample_threshold_accounting`, with nested schema
`wsp-prospective-support-sample-threshold-accounting/1`. It preserves the frozen
assigned denominator and separates admission, retained scoring and missing
endpoints:

- `assigned_count` counts every frozen assignment; `admitted_count` counts
  admitted proposals, including eligible no-ops, rather than files on disk.
  `scored_count` counts retained successful admitted outcomes in the final
  accepted report. It does not count physical computation or all durable
  records: a verification downgrade clears the report's scored outcomes even
  if a durable successful callback record exists.
- `target_gain_observed_count` and `protected_harm_observed_count` count the
  corresponding observed endpoints. `complete_endpoint_count` requires both.
  `missing_endpoint_count` is assigned minus complete;
  `admitted_missing_endpoint_count` is admitted minus complete. An unavailable
  retained metric endpoint is `null`; unscored hypothetical endpoints remain
  absent. Neither is imputed as a measured failure.
- `sample_threshold_success_count` requires a retained successful admitted
  outcome with both endpoints passing the frozen sample admission thresholds.
  `artifact_coverage` divides admitted by assigned;
  `all_assigned_sample_threshold_success_rate` divides successes by assigned;
  `conditional_sample_threshold_success_rate` divides successes by admitted,
  counting missing admitted outcomes as no threshold success. The conditional
  rate is `null` when nothing was admitted.
- `conditional_target_gain_mean` and `conditional_protected_harm_mean` each
  use all admitted artifacts only when at least one was admitted and that
  endpoint is present for every one; otherwise the mean is `null`. The reducer
  validates the producer's endpoint
  contract: non-boolean finite numbers in `[-1, 1]`, or `null`. Complete means
  use `math.fsum(values) / admitted_count`. Stored threshold flags must be
  booleans matching the frozen comparisons.

The pure reducer checks frozen assignment identity, order and proposal status
against terminal evaluation status. Successful accounting is committed inside
the callback result; the success consumer compares it canonically with the
report and a deterministic reduction of recorded assignments, without replaying
predictions or metrics. Terminal reports include abstention, rejection, failure
and cancellation. When terminal persistence completes, the same accounting is
copied into the terminal manifest. The report write and manifest replacement
are separate operations. Failed or rejected reports retain no scored endpoints.
Preparation and `outcome_unknown` manifests receive no final rates; errors
before a report exists and abrupt process death do not manufacture a terminal
report.

New result, evaluation and manifest schemas use `/2`, and the evaluator is
`prospective-support-native-readout/2`; protocol and binding schemas remain
`/1`. Existing source-identity checks still apply. Earlier prepared runs and
saved reports are not migrated, rewritten or backfilled, and the new success
consumer does not accept earlier report schemas or promote nonsuccess reports.
This accounting supplies neither scientific usefulness nor complete costs;
the [rubrics](prospective-protocol-rubrics.md) retain those separate requirements
and include a mixed-denominator example. Issue #2 c2/c7 remain partial.

This runner requires a separately supplied cohort committed by its frozen
configuration. A caller declaration of no development use remains a declaration,
and frozen sample admission thresholds are not themselves a scientific
useful-effect analysis. The source contract creates neither a new cohort nor
proof that the full prospective rubrics below are satisfied. Synthetic fixtures
exercise only their declared execution and rejection behavior. The
[durable lifecycle verification receipt](../../provenance/verification/durable-evaluation-lifecycle-20261109.json)
records passing focused and configured engineering checks completed on
November 9, 2026. Captured source identities were unchanged before and after
execution; the configured selection retains three declared frozen-study
deselections. These checks establish the listed engineering behaviors only.
That receipt predates sample-threshold accounting and does not validate the
new `/2` result, evaluation or manifest schemas.

The [five prospective protocol rubrics](prospective-protocol-rubrics.md) now
define reading, diagnosis, editing, creation and transfer separately. Each
specifies allowed context/weights/examples/computation, independent units and
ancestry/splits, candidate selection, comparisons, useful-effect margins and
abstention/missingness/sensitivity rules. They are definitions without invented
cohorts, numerical margins or results. Issue #2 c0 is supported only for those
five complete rubric definitions and their requested field inventory. A runnable
study must freeze its actual
values and assignments before selection or outcomes. Any separate support-routing
fixture remains synthetic and cannot stand in for a fresh cohort or enterprise
service outcomes.

All lifecycle records retain `provenance_authenticated: false`,
`semantic_independence_verified: false`, `scientific_freshness_verified: false`
and `evaluator_execution_attested: false`. The source implements a cooperative
boundary over captured finite JSON and observed declarations. It cannot
authenticate source provenance, discover undeclared semantic near duplicates,
attest arbitrary execution or prove fresh sampling. Issue #2 c2 and c7 remain
partial: observed lineage and exact aliases are enforceable within this scope,
while semantic independence, complete historical-attempt capture and a fresh
prospective cohort require separate evidence. The historical 13-route c7 failure,
two historical-development controls, six working guards and original object
identities above remain unchanged.

## Registered session evidence verifier supplement: November 9, 2026

**Later source and engineering account, November 9, 2026, America/Chicago.**
`workflow.verify_registered_evaluation` checks saved registered session evidence
against a caller-owned registry, selected realized artifact and independently
trusted expectations. The separate [synthetic verifier receipt](../../provenance/verification/registered-evaluation-verifier-20261109.json)
records one completed check against captured source. It covers the trusted
identity, terminal-attempt and accounting boundaries, including rehashed
substitutions and failed or unknown attempts. It is not a scientific study and
does not extend the earlier guard or durable lifecycle receipts to this verifier.

The API accepts an `EvaluationReceipt` or its finite-JSON dictionary and
requires `registry`, `candidate`, `expected_binding`, `expected_attempt_id`,
`expected_manifest_sha256` and `expected_result_sha256`. The registry is the
trusted `DataRoleRegistry`; the candidate argument is the selected `Network` or
`TensorArtifact`, not its session wrapper. The caller must retain all expected
identities independently of the untrusted receipt. Reading them back from that
receipt supplies no independent authorization.

The exact `wsp-session-evaluation-binding/1` field set is `schema`,
`protocol_sha256`, `task_sha256`, `parent_sha256`, `candidate_sha256`,
`selection_sha256`, `evaluator_id` and `split_name`. The candidate hash binds
the selected session wrapper, distinct from its realized artifact hash.
The selection hash commits the selected wrapper, all successful candidates and
the seal event. The protocol hash identifies the session configuration; it
does not by itself commit a complete prospective scientific protocol.

Verification checks receipt and configuration commitments, the wrapper and
exact supplied artifact, membership in the saved candidate list, trace links
and cumulative accounting. It requires one successful seal immediately before
the sole terminal successful held-out evaluation, matching selection and
result, no held-out proposer charges and unchanged proposer usage. It accepts
only the registered artifact/program success lane, not answer-only, abstained,
rejected, failed or opaque legacy evidence.

The only registry access is `get_attempt(expected_attempt_id)`. That lookup
must return a durable successful attempt with reason `observed_boundary_passed`,
one cooperating callback invocation and evidence role
`evaluation_consumed_on_authorization`. Registry, attempt, manifest, binding
and result identities must match the trusted expectations. The embedded
decision and attempt must agree with the durable record, including its
reservation/lifecycle commitments and recorded reservation revisions and state
hashes. The receipt result must match both the expected result hash and durable
result. A rehashed substituted wrapper, binding, result or attempt cannot
inherit the original durable authorization. Missing, failed and
`authorized_outcome_unknown` attempts do not satisfy verification.

No evaluation, callback, scoring, batch materialization or current registry
snapshot occurs during verification. Later registry changes do not invalidate
a recorded successful attempt. Development `consume(...)` and configured
session exposure guards remain separate operations before development use;
the verifier neither consumes data nor reconstructs unrecorded exposure or
historical attempts.

The registry and expectations remain caller-trusted inputs. The verifier
does not attest evaluator execution or source provenance, check evaluator
output for leakage, authenticate undeclared ancestry, establish semantic
independence or scientific freshness, or certify useful behavior. Provenance,
independence, freshness and evaluator-attestation fields remain false.
Issue #2 c2 and c7 remain partial, with 6 supported criteria, 2 partial criteria
and a keep-open recommendation. All historical outcomes, dates and the earlier
13-route c7 failure, two development controls and six working guards remain
unchanged.

## Development consumption purpose binding: November 10, 2026

The current `consume(examples_or_batch, reason=...)` decision binds the caller's
validated purpose in `consumption_reason_sha256`, the SHA-256 digest of the
reason's canonical finite-JSON bytes. The field is included in the canonical
decision body committed by `decision_sha256`. The generic outcome reason remains
`development_consumption_committed`; the supplied purpose text is not copied
into the decision.

The invariant is that changing a valid declared purpose changes its decision
commitment even when the scope was already consumed. The precise assumptions
are a trusted, valid registry; successful transactions; the same captured rows
and observed registered scope; no intervening registration or consumption;
and every key reachable by conservative scope propagation already consumed.
Purpose and decision identity separation additionally assume no SHA-256
collision for the compared canonical bytes. The registry does not normalize a
valid purpose string before committing it.

Under these assumptions, `_consume_keys` has no new keys to insert, so it leaves
the original consumed-key reasons, their revisions and the registry revision
unchanged. The registry state hash covers registrations and consumed keys, not
the decisions table, so recording either decision leaves that state hash
unchanged too. Two different purpose commitments nevertheless give different
canonical decision bodies and, under the collision assumption, different
decision identities. Both decisions can therefore persist through
`INSERT OR IGNORE` without rewriting consumption history. Repeating the same
purpose at the same state may still return the same decision identity; this
contract does not assign a unique identity to every invocation.

The focused regression warms consumption first and then supplies two different
purposes for the same scope, separately for raw rows and a `RegisteredBatch`.
It checks exact purpose commitments, canonical decision binding and durable
storage after reopening the registry, unchanged state and consumed-key history,
absence of the purpose text from decisions, and continued scoring rejection
before callback entry. These are synthetic engineering checks, not a new
scientific study or a retrospective extension of the November 9 receipts.

The purpose is still a caller declaration. Its hash is neither encryption nor
authenticated provenance. This additive decision field does not establish
freshness, semantic independence or evaluator execution, change the false
authority flags, migrate earlier decisions, or close Issue #2 c2/c7.
