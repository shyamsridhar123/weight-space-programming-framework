# Changelog

This file records changes to the standalone Weight Space Programming Framework.
Publication versions are separate from the historical `v1` and `v2`
experiment/report directory names. The current version remains an unpublished
research draft.

## Unreleased

- Record effective broker request-count and canonical-JSON byte limits in
  successful metadata and trusted audits as `wsp-broker-request-limits/1`.
  The [control inventory](docs/research/broker-control-inventory.md) distinguishes
  configured policy, enforced admission checks and residual host trust. The
  [authored engineering receipt](provenance/verification/broker-request-limits-20261109.json)
  records the passing boundary check. Metadata and transcript hashes change;
  older receipts without the field leave these limits unrecorded, and exact
  historical replay may require its frozen source. #5 c1 remains partial/open.
  The separate [fixture integration receipt](provenance/verification/broker-fixture-source-binding-20261110.json)
  records a passing check of the reviewed broker identity in literal engineering
  fixtures. Production selection-driver pins remain frozen; default rejection,
  subsequent source drift and permanent invalidation are checked explicitly.

- Sanitize decoded JSON keys as well as string values in public event derivatives.
  Same-object profile-redaction collisions omit the whole event and roll back
  its count; the public string marker may yield `invalid_event`, with no claimed
  raw/public parser equivalence. The [publication contract](docs/research/live-studies.md)
  and [authored offline receipt](provenance/verification/provider-public-event-keys-20261109.json)
  describe one passing synthetic AST regression with a disclosed test-local
  JSON sink. Production writer, live-provider and historical-repair evidence
  remain outside its scope; #5 c2 and #8 c6 remain partial/open.

- Correct pre-return [fit-cost missingness](docs/weight-space/resource-accounting.md):
  `fit_seconds_all_options` remains unknown (`None` / JSON `null`) until eager
  `fit_options` returns, including failures after partial internal fitting.
  Returned options retain their summed cost through later calibration failure;
  returned explicit no-fit options retain observed zero. Existing timer
  boundaries and nesting are unchanged. The [engineering receipt](provenance/verification/fit-failure-cost-20261109.json)
  records one passing regression of selected AST-extracted statements with
  stdlib fakes and literal costs, without full `fit_case_arm` execution, fitting,
  timer measurement, scientific execution or historical cost recovery.
  #14 c0/c5 remain partial/open.

- Extend fresh posthoc [resource accounting](docs/weight-space/resource-accounting.md)
  with coverage and stage statistics for eight fields, retaining known costs on
  missing evaluations. Pair the two enclosing incremental timings within each
  assignment only when both are observed; complementary partial rows never
  manufacture a pair. Exact byte sums remain separate from approximate
  binary64 statistics. Fresh source output uses `wsp-trained-posthoc-analysis/2`
  with the current metrics hash; historical `/1` analyses and geometry
  distributions remain unchanged. The [engineering receipt](provenance/verification/posthoc-resource-coverage-20261109.json)
  records one passing scalar check of four AST-extracted helpers, without full
  analysis, persistence, study replay or scientific execution. Producer early-fit
  undercount remains deferred to a separate PR, and #14 c0/c5 remain partial/open.

- Preserve per-field [resource availability](docs/weight-space/resource-accounting.md)
  across all assigned trained outcomes, including known costs on missing
  evaluations. Fresh `/2` study summaries expose observed sums and counts;
  existing total aliases become null when any assigned value is unknown.
  The [engineering receipt](provenance/verification/resource-coverage-summary-20261109.json)
  records one passing authored scalar seam check. Runtime identity was captured
  and unchanged; lockfile agreement was not enforced. Timers retain their
  producer limits, `analyze_study` remains deferred, and no scientific execution
  or study replay occurred. Historical results remain unchanged and #14 c0/c5
  stay partial/open.

- Add generic Boolean [correctness-transition counts](docs/weight-space/classification-metrics.md)
  so corrections and newly incorrect cases remain visible when net accuracy
  cancels them. The caller owns decoding, regions and rates; existing queue
  metrics stay unchanged. The [authored engineering receipt](provenance/verification/correctness-transition-counts-20261109.json)
  records the literal cancellation example and strict Boolean input boundary.
  This utility is not integrated into scientific evaluation or admission, and
  #9 c1 remains partial/open.

- Specify a non-admitting draft evidence table and seven ordered gates for the
  [prospective staged study](docs/research/staged-scientific-compatibility.md#draft-admission-package).
  Native evaluation, scientific ownership, source/seed/observation evidence and
  margins remain unresolved. No protocol is frozen and no study is selected
  or executed; #9 c1 remains partial/open and #6 c4 remains partial.

- Connect the declared sequential runtime to registered cohort slot evaluation.
  The helper restores against the retained seal, validates every row before
  forwarding, and records predictions, input commitments and a detached runtime
  receipt. A [focused authored fixture](provenance/verification/sequential-slot-evaluator-20261109.json)
  checks the prescribed bias correction, receipt bindings and
  failure that consumes authorization and stops later slots. This is engineering
  evidence for two declared float64 slots; the proposed native-float32 scientific
  companion remains unimplemented. Generic evidence flags stay false, earlier
  receipt scopes and historical outcomes stay unchanged, and #9 c1 stays partial.

- Add an owned runtime for declared `sequential-affine-activation/1` artifacts
  with explicit NumPy/float64/CPU execution, retained expected-hash agreement,
  finite left-to-right affine evaluation and detached state/output. The
  [focused synthetic engineering receipt](provenance/verification/sequential-runtime-restoration-20261109.json)
  records one captured regression selection with three NumPy 2.5 deprecation
  warnings from intentional array-header mutation checks. Coverage is limited
  to the declared adapter; caller trust in the expected hash remains required.
  Refresh selection-only admission's reviewed arithmetic-source pin for the
  shared helper change, retaining its exact source checks and all earlier
  capture and seal records. The earlier runtime receipt does not cover this
  later source-pin update. Generic cohort restoration/usefulness flags remain
  false. No scientific outcome or earlier receipt scope changes; #9 c1 stays
  partial and open.

- Leave descendant cleanup unknown (`None`/JSON `null`) when pipes drain after
  a timeout; retain `False` when cleanup waits are exhausted and preserve
  ordinary completion. The
  [authored offline receipt](provenance/verification/provider-timeout-cleanup-20261109.json)
  records one passing regression with four mocked communication sequences,
  retained output and exact requested wait/kill calls. This establishes no
  native cleanup, containment, output-size bound, provider or scientific
  attestation. Earlier evidence and historical outcomes remain unchanged;
  #5 c1 stays partial and open, and #5 c2/#8 c6 remain partial/open.

- Serialize decoded, sanitized provider events with ASCII-safe JSON escapes so
  a lone surrogate cannot prevent writing the public UTF-8 derivative. The
  [authored offline receipt](provenance/verification/provider-surrogate-publication-20261109.json)
  checks escaped synthetic profile redaction, retained ordinary Unicode, the
  same raw/public parser result for that surrogate fixture and later
  response/usage, unchanged raw bytes
  and inert copied metadata, separate hashes and the exact existing publication
  record. This handcrafted fixture does not establish exhaustive Unicode,
  malformed raw UTF-8 or arbitrary-data confidentiality coverage. No live CLI
  or historical replay was run; Issue #5 c2 and #8 c6 remain partial and open.

- Catch `UnicodeError` from the strict UTF-8 CLI version probe in the existing
  startup handler, retaining a failed attempt/trial and normal broker abstention.
  The [authored offline receipt](provenance/verification/provider-version-utf8-20261109.json)
  records one passing regression that injects `UnicodeDecodeError` at mocked
  `subprocess.run`; actual CLI decoding is not exercised. Its assertions cover
  one probe, no workers or retries despite `max_rounds=3`, empty streams,
  null version/response/returncode and unknown usage/cost, with no usage admission
  or proposal before sealing. Zero admission
  does not establish zero consumption. No historical stream was inspected,
  repaired or rerun; the 90 failures' unknown usage, charges and deployment
  revision, absence of a finite total-cost bound and provider-origin attestation
  remain unresolved. Issue #5 c2 and #8 c6 remain partial and open.

- Reject duplicate JSON object keys in provider events, final responses, encoded
  broker actions and event publication. Retain failed attempts/trials and broker
  abstention; ambiguous usage stays unknown while separately valid usage remains
  recorded. Public copies replace ambiguous event lines with a disclosed JSON
  string marker, retaining the existing metadata shape and transform. The
  [authored offline receipt](provenance/verification/provider-duplicate-json-20261109.json)
  records one passing regression with mocked transport and version probing.
  Raw/public identities remain separate. No historical stream was inspected or
  repaired; unknown usage, charges, deployment revision and provider origin
  remain unresolved. Issue #5 c2 and #8 c6 remain partial and open.

- Rebuild saved-attempt broker trials through the shared driver and compare the
  complete evaluation, including usage-ledger ordering, requested counters,
  denial, admitted charges and totals. Pin requests, schemas and publication
  identities; translate opaque handles while checking every broker event. The
  [authored offline receipt](provenance/verification/broker-usage-replay-20261109.json)
  records one passing capture of ordinary sealing, failed known/unknown usage,
  and denied usage, with re-signed corruption checks. Replay uses a fresh
  temporary directory, not a fresh process, and never constructs a live provider.
  Earlier failed-response evidence retains its captured scope; this later
  supplement supplies no provider regeneration, origin attestation, historical
  repair, real-charge measurement or finite total-cost bound. Issue #8 c6 and
  #17 c2/c3 remain partial and open.

- Add a separately pinned raw-text serving path through preprocessing, encoder,
  native CPU float32 head inference and queue mapping, with exact parent fallback
  after rejected edits. Reuse the captured-parent validator; retain all assigned
  jobs, bounded subprocess failures and partial stage evidence. The
  [authored engineering receipt](provenance/verification/raw-text-serving-20261109.json)
  records one passing capture including one real baseline child. Other deployment
  selections run in parent workers; supervisor fault cases are controlled.
  Whole-batch stage/raw-call/process timers are nested and include disclosed
  instrumentation boundaries. This supplies no trained-lineage benchmark,
  controlled cold-cache result, direct load-through-first-output metric,
  production latency, memory/energy measurement, historical cost recovery or ROI.
  Issues #7 c3 and #14 c1/c5 remain partial and open.

- Attempt budget accounting once for valid, unambiguous reported usage from
  failed provider receipts before abstention, retaining failure status and
  attempt/trial records. Missing, invalid or ambiguous usage stays unknown.
  Denied events preserve requested counters separately from admitted totals,
  which do not measure total consumption. The
  [scoped offline receipt](provenance/verification/provider-failed-response-usage-20261109.json)
  records one execution of six synthetic provider/broker cases. Timeout status
  shares the source guard but has no separately asserted fixture. This supplies
  no replay usage reconstruction at that snapshot, live attestation, historical usage/charge or
  deployment-revision recovery, or finite total-cost bound. Issue #8 c6 and
  #17 c2/c3 remain partial and open.

- Preserve known fitted-option timings and broad fit/selection/materialization
  elapsed when calibration fails, retaining the original error and unchanged-parent
  fallback. The original stdlib-only regression covers its captured attempt-writer
  snapshot. Summaries retain fit/selection records with null calibration, report
  unobserved metrics and means as null, and preserve the all-assigned success
  denominator, including assigned lineages with no record. A separate
  [synthetic summary receipt](provenance/verification/trained-calibration-analysis-20261109.json)
  covers extracted summary statements only; the original regression was not rerun.
  Neither check measures full lifecycle cost or recovers missing historical timings.

- Draft a separately scoped [recognition-repair study](docs/research/recognition-repair-next-study.md)
  with an exposure inventory, development-justified useful gain, matched ordinary
  controls, paired trained-lineage inference and complete lifecycle accounting.
  Freeze and execution gates retain failed assignments and unknown costs.
  This source-only proposal preserves historical criteria and negative results;
  measured complete lifecycle cost is still required for Issue #7 c3, which
  remains partial and open alongside the unresolved Issue #14 and #11 scopes.

- Reject duplicate provider completions and completed agent messages after
  completion through the existing failed-attempt/trial path, retaining the
  parsed response and unknown usage before broker charges or proposed actions.
  The [scoped offline receipt](provenance/verification/provider-completion-lifecycle-20261109.json)
  records one synthetic regression with an accepted preterminal-message control.
  Future shared-parser revalidation of ambiguous prior streams can differ or
  fail; historical occurrences were not inspected. Historical charges remain
  unknown, and Issue #5 c2 and #8 c6 remain partial.

- Add `verify_registered_cohort_evaluation` for a saved successful cohort
  terminal, independently retained commitments and binding, and its durable
  successful registry attempt. The verification-only commitment join uses
  `get_attempt` without scoring, row materialization, endpoint reconstruction or
  a current registry snapshot. The separate
  [verifier engineering receipt](provenance/verification/staged-cohort-evaluation-verifier-20261109.json)
  records one passing literal offline regression at its captured source snapshot,
  with no retry or timeout. Document the independent-trust requirement in the
  [staged diagnosis guide](docs/research/staged-diagnosis.md#saved-cohort-success-verification).
  Repair only the selection consumer's staged-cohort compatibility pin before
  capture; the driver was preserved but not executed. Later publication edits
  retain separate validation scope. This adds no runtime/provenance/freshness or
  scientific claim. Issue #2 c7 and #9 c1 remain partial and open; historical
  failures and evidence scopes are unchanged.

- Add optional ambiguity-aware hidden alignment with a mandatory
  finite nonnegative threshold, global edge-exclusion separation and explicit
  numerical or ambiguity abstention. Preserve legacy matcher arithmetic,
  diagnostic keys and callers. One literal synthetic regression passed in the
  [captured offline check](provenance/verification/checked-hidden-alignment-20261109.json).
  The numerical guard remains an engineering policy without a proof of a
  rounding bound. Native equivalence remains unverified. No training, frozen
  replay or historical evidence is rerun or upgraded. Issue #16 remains five
  supported, two partial and one open criterion. See the
  [design notes](docs/weight-space/design.md).

- Capture supported NumPy string semantic labels as plain Python strings at the
  immutable label boundary, preserving text, order and model identity for JSON
  admission binding. Reject custom string subtypes before conversion. Add a focused
  synthetic regression; historical replay results and partial issue statuses
  remain unchanged.

- Retain validated live-provider receipts before publication. Once a publication
  failure marker is persisted, recovery preserves provider completion and its
  response while selecting the exact parent as interrupted, without another
  provider call, publication attempt or proposed-edit interpretation. Missing
  evaluation and per-field token/latency coverage stay explicit; financial cost
  remains unknown. The [scoped offline receipt](provenance/verification/publication-terminal-20261109.json)
  records one passing literal regression, including a simulated crash after the
  marker and before selection persistence. This does not establish OS crash
  durability, complete publication atomicity or new live-study evidence.

- Retain a disclosed public event derivative when JSON decoding, recursive
  sanitation or serialization raises `RecursionError`. The shared fallback
  redacts exact home paths, JSON-escaped Windows profile prefixes and quoted
  POSIX profile prefixes, counting only retained replacements. The
  [scoped offline receipt](provenance/verification/provider-public-recursion-20261109.json)
  records one passing synthetic invocation with two home-path cases and a
  following shallow event. This change adds no live evidence and does not
  establish complete caller-specific selection terminality.

- Retain provider attempts and brokered trials when deeply nested JSON exceeds
  the interpreter's recursion limit. Event and final-response parsing use their
  existing failure paths; an over-nested action preserves provider completion
  and charged usage while making the trial invalid and sealing by abstention.
  The [scoped offline receipt](provenance/verification/provider-json-depth-20261109.json)
  records the mocked-transport regression. Historical results, unknown usage and
  charges, and provider-attestation limits remain unchanged; Issue #5 c2 and
  #8 c6 remain partial.

- Add a [prospective scientific compatibility companion](docs/research/staged-scientific-compatibility.md)
  mapping seven engineering slots to eleven scientific entries, with separate
  operational fallback and benchmark-specific analysis rules. This documentation
  adds no implementation, protocol freeze, execution or scientific admission;
  historical results and receipt scopes remain unchanged, and Issue #9 c1 stays
  partial and open.

- Reject a present nonstring provider event `type` through the existing
  parse-failure path, preserving failed-attempt/trial accounting, saved event
  commitments and broker abstention. Missing types and unknown string types
  retain their prior behavior. The
  [scoped engineering receipt](provenance/verification/provider-event-type-20261109.json)
  records the mocked-transport regression. Earlier receipts retain their
  captured source scope. Historical unknown usage/costs and provider-attestation
  limits remain unchanged; Issue #5 c2 and #8 c6 remain partial.

- Reject non-finite decoded provider events and final responses before receipt
  hashing, using the existing canonical serializer and parse-failure paths.
  The [scoped offline engineering receipt](provenance/verification/provider-finite-json-20261109.json)
  records retained failed attempts, raw-event commitments and broker abstention.
  Earlier serialization-limit observations retain their original source scope;
  historical results, unknown usage/costs and provider-attestation limits remain
  unchanged. Issue #5 c2 and #8 c6 remain partial.

- Add an optional endpoint-only plan to the study-specific `SelectionAdmission`,
  bound with native-control recipes before preparation. The one-shot
  `seal_study_endpoints()` retains all assigned members and seven ordered logical
  slots, including equal-hash endpoints, in a canonical, exclusively written,
  flushed and fsynced receipt. Selected no-ops remain selected; only absent
  selection can use the frozen parent fallback. Retain actual Candidate identity,
  full artifacts and native proposal/program/lowering payloads, with independent
  materialization, structure and matching outcomes. `verify_study_endpoints()`
  rechecks the saved seal and live retained endpoints without reconstruction;
  integrity or persistence failure permanently closes the originating owner.
  Add one focused synthetic regression and an
  [engineering receipt](provenance/verification/staged-study-seal-20261109.json)
  preserving its initial strict-JSON tuple-key failure and the passing invocation
  after a direct ordered-key comparison fixed production code. Each invocation
  records unchanged before/after source identities and no automatic retry or
  timeout. This accounting seal supplies no final
  scientific study seal, runtime restoration, holdouts or scoring. Prospective
  scientific conditions remain required before preparation. The seven-slot,
  absent-selection fallback plan still needs a prospective decision reconciling
  the earlier eleven-slot design and its usability-based fallback. Historical results
  and receipt scopes and #2/#6/#9 limitations are unchanged; #9 c1 stays partial
  and open.

- Add opt-in, prebound native-control recipes to the study-specific
  `SelectionAdmission` adapter, preserving legacy empty controls and the generic
  framework. `check_native_controls()` constructs random, three shuffled and
  opposite controls through public native lowering and returns a digest-bound
  report with separate materialization, structural usability and realized-norm
  matching outcomes. Recipes fix magnitude 2, float32 rounding and the
  main-referenced sum tolerance; unavailable or unusable mains have no norm
  reference. Retain all assigned members, actual selected Candidate identities
  and separate proposal/program/lowering/artifact hashes, without fabricating
  control Candidates or writing evidence files. The separate
  [native-control engineering receipt](provenance/verification/staged-native-controls-20261109.json)
  records one passing focused synthetic regression covering recipe binding,
  rounded mismatches/no-ops, abstention, unavailable parents and permanent
  live-admission invalidation, without automatic retry or timeout. Its saved
  execution scope is distinct from later documentation, manifest and publication
  integration. The report supplies no final study seal, scoring,
  randomization or prospective registration evidence and leaves U's requirement
  concerning the three shuffled controls unchanged. Historical receipt scopes and
  results remain unchanged; Issue #9 c1 remains partial and open.

- Add `SelectionAdmission.check_repair_usability()` as a separate, digest-bound
  [common-repair structural report](docs/research/staged-diagnosis.md#selection-only-admission-for-fixed-diagnosis-policies).
  It checks retained endpoints around unchanged live admission, preserves the
  complete roster and selected Candidate/artifact identities, and writes no
  evidence files. Native float32 support, direction, metadata and byte checks on
  retained canonical artifacts distinguish usable repairs, authentic unusable
  repairs and unavailable selections, including rounded no-ops and stranded
  attempted artifacts. Signed zeros are canonicalized at artifact capture;
  original checkpoint signed-zero preservation is not established. Realized
  magnitude may differ from the prescribed 2. The existing admission schema and
  preliminary `repair_usable` flag retain their meanings. The separate
  [structural engineering receipt](provenance/verification/staged-repair-usability-20261109.json)
  preserves the first signed-zero assertion failure and the corrected focused
  pass, each with one invocation and no automatic retry or timeout. The check
  separates literal helper cases from genuine admitted endpoints; the earlier
  selection-admission receipt keeps its original source-snapshot scope.
  This structural method supplies no control matching. Final study sealing and
  scientific usefulness remain pending; Issue #9 c1 remains partial and open.

- Retain allowlisted [factory acceptance diagnostics](docs/research/dark-factory.md)
  for completed provider attempts: known outcome, first rejection code and only
  validated criterion IDs. Acceptance, short-circuit and exception behavior stay
  unchanged; legacy receipts are not backfilled. The separate
  [engineering receipt](provenance/verification/factory-assessment-diagnostics-20261109.json)
  records one passing focused offline invocation without retry or timeout.
  Mocked worker checkpoint and validator checks establish no live provider,
  scheduler, recovery or research outcome.

- Add [selection-only admission for fixed diagnosis policies](docs/research/staged-diagnosis.md#selection-only-admission-for-fixed-diagnosis-policies)
  through the study-specific `SelectionAdmission` consumer of public cohort
  phases. It retains three successful normal returns, exact receipt bytes and
  owned artifacts; changed evidence permanently closes the live owner. The
  fixed trained-head interface and policies remain separate from the generic
  framework. Genuine unavailable and terminal failure outcomes are retained,
  while `repair_usable` is only a selected/non-no-op preliminary flag. The
  separate [engineering receipt](provenance/verification/staged-selection-admission-20261109.json)
  records one passing focused genuine-receipt regression: exit 0, one invocation,
  no retry or timeout, using literal artifacts and no registered batches at its
  captured source snapshot. No structural usability, evaluation, final study
  seal, trained ancestry or causal usefulness is established. Source pins cover
  reviewed disk format compatibility only. Issue #9 c1 remains partial and open;
  earlier receipt scopes and historical results are unchanged.

- Route new accepted [factory publication](docs/research/dark-factory.md) through
  deterministic `factory/<run_id>` branches and durably journalled pull requests.
  Pending PRs block the queue until a separate merge is observed and fetched
  remote `main` contains both the exact candidate and reported merge commit.
  The factory does not merge PRs; squash/rebase merges that lose candidate
  ancestry remain pending. Recovery reconciles branch and response loss without
  overwriting conflicting branches or blindly repeating a PR POST. Legacy accepted
  or failed candidates require existing containment in fetched remote `main`;
  unpublished legacy candidates remain pending. Only runs without a checkpointed
  candidate use the existing direct-main failure-accounting path. The separate
  [engineering receipt](provenance/verification/factory-pr-publication-20261109.json)
  records one local offline attempt that timed out after 300 seconds without
  retry; completed test results, JUnit, actual import inventory and runner after
  artifacts were unavailable. A separate post-termination source observation
  matched all 68 guarded launch files before documentation edits. Ordinary
  new-head hosted PR CI remains the next validation. Live GitHub recovery and
  scientific outcomes remain unverified.

- Add a [registered cohort evaluation bridge](docs/research/staged-diagnosis.md#registered-cohort-evaluation)
  after endpoint sealing. A one-shot lazy loader supplies every declared holdout;
  one aggregate registry score authorization covers their rows in plan order,
  including unused holdouts. Trusted slot callbacks receive retained sealed
  endpoints and copied context/rows, with exact finite-JSON result keys. Failures
  stop later slots and preserve unavailable, not-run and unknown registry states
  without retry. The separate [engineering receipt](provenance/verification/staged-cohort-evaluation-bridge-20261109.json)
  records one focused regression: exit 0, `1 passed in 18.47s`, no retry or timeout,
  with unchanged before/after source inventory and tracked diff. This covers its
  captured source snapshot. The generic framework bridge leaves runtime restoration
  and causal usefulness unverified. Issue #9 c1 remains partial and the issue
  remains open; earlier receipt scopes and historical results are unchanged.

- Add an explicit [cohort endpoint seal](docs/research/staged-diagnosis.md#endpoint-seal)
  after the complete repair barrier, materializing referenced origins once while
  retaining every logical comparison slot. Selected endpoints verify saved literal
  replay; structured controls reproduce the complete declared lowering receipt.
  Static capture metadata and construction native maps remain distinct. The separate
  [engineering receipt](provenance/verification/staged-cohort-endpoint-seal-20261109.json)
  records one focused regression: exit 0, `1 passed in 4.30s`, no retry or timeout,
  with unchanged before/after source inventory and tracked diff. Cohort scoring,
  evaluation, runtime restoration and causal usefulness remain unavailable
  or unverified. The single-score bridge remains pending, Issue #9 c1 remains
  partial and the issue remains open; historical results are unchanged.

- Fetch the exact advertised remote commit before reconciling failure-accounting
  ancestry. The [focused engineering receipt](provenance/verification/factory-recovery-remote-fetch-20261109.json)
  records the offline regression for a remote descendant, divergence and fetch
  failure, preserving the original failure and unknown-cost accounting. This
  change does not execute factory recovery or establish live Git transfer behavior.

- Add an opt-in [cohort evaluation-plan declaration](docs/research/staged-diagnosis.md#optional-evaluation-plan-declaration)
  with exact roster/order, holdout/control references, execution declarations and
  immutable capture. Control payloads receive schema-label checks; full parsing
  and materialization remain deferred. Existing arbitrary finite-JSON controls
  remain supported without a plan. The separate [engineering receipt](provenance/verification/staged-cohort-evaluation-plan-20261109.json)
  records one passing focused declaration regression with no retry and no runtime
  availability claim; the permitted case with no selected-source slots was not
  directly exercised. Cohort scoring and evaluation remain disabled, with the
  endpoint/control seal and single-score bridge pending. Issue #9 c1 remains
  partial and the issue remains open; historical results are unchanged.

- Add a [complete ordered repair barrier](docs/research/staged-diagnosis.md#complete-repair-barrier)
  after all cohort repair dispositions, binding diagnosis, preparation,
  disposition and lane evidence. Selected candidates and live artifacts are
  checked before persistence and during readback; evidence mutation, persistence failure
  and cancellation permanently invalidate the attempt. The return list is
  preserved. A separate [engineering receipt](provenance/verification/staged-cohort-repair-barrier-20261109.json)
  records one passing literal regression with no retry. Cohort scoring and evaluation
  remain disabled; the endpoint/control seal and single-score bridge are pending.
  Issue #9 c1 remains partial and the issue remains open, with historical results
  unchanged.

- Add [cohort preparation and selection](docs/research/staged-diagnosis.md#cohort-preparation-and-selection)
  through `PreparedMember` and `StagedCohort`, retaining the complete assigned
  roster and requiring a verified whole-cohort diagnosis barrier before repair
  construction. Supplied subset/native-format configuration is checked before
  lane construction; ordinary preflight failures remain assigned as unavailable.
  Owned lanes reject direct public selection and evaluation. The separate
  [engineering receipt](provenance/verification/staged-cohort-selection-20261109.json)
  records one passing selection regression with no retry at its captured source
  snapshot.
  This adds no scoring bridge or new study evidence; Issue #9 c1 remains partial
  and the issue remains open, with historical outcomes unchanged.

- Add canonical terminal selection outcomes and failure receipts to the
  [staged diagnosis lane](docs/research/staged-diagnosis.md), preserving original
  exceptions and invalidating failed finalization without overwrite or retry.
  The [separate engineering receipt](provenance/verification/staged-terminal-failures-20261109.json)
  retains failed-invocation evidence and both test-source identities alongside
  the passing test-only correction against unchanged production source; the older
  guard covers its original snapshot.
  Issue #9 c1 remains partial and the issue remains open, with historical results
  unchanged.

- Require complete, nonnegative exact-integer input/output token usage before
  provider completion and common broker action admission. Serializable rejected
  usage and event identity are retained in failed receipts; missing counts are
  never defaulted to zero. Non-finite JSON receipt serialization remains limited.
  The [prospective boundary note](docs/research/live-studies.md) and
  [sanitized execution receipt](provenance/verification/provider-usage-admission-20261109.json)
  keep historical unknown usage and costs explicit, with Issue #5 c2 and #8 c6
  still partial.

- Deny recursively nested broker requests through the shared malformed-request
  accounting path, preserving trusted sealing after the request allowance is used.
  The [single synthetic engineering receipt](provenance/verification/broker-ingress-accounting-20261109.json)
  records one local passing test at its captured source snapshot. Issue #5 c1/c3
  remain partial; general host/network containment is not established.

- Added a cooperative [registered session evidence verifier](docs/research/evaluation-data-roles.md#registered-session-evidence-verifier-supplement-november-9-2026)
  over a caller-owned registry, exact selected artifact and trusted expected
  binding, attempt, manifest and result identities. It reads only the durable
  attempt, without evaluation, callbacks, batch materialization or current-state
  snapshotting. A separate [synthetic verification receipt](provenance/verification/registered-evaluation-verifier-20261109.json)
  records checks against rehashed substitutions, missing or failed attempts and
  added held-out charges. Development consumption remains a separate boundary.
  This adds no source attestation or scientific freshness; earlier receipts
  retain their original scope, and Issue #2 c2/c7 remain partial.

- Published a [symmetry-family engineering check](docs/research/symmetry-family-boundary-check.md)
  and [sanitized receipt](provenance/verification/symmetry-family-boundary-20261109.json)
  for one standalone invocation covering finite ReLU gauge/edit transport,
  raw-coordinate negatives and family boundaries. The receipt binds the captured
  source; later integration remains unverified. Issue #16 c0 remains partial,
  with the frozen pilot's 12/48 failed edited-score controls and absence of native
  replay success receipts preserved.

- Added an opt-in [supplied-feature exact-score compiler guard](docs/research/feature-feasibility.md)
  and CLI field. It rejects infeasible or unresolved score equations and
  nonexact stored target/protected residuals. The default numerical path remains
  available. The [recorded standalone check](provenance/verification/exact-score-admission-20261109.json)
  covered approximate compilation, rounded-target rejection, exact acceptance
  and rejection of integer `1` as the boolean option once. The frozen receipt
  does not verify later integration; Issue #6 c4 remains partial.

- Published the November 9 [synthetic trained-development retention comparison](docs/research/retention-development.md),
  its [finite summary](reports/research/retention-development-20261109/summary.json)
  and [verification receipt](provenance/verification/retention-development-20261109.json).
  The frozen cutoff, budget and margin sweep found useful edits only at the
  loosest score budget; the smallest hard-projection cutoffs yielded no-ops.
  Fitting and calibration both participate in development gates. Issue #6 c1
  is supported at this bounded comparison scope; c4 remains partial. Historical
  outcomes remain unchanged, and no policy winner or deployment result is claimed.

- Added a prospective [diagnosis before repair lane](docs/research/staged-diagnosis.md)
  with separate sealed sessions, durable exact-selection receipts and registered
  evaluation gating. The [literal guard receipt](provenance/verification/staged-diagnosis-guard-20261109.json)
  preserves both launch outcomes and its recorded source snapshot; later
  integration validation is not inferred. Issue #9 c1 remains partial and
  the historical negative joint-selection result is preserved.

- Added [bounded supplied-feature score feasibility](docs/research/feature-feasibility.md)
  to the shared readout diagnostics: joint exact consistency, checkable conflict
  witnesses, and minimum squared edit norms, with explicit resource and numerical
  limits. Stored endpoint residuals remain separate from the numerical optimizer.
  The [November 9 verification receipt](provenance/verification/bounded-feature-feasibility-20261109.json)
  binds passing focused and configured engineering checks to unchanged source
  identities. A separate [affine endpoint correction receipt](provenance/verification/affine-endpoint-certificate-correction-20261109.json)
  preserves the subsequent hosted assertion failure and records the corrected
  exact-fraction endpoint check; no tolerance or implementation change was made.
  A later [numerical diagnostic correction](provenance/verification/edit-coordinate-underflow-correction-20261109.json)
  marks edit norms unavailable when final coordinate conversion underflows,
  preserving independently valid residual fields. Its focused check passed;
  the earlier affine-corrected hosted pass remains scoped to its own head.
  Issue #6 c0 is supported at this diagnostic scope; its trained
  cutoff/budget sweep and generic exact-target admission gate remain partial.
  Historical trained outcomes are preserved, with no new scientific study or
  enterprise operational-effect claim.

- Added the November 9 durable evaluation lifecycle: authorization reserves an
  attempt and conservatively consumes observed scope before the callback, releases
  the writer lock during scoring and records terminal or unknown outcomes without
  reuse. A separate prospective support runner provides `prepare`, `evaluate`
  and `verify` stages with preprocessing and predictions inside authorization.
  The [implementation note](docs/research/evaluation-data-roles.md) retains the
  prospective migration, reservation-snapshot and cooperative-scope limits. The
  [five rubric definitions](docs/research/prospective-protocol-rubrics.md) support
  Issue #2 c0 only at definition scope; c2 and c7 remain partial. The
  [durable lifecycle verification receipt](provenance/verification/durable-evaluation-lifecycle-20261109.json)
  records passing focused and configured engineering checks completed on
  November 9, 2026, with captured source identities unchanged. Synthetic fixtures
  and these definitions establish no fresh scientific
  cohort or enterprise service effect.

- Added a shared SQLite data-role registry and optional registered scoring in
  sessions and brokers. Development exposure commits consumption before the
  callback; exact JSON aliases and observed family/ancestry scopes stay
  invalidated across registry restart. Synthetic checks cover rejection before
  scorer entry, callback failures, cancellation and concurrent ordering. The
  [implementation note](docs/research/evaluation-data-roles.md) and
  [verification receipt](provenance/verification/registered-data-role-guard-20261109.json)
  retain the original c7 failure, cooperative scope, absent scientific freshness,
  remaining caller migration and frozen-auditor compatibility limits. Issue #2
  remains partial; this change establishes no useful live-model edit.

- Recorded the November 9 GitHub issue reconciliation in a separate
  [observation receipt](provenance/issue-reconciliation-observation-20261109.json).
  Seven issues closed and eleven remained open after applying the reviewed
  acceptance assessment. All original criterion wording and statuses remain
  preserved; the receipt describes saved observations and journals rather than
  independent server-side causal provenance.

- Integrated the November 8, 2026 study evidence during the November 9 final
  review (America/Chicago). [Trained applications](docs/research/trained-applications.md)
  found useful ownership changes but no useful recognition repair; hard
  protected-score candidates abstained. In the [live studies](docs/research/live-studies.md),
  all 80 primary weight-access assignments abstained and diagnosis useful
  coverage missed its threshold. The [original-source repeat](docs/research/original-source-repeat.md)
  and [construction pilot](docs/research/construction-steering-pilot.md)
  retained four and twelve failed fresh assignments respectively, with no
  completed fresh responses. The [synthetic unlearning study](docs/research/synthetic-unlearning.md)
  completed training, evaluation, and independent checking; both edited methods
  failed the weight-only and archive-inclusive joint gates across all transforms,
  while some individual component gates passed. Scoped replay and control success do not
  establish live-model success, privacy, or native-execution guarantees.
  Package version remains `0.2.0.dev4`; `v0.2.0-draft.4` remains unpublished.

- Added a November 8, 2026 primary-source research review and actionable GitHub
  roadmap. The plan specifies a trained public support-routing study against
  rules, routing overlays, and retraining; feasibility and abstention; candidate
  and proof binding; numerical execution; brokered live access; causal audits;
  cumulative edits; a second trained domain; cost; source repetition; and gated
  transfer, no-fitting construction, and synthetic unlearning. Published a source
  register with acquired-byte hashes, reviewed methods and limitations, and a
  versioned issue index/body snapshot with dependencies and falsification criteria.
  That review recorded planned studies; the completed outcomes are linked above.
  Historical records retain their original scope. See
  [the research review](docs/weight-space/v2/research-roadmap-2026-11.md).

- Rebuilt the enterprise explanation around one outage ticket using STE-Pro-Max
  storytelling and original SVG mathematical animations. The 90-second story
  explains queue scores, feature contributions, the target/protection tradeoff,
  a permitted weight-space direction, the literal edit, potential operational
  value, abstention, rejection, and restoration. The local page recomputes actual
  framework outcomes before playback and offers pause, scrub, chapter navigation,
  reset, light/dark themes, a static reading path, and an understanding check.
  Replaced the earlier recording with a full, uninterrupted browser capture;
  added continuous visible subtitles, VTT/SRT sidecars, an embedded subtitle
  track, source/claim registers, and matching player chapters. Retained the
  credited quiet music. Synthetic and fixed-feature limits remain explicit.

- Cut the real browser demo to 90 seconds at normal speed, retaining the contract,
  candidate comparison, protected edit, final gate, impact, rollback, abstention,
  and rejection. Added a restrained excerpt of Scott Buckley's “Aphelion” under
  CC BY 4.0, with fades, attribution, a reproducible edit plan, embedded chapters,
  and updated player timings and provenance.

- Added a simple local enterprise-routing interface backed by actual Python
  execution and a browser screen recording. The walkthrough explains the
  task contract, candidate tradeoff, literal protected edit, sealed final gate,
  measured routing impact, and exact-parent restoration. It also shows
  infeasible protection and feature-coverage rejection. Published the video,
  captured poster, recording provenance, local launch instructions, and scoped
  visual design guidance.

- Added a runnable enterprise support-routing use case using the generic
  frozen-feature compiler, structural and behavioral admission, exact retention
  receipts, cooperative sessions, sealed evaluation, and rollback. Synthetic
  records cover a successful policy change, infeasible protection, and rejection
  on unseen symptom coverage. Added reproducible evidence bundles, business
  reports, image flowcharts, a measured score plot, and integration checks.
  The walkthrough connects the example to the original source review and current
  research gaps without claiming enterprise deployment or live-provider results.

- Adopted the MIT license for project-authored code, documentation, and diagrams;
  added package license metadata and updated the current rights statements.
  Retained third-party licenses and notices remain applicable.

- Refined the README using STE-Pro-Max's writing and visual guidance. The opening
  now leads with the generic framework and a small edit; an original image shows
  the intersecting target and protection conditions. A comparison table preserves
  both sides of the measured tradeoff, and publication details sit at the end.

The [research plan](ROADMAP.md) defines the next decisions: generic behavioral
admission, candidate-specific formal binding, numerical execution correspondence,
a restricted/provider harness, fresh task studies, and cumulative edit contracts.
Each needs evidence within a declared scope before it can support a stronger claim.

## 0.2.0-draft.4 — prepared November 7, 2026

### Public snapshot candidate

- Prepared a sanitized snapshot from private draft.3 commit
  `2424c592fd8ed6b57217bd7b55224f77d0094455`. The publication plan uses a
  separate repository, fresh Git history, and a noreply author identity while
  retaining the original repository privately. The draft release remains
  unpublished; this entry does not establish a completed public transition.
- Retained the original source identity hashes in
  [the source inventory](provenance/source-files.json). The
  [public sanitation ledger](provenance/public-sanitization.json) binds each
  disclosed derivative or omission to its original bytes and SHA-256.
- Redacted internal package-environment and workstation locators, omitted ten
  Python bytecode cache files, and replaced a raw GitHub UI capture with a
  disclosed minimal source reference. Scientific numerical results, protocol
  identities, and recorded source hashes are not revised to hide these changes.
- Kept historical study freeze, audit, and report checks strict. A sanitized
  frozen lock or changed historical inventory fails original-byte checks;
  saved passing receipts describe their original runs. A new study needs clean
  inputs, a newly fixed protocol, and a new freeze and output tree.
- Added [PUBLICATION.md](PUBLICATION.md) and
  [draft.4 notes](releases/v0.2.0-draft.4.md). The current publication manifest
  describes this candidate; the prior draft.3 manifest is retained under
  [historical publication records](provenance/historical-publication/draft3-publication-manifest.json).

### Validation scope

For the November 7 candidate, validation concerned the publication verifier,
packaging, verifier tests, and the Python suite. That preparation did not rerun
scientific studies or Lean proofs; later study executions are recorded under
Unreleased above. Public npm verification stopped at TLS before metadata or
tarball validation. The root lock omits environment-specific `resolved` URLs
while retaining versions, integrity values, and the dependency graph; public
portability remains unverified. No new hosted-CI success is claimed here.
[VERIFICATION.md](VERIFICATION.md) distinguishes current outcomes from retained
historical results.

## 0.2.0-draft.3 — prepared November 7, 2026

### Changed

- Rewrote the entry guides and framework documents using the pinned
  [STE-Pro-Max guidance](provenance/documentation-revision.json). Explanations
  now start from a concrete edit, show the mechanism and intermediate states,
  and connect each conclusion to its assumptions and evidence. Document length,
  test totals, and theorem totals are no longer presented as quality claims.
- Added a continuous path from the original author report through the generic
  framework, implementation obligations, study design, results, limitations,
  and research decisions. RTS remains a historical adapter; it does not define
  the common interfaces.
- Explained the observed target–retention tradeoff with baseline values,
  normalization, paired comparisons, and world-cluster uncertainty. The results
  retain full-protection no-ops, numerical residuals, precision effects, repeated
  edit failures, and the historical negative RTS result.
- Recast the proof guide around premises and consequences: what preservation
  on a feature span establishes, what sequential drift bounds require, and
  which runtime, extraction, and floating-point obligations remain open.
- Organized verification by the question a check answers. Historical proof,
  scientific, and hosted executions retain their own identities and limits.
  This documentation revision does not report a new scientific or Lean run.
- Updated the Python distribution to `0.2.0.dev3` and the installed-wheel CI
  check to run byte-copied examples outside the source checkout. The wheel's
  metadata contains the revised README.

### Added

- A [worked example](docs/weight-space/v2/worked-example.md) that derives an
  exact target-only edit, a protected edit, and an incompatible request, then
  follows the corresponding API operations and numerical fitting behavior.
- Runnable [edit](examples/edit_retention.py) and
  [session](examples/session_edit.py) examples. The session makes the caller's
  validation, ranking, sealing, and evaluation responsibilities visible.
  Its analytic evaluation rows illustrate the interface; they are not new
  experimental confirmation.
- A standalone [interactive explanation](artifacts/explanations/edit-retention/index.html)
  with an inspectable source model, static comparison, and readable no-JavaScript
  and print paths. It lets readers change the edit and protected set. The
  lifecycle, architecture, and evidence flowcharts remain PNG/SVG images.
- Revision provenance and review records that identify the applied writing
  guidance, document bytes, excerpt execution, arithmetic checks, browser
  observations, and unverified interaction surfaces.

### Preservation and verification

- Preserved original document bytes at explicit inventory mappings before
  replacing the live guides. The portable proof checker resolves its historical
  document inputs through those mappings; the saved proof sources, receipts,
  and historical checker retain their recorded identities.
- Built the current wheel and installed it in a fresh core-only environment.
  The copied examples, dependency check, isolated imports, and CLI entry points
  succeeded. The exact Python fences in the worked guide also ran against that
  installed wheel. Environment and registry limits are in
  [VERIFICATION.md](VERIFICATION.md).
- Reviewed the interactive artifact's arithmetic, desktop and narrow layouts,
  presets, protected-set choices, reset, themes, reduced motion, print emulation,
  and static reading path. Native keyboard activation and slider dragging remain
  unverified; browser rendering does not establish accessibility conformance or
  reader understanding.
- Preserved draft.2 hosted evidence independently of release-asset replacement.
  Final tree, staging, archive, and draft.3 hosted outcomes belong to the exact
  records listed in the [verification guide](VERIFICATION.md#final-draft3-publication-and-hosted-records).

## 0.2.0-draft.2 — prepared November 7, 2026

### Fixed

- Placed pytest temporary fixtures inside the checkout in hosted CI and the
  installation, reproduction, and contribution guides. The first Ubuntu run
  used `/tmp`; the source-freeze containment guard rejected fixtures outside
  the declared source root. The correction supplies repository-local
  `--basetemp` and preserves the guard and original tests.
- Updated the distribution to `0.2.0.dev2` and draft references to
  `v0.2.0-draft.2`. The first draft tag and release notes remain historical.

### Verification and evidence

- Retained the hosted failure, diagnosis, and exact log. Added local suite,
  workflow-validation, wheel-build, and core-installation receipts for the
  corrected snapshot. Its subsequent hosted run succeeded for the recorded
  commit; [VERIFICATION.md](VERIFICATION.md) links the archived receipt and log.
- Regenerated the publication inventory, checksums, and source asset.
  Framework implementation, tests, frozen studies, proof sources, and original
  source identities were unchanged. Earlier successful proof, compatibility,
  and saved-evidence checks were not represented as newly repeated runs.

## 0.2.0-draft.1 — prepared November 7, 2026

### Added for standalone publication

- A Python distribution with a NumPy core and optional PyTorch, test, and
  plotting dependencies; the existing CLI is available as `weight-space`.
  The original `rlcd.weight_space` import namespace was retained.
- Root navigation, proof and reproduction guides, an evidence map, verification
  records, a research roadmap, attribution/rights notes, and contribution guidance.
- PNG/SVG flowcharts with editable rendering source and image identities.
  Original documents containing Mermaid were retained as evidence copies.
- Publication checksums and inventory, plus CI entry points for the generic
  Python suite, publication integrity, explicit generic Lean target, and
  optional historical TypeScript compatibility controls.
- Separate records for standalone Python behavior, historical compatibility,
  core-wheel installation, Lean elaboration/axiom audit, and saved-evidence
  replay. Failed and interrupted setup attempts remain available.

### Preserved from the framework work

- Generic read, diagnose, edit, and create task routes; declared access and
  budgets; candidate histories; validation/selection; sealing; and evaluation
  receipts within their documented caller responsibilities.
- Named-tensor edit algebra, structured lowering, dtype projection receipts,
  frozen-feature editing, exact residual receipts, finite-depth sequential
  semantics/bounds, and a PyTorch bridge.
- Original-report review, implementation requirements, formal sources and
  pinned dependencies, generic study records, saved-evidence audit, and
  integration history.
- Earlier generic/RTS studies, negative and no-op outcomes, receipt repairs,
  and their original reports and identities.

### Historical draft.1 boundaries

These statements describe draft.1. Later live studies and adoption of the
project MIT license are recorded above; their outcomes do not erase the other
unsupported claims.

- The older behavioral `Contract` is specialized to a two-layer network;
  generic admission remains incomplete.
- Real-valued proofs do not establish native arithmetic, compiler/extractor
  correctness, or automatic candidate-to-proof correspondence.
- No live provider experiment, hostile-code isolation, execution attestation,
  fresh confirmatory task set, or useful large-model editing claim is included.
- Existing worlds/seeds are already observed. New runs on them are reproductions.
- No project-wide software license has been selected; archived upstream
  licenses remain intact.

## Historical version labels

`artifacts/weight-space/report-v2` is the second rendering of the first study.
The authoritative generic second-study report is
`artifacts/weight-space/study-v2-report-complete`. The earlier `study-v2-report`
is a preserved incomplete rendering attempt. These directory names are not
standalone software release versions.
