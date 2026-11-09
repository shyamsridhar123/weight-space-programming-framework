# Staged diagnosis before repair

`StagedDiagnosisRepair` adds a prospective lane in which a diagnosis is selected, sealed and durably recorded before a separate repair session is created. It works over the framework's generic captured tensor artifacts. A later evaluator can intervene only after both exact selections verify.

This closes an engineering ordering gap. **Issue #9 c1 remains partial.** The historical blind-diagnosis study selected diagnosis and repair jointly, and its negative result remains unchanged: gain **0.028333**, interval **[-0.019172, 0.075839]**, useful coverage **0.083333**. This lane and its literal guard establish neither causal repair usefulness nor natural fault or backdoor discovery. See the [acceptance record](issue-acceptance-status.md), [live-study record](live-studies.md) and [prospective rubric](prospective-protocol-rubrics.md).

The sequence is:

1. Create a new, exclusive run directory and write its protocol receipt.
2. Select one diagnosis through metadata/weight-only JSON queries. Validate the strict schema, seal the diagnosis session, write its private receipt with flush and file `fsync`, and verify its exact bytes and terminal selection.
3. Create a separate repair session whose task constraints bind the protocol and the exact diagnosis receipt, candidate, task, session and terminal-seal identities. Select and seal one structured repair, then persist and verify its private receipt.
4. Verify both stages before using the existing registered-evaluation authorization path. Verify them again immediately before the trusted evaluator callback, and persist the resulting evaluation audit.

This lane supersedes the rubric's joint diagnosis-plus-intervention submission rule only for new runs using this module. It changes no frozen protocols or outcomes.

## Minimal integration

The trusted harness supplies an immutable `base` artifact, a shared `registry`, an existing `registered_batch`, and its evaluator. Keep the same registry across the runs whose data-role constraints must be coordinated. The run directory must not already exist.

```python
from rlcd.weight_space.staged_diagnosis import StagedDiagnosisRepair

lane = StagedDiagnosisRepair(
    "runs/staged-diagnosis-001",
    task_id="staged-diagnosis-001",
    objective="Diagnose a captured tensor fault before proposing its repair",
    base=base,
    registry=registry,
)
diagnosis_receipt = lane.select_diagnosis(choose_diagnosis)
repair_receipt = lane.select_repair(choose_repair)
evaluation_receipt = lane.evaluate_registered(
    registered_batch,
    evaluate_candidate,
    evaluator_id="registered-evaluator-v1",
)
```

This is an integration sketch: `choose_diagnosis`, `choose_repair`, `evaluate_candidate` and the harness-owned values must be supplied. The module does not run a provider, train a model or generate a study on its own. Optional `view`, `limits` and `native_dtypes` reuse the existing broker controls.

`choose_diagnosis(query)` receives a JSON query function. Allowed operations are `metadata` and `weights`, subject to the broker's view and budgets. It returns exactly these fields:

```json
{
  "location": {"tensor": "layer.weight", "index_prefix": [0]},
  "mechanism": "A proposed explanation for the captured fault",
  "predicted_effect": "The behavior a later intervention is predicted to change",
  "confidence": 0.5,
  "uncertainty": "What the available evidence does not establish",
  "abstain": false
}
```

Tensor names come from the captured artifact. The integer prefix must be in bounds; an empty prefix denotes the entire tensor, including a scalar. The three explanation fields must be nonblank text. Confidence must be finite and between zero and one; booleans are rejected. An explicit abstention requires `abstain: true` and `location: null` and prevents repair construction. Extra fields, including repair commands, are rejected.

`choose_repair(frozen_diagnosis, query)` is invoked only after diagnosis verification and after construction of its separately bound repair session. It receives a copy of the selected diagnosis and the same restricted JSON query surface. It returns exactly `{"commands": [...], "intent": "..."}` using the existing structured-edit vocabulary, or `None` to abstain. A failed or invalid selection closes that stage; a valid, nonabstaining endpoint is required for evaluation.

`evaluate_candidate(frozen_diagnosis, candidate, rows)` is a trusted harness callback. Proposer callbacks receive neither the coordinator nor a broker, candidate endpoint, evaluation rows or evaluator through this API. The registered broker performs the existing data-role authorization and records its evaluator attempt. The returned receipt binds the diagnosis and repair receipt identities to that audit.

## Cohort preparation and selection

[`StagedCohort`](../../rlcd/weight_space/staged_cohort.py) owns the lanes for a fixed, nonempty ordered roster of unique member IDs. Its constructor records that complete roster, protocol, controls, registry identity and evaluator/attempt identities before any preparation callback runs. The trusted `prepare_member(row)` callback receives `member_index` and `member_id` and returns `PreparedMember` with the captured `base`, `task_id`, `objective` and JSON `preparation` record; optional `view`, `limits` and `native_dtypes` use the existing lane controls. The controller constructs each lane itself.

```python
from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort

cohort = StagedCohort(
    "runs/staged-cohort-001",
    member_ids=("case-001", "case-002"),
    protocol=protocol,
    controls=controls,
    evaluator_id="registered-evaluator-v1",
    registry=registry,
    attempt_id="staged-cohort-001",
)
preparation_dispositions = cohort.prepare(prepare_member)
diagnosis_barrier = cohort.select_diagnoses(choose_member_diagnosis)
repair_dispositions = cohort.select_repairs(choose_member_repair)
```

This is a selection-only integration sketch. The harness supplies the protocol, controls and callbacks. `choose_member_diagnosis(row, query)` uses the diagnosis schema above; `choose_member_repair(row, frozen_diagnosis, query)` uses the repair schema above.

Preparation finishes across the roster before diagnosis selection begins. Before constructing a lane or broker, the controller validates the supplied task and configuration, including subset coordinates against the captured tensors. A supplied native-format mapping must cover every captured tensor exactly with supported formats, and the base must already be representable in those formats. Ordinary preparation callback and configuration-preflight failures remain assigned as `unavailable`. A later lane-setup failure invalidates the cohort attempt. The diagnosis pass records every original member's disposition and verifies each prepared lane's terminal diagnosis receipt before writing and verifying the complete diagnosis barrier. No repair task or broker is constructed before that barrier exists. The controller rechecks the full barrier before each repair and binds the repair to the cohort, roster, member, preparation and diagnosis evidence.

Only a `selected` diagnosis can proceed to repair. Unavailable preparation, diagnosis abstention, invalid submissions and selector failures remain in the ordered dispositions and receive an unavailable repair disposition without a fabricated repair receipt. Repair selections retain their own terminal outcomes. These records preserve the assigned denominator; they do not supply scored endpoints or turn a conditional selection rate into all-assigned success.

Each phase is available once. An ordinary selector exception permits continuation only when its terminal receipt verifies; missing or changed evidence, persistence failure and cancellation invalidate the cohort attempt. Cohort-owned lanes reject direct public diagnosis/repair selection, so an individual lane cannot bypass the full-cohort barrier. Their public evaluation method is also denied: cohort evaluation uses the separate registered bridge below and adds no study evidence by itself. It is a trusted cooperative controller, not a Python sandbox, crash-recovery mechanism or independent execution attestation.

### Selection-only admission for fixed diagnosis policies

[`SelectionAdmission`](../../scripts/staged_diagnosis_selection.py) is a study-specific consumer of the public `StagedCohort` phases. The framework remains generic over captured tensor artifacts; this helper fixes the prospective `bias-outlier` and `metadata-uniform` policies, their views and budgets, and the common repair. It accepts one fresh, exact `StagedCohort` with matching roster, protocol and identities, and no `evaluation_plan` key. Controls default to the legacy empty object; the optional native-control recipe below must instead be supplied identically to both constructors before `run()`. The cohort constructor creates the previously nonexistent run directory before the admission owner is constructed. Do not call the cohort phases separately.

After the trusted harness has selected reviewed source bytes, a concrete integration is:

```python
import hashlib
from pathlib import Path

from rlcd.weight_space.staged_cohort import PreparedMember, StagedCohort
from scripts import staged_diagnosis_selection as selection

run_dir = Path("runs/staged-selection-001")
assignments = [
    {"member_id": "case-outlier", "policy": "bias-outlier", "metadata_choice": None},
    {
        "member_id": "case-metadata",
        "policy": "metadata-uniform",
        "metadata_choice": {"row": 2, "direction": "bias_down"},
    },
]
protocol = {"purpose": "selection-only fixed-policy engineering integration"}
evaluator_id = "fixed-policy-evaluator-v1"
attempt_id = "staged-selection-001"
driver_sha256 = hashlib.sha256(Path(selection.__file__).read_bytes()).hexdigest()

cohort = StagedCohort(
    run_dir,
    member_ids=[item["member_id"] for item in assignments],
    protocol=protocol,
    controls={},
    evaluator_id=evaluator_id,
    registry=registry,
    attempt_id=attempt_id,
)
admission = selection.SelectionAdmission(
    cohort,
    run_dir,
    assignments=assignments,
    protocol=protocol,
    registry_id=registry.registry_id,
    evaluator_id=evaluator_id,
    attempt_id=attempt_id,
    driver_sha256=driver_sha256,
)
evidence = admission.run(prepare_member)
checked_evidence = admission.admit()
repair_structure = admission.check_repair_usability()
```

The harness supplies the shared `registry` and `prepare_member(row)`. The callback receives a copied `member_index`/`member_id` row and returns an exact `PreparedMember` with its base, nonblank task ID and objective, and finite JSON preparation object. The helper supplies the fixed view, budgets and native map, overriding optional callback settings. The assignment list is nonempty, ordered and finite JSON, with unique nonblank member IDs and exactly the fields shown. An outlier assignment requires `metadata_choice=None`; a metadata assignment requires an integer row from 0 through 7 and direction `bias_up` or `bias_down`. These supplied choices are frozen inputs, not evidence that randomization occurred. Computing the driver digest in this example does not itself constitute source review.

This consumer requires an exact `TensorArtifact` with architecture `trained-frozen-readout/1`, exactly `head.weight` of shape 9 by 32 and `head.bias` of shape 9, and finite values exactly representable in float32. The tensor schema still serializes values as binary64 (`dtype="float64"`). Weight row 8 must contain numeric zeros and bias row 8 must equal -64. Required metadata identities `encoder_sha256`, `preprocessing_sha256` and `feature_id` must each be lowercase 64-character hexadecimal strings. The helper clones the captured base through public serialization and uses `{"head.weight": "float32", "head.bias": "float32"}` as its construction native map. These checks establish the captured interface and declared identities; they do not establish training or genuine trained ancestry.

The outlier policy makes one weight request for the eight ordinary bias coordinates, under a subset view with two tool calls and eight weight values. The metadata policy has no weight view, one tool call and zero weight values, and makes no queries. Both sessions allow one proposal and zero feedback, execution and token budgets; no feedback, forward or activation callback is registered. Repair makes no queries, including metadata. It uses only the sealed diagnosis to add -2 for `bias_up` or +2 for `bias_down` at the diagnosed ordinary bias coordinate, with intent `common-repair-v1`. The helper reproduces structured lowering with float32 native formats and `cast="round"`, verifies the lowering, parses the public literal program and checks exact-parent application. It also checks the fixed diagnosis's strict nested payloads.

`run()` is one-shot. It captures independent finite-JSON snapshots of successful normal returns from preparation, diagnosis selection and repair selection. It verifies the complete diagnosis evidence before any repair construction, then checks the framework's complete repair barrier. Files left by an interrupted or failed phase cannot recreate its missing normal return. A premature `admit()` or repeated `run()` closes the owner. After a successful run, `admit()` rechecks the retained evidence and returns a fresh JSON copy of the frozen admission record.

Admission rejects duplicate JSON keys, nonfinite values, unexpected schema fields, noncanonical receipt bytes, links or reparse points, and unexpected or missing evidence files. It retains accepted receipt bytes, phase-return snapshots and owned parent/repair artifact clones, then rechecks their serialization and hashes. The receipt body digest excludes the `sha256` field; the separate file digest covers the complete canonical receipt bytes and final newline. Missing or changed evidence, source bytes or owned objects permanently makes this live owner ineligible, even if the original bytes are restored. Genuine preparation callback failures can remain assigned as unavailable, and genuine denied, failed, invalid or abstained terminal stages retain their outcomes and evidence. No unavailable lane receipt is fabricated. A successful proposal followed by a denied seal remains an attempted candidate in the genuine receipt's `candidates`; the selected `candidate` and exported `repair_candidate` remain null.

The returned `wsp-staged-selection-admission/1` record binds assignments, run identities, source hashes, phase-return hashes, both barriers and each receipt's body/file hashes. Its complete member entries include the captured parent, diagnosis candidate, selected repair candidate, outcomes and `repair_usable`. The existing schema and that field's preliminary meaning remain unchanged: `repair_usable` reports selected-candidate presence and `Candidate.no_op` only. Full captured artifacts and receipt details remain private harness evidence.

`check_repair_usability()` adds a separate structural check after successful admission. It uses the retained driver-owned parents and endpoints, checks admission before and after the calculation, and writes nothing into the evidence directory. Every call returns a fresh finite-JSON `wsp-common-repair-structure/1` report. Its `selection_admission_sha256` binds the exact admitted document, including its transitive phase, receipt and source identities; its own `sha256` hashes the report body without that field. The header retains the run and roster identities. Member entries preserve the original roster order, identity, policy, both stage outcomes and diagnosis/repair receipt links, plus `parent_sha256`. Selected entries distinguish `selected_repair_candidate_sha256` from `selected_repair_artifact_sha256` and record `intended_row`, `diagnosed_direction` and `prescribed_delta`.

| Structural status | Meaning |
| --- | --- |
| `usable` | An authentic selected repair passes all common-repair structural criteria. |
| `unusable` | An authentic selected repair fails a structural criterion; its selected identities remain recorded. |
| `unavailable` | There is no selected repair. Selected hashes, direction and all repair numerical fields are null. Original outcomes and receipt links retain the reason the stage was unavailable or failed. |

The structural criteria require unchanged architecture, metadata, tensor names and shapes, and finite values exactly representable in float32. The diagnosis must name a strict integer ordinary row from 0 through 7. Exactly one numerical coordinate must change, at that row of `head.bias`, with a nonzero delta opposite the diagnosed mechanism: negative for `bias_up`, positive for `bias_down`. Native float32 bytes of the retained canonical artifacts must match everywhere outside that coordinate across both head tensors, including reserved row 8. `TensorArtifact` canonicalizes signed zeros to `+0.0` during capture, so the report does not establish preservation of the original checkpoint's signed-zero bits. An endpoint retained from an attempted proposal followed by a denied seal remains unavailable because its receipt selected no Candidate.

`realized_native_delta` is the endpoint float32 value minus the parent float32 value after promotion to binary64. `realized_native_norm` is its absolute value once the permitted support is established. An authentic native no-op is `unusable` with reason `no_op` and both values `0.0`. A wrong direction retains the singleton norm; changes outside the intended support have reason `outside_intended_support` and null norm, although the intended-row delta remains available. Incompatible interfaces or non-native values have null delta and norm. The prescribed command remains magnitude 2; structural usability does not require the realized magnitude to equal 2. At native bias `33554436.0`, adding or subtracting 2 can realize a delta of +4 or -4 and pass; `33554432.0 + 2` realizes no movement and fails. The separate native-control report compares realized norms.

Integrity and identity-binding failures propagate permanent admission rejection, including mutation observed during the calculation; they do not become ordinary `unusable` results. This structural report supplies no operational fallback, matched controls, final study seal or behavioral success evidence.

`check_native_controls()` optionally constructs comparison endpoints after successful live admission. Bind the complete recipe through `controls=controls` in both the `StagedCohort` and `SelectionAdmission` constructors. The shape for a single-member roster is:

```json
{
  "schema": "wsp-native-control-recipes/1",
  "members": [{
    "member_index": 0,
    "member_id": "case-outlier",
    "random": {"row": 1, "direction": "bias_down"},
    "shuffles": [
      [1, 2, 3, 4, 5, 6, 7, 0],
      [2, 3, 4, 5, 6, 7, 0, 1],
      [3, 4, 5, 6, 7, 0, 1, 2]
    ]
  }]
}
```

The recipe has exactly these fields and must contain every original member in exact roster order. Each random choice has a strict integer row from 0 through 7 and direction `bias_up` or `bias_down`. Each member supplies exactly three eight-element integer permutations, each with no fixed points. A map's orientation is `mapping[diagnosed_row]`: it selects the shuffled target row while retaining the diagnosed mechanism. The random direction is likewise a supplied diagnosis mechanism; common repair applies the opposite sign. The opposite control reverses the mechanism at the diagnosed row. The schema fixes intent `common-repair-v1`, prescribed magnitude 2, float32 for both head tensors and `cast="round"`; none is an adjustable recipe field.

Construction checks exact recipe equality and its digest against the cohort's already persisted `protocol.json` before any phases run. It captures independent canonical recipe bytes, so later caller mutations cannot change the bound inputs. Omitting `controls` means `{}` and cannot admit a nonempty-controls cohort. Calling `check_native_controls()` without prebound recipes rejects and permanently closes the owner.

Each report retains five logical controls in order: `random`, `shuffled_1`, `shuffled_2`, `shuffled_3`, `opposite`. Random construction requires only a retained parent, including after diagnosis abstention. Shuffled and opposite controls additionally require an authentic selected, nonabstaining diagnosis. They can still be constructed when the main repair is unusable or unavailable. Without a parent, controls have no operational hashes or realized native effects; the random recipe's prescribed row and direction may still be shown. Controls use public structured lowering, lowering verification and exact-parent literal application, followed by the same structural criteria as the main repair. They are driver-owned artifacts, with no broker Candidate fabricated for a control. Coincident instructions or endpoints can legitimately share hashes.

The report separates `materialized` and `unavailable_reason` from each control's `structure` and `matching`. A constructed native no-op is structurally `unusable`. Only a structurally usable selected main supplies `main_reference_norm`; otherwise that field is null and all comparisons report matching `unavailable`, with reason `no_usable_main_reference`, even if a control itself is usable. An unusable main's norm of zero never becomes a matching reference. With a usable main, unavailable or unusable controls retain those reasons instead of becoming mismatches. Two usable endpoints are `matched` exactly when `abs(control_norm - main_norm) <= 1e-5 + 1e-5 * abs(main_norm)`; otherwise they are `mismatched`. This version fixes that main-referenced sum tolerance and never retunes commands after rounding or mismatch.

The detached finite-JSON `wsp-native-control-matching/1` report binds the admission and main structural-report digests, controls digest, run/roster/protocol identities, both barriers and reviewed source identities. Its body digest excludes `sha256`. Every member retains the complete main structural entry and diagnosis Candidate identity when present; materialized controls record separate proposal, literal program, lowering-receipt and artifact hashes, prescribed deltas, realized native effects and matching outcomes. The calculation rechecks local owned control snapshots and live admission, including after construction. Integrity failures permanently close the owner. It writes no evidence files, retains no final study endpoint seal and authorizes no evaluation. A future study seal must capture and revalidate the actual comparison endpoints.

These recipes can be bound for engineering without registering prospective v2/v3. Supplied choices and maps establish neither an independent randomization, seed or pairing procedure nor prospective registration. This checker does not compute the prospective usefulness indicator `U` or change its existing requirement concerning the three shuffled controls; random and opposite controls add no new prerequisites to `U`. Its evaluation, final study seal and causal usefulness flags remain false.

The source-pin scope is `reviewed_disk_format_compatibility_only`. Pins check reviewed on-disk format compatibility, not loaded-code identity, execution provenance or malicious Python behavior. Ownership rejects a second admission owner for the same live cohort object in the same process; it provides no durable identity, reload/resume, race defense or sandbox. This helper uses public framework methods and parsers rather than private framework state. It supplies no final study endpoint seal, seeds, training, holdouts or scoring, and invokes neither `seal_endpoints` nor `evaluate_registered`. The framework repair barrier is not a final study seal. The admission's `evaluation_available`, `final_study_seal`, `trained_ancestry_established` and `causal_usefulness_established` remain `false`. Issue #9 c1 remains partial and the issue remains open; historical results and earlier receipt scopes are unchanged.

The separate [selection-admission engineering receipt](../../provenance/verification/staged-selection-admission-20261109.json) records one passing focused genuine-receipt regression at its captured source snapshot: exit 0, one invocation, no retry or timeout. It used literal artifacts and no registered batches to check both fixed policies, preserved failure outcomes, duplicate live ownership, integrity-change/restoration rejection and interrupted phase returns. That earlier receipt does not cover the added structural or native-control checkers or later source changes, and establishes no trained ancestry or scientific readiness.

### Optional evaluation-plan declaration

Supplying `protocol["evaluation_plan"]` opts into a declaration contract validated before the exclusive run directory is created. Without that key, `controls` retains its existing contract of an arbitrary finite JSON object. The constructor captures the normalized protocol and controls in the immutable protocol receipt; later changes to caller inputs or returned copies do not change that capture.

The plan has exactly `schema`, `result_schema`, `aggregate`, `holdouts` and `members`. Its schema is `wsp-staged-cohort-evaluation-plan/1` and its result schema is `wsp-staged-cohort-result/1`. `aggregate` has exactly `source_id` and `family_id`. Each entry in the nonempty ordered `holdouts` list has exactly `holdout_id`, `source_id`, `family_id` and a strict positive-integer `row_count`; booleans are rejected. Holdout IDs are unique, and source IDs are unique across the aggregate and holdouts. Family IDs may be shared. Identities must be nonblank UTF-8 strings and retain their exact spelling, including surrounding whitespace.

`members` must match the complete original roster exactly and in order. Each member has exactly `member_id`, a declared `holdout_id`, and a nonempty ordered `slots` list. A slot has exactly `slot_id`, `source`, `execution` and `result_keys`. Slot IDs are unique within that member; result keys form a nonempty list of unique identities. A source is either `{"kind": "selected", "fallback_control_id": null}` (or a same-member parent-control ID in place of `null`) or `{"kind": "control", "control_id": "..."}` naming a same-member control. These are declarations of sources and fallback policy, not executed fallbacks.

With a plan, `controls` must map exactly the original member IDs to control-definition objects. Each named definition has one of these exact envelopes:

| Kind | Definition fields | Payload schema labels |
| --- | --- | --- |
| `parent` | `kind` | No payload. |
| `reference` | `kind`, `artifact` | `wsp-network/1` or `wsp-tensors/1` |
| `structured` | `kind`, `proposal`, `lowering` | `wsp-structured-program/1` and `wsp-lowering/1`, respectively |

At declaration time, payloads are checked as finite JSON objects with those schema labels only. Full artifact/proposal/lowering parsing, endpoint construction and control materialization are deferred to the explicit endpoint seal below. A member may have an empty control map when no slot or fallback refers to a control. Controls and holdouts may remain unused, and multiple members may share a holdout. No selected-source slot is required. Distinct slots may repeat the same selected or control source under different execution declarations.

For a one-member roster `member_ids=("case-001",)`, a compact declaration is:

```python
execution = {
    "backend": "numpy", "dtype": "float64", "device": "cpu",
    "accumulation": "matmul", "endpoint_cast": "reject",
    "error_model": "empirical",
}
controls = {"case-001": {"parent": {"kind": "parent"}}}
evaluation_plan = {
    "schema": "wsp-staged-cohort-evaluation-plan/1",
    "result_schema": "wsp-staged-cohort-result/1",
    "aggregate": {"source_id": "aggregate", "family_id": "repair-study"},
    "holdouts": [{"holdout_id": "heldout", "source_id": "heldout-source",
                  "family_id": "repair-study", "row_count": 20}],
    "members": [{"member_id": "case-001", "holdout_id": "heldout", "slots": [{
        "slot_id": "selected",
        "source": {"kind": "selected", "fallback_control_id": "parent"},
        "execution": execution,
        "result_keys": ["loss"],
    }]}],
}
# Pass protocol={"evaluation_plan": evaluation_plan} and controls to StagedCohort.
```

Each `execution` contains exactly the six string fields shown above and must pass the existing `ExecutionSpec` syntax checks. Acceptance of that declaration does not establish backend/device availability, numerical support or successful execution. This declaration alone enables no cohort scoring or evaluation. The explicit endpoint seal and separate registered bridge below are required beyond declaration validation.

### Complete repair barrier

After all original members have terminal repair dispositions, `select_repairs` writes and verifies `repair-barrier.json` before returning its existing ordered disposition list. Changing that returned list does not change the retained barrier. The private barrier uses schema `wsp-staged-cohort-repair-barrier/1`. It binds the cohort protocol, roster, attempt, registry and evaluator identities to the complete diagnosis barrier. Each ordered member entry links its preparation receipt, diagnosis disposition, diagnosis receipt and outcome, repair disposition, and terminal lane repair receipt and outcome. Members unavailable for repair remain present with no fabricated lane receipt.

Before persisting the barrier, and again during its readback verification, the controller verifies the full diagnosis barrier, member dispositions and terminal lane evidence. For a selected repair it checks the actual selected `Candidate`, its parent/task/session identities, and the live artifact's hash and content against the recorded candidate. A cached candidate record alone cannot establish that the live artifact is unchanged. For other terminal repair outcomes, no selected candidate may remain.

Changed or missing evidence, a substituted live artifact, persistence failure or cancellation permanently invalidates the cohort attempt. Restoring the original evidence does not reopen it; finalization is not retried. Cancellation preserves the original exception even if writing its disposition also fails. This barrier alone enables no scoring or evaluation. The separate endpoint seal and registered bridge below require it; Issue #9 c1 remains partial with the issue open.

### Endpoint seal

With an opt-in evaluation plan and a complete durable repair barrier, the trusted harness may call the one-shot `seal_endpoints()` phase:

```python
endpoint_seal = cohort.seal_endpoints()
```

The call materializes only referenced member/source origins, writes `endpoint-seal.json`, and verifies it against the retained objects and complete repair evidence before returning a copied receipt. Schema `wsp-staged-cohort-endpoint-seal/1` binds the protocol, roster, evaluation plan, controls, attempt, registry, evaluator, diagnosis barrier and repair barrier. Changed configuration or evidence, endpoint construction or persistence failure, and cancellation invalidate the attempt; the phase cannot be retried.

The receipt distinguishes `origins` from ordered `members[*].slots`. An origin is identified by member and either `selected` or a control ID and is materialized once. Multiple logical slots can resolve to that origin while retaining their own slot ID, declared source, resolution, execution declaration and result keys. Distinct origins remain distinct even when their endpoint hashes coincide. Each materialized origin records its full artifact and hash. For a selected origin, the controller retains and checks the actual selected `Candidate` identity, and the receipt adds its candidate hash. Slot execution declarations are retained without execution or scoring.

A selected origin uses the actual selected candidate artifact. The seal checks the broker's saved proposal, literal program and lowering-receipt bindings and reproduces the endpoint by replaying the saved literal program. Its construction verification is `saved_literal_replay_only; structured_lowering_is_broker_provenance`: this does not independently reproduce the selected proposal's structured lowering. A referenced structured control instead parses its proposal, reruns `lower_structured` once with its declared native map and cast policy, requires equality with the complete supplied lowering receipt, and applies the resulting literal program. Its verification is `complete_lowering_receipt_reproduced_once`. A parent control uses the member's parent artifact; a reference control parses its supplied artifact. Every materialized endpoint must be compatible with that parent.

An unselected repair uses a declared same-member parent fallback when one exists; otherwise its selected origin and slots remain `unavailable`. A member without a prepared parent remains unavailable even with a fallback or control declaration. A selected no-op remains selected and does not trigger fallback. Referenced no-op controls remain valid controls, and repeated endpoint hashes do not erase logical slots. `unused_controls` lists unreferenced control IDs: their payloads remain frozen declarations and receive no full parsing or materialization during sealing.

For retained Torch capture metadata, the seal validates saved module/state declarations, shapes, supported dtype labels, devices and alias groups, including equality of declared aliased tensor data. It records `captured_native_dtypes`, `captured_devices` and `captured_alias_groups` separately from the construction native map. `construction_capture_agreement` is `absent`, `not_applicable`, `agree` or `mismatch`; a map mismatch is recorded rather than treated as runtime restoration or silently reconciled. These are static capture declarations, not restored modules, verified device availability or executed native semantics.

The separate [endpoint-seal engineering receipt](../../provenance/verification/staged-cohort-endpoint-seal-20261109.json) records one passing focused regression with no retry or timeout at its captured source snapshot. Earlier engineering receipts cover only their own captured source snapshots. The endpoint seal itself enables no cohort scoring or evaluation: its `evaluation_available`, `runtime_restoration_verified` and `causal_usefulness_established` flags remain `false`. The separate registered bridge below evaluates sealed slots without establishing runtime restoration or causal usefulness. Issue #9 c1 remains partial and the issue remains open; historical results are unchanged.

### Registered cohort evaluation

After `seal_endpoints()`, the trusted harness may call the one-shot cohort bridge:

```python
evaluation_receipt = cohort.evaluate_registered(load_holdouts, evaluate_slot)
```

`load_holdouts()` is a lazy trusted callback returning a dictionary from exactly every declared holdout ID to an exact `RegisteredBatch` handle from the cohort's registry. The bridge writes and verifies `evaluation-start.json` before invoking that loader once. It checks each handle's registry, source/family declaration, row count, content hash and ordered row hashes. It concatenates every declared holdout in plan order, including holdouts unused by any member, and registers one aggregate with every child batch as a parent. A loader or validation failure stops before scoring; otherwise the bridge requests `registry.score` once under the frozen attempt ID, binding the protocol, roster, plan, evaluator, endpoint seal, start receipt and child/aggregate manifests.

The registry's existing authorization consumes the aggregate's declared scope before entering its one scoring callback. Inside that callback, the bridge verifies the exact authorized aggregate and child row slices, then visits every member and logical slot in frozen order. Each available slot reached calls `evaluate_slot(context, endpoint, rows)` once. `endpoint` is the retained sealed artifact object; evaluation does not reconstruct, relower or reapply its program. `context` is a JSON copy of the member record plus the sealed slot declaration, including its execution specification; `rows` is a fresh JSON copy of that member's declared holdout slice. Repeated slots remain separate evaluator invocations even when they share an origin or endpoint hash. The trusted evaluator is responsible for honoring the declared execution specification; the bridge does not supply a model adapter, restore a runtime module or establish backend/device support.

Each evaluator result must be a finite JSON object with exactly the declared result keys. The bridge rechecks the start receipt, endpoint seal and prior slot receipts before and after slot work. It writes ordered private `evaluation-MEMBER-SLOT.json` dispositions using zero-padded numeric indices. Unavailable endpoints receive `outcome: unavailable`, zero evaluator invocations and no fabricated result. Available slots begin as `not_run`, become `succeeded` only after result validation and evidence verification, and become `failed` if their active work fails. A failed active slot can have zero or one evaluator invocation, depending on the failure phase. A failure stops subsequent slots; those slots retain `not_run` or their original `unavailable` state. A failed slot receipt write can leave its receipt hash null, and failed terminal persistence can leave no `evaluation.json`.

When persistence succeeds, terminal schema `wsp-staged-cohort-evaluation/1` retains the full result roster, holdout/aggregate manifests, slot receipt hashes, phase and exception type, registry decision and inspected attempt. Its outcome is `succeeded`, `failed` or `cancelled`. `score_requested` records that the registry boundary was attempted, while `score_callback_entered` records entry into the bridge's aggregate scorer; neither replaces the registry's authorization record. Loader failure or authorization rejection does not fabricate a registry attempt. Failed inspection records its exception type while the attempt remains unknown; an observed `authorized_outcome_unknown` status and null callback count remain unresolved rather than becoming zero or a successful result. Once authorization commits, failure or cancellation does not release its consumed scope.

Evidence mutation, loader/evaluator failure, invalid output, persistence failure and cancellation close the attempt without retry; a successful call also cannot repeat. The original loader or evaluator exception is preserved even if registry wrapping, metadata inspection or cleanup also fails. This cooperative bridge supplies no sandbox, crash recovery, authenticated provenance or scientific result. `runtime_restoration_verified` and `causal_usefulness_established` remain `false`. The separate [bridge engineering receipt](../../provenance/verification/staged-cohort-evaluation-bridge-20261109.json) records one focused regression: exit 0, `1 passed in 18.47s`, with no retry or timeout at its captured source snapshot. Earlier receipts retain their own source-snapshot scopes, Issue #9 c1 remains partial and the issue remains open.

## Receipt and rejection behavior

A run writes private `protocol.json`, `diagnosis.json`, `repair.json` and, following the registered evaluation path, `evaluation.json`. Selection receipts contain configuration, candidate and trace details and should remain under the trusted harness's control. They are separate from the sanitized [public guard receipt](../../provenance/verification/staged-diagnosis-guard-20261109.json).

Selection receipts add a canonical `outcome` field:

| Outcome | Meaning |
| --- | --- |
| `selected` | An admitted, nonabstaining candidate was sealed and recorded. |
| `abstained` | A valid explicit diagnosis abstention or `None` repair was finalized. |
| `invalid_submission` | The selector returned, but its submission failed schema validation or broker admission, or could not be finalized as selected. |
| `selector_failed` | The selector callback raised before returning a submission. |

The private receipt schema remains `wsp-staged-selection/1`. External consumers that require an exact set of keys must accept the additive `outcome` key and handle all four values before consuming new receipts. New receipt hashes include that field; existing saved receipts and their hashes remain unchanged.

When a broker exists and sealing, persistence and verification succeed, a callback or submission failure leaves a terminal failure receipt. If the failure raised an exception, the selection call still propagates that exact original exception, including `RuntimeError`, `KeyboardInterrupt` or `SystemExit`; cleanup does not replace it with a receipt or a cleanup error. The coordinator adds no exception-message copy to its terminal outcome fields; existing private session and broker traces retain their own diagnostic fields. A returned submission that fails schema validation is classified separately from a callback failure; a broker admission denial cannot become a successful selection.

If the broker denies a requested seal because its request cap is exhausted, trusted harness abstention closes the stage without consuming another proposer request. A seal or receipt-write error invalidates the run; finalization is not retried and an existing receipt is never overwritten. Repair setup can fail before a repair broker exists. That failure invalidates the run and propagates the original exception without claiming a terminal repair receipt.

Repair construction requires the exact persisted and in-memory diagnosis with `outcome: selected`; evaluation requires both exact selected stages. An observed receipt replacement permanently invalidates the run; restoring the original bytes does not reopen it. Premature and duplicate selections are rejected. Retained query closures are denied after sealing or invalidation without altering terminal traces. The exclusive directory prevents opening a second coordinator for the same run; this API does not provide run resumption.

The implementation reuses `ProposerBroker`, `ProgrammerSession`, `TaskSpec`, structured lowering and the durable data-role registry. Its terminal snapshots inspect existing private broker/session state because there is no public harness-only selection snapshot. Changes to those internals must preserve this boundary.

## Completed engineering evidence

The [source module](../../rlcd/weight_space/staged_diagnosis.py) contains one embedded assert guard. On November 9, 2026 there were two process launch attempts against module SHA-256 `7a035773d13496f18600d5a55f5165e521231c04d8a9706451749bf7d3cb531c`:

| Attempt | Recorded result | Process duration | Evaluator callbacks |
| --- | --- | --- | --- |
| Default Python 3.12 | Missing NumPy during package import; no guard executed | 0.2908897 seconds | 0 |
| Existing project virtual environment, one explicitly authorized correction | Embedded guard passed | 1.0615844 seconds | 1 literal synthetic callback |

The corrected invocation used `-B`, source-bound `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONNOUSERSITE=1`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. No guard was rerun to prepare this guide or its public receipt. The receipt preserves both launch outcomes and source identities while omitting workstation paths, raw logs and registry-row contents. Process durations are execution records, not performance benchmarks.

The guard exercised strict diagnosis validation, diagnosis-before-repair ordering, exact binding, copied-diagnosis mutation, premature and duplicate rejection, post-seal queries, replacement and restoration rejection, explicit abstention, and one durably recorded synthetic evaluator authorization. It used literal tensor values and one temporary-registry synthetic row; the callback checked a literal repaired endpoint. No training, provider, model study, scientific scoring or held-out study ran. All recorded source and frozen-file hashes matched before and after both invocations. The original startup-failure evidence remained unchanged around the corrected invocation.

Those two saved launch outcomes and source identities do not cover the later outcome classification and terminal failure receipt changes described above.

A separate [terminal failure engineering receipt](../../provenance/verification/staged-terminal-failures-20261109.json) records invocation evidence and test-source identities for these November 9, 2026 attempts:

| Test version | UTC start to finish | Recorded result |
| --- | --- | --- |
| Initial fixture | 15:42:14.3240073Z to 15:42:18.6940761Z | Exit 1; `1 failed in 2.84s`, `KeyError: 'repair'` |
| Corrected fixture | 15:45:55.2504050Z to 15:45:59.5488061Z | Exit 0; `1 passed in 3.08s`; no timeout |

The initial fixture looked up the lazy repair broker before its construction. A test-only correction uses standard-library `ExitStack` to attach instrumentation inside the callback; production source and base commit stayed the same. The corrected invocation's source inventory matched before and after execution, and independent source review found no findings. Each test version was invoked exactly once; the failed invocation was not retried.

The initial failing test's raw bytes were not saved; the untracked test was absent from the retained tracked diff. Its recorded test-source hash, invocation logs and import inventory are retained. The corrected raw test source was saved.

The synthetic check exercises terminal outcome classification and failure receipts, exact original `RuntimeError`/`KeyboardInterrupt`/`SystemExit` propagation, trusted abstention after capped seal denial, and invalidation without overwrite or retry for seal, persistence and pre-broker repair setup failures. No provider, training or scientific study ran; this focused check was not a completed-tree gate. These are engineering behaviors, not evidence of useful causal repair; Issue #9 c1 remains partial and the issue remains open.

A separate [cohort selection engineering receipt](../../provenance/verification/staged-cohort-selection-20261109.json) records one fresh regression invocation on November 9, 2026, from `16:12:27.1106180Z` to `16:12:31.8024243Z`: exit 0, `1 passed in 3.52s`, with no retry. The production/test source audit passed. The exact 37-source inventories and recorded diff/state were unchanged before and after the invocation, and the raw affected source and test snapshots were saved. This is a focused selection regression at its recorded source snapshot, not a completed-tree gate or evidence of useful causal repair. It supplies no cohort scoring bridge or new study evidence.

A separate [repair-barrier engineering receipt](../../provenance/verification/staged-cohort-repair-barrier-20261109.json) records one invocation of [`test_complete_repair_barrier_and_fail_stop`](../../tests/test_staged_cohort_repair_barrier.py) on November 9, 2026, from `16:31:53.8747550Z` to `16:32:00.7294848Z`: `1 passed in 5.29s`, with no retry. Independent production/test source review found no findings. The captures at `16:31:48.6786406Z` and `16:32:08.5980405Z` recorded an identical 37-source inventory, SHA-256 `ab556530f586292a1b11e2bed7af1671659fc301d3d9d85e4154eeafedb6b123`, and identical tracked diff and state apart from capture time/phase. Raw affected source and test snapshots were saved privately.

That literal regression covers the complete ordered repair barrier, selected/unavailable/failed/abstained/invalid dispositions, exact linked evidence, and live artifact validation. It also checks permanent invalidation after changed evidence or a substituted live artifact, barrier persistence failure, and cancellation with a secondary disposition-write failure. No provider, study, evaluation data or scoring authorization was used. This focused regression covers its recorded source snapshot; it is not a completed-tree gate or a scientific result.

A separate [evaluation-plan engineering receipt](../../provenance/verification/staged-cohort-evaluation-plan-20261109.json) records one invocation of [`test_frozen_evaluation_plan_and_legacy_compatibility`](../../tests/test_staged_cohort_evaluation_plan.py) on November 9, 2026, from `16:49:52.1333908Z` to `16:49:54.1348456Z`: exit 0, `1 passed in 0.79s`, with no retry or timeout. Independent implementation/test source review found no material findings. The captures at `16:49:47.7425143Z` and `16:49:59.2180664Z` recorded an identical 37-source inventory, SHA-256 `a8333475f09174ace15e4a6731880689d575b4b1f0e34d879a9f9ce8b552cd08`, and identical tracked diff and state apart from capture time/phase. Raw affected source snapshots were retained privately.

This focused declaration regression checks immutable capture, exact field/reference validation, supported declaration variants and legacy finite-JSON controls. No preparation, provider calls, scoring data, scoring authorization or study execution occurred; the check establishes no runtime availability. The permitted case with no selected-source slots was not directly exercised. The receipt covers its recorded source snapshot, not a final-tree gate or a scientific result; Issue #9 c1 remains partial and the issue remains open.

A separate [endpoint-seal engineering receipt](../../provenance/verification/staged-cohort-endpoint-seal-20261109.json) records one focused regression invocation on November 9, 2026, from `17:18:35.4154661Z` to `17:18:41.1152709Z`: exit 0, `1 passed in 4.30s`, with no retry or timeout. The before/after source inventory and tracked diff were unchanged. This check covers its captured source snapshot; it is not a completed-tree gate or a scientific result. It supplies no cohort scoring or evaluation bridge and establishes neither runtime restoration nor causal usefulness; Issue #9 c1 remains partial and the issue remains open.

A separate [cohort evaluation-bridge engineering receipt](../../provenance/verification/staged-cohort-evaluation-bridge-20261109.json) records one focused regression invocation on November 9, 2026, from `2026-11-09T17:43:37.9968303Z` to `2026-11-09T17:43:57.5622134Z`: exit 0, `1 passed in 18.47s`, `timed_out=false`, `execution_count=1` and `automatic_retry_performed=false`. The tested and integrated clean source HEAD was `6c7da8e30927163493c6b2de6127c7140e6fd6d8`, based on `7c145e5631c2771b22109ebaa02d974c9d9e9068`. The before/after 65-file source inventory was unchanged at SHA-256 `a1039df6d862b81e7816123ddf8b9ad57ed90290ffa6c283edf67ebf77409995`; the unchanged tracked diff had SHA-256 `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b`, with a clean tree at both captures. This check covers its captured source snapshot, not a completed-tree gate or a scientific result. It supplies generic framework integration without a real held-out study, runtime-restoration verification or causal-usefulness evidence. Earlier receipt scopes and historical results remain unchanged; Issue #9 c1 remains partial and the issue remains open.

## Limits and scientific next step

This is a cooperative trusted coordinator. It enforces sequencing at its JSON interface and checks exact recorded evidence. It does not sandbox Python callbacks or attest a proposer's private reasoning, prior knowledge, provenance, independent decision-making or storage hardware. The caller must control the receipt directory and shared registry. File flush and `fsync` are persistence steps; the literal guard is not a crash-recovery or adversarial-filesystem study.

The older public guard receipt covers its recorded source snapshot and guard execution; the terminal failure, cohort selection and repair-barrier receipts each cover their separately recorded snapshots. None verifies later source changes or integrated revisions. Their private-receipt hashes are commitments to retained evidence, not publicly reproducible raw evidence.

A future preregistered study must use separate prospective diagnosis and repair selections, an appropriate evaluator and fresh authorized data to assess whether the frozen diagnosis predicts a useful intervention. That study must preserve the historical joint-selection result and report uncertainty and abstentions. Until such evidence satisfies the acceptance criterion, #9 c1 remains partial.
