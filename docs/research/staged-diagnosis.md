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

Each phase is available once. An ordinary selector exception permits continuation only when its terminal receipt verifies; missing or changed evidence, persistence failure and cancellation invalidate the cohort attempt. Cohort-owned lanes reject direct public diagnosis/repair selection, so an individual lane cannot bypass the full-cohort barrier. Their public evaluation method is also denied: this API has no cohort scoring bridge and adds no study evidence. It is a trusted cooperative controller, not a Python sandbox, crash-recovery mechanism or independent execution attestation.

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

Payloads are checked as finite JSON objects with those schema labels only. Full artifact/proposal/lowering parsing, endpoint construction and control materialization are deferred. A member may have an empty control map when no slot or fallback refers to a control. Controls and holdouts may remain unused, and multiple members may share a holdout. No selected-source slot is required. Distinct slots may repeat the same selected or control source under different execution declarations.

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

Each `execution` contains exactly the six string fields shown above and must pass the existing `ExecutionSpec` syntax checks. Acceptance of that declaration does not establish backend/device availability, numerical support or successful execution. This opt-in contract enables no cohort scoring or evaluation; the endpoint/control seal and single-score bridge remain pending.

### Complete repair barrier

After all original members have terminal repair dispositions, `select_repairs` writes and verifies `repair-barrier.json` before returning its existing ordered disposition list. Changing that returned list does not change the retained barrier. The private barrier uses schema `wsp-staged-cohort-repair-barrier/1`. It binds the cohort protocol, roster, attempt, registry and evaluator identities to the complete diagnosis barrier. Each ordered member entry links its preparation receipt, diagnosis disposition, diagnosis receipt and outcome, repair disposition, and terminal lane repair receipt and outcome. Members unavailable for repair remain present with no fabricated lane receipt.

Before persisting the barrier, and again during its readback verification, the controller verifies the full diagnosis barrier, member dispositions and terminal lane evidence. For a selected repair it checks the actual selected `Candidate`, its parent/task/session identities, and the live artifact's hash and content against the recorded candidate. A cached candidate record alone cannot establish that the live artifact is unchanged. For other terminal repair outcomes, no selected candidate may remain.

Changed or missing evidence, a substituted live artifact, persistence failure or cancellation permanently invalidates the cohort attempt. Restoring the original evidence does not reopen it; finalization is not retried. Cancellation preserves the original exception even if writing its disposition also fails. This barrier enables no scoring or evaluation. The separate endpoint/control seal and single-score bridge remain pending, and Issue #9 c1 remains partial with the issue open.

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

## Limits and scientific next step

This is a cooperative trusted coordinator. It enforces sequencing at its JSON interface and checks exact recorded evidence. It does not sandbox Python callbacks or attest a proposer's private reasoning, prior knowledge, provenance, independent decision-making or storage hardware. The caller must control the receipt directory and shared registry. File flush and `fsync` are persistence steps; the literal guard is not a crash-recovery or adversarial-filesystem study.

The older public guard receipt covers its recorded source snapshot and guard execution; the terminal failure, cohort selection and repair-barrier receipts each cover their separately recorded snapshots. None verifies later source changes or integrated revisions. Their private-receipt hashes are commitments to retained evidence, not publicly reproducible raw evidence.

A future preregistered study must use separate prospective diagnosis and repair selections, an appropriate evaluator and fresh authorized data to assess whether the frozen diagnosis predicts a useful intervention. That study must preserve the historical joint-selection result and report uncertainty and abstentions. Until such evidence satisfies the acceptance criterion, #9 c1 remains partial.
