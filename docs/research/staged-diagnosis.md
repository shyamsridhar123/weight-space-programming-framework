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

## Receipt and rejection behavior

A run writes private `protocol.json`, `diagnosis.json`, `repair.json` and, following the registered evaluation path, `evaluation.json`. Selection receipts contain configuration, candidate and trace details and should remain under the trusted harness's control. They are separate from the sanitized [public guard receipt](../../provenance/verification/staged-diagnosis-guard-20261109.json).

Repair construction and evaluator intervention require the exact persisted and in-memory terminal selections. An observed receipt replacement permanently invalidates the run; restoring the original bytes does not reopen it. Premature and duplicate selections are rejected. Selector failures seal their brokers, and retained query closures are denied after sealing without altering terminal traces. The exclusive directory prevents opening a second coordinator for the same run; this API does not provide run resumption.

The implementation reuses `ProposerBroker`, `ProgrammerSession`, `TaskSpec`, structured lowering and the durable data-role registry. Its terminal snapshots inspect existing private broker/session state because there is no public harness-only selection snapshot. Changes to those internals must preserve this boundary.

## Completed engineering evidence

The [source module](../../rlcd/weight_space/staged_diagnosis.py) contains one embedded assert guard. On November 9, 2026 there were two process launch attempts against module SHA-256 `7a035773d13496f18600d5a55f5165e521231c04d8a9706451749bf7d3cb531c`:

| Attempt | Recorded result | Process duration | Evaluator callbacks |
| --- | --- | --- | --- |
| Default Python 3.12 | Missing NumPy during package import; no guard executed | 0.2908897 seconds | 0 |
| Existing project virtual environment, one explicitly authorized correction | Embedded guard passed | 1.0615844 seconds | 1 literal synthetic callback |

The corrected invocation used `-B`, source-bound `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONNOUSERSITE=1`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. No guard was rerun to prepare this guide or its public receipt. The receipt preserves both launch outcomes and source identities while omitting workstation paths, raw logs and registry-row contents. Process durations are execution records, not performance benchmarks.

The guard exercised strict diagnosis validation, diagnosis-before-repair ordering, exact binding, copied-diagnosis mutation, premature and duplicate rejection, post-seal queries, replacement and restoration rejection, explicit abstention, and one durably recorded synthetic evaluator authorization. It used literal tensor values and one temporary-registry synthetic row; the callback checked a literal repaired endpoint. No training, provider, model study, scientific scoring or held-out study ran. All recorded source and frozen-file hashes matched before and after both invocations. The original startup-failure evidence remained unchanged around the corrected invocation.

## Limits and scientific next step

This is a cooperative trusted coordinator. It enforces sequencing at its JSON interface and checks exact recorded evidence. It does not sandbox Python callbacks or attest a proposer's private reasoning, prior knowledge, provenance, independent decision-making or storage hardware. The caller must control the receipt directory and shared registry. File flush and `fsync` are persistence steps; the literal guard is not a crash-recovery or adversarial-filesystem study.

The public receipt covers the recorded source snapshot and guard execution. It does not verify later source changes or integrated revisions. Its private-receipt hashes are commitments to retained evidence, not publicly reproducible raw evidence.

A future preregistered study must use separate prospective diagnosis and repair selections, an appropriate evaluator and fresh authorized data to assess whether the frozen diagnosis predicts a useful intervention. That study must preserve the historical joint-selection result and report uncertainty and abstentions. Until such evidence satisfies the acceptance criterion, #9 c1 remains partial.
