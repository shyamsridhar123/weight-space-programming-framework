# Enterprise support routing: policy-change

**Decision: admitted_for_synthetic_trial.** Production deployment is not authorized by this demo.

New policy: telemetry outage tickets should reach Platform Operations; existing routing scores and queues are protected.

The baseline and business records are synthetic. Scores are raw routing scores, not probabilities. A winner margin below 0.5 sends a ticket to manual review.

## Public candidate review

| Candidate | Target shift error (score units) | Protected score drift | Target routes correct | Protected routes unchanged | Exact anchor retention | Eligible |
| --- | ---: | ---: | --- | --- | --- | --- |
| keep-parent | 3.000000 | 0.000000 | False | True | True | False |
| target-only | 0.042784 | 1.541679 | True | False | False | False |
| soft-retention | 0.031531 | 0.011449 | True | True | False | False |
| protected-edit | 0.029703 | 0.000000 | True | True | True | True |

Acceptance requires target shift error <= 0.05, correct target queues with margin >= 0.5, exact zero residual on protection anchors, protected sampled score drift <= 1e-12, and unchanged correct protected queues. Structural admission and certificate binding must also pass.

- **keep-parent:** target_score_shift, target_routing.
- **target-only:** protected_runtime_scores, protected_routes, protected_exact_retention.
- **soft-retention:** protected_runtime_scores, protected_exact_retention.

## Sealed final evaluation

Selected: **protected-edit**.

Target shift error: **0.029703 score units**. Protected score drift: **0.000000**.

| Ticket | Cohort | Expected queue | Parent queue | Sealed queue | Sealed winner margin |
| --- | --- | --- | --- | --- | ---: |
| final-outage-enterprise | target | platform | general | platform | 1.220297 |
| final-outage-standard | target | platform | general | platform | 1.220297 |
| final-general-enterprise | protected | general | general | general | 1.937500 |
| final-core-outage | protected | general | general | general | 1.812500 |
| final-billing-telemetry | protected | billing | billing | billing | 2.625000 |
| final-security-telemetry | protected | security | security | security | 4.062500 |
| final-existing-platform | protected | platform | platform | platform | 2.812500 |
| final-telemetry-help | protected | general | general | general | 1.750000 |

Final gate failures: none.

## Evidence and operational boundary

The bundle binds the parent, contract, task, literal edits, compiler lowering, certificates, candidate review, sealed evaluation, and rollback. Budget usage describes session interfaces; external fitting and final evaluation are outside proposer charges. No external provider ran.

Exact receipts concern the real linear head on supplied features and their span under the stated premises. This run checks NumPy float64 scores on the recorded rows; it does not prove a production runtime or text extractor. Declared split identifiers are disjoint, but constructed public fixtures are not independent confirmation.

Rollback restores the exact model artifact. It does not reverse ticket assignments, external side effects, or other model revisions. A real release needs approved data, an independently owned untouched evaluation, feature and runtime correspondence, and an operational rollout decision.

Contract SHA-256: `0f18734f5131cf7fbe79677ef22b6a3e10dd622570211e2e1b5e5653d112ce23`. Final admission SHA-256: `a6558e1a41d0ba70da348a6e113cb68d22ec1d2ec0c78b99838dcd3c083da82e`.
