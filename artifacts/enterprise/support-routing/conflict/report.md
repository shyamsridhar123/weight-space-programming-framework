# Enterprise support routing: conflict

**Decision: abstained.** Production deployment is not authorized by this demo.

New policy: telemetry outage tickets should reach Platform Operations; existing routing scores and queues are protected.

The baseline and business records are synthetic. Scores are raw routing scores, not probabilities. A winner margin below 0.5 sends a ticket to manual review.

## Public candidate review

| Candidate | Target shift error (score units) | Protected score drift | Target routes correct | Protected routes unchanged | Exact anchor retention | Eligible |
| --- | ---: | ---: | --- | --- | --- | --- |
| keep-parent | 3.000000 | 0.000000 | False | True | True | False |
| target-only | 0.042784 | 1.541679 | True | False | False | False |
| soft-retention | 1.374431 | 0.189195 | False | True | False | False |
| protected-edit | 3.000000 | 0.000000 | False | True | True | False |

Acceptance requires target shift error <= 0.05, correct target queues with margin >= 0.5, exact zero residual on protection anchors, protected sampled score drift <= 1e-12, and unchanged correct protected queues. Structural admission and certificate binding must also pass.

- **keep-parent:** target_score_shift, target_routing.
- **target-only:** protected_runtime_scores, protected_routes, protected_exact_retention.
- **soft-retention:** target_score_shift, target_routing, protected_runtime_scores, protected_exact_retention.
- **protected-edit:** target_score_shift, target_routing.

## Sealed final evaluation

Selected: **None**.

No candidate met the public contract. The session abstained and did not call the final evaluator.

The protected features now span every editable feature direction. Exact preservation forces a zero update; the requested score shift cannot be admitted.

## Evidence and operational boundary

The bundle binds the parent, contract, task, literal edits, compiler lowering, certificates, candidate review, sealed evaluation, and rollback. Budget usage describes session interfaces; external fitting and final evaluation are outside proposer charges. No external provider ran.

Exact receipts concern the real linear head on supplied features and their span under the stated premises. This run checks NumPy float64 scores on the recorded rows; it does not prove a production runtime or text extractor. Declared split identifiers are disjoint, but constructed public fixtures are not independent confirmation.

Rollback restores the exact model artifact. It does not reverse ticket assignments, external side effects, or other model revisions. A real release needs approved data, an independently owned untouched evaluation, feature and runtime correspondence, and an operational rollout decision.

Contract SHA-256: `5fc8fe1966226ad5585a02f6bb0b3e722516546f48104b2441e22cd681db539f`. Final admission SHA-256: `8dee9278d512cab3aaa3667cc5f428aead32c59fd86ad3fc2b5fac97fb8969fc`.
