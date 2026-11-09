# Trained optical-digit application: evidence artifacts

This directory contains the pilot and six independent main training lineages, followed by immutable per-case/per-arm candidate, admission, prediction, verification and rollback records. These are research artifacts for a declared frozen-feature head-edit task.

[Read the heldout report](../../../docs/research/trained-applications.md) for concrete use cases, deployed outcomes, denominators, adverse findings, uncertainty, costs and boundaries. [Dataset attribution and publication scope](../../../experiments/research/trained-nontext/DATASET-NOTICE.md) and [sanitized source identities](../../../experiments/research/trained-nontext/DATASET-PROVENANCE.json) accompany the artifacts.

[The frozen protocol, global seal, all-assigned ledger, summary and independent replay](../../../experiments/research/trained-nontext/) are the study index. Pilot outputs are excluded from main inference. Ledger assignments are not independent sample units; the main independent unit is a full from-scratch encoder/head training lineage.

Each main lineage records its encoder, preprocessing, feature and head identities in `lineage.json`. Its optional `evaluation-features.json` separately hashes the test/stress learned-feature bundle after sealing. Each `attempts/` subdirectory records one selected candidate for a case/arm, the corresponding `program.json` and `rollback.json`, admission trace, and evaluation artifacts. An external overlay/context intervention can have an unchanged tensor artifact while changing deployed codes; admission, tensor no-op and useful behavior are separate fields.

`outcome.json` distinguishes **deployed** primary/secondary/nearby/stress behavior from **candidate-only** diagnostics. Rejected attempts deploy the parent and remain in the useful-success denominator. `native_diagnostics` describes the selected candidate head even if rejected. Only methods with `retention.json` have an exact finite-anchor retention receipt; the receipt records nonzero residuals as nonzero values. Application and rollback identities are independently verified for every main assignment.

No raw acquired query/image records are included. No live provider/editor ability, whole-domain raw-input/native guarantee or enterprise savings follows from these artifacts. Use a disposable checkout for archived replay commands: replay regenerates its verification record.
