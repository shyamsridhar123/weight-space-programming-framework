# Contributing to the draft

Start a contribution with the concrete problem: which input, request, or claim is currently unsupported or wrong? Explain how the change addresses it, then validate the resulting behavior at the boundary it affects. A tensor edit, a retention guarantee, and a useful task result need different evidence.

Read the [requirements](docs/weight-space/v2/requirements.md), [architecture](docs/weight-space/v2/architecture.md), and [limitations](docs/weight-space/v2/limitations-and-next-steps.md) before extending a claim. The [worked example](docs/weight-space/v2/worked-example.md) shows the distinction on a small edit. Application-specific behavior belongs in explicit adapters and task evaluators.

## Development setup

Use Python 3.11 or newer. From the repository root:

```bash
python -m pip install -e ".[test,plots]"
python examples/edit_retention.py
python examples/session_edit.py
python -m pytest -q --basetemp .runs/pytest-contributor-1
```

Choose an unused `--basetemp` directory inside this checkout each time. Freeze tests require repository-contained fixtures, and pytest clears the supplied directory before use. Run focused tests while changing a boundary, then complete the relevant suite and report failures or skips. The examples are constructed API demonstrations, separate from the retained study evidence.

[REPRODUCING.md](REPRODUCING.md) covers installation variants, independent replay, report/diagram rendering, and optional historical TypeScript checks. [PROOFS.md](PROOFS.md) covers the pinned formal toolchain. The [interactive explanation](artifacts/explanations/edit-retention/index.html) opens locally without a build or network access; its [static guide](docs/weight-space/v2/worked-example.md) remains available for review.

## Preserve evidence and scope

Closed studies, freezes, proof receipts, reports, and verification directories are historical evidence. Put new work in a fresh destination and retain incomplete attempts and failures. Preserve original bytes and source identities when revising an explanatory guide.

Use the obligation that matches the change:

| Contribution | Mechanism to make explicit | Validation needed for its claim |
| --- | --- | --- |
| Edit operation or numerical compiler | Parent binding, ordered operations, dtype conversion, and realized endpoint. | Reject invalid parents and inputs; check stored changes and rounding. Measure target behavior and retention separately when claiming benefit. |
| Architecture or adapter | Tensor shapes, feature/output order, state, and execution equation. | Test the declared semantics. A certificate needs a matching proposition, domain, and endpoint binding in addition to an evaluator. |
| Proposer or workflow | Supplied information, allowed access, charged work, candidate history, and selection rule. | Check failures and sealing order. Strong access restrictions need an enforceable boundary beyond the cooperative trace. |
| Scientific result | Baseline, metric and units, independent units, selection, uncertainty, and failure policy. | Freeze before observing new outcomes; retain all attempted units and relevant negative findings. |
| Explanatory guide or visualization | The question, mechanism, evidence source, and limit of each claim. | Check equations and examples against the implementation, verify links, and inspect rendered output. |

Keep structural validity, real-valued bounds, native execution, and measured utility as separate conclusions. Retain the exact parent/candidate identities, domains, adapters, witnesses, and protocol needed to assess each one. A small edit or a successful reconstruction alone does not establish retained behavior.

Published seeds and archived holdouts are observed. Fresh confirmation requires untouched units and a protocol fixed before their outcomes. Update requirement and roadmap status only when its stated acceptance obligation is met; keep contrary results and unresolved limits beside the supported conclusion.

## Change record and review

Write the review description for a reader who has not seen the discussion. State the triggering problem, the resulting behavior, why the mechanism addresses it, how it was checked, and any remaining limit. Add that result and validation under `Unreleased` in [CHANGELOG.md](CHANGELOG.md). Report observations and their denominators; file, test, and theorem totals are not measures of research quality.

A source change creates a new lineage. Preserve the earlier records and identify the new source freeze and publication manifest. In particular, the revised source-review differs from its historical study input: saved-data replay reports that live difference as informational, while strict execution against the old freeze rejects it. Create a fresh freeze for fresh execution, as shown in [REPRODUCING.md](REPRODUCING.md).

Proof work must state the changed proposition or premise and use the explicit `WeightSpace.GenericAudit` target and combined axiom audit. The portable wrapper checks the preserved proof snapshot. Its narrow mapping resolves the historical `docs/weight-space/v2/formal.md` and `formal/weight-space/README.md` entries to archived originals while retaining their original hashes; every other input keeps its recorded path and digest. See the [proof receipt mapping](PROOFS.md#mapping-the-saved-receipts-into-this-repository). A deliberate proof-source change requires a new provenance record and review, rather than edits to historical receipts. Retain new proof logs and receipts in a fresh `.proof-runs/` directory. Checking archive mappings or editing prose does not constitute a new Lean run.

For a flowchart change, edit [scripts/render_flowcharts.py](scripts/render_flowcharts.py), regenerate PNG/SVG files, and inspect each output. Keep the graph manifest, alt text, and explanatory prose aligned. Review changes to the standalone HTML explanation directly in a browser, including its initial state and interactive controls, and keep the static guide consistent with the same mechanism.

The [publication verifier](scripts/verify_publication.py) rejects a changed file against the prior checksums. For a new release, review the complete diff, preserve archived originals, and generate new checksums and a new publication record. Retain mismatches until the intentional change is explained; keep the verifier's preservation obligations intact.

## Rights and attribution

Review [RIGHTS.md](RIGHTS.md) before importing external code, data, or prose. Record its source and applicable notices. Keep third-party licenses and ownership distinct from this project's terms, and describe attribution without implying upstream endorsement.
