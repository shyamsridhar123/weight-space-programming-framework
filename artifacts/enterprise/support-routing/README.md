# Synthetic enterprise support-routing records

These snapshots accompany the [enterprise walkthrough](../../../docs/use-cases/enterprise-support-routing.md)
and [runnable implementation](../../../examples/enterprise_support_routing.py).
They use invented business records and a hand-constructed linear head. They are
integration evidence, not customer data or enterprise deployment evidence.

The [local validation record](validation.json) names the checked behavior and
environment. It distinguishes integration checks from formal, independent,
provider, and deployment claims.

| Scenario | Business report | Complete bundle | Decision |
| --- | --- | --- | --- |
| Policy change | [Read report](policy-change/report.md) | [Inspect bundle](policy-change/bundle.json) | Admit for the synthetic trial |
| Conflicting protection | [Read report](conflict/report.md) | [Inspect bundle](conflict/bundle.json) | Abstain before final evaluation |
| Symptom coverage gap | [Read report](coverage-gap/report.md) | [Inspect bundle](coverage-gap/bundle.json) | Reject after final evaluation |

For each scenario, `checksums.json` binds the generated files. Selected cases
also export the candidate artifact, exact-parent edit, and inverse program.
An exported rejected candidate remains rejected; files are review artifacts,
not a deployment action.

Verify a snapshot from the repository root after installing the project:

```bash
python examples/enterprise_support_routing.py --verify artifacts/enterprise/support-routing/policy-change
```

Generate a separate local run with `python examples/enterprise_support_routing.py`.
It writes to `.runs/` by default. To rebuild snapshots intentionally, pass
`--output artifacts/enterprise/support-routing`, then update the current
publication manifest. Exact verification expects the recorded source, fixture,
schema, and lowering environment.

The [workflow PNG](routing-workflow.png) and [score tradeoff PNG](score-tradeoff.png)
have editable SVG counterparts. Their [source record](visual-source.json)
binds the measured bundle and names the metric fields. Re-render with:

```bash
python -m pip install -e '.[plots]'
python scripts/render_enterprise_routing.py
```

Project-authored code, records, and diagrams use the repository's MIT license.
The [framework proof guide](../../../PROOFS.md) defines the scope of the retained
mathematical work; these runs do not add candidate-specific Lean proofs.
