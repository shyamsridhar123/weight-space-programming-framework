# Review a weight-space change with Distro

A support owner wants telemetry outages routed to Platform Operations. The
fresh demo changes one readout weight so its supplied outage feature moves from
`ordinary-support` to `platform-operations`. All scores on the supplied billing,
security and ordinary-support feature directions stay unchanged.

This is a fixed-feature example. Raw-text recognition, unseen-ticket behavior
and customer savings remain unverified. The review interface accepts other
framework use cases through the same [change descriptor](distro-integration.md#one-change-file).

Start with the property and its assumptions. The demo generator applies the
exact-parent program, checks structural admission and verifies the supplied-feature
retention receipt. The governance tools bind those evidence bytes to the change;
they do not rerun proofs or behavioral checks. Constraints are committed data,
not automatically evaluated conditions.

## Create the demo and load the companion

From the repository root, using the Python environment you want the server to
run in:

```sh
python -m pip install '.[distro]'
python -m rlcd.weight_space distro demo --workspace ../wsp-distro-demo
```

Use a **new directory**: the generator refuses an existing workspace. It writes
`change.json`, the exact artifacts under `change/`, review evidence under
`evidence/`, and a companion plugin under `.runs/distro-plugin`. It creates no
approval. The plugin contains the governance skill and native Claude/Copilot
manifests, bound to this interpreter and resolved workspace. Keep this generated
configuration local.

With Distro and your chosen coding host installed, change to the new workspace
and launch either host:

```sh
cd ../wsp-distro-demo
distro claude --plugin-dir .runs/distro-plugin
```

The Copilot alternative is:

```sh
distro copilot --plugin-dir .runs/distro-plugin
```

These are the generated launch paths. The native checks below establish the
observed host behavior; live Distro/model sessions and enterprise identity
enforcement still need their own validation.

## Native host validation

On November 10, 2026, the checks used an installed wheel in an isolated Python
environment and a freshly generated companion. The native management commands
ran directly, without model prompts or changes to global host configuration.
In the commands below, `<plugin>` is the generated `.runs/distro-plugin` path.

| Native host | Command | Observed result |
| --- | --- | --- |
| Claude Code 2.1.317 | `claude plugin validate <plugin> --json` | Exited successfully with `success: true`. |
| Claude Code 2.1.317 | `claude --plugin-dir <plugin> mcp get plugin:weight-space-governance:weight-space` | Exited successfully and reported `Connected`. |
| GitHub Copilot CLI 1.0.99-2 | `copilot --plugin-dir <plugin> plugin list --json` | Discovered `weight-space-governance` as an enabled external plugin. |
| GitHub Copilot CLI 1.0.99-2 | `copilot --plugin-dir <plugin> mcp list --json` | Exited successfully but omitted the framework server. |

The Copilot management result is limited by the installed version. Its
[official 1.0.109-0 release notes](https://github.com/github/copilot-cli/releases/tag/v1.0.109-0)
document a later fix making `mcp list` and `mcp get` honor `--plugin-dir` and
`--additional-mcp-config`. An omitted server on 1.0.99-2 therefore does not
establish whether a live session loads it. That later version was not installed
or tested here.

Claude's reported connection and Copilot's plugin discovery are narrower than
a live tool invocation. Native host tool calls, live Distro-launched model
sessions and enterprise identity enforcement remain untested. The independent
installed-wheel MCP smoke check covers the four framework operations and
invalid-pin rejection; it does not establish those host behaviors.

## Inspect, prepare, obtain a human decision, then export

Ask the host:

> Use the weight-space governance skill to inspect `change.json`. Explain the
> edit, the supplied-feature retention evidence and its limits. Prepare a review
> at `review/requested.json` and return both pins for me to retain independently.
> Leave the decision to the human reviewer.

The companion exposes exactly four tools. All arguments are strings; all paths
are relative to the configured workspace. Every output path must be new.

| Tool | Arguments for this example | What it does |
| --- | --- | --- |
| `wsp_inspect_change` | `change: "change.json"` | Shows the exact draft request, program and bounded evidence previews. |
| `wsp_prepare_review` | `change: "change.json"`, `out: "review/requested.json"` | Creates a requested review and returns `request_sha256` and initial `head_sha256`. |
| `wsp_review_status` | `change: "change.json"`, `bundle: "review/approved.json"`, `expected_request_sha256`, `expected_head_sha256` | Checks the complete current change and review history against both independently retained pins. |
| `wsp_export_review` | The status arguments plus `out: "review/audit-copy.json"` | Performs the same validation and copies the complete bundle bytes to a new file. |

Retain the preparation's two pins in a record you control **outside later
received bundles**. The original request pin remains fixed. After each
independent human decision, retain its new head alongside that original request
pin. Never fill an expected pin from the bundle being checked.

An independently authorized human reviews the exact artifacts and evidence.
For this synthetic example the requester label is `alice` and the designated
approver label is `bob`; labels do not authenticate either person. The human,
from the demo workspace, can run this command after substituting the two pins
from their independent record:

```sh
python -m rlcd.weight_space governance decide review/requested.json --expected-request-sha256 "<retained-request-pin>" --expected-head-sha256 "<retained-initial-head-pin>" --actor bob --action approve --reason "Reviewed the exact synthetic change and supplied-feature evidence for a local pilot" --out review/approved.json
```

Use a new output path and retain the returned new head. This command is for the
human reviewer. The agent must not execute approval, rejection or revocation
through a shell or another tool. The CLI also supports `reject` and `revoke`
under the [existing transition rules](runtime.md); each decision needs both
independent current pins and a new output path.

Return the approved bundle's location and the independently retained original
request pin and new head to the host. Ask it to check the status, then export to
`review/audit-copy.json`. Changed artifacts, evidence, scope, request fields or
a history inconsistent with either pin must fail validation. Requested,
rejected and revoked records can still be checked or exported successfully;
they remain ineligible.

An approved, correctly bound review can return `local_pilot_eligible: true`.
`production_authorized` and `scientific_authorized` remain `false`. The adapter
has no trusted latest-head store: a retained newer revocation head rejects an
older approval, while supplying an old pin does not establish freshness.
Authentication, current-head retention, concurrent decision serialization and
operational deployment/revocation remain external responsibilities.

For your own use case, prepare a [descriptor and evidence](distro-integration.md),
then run `python -m rlcd.weight_space distro setup --workspace "<existing-directory>"`.
Read the [enterprise handoff](distro-handoff.md) before extending this local
review to organizational systems.
