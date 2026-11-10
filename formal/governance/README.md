# Abstract local-pilot governance model

[Governance.lean](Governance.lean) states the governance obligations fixed in
[the contract](../../docs/governance/contract.md) before runtime implementation.
It is a standalone Lean source importing only `Std`, `Lean.Elab.Command` and
`Lean.Util.CollectAxioms`, all shipped with Lean, separate from the historical
weight-semantics proof project. `Lean.Elab.Command` supplies the command
elaboration support required by the audit's `run_cmd`. It needs no Mathlib,
Lake project, dependency installation, historical wrapper, or research
execution.

## Verification status

**Hosted direct check passed on November 10, 2026.**
[Run `42828945211`](https://github.com/shyamsridhar123/weight-space-programming-framework/actions/runs/42828945211),
job `129873557794`, completed successfully, including the direct compiler
invocation, every axiom report, the final transitive audit and source-byte
verification. This checks the abstract model; it does not verify a Python
runtime or discharge organizational or scientific obligations.

| Observed identity | Value |
| --- | --- |
| PR head associated with the run | `c131ff0437e344938038075624b9899f7cd70394` |
| Actual checked PR merge commit | `e72d9da64efae72cc5b797cd6fc26ea69ccf0d3e` |
| `Governance.lean` SHA-256 | `19d9d9de98f64672d0d1b49b6eb3cf058fb10123749369737ff183ee6a83178f` |
| Compiler | Lean `4.24.0`, `x86_64-unknown-linux-gnu`, release commit `797c613eb9b6d4ec95db23e3e00af9ac6657f24b` |
| Direct command | `lean +leanprover/lean4:v4.24.0 formal/governance/Governance.lean` |
| Compiler and audit outcome | Exit 0; namespace transitive axiom audit passed |
| Retained hosted artifact | `governance-proof-42828945211-1`, artifact `13768959809` |
| Downloaded artifact ZIP SHA-256 | `fad4db884a6cd8e5f6fbe842e7f4bc65c05661ab4fc168ff634cd28221de355d` |

The artifact's `commit.txt`, `source.sha256`, `lean-version.txt` and complete
`governance-proof.log` were downloaded and inspected. Its archive identity
matches the upload log, and its source identity matches the published source.
The complete compiler proof output was:

```text
'Governance.step_preserves_request' does not depend on any axioms
'Governance.step_is_bound' does not depend on any axioms
'Governance.mismatched_pins_cannot_step' does not depend on any axioms
'Governance.path_preserves_request' does not depend on any axioms
'Governance.reachable_preserves_request' does not depend on any axioms
'Governance.reachable_step_binds_original' does not depend on any axioms
'Governance.step_approval_is_separate_and_bound' does not depend on any axioms
'Governance.reachable_approval_invariant' does not depend on any axioms
'Governance.reachable_approval_is_separate_and_bound' does not depend on any axioms
'Governance.terminal_no_step' does not depend on any axioms
'Governance.terminal_path_stays' does not depend on any axioms
'Governance.revoked_path_stays' does not depend on any axioms
'Governance.authorized_pins_and_context' does not depend on any axioms
'Governance.authorized_approval_is_separate_and_bound' does not depend on any axioms
'Governance.authorized_is_local' does not depend on any axioms
'Governance.production_authorization_impossible' does not depend on any axioms
'Governance.scientific_authorization_impossible' does not depend on any axioms
'Governance.terminal_cannot_authorize' does not depend on any axioms
Governance transitive axiom audit passed.
```

**Earlier hosted failure retained.** Lean 4.24.0 run `42827681264`, job `129869167142`,
failed because `prefix` was used as a binder and induction-pattern identifier,
but is reserved syntax. The resulting parse errors caused cascading `Path`
failures, and the final transitive axiom audit rejected `sorryAx`. The binder
and affected patterns now use `history`. The successful check above applies
to that corrected source; it does not change the earlier failed result. The
contract and checked model are published before runtime implementation.

**UNCHECKED locally as of November 10, 2026.** No installed Lean executable
was found on PATH or in the existing user/tool locations checked. No local
compiler version or kernel check result is available. The following exact
PowerShell verification command was attempted from the worktree root:

```powershell
$governanceLean = Get-Command lean -CommandType Application -ErrorAction SilentlyContinue
if ($null -eq $governanceLean) {
  Write-Output 'UNCHECKED: no installed lean executable on PATH; no compilation attempted.'
  exit 127
}
& $governanceLean.Source --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $governanceLean.Source 'formal/governance/Governance.lean'
exit $LASTEXITCODE
```

Observed status: **exit 127**, with exactly this diagnostic:

```text
UNCHECKED: no installed lean executable on PATH; no compilation attempted.
```

The command stopped before invoking a compiler. No toolchain was installed,
no historical proof wrapper was run, and no scientific artifact or receipt was
used. A static source review is not a Lean kernel check.

**Local axiom reports and audit result remain unavailable; the hosted result
is retained above.**
Every named theorem retains a corresponding `#print axioms` command. The
source finishes with a transitive
axiom audit of every declaration in the `Governance` namespace, including
private definitions and generated declarations. An empty namespace match is
an error. The audit permits only `propext`,
`Classical.choice` and `Quot.sound`; any other dependency, including `sorryAx`
or an added axiom, causes an error. Neither the reports nor the audit have
executed locally. The absence of source-level `axiom` or `sorry` declarations
does not establish the transitive axiom dependencies of an unchecked proof.

With an already installed Lean 4.24.0 toolchain, the direct standalone check
from the repository root is:

```powershell
lean --version
lean formal/governance/Governance.lean
```

The direct Lean invocation checks the source, emits each `#print axioms`
report, and runs the final namespace audit. These commands are reproduction
instructions; the observed hosted run above is the retained successful
result. Any changed source needs its own observed compiler and audit result.

## Model and obligations

A `Request` contains the exact abstract identities of the base, program,
candidate and constraints; the environment, use case and target; purpose;
the ordered evidence identities; and requester and configured approver labels.
Request equality covers all these fields. The model stores them immutably in
every state, rather than deriving immutability from a hash assumption.

The initial status is `requested`. Approval and rejection require a configured
actor distinct from the requester. Revocation requires an approved state and
the requester or a configured actor. Each transition requires explicit
`ExpectedPins`, matching the current request hash and head, and a decision
that itself names the same request hash and previous head. An abstract
`Approval` retains the actor and request hash recorded in the validated
history. Rejection and revocation admit no successor.

`Authorized` requires a reachable state, both expected pins, exact current
artifact/constraint/scope context, local-pilot scope, approved status, and
local-pilot use. It cannot authorize production or scientific work. A supplied
status value alone does not satisfy this predicate.

| Obligation | Principal theorem statements in the source |
| --- | --- |
| Every transition and reachable path preserves the entire request | `step_preserves_request`, `path_preserves_request`, `reachable_preserves_request` |
| Decisions and expected pins bind the original request and current previous head | `step_is_bound`, `mismatched_pins_cannot_step`, `reachable_step_binds_original` |
| Every reachable approval uses an originally configured, distinct actor and the original request hash | `reachable_approval_is_separate_and_bound` |
| Rejection and revocation have no next transition; revocation cannot be reversed by a path | `terminal_no_step`, `terminal_path_stays`, `revoked_path_stays` |
| Authorization requires exact original context, matching pins, and a separate bound approval | `authorized_pins_and_context`, `authorized_approval_is_separate_and_bound` |
| Authorization is confined to local use in every state | `authorized_is_local`, `production_authorization_impossible`, `scientific_authorization_impossible` |
| Terminal states cannot authorize any use | `terminal_cannot_authorize` |

The hosted check above verified these theorem statements and proof terms,
including their axiom reports and the namespace-wide transitive audit. This
does not verify a Python implementation or establish that a candidate has
scientific or operational value.

## Assumptions and correspondence limits

`Encoding` supplies arbitrary request, initial-head and decision-head
functions. The model introduces no axiom asserting injectivity, collision
resistance or SHA-256 correctness. Consequently, it proves equality guards
and immutable state transitions; it does not prove that different serialized
requests have different hashes or that a coherently substituted request is
cryptographically detectable.

The following links to the runtime remain unproved:

- **Parsing and encoding:** strict JSON schemas, duplicate-key and nonfinite
  rejection, canonical serialization, hash implementation, exact artifact
  identities, and conversion between runtime records and these Lean types.
  Evidence is modeled as an ordered list of opaque identities. Runtime
  metadata is covered only if the correspondence includes it in those
  identities or extends the model.
- **Runtime refinement:** Python replay and checking implement these
  transitions; event positions, nonblank reasons and canonical actor labels
  satisfy their additional runtime requirements; and the supplied candidate
  is the result of applying the supplied program to the base. The last
  structural application relation is not modeled here.
- **Independent expectations:** the caller obtains the request and current
  head from a trusted independent source. Explicit arguments do not prove
  where their values came from. A stale approved snapshot can still satisfy
  the model with stale matching pins. Freshness, concurrent forks and a
  trusted serialized head store are outside this model.
- **Authenticated actors:** an external system maps the labels to people and
  obtains their actual decisions. Unequal strings establish modeled label
  separation, not identity, organizational independence or consent.
- **Storage and export:** JSON bundles remain immutable, outputs use exclusive
  new-file creation, and export copies the complete validated history without
  changing authority. File I/O, persistence, crash behavior and the export
  implementation are not formalized.
- **Effect and enforcement:** evidence commitments are truthful and complete
  only if established elsewhere; local eligibility does not prove beneficial
  behavior or scientific validity. Callers must honor the result. Unmodeled
  execution and deployment routes are not governed by these definitions.

The successful hosted Lean check validates the abstract implications under
these definitions. It does not discharge these correspondence assumptions
or replace the separate synthetic runtime checks.
