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

**UNCHECKED locally as of November 10, 2026.** No installed Lean executable
was found on PATH or in the existing user/tool locations checked. No compiler
version or kernel check result is available. The following exact PowerShell
verification command was attempted from the worktree root:

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

**Axiom report and audit result: unavailable.** Every named theorem retains a
corresponding `#print axioms` command. The source finishes with a transitive
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
report, and runs the final namespace audit. These are verification
instructions, not a record of a successful run. Retain the actual version,
compiler exit status, complete `#print axioms` output and audit result when
this check is performed. A hosted workflow result also must be observed
before it can be reported as passing.

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

These are proposed theorem statements with proof terms, pending compiler
verification. Their number is not evidence that the implementation is correct
or that a candidate has scientific or operational value.

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

A future successful Lean check would validate the abstract implications under
these definitions. It would not discharge these correspondence assumptions
or replace the separate synthetic runtime checks.
