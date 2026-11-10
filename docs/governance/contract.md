# Governance contract

This contract was written on November 10, 2026, before the governance runtime.
It defines an optional local pilot for reviewing a weight-space change. It is
generic across architectures and applications. Microsoft Distro may help
prepare and review the handoff described in [the guide](distro-handoff.md);
the bundle format and checker do not depend on Distro.

A concrete example is a support-routing owner proposing an edit to a routing
head. The owner supplies the exact base, edit program, resulting candidate,
constraints, target environment, purpose and evidence commitments. A distinct
configured reviewer may approve that request for a local pilot. Approval does
not establish that the routing edit improves outcomes. Those outcomes need
separate, predeclared evaluation and complete attempt accounting.

## Invariants before implementation

An immutable request binds the content identities of the base, program,
candidate and constraints; the complete scope, purpose and evidence list; and
the requester and configured approver labels. The only supported environment
is `local-pilot`. Scope also names the use case and target. Identical artifacts
with different constraints, scope or purpose constitute different requests.

The initial status is `requested`. A configured approver whose canonical label
differs from the requester may approve or reject it. An approved request may
be revoked by the requester or a configured approver. Rejected and revoked
states are terminal. Every decision commits the original request identity,
the previous head, its ordered position, actor, action and nonblank reason.
No transition edits the request.

Every decision and eligibility check requires independently retained expected
request and current-head identities. The checker recomputes the complete
history and compares both expectations. Eligibility also requires the exact
current base, program, candidate, constraints and scope, and approved status.
Removing a revocation or substituting a coherently rehashed request fails
against the appropriate independent expectations.

The checker always reports `production_authorized = false` and
`scientific_authorized = false`. It does not execute the candidate, a provider,
an admission gate or a study. Export validates and copies the same complete
bundle; it cannot change authority. Existing application APIs remain separate;
this pilot is not mandatory enforcement around every framework caller.

## Proof obligations and implementation correspondence

The [standalone formal model](../../formal/governance/README.md) is deliberately
separate from the historical weight-semantics proofs. Its validation status
and exact command are recorded there. An unchecked source is not a checked
proof, and checking the abstract model is not verification of the Python
implementation.

| Obligation | Runtime boundary | Remaining premise |
| --- | --- | --- |
| Reachable transitions preserve the complete request | Exact request schema, canonical request digest, append-only transition construction, complete-history validation | Canonical serialization and SHA-256 correspondence; trusted starting expectation |
| Approval uses a configured label distinct from requester | Canonical actor validation, original approver membership and requester inequality on every replayed approval | An external authenticated system maps labels to people and establishes organizational independence |
| Decisions bind the original request and previous head | Exact request/head pins, consecutive event positions, recomputation of every event digest | The caller retains independent pins and serializes decisions; a hash does not prevent concurrent forks |
| Rejection and revocation are terminal | No transition out of either status; full history replay before checking or export | The trusted head reflects the most recent decision |
| Local eligibility requires approved status and exact context | Exact current artifact/program/constraint/scope comparison, valid application relation, mandatory expected request/head pins | Inputs are the artifacts actually used; local eligibility does not establish useful behavior |
| No production or scientific escalation | Output flags are fixed false in every status, including approval | The caller honors the result; no unmodeled deployment route is governed |
| Export preserves authority | Validated complete bundle is copied without dropping history or changing request/head identities | External storage preserves bytes and supplies current independent expectations |

## Trusted inputs and explicit limits

Actor labels use one canonical lowercase ASCII spelling. Label inequality is
a modeled separation rule, not authentication. This pilot has no signatures,
identity provider, credential handling, approval service or deployment hook.
The caller must authenticate actors and obtain their actual decisions outside
this code. Possession of a bundle is not permission to act for its labels.

The trusted current head is essential. An offline approved bundle cannot
prove that a later revocation does not exist. A check establishes eligibility
as of the independently pinned head only. Supplying the bundle's own values
as expectations verifies internal consistency but supplies no freshness or
substitution protection. An organizational deployment needs a trusted current
head store, serialized updates and authenticated actors before relying on it.

The supplied constraints are committed as strict finite JSON. The checker
binds their exact canonical representation; it does not interpret them as a
behavioral contract or establish that a candidate satisfies them. Evidence
hashes commit named bytes or records supplied by the caller; they do not prove
truth, completeness, provider origin, provenance, useful impact or freshness.
Artifact identities use the framework's existing canonical artifact/program
semantics; constraints use canonical JSON rather than original file formatting.

Request creation and checking recompute `apply_program(base, program)` and
require its identity to equal the supplied candidate. This is a structural
application relation. It adds no behavioral guarantee and runs no evaluation.
Historical proof assumptions, study protocols and failure receipts remain
unchanged.

All JSON entry points reject duplicate keys, unknown envelope fields, invalid
types and nonfinite numbers, including overflowed exponents. The public Python
boundary validates and snapshots supplied JSON values as well. Output files
are created exclusively. File output is not a durable transaction, a global
append-only ledger, an attestation of successful execution, or an atomic update
of a trusted head. A failed write must be treated as failed; existing files
cannot reconstruct a missing original successful return.

## Verification scope

Focused synthetic fixtures should cover the valid request/approval/revocation
workflow and its boundaries: self/unknown approvers; changed artifacts,
constraints, purpose, evidence and scope; coherently rehashed substitution;
reordered or truncated histories; removed revocation against the newer head;
terminal transitions; malformed JSON/types/fields; export preservation; and
permanently false production/scientific flags. They use newly constructed
artifacts and temporary files. They are implementation evidence, not an
enterprise deployment or a scientific benefit demonstration.

The mathematical obligation, its assumptions, implementation checks and actual
verification result must remain separately visible in the documentation and PR.
No acceptance criterion is promoted using theorem counts or test totals.
