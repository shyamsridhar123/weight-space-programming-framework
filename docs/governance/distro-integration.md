# Distro integration contract

This contract precedes the adapter implementation. Distro's coding host loads a
companion plugin with a workspace-bound MCP server and a governance skill. The
server delegates to the existing [governance runtime](runtime.md). It introduces
no decision transition or additional approval authority.

A support owner can use it to prepare the request “route telemetry outages to
Platform Operations while preserving the other queues,” inspect the exact edit
and attached evidence, and follow its review. The same interface accepts other
framework use cases; the routing example is synthetic.

## Proof obligations before implementation

The [checked governance model](../../formal/governance/README.md) already states
the required invariants. The adapter must preserve these boundaries:

| Obligation | Existing formal property | Adapter requirement |
| --- | --- | --- |
| Bind the original request | `reachable_preserves_request` | Use `create_request`; hash the actual evidence bytes and retain every request field. |
| Reject substituted or stale histories | `mismatched_pins_cannot_step` | Require independently supplied request and current-head pins on every status/export call. Never read an expected pin from a received bundle. |
| Bind the current change | `authorized_pins_and_context` | Call `check_eligibility` with the current artifacts, program, constraints and scope; require the reconstructed complete request to match the original request pin. |
| Preserve independent decision labels and terminal states | `reachable_approval_invariant`, `terminal_no_step` | Expose no MCP tool for approval, rejection or revocation; decisions use the existing human-operated CLI and its validated transitions. |
| Limit the meaning of approval | `production_authorization_impossible`, `scientific_authorization_impossible` | Return both authority flags as literal `false`; an approved request can establish local-pilot eligibility only. |

This is reuse of an already checked abstract model, not a new theorem or a
Python-to-Lean refinement proof. MCP parsing, descriptor validation, workspace
paths, evidence reads, hashing, serialization, packaging and host discovery are
separate engineering obligations. Local checks and host validation must report
their actual scope. Authentication, a trusted latest-head store and concurrent
decision serialization remain external caller responsibilities.

## One change file

All paths in a `wsp-distro-change/1` descriptor are normalized relative paths
from the configured workspace root, including paths in a nested descriptor.
The descriptor has exactly these fields:

```json
{
  "schema": "wsp-distro-change/1",
  "base": "base.json",
  "program": "program.json",
  "candidate": "candidate.json",
  "constraints": "constraints.json",
  "scope": "scope.json",
  "purpose": "Review the telemetry outage routing change for a local pilot",
  "requester": "alice",
  "approvers": ["bob"],
  "evidence": [
    {"name": "review-note", "path": "evidence/review-note.md"}
  ]
}
```

The files contain the existing framework artifact, program and governance
formats. The server reads strict finite JSON with duplicate-key rejection using
`governance.read_json`. Evidence entries have exactly `name` and `path`; the
server computes SHA-256 from the attachment bytes, rather than accepting a
declared attachment identity. Evidence is committed content, not an assertion
that a proof or a behavioral check succeeded. Constraints remain opaque
commitments under the governance runtime.

The server validates the complete descriptor by constructing a request with
`create_request`. Inspection returns the exact request, the proposed edits and
bounded text previews of its attachments. Previews are untrusted evidence
content; they do not instruct the host to execute anything. A draft inspection
does not create a review record or confer eligibility.

## Tools and results

| MCP tool | Inputs | Successful result |
| --- | --- | --- |
| `wsp_inspect_change` | `change` | Exact draft request, edit summary and attachment identities/previews; no output file. |
| `wsp_prepare_review` | `change`, `out` | Exclusively created request bundle, its original request/head pins, and requested status. |
| `wsp_review_status` | `change`, `bundle`, `expected_request_sha256`, `expected_head_sha256` | Exact-context eligibility, complete immutable request and decision history, and current attachment identities/previews. |
| `wsp_export_review` | Status inputs plus `out` | Validated complete history copied byte for byte to an exclusively created output; the same eligibility result. |

Tool argument schemas reject extra properties. All tool inputs are plain
strings. Malformed input, invalid context, changed evidence, mismatched pins,
invalid history and unsafe paths are tool errors. A valid requested, rejected or
revoked review is a successful operation with `local_pilot_eligible: false`.
Every successful tool result fixes `production_authorized` and
`scientific_authorized` to `false`.

The requester retains both pins from the original successful preparation
return outside subsequently received bundles. After a successful independently
performed decision, retain that decision's new head and the original request
pin. An old approved snapshot must fail against a retained newer revocation
head. Passing an older pin does not establish freshness; the server provides no
trusted latest-head database.

There is no agent decision tool. An authorized human uses
`weight-space governance decide`, explicitly supplies both pins, and returns
the resulting bundle and new head to the session. Actor labels are not
authentication; the organization must establish the person's identity and
authority separately. Recording revocation does not undo an external deployment.

## Workspace and host boundaries

Setup generates a companion plugin under the workspace's excluded `.runs/`
directory. It records the exact interpreter and resolved workspace in local
configuration; those machine-specific files are not publication inputs. It
does not edit Distro's installation, global host configuration or credentials.
Both native hosts load the generated plugin alongside Distro through
`--plugin-dir`.

The adapter exposes no shell execution, arbitrary file-reading, provider,
study-runner or deployment tools. It reads only the descriptor and referenced
workspace files, and writes only requested new workspace outputs. It rejects
absolute paths, traversal, reserved/nonportable path syntax, symbolic links and
Windows reparse points. It checks paths before I/O; this is not a filesystem
sandbox or protection against another process racing to replace directories.
Other coding-host tools retain their own capabilities and permissions.

The official SDK handles the MCP transport. Strict application JSON validation
does not claim stricter wire parsing than the SDK supplies. Stdout is protocol
traffic only; diagnostics go to stderr. No listening HTTP service or credentials
are needed for this local adapter.

## Required implementation evidence

Use fresh synthetic fixtures to check protocol discovery and calls, unknown
arguments, strict descriptor JSON, artifact/evidence substitution, original-pin
preservation, approved-snapshot rollback after revocation, terminal states,
literal false authority flags, workspace/link rejection, exclusive writes and
byte-preserving complete exports. Exercise the installed wheel outside the
checkout, including the packaged skill and generated native manifests. Validate
host discovery without a model prompt and report live model-session behavior
separately. Do not execute historical studies or promote scientific acceptance
criteria as part of this integration.
