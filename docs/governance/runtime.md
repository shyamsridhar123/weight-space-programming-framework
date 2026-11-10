# Local-pilot governance runtime

This optional workflow records review of an exact weight-space edit for a local
pilot. It connects the [contract](contract.md) and
[checked abstract model](../../formal/governance/README.md) to a JSON bundle,
Python API and `weight-space governance` commands. It does not evaluate routing
behavior or authorize production use or scientific work.

## Implementation and verification status

| Layer | Status as of November 10, 2026 |
| --- | --- |
| Proposed obligations | The contract fixes request binding, allowed transitions, independent pins, context matching and permanently false production/scientific authority. |
| Observed abstract proof | The standalone model passed a hosted Lean 4.24.0 check and transitive axiom audit. Proof-only PR #87 merged before runtime implementation; the formal guide retains the command, assumptions and observed identities. |
| Runtime implementation | The API and CLI described below are implemented and have a static correspondence review. |
| Observed runtime verification | Focused synthetic governance checks passed for the source identities below. Consult the publishing PR's checks for hosted verification of its exact commit. This walkthrough has not been executed; no Python-to-Lean correspondence proof or behavioral result is claimed. |

The abstract proof concerns its modeled transitions and premises. Runtime
validation must separately check serialization, hashing, parsing, file handling
and correspondence to that model.

## Proof-to-code correspondence review

A static review compared [the Python runtime](../../rlcd/weight_space/governance.py)
with the contract and [Lean source](../../formal/governance/Governance.lean).
This is engineering evidence, not a proof that Python implements the model.

| Runtime boundary | Compared formal obligation | Reviewed implementation boundary |
| --- | --- | --- |
| `_request` | `reachable_preserves_request` | Validates the complete request before hashing; decisions retain that request and replay checks its independently pinned identity. |
| `_context` | `authorized_pins_and_context`, `authorized_is_local` | Builds the artifact/program/constraint/scope context, checks exact application identity, and restricts scope to `local-pilot`; eligibility compares it with the reviewed context. |
| `_checked` | `reachable_step_binds_original`, `mismatched_pins_cannot_step` | Replays from genesis, checks every event's request, previous head, consecutive position and digest, and compares both independently supplied pins. |
| `_transition` | `reachable_approval_invariant`, `terminal_no_step` | Enforces original approver membership and requester separation for approval/rejection; permits approved-state revocation and no terminal-state transition. |

The runtime also fixes both authority flags to false, consistent with
`production_authorization_impossible` and `scientific_authorization_impossible`.
JSON validation, canonical encoding, SHA-256 correspondence, event positions
and reasons, artifact normalization and application, Python execution and file
I/O remain unproved by Lean. Export has **no separate Lean theorem**: its
preservation argument relies on Python validating one captured source snapshot
and copying those exact bytes. Independent pins, freshness, authentication and
serialized decisions remain caller premises.

## Observed focused runtime checks

The focused synthetic checks in `tests/test_governance.py` passed with exit 0 on
November 10, 2026, finishing at `2026-11-10T14:22:00.234323Z`. The retained
invocation and result record Python 3.12.10, pytest 9.1.1, disabled bytecode and
plugin autoload, and matching before/after source identities. The worktree base
was `8fd6cd5a59105ba0a010da497ddcf0aa27150d83`; the hashes below identify the
runtime files actually checked.

The private receipt directory, outside the worktree, is
`<workspace>/.runs/observed-governance-runtime-20261110-d4b40d659e`.
It retains `invocation.json`, `result.json`, `pytest.log` and `pytest.xml`.
The receipt retains the original absolute paths. For this public display,
`<workspace>` normalizes the workspace path, `<run>` denotes that directory,
and `python` denotes the recorded virtual-environment interpreter:

```text
python -m pytest -q -c pytest-engineering.ini tests/test_governance.py --basetemp <run>/pytest-tmp --junitxml <run>/pytest.xml
```

| Checked file or retained output | SHA-256 |
| --- | --- |
| `rlcd/weight_space/governance.py` | `23021f2c88883c2ccc41870582e49cb70f5b5804f2d460ac865d3bbf525fc8f1` |
| `rlcd/weight_space/__main__.py` | `b68005046dfa82746460e0a4eb98a29ff5b260c573dd32822e1c6138b58a6b9c` |
| `tests/test_governance.py` | `36b476cbcc715fb2c07aff196622b5dd79cee1a2e29cf7562b9668a362b33ceb` |
| `pytest-engineering.ini` | `4542f215c88362e662242f1bfef71cebd74f3b786f7910d9472dd7f1dfd9f882` |
| `pytest.log` | `b0abdc8997fd10092f711096259801252c22a63074f263dbd443c2ea31156fb9` |

This is observed engineering evidence for those sources. Hosted results belong
to the exact commits identified by the publishing PR's checks. The walkthrough
below has not been executed. The checks do
not establish refinement or correspondence to Lean, reviewer authentication,
current-head freshness, routing behavior, production authority or scientific
evidence.

## Prepare a synthetic support-routing request

Suppose Alice proposes routing telemetry outages to Platform Operations and Bob
reviews it for a local pilot. The following data changes a synthetic two-weight
head from `(1, 1)` to `(1, 2)`. The architecture and feature labels are declared
data; no ticket routing or protected behavior is measured here.

With the package installed, save this as `prepare_governance.py` in a fresh
directory and run `python prepare_governance.py`:

```python
import hashlib
import json
from pathlib import Path

from rlcd.weight_space import TensorArtifact, difference

base = TensorArtifact(
    "synthetic-support-routing/1",
    {"head.weight": [[1.0, 1.0]]},
    {"feature_id": "declared-support-features/1"},
)
candidate = base.replace(**{"head.weight": [[1.0, 2.0]]})
program = difference(
    base, candidate,
    "Synthetic request: route telemetry outages to Platform Operations",
)
constraints = {"protected_queues": ["billing", "security", "ordinary-support"]}
scope = {
    "environment": "local-pilot",
    "use_case": "synthetic-support-routing",
    "target": "local-synthetic-head",
}
note = b"Constructed request data only; routing behavior has not been evaluated.\n"
with Path("synthetic-note.txt").open("xb") as stream:
    stream.write(note)
evidence = [{"name": "synthetic-note", "sha256": hashlib.sha256(note).hexdigest()}]

for name, value in {
    "base": base.to_dict(), "program": program.to_dict(),
    "candidate": candidate.to_dict(), "constraints": constraints,
    "scope": scope, "evidence": evidence,
}.items():
    with Path(f"{name}.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
```

The evidence hash commits the note's bytes. It supplies no approval, behavioral
check or evidence of useful impact. Constraints are opaque JSON commitments:
the runtime binds their identity but does not check the protected queues.

## Create, approve and check

Run the following commands in that directory. The examples use PowerShell;
replace the pin placeholders with the full lowercase SHA-256 values retained
from the original successful command returns.

```powershell
weight-space governance request base.json program.json candidate.json `
  --constraints constraints.json --scope scope.json `
  --purpose "Review the synthetic telemetry-routing edit for a local pilot" `
  --evidence evidence.json --requester alice --approver bob --out request.json
```

Retain `request_sha256` and the initial `head_sha256` from your own trusted,
successful request creation. Keep these expectations outside any subsequently
received bundle. An output file's existence is not evidence that the operation
returned successfully.

```powershell
$requestPin = "REPLACE_WITH_REQUEST_SHA256_FROM_TRUSTED_CREATION_RETURN"
$headPin = "REPLACE_WITH_HEAD_SHA256_FROM_TRUSTED_CREATION_RETURN"

weight-space governance decide request.json --actor bob --action approve `
  --reason "Approve review of this exact synthetic request for the local pilot" `
  --out approved.json `
  --expected-request-sha256 $requestPin --expected-head-sha256 $headPin
```

After the decision returns successfully, retain its new `head_sha256`. Keep the
original request pin. The actor label `bob` must be a configured approver and
distinct from `alice`; the caller must separately authenticate Bob and obtain
his actual decision. Different strings do not establish different people or
organizational independence.

```powershell
$headPin = "REPLACE_WITH_HEAD_SHA256_FROM_ORIGINAL_SUCCESSFUL_APPROVAL_RETURN"

weight-space governance check approved.json base.json program.json candidate.json `
  --constraints constraints.json --scope scope.json `
  --expected-request-sha256 $requestPin --expected-head-sha256 $headPin
```

The check replays the full history, compares both pins and the exact current
context, and recomputes `apply_program(base, program)` to match the supplied
candidate identity. This application check is structural; it runs no evaluator.
A valid approved history with matching context yields
`local_pilot_eligible: true`. Both `production_authorized` and
`scientific_authorized` remain `false`.

A valid requested, rejected or revoked history returns exit code **0** with
`local_pilot_eligible: false`. Exit code **2** rejects invalid input, malformed or
mismatched context, invalid history, or mismatched pins. Treat the derived
eligibility value separately from command success.

## Revoke and export the complete history

The requester or a configured approver can revoke an approved request. Use the
current approval head retained above:

```powershell
weight-space governance decide approved.json --actor alice --action revoke `
  --reason "Stop this local pilot pending a new review" --out revoked.json `
  --expected-request-sha256 $requestPin --expected-head-sha256 $headPin

$headPin = "REPLACE_WITH_HEAD_SHA256_FROM_ORIGINAL_SUCCESSFUL_REVOCATION_RETURN"

weight-space governance check revoked.json base.json program.json candidate.json `
  --constraints constraints.json --scope scope.json `
  --expected-request-sha256 $requestPin --expected-head-sha256 $headPin

weight-space governance export revoked.json --out exported.json `
  --expected-request-sha256 $requestPin --expected-head-sha256 $headPin
```

The revoked check is ineligible. Rejection and revocation are terminal; changing
the purpose, constraints, scope or candidate requires a new request. Export
validates the full bundle against both pins and copies its bytes unchanged.
It preserves the history and false authority flags.

All output paths must be new: the commands create files exclusively. A failed
write is a failed operation. Files do not form a transaction or trusted global
ledger, and surviving files cannot reconstruct a missing original successful
return. Resolve such a failure through the caller's trusted coordination process
before making another decision.

## Python boundary and JSON format

The equivalent request API accepts built-in JSON values. Using the variables
from the preparation example:

```python
from rlcd.weight_space.governance import create_request

bundle = create_request(
    base.to_dict(), program.to_dict(), candidate.to_dict(),
    constraints=constraints, scope=scope,
    purpose="Review the synthetic telemetry-routing edit for a local pilot",
    evidence=evidence, requester="alice", approvers=["bob"],
)
request_pin = bundle["request_sha256"]
head_pin = bundle["head_sha256"]
```

Retain these pins directly from this trusted return. Pass `.to_dict()` values,
not artifact or program objects. The module also exposes `decide`,
`check_eligibility`, `export_bundle` and the strict `read_json` helper. Decisions,
checks and exports require both pins explicitly; checks also require the current
context.

Bundles use schema `wsp-governance-bundle/1` and contain the immutable request,
`request_sha256`, ordered `events`, `head_sha256` and false authority flags.
Check results derive status and eligibility from validation. Input rules include:

- Strict finite JSON at file and Python boundaries: duplicate keys, nonfinite
  numbers (including overflowed exponents), unknown envelope fields and invalid
  types are rejected.
- `constraints` may be any strict finite JSON value. `scope` has exactly
  `environment`, `use_case` and `target`; the environment is `local-pilot`, and
  the other values are nonblank strings.
- `evidence` is a nonempty list of objects with exactly `name` and `sha256`.
  Names are unique nonblank strings; digests contain 64 lowercase hexadecimal
  characters. Evidence commitments do not establish truth or freshness.

## Caller obligations

Never obtain an expected head by inspecting an untrusted bundle. Retain each
next head from the original successful decision return and supply the latest
trusted one to every operation. An old approved snapshot can still validate
against old pins after a revocation; offline validation cannot discover that a
newer decision exists. Hashes also do not prevent concurrent decision forks.

The caller must supply external actor authentication, a trusted current-head
store and serialized decisions. This optional pilot adds no identity provider,
signature, deployment hook, behavioral admission, scientific authorization or
enforcement around other application APIs. The
[Distro handoff](distro-handoff.md) is an optional authoring and review aid;
the workflow does not depend on Distro.
