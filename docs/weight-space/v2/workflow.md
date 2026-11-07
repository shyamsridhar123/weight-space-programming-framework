# Select a proposal, then evaluate it

A candidate can have a valid program and still be the wrong choice. The session
workflow records what the proposer could observe, which candidates it submitted,
which candidate the caller selected, and what the final evaluator reported. Selection
happens before the held-out evaluation.

This workflow serves reading, diagnosis, editing, and creation across declared
architectures. The session owns accounting and immutable records. The caller
owns model semantics, success criteria, and the evaluation data. It is a
cooperative Python interface: code in the same process can access information
outside the intended tool methods. A trace therefore describes interface use;
it does not establish a restricted execution environment.

## Follow the worked edit through selection

Start with the same parent and proposals as the [worked example](worked-example.md).
The parent is `w = (1, 1)`. The public target `(1, 1)` should move from 2 to 3;
the protected row `(1, 0)` should stay at 1.

| Explicit proposal | Public target error | Public protected drift | Eligible under exact public protection? |
| --- | ---: | ---: | --- |
| Keep `(1, 1)` | 1 | 0 | Yes, but it misses the target |
| Change to `(1.5, 1.5)` | 0 | 0.5 | No |
| Change to `(1, 2)` | 0 | 0 | Yes |

Fix the selection rule before seeing the final examples: retain candidates with
zero public protected drift, then select the one with the smallest public
target error. The no-op remains eligible. If no candidate meets the rule, the
caller can abstain instead of forcing an edit.

The runnable [session example](../../../examples/session_edit.py) also declares
the broader formula `desired output change = x2`. That is a strong supplied
hint. It writes the proposals explicitly, so it demonstrates the API without
measuring whether a language model can discover the rule or the edit.

```bash
python examples/session_edit.py
```

The script performs the following sequence:

| Stage | Concrete action | Record or decision |
| --- | --- | --- |
| Declare | Bind the exact parent, feature order, formula hint, access policy, budgets, and split IDs. | A `TaskSpec` and session configuration. |
| Observe | Read metadata and the two parent coefficients through the session. | Charged observations in the trace. |
| Propose | Explicitly check each literal program with a structural policy, then submit it. | Immutable candidates; structural acceptance still leaves behavior unchecked. |
| Compare | Evaluate the public target and protected row for each candidate. | The table above, retained in feedback events. |
| Select | Apply the fixed rule and call `seal(selected)`. | The protected edit is selected; proposer methods close. |
| Evaluate | The caller checks new rows with `x2 = 1` and `x2 = 0`. | Zero target error and zero protected drift for this constructed formula. |

The central selection code is an excerpt from that complete script:

```python
eligible = [row for row in candidates if row[2]["protected_drift"] == 0.0]
selected_label, selected, _ = min(eligible, key=lambda row: row[2]["target_error"])
session.seal(selected)
receipt = session.evaluate_heldout(
    evaluate, evaluator_id="illustrative-linear-spec/1", split_name="test",
)
```

These simple stored coefficients and integer features make zero an exact
arithmetic case here. For another task, the caller must define tolerances,
metrics, and numerical checks deliberately. The script's final rows are new to
the public feedback function, but they share an analytic construction and a
known formula. They are an illustration of the selection boundary, not an
independent empirical confirmation.

## Declare the task and its success criterion

`TaskSpec` binds the objective, architecture/interface identifier, optional
constraints, and exact parent hash. All modes except `create` require a parent;
`create` forbids one. The session checks that the supplied immutable `Network`
or `TensorArtifact` matches the task. The built-in `Network` interface identifier
is `dense-relu-dense/1`; named tensor artifacts use the caller's identifier.

| Mode | Output submission | Meaning supplied by the caller |
| --- | --- | --- |
| `read` | `submit_answer(finite_json)` | Identification, description, or another interpretation rubric. |
| `diagnose` | `submit_answer(finite_json)` | A mechanism or failure hypothesis and its scoring rubric. |
| `edit` | `propose_program(program)` or `propose_artifact(artifact)` | A changed artifact with the parent's architecture, tensor shapes, and semantic metadata. |
| `create` | `propose_artifact(artifact)` | A new artifact with the declared architecture and no parent. |

The mode routes outputs; it does not supply a correctness rubric. Constraints
such as `preserve_bias`, forbidden fitting, writable tensors, or an accuracy
threshold remain declared context unless the caller adds checks. In the example,
the evaluator explicitly measures target error and protected output drift. Metadata, shape, and hash
checks alone do not establish that behavior is preserved.

Every successful submission returns a `Candidate`, including a valid no-op. Its
`sha256` commits the task, session configuration, parent, proposal index, proposal
kind, and realized output. `candidate.artifact.sha256` identifies the realized
artifact and is a different identity. Answer candidates have `artifact=None`
and a detached finite-JSON `answer`. `feedback`, candidate execution, and `seal`
accept only the exact candidate object produced by that session; reconstructing
or copying an equal-hash candidate does not grant selection rights. A session's
`sha256` identifies its configuration, not a globally unique execution. Identical
deterministic runs can produce identical hashes.

## Specify what the proposer can observe

`AccessPolicy` independently permits metadata, weight reads, public feedback,
forward execution, and activation inspection. Metadata includes task constraints
and artifact semantic names. Record supplied labels, training information,
privileged hints, and semantic names explicitly in the task and experiment
configuration. Public accounting commits hidden constraint contents and split
identifiers by hash; an authorized `metadata()` call intentionally reveals the
constraints. Other configuration fields, such as the objective and architecture,
are public.

`weights(tensor, indices=None)` reads a whole parent tensor or explicit coordinates.
It counts returned scalar entries, including repeated coordinates. Setting
`weight_decimals` to an integer from 0 through 15 rounds only these read results;
the immutable parent retains its original values. Exact access uses `None`.
Python's numeric rounding is used, not a promise of fixed-width decimal text.

Program proposals return the entire realized artifact through
`Candidate.artifact`. They therefore require both metadata and exact weight
access, charge the full artifact scalar count on every admitted program attempt, and
reject rounded-only access. This includes no-ops and failed program attempts.
`propose_artifact` returns data the caller already supplied, so it does not
require parent-disclosure capabilities or charge a weight read. It still checks
the declared architecture and, for edits, compatibility with the parent. Neither
route makes an untrusted callback safe to expose private information.

## Supply callbacks with explicit responsibilities

| Callback | Arguments | Required output |
| --- | --- | --- |
| `public_feedback` | `(candidate, examples)` | Finite JSON with public validation observations or scores. |
| `forward` | `(artifact, examples)` | Finite JSON containing the permitted execution result. |
| `activations` | `(artifact, examples)` | Finite JSON containing the permitted intermediate observations. |
| Held-out evaluator | `(selected_candidate)` | Finite JSON for the final receipt. |

Tool example batches must be nonempty lists, tuples, or NumPy arrays. NumPy
scalars and arrays are normalized to finite JSON, and input/output snapshots are
detached from caller mutations. A callback returning a Python object, a
`Candidate`, NaN, or infinity fails output validation. Public callbacks cannot
reenter an active session. Adapters are actual caller-owned callables; this
interface does not import callback names or execute serialized source code.

## Account for admitted attempts and failures

`BudgetLimits` and `BudgetUsage` track the following independently:

| Counter | Unit charged by the interface |
| --- | --- |
| `tool_calls` | Metadata, weights, proposal, feedback, forward, or activation attempts admitted for execution. |
| `weight_values` | Scalars disclosed by weight reads or the full realized-artifact size reserved by a program attempt. |
| `proposals` | Program, full-artifact, or answer submission attempts. |
| `feedback_calls`, `feedback_examples` | Admitted public feedback attempts and the number of submitted rows. |
| `execution_calls`, `execution_examples` | Admitted forward and activation attempts, sharing counters, and their submitted rows. |
| `input_tokens`, `output_tokens` | Counts explicitly supplied through `record_provider_usage`. |

All limits are nonnegative integers; booleans are rejected. Budget additions are
atomic: an attempt exceeding any applicable limit changes no counters. Denied
capabilities, invalid modes, malformed coordinates/batch containers, and foreign
candidate references are recorded as rejected actions without execution charges.
Once an operation is admitted, its full planned charge is applied before
execution, including adapter failures, nonfinite results, stale edit parents,
precondition failures, and proposal parser failures. Malformed serialized proposal
objects that are finite JSON retain the submitted data and digest in the trace. Failed submissions
do not become candidates. Token telemetry is provider-reported and does not add
a tool call; it is not a locally measured tokenizer count.

The example charges eight weight values: two explicitly read coefficients and
two coefficients for each of three program proposals. Its three validation
callbacks account for six submitted examples. These quantities explain what
the interface disclosed and charged; they do not measure scientific quality. These counters do not observe how many model calls,
optimizer steps, samples, or tokens a callback uses internally. Human selection,
external editor attempts, provider requests, wall time, and work outside the
interface require separate experiment records. These budgets do not implement a
no-fitting or no-network regime.

## Seal selection before the final evaluation

`seal(candidate)` fixes selection and closes every proposer operation. Later
proposer attempts raise `SessionSealed` without changing the sealed history.
`seal()` or `seal(None)` records abstention. Held-out data stay in a caller-owned
closure and are not passed to proposer tools by this interface. The evaluator
owner calls `evaluate_heldout` only after selection is sealed. A valid invocation
runs the evaluator at most once, including a failed evaluation; abstention
returns an `abstained` receipt without calling it. Evaluator exceptions produce a
`failed` receipt, with their messages redacted to avoid copying private samples.
System interrupts also produce a receipt, then propagate the interrupt.

The receipt retains the selected candidate, all successful candidates, failures
in the trace, configuration, final usage, evaluator ID, split name, and finite
result. Held-out invocation count is separate from the proposer budget; held-out
example count and internal evaluator work are unobserved. The evaluator ID is a
caller-supplied label, not a verified code digest. The evaluator's returned JSON
is not checked for data leakage. Keep its outputs scoped to intended metrics and
retain the scorer's version, data manifest, randomness, and actual execution
counts in the surrounding experiment record. A finite score or successful
receipt is neither a formal certificate nor evidence of useful improvement by
itself.

## Keep split labels separate from independence

`SplitSpec` records sample IDs and optional lineage IDs. `validate_splits` rejects
duplicate IDs within a split, repeated split names, sample overlap across splits,
and overlap among declared lineage sets. A lineage list is the set of lineage
identifiers represented in that split, not necessarily one entry per sample.
Validation commits each manifest and reports counts without exposing its IDs.
It does not infer ancestry, inspect actual data, detect undeclared duplicates,
or establish independence. When splits are supplied, the held-out split name
must be declared. Callers remain responsible for keeping public validation and
private evaluation contents separate.

## Replay a finite action record

`ReplayProvider` is a deterministic data script useful for exercising the
interface or replaying already recorded actions. It accepts a list or tuple of
at most 10,000 finite-JSON action objects. It uses a fixed whitelist, with no
dynamic imports, network requests, or live provider calls. Registered session
callbacks can still perform caller-owned work. This self-contained example
submits and selects a no-op:

```python
from rlcd.weight_space import Program, TensorArtifact
from rlcd.weight_space.workflow import ProgrammerSession, ReplayProvider, TaskSpec

base = TensorArtifact("illustrative-linear-head/1", {"weight": [[1.0, 1.0]]})
task = TaskSpec("replay-no-op", "edit", "Keep the parent", base.architecture, base.sha256)
replay_session = ProgrammerSession(task, base=base)
actions = [
    {"op": "metadata"},
    {"op": "weights", "tensor": "weight", "indices": [[0, 0]]},
    {"op": "propose_program",
     "program": Program(base.sha256, (), "Retain the baseline").to_dict()},
    {"op": "seal", "candidate": 0},
]
outcomes = ReplayProvider(actions).run(replay_session)
assert replay_session.is_sealed
assert all(outcome["status"] == "succeeded" for outcome in outcomes)
```

The remaining action forms are `propose_artifact` with `artifact`, `submit_answer`
with `answer`, `feedback` with `candidate` and `examples`, `forward` or
`activations` with `examples` and an optional `candidate`, and `provider_usage`
with `input_tokens` and `output_tokens`. Candidate indexes are zero-based among
successful replay proposals; a failed parse does not occupy an index. Omitting
the candidate for execution uses the parent, and omitting it for `seal` abstains.
Replay records ordinary action failures and continues until it seals or runs out of
actions; system interrupts propagate. The caller must seal explicitly if the script ends without selection.
Held-out evaluation is deliberately outside the replay action set. The trace
labels replay as `scripted_replay` and `model_ability_measured=False`.

## Read the receipt without extending its claim

Trace events contain a sequence number, previous-event hash, request, result or
error, action charges, and cumulative usage. `verify_trace` checks the hash links
and cumulative accounting. Preserve `receipt.to_dict()` together with its
`receipt.sha256` or a trusted `trace_head_sha256` outside the mutable experiment
directory. A hash chain can reveal edits or truncation relative to that trusted
commitment; it cannot detect wholesale replacement if the attacker also replaces
the claimed head. Successful verification does not establish that callbacks did
what their labels claim, that no work occurred outside the interface, or that a
proposer could not access private attributes in the same Python process.


For a study, pair this interface record with the [methods](methods.md): the
proposer mechanism, actual computation, independent units, comparator, selection
rule, and failure policy must be frozen together. For a stronger execution
boundary, follow the unresolved access and provider work in the
[limitations and next steps](limitations-and-next-steps.md).
