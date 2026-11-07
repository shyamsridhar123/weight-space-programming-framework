# Carry an edit from a request to an evaluated artifact

In the [worked example](worked-example.md), the parent weights are `(1, 1)`. We
want the output at `(1, 1)` to rise from 2 to 3 while the output at `(1, 0)` stays
at 1. The endpoint `(1, 2)` satisfies both conditions in the declared linear
model. The framework makes that proposal inspectable: it binds the parent,
records the changed coefficient, checks the realized endpoint, and retains the
evidence used to select it.

This guide follows those boundaries through the implementation. The same
artifact and session interfaces can serve a regression head, an embedding-based
predictor, a controller, or a game scorer. Their forward equations, protected
domains, and success metrics must be supplied separately. A common tensor
format does not make those meanings interchangeable.

## Decide what each check must establish

A proposal passes through several checks because each answers a different
question. In the example, a valid assignment from 1 to 2 establishes the stored
change. Evaluating the target establishes the requested output. Checking the
protected feature establishes retention there. None of those steps establishes
coverage of an unseen input population.

| Layer | What can be established | What remains to check |
| --- | --- | --- |
| Artifact and literal program | Exact parent identity, finite stored values, valid coordinates, preserved tensor/interface structure | Meaning of the architecture and task behavior |
| Structural admission | Allowed tensors, effective edit count, parameter magnitude, per-coordinate change limits | Output preservation or useful repair |
| Structured lowering | Ordered command realization, declared dtype projection, actual endpoint and program identities | Correctness of the numerical compiler and behavior of the endpoint |
| Architecture-specific certificate | A stated real-valued inequality on its declared domain, or exact residuals on supplied feature rows | Floating execution, extractor correspondence, downstream decisions, reward, and generalization |
| Session and evaluation receipt | Declared access, interface usage, candidate history, selection, and a caller's evaluation result | Process isolation, evaluator honesty, hidden work, leakage, and scientific independence |
| Empirical study | Measurements under a separately frozen protocol | Claims about unmeasured tasks, models, proposer mechanisms, or populations |

The usual flow is:

![Generic framework architecture. A StructuredProgram passes through explicit lowering to a literal Program. The literal Program, TaskSpec with access and budgets, and an answer or complete artifact feed ProgrammerSession. Candidate history proceeds to caller checks and selection, sealing a candidate or abstaining, a caller-owned held-out evaluator, and EvaluationReceipt. Callers must run checks and retain structured proposals and lowering receipts.](../../images/architecture.png)

[Editable SVG](../../images/architecture.svg)

Checks in the diagram are caller responsibilities. A session does not automatically run a structural policy, a behavioral contract, or a feature certificate. Structured proposal text and lowering receipts also need to be retained by the caller; submitting the resulting literal program does not automatically retain those upstream files.

## Locate the implementation

Public convenience imports are in [`rlcd.weight_space`](../../../rlcd/weight_space/__init__.py).
Use this map when inspecting a particular boundary. The study runner is a
consumer of the framework; its experimental outcomes are in [results](results.md).

| Module | Responsibility |
| --- | --- |
| [`artifact.py`](../../../rlcd/weight_space/artifact.py), [`model.py`](../../../rlcd/weight_space/model.py) | Immutable named tensors, canonical identities, the specialized two-layer `Network`, and `InputBox` |
| [`program.py`](../../../rlcd/weight_space/program.py) | Literal edits, exact-parent application, differences, composition, rollback, interpolation |
| [`structured.py`](../../../rlcd/weight_space/structured.py) | Data-only block and factor commands, explicit native projection, replayable lowering receipts |
| [`semantics.py`](../../../rlcd/weight_space/semantics.py), [`sequential.py`](../../../rlcd/weight_space/sequential.py) | Caller-owned adapter registry and finite sequential execution/certification |
| [`features.py`](../../../rlcd/weight_space/features.py), [`compiler.py`](../../../rlcd/weight_space/compiler.py) | Generic frozen-feature fitting and endpoint receipts; the earlier specialized ReLU readout fitter |
| [`contracts.py`](../../../rlcd/weight_space/contracts.py), [`symmetry.py`](../../../rlcd/weight_space/symmetry.py) | Two-layer real drift admission; explicit ReLU hidden-coordinate transformations and transfer |
| [`torch_bridge.py`](../../../rlcd/weight_space/torch_bridge.py) | Optional capture and exact native restoration for an existing caller-supplied PyTorch module |
| [`workflow.py`](../../../rlcd/weight_space/workflow.py) | Task, access, budgets, candidates, traces, sealing, held-out receipts, and finite replay |
| [`__main__.py`](../../../rlcd/weight_space/__main__.py) | JSON artifact/program commands and explicit verification commands |
| [`study_v2.py`](../../../rlcd/weight_space/study_v2.py), [`study_tasks_v2.py`](../../../rlcd/weight_space/study_tasks_v2.py) | Protocol-driven experiments and deterministic auxiliary task fixtures, separate from the core interfaces |

## Bind the parent and write a literal edit

`TensorArtifact(architecture, tensors, metadata)` captures arbitrary-rank named finite binary64 tensors, including scalars. Architecture and metadata are declarations, not executable code. A convolution kernel can be represented without giving the generic artifact a convolution evaluator. Tensor names, shapes, architecture, metadata, and values contribute to its canonical JSON identity. Signed zero is normalized; this identity is not a hash of an arbitrary original checkpoint file or of Python forward code.

`WeightArtifact` is the protocol consumed by the edit algebra: `tensors`, `sha256`, `to_dict()`, `compatible()`, and `replace()`. The CLI and current session implementation accept the built-in `Network` and `TensorArtifact` formats. Supporting another implementation of the protocol in edit helpers does not automatically teach those loaders about it.

An `Edit` specifies a tensor, a coordinate tuple, its exact stored `before` value, and its `after` value. A scalar uses `index=()`. A `Program` binds an ordered tuple of distinct edited coordinates to an exact parent SHA-256 and a human-readable intent. The intent remains a hypothesis. Invalid values, unknown fields, duplicate coordinates, stale parents, out-of-range coordinates, and wrong `before` values are rejected. Application validates the complete edit set before constructing a new artifact; it does not modify the parent.

`difference(base, candidate, intent)` emits effective changed coordinates. `compose(base, programs)` requires the actual intermediate parents in sequence. `rollback(base, program)` constructs literal assignments back to the original stored values; it does not rely on subtracting a floating update. `interpolate` performs numerical coordinate interpolation and makes no behavioral claim. Compatibility includes semantic metadata, so a change of feature order or output meaning cannot silently masquerade as an ordinary edit.

The following code makes the protected edit from the worked example. After
[installing the package](../../../REPRODUCING.md#2-install-the-python-distribution),
run it with that environment's Python interpreter. It checks structure and
rollback; the [worked example](worked-example.md#carry-the-same-example-through-the-api)
adds the separate behavioral receipt.

```python
from rlcd.weight_space import (
    Edit, Program, StructuralPolicy, TensorArtifact,
    apply_program, rollback,
)

base = TensorArtifact(
    "illustrative-linear-head/1", {"head.weight": [[1.0, 1.0]]},
    {"feature_id": "two-declared-features/1"},
)
program = Program(
    base.sha256,
    (Edit("head.weight", (0, 1), 1.0, 2.0),),
    "Raise the target; preserve (1, 0)",
)
policy = StructuralPolicy(
    writable=("head.weight",), max_edits=1,
    max_parameter_abs=2.0, max_change_abs=1.0,
)
admission = policy.check(base, program)
assert admission["structural_accepted"]
assert admission["behavioral_status"] == "not_checked"
candidate = apply_program(base, program)
assert candidate.tensors["head.weight"].tolist() == [[1.0, 2.0]]
assert apply_program(candidate, rollback(base, program)).sha256 == base.sha256
assert Program.from_dict(program.to_dict()).sha256 == program.sha256
```

`StructuralPolicy.check` returns an admission record; callers must act on its `structural_accepted` field. `apply_program` alone checks identities and edit validity, without imposing such a policy. The CLI's `apply --policy` performs both steps and refuses rejected candidates.

## Realize the proposal in its native format

A proposer may describe a whole block or a factored update. The endpoint still
has to become actual stored values before it can be checked. `StructuredProgram`
is immutable JSON data with four operations. No command imports a module or
evaluates provider-authored source.

| Operation | Operand | Effect on the selected tensor or block |
| --- | --- | --- |
| `replace` | `values` | Assign an equally shaped array |
| `add` | `values` | Add an equally shaped array |
| `scale` | `factor` | Multiply by a finite scalar |
| `low_rank` | `left`, `right`, `alpha` | Add `alpha * (left @ right)` to a matrix |

An optional `block` contains one half-open `[start, stop]` pair per tensor axis. Broadcasting, empty blocks, unknown tensors, and incompatible factor dimensions are rejected. Commands run sequentially on temporary binary64 arrays. Lowering projects to native formats once, after all commands. It then creates the ordinary literal `Program` from the realized endpoint.

`native_dtypes` must cover every captured tensor exactly; its default is all `float64`. Supported formats are `float64`, `float32`, `float16`, and optional PyTorch-backed `bfloat16`. The source must already be representable in its declared formats. `cast="reject"` rejects any further rounding; `cast="round"` permits finite projection and records it. Nonfinite overflow is rejected in both modes. This prevents a dtype declaration from silently recasting unrelated source values.

The receipt distinguishes the proposal, requested binary64 endpoint, realized endpoint, and literal program hashes. For each tensor it reports touched coordinates, requested changes, effective changes, rounded coordinates, and the maximum exact difference between the requested and projected binary64 values. Touched-coordinate counts are unions across commands, not counts of assignments. Compact factors can change many stored cells, and rounding can change the algebraic properties of their intended product.

```python
from rlcd.weight_space import (
    StructuredProgram, TensorArtifact, apply_program,
    lower_structured, verify_lowering,
)

base = TensorArtifact("example-affine/1", {"weight": [[1.0, 0.0]]})
proposal = StructuredProgram(base.sha256, [
    {"op": "add", "tensor": "weight", "values": [[2.0**-13, 0.0]]},
    {"op": "low_rank", "tensor": "weight",
     "left": [[1.0]], "right": [[0.0, 0.5]], "alpha": 1.0},
], "Show a requested change that disappears in float16")
lowering = lower_structured(
    base, proposal, native_dtypes={"weight": "float16"}, cast="round",
)
candidate = apply_program(base, lowering.program)
assert candidate.tensors["weight"].tolist() == [[1.0, 0.5]]
counts = lowering.receipt["projections"]["weight"]
assert counts["requested_changed_cells"] == 2
assert counts["effective_changed_cells"] == 1
assert counts["rounded_cells"] == 1
assert verify_lowering(base, proposal, lowering.receipt)
```

This check concerns the installed NumPy/PyTorch conversion behavior. The exact rounding-error field measures endpoint differences; it is not a theorem about every intermediate operation, hardware kernel, or compiler. `verify_lowering` reruns lowering and compares the complete receipt, including types and recorded versions. It is an executable consistency check, not a separately verified compiler.

## Give the artifact executable meaning

The linear equation in the worked example comes from the example's declared
model. For an application, a caller must supply that meaning through an adapter.
`SemanticsRegistry` is caller-owned. `SemanticsAdapter` registers an architecture
identifier, an evaluator, and an optional certificate function with an explicit
scope. Unknown architectures and unavailable certificate backends fail
explicitly. A custom evaluator is trusted application code and should validate
its own interface, tensor layout, input domain, and output shape. The registry
checks finite numerical outputs; it does not prove that the evaluator implements
the declared architecture.

`default_registry()` registers these semantic families:

| Family | Stored layout and numerical execution | Certificate scope |
| --- | --- | --- |
| `dense-relu-dense/1` | `w1[H,F]`, `b1[H]`, `w2[O,H]`, `b2[O]`; `ReLU(x @ w1.T + b1) @ w2.T + b2` | Simultaneous all-layer parameter drift over every real input in a declared finite `InputBox` |
| `sequential-affine-activation/1` | A nonempty finite chain of affine layers with changing widths; `identity`, `relu`, `abs`, or `tanh` activation after each layer | Same-input drift for shared `identity`, `relu`, and `abs` activations over a declared box; `tanh` execution has no registered certificate |

`Network` is the specialized first family. `to_tensor_artifact` and `as_relu_network` convert between its representation and the strict named-tensor interface. Their artifact hashes can differ because their serialization schemas differ. Adapter certificates retain the outer artifact identities alongside the specialized semantic receipt. Do not compare a specialized network hash with a named-tensor artifact hash as if they were interchangeable.

The sequential constructor rejects disconnected dimensions, unrecognized layers, reused layer tensors, and uninterpreted extra tensors. Its metadata fixes layer order, activation, and input/output names. `sequential_forward(..., through_layer=k)` executes a prefix of `k` layers; this is a numerical observation, not an additional certificate. No default backend interprets residual graphs, attention, convolution, normalization, recurrent state, arbitrary PyTorch code, or a transformer as a whole. Their tensors can still be stored, and a suitable frozen head can use the feature interface described below.

```python
from rlcd.weight_space import (
    InputBox, certify_sequential_drift, default_registry,
    sequential_artifact, verify_sequential_certificate,
)

base = sequential_artifact(
    weights=[[[1.0]], [[2.0]], [[-1.0]]],
    biases=[[0.0], [0.5], [0.0]],
    activations=["relu", "abs", "identity"],
    input_names=["x"], output_names=["score"],
)
candidate = base.replace(**{"layers.2.bias": [0.125]})
box = InputBox([-1.0], [1.0])
certificate = certify_sequential_drift(base, candidate, box)
assert certificate["output_bounds"] == ["1/8"]
assert verify_sequential_certificate(base, candidate, certificate)
registry = default_registry()
assert registry.evaluate(candidate, [[0.0]]).tolist() == [[-0.375]]
assert registry.certify(base, candidate, box)["certificate"] == certificate
```

For sequential layers, the checker maintains a baseline magnitude radius `r` and an endpoint drift radius `d`, starting with `r[j] = max(abs(lower[j]), abs(upper[j]))` and `d[j] = 0`:

```text
r_next[i] = abs(b[i]) + sum_j abs(W[i,j]) * r[j]
d_next[i] = abs(b_new[i] - b[i])
            + sum_j (abs(W_new[i,j] - W[i,j]) * r[j]
                     + abs(W_new[i,j]) * d[j])
```

The supported activations preserve the needed magnitude bound and are one-Lipschitz. The Python checker computes this recurrence with exact rationals lifted from stored binary64 endpoints. Absolute-value propagation loses cancellation and correlation, so bounds may be loose. Exceeding a requested budget means this recurrence did not certify it; it does not prove that a violating input exists.

## State what a real-valued certificate covers

Finite binary64 coefficients denote exact dyadic rationals. A real-valued
certificate interprets those coefficients in a mathematical network over real
inputs. An `InputBox` states the entire certified input domain; testing points
in a box does not establish that deployed inputs always remain in it. Native
dtype projection chooses which coefficients to certify. NumPy or PyTorch
subsequently evaluates them using floating operations whose accumulated error
is outside these receipts.

The Lean results in [formal.md](formal.md) prove the stated algebra and inequalities under their premises. Runtime `verify_*` functions recompute receipts and check identities; they do not prove Python, NumPy, serialization, or feature-extractor correspondence. A receipt can therefore be valid while an end-to-end deployment claim remains unestablished. Output drift is also distinct from a decision: strict argmax preservation requires a baseline winner's margin over each competitor to exceed the sum of their coordinate drift bounds. Ties, sampling, decoded text, and downstream task utility need their own arguments or measurements.

`Contract` and `check_contract` combine the two-layer ReLU bound with write-set, magnitude, and edit-count requirements. There is no general `Contract` implementation for every registered architecture: for sequential models, callers obtain the registered receipt and explicitly compare its bounds with their policy. The symmetry helpers likewise apply to the specialized two-layer ReLU semantics; positive hidden-unit rescaling/permutation and numerical alignment do not supply automatic transfer between unrelated architectures.

## Fit an edit over a frozen feature map

The protected edit in the worked example was written explicitly. When a caller
wants a numerical proposal, `compile_feature_edit` fits one from supplied
feature rows. `ReadoutHead(weight, bias=None, feature_id=...)` identifies a
captured `[outputs, features]` matrix and optional output bias. The caller is
responsible for the equation `output = feature @ weight.T + bias`. The feature
identity records that declaration; a string cannot verify an extractor. This
interface can describe a fixed backbone's final head without claiming to
certify the backbone's operators or arbitrary edits to it.

`compile_feature_edit` receives target features, desired output deltas, and optional retention features. It supports:

| Retention mode | Construction | Limitation |
| --- | --- | --- |
| `none` | Ridge least squares on target rows | No retention objective; supplied anchors can still receive a diagnostic receipt |
| `soft` | Add weighted retention residuals to the ridge objective | Trades requested target change against anchor drift |
| `nullspace` | Use an SVD-selected nullspace of the retention design | Numerical rank depends on `nullspace_rtol`; actual endpoint residuals must still be checked |

Bias is editable only with `edit_bias=True` and an explicitly named bias tensor. A full-rank retention design can leave no permitted direction, producing a recorded no-op and unmet target request. Ridge regularization can also leave target residuals. Successful fitting or zero anchor drift is not, by itself, useful editing.

The fitter creates a structured proposal, lowers it with the requested native formats, and computes diagnostics and exact anchor residuals on the actual endpoint. `certify_feature_retention` refuses changes outside the declared head. It augments feature rows with a constant coordinate for bias and computes exact endpoint residuals, not solver tolerances. If an unseen augmented feature has an explicit representation `sum_k c[k] * anchor[k]`, its real readout drift is bounded by `sum_k abs(c[k]) * residual[k]` per output. Exact zero anchor residuals imply exact preservation on that span. Large coefficients amplify nonzero residuals; an unrepresented feature component needs an additional bound. The receipt neither discovers nor certifies domain-wide span coverage.

```python
import numpy as np
from rlcd.weight_space import (
    ReadoutHead, TensorArtifact, apply_program,
    compile_feature_edit, verify_feature_retention,
)

def features(points):
    xy = np.asarray(points, dtype=np.float64)
    return np.column_stack((np.ones(len(xy)), xy[:, 0], xy[:, 1] ** 2))

base = TensorArtifact(
    "example-polynomial-head/1", {"head.weight": [[0.0, 0.0, 0.0]]},
    {"feature_id": "constant-x-y-squared/1"},
)
head = ReadoutHead("head.weight", feature_id="constant-x-y-squared/1")
anchors = features([[0.0, 0.0], [1.0, 0.0]])
compiled = compile_feature_edit(
    base, head, features([[0.0, 1.0]]), [[0.5]], anchors,
    retention="nullspace", ridge=0.01,
)
candidate = apply_program(base, compiled.program)
receipt = compiled.retention_certificate
assert receipt is not None
assert receipt["exactly_zero_on_supplied_features"]
assert verify_feature_retention(base, candidate, head, anchors, receipt)
assert not compiled.diagnostics["no_op"]
```

In this declared mathematical feature map, `(1,t,0)` is `(1-t)` times the first anchor plus `t` times the second, so the exact span premise has a direct witness along `y=0`. That reasoning is supplied by the example, not inferred by the fitter. The NumPy feature calculation is still a separate implementation boundary.

The earlier `compile_readout` computes hidden ReLU features directly from a `Network` and supports a soft retention objective and a selected hidden-coordinate basis. Both fitters perform numerical optimization. Computing their features can also execute a model. Neither fitting nor feature extraction is justified by merely calling a protocol “no-gradient”; no-fitting and no-forward restrictions must be stated and enforced separately. See the [source review](source-review.md) for why these access regimes cannot be conflated.

## Apply captured state to a PyTorch module

`snapshot_torch` captures persistent dense floating parameters and buffers from an already constructed module, along with declared architecture, native dtypes, modes, and exact alias groups. `apply_torch_program` edits a detached copy, checks tied aliases and native round trips, loads it, and resnapshots to verify the resulting artifact identity. It rejects edits that would silently round. To permit projection, first lower with the complete native dtype map and `cast="round"`, then apply the realized literal program through the bridge.

Integer, sparse, quantized, lazy, meta, nonpersistent, and extra state are outside the current bridge; distinct views of shared storage are rejected. Caller forward code, hooks, external resources, RNG, and arbitrary hidden mutable state are not certified by a state snapshot. `network_from_torch` accepts only an ordinary built-in `Sequential(Linear, ReLU, Linear)` with the documented hook/forward restrictions. It supplies the supported mathematical specialization, without establishing native floating execution equivalence. See [`torch_bridge.py`](../../../rlcd/weight_space/torch_bridge.py) for the precise checks.

## Record observations and selection before final evaluation

The same workflow represents four task kinds. Each needs its own success
criterion. A diagnosis is a statement to evaluate; an edit is a changed artifact
whose consequences must be measured.

| `TaskSpec.mode` | Parent | Submission | Caller evaluator must establish |
| --- | --- | --- | --- |
| `read` | Exact parent required | Finite JSON answer | Correctness of a claimed interpretation or reconstruction |
| `diagnose` | Exact parent required | Finite JSON answer | Correctness of a diagnosis; causal repair requires separate intervention evidence |
| `edit` | Exact parent required | Literal program or compatible complete artifact | Structural validity, preservation, and target utility appropriate to the task |
| `create` | Parent forbidden | Complete artifact of the declared architecture | Satisfaction of the requested specification and evaluation protocol |

`AccessPolicy` independently permits semantic metadata, weights, public feedback, forward calls, and activations. `weight_decimals` can expose rounded weights. Metadata can include semantic names and task constraints, so a metadata-only condition may still carry informative hints. A session's declared constraints are context, not automatically enforced predicates.

`BudgetLimits` counts tool calls, exposed weight scalars, feedback calls and submitted examples, execution calls and submitted examples, proposals, and reported input/output tokens. Forward and activation calls share the execution counters. Counts describe interface operations; callback-internal computation and provider token accounting are not locally attested. Held-out evaluation is outside proposer budgets, and its internal example count is not automatically measured.

An admitted attempt is charged before parsing/execution, and a failed admitted attempt retains its charge. Preflight or capability denials do not consume those counters, but remain in the trace. `propose_program` returns the complete realized artifact, so it requires metadata and exact weight access and charges every artifact scalar on every admitted attempt, including no-ops and failures. It is unavailable under rounded-only access. `propose_artifact` accepts caller-supplied data and does not charge it as a base-weight observation. The [workflow guide](workflow.md) gives the full accounting and failure behavior.

```python
from rlcd.weight_space import (
    AccessPolicy, BudgetLimits, ProgrammerSession, SplitSpec,
    TaskSpec, TensorArtifact, verify_trace,
)

base = TensorArtifact("example-affine/1", {"weight": [[0.5]], "bias": [0.0]})
task = TaskSpec(
    "guide-read-sign", "read", "Report the sign of weight[0,0]",
    base.architecture, base.sha256,
)
session = ProgrammerSession(
    task, base=base,
    access=AccessPolicy(metadata=False, weights=True),
    limits=BudgetLimits(
        tool_calls=2, weight_values=1, proposals=1,
        feedback_calls=0, feedback_examples=0,
        execution_calls=0, execution_examples=0,
        input_tokens=0, output_tokens=0,
    ),
    splits=(SplitSpec("test", ("guide-case",), ("guide-lineage",)),),
)

# Only this function represents the cooperative proposer.
def read_sign(tools):
    value = tools.weights("weight", [(0, 0)])[0]
    return tools.submit_answer({"sign": 1 if value > 0 else -1 if value < 0 else 0})

selected = read_sign(session)
session.seal(selected)
receipt = session.evaluate_heldout(
    lambda candidate: {"correct": candidate.answer["sign"] == 1},
    evaluator_id="guide-sign-check/1", split_name="test",
)
record = receipt.to_dict()
assert record["status"] == "succeeded" and record["result"]["correct"]
assert record["usage"]["weight_values"] == 1
assert verify_trace(record["trace"], expected_head=record["trace_head_sha256"])
```

`Candidate.sha256` identifies the task/session/proposal record; `Candidate.artifact.sha256` identifies its artifact, if any. They are different identities. Selection, feedback, and execution require the exact candidate object issued by that session. `seal(candidate)` fixes selection and closes proposer operations; `seal(None)` records abstention. Held-out invocation is allowed at most once, including a failed invocation. An abstention does not call the evaluator. Receipt status `succeeded` means that the callback returned successfully; the task's metric can still report failure.

`SplitSpec` and `validate_splits` reject overlapping declared sample and lineage identifiers. They do not infer shared ancestry, deduplicate equivalent examples, or prove independence. Choose the independence unit before the study and assign related checkpoints/tasks accordingly. The caller owns hidden examples, selection rules, evaluator versioning, and the interpretation of uncertainty.

Sessions are cooperative same-process interfaces, not security boundaries. Python code with access to private attributes, closures, files, or the network can bypass the intended observations. Public callbacks cannot reenter the session, but their hidden work and outputs still need review; held-out output leakage is not checked. A trace hash chain detects inconsistency relative to a trusted external head or receipt. It cannot authenticate a wholly replaced history by itself. Retain the trace, configuration, selected and rejected candidates, upstream proposals, exact artifacts, and an externally retained receipt when those facts matter to a claim.

`ReplayProvider` executes a bounded finite JSON action list with a fixed whitelist. It is useful for deterministic harness tests. The generic workflow has no implemented live LLM provider: it does not call a model, compile unrestricted natural language, or measure LLM programming ability. Analytic constructors, deterministic replay, and least-squares fitting must be identified as their actual proposer mechanisms. Provider-supplied token counts alone are not evidence that a model was called.

## Run the commands on a temporary fixture

This PowerShell block runs from the repository root after activating the package
environment. It creates a fresh temporary directory and leaves its records there
for inspection. It uses deterministic two-dimensional data, with no training run
or study seeds. The `lower` and `compile-features` output directories and `apply`
output files must be new. The commands retain separate lowering, application,
drift, and retention receipts.

```powershell
$wspPython = (Get-Command python).Source
$env:WSP_GUIDE_DIR = Join-Path $env:TEMP ("wsp-guide-" + [guid]::NewGuid().ToString("N"))
@'
import json
import os
from pathlib import Path
from rlcd.weight_space import StructuredProgram, sequential_artifact

root = Path(os.environ["WSP_GUIDE_DIR"])
root.mkdir()
base = sequential_artifact(
    [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]]],
    [[0.0, 0.0], [0.0]], ["relu", "identity"],
)
proposal = StructuredProgram(base.sha256, [
    {"op": "add", "tensor": "layers.1.bias", "values": [0.125]},
], "Temporary guide output shift")
head = {"weight": "layers.1.weight", "bias": "layers.1.bias",
        "feature_id": "guide-first-relu/1"}
files = {
    "base.json": base.to_dict(), "proposal.json": proposal.to_dict(),
    "native.json": {name: "float16" for name in base.tensors},
    "box.json": {"lower": [-1.0, -1.0], "upper": [1.0, 1.0]},
    "feature-fit.json": {"head": head, "target_features": [[1.0, 0.0]],
                         "desired_delta": [[0.5]], "retain_features": [[0.0, 1.0]],
                         "retention": "nullspace"},
    "retention-spec.json": {"head": head, "anchors": [[0.0, 1.0]]},
}
for name, data in files.items():
    (root / name).write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
print(root)
'@ | & $wspPython -

& $wspPython -m rlcd.weight_space inspect "$env:WSP_GUIDE_DIR/base.json"
& $wspPython -m rlcd.weight_space lower "$env:WSP_GUIDE_DIR/base.json" "$env:WSP_GUIDE_DIR/proposal.json" --native-dtypes "$env:WSP_GUIDE_DIR/native.json" --out "$env:WSP_GUIDE_DIR/lowered"
& $wspPython -m rlcd.weight_space verify-lowering "$env:WSP_GUIDE_DIR/base.json" "$env:WSP_GUIDE_DIR/proposal.json" "$env:WSP_GUIDE_DIR/lowered/lowering.json"
& $wspPython -m rlcd.weight_space apply "$env:WSP_GUIDE_DIR/base.json" "$env:WSP_GUIDE_DIR/lowered/program.json" --out "$env:WSP_GUIDE_DIR/candidate.json"
& $wspPython -m rlcd.weight_space certify "$env:WSP_GUIDE_DIR/base.json" "$env:WSP_GUIDE_DIR/candidate.json" --box "$env:WSP_GUIDE_DIR/box.json" --out "$env:WSP_GUIDE_DIR/drift.json"
& $wspPython -m rlcd.weight_space verify "$env:WSP_GUIDE_DIR/base.json" "$env:WSP_GUIDE_DIR/candidate.json" "$env:WSP_GUIDE_DIR/drift.json"
& $wspPython -m rlcd.weight_space compile-features "$env:WSP_GUIDE_DIR/base.json" "$env:WSP_GUIDE_DIR/feature-fit.json" --out "$env:WSP_GUIDE_DIR/compiled"
& $wspPython -m rlcd.weight_space apply "$env:WSP_GUIDE_DIR/base.json" "$env:WSP_GUIDE_DIR/compiled/program.json" --out "$env:WSP_GUIDE_DIR/feature-candidate.json"
& $wspPython -m rlcd.weight_space certify-retention "$env:WSP_GUIDE_DIR/base.json" "$env:WSP_GUIDE_DIR/feature-candidate.json" "$env:WSP_GUIDE_DIR/retention-spec.json" --out "$env:WSP_GUIDE_DIR/retention.json"
& $wspPython -m rlcd.weight_space verify-retention "$env:WSP_GUIDE_DIR/base.json" "$env:WSP_GUIDE_DIR/feature-candidate.json" "$env:WSP_GUIDE_DIR/retention-spec.json" "$env:WSP_GUIDE_DIR/retention.json"
& $wspPython -m rlcd.weight_space --help
& $wspPython -m rlcd.weight_space.study_v2 --help
```

The generic CLI also exposes `diff`, `compose`, `rollback`, `interpolate`, the specialized `compile`, and explicit `apply --policy` / `apply --contract` admission. Those operations have the same limits as their APIs. The `study_v2` module has its own `freeze`, `verify-freeze`, and `run` entry points; it is separate from the original study commands in the generic CLI. This walkthrough deliberately performs no study execution or source freeze.

## Extend the framework with an explicit new obligation

Add a new architecture by first declaring its tensor shapes, feature/output order, state, native format, and execution meaning. Register a caller-owned evaluator. A certificate backend needs a matching semantic proposition, explicit domain, endpoint binding, and a verifier; registering an evaluator alone leaves certification unavailable. For a fixed backbone and linear head, reuse `ReadoutHead` only after establishing its feature equation and keeping all non-head captured tensors unchanged.

Add a proposer by passing it the allowed session interface, recording every submitted candidate and provider usage, and freezing its observation/selection rules before evaluation. A live LLM adapter, provider transcript retention, and isolation from hidden evaluators remain implementation work. External process/network controls and independently audited callbacks are needed for restrictions that must resist an adversarial proposer. They are not supplied by the current Python session.

The reference format uses dense binary64 JSON and performs full tensor scans/copies. Literal programs enumerate changed coordinates; structured factors are expanded during lowering. Readout fitting uses dense SVD/least squares, and exact rational certificates can grow costly with parameter count and depth. These facilities make small artifacts inspectable; they do not establish large-model storage, runtime, or certification scalability. Sparse changes do not prove preservation, successful unlearning, or causal understanding. Use the [limitations and next steps](limitations-and-next-steps.md) and [requirements](requirements.md) to identify the additional evidence needed for a new domain or stronger claim.
