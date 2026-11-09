# Admission correspondence and native execution

Research date: November 8, 2026. This is an adjacent, executable supplement to
the [v2 precision study](../weight-space/v2/results.md). Its historical outputs
are unchanged. These examples are designed falsifiers, not a sample of
enterprise traffic; their frequency supplies no business or population estimate.

The resulting claim is narrow and useful: an integrated admission decision
identifies one immutable request, and its native decision evidence identifies
one measured execution scope. Exact real arithmetic, feature retention, native
decisions and useful behavior retain separate results.

## Evidence and reproduction

The frozen study is in
[`artifacts/research/admission-runtime/2026-11-08`](../../artifacts/research/admission-runtime/2026-11-08).
Its [`protocol.json`](../../artifacts/research/admission-runtime/2026-11-08/protocol.json)
contains all designed inputs and substitutions before gate execution.
[`results.json`](../../artifacts/research/admission-runtime/2026-11-08/results.json)
retains the full admission records, rehashed substituted records and native
receipts. Relative Python source snapshots, source hashes and a file manifest
accompany the results. The manifest's SHA-256 is
`2ffbc77abe14072883566385246e294d4a7cd2dc5abbedc5c676f296af5f1458`.

From the repository root:

```sh
python -m rlcd.weight_space.runtime_research verify artifacts/research/admission-runtime/2026-11-08
python -m pytest tests/test_research_admission_runtime.py tests/test_weight_space.py tests/test_weight_space_generic.py tests/test_weight_space_boundaries.py -q
```

The study was executed and independently replayed in separate processes. A new
run must use a new directory:

```sh
python -m rlcd.weight_space.runtime_research run artifacts/research/admission-runtime/my-new-run
```

The runner refuses to overwrite evidence, checks each manifest file, compares
current execution sources to their frozen hashes and recomputes every result.
Changing a source invalidates verification against that historical source
snapshot; a new run should record the new source identity. File hashes establish
identity. Authentication requires retaining a trusted manifest or request hash
outside the mutable evidence bundle.

## Issue #3: one request shared by every admission gate

[`AdmissionRequest` and `AdmissionRecord`](../../rlcd/weight_space/contracts.py)
freeze canonical artifact and request bytes under versioned schemas. The binding
includes parent, candidate, atomic program, task, ordered inputs and outputs,
C-order tensor coordinates, explicit post-dot bias convention, semantic adapter
version and callback descriptor, real input box, structural policy, per-output
score budgets, decision semantics, witnesses, optional feature head/anchors and
optional native execution/environment scope.

These are canonical framework artifact bytes. They do not attest to an arbitrary
external model's storage layout, extractor correctness or deployment state.
The program is reapplied to its captured parent, and the resulting candidate
must match the captured candidate's canonical bytes. The task, interface,
domain and evidence are then checked against that same request. Detached
dictionary edits cannot mutate either frozen object.

Every gate records the request hash, whether it is required, a status, a reason
and its own evidence. Status meanings are explicit:

| Status | Interpretation |
|---|---|
| `invalid` | The proposed request/application violates a binding or format premise. |
| `unsupported` | The declared semantics, operator or evidence lane is unavailable. |
| `missing_correspondence` | The available implementation or receipt does not match the frozen claim. |
| `conservative_rejection` | A sufficient bound or strict-margin condition fails; no witnessed counterexample is implied. |
| `observed_violation` | A supplied witness, native decision or structural check violates the declared rule. |
| `pass` | This gate's stated condition passed within its stated scope. |

Admission requires every required gate to pass. Real and native strict-margin
gates are optional sufficient conditions, so their conservative rejection does
not erase a directly checked witness decision. A native lane is optional when
no execution scope was requested; its `unsupported` status remains visible,
and an accepted real-only record makes no native preservation claim.

### Substitution and replay

`verify_admission(record, expected_binding_sha256)` requires an externally
trusted request hash. It reconstructs the request and independently recomputes
the complete record. A hash supplied by the record itself would allow a caller
to substitute an entirely different, internally consistent claim.

The public evidence deliberately substitutes each endpoint, program, input
domain, adapter version/configuration, input/output/tensor order, bias
convention, witnesses, feature names/anchors/head/bias, decision rule, score
budgets, structural policy, task and backend environment. Each substituted
request repairs its top-level hashes. Every substituted record fails
verification under the original trusted request hash. Each remains replayable
under its own new request hash, including records whose new claim is rejected.
A well-hashed new claim is still a new claim.

The executable checks also repair record hashes after altering a gate's
requirement, evidence or status, the aggregate acceptance bit, or a formal
claim bit. Independent recomputation rejects those changes. Mutation of a
JSON-compatible callback closure or the bound backend environment produces
`missing_correspondence`. A callback fingerprint is correspondence accounting;
it excludes mutable Python globals, sandboxing and proof of semantic correctness.

### Frozen boundary results

| Designed request | Outcome and reason |
|---|---|
| Valid edit | Accepted; captured candidate, bounds, witness scores/decisions and requested native scope replay. |
| No-op | Accepted in its requested real-only scope; native lanes remain unsupported. |
| Stale program | Rejected as invalid because its parent identity differs. |
| Witness budget violation | Rejected with observed score and feature violations; the box bound separately rejects conservatively. |
| Conservative box bound | Rejected despite zero supplied-anchor residual and matching witness outputs. |
| Stochastic decision | Rejected as unsupported; it is never silently interpreted as argmax. |
| Unregistered semantics | Rejected as unsupported. |
| Tied unique argmax | Rejected by the declared unique-winner rule. |
| Tied first/last index argmax | Accepted under the explicitly declared stable tie rule; strict-margin tests remain conservative. |
| Real checks pass, native decision changes | Rejected by the requested native decision lane. |

For the conservative example, the parent first score is `x1 + x2 + 2`, the
candidate first score is `2*x1 + 2`, and the witness is `(0.5, 0.5)`. Both witness
scores are 3. The requested zero drift over `[-1, 1]^2` cannot be certified by
the positive triangle bound. The zero witness residual is not an all-input
retention guarantee, and rejection at the witness is not an observed violation.

Unique argmax, first-index ties, last-index ties and scores-only requests have
separate semantics. Nonfinite scores violate even a scores-only request.
Stochastic sampling has no deterministic replay model here.

## Issue #4: arithmetic and decisions share a specified execution scope

[`ExecutionSpec`](../../rlcd/weight_space/structured.py) declares backend,
dtype, device, accumulation order, endpoint-cast policy and empirical/formal
error lane. The operator model performs dot product, explicit bias addition
and activation. Serial arms use separate multiply and add operations in the
specified order. The `matmul` arm measures the installed backend's implementation
without claiming a portable reduction order or absence of internal fusion.

Each receipt retains original and projected endpoints and hashes, original and
projected witnesses, exact outputs at each stage, native outputs, per-coordinate
errors and decision effects. Projecting the parent changes its baseline and is
always reported. `endpoint_cast="reject"` rejects a changed projected endpoint;
`"round"` explicitly permits and identifies it. Input rounding is recorded
separately.

For each finite output coordinate, the receipt records the signed telescoping
decomposition:

```text
endpoint error     = real(projected endpoint, original input) - real(stored endpoint, original input)
input error        = real(projected endpoint, projected input) - real(projected endpoint, original input)
intermediate error = native(projected endpoint, projected input) - real(projected endpoint, projected input)
total error        = endpoint error + input error + intermediate error
```

Exact outputs use the rational values of stored binary floating-point numbers
through supported affine, identity, ReLU and absolute-value paths. This
calculation supplies a mathematical reference; it does not claim to execute
the backend kernel. The measured intermediate error includes all arithmetic
and activation effects after endpoint/input projection.

### Cancellation with an exactly zero anchor residual

At witness `(1, 1, 1)`, compare first-row weights `(L, -L, 1)` with
`(L, 1, -L)`. The second score is the constant `0.5`. Both exact real first
scores are 1, and the affine-anchor residual is exactly zero. `L` is `2^54`
for float64, `2^25` for float32, `2^12` for float16 and `2^9` for bfloat16.

The following are measurements from this frozen CPU environment:

| Backend and dtype | Left-to-right first scores (parent, candidate) | Right-to-left | Installed `matmul` |
|---|---|---|---|
| NumPy float64 | `(1, 0)` | `(0, 0)` | `(1, 0)` |
| NumPy float32 | `(1, 0)` | `(0, 0)` | `(1, 0)` |
| NumPy float16 | `(1, 0)` | `(0, 0)` | `(1, 1)` |
| Torch float64 | `(1, 0)` | `(0, 0)` | `(1, 0)` |
| Torch float32 | `(1, 0)` | `(0, 0)` | `(1, 0)` |
| Torch float16 | `(1, 0)` | `(0, 0)` | `(1, 1)` |
| Torch bfloat16 | `(1, 0)` | `(0, 0)` | `(1, 1)` |
| NumPy bfloat16 | Unsupported | Unsupported | Unsupported |

Every `(1, 0)` pair changes the winner between endpoints. Every `(0, 0)`
pair preserves the native winner but disagrees with the real winner for both
endpoints. Native preservation therefore does not establish real-to-native
decision correspondence. The tests impose the declared serial effects and
replay the measured `matmul` results without requiring these `matmul` values
on every machine.

### Near-tie endpoint rounding blocks admission

With zero weights, the stored parent biases are
`(1 + 2^-25, 1 + 2^-24)` and the candidate biases are
`(1 + 2^-25, 1 + 2^-23)`. Both real winners are coordinate 1. Under
float32 endpoint rounding and first-index ties, the native parent becomes
`(1, 1)` with winner 0, while the candidate becomes `(1, 1 + 2^-23)` with
winner 1. The captured real box drift, real witness scores/decisions and feature
budget pass. The native decision gate reports an observed violation and rejects
admission. Input and intermediate errors are zero: endpoint quantization alone
causes the failure.

The optional native margin condition uses the real drift allowance plus the
absolute total arithmetic errors of both endpoints on both competing
coordinates. The near-tie record retains a zero native-parent margin and total
allowance `3/16777216`, so the strict sufficient condition rejects. Arithmetic
errors from only the candidate would be insufficient. These are measured
witness allowances; they do not bound unobserved inputs.

A native no-op on the rounded parent preserves its native decision and reports
`no_op=true`, while its real-to-native decision effect still reports a
violation. Unsupported and invalid no-op receipts also retain `no_op` consistently.

### Coverage and unsupported cases

Recorded environment: Python 3.12.10, NumPy 2.5.3, Torch 2.8.0+cpu,
Windows/AMD64 CPU. Torch used 8 intra-operation and 8 inter-operation threads,
`float32_matmul_precision="highest"` and deterministic algorithms disabled.
Thread environment variables and backend flags are retained. The NumPy build
configuration is hashed instead of publishing installation paths.

| Scope | Recorded disposition |
|---|---|
| CPU NumPy float64/float32/float16 | Executed for identity arithmetic and the declared serial/`matmul` arms. |
| CPU Torch float64/float32/float16/bfloat16 | Executed for the same arms. |
| Finite float16 endpoints with intermediate overflow | Native `+Infinity` is retained as JSON-safe evidence; the receipt reports an observed violation and no finite total-error budget. |
| Input-only float32 rounding | Input error is retained while endpoint and intermediate errors remain zero. |
| Tanh | Native outputs are retained; exact error accounting and the combined retention lane are unsupported. |
| Unregistered operator model | Unsupported. |
| NumPy bfloat16 | Unsupported. |
| CUDA | Unavailable in this environment and explicitly unsupported. |
| Formal native lane | Unsupported: no rounding/domain proof backend is registered. |

This operator model supports affine/identity/ReLU/absolute-value exact
accounting; the frozen cancellation and boundary experiments exercise affine
identity cases and retain a tanh exclusion. It does not cover convolutions,
attention, sparse/custom kernels, arbitrary layouts, integer quantization,
compiler transforms or deployment hardware beyond the recorded model.

PyTorch's primary numerical-accuracy documentation explains why mathematically
equivalent reductions and backend/device choices need not produce identical
floating-point outputs. The
[v2.8 source](https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/numerical_accuracy.rst)
describes that numerical boundary; the installed kernel behavior above is
established by the retained measurements, not inferred from documentation.
The [research register's TorchLean scope review](../weight-space/v2/research-roadmap-2026-11.md#s12-torchlean-formalizing-neural-networks-in-lean)
motivates keeping mathematical semantics and executed hardware correspondence
separate.

## Acceptance disposition and next boundary

Issue #3's deliverables and acceptance cases are satisfied within the canonical
artifact/caller-owned adapter scope: immutable request binding, separate gate
statuses, declared decisions, repaired-hash substitutions and independent
replay are implemented and recorded. The falsifier is an altered claim passing
under the original externally trusted request hash. None did in the designed
study. This is not a proof of all possible Python implementations or extractors.

Issue #4's empirical deliverables and acceptance cases are satisfied within the
recorded CPU operator model: cancellation/order, endpoint/input/intermediate
error accounting, tie margins, no-op/baseline reporting, overflow and unsupported
coverage are recorded. A violated requested native rule blocks admission.
The formal lane remains explicitly unavailable; future work must specify
rounding, operation order/fusion, finite-domain/overflow premises and a total
decision-error budget before attaching a formal native claim.

The next substantive experiment is cumulative admission: compare each new
endpoint with its original model and all earlier target requests, rather than
letting local checks reset retention budgets after every accepted step.
Trained-use-case utility and sequence uncertainty belong in that separate
evaluation. Neither these designed examples nor their runnable-check count
establishes business impact.
