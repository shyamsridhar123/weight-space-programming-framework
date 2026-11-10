# Owned sequential runtime

[`restore_sequential_runtime`](../../rlcd/weight_space/sequential_runtime.py)
constructs an owned runtime for one declared `sequential-affine-activation/1`
`TensorArtifact`. The caller supplies the exact retained endpoint SHA-256 and an
explicit execution declaration. Construction checks both the supplied artifact
and a fresh snapshot of the runtime's owned state against that hash.

The retained hash needs an independent trust basis: keep it in the trusted harness
before accepting an endpoint for restoration. Computing a new hash from an
untrusted replacement only checks that replacement's self-consistency. Hash
agreement identifies the declared artifact; it does not authenticate training,
ancestry or the identity of an original Python module.

The runtime owns independent tensor and metadata state. `snapshot()` returns a
detached `TensorArtifact`; `receipt` returns a detached JSON object. Changing
caller arrays, returned array headers, snapshots, receipts or outputs does not
change the runtime. The restoration receipt has schema
`wsp-sequential-runtime-restoration/1` and records endpoint/snapshot hashes,
ordered layers, shapes, labels, execution settings, numerical environment and an
empty alias declaration. It describes this owned restoration, with no task score
or forward-call result recorded in it.

All six execution fields are fixed:

```python
{
    "backend": "numpy",
    "dtype": "float64",
    "device": "cpu",
    "accumulation": "left_to_right",
    "endpoint_cast": "reject",
    "error_model": "empirical",
}
```

Supply either an exact `ExecutionSpec` or a plain `dict` with exactly these plain
string keys and values. Missing, extra, implicit or different settings are
rejected, as are string subclasses and declaration subclasses. The endpoint must
be an exact `TensorArtifact` with plain architecture and tensor-name strings.
Its metadata contains only `layers`, `input_names`
and `output_names`; each ordered layer contains only `weight`, `bias` and
`activation`. Every tensor must be used exactly once, layer widths must connect,
and labels must be unique and match the declared dimensions. Native dtype,
device, alias and arbitrary capture metadata are outside this adapter's scope.

Supported activations are `identity`, `relu` and `abs`. For each affine output,
execution multiplies and adds coordinates from left to right, adds the bias
after the completed dot product, then applies the activation. This ordering is
part of the supported scope. Overflow and invalid arithmetic are rejected before
activation, including negative overflow that ReLU could otherwise hide. Finite
underflow results are allowed under NumPy's recorded environment.

`forward` accepts a finite input vector or a nonempty two-dimensional batch with
the declared input width, and preserves vector/batch output shape. Input
conversion must be exact binary64; nonfinite values, booleans, strings and
inexact integer conversions are rejected.

This standalone source example uses handcrafted tensors. It has not been run as
part of preparing this guide; the output comments are hand calculations.

```python
from rlcd.weight_space.sequential import sequential_artifact
from rlcd.weight_space.sequential_runtime import restore_sequential_runtime
from rlcd.weight_space.structured import ExecutionSpec

endpoint = sequential_artifact(
    weights=[[[1.0, -2.0]]],
    biases=[[0.5]],
    activations=["abs"],
    input_names=("left", "right"),
    output_names=("magnitude",),
)
execution = ExecutionSpec.from_dict({
    "backend": "numpy", "dtype": "float64", "device": "cpu",
    "accumulation": "left_to_right", "endpoint_cast": "reject",
    "error_model": "empirical",
})

# Self-contained illustration; a real harness supplies its independently retained hash.
retained_sha256 = endpoint.sha256
runtime = restore_sequential_runtime(
    endpoint, expected_endpoint_sha256=retained_sha256, execution=execution,
)
vector = runtime.forward([1.0, -1.0])                # Expected [3.5], shape (1,).
batch = runtime.forward([[1.0, -1.0], [0.0, 1.0]]) # Expected [[3.5], [1.5]].
snapshot = runtime.snapshot()
restoration_receipt = runtime.receipt
```

The adapter restores the declared sequential artifact. Arbitrary original Torch
modules, aliases, buffers, caches, random state and deployment effects are outside
its contract. Its empirical floating-point execution supplies neither a formal
floating-point proof nor evidence of useful or causal repair.

## Registered sequential slot evaluation

[`evaluate_sequential_slot`](../../rlcd/weight_space/sequential_slot_evaluator.py)
is an optional trusted evaluator for the
[registered cohort bridge](../research/staged-diagnosis.md#registered-cohort-evaluation).
Freeze every applicable slot with exactly `result_keys: ["sequential"]` and the
six-field execution declaration above before constructing the cohort. For example,
this selected slot uses the earlier `execution` object:

```python
slot = {
    "slot_id": "repaired",
    "source": {"kind": "selected", "fallback_control_id": None},
    "execution": execution.to_dict(),
    "result_keys": ["sequential"],
}
```

After the separate diagnosis and repair passes, complete repair barrier and
endpoint seal, supply the helper directly:

```python
from rlcd.weight_space.sequential_slot_evaluator import evaluate_sequential_slot

evaluation_receipt = cohort.evaluate_registered(load_holdouts, evaluate_sequential_slot)
```

Each supplied row is a plain JSON object containing an `inputs` object with
exactly the runtime's input names. Other finite JSON row fields are allowed and
remain part of the full-row commitment. The helper validates the context and
every row as plain finite JSON, restores against
`context["slot"]["endpoint_sha256"]`, and passes
`context["slot"]["execution"]` unchanged to the runtime. That endpoint hash comes
from the cohort's verified retained seal; the helper does not derive its expected
hash from the supplied endpoint. A caller outside the cohort must establish its
own independent trust basis for the context.

The helper orders inputs by `runtime.input_names`, converts each value exactly
with `finite_binary64`, and prepares all rows before one batched `forward` call.
Booleans, strings, nonfinite or inexact numeric inputs, missing or extra input
names, tuples, NumPy values and subclasses received at this boundary are rejected.
Earlier providers or cohort APIs may already have normalized some types into
JSON; this helper does not attest those pre-normalization types. The generic
cohort examples' `accumulation: "matmul"` declaration is unsupported here and
fails; it is never silently changed to `left_to_right`.

The sole result key `sequential` contains schema
`wsp-sequential-slot-evaluation/1`, the detached `restoration` receipt,
`rows_sha256` for the complete ordered rows, `ordered_inputs_sha256` for
`{"input_names": [...], "inputs": [...]}` using the normalized binary64 floats,
and plain JSON `predictions` in `[row][output]` shape, including a one-row batch.
The existing cohort slot, terminal and durable registry result commitments retain
this nested result. Errors propagate through the bridge's existing failure path:
later slots stay unrun and committed scoring authorization remains consumed.

The [authored integration check](../../tests/test_sequential_slot_evaluator.py)
uses two literal layers with
`score = 2 * relu(left) - relu(right) + 3`. Its separate diagnosis names the
authored `layers.1.bias` offset and predicts that subtracting three from that bias
lowers every output by three. The repair selects that literal edit before sealing
repaired and parent-control slots. These are hand calculations, not study results:

| Inputs `[left, right]` | Parent | Repaired |
| --- | --- | --- |
| `[1, 2]` | 3 | 0 |
| `[3, 1]` | 8 | 5 |
| `[-1, 2]` | 1 | -2 |

This integration source and its authored check have not been executed as part of
this change. They add no validation capture or scientific result. The helper
establishes no automatic diagnosis, trained-model benefit, original-module
restoration or evaluator attestation. Generic cohort
`runtime_restoration_verified` and `causal_usefulness_established` flags remain
`false`; their schemas and earlier evidence scopes are unchanged. Issue #9 c1
remains partial and open, and the historical diagnosis/repair result is unchanged.
