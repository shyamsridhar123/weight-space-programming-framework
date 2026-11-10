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

The [staged diagnosis guide](../research/staged-diagnosis.md#registered-cohort-evaluation)
describes the separate generic cohort evaluation path. This adapter supplies no
cohort integration or automatic runtime receipt binding. Generic cohort
`runtime_restoration_verified` and `causal_usefulness_established` flags remain
`false`; their receipt schemas and earlier evidence scopes are unchanged.
Issue #9 c1 remains partial, and the historical diagnosis/repair result is
unchanged. The separate
[engineering receipt](../../provenance/verification/sequential-runtime-restoration-20261109.json)
records one focused synthetic regression capture at its captured source snapshot:
`63 passed, 3 warnings in 1.47s`. Counts and elapsed time are bookkeeping, not
impact or performance evidence. The three warnings concern NumPy 2.5 array-header
mutation deprecations triggered by the isolation fixtures. This supplies
engineering coverage of the owned sequential adapter and adds no scientific
result. The standalone source example above remains unexecuted; later publication
edits retain separate validation scope.
