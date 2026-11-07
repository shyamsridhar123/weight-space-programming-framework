# Weight-space programming framework

> **Historical framework entry point.** The examples below belong to the original
> RLCD documentation. Begin with the [current overview](v2/README.md) for the task
> and evaluation workflow. Use the [standalone reproduction guide](../../REPRODUCING.md)
> for installation and current commands. The [original guide](../../provenance/original-documents/docs/weight-space/README.md.txt)
> is preserved separately.

This is a generic framework for **reading, diagnosing, editing, and creating
neural-network weights**. Its core represents immutable named tensors, checks
parent identities and edit preconditions, applies atomic programs, composes
programs, and constructs exact rollbacks. Execution, proposal generation,
behavioral verification, and task evaluation use explicit adapters.

**Start with the [comprehensive current framework guide](v2/README.md).** It builds
on a fresh review of the original report, its future directions and release
caveats, and the limitations of this repository's first version. RTS is one
historical application. The current generic study uses ReLU MLP, tanh MLP, and
convolution-plus-tanh models, together with independent read/diagnose/create
controls, without a game or provider dependency.

| Purpose | Current document |
| --- | --- |
| Understand original evidence, limitations, and source omissions | [Source review](v2/source-review.md) |
| Trace implementation obligations and open scope | [60 requirements](v2/requirements.md) |
| Use structured edits, feature compilers, native projection, and task sessions | [Architecture and runnable examples](v2/architecture.md), [workflow](v2/workflow.md) |
| Inspect generic-feature and arbitrary-depth real-valued guarantees | [Formal extension and verification](v2/formal.md) |
| Assess the 12-world, 36-checkpoint study and its independent audit | [Methods](v2/methods.md), [results](v2/results.md) |
| Plan the next implementation and experiments | [Limitations and 11-part research program](v2/limitations-and-next-steps.md) |

The original API examples below remain useful. The earlier [design](design.md),
[research review](research.md), [results](results.md), and
[completion audit](requirements.md#completion-audit) describe version one; their
test counts, study conclusions, and proof scope are historical. The
[project memory](../PROJECT_MEMORY.md) keeps both versions' decisions and evidence
distinct.

## Run the standalone examples

From the repository root, using the existing Python environment:

```powershell
.venv/Scripts/python.exe -m rlcd.weight_space --help
.venv/Scripts/python.exe -m rlcd.weight_space demo --out artifacts/weight-space/my-demo
```

Choose a new output directory. Evidence commands refuse to overwrite prior files.
The demo needs NumPy and has no game or provider dependency. It writes:

- A convolution-shaped tensor/Tanh scorer with a scalar bias, a structural
  policy, literal edit program, before/after outputs, and exact rollback.
- A ReLU model, behavioral example specification, compiled readout program,
  and exact rational drift receipts. Its edit has zero real-valued effect on
  the specified nonpositive input region.

The Tanh example reports `behavioral_status: "not_checked"`. Successful tensor
editing alone makes no claim about model behavior. The ReLU receipts state a
bounded real-score guarantee and explicitly exclude floating execution and task
reward. The checked-in example evidence is in
[`artifacts/weight-space/demo-v1`](../../artifacts/weight-space/demo-v1/).

## Use arbitrary named tensors

```python
import numpy as np
from rlcd.weight_space import (
    TensorArtifact, StructuralPolicy, SemanticsAdapter, SemanticsRegistry,
    difference, apply_program, rollback,
)

base = TensorArtifact(
    "affine-scalar/1",
    {"weight": np.array([1.0, -0.5]), "bias": np.array(0.0)},
    {"input_names": ["temperature", "pressure"], "output_names": ["score"]},
)
proposal = base.replace(bias=np.array(0.25))
program = difference(base, proposal, "Raise the scalar score by 0.25")
policy = StructuralPolicy(
    writable=("bias",), max_edits=1,
    max_parameter_abs=8.0, max_change_abs=0.25,
)
receipt = policy.check(base, program)
assert receipt["structural_accepted"]
candidate = apply_program(base, program)
undo = rollback(base, program)
assert apply_program(candidate, undo).sha256 == base.sha256

registry = SemanticsRegistry()
registry.register(SemanticsAdapter(
    "affine-scalar/1",
    lambda artifact, x: (
        np.asarray(x) @ artifact.tensors["weight"]
        + float(artifact.tensors["bias"])
    )[..., None],
))
print(registry.evaluate(candidate, [[0.0, 0.0], [1.0, 2.0]]))
```

Tensor names and ranks are arbitrary, including rank-zero scalars. The reference
storage format contains finite binary64 values; it does not support arbitrary
dtypes or integer-only model state. Non-exact integer conversions are rejected.
Architecture, shapes, names, and JSON interface metadata participate in identity.
Returned arrays do not expose mutable model storage. A program binds its exact
parent and checks every old value before producing a new artifact.

The caller supplies execution code through `SemanticsRegistry`. Artifact JSON
cannot select a Python import or executable module. Register a separate
certificate backend when adding an architecture; requesting a certificate
without one fails. A custom backend remains the caller's responsibility and is
not automatically covered by this project's Lean proofs.

## Edit an existing PyTorch module

```python
import torch
from torch import nn
from rlcd.weight_space import difference
from rlcd.weight_space.torch_bridge import snapshot_torch, apply_torch_program

model = nn.Sequential(
    nn.Linear(4, 8, dtype=torch.float64), nn.Tanh(),
    nn.Linear(8, 2, dtype=torch.float64),
).eval()
metadata = {"input_order": ["a", "b", "c", "d"], "output_order": ["left", "right"]}
base = snapshot_torch(model, architecture="my-tanh-model/1", metadata=metadata)
bias = base.tensors["2.bias"].copy()
bias[0] += 0.125
program = difference(base, base.replace(**{"2.bias": bias}), "Adjust first output bias")
edited_model = apply_torch_program(
    model, program, architecture="my-tanh-model/1", metadata=metadata,
)
```

The bridge edits a copy and retains native tensor names, shapes, dtypes, devices,
module modes, and gradient flags. It captures floating persistent buffers and
checks tied aliases. Native restoration must be exact: an edit that would round
when restored to float32, float16, or bfloat16 is rejected. Integer buffers,
nonpersistent state, extra state, sparse or quantized tensors, meta/lazy tensors,
and incompatible views of shared storage are explicitly unsupported.

The caller identifies the forward semantics and non-state configuration. The
snapshot does not certify arbitrary Python forward code or capture the global
random state. `network_from_torch` separately extracts the supported ordinary
`Sequential(Linear, ReLU, Linear)` family with explicit input/output names; it
rejects custom forwards and local execution hooks.

## Inspect, edit, compile, and check from the CLI

The CLI accepts `wsp-tensors/1` and `wsp-network/1` JSON. A `wsp-program/1` names
the exact parent and literal coordinate endpoints. Use `diff` to construct a
program from compatible artifacts:

```powershell
.venv/Scripts/python.exe -m rlcd.weight_space inspect artifacts/weight-space/my-demo/generic/base.json
.venv/Scripts/python.exe -m rlcd.weight_space diff base.json proposed.json --intent "Adjust readout" --out program.json
.venv/Scripts/python.exe -m rlcd.weight_space apply base.json program.json --policy policy.json --out edited.json
.venv/Scripts/python.exe -m rlcd.weight_space rollback base.json program.json --out undo.json
.venv/Scripts/python.exe -m rlcd.weight_space apply edited.json undo.json --out restored.json
```

`apply` writes a new artifact and an adjacent `.receipt.json`. A structural
policy can restrict writable tensor names, edit count, parameter magnitude, and
maximum coordinate change. Add `--contract contract.json` for supported ReLU
behavioral admission. Rejected programs are not applied by this command.

`compose BASE PROGRAMS... --out FILE` checks each intermediate parent before
constructing the combined program. `interpolate BASE CANDIDATE --alpha 0.5
--out FILE` constructs a coordinate interpolation proposal; interpolate and
generic composition still need appropriate behavioral checks.

For the supported ReLU family, compilation fits a requested output change using
the existing hidden features. The specification provides `target_inputs`,
`desired_delta`, optional `retain_inputs`, `retention_weight`, positive `ridge`,
optional `editable_hidden`, and optional `edit_bias`. It solves a regularized
least-squares problem and never consumes evaluation inputs internally.

```powershell
.venv/Scripts/python.exe -m rlcd.weight_space compile artifacts/weight-space/my-demo/relu/base.json artifacts/weight-space/my-demo/relu/spec.json --contract artifacts/weight-space/my-demo/relu/contract.json --out artifacts/weight-space/my-compiled
```

Compilation retains the proposed program, fit diagnostics, and any admission
result. A rejected proposal is retained as evidence. Applying it is a separate
operation. The compiler does not synthesize arbitrary architectures or convert
unrestricted natural language into a behavioral theorem.

The standalone certificate commands are:

```powershell
.venv/Scripts/python.exe -m rlcd.weight_space certify base.json edited.json --box box.json --out certificate.json
.venv/Scripts/python.exe -m rlcd.weight_space verify base.json edited.json certificate.json
```

A box is `{"lower": [-1.0], "upper": [1.0]}`, with one endpoint per input.
A contract adds `max_output_drift` per output coordinate, `allowed_tensors`,
`max_parameter_abs`, and `max_edits`; the demo writes complete examples. The
checker lifts serialized binary64 endpoints to exact rationals and recomputes
the all-layer bound. It rejects altered receipts and incompatible semantics.

## Understand the guarantees

| Layer | What acceptance establishes | Additional obligation |
| --- | --- | --- |
| Generic program | Exact parent, valid coordinates/old values, finite atomic result | Meaning of architecture and outputs |
| Structural policy | Declared writable set and numeric parameter limits | Behavioral effect |
| ReLU drift receipt | Exact rational evaluation of a real-score bound on an input box | Correct exporter/checker and floating execution |
| Lean theorem | Universal real-valued bound, symmetry, or composition under stated assumptions | Connection from external artifacts to explicit definitions |
| Environment evaluation | Observed results for recorded units and protocol | Other models, datasets, environments, or horizons |

The first version's proved semantic family is a finite two-layer ReLU network. All-layer
edits, positive rescaling, hidden permutation, edit transport, sequential drift
composition, and strict argmax preservation are covered by the formal library.
The vector-output runtime applies the scalar theorem per coordinate. The strict
argmax result requires a sufficient margin; a small score bound alone does not
preserve a selected action.

The [current formal extension](v2/formal.md) adds arbitrary frozen feature maps,
exact and approximate span-retention results, infeasibility and factored-edit
results, and drift bounds for arbitrary finite-depth, variable-width sequential
networks. Supported activation assumptions remain explicit. New runtime
candidates are not automatically bound to generated Lean instances; generic
behavioral admission and native-execution correspondence remain open work.

Known permutations and positive ReLU rescalings are real-function symmetries.
Incoming-weight alignment between independently trained networks is a heuristic.
`transfer_edit` returns a new proposal that requires fresh admission and
evaluation. Equivalent real functions can still differ numerically when their
floating-point accumulation order changes.

This implementation favors auditability: dense binary64 JSON, full snapshots,
exact rational arithmetic, and cubic hidden-unit assignment. It is a small-model
research reference, with substantial time and memory limits at transformer scale.
An architecture-independent tensor interface does not remove those limits.

## Reproduce the original research and proofs

This section reproduces **version one**. For the completed second study, use the
[current methods](v2/methods.md), [audit and rendering commands](v2/results.md#independent-audit-and-verification),
and [generic proof commands](v2/formal.md). All published seeds in both studies are
observed; fresh confirmation needs a new protocol and untouched units.

The original fixed protocol is [`experiments/weight-space/protocol.json`](../../experiments/weight-space/protocol.json).
Its primary experiment uses independently trained three-output regression
networks. The RTS study is an environment adapter with its own legal-action
semantics, fixtures, validation selection, and held-out worlds. Neither study
requires new provider calls.

The [original completed report](../../artifacts/weight-space/report-v2/report.md) supports
the fixed H1/H2 rules on this regression study; H3 is unsupported because validation
selected the unchanged RTS baseline. The [results guide](results.md) separates
numerical outcomes from admission and explains the archived study identity and
subsequent CLI receipt repair. The directory name `report-v2` denotes a chart
revision of that first study. The second study's report is
[`study-v2-report-complete`](../../artifacts/weight-space/study-v2-report-complete/report.md).

Run the [reproduction script](../../scripts/run_weight_space_study.ps1) with a new
output directory after installing the repository's locked dependencies. It
freezes sources before executing any units, retains every arm and failure, and
selects the RTS candidate before opening its test split. The result report links
the exact frozen inputs, raw records, admission decisions, and proof evidence.

```powershell
pwsh -File scripts/run_weight_space_study.ps1 -Out artifacts/weight-space/my-study
pwsh -File scripts/check_weight_space_proofs.ps1 -Bootstrap
.venv/Scripts/python.exe -m pytest tests/test_weight_space.py tests/test_weight_space_generic.py tests/test_weight_space_torch.py tests/test_weight_space_boundaries.py -q
```

The proof bootstrap pins Lean 4.24.0 and Mathlib, verifies the toolchain archive,
builds the library, and audits every `WeightSpace` declaration. An initialized
installation can use `-SkipCache`. No Python certificate receipt claims that
the Python implementation, the optimizer, or target hardware has been verified.
