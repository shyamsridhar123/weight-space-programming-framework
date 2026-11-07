# Change one output and preserve another

An edit can satisfy its target and still damage another behavior. This example
shows why. It also shows which checks the framework supplies and which conclusions
need more evidence.

The model is deliberately small. The arithmetic is an **illustrative exact case**,
not an observation from the study or a trial of a language model. Its mechanism
applies to a linear output head over any declared, unchanged feature map. That
feature map could come from a neural network, an embedding, or an analytic function.

## The request

Let the output be a dot product:

\[
y(\phi)=w\cdot\phi,\qquad w=(1,1).
\]

We want to increase the output at target feature \(t=(1,1)\) by one. We also want
to preserve the output at protected feature \(p=(1,0)\).

| Input feature | Parent output | Requested output |
| --- | ---: | ---: |
| Target \(t=(1,1)\) | 2 | 3 |
| Protected \(p=(1,0)\) | 1 | 1 |

The two outputs share the first coefficient. Changing that coefficient affects
both inputs. Calling the change a repair does not remove this coupling.

## A target-only edit

Write the parameter update as \(\Delta=(a,b)\). Meeting the target requires
\(a+b=1\). Among exact solutions, the smallest Euclidean update is
\(\Delta=(1/2,1/2)\), because

\[
a^2+(1-a)^2=2(a-1/2)^2+1/2.
\]

The new weights are \((1.5,1.5)\). The target output becomes 3. The protected
output becomes 1.5. The edit has achieved the target and introduced collateral
change of 0.5 in the same output units.

This is the minimum-norm **exact** solution to this toy request. It is not a claim
about the repository's ridge solver, which solves a regularized objective.

## A protected edit

Exact protection requires \(\Delta\cdot p=a=0\). Combining that condition with
\(a+b=1\) gives \(\Delta=(0,1)\). The new weights are \((1,2)\).

| Proposal | Actual weights | Target output | Protected output | Interpretation |
| --- | --- | ---: | ---: | --- |
| Keep the parent | \((1,1)\) | 2 | 1 | Preserves behavior but misses the requested change |
| Target-only exact edit | \((1.5,1.5)\) | 3 | 1.5 | Meets the target with collateral change |
| Protected exact edit | \((1,2)\) | 3 | 1 | Meets both conditions in this example |

The protected edit has a larger parameter norm. Parameter distance alone would
prefer the exact target-only edit even though it violates the protected behavior. The task
must say which outputs matter and what change is allowed.

Open the [interactive illustration](../../../artifacts/explanations/edit-retention/index.html)
to move the update between these choices. It also shows the infeasible cases
below. The table and equations here provide the complete static reading path.

## Carry the same example through the API

The framework stores the parent, turns the requested endpoint into a literal
program, and applies that program only to the exact parent. The following code
uses the protected edit. It does not ask a solver to discover it.

```python
from rlcd.weight_space import (
    ReadoutHead, TensorArtifact, apply_program, certify_feature_retention,
    difference, rollback, verify_feature_retention,
)

parent = TensorArtifact(
    "illustrative-linear-head/1", {"head.weight": [[1.0, 1.0]]},
    {"feature_id": "two-declared-features/1"},
)
requested = parent.replace(**{"head.weight": [[1.0, 2.0]]})
program = difference(parent, requested, "Raise the target; preserve (1, 0)")
candidate = apply_program(parent, program)
head = ReadoutHead("head.weight", feature_id="two-declared-features/1")
anchors = [[1.0, 0.0]]
receipt = certify_feature_retention(parent, candidate, head, anchors)

assert candidate.sha256 == requested.sha256
assert receipt["maximum_residuals_exact"] == ["0"]
assert verify_feature_retention(parent, candidate, head, anchors, receipt)
assert apply_program(candidate, rollback(parent, program)).sha256 == parent.sha256
```

These checks establish different facts:

1. `difference` records the actual changed coefficient, from 1 to 2.
2. `apply_program` checks the parent identity and stored before-value. It creates
   a new artifact after validating the complete edit.
3. The retention receipt interprets the stored endpoints and supplied feature
   values as exact rationals. Their dot-product difference at \((1,0)\) is zero.
4. `verify_feature_retention` recomputes that receipt. It is executable checking;
   it does not invoke Lean or verify the Python implementation.
5. `rollback` restores the parent artifact identity. It does not reverse effects
   that a deployed system may already have caused.

The architecture name and feature ID are declarations. The code above does not
prove that a separate application computes those features, uses that output
equation, or remains within a specified input domain.

Run the complete example after [installing the package](../../../README.md#install-and-run-the-example):

```bash
python examples/edit_retention.py
```

The [source](../../../examples/edit_retention.py) checks all three exact cases,
an actual numerical compilation, and a full-protection no-op. Its output reports
the target change and protected residual for each case. Those are the quantities
to inspect; successful execution alone is not the behavioral conclusion.

## What the numerical compiler actually does

`compile_feature_edit` receives feature rows and desired output changes. It fits
an update with positive ridge regularization. For this one target, nullspace
protection enforces \(a=0\), and the ideal real-valued fitting problem becomes

\[
\min_b\ (b-1)^2+\lambda b^2,
\qquad b=\frac{1}{1+\lambda}.
\]

With `ridge=0.01`, this ideal update is approximately 0.990099. It preserves the
protected output but falls short of the target change of one. The implementation
uses numerical linear algebra, realizes a stored endpoint, and then checks that
endpoint. It need not store the ideal rational minimizer exactly.

```python
from rlcd.weight_space import compile_feature_edit

compiled = compile_feature_edit(
    parent, head,
    target_features=[[1.0, 1.0]], desired_delta=[[1.0]],
    retain_features=[[1.0, 0.0]], retention="nullspace", ridge=0.01,
)
realized = apply_program(parent, compiled.program)
assert compiled.retention_certificate["exactly_zero_on_supplied_features"]
assert compiled.diagnostics["target_rmse_after_projection"] > 0
```

In general, a numerically chosen nullspace need not give exact zero residual
after fitting and storage. This coordinate-aligned example happens to preserve
the anchor exactly. The generic study contains small **nonzero** realized
residuals. A successful receipt verifies the residual it records, including a
nonzero one. It does not turn a tolerance into an equality.

The compiler also offers target-only fitting and soft retention. Soft retention
adds a penalty for changing protected outputs. Its weight controls a tradeoff;
it is not a promise that both objectives improve. See the [methods](methods.md)
for the objective and the [results](results.md) for what happened in the study.

## When the request is impossible

Suppose the protected feature is the target itself: \(p=t=(1,1)\). The same
quantity \(a+b\) would have to equal both zero and one. No update satisfies both
conditions under this fixed-head model.

If the protected set contains \((1,0)\) and \((0,1)\), exact protection requires
both \(a=0\) and \(b=0\). Only the unchanged model remains. Its perfect retention
does not make it a successful repair.

More generally, exact protection on an anchor span prevents a nonzero change at
a target in that span. A larger protected set can remove the directions needed
for the requested edit. Anchor quantity alone therefore says little about edit
quality; their geometry and their relation to the target matter.

## What the anchor tells us about other inputs

For the explicit protected update \((0,1)\), the real output change at an arbitrary
feature \((x_1,x_2)\) is \(x_2\). It vanishes on the line \(x_2=0\) and can be large
elsewhere. Protection at one row is not global protection.

The formal feature theorem generalizes this reasoning. Exact zero effects on
anchors extend to their feature span. Nonzero anchor residuals are multiplied
by the absolute coefficients used to represent a new feature. A component
outside the span needs an additional error term. The theorem needs those
representation premises; a list of anchors does not establish them.

This example has no bias. If bias can change, append a constant feature to the
head and use **augmented** feature rows for span arguments. The runtime receipt
always records that convention. Feature order and a frozen extractor are also
part of the premise. The [formal guide](formal.md) gives the precise statements
and trust boundary.

## Why native precision is a separate step

A requested coefficient change may disappear when stored in a lower-precision
format. For example, adding \(2^{-13}\) to 1 rounds back to 1 in the supported
float16 projection. A receipt must refer to the realized endpoint, not merely
the requested update. The [architecture example](architecture.md#realize-the-proposal-in-its-native-format)
shows this through `lower_structured`.

Even after endpoint projection, floating execution can differ from the real
dot product. Intermediate rounding and kernel behavior need a separate model
or measurement. The current Lean results prove real-valued statements; they
do not certify an arbitrary hardware execution.

## How this example connects to the research

The study asks whether retention constraints improve the balance between target
change and collateral behavior on held-out synthetic inputs. It uses trained
heads over different frozen feature extractors and compares fixed editing
methods on the same checkpoints. Its main finding is a tradeoff: soft retention
reduced collateral error and left more target error than target-only editing.

This example explains why that tradeoff is plausible. It does not prove that the
same tradeoff holds for every task, that an LLM can find the edit, or that a
feature span covers unseen real-world data. Those questions require their own
experiments. Continue with the [study results](results.md), the
[session workflow](workflow.md), or the [next research decisions](limitations-and-next-steps.md).
