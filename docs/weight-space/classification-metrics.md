# Corrections and newly incorrect cases

An edit can leave average accuracy unchanged while breaking previously correct
cases. Keep the two directions of change visible before computing a net rate.

[`correctness_transition_counts`](../../rlcd/weight_space/metrics.py) accepts
three aligned, one-dimensional Boolean arrays: parent correctness, candidate
correctness and a row-selection mask. Boolean lists and tuples are also accepted.
Numeric labels, scores, scalar Booleans, object arrays, masked arrays, missing
sequence values and mismatched shapes are rejected; the helper does not coerce
them into correctness or discard a missing-value mask.

The caller defines prediction decoding, expected outputs and each region. The
same primitive works for target, protected, overlapping or other caller-defined
groups. It imposes no task, queue mapping, class vocabulary or region policy.

```python
import numpy as np
from rlcd.weight_space.metrics import correctness_transition_counts

expected = np.array([0, 0, 1, 1, 2, 2])
parent = np.array([1, 0, 0, 1, 2, 0])
candidate = np.array([0, 1, 1, 1, 2, 2])
target = np.array([True, False, True, False, False, False])

target_counts = correctness_transition_counts(
    parent == expected, candidate == expected, target)
protected_counts = correctness_transition_counts(
    parent == expected, candidate == expected, ~target)
```

These are authored examples, not measured repair results:

| Count | Target | Protected |
| --- | ---: | ---: |
| `rows` | 2 | 4 |
| `stayed_correct` | 0 | 2 |
| `corrected` | 2 | 1 |
| `newly_incorrect` | 0 | 1 |
| `stayed_incorrect` | 0 | 0 |

All returned values are native Python integers. The four transitions partition
the selected rows. The gain numerator is `corrected - newly_incorrect`; the
net-harm numerator is `newly_incorrect - corrected`. Divide by `rows` only when
it is nonzero. Empty Boolean arrays and empty selected regions return five
zeroes; their rates remain undefined (`None`) in caller-owned calculations.

The protected example has zero net harm despite one newly incorrect case. Its
two decision flips can be counted from decoded outputs separately. Boolean
correctness alone cannot distinguish one wrong decision from another; both
remain `stayed_incorrect`.

This utility adds no scoring callback, score decoder, missing-study outcome,
eligibility rule, uncertainty estimate or success decision. Existing queue
metrics and their callers retain their behavior. The [authored engineering
check](../../tests/test_correctness_transition_counts.py) exercises the literal
example and input boundary. Its separate
[verification receipt](../../provenance/verification/correctness-transition-counts-20261109.json)
records one passing invocation at the captured source snapshot, with no reported
warnings, retry or timeout and unchanged source, runner, Git and runtime
identities. It checks net cancellation, wrong-to-wrong decisions, native integer
counts, the transition partition, empty Boolean inputs, Boolean sequences and
invalid inputs in all three argument positions. Execution counts and elapsed
time are bookkeeping. Documentation snippets are not separately executed, and
the captured check does not verify later documentation or publication edits.

The helper is available to future explicitly decoded classification callers.
It is not integrated into the [prospective scientific
evaluator](../research/staged-scientific-compatibility.md), its whole-slot
receipts or admission gates. Adding a package module changes source inventories
that include all package files; existing historical freezes and receipts retain
their original identities. No scientific criterion is promoted by this utility.
