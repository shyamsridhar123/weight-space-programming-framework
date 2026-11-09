# Symmetry-family engineering check — verified inline fixtures

The [standalone check](../../scripts/check_symmetry_family_boundary.py) passed once
on November 9, 2026. It supplies inline functional engineering fixtures relevant
to issue #16 c0, separate from the frozen trained-lineage pilot. The
[sanitized verification receipt](../../provenance/verification/symmetry-family-boundary-20261109.json)
records the captured source inventory, timing, outcome and scope.

For a dense affine/ReLU/affine network, hidden permutations and strictly positive
incoming-row/bias scales with reciprocal outgoing-column scales preserve the real
function. The fixture uses exactly representable dyadic parameters and scales;
its score comparisons use the existing rational `exact_score` oracle on a fixed
grid that includes both hidden activation boundaries.

The execution observed the following finite fixture behavior:

| Fixture | Observed result |
|---|---|
| Supported ReLU artifact | The existing specialized conversion roundtrip retained the model hash. |
| Permutation, scaling, and combined gauges | Parent and nonzero transported-edit scores agreed exactly with their source endpoints on the fixed grid, including nonzero hidden biases. |
| Raw-coordinate transfer | At `x = 1`, the edited source score was `3`; raw transfers produced `3/2` for permutation, `9/2` for scaling, and `3/2` for the combined gauge. Each mismatched despite the same-function parent control. |
| Tanh scaling | Existing sequential execution produced approximately `0.761594155956` for `tanh(x)` and `0.482013790038` for `tanh(2*x)/2` at `x = 1`; the difference exceeded `0.25`. This is a numerical counterexample, not a certificate or a replacement for a native equality tolerance. |
| Tanh and deeper ReLU chains | Valid sequential artifacts were rejected by `as_relu_network`, whose declared family is `dense-relu-dense/1`. The deeper chain also executed through its own sequential backend, returning `1` at `x = 1`. |

The conversion rejection concerns the specialized two-layer ReLU adapter. The
generic framework retains its separate sequential execution path for these graphs.
This check supplies no learned alignment or inferred correspondence, trained
transfer utility, causal intervention, or all-family symmetry guarantee.

The once-only invocation is shown below with a placeholder interpreter; its
actual workstation path and environment are omitted from the public receipt:

```powershell
[PYTHON] -B -m scripts.check_symmetry_family_boundary
```

The recorded invocation started at `2026-11-09T13:15:04.732950+00:00` and finished
at `2026-11-09T13:15:05.334353+00:00`, with measured wall time
`0.5971321999677457` seconds, return code `0`, empty standard error, one process
launch and no retries. It ran no scientific study and made no provider calls.

The tested base revision was `d36c3b977d3c618a9778a702f0aa39f691ef5415` with the
fixture and prepared report uncommitted. Captured source hashes were unchanged
across that invocation. The inventory commits to the captured package files,
check and prepared report; it is not an executed-module trace. This updated
report and the public receipt were added afterward. The execution does not
validate later integration, rebases, commits or publication bookkeeping. The
sanitized projection and its hashes are not provenance authentication.

Issue #16 c0 remains **partial**. This later engineering fixture cannot add
unsupported-family controls to the frozen pilot. Its held-out edited score controls
still pass only **36/48**, leaving **12/48 failed**, and neither native replay
attempt produced a success receipt. The
[original criterion map](trained-symmetry-pilot-criterion-map.md) and
[pilot report](trained-symmetry-pilot.md) retain those unresolved limitations.
