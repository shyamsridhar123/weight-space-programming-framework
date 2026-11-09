# Main-study artifact verification

The saved independent comparison recorded 224 matching deterministic outcomes,
zero mismatches, 2,760 evaluator input files, 18 frozen source files and 448
outcome files. The check completed at 2026-11-09T04:30:17.361270+00:00.
The JSON files in this directory are newly sanitized derivatives of existing
receipts. Their source raw hashes identify the retained originals; their own
canonical and raw hashes identify the new derivatives.

`compare_released_live_replay.py` is a portable derivation of the original
comparison implementation. From the repository root, an auditor can run:

```text
python artifacts/research/live-studies/verification/compare_released_live_replay.py --repo .
```

The program requires Python 3.11 or later and the standard library. It prints
a fresh artifact-comparison report to stdout and does not write evidence
files. NumPy and PyTorch do not need to be installed to compare the saved
artifacts. This derivative was not executed against the study during
packaging; validation was limited to AST parsing, static source comparison,
receipt transformation, hashes and privacy checks.

The algorithm retains the original gate/lock anchors, assignment accounting,
evaluator input inventory, evaluation/replay start bindings, complete outcome
inventory, deterministic equality, source snapshots and current source hashes.
`main-comparison-provenance.json` lists the exact preserved checks, code changes,
source identities and derivative file hashes. The repository location comes
from `--repo`. The prior replay exit is consumed through its pinned sanitized
derivative; original byte identities and derivative identities remain distinct.

The tool compares released artifacts. It does not execute the evaluator,
reconstruct model behavior, or independently establish provider provenance.
The original report's zero-provider-call assertion remains a labeled capture
claim in `main-replay-comparison.sanitized.json`. The portable tool's own zero
call field describes its stdlib artifact-reading behavior. Timing and opaque
handles remain outside deterministic equality.

`main-replay-exit.sanitized.json` retains replay status, timing and manifest
bindings. `main-comparison-exit.sanitized.json` retains comparison status,
timing and all nine tracked input identities with relative paths or roles.
`main-environment.sanitized.json` retains the six runtime declarations copied
from saved records. It does not attest a complete package lock, interpreter
binary identity, environment-variable values or a hidden provider runtime.

Sanitization uses explicit field whitelists and documented role rewrites.
Process identifiers, absolute machine/profile locators and raw transport
contents are omitted. Existing source evidence was left unchanged.

For signed JSON in this directory, `sha256` is SHA-256 of UTF-8 JSON with
sorted keys, compact separators, non-ASCII text retained, nonfinite numbers
rejected, and the top-level `sha256` field excluded. Raw SHA-256 hashes cover
the exact serialized bytes, including whitespace and the final newline.
