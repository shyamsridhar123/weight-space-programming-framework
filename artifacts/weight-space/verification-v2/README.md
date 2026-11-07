# Python integration verification, version 2

The full Python suite passed: **341 passed, zero failed, zero errors, zero skipped**.
Pytest reported two dependency deprecation warnings. The subprocess exited with code 0.

The run started on **2026-11-07 at 02:42:38.308393 UTC** and ended at
**02:43:50.127683 UTC**, taking **71.8195598 seconds** of wall time. Pytest's own
reported duration was **69.17 seconds**. The local date was November 6, 2026 in
America/Chicago.

[receipt.json](receipt.json) binds the exact command, runtime, outcomes, source
manifests, import review, environment, dependencies, and output hashes. Its file
SHA-256 is `98be94100ed8fcf465593c99734860944680f1c7894be9069ceff7bb7251e937`.
All 13 files bound by that receipt were independently rehashed after completion;
none differed.

- [python-tests.command.json](python-tests.command.json): exact absolute argument
  vector, working directory, environment overrides, and runner hash. The full
  `tests` directory was executed without deselection, with bytecode and pytest
  cache writes disabled and a fresh `pytest-tmp` directory.
- [python-tests.stdout.log](python-tests.stdout.log) and
  [python-tests.stderr.log](python-tests.stderr.log): persistent test output.
- [python-tests.junit.xml](python-tests.junit.xml),
  [python-tests.results.json](python-tests.results.json), and
  [python-tests.status.json](python-tests.status.json): per-test outcomes, counts,
  failure details, timestamps, and exit status. Stderr was empty.
- [environment.json](environment.json): Python 3.12.10, Windows environment,
  installed distribution names and versions, and numerical threading environment.
  Relevant versions were NumPy 2.5.3, PyTorch 2.8.0+cpu, pytest 9.1.1,
  Transformers 5.0.0, FastAPI 0.141.1, and HTTPX 0.28.1.
- [sources-before.json](sources-before.json) and
  [sources-after.json](sources-after.json): identical hashes and byte counts for
  66 source and configuration inputs, covering the Python packages, scripts,
  tests, and all frozen study inputs.
- [freeze-before.json](freeze-before.json) and
  [freeze-after.json](freeze-after.json): all 31 files still match the study freeze
  with declared digest
  `0aef8bc624bc5ea501558a807c3c90afa46b3b8ab5d27ea1423943cbb986349e`.
- [import-coverage.json](import-coverage.json): all 16 local modules loaded by
  importing `study_v2`, plus its auxiliary task module, are covered by the freeze.
  The three excluded modules are imported only in unrelated CLI command branches.
  This is an import audit and static review, not exhaustive tracing of arbitrary
  dynamic imports.
- [run_check.py](run_check.py): the recorded verification wrapper.

No numerical, protocol, formal, or prior evidence files were edited. The version 2
study tests use smoke seeds 900001 through 900008; auxiliary tests use 900001
through 900004. This verification did not execute reserved study seeds 7300
through 7311. The actual study is separate evidence owned by the parent run.

The warnings concern Starlette's HTTPX test client and AnyIO's deprecated
`BlockingPortal` alias. This verification covers Python tests; it does not claim
a browser, TypeScript, or formal proof build. Test temporary artifacts are retained
under `pytest-tmp`, but the receipt does not hash every temporary test artifact.

Preserve this directory as run evidence. Any rerun should use a new versioned
verification directory and a fresh temporary directory.
