# Reproducing the draft

[PUBLICATION.md](PUBLICATION.md) explains the draft.4 public derivatives and
omissions. Original identities are preserved in the records, but some
delivered files no longer have their original bytes. Historical freeze and
audit checks remain strict; a publication-integrity result does not establish
an exact replay of those changed inputs.

Choose the procedure for the claim you want to check. Reconstructing an edit, recomputing a saved result, and repeating a proof build establish different things. The repository retains the inputs and prior outcomes needed to inspect each one.

| What you want to establish | Procedure and its limit |
| --- | --- |
| The checkout matches its declared publication | Run the publication verifier. Checksums show agreement with the declared package, not who executed an experiment. |
| An edit changes a target while preserving a specified feature | Run the small examples after installing the core package. Their explicit formulas illustrate the mechanism; they do not measure an editor's ability. |
| The implementation still meets its tested obligations | Run the relevant Python tests. Development fixtures do not supply a new study result. |
| The saved generic results agree with the retained candidates and data | Run the independent auditor with its expected identities and inspect its actual outcome. It recomputes saved observations without retraining; disclosed sanitation does not waive original-byte checks. |
| The study procedure can be executed again | Use clean inputs, a fresh freeze, and new output directories. Published seeds are already observed; fresh confirmation also needs a newly fixed protocol and untouched units. |
| The stated real-valued theorems check under the pinned dependencies | Build the explicit generic proof target and retain a new receipt. The theorem premises still bound the claim. |

Run commands from the repository root unless a step says otherwise. Use an unused output path for each run. The examples, Python tests, and generic study require neither a live model provider nor an RTS environment. The optional RTS adapter retains an earlier application and its selected no-op result.

[EVIDENCE.md](EVIDENCE.md) connects claims to their historical records. [VERIFICATION.md](VERIFICATION.md) identifies checks actually performed on this publication. New output supplements those records; it does not replace the original studies, freezes, proof receipts, failed attempts, or negative results.

## 1. Check the publication before installing dependencies

Use the standard-library verifier to check the package before installing dependencies:

```bash
python scripts/verify_publication.py
```

It checks the published file set and SHA-256 values in `MANIFEST.sha256`, original identities in `provenance/source-files.json`, disclosed public derivatives and omissions in `provenance/public-sanitization.json`, local guide links, and image-only flowcharts. To retain this check, choose a new receipt path outside the historical archive:

```bash
python scripts/verify_publication.py --out .runs/publication-check-1.json
```

The output filename must be unused. Local dependencies, build output, and new `.runs/` or `.proof-runs/` receipts are excluded from publication checksums. Historical files inside `artifacts/` remain included, even when their names resemble temporary directories. The verifier defines the exact exclusions.

Checksums establish agreement with this package. They do not authenticate who executed a historical experiment, create trusted timestamps, or prove that a source file produced an output. Obtain the manifest from the repository commit you intend to examine.

## 2. Install the Python distribution

Use Python 3.11 or newer. The recorded local verification environment uses Python 3.12. The distribution is named `weight-space-programming-framework`; imports remain `rlcd.weight_space` to preserve the implementation and its source hashes.

```bash
python -m venv .venv
```

Activate it using the command for your shell:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

Install the NumPy core and inspect the CLI:

```bash
python -m pip install -e .
weight-space --help
python -m rlcd.weight_space --help
```

Start with the [worked example](docs/weight-space/v2/worked-example.md), then run its executable companions:

```bash
python examples/edit_retention.py
python examples/session_edit.py
```

The [edit example](examples/edit_retention.py) starts from weights `(1, 1)`. It compares a no-op, an exact target edit that disturbs the protected output, and an exact protected edit. It then shows two numerical-fitting limits: positive ridge leaves some target error, and protecting both independent feature directions forces a no-op. Its exact residual checks and rollback assertions concern this declared linear model.

The [session example](examples/session_edit.py) supplies the desired formula and explicit proposals. It measures public target error and protected drift, applies a fixed selection rule, seals the selected candidate, and evaluates new rows. Both scripts print JSON and check their stated outcomes with assertions. They illustrate the API with constructed examples; they do not create fresh empirical evidence or measure a live LLM.

You can also open the [interactive explanation](artifacts/explanations/edit-retention/index.html) directly in a browser. It is a standalone local HTML artifact with no build or network requirement. The [static worked guide](docs/weight-space/v2/worked-example.md) gives the same core explanation, and the Python examples exercise the implementation. The [README quickstart](README.md) gives the shorter installation and example-running path.

Install the extra dependencies when you need the PyTorch bridge, native-execution replay, the engineering Python suite, or report plotting:

```bash
python -m pip install -e ".[test,plots]"
```

PyTorch is pinned to 2.8.0 for compatibility with the preserved implementation. For a CPU-only installation, install its CPU wheel first and then the package extras:

```bash
python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,plots]"
```

Wheel availability depends on the platform and interpreter. The broad Python requirement expresses the package's declared support; [the verification record](VERIFICATION.md) identifies configurations actually checked. Core edit operations do not need Torch. Plotting is not required for execution or independent evidence replay.

The root `requirements.txt` and `requirements.lock.txt` are preserved historical RLCD inputs. They contain unrelated dependencies and the editable path `./upstream/laya`. Use the standalone `pyproject.toml` installation above. Do not edit the historical lock to make a new environment appear identical to the recorded one.

## 3. Run the tests

Run the reviewed engineering selection used by publication CI. Disable automatic loading of third-party pytest plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -c pytest-engineering.ini --basetemp .runs/pytest-review-1
```

In PowerShell, set `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"` first, then run the same command starting at `python`.

Use this selection to check edit algebra, certificate boundaries, tensor/adaptor compatibility, lowering, native precision, feature editing, sequential bounds, workflow state, factory behavior, publication sanitation, and the proof bridge. [pytest-engineering.ini](pytest-engineering.ini) selects the engineering files and deselects three generic study tests. The independent-auditor fixtures and other research suites are outside this selection. The historical `pytest.ini` remains unchanged; omitting `-c pytest-engineering.ini` uses its broader collection.

The configured suite excludes [tests/test_raw_text_serving.py](tests/test_raw_text_serving.py). This retained module is separate opt-in validation: selecting it constructs authored model/package fixtures, restores model bytes, runs preprocessing and native CPU inference, and launches one actual baseline child. Select it only when that execution is within the agreed validation scope. A passing configured engineering suite therefore does not revalidate raw-text serving. The remaining selection still includes native and numerical work; configured CI is narrower, not native-free. Explicit test paths can select files outside this whitelist; omitting the engineering config uses broader collection.

Keep `--basetemp` inside this checkout and choose an unused `.runs/` directory for each invocation: pytest clears that directory before use.

For focused work, choose files listed in `pytest-engineering.ini`. Explicit test paths override its `testpaths` list:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -c pytest-engineering.ini --basetemp .runs/pytest-focused-1 tests/test_weight_space_v2.py tests/test_weight_space_workflow.py
```

Inspect failures, deselections, and skipped tests before interpreting the result. The selected Torch checks require Torch. These tests use development fixtures; they do not execute the full published study or establish compiler correctness, useful model editing, or confirmation on untouched tasks. Saved-data replay and fresh study execution below are separate, optional scientific procedures.

## 4. Replay the saved generic study independently

Use saved-data replay to ask whether the retained endpoints, data, and claimed outcomes agree. The independent auditor implements the equations without importing `rlcd`. It checks frozen inputs, endpoints, exact rational receipts, numerical observations, workflow counters and decisions, and world-cluster statistics. It does not retrain heads or generate candidate edits.

The command below retains the generic auditor's historical expected identities.
It is a procedure to attempt, not a fresh passing result for draft.4. Any
sanitized input remains subject to the auditor's original checks. Inspect the
actual report and exit status; do not remove expected hashes to make the public
copy pass.

Use a new output filename:

```bash
python scripts/audit_weight_space_v2.py --study-root artifacts/weight-space/study-v2 --freeze artifacts/weight-space/study-v2-freeze/receipt.json --expected-freeze 0aef8bc624bc5ea501558a807c3c90afa46b3b8ab5d27ea1423943cbb986349e --expected-protocol c809fc30b69d9fec2fd720c2e80e0d35a08d49a1dd7f6e2f352a6b78e2cbb3de --repo . --out .runs/saved-evidence-audit-1.json
```

The expected freeze and protocol values above are **canonical JSON identities**, not raw file SHA-256 values. Inspect the new report's `status`, `complete`, individual checks, input inventories, source comparison, and limitations. A new replay has its own identity and result.

The auditor shares NumPy and PyTorch with the study. Numerical observations use absolute tolerance `1e-12` and relative tolerance `1e-10`; hashes, types, counters, literal edits, and rational payloads use exact checks. Replay does not recreate RNG draws, training trajectories, hidden-access restrictions, solver optimality, native behavior on all inputs, or a Lean build.

The historical freeze includes `docs/weight-space/v2/source-review.md`. Its live explanatory revision now differs from that input. The original bytes remain in the frozen sources and [archived publication copy](provenance/original-documents/docs/weight-space/v2/source-review.md.txt). With `--repo .`, the saved-data auditor reports current-source differences as informational while checking the historical archive. A successful replay therefore does not claim that the rewritten guide matches the old freeze.

The first study has a separate, stricter public-copy limitation. Sanitizing
`artifacts/weight-space/study-v1/freeze/sources/package-lock.json` breaks its
original frozen-source hash, and changed logs and omitted caches break its
saved audit inventory. The unchanged
[first-study auditor](scripts/audit_weight_space_study.py) and
[reporter](scripts/report_weight_space.py) must reject those original-byte
claims. Its saved passing audit describes the historical private inputs.
[The sanitation ledger](provenance/public-sanitization.json) validates the
disclosed public representation separately; it does not repair that freeze.

## 5. Reproduce the study procedure in new directories

Use a new execution to reproduce the procedure, including generation, initial training, editing, selection, and evaluation. The [generic protocol](docs/weight-space/v2/protocol.json) trains output heads over frozen random features and compares the declared deterministic arms. It contains no live LLM/provider arm. Read the [methods](docs/weight-space/v2/methods.md) for its populations, access rules, metrics, and failure policy.

Strict `study_v2.verify_freeze(current_sources=True)` and `run()` compare live sources with the supplied freeze. They reject the rewritten source-review against the historical receipt. **Use clean inputs and a fresh freeze for fresh execution** and retain the original receipt as historical evidence. A new scientific study also needs its own protocol fixed before the new outcomes; do not revise a preserved protocol or receipt to describe a different run.

Create a new source freeze, verify it, and execute into another unused directory:

```bash
python -m rlcd.weight_space.study_v2 freeze --protocol docs/weight-space/v2/protocol.json --out .runs/reproduction-1-freeze
python -m rlcd.weight_space.study_v2 verify-freeze --protocol docs/weight-space/v2/protocol.json --freeze .runs/reproduction-1-freeze/receipt.json
python -m rlcd.weight_space.study_v2 run --protocol docs/weight-space/v2/protocol.json --freeze .runs/reproduction-1-freeze/receipt.json --out .runs/reproduction-1-study
```

All published world seeds, 7300–7311, have already been observed. These commands reproduce the procedure on known units. Confirmation requires untouched units or task lineages and a protocol fixed before their outcomes, including the comparison, selection rule, independent unit, and failure policy.

Retain the new freeze identity and explain its source differences. Evidence writers refuse overwrite. Keep audits and reports outside the study tree, whose own manifest is closed, and preserve incomplete runs and failures under their own identities.

## 6. Render a report or the image flowcharts

The authoritative existing generic report is [study-v2-report-complete/report.md](artifacts/weight-space/study-v2-report-complete/report.md). To make another rendering without replacing it:

```bash
python scripts/report_weight_space_v2.py --study artifacts/weight-space/study-v2 --out .runs/report-1
```

The live framework flowcharts use PNG/SVG files. Their editable source is [scripts/render_flowcharts.py](scripts/render_flowcharts.py); the graph manifest records nodes, edges, source diagram hashes, alt text, image dimensions, and renderer versions. Render into a separate directory for comparison:

```bash
python scripts/render_flowcharts.py --output-dir .runs/flowcharts-1
```

Byte-identical image reproduction assumes the recorded Python, Matplotlib, and font environment. After a diagram edit, inspect each PNG and SVG for the intended graph and readable labels. Update the image manifest and publication change record together. The standalone [interactive explanation](artifacts/explanations/edit-retention/index.html) is separate from this renderer and opens directly without a build.

## 7. Check the formal proofs

Follow [PROOFS.md](PROOFS.md) to prepare Lean 4.24.0 and the exact dependencies in `formal/weight-space/lake-manifest.json`. The generic target must be explicit:

```bash
cd formal/weight-space
lake build WeightSpace.GenericAudit
lake env lean WeightSpace/GenericAudit.lean
```

The portable wrapper checks saved proof-input identities and dependency revisions before and after verification, builds the explicit target, parses the combined axiom audit, and writes a new receipt. From the repository root, after preparing the toolchain:

```bash
python scripts/check_proofs.py --output .proof-runs/check-1
```

Choose a new subdirectory of this checkout's `.proof-runs/`; the wrapper refuses an existing destination, including a failed attempt's directory. The `.ps1` and `.sh` wrappers call the same runner. Plain `lake build` uses the historical default target and is insufficient for the generic extension.

Two historical documentation inputs now resolve to archived original bytes:

| Historical receipt path | Preserved input used by the wrapper |
| --- | --- |
| `docs/weight-space/v2/formal.md` | [Original formal guide](provenance/original-documents/docs/weight-space/v2/formal.md.txt) |
| `formal/weight-space/README.md` | [Original proof README](provenance/original-documents/formal/weight-space/README.md.txt) |

The mapping retains their original expected hashes. Other historical inputs remain at their recorded repository paths and keep their original hash checks. The saved receipt describes its historical proof run; revised explanatory prose and a successful mapping check are not a new Lean run. A new proof-verification claim needs the successful commands, logs, and receipt from that run.

Report whether the build used downloaded mathlib caches or rebuilt dependencies. The wrapper does not promise a fully offline build or a toolchain bootstrap. The proofs establish conditional real-valued statements under the premises and trust boundaries in [PROOFS.md](PROOFS.md).

## 8. Optional historical RTS compatibility checks

The root Node package supports the historical adapter and its compatibility tests. It has an exact `tsx` dependency and a public derivative of the historical dependency lock. The root lock omits environment-specific `resolved` URLs while retaining versions, integrity values, and the dependency graph; npm resolves endpoints through your configured registry. The changes are disclosed in [the sanitation ledger](provenance/public-sanitization.json). The draft.4 public npm attempt stopped at TLS before metadata retrieval or tarball verification, so public-registry availability and installation portability remain unverified. The standalone interactive explanation needs none of these dependencies.

```bash
npm ci
npm run test:historical:rts
```

Use Node 20 or newer. The retained runner accepts `npm run historical:rts -- export ...` or `npm run historical:rts -- evaluate ...`; consult [the historical results and reproduction guide](docs/weight-space/results.md) and [the retained orchestration script](scripts/run_weight_space_study.ps1) for the original invocation context. These controls concern that application. Keep any new orchestration output separate from occupied historical directories.

## 9. Record a new result honestly

Record what the run was intended to establish, then retain the evidence needed to assess that claim: exact command, interpreter and dependencies, platform, elapsed time, exit status, logs, source/input identities, output manifest, and preservation checks. Identify it as a test suite, saved-data replay, study reproduction, fresh confirmation, proof build using caches, or dependency rebuild. Include failed and skipped checks.

For a scientific result, also report the baseline, metric and units, independent population, selection rule, uncertainty, and relevant negative outcomes. A package check or proof build cannot supply an empirical comparison. The [evidence guide](EVIDENCE.md), [verification record](VERIFICATION.md), and [contribution guidance](CONTRIBUTING.md) show how these records stay distinct.
