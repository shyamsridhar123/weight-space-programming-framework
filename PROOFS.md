# Proof scope, provenance, and reproduction

The weight-semantics library proves preservation and drift results for explicit
real-valued model semantics. Its two generic families are linear heads over
fixed feature maps and finite affine/activation chains. Input and output
meanings are application parameters. The RTS certificate is a preserved
specialization, alongside generic mathematical examples without an RTS
dependency.

The reproduction target is **`WeightSpace.GenericAudit`**. Run both
`lake build WeightSpace.GenericAudit` and
`lake env lean WeightSpace/GenericAudit.lean` from `formal/weight-space`.
Plain `lake build` uses the historical root in the preserved `lakefile.toml`
and does not verify the generic extension.

The saved successful v2 verification was recorded at
`2026-11-07T02:26:56.8929922+00:00`, the evening of November 6, 2026 in Central
time. It is evidence of that historical run. A fresh verification needs its
own successful commands and new output under `.proof-runs/`. Revising the
explanatory documentation does not create a new proof run.

## Separate local-pilot governance model

The generic [governance contract](docs/governance/contract.md) fixes the intended
invariants before runtime implementation. Its
[standalone Lean source](formal/governance/Governance.lean) models immutable
requests, approval by a configured label distinct from the requester, explicit
request/head pins, terminal rejection and revocation, and authorization confined
to local-pilot use. It is separate from the historical weight-semantics library
and its saved evidence.

**The governance model is locally UNCHECKED as of November 10, 2026.** The
[model guide](formal/governance/README.md) records the failed tool-discovery
attempt, proposed obligations, exact direct verification command, and unavailable
axiom report. The [standalone CI workflow](.github/workflows/governance-proof.yml)
is pending; no passing hosted result is established here. Its source contains
proof terms and a transitive axiom audit, but neither has been checked locally.
The historical reproduction commands below do not check this model.

The model uses arbitrary encoding functions without a collision-resistance or
injectivity axiom. It does not verify JSON parsing, canonical serialization,
SHA-256, Python refinement, file I/O, export, or the candidate's application
relation. Callers must obtain independent current request/head identities,
authenticate actors, serialize decisions, and honor the result. Matching stale
pins cannot establish freshness; unequal labels cannot establish organizational
independence. The contract and model guide state these correspondence limits.

A future successful Lean check would establish conditional mathematical claims
about these definitions. Synthetic implementation checks, authenticated
organizational authority, and scientific evidence of useful behavior remain
separate obligations. The optional [Distro handoff](docs/governance/distro-handoff.md)
supplies an authoring and review procedure, not evidence that these obligations
have been discharged.

## What the library proves

The theorem statements and hypotheses are authoritative. The
[formal guide](docs/weight-space/v2/formal.md) develops them with a small head
edit, exact assumptions, intermediate calculations, checked examples, and
boundaries.

| Source | Established result | Essential premise or boundary |
| --- | --- | --- |
| [Generic.lean](formal/weight-space/WeightSpace/Generic.lean) | Exact readout effects, endpoint differences, composition, and rollback | Fixed feature map, finite feature coordinates, and real-valued linear heads |
| [Generic.lean](formal/weight-space/WeightSpace/Generic.lean) | Exact protection throughout a represented feature span | Explicit span coefficients and zero actual endpoint residuals at the anchors |
| [Generic.lean](formal/weight-space/WeightSpace/Generic.lean) | Approximate protection, including conditioning and off-span error | Bounds on anchor residuals and, where needed, feature representation error |
| [Generic.lean](formal/weight-space/WeightSpace/Generic.lean) | Factored effects, protection, and an infeasibility condition | A supplied finite factorization and kernel premises; exact protection prevents nonzero changes inside its feature span |
| [Sequential.lean](formal/weight-space/WeightSpace/Sequential.lean) | Coordinatewise parameter and input drift at any finite depth | Shared layer coordinates and activations, magnitude and one-Lipschitz premises, and input bounds |
| [Contracts.lean](formal/weight-space/WeightSpace/Contracts.lean) | Composable drift contracts and strict argmax preservation | Common domains/endpoints, certified coordinate budgets, and winner margins exceeding the relevant budgets |
| [Core.lean](formal/weight-space/WeightSpace/Core.lean), [Gauge.lean](formal/weight-space/WeightSpace/Gauge.lean), [Bounds.lean](formal/weight-space/WeightSpace/Bounds.lean) | Historical two-layer ReLU results: hidden permutations, positive rescaling, edit transport, exact readout effects, and box drift bounds | The architecture and real-valued domain premises in those sources |
| [GenericExamples.lean](formal/weight-space/WeightSpace/GenericExamples.lean) | Rational instances of the generic results | Hand-written mathematical definitions; no formal binding to Python study artifacts |
| [RTS.lean](formal/weight-space/WeightSpace/RTS.lean) | Historical endpoint, score-drift, and protected-box contracts | Embedded coefficients, explicit feature coordinates, and stated boxes |

For a fixed feature map, a head has the form
`h_W(x)[o] = sum_f W[o,f] * phi(x)[f]`. `endpoint_exact` uses the actual
difference `Q − W`. A requested update is interchangeable with that difference
only if realization preserves it. Bias requires a declared constant feature
and matching coordinate convention.

The span guarantee also has a specific premise. If
`phi(x) = sum_a c[a] * phi(anchor[a])`, anchor residuals bounded by
`epsilon[a]` give:

```text
abs(h_Q(x)[o] - h_W(x)[o]) <= sum_a abs(c[a]) * epsilon[a]
```

Thus coefficients `(-2,3)` and uniform anchor error `0.01` permit drift `0.05`.
A checked example attains it. With coordinatewise feature representation error
`rho[f]`, the bound also includes `sum_f abs((Q-W)[o,f]) * rho[f]`.
Numerical rank estimates and solver-success flags do not prove these premises.
The factor results likewise require exact or bounded right-factor residuals;
they do not certify SVD exactness, minimum rank, or solver optimality.

The sequential recurrence separately propagates baseline magnitudes and drift:

```text
r_next[g] = abs(b[g]) + sum_f abs(W[g,f]) * r[f]
d_next[g] = abs(b'[g] - b[g])
          + sum_f (abs(W'[g,f] - W[g,f]) * r[f] + abs(W'[g,f]) * d[f])
```

The activation assumptions are `abs(sigma(a)) <= abs(a)` and
`abs(sigma(a)-sigma(b)) <= abs(a-b)`. Supplied proofs cover identity, ReLU,
and absolute value. No activation certificate is supplied for tanh, sigmoid,
GELU, softmax, attention, or normalization. All affine parameters may change;
the old and edited chains keep the same activation functions and coordinate
types. A separately bounded input perturbation is allowed. A conservative
bound above budget means this method failed to certify the candidate, without
by itself demonstrating a behavioral violation.

To preserve a strict winner `w`, every competitor `c` needs
`base[w] - base[c] > epsilon[w] + epsilon[c]`. This conclusion concerns the
declared scores and candidate set. It neither creates a margin nor handles
ties, stochastic sampling, or downstream reward.

The checked examples make the scope tangible. Features `(1,x,y²)` permit an
edit that protects all `(t,0)` while changing the two outputs at `(2,2)` by
`2` and `−1`. A separate instance rules out a nonzero change within an exactly
protected span. The three-layer ReLU/absolute-value/identity witness has
uniform drift at most `987/320` on `|x| ≤ 1`; its signed difference at `x=1`
is `−443/320`. These are mathematical witnesses, not evaluations of the
runtime study.

## What the axiom audit checks

[GenericAudit.lean](formal/weight-space/WeightSpace/GenericAudit.lean) loads
the historical library and generic modules. It enumerates every loaded
`WeightSpace` declaration and follows its transitive dependencies with
`Lean.collectAxioms`. A dependency outside this allowlist fails elaboration:

```text
propext
Classical.choice
Quot.sound
```

Representative theorem printouts help inspection, but do not define audit
coverage. The full traversal rejects proof placeholders and added axioms,
including additional axioms introduced by native-evaluation proof shortcuts.

The [expected output](#expected-new-output) includes frozen declaration totals.
They identify the environment checked by this snapshot; they are not measures
of explanation quality or independent scientific results. The loaded
environment includes historical and compiler-generated declarations. An audit
theorem declaration is a `Lean.ConstantInfo.thmInfo`, which may be generated.
Those totals are therefore different from counts of explicitly written
`theorem` statements or counts in earlier historical receipts.

## Locked toolchain and dependencies

The [toolchain file](formal/weight-space/lean-toolchain) selects
`leanprover/lean4:v4.24.0`. The saved successful run reports Lean commit
`797c613eb9b6d4ec95db23e3e00af9ac6657f24b`. The complete dependency lock is
[lake-manifest.json](formal/weight-space/lake-manifest.json). Its `rev` values
are the required commits. A floating `inputRev`, such as `main`, `master`, or
a tag, does not replace them.

| Package | Locked commit |
| --- | --- |
| `mathlib` | `f897ebcf72cd16f89ab4577d0c826cd14afaafc7` |
| `plausible` | `dfd06ebfe8d0e8fa7faba9cb5e5a2e74e7bd2805` |
| `LeanSearchClient` | `99657ad92e23804e279f77ea6dbdeebaa1317b98` |
| `importGraph` | `d768126816be17600904726ca7976b185786e6b9` |
| `proofwidgets` | `556caed0eadb7901e068131d1be208dd907d07a2` |
| `aesop` | `725ac8cd67acd70a7beaf47c3725e23484c1ef50` |
| `Qq` | `dea6a3361fa36d5a13f87333dc506ada582e025c` |
| `batteries` | `8da40b72fece29b7d3fe3d768bac4c8910ce9bee` |
| `Cli` | `91c18fa62838ad0ab7384c03c9684d99d306e1da` |

Use the preserved manifest. `lake update` resolves dependencies again and can
change the lock, so it is outside this reproduction procedure. Preserve file
bytes, including newline encodings, when copying or checking out hash-bound
inputs. Source checks deliberately reject newline normalization of those
files. The live documentation and original archived documentation have
different roles, explained in the [receipt mapping](#mapping-the-saved-receipts-into-this-repository).

## Fresh-clone procedure

Start in a fresh standalone checkout. Install Git, curl, and Elan using the
official Lean instructions for your platform, and make `elan`, `lean`, and
`lake` available on `PATH`. The portable receipt runner requires Python 3.10
or later and uses only its standard library. The PowerShell entry point
requires PowerShell 7 or later. The two direct Lake proof commands do not
require Python.

Elan selects the version named in `lean-toolchain`. The preparation below
requests that release and retrieves compiled Mathlib artifacts for the imports
used here. Lake obtains missing Git dependencies using the existing manifest.
Preparation needs network access when components are absent. Before running
the checker, the dependency checkouts must exist; the checker validates their
locked revisions and tracked-source cleanliness.

On Linux or macOS with Bash:

```bash
cd formal/weight-space
elan toolchain install leanprover/lean4:v4.24.0
lake exe cache get Mathlib.Data.Real.Basic Mathlib.Algebra.BigOperators.Fin Mathlib.Algebra.Order.BigOperators.Group.Finset Mathlib.Tactic.Ring Mathlib.Tactic.Linarith Mathlib.Tactic.NormNum
cd ../..
bash scripts/check_proofs.sh .proof-runs/first-linux-check
```

On Windows with PowerShell:

```powershell
Set-Location formal/weight-space
elan toolchain install leanprover/lean4:v4.24.0
lake exe cache get Mathlib.Data.Real.Basic Mathlib.Algebra.BigOperators.Fin Mathlib.Algebra.Order.BigOperators.Group.Finset Mathlib.Tactic.Ring Mathlib.Tactic.Linarith Mathlib.Tactic.NormNum
Set-Location ../..
pwsh -NoLogo -NoProfile -File scripts/check_proofs.ps1 -OutputDirectory .proof-runs/first-windows-check
```

Run the commands sequentially and resolve preparation errors before invoking
the runner. If that exact Lean toolchain is installed, omit its installation
step. To select a different Python executable or shell, invoke the common
runner from the repository root:

```text
python scripts/check_proofs.py --output .proof-runs/first-direct-check
```

Use `python3` or `py -3` in place of `python` if appropriate. Each output must
be a **new subdirectory of this checkout's `.proof-runs/`**. The runner refuses
an existing directory, including one left by a failed attempt. Use a fresh
run name for the next attempt.

The [PowerShell wrapper](scripts/check_proofs.ps1) and
[Bash wrapper](scripts/check_proofs.sh) call the same
[Python runner](scripts/check_proofs.py). That runner:

1. Verifies the saved v2 receipt against its fixed raw SHA-256, then checks the
   proof inputs, preserved earlier inputs, and saved logs. Its explicit
   document mapping resolves original hash-bound guides to their archive copies.
2. Checks the toolchain pin, selected Lean version, and locked dependency
   commits. Modified tracked dependency sources cause failure.
3. Runs `lake build WeightSpace.GenericAudit`, followed by
   `lake env lean WeightSpace/GenericAudit.lean`, from the proof project.
4. Requires the complete audit output, matching frozen declaration/namespace
   totals, and the permitted observed axiom set.
5. Rechecks proof inputs, its current driver files, dependency state, and the
   selected Lean binary before writing a successful new receipt.

The checker preserves historical receipts. It does not install tools, fetch
caches explicitly, change the lock, regenerate Lean sources, or run studies.
Lake can reuse build artifacts and may need artifacts absent locally. This
procedure does not promise a fully offline build or a from-source bootstrap
of Lean and Mathlib.

After preparation, the direct proof commands are:

```text
cd formal/weight-space
lake build WeightSpace.GenericAudit
lake env lean WeightSpace/GenericAudit.lean
```

Both must return zero. They check the proof target without the portable
runner's historical-byte checks or new standalone receipt.

The [archived historical README](provenance/original-documents/formal/weight-space/README.md.txt)
and [v2 Windows checker](formal/weight-space/verification-v2/check.ps1)
document the original RLCD environment. That checker expects the local
toolchain under `.cache/weight-space/lean-4.24.0-windows/bin`. The original
`scripts/check_weight_space_proofs.ps1` builds the historical root and audit.
Use the portable procedure above for the full generic extension in a new
checkout.

Official command references used to prepare the procedure are retained as
source locators. The Lake and Mathlib references are pinned to this project:

```text
https://lean-lang.org/install/manual/
https://github.com/leanprover/elan/blob/master/README.md
https://raw.githubusercontent.com/leanprover/lean4/v4.24.0/src/lake/README.md
https://raw.githubusercontent.com/leanprover-community/mathlib4/f897ebcf72cd16f89ab4577d0c826cd14afaafc7/README.md
```

## Expected new output

A successful portable run creates the following evidence in its chosen run
directory. Ordinary Lake build products are created under the proof project's
`.lake` directory.

```text
.proof-runs/<new-run-name>/
  build.log
  axiom-audit.log
  verification.json
```

The audit log must contain these frozen totals. Lean may prefix messages with
source locations; the order of the three observed axioms is immaterial.

```text
V2 full transitive axiom audit passed: 376 declarations; 215 theorem declarations.
V2 new namespaces: 140 declarations; 94 theorem declarations.
V2 namespace WeightSpace.Generic: 27 declarations; 22 theorem declarations.
V2 namespace WeightSpace.Sequential: 62 declarations; 32 theorem declarations.
V2 namespace WeightSpace.GenericExamples: 51 declarations; 40 theorem declarations.
V2 observed axioms: [propext, Quot.sound, Classical.choice].
```

The receipt schema is `wsp-standalone-proof-verification/1`. It records the
platform, Python and Lean versions, selected Lean executable hash, dependency
commits, command exits, audit totals, hashes of preserved proof inputs and
current driver files, and hashes of the new logs. `preserved_document_paths`
records the archive resolution for original document names. Schema and
timestamps distinguish it from the historical
`wsp-generic-formal-verification/2` receipt. Timings, executable hashes,
platform text, and build logs need not equal the original Windows run.

Failure returns nonzero and does not create a successful `verification.json`.
Partial logs may remain; preflight can fail before creating the run directory.
A deliberate proof-source change needs a new provenance record and review.
Editing old receipt hashes or relaxing expected audit totals would no longer
check the preserved snapshot.

## Mapping the saved receipts into this repository

Historical source paths retain their original repository-relative names. For
example, `formal/weight-space/WeightSpace/Generic.lean` resolves under this
standalone checkout. The outer `exports/weight-space-programming-framework`
directory belongs to the source RLCD workspace and is not part of the new
repository's path scheme. Machine-absolute paths in old logs describe the
machine that produced them.

Two documentation files now have live explanatory revisions. Their original
bytes are archived; the portable checker uses the following explicit mapping
when resolving their historical receipt entries:

| Historical receipt path | Original bytes used for preservation checks | Current reader guide |
| --- | --- | --- |
| `docs/weight-space/v2/formal.md` | [Archived formal guide](provenance/original-documents/docs/weight-space/v2/formal.md.txt) | [Formal guarantees](docs/weight-space/v2/formal.md) |
| `formal/weight-space/README.md` | [Archived proof README](provenance/original-documents/formal/weight-space/README.md.txt) | [Proof project guide](formal/weight-space/README.md) |

The mapping changes where historical bytes are read. It does not rewrite a
saved receipt, substitute a new proof source, or claim that a revised guide
was an input to an earlier run. The archived formal guide's SHA-256 remains
`a3a92404751b917afdde19cefadebcc06fd1d11a5531440fbe528ec10d233603`.
The [source preservation inventory](provenance/source-files.json) records
the broader export mapping.

| Preserved evidence | Interpretation |
| --- | --- |
| [verification-v2/actual/verification.json](formal/weight-space/verification-v2/actual/verification.json) | Saved generic-extension run; original `source_sha256` names and digests remain intact |
| [verification-v2/actual/build.log](formal/weight-space/verification-v2/actual/build.log) and [axiom-audit.log](formal/weight-space/verification-v2/actual/axiom-audit.log) | Original logs; short log names in the receipt resolve beside it |
| [preserved-v1-inputs.json](formal/weight-space/verification-v2/preserved-v1-inputs.json) | Its top-level `sha256` maps preserved paths to file digests; it is not the snapshot file's own digest |
| [verification/actual-verification.json](formal/weight-space/verification/actual-verification.json) | Earlier RTS-specialization record, with its original source, model, and log references |
| [verification/generic-verification.json](formal/weight-space/verification/generic-verification.json) | An earlier generic-only stage, before the RTS import and later modules; its root/audit hashes describe that stage |

The earlier generic-only receipt's hashes for `WeightSpace.lean` and
`AxiomAudit.lean` intentionally differ from their later versions. Preserving
that receipt does not assert that every source version it mentions is current.
The portable runner checks the direct v2 closure and all entries in its
preservation snapshot. It does not substitute current files recursively into
every past-stage receipt.

The saved v2 receipt's raw SHA-256 is:

```text
7d3435dac048bd2ae4c28b90e7e9a035c5134cd0edee75ae200b32c2d68a9eac
```

The preservation snapshot's raw SHA-256 is:

```text
bfde17569c918303934b6a3beb5f20f759e76fd3a8f4567cfdbab2476da161ba
```

The historical Windows `lean.exe` hash is
`20bc808248e38cf072f84f4eb389f8f98e5925d703bcbfa400d9dd44f310c4db`.
It identifies that platform-specific execution. Linux or macOS runs record
their own selected binary rather than requiring the Windows hash.

For a complete proof source and evidence copy, retain the following original
inputs. The two documentation entries use the archive mapping above. Toolchain
caches and `.lake` contents do not replace source and lock files.

```text
formal/weight-space/
  WeightSpace.lean
  WeightSpace/
    Core.lean
    Gauge.lean
    Bounds.lean
    Contracts.lean
    Examples.lean
    RTS.lean
    AxiomAudit.lean
    Generic.lean
    Sequential.lean
    GenericExamples.lean
    GenericAudit.lean
  lakefile.toml
  lake-manifest.json
  lean-toolchain
  README.md                         [original bytes in archive]
  generate_rts_certificate.py
  verification/
    generic-verification.json
    generic-build.log
    generic-axiom-audit.log
    actual-verification.json
    actual-build-inputs.json
    actual-build.log
    actual-axiom-audit.log
    actual-regeneration.log
  verification-v2/
    check.ps1
    preserved-v1-inputs.json
    actual/
      verification.json
      build.log
      axiom-audit.log
docs/weight-space/v2/formal.md        [original bytes in archive]
```

For independent inspection of the historical RTS receipt's external source
and input references, also retain:

```text
scripts/check_weight_space_proofs.ps1
artifacts/weight-space/study-v1/rts-candidates/exact-policy.json
artifacts/weight-space/study-v1/rts-export/model.json
```

These additional files explain the earlier generation and source-comparison
claims. They are unnecessary for elaborating the already generated Lean
modules. The portable check does not rerun the generator or the empirical RTS
study. Broader framework, experiment, source-retrieval, and integrity evidence
has a separate scope.

## Trust boundary and interpretation

Lean establishes statements about the supplied definitions under their stated
hypotheses. The trusted base includes the kernel and toolchain, allowed
foundational axioms, imported libraries and compiled artifacts, and execution
environment. A cached build does not reconstruct Lean's compiler or recheck
all imported Mathlib artifacts from source. The wrappers add operational
checks and records; their output is not itself a Lean theorem about a
filesystem or build system.

The semantics use real numbers and exact rational constants. A binary64
coefficient must be interpreted as its exact represented rational when
connecting it to those definitions. A short decimal display is not
interchangeable with that value. A hash label alone does not prove correct
extraction into Lean. The historical RTS generator performed an external
comparison; the generic examples are hand-written witnesses. Neither proves
general Python, NumPy, PyTorch, JavaScript, or hardware execution equivalence.

Additional obligations remain for floating-point error, solver correctness,
feature-extractor correspondence, domain coverage, arbitrary operators,
cross-architecture transfer, causal repair, unlearning, retention, reward, and
source-paper replication. The
[formal guide's correspondence table](docs/weight-space/v2/formal.md#what-the-proof-does-not-connect-yet)
and [research plan](docs/weight-space/v2/limitations-and-next-steps.md) specify
what each needs. Local hashes and timestamps identify supplied evidence; they
are not signed independent attestations.

For standalone CI, install the pinned toolchain, preserve locked dependencies
and original file bytes, run the generic target and separate audit, and retain
the newly created `.proof-runs/<run>` directory. Cache keys should distinguish
the operating system, toolchain, and manifest. Old logs remain historical
evidence. A source build or an extended proof library can have a separate
verification mode with its own inputs and provenance.
