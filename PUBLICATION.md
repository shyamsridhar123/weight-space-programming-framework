# Public snapshot and historical evidence

**`v0.2.0-draft.4` publication candidate — prepared November 7, 2026. The draft
release remains unpublished.**

Draft.4 is derived from private draft.3 commit
`2424c592fd8ed6b57217bd7b55224f77d0094455`. It prepares a separate public
snapshot with disclosed privacy transformations. The publication plan uses a
fresh Git history and a noreply author identity; the original repository and
its earlier history remain private. Preparing the candidate does not establish
that the public transition or a hosted workflow has completed.

The framework, worked examples, image flowcharts, research conclusions, and
conditional proof claims keep their existing scope. This publication work
does not add an experiment, expand a theorem, or establish a new scientific
result.

## Which record describes which bytes

| Record | Purpose |
| --- | --- |
| [Original source inventory](provenance/source-files.json) | Retains the recorded original paths, byte lengths, SHA-256 values, and preservation mappings. Original identity hashes are not replaced with hashes of sanitized copies. |
| [Public sanitation ledger](provenance/public-sanitization.json) | Binds each changed or omitted path to its original identity, public identity or omission, and reasons. |
| [Current publication manifest](provenance/publication-manifest.json) | Describes the draft.4 publication tree and its transformations. |
| [Historical draft.3 publication manifest](provenance/historical-publication/draft3-publication-manifest.json) | Preserves the prior publication record in its historical scope. It is not the manifest for draft.4. |
| [Publication checksums](MANIFEST.sha256) | Identify the delivered public bytes according to the publication verifier's coverage rules. |
| [Verification guide](VERIFICATION.md) and [archive index](provenance/verification/index.json) | Separate checks of this candidate from records of earlier executions. |

The ledger uses schema `wspf-public-sanitization/1`. It records the private
source commit and, for each affected path, the original byte length and
SHA-256, the public byte length and SHA-256 or `null` for an omission, and the
reasons for the change. The public verifier checks the disclosed public
representation. A ledger entry does not make changed bytes equal to their
original hash, and hashes alone cannot recover unavailable private bytes or
authenticate who produced them.

## Disclosed transformations

Internal package-environment locators and author-workstation paths are
redacted from retained text and environment records. These are public
derivatives wherever the delivered bytes differ. The root package lock omits
environment-specific `resolved` URLs while retaining its versions, integrity
values, and dependency graph. npm must resolve endpoints through the user's
configured registry. The public npm verification attempt stopped at TLS
before retrieving metadata or verifying any tarball. Public-registry
availability and installation portability remain unverified; omitting URLs
does not establish either result.

Ten Python bytecode cache files are omitted. Their original identities remain
recorded, and the ledger explicitly records their absence. They are not
silently removed from the historical evidence inventories that named them.

The raw GitHub UI capture at
[`artifacts/weight-space/source-review-v2/repository-tree.html`](artifacts/weight-space/source-review-v2/repository-tree.html)
is replaced with a minimal safe source reference. It identifies the reviewed
source without retaining the captured page's transient request, visitor, and
form-token values. The replacement is disclosed as a derivative, not presented
as the original captured HTML.

These changes remove environment and capture details while retaining the
scientific numerical payloads and their original protocol and source
identities. The ledger, rather than a blanket preservation claim, identifies
which delivered files differ. Earlier reports, receipts, and release notes
remain historical records even when their environment fields need a public
derivative.

## Historical checks remain strict

The original freeze receipts and their source hashes are retained. The
historical study's freeze, independent audit, and report checks are not relaxed
to make the sanitized copy pass as an original. In particular:

- Redacting the first study's frozen `package-lock.json` breaks its original
  frozen-source hash check.
- Redacting stage logs and omitting caches breaks the saved audit's inventory
  of original file bytes. The original audit also names its historical study
  location.
- The first-study report requires a passing audit of the actual inputs and
  checks the frozen source manifest. A saved historical `passed: true` cannot
  validate this changed public tree.

Those checks are in [the first-study auditor](scripts/audit_weight_space_study.py),
[reporter](scripts/report_weight_space.py), and
[runtime evidence verification](rlcd/weight_space/evidence.py). Their expected
rejection of changed original inputs is consistent with a separate successful
publication-integrity check. The latter can establish that a public derivative
matches its disclosure; it cannot establish exact replay of the private
freeze.

The generic study and proof procedures retain their own expected identities
and limits. Do not remove expected hashes or rewrite a freeze receipt to force
a result. Any new replay must report its actual outcome on its actual inputs.
No fresh exact scientific replay or Lean run is claimed for draft.4. Historical
passing receipts and report prose describe their identified historical runs.

For new scientific work, prepare clean inputs, fix a new protocol before
observing its outcomes, create a new source freeze, and use unused output
directories. Reusing the published worlds is reproduction on observed units;
fresh confirmation needs untouched units or task lineages. See
[REPRODUCING.md](REPRODUCING.md) for the distinction and executable procedures.

## Current validation and release status

The local Python suite, isolated wheel build, and fresh external wheel
installation passed for their recorded inputs. The
[verification record](VERIFICATION.md) links those current receipts and
states their environment limits. Final publication, source-archive,
confidentiality, and hosted receipts separately bind the completed tree and
commit; historical successes do not supply those outcomes. The public npm attempt was blocked by
TLS, with no metadata or tarball verified. The public repository transition
and a hosted result for the new commit are not claimed as complete here. The
draft.4 release remains unpublished.

Earlier release-asset links and commit identifiers refer to the private
repository's historical record and may be unavailable to a public reader.
Their presence is not evidence that corresponding assets exist in the new
repository. No project-wide license grant or stable API commitment is added
by preparing this snapshot; [RIGHTS.md](RIGHTS.md) retains the attribution and
rights boundaries.
