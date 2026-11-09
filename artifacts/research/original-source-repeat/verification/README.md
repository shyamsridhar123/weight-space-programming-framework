# Original-source replay and audit receipts

The saved corrected offline replay and its standalone independent audit passed. The underlying live result remains four attempted harness calls, four failed provider receipts, zero completed provider receipts, zero valid responses, and zero credited correct outcomes. These failures do not measure model capability accuracy.

This directory contains sanitized derivatives of existing evidence. The packaging operation did not rerun the replay or audit, execute a comparator or evaluator, import study code, or make provider calls. No original capture was overwritten. It sits outside the frozen 31-file preparation inventory and 21-file live bundle.

## Saved results and timing

| Phase | Started (UTC) | Child finished (UTC) | Verification finished (UTC) | Saved child/wrapper exit | Captured elapsed seconds | Stdout / stderr bytes |
| --- | --- | --- | --- | --- | --- | --- |
| Corrected offline replay | 2026-11-09 04:41:04.159931 | 2026-11-09 04:41:05.681323 | 2026-11-09 04:41:06.599577 | 0 / 0 | 2.4168554000207223 | 259 / 0 |
| Standalone audit | 2026-11-09 04:43:37.213438 | 2026-11-09 04:43:38.739015 | 2026-11-09 04:43:38.765537 | 0 / 0 | 1.5510153999784961 | 487 / 0 |

The audit recorded 131 authenticated inputs, 31 preparation files, 21 live-bundle files, 18 current/frozen source pairs and 44 acquired assets checked using SHA-256 and Git blob SHA-1. It required each failed response to be null, independently reconstructed all four empty-response scores, and checked four publication hash linkages. The four policies remain in frozen order: `a9f535cc/breakout`, `dfc6cb65/pong`, `bdf99279/pacman`, `0bbcc3ea/freeway`.

All four receipts lack each input, cached-input and output usage field. The recorded sum of observed values is a lower bound of zero; actual usage is unknown. There is no finite usage upper bound and all four request monetary amounts remain unknown. The broad `transport_or_response_failed` category does not diagnose a network, authentication, quota or capability cause.

## Public files and identities

- [original-replay-exit.sanitized.json](original-replay-exit.sanitized.json) retains saved exit/timing/stream-hash fields and rewrites all 131 input identities into 120 repository-relative paths and 11 logical roles.
- [original-audit.sanitized.json](original-audit.sanitized.json) retains the audit counts, four ordered attempt rows, usage missingness, scope limitations and five capture byte identities.
- [original-audit-exit.sanitized.json](original-audit-exit.sanitized.json) retains the audit process result, report binding and before/after input commitments.
- [original-environment.sanitized.json](original-environment.sanitized.json) retains the recorded Python 3.12.10 / NumPy 2.5.3 Windows runtime declaration, finite binary64 arithmetic, C/output-then-input weight order, interpreter commitment and 428-file NumPy inventory commitment.
- [original-prelaunch-history.sanitized.json](original-prelaunch-history.sanitized.json) preserves the retrospective failed-prelaunch record and final successful preflight, with later diagnosis separately labeled.
- [original-verification-provenance.json](original-verification-provenance.json) binds original source identities and every other released file in this directory.

Every receipt identifies its original source bytes separately from its derivative identity. Its explicit field whitelist and rewrite notes describe sanitation. Raw SHA-256 hashes identify exact serialized bytes. Canonical payload hashes use sorted-key compact UTF-8 JSON, `ensure_ascii=false`, `allow_nan=false`, and omit only the top-level `sha256` field. A `canonical_json_sha256` instead hashes the entire parsed original object, including any existing self-hash.

The replay's saved `evidence_after_sha256` commits to the original private locator map. The derivative's `sanitized_input_inventory_sha256` identifies the public rewritten inventory. They intentionally identify different objects. Private input bytes, raw provider transport, process identifiers and absolute machine locators are not published here. The package does not provide a portable original-audit executable; the main study's separately reviewed portable artifact comparator is in [the main verification directory](../../live-studies/verification/README.md).

## Preserved prelaunch history

The first wrapper invocation exited 1 before child launch; all five expected phase capture files were absent afterward. Its safe output was `{"exception_type":"ValueError","status":"blocked"}`. The saved block record was written retrospectively: exact failed-invocation start/finish times are unavailable, and the shell-reported duration was approximately 1.2 seconds. Its earlier `diagnosis_pending=true` value remains intact.

The diagnosed Windows differences were ctime values between `Path.stat()` and `os.fstat()`, and differing executable permission bits between path and handle APIs. Only the private audit helpers were corrected: cross-API checks compare device, inode, file type, size and mtime, while full same-API before/after metadata and both no-link checks remain. The archived first-version and final helper byte identities are retained.

A later read-only preflight stopped on an extra README inside the strict preparation inventory. The notice was moved byte for byte to `docs/research/original-source-inputs-NOTICE.md`, outside frozen inputs. This was an additional blocked preflight, not another replay launch. The final preflight passed from 2026-11-09 04:40:50.569132 to 04:40:51.459309 UTC, authenticating 131 inputs and four terminal attempts. The corrected replay and standalone audit then each ran once. Earlier parent replay evidence and the failed prelaunch history remain preserved.

## Trust and interpretation limits

The saved zero-fresh-call values are captured claims supported by the authenticated offline source, command and exact expected stdout. This packaging operation does not independently attest OS or network behavior, hidden provider retries or provider origin. Public raw-hash linkage does not reconstruct withheld provider events.

The stdlib audit hashed public event bytes without independently parsing event semantics; those checks rely on the authenticated runner's exact safe stdout. It did not independently rehash installed NumPy or the complete runtime import closure. The runtime derivative recalculates only the digest of the saved 428-file inventory, not installed files. Interpreter, standard-library/native-runtime correctness, filesystem observations and local clocks remain trusted.

The main comparison was a raw-hash-bound prerequisite; no main outcome files were opened by the original audit, and main evaluation, frozen replay and comparison were not repeated. These results do not reproduce original training, game-environment/reward trajectories or prose-mechanistic claims.
