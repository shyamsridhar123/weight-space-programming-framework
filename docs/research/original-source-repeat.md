**Original-source reading-task repeat: results, provenance, and reproduction limits**

**Evidence review: November 8, 2026, America/Chicago.** The saved live requests,
corrected offline replay, and independent audit have November 9 UTC timestamps;
they occurred on November 8 in the local review timezone. This report assesses
the completed evidence already retained in the repository. Writing the report
did not launch a study, replay, audit, comparator, training run, or provider call.

The fresh adapted task produced four attempted harness calls, four failed
provider receipts, zero completed provider receipts, zero valid responses, and
zero credited correct outcomes. Every response is null. This is a transport or
response failure result, not measured model capability accuracy and not evidence
of four incorrect model answers. The corrected offline readback and its separate
independent audit passed; they authenticate and reconstruct the saved failure
accounting. Their success does not turn those failed requests into completed
answers or reproduce the source authors' training, reward trajectories, or
mechanistic explanations.

The evidence supports four distinct conclusions:

| Evidence layer | Saved result | Supported interpretation |
| --- | --- | --- |
| Released historical answers | Four reconstructed correct Game fields out of four | The retained replies agree with the released labels under an explicitly reconstructed keyword rule. |
| Actor conversion and bounded arithmetic | All 17,253 coordinates checked per actor; eight representation checks within tolerance on six synthetic grids | The adapted serialization and numerical execution correspond to the declared source orientation on the checked inputs. |
| Fresh adapted task | Four attempted calls, four failed receipts, no completed answer | The four-assignment protocol retained its failures and missing observations; answer accuracy is unavailable. |
| Corrected offline replay and independent audit | Each completed once after final preflight, with verified stdout and exit 0 | The identified saved bundle is repeatable and its specified byte, accounting, and empty-score checks passed in the recorded environment. |

The [archive overview](../../artifacts/research/original-source-repeat/README.md),
[frozen protocol](../../experiments/research/original-source-repeat/protocol.json),
[live outcomes](../../artifacts/research/original-source-repeat/live-20261108-001/outcomes.json),
and [verification handoff](../../artifacts/research/original-source-repeat/verification/README.md)
provide the corresponding public entry points. These layers should remain
separate in summaries, acceptance decisions, and comparisons with other tasks.

This task matters to a general weight-space programming framework because it
exercises provenance, supplied information, access restrictions, exact artifact
realization, selection, and missingness together. The same questions arise when
reading a classifier's weights, proposing a controller change, or auditing an
enterprise routing model: which object was supplied, what context was available,
which operation was permitted, which answer or edit was actually returned, and
which claim did the evaluator test? The [selected source review](../weight-space/v2/source-review.md)
and [primary-paper review](primary-paper-review.md) motivate that broader workflow.
The four game policies provide a small source-grounded case for its evidence
contracts. They establish neither business benefit nor a general reader or
editor, production readiness, broad model superiority, or cost advantage. A
correct task label would also leave the proposed causal explanation and any
weight edit to be tested separately.

The retained upstream source is `RikiyaT/lm_understand-logs`, pinned at commit
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. The accompanying author report,
*Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights*, is by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. The
[acquisition record](../../artifacts/research/original-source-repeat/acquisition.json)
identifies all 44 acquired experiment 1 assets with original byte counts,
SHA-256 hashes, Git blob SHA-1 identities, and public source URLs. The archive
retains those bytes unchanged. All four complete experiment 1 checkpoints were
selected on full asset availability and their smaller prompt and weight
footprint than experiment 2, before fresh outcomes were inspected. This is an
availability-based choice of a released task, not a representative sample of
networks or enterprises.

The upstream [LICENSE](../../artifacts/research/original-source-repeat/source/LICENSE)
assigns the experiment 1 logs, data, and networks to CC BY 4.0. The upstream MIT
clause applies to its authored scripts, and its GPL-3.0 clause applies to MinAtar
source embedded in experiment 5; neither clause assigns those terms to this
experiment 1 archive. The framework's MIT license does not relicense the
retained source materials or adapted actor values. The
[conversion notice](original-source-inputs-NOTICE.md) preserves attribution,
license information, and the stated changes. Source-author endorsement of the
conversion, repetition, or framework is not implied. The original paper text
was not newly downloaded or copied into this report.

The source [experiment 1 README](../../artifacts/research/original-source-repeat/source/exp1_which_game/README.md)
describes four PPO policies for the authors' 10 by 10 grid versions of Breakout,
Pong, Pac-Man, and Freeway. Each actor has the same 500 → 32 → 32 → 5 architecture
with tanh hidden activations and 17,253 actor parameters. The supplied prompt
describes the grid, input channels, action meanings, and forward formula, then
provides the actor values rounded to three decimal places. The channels are
agent, moving objects, reward targets, obstacles, and previous frame; the
actions are no-op, up, down, left, and right. No game name or list of possible
games is supplied to the reader. The critic remains outside the reader's input.

Consequently, the task is contextualized structural reading of weights with
known architecture and observation/action semantics. It does not test inference
from an anonymous array stripped of those meanings. One network is supplied in
each fresh conversation and one answer is requested, without execution or a
calculator. The original release requests `gpt-6-astra` with high reasoning
effort through the Codex CLI. Its reported training schedule is approximately
one billion environment steps, recorded as 999,997,440 steps for the chosen
policies. These are upstream training records, not training independently
repeated here.

Three training seeds were tried per game, and the released checkpoint for each
game is the one with the highest mean return in the authors' sanity check.
Breakout and Pong use seed 0; Pac-Man and Freeway use seed 2. Selecting all four
released checkpoints preserves the complete available reading-task set, but it
does not remove that upstream best-of-three selection. The four endpoints
cannot be treated as twelve unselected training replicates, and their released
answers cannot estimate performance on unseen architectures, training
lineages, or hidden tasks. Familiarity with these public source artifacts
cannot be ruled out for either the historical model or a future reader.

The [historical reconstruction](../../experiments/research/original-source-repeat/historical-replay.json)
reads the released answers and reconstructs only their Game-field scores:

| Policy ID | Released truth | Selected seed | Reconstructed Game-field interpretation | Reported confidence | Correct under reconstruction |
| --- | --- | ---: | --- | ---: | --- |
| `a9f535cc` | Breakout | 0 | Breakout, a grid brick-breaking variant | 99% | Yes |
| `dfc6cb65` | Pong | 0 | Pong, a vertical-paddle wall variant | 96% | Yes |
| `bdf99279` | Pac-Man | 2 | Pac-Man or a simplified clone of the same game | 96% | Yes |
| `0bbcc3ea` | Freeway | 2 | Frogger, accepted as Freeway under the released alias rule | 97% | Yes |

The original executable scorer is unavailable. The reconstruction uses keywords
on the Game field alone, accepts Frogger as Freeway as the release describes,
and abstains when the field is missing or identifies conflicting game
alternatives. This alias is part of the scoring convention, not evidence that
Frogger and Freeway have identical mechanics. Descriptions and observations are
not mechanistic truth tests. Confidence is a diagnostic number supplied by the
answer, not a calibrated probability. A correct mapped label therefore does
not validate every sentence of a reply.

The historical record checks that the retained final event answer agrees with
the released answer after ignoring outer whitespace, and observes zero tool
events in those released logs. That is a bounded log observation, not a general
host or network containment guarantee. The released prompts have 115,478,
115,269, 115,394, and 115,259 bytes in frozen policy order. The reconstruction
also preserves upstream-recorded input usage of 73,749, 73,723, 73,723, and
73,695 tokens and wall times of 107.2, 82.7, 70.8, and 70.9 seconds. Those fields
are historical source records, not independently verified current request
usage or prices. The historical reconstruction made zero fresh provider calls.
The separate saved [offline preparation replay](../../artifacts/research/original-source-repeat/offline-replay-20261108-001.json)
records pass, four reconstructed correct labels, complete actor coordinates,
zero fresh calls, no original environment/reward repetition, and no Lean check.

The fresh task is an adapted repetition whose differences were declared in the
protocol before its four calls:

| Aspect | Released historical task | Frozen fresh task |
| --- | --- | --- |
| Requested model and effort | `gpt-6-astra`, high | `gpt-6.1-sol`, low |
| Supplied user prompt | Description, question, rounded actor values | Exact original prompt bytes decoded as UTF-8; no historical answer, truth, or answer key added |
| Answer transport | Markdown reply | Original Markdown reply carried in required JSON `{"answer": "<string>"}` |
| Calls and retry rule | One reply per network in the release | Four assigned calls; no harness retries; 180-second timeout per call |
| System instructions | Released historical instructions | Current CLI system scaffolding is not exported or asserted identical |
| Task access | No tools or calculator | Tool-free CLI configuration, ephemeral isolated proposer directory, read-only sandbox, user configuration disabled |

The declared fresh configuration disables plugins, MCP, web search, shell and
execution tools, apps, agents, skill discovery, images, hooks, browser access,
and memories. The task and harness were authored before trial, with no
midtrial human help. The receipts identify Codex CLI 0.163.0 and bind the local
invocation and prompt. These declarations and observed event checks define the
recorded access boundary; they do not attest OS hermeticity, provider-side
containment, or hidden provider retries. The actual deployment revision was
not exposed and internal model computation was not observed. The changed
requested model, effort, JSON transport, and unasserted system-scaffolding
identity prevent an exact historical model-session reproduction claim even if
future transport succeeds.

The frozen success rule requires completed transport plus one unambiguous
correct Game field, while retaining failed requests in the four-assignment
denominator. The [protocol lock](../../experiments/research/original-source-repeat/protocol-lock.json)
binds 31 preparation input files, the converted actors, historical
reconstruction, bounded arithmetic record, 18 listed project source files and
their frozen counterparts, and runtime identities. The lock itself is the
control artifact for that inventory. The live
[bundle manifest](../../artifacts/research/original-source-repeat/live-20261108-001/manifest.json)
binds the separate 21-file bundle and a global selection gate declaring that
main evaluation outcomes had not been inspected at launch.

| Commitment | Canonical payload SHA-256 |
| --- | --- |
| Acquisition | `7b9fb07e48225e40246fde2e3cd79869a8320ef3ca63341935feb08e4d05d606` |
| Protocol | `5c0015a1b1a5b5d8e612fb733173b60780d633f13c6056e0e2c565ed775b6533` |
| Protocol lock | `a127b29dc0971163ddb109c66dba569168eade67ea700edc516fdc46d41de1fc` |
| Fresh bundle | `88d28359ddc8b1b3bbff75a997739609803c8dfb9c58650c975c5cf31c840fec` |
| Global selection gate | `601897dc212592ae76cb2eed4b56e120de6ed9d66fa292b7e42ba5b924a13d0f` |

The [conversion audit](../../experiments/research/original-source-repeat/conversion-audit.json)
accounts for every actor coordinate. Source matrices use input/output
orientation and are explicitly transposed into the framework's output/input
coordinate convention. Exact source float32 coordinates are lifted into
binary64 for the full-precision artifact, and the released rounded decimal
values are retained in a separate artifact. Released negative rounded zero
tokens are normalized to `0.000`; that canonicalization does not change the
exact original prompt bytes supplied in the fresh task. The actor-only
[converted artifacts](../../experiments/research/original-source-repeat/converted/)
and conversion notice identify these adaptations and exclude the critic.

The [arithmetic correspondence record](../../experiments/research/original-source-repeat/arithmetic-correspondence.json)
compares framework execution with an independent source-orientation reference
using Python `math.fsum` and `math.tanh`. Each of the four actors is checked in
both full-precision and rounded representations on six synthetic binary grids,
at absolute tolerance `1e-10`:

| Policy | Largest source-to-rounded coordinate difference | Negative rounded zeros normalized | Full-precision maximum logit error | Rounded maximum logit error |
| --- | ---: | ---: | ---: | ---: |
| `a9f535cc` | 0.000499947845935822 | 30 | 7.105427357601002e-15 | 7.105427357601002e-15 |
| `dfc6cb65` | 0.000499947845935822 | 30 | 3.552713678800501e-15 | 2.0212997942081756e-14 |
| `bdf99279` | 0.0004998995065689099 | 32 | 8.881784197001252e-15 | 7.105427357601002e-15 |
| `0bbcc3ea` | 0.0004999132752418478 | 35 | 5.329070518200751e-15 | 7.105427357601002e-15 |

The largest numerical discrepancy across those eight checks is
`2.0212997942081756e-14`, below the frozen tolerance. Each representation is
compared with its corresponding reference; the table does not assert that
rounding preserves all logits, selected actions, trajectories, or returns.
Six synthetic grids are a bounded correspondence check, not a proof of
whole-domain floating-point equivalence or a repetition of the original game
environment. The actor has tanh execution support only in this task. The
framework's real-drift and Lean bridge do not certify this actor. No Lean
execution or theorem about its reward behavior is claimed.

The fresh receipts preserve all four attempts in frozen order:

| Policy and public receipt | Started UTC on November 9, 2026 | Captured elapsed seconds | Terminal result |
| --- | --- | ---: | --- |
| [`a9f535cc`, Breakout](../../artifacts/research/original-source-repeat/live-20261108-001/a9f535cc/receipt.json) | 04:12:04.251973 | 3.4675661000073887 | Failed; response null; no credit |
| [`dfc6cb65`, Pong](../../artifacts/research/original-source-repeat/live-20261108-001/dfc6cb65/receipt.json) | 04:12:07.821867 | 2.618603000009898 | Failed; response null; no credit |
| [`bdf99279`, Pac-Man](../../artifacts/research/original-source-repeat/live-20261108-001/bdf99279/receipt.json) | 04:12:10.578282 | 3.1923577000270598 | Failed; response null; no credit |
| [`0bbcc3ea`, Freeway](../../artifacts/research/original-source-repeat/live-20261108-001/0bbcc3ea/receipt.json) | 04:12:13.890054 | 2.712051499984227 | Failed; response null; no credit |

All four have return code 1 and the broad category
`transport_or_response_failed`. The evidence does not establish a network,
authentication, quota, or model-capability cause. The outcome scores contain
null guesses, Game fields, and confidence fields, with missing-field warnings;
these are empty-response bookkeeping, not substantive wrong answers. Four
attempts were assigned and accounted, with no local harness-error rows and no
allowed harness retries. The saved `fresh_calls=4` counts attempted
`provider.call` invocations, not independently attested completed model
executions or all provider-side activity.

For each policy the authenticated runner's event accounting records two error
events, one diagnostic item, and zero unapproved events. Each publication
record declares one redaction. Those counts and the four publication hash
linkages preserve the public derivative's relationship to the disclosed raw
hashes; this report does not copy provider event text or withheld raw transport.
Hashes identify bytes and their declared relationships. They neither
reconstruct withheld bytes nor independently establish provider origin, event
semantics, or host/network containment.

All fresh usage remains missing. The receipts have empty usage objects; the
independent audit represents each missing field as null and counts its
missingness explicitly:

| Fresh usage field | Receipts with an observed value | Receipts missing the value | Sum of observed values, as a lower bound | Exact total | Finite upper bound |
| --- | ---: | ---: | ---: | --- | --- |
| Input tokens | 0 | 4 | 0 | Unknown/null | None/null |
| Cached input tokens | 0 | 4 | 0 | Unknown/null | None/null |
| Output tokens | 0 | 4 | 0 | Unknown/null | None/null |

An observed-value sum of zero does not mean actual usage was zero. A missing
field provides no finite upper bound, and cached input is not added to input
tokens. All four request monetary amounts are USD null/unknown. Their recorded
reason is that the authenticated CLI subscription did not provide request
pricing. The historical source usage cannot fill these fresh gaps, and an
existing subscription does not yield a justified per-request zero price. This
record supports no amortization, ROI, or cost-superiority calculation.

The saved [self-check record](../../artifacts/research/original-source-repeat/self-check-20261108-001.json)
reports pass for Game-field extraction, ambiguity and alternative rejection,
confidence validation, coefficient and nested inventory mutation checks,
destination guards, interrupted four-assignment accounting, and external
global-gate mutations. It records zero fresh provider calls and one mocked
provider invocation. These are saved harness checks; none was rerun to prepare
this report, and a mocked invocation is not a live model answer.

The corrected offline replay and standalone independent audit followed the
final successful read-only preflight. Their public process evidence retains
the following saved results:

| Phase | Started UTC on November 9, 2026 | Child finished UTC | Verification finished UTC | Child / wrapper / recorded shell exit | Captured elapsed seconds | Stdout / stderr bytes |
| --- | --- | --- | --- | --- | ---: | --- |
| Corrected offline replay | 04:41:04.159931 | 04:41:05.681323 | 04:41:06.599577 | 0 / 0 / 0 | 2.4168554000207223 | 259 / 0 |
| Standalone independent audit | 04:43:37.213438 | 04:43:38.739015 | 04:43:38.765537 | 0 / 0 / 0 | 1.5510153999784961 | 487 / 0 |

The replay's exact expected stdout and the audit's exact safe stdout were
verified. Each phase ran once after that final preflight. The replay command
uses the captured interpreter, the frozen runner's `replay-live` operation,
and the expected lock and bundle commitments listed above. The independent
verifier runs in a separate process and uses the standard library without
importing study code. The saved source, command, and exact stdout support its
zero-fresh-provider-call claim; the later public packaging retains that claim
as captured evidence and does not provide an independent OS/network
attestation.

The audit authenticated 131 input files, checked 31 preparation files, the
21-file live bundle, 18 current/frozen project source pairs, and all 44 acquired
assets using SHA-256 and Git blob SHA-1. It checked protocol, acquisition, lock,
bundle, interpreter, control-receipt, and prerequisite commitments; required
all failed responses to be null; independently reconstructed all four empty
Game-field scores; and verified four publication linkages and unknown-usage
accounting. All authenticated inputs were unchanged. Its input-evidence
canonical commitment is
`1841cf35290f9e0c53b6676840408133d676f97f96446deaf96b2591edb7409b`.
The completed main comparison was an authenticated prerequisite with 224
deterministic matches and zero mismatches. The original-source audit opened
zero main evaluation outcome files; it did not repeat main evaluation, main
frozen replay, or main comparison, and those 224 observations are not additional
original-source reading trials.

All seven files in the public verification directory are part of this report's
evidence basis. The following hashes identify their exact released derivative
bytes, separately from each receipt's original-source commitments:

| Public verification file | Purpose | Raw SHA-256 of public file |
| --- | --- | --- |
| [README.md](../../artifacts/research/original-source-repeat/verification/README.md) | Saved result, timing, sanitation, and trust handoff | `492bfd1022d00c4eac9ff4548873472203632a5a30d3de46058a79fb56c84487` |
| [original-replay-exit.sanitized.json](../../artifacts/research/original-source-repeat/verification/original-replay-exit.sanitized.json) | Replay process result, stream commitments, rewritten 131-input inventory | `5688e374dc60d1e88cc6e5f063303728fdba0f89949de947a8eb1be7b0bd2442` |
| [original-audit.sanitized.json](../../artifacts/research/original-source-repeat/verification/original-audit.sanitized.json) | Independent byte/accounting checks, four attempt rows, missing usage, limits | `51a77f2e204441c92eb0afd30c06790aab8966b1d44a5f74906cfa87e3ff5261` |
| [original-audit-exit.sanitized.json](../../artifacts/research/original-source-repeat/verification/original-audit-exit.sanitized.json) | Audit process result, report binding, unchanged before/after inputs | `f63c82f864ecb548fc486f743b7c6dbe2bfe59c0904debb343d77a815657a65b` |
| [original-environment.sanitized.json](../../artifacts/research/original-source-repeat/verification/original-environment.sanitized.json) | Recorded runtime declaration and interpreter/NumPy inventory commitments | `fc6203d3ca273c8f4d20687eef7e637ac7770b0371dac7af8ae1f56dbae84171` |
| [original-prelaunch-history.sanitized.json](../../artifacts/research/original-source-repeat/verification/original-prelaunch-history.sanitized.json) | Preserved first prelaunch failure, later diagnosis, inventory block, final preflight | `64126ca63967dda3c349a8e876b22368c74cf8cdf9b7dfbd26059b787b844e4f` |
| [original-verification-provenance.json](../../artifacts/research/original-source-repeat/verification/original-verification-provenance.json) | Source and export identities, hash conventions, field fidelity and packaging scope | `529fa0ebd0cac8353d848ad7a24269f7ed1610aa380d3a9a1e8d17eb7819e1eb` |

The provenance records 24 checked source byte identities and binds the six
other released verification files. Sanitation uses explicit field whitelists
and documented path/role rewrites. It changes no original capture and performs
no study, verifier, comparator, or provider execution. The replay's 131 input
identities become 120 repository-relative paths and 11 logical roles, retaining
their order, sizes, and raw hashes. Process identifiers and absolute machine
locators are omitted. The original input-evidence digest commits to the
original locator map; the public rewritten inventory has the distinct digest
`c597a347fa450c033104251192ad5fa91499db7760e30e4e5f2adb595de5badd`.
Those hashes intentionally identify different objects.

Raw SHA-256 means the hash of exact serialized file bytes. Canonical payload
hashes use sorted-key compact UTF-8 JSON, `ensure_ascii=false`, no NaN, and omit
only the top-level `sha256` field. A `canonical_json_sha256` hashes the entire
parsed original JSON object, including any existing self-hash. The public
derivative hash, original raw hash, original canonical payload hash, and
rewritten-inventory hash should not be substituted for one another. For
example, the replay exit derivative retains original raw commitment
`8eba0adae7e70ae6fd3401bb74e23e3b51fdae767def4cbbd86cd190b2fe3000`
and original payload commitment
`862fe9cc8fa190c3519c344c288524c22062fea4c50cf3bd4cf0f43f0d31afdd`;
neither is its public raw hash in the table. The audit derivative retains
original raw commitment
`0c5d727aad1a0471d83d3575032f3d663f678551cce0766e155318511b4a51ce`
and original payload commitment
`8aa41cf4f5cdebc4bf557f0553f922a1456074bcf8e8144152df2fc119af726c`.
These anchors identify preserved source evidence without publishing private
locators or captures. No portable original-audit executable is published in
this package. The main study's separate
[portable artifact comparator](../../artifacts/research/live-studies/verification/README.md)
has its own scope and does not replace an original-source audit executable.

The environment derivative records Python 3.12.10 and NumPy 2.5.3 on Windows
AMD64 (`win32`), little-endian finite binary64 arithmetic, and
`C/output-then-input` weight order. It retains interpreter raw SHA-256
`0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14`
and a 428-file NumPy inventory commitment
`c02cb829d7edd63b46243e73c7b28786d167f044aefe45c935715cf6b306f9fc`.
The authenticated runner supplies installed NumPy and runtime import-closure
checks. The independent standard-library audit does not separately rehash the
installed NumPy files or complete runtime import closure, and the public
environment packaging recalculates the saved inventory digest only. It does
not inspect installed packages anew. Interpreter, standard-library and native
runtime correctness, filesystem observations, and local clocks remain trusted.

The independent audit hashes public event streams without independently
parsing their event semantics. For those semantics it relies on the successful
exact stdout of the authenticated runner. Withheld raw events and provider
origin remain independently unverified. These are material trust limits of the
successful offline check, rather than missing fresh answers that can be filled
by an audit. Source authentication, same-implementation repeatability,
independent accounting, model-session provenance, and scientific reproduction
have different evidential scopes.

The preserved prelaunch history is part of the result. The first independent
wrapper invocation exited 1 before child launch. Its safe recorded output was
`{"exception_type":"ValueError","status":"blocked"}`, and all five expected
phase capture files were absent afterward. The block record was written
retrospectively; exact failed-invocation start and finish timestamps are
unavailable, while the shell-reported elapsed duration was approximately 1.2
seconds. Its earlier `diagnosis_pending=true` field is retained, with the later
diagnosis separately labeled rather than rewritten into that earlier record.

The later diagnosis found differences between Windows `Path.stat()` and
`os.fstat()` observations of ctime and executable permission bits. Only the
private audit helpers were corrected. Cross-API comparisons now use device,
inode, file type, size, and mtime, while complete same-API before/after metadata,
regular-file requirements, and both no-link checks remain. Archived first and
final helper identities are preserved in the history and provenance. The
correction did not edit frozen study source or evidence.

A subsequent read-only preflight stopped on an extra README inside the strict
preparation inventory. The conversion notice was moved byte for byte outside
that inventory to its current documentation path. The history records the
move-time raw commitment
`e579f9b54d1c1ed2356c775ab81a0016cbd580a58163707cb88751739174c5d2`;
that is a historical identity at the move, not a claim about its current bytes
after documentation links were repaired. This inventory-only block was not
another replay launch or provider attempt. There is no README inside the
frozen preparation directory. The final preflight passed from November 9,
2026, 04:40:50.569132 to 04:40:51.459309 UTC, authenticating 131 inputs and four
terminal attempts with no subprocess or provider calls. The corrected replay
and audit then each ran once. Earlier parent replay evidence and both prelaunch
blocks remain preserved; later success does not erase them.

The [existing acceptance map](issue-acceptance-status.md) remains the separate
criterion-by-criterion assessment. This report adds an explanation of the
original-source evidence; it changes no original criterion, checkbox, source
record, frozen preparation input, live bundle, or issue state. In particular,
successful offline authentication cannot support a fresh-answer accuracy
claim, an original-training reproduction claim, an actor certificate outside
its backend, or a general capability and business-value claim. Results from
different reading, editing, construction, or behavioral tasks should not be
pooled into a single success rate.

A concrete next protocol should first resolve the transport uncertainty on a
non-study input, with a separately identified diagnostic record. That check
should establish completed response transport and schema handling before
spending or interpreting another four-policy task budget. It should record
observed causes rather than infer them from this broad failure category. The
present lock and failed bundle remain immutable. Any changed model, system
instructions, transport schema, access controls, timeout, retry policy, or
scorer requires a new frozen amendment and run identity before study calls.

The new study protocol should bind the requested model and effort, the actual
deployment revision if exposed, CLI identity, full available instruction
identity, exact prompt and schema bytes, supplied metadata, and declared versus
enforced access. It should define the Game-field parser, ambiguity and alias
rules, whether confidence is collected, the success denominator, completion
criteria, retry limit, and failure retention before launch. Each assigned
attempt should retain response status, elapsed time, observed tool events,
publication transformations, token-field missingness, and request cost or its
explicit unknown status. Failed attempts must stay in the record even if a
later attempt succeeds. Obtaining a portable independently reviewed audit
executable and stronger provider provenance would improve external verification;
neither is supplied by the present derivative receipts.

For general reading ability, a separate expanded study should freeze unseen
training lineages and architectures, preserve unselected seeds and checkpoint
selection rules, identify independent units, and keep task families separate.
Matched metadata-only, weight-only, and combined-context arms can test how much
identification comes from the architecture and observation/action description.
Declared weight or metadata ablations and familiarity controls would be needed
to assess what the numerical values contribute. Sample size and uncertainty
should match that expanded claim rather than treating these four public
checkpoints as a general estimate. An enterprise application would additionally
need a real target behavior, retention measures, a suitable unchanged-model or
policy-level baseline, and observed cost before claiming useful deployment
benefit.

Original training and reward reproduction require their own protocol with
source environment implementation, dependencies, seeds, PPO settings,
checkpoint-selection procedure, action sampling, and trajectory/return
evaluation. Mechanistic validation requires preregistered interventions or
other tests that distinguish the proposed explanation from matched controls.
Numerical and formal guarantees require backend-specific assumptions and
checks. These are proposed next phases, not executions completed by this
report.
