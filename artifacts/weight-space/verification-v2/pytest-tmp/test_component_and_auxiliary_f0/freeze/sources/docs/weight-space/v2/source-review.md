# Source review for a generic weight-space programming framework

Reviewed November 6, 2026, America/Chicago. This review reads original sources and saved evidence; it does not run an editor, train a model, reconstruct a large checkpoint, or repeat an experiment.

The released work supports treating weight-space programming as a family of tasks: identifying a model, diagnosing a mechanism, designing weights, and repairing behavior under different access rules. It does not support collapsing these tasks into one edit-success measure. A generic framework needs explicit task semantics, access and budget records, candidate realization, and separate mathematical and empirical evidence. A parameter-edit format is a useful foundation, but it does not by itself supply these distinctions.

## Sources and review boundary

The original is Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab's **Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights**, published November 2, 2026. It is an author report/blog, not a peer-reviewed paper. This review distinguishes the authors' interpretations, observations in their released records, and requirements proposed for this repository. [Report]

The accompanying repository is RikiyaT/lm_understand-logs, pinned here to commit **4d09ba6d6a7e891dd8063974dfe81b3d78e3494f**. The [archive manifest][Manifest] records 77 successful retrievals: the report HTML, the pinned repository-root HTML, and 75 selected release files. Every retrieved file's byte count and SHA-256 were checked. The report and all 13 release files previously listed in [references.json](../references.json) match their earlier hashes. The report hash is a326bc5b34b3152a93bf38013f412bb140c9e0a299c72491d27fb17b1b7d9cbb. [Evidence]

All experiment READMEs and the report were read; selected prompts, results, notices, manifests, and implementation files received further inspection. Downloading a file is not a claim to have audited all of its contents. The evidence companion records per-file review scope. Raw source bytes are preserved; article-text.txt is a separately identified reading aid. Retrieval timestamps are November 7 in UTC and November 6 in the local timezone. [Manifest] [Evidence]

There are seven recorded acquisition failures: five initial incorrect path guesses returned 404, and the recursive-tree and commit-metadata API requests returned 504. The successful root HTML verifies the pinned object ID and shows eight experiment directories, harness, README.md, and LICENSE. Consequently, this is a selected-artifact review, not a complete recursive repository audit. The old references file's commit timestamp was not independently reverified in this pass. [Manifest] [Tree]

No separate limitations or next-steps document was found in the inspected release root or the bounded local repository scan. The actual original discussion is the report's **What Comes Next?** section and the experiment READMEs' setup, results, and omitted-artifact sections. The local limitations are in the existing design, research, results, project-memory, and formal-scope sections identified below. This review does not invent documents to stand in for those sources.

## What the tasks actually expose and measure

The following are observations from the released protocols and records, not independent replications. “No tools” describes an editor interface; “no target forward pass” is a different restriction.

| Task | Information and permitted interaction | Reported evidence and its unit |
| --- | --- | --- |
| 1. Identify a game's policy | Rounded weights of a 500→32→32→5 tanh policy; architecture, observation channels, action meanings, and forward formula; no execution. | Four checkpoints, each selected as the best of three training seeds. Four labels scored correct; Frogger was accepted for Freeway. Four successes do not calibrate the reported confidence. [E1] [E1Scores] |
| 2. Identify a diffusion model's output | Three 12×12 generators; architecture, pixel meaning, noise schedule, and training objective; 23,824 rounded values per model; no execution. | Three shape categories correct. All three descriptions call the shapes hollow or outlines, while the README describes filled training targets. Category accuracy and output-description fidelity differ. [E2] [E2Scores] [E2Key] |
| 3. Identify language and attention-head roles | Exact float32 middle-block weights, forward code, training procedure, and three candidate corpus descriptions; embeddings/output omitted; Python analysis of weights, but no layer or network forward computation. | Three independently trained checkpoints, two conversations each: six language answers correct. All six training-order answers concern unshuffled corpora. Head accuracy depends on a declared induction-score threshold. [E3] [E3Prompt] [E3Scores] |
| 4. Repair the Kinetix car | Scene and rules, observation layout, actor formula, rounded weights, a failed rollout, and base evaluation; one reply without tools or feedback. | One editor proposal, 51 edit statements, 6,073 changed values among 171,664 actor values, measured from the rounded shown base. The result records 1,024/1,024 successes; the base record is 0/256. Episodes are repeated evaluations of one edit. [E4] [E4Result] |
| 5. Synthesize MinAtar policies | Zero-initialized 600→32→32→6 tanh network; exact game source, rules, observation/action semantics, and scoring; one reply per game without execution or feedback. | One written policy per game. Mean rewards versus the best 500M-step PPO seed: Breakout 517.411 versus 25.344; Seaquest 53.110 versus 88.884; Asterix 53.178 versus 55.204. Each value summarizes 1,000 episodes. [E5] [E5Prompt] [E5Eval] |
| 6. Synthesize a Go policy | Zero-initialized 1,480,274-value policy with six hidden ReLU layers; game rules and interface; Python weight construction; 0/10/25/50 target-pass budgets. | One editor run per budget. At zero feedback the policy wins 54.1/54.9% against small PPO opponents, but 42.6/26.3/36.3% against tuned same-size PPO seeds. At 50 passes the latter comparison is 74.2/70.0/78.5%. These are different baselines, not interchangeable claims. [E6] [Go0] [GoSameSize] |
| 7. Identify sycophancy and nominate patches | Three matched fine-tune pairs; weights-only rounds vary embedding/tokenizer access; another arm exposes attention from requested texts. The prompt describes how the fine-tunes differ. | Weights-only rounds 1–3 return P(A)=0.5 with forced A labels; round 4 is incomplete. Attention access gives 3/3 classification. The inspected causal patch records do not show successful behavioral repair. [E7] [E7Prompt] [E7Attention] [Patch1] [Patch2] [Patch3] |
| 8. Repair Qwen math behavior | Weights, tokenizer, labeled practice problems and solutions, Python, generation/teacher-forced feedback, activations, and explicit diagnosis guidance; hidden test items but disclosed evaluation format. | At 50 charged passes, MATH-500 rises 7.9→48.1%, GSM8K 13.8→67.1%, and IFEval 21.0→62.3%. The final patch changes 130,188 bf16 values. This is an adaptive, informed repair experiment. [E8Prompt] [QwenBase] [Qwen50] [QwenManifest] |
| 8 follow-up. Improve Gemma math | Similar adaptive access, a different model and decoding/grading protocol, 5/10/25/50 passes, and explicit DEV/HOLDOUT instructions. | Raw format-agnostic MATH-500 accuracy is 6.8% base and 7.05/7.6/7.7/7.0% after edits. The release presents no established improvement. [Gemma] [GemmaBase] [Gemma5] [Gemma10] [Gemma25] [Gemma50] |

These are heterogeneous tasks with heterogeneous evidence units. Four games, three generators, three independently trained language checkpoints, three fine-tune pairs, and many samples from one final edit must not be added into a common “number of successful experiments.” Likewise, repeated generations or rollouts reduce some measurement uncertainty while leaving variability across editor runs, model lineages, and task families largely unmeasured.

## Critical findings

### Supplied semantics and hints are part of the task

The small-model demonstrations use contextualized weights. Game-channel meanings and actions are given; diffusion targets come with image and training semantics; the language experiment gives three candidate corpus descriptions. The synthesis prompt even supplies exact MinAtar source. These conditions can enable useful structural reasoning, but they do not establish identification from arbitrary anonymous arrays. [E1Description] [E2Description] [E3Prompt] [E5Prompt]

For Qwen, the first prompt explicitly suggests repetition as a repair target, describes causal localization and strength sweeps, names MATH-500/GSM8K, and supplies the grading and decoding protocol. It also exposes 7,469 GSM8K, 7,275 MATH, and 714 AIME practice problems, with reference solutions where available. Hidden evaluation items are a meaningful boundary; they do not mean the evaluator specification or labeled practice data is hidden. A generic task record should enumerate these inputs instead of using an unqualified “weights only” or “zero data” label. [E8Prompt]

One useful control is therefore to hold the task fixed while varying access to weights, semantic metadata, labels, activations, and target feedback. Such a control is a proposed experiment, not a result of this review. The language study's all-unshuffled cases also leave the shuffled-text discrimination task untested in the inspected release. [E3]

### Identification, mechanistic diagnosis, and causal repair are separate claims

The diffusion descriptions illustrate partial success: the label is right while an important visual property is wrong. The README describes filled targets and explicitly notes the outline error; the answer key supplies shape labels without a filled/hollow field. A single category-success flag hides this distinction. [E2] [E2Scores] [E2Key]

Sycophancy provides stronger negative evidence. All nine Astra gap_closed_h2s values in the three weights-only patch-score files are 0.0, although their continuous margin-closure values are nonzero. The attention arm's inspected pair-1 causal-check log also reports zero behavioral-gap closure; its margin closure is 0.04, versus 0.08 for largest-change components and 0.19 for the random-control mean. Thus correct pair identification and an attention-based explanation are not evidence of a successful repair in these checked records. Zero classification-gap closure does not mean no measurable continuous effect. This does not claim that every possible patch was tested. [Patch1] [Patch2] [Patch3] [AttentionPatch]

The Gemma 50-pass final answer itself reports a HOLDOUT gain of 0.421 percentage points over 128 problems, with 16,384 answers per model, problem-cluster SE 0.365 points, and approximate 95% interval [−0.300, +1.143]. Cutoffs worsen from 7.31% to 10.50%. The interval crosses zero; the large answer count does not create 16,384 independent problems. Its prompt requires submitting the best edit even if insignificant and forbids restoring the original, so a submitted patch must not automatically count as a useful outcome. [GemmaAnswer] [GemmaPrompt]

### Metric definitions and incomplete attempts belong in the record

The language README explicitly discloses a threshold mismatch: the post uses induction score >0.2, whereas true_copy_heads in scores.json uses >0.3. Python scores 0.245, 0.269, and 0.546 and the romaji score 0.220 change inclusion under those cutoffs. The correct response is to bind the raw values, threshold, and scorer version to each result, not silently choose the more favorable one. [E3] [E3Heads] [E3Scores]

Sycophancy rounds 1–3 choose A at exactly 0.5, yielding nominal one-third accuracy by a forced tie-break. Round 4 was stopped at 29–30 runs per pair; only pair 2 completed. It should not be encoded as three complete trials, nor should uncertainty be translated into confident wrong predictions. [E7]

Gemma uses a format-agnostic extraction rule and also retains strict boxed-answer scores; Qwen uses its own specified rule and sampling settings. These are separate evaluator configurations. The main release also explains that the figure's 24.5% base GPQA figure was an interim subset, whereas the final merged base score is 22.7%. Final, interim, strict, and lenient results require explicit identities. [Gemma] [E8] [QwenBase]

### “No training,” “tool free,” and “few passes” need operational definitions

The Go protocol forbids gradients and fitting weights to examples, including regression and least squares. Writing weights from formulas and analyzing feedback are allowed. A constrained least-squares compiler may be a useful framework proposer, but it belongs to a different experimental arm from this no-fitting protocol. Qwen's adaptive weight editing is also not a no-feedback experiment merely because it prohibits gradient training. [GoPrompt] [E8Prompt]

The root audit's 1,125 tool-free turns refer to built-in Codex tools. Python reply blocks and external target tools still execute in experiments 3, 6, 7, and 8. The inspected sandbox provides process/network/resource isolation and validates saved tensor shape, float32 conversion, and finiteness; it is not a general semantic proof that arbitrary Python performs no forbidden forward computation or fitting. Record declared restrictions, enforced capabilities, and compliance-audit evidence separately. [Root] [Sandbox] [E3] [E6] [E7] [E8]

A Go pass may contain 128 games or 256 positions; a position call can expose all hidden activations. A Qwen generation pass may return 512 answers, or a teacher-forced pass may read 1,024 texts totaling up to 524,288 tokens. A Gemma generation pass permits 2,048 answers. Equal pass counts are therefore not equal compute, feedback, or information budgets. Qwen's pass-size notice expressly says duration and batch size are not charged separately. [GoPrompt] [E8Prompt] [GemmaPrompt] [PassNotice]

Qwen p50c's final state records 80 Python runs, 69 requests, 50 charged passes, and three handoffs. The Go release retains disk-quota failures, and the attention experiment describes retries after a missing-package failure that the editor never saw. Count attempted and completed work, retries, hidden infrastructure work, and editor-visible feedback separately. Also preserve the actual observed output: the inspected Go sandbox returns up to 20,000 characters but its tool-call log saves only the first 6,000. [QwenState] [E6] [E7] [Sandbox]

### Exact candidate realization matters

The Kinetix edit count is relative to the rounded network shown in the prompt, not silently interchangeable with the full-precision trained checkpoint. MinAtar Seaquest and Asterix replies were reparsed after a whitespace-handling fix; no new editor reply was generated. Proposal text, parser version, parsed program, parent checkpoint, and realized candidate should have separate identities. [E4] [E5]

Qwen p50c touches two 64×1,024 column blocks, or 131,072 entries, but only 130,188 stored bf16 entries differ. Its released reconstruction checks parent tensor dtype, shape, and hash, then verifies each reconstructed edited tensor hash. The checkpoint builder casts submitted values to the original tensor dtype. For p10c, low-rank reconstruction includes correction entries for rounding; its manifest records 30 changed tensors and 72,986,982 effective changed entries. A compact low-rank program is not necessarily a sparse stored change. [QwenManifest] [ApplyEdit] [BuildCheckpoint] [NativeConversion] [Qwen10Manifest]

This motivates explicit native-dtype projection and endpoint verification, with distinct touched-coordinate and effective-change counts. It does not imply that changing few entries preserves unrelated behavior. Nor does a native-byte reconstruction prove numerical equivalence across execution backends or establish a real-valued bound.

### Baseline selection and omitted history limit generality

The early game-identification checkpoints are selected training seeds, MinAtar shows best-seed PPO comparisons, and Go reports both smaller and tuned same-size opponents. Go has one editor run at each budget; a one-pass run and another prompt version are omitted. These are informative demonstrations, but they do not estimate editor variance or establish a causal scaling law in the number of passes. [E1] [E5] [E6]

Qwen and Gemma also disclose other prompt versions/runs outside the reported subset. Qwen's notices change tool availability and clarify non-reproducible batching; a later notice corrects failure counts that previously classified wrong stated answers as missing answers. Protocol revisions, rejected attempts, selection criteria, and interventions by experimenters belong in a session's history. A final patch plus a headline score cannot substitute for that history. [E8] [Gemma] [ToolNotice] [FailureNotice]

## Reproducibility is a set of distinct obligations

| Level | What the inspected sources support | What this review does not establish |
| --- | --- | --- |
| Inspect retained bytes and records | Pinned source files, exact archive hashes, reported scores, prompts, notices, and selected code can be checked. | The truth of every logged event or the completeness of every experiment. |
| Reconstruct a final candidate | The release inventories final small-model assets; Qwen/Gemma provide diff-based reconstruction against pinned public bases with tensor-hash checks. | This review did not download all model bodies or execute reconstruction. A recipe is not a completed reproduction. |
| Repeat final evaluation | Several evaluators and protocol details are supplied. | External packages, datasets, hardware, omitted graders, and stochastic execution still require a separately documented run. |
| Replay the editor session | Many prompts, replies, and summaries are retained. | Omitted observations, intermediate states, harness code, and unreported runs prevent a blanket full-replay claim. Provider behavior is another dependency. |
| Independently repeat the scientific result | A new run could use a frozen protocol, independent editor trials, and untouched lineages/tasks. | No such experiment was performed here. The report credits a third-party Qwen rebuild/evaluation; that work was not independently inspected in this review. |

The missing pieces are concrete. The language experiment releases only middle blocks; its Python-corpus reconstruction recovers 81.5% of characters exactly while 18.5% remains unidentified. MinAtar PPO checkpoint bodies are absent. Go omits intermediate full networks and per-game boards/valid masks from feedback passes. The sycophancy checkpoint bodies are absent, and its rebuild description warns that GPU bf16 training is not bit-reproducible. Most attention arrays are omitted, with a specific figure example retained. Qwen omits work directories, its agent harness, some imported graders, and other prompt-version runs. A release can be highly useful while failing stronger notions of replay completeness. [E3] [PythonCorpus] [E5] [E6] [E7] [E8]

## Actual limitations and next steps in the original report

In **What Comes Next?**, the authors identify context-length and harness inefficiency. They propose auditing without evaluation, reducing optimization cost, zero-data steering/unlearning, recursive self-improvement, and grafting circuits across architectures; they also say they are building a benchmark and harness. These are research directions. The demonstrated repair results do not prove backdoor detection, successful unlearning, absence of catastrophic forgetting, recursive self-improvement, or cross-architecture transfer. The inference from sparse editing to preservation needs independent justification. [Report]

The Gemma follow-up is particularly important because it tests a model where looping is less prevalent and does not establish a gain. It supports keeping failure and abstention as first-class outcomes. It does not prove that weight editing can never improve an initially absent ability: that universal conclusion would go beyond these tasks, models, and budgets. [Gemma] [GemmaAnswer]

## Prior documented boundaries in this repository

The existing framework already has generic named tensors, atomic exact-parent edits, structural admission, and caller-supplied execution. It would be inaccurate to describe those facilities as absent. The following are the **prior documented boundaries read during this review**, not claims that v2 additions have already solved them.

| Existing source and exact section | Boundary relevant to v2 |
| --- | --- |
| [README: Understand the guarantees](../README.md#understand-the-guarantees) | Generic admission, real-valued certificates, formal theorems, and observed outcomes establish different things. The documented first proof family is two-layer ReLU. |
| [Design: Readout compilation and its limits](../design.md) | The existing compiler uses least squares with soft retention. Exact anchor constraints are a distinct extension; anchors alone do not characterize all future feature rows. The floating solver is not a compiler-correctness theorem. |
| [Design: Caller-supplied PyTorch modules](../design.md) | Snapshot metadata does not capture arbitrary forward-code meaning, hidden mutable state, RNG, hooks, or external resources. Exact tied aliases and native dtype behavior matter. |
| [Design: Proof boundary and evaluation boundary; Use and scale](../design.md) | Tensor schemas do not supply architecture semantics. Dense binary64 JSON, full scans/copies, rational arithmetic, and dense alignment/compilation have explicit scale limits. |
| [Research: Follow-up hypotheses and Project-specific interpretation and remaining gaps](../research.md) | Generality, cumulative edits, alignment transfer, and fresh task utility require new protocols and whole-lineage/task splits. |
| [Results: Reproduce or extend](../results.md#reproduce-or-extend), plus the preceding provenance paragraph | Initial and most final RTS checkpoint bodies are absent. Observed seeds are reproduction inputs, not untouched confirmatory data. |
| [Project memory: Follow-up research and existing applications](../../PROJECT_MEMORY.md#follow-up-research-and-existing-applications) | Existing app policies have different feature and decision semantics; their claims cannot be transferred without an adapter. No broad operator-superiority or Astra proposal-quality result is established. |
| [Formal README: Semantic scope and trust boundary](../../../formal/weight-space/README.md#semantic-scope-and-trust-boundary) | Theorems concern explicit finite real-valued networks. They do not prove deployed floating execution, simulator reachability, reward gains, or a different architecture's equivalence. |

The evidence companion records local source hashes and line/section locators. The bounded scan did not find a standalone LIMITATIONS.md, NEXT_STEPS.md, or roadmap for this subsystem. The original frozen study, its failures, and its disclosed missing checkpoint bodies should remain preserved rather than rewritten to match a new framework.

## Requirements derived from the review

These are design and acceptance requirements, not empirical findings or an assertion that every item is already implemented.

| Source gap or ambiguity | Required framework behavior | Reviewable acceptance evidence |
| --- | --- | --- |
| Reading, diagnosis, synthesis, and repair have different outputs | A TaskSpec names the task kind, claim, model/interface, allowed observations, and evaluator. | A classification candidate can be assessed without pretending it is a weight patch; a diagnosis does not inherit repair success. |
| Hints, labels, activations, and evaluator knowledge vary | Record supplied information, permitted operations, hidden-evaluation ownership, and declared versus enforced restrictions. | A receipt enumerates actual observation types and the responsible enforcement/audit boundary. |
| Real examples use deeper and mixed architectures | Separate tensor representation, executable semantics, and architecture-specific certificate backends. | Unsupported semantics produce an explicit unsupported/unchecked result rather than borrowing a ReLU theorem. |
| Dense, slice, sparse, and low-rank programs realize differently | Bind source proposal, parser/compiler version, exact parent, ordered operations, dtype conversion, and final tensor hashes. | Replaying a candidate verifies its realized endpoints; projection loss and effective change counts are visible. |
| Success at one level does not imply another | Use separate structural, mathematical, numerical, behavioral, generalization, and reproduction evidence gates. | An unchecked or failed gate remains visible after another gate passes. |
| “Pass” hides work and information | Keep tokens/bytes/scalars, sandbox attempts, target calls, examples/samples/tokens per call, target compute, timing, retries, and human selection. | Budget comparisons state their denominator and retain failed/unbilled work. |
| Selection and protocol changes are consequential | ProgrammerSession retains candidates, failures, notices, access events, and parent/child lineage. | The selected Candidate resolves to an exact proposal and realized checkpoint within the retained population. |
| Thresholds, grading, ties, and incomplete runs change results | EvaluationReceipt binds evaluator/configuration, raw metric units, completion status, uncertainty unit, and claim scope. | Recompute displayed metrics from retained values without silently changing thresholds or counting an incomplete trial as complete. |
| Many samples share one problem, editor, or checkpoint | Separate editor trials, training lineages, tasks, rollout seeds, problems, and samples. | Splits and uncertainty calculations match the claimed independent unit. |
| Sparse edits can still cause broad regressions | Require retention and task utility evaluation appropriate to the claim; always allow the unchanged baseline and no established gain. | A valid patch can fail behavioral admission or be rejected by final selection. |
| Final bytes and full replay are different deliverables | Declare reproduction tiers and an explicit inventory of missing inputs. | A hash-verifiable diff is labeled candidate reconstruction, not full-session replay. |

The intended generic workflow is **TaskSpec → ProgrammerSession → Candidate → EvaluationReceipt**. Its purpose is to make the experiment and claim inspectable across proposer types. It should admit an LLM, an analytic construction, or a fitting algorithm while recording which mechanism was used. A workflow data model alone is not a security sandbox, a hidden-test service, or a proof that a declared rule was obeyed.

## Testable follow-up work

The following agenda is proposed; no item was run for this review.

1. **Access ablations.** On new model lineages and tasks, compare semantic metadata alone, weights plus semantics, and progressively richer feedback at matched declared budgets. Include unfamiliar task variants and explicit abstention. This separates weight information from supplied rules and hints.
2. **Mechanism and repair controls.** Freeze a causal prediction before intervention; compare the proposed edit with matched random, magnitude, opposite-direction, and restoration controls. Report target behavior and retention on untouched units. For loop repair, measure finishing, correctness, and reasoning errors separately.
3. **Native realization checks.** Exercise serialization, underflow/overflow, rounding, aliases, parent mismatches, and parser revisions on small fixtures. Compare the proposed arithmetic with the actual stored candidate before issuing any certificate or evaluation receipt.
4. **Sequential edits.** Freeze per-step selection rules, preserve complete lineage, and recheck cumulative contracts and held-out retention after every edit. Individually admissible edits need not yield an admissible composition under the same overall budget.
5. **Architecture and scale extensions.** Add explicit semantics and trusted-boundary descriptions for each supported backend. Measure storage, copying, compilation, and certificate cost before claiming large-model applicability; chunked storage and compact programs need their own correctness and performance evidence.
6. **Independent repetitions with full populations.** Freeze new protocols and untouched evaluation units, repeat editors and checkpoints, retain every attempted candidate and failure, and publish unconditional joint success as well as success among admitted edits. Keep the unchanged baseline eligible and preserve negative results.

[Report]: ../../../artifacts/weight-space/source-review-v2/article.html
[Root]: ../../../artifacts/weight-space/source-review-v2/release/README.md
[Manifest]: ../../../artifacts/weight-space/source-review-v2/provenance.json
[Evidence]: ../../../artifacts/weight-space/source-review-v2/review-evidence.json
[Tree]: ../../../artifacts/weight-space/source-review-v2/repository-tree.html
[E1]: ../../../artifacts/weight-space/source-review-v2/release/exp1_which_game/README.md
[E1Scores]: ../../../artifacts/weight-space/source-review-v2/release/exp1_which_game/eval/scores.json
[E1Description]: ../../../artifacts/weight-space/source-review-v2/release/exp1_which_game/networks/description.txt
[E2]: ../../../artifacts/weight-space/source-review-v2/release/exp2_diffusion/README.md
[E2Scores]: ../../../artifacts/weight-space/source-review-v2/release/exp2_diffusion/eval/scores.json
[E2Key]: ../../../artifacts/weight-space/source-review-v2/release/exp2_diffusion/eval/answer_key.json
[E2Description]: ../../../artifacts/weight-space/source-review-v2/release/exp2_diffusion/networks/description.txt
[E3]: ../../../artifacts/weight-space/source-review-v2/release/exp3_language/README.md
[E3Prompt]: ../../../artifacts/weight-space/source-review-v2/release/exp3_language/runs/python_r0/message.txt
[E3Scores]: ../../../artifacts/weight-space/source-review-v2/release/exp3_language/eval/scores.json
[E3Heads]: ../../../artifacts/weight-space/source-review-v2/release/exp3_language/eval/heads.json
[PythonCorpus]: ../../../artifacts/weight-space/source-review-v2/release/exp3_language/python_dataset/stats.json
[E4]: ../../../artifacts/weight-space/source-review-v2/release/exp4_fix_car/README.md
[E4Result]: ../../../artifacts/weight-space/source-review-v2/release/exp4_fix_car/eval/oneshot_result.json
[E5]: ../../../artifacts/weight-space/source-review-v2/release/exp5_write_from_scratch/README.md
[E5Prompt]: ../../../artifacts/weight-space/source-review-v2/release/exp5_write_from_scratch/runs/seaquest/message.txt
[E5Eval]: ../../../artifacts/weight-space/source-review-v2/release/exp5_write_from_scratch/eval/minatar_eval.json
[E6]: ../../../artifacts/weight-space/source-review-v2/release/exp6_go_from_scratch/README.md
[GoPrompt]: ../../../artifacts/weight-space/source-review-v2/release/exp6_go_from_scratch/runs/budget_50/message.txt
[Go0]: ../../../artifacts/weight-space/source-review-v2/release/exp6_go_from_scratch/eval/budget_0/eval_submission.json
[GoSameSize]: ../../../artifacts/weight-space/source-review-v2/release/exp6_go_from_scratch/eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json
[E7]: ../../../artifacts/weight-space/source-review-v2/release/exp7_sycophancy/README.md
[E7Prompt]: ../../../artifacts/weight-space/source-review-v2/release/exp7_sycophancy/runs/attention_tool/prompt.txt
[E7Attention]: ../../../artifacts/weight-space/source-review-v2/release/exp7_sycophancy/eval/attention_tool/score.json
[Patch1]: ../../../artifacts/weight-space/source-review-v2/release/exp7_sycophancy/eval/weights_only_1/patch_score.json
[Patch2]: ../../../artifacts/weight-space/source-review-v2/release/exp7_sycophancy/eval/weights_only_2/patch_score.json
[Patch3]: ../../../artifacts/weight-space/source-review-v2/release/exp7_sycophancy/eval/weights_only_3/patch_score.json
[AttentionPatch]: ../../../artifacts/weight-space/source-review-v2/release/exp7_sycophancy/runs/attention_tool/ask_devemb_s2.log
[E8]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/README.md
[E8Prompt]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/runs/p50c/turns/t001/prompt.txt
[QwenBase]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/eval/unedited/merged.json
[Qwen50]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/eval/p50c/merged.json
[QwenManifest]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/runs/p50c/edit/manifest.json
[Qwen10Manifest]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/runs/p10c/edit/manifest.json
[QwenState]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/runs/p50c/state.json
[ApplyEdit]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/code/apply_edit.py
[BuildCheckpoint]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/code/build_ckpt.py
[NativeConversion]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/code/st_numpy.py
[Gemma]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/gemma_followup/README.md
[GemmaPrompt]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/gemma_followup/runs/g25b/turns/t001/prompt.txt
[GemmaAnswer]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/gemma_followup/runs/g50b/final_answer.md
[GemmaBase]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/gemma_followup/eval/unedited/merged.json
[Gemma5]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/gemma_followup/eval/g05b/merged.json
[Gemma10]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/gemma_followup/eval/g10b/merged.json
[Gemma25]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/gemma_followup/eval/g25b/merged.json
[Gemma50]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/gemma_followup/eval/g50b/merged.json
[PassNotice]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/notices/001_pass_size.txt
[ToolNotice]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/notices/002_tool_facts.txt
[FailureNotice]: ../../../artifacts/weight-space/source-review-v2/release/exp8_math/notices/004_failure_counts.txt
[Sandbox]: ../../../artifacts/weight-space/source-review-v2/release/harness/sandbox.py
