# Source review for a generic weight-space programming framework

Can an LLM use a network's weights to identify what it does, explain a mechanism, or make a useful change? The original report offers examples of all three, under different access rules. Those examples require different tests. A correct label does not establish a causal explanation, and a plausible explanation does not establish a successful repair.

This review asks what a framework must record to make those claims assessable. Its answer is concrete: record the information supplied to the editor, the exact candidate produced, the comparison used, and the unit over which the result varies. Keep target behavior, retained behavior, and candidate reconstruction as separate outcomes.

The review used saved original sources on November 6, 2026, America/Chicago. It did not run an editor, train a model, reconstruct a large checkpoint, or repeat an experiment. Statements about released scores below are observations of those records. The authors' explanations and this repository's proposed work are identified separately.

## Sources and review boundary

The source is Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab's **Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights**, published November 2, 2026. It is an author report/blog, not a peer-reviewed paper. The accompanying repository, `RikiyaT/lm_understand-logs`, is pinned to commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. [Report] [Tree]

The local archive preserves the report, the pinned repository-root HTML, and selected release files. The report and every experiment README were read. Selected prompts, results, notices, manifests, and code received further inspection. The [evidence companion][Evidence] records that scope per file; a downloaded file is not necessarily a fully reviewed file. The [manifest][Manifest] records byte identities and acquisition outcomes. Raw source bytes remain separate from `article-text.txt`, a reading aid.

The archive has seven acquisition failures: five incorrect path guesses returned 404, and two API requests—for the recursive tree and commit metadata—returned 504. The successful root HTML verifies the pinned object and its listed experiments. These limits make this a selected-artifact review, not a complete repository audit. The old [references file](../references.json)'s commit timestamp was not independently reverified. Retrievals dated November 7 in UTC occurred on November 6 locally. [Manifest] [Tree]

The checked source bytes match the archive's hashes and the prior reference records where both exist. The report's raw SHA-256 is `a326bc5b34b3152a93bf38013f412bb140c9e0a299c72491d27fb17b1b7d9cbb`. These identities let a reader inspect the same material; they do not establish that each reported event occurred or that omitted trials were unimportant. [Evidence]

No separate limitations or next-steps document appeared in the inspected release root or bounded local scan. The original discussion is the report's **What Comes Next?** section and the experiment READMEs. The local limitations are in the design, research, results, project-memory, and formal-scope sections linked below.

## What the tasks actually expose and measure

The identification tasks give weights with substantial context. “No tools” describes an editor interface; “no target forward pass” describes a different restriction.

| Task | What the editor received | Observation in the released records |
| --- | --- | --- |
| Identify a game's policy | Rounded weights of a 500→32→32→5 tanh policy, its forward formula, observation channels, and action meanings; no execution. | Four checkpoints, each the best of three training seeds. All four labels were accepted, including Frogger for Freeway. This does not calibrate the reported confidence. [E1] [E1Scores] |
| Identify a diffusion model's output | Three 12×12 generators, architecture, pixel meaning, noise schedule, training objective, and rounded weights; no execution. | All three shape categories were correct. All descriptions called the shapes hollow or outlines, although the training targets were filled. Category accuracy and output fidelity differ. [E2] [E2Scores] [E2Key] |
| Identify language and attention-head roles | Exact float32 middle-block weights, forward code, training procedure, and three candidate corpus descriptions; Python weight analysis, but no layer or network forward computation. | Three independently trained checkpoints, two conversations each. All six language answers were correct. Every training-order case was unshuffled. Head-role accuracy depends on the induction threshold. [E3] [E3Prompt] [E3Scores] |
| Identify sycophancy and nominate patches | Three matched fine-tune pairs and a description of their training differences. Weights-only rounds vary tokenizer/embedding access; another arm exposes attention from requested texts. | Weights-only rounds 1–3 returned P(A)=0.5 with forced A labels. Round 4 was incomplete. Attention access classified 3/3 pairs, but inspected intervention records did not show successful behavioral repair. [E7] [E7Prompt] [E7Attention] [Patch1] [Patch2] [Patch3] |

The writing and repair tasks use different baselines and feedback budgets:

| Task | What the editor could do | Observation and comparison |
| --- | --- | --- |
| Repair the Kinetix car | Read the scene, rules, actor formula, rounded weights, failed rollout, and base evaluation; make one reply without tools or feedback. | One proposal changed 6,073 of 171,664 actor values through 51 edit statements, relative to the shown rounded base. Its policy succeeded in 1,024/1,024 episodes; the base succeeded in 0/256. These episodes evaluate one edit, not independent editor runs. [E4] [E4Result] |
| Synthesize MinAtar policies | Write a zero-initialized 600→32→32→6 tanh network from exact game source, rules, observation/action meanings, and scoring; one reply per game without execution or feedback. | Mean reward over 1,000 episodes, written policy versus best 500M-step PPO seed: Breakout **517.411 vs 25.344**; Seaquest **53.110 vs 88.884**; Asterix **53.178 vs 55.204**. The strong Breakout result does not extend to all three games. [E5] [E5Prompt] [E5Eval] |
| Synthesize a Go policy | Write a zero-initialized policy with six hidden ReLU layers and 1,480,274 values, using game rules and Python construction; budgets of 0, 10, 25, or 50 target passes. | One editor run per budget. At zero feedback, win rates were **54.1/54.9%** against two small PPO opponents and **42.6/26.3/36.3%** against three tuned same-size PPO seeds. At 50 passes, the latter rates were **74.2/70.0/78.5%**. Each opponent comparison used 2,000 games, balanced by color. Opponent size and tuning change the claim. [E6] [Go0] [GoSameSize] |
| Repair Qwen math behavior | Inspect weights, tokenizer, labeled practice problems and solutions; use Python, generation, teacher-forced feedback, activations, and explicit diagnosis guidance. | At 50 charged passes, MATH-500 accuracy rose **7.9→48.1%**, GSM8K **13.8→67.1%**, and IFEval **21.0→62.3%**. The final patch changed 130,188 stored bf16 values. This was an adaptive repair with informed access. [E8Prompt] [QwenBase] [Qwen50] [QwenManifest] |
| Improve Gemma math, follow-up | Similar adaptive access under a different model, decoding/grading protocol, and explicit DEV/HOLDOUT instructions; budgets of 5, 10, 25, or 50 passes. | Format-agnostic MATH-500 accuracy was **6.8%** before editing and **7.05/7.6/7.7/7.0%** afterward. The released follow-up did not establish improvement. [Gemma] [GemmaBase] [Gemma5] [Gemma10] [Gemma25] [Gemma50] |

Qwen's reported accuracy averages samples within benchmark problems: MATH-500 uses 500 problems with four samples each; GSM8K uses 1,319 with two; IFEval uses 541 with two. IFEval is strict prompt-level accuracy. Gemma uses the same 500 MATH problems with four samples each, but different sampling and answer extraction. These evaluation populations support comparisons between the retained endpoints under their own protocols. They do not measure variation across independent editor runs. [E8] [QwenBase] [Qwen50] [Gemma]

The tasks cannot be pooled into one success rate. Game checkpoints, generator categories, language checkpoints, fine-tune pairs, problems, and repeated rollouts are different units. More samples from one final edit can reduce measurement noise while leaving uncertainty across editors, lineages, and tasks unresolved.

## Critical findings

### Supplied semantics and hints are part of the task

Weights in the small-model tasks are contextualized. The editor knows game channels and actions, diffusion training semantics, or candidate corpus descriptions. The MinAtar prompt supplies exact game source. These conditions can support structural reasoning; identification from anonymous arrays is a different claim. [E1Description] [E2Description] [E3Prompt] [E5Prompt]

Qwen's first prompt suggests repetition as a repair target, describes localization and strength sweeps, names MATH-500/GSM8K, and discloses grading and decoding. It supplies 7,469 GSM8K, 7,275 MATH, and 714 AIME practice problems, with reference solutions where available. Hidden test items are a meaningful restriction, but they do not make the evaluator specification or labeled practice data hidden. [E8Prompt]

A useful next comparison would hold the task fixed and vary semantic metadata, weights, labels, activations, and feedback. That is a proposed access ablation, not a result of this review. Shuffled and unshuffled language cases would also be needed to test training-order discrimination; the inspected cases cover only the latter. [E3]

### Identification, mechanistic diagnosis, and causal repair are separate claims

The diffusion task shows why scoring must match the claim. Its answer key records shape categories without a filled/hollow field. A correct category can therefore coexist with the wrong visual description. The README acknowledges that error. [E2] [E2Scores] [E2Key]

Sycophancy makes the distinction causal. All nine inspected weights-only `gap_closed_h2s` values were zero, although continuous margin-closure values were nonzero. The attention arm's inspected pair-1 intervention also closed zero of the discrete behavioral gap. Its margin closure was **0.04**, compared with **0.08** for the largest-change control and **0.19** for the random-control mean. The checked interventions therefore do not support successful repair, even though pair classification succeeded. This is a finding about inspected patches; it is not a claim that all possible patches fail. [Patch1] [Patch2] [Patch3] [AttentionPatch]

The Qwen authors interpret the successful edit as suppressing a loop direction. The final 50-pass construction subtracts that direction from selected MLP output columns in layers 15 and 19. The score change establishes behavior of that retained candidate under the stated evaluation. A more specific claim about the direction's causal role would require matched interventions that distinguish it from other changes of similar size. [E8] [QwenManifest]

The Gemma follow-up did not establish improvement. The 50-pass final answer reports a HOLDOUT gain of **0.421 percentage points**, with problem-cluster SE **0.365 points** and approximate 95% interval **[−0.300, +1.143]**. It uses 128 problems and 16,384 answers per model. The interval crosses zero, and cutoff frequency worsens from **7.31% to 10.50%**. This internal HOLDOUT result is distinct from the MATH-500 score above. Its prompt requires a best edit even if insignificant and prohibits restoration of the original. Submission alone is therefore not evidence of benefit. [GemmaAnswer] [GemmaPrompt]

### Metric definitions and incomplete attempts belong in the record

The language README discloses an induction-head threshold mismatch: the post uses score >0.2, while `true_copy_heads` in `scores.json` uses >0.3. Python scores 0.245 and 0.269 and the romaji score 0.220 change inclusion under those cutoffs. The Python score 0.546 exceeds both thresholds and remains included. A review needs the raw values, threshold, and scorer identity together. [E3] [E3Heads] [E3Scores]

Sycophancy rounds 1–3 force A at P(A)=0.5, producing nominal one-third accuracy from a tie-break. Round 4 stopped at 29–30 runs per pair; only pair 2 completed. These records must retain both uncertainty and incomplete status. They cannot be treated as three completed, confident predictions. [E7]

Gemma retains both format-agnostic and strict boxed-answer scores; Qwen uses its own extraction and sampling rules. The Qwen release also corrects the figure's interim base GPQA score of 24.5% to the final merged score of 22.7%. Final, interim, strict, and lenient results need separate identities. [Gemma] [E8] [QwenBase]

### “No training,” “tool free,” and “few passes” need operational definitions

Go forbids gradients and fitting weights to examples, including regression and least squares. It permits formula-based construction and feedback analysis. A least-squares proposer is therefore a different experimental arm from the original Go task. Likewise, a prohibition on gradient training does not make Qwen's adaptive editing a no-feedback experiment. [GoPrompt] [E8Prompt]

The root audit's “tool-free” description concerns built-in Codex tools. Python reply blocks and external target tools still execute in experiments 3, 6, 7, and 8. The inspected sandbox provides process, network, and resource isolation, then checks tensor shape, float32 conversion, and finiteness. Those checks do not prove that arbitrary Python obeys every no-forward or no-fitting rule. A task should record declared rules, enforced capabilities, and compliance checks separately. [Root] [Sandbox] [E3] [E6] [E7] [E8]

A pass is an accounting unit, not a common amount of information. Go permits up to 128 games or 256 positions per pass; a position request can expose hidden activations. Qwen permits 512 generated answers, or a teacher-forced read of 1,024 texts totaling up to 524,288 tokens. Gemma permits 2,048 generated answers. Qwen's notice explicitly leaves duration and batch size uncharged. Equal pass budgets do not imply equal compute or feedback. [GoPrompt] [E8Prompt] [GemmaPrompt] [PassNotice]

Qwen p50c records 80 Python runs, 69 requests, 50 charged passes, and three handoffs. Go retains disk-quota failures, and the attention experiment describes infrastructure retries hidden from the editor. Attempts, completions, retries, and editor-visible observations each matter. The inspected Go sandbox can return 20,000 characters, while its log retains only the first 6,000; the log is not always the complete observation. [QwenState] [E6] [E7] [Sandbox]

### Exact candidate realization matters

The Kinetix change count is measured from the rounded network shown to the editor. It must not be silently substituted for a comparison against the full-precision checkpoint. MinAtar's Seaquest and Asterix replies were reparsed after a whitespace fix, without new editor replies. The proposal, parser, program, parent, and realized candidate are distinct objects. [E4] [E5]

For Qwen p50c, two 64×1,024 column blocks touch **131,072** entries, but only **130,188** stored bf16 entries differ. Reconstruction checks the base tensor's dtype, shape, and hash, then verifies each reconstructed edited tensor. The builder casts submitted values to the original dtype. For p10c, low-rank reconstruction needs rounding corrections; the manifest records 30 changed tensors and **72,986,982** effective changed entries. A compact formula can produce a large stored edit. [QwenManifest] [ApplyEdit] [BuildCheckpoint] [NativeConversion] [Qwen10Manifest]

These records motivate explicit rounding and endpoint checks in this framework. They do not show that sparse edits preserve unrelated behavior. Nor does a native-byte reconstruction establish equivalence across execution backends or a real-valued behavioral bound.

### Baseline selection and omitted history limit generality

The policy-identification models are selected training seeds. MinAtar compares against the best PPO seed. Go distinguishes smaller opponents from tuned same-size opponents and retains only one editor run at each reported budget. Another prompt version and a one-pass run are omitted. These demonstrations do not estimate editor variance or isolate a causal law relating pass budget to performance. [E1] [E5] [E6]

Qwen and Gemma also disclose runs outside the reported subset. Qwen's notices alter tool availability and clarify batching that cannot be reproduced exactly. A later notice corrects failure counts that previously labeled wrong stated answers as missing answers. Protocol changes, rejected attempts, selection criteria, and experimenter interventions are part of the method, even when the final candidate is reproducible. [E8] [Gemma] [ToolNotice] [FailureNotice]

## Reproducibility is a set of distinct obligations

| Reproduction question | What the inspected release supports | Work still required |
| --- | --- | --- |
| Can a reader inspect the same records? | Pinned files, prompts, notices, selected code, and score records. | Establish completeness and independently verify events claimed in the logs. |
| Can a reader reconstruct the candidate? | Final small-model assets are inventoried; Qwen/Gemma provide diffs against public bases with tensor-hash checks. | Download the required bodies and execute reconstruction. This review did neither. |
| Can the final evaluation be repeated? | Several evaluators and protocol details are supplied. | Resolve packages, datasets, hardware, omitted graders, and stochastic execution in a separate run. |
| Can the editor session be replayed? | Many prompts, replies, and summaries are retained. | Recover missing observations, intermediate states, harness behavior, and provider dependencies. |
| Does an independent repetition reach the same scientific result? | The release provides cases from which to design a new protocol. | Run independent editors and untouched lineages/tasks under a frozen comparison. The report's credited third-party Qwen rebuild was not independently inspected here. |

The missing inputs have specific consequences. The language release contains only middle blocks; Python-corpus reconstruction recovers 81.5% of characters exactly and leaves 18.5% unidentified. MinAtar PPO checkpoint bodies are absent. Go omits intermediate full networks and feedback boards/valid masks. Sycophancy checkpoint bodies and most attention arrays are absent, and its rebuild description warns that GPU bf16 training is not bit-reproducible. Qwen omits work directories, its agent harness, some graders, and other prompt-version runs. A reconstruction recipe does not resolve those session-replay gaps. [E3] [PythonCorpus] [E5] [E6] [E7] [E8]

## Actual limitations and next steps in the original report

In **What Comes Next?**, the authors identify context length and harness inefficiency as limits. They propose auditing without evaluation, cheaper optimization, zero-data steering and unlearning, recursive self-improvement, and circuit transfer across architectures. They also describe a benchmark and harness under development. These are attributed research directions, not demonstrated outcomes. The retained repairs do not establish backdoor detection, successful unlearning, absence of catastrophic forgetting, recursive self-improvement, or cross-architecture transfer. [Report]

The Gemma result narrows the interpretation of the positive Qwen case: repair of a model with a prominent looping failure does not establish reliable improvement on a model with less looping. It also does not prove that weight editing can never improve an initially absent ability. That universal conclusion exceeds the observed models, tasks, and budgets. [Gemma] [GemmaAnswer]

## Prior documented boundaries in this repository

This review builds on an existing framework for named tensors, atomic edits bound to an exact parent, structural admission, and caller-supplied execution. The sources below describe its earlier boundaries. They should not be read as claims that the v2 work has already resolved each gap.

| Existing source | Boundary relevant to this study |
| --- | --- |
| [README: Understand the guarantees](../README.md#understand-the-guarantees) | Admission, real-valued certificates, formal theorems, and observations establish different claims. The first proof family is two-layer ReLU. |
| [Design: Readout compilation and its limits](../design.md) | Soft least squares differs from exact anchor constraints. Supplied anchors need not span future feature rows, and the floating solver has no compiler-correctness theorem. |
| [Design: Caller-supplied PyTorch modules](../design.md) | A tensor snapshot does not capture arbitrary forward code, hidden state, RNG, hooks, or external resources. Aliases and native dtypes affect realization. |
| [Design: Proof boundary and evaluation boundary; Use and scale](../design.md) | Tensor schemas do not provide architecture semantics. Dense JSON, copying, scans, rational arithmetic, alignment, and fitting impose scale costs. |
| [Research: Follow-up hypotheses and remaining gaps](../research.md) | Transfer, cumulative edits, and fresh utility require new protocols and whole-lineage/task splits. |
| [Results: Reproduce or extend](../results.md#reproduce-or-extend) | Initial and most final RTS checkpoint bodies are missing. Published seeds are reproduction inputs, not untouched confirmation data. |
| [Project memory: Follow-up research and existing applications](../../PROJECT_MEMORY.md#follow-up-research-and-existing-applications) | Existing app policies have different feature and decision meanings. No broad operator-superiority or Astra proposal-quality result is established. |
| [Formal README: Semantic scope and trust boundary](../../../formal/weight-space/README.md#semantic-scope-and-trust-boundary) | Theorems concern explicit finite real-valued networks. They do not establish deployed arithmetic, simulator reachability, reward, or another architecture's equivalence. |

The [evidence companion][Evidence] retains source hashes and section locators. The bounded scan found no standalone `LIMITATIONS.md`, `NEXT_STEPS.md`, or subsystem roadmap. The original frozen study, its failures, and its missing checkpoint bodies remain historical evidence. In particular, its RTS selection chose the no-op and produced zero held-out effect; a broader framework does not change that result.

## Requirements derived from the review

The framework's proposed workflow is **TaskSpec → ProgrammerSession → Candidate → EvaluationReceipt**. It connects a requested claim to the information available, the object submitted, and the evaluation performed. The proposer can be an LLM, an analytic construction, or a fitting algorithm; its mechanism must remain explicit.

| Source problem | Required behavior | Evidence a reviewer should be able to inspect |
| --- | --- | --- |
| Identification, diagnosis, synthesis, and repair have different outputs | Declare the task and claim before evaluation. | A classification answer can be correct while its proposed intervention fails. |
| Metadata, labels, hints, activations, and feedback vary | Record supplied information and distinguish rules from enforcement. | The actual observations and the access boundary for that session. |
| Architectures have different execution semantics | Separate stored tensors, executable meaning, and certificate support. | An unsupported backend remains explicitly unchecked. |
| Programs and native endpoints can differ | Bind proposal, parser/compiler, parent, operations, conversion, and final tensors. | Requested changes, rounding loss, and effective changes. |
| Structural validity does not establish benefit | Keep structural, mathematical, numerical, behavioral, and generalization evidence separate. | Failed or unchecked claims remain visible after another check passes. |
| Pass counts hide work | Record calls, batch sizes, tokens, examples, timing, retries, and selection. | Budget comparisons with common denominators and retained failed work. |
| Selection and incomplete attempts affect interpretation | Preserve candidate lineage, failures, notices, completion, scorer version, and selection rule. | The selected endpoint resolves to its place in the full attempted population. |
| Repeated observations share editors, models, or problems | Identify independent units before choosing splits and uncertainty estimates. | Variation across the units named in the claim. |
| Sparse changes can still cause broad regression | Measure target behavior and retention; keep the no-op eligible. | A valid patch can be rejected or yield no established gain. |
| Candidate reconstruction differs from full replay | State the reproduction level and missing inputs. | A diff can be hash-verifiable while session replay remains incomplete. |

These are design and acceptance requirements. A workflow data model alone supplies neither a security sandbox nor proof that a declared rule was obeyed. The [methods](methods.md) and [results](results.md) identify which parts the local deterministic study exercises.

## Testable follow-up work

A stronger follow-up would state in advance which observation would weaken its claim:

1. **Weight information.** Compare semantic metadata alone with weights plus the same metadata on untouched lineages, then add labels, activations, and feedback in separate arms. Little or no advantage from weights would weaken a claim that weight information drives identification.
2. **Causal repair.** Freeze a mechanistic prediction, then compare its edit with matched random, magnitude, opposite-direction, and restoration controls. For loop repair, separate finishing from correctness. Similar improvement from matched controls would weaken the claimed direction-specific mechanism.
3. **Candidate realization.** Exercise parser revisions, aliases, parent mismatches, overflow, underflow, and rounding on small fixtures. A proposal that reconstructs a different endpoint should fail realization before receiving a behavioral claim.
4. **Cumulative usefulness.** Freeze per-step selection and recheck all earlier target and retention conditions after each edit. Loss of an earlier condition would refute a claim that this procedure preserves the cumulative contract, even if composition remains exact.
5. **Architecture and scale.** Add explicit execution semantics and measure storage, copying, fitting, and certificate cost on the intended model sizes. Unsupported semantics or impractical resource use would limit applicability despite successful small fixtures.
6. **Generalization.** Repeat editors and checkpoint lineages under a new frozen protocol, retain failed attempts, and report useful-and-admitted outcomes over the whole attempted population. Failure on untouched units would weaken generality; a submitted patch or a successful reconstruction alone would not rescue it.

No item in this agenda was run for the source review. It separates the next test from the result already observed.

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
