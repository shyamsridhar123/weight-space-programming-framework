# Weight-space programming: completion requirements

> **Original delivery requirements.** This page records the obligations and audit
> for the first implementation. Use the [current requirements](v2/requirements.md)
> to assess the extended framework and the [research roadmap](v2/limitations-and-next-steps.md)
> to assess proposed work. The [original document](../../provenance/original-documents/docs/weight-space/requirements.md.txt)
> retains the source record unchanged.

This work follows the user's request to read project memory, research weight-space
learning deeply, formulate a hypothesis using the scientific method, develop Lean
proofs, and build a comprehensive framework. “Wright space” is interpreted as
**weight space**, consistent with the supplied Awesome-Weight-Space-Learning index.

The framework is an executable research foundation for programs that produce or
modify neural parameters under explicit behavioral specifications. The concrete
verified model family is a two-layer ReLU network. Supporting an architecture in a
proposal interface does not make its semantics or implementation formally verified.

| Requirement | Evidence required for completion |
| --- | --- |
| Project memory | Current code audit, durable project decisions, explicit separation of the RTS scorer and stochastic living-world policy |
| Deep research | Primary-source evidence matrix, pinned supplied index, substantial coverage of programming, equivariance, generation, merging/editing, verification and negative findings |
| Hypotheses | Observable endpoints, nulls, controls, fresh splits, fixed selection rules, uncertainty estimates and pre-result protocol hash |
| Mathematical framework | Parameter spaces, semantics, symmetry actions, edit algebra, contracts, composition, compiler and proof obligations |
| Executable framework | Validated immutable artifacts, typed atomic edits, provenance, synthesis, symmetry transport/alignment, exact rational certificates and a documented CLI |
| Formal proof | Pinned Lean build with no proof holes/custom axioms; substantive network theorems; at least one actual-policy certificate instantiated and checked |
| Scientific experiments | All arms and failures retained; positive and negative controls; independently trained networks and actual RTS simulation; held-out evaluation distinct from synthesis/selection |
| Integration | Import/export and numerical agreement with the actual 32→38→1 RTS policy; evaluated proposals and lineage without changing the live game |
| Reproducibility | Commands, versions, source/protocol/artifact hashes, raw trial records, report generated from those records, meaningful tests |
| Honest assessment | Requirement-level completion audit; real arithmetic/float and finite-horizon/generalization boundaries; unsupported hypotheses remain unsupported |

The existing Astra provider is a source of candidate edits. Access to that provider
is not a prerequisite for testing the compiler, semantics or admission mechanism.
No claim about Astra's proposal quality will be made without new measured calls.
The framework does not infer overall task success from a bounded score change.

## Current-state audit

The repository README, `docs/EMPIRES.md`, the existing whiteboard notes and
`presentations/living-world/sources/READING.md` preserve relevant project context.
The contributor memory at `~/.codex/memories/MEMORY.md` currently contains ECorp
records and no RLCD task group. Current executable sources take precedence over
older teaching notes that predate the live Astra programmer.

The RTS policy has 1,293 authored parameters. Its output is a **score for one
candidate**, not a probability of success. First Conquest selects the highest
score among legal candidates, breaking ties by candidate ID. Shared output bias
and temperature are therefore decision-invariant in real arithmetic. The living
world uses a different input schema and samples a softmax; its temperature has a
different meaning. Existing paired simulations show reproducible interventions,
but their previously used seeds are not untouched research holdouts.

## Scope and review triggers

Changing feature order, activation, action selection, numeric representation or
legal-action filtering requires a new semantic adapter and rechecking assumptions.
A same-function symmetry is not an arbitrary correspondence between independently
trained neurons. A real-valued theorem does not certify IEEE-754 execution or
prove task reward, robustness outside its domain, or universal LLM weight synthesis.

## Completion audit

The requested research foundation is delivered within the generic scope below.
Completion means the implementation, checks, experiments, and evidence are
available; it does not mean every empirical hypothesis succeeded. The
[results guide](results.md) and [generated report](../../artifacts/weight-space/report-v2/report.md)
provide the measured conclusions and their limits.

| Requirement | Delivered evidence | Boundary |
| --- | --- | --- |
| Project memory | [Durable memory](../PROJECT_MEMORY.md), current-source audit above, and framework guide | Existing Laya/Astra and living-world results retain their separate meaning |
| Deep research | [Research review](research.md), [28 source records](references.json), pinned supplied index, 13 inspected released artifacts, and 41 checked citation mappings | Author-reported results are identified; no first-of-kind or universal-superiority claim |
| Hypotheses | [Frozen protocol](../../artifacts/weight-space/study-v1/freeze/protocol.json), [pre-result source receipt](../../artifacts/weight-space/study-v1/freeze/receipt.json), fixed units/controls/selection/bootstrap rules | Local pre-result freeze, not externally timestamped preregistration; exploratory unadjusted intervals |
| Mathematical framework | [Design](design.md) defines parameter artifacts, semantics, symmetries, programs, contracts, composition, and proof obligations | Proved behavioral semantics are for finite two-layer ReLU networks |
| Executable framework | [Generic guide](README.md), [source](../../rlcd/weight_space/), [standalone evidence](../../artifacts/weight-space/demo-v1/), tensor/Tanh and PyTorch examples | Generic structure and execution adapters do not provide proofs for arbitrary forward code; reference-scale storage and algorithms |
| Formal proof | [Lean project](../../formal/weight-space/README.md), [actual verification receipt](../../formal/weight-space/verification/actual-verification.json), successful build and transitive axiom audit | 59 explicit theorem statements; all 1,293 actual-policy coefficients and four candidate contracts checked over real arithmetic; external artifact mapping remains outside Lean |
| Scientific experiments | [Synthetic evidence](../../artifacts/weight-space/study-v1/synthetic/results.json), [RTS evidence](../../artifacts/weight-space/study-v1/rts-test/results.json), [independent audit](../../artifacts/weight-space/study-v1-audit.json), and [report](../../artifacts/weight-space/report-v2/report.md) | All 40 networks/20 pairs completed; H1/H2 supported locally; H3 unsupported with no nonzero held-out RTS edit |
| Integration | [RTS export](../../artifacts/weight-space/study-v1/rts-export/), [candidate/admission artifacts](../../artifacts/weight-space/study-v1/rts-candidates/), and recorded 256-fixture/1,200-score agreement | Specific adapter checks; no changes deployed into a running game and no new provider-quality claim |
| Reproducibility | [Study wrapper](../../scripts/run_weight_space_study.ps1), [audit script](../../scripts/audit_weight_space_study.py), [report script](../../scripts/report_weight_space.py), manifests, raw records, [test/build evidence](../../artifacts/weight-space/verification-v1/summary.json), and [repair evidence](../../artifacts/weight-space/post-study-repair-v1/receipt.json) | Omitted RTS checkpoint bodies prevent independent reconstruction of those recorded checkpoint hashes; used seeds are no longer fresh holdouts |
| Honest assessment | [Results and trust boundaries](results.md), all admission denominators, negative RTS finding, and preserved before/after repair provenance | No inference from bounded score drift to reward, task success, floating execution, or other architectures |

The independent audit and report use retained data; they do not train models,
select a replacement candidate, or rerun reserved worlds. The broad pre-repair
checks passed 239 Python tests, 386 TypeScript tests, type checking, and the
production build. The current receipt repair passed all 76 focused weight-space
tests. Original study sources and the repaired workspace are deliberately distinct.

Future extensions include additional semantic/proof backends, scalable tensor
storage and alignment, independent datasets and architecture families, and an RTS
protocol that reaches more difficult opponent phases. These are new research
tasks; the present results do not depend on assuming that they will succeed.
