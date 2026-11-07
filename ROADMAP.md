# Research and engineering roadmap

The next work must connect a declared task to an exact candidate, a justified
guarantee, and a useful measured outcome. The framework already has generic
artifact and workflow interfaces, real-valued proofs for explicit semantic
families, and deterministic study evidence. The open questions concern the
connections between them and their behavior on independent tasks.

A small example shows the issue. For a linear head, an edit can raise the
target score while also moving a protected score. A different edit can
preserve that score exactly. If target and protected features coincide, the
requested nonzero change is impossible under exact protection. A useful
framework must report all three outcomes clearly. The
[worked explanation](docs/weight-space/v2/formal.md#a-small-edit-that-exposes-the-problem)
connects this example to the existing theorems.

The detailed [limitations and next-steps plan](docs/weight-space/v2/limitations-and-next-steps.md)
is the authoritative research program. It builds on the original report's
future directions, released access protocols and omitted artifacts, earlier
framework limitations, and the current study. Each package states the failure
it addresses, the research needed, and a completion or falsification condition.

## Work packages

| Priority | Package | Why it matters | What completion would establish |
| --- | --- | --- | --- |
| P0 | [Bind admission to one exact candidate and contract](docs/weight-space/v2/limitations-and-next-steps.md#1-bind-admission-to-one-exact-candidate-and-contract) | A sound certificate can be attached to the wrong endpoint or domain | Structural, semantic, and behavioral decisions identify the same endpoints, adapter, domain, and output meanings; substitutions are rejected |
| P0 | [Generate and check new formal instances](docs/weight-space/v2/limitations-and-next-steps.md#2-generate-and-check-new-formal-instances) | Hand-written Lean witnesses do not automatically certify runtime artifacts | A supported new candidate has a checked formal instance, exact extraction records, and explicit correspondence obligations |
| P0 | [Bound or measure numerical execution](docs/weight-space/v2/limitations-and-next-steps.md#3-bound-or-explicitly-measure-numerical-execution) | Stored parameters and real semantics do not account for intermediate native arithmetic | The claim identifies its arithmetic/backend model and either bounds execution error or reports the limits of empirical measurements |
| P0 | [Build an auditable restricted and live-provider harness](docs/weight-space/v2/limitations-and-next-steps.md#4-build-an-auditable-restricted-and-live-provider-harness) | Work outside cooperative callbacks can defeat a claimed access restriction | Information, computation, attempts, failures, and selection are auditable under specifically supported restrictions |
| P1 | [Run independent task and access experiments](docs/weight-space/v2/limitations-and-next-steps.md#5-run-independent-task-and-access-experiments) | Rich context can explain success that is attributed to weights | Weight access is compared with identical context under untouched tasks, lineages, and declared independent units |
| P1 | [Establish feature coverage and feasibility limits](docs/weight-space/v2/limitations-and-next-steps.md#6-establish-feature-coverage-and-feasibility-limits) | Small anchor residuals can hide large domain drift or a conflicting target request | Span coverage, conditioning, representation error, realized target gain, and infeasibility are distinguished |
| P1 | [Test causal diagnosis and repair](docs/weight-space/v2/limitations-and-next-steps.md#7-test-causal-diagnosis-and-repair) | Correct identification or a score shift can coexist with failed repair | Frozen intervention predictions are tested against suitable controls, with separate target-utility and retention criteria |
| P1 | [Admit and evaluate cumulative edit sequences](docs/weight-space/v2/limitations-and-next-steps.md#8-admit-and-evaluate-cumulative-edit-sequences) | Individually accepted steps can exceed a global budget or erase earlier gains | Each endpoint meets its declared cumulative contract, while useful progress is evaluated across independent lineages and task orders |
| P2 | [Extend operators and tighten bounds](docs/weight-space/v2/limitations-and-next-steps.md#9-extend-operators-and-tighten-bounds) | Unsupported operators and conservative bounds limit useful certification | Each extension has explicit semantics and soundness obligations, plus measured coverage, tightness, and cost |
| P2 | [Measure scale and total cost](docs/weight-space/v2/limitations-and-next-steps.md#10-measure-and-improve-scale-and-total-cost) | Compact edits can still require expensive dense preparation or evaluation | Complete representation, proposal, verification, execution, and evaluation costs support a bounded efficiency claim at matched quality |
| P2 | [Reconstruct and repeat selected source tasks](docs/weight-space/v2/limitations-and-next-steps.md#11-reconstruct-and-independently-repeat-selected-source-tasks) | A reconstructed patch is easily mistaken for a reproduced editing process | Inspection, reconstruction, reevaluation, replay, and independent replication have separately justified scope |

## Dependencies and experiment rules

The first four packages can proceed in parallel where their artifacts and
interfaces are independent. Candidate identity is needed for an integrated
admission claim. A formal-instance bridge is needed for a new candidate-specific
machine-checked guarantee. A numerical model is needed to extend a real bound
to native execution. An audited harness is needed for enforced-access and
live-editor claims.

These are dependencies of claims, not a requirement to finish all proofs
before measuring behavior. An empirical study may proceed with a clearly
stated implementation or proof gap. It must not report that gap as closed.

The next task experiments need frozen rules and untouched units. All published
seeds, including worlds `7300–7311`, are already observed. Their reruns provide
reproduction evidence. New confirmation needs new units selected before their
outcomes are known. Additional rows, episodes, or sampled answers from one
final edit do not supply independent editor trials or task families.

There is no live language-model/provider study in the current evidence. The
fitting-permitted compiler and analytic controls answer narrower questions.
The source Go protocol explicitly forbids regression and least squares; a
future protocol-replication arm must preserve that restriction.

## What a completed package should leave behind

Engineering work should provide versioned implementation or checked proof
obligations, exact artifact identities, focused validation, and an explicit
unsupported boundary. Experimental work should provide a frozen protocol, raw
per-unit records, a complete attempt and selection ledger, reproducible
analysis, and conclusions matched to the independent units. Both should retain
negative results and make the stated falsification checks reproducible.

The report's proposals for auditing, zero-data steering, unlearning,
optimization savings, recursive improvement, and cross-architecture circuit
grafting remain separate research hypotheses. Their
[minimum credible tests](docs/weight-space/v2/limitations-and-next-steps.md#source-future-directions-and-their-unresolved-tests)
are specified in the detailed plan. The
[requirements ledger](docs/weight-space/v2/requirements.md) records current
status; roadmap entries are not implementation claims.
