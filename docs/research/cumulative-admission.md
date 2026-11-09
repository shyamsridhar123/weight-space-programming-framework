# Cumulative admission over frozen edit lineages

Research review: November 8, 2026. Roadmap issue: #10.

The controller binds each proposed head edit to the original model and every earlier accepted target insertion. It records numerical score residuals, native queue behavior, and evaluation utility as separate claims. A candidate that passes against its current parent can still fail its original-baseline or prior-target obligation.

The candidates and calibration admissions were sealed before the coordinated research release. Evaluation and separate-process outcome replay have now passed with the pinned identities unchanged. [The released result report](cumulative-and-bounds.md) accounts for every assigned sequence: no trained proposal was admitted, so there is no trained usefulness or successful prior-gain retention result. The design below records the frozen rules; it claims no continual learning, enterprise deployment impact, or recursive improvement.

## What the contract remembers

A sequence contract fixes the original artifact, readout head, feature identity, queue map, protected calibration features and labels, ordered target requests, controller, and policy. Each step binds its current parent, actual rounded candidate, producing program, original ancestor, sequence contract hash, previous history hash, execution environment, and insertion number.

Every accepted request remains a separate memory. A repeated subject does not overwrite an earlier insertion. Its retained record contains its request hash, insertion checkpoint, insertion step, calibration features, desired code and queue. Replay reconstructs the complete memory list from the earlier accepted records. Removing an obligation and recomputing the outer hashes fails this reconstruction.

The controller checks that the program produces its captured candidate, endpoint identities match their artifacts, feature arrays match their recorded hashes, head and feature identities remain fixed, and no captured tensor outside the head changes. Requests also require valid target labels, code-to-queue correspondence, distinct accepted request identities, and strictly increasing insertion steps.

## Admission rules and arithmetic

Both controllers use the same score cap, derived once from the original model: half its median winning score margin on the frozen protected calibration sample. This is a model-score feasibility cap. It has no agreed enterprise SLA or cost interpretation, and it is never enlarged after a rejection.

For a frozen feature row `h`, each score change is computed as an exact rational quantity:

```text
change_i(h) = (candidate_bias_i - reference_bias_i)
            + sum_j((candidate_weight_ij - reference_weight_ij) * h_j)
```

Each finite stored endpoint and feature value is lifted to its exact binary value before subtraction and multiplication. The receipt reports the maximum absolute residual across the supplied rows and output coordinates. This proves a statement about those feature rows. It does not cover their full feature box, arbitrary future features, or the raw-input encoder.

Native scores are separately executed with `torch.nn.functional.linear` on CPU using float32, then the deterministic lowest-index argmax and frozen queue map. Tensor layout, feature and endpoint casts, actual operator, bias convention, Torch version, determinism flags, and numerical environment are captured. Fusion and reduction order remain controlled by the backend. These native observations are not a formal kernel correspondence theorem.

The local controller requires local protected score residuals within the fixed cap, protected queue flips and net harm at most 0.01, and new-target calibration gain at least 0.10. It records original and prior-insertion checks without making them required.

The cumulative controller additionally requires the original protected score residuals within that same cap, original protected queue flips and net harm at most 0.01, and every accepted insertion's score residual within the cap, queue flips at most 0.01, and accuracy loss from its insertion checkpoint at most 0.01. Each earlier insertion therefore gets its own comparison, age, and evidence receipt.

Score preservation, queue preservation, accuracy loss, and usefulness differ. Preserved argmax can tolerate changed losing scores; a small score bound can still be insufficient near a tie; a no-op can satisfy retention while producing no new gain. The ledger retains these distinctions.

## Conflicts, rejection, and rollback

An accepted subject's contrary desired code or queue is a conflict and is rejected. The policy does not silently revise the earlier obligation. Malformed proposals and solver failures become recorded rejected no-ops. A rejected attempt keeps the last accepted checkpoint, and all assigned attempts remain in the sequence.

The designed boundary sequences include a chain whose increments pass local score checks while the accumulated change breaches the original cap. Local admission accepts its relevant steps; cumulative admission rejects the violating endpoint. These constructed examples test the policy boundary and are not samples of enterprise traffic.

Replay verifies every producing program, required gate, insertion memory, deployed checkpoint, composition, and named rollback. The inverse composition restores only the captured readout artifact. It does not restore an encoder mutation, cache, RNG, deployment process, queue assignment, or an already executed business action. A forged assertion that external state was restored fails verification.

## Frozen trained design

The sequence study reuses two previously trained domains with six independent full training lineages per domain. The support task uses BANKING77; the nontext task uses optical digit recognition. Each sequence is evaluated in forward and reverse request order with local and cumulative controllers. Orders and controllers share a trained ancestor and are averaged within that training lineage rather than treated as independent samples.

Support requests contain two ownership changes, for `card_arrival` and `pending_cash_withdrawal`, routed to the reserved specialist code and its queue, plus two recognition requests, for `top_up_by_bank_transfer_charge` and `atm_support`, retaining their original code and queue. Digits contain an ownership change for digit 8 and recognition for digit 4. Ownership and recognition remain separate interpretations of the same generic head contract.

The proposer is the existing soft feature editor with ridge 0.01, retention weight 8, and the declared nullspace cutoff 1e-12. It uses the scheduled label's target-role rows and at most 128 deterministic ID-hash-ordered protected fitting rows. Original calibration protection excludes all scheduled labels and uses at most 128 rows under the same fixed ordering. Each target memory uses all its scheduled-label calibration rows. Fitting features and admission features remain role-separated.

All candidates, policies, orders, reference checkpoints, feature arrays, and admissions were frozen before this study opened its evaluation arrays. The source freeze includes the complete project import closure, package initializers, core admission code, broker and provider dependencies, and the training script. Verification rejects changes to any frozen executable source or input identity.

The heldout feature rows already exist from the earlier study. Reusing them for this adjacent sequence question is an explicitly secondary analysis. It is not fresh independent confirmation, even though this study preserves its own candidate-before-evaluation boundary. The primary evaluation population is the earlier study's novel official-test feature rows; complete official test is a secondary sensitivity endpoint.

## Evaluation and uncertainty

For every insertion, the utility matrix preserves its insertion accuracy, final accuracy, original-baseline gain, loss since insertion, age, and later endpoints. Rejected attempts use their retained deployed parent; missing outcomes remain assigned sequences.

Joint sequence usefulness requires all final target gains at least 0.10, original protected net harm throughout at most 0.01, and every earlier target's loss throughout at most 0.01. The protocol also records the requested coverage threshold of 0.75. A numeric retention receipt alone cannot satisfy this joint utility definition.

Descriptive paired Student-t intervals use six training-lineage means per domain. Forward/reverse orders and target steps are aggregated inside their lineage. Missing gains and harms are represented by zero-filled descriptive estimates with explicit sensitivity ranges of [-1,1]; missing joint utility receives a failure value with a [0,1] sensitivity range. This preserves the all-assigned-sequence denominator and exposes incomplete pairs. Intervals do not establish general population validity or an enterprise SLA.

## Evidence and commands

The candidate ledger is in `artifacts/research/cumulative-admission/2026-11-08`.

| Frozen object | SHA256 |
|---|---|
| Candidate manifest | `208802510de6675f1ae09efcac54932c4dae9651df5d9d08f395e1b364b1309d` |
| Protocol | `f2adf3ed83a402b696c2ff8f7085454c0670e3d6e005474a5637943e18752cc8` |
| Designed boundary records | `0875ae033c14fc7de644165a250acb7100557d788cc4884fba16c8b8ddd88aef` |

The manifest identifies the source snapshots, protocol, complete candidate pool, calibration arrays, sequences, and global candidate seal. Keep its identity outside the mutable evidence directory when authenticating a replay. Internal hashes provide correspondence and corruption detection; they are not an independent authorization for a substituted ledger.

```powershell
python -m rlcd.weight_space.cumulative_research verify-seal artifacts/research/cumulative-admission/2026-11-08
python -m pytest tests/test_research_cumulative_admission.py -q
```

`seal` creates a new evidence directory exclusively; `verify-seal` replays candidate admission and never loads evaluation features. After the coordinated release, `evaluate` verifies the immutable candidate seal and appends exclusive `results.json` and `evaluation-manifest.json` files. The original candidate manifest remains unchanged. `verify` then independently replays both the candidate and outcome evidence. `run` combines seal and evaluation for a new study directory and is not the pre-release command.

The engineering checks cover global drift rejection, retained insertion reconstruction, conflicts, rehashed omissions, invalid inputs, non-head mutations, no-op failures, matrix bookkeeping, source correspondence, and named rollback. Their passing status supports implementation behavior. Test counts are not evidence of task utility.

## Scope of the result

This implementation supports cumulative sampled admission over stated frozen head lineages, with original and earlier-target obligations that replay. The released trained evaluation found zero admitted steps and zero whole-sequence useful coverage; unchanged-head zero harm does not establish successful learned retention. Extending it to uncertain feature extraction, broad raw-input domains, changing deployments, or external actions requires separate contracts and evidence. The generic framework keeps those boundaries visible instead of treating a head checkpoint restore as a system-wide guarantee.
