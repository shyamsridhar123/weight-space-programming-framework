# From weight-space programming to a visible learning experiment

The conceptual starting point is **“Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights”**, by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, published November 2, 2026:

https://rikiyat.github.io/blog/weight-space-programming/

The article studies general-purpose language models reading and writing the numerical parameters of other neural networks. A natural-language specification describes the desired behavior; the produced weights implement a small neural program without gradient training during synthesis.

Crown & Charter makes the relationship between weights and behavior tangible. Its civilizations use small, authored neural policies. A learner can edit real output weights, hold observations fixed to understand an immediate probability change, and compare two simulated futures from a common checkpoint. Optional native Laya selects among a bounded set of authored weight patches and the option to keep the current policy. General natural-language specification-to-weight synthesis is outside this implementation.

## Read the implementation

- `policy.ts`: the actual network, observations, scores, temperature, and sampling.
- `learning-experiment.ts`: checkpoint capture and paired simulation.
- `coach.ts`: the optional native Laya selector and bounded patch validation.
- `EMPIRES.md`: application guide, controls, system limits, and learning workflow.
- `manifest.json`: provenance and SHA-256 hashes of the copied workspace source files.

The deck's embedded laboratories import the same policy and simulation code used by the game. They run locally against their own deterministic fixture and do not call Laya or read or change the game's saved world.

## Read an experiment carefully

At a fixed observation, changing an output weight changes a candidate's score by the hidden activation multiplied by the weight change. Softmax then converts all eight scores into probabilities. A probability change is an immediate model effect, not a guarantee of the next action.

Paired simulations start from identical world and random state. Different decisions can change later random draws, so a result describes one reproducible scenario. Scheduled decision counts do not guarantee successful execution, and ecological comparisons follow regions occupied at the starting checkpoint. The deck lets you compare several horizons in one fixed scenario. Use the full application to try other societies and seeds before making broad claims.

The prerecorded film is observational footage. It contains no controlled weight intervention; use the live labs to test one.
