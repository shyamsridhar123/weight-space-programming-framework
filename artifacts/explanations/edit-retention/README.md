# Change one output. Keep another.

Open `index.html` in a browser. It is a standalone explanation of the exact,
bias-free linear-head example in the [worked guide](../../../docs/weight-space/v2/worked-example.md).
The HTML contains its styles, original SVG figure, and script. It needs no build,
downloaded dependency, server, or external network request.

Start with the exact target-only edit. Compare the edit that preserves `(1, 0)`
and the no-op. Move the slider to inspect other exact-target edits, then protect
the target or both coordinate directions to expose infeasibility. The output
table keeps target success separate from retention. Reset restores the initial
edit, protected set, and system appearance.

The controls use native buttons, range input, and select elements. Keyboard
users can tab between them, use arrow keys on the slider/select, and activate
buttons with Enter or Space. A static initial figure, comparison table, and prose
remain readable with JavaScript disabled and in print.

The [source record](source.json) defines the supplied model, arithmetic,
defaults, interactions, and limits. Equality decisions use integer numerators
over a fixed denominator of 20; displayed decimals do not decide success.
This example is illustrative. It creates no new experimental evidence.

The [documentation verification receipt](../../../provenance/verification/runs/documentation-draft3.json)
records checks actually performed. These checks do not claim accessibility
certification, learning efficacy, or STE engine validation.
