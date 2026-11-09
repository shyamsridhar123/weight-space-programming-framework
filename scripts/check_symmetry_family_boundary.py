"""Standalone engineering fixture for #16; separate from the frozen pilot.

Run from the repository root: python -m scripts.check_symmetry_family_boundary
"""

from rlcd.weight_space.contracts import exact_score
from rlcd.weight_space.model import Network
from rlcd.weight_space.program import apply_program, difference
from rlcd.weight_space.semantics import as_relu_network, to_tensor_artifact
from rlcd.weight_space.sequential import sequential_artifact, sequential_forward
from rlcd.weight_space.symmetry import Gauge, transfer_edit, transform, transport


def main() -> None:
    # Dyadic parameters/scales keep these transformations exactly representable.
    base = Network([[1.], [-1.]], [.5, .25], [[1., 2.]], [0.])
    edited = base.replace(w2=[[2., 2.]])
    program = difference(base, edited, "Increase the positive-input readout")
    witness = [1.]
    edited_score = exact_score(edited, witness)
    assert program.edits and exact_score(base, witness) != edited_score
    assert as_relu_network(to_tensor_artifact(base)).sha256 == base.sha256

    gauges = (
        ("permutation", Gauge((1, 0), (1., 1.))),
        ("positive dyadic scaling", Gauge((0, 1), (2., .5))),
        ("combined permutation/scaling", Gauge((1, 0), (2., .5))),
    )
    for name, gauge in gauges:
        target = transform(base, gauge)
        transported_parent, transported_program = transport(base, program, gauge)
        assert target.sha256 == transported_parent.sha256
        assert transported_program.edits
        transported_edit = apply_program(target, transported_program)
        for value in (-2., -.5, 0., .25, 1., 2.):
            inputs = [value]
            assert exact_score(base, inputs) == exact_score(target, inputs), name
            assert exact_score(edited, inputs) == exact_score(transported_edit, inputs), name
        raw_edit = apply_program(target, transfer_edit(base, program, target))
        raw_score = exact_score(raw_edit, witness)
        assert raw_score != edited_score, name
        print(f"{name}: exact parent/edit controls agree; "
              f"raw transfer at x=1 gives {raw_score[0]}, expected {edited_score[0]}.")

    tanh = sequential_artifact([[[1.]], [[1.]]], [[0.], [0.]], ["tanh", "identity"])
    scaled_tanh = tanh.replace(**{
        "layers.0.weight": [[2.]], "layers.1.weight": [[.5]],
    })
    before = float(sequential_forward(tanh, witness)[0])
    after = float(sequential_forward(scaled_tanh, witness)[0])
    # A large numerical counterexample, not an equality tolerance or certificate.
    assert before - after > .25
    print(f"Tanh scaling changes the function at x=1: {before:.12g} versus {after:.12g}.")

    deeper_relu = sequential_artifact(
        [[[1.]], [[1.]], [[1.]]], [[0.], [0.], [0.]], ["relu", "relu", "identity"],
    )
    assert float(sequential_forward(deeper_relu, witness)[0]) == 1.
    for name, artifact in (("tanh chain", tanh), ("deeper ReLU chain", deeper_relu)):
        try:
            as_relu_network(artifact)
        except ValueError:
            print(f"Specialized two-layer ReLU conversion rejects the {name}.")
        else:
            raise AssertionError(f"Unsupported conversion accepted: {name}")

    print("Standalone family-boundary fixture passed; frozen-pilot c0 remains partial.")


if __name__ == "__main__":
    main()
