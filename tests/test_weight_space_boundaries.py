"""Regressions for immutable identities and exact binary64 admission boundaries."""
import copy
import json
import warnings

import numpy as np
import pytest

from rlcd.weight_space import (
    Contract, DecisionRule, Edit, InputBox, Network, Program, StructuralPolicy, TaskSpec, TensorArtifact,
    admit, apply_program, bind_admission, certify_drift, check_contract, difference,
    replay_admission, rollback, verify_certificate,
)
from rlcd.weight_space.contracts import rational


def network():
    return Network([[1., 2.], [3., 4.]], [0., 1.], [[1., -1.]], [0.])


def test_numpy_semantic_labels_are_captured_before_admission_json():
    inputs, outputs = (" β ", "alpha"), ("positive", "negative")
    input_names, output_names = np.array(inputs), np.array(outputs)
    model = Network([[1., 2.]], [0.], [[1.], [-1.]], [.5, -.5],
                    input_names=tuple(input_names), output_names=tuple(output_names))
    plain = Network(**model.tensors, input_names=inputs, output_names=outputs)
    input_names[0], output_names[0] = "changed", "changed"
    assert model.input_names == inputs and model.output_names == outputs
    assert all(type(name) is str for name in (*model.input_names, *model.output_names))
    assert model.to_dict() == plain.to_dict() and model.sha256 == plain.sha256
    restored = Network.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.sha256 == model.sha256
    for name in model.tensors:
        np.testing.assert_array_equal(restored.tensors[name], plain.tensors[name])

    request = bind_admission(
        model, model, Program(model.sha256, (), "Synthetic label boundary no-op"),
        task=TaskSpec("numpy-label-boundary", "edit", "Check label capture",
                      "dense-relu-dense/1", model.sha256),
        policy=StructuralPolicy(("b2",), 0, 2., 0.),
        box=InputBox([0., 0.], [1., 1.]), max_output_drift=(0., 0.),
        decision=DecisionRule("scores_only"), witnesses=[[0., 0.]],
    )
    record = admit(request)
    assert record.accepted
    assert replay_admission(record.to_dict(), expected_binding_sha256=request.sha256).to_dict() == record.to_dict()

    class StringSubtype(str):
        def __str__(self):
            raise AssertionError("Custom label conversion must not execute")

    class NumpyStringSubtype(np.str_):
        __str__ = StringSubtype.__str__

    class StringTypeEqualityTrap(type):
        def __eq__(cls, other):
            raise AssertionError("Custom label type equality must not execute")

    class MetaclassStringSubtype(StringSubtype, metaclass=StringTypeEqualityTrap):
        pass

    for invalid in (
        (StringSubtype("beta"), "alpha"), (NumpyStringSubtype("beta"), "alpha"),
        (MetaclassStringSubtype("beta"), "alpha"),
        (np.bytes_("beta"), "alpha"), (object(), "alpha"),
        (np.str_(" "), "alpha"), (np.str_("alpha"), "alpha"), (np.str_("alpha"),),
    ):
        with pytest.raises(ValueError, match="Semantic labels"):
            Network(**model.tensors, input_names=invalid, output_names=outputs)


def mutate_headers(array):
    """Exercise the data and header of every accessible ndarray in the chain."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        while isinstance(array, np.ndarray):
            with pytest.raises(ValueError):
                array.flags.writeable = True
            with pytest.raises(ValueError):
                array.flat[0] = 0
            array.shape = (array.size,)
            array.dtype = np.int64
            array = array.base


def test_retrieved_array_headers_cannot_change_stored_identity_or_behavior():
    model = network()
    artifact = TensorArtifact("audit/1", {"x": [[1., 2.]], "scalar": np.array(3.)})
    box = InputBox([0., 0.], [1., 1.])
    model_data, artifact_data, box_data = model.to_dict(), artifact.to_dict(), box.to_dict()
    model_id, artifact_id = model.sha256, artifact.sha256
    score = model([.25, .75])
    certificate = certify_drift(model, model, box)

    for array in (*model.tensors.values(), model.w1, model.b1, model.w2, model.b2,
                  *artifact.tensors.values(), box.lower, box.upper):
        mutate_headers(array)

    assert model.to_dict() == model_data and model.sha256 == model_id
    assert artifact.to_dict() == artifact_data and artifact.sha256 == artifact_id
    assert model.shape == (2, 2, 1) and artifact.tensors["scalar"].shape == ()
    assert box.to_dict() == box_data
    np.testing.assert_array_equal(model([.25, .75]), score)
    np.testing.assert_array_equal(box.contains([[0., 1.], [0., 2.]]), [True, False])
    assert certify_drift(model, model, box) == certificate
    assert verify_certificate(model, model, certificate)
    assert all(t.dtype == np.float64 for t in (*model.tensors.values(), *artifact.tensors.values()))


def test_tensor_name_self_supports_direct_replacement_application_and_rollback():
    base = TensorArtifact("audit/1", {"self": [1.]})
    candidate = base.replace(self=[2.])
    program = difference(base, candidate, "Edit a valid tensor named self")
    assert apply_program(base, program).sha256 == candidate.sha256
    assert apply_program(candidate, rollback(base, program)).sha256 == base.sha256


INVALID_SCALARS = [
    pytest.param(True, id="bool"), pytest.param(np.bool_(False), id="numpy-bool"),
    pytest.param(2**53 + 1, id="rounded-down-int"),
    pytest.param(2**53 + 3, id="rounded-up-int"),
    pytest.param(-(2**53 + 1), id="negative-rounded-int"),
    pytest.param(np.int64(2**53 + 1), id="numpy-int"),
    pytest.param(np.uint64(2**64 - 1), id="numpy-uint"),
    pytest.param(10**10000, id="huge-int"),
    pytest.param(float("inf"), id="infinite"),
    pytest.param(float("nan"), id="nan"), pytest.param("1", id="string"),
]


@pytest.mark.parametrize("value", INVALID_SCALARS)
def test_edits_and_budgets_reject_inexact_or_non_numeric_scalars(value):
    box = InputBox([0.], [1.])
    constructors = (
        lambda: Edit("x", (), value, 0.), lambda: Edit("x", (), 0., value),
        lambda: StructuralPolicy(("x",), 1, value, 1.),
        lambda: StructuralPolicy(("x",), 1, 1., value),
        lambda: Contract(box, (value,)),
        lambda: Contract(box, (1.,), max_parameter_abs=value),
        lambda: rational(value),
    )
    for construct in constructors:
        with pytest.raises(ValueError, match="binary64"):
            construct()


@pytest.mark.parametrize("value", [1, 2**53 + 2, 2**1000, np.int64(2**53 + 2), np.float32(.5)])
def test_exact_scalars_are_normalized_and_edited_programs_roundtrip(value):
    base = TensorArtifact("audit/1", {"x": np.array(0.)})
    edit = Edit("x", (), 0, value)
    assert type(edit.before) is float and type(edit.after) is float
    program = Program(base.sha256, (edit,), "An exactly representable change")
    restored = Program.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored.sha256 == program.sha256
    assert apply_program(base, restored).sha256 != base.sha256
    assert float(apply_program(base, restored).tensors["x"]) == float(value)
    policy = StructuralPolicy(("x",), 1, value, value)
    assert type(policy.max_parameter_abs) is float and type(policy.max_change_abs) is float
    assert policy.check(base, restored)["structural_accepted"]
    contract = Contract(InputBox([0.], [1.]), (value,), max_parameter_abs=value)
    assert type(contract.max_parameter_abs) is float and type(contract.max_output_drift[0]) is float


def test_unrepresentable_edit_cannot_collapse_to_noop_and_signed_zero_is_canonical():
    with pytest.raises(ValueError, match="binary64"):
        Edit("x", (0,), float(2**53), 2**53 + 1)
    with pytest.raises(ValueError, match="Unchanged"):
        Edit("x", (0,), -0.0, 0)
    assert Edit("x", (0,), -0.0, 1).to_dict()["before"] == 0.0
    assert not np.signbit(Edit("x", (0,), -0.0, 1).before)


def test_rounded_up_integer_limits_cannot_falsely_admit_a_larger_edit():
    actual, smaller, rounded = 2**53 + 4, 2**53 + 2, 2**53 + 3
    base = TensorArtifact("audit/1", {"x": [0.]})
    program = Program(base.sha256, (Edit("x", (0,), 0., actual),), "Exact large edit")
    with pytest.raises(ValueError, match="binary64"):
        StructuralPolicy(("x",), 1, rounded, rounded)
    receipt = StructuralPolicy(("x",), 1, smaller, smaller).check(base, program)
    assert set(receipt["issues"]) == {"parameter-bound", "change-bound"}
    assert StructuralPolicy(("x",), 1, actual, actual).check(base, program)["structural_accepted"]

    model = Network([[0.]], [0.], [[0.]], [0.])
    edit = difference(model, model.replace(b2=[actual]), "Exact large output drift")
    box = InputBox([0.], [1.])
    with pytest.raises(ValueError, match="binary64"):
        Contract(box, (rounded,), max_parameter_abs=actual)
    with pytest.raises(ValueError, match="binary64"):
        Contract(box, (actual,), max_parameter_abs=rounded)
    receipt = check_contract(model, edit, Contract(box, (smaller,), max_parameter_abs=smaller))
    assert set(receipt["issues"]) == {"parameter-bound", "behavioral-drift"}
    assert check_contract(model, edit, Contract(box, (actual,), max_parameter_abs=actual))["accepted"]


@pytest.mark.parametrize("values", [
    [2**53 + 1, 0.], [2**53 + 3, 1], [True, 1.], [False, 1], ["1", 2],
    np.array([2**53 + 1], dtype=np.int64), np.array([2**64 - 1], dtype=np.uint64),
    np.array([True]), np.array([1.], dtype=object),
    pytest.param([10**10000], id="huge-integer-list"),
])
def test_tensor_conversion_rejects_lost_integer_precision_and_mixed_booleans(values):
    with pytest.raises(ValueError):
        TensorArtifact("audit/1", {"x": values})


def test_exact_integer_tensor_extremes_remain_supported():
    artifact = TensorArtifact("audit/1", {
        "signed": np.array([-2**63, 2**53 + 2], dtype=np.int64),
        "unsigned": np.array([2**63, 2**64 - 2048], dtype=np.uint64),
        "mixed": [2**53 + 2, .5], "large": [2**1000],
    })
    assert TensorArtifact.from_dict(artifact.to_dict()).sha256 == artifact.sha256
    assert int(artifact.tensors["unsigned"][1]) == 2**64 - 2048
    assert int(artifact.tensors["large"][0]) == 2**1000
    with pytest.raises(ValueError, match="binary64"):
        Network([[2**53 + 1]], [0.], [[1.]], [0.])
    with pytest.raises(ValueError, match="binary64"):
        InputBox([0.], [2**53 + 1])


@pytest.mark.parametrize("bad", ["", {}, None, 0, False])
def test_program_serialization_requires_edit_and_coordinate_arrays(bad):
    base = TensorArtifact("audit/1", {"x": np.array(0.)})
    data = Program(base.sha256, (Edit("x", (), 0., 1.),), "Scalar edit").to_dict()
    with pytest.raises(ValueError):
        Program.from_dict(data | {"edits": bad})
    malformed = copy.deepcopy(data)
    malformed["edits"][0]["index"] = bad
    with pytest.raises(ValueError):
        Program.from_dict(malformed)
    with pytest.raises(ValueError):
        Program(base.sha256, bad, "Invalid direct constructor")
    with pytest.raises(ValueError):
        Edit("x", bad, 0., 1.)


@pytest.mark.parametrize("loader", [Network.from_dict, TensorArtifact.from_dict, Program.from_dict, Contract.from_dict])
@pytest.mark.parametrize("bad", [None, "schema", [], ["schema"]])
def test_serialized_schemas_require_objects(loader, bad):
    with pytest.raises(ValueError):
        loader(bad)


def test_metadata_labels_contract_fields_and_parent_ids_keep_declared_types():
    artifact = TensorArtifact("audit/1", {"x": [0.]})
    with pytest.raises(ValueError):
        TensorArtifact.from_dict(artifact.to_dict() | {"metadata": None})
    with pytest.raises(ValueError):
        Network.from_dict(network().to_dict() | {"input_names": "xy"})
    with pytest.raises(ValueError):
        Network.from_dict(network().to_dict() | {"output_names": {"y": 1}})
    with pytest.raises(ValueError):
        Program(["a"] * 64, (), "Invalid parent type")
    contract = Contract(InputBox([0.], [1.]), (1.,)).to_dict()
    for key, value in (("box", []), ("max_output_drift", "1"), ("allowed_tensors", "w1")):
        with pytest.raises(ValueError):
            Contract.from_dict(contract | {key: value})


def test_certificate_verification_distinguishes_boolean_claims_from_numbers():
    model = network()
    certificate = certify_drift(model, model, InputBox([0., 0.], [1., 1.]))
    assert verify_certificate(model, model, certificate)
    assert not verify_certificate(model, model, certificate | {"implementation_verified": 0})
