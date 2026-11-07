"""Generic native execution and the explicit ReLU extraction boundary."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rlcd.weight_space.artifact import TensorArtifact
from rlcd.weight_space.program import Edit, Program, difference
from rlcd.weight_space.torch_bridge import (
    BRIDGE_METADATA_KEY, apply_torch_program, network_from_torch, snapshot_torch,
)


def tanh_model():
    with torch.random.fork_rng():
        torch.manual_seed(917041)
        model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 1))
    with torch.no_grad():
        model[-1].bias.zero_()
    return model


def test_generic_deep_tanh_edit_roundtrips_and_executes_without_mutating_original():
    model = tanh_model().eval()
    metadata = {"inputs": ["left", "right"], "outputs": ["response"]}
    base = snapshot_torch(model, architecture="test.deep-tanh/v1", metadata=metadata)
    assert TensorArtifact.from_dict(json.loads(json.dumps(base.to_dict()))).sha256 == base.sha256
    assert base.metadata[BRIDGE_METADATA_KEY]["state"]["4.bias"]["dtype"] == "torch.float32"
    candidate = base.replace(**{"4.bias": np.array([.5])})
    program = Program.from_dict(difference(base, candidate, "Raise the output by .5").to_dict())
    inputs = torch.tensor([[.2, -.4], [1., .5]])
    before = model(inputs).detach().clone()
    edited = apply_torch_program(model, program, architecture=base.architecture, metadata=metadata)
    assert edited is not model and edited[0].weight.data_ptr() != model[0].weight.data_ptr()
    assert not edited.training
    assert snapshot_torch(edited, architecture=base.architecture, metadata=metadata).sha256 == candidate.sha256
    assert snapshot_torch(model, architecture=base.architecture, metadata=metadata).sha256 == base.sha256
    torch.testing.assert_close(edited(inputs) - before, torch.full_like(before, .5))
    torch.testing.assert_close(model(inputs), before, rtol=0, atol=0)
    metadata["inputs"][0] = "changed"
    assert base.metadata["inputs"][0] == "left"


def test_native_parent_mode_interface_and_rounding_checks_fail_before_mutation():
    model = tanh_model()
    base = snapshot_torch(model, architecture="test.deep-tanh/v1")
    patch = Program(base.sha256, (Edit("4.bias", (0,), 0., .5),), "Representable edit")
    with pytest.raises(ValueError, match="parent"):
        apply_torch_program(model, patch, architecture="other-interface")
    model.eval()
    with pytest.raises(ValueError, match="parent"):
        apply_torch_program(model, patch, architecture=base.architecture)
    model.train()
    with torch.no_grad():
        model[0].bias[0] += 1
    with pytest.raises(ValueError, match="parent"):
        apply_torch_program(model, patch, architecture=base.architecture)
    current = snapshot_torch(model, architecture=base.architecture)
    for after in (.1, 1e100):
        bad = Program(current.sha256, (Edit("4.bias", (0,), 0., after),), "Unrepresentable edit")
        with pytest.raises(ValueError, match="exactly representable"):
            apply_torch_program(model, bad, architecture=base.architecture)
        assert snapshot_torch(model, architecture=base.architecture).sha256 == current.sha256
    with pytest.raises(ValueError, match="reserved"):
        snapshot_torch(model, architecture=base.architecture, metadata={BRIDGE_METADATA_KEY: {"dtype": "fake"}})


def test_dropout_modes_are_bound_and_preserved_without_random_forward_evaluation():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(.25), nn.Linear(2, 1))
    model.train()
    model[1].eval()
    base = snapshot_torch(model, architecture="test.dropout/v1")
    no_op = Program(base.sha256, (), "Preserve mixed modes")
    edited = apply_torch_program(model, no_op, architecture=base.architecture)
    assert [child.training for child in edited.modules()] == [child.training for child in model.modules()]
    model[1].train()
    with pytest.raises(ValueError, match="parent"):
        apply_torch_program(model, no_op, architecture=base.architecture)


def test_arbitrary_rank_tensors_and_floating_buffers_restore_but_integer_state_is_explicitly_rejected():
    class ConvScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 2, bias=False)
            self.scale = nn.Parameter(torch.tensor(1.))
            self.register_buffer("offset", torch.tensor(.25))

        def forward(self, inputs):
            return torch.tanh(self.conv(inputs)) * self.scale + self.offset

    model = ConvScale()
    base = snapshot_torch(model, architecture="test.conv-scale/v1")
    assert base.tensors["conv.weight"].shape == (1, 1, 2, 2)
    assert base.tensors["scale"].shape == ()
    assert base.metadata[BRIDGE_METADATA_KEY]["state"]["offset"]["role"] == "buffer"
    candidate = base.replace(scale=np.array(.5), offset=np.array(.75))
    edited = apply_torch_program(model, difference(base, candidate, "Scale and shift"), architecture=base.architecture)
    inputs = torch.ones(1, 1, 3, 3)
    torch.testing.assert_close(edited(inputs), (model(inputs) - .25) * .5 + .75)
    assert float(model.scale.detach()) == 1. and float(model.offset) == .25
    with pytest.raises(ValueError, match="num_batches_tracked.*torch.int64"):
        snapshot_torch(nn.BatchNorm1d(2), architecture="test.batchnorm/v1")
    model.register_buffer("ephemeral", torch.tensor(2.), persistent=False)
    with pytest.raises(ValueError, match="Nonpersistent"):
        snapshot_torch(model, architecture=base.architecture)


def test_tied_state_edits_must_agree_and_aliases_survive_the_copy():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(2, 2, bias=False)
            self.right = nn.Linear(2, 2, bias=False)
            self.right.weight = self.left.weight

        def forward(self, inputs):
            return self.right(torch.tanh(self.left(inputs)))

    model = Tied()
    base = snapshot_torch(model, architecture="test.tied/v1")
    replacement = np.full((2, 2), .5)
    partial = difference(base, base.replace(**{"left.weight": replacement}), "Break alias")
    with pytest.raises(ValueError, match="tied"):
        apply_torch_program(model, partial, architecture=base.architecture)
    candidate = base.replace(**{"left.weight": replacement, "right.weight": replacement})
    edited = apply_torch_program(model, difference(base, candidate, "Preserve alias"), architecture=base.architecture)
    assert edited.left.weight is edited.right.weight
    assert edited.left.weight is not model.left.weight
    assert snapshot_torch(model, architecture=base.architecture).sha256 == base.sha256


def test_only_explicit_linear_relu_linear_models_enter_the_proven_specialization():
    with torch.random.fork_rng():
        torch.manual_seed(917042)
        module = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2, bias=False)).double().eval()
    network = network_from_torch(module, input_names=("left", "right"), output_names=("yes", "no"))
    inputs = torch.tensor([[-1., .5], [0., 0.], [.5, 1.], [2., -3.]], dtype=torch.float64)
    np.testing.assert_allclose(network(inputs.numpy()), module(inputs).detach().numpy(), rtol=1e-14, atol=1e-14)
    assert network.input_names == ("left", "right") and network.output_names == ("yes", "no")
    with pytest.raises(ValueError, match="exactly Linear"):
        network_from_torch(tanh_model(), input_names=("left", "right"), output_names=("response",))
    with pytest.raises(ValueError, match="explicit"):
        network_from_torch(module, input_names=(), output_names=("yes", "no"))
    hook = module.register_forward_hook(lambda _m, _args, output: output * 2)
    try:
        with pytest.raises(ValueError, match="hooks"):
            network_from_torch(module, input_names=("left", "right"), output_names=("yes", "no"))
    finally:
        hook.remove()

    class CustomSequential(nn.Sequential):
        def forward(self, value):
            return super().forward(value) * 2

    with pytest.raises(ValueError, match="ordinary"):
        network_from_torch(CustomSequential(*module.children()), input_names=("left", "right"), output_names=("yes", "no"))
