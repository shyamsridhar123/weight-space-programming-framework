"""Data-only weight programs for caller-supplied PyTorch modules.

No model class is imported or reconstructed from an artifact. Architecture and
caller metadata declare the execution interface; arbitrary Python forward code,
global random state, and task behavior are outside the snapshot's guarantee.
The reference format uses binary64 JSON, so native restoration must be exact.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import inspect
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .artifact import TensorArtifact
from .model import Network
from .program import Program, apply_program


BRIDGE_METADATA_KEY = "torch_bridge"
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _module_type(module: nn.Module) -> str:
    return f"{type(module).__module__}.{type(module).__qualname__}"


def _state(module: nn.Module) -> tuple[Mapping[str, torch.Tensor], list[list[str]]]:
    if not isinstance(module, nn.Module):
        raise TypeError("Supply an existing torch.nn.Module")
    state = module.state_dict()
    if not state:
        raise ValueError("The module has no persistent state tensors to edit")
    buffers = dict(module.named_buffers(remove_duplicate=False))
    missing = sorted(set(buffers) - set(state))
    if missing:
        raise ValueError(f"Nonpersistent buffers are not captured by state_dict: {missing}")
    storages: dict[tuple[str, int], list[str]] = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Unsupported non-tensor or extra state at {name}")
        if torch.nn.parameter.is_lazy(tensor) or tensor.is_meta:
            raise ValueError(f"Initialize concrete tensor storage before snapshotting {name}")
        if tensor.layout != torch.strided or tensor.dtype not in _FLOAT_DTYPES:
            raise ValueError(f"Unsupported state tensor {name}: {tensor.dtype}, {tensor.layout}; "
                             "only dense floating state is supported, including buffers")
        if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"State tensor {name} must be nonempty and finite")
        storage = (str(tensor.device), tensor.untyped_storage().data_ptr())
        storages.setdefault(storage, []).append(name)
    aliases = []
    for names in storages.values():
        if len(names) < 2:
            continue
        first = state[names[0]]
        if any((tensor.shape, tensor.stride(), tensor.storage_offset(), tensor.dtype)
               != (first.shape, first.stride(), first.storage_offset(), first.dtype)
               for tensor in (state[name] for name in names[1:])):
            raise ValueError(f"Different views of shared storage are unsupported: {names}")
        aliases.append(sorted(names))
    return state, sorted(aliases)


def snapshot_torch(module: nn.Module, *, architecture: str,
                   metadata: dict[str, Any] | None = None) -> TensorArtifact:
    """Capture persistent floating parameters/buffers and their native interface.

    The caller must identify forward semantics and any non-state configuration
    in ``architecture``/``metadata``. Module modes and dropout configuration are
    captured without calling ``eval()`` or changing the source module. Integer
    buffers, extra state, sparse/quantized state, and nonpersistent buffers are
    explicitly unsupported; nothing from state_dict is silently dropped.
    """
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Caller metadata must be a JSON object")
    if metadata is not None and BRIDGE_METADATA_KEY in metadata:
        raise ValueError(f"{BRIDGE_METADATA_KEY} is reserved bridge metadata")
    state, aliases = _state(module)
    parameters = dict(module.named_parameters(remove_duplicate=False))
    buffers = dict(module.named_buffers(remove_duplicate=False))
    try:
        forward_signature = str(inspect.signature(module.forward))
    except (TypeError, ValueError):
        forward_signature = "unavailable; caller declares the architecture/interface"
    context = copy.deepcopy(metadata or {})
    context[BRIDGE_METADATA_KEY] = {
        "schema": "wsp-torch-bridge/1", "torch_version": str(torch.__version__),
        "module_type": _module_type(module), "module_repr": repr(module),
        "forward_signature": forward_signature,
        "modules": {name: {"type": _module_type(child), "training": bool(child.training)}
                    for name, child in module.named_modules(remove_duplicate=False)},
        "state": {name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                         "device": str(tensor.device),
                         "role": "parameter" if name in parameters else "buffer" if name in buffers else "state",
                         "requires_grad": bool(parameters[name].requires_grad) if name in parameters else False}
                  for name, tensor in state.items()},
        "alias_groups": aliases,
        "execution_scope": "Caller-supplied forward code; modes are captured, global RNG state and behavior are not certified.",
    }
    arrays = {name: tensor.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
              for name, tensor in state.items()}
    return TensorArtifact(architecture, arrays, context)


def apply_torch_program(module: nn.Module, program: Program, *, architecture: str,
                        metadata: dict[str, Any] | None = None) -> nn.Module:
    """Apply an exact-parent program to a detached copy, preserving native modes.

    All native dtype conversions are checked before loading the copied module.
    A binary64 edit that would round, overflow, or break a tied weight is rejected.
    Use the same caller metadata supplied to ``snapshot_torch``; bridge metadata
    is reconstructed from the actual source, never trusted from program data.
    """
    base = snapshot_torch(module, architecture=architecture, metadata=metadata)
    candidate = apply_program(base, program)
    state, aliases = _state(module)
    for names in aliases:
        if any(not np.array_equal(candidate.tensors[names[0]], candidate.tensors[name])
               for name in names[1:]):
            raise ValueError(f"Edits must agree across tied state tensors: {names}")
    native = OrderedDict()
    for name, original in state.items():
        values = np.array(candidate.tensors[name], copy=True)
        restored = torch.from_numpy(values).to(device=original.device, dtype=original.dtype)
        roundtrip = restored.detach().to(device="cpu", dtype=torch.float64).numpy()
        if not np.isfinite(roundtrip).all() or not np.array_equal(roundtrip, values):
            raise ValueError(f"Edit to {name} is not exactly representable as native {original.dtype}; "
                             "rounding or nonfinite conversion is forbidden")
        native[name] = restored
    # PyTorch uses per-module state_dict version metadata during strict loading.
    if hasattr(state, "_metadata"):
        native._metadata = copy.deepcopy(state._metadata)
    edited = copy.deepcopy(module)
    if snapshot_torch(edited, architecture=architecture, metadata=metadata).sha256 != base.sha256:
        raise ValueError("Deep copying changed the module's captured state or interface")
    edited.load_state_dict(native, strict=True)
    if snapshot_torch(edited, architecture=architecture, metadata=metadata).sha256 != candidate.sha256:
        raise ValueError("Loaded module does not match the exact edited artifact")
    return edited


def network_from_torch(module: nn.Module, *, input_names: Sequence[str],
                       output_names: Sequence[str]) -> Network:
    """Extract the supported mathematical Linear-ReLU-Linear specialization.

    Only an ordinary three-layer ``nn.Sequential`` with exact built-in layer
    types is accepted. Custom forwards and active forward hooks are rejected.
    The result lifts stored weights to binary64; this does not certify PyTorch
    floating-point execution or confer ReLU proofs on arbitrary modules.
    """
    if type(module) is not nn.Sequential:
        raise ValueError("ReLU extraction requires an ordinary nn.Sequential")
    layers = list(module.children())
    if len(layers) != 3 or tuple(map(type, layers)) != (nn.Linear, nn.ReLU, nn.Linear):
        raise ValueError("ReLU extraction requires exactly Linear, ReLU, Linear")
    for child in (module, *layers):
        if "forward" in child.__dict__ or child._forward_hooks or child._forward_pre_hooks:
            raise ValueError("Custom forwards and forward hooks are outside ReLU extraction semantics")
    first, _, last = layers
    if (isinstance(input_names, (str, bytes)) or isinstance(output_names, (str, bytes))
            or len(input_names) != first.in_features or len(output_names) != last.out_features):
        raise ValueError("Supply explicit dimension-matched input and output labels")
    if first.out_features != last.in_features:
        raise ValueError("The two linear layers have incompatible dimensions")
    _state(module)  # Apply the same finite/native-state admission boundary.

    def array(tensor: torch.Tensor | None, width: int) -> np.ndarray:
        return np.zeros(width) if tensor is None else tensor.detach().to(device="cpu", dtype=torch.float64).numpy()

    return Network(array(first.weight, 0), array(first.bias, first.out_features),
                   array(last.weight, 0), array(last.bias, last.out_features),
                   tuple(input_names), tuple(output_names))
