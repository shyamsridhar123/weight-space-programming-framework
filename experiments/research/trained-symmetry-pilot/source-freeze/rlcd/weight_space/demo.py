"""Small end-to-end examples that do not import a game or an environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .artifact import StructuralPolicy, TensorArtifact
from .compiler import compile_readout
from .contracts import Contract, check_contract
from .evidence import write_json
from .model import InputBox, Network
from .program import apply_program, difference, rollback
from .semantics import SemanticsAdapter, SemanticsRegistry, to_tensor_artifact


def run_demo(out: str | Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    # A rank-four patch convolution with Tanh and a scalar output. The generic
    # edit engine only sees named tensors; the caller owns this execution code.
    base = TensorArtifact("patch-convolution-tanh/1", {
        "conv.weight": np.array([[[[1., -.5], [0., .5]]], [[[-.5, 0.], [1., .5]]]]),
        "conv.bias": np.array([0., .25]), "head.weight": np.array([[.5, -.25]]),
        "head.bias": np.array(.125),
    }, {"input_shape": [1, 2, 2], "input_channels": ["intensity"], "output_names": ["score"]})
    registry = SemanticsRegistry()
    def evaluate(artifact: TensorArtifact, inputs: Any) -> np.ndarray:
        x, t = np.asarray(inputs, dtype=np.float64), artifact.tensors
        if x.ndim != 4 or x.shape[1:] != (1, 2, 2):
            raise ValueError("Expected a batch of 1x2x2 input patches")
        hidden = np.tanh(np.einsum("nchw,ochw->no", x, t["conv.weight"]) + t["conv.bias"])
        return hidden @ t["head.weight"].T + t["head.bias"]
    registry.register(SemanticsAdapter(base.architecture, evaluate))
    candidate = base.replace(**{"head.weight": [[.75, -.25]], "head.bias": np.array(.25)})
    program = difference(base, candidate, "Increase the first filter readout and output bias; behavior is empirical")
    policy = StructuralPolicy(("head.weight", "head.bias"), 2, 8., .5)
    admission = policy.check(base, program)
    applied = apply_program(base, program)
    inputs = np.array([[[[0., 0.], [0., 0.]]], [[[1., 0.], [0., 1.]]], [[[0., 1.], [1., 0.]]]])
    generic_result = {"base_sha256": base.sha256, "candidate_sha256": applied.sha256,
                      "admission": admission, "inputs": inputs.tolist(),
                      "before": registry.evaluate(base, inputs).tolist(),
                      "after": registry.evaluate(applied, inputs).tolist(),
                      "rollback_restores_parent": apply_program(applied, rollback(base, program)).sha256 == base.sha256,
                      "scope": "generic tensor editing and caller-owned numerical execution; no behavioral proof for Tanh"}
    write_json(out / "generic" / "base.json", base.to_dict())
    write_json(out / "generic" / "candidate.json", applied.to_dict())
    write_json(out / "generic" / "program.json", program.to_dict())
    write_json(out / "generic" / "policy.json", admission["policy"])
    write_json(out / "generic" / "result.json", generic_result)

    relu = Network([[1.], [-1.]], [0., 0.], [[1., 1.]], [0.], ("position",), ("score",))
    spec = {"target_inputs": [[.25], [.5], [1.]], "desired_delta": [[.25], [.5], [1.]],
            "retain_inputs": [[-.25], [-.5], [-1.]], "ridge": 1e-8, "editable_hidden": [0],
            "intent": "Raise positive-position scores while preserving every nonpositive position"}
    compilation = compile_readout(relu, **spec)
    relu_candidate = apply_program(relu, compilation.program)
    contract = Contract(InputBox([-1.], [1.]), (1.,), ("w2",), max_edits=1)
    protected = Contract(InputBox([-1.], [0.]), (0.,), ("w2",), max_edits=1)
    relu_result = {"global": check_contract(relu, compilation.program, contract),
                   "protected": check_contract(relu, compilation.program, protected),
                   "diagnostics": compilation.diagnostics,
                   "scope": "exact rational real-score bounds; float execution and task utility remain separate"}
    if not relu_result["global"]["accepted"] or not relu_result["protected"]["accepted"]:
        raise ValueError("The ReLU demonstration did not satisfy its declared contracts")
    for name, value in (("base", relu.to_dict()), ("candidate", relu_candidate.to_dict()),
                        ("tensor-base", to_tensor_artifact(relu).to_dict()), ("spec", spec),
                        ("program", compilation.program.to_dict()), ("contract", contract.to_dict()),
                        ("protected-contract", protected.to_dict()), ("result", relu_result)):
        write_json(out / "relu" / f"{name}.json", value)
    result = {"schema": "wsp-generic-demo/1", "generic": generic_result, "relu": relu_result}
    write_json(out / "results.json", result)
    return result
