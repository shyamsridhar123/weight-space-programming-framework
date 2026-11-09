"""Data-only tensor programs and explicit projection to native floating formats.

Structured commands are lowered to the existing exact-parent literal Program.
They never name Python imports or execute provider-authored source code. Projection
is a proposal operation: the resulting endpoint still needs admission/evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from typing import Any, Mapping

import numpy as np

from .artifact import WeightArtifact, valid_tensor_name
from .model import _TensorData, canonical_bytes, digest, finite_binary64
from .program import Program, difference


NATIVE_DTYPES = ("float64", "float32", "float16", "bfloat16")


def _array(value: Any) -> np.ndarray:
    return _TensorData.capture(value).array()


def _validate_command(command: Any) -> None:
    if not isinstance(command, dict):
        raise ValueError("Commands must be JSON objects")
    kind = command.get("op")
    operands = {"replace": {"values"}, "add": {"values"}, "scale": {"factor"},
                "low_rank": {"left", "right", "alpha"}}
    if not isinstance(kind, str) or kind not in operands:
        raise ValueError("Unsupported structured operation")
    required = {"op", "tensor"} | operands[kind]
    if set(command) not in (required, required | {"block"}):
        raise ValueError("Unknown or missing command fields")
    if not valid_tensor_name(command["tensor"]):
        raise ValueError("Invalid tensor name")
    if "block" in command:
        block = command["block"]
        if not isinstance(block, list) or any(
            not isinstance(axis, list) or len(axis) != 2 or
            any(type(n) is not int for n in axis) or axis[0] < 0 or axis[1] <= axis[0]
            for axis in block
        ):
            raise ValueError("Blocks contain half-open nonempty [start, stop] axes")
    for key in operands[kind]:
        if key in ("factor", "alpha"):
            finite_binary64(command[key])
        else:
            _array(command[key])


class StructuredProgram:
    """Immutable JSON commands with sequential, atomic lowering semantics."""

    __slots__ = ("_bytes",)

    def __init__(self, base_sha256: str, commands: list[dict[str, Any]], intent: str):
        Program(base_sha256, (), intent)  # Reuse the literal identity/intent contract.
        if not isinstance(commands, list):
            raise ValueError("Commands must be a JSON array")
        for command in commands:
            _validate_command(command)
        object.__setattr__(self, "_bytes", canonical_bytes({
            "schema": "wsp-structured-program/1", "base_sha256": base_sha256,
            "intent": intent, "commands": commands,
        }))

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Structured programs are immutable")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._bytes)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StructuredProgram:
        if (not isinstance(value, dict) or
                set(value) != {"schema", "base_sha256", "intent", "commands"} or
                value["schema"] != "wsp-structured-program/1"):
            raise ValueError("Unknown structured-program schema or fields")
        return cls(value["base_sha256"], value["commands"], value["intent"])

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


def native_round(values: Any, dtype: str) -> np.ndarray:
    """Return the finite endpoint actually representable in a declared format.

    bfloat16 uses the installed PyTorch conversion, recorded as a trusted runtime
    dependency. This is measured conversion, not a proof of hardware rounding.
    """
    if dtype not in NATIVE_DTYPES:
        raise ValueError(f"Unsupported native floating format: {dtype}")
    source = _array(values)
    if dtype == "bfloat16":
        try:
            import torch
        except ImportError as error:
            raise ValueError("bfloat16 projection requires the optional PyTorch bridge") from error
        result = torch.from_numpy(source.copy()).to(torch.bfloat16).to(torch.float64).numpy()
    else:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            result = source.astype(dtype).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Native projection overflowed to a nonfinite value")
    return _array(result)


@dataclass(frozen=True)
class Lowering:
    program: Program
    _receipt_bytes: bytes

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_bytes)


def lower_structured(base: WeightArtifact, proposal: StructuredProgram, *,
                     native_dtypes: Mapping[str, str] | None = None,
                     cast: str = "reject") -> Lowering:
    """Apply commands to a temporary copy, project once, and bind literal edits.

    ``reject`` refuses any rounding. ``round`` explicitly records every rounded
    endpoint. The *base* must already be representable in its declared formats,
    so projection cannot silently change unrelated source state.
    """
    if cast not in ("reject", "round"):
        raise ValueError("Cast policy must be reject or round")
    data = proposal.to_dict()
    if data["base_sha256"] != base.sha256:
        raise ValueError("Structured proposal has a stale or wrong parent")
    original = base.tensors
    dtypes = ({name: "float64" for name in original} if native_dtypes is None
              else dict(native_dtypes))
    if set(dtypes) != set(original) or any(t not in NATIVE_DTYPES for t in dtypes.values()):
        raise ValueError("Native formats must cover every captured tensor exactly")
    for name, array in original.items():
        if not np.array_equal(native_round(array, dtypes[name]), array):
            raise ValueError(f"Base tensor {name} is not representable in its declared format")
    values = {name: array.copy() for name, array in original.items()}
    touched = {name: np.zeros(array.shape, dtype=bool) for name, array in original.items()}
    for command in data["commands"]:
        name = command["tensor"]
        if name not in values:
            raise ValueError("Command names an undeclared tensor")
        target = values[name]
        if "block" in command:
            block = command["block"]
            if len(block) != target.ndim or any(stop > size for (_, stop), size in zip(block, target.shape)):
                raise ValueError("Block is outside its tensor")
            index = tuple(slice(start, stop) for start, stop in block)
        else:
            index = tuple(slice(None) for _ in target.shape)
        current = target[index]
        kind = command["op"]
        with np.errstate(over="raise", invalid="raise"):
            if kind in ("replace", "add"):
                operand = _array(command["values"])
                if operand.shape != current.shape:
                    raise ValueError("Block operand shape must match exactly; broadcasting is forbidden")
                updated = operand if kind == "replace" else current + operand
            elif kind == "scale":
                updated = current * command["factor"]
            else:
                left, right = _array(command["left"]), _array(command["right"])
                if (current.ndim != 2 or left.ndim != 2 or right.ndim != 2 or
                        left.shape[1] != right.shape[0] or
                        (left.shape[0], right.shape[1]) != current.shape):
                    raise ValueError("Low-rank factors must multiply to the selected matrix shape")
                updated = current + command["alpha"] * (left @ right)
        target[index] = _array(updated)
        touched[name][index] = True
    requested = base.replace(**values)
    endpoints, projections = {}, {}
    for name, array in values.items():
        endpoint = native_round(array, dtypes[name])
        changed = array != endpoint
        if cast == "reject" and np.any(changed):
            raise ValueError(f"Native format would round tensor {name}; explicitly choose cast='round'")
        errors = [abs(Fraction.from_float(float(a))-Fraction.from_float(float(b)))
                  for a, b in zip(array[changed].flat, endpoint[changed].flat)]
        projections[name] = {
            "dtype": dtypes[name], "touched_cells": int(touched[name].sum()),
            "requested_changed_cells": int(np.count_nonzero(original[name] != array)),
            "effective_changed_cells": int(np.count_nonzero(original[name] != endpoint)),
            "rounded_cells": int(changed.sum()),
            "maximum_rounding_error_exact": str(max(errors, default=Fraction(0))),
        }
        endpoints[name] = endpoint
    candidate = base.replace(**endpoints)
    program = difference(base, candidate, data["intent"])
    payload = {
        "schema": "wsp-lowering/1", "proposal_sha256": proposal.sha256,
        "base_sha256": base.sha256, "requested_candidate_sha256": requested.sha256,
        "candidate_sha256": candidate.sha256, "program_sha256": program.sha256,
        "command_count": len(data["commands"]), "cast": cast,
        "native_dtypes": dtypes, "projections": projections,
        "effective_edits": len(program.edits), "behavioral_status": "not_checked",
        "numerics": "NumPy binary64 operations; explicit native endpoint projection",
        "numpy_version": np.__version__,
    }
    if "bfloat16" in dtypes.values():
        import torch
        payload["torch_version"] = torch.__version__
    return Lowering(program, canonical_bytes(payload | {"sha256": digest(payload)}))


def verify_lowering(base: WeightArtifact, proposal: StructuredProgram,
                    receipt: dict[str, Any]) -> bool:
    """Recompute the lowering; compare JSON types as well as numerical values."""
    try:
        actual = lower_structured(base, proposal, native_dtypes=receipt["native_dtypes"],
                                  cast=receipt["cast"]).receipt
        return canonical_bytes(actual) == canonical_bytes(receipt)
    except (ValueError, TypeError, KeyError, FloatingPointError):
        return False


@dataclass(frozen=True)
class ExecutionSpec:
    """One measured operator/backend model, never a universal float proof."""

    backend: str = "numpy"
    dtype: str = "float64"
    device: str = "cpu"
    accumulation: str = "matmul"
    endpoint_cast: str = "reject"
    error_model: str = "empirical"

    def __post_init__(self) -> None:
        import re
        if (self.backend not in ("numpy", "torch") or self.dtype not in NATIVE_DTYPES
                or not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device)
                or self.accumulation not in ("matmul", "left_to_right", "right_to_left")
                or self.endpoint_cast not in ("reject", "round")
                or self.error_model not in ("empirical", "formal")):
            raise ValueError("Unknown native execution scope")

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "dtype": self.dtype, "device": self.device,
                "accumulation": self.accumulation, "endpoint_cast": self.endpoint_cast,
                "error_model": self.error_model}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExecutionSpec:
        if not isinstance(value, dict) or set(value) != {
                "backend", "dtype", "device", "accumulation", "endpoint_cast", "error_model"}:
            raise ValueError("Unknown execution-spec fields")
        return cls(**value)


def _execution_layers(artifact: WeightArtifact) -> list[tuple[np.ndarray, np.ndarray, str]]:
    from .model import Network
    from .semantics import RELU_ARCHITECTURE, as_relu_network
    from .sequential import SEQUENTIAL_ARCHITECTURE, validate_sequential
    if isinstance(artifact, Network):
        model = artifact
    elif artifact.architecture == RELU_ARCHITECTURE:
        model = as_relu_network(artifact)
    elif artifact.architecture == SEQUENTIAL_ARCHITECTURE:
        tensors = artifact.tensors
        return [(tensors[layer["weight"]], tensors[layer["bias"]], layer["activation"])
                for layer in validate_sequential(artifact)]
    else:
        raise ValueError("Native accounting has no operator model for this architecture")
    return [(model.w1, model.b1, "relu"), (model.w2, model.b2, "identity")]


def _exact_execution(layers: list[tuple[np.ndarray, np.ndarray, str]],
                     inputs: np.ndarray) -> list[list[Fraction]]:
    q = lambda value: Fraction.from_float(float(value))
    result = []
    for sample in inputs:
        values = list(map(q, sample))
        for weights, bias, activation in layers:
            values = [q(b) + sum((q(w)*x for w, x in zip(row, values)), Fraction(0))
                      for row, b in zip(weights, bias)]
            if activation == "relu":
                values = [max(Fraction(0), value) for value in values]
            elif activation == "abs":
                values = list(map(abs, values))
            elif activation != "identity":
                raise ValueError("Exact real execution is unsupported for this activation")
        result.append(values)
    return result


def _native_execution(layers: list[tuple[np.ndarray, np.ndarray, str]],
                      inputs: np.ndarray, spec: ExecutionSpec) -> np.ndarray:
    if spec.backend == "numpy":
        if spec.dtype == "bfloat16" or spec.device != "cpu":
            raise ValueError("NumPy execution requires CPU float64/float32/float16")
        dtype = np.dtype(spec.dtype)
        values = inputs.astype(dtype)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for weights, bias, activation in layers:
                weight, offset = weights.astype(dtype), bias.astype(dtype)
                if spec.accumulation == "matmul":
                    values = values @ weight.T + offset
                else:
                    summed = np.zeros((len(values), len(offset)), dtype=dtype)
                    order = range(weight.shape[1])
                    if spec.accumulation == "right_to_left":
                        order = reversed(order)
                    for j in order:
                        summed = summed + values[:, j:j+1]*weight[:, j]
                    values = summed + offset
                if activation == "relu":
                    values = np.maximum(values, dtype.type(0))
                elif activation == "abs":
                    values = np.abs(values)
                elif activation == "tanh":
                    values = np.tanh(values)
        return values.astype(np.float64)
    import torch
    dtype, device = getattr(torch, spec.dtype), torch.device(spec.device)
    with torch.no_grad():
        values = torch.tensor(inputs.copy(), dtype=dtype, device=device)
        for weights, bias, activation in layers:
            weight = torch.tensor(weights.copy(), dtype=dtype, device=device)
            offset = torch.tensor(bias.copy(), dtype=dtype, device=device)
            if spec.accumulation == "matmul":
                values = values @ weight.T + offset
            else:
                summed = torch.zeros((len(values), len(offset)), dtype=dtype, device=device)
                order = range(weight.shape[1])
                if spec.accumulation == "right_to_left":
                    order = reversed(order)
                for j in order:
                    summed = summed + values[:, j:j+1]*weight[:, j]
                values = summed + offset
            if activation == "relu":
                values = torch.relu(values)
            elif activation == "abs":
                values = torch.abs(values)
            elif activation == "tanh":
                values = torch.tanh(values)
        return values.to(device="cpu", dtype=torch.float64).numpy()


def native_execution_receipt(base: WeightArtifact, candidate: WeightArtifact, inputs: Any,
                             spec: ExecutionSpec = ExecutionSpec(), *, decision: Any = None,
                             binding_sha256: str | None = None) -> dict[str, Any]:
    """Retain endpoint/input quantization, intermediate errors, and decisions.

    Exact outputs concern mathematical affine/identity/ReLU/abs paths. Native
    values are measurements on the pinned backend. Backend matmul may reassociate
    or fuse arithmetic; the two serial arms expose specified multiply/add order.
    """
    from .contracts import DecisionRule, _result_status
    from .evidence import numerical_environment
    if not base.compatible(candidate) or not isinstance(spec, ExecutionSpec):
        raise ValueError("Native comparison needs compatible endpoints and an ExecutionSpec")
    rule = DecisionRule() if decision is None else decision
    if not isinstance(rule, DecisionRule):
        raise ValueError("Supply explicit decision semantics")
    samples = _array(inputs)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2:
        raise ValueError("Native witnesses must be vectors or a nonempty matrix")
    payload: dict[str, Any] = {
        "schema": "wsp-native-execution/1", "binding_sha256": binding_sha256,
        "base_sha256": base.sha256, "candidate_sha256": candidate.sha256,
        "stored_parent": base.to_dict(), "stored_candidate": candidate.to_dict(),
        "no_op": base.sha256 == candidate.sha256,
        "witness_inputs": samples.tolist(), "witness_sha256": digest(samples.tolist()),
        "execution": spec.to_dict(), "decision_rule": rule.to_dict(),
        "operator_order": "dot product then explicit bias then activation; separate multiply/add in serial arms",
        "formal_execution_verified": False,
        "scope": "Empirical paired execution on these witnesses/backend; no all-input or cross-backend guarantee",
    }
    def finish(status: str, reason: str | None = None) -> dict[str, Any]:
        payload["status"] = status
        payload["reason"] = reason
        return payload | {"sha256": digest(payload)}
    try:
        payload["environment"] = numerical_environment(spec.backend, spec.device)
        before_layers, after_layers = _execution_layers(base), _execution_layers(candidate)
    except (ValueError, ImportError, RuntimeError) as error:
        return finish("unsupported", str(error))
    if samples.shape[1] != before_layers[0][0].shape[1]:
        raise ValueError("Native witness coordinate order/width differs from the model")
    payload["operator_coverage"] = [activation for _, _, activation in before_layers]
    if spec.error_model == "formal":
        return finish("unsupported", "No formal native rounding/domain proof backend is registered")
    try:
        projected_base = base.replace(**{name: native_round(tensor, spec.dtype) for name, tensor in base.tensors.items()})
        projected_candidate = candidate.replace(**{name: native_round(tensor, spec.dtype) for name, tensor in candidate.tensors.items()})
        projected_inputs = native_round(samples, spec.dtype)
    except ValueError as error:
        return finish("invalid", "Endpoint/input projection failed: " + str(error))
    payload.update({"projected_parent": projected_base.to_dict(), "projected_candidate": projected_candidate.to_dict(),
                    "projected_parent_sha256": projected_base.sha256,
                    "projected_candidate_sha256": projected_candidate.sha256,
                    "projected_witness_inputs": projected_inputs.tolist(),
                    "parent_projection_changed": projected_base.sha256 != base.sha256,
                    "candidate_projection_changed": projected_candidate.sha256 != candidate.sha256,
                    "input_projection_changed": not np.array_equal(samples, projected_inputs)})
    if spec.endpoint_cast == "reject" and (payload["parent_projection_changed"] or payload["candidate_projection_changed"]):
        return finish("invalid", "Declared endpoint cast forbids quantization; the projected parent is a different baseline")
    exact_available = all(a in ("identity", "relu", "abs") for _, _, a in before_layers)
    payload["exact_real_supported"] = exact_available
    observations = {}
    for name, original, projected in (("parent", base, projected_base), ("candidate", candidate, projected_candidate)):
        original_layers, projected_layers = _execution_layers(original), _execution_layers(projected)
        try:
            native = _native_execution(projected_layers, projected_inputs, spec)
        except (ValueError, RuntimeError, ImportError, TypeError) as error:
            return finish("unsupported", "Native operator execution failed: " + str(error))
        encoded = [[float(x) if np.isfinite(x) else "NaN" if np.isnan(x) else
                    "+Infinity" if x > 0 else "-Infinity" for x in row] for row in native]
        lanes = {"native_outputs": encoded, "native_decisions": [rule.classify(row) for row in native]}
        if exact_available:
            original_exact = _exact_execution(original_layers, samples)
            endpoint_exact = _exact_execution(projected_layers, samples)
            input_exact = _exact_execution(projected_layers, projected_inputs)
            lanes.update({"stored_exact_outputs": [[str(x) for x in row] for row in original_exact],
                          "projected_endpoint_exact_outputs": [[str(x) for x in row] for row in endpoint_exact],
                          "projected_input_exact_outputs": [[str(x) for x in row] for row in input_exact],
                          "stored_exact_decisions": [rule.classify(row) for row in original_exact],
                          "projected_endpoint_exact_decisions": [rule.classify(row) for row in endpoint_exact],
                          "projected_input_exact_decisions": [rule.classify(row) for row in input_exact]})
            effects = []
            for a, b, c, executed in zip(original_exact, endpoint_exact, input_exact, native):
                effects.append([{"endpoint_error_exact": str(y-x), "input_error_exact": str(z-y),
                                 "intermediate_error_exact": None if not np.isfinite(n) else str(Fraction.from_float(float(n))-z),
                                 "total_error_exact": None if not np.isfinite(n) else str(Fraction.from_float(float(n))-x)}
                                for x, y, z, n in zip(a, b, c, executed)])
            lanes["per_coordinate_errors"] = effects
            lanes["decision_effects"] = {
                "endpoint_quantization": [rule.preservation(a, b) for a, b in zip(
                    lanes["stored_exact_decisions"], lanes["projected_endpoint_exact_decisions"])],
                "input_quantization": [rule.preservation(a, b) for a, b in zip(
                    lanes["projected_endpoint_exact_decisions"], lanes["projected_input_exact_decisions"])],
                "intermediate_arithmetic": [rule.preservation(a, b) for a, b in zip(
                    lanes["projected_input_exact_decisions"], lanes["native_decisions"])],
                "stored_real_to_native": [rule.preservation(a, b) for a, b in zip(
                    lanes["stored_exact_decisions"], lanes["native_decisions"])],
            }
        observations[name] = lanes
    payload["observations"] = observations
    payload["native_decision_preserved"] = [rule.preservation(a, b) for a, b in zip(
        observations["parent"]["native_decisions"], observations["candidate"]["native_decisions"])]
    if exact_available:
        payload["stored_real_decision_preserved"] = [rule.preservation(a, b) for a, b in zip(
            observations["parent"]["stored_exact_decisions"], observations["candidate"]["stored_exact_decisions"])]
        paired = []
        for index, (old, new) in enumerate(zip(observations["parent"]["stored_exact_outputs"],
                                               observations["candidate"]["stored_exact_outputs"])):
            paired.append([{
                "stored_real_drift_exact": str(Fraction(b)-Fraction(a)),
                "native_drift_exact": None if not (np.isfinite(left) and np.isfinite(right)) else
                                      str(Fraction.from_float(float(right))-Fraction.from_float(float(left))),
                "parent_total_error_exact": observations["parent"]["per_coordinate_errors"][index][j]["total_error_exact"],
                "candidate_total_error_exact": observations["candidate"]["per_coordinate_errors"][index][j]["total_error_exact"],
            } for j, (a, b, left, right) in enumerate(zip(old, new,
                [float(x) if not isinstance(x, str) else float("nan") for x in observations["parent"]["native_outputs"][index]],
                [float(x) if not isinstance(x, str) else float("nan") for x in observations["candidate"]["native_outputs"][index]]))])
        payload["paired_coordinate_effects"] = paired
    if any(any(isinstance(x, str) for x in row) for lanes in observations.values() for row in lanes["native_outputs"]):
        return finish("observed_violation", "Native execution produced nonfinite outputs")
    if not exact_available:
        return finish("unsupported", "Native outputs retained; exact real-to-native error accounting is unsupported for this activation")
    decision_status = _result_status(payload["native_decision_preserved"])
    return finish(decision_status, None if decision_status == "pass" else "The declared native decision-preservation rule did not pass")


def verify_native_execution(base: WeightArtifact, candidate: WeightArtifact,
                            receipt: dict[str, Any]) -> bool:
    from .contracts import DecisionRule
    try:
        actual = native_execution_receipt(base, candidate, receipt["witness_inputs"],
            ExecutionSpec.from_dict(receipt["execution"]),
            decision=DecisionRule.from_dict(receipt["decision_rule"]),
            binding_sha256=receipt["binding_sha256"])
        return canonical_bytes(actual) == canonical_bytes(receipt)
    except (ValueError, TypeError, KeyError, OverflowError):
        return False
