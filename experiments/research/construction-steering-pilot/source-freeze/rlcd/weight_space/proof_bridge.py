"""Generate a small, closed Lean instance and decode its actual value declarations.

This bridge proves real arithmetic for captured finite affine chains. The decoder
is a separate data path: it reads the declarations used by the theorem and walks
the original nested runtime rows, rather than trusting the emitter's flattening.
Neither a Python serializer nor a frozen feature extractor is formally verified.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import re
from typing import Any, Sequence

from .artifact import TensorArtifact, WeightArtifact
from .features import ReadoutHead, _endpoint_delta
from .model import InputBox, Network, _labels, canonical_bytes, digest, finite_binary64
from .sequential import CERTIFIED_ACTIVATIONS, validate_sequential


BRIDGE_SCHEMA = "wsp-lean-runtime-instance/1"
GENERATOR_VERSION = "1"
_ACTIVATION_LEAN = {"identity": "identity", "relu": "relu", "abs": "absolute"}
_DECLARATIONS = ("descriptorHex", "widths", "depth", "certifiedActivations",
                 "parentWeights", "parentBiases", "candidateWeights", "candidateBiases",
                 "domainLower", "domainUpper", "outputBudget")
_HEADER = """import WeightSpace.Bridge
import Lean.Util.CollectAxioms

noncomputable section
open scoped BigOperators
namespace WeightSpace.RuntimeInstance
open WeightSpace WeightSpace.Sequential WeightSpace.Bridge
set_option maxRecDepth 200000
set_option maxHeartbeats 2000000

"""


def _runtime_layers(base: WeightArtifact, candidate: WeightArtifact,
                    head: ReadoutHead | None, feature_names: Sequence[str],
                    output_names: Sequence[str]) -> dict[str, Any]:
    if type(base) not in (TensorArtifact, Network) or type(candidate) is not type(base):
        raise ValueError("Bridge endpoints must be immutable Network or TensorArtifact snapshots")
    if not base.compatible(candidate):
        raise ValueError("Bridge endpoints differ in architecture, shapes or semantic metadata")
    before, after = base.tensors, candidate.tensors
    if head is not None:
        if not isinstance(head, ReadoutHead) or not feature_names or not output_names:
            raise ValueError("A frozen head needs its identity and explicit ordered feature/output names")
        width, outputs = head.dimensions(base)
        _endpoint_delta(base, candidate, head)  # Reject any edit outside the frozen head.
        inputs = list(_labels(feature_names, width, "h"))
        names = list(_labels(output_names, outputs, "y"))
        layers = [{"weight": head.weight, "bias": head.bias, "activation": "identity"}]
        scope = "frozen-affine-head"
    elif type(base) is Network:
        inputs, names = list(base.input_names), list(base.output_names)
        layers = [{"weight": "w1", "bias": "b1", "activation": "relu"},
                  {"weight": "w2", "bias": "b2", "activation": "identity"}]
        scope = "dense-relu-dense"
    else:
        from .semantics import RELU_ARCHITECTURE, as_relu_network
        if base.architecture == RELU_ARCHITECTURE:
            model = as_relu_network(base)
            inputs, names = list(model.input_names), list(model.output_names)
            layers = [{"weight": "w1", "bias": "b1", "activation": "relu"},
                      {"weight": "w2", "bias": "b2", "activation": "identity"}]
            scope = "dense-relu-dense"
        else:
            layers = validate_sequential(base)
            validate_sequential(candidate)
            inputs, names = base.metadata["input_names"], base.metadata["output_names"]
            scope = "sequential-affine-activation"
    if feature_names and list(feature_names) != inputs:
        raise ValueError("Feature/input order differs from the interpreted interface")
    if output_names and list(output_names) != names:
        raise ValueError("Output order differs from the interpreted interface")
    if any(layer["activation"] not in CERTIFIED_ACTIVATIONS for layer in layers):
        raise ValueError("The Lean bridge supports only identity, ReLU and absolute value")
    widths = [int(before[layers[0]["weight"]].shape[1])]
    for layer in layers:
        width, inputs_count = map(int, before[layer["weight"]].shape)
        if inputs_count != widths[-1]:
            raise ValueError("Layer widths do not compose")
        widths.append(width)
    return {"scope": scope, "layers": layers, "widths": widths,
            "input_names": inputs, "output_names": names, "before": before, "after": after}


def _lift_emitted(runtime: dict[str, Any]) -> dict[str, Any]:
    """Emitter path: NumPy C-order flattening plus Fraction.from_float."""
    result: dict[str, Any] = {}
    for label, tensors in (("parent", runtime["before"]), ("candidate", runtime["after"])):
        result[label + "Weights"] = [
            [Fraction.from_float(float(v)) for v in tensors[layer["weight"]].ravel(order="C")]
            for layer in runtime["layers"]]
        result[label + "Biases"] = [
            [Fraction(0)] * runtime["widths"][i + 1] if layer["bias"] is None else
            [Fraction.from_float(float(v)) for v in tensors[layer["bias"]]]
            for i, layer in enumerate(runtime["layers"])]
    return result


def _lift_original(runtime: dict[str, Any]) -> dict[str, Any]:
    """Validator path: explicit output/input coordinates and integer ratios.

    No ravel/flatten operation or emitter-lift helper is used here. The row-major
    correspondence is checked against each original row and column separately.
    """
    result: dict[str, Any] = {}
    for label, tensors in (("parent", runtime["before"]), ("candidate", runtime["after"])):
        weights, biases = [], []
        for i, layer in enumerate(runtime["layers"]):
            matrix = tensors[layer["weight"]]
            weights.append([Fraction(*float(matrix[o, f]).as_integer_ratio())
                            for o in range(runtime["widths"][i + 1])
                            for f in range(runtime["widths"][i])])
            biases.append([Fraction(0) if layer["bias"] is None else
                           Fraction(*float(tensors[layer["bias"]][o]).as_integer_ratio())
                           for o in range(runtime["widths"][i + 1])])
        result[label + "Weights"], result[label + "Biases"] = weights, biases
    return result


def _drift(widths: list[int], data: dict[str, Any], lower: list[Fraction],
           upper: list[Fraction]) -> list[Fraction]:
    radius = [max(abs(lo), abs(hi)) for lo, hi in zip(lower, upper)]
    drift = [Fraction(0)] * widths[0]
    for n, outputs in enumerate(widths[1:]):
        pw, qw = data["parentWeights"][n], data["candidateWeights"][n]
        pb, qb = data["parentBiases"][n], data["candidateBiases"][n]
        next_radius, next_drift = [], []
        for o in range(outputs):
            start = o * widths[n]
            next_radius.append(abs(pb[o]) + sum((abs(pw[start + f]) * radius[f]
                                                for f in range(widths[n])), Fraction(0)))
            next_drift.append(abs(qb[o] - pb[o]) + sum((
                abs(qw[start + f] - pw[start + f]) * radius[f] + abs(qw[start + f]) * drift[f]
                for f in range(widths[n])), Fraction(0)))
        radius, drift = next_radius, next_drift
    return drift


def _binding(value: Any, base: WeightArtifact, candidate: WeightArtifact, box: InputBox,
             budget: list[Fraction], runtime: dict[str, Any], head: ReadoutHead | None) -> dict[str, Any] | None:
    if value is None:
        return None
    from .artifact import StructuralPolicy
    from .contracts import AdmissionRequest, _semantic_artifact, _task
    from .program import Program, apply_program
    from .features import verify_feature_retention
    data = value.to_dict() if isinstance(value, AdmissionRequest) else value
    data = AdmissionRequest.from_dict(data).to_dict()  # Schema and complete request digest.
    for name, original in (("parent", base), ("candidate", candidate)):
        if data[name + "_sha256"] != original.sha256 or canonical_bytes(data[name]) != canonical_bytes(original.to_dict()):
            raise ValueError("Admission binding substitutes the " + name + " artifact")
    program, task = Program.from_dict(data["program"]), _task(data["task"])
    if (program.sha256 != data["program_sha256"] or task.sha256 != data["task_sha256"]
            or task.base_sha256 != base.sha256 or task.architecture != _semantic_artifact(base).architecture
            or task.mode != "edit" or digest(data["witness_inputs"]) != data["witness_sha256"]):
        raise ValueError("Admission program, task or witness identity differs")
    applied = apply_program(base, program)
    if applied.sha256 != candidate.sha256 or canonical_bytes(applied.to_dict()) != canonical_bytes(candidate.to_dict()):
        raise ValueError("Admission program does not produce the proved candidate")
    if not StructuralPolicy.from_dict(data["structural_policy"]).check(base, program)["structural_accepted"]:
        raise ValueError("Admission candidate violates its captured structural policy")
    if canonical_bytes(data["domain"]) != canonical_bytes(box.to_dict()):
        raise ValueError("Admission domain differs; a raw-input box is not an inferred feature box")
    if not bool(box.contains(data["witness_inputs"]).all()):
        raise ValueError("An admission witness lies outside the proved domain")
    if [Fraction(*float(finite_binary64(v)).as_integer_ratio()) for v in data["max_output_drift"]] != budget:
        raise ValueError("Admission output budget differs from the proved budget")
    interface = data["interface"]
    if (interface["input_names"] != runtime["input_names"] or interface["output_names"] != runtime["output_names"]
            or interface["tensor_coordinate_order"] != "C"):
        raise ValueError("Admission input/output or tensor order differs")
    if head is None and interface["bias_convention"] != "explicit-per-layer-vector-after-dot/1":
        raise ValueError("Sequential binding does not declare its explicit bias convention")
    if head is not None:
        if interface["bias_convention"] not in ("explicit-per-layer-vector-after-dot/1", "caller-declared/1"):
            raise ValueError("Unknown declared head bias convention")
        feature = data["feature_retention"]
        if feature is not None:
            if feature["head"] != head.to_dict() or feature["feature_names"] != runtime["input_names"]:
                raise ValueError("Admission frozen-feature identity or order differs")
            if feature["receipt"] is None or not verify_feature_retention(base, candidate, head, feature["anchors"], feature["receipt"]):
                raise ValueError("Admission feature receipt does not independently replay")
    return data


def _descriptor(base: WeightArtifact, candidate: WeightArtifact, box: InputBox,
                budget: list[Fraction], runtime: dict[str, Any], head: ReadoutHead | None,
                binding: Any) -> dict[str, Any]:
    return {"schema": BRIDGE_SCHEMA, "generator_version": GENERATOR_VERSION,
            "scope": runtime["scope"], "parent": base.to_dict(), "candidate": candidate.to_dict(),
            "parent_sha256": base.sha256, "candidate_sha256": candidate.sha256,
            "layers": runtime["layers"], "head": None if head is None else head.to_dict(),
            "interface": {"input_names": runtime["input_names"], "output_names": runtime["output_names"],
                          "weight_order": "row-major-output-then-input/1", "bias_order": "explicit-after-dot-vector/1",
                          "absent_bias_interpretation": "zero-vector-with-absence-retained-in-head"},
            "domain": box.to_dict(), "output_budget_exact": list(map(str, budget)),
            "binding": _binding(binding, base, candidate, box, budget, runtime, head)}


def _lean_values(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_lean_values(item) for item in value) + "]"
    if isinstance(value, Fraction):
        return f"({value.numerator} / {value.denominator} : ℝ)"
    raise ValueError("Only exact rational lists may be emitted")


def _footer(input_width: int, outputs: int) -> str:
    return f"""

theorem layout_valid : LayoutValid widths depth certifiedActivations parentWeights parentBiases
    candidateWeights candidateBiases domainLower domainUpper outputBudget := by
  unfold LayoutValid
  decide

theorem domain_valid : (inputBox widths domainLower domainUpper).Valid := by
  change ∀ i : Fin {input_width}, _
  norm_num [inputBox, widthAt, widths, domainLower, domainUpper, Fin.forall_fin_succ]

theorem numeric_budget : ∀ o : Coord widths depth,
    driftChain (D := Coord widths) (layers widths parentWeights parentBiases)
      (layers widths candidateWeights candidateBiases)
      (inputRadius (inputBox widths domainLower domainUpper)) (fun _ => 0) depth o ≤
      outputBudget.getD o.val 0 := by
  change ∀ o : Fin {outputs}, _
  dsimp only [Coord, widthAt, widths, depth, driftChain, driftStep, radiusChain,
    radiusStep, layers, inputRadius, inputBox, parentWeights, parentBiases,
    candidateWeights, candidateBiases, domainLower, domainUpper, outputBudget]
  norm_num [Fin.sum_univ_succ, Fin.forall_fin_succ]

theorem certified_output_drift (x : Coord widths 0 → ℝ)
    (hx : InBox (inputBox widths domainLower domainUpper) x) (o : Coord widths depth) :
    |evalChain (D := Coord widths) (layers widths candidateWeights candidateBiases)
       (activationsAt widths certifiedActivations) x depth o -
     evalChain (D := Coord widths) (layers widths parentWeights parentBiases)
       (activationsAt widths certifiedActivations) x depth o| ≤ outputBudget.getD o.val 0 := by
  exact (box_drift widths parentWeights parentBiases candidateWeights candidateBiases
    certifiedActivations domainLower domainUpper x hx depth o).trans (numeric_budget o)

theorem affine_endpoint_identity (n : Fin depth) (x : Coord widths n.val → ℝ)
    (o : Coord widths (n.val + 1)) :
    affineValue (layers widths candidateWeights candidateBiases n.val) x o -
      affineValue (layers widths parentWeights parentBiases n.val) x o =
      ((layers widths candidateWeights candidateBiases n.val).bias o -
        (layers widths parentWeights parentBiases n.val).bias o) +
      ∑ i, ((layers widths candidateWeights candidateBiases n.val).weight o i -
        (layers widths parentWeights parentBiases n.val).weight o i) * x i := by
  exact affine_endpoint_exact _ _ x o

#print axioms certified_output_drift
#print axioms affine_endpoint_identity

run_cmd do
  let allowed : Array Lean.Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let env ← Lean.getEnv
  let checked : Array Lean.Name := env.constants.fold (fun names name _ =>
    if (`WeightSpace.Bridge).isPrefixOf name || (`WeightSpace.RuntimeInstance).isPrefixOf name
    then names.push name else names) #[]
  for decl in checked do
    for axiomName in (← Lean.collectAxioms decl) do
      unless allowed.contains axiomName do
        throwError "{{decl}} depends on disallowed axiom {{axiomName}}"
  Lean.logInfo m!"Runtime instance transitive axiom audit passed."

end WeightSpace.RuntimeInstance
"""


def _render(descriptor: dict[str, Any], widths: list[int], activations: list[str],
            data: dict[str, Any]) -> str:
    encoded = canonical_bytes(descriptor).hex()
    declarations = [f'def descriptorHex : String := "{encoded}"',
                    "def widths : List ℕ := " + json.dumps(widths),
                    f"def depth : ℕ := {len(widths) - 1}",
                    "def certifiedActivations : List Activation := [" +
                    ", ".join("." + _ACTIVATION_LEAN[a] for a in activations) + "]"]
    declarations += [f"def {name} : " + ("List (List ℝ)" if name.endswith(("Weights", "Biases")) else "List ℝ") +
                     " := " + _lean_values(data[name]) for name in _DECLARATIONS[4:]]
    return _HEADER + "\n".join(declarations) + _footer(widths[0], widths[-1])


def generate_instance(base: WeightArtifact, candidate: WeightArtifact, box: InputBox, *,
                      head: ReadoutHead | None = None, feature_names: Sequence[str] = (),
                      output_names: Sequence[str] = (), max_output_drift: Sequence[float] | None = None,
                      binding: Any = None) -> str:
    """Emit a closed real-arithmetic instance; this function does not run Lean."""
    if not isinstance(box, InputBox):
        raise ValueError("An explicit finite real input/feature box is required")
    runtime = _runtime_layers(base, candidate, head, feature_names, output_names)
    if len(box.lower) != runtime["widths"][0]:
        raise ValueError("Input/feature box width differs from the interpreted model")
    data = _lift_emitted(runtime)
    data["domainLower"] = [Fraction.from_float(float(v)) for v in box.lower]
    data["domainUpper"] = [Fraction.from_float(float(v)) for v in box.upper]
    bound = _drift(runtime["widths"], data, data["domainLower"], data["domainUpper"])
    budget = bound if max_output_drift is None else [Fraction.from_float(finite_binary64(v)) for v in max_output_drift]
    if len(budget) != runtime["widths"][-1] or any(v < 0 for v in budget):
        raise ValueError("A finite nonnegative budget is required for every output")
    if any(actual > allowed for actual, allowed in zip(bound, budget)):
        raise ValueError("Conservative real drift recurrence exceeds the requested output budget")
    data["outputBudget"] = budget
    descriptor = _descriptor(base, candidate, box, budget, runtime, head, binding)
    return _render(descriptor, runtime["widths"], [a["activation"] for a in runtime["layers"]], data)


_REAL_TOKEN = re.compile(r"\((-?(?:0|[1-9][0-9]*)) / ([1-9][0-9]*) : ℝ\)")


def _decode_values(text: str) -> Any:
    """Strict list/rational parser; no eval and no Lean execution."""
    index = 0
    def item(depth: int = 0) -> Any:
        nonlocal index
        if index < len(text) and text[index] == "[":
            if depth >= 2:
                raise ValueError("Lean value lists exceed the supported tensor rank")
            index += 1
            result = []
            if index < len(text) and text[index] == "]":
                index += 1
                return result
            while True:
                result.append(item(depth + 1))
                if index < len(text) and text[index] == "]":
                    index += 1
                    return result
                if text[index:index + 2] != ", ":
                    raise ValueError("Malformed Lean list separator")
                index += 2
        match = _REAL_TOKEN.match(text, index)
        if match is None:
            raise ValueError("Unsupported or noncanonical Lean real literal")
        index = match.end()
        numerator, denominator = map(int, match.groups())
        result = Fraction(numerator, denominator)
        if result.numerator != numerator or result.denominator != denominator or match.group(1) == "-0":
            raise ValueError("Lean rational literal must be reduced and canonical")
        return result
    result = item()
    if index != len(text) or not isinstance(result, list):
        raise ValueError("Trailing or non-list Lean value source")
    return result


def decode_instance(source: str) -> dict[str, Any]:
    """Decode the actual theorem inputs and reject any unsupported proof scaffold.

    A successful return says the source matches the reviewed grammar, not that it
    typechecks or identifies a particular runtime artifact. validate_instance and
    an actual Lean check are both required for a checked bridge bundle.
    """
    if not isinstance(source, str) or not source.startswith(_HEADER):
        raise ValueError("Unsupported Lean bridge header")
    lines = source[len(_HEADER):].splitlines()
    if len(lines) < len(_DECLARATIONS):
        raise ValueError("Missing concrete Lean declarations")
    raw = {}
    for name, line in zip(_DECLARATIONS, lines):
        match = re.fullmatch(r"def " + name + r" : (.+) := (.+)", line)
        if match is None:
            raise ValueError("Missing, reordered or replaced Lean value declaration: " + name)
        raw[name] = match.groups()
    if raw["descriptorHex"][0] != "String" or not re.fullmatch(r'"[0-9a-f]+"', raw["descriptorHex"][1]):
        raise ValueError("Descriptor must be a canonical hexadecimal JSON string")
    try:
        descriptor_bytes = bytes.fromhex(raw["descriptorHex"][1][1:-1])
        descriptor = json.loads(descriptor_bytes)
        if canonical_bytes(descriptor) != descriptor_bytes or descriptor["schema"] != BRIDGE_SCHEMA:
            raise ValueError("Noncanonical or unknown instance descriptor")
        widths = json.loads(raw["widths"][1])
        if (raw["widths"][0] != "List ℕ" or not isinstance(widths, list) or len(widths) < 2
                or any(type(w) is not int or w < 1 for w in widths)):
            raise ValueError("Nonpositive or malformed declared widths")
        if raw["depth"] != ("ℕ", str(len(widths) - 1)):
            raise ValueError("Declared depth does not match widths")
        activation_source = raw["certifiedActivations"]
        if activation_source[0] != "List Activation" or not re.fullmatch(r"\[\.(identity|relu|absolute)(, \.(identity|relu|absolute))*\]", activation_source[1]):
            raise ValueError("Unsupported or malformed certified activation list")
        inverse = {v: k for k, v in _ACTIVATION_LEAN.items()}
        activations = [inverse[a[1:]] for a in activation_source[1][1:-1].split(", ")]
        if len(activations) != len(widths) - 1:
            raise ValueError("Activation depth differs")
        data = {}
        for name in _DECLARATIONS[4:]:
            expected_type = "List (List ℝ)" if name.endswith(("Weights", "Biases")) else "List ℝ"
            if raw[name][0] != expected_type:
                raise ValueError("Changed Lean value type: " + name)
            data[name] = _decode_values(raw[name][1])
        for name in _DECLARATIONS[4:8]:
            if len(data[name]) != len(widths) - 1:
                raise ValueError("Wrong declared layer count: " + name)
            for n, values in enumerate(data[name]):
                expected = widths[n + 1] * widths[n] if name.endswith("Weights") else widths[n + 1]
                if not isinstance(values, list) or len(values) != expected or not all(isinstance(v, Fraction) for v in values):
                    raise ValueError("Wrong concrete weight/bias shape: " + name)
        for name, width in (("domainLower", widths[0]), ("domainUpper", widths[0]), ("outputBudget", widths[-1])):
            if len(data[name]) != width or not all(isinstance(v, Fraction) for v in data[name]):
                raise ValueError("Wrong domain/budget shape: " + name)
        if any(lo > hi for lo, hi in zip(data["domainLower"], data["domainUpper"])) or any(v < 0 for v in data["outputBudget"]):
            raise ValueError("Invalid declared domain or negative output budget")
        if source != _render(descriptor, widths, activations, data):
            raise ValueError("Unsupported proof scaffold, extra declarations or noncanonical source")
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError, IndexError) as error:
        raise ValueError("Malformed concrete Lean runtime instance") from error
    def serial(value: Any) -> Any:
        return [serial(v) for v in value] if isinstance(value, list) else str(value)
    return {"schema": "wsp-lean-decoded-instance/1", "descriptor": descriptor,
            "widths": widths, "depth": len(widths) - 1, "activations": activations,
            "values": {name: serial(values) for name, values in data.items()},
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}


def validate_instance(source: str, base: WeightArtifact, candidate: WeightArtifact, box: InputBox, *,
                      head: ReadoutHead | None = None, feature_names: Sequence[str] = (),
                      output_names: Sequence[str] = (), max_output_drift: Sequence[float] | None = None,
                      binding: Any = None) -> dict[str, Any]:
    """Independently bind decoded theorem values to the original immutable inputs.

    Raises ValueError on any substitution. The returned receipt deliberately says
    lean_checked=False; only the verification runner can archive a successful
    actual Lean invocation and produce a checked bundle.
    """
    decoded = decode_instance(source)
    runtime = _runtime_layers(base, candidate, head, feature_names, output_names)
    original = _lift_original(runtime)
    original["domainLower"] = [Fraction(*float(v).as_integer_ratio()) for v in box.lower]
    original["domainUpper"] = [Fraction(*float(v).as_integer_ratio()) for v in box.upper]
    if len(original["domainLower"]) != runtime["widths"][0]:
        raise ValueError("Original input/feature domain width differs")
    bound = _drift(runtime["widths"], original, original["domainLower"], original["domainUpper"])
    budget = bound if max_output_drift is None else [Fraction(*float(finite_binary64(v)).as_integer_ratio()) for v in max_output_drift]
    if len(budget) != runtime["widths"][-1] or any(v < 0 for v in budget) or any(x > y for x, y in zip(bound, budget)):
        raise ValueError("Original output budget does not cover the conservative real recurrence")
    original["outputBudget"] = budget
    if decoded["widths"] != runtime["widths"] or decoded["activations"] != [a["activation"] for a in runtime["layers"]]:
        raise ValueError("Decoded shape or activation differs from original runtime semantics")
    for name, expected in original.items():
        if name.endswith(("Weights", "Biases")):
            actual = [[Fraction(v) for v in layer] for layer in decoded["values"][name]]
        else:
            actual = [Fraction(v) for v in decoded["values"][name]]
        if actual != expected:
            raise ValueError("Decoded theorem values differ from original runtime data: " + name)
    descriptor = _descriptor(base, candidate, box, budget, runtime, head, binding)
    if canonical_bytes(decoded["descriptor"]) != canonical_bytes(descriptor):
        raise ValueError("Descriptor artifact, metadata, feature identity, domain or full admission binding differs")
    payload = {"schema": "wsp-lean-independent-decoding/1", "source_sha256": decoded["source_sha256"],
               "parent_sha256": base.sha256, "candidate_sha256": candidate.sha256,
               "binding_sha256": None if descriptor["binding"] is None else descriptor["binding"]["sha256"],
               "decoded_sha256": digest(decoded), "descriptor_sha256": digest(descriptor),
               "output_bounds_exact": list(map(str, bound)), "output_budget_exact": list(map(str, budget)),
               "scope": runtime["scope"], "independent_decoding_passed": True, "lean_checked": False,
               "feature_extractor_verified": False, "floating_execution_verified": False,
               "full_model_semantics_verified": False, "useful_business_behavior_verified": False,
               "trust_boundary": "Exact binary64 endpoints lifted to real rationals; decoder/Python, external extractor and native backend remain outside the Lean theorem"}
    return payload | {"sha256": digest(payload)}
