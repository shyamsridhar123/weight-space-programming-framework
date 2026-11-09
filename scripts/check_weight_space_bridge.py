#!/usr/bin/env python3
"""Freshly check runtime-candidate Lean instances and archive replayable evidence.

No dependency compilation claim is inferred from a downloaded .olean cache.
Pinned dependency sources and the actual cached artifacts are hashed separately.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import AbstractContextManager
from fractions import Fraction
import hashlib
from importlib.util import resolve_name
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlcd.weight_space.artifact import TensorArtifact, StructuralPolicy
from rlcd.weight_space.contracts import DecisionRule, bind_admission
from rlcd.weight_space.features import ReadoutHead
from rlcd.weight_space.model import InputBox, Network, canonical_bytes, digest
from rlcd.weight_space.program import difference
from rlcd.weight_space.proof_bridge import decode_instance, generate_instance, validate_instance
from rlcd.weight_space.sequential import sequential_artifact
from rlcd.weight_space.workflow import TaskSpec

MODULES = ("Core", "Bounds", "Sequential", "Bridge")
IMPORT_HELPER = "ImportClosure"
INPUT_FIELDS = {"schema", "parent", "candidate", "domain", "head", "feature_names", "output_names",
                "max_output_drift", "binding"}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(data) + b"\n")


def checker_source_closure() -> list[Path]:
    """Include the real package initializers and transitive project imports."""
    pending = ["rlcd", "rlcd.weight_space", "rlcd.weight_space.proof_bridge"]
    seen: set[str] = set()
    paths: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        prefix = ROOT.joinpath(*module.split("."))
        path = prefix / "__init__.py" if (prefix / "__init__.py").is_file() else prefix.with_suffix(".py")
        if not path.is_file():
            raise ValueError("Missing project checker dependency: " + module)
        paths.add(path)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imports = []
            if isinstance(node, ast.ImportFrom):
                name = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
                if name and (name == "rlcd" or name.startswith("rlcd.")):
                    imports = [name] if node.module else [name + "." + alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names if alias.name.startswith("rlcd.")]
            pending.extend(imports)
    return sorted(paths)


def python_runtime_identity() -> dict[str, Any]:
    import numpy
    package = Path(numpy.__file__).resolve().parent
    # ponytail: installed source/binary hashes bind this execution without packaging a new environment.
    numpy_files = {"numpy/" + path.relative_to(package).as_posix(): file_hash(path)
                   for path in sorted(package.rglob("*"))
                   if path.is_file() and path.suffix.lower() in (".py", ".pyd", ".dll")}
    library_dir = package.parent / "numpy.libs"
    numpy_files.update({"numpy.libs/" + path.relative_to(library_dir).as_posix(): file_hash(path)
                        for path in sorted(library_dir.rglob("*.dll")) if path.is_file()})
    return {"version": sys.version, "numpy_version": numpy.__version__,
            "executable_sha256": file_hash(Path(sys.executable)),
            "numpy_installation_sha256": digest(numpy_files), "numpy_files_sha256": numpy_files,
            "platform": sys.platform, "byteorder": sys.byteorder,
            "tensor_dtype": "finite-binary64", "weight_order": "C/output-then-input",
            "source_package": "checker-sources/rlcd",
            "trust_boundary": "Interpreter executable and installed NumPy source/binaries are hashed; Python standard-library/native runtime correctness remains trusted"}


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None,
        timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


def resolve_lean(value: str | None) -> Path:
    if value:
        result = Path(value).resolve()
    else:
        candidates = [ROOT / ".cache/weight-space/lean-4.24.0-windows/bin/lean.exe",
                      ROOT.parents[1] / ".cache/weight-space/lean-4.24.0-windows/bin/lean.exe"]
        found = next((path for path in candidates if path.is_file()), None)
        executable = shutil.which("lean")
        if found is None and executable is None:
            raise ValueError("Supply --lean with the pinned Lean 4.24.0 executable")
        result = found or Path(executable).resolve()
    if not result.is_file():
        raise ValueError("Lean executable does not exist")
    return result


def resolve_cache(value: str | None) -> Path:
    if value:
        result = Path(value).resolve()
    else:
        candidates = [ROOT / "formal/weight-space/.lake/packages",
                      ROOT.parents[1] / "formal/weight-space/.lake/packages"]
        result = next((p for p in candidates if (p / "mathlib/.lake/build/lib/lean").is_dir()), None)
        if result is None:
            raise ValueError("Supply --dependency-cache with the pinned Lake packages directory")
    if not (result / "mathlib/.lake/build/lib/lean").is_dir():
        raise ValueError("Pinned Mathlib compiled dependency cache is missing")
    return result


def dependency_audit(lean: Path, cache: Path, framework: Path, *, timeout: float = 60) -> dict[str, Any]:
    """Hash the explicit source import closure and the artifacts Lean imports.

    Git blob identities validate only source bytes at the locked revision.
    Correspondence between those sources and cached .olean bytes is retained as
    an external assumption, because this runner does not rebuild all Mathlib.
    """
    lock = json.loads((ROOT / "formal/weight-space/lake-manifest.json").read_text())
    packages, source_trees = {}, {}
    for package in lock["packages"]:
        directory = cache / package["name"]
        head = run(["git", "rev-parse", "HEAD"], cwd=directory)
        if head.returncode or head.stdout.strip() != package["rev"]:
            raise ValueError("Dependency checkout differs from the lock: " + package["name"])
        tree = subprocess.run(["git", "ls-tree", "-r", "-z", "HEAD"], cwd=directory,
                              capture_output=True, timeout=30)
        if tree.returncode:
            raise ValueError("Could not read pinned dependency source tree")
        source_trees[package["name"]] = {}
        for record in tree.stdout.split(b"\0"):
            if not record:
                continue
            info, name = record.split(b"\t", 1)
            source_trees[package["name"]][name.decode("utf-8")] = info.split()[2].decode("ascii")
        packages[package["name"]] = {"url": package["url"], "revision": package["rev"],
                                     "checkout_head_matches_lock": True}
    distribution = lean.parent.parent
    roots = [(name, cache / name, cache / name / ".lake/build/lib/lean") for name in packages]
    roots.append(("lean-distribution", distribution / "src/lean", distribution / "lib/lean"))
    # ponytail: Lean's own parser handles comment/string examples and implicit Init.
    helper = framework / "WeightSpace" / (IMPORT_HELPER + ".lean")
    with tempfile.TemporaryDirectory(prefix="wsp-import-audit-") as temporary:
        config = Path(temporary) / "imports.json"
        write_json(config, {"roots": [str(framework)] + [str(source) for _, source, _ in roots],
                           "seeds": ["WeightSpace." + name for name in MODULES + (IMPORT_HELPER,)]
                                    + ["Lean.Util.CollectAxioms"]})
        parsed = run([str(lean), "--run", str(helper), str(config)], cwd=framework,
                     env=dict(os.environ, LEAN_PATH=""), timeout=timeout)
        if parsed.returncode:
            raise ValueError("Lean's native source-import parser failed: " + parsed.stdout + parsed.stderr)
        import_rows = json.loads(parsed.stdout)
    closure = []
    for parsed_row in import_rows:
        module = parsed_row["moduleName"]
        relative = Path(*module.split("."))
        if module.startswith("WeightSpace."):
            source = framework / relative.with_suffix(".lean")
            package_name, compiled = "framework", None
        else:
            match = next(((name, src / relative.with_suffix(".lean"), binary / relative.with_suffix(".olean"))
                          for name, src, binary in roots if (src / relative.with_suffix(".lean")).is_file()), None)
            if match is None:
                raise ValueError("Imported dependency source is missing: " + module)
            package_name, source, compiled = match
        if source.resolve() != Path(parsed_row["source"]).resolve():
            raise ValueError("Native parser selected a different source root: " + module)
        source_bytes = source.read_bytes()
        row = {"module": module, "package": package_name,
               "source_sha256": hashlib.sha256(source_bytes).hexdigest()}
        if package_name in packages:
            relative_source = source.relative_to(cache / package_name).as_posix()
            expected_blob = source_trees[package_name].get(relative_source)
            actual_blob = hashlib.sha1(b"blob " + str(len(source_bytes)).encode() + b"\0" + source_bytes).hexdigest()
            normalized = source_bytes.replace(b"\r\n", b"\n")
            normalized_blob = hashlib.sha1(b"blob " + str(len(normalized)).encode() + b"\0" + normalized).hexdigest()
            if actual_blob != expected_blob:
                if normalized_blob != expected_blob:
                    raise ValueError("Imported source differs from pinned Git blob beyond CRLF checkout conversion: " + module)
            row.update({"source_git_blob": expected_blob, "raw_source_matches_locked_git_blob": actual_blob == expected_blob,
                        "checkout_conversion": None if actual_blob == expected_blob else "CRLF-to-LF-only",
                        "source_matches_locked_revision_after_declared_conversion": True})
        else:
            row["raw_source_matches_locked_git_blob"] = None
        if compiled is not None:
            if not compiled.is_file():
                raise ValueError("Imported compiled artifact is missing: " + module)
            binaries = [compiled, Path(str(compiled) + ".private"), Path(str(compiled) + ".server"),
                        compiled.with_suffix(".ir")]
            row["cached_artifacts"] = {path.name: file_hash(path) for path in binaries if path.is_file()}
        row["source_header_imports"] = parsed_row["imports"]
        closure.append(row)
    payload = {"schema": "wsp-lean-dependency-audit/1", "packages": packages,
               "explicit_import_closure_plus_default_prelude": sorted(closure, key=lambda row: row["module"]),
               "source_header_parser": {"engine": "Pinned Lean Lean.parseImports'",
                                        "source_sha256": file_hash(helper)},
               "dependency_compilation_rechecked": False,
               "trust_boundary": "Locked imported source Git blobs are checked allowing only declared CRLF checkout conversion; exact working source and cached .olean/.ir bytes are hashed. Cached dependencies and Lean distribution remain trusted, not rebuilt or kernel-rechecked by this runner"}
    return payload | {"sha256": digest(payload)}


class LeanSession(AbstractContextManager):
    def __init__(self, output: Path, lean: Path, cache: Path, timeout: float = 60):
        self.output, self.lean, self.cache, self.timeout = output, lean, cache, timeout
        self.temporary = tempfile.TemporaryDirectory(prefix="wsp-lean-bridge-")
        self.build = Path(self.temporary.name)
        self.framework = output / "framework"
        self.framework.mkdir(parents=True)
        version = run([str(lean), "--version"], cwd=ROOT)
        if version.returncode or "version 4.24.0," not in version.stdout:
            raise ValueError("Bridge requires the repository's pinned Lean 4.24.0")
        self.compiler = {"version": version.stdout.strip(), "executable_sha256": file_hash(lean),
                         "distribution_dll_sha256": {path.name: file_hash(path)
                                                     for path in sorted(lean.parent.glob("*.dll"))},
                         "invocation_options": ["-o", "<fresh-build>/<module>.olean", "<archived-source>"],
                         "imported_modules_trust_level": "Lean default; cached imports trusted",
                         "project_toolchain_sha256": file_hash(ROOT / "formal/weight-space/lean-toolchain"),
                         "project_lock_sha256": file_hash(ROOT / "formal/weight-space/lake-manifest.json"),
                         "project_config_sha256": file_hash(ROOT / "formal/weight-space/lakefile.toml")}
        for name in MODULES + (IMPORT_HELPER,):
            target = self.framework / "WeightSpace" / (name + ".lean")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "formal/weight-space/WeightSpace" / (name + ".lean"), target)
        for name in ("lean-toolchain", "lake-manifest.json", "lakefile.toml"):
            shutil.copyfile(ROOT / "formal/weight-space" / name, self.framework / name)
        source_dir = output / "checker-sources"
        source_dir.mkdir()
        shutil.copyfile(ROOT / "rlcd/weight_space/proof_bridge.py", source_dir / "proof_bridge.py")
        shutil.copyfile(Path(__file__), source_dir / "check_weight_space_bridge.py")
        for path in checker_source_closure():
            target = source_dir / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        python_runtime = python_runtime_identity()
        write_json(output / "python-runtime.json", python_runtime)
        self.python_runtime = python_runtime
        started = time.perf_counter()
        audit = dependency_audit(lean, cache, self.framework, timeout=self.timeout)
        write_json(output / "dependencies.json", audit)
        print("Pinned dependency source/artifact audit completed", flush=True)
        self.audit_seconds = time.perf_counter() - started
        paths = [self.build] + [cache / name / ".lake/build/lib/lean" for name in audit["packages"]
                               if (cache / name / ".lake/build/lib/lean").is_dir()]
        self.env = dict(os.environ, LEAN_PATH=os.pathsep.join(map(str, paths)))
        self.framework_results = []
        for name in MODULES:
            self.framework_results.append(self.compile(self.framework / "WeightSpace" / (name + ".lean"),
                                                       "framework-" + name, "WeightSpace/" + name))
        write_json(output / "compiler.json", self.compiler)
        write_json(output / "framework-checks.json", self.framework_results)

    def compile(self, source: Path, log_name: str, module: str, *, expect_success: bool = True) -> dict[str, Any]:
        target = self.build / (module + ".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [str(self.lean), "-o", str(target), str(source)]
        started = time.perf_counter()
        try:
            result = run(command, cwd=source.parent if module == "Instance" else self.framework,
                         env=self.env, timeout=self.timeout)
            stdout = result.stdout.replace(str(self.build), "<fresh-build>").replace(str(self.output), "<bundle>")
            stdout = stdout.replace(str(self.cache), "<dependency-cache>")
            stderr = result.stderr.replace(str(self.build), "<fresh-build>").replace(str(self.output), "<bundle>")
            stderr = stderr.replace(str(self.cache), "<dependency-cache>")
            returncode, timed_out = result.returncode, False
        except subprocess.TimeoutExpired as error:
            stdout = (error.stdout or b"").decode("utf-8", errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr, returncode, timed_out = "Lean invocation exceeded the recorded timeout", None, True
        logs = self.output / "logs"
        logs.mkdir(exist_ok=True)
        (logs / (log_name + ".log")).write_text(stdout + stderr, encoding="utf-8")
        receipt = {"source_sha256": file_hash(source), "returncode": returncode, "timed_out": timed_out,
                   "wall_seconds": round(time.perf_counter() - started, 6), "timeout_seconds": self.timeout,
                   "olean_sha256": file_hash(target) if target.is_file() and returncode == 0 else None,
                   "axiom_audit_passed": "Runtime instance transitive axiom audit passed." in stdout,
                   "log": "logs/" + log_name + ".log", "lean_checked": returncode == 0}
        write_json(logs / (log_name + ".receipt.json"), receipt)
        print(log_name + ": " + ("Lean passed" if returncode == 0 else "Lean failed/timeout"), flush=True)
        if expect_success and returncode != 0:
            raise ValueError("Lean compilation failed; inspect " + receipt["log"])
        return receipt

    def __exit__(self, *args: Any) -> None:
        self.temporary.cleanup()


def restore_input(data: dict[str, Any]) -> tuple[Any, Any, InputBox, dict[str, Any]]:
    if not isinstance(data, dict) or set(data) != INPUT_FIELDS or data["schema"] != "wsp-lean-bridge-input/1":
        raise ValueError("Unknown strict bridge input schema")
    def artifact(raw: dict[str, Any]) -> Any:
        return Network.from_dict(raw) if raw.get("schema") == "wsp-network/1" else TensorArtifact.from_dict(raw)
    options = {key: data[key] for key in ("feature_names", "output_names", "max_output_drift", "binding")}
    options["head"] = None if data["head"] is None else ReadoutHead(**data["head"])
    return artifact(data["parent"]), artifact(data["candidate"]), InputBox(**data["domain"]), options


def input_data(base: Any, candidate: Any, box: InputBox, *, head: ReadoutHead | None = None,
               feature_names: tuple[str, ...] = (), output_names: tuple[str, ...] = (),
               max_output_drift: tuple[float, ...] | None = None, binding: Any = None) -> dict[str, Any]:
    return {"schema": "wsp-lean-bridge-input/1", "parent": base.to_dict(), "candidate": candidate.to_dict(),
            "domain": box.to_dict(), "head": None if head is None else head.to_dict(),
            "feature_names": list(feature_names), "output_names": list(output_names),
            "max_output_drift": None if max_output_drift is None else list(max_output_drift),
            "binding": None if binding is None else binding.to_dict() if hasattr(binding, "to_dict") else binding}


def check_candidate(session: LeanSession, name: str, data: dict[str, Any], *,
                    source: str | None = None) -> dict[str, Any]:
    base, candidate, box, options = restore_input(data)
    regenerated = generate_instance(base, candidate, box, **options)
    if regenerated != generate_instance(base, candidate, box, **options):
        raise ValueError("Repeated generation changed the concrete source")
    if source is not None and source != regenerated:
        raise ValueError("Archived candidate source differs from deterministic regeneration")
    source = regenerated if source is None else source
    folder = session.output / "candidates" / name
    folder.mkdir(parents=True)
    write_json(folder / "inputs.json", data)
    path = folder / "Instance.lean"
    path.write_bytes(source.encode("utf-8"))
    decoded = validate_instance(source, base, candidate, box, **options)
    write_json(folder / "independent-decoding.json", decoded)
    write_json(folder / "decoded-values.json", decode_instance(source))
    lean = session.compile(path, name, "Instance")
    if not lean["axiom_audit_passed"]:
        raise ValueError("Generated candidate did not complete its transitive axiom audit")
    payload = {"schema": "wsp-lean-checked-candidate/1", "name": name, "input_sha256": digest(data),
               "parent_sha256": base.sha256, "candidate_sha256": candidate.sha256,
               "binding_sha256": decoded["binding_sha256"], "source_sha256": file_hash(path),
               "generated_source_bytes": path.stat().st_size, "deterministic_regeneration_passed": True,
               "independent_decoding_sha256": decoded["sha256"], "lean": lean,
               "real_theorem_checked": True, "independent_decoding_passed": True,
               "scope": decoded["scope"], "output_bounds_exact": decoded["output_bounds_exact"],
               "output_budget_exact": decoded["output_budget_exact"],
               "native_execution_formally_verified": False, "feature_extractor_verified": False,
               "useful_business_behavior_verified": False}
    receipt = payload | {"sha256": digest(payload)}
    write_json(folder / "checked-candidate.json", receipt)
    return receipt


def fixtures() -> dict[str, dict[str, Any]]:
    inputs, outputs = ("feature-a", "feature-b"), ("accept", "review")
    base = TensorArtifact("frozen-linear-fixture/1", {
        "head.weight": [[1., -.5], [.25, 1.]], "head.bias": [.25, -.25]},
        {"input_names": list(inputs), "output_names": list(outputs), "feature_id": "fixture-features/1"})
    candidate = base.replace(**{"head.weight": [[1.125, -.5], [.25, 1.]], "head.bias": [.25, -.1875]})
    head, box = ReadoutHead("head.weight", "head.bias", "fixture-features/1"), InputBox([-1., -2.], [1., 2.])
    program = difference(base, candidate, "Change the declared frozen head")
    task = TaskSpec("lean-head-fixture", "edit", "Conditional head score drift", base.architecture, base.sha256, {})
    binding = bind_admission(base, candidate, program, task=task,
        policy=StructuralPolicy((head.weight, head.bias), 4, 4., 1.), box=box,
        max_output_drift=(1., 1.), decision=DecisionRule("scores_only"),
        witnesses=[[0., 0.], [1., 2.]], feature_retention={"head": head, "feature_names": inputs, "anchors": [[0., 0.]]})
    sequential = sequential_artifact([[[1., .5], [0., 1.]], [[1., -1.]]], [[0., .25], [0.]],
                                    ["relu", "abs"], input_names=inputs, output_names=("score",))
    changed = sequential.replace(**{"layers.0.weight": [[1.125, .5], [0., 1.]],
                                     "layers.1.weight": [[1., -.875]], "layers.1.bias": [.0625]})
    no_bias = TensorArtifact("frozen-bias-free-fixture/1", {"head.weight": [[1., -.5]]},
                            {"input_names": list(inputs), "output_names": ["score"]})
    return {
        "head-zero": input_data(base, base, box, head=head, feature_names=inputs, output_names=outputs),
        "head-nonzero-bound": input_data(base, candidate, box, head=head, feature_names=inputs,
                                        output_names=outputs, max_output_drift=(1., 1.), binding=binding),
        "head-bias-free": input_data(no_bias, no_bias.replace(**{"head.weight": [[1.125, -.5]]}), box,
                                     head=ReadoutHead("head.weight", feature_id="fixture-features/1"),
                                     feature_names=inputs, output_names=("score",)),
        "sequential-zero": input_data(sequential, sequential, box),
        "sequential-nonzero": input_data(sequential, changed, box),
    }


def mutation_checks(session: LeanSession, data: dict[str, Any]) -> list[dict[str, Any]]:
    base, candidate, box, options = restore_input(data)
    source = generate_instance(base, candidate, box, **options)
    folder = session.output / "mutations"
    folder.mkdir()
    rows = []
    def reject(name: str, altered: str = source, *, kwargs: dict[str, Any] | None = None,
               altered_box: InputBox = box) -> None:
        try:
            validate_instance(altered, base, candidate, altered_box, **(options if kwargs is None else kwargs))
        except ValueError as error:
            rows.append({"mutation": name, "independent_decoder_rejected": True, "reason": str(error),
                         "source_sha256": hashlib.sha256(altered.encode()).hexdigest()})
        else:
            raise ValueError("Mutation was incorrectly accepted: " + name)
    lines = source.splitlines(keepends=True)
    def replace_declaration(name: str, old: str, new: str) -> str:
        return "".join(line.replace(old, new, 1) if line.startswith("def " + name + " :") else line for line in lines)
    wrong_endpoint = replace_declaration("candidateWeights", "(9 / 8 : ℝ)", "(17 / 16 : ℝ)")
    if wrong_endpoint == source:
        raise ValueError("Wrong-endpoint fixture did not alter a theorem declaration")
    path = folder / "LeanValidWrongEndpoint.lean"
    path.write_bytes(wrong_endpoint.encode("utf-8"))
    lean = session.compile(path, "lean-valid-wrong-endpoint", "Instance")
    reject("lean-valid-wrong-endpoint", wrong_endpoint)
    rows[-1]["lean"] = lean
    rows[-1]["requirement"] = "A theorem about substituted endpoint values remains Lean-valid and must be rejected"
    reject("row-order", replace_declaration("candidateWeights", "(9 / 8 : ℝ), (-1 / 2 : ℝ)", "(-1 / 2 : ℝ), (9 / 8 : ℝ)"))
    reject("bias-value", replace_declaration("candidateBiases", "(-3 / 16 : ℝ)", "(-1 / 8 : ℝ)"))
    reject("shape", replace_declaration("widths", "[2, 2]", "[1, 4]"))
    reject("activation", replace_declaration("certifiedActivations", ".identity", ".tanh"))
    reject("domain-value", replace_declaration("domainLower", "(-1 / 1 : ℝ)", "(-2 / 1 : ℝ)"))
    reject("domain-substitution", altered_box=InputBox([-2., -2.], [1., 2.]))
    reject("feature-identity", kwargs=options | {"head": ReadoutHead("head.weight", "head.bias", "substituted-features/1")})
    reject("feature-order", kwargs=options | {"feature_names": list(reversed(options["feature_names"]))})
    reject("budget", replace_declaration("outputBudget", "(1 / 1 : ℝ)", "(2 / 1 : ℝ)"))
    reject("extra-axiom", source + "\naxiom unrelatedEscape : False\n")
    bad_binding = json.loads(json.dumps(options["binding"]))
    bad_binding["program"]["base_sha256"] = "0" * 64
    bad_binding["program_sha256"] = digest(bad_binding["program"])
    bad_binding["sha256"] = digest({k: v for k, v in bad_binding.items() if k != "sha256"})
    reject("self-consistent-stale-program-binding", kwargs=options | {"binding": bad_binding})
    bad_binding = json.loads(json.dumps(options["binding"]))
    bad_binding["task"]["objective"] = "Substituted task"
    bad_binding["task_sha256"] = digest(bad_binding["task"])
    bad_binding["sha256"] = digest({k: v for k, v in bad_binding.items() if k != "sha256"})
    reject("self-consistent-different-task-binding", kwargs=options | {"binding": bad_binding})
    unsupported = sequential_artifact([[[1.]]], [[0.]], ["tanh"])
    try:
        generate_instance(unsupported, unsupported, InputBox([-1.], [1.]))
    except ValueError as error:
        rows.append({"mutation": "unsupported-executable-activation", "generator_rejected": True, "reason": str(error)})
    else:
        raise ValueError("Tanh execution was incorrectly promoted to a Lean certificate")
    write_json(folder / "rejections.json", rows)
    return rows


def finalize(output: Path, candidates: list[dict[str, Any]], mutations: list[dict[str, Any]],
             *, replay_of: str | None = None) -> dict[str, Any]:
    files = {path.relative_to(output).as_posix(): file_hash(path)
             for path in sorted(output.rglob("*")) if path.is_file() and path != output / "manifest.json"}
    payload = {"schema": "wsp-lean-bridge-bundle/1", "candidates": candidates, "mutations": mutations,
               "files_sha256": files, "replay_of_manifest_sha256": replay_of,
               "all_candidate_real_theorems_checked": all(c["real_theorem_checked"] for c in candidates),
               "all_candidate_decodings_passed": all(c["independent_decoding_passed"] for c in candidates),
               "scope": "Pinned Lean checks of finite real affine/identity/ReLU/abs instances independently bound to captured endpoint/interface/domain/request bytes; excludes encoder, arbitrary operators, native kernels and business usefulness",
               "dependency_cache_compilation_rechecked": False,
               "reproduction_tiers": {"snapshot_integrity": "Bundle file hashes checked against an external expected manifest identity",
                                      "proof_replay": "Fresh framework and candidate Lean applications with pinned dependency source identities and independently decoded original inputs",
                                      "native_backend": "Not provided by this real-arithmetic bridge",
                                      "business_outcome": "Not provided by fixture proofs"}}
    manifest = payload | {"sha256": digest(payload)}
    write_json(output / "manifest.json", manifest)
    return manifest


def check_manifest(path: Path, expected: str) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["sha256"] != expected or digest({k: v for k, v in manifest.items() if k != "sha256"}) != expected:
        raise ValueError("Bundle manifest differs from its external expected identity")
    present = {file.relative_to(path).as_posix() for file in path.rglob("*")
               if file.is_file() and file != path / "manifest.json"}
    if present != set(manifest["files_sha256"]):
        raise ValueError("Bundle contains missing or unlisted files")
    for name, sha256 in manifest["files_sha256"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file() or file_hash(target) != sha256:
            raise ValueError("Bundle file bytes differ: " + name)
        if name.startswith("checker-sources/rlcd/"):
            local = ROOT / name.removeprefix("checker-sources/")
            if not local.is_file() or file_hash(local) != sha256:
                raise ValueError("Current runtime source package differs from the archived checker: " + name)
    if file_hash(ROOT / "rlcd/weight_space/proof_bridge.py") != manifest["files_sha256"]["checker-sources/proof_bridge.py"]:
        raise ValueError("Current generator/decoder differs from the archived checker")
    if file_hash(Path(__file__)) != manifest["files_sha256"]["checker-sources/check_weight_space_bridge.py"]:
        raise ValueError("Current verification runner differs from the archived runner")
    for module in MODULES + (IMPORT_HELPER,):
        key = "framework/WeightSpace/" + module + ".lean"
        if file_hash(ROOT / "formal/weight-space/WeightSpace" / (module + ".lean")) != manifest["files_sha256"][key]:
            raise ValueError("Current framework source differs from the archived instance premise: " + module)
    for name in ("lean-toolchain", "lake-manifest.json", "lakefile.toml"):
        if file_hash(ROOT / "formal/weight-space" / name) != manifest["files_sha256"]["framework/" + name]:
            raise ValueError("Current toolchain/dependency/config declaration differs: " + name)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Strict original candidate inputs JSON; omit for the complete fixture suite")
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory")
    parser.add_argument("--lean")
    parser.add_argument("--dependency-cache")
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.timeout_seconds > 600:
        raise ValueError("Use a positive recorded per-file timeout of at most 600 seconds")
    if args.replay and not args.expected_manifest_sha256:
        raise ValueError("Replay requires the trusted original manifest SHA256")
    if args.replay and args.input:
        raise ValueError("Choose candidate input or an existing replay bundle")
    original = check_manifest(args.replay, args.expected_manifest_sha256) if args.replay else None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        with LeanSession(output, resolve_lean(args.lean), resolve_cache(args.dependency_cache), args.timeout_seconds) as session:
            candidates, mutations = [], []
            if original is not None:
                old_compiler = json.loads((args.replay / "compiler.json").read_text(encoding="utf-8"))
                if session.compiler != old_compiler:
                    raise ValueError("Strict replay compiler, distribution or configuration identity differs")
                old_python = json.loads((args.replay / "python-runtime.json").read_text(encoding="utf-8"))
                if session.python_runtime != old_python:
                    raise ValueError("Strict replay Python interpreter or NumPy source/binary identity differs")
                old_dependencies = json.loads((args.replay / "dependencies.json").read_text())
                current_dependencies = json.loads((output / "dependencies.json").read_text())
                if old_dependencies != current_dependencies:
                    raise ValueError("Strict replay dependency source or compiled artifact identity differs")
                for old in original["candidates"]:
                    folder = args.replay / "candidates" / old["name"]
                    data = json.loads((folder / "inputs.json").read_text(encoding="utf-8"))
                    source = (folder / "Instance.lean").read_text(encoding="utf-8")
                    candidates.append(check_candidate(session, old["name"], data, source=source))
                if original["mutations"]:
                    bound = next(c for c in original["candidates"] if c["name"] == "head-nonzero-bound")
                    data = json.loads((args.replay / "candidates" / bound["name"] / "inputs.json").read_text())
                    mutations = mutation_checks(session, data)
            elif args.input:
                data = json.loads(args.input.read_text(encoding="utf-8"))
                candidates.append(check_candidate(session, "candidate", data))
            else:
                suite = fixtures()
                candidates = [check_candidate(session, name, data) for name, data in suite.items()]
                mutations = mutation_checks(session, suite["head-nonzero-bound"])
            manifest = finalize(output, candidates, mutations, replay_of=None if original is None else original["sha256"])
        print(json.dumps({"status": "pass", "manifest_sha256": manifest["sha256"], "output": str(output)}, indent=2))
        return 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        write_json(output / "failure.json", {"status": "failed", "reason": str(error)})
        print("Bridge check failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
