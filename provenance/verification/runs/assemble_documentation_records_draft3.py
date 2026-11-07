"""Bind the reviewed draft.3 explanations to their actual inputs and check records.

This writes only the two new documentation records. It does not rerun or relabel
historical scientific/proof work, publish a release, or infer final tree/CI results.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = "93e8a9b7f2cfbbffab6c7d10f831b2cde4b69176"
TAG = "v0.2.0-draft.3"
VERSION = "0.2.0.dev3"
STE_COMMIT = "a7da74e83e5fcf8ea23dce075e64b463e5e24584"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def binding(relative: str, *, published_path: str | None = None) -> dict:
    path = (ROOT / relative).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Input escapes repository: {relative}")
    data = path.read_bytes()
    return {
        "path": published_path or relative,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def receipt(relative: str) -> dict:
    if not relative.startswith(".runs/"):
        raise ValueError("Only explicit local .runs evidence can be archive-mapped")
    return binding(relative, published_path="provenance/verification/runs/" + relative[6:])


def write_new(relative: str, value: dict) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    outputs = [
        "provenance/documentation-revision.json",
        "provenance/verification/runs/documentation-draft3.json",
    ]
    if any((ROOT / path).exists() for path in outputs):
        raise FileExistsError("Documentation records already exist; do not overwrite a completed assembly")

    guide_paths = [
        "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "EVIDENCE.md",
        "PROOFS.md", "REPRODUCING.md", "ROADMAP.md", "VERIFICATION.md",
        "releases/v0.2.0-draft.3.md",
        "docs/weight-space/v2/README.md", "docs/weight-space/v2/worked-example.md",
        "docs/weight-space/v2/architecture.md", "docs/weight-space/v2/workflow.md",
        "docs/weight-space/v2/requirements.md", "docs/weight-space/v2/source-review.md",
        "docs/weight-space/v2/methods.md", "docs/weight-space/v2/results.md",
        "docs/weight-space/v2/formal.md",
        "docs/weight-space/v2/limitations-and-next-steps.md",
    ]
    navigation_paths = [
        "docs/PROJECT_MEMORY.md", "docs/weight-space/README.md",
        "docs/weight-space/design.md", "docs/weight-space/requirements.md",
        "docs/weight-space/research.md", "docs/weight-space/results.md",
        "formal/weight-space/README.md",
    ]
    example_paths = ["examples/edit_retention.py", "examples/session_edit.py"]
    explanation_paths = [
        "artifacts/explanations/edit-retention/index.html",
        "artifacts/explanations/edit-retention/source.json",
        "artifacts/explanations/edit-retention/README.md",
    ]

    editorial = load(".runs/documentation-editorial-review-draft3.json")
    reviewed = {item["path"]: item for item in editorial["reviewed_documents"]}
    for path in guide_paths:
        if path not in reviewed or any(reviewed[path][key] != binding(path)[key] for key in ("bytes", "sha256")):
            raise ValueError(f"Final editorial receipt does not bind {path}")
    build = load(".runs/wheel-build-draft3.json")
    smoke = load(".runs/wheel-install-smoke-draft3.json")
    excerpts = load(".runs/worked-example-excerpts-draft3.json")
    browser = load(".runs/documentation-browser-draft3.json")
    arithmetic = load(".runs/illustration-arithmetic-draft3.json")
    integration = load(".runs/integration-draft3-verification.json")
    freeze = load(".runs/source-freeze-draft3-review.json")
    workflow = load(".runs/ci-draft3-validation/validation.json")
    wrapper = load(".runs/ci-draft3-validation/local-execution/validation.json")
    helpers = load(".runs/publication-helper-validation-draft3.json")
    if build["status"] != "passed" or smoke["status"] != "passed":
        raise ValueError("Current wheel build and installation must pass")
    if excerpts["result"] != "passed" or excerpts["exit_code"] != 0 or not excerpts["source_unchanged"]:
        raise ValueError("Worked-guide excerpts did not complete against unchanged document bytes")
    if browser["status"] != "passed_with_declared_unverified_surfaces" or browser["failures"]:
        raise ValueError("Browser review has unexpected status or failures")
    if not browser["checks"] or not all(item["passed"] is True for item in browser["checks"]):
        raise ValueError("Recorded browser checks did not pass")
    if browser["artifact"] != binding(explanation_paths[0]) or browser["source_model"] != binding(explanation_paths[1]):
        raise ValueError("Browser inputs changed after review")
    if not arithmetic["outcome"].startswith("passed") or integration["passed"] is not True:
        raise ValueError("Arithmetic or focused integration review did not pass")
    if freeze["snapshot_integrity_passed"] is not True or freeze["scientific_implementation_changed"] is not False:
        raise ValueError("Historical source-freeze boundary is not established by its receipt")
    if not freeze["strict_current_source_rejection"]:
        raise ValueError("Expected fresh-execution rejection was not recorded")
    if any(record["status"] != "passed" for record in (workflow, wrapper, helpers)):
        raise ValueError("Workflow/helper checks did not pass in their declared scopes")

    guidance = load(".runs/ste-reference/source.json")
    if guidance["repository"] != "shyamsridhar123/STE-Pro-Max" or guidance["commit"] != STE_COMMIT:
        raise ValueError("Unexpected writing-guidance source")
    guidance_files = []
    for item in guidance["files"]:
        raw = (ROOT / ".runs/ste-reference" / item["path"]).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if git_blob != item["blob_sha"]:
            raise ValueError(f"Retrieved guidance does not match pinned Git blob: {item['path']}")
        guidance_files.append({
            "repository_path": item["path"], "git_blob_sha": git_blob,
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        })

    timestamp = datetime.now(timezone.utc).isoformat()
    revision = {
        "schema": "wspf-documentation-revision/1",
        "assembled_at_utc": timestamp,
        "revision": TAG,
        "python_distribution": VERSION,
        "parent_publication": {"tag": "v0.2.0-draft.2", "commit": PARENT},
        "purpose": "Explain the generic framework through a concrete edit, its mechanism, the evidence and its limits, and the next research decisions.",
        "guidance": {
            "repository": guidance["repository"], "commit": STE_COMMIT,
            "acquisition_record": receipt(".runs/ste-reference/source.json"),
            "retrieval_byte_verification": receipt(".runs/ste-reference/source-byte-verification-draft3.json"),
            "reviewed_files": guidance_files,
            "applied": [
                "Factual fidelity and requested scope precede clarity and style.",
                "Start with a supplied concrete example, show intermediate states, and explain the mechanism before generalizing.",
                "Keep baselines, units, populations, uncertainty, contrary outcomes, and attribution attached to findings.",
                "Distinguish authored illustrations, recorded observations, conditional mathematical results, interpretation, and proposed work.",
                "Use a useful reader-controlled visual with inspectable model assumptions and static reading; test the actual artifact and record untested surfaces.",
                "Assess the explanation by supported conclusions and usable reader actions, not document length, test totals, theorem totals, or a readability score.",
            ],
            "application_boundary": "Host-authored Markdown and original HTML/CSS/JavaScript/SVG. The optional STE engine and its validators were not executed. No ASD-STE100 certification, learning benefit, or accessibility certification is claimed.",
            "upstream_reuse": "Guidance and examples were read. No upstream runtime or figures were copied into the explanation; pinned repository paths and content identities record the sources.",
        },
        "reader_questions": [
            {"question": "What changes when an edit protects another output?", "answer_path": "docs/weight-space/v2/worked-example.md", "mechanism": "Target equality and protected directions constrain the same update. Compatible constraints intersect; protecting the target makes the requested change infeasible."},
            {"question": "How does a candidate become an evaluated artifact?", "answer_path": "docs/weight-space/v2/workflow.md", "mechanism": "The caller validates and ranks candidates, records selection, seals it, and supplies evaluation results. The receipt binds records; it does not attest historical execution."},
            {"question": "What did the original report and local study establish?", "answer_paths": ["docs/weight-space/v2/source-review.md", "docs/weight-space/v2/methods.md", "docs/weight-space/v2/results.md"], "mechanism": "Separate original tasks and restrictions from local synthetic output-head studies, retain negative/no-op outcomes, and report the target-retention tradeoff with normalized errors and uncertainty."},
            {"question": "Which premises support a guarantee?", "answer_paths": ["docs/weight-space/v2/formal.md", "PROOFS.md"], "mechanism": "Feature-span and sequential real-valued results depend on declared endpoints, coordinates, domains, and hypotheses. Runtime extraction, compilation, native arithmetic, and candidate-to-proof binding remain separate obligations."},
            {"question": "What evidence is needed next?", "answer_paths": ["docs/weight-space/v2/limitations-and-next-steps.md", "ROADMAP.md"], "mechanism": "Define questions, controls, fresh units, access conditions, and acceptance/rejection criteria before claiming stronger scope."},
        ],
        "documents": [binding(path) for path in guide_paths],
        "navigation_and_historical_entry_points": [binding(path) for path in navigation_paths],
        "runnable_illustrations": [binding(path) for path in example_paths],
        "interactive_illustration": [binding(path) for path in explanation_paths],
        "supporting_publication_changes": [binding(path) for path in ["pyproject.toml", ".github/workflows/publication-python.yml", "scripts/check_proofs.py"]],
        "source_preservation": {
            "inventory": binding("provenance/source-files.json"),
            "mapping_review": receipt(".runs/docs-preservation-draft3.json"),
            "proof_lookup_review": receipt(".runs/integration-draft3-verification.json"),
            "freeze_review": receipt(".runs/source-freeze-draft3-review.json"),
            "mapping_rule": "Only explicit preserved_path values move original document inputs. Original inventory paths, hashes, byte lengths, order, and metadata retain their identity.",
            "original_document_examples": [binding(path) for path in [
                "provenance/original-documents/docs/weight-space/v2/source-review.md.txt",
                "provenance/original-documents/docs/weight-space/v2/limitations-and-next-steps.md.txt",
                "provenance/original-documents/docs/weight-space/v2/formal.md.txt",
                "provenance/original-documents/formal/weight-space/README.md.txt",
            ]],
        },
        "scientific_and_formal_scope": {
            "new_scientific_study": False,
            "new_lean_run": False,
            "historical_reference_inputs": [binding(path) for path in [
                "artifacts/weight-space/study-v2-report-complete/report.md",
                "artifacts/weight-space/study-v2-report-complete/summary.json",
                "formal/weight-space/WeightSpace/Generic.lean",
                "formal/weight-space/WeightSpace/Sequential.lean",
                "formal/weight-space/WeightSpace/GenericAudit.lean",
            ]],
            "interpretation": "The retained local study concerns synthetic feature maps and trained/edited output heads. It supports a target-retention tradeoff within that design. Existing seeds are observed; replay is not fresh confirmation. The conditional Lean results concern declared real-valued semantics. No live LLM/provider arm, useful large-model editing result, or automatic runtime-to-proof correspondence is added.",
        },
        "verification_record": "provenance/verification/runs/documentation-draft3.json",
        "final_publication_record_policy": "Final links, manifest/tree verification, staging, archive binding, exact-commit hosted CI, and remote release verification have separate records. This assembly record does not infer their outcomes.",
    }

    checks = [
        {
            "question": "Do the live explanations retain the supported conclusions, assumptions, and reader actions?",
            "status": "passed",
            "evidence": [receipt(".runs/documentation-editorial-review-draft3.json")],
            "scope": "Bounded static editorial review of the identified guides, source interpretation, equations as text, release notes, and retained diagram captions. This is a review finding, not a proof of completeness or a learning evaluation.",
        },
        {
            "question": "Do the exact Python excerpts in the worked guide run against the identified installed distribution?",
            "status": "passed",
            "evidence": [receipt(".runs/worked-example-excerpts-draft3.json"), receipt(".runs/verify_documentation_excerpts_draft3.py")],
            "scope": "The guide's Python fences ran in document order with -I -B against the already installed dev3 wheel. Imports resolved under that isolated environment. Endpoint identity, exact supplied-anchor residual, receipt verification, rollback, and positive-ridge target error were exercised. This run reused the fresh environment created by the separate installation check.",
        },
        {
            "question": "Can a reader use the current wheel and runnable examples outside the source checkout?",
            "status": "passed",
            "evidence": [receipt(path) for path in [".runs/wheel-build-draft3.json", ".runs/wheel-build-draft3.log", ".runs/wheel-install-smoke-draft3.json", ".runs/wheel-install-smoke-draft3.log"]],
            "scope": "Fresh PEP 517 build and core-only isolated installation, byte-copied examples, import-prefix checks, pip check, and module/console CLI help. The receipt identifies the wheel, source inputs, registry, Python/NumPy environment, and absent Torch. It is not a full optional-dependency or platform-support result.",
        },
        {
            "question": "Does the interactive illustration calculate its declared exact model correctly?",
            "status": "passed",
            "evidence": [receipt(".runs/illustration-arithmetic-draft3.json")],
            "scope": "The original inline script ran under a minimal DOM stub. An independent BigInt oracle checked the declared slider grid, protected sets, presets, no-op transitions, statuses, output tables, reset, and geometry. Supplied illustrative arithmetic is separate from native model execution and scientific evidence.",
        },
        {
            "question": "Does the authored explanation render and respond in the browser surfaces actually exercised?",
            "status": "passed",
            "evidence": [receipt(".runs/documentation-browser-draft3.json")],
            "scope": "Observed presets, protection choices, supported slider fill, reset, themes, desktop/narrow layouts, reduced-motion state, print emulation, no-JavaScript reading, and local resource-tree inspection. The receipt binds the final artifact and screenshots and explicitly identifies unverified keyboard/drag, assistive-technology, touch, offline-disconnection, and print-pagination surfaces.",
        },
        {
            "question": "Can historical proof/document inputs be resolved without silently changing their scientific identity?",
            "status": "passed",
            "evidence": [receipt(path) for path in [".runs/docs-preservation-draft3.json", ".runs/integration-draft3-verification.json", ".runs/source-freeze-draft3-review.json", ".runs/source-freeze-draft3-provenance.json"]],
            "scope": "Explicit preserved-document lookups and simulated digest mismatch rejection passed. Saved-freeze integrity was accepted in replay mode; strict current-source verification rejected the rewritten live source-review as expected. These are focused checks, not a new proof build, numerical study, or complete-tree verifier result.",
        },
        {
            "question": "Are the revised workflow and publication helpers consistent in the locally inspected scope?",
            "status": "passed",
            "evidence": [receipt(path) for path in [".runs/ci-draft3-validation/validation.json", ".runs/ci-draft3-validation/local-execution/validation.json", ".runs/ci-draft3-validation/local-execution/wheel-examples.json", ".runs/publication-helper-validation-draft3.json"]],
            "scope": "Workflow syntax/embedded code checks and static helper interfaces passed. The example-wrapper execution used the preserved dev2 wheel in an existing environment; the separate fresh dev3 records cover the current wheel. These checks do not establish hosted execution or a remote publication result.",
        },
    ]
    document_receipt = {
        "schema": "wspf-documentation-verification/1",
        "recorded_at_utc": timestamp,
        "revision": TAG,
        "result": "passed_with_declared_limits",
        "failures": [],
        "checks": checks,
        "historical_assembly_and_attempt_records": [receipt(path) for path in [
            ".runs/documentation-editorial-review-draft3-before-assembly.json",
            ".runs/documentation-links-draft3-assembly.json",
            ".runs/illustration-arithmetic-draft3-before-layout.json",
            ".runs/worked-example-excerpts-draft3-launch.json",
            ".runs/wheel-build-draft3-launch.json",
        ]],
        "final_local_link_record": {
            "path": "provenance/verification/runs/documentation-links-draft3.json",
            "interpretation": "Read its actual final outcome and document bindings. Publication finalization requires ok=true, no failures, and no pending-path exceptions. This path-only reference avoids a receipt hash cycle.",
        },
        "separate_publication_checks": [
            {"record": "publication-final-verification-draft3-release.json", "question": "Does the finalized publication tree satisfy its preservation and checksum checks?"},
            {"record": "staging-verification-draft3-release.json", "question": "Does the staged Git tree bind that exact verified snapshot?"},
            {"record": "source-archive-verification-draft3.json", "question": "Do the archive, wheel, manifest, and source commit match their inputs?"},
            {"record": "hosted-ci-verification-draft3.json", "question": "Did the required workflow and steps complete successfully for the released commit?"},
            {"record": "github-draft3-verify operation receipt", "question": "Do remote branch/tag/release/assets match the intended private unpublished draft?"},
        ],
        "limitations": [
            "This documentation receipt summarizes checks with different scopes. Their totals are not a measure of correctness, scientific strength, or explanation quality.",
            "No new scientific population, live LLM/provider run, or Lean elaboration is reported here. Historical evidence retains its original identities and boundaries.",
            "The exact toy model, useful edit, and analytic session rows are supplied. Zero toy errors do not establish autonomous discovery or experimental confirmation.",
            "Native keyboard activation and slider drag were not verified. No touch, screen-reader/accessibility certification, user-learning evaluation, or full offline-disconnection test was performed. Print emulation does not establish PDF pagination.",
            "The unusable tiled full-page screenshot and the pre-layout observations are retained with their actual limitations. Final viewport captures bind the reviewed layout.",
            "Fresh core installation used the configured Microsoft mirror on the recorded Windows/Python/NumPy environment without Torch. Public-PyPI portability and every supported platform/optional dependency are not established by it.",
            "Local destination/fragment checks do not validate external URLs or unpublished release assets. Their actual publication and remote outcomes require separate records.",
            "The optional STE engine validators were not run. Authored artifact review is not ASD-STE100 certification, factual attestation, or a guarantee of comprehension.",
        ],
    }
    write_new(outputs[0], revision)
    write_new(outputs[1], document_receipt)
    print(json.dumps({"result": "assembled", "outputs": [binding(path) for path in outputs]}))


if __name__ == "__main__":
    main()
