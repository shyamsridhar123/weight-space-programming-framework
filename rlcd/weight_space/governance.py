"""Optional local-pilot review records; see docs/governance/contract.md.

Labels are not authenticated. Independently retained, current request/head pins
and externally serialized decisions are caller obligations. No operation grants
production or scientific authority, interprets constraints, or runs a candidate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from .artifact import TensorArtifact, WeightArtifact
from .model import Network, canonical_bytes, digest
from .program import Program, apply_program


def _snapshot(value: Any) -> Any:
    """Own plain finite JSON without calling user-defined conversion protocols."""
    active: set[int] = set()

    def copy(item: Any) -> Any:
        kind = type(item)
        if item is None or kind in (bool, int, str):
            return item
        if kind is float and math.isfinite(item):
            return item
        if kind not in (dict, list):
            raise ValueError("Governance inputs must be plain finite JSON values")
        if id(item) in active:
            raise ValueError("Governance JSON cannot contain cycles")
        active.add(id(item))
        try:
            if kind is list:
                return [copy(child) for child in item]
            if any(type(key) is not str for key in item):
                raise ValueError("Governance JSON object keys must be strings")
            return {key: copy(child) for key, child in item.items()}
        finally:
            active.remove(id(item))

    try:
        result = copy(value)
        canonical_bytes(result)
        return result
    except RecursionError:
        raise ValueError("Governance JSON is nested too deeply") from None


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate governance JSON object key")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise ValueError("Nonfinite governance JSON number")


def _decode(content: bytes) -> Any:
    try:
        return _snapshot(json.loads(content.decode("utf-8-sig"),
                                    object_pairs_hook=_pairs, parse_constant=_constant))
    except RecursionError:
        raise ValueError("Governance JSON is nested too deeply") from None


def read_json(path: str | Path) -> Any:
    """Read strict finite JSON; the consuming API validates its exact schema."""
    return _decode(Path(path).read_bytes())


def _fields(value: Any, schema: str, fields: str) -> None:
    if (type(value) is not dict or set(value) != {"schema", *fields.split()}
            or value["schema"] != schema):
        raise ValueError(f"Invalid {schema} schema or fields")


def _text(value: Any) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("Governance text must be a nonblank string")


def _actor(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9._-]{0,63}", value) is None:
        raise ValueError("Governance actors require canonical lowercase ASCII labels")


def _sha(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Governance identities require lowercase SHA-256 values")


def _scope(value: Any) -> None:
    if (type(value) is not dict or set(value) != {"environment", "use_case", "target"}
            or value["environment"] != "local-pilot"):
        raise ValueError("Governance scope requires a local-pilot environment, use_case and target")
    _text(value["use_case"])
    _text(value["target"])


def _request(value: Any) -> None:
    _fields(value, "wsp-governance-request/1",
            "base_sha256 program_sha256 candidate_sha256 constraints_sha256 "
            "scope purpose evidence requester approvers")
    for key in ("base_sha256", "program_sha256", "candidate_sha256", "constraints_sha256"):
        _sha(value[key])
    _scope(value["scope"])
    _text(value["purpose"])
    _actor(value["requester"])
    approvers = value["approvers"]
    if type(approvers) is not list or not approvers:
        raise ValueError("Configure a nonempty list of approver labels")
    for actor in approvers:
        _actor(actor)
    if len(set(approvers)) != len(approvers):
        raise ValueError("Approver labels must be unique")
    evidence = value["evidence"]
    if type(evidence) is not list or not evidence:
        raise ValueError("Supply nonempty named evidence commitments")
    names: set[str] = set()
    for entry in evidence:
        if type(entry) is not dict or set(entry) != {"name", "sha256"}:
            raise ValueError("Evidence entries require exactly name and sha256")
        _text(entry["name"])
        _sha(entry["sha256"])
        if entry["name"] in names:
            raise ValueError("Evidence names must be unique")
        names.add(entry["name"])


def _artifact(value: Any) -> WeightArtifact:
    if type(value) is not dict:
        raise ValueError("Pass an artifact's plain JSON to_dict() value")
    if value.get("schema") == "wsp-tensors/1":
        return TensorArtifact.from_dict(value)
    if value.get("schema") == "wsp-network/1":
        return Network.from_dict(value)
    raise ValueError("Unsupported governance artifact schema")


def _context(base: Any, program: Any, candidate: Any,
             constraints: Any, scope: Any) -> dict[str, Any]:
    base, program, candidate, constraints, scope = _snapshot(
        [base, program, candidate, constraints, scope])
    base_artifact = _artifact(base)
    edit_program = Program.from_dict(program)
    candidate_artifact = _artifact(candidate)
    _scope(scope)
    if apply_program(base_artifact, edit_program).sha256 != candidate_artifact.sha256:
        raise ValueError("Governance candidate is not the exact application of this program")
    return {"base_sha256": base_artifact.sha256, "program_sha256": edit_program.sha256,
            "candidate_sha256": candidate_artifact.sha256,
            "constraints_sha256": digest(constraints), "scope": scope}


def _genesis(request_sha256: str) -> str:
    return digest({"schema": "wsp-governance-genesis/1", "request_sha256": request_sha256})


def _transition(request: dict[str, Any], status: str, actor: Any, action: Any) -> str:
    _actor(actor)
    if (status == "requested" and action in ("approve", "reject")
            and actor in request["approvers"] and actor != request["requester"]):
        return "approved" if action == "approve" else "rejected"
    if (status == "approved" and action == "revoke"
            and (actor == request["requester"] or actor in request["approvers"])):
        return "revoked"
    raise ValueError("Governance decision is not an allowed actor/action transition")


def _checked(bundle: Any, expected_request_sha256: str,
             expected_head_sha256: str) -> tuple[dict[str, Any], str]:
    _sha(expected_request_sha256)
    _sha(expected_head_sha256)
    bundle = _snapshot(bundle)
    _fields(bundle, "wsp-governance-bundle/1",
            "request request_sha256 events head_sha256 production_authorized scientific_authorized")
    if bundle["production_authorized"] is not False or bundle["scientific_authorized"] is not False:
        raise ValueError("Local governance cannot grant production or scientific authority")
    _request(bundle["request"])
    _sha(bundle["request_sha256"])
    _sha(bundle["head_sha256"])
    request_sha256 = digest(bundle["request"])
    if request_sha256 != bundle["request_sha256"] or request_sha256 != expected_request_sha256:
        raise ValueError("Governance request does not match its independently retained identity")
    if type(bundle["events"]) is not list:
        raise ValueError("Governance history must be a JSON array")
    status, head = "requested", _genesis(request_sha256)
    for position, event in enumerate(bundle["events"], 1):
        _fields(event, "wsp-governance-event/1",
                "request_sha256 previous_head_sha256 position actor action reason sha256")
        _text(event["reason"])
        _sha(event["sha256"])
        if (type(event["position"]) is not int or event["position"] != position
                or event["request_sha256"] != request_sha256
                or event["previous_head_sha256"] != head):
            raise ValueError("Governance history is not consecutively bound to this request and head")
        event_sha256 = digest({key: value for key, value in event.items() if key != "sha256"})
        if event_sha256 != event["sha256"]:
            raise ValueError("Governance event identity does not match its complete content")
        status = _transition(bundle["request"], status, event["actor"], event["action"])
        head = event_sha256
    if head != bundle["head_sha256"] or head != expected_head_sha256:
        raise ValueError("Governance history does not match the independently retained current head")
    return bundle, status


def create_request(base: dict[str, Any], program: dict[str, Any], candidate: dict[str, Any], *,
                   constraints: Any, scope: dict[str, Any], purpose: str,
                   evidence: list[dict[str, str]], requester: str,
                   approvers: list[str]) -> dict[str, Any]:
    """Commit a fresh exact change for local review; retain the returned pins."""
    metadata = _snapshot({"purpose": purpose, "evidence": evidence,
                          "requester": requester, "approvers": approvers})
    request = {"schema": "wsp-governance-request/1",
               **_context(base, program, candidate, constraints, scope),
               **metadata}
    _request(request)
    request_sha256 = digest(request)
    return {"schema": "wsp-governance-bundle/1", "request": request,
            "request_sha256": request_sha256, "events": [], "head_sha256": _genesis(request_sha256),
            "production_authorized": False, "scientific_authorized": False}


def decide(bundle: dict[str, Any], *, expected_request_sha256: str, expected_head_sha256: str,
           actor: str, action: str, reason: str) -> dict[str, Any]:
    """Append a valid decision to an owned copy; no authentication is provided."""
    result, status = _checked(bundle, expected_request_sha256, expected_head_sha256)
    actor, action, reason = _snapshot(actor), _snapshot(action), _snapshot(reason)
    _text(reason)
    _transition(result["request"], status, actor, action)
    event = {"schema": "wsp-governance-event/1", "request_sha256": result["request_sha256"],
             "previous_head_sha256": result["head_sha256"], "position": len(result["events"]) + 1,
             "actor": actor, "action": action, "reason": reason}
    event["sha256"] = digest(event)
    result["events"].append(event)
    result["head_sha256"] = event["sha256"]
    return result


def check_eligibility(bundle: dict[str, Any], *, expected_request_sha256: str,
                      expected_head_sha256: str, base: dict[str, Any], program: dict[str, Any],
                      candidate: dict[str, Any], constraints: Any, scope: dict[str, Any]) -> dict[str, Any]:
    """Check local eligibility as of both supplied pins and the exact context."""
    bundle, status = _checked(bundle, expected_request_sha256, expected_head_sha256)
    context = _context(base, program, candidate, constraints, scope)
    if context != {key: bundle["request"][key] for key in context}:
        raise ValueError("Current governance context differs from the reviewed request")
    return {"schema": "wsp-governance-check/1", "request_sha256": bundle["request_sha256"],
            "head_sha256": bundle["head_sha256"], "status": status,
            "local_pilot_eligible": status == "approved",
            "production_authorized": False, "scientific_authorized": False}


def export_bundle(source: str | Path, destination: str | Path, *,
                  expected_request_sha256: str, expected_head_sha256: str) -> dict[str, Any]:
    """Validate one source snapshot and exclusively copy its identical bytes.

    A successful return is not a transaction, deployment or eligibility check.
    Treat any write failure as failure even if bytes survive at the destination.
    """
    content = Path(source).read_bytes()
    bundle, _ = _checked(_decode(content), expected_request_sha256, expected_head_sha256)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(content)
    return bundle
