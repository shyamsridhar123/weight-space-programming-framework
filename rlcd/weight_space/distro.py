"""Workspace-scoped Distro review adapter; no decisions or new authority.

See docs/governance/distro-integration.md. Path checks reject known unsafe
paths; they are not a filesystem sandbox or a concurrent-writer guarantee.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from . import governance


_CONTEXT = ("base", "program", "candidate", "constraints", "scope")
_DESCRIPTOR_FIELDS = {"schema", *_CONTEXT, "purpose", "requester", "approvers", "evidence"}
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])$", re.IGNORECASE)
_PREVIEW_BYTES = 4096


def _not_link(path: Path, entry: os.stat_result) -> None:
    if (stat.S_ISLNK(entry.st_mode)
            or getattr(entry, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
        raise ValueError(f"Workspace paths cannot cross a symbolic link or reparse point: {path.name}")


class DistroWorkspace:
    """One configured local workspace, using the existing governance runtime."""

    def __init__(self, root: Path):
        path = Path(os.path.abspath(root))
        for ancestor in (*reversed(path.parents), path):
            entry = ancestor.lstat()
            _not_link(ancestor, entry)
            if not stat.S_ISDIR(entry.st_mode):
                raise ValueError("The configured workspace and its ancestors must be directories")
        self.root = path.resolve(strict=True)

    def _path(self, relative: str, *, output: bool = False) -> Path:
        if (type(relative) is not str or not relative or relative != relative.strip()
                or any(c in relative for c in '\\<>:"|?*')
                or any(ord(c) < 32 or ord(c) == 127 for c in relative)):
            raise ValueError("Use a normalized portable path relative to the configured workspace")
        parts = relative.split("/")
        if any(not part or part in (".", "..") or part.endswith((".", " "))
               or _RESERVED.fullmatch(part.split(".", 1)[0].rstrip(" ")) for part in parts):
            raise ValueError("Use a normalized portable path relative to the configured workspace")
        target = self.root
        for index, part in enumerate(parts):
            target /= part
            try:
                entry = target.lstat()
            except FileNotFoundError:
                if not output:
                    raise
                continue
            _not_link(target, entry)
            if index < len(parts) - 1 and not stat.S_ISDIR(entry.st_mode):
                raise ValueError("Workspace path has a non-directory ancestor")
            if index == len(parts) - 1 and not stat.S_ISREG(entry.st_mode):
                raise ValueError("Review inputs and outputs must be regular files")
        return target

    def _output(self, relative: str) -> Path:
        target = self._path(relative, output=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Recheck created directories. Concurrent directory replacement remains
        # outside this adapter's filesystem guarantees.
        return self._path(relative, output=True)

    def _load(self, change: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        descriptor = governance.read_json(self._path(change))
        if (type(descriptor) is not dict or set(descriptor) != _DESCRIPTOR_FIELDS
                or descriptor["schema"] != "wsp-distro-change/1"):
            raise ValueError("Invalid wsp-distro-change/1 schema or fields")

        documents: dict[Path, Any] = {}
        context: dict[str, Any] = {}
        for name in _CONTEXT:
            path = self._path(descriptor[name])
            if path not in documents:
                documents[path] = governance.read_json(path)
            context[name] = documents[path]

        entries = descriptor["evidence"]
        if type(entries) is not list:
            raise ValueError("Evidence must be a nonempty array of named workspace paths")
        captures: dict[Path, bytes] = {}
        attachments = []
        for entry in entries:
            if type(entry) is not dict or set(entry) != {"name", "path"}:
                raise ValueError("Distro evidence entries require exactly name and path")
            path = self._path(entry["path"])
            if path not in captures:
                captures[path] = path.read_bytes()
            content = captures[path]
            attachments.append({
                "name": entry["name"], "path": entry["path"],
                "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
                "preview": content[:_PREVIEW_BYTES].decode("utf-8", errors="replace"),
                "preview_truncated": len(content) > _PREVIEW_BYTES,
            })

        request = governance.create_request(
            **context, purpose=descriptor["purpose"], requester=descriptor["requester"],
            approvers=descriptor["approvers"],
            evidence=[{"name": item["name"], "sha256": item["sha256"]} for item in attachments],
        )
        return context, request, attachments

    @staticmethod
    def _details(context: dict[str, Any], bundle: dict[str, Any],
                 attachments: list[dict[str, Any]]) -> dict[str, Any]:
        return {"request": bundle["request"], "events": bundle["events"],
                "program": context["program"], "attachments": attachments}

    @staticmethod
    def _require_request(request: dict[str, Any], expected_request_sha256: str) -> None:
        if request["request_sha256"] != expected_request_sha256:
            raise ValueError("Current complete change differs from the independently retained request identity")

    def inspect_change(self, change: str) -> dict[str, Any]:
        """Inspect an exact draft without creating a review or running a model."""
        context, request, attachments = self._load(change)
        return {"status": "draft", "request_sha256": request["request_sha256"],
                "local_pilot_eligible": False, "production_authorized": False,
                "scientific_authorized": False, **self._details(context, request, attachments)}

    def prepare_review(self, change: str, out: str) -> dict[str, Any]:
        """Create a new core bundle; retain both returned pins independently."""
        context, request, attachments = self._load(change)
        target = self._output(out)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(request, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return {"bundle": out, "status": "requested", "local_pilot_eligible": False,
                "request_sha256": request["request_sha256"], "head_sha256": request["head_sha256"],
                "production_authorized": False, "scientific_authorized": False,
                **self._details(context, request, attachments)}

    def review_status(self, change: str, bundle: str,
                      expected_request_sha256: str, expected_head_sha256: str) -> dict[str, Any]:
        """Check the complete current change against both independent pins."""
        context, request, attachments = self._load(change)
        self._require_request(request, expected_request_sha256)
        current = governance.read_json(self._path(bundle))
        result = governance.check_eligibility(
            current, expected_request_sha256=expected_request_sha256,
            expected_head_sha256=expected_head_sha256, **context,
        )
        return {**result, "bundle": bundle, **self._details(context, current, attachments)}

    def export_review(self, change: str, bundle: str,
                      expected_request_sha256: str, expected_head_sha256: str,
                      out: str) -> dict[str, Any]:
        """Validate and export one complete source snapshot without reencoding."""
        context, request, attachments = self._load(change)
        self._require_request(request, expected_request_sha256)
        source, target = self._path(bundle), self._output(out)
        current = governance.export_bundle(
            source, target, expected_request_sha256=expected_request_sha256,
            expected_head_sha256=expected_head_sha256,
        )
        result = governance.check_eligibility(
            current, expected_request_sha256=expected_request_sha256,
            expected_head_sha256=expected_head_sha256, **context,
        )
        return {**result, "bundle": bundle, "out": out,
                **self._details(context, current, attachments)}
