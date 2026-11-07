#!/usr/bin/env python3
"""Verify preserved source bytes, publication coverage, and live documentation.

This standard-library-only checker does not import the framework or run studies.
Its default operation is read-only. Optional JSON receipts must be new files;
inside the repository they belong under the excluded ``.runs/`` directory.
An optional public sanitization ledger binds disclosed derivatives or omissions
to original identities; only unchanged files count as verified original bytes.

``publication_files(root)`` and ``is_excluded_publication_path(path)`` define the
same publication selection policy used to create MANIFEST.sha256. The manifest
excludes itself. Historical caches inside artifacts/ remain published evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePath
import re
import stat
import string
import sys
from datetime import datetime, timezone
from urllib.parse import unquote, urlsplit


SCHEMA = "weight-space-programming-framework.publication-verification.v1"
MANIFEST = "MANIFEST.sha256"
SOURCE_INDEX = "provenance/source-files.json"
PUBLIC_SANITIZATION = "provenance/public-sanitization.json"
PUBLIC_SANITIZATION_SCHEMA = "wspf-public-sanitization/1"
ROOT_EXCLUSIONS = frozenset({
    ".git", ".venv", ".proof-runs", ".runs", ".pytest_cache", ".pytest-tmp",
    ".cache", "node_modules", "build", "dist",
})
NONARCHIVE_CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache"})
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
MANIFEST_LINE = re.compile(r"([0-9a-fA-F]{64})  (.+)\Z")
MARKDOWN_ESCAPE = re.compile(r"\\([" + re.escape(string.punctuation) + r"])")
URI_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
WINDOWS_DEVICES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def is_excluded_publication_path(relative_path: str | PurePath) -> bool:
    """Return whether a repository-relative path is outside manifest coverage.

    Exclusions are deliberately narrow. In particular, artifacts/ is included
    before considering cache-directory names at any depth.
    """
    name = relative_path.as_posix() if isinstance(relative_path, PurePath) else relative_path
    parts = name.split("/")
    if parts[0] == "artifacts":
        return False
    if name == MANIFEST:
        return True
    if parts[0] in ROOT_EXCLUSIONS or parts[0].endswith(".egg-info"):
        return True
    if any(part in NONARCHIVE_CACHE_DIRS for part in parts):
        return True
    return parts[:3] == ["formal", "weight-space", ".lake"]


def _failure(failures: list[dict], code: str, message: str,
             path: str | None = None, line: int | None = None) -> None:
    item = {"code": code, "message": message}
    if path is not None:
        item["path"] = path
    if line is not None:
        item["line"] = line
    failures.append(item)


def _safe_relative_path(value: object) -> str:
    """Require a portable, normalized, relative POSIX file path."""
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a nonempty string")
    if any(char in value for char in '\\:<>\"|?*') or any(
        ord(char) < 32 or ord(char) == 127 for char in value
    ):
        raise ValueError("path contains a nonportable filename or control character")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path must be relative and normalized, without empty, '.' or '..' components")
    if any(part.endswith((".", " ")) or part.startswith(" ") for part in parts):
        raise ValueError("path components cannot begin with spaces or end with dots or spaces")
    if any(part.split(".", 1)[0].upper() in WINDOWS_DEVICES for part in parts):
        raise ValueError("path contains a reserved Windows device name")
    return value


def _is_link(file_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(file_stat.st_mode) or bool(
        getattr(file_stat, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _scan_publication_files(root: Path) -> tuple[list[str], list[dict]]:
    files: list[str] = []
    failures: list[dict] = []
    seen: dict[str, str] = {}

    def visit(directory: Path, prefix: str = "") -> None:
        try:
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except OSError as exc:
            _failure(failures, "publication_scan", str(exc), prefix or ".")
            return
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            if is_excluded_publication_path(relative):
                continue
            try:
                _safe_relative_path(relative)
                entry_stat = entry.stat(follow_symlinks=False)
                if _is_link(entry_stat):
                    raise ValueError("symbolic links and filesystem reparse points are not publishable")
                folded = relative.casefold()
                if folded in seen:
                    raise ValueError(f"case-insensitive path collision with {seen[folded]!r}")
                seen[folded] = relative
                if stat.S_ISDIR(entry_stat.st_mode):
                    visit(Path(entry.path), relative)
                elif stat.S_ISREG(entry_stat.st_mode):
                    files.append(relative)
                else:
                    raise ValueError("publication entries must be regular files or directories")
            except (OSError, ValueError) as exc:
                _failure(failures, "publication_path", str(exc), relative)

    visit(root)
    return sorted(files), failures


def publication_files(root: Path) -> list[str]:
    """Return sorted relative POSIX file paths; reject unsafe filesystem entries.

    This helper is safe to import for manifest generation. It does not write
    files, calculate hashes, inspect Markdown, or import project modules.
    """
    files, failures = _scan_publication_files(Path(root).resolve())
    if failures:
        first = failures[0]
        raise ValueError(f"{first.get('path', '.')}: {first['message']} ({len(failures)} scan failure(s))")
    return files


def _repository_file(root: Path, relative: str) -> Path:
    _safe_relative_path(relative)
    target = root
    for part in relative.split("/"):
        target /= part
        if _is_link(target.lstat()):
            raise ValueError("path crosses a symbolic link or filesystem reparse point")
    if not target.resolve().is_relative_to(root):
        raise ValueError("path resolves outside the repository")
    if not target.is_file():
        raise ValueError("path is not a regular file")
    return target


def _digest(root: Path, relative: str, cache: dict[str, tuple[int, str]]) -> tuple[int, str]:
    if relative not in cache:
        target = _repository_file(root, relative)
        digest = hashlib.sha256()
        size = 0
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        cache[relative] = (size, digest.hexdigest())
    return cache[relative]


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _sanitization_identity(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    size, digest = value.get("bytes"), value.get("sha256")
    if type(size) is not int or size < 0:
        raise ValueError(f"{label}.bytes must be a nonnegative integer")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ValueError(f"{label}.sha256 must contain exactly 64 hexadecimal characters")
    return {"bytes": size, "sha256": digest.lower()}


def _require_absent(root: Path, relative: str) -> None:
    """Reject present entries, including broken links and linked ancestors."""
    _safe_relative_path(relative)
    target = root
    parts = relative.split("/")
    for number, part in enumerate(parts):
        target /= part
        try:
            entry_stat = target.lstat()
        except FileNotFoundError:
            return
        if _is_link(entry_stat):
            raise ValueError("omitted path crosses a symbolic link or filesystem reparse point")
        if number == len(parts) - 1:
            raise ValueError("omitted path is still present")
        if not stat.S_ISDIR(entry_stat.st_mode):
            raise ValueError("omitted path has a non-directory ancestor")


def _indexed_original_bindings(root: Path) -> dict[str, tuple[str, dict]]:
    try:
        index = json.loads(_repository_file(root, SOURCE_INDEX).read_text(encoding="utf-8"),
                           object_pairs_hook=_unique_json_object)
    except FileNotFoundError:
        # Standalone historical consumers may have no source index. The full
        # publication verifier independently requires one.
        return {}
    records = index.get("files") if isinstance(index, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("source index must contain a nonempty 'files' array")
    bindings: dict[str, tuple[str, dict]] = {}
    sources: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("source entry must be an object")
        source = _safe_relative_path(record.get("path"))
        preserved = _safe_relative_path(record.get("preserved_path"))
        if source.casefold() in sources or preserved.casefold() in bindings:
            raise ValueError("source index has a duplicate original or preserved path")
        sources.add(source.casefold())
        bindings[preserved.casefold()] = (
            preserved, _sanitization_identity(record, "indexed original"),
        )
    return bindings


def load_public_sanitization(
    root: Path, cache: dict[str, tuple[int, str]] | None = None,
) -> dict[str, dict]:
    """Load and validate the optional ledger, returning records keyed by path.

    Every public derivative must match its recorded size and raw SHA-256;
    omissions must actually be absent. Originals mapped by the source index's
    preserved_path must match its unchanged size and hash. Other original
    identities and source_commit are disclosures, not independently
    authenticated historical bytes. Invalid ledgers raise rather than exposing
    a partially accepted set of records. This function performs no writes.
    """
    root = Path(root).resolve()
    try:
        ledger = json.loads(_repository_file(root, PUBLIC_SANITIZATION).read_text(encoding="utf-8"),
                            object_pairs_hook=_unique_json_object)
    except FileNotFoundError:
        return {}
    if not isinstance(ledger, dict) or ledger.get("schema") != PUBLIC_SANITIZATION_SCHEMA:
        raise ValueError("unsupported public sanitization schema")
    commit = ledger.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("source_commit must be a full 40-character hexadecimal commit identity")
    records = ledger.get("files")
    if not isinstance(records, list):
        raise ValueError("public sanitization ledger must contain a 'files' array")
    bindings = _indexed_original_bindings(root)
    cache = {} if cache is None else cache
    result: dict[str, dict] = {}
    seen: set[str] = set()
    for number, record in enumerate(records, 1):
        try:
            if not isinstance(record, dict):
                raise ValueError("entry must be an object")
            relative = _safe_relative_path(record.get("path"))
            folded = relative.casefold()
            if folded in seen:
                raise ValueError("duplicate or case-colliding ledger path")
            if any(folded.startswith(path + "/") or path.startswith(folded + "/") for path in seen):
                raise ValueError("ledger file paths cannot be ancestors of one another")
            seen.add(folded)
            if (is_excluded_publication_path(relative) or is_excluded_publication_path(folded)
                    or folded == MANIFEST.casefold()):
                raise ValueError("ledger target is outside publication coverage")
            if folded == PUBLIC_SANITIZATION.casefold():
                raise ValueError("ledger cannot describe itself")
            original = _sanitization_identity(record.get("original"), "original")
            if "public" not in record:
                raise ValueError("entry must contain 'public'; use null for an omission")
            public = None if record["public"] is None else _sanitization_identity(record["public"], "public")
            reasons = record.get("reasons")
            if not isinstance(reasons, list) or not reasons or any(
                not isinstance(reason, str) or not reason.strip() for reason in reasons
            ):
                raise ValueError("reasons must be a nonempty array of nonempty strings")
            if folded in bindings:
                preserved, expected = bindings[folded]
                if relative != preserved:
                    raise ValueError("ledger path must use the indexed preserved_path spelling")
                if original != expected:
                    raise ValueError("original binding does not match the source index")
            if public is None:
                _require_absent(root, relative)
            else:
                if public == original:
                    raise ValueError("a redacted public binding must differ from its original")
                actual_size, actual_sha = _digest(root, relative, cache)
                if public != {"bytes": actual_size, "sha256": actual_sha}:
                    raise ValueError("public binding does not match the published bytes")
            result[relative] = {
                "path": relative, "original": original, "public": public, "reasons": list(reasons),
            }
        except (OSError, ValueError) as exc:
            raise ValueError(f"public sanitization entry {number}: {exc}") from exc
    return result


def match_public_sanitization(
    entries: dict[str, dict], relative: str, original_sha256: str,
    original_bytes: int | None = None,
) -> str | None:
    """Match a historical identity to a freshly validated ledger entry.

    Return 'redacted', 'omitted', or None. Call load_public_sanitization first;
    this helper matches the disclosure, not the current filesystem. Supply
    original_bytes whenever the historical record contains a byte count.
    """
    _safe_relative_path(relative)
    if not isinstance(original_sha256, str) or not SHA256.fullmatch(original_sha256):
        raise ValueError("original_sha256 must contain exactly 64 hexadecimal characters")
    if original_bytes is not None and (type(original_bytes) is not int or original_bytes < 0):
        raise ValueError("original_bytes must be a nonnegative integer")
    entry = entries.get(relative)
    if entry is None or entry["original"]["sha256"] != original_sha256.lower():
        return None
    if original_bytes is not None and entry["original"]["bytes"] != original_bytes:
        return None
    return "omitted" if entry["public"] is None else "redacted"


def _verify_originals(root: Path, cache: dict[str, tuple[int, str]],
                      counts: dict[str, int], failures: list[dict],
                      sanitization: dict[str, dict] | None = None) -> None:
    sanitization = {} if sanitization is None else sanitization
    try:
        index = json.loads(_repository_file(root, SOURCE_INDEX).read_text(encoding="utf-8"))
        records = index.get("files") if isinstance(index, dict) else None
        if not isinstance(records, list) or not records:
            raise ValueError("source index must contain a nonempty 'files' array")
    except (OSError, ValueError) as exc:
        _failure(failures, "source_index", str(exc), SOURCE_INDEX)
        return
    counts["source_records"] = len(records)
    seen_sources: set[str] = set()
    seen_preserved: set[str] = set()
    for number, record in enumerate(records, 1):
        relative = SOURCE_INDEX
        try:
            if not isinstance(record, dict):
                raise ValueError("source entry must be an object")
            original = _safe_relative_path(record.get("path"))
            relative = _safe_relative_path(record.get("preserved_path"))
            if original.casefold() in seen_sources or relative.casefold() in seen_preserved:
                raise ValueError("duplicate original or preserved path")
            seen_sources.add(original.casefold())
            seen_preserved.add(relative.casefold())
            expected_size = record.get("bytes")
            expected_sha = record.get("sha256")
            if type(expected_size) is not int or expected_size < 0:
                raise ValueError("'bytes' must be a nonnegative integer")
            if not isinstance(expected_sha, str) or not SHA256.fullmatch(expected_sha):
                raise ValueError("'sha256' must contain exactly 64 hexadecimal characters")
            if is_excluded_publication_path(relative):
                raise ValueError("preserved original is outside publication coverage")
            disposition = match_public_sanitization(sanitization, relative, expected_sha, expected_size)
            if disposition == "omitted":
                _require_absent(root, relative)
                counts["source_files_omitted"] += 1
                continue
            actual_size, actual_sha = _digest(root, relative, cache)
            if disposition == "redacted" and sanitization[relative]["public"] == {
                "bytes": actual_size, "sha256": actual_sha,
            }:
                counts["source_files_redacted"] += 1
                continue
            matches = True
            if actual_size != expected_size:
                matches = False
                _failure(failures, "source_size", f"expected {expected_size} bytes, found {actual_size}", relative)
            if actual_sha != expected_sha.lower():
                matches = False
                _failure(failures, "source_sha256", f"expected {expected_sha.lower()}, found {actual_sha}", relative)
            if matches:
                counts["source_files_verified"] += 1
                counts["source_bytes_verified"] += actual_size
        except (OSError, ValueError) as exc:
            _failure(failures, "source_entry", f"entry {number}: {exc}", relative)


def _verify_manifest(root: Path, files: list[str], cache: dict[str, tuple[int, str]],
                     counts: dict[str, int], failures: list[dict]) -> None:
    try:
        lines = _repository_file(root, MANIFEST).read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError) as exc:
        _failure(failures, "manifest_read", str(exc), MANIFEST)
        return
    entries: dict[str, str] = {}
    seen: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        if not line:
            continue
        match = MANIFEST_LINE.fullmatch(line)
        if not match:
            _failure(failures, "manifest_format", "expected '<64 hex characters>  <relative POSIX path>'", MANIFEST, number)
            continue
        digest, relative = match.groups()
        try:
            _safe_relative_path(relative)
            folded = relative.casefold()
            if folded in seen:
                raise ValueError(f"duplicate or case-colliding manifest path: {seen[folded]!r}")
            seen[folded] = relative
            if is_excluded_publication_path(relative):
                raise ValueError("manifest entry is excluded by the publication policy")
            entries[relative] = digest.lower()
        except ValueError as exc:
            _failure(failures, "manifest_path", str(exc), MANIFEST, number)
    counts["manifest_entries"] = len(entries)
    actual_paths = set(files)
    for relative in sorted(actual_paths - entries.keys()):
        _failure(failures, "manifest_missing_entry", "published file has no manifest entry", relative)
    for relative in sorted(entries.keys() - actual_paths):
        _failure(failures, "manifest_extra_entry", "manifest path is absent from the publishable file set", relative)
    for relative, expected in entries.items():
        try:
            _, actual = _digest(root, relative, cache)
            if actual != expected:
                _failure(failures, "manifest_sha256", f"expected {expected}, found {actual}", relative)
            else:
                counts["manifest_files_verified"] += 1
        except (OSError, ValueError) as exc:
            _failure(failures, "manifest_file", str(exc), relative)


def _blank(text: str) -> str:
    return "".join("\n" if char == "\n" else " " for char in text)


def _fence_line(line: str) -> str:
    """Remove ordinary blockquote and same-line list containers."""
    while True:
        match = re.match(r"^ {0,3}>[ \t]?", line)
        if not match:
            break
        line = line[match.end():]
    return re.sub(r"^ {0,3}(?:[-+*]|\d{1,9}[.)])[ \t]+", "", line, count=1)


def _visible_markdown(text: str, relative: str, failures: list[dict]) -> str:
    """Mask fenced code and comments while retaining offsets and line numbers."""
    result: list[str] = []
    fence: tuple[str, int] | None = None
    in_comment = False
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        if fence is not None:
            candidate = _fence_line(line)
            if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(fence[1]) + r",}[ \t]*(?:\r?\n)?", candidate):
                fence = None
            result.append(_blank(line))
            continue
        parts: list[str] = []
        cursor = 0
        while cursor < len(line):
            if in_comment:
                end = line.find("-->", cursor)
                end = len(line) if end < 0 else end + 3
                parts.append(_blank(line[cursor:end]))
                in_comment = not line[cursor:end].endswith("-->")
                cursor = end
            else:
                start = line.find("<!--", cursor)
                if start < 0:
                    parts.append(line[cursor:])
                    break
                parts.append(line[cursor:start])
                cursor = start
                in_comment = True
        visible = "".join(parts)
        candidate = _fence_line(visible)
        match = re.match(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)", candidate)
        if match and not (match[1][0] == "`" and "`" in match[2]):
            fence = (match[1][0], len(match[1]))
            language = match[2].strip().split(maxsplit=1)
            if language and language[0].strip("{}.").casefold() == "mermaid":
                _failure(failures, "markdown_mermaid", "live Markdown must use a rendered diagram instead of a Mermaid fence", relative, number)
            visible = _blank(line)
        result.append(visible)
    visible = "".join(result)
    return re.sub(r"<(pre|code|script|style)\b[^>]*>.*?</\1\s*>",
                  lambda match: _blank(match[0]), visible, flags=re.IGNORECASE | re.DOTALL)


def _destination(text: str, start: int) -> tuple[str, int] | None:
    """Read an angle-delimited or balanced bare Markdown destination."""
    if start >= len(text):
        return None
    if text[start] == "<":
        cursor = start + 1
        while cursor < len(text):
            if text[cursor] == "\\" and cursor + 1 < len(text):
                cursor += 2
            elif text[cursor] == ">":
                return text[start + 1:cursor], cursor + 1
            elif text[cursor] in "\r\n<":
                return None
            else:
                cursor += 1
        return None
    cursor, depth = start, 0
    while cursor < len(text):
        char = text[cursor]
        if char == "\\" and cursor + 1 < len(text):
            cursor += 2
            continue
        if char.isspace() or (char == ")" and depth == 0):
            break
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char in "<>":
            return None
        cursor += 1
    if depth:
        return None
    return text[start:cursor], cursor


def _inline_destination(text: str, start: int) -> tuple[str, int] | None:
    cursor = start
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    parsed = _destination(text, cursor)
    if parsed is None:
        return None
    target, cursor = parsed
    before_space = cursor
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor > before_space and cursor < len(text) and text[cursor] in "\"'(":
        close = ")" if text[cursor] == "(" else text[cursor]
        cursor += 1
        while cursor < len(text):
            if text[cursor] == "\\" and cursor + 1 < len(text):
                cursor += 2
            elif text[cursor] == close:
                cursor += 1
                break
            else:
                cursor += 1
        else:
            return None
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    if cursor < len(text) and text[cursor] == ")":
        return target, cursor + 1
    return None


def _inline_targets(text: str) -> tuple[list[tuple[str, int, bool]], str]:
    targets: list[tuple[str, int, bool]] = []
    brackets: list[int] = []
    code_spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "`":
            run = re.match(r"`+", text[cursor:])[0]
            close = re.search(r"(?<!`)" + re.escape(run) + r"(?!`)", text[cursor + len(run):])
            if close:
                end = cursor + len(run) + close.end()
                code_spans.append((cursor, end))
                cursor = end
                continue
            cursor += len(run)
            continue
        if char == "<":
            tag = re.match(r"<(?:[^\"'>]|\"[^\"]*\"|'[^']*')*>", text[cursor:])
            if tag:
                cursor += tag.end()
                continue
        if char == "[":
            brackets.append(cursor)
        elif char == "]" and brackets:
            opening = brackets.pop()
            if cursor + 1 < len(text) and text[cursor + 1] == "(":
                parsed = _inline_destination(text, cursor + 2)
                if parsed:
                    target, end = parsed
                    targets.append((target, text.count("\n", 0, opening) + 1,
                                    opening > 0 and text[opening - 1] == "!"))
                    cursor = end
                    continue
        cursor += 1
    # HTML and reference-definition scanning must ignore the same code spans.
    pieces: list[str] = []
    cursor = 0
    for start, end in code_spans:
        pieces.extend((text[cursor:start], _blank(text[start:end])))
        cursor = end
    pieces.append(text[cursor:])
    return targets, "".join(pieces)


class _HTMLTargets(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.targets: list[tuple[str, int, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is not None and (
                (name == "href" and tag in {"a", "link"})
                or (name == "src" and tag in {"img", "source", "video", "audio", "iframe", "script"})
                or (name == "poster" and tag == "video")
            ):
                self.targets.append((value, self.getpos()[0], tag in {"img", "source"} or name == "poster"))


def _markdown_targets(text: str) -> list[tuple[str, int, bool]]:
    targets, text = _inline_targets(text)
    # Checking definitions also covers full, collapsed, and shortcut references.
    # Undefined bracketed text is ordinary Markdown text, not a local file URL.
    for match in re.finditer(r"(?m)^ {0,3}\[(?:\\.|[^\]\\\n])+\]:[ \t]*(?:\r?\n[ \t]*)?", text):
        parsed = _destination(text, match.end())
        if parsed:
            targets.append((parsed[0], text.count("\n", 0, match.start()) + 1, False))
    parser = _HTMLTargets()
    parser.feed(text)
    parser.close()
    targets.extend(parser.targets)
    return targets


def _verify_markdown(root: Path, files: list[str], counts: dict[str, int], failures: list[dict]) -> None:
    published = set(files)
    if (root / MANIFEST).is_file():
        published.add(MANIFEST)
    directories = {"."}
    for relative in published:
        directories.update(parent.as_posix() for parent in Path(relative).parents)
    for relative in files:
        parts = relative.split("/")
        if Path(relative).suffix.lower() not in {".md", ".markdown"}:
            continue
        if len(parts) != 1 and parts[0] not in {"docs", "formal"}:
            continue
        counts["markdown_files"] += 1
        try:
            raw = _repository_file(root, relative).read_text(encoding="utf-8")
            visible = _visible_markdown(raw, relative, failures)
            targets = _markdown_targets(visible)
        except (OSError, ValueError) as exc:
            _failure(failures, "markdown_read", str(exc), relative)
            continue
        for raw_target, number, is_image in targets:
            counts["markdown_destinations"] += 1
            target = html.unescape(MARKDOWN_ESCAPE.sub(r"\1", raw_target))
            try:
                if re.match(r"^[A-Za-z]:", target):
                    raise ValueError("local link uses an absolute Windows path")
                if URI_SCHEME.match(target) or target.startswith("//"):
                    counts["markdown_external_destinations"] += 1
                    continue
                path = unquote(urlsplit(target).path, encoding="utf-8", errors="strict")
                if not path:
                    counts["markdown_fragment_destinations"] += 1
                    continue
                counts["markdown_local_destinations"] += 1
                if path.startswith("/") or "\\" in path or ":" in path or "\0" in path:
                    raise ValueError("local link must use a relative repository path")
                candidate = root.joinpath(*parts[:-1], path)
                resolved = candidate.resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError("local link escapes the repository")
                lexical = Path(os.path.abspath(candidate))
                if not lexical.is_relative_to(root):
                    raise ValueError("local link escapes the repository")
                normalized = lexical.relative_to(root).as_posix()
                if normalized in published:
                    _repository_file(root, normalized)
                elif not is_image and normalized in directories and lexical.is_dir():
                    pass
                else:
                    raise ValueError("local target is absent from the published files or directories")
            except (OSError, ValueError) as exc:
                _failure(failures, "markdown_local_link", f"{raw_target!r}: {exc}", relative, number)


def verify(root: Path) -> dict:
    """Perform read-only checks and return a JSON-serializable receipt."""
    root = Path(root).resolve()
    files, failures = _scan_publication_files(root)
    counts = dict.fromkeys((
        "publication_files", "source_records", "source_files_verified", "source_bytes_verified",
        "source_files_redacted", "source_files_omitted", "sanitization_records",
        "sanitization_public_files_verified", "sanitization_omissions_verified",
        "manifest_entries", "manifest_files_verified", "markdown_files", "markdown_destinations",
        "markdown_local_destinations", "markdown_external_destinations", "markdown_fragment_destinations",
    ), 0)
    counts["publication_files"] = len(files)
    cache: dict[str, tuple[int, str]] = {}
    try:
        sanitization = load_public_sanitization(root, cache)
    except (OSError, ValueError) as exc:
        _failure(failures, "public_sanitization", str(exc), PUBLIC_SANITIZATION)
        sanitization = {}
    counts["sanitization_records"] = len(sanitization)
    counts["sanitization_omissions_verified"] = sum(
        entry["public"] is None for entry in sanitization.values()
    )
    counts["sanitization_public_files_verified"] = (
        len(sanitization) - counts["sanitization_omissions_verified"]
    )
    _verify_originals(root, cache, counts, failures, sanitization)
    _verify_manifest(root, files, cache, counts, failures)
    _verify_markdown(root, files, counts, failures)
    counts["failures"] = len(failures)
    return {
        "schema": SCHEMA,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "ok": not failures,
        "counts": counts,
        "policy": {
            "root_exclusions": sorted(ROOT_EXCLUSIONS) + ["*.egg-info", MANIFEST],
            "nonarchive_cache_directories": sorted(NONARCHIVE_CACHE_DIRS),
            "formal_build_directory": "formal/weight-space/.lake",
            "artifacts": "all files included, including historical caches; links rejected",
            "markdown_scope": "root Markdown files and Markdown below docs/ and formal/",
            "markdown_fragments": "ignored; local file and directory existence checked",
            "external_urls": "ignored; no network requests",
            "public_sanitization": (
                f"optional {PUBLIC_SANITIZATION}; public hashes or omissions verified; "
                "mapped original identities checked against source-index preserved_path; "
                "other original identities and source_commit are disclosed, not authenticated"
            ),
            "source_counts": (
                "source_files_verified and source_bytes_verified count exact originals only; "
                "disclosed redactions and omissions are counted separately"
            ),
        },
        "failures": failures,
    }


def _receipt_destination(value: str, root: Path) -> Path:
    requested = Path(os.path.abspath(Path(value).expanduser()))
    resolved = requested.resolve()
    for candidate in (requested, resolved):
        if any(part.casefold() in {"artifacts", "provenance"} for part in candidate.parts):
            raise ValueError("receipt output cannot be inside artifacts/ or provenance/ archives")
        if candidate.is_relative_to(root):
            relative = candidate.relative_to(root)
            if len(relative.parts) < 2 or relative.parts[0] != ".runs":
                raise ValueError("receipt output inside the repository must be a new file under .runs/")
    if requested.exists() or requested.is_symlink() or resolved.exists():
        raise ValueError("receipt output already exists; choose a new path")
    return requested


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", metavar="NEW_JSON", help="write a new JSON receipt; default is read-only")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = None
    if args.out is not None:
        try:
            output = _receipt_destination(args.out, root)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    receipt = verify(root)
    counts = receipt["counts"]
    print(f"Publication verification: {'PASS' if receipt['ok'] else 'FAIL'}")
    print(f"Source records: {counts['source_records']}; exact originals: {counts['source_files_verified']}; "
          f"disclosed redactions: {counts['source_files_redacted']}; "
          f"disclosed omissions: {counts['source_files_omitted']}")
    print(f"Sanitization records: {counts['sanitization_records']}; "
          f"manifest digests: {counts['manifest_files_verified']}/{counts['manifest_entries']}; "
          f"publishable files: {counts['publication_files']}")
    print(f"Live Markdown: {counts['markdown_files']} files, "
          f"{counts['markdown_local_destinations']} local destinations; "
          f"failures: {counts['failures']}")
    for failure in receipt["failures"][:30]:
        location = failure.get("path", ".")
        if "line" in failure:
            location += f":{failure['line']}"
        print(f"  {failure['code']}: {location}: {failure['message']}")
    if counts["failures"] > 30:
        print(f"  ... {counts['failures'] - 30} additional failure(s); use --out for the complete receipt")
    if output is not None:
        try:
            _receipt_destination(str(output), root)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(receipt, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            print(f"Receipt: {output}")
        except (OSError, ValueError) as exc:
            print(f"Cannot write receipt: {exc}", file=sys.stderr)
            return 2
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
