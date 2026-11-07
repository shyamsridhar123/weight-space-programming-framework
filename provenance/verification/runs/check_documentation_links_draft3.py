#!/usr/bin/env python3
"""Check local destinations and heading fragments in the live publication docs.

This supplements scripts/verify_publication.py without changing that checker.
It reads publication files and writes only the requested receipt under .runs/.
No framework modules, examples, tests, proof builds, or network calls are run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import runpy
import unicodedata
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PREFIXES = (
    "artifacts/",
    "provenance/original-documents/",
    "provenance/previous-publications/",
)
MARKDOWN_SUFFIXES = {".md", ".markdown"}


class ExplicitAnchors(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if value is not None and (key == "id" or (tag == "a" and key == "name")):
                self.anchors.add(value)


def heading_slug(value: str) -> str:
    """Produce GitHub-style slugs for the headings used in this publication."""
    value = re.sub(r"\s+#+\s*$", "", value.strip())
    value = re.sub(r"!?\[([^]]*)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]*>", "", value)
    value = re.sub(r"[`*~]", "", value)
    value = html.unescape(value).lower()
    value = "".join(
        char for char in value
        if char in "-_" or char.isspace()
        or unicodedata.category(char)[0] in {"L", "N", "M"}
    )
    return re.sub(r"\s", "-", value)


def inspect(root: Path, pending_paths: set[str]) -> dict:
    checker = runpy.run_path(
        str(root / "scripts/verify_publication.py"), run_name="documentation_link_review"
    )
    publication = checker["publication_files"](root)
    published = set(publication)
    if (root / "MANIFEST.sha256").is_file():
        published.add("MANIFEST.sha256")
    directories = {"."}
    for relative in published:
        directories.update(parent.as_posix() for parent in Path(relative).parents)
    all_targets = published | directories
    casefolded = {relative.casefold(): relative for relative in all_targets}
    live = [
        relative for relative in publication
        if Path(relative).suffix.lower() in MARKDOWN_SUFFIXES
        and not relative.startswith(HISTORICAL_PREFIXES)
    ]
    failures: list[dict] = []
    pending: list[dict] = []
    verified: list[dict] = []
    external: list[dict] = []
    snapshots: dict[str, dict] = {}
    anchor_cache: dict[str, tuple[set[str], set[str]]] = {}
    supplemental: list[dict] = []

    def read(relative: str) -> str:
        raw = checker["_repository_file"](root, relative).read_bytes()
        snapshots[relative] = {
            "path": relative,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        return raw.decode("utf-8")

    def anchors(relative: str) -> tuple[set[str], set[str]]:
        if relative in anchor_cache:
            return anchor_cache[relative]
        raw = read(relative)
        markdown = Path(relative).suffix.lower() in MARKDOWN_SUFFIXES
        visible = checker["_visible_markdown"](raw, relative, failures) if markdown else raw
        generated: set[str] = set()
        occurrences: dict[str, int] = {}

        def add_heading(value: str) -> None:
            original = heading_slug(value)
            candidate = original
            while candidate in occurrences:
                occurrences[original] += 1
                candidate = f"{original}-{occurrences[original]}"
            occurrences[candidate] = 0
            generated.add(candidate)

        if markdown:
            previous = ""
            for line in visible.splitlines():
                match = re.match(r"^ {0,3}#{1,6}(?:[ \t]+(.*)|[ \t]*)$", line)
                if match:
                    add_heading(match[1] or "")
                    previous = ""
                    continue
                if previous and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", line):
                    add_heading(previous)
                    previous = ""
                    continue
                previous = line.strip() if line and not line.startswith(("    ", "\t")) else ""
            # Keep inline code text in heading slugs, but do not parse tags inside it.
            visible = checker["_inline_targets"](visible)[1]
        parser = ExplicitAnchors()
        parser.feed(visible)
        parser.close()
        anchor_cache[relative] = generated, parser.anchors
        return anchor_cache[relative]

    for source in live:
        text = read(source)
        visible = checker["_visible_markdown"](text, source, failures)
        targets = set(checker["_markdown_targets"](visible))
        # A second, line-scoped pass detects any links hidden by accidental
        # cross-paragraph backtick pairing in the full-document parser.
        for number, line in enumerate(visible.splitlines(), 1):
            for target, _, is_image in checker["_markdown_targets"](line):
                item = (target, number, is_image)
                if item not in targets:
                    supplemental.append({"path": source, "line": number, "target": target})
                    targets.add(item)
        for raw_target, number, is_image in sorted(targets, key=lambda item: (item[1], item[0])):
            record = {"path": source, "line": number, "target": raw_target}
            target = html.unescape(checker["MARKDOWN_ESCAPE"].sub(r"\1", raw_target))
            try:
                if re.match(r"^[A-Za-z]:", target):
                    raise ValueError("absolute Windows paths are not portable publication links")
                if checker["URI_SCHEME"].match(target) or target.startswith("//"):
                    external.append({**record, "status": "unchecked", "reason": "no network requests"})
                    continue
                split = urlsplit(target)
                path = unquote(split.path, encoding="utf-8", errors="strict")
                fragment = unquote(split.fragment, encoding="utf-8", errors="strict")
                if path.startswith("/") or "\\" in path or ":" in path or "\0" in path:
                    raise ValueError("local links must use relative repository paths")
                candidate = root / Path(source).parent / path if path else root / source
                lexical = Path(os.path.abspath(candidate))
                if not lexical.is_relative_to(root) or not candidate.resolve().is_relative_to(root):
                    raise ValueError("local link escapes the publication root")
                normalized = lexical.relative_to(root).as_posix()
                record["resolved_path"] = normalized
                if fragment:
                    record["fragment"] = fragment
                if normalized not in all_targets:
                    if normalized in pending_paths and not lexical.exists():
                        pending.append({**record, "status": "pending_assembly"})
                        continue
                    actual_case = casefolded.get(normalized.casefold())
                    if actual_case:
                        raise ValueError(f"path case differs from the publication: {actual_case}")
                    raise ValueError("target is absent from the published files or directories")
                if normalized in published:
                    checker["_repository_file"](root, normalized)
                    kind = "file"
                elif not is_image and lexical.is_dir():
                    kind = "directory"
                else:
                    raise ValueError("image destination must be a published file")
                if fragment:
                    fragment_path = normalized
                    if kind == "directory":
                        readmes = [
                            value for value in published
                            if Path(value).parent.as_posix() == normalized
                            and Path(value).stem.lower() == "readme"
                            and Path(value).suffix.lower() in MARKDOWN_SUFFIXES
                        ]
                        if len(readmes) != 1:
                            raise ValueError("directory fragment needs one unambiguous Markdown README")
                        fragment_path = readmes[0]
                        record["fragment_document"] = fragment_path
                    line_range = re.fullmatch(r"L([1-9][0-9]*)(?:-L?([1-9][0-9]*))?", fragment)
                    if line_range and Path(fragment_path).suffix.lower() not in MARKDOWN_SUFFIXES:
                        start = int(line_range[1])
                        end = int(line_range[2] or line_range[1])
                        length = len(read(fragment_path).splitlines())
                        if not 1 <= start <= end <= length:
                            raise ValueError(f"line fragment is outside the file's {length} lines")
                        kind = "source_lines"
                    elif Path(fragment_path).suffix.lower() in MARKDOWN_SUFFIXES | {".html", ".htm", ".svg"}:
                        generated, explicit = anchors(fragment_path)
                        if fragment in generated:
                            kind = "markdown_heading"
                        elif fragment in explicit:
                            kind = "explicit_anchor"
                        else:
                            raise ValueError("fragment does not match a heading or explicit HTML anchor")
                    else:
                        raise ValueError("fragment type is not supported by this local checker")
                verified.append({**record, "verified_as": kind})
            except (OSError, ValueError, UnicodeError) as exc:
                failures.append({**record, "code": "local_documentation_link", "message": str(exc)})

    return {
        "schema": "weight-space-programming-framework.documentation-links.v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "status": "fail" if failures else ("pending_assembly" if pending else "pass"),
        "ok": not failures and not pending,
        "existing_destinations_ok": not failures,
        "policy": {
            "live_sources": "all publishable .md and .markdown outside historical subtrees",
            "historical_source_exclusions": list(HISTORICAL_PREFIXES),
            "preserved_md_txt": "not live Markdown; still checked when linked as a destination",
            "publication_selection": "scripts/verify_publication.py:publication_files",
            "local_destinations": "exact-case publication file or directory; repository escape rejected",
            "fragments": "GitHub-style ATX/Setext heading slugs, duplicate suffixes, explicit id/name anchors; source line ranges when applicable",
            "external_destinations": "unchecked; no HTTP, DNS, authentication, or release-asset validation",
            "execution_scope": "documentation inspection only; no examples, tests, studies, or proof builds",
            "pending_paths": sorted(pending_paths),
        },
        "counts": {
            "live_markdown_files": len(live),
            "local_destinations_verified": len(verified),
            "fragments_verified": sum("fragment" in value for value in verified),
            "external_destinations_unchecked": len(external),
            "supplemental_parser_candidates": len(supplemental),
            "pending_assembly_destinations": len(pending),
            "failures": len(failures),
        },
        "live_markdown_files": live,
        "read_file_snapshots": [snapshots[key] for key in sorted(snapshots)],
        "verified_local_destinations": verified,
        "external_destinations_unchecked": external,
        "supplemental_parser_candidates": supplemental,
        "pending_assembly_destinations": pending,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".runs/documentation-links-draft3.json")
    parser.add_argument("--pending-path", action="append", default=[])
    args = parser.parse_args()
    output = Path(args.out)
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(ROOT / ".runs"):
        parser.error("receipt output must stay under this publication's .runs directory")
    receipt = inspect(ROOT, set(args.pending_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({
        "status": receipt["status"],
        "counts": receipt["counts"],
        "pending_assembly_destinations": receipt["pending_assembly_destinations"],
        "failures": receipt["failures"],
        "supplemental_parser_candidates": receipt["supplemental_parser_candidates"],
        "receipt": str(output),
    }, indent=2))
    return 0 if receipt["ok"] else (2 if receipt["failures"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
