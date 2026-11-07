"""Review current guide destinations and fragments without editing documents."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
FOCUS = {ROOT / "README.md", ROOT / "docs/weight-space/v2/README.md",
         ROOT / "docs/weight-space/v2/worked-example.md"}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def without_fences(text: str) -> str:
    return re.sub(r"(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[^\n]*$", "", text)


def anchors(text: str) -> set[str]:
    text = without_fences(text)
    found = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", text))
    seen = {}
    for match in re.finditer(r"(?m)^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", text):
        title = html.unescape(re.sub(r"<[^>]*>", "", match.group(1)))
        title = re.sub(r"\[([^]]*)\]\([^)]*\)", r"\1", title)
        slug = re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")
        suffix = seen.get(slug, 0)
        seen[slug] = suffix + 1
        found.add(slug if suffix == 0 else f"{slug}-{suffix}")
    return found


def main() -> None:
    sources = sorted({*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md"),
                      *(ROOT / "formal").rglob("README.md"), *(ROOT / "releases").glob("*.md")})
    content = {path: path.read_bytes() for path in sources}
    receipt = {
        "schema": "guide-navigation-review/1",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Outbound local links from root README, v2 README, and worked-example; inbound links to those guides from top-level, docs, formal READMEs, and release Markdown. Archived artifacts and external URLs were excluded.",
        "reviewer_sha256": digest(Path(__file__).read_bytes()),
        "source_snapshots": {p.relative_to(ROOT).as_posix(): digest(b) for p, b in content.items()},
        "fragment_checks": [], "missing_paths": [], "unresolved_fragments": [],
    }
    for source, raw in content.items():
        text = without_fences(raw.decode("utf-8"))
        for match in re.finditer(r"!?\[[^\]]*\]\(([^\n)]+)\)", text):
            url = match.group(1).strip().split(' "', 1)[0].strip("<>")
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc:
                continue
            target = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source
            if source not in FOCUS and target not in FOCUS:
                continue
            record = {"source": source.relative_to(ROOT).as_posix(), "destination": url}
            if not target.is_file():
                receipt["missing_paths"].append(record)
                continue
            if not parsed.fragment:
                continue
            raw_target = content.get(target)
            if raw_target is None:
                raw_target = target.read_bytes()
            target_text = raw_target.decode("utf-8")
            target_anchors = anchors(target_text)
            ok = unquote(parsed.fragment) in target_anchors
            record.update(target_sha256=digest(raw_target), resolved=ok)
            receipt["fragment_checks"].append(record)
            if not ok:
                receipt["unresolved_fragments"].append(record)
    receipt["sources_changed_during_review"] = [
        p.relative_to(ROOT).as_posix() for p, b in content.items() if p.read_bytes() != b
    ]
    receipt["fragment_status"] = "failed" if receipt["unresolved_fragments"] else "passed"
    output = ROOT / ".runs/navigation-draft3-review.json"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"receipt": output.relative_to(ROOT).as_posix(),
                      "fragment_status": receipt["fragment_status"],
                      "missing_paths": receipt["missing_paths"],
                      "unresolved_fragments": receipt["unresolved_fragments"],
                      "sources_changed_during_review": receipt["sources_changed_during_review"]}))


if __name__ == "__main__":
    main()
