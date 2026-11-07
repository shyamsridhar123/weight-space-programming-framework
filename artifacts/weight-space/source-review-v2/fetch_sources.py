"""Archive selected public source bytes; never execute the released experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parent
COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
REPO = "RikiyaT/lm_understand-logs"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
ARTICLE = "https://rikiyat.github.io/blog/weight-space-programming/"
INITIAL_PATHS = [
    "README.md",
    "exp1_which_game/README.md",
    "exp2_diffusion/README.md",
    "exp3_language/README.md",
    "exp4_fix_car/README.md",
    "exp5_write_from_scratch/README.md",
    "exp6_go_from_scratch/README.md",
    "exp7_sycophancy/README.md",
    "exp8_math/README.md",
    "exp8_math/gemma_followup/README.md",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(item):
    url, relpath, kind = item
    target = ROOT / relpath
    record = {"url": url, "local_path": relpath, "source_type": kind}
    record["retrieved_at_utc"] = now()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RLCD-source-review/2"})
        with urllib.request.urlopen(req, timeout=25) as response:
            data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError("source exceeds archival size limit; select smaller evidence file")
            record.update({
                "status": "retrieved",
                "http_status": response.status,
                "resolved_url": response.geturl(),
                "content_type": response.headers.get("Content-Type"),
                "etag": response.headers.get("ETag"),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            })
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError("refusing to overwrite an existing source with different bytes")
        target.write_bytes(data)
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--github-tree", action="store_true")
    parser.add_argument("--from-file", type=Path)
    args = parser.parse_args()
    paths = INITIAL_PATHS if args.initial else args.paths
    if args.from_file:
        paths += [p.strip() for p in args.from_file.read_text(encoding="utf-8").splitlines() if p.strip() and not p.startswith("#")]
    requests = [(RAW + p, "release/" + p, "commit_pinned_release_file") for p in paths]
    if args.github_tree:
        requests.append((f"https://github.com/{REPO}/tree/{COMMIT}", "repository-tree.html", "commit_pinned_repository_listing"))
    if args.initial:
        requests += [
            (ARTICLE, "article.html", "primary_author_report"),
            (f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1", "repository-tree.json", "commit_pinned_repository_tree"),
            (f"https://api.github.com/repos/{REPO}/commits/{COMMIT}", "repository-commit.json", "repository_commit_metadata"),
        ]
    manifest_path = ROOT / "provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "schema": "wsp-source-review-provenance/2",
        "created_at_utc": now(),
        "review_local_date": "2026-11-06",
        "local_timezone": "America/Chicago",
        "original_report_publication_date": "2026-11-02",
        "release_repository": REPO,
        "release_commit": COMMIT,
        "independent_experiments_run": False,
        "records": [],
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fetch, requests))
    manifest["records"].extend(results)
    manifest["updated_at_utc"] = now()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for r in results:
        print(json.dumps({k: r[k] for k in ("local_path", "status", "bytes", "error") if k in r}))


if __name__ == "__main__":
    main()
