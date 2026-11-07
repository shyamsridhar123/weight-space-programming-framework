from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


RUN = Path(__file__).resolve().parent
STAGING = RUN.parent.parent


def load(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def record(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    baseline = load("baseline.json")
    installation = load("npm-ci-original-lock.json")
    tests = load("historical-rts-tests.json")
    local = load("local-dependency-resolution.stdout.log")
    lock = json.loads((STAGING / "package-lock.json").read_text(encoding="utf-8"))
    installed_lock = json.loads((STAGING / "node_modules" / ".package-lock.json").read_text(encoding="utf-8"))
    cache_entries = {}
    for index_file in (RUN / "npm-cache" / "_cacache" / "index-v5").rglob("*"):
        if not index_file.is_file():
            continue
        for line in index_file.read_text(encoding="utf-8").splitlines():
            if not line or "\t" not in line:
                continue
            entry = json.loads(line.split("\t", 1)[1])
            if entry.get("key", "").startswith("make-fetch-happen:request-cache:"):
                url = entry["key"].removeprefix("make-fetch-happen:request-cache:")
                cache_entries[url] = entry
    installed_sources = []
    for package_path, package in installed_lock["packages"].items():
        original = lock["packages"][package_path]
        source = {
            "package_path": package_path,
            "version": package["version"],
            "resolved": package.get("resolved"),
            "registry_host": urlparse(package.get("resolved", "")).hostname,
            "integrity": package.get("integrity"),
            "installed_lock_matches_staged_lock": all(package.get(key) == original.get(key) for key in ("version", "resolved", "integrity")),
        }
        cached = cache_entries.get(package.get("resolved"))
        source["cache_lookup_method"] = "locked_resolved_url"
        if cached is None:
            candidates = [entry for entry in cache_entries.values() if entry.get("integrity") == package.get("integrity")]
            if len(candidates) == 1:
                cached = candidates[0]
                source["cache_lookup_method"] = "exact_locked_integrity_for_registry_blob_redirect"
        source["download_record_in_fresh_run_cache"] = cached is not None
        if cached is not None:
            download_url = urlparse(cached["key"].removeprefix("make-fetch-happen:request-cache:"))
            cache_integrity = cached["integrity"].split()[0]
            algorithm, encoded_digest = cache_integrity.split("-", 1)
            hexadecimal = base64.b64decode(encoded_digest).hex()
            content_file = RUN / "npm-cache" / "_cacache" / "content-v2" / algorithm / hexadecimal[:2] / hexadecimal[2:4] / hexadecimal[4:]
            data = content_file.read_bytes()
            original_algorithm, original_digest = package["integrity"].split()[0].split("-", 1)
            computed_digest = base64.b64encode(hashlib.new(original_algorithm, data).digest()).decode("ascii")
            source.update({
                "cache_content_file": str(content_file.relative_to(RUN)),
                "cache_content_bytes": len(data),
                "cache_content_sha256": hashlib.sha256(data).hexdigest(),
                "cache_integrity": cache_integrity,
                "downloaded_tarball_matches_locked_integrity": computed_digest == original_digest,
                "download_time_unix_ms": cached.get("time"),
                "download_host": download_url.hostname,
                "download_url_without_query": download_url._replace(query="", fragment="").geturl(),
            })
        installed_sources.append(source)
    output = (RUN / tests["stdout"]["file"]).read_text(encoding="utf-8")
    counts = {}
    for key in ("tests", "suites", "pass", "fail", "cancelled", "skipped", "todo"):
        matches = re.findall(r"^\S+ " + key + r" (\d+)\s*$", output, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one test-summary value for {key}; got {matches!r}")
        counts[key] = int(matches[0])
    duration = re.search(r"^\S+ duration_ms ([\d.]+)\s*$", output, flags=re.MULTILINE)
    if duration:
        counts["reported_duration_ms"] = float(duration.group(1))
    files_after = {name: record(STAGING / name) for name in ("package.json", "package-lock.json")}
    command_files = ["npm-ci-original-lock.json", "local-dependency-resolution.json", "historical-rts-tests.json", "proof-shell-syntax.json", "git-bash-version.json", "npm-public-tsx-view.json", "public-registry-curl.json"]
    receipt = {
        "receipt_kind": "new_optional_historical_rts_runtime_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "staging": str(STAGING),
        "run_directory_relative_to_staging": str(RUN.relative_to(STAGING)),
        "scope": "Install staged locked dependencies, confirm local resolution, run only the three retained test files, and syntax-check the proof shell wrapper; no historical study runner commands executed.",
        "versions": baseline["versions"],
        "node_modules_existed_before": baseline["node_modules_existed_before"],
        "commands": {name.removesuffix(".json"): {"receipt": name, "argv": load(name)["argv"], "exit_code": load(name)["exit_code"], "stdout_log": load(name)["stdout"]["file"], "stderr_log": load(name)["stderr"]["file"], "receipt_file": record(RUN / name)} for name in command_files},
        "install_exit_code": installation["exit_code"],
        "test_exit_code": tests["exit_code"],
        "test_counts": counts,
        "test_files": ["tests/weight-space-empires.test.ts", "tests/empires-weights.test.ts", "tests/empires-weight-coach.test.ts"],
        "local_dependency_resolution": local,
        "installed_package_sources": installed_sources,
        "all_installed_sources_verified_against_locked_tarball_integrity": all(source["installed_lock_matches_staged_lock"] and source.get("downloaded_tarball_matches_locked_integrity") for source in installed_sources),
        "package_files_before": baseline["files_before"],
        "package_files_after": files_after,
        "package_files_unchanged": baseline["files_before"] == files_after,
        "temporary_file_directory": str(RUN / "tmp"),
        "npm_cache_directory": str(RUN / "npm-cache"),
        "public_registry_portability": {
            "verified": False,
            "reason": "Direct official npm registry requests failed at TLS handshake in Python urllib, npm, and native curl; metadata availability and public tarball byte identity could not be assessed.",
            "public_registry_receipts": ["public-registry-check.json", "npm-public-tsx-view.json", "public-registry-curl.json", "network-config-sanitized.json"],
            "installed_sources_are_public_npm_verified": False,
            "lock_changed_or_dependencies_upgraded": False,
        },
        "notes": [
            "npm ci emitted an esbuild install-script allowScripts warning; the installed native Windows optional dependency worked in a direct local esbuild transform and all retained tests passed.",
            "NODE_PATH was explicitly empty for the local-resolution check and retained test command; the staged npm launcher points directly to staged node_modules/tsx/dist/cli.mjs.",
            "The test suite creates and removes its own adapter fixture and receipt files under the run temporary directory; those are new test execution artifacts, not regenerated historical studies.",
        ],
    }
    assert receipt["package_files_unchanged"]
    assert receipt["all_installed_sources_verified_against_locked_tarball_integrity"]
    assert receipt["install_exit_code"] == receipt["test_exit_code"] == 0
    assert counts["tests"] == counts["pass"] == 61
    assert all(counts[key] == 0 for key in ("fail", "cancelled", "skipped", "todo"))
    (RUN / "validation-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "receipt": str((RUN / "validation-receipt.json").relative_to(STAGING)),
        "receipt_file": record(RUN / "validation-receipt.json"),
        "test_counts": counts,
        "installed_package_sources": installed_sources,
        "package_files_unchanged": receipt["package_files_unchanged"],
        "public_registry_verified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
