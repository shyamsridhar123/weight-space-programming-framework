from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


RUN = Path(__file__).resolve().parent
STAGING = RUN.parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def file_record(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def check_package(item):
    package_path, package = item
    name = package.get("name", package_path.removeprefix("node_modules/"))
    version = package["version"]
    filename = name.replace("/", "__") + "-" + version
    metadata_url = "https://registry.npmjs.org/" + urllib.parse.quote(name, safe="@") + "/" + version
    result = {
        "package_path": package_path,
        "name": name,
        "version": version,
        "original_resolved": package.get("resolved"),
        "original_integrity": package.get("integrity"),
        "metadata_url": metadata_url,
        "started_at": now(),
    }
    try:
        request = urllib.request.Request(metadata_url, headers={"User-Agent": "staged-historical-rts-portability-audit"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
            result["metadata_http_status"] = response.status
            result["metadata_final_url"] = response.url
        metadata_path = RUN / "registry-metadata" / (filename + ".json")
        metadata_path.write_bytes(data)
        result["metadata_file"] = str(metadata_path.relative_to(RUN))
        result["metadata_sha256"] = hashlib.sha256(data).hexdigest()
        metadata = json.loads(data)
        result["public_name"] = metadata.get("name")
        result["public_version"] = metadata.get("version")
        result["public_dist"] = metadata.get("dist")
        result["public_dependencies"] = metadata.get("dependencies", {})
        result["public_optional_dependencies"] = metadata.get("optionalDependencies", {})
        result["exact_version_available"] = metadata.get("name") == name and metadata.get("version") == version
        if not result["exact_version_available"]:
            raise ValueError("Public metadata identity mismatch")
        tarball_url = metadata["dist"]["tarball"]
        if urllib.parse.urlparse(tarball_url).hostname != "registry.npmjs.org":
            raise ValueError("Unexpected public tarball host")
        tarball_path = RUN / "public-registry-tarballs" / (filename + ".tgz")
        digests = {algorithm: hashlib.new(algorithm) for algorithm in ("sha1", "sha256", "sha512")}
        size = 0
        with urllib.request.urlopen(tarball_url, timeout=45) as response, tarball_path.open("wb") as target:
            result["tarball_http_status"] = response.status
            for block in iter(lambda: response.read(1024 * 1024), b""):
                target.write(block)
                size += len(block)
                for digest in digests.values():
                    digest.update(block)
        result["tarball_file"] = str(tarball_path.relative_to(RUN))
        result["tarball_bytes"] = size
        result["tarball_sha256"] = digests["sha256"].hexdigest()
        result["tarball_integrities"] = {algorithm: algorithm + "-" + base64.b64encode(digest.digest()).decode("ascii") for algorithm, digest in digests.items()}
        original_integrity = package.get("integrity", "")
        result["matches_original_locked_integrity"] = any(token.split("?", 1)[0] in result["tarball_integrities"].values() for token in original_integrity.split())
        result["matches_public_integrity"] = any(token.split("?", 1)[0] in result["tarball_integrities"].values() for token in metadata["dist"].get("integrity", "").split())
        result["matches_public_shasum"] = metadata["dist"].get("shasum") == digests["sha1"].hexdigest()
    except urllib.error.HTTPError as error:
        result["error"] = str(error)
        result["http_error_status"] = error.code
        result["error_body"] = error.read().decode("utf-8", errors="replace")
        result.setdefault("exact_version_available", False)
    except Exception as error:
        result["error"] = repr(error)
    result["finished_at"] = now()
    return result


def main():
    for subdir in ("registry-metadata", "public-registry-tarballs", "tmp", "npm-cache"):
        (RUN / subdir).mkdir(exist_ok=True)
    baseline = {
        "started_at": now(),
        "staging": str(STAGING),
        "run_directory": str(RUN),
        "platform": platform.platform(),
        "node_modules_existed_before": (STAGING / "node_modules").exists(),
        "files_before": {},
        "versions": {},
    }
    for name in ("package.json", "package-lock.json"):
        baseline["files_before"][name] = file_record(STAGING / name)
        shutil.copyfile(STAGING / name, RUN / ("before-" + name))
    for name, executable in (("node", "node.exe"), ("npm", "npm.cmd"), ("python", "python.exe"), ("git", "git.exe")):
        path = shutil.which(executable)
        args = [path, "--version"]
        completed = subprocess.run(args, cwd=STAGING, capture_output=True, text=True, timeout=20)
        baseline["versions"][name] = {"executable": path, "argv": args, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    if not (RUN / "baseline.json").exists():
        write_json(RUN / "baseline.json", baseline)
    lock = json.loads((STAGING / "package-lock.json").read_text(encoding="utf-8"))
    items = [(path, package) for path, package in lock["packages"].items() if path]
    results = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(check_package, items):
            results.append(result)
            print(json.dumps({key: result.get(key) for key in ("name", "version", "exact_version_available", "matches_original_locked_integrity", "error")}), flush=True)
    receipt = {
        "purpose": "Verify availability and byte identity of exact original locked versions on the public npm registry",
        "command": "python .runs/historical-rts-validation-20261106T2134/check_public_registry.py",
        "finished_at": now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "packages_checked": len(results),
        "exact_versions_available": sum(result.get("exact_version_available", False) for result in results),
        "matching_original_tarballs": sum(result.get("matches_original_locked_integrity", False) for result in results),
        "results": results,
    }
    receipt["all_exact_versions_and_bytes_match"] = all(result.get("exact_version_available") and result.get("matches_original_locked_integrity") and result.get("matches_public_integrity") and result.get("matches_public_shasum") for result in results)
    write_json(RUN / "public-registry-check.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
