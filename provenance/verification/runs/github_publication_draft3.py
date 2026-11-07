"""Local publication driver. Credentials remain in subprocess memory only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = "shyamsridhar123/weight-space-programming-framework"
OWNER = "shyamsridhar123"
OWNER_ID = 117464342
TAG = "v0.2.0-draft.3"
RELEASE_ID = 437312686
PREVIOUS_TAG = "v0.2.0-draft.2"
HISTORICAL_TAGS = {
    "v0.2.0-draft.1": "1f4d48ee14e85f9d1d16f163ad9254a1515c8919",
    PREVIOUS_TAG: "93e8a9b7f2cfbbffab6c7d10f831b2cde4b69176",
}
PREVIOUS_VERIFICATION = ".runs/github-draft2-verify-20261107T041913.json"
WHEEL_SHA256 = "9d018555b7b06a2de57e184227550e9f764160fa668bfe122a0fc3cceaedb89a"
WORKFLOW = ".github/workflows/publication-python.yml"
REQUIRED_STEPS = (
    "Preserve checked-out file bytes",
    "Check out publication",
    "Set up Python 3.12",
    "Verify publication before installing dependencies",
    "Install CPU Torch and the standalone test dependencies",
    "Run the standalone Python suite",
    "Build the wheel",
    "Validate a fresh wheel installation outside the checkout",
    "Verify publication after tests and packaging",
    "Upload new validation receipts and wheel",
)
ASSET_PATHS = (
    "dist/weight_space_programming_framework-0.2.0.dev3-py3-none-any.whl",
    f"dist/weight-space-programming-framework-{TAG}-source.zip",
    "MANIFEST.sha256",
    ".runs/publication-final-verification-draft3-release.json",
    ".runs/staging-verification-draft3-release.json",
    ".runs/source-archive-verification-draft3.json",
    ".runs/hosted-ci-verification-draft3.json",
    ".runs/hosted-ci-job-draft3.log",
)
GH = r"C:\Program Files\GitHub CLI\gh.exe"
GIT = r"C:\Program Files\Git\cmd\git.exe"
GCM = r"C:\Program Files\Git\ucrt64\bin\git-credential-manager.exe"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "create", "push", "refresh", "verify", "status"])
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", GH_PROMPT_DISABLED="1", GH_PAGER="cat", GH_HOST="github.com")
    for name in ("GH_DEBUG", "GH_FORCE_TTY"):
        env.pop(name, None)
    credential = subprocess.run([GCM, "get"], input=f"protocol=https\nhost=github.com\nusername={OWNER}\n\n", text=True, capture_output=True, env=env, timeout=30)
    if credential.returncode:
        raise RuntimeError("Git Credential Manager could not supply existing GitHub authentication")
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if not token:
        raise RuntimeError("No GitHub credential was returned")
    env["GH_TOKEN"] = token
    operations = []

    def run(command: list[str], *, timeout: int = 120, allow_failure: bool = False) -> subprocess.CompletedProcess:
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout, encoding="utf-8", errors="replace")
        operations.append({"command": command, "exit_code": result.returncode})
        if result.returncode and not allow_failure:
            raise RuntimeError(f"Command failed ({result.returncode}): {command[:4]}; subprocess output withheld to protect authentication")
        return result

    def api(path: str) -> dict | list:
        return json.loads(run([GH, "api", "--method", "GET", "--hostname", "github.com", path]).stdout)

    def remote_commit(ref: str) -> str:
        value = api(f"repos/{REPO}/git/ref/{ref}")["object"]
        while value["type"] == "tag":
            value = api(f"repos/{REPO}/git/tags/{value['sha']}")["object"]
        if value["type"] != "commit":
            raise RuntimeError("Remote reference does not resolve to a commit")
        return value["sha"]

    def preserve_historical_tags() -> dict:
        observed = {tag: remote_commit("tags/" + tag) for tag in HISTORICAL_TAGS}
        if observed != HISTORICAL_TAGS:
            raise RuntimeError("A historical draft tag changed unexpectedly")
        return observed

    identity = api("user")
    if identity.get("login") != OWNER or identity.get("id") != OWNER_ID:
        raise RuntimeError("Authenticated GitHub identity differs from the intended owner")
    top = Path(run([GIT, "rev-parse", "--show-toplevel"]).stdout.strip()).resolve()
    if top != ROOT:
        raise RuntimeError("Refusing publication from a parent repository")
    repo_result = run([GH, "api", "--method", "GET", "--hostname", "github.com", "repos/" + REPO], allow_failure=True)
    exists = repo_result.returncode == 0
    if not exists and "404" not in repo_result.stderr:
        raise RuntimeError("Repository availability could not be established")
    repository = json.loads(repo_result.stdout) if exists else None
    result = {"schema": "wspf-github-publication-operation/2", "action": args.action,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "authenticated_login": OWNER, "repository": REPO,
              "repository_exists": exists}

    if args.action == "create":
        if exists:
            raise RuntimeError("Repository already exists; inspect before taking another publication action")
        top = Path(run([GIT, "rev-parse", "--show-toplevel"]).stdout.strip()).resolve()
        if top != ROOT:
            raise RuntimeError("Refusing publication from a parent repository")
        run([GIT, "rev-parse", "--verify", "HEAD"])
        run([GH, "repo", "create", REPO, "--private", "--source", str(ROOT), "--remote", "origin",
             "--disable-wiki", "--description", "Draft generic framework for reading, diagnosing, editing, and creating neural-network weights; specifications, proofs, experiments, and evidence."])
        repository = api("repos/" + REPO)
        result["repository_exists"] = True

    if args.action in {"push", "refresh", "verify", "status"}:
        if (not repository or repository.get("private") is not True or repository.get("full_name") != REPO
                or repository.get("owner", {}).get("login") != OWNER
                or repository.get("owner", {}).get("id") != OWNER_ID):
            raise RuntimeError("The intended private repository is not available")
        remote = run([GIT, "remote", "get-url", "origin"]).stdout.strip()
        if remote not in {f"https://github.com/{REPO}.git", f"https://github.com/{REPO}", f"git@github.com:{REPO}.git"}:
            raise RuntimeError("Local origin does not match the intended repository")
        head = run([GIT, "rev-parse", "HEAD"]).stdout.strip()
        result["local_commit"] = head

    if args.action == "push":
        top = Path(run([GIT, "rev-parse", "--show-toplevel"]).stdout.strip()).resolve()
        local_main = run([GIT, "rev-parse", "--verify", "refs/heads/main"]).stdout.strip()
        local_tag = run([GIT, "rev-parse", "--verify", "refs/tags/" + TAG + "^{}"]).stdout.strip()
        if top != ROOT or local_main != head or local_tag != head:
            raise RuntimeError("Local repository, main, and draft tag must identify the intended checkout before push")
        result["historical_tags_before_push"] = preserve_historical_tags()
        run([GIT, "push", "--atomic", "-u", "origin", "main", "refs/tags/" + TAG], timeout=480)
        result["remote_main"] = remote_commit("heads/main")
        result["remote_tag_commit"] = remote_commit("tags/" + TAG)
        result["preserved_historical_tags"] = preserve_historical_tags()
        if result["remote_main"] != head or result["remote_tag_commit"] != head:
            raise RuntimeError("Remote main or draft tag differs from local publication commit")

    if args.action in {"refresh", "verify"}:
        local_tag = run([GIT, "rev-parse", "--verify", "refs/tags/" + TAG + "^{}"]).stdout.strip()
        if local_tag != head:
            raise RuntimeError("The local draft.3 tag must identify HEAD")
        result["preserved_historical_tags"] = preserve_historical_tags()
        for ref in ("heads/main", "tags/" + TAG):
            if remote_commit(ref) != head:
                raise RuntimeError("Remote main and draft tag must identify the intended commit")
        assets = list(ASSET_PATHS)
        new_assets = {Path(relative).name: ROOT / relative for relative in assets}
        for name, path in new_assets.items():
            if not path.is_file():
                raise RuntimeError(f"Missing release asset: {name}")
        if sha(ROOT / ASSET_PATHS[0]) != WHEEL_SHA256:
            raise RuntimeError("Release wheel differs from the fresh draft.3 validation")
        ci = json.loads((ROOT / ".runs/hosted-ci-verification-draft3.json").read_text(encoding="utf-8"))
        if (ci.get("verification_passed") is not True or ci.get("verification_failures") != []
                or ci.get("head_sha") != head or ci.get("status") != "completed" or ci.get("conclusion") != "success"
                or ci.get("repository") != REPO or ci.get("workflow_path") != WORKFLOW
                or ci.get("workflow_name") != "Publication and Python"
                or ci.get("event") != "push" or ci.get("head_branch") != "main"
                or ci.get("required_steps") != list(REQUIRED_STEPS)):
            raise RuntimeError("Successful exact-commit hosted verification, including every required step, is required")
        if ci.get("log") != binding(ROOT / ".runs/hosted-ci-job-draft3.log"):
            raise RuntimeError("Hosted verification does not bind the intended job log bytes")
        if ci.get("secrets_pattern_scan", {}).get("clean_after_redaction") is not True:
            raise RuntimeError("Hosted verification does not record a clean exported log")
        hosted = api(f"repos/{REPO}/actions/runs/{ci['run_id']}")
        for key, expected in (("head_sha", head), ("status", "completed"), ("conclusion", "success"),
                              ("path", WORKFLOW), ("name", "Publication and Python"),
                              ("event", "push"), ("head_branch", "main"), ("run_attempt", ci["run_attempt"])):
            if hosted.get(key) != expected:
                raise RuntimeError("Hosted workflow state no longer matches the successful verification receipt")
        jobs_data = api(f"repos/{REPO}/actions/runs/{ci['run_id']}/attempts/{ci['run_attempt']}/jobs?per_page=100")
        jobs = jobs_data.get("jobs", [])
        if (not jobs or jobs_data.get("total_count") != len(jobs)
                or sorted(job["id"] for job in jobs) != sorted(ci.get("job_ids", []))):
            raise RuntimeError("Hosted jobs no longer match the complete verified attempt")
        for job in jobs:
            if (job.get("run_id") != ci["run_id"] or job.get("head_sha") != head
                    or job.get("run_attempt", ci["run_attempt"]) != ci["run_attempt"]
                    or job.get("status") != "completed" or job.get("conclusion") != "success"):
                raise RuntimeError("A hosted job does not match the successful exact-commit attempt")
        publication_jobs = [job for job in jobs if job.get("name") == "publication-python"]
        if len(publication_jobs) != 1 or publication_jobs[0]["id"] != ci.get("job_id"):
            raise RuntimeError("Hosted publication job differs from the monitored job")
        for name in REQUIRED_STEPS:
            matches = [step for step in publication_jobs[0].get("steps", []) if step.get("name") == name]
            if len(matches) != 1 or matches[0].get("status") != "completed" or matches[0].get("conclusion") != "success":
                raise RuntimeError(f"Required hosted step did not complete successfully: {name}")
        archive = json.loads((ROOT / ".runs/source-archive-verification-draft3.json").read_text(encoding="utf-8"))
        local_tree = run([GIT, "rev-parse", "HEAD^{tree}"]).stdout.strip()
        if (archive.get("commit") != head or archive.get("tag") != TAG or archive.get("tree") != local_tree
                or archive.get("python_distribution") != "0.2.0.dev3"
                or archive.get("exact_git_tree_membership") is not True
                or archive.get("all_archive_file_bytes_match_commit_and_manifest") is not True
                or archive.get("staged_tree_matches_committed_tree") is not True):
            raise RuntimeError("Source archive receipt does not bind the intended committed draft.3 tree")
        expected_bindings = {
            "source_zip": f"dist/weight-space-programming-framework-{TAG}-source.zip",
            "manifest": "MANIFEST.sha256",
            "wheel": ASSET_PATHS[0],
            "publication_verification": ".runs/publication-final-verification-draft3-release.json",
            "git_staging_verification": ".runs/staging-verification-draft3-release.json",
            "wheel_build_verification": ".runs/wheel-build-draft3.json",
            "wheel_build_log": ".runs/wheel-build-draft3.log",
            "wheel_install_smoke_verification": ".runs/wheel-install-smoke-draft3.json",
            "wheel_install_smoke_log": ".runs/wheel-install-smoke-draft3.log",
        }
        for key, relative in expected_bindings.items():
            if archive.get(key) != binding(ROOT / relative):
                raise RuntimeError(f"Source archive receipt has a stale or unexpected asset binding: {key}")
        final = json.loads((ROOT / expected_bindings["publication_verification"]).read_text(encoding="utf-8"))
        staging = json.loads((ROOT / expected_bindings["git_staging_verification"]).read_text(encoding="utf-8"))
        if (final.get("ok") is not True or final.get("failures") != []
                or staging.get("staged_tree") != local_tree
                or staging.get("manifest") != archive["manifest"]
                or staging.get("publication_verification") != archive["publication_verification"]):
            raise RuntimeError("Final verification and staging evidence do not bind this release")
        result["hosted_ci_verification"] = binding(ROOT / ".runs/hosted-ci-verification-draft3.json")
        result["source_archive_verification"] = binding(ROOT / ".runs/source-archive-verification-draft3.json")
        prepared_assets = {name: binding(path) for name, path in new_assets.items()}
        notes_path = ROOT / f"releases/{TAG}.md"
        committed_notes = run([GIT, "show", f"{head}:releases/{TAG}.md"]).stdout
        if notes_path.read_text(encoding="utf-8") != committed_notes:
            raise RuntimeError("Prepared release notes differ from the committed publication")
        notes_binding = binding(notes_path)

    if args.action == "refresh":
        release = api(f"repos/{REPO}/releases/{RELEASE_ID}")
        if release["id"] != RELEASE_ID or release["tag_name"] not in {PREVIOUS_TAG, TAG}:
            raise RuntimeError("Unexpected release identity or tag; refusing draft update")
        if not release["draft"] or not release["prerelease"] or release["published_at"] is not None:
            raise RuntimeError("Only the existing unpublished prerelease may be updated")
        previous = json.loads((ROOT / PREVIOUS_VERIFICATION).read_text(encoding="utf-8"))
        if (previous.get("repository") != REPO or previous.get("local_commit") != HISTORICAL_TAGS[PREVIOUS_TAG]
                or previous.get("all_release_asset_sha256_verified") is not True
                or previous["release"]["id"] != RELEASE_ID
                or previous["release"]["tag_name"] != PREVIOUS_TAG
                or previous["release"]["target_commitish"] != HISTORICAL_TAGS[PREVIOUS_TAG]):
            raise RuntimeError("Previous release verification does not identify the preserved draft.2 publication")
        old_asset_bindings = {asset["name"]: asset for asset in previous["release"]["assets"]}
        for asset in release["assets"]:
            old = old_asset_bindings.get(asset["name"])
            is_known_original = (old is not None and old["id"] == asset["id"]
                                 and old["digest"] == asset["digest"]
                                 and old["size"] == asset["size"] and asset["state"] == "uploaded")
            target = new_assets.get(asset["name"])
            is_exact_current = (target is not None and target.is_file() and asset["state"] == "uploaded"
                                and asset["size"] == target.stat().st_size and asset["digest"] == "sha256:"+sha(target))
            if not (is_known_original or is_exact_current):
                raise RuntimeError("Unexpected existing asset identity; refusing to replace or remove it")
        backup = ROOT / ".runs" / ("github-draft3-before-refresh-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
        with backup.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(release, stream, indent=2)
            stream.write("\n")
        if run([GIT, "rev-parse", "HEAD"]).stdout.strip() != head:
            raise RuntimeError("Local HEAD changed before draft refresh")
        if (prepared_assets != {name: binding(path) for name, path in new_assets.items()}
                or binding(notes_path) != notes_binding):
            raise RuntimeError("Prepared release assets or notes changed before draft refresh")
        run([GH, "release", "edit", release["tag_name"], "--repo", REPO, "--tag", TAG,
             "--draft=true", "--prerelease=true", "--latest=false", "--verify-tag", "--target", head,
             "--title", "Weight Space Programming Framework v0.2.0-draft.3",
             "--notes-file", "releases/v0.2.0-draft.3.md"], timeout=120)
        run([GH, "release", "upload", TAG, *assets, "--repo", REPO, "--clobber"], timeout=480)
        uploaded = api(f"repos/{REPO}/releases/{RELEASE_ID}")
        if (uploaded["id"] != RELEASE_ID or uploaded["tag_name"] != TAG or uploaded["target_commitish"] != head
                or uploaded["draft"] is not True or uploaded["prerelease"] is not True or uploaded["published_at"] is not None):
            raise RuntimeError("Updated release identity or unpublished state differs; old assets retained")
        uploaded_by_name = {a["name"]:a for a in uploaded["assets"]}
        for name,path in new_assets.items():
            asset=uploaded_by_name.get(name)
            expected = prepared_assets[name]
            if (asset is None or asset["state"] != "uploaded" or asset["size"] != expected["bytes"]
                    or asset["digest"] != "sha256:" + expected["sha256"] or binding(path) != expected):
                raise RuntimeError(f"Uploaded asset did not verify; old assets retained: {name}")
        for asset in uploaded["assets"]:
            if asset["name"] not in new_assets:
                old=old_asset_bindings.get(asset["name"])
                if (old is None or old["id"] != asset["id"] or old["digest"] != asset["digest"]
                        or old["size"] != asset["size"] or asset["state"] != "uploaded"):
                    raise RuntimeError("Unexpected stale asset encountered during guarded cleanup")
                run([GH, "api", "--method", "DELETE", "--hostname", "github.com", f"repos/{REPO}/releases/assets/{asset['id']}"])
        result["updated_existing_release_id"] = RELEASE_ID
        result["preserved_historical_tags"] = preserve_historical_tags()
        result["previous_asset_verification"] = binding(ROOT / PREVIOUS_VERIFICATION)
        result["release_metadata_backup"] = binding(backup)
        result["local_assets"] = [{"name": name, **expected} for name, expected in prepared_assets.items()]

    if args.action in {"refresh", "verify"}:
        release_list = api(f"repos/{REPO}/releases?per_page=100")
        matching = [r for r in release_list if r["tag_name"] == TAG]
        if len(matching) != 1:
            raise RuntimeError("Expected exactly one draft release")
        release = matching[0]
        if (release["id"] != RELEASE_ID or release["target_commitish"] != head
                or release["draft"] is not True or release["prerelease"] is not True or release["published_at"] is not None):
            raise RuntimeError("Release does not have the requested unpublished draft/prerelease status")
        result["release"] = {k: release.get(k) for k in ("id", "html_url", "tag_name", "target_commitish", "draft", "prerelease", "published_at", "created_at")}
        result["release"]["assets"] = [{k: a.get(k) for k in ("id", "name", "size", "state", "digest", "browser_download_url")} for a in release["assets"]]
        result["release_notes_match"] = release["body"].replace("\r\n", "\n") == committed_notes
        if not result["release_notes_match"]:
            raise RuntimeError("Remote release notes differ from the prepared draft")
        if (len(release["assets"]) != len(new_assets)
                or {asset["name"] for asset in release["assets"]} != set(new_assets)):
            raise RuntimeError("Release asset membership differs from the prepared asset set")
        for asset in release["assets"]:
            local = new_assets[asset["name"]]
            expected = prepared_assets[asset["name"]]
            if (asset["size"] != expected["bytes"] or asset["state"] != "uploaded"
                    or asset.get("digest") != "sha256:" + expected["sha256"] or binding(local) != expected):
                raise RuntimeError(f"Release asset server identity differs: {asset['name']}")
        result["all_release_asset_sha256_verified"] = True
        if binding(notes_path) != notes_binding:
            raise RuntimeError("Local release notes changed during publication")

    if args.action == "verify":
        result["preserved_historical_tags"] = preserve_historical_tags()
        result["remote_main"] = remote_commit("heads/main")
        result["remote_tag_commit"] = remote_commit("tags/" + TAG)
        if result["remote_main"] != head or result["remote_tag_commit"] != head:
            raise RuntimeError("Remote branch/tag do not identify the publication commit")
        tree = api(f"repos/{REPO}/git/trees/{head}?recursive=1")
        if tree.get("truncated"):
            raise RuntimeError("Remote Git tree is truncated")
        local_entries = {}
        local_tree = run([GIT, "ls-tree", "-rz", "--full-tree", "HEAD"]).stdout
        for entry in local_tree.split("\0"):
            if not entry:
                continue
            metadata, name = entry.split("\t", 1)
            mode, kind, object_id = metadata.split()
            local_entries[name] = (mode, kind, object_id)
        remote_entries = {e["path"]: (e["mode"],e["type"],e["sha"]) for e in tree["tree"] if e["type"] != "tree"}
        if remote_entries != local_entries:
            raise RuntimeError("Remote tree paths, modes, or object identities differ from local publication")
        result["remote_tree_matches_local_commit"] = True
        result["local_tree"] = run([GIT, "rev-parse", "HEAD^{tree}"]).stdout.strip()
        result["manifest_sha256"] = sha(ROOT/"MANIFEST.sha256")

    if args.action in {"status", "verify"}:
        workflows = api(f"repos/{REPO}/actions/runs?head_sha={head}&per_page=20")
        result["workflow_runs"] = [{k:r.get(k) for k in ("id", "name", "head_sha", "status", "conclusion", "html_url", "run_attempt", "created_at", "updated_at")} for r in workflows["workflow_runs"]]

    if repository:
        result["repository_metadata"] = {k: repository.get(k) for k in ("full_name", "private", "visibility", "html_url", "default_branch")}
        if (repository.get("private") is not True or repository.get("full_name") != REPO
                or repository.get("owner", {}).get("login") != OWNER
                or repository.get("owner", {}).get("id") != OWNER_ID):
            raise RuntimeError("Repository owner or private visibility differs")
    if args.action in {"push", "refresh", "verify", "status"} and run([GIT, "rev-parse", "HEAD"]).stdout.strip() != head:
        raise RuntimeError("Local HEAD changed during the publication operation")
    result["operations"] = operations
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = ROOT/".runs"/("github-draft3-" + args.action + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    result["local_receipt"] = str(path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
