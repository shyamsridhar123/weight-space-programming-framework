"""Read one exact commit's hosted CI state; run again while exit code is 2.

No remote calls occur when this module is imported or --help is requested.
Only GET requests are used. Completed observations are preserved under new
run/attempt names; the canonical draft.3 assets are written only after success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = "shyamsridhar123/weight-space-programming-framework"
OWNER = "shyamsridhar123"
OWNER_ID = 117464342
WORKFLOW = ".github/workflows/publication-python.yml"
WORKFLOW_NAME = "Publication and Python"
GH = r"C:\Program Files\GitHub CLI\gh.exe"
GIT = r"C:\Program Files\Git\cmd\git.exe"
GCM = r"C:\Program Files\Git\ucrt64\bin\git-credential-manager.exe"
RECEIPT = ROOT / ".runs/hosted-ci-verification-draft3.json"
PROGRESS = ROOT / ".runs/hosted-ci-progress-draft3.json"
LOG = ROOT / ".runs/hosted-ci-job-draft3.log"
PYTEST_STEP = "Run the standalone Python suite"
REQUIRED_STEPS = (
    "Preserve checked-out file bytes",
    "Check out publication",
    "Set up Python 3.12",
    "Verify publication before installing dependencies",
    "Install CPU Torch and the standalone test dependencies",
    PYTEST_STEP,
    "Build the wheel",
    "Validate a fresh wheel installation outside the checkout",
    "Verify publication after tests and packaging",
    "Upload new validation receipts and wheel",
)
ANSI_BYTES = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")
SECRET_PATTERNS = {
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "api_key_prefix": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "jwt": re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----|\Z)"),
    "authorization_value": re.compile(rb"\bauthorization\s*[:=]\s*(?:basic|bearer)\s+[A-Za-z0-9._~+/=-]{8,}", re.I),
    "credential_assignment": re.compile(rb"\b(?:password|passwd|secret|access[_-]?token|api[_-]?key)\s*[:=]\s*[\"']?[A-Za-z0-9_+./=-]{16,}", re.I),
    "credential_in_url": re.compile(rb"\bhttps?://[^/\s:@]+:[^/\s@]+@[A-Za-z0-9.-]+", re.I),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def elapsed(start: str | None, end: str | None) -> float | None:
    return round((instant(end) - instant(start)).total_seconds(), 3) if start and end else None


def write_progress(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_new_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def scan_and_redact(raw: bytes, credential: bytes) -> tuple[bytes, dict]:
    """Return category counts only; no matched value or surrounding text leaves here."""
    patterns = {"authenticated_credential_literal": re.compile(re.escape(credential)), **SECRET_PATTERNS}
    before = {name: len(pattern.findall(raw)) for name, pattern in patterns.items()}
    normalized_before = {name: len(pattern.findall(ANSI_BYTES.sub(b"", raw))) for name, pattern in patterns.items()}
    safe = raw
    replacements = {}
    for name, pattern in patterns.items():
        safe, count = pattern.subn(b"[REDACTED]", safe)
        replacements[name] = count
    # Preserve downloaded bytes unless normalization is needed to remove a match
    # interrupted by ANSI formatting.
    normalized = ANSI_BYTES.sub(b"", safe)
    ansi_removed = any(pattern.search(normalized) for pattern in patterns.values())
    if ansi_removed:
        safe = normalized
        for name, pattern in patterns.items():
            safe, count = pattern.subn(b"[REDACTED]", safe)
            replacements[name] += count
    after = {name: len(pattern.findall(safe)) for name, pattern in patterns.items()}
    normalized_after = {name: len(pattern.findall(ANSI_BYTES.sub(b"", safe))) for name, pattern in patterns.items()}
    return safe, {
        "method": "Known authentication credential literal plus named byte-regex signatures, also scanned after ANSI removal; matched values are never recorded.",
        "counts_before_redaction": before,
        "counts_before_redaction_after_ansi_removal": normalized_before,
        "redactions_by_pattern": replacements,
        "counts_after_redaction": after,
        "counts_after_redaction_after_ansi_removal": normalized_after,
        "total_redactions": sum(replacements.values()),
        "clean_after_redaction": not any(after.values()) and not any(normalized_after.values()),
        "ansi_removed_to_redact_an_interrupted_match": ansi_removed,
        "masked_placeholders_are_not_secret_values": True,
    }


def checked_receipt_bytes(data: dict, credential: bytes) -> bytes:
    payload = {**data, "receipt_secrets_pattern_scan": {"total_matches": 0, "clean": True}}
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    _, scan = scan_and_redact(encoded, credential)
    if scan["total_redactions"] or not scan["clean_after_redaction"]:
        raise RuntimeError("Receipt metadata matched a configured secret pattern; no receipt was written")
    return encoded


def git(*args: str) -> str:
    result = subprocess.run([GIT, *args], cwd=ROOT, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Local Git preflight failed")
    return result.stdout.decode("utf-8").strip()


class Session:
    def __init__(self) -> None:
        self.operations = []
        self.env = os.environ.copy()
        self.env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", GH_PROMPT_DISABLED="1", GH_PAGER="cat", GH_HOST="github.com", NO_COLOR="1")
        for name in ("GH_DEBUG", "GH_FORCE_TTY"):
            self.env.pop(name, None)
        result = subprocess.run(
            [GCM, "get"], input=f"protocol=https\nhost=github.com\nusername={OWNER}\n\n".encode(),
            capture_output=True, env=self.env, timeout=30,
        )
        if result.returncode:
            raise RuntimeError("Git Credential Manager could not supply existing authentication")
        fields = dict(line.split("=", 1) for line in result.stdout.decode("utf-8").splitlines() if "=" in line)
        self.token = fields.get("password", "")
        if not self.token:
            raise RuntimeError("No GitHub credential was returned")
        self.env["GH_TOKEN"] = self.token

    def get_bytes(self, endpoint: str) -> bytes:
        started = time.monotonic()
        result = subprocess.run(
            [GH, "api", "--method", "GET", "--hostname", "github.com", "--allow-escape-sequences", endpoint],
            cwd=ROOT, env=self.env, capture_output=True, timeout=60,
        )
        self.operations.append({"method": "GET", "endpoint": endpoint, "exit_code": result.returncode, "seconds": round(time.monotonic() - started, 3)})
        if result.returncode:
            # Never include raw stderr/stdout: either could contain authentication material.
            raise RuntimeError(f"GitHub GET failed for {endpoint} (exit {result.returncode})")
        return result.stdout

    def get(self, endpoint: str) -> dict:
        return json.loads(self.get_bytes(endpoint))


def brief_job(job: dict) -> dict:
    fields = ("id", "run_id", "run_attempt", "head_sha", "name", "status", "conclusion", "started_at", "completed_at", "html_url", "runner_name", "runner_group_name", "labels")
    item = {key: job.get(key) for key in fields}
    item["elapsed_seconds"] = elapsed(job.get("started_at"), job.get("completed_at"))
    item["steps"] = [
        {**{key: step.get(key) for key in ("number", "name", "status", "conclusion", "started_at", "completed_at")},
         "elapsed_seconds": elapsed(step.get("started_at"), step.get("completed_at"))}
        for step in job.get("steps", [])
    ]
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-sha", required=True, help="Full pushed commit SHA; must equal local HEAD")
    args = parser.parse_args()
    head = args.head_sha.lower()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise RuntimeError("A full 40-character commit SHA is required")
    if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT or git("rev-parse", "HEAD") != head:
        raise RuntimeError("Expected commit does not match this standalone repository's HEAD")
    if git("remote", "get-url", "origin") not in {f"https://github.com/{REPO}.git", f"https://github.com/{REPO}", f"git@github.com:{REPO}.git"}:
        raise RuntimeError("Local origin does not match the intended repository")
    if any(path.exists() or path.is_symlink() for path in (RECEIPT, LOG)):
        raise RuntimeError("Canonical draft.3 hosted evidence already exists; it must not be overwritten")
    session = Session()
    identity = session.get("user")
    if identity.get("login") != OWNER or identity.get("id") != OWNER_ID:
        raise RuntimeError("Authenticated GitHub identity differs from the intended owner")
    repository = session.get(f"repos/{REPO}")
    if (repository.get("full_name") != REPO or repository.get("private") is not True
            or repository.get("owner", {}).get("login") != OWNER
            or repository.get("owner", {}).get("id") != OWNER_ID):
        raise RuntimeError("The intended private repository is not available")
    receipt = {
        "schema": "wspf-hosted-ci-verification/2",
        "checked_at_utc": now(),
        "repository": REPO,
        "workflow_path": WORKFLOW,
        "workflow_name": WORKFLOW_NAME,
        "head_sha": head,
        "local_head_sha": head,
        "authenticated_login": OWNER,
        "authenticated_user_id": OWNER_ID,
        "required_steps": list(REQUIRED_STEPS),
        "success_rule": "Exact push commit, workflow and attempt; every hosted job and each required publication step completed successfully.",
        "test_total_quality_gate": False,
        "remote_mutations": [],
    }
    runs = session.get(f"repos/{REPO}/actions/workflows/publication-python.yml/runs?head_sha={head}&event=push&per_page=100")
    candidates = [run for run in runs.get("workflow_runs", []) if run.get("head_sha") == head and run.get("path") == WORKFLOW and run.get("event") == "push" and run.get("head_branch") == "main"]
    receipt["matching_run_ids"] = [run["id"] for run in candidates]
    if not candidates:
        receipt.update(status="not_found", conclusion=None, operations=session.operations)
        write_progress(PROGRESS, checked_receipt_bytes(receipt, session.token.encode("utf-8")))
        print(json.dumps({"status": "not_found", "head_sha": head, "receipt": str(PROGRESS)}))
        return 2
    selected = max(candidates, key=lambda run: run["id"])
    run = session.get(f"repos/{REPO}/actions/runs/{selected['id']}")
    if run.get("head_sha") != head or run.get("path") != WORKFLOW or run.get("name") != WORKFLOW_NAME or run.get("event") != "push" or run.get("head_branch") != "main":
        raise RuntimeError("Selected hosted run does not match the requested commit and workflow")
    run_id, attempt = run["id"], run["run_attempt"]
    receipt.update({
        "run_id": run_id, "run_attempt": attempt, "workflow_id": run.get("workflow_id"),
        "url": run.get("html_url"), "html_url": run.get("html_url"),
        "event": run.get("event"), "head_branch": run.get("head_branch"),
        "status": run.get("status"), "conclusion": run.get("conclusion"),
        "created_at": run.get("created_at"), "run_started_at": run.get("run_started_at"), "updated_at": run.get("updated_at"),
        "run_started_to_last_update_seconds": elapsed(run.get("run_started_at"), run.get("updated_at")),
    })
    jobs_data = session.get(f"repos/{REPO}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100")
    jobs = jobs_data.get("jobs", [])
    if jobs_data.get("total_count") != len(jobs):
        raise RuntimeError("Hosted job response is incomplete")
    receipt["jobs"] = [brief_job(job) for job in jobs]
    if run.get("status") != "completed" or any(job.get("status") != "completed" for job in jobs):
        receipt["operations"] = session.operations
        write_progress(PROGRESS, checked_receipt_bytes(receipt, session.token.encode("utf-8")))
        print(json.dumps({key: receipt[key] for key in ("head_sha", "run_id", "run_attempt", "status", "conclusion")}))
        return 2
    matching_jobs = [job for job in jobs if job.get("name") == "publication-python"]
    if len(matching_jobs) != 1:
        raise RuntimeError("Expected exactly one publication-python job")
    job = matching_jobs[0]
    for item in jobs:
        if item.get("run_id") != run_id or item.get("head_sha") != head or item.get("run_attempt", attempt) != attempt:
            raise RuntimeError("Hosted job does not match the selected run, attempt and commit")
    raw_log = session.get_bytes(f"repos/{REPO}/actions/jobs/{job['id']}/logs")
    safe_log, scan = scan_and_redact(raw_log, session.token.encode("utf-8"))
    if not scan["clean_after_redaction"]:
        raise RuntimeError("Log still contains a configured secret pattern after redaction")
    steps = {step["name"]: step for step in job.get("steps", [])}
    pytest_step = steps.get(PYTEST_STEP)
    receipt.update({
        "job_id": job["id"],
        "job_ids": [item["id"] for item in jobs],
        "python_suite_step": {key: pytest_step.get(key) for key in ("number", "name", "status", "conclusion", "started_at", "completed_at")} if pytest_step is not None else None,
        "downloaded_log": {"bytes": len(raw_log), "sha256": digest(raw_log)},
        "log_bytes_preserved": raw_log == safe_log,
        "log_transport": "Binary stdout from gh api GET; no newline or text-encoding transformation.",
        "secrets_pattern_scan": scan,
        "operations": session.operations,
    })
    failures = []
    if run.get("conclusion") != "success":
        failures.append("Hosted run conclusion is not success")
    if not jobs or any(item.get("conclusion") != "success" for item in jobs):
        failures.append("Not every hosted job concluded success")
    for name in REQUIRED_STEPS:
        named_steps = [step for step in job.get("steps", []) if step.get("name") == name]
        if len(named_steps) != 1 or named_steps[0].get("status") != "completed" or named_steps[0].get("conclusion") != "success":
            failures.append(f"Expected exactly one successful completed step: {name}")
    final_run = session.get(f"repos/{REPO}/actions/runs/{run_id}")
    for key in ("id", "head_sha", "path", "name", "event", "head_branch", "run_attempt", "status", "conclusion"):
        if final_run.get(key) != run.get(key):
            raise RuntimeError("Hosted run changed during monitoring; no final evidence was written")
    if git("rev-parse", "HEAD") != head:
        raise RuntimeError("Local HEAD changed during monitoring; no final evidence was written")
    observed = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    observation = f"draft3-run-{run_id}-attempt-{attempt}-{observed}"
    attempt_receipt = ROOT / ".runs" / f"hosted-ci-verification-{observation}.json"
    attempt_log = ROOT / ".runs" / f"hosted-ci-job-{observation}.log"
    log_path = attempt_log if failures else LOG
    receipt.update(
        verification_passed=not failures, verification_failures=failures,
        completed_at_utc=now(), operations=session.operations,
        log={"path": log_path.relative_to(ROOT).as_posix(), "bytes": len(safe_log), "sha256": digest(safe_log)},
        attempt_log={"path": attempt_log.relative_to(ROOT).as_posix(), "bytes": len(safe_log), "sha256": digest(safe_log)},
        attempt_receipt=attempt_receipt.relative_to(ROOT).as_posix(),
        canonical_receipt=RECEIPT.relative_to(ROOT).as_posix() if not failures else None,
    )
    # Scan every receipt before writing, without silently changing metadata.
    receipt_bytes = checked_receipt_bytes(receipt, session.token.encode("utf-8"))
    write_new_bytes(attempt_log, safe_log)
    write_new_bytes(attempt_receipt, receipt_bytes)
    if not failures:
        write_new_bytes(LOG, safe_log)
        write_new_bytes(RECEIPT, receipt_bytes)
    print(json.dumps({key: receipt[key] for key in ("head_sha", "run_id", "run_attempt", "status", "conclusion", "verification_passed", "verification_failures", "python_suite_step", "log", "attempt_receipt", "canonical_receipt")}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Hosted CI monitor error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        # Exception messages are our own fixed context; never include subprocess output.
        print(f"Hosted CI monitor error: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1)
