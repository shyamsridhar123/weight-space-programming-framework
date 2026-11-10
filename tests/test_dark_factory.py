"""Offline orchestration checks; fake providers are never scientific trials."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import dark_factory as factory
from scripts.audit_public_content import PUBLIC_CI_LOGS, audit


def assessment(evidence="README.md"):
    return {"outcome": "ready", "summary": "Implemented", "claim_scope": "offline orchestration fixture",
            "attempt_accounting": "one assigned attempt", "criteria": [
                {"id": 0, "met": True, "evidence": [evidence], "explanation": "file inspected"}]}


def test_requirements_dependencies_and_public_evidence_fail_closed(tmp_path):
    issue = {"number": 8, "state": "open", "title": "[P1] study",
             "body": "- [ ] execute\r\nDependencies for the final claim: #2, #5 Pilot/design may proceed"}
    assert factory.requirements(issue) == ["execute"]
    assert factory.dependencies(issue) == {2, 5}
    assert not factory.eligible([issue, {"number": 2, "state": "closed"}, {"number": 5, "state": "open"}])
    assert factory.eligible([issue, {"number": 2, "state": "closed"}, {"number": 5, "state": "closed"}]) == [issue]
    assert not factory.eligible([issue | {"body": "- [ ] missing dependency declaration"}])
    (tmp_path / "README.md").write_text("evidence")
    for relative in (".runs/private.json", ".proof-runs/proof.json", ".venv/result.json"):
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
        assert not factory.assessment_ready(assessment(relative), ["execute"], tmp_path)
    assert factory.assessment_ready(assessment(), ["execute"], tmp_path)
    for relative in ("../README.md", "./README.md", "README.md:1"):
        assert not factory.assessment_ready(assessment(relative), ["execute"], tmp_path)
    bad = assessment()
    bad["criteria"][0]["id"] = False
    assert not factory.assessment_ready(bad, ["execute"], tmp_path)
    assert factory.protected_changes([".github/new.yml", "artifacts/old.json", "artifacts/new.json", "pytest-engineering.ini"],
                                     {"artifacts/old.json"}) == [".github/new.yml", "artifacts/old.json", "pytest-engineering.ini"]


def test_assessment_diagnostics_preserve_short_circuit_and_safe_fields(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("evidence")
    scans = []
    def published(root):
        scans.append(root)
        return ["README.md", "missing.md"]
    monkeypatch.setattr(factory, "publication_files", published)
    def diagnose(response):
        return factory.assessment_diagnostics(response, ["execute"], tmp_path)
    ready = {"outcome": "ready", "reason_codes": [], "rejected_criterion_ids": []}
    assert diagnose(assessment()) == ready
    assert factory.assessment_ready(assessment(), ["execute"], tmp_path)
    assert scans == [tmp_path, tmp_path]
    scans.clear()
    assert diagnose(None) == ready | {"outcome": None, "reason_codes": ["response_not_object"]}
    for outcome, known in (("partial", "partial"), ("blocked", "blocked"),
                           ("private outcome", None), ({"private": "outcome"}, None)):
        assert diagnose(assessment() | {"outcome": outcome}) == ready | {
            "outcome": known, "reason_codes": ["outcome_not_ready"]}
    assert not factory.assessment_ready(assessment() | {"outcome": "partial"}, ["execute"], tmp_path)
    assert diagnose(assessment() | {"summary": " "}) == ready | {"reason_codes": ["required_text_missing"]}
    assert diagnose(assessment() | {"criteria": None}) == ready | {"reason_codes": ["criteria_invalid"]}
    for criterion_id in (False, 99, "private criterion"):
        response = assessment()
        response["criteria"][0]["id"] = criterion_id
        assert diagnose(response) == ready | {"reason_codes": ["criteria_invalid"]}
    assert scans == []
    for changes, reason in (
            ({"met": False, "evidence": None, "explanation": None}, "criterion_not_met"),
            ({"evidence": [], "explanation": None}, "criterion_evidence_missing"),
            ({"explanation": " "}, "criterion_explanation_missing"),
            ({"evidence": ["../private-evidence.txt"]}, "evidence_path_invalid"),
            ({"evidence": ["private-evidence.txt"]}, "evidence_not_published"),
            ({"evidence": ["missing.md"]}, "evidence_file_invalid")):
        response = assessment()
        response["summary"] = "private assessment prose"
        response["criteria"][0].update(changes)
        assert diagnose(response) == ready | {"reason_codes": [reason], "rejected_criterion_ids": [0]}
    assert scans == [tmp_path] * 6
    def unavailable(root):
        raise OSError("publication scanner unavailable")
    monkeypatch.setattr(factory, "publication_files", unavailable)
    with pytest.raises(OSError, match="publication scanner unavailable"):
        diagnose(assessment())


@pytest.mark.parametrize("provider_status", ["completed", "timeout"])
def test_assessment_diagnostics_checkpoint_completed_attempt_once(tmp_path, monkeypatch, provider_status):
    root, state = tmp_path / "project", tmp_path / "state"
    root.mkdir()
    state.mkdir()
    commands, provider_calls, validations = [], [], []
    outputs = {"working_tree": "", "branch": "main", "base": "a" * 40,
               "remote_base": "a" * 40 + "\trefs/heads/main", "historical_paths": "README.md", "worktree_add": ""}
    def command(name, argv, **kwargs):
        commands.append(name)
        if name == "worktree_add":
            worktree = Path(argv[-2])
            worktree.mkdir()
            (worktree / "README.md").write_text("evidence")
        return outputs[name]
    response = assessment("private-evidence.txt")
    response["summary"] = "private assessment prose"
    class Provider:
        def __init__(self, **options):
            pass
        def call(self, prompt, *, directory, cwd, schema):
            provider_calls.append(directory.name)
            return {"status": provider_status, "sha256": "0" * 64, "elapsed_seconds": 0.01,
                    "usage": {}, "cost": {"amount": None, "currency": "USD"}, "response": response}
    original = factory.assessment_diagnostics
    def diagnose(response, criteria, worktree):
        validations.append(True)
        retained = json.loads((worktree.parent / "result.json").read_text())
        assert retained["attempts"][0]["status"] == "completed"
        assert "assessment" not in retained["attempts"][0]
        return original(response, criteria, worktree)
    monkeypatch.setattr(factory, "ROOT", root)
    monkeypatch.setattr(factory, "command", command)
    monkeypatch.setattr(factory, "CodexProvider", Provider)
    monkeypatch.setattr(factory, "assessment_diagnostics", diagnose)
    monkeypatch.setattr(factory, "publication_files", lambda root: ["README.md"])
    monkeypatch.setattr(factory, "account_failure", lambda *args: None)
    issue = {"number": 2, "title": "offline diagnostics fixture", "state": "open",
             "body": "- [ ] execute\nDependencies for the final claim: None; independent."}
    result = factory.cycle(issue, state=state, publish=False, worker_seconds=1)
    assert result["status"] == "failed" and result["error"] == "worker_did_not_fulfill_requirements"
    assert commands == list(outputs) and provider_calls == ["worker"]
    assert len(validations) == (1 if provider_status == "completed" else 0)
    entry = result["attempts"][0]
    assert entry["status"] == provider_status
    if provider_status == "completed":
        assert entry["assessment"] == {"outcome": "ready", "reason_codes": ["evidence_not_published"],
                                       "rejected_criterion_ids": [0]}
    else:
        assert "assessment" not in entry
    retained = json.loads((state / result["run_id"] / "result.json").read_text())
    assert retained["attempts"] == [entry]
    summary = factory.failure_summary(result)
    assert summary["attempts"] == [entry]
    for private in (response["summary"], response["criteria"][0]["evidence"][0], response["criteria"][0]["explanation"]):
        assert private not in json.dumps(summary)


def test_failed_native_startup_and_deadline_leave_receipts(tmp_path):
    ledger = []
    with pytest.raises(RuntimeError, match="startup_failed"):
        factory.command("missing", [str(tmp_path / "unavailable-executable")], cwd=tmp_path,
                        run=tmp_path, ledger=ledger)
    with pytest.raises(RuntimeError, match="timeout"):
        factory.command("deadline", [sys.executable, "-c", "import time; time.sleep(10)"], cwd=tmp_path,
                        run=tmp_path, ledger=ledger, timeout=0.05)
    assert [r["status"] for r in ledger] == ["startup_failed", "timeout"]
    assert ledger[1]["timeout_cleanup_complete"] is None
    for i, record in enumerate(ledger):
        assert json.loads((tmp_path / f"command-{i:03}.json").read_text()) == record
        assert (tmp_path / f"command-{i:03}.stdout.private").is_file()


def test_command_timeout_retains_capture_without_attesting_cleanup(tmp_path, monkeypatch):
    import hashlib

    stdout, stderr = b"retained stdout", b"retained stderr"
    process = SimpleNamespace(returncode=-9)
    capture = {"max_stdout_bytes": 64, "max_stderr_bytes": 32,
               "stdout_bytes": len(stdout), "stderr_bytes": len(stderr),
               "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
               "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
               "stdout_eof": True, "stderr_eof": True, "overflow": [], "read_errors": [],
               "stdin_error": None, "returncode": -9, "cleanup_complete": None, "complete": False}

    def popen(argv, **kwargs):
        assert kwargs["bufsize"] == 0 and not kwargs.get("text", False)
        assert kwargs["stdin"] == subprocess.DEVNULL
        return process

    def communicate(candidate, **kwargs):
        assert candidate is process
        assert kwargs == {"timeout": 1, "max_stdout_bytes": 64, "max_stderr_bytes": 32}
        return stdout, stderr, True, None, capture

    monkeypatch.setattr(factory.subprocess, "Popen", popen)
    monkeypatch.setattr(factory, "_communicate_bounded", communicate)
    ledger = []
    with pytest.raises(RuntimeError, match="authored:timeout"):
        factory.command("authored", ["offline-placeholder"], cwd=tmp_path, run=tmp_path,
                        ledger=ledger, timeout=1, max_stdout_bytes=64, max_stderr_bytes=32)
    assert ledger[0]["status"] == "timeout" and ledger[0]["timeout_cleanup_complete"] is None
    assert ledger[0]["output_capture"] == capture
    assert json.loads((tmp_path / "command-000.json").read_text(encoding="utf-8")) == ledger[0]
    assert (tmp_path / "command-000.stdout.private").read_bytes() == stdout
    assert (tmp_path / "command-000.stderr.private").read_bytes() == stderr


def test_command_preserves_raw_output_left_before_receipt_checkpoint(tmp_path):
    abandoned = tmp_path / "command-000.stdout.private"
    abandoned.write_bytes(b"output retained after an interruption")
    ledger = []
    assert factory.command("next", [sys.executable, "-c", "print('continued')"],
                           cwd=tmp_path, run=tmp_path, ledger=ledger) == "continued"
    assert abandoned.read_bytes() == b"output retained after an interruption"
    assert json.loads((tmp_path / "command-001.json").read_text()) == ledger[0]


def test_content_audit_reports_locations_without_values(tmp_path):
    secret = "ghp_" + "A" * 40
    user_path = "C:" + "\\Users\\" + "fixture" + "_person" + "\\file"
    (tmp_path / "unsafe.json").write_text(json.dumps({"secret": secret, "path": user_path}))
    (tmp_path / ".env").write_text("EXAMPLE=true")
    result = audit(tmp_path)
    assert not result["ok"]
    assert {f["kind"] for f in result["findings"]} == {"github_token", "private_user_path", "credential_filename"}
    assert secret not in json.dumps(result) and user_path not in json.dumps(result)
    (tmp_path / "unsafe.json").write_text(json.dumps({"path": user_path.replace("fixture_person", "username")}))
    (tmp_path / ".env").unlink()
    assert audit(tmp_path)["ok"]


@pytest.mark.parametrize("profile_root", [
    "/home/",
    "/Users/",
    "C:/Users/",
    "C:/Documents and Settings/",
    "C:" + "\\Users\\",
    "C:" + "\\Documents and Settings\\",
])
def test_content_audit_checks_fixture_paths_and_tokens(tmp_path, profile_root):
    relative = "tests/test_research_broker.py"
    source = tmp_path / relative
    source.parent.mkdir()
    cases = [
        ("private-person", "", []),
        ("fixture" + "_person", "", [
            {"path": relative, "kind": "private_user_path"},
        ]),
        ("private-person", "ghp_" + "A" * 40, [
            {"path": relative, "kind": "github_token"},
        ]),
    ]
    for person, token, expected in cases:
        profile = profile_root + person + profile_root[-1] + "events.jsonl"
        source.write_text(
            f"profile = {profile!r}\ntoken = {token!r}\n",
            encoding="utf-8",
        )
        result = audit(tmp_path)
        assert result["files_scanned"] == 1
        assert result["findings"] == expected
        assert result["ok"] is (not expected)


def test_ci_service_path_exception_is_limited_to_known_public_logs(tmp_path):
    log = tmp_path / next(iter(PUBLIC_CI_LOGS))
    log.parent.mkdir(parents=True)
    log.write_text("/home/" + "runner/work/public-repository")
    assert audit(tmp_path)["ok"]
    log.write_text("/home/" + "another-person/private")
    assert not audit(tmp_path)["ok"]
    log.write_text("C:" + "\\Users\\runner\\private")
    assert not audit(tmp_path)["ok"]
    log.write_text("/home/" + "runner/work/public-repository")
    (tmp_path / "README.md").write_text("/home/" + "runner/private")
    assert not audit(tmp_path)["ok"]


class SimulatedCrash(BaseException):
    pass


@pytest.fixture
def factory_env(tmp_path, monkeypatch):
    root, remote, state = (tmp_path / name for name in ("project", "remote.git", "private"))
    root.mkdir()
    state.mkdir()
    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    git(tmp_path, "init", "--bare", str(remote))
    git(root, "init", "-b", "main")
    # Private pytest roots can exceed Git for Windows' default path limit.
    git(root, "config", "core.longpaths", "true")
    git(root, "config", "user.name", "Offline fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / "README.md").write_text("original")
    git(root, "add", "README.md")
    git(root, "commit", "-m", "base")
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "origin", "main")
    issue = {"number": 2, "state": "open", "title": "[P0] offline fixture",
             "body": "- [ ] implement\nDependencies for the final claim: None; independent."}
    env = SimpleNamespace(root=root, remote=remote, state=state, git=git, issue=issue, issues=[issue],
                          calls=[], comments=[], pull_requests=[], commands=[], provider_calls=[],
                          mode="accept", auto_merge=True, crashed=False,
                          before_github=lambda *args: None, after_github=lambda *args: None,
                          after_command=lambda *args: None)
    repository = {"id": factory.REPOSITORY_ID, "full_name": factory.REPOSITORY}

    def merge_pull_request(pull_request=None, *, delete_branch=False):
        pull_request = pull_request or env.pull_requests[-1]
        if not pull_request["merged"]:
            candidate = pull_request["head"]["sha"]
            base = git(remote, "rev-parse", "refs/heads/main")
            tree = git(remote, "rev-parse", candidate + "^{tree}")
            merged = git(remote, "-c", "user.name=Offline merger", "-c", "user.email=fixture@example.invalid",
                         "commit-tree", tree, "-p", base, "-p", candidate, "-m", "Externally merged offline PR")
            git(remote, "update-ref", "refs/heads/main", merged, base)
            pull_request.update(state="closed", merged=True, merged_at="2026-11-09T00:00:00Z",
                                merge_commit_sha=merged)
        if delete_branch:
            git(remote, "update-ref", "-d", "refs/heads/" + pull_request["head"]["ref"])
        return pull_request["merge_commit_sha"]

    def create_pull_request(data):
        branch = data["head"]
        candidate = git(remote, "rev-parse", "refs/heads/" + branch)
        pull_request = {"number": len(env.pull_requests) + 1, "id": len(env.pull_requests) + 100,
                        "title": data["title"], "body": data["body"], "state": "open", "draft": data["draft"],
                        "merged": False, "merged_at": None, "merge_commit_sha": None,
                        "html_url": "https://github.com/" + factory.REPOSITORY + "/pull/" + str(len(env.pull_requests) + 1),
                        "head": {"ref": branch, "sha": candidate, "repo": dict(repository)},
                        "base": {"ref": data["base"], "sha": git(remote, "rev-parse", "refs/heads/main"),
                                 "repo": dict(repository)}}
        env.pull_requests.append(pull_request)
        if env.auto_merge:
            merge_pull_request(pull_request)
        return pull_request

    env.merge_pull_request = merge_pull_request
    env.create_pull_request = create_pull_request

    def github(method, path, data=None):
        env.calls.append((method, path))
        if method in {"POST", "PATCH"}:
            is_pull_request = path == f"/repos/{factory.REPOSITORY}/pulls"
            if not is_pull_request:
                assert git(root, "rev-parse", "HEAD") == git(remote, "rev-parse", "refs/heads/main")
            operation = "pull-request" if is_pull_request else "comment" if method == "POST" else "close"
            journal_path = next(state.glob(f"*/github-{operation}.json"))
            journal = json.loads(journal_path.read_text())
            retained = json.loads((journal_path.parent / "result.json").read_text())
            assert journal["status"] == "pending" and journal["method"] == method and journal["path"] == path
            assert journal["candidate_commit"] == retained["candidate_commit"]
            assert journal["run_id"] == retained["run_id"] and journal["issue"] == issue["number"]
            assert journal["request_sha256"] == factory.digest(data)
            if is_pull_request:
                assert method == "POST" and journal["operation"] == "pull-request"
                assert journal["repository"] == factory.REPOSITORY and journal["repository_id"] == factory.REPOSITORY_ID
                assert journal["issue_sha256"] == retained["issue_sha256"]
                assert journal["base_commit"] == retained["base_commit"]
                assert data["head"] == "factory/" + retained["run_id"]
                assert data["base"] == "main" and data["draft"] is False
                assert data["title"] == f"factory: accepted work for #{issue['number']}"
                assert git(root, "rev-parse", "HEAD") == retained["base_commit"]
                assert git(remote, "rev-parse", "refs/heads/main") == retained["base_commit"]
        env.before_github(method, path, data)
        if method == "GET" and "/pulls?" in path:
            page = int(path.rsplit("page=", 1)[1])
            return json.loads(json.dumps(env.pull_requests[(page - 1) * 100:page * 100]))
        if method == "GET" and "/pulls/" in path:
            number = int(path.rsplit("/", 1)[1])
            return json.loads(json.dumps(next(pr for pr in env.pull_requests if pr["number"] == number)))
        if method == "GET" and "/comments?" in path:
            page = int(path.rsplit("page=", 1)[1])
            return env.comments[(page - 1) * 100:page * 100]
        if method == "GET" and "/issues?" in path:
            page = int(path.rsplit("page=", 1)[1])
            return [dict(i) for i in env.issues[(page - 1) * 100:page * 100]]
        if method == "GET" and path == "/repos/" + factory.REPOSITORY:
            return {"id": factory.REPOSITORY_ID, "full_name": factory.REPOSITORY,
                    "visibility": "public", "default_branch": "main", "archived": False}
        if method == "POST" and path == f"/repos/{factory.REPOSITORY}/pulls":
            pull_request = create_pull_request(data)
            env.after_github(method, path, data)
            return json.loads(json.dumps(pull_request))
        if method == "POST":
            comment = {"id": len(env.comments) + 1, "body": data["body"]}
            env.comments.append(comment)
            env.after_github(method, path, data)
            return dict(comment)
        if method == "PATCH":
            issue["state"] = "closed"
            issue["state_reason"] = "completed"
        env.after_github(method, path, data)
        return dict(issue)
    class Provider:
        def __init__(self, **options):
            self.role = "reviewer" if options.get("sandbox") == "read-only" else "worker"
        def call(self, prompt, *, directory, cwd, schema):
            env.provider_calls.append(self.role)
            directory.mkdir()
            response = assessment()
            if self.role == "worker":
                (cwd / "README.md").write_text("implemented")
                if env.mode == "partial":
                    response["outcome"] = "partial"
            elif env.mode == "reviewer_mutates":
                (cwd / "README.md").write_text("mutated same path")
            receipt = {"status": "completed", "sha256": "0" * 64, "elapsed_seconds": 0.01,
                       "usage": {"input_tokens": 1, "output_tokens": 1},
                       "cost": {"amount": None, "currency": "USD"}, "response": response}
            for name, value in (("request.json", {}), ("response-schema.json", schema), ("receipt.json", receipt)):
                (directory / name).write_text(json.dumps(value))
            (directory / "events.jsonl").write_text("{}\n")
            return receipt
    original_command = factory.command
    def command(name, argv, **kwargs):
        env.commands.append(name)
        if name in {"integrate", "pr_branch_push"}:
            retained = json.loads((kwargs["run"] / "result.json").read_text())
            assert retained["candidate_commit"] and retained["status"] in {"accepted", "pending"}
        if name == "python_suite":
            # The test exercises orchestration, not a nested research suite.
            assert argv[argv.index("-c") + 1] == "pytest-engineering.ini"
            assert kwargs["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
            argv = [sys.executable, "-c", "print('offline validation fixture')"]
        output = original_command(name, argv, **kwargs)
        env.after_command(name, kwargs["run"])
        return output
    original_checkpoint = factory.checkpoint
    def checkpoint(*args, **kwargs):
        if env.crashed:
            # A killed process cannot execute cycle's finally checkpoint.
            raise SimulatedCrash()
        return original_checkpoint(*args, **kwargs)
    monkeypatch.setattr(factory, "ROOT", root)
    monkeypatch.setattr(factory, "github", github)
    monkeypatch.setattr(factory, "CodexProvider", Provider)
    monkeypatch.setattr(factory, "command", command)
    monkeypatch.setattr(factory, "checkpoint", checkpoint)
    monkeypatch.setattr(factory, "publication_gate", lambda p: {"publication": {"ok": True}, "content": {"ok": True}})
    # The failure-accounting path stages a manifest as well as its safe receipt.
    (root / "MANIFEST.sha256").write_text("")
    git(root, "add", "MANIFEST.sha256")
    git(root, "commit", "-m", "manifest fixture")
    git(root, "push", "origin", "main")
    return env


def mutations(env, method):
    return sum(actual == method and "/pulls" not in path for actual, path in env.calls)


def pull_request_mutations(env):
    return sum(method == "POST" and path == f"/repos/{factory.REPOSITORY}/pulls" for method, path in env.calls)


def run_cycle(env):
    return factory.cycle(dict(env.issue), state=env.state, publish=True, worker_seconds=30)


def crash(env):
    env.crashed = True
    raise SimulatedCrash()


def assert_pending_candidate(env, result):
    assert result["status"] == "pending" and not result["recovery_complete"]
    assert result["candidate_commit"] and env.issue["state"] == "open"
    assert "failure_commit" not in result and "failure_publication" not in result
    assert not (env.state / result["run_id"] / "failure-public.json").exists()
    assert not (env.root / "artifacts/factory" / result["run_id"] / "failure.json").exists()
    assert not any(name.startswith("failure_") for name in env.commands)
    assert "push" not in env.commands
    assert mutations(env, "PATCH") == 0


@pytest.mark.parametrize("mode", ["accept", "partial", "reviewer_mutates"])
def test_cycle_waits_for_pr_merge_before_closing_and_quarantines_failed_work(factory_env, mode):
    env = factory_env
    env.mode = mode
    result = run_cycle(env)
    if mode == "accept":
        assert result["status"] == "completed" and env.issue["state"] == "closed"
        assert result["publication_mode"] == "pull_request"
        assert pull_request_mutations(env) == 1 and env.pull_requests[0]["merged"]
        assert mutations(env, "POST") == mutations(env, "PATCH") == 1
        assert "push" not in env.commands and "failure_push" not in env.commands
        assert env.commands.index("merged_remote_fetch") < env.commands.index("merged_candidate_ancestor")
        assert env.commands.index("merged_candidate_ancestor") < env.commands.index("integrate")
        assert (env.root / "README.md").read_text() == "implemented"
        assert result["stages"].index("published_commit_verified") < result["stages"].index("issue_closed_after_publication")
    else:
        assert result["status"] == "failed" and env.issue["state"] == "open"
        assert mutations(env, "POST") == mutations(env, "PATCH") == 0
        assert result["failure_publication"] == "published"
        assert (env.root / "README.md").read_text() == "original"
        failure = json.loads((env.root / "artifacts/factory" / result["run_id"] / "failure.json").read_text())
        assert failure["claim"] == "failed_attempt_accounting;_no_issue_completion"
        assert (env.state / result["run_id"] / "worktree").is_dir()


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_lost_github_response_is_reconciled_without_repeating_mutation(factory_env, method):
    env = factory_env
    def lose_response(actual, *_):
        if actual == method:
            raise ConnectionError("offline lost response")
    env.after_github = lose_response
    result = run_cycle(env)
    assert result["status"] == "completed" and result["recovery_complete"]
    assert mutations(env, "POST") == mutations(env, "PATCH") == 1
    operation = "comment" if method == "POST" else "close"
    journal = json.loads((env.state / result["run_id"] / f"github-{operation}.json").read_text())
    assert journal["status"] == "confirmed" and journal["response_error"] == "ConnectionError"


@pytest.mark.parametrize("boundary", ["integrate", "pr_branch_push", "POST", "PATCH"])
def test_crash_recovers_observed_effects_and_preserves_receipts(factory_env, boundary):
    env = factory_env
    # Recovery must locate its marker even when it is on the second page.
    env.comments = [{"id": n + 1, "body": f"unrelated comment {n}"} for n in range(100)]
    def interrupt(actual, *_):
        if actual == boundary:
            crash(env)
    env.after_command = interrupt
    env.after_github = interrupt
    with pytest.raises(SimulatedCrash):
        run_cycle(env)
    run = next(env.state.glob("*/result.json")).parent
    before = {p.name: p.read_bytes() for p in run.glob("command-*.*")}
    retained = json.loads((run / "result.json").read_text())
    assert retained["candidate_commit"] and not retained["recovery_complete"]
    env.crashed = False
    env.after_command = env.after_github = lambda *args: None
    recovered, = factory.recover_pending(env.state, publish=True)
    assert recovered["status"] == "completed" and recovered["recovery_complete"]
    assert env.issue["state"] == "closed"
    assert mutations(env, "POST") == mutations(env, "PATCH") == 1
    assert pull_request_mutations(env) == 1
    assert env.commands.count("integrate") == env.commands.count("pr_branch_push") == 1
    assert "push" not in env.commands and "failure_push" not in env.commands
    assert all((run / name).read_bytes() == content for name, content in before.items())
    assert any("comments?per_page=100&page=2" in path for _, path in env.calls)
    assert factory.recover_pending(env.state, publish=True) == []


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_unobserved_request_remains_pending_and_blocks_queue(factory_env, monkeypatch, method):
    env = factory_env
    delayed_request = {}
    def lose_request(actual, path, data):
        if actual == method and "/issues/" in path:
            delayed_request.update(data)
            raise ConnectionError("outcome unknown")
    env.before_github = lose_request
    result = run_cycle(env)
    assert result["status"] == "pending" and not result["recovery_complete"]
    assert env.issue["state"] == "open" and mutations(env, method) == 1
    env.before_github = lambda *args: None
    recovered, = factory.recover_pending(env.state, publish=True)
    assert recovered["status"] == "pending" and not recovered["recovery_complete"]
    assert mutations(env, method) == 1
    monkeypatch.setattr(factory, "cycle", lambda *args, **kwargs: pytest.fail("selected another issue before recovery"))
    monkeypatch.setattr(sys, "argv", ["dark_factory.py", "run", "--state-directory", str(env.state),
                                      "--publish", "--max-issues", "2"])
    assert factory.main() == 1
    assert mutations(env, method) == 1 and not (env.state / "factory.lock").exists()
    # A delayed remote effect can later become observable without another request.
    if method == "POST":
        env.comments.append({"id": 1, "body": delayed_request["body"]})
    else:
        env.issue.update(delayed_request)
    completed, = factory.recover_pending(env.state, publish=True)
    assert completed["status"] == "completed" and completed["recovery_complete"]
    assert "recovery_error" not in completed and mutations(env, method) == 1


@pytest.mark.parametrize("boundary", ["pr_branch_remote", "pr_branch_verify", "merged_remote_fetch", "integrate", "comment_read"])
def test_ordinary_errors_after_candidate_remain_pending_without_failure_accounting(factory_env, monkeypatch, boundary):
    env = factory_env
    private_error = "ghp_" + "Z" * 40
    def fail_command(name, *_):
        if boundary == name:
            raise RuntimeError(private_error)
    def fail_comment_read(method, path, _):
        if boundary == "comment_read" and method == "GET" and "/comments?" in path:
            raise RuntimeError(private_error)
    env.after_command = fail_command
    env.before_github = fail_comment_read
    result = run_cycle(env)
    assert_pending_candidate(env, result)
    assert result["candidate_commit"] != result["base_commit"]
    assert private_error not in json.dumps(result)
    def fail_finish(*args):
        raise RuntimeError(private_error)
    monkeypatch.setattr(factory, "finish_candidate", fail_finish)
    recovered, = factory.recover_pending(env.state, publish=True)
    assert_pending_candidate(env, recovered)
    assert private_error not in json.dumps(recovered)
    assert env.provider_calls == ["worker", "reviewer"]


def test_pr_body_keeps_review_closing_keywords_in_retained_artifact(factory_env, monkeypatch):
    env = factory_env
    env.auto_merge = False
    original_assessment = assessment
    def closing_keyword_assessment(evidence="README.md"):
        return original_assessment(evidence) | {"summary": "Fixes #2", "claim_scope": "Resolves #2"}
    monkeypatch.setitem(globals(), "assessment", closing_keyword_assessment)
    result = run_cycle(env)
    assert_pending_candidate(env, result)
    body = env.pull_requests[0]["body"]
    assert "Fixes #2" not in body and "Resolves #2" not in body
    assert f"/blob/{result['candidate_commit']}/artifacts/factory/{result['run_id']}/cycle.json" in body
    artifact = env.state / result["run_id"] / "worktree" / "artifacts/factory" / result["run_id"] / "cycle.json"
    retained_review = json.loads(artifact.read_text())["reviewer_assessment"]
    assert retained_review["summary"] == "Fixes #2"
    assert retained_review["claim_scope"] == "Resolves #2"


def test_open_pr_blocks_queue_without_repeating_worker_or_publication(factory_env, monkeypatch):
    env = factory_env
    env.auto_merge = False
    result = run_cycle(env)
    assert_pending_candidate(env, result)
    assert env.git(env.root, "rev-parse", "HEAD") == result["base_commit"]
    assert env.git(env.remote, "rev-parse", "refs/heads/main") == result["base_commit"]
    assert env.git(env.remote, "rev-parse", "refs/heads/factory/" + result["run_id"]) == result["candidate_commit"]
    assert pull_request_mutations(env) == 1 and mutations(env, "POST") == 0
    recovered, = factory.recover_pending(env.state, publish=True)
    assert_pending_candidate(env, recovered)
    monkeypatch.setattr(factory, "cycle", lambda *args, **kwargs: pytest.fail("selected work while PR pending"))
    monkeypatch.setattr(sys, "argv", ["dark_factory.py", "run", "--state-directory", str(env.state),
                                      "--publish", "--max-issues", "2"])
    assert factory.main() == 1
    assert env.provider_calls == ["worker", "reviewer"]
    assert env.commands.count("pr_branch_push") == pull_request_mutations(env) == 1
    assert not (env.state / "factory.lock").exists()


def test_merged_pr_completes_after_source_branch_deletion(factory_env):
    env = factory_env
    env.auto_merge = False
    result = run_cycle(env)
    assert_pending_candidate(env, result)
    merged = env.merge_pull_request(delete_branch=True)
    recovered, = factory.recover_pending(env.state, publish=True)
    assert recovered["status"] == "completed" and recovered["recovery_complete"]
    assert env.issue["state"] == "closed"
    assert env.git(env.root, "rev-parse", "HEAD") == merged
    assert env.git(env.remote, "rev-parse", "refs/heads/main") == merged
    assert env.commands.count("pr_branch_push") == pull_request_mutations(env) == 1
    assert mutations(env, "POST") == mutations(env, "PATCH") == 1
    assert env.provider_calls == ["worker", "reviewer"]


def test_pr_reconciliation_reads_every_page_and_full_detail(factory_env):
    env = factory_env
    base = env.git(env.root, "rev-parse", "HEAD")
    repository = {"id": factory.REPOSITORY_ID, "full_name": factory.REPOSITORY}
    env.pull_requests = [{"number": number, "id": number + 100, "title": "unrelated PR",
                          "body": f"unrelated PR {number}", "state": "closed", "draft": False,
                          "merged": False, "merged_at": None, "merge_commit_sha": None,
                          "head": {"ref": f"unrelated-{number}", "sha": base, "repo": dict(repository)},
                          "base": {"ref": "main", "sha": base, "repo": dict(repository)}}
                         for number in range(1, 101)]
    result = run_cycle(env)
    assert result["status"] == "completed" and result["recovery_complete"]
    assert result["pull_request_number"] == 101 and pull_request_mutations(env) == 1
    assert any("pulls?state=all&per_page=100&page=2" in path for _, path in env.calls)
    assert ("GET", f"/repos/{factory.REPOSITORY}/pulls/101") in env.calls


@pytest.mark.parametrize("conflict", ["unpublished_merge", "uncontained_merge_commit", "changed_head"])
def test_merged_pr_requires_remote_candidate_and_merge_containment(factory_env, conflict):
    env = factory_env
    env.auto_merge = False
    result = run_cycle(env)
    assert_pending_candidate(env, result)
    env.merge_pull_request()
    pull_request = env.pull_requests[0]
    if conflict == "unpublished_merge":
        env.git(env.remote, "update-ref", "refs/heads/main", result["base_commit"])
    elif conflict == "uncontained_merge_commit":
        tree = env.git(env.remote, "rev-parse", result["candidate_commit"] + "^{tree}")
        pull_request["merge_commit_sha"] = env.git(
            env.remote, "-c", "user.name=Offline merger", "-c", "user.email=fixture@example.invalid",
            "commit-tree", tree, "-p", result["candidate_commit"], "-m", "Uncontained offline merge receipt")
    else:
        pull_request["head"]["sha"] = result["base_commit"]
    recovered, = factory.recover_pending(env.state, publish=True)
    assert_pending_candidate(env, recovered)
    assert env.git(env.root, "rev-parse", "HEAD") == result["base_commit"]
    assert "integrate" not in env.commands
    assert pull_request_mutations(env) == env.commands.count("pr_branch_push") == 1
    assert env.provider_calls == ["worker", "reviewer"]


@pytest.mark.parametrize("boundary", ["pr_branch_push", "pull_request_post"])
def test_lost_publication_response_reconciles_one_branch_and_pr(factory_env, boundary):
    env = factory_env
    def lose_command(name, *_):
        if boundary == name:
            raise ConnectionError("offline lost branch push response")
    def lose_response(method, path, _):
        if boundary == "pull_request_post" and method == "POST" and path.endswith("/pulls"):
            raise ConnectionError("offline lost PR response")
    env.after_command = lose_command
    env.after_github = lose_response
    result = run_cycle(env)
    assert result["status"] == "completed" and result["recovery_complete"]
    assert env.commands.count("pr_branch_push") == pull_request_mutations(env) == 1
    assert mutations(env, "POST") == mutations(env, "PATCH") == 1
    assert "push" not in env.commands and "failure_push" not in env.commands
    if boundary == "pull_request_post":
        journal = json.loads((env.state / result["run_id"] / "github-pull-request.json").read_text())
        assert journal["status"] == "confirmed" and journal["response_error"] == "ConnectionError"


def test_unobserved_pr_request_is_not_repeated_and_later_observation_recovers(factory_env):
    env = factory_env
    delayed_request = {}
    def lose_request(method, path, data):
        if method == "POST" and path.endswith("/pulls"):
            delayed_request.update(data)
            raise ConnectionError("offline delayed PR creation")
    env.before_github = lose_request
    result = run_cycle(env)
    assert_pending_candidate(env, result)
    assert pull_request_mutations(env) == 1 and not env.pull_requests
    env.before_github = lambda *args: None
    recovered, = factory.recover_pending(env.state, publish=True)
    assert_pending_candidate(env, recovered)
    assert pull_request_mutations(env) == 1 and not env.pull_requests
    env.create_pull_request(delayed_request)
    completed, = factory.recover_pending(env.state, publish=True)
    assert completed["status"] == "completed" and completed["recovery_complete"]
    assert pull_request_mutations(env) == 1
    assert env.provider_calls == ["worker", "reviewer"]


@pytest.mark.parametrize("conflict", ["duplicate", "head", "closed", "binding", "body", "repository", "base"])
def test_pr_or_journal_conflicts_remain_pending_without_republication(factory_env, conflict):
    env = factory_env
    env.auto_merge = False
    result = run_cycle(env)
    assert_pending_candidate(env, result)
    pull_request = env.pull_requests[0]
    if conflict == "duplicate":
        env.pull_requests.append(json.loads(json.dumps(pull_request)) | {"number": 2, "id": 101})
    elif conflict == "head":
        pull_request["head"]["sha"] = result["base_commit"]
    elif conflict == "closed":
        pull_request["state"] = "closed"
    elif conflict == "body":
        pull_request["body"] += "changed"
    elif conflict == "repository":
        pull_request["head"]["repo"]["id"] += 1
    elif conflict == "base":
        pull_request["base"]["ref"] = "another-base"
    else:
        journal_path = env.state / result["run_id"] / "github-pull-request.json"
        journal = json.loads(journal_path.read_text())
        factory.write_json(journal_path, journal | {"candidate_commit": "f" * 40})
    recovered, = factory.recover_pending(env.state, publish=True)
    assert_pending_candidate(env, recovered)
    assert pull_request_mutations(env) == env.commands.count("pr_branch_push") == 1
    assert env.provider_calls == ["worker", "reviewer"]


@pytest.mark.parametrize("arrival", ["candidate", "pr_branch_remote"])
def test_existing_or_racing_conflicting_branch_is_not_overwritten(factory_env, arrival):
    env = factory_env
    env.auto_merge = False
    def occupy_branch(name, run):
        if name == arrival:
            retained = json.loads((run / "result.json").read_text())
            env.git(env.remote, "update-ref", "refs/heads/factory/" + retained["run_id"], retained["base_commit"])
    env.after_command = occupy_branch
    result = run_cycle(env)
    assert_pending_candidate(env, result)
    assert pull_request_mutations(env) == 0
    assert env.git(env.remote, "rev-parse", "refs/heads/factory/" + result["run_id"]) == result["base_commit"]
    assert env.git(env.remote, "rev-parse", "refs/heads/main") == result["base_commit"]


@pytest.mark.parametrize("publication_mode", ["pull_request", None])
def test_failure_accounting_defensively_refuses_accepted_candidate_before_writing(factory_env, publication_mode):
    env = factory_env
    env.auto_merge = False
    result = run_cycle(env)
    assert_pending_candidate(env, result)
    if publication_mode is None:
        # A legacy post-integration error must not publish the candidate through failure accounting.
        result.pop("publication_mode")
        env.git(env.root, "merge", "--ff-only", result["candidate_commit"])
    assert env.git(env.remote, "rev-parse", "refs/heads/main") == result["base_commit"]
    before = len(env.commands)
    def forbidden_git(*args):
        pytest.fail("accepted candidate reached failure-accounting Git mutation")
    with pytest.raises(factory.RecoveryPending):
        factory.account_failure(env.state / result["run_id"], result | {"status": "failed"}, forbidden_git, publish=True)
    assert len(env.commands) == before
    assert_pending_candidate(env, result)


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("retained_status", ["accepted", "failed"])
def test_legacy_integrated_candidate_only_completes_if_already_published(
        factory_env, monkeypatch, published, retained_status):
    env = factory_env
    original_finish = factory.finish_candidate
    def interrupt_finish(*args):
        raise SimulatedCrash()
    monkeypatch.setattr(factory, "finish_candidate", interrupt_finish)
    with pytest.raises(SimulatedCrash):
        run_cycle(env)
    monkeypatch.setattr(factory, "finish_candidate", original_finish)
    path = next(env.state.glob("*/result.json"))
    result = json.loads(path.read_text())
    result.pop("publication_mode")
    result["status"] = retained_status
    factory.write_json(path, result)
    env.git(env.root, "merge", "--ff-only", result["candidate_commit"])
    if published:
        # This is the retained remote effect of the historical direct-push path.
        env.git(env.root, "push", "origin", result["candidate_commit"] + ":refs/heads/main")
    recovered, = factory.recover_pending(env.state, publish=True)
    assert pull_request_mutations(env) == 0 and "push" not in env.commands
    assert env.provider_calls == ["worker", "reviewer"]
    if published:
        assert recovered["status"] == "completed" and recovered["recovery_complete"]
        assert env.issue["state"] == "closed"
        assert env.commands.index("legacy_remote_fetch") < env.commands.index("legacy_candidate_ancestor")
    else:
        assert_pending_candidate(env, recovered)
        assert env.git(env.remote, "rev-parse", "refs/heads/main") == result["base_commit"]


def test_unfinished_accepted_candidate_recovery_remains_pending(factory_env, monkeypatch):
    env = factory_env
    # An interrupted implementation may return without marking its retained candidate complete.
    monkeypatch.setattr(factory, "finish_candidate", lambda *args: None)
    result = run_cycle(env)
    assert result["candidate_commit"] and not result["recovery_complete"]
    recovered, = factory.recover_pending(env.state, publish=True)
    assert_pending_candidate(env, recovered)
    assert env.git(env.root, "rev-parse", "HEAD") == result["base_commit"]
    assert env.git(env.remote, "rev-parse", "refs/heads/main") == result["base_commit"]
    assert pull_request_mutations(env) == 0 and "pr_branch_push" not in env.commands
    assert env.provider_calls == ["worker", "reviewer"]


@pytest.mark.parametrize("boundary", ["failure_commit", "failure_push"])
def test_failure_accounting_recovers_without_duplicate_commit_or_push(factory_env, boundary):
    env = factory_env
    env.mode = "partial"
    def interrupt(name, *_):
        if name == boundary:
            crash(env)
    env.after_command = interrupt
    with pytest.raises(SimulatedCrash):
        run_cycle(env)
    head = env.git(env.root, "rev-parse", "HEAD")
    run = next(env.state.glob("*/result.json")).parent
    frozen_summary = (run / "failure-public.json").read_bytes()
    env.crashed = False
    env.after_command = lambda *args: None
    recovered, = factory.recover_pending(env.state, publish=True)
    assert recovered["status"] == "failed" and recovered["recovery_complete"]
    assert recovered["failure_publication"] == "published"
    assert env.git(env.root, "rev-parse", "HEAD") == env.git(env.remote, "rev-parse", "refs/heads/main") == head
    assert env.commands.count("failure_commit") == env.commands.count("failure_push") == 1
    assert (run / "failure-public.json").read_bytes() == frozen_summary
    assert factory.recover_pending(env.state, publish=True) == []


@pytest.mark.parametrize("outcome", ["descendant", "diverged", "fetch_failed"])
def test_failure_publication_fetches_exact_remote_before_ancestry(tmp_path, outcome):
    base, failure, remote = (letter * 40 for letter in "abc")
    result = {"run_id": "20261109T000000Z-00000000", "issue": 2, "issue_sha256": "0" * 64,
              "base_commit": base, "status": "failed", "error": "worker_did_not_fulfill_requirements",
              "error_type": "RuntimeError", "commands": [], "stages": ["worker_attempted"],
              "attempts": [{"role": "worker", "status": "completed", "cost": {"amount": None,
                  "currency": "USD", "reason": "authenticated_CLI_subscription_not_request_pricing"}}],
              "elapsed_seconds": 1, "failure_base_commit": base, "failure_commit": failure,
              "failure_publication": "committed_locally", "recovery_complete": False}
    retained = json.loads(json.dumps(result))
    summary = tmp_path / "failure-public.json"
    factory.write_json(summary, factory.failure_summary(result))
    frozen_summary = summary.read_bytes()
    calls = []
    fetch = ["fetch", "--no-tags", "--no-write-fetch-head", "origin", remote]

    def git(name, args):
        calls.append((name, args))
        if name == "failure_branch":
            assert args == ["branch", "--show-current"]
            return "main"
        if name == "failure_base":
            assert args == ["rev-parse", "HEAD"]
            return failure
        if name == "failure_remote_base":
            assert args == ["ls-remote", "origin", "refs/heads/main"]
            return remote + "\trefs/heads/main"
        if name == "failure_remote_fetch":
            assert args == fetch
            if outcome == "fetch_failed":
                raise RuntimeError("failure_remote_fetch:failed")
            return ""
        if name == "failure_remote_ancestor":
            assert calls[-2] == ("failure_remote_fetch", fetch)
            assert args == ["merge-base", remote, failure]
            return failure if outcome == "descendant" else "d" * 40
        pytest.fail("unexpected Git operation: " + name)

    if outcome == "descendant":
        factory.account_failure(tmp_path, result, git, publish=True)
        assert result == retained | {"failure_publication": "published", "recovery_complete": True}
    else:
        error = factory.RecoveryPending if outcome == "diverged" else RuntimeError
        message = "failure_publication_branch_conflict" if outcome == "diverged" else "failure_remote_fetch:failed"
        with pytest.raises(error, match=message):
            factory.account_failure(tmp_path, result, git, publish=True)
        assert result == retained
    assert calls[3] == ("failure_remote_fetch", fetch)
    if outcome == "fetch_failed":
        assert len(calls) == 4
    assert summary.read_bytes() == frozen_summary


def test_failure_accounting_does_not_push_unpublished_main_changes(factory_env):
    env = factory_env
    remote = env.git(env.remote, "rev-parse", "refs/heads/main")
    (env.root / "unrelated.txt").write_text("unpublished local work")
    env.git(env.root, "add", "unrelated.txt")
    env.git(env.root, "commit", "-m", "unrelated local work")
    result = run_cycle(env)
    assert result["status"] == "failed" and not result["recovery_complete"]
    assert result["failure_publication_error"] == "RecoveryPending"
    assert env.git(env.remote, "rev-parse", "refs/heads/main") == remote
    assert not (env.root / "artifacts").exists()
    assert mutations(env, "POST") == mutations(env, "PATCH") == 0


@pytest.mark.parametrize("conflict", ["duplicate", "body", "binding"])
def test_reconciliation_rejects_marker_or_journal_conflicts(factory_env, conflict):
    env = factory_env
    run = env.state / "20261108T000000Z-00000000"
    run.mkdir()
    result = {"run_id": run.name, "issue": env.issue["number"], "issue_sha256": factory.digest(env.issue),
              "candidate_commit": env.git(env.root, "rev-parse", "HEAD"), "reviewer_assessment": assessment()}
    factory.write_json(run / "issue.json", env.issue)
    factory.write_json(run / "result.json", result)
    factory.complete_issue(run, result)
    if conflict == "duplicate":
        env.comments.append(env.comments[0] | {"id": 2})
    elif conflict == "body":
        env.comments[0]["body"] += "conflicting content"
    else:
        journal_path = run / "github-comment.json"
        journal = json.loads(journal_path.read_text())
        factory.write_json(journal_path, journal | {"candidate_commit": "f" * 40})
    with pytest.raises(factory.RecoveryPending):
        factory.complete_issue(run, result)
    assert mutations(env, "POST") == mutations(env, "PATCH") == 1
