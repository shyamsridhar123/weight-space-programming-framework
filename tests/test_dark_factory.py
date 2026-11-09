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


def test_failed_native_startup_and_deadline_leave_receipts(tmp_path):
    ledger = []
    with pytest.raises(RuntimeError, match="startup_failed"):
        factory.command("missing", [str(tmp_path / "unavailable-executable")], cwd=tmp_path,
                        run=tmp_path, ledger=ledger)
    with pytest.raises(RuntimeError, match="timeout"):
        factory.command("deadline", [sys.executable, "-c", "import time; time.sleep(10)"], cwd=tmp_path,
                        run=tmp_path, ledger=ledger, timeout=0.05)
    assert [r["status"] for r in ledger] == ["startup_failed", "timeout"]
    assert ledger[1]["timeout_cleanup_complete"] is True
    for i, record in enumerate(ledger):
        assert json.loads((tmp_path / f"command-{i:03}.json").read_text()) == record
        assert (tmp_path / f"command-{i:03}.stdout.private").is_file()


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
                          calls=[], comments=[], commands=[], mode="accept", crashed=False,
                          before_github=lambda *args: None, after_github=lambda *args: None,
                          after_command=lambda *args: None)
    def github(method, path, data=None):
        env.calls.append((method, path))
        if method in {"POST", "PATCH"}:
            assert git(root, "rev-parse", "HEAD") == git(remote, "rev-parse", "refs/heads/main")
            operation = "comment" if method == "POST" else "close"
            journal_path = next(state.glob(f"*/github-{operation}.json"))
            journal = json.loads(journal_path.read_text())
            retained = json.loads((journal_path.parent / "result.json").read_text())
            assert journal["status"] == "pending" and journal["method"] == method and journal["path"] == path
            assert journal["candidate_commit"] == retained["candidate_commit"]
            assert journal["run_id"] == retained["run_id"] and journal["issue"] == issue["number"]
            assert journal["request_sha256"] == factory.digest(data)
        env.before_github(method, path, data)
        if method == "GET" and "/comments?" in path:
            page = int(path.rsplit("page=", 1)[1])
            return env.comments[(page - 1) * 100:page * 100]
        if method == "GET" and "/issues?" in path:
            page = int(path.rsplit("page=", 1)[1])
            return [dict(i) for i in env.issues[(page - 1) * 100:page * 100]]
        if method == "GET" and path == "/repos/" + factory.REPOSITORY:
            return {"id": factory.REPOSITORY_ID, "full_name": factory.REPOSITORY,
                    "visibility": "public", "default_branch": "main", "archived": False}
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
        if name in {"integrate", "push"}:
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
    return sum(actual == method for actual, _ in env.calls)


def run_cycle(env):
    return factory.cycle(dict(env.issue), state=env.state, publish=True, worker_seconds=30)


def crash(env):
    env.crashed = True
    raise SimulatedCrash()


@pytest.mark.parametrize("mode", ["accept", "partial", "reviewer_mutates"])
def test_cycle_pushes_before_closing_and_quarantines_failed_work(factory_env, mode):
    env = factory_env
    env.mode = mode
    result = run_cycle(env)
    if mode == "accept":
        assert result["status"] == "completed" and env.issue["state"] == "closed"
        assert mutations(env, "POST") == mutations(env, "PATCH") == 1
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


@pytest.mark.parametrize("boundary", ["integrate", "push", "POST", "PATCH"])
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
    assert env.commands.count("integrate") == env.commands.count("push") == 1
    assert all((run / name).read_bytes() == content for name, content in before.items())
    assert any("comments?per_page=100&page=2" in path for _, path in env.calls)
    assert factory.recover_pending(env.state, publish=True) == []


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_unobserved_request_remains_pending_and_blocks_queue(factory_env, monkeypatch, method):
    env = factory_env
    delayed_request = {}
    def lose_request(actual, path, data):
        if actual == method:
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


@pytest.mark.parametrize("boundary", ["integrate", "comment_read"])
def test_failures_after_integration_or_push_publish_safe_accounting(factory_env, boundary):
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
    assert result["status"] == "failed" and result["recovery_complete"]
    assert result["failure_publication"] == "published"
    assert result["candidate_commit"] != result["base_commit"]
    assert result["failure_base_commit"] == result["candidate_commit"]
    assert env.git(env.remote, "rev-parse", "refs/heads/main") == result["failure_commit"]
    assert env.issue["state"] == "open" and mutations(env, "POST") == mutations(env, "PATCH") == 0
    failure_path = env.root / "artifacts/factory" / result["run_id"] / "failure.json"
    failure = json.loads(failure_path.read_text())
    assert failure["error"] == "RuntimeError" and private_error not in failure_path.read_text()
    assert failure["candidate_commit"] == result["candidate_commit"]
    assert audit(env.root)["ok"]


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
