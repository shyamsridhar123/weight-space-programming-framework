"""Finite, authored transport fixtures; no provider or scientific work is run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from rlcd.weight_space import live_provider
from scripts import dark_factory as factory


_EMITTER = r'''
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

# Even a broken parent that stops servicing pipes cannot leave this fixture alive.
watchdog = threading.Timer(10, lambda: os._exit(97))
watchdog.daemon = True
watchdog.start()
if sys.argv[1] == "--hold":
    time.sleep(float(sys.argv[2]))
    os._exit(0)
plan_path = Path(sys.argv[1])
plan = json.loads(plan_path.read_text(encoding="utf-8"))

def write(fd, data):
    while data:
        try:
            count = os.write(fd, data[:8192])
        except BrokenPipeError:
            os._exit(0)  # Closing an overflowed pipe must not create fixture diagnostics.
        data = data[count:]

if plan["stdin_mode"] != "ignore":
    incoming = sys.stdin.buffer.read(1048577)
    if plan["stdin_mode"] == "echo":
        write(1, incoming)
outputs = [bytes.fromhex(plan[name]) for name in ("stdout", "stderr")]
if plan["parallel"]:
    writers = [threading.Thread(target=write, args=(fd, data))
               for fd, data in zip((1, 2), outputs)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join()
else:
    for fd, data in zip((1, 2), outputs):
        write(fd, data)
if plan["descendant_seconds"]:
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    subprocess.Popen([sys.executable, __file__, "--hold", str(plan["descendant_seconds"])],
        stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr, **options)
plan_path.with_suffix(".ready").write_text("ready", encoding="ascii")
time.sleep(plan["sleep_seconds"])
'''


@pytest.fixture
def finite_emitter(tmp_path):
    """One fixed local program, bounded payloads, and tracked direct children."""
    script = tmp_path / "finite_emitter.py"
    script.write_text(_EMITTER, encoding="utf-8")
    native_popen = subprocess.Popen
    processes, plans = [], []

    def command(*, stdout=b"", stderr=b"", parallel=False, stdin_mode="read",
                sleep_seconds=0, descendant_seconds=0):
        assert max(len(stdout), len(stderr)) <= 512 * 1024
        assert 0 <= sleep_seconds <= 8 and 0 <= descendant_seconds <= 3
        plan = tmp_path / f"emitter-{len(plans)}.json"
        plan.write_text(json.dumps({"stdout": stdout.hex(), "stderr": stderr.hex(),
            "parallel": parallel, "stdin_mode": stdin_mode, "sleep_seconds": sleep_seconds,
            "descendant_seconds": descendant_seconds}), encoding="utf-8")
        plans.append(plan)
        return [sys.executable, "-u", str(script), str(plan)]

    def start(*, wait_ready=False, **plan):
        argv = command(**plan)
        options = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
                   else {"start_new_session": True})
        process = native_popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0, **options)
        processes.append(process)
        if wait_ready:
            ready = Path(argv[-1]).with_suffix(".ready")
            deadline = time.monotonic() + 3
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "Finite emitter did not reach its authored wait"
        return process

    yield SimpleNamespace(command=command, start=start, native_popen=native_popen,
                          processes=processes)
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def assert_capture_identity(capture, stdout, stderr, limits):
    for name, raw, limit in zip(("stdout", "stderr"), (stdout, stderr), limits):
        assert capture[f"max_{name}_bytes"] == limit
        assert capture[f"{name}_bytes"] == len(raw) <= limit
        assert capture[f"{name}_sha256"] == hashlib.sha256(raw).hexdigest()
        assert type(capture[f"{name}_eof"]) is bool
    assert isinstance(capture["read_errors"], list)
    assert capture["stdin_error"] is None or isinstance(capture["stdin_error"], str)


@pytest.mark.parametrize("field", ["max_stdout_bytes", "max_stderr_bytes"])
@pytest.mark.parametrize("limit", [True, False, 0, -1], ids=["true", "false", "zero", "negative"])
def test_invalid_output_limits_are_rejected_before_launch(tmp_path, monkeypatch, field, limit):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Invalid output limits must be rejected before process launch")

    monkeypatch.setattr(factory.subprocess, "Popen", unexpected_launch)
    with pytest.raises(ValueError, match="Invalid live-provider configuration"):
        live_provider.CodexProvider(**{field: limit})
    ledger, run = [], tmp_path / "uncreated-run"
    with pytest.raises(ValueError, match="positive output limits"):
        factory.command("authored", ["authored-emitter"], cwd=tmp_path, run=run,
                        ledger=ledger, **{field: limit})
    assert ledger == [] and not run.exists()


@pytest.mark.parametrize("timeout", [
    pytest.param(True, id="true"), pytest.param(False, id="false"),
    pytest.param(0, id="zero-int"), pytest.param(-1, id="negative-int"),
    pytest.param(0.0, id="zero-float"), pytest.param(-0.5, id="negative-float"),
    pytest.param(float("nan"), id="nan"), pytest.param(float("inf"), id="positive-inf"),
    pytest.param(float("-inf"), id="negative-inf"), pytest.param(10**400, id="oversized-int"),
    pytest.param(None, id="none"), pytest.param("1", id="string"),
])
def test_invalid_timeouts_are_rejected_before_launch(tmp_path, monkeypatch, timeout):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Invalid timeouts must be rejected before process launch")

    monkeypatch.setattr(factory.subprocess, "Popen", unexpected_launch)
    with pytest.raises(ValueError, match="Invalid live-provider configuration"):
        live_provider.CodexProvider(timeout_seconds=timeout)
    ledger, run = [], tmp_path / "uncreated-run"
    with pytest.raises(ValueError, match="finite deadline"):
        factory.command("authored", ["authored-emitter"], cwd=tmp_path, run=run,
                        ledger=ledger, timeout=timeout)
    with pytest.raises(ValueError, match="finite deadline"):
        live_provider._communicate_bounded(None, timeout=timeout)
    assert ledger == [] and not run.exists()


def test_provider_timeout_retains_integer_contract():
    with pytest.raises(ValueError, match="Invalid live-provider configuration"):
        live_provider.CodexProvider(timeout_seconds=1.0)


@pytest.mark.parametrize("stdout,stderr,limits,parallel,overflow", [
    pytest.param(b"normal\n", b"diagnostic\n", (64, 64), False, [], id="normal"),
    pytest.param(b"a" * 4096, b"b" * 3072, (4096, 3072), False, [], id="exact-both"),
    pytest.param(b"a" * 4097, b"", (4096, 3072), False, ["stdout"], id="stdout-plus-one"),
    pytest.param(b"", b"b" * 3073, (4096, 3072), False, ["stderr"], id="stderr-plus-one"),
    pytest.param(b"a" * 262144, b"b" * 262144, (262144, 262144), True, [],
                 id="simultaneous-pipe-pressure"),
    pytest.param(b"\xff" + b"\xe2\x82\xac" * 65536, b"\x00\r\xff", (196609, 3), False, [],
                 id="long-raw-record-without-newline"),
    pytest.param(b"\xff" + b"\xe2\x82\xac" * 65536, b"", (196607, 64), False, ["stdout"],
                 id="cap-splits-multibyte-sequence"),
])
def test_capture_retains_exact_raw_prefixes(finite_emitter, stdout, stderr, limits, parallel, overflow):
    process = finite_emitter.start(stdout=stdout, stderr=stderr, parallel=parallel)
    started = time.monotonic()
    out, err, timed_out, cleanup, capture = live_provider._communicate_bounded(
        process, timeout=3, input=b"authored input", max_stdout_bytes=limits[0],
        max_stderr_bytes=limits[1])
    assert time.monotonic() - started < 9
    assert (out, err) == (stdout[:limits[0]], stderr[:limits[1]])
    assert_capture_identity(capture, out, err, limits)
    assert timed_out is False
    assert capture["overflow"] == overflow
    assert capture["complete"] is (not overflow)
    assert capture["cleanup_complete"] is cleanup
    if overflow:
        assert cleanup is not True
    else:
        assert cleanup is True and process.returncode == 0
        assert capture["stdout_eof"] is capture["stderr_eof"] is True
        assert capture["read_errors"] == [] and capture["stdin_error"] is None


def test_capture_advances_after_short_stdin_writes(finite_emitter):
    process = finite_emitter.start(stdin_mode="echo")
    native_stdin, writes = process.stdin, []

    class ShortWriter:
        def write(self, data):
            writes.append(len(data))
            return native_stdin.write(data[:7])

        def close(self):
            native_stdin.close()

    process.stdin = ShortWriter()
    raw = bytes(range(256)) + b"last"
    out, err, timed_out, cleanup, capture = live_provider._communicate_bounded(
        process, timeout=3, input=raw, max_stdout_bytes=len(raw), max_stderr_bytes=64)
    assert out == raw and err == b""
    assert len(writes) > 1 and writes[0] > 7
    assert timed_out is False and cleanup is True and capture["complete"] is True
    assert_capture_identity(capture, out, err, (len(raw), 64))


@pytest.mark.parametrize("unread_input", [False, True], ids=["idle-worker", "blocked-stdin"])
def test_deadline_covers_process_and_unread_stdin(finite_emitter, unread_input):
    process = finite_emitter.start(stdout=b"started\n", stdin_mode="ignore",
                                   sleep_seconds=8, wait_ready=True)
    started = time.monotonic()
    out, err, timed_out, cleanup, capture = live_provider._communicate_bounded(
        process, timeout=0.1, input=b"x" * 524288 if unread_input else None,
        max_stdout_bytes=64, max_stderr_bytes=64)
    assert time.monotonic() - started < 6
    assert out == b"started\n" and err == b""
    assert timed_out is True and cleanup is None
    assert capture["complete"] is False and capture["overflow"] == []
    assert_capture_identity(capture, out, err, (64, 64))


def test_exited_leader_does_not_end_deadline_with_inherited_pipe(finite_emitter):
    process = finite_emitter.start(stdout=b"leader exited\n", stdin_mode="ignore",
                                   descendant_seconds=3)
    assert process.wait(timeout=3) == 0
    started = time.monotonic()
    out, err, timed_out, cleanup, capture = live_provider._communicate_bounded(
        process, timeout=0.05, max_stdout_bytes=64, max_stderr_bytes=64)
    assert time.monotonic() - started < 6
    assert out == b"leader exited\n" and err == b""
    assert timed_out is True and cleanup is not True
    assert capture["complete"] is False and capture["overflow"] == []
    assert_capture_identity(capture, out, err, (64, 64))


def test_cleanup_budget_reports_lingering_process_and_pipes(finite_emitter, monkeypatch):
    process = finite_emitter.start(stdout=b"still alive\n", stdin_mode="ignore",
                                   sleep_seconds=8, wait_ready=True)
    kill_budgets = []
    monkeypatch.setattr(live_provider, "_kill_tree",
                        lambda process, *, timeout: kill_budgets.append(timeout))
    started = time.monotonic()
    out, err, timed_out, cleanup, capture = live_provider._communicate_bounded(
        process, timeout=0.05, max_stdout_bytes=64, max_stderr_bytes=64)
    assert time.monotonic() - started < 6
    assert len(kill_budgets) == 1 and 0 < kill_budgets[0] <= 5
    assert process.poll() is None
    assert out == b"still alive\n" and err == b""
    assert timed_out is True and cleanup is False
    assert capture["cleanup_complete"] is False and capture["complete"] is False
    assert capture["stdout_eof"] is capture["stderr_eof"] is False


def completed_events():
    response = {"actions": [json.dumps({"op": "forbidden_returned_action"})],
                "reason": "authored transport fixture"}
    usage = {"input_tokens": 11, "output_tokens": 7}
    events = [{"type": "item.completed", "item": {
        "type": "agent_message", "text": json.dumps(response)}},
        {"type": "turn.completed", "usage": usage}]
    return ("\n".join(json.dumps(event) for event in events) + "\n").encode(), response, usage


def route_provider(monkeypatch, tmp_path, finite_emitter, *, probe, worker):
    """Only the deliberately named CLI is substituted; taskkill stays real."""
    cli = tmp_path / "authored-cli-placeholder"
    cli.write_bytes(b"authored placeholder, never executed")
    calls = []
    monkeypatch.setattr(live_provider.shutil, "which", lambda name: str(cli))

    def popen(argv, **kwargs):
        if argv[0] != str(cli):
            # Windows process-tree termination is a separate native command.
            assert Path(argv[0]).name.lower() in {"taskkill", "taskkill.exe"}
            return finite_emitter.native_popen(argv, **kwargs)
        role = "version_probe" if argv[1:] == ["--version"] else "worker"
        calls.append(role)
        assert kwargs["bufsize"] == 0 and not kwargs.get("text", False)
        assert kwargs["stdout"] == kwargs["stderr"] == subprocess.PIPE
        plan = probe if role == "version_probe" else worker
        assert plan is not None, "Rejected version probe must prevent worker launch"
        process = finite_emitter.native_popen(finite_emitter.command(**plan), **kwargs)
        finite_emitter.processes.append(process)
        return process

    monkeypatch.setattr(live_provider.subprocess, "Popen", popen)
    return calls


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_version_overflow_prevents_worker_launch(tmp_path, monkeypatch, finite_emitter, stream):
    calls = route_provider(monkeypatch, tmp_path, finite_emitter,
                           probe={stream: b"v" * 65}, worker=None)
    directory = tmp_path / "attempt"
    receipt = live_provider.CodexProvider(max_stdout_bytes=64, max_stderr_bytes=64).call(
        "Authored request", directory=directory, cwd=tmp_path / "cwd", schema={"type": "object"})
    assert calls == ["version_probe"]
    assert receipt["status"] == "failed" and receipt["error"] == "version_probe_output_limit_exceeded"
    assert receipt["response"] is None and receipt["usage"] == {}
    captures = receipt["invocation"]["output_capture"]
    assert captures["worker"] is None
    assert captures["version_probe"]["overflow"] == [stream]
    assert captures["version_probe"]["complete"] is False
    out = (directory / "version.stdout.private").read_bytes()
    err = (directory / "version.stderr.private").read_bytes()
    assert (out, err) == ((b"v" * 64, b"") if stream == "stdout" else (b"", b"v" * 64))
    assert_capture_identity(captures["version_probe"], out, err, (64, 64))
    assert (directory / "events.jsonl").read_bytes() == b""
    assert (directory / "transport.stderr.private").read_bytes() == b""
    assert json.loads((directory / "receipt.json").read_text(encoding="utf-8")) == receipt


def test_invalid_version_utf8_prevents_worker_launch(tmp_path, monkeypatch, finite_emitter):
    calls = route_provider(monkeypatch, tmp_path, finite_emitter,
                           probe={"stdout": b"\xff"}, worker=None)
    directory = tmp_path / "attempt"
    receipt = live_provider.CodexProvider(max_stdout_bytes=64, max_stderr_bytes=64).call(
        "Authored request", directory=directory, cwd=tmp_path / "cwd", schema={"type": "object"})
    assert calls == ["version_probe"]
    assert receipt["status"] == "failed"
    assert receipt["error"] == "transport_startup_failed:UnicodeDecodeError"
    assert receipt["response"] is None and receipt["usage"] == {}
    assert receipt["invocation"]["output_capture"]["worker"] is None
    assert (directory / "version.stdout.private").read_bytes() == b"\xff"
    assert_capture_identity(receipt["invocation"]["output_capture"]["version_probe"],
                            b"\xff", b"", (64, 64))


@pytest.mark.parametrize("case", ["normal", "stdout-trailer", "stderr-trailer", "raw-invalid", "invalid-utf8"])
def test_provider_capture_controls_authority_and_raw_receipts(tmp_path, monkeypatch, finite_emitter, case):
    raw, response, usage = completed_events()
    limits = (len(raw), 64)
    stdout = raw + (b"x" if case == "stdout-trailer" else b"")
    stderr = b"e" * 65 if case == "stderr-trailer" else b""
    if case == "raw-invalid":
        stdout, limits = b"\xff\xe2\x82\xac" + raw, (len(raw) + 2, 64)
    elif case == "invalid-utf8":
        stdout, limits = raw + b"\xff\n", (len(raw) + 2, 64)
        stderr = b"private stderr fixture"
    calls = route_provider(monkeypatch, tmp_path, finite_emitter,
        probe={"stdout": b"offline-fixture\n"}, worker={"stdout": stdout, "stderr": stderr})
    before_decode, capture_raw = [], live_provider._communicate_bounded

    def record_capture(*args, **kwargs):
        result = capture_raw(*args, **kwargs)
        before_decode.append(dict(result[-1]))
        return result

    monkeypatch.setattr(live_provider, "_communicate_bounded", record_capture)
    directory = tmp_path / "attempt"
    receipt = live_provider.CodexProvider(max_stdout_bytes=limits[0], max_stderr_bytes=limits[1]).call(
        "Authored request", directory=directory, cwd=tmp_path / "cwd", schema={"type": "object"})
    assert calls == ["version_probe", "worker"]
    out = (directory / "events.jsonl").read_bytes()
    err = (directory / "transport.stderr.private").read_bytes()
    assert (out, err) == (stdout[:limits[0]], stderr[:limits[1]])
    capture = receipt["invocation"]["output_capture"]["worker"]
    assert_capture_identity(capture, out, err, limits)
    assert receipt["events_sha256"] == hashlib.sha256(out).hexdigest()
    assert json.loads((directory / "receipt.json").read_text(encoding="utf-8")) == receipt
    if case == "normal":
        assert receipt["status"] == "completed" and receipt["error"] is None
        assert receipt["response"] == response and receipt["usage"] == usage
        assert capture["complete"] is True
    else:
        assert receipt["status"] == "failed"
        assert receipt["error"] == ("invalid_event_utf8" if case == "invalid-utf8"
                                    else "provider_output_limit_exceeded")
        assert receipt["response"] is None and receipt["usage"] == {}
        assert capture["complete"] is False
        assert capture["overflow"] == ([] if case == "invalid-utf8" else
                                       ["stderr"] if case == "stderr-trailer" else ["stdout"])
        if case in ("raw-invalid", "invalid-utf8"):
            assert capture["text_decoding_error"] == "UnicodeDecodeError"
        if case == "invalid-utf8":
            assert len(before_decode) == 2 and before_decode[-1]["complete"] is True
            assert before_decode[-1]["stdout_eof"] is before_decode[-1]["stderr_eof"] is True
            assert before_decode[-1]["overflow"] == []
            assert_capture_identity(before_decode[-1], out, err, limits)
            public_directory = tmp_path / "public"
            publication = live_provider.publish_attempt(directory, public_directory)
            public = (public_directory / "events.jsonl").read_bytes()
            assert public == raw + b"\xef\xbf\xbd\n"
            assert (directory / "events.jsonl").read_bytes() == stdout
            assert (directory / "transport.stderr.private").read_bytes() == stderr
            assert publication["transform"] == (
                "UTF8_replacement_then_JSON_reserialization_and_user_profile_prefix_redaction")
            assert publication["raw_events_sha256"] == receipt["events_sha256"]
            assert publication["public_events_sha256"] == hashlib.sha256(public).hexdigest()
            assert publication["raw_events_sha256"] != publication["public_events_sha256"]
            assert publication["raw_bytes_published"] is False
            assert {path.name for path in public_directory.iterdir()} == {
                "request.json", "response-schema.json", "receipt.json", "events.jsonl",
                "event-publication.json"}
            assert json.loads((public_directory / "receipt.json").read_text(encoding="utf-8")) == receipt
            assert json.loads((public_directory / "event-publication.json").read_text(
                encoding="utf-8")) == publication


def test_overflowed_valid_completion_never_reaches_broker_actions(tmp_path, monkeypatch, finite_emitter):
    raw, _, _ = completed_events()
    calls = route_provider(monkeypatch, tmp_path, finite_emitter,
        probe={"stdout": b"offline-fixture\n"}, worker={"stdout": raw + b"x"})

    class RecordingBroker:
        is_sealed = False

        def __init__(self):
            self.requests, self.charges, self.abstentions = [], [], 0

        def request(self, action):
            self.requests.append(action)
            assert action == {"op": "metadata"}, "Rejected output reached a broker action"
            return {"fixture": "metadata"}

        def record_usage(self, *counts):
            self.charges.append(counts)

        def abstain(self):
            self.abstentions += 1
            self.is_sealed = True

    broker = RecordingBroker()
    result = live_provider.run_brokered(
        live_provider.CodexProvider(max_stdout_bytes=len(raw), max_stderr_bytes=64), broker,
        prompt="Authored request", directory=tmp_path / "trial", max_rounds=3)
    assert calls == ["version_probe", "worker"]
    assert result["status"] == "failed" and len(result["attempts"]) == 1
    assert result["attempts"][0]["response"] is None and result["attempts"][0]["usage"] == {}
    assert broker.requests == [{"op": "metadata"}] and broker.charges == []
    assert broker.abstentions == 1 and broker.is_sealed


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_factory_rejects_overflow_after_zero_exit(tmp_path, monkeypatch, finite_emitter, stream):
    raw = b"private fixture bytes:" + b"\xff\xe2\x82\xac" * 10
    process = finite_emitter.start(**{stream: raw}, stdin_mode="ignore")
    assert process.wait(timeout=3) == 0
    calls = []

    def exited_process(argv, **kwargs):
        calls.append(argv)
        assert kwargs["bufsize"] == 0 and kwargs["stdin"] == subprocess.DEVNULL
        return process

    monkeypatch.setattr(factory.subprocess, "Popen", exited_process)
    ledger = []
    with pytest.raises(RuntimeError, match="authored:failed"):
        factory.command("authored", ["authored-emitter"], cwd=tmp_path, run=tmp_path,
                        ledger=ledger, max_stdout_bytes=32, max_stderr_bytes=32)
    assert calls == [["authored-emitter"]]
    record = ledger[0]
    assert record["status"] == "failed" and record["returncode"] == 0
    assert record["error"] == "command_output_limit_exceeded"
    out = (tmp_path / "command-000.stdout.private").read_bytes()
    err = (tmp_path / "command-000.stderr.private").read_bytes()
    assert (out, err) == ((raw[:32], b"") if stream == "stdout" else (b"", raw[:32]))
    assert_capture_identity(record["output_capture"], out, err, (32, 32))
    assert record["stdout_sha256"] == hashlib.sha256(out).hexdigest()
    assert record["stderr_sha256"] == hashlib.sha256(err).hexdigest()
    assert record["output_capture"]["overflow"] == [stream]
    assert record["output_capture"]["complete"] is False
    assert json.loads((tmp_path / "command-000.json").read_text(encoding="utf-8")) == record
    summary = factory.failure_summary({"commands": ledger, "run_id": "authored-fixture", "issue": 0,
        "issue_sha256": "0" * 64, "attempts": [], "stages": [], "elapsed_seconds": 0})
    assert summary["commands"][0]["output_capture"] == record["output_capture"]
    assert "private fixture bytes" not in json.dumps(summary)
