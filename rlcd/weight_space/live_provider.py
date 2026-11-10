"""Bounded noninteractive Codex calls with credentials excluded from receipts.

The CLI is an installed transport, not a learned weight reader. Local configured
tool restrictions and observed events are recorded; provider-side computation and
deployment revisions are not attested by the CLI protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any

from .model import canonical_bytes, digest


DEFAULT_MAX_STDOUT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_STDERR_BYTES = 1024 * 1024


TOOL_FREE_CONFIG = {
    "plugins._default.enabled": "false", "mcp_servers": "{}",
    "web_search": '"disabled"', "features.shell_tool": "false",
    "features.unified_exec": "false", "features.apps": "false",
    "features.multi_agent": "false", "features.multi_agent_mode": "false",
    "features.code_mode": "false", "features.js_repl": "false",
    "features.tool_search": "false", "features.skill_search": "false",
    "features.skip_host_skill_discovery": "true",
    "features.view_image": "false",
    "features.image_generation": "false", "features.hooks": "false",
    "features.browser_use": "false", "features.in_app_browser": "false",
    "features.memories": "false", "features.search_tool": "false",
    "features.stable_environment_tools": "false", "features.incremental_tools": "false",
    "features.request_permissions_tool": "false", "tools.update_plan.enabled": "false",
    "project_doc_max_bytes": "0",
}


def write_json(path: Path, value: Any) -> None:
    """Replace a complete JSON receipt atomically; keep the old file on failure."""
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _kill_tree(process: subprocess.Popen, *, timeout: float = 5) -> None:
    """Best-effort termination of the local transport/worker process tree."""
    try:
        if os.name == "nt":
            if process.poll() is not None:
                return  # An exited Windows leader cannot attest descendant cleanup.
            result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            if result.returncode == 0:
                return
        else:
            import signal
            os.killpg(process.pid, signal.SIGKILL)
            return  # The group may outlive its leader.
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
    except OSError:
        pass


def _positive_finite_timeout(value: Any) -> bool:
    if type(value) not in (int, float) or value <= 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _communicate_bounded(process: subprocess.Popen, *, timeout: float, input: bytes | None = None,
                         max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES,
                         max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES):
    """Capture finite raw prefixes; process/pipe completion never attests descendants."""
    if (not _positive_finite_timeout(timeout)
            or any(type(limit) is not int or limit <= 0
                   for limit in (max_stdout_bytes, max_stderr_bytes))
            or (input is not None and not isinstance(input, bytes))
            or process.stdout is None or process.stderr is None):
        raise ValueError("Bounded capture requires binary pipes, positive limits and a finite deadline")
    limits = (max_stdout_bytes, max_stderr_bytes)
    buffers = [bytearray(), bytearray()]
    eof, overflow, read_errors = [False, False], [False, False], [None, None]
    stdin_error = None
    failed, lock = threading.Event(), threading.Lock()
    deadline = time.monotonic() + timeout

    def read_stream(stream, index):
        try:
            while True:
                with lock:
                    remaining = limits[index] - len(buffers[index])
                # The extra byte distinguishes exact-cap EOF from overflow.
                chunk = stream.read(min(65536, remaining + 1))
                with lock:
                    if not chunk:
                        eof[index] = True
                        break
                    buffers[index].extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        overflow[index] = True
                        failed.set()
                        break
        except (OSError, ValueError) as failure:
            with lock:
                read_errors[index] = type(failure).__name__
            failed.set()
        finally:
            try:
                stream.close()
            except (OSError, ValueError) as failure:
                with lock:
                    read_errors[index] = type(failure).__name__
                failed.set()

    def write_input():
        nonlocal stdin_error
        try:
            remaining = memoryview(input or b"")
            while remaining:
                written = process.stdin.write(remaining[:65536])
                if written is None or written <= 0:
                    raise OSError("Incomplete stdin write")
                remaining = remaining[written:]
        except (OSError, ValueError) as failure:
            with lock:
                stdin_error = type(failure).__name__
            failed.set()
        finally:
            try:
                process.stdin.close()
            except (OSError, ValueError) as failure:
                with lock:
                    stdin_error = type(failure).__name__
                failed.set()

    threads = [threading.Thread(target=read_stream, args=(stream, index), daemon=True)
               for index, stream in enumerate((process.stdout, process.stderr))]
    if process.stdin is not None:
        threads.append(threading.Thread(target=write_input, daemon=True))
    for thread in threads:
        thread.start()
    timed_out = False
    while not failed.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        if process.poll() is not None and all(not thread.is_alive() for thread in threads):
            break
        failed.wait(min(0.02, remaining))
    # Recheck after readers finish: an exited leader can still overflow a pipe.
    interrupted = timed_out or failed.is_set()
    cleanup_complete = True
    if interrupted:
        cleanup_deadline = time.monotonic() + 5
        _kill_tree(process, timeout=max(0.001, cleanup_deadline - time.monotonic()))
        remaining = cleanup_deadline - time.monotonic()
        if process.poll() is None and remaining > 0:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
        for thread in threads:
            thread.join(timeout=max(0, cleanup_deadline - time.monotonic()))
        cleanup_complete = (None if process.poll() is not None
                            and all(not thread.is_alive() for thread in threads) else False)
    with lock:
        stdout, stderr = (bytes(buffer) for buffer in buffers)
        capture = {"max_stdout_bytes": max_stdout_bytes, "max_stderr_bytes": max_stderr_bytes,
                   "stdout_bytes": len(stdout), "stderr_bytes": len(stderr),
                   "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                   "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                   "stdout_eof": eof[0], "stderr_eof": eof[1],
                   "overflow": [name for name, exceeded in zip(("stdout", "stderr"), overflow) if exceeded],
                   "read_errors": [name + ":" + error for name, error in
                                   zip(("stdout", "stderr"), read_errors) if error],
                   "stdin_error": stdin_error, "returncode": process.returncode,
                   "cleanup_complete": cleanup_complete,
                   "complete": (not interrupted and all(eof) and not any(overflow)
                                and not any(read_errors) and stdin_error is None)}
    return stdout, stderr, timed_out, cleanup_complete, capture


class _DuplicateJSONKey(ValueError):
    pass


class _RedactedJSONKeyCollision(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey("Duplicate JSON object key")
        result[key] = value
    return result


def _jsonl_lines(text: str) -> list[str]:
    """Split LF/CRLF records, preserving blanks and ignoring one final delimiter."""
    return text.replace("\r\n", "\n").removesuffix("\n").split("\n") if text else []


def parse_events(output: str) -> tuple[Any, dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
    """Parse the final message, and inspect tool events at every lifecycle stage."""
    final, usage, unapproved, errors, diagnostics = None, {}, [], [], []
    completion_seen = False
    for line in _jsonl_lines(output):
        try:
            event = json.loads(line, object_pairs_hook=_unique_json_object)
            if type(event) is not dict:
                raise ValueError("Event must be an object")
            canonical_bytes(event)
            if not isinstance(event.get("type", ""), str):
                raise ValueError("Event type must be a string")
            item = event.get("item")
            if event.get("type", "").startswith("item."):
                if not isinstance(item, dict):
                    raise ValueError("Event item must be an object")
                kind = item.get("type", "unknown")
                if kind == "error":
                    # CLI warnings use error items too; a failed turn/exit is
                    # classified separately. Neither is a native tool invocation.
                    diagnostics.append({"stage": event["type"], "item": item})
                elif kind not in ("reasoning", "agent_message"):
                    unapproved.append(f"{event['type']}:{kind}")
                if event.get("type") == "item.completed" and kind == "agent_message":
                    if completion_seen:
                        errors.append("agent_message_after_turn_completed")
                        usage = {}
                    final = item.get("text")
            if event.get("type") == "turn.completed":
                if completion_seen:
                    errors.append("duplicate_turn_completed")
                usage = {} if completion_seen else event.get("usage", {})
                completion_seen = True
            if event.get("type") in ("error", "turn.failed"):
                errors.append(event["type"])
        except (ValueError, TypeError, RecursionError):
            errors.append("invalid_event")
    try:
        response = json.loads(final, object_pairs_hook=_unique_json_object) if isinstance(final, str) else None
        canonical_bytes(response)
    except (ValueError, RecursionError):
        response = None
        errors.append("invalid_final_response")
    return response, usage, unapproved, errors, diagnostics


def _usage_counts(usage: Any) -> tuple[int, int]:
    """Require both provider counters without converting unknown usage to zero."""
    if type(usage) is not dict or any(type(usage.get(key)) is not int or usage[key] < 0
                                    for key in ("input_tokens", "output_tokens")):
        raise ValueError("Provider usage requires nonnegative integer input_tokens and output_tokens")
    return usage["input_tokens"], usage["output_tokens"]


@dataclass(frozen=True)
class CodexProvider:
    model: str = "gpt-6.1-sol"
    timeout_seconds: int = 180
    tool_free: bool = True
    reasoning_effort: str = "low"
    sandbox: str | None = None
    max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES

    def __post_init__(self) -> None:
        if (not isinstance(self.model, str) or not self.model.strip()
                or type(self.timeout_seconds) is not int or not _positive_finite_timeout(self.timeout_seconds)
                or type(self.tool_free) is not bool
                or any(type(limit) is not int or limit <= 0
                       for limit in (self.max_stdout_bytes, self.max_stderr_bytes))
                or self.sandbox not in (None, "read-only", "workspace-write")
                or (self.tool_free and self.sandbox == "workspace-write")
                or self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max")):
            raise ValueError("Invalid live-provider configuration")

    def configuration(self) -> dict[str, Any]:
        return {"schema": "wsp-codex-provider-config/1", "provider": "openai",
                "requested_model": self.model, "reasoning_effort": self.reasoning_effort,
                "tool_free": self.tool_free, "user_config_loaded": False,
                "ephemeral": True, "timeout_seconds": self.timeout_seconds,
                "max_stdout_bytes": self.max_stdout_bytes, "max_stderr_bytes": self.max_stderr_bytes,
                "configuration_overrides": TOOL_FREE_CONFIG if self.tool_free else {
                    "plugins._default.enabled": "false", "mcp_servers": "{}"},
                "sandbox": self.sandbox or ("read-only" if self.tool_free else "workspace-write"),
                "deployment_revision_exposed": False,
                "internal_model_computation_observed": False,
                "human_assistance": "task_and_harness_authored_before_trial;_no_midtrial_help"}

    def call(self, prompt: str, *, directory: Path, cwd: Path,
             schema: dict[str, Any]) -> dict[str, Any]:
        """Keep every failed/timed-out invocation; never silently retry."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonblank provider prompt is required")
        directory, cwd = Path(directory).resolve(), Path(cwd).resolve()
        if directory.exists():
            raise ValueError("Provider attempt directory must be new")
        directory.mkdir(parents=True)
        cwd.mkdir(parents=True, exist_ok=True)
        cli = shutil.which("codex")
        version, executable_sha256 = None, None
        schema_path = directory / "response-schema.json"
        write_json(schema_path, schema)
        write_json(directory / "request.json", {"prompt": prompt, "configuration": self.configuration()})
        overrides = TOOL_FREE_CONFIG if self.tool_free else {
            "plugins._default.enabled": "false", "mcp_servers": "{}"}
        command = [cli or "codex", "exec", "--ignore-user-config", "--ignore-rules",
                   "--skip-git-repo-check", "--ephemeral", "--json", "--color", "never",
                   "-m", self.model, "-c", 'model_provider="openai"',
                   "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
                   "-c", 'approval_policy="never"',
                   "-s", self.configuration()["sandbox"],
                   "-C", str(cwd), "--output-schema", str(schema_path)]
        for key, value in overrides.items():
            command.extend(["-c", f"{key}={value}"])
        command.append("-")  # Prompts are stdin data, never shell command text.
        started = datetime.now(timezone.utc).isoformat()
        tick = time.perf_counter()
        status, error, returncode = "failed", None, None
        cleanup_complete = None
        output, stderr = b"", b""
        output_capture = {"version_probe": None, "worker": None}
        capture_options = {"max_stdout_bytes": self.max_stdout_bytes,
                           "max_stderr_bytes": self.max_stderr_bytes}
        options: dict[str, Any] = {}
        if __import__("os").name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            options["start_new_session"] = True
        try:
            if cli is None:
                raise FileNotFoundError("Codex CLI unavailable")
            executable_sha256 = hashlib.sha256(Path(cli).read_bytes()).hexdigest()
            probe = subprocess.Popen([cli, "--version"], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, **options)
            probe_stdout, probe_stderr, timed_out, cleanup_complete, capture = _communicate_bounded(
                probe, timeout=15, **capture_options)
            output_capture["version_probe"] = capture
            (directory / "version.stdout.private").write_bytes(probe_stdout)
            (directory / "version.stderr.private").write_bytes(probe_stderr)
            if timed_out:
                status, error = "timeout", "provider_deadline"
            elif capture["overflow"]:
                error = "version_probe_output_limit_exceeded"
            elif not capture["complete"]:
                error = "version_probe_capture_incomplete"
            elif probe.returncode != 0:
                raise subprocess.CalledProcessError(probe.returncode, [cli, "--version"])
            else:
                version = probe_stdout.decode("utf-8").strip()
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, bufsize=0, **options)
                output, stderr, timed_out, cleanup_complete, capture = _communicate_bounded(
                    process, input=prompt.encode("utf-8"), timeout=self.timeout_seconds, **capture_options)
                output_capture["worker"] = capture
                returncode = process.returncode
                if timed_out:
                    status, error = "timeout", "provider_deadline"
                elif capture["overflow"]:
                    error = "provider_output_limit_exceeded"
                elif not capture["complete"]:
                    error = "provider_capture_incomplete"
        except subprocess.TimeoutExpired:
            # The bounded version probe timed out before a worker was started.
            status, error = "timeout", "provider_deadline"
        except (OSError, subprocess.CalledProcessError, UnicodeError) as failure:
            error = "transport_startup_failed:" + type(failure).__name__
        elapsed = time.perf_counter() - tick
        # Raw stderr can contain transport/environment paths. It stays private.
        (directory / "events.jsonl").write_bytes(output)
        (directory / "transport.stderr.private").write_bytes(stderr)
        try:
            event_text = output.decode("utf-8")
        except UnicodeDecodeError:
            event_text = output.decode("utf-8", errors="replace")
            output_capture["worker"]["text_decoding_error"] = "UnicodeDecodeError"
            output_capture["worker"]["complete"] = False
            if error is None:
                error = "invalid_event_utf8"
        response, usage, unapproved, event_errors, diagnostics = parse_events(event_text)
        if output_capture["worker"] is None or output_capture["worker"]["complete"] is not True:
            # A retained prefix is forensic evidence, never authority or known usage.
            response, usage = None, {}
        if status != "timeout" and error is None:
            if self.tool_free and unapproved:
                error = "unapproved_tool_event"
            elif returncode != 0 or response is None or event_errors:
                error = "transport_or_response_failed"
            else:
                try:
                    _usage_counts(usage)
                except ValueError:
                    error = "invalid_provider_usage"
                else:
                    status = "completed"
        invocation = {"executable_sha256": executable_sha256,
                      "cli_version": version, "configuration": self.configuration(),
                      "output_capture": output_capture,
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "schema_sha256": digest(schema), "cwd_role": (
                          "isolated_proposer_directory" if self.tool_free else
                          "readonly_review_worktree" if self.sandbox == "read-only" else "coding_worktree"),
                      "local_invocation_sha256": digest(command)}
        payload = {"schema": "wsp-live-provider-attempt/1", "started_utc": started,
                   "status": status, "error": error, "returncode": returncode,
                   "timeout_cleanup_complete": cleanup_complete if status == "timeout" else None,
                   "elapsed_seconds": elapsed, "invocation": invocation,
                   "response": response, "usage": usage, "unapproved_events": unapproved,
                   "event_errors": event_errors,
                   "diagnostic_items": len(diagnostics),
                   "events_sha256": hashlib.sha256(output).hexdigest(),
                   "cost": {"currency": "USD", "amount": None,
                            "reason": "authenticated_CLI_subscription_not_request_pricing"},
                   "prompt_scope": "exact_harness_prompt;_CLI_system_scaffolding_not_exported",
                   "restrictions_scope": "configured_CLI_tools_and_observed_events;_not_OS_hermeticity"}
        receipt = payload | {"sha256": digest(payload)}
        write_json(directory / "receipt.json", receipt)
        return receipt


ACTION_SCHEMA = {
    "type": "object", "properties": {
        "actions": {"type": "array", "maxItems": 8,
                    "items": {"type": "string"}},
        "reason": {"type": "string"}},
    "required": ["actions", "reason"], "additionalProperties": False}


def run_brokered(provider: CodexProvider, broker: Any, *, prompt: str,
                 directory: Path, max_rounds: int = 3) -> dict[str, Any]:
    """Fresh subprocesses exchange JSON actions; the evaluator is never passed."""
    if type(max_rounds) is not int or max_rounds <= 0 or not provider.tool_free:
        raise ValueError("Brokered trials require bounded rounds and restricted CLI tools")
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Trial directory must be new")
    directory.mkdir(parents=True)
    history = [{"role": "harness", "data": broker.request({"op": "metadata"})}]
    attempts, status = [], "failed"
    for round_number in range(max_rounds):
        request_prompt = (prompt + "\nEncode each action as a JSON string in the actions array. "
                          'For example: {"actions":["{\\\"op\\\":\\\"metadata\\\"}"],"reason":"inspect"}.'
                          "\n\nJSON transcript:\n" + json.dumps(history, ensure_ascii=False))
        attempt = provider.call(request_prompt, directory=directory / f"provider-{round_number}",
                                cwd=directory / "proposer-cwd", schema=ACTION_SCHEMA)
        attempts.append(attempt)
        try:
            broker.record_usage(*_usage_counts(attempt["usage"]))
            if attempt["status"] != "completed":
                status = attempt["status"]
                break
            actions = attempt["response"]["actions"]
            if not isinstance(actions, list) or not actions or len(actions) > 8:
                raise ValueError("Invalid action batch")
            history.append({"role": "proposer", "data": attempt["response"]})
            for action in actions:
                if type(action) is not str:
                    raise ValueError("Action must be encoded as a JSON string")
                result = broker.request(json.loads(action, object_pairs_hook=_unique_json_object))
                history.append({"role": "broker", "data": result})
                if broker.is_sealed:
                    status = "sealed"
                    break
            if broker.is_sealed:
                break
        except (ValueError, KeyError, TypeError, RecursionError):
            status = attempt["status"] if attempt["status"] != "completed" else "invalid"
            break
    if not broker.is_sealed:
        broker.abstain()
    payload = {"schema": "wsp-live-brokered-trial/1", "status": status,
               "assigned_trials": 1, "attempts": attempts, "transcript": history,
               "selection_rule": "provider_selection_within_frozen_round_budget;_else_abstain",
               "retry_rule": "none;_all_rounds_retained", "max_rounds": max_rounds}
    result = payload | {"sha256": digest(payload)}
    write_json(directory / "trial.json", result)
    return result


def publish_attempt(source: Path, destination: Path) -> dict[str, Any]:
    """Publish a derivative; ambiguous-key events become disclosed nonobject markers."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("request.json", "response-schema.json", "receipt.json"):
        shutil.copyfile(source / name, destination / name)
    raw = (source / "events.jsonl").read_bytes()
    transform = "JSON_reserialization_and_user_profile_prefix_redaction"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        transform = "UTF8_replacement_then_" + transform
    redactions = 0
    profile = re.compile(
        r'(?:[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\\/"\s]+'
        r'|/(?:home|Users)/[^\\/"\s]+)')

    def sanitize(value: Any) -> Any:
        nonlocal redactions
        if isinstance(value, str):
            result, count = profile.subn("[USER_PROFILE]", value)
            redactions += count
            return result
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                redacted_key = sanitize(key)
                if redacted_key in result:
                    raise _RedactedJSONKeyCollision(
                        "JSON object key collision after profile redaction")
                result[redacted_key] = sanitize(item)
            return result
        return value

    lines = []
    for line in _jsonl_lines(text):
        previous_redactions = redactions
        try:
            value = json.loads(line, object_pairs_hook=_unique_json_object)
            lines.append(json.dumps(sanitize(value), ensure_ascii=True, allow_nan=False))
        except _DuplicateJSONKey:
            # Raw fallback cannot safely sanitize arbitrary JSON escape spellings.
            lines.append(json.dumps("public event omitted: duplicate JSON object key"))
        except _RedactedJSONKeyCollision:
            redactions = previous_redactions
            lines.append(json.dumps(
                "public event omitted: JSON object key collision after profile redaction"))
        except (ValueError, RecursionError):
            # Retain malformed or overnested lines as disclosed derivatives,
            # redacting escaped profile prefixes without counting abandoned work.
            redactions = previous_redactions
            escaped_home = str(Path.home()).replace("\\", "\\\\")
            redactions += line.count(escaped_home)
            value = line.replace(escaped_home, "[USER_PROFILE]")
            lines.append(sanitize(value))
    public = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    (destination / "events.jsonl").write_bytes(public)
    record = {"schema": "wsp-public-event-derivative/1",
              "raw_events_sha256": hashlib.sha256(raw).hexdigest(),
              "public_events_sha256": hashlib.sha256(public).hexdigest(),
              "redactions": redactions,
              "transform": transform,
              "raw_bytes_published": False,
              "scope": "public_derivative_replay;_raw_identity_disclosed_not_publicly_replayed"}
    write_json(destination / "event-publication.json", record)
    return record
