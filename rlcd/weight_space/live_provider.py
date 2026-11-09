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
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from .model import canonical_bytes, digest


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


def _kill_tree(process: subprocess.Popen) -> None:
    """Timeout the entire local transport/worker tree, not just its wrapper."""
    if process.poll() is not None:
        return
    try:
        if __import__("os").name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=15)
        else:
            import os
            import signal
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()


def _communicate_bounded(process: subprocess.Popen, *, timeout: float, input=None):
    """Retain partial output even if descendants keep a pipe open after timeout."""
    try:
        return (*process.communicate(input, timeout=timeout), False, True)
    except subprocess.TimeoutExpired as expired:
        _kill_tree(process)
        try:
            output = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                output = process.communicate(timeout=5)
            except subprocess.TimeoutExpired as incomplete:
                return (incomplete.output or expired.output or b"",
                        incomplete.stderr or expired.stderr or b"", True, False)
        return (*output, True, True)


def parse_events(output: str) -> tuple[Any, dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
    """Parse the final message, and inspect tool events at every lifecycle stage."""
    final, usage, unapproved, errors, diagnostics = None, {}, [], [], []
    for line in output.splitlines():
        try:
            event = json.loads(line)
            if type(event) is not dict:
                raise ValueError("Event must be an object")
            canonical_bytes(event)
            item = event.get("item")
            if isinstance(item, dict) and event.get("type", "").startswith("item."):
                kind = item.get("type", "unknown")
                if kind == "error":
                    # CLI warnings use error items too; a failed turn/exit is
                    # classified separately. Neither is a native tool invocation.
                    diagnostics.append({"stage": event["type"], "item": item})
                elif kind not in ("reasoning", "agent_message"):
                    unapproved.append(f"{event['type']}:{kind}")
                if event.get("type") == "item.completed" and kind == "agent_message":
                    final = item.get("text")
            if event.get("type") == "turn.completed":
                usage = event.get("usage", {})
            if event.get("type") in ("error", "turn.failed"):
                errors.append(event["type"])
        except (ValueError, TypeError):
            errors.append("invalid_event")
    try:
        response = json.loads(final) if isinstance(final, str) else None
        canonical_bytes(response)
    except ValueError:
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

    def __post_init__(self) -> None:
        if (not isinstance(self.model, str) or not self.model.strip()
                or type(self.timeout_seconds) is not int or self.timeout_seconds <= 0
                or type(self.tool_free) is not bool
                or self.sandbox not in (None, "read-only", "workspace-write")
                or (self.tool_free and self.sandbox == "workspace-write")
                or self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max")):
            raise ValueError("Invalid live-provider configuration")

    def configuration(self) -> dict[str, Any]:
        return {"schema": "wsp-codex-provider-config/1", "provider": "openai",
                "requested_model": self.model, "reasoning_effort": self.reasoning_effort,
                "tool_free": self.tool_free, "user_config_loaded": False,
                "ephemeral": True, "timeout_seconds": self.timeout_seconds,
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
        output, stderr = "", ""
        options: dict[str, Any] = {}
        if __import__("os").name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            options["start_new_session"] = True
        try:
            if cli is None:
                raise FileNotFoundError("Codex CLI unavailable")
            executable_sha256 = hashlib.sha256(Path(cli).read_bytes()).hexdigest()
            version = subprocess.run([cli, "--version"], capture_output=True, text=True,
                                     encoding="utf-8", timeout=15, check=True).stdout.strip()
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                       errors="replace", **options)
            output, stderr, timed_out, cleanup_complete = _communicate_bounded(
                process, input=prompt, timeout=self.timeout_seconds)
            returncode = process.returncode
            if timed_out:
                status, error = "timeout", "provider_deadline"
        except subprocess.TimeoutExpired:
            # The bounded version probe timed out before a worker was started.
            status, error = "timeout", "provider_deadline"
        except (OSError, subprocess.CalledProcessError) as failure:
            error = "transport_startup_failed:" + type(failure).__name__
        elapsed = time.perf_counter() - tick
        output = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
        stderr = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
        # Raw stderr can contain transport/environment paths. It stays private.
        (directory / "events.jsonl").write_text(output, encoding="utf-8", newline="\n")
        (directory / "transport.stderr.private").write_text(stderr, encoding="utf-8", newline="\n")
        response, usage, unapproved, event_errors, diagnostics = parse_events(output)
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
                   "events_sha256": hashlib.sha256(output.encode()).hexdigest(),
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
        if attempt["status"] != "completed":
            status = attempt["status"]
            break
        try:
            broker.record_usage(*_usage_counts(attempt["usage"]))
            actions = attempt["response"]["actions"]
            if not isinstance(actions, list) or not actions or len(actions) > 8:
                raise ValueError("Invalid action batch")
            history.append({"role": "proposer", "data": attempt["response"]})
            for action in actions:
                if type(action) is not str:
                    raise ValueError("Action must be encoded as a JSON string")
                result = broker.request(json.loads(action))
                history.append({"role": "broker", "data": result})
                if broker.is_sealed:
                    status = "sealed"
                    break
            if broker.is_sealed:
                break
        except (ValueError, KeyError, TypeError):
            status = "invalid"
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
    """Publish a disclosed event derivative; raw transport bytes stay private."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("request.json", "response-schema.json", "receipt.json"):
        shutil.copyfile(source / name, destination / name)
    raw = (source / "events.jsonl").read_bytes()
    redactions = 0
    profile = re.compile(r"(?:[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s]+|/(?:home|Users)/[^/\s]+)")

    def sanitize(value: Any) -> Any:
        nonlocal redactions
        if isinstance(value, str):
            result, count = profile.subn("[USER_PROFILE]", value)
            redactions += count
            return result
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        return value

    lines = []
    for line in raw.decode("utf-8").splitlines():
        try:
            value = json.loads(line)
            lines.append(json.dumps(sanitize(value), ensure_ascii=False, allow_nan=False))
        except ValueError:
            # Failed transports may emit a non-JSON line. Retain it as a
            # disclosed derivative, including escaped profile prefixes.
            value = line.replace(str(Path.home()).replace("\\", "\\\\"), "[USER_PROFILE]")
            lines.append(sanitize(value))
    public = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    (destination / "events.jsonl").write_bytes(public)
    record = {"schema": "wsp-public-event-derivative/1",
              "raw_events_sha256": hashlib.sha256(raw).hexdigest(),
              "public_events_sha256": hashlib.sha256(public).hexdigest(),
              "redactions": redactions,
              "transform": "JSON_reserialization_and_user_profile_prefix_redaction",
              "raw_bytes_published": False,
              "scope": "public_derivative_replay;_raw_identity_disclosed_not_publicly_replayed"}
    write_json(destination / "event-publication.json", record)
    return record
