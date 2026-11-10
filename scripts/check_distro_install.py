"""Check the installed Distro companion with a fresh fixture and official MCP client.

Run with the installed wheel's Python -I, outside the source checkout. No host
model, decision, provider, historical study, or deployment is invoked.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import rlcd.weight_space as wsp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="New synthetic workspace directory")
    args = parser.parse_args()
    package = Path(wsp.__file__).resolve()
    assert sys.flags.isolated, "Run the installed interpreter with -I"
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    result = subprocess.run(
        [sys.executable, "-I", "-m", "rlcd.weight_space", "distro", "demo",
         "--workspace", str(args.workspace)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    demo = json.loads(result.stdout)
    root = Path(demo["workspace"])
    plugin = Path(demo["plugin"]["plugin_path"])
    server = json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["weight-space"]
    skill = (plugin / "skills/weight-space-governance/SKILL.md").read_bytes()
    assert skill == files("rlcd.weight_space").joinpath("distro_skill.md").read_bytes()
    assert Path(server["command"]).resolve() == Path(sys.executable).resolve()
    assert server["args"] == ["-I", "-m", "rlcd.weight_space.distro_mcp", "--workspace", str(root)]

    def payload(response):
        assert response.is_error is False, response
        data = response.structured_content
        assert data == json.loads(response.content[0].text)
        assert data["production_authorized"] is False
        assert data["scientific_authorized"] is False
        assert data["local_pilot_eligible"] is False
        return data

    async def exercise():
        parameters = StdioServerParameters(command=server["command"], args=server["args"])
        with anyio.fail_after(20):
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=10.0) as session:
                    await session.initialize()
                    names = {tool.name for tool in (await session.list_tools()).tools}
                    assert names == {"wsp_inspect_change", "wsp_prepare_review",
                                     "wsp_review_status", "wsp_export_review"}
                    draft = payload(await session.call_tool("wsp_inspect_change", {"change": "change.json"}))
                    assert draft["status"] == "draft"
                    prepared = payload(await session.call_tool("wsp_prepare_review", {
                        "change": "change.json", "out": "review/requested.json",
                    }))
                    assert prepared["status"] == "requested"
                    pins = {"expected_request_sha256": prepared["request_sha256"],
                            "expected_head_sha256": prepared["head_sha256"]}
                    arguments = {"change": "change.json", "bundle": "review/requested.json", **pins}
                    status = payload(await session.call_tool("wsp_review_status", arguments))
                    assert status["status"] == "requested"
                    exported = payload(await session.call_tool("wsp_export_review", {
                        **arguments, "out": "review/exported.json",
                    }))
                    assert exported["events"] == status["events"]
                    assert (root / "review/requested.json").read_bytes() == (root / "review/exported.json").read_bytes()
                    invalid = await session.call_tool("wsp_review_status", {
                        **arguments, "expected_head_sha256": "0" * 64,
                    })
                    assert invalid.is_error is True
                    return sorted(names)

    tools = anyio.run(exercise)
    print(json.dumps({
        "schema": "wsp-distro-installed-check/1", "status": "passed",
        "distribution_version": version("weight-space-programming-framework"),
        "mcp_version": version("mcp"), "isolated_python": True,
        "tools": tools, "before_queue": demo["before_queue"], "after_queue": demo["after_queue"],
        "supplied_protected_scores_unchanged": demo["supplied_protected_scores_unchanged"],
        "review_status": "requested", "local_pilot_eligible": False,
        "production_authorized": False, "scientific_authorized": False,
        "native_host_discovery_checked": False, "live_host_model_session_checked": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
