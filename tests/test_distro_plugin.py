"""Native companion configuration checks without model or provider execution."""
from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from rlcd.weight_space.distro_plugin import setup_plugin


def _snapshot(directory: Path):
    return {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


def test_native_manifests_use_exact_interpreter_workspace_and_packaged_skill(tmp_path):
    result = setup_plugin(tmp_path)
    root = tmp_path.resolve()
    plugin = root / ".runs/distro-plugin"
    assert result == {
        "schema": "wsp-distro-plugin/1", "workspace": str(root), "plugin_path": str(plugin),
        "launch_args": {host: ["distro", host, "--plugin-dir", str(plugin)]
                        for host in ("claude", "copilot")},
        "production_authorized": False, "scientific_authorized": False,
    }
    assert result["production_authorized"] is False
    assert result["scientific_authorized"] is False
    claude = json.loads((plugin / ".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    copilot = json.loads((plugin / ".github/plugin/plugin.json").read_text(encoding="utf-8"))
    mcp = json.loads((plugin / ".mcp.json").read_text(encoding="utf-8"))
    assert claude["name"] == copilot["name"] == "weight-space-governance"
    assert claude["version"] == copilot["version"]
    assert claude["license"] == copilot["license"] == "MIT"
    assert copilot["skills"] == ["./skills/weight-space-governance"]
    assert mcp["mcpServers"] == copilot["mcpServers"] == {
        "weight-space": {
            "type": "stdio", "command": sys.executable,
            "args": ["-I", "-m", "rlcd.weight_space.distro_mcp", "--workspace", str(root)],
        },
    }
    skill = (plugin / "skills/weight-space-governance/SKILL.md").read_bytes()
    assert skill == files("rlcd.weight_space").joinpath("distro_skill.md").read_bytes()
    text = skill.decode("utf-8")
    for tool in ("wsp_inspect_change", "wsp_prepare_review", "wsp_review_status", "wsp_export_review"):
        assert tool in text
    for boundary in ("independently retained pins", "untrusted", "governance decide",
                     "production_authorized", "scientific_authorized", "Proof first"):
        assert boundary in text


def test_repeated_setup_does_not_rewrite_matching_files(tmp_path):
    original = setup_plugin(tmp_path)
    plugin = Path(original["plugin_path"])
    before = _snapshot(plugin)
    assert setup_plugin(tmp_path) == original
    assert _snapshot(plugin) == before


@pytest.mark.parametrize("relative", [".mcp.json", "skills/weight-space-governance/SKILL.md"])
def test_differing_existing_content_is_preserved_and_prevents_all_writes(tmp_path, relative):
    plugin = tmp_path / ".runs/distro-plugin"
    existing = plugin / relative
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"Do not replace this existing file.\n")
    before = _snapshot(plugin)
    with pytest.raises(ValueError, match="differs"):
        setup_plugin(tmp_path)
    assert _snapshot(plugin) == before


def test_unowned_existing_plugin_content_is_rejected_without_changes(tmp_path):
    plugin = Path(setup_plugin(tmp_path)["plugin_path"])
    (plugin / "hooks.json").write_text("{}", encoding="utf-8")
    before = _snapshot(plugin)
    with pytest.raises(ValueError, match="Unexpected existing plugin content"):
        setup_plugin(tmp_path)
    assert _snapshot(plugin) == before


@pytest.mark.parametrize("relative", [".runs", ".runs/distro-plugin", ".runs/distro-plugin/.github"])
def test_file_in_place_of_setup_directory_is_rejected_before_writes(tmp_path, relative):
    existing = tmp_path / relative
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="directory"):
        setup_plugin(tmp_path)
    assert existing.read_bytes() == b"preserve"


@pytest.mark.parametrize("relative", [
    ".runs", ".runs/distro-plugin", ".runs/distro-plugin/.github",
    ".runs/distro-plugin/skills/weight-space-governance",
])
def test_linked_setup_directory_is_rejected_without_writing_target(tmp_path, relative):
    root, outside = tmp_path / "workspace", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    linked = root / relative
    linked.parent.mkdir(parents=True, exist_ok=True)
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symbolic links unavailable: {error}")
    with pytest.raises(ValueError, match="[Ss]ymbolic link|reparse"):
        setup_plugin(root)
    assert list(outside.iterdir()) == []


def test_linked_generated_file_is_rejected_without_replacing_target(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    target = tmp_path / "outside.json"
    target.write_bytes(b"preserve")
    linked = root / ".runs/distro-plugin/.mcp.json"
    linked.parent.mkdir(parents=True)
    try:
        linked.symlink_to(target)
    except OSError as error:
        pytest.skip(f"File symbolic links unavailable: {error}")
    with pytest.raises(ValueError, match="[Ss]ymbolic link|reparse"):
        setup_plugin(root)
    assert target.read_bytes() == b"preserve"


def test_reparse_attribute_is_rejected_even_without_symlink_mode(tmp_path, monkeypatch):
    original = Path.lstat
    reparse = tmp_path / ".runs"

    def lstat(path, *args, **kwargs):
        if path == reparse:
            return SimpleNamespace(st_mode=stat.S_IFDIR,
                                   st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        setup_plugin(tmp_path)
    assert not reparse.exists()
