"""Generate a workspace-local companion for Distro's native coding hosts."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
import stat
import sys

from .distro import DistroWorkspace


def _safe_stat(path: Path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or (
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise ValueError(f"Symbolic links and reparse points are not allowed: {path}")
    return info


def setup_plugin(workspace: Path) -> dict:
    """Create the fixed companion, or reuse its exact existing contents.

    Paths are checked before I/O; concurrent directory replacement is outside
    this adapter's filesystem guarantees.
    """
    root = DistroWorkspace(workspace).root
    plugin = root / ".runs" / "distro-plugin"
    server = {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-I", "-m", "rlcd.weight_space.distro_mcp", "--workspace", str(root)],
    }
    metadata = {
        "name": "weight-space-governance",
        "version": "0.1.0",
        "license": "MIT",
        "description": "Proof-first weight-space review with explicit human governance.",
    }
    manifests = {
        ".claude-plugin/plugin.json": metadata,
        ".github/plugin/plugin.json": {
            **metadata,
            "skills": ["./skills/weight-space-governance"],
            "mcpServers": {"weight-space": server},
        },
        ".mcp.json": {"mcpServers": {"weight-space": server}},
    }
    contents = {
        Path(name): (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        for name, value in manifests.items()
    }
    contents[Path("skills/weight-space-governance/SKILL.md")] = (
        files("rlcd.weight_space").joinpath("distro_skill.md").read_bytes()
    )
    directories = {parent for relative in contents for parent in relative.parents}
    ordered_directories = sorted(directories, key=lambda path: (len(path.parts), str(path)))
    directory_paths = [root / ".runs", *(plugin / relative for relative in ordered_directories)]

    # Preflight the complete companion before creating any missing contents.
    for path in directory_paths:
        info = _safe_stat(path)
        if info is not None and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Expected a plugin directory: {path}")
    missing = []
    for relative, content in contents.items():
        path = plugin / relative
        info = _safe_stat(path)
        if info is None:
            missing.append((path, content))
        elif not stat.S_ISREG(info.st_mode) or info.st_size != len(content) or path.read_bytes() != content:
            raise ValueError(f"Existing plugin content differs; refusing to replace it: {path}")
    allowed = directories | contents.keys()
    for relative in ordered_directories:
        directory = plugin / relative
        if directory.exists():
            for child in directory.iterdir():
                _safe_stat(child)
                if child.relative_to(plugin) not in allowed:
                    raise ValueError(f"Unexpected existing plugin content: {child}")

    for path in directory_paths:
        path.mkdir(exist_ok=True)
    for path, content in missing:
        with path.open("xb") as stream:
            stream.write(content)
    return {
        "schema": "wsp-distro-plugin/1",
        "workspace": str(root),
        "plugin_path": str(plugin),
        "launch_args": {
            host: ["distro", host, "--plugin-dir", str(plugin)]
            for host in ("claude", "copilot")
        },
        "production_authorized": False,
        "scientific_authorized": False,
    }
