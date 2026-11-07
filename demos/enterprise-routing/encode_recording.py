"""Encode captured browser frames at their actual presentation times.

Requires ffmpeg on PATH. Capture input stays local; the exported provenance
contains repository-relative source names and their hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = ("demos/enterprise-routing/index.html", "demos/enterprise-routing/style.css",
                "demos/enterprise-routing/app.js", "demos/enterprise-routing/serve.py",
                "demos/enterprise-routing/encode_recording.py",
                "examples/enterprise_support_routing.py",
                "examples/data/enterprise_support_tickets.json")
STORY_SOURCE_NAMES = tuple("artifacts/explanations/enterprise-story/" + name for name in
                           ("index.html", "story.css", "story.js", "story-data.js", "source.json",
                            "story.md", "subtitles.vtt", "subtitles.srt")) + (
    "demos/enterprise-routing/serve.py", "demos/enterprise-routing/build_story_source.py",
    "demos/enterprise-routing/encode_recording.py", "examples/enterprise_support_routing.py",
    "examples/data/enterprise_support_tickets.json")


def ffconcat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="local JSON frame/timing receipt")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/enterprise/support-routing/demo")
    parser.add_argument("--source-set", choices=("technical", "story"), default="technical")
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    frames = capture["frames"]
    if not frames or capture["capture_method"] != "Chrome DevTools Protocol Page.screencastFrame":
        raise ValueError("Expected actual browser screencast frames")
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["ffconcat version 1.0"]
    for i, frame in enumerate(frames):
        path = args.capture.parent / frame["file"]
        if not path.is_file() or not path.resolve().is_relative_to(args.capture.parent.resolve()):
            raise ValueError("Capture frame is missing or outside the local capture directory")
        end = frames[i+1]["time"] if i+1 < len(frames) else capture["duration_seconds"]
        duration = end - frame["time"]
        if duration <= 0:
            raise ValueError("Frame presentation times must increase")
        lines += [f"file '{ffconcat_path(path)}'", f"duration {duration:.6f}"]
    lines += [f"file '{ffconcat_path(args.capture.parent / frames[-1]['file'])}'"]
    concat = args.capture.parent / "frames.ffconcat"
    concat.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    video = args.out / "walkthrough.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", str(concat), "-vf",
                    "fps=24,scale=in_range=full:out_range=tv,format=yuv420p", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                    "-color_range", "tv", "-t", str(capture["duration_seconds"]),
                    "-movflags", "+faststart", str(video)], check=True)
    metadata = {"schema": "enterprise-routing-screen-recording/1",
                "capture_method": capture["capture_method"],
                "capture_date": capture["capture_date"],
                "viewport": capture["viewport"], "duration_seconds": capture["duration_seconds"],
                "audio": "Silent; explanations appear in the captured interface",
                "timing": "Captured browser frames retain their observed presentation timing; unchanged frames are held between events, encoded at 24 fps",
                "editing": "No generated scenes or substituted UI footage; browser content capture excludes browser chrome and workstation desktop",
                "chapters": capture["chapters"],
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in (STORY_SOURCE_NAMES if args.source_set == "story" else SOURCE_NAMES)},
                "scope": "Synthetic structured-field routing example. No trained text classifier, customer deployment, or measured financial impact."}
    (args.out / "recording.json").write_bytes((json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
    print(f"Encoded {video.name}: {capture['duration_seconds']:.1f}s, {video.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
