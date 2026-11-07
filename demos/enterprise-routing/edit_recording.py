"""Cut the actual browser recording to 90 seconds and add a quiet music bed.

Requires ffmpeg and ffprobe on PATH. Inputs are bound by edit-plan.json hashes.
Temporary footage remains under .runs; public receipts contain no local paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)
    ], text=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original, uncut MP4")
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--music", type=Path, required=True, help="original credited Aphelion MP3")
    parser.add_argument("--plan", type=Path, default=HERE / "edit-plan.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/enterprise/support-routing/demo")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    original = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    music = plan["music"]
    source_hash = sha256(args.source)
    if source_hash != plan["source_video_sha256"] or original["video_sha256"] != source_hash:
        raise ValueError("Source footage does not match the recorded edit plan and original receipt")
    if sha256(args.music) != music["source_sha256"]:
        raise ValueError("Music does not match the credited source bytes")
    if args.source.resolve() == (args.out / "walkthrough.mp4").resolve():
        raise ValueError("Keep the original footage separate from the output")
    if plan["speed"] != 1 or plan["duration_seconds"] != 90:
        raise ValueError("This edit uses normal speed and an exact 90-second duration")
    source_duration = float(probe(args.source)["format"]["duration"])
    clips, chapters, cursor = [], [], 0.0
    for chapter in plan["chapters"]:
        chapters.append({"title": chapter["title"], "start_seconds": cursor})
        for start, end in chapter["ranges"]:
            duration = end - start
            if start < 0 or end > source_duration or duration <= 0:
                raise ValueError("Invalid source range")
            if abs(duration * 24 - round(duration * 24)) > 1e-6:
                raise ValueError("Clip duration must contain a whole number of frames at 24 fps")
            clips.append((start, duration))
            cursor += duration
    if abs(cursor - 90) > 1e-6 or music["duration_seconds"] != 90:
        raise ValueError("The footage and music must cover exactly 90 seconds")
    args.out.mkdir(parents=True, exist_ok=True)
    work_root = ROOT / ".runs"
    work_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="video-edit-", dir=work_root) as work:
        work = Path(work)
        concat_lines = ["ffconcat version 1.0"]
        for i, (start, duration) in enumerate(clips):
            clip = work / f"clip-{i:02}.mp4"
            run("-ss", str(start), "-i", str(args.source), "-frames:v", str(round(duration * 24)),
                "-map", "0:v:0", "-an", "-map_metadata", "-1", "-vf",
                "setpts=PTS-STARTPTS,fps=24,format=yuv420p",
                "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-color_range", "tv", str(clip))
            concat_lines.append(f"file '{clip.name}'")
        concat = work / "clips.ffconcat"
        concat.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        meta_lines = [";FFMETADATA1", "title=Weight Space Programming - Enterprise routing demo",
                      "comment=" + music["attribution"] + " | " + music["changes"],
                      "copyright=Music: Scott Buckley, CC BY 4.0 | project-authored material: MIT"]
        for i, chapter in enumerate(chapters):
            end = chapters[i+1]["start_seconds"] if i+1 < len(chapters) else 90
            meta_lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(chapter['start_seconds']*1000)}",
                           f"END={round(end*1000)}", "title=" + chapter["title"]]
        meta = work / "chapters.ffmetadata"
        meta.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
        audio_filter = (f"volume={music['gain_db']}dB,afade=t=in:st=0:d={music['fade_in_seconds']},"
                        f"afade=t=out:st={90-music['fade_out_seconds']}:d={music['fade_out_seconds']}")
        video = work / "walkthrough.mp4"
        run("-f", "concat", "-safe", "0", "-i", str(concat), "-ss", str(music["start_seconds"]),
            "-i", str(args.music), "-i", str(meta), "-map", "0:v:0", "-map", "1:a:0",
            "-map_metadata", "2", "-map_chapters", "2", "-c:v", "libx264", "-preset", "medium",
            "-crf", "19", "-vf", "fps=24,format=yuv420p", "-color_range", "tv",
            "-frames:v", "2160", "-c:a", "aac",
            "-b:a", "160k", "-ar", "48000", "-af", audio_filter, "-t", "90",
            "-movflags", "+faststart", str(video))
        info = probe(video)
        video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        audio_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
        if (abs(float(info["format"]["duration"]) - 90) > .001
                or video_stream["nb_frames"] != "2160" or audio_stream["codec_name"] != "aac"):
            raise ValueError(f"Incorrect output: duration={info['format']['duration']}, "
                             f"frames={video_stream['nb_frames']}, audio={audio_stream['codec_name']}")
        video.replace(args.out / "walkthrough.mp4")
    metadata = dict(original)
    metadata.update({
        "schema": "enterprise-routing-screen-recording/2",
        "duration_seconds": 90,
        "audio": "Quiet cinematic instrumental background music; explanations remain on screen; no narration",
        "timing": "Normal-speed excerpts of the actual browser capture, joined with hard cuts at 24 fps; pauses shortened",
        "editing": "90-second edit of the previously published real screencast. No generated scenes, substituted UI, or speed-up. Music added separately.",
        "chapters": chapters,
        "video_sha256": sha256(args.out / "walkthrough.mp4"),
        "original_recording": {"video_sha256": source_hash,
                               "receipt_sha256": sha256(args.source_receipt),
                               "duration_seconds": source_duration,
                               "git_commit": plan["source_git_commit"],
                               "path": plan["source_video_path"]},
        "edit": plan,
        "edit_source_sha256": {
            "demos/enterprise-routing/edit-plan.json": sha256(args.plan),
            "demos/enterprise-routing/edit_recording.py": sha256(Path(__file__))
        },
        "music": music
    })
    (args.out / "recording.json").write_bytes((json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
    print("Exported 90-second browser walkthrough with credited AAC background music")


if __name__ == "__main__":
    main()
