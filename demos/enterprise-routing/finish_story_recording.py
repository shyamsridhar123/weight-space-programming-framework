"""Add the credited instrumental bed and subtitle track to a real 90-second capture.

This retains the full captured story at normal speed. It does not render or
replace visual frames. Raw capture and music inputs stay outside publication.
Requires ffmpeg and ffprobe on PATH.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MUSIC_HASH = "57ae03dbc122f0859e0f9a273e59503e34883728ddb5b06761fbf8244b0eb23e"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--capture-receipt", required=True, type=Path)
    parser.add_argument("--music", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/enterprise/support-routing/demo")
    args = parser.parse_args()
    original = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    capture = json.loads(args.capture_receipt.read_text(encoding="utf-8"))
    assert original["video_sha256"] == sha(args.source)
    assert original["duration_seconds"] == capture["duration_seconds"] == 90
    assert capture["capture_method"] == "Chrome DevTools Protocol Page.screencastFrame"
    assert sha(args.music) == MUSIC_HASH, "Music bytes differ from the credited source"
    assert args.source.resolve() != (args.out / "walkthrough.mp4").resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    story = ROOT / "artifacts/explanations/enterprise-story"
    for name in ("subtitles.vtt", "subtitles.srt"):
        shutil.copyfile(story / name, args.out / name)
    chapters = original["chapters"]
    music = {"title": "Aphelion", "artist": "Scott Buckley", "license": "CC BY 4.0",
             "track_url": "https://www.scottbuckley.com.au/library/aphelion/",
             "source_url": "https://www.scottbuckley.com.au/library/wp-content/uploads/2026/04/Aphelion.mp3",
             "license_url": "https://creativecommons.org/licenses/by/4.0/", "source_sha256": MUSIC_HASH,
             "start_seconds": 60, "duration_seconds": 90, "gain_db": -8.17,
             "fade_in_seconds": 2, "fade_out_seconds": 6,
             "attribution": "Aphelion by Scott Buckley - www.scottbuckley.com.au - CC BY 4.0",
             "changes": "Excerpt 60-150 seconds; gain lowered by 8.17 dB; 2-second fade in and 6-second fade out; encoded as AAC."}
    work = ROOT / ".runs/story-finish"
    work.mkdir(parents=True, exist_ok=True)
    lines = [";FFMETADATA1", "title=An outage. The right team. - Weight Space Programming",
             "comment=" + music["attribution"] + " | " + music["changes"],
             "copyright=Music: Scott Buckley, CC BY 4.0 | project-authored material: MIT"]
    for i, chapter in enumerate(chapters):
        end = chapters[i+1]["start_seconds"] if i+1 < len(chapters) else 90
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(chapter['start_seconds']*1000)}",
                  f"END={round(end*1000)}", "title=" + chapter["title"]]
    meta = work / "chapters.ffmetadata"
    meta.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    video = args.out / "walkthrough.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.source),
                    "-ss", "60", "-i", str(args.music), "-i", str(meta), "-i", str(args.out / "subtitles.srt"),
                    "-map", "0:v:0", "-map", "1:a:0", "-map", "3:s:0", "-map_metadata", "2", "-map_chapters", "2",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
                    "-af", "volume=-8.17dB,afade=t=in:st=0:d=2,afade=t=out:st=84:d=6",
                    "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-t", "90", "-movflags", "+faststart", str(video)], check=True)
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-show_chapters", "-of", "json", str(video)], text=True))
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert abs(float(info["format"]["duration"]) - 90) <= .001
    assert vs["nb_frames"] == "2160" and (vs["width"], vs["height"]) == (1280, 720)
    assert len(info["chapters"]) == len(chapters)
    assert any(s["codec_name"] == "mov_text" for s in info["streams"])
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "64", "-i", str(video),
                    "-frames:v", "1", "-q:v", "2", str(args.out / "poster.jpg")], check=True)
    receipt = dict(original)
    receipt.update({"schema": "enterprise-routing-screen-recording/3",
                    "audio": "Quiet cinematic instrumental music; visible subtitles; no spoken narration",
                    "timing": "Full 90-second story at normal speed. Actual event timestamps determine frame timing; unchanged frames are held. Encoded at 24 fps.",
                    "editing": "Real browser capture of an authored SVG explanation. Only the idle lead-in and trailing capture hold were removed; no scene cuts, speed-up, or substituted frames. Music and selectable subtitles added separately.",
                    "capture_observations": {key: capture[key] for key in ("raw_frame_count", "retained_frame_count", "raw_duration_seconds", "lead_in_trim_seconds", "start_alignment", "observed_outcomes")},
                    "video_sha256": sha(video), "poster_sha256": sha(args.out / "poster.jpg"),
                    "source_recording_sha256": sha(args.source), "source_receipt_sha256": sha(args.source_receipt),
                    "subtitles": {"language": "English", "visible_in_capture": True, "embedded_format": "mov_text",
                                  "sidecars_sha256": {name: sha(args.out / name) for name in ("subtitles.vtt", "subtitles.srt")},
                                  "cue_count": 18, "continuous_seconds": [0, 90]},
                    "music": music})
    receipt["source_sha256"]["demos/enterprise-routing/finish_story_recording.py"] = sha(Path(__file__))
    (args.out / "recording.json").write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print("Finished: 90 seconds, 1280x720, 2160 frames, AAC music, English subtitles, nine chapters")

if __name__ == "__main__":
    main()
