"""Read-only technical measurement of the final encoded AAC; no listening review."""
from __future__ import annotations

import array
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "renders" / "first-conquest-90s.mp4"
EXPECTED_SHA256 = "29ee64ca17aaada840508f42527739400c5b868cd3cf10a2b5a20b5d3a880991"
COMMANDS: list[dict] = []


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def command(name: str, args: list[str], *, binary=False):
    COMMANDS.append({"name": name, "argv": args, "windows_command_line": subprocess.list2cmdline(args)})
    (HERE / "commands.json").write_text(json.dumps(COMMANDS, indent=2) + "\n", encoding="utf-8")
    (HERE / "commands.txt").write_text("\n\n".join(f"{item['name']}\n{item['windows_command_line']}" for item in COMMANDS) + "\n", encoding="utf-8")
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stderr = result.stderr.decode("utf-8", errors="replace")
    stdout = "" if binary else result.stdout.decode("utf-8", errors="replace")
    (HERE / f"{name}.log").write_text(f"Command: {subprocess.list2cmdline(args)}\nExit code: {result.returncode}\n\nSTDERR\n{stderr}\nSTDOUT\n{stdout}", encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{name} failed; see saved log")
    return result.stdout if binary else (stdout, stderr)


def db(value: float):
    return round(20 * math.log10(value), 4) if value > 0 else None


def statistics(samples, start_frame: int, end_frame: int, channels: int):
    segment = samples[start_frame * channels:end_frame * channels]
    rms = math.sqrt(math.fsum(value * value for value in segment) / len(segment))
    peak = max(abs(value) for value in segment)
    return {"start_seconds": start_frame / 48000, "end_seconds": end_frame / 48000,
            "rms_dbfs": db(rms), "sample_peak_dbfs": db(peak), "sample_peak_linear": peak}


def main():
    digest_before = sha256(SOURCE)
    if digest_before != EXPECTED_SHA256:
        raise RuntimeError(f"Final MP4 identity differs: {digest_before}")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe are required")
    command("ffmpeg-version", [ffmpeg, "-version"])
    probe_stdout, _ = command("ffprobe", [ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_streams", "-show_format", "-of", "json", str(SOURCE)])
    probe = json.loads(probe_stdout)
    (HERE / "ffprobe.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    stream = probe["streams"][0]
    channels = int(stream["channels"])
    sample_rate = int(stream["sample_rate"])
    if channels != 2 or sample_rate != 48000:
        raise RuntimeError(f"Unexpected audio format: {channels} channels / {sample_rate} Hz")
    common = [ffmpeg, "-hide_banner", "-nostats", "-threads", "1", "-i", str(SOURCE),
              "-map", "0:a:0", "-vn", "-sn", "-dn", "-filter_threads", "1"]
    _, loudness_log = command("ffmpeg-loudness", common + ["-af",
        "loudnorm=I=-24:TP=-1:LRA=11:print_format=json", "-f", "null", "NUL"])
    match = re.search(r'\{\s*"input_i"[\s\S]*?\}', loudness_log)
    if not match:
        raise RuntimeError("No loudnorm input measurements found")
    measured = json.loads(match.group())
    _, r128_log = command("ffmpeg-ebur128", common + ["-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "NUL"])
    secondary_ebur128 = {
        "integrated_loudness_lufs": float(re.findall(r"Integrated loudness:\s*I:\s*([-\d.]+)\s*LUFS", r128_log)[-1]),
        "loudness_range_lu": float(re.findall(r"LRA:\s*([-\d.]+)\s*LU", r128_log)[-1]),
        "true_peak_dbtp": float(re.findall(r"Peak:\s*([-\d.]+)\s*dBFS", r128_log)[-1]),
    }
    raw = command("ffmpeg-decode", common + ["-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"], binary=True)
    samples = array.array("f")
    samples.frombytes(raw)
    del raw
    if sys.byteorder != "little":
        samples.byteswap()
    frame_count = len(samples) // channels
    duration = frame_count / sample_rate
    windows = [statistics(samples, start, min(start + 12000, frame_count), channels)
               for start in range(0, frame_count, 12000)]
    with (HERE / "signal-windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    edge_intervals = [(0, .05), (0, .25), (.25, .5), (.5, 1), (1, 2), (2, 4), (4, 6),
                      (duration - 6, duration - 4), (duration - 4, duration - 2),
                      (duration - 2, duration - 1), (duration - 1, duration - .5),
                      (duration - .5, duration - .25), (duration - .25, duration),
                      (duration - .05, duration)]
    edge = [statistics(samples, round(start * sample_rate), round(end * sample_rate), channels)
            for start, end in edge_intervals]
    overall = statistics(samples, 0, frame_count, channels)
    channel_rms = [db(math.sqrt(math.fsum(value * value for value in samples[channel::channels]) / frame_count))
                   for channel in range(channels)]
    digest_after = sha256(SOURCE)
    result = {
        "measurement_type": "Technical decoding and measurement only; not a listening review",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE), "sha256_before": digest_before, "sha256_after": digest_after,
        "source_unchanged": digest_before == digest_after,
        "codec": stream["codec_name"], "sample_rate_hz": sample_rate,
        "channels": channels, "channel_layout": stream.get("channel_layout"),
        "container_duration_seconds": float(probe["format"]["duration"]),
        "audio_stream_duration_seconds": float(stream["duration"]),
        "decoded_pcm_frames": frame_count, "decoded_pcm_duration_seconds": duration,
        "integrated_loudness_lufs": float(measured["input_i"]),
        "true_peak_dbtp": float(measured["input_tp"]),
        "loudness_range_lu": float(measured["input_lra"]),
        "integrated_threshold_lufs": float(measured["input_thresh"]),
        "loudnorm_input_measurements": {key: value for key, value in measured.items() if key.startswith("input_")},
        "secondary_ebur128": secondary_ebur128,
        "aggregate_signal": overall, "channel_rms_dbfs": channel_rms,
        "nonzero_interleaved_samples": sum(value != 0 for value in samples),
        "edge_windows": edge,
        "prepared_wav_reference": {"integrated_loudness_lufs": -24.00, "true_peak_dbtp": -7.70, "loudness_range_lu": 11.20},
        "notes": ["Only loudnorm input_* fields describe the delivered AAC; filter output was discarded.",
                  "PCM decoding includes any AAC end padding; use stream/container duration for presentation length.",
                  "Window levels measure combined stereo RMS and sample peaks, not gated loudness or true peaks.",
                  "Edge-level trends can corroborate fades but do not assess audibility, musical quality, or transitions."]
    }
    (HERE / "measurement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
