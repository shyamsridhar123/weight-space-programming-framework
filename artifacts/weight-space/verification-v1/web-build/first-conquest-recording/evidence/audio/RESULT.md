# Final encoded audio: technical measurement

Measured the delivered `renders/first-conquest-90s.mp4` on 2026-11-06 at 01:20 UTC using FFmpeg 9.0.1. **This is a technical measurement, not a listening review.** The MP4 was decoded for analysis and left unchanged.

| Measurement | Final encoded AAC | Prepared WAV reference |
|---|---:|---:|
| Integrated loudness, FFmpeg loudnorm input | **−24.02 LUFS** | −24.00 LUFS |
| True peak, FFmpeg loudnorm input | **−7.68 dBTP** | −7.70 dBTP |
| Loudness range, FFmpeg loudnorm input | **11.20 LU** | 11.20 LU |
| Codec / rate / layout | **AAC / 48,000 Hz / stereo** | — |
| Container / audio-stream / decoded-PCM duration | **90.000 s / 90.000 s / 90.000 s** | 90 s |

The final AAC differs from the prepared-WAV reference by −0.02 LUFS integrated and +0.02 dB in true peak, with the same reported loudness range. These comparisons use the `loudnorm` **input** measurements. The filter's normalized output was discarded to `NUL`; its `output_*` fields do not describe the delivered file.

A separate FFmpeg `ebur128=peak=true` pass reports **−23.9 LUFS**, **11.3 LU LRA**, and a **−7.7 dB true peak**. The slight numerical disagreement between the two measurement passes is preserved in the logs and JSON rather than treated as exact agreement.

## Signal and ending evidence

- Both channels contain signal: left/right RMS **−25.9735 / −26.4376 dBFS**. Across the 8,640,000 interleaved samples, **8,559,104 are nonzero**.
- Maximum decoded sample magnitude is **0.412901**, or **−7.6831 dBFS**; measured true peak remains **7.68 dB below 0 dBTP**. No sample reaches full scale.
- The first **0.75 seconds are digital silence**. RMS rises from **−118.777 dBFS** over 0.75–1.00 s to **−37.224 dBFS** over 1.00–1.25 s and **−31.937 dBFS** over 1.75–2.00 s. This supports a quiet opening and rising signal; it does not isolate a fade-in envelope from the music's own onset.
- Ending RMS declines from **−24.822 dBFS** over 86–88 s to **−30.393** over 88–89 s, **−36.418** over 89–89.5 s, and **−50.272** over the final 0.25 s. The last 0.05 s measures **−65.658 dBFS**. This is strong measured evidence of the fade-out.
- All decoding and measurement commands completed with exit code 0. These measurements do not assess musical timing, transitions, subjective volume, or perceived quality.

## Evidence and reproduction

- `commands.txt` and `commands.json`: exact executable paths and arguments.
- `ffmpeg-loudness.log`: full primary loudnorm log, including delivered-input measurements.
- `ffmpeg-ebur128.log`: separate EBU R128 / true-peak measurement log.
- `ffmpeg-decode.log`: audio-only decoding command and completion log.
- `ffprobe.json` and `ffprobe.log`: encoded stream and container metadata.
- `signal-windows.csv`: 0.25-second combined-stereo RMS and sample-peak windows; blank dB values mean digital silence.
- `measurement.json`: full numeric results, edge windows, and before/after hashes.
- `ffmpeg-version.log`: exact FFmpeg build.
- `measure_final_audio.py`: reproducible read-only analysis; writes only in this review directory.

SHA-256 before and after measurement, matching the supplied final file identity:

```text
29ee64ca17aaada840508f42527739400c5b868cd3cf10a2b5a20b5d3a880991
```
