# Enterprise routing demo

A guided interface and real browser recording of the Weight Space Programming
Framework changing a support-routing policy. The walkthrough makes the task
contract, candidate comparison, literal weight edit, protection evidence, sealed
evaluation, impact, and inverse restoration visible.

[Watch the screen recording](../../artifacts/enterprise/support-routing/demo/walkthrough.mp4)
· [Open the video player locally](../../artifacts/enterprise/support-routing/demo/index.html)
· [Read the enterprise use case](../../docs/use-cases/enterprise-support-routing.md)

![Captured interface showing the observed routing change, measured protection results, and potential enterprise value.](../../artifacts/enterprise/support-routing/demo/poster.jpg)

## Run the interface

From the repository root, with Python 3.11 or newer:

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -e .
python demos/enterprise-routing/serve.py
```

Open `http://127.0.0.1:8876`. Stop the server with Ctrl+C. Use `--port 8877`
if the default port is occupied. This is a loopback development server, with
no external model provider, credentials, or frontend build step.

Choose a scenario to run the framework, or use **Run framework** to recompute it.
Then follow the six walkthrough steps. Select candidate rows to inspect each
alternative's public failures. **Download full evidence** saves the actual
JSON bundle for the current run.

The server executes compilation, public admission, selection, sealing, and
final evaluation together. Navigation reveals their completed receipts;
**Make the decision** is a walkthrough step, not a separate evaluator call.
Candidate-row selection changes the inspected record, not the sealed candidate.

**Restore parent model** makes a separate Python request. It reconstructs the
admitted candidate, applies its bound inverse, compares the restored artifact
with the parent hash, and evaluates the restored routing. It restores a model
artifact; it does not undo external ticket assignments or control a live system.

## What to explore

| Scenario | Framework outcome | Concept exposed |
| --- | --- | --- |
| Policy change | Admitted for synthetic trial | A target improvement must also satisfy protection and the final gate |
| Conflicting protection | Abstained; final evaluator not called | A target that must remain fixed makes the requested change infeasible |
| Unseen symptom | Sealed candidate rejected after final evaluation | A public pass does not establish feature coverage on the final split |

For the successful policy change, the protected edit changes the Platform /
telemetry-outage coefficient from `0.25` to `3.22029702970297`. Final recognized
outage targets move from General Support to Platform Operations, with a minimum
winning margin of `1.220297` raw score units. Maximum sampled protected score
drift is `0`.

These are synthetic structured fields and a supplied linear head. The exact
retention certificate covers the supplied fixed-feature anchors and their span
under real-valued linear semantics. It does not prove floating-point runtime
execution or text understanding. The interface separates those observed results
from possible enterprise benefits: handling time, savings, and customer outcomes
have not been measured.

## Recording and reproducibility

The silent MP4 captures actual interactions with the local browser interface.
Explanations are on screen. Chrome DevTools Protocol `Page.screencastFrame`
supplied JPEG frames and presentation timestamps; unchanged browser frames are
held between events, then encoded as H.264 at 24 fps. Browser chrome and the
workstation desktop are outside the capture. No generated scenes replace the UI.

[The recording receipt](../../artifacts/enterprise/support-routing/demo/recording.json)
contains its duration, chapters, video hash, and published source snapshot hashes.
Those hashes describe the files supplied with the demo; the receipt is a local
provenance record, not independent runtime attestation. Private raw capture paths
and review screenshots stay outside the publication.

`encode_recording.py` encodes an existing real capture; it does not drive the
browser. Given a local JSON receipt with `capture_method`, `capture_date`,
`viewport`, `duration_seconds`, `chapters`, and a `frames` list containing
relative JPEG filenames and increasing presentation times:

```bash
python demos/enterprise-routing/encode_recording.py .runs/demo-capture/capture.json
```

This export needs `ffmpeg` on PATH. Running the interface requires only the core
framework's Python dependencies. The published evidence can also be reproduced
and verified through the commands in the enterprise use-case document.

The demo's scoped visual tokens and component guidance live in
[DESIGN.md](DESIGN.md). Its interface uses local CSS, system fonts, readable
tables, and responsive navigation.
