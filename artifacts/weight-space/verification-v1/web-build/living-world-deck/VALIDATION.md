# Validation record

**Validation date: November 4, 2026.** This record covers the eight-slide, 129-second presentation, its portable server, the bundled learning labs, and the recorded gameplay. It distinguishes automated checks from observed browser interactions and identifies the checks still awaiting final confirmation.

## Official composition check

The official **HyperFrames 0.8.181** check completed with `ok: true` and `browserSkipped: false`. The browser phase ran successfully. The final report is preserved verbatim in [sources/verification/hyperframes-check.json](sources/verification/hyperframes-check.json).

| Check | Verified result |
| --- | --- |
| HTML/composition lint | Zero errors, warnings, or informational findings |
| Browser runtime | Zero errors, warnings, or informational findings |
| Layout | Zero findings across all eight requested positions |
| Contrast | **125 of 125 checks passed**, with zero findings |
| Composition duration | **129 seconds** |
| Motion assertions | Disabled; no motion-assertion pass is claimed |

Layout sampling positions were **1, 13, 58, 70, 82, 94, 106, and 118 seconds**. Contrast sampling and the **five saved snapshots** used **1, 13, 70, 94, and 118 seconds**. Eight layout positions do not mean eight saved images. The final recording and policy-lab snapshots were also visually inspected.

The report ID is `0dab98dd-a628-4167-a56c-640f3bbba5bc`. Its snapshot paths refer to the original authoring project's generated artifacts, under run `7f7755d5-20f2-4bb4-9c02-769664693fd5`; those five image files are not included in this portable report copy. The check was not run in strict-warning mode, but there were no warnings to waive.

The root composition declares the full 129-second duration. All eight scene identifiers and timings are valid and contiguous, and the portable wrapper and composition contain identical slideshow manifests. Final staging included the contrast corrections, full-page lab links, and removal of temporary pointer diagnostics.

## Code and numerical checks

Strict TypeScript compilation passed for the lab. **All 13 lab tests passed**, with no failures or skipped tests, and the portable lab bundle built successfully. The machine-readable record and exact commands are in [lab/verification.json](lab/verification.json); implementation and reproduction notes are in [lab/README.md](lab/README.md).

These tests cover independently calculated dense multiplication, ReLU and softmax arithmetic; replay using the actual world simulation; supported comparison horizons; checkpoint preservation; and experiment export and state restoration. The bundled lab uses the game's actual policy and simulation code. It does not substitute displayed demonstration values for computation.

The reference experiment uses **Iron Ridge, seed 42, year 10**:

| Quantity | Verified value |
| --- | --- |
| Conservation output weight `W2[2]` | Original **−0.5**; edited **4** |
| Selected hidden activation | **1** |
| Change in the conservation score | **+4.5** |
| Original conservation probability | **0.059019727625896606** (about **5.9%**) |
| Edited conservation probability | **0.637442943401737** (about **63.7%**) |
| Three-year scheduled conservation selections | **4 → 22** |
| Three-year forest difference, edited minus original | **0.009814642091766634**, or about **+0.98 percentage points** |
| Three-year population difference, edited minus original | **−0.13827262516454653** |

The unchanged-policy control produces identical futures and preserves the starting checkpoint. The edited result is one reproducible paired scenario, not an average causal effect. Scheduled selections do not guarantee successful actions, and a higher conservation probability does not guarantee improvement in every outcome.

JavaScript syntax checks passed for the runtime and server. **All five server integration tests passed**. They cover same-origin asset serving, HTML/module/source MIME types, HEAD and OPTIONS behavior, exact bounded/open/suffix video byte ranges, invalid ranges, and malformed, missing, hidden, or traversal paths. The local server suite runs with `node runtime/server.test.mjs` from this presentation folder. Lab TypeScript and source-test commands require the original repository and its development dependencies.

## Observed browser interactions

The full-page lab at `lab/index.html` reuses the same bundled model and UI command handlers as the embedded panels. Browser verification confirmed the following:

| Interaction | Observed result |
| --- | --- |
| Apply the conservation preset | Weight changed **−0.5 → 4** and conservation probability changed **5.9% → 63.7%** |
| Increment and decrement the weight | Controls updated the weight and recomputed the displayed policy |
| Change temperature **1.35 → 2.5** after the preset | Conservation probability decreased to about **38.2%**, while its score remained unchanged |
| Open the network inspector and inspect connections | Inspector and connection views opened and displayed the policy arithmetic |
| Restore the edited weight and reset the lab | Original values and the starting experiment configuration were restored |
| Run a one-year comparison with an unchanged policy | Displayed differences were zero |
| Run the actual three-year edited-policy comparison | Browser results matched the verified numerical fixture above |
| Edit parameters after a completed comparison | Stale results were invalidated |
| Change the trajectory metric | The selected trajectory view updated |
| Open the method dialog | Method and interpretation information was available |
| Start and cancel a ten-year comparison | The run stopped without presenting partial metrics as completed results |

The JSON download button was clicked. **Receipt of a downloaded file was not verified.** Export data and state restoration were covered by the numerical tests; that does not certify browser download delivery.

In the embedded presentation, filling the range input through automation worked. For automated pointer clicks, instrumentation showed synthetic events dispatched to the outer `IFRAME` element and no pointer event entering its inner document. **Native clicks inside the embedded iframe are therefore not certified by this browser run.** This is an automation-routing limitation, not evidence of a lab click-handler defect. The successful full-page interactions above exercise the same model and handlers without that iframe route.

## Presenter, audience, and video verification

Choosing **Present** displayed editable speaker notes and the correct up-next notes. It did not create an audience tab in the in-app browser used for this review. An audience view was therefore opened manually at the same presentation path with `?mode=audience`.

With that audience view open, **Next** navigated both views to slides 2 and 4. Filling the embedded `W2` slider with **4** synchronized the audience's conservation probability to **63.7%** and its displayed score contribution to **+4.5**. This verifies navigation and lab-input synchronization through the tested path; automatic audience-window creation remains unverified.

The fullscreen control changed to **Exit full screen**, visibly expanded the stage, and returned it to the previous layout when toggled again. `document.fullscreenElement` remained `null`. The control and layout toggle are verified; operating-system/browser fullscreen is not certified by that observation.

The bundled MP4 was also opened directly in the browser's native video viewer. Its actual video element reported a duration of **36.133333 seconds**, a playhead advancing from zero to **18.61534 seconds**, `paused: false`, `readyState: 4`, decoded dimensions of **1280 × 720**, and `error: null`. The corresponding screenshot showed the recorded world at year 26. Playback subsequently reached the end with `currentTime: 36.133333`, `ended: true`, and `error: null`.

This verifies playback of the bundled movie through the direct video route. Native play/pause, seeking, caption toggling, and stopping on slide departure **inside the player iframe were not automated end to end**, owing to the pointer-routing limitation described above. No pass for those specific embedded controls is claimed here.

## Source integrity and recording provenance

The package integrity audit verifies the required local assets, eight-scene continuity and manifest agreement, the four copied source files against [sources/manifest.json](sources/manifest.json), and the movie against the SHA-256 in [assets/recording.json](assets/recording.json). The source-snapshot and recording checksum checks passed. The bundled framework versions, origins, licenses, and hashes are documented in [vendor/THIRD-PARTY-NOTICES.md](vendor/THIRD-PARTY-NOTICES.md).

The movie contains **actual gameplay** captured from the running game, with the following recorded provenance:

| Property | Recorded value |
| --- | --- |
| Source captures | **268 screenshots**, with capture timestamps |
| Active captured time | **72.004 seconds** |
| Encoded movie duration | **36.133333 seconds** |
| Recording retime | **2×** the measured capture cadence, with pauses between views omitted |
| Game simulation setting | **32×** |
| World | **Seed 42; years 1–49; five founding societies** |
| Weight intervention | **None**; this footage is observational |
| Encoded format | **1280 × 720, H.264, 30 fps, silent** |
| Movie SHA-256 | `8f788b613f514195c251edaa955da568904cec1810e51e1a0b6930007a19e902` |

The 32× game setting and 2× recording retime describe different operations. No 64× effective simulation-speed claim is made. The recording is not a controlled comparison of a weight edit; the interactive labs provide that separate experiment.

## Portable delivery scope

The portable folder includes the local HyperFrames player and slideshow, GSAP, the compiled labs, movie, poster, captions, source snapshots, licenses, and this validation evidence. Viewing them through the bundled Node server requires no package installation or internet connection once the folder and Node are available.

The full game is a separate application and its link requires the game server. The original research article is an external destination. The labs operate on detached bundled state and do not call native Laya or modify the separately running game's saved world. General natural-language specification-to-weight synthesis and gradient training during play are outside the verified implementation; see [sources/READING.md](sources/READING.md).
