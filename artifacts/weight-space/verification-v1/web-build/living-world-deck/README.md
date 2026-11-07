# A living world. A visible mind.

A portable, eight-slide HyperFrames presentation for Crown & Charter. It combines a recording of the running game, an editable neural-policy experiment, paired simulation futures, and inspectable source copies.

## Open the deck

Install Node.js 22 or newer. No package install is needed to view the deck.

- **Windows:** double-click `start-deck.cmd`, then open the address printed in the terminal. The launcher keeps the server running; it does not open a browser automatically.
- **Any platform:** open a terminal in this folder and run `node serve.mjs`. Open `http://127.0.0.1:4180/` and keep the terminal open.
- **Another port:** run `node serve.mjs 4181` and open the printed address.

Stop the server with Ctrl+C. Opening `index.html` directly from the filesystem displays these server instructions: the interactive player, video byte-range requests, and audience synchronization need an HTTP origin.

The player, slideshow, GSAP, lab, video, poster, captions, and source copies are bundled locally. They work without an internet connection once this folder and Node are available. The final slide's **live game** link requires the separate game server at `http://127.0.0.1:4176/empires.html`; the research article is an external website.

## Present and explore

Use the official slideshow's Previous/Next controls or arrow keys. Click **Present**, or press **P**, to open the audience window. Speaker notes remain in the presenter view and edits are saved in that browser. The official controls also provide fullscreen and mute. If a browser blocks the audience window, allow the presentation's popup and retry Present.

On slide 2, press the video's native Play control. Pause and seek with those same controls. The 36.133-second, silent recording has a Scene guide caption track. It never advances the slide automatically; use Next when ready. Video playback stops when leaving the slide.

On slide 4, change one actual policy weight, compare action probabilities at a frozen checkpoint, and open the arithmetic trace. Slide 5 carries the edit into two reproducible simulation copies. Choose a horizon, run the comparison, and inspect both society and ecology outcomes. Both slides also have an **Open full-page lab** link to the same experiments in a scrollable page. That page opens a fresh experiment and keeps your edits when switching between its Weight lab and Paired futures tabs. The labs operate on bundled, detached state; they do not change the separately running game or its saves. The audience receives experiment inputs and recomputes results locally, so longer experiments may finish a little later there.

Text fields, sliders, and native media controls keep their own keyboard interactions. Click-to-play is disabled on the outer player so operating a lab does not start the composition timeline. The local composition iframe permits user-triggered JSON downloads and links that open a separate tab.

## Contents and provenance

| Location | Purpose |
| --- | --- |
| `index.html` | Portable wrapper using the official HyperFrames player and slideshow |
| `composition/` | Eight timed scenes, presentation styling, and speaker-note manifest |
| `runtime/` | Paused timeline, scene visibility, lab synchronization, preparation scripts, and server checks |
| `lab/` | Self-contained experiments built from the game's actual policy and simulation code |
| `assets/recording.json` | Recording intervals, playback speed, file hash, and footage provenance |
| `assets/recording-transcript.txt` | Accessible description of the silent footage |
| `sources/` | Portable source snapshots, hashes, implementation guide, and research attribution |
| `vendor/THIRD-PARTY-NOTICES.md` | Bundled framework versions and licenses |

The presentation distinguishes manual auxiliary-policy edits from the research's broader specification-to-weight synthesis. The optional fixed Laya selector belongs to the live application; these embedded experiments do not call Laya. Read `sources/READING.md` for the attribution and scope, and `VALIDATION.md` when included for the delivery's verified checks and remaining limits.

## Maintain or validate the presentation

The portable entry page is already a slideshow wrapper. The official CLI expects a raw composition at the project root, so do **not** pass this root folder directly to `hyperframes present`. The preparation script creates `.hyperframes-project/`, copies the composition and local dependencies, adjusts relative asset paths, and synchronizes the wrapper's slide manifest.

```sh
npm run sync:manifest
npm run prepare:hyperframes
npm run present
```

`present` runs the pinned official HyperFrames CLI **0.8.181** against that staged folder. Its first use may download the CLI through npm. Re-run preparation after edits; the staged folder is generated and should not be edited directly. The portable `node serve.mjs` launcher remains the viewing path that requires no dependency installation.

Run the local server integration checks and the official composition check:

```sh
npm test
npm run check:hyperframes
```

The server checks cover HTML/module/source MIME types, HEAD and OPTIONS, exact video byte ranges, invalid ranges, and invalid paths. The official check includes lint, browser runtime, layout inspection at all eight scene positions, and contrast inspection with selected snapshots; it requires the CLI's browser dependencies and local process-launch permission. A browser startup failure is an incomplete official check, not a passing visual audit. Set `HYPERFRAMES_HISTORY_ROOT` to a writable directory if the CLI's default history cache is unavailable.

The two HTML files contain the same static slide manifest. After editing scene timing or notes, run `npm run sync:manifest` before distributing the folder. The source project also contains `tools/package-deck.mjs`, which verifies local assets, scene continuity, source hashes, and the recording hash before copying the portable files into the game server's public deck directory. Packaging is separate from the local viewer and the official CLI stage.
