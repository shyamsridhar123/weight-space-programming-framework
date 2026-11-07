# First Conquest: evidence for the 90-second film

The [film](../renders/first-conquest-90s.mp4) records a real weight edit, a controlled comparison, and a manually commanded mission through victory. Automated export checks passed, and browser playback plus all ten chapter controls have been observed. Audio has been measured; no complete audiovisual watch or perceptual listening review is claimed.

## What the recording demonstrates

The fresh match began paused at **00:00**, with Blue under **I command**. An actual coefficient in the authored, untrained auxiliary game policy changed: **`W2[19]`, `army_train_bias`, 0.8 → 24**, advancing policy revision 0 → 1.

1. **Edit → preview.** The preferred action changed from Train Villager (score **7.86**) to Train Spearman (score **25.98**). Spearman's own score rose **2.78 → 25.98**. These are preference scores, not probabilities; preview alone issued no order.
2. **Compare 60 seconds.** Two simulations started from the same captured world, using the baseline or edited Blue policy and the same Red policy. The live match stayed at 00:00. Their first different issued decision occurred at **+1.6 seconds**; they issued **25 versus 26 orders**.
3. **Execute → paid queue → completed unit.** Run one decision issued Train Spearman. Food fell **500 → 450**, wood **300 → 275**, and the barracks showed one already-paid recruit with **15 seconds remaining**. By **00:48**, that recruit had completed and the army numbered three.

| Outcome after 60 simulated seconds | Baseline | Edited |
| --- | ---: | ---: |
| Food brought home | 220 | 160 |
| Wood brought home | 120 | 98 |
| Soldiers completed | 2 | 3 |
| Army alive, including two starting guards | 4 | 5 |
| Villagers alive | 10 | 9 |
| Blue / Red Town Center health | 1800 / 1800 | 1800 / 1800 |

This comparison measures a recruitment/economy trade-off; it does not establish a superior strategy.

Manual gathering, house construction, and seven further recruitment orders produced an army of **ten: two starting guards, one policy recruit, and seven manual recruits**. A manual attack at **03:30** won at **04:34**, with **8 soldiers trained, 3 enemies defeated, and 6 Blue survivors**. The weight edit did not autonomously win this battle. Selected segment 38 images begin after the first of its two manual recruitment purchases; segment 41 confirms the completed ten-soldier army.

At **03:30 / tick 2110**, native **Laya** (`laya-game`, CPU, checkpoint **`a28adbc55d13`**) proposed changing attack bias **`W2[30]`, −5 → −4**. That bounded proposal was **not applied**: the policy stayed at revision 1. Laya's checkpoint remained fixed; the demonstration did not train or rewrite it, and its RTS strategy is unvalidated.

## Capture and export

The film uses only the fresh run's segments **25–43**: **181.6 seconds of selected capture**, represented by **546 distinct JPEG source files**. These are direct screenshots of the controlled game tab, captured at roughly **3 fps**. The **30 fps** export repeats the last observed screenshot according to recorded timestamps; it does not contain 30 unique captured images per second or invented intermediate motion. Resumed gameplay used UI **2×** speed, with additional editorial acceleration separately labeled. Earlier wrong-tab captures 01–23 are quarantined and excluded.

[Export verification](verification.json) reports every automated check true, with no warnings:

| Property | Final file |
| --- | --- |
| Duration | 90.000 seconds, container / video / audio |
| Picture | 1280×880: complete 1280×720 gameplay plus a 160-pixel caption band |
| Video | H.264, `yuv420p`, 30 fps, 2,700 decoded frames |
| Audio | AAC, stereo, 48,000 Hz |
| Size | 5,670,591 bytes |
| Validation | Full decode passed; source images and music unchanged |

SHA-256:

```text
29ee64ca17aaada840508f42527739400c5b868cd3cf10a2b5a20b5d3a880991
```

## Review scope and music

[Encoded-frame review](encoded-frame-review.md) covers 16 samples; the production reviewer separately inspected 7, 21, 28, and 88 seconds, for **20 sampled frames total**. No material cropping, caption clipping, or readability issue was observed at native resolution. This was not a complete watch. One minor timing issue remains: at **84 seconds**, a destruction caption appears while the enemy town still has **104 health**; victory follows at approximately **85.825 seconds** and is visible in the reviewed 88-second frame.

[Browser playback and chapter review](browser-review.json): **mouse playback, pause, all ten chapters, ending, replay, and the live-game destination observed**. On the final viewer, clicking Play film started playback at **0.017499 seconds**; it progressed to **12.159805 seconds**, and clicking Pause film stopped it at **12.320002 seconds**, without a media error. All ten chapter buttons were retested by mouse and sought to exactly **0, 4, 11, 17, 25, 33, 40, 59, 70, and 86 seconds**, each with the matching explanatory heading, paused state, and no error.

Clicking **Play film** at **86 seconds** advanced to **90.000 seconds**, with **`ended=true`**, **`paused=true`**, and no error; the button changed to **Replay film**. Clicking Replay film restarted at **0 seconds**, with **`paused=false`**. Pause film and the baseline chapter then left the player ready at **0 seconds**, paused. The live-game link destination opened in a fresh tab and displayed the original playable mission entry. These checks do not constitute a complete uninterrupted watch or listening review.

The instrumental soundtrack is **“Horizons” by Scott Buckley**, under **CC BY 4.0**. Its prepared 90-second bed retains level adjustment, volume shaping, and fades; gameplay acceleration does not speed up the music. Carry the [music credit and modification notice](../MUSIC-CREDITS.md) with shared copies. There is no spoken narration or captured game audio.

[Final encoded-audio measurements](audio/RESULT.md) report **−24.02 LUFS integrated**, **−7.68 dBTP true peak**, and **11.20 LU loudness range**, with signal in both channels, a quiet opening, and a measured fade-out. These are technical measurements, **not a listening review**; musical transitions and subjective playback quality remain unreviewed.

The detailed raw audit, historical corrections, failed first build, and earlier test evidence remain in the workspace at `artifacts/empires/recording-readiness/README.md`. That path is a workspace reference, not a file included in this portable evidence bundle.
