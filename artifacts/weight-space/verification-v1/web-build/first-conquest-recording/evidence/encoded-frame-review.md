# Encoded-frame visual review

Result: the 16 sampled encoded frames retain the complete captured game viewport and readable editorial captions. Visible numerical claims agree with the interface, with one minor caption-timing issue noted below. This is a static frame review, not a complete watch or playtest.

## Scope

Reviewed these original-resolution, 1280 × 880 PNGs from this directory: `frame-002.0.png`, `frame-010.0.png`, `frame-014.0.png`, `frame-031.5.png`, `frame-036.0.png`, `frame-039.0.png`, `frame-043.0.png`, `frame-048.0.png`, `frame-055.0.png`, `frame-058.8.png`, `frame-061.0.png`, `frame-064.5.png`, `frame-068.0.png`, `frame-073.0.png`, `frame-079.0.png`, and `frame-084.0.png`.

The parent reviewer separately inspected 007.0, 021.0, 028.0, and 088.0 seconds. Those four samples are not included in this review's count or independent findings.

Parent-supplied final MP4 SHA256: `29ee64ca17aaada840508f42527739400c5b868cd3cf10a2b5a20b5d3a880991`. This reviewer did not recompute that checksum or independently extract these frames.

## Findings

| Video time | Visible evidence and caption agreement |
| --- | --- |
| 00:02.0 | Opening is paused at game time 00:00 with 500 food, 300 wood, and population 8/15. Weight programming is open and Capture baseline is highlighted. Caption identifies a small authored neural policy. |
| 00:10.0 | Weight editor shows the army-training bias, `W2[19] = 24.000`, and policy revision 1. This agrees with the edit's destination value. The original 0.8 value is not independently visible in this sample. |
| 00:14.0 | Preview shows Train Spearman, score 25.98, and Train Villager → Train Spearman. The executed-order panel explicitly says no order has been issued. Caption correctly distinguishes preview from execution. |
| 00:31.5 | Barracks shows 15 seconds remaining and one already-paid queued recruit. Resources are 450 food and 275 wood; army is 2 +1. Caption correctly describes remaining training time rather than a completed soldier. |
| 00:36.0, 00:39.0 | Army is 3 with no queued recruit. Food and wood rise between samples while villagers gather. Caption correctly says the policy recruit finishes and labels gathering as manual orders. |
| 00:43.0 | The new house is visible; population capacity is 20, increased from 15. Caption explicitly attributes the house to manual gameplay. |
| 00:48.0 | Barracks shows five already-paid queued recruits, three living soldiers, and seven seconds remaining on the active recruit. Caption says these recruitment orders are manual. |
| 00:55.0, 00:58.8 | Army changes from eight living soldiers plus two queued to ten living soldiers and an empty queue. Resource stocks continue increasing. This agrees with the recruitment/gathering chapter. |
| 01:01.0, 01:04.5 | Paused Laya request is followed by a visible proposed `W2[30]` attack-bias edit, −5.00 → −4.00. Policy header remains revision 1; caption labels the proposal suggested and not applied. The model-checkpoint claim cannot be independently established from still images. |
| 01:08.0 | Ten spearmen are selected. Caption's total agrees with the UI. Its breakdown into two starting guards, one policy recruit, and seven manual recruits relies on the preceding recording sequence, not this still alone. |
| 01:13.0 | Ten soldiers are engaging the red Town Center, whose health is 1742/1800. Caption labels the attack as manually ordered. |
| 01:19.0 | Army is eight; enemy Town Center health is 718/1800. Caption about combat losses and falling building health agrees with the earlier battle sample. |
| 01:24.0 | Army is six; enemy Town Center health is 104/1800 at game time 04:30. This supports a near-complete manual assault, but destruction has not yet occurred in this sample. |

## Issue and limitations

- Minor caption-timing issue at 01:24.0: “The manual assault destroys the enemy Town Center” appears before its visible health reaches zero. The parent reviewer reports inspecting the later 01:28.0 victory frame. A stricter pre-victory caption would be “Finish the manual assault” or “The enemy Town Center is close to destruction.”
- No accidental export cropping, missing caption text, or material legibility problem was observed at native resolution. Resources, action controls, minimap, and the separate caption band remain inside the encoded frame. Readability at reduced player sizes was not evaluated here.
- The sampled sequence distinguishes previews, paid queues, completed soldiers, manual gameplay, and an unapplied Laya suggestion. It does not attribute the manual victory to the paired weight experiment.
- This review does not independently verify implementation details, neural-model training history, Laya checkpoint immutability, or causality from still images. The 60-second comparison and final victory statistics occur in samples handled by the parent reviewer.
- No complete real-time watch, audio listening, browser actions, independent checksum verification, or frame-extraction validation was performed for this subtask.
