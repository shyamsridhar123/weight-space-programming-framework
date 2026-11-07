# Interactive learning labs

These two presentation panels run the application's real policy and world simulation. The bundled `lab.js` contains its dependencies and needs no network service, Laya process, model download, or browser storage. It mounts into `#policy-lab` and `#future-lab` when the document is ready. `lab.css` supplies the scoped presentation layout.

## Focused application

Open `lab/index.html` from the presentation's server for a full-page learning application. **Weight lab** and **Paired futures** use the same mounted `lab.js` instance, so switching views preserves the edited policy, running comparison, and results. The **Back to presentation** link returns to the parent presentation. This entrypoint also works when the presentation folder is served as a portable static site.

The layout fits a 1280 × 720 window and scrolls when content needs more room. Supplemental styles in `standalone.css` are scoped to this page. `standalone.js` only controls view visibility and navigation; it contains no model or simulation logic. Tab through controls, use Left/Right or Home/End while a lab tab is focused, and use Enter/Space to activate buttons. Direct links ending in `#weights` or `#futures` open the corresponding view. A page reload starts a fresh checkpoint; changes are retained while switching labs within the page.

## Teaching path

1. Inspect Iron Ridge at the start of year 10 in seed 42. The actual model, world settings, and random state are preserved.
2. Predict how increasing conservation will affect its probability, then choose **Try conservation = 4**. All eight probabilities update against the same frozen observations.
3. Open **Inspect the network** to follow the selected input terms, ReLU activation, output connection, score, and softmax probability. Its action and neuron selectors are independent, so an inactive action gate can be inspected too.
4. Choose **Compare futures** for 1, 3, 5, or 10 years. Two detached world copies start from the same checkpoint. Only Iron Ridge's candidate policy differs.
5. Compare population, food, forest, and knowledge; change the trajectory metric; inspect scheduled focus counts; download the full reproducible experiment.

Weight and temperature edits immediately clear stale outcomes. Changing duration also clears results. Cancellation presents no partial result. Reset restores the checkpoint's original weights and temperature, the conservation view, and the three-year horizon.

## Model and interpretation

- The authored, untrained scorer has 32 inputs, 38 ReLU hidden units, and one score per candidate action: 1,293 network parameters. Temperature is separate sampling configuration.
- Inputs are 16 local state features, eight action indicators, and eight reserved zeros.
- These panels edit the 38 W2 output connections and temperature. They do not train a model or call native Laya.
- For an output edit at a frozen state, the selected connection contributes `hidden activation × weight change`. The total score difference sums all edited connections.
- Scheduled focus choices can fail because of resource or world constraints. Their counts are not counts of successful actions.
- Population and food include traveling groups. Ecological metrics follow the original occupied regions; lost observations become unavailable.
- Both branches have the same initial random state. Different decisions can alter subsequent random draws. A single pair is a reproducible scenario, not an average causal effect.
- The chart axis fits its observed range and uses quarterly samples. Forest and fertility deltas are percentage points in the UI and fractions in the JSON.
- Simulation retention caps are 2,048 regions, 32 civilizations, 128 settlements, and 128 mobile groups. The method panel reports observed limit warnings.

## Integration API

`window.LivingWorldLab` exposes:

```ts
{
  version: 1,
  ready: Promise<void>,
  snapshot(): PortableLabState | null,
  restore(state: PortableLabState): Promise<void>,
  run(): Promise<void>,
  reset(): Promise<void>,
  result(): LearningExperimentResult | null
}
```

Snapshots include checkpoint identity, all W2 values, temperature, selected action and neuron, duration, trajectory metric, open panel, and whether a comparison is complete. `restore` validates the checkpoint and numeric ranges. Completed remote states are replayed locally instead of accepting transferred metrics. Results returned through the API are detached copies.

Local changes dispatch `living-world:lab-state` on `document`, with `{ state: snapshot() }` in the event detail. Remote restores suppress echo. Lab command handlers are attached to the lab mounts and dialog container, so upstream document-level slideshow click handlers cannot consume them first.

## Source and verification

- `model.ts` adapts `game/src/world/policy.ts`, `simulation.ts`, and `learning-experiment.ts` without replacing their arithmetic or dynamics.
- `lab.ts` owns controls, rendering, async progress, invalidation, export, and state synchronization.
- `lab.test.ts` independently calculates dense multiplication, ReLU, and softmax, then checks real experiment replay, all supported horizons, preserved checkpoints, export, and restore.
- `inspect-demo.ts` prints exact policy and three-year demonstration values.

From the repository root:

```powershell
.\node_modules\.bin\tsc.cmd --noEmit -p presentations/living-world/lab/tsconfig.json
.\node_modules\.bin\tsx.cmd --test --test-reporter=spec presentations/living-world/lab/lab.test.ts
node presentations/living-world/lab/build.mjs
```

The build produces the portable IIFE `lab.js` and `bundle-meta.json`. Re-run the presentation's staging command after a lab rebuild when testing the official CLI, because `.hyperframes-project` contains copied assets.

The verified demonstration changes W2[2] from −0.5 to 4, with activation 1 and score contribution +4.5. Conservation probability changes from 0.059019727625896606 to 0.637442943401737. Over three years its scheduled selections change from 4 to 22; the forest difference is 0.009814642091766634, or about +0.98 percentage points. Other outcomes need not improve: population changes by −0.13827262516454653 and wealth by −29.97121311673007 in this pair.
