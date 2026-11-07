import type { PolicyModel, PolicyTrace, RankedCandidate } from './policy';
import type { WeightEdit, WeightSnapshot } from './weight-runtime';
import type { WeightExperimentConfig, WeightExperimentProgress, WeightExperimentResult } from './weight-experiments';
import './weight-lab.css';

export interface WeightLabPreset {
  id: string;
  name: string;
  description: string;
  mappings?: ReadonlyArray<{ hidden: string; value: number }>;
}

export interface WeightLabSearchState {
  state: 'idle' | 'running' | 'complete' | 'cancelled' | 'error';
  config: WeightExperimentConfig;
  progress?: WeightExperimentProgress;
  result?: WeightExperimentResult;
  error?: string;
  autoPromote: boolean;
  /** The session owner must check validation and the source revision before enabling promotion. */
  canPromote: boolean;
}

export interface WeightLabSnapshot {
  active: WeightSnapshot;
  baseline: WeightSnapshot;
  history: readonly WeightSnapshot[];
  /** Staged values are editable, but rankings always describe the active model. */
  draft?: PolicyModel;
  liveApply: boolean;
  stagedEdits: number;
  rankings: { active: readonly RankedCandidate[]; baseline: readonly RankedCandidate[] };
  inspection?: { candidateId: string; active: PolicyTrace; baseline?: PolicyTrace };
  presets: readonly WeightLabPreset[];
  canUndo: boolean;
  search: WeightLabSearchState;
  coach?: { available: boolean; busy: boolean; message?: string; suggestion?: string };
  status?: string;
  error?: string;
  currentTick?: number;
  context?: { tick: number; seconds: number; pilot: string; paused: boolean; matchId: string };
}

type Operation = unknown | Promise<unknown>;
export interface WeightLabCallbacks {
  setLiveApply(enabled: boolean): Operation;
  editTensor(edit: WeightEdit): Operation;
  applyStaged(): Operation;
  discardStaged(): Operation;
  applyPreset(id: string): Operation;
  pinBaseline(): Operation;
  undo(): Operation;
  rollback(id: string): Operation;
  importCheckpoint(json: string): Operation;
  /** Return validated checkpoint JSON. The lab displays it and offers a browser download. */
  exportCheckpoint(): string | Promise<string>;
  selectCandidate(id: string): Operation;
  startSearch(config: WeightExperimentConfig, autoPromote: boolean): Operation;
  stopSearch(): Operation;
  promoteSearchCandidate(id: string): Operation;
  requestLayaCoach?(): Operation;
  onOpenChange?(open: boolean): void;
}

type Tab = 'live' | 'tensors' | 'experiments' | 'history';
type Tensor = WeightEdit['tensor'];
const TABS: Tab[] = ['live', 'tensors', 'experiments', 'history'];
const TENSORS: Tensor[] = ['w1', 'b1', 'w2', 'b2', 'temperature'];
const ESSENTIALS = new Set([0, 1, 6, 9, 16, 25, 27, 31]);
const pretty = (name: string) => name.replaceAll('_', ' ').replace(/^./, char => char.toUpperCase());
const number = (value: number, digits = 3) => Number.isFinite(value) ? String(Number(value.toFixed(digits))) : '—';
const signed = (value: number, digits = 3) => `${value > 0 ? '+' : ''}${number(value, digits)}`;
const percent = (value: number) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const setText = (node: HTMLElement, value: string) => { if (node.textContent !== value) node.textContent = value; };
const inputValue = (node: HTMLInputElement | HTMLSelectElement, value: string) => {
  if (document.activeElement !== node && node.value !== value) node.value = value;
};

interface NeuronRow {
  root: HTMLElement;
  range: HTMLInputElement;
  input: HTMLInputElement;
  contribution: HTMLElement;
}
interface RankRow {
  root: HTMLButtonElement;
  index: HTMLElement;
  label: HTMLElement;
  activeScore: HTMLElement;
  activeProbability: HTMLElement;
  baselineScore: HTMLElement;
  baselineProbability: HTMLElement;
  delta: HTMLElement;
  bar: HTMLElement;
}
interface TraceRow { root: HTMLElement; label: HTMLElement; value: HTMLElement; bar: HTMLElement }

/** A persistent DOM workbench. Periodic snapshots update values, never replace focused editors. */
export class WeightLab {
  private readonly panel: HTMLElement;
  private readonly abort = new AbortController();
  private readonly refs = new Map<string, HTMLElement>();
  private snapshot?: WeightLabSnapshot;
  private tab: Tab = 'live';
  private tensor: Tensor = 'w1';
  private row = 0;
  private column = 0;
  private shapeSignature = '';
  private valuesSignature = '';
  private presetSignature = '';
  private historySignature = '';
  private renderedResult?: WeightExperimentResult;
  private rankRows = new Map<string, RankRow>();
  private neurons: NeuronRow[] = [];
  private featureRows: TraceRow[] = [];
  private contributionRows: TraceRow[] = [];
  private vectorButtons: HTMLButtonElement[] = [];
  private configInitialized = false;
  private configDirty = false;
  private lastSearchState: WeightLabSearchState['state'] = 'idle';
  private localError = '';
  private localStatus = '';
  private allRanks = false;
  private selectedCandidate = '';

  constructor(host: HTMLElement, private readonly callbacks: WeightLabCallbacks) {
    this.panel = document.createElement('aside');
    this.panel.className = 'weight-lab';
    this.panel.id = 'weight-lab';
    this.panel.hidden = true;
    this.panel.setAttribute('aria-label', 'Live Weight Lab');
    this.panel.innerHTML = `
      <header class="wl-header">
        <div><div class="wl-eyebrow"><span class="wl-beacon"></span> BLUE STAG · WEIGHT LAB</div><h2>Write the next move.</h2></div>
        <button type="button" class="wl-icon-button" data-ref="close" data-action="close" aria-label="Close Weight Lab" title="Close · Escape">×</button>
      </header>
      <div class="wl-identity">
        <div><span class="wl-caption">ACTIVE REVISION</span><strong data-ref="revision">—</strong><code data-ref="fingerprint" title="Fingerprint of the numerical model">—</code></div>
        <div><span class="wl-caption">ACTUAL PARAMETERS</span><strong data-ref="parameter-count">—</strong><span data-ref="architecture" class="wl-micro">—</span></div>
        <div class="wl-provenance"><span class="wl-caption">PROVENANCE</span><strong data-ref="source">—</strong><span data-ref="model-version" class="wl-micro">—</span></div>
      </div>
      <div class="wl-context" data-ref="context" hidden><span data-ref="context-state"></span><span data-ref="context-clock"></span><span data-ref="context-pilot"></span></div>
      <div class="wl-apply-bar">
        <label class="wl-switch"><input type="checkbox" data-ref="live-apply" checked><span class="wl-switch-track" aria-hidden="true"></span><span>Apply edits live</span></label>
        <span class="wl-edit-state" data-ref="edit-state">Next decision</span>
        <button type="button" data-ref="apply" data-action="apply" class="wl-button wl-button-gold" disabled>Apply <span data-ref="staged-count"></span></button>
        <button type="button" data-ref="discard" data-action="discard" class="wl-button wl-button-quiet" disabled>Discard</button>
      </div>
      <nav class="wl-tabs" role="tablist" aria-label="Weight Lab views">
        ${TABS.map((tab, index) => `<button type="button" id="wl-tab-${tab}" role="tab" data-tab="${tab}" aria-controls="wl-pane-${tab}" aria-selected="${index === 0}" tabindex="${index === 0 ? 0 : -1}">${pretty(tab)}</button>`).join('')}
      </nav>
      <div class="wl-scroll">
        <section id="wl-pane-live" class="wl-pane" data-pane="live" role="tabpanel" aria-labelledby="wl-tab-live">
          <div class="wl-scope-note">Edits program the <b>auxiliary neural commander</b> for your blue kingdom. Laya uses its shortlist; these controls do not edit Laya’s encoder.</div>
          <div class="wl-section-heading"><div><span class="wl-kicker">COUNTERFACTUAL</span><h3>Same world. Different weights.</h3></div><button type="button" class="wl-button" data-action="pin">Pin baseline</button></div>
          <div class="wl-comparison-labels"><span><i class="wl-dot wl-dot-gold"></i> Active <b data-ref="active-tag">—</b></span><span><i class="wl-dot wl-dot-blue"></i> Frozen baseline <b data-ref="baseline-tag">—</b></span><span class="wl-micro" data-ref="tick"></span></div>
          <div class="wl-rank-heading" aria-hidden="true"><span>LEGAL CANDIDATE</span><span>ACTIVE<br><small>score · probability</small></span><span>BASELINE<br><small>score · probability</small></span><span>Δ SCORE</span></div>
          <div class="wl-ranks" data-ref="ranks" aria-label="Active and frozen baseline action rankings"></div>
          <p class="wl-empty" data-ref="rank-empty">Start a kingdom to inspect legal candidates.</p>
          <button type="button" class="wl-disclosure" data-action="more-ranks" data-ref="more-ranks" hidden>Show all candidates</button>
          <p class="wl-footnote">Probabilities are a softmax over legal candidates, not win probabilities. A pinned baseline is separate from the opposing commander.</p>
          <details class="wl-details" open>
            <summary>Inside the selected decision <span data-ref="inspection-label">—</span></summary>
            <div class="wl-details-body">
              <div class="wl-equation"><code>score = b₂ + Σ W₂[h] · ReLU(W₁[h] · x + b₁[h])</code><strong data-ref="trace-score">—</strong></div>
              <p class="wl-footnote" data-ref="trace-detail">Select a candidate above to inspect its real forward pass.</p>
              <details class="wl-nested"><summary>Input features <span data-ref="feature-count">32</span></summary><div class="wl-trace-list" data-ref="features"></div></details>
              <details class="wl-nested" open><summary>Neuron contributions <span data-ref="hidden-count">38</span></summary><div class="wl-trace-legend"><span>ACTIVE CONTRIBUTION</span><span>BASELINE</span></div><div class="wl-trace-list" data-ref="contributions"></div></details>
            </div>
          </details>
          <div class="wl-section-heading"><div><span class="wl-kicker">SEMANTIC CONTROLS</span><h3>Tune the commander.</h3></div><select data-ref="neuron-filter" aria-label="Filter semantic output weights"><option value="essentials">8 essentials</option><option value="all">All neurons</option><option value="economy">Economy</option><option value="military">Military</option><option value="progression">Progression</option><option value="general">General</option></select></div>
          <p class="wl-footnote">Each control writes the labeled neuron’s actual W₂ coefficient. Positive values reward its activation; negative values penalize it.</p>
          <div class="wl-neurons" data-ref="neurons"></div>
          <div class="wl-preset-box"><label class="wl-field">Strategy preset<select data-ref="preset" aria-label="Strategy preset"></select></label><button type="button" class="wl-button" data-action="preset" data-ref="preset-apply">Use preset</button><p data-ref="preset-description"></p><div class="wl-preset-mappings" data-ref="preset-mappings"></div></div>
          <section class="wl-coach" aria-label="Optional Laya coach"><div class="wl-section-heading"><div><span class="wl-kicker">OPTIONAL · LOCAL LAYA</span><h3>Consult the war council.</h3></div><button type="button" class="wl-button" data-ref="coach" data-action="coach" disabled>Ask Laya</button></div><p data-ref="coach-status">Connect Laya to request a tactical suggestion.</p><pre data-ref="coach-result" hidden></pre></section>
        </section>
        <section id="wl-pane-tensors" class="wl-pane" data-pane="tensors" role="tabpanel" aria-labelledby="wl-tab-tensors" hidden>
          <div class="wl-section-heading"><div><span class="wl-kicker">LITERAL PARAMETER EDITOR</span><h3>Every number is addressable.</h3></div><span class="wl-badge" data-ref="tensor-shape">38 × 32</span></div>
          <div class="wl-tensor-tabs" role="group" aria-label="Choose tensor">${TENSORS.map(tensor => `<button type="button" data-tensor="${tensor}" aria-pressed="${tensor === 'w1'}">${tensor === 'temperature' ? 'Temperature' : tensor.toUpperCase()}</button>`).join('')}</div>
          <p class="wl-footnote" data-ref="tensor-description">W₁ maps 32 features to 38 hidden neurons. Select a cell or use the labeled selectors.</p>
          <div class="wl-matrix" data-ref="matrix">
            <div class="wl-matrix-scale"><span><i class="wl-swatch wl-swatch-negative"></i> Negative</span><span><i class="wl-swatch wl-swatch-zero"></i> Zero</span><span><i class="wl-swatch wl-swatch-positive"></i> Positive</span><span data-ref="matrix-range"></span></div>
            <div class="wl-matrix-viewport"><div class="wl-matrix-layout"><div class="wl-matrix-corner">NEURON ↓<br>FEATURE →</div><div class="wl-matrix-columns" data-ref="matrix-columns"></div><div class="wl-matrix-rows" data-ref="matrix-rows"></div><canvas data-ref="matrix-canvas" width="640" height="760" tabindex="0" role="img" aria-label="W1 weight matrix. Arrow keys select a cell; Enter focuses its value editor." aria-describedby="wl-cell-address"></canvas></div></div>
          </div>
          <div class="wl-vector" data-ref="vector" hidden aria-label="Tensor values"></div>
          <div class="wl-cell-editor">
            <div class="wl-section-heading"><code id="wl-cell-address" data-ref="cell-address">W1[0, 0]</code><span class="wl-micro" data-ref="cell-baseline">Baseline —</span></div>
            <label class="wl-field" data-ref="row-field">Hidden neuron<select data-ref="row-select" aria-label="Tensor hidden neuron"></select></label>
            <label class="wl-field" data-ref="column-field">Input feature<select data-ref="column-select" aria-label="W1 input feature"></select></label>
            <div class="wl-cell-value"><label class="wl-field">Parameter value<input data-ref="cell-value" type="number" min="-64" max="64" step="any" inputmode="decimal" aria-label="Selected tensor parameter value"></label><button type="button" class="wl-button wl-button-gold" data-action="write-cell">Write value</button><button type="button" class="wl-button" data-action="baseline-cell">Use baseline value</button></div>
            <p class="wl-footnote" data-ref="cell-note">Valid range: −64 to 64. Editing updates the selected parameter only.</p>
          </div>
          <div class="wl-network-note"><span>32 FEATURES</span><i>→</i><span>38 ReLU NEURONS</span><i>→</i><span>1 SCORE</span></div>
          <p class="wl-footnote">Temperature changes the softmax distribution. The shared output bias shifts every candidate score equally; neither changes the top-ranked action by itself.</p>
        </section>
        <section id="wl-pane-experiments" class="wl-pane" data-pane="experiments" role="tabpanel" aria-labelledby="wl-tab-experiments" hidden>
          <div class="wl-section-heading"><div><span class="wl-kicker">BACKGROUND SIMULATION</span><h3>Test before you promote.</h3></div><span class="wl-badge">W₂ SEARCH</span></div>
          <p class="wl-intro">Run seeded matches against a frozen copy of the starting commander while your kingdom continues. Search output-weight mutations on development seeds, then evaluate the finalist once on held-out seeds.</p>
          <form data-ref="search-form" class="wl-search-form">
            <fieldset data-ref="search-fields"><legend class="wl-sr-only">Mutation search configuration</legend>
              <div class="wl-form-grid">
                <label class="wl-field">Candidates<input type="number" data-config="candidates" min="1" max="16" step="1" required><small>At most 16 proposals</small></label>
                <label class="wl-field">Seconds per match<input type="number" data-config="maxSeconds" min="0.1" max="600" step="0.1" required><small>Simulation time, not wall time</small></label>
                <label class="wl-field">Mutation scale<input type="number" data-config="mutationScale" min="0.001" max="2" step="any" required><small>Seeded W₂ perturbation size</small></label>
                <label class="wl-field">Weights per candidate<input type="number" data-config="editsPerCandidate" min="1" max="38" step="1" required><small>Actual W₂ coordinates</small></label>
                <label class="wl-field wl-field-wide">Development seeds<input type="text" data-config="developmentSeeds" required spellcheck="false" placeholder="11, 47"><small>Comma-separated map seeds · up to 8</small></label>
                <label class="wl-field wl-field-wide">Held-out validation seeds<input type="text" data-config="validationSeeds" required spellcheck="false" placeholder="93, 137"><small>Keep separate from development seeds</small></label>
              </div>
              <details class="wl-nested wl-search-advanced"><summary>Evaluation controls</summary><div class="wl-form-grid"><label class="wl-field">Mutation random seed<input type="number" data-config="mutationSeed" min="0" max="4294967295" step="1" required></label><label class="wl-field">Decision interval · seconds<input type="number" data-config="decisionSeconds" min="0.1" max="10" step="0.1" required></label><label class="wl-field wl-field-wide">Minimum validation fitness gain<input type="number" data-config="minFitnessGain" min="0" max="1" step="any" required><small>Shaped fitness is reported separately from wins.</small></label></div></details>
              <label class="wl-auto-promote"><input type="checkbox" data-ref="auto-promote"><span>Auto-promote once if validation passes<small>Only if the source revision is still active. Other runs need a new choice.</small></span></label>
            </fieldset>
            <div class="wl-search-actions"><button type="submit" class="wl-button wl-button-gold" data-ref="search-start">Run mutation search <span aria-hidden="true">↗</span></button><button type="button" class="wl-button" data-ref="search-stop" data-action="stop-search" disabled>Stop</button></div>
          </form>
          <section class="wl-search-progress" aria-label="Experiment progress"><div class="wl-section-heading"><strong data-ref="search-state">Ready to experiment</strong><span data-ref="search-percent">—</span></div><progress data-ref="search-progress" max="100" value="0" aria-label="Experiment completion"></progress><p data-ref="search-detail">No simulations have run in this session.</p><span class="wl-micro" data-ref="search-best"></span></section>
          <section data-ref="search-results" class="wl-search-results" hidden><div class="wl-section-heading"><div><span class="wl-kicker">HELD-OUT RESULT</span><h3 data-ref="result-title">Validation complete.</h3></div><code data-ref="result-fingerprint"></code></div><div class="wl-result-metrics" data-ref="result-metrics"></div><p data-ref="promotion-reason" class="wl-promotion-reason"></p><button type="button" class="wl-button wl-button-gold" data-ref="promote" data-action="promote" disabled>Promote validated candidate</button><details class="wl-nested"><summary>Development evaluations</summary><div data-ref="development-results"></div></details></section>
          <p class="wl-footnote">These are independent simulations, not replays of this battle. A short test can end in a time-limit draw. No backpropagation or Laya encoder training runs here.</p>
        </section>
        <section id="wl-pane-history" class="wl-pane" data-pane="history" role="tabpanel" aria-labelledby="wl-tab-history" hidden>
          <div class="wl-section-heading"><div><span class="wl-kicker">CHECKPOINTS & PROVENANCE</span><h3>Keep the experiments reversible.</h3></div><button type="button" data-ref="undo" data-action="undo" class="wl-button">Undo</button></div>
          <p class="wl-intro">Export the exact model, import a validated checkpoint, or restore a recorded revision. Each applied change carries its own fingerprint and source.</p>
          <div class="wl-checkpoint-actions"><button type="button" data-action="export" class="wl-button wl-button-gold">Export checkpoint ↓</button><button type="button" data-action="choose-file" class="wl-button">Choose JSON file</button><input data-ref="checkpoint-file" type="file" accept=".json,application/json" hidden></div>
          <label class="wl-field wl-json-field">Checkpoint JSON<textarea data-ref="checkpoint" rows="8" spellcheck="false" placeholder="Paste an exported checkpoint here, or choose a JSON file." aria-label="Checkpoint JSON text"></textarea></label>
          <div class="wl-checkpoint-actions"><button type="button" data-action="import" class="wl-button">Validate & import</button><span class="wl-micro">Import is an explicit model change.</span></div>
          <div class="wl-section-heading wl-history-heading"><h3>Revision history</h3><span class="wl-micro" data-ref="history-count">—</span></div>
          <ol class="wl-history" data-ref="history"></ol>
        </section>
      </div>
      <footer class="wl-footer"><span data-ref="status" role="status" aria-live="polite">Live edits affect the next commander decision.</span><span class="wl-footer-key">ESC <i>close</i></span></footer>`;
    host.append(this.panel);
    for (const node of this.panel.querySelectorAll<HTMLElement>('[data-ref]')) this.refs.set(node.dataset.ref!, node);
    this.bindEvents();
  }

  get isOpen(): boolean { return !this.panel.hidden; }

  setOpen(open: boolean): void {
    if (open === this.isOpen) return;
    const returnFocus = this.panel.contains(document.activeElement);
    this.panel.hidden = !open;
    const toggle = document.getElementById('weight-lab-toggle');
    toggle?.setAttribute('aria-expanded', String(open));
    toggle?.setAttribute('aria-controls', this.panel.id);
    this.callbacks.onOpenChange?.(open);
    if (open) {
      this.ref<HTMLButtonElement>('close').focus({ preventScroll: true });
      this.drawMatrix();
    } else if (returnFocus) toggle?.focus({ preventScroll: true });
  }

  update(snapshot: WeightLabSnapshot): void {
    this.snapshot = snapshot;
    const model = snapshot.draft ?? snapshot.active.model;
    const shape = `${model.featureNames.join('|')}::${model.hiddenNames.join('|')}`;
    if (shape !== this.shapeSignature) { this.buildModelControls(model); this.shapeSignature = shape; }
    const values = JSON.stringify([model.w1, model.b1, model.w2, model.b2, model.temperature]);
    const modelChanged = values !== this.valuesSignature;
    this.valuesSignature = values;
    const parameters = model.w1.reduce((sum, row) => sum + row.length, 0) + model.b1.length + model.w2.length + 1;
    setText(this.ref('revision'), `r${snapshot.active.revision}`);
    setText(this.ref('fingerprint'), snapshot.active.fingerprint.slice(0, 12));
    this.ref('fingerprint').title = snapshot.active.fingerprint;
    setText(this.ref('parameter-count'), parameters.toLocaleString());
    setText(this.ref('architecture'), `${model.featureNames.length} → ${model.b1.length} → 1 · + temperature`);
    setText(this.ref('source'), pretty(snapshot.active.provenance.source));
    this.ref('source').title = snapshot.active.provenance.reason;
    setText(this.ref('model-version'), `${snapshot.active.model.version} · ${snapshot.active.model.provenance}`);
    const context = snapshot.context;
    this.ref('context').hidden = !context;
    if (context) {
      const seconds = Math.max(0, Math.floor(context.seconds));
      this.ref('context').title = `Match ${context.matchId}`;
      this.ref('context').classList.toggle('wl-context-paused', context.paused);
      setText(this.ref('context-state'), context.paused ? 'PAUSED' : 'RUNNING');
      setText(this.ref('context-clock'), `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')} · tick ${context.tick}`);
      setText(this.ref('context-pilot'), `${pretty(context.pilot)} pilot`);
    }
    const live = this.ref<HTMLInputElement>('live-apply');
    live.checked = snapshot.liveApply;
    setText(this.ref('edit-state'), snapshot.stagedEdits > 0 ? `${snapshot.stagedEdits} staged` : snapshot.liveApply ? 'Next decision' : 'Stage changes');
    this.ref('edit-state').classList.toggle('wl-has-edits', snapshot.stagedEdits > 0);
    setText(this.ref('staged-count'), snapshot.stagedEdits ? `(${snapshot.stagedEdits})` : '');
    this.ref<HTMLButtonElement>('apply').disabled = !snapshot.stagedEdits;
    this.ref<HTMLButtonElement>('discard').disabled = !snapshot.stagedEdits;
    this.ref<HTMLButtonElement>('undo').disabled = !snapshot.canUndo;
    setText(this.ref('active-tag'), `r${snapshot.active.revision}`);
    setText(this.ref('baseline-tag'), `r${snapshot.baseline.revision} · ${snapshot.baseline.fingerprint.slice(0, 6)}`);
    const tick = context?.tick ?? snapshot.currentTick;
    setText(this.ref('tick'), tick === undefined ? '' : `TICK ${tick}`);
    this.updateRankings();
    this.updateTrace();
    this.updateNeurons(model);
    this.updatePresets();
    this.updateTensorEditor(modelChanged);
    this.updateSearch();
    this.updateHistory();
    this.updateCoach();
    this.updateStatus();
  }

  dispose(): void { this.abort.abort(); this.panel.remove(); this.refs.clear(); }

  private ref<T extends HTMLElement = HTMLElement>(name: string): T { return this.refs.get(name)! as T; }
  private get model(): PolicyModel | undefined { return this.snapshot?.draft ?? this.snapshot?.active.model; }

  private bindEvents(): void {
    const signal = this.abort.signal;
    // UI pointer/keyboard activity must not become battlefield orders or camera movement.
    for (const type of ['pointerdown', 'pointerup', 'pointermove', 'mousedown', 'mouseup', 'touchstart', 'touchmove', 'wheel', 'contextmenu', 'keyup']) {
      this.panel.addEventListener(type, event => event.stopPropagation(), { signal, passive: type === 'wheel' || type.startsWith('touch') });
    }
    document.addEventListener('keydown', event => {
      if (event.key !== 'Escape' || !this.isOpen || document.querySelector('.modal:not([hidden])')) return;
      event.preventDefault(); event.stopPropagation(); this.setOpen(false);
    }, { signal, capture: true });
    this.panel.addEventListener('keydown', event => {
      event.stopPropagation();
      const target = event.target as HTMLElement;
      if (target.dataset.tab && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const index = TABS.indexOf(this.tab);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? TABS.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + TABS.length) % TABS.length;
        this.selectTab(TABS[next], true);
      }
      if (target === this.ref('matrix-canvas')) this.matrixKey(event);
      if (target === this.ref('cell-value') && event.key === 'Enter') { event.preventDefault(); this.writeCell(); }
    }, { signal });
    this.panel.addEventListener('click', event => {
      event.stopPropagation();
      const target = (event.target as HTMLElement).closest<HTMLElement>('button');
      if (!target || target.hasAttribute('disabled') || target.getAttribute('aria-disabled') === 'true') return;
      if (target.dataset.tab) { this.selectTab(target.dataset.tab as Tab); return; }
      if (target.dataset.tensor) { this.tensor = target.dataset.tensor as Tensor; this.updateTensorEditor(true); return; }
      if (target.dataset.candidate) {
        this.selectedCandidate = target.dataset.candidate;
        this.perform(() => this.callbacks.selectCandidate(this.selectedCandidate)); this.updateRankings(); return;
      }
      if (target.dataset.vector !== undefined) { this.row = Number(target.dataset.vector); this.updateTensorEditor(true); return; }
      if (target.dataset.restore) { this.perform(() => this.callbacks.rollback(target.dataset.restore!)); return; }
      switch (target.dataset.action) {
        case 'close': this.setOpen(false); break;
        case 'apply': this.perform(() => this.callbacks.applyStaged()); break;
        case 'discard': this.perform(() => this.callbacks.discardStaged()); break;
        case 'pin': this.perform(() => this.callbacks.pinBaseline(), 'Frozen comparison baseline updated.'); break;
        case 'undo': this.perform(() => this.callbacks.undo()); break;
        case 'more-ranks': this.allRanks = !this.allRanks; this.updateRankings(); break;
        case 'preset': this.perform(() => this.callbacks.applyPreset(this.ref<HTMLSelectElement>('preset').value)); break;
        case 'write-cell': this.writeCell(); break;
        case 'baseline-cell': this.restoreCell(); break;
        case 'stop-search': this.perform(() => this.callbacks.stopSearch()); break;
        case 'promote': if (this.snapshot?.search.canPromote && this.snapshot.search.result?.promotion.recommended) this.perform(() => this.callbacks.promoteSearchCandidate(this.snapshot!.search.result!.candidate.id)); break;
        case 'coach': if (this.callbacks.requestLayaCoach) this.perform(() => this.callbacks.requestLayaCoach!()); break;
        case 'export': void this.exportCheckpoint(); break;
        case 'choose-file': this.ref<HTMLInputElement>('checkpoint-file').click(); break;
        case 'import': this.importCheckpoint(); break;
      }
    }, { signal });
    const live = this.ref<HTMLInputElement>('live-apply');
    live.addEventListener('change', () => {
      const enabled = live.checked;
      this.perform(async () => {
        try {
          await this.callbacks.setLiveApply(enabled);
        } finally {
          if (this.snapshot) live.checked = this.snapshot.liveApply;
        }
      });
    }, { signal });
    this.ref('neuron-filter').addEventListener('change', () => this.filterNeurons(), { signal });
    this.ref('preset').addEventListener('change', () => this.updatePresetDescription(), { signal });
    this.ref('row-select').addEventListener('change', event => { this.row = Number((event.target as HTMLSelectElement).value); this.updateTensorEditor(true); }, { signal });
    this.ref('column-select').addEventListener('change', event => { this.column = Number((event.target as HTMLSelectElement).value); this.updateTensorEditor(true); }, { signal });
    this.ref<HTMLInputElement>('cell-value').addEventListener('input', () => {
      const field = this.ref<HTMLInputElement>('cell-value');
      if (field.value !== '' && field.validity.valid) this.writeCell();
    }, { signal });
    this.ref<HTMLInputElement>('cell-value').addEventListener('blur', () => this.updateTensorEditor(false), { signal });
    const matrix = this.ref<HTMLCanvasElement>('matrix-canvas');
    matrix.addEventListener('pointerdown', event => {
      if (!this.model) return;
      const rect = matrix.getBoundingClientRect();
      this.column = clamp(Math.floor((event.clientX - rect.left) / rect.width * this.model.featureNames.length), 0, this.model.featureNames.length - 1);
      this.row = clamp(Math.floor((event.clientY - rect.top) / rect.height * this.model.hiddenNames.length), 0, this.model.hiddenNames.length - 1);
      matrix.focus({ preventScroll: true }); this.updateTensorEditor(true);
    }, { signal });
    matrix.addEventListener('pointermove', event => {
      const model = this.model; if (!model) return;
      const rect = matrix.getBoundingClientRect();
      const column = clamp(Math.floor((event.clientX - rect.left) / rect.width * model.featureNames.length), 0, model.featureNames.length - 1);
      const row = clamp(Math.floor((event.clientY - rect.top) / rect.height * model.hiddenNames.length), 0, model.hiddenNames.length - 1);
      matrix.title = `W1[${row}, ${column}] = ${model.w1[row][column]} · ${model.hiddenNames[row]} ← ${model.featureNames[column]}`;
    }, { signal });
    this.ref<HTMLFormElement>('search-form').addEventListener('input', () => { this.configDirty = true; }, { signal });
    this.ref<HTMLFormElement>('search-form').addEventListener('submit', event => { event.preventDefault(); this.startSearch(); }, { signal });
    this.ref<HTMLInputElement>('checkpoint-file').addEventListener('change', () => { void this.readCheckpointFile(); }, { signal });
    this.panel.addEventListener('focusout', () => queueMicrotask(() => { if (this.snapshot) this.updateHistory(); }), { signal });
  }

  private perform(action: () => Operation, message = ''): void {
    this.localError = ''; this.localStatus = '';
    try {
      const result = action();
      void Promise.resolve(result).then(() => { if (message) this.localStatus = message; this.updateStatus(); }, error => this.showError(error));
    } catch (error) { this.showError(error); }
    this.updateStatus();
  }

  private showError(error: unknown): void {
    this.localError = error instanceof Error ? error.message : String(error);
    this.localStatus = ''; this.updateStatus();
  }

  private updateStatus(): void {
    const error = this.localError || this.snapshot?.error || '';
    const text = error || this.localStatus || this.snapshot?.status || (this.snapshot?.stagedEdits ? 'Rankings use active weights. Apply staged changes to update decisions.' : 'Live edits affect the next commander decision.');
    const status = this.ref('status'); setText(status, text); status.classList.toggle('wl-status-error', !!error);
  }

  private selectTab(tab: Tab, focus = false): void {
    this.tab = tab;
    for (const button of this.panel.querySelectorAll<HTMLButtonElement>('[data-tab]')) {
      const selected = button.dataset.tab === tab;
      button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
      if (selected && focus) button.focus();
    }
    for (const pane of this.panel.querySelectorAll<HTMLElement>('[data-pane]')) pane.hidden = pane.dataset.pane !== tab;
    if (tab === 'tensors') this.drawMatrix();
  }

  private buildModelControls(model: PolicyModel): void {
    const rowSelect = this.ref<HTMLSelectElement>('row-select');
    const columnSelect = this.ref<HTMLSelectElement>('column-select');
    rowSelect.replaceChildren(...model.hiddenNames.map((name, index) => new Option(`${index.toString().padStart(2, '0')} · ${pretty(name)}`, String(index))));
    columnSelect.replaceChildren(...model.featureNames.map((name, index) => new Option(`${index.toString().padStart(2, '0')} · ${pretty(name)}`, String(index))));
    const rows = this.ref('matrix-rows'), columns = this.ref('matrix-columns');
    rows.replaceChildren(); columns.replaceChildren();
    for (const [index, name] of model.hiddenNames.entries()) {
      const label = document.createElement('span'); label.textContent = `${index.toString().padStart(2, '0')} ${pretty(name)}`; label.title = name; rows.append(label);
    }
    for (const [index, name] of model.featureNames.entries()) {
      const label = document.createElement('span'); label.textContent = `${index.toString().padStart(2, '0')} ${pretty(name)}`; label.title = name; columns.append(label);
    }
    this.ref('neurons').replaceChildren(); this.neurons = [];
    for (const [index, name] of model.hiddenNames.entries()) {
      const root = document.createElement('div'); root.className = 'wl-neuron';
      const label = document.createElement('label'); label.htmlFor = `wl-w2-${index}`; label.textContent = pretty(name); label.title = `W2[${index}] · ${name}`;
      const address = document.createElement('code'); address.textContent = `W₂[${index}]`;
      const head = document.createElement('div'); head.className = 'wl-neuron-label'; head.append(label, address);
      const range = document.createElement('input'); range.type = 'range'; range.min = '-12'; range.max = '12'; range.step = '0.05'; range.setAttribute('aria-label', `${pretty(name)} output weight slider`);
      const input = document.createElement('input'); input.type = 'number'; input.id = `wl-w2-${index}`; input.min = '-64'; input.max = '64'; input.step = 'any'; input.inputMode = 'decimal';
      const controls = document.createElement('div'); controls.className = 'wl-neuron-controls'; controls.append(range, input);
      const contribution = document.createElement('span'); contribution.className = 'wl-neuron-contribution';
      root.append(head, controls, contribution); this.ref('neurons').append(root);
      const edit = (field: HTMLInputElement) => {
        if (field.value === '' || !field.validity.valid) return;
        const value = Number(field.value); if (!Number.isFinite(value) || value === this.model?.w2[index]) return;
        this.perform(() => this.callbacks.editTensor({ tensor: 'w2', row: index, value }));
      };
      range.addEventListener('input', () => edit(range), { signal: this.abort.signal });
      input.addEventListener('input', () => edit(input), { signal: this.abort.signal });
      input.addEventListener('blur', () => { if (this.model) this.updateNeurons(this.model); }, { signal: this.abort.signal });
      this.neurons.push({ root, range, input, contribution });
    }
    this.featureRows = this.buildTraceRows(this.ref('features'), model.featureNames);
    this.contributionRows = this.buildTraceRows(this.ref('contributions'), model.hiddenNames);
    this.ref('vector').replaceChildren(); this.vectorButtons = [];
    for (const [index, name] of model.hiddenNames.entries()) {
      const button = document.createElement('button'); button.type = 'button'; button.dataset.vector = String(index); button.className = 'wl-vector-cell'; button.title = name;
      const label = document.createElement('span'); label.textContent = `${index} · ${pretty(name)}`;
      const value = document.createElement('code'); button.append(label, value); this.ref('vector').append(button); this.vectorButtons.push(button);
    }
    setText(this.ref('feature-count'), String(model.featureNames.length)); setText(this.ref('hidden-count'), String(model.hiddenNames.length));
    this.filterNeurons();
  }

  private filterNeurons(): void {
    const filter = this.ref<HTMLSelectElement>('neuron-filter').value;
    this.neurons.forEach((item, index) => {
      const visible = filter === 'all' || filter === 'essentials' && ESSENTIALS.has(index) || filter === 'economy' && index < 16 || filter === 'military' && (index >= 16 && index < 20 || index >= 25 && index < 36) || filter === 'progression' && index >= 20 && index < 25 || filter === 'general' && index >= 36;
      item.root.hidden = !visible;
    });
  }

  private updateNeurons(model: PolicyModel): void {
    this.neurons.forEach((item, index) => {
      const value = model.w2[index];
      if (document.activeElement !== item.range) { item.range.min = String(Math.min(-12, Math.floor(value))); item.range.max = String(Math.max(12, Math.ceil(value))); }
      inputValue(item.input, String(value)); inputValue(item.range, String(value));
      const contribution = this.snapshot?.inspection?.active.contributions[index];
      setText(item.contribution, contribution === undefined ? 'Choose a candidate to see its contribution.' : `${signed(contribution)} to selected active score`);
      item.root.classList.toggle('wl-neuron-negative', value < 0);
    });
  }

  private makeRankRow(id: string): RankRow {
    const root = document.createElement('button'); root.type = 'button'; root.className = 'wl-rank'; root.dataset.candidate = id;
    const name = document.createElement('span'); name.className = 'wl-rank-name';
    const index = document.createElement('i'), label = document.createElement('span'); name.append(index, label);
    const active = document.createElement('span'); active.className = 'wl-rank-values wl-rank-active';
    const activeScore = document.createElement('b'), activeProbability = document.createElement('small'); active.append(activeScore, activeProbability);
    const baseline = document.createElement('span'); baseline.className = 'wl-rank-values wl-rank-baseline';
    const baselineScore = document.createElement('b'), baselineProbability = document.createElement('small'); baseline.append(baselineScore, baselineProbability);
    const delta = document.createElement('span'); delta.className = 'wl-rank-delta';
    const bar = document.createElement('span'); bar.className = 'wl-rank-bar'; bar.setAttribute('aria-hidden', 'true');
    root.append(name, active, baseline, delta, bar);
    this.ref('ranks').append(root);
    return { root, index, label, activeScore, activeProbability, baselineScore, baselineProbability, delta, bar };
  }

  private updateRankings(): void {
    if (!this.snapshot) return;
    const { active, baseline } = this.snapshot.rankings;
    const byId = new Map(baseline.map(candidate => [candidate.id, candidate]));
    const ids = new Set(active.map(candidate => candidate.id));
    if (this.snapshot.inspection) this.selectedCandidate = this.snapshot.inspection.candidateId;
    if (!ids.has(this.selectedCandidate)) this.selectedCandidate = active[0]?.id ?? '';
    for (const [id, row] of this.rankRows) {
      if (ids.has(id)) continue;
      // Preserve a keyboard user's focused control until focus has moved elsewhere.
      if (document.activeElement === row.root) { row.root.setAttribute('aria-disabled', 'true'); row.root.tabIndex = -1; row.root.title = 'This candidate is no longer legal.'; }
      else { row.root.remove(); this.rankRows.delete(id); }
    }
    active.forEach((candidate, index) => {
      let row = this.rankRows.get(candidate.id);
      if (!row) { row = this.makeRankRow(candidate.id); this.rankRows.set(candidate.id, row); }
      const comparison = byId.get(candidate.id);
      row.root.disabled = false;
      row.root.removeAttribute('aria-disabled'); row.root.tabIndex = 0;
      row.root.style.order = String(index);
      row.root.hidden = !this.allRanks && index >= 6 && candidate.id !== this.selectedCandidate && document.activeElement !== row.root;
      row.root.setAttribute('aria-pressed', String(candidate.id === this.selectedCandidate));
      row.root.title = candidate.description;
      setText(row.index, String(index + 1).padStart(2, '0')); setText(row.label, candidate.label);
      setText(row.activeScore, signed(candidate.score)); setText(row.activeProbability, percent(candidate.probability));
      setText(row.baselineScore, comparison ? signed(comparison.score) : '—'); setText(row.baselineProbability, comparison ? percent(comparison.probability) : '—');
      setText(row.delta, comparison ? signed(candidate.score - comparison.score, 2) : '—');
      row.delta.classList.toggle('wl-negative', !!comparison && candidate.score < comparison.score);
      row.bar.style.width = `${clamp(candidate.probability * 100, 0, 100)}%`;
      row.root.setAttribute('aria-label', `${candidate.label}. Active score ${number(candidate.score)}, probability ${percent(candidate.probability)}. Baseline ${comparison ? `score ${number(comparison.score)}, probability ${percent(comparison.probability)}` : 'unavailable'}. Inspect decision.`);
    });
    this.ref('rank-empty').hidden = active.length > 0;
    this.ref('more-ranks').hidden = active.length <= 6;
    setText(this.ref('more-ranks'), this.allRanks ? 'Show leading candidates' : `Show all ${active.length} candidates`);
  }

  private buildTraceRows(host: HTMLElement, labels: string[]): TraceRow[] {
    host.replaceChildren();
    return labels.map((name, index) => {
      const root = document.createElement('div'); root.className = 'wl-trace-row';
      const label = document.createElement('span'); label.className = 'wl-trace-name'; label.textContent = `${index.toString().padStart(2, '0')} ${pretty(name)}`; label.title = name;
      const value = document.createElement('code');
      const bar = document.createElement('i'); bar.className = 'wl-trace-bar'; bar.setAttribute('aria-hidden', 'true');
      root.append(label, value, bar); host.append(root); return { root, label, value, bar };
    });
  }

  private updateTrace(): void {
    const inspection = this.snapshot?.inspection;
    const candidate = this.snapshot?.rankings.active.find(item => item.id === inspection?.candidateId);
    setText(this.ref('inspection-label'), candidate?.label ?? 'Choose a candidate');
    setText(this.ref('trace-score'), inspection ? signed(inspection.active.score) : '—');
    setText(this.ref('trace-detail'), inspection ? `Output bias ${signed(inspection.active.outputBias)} · ${inspection.active.hiddenActivations.filter(value => value > 0).length} active neurons${inspection.baseline ? ` · baseline score ${signed(inspection.baseline.score)}` : ''}.` : 'Select a candidate above to inspect its real forward pass.');
    const maxContribution = Math.max(1, ...inspection?.active.contributions.map(Math.abs) ?? []);
    this.featureRows.forEach((item, index) => {
      const value = inspection?.active.input[index]; setText(item.value, value === undefined ? '—' : number(value, 4));
      item.bar.style.width = `${clamp(Math.abs(value ?? 0) * 100, 0, 100)}%`; item.root.classList.toggle('wl-trace-inactive', !value);
    });
    this.contributionRows.forEach((item, index) => {
      const value = inspection?.active.contributions[index], baseline = inspection?.baseline?.contributions[index];
      setText(item.value, value === undefined ? '—' : `${signed(value)}${baseline === undefined ? '' : `  /  ${signed(baseline)}`}`);
      item.root.title = inspection ? `Pre-activation ${number(inspection.active.preActivations[index], 4)} · ReLU activation ${number(inspection.active.hiddenActivations[index], 4)} · W2 ${number(this.snapshot!.active.model.w2[index], 4)}` : '';
      item.bar.style.width = `${Math.abs(value ?? 0) / maxContribution * 100}%`;
      item.root.classList.toggle('wl-trace-negative', (value ?? 0) < 0); item.root.classList.toggle('wl-trace-inactive', !value);
    });
  }

  private updatePresets(): void {
    const presets = this.snapshot?.presets ?? [];
    const signature = presets.map(preset => preset.id).join('|');
    if (signature !== this.presetSignature) {
      const select = this.ref<HTMLSelectElement>('preset'), selected = select.value;
      select.replaceChildren(...presets.map(preset => new Option(preset.name, preset.id)));
      if (presets.some(preset => preset.id === selected)) select.value = selected;
      this.presetSignature = signature;
    }
    this.ref<HTMLButtonElement>('preset-apply').disabled = presets.length === 0;
    this.updatePresetDescription();
  }

  private updatePresetDescription(): void {
    const preset = this.snapshot?.presets.find(item => item.id === this.ref<HTMLSelectElement>('preset').value);
    setText(this.ref('preset-description'), preset?.description ?? 'No strategy presets available.');
    const mappings = preset?.mappings?.map(mapping => {
      const row = this.model?.hiddenNames.indexOf(mapping.hidden) ?? -1;
      return row >= 0 ? `W₂[${row}] = ${signed(mapping.value)}` : '';
    }).filter(Boolean) ?? [];
    setText(this.ref('preset-mappings'), mappings.join(' · '));
  }

  private cellValue(model: PolicyModel): number {
    return this.tensor === 'w1' ? model.w1[this.row][this.column] : this.tensor === 'b1' ? model.b1[this.row] : this.tensor === 'w2' ? model.w2[this.row] : model[this.tensor];
  }

  private cellEdit(value: number): WeightEdit {
    return this.tensor === 'w1' ? { tensor: this.tensor, row: this.row, column: this.column, value }
      : this.tensor === 'w2' || this.tensor === 'b1' ? { tensor: this.tensor, row: this.row, value } : { tensor: this.tensor, value };
  }

  private writeCell(): void {
    const model = this.model, field = this.ref<HTMLInputElement>('cell-value');
    if (!model || !field.reportValidity() || field.value === '') return;
    const value = Number(field.value);
    if (!Number.isFinite(value) || value === this.cellValue(model)) return;
    this.perform(() => this.callbacks.editTensor(this.cellEdit(value)));
  }

  private restoreCell(): void {
    if (!this.snapshot) return;
    const value = this.cellValue(this.snapshot.baseline.model);
    this.perform(() => this.callbacks.editTensor(this.cellEdit(value)));
  }

  private updateTensorEditor(draw: boolean): void {
    const model = this.model; if (!model) return;
    this.row = clamp(this.row, 0, model.hiddenNames.length - 1); this.column = clamp(this.column, 0, model.featureNames.length - 1);
    const matrix = this.tensor === 'w1', vector = this.tensor === 'b1' || this.tensor === 'w2';
    this.ref('matrix').hidden = !matrix; this.ref('vector').hidden = !vector;
    this.ref('row-field').hidden = !matrix && !vector; this.ref('column-field').hidden = !matrix;
    for (const button of this.panel.querySelectorAll<HTMLElement>('[data-tensor]')) button.setAttribute('aria-pressed', String(button.dataset.tensor === this.tensor));
    const shape = matrix ? `${model.w1.length} × ${model.featureNames.length} · ${model.w1.flat().length} values` : vector ? `${model.hiddenNames.length} coefficients` : '1 scalar';
    setText(this.ref('tensor-shape'), shape);
    const descriptions: Record<Tensor, string> = {
      w1: `W₁ maps ${model.featureNames.length} features to ${model.hiddenNames.length} hidden neurons. Every cell is selectable; arrow keys move through the matrix.`,
      b1: 'b₁ is added before ReLU. A more negative bias requires stronger input before that neuron activates.',
      w2: 'W₂ turns each hidden activation into a signed contribution to the candidate score. These are the same coefficients as the semantic controls.',
      b2: 'b₂ is the shared scalar output bias. Adding it shifts all candidate scores equally.',
      temperature: 'Temperature scales scores before softmax. Lower values concentrate probability; higher values spread it across candidates.',
    };
    setText(this.ref('tensor-description'), descriptions[this.tensor]);
    const address = matrix ? `W1[${this.row}, ${this.column}]` : vector ? `${this.tensor.toUpperCase()}[${this.row}]` : this.tensor === 'b2' ? 'b2' : 'temperature';
    setText(this.ref('cell-address'), address);
    this.ref('cell-address').title = matrix ? `${model.hiddenNames[this.row]} ← ${model.featureNames[this.column]}` : vector ? model.hiddenNames[this.row] : descriptions[this.tensor];
    inputValue(this.ref<HTMLSelectElement>('row-select'), String(this.row)); inputValue(this.ref<HTMLSelectElement>('column-select'), String(this.column));
    const value = this.ref<HTMLInputElement>('cell-value'); value.min = this.tensor === 'temperature' ? '0.05' : '-64'; value.max = this.tensor === 'temperature' ? '20' : '64';
    value.setAttribute('aria-label', `${address} parameter value`); inputValue(value, String(this.cellValue(model)));
    setText(this.ref('cell-baseline'), this.snapshot ? `Frozen baseline ${number(this.cellValue(this.snapshot.baseline.model), 6)}` : '');
    setText(this.ref('cell-note'), this.tensor === 'temperature' ? 'Valid range: 0.05 to 20. Temperature is stored separately from the 1,293 neural parameters.' : `Valid range: −64 to 64. ${this.snapshot?.liveApply ? 'Valid edits apply live.' : 'Edits are staged until you apply them.'}`);
    if (vector) this.vectorButtons.forEach((button, index) => {
      setText(button.querySelector('code')!, number(model[this.tensor as 'b1' | 'w2'][index]));
      button.setAttribute('aria-pressed', String(index === this.row));
    });
    if (draw) this.drawMatrix();
  }

  private drawMatrix(): void {
    const model = this.model; if (!model || !this.isOpen || this.tab !== 'tensors' || this.tensor !== 'w1') return;
    const canvas = this.ref<HTMLCanvasElement>('matrix-canvas'), context = canvas.getContext('2d'); if (!context) return;
    const rows = model.hiddenNames.length, columns = model.featureNames.length;
    const width = canvas.width / columns, height = canvas.height / rows;
    const magnitude = Math.max(1, ...model.w1.flat().map(Math.abs));
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
      const value = model.w1[row][column], intensity = Math.sqrt(Math.abs(value) / magnitude);
      const start = [27, 53, 54], end = value < 0 ? [110, 180, 199] : [233, 191, 111];
      context.fillStyle = `rgb(${start.map((base, index) => Math.round(base + (end[index] - base) * intensity)).join(',')})`;
      context.fillRect(column * width + 1, row * height + 1, width - 2, height - 2);
    }
    context.strokeStyle = '#fff3d8'; context.lineWidth = 2.5;
    context.strokeRect(this.column * width + 1.5, this.row * height + 1.5, width - 3, height - 3);
    setText(this.ref('matrix-range'), `SCALE ±${number(magnitude)}`);
    for (const [index, label] of [...this.ref('matrix-rows').children].entries()) label.classList.toggle('wl-axis-selected', index === this.row);
    for (const [index, label] of [...this.ref('matrix-columns').children].entries()) label.classList.toggle('wl-axis-selected', index === this.column);
    canvas.setAttribute('aria-label', `W1 ${rows} by ${columns} matrix. Selected ${model.hiddenNames[this.row]}, ${model.featureNames[this.column]}, value ${model.w1[this.row][this.column]}. Arrow keys select; Enter edits.`);
  }

  private matrixKey(event: KeyboardEvent): void {
    if (!this.model) return;
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'Enter'].includes(event.key)) event.preventDefault();
    if (event.key === 'ArrowLeft') this.column = Math.max(0, this.column - 1);
    else if (event.key === 'ArrowRight') this.column = Math.min(this.model.featureNames.length - 1, this.column + 1);
    else if (event.key === 'ArrowUp') this.row = Math.max(0, this.row - 1);
    else if (event.key === 'ArrowDown') this.row = Math.min(this.model.hiddenNames.length - 1, this.row + 1);
    else if (event.key === 'Home') this.column = 0;
    else if (event.key === 'End') this.column = this.model.featureNames.length - 1;
    else if (event.key === 'Enter') { this.ref<HTMLInputElement>('cell-value').focus(); this.ref<HTMLInputElement>('cell-value').select(); }
    else return;
    this.updateTensorEditor(true);
  }

  private readConfig(): WeightExperimentConfig {
    if (!this.snapshot) throw new Error('The commander model is not ready.');
    const config = { ...this.snapshot.search.config };
    for (const field of this.panel.querySelectorAll<HTMLInputElement>('[data-config]')) {
      const key = field.dataset.config as keyof WeightExperimentConfig;
      if (key === 'developmentSeeds' || key === 'validationSeeds') {
        const parts = field.value.split(/[\s,]+/).filter(Boolean);
        const seeds = parts.map(Number);
        if (!seeds.length || seeds.length > 8 || seeds.some(seed => !Number.isInteger(seed) || seed < 1 || seed > 0xffffffff) || new Set(seeds).size !== seeds.length)
          throw new Error(`${key === 'developmentSeeds' ? 'Development' : 'Validation'} seeds must contain 1–8 unique whole numbers from 1 to 4294967295.`);
        config[key] = seeds;
      } else {
        const value = Number(field.value); if (!Number.isFinite(value)) throw new Error('Experiment settings must be finite numbers.');
        config[key] = value;
      }
    }
    if (config.validationSeeds.some(seed => config.developmentSeeds.includes(seed))) throw new Error('Held-out validation seeds must not overlap development seeds.');
    return config;
  }

  private startSearch(): void {
    const form = this.ref<HTMLFormElement>('search-form'); if (!form.reportValidity() || this.snapshot?.search.state === 'running') return;
    try {
      const config = this.readConfig(), autoPromote = this.ref<HTMLInputElement>('auto-promote').checked;
      this.perform(() => this.callbacks.startSearch(config, autoPromote));
    } catch (error) { this.showError(error); }
  }

  private updateSearch(): void {
    const search = this.snapshot?.search; if (!search) return;
    const running = search.state === 'running';
    if (running && this.lastSearchState !== 'running') this.configDirty = false;
    if (!this.configInitialized || !this.configDirty) {
      for (const field of this.panel.querySelectorAll<HTMLInputElement>('[data-config]')) {
        const value = search.config[field.dataset.config as keyof WeightExperimentConfig];
        inputValue(field, Array.isArray(value) ? value.join(', ') : String(value));
      }
    }
    if (!this.configInitialized) this.ref<HTMLInputElement>('auto-promote').checked = search.autoPromote;
    if (this.lastSearchState === 'running' && !running) this.ref<HTMLInputElement>('auto-promote').checked = false;
    this.configInitialized = true; this.lastSearchState = search.state;
    this.ref<HTMLFieldSetElement>('search-fields').disabled = running;
    this.ref<HTMLButtonElement>('search-start').disabled = running;
    this.ref<HTMLButtonElement>('search-stop').disabled = !running;
    const progress = search.progress;
    const titles = { idle: 'Ready to experiment', running: progress?.phase === 'validation' ? 'Validating the finalist' : 'Searching development seeds', complete: 'Experiment complete', cancelled: 'Experiment stopped', error: 'Experiment needs attention' };
    setText(this.ref('search-state'), titles[search.state]);
    const completion = search.state === 'complete' ? 100 : progress ? clamp(progress.percent, 0, 100) : 0;
    this.ref<HTMLProgressElement>('search-progress').value = completion;
    setText(this.ref('search-percent'), progress || search.state === 'complete' ? `${Math.round(completion)}%` : '—');
    const detail = search.error || (progress ? `${progress.matchesCompleted}/${progress.totalMatches} matches · candidate ${progress.candidateIndex + 1}/${progress.candidateCount} · seed ${progress.seed} · ${number(progress.simulationSeconds, 1)}/${number(progress.maxSeconds, 1)}s` : running ? 'Starting independent seeded worlds…' : 'No simulations have run in this session.');
    setText(this.ref('search-detail'), detail);
    setText(this.ref('search-best'), progress?.bestDevelopmentFitness === null || progress?.bestDevelopmentFitness === undefined ? '' : `Best development shaped fitness ${signed(progress.bestDevelopmentFitness, 4)}${running && search.autoPromote ? ' · one-shot promotion armed' : ''}`);
    const result = search.result;
    this.ref('search-results').hidden = !result;
    if (!result) { this.renderedResult = undefined; return; }
    setText(this.ref('result-title'), result.promotion.recommended ? 'Validation gate passed.' : 'Keep the active commander.');
    setText(this.ref('result-fingerprint'), result.candidate.fingerprint.slice(0, 10)); this.ref('result-fingerprint').title = result.candidate.fingerprint;
    const promote = this.ref<HTMLButtonElement>('promote'); promote.disabled = !search.canPromote || !result.promotion.recommended;
    const alreadyActive = result.candidate.fingerprint === this.snapshot!.active.fingerprint;
    const sourceChanged = result.baseRevision !== this.snapshot!.active.revision || result.baseFingerprint !== this.snapshot!.active.fingerprint;
    setText(this.ref('promotion-reason'), `${result.promotion.reason}${result.promotion.recommended && sourceChanged && !alreadyActive ? ` Active weights have changed since source r${result.baseRevision}; run a new experiment before promoting.` : ''}`);
    setText(promote, alreadyActive ? 'Candidate is active' : 'Promote validated candidate');
    if (result !== this.renderedResult) {
      const metrics = this.ref('result-metrics'); metrics.replaceChildren();
      const validation = result.validation;
      const values: [string, string][] = [['WINS', String(validation.wins)], ['LOSSES', String(validation.losses)], ['DRAWS', String(validation.draws)], ['SHAPED FITNESS', signed(validation.meanShapedFitness, 4)]];
      for (const [label, value] of values) { const block = document.createElement('div'), strong = document.createElement('b'), caption = document.createElement('span'); strong.textContent = value; caption.textContent = label; block.append(strong, caption); metrics.append(block); }
      const footnote = document.createElement('p'); footnote.className = 'wl-footnote wl-result-footnote'; footnote.textContent = `${validation.timeLimitDraws} time-limit draws · ${validation.illegalCommands} illegal commands · source r${result.baseRevision}`; metrics.append(footnote);
      const development = this.ref('development-results'); development.replaceChildren();
      result.development.forEach((evaluation, index) => {
        const row = document.createElement('div'); row.className = 'wl-evaluation-row';
        const name = document.createElement('span'), score = document.createElement('code'), record = document.createElement('small');
        name.textContent = `Evaluation ${index + 1}`; score.textContent = signed(evaluation.meanShapedFitness, 4); record.textContent = `${evaluation.wins}W / ${evaluation.losses}L / ${evaluation.draws}D`;
        row.append(name, score, record); development.append(row);
      });
      this.renderedResult = result;
    }
  }

  private updateHistory(): void {
    if (!this.snapshot) return;
    const history = this.snapshot.history;
    setText(this.ref('history-count'), `${history.length} retained revisions`);
    const signature = `${this.snapshot.active.id}:${history.map(entry => entry.id).join('|')}`;
    const host = this.ref('history');
    if (signature === this.historySignature || host.contains(document.activeElement)) return;
    host.replaceChildren();
    for (const entry of [...history].sort((a, b) => b.revision - a.revision)) {
      const active = entry.id === this.snapshot.active.id;
      const item = document.createElement('li'); item.className = `wl-history-item${active ? ' wl-history-active' : ''}`;
      const head = document.createElement('div'); head.className = 'wl-history-top';
      const revision = document.createElement('b'); revision.textContent = `r${entry.revision}`;
      const fingerprint = document.createElement('code'); fingerprint.textContent = entry.fingerprint.slice(0, 12); fingerprint.title = entry.fingerprint;
      const time = document.createElement('time'); time.dateTime = entry.createdAt; time.title = entry.createdAt; time.textContent = new Date(entry.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      head.append(revision, fingerprint, time);
      const source = document.createElement('span'); source.className = 'wl-history-source'; source.textContent = `${pretty(entry.provenance.source)}${entry.provenance.tick === null ? '' : ` · tick ${entry.provenance.tick}`}`;
      const reason = document.createElement('p'); reason.textContent = entry.provenance.reason;
      const restore = document.createElement('button'); restore.type = 'button'; restore.className = 'wl-button'; restore.dataset.restore = entry.id; restore.disabled = active; restore.textContent = active ? 'Active' : 'Restore'; restore.setAttribute('aria-label', active ? `Revision ${entry.revision} is active` : `Restore revision ${entry.revision}`);
      item.append(head, source, reason, restore); host.append(item);
    }
    this.historySignature = signature;
  }

  private updateCoach(): void {
    const coach = this.snapshot?.coach;
    const button = this.ref<HTMLButtonElement>('coach'); button.disabled = !coach?.available || !!coach.busy || !this.callbacks.requestLayaCoach;
    setText(button, coach?.busy ? 'Laya is considering…' : 'Ask Laya');
    setText(this.ref('coach-status'), coach?.message || (coach?.available ? 'Ask the connected checkpoint for a tactical suggestion. Its RTS ability remains unvalidated.' : 'Connect Laya to request a tactical suggestion.'));
    const result = this.ref('coach-result'); result.hidden = !coach?.suggestion; setText(result, coach?.suggestion ?? '');
  }

  private async exportCheckpoint(): Promise<void> {
    try {
      const json = await this.callbacks.exportCheckpoint();
      JSON.parse(json);
      this.ref<HTMLTextAreaElement>('checkpoint').value = json;
      const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
      const link = document.createElement('a'); link.href = url; link.download = `crown-charter-weights-r${this.snapshot?.active.revision ?? 0}.json`; link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      this.localError = ''; this.localStatus = 'Exact checkpoint exported. Its JSON is also available above.'; this.updateStatus();
    } catch (error) { this.showError(error); }
  }

  private importCheckpoint(): void {
    const text = this.ref<HTMLTextAreaElement>('checkpoint').value.trim();
    try {
      if (!text) throw new Error('Paste checkpoint JSON or choose a JSON file first.');
      JSON.parse(text);
      this.perform(() => this.callbacks.importCheckpoint(text), 'Checkpoint import completed.');
    } catch (error) { this.showError(error); }
  }

  private async readCheckpointFile(): Promise<void> {
    const field = this.ref<HTMLInputElement>('checkpoint-file'), file = field.files?.[0]; if (!file) return;
    try {
      if (file.size > 2_000_000) throw new Error('Choose a commander checkpoint smaller than 2 MB.');
      const text = await file.text(); JSON.parse(text);
      this.ref<HTMLTextAreaElement>('checkpoint').value = text;
      this.localError = ''; this.localStatus = `${file.name} loaded. Select Validate & import to apply it.`; this.updateStatus();
    } catch (error) { this.showError(error); }
    field.value = '';
  }
}
