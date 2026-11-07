import { createDefaultModel, forwardPolicy, policyFeatures, tracePolicy } from './policy';
import { World, type WorldCheckpoint } from './simulation';
import type { Command, Team } from './types';
import { getPlayCandidates, pickPlayPolicy, shouldPlayPolicyStep } from './play-policy';
import { WeightRuntime, type WeightEdit, type WeightSnapshot } from './weight-runtime';
import { LayaWeightCoach, type WeightCoachDecision } from './weight-coach';
import { getPlayLayaStatus, getPlayLayaProposalEligibility, playLayaWeightProposals, type PlayLayaStatus } from './play-laya';
import type { PlayExperimentChoice, PlayExperimentRequest, PlayExperimentResult } from './play-experiment';
import { issueTrackedPlayOrder, observePlayOrder, type PlayOrderImpact } from './play-impact';
import { PlayAstraLoop } from './play-astra';
import { PlayAstraPanel } from './play-astra-panel';

type Control = 'manual' | 'policy';
type Tab = 'astra' | 'learn' | 'weights' | 'laya';
type Tensor = 'w1' | 'b1' | 'w2' | 'b2';
interface Capture { checkpoint: WorldCheckpoint; time: number; tick: number }
interface IssuedOrder { label: string; time: number; revision: number | null; source: 'policy' | 'you' }
interface ExperimentReceipt { result: PlayExperimentResult; team: Team; revision: number; baselineRevision: number; fingerprint: string; otherRevision: number }
interface ProposalReceipt { decision: WeightCoachDecision; team: Team; time: number }
interface WatchedOrder {
  label: string; baselineLabel: string; revision: number; fingerprint: string; time: number;
  impact: PlayOrderImpact | null;
}

export interface PlayProgrammingHooks {
  world(): World;
  available(): boolean;
  paused(): boolean;
  pause(): void;
  resume(): void;
  watchOrder(team: Team, command: Command): void;
  onOrder(team: Team, command: Command, label: string, revision: number): void;
  notify(message: string): void;
}

const authored = createDefaultModel();
const recruitmentRow = authored.hiddenNames.indexOf('army_train_bias');
export const PLAY_WEIGHT_CONTROLS = [
  { name: 'labor_coverage_gap', label: 'Gather resources', hint: 'Value putting workers on an uncovered resource job.' },
  { name: 'army_train_bias', label: 'Recruit soldiers', hint: 'Value training a soldier when a building can afford it.' },
  { name: 'attack_bias', label: 'Launch attacks', hint: 'Value sending available troops against the other town.' },
  { name: 'visible_threat_defend', label: 'Defend the village', hint: 'Value a defense order when enemies threaten home.' },
].map(control => ({ ...control, row: authored.hiddenNames.indexOf(control.name) }));

const element = <T extends HTMLElement = HTMLElement>(id: string) => {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing programming control: ${id}`);
  return found as T;
};
const escape = (value: unknown) => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]!));
const clock = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const signed = (value: number, precision = 1) => `${value > 0 ? '+' : ''}${value.toFixed(precision)}`;
const human = (name: string) => name.replaceAll('_', ' ');
const teamName = (team: Team) => team === 0 ? 'Blue' : 'Red';
const setText = (id: string, value: string) => { const node = element(id); if (node.textContent !== value) node.textContent = value; };
const renderedHTML = new WeakMap<HTMLElement, string>();
const setHTML = (id: string, value: string) => {
  const node = element(id);
  if (renderedHTML.get(node) === value) return;
  node.innerHTML = value; renderedHTML.set(node, value);
};

/** One policy surface for the same orders, economy and combat visible on the battlefield. */
export class PlayProgramming {
  private readonly runtimes: [WeightRuntime, WeightRuntime] = [new WeightRuntime(), new WeightRuntime()];
  private readonly controls: [Control, Control] = ['manual', 'policy'];
  private readonly captures: [Capture | null, Capture | null] = [null, null];
  private readonly issued: [IssuedOrder | null, IssuedOrder | null] = [null, null];
  private readonly watched: [WatchedOrder | null, WatchedOrder | null] = [null, null];
  private readonly undo: [WeightSnapshot[], WeightSnapshot[]] = [[], []];
  private readonly lastDecision: [string | null, string | null] = [null, null];
  private readonly coach = new LayaWeightCoach(undefined, undefined, playLayaWeightProposals);
  private readonly astra: PlayAstraLoop;
  private readonly astraPanel: PlayAstraPanel;
  private team: Team = 0;
  private tab: Tab = 'astra';
  private matchId = crypto.randomUUID();
  private gesture: string | null = null;
  private editorSignature = '';
  private neuronNamesSignature = '';
  private historySignature = '';
  private matrixSignature = '';
  private worker: Worker | null = null;
  private experimentId = 0;
  private experimentBusy = false;
  private experimentReceipt: ExperimentReceipt | null = null;
  private resultSignature = '';
  private experimentMessage = 'Capture a state, edit a weight, then compare two real 60-second simulations.';
  private layaStatus: PlayLayaStatus | null = null;
  private statusGeneration = 0;
  private proposalGeneration = 0;
  private proposal: ProposalReceipt | null = null;
  private layaMessage = 'Checking the local native Laya service…';
  private disposed = false;

  constructor(private readonly hooks: PlayProgrammingHooks) {
    this.astra = new PlayAstraLoop({
      world: hooks.world, available: hooks.available, paused: hooks.paused,
      matchId: () => this.matchId, runtime: team => this.runtimes[team],
      policyEnabled: team => this.controls[team] === 'policy',
      startPolicy: () => {
        this.controls[0] = 'policy'; this.team = 0; this.tab = 'astra';
        element<HTMLSelectElement>('program-team').value = '0';
        this.editorSignature = ''; this.matrixSignature = ''; this.historySignature = '';
        this.renderTabs(); this.hooks.resume();
      },
      onApplied: (before, after) => {
        this.undo[0].push(before);
        if (this.undo[0].length > 40) this.undo[0].shift();
        this.gesture = null; this.invalidateProposal();
        this.cancelExperiment(false);
        this.experimentReceipt = null;
        this.experimentMessage = 'Astra changed the live weights. Capture a new state to run a manual comparison.';
        this.editorSignature = ''; this.matrixSignature = ''; this.historySignature = '';
        this.hooks.notify(`Astra applied policy revision ${after.revision}. Blue is now playing with those weights.`);
      },
    });
    this.astraPanel = new PlayAstraPanel(this.astra, {
      world: hooks.world, available: hooks.available, paused: hooks.paused,
      changed: () => this.render(),
    });
    this.buildControls();
    this.bind();
    this.renderTabs();
    this.render();
    void this.refreshLaya();
    void this.astra.refreshConnection().then(() => this.render());
  }

  private get runtime() { return this.runtimes[this.team]; }

  private buildControls() {
    element('program-controls').innerHTML = PLAY_WEIGHT_CONTROLS.map(control => `<div class="weight-control">
      <label for="program-weight-${control.row}">${escape(control.label)}<output id="program-output-${control.row}" for="program-weight-${control.row}"></output></label>
      <p id="program-hint-${control.row}">${escape(control.hint)} <code>W2[${control.row}]</code></p>
      <input id="program-weight-${control.row}" data-weight-row="${control.row}" type="range" min="-12" max="24" step="any" aria-describedby="program-hint-${control.row}">
    </div>`).join('');
    element<HTMLSelectElement>('program-neuron').innerHTML = authored.hiddenNames.map((name, row) => `<option value="${row}">${row} · ${escape(human(name))}</option>`).join('');
    element<HTMLSelectElement>('program-feature').innerHTML = authored.featureNames.map((name, column) => `<option value="${column}">${column} · ${escape(human(name))}</option>`).join('');
  }

  private bind() {
    element('program-team').addEventListener('change', () => {
      this.cancelExperiment(false);
      this.invalidateProposal();
      this.team = element<HTMLSelectElement>('program-team').value === '1' ? 1 : 0;
      this.gesture = null; this.editorSignature = ''; this.matrixSignature = ''; this.historySignature = '';
      this.experimentReceipt = null;
      this.experimentMessage = 'Experiments compare this civilization with its saved baseline.';
      this.render();
    });
    element('program-control').addEventListener('click', event => {
      const button = (event.target as HTMLElement).closest<HTMLButtonElement>('[data-control]');
      if (!button || !this.hooks.available()) return;
      this.controls[this.team] = button.dataset.control === 'policy' ? 'policy' : 'manual';
      if (this.team === 0 && this.controls[0] === 'manual') this.astra.stop('You took manual control. Pending Astra edits were canceled.');
      if (this.controls[this.team] === 'policy') this.hooks.resume();
      this.hooks.notify(this.controls[this.team] === 'policy'
        ? `${teamName(this.team)} network is playing. It issues the highest-scoring legal order every two game seconds.`
        : `${teamName(this.team)} policy stopped issuing orders. Units finish their existing orders.`);
      this.render();
    });
    document.querySelectorAll<HTMLButtonElement>('[data-program-tab]').forEach(button => button.addEventListener('click', () => {
      this.tab = button.dataset.programTab as Tab;
      if (this.tab === 'astra') {
        this.team = 0; element<HTMLSelectElement>('program-team').value = '0';
        this.editorSignature = ''; this.matrixSignature = ''; this.historySignature = '';
      }
      this.renderTabs(); this.render();
    }));
    element('program-controls').addEventListener('input', event => {
      const input = event.target as HTMLInputElement;
      if (!input.matches('input[data-weight-row]')) return;
      const row = Number(input.dataset.weightRow);
      this.edit([{ tensor: 'w2', row, value: Number(input.value) }], `Adjusted W2[${row}] (${authored.hiddenNames[row]})`, `slider:${this.team}:${row}`);
    });
    element('program-controls').addEventListener('change', () => { this.gesture = null; });
    for (const id of ['program-focus-value', 'program-focus-slider']) {
      const input = element<HTMLInputElement>(id);
      input.addEventListener('input', () => {
        const value = Number(input.value);
        const valid = input.value.trim() !== '' && Number.isFinite(value) && value >= -64 && value <= 64;
        input.setCustomValidity(valid ? '' : 'Enter a finite weight between −64 and +64.');
        if (valid) this.edit([{ tensor: 'w2', row: recruitmentRow, value }], `Set recruitment weight W2[${recruitmentRow}] to ${value}`, `focus:${this.team}`);
      });
      input.addEventListener('change', () => { this.gesture = null; input.reportValidity(); });
      input.addEventListener('blur', () => {
        input.setCustomValidity(''); input.value = String(this.runtime.current.model.w2[recruitmentRow]);
        this.gesture = null;
      });
    }
    element('program-step').addEventListener('click', () => {
      if (!this.hooks.available()) return;
      this.astra.stop('You chose to execute a manual policy step. Pending Astra edits were canceled.');
      this.hooks.pause();
      this.controls[this.team] = 'manual';
      this.execute(this.team, true);
      this.render();
    });
    element('program-pause').addEventListener('click', () => {
      if (!this.hooks.available()) return;
      if (this.hooks.paused()) { this.hooks.resume(); this.render(); return; }
      this.hooks.pause(); this.capture(false); this.invalidateExperiment();
      this.experimentMessage = `State frozen at ${clock(this.hooks.world().time)}. Your baseline weights are preserved; edits can now be compared on this state.`;
      this.render();
    });
    element('program-pin').addEventListener('click', () => {
      if (!this.hooks.available()) return;
      this.hooks.pause(); this.capture(true); this.invalidateExperiment();
      this.experimentMessage = `Baseline r${this.runtime.baseline.revision} captured at ${clock(this.hooks.world().time)}. Change a weight to test another policy.`;
      this.render();
    });
    element('program-demo').addEventListener('click', () => this.tryRecruitmentEdit());
    element('program-undo').addEventListener('click', () => {
      const previous = this.undo[this.team].pop();
      if (!previous) return;
      this.runtime.apply(previous.model, { source: 'play-undo', reason: `Undo to values from revision ${previous.revision}`, tick: this.hooks.world().tick });
      this.afterEdit('Undid the last edit.');
    });
    element('program-reset').addEventListener('click', () => {
      this.rememberUndo();
      this.runtime.reset({ source: 'play-baseline', reason: 'Restore the captured baseline weights', tick: this.hooks.world().tick });
      this.afterEdit('Baseline weights restored.');
    });
    element('program-compare').addEventListener('click', () => this.compare());
    element('program-compare-cancel').addEventListener('click', () => this.cancelExperiment(true));
    for (const id of ['program-neuron', 'program-feature', 'program-tensor']) element(id).addEventListener('change', () => {
      this.editorSignature = ''; this.matrixSignature = ''; this.renderEditor();
    });
    element('program-apply').addEventListener('click', () => {
      const value = Number(element<HTMLInputElement>('program-value').value);
      if (element<HTMLInputElement>('program-value').value.trim() === '' || !Number.isFinite(value) || value < -64 || value > 64) {
        setText('program-tensor-detail', 'Enter a finite weight between −64 and +64.'); return;
      }
      const coordinate = this.coordinate();
      this.edit([{ ...coordinate, value }], `Set ${this.coordinateLabel()} to ${value}`);
    });
    element('program-matrix').addEventListener('click', event => {
      const button = (event.target as HTMLElement).closest<HTMLButtonElement>('[data-weight-cell]');
      if (!button) return;
      const tensor = this.coordinate().tensor;
      if (tensor === 'w1') element<HTMLSelectElement>('program-feature').value = button.dataset.weightCell!;
      else if (tensor !== 'b2') element<HTMLSelectElement>('program-neuron').value = button.dataset.weightCell!;
      this.editorSignature = ''; this.matrixSignature = ''; this.renderEditor();
      element<HTMLInputElement>('program-value').focus();
    });
    element('program-history').addEventListener('click', event => {
      const button = (event.target as HTMLElement).closest<HTMLButtonElement>('[data-restore]');
      if (!button) return;
      this.rememberUndo();
      this.runtime.restore(button.dataset.restore!, { source: 'play-history', reason: 'Restored a prior policy revision', tick: this.hooks.world().tick });
      this.afterEdit('Policy revision restored.');
    });
    element('program-export').addEventListener('click', () => {
      const url = URL.createObjectURL(new Blob([this.runtime.exportCheckpoint()], { type: 'application/json' }));
      const anchor = document.createElement('a');
      anchor.href = url; anchor.download = `crown-charter-${teamName(this.team).toLowerCase()}-r${this.runtime.current.revision}.json`;
      anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    element('program-import-trigger').addEventListener('click', () => element<HTMLInputElement>('program-import').click());
    element('program-import').addEventListener('change', () => { void this.importWeights(); });
    element('program-laya-refresh').addEventListener('click', () => { void this.refreshLaya(); });
    element('program-laya-request').addEventListener('click', () => { void this.requestLaya(); });
    element('program-laya-apply').addEventListener('click', () => this.applyLaya());
    element('program-laya-reject').addEventListener('click', () => {
      const pending = this.coach.busy;
      this.invalidateProposal();
      this.layaMessage = pending ? 'Request canceled. No weights changed.' : 'Proposal discarded. No weights changed.';
      this.renderLaya();
    });
  }

  /** Battle restarts retain policy work; a fresh lesson restores authored weights. */
  startMatch(freshLesson = false) {
    this.astra.reset();
    this.cancelExperiment(false); this.invalidateProposal();
    this.matchId = crypto.randomUUID();
    this.controls[0] = 'manual'; this.controls[1] = 'policy';
    this.team = 0; element<HTMLSelectElement>('program-team').value = '0';
    this.tab = freshLesson ? 'astra' : 'learn'; this.renderTabs();
    for (const team of [0, 1] as const) {
      if (freshLesson) this.runtimes[team] = new WeightRuntime();
      this.runtimes[team].pinBaseline(); this.captures[team] = null;
      this.issued[team] = null; this.watched[team] = null; this.lastDecision[team] = null; this.undo[team] = [];
    }
    this.experimentReceipt = null; this.resultSignature = ''; this.gesture = null;
    this.editorSignature = ''; this.matrixSignature = ''; this.historySignature = '';
    this.experimentMessage = 'Edit W2[19] and execute its decision in the live world. You can also compare two simulated 60-second futures below.';
    if (freshLesson) this.astra.start();
    this.render();
  }

  takeCommand() {
    this.astra.stop('You took command of blue. Pending Astra edits were canceled; existing unit orders continue.');
    this.controls[0] = 'manual';
    this.render();
  }

  manualOrder(label: string) {
    const wasPolicy = this.controls[0] === 'policy';
    this.astra.stop('Your manual order took control of blue. Pending Astra edits were canceled.');
    this.controls[0] = 'manual';
    this.issued[0] = { label, time: this.hooks.world().time, revision: null, source: 'you' };
    this.lastDecision[0] = null;
    if (this.proposal || this.coach.busy) {
      this.invalidateProposal(); this.layaMessage = 'A game order changed the state. Request a fresh Laya proposal.';
    }
    this.render();
    return wasPolicy;
  }

  /** Called after each fixed simulation step; shares its schedule with experiments. */
  tick() {
    const world = this.hooks.world();
    this.astra.tick();
    for (const receipt of this.watched) if (receipt?.impact) observePlayOrder(world, receipt.impact);
    if (!shouldPlayPolicyStep(world)) return;
    for (const team of [0, 1] as const) if (this.controls[team] === 'policy') this.execute(team, false);
  }

  private execute(team: Team, explicit: boolean) {
    const world = this.hooks.world(), snapshot = this.runtimes[team].current;
    const candidate = pickPlayPolicy(world, team, snapshot.model);
    if (!candidate) { if (explicit) this.hooks.notify('No legal decisions are available.'); return; }
    this.lastDecision[team] = candidate.label;
    const receipt: WatchedOrder | null = explicit ? {
      label: candidate.label, baselineLabel: pickPlayPolicy(world, team, this.runtimes[team].baseline.model)?.label ?? 'No order',
      revision: snapshot.revision, fingerprint: snapshot.fingerprint, time: world.time, impact: null,
    } : null;
    if (!candidate.command) {
      if (explicit) {
        this.watched[team] = receipt;
        this.hooks.resume();
        this.hooks.notify('The network chose to keep existing orders. The simulation is running; no new order was issued.');
      }
      return;
    }
    const response = explicit ? issueTrackedPlayOrder(world, team, candidate.command) : world.command(team, candidate.command);
    if (!response.ok) { if (explicit) this.hooks.notify(response.reason ?? 'This order is no longer available.'); return; }
    if (this.proposal || this.coach.busy) {
      this.invalidateProposal(); this.layaMessage = 'A game order changed the state. Request a fresh Laya proposal.';
    }
    this.issued[team] = { label: candidate.label, time: world.time, revision: snapshot.revision, source: 'policy' };
    if (team === 0) this.astra.recordOrder(candidate.label, snapshot.revision);
    if (receipt && 'impact' in response) {
      receipt.impact = response.impact as PlayOrderImpact;
      this.watched[team] = receipt;
    }
    this.hooks.onOrder(team, candidate.command, candidate.label, snapshot.revision);
    if (explicit) {
      this.hooks.watchOrder(team, candidate.command);
      this.hooks.resume();
      this.hooks.notify(`${teamName(team)} network r${snapshot.revision} chose ${candidate.label}. Watch its real progress below.`);
    }
  }

  private capture(pin: boolean) {
    if (pin) this.runtime.pinBaseline();
    const world = this.hooks.world();
    this.captures[this.team] = { checkpoint: world.checkpoint(), tick: world.tick, time: world.time };
  }

  private rememberUndo() {
    this.undo[this.team].push(this.runtime.current);
    if (this.undo[this.team].length > 40) this.undo[this.team].shift();
  }

  private edit(edits: WeightEdit[], reason: string, gesture?: string) {
    if (!this.hooks.available()) return;
    const current = this.runtime.current;
    if (edits.every(edit => this.valueOf(edit) === edit.value)) return;
    if (!this.captures[this.team]) this.capture(false);
    try {
      this.runtime.patch(edits, { source: 'play-editor', reason, tick: this.hooks.world().tick });
      if (!gesture || this.gesture !== gesture) {
        this.undo[this.team].push(current);
        if (this.undo[this.team].length > 40) this.undo[this.team].shift();
      }
      this.gesture = gesture ?? null;
      this.afterEdit(reason, !!gesture);
    } catch (error) { this.hooks.notify(error instanceof Error ? error.message : 'Could not change this weight.'); }
  }

  private afterEdit(message: string, preserveGesture = false) {
    this.astra.stop('You changed the weights manually. Pending Astra edits were canceled. Start AI + Astra to resume automatic reviews.');
    if (!preserveGesture) this.gesture = null;
    this.invalidateProposal();
    if (this.experimentBusy) this.cancelExperiment(false);
    if (this.experimentReceipt) this.experimentMessage = 'Weights changed after the displayed experiment. Run it again to test this revision.';
    this.editorSignature = ''; this.matrixSignature = ''; this.historySignature = '';
    this.render();
    if (!preserveGesture) this.hooks.notify(message);
  }

  private tryRecruitmentEdit() {
    if (!this.hooks.available()) return;
    const world = this.hooks.world();
    const candidates = getPlayCandidates(world, this.team);
    const target = candidates.find(candidate => candidate.command?.type === 'train' && candidate.command.kind !== 'villager');
    if (!target) {
      this.hooks.notify(this.team === 1 && world.time < 270
        ? 'Red cannot recruit before 04:30 in this beginner mission. Choose Your civilization to try recruitment now.'
        : 'No soldier can be trained right now. You need an available Barracks, population space, 50 food and 25 wood.');
      return;
    }
    this.hooks.pause(); this.capture(false); this.invalidateExperiment();
    this.edit([{ tensor: 'w2', row: recruitmentRow, value: 24 }], `Set the actual recruitment weight, W2[${recruitmentRow}], to 24`);
    this.experimentMessage = `Game paused at ${clock(world.time)}. Your original baseline is preserved. Execute & watch runs the current winning order in this world.`;
    this.render();
  }

  private coordinate(): Omit<WeightEdit, 'value'> & { tensor: Tensor } {
    const tensor = element<HTMLSelectElement>('program-tensor').value as Tensor;
    const row = Number(element<HTMLSelectElement>('program-neuron').value);
    const column = Number(element<HTMLSelectElement>('program-feature').value);
    return tensor === 'w1' ? { tensor, row, column } : tensor === 'b2' ? { tensor } : { tensor, row };
  }
  private coordinateLabel() {
    const value = this.coordinate();
    return `${value.tensor.toUpperCase()}${value.row === undefined ? '' : `[${value.row}]`}${value.column === undefined ? '' : `[${value.column}]`}`;
  }
  private valueOf(edit: Pick<WeightEdit, 'tensor' | 'row' | 'column'>, snapshot = this.runtime.current) {
    const model = snapshot.model;
    return edit.tensor === 'w1' ? model.w1[edit.row!][edit.column!] : edit.tensor === 'b1' || edit.tensor === 'w2' ? model[edit.tensor][edit.row!] : model[edit.tensor];
  }

  private async importWeights() {
    const input = element<HTMLInputElement>('program-import'), file = input.files?.[0];
    if (!file) return;
    const team = this.team, match = this.matchId, revision = this.runtime.current.revision;
    try {
      if (file.size > 250_000) throw new Error('Weight checkpoints must be smaller than 250 KB.');
      const text = await file.text();
      if (team !== this.team || match !== this.matchId || revision !== this.runtime.current.revision) throw new Error('The active policy changed while reading the file. Select it again to import.');
      const previous = this.runtime.current;
      if (!this.captures[team]) this.capture(false);
      this.runtime.importCheckpoint(text, { source: 'play-import', reason: `Imported ${file.name.slice(0, 120)}`, tick: this.hooks.world().tick, expectedRevision: revision });
      this.undo[team].push(previous);
      this.afterEdit('Validated checkpoint imported into the selected civilization.');
    } catch (error) { this.hooks.notify(error instanceof Error ? error.message : 'The checkpoint could not be imported.'); }
    finally { input.value = ''; }
  }

  private compare() {
    if (!this.hooks.available() || this.experimentBusy) return;
    this.hooks.pause();
    if (!this.captures[this.team]) this.capture(false);
    const capture = this.captures[this.team]!, runtime = this.runtime;
    const team = this.team, other = (1 - team) as Team;
    const revision = runtime.current.revision, fingerprint = runtime.current.fingerprint;
    const baselineRevision = runtime.baseline.revision, otherRevision = this.runtimes[other].current.revision;
    const request: PlayExperimentRequest = {
      checkpoint: capture.checkpoint, team, baselineModel: runtime.baseline.model, candidateModel: runtime.current.model,
      otherModel: this.runtimes[other].current.model, otherControl: this.controls[other] === 'policy' ? 'policy' : 'hold', seconds: 60,
    };
    this.experimentBusy = true; this.experimentReceipt = null; this.resultSignature = '';
    this.experimentMessage = `Simulating two futures from ${clock(capture.time)}. ${teamName(team)} policy runs in both, even if you command it manually in the live game.`;
    const id = ++this.experimentId;
    try {
      this.worker = new Worker(new URL('./play-experiment-worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = event => {
        const message = event.data;
        if (message.id !== this.experimentId || id !== this.experimentId) return;
        if (message.type === 'progress') {
          this.experimentMessage = `Comparing from ${clock(capture.time)} · ${Math.round(message.progress * 100)}% · live game paused`;
        } else if (message.type === 'result') {
          this.experimentBusy = false;
          this.experimentReceipt = { result: message.result, team, revision, baselineRevision, fingerprint, otherRevision };
          this.experimentMessage = `Completed from ${clock(capture.time)}. Same starting world and other civilization; only ${teamName(team)} weights differ.`;
          this.worker?.terminate(); this.worker = null;
        } else if (message.type === 'error') {
          this.experimentBusy = false; this.experimentMessage = `Experiment failed: ${message.message}`;
          this.worker?.terminate(); this.worker = null;
        }
        this.renderExperiment();
      };
      this.worker.onerror = () => {
        if (id !== this.experimentId) return;
        this.experimentBusy = false; this.experimentMessage = 'The comparison worker could not run. No result or live-game change was produced.';
        this.worker?.terminate(); this.worker = null; this.renderExperiment();
      };
      this.worker.postMessage({ type: 'run', id, request });
    } catch (error) {
      this.experimentBusy = false;
      this.experimentMessage = `Could not start the experiment: ${error instanceof Error ? error.message : 'worker unavailable'}`;
      this.worker?.terminate(); this.worker = null;
    }
    this.renderExperiment();
  }

  private cancelExperiment(announce: boolean) {
    this.experimentId++; this.worker?.terminate(); this.worker = null;
    const busy = this.experimentBusy; this.experimentBusy = false;
    if (busy) this.experimentMessage = 'Comparison canceled. Neither simulated future was applied to the live game.';
    if (announce) this.renderExperiment();
  }
  private invalidateExperiment() {
    this.cancelExperiment(false); this.experimentReceipt = null; this.resultSignature = '';
  }

  private invalidateProposal() {
    this.proposalGeneration++; this.coach.cancel(); this.proposal = null;
    if (this.layaStatus) this.layaMessage = this.layaStatus.available ? 'Ask Laya to select one of the preset edits for this state. Review its selection before applying.' : this.layaStatus.detail;
  }

  private async refreshLaya() {
    const generation = ++this.statusGeneration;
    setText('program-laya-status', 'Checking local Laya…');
    const status = await getPlayLayaStatus();
    if (generation !== this.statusGeneration || this.disposed) return;
    const changed = this.layaStatus?.checkpoint !== status.checkpoint;
    this.layaStatus = status;
    if (changed) this.invalidateProposal();
    this.layaMessage = status.available ? 'Native model available. Request a proposal for the selected civilization.' : status.detail;
    this.renderLaya();
  }

  private async requestLaya() {
    if (!this.hooks.available() || !this.layaStatus?.available || this.coach.busy) return;
    this.hooks.pause(); this.proposal = null;
    const generation = ++this.proposalGeneration;
    const team = this.team, match = this.matchId, world = this.hooks.world(), time = world.time;
    const snapshot = this.runtime.current;
    this.layaMessage = 'Game paused. Native Laya is scoring the four preset edits and “Keep unchanged”…';
    const pending = this.coach.request(world, team, match, snapshot);
    this.renderLaya();
    try {
      const decision = await pending;
      if (generation !== this.proposalGeneration || match !== this.matchId || team !== this.team || this.disposed) return;
      if (!decision) this.layaMessage = 'Request canceled or its state became stale. No weights changed.';
      else {
        this.proposal = { decision, team, time };
        const eligibility = this.proposalEligibility();
        this.layaMessage = eligibility.allowed ? 'Native proposal received. Review the exact values, then apply or discard.' : eligibility.reason;
      }
    } catch (error) {
      if (generation !== this.proposalGeneration || match !== this.matchId || team !== this.team || this.disposed) return;
      this.layaMessage = error instanceof Error ? error.message : 'Native Laya request failed. No weights changed.';
    }
    this.renderLaya();
  }

  private proposalEligibility() {
    return getPlayLayaProposalEligibility(this.proposal?.decision, {
      snapshot: this.runtime.current, matchId: this.matchId, tick: this.hooks.world().tick,
      status: this.hooks.world().status, checkpoint: this.layaStatus?.checkpoint,
      time: this.hooks.world().time, requestedTime: this.proposal?.time,
    });
  }

  private applyLaya() {
    if (!this.proposal || this.proposal.team !== this.team || !this.hooks.available()) return;
    const eligibility = this.proposalEligibility();
    if (!eligibility.allowed) { this.layaMessage = eligibility.reason; this.renderLaya(); return; }
    const proposal = this.proposal.decision;
    if (!proposal.proposal.edits.length) return;
    if (!this.captures[this.team]) this.capture(false);
    const previous = this.runtime.current;
    try {
      this.runtime.patch(proposal.proposal.edits.map(edit => ({ tensor: 'w2', row: edit.row, value: edit.value })), {
        source: 'native-laya', reason: `Native Laya chose ${proposal.proposal.id}; checkpoint ${this.layaStatus?.checkpoint ?? 'unknown'}`,
        tick: this.hooks.world().tick, expectedRevision: proposal.response.base_revision, expectedFingerprint: proposal.response.base_fingerprint,
      });
      this.undo[this.team].push(previous);
      this.afterEdit('The preset weight edit Laya selected is now applied. Execute & watch runs the resulting decision.');
      this.layaMessage = `Applied preset ${proposal.proposal.id} as revision ${this.runtime.current.revision}. Its values were authored by the developer, selected by Laya.`;
    } catch (error) { this.layaMessage = error instanceof Error ? error.message : 'Patch could not be applied.'; }
    this.renderLaya();
  }

  render() {
    if (this.disposed) return;
    const runtime = this.runtime, snapshot = runtime.current, world = this.hooks.world();
    const active = this.hooks.available();
    setText('program-revision', `${teamName(this.team)} · revision ${snapshot.revision}`);
    setText('program-fingerprint', snapshot.fingerprint.slice(-8));
    element('program-fingerprint').title = `Content fingerprint: ${snapshot.fingerprint}`;
    for (const button of element('program-control').querySelectorAll<HTMLButtonElement>('[data-control]')) {
      const selected = this.controls[this.team] === button.dataset.control;
      button.setAttribute('aria-pressed', `${selected}`); button.classList.toggle('active', selected); button.disabled = !active;
      button.textContent = button.dataset.control === 'manual' ? 'One at a time' : 'Network plays';
    }
    setText('program-mode-status', this.controls[this.team] === 'policy'
      ? `${teamName(this.team)} network ${this.hooks.paused() ? 'paused' : 'running · chooses every 2 game seconds'}. Manual orders take back control.`
      : this.hooks.paused() ? 'Paused for edits. Execute & watch runs one network decision.' : 'One decision per click. Units finish their existing orders.');
    const focusedWeight = snapshot.model.w2[recruitmentRow];
    for (const id of ['program-focus-value', 'program-focus-slider']) {
      const input = element<HTMLInputElement>(id);
      if (id === 'program-focus-slider') {
        input.min = String(Math.min(-12, focusedWeight)); input.max = String(Math.max(24, focusedWeight));
      }
      if (document.activeElement !== input && Number(input.value) !== focusedWeight) input.value = String(focusedWeight);
      input.disabled = !active;
    }
    setText('program-focus-baseline', `Saved baseline: ${runtime.baseline.model.w2[recruitmentRow].toFixed(2)}`);
    setText('program-pause', this.hooks.paused() ? 'Resume game' : 'Pause & inspect');
    for (const control of PLAY_WEIGHT_CONTROLS) {
      const value = snapshot.model.w2[control.row];
      const input = element<HTMLInputElement>(`program-weight-${control.row}`);
      input.min = String(Math.min(-12, value)); input.max = String(Math.max(24, value));
      if (Number(input.value) !== value) input.value = String(value);
      input.disabled = !active;
      setText(`program-output-${control.row}`, signed(value, Number.isInteger(value * 10) ? 1 : 2));
      const inputsChanged = snapshot.model.b1[control.row] !== authored.b1[control.row]
        || snapshot.model.w1[control.row].some((weight, column) => weight !== authored.w1[control.row][column]);
      setHTML(`program-hint-${control.row}`, `${escape(control.hint)} <code>W2[${control.row}]</code>${inputsChanged ? ' Its input weights have also changed; this label describes its original role. Inspect the trace.' : ''}`);
    }
    for (const id of ['program-step', 'program-pause', 'program-pin', 'program-demo', 'program-apply', 'program-import-trigger']) element<HTMLButtonElement>(id).disabled = !active;
    element<HTMLButtonElement>('program-undo').disabled = !active || !this.undo[this.team].length;
    element<HTMLButtonElement>('program-reset').disabled = !active || snapshot.fingerprint === runtime.baseline.fingerprint;
    this.renderDecision(world);
    this.renderImpact(world);
    this.renderExperiment();
    if (this.tab === 'weights') this.renderEditor();
    this.renderLaya();
    this.astraPanel.render(this.tab === 'astra');
  }

  private renderTabs() {
    document.body.classList.toggle('astra-mode', this.tab === 'astra');
    for (const tab of ['astra', 'learn', 'weights', 'laya'] as const) {
      element(`program-${tab}`).hidden = tab !== this.tab;
      const button = document.querySelector<HTMLButtonElement>(`[data-program-tab="${tab}"]`)!;
      button.setAttribute('aria-selected', `${tab === this.tab}`); button.classList.toggle('active', tab === this.tab);
    }
  }

  private renderDecision(world: World) {
    const model = this.runtime.current.model, baseline = this.runtime.baseline.model;
    const rows = getPlayCandidates(world, this.team).map(candidate => {
      const input = policyFeatures(world, this.team, candidate);
      return { candidate, score: forwardPolicy(input, model), baseline: forwardPolicy(input, baseline), input };
    }).sort((a, b) => b.score - a.score || (a.candidate.id < b.candidate.id ? -1 : a.candidate.id > b.candidate.id ? 1 : 0));
    const previous = [...rows].sort((a, b) => b.baseline - a.baseline || (a.candidate.id < b.candidate.id ? -1 : a.candidate.id > b.candidate.id ? 1 : 0))[0];
    const choice = rows[0];
    setText('program-baseline-choice', previous ? `${previous.candidate.label} · ${previous.baseline.toFixed(2)}` : 'No legal order');
    setText('program-edited-choice', choice ? `${choice.candidate.label} · ${choice.score.toFixed(2)}` : 'No legal order');
    const recruitment = rows.find(row => row.candidate.command?.type === 'train' && row.candidate.command.kind !== 'villager');
    const mathChoice = recruitment ?? choice;
    if (mathChoice) {
      const trace = tracePolicy(mathChoice.input, model);
      const activation = trace.hiddenActivations[recruitmentRow], weight = model.w2[recruitmentRow];
      const other = trace.score - activation * weight;
      setText('program-causal-math', `${mathChoice.candidate.label}: ${other.toFixed(2)} + (${activation.toFixed(2)} × ${weight.toFixed(2)}) = ${trace.score.toFixed(2)}`);
      element('program-causal-math').title = 'Current score = all other network terms + recruitment-neuron activation × W2[19]. These are preference scores, not success probabilities.';
    } else setText('program-causal-math', 'No legal action is available to score.');
    const changed = this.runtime.current.fingerprint !== this.runtime.baseline.fingerprint;
    const differentChoice = !!choice && !!previous && choice.candidate.id !== previous.candidate.id;
    setText('program-cause-status', !choice ? 'No legal order is available. Resource and population limits still apply.'
      : differentChoice ? 'Your weights changed the top decision on this same world state.'
      : changed ? 'Weights changed; the same decision still leads on this state.'
      : 'Same weights, same decision. Change W2[19] to change its preference.');
    if (this.tab !== 'astra' && !this.watched[this.team]) {
      setText('wsp-field-step', differentChoice ? 'Your edit changed the next order.' : 'Make the network recruit.');
      setText('wsp-field-copy', differentChoice
        ? 'Click Execute & watch. The winning decision becomes an order; follow its cost and progress below.'
        : 'Set the recruitment weight to 24. Watch the two scores, then execute the winning decision.');
    }
    setText('program-choice', choice ? `${choice.candidate.label} · score ${choice.score.toFixed(2)}` : 'No orders available.');
    setText('program-effect', choice && previous
      ? choice.candidate.id !== previous.candidate.id
        ? `${previous.candidate.label} → ${choice.candidate.label}. The top decision changes on this same state.`
        : `${choice.candidate.label} still leads. A different weight does not always change the chosen order.`
      : 'The match has no available decisions.');
    const shown = rows.slice(0, 4);
    if (previous && !shown.includes(previous)) shown[shown.length - 1] = previous;
    setHTML('program-scores', shown.map(row => `<tr${row === choice ? ' class="program-leading"' : ''}><td>${escape(row.candidate.label)}</td><td>${row.baseline.toFixed(2)}</td><td>${row.score.toFixed(2)}</td></tr>`).join(''));
    const issued = this.issued[this.team];
    setText('program-executed', issued
      ? `${clock(issued.time)} · ${issued.source === 'you' ? 'Your manual order' : `Policy r${issued.revision}`} · ${issued.label}${this.lastDecision[this.team] && this.lastDecision[this.team] !== issued.label ? ` · Latest decision: ${this.lastDecision[this.team]}` : ''}`
      : this.lastDecision[this.team] ? `No new order issued. Decision: ${this.lastDecision[this.team]}.` : 'No order has been issued. A preview is not an executed action.');
    const capture = this.captures[this.team];
    setText('program-constraints', `${this.hooks.paused() ? 'Paused' : 'Live'} at ${clock(world.time)} · only legal, affordable orders are scored.${this.team === 1 ? ' Red recruitment starts at 04:30; raids at 05:30, with a limited army.' : ' Costs, population limits, construction and training times still apply.'}${capture ? ` Experiment state: ${clock(capture.time)}, baseline r${this.runtime.baseline.revision}.` : ''}`);
    if (this.tab === 'weights') {
      if (!choice) { setHTML('program-trace', '<p>No candidate is available to trace.</p>'); return; }
      const trace = tracePolicy(choice.input, model);
      const contributions = trace.contributions.map((value, row) => ({ value, row })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 5);
      setHTML('program-trace', `<p>For <strong>${escape(choice.candidate.label)}</strong> on the current state:</p><code>score = Σ W2[h] × ReLU(W1[h] · inputs + B1[h]) + B2</code><ol>${contributions.map(({ row, value }) => `<li><span>${escape(human(model.hiddenNames[row]))}</span><code>${trace.hiddenActivations[row].toFixed(3)} × ${model.w2[row].toFixed(2)} = ${signed(value, 3)}</code></li>`).join('')}</ol><p>All ${model.hiddenNames.length} terms + B2 (${model.b2.toFixed(2)}) = <b>${trace.score.toFixed(3)}</b>. These are preference scores, not success probabilities.</p>`);
    }
  }

  private renderImpact(world: World) {
    const receipt = this.watched[this.team];
    const progress = element<HTMLProgressElement>('program-impact-progress');
    if (!receipt) {
      const units = world.entities.filter(entity => entity.owner === this.team && entity.hp > 0 && entity.category === 'unit');
      const workers = units.filter(entity => entity.kind === 'villager').length;
      setText('program-impact-label', 'No order executed yet');
      setText('program-impact-detail', 'Execute & watch sends the winning decision into this live world.');
      setText('program-impact-counts', `Live army ${units.length - workers} · villagers ${workers}`);
      progress.value = 0; progress.hidden = true;
      return;
    }
    const revision = this.runtime.current.fingerprint === receipt.fingerprint ? `r${receipt.revision}` : `r${receipt.revision} (current: r${this.runtime.current.revision})`;
    element('program-impact-label').title = `At ${clock(receipt.time)}: baseline chose ${receipt.baselineLabel}; executed ${receipt.label} with ${receipt.fingerprint}.`;
    if (!receipt.impact) {
      setText('program-impact-label', 'Keeping existing orders');
      setText('program-impact-detail', `${revision} chose ${receipt.label}. No new order or resource cost.`);
      setText('program-impact-counts', 'The world continues. Adjust weights to prefer another legal action.');
      setText('wsp-field-step', 'The network chose to wait.');
      setText('wsp-field-copy', 'Existing tasks continue. Adjust a weight and execute again to change its next decision.');
      progress.value = 0; progress.hidden = true;
      return;
    }
    const impact = receipt.impact, result = observePlayOrder(world, impact);
    const costs = (['food', 'wood', 'gold', 'stone'] as const)
      .map(resource => ({ resource, amount: impact.beforeResources[resource] - impact.afterResources[resource] }))
      .filter(cost => cost.amount > 0).map(cost => `${Math.round(cost.amount)} ${cost.resource}`).join(' + ');
    setText('program-impact-label', result.label);
    setText('program-impact-detail', `Executed ${revision}${costs ? ` · paid ${costs}` : ''}. ${result.detail}`);
    setText('program-impact-counts', `Live totals · army ${impact.beforeArmy} → ${result.army} · villagers ${impact.beforeVillagers} → ${result.villagers}`);
    element('program-impact-counts').title = 'Counts when this order was issued → counts now. Other orders and combat can also change these totals.';
    progress.hidden = result.progress === null;
    progress.value = result.progress ?? 0;
    setText('program-impact-progress', result.progress === null ? '' : `${Math.round(result.progress * 100)}%`);
    progress.setAttribute('aria-label', `${result.label}${result.progress === null ? '' : `, ${Math.round(result.progress * 100)} percent`}`);
    setText('wsp-field-step', result.status === 'completed' ? 'The network’s order finished.'
      : result.status === 'interrupted' ? 'The order was interrupted.' : 'Watch the order happen.');
    setText('wsp-field-copy', result.status === 'completed'
      ? `${result.detail} Try another weight, or take this network into the full battle.`
      : result.status === 'interrupted' ? result.detail
      : `${receipt.label} is an executed order from ${revision}. ${this.hooks.paused() ? 'Resume game to continue its progress.' : result.detail}`);
  }

  private renderEditor() {
    const snapshot = this.runtime.current;
    const namesSignature = JSON.stringify(snapshot.model.hiddenNames);
    if (namesSignature !== this.neuronNamesSignature) {
      this.neuronNamesSignature = namesSignature;
      const select = element<HTMLSelectElement>('program-neuron'), selected = select.value;
      select.innerHTML = snapshot.model.hiddenNames.map((name, row) => `<option value="${row}">${row} · ${escape(human(name))}</option>`).join('');
      select.value = selected;
    }
    const coordinate = this.coordinate();
    const signature = `${snapshot.fingerprint}:${JSON.stringify(coordinate)}`;
    if (signature !== this.editorSignature) {
      this.editorSignature = signature;
      element<HTMLInputElement>('program-value').value = String(this.valueOf(coordinate));
      element<HTMLSelectElement>('program-neuron').disabled = coordinate.tensor === 'b2';
      element<HTMLSelectElement>('program-feature').disabled = coordinate.tensor !== 'w1';
      setText('program-tensor-detail', `${this.coordinateLabel()} = ${this.valueOf(coordinate).toFixed(3)}.${coordinate.tensor === 'w1' ? ` Input: ${human(snapshot.model.featureNames[coordinate.column!])}; neuron: ${human(snapshot.model.hiddenNames[coordinate.row!])}.` : coordinate.tensor === 'b2' ? ' This shared bias shifts every score equally, so it cannot change the top decision.' : ` Neuron: ${human(snapshot.model.hiddenNames[coordinate.row!])}.`}`);
    }
    if (this.matrixSignature !== signature) {
      this.matrixSignature = signature;
      const model = snapshot.model;
      const values = coordinate.tensor === 'w1' ? model.w1[coordinate.row!] : coordinate.tensor === 'b2' ? [model.b2] : model[coordinate.tensor];
      element('program-matrix').innerHTML = values.map((value, index) => {
        const selected = coordinate.tensor === 'w1' ? index === coordinate.column : coordinate.tensor === 'b2' || index === coordinate.row;
        const label = coordinate.tensor === 'w1' ? model.featureNames[index] : coordinate.tensor === 'b2' ? 'shared output bias' : model.hiddenNames[index];
        return `<button type="button" data-weight-cell="${index}" class="weight-cell ${value < 0 ? 'negative' : value > 0 ? 'positive' : 'zero'}${selected ? ' selected' : ''}" aria-pressed="${selected}" title="${escape(label)}: ${value}" aria-label="${escape(label)}, weight ${value}"><small>${index}</small><b>${value.toFixed(1)}</b></button>`;
      }).join('');
    }
    const historySignature = `${this.team}:${snapshot.id}`;
    if (this.historySignature !== historySignature) {
      this.historySignature = historySignature;
      element('program-history').innerHTML = this.runtime.history.slice(0, 8).map((revision, index) => `<div class="program-history-row"><span><b>r${revision.revision}</b> · ${escape(revision.provenance.source)}<small>${escape(revision.provenance.reason)}</small></span><button type="button" data-restore="${escape(revision.id)}"${index === 0 ? ' disabled' : ''}>${index === 0 ? 'Current' : 'Restore'}</button></div>`).join('');
    }
  }

  private renderExperiment() {
    element<HTMLButtonElement>('program-compare').disabled = !this.hooks.available() || this.experimentBusy;
    element<HTMLButtonElement>('program-compare-cancel').disabled = !this.experimentBusy;
    setText('program-compare-status', this.experimentMessage);
    const receipt = this.experimentReceipt;
    if (!receipt || receipt.team !== this.team) {
      if (this.resultSignature !== 'empty') { element('program-results').replaceChildren(); this.resultSignature = 'empty'; }
      return;
    }
    const stale = receipt.fingerprint !== this.runtime.current.fingerprint;
    const signature = `${receipt.revision}:${receipt.result.startTick}:${stale}`;
    if (signature === this.resultSignature) return;
    this.resultSignature = signature;
    const result = receipt.result, before = result.baseline, after = result.candidate;
    const initial = result.initialMetrics;
    const metrics: [string, number, number][] = [
      ['Food brought home', before.metrics.gathered.food - initial.gathered.food, after.metrics.gathered.food - initial.gathered.food],
      ['Wood brought home', before.metrics.gathered.wood - initial.gathered.wood, after.metrics.gathered.wood - initial.gathered.wood],
      ['Soldiers completed', before.metrics.recruitsCompleted - initial.recruitsCompleted, after.metrics.recruitsCompleted - initial.recruitsCompleted],
      ['Army alive', before.metrics.army, after.metrics.army],
      ['Villagers alive', before.metrics.villagers, after.metrics.villagers],
      [`${teamName(receipt.team)} town health`, before.metrics.townHealth, after.metrics.townHealth],
      [`${teamName((1 - receipt.team) as Team)} town health`, before.otherMetrics.townHealth, after.otherMetrics.townHealth],
    ];
    const changes = metrics.filter(([, a, b]) => Math.abs(a - b) >= .5);
    const headline = changes.length ? changes.slice(0, 2).map(([label, a, b]) => `${label.toLowerCase()}: ${signed(b - a, 0)}`).join(' · ') : 'No measured outcome difference in this 60-second experiment.';
    const divergence = result.firstChangedChoice;
    const choiceLabel = (choice: PlayExperimentChoice | null) => choice
      ? `${choice.label}${choice.executed ? ' (order issued)' : ' (no new order)'}` : 'no available decision';
    const cause = divergence
      ? `First different decision at +${divergence.elapsedSeconds.toFixed(1)}s: baseline ${choiceLabel(divergence.baseline)} → edited ${choiceLabel(divergence.candidate)}.`
      : 'Both policies chose the same orders throughout this experiment.';
    element('program-results').innerHTML = `<div class="program-result-heading"><b>${escape(headline)}</b>${stale ? '<span class="program-stale">Earlier weights · rerun to update</span>' : ''}</div><p>From ${clock(result.startTime)} · baseline r${receipt.baselineRevision} vs edited r${receipt.revision}. ${teamName(receipt.team)} uses its policy in both futures; ${teamName((1 - receipt.team) as Team)} ${result.otherControl === 'policy' ? `uses the same r${receipt.otherRevision} policy` : 'keeps its existing orders'}.</p><p><strong>${escape(cause)}</strong></p><table aria-label="Measured outcomes of the two simulations"><thead><tr><th>After up to 60s</th><th>Baseline</th><th>Edited</th><th>Change</th></tr></thead><tbody>${metrics.map(([label, a, b]) => `<tr><td>${escape(label)}</td><td>${Math.round(a)}</td><td>${Math.round(b)}</td><td>${signed(b - a, 0)}</td></tr>`).join('')}</tbody></table><p>Actual orders issued: ${before.executedOrders} → ${after.executedOrders}. Simulated time: ${before.elapsedSeconds.toFixed(1)}s / ${after.elapsedSeconds.toFixed(1)}s. A battle ending stops that branch. The live battlefield is unchanged.</p>`;
  }

  private renderLaya() {
    const status = this.layaStatus;
    setText('program-laya-status', status?.label ?? 'Checking local Laya…');
    setText('program-laya-detail', status?.available
      ? `${status.model} · ${status.device ?? 'native runtime'} · checkpoint ${status.checkpoint?.slice(0, 12)}. Laya chooses among four developer-written +1 weight edits or “keep”. It cannot generate new numerical weights.`
      : status?.detail ?? 'Checking /api/status. Manual weight programming works independently.');
    element('program-laya-detail').title = status?.checkpoint ? `Native checkpoint SHA256: ${status.checkpoint}` : '';
    element<HTMLButtonElement>('program-laya-request').disabled = !this.hooks.available() || !status?.available || this.coach.busy;
    element<HTMLButtonElement>('program-laya-refresh').disabled = this.coach.busy;
    const decision = this.proposal?.decision;
    const eligibility = this.proposalEligibility();
    element<HTMLButtonElement>('program-laya-apply').disabled = !this.hooks.available() || !decision?.proposal.edits.length || !eligibility.allowed;
    element<HTMLButtonElement>('program-laya-reject').disabled = !decision && !this.coach.busy;
    setText('program-laya-reject', this.coach.busy ? 'Cancel request' : 'Discard');
    const edits = decision?.proposal.edits;
    setHTML('program-laya-proposal', `<p role="status">${escape(this.layaMessage)}</p>${decision ? `<b>${escape(decision.proposal.description)}</b>${edits?.length ? `<ul>${edits.map(edit => `<li><code>W2[${edit.row}]</code> ${escape(human(this.runtime.current.model.hiddenNames[edit.row]))}: <b>${edit.before.toFixed(2)} → ${edit.value.toFixed(2)}</b></li>`).join('')}</ul>` : '<p>Laya selected “keep”. No numerical edits are proposed.</p>'}<p>Requested at tick ${decision.requestedTick} against revision ${decision.response.base_revision}.${!eligibility.allowed ? ` ${escape(eligibility.reason)}` : ''}</p>` : ''}`);
  }

  dispose() {
    this.astra.dispose();
    this.disposed = true; this.statusGeneration++; this.cancelExperiment(false); this.coach.cancel();
  }
}
