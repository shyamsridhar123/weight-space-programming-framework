import { comparePolicies, policyFeatures, tracePolicy, type PolicyModel } from './policy';
import type { World } from './simulation';
import type { Team } from './types';
import { WeightRuntime, STRATEGIC_PRESETS, cloneWeightModel, fingerprintWeightModel, presetEdits,
  type WeightContext, type WeightEdit, type WeightSnapshot } from './weight-runtime';
import { assessWeightPromotion, validateWeightExperimentConfig, weightExperimentBudget, type WeightExperimentConfig,
  type WeightExperimentResult } from './weight-experiments';
import { LayaWeightCoach } from './weight-coach';
import type { WeightLabCallbacks, WeightLabSearchState, WeightLabSnapshot } from './weight-lab';
import type { WeightWorkerReply, WeightWorkerRequest } from './weight-worker';

export const WEIGHT_SESSION_KEY = 'crown-charter-weight-session-v1';
const EVIDENCE_LIMIT = 120;
type WorkerHandle = Pick<Worker, 'onmessage' | 'onerror' | 'onmessageerror' | 'postMessage' | 'terminate'>;
type StorageHandle = Pick<Storage, 'getItem' | 'setItem'>;

export interface WeightWorldContext {
  world: World;
  team: Team;
  matchId: string;
  started: boolean;
  paused: boolean;
  pilot: string;
}

export interface WeightWorkbenchOptions {
  context(): WeightWorldContext;
  update?(snapshot: WeightLabSnapshot): void;
  weightsChanged?(snapshot: WeightSnapshot): void;
  coachBusyChanged?(busy: boolean): void;
  storage?: StorageHandle;
  storageError?: string;
  fetcher?: typeof fetch;
  workerFactory?: () => WorkerHandle;
}

type WeightEvidence = { kind: string; tick: number; matchId: string; at: string; detail: unknown };

/** Compact identity for command logs; full tensors live in the weight evidence. */
export function weightIdentity(snapshot: WeightSnapshot) {
  return { id: snapshot.id, revision: snapshot.revision, fingerprint: snapshot.fingerprint,
    provenance: snapshot.provenance, createdAt: snapshot.createdAt };
}

export function countWeightDifferences(a: PolicyModel, b: PolicyModel): number {
  let count = Number(a.b2 !== b.b2) + Number(a.temperature !== b.temperature);
  for (let row = 0; row < a.w1.length; row++) {
    count += Number(a.b1[row] !== b.b1[row]) + Number(a.w2[row] !== b.w2[row]);
    for (let column = 0; column < a.w1[row].length; column++) count += Number(a.w1[row][column] !== b.w1[row][column]);
  }
  return count;
}

/** The UI, live match, and worker exchange snapshots; none shares a mutable model. */
export class WeightWorkbench {
  readonly runtime: WeightRuntime;
  readonly opponent = new WeightRuntime().current;
  readonly callbacks: WeightLabCallbacks;
  private readonly fetcher: typeof fetch;
  private readonly coach: LayaWeightCoach;
  private draft?: PolicyModel;
  private draftBase?: WeightSnapshot;
  private liveApply = true;
  private selectedCandidate = '';
  private status = 'Live edits affect blue’s next neural decision. Red keeps the original commander.';
  private error = '';
  private persistenceError = '';
  private coachAvailable = false;
  private coachPending = false;
  private coachGeneration = 0;
  private coachMessage = 'Checking the local Laya connection…';
  private coachSuggestion = '';
  private availabilityPending = false;
  private lastAvailabilityCheck = -Infinity;
  private availabilityAbort: AbortController | null = null;
  private worker: WorkerHandle | null = null;
  private runId = '';
  private runSerial = 0;
  private runBase?: WeightSnapshot;
  private runConfig?: WeightExperimentConfig;
  private search: WeightLabSearchState = { state: 'idle', config: validateWeightExperimentConfig(), autoPromote: false, canPromote: false };
  private starting: WeightSnapshot;
  private evidence: WeightEvidence[] = [];
  private droppedEvidence = 0;
  private disposed = false;

  constructor(private readonly options: WeightWorkbenchOptions) {
    let runtime = new WeightRuntime();
    this.persistenceError = options.storageError ?? '';
    if (options.storage) {
      try {
        const saved = options.storage.getItem(WEIGHT_SESSION_KEY);
        if (saved !== null) {
          runtime = WeightRuntime.fromSession(saved);
          this.status = `Restored revision ${runtime.current.revision}, its history, and the pinned comparison baseline.`;
        }
      } catch (error) {
        this.persistenceError = `Saved weights could not be restored: ${this.message(error)}. The original commander is active; the saved data has not been overwritten.`;
      }
    }
    this.runtime = runtime;
    this.starting = runtime.current;
    this.fetcher = options.fetcher ?? globalThis.fetch.bind(globalThis);
    this.coach = new LayaWeightCoach(this.fetcher);
    this.callbacks = {
      setLiveApply: enabled => this.setLiveApply(enabled),
      editTensor: edit => this.editTensor(edit),
      applyStaged: () => this.applyStaged(),
      discardStaged: () => this.discardStaged(),
      applyPreset: id => this.applyPreset(id),
      pinBaseline: () => this.pinBaseline(),
      undo: () => this.undo(),
      rollback: id => this.rollback(id),
      importCheckpoint: json => this.importCheckpoint(json),
      exportCheckpoint: () => this.runtime.exportCheckpoint(),
      selectCandidate: id => { this.selectedCandidate = id; this.refresh(); },
      startSearch: (config, autoPromote) => this.startSearch(config, autoPromote),
      stopSearch: () => this.stopSearch(),
      promoteSearchCandidate: id => this.promoteSearchCandidate(id),
      requestLayaCoach: () => this.requestLayaCoach(),
    };
  }

  get current(): WeightSnapshot { return this.runtime.current; }
  get coachBusy(): boolean { return this.coachPending; }
  get stagedEdits(): number { return this.draft ? countWeightDifferences(this.draft, this.current.model) : 0; }

  snapshot(): WeightLabSnapshot {
    const context = this.options.context();
    const rankings = comparePolicies(context.world, context.team, this.current.model, this.runtime.baseline.model);
    const candidate = rankings.active.find(entry => entry.id === this.selectedCandidate) ?? rankings.active[0];
    this.selectedCandidate = candidate?.id ?? '';
    const input = candidate ? policyFeatures(context.world, context.team, candidate) : null;
    return {
      active: this.current, baseline: this.runtime.baseline, history: this.runtime.history,
      draft: this.draft, liveApply: this.liveApply, stagedEdits: this.stagedEdits, rankings,
      inspection: candidate && input ? { candidateId: candidate.id,
        active: tracePolicy(input, this.current.model), baseline: tracePolicy(input, this.runtime.baseline.model) } : undefined,
      presets: STRATEGIC_PRESETS, canUndo: this.runtime.history.length > 1 && !this.draft,
      search: { ...this.search, canPromote: this.canPromote() },
      coach: { available: this.coachAvailable && context.started && context.world.status === 'playing',
        busy: this.coachPending, message: !context.started ? 'Start a match to ask Laya about its current kingdom.' : this.coachMessage,
        suggestion: this.coachSuggestion },
      status: this.status, error: this.error || this.persistenceError, currentTick: context.world.tick,
      context: { tick: context.world.tick, seconds: context.world.time, pilot: context.pilot,
        paused: context.paused || !context.started, matchId: context.matchId },
    };
  }

  refresh(): void { if (!this.disposed) this.options.update?.(this.snapshot()); }

  setLiveApply(enabled: boolean): void {
    if (this.draft) throw new Error('Apply or discard the staged changes before switching editing modes.');
    this.liveApply = enabled;
    this.error = '';
    this.status = enabled ? 'Live mode: numerical edits become active immediately.' : 'Staged mode: the running commander keeps its applied weights until you select Apply.';
    this.refresh();
  }

  editTensor(edit: WeightEdit): void {
    const address = `${edit.tensor}${edit.row === undefined ? '' : `[${edit.row}${edit.column === undefined ? '' : `,${edit.column}`}]`}`;
    this.edit([edit], `Set ${address} to ${edit.value}.`, 'tensor-editor');
  }

  applyPreset(id: string): void {
    const preset = STRATEGIC_PRESETS.find(entry => entry.id === id);
    if (!preset) throw new Error('Unknown strategic weight preset.');
    this.edit(presetEdits(this.draft ?? this.current.model, id), `Applied ${preset.name} W2 profile.`, 'strategic-preset');
  }

  private edit(edits: WeightEdit[], reason: string, source: string): void {
    // Validate the entire patch before either editing path can change live state.
    const scratch = new WeightRuntime(this.draft ?? this.current.model);
    const validated = scratch.patch(edits, { source, reason }).model;
    if (!countWeightDifferences(validated, this.draft ?? this.current.model)) return;
    if (this.liveApply) {
      const next = this.runtime.apply(validated, this.context(source, reason));
      this.changed(next);
    } else {
      this.draftBase ??= this.current;
      this.draft = cloneWeightModel(validated);
      if (countWeightDifferences(this.draft, this.current.model) === 0) { this.draft = undefined; this.draftBase = undefined; }
      this.search.autoPromote = false;
      this.cancelCoach('Laya’s pending suggestion was cancelled because changes are being staged.');
      this.status = `${this.stagedEdits} numerical changes staged; the applied commander is still r${this.current.revision}.`;
      this.error = ''; this.refresh();
    }
  }

  applyStaged(): void {
    if (!this.draft || !this.draftBase) throw new Error('There are no staged changes to apply.');
    const count = this.stagedEdits;
    const next = this.runtime.apply(this.draft, this.context('staged-editor', `Applied ${count} staged numerical changes atomically.`, this.draftBase));
    this.draft = undefined; this.draftBase = undefined;
    this.changed(next);
  }

  discardStaged(): void {
    this.draft = undefined; this.draftBase = undefined;
    this.status = 'Staged changes discarded. The applied commander is unchanged.';
    this.error = ''; this.refresh();
  }

  pinBaseline(): void {
    this.requireNoDraft();
    const baseline = this.runtime.pinBaseline();
    this.addEvidence('baseline-pinned', weightIdentity(baseline));
    this.persist(); this.error = '';
    this.status = `Comparison baseline pinned to r${baseline.revision}. Red’s opponent weights are still the original checkpoint.`;
    this.refresh();
  }

  undo(): void {
    this.requireNoDraft();
    const previous = this.runtime.history[1];
    if (!previous) throw new Error('There is no earlier retained revision to restore.');
    this.rollback(previous.id);
  }

  rollback(id: string): void {
    this.requireNoDraft();
    this.changed(this.runtime.restore(id, this.context('rollback', `Restored retained snapshot ${id}.`)));
  }

  importCheckpoint(json: string): void {
    this.requireNoDraft();
    this.changed(this.runtime.importCheckpoint(json, this.context('checkpoint-import', 'Imported a validated commander checkpoint.')));
  }

  private changed(snapshot: WeightSnapshot): void {
    this.cancelCoach('The pending Laya suggestion was cancelled by a newer weight revision.');
    this.search.autoPromote = false;
    this.addEvidence('weights-applied', snapshot);
    this.persist(); this.error = '';
    this.status = `r${snapshot.revision} applied at tick ${this.options.context().world.tick}. Blue will use it on its next neural decision.`;
    this.options.weightsChanged?.(snapshot);
    this.refresh();
  }

  private persist(): void {
    if (!this.options.storage) return;
    try { this.options.storage.setItem(WEIGHT_SESSION_KEY, this.runtime.exportSession()); this.persistenceError = ''; }
    catch (error) { this.persistenceError = `Weights are active, but this browser could not save the session: ${this.message(error)}. Export a checkpoint to keep it.`; }
  }

  private context(source: string, reason: string, base = this.current): WeightContext {
    return { source, reason, tick: this.options.context().world.tick, expectedRevision: base.revision, expectedFingerprint: base.fingerprint };
  }

  private requireNoDraft(): void {
    if (this.draft) throw new Error('Apply or discard the staged changes before this operation.');
  }

  startSearch(config: WeightExperimentConfig, autoPromote: boolean): void {
    this.requireNoDraft();
    if (this.worker) throw new Error('An experiment is already running. Stop it before starting another.');
    const validated = validateWeightExperimentConfig(config);
    const factory = this.options.workerFactory ?? (() => new Worker(new URL('./weight-worker.ts', import.meta.url), { type: 'module', name: 'commander-weight-experiment' }));
    const worker = factory(), base = this.current, runId = `weights-${Date.now()}-${++this.runSerial}`;
    this.worker = worker; this.runId = runId; this.runBase = base;
    const submitted = validateWeightExperimentConfig(validated);
    this.runConfig = submitted;
    this.search = { state: 'running', config: validated, autoPromote, canPromote: false };
    this.error = ''; this.status = 'Experiment running in independent seeded worlds. The live match keeps its current weights.';
    this.addEvidence('experiment-started', { runId, base, config: validated, autoPromote });
    worker.onmessage = (event: MessageEvent<WeightWorkerReply>) => {
      if (this.disposed || this.worker !== worker || event.data.runId !== runId) return;
      const reply = event.data;
      if (reply.type === 'progress') { this.search.progress = reply.progress; return; }
      worker.terminate(); this.worker = null;
      const armed = this.search.autoPromote; this.search.autoPromote = false;
      if (reply.type === 'cancelled') { this.search.state = 'cancelled'; this.status = 'Experiment cancelled.'; }
      else if (reply.type === 'error') this.searchFailed(reply.message);
      else if (reply.type === 'result') {
        try {
          this.validateResult(reply.result, base, submitted);
          this.search = { ...this.search, state: 'complete', result: reply.result };
          this.addEvidence('experiment-completed', { runId, result: reply.result });
          this.status = this.canPromote() ? 'The finalist passed the configured smoke test. Review its evidence before promotion.'
            : this.current.revision !== base.revision || this.current.fingerprint !== base.fingerprint
              ? 'Experiment complete. Its source revision is no longer active, so this result cannot be promoted.'
              : 'Experiment complete. The finalist did not pass the promotion gate; current weights are retained.';
          if (armed && this.canPromote()) this.promoteSearchCandidate(reply.result.candidate.id);
        } catch (error) { this.searchFailed(this.message(error)); }
      }
      this.refresh();
    };
    worker.onerror = event => {
      event.preventDefault();
      if (this.worker !== worker) return;
      worker.terminate(); this.worker = null; this.searchFailed(event.message || 'The experiment worker stopped unexpectedly.'); this.refresh();
    };
    worker.onmessageerror = () => {
      if (this.worker !== worker) return;
      worker.terminate(); this.worker = null; this.searchFailed('The experiment worker returned an unreadable message.'); this.refresh();
    };
    try {
      const message: WeightWorkerRequest = { type: 'start', runId, baseRevision: base.revision, baseModel: cloneWeightModel(base.model), config: validated };
      worker.postMessage(message);
    } catch (error) { worker.terminate(); this.worker = null; this.searchFailed(this.message(error)); }
    this.refresh();
  }

  stopSearch(): void {
    if (!this.worker) return;
    this.worker.terminate(); this.worker = null;
    this.search = { ...this.search, state: 'cancelled', autoPromote: false, canPromote: false };
    this.addEvidence('experiment-cancelled', { runId: this.runId });
    this.status = 'Experiment stopped. No candidate weights were applied.';
    this.refresh();
  }

  private searchFailed(message: string): void {
    this.search = { ...this.search, state: 'error', error: message, autoPromote: false, canPromote: false };
    this.addEvidence('experiment-error', { runId: this.runId, message });
    this.status = 'Experiment failed. The running commander retains its weights.';
  }

  private validateResult(result: WeightExperimentResult, base: WeightSnapshot, submitted: WeightExperimentConfig): void {
    if (result.schemaVersion !== 1 || result.baseRevision !== base.revision || result.baseFingerprint !== base.fingerprint)
      throw new Error('Experiment returned a result for a different source revision.');
    if (JSON.stringify(validateWeightExperimentConfig(result.config)) !== JSON.stringify(submitted))
      throw new Error('Experiment returned a configuration different from the submitted settings.');
    const budget = weightExperimentBudget(submitted);
    if (result.budget.matches !== budget.matches || result.budget.simulatedSeconds !== budget.simulatedSeconds
      || result.budget.policyDecisions !== budget.policyDecisions)
      throw new Error('Experiment returned a budget different from the submitted settings.');
    if (fingerprintWeightModel(result.candidate.model) !== result.candidate.fingerprint)
      throw new Error('Experiment candidate fingerprint does not match its actual weights.');
    const development = result.development.find(entry => entry.candidateId === result.candidate.id);
    if (!development || result.validation.candidateId !== result.candidate.id)
      throw new Error('Experiment has no matching development and validation evidence for its finalist.');
    if (result.development.length !== submitted.candidates
      || result.development.some(entry => entry.phase !== 'development' || JSON.stringify(entry.seeds) !== JSON.stringify(submitted.developmentSeeds))
      || result.validation.phase !== 'validation' || JSON.stringify(result.validation.seeds) !== JSON.stringify(submitted.validationSeeds))
      throw new Error('Experiment evidence does not match the submitted development and validation splits.');
    const promotion = assessWeightPromotion(development, result.validation, submitted.minFitnessGain);
    if (JSON.stringify(promotion) !== JSON.stringify(result.promotion)) throw new Error('Experiment promotion claim does not match its evaluation evidence.');
    const scratch = new WeightRuntime(base.model);
    const reconstructed = scratch.patch(result.candidate.edits.map(edit => {
      if (edit.tensor !== 'w2' || base.model.w2[edit.index] !== edit.before || Math.abs(edit.after - edit.before - edit.delta) > 1e-9)
        throw new Error('Experiment patch does not match its source weights.');
      return { tensor: 'w2', row: edit.index, value: edit.after };
    }), { source: 'experiment-validation', reason: 'Reconstruct candidate from its numeric patch.' });
    if (reconstructed.fingerprint !== result.candidate.fingerprint) throw new Error('Experiment changed weights outside its recorded W2 patch.');
  }

  private canPromote(): boolean {
    const result = this.search.result;
    return this.search.state === 'complete' && !!result?.promotion.recommended && !this.draft
      && result.baseRevision === this.current.revision && result.baseFingerprint === this.current.fingerprint
      && result.candidate.fingerprint !== this.current.fingerprint;
  }

  promoteSearchCandidate(id: string): void {
    this.requireNoDraft();
    const result = this.search.result;
    if (!result || result.candidate.id !== id || !this.canPromote() || !this.runBase || !this.runConfig)
      throw new Error('This candidate cannot be promoted: validation must pass and its source revision must still be active.');
    this.validateResult(result, this.runBase, this.runConfig);
    const next = this.runtime.apply(result.candidate.model, this.context('validated-search',
      `Promoted ${id}; development fitness ${result.promotion.developmentGain.toFixed(4)}, validation ${result.promotion.validationGain.toFixed(4)}.`, this.runBase));
    this.addEvidence('experiment-promoted', { runId: this.runId, candidateId: id, source: weightIdentity(this.runBase), applied: weightIdentity(next) });
    this.changed(next);
  }

  async checkLayaAvailability(force = false): Promise<void> {
    if (this.disposed || this.availabilityPending || !force && Date.now() - this.lastAvailabilityCheck < 15000) return;
    this.lastAvailabilityCheck = Date.now(); this.availabilityPending = true;
    const controller = new AbortController(); this.availabilityAbort = controller;
    const timer = setTimeout(() => controller.abort(), 4000);
    try {
      const response = await this.fetcher('/api/status', { signal: controller.signal });
      if (!response.ok) throw new Error('Local Laya service is unavailable.');
      const status = await response.json() as { ready?: boolean; available_modes?: string[] };
      if (!status.ready || !Array.isArray(status.available_modes) || !status.available_modes.includes('laya'))
        throw new Error('The local service has no ready native Laya checkpoint.');
      const schemaResponse = await this.fetcher('/openapi.json', { signal: controller.signal });
      if (!schemaResponse.ok) throw new Error('The local service does not expose the weight coach route.');
      const schema = await schemaResponse.json() as { paths?: Record<string, { post?: unknown }> };
      if (!schema.paths?.['/api/empires/weight-patch']?.post) throw new Error('Restart the local Laya service to load its weight coach route.');
      this.coachAvailable = true;
      if (!this.coachPending && !this.coachSuggestion) this.coachMessage = 'Native Laya can select an explicit W2 patch for this kingdom. Its RTS ability is unvalidated.';
    } catch (error) {
      this.coachAvailable = false;
      if (!this.coachPending) this.coachMessage = `${this.message(error)} Start scripts/start-empires.ps1 -WithLaya to connect.`;
    } finally {
      clearTimeout(timer); this.availabilityPending = false;
      if (this.availabilityAbort === controller) this.availabilityAbort = null;
      this.refresh();
    }
  }

  async requestLayaCoach(): Promise<void> {
    this.requireNoDraft();
    if (this.coachPending) return;
    const context = this.options.context(), base = this.current;
    if (!context.started || context.world.status !== 'playing') throw new Error('Start a playing match before asking Laya for a weight patch.');
    if (!this.coachAvailable) throw new Error(this.coachMessage);
    const generation = ++this.coachGeneration;
    this.coachPending = true; this.coachSuggestion = ''; this.error = '';
    this.coachMessage = 'Laya is choosing among bounded, explicit numerical patches. The world keeps running.';
    this.options.coachBusyChanged?.(true); this.refresh();
    try {
      const result = await this.coach.request(context.world, context.team, context.matchId, base);
      const currentContext = this.options.context();
      if (this.disposed || generation !== this.coachGeneration) return;
      if (!result || currentContext.world !== context.world || currentContext.matchId !== context.matchId
        || this.current.revision !== base.revision || this.current.fingerprint !== base.fingerprint) {
        this.coachMessage = 'The suggestion expired before it could be applied. Ask again with the current match and revision.';
        this.addEvidence('coach-expired', { source: weightIdentity(base) }); return;
      }
      this.addEvidence('laya-weight-decision', { source: weightIdentity(base), decision: result });
      const patch = result.proposal.edits;
      const summary = patch.length ? patch.map(edit => `W2[${edit.row}] ${edit.before} → ${edit.value}`).join('; ') : 'Keep the current weights';
      if (patch.length) {
        const next = this.runtime.patch(patch.map(edit => ({ tensor: 'w2', row: edit.row, value: edit.value })),
          this.context('laya-coach', `Native Laya selected ${result.proposal.id}: ${summary}.`, base));
        this.changed(next);
      }
      this.coachSuggestion = `${summary}. ${Math.round(result.response.inference_ms)} ms inference; requested at tick ${result.requestedTick}.`;
      this.coachMessage = patch.length ? `Laya’s numerical patch is active at r${this.current.revision}.` : 'Laya selected keep. No revision was created.';
      this.status = this.coachMessage;
    } catch (error) {
      if (generation !== this.coachGeneration) return;
      this.coachMessage = this.message(error); this.error = this.coachMessage;
      this.addEvidence('coach-error', { source: weightIdentity(base), message: this.coachMessage });
    } finally {
      if (generation === this.coachGeneration) { this.coachPending = false; this.options.coachBusyChanged?.(false); }
      this.refresh();
    }
  }

  cancelCoach(message = 'Laya’s pending weight request was cancelled.'): void {
    if (!this.coachPending) return;
    this.coachGeneration++; this.coach.cancel(); this.coachPending = false;
    this.coachMessage = message;
    this.options.coachBusyChanged?.(false);
  }

  beginMatch(): void {
    this.cancelCoach('A new match cancelled the previous Laya weight request.');
    this.starting = this.current; this.evidence = []; this.droppedEvidence = 0;
    this.search.autoPromote = false;
    this.addEvidence('match-weight-start', { active: weightIdentity(this.current), opponent: weightIdentity(this.opponent) });
    this.refresh();
  }

  exportEvidence() {
    return {
      schema: 'crown-charter-weight-evidence-v1', starting: this.starting, current: this.current,
      comparisonBaseline: this.runtime.baseline, frozenOpponent: this.opponent, retainedHistory: this.runtime.history,
      events: structuredClone(this.evidence), droppedEvents: this.droppedEvidence, eventLimit: EVIDENCE_LIMIT,
      historyNote: 'Runtime history retains the most recent 40 revisions. Match weight events retain at most 120 entries; droppedEvents reports truncation. Checkpoint fingerprints identify content and are noncryptographic.',
      experimentNote: 'Experiments use separate worlds and the submitted source model as their frozen opponent. The live red commander always uses the original model. Shaped fitness is not a win rate.',
      coachNote: 'Native Laya selects among explicit authored W2 patches, including keep. Only the auxiliary 1,293-parameter commander changes; the Laya checkpoint remains fixed.',
    };
  }

  private addEvidence(kind: string, detail: unknown): void {
    const context = this.options.context();
    this.evidence.push({ kind, tick: context.world.tick, matchId: context.matchId, at: new Date().toISOString(), detail });
    if (this.evidence.length > EVIDENCE_LIMIT) { this.evidence.shift(); this.droppedEvidence++; }
  }

  private message(error: unknown): string { return error instanceof Error ? error.message : String(error); }

  dispose(): void {
    this.disposed = true; this.cancelCoach(); this.worker?.terminate(); this.worker = null;
    this.availabilityAbort?.abort();
  }
}
