import {
  PlayCoachMonitor, applyPlayCoachPatch, buildPlayCoachContext, parsePlayCoachPatch,
  summarizePlayCoachComparison, type PlayCoachGoal, type PlayCoachPerformance, type PlayCoachProposal,
} from './play-coach';
import { pickPlayPolicy, shouldPlayPolicyStep } from './play-policy';
import type { PlayExperimentRequest, PlayExperimentResult, PlayExperimentWorkerReply } from './play-experiment';
import type { World } from './simulation';
import type { Team } from './types';
import type { WeightRuntime, WeightSnapshot } from './weight-runtime';

export interface AstraConnection {
  configured: boolean;
  verified: boolean;
  in_flight: boolean;
  requested_model: string;
  last_error: string | null;
}

export interface AstraCycle {
  proposal: PlayCoachProposal;
  goal: PlayCoachGoal;
  before: PlayCoachPerformance;
  appliedRevision: number | null;
  appliedFingerprint: string | null;
  appliedTime: number | null;
  beforeChoice: string;
  afterChoice: string;
  firstOrder: { label: string; time: number; revision: number } | null;
  after: PlayCoachPerformance | null;
  measurementComplete: boolean;
  measurementCanceled: boolean;
  comparison: PlayExperimentResult | null;
  comparisonMessage: string;
}

export interface PlayAstraHooks {
  world(): World;
  matchId(): string;
  runtime(team: Team): WeightRuntime;
  available(): boolean;
  paused(): boolean;
  policyEnabled(team: Team): boolean;
  startPolicy(): void;
  onApplied(before: WeightSnapshot, after: WeightSnapshot): void;
}

type Comparison = (request: PlayExperimentRequest, signal: AbortSignal) => Promise<PlayExperimentResult>;
export interface PlayAstraDependencies {
  fetch?: typeof fetch;
  now?: () => number;
  compare?: Comparison;
}

const REVIEW_SECONDS = 30;
const MIN_REQUEST_INTERVAL_MS = 20_000;
const MAX_CONTEXT_AGE_SECONDS = 120;

/** Isolated copies use exactly the same world engine and rival policy as the live game. */
const compareInWorker: Comparison = (request, signal) => new Promise((resolve, reject) => {
  if (signal.aborted) { reject(new Error('Comparison canceled.')); return; }
  const worker = new Worker(new URL('./play-experiment-worker.ts', import.meta.url), { type: 'module' });
  const finish = () => { worker.terminate(); signal.removeEventListener('abort', abort); clearTimeout(timeout); };
  const abort = () => { finish(); reject(new Error('Comparison canceled.')); };
  const timeout = setTimeout(() => { finish(); reject(new Error('Comparison exceeded 30 seconds.')); }, 30_000);
  signal.addEventListener('abort', abort, { once: true });
  worker.onmessage = ({ data }: MessageEvent<PlayExperimentWorkerReply>) => {
    if (data.type === 'result') { finish(); resolve(data.result); }
    else if (data.type === 'error') { finish(); reject(new Error(data.message)); }
  };
  worker.onerror = () => { finish(); reject(new Error('The isolated comparison could not run.')); };
  worker.postMessage({ type: 'run', id: 1, request });
});

/** The supervisor only writes validated model output; ordinary game orders remain the policy's job. */
export class PlayAstraLoop {
  active = false;
  goal: PlayCoachGoal = 'grow-army';
  phase: 'idle' | 'observe' | 'request' | 'apply' | 'measure' | 'error' = 'idle';
  message = 'Start the AI. After 30 game seconds of measured play, Astra reviews a shortfall and writes a correction.';
  connection: AstraConnection | null = null;
  connectionError = '';
  checkingConnection = false;
  report: PlayCoachPerformance | null = null;
  latest: AstraCycle | null = null;
  lastApplied: AstraCycle | null = null;
  events: { time: number; label: string; detail: string }[] = [];
  requestStartedAt: number | null = null;
  private readonly monitor = new PlayCoachMonitor({ windowSeconds: REVIEW_SECONDS, minimumSeconds: REVIEW_SECONDS });
  private readonly fetcher: typeof fetch;
  private readonly now: () => number;
  private readonly compare: Comparison;
  private generation = 0;
  private statusGeneration = 0;
  private sessionMatch = '';
  private requestAbort: AbortController | null = null;
  private comparisonAbort: AbortController | null = null;
  private pending: { proposal: PlayCoachProposal; before: PlayCoachPerformance; requestedTime: number } | null = null;
  private lastRequestAt = -Infinity;

  constructor(private readonly hooks: PlayAstraHooks, dependencies: PlayAstraDependencies = {}) {
    this.fetcher = dependencies.fetch ?? ((...args) => fetch(...args));
    this.now = dependencies.now ?? (() => performance.now());
    this.compare = dependencies.compare ?? compareInWorker;
  }

  async refreshConnection(): Promise<void> {
    const generation = ++this.statusGeneration;
    this.checkingConnection = true;
    try {
      const response = await this.fetcher('/api/empires/astra/status', { signal: AbortSignal.timeout(7000) });
      if (!response.ok) throw new Error(`Astra service returned HTTP ${response.status}.`);
      const status = await response.json() as AstraConnection;
      if (typeof status.configured !== 'boolean' || typeof status.verified !== 'boolean' || typeof status.requested_model !== 'string')
        throw new Error('The service did not return an Astra connection status.');
      if (generation !== this.statusGeneration) return;
      this.connection = status;
      this.connectionError = '';
    } catch (error) {
      if (generation !== this.statusGeneration) return;
      this.connection = null;
      this.connectionError = error instanceof Error ? error.message : 'Astra service could not be reached.';
    } finally { if (generation === this.statusGeneration) this.checkingConnection = false; }
  }

  start(): void {
    if (!this.hooks.available() || this.hooks.world().status !== 'playing') return;
    this.cancelPending();
    this.active = true;
    this.phase = 'observe';
    this.sessionMatch = this.hooks.matchId();
    this.hooks.startPolicy();
    this.monitor.reset(this.hooks.world(), 0, this.hooks.runtime(0).current);
    this.report = this.monitor.observe(this.hooks.world(), 0, this.hooks.runtime(0).current, this.goal);
    this.message = 'Blue’s neural policy is playing. Measuring 30 game seconds before asking Astra for a correction.';
    this.event('AI playing', 'Orders come from the game policy. Astra reviews measured shortfalls.');
  }

  stop(message = 'Astra stopped. The game policy keeps playing with its current weights.'): void {
    const wasActive = this.active;
    this.cancelPending();
    this.active = false;
    this.phase = 'idle';
    this.message = message;
    if (wasActive) this.event('Astra stopped', message);
  }

  reset(): void {
    this.stop();
    this.monitor.reset(); this.report = null; this.latest = null; this.lastApplied = null; this.events = [];
    this.message = 'Start AI + Astra to observe play and automatically correct a measured shortfall.';
    this.lastRequestAt = -Infinity;
  }

  setGoal(goal: PlayCoachGoal): void {
    if (goal === this.goal) return;
    const restart = this.active;
    this.stop('Goal changed. Start AI + Astra to measure this goal.');
    this.goal = goal; this.report = null;
    if (restart) this.start();
  }

  private event(label: string, detail: string): void {
    this.events.unshift({ time: this.hooks.world().time, label, detail });
    if (this.events.length > 12) this.events.length = 12;
  }

  private cancelPending(): void {
    this.generation++;
    this.requestAbort?.abort(); this.requestAbort = null;
    this.comparisonAbort?.abort(); this.comparisonAbort = null;
    this.pending = null; this.requestStartedAt = null;
    if (this.latest && this.latest.appliedRevision !== null && !this.latest.measurementComplete)
      this.latest.measurementCanceled = true;
    if (this.lastApplied && !this.lastApplied.comparison)
      this.lastApplied.comparisonMessage = 'Comparison canceled; the live weight revision is unchanged.';
  }

  private isMeasuring(cycle: AstraCycle | null, report: PlayCoachPerformance): cycle is AstraCycle {
    return !!cycle && cycle.appliedRevision !== null && !cycle.measurementCanceled && !cycle.measurementComplete
      && cycle.goal === this.goal && cycle.appliedRevision === report.revision
      && cycle.appliedFingerprint === report.fingerprint && cycle.appliedTime !== null
      && Math.abs(cycle.appliedTime - report.startTime) < 1e-6;
  }

  private recordMeasurement(report: PlayCoachPerformance): void {
    const cycle = this.latest;
    if (!this.isMeasuring(cycle, report)) return;
    cycle.after = report;
    if (report.ready) {
      cycle.measurementComplete = true;
      this.phase = 'observe';
      this.message = `Measured ${report.elapsedSeconds.toFixed(0)}s after Astra’s edit: ${report.recruitsCompleted} soldiers completed, ${Math.round(report.returned.food)} food and ${Math.round(report.returned.wood)} wood delivered.`;
      this.event('Live result measured', this.message);
    }
  }

  /** Called on a real fixed game tick, before the next policy decision. */
  tick(): void {
    if (!this.active) return;
    const world = this.hooks.world();
    if (this.sessionMatch !== this.hooks.matchId() || !this.hooks.policyEnabled(0)) {
      this.stop('The match or command control changed. Astra stopped.'); return;
    }
    if (world.status !== 'playing') {
      this.report = this.monitor.observe(world, 0, this.hooks.runtime(0).current, this.goal);
      this.recordMeasurement(this.report);
      this.stop('The match ended. Astra stopped; the final observed results are retained.'); return;
    }
    if (this.hooks.paused()) return;
    if (this.pending) this.applyPending();
    if (!this.active) return;
    this.report = this.monitor.observe(world, 0, this.hooks.runtime(0).current, this.goal);
    this.recordMeasurement(this.report);
    const measuring = this.isMeasuring(this.latest, this.report);
    if (!this.requestAbort && !this.pending && !measuring && this.report.ready && this.report.underperforming
      && this.now() - this.lastRequestAt >= MIN_REQUEST_INTERVAL_MS) void this.request();
  }

  /** Optional explicit review still uses real, accumulated telemetry and actual model generation. */
  async reviewNow(): Promise<void> {
    if (!this.active || !this.report?.ready || this.requestAbort || this.pending) return;
    if (this.now() - this.lastRequestAt < MIN_REQUEST_INTERVAL_MS) {
      this.message = 'Reviews are spaced at least 20 real seconds apart. The current policy continues playing.';
      return;
    }
    await this.request();
  }

  private async request(): Promise<void> {
    const world = this.hooks.world(), report = this.report;
    if (!this.active || !report?.ready || this.requestAbort || this.pending) return;
    const generation = this.generation, match = this.sessionMatch, snapshot = this.hooks.runtime(0).current;
    if (report.team !== 0 || report.goal !== this.goal || report.revision !== snapshot.revision || report.fingerprint !== snapshot.fingerprint) {
      this.cancelPending();
      this.monitor.reset(world, 0, snapshot);
      this.report = this.monitor.observe(world, 0, snapshot, this.goal);
      this.phase = 'observe';
      this.message = 'The policy changed. Measuring a fresh window before asking Astra to edit these weights.';
      return;
    }
    const requestedTime = world.time;
    const previous = this.lastApplied ?? this.latest;
    const previousFeedback = previous ? {
      goal: previous.goal, appliedRevision: previous.appliedRevision, generatedPatch: previous.proposal.patch,
      predictedEffect: previous.proposal.expected_effect, observedAfter: previous.after,
      observationComplete: previous.measurementComplete, observationCanceled: previous.measurementCanceled,
      comparison: previous.comparison ? summarizePlayCoachComparison(previous.comparison, previous.goal) : null,
      latestReviewKeptWeights: this.latest?.proposal.no_change ?? false,
      note: 'Sequential observations have different world states. The paired comparison isolates the weights at its saved checkpoint.',
    } : undefined;
    const request = buildPlayCoachContext(world, 0, snapshot, report, { matchId: match, goal: this.goal, previousFeedback });
    const abort = new AbortController(); this.requestAbort = abort;
    const timeout = setTimeout(() => abort.abort(), 95_000);
    this.lastRequestAt = this.now(); this.requestStartedAt = this.now();
    this.phase = 'request';
    this.message = 'Astra is reading measured gameplay and the current weights. The policy keeps playing while it writes a correction.';
    this.event('Astra reviewing', report.summary);
    try {
      const response = await this.fetcher('/api/empires/astra/program', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request), signal: abort.signal,
      });
      const text = await response.text();
      if (generation !== this.generation || !this.active || match !== this.hooks.matchId()) return;
      if (!response.ok) {
        let detail = `Astra returned HTTP ${response.status}.`;
        try { const body = JSON.parse(text); if (typeof body.detail === 'string') detail = body.detail; } catch { /* HTTP status remains useful. */ }
        throw new Error(detail);
      }
      const proposal = parsePlayCoachPatch(text, this.hooks.runtime(0).current);
      if (proposal.match_id !== match || proposal.team !== 0 || proposal.tick !== request.tick)
        throw new Error('Astra returned a response for a different match, team, or observation.');
      if (this.connection && proposal.requested_model !== this.connection.requested_model)
        throw new Error('The returned program does not match the configured Astra model.');
      this.pending = { proposal, before: report, requestedTime };
      this.phase = 'apply';
      this.message = this.hooks.paused()
        ? 'Astra returned a validated edit. Resume the game to apply it on the next simulation tick.'
        : 'Astra returned a validated edit. Applying it on the next simulation tick.';
      this.connection = { configured: true, verified: true, in_flight: false, requested_model: proposal.requested_model, last_error: null };
      this.connectionError = '';
    } catch (error) {
      if (generation !== this.generation || !this.active) return;
      this.active = false; this.phase = 'error';
      this.message = `${error instanceof Error ? error.message : 'Astra request failed.'} No weights from this request were applied. Start again to retry.`;
      this.event('Review failed', this.message);
    } finally {
      clearTimeout(timeout);
      if (generation === this.generation) { this.requestAbort = null; this.requestStartedAt = null; }
    }
  }

  private applyPending(): void {
    const pending = this.pending;
    if (!pending) return;
    const { proposal, before, requestedTime } = pending;
    const world = this.hooks.world(), runtime = this.hooks.runtime(0), old = runtime.current;
    // Paired continuations start by advancing the checkpoint. Capture between
    // decisions so neither branch skips an order still due on this live tick.
    if (proposal.patch.length && shouldPlayPolicyStep(world)) return;
    this.pending = null;
    let committed: WeightSnapshot | null = null;
    try {
      if (world.time - requestedTime > MAX_CONTEXT_AGE_SECONDS)
        throw new Error('The game advanced more than 120 seconds while Astra was responding. That observation is too old.');
      const checkpoint = world.checkpoint();
      const beforeChoice = pickPlayPolicy(world, 0, old.model)?.label ?? 'No legal order';
      const snapshot = applyPlayCoachPatch(runtime, proposal, world.tick);
      committed = snapshot;
      const cycle: AstraCycle = {
        proposal, goal: this.goal, before, appliedRevision: snapshot?.revision ?? null,
        appliedFingerprint: snapshot?.fingerprint ?? null, appliedTime: snapshot ? world.time : null,
        beforeChoice, afterChoice: pickPlayPolicy(world, 0, runtime.current.model)?.label ?? 'No legal order',
        firstOrder: null, after: null, measurementComplete: false, measurementCanceled: false, comparison: null,
        comparisonMessage: snapshot ? 'Running two isolated 60-second continuations from the exact application state…' : 'Astra kept the weights; no two-policy comparison is needed.',
      };
      this.latest = cycle;
      this.monitor.reset(world, 0, runtime.current);
      if (!snapshot) {
        this.phase = 'observe';
        this.message = 'Astra reviewed the evidence and kept the current weights. Measuring another window.';
        this.event('Astra kept weights', proposal.explanation);
        return;
      }
      this.lastApplied = cycle;
      this.phase = 'measure';
      this.message = `Astra changed ${proposal.patch.length} weights. Blue now plays with revision ${snapshot.revision}; measuring the next 30 game seconds.`;
      this.event(`Applied revision ${snapshot.revision}`, proposal.patch.map(edit => `${edit.tensor}${edit.row === null ? '' : `[${edit.row}]`}${edit.column === null ? '' : `[${edit.column}]`}: ${edit.before} → ${edit.value}`).join(' · '));
      this.hooks.onApplied(old, snapshot);
      this.comparisonAbort?.abort();
      const abort = new AbortController(); this.comparisonAbort = abort;
      const generation = this.generation;
      void Promise.resolve().then(() => this.compare({ checkpoint, team: 0, baselineModel: old.model, candidateModel: snapshot.model,
        otherModel: this.hooks.runtime(1).current.model, otherControl: this.hooks.policyEnabled(1) ? 'policy' : 'hold', seconds: 60,
      }, abort.signal)).then(result => {
        if (generation !== this.generation || cycle !== this.lastApplied) return;
        cycle.comparison = result;
        cycle.comparisonMessage = summarizePlayCoachComparison(result, cycle.goal).summary;
      }).catch(error => {
        if (generation !== this.generation || cycle !== this.lastApplied) return;
        cycle.comparisonMessage = error instanceof Error ? error.message : 'Comparison unavailable; no outcome was fabricated.';
      }).finally(() => { if (this.comparisonAbort === abort) this.comparisonAbort = null; });
    } catch (error) {
      this.active = false; this.phase = 'error';
      const detail = error instanceof Error ? error.message : 'The weight update could not finish.';
      this.message = committed
        ? `Revision ${committed.revision} was applied, but its evidence could not finish updating: ${detail} Astra stopped; those weights remain active.`
        : `${detail} No new patch was applied.`;
      this.event(committed ? 'Applied; evidence interrupted' : 'Patch rejected', this.message);
    }
  }

  recordOrder(label: string, revision: number): void {
    const cycle = this.latest;
    if (!this.active || !cycle || cycle.measurementCanceled || cycle.appliedRevision !== revision || cycle.firstOrder) return;
    cycle.firstOrder = { label, revision, time: this.hooks.world().time };
    this.event('New policy issued an order', `r${revision}: ${label}`);
  }

  dispose(): void { this.stop(); this.statusGeneration++; }
}
