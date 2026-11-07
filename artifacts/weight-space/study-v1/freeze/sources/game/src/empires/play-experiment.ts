import { validateModel, type PolicyModel } from './policy';
import { rankPlayPolicy, shouldPlayPolicyStep } from './play-policy';
import { World, type WorldCheckpoint } from './simulation';
import { DT, type Command, type ProductionStats, type Resources, type Team } from './types';

/** 420 seconds is the offline research horizon; the live inspector still offers its short windows. */
export type PlayExperimentSeconds = 30 | 60 | 120 | 420;
export type PlayOtherControl = 'policy' | 'hold';

export interface PlayExperimentRequest {
  checkpoint: WorldCheckpoint;
  team: Team;
  baselineModel: PolicyModel;
  candidateModel: PolicyModel;
  otherModel: PolicyModel;
  otherControl: PlayOtherControl;
  seconds: PlayExperimentSeconds;
}

/** Lifetime counters are retained; subtract initialMetrics to measure the experiment window. */
export interface PlayExperimentMetrics {
  gathered: Resources;
  stockpile: Resources;
  trained: ProductionStats['trained'];
  constructed: ProductionStats['constructed'];
  recruitsCompleted: number;
  buildingsCompleted: number;
  units: number;
  army: number;
  villagers: number;
  /** Living, completed buildings; foundations do not count. */
  buildings: number;
  /** Population includes reserved places for queued recruits. */
  population: number;
  capacity: number;
  queued: number;
  townHealth: number;
  kills: number;
  /** The world's outcome is always from blue/team 0's perspective, even when team is 1. */
  status: World['status'];
  age: number;
}

export interface PlayExperimentChoice {
  id: string;
  label: string;
  score: number;
  command: Command | null;
  /** A wait choice, or a rejected command, did not execute an order. */
  executed: boolean;
}

export interface PlayExperimentOrder extends PlayExperimentChoice {
  team: Team;
  tick: number;
  time: number;
  elapsedSeconds: number;
  command: Command;
  executed: true;
}

export interface PlayExperimentChangedChoice {
  tick: number;
  time: number;
  elapsedSeconds: number;
  baseline: PlayExperimentChoice | null;
  candidate: PlayExperimentChoice | null;
}

export interface PlayExperimentBranch {
  elapsedSeconds: number;
  metrics: PlayExperimentMetrics;
  otherMetrics: PlayExperimentMetrics;
  executedOrders: number;
  otherExecutedOrders: number;
  /** Number of selected-team policy choices, including wait. */
  decisions: number;
  waitDecisions: number;
  /** Successful non-null commands from both teams, in their actual execution order. */
  orders: PlayExperimentOrder[];
  finalCheckpoint: WorldCheckpoint;
}

export interface PlayExperimentResult {
  team: Team;
  otherControl: PlayOtherControl;
  requestedSeconds: PlayExperimentSeconds;
  startTick: number;
  startTime: number;
  initialMetrics: PlayExperimentMetrics;
  initialOtherMetrics: PlayExperimentMetrics;
  baseline: PlayExperimentBranch;
  candidate: PlayExperimentBranch;
  firstChangedChoice: PlayExperimentChangedChoice | null;
}

export type PlayExperimentId = string | number;
export type PlayExperimentWorkerRequest =
  | { type: 'run'; id: PlayExperimentId; request: PlayExperimentRequest }
  | { type: 'cancel'; id: PlayExperimentId };

export type PlayExperimentWorkerReply =
  | { type: 'progress'; id: PlayExperimentId; progress: number }
  | { type: 'result'; id: PlayExperimentId; result: PlayExperimentResult }
  | { type: 'error'; id: PlayExperimentId; message: string }
  | { type: 'cancelled'; id: PlayExperimentId };

/** Read actual simulation values without inventing rewards or projecting an outcome. */
export function getPlayExperimentMetrics(world: World, team: Team): PlayExperimentMetrics {
  const player = world.players[team];
  const own = world.entities.filter(entity => entity.owner === team && entity.hp > 0);
  const units = own.filter(entity => entity.category === 'unit');
  const production = structuredClone(world.production[team]);
  return {
    gathered: { ...player.gathered }, stockpile: { ...player.resources }, ...production,
    recruitsCompleted: Object.entries(production.trained).reduce((sum, [kind, count]) => sum + (kind === 'villager' ? 0 : count), 0),
    buildingsCompleted: Object.values(production.constructed).reduce((sum, count) => sum + count, 0),
    units: units.length, army: units.filter(entity => entity.kind !== 'villager').length,
    villagers: units.filter(entity => entity.kind === 'villager').length,
    buildings: own.filter(entity => entity.category === 'building' && entity.construction >= 1).length,
    population: world.population(team), capacity: world.capacity(team),
    queued: own.reduce((sum, entity) => sum + entity.queue.length, 0),
    townHealth: own.filter(entity => entity.kind === 'towncenter').reduce((sum, entity) => sum + entity.hp, 0),
    kills: player.kills, status: world.status, age: player.age,
  };
}

interface BranchState {
  world: World;
  model: PolicyModel;
  executedOrders: number;
  otherExecutedOrders: number;
  decisions: number;
  waitDecisions: number;
  orders: PlayExperimentOrder[];
}

interface TimedChoice {
  tick: number;
  time: number;
  choice: PlayExperimentChoice;
}

function captureModel(model: PolicyModel): PolicyModel {
  const captured = structuredClone(model);
  validateModel(captured);
  return captured;
}

function sameChoice(a: PlayExperimentChoice | null, b: PlayExperimentChoice | null): boolean {
  if (!a || !b) return a === b;
  // Candidate commands are constructed with a stable property order. Target IDs,
  // unit IDs, positions and other command fields all matter; a score change alone does not.
  return a.id === b.id && JSON.stringify(a.command) === JSON.stringify(b.command);
}

/**
 * Two isolated continuations of one captured world. The selected side runs its
 * policy in both futures. The other side either runs the same captured policy or
 * keeps existing orders with no new commands. All costs, timers and combat use World.
 */
export class PlayExperiment {
  private readonly team: Team;
  private readonly otherTeam: Team;
  private readonly otherControl: PlayOtherControl;
  private readonly otherModel: PolicyModel;
  private readonly seconds: PlayExperimentSeconds;
  private readonly startTick: number;
  private readonly startTime: number;
  private readonly initialMetrics: PlayExperimentMetrics;
  private readonly initialOtherMetrics: PlayExperimentMetrics;
  private readonly horizonTicks: number;
  private readonly baseline: BranchState;
  private readonly candidate: BranchState;
  private elapsedTicks = 0;
  private firstChangedChoice: PlayExperimentChangedChoice | null = null;

  constructor(request: PlayExperimentRequest) {
    if (request.team !== 0 && request.team !== 1) throw new Error('Choose blue or red for the experiment.');
    if (![30, 60, 120, 420].includes(request.seconds)) throw new Error('Experiment duration must be 30, 60, 120, or 420 seconds.');
    if (request.otherControl !== 'policy' && request.otherControl !== 'hold') throw new Error('Other-side control must be policy or hold.');
    this.team = request.team;
    this.otherTeam = request.team === 0 ? 1 : 0;
    this.otherControl = request.otherControl;
    this.otherModel = captureModel(request.otherModel);
    this.seconds = request.seconds;
    this.horizonTicks = Math.round(request.seconds / DT);
    const branch = (model: PolicyModel): BranchState => ({
      world: World.fromCheckpoint(request.checkpoint), model: captureModel(model),
      executedOrders: 0, otherExecutedOrders: 0, decisions: 0, waitDecisions: 0, orders: [],
    });
    this.baseline = branch(request.baselineModel);
    this.candidate = branch(request.candidateModel);
    this.startTick = this.baseline.world.tick;
    this.startTime = this.baseline.world.time;
    this.initialMetrics = getPlayExperimentMetrics(this.baseline.world, this.team);
    this.initialOtherMetrics = getPlayExperimentMetrics(this.baseline.world, this.otherTeam);
  }

  get done(): boolean {
    return this.elapsedTicks >= this.horizonTicks
      || (this.baseline.world.status !== 'playing' && this.candidate.world.status !== 'playing');
  }

  get progress(): number {
    return this.done ? 1 : this.elapsedTicks / this.horizonTicks;
  }

  /** Advance a bounded number of paired fixed ticks; return true when results are ready. */
  advance(maxTicks = 20): boolean {
    if (!Number.isSafeInteger(maxTicks) || maxTicks <= 0) throw new Error('Batch size must be a positive whole number.');
    for (let count = 0; count < maxTicks && !this.done; count++) {
      const baseline = this.stepBranch(this.baseline);
      const candidate = this.stepBranch(this.candidate);
      this.elapsedTicks++;
      if (!this.firstChangedChoice && !sameChoice(baseline?.choice ?? null, candidate?.choice ?? null)) {
        const when = baseline ?? candidate!;
        this.firstChangedChoice = {
          tick: when.tick, time: when.time, elapsedSeconds: when.time - this.startTime,
          baseline: baseline?.choice ?? null, candidate: candidate?.choice ?? null,
        };
      }
    }
    return this.done;
  }

  /** Every return is detached; inspecting or editing a result cannot affect either future. */
  result(): PlayExperimentResult {
    if (!this.done) throw new Error('The paired futures are still running.');
    return {
      team: this.team, otherControl: this.otherControl, requestedSeconds: this.seconds,
      startTick: this.startTick, startTime: this.startTime,
      initialMetrics: structuredClone(this.initialMetrics), initialOtherMetrics: structuredClone(this.initialOtherMetrics),
      baseline: this.branchResult(this.baseline), candidate: this.branchResult(this.candidate),
      firstChangedChoice: structuredClone(this.firstChangedChoice),
    };
  }

  private stepBranch(branch: BranchState): TimedChoice | null {
    const { world } = branch;
    if (world.status !== 'playing') return null;
    world.step(DT);
    if (!shouldPlayPolicyStep(world)) return null;
    let selected: TimedChoice | null = null;
    // Same order as live play. Red observes any blue order issued on this tick.
    for (const team of [0, 1] as const) {
      if (team !== this.team && this.otherControl === 'hold') continue;
      const ranked = rankPlayPolicy(world, team, team === this.team ? branch.model : this.otherModel)[0];
      if (!ranked) continue;
      const command = structuredClone(ranked.candidate.command);
      const executed = command !== null && world.command(team, command).ok;
      const choice: PlayExperimentChoice = {
        id: ranked.candidate.id, label: ranked.candidate.label, score: ranked.score, command, executed,
      };
      if (team === this.team) {
        branch.decisions++;
        if (!command) branch.waitDecisions++;
        selected = { tick: world.tick, time: world.time, choice };
      }
      if (command && executed) {
        if (team === this.team) branch.executedOrders++;
        else branch.otherExecutedOrders++;
        branch.orders.push({ ...choice, team, tick: world.tick, time: world.time, elapsedSeconds: world.time - this.startTime, command, executed: true });
      }
    }
    return selected;
  }

  private branchResult(branch: BranchState): PlayExperimentBranch {
    return {
      elapsedSeconds: branch.world.time - this.startTime,
      metrics: getPlayExperimentMetrics(branch.world, this.team),
      otherMetrics: getPlayExperimentMetrics(branch.world, this.otherTeam),
      executedOrders: branch.executedOrders, otherExecutedOrders: branch.otherExecutedOrders,
      decisions: branch.decisions, waitDecisions: branch.waitDecisions,
      orders: structuredClone(branch.orders), finalCheckpoint: branch.world.checkpoint(),
    };
  }
}

export function runPlayExperiment(request: PlayExperimentRequest): PlayExperimentResult {
  const experiment = new PlayExperiment(request);
  while (!experiment.advance()) { /* The worker uses one batch per event-loop turn. */ }
  return experiment.result();
}
