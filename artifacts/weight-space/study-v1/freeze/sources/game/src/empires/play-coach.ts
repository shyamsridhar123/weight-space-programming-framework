import { getPlayExperimentMetrics, type PlayExperimentResult } from './play-experiment';
import { rankPlayPolicy } from './play-policy';
import { policyFeatures, type PolicyModel } from './policy';
import type { World } from './simulation';
import { RESOURCES, UNIT_DEFS, type Resources, type Team } from './types';
import {
  cloneWeightModel, WeightRevisionConflictError, WeightRuntime, WeightValidationError,
  type WeightEdit, type WeightSnapshot,
} from './weight-runtime';

export const PLAY_COACH_GOALS = Object.freeze({
  'grow-army': 'Grow a sustainable army of at least 8 living soldiers, targeting 4 completed soldiers per game minute until that size is reached. Keep villagers delivering food and wood, maintain housing, and preserve the Town Center. Avoid sacrificing an undersized army.',
  'grow-economy': 'Increase food and wood actually delivered by villagers. Put idle workers to work, maintain housing, and preserve the Town Center. Spending stored resources is not evidence of gathering.',
  defend: 'Preserve the Town Center and villagers. Respond to visible nearby enemies, replace lost soldiers, and keep the economy working.',
  attack: 'Use a sufficiently large army to defeat the rival while protecting home. Measure completed recruits, surviving troops, kills, and damage observed on visible enemies; do not assume hidden enemy state.',
});
export type PlayCoachGoal = keyof typeof PLAY_COACH_GOALS;

export interface PlayCoachMeasurements {
  tick: number;
  time: number;
  stockpile: Resources;
  gathered: Resources;
  army: number;
  villagers: number;
  idleWorkers: number;
  idleArmy: number;
  queuedArmy: number;
  /** Training time left through the first queued soldier; excludes travel after spawning. */
  nextRecruitSeconds: number | null;
  queuedVillagers: number;
  recruitsCompleted: number;
  villagersCompleted: number;
  population: number;
  capacity: number;
  townHealth: number;
  kills: number;
  canRecruit: boolean;
  threatsNearHome: number;
  status: World['status'];
}

export interface PlayCoachProblem {
  code: 'idle-workers' | 'no-resource-returns' | 'recruit-stagnation' | 'recruit-rate' | 'production-stalled'
    | 'town-damage' | 'army-losses' | 'worker-losses' | 'idle-army' | 'housing-blocked';
  detail: string;
}

export interface PlayCoachPerformance {
  team: Team;
  goal: PlayCoachGoal;
  revision: number | null;
  fingerprint: string | null;
  startTime: number;
  endTime: number;
  startTick: number;
  endTick: number;
  elapsedSeconds: number;
  minimumSeconds: number;
  sampleSeconds: number;
  ready: boolean;
  underperforming: boolean;
  baseline: PlayCoachMeasurements;
  current: PlayCoachMeasurements;
  returned: Resources;
  returnedPerMinute: Resources;
  recruitsCompleted: number;
  recruitsPerMinute: number;
  targetRecruitsPerMinute: number;
  targetArmy: number;
  villagersCompleted: number;
  armyChange: number;
  /** Time-weighted sampled idle villagers / all villagers, never a resource-return proxy. */
  idleWorkerFraction: number;
  ownDamage: number;
  townDamage: number;
  /** Lower bound: only HP reductions on enemies visible in two consecutive samples. */
  observedEnemyDamage: number;
  armyLost: number;
  villagersLost: number;
  kills: number;
  problems: PlayCoachProblem[];
  summary: string;
}

interface Sample {
  measures: PlayCoachMeasurements;
  own: Map<number, { hp: number; kind: string; category: string }>;
  visibleEnemies: Map<number, number>;
  workersSeconds: number;
  idleWorkerSeconds: number;
  ownDamage: number;
  townDamage: number;
  observedEnemyDamage: number;
  armyLost: number;
  villagersLost: number;
}

function measurements(world: World, team: Team): PlayCoachMeasurements {
  const metrics = getPlayExperimentMetrics(world, team);
  const own = world.entities.filter(entity => entity.owner === team && entity.hp > 0);
  const home = own.find(entity => entity.kind === 'towncenter');
  const firstRecruits = own.flatMap(entity => {
    let remaining = 0;
    for (const job of entity.queue) {
      remaining += Math.max(0, job.remaining);
      if (job.kind !== 'villager') return [remaining];
    }
    return [];
  });
  return {
    tick: world.tick, time: world.time, stockpile: metrics.stockpile, gathered: metrics.gathered,
    army: metrics.army, villagers: metrics.villagers,
    idleWorkers: own.filter(entity => entity.kind === 'villager' && entity.order.type === 'idle').length,
    idleArmy: own.filter(entity => entity.category === 'unit' && entity.kind !== 'villager' && entity.order.type === 'idle').length,
    queuedArmy: own.reduce((sum, entity) => sum + entity.queue.filter(job => job.kind !== 'villager').length, 0),
    nextRecruitSeconds: firstRecruits.length ? Math.min(...firstRecruits) : null,
    queuedVillagers: own.reduce((sum, entity) => sum + entity.queue.filter(job => job.kind === 'villager').length, 0),
    recruitsCompleted: metrics.recruitsCompleted, villagersCompleted: metrics.trained.villager ?? 0,
    population: metrics.population, capacity: metrics.capacity, townHealth: metrics.townHealth,
    kills: metrics.kills, status: world.status,
    canRecruit: own.some(entity => entity.kind === 'barracks' && world.canCommand(team, {
      type: 'train', buildingId: entity.id, kind: 'spearman',
    }).ok),
    threatsNearHome: !home ? 0 : world.entities.filter(entity => entity.owner !== team && entity.hp > 0
      && entity.category === 'unit' && world.isVisible(team, entity.x, entity.z)
      && Math.hypot(entity.x - home.x, entity.z - home.z) < 18).length,
  };
}

const resourceDelta = (a: Resources, b: Resources): Resources => Object.fromEntries(
  RESOURCES.map(resource => [resource, b[resource] - a[resource]]),
) as Resources;

/** Observes the actual running world. It never issues orders, writes weights, or advances time. */
export class PlayCoachMonitor {
  readonly windowSeconds: number;
  readonly minimumSeconds: number;
  readonly sampleSeconds: number;
  private world: World | null = null;
  private team: Team = 0;
  private revision: number | null = null;
  private fingerprint: string | null = null;
  private samples: Sample[] = [];

  constructor(options: { windowSeconds?: number; minimumSeconds?: number; sampleSeconds?: number } = {}) {
    this.windowSeconds = options.windowSeconds ?? 30;
    this.minimumSeconds = options.minimumSeconds ?? 20;
    this.sampleSeconds = options.sampleSeconds ?? 1;
    if (!Number.isFinite(this.windowSeconds) || this.windowSeconds < 2 || this.windowSeconds > 300
      || !Number.isFinite(this.minimumSeconds) || this.minimumSeconds <= 0 || this.minimumSeconds > this.windowSeconds
      || !Number.isFinite(this.sampleSeconds) || this.sampleSeconds < 0.1 || this.sampleSeconds > this.minimumSeconds)
      throw new Error('Coach windows must be finite: 2..300 seconds, with a positive minimum and a 0.1s-or-longer sampling interval.');
  }

  reset(world?: World, team: Team = 0, snapshot?: WeightSnapshot): void {
    this.world = world ?? null;
    this.team = team;
    this.revision = snapshot?.revision ?? null;
    this.fingerprint = snapshot?.fingerprint ?? null;
    this.samples = [];
    if (world) this.sample(world, team);
  }

  observe(world: World, team: Team, snapshot?: WeightSnapshot, goal: PlayCoachGoal = 'grow-army'): PlayCoachPerformance {
    if (team !== 0 && team !== 1) throw new Error('Coach must observe blue or red.');
    if (!Object.hasOwn(PLAY_COACH_GOALS, goal)) throw new Error('Unknown coach goal.');
    const last = this.samples.at(-1);
    if (world !== this.world || team !== this.team || (last && world.tick < last.measures.tick)
      || this.revision !== (snapshot?.revision ?? null) || this.fingerprint !== (snapshot?.fingerprint ?? null)) {
      this.reset(world, team, snapshot);
    } else if (!last || world.time - last.measures.time >= this.sampleSeconds - 1e-7
      || (last.measures.status === 'playing' && world.status !== 'playing')) {
      this.sample(world, team);
    }
    return this.report(goal);
  }

  private sample(world: World, team: Team): void {
    const previous = this.samples.at(-1);
    const own = new Map(world.entities.filter(entity => entity.owner === team && entity.hp > 0)
      .map(entity => [entity.id, { hp: entity.hp, kind: entity.kind, category: entity.category }]));
    const visibleEnemies = new Map(world.entities.filter(entity => entity.owner !== team && entity.hp > 0
      && world.isVisible(team, entity.x, entity.z)).map(entity => [entity.id, entity.hp]));
    const sample: Sample = {
      measures: measurements(world, team), own, visibleEnemies,
      workersSeconds: previous?.workersSeconds ?? 0, idleWorkerSeconds: previous?.idleWorkerSeconds ?? 0,
      ownDamage: previous?.ownDamage ?? 0, townDamage: previous?.townDamage ?? 0,
      observedEnemyDamage: previous?.observedEnemyDamage ?? 0,
      armyLost: previous?.armyLost ?? 0, villagersLost: previous?.villagersLost ?? 0,
    };
    if (previous) {
      const elapsed = world.time - previous.measures.time;
      sample.workersSeconds += previous.measures.villagers * elapsed;
      sample.idleWorkerSeconds += previous.measures.idleWorkers * elapsed;
      for (const [id, before] of previous.own) {
        const after = own.get(id);
        const damage = Math.max(0, before.hp - (after?.hp ?? 0));
        sample.ownDamage += damage;
        if (before.kind === 'towncenter') sample.townDamage += damage;
        if (!after && before.category === 'unit') {
          if (before.kind === 'villager') sample.villagersLost++;
          else sample.armyLost++;
        }
      }
      for (const [id, beforeHp] of previous.visibleEnemies) {
        const afterHp = visibleEnemies.get(id);
        // Disappearing into fog cannot be credited as enemy damage or a kill.
        if (afterHp !== undefined) sample.observedEnemyDamage += Math.max(0, beforeHp - afterHp);
      }
    }
    this.samples.push(sample);
    // Fixed 0.1s ticks accumulate rounding error. Match the sampling/readiness
    // tolerance so a boundary sample cannot stretch a 30s window to 31s.
    while (this.samples.length > 2 && this.samples[1].measures.time <= world.time - this.windowSeconds + 1e-7)
      this.samples.shift();
  }

  private report(goal: PlayCoachGoal): PlayCoachPerformance {
    const first = this.samples[0], last = this.samples.at(-1)!;
    const a = first.measures, b = last.measures, elapsedSeconds = Math.max(0, b.time - a.time);
    const returned = resourceDelta(a.gathered, b.gathered);
    const recruitsCompleted = b.recruitsCompleted - a.recruitsCompleted;
    const idleWorkerFraction = (last.idleWorkerSeconds - first.idleWorkerSeconds)
      / Math.max(1e-9, last.workersSeconds - first.workersSeconds);
    const townDamage = last.townDamage - first.townDamage;
    const armyLost = last.armyLost - first.armyLost, villagersLost = last.villagersLost - first.villagersLost;
    const ready = elapsedSeconds >= this.minimumSeconds - 1e-7;
    const problems: PlayCoachProblem[] = [];
    if (ready && b.status === 'playing') {
      if (b.villagers > 0 && idleWorkerFraction >= 0.25)
        problems.push({ code: 'idle-workers', detail: `${Math.round(idleWorkerFraction * 100)}% of sampled worker time was idle; ${b.idleWorkers} villagers are idle now.` });
      if (b.villagers > 0 && returned.food + returned.wood < 1)
        problems.push({ code: 'no-resource-returns', detail: `No food or wood was delivered in ${elapsedSeconds.toFixed(1)} game seconds. Carrying and stockpile spending are not deliveries.` });
      if (townDamage > 0)
        problems.push({ code: 'town-damage', detail: `The Town Center lost ${townDamage.toFixed(1)} HP during this window.` });
      if (armyLost > 0)
        problems.push({ code: 'army-losses', detail: `${armyLost} soldiers died during this window; ${recruitsCompleted} recruits completed.` });
      if (villagersLost > 0)
        problems.push({ code: 'worker-losses', detail: `${villagersLost} villagers died during this window.` });
      if (goal !== 'grow-economy' && b.army < 8 && recruitsCompleted === 0 && b.queuedArmy === 0)
        problems.push({ code: 'recruit-stagnation', detail: `No soldiers completed or remain queued; army ${b.army}/8. ${b.canRecruit ? 'A spearman is currently affordable and legal.' : 'Recruitment is currently blocked by resources, housing, or the producer.'}` });
      // Initial commands take up to 2s to be chosen. A recruit completing within
      // that interval is counted soon rather than branded a failure at the boundary.
      if (goal === 'grow-army' && b.army < 8 && elapsedSeconds >= 30 - 1e-7
        && recruitsCompleted * 60 / elapsedSeconds < 4 - 1e-7
        && (b.nextRecruitSeconds === null || b.nextRecruitSeconds > 2 + 1e-7)
        && !problems.some(problem => problem.code === 'recruit-stagnation'))
        problems.push({ code: 'recruit-rate', detail: `${recruitsCompleted} soldiers completed in ${elapsedSeconds.toFixed(1)}s (${(recruitsCompleted * 60 / elapsedSeconds).toFixed(1)}/minute); target 4/minute until army 8. ${b.queuedArmy} queued${b.nextRecruitSeconds === null ? '' : `; next needs at least ${b.nextRecruitSeconds.toFixed(1)}s`}.` });
      if (goal !== 'grow-economy' && recruitsCompleted === 0 && a.queuedArmy > 0 && b.queuedArmy > 0
        && elapsedSeconds >= UNIT_DEFS.spearman.time + 5 && this.samples.every(sample => sample.measures.queuedArmy > 0))
        problems.push({ code: 'production-stalled', detail: `Soldiers were queued throughout ${elapsedSeconds.toFixed(1)} seconds but no recruit completed. Normal spearman training takes ${UNIT_DEFS.spearman.time}s; inspect blocked spawning or production.` });
      if (goal !== 'grow-economy' && b.army < 8 && b.population >= b.capacity && b.capacity < 100)
        problems.push({ code: 'housing-blocked', detail: `Population is ${b.population}/${b.capacity}; another recruit needs completed housing.` });
      if ((goal === 'attack' && b.army >= 4) || (goal === 'defend' && b.threatsNearHome > 0)) {
        if (b.idleArmy >= Math.max(1, Math.ceil(b.army / 2)))
          problems.push({ code: 'idle-army', detail: `${b.idleArmy}/${b.army} soldiers are idle${b.threatsNearHome ? ` with ${b.threatsNearHome} visible threats near home` : ' while the goal is to attack'}.` });
      }
    }
    return {
      team: this.team, goal, revision: this.revision, fingerprint: this.fingerprint,
      startTime: a.time, endTime: b.time, startTick: a.tick, endTick: b.tick, elapsedSeconds,
      minimumSeconds: this.minimumSeconds, sampleSeconds: this.sampleSeconds, ready,
      underperforming: ready && problems.length > 0,
      baseline: structuredClone(a), current: structuredClone(b), returned,
      returnedPerMinute: Object.fromEntries(RESOURCES.map(resource => [resource, elapsedSeconds ? returned[resource] * 60 / elapsedSeconds : 0])) as Resources,
      recruitsCompleted, recruitsPerMinute: elapsedSeconds ? recruitsCompleted * 60 / elapsedSeconds : 0,
      targetRecruitsPerMinute: 4, targetArmy: 8,
      villagersCompleted: b.villagersCompleted - a.villagersCompleted, armyChange: b.army - a.army,
      idleWorkerFraction, ownDamage: last.ownDamage - first.ownDamage, townDamage,
      observedEnemyDamage: last.observedEnemyDamage - first.observedEnemyDamage, armyLost, villagersLost,
      kills: b.kills - a.kills, problems,
      summary: !ready ? `Observing ${elapsedSeconds.toFixed(0)}/${this.minimumSeconds} game seconds before assessing performance.`
        : b.status !== 'playing' ? 'Match ended; no automatic correction requested.'
          : problems.length ? problems.map(problem => problem.detail).join(' ')
            : `${elapsedSeconds.toFixed(0)}s observed: ${Math.floor(returned.food)} food + ${Math.floor(returned.wood)} wood delivered, ${recruitsCompleted} recruits completed, army ${a.army} → ${b.army}. No configured underperformance trigger fired.`,
    };
  }
}

export interface PlayCoachRequest {
  match_id: string;
  team: Team;
  tick: number;
  goal: string;
  base_revision: number;
  base_fingerprint: string;
  model: PolicyModel;
  state: Record<string, unknown>;
  performance: PlayCoachPerformance;
  decisions: { id: string; label: string; input: number[]; score: number }[];
}

/** Full real network tensors plus a fog-filtered observation; there is no generated patch here. */
export function buildPlayCoachContext(
  world: World, team: Team, snapshot: WeightSnapshot, performance: PlayCoachPerformance,
  options: { matchId: string; goal?: PlayCoachGoal; goalText?: string; previousFeedback?: unknown },
): PlayCoachRequest {
  if (performance.team !== team || performance.revision !== snapshot.revision || performance.fingerprint !== snapshot.fingerprint)
    throw new WeightRevisionConflictError('Performance feedback belongs to another team or weight revision. Observe this policy before requesting a correction.');
  const goal = options.goal ?? performance.goal;
  if (!Object.hasOwn(PLAY_COACH_GOALS, goal)) throw new Error('Unknown coach goal.');
  const state = JSON.parse(world.observe(team)) as Record<string, unknown>;
  state.production = structuredClone(world.production[team]);
  state.training = world.entities.filter(entity => entity.owner === team && entity.hp > 0 && entity.queue.length)
    .map(entity => ({ buildingId: entity.id, kind: entity.kind, queue: entity.queue.map(job => ({ ...job })) }));
  state.policy = {
    architecture: '32 inputs → 38 ReLU hidden neurons → 1 candidate score',
    decisionRule: 'Highest score among the legal candidates. A decision is taken every 2 game seconds. Temperature is not used for this argmax decision.',
    forward: 'score = b2 + sum_h(w2[h] * max(0, b1[h] + sum_f(w1[h][f] * input[f])))',
    constraints: 'Only food/wood gathering, villagers/spearmen, houses/farms/replacement barracks, legal movement/combat, and continue-orders are available. Costs, population limits, construction, travel, and training still apply.',
    training: 'Authored initial policy. Corrections directly edit numerical weights; no gradient training or fine-tuning runs.',
  };
  state.costs = { spearman: { ...UNIT_DEFS.spearman.cost, seconds: UNIT_DEFS.spearman.time }, villager: { ...UNIT_DEFS.villager.cost, seconds: UNIT_DEFS.villager.time } };
  if (options.previousFeedback !== undefined) state.previousFeedback = structuredClone(options.previousFeedback);
  return {
    match_id: options.matchId, team, tick: world.tick,
    goal: options.goalText?.trim() || PLAY_COACH_GOALS[goal],
    base_revision: snapshot.revision, base_fingerprint: snapshot.fingerprint,
    model: cloneWeightModel(snapshot.model), state, performance: structuredClone(performance),
    decisions: rankPlayPolicy(world, team, snapshot.model).slice(0, 32).map(({ candidate, score }) => ({
      id: candidate.id, label: candidate.label, input: policyFeatures(world, team, candidate), score,
    })),
  };
}

export interface PlayCoachScalarEdit {
  tensor: 'w1' | 'b1' | 'w2' | 'b2';
  row: number | null;
  column: number | null;
  before: number;
  value: number;
}

export interface PlayCoachProposal {
  kind: 'astra-weight-program';
  match_id: string;
  team: Team;
  tick: number;
  base_revision: number;
  base_fingerprint: string;
  patch: PlayCoachScalarEdit[];
  explanation: string;
  expected_effect: string;
  no_change: boolean;
  requested_model: string;
  returned_model: string;
  response_id: string;
  latency_ms: number;
  request_evidence_id: string;
  response_evidence_id: string;
  evidence_url: string;
  source: 'astra';
  applied: false;
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)
    || ![Object.prototype, null].includes(Object.getPrototypeOf(value)))
    throw new WeightValidationError(`${label} must be a plain object.`);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || !Object.hasOwn(Object.getOwnPropertyDescriptor(value, key)!, 'value'))
      throw new WeightValidationError(`${label} must contain data fields.`);
  }
  return value as Record<string, unknown>;
}

function text(value: unknown, label: string, maximum = 3000): string {
  if (typeof value !== 'string' || !value.trim() || value.length > maximum || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value))
    throw new WeightValidationError(`${label} must be nonempty text of at most ${maximum} characters.`);
  return value;
}

function integer(value: unknown, label: string, maximum = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value > maximum)
    throw new WeightValidationError(`${label} must be an integer from 0 to ${maximum}.`);
  return value;
}

function scalar(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < -64 || value > 64)
    throw new WeightValidationError(`${label} must be finite and between -64 and 64.`);
  return Object.is(value, -0) ? 0 : value;
}

function guardBase(proposal: Pick<PlayCoachProposal, 'base_revision' | 'base_fingerprint'>, snapshot: WeightSnapshot): void {
  if (proposal.base_revision !== snapshot.revision || proposal.base_fingerprint !== snapshot.fingerprint)
    throw new WeightRevisionConflictError('Astra response is stale: the active policy changed after the request.');
}

/** Parse the server response again at the browser boundary before touching any live tensor. */
export function parsePlayCoachPatch(raw: unknown, snapshot: WeightSnapshot): PlayCoachProposal {
  let decoded = raw;
  if (typeof raw === 'string') {
    if (new TextEncoder().encode(raw).byteLength > 65_536) throw new WeightValidationError('Astra response is too large.');
    try { decoded = JSON.parse(raw); } catch { throw new WeightValidationError('Astra response must be valid JSON.'); }
  }
  const value = object(decoded, 'Astra response');
  if (value.kind !== 'astra-weight-program' || value.source !== 'astra' || value.applied !== false)
    throw new WeightValidationError('Expected an unapplied Astra weight program.');
  const base_revision = integer(value.base_revision, 'base_revision');
  const base_fingerprint = text(value.base_fingerprint, 'base_fingerprint', 80);
  guardBase({ base_revision, base_fingerprint }, snapshot);
  if (value.team !== 0 && value.team !== 1) throw new WeightValidationError('Astra response must identify blue or red.');
  if (typeof value.no_change !== 'boolean') throw new WeightValidationError('no_change must be a boolean.');
  if (!Array.isArray(value.patch) || value.patch.length > 16)
    throw new WeightValidationError('Astra may return at most 16 scalar edits.');
  if (value.no_change !== (value.patch.length === 0))
    throw new WeightValidationError('A no-change decision requires an empty patch; a correction requires at least one edit.');
  const entries: unknown[] = [];
  for (let index = 0; index < value.patch.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value.patch, String(index));
    if (!descriptor || !Object.hasOwn(descriptor, 'value'))
      throw new WeightValidationError('Astra patch must be a dense array of data fields.');
    entries.push(descriptor.value);
  }
  const seen = new Set<string>();
  const patch = entries.map((entry, index): PlayCoachScalarEdit => {
    const edit = object(entry, `Patch edit ${index}`);
    if (Object.keys(edit).some(key => !['tensor', 'row', 'column', 'before', 'value'].includes(key)))
      throw new WeightValidationError('Astra edits may only contain tensor, row, column, before, and value.');
    const tensor = edit.tensor;
    if (tensor !== 'w1' && tensor !== 'b1' && tensor !== 'w2' && tensor !== 'b2')
      throw new WeightValidationError('Astra may edit only w1, b1, w2, or b2.');
    const row = tensor === 'b2' ? null : integer(edit.row, `${tensor} row`, 37);
    const column = tensor === 'w1' ? integer(edit.column, 'w1 column', 31) : null;
    if ((row === null && edit.row !== null) || (column === null && edit.column !== null))
      throw new WeightValidationError('Unused scalar coordinates must be null.');
    const before = scalar(edit.before, 'before'), after = scalar(edit.value, 'value');
    const current = tensor === 'w1' ? snapshot.model.w1[row!][column!]
      : tensor === 'b2' ? snapshot.model.b2 : snapshot.model[tensor][row!];
    if (before !== current) throw new WeightRevisionConflictError(`Astra edit ${tensor}[${row ?? ''}] has a stale before-value.`);
    if (before === after) throw new WeightValidationError('Astra correction contains a no-op edit.');
    const key = `${tensor}:${row}:${column}`;
    if (seen.has(key)) throw new WeightValidationError('Astra cannot edit one scalar twice in the same patch.');
    seen.add(key);
    return { tensor, row, column, before, value: after };
  });
  if (typeof value.latency_ms !== 'number' || !Number.isFinite(value.latency_ms) || value.latency_ms < 0)
    throw new WeightValidationError('latency_ms must be a finite nonnegative number.');
  const requested_model = text(value.requested_model, 'requested_model', 200);
  const returned_model = text(value.returned_model, 'returned_model', 200);
  if (requested_model !== returned_model)
    throw new WeightValidationError('Astra response model does not match the requested model.');
  return {
    kind: 'astra-weight-program', source: 'astra', applied: false,
    match_id: text(value.match_id, 'match_id', 200), team: value.team, tick: integer(value.tick, 'tick'),
    base_revision, base_fingerprint, patch, no_change: value.no_change,
    explanation: text(value.explanation, 'explanation'), expected_effect: text(value.expected_effect, 'expected_effect'),
    requested_model, returned_model,
    response_id: text(value.response_id, 'response_id', 4096), latency_ms: value.latency_ms,
    request_evidence_id: text(value.request_evidence_id, 'request_evidence_id', 200),
    response_evidence_id: text(value.response_evidence_id, 'response_evidence_id', 200),
    evidence_url: text(value.evidence_url, 'evidence_url', 1000),
  };
}

function weightEdits(proposal: PlayCoachProposal): WeightEdit[] {
  return proposal.patch.map(edit => ({
    tensor: edit.tensor, value: edit.value,
    ...(edit.row === null ? {} : { row: edit.row }), ...(edit.column === null ? {} : { column: edit.column }),
  }));
}

/** Detached candidate for paired evaluation. Creating it does not change the live revision/history. */
export function previewPlayCoachPatch(snapshot: WeightSnapshot, raw: unknown): PolicyModel {
  const proposal = parsePlayCoachPatch(raw, snapshot);
  if (proposal.no_change) return cloneWeightModel(snapshot.model);
  const isolated = new WeightRuntime(snapshot.model);
  return cloneWeightModel(isolated.patch(weightEdits(proposal), { source: 'astra-preview', reason: proposal.explanation.slice(0, 800) }).model);
}

/** One validated atomic commit. An explicit keep never manufactures a new revision. */
export function applyPlayCoachPatch(runtime: WeightRuntime, raw: unknown, tick: number): WeightSnapshot | null {
  const proposal = parsePlayCoachPatch(raw, runtime.current);
  if (proposal.no_change) return null;
  return runtime.patch(weightEdits(proposal), {
    source: 'astra', reason: proposal.explanation.slice(0, 800), tick,
    expectedRevision: proposal.base_revision, expectedFingerprint: proposal.base_fingerprint,
  });
}

export interface PlayCoachComparison {
  goal: PlayCoachGoal;
  horizonSeconds: number;
  improved: boolean;
  regressed: boolean;
  summary: string;
  changes: { deliveredFood: number; deliveredWood: number; army: number; recruitsCompleted: number; villagers: number; townHealth: number; enemyTownHealth: number; kills: number };
}

/** A transparent short-horizon measurement, not a promise that an edit wins future games. */
export function summarizePlayCoachComparison(result: PlayExperimentResult, goal: PlayCoachGoal = 'grow-army'): PlayCoachComparison {
  const before = result.baseline.metrics, after = result.candidate.metrics;
  const changes = {
    deliveredFood: after.gathered.food - before.gathered.food,
    deliveredWood: after.gathered.wood - before.gathered.wood,
    army: after.army - before.army, recruitsCompleted: after.recruitsCompleted - before.recruitsCompleted,
    villagers: after.villagers - before.villagers, townHealth: after.townHealth - before.townHealth,
    enemyTownHealth: result.candidate.otherMetrics.townHealth - result.baseline.otherMetrics.townHealth,
    kills: after.kills - before.kills,
  };
  const ownLoss = result.team === 0 ? 'lost' : 'won';
  const newDefeat = after.status === ownLoss && before.status !== ownLoss;
  const workerDeaths = (metrics: typeof before) => Math.max(0, result.initialMetrics.villagers
    + (metrics.trained.villager ?? 0) - (result.initialMetrics.trained.villager ?? 0) - metrics.villagers);
  // Training fewer new workers is a visible production tradeoff, not a death.
  const regressed = newDefeat || changes.townHealth < -1e-6 || workerDeaths(after) > workerDeaths(before);
  const objectiveImproved = goal === 'grow-economy' ? changes.deliveredFood + changes.deliveredWood > 1e-6
    : goal === 'defend' ? changes.townHealth > 1e-6 || changes.army > 0 || changes.kills > 0
      : goal === 'attack' ? changes.enemyTownHealth < -1e-6 || changes.kills > 0
        : changes.army > 0;
  const improved = objectiveImproved && !regressed;
  const amount = (number: number) => `${number > 0 ? '+' : ''}${Math.round(number)}`;
  return {
    goal, horizonSeconds: result.requestedSeconds, improved, regressed, changes,
    summary: `${result.requestedSeconds}s paired simulation from one checkpoint: army ${amount(changes.army)}, completed recruits ${amount(changes.recruitsCompleted)}, returned food ${amount(changes.deliveredFood)}, returned wood ${amount(changes.deliveredWood)}, home HP ${amount(changes.townHealth)} versus unchanged weights. ${regressed ? 'A measured safety outcome regressed.' : improved ? 'The selected goal improved in this comparison.' : 'No improvement in the selected goal was measured.'}`,
  };
}
