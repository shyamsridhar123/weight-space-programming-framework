import { chooseNeural, type PolicyModel } from './policy';
import { World } from './simulation';
import { BUILDING_DEFS, DT, RESOURCES, UNIT_DEFS, type BuildingKind, type Resources, type Team, type UnitKind } from './types';
import { cloneWeightModel, fingerprintWeightModel, validateWeightModel, WEIGHT_LIMITS } from './weight-runtime';

export interface WeightExperimentConfig {
  mutationSeed: number;
  candidates: number;
  developmentSeeds: number[];
  validationSeeds: number[];
  maxSeconds: number;
  decisionSeconds: number;
  mutationScale: number;
  editsPerCandidate: number;
  minFitnessGain: number;
}

export const DEFAULT_WEIGHT_EXPERIMENT_CONFIG: WeightExperimentConfig = Object.freeze({
  mutationSeed: 20261104,
  candidates: 2,
  developmentSeeds: Object.freeze([11, 47]) as unknown as number[],
  validationSeeds: Object.freeze([93, 137]) as unknown as number[],
  maxSeconds: 90,
  decisionSeconds: 2,
  mutationScale: 0.3,
  editsPerCandidate: 4,
  minFitnessGain: 0.01,
});

export const WEIGHT_EXPERIMENT_LIMITS = Object.freeze({
  candidates: 16,
  seedsPerSplit: 8,
  maxSeconds: 600,
  maxDecisionSeconds: 10,
  maxSimulatedSeconds: 19200,
  maxPolicyDecisions: 40000,
});

export const WEIGHT_EXPERIMENT_METHODOLOGY = Object.freeze({
  mutation: 'Seeded W2 output-weight mutation only. Consecutive candidates use opposite perturbations; values are clipped to [-64, 64], with unchanged boundary edits reflected inward.',
  opponent: 'The submitted starting model is frozen for the entire run. Every match starts a fresh simulation; each seed is played with both candidate sides.',
  selection: 'Select once on development by wins-minus-losses per match, then mean shaped fitness, then candidate index. Evaluate only that finalist on the disjoint validation seeds.',
  winner: 'Destroy every opposing Town Center. A match still playing at the time limit is a draw, regardless of shaped fitness.',
  fitness: 'Endpoint candidate-minus-baseline ratios: 30% gathered resources, 10% workers, 10% age, 20% health-adjusted military cost, 10% kills, 10% Town Center health, 5% completed/health-adjusted building cost, 5% stockpiles. Each ratio is (candidate - baseline) / (candidate + baseline + scale); scales are 400, 6, 2, 150, 2, 1800, 350, 400 respectively. The result lies in [-1, 1] and is not a win rate.',
  promotion: 'Both development and held-out mean shaped fitness must be positive and at least minFitnessGain, with wins >= losses in each split and zero illegal commands by either side in all finalist matches. This is a small in-engine smoke criterion, not evidence of strong RTS play.',
  observation: 'Both policies use their ordinary fog-limited candidate and feature APIs. Only the endpoint evaluator reads full simulation state.',
});

export interface WeightMutationEdit {
  tensor: 'w2';
  index: number;
  before: number;
  after: number;
  delta: number;
}

export interface WeightExperimentCandidate {
  id: string;
  index: number;
  model: PolicyModel;
  fingerprint: string;
  edits: WeightMutationEdit[];
}

/** Counts are means in an EvaluationSummary, and exact counts in a match record. */
export interface EvaluationMetrics {
  age: number;
  workers: number;
  army: number;
  militaryValue: number;
  buildings: number;
  buildingValue: number;
  population: number;
  capacity: number;
  townCenterHealth: number;
  kills: number;
  gathered: Resources;
  gatheredTotal: number;
  stockpiles: Resources;
  stockpileTotal: number;
  decisions: number;
  executedCommands: number;
  illegalCommands: number;
  waits: number;
}

export interface ExperimentMatch {
  seed: number;
  candidateTeam: Team;
  baselineTeam: Team;
  outcome: 'win' | 'loss' | 'draw';
  winnerTeam: Team | null;
  terminalStatus: World['status'];
  timeLimitDraw: boolean;
  simulationSeconds: number;
  shapedFitness: number;
  candidateMetrics: EvaluationMetrics;
  baselineMetrics: EvaluationMetrics;
}

export interface EvaluationSummary {
  candidateId: string;
  candidateIndex: number;
  phase: 'development' | 'validation';
  seeds: number[];
  matchCount: number;
  wins: number;
  losses: number;
  draws: number;
  timeLimitDraws: number;
  meanShapedFitness: number;
  terminalBalance: number;
  illegalCommands: number;
  candidateIllegalCommands: number;
  baselineIllegalCommands: number;
  candidateMetrics: EvaluationMetrics;
  baselineMetrics: EvaluationMetrics;
  matches: ExperimentMatch[];
}

export interface WeightExperimentProgress {
  phase: 'development' | 'validation';
  candidateIndex: number;
  candidateCount: number;
  matchesCompleted: number;
  totalMatches: number;
  seed: number;
  candidateTeam: Team;
  simulationSeconds: number;
  maxSeconds: number;
  percent: number;
  bestDevelopmentFitness: number | null;
}

export interface WeightPromotion {
  recommended: boolean;
  reason: string;
  minFitnessGain: number;
  developmentGain: number;
  validationGain: number;
  developmentTerminalBalance: number;
  validationTerminalBalance: number;
  finalistIllegalCommands: number;
}

export interface WeightExperimentResult {
  schemaVersion: 1;
  baseRevision: number;
  baseFingerprint: string;
  config: WeightExperimentConfig;
  methodology: typeof WEIGHT_EXPERIMENT_METHODOLOGY;
  budget: { matches: number; simulatedSeconds: number; policyDecisions: number };
  candidate: WeightExperimentCandidate;
  development: EvaluationSummary[];
  validation: EvaluationSummary;
  promotion: WeightPromotion;
}

export interface RunWeightExperimentOptions {
  baseModel: PolicyModel;
  baseRevision: number;
  config?: Partial<WeightExperimentConfig>;
  signal?: AbortSignal;
  onProgress?: (progress: WeightExperimentProgress) => void;
  /** Must yield to the task queue in a worker so cancel messages can be received. */
  yieldControl?: () => Promise<void>;
}

export class WeightExperimentCancelledError extends Error {
  constructor() { super('Weight experiment cancelled.'); this.name = 'AbortError'; }
}

const emptyResources = (): Resources => ({ food: 0, wood: 0, gold: 0, stone: 0 });
const totalResources = (resources: Partial<Resources>): number => RESOURCES.reduce((sum, kind) => sum + (resources[kind] ?? 0), 0);
const simulationTime = (ticks: number): number => Math.round(ticks * DT * 1e6) / 1e6;
const yieldTask = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 0));
const checkCancelled = (signal?: AbortSignal): void => { if (signal?.aborted) throw new WeightExperimentCancelledError(); };

function integer(value: number, min: number, max: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < min || value > max)
    throw new Error(`${name} must be an integer from ${min} to ${max}.`);
}

function bounded(value: number, min: number, max: number, name: string): void {
  if (!Number.isFinite(value) || value < min || value > max)
    throw new Error(`${name} must be a finite number from ${min} to ${max}.`);
}

function tickDuration(value: number, max: number, name: string): number {
  bounded(value, DT, max, name);
  const ticks = Math.round(value / DT);
  if (Math.abs(ticks * DT - value) > 1e-8) throw new Error(`${name} must be a multiple of ${DT} seconds.`);
  return simulationTime(ticks);
}

function validateSeeds(seeds: number[], name: string): number[] {
  if (!Array.isArray(seeds) || seeds.length < 1 || seeds.length > WEIGHT_EXPERIMENT_LIMITS.seedsPerSplit)
    throw new Error(`${name} must contain 1 to ${WEIGHT_EXPERIMENT_LIMITS.seedsPerSplit} seeds.`);
  for (const seed of seeds) integer(seed, 1, 0xffffffff, `${name} seed`);
  if (new Set(seeds).size !== seeds.length) throw new Error(`${name} cannot contain duplicate seeds.`);
  return [...seeds];
}

export function weightExperimentBudget(config: WeightExperimentConfig): WeightExperimentResult['budget'] {
  const matches = 2 * (config.candidates * config.developmentSeeds.length + config.validationSeeds.length);
  const ticks = Math.round(config.maxSeconds / DT), cadence = Math.round(config.decisionSeconds / DT);
  return { matches, simulatedSeconds: simulationTime(matches * ticks), policyDecisions: matches * 2 * Math.ceil(ticks / cadence) };
}

/** Copy and validate before the first asynchronous yield; caller edits cannot change a running budget. */
export function validateWeightExperimentConfig(input: Partial<WeightExperimentConfig> = {}): WeightExperimentConfig {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Experiment configuration must be an object.');
  for (const key of Object.keys(input)) if (!Object.hasOwn(DEFAULT_WEIGHT_EXPERIMENT_CONFIG, key)) throw new Error(`Unknown experiment setting: ${key}.`);
  const config = { ...DEFAULT_WEIGHT_EXPERIMENT_CONFIG, ...input };
  integer(config.mutationSeed, 0, 0xffffffff, 'mutationSeed');
  integer(config.candidates, 1, WEIGHT_EXPERIMENT_LIMITS.candidates, 'candidates');
  config.developmentSeeds = validateSeeds(config.developmentSeeds, 'developmentSeeds');
  config.validationSeeds = validateSeeds(config.validationSeeds, 'validationSeeds');
  if (config.validationSeeds.some(seed => config.developmentSeeds.includes(seed)))
    throw new Error('Development and validation seeds must be disjoint.');
  config.maxSeconds = tickDuration(config.maxSeconds, WEIGHT_EXPERIMENT_LIMITS.maxSeconds, 'maxSeconds');
  config.decisionSeconds = tickDuration(config.decisionSeconds, WEIGHT_EXPERIMENT_LIMITS.maxDecisionSeconds, 'decisionSeconds');
  bounded(config.mutationScale, 0.001, 2, 'mutationScale');
  integer(config.editsPerCandidate, 1, WEIGHT_LIMITS.hidden, 'editsPerCandidate');
  bounded(config.minFitnessGain, 0, 1, 'minFitnessGain');
  const budget = weightExperimentBudget(config);
  if (budget.simulatedSeconds > WEIGHT_EXPERIMENT_LIMITS.maxSimulatedSeconds)
    throw new Error(`Experiment exceeds the ${WEIGHT_EXPERIMENT_LIMITS.maxSimulatedSeconds} simulated-second budget. Reduce candidates, seeds, or duration.`);
  if (budget.policyDecisions > WEIGHT_EXPERIMENT_LIMITS.maxPolicyDecisions)
    throw new Error(`Experiment exceeds the ${WEIGHT_EXPERIMENT_LIMITS.maxPolicyDecisions} policy-decision budget. Reduce the workload or increase decisionSeconds.`);
  return config;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x100000000;
  };
}

/** Numerical search only: no features, candidate generation, legality checks, or W1 parameters change. */
export function createWeightCandidates(baseModel: PolicyModel, input: Partial<WeightExperimentConfig> = {}): WeightExperimentCandidate[] {
  const base = validateWeightModel(baseModel), config = validateWeightExperimentConfig(input);
  const random = seededRandom(config.mutationSeed), baseId = fingerprintWeightModel(base).slice(-8);
  const candidates: WeightExperimentCandidate[] = [];
  let perturbation: { index: number; delta: number }[] = [];
  for (let index = 0; index < config.candidates; index++) {
    if (index % 2 === 0) {
      const coordinates = Array.from({ length: base.w2.length }, (_, h) => h);
      perturbation = [];
      for (let edit = 0; edit < config.editsPerCandidate; edit++) {
        const picked = edit + Math.floor(random() * (coordinates.length - edit));
        [coordinates[edit], coordinates[picked]] = [coordinates[picked], coordinates[edit]];
        const coordinate = coordinates[edit];
        const delta = (random() < 0.5 ? -1 : 1) * (0.25 + 0.75 * random()) * config.mutationScale * (1 + Math.abs(base.w2[coordinate]));
        perturbation.push({ index: coordinate, delta });
      }
    }
    const model = cloneWeightModel(base), edits: WeightMutationEdit[] = [];
    for (const edit of perturbation) {
      const before = base.w2[edit.index], delta = edit.delta * (index % 2 === 0 ? 1 : -1);
      let after = Math.max(WEIGHT_LIMITS.minimum, Math.min(WEIGHT_LIMITS.maximum, before + delta));
      if (after === before) after = Math.max(WEIGHT_LIMITS.minimum, Math.min(WEIGHT_LIMITS.maximum, before - delta));
      model.w2[edit.index] = after;
      edits.push({ tensor: 'w2', index: edit.index, before, after, delta: after - before });
    }
    candidates.push({ id: `w2-${baseId}-${config.mutationSeed}-${index + 1}`, index, model, fingerprint: fingerprintWeightModel(model), edits });
  }
  return candidates;
}

type DecisionMetrics = Pick<EvaluationMetrics, 'decisions' | 'executedCommands' | 'illegalCommands' | 'waits'>;
const emptyDecisions = (): DecisionMetrics => ({ decisions: 0, executedCommands: 0, illegalCommands: 0, waits: 0 });

function endpointMetrics(world: World, team: Team, decisions: DecisionMetrics): EvaluationMetrics {
  const entities = world.entities.filter(entity => entity.owner === team && entity.hp > 0);
  const army = entities.filter(entity => entity.category === 'unit' && entity.kind !== 'villager');
  const buildings = entities.filter(entity => entity.category === 'building');
  const player = world.players[team];
  return {
    age: player.age, workers: entities.filter(entity => entity.kind === 'villager').length,
    army: army.length,
    militaryValue: army.reduce((sum, entity) => sum + totalResources(UNIT_DEFS[entity.kind as UnitKind].cost) * entity.hp / entity.maxHp, 0),
    buildings: buildings.length,
    buildingValue: buildings.reduce((sum, entity) => sum + totalResources(BUILDING_DEFS[entity.kind as BuildingKind].cost) * entity.construction * entity.hp / entity.maxHp, 0),
    population: world.population(team), capacity: world.capacity(team),
    townCenterHealth: buildings.filter(entity => entity.kind === 'towncenter').reduce((sum, entity) => sum + entity.hp, 0),
    kills: player.kills, gathered: { ...player.gathered }, gatheredTotal: totalResources(player.gathered),
    stockpiles: { ...player.resources }, stockpileTotal: totalResources(player.resources), ...decisions,
  };
}

/** This bounded endpoint score never determines a terminal winner. See the exported methodology. */
export function shapedEndpointFitness(candidate: EvaluationMetrics, baseline: EvaluationMetrics): number {
  const relative = (key: 'gatheredTotal' | 'workers' | 'age' | 'militaryValue' | 'kills' | 'townCenterHealth' | 'buildingValue' | 'stockpileTotal', scale: number) =>
    (candidate[key] - baseline[key]) / (candidate[key] + baseline[key] + scale);
  return Math.max(-1, Math.min(1,
    0.3 * relative('gatheredTotal', 400) + 0.1 * relative('workers', 6) + 0.1 * relative('age', 2)
    + 0.2 * relative('militaryValue', 150) + 0.1 * relative('kills', 2) + 0.1 * relative('townCenterHealth', 1800)
    + 0.05 * relative('buildingValue', 350) + 0.05 * relative('stockpileTotal', 400)));
}

function averageMetrics(metrics: EvaluationMetrics[]): EvaluationMetrics {
  const result: EvaluationMetrics = {
    age: 0, workers: 0, army: 0, militaryValue: 0, buildings: 0, buildingValue: 0, population: 0, capacity: 0,
    townCenterHealth: 0, kills: 0, gathered: emptyResources(), gatheredTotal: 0,
    stockpiles: emptyResources(), stockpileTotal: 0, ...emptyDecisions(),
  };
  for (const values of metrics) for (const key of Object.keys(result) as (keyof EvaluationMetrics)[]) {
    if (key === 'gathered' || key === 'stockpiles') for (const resource of RESOURCES) result[key][resource] += values[key][resource] / metrics.length;
    else result[key] += values[key] / metrics.length;
  }
  return result;
}

function summarizeEvaluation(candidate: WeightExperimentCandidate, phase: EvaluationSummary['phase'], seeds: number[], matches: ExperimentMatch[]): EvaluationSummary {
  const wins = matches.filter(match => match.outcome === 'win').length;
  const losses = matches.filter(match => match.outcome === 'loss').length;
  const candidateIllegalCommands = matches.reduce((sum, match) => sum + match.candidateMetrics.illegalCommands, 0);
  const baselineIllegalCommands = matches.reduce((sum, match) => sum + match.baselineMetrics.illegalCommands, 0);
  return {
    candidateId: candidate.id, candidateIndex: candidate.index, phase, seeds: [...seeds], matchCount: matches.length,
    wins, losses, draws: matches.length - wins - losses, timeLimitDraws: matches.filter(match => match.timeLimitDraw).length,
    meanShapedFitness: matches.reduce((sum, match) => sum + match.shapedFitness, 0) / matches.length,
    terminalBalance: (wins - losses) / matches.length,
    illegalCommands: candidateIllegalCommands + baselineIllegalCommands, candidateIllegalCommands, baselineIllegalCommands,
    candidateMetrics: averageMetrics(matches.map(match => match.candidateMetrics)),
    baselineMetrics: averageMetrics(matches.map(match => match.baselineMetrics)), matches,
  };
}

type PromotionEvaluation = Pick<EvaluationSummary, 'meanShapedFitness' | 'terminalBalance' | 'illegalCommands'>;

export function assessWeightPromotion(development: PromotionEvaluation, validation: PromotionEvaluation, minFitnessGain: number): WeightPromotion {
  bounded(minFitnessGain, 0, 1, 'minFitnessGain');
  for (const [phase, summary] of [['development', development], ['validation', validation]] as const) {
    bounded(summary.meanShapedFitness, -1, 1, `${phase} meanShapedFitness`);
    bounded(summary.terminalBalance, -1, 1, `${phase} terminalBalance`);
    integer(summary.illegalCommands, 0, Number.MAX_SAFE_INTEGER, `${phase} illegalCommands`);
  }
  const finalistIllegalCommands = development.illegalCommands + validation.illegalCommands;
  let reason = 'Meets the configured smoke criterion: positive development and held-out fitness gains, no net match losses in either split, and zero illegal commands.';
  let recommended = false;
  if (finalistIllegalCommands !== 0) reason = 'Do not promote: a finalist match contained an illegal command.';
  else if (development.terminalBalance < 0 || validation.terminalBalance < 0) reason = 'Do not promote: the finalist lost more matches than it won in at least one split.';
  else if (!(development.meanShapedFitness > 0 && development.meanShapedFitness >= minFitnessGain)) reason = 'Do not promote: development fitness did not meet the required positive gain.';
  else if (!(validation.meanShapedFitness > 0 && validation.meanShapedFitness >= minFitnessGain)) reason = 'Do not promote: held-out validation fitness did not meet the required positive gain.';
  else recommended = true;
  return {
    recommended, reason, minFitnessGain, developmentGain: development.meanShapedFitness, validationGain: validation.meanShapedFitness,
    developmentTerminalBalance: development.terminalBalance, validationTerminalBalance: validation.terminalBalance, finalistIllegalCommands,
  };
}

/** A single bounded search run. No incumbent changes, recursive rounds, fitting claims, or partial promotions. */
export async function runWeightExperiment(options: RunWeightExperimentOptions): Promise<WeightExperimentResult> {
  checkCancelled(options.signal);
  integer(options.baseRevision, 0, Number.MAX_SAFE_INTEGER, 'baseRevision');
  const baseRevision = options.baseRevision;
  const baseline = validateWeightModel(options.baseModel);
  const baseFingerprint = fingerprintWeightModel(baseline);
  const config = validateWeightExperimentConfig(options.config);
  const budget = weightExperimentBudget(config);
  const candidates = createWeightCandidates(baseline, config);
  const signal = options.signal, onProgress = options.onProgress, yieldControl = options.yieldControl ?? yieldTask;
  const lastTick = Math.round(config.maxSeconds / DT), decisionTicks = Math.round(config.decisionSeconds / DT);
  const development: EvaluationSummary[] = [];
  let matchesCompleted = 0, bestDevelopmentFitness: number | null = null;

  // Both the baseline and candidate tensors are private copies before any callback or await.
  for (const row of baseline.w1) Object.freeze(row);
  for (const array of [baseline.w1, baseline.b1, baseline.w2, baseline.featureNames, baseline.hiddenNames]) Object.freeze(array);
  Object.freeze(baseline);
  await yieldControl();
  checkCancelled(signal);

  async function evaluate(candidate: WeightExperimentCandidate, phase: EvaluationSummary['phase'], seeds: number[]): Promise<EvaluationSummary> {
    const matches: ExperimentMatch[] = [];
    for (const seed of seeds) for (const candidateTeam of [0, 1] as const) {
      checkCancelled(signal);
      const world = new World(seed), baselineTeam = (1 - candidateTeam) as Team;
      const decisions: [DecisionMetrics, DecisionMetrics] = [emptyDecisions(), emptyDecisions()];
      const progress = (finished = false): void => onProgress?.({
        phase, candidateIndex: candidate.index, candidateCount: candidates.length,
        matchesCompleted, totalMatches: budget.matches, seed, candidateTeam,
        simulationSeconds: simulationTime(world.tick), maxSeconds: config.maxSeconds,
        percent: Math.min(100, 100 * (matchesCompleted + (finished ? 0 : world.tick / lastTick)) / budget.matches), bestDevelopmentFitness,
      });
      progress();
      while (world.status === 'playing' && world.tick < lastTick) {
        checkCancelled(signal);
        if (world.tick % decisionTicks === 0) {
          const first = (Math.floor(world.tick / decisionTicks) % 2) as Team;
          for (const team of [first, 1 - first] as Team[]) {
            const stats = decisions[team];
            const selected = chooseNeural(world, team, team === candidateTeam ? candidate.model : baseline);
            stats.decisions++;
            if (selected.command === null) stats.waits++;
            else if (world.command(team, selected.command).ok) stats.executedCommands++;
            else stats.illegalCommands++;
          }
        }
        world.step(DT);
        if (world.tick % 100 === 0 && world.tick < lastTick && world.status === 'playing') progress();
        if (world.tick % 20 === 0) { await yieldControl(); checkCancelled(signal); }
      }
      const winnerTeam: Team | null = world.status === 'won' ? 0 : world.status === 'lost' ? 1 : null;
      const candidateMetrics = endpointMetrics(world, candidateTeam, decisions[candidateTeam]);
      const baselineMetrics = endpointMetrics(world, baselineTeam, decisions[baselineTeam]);
      matches.push({
        seed, candidateTeam, baselineTeam, outcome: winnerTeam === null ? 'draw' : winnerTeam === candidateTeam ? 'win' : 'loss',
        winnerTeam, terminalStatus: world.status, timeLimitDraw: world.status === 'playing', simulationSeconds: simulationTime(world.tick),
        shapedFitness: shapedEndpointFitness(candidateMetrics, baselineMetrics), candidateMetrics, baselineMetrics,
      });
      matchesCompleted++;
      progress(true);
      await yieldControl();
      checkCancelled(signal);
    }
    return summarizeEvaluation(candidate, phase, seeds, matches);
  }

  for (const candidate of candidates) {
    const summary = await evaluate(candidate, 'development', config.developmentSeeds);
    development.push(summary);
    bestDevelopmentFitness = Math.max(bestDevelopmentFitness ?? -Infinity, summary.meanShapedFitness);
  }
  const selected = [...development].sort((a, b) => b.terminalBalance - a.terminalBalance || b.meanShapedFitness - a.meanShapedFitness || a.candidateIndex - b.candidateIndex)[0];
  const candidate = candidates[selected.candidateIndex];
  const validation = await evaluate(candidate, 'validation', config.validationSeeds);
  checkCancelled(signal);
  return {
    schemaVersion: 1, baseRevision, baseFingerprint, config, methodology: WEIGHT_EXPERIMENT_METHODOLOGY, budget,
    candidate, development, validation, promotion: assessWeightPromotion(selected, validation, config.minFitnessGain),
  };
}
