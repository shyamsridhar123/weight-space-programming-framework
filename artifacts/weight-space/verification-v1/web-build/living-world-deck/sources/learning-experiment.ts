import type { PolicyModel } from '../empires/policy';
import {
  fingerprintWorldModel, rankWorldActions, validateWorldModel, worldPolicyContext,
  type WorldPolicyContext,
} from './policy';
import { LivingWorld, WORLD_LIMITS, WORLD_STEP } from './simulation';
import { CHUNK_SIZE } from './terrain';
import { WORLD_ACTIONS, type ActionScore, type Region, type WorldAction, type WorldSave } from './types';

export const LEARNING_HORIZONS = [1, 3, 5, 10] as const;
export type LearningHorizon = typeof LEARNING_HORIZONS[number];
export const LEARNING_TICKS_PER_YEAR = 48 / WORLD_STEP;
const SAMPLE_INTERVAL = LEARNING_TICKS_PER_YEAR / 4;
const MAX_TICK = 1_000_000_000_000;
const MAX_BATCH = 192;

export interface LearningExperimentRequest {
  snapshot: WorldSave;
  civId: number;
  /** Must be the model actually present in snapshot, including its metadata. */
  baselineModel: PolicyModel;
  candidateModel: PolicyModel;
  years: LearningHorizon;
}

export interface LearningMetrics {
  /** All people belonging to this society, including mobile groups. */
  population: number;
  settledPopulation: number;
  travelingPopulation: number;
  towns: number;
  /** Food in settlement stores plus the payload of this society's mobile groups. */
  food: number;
  storedFood: number;
  travelingFood: number;
  /** Null if the society's record has retired; zero would be a different observation. */
  knowledge: number | null;
  wealth: number | null;
  farms: number;
  health: number | null;
  cohesion: number | null;
  /** Mean cover/fertility and wildlife totals over the original occupied-region cohort. */
  forest: number | null;
  fertility: number | null;
  wildlife: number | null;
  herbivores: number | null;
  predators: number | null;
  wars: number;
  tradePartners: number;
  worldPopulation: number;
  worldSocieties: number;
  worldTowns: number;
  worldWars: number;
}

export const LEARNING_METRIC_KEYS = [
  'population', 'settledPopulation', 'travelingPopulation', 'towns', 'food', 'storedFood',
  'travelingFood', 'knowledge', 'wealth', 'farms', 'health', 'cohesion', 'forest', 'fertility',
  'wildlife', 'herbivores', 'predators', 'wars', 'tradePartners', 'worldPopulation',
  'worldSocieties', 'worldTowns', 'worldWars',
] as const satisfies readonly (keyof LearningMetrics)[];

export interface LearningObservation extends LearningMetrics {
  tick: number;
  elapsedYears: number;
  retained: boolean;
  alive: boolean;
  cohort: {
    expected: number;
    /** Original region objects still continuously retained, excluding regenerated ones. */
    retained: number;
    complete: boolean;
    missingRegionKeys: string[];
    replacedRegionKeys: string[];
    /** Cumulative loss: recovering the same coordinate cannot recover retired ecology. */
    lostRegionKeys: string[];
  };
}

export type LearningCap = 'regions' | 'civilizations' | 'settlements' | 'groups';
export interface LearningBranchResult {
  initial: LearningObservation;
  final: LearningObservation;
  /** Scored on the identical frozen context, before either branch steps. */
  probabilities: ActionScore[];
  /** Scheduled focus selections, not previews and not guaranteed successful effects. */
  actionCounts: Record<WorldAction, number>;
  trajectory: LearningObservation[];
  warnings: string[];
  firstExtinctTick: number | null;
  firstRetiredTick: number | null;
  capFirstTick: Record<LearningCap, number | null>;
}

export interface LearningExperimentResult {
  format: 'crown-charter-learning-experiment';
  version: 1;
  civId: number;
  civName: string;
  context: WorldPolicyContext;
  cohortRegionKeys: string[];
  reproducibility: {
    seed: number;
    startTick: number;
    endTick: number;
    years: LearningHorizon;
    ticksPerYear: number;
    sampleEveryTicks: number;
    initialRngState: number;
    /** Content fingerprints are comparison aids, not cryptographic signatures. */
    checkpointFingerprint: string;
    baselineFingerprint: string;
    candidateFingerprint: string;
    changedParameters: number;
  };
  baseline: LearningBranchResult;
  candidate: LearningBranchResult;
  /** Candidate final value minus baseline final value; null means unavailable comparison. */
  differences: { [K in keyof LearningMetrics]: number | null };
  notes: string[];
}

export interface LearningExperimentProgress {
  completedTicks: number;
  totalTicks: number;
  fraction: number;
  elapsedYears: number;
  done: boolean;
}

export interface LearningExperimentRunner {
  /** Advances at most this many ticks in EACH branch; accepted range 1–192. */
  step(maxTicks?: number): LearningExperimentProgress;
  readonly progress: LearningExperimentProgress;
  /** Returns a detached result after completion, otherwise null. */
  readonly result: LearningExperimentResult | null;
}

export interface LearningWorkerRequest { type: 'run'; id: number; request: LearningExperimentRequest }
export type LearningWorkerMessage =
  | { type: 'progress'; id: number; progress: LearningExperimentProgress }
  | { type: 'result'; id: number; result: LearningExperimentResult }
  | { type: 'error'; id: number; error: string };

interface Branch {
  world: LivingWorld;
  cohort: Map<string, Region>;
  lost: Set<string>;
  result: LearningBranchResult;
}

const keyAt = (point: { x: number; z: number }): string => `${Math.floor(point.x / CHUNK_SIZE)},${Math.floor(point.z / CHUNK_SIZE)}`;
const actionCounts = (): Record<WorldAction, number> => Object.fromEntries(WORLD_ACTIONS.map(action => [action, 0])) as Record<WorldAction, number>;
const capKeys: readonly LearningCap[] = ['regions', 'civilizations', 'settlements', 'groups'];

function validateEnvelope(input: unknown): asserts input is LearningExperimentRequest {
  if (!input || typeof input !== 'object' || Array.isArray(input) || ![Object.prototype, null].includes(Object.getPrototypeOf(input)))
    throw new Error('Experiment request must be a plain object.');
  const keys = ['snapshot', 'civId', 'baselineModel', 'candidateModel', 'years'];
  if (keys.some(key => !Object.hasOwn(input, key))) throw new Error('Experiment request is missing required fields.');
  for (const key of Reflect.ownKeys(input)) {
    if (typeof key !== 'string' || !keys.includes(key)) throw new Error('Experiment request contains an unknown field.');
    if (!Object.hasOwn(Object.getOwnPropertyDescriptor(input, key)!, 'value')) throw new Error('Experiment request cannot contain accessors.');
  }
  const value = input as LearningExperimentRequest;
  if (!Number.isSafeInteger(value.civId) || value.civId < 1) throw new Error('Select a valid civilization for the experiment.');
  if (!LEARNING_HORIZONS.includes(value.years)) throw new Error('Experiment horizon must be 1, 3, 5, or 10 years.');
}

/** Small reproducibility checksum. Model identity uses the existing 64-bit model fingerprint. */
function checkpointFingerprint(save: WorldSave): string {
  const content = JSON.stringify(save);
  let hash = 2166136261;
  for (let index = 0; index < content.length; index++) hash = Math.imul(hash ^ content.charCodeAt(index), 16777619);
  return `fnv1a32_${(hash >>> 0).toString(16).padStart(8, '0')}`;
}

function changedParameters(a: PolicyModel, b: PolicyModel): number {
  const first = [...a.w1.flat(), ...a.b1, ...a.w2, a.b2, a.temperature];
  const second = [...b.w1.flat(), ...b.b1, ...b.w2, b.b2, b.temperature];
  return first.reduce((sum, value, index) => sum + Number(value !== second[index]), 0);
}

function observation(branch: Branch, civId: number, startTick: number): LearningObservation {
  const { world, cohort, lost } = branch;
  const civ = world.civilizations.find(entry => entry.id === civId);
  const towns = world.settlements.filter(town => town.civId === civId && town.population > 0);
  const groups = world.groups.filter(group => group.civId === civId);
  const settledPopulation = towns.reduce((sum, town) => sum + town.population, 0);
  const travelingPopulation = groups.reduce((sum, group) => sum + group.members, 0);
  const storedFood = towns.reduce((sum, town) => sum + town.food, 0);
  const travelingFood = groups.reduce((sum, group) => sum + group.payload, 0);
  const missingRegionKeys: string[] = [], replacedRegionKeys: string[] = [];
  const original: Region[] = [];
  for (const [key, region] of cohort) {
    const current = world.regions.get(key);
    if (!current) missingRegionKeys.push(key);
    else if (current !== region) replacedRegionKeys.push(key);
    if (current === region && !lost.has(key)) original.push(region);
  }
  const complete = cohort.size > 0 && lost.size === 0 && original.length === cohort.size;
  const ecologicalMean = (key: 'forest' | 'fertility'): number | null => complete ? original.reduce((sum, region) => sum + region[key], 0) / cohort.size : null;
  const ecologicalSum = (key: 'herbivores' | 'predators'): number | null => complete ? original.reduce((sum, region) => sum + region[key], 0) : null;
  const herbivores = ecologicalSum('herbivores'), predators = ecologicalSum('predators');
  const relations = world.relations.filter(relation => relation.a === civId || relation.b === civId);
  return {
    tick: world.tick, elapsedYears: (world.tick - startTick) / LEARNING_TICKS_PER_YEAR,
    retained: Boolean(civ), alive: Boolean(civ?.alive),
    population: settledPopulation + travelingPopulation, settledPopulation, travelingPopulation,
    towns: towns.length, food: storedFood + travelingFood, storedFood, travelingFood,
    knowledge: civ?.knowledge ?? null,
    wealth: civ ? civ.treasury + towns.reduce((sum, town) => sum + town.wealth, 0) : null,
    farms: towns.reduce((sum, town) => sum + town.farms, 0),
    health: settledPopulation > 0 ? towns.reduce((sum, town) => sum + town.health * town.population, 0) / settledPopulation : null,
    cohesion: civ?.cohesion ?? null, forest: ecologicalMean('forest'), fertility: ecologicalMean('fertility'),
    wildlife: herbivores === null || predators === null ? null : herbivores + predators, herbivores, predators,
    wars: relations.filter(relation => relation.status === 'war').length,
    tradePartners: relations.filter(relation => relation.status === 'trade' || relation.status === 'alliance').length,
    worldPopulation: world.settlements.reduce((sum, town) => sum + town.population, 0) + world.groups.reduce((sum, group) => sum + group.members, 0),
    worldSocieties: world.civilizations.filter(entry => entry.alive).length,
    worldTowns: world.settlements.filter(town => town.population > 0).length,
    worldWars: world.relations.filter(relation => relation.status === 'war').length,
    cohort: { expected: cohort.size, retained: original.length, complete, missingRegionKeys, replacedRegionKeys, lostRegionKeys: [...lost] },
  };
}

/** Inspect every step so a retired/recreated region between chart samples cannot go unnoticed. */
function trackBranch(branch: Branch, civId: number, countDecision: boolean): void {
  const { world, result } = branch;
  const civ = world.civilizations.find(entry => entry.id === civId);
  if (!civ && result.firstRetiredTick === null) result.firstRetiredTick = world.tick;
  if (!civ?.alive && result.firstExtinctTick === null) result.firstExtinctTick = world.tick;
  if (countDecision && world.tick % 16 === 0 && civ?.decision?.tick === world.tick)
    result.actionCounts[civ.decision.action]++;
  for (const [key, region] of branch.cohort) if (world.regions.get(key) !== region) branch.lost.add(key);
  const sizes: Record<LearningCap, number> = {
    regions: world.regions.size, civilizations: world.civilizations.length,
    settlements: world.settlements.length, groups: world.groups.length,
  };
  for (const cap of capKeys) if (sizes[cap] >= WORLD_LIMITS[cap] && result.capFirstTick[cap] === null) result.capFirstTick[cap] = world.tick;
}

function prepareBranch(world: LivingWorld, model: PolicyModel, civId: number, context: WorldPolicyContext, cohortKeys: string[]): Branch {
  // Both branches get the exact same bookkeeping edit. No RNG draw, timestep, focus choice,
  // or initial ecology differs because of experiment setup; only the assigned tensor can differ.
  world.applyModel(civId, model, 'Learning experiment policy');
  const branch: Branch = {
    world, cohort: new Map(cohortKeys.map(key => [key, world.regions.get(key)!])), lost: new Set(),
    result: {
      initial: undefined as unknown as LearningObservation, final: undefined as unknown as LearningObservation,
      probabilities: rankWorldActions(context, model), actionCounts: actionCounts(), trajectory: [], warnings: [],
      firstExtinctTick: null, firstRetiredTick: null,
      capFirstTick: { regions: null, civilizations: null, settlements: null, groups: null },
    },
  };
  trackBranch(branch, civId, false);
  branch.result.initial = observation(branch, civId, world.tick);
  branch.result.final = branch.result.initial;
  branch.result.trajectory.push(branch.result.initial);
  return branch;
}

function finishBranch(branch: Branch, civId: number, startTick: number): LearningBranchResult {
  const { result } = branch;
  result.final = observation(branch, civId, startTick);
  if (result.trajectory.at(-1)?.tick !== result.final.tick) result.trajectory.push(result.final);
  if (result.firstExtinctTick !== null) result.warnings.push(`The selected society had no surviving population at tick ${result.firstExtinctTick}; its loss is included in these results.`);
  if (result.firstRetiredTick !== null) result.warnings.push(`The selected society's record retired at tick ${result.firstRetiredTick}; unavailable society metrics are null.`);
  for (const cap of capKeys) if (result.capFirstTick[cap] !== null)
    result.warnings.push(`The world reached the ${WORLD_LIMITS[cap]} ${cap} retention cap at tick ${result.capFirstTick[cap]}; this limit can constrain later outcomes.`);
  if (branch.cohort.size === 0) result.warnings.push('This society occupied no ecological regions at the checkpoint; local ecology metrics are unavailable.');
  if (branch.lost.size) result.warnings.push(`${branch.lost.size} original occupied ecological region(s) retired or regenerated. Cohort metrics after loss are unavailable, not replaced with fresh terrain.`);
  return result;
}

/**
 * A paired, deterministic counterfactual, using two detached copies of one validated save.
 * The baseline is the snapshot's actual model. Other societies retain their original models.
 * Step budgets affect responsiveness only; they never change simulation time or sampling.
 */
export function createLearningExperiment(input: LearningExperimentRequest): LearningExperimentRunner {
  validateEnvelope(input);
  const baselineModel = validateWorldModel(input.baselineModel), candidateModel = validateWorldModel(input.candidateModel);
  const baselineWorld = LivingWorld.load(input.snapshot);
  const snapshot = baselineWorld.save();
  const { civId, years } = input, startTick = snapshot.tick, totalTicks = years * LEARNING_TICKS_PER_YEAR;
  if (startTick > MAX_TICK - totalTicks) throw new Error('Experiment would exceed the supported world simulation clock.');
  if (snapshot.nextId > Number.MAX_SAFE_INTEGER - 1_000_000) throw new Error('Experiment checkpoint has insufficient entity ID headroom.');
  const civ = baselineWorld.civilizations.find(entry => entry.id === civId);
  if (!civ) throw new Error('Selected civilization is absent from the frozen checkpoint.');
  if (civ.mind.revision === Number.MAX_SAFE_INTEGER) throw new Error('The selected mind has reached its supported revision limit.');
  if (JSON.stringify(baselineModel) !== JSON.stringify(civ.mind.model)) throw new Error('Baseline model must match the selected civilization in the frozen checkpoint.');
  const context = worldPolicyContext(baselineWorld, civ);
  const cohortRegionKeys = [...new Set([
    ...baselineWorld.settlements.filter(town => town.civId === civId && town.population > 0).map(keyAt),
    ...baselineWorld.groups.filter(group => group.civId === civId && group.members > 0).map(keyAt),
  ])].sort();
  const baseline = prepareBranch(baselineWorld, baselineModel, civId, context, cohortRegionKeys);
  const candidate = prepareBranch(LivingWorld.load(snapshot), candidateModel, civId, context, cohortRegionKeys);
  const reproducibility: LearningExperimentResult['reproducibility'] = {
    seed: snapshot.seed, startTick, endTick: startTick + totalTicks, years, ticksPerYear: LEARNING_TICKS_PER_YEAR,
    sampleEveryTicks: SAMPLE_INTERVAL, initialRngState: snapshot.rngState,
    checkpointFingerprint: checkpointFingerprint(snapshot),
    baselineFingerprint: fingerprintWorldModel(baselineModel), candidateFingerprint: fingerprintWorldModel(candidateModel),
    changedParameters: changedParameters(baselineModel, candidateModel),
  };
  let completedTicks = 0, finished: LearningExperimentResult | null = null;
  const progress = (): LearningExperimentProgress => ({
    completedTicks, totalTicks, fraction: completedTicks / totalTicks,
    elapsedYears: completedTicks / LEARNING_TICKS_PER_YEAR, done: completedTicks === totalTicks,
  });
  return {
    get progress() { return progress(); },
    get result() { return finished === null ? null : structuredClone(finished); },
    step(maxTicks = 32): LearningExperimentProgress {
      if (!Number.isInteger(maxTicks) || maxTicks < 1 || maxTicks > MAX_BATCH) throw new Error(`Experiment step budget must be an integer from 1 to ${MAX_BATCH}.`);
      const end = Math.min(totalTicks, completedTicks + maxTicks);
      while (completedTicks < end) {
        for (const branch of [baseline, candidate]) { branch.world.step(); trackBranch(branch, civId, true); }
        completedTicks++;
        if (completedTicks % SAMPLE_INTERVAL === 0 || completedTicks === totalTicks)
          for (const branch of [baseline, candidate]) branch.result.trajectory.push(observation(branch, civId, startTick));
      }
      if (completedTicks === totalTicks && !finished) {
        const original = finishBranch(baseline, civId, startTick), changed = finishBranch(candidate, civId, startTick);
        const differences = Object.fromEntries(LEARNING_METRIC_KEYS.map(key => [key,
          original.final[key] === null || changed.final[key] === null ? null : changed.final[key]! - original.final[key]!,
        ])) as LearningExperimentResult['differences'];
        finished = {
          format: 'crown-charter-learning-experiment', version: 1, civId, civName: civ.name,
          context, cohortRegionKeys, reproducibility, baseline: original, candidate: changed, differences,
          notes: [
            'Both branches start from one frozen world and RNG state. Only the selected civilization model is changed; the live game is untouched.',
            'Action counts are scheduled focus choices, not guaranteed successful actions. Resource limits can prevent a chosen action from taking effect.',
            'Probabilities are evaluated on the same starting context. The existing focus remains until the next normal decision, every 16 ticks.',
            'The same initial RNG does not synchronize later random draws after choices diverge. This is one paired scenario, not an average causal effect across seeds.',
            'Quarter-year samples show the selected society, excluding independent descendants. Local ecology uses the fixed original occupied-region cohort.',
            'The policy is an authored neural scorer. This experiment performs no training and makes no native Laya call.',
          ],
        };
      }
      return progress();
    },
  };
}
