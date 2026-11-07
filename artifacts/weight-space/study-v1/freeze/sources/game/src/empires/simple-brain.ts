import { createDefaultModel, policyFeatures, tracePolicy, type Candidate, type PolicyModel } from './policy';
import type { World } from './simulation';

export type SimpleWeightId = 'gathering' | 'recruitment' | 'aggression';

export interface SimpleWeightControl {
  readonly id: SimpleWeightId;
  readonly label: string;
  readonly hint: string;
  readonly tensor: 'w2';
  readonly hiddenName: string;
  readonly index: number;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly defaultValue: number;
}

const authored = createDefaultModel();
const controls = [
  { id: 'gathering', label: 'Gather resources', hiddenName: 'labor_coverage_gap',
    hint: 'Scores gathering orders when a resource has too few workers.' },
  { id: 'recruitment', label: 'Recruit soldiers', hiddenName: 'army_train_bias',
    hint: 'Adds to the score of each affordable soldier recruitment order.' },
  { id: 'aggression', label: 'Attack your town', hiddenName: 'attack_bias',
    hint: 'Adds to the score of an available attack order. Mission timing still applies.' },
] as const;

/** These sliders write three literal output coefficients of the authored MLP. */
export const SIMPLE_WEIGHT_CONTROLS: readonly SimpleWeightControl[] = Object.freeze(controls.map(control => {
  const index = authored.hiddenNames.indexOf(control.hiddenName);
  if (index < 0) throw new Error(`Missing commander neuron: ${control.hiddenName}`);
  return Object.freeze({ ...control, tensor: 'w2' as const, index,
    min: -12, max: 12, step: 0.1, defaultValue: authored.w2[index] });
}));

export const SIMPLE_BRAIN_DESCRIPTION = 'The rival uses a small authored, untrained neural network. These controls edit its actual weights; they do not train it.';
export const SIMPLE_DECISION_NOTE = 'Default and edited scores use the same current situation and available orders. Scores are preferences, not chances of success. Higher scores cannot bypass resource costs or mission rules.';

export interface SimpleBrain {
  /** Stable model identity: the rival reads these live coefficients each decision. */
  readonly model: PolicyModel;
  /** An independent frozen checkpoint for same-state comparisons and reset. */
  readonly baseline: PolicyModel;
  revision: number;
}

function cloneModel(model: PolicyModel): PolicyModel {
  return { ...model, featureNames: [...model.featureNames], hiddenNames: [...model.hiddenNames],
    w1: model.w1.map(row => [...row]), b1: [...model.b1], w2: [...model.w2] };
}

function freezeModel(model: PolicyModel): PolicyModel {
  for (const row of model.w1) Object.freeze(row);
  for (const values of [model.featureNames, model.hiddenNames, model.w1, model.b1, model.w2]) Object.freeze(values);
  return Object.freeze(model);
}

export function createSimpleBrain(): SimpleBrain {
  return { model: createDefaultModel(), baseline: freezeModel(createDefaultModel()), revision: 0 };
}

function controlById(id: SimpleWeightId): SimpleWeightControl {
  const control = SIMPLE_WEIGHT_CONTROLS.find(candidate => candidate.id === id);
  if (!control) throw new Error(`Unknown rival weight: ${id}`);
  return control;
}

export function simpleWeightValue(brain: SimpleBrain, id: SimpleWeightId): number {
  return brain.model.w2[controlById(id).index];
}

/** Apply immediately; finite values are bounded and quantized to the visible slider step. */
export function setSimpleWeight(brain: SimpleBrain, id: SimpleWeightId, value: number): boolean {
  const control = controlById(id);
  if (!Number.isFinite(value)) throw new RangeError('Rival weights must be finite numbers.');
  const next = Number(Math.max(control.min, Math.min(control.max, value)).toFixed(1));
  if (brain.model.w2[control.index] === next) return false;
  brain.model.w2[control.index] = next;
  brain.revision++;
  return true;
}

/** Restore all authored parameters without invalidating a rival's model reference. */
export function resetSimpleBrain(brain: SimpleBrain): void {
  Object.assign(brain.model, cloneModel(brain.baseline));
  brain.revision++;
}

export interface SimpleWeightEffect {
  id: SimpleWeightId;
  hiddenName: string;
  index: number;
  activation: number;
  baselineWeight: number;
  weight: number;
  baselineContribution: number;
  contribution: number;
}

export interface SimpleScoredCandidate extends Candidate {
  baselineScore: number;
  score: number;
  delta: number;
  effects: SimpleWeightEffect[];
}

export interface SimpleDecisionTrace {
  team: 1;
  tick: number;
  time: number;
  revision: number;
  selected: SimpleScoredCandidate | null;
  baselineSelected: SimpleScoredCandidate | null;
  orderChanged: boolean;
  /** Only the legal, mission-permitted candidates supplied by the rival controller. */
  candidates: SimpleScoredCandidate[];
}

const byId = (a: Candidate, b: Candidate): number => a.id < b.id ? -1 : a.id > b.id ? 1 : 0;

/**
 * Read-only evidence for the rival (team 1), never an automatic player controller.
 * Pass getMissionRivalCandidates(world) so the lesson and the actual rival see the
 * same choices. An unavailable order is never invented just to show an effect.
 */
export function snapshotSimpleDecision(world: World, brain: SimpleBrain, available: readonly Candidate[]): SimpleDecisionTrace {
  const candidates = available.filter(candidate => !candidate.command || world.canCommand(1, candidate.command).ok).map(candidate => {
    const input = policyFeatures(world, 1, candidate);
    const baseline = tracePolicy(input, brain.baseline), live = tracePolicy(input, brain.model);
    return {
      id: candidate.id, label: candidate.label, description: candidate.description,
      command: candidate.command ? structuredClone(candidate.command) : null,
      baselineScore: baseline.score, score: live.score, delta: live.score - baseline.score,
      effects: SIMPLE_WEIGHT_CONTROLS.map(control => ({
        id: control.id, hiddenName: control.hiddenName, index: control.index,
        activation: live.hiddenActivations[control.index], baselineWeight: brain.baseline.w2[control.index],
        weight: brain.model.w2[control.index], baselineContribution: baseline.contributions[control.index],
        contribution: live.contributions[control.index],
      })),
    };
  }).sort((a, b) => b.score - a.score || byId(a, b));
  const selected = candidates[0] ?? null;
  const baselineSelected = [...candidates].sort((a, b) => b.baselineScore - a.baselineScore || byId(a, b))[0] ?? null;
  return { team: 1, tick: world.tick, time: world.time, revision: brain.revision, selected, baselineSelected,
    orderChanged: selected?.id !== baselineSelected?.id, candidates };
}
