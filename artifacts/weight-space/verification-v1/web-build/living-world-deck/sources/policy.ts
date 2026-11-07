import { forwardPolicy, tracePolicy, type PolicyModel, type PolicyTrace } from '../empires/policy';
import {
  WORLD_ACTIONS, type ActionScore, type Civilization, type MobileGroup, type Region,
  type Relation, type Settlement, type WorldAction, type WorldMind,
} from './types';

export const WORLD_POLICY_LIMITS = Object.freeze({
  inputs: 32, hidden: 38, parameters: 1293, minimum: -64, maximum: 64,
  minimumTemperature: 0.05, maximumTemperature: 20, maximumModelBytes: 250_000,
});

/** Every state feature is a local, normalized quantity in [0, 1]. */
export interface WorldPolicyContext {
  foodNeed: number; fertility: number; forest: number; pressure: number;
  health: number; unrest: number; cohesion: number; wealth: number;
  knowledge: number; development: number; crowding: number; tradeOpportunity: number;
  threat: number; militaryStrength: number; frontierRoom: number; migrationNeed: number;
}

export interface WorldPolicySource {
  tick: number;
  settlements: readonly Settlement[];
  civilizations: readonly Civilization[];
  groups: readonly MobileGroup[];
  relations: readonly Relation[];
  regions: ReadonlyMap<string, Region> | readonly Region[];
}

const CONTEXT_KEYS: readonly (keyof WorldPolicyContext)[] = Object.freeze([
  'foodNeed', 'fertility', 'forest', 'pressure', 'health', 'unrest', 'cohesion', 'wealth',
  'knowledge', 'development', 'crowding', 'tradeOpportunity', 'threat', 'militaryStrength', 'frontierRoom', 'migrationNeed',
]);
export const WORLD_FEATURE_NAMES: readonly string[] = Object.freeze([
  'food_need', 'local_fertility', 'forest_cover', 'land_pressure', 'public_health', 'unrest', 'cohesion', 'wealth_per_person',
  'knowledge', 'age_development', 'settlement_crowding', 'trade_opportunity', 'military_threat', 'army_strength', 'frontier_room', 'migration_need',
  ...WORLD_ACTIONS.map(action => `action_${action}`),
  ...Array.from({ length: 8 }, (_, index) => `reserved_${index + 1}`),
]);

interface StateNeuron {
  name: string; action: WorldAction; feature: keyof WorldPolicyContext; weight: number; complement?: boolean;
}
const STATE_NEURONS: readonly StateNeuron[] = [
  { name: 'cultivate_food_need', action: 'cultivate', feature: 'foodNeed', weight: 6 },
  { name: 'cultivate_fertile_land', action: 'cultivate', feature: 'fertility', weight: 1.2 },
  { name: 'cultivate_land_pressure', action: 'cultivate', feature: 'pressure', weight: -3 },
  { name: 'expand_crowding', action: 'expand', feature: 'crowding', weight: 4.2 },
  { name: 'expand_frontier', action: 'expand', feature: 'frontierRoom', weight: 2 },
  { name: 'expand_wealth', action: 'expand', feature: 'wealth', weight: 1 },
  { name: 'expand_food_need', action: 'expand', feature: 'foodNeed', weight: -5 },
  { name: 'conserve_land_pressure', action: 'conserve', feature: 'pressure', weight: 5.5 },
  { name: 'conserve_forest_loss', action: 'conserve', feature: 'forest', weight: 2, complement: true },
  { name: 'conserve_food_need', action: 'conserve', feature: 'foodNeed', weight: -1.5 },
  { name: 'trade_opportunities', action: 'trade', feature: 'tradeOpportunity', weight: 2.6 },
  { name: 'trade_wealth_need', action: 'trade', feature: 'wealth', weight: 1.7, complement: true },
  { name: 'trade_war_risk', action: 'trade', feature: 'threat', weight: -3 },
  { name: 'trade_cohesion', action: 'trade', feature: 'cohesion', weight: 0.8 },
  { name: 'research_wealth', action: 'research', feature: 'wealth', weight: 2.1 },
  { name: 'research_food_need', action: 'research', feature: 'foodNeed', weight: -3.5 },
  { name: 'research_knowledge_need', action: 'research', feature: 'knowledge', weight: 0.8, complement: true },
  { name: 'fortify_threat', action: 'fortify', feature: 'threat', weight: 5.2 },
  { name: 'fortify_army_need', action: 'fortify', feature: 'militaryStrength', weight: 0.8, complement: true },
  { name: 'fortify_unrest', action: 'fortify', feature: 'unrest', weight: 2 },
  { name: 'raid_army_strength', action: 'raid', feature: 'militaryStrength', weight: 4 },
  { name: 'raid_war_pressure', action: 'raid', feature: 'threat', weight: 1.2 },
  { name: 'raid_cohesion', action: 'raid', feature: 'cohesion', weight: 1 },
  { name: 'raid_food_need', action: 'raid', feature: 'foodNeed', weight: -3 },
  { name: 'migrate_displacement', action: 'migrate', feature: 'migrationNeed', weight: 5.2 },
  { name: 'migrate_land_pressure', action: 'migrate', feature: 'pressure', weight: 2 },
  { name: 'migrate_food_need', action: 'migrate', feature: 'foodNeed', weight: 1.4 },
  { name: 'migrate_cohesion', action: 'migrate', feature: 'cohesion', weight: -1.2 },
  { name: 'cultivate_health_need', action: 'cultivate', feature: 'health', weight: 2, complement: true },
  { name: 'conserve_war_risk', action: 'conserve', feature: 'threat', weight: -1 },
];
export const WORLD_HIDDEN_NAMES: readonly string[] = Object.freeze([
  ...WORLD_ACTIONS.map(action => `priority_${action}`), ...STATE_NEURONS.map(neuron => neuron.name),
]);

export interface WorldWeightControl {
  id: WorldAction; action: WorldAction; tensor: 'w2'; row: number;
  label: string; description: string; min: number; max: number; step: number;
}
const ACTION_LABELS: Record<WorldAction, string> = {
  cultivate: 'Cultivation', expand: 'Expansion', conserve: 'Conservation', trade: 'Trade',
  research: 'Knowledge', fortify: 'Defense', raid: 'Raiding', migrate: 'Migration',
};
const ACTION_DESCRIPTIONS: Record<WorldAction, string> = {
  cultivate: 'Favor food production and farms.', expand: 'Favor founding additional settlements.',
  conserve: 'Favor restoring soil and forests.', trade: 'Favor commerce and peaceful exchange.',
  research: 'Favor discovery and advancement.', fortify: 'Favor defensive preparation.',
  raid: 'Favor military expeditions.', migrate: 'Favor moving people away from stressed land.',
};
export const WORLD_WEIGHT_CONTROLS: readonly WorldWeightControl[] = Object.freeze(WORLD_ACTIONS.map((action, row) => Object.freeze({
  id: action, action, tensor: 'w2' as const, row, label: ACTION_LABELS[action], description: ACTION_DESCRIPTIONS[action],
  min: -6, max: 8, step: 0.1,
})));

export interface WorldPreset {
  id: string; name: string; description: string; priorities: Readonly<Record<WorldAction, number>>; temperature: number;
}
function preset(id: string, name: string, description: string, values: readonly number[], temperature = 1.35): WorldPreset {
  return Object.freeze({ id, name, description, temperature,
    priorities: Object.freeze(Object.fromEntries(WORLD_ACTIONS.map((action, index) => [action, values[index]])) as Record<WorldAction, number>) });
}
export const WORLD_PRESETS: readonly WorldPreset[] = Object.freeze([
  preset('growth', 'Agrarian growth', 'Build food reserves and settle new land.', [1.5, 2.2, 0, 0.2, 0, 0, -0.5, 0]),
  preset('trade', 'Merchant cities', 'Favor exchange, wealth, and stable settlements.', [0.4, 0.4, 0.2, 2.8, 0.7, 0, -1, 0]),
  preset('conservation', 'Forest stewards', 'Favor ecological recovery and restrained expansion.', [0.5, -0.5, 2.7, 0.5, 0.4, 0, -1.5, 0.4]),
  preset('military', 'Martial frontier', 'Favor armies, fortifications, and expeditions.', [0.5, 0.6, -0.5, -0.4, 0, 1.8, 2.5, -0.4]),
  preset('knowledge', 'Scholarly cities', 'Favor research supported by food and commerce.', [0.4, 0.2, 0.4, 0.7, 2.8, 0, -1.2, 0]),
  preset('balanced', 'Balanced council', 'Let changing local needs dominate action priorities.', [0.6, 0.4, 0.3, 0.3, 0.3, 0.1, -0.6, 0]),
]);

const clamp = (value: number): number => Math.max(0, Math.min(1, value));
const distance = (a: { x: number; z: number }, b: { x: number; z: number }): number => Math.hypot(a.x - b.x, a.z - b.z);
const round = (value: number): number => Math.round(value * 1000) / 1000;

function seedNumber(value: string | number): number {
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value)) throw new Error('World policy seed must be a safe integer');
    return Math.abs(value);
  }
  let hash = 2166136261;
  for (const character of value) hash = Math.imul(hash ^ character.charCodeAt(0), 16777619);
  return hash >>> 0;
}

function culturePreset(value: string | number): WorldPreset {
  if (typeof value === 'string') {
    const name = value.toLowerCase();
    const exact = WORLD_PRESETS.find(entry => entry.id === name || entry.name.toLowerCase() === name);
    if (exact) return exact;
    const aliases: [RegExp, string][] = [
      [/agri|grow|farmer|river/, 'growth'], [/trade|merchant|market|coast/, 'trade'],
      [/conserv|forest|grove|steward|keeper/, 'conservation'], [/militar|martial|warrior|iron|highland/, 'military'],
      [/know|scholar|research|academy|star/, 'knowledge'],
    ];
    const alias = aliases.find(([pattern]) => pattern.test(name));
    if (alias) return WORLD_PRESETS.find(entry => entry.id === alias[1])!;
  }
  return WORLD_PRESETS[seedNumber(value) % 5];
}

/** Authored auxiliary network, not trained weights or a modification of native Laya. */
export function createWorldModel(cultureOrSeed: string | number = 0): PolicyModel {
  const selected = culturePreset(cultureOrSeed);
  const model: PolicyModel = {
    name: `${selected.name} mind`, version: 'living-world-v1', provenance: 'authored, untrained',
    description: 'Independent civilization candidate scorer. Sixteen local features, eight actions, eight reserved inputs. Native Laya stays fixed.',
    featureNames: [...WORLD_FEATURE_NAMES], hiddenNames: [...WORLD_HIDDEN_NAMES],
    w1: Array.from({ length: 38 }, () => Array<number>(32).fill(0)), b1: Array<number>(38).fill(0),
    w2: Array<number>(38).fill(0), b2: 0, temperature: selected.temperature,
  };
  for (let action = 0; action < WORLD_ACTIONS.length; action++) {
    model.w1[action][16 + action] = 1;
    model.w2[action] = selected.priorities[WORLD_ACTIONS[action]];
  }
  for (let index = 0; index < STATE_NEURONS.length; index++) {
    const neuron = STATE_NEURONS[index], row = index + 8;
    model.w1[row][16 + WORLD_ACTIONS.indexOf(neuron.action)] = 1;
    model.w1[row][CONTEXT_KEYS.indexOf(neuron.feature)] = neuron.complement ? -1 : 1;
    model.b1[row] = neuron.complement ? 0 : -1;
    model.w2[row] = neuron.weight;
  }
  return model;
}

export function createWorldMind(cultureOrSeed: string | number = 0): WorldMind {
  const model = createWorldModel(cultureOrSeed);
  return { model, revision: 0, history: [{ revision: 0, tick: 0, label: 'Authored starting mind', model: validateWorldModel(model) }] };
}

/** Reads settled land and nearby activity only, and never creates regions or changes the world. */
export function worldPolicyContext(world: WorldPolicySource, civ: Civilization): WorldPolicyContext {
  const own = world.settlements.filter(town => town.civId === civ.id && town.population > 0);
  const people = own.reduce((total, town) => total + town.population, 0), divisor = Math.max(1, people);
  const mean = (read: (town: Settlement) => number): number => own.reduce((total, town) => total + read(town) * town.population, 0) / divisor;
  const regions: ReadonlyMap<string, Region> = Array.isArray(world.regions)
    ? new Map(world.regions.map(region => [region.key, region])) : world.regions as ReadonlyMap<string, Region>;
  const regionOf = (town: Settlement): Region | undefined => regions.get(`${Math.floor(town.x / 48)},${Math.floor(town.z / 48)}`);
  const fertility = clamp(mean(town => regionOf(town)?.fertility ?? 0.5));
  const forest = clamp(mean(town => regionOf(town)?.forest ?? 0.5));
  const pressure = clamp(mean(town => regionOf(town)?.pressure ?? 0));
  const health = clamp(mean(town => town.health)), unrest = clamp(mean(town => town.unrest));
  const food = own.reduce((total, town) => total + town.food, 0);
  const foodNeed = people > 0 ? clamp(1 - food / (divisor * 2.5)) : 0;
  const relations = world.relations.filter(relation => relation.a === civ.id || relation.b === civ.id);
  const warIds = new Set(relations.filter(relation => relation.status === 'war').map(relation => relation.a === civ.id ? relation.b : relation.a));
  const nearHome = (point: { x: number; z: number }, radius: number): boolean => own.some(town => distance(town, point) <= radius);
  const nearby = world.settlements.filter(town => town.civId !== civ.id && town.population > 0 && nearHome(town, 180));
  const peacefulNeighbors = new Set(nearby.filter(town => !warIds.has(town.civId)).map(town => town.civId)).size;
  const enemyArmies = world.groups.filter(group => group.kind === 'army' && warIds.has(group.civId) && nearHome(group, 100));
  const army = world.groups.filter(group => group.kind === 'army' && group.civId === civ.id).reduce((total, group) => total + group.members, 0);
  const threat = clamp(warIds.size * 0.3 + enemyArmies.reduce((total, group) => total + group.members, 0) / Math.max(10, divisor * 0.2));
  const context: WorldPolicyContext = {
    foodNeed, fertility, forest, pressure, health, unrest, cohesion: clamp(civ.cohesion),
    wealth: clamp((civ.treasury + own.reduce((total, town) => total + town.wealth, 0)) / (divisor * 4)),
    knowledge: clamp(civ.knowledge / 100), development: clamp((civ.age - 1) / 4),
    crowding: clamp((people / Math.max(1, own.length) - 48) / 64 + pressure * 0.35),
    tradeOpportunity: clamp(peacefulNeighbors / 3), threat, militaryStrength: clamp(army / Math.max(10, divisor * 0.2)),
    frontierRoom: clamp(fertility * (1 - pressure) * (1 - Math.min(0.8, nearby.length * 0.1))),
    migrationNeed: clamp(Math.max(foodNeed * (1 - fertility), unrest, 1 - health, pressure * 0.6)),
  };
  return context;
}

export function worldPolicyFeatures(context: WorldPolicyContext, action: WorldAction): number[] {
  const index = WORLD_ACTIONS.indexOf(action);
  if (index < 0) throw new Error('Unknown world action');
  const values = CONTEXT_KEYS.map(key => context[key]);
  if (!values.every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1))
    throw new Error('World policy features must be finite normalized values between 0 and 1');
  const input = [...values, ...Array<number>(16).fill(0)];
  input[16 + index] = 1;
  return input;
}

export function rankWorldActions(context: WorldPolicyContext, model: PolicyModel): ActionScore[];
export function rankWorldActions(world: WorldPolicySource, civ: Civilization, model?: PolicyModel): ActionScore[];
export function rankWorldActions(source: WorldPolicyContext | WorldPolicySource, civOrModel: Civilization | PolicyModel, explicitModel?: PolicyModel): ActionScore[] {
  const isWorld = 'settlements' in source;
  const context = isWorld ? worldPolicyContext(source as WorldPolicySource, civOrModel as Civilization) : source as WorldPolicyContext;
  const model = isWorld ? explicitModel ?? (civOrModel as Civilization).mind.model : civOrModel as PolicyModel;
  if (!Number.isFinite(model.temperature) || model.temperature <= 0) throw new Error('World policy temperature must be positive and finite');
  const ranked = WORLD_ACTIONS.map(action => {
    const input = worldPolicyFeatures(context, action);
    return { action, input, score: forwardPolicy(input, model), probability: 0 };
  });
  const maximum = Math.max(...ranked.map(entry => entry.score));
  const total = ranked.reduce((sum, entry) => sum + Math.exp((entry.score - maximum) / model.temperature), 0);
  for (const entry of ranked) entry.probability = Math.exp((entry.score - maximum) / model.temperature) / total;
  return ranked.sort((a, b) => b.score - a.score || WORLD_ACTIONS.indexOf(a.action) - WORLD_ACTIONS.indexOf(b.action));
}

/** Trace the exact dense forward pass used by rankWorldActions. No default/global model. */
export function traceWorldPolicy(input: readonly number[], model: PolicyModel): PolicyTrace { return tracePolicy(input, model); }
export { traceWorldPolicy as traceWorldAction };

const MODEL_KEYS = ['name', 'version', 'provenance', 'description', 'featureNames', 'hiddenNames', 'w1', 'b1', 'w2', 'b2', 'temperature'];
function record(value: unknown, label: string, keys: readonly string[], optional: readonly string[] = []): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be a plain object`);
  const proto = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) throw new Error(`${label} must be a plain object`);
  if (keys.some(key => !Object.hasOwn(value, key))) throw new Error(`${label} is missing required fields`);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || !keys.includes(key) && !optional.includes(key)) throw new Error(`${label} contains an unknown field`);
    if (!Object.hasOwn(Object.getOwnPropertyDescriptor(value, key)!, 'value')) throw new Error(`${label} cannot contain accessors`);
  }
  return value as Record<string, unknown>;
}
function dense(value: unknown, length: number, label: string): unknown[] {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype || value.length !== length || Reflect.ownKeys(value).length !== length + 1)
    throw new Error(`${label} must be a dense array of exactly ${length} entries`);
  for (let index = 0; index < length; index++) {
    if (!Object.hasOwn(value, index) || !Object.hasOwn(Object.getOwnPropertyDescriptor(value, index)!, 'value'))
      throw new Error(`${label} must contain dense data entries`);
  }
  return value;
}
function finite(value: unknown, label: string, min: number = WORLD_POLICY_LIMITS.minimum, max: number = WORLD_POLICY_LIMITS.maximum): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max)
    throw new Error(`${label} must be finite and between ${min} and ${max}`);
  return Object.is(value, -0) ? 0 : value;
}
function text(value: unknown, label: string, maximum: number): string {
  if (typeof value !== 'string' || !value.trim() || value.length > maximum || /[\u0000-\u001f\u007f]/.test(value))
    throw new Error(`${label} must be nonempty text of at most ${maximum} characters without control characters`);
  return value;
}

/** Strict import boundary; world labels cannot be replaced with skirmish feature semantics. */
export function validateWorldModel(input: unknown): PolicyModel {
  if (typeof input === 'string') {
    if (input.length > WORLD_POLICY_LIMITS.maximumModelBytes || new TextEncoder().encode(input).length > WORLD_POLICY_LIMITS.maximumModelBytes)
      throw new Error('World weight model is too large');
    try { input = JSON.parse(input); } catch { throw new Error('World weight model is not valid JSON'); }
  }
  const model = record(input, 'World weight model', MODEL_KEYS);
  const labels = (value: unknown, expected: readonly string[], name: string): string[] => dense(value, expected.length, name).map((entry, index) => {
    if (entry !== expected[index]) throw new Error(`${name}[${index}] must be ${expected[index]}; world labels and order cannot change`);
    return entry as string;
  });
  const numeric = (value: unknown, length: number, name: string): number[] => dense(value, length, name).map((entry, index) => finite(entry, `${name}[${index}]`));
  return {
    name: text(model.name, 'name', 120), version: text(model.version, 'version', 64),
    provenance: text(model.provenance, 'provenance', 512), description: text(model.description, 'description', 2048),
    featureNames: labels(model.featureNames, WORLD_FEATURE_NAMES, 'featureNames'), hiddenNames: labels(model.hiddenNames, WORLD_HIDDEN_NAMES, 'hiddenNames'),
    w1: dense(model.w1, 38, 'w1').map((row, index) => numeric(row, 32, `w1[${index}]`)),
    b1: numeric(model.b1, 38, 'b1'), w2: numeric(model.w2, 38, 'w2'), b2: finite(model.b2, 'b2'),
    temperature: finite(model.temperature, 'temperature', WORLD_POLICY_LIMITS.minimumTemperature, WORLD_POLICY_LIMITS.maximumTemperature),
  };
}

export interface WorldWeightEdit { tensor: 'w1' | 'b1' | 'w2' | 'b2' | 'temperature'; row?: number; column?: number; value: number }
export function patchWorldModel(input: PolicyModel, editOrEdits: WorldWeightEdit | readonly WorldWeightEdit[]): PolicyModel {
  const model = validateWorldModel(input), edits: readonly WorldWeightEdit[] = Array.isArray(editOrEdits) ? editOrEdits : [editOrEdits as WorldWeightEdit];
  if (edits.length > 1294) throw new Error('Too many world weight edits');
  const edited = new Set<string>();
  for (const inputEdit of edits) {
    const edit = record(inputEdit, 'World weight edit', ['tensor', 'value'], ['row', 'column']);
    if (!['w1', 'b1', 'w2', 'b2', 'temperature'].includes(edit.tensor as string)) throw new Error('Unknown world weight tensor');
    const tensor = edit.tensor as WorldWeightEdit['tensor'];
    const scalar = tensor === 'b2' || tensor === 'temperature';
    if (scalar && edit.row !== undefined || tensor !== 'w1' && edit.column !== undefined) throw new Error('Invalid coordinate for world weight tensor');
    if (!scalar && (!Number.isInteger(edit.row) || (edit.row as number) < 0 || (edit.row as number) >= 38)) throw new Error('World weight row must be an integer from 0 through 37');
    if (tensor === 'w1' && (!Number.isInteger(edit.column) || (edit.column as number) < 0 || (edit.column as number) >= 32)) throw new Error('World weight column must be an integer from 0 through 31');
    const key = `${tensor}:${edit.row ?? ''}:${edit.column ?? ''}`;
    if (edited.has(key)) throw new Error('A world weight patch cannot edit the same coordinate twice');
    edited.add(key);
    const value = tensor === 'temperature'
      ? finite(edit.value, 'temperature', WORLD_POLICY_LIMITS.minimumTemperature, WORLD_POLICY_LIMITS.maximumTemperature) : finite(edit.value, tensor);
    if (scalar) model[tensor] = value;
    else if (tensor === 'w1') model.w1[edit.row as number][edit.column as number] = value;
    else model[tensor][edit.row as number] = value;
  }
  return model;
}

/** Presets replace eight semantic output priorities and temperature, preserving all other tensor edits. */
export function applyWorldPreset(input: PolicyModel, selected: string | WorldPreset): PolicyModel {
  const id = typeof selected === 'string' ? selected : selected.id;
  const choice = WORLD_PRESETS.find(entry => entry.id === id);
  if (!choice) throw new Error('Unknown world mind preset');
  return patchWorldModel(input, [
    ...WORLD_ACTIONS.map((action, row) => ({ tensor: 'w2' as const, row, value: choice.priorities[action] })),
    { tensor: 'temperature', value: choice.temperature },
  ]);
}

/** Canonical model content identity, including labels/metadata; not a cryptographic signature. */
export function fingerprintWorldModel(input: PolicyModel): string {
  const content = JSON.stringify(validateWorldModel(input));
  let hash = 0xcbf29ce484222325n;
  for (let index = 0; index < content.length; index++) {
    const code = content.charCodeAt(index);
    hash = BigInt.asUintN(64, (hash ^ BigInt(code & 255)) * 0x100000001b3n);
    hash = BigInt.asUintN(64, (hash ^ BigInt(code >>> 8)) * 0x100000001b3n);
  }
  return `fnv1a64_${hash.toString(16).padStart(16, '0')}`;
}

/** Short observation for the fixed native selector; full tensors stay in the browser. */
export function compactWorldObservation(world: WorldPolicySource, civ: Civilization): string {
  const context = worldPolicyContext(world, civ);
  return `Living world civ ${civ.id}; age ${civ.age}; local needs 0-1: food ${round(context.foodNeed)}, soil ${round(context.fertility)}, ` +
    `forest ${round(context.forest)}, pressure ${round(context.pressure)}, threat ${round(context.threat)}, unrest ${round(context.unrest)}. ` +
    `Wealth ${round(context.wealth)}, trade ${round(context.tradeOpportunity)}, knowledge ${round(context.knowledge)}. ` +
    'Patch this civilization auxiliary mind only.';
}
