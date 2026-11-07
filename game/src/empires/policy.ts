import serialized from './weights/commander-v1.json';
import { World } from './simulation';
import {
  BUILDING_DEFS, RESOURCES, UNIT_DEFS, UPGRADE_DEFS,
  type BuildingKind, type Command, type Entity, type Resource, type ResourceNode,
  type Team, type UnitKind,
} from './types';

export interface Candidate {
  id: string;
  label: string;
  description: string;
  command: Command | null;
}
export type RankedCandidate = Candidate & { score: number; probability: number };
export interface PolicyModel {
  name: string; version: string; provenance: string; description: string;
  featureNames: string[]; hiddenNames: string[];
  w1: number[][]; b1: number[]; w2: number[]; b2: number; temperature: number;
}

// Keep the authored checkpoint private. Live revisions are always explicit arguments.
const model: PolicyModel = {
  ...serialized, featureNames: [...serialized.featureNames], hiddenNames: [...serialized.hiddenNames],
  w1: serialized.w1.map(row => [...row]), b1: [...serialized.b1], w2: [...serialized.w2],
};
export const FEATURE_NAMES: readonly string[] = Object.freeze([...model.featureNames]);
export const MODEL_METADATA = Object.freeze({
  name: model.name, version: model.version, provenance: model.provenance,
  label: 'Authored, untrained neural commander', description: model.description,
  inputSize: model.featureNames.length, hiddenSize: model.b1.length, outputSize: 1,
  parameters: model.w1.length * model.featureNames.length + model.b1.length + model.w2.length + 1,
  architecture: `${model.featureNames.length} → ${model.b1.length} ReLU → 1 candidate score`,
  trained: false,
});

export function validateModel(candidate: PolicyModel): void {
  const input = candidate.featureNames.length, hidden = candidate.b1.length;
  if (input !== 32 || hidden === 0 || candidate.w1.length !== hidden || candidate.w2.length !== hidden)
    throw new Error('Invalid commander network shape');
  if (candidate.hiddenNames.length !== hidden || new Set(candidate.featureNames).size !== input)
    throw new Error('Invalid commander feature or hidden labels');
  if (candidate.w1.some(row => row.length !== input)) throw new Error('Invalid commander matrix row');
  const values = [...candidate.w1.flat(), ...candidate.b1, ...candidate.w2, candidate.b2, candidate.temperature];
  if (!values.every(Number.isFinite) || candidate.temperature <= 0)
    throw new Error('Commander parameters must be finite, with positive temperature');
}
validateModel(model);
for (const row of model.w1) Object.freeze(row);
for (const values of [model.featureNames, model.hiddenNames, model.w1, model.b1, model.w2]) Object.freeze(values);
Object.freeze(model);

/** A mutable copy; editing it never changes the checked-in policy or another controller. */
export function createDefaultModel(): PolicyModel {
  return {
    ...model, featureNames: [...model.featureNames], hiddenNames: [...model.hiddenNames],
    w1: model.w1.map(row => [...row]), b1: [...model.b1], w2: [...model.w2],
  };
}

export interface PolicyTrace {
  input: number[];
  preActivations: number[];
  hiddenActivations: number[];
  /** Hidden-neuron contributions W2[h] * ReLU(W1[h] x + b1[h]), before b2. */
  contributions: number[];
  outputBias: number;
  score: number;
}

/** Dense execution trace of the same forward pass used to choose commands. */
export function tracePolicy(input: readonly number[], weights: PolicyModel = model): PolicyTrace {
  if (input.length !== weights.featureNames.length || !input.every(Number.isFinite))
    throw new Error('Commander input has the wrong shape or contains a nonfinite value');
  const preActivations: number[] = [], hiddenActivations: number[] = [], contributions: number[] = [];
  let score = weights.b2;
  for (let h = 0; h < weights.w1.length; h++) {
    let activation = weights.b1[h];
    for (let f = 0; f < input.length; f++) activation += weights.w1[h][f] * input[f];
    const hidden = Math.max(0, activation), contribution = weights.w2[h] * hidden;
    preActivations.push(activation); hiddenActivations.push(hidden); contributions.push(contribution);
    score += contribution;
  }
  if (!Number.isFinite(score)) throw new Error('Commander produced a nonfinite score');
  return { input: [...input], preActivations, hiddenActivations, contributions, outputBias: weights.b2, score };
}

/** A real, dense matrix-vector forward pass. The checked-in values were authored; no fitting occurs here. */
export function forwardPolicy(input: readonly number[], weights: PolicyModel = model): number {
  if (input.length !== weights.featureNames.length || !input.every(Number.isFinite))
    throw new Error('Commander input has the wrong shape or contains a nonfinite value');
  let score = weights.b2;
  for (let h = 0; h < weights.w1.length; h++) {
    let activation = weights.b1[h];
    const row = weights.w1[h];
    for (let f = 0; f < input.length; f++) activation += row[f] * input[f];
    score += weights.w2[h] * Math.max(0, activation);
  }
  if (!Number.isFinite(score)) throw new Error('Commander produced a nonfinite score');
  return score;
}

const clamp = (x: number, max = 1): number => Math.max(0, Math.min(max, x));
const distance = (a: { x: number; z: number }, b: { x: number; z: number }): number => Math.hypot(a.x - b.x, a.z - b.z);
const alphabetical = (a: { id: string }, b: { id: string }): number => a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
const military = (e: Entity): boolean => e.category === 'unit' && e.kind !== 'villager';
const productionKinds: BuildingKind[] = ['barracks', 'range', 'stable', 'forge'];
const unitKinds: UnitKind[] = ['villager', 'spearman', 'archer', 'knight', 'ram'];
const buildKinds = Object.keys(BUILDING_DEFS) as BuildingKind[];
const waitCandidate = (): Candidate => ({ id: 'wait', label: 'Continue orders', description: 'Continue current orders; spend no resources.', command: null });

interface PolicyView {
  world: World; team: Team; own: Entity[]; workers: Entity[]; idleWorkers: Entity[];
  army: Entity[]; idleArmy: Entity[]; buildings: Entity[]; visibleEnemies: Entity[];
  threats: Entity[]; nodes: ResourceNode[]; home: { x: number; z: number };
  workerCount: number; armyCount: number; activeFarms: number; assigned: Record<Resource, number>;
  ownKindCount: Map<string, number>; queued: Map<string, number>;
}

function view(world: World, team: Team): PolicyView {
  const own = world.entities.filter(e => e.owner === team && e.hp > 0);
  const workers = own.filter(e => e.kind === 'villager');
  const army = own.filter(military);
  const buildings = own.filter(e => e.category === 'building');
  const base = buildings.find(e => e.kind === 'towncenter') ?? buildings[0] ?? own[0];
  const home = base ? { x: base.x, z: base.z } : { x: team === 0 ? 10 : world.size - 10, z: team === 0 ? 10 : world.size - 10 };
  // Hidden enemies are never read by features, candidate targets, or the rule baseline.
  const visibleEnemies = world.entities.filter(e => e.owner !== team && e.hp > 0 && world.isVisible(team, e.x, e.z));
  const threats = visibleEnemies.filter(e => e.category === 'unit' && distance(e, home) < 18);
  const nodes = world.nodes.filter(n => world.isVisible(team, n.x, n.z) && n.amount > 0 && (n.owner === undefined || n.owner === team));
  const assigned = { food: 0, wood: 0, gold: 0, stone: 0 };
  for (const worker of workers) {
    if (worker.order.type !== 'gather') continue;
    const resource = worker.order.resource ?? nodes.find(n => n.id === worker.order.targetId)?.kind;
    if (resource) assigned[resource]++;
  }
  const ownKindCount = new Map<string, number>(), queued = new Map<string, number>();
  for (const entity of own) {
    ownKindCount.set(entity.kind, (ownKindCount.get(entity.kind) ?? 0) + 1);
    for (const job of entity.queue) queued.set(job.kind, (queued.get(job.kind) ?? 0) + 1);
  }
  const workerCount = workers.length + (queued.get('villager') ?? 0);
  const armyCount = army.length + [...queued.entries()].reduce((sum, [kind, count]) => sum + (kind === 'villager' ? 0 : count), 0);
  const activeFarms = buildings.filter(e => e.kind === 'farm' && nodes.some(n => n.buildingId === e.id)).length;
  return {
    world, team, own, workers, army, buildings, visibleEnemies, threats, nodes, home,
    idleWorkers: workers.filter(e => e.order.type === 'idle'),
    idleArmy: army.filter(e => e.order.type === 'idle'), workerCount, armyCount, activeFarms, assigned, ownKindCount, queued,
  };
}

function laborShares(age: number): Record<Resource, number> {
  return age < 2 ? { food: 0.52, wood: 0.32, gold: 0.13, stone: 0.03 }
    : { food: 0.43, wood: 0.3, gold: 0.22, stone: 0.05 };
}

function armyShares(age: number): Partial<Record<UnitKind, number>> {
  return age < 2 ? { spearman: 1 } : age < 3 ? { spearman: 0.6, archer: 0.4 }
    : { spearman: 0.35, archer: 0.3, knight: 0.25, ram: 0.1 };
}

function candidateGap(v: PolicyView, candidate?: Candidate): number {
  const cmd = candidate?.command;
  if (!cmd) return 0;
  if (cmd.type === 'gather') {
    const resource = v.nodes.find(n => n.id === cmd.targetId)?.kind;
    if (!resource) return 0;
    const desired = laborShares(v.world.players[v.team].age)[resource] * v.workers.length;
    return clamp((desired - v.assigned[resource]) / Math.max(1, desired));
  }
  if (cmd.type === 'build') {
    // A depleted field no longer covers food needs even though its building remains.
    const have = cmd.kind === 'farm' ? v.activeFarms + v.buildings.filter(e => e.kind === 'farm' && e.construction < 1).length
      : v.ownKindCount.get(cmd.kind) ?? 0;
    const target = cmd.kind === 'farm' ? Math.max(1, Math.ceil(v.workers.length / 6))
      : cmd.kind === 'towncenter' ? Math.max(1, Math.ceil(v.workers.length / 30)) : 1;
    return clamp((target - have) / target);
  }
  if (cmd.type === 'train' && cmd.kind !== 'villager') {
    const desired = armyShares(v.world.players[v.team].age)[cmd.kind] ?? 0;
    const fraction = ((v.ownKindCount.get(cmd.kind) ?? 0) + (v.queued.get(cmd.kind) ?? 0)) / Math.max(1, v.armyCount);
    return desired > 0 ? clamp((desired - fraction) / desired) : 0;
  }
  if (candidate?.id.startsWith('scout')) return clamp(1 - v.army.filter(e => distance(e, v.home) > 18).length / 2);
  return 0;
}

function candidateDisruption(v: PolicyView, candidate?: Candidate): number {
  const cmd = candidate?.command;
  if (!cmd) return 0;
  if (cmd.type === 'gather') {
    const members = v.workers.filter(e => cmd.ids.includes(e.id));
    return 0.9 * members.filter(e => e.order.type !== 'idle').length / Math.max(1, members.length);
  }
  if (cmd.type === 'build') {
    const members = v.workers.filter(e => cmd.workerIds?.includes(e.id));
    return 0.3 * members.filter(e => e.order.type !== 'idle').length / Math.max(1, members.length);
  }
  return 0;
}

function features(v: PolicyView, candidate?: Candidate): number[] {
  const player = v.world.players[v.team], stock = player.resources;
  const activeProduction = v.buildings.filter(e => productionKinds.includes(e.kind as BuildingKind)).length;
  const enemyArmy = v.visibleEnemies.filter(military).length;
  const input = [
    clamp(1 - stock.food / 240), clamp(1 - stock.wood / 240), clamp(1 - stock.gold / 220), clamp(1 - stock.stone / 160),
    clamp(v.idleWorkers.length / 6), clamp((28 - v.workerCount) / 28),
    clamp(1 - (v.world.capacity(v.team) - v.world.population(v.team)) / 4),
    clamp(v.army.length / 10), clamp(v.threats.length / 6), clamp((3 - activeProduction) / 3),
    clamp((player.age - 1) / 3), clamp(RESOURCES.reduce((sum, kind) => sum + stock[kind], 0) / 1600),
    clamp(0.5 + (v.army.length - enemyArmy) / 36), clamp(1 - v.armyCount / 20),
    candidateGap(v, candidate), candidateDisruption(v, candidate),
    ...Array<number>(16).fill(0),
  ];
  const cmd = candidate?.command;
  let category = -1;
  if (candidate?.id === 'wait') category = 31;
  else if (cmd?.type === 'gather') {
    const resource = v.nodes.find(n => n.id === cmd.targetId)?.kind;
    if (resource) category = 16 + RESOURCES.indexOf(resource);
  } else if (cmd?.type === 'train') category = cmd.kind === 'villager' ? 20 : 24;
  else if (cmd?.type === 'build') category = cmd.kind === 'house' ? 21 : cmd.kind === 'farm' ? 22
    : cmd.kind === 'tower' || cmd.kind === 'castle' ? 30 : 23;
  else if (cmd?.type === 'advance') category = 25;
  else if (cmd?.type === 'research') category = 26;
  else if (candidate?.id.startsWith('defend')) category = 27;
  else if (candidate?.id.startsWith('assault')) category = 28;
  else if (candidate?.id.startsWith('scout')) category = 29;
  if (category >= 0) input[category] = 1;
  return input;
}

export function policyFeatures(world: World, team: Team, candidate?: Candidate): number[] {
  return features(view(world, team), candidate);
}

function assaultApproach(v: PolicyView, target: { x: number; z: number }): Command | undefined {
  const ids = v.idleArmy.map(e => e.id);
  const move = (point: { x: number; z: number }): Command => ({ type: 'move', ids, ...point, attackMove: true });
  const exact = move(target);
  if (v.world.canCommand(v.team, exact).ok) return exact;
  // A foundation can occupy the inferred point. Probe nearby cell centers via
  // ordinary command legality without reading the identity of hidden blockers.
  const nearby: { x: number; z: number }[] = [];
  for (let z = Math.floor(target.z) - 6; z <= Math.floor(target.z) + 6; z++) {
    for (let x = Math.floor(target.x) - 6; x <= Math.floor(target.x) + 6; x++) {
      const point = { x: x + 0.5, z: z + 0.5 };
      if (distance(point, target) <= 6) nearby.push(point);
    }
  }
  nearby.sort((a, b) => distance(a, target) - distance(b, target)
    || distance(a, v.home) - distance(b, v.home) || a.z - b.z || a.x - b.x);
  for (const point of nearby) {
    const command = move(point);
    if (v.world.canCommand(v.team, command).ok) return command;
  }
  return undefined;
}

function candidates(v: PolicyView): Candidate[] {
  const { world, team } = v;
  if (world.status !== 'playing') return [waitCandidate()];
  const result: Candidate[] = [];
  const add = (id: string, label: string, description: string, command: Command): void => {
    // This authoritative mask runs before either the MLP, softmax, or rule selection sees an action.
    if (world.canCommand(team, command).ok) result.push({ id, label, description, command });
  };
  for (const resource of RESOURCES) {
    const available = v.idleWorkers.length ? v.idleWorkers.slice(0, 3) : v.workers.filter(worker =>
      worker.order.type === 'gather' && worker.order.resource !== resource &&
      !v.nodes.some(n => n.kind === resource && n.id === worker.order.targetId)).slice(0, 1);
    if (!available.length) continue;
    const node = v.nodes.filter(n => n.kind === resource).sort((a, b) => distance(a, available[0]) - distance(b, available[0]) || a.id - b.id)[0];
    if (node) add(`gather_${resource}_${node.id}`, `Gather ${resource}`, `Assign ${available.length} worker${available.length === 1 ? '' : 's'} to ${resource}.`, { type: 'gather', ids: available.map(e => e.id), targetId: node.id });
  }
  for (const kind of buildKinds) {
    // Houses cannot raise the simulation's hard 100-person cap.
    if (kind === 'house' && world.capacity(team) >= 100) continue;
    // A macro starts one construction project per kind; existing projects retain their builders.
    if (v.buildings.some(e => e.kind === kind && e.construction < 1)) continue;
    const builders = v.workers.filter(e => e.order.type === 'idle' || e.order.type === 'gather')
      .sort((a, b) => Number(a.order.type !== 'idle') - Number(b.order.type !== 'idle') || a.id - b.id);
    if (!builders.length) continue;
    const site = world.findBuildSite(team, kind);
    if (site) add(`build_${kind}`, `Build ${BUILDING_DEFS[kind].name}`, `Build a ${BUILDING_DEFS[kind].name.toLowerCase()}.`, { type: 'build', kind, ...site, workerIds: [builders[0].id] });
  }
  for (const kind of unitKinds) {
    const producer = v.buildings.filter(e => e.construction >= 1 && BUILDING_DEFS[e.kind as BuildingKind].trainable.includes(kind))
      .sort((a, b) => a.queue.length - b.queue.length || a.id - b.id)
      .find(e => world.canCommand(team, { type: 'train', buildingId: e.id, kind }).ok);
    if (producer) add(`train_${kind}_${producer.id}`, `Train ${UNIT_DEFS[kind].name}`, `Queue one ${UNIT_DEFS[kind].name.toLowerCase()}.`, { type: 'train', buildingId: producer.id, kind });
  }
  add('advance_age', 'Advance age', `Begin the next age; unlock stronger technology.`, { type: 'advance' });
  for (const upgrade of ['economy', 'weapons', 'armor'] as const)
    add(`research_${upgrade}`, UPGRADE_DEFS[upgrade].name, `Research ${UPGRADE_DEFS[upgrade].name.toLowerCase()}.`, { type: 'research', upgrade });
  const threat = [...v.threats].sort((a, b) => distance(a, v.home) - distance(b, v.home) || a.id - b.id)[0];
  if (threat) {
    const defenders = v.army.filter(e => e.order.type !== 'attack' || e.order.targetId !== threat.id);
    if (defenders.length) add(`defend_${threat.id}`, 'Defend the settlement', `Send ${defenders.length} troops at a visible threat near home.`, { type: 'attack', ids: defenders.map(e => e.id), targetId: threat.id });
  }
  if (v.idleArmy.length) {
    const visibleTarget = [...v.visibleEnemies].sort((a, b) => Number(b.kind === 'towncenter') - Number(a.kind === 'towncenter') || distance(a, v.home) - distance(b, v.home) || a.id - b.id)[0];
    if (visibleTarget) add(`assault_visible_${visibleTarget.id}`, 'Attack visible enemy', `Send ${v.idleArmy.length} reserve troops at a visible enemy.`, { type: 'attack', ids: v.idleArmy.map(e => e.id), targetId: visibleTarget.id });
    // Map symmetry is a prior, not a read of the opponent's hidden position or resources.
    const inferred = { x: clamp(world.size - v.home.x, world.size - 2), z: clamp(world.size - v.home.z, world.size - 2) };
    const approach = assaultApproach(v, inferred);
    if (approach) add('assault_inferred_base', 'March on rival territory', `Attack-move ${v.idleArmy.length} reserves toward inferred rival land.`, approach);
    const scout = v.idleArmy[0];
    const angle = Math.atan2(inferred.z - v.home.z, inferred.x - v.home.x);
    const target = { x: clamp(v.home.x + Math.cos(angle) * world.size * 0.46, world.size - 2), z: clamp(v.home.z + Math.sin(angle) * world.size * 0.46, world.size - 2) };
    if (!world.isExplored(team, target.x, target.z))
      add(`scout_${scout.id}`, 'Scout the frontier', 'Send one reserve troop to unexplored central ground.', { type: 'move', ids: [scout.id], ...target });
  }
  result.push(waitCandidate());
  return result.sort(alphabetical);
}

export function buildCandidates(world: World, team: Team): Candidate[] {
  return candidates(view(world, team));
}

function rankPrepared(prepared: { candidate: Candidate; input: number[] }[], weights: PolicyModel): RankedCandidate[] {
  const scored = prepared.map(({ candidate, input }) => ({ ...candidate, score: forwardPolicy(input, weights), probability: 0 }));
  const peak = Math.max(...scored.map(candidate => candidate.score));
  const denominator = scored.reduce((sum, candidate) => sum + Math.exp((candidate.score - peak) / weights.temperature), 0);
  for (const candidate of scored) candidate.probability = Math.exp((candidate.score - peak) / weights.temperature) / denominator;
  return scored.sort((a, b) => b.score - a.score || alphabetical(a, b));
}

export function rankActions(world: World, team: Team, weights: PolicyModel = model): RankedCandidate[] {
  const observation = view(world, team);
  return rankPrepared(candidates(observation).map(candidate => ({ candidate, input: features(observation, candidate) })), weights);
}

/** Generate the legal candidates and feature vectors once, then compare two explicit checkpoints. */
export function comparePolicies(world: World, team: Team, active: PolicyModel, baseline: PolicyModel): {
  active: RankedCandidate[]; baseline: RankedCandidate[];
} {
  const observation = view(world, team);
  const prepared = candidates(observation).map(candidate => ({ candidate, input: features(observation, candidate) }));
  return { active: rankPrepared(prepared, active), baseline: rankPrepared(prepared, baseline) };
}

export function chooseNeural(world: World, team: Team, weights: PolicyModel = model): Candidate {
  return rankActions(world, team, weights)[0];
}

/** A separately labelled, explicit decision tree; it is a comparator, not the network implementation. */
export function chooseRule(world: World, team: Team): Candidate {
  const v = view(world, team), legal = candidates(v), player = world.players[team];
  const find = (prefix: string): Candidate | undefined => legal.find(c => c.id.startsWith(prefix));
  const defend = find('defend');
  if (defend) return defend;
  const gather = legal.filter(c => c.command?.type === 'gather').sort((a, b) => {
    const resource = (c: Candidate): Resource => v.nodes.find(n => n.id === (c.command as Extract<Command, { type: 'gather' }>).targetId)!.kind;
    const need = (c: Candidate): number => laborShares(player.age)[resource(c)] * Math.max(1, v.workers.length) - v.assigned[resource(c)];
    return need(b) - need(a) || alphabetical(a, b);
  });
  if (v.idleWorkers.length && gather[0]) return gather[0];
  const house = find('build_house');
  if (world.capacity(team) - world.population(team) <= 3 && house) return house;
  const worker = find('train_villager');
  if (v.workerCount < 22 && worker) return worker;
  const advance = find('advance_age');
  if (v.workers.length >= 10 && advance) return advance;
  if (v.idleArmy.length >= 7 || (v.army.length >= 12 && v.idleArmy.length >= 2)) {
    const assault = find('assault_visible') ?? find('assault_inferred');
    if (assault) return assault;
  }
  for (const kind of ['barracks', 'range', 'stable', 'forge'] as const) {
    const building = find(`build_${kind}`);
    if (building && !v.ownKindCount.has(kind)) return building;
  }
  const foodRemaining = v.nodes.filter(n => n.kind === 'food').reduce((sum, n) => sum + n.amount, 0);
  const farm = find('build_farm');
  if (farm && foodRemaining < v.workers.length * 80 && candidateGap(v, farm) > 0) return farm;
  const train = legal.filter(c => c.command?.type === 'train' && c.command.kind !== 'villager')
    .sort((a, b) => candidateGap(v, b) - candidateGap(v, a) || alphabetical(a, b));
  if (train[0]) return train[0];
  const research = find('research_economy') ?? find('research_weapons') ?? find('research_armor');
  if (research && v.army.length >= 5) return research;
  const castle = find('build_castle'), tower = find('build_tower');
  if (castle && !v.ownKindCount.has('castle')) return castle;
  if (tower && !v.ownKindCount.has('tower')) return tower;
  if (gather[0] && candidateGap(v, gather[0]) > 0.6) return gather[0];
  const scout = find('scout');
  if (scout && v.army.length >= 3) return scout;
  return find('wait')!;
}

function family(candidate: Candidate): string {
  const cmd = candidate.command;
  if (!cmd) return 'wait';
  if (cmd.type === 'train') return cmd.kind === 'villager' ? 'workers' : 'army';
  if (cmd.type === 'build') return cmd.kind === 'house' ? 'housing' : 'construction';
  if (candidate.id.startsWith('defend')) return 'defense';
  return cmd.type;
}

/** Four diverse legal options plus continuation keep Laya's candidate head compact. */
export function layaCandidates(world: World, team: Team, weights: PolicyModel = model): Candidate[] {
  const ranked = rankActions(world, team, weights), selected: Candidate[] = [], families = new Set<string>();
  for (const candidate of ranked) {
    const group = family(candidate);
    if (group === 'wait' || families.has(group)) continue;
    selected.push(candidate); families.add(group);
    if (selected.length === 4) break;
  }
  for (const candidate of ranked) {
    if (selected.length === 4) break;
    if (candidate.id !== 'wait' && !selected.some(chosen => chosen.id === candidate.id)) selected.push(candidate);
  }
  selected.push(ranked.find(candidate => candidate.id === 'wait')!);
  return selected;
}

export function compactObservation(world: World, team: Team): string {
  const v = view(world, team), player = world.players[team];
  const stock = RESOURCES.map(resource => `${resource} ${Math.floor(player.resources[resource])}`).join(', ');
  const labor = RESOURCES.map(resource => `${resource[0]}${v.assigned[resource]}`).join('/');
  return `t=${Math.floor(world.time)}s. Age ${player.age}${player.advancing > 0 ? ' (advancing)' : ''}. ${stock}. ` +
    `Population ${world.population(team)}/${world.capacity(team)}. Workers ${v.workers.length}, idle ${v.idleWorkers.length}, gathering ${labor}. ` +
    `Troops ${v.army.length}, reserves ${v.idleArmy.length}. Visible foes ${v.visibleEnemies.length}; near home ${v.threats.length}. ` +
    `Production ${v.buildings.filter(e => productionKinds.includes(e.kind as BuildingKind)).length}; productive farms ${v.activeFarms}. Hidden enemy state unknown.`;
}
