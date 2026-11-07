import type { World } from './simulation';
import { BUILDING_DEFS, UNIT_DEFS, UPGRADE_DEFS } from './types';
import type { BuildingKind, Command, Entity, Resources, Team } from './types';

export interface PlayOrderImpact {
  readonly team: Team;
  readonly time: number;
  readonly command: Command;
  readonly beforeResources: Resources;
  readonly afterResources: Resources;
  readonly beforeArmy: number;
  readonly beforeVillagers: number;
}

export interface PlayOrderObservation {
  status: 'queued' | 'running' | 'completed' | 'interrupted' | 'issued';
  progress: number | null;
  label: string;
  detail: string;
  army: number;
  villagers: number;
}

type Outcome = Omit<PlayOrderObservation, 'army' | 'villagers'>;
interface OrderRecord {
  world: World;
  issued: Outcome;
  training?: { building: Entity; entry: Entity['queue'][number] };
  terminal?: Outcome;
}

// Queue identity belongs to the live world. It is intentionally not a checkpoint.
const records = new WeakMap<PlayOrderImpact, OrderRecord>();
const TRAINING_EPSILON = 1e-7;

function population(world: World, team: Team) {
  let army = 0, villagers = 0;
  for (const entity of world.entities) {
    if (entity.owner !== team || entity.hp <= 0 || entity.category !== 'unit') continue;
    if (entity.kind === 'villager') villagers++;
    else army++;
  }
  return { army, villagers };
}

function issuedOutcome(world: World, command: Command): Outcome {
  let detail: string;
  const units = 'ids' in command ? new Set(command.ids).size : 0;
  const group = `${units} ${units === 1 ? 'unit' : 'units'}`;
  switch (command.type) {
    case 'train': detail = `${UNIT_DEFS[command.kind].name} training ordered.`; break;
    case 'move': detail = `${group} ordered to ${command.attackMove ? 'advance and engage enemies' : 'move to the destination'}.`; break;
    case 'gather': {
      const source = world.nodes.find(node => node.id === command.targetId || node.buildingId === command.targetId);
      detail = `${group} ordered to gather ${source?.kind ?? 'resources'}.`;
      break;
    }
    case 'construct': {
      const target = world.entity(command.targetId);
      const name = target ? BUILDING_DEFS[target.kind as BuildingKind]?.name : undefined;
      detail = `${group} ordered to continue ${name ?? 'building'} construction.`;
      break;
    }
    case 'attack': detail = `${group} ordered to attack the selected enemy.`; break;
    case 'build': detail = `${BUILDING_DEFS[command.kind].name} foundation placed; construction ordered.`; break;
    case 'advance': detail = 'Age advancement ordered.'; break;
    case 'research': detail = `${UPGRADE_DEFS[command.upgrade].name} research ordered.`; break;
    case 'stop': detail = `${group} ordered to stop.`; break;
  }
  return { status: 'issued', progress: null, label: 'Order issued', detail };
}

/** Execute once and capture the actual payment and the exact new training job. */
export function issueTrackedPlayOrder(world: World, team: Team, command: Command):
  { ok: false; reason?: string } | { ok: true; impact: PlayOrderImpact } {
  const beforeResources = { ...world.players[team].resources };
  const before = population(world, team);
  const recordedCommand = structuredClone(command);
  const time = world.time;
  const building = command.type === 'train' ? world.entity(command.buildingId) : undefined;
  const previousEntries = new Set(building?.queue);
  const result = world.command(team, command);
  if (!result.ok) return { ok: false, ...(result.reason === undefined ? {} : { reason: result.reason }) };

  const impact: PlayOrderImpact = {
    team, time, command: recordedCommand, beforeResources,
    afterResources: { ...world.players[team].resources },
    beforeArmy: before.army, beforeVillagers: before.villagers,
  };
  const record: OrderRecord = { world, issued: issuedOutcome(world, recordedCommand) };
  const entry = building?.queue.find(item => !previousEntries.has(item));
  if (building && entry) record.training = { building, entry };
  records.set(impact, record);
  return { ok: true, impact };
}

/** Observe actual queue progress; population reservations never count as recruits. */
export function observePlayOrder(world: World, impact: PlayOrderImpact): PlayOrderObservation {
  const counts = population(world, impact.team);
  const record = records.get(impact);
  if (!record) return { ...issuedOutcome(world, impact.command), ...counts };
  if (record.terminal) return { ...record.terminal, ...counts };
  if (!record.training) return { ...record.issued, ...counts };

  const { building, entry } = record.training;
  const name = UNIT_DEFS[entry.kind].name;
  const currentBuilding = world.entities.find(entity => entity.id === building.id);
  const position = building.queue.indexOf(entry);
  const finish = (outcome: Outcome) => {
    record.terminal = outcome;
    return { ...outcome, ...counts };
  };
  if (world !== record.world || (currentBuilding && currentBuilding !== building)) {
    return finish({ status: 'interrupted', progress: null, label: 'Tracking interrupted', detail: `${name} belongs to an earlier world or building.` });
  }

  // Only successful spawning shifts a finished entry. Destruction leaves the
  // original queue intact, including when a building dies later in that tick.
  if (position === -1 && entry.remaining < TRAINING_EPSILON) {
    return finish({ status: 'completed', progress: 1, label: 'Recruit completed', detail: `${name} joined ${entry.kind === 'villager' ? 'the workforce' : 'the army'}.` });
  }
  if (!currentBuilding || building.hp <= 0 || building.owner !== impact.team) {
    return finish({ status: 'interrupted', progress: null, label: 'Training interrupted', detail: `${name} was not recruited; its building was destroyed or lost.` });
  }
  if (position === -1) {
    return finish({ status: 'interrupted', progress: null, label: 'Training interrupted', detail: `${name} left the queue before training finished.` });
  }
  if (position > 0) {
    return { status: 'queued', progress: 0, label: 'Waiting in queue', detail: `${name} is waiting behind ${position} ${position === 1 ? 'recruit' : 'recruits'}.`, ...counts };
  }
  const progress = Math.max(0, Math.min(1, 1 - entry.remaining / entry.total));
  if (entry.remaining < TRAINING_EPSILON) {
    return { status: 'queued', progress: 1, label: 'Awaiting space', detail: `${name} is ready, but needs a clear space to leave the building.`, ...counts };
  }
  return { status: 'queued', progress, label: `Training ${name}`, detail: `Training ${name}: ${Math.ceil(entry.remaining)} game seconds remaining.`, ...counts };
}
