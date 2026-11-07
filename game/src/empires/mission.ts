import { buildCandidates, forwardPolicy, policyFeatures, type Candidate, type PolicyModel } from './policy';
import { World } from './simulation';
import type { Entity, Team } from './types';

/** These tutorial rules constrain available orders; the neural weights rank them. */
export const MISSION_RULES = Object.freeze({
  rivalGraceSeconds: 330,
  rivalRecruitStartSeconds: 270,
  rivalArmyLimit: 6,
  rivalRecruitBudget: 10,
  rivalRaidSize: 6,
  foodGoal: 50,
  woodGoal: 30,
  trainedGoal: 8,
});

export interface MissionGoal {
  id: 'gather' | 'house' | 'army' | 'victory';
  title: string;
  detail: string;
  done: boolean;
  current: number;
  target: number;
}

export interface MissionProgress {
  phase: 'gather' | 'house' | 'army' | 'attack' | 'won' | 'lost' | 'draw';
  gatheredFood: number;
  gatheredWood: number;
  housesBuilt: number;
  soldiersTrained: number;
  armySize: number;
  playerTownCenterHp: number;
  rivalTownCenterHp: number;
  rivalGraceRemaining: number;
  goals: MissionGoal[];
}

const military = (entity: Entity) => entity.category === 'unit' && entity.kind !== 'villager';

export function createMissionWorld(seed = 42): World {
  return new World(seed, { scenario: 'first-victory' });
}

/** A legal approach makes attack-move engage guards before the Town Center. */
export function missionAttackPoint(world: World, team: Team = 0): { x: number; z: number } | null {
  const enemy = world.entities.find(entity => entity.owner !== team && entity.kind === 'towncenter' && entity.hp > 0);
  const ids = world.entities.filter(entity => entity.owner === team && entity.hp > 0 && military(entity)).map(entity => entity.id);
  if (!enemy || !ids.length) return null;
  for (const distance of [4, 5, 6]) {
    for (const offset of [0, -1, 1, -2, 2]) {
      const point = { x: enemy.x + (team === 0 ? -distance : distance), z: enemy.z + offset };
      if (world.canCommand(team, { type: 'move', ids, ...point, attackMove: true }).ok) return point;
    }
  }
  return null;
}

/** Read actual deposits, completed production, live troops, and target health. */
export function getMissionProgress(world: World): MissionProgress {
  const gatheredFood = world.players[0].gathered.food;
  const gatheredWood = world.players[0].gathered.wood;
  const housesBuilt = world.production[0].constructed.house ?? 0;
  const soldiersTrained = Object.entries(world.production[0].trained)
    .reduce((sum, [kind, count]) => sum + (kind === 'villager' ? 0 : count), 0);
  const armySize = world.entities.filter(entity => entity.owner === 0 && entity.hp > 0 && military(entity)).length;
  const home = world.entities.find(entity => entity.owner === 0 && entity.kind === 'towncenter' && entity.hp > 0);
  const rival = world.entities.find(entity => entity.owner === 1 && entity.kind === 'towncenter' && entity.hp > 0);
  const playerTownCenterHp = Math.max(0, home?.hp ?? 0);
  const rivalTownCenterHp = Math.max(0, rival?.hp ?? 0);
  const rivalMaxHp = rival?.maxHp ?? 1800;
  const goals: MissionGoal[] = [
    {
      id: 'gather', title: 'Gather food and wood',
      detail: `${Math.floor(gatheredFood)} / ${MISSION_RULES.foodGoal} food · ${Math.floor(gatheredWood)} / ${MISSION_RULES.woodGoal} wood returned to your Town Center`,
      current: Math.min(MISSION_RULES.foodGoal, gatheredFood) + Math.min(MISSION_RULES.woodGoal, gatheredWood),
      target: MISSION_RULES.foodGoal + MISSION_RULES.woodGoal,
      done: gatheredFood >= MISSION_RULES.foodGoal - 1e-6 && gatheredWood >= MISSION_RULES.woodGoal - 1e-6,
    },
    {
      id: 'house', title: 'Build a house',
      detail: 'Select a villager, place a house on clear ground, and let construction finish. A completed house adds 5 population.',
      current: housesBuilt, target: 1, done: housesBuilt >= 1,
    },
    {
      id: 'army', title: `Train ${MISSION_RULES.trainedGoal} spearmen`,
      detail: `${soldiersTrained} recruits completed · ${armySize} troops alive. Train at your Barracks; each costs 50 food and 25 wood.`,
      current: soldiersTrained, target: MISSION_RULES.trainedGoal, done: soldiersTrained >= MISSION_RULES.trainedGoal,
    },
    {
      id: 'victory', title: 'Destroy the red Town Center',
      detail: 'Train 8 extra spearmen, keep your army together, and order it to attack the red Town Center. Soldiers fight enemy guards on the way. Recruit replacements as needed.',
      current: rivalMaxHp - rivalTownCenterHp, target: rivalMaxHp, done: world.status === 'won',
    },
  ];
  const next = goals.find(goal => !goal.done)?.id ?? 'victory';
  return {
    phase: world.status === 'playing' ? next === 'victory' ? 'attack' : next : world.status,
    gatheredFood, gatheredWood, housesBuilt, soldiersTrained, armySize,
    playerTownCenterHp, rivalTownCenterHp,
    rivalGraceRemaining: Math.max(0, MISSION_RULES.rivalGraceSeconds - world.time),
    goals,
  };
}

/**
 * Exact eligible set for both rival play and the optional weight inspector.
 * Every order still passes World.canCommand, pays its normal cost, and takes
 * normal simulation time. Grace stops raids; troops may defend their home.
 */
export function getMissionRivalCandidates(world: World): Candidate[] {
  if (world.status !== 'playing') return [];
  const own = world.entities.filter(entity => entity.owner === 1 && entity.hp > 0);
  const army = own.filter(military);
  const idleArmy = army.filter(entity => entity.order.type === 'idle');
  const queuedArmy = own.reduce((sum, entity) => sum + entity.queue.filter(job => job.kind !== 'villager').length, 0);
  const trained = Object.entries(world.production[1].trained)
    .reduce((sum, [kind, count]) => sum + (kind === 'villager' ? 0 : count), 0);
  const houseCount = own.filter(entity => entity.kind === 'house').length;
  const farmCount = own.filter(entity => entity.kind === 'farm').length;
  const foodRemaining = world.nodes.filter(node => node.kind === 'food' && node.amount > 0
    && (node.owner === undefined || node.owner === 1) && node.x > world.size / 2)
    .reduce((sum, node) => sum + node.amount, 0);
  const raiding = army.some(entity => entity.order.type === 'attackMove'
    || (entity.order.type === 'attack' && (world.entity(entity.order.targetId ?? -1)?.x ?? world.size) < world.size / 2));

  return buildCandidates(world, 1).filter(candidate => {
    const command = candidate.command;
    if (!command) return true;
    if (command.type === 'gather') {
      const source = world.nodes.find(node => node.id === command.targetId);
      return source?.kind === 'food' || source?.kind === 'wood';
    }
    if (command.type === 'train') {
      return command.kind === 'spearman'
        && world.time >= MISSION_RULES.rivalRecruitStartSeconds
        && army.length + queuedArmy < MISSION_RULES.rivalArmyLimit
        && trained + queuedArmy < MISSION_RULES.rivalRecruitBudget;
    }
    if (command.type === 'build') {
      return (command.kind === 'house' && houseCount < 2 && world.capacity(1) - world.population(1) <= 3)
        || (command.kind === 'farm' && farmCount < 2 && foodRemaining < 480);
    }
    if (candidate.id.startsWith('defend')) return true;
    if (candidate.id.startsWith('assault')) {
      return world.time >= MISSION_RULES.rivalGraceSeconds
        && (idleArmy.length >= MISSION_RULES.rivalRaidSize || (raiding && idleArmy.length > 0));
    }
    return false;
  });
}

/** No hidden heuristics alter network scores after the mission legality mask. */
export function chooseMissionRival(world: World, model: PolicyModel): Candidate | null {
  return getMissionRivalCandidates(world)
    .map(candidate => ({ candidate, score: forwardPolicy(policyFeatures(world, 1, candidate), model) }))
    .sort((a, b) => b.score - a.score || (a.candidate.id < b.candidate.id ? -1 : a.candidate.id > b.candidate.id ? 1 : 0))[0]?.candidate ?? null;
}
