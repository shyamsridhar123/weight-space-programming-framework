import { getMissionRivalCandidates } from './mission';
import { buildCandidates, forwardPolicy, policyFeatures, type Candidate, type PolicyModel } from './policy';
import type { World } from './simulation';
import type { Team } from './types';

/** At the fixed 0.1-second simulation step, policies choose every two game seconds. */
export const PLAY_POLICY_TICKS = 20;

/** Call after world.step(); absolute ticks prevent a fork from restarting the policy clock. */
export function shouldPlayPolicyStep(world: World): boolean {
  return world.status === 'playing' && world.tick > 0 && world.tick % PLAY_POLICY_TICKS === 0;
}

/** Match the beginner mission's visible commands before scoring either side's choices. */
export function getPlayCandidates(world: World, team: Team): Candidate[] {
  if (team === 1) return getMissionRivalCandidates(world);
  const hasBarracks = world.entities.some(entity => entity.owner === 0 && entity.hp > 0 && entity.kind === 'barracks');
  return buildCandidates(world, team).filter(candidate => {
    const command = candidate.command;
    if (!command) return true;
    if (command.type === 'gather') {
      const source = world.nodes.find(node => node.id === command.targetId);
      return source?.kind === 'food' || source?.kind === 'wood';
    }
    if (command.type === 'train') return command.kind === 'villager' || command.kind === 'spearman';
    if (command.type === 'build') {
      return command.kind === 'house' || command.kind === 'farm' || (command.kind === 'barracks' && !hasBarracks);
    }
    // Scouting, marching, attacking and defending use the same visible movement/combat orders.
    return command.type === 'move' || command.type === 'attack';
  });
}

export interface RankedPlayPolicy {
  candidate: Candidate;
  score: number;
}

/** No probability conversion or hidden heuristic changes the network's candidate scores. */
export function rankPlayPolicy(world: World, team: Team, model: PolicyModel): RankedPlayPolicy[] {
  return getPlayCandidates(world, team)
    .map(candidate => ({ candidate, score: forwardPolicy(policyFeatures(world, team, candidate), model) }))
    .sort((a, b) => b.score - a.score || (a.candidate.id < b.candidate.id ? -1 : a.candidate.id > b.candidate.id ? 1 : 0));
}

export function pickPlayPolicy(world: World, team: Team, model: PolicyModel): Candidate | null {
  return rankPlayPolicy(world, team, model)[0]?.candidate ?? null;
}
