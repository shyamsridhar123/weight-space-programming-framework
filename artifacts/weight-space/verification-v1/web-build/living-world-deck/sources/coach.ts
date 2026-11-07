import { layaErrorMessage, validateLayaResponse } from '../empires/laya';
import type { NativeWeightEdit, NativeWeightProposal, WeightCoachResponse } from '../empires/weight-coach';
import type { PolicyModel } from '../empires/policy';
import {
  compactWorldObservation, fingerprintWorldModel, patchWorldModel, validateWorldModel,
  type WorldPolicySource,
} from './policy';
import type { Civilization } from './types';

export interface WorldCoachDecision {
  civId: number; proposal: NativeWeightProposal; response: WeightCoachResponse;
  requestedTick: number; ageTicks: number;
}

/** Three small authored alternatives plus keep fit the native bounded choice head. */
export function nativeWorldProposals(input: PolicyModel): NativeWeightProposal[] {
  const model = validateWorldModel(input);
  const definitions: { id: string; description: string; changes: [number, number][] }[] = [
    { id: 'sustain', description: 'Favor food and ecological recovery', changes: [[0, 0.75], [2, 0.5]] },
    { id: 'connect', description: 'Favor commerce and discovery', changes: [[3, 0.7], [4, 0.5]] },
    { id: 'secure', description: 'Favor defense and safe migration', changes: [[5, 0.65], [7, 0.4]] },
  ];
  const proposals: NativeWeightProposal[] = [];
  for (const definition of definitions) {
    const edits: NativeWeightEdit[] = [];
    for (const [row, increase] of definition.changes) {
      const before = model.w2[row];
      const value = Math.min(64, Math.round((before + increase) * 1_000_000) / 1_000_000);
      if (value !== before) edits.push({ tensor: 'w2', row, before, value });
    }
    if (edits.length) proposals.push({ id: definition.id, description: definition.description, edits });
  }
  proposals.push({ id: 'keep', description: 'Keep the current civilization weights', edits: [] });
  return proposals;
}

function validatePatch(raw: unknown, choices: ReadonlyMap<string, NativeWeightProposal>, matchId: string,
  tick: number, revision: number, fingerprint: string): WeightCoachResponse {
  const response = validateLayaResponse(raw, choices, matchId, tick), body = raw as Record<string, unknown>;
  if (body.kind !== 'auxiliary-weight-patch' || body.base_revision !== revision || body.base_fingerprint !== fingerprint)
    throw new Error('Laya returned a patch for a different world mind revision');
  if (response.model.checkpoint_scope !== 'unvalidated-for-rts' || typeof response.model.weights_sha256 !== 'string'
    || !/^[0-9a-f]{64}$/i.test(response.model.weights_sha256))
    throw new Error('Laya returned missing or invalid fixed-checkpoint provenance');
  const expected = choices.get(response.choice)!.edits;
  if (!Array.isArray(body.patch) || body.patch.length !== expected.length) throw new Error('Laya changed the proposed numerical world patch');
  for (let index = 0; index < expected.length; index++) {
    const edit = body.patch[index] as Record<string, unknown> | null;
    if (!edit || typeof edit !== 'object' || Array.isArray(edit) || Object.keys(edit).length !== 4
      || edit.tensor !== 'w2' || edit.row !== expected[index].row
      || edit.before !== expected[index].before || edit.value !== expected[index].value)
      throw new Error('Laya changed the proposed numerical world patch');
  }
  return { ...response, kind: 'auxiliary-weight-patch', base_revision: revision, base_fingerprint: fingerprint,
    patch: expected.map(edit => ({ ...edit })) };
}

function unchanged(world: WorldPolicySource, civ: Civilization, tick: number, revision: number, fingerprint: string): boolean {
  return civ.alive && world.civilizations.includes(civ) && Number.isSafeInteger(world.tick)
    && world.tick >= tick && world.tick - tick <= 60 && civ.mind.revision === revision
    && fingerprintWorldModel(civ.mind.model) === fingerprint;
}

/** Recheck immediately before handing a selected patch to the simulation's revision/history method. */
export function applyWorldCoachPatch(world: WorldPolicySource, civ: Civilization, matchId: string, decision: WorldCoachDecision): PolicyModel {
  const response = decision.response;
  if (decision.civId !== civ.id || response.match_id !== matchId || response.tick !== decision.requestedTick
    || !unchanged(world, civ, decision.requestedTick, response.base_revision, response.base_fingerprint))
    throw new Error('The world mind changed while the native patch was pending; request a new patch');
  const proposals = new Map(nativeWorldProposals(civ.mind.model).map(proposal => [proposal.id, proposal]));
  const validated = validatePatch(response, proposals, matchId, decision.requestedTick, civ.mind.revision, response.base_fingerprint);
  return patchWorldModel(civ.mind.model, validated.patch.map(({ tensor, row, value }) => ({ tensor, row, value })));
}

/** Fixed native Laya selects exact numeric alternatives. It never edits itself or applies an unreviewed response. */
export class LayaWorldCoach {
  private generation = 0;
  private active: AbortController | null = null;

  constructor(private readonly fetcher: typeof fetch = globalThis.fetch.bind(globalThis), private readonly timeoutMs = 20_000) {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Laya timeout must be a positive number');
  }

  get busy(): boolean { return this.active !== null; }

  /** Call on world replacement or selection change, even if the transport ignores AbortSignal. */
  cancel(): void {
    this.generation++;
    const active = this.active;
    this.active = null;
    active?.abort();
  }

  async request(world: WorldPolicySource, civ: Civilization, matchId: string,
    isCurrent: () => boolean = () => true): Promise<WorldCoachDecision | null> {
    if (this.busy || !civ.alive || !world.civilizations.includes(civ) || !isCurrent()) return null;
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,89}$/.test(matchId)) throw new Error('Invalid living-world match identity');
    if (!Number.isSafeInteger(world.tick) || world.tick < 0 || !Number.isSafeInteger(civ.mind.revision) || civ.mind.revision < 0)
      throw new Error('World tick and mind revision must be nonnegative safe integers');
    const model = validateWorldModel(civ.mind.model), revision = civ.mind.revision, fingerprint = fingerprintWorldModel(model);
    const proposals = nativeWorldProposals(model);
    if (proposals.length < 2) throw new Error('The bounded world coach has no available numerical changes');
    const byId = new Map(proposals.map(proposal => [proposal.id, proposal]));
    const requestedTick = world.tick;
    const body = JSON.stringify({ state: compactWorldObservation(world, civ), proposals,
      match_id: matchId, tick: requestedTick, base_revision: revision, base_fingerprint: fingerprint });
    const generation = ++this.generation, controller = new AbortController();
    this.active = controller;
    let timedOut = false;
    let onAbort!: () => void;
    const aborted = new Promise<never>((_, reject) => {
      onAbort = () => reject(new Error('Laya world weight request canceled'));
      controller.signal.addEventListener('abort', onAbort, { once: true });
    });
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, this.timeoutMs);
    try {
      const request = (async () => {
        const response = await this.fetcher('/api/empires/weight-patch', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body, signal: controller.signal,
        });
        let parsed: unknown;
        try { parsed = await response.json(); }
        catch { if (response.ok) throw new Error('Laya returned a response that is not valid JSON'); }
        if (!response.ok) throw new Error(layaErrorMessage(parsed) ?? `Laya world patch request failed (HTTP ${response.status})`);
        return parsed;
      })();
      const raw = await Promise.race([request, aborted]);
      if (generation !== this.generation || !isCurrent()) return null;
      const response = validatePatch(raw, byId, matchId, requestedTick, revision, fingerprint);
      if (!unchanged(world, civ, requestedTick, revision, fingerprint)) return null;
      return { civId: civ.id, proposal: { ...byId.get(response.choice)!, edits: response.patch.map(edit => ({ ...edit })) },
        response, requestedTick, ageTicks: world.tick - requestedTick };
    } catch (error) {
      if (generation !== this.generation || !isCurrent()) return null;
      if (timedOut) throw new Error(`Laya world patch request timed out after ${this.timeoutMs / 1000} seconds`);
      throw error instanceof Error ? error : new Error('Laya world patch request failed');
    } finally {
      clearTimeout(timer);
      controller.signal.removeEventListener('abort', onAbort);
      if (generation === this.generation) this.active = null;
    }
  }
}
