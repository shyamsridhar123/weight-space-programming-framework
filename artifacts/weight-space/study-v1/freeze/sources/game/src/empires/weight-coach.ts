import { compactObservation } from './policy';
import { layaErrorMessage, validateLayaResponse, type LayaResponse } from './laya';
import { getWeightProposals, WEIGHT_LIMITS, type WeightSnapshot } from './weight-runtime';
import type { World } from './simulation';
import type { Team } from './types';

export interface NativeWeightEdit { tensor: 'w2'; row: number; before: number; value: number }
export interface NativeWeightProposal { id: string; description: string; edits: NativeWeightEdit[] }
export type NativeWeightProposalFactory = (snapshot: WeightSnapshot) => readonly NativeWeightProposal[];
export interface WeightCoachResponse extends LayaResponse {
  kind: 'auxiliary-weight-patch';
  base_revision: number;
  base_fingerprint: string;
  patch: NativeWeightEdit[];
}
export interface WeightCoachDecision {
  proposal: NativeWeightProposal;
  response: WeightCoachResponse;
  requestedTick: number;
  ageSeconds: number;
}

export function nativeWeightProposals(snapshot: WeightSnapshot): NativeWeightProposal[] {
  return getWeightProposals(snapshot.model).map(proposal => ({
    id: proposal.id, description: proposal.description,
    edits: proposal.edits.map(edit => {
      if (edit.tensor !== 'w2' || edit.row === undefined) throw new Error('Coach proposals must edit an explicit W2 factor');
      return { tensor: 'w2', row: edit.row, before: snapshot.model.w2[edit.row], value: edit.value };
    }),
  }));
}

/** Freeze the request's alternatives by value before an asynchronous native call. */
function prepareProposals(factory: NativeWeightProposalFactory, snapshot: WeightSnapshot): NativeWeightProposal[] {
  const proposals = factory(snapshot);
  if (!Array.isArray(proposals) || proposals.length < 2)
    throw new Error('The bounded coach has no available numerical changes');
  if (proposals.length > 5) throw new Error('The bounded coach accepts at most five proposals');
  const ids = new Set<string>();
  const copied = proposals.map(proposal => {
    if (!proposal || typeof proposal.id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$/.test(proposal.id)
      || ids.has(proposal.id) || typeof proposal.description !== 'string' || !proposal.description.trim()
      || proposal.description.length > 60 || /[\u0000-\u001f\u007f]/.test(proposal.description)
      || !Array.isArray(proposal.edits) || proposal.edits.length > 2
      || (proposal.id === 'keep') !== (proposal.edits.length === 0))
      throw new Error('The bounded coach returned an invalid proposal');
    ids.add(proposal.id);
    const rows = new Set<number>();
    const edits = proposal.edits.map((edit: NativeWeightEdit) => {
      if (!edit || edit.tensor !== 'w2' || !Number.isInteger(edit.row) || edit.row < 0 || edit.row >= WEIGHT_LIMITS.hidden
        || rows.has(edit.row) || !Number.isFinite(edit.before) || !Number.isFinite(edit.value)
        || edit.before < WEIGHT_LIMITS.minimum || edit.before > WEIGHT_LIMITS.maximum
        || edit.value < WEIGHT_LIMITS.minimum || edit.value > WEIGHT_LIMITS.maximum
        || edit.before !== snapshot.model.w2[edit.row] || edit.value === edit.before || Math.abs(edit.value - edit.before) > 4)
        throw new Error('The bounded coach returned an invalid numerical edit');
      rows.add(edit.row);
      return { tensor: 'w2' as const, row: edit.row, before: edit.before, value: edit.value };
    });
    return { id: proposal.id, description: proposal.description, edits };
  });
  if (!ids.has('keep')) throw new Error('The bounded coach requires a keep proposal');
  return copied;
}

function validatePatch(body: unknown, proposals: ReadonlyMap<string, NativeWeightProposal>, snapshot: WeightSnapshot,
  matchId: string, tick: number): WeightCoachResponse {
  const choice = validateLayaResponse(body, proposals, matchId, tick);
  const raw = body as Record<string, unknown>;
  if (raw.kind !== 'auxiliary-weight-patch' || raw.base_revision !== snapshot.revision || raw.base_fingerprint !== snapshot.fingerprint)
    throw new Error('Laya returned a patch for a different weight revision');
  const expected = proposals.get(choice.choice)!.edits;
  if (!Array.isArray(raw.patch) || raw.patch.length !== expected.length) throw new Error('Laya changed the proposed numerical patch');
  for (let i = 0; i < expected.length; i++) {
    const edit: unknown = raw.patch[i];
    if (!edit || typeof edit !== 'object' || Array.isArray(edit)) throw new Error('Laya returned an invalid numerical patch');
    const value = edit as Record<string, unknown>;
    if (Object.keys(value).length !== 4 || value.tensor !== 'w2' || value.row !== expected[i].row
      || value.before !== expected[i].before || value.value !== expected[i].value)
      throw new Error('Laya changed the proposed numerical patch');
  }
  return { ...choice, kind: 'auxiliary-weight-patch', base_revision: snapshot.revision,
    base_fingerprint: snapshot.fingerprint, patch: expected.map(edit => ({ ...edit })) };
}

/** Selects among authored numeric patches using the native fixed Laya checkpoint. */
export class LayaWeightCoach {
  private generation = 0;
  private active: AbortController | null = null;

  constructor(
    private readonly fetcher: typeof fetch = globalThis.fetch.bind(globalThis),
    private readonly timeoutMs = 20_000,
    private readonly proposalFactory: NativeWeightProposalFactory = nativeWeightProposals,
  ) {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Laya timeout must be a positive number');
    if (typeof proposalFactory !== 'function') throw new Error('Laya proposal factory must be a function');
  }

  get busy(): boolean { return this.active !== null; }

  cancel(): void {
    this.generation++;
    const active = this.active;
    this.active = null;
    active?.abort();
  }

  async request(world: World, team: Team, matchId: string, snapshot: WeightSnapshot): Promise<WeightCoachDecision | null> {
    if (this.busy || world.status !== 'playing') return null;
    const proposals = prepareProposals(this.proposalFactory, snapshot);
    const byId = new Map(proposals.map(proposal => [proposal.id, proposal]));
    const requestedTick = world.tick, requestedTime = world.time;
    const body = JSON.stringify({ state: compactObservation(world, team), proposals,
      match_id: matchId, tick: requestedTick, base_revision: snapshot.revision, base_fingerprint: snapshot.fingerprint });
    const generation = ++this.generation, controller = new AbortController();
    this.active = controller;
    let timedOut = false;
    let onAbort!: () => void;
    const aborted = new Promise<never>((_, reject) => {
      onAbort = () => reject(new Error('Laya weight request canceled'));
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
        if (!response.ok) throw new Error(layaErrorMessage(parsed) ?? `Laya weight request failed (HTTP ${response.status})`);
        return parsed;
      })();
      const raw = await Promise.race([request, aborted]);
      if (generation !== this.generation) return null;
      const response = validatePatch(raw, byId, snapshot, matchId, requestedTick);
      const ageSeconds = world.time - requestedTime;
      if (world.status !== 'playing' || world.tick < requestedTick || ageSeconds < 0 || ageSeconds > 15) return null;
      return { proposal: byId.get(response.choice)!, response, requestedTick, ageSeconds };
    } catch (error) {
      if (generation !== this.generation) return null;
      if (timedOut) throw new Error(`Laya weight request timed out after ${this.timeoutMs / 1000} seconds`);
      throw error instanceof Error ? error : new Error('Laya weight request failed');
    } finally {
      clearTimeout(timer);
      controller.signal.removeEventListener('abort', onAbort);
      if (generation === this.generation) this.active = null;
    }
  }
}
