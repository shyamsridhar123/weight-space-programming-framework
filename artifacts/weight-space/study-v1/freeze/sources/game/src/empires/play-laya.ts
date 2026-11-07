import { validateLayaResponse } from './laya';
import { createDefaultModel } from './policy';
import type { World } from './simulation';
import type { NativeWeightEdit, NativeWeightProposal, WeightCoachDecision } from './weight-coach';
import { fingerprintWeightModel, validateWeightModel, WEIGHT_LIMITS, type WeightSnapshot } from './weight-runtime';

const authoredNames = createDefaultModel().hiddenNames;
const playProposalDefinitions = [
  { id: 'gather', description: 'Favor gathering when a resource job is uncovered.', name: 'labor_coverage_gap' },
  { id: 'recruit', description: 'Favor recruiting soldiers when recruitment is legal.', name: 'army_train_bias' },
  { id: 'attack', description: 'Favor legal attack orders for available soldiers.', name: 'attack_bias' },
  { id: 'defend', description: 'Favor defense orders when enemies threaten home.', name: 'visible_threat_defend' },
].map(definition => ({ ...definition, row: authoredNames.indexOf(definition.name) }));

/** The four editable play controls; mission rules still decide which orders are legal. */
export function playLayaWeightProposals(snapshot: WeightSnapshot): NativeWeightProposal[] {
  const model = validateWeightModel(snapshot.model);
  const proposals: NativeWeightProposal[] = [];
  for (const definition of playProposalDefinitions) {
    // Use the authored row, as the play controls do. Imported labels cannot
    // redirect a proposal to an unrelated coefficient.
    const before = model.w2[definition.row];
    const value = Math.min(WEIGHT_LIMITS.maximum, before + 1);
    if (value !== before) proposals.push({ id: definition.id, description: definition.description,
      edits: [{ tensor: 'w2', row: definition.row, before, value }] });
  }
  proposals.push({ id: 'keep', description: 'Keep current weights.', edits: [] });
  return proposals;
}

export interface PlayLayaStatus {
  available: boolean;
  ready: boolean;
  label: string;
  detail: string;
  /** Basename of the model directory actually loaded by the native runtime. */
  model: string | null;
  /** SHA256 of the checkpoint actually loaded; never inferred from static provenance. */
  checkpoint: string | null;
  device: string | null;
}

const STATUS_TIMEOUT_MS = 5_000;
const MAX_PROPOSAL_AGE_SECONDS = 15;
const FIXED_STEP_SECONDS = 0.1;
const sha256 = /^[a-f0-9]{64}$/i;
const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const validText = (value: unknown, maximum: number): value is string =>
  typeof value === 'string' && value.trim().length > 0 && value.length <= maximum
    && !/[\u0000-\u001f\u007f]/.test(value);

function unavailable(detail: string, ready = false): PlayLayaStatus {
  return { available: false, ready, label: 'Laya unavailable', detail, model: null, checkpoint: null, device: null };
}

function statusFromResponse(body: unknown): PlayLayaStatus {
  if (!isRecord(body) || typeof body.ready !== 'boolean')
    return unavailable('The service returned an invalid status. Check the native Laya connection and try again.');
  const ready = body.ready;
  if (!Array.isArray(body.available_modes) || !body.available_modes.every(mode => typeof mode === 'string')
    || !isRecord(body.provenance) || body.provenance.project !== 'RLCD / Cinder Station')
    return unavailable('The connected service did not identify this game’s native Laya runtime.', ready);
  if (!body.available_modes.includes('laya'))
    return unavailable('This service has no native Laya model. Start the game with Laya enabled, then check again.', ready);
  if (!ready)
    return unavailable('The native Laya model is not ready. Wait for startup to finish, then check again.');
  const metadata = body.metadata;
  if (!isRecord(metadata) || !validText(metadata.model_dir, 2048) || !validText(metadata.device, 80)
    || typeof metadata.weights_sha256 !== 'string' || !sha256.test(metadata.weights_sha256))
    return unavailable('The service is ready but did not identify its loaded checkpoint. A verified model is required.', ready);
  const model = metadata.model_dir.replace(/[\\/]+$/, '').split(/[\\/]/).at(-1);
  if (!model || model === '.' || model === '..')
    return unavailable('The service did not identify a valid loaded model directory.', ready);
  return {
    available: true, ready, label: 'Native Laya ready', model,
    checkpoint: metadata.weights_sha256.toLowerCase(), device: metadata.device,
    detail: `${model} is loaded on ${metadata.device}. Laya proposes an edit; you choose whether to apply it. This checkpoint is unvalidated for RTS.`,
  };
}

/** Probe real runtime metadata; unavailable services never produce a synthetic model status. */
export async function getPlayLayaStatus(
  fetcher: typeof fetch = globalThis.fetch.bind(globalThis), signal?: AbortSignal,
): Promise<PlayLayaStatus> {
  if (signal?.aborted) return unavailable('The Laya connection check was canceled.');
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort();
  let onAbort!: () => void;
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => reject(new Error('Status check canceled'));
    controller.signal.addEventListener('abort', onAbort, { once: true });
  });
  signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, STATUS_TIMEOUT_MS);
  try {
    const request = (async () => {
      const response = await fetcher('/api/status', { cache: 'no-store', signal: controller.signal });
      if (!response.ok)
        return unavailable(`The native Laya service could not be reached (HTTP ${response.status}). Check its connection and try again.`);
      let body: unknown;
      try { body = await response.json(); }
      catch { return unavailable('The native Laya service returned a status that is not valid JSON.'); }
      return statusFromResponse(body);
    })();
    return await Promise.race([request, aborted]);
  } catch {
    if (timedOut) return unavailable('The native Laya service did not respond within 5 seconds. Check its connection and try again.');
    if (controller.signal.aborted) return unavailable('The Laya connection check was canceled.');
    return unavailable('The native Laya service is offline or unreachable. Start it, then check the connection again.');
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
    controller.signal.removeEventListener('abort', onAbort);
  }
}

export interface PlayLayaProposalContext {
  snapshot: WeightSnapshot;
  matchId: string;
  tick: number;
  status?: World['status'];
  /** Current native checkpoint identity, when a connection check is available. */
  checkpoint?: string | null;
  /** Supply both values when authoritative simulation times were captured. */
  time?: number;
  requestedTime?: number;
}

export interface PlayLayaProposalEligibility { allowed: boolean; reason: string }
const reject = (reason: string): PlayLayaProposalEligibility => ({ allowed: false, reason });

function sameEdits(actual: unknown, expected: readonly NativeWeightEdit[]): boolean {
  return Array.isArray(actual) && actual.length === expected.length && actual.every((edit, index) => {
    const wanted = expected[index];
    return isRecord(edit) && Object.keys(edit).length === 4 && edit.tensor === wanted.tensor && edit.row === wanted.row
      && edit.before === wanted.before && edit.value === wanted.value;
  });
}

/** Recheck a preview at Apply time. This function never changes weights or game state. */
export function getPlayLayaProposalEligibility(
  decision: WeightCoachDecision | null | undefined, context: PlayLayaProposalContext,
): PlayLayaProposalEligibility {
  if (!decision) return reject('Ask Laya for a proposal first.');
  if (context.status !== undefined && context.status !== 'playing')
    return reject('This match has ended. Start another match before asking Laya.');
  const { response, proposal } = decision;
  if (!response || !proposal || response.match_id !== context.matchId)
    return reject('This proposal belongs to another match. Ask Laya again.');
  if (!Number.isSafeInteger(context.tick) || context.tick < 0 || !Number.isSafeInteger(decision.requestedTick)
    || decision.requestedTick < 0 || context.tick < decision.requestedTick || response.tick !== decision.requestedTick)
    return reject('The match changed after this request. Ask Laya again.');
  if (!Number.isFinite(decision.ageSeconds) || decision.ageSeconds < 0)
    return reject('This proposal has invalid timing. Ask Laya again.');
  // First Conquest advances at a fixed 0.1 seconds per tick. The optional time
  // pair also detects rewinds and callers using a different stepping interval.
  let ageSeconds = Math.max(decision.ageSeconds, (context.tick - decision.requestedTick) * FIXED_STEP_SECONDS);
  if (context.time !== undefined || context.requestedTime !== undefined) {
    if (!Number.isFinite(context.time) || !Number.isFinite(context.requestedTime)
      || context.time! < 0 || context.requestedTime! < 0 || context.time! < context.requestedTime!
      || context.time! - context.requestedTime! + 1e-8 < decision.ageSeconds)
      return reject('The match clock changed after this request. Ask Laya again.');
    ageSeconds = Math.max(ageSeconds, context.time! - context.requestedTime!);
  }
  if (ageSeconds > MAX_PROPOSAL_AGE_SECONDS)
    return reject('This proposal is more than 15 game seconds old. Ask Laya for a fresh proposal.');
  const snapshot = context.snapshot;
  if (response.base_revision !== snapshot.revision || response.base_fingerprint !== snapshot.fingerprint)
    return reject('The weights changed after this request. Ask Laya for a fresh proposal.');
  try {
    if (fingerprintWeightModel(snapshot.model) !== snapshot.fingerprint)
      return reject('The current weights no longer match their recorded revision. Ask Laya again.');
  } catch {
    return reject('The current weights are invalid. Reset the weights before asking Laya.');
  }
  const model = response.model;
  if (!isRecord(model) || !validText(model.model_dir, 2048)
    || typeof model.weights_sha256 !== 'string' || !sha256.test(model.weights_sha256)
    || model.checkpoint_scope !== 'unvalidated-for-rts')
    return reject('This proposal did not identify a native checkpoint. Check the Laya connection and ask again.');
  if (context.checkpoint !== undefined && (typeof context.checkpoint !== 'string'
    || context.checkpoint.toLowerCase() !== model.weights_sha256.toLowerCase()))
    return reject('The Laya checkpoint changed or is unavailable. Check the connection and ask again.');
  const proposals = playLayaWeightProposals(snapshot);
  const expected = proposals.find(candidate => candidate.id === response.choice);
  try {
    validateLayaResponse(response, new Map(proposals.map(candidate => [candidate.id, candidate])), context.matchId, decision.requestedTick);
  } catch {
    return reject('The native decision evidence is invalid. Ask Laya for a fresh proposal.');
  }
  if (response.kind !== 'auxiliary-weight-patch' || !expected || proposal.id !== expected.id
    || !sameEdits(response.patch, expected.edits) || !sameEdits(proposal.edits, expected.edits))
    return reject('The numerical proposal changed after the request. Ask Laya again.');
  if (expected.edits.length === 0) return reject('Laya recommends keeping the current weights. There is no change to apply.');
  return { allowed: true, reason: 'This proposal matches the current match and weights. You can apply it.' };
}

export function canApplyPlayLayaProposal(
  decision: WeightCoachDecision | null | undefined, context: PlayLayaProposalContext,
): boolean {
  return getPlayLayaProposalEligibility(decision, context).allowed;
}
