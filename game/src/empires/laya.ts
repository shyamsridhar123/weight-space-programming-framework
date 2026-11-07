import { compactObservation, layaCandidates, type Candidate, type PolicyModel } from './policy';
import type { World } from './simulation';
import type { Team } from './types';

export interface LayaResponse {
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  confidence_kind: 'normalized_entropy';
  inference_ms: number;
  queue_ms: number;
  match_id: string;
  tick: number;
  model: Record<string, unknown>;
}

export interface LayaDecision {
  candidate: Candidate;
  requestedTick: number;
  response: LayaResponse;
  ageSeconds: number;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const finite = (value: unknown, maximum = Infinity): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= maximum;

function capture(candidate: Candidate): Candidate {
  const command = candidate.command === null ? null : structuredClone(candidate.command);
  if (command) {
    if ('ids' in command) Object.freeze(command.ids);
    if ('workerIds' in command && command.workerIds) Object.freeze(command.workerIds);
    Object.freeze(command);
  }
  return Object.freeze({ id: candidate.id, label: candidate.label, description: candidate.description, command });
}

export function layaErrorMessage(body: unknown): string | undefined {
  if (!isRecord(body)) return undefined;
  if (typeof body.detail === 'string' && body.detail.trim()) return body.detail;
  if (!Array.isArray(body.detail)) return undefined;
  // FastAPI validation details are objects, not strings. Report their messages,
  // without dumping request inputs or coercing them to "[object Object]".
  const messages = body.detail.flatMap(detail => {
    if (typeof detail === 'string') return detail.trim() ? [detail] : [];
    return isRecord(detail) && typeof detail.msg === 'string' ? [detail.msg] : [];
  });
  return messages.length ? messages.join('; ') : undefined;
}

export function validateLayaResponse(body: unknown, candidates: ReadonlyMap<string, unknown>, matchId: string, tick: number): LayaResponse {
  if (!isRecord(body)) throw new Error('Laya returned an invalid decision object');
  if (body.match_id !== matchId || body.tick !== tick)
    throw new Error('Laya returned a decision for a different match or tick');
  if (typeof body.choice !== 'string' || !candidates.has(body.choice))
    throw new Error('Laya selected an unknown candidate');
  if (!isRecord(body.probabilities) || Object.keys(body.probabilities).length !== candidates.size)
    throw new Error('Laya probabilities do not match the requested candidates');
  const entries: [string, number][] = [];
  for (const id of candidates.keys()) {
    const probability = body.probabilities[id];
    if (!Object.hasOwn(body.probabilities, id) || !finite(probability, 1))
      throw new Error('Laya returned a missing or invalid candidate probability');
    entries.push([id, probability]);
  }
  const probabilities = Object.fromEntries(entries);
  // Native Laya rounds to four decimals. Match the server's tolerance and retain
  // its probabilities rather than silently renormalizing the evidence.
  if (Math.abs(entries.reduce((sum, [, p]) => sum + p, 0) - 1) > 0.02)
    throw new Error('Laya probabilities do not sum to one');
  if (probabilities[body.choice] < Math.max(...entries.map(([, p]) => p)))
    throw new Error('Laya choice does not agree with its probabilities');
  if (body.confidence_kind !== 'normalized_entropy' || !finite(body.confidence, 1))
    throw new Error('Laya returned invalid normalized entropy confidence');
  if (!finite(body.inference_ms) || !finite(body.queue_ms))
    throw new Error('Laya returned invalid inference timing');
  if (!isRecord(body.model)) throw new Error('Laya returned invalid model metadata');
  return {
    choice: body.choice, probabilities, confidence: body.confidence,
    confidence_kind: 'normalized_entropy', inference_ms: body.inference_ms,
    queue_ms: body.queue_ms, match_id: matchId, tick, model: { ...body.model },
  };
}

/** One native Laya request at a time. The caller remains responsible for applying its returned order. */
export class LayaCommander {
  private generation = 0;
  private active: AbortController | null = null;

  constructor(private readonly fetcher: typeof fetch = globalThis.fetch.bind(globalThis), private readonly timeoutMs = 20_000) {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Laya timeout must be a positive number');
  }

  get busy(): boolean { return this.active !== null; }

  cancel(): void {
    this.generation++;
    const active = this.active;
    this.active = null;
    active?.abort();
  }

  async decide(world: World, team: Team, matchId: string, weights?: PolicyModel): Promise<LayaDecision | null> {
    if (this.busy || world.status !== 'playing') return null;
    const candidates = layaCandidates(world, team, weights).map(capture);
    if (candidates.length <= 1) return null;
    const byId: ReadonlyMap<string, Candidate> = new Map(candidates.map(candidate => [candidate.id, candidate]));
    const requestedTick = world.tick, requestedTime = world.time;
    const body = JSON.stringify({
      state: compactObservation(world, team),
      candidates: candidates.map(({ id, description }) => ({ id, description })),
      match_id: matchId, tick: requestedTick,
    });
    const generation = ++this.generation, controller = new AbortController();
    this.active = controller;
    let timedOut = false;
    let onAbort!: () => void;
    const aborted = new Promise<never>((_, reject) => {
      onAbort = () => reject(new Error('Laya request canceled'));
      controller.signal.addEventListener('abort', onAbort, { once: true });
    });
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, this.timeoutMs);
    try {
      const request = (async () => {
        const response = await this.fetcher('/api/empires/decision', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body, signal: controller.signal,
        });
        let parsed: unknown;
        try { parsed = await response.json(); }
        catch {
          if (response.ok) throw new Error('Laya returned a response that is not valid JSON');
        }
        if (!response.ok) {
          const detail = layaErrorMessage(parsed);
          throw new Error(detail ?? `Laya request failed (HTTP ${response.status})`);
        }
        return parsed;
      })();
      // The race covers both headers and body parsing, even for a fetcher that
      // ignores AbortSignal. Cancellation must not strand the next match.
      const result: unknown = await Promise.race([request, aborted]);
      if (generation !== this.generation) return null;
      const response = validateLayaResponse(result, byId, matchId, requestedTick);
      const candidate = byId.get(response.choice)!;
      const ageSeconds = world.time - requestedTime;
      if (world.status !== 'playing' || ageSeconds > 15 || ageSeconds < 0 || world.tick < requestedTick) return null;
      if (candidate.command !== null && !world.canCommand(team, candidate.command).ok) return null;
      return { candidate, requestedTick, response, ageSeconds };
    } catch (error) {
      if (generation !== this.generation) return null;
      if (timedOut) throw new Error(`Laya request timed out after ${this.timeoutMs / 1000} seconds`);
      throw error instanceof Error ? error : new Error('Laya request failed');
    } finally {
      clearTimeout(timer);
      controller.signal.removeEventListener('abort', onAbort);
      if (generation === this.generation) this.active = null;
    }
  }
}
