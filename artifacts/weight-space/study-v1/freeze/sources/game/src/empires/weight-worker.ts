import { runWeightExperiment, type RunWeightExperimentOptions, type WeightExperimentConfig, type WeightExperimentProgress, type WeightExperimentResult } from './weight-experiments';
import type { PolicyModel } from './policy';

export type WeightWorkerRequest =
  | { type: 'start'; runId: string; baseRevision: number; baseModel: PolicyModel; config?: Partial<WeightExperimentConfig> }
  | { type: 'cancel'; runId: string };

export type WeightWorkerReply =
  | { type: 'progress'; runId: string; progress: WeightExperimentProgress }
  | { type: 'result'; runId: string; result: WeightExperimentResult }
  | { type: 'cancelled'; runId: string }
  | { type: 'error'; runId: string; message: string };

interface MessageEvent { data: unknown; }
export interface WeightWorkerPort {
  postMessage(message: WeightWorkerReply): void;
  addEventListener(type: 'message', listener: (event: MessageEvent) => void): void;
  removeEventListener?(type: 'message', listener: (event: MessageEvent) => void): void;
}

/** One job per worker. A cancellation reply is sent only once computation has actually stopped. */
export function installWeightExperimentWorker(
  port: WeightWorkerPort,
  run: (options: RunWeightExperimentOptions) => Promise<WeightExperimentResult> = runWeightExperiment,
): () => void {
  let active: { runId: string; controller: AbortController } | null = null;
  let disposed = false;

  const onMessage = (event: MessageEvent): void => {
    if (disposed || !event.data || typeof event.data !== 'object') return;
    const message = event.data as WeightWorkerRequest;
    if (typeof message.runId !== 'string' || !message.runId || message.runId.length > 128) return;
    if (message.type === 'cancel') {
      if (active?.runId === message.runId) active.controller.abort();
      return;
    }
    if (message.type !== 'start') return;
    if (active) {
      port.postMessage({ type: 'error', runId: message.runId, message: 'A weight experiment is already running. Cancel it and wait for cancellation before starting another.' });
      return;
    }
    const job = { runId: message.runId, controller: new AbortController() };
    active = job;
    void (async () => {
      let reply: WeightWorkerReply;
      try {
        const result = await run({
          baseModel: message.baseModel, baseRevision: message.baseRevision, config: message.config, signal: job.controller.signal,
          onProgress: progress => {
            if (!disposed && !job.controller.signal.aborted) port.postMessage({ type: 'progress', runId: job.runId, progress });
          },
        });
        reply = job.controller.signal.aborted ? { type: 'cancelled', runId: job.runId } : { type: 'result', runId: job.runId, result };
      } catch (error) {
        reply = job.controller.signal.aborted || (error instanceof Error && error.name === 'AbortError')
          ? { type: 'cancelled', runId: job.runId }
          : { type: 'error', runId: job.runId, message: error instanceof Error ? error.message : 'Weight experiment failed.' };
      }
      if (active === job) active = null;
      if (!disposed) port.postMessage(reply);
    })();
  };

  port.addEventListener('message', onMessage);
  return () => {
    disposed = true;
    active?.controller.abort();
    port.removeEventListener?.('message', onMessage);
  };
}

// DOM types cover the main app; this small port also permits worker imports in Node tests.
const scope = globalThis as unknown as WeightWorkerPort;
if (!('document' in globalThis) && typeof scope.postMessage === 'function' && typeof scope.addEventListener === 'function')
  installWeightExperimentWorker(scope);
