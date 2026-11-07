import {
  PlayExperiment, type PlayExperimentId, type PlayExperimentWorkerReply, type PlayExperimentWorkerRequest,
} from './play-experiment';

export interface PlayExperimentWorkerPort {
  postMessage(message: PlayExperimentWorkerReply): void;
  addEventListener(type: 'message', listener: (event: { data: unknown }) => void): void;
  removeEventListener?(type: 'message', listener: (event: { data: unknown }) => void): void;
}

/** A new run replaces its predecessor. Each batch yields so cancellation stays responsive. */
export function installPlayExperimentWorker(
  port: PlayExperimentWorkerPort,
  schedule: (callback: () => void) => void = callback => { setTimeout(callback, 0); },
): () => void {
  let active: { id: PlayExperimentId; experiment: PlayExperiment } | null = null;
  let disposed = false;

  const cancel = (): void => {
    if (!active) return;
    const id = active.id;
    active = null;
    if (!disposed) port.postMessage({ type: 'cancelled', id });
  };

  const onMessage = (event: { data: unknown }): void => {
    if (disposed || !event.data || typeof event.data !== 'object') return;
    const message = event.data as PlayExperimentWorkerRequest;
    if (!((typeof message.id === 'number' && Number.isFinite(message.id))
      || (typeof message.id === 'string' && message.id.length > 0))) return;
    if (message.type === 'cancel') {
      if (active?.id === message.id) cancel();
      return;
    }
    if (message.type !== 'run') return;
    cancel();
    try {
      const job = { id: message.id, experiment: new PlayExperiment(message.request) };
      active = job;
      port.postMessage({ type: 'progress', id: job.id, progress: job.experiment.progress });
      const batch = (): void => {
        if (disposed || active !== job) return;
        try {
          const done = job.experiment.advance(20);
          port.postMessage({ type: 'progress', id: job.id, progress: job.experiment.progress });
          if (done) {
            const result = job.experiment.result();
            active = null;
            port.postMessage({ type: 'result', id: job.id, result });
          } else schedule(batch);
        } catch (error) {
          active = null;
          port.postMessage({ type: 'error', id: job.id, message: error instanceof Error ? error.message : 'The paired futures could not finish.' });
        }
      };
      schedule(batch);
    } catch (error) {
      active = null;
      port.postMessage({ type: 'error', id: message.id, message: error instanceof Error ? error.message : 'The paired futures could not start.' });
    }
  };

  port.addEventListener('message', onMessage);
  return () => {
    disposed = true;
    active = null;
    port.removeEventListener?.('message', onMessage);
  };
}

// Small structural port avoids requiring WebWorker globals in the DOM-based app build.
const scope = globalThis as unknown as PlayExperimentWorkerPort;
if (!('document' in globalThis) && typeof scope.postMessage === 'function' && typeof scope.addEventListener === 'function')
  installPlayExperimentWorker(scope);
