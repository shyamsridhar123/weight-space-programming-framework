import type { AstraCycle, PlayAstraLoop } from './play-astra';
import { PLAY_COACH_GOALS, type PlayCoachGoal, type PlayCoachPerformance, type PlayCoachScalarEdit } from './play-coach';
import { getPlayExperimentMetrics, type PlayExperimentResult } from './play-experiment';
import type { World } from './simulation';

export interface PlayAstraPanelHooks {
  world(): World;
  available(): boolean;
  paused(): boolean;
  changed(): void;
}

const escape = (value: unknown): string => String(value).replace(/[&<>"']/g,
  character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]!));
const number = (value: number, digits = 0): string => Number.isFinite(value) ? value.toFixed(digits) : '—';
const clock = (seconds: number): string => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const shortId = (id: string): string => id.length > 38 ? `${id.slice(0, 20)}…${id.slice(-12)}` : id;
const evidenceId = /^astra_[0-9a-f]{32}_(request|response|receipt)$/;
const element = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing Astra panel element: ${id}`);
  return found as T;
};
const setText = (id: string, value: string): void => { element(id).textContent = value; };
const setHtml = (id: string, value: string): void => {
  const node = element(id);
  // Avoid replacing unchanged content while readers select text or follow evidence links.
  if (node.innerHTML !== value) node.innerHTML = value;
};

function safeEvidenceLink(id: string): string {
  const label = `<code title="${escape(id)}">${escape(shortId(id))}</code>`;
  return evidenceId.test(id)
    ? `<a href="/api/empires/astra/evidence/${encodeURIComponent(id)}" target="_blank" rel="noopener noreferrer">${label}</a>`
    : label;
}

function address(edit: PlayCoachScalarEdit): string {
  return `${edit.tensor.startsWith('w') ? edit.tensor.toUpperCase() : edit.tensor}${edit.row === null ? '' : `[${edit.row}]`}${edit.column === null ? '' : `[${edit.column}]`}`;
}

function metric(label: string, value: string, detail: string): string {
  return `<div><span>${escape(label)}</span><strong>${escape(value)}</strong><small>${escape(detail)}</small></div>`;
}

function target(goal: PlayCoachGoal): string {
  if (goal === 'grow-army') return 'Target: 4 completed soldiers per game minute until 8 soldiers are alive.';
  if (goal === 'grow-economy') return 'Target: food and wood delivered, active workers, and enough housing.';
  if (goal === 'defend') return 'Target: preserve the Town Center and villagers; respond to visible threats.';
  return 'Target: damage the visible rival, keep soldiers alive, and protect home.';
}

/** Displays measured telemetry and model receipts; it never creates a patch or advances the game. */
export class PlayAstraPanel {
  private selected = false;
  private actionError = '';

  constructor(private readonly loop: PlayAstraLoop, private readonly hooks: PlayAstraPanelHooks) {
    element('astra-connect').addEventListener('click', () => this.act(() => this.loop.refreshConnection()));
    element('astra-start').addEventListener('click', () => this.act(() => {
      this.loop.start();
      return this.loop.refreshConnection();
    }));
    element('astra-stop').addEventListener('click', () => this.act(() => this.loop.stop()));
    element('astra-review').addEventListener('click', () => this.act(() => this.loop.reviewNow()));
    element<HTMLSelectElement>('astra-goal').addEventListener('change', event => {
      const goal = (event.currentTarget as HTMLSelectElement).value;
      if (Object.hasOwn(PLAY_COACH_GOALS, goal)) this.act(() => this.loop.setGoal(goal as PlayCoachGoal));
    });
  }

  private act(action: () => void | Promise<void>): void {
    this.actionError = '';
    let result: void | Promise<void> = undefined;
    try { result = action(); }
    catch (error) { this.actionError = error instanceof Error ? error.message : 'The action could not complete.'; }
    this.changed();
    if (result) void result.catch(error => {
      this.actionError = error instanceof Error ? error.message : 'The action could not complete.';
    }).finally(() => this.changed());
  }

  private changed(): void {
    this.hooks.changed();
    this.render(this.selected);
  }

  render(selected: boolean): void {
    this.selected = selected;
    const { loop } = this;
    const available = this.hooks.available() && this.hooks.world().status === 'playing';
    const paused = this.hooks.paused();
    const busy = loop.phase === 'request' || loop.phase === 'apply';
    element<HTMLButtonElement>('astra-start').disabled = !available || loop.active;
    element<HTMLButtonElement>('astra-stop').disabled = !loop.active;
    element<HTMLButtonElement>('astra-connect').disabled = loop.checkingConnection;
    element<HTMLButtonElement>('astra-review').disabled = !available || !loop.active || !loop.report?.ready || busy || paused || loop.phase === 'measure';
    element('astra-review').title = !loop.report?.ready
      ? 'Collect a complete measured window before requesting an Astra review.'
      : 'Ask Astra to review the current measurements. Requests are spaced at least 20 real seconds apart.';
    element<HTMLSelectElement>('astra-goal').value = loop.goal;
    this.renderConnection();
    const status = element('astra-status');
    status.textContent = this.actionError || (paused && loop.active ? `Game paused. ${loop.message}` : loop.message);
    status.dataset.state = this.actionError || loop.phase === 'error' ? 'error' : loop.phase;
    element('program-astra').setAttribute('aria-busy', String(busy));
    for (const phase of document.querySelectorAll<HTMLElement>('[data-astra-phase]')) {
      const active = phase.dataset.astraPhase === loop.phase;
      phase.dataset.state = active ? 'active' : 'waiting';
      if (active) phase.setAttribute('aria-current', 'step');
      else phase.removeAttribute('aria-current');
    }
    const report = loop.report;
    const progress = report
      ? `${number(report.elapsedSeconds)}/${number(report.minimumSeconds)} game seconds observed${report.revision === null ? '' : ` · policy r${report.revision}`}.`
      : 'Reviews use 30 game seconds of actual play.';
    setText('astra-window', `${target(loop.goal)} ${!loop.active && report ? `Last observation at ${clock(report.endTime)}. ` : ''}${progress}`);
    this.renderMetrics(report);
    const cycle = loop.latest;
    const showTrigger = cycle && !cycle.measurementCanceled && (loop.phase === 'measure' || (!report?.ready && cycle.appliedRevision !== null));
    setText('astra-observation', showTrigger
      ? `Shortfall sent to Astra: ${cycle.before.summary}`
      : report?.summary ?? 'No performance shortfall measured yet. Start the AI to collect observations.');
    this.renderCycle(loop.lastApplied ?? cycle);
    setHtml('astra-history', loop.events.length ? `<ol>${loop.events.map(event =>
      `<li><time>${clock(event.time)}</time> <strong>${escape(event.label)}</strong><p>${escape(event.detail)}</p></li>`).join('')}</ol>`
      : '<p class="program-empty">No feedback events yet.</p>');
    if (selected) this.renderFieldGuide(paused);
  }

  private renderConnection(): void {
    const { loop } = this, connection = loop.connection;
    const node = element('astra-connection');
    if (loop.checkingConnection) {
      node.textContent = 'Checking the Astra service…'; node.dataset.state = 'checking';
    } else if (loop.connectionError) {
      node.textContent = loop.connectionError; node.dataset.state = 'unavailable';
    } else if (connection?.verified) {
      node.textContent = `${connection.requested_model} · real generation verified`;
      node.dataset.state = 'connected';
    } else if (connection?.configured) {
      node.textContent = `${connection.requested_model} configured · generation not verified yet`;
      node.dataset.state = 'configured';
    } else {
      node.textContent = connection ? 'Astra is not configured.' : 'Astra connection not checked';
      node.dataset.state = connection ? 'unavailable' : 'unchecked';
    }
    node.title = connection?.last_error ?? '';
  }

  private renderMetrics(report: PlayCoachPerformance | null): void {
    if (!report && !this.hooks.available()) {
      setHtml('astra-metrics', '<p class="program-empty">Start the match to see actual blue army, recruitment, and resource measurements.</p>');
      return;
    }
    const world = this.hooks.world();
    const current = getPlayExperimentMetrics(world, 0);
    const own = world.entities.filter(entity => entity.owner === 0 && entity.hp > 0);
    const queued = own.reduce((count, entity) => count + entity.queue.filter(job => job.kind !== 'villager').length, 0);
    const firstRecruits = own.flatMap(entity => {
      let remaining = 0;
      for (const job of entity.queue) {
        remaining += Math.max(0, job.remaining);
        if (job.kind !== 'villager') return [remaining];
      }
      return [];
    });
    const nextRecruit = firstRecruits.length ? Math.min(...firstRecruits) : null;
    const interval = report ? `${number(report.elapsedSeconds)}s window ending ${clock(report.endTime)}` : 'measurement not started';
    setHtml('astra-metrics', [
      metric('Army alive', `${current.army}${this.loop.goal === 'grow-army' ? ' / 8' : ''}`, 'living soldiers now'),
      metric('Recruitment', report?.elapsedSeconds ? `${number(report.recruitsPerMinute, 1)} / min` : '—', report ? `${report.recruitsCompleted} completed · through ${clock(report.endTime)}` : 'waiting for a play window'),
      metric('Training now', String(queued), nextRecruit === null ? 'soldiers in the queue' : `next in ${number(nextRecruit, 1)}s`),
      metric('Food delivered', report ? number(report.returned.food) : '—', interval),
      metric('Wood delivered', report ? number(report.returned.wood) : '—', interval),
      metric('Town health', number(current.townHealth), 'live Town Center HP'),
    ].join(''));
  }

  private renderCycle(cycle: AstraCycle | null): void {
    this.renderLatestReview(cycle);
    element('astra-reasoning').hidden = !cycle;
    if (!cycle) {
      element<HTMLDetailsElement>('astra-reasoning').open = false;
      setText('astra-edit-summary', this.loop.phase === 'request'
        ? 'Astra is reading the observations and current weights. A validated response will appear here.'
        : 'After a measured shortfall, Astra receives the game observations and current policy weights.');
      setText('astra-explanation', '');
      setText('astra-prediction', '');
      setText('astra-decision-change', 'No Astra edit applied yet.');
      setHtml('astra-patches', '<p class="program-empty">The exact generated edits will appear here: coordinate · old value → new value.</p>');
      setHtml('astra-after', '<p class="program-empty">Subsequent gameplay will show whether the edit helped the chosen goal.</p>');
      setHtml('astra-comparison', '<p class="program-empty">No paired comparison yet.</p>');
      setHtml('astra-request-info', '<p class="program-empty">This match has no completed Astra response yet. A configured connection is not proof of generation.</p>');
      return;
    }
    const { proposal } = cycle;
    const kept = cycle.appliedRevision === null;
    setText('astra-edit-summary', kept ? `Latest review kept policy r${proposal.base_revision}. No Astra edit has been applied.`
      : `Most recent applied edit · r${proposal.base_revision} → r${cycle.appliedRevision} · ${clock(cycle.appliedTime!)}`);
    setText('astra-reasoning-title', kept ? 'Why Astra kept the weights' : 'Why Astra wrote this edit');
    setText('astra-explanation', proposal.explanation);
    setText('astra-prediction', `Astra’s prediction: ${proposal.expected_effect}`);
    const sameChoice = cycle.beforeChoice === cycle.afterChoice;
    setText('astra-decision-change', kept ? `Choice at that review: ${cycle.afterChoice}.`
      : sameChoice ? `At application, the next choice stayed “${cycle.afterChoice}”. Changed weights do not always change the winning order.`
        : `Same game state · before: ${cycle.beforeChoice} → after: ${cycle.afterChoice}. Applied as policy r${cycle.appliedRevision}.`);
    setHtml('astra-patches', proposal.patch.length
      ? `<table aria-label="Exact weight values generated by Astra"><thead><tr><th scope="col">Weight</th><th scope="col">Before</th><th scope="col">After</th></tr></thead><tbody>${proposal.patch.map(edit =>
        `<tr><th scope="row"><code>${escape(address(edit))}</code></th><td><code>${escape(edit.before)}</code></td><td><code>${escape(edit.value)}</code></td></tr>`).join('')}</tbody></table>`
      : '<p>Astra returned a keep decision. No parameter changed and no new weight revision was created.</p>');
    this.renderAfter(cycle);
    this.renderComparison(cycle);
    setHtml('astra-request-info', this.requestInfo(cycle));
  }

  private renderLatestReview(displayed: AstraCycle | null): void {
    const latest = this.loop.latest;
    const keptAfterEdit = latest && displayed && latest !== displayed && latest.appliedRevision === null;
    element('astra-latest-review').hidden = !keptAfterEdit;
    element('astra-keep-review').hidden = !keptAfterEdit;
    if (!keptAfterEdit) {
      element<HTMLDetailsElement>('astra-keep-review').open = false;
      setText('astra-latest-review', '');
      setHtml('astra-keep-explanation', '');
      return;
    }
    const sameWeights = latest.proposal.base_revision === displayed.appliedRevision
      && latest.proposal.base_fingerprint === displayed.appliedFingerprint;
    setText('astra-latest-review', sameWeights
      ? `Newest review: Astra kept r${latest.proposal.base_revision}. The applied edit and its results remain below.`
      : `Newest review: Astra kept r${latest.proposal.base_revision}. Below is the most recent Astra edit, r${displayed.appliedRevision}.`);
    setHtml('astra-keep-explanation', `<p>${escape(latest.proposal.explanation)}</p><p><strong>Astra’s prediction:</strong> ${escape(latest.proposal.expected_effect)}</p>${this.requestInfo(latest)}`);
  }

  private requestInfo(cycle: AstraCycle): string {
    const { proposal } = cycle;
    const kept = cycle.appliedRevision === null;
    const receipt = /^\/api\/empires\/astra\/evidence\/(astra_[0-9a-f]{32}_receipt)$/.exec(proposal.evidence_url)?.[1];
    const context = kept ? `Evidence for the review that kept r${proposal.base_revision}. It made no parameter change.`
      : `Evidence for the displayed edit: r${proposal.base_revision} → r${cycle.appliedRevision}, applied at ${clock(cycle.appliedTime!)}.`;
    return `<p class="program-note">${context}</p><dl><dt>Requested model</dt><dd>${escape(proposal.requested_model)}</dd><dt>Returned model</dt><dd>${escape(proposal.returned_model)}</dd><dt>Provider response</dt><dd><code title="${escape(proposal.response_id)}">${escape(shortId(proposal.response_id))}</code></dd><dt>Response time</dt><dd>${number(proposal.latency_ms / 1000, 1)} seconds</dd><dt>Request evidence</dt><dd>${safeEvidenceLink(proposal.request_evidence_id)}</dd><dt>Response evidence</dt><dd>${safeEvidenceLink(proposal.response_evidence_id)}</dd>${receipt ? `<dt>Validated receipt</dt><dd>${safeEvidenceLink(receipt)}</dd>` : ''}<dt>Input policy</dt><dd>r${proposal.base_revision} · <code>${escape(proposal.base_fingerprint)}</code></dd><dt>Applied policy</dt><dd>${kept ? 'Unchanged' : `r${cycle.appliedRevision} at ${clock(cycle.appliedTime!)}`}</dd></dl><p class="program-note">These records identify the actual model request and returned program. Astra’s prediction is a hypothesis; the gameplay measurements test its effect.</p>`;
  }

  private renderAfter(cycle: AstraCycle): void {
    if (cycle.appliedRevision === null) {
      setHtml('astra-after', '<p>No new weights were applied. The existing policy continues; this review has no weight-change effect to measure.</p>');
      return;
    }
    const first = cycle.firstOrder;
    const order = first
      ? `<p><strong>First actual order:</strong> ${escape(first.label)} at ${clock(first.time)} using r${first.revision}.</p>`
      : cycle.measurementCanceled
        ? '<p>No order was recorded before this observation window was interrupted.</p>'
        : '<p>Waiting for the first real order from the edited policy.</p>';
    const after = cycle.after;
    if (!after) {
      const note = cycle.measurementCanceled
        ? 'The live observation window was interrupted before a result was recorded. No improvement conclusion can be drawn.'
        : 'The post-edit observation window has not produced a measurement yet.';
      setHtml('astra-after', `${order}<p class="program-note">${note}</p>`);
      return;
    }
    const label = cycle.measurementCanceled ? 'Interrupted live result' : cycle.measurementComplete ? 'Measured live result' : this.loop.active ? 'Measuring live result' : 'Last recorded live result';
    const note = cycle.measurementCanceled
      ? 'This observation window was interrupted before completion. These partial measurements do not establish an improvement.'
      : cycle.measurementComplete
        ? 'Before and after use different time windows and game states. The paired comparison below isolates the weights at one saved state.'
        : 'This is an incomplete observation window. No improvement conclusion yet.';
    setHtml('astra-after', `${order}<p><strong>${label} · ${number(after.elapsedSeconds)}/${number(after.minimumSeconds)}s:</strong> ${after.recruitsCompleted} soldiers completed; army ${after.baseline.army} → ${after.current.army}. Villagers delivered ${number(after.returned.food)} food and ${number(after.returned.wood)} wood.</p><p>Recruitment ${number(cycle.before.recruitsPerMinute, 1)} → ${number(after.recruitsPerMinute, 1)} soldiers/min. Town Center lost ${number(after.townDamage)} HP; ${after.armyLost} soldiers and ${after.villagersLost} villagers lost.</p><small>${note}</small>`);
  }

  private renderComparison(cycle: AstraCycle): void {
    const result = cycle.comparison;
    if (!result) { setHtml('astra-comparison', `<p>${escape(cycle.comparisonMessage)}</p>`); return; }
    const a = result.baseline.metrics, b = result.candidate.metrics, start = result.initialMetrics;
    const rows: [string, number, number][] = [
      ['Soldiers completed', a.recruitsCompleted - start.recruitsCompleted, b.recruitsCompleted - start.recruitsCompleted],
      ['Army alive', a.army, b.army],
      ['Food delivered', a.gathered.food - start.gathered.food, b.gathered.food - start.gathered.food],
      ['Wood delivered', a.gathered.wood - start.gathered.wood, b.gathered.wood - start.gathered.wood],
      ['Villagers alive', a.villagers, b.villagers],
      ['Own town HP', a.townHealth, b.townHealth],
      ['Rival town HP', result.baseline.otherMetrics.townHealth, result.candidate.otherMetrics.townHealth],
    ];
    setHtml('astra-comparison', `<p>${escape(cycle.comparisonMessage)}</p><table aria-label="Isolated gameplay comparison with unchanged and Astra-edited weights"><thead><tr><th scope="col">Outcome</th><th scope="col">Unchanged</th><th scope="col">Astra edit</th></tr></thead><tbody>${rows.map(([label, before, after]) => `<tr><th scope="row">${escape(label)}</th><td>${number(before)}</td><td>${number(after)}</td></tr>`).join('')}</tbody></table><p>${this.comparisonChoice(result)}</p><small>Separate simulated continuations from ${clock(result.startTime)}. Unchanged ran ${number(result.baseline.elapsedSeconds, 1)}s; edited ran ${number(result.candidate.elapsedSeconds, 1)}s, up to ${result.requestedSeconds}s or match end. These figures are not the live battle and do not guarantee future improvement.</small>`);
  }

  private comparisonChoice(result: PlayExperimentResult): string {
    const first = result.firstChangedChoice;
    return first ? `First different choice at +${number(first.elapsedSeconds, 1)}s: ${escape(first.baseline?.label ?? 'No available order')} → ${escape(first.candidate?.label ?? 'No available order')}.`
      : 'No different policy choice occurred during this comparison.';
  }

  private renderFieldGuide(paused: boolean): void {
    const { loop } = this;
    let title: string, copy: string;
    if (loop.active && paused) {
      title = 'Game paused.';
      copy = loop.phase === 'apply' ? 'Astra returned an edit. Resume to apply it on the next game tick.' : 'Resume the game to continue observing the policy’s orders and outcomes.';
    } else if (!loop.active) {
      title = loop.phase === 'error' ? 'Astra’s review needs attention.' : loop.events.length ? 'Astra is stopped.' : 'AI plays. Astra programs its weights.';
      copy = loop.events.length || loop.phase === 'error' ? loop.message : 'Start AI + Astra. Blue plays; a measured shortfall triggers numerical policy edits. Watch the exact change in the panel.';
    } else if (loop.phase === 'request') {
      title = 'Astra is writing new policy weights.';
      copy = 'The game policy keeps playing. Astra is reviewing the observed shortfall and the current network; no edit is applied until its response is validated.';
    } else if (loop.phase === 'apply') {
      title = 'Astra’s numerical edit is ready.';
      copy = 'The validated weights will apply together on the next game tick. Blue’s next decisions then use the new policy revision.';
    } else if (loop.phase === 'measure') {
      title = 'New weights. Watch blue’s next orders.';
      copy = loop.latest?.firstOrder ? `First actual order: ${loop.latest.firstOrder.label}. Measuring completed recruitment, resource deliveries, and losses after the edit.`
        : 'The edit is applied. Waiting for the next policy decision, then measuring what happens in the game.';
    } else {
      title = 'Blue is playing. Performance is measured.';
      copy = `${target(loop.goal)} A measured shortfall asks Astra to write a correction.`;
    }
    setText('wsp-field-step', title);
    setText('wsp-field-copy', copy);
  }
}
