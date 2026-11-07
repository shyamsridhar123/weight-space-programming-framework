'use strict';

const steps = ['Define the goal', 'Compare edits', 'Protect behavior', 'Make the decision', 'Understand the impact', 'Restore the parent'];
const names = { 'keep-parent': 'Unchanged parent', 'target-only': 'Target-only edit', 'soft-retention': 'Soft retention', 'protected-edit': 'Protected edit' };
const queues = { general: 'General Support', billing: 'Billing', security: 'Security', platform: 'Platform Operations', manual_review: 'Manual review' };
const concepts = [
  '<strong>Task contract.</strong> Declare what should change, what must be preserved, and how success will be evaluated.',
  '<strong>Candidate admission.</strong> Better target behavior is only one obligation. Utility, protection, structure, and evidence binding must all pass.',
  '<strong>Weight space programming.</strong> Compile intent into a literal edit, then check the endpoint that the program actually produces.',
  '<strong>Sealed evaluation.</strong> Select using public evidence, seal that candidate, and evaluate it on the declared final split.',
  '<strong>Scoped impact.</strong> Explain the observed behavior and the operational opportunity separately.',
  '<strong>Exact-parent rollback.</strong> Apply a bound inverse to the edited artifact and verify its identity against the parent.'
];
let bundle = null;
let step = 0;
let candidateLabel = 'protected-edit';
let restored = null;
let busy = false;
let evidenceUrl = null;
const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const number = (value) => value === 0 ? '0' : Number(value).toFixed(6);
const queue = (value) => queues[value] || value;
const arrow = '<svg class="arrow" viewBox="0 0 28 20" aria-hidden="true"><path d="M1 10h23M17 3l7 7-7 7" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>';
const record = (label = candidateLabel) => bundle.candidates.find(row => row.label === label);
const admitted = () => bundle.final_admission.decision === 'admitted_for_synthetic_trial';
const failureNames = {
  target_score_shift: 'Requested score shift missed', target_routing: 'Target routing failed',
  protected_runtime_scores: 'Protected scores changed', protected_routes: 'Protected routes changed',
  protected_exact_retention: 'Exact protection not satisfied', certificate_binding: 'Receipt binding failed',
  structural_policy: 'Structural policy failed', no_publicly_eligible_edit: 'No publicly eligible edit'
};

function identities() {
  const final = bundle.final_admission;
  return `<details><summary>Inspect bound evidence</summary><dl>
    <dt>Exact parent artifact</dt><dd>${esc(final.parent_sha256)}</dd>
    <dt>Task contract</dt><dd>${esc(final.contract_sha256)}</dd>
    <dt>Final admission receipt</dt><dd>${esc(final.sha256)}</dd>
    <dt>Cooperative session trace head</dt><dd>${esc(final.trace_head_sha256)}</dd>
    </dl><p class="small">Hashes bind these records to specific artifacts. Cooperative traces do not establish process isolation or independent attestation.</p></details>`;
}

function policyView() {
  const conflict = bundle.scenario === 'conflict';
  const metrics = record('keep-parent').admission.metrics;
  const target = metrics.tickets.find(ticket => ticket.role === 'target');
  return `<h1>Change one routing policy.<br>Protect the rest.</h1>
    <p class="lead">Telemetry outage tickets should reach Platform Operations. The existing model sends them to General Support.</p>
    <div class="columns">
      <section class="policy"><h2>What should change</h2><p>Recognized telemetry outages move to the team that owns the platform.</p>
        <p class="small">Structured match: general topic + telemetry product + outage symptom.</p>
        <div class="formula">Desired Platform score shift: +3</div></section>
      <section class="policy"><h2>What must be preserved</h2><p>General Support, Billing, Security, and existing Platform behavior.</p>
        <p class="small">Preserve all queue scores on the supplied protection anchors and their span, under the fixed-feature linear model.</p>
        <div class="formula">Protected score change: 0</div></section>
    </div>
    ${conflict ? '<div class="decision warning"><h2>This request conflicts with protection</h2><p>The target is also protected. Changing its output and preserving it exactly cannot both hold.</p></div>' : `<div class="route-change"><p class="small">A synthetic telemetry outage before the edit</p><span class="route">${esc(queue(target.before_queue))}</span>${arrow}<span class="route new">Desired: Platform Operations</span></div>`}
    <p class="small">Acceptance also requires target shift error ≤ 0.05 score units, a winning margin ≥ 0.5, correct protected queues, structural admission, and a verified retention receipt. Scores are not probabilities.</p>`;
}

function compareView() {
  const row = record();
  const a = row.admission;
  const headings = bundle.scenario === 'conflict' ? 'Sometimes the correct answer is no edit.' : 'A useful edit must pass every obligation.';
  return `<h1>${headings}</h1>
    <p class="lead">The compiler fits three alternatives against the same parent. Target-only fitting can improve the requested route while disturbing protected scores.</p>
    <div class="table-wrap"><table><thead><tr><th>Candidate</th><th class="numeric">Target shift error</th><th class="numeric">Protected drift</th><th>Exact retention</th><th>Eligible</th></tr></thead><tbody>
      ${bundle.candidates.map(item => { const m = item.admission.metrics; return `<tr class="${item.label === candidateLabel ? 'selected' : ''}"><td><button class="candidate-button" data-candidate="${item.label}" aria-pressed="${item.label === candidateLabel}">${names[item.label]}</button></td><td class="numeric">${number(m.maximum_target_shift_error)}</td><td class="numeric">${number(m.maximum_protected_score_drift)}</td><td class="${item.retention_certificate.exactly_zero_on_supplied_features ? 'pass' : 'fail'}">${item.retention_certificate.exactly_zero_on_supplied_features ? 'Pass' : 'Fail'}</td><td class="${item.admission.eligible ? 'pass' : 'fail'}">${item.admission.eligible ? 'Yes' : 'No'}</td></tr>`; }).join('')}
    </tbody></table></div>
    <p class="small" style="margin-top:12px">Raw score units; lower errors are better. Target error ≤ 0.05 and sampled protected drift ≤ 10⁻¹². Exact retention is a separate endpoint check.</p>
    <div class="review-detail"><strong>${names[candidateLabel]}:</strong> ${a.eligible ? 'Meets the public contract. The selection rule chooses this candidate.' : esc(a.failures.map(name => failureNames[name] || name).join('; ')) + '.'}</div>
    <p class="small" style="margin-top:15px">Only publicly eligible edits can be selected. The rule minimizes changed cells, then uses the program hash to break ties. Fitting uses ridge regularization, so target error need not be zero.</p>`;
}

function protectView() {
  const protectedRecord = record('protected-edit');
  const edits = protectedRecord.program.edits;
  const changed = new Map(edits.map(edit => [edit.index.join(','), edit]));
  const weights = bundle.parent.tensors['head.weight'];
  const label = ['General', 'Billing', 'Security', 'Platform'];
  const features = ['General', 'Billing', 'Security', 'Platform', 'Enterprise', 'Urgency', 'Telemetry<br>outage'];
  return `<h1>${edits.length ? 'One coefficient carries the policy change.' : 'Full protection leaves no useful direction.'}</h1>
    <p class="lead">With frozen features, the protected edit uses a direction that is zero on the protection anchors. The realized endpoint is checked after numerical compilation.</p>
    <div class="table-wrap"><table class="matrix"><caption class="small" style="text-align:left;padding-bottom:8px">Output-head weights · rows are queues, columns are input features</caption><thead><tr><th>Queue</th>${features.map(feature => `<th>${feature}</th>`).join('')}</tr></thead><tbody>
    ${weights.map((values, i) => `<tr><td>${label[i]}</td>${values.map((value, j) => `<td class="${changed.has(`${i},${j}`) ? 'changed' : ''}">${changed.has(`${i},${j}`) ? Number(changed.get(`${i},${j}`).after).toFixed(4) : value}</td>`).join('')}</tr>`).join('')}
    </tbody></table></div>
    <div class="decision ${edits.length ? '' : 'warning'}"><h2>${edits.length ? 'Protected behavior stays fixed' : 'The framework abstains'}</h2>
    <p>${edits.length ? `Platform / telemetry_outage: <strong>${edits[0].before}</strong> becomes <strong>${edits[0].after}</strong>. The other weights and bias are unchanged.` : 'The target joins the protected set. Those anchors span every editable feature direction, so exact preservation forces a zero update.'}</p></div>
    <p class="small">The exact certificate covers the supplied fixed features and their span under real-valued linear semantics. The run also observes NumPy float64 scores; it does not prove floating-point execution or a text extractor.</p>
    ${identities()}`;
}

function decisionView() {
  const final = bundle.final_admission;
  const result = bundle.session.result;
  const abstain = final.decision === 'abstained';
  const gap = final.decision === 'rejected_after_holdout';
  const title = abstain ? 'No eligible edit. The session abstains.' : gap ? 'A public pass can still fail the final gate.' : 'Seal the candidate. Then evaluate.';
  return `<h1>${title}</h1>
    <p class="lead">Selection uses public feedback. The selected candidate is sealed before final evaluation; the final result cannot choose a different edit.</p>
    <ul class="check-list">
      <li><span>Public contract</span><strong class="${abstain ? 'fail' : 'pass'}">${abstain ? 'No eligible candidate' : 'Protected edit eligible'}</strong></li>
      <li><span>Sealed selection</span><strong>${abstain ? 'Abstention' : 'Protected edit'}</strong></li>
      <li><span>Declared final split</span><strong class="${abstain || gap ? 'fail' : 'pass'}">${abstain ? 'Evaluator not called' : gap ? 'Behavioral gate failed' : 'Behavioral gate passed'}</strong></li>
    </ul>
    <div class="decision ${admitted() ? '' : 'warning'}"><h2>${admitted() ? 'Admitted for a synthetic trial' : abstain ? 'Abstained' : 'Rejected after final evaluation'}</h2>
      <p>${admitted() ? 'Recognized outages reach Platform Operations; protected queues and sampled scores are unchanged.' : abstain ? 'The request cannot meet its public contract. No candidate is sent to the final evaluator.' : 'Final targets use intermittent_errors. The extractor recognizes outage only, so the new feature stays zero and routing does not change.'}</p></div>
    ${result ? `<div class="table-wrap"><table><thead><tr><th>Final target</th><th>Parent route</th><th>Sealed route</th><th class="numeric">Winner margin</th></tr></thead><tbody>${result.tickets.filter(ticket => ticket.role === 'target').map((ticket, i) => `<tr><td>${i === 0 ? 'Enterprise tier' : 'Standard tier'}</td><td>${esc(queue(ticket.before_queue))}</td><td class="${admitted() ? 'pass' : 'fail'}">${esc(queue(ticket.after_queue))}</td><td class="numeric">${number(ticket.after_margin)}</td></tr>`).join('')}</tbody></table></div>` : ''}
    <p class="small" style="margin-top:16px">${abstain ? 'The session stops before final evaluation.' : 'The final rows have disjoint declared synthetic identifiers. They are constructed public fixtures, not independent confirmation.'} No production rollout is authorized.</p>`;
}

function impactView() {
  const result = bundle.session.result;
  if (!admitted()) {
    return `<h1>A blocked rollout is an informative outcome.</h1><p class="lead">${bundle.scenario === 'conflict' ? 'The framework exposes an impossible contract before final evaluation.' : 'Final evaluation exposes a missing feature before the synthetic trial is admitted.'}</p>
      <div class="columns"><section class="policy"><h2>Observed in this run</h2><p>${bundle.scenario === 'conflict' ? 'No edit meets the protection contract. The session abstains.' : 'The sealed edit fails on intermittent_errors. The final gate rejects it.'}</p></section><section class="policy"><h2>What the team learns</h2><p>${bundle.scenario === 'conflict' ? 'Revise the objective or obtain explicit approval to change the protection requirement.' : 'Expand and validate the feature mapping, then start a new contract and evaluation cycle.'}</p></section></div>
      <h2>Potential operational value</h2><p>Expose unsuitable changes early and give reviewers concrete evidence for the next decision. This demo does not measure incident prevention, costs, or customer outcomes.</p>`;
  }
  const target = result.tickets.find(ticket => ticket.role === 'target');
  return `<h1>The policy changes. Existing behavior holds.</h1>
    <p class="lead">The observed result is a targeted routing change with preserved protected scores in the declared synthetic evaluation.</p>
    <div class="route-change"><span class="route">${esc(queue(target.before_queue))}</span>${arrow}<span class="route new">${esc(queue(target.after_queue))}</span><p class="small">Final telemetry outage targets · winner margin ${number(result.minimum_target_margin)} raw score units</p></div>
    <div class="impact-grid"><section><h2>Measured in the demo</h2><ul>
      <li>Recognized final outage targets reach Platform.</li>
      <li>Protected final routes remain correct and unchanged.</li>
      <li>Maximum sampled protected score drift: <strong>${number(result.maximum_protected_score_drift)}</strong>.</li>
      <li>The edit and decision bind to exact artifacts.</li>
    </ul></section><section><h2>Potential enterprise value</h2><p>Make policy updates reviewable, preserve agreed behavior, and restore a known model if needed.</p><p class="small">Resolution time, savings, customer outcomes, and production performance have not been measured.</p></section></div>
    <p class="small" style="margin-top:22px">The generic pattern is artifact → program → realized candidate → admission → sealed evaluation. This example supplies one domain adapter and contract.</p>`;
}

function rollbackView() {
  if (!admitted()) return `<h1>No trial was admitted.</h1><p class="lead">${bundle.scenario === 'conflict' ? 'The session abstained without selecting an edit.' : 'The sealed candidate was rejected by final evaluation.'} The demo has no admitted model to restore.</p><div class="decision warning"><h2>Return to the policy change</h2><p>Choose Policy change and run the framework to explore the exact-parent rollback.</p></div>`;
  return `<h1>Restore the exact parent model.</h1><p class="lead">The inverse program is bound to the edited artifact. Applying it restores the parent tensors and metadata, then checks the artifact identity.</p>
    <div class="formula">Platform / telemetry_outage: 3.22029702970297 ${arrow} 0.25</div>
    ${restored ? `<div class="decision restored"><h2>Exact parent restored</h2><p>The restored artifact hash matches the parent. Re-evaluating the restored model sends the synthetic outage targets to General Support.</p></div><details open><summary>Restoration receipt</summary><dl><dt>Restored artifact SHA-256</dt><dd>${esc(restored.restored_parent_sha256)}</dd><dt>Original parent SHA-256</dt><dd>${esc(restored.parent_sha256)}</dd></dl></details>` : '<div class="decision"><h2>Ready to apply the inverse</h2><p>Run the actual rollback in Python and inspect the result.</p></div><button id="restore" class="primary">Restore parent model</button>'}
    <h2>Scope of the restoration</h2><p>Rollback restores the model artifact. It does not undo ticket assignments, external side effects, or unrelated model revisions.</p><p class="small">The local interface demonstrates application to a freshly reconstructed admitted candidate. It does not control a live support system.</p>`;
}

function render() {
  document.querySelectorAll('[data-step]').forEach(button => {
    if (Number(button.dataset.step) === step) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
    button.disabled = !bundle || busy;
  });
  $('#back').hidden = step === 0;
  $('#back').disabled = !bundle || busy;
  $('#next').disabled = !bundle || busy;
  $('#next').textContent = step === 5 ? 'Return to the goal' : steps[step + 1];
  $('#concept').innerHTML = concepts[step];
  if (!bundle) {
    $('#view').innerHTML = '<h1>One policy change.<br>Every obligation visible.</h1><p class="lead">Run a synthetic support-routing example using the actual Weight Space Programming Framework.</p><div class="loading">Choose a scenario and run the framework to inspect its evidence.</div>';
    return;
  }
  $('#view').innerHTML = [policyView, compareView, protectView, decisionView, impactView, rollbackView][step]();
  document.querySelectorAll('[data-candidate]').forEach(button => button.addEventListener('click', () => {
    candidateLabel = button.dataset.candidate; render();
    $(`[data-candidate="${candidateLabel}"]`).focus({ preventScroll: true });
  }));
  $('#restore')?.addEventListener('click', restore);
}

async function request(endpoint) {
  const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario: $('#scenario').value }) });
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || 'The run did not complete. Run the framework again.');
  return data.result;
}

function setBusy(value) {
  busy = value;
  $('#run').disabled = value;
  $('#scenario').disabled = value;
}

async function run() {
  setBusy(true);
  $('#error').hidden = true;
  $('#run-status').textContent = 'Running compiler and evaluation…';
  bundle = null; restored = null; step = 0; candidateLabel = 'protected-edit';
  $('#download').hidden = true;
  render();
  try {
    bundle = await request('/api/run');
    if (evidenceUrl) URL.revokeObjectURL(evidenceUrl);
    evidenceUrl = URL.createObjectURL(new Blob([JSON.stringify(bundle, null, 2) + '\n'], { type: 'application/json' }));
    $('#download').href = evidenceUrl;
    $('#download').download = `${bundle.scenario}-evidence.json`;
    $('#download').hidden = false;
    $('#run-status').textContent = 'Python run completed · explore the receipts';
  } catch (error) {
    $('#error').textContent = `${error.message} Make sure the local Python server is running, then try again.`;
    $('#error').hidden = false;
    $('#run-status').textContent = 'Run did not complete';
  } finally { setBusy(false); render(); }
}

async function restore() {
  $('#restore').disabled = true;
  $('#restore').textContent = 'Applying inverse…';
  $('#error').hidden = true;
  setBusy(true);
  try {
    restored = await request('/api/rollback');
    $('#run-status').textContent = 'Python rollback verified · exact parent restored';
  } catch (error) {
    $('#error').textContent = error.message;
    $('#error').hidden = false;
  } finally { setBusy(false); render(); }
}

function navigate(value) {
  if (!bundle || busy) return;
  step = value;
  render();
  $('#main').focus({ preventScroll: true });
  if (window.innerWidth <= 700) $('#main').scrollIntoView({ behavior: 'auto', block: 'start' });
}
document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => navigate(Number(button.dataset.step))));
$('#back').addEventListener('click', () => navigate(Math.max(0, step - 1)));
$('#next').addEventListener('click', () => navigate(step === 5 ? 0 : step + 1));
$('#run').addEventListener('click', run);
$('#scenario').addEventListener('change', run);
render();
run();
