import './style.css';
import { World, AGE_NAMES, AGE_COSTS, AGE_TIMES, BUILDING_DEFS, UNIT_DEFS, UPGRADE_DEFS, RESOURCES, DT } from './simulation';
import type { BuildingKind, Command, Entity, EntityKind, Point, Resource, Resources, Team, UnitKind, Upgrade } from './types';
import { EmpiresView } from './view';
import { buildCandidates, chooseNeural, chooseRule, compactObservation, MODEL_METADATA, rankActions } from './policy';
import type { Candidate } from './policy';
import { LayaCommander } from './laya';
import { WeightLab } from './weight-lab';
import { WeightWorkbench, weightIdentity } from './weight-workbench';

const element = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const canvas = element<HTMLCanvasElement>('battlefield');
const minimap = element<HTMLCanvasElement>('minimap');
const roman = ['', 'I', 'II', 'III', 'IV'];
const symbols: Record<EntityKind | 'weapons' | 'armor' | 'economy', string> = {
  towncenter: '♜', house: '⌂', farm: '≋', barracks: '⚔', range: '➶', stable: '♞', tower: '♖', forge: '⚒', castle: '♜',
  villager: '♙', spearman: '♟', archer: '➶', knight: '♞', ram: '▰', weapons: '⚔', armor: '♢', economy: '☷',
};
const resourceSymbols: Record<Resource, string> = { food: '♧', wood: '♠', gold: '◆', stone: '⬟' };
type Pilot = 'manual' | 'neural' | 'rule' | 'laya';
type Tab = 'context' | 'build' | 'tech';
type MatchEvent = { tick: number; team: Team; source: string; action: string; command: Command | null; observation: string; latencyMs?: number; model?: unknown };
let world = new World(42);
let selected = new Set<number>([world.entities.find(e => e.owner === 0 && e.kind === 'towncenter')!.id]);
let selectedNode: number | null = null;
let view: EmpiresView;
let pilot: Pilot = 'manual';
let tab: Tab = 'context';
let started = false;
let paused = false;
let helpWasPaused = false;
let speed = 1;
let placement: BuildingKind | null = null;
let pointOrder: 'move' | 'attackMove' | null = null;
let pointer: Point | null = null;
let toastDeadline = 0;
let hudAge = 0;
let lastEvent = world.events.at(-1);
let actionSignature = '';
let matchId = `empires-${Date.now()}`;
let matchStartedAt: string | null = null;
let records: MatchEvent[] = [];
let lastPolicyAt = 0;
let nextLayaAt = 0;
let accumulator = 0;
let lastFrame = performance.now();
let finished = false;
let renderFailed = false;
let workbench: WeightWorkbench | null = null;
let weightLab: WeightLab | null = null;
let labAge = 0;
const heldKeys = new Set<string>();
const laya = new LayaCommander();
let layaGeneration = 0;
const actionHandlers = new Map<string, () => void>();

function cancelLaya() { layaGeneration++; laya.cancel(); }

const escapeHTML = (text: unknown): string => String(text).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]!));
const costLabel = (cost: Partial<Resources>): string => RESOURCES.filter(r => cost[r]).map(r => `${cost[r]}${resourceSymbols[r]}`).join(' ');
const fullCost = (cost: Partial<Resources>): string => RESOURCES.filter(r => cost[r]).map(r => `${cost[r]} ${r}`).join(', ');
const canAfford = (cost: Partial<Resources>) => RESOURCES.every(r => world.players[0].resources[r] >= (cost[r] ?? 0));
const clock = (time: number) => `${Math.floor(time / 60).toString().padStart(2, '0')}:${Math.floor(time % 60).toString().padStart(2, '0')}`;
const entities = () => [...selected].map(id => world.entity(id)).filter((e): e is Entity => !!e);
const ownUnits = () => entities().filter(e => e.owner === 0 && e.category === 'unit');
const workers = () => ownUnits().filter(e => e.kind === 'villager');

function toast(message: string, duration = 3600) {
  element('toast').textContent = message; element('toast').hidden = false; toastDeadline = performance.now() + duration;
}
function centerHome() {
  const home = world.entities.find(e => e.owner === 0 && e.kind === 'towncenter' && e.hp > 0);
  if (home) view.focus(home.x + 2, home.z - 1);
}
function cancelPlacement() {
  placement = null; pointOrder = null; view.preview(null);
  element('placement-hint').hidden = true; canvas.style.cursor = 'default'; actionSignature = '';
}
function setPilot(value: Pilot) {
  pilot = value; cancelLaya(); workbench?.cancelCoach('Changing commander cancelled the pending weight suggestion.'); nextLayaAt = world.time; cancelPlacement();
  element<HTMLSelectElement>('pilot').value = value;
  element('pilot-note').textContent = value === 'manual' ? 'Your orders shape the kingdom. Council advice is optional.'
    : value === 'neural' ? `1,293 editable weights · r${workbench?.current.revision ?? 0}. Open the Weight Lab to change the next decision.`
      : value === 'rule' ? 'A fixed decision tree for comparison. No neural inference.'
        : 'Existing Laya checkpoint; RTS ability unvalidated. The world keeps running during inference.';
  element('decision-status').textContent = value === 'laya' ? 'AWAITING LAYA' : value === 'manual' ? 'COUNCIL ADVICE' : 'COMMANDER ACTIVE';
  element('decision-label').textContent = value === 'laya' ? 'Connecting on the next strategic turn…' : 'Your council is assessing the kingdom.';
  element('decision-ranks').replaceChildren(); element('decision-latency').textContent = '—';
  document.body.dataset.pilot = value;
  if (value !== 'laya') displayAdvice();
}
function chooseTab(value: Tab) {
  tab = value; actionSignature = '';
  for (const name of ['context', 'build', 'tech'] as const) {
    element(`tab-${name}`).classList.toggle('active', value === name);
    element(`tab-${name}`).setAttribute('aria-selected', String(value === name));
  }
  renderActions();
}
function select(ids: number[], additive = false) {
  if (!additive) selected.clear();
  for (const id of ids) if (world.entity(id)) selected.add(id);
  selectedNode = null; actionSignature = ''; renderHUD();
}
function record(team: Team, source: string, candidate: Candidate, observation: string, latencyMs?: number, model?: unknown) {
  records.push({ tick: world.tick, team, source, action: candidate.id, command: candidate.command ? structuredClone(candidate.command) : null, observation, latencyMs, model });
}
function issue(command: Command, label: string, point?: Point) {
  if (!started || world.status !== 'playing') return false;
  const observation = compactObservation(world, 0);
  const result = world.command(0, command);
  if (!result.ok) { toast(result.reason ?? 'That order is unavailable.'); return false; }
  if (pilot !== 'manual') { setPilot('manual'); toast('You have taken command.'); }
  record(0, 'manual', { id: command.type, label, description: label, command }, observation);
  if (point) view.ping(point, command.type === 'attack' || command.type === 'move' && !!command.attackMove);
  element('decision-status').textContent = 'YOUR LAST ORDER';
  element('decision-label').textContent = label;
  renderHUD(); return true;
}

function start(watch = false) {
  const input = element<HTMLInputElement>('seed');
  const value = Number(input.value);
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) { input.reportValidity(); toast('Choose a whole-number map seed from 0 to 4294967295.'); return; }
  cancelLaya(); world = new World(value); view.setWorld(world);
  selected = new Set([world.entities.find(e => e.owner === 0 && e.kind === 'towncenter')!.id]); selectedNode = null;
  started = true; paused = false; finished = false; speed = 1; accumulator = 0;
  records = []; matchId = `empires-${Date.now()}-${value}`; matchStartedAt = new Date().toISOString();
  workbench?.beginMatch();
  lastPolicyAt = 0; lastEvent = world.events.at(-1); nextLayaAt = 0; heldKeys.clear();
  for (const id of ['welcome', 'help-modal', 'outcome', 'paused-banner']) element(id).hidden = true;
  element('pause').textContent = 'Ⅱ'; element('pause').setAttribute('aria-label', 'Pause game');
  element('speed').textContent = '1×'; element('seed-label').textContent = `SEED ${world.seed}`;
  setPilot(watch ? 'neural' : 'manual'); chooseTab('context'); centerHome(); renderHUD();
  toast(watch ? 'The neural commanders are in charge. Choose Manual to take over.' : 'Your charter begins. Train villagers at the Town Center, then build houses to grow.', 6000);
  canvas.focus({ preventScroll: true });
}

function displayRanks(ranked: { label: string; probability: number }[]) {
  element('decision-ranks').innerHTML = ranked.slice(0, 3).map(r => `<div class="rank"><div><span>${escapeHTML(r.label)}</span><span>${Math.round(r.probability * 100)}%</span></div><div class="rank-track"><i style="width:${Math.max(0, Math.min(100, r.probability * 100))}%"></i></div></div>`).join('');
}
function displayAdvice() {
  if (world.status !== 'playing' || pilot === 'laya') return;
  const before = performance.now();
  const ranked = rankActions(world, 0, workbench?.current.model);
  const choice = pilot === 'rule' ? chooseRule(world, 0) : ranked[0];
  element('decision-status').textContent = pilot === 'manual' ? `COUNCIL ADVICE · r${workbench?.current.revision ?? 0}` : pilot === 'rule' ? 'RULE BASELINE' : `NEURAL COMMANDER · r${workbench?.current.revision ?? 0}`;
  element('decision-latency').textContent = `${(performance.now() - before).toFixed(1)} ms`;
  element('decision-label').textContent = choice.description;
  if (pilot === 'rule') element('decision-ranks').replaceChildren(); else displayRanks(ranked);
}
function localDecision(team: Team, kind: 'neural' | 'rule') {
  const observation = compactObservation(world, team), before = performance.now();
  const snapshot = team === 0 ? workbench?.current : workbench?.opponent;
  const action = kind === 'neural' ? chooseNeural(world, team, snapshot?.model) : chooseRule(world, team);
  const latency = performance.now() - before;
  const result = action.command ? world.command(team, action.command) : { ok: true };
  if (result.ok) record(team, kind === 'neural' ? team === 0 ? 'editable-neural' : 'authored-neural' : 'rule-baseline', action, observation, latency,
    kind === 'neural' ? { ...MODEL_METADATA, label: team === 0 ? 'Editable auxiliary neural commander' : MODEL_METADATA.label,
      ...(snapshot ? { weights: weightIdentity(snapshot) } : {}) } : undefined);
  if (team === 0) {
    displayAdvice(); element('decision-label').textContent = action.description;
    element('decision-latency').textContent = `${latency.toFixed(1)} ms`;
  }
}
async function layaDecision() {
  if (laya.busy || workbench?.coachBusy || world.time < nextLayaAt || pilot !== 'laya' || paused) return;
  const requestedWorld = world, requestedMatch = matchId, requestedGeneration = layaGeneration;
  const requestedWeights = workbench?.current;
  const observation = compactObservation(world, 0);
  const candidates = buildCandidates(world, 0);
  element('decision-status').textContent = 'LAYA IS DELIBERATING';
  element('decision-label').textContent = 'The kingdom continues while your commander chooses a legal order.';
  element('decision-ranks').replaceChildren();
  nextLayaAt = world.time + 2;
  try {
    const result = await laya.decide(requestedWorld, 0, requestedMatch, requestedWeights?.model);
    if (world !== requestedWorld || matchId !== requestedMatch || layaGeneration !== requestedGeneration || pilot !== 'laya' || paused || world.status !== 'playing'
      || workbench?.current.revision !== requestedWeights?.revision || workbench?.current.fingerprint !== requestedWeights?.fingerprint) return;
    nextLayaAt = world.time + 2;
    if (!result) { element('decision-status').textContent = 'LAYA · WAITING'; element('decision-label').textContent = 'Continuing existing orders; the next turn will use a fresh observation.'; return; }
    const action = result.candidate;
    if (action.command && !world.canCommand(0, action.command).ok) { element('decision-label').textContent = 'The chosen order expired. Requesting a fresh decision.'; return; }
    const accepted = action.command ? world.command(0, action.command) : { ok: true };
    if (!accepted.ok) return;
    record(0, 'laya', action, observation, result.response.inference_ms, { ...result.response.model, requested_tick: result.requestedTick, applied_tick: world.tick,
      shortlist_weights: requestedWeights ? weightIdentity(requestedWeights) : null,
      confidence: result.response.confidence, confidence_kind: result.response.confidence_kind, queue_ms: result.response.queue_ms });
    element('decision-status').textContent = 'LAYA · EXPERIMENTAL';
    element('decision-latency').textContent = `${Math.round(result.response.inference_ms)} ms`;
    element('decision-label').textContent = action.description;
    displayRanks(Object.entries(result.response.probabilities).map(([id, probability]) => ({ label: candidates.find(c => c.id === id)?.label ?? id, probability })).sort((a, b) => b.probability - a.probability));
    element('pilot-note').textContent = `Checkpoint unvalidated for RTS. Entropy concentration ${Math.round(result.response.confidence * 100)}%; this is not accuracy.`;
  } catch (error) {
    if (world !== requestedWorld || matchId !== requestedMatch || layaGeneration !== requestedGeneration || pilot !== 'laya' || paused || world.status !== 'playing') return;
    nextLayaAt = world.time + 12;
    element('decision-status').textContent = 'LAYA CONNECTION NEEDS ATTENTION';
    const message = error instanceof Error ? error.message : 'The commander request failed.';
    element('decision-label').textContent = message;
    element('decision-latency').textContent = '—';
    element('pilot-note').textContent = 'Existing orders continue. Run scripts/start-empires.ps1 -WithLaya to connect, or choose another commander.';
  }
}

type ActionSpec = { id: string; name: string; glyph: string; cost?: Partial<Resources>; detail?: string; enabled?: boolean; hotkey?: string; selected?: boolean; run: () => void };
function renderActions() {
  const current = entities(), own = current.filter(e => e.owner === 0), first = own[0];
  const specs: ActionSpec[] = [];
  const add = (spec: ActionSpec) => specs.push(spec);
  if (tab === 'build') {
    const haveWorker = world.entities.some(e => e.owner === 0 && e.kind === 'villager' && e.hp > 0);
    for (const kind of Object.keys(BUILDING_DEFS) as BuildingKind[]) {
      const def = BUILDING_DEFS[kind];
      add({ id: `build-${kind}`, name: kind === 'towncenter' ? 'Town Center' : kind === 'range' ? 'Range' : kind === 'forge' ? 'Forge' : def.name, glyph: symbols[kind], cost: def.cost,
        detail: `${def.name} · ${fullCost(def.cost)} · ${def.time}s · ${AGE_NAMES[def.requiredAge]}${def.population ? ` · +${def.population} population` : ''}`,
        enabled: haveWorker && world.players[0].age >= def.requiredAge && canAfford(def.cost), selected: placement === kind,
        run: () => { cancelPlacement(); placement = kind; element('placement-hint').hidden = false; element('placement-name').textContent = `Place ${def.name}`; canvas.style.cursor = 'crosshair'; actionSignature = ''; if (pointer) updatePreview(); renderActions(); },
      });
    }
  } else if (tab === 'tech') {
    for (const upgrade of Object.keys(UPGRADE_DEFS) as Upgrade[]) {
      const def = UPGRADE_DEFS[upgrade], command: Command = { type: 'research', upgrade }, result = world.canCommand(0, command);
      add({ id: `research-${upgrade}`, name: world.players[0].upgrades.includes(upgrade) ? 'Researched' : def.name, glyph: symbols[upgrade], cost: def.cost,
        detail: `${def.name} · ${fullCost(def.cost)}${result.reason ? ` · ${result.reason}` : ''}`, enabled: result.ok,
        run: () => { if (issue(command, `${def.name} researched.`)) toast(`${def.name} is now active.`); },
      });
    }
  } else if (first?.category === 'building') {
    for (const kind of BUILDING_DEFS[first.kind as BuildingKind].trainable) {
      const command: Command = { type: 'train', buildingId: first.id, kind }, def = UNIT_DEFS[kind], result = world.canCommand(0, command);
      add({ id: `train-${kind}`, name: def.name, glyph: symbols[kind], cost: def.cost, hotkey: kind === 'villager' ? 'V' : '',
        enabled: result.ok, detail: `${def.name} · ${fullCost(def.cost)} · ${def.time}s${result.reason ? ` · ${result.reason}` : ''}`,
        run: () => { if (issue(command, `Training a ${def.name.toLowerCase()}.`)) toast(`${def.name} added to the training queue.`, 1800); },
      });
    }
    if (first.kind === 'farm' && first.construction >= 1) {
      const villager = world.entities.filter(e => e.owner === 0 && e.kind === 'villager').sort((a, b) => Math.hypot(a.x - first.x, a.z - first.z) - Math.hypot(b.x - first.x, b.z - first.z))[0];
      const command: Command = { type: 'gather', ids: villager ? [villager.id] : [], targetId: first.id };
      add({ id: 'work-farm', name: 'Assign villager', glyph: symbols.villager, enabled: world.canCommand(0, command).ok, run: () => issue(command, 'A nearby villager will harvest this farm.') });
    }
    add({ id: 'open-build', name: 'Build', glyph: '⌂', detail: 'Place a new building with a villager.', hotkey: 'B', run: () => chooseTab('build') });
    add({ id: 'open-tech', name: 'Research', glyph: '⚒', detail: 'Improve your economy and army.', run: () => chooseTab('tech') });
  } else if (ownUnits().length) {
    const ids = ownUnits().map(e => e.id);
    for (const kind of ['move', 'attackMove'] as const) add({ id: kind, name: kind === 'move' ? 'Move' : 'Attack move', glyph: kind === 'move' ? '↗' : '⚔', hotkey: kind === 'move' ? 'G' : 'F', selected: pointOrder === kind,
      run: () => { cancelPlacement(); pointOrder = kind; element('placement-hint').hidden = false; element('placement-name').textContent = kind === 'move' ? 'Choose a destination' : 'Advance and engage'; canvas.style.cursor = 'crosshair'; renderActions(); },
    });
    add({ id: 'stop', name: 'Stop', glyph: '⊠', hotkey: 'X', run: () => issue({ type: 'stop', ids }, 'Selected units are holding their position.') });
    if (workers().length) {
      add({ id: 'open-build', name: 'Build', glyph: '⌂', hotkey: 'B', run: () => chooseTab('build') });
      const firstWorker = workers()[0];
      const food = world.nodes.filter(n => n.kind === 'food' && n.amount > 0 && (n.owner === undefined || n.owner === 0) && world.isVisible(0, n.x, n.z)).sort((a, b) => Math.hypot(a.x - firstWorker.x, a.z - firstWorker.z) - Math.hypot(b.x - firstWorker.x, b.z - firstWorker.z))[0];
      add({ id: 'gather-food', name: 'Gather food', glyph: '♧', enabled: !!food, detail: 'Send selected villagers to the nearest visible food source.', run: () => { if (food) issue({ type: 'gather', ids: workers().map(e => e.id), targetId: food.id }, 'Villagers are gathering food.', food); } });
    }
  } else {
    add({ id: 'home-select', name: 'Town Center', glyph: '♜', hotkey: 'H', run: () => { const home = world.entities.find(e => e.owner === 0 && e.kind === 'towncenter'); if (home) { select([home.id]); centerHome(); } } });
    add({ id: 'open-build', name: 'Build', glyph: '⌂', hotkey: 'B', run: () => chooseTab('build') });
    add({ id: 'army-select', name: 'Select army', glyph: '⚔', hotkey: 'M', run: selectArmy });
  }
  actionHandlers.clear(); for (const spec of specs) actionHandlers.set(spec.id, spec.run);
  const signature = JSON.stringify([tab, specs.map(({ run: _, ...spec }) => spec)]);
  if (signature === actionSignature) return;
  actionSignature = signature;
  element('actions').classList.toggle('build-grid', tab === 'build');
  element('actions').innerHTML = specs.map(spec => `<button class="action-card${spec.selected ? ' selected' : ''}" data-action="${spec.id}" ${spec.enabled === false ? 'disabled' : ''} title="${escapeHTML(spec.detail ?? spec.name)}" aria-label="${escapeHTML(spec.name)}"><span class="glyph" aria-hidden="true">${spec.glyph}</span><span class="action-name">${escapeHTML(spec.name)}</span><span class="cost">${spec.cost ? costLabel(spec.cost) : 'ORDER'}</span>${spec.hotkey ? `<span class="hotkey">${spec.hotkey}</span>` : ''}</button>`).join('');
}

function renderHUD() {
  const player = world.players[0];
  for (const id of selected) {
    const entity = world.entity(id);
    if (!entity || entity.hp <= 0 || (entity.owner !== 0 && !world.isVisible(0, entity.x, entity.z))) selected.delete(id);
  }
  const current = entities();
  const ownWorkers = world.entities.filter(e => e.owner === 0 && e.kind === 'villager' && e.hp > 0);
  for (const resource of RESOURCES) {
    element(resource).textContent = Math.floor(player.resources[resource]).toLocaleString('en-US');
    element(`workers-${resource}`).textContent = String(ownWorkers.filter(e => e.order.type === 'gather' && e.order.resource === resource).length);
  }
  element('population').textContent = `${world.population(0)} / ${world.capacity(0)}`;
  element('clock').textContent = clock(world.time);
  element('age-label').textContent = `${roman[player.age]} · ${AGE_NAMES[player.age].toUpperCase()}`;
  const advance = element<HTMLButtonElement>('advance'), advancement = world.canCommand(0, { type: 'advance' });
  advance.disabled = !advancement.ok;
  advance.innerHTML = player.advancing > 0 ? `Advancing · ${Math.ceil(player.advancing)}s` : player.age === 4 ? 'Imperial' : 'Advance age <span>↗</span>';
  advance.title = player.age < 4 ? `${AGE_NAMES[player.age + 1]} · ${fullCost(AGE_COSTS[player.age + 1])}${advancement.reason ? ` · ${advancement.reason}` : ''}` : 'Your kingdom has reached the final age.';
  element('age-progress').style.width = `${player.advancing > 0 ? 100 * (1 - player.advancing / AGE_TIMES[player.age + 1]) : 0}%`;
  element('idle-count').textContent = String(ownWorkers.filter(e => e.order.type === 'idle').length);
  element('selection-count').textContent = selected.size > 1 ? `${selected.size} SELECTED` : '';
  const first = current[0], node = world.nodes.find(n => n.id === selectedNode && n.amount > 0 && world.isVisible(0, n.x, n.z));
  if (!node) selectedNode = null;
  let title = 'Your kingdom', subtitle = 'Select a unit or building', icon = '♛', hp = 0, maxHp = 0, order = 'AWAITING ORDERS';
  if (node) {
    title = node.kind === 'wood' ? 'Woodland' : node.kind === 'food' ? 'Berry grove' : node.kind === 'gold' ? 'Gold deposit' : 'Stone outcrop';
    subtitle = 'Right click with villagers to gather'; icon = resourceSymbols[node.kind]; hp = node.amount; maxHp = node.maxAmount; order = 'RESOURCE';
  } else if (first) {
    title = current.length > 1 ? `${current.length} ${current.every(e => e.kind === 'villager') ? 'villagers' : 'units'} selected` : (first.category === 'building' ? BUILDING_DEFS[first.kind as BuildingKind] : UNIT_DEFS[first.kind as UnitKind]).name;
    icon = current.length > 1 ? '⚔' : symbols[first.kind];
    hp = current.reduce((sum, e) => sum + e.hp, 0); maxHp = current.reduce((sum, e) => sum + e.maxHp, 0);
    subtitle = first.owner === 1 ? 'The Ashen Crown' : first.category === 'building' ? first.construction < 1 ? `Under construction · ${Math.round(first.construction * 100)}%` : first.queue.length ? `${first.queue.length} in the production queue` : 'House of the Blue Stag'
      : first.kind === 'villager' ? `${first.carrying ? `Carrying ${Math.floor(first.carrying)} ${first.carryResource}` : 'The hands that build an empire'}` : 'House of the Blue Stag';
    order = first.owner !== 0 ? 'IN SIGHT' : first.construction < 1 ? 'BUILDING' : first.category === 'building' ? first.queue.length ? 'TRAINING' : 'READY' : first.order.type === 'gather' ? `GATHER ${first.order.resource?.toUpperCase() ?? ''}` : first.order.type === 'attackMove' ? 'ON THE MARCH' : first.order.type.toUpperCase();
  }
  element('selection-name').textContent = title; element('selection-subtitle').textContent = subtitle; element('selection-icon').textContent = icon;
  element('selection-owner').textContent = node ? 'THE LAND’S RICHES' : first?.owner === 1 ? 'RIVAL KINGDOM' : 'YOUR KINGDOM';
  element('selection-health').style.width = `${maxHp ? Math.max(0, hp / maxHp * 100) : 0}%`;
  element('selection-hp').textContent = maxHp ? `${Math.ceil(hp)} / ${Math.ceil(maxHp)} ${node ? 'REMAINING' : 'HP'}` : 'YOUR CHARTER AWAITS';
  element('selection-order').textContent = order;
  element('production').innerHTML = first?.owner === 0 ? first.queue.map(job => `<div class="queue-slot" title="${UNIT_DEFS[job.kind].name} · ${Math.ceil(job.remaining)}s"><span>${UNIT_DEFS[job.kind].name.slice(0, 3).toUpperCase()}</span><i style="width:${100 * (1 - job.remaining / job.total)}%"></i></div>`).join('') : '';
  renderActions(); view.minimap(minimap, selected);
  document.body.dataset.matchStatus = !started ? 'ready' : paused ? 'paused' : world.status;
  document.body.dataset.tick = String(world.tick);
}

function selectArmy() {
  select(world.entities.filter(e => e.owner === 0 && e.category === 'unit' && e.kind !== 'villager' && e.hp > 0).map(e => e.id));
  chooseTab('context'); if (!selected.size) toast('Recruit soldiers at the Barracks first.');
}
function selectIdle() {
  const idle = world.entities.filter(e => e.owner === 0 && e.kind === 'villager' && e.order.type === 'idle' && e.hp > 0);
  if (!idle.length) { toast('All villagers have orders.'); return; }
  select(idle.map(e => e.id)); view.focus(idle[0].x, idle[0].z); chooseTab('context');
}
function updatePreview() {
  if (!placement || !pointer) { view.preview(null); return; }
  const site = { x: Math.round(pointer.x), z: Math.round(pointer.z) };
  const ids = workers().map(e => e.id);
  view.preview(placement, site, world.canCommand(0, { type: 'build', kind: placement, ...site, ...(ids.length ? { workerIds: ids } : {}) }).ok);
}
function place(point: Point) {
  if (placement) {
    const site = { x: Math.round(point.x), z: Math.round(point.z) }, ids = workers().map(e => e.id), kind = placement;
    if (issue({ type: 'build', kind, ...site, ...(ids.length ? { workerIds: ids } : {}) }, `Constructing a ${BUILDING_DEFS[kind].name.toLowerCase()}.`, site)) {
      cancelPlacement(); toast(`${BUILDING_DEFS[kind].name} foundation placed. A villager is on the way.`);
    }
  } else if (pointOrder) {
    if (issue({ type: 'move', ids: ownUnits().map(e => e.id), ...point, attackMove: pointOrder === 'attackMove' }, pointOrder === 'attackMove' ? 'Advance and engage enemies along the way.' : 'Moving to the chosen destination.', point)) cancelPlacement();
  }
}
function contextualOrder(clientX: number, clientY: number) {
  if (placement || pointOrder) { cancelPlacement(); return; }
  const hit = view.pick(clientX, clientY); if (!hit) return;
  const target = hit.entityId === undefined ? undefined : world.entity(hit.entityId);
  const node = hit.nodeId === undefined ? undefined : world.nodes.find(n => n.id === hit.nodeId);
  const units = ownUnits();
  if (!units.length) { toast('Select villagers or soldiers first.'); return; }
  if (target && target.owner !== 0) issue({ type: 'attack', ids: units.map(e => e.id), targetId: target.id }, 'Engage the selected enemy.', target);
  else if ((node || target?.kind === 'farm') && workers().length) issue({ type: 'gather', ids: workers().map(e => e.id), targetId: node?.id ?? target!.id }, `Gather ${node?.kind ?? 'food'} and return it to town.`, node ?? target);
  else issue({ type: 'move', ids: units.map(e => e.id), ...hit.point }, 'Move to the chosen destination.', hit.point);
}

let drag: { x: number; y: number; lastX: number; lastY: number; button: number; moved: boolean; shift: boolean; touch: boolean } | null = null;
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('pointerdown', event => {
  if (!started || finished) return;
  canvas.focus({ preventScroll: true }); canvas.setPointerCapture(event.pointerId);
  drag = { x: event.clientX, y: event.clientY, lastX: event.clientX, lastY: event.clientY, button: event.button, moved: false, shift: event.shiftKey, touch: event.pointerType === 'touch' };
});
canvas.addEventListener('pointermove', event => {
  pointer = view.ground(event.clientX, event.clientY); updatePreview();
  if (!drag) return;
  if (Math.hypot(event.clientX - drag.x, event.clientY - drag.y) > 6) drag.moved = true;
  if (drag.moved && (drag.button === 1 || drag.button === 2)) {
    view.pan(-(event.clientX - drag.lastX) * view.span / canvas.clientHeight, -(event.clientY - drag.lastY) * view.span / canvas.clientHeight * 1.4);
  } else if (drag.moved && drag.button === 0 && !placement && !pointOrder) {
    const box = element('selection-box'); box.hidden = false;
    Object.assign(box.style, { left: `${Math.min(drag.x, event.clientX)}px`, top: `${Math.min(drag.y, event.clientY)}px`, width: `${Math.abs(event.clientX - drag.x)}px`, height: `${Math.abs(event.clientY - drag.y)}px` });
  }
  drag.lastX = event.clientX; drag.lastY = event.clientY;
});
canvas.addEventListener('pointerup', event => {
  if (!drag) return;
  const state = drag; drag = null; element('selection-box').hidden = true;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  if (state.button === 2 && !state.moved) { contextualOrder(event.clientX, event.clientY); return; }
  if (state.button !== 0) return;
  const point = view.ground(event.clientX, event.clientY);
  if (placement || pointOrder) { if (point) place(point); return; }
  if (state.moved) {
    const ids = world.entities.filter(e => {
      if (e.owner !== 0 || e.category !== 'unit' || e.hp <= 0) return false;
      const p = view.screen(e.x, e.z, .6);
      return p.x >= Math.min(state.x, event.clientX) && p.x <= Math.max(state.x, event.clientX) && p.y >= Math.min(state.y, event.clientY) && p.y <= Math.max(state.y, event.clientY);
    }).map(e => e.id);
    select(ids, state.shift); chooseTab('context');
  } else {
    const hit = view.pick(event.clientX, event.clientY);
    if (state.touch && ownUnits().length && hit && (hit.nodeId !== undefined || hit.entityId === undefined || world.entity(hit.entityId)?.owner === 1)) { contextualOrder(event.clientX, event.clientY); return; }
    if (hit?.entityId !== undefined) { select([hit.entityId], state.shift); chooseTab('context'); }
    else if (hit?.nodeId !== undefined) { selected.clear(); selectedNode = hit.nodeId; renderHUD(); }
    else if (!state.shift) select([]);
  }
});
canvas.addEventListener('pointercancel', () => { drag = null; element('selection-box').hidden = true; });
canvas.addEventListener('wheel', event => { event.preventDefault(); view.zoom(event.deltaY); updatePreview(); }, { passive: false });
minimap.addEventListener('pointerdown', event => {
  const rect = minimap.getBoundingClientRect(); view.focus((event.clientX - rect.left) / rect.width * 64, (event.clientY - rect.top) / rect.height * 64); renderHUD();
});

function pauseToggle() {
  if (!started || finished || !element('help-modal').hidden) return;
  paused = !paused; cancelLaya(); workbench?.cancelCoach('Changing pause state cancelled the pending Laya suggestion.'); heldKeys.clear(); accumulator = 0;
  element('paused-banner').hidden = !paused;
  element('pause').textContent = paused ? '▷' : 'Ⅱ'; element('pause').setAttribute('aria-label', paused ? 'Resume game' : 'Pause game'); renderHUD();
}
function openHelp() {
  if (!started || finished || !element('help-modal').hidden) return;
  helpWasPaused = paused; paused = true; heldKeys.clear(); cancelLaya(); workbench?.cancelCoach(); weightLab?.setOpen(false); element('help-modal').hidden = false;
  element('close-help').focus(); renderHUD();
}
function closeHelp() {
  element('help-modal').hidden = true; paused = helpWasPaused; accumulator = 0;
  element('paused-banner').hidden = !paused;
  element('pause').textContent = paused ? '▷' : 'Ⅱ'; element('pause').setAttribute('aria-label', paused ? 'Resume game' : 'Pause game');
  canvas.focus(); renderHUD();
}
function showRenderError(message: string) {
  renderFailed = true; started = false; paused = true; accumulator = 0;
  cancelLaya(); workbench?.dispose(); weightLab?.setOpen(false); heldKeys.clear(); drag = null; placement = null; pointOrder = null;
  for (const id of ['welcome', 'help-modal', 'outcome', 'paused-banner', 'placement-hint', 'selection-box']) element(id).hidden = true;
  canvas.style.cursor = 'default';
  element('render-error-detail').textContent = message;
  element('render-error').hidden = false; element('reload').focus();
}
function showOutcome() {
  if (finished || world.status === 'playing') return;
  finished = true; cancelLaya(); workbench?.cancelCoach('The match ended before the weight suggestion was applied.'); weightLab?.setOpen(false); cancelPlacement(); heldKeys.clear();
  element('outcome').hidden = false;
  element('outcome-title').textContent = world.status === 'won' ? 'A crown claimed.' : world.status === 'lost' ? 'The charter falls.' : 'Two crowns endure.';
  element('outcome-detail').textContent = world.status === 'won' ? 'The rival Town Center has fallen. The Riverlands answer to the Blue Stag.' : world.status === 'lost' ? 'Your last Town Center has fallen. A new seed is another chance to build an empire.' : 'Neither kingdom holds the field.';
  element('outcome-stats').innerHTML = `<div><b>${clock(world.time)}</b><span>TIME IN THE FIELD</span></div><div><b>${roman[world.players[0].age]}</b><span>AGE REACHED</span></div><div><b>${world.players[0].kills}</b><span>ENEMIES DEFEATED</span></div>`;
  element('play-again').focus(); renderHUD();
}
function exportMatch() {
  const data = { schema: 'crown-charter-match-v2', match_id: matchId, seed: world.seed, started_at: matchStartedAt, exported_at: new Date().toISOString(), status: world.status,
    seconds: world.time, ticks: world.tick, note: 'Debug match record includes both sides. Commands reproduce deterministic play at their recorded ticks; this is not an RTS training dataset or a rendered replay.',
    controllers: { blue: pilot, red: 'authored-neural' }, policy: MODEL_METADATA, weights: workbench?.exportEvidence(), decisions: records, final_state: world.snapshot(), events: world.events, events_note: 'Only the most recent 80 simulation notifications are retained. The decisions array contains the full command history; weight history limits are stated in weights.' };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }), url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = `${matchId}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); toast('Match record exported, including both sides and policy provenance.');
}

element('start').addEventListener('click', () => start());
element('watch').addEventListener('click', () => start(true));
element('pause').addEventListener('click', pauseToggle);
element('help').addEventListener('click', openHelp);
element('close-help').addEventListener('click', closeHelp);
element('speed').addEventListener('click', () => { speed = speed === 1 ? 2 : speed === 2 ? 4 : 1; element('speed').textContent = `${speed}×`; });
element('home').addEventListener('click', centerHome);
element('zoom-in').addEventListener('click', () => view.zoom(-180));
element('zoom-out').addEventListener('click', () => view.zoom(180));
element('select-army').addEventListener('click', selectArmy);
element('idle-workers').addEventListener('click', selectIdle);
element('advance').addEventListener('click', () => { if (issue({ type: 'advance' }, `Advancing toward ${AGE_NAMES[world.players[0].age + 1]}.`)) toast('A new age is on the horizon. Your settlement keeps working.'); });
for (const name of ['context', 'build', 'tech'] as const) element(`tab-${name}`).addEventListener('click', () => chooseTab(name));
element('actions').addEventListener('click', event => { const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-action]'); if (button && !button.disabled) actionHandlers.get(button.dataset.action!)?.(); });
element<HTMLSelectElement>('pilot').addEventListener('change', event => setPilot((event.target as HTMLSelectElement).value as Pilot));
element('commander-toggle').addEventListener('click', () => { const content = element('commander-content'); content.hidden = !content.hidden; element('commander-toggle').setAttribute('aria-expanded', String(!content.hidden)); element('commander-chevron').textContent = content.hidden ? '+' : '−'; });
element('weight-lab-toggle').addEventListener('click', () => weightLab?.setOpen(!weightLab.isOpen));
for (const id of ['export', 'export-result']) element(id).addEventListener('click', exportMatch);
for (const id of ['restart', 'play-again']) element(id).addEventListener('click', () => { cancelLaya(); workbench?.cancelCoach(); weightLab?.setOpen(false); started = false; paused = false; heldKeys.clear(); cancelPlacement(); for (const modal of ['help-modal', 'outcome', 'paused-banner']) element(modal).hidden = true; element('welcome').hidden = false; element('start').focus(); });
element('reload').addEventListener('click', () => location.reload());
window.addEventListener('keydown', event => {
  const modal = document.querySelector<HTMLElement>('.modal:not([hidden])');
  if (modal) {
    if (event.code === 'Escape' && modal.id === 'help-modal') { event.preventDefault(); closeHelp(); }
    if (event.code === 'Tab') {
      const controls = [...modal.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), a[href], [tabindex="0"]')];
      const first = controls[0], last = controls.at(-1);
      if (!modal.contains(document.activeElement) || event.shiftKey && document.activeElement === first || !event.shiftKey && document.activeElement === last) {
        event.preventDefault(); (event.shiftKey ? last : first)?.focus();
      }
    }
    return;
  }
  if ((event.target as HTMLElement).closest('input,select,textarea,button,[contenteditable="true"],#weight-lab-host') || !started || finished) return;
  if (['Space', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.code)) event.preventDefault();
  if (event.repeat) return;
  if (event.code === 'Escape') { if (placement || pointOrder) cancelPlacement(); else if (!element('help-modal').hidden) closeHelp(); else pauseToggle(); return; }
  if (!element('help-modal').hidden) return;
  heldKeys.add(event.code);
  if (event.code === 'Space') pauseToggle();
  if (event.code === 'KeyH') { centerHome(); const home = world.entities.find(e => e.owner === 0 && e.kind === 'towncenter'); if (home) { select([home.id]); chooseTab('context'); } }
  if (event.code === 'KeyI') selectIdle();
  if (event.code === 'KeyM') selectArmy();
  if (event.code === 'KeyB') chooseTab('build');
  if (event.code === 'KeyV') actionHandlers.get('train-villager')?.();
  if (event.code === 'KeyF' && ownUnits().length) { chooseTab('context'); actionHandlers.get('attackMove')?.(); }
  if (event.code === 'KeyG' && ownUnits().length) { chooseTab('context'); actionHandlers.get('move')?.(); }
  if (event.code === 'KeyX' && ownUnits().length) issue({ type: 'stop', ids: ownUnits().map(e => e.id) }, 'Selected units are holding their position.');
});
window.addEventListener('keyup', event => heldKeys.delete(event.code));
window.addEventListener('blur', () => { heldKeys.clear(); drag = null; element('selection-box').hidden = true; });
document.addEventListener('visibilitychange', () => { lastFrame = performance.now(); accumulator = 0; heldKeys.clear(); });
window.addEventListener('pagehide', () => { workbench?.dispose(); cancelLaya(); });
window.addEventListener('empires-render-error', event => showRenderError((event as CustomEvent<string>).detail));

function frame(now: number) {
  if (renderFailed) return;
  requestAnimationFrame(frame);
  const dt = Math.min(.15, (now - lastFrame) / 1000); lastFrame = now;
  if (started && element('help-modal').hidden && !finished) {
    const horizontal = Number(heldKeys.has('KeyD') || heldKeys.has('ArrowRight')) - Number(heldKeys.has('KeyA') || heldKeys.has('ArrowLeft'));
    const vertical = Number(heldKeys.has('KeyS') || heldKeys.has('ArrowDown')) - Number(heldKeys.has('KeyW') || heldKeys.has('ArrowUp'));
    if (horizontal || vertical) view.pan(horizontal * dt * view.span * .5, vertical * dt * view.span * .5);
  }
  if (started && !paused && world.status === 'playing') {
    accumulator = Math.min(accumulator + dt * speed, 1.2);
    while (accumulator >= DT && world.status === 'playing') {
      world.step(DT); accumulator -= DT;
      if (world.time - lastPolicyAt >= 2 - 1e-6) {
        lastPolicyAt = world.time;
        if (pilot === 'neural' || pilot === 'rule') localDecision(0, pilot);
        else if (pilot === 'manual') displayAdvice();
        localDecision(1, 'neural');
      }
    }
    if (pilot === 'laya') void layaDecision();
    if (world.events.at(-1) !== lastEvent) {
      const fresh = world.events.slice(lastEvent ? world.events.indexOf(lastEvent) + 1 : 0).filter(e => e.team === 0 || e.team === undefined); lastEvent = world.events.at(-1);
      if (fresh.length && performance.now() > toastDeadline) toast(fresh[fresh.length - 1].text, 2600);
    }
    if (world.status !== 'playing') showOutcome();
  }
  view.render(world, selected, dt); hudAge += dt;
  if (hudAge > .18) { hudAge = 0; renderHUD(); if (placement) updatePreview(); }
  labAge += dt;
  if (labAge > .5) { labAge = 0; if (weightLab?.isOpen) { workbench?.refresh(); void workbench?.checkLayaAvailability(); } }
  if (now > toastDeadline) element('toast').hidden = true;
}

try {
  view = new EmpiresView(canvas, world);
  let storage: Storage | undefined, storageError = '';
  try { storage = window.localStorage; } catch { storageError = 'This browser does not allow session storage. Export a checkpoint to keep your weights.'; }
  workbench = new WeightWorkbench({
    context: () => ({ world, team: 0, matchId, started, paused, pilot }), storage, storageError,
    update: snapshot => {
      weightLab?.update(snapshot);
      element('weight-revision').textContent = `r${snapshot.active.revision}`;
    },
    weightsChanged: snapshot => {
      cancelLaya(); nextLayaAt = world.time;
      if (pilot === 'neural') element('pilot-note').textContent = `1,293 editable weights · r${snapshot.revision}. The next strategic turn uses this revision.`;
      if (pilot === 'laya') {
        element('decision-status').textContent = `LAYA · NEW SHORTLIST r${snapshot.revision}`;
        element('decision-label').textContent = 'Weights changed. Requesting a fresh decision using the new shortlist.';
      } else displayAdvice();
    },
    coachBusyChanged: busy => { if (busy) cancelLaya(); else nextLayaAt = world.time + 2; },
  });
  weightLab = new WeightLab(element('weight-lab-host'), { ...workbench.callbacks,
    onOpenChange: open => {
      heldKeys.clear(); drag = null; element('selection-box').hidden = true; cancelPlacement();
      if (open) { workbench?.refresh(); void workbench?.checkLayaAvailability(true); }
    },
  });
  workbench.refresh();
  if (innerWidth <= 650) { element('commander-content').hidden = true; element('commander-toggle').setAttribute('aria-expanded', 'false'); element('commander-chevron').textContent = '+'; }
  renderHUD(); element('start').focus(); requestAnimationFrame(frame);
} catch (error) {
  showRenderError(`${error instanceof Error ? error.message : error} Enable WebGL and hardware acceleration, then reload.`);
}
