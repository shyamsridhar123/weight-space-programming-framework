import './play.css';
import { BUILDING_DEFS, DT, UNIT_DEFS } from './simulation';
import type { BuildingKind, Command, Entity, Point, Resource, ResourceNode, UnitKind } from './types';
import { EmpiresView } from './view';
import { createMissionWorld, getMissionProgress, missionAttackPoint, MISSION_RULES } from './mission';
import { PlayProgramming } from './play-programming';

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = el<HTMLCanvasElement>('battlefield');
const minimap = el<HTMLCanvasElement>('minimap');
const field = el('field');
const escape = (value: unknown) => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
const clock = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.z - b.z);
const names = (e: Entity) => e.category === 'unit' ? UNIT_DEFS[e.kind as UnitKind].name : BUILDING_DEFS[e.kind as BuildingKind].name;
let world = createMissionWorld();
let view: EmpiresView;
let programming: PlayProgramming;
let selected = new Set<number>();
let selectedNode: number | null = null;
let started = false, paused = false, helpWasPaused = false, ended = false, failed = false;
let labMode = false;
let speed = 1, accumulator = 0, lastFrame = performance.now(), hudAge = 0, brainAge = 0;
let toastUntil = 0;
let lastEvent = world.events.at(-1);
let placement: BuildingKind | null = null;
let placementWorkers: number[] = [];
let pointOrder: 'move' | 'attackMove' | null = null;
let cameraDragMode = false;
let pointer: Point | null = null;
let activeModal: string | null = 'welcome';
let actionsSignature = '';
let objectiveAction: () => void = () => selectVillagers();
let drag: { x: number; y: number; lastX: number; lastY: number; button: number; moved: boolean; shift: boolean; pointerId: number; pan: boolean } | null = null;
const heldKeys = new Set<string>();
const actionHandlers = new Map<string, () => void>();
const mapLabels = new Map<string, HTMLButtonElement>();
const labelActions = new Map<string, () => void>();
const labelOrders = new Map<string, () => void>();

const own = () => world.entities.filter(e => e.owner === 0 && e.hp > 0);
const home = () => own().find(e => e.kind === 'towncenter');
const rivalHome = () => world.entities.find(e => e.owner === 1 && e.kind === 'towncenter' && e.hp > 0);
const villagers = () => own().filter(e => e.kind === 'villager');
const army = () => own().filter(e => e.category === 'unit' && e.kind !== 'villager');
const selection = () => [...selected].map(id => world.entity(id)).filter((e): e is Entity => !!e);
const selectedUnits = () => selection().filter(e => e.owner === 0 && e.category === 'unit');
const selectedWorkers = () => selectedUnits().filter(e => e.kind === 'villager');
const workersOn = (resource: Resource) => villagers().filter(e => e.order.type === 'gather' && e.order.resource === resource);
const canPlay = () => started && !activeModal && !ended && !failed;

function toast(message: string, duration = 4500) {
  el('toast').textContent = message;
  el('toast').hidden = false;
  toastUntil = performance.now() + duration;
}

function focusEntities(items: Entity[]) {
  if (!items.length) return;
  view.focus(items.reduce((sum, e) => sum + e.x, 0) / items.length, items.reduce((sum, e) => sum + e.z, 0) / items.length);
}

function centerHome() {
  const town = home();
  if (town) view.focus(town.x + 1, town.z - 1);
}

function select(ids: number[], focus = false, additive = false) {
  setCameraDragMode(false);
  if (!additive) selected.clear();
  for (const id of ids) if (world.entity(id)) selected.add(id);
  selectedNode = null;
  if (focus) focusEntities(selection());
  actionsSignature = '';
  renderHUD();
}

function selectVillagers() {
  cancelCommand();
  const all = villagers(), idle = all.filter(e => e.order.type === 'idle');
  const group = idle.length ? idle.slice(0, 3) : all;
  select(group.map(e => e.id), true);
  if (!group.length) toast('No villagers remain. Train one at your Town Center.');
  else if (idle.length) toast(`${group.length} idle villager${group.length === 1 ? '' : 's'} selected. Choose Gather food or Chop wood below.`, 3500);
}

function selectArmy(focus = true) {
  cancelCommand();
  const troops = army();
  select(troops.map(e => e.id), focus);
  if (!troops.length) toast('Your army needs recruits. Select Barracks, then Train spearman.');
}

function selectBuilding(kind: BuildingKind) {
  cancelCommand();
  const building = own().find(e => e.kind === kind && e.construction >= 1) ?? own().find(e => e.kind === kind);
  if (building) select([building.id], true);
  else toast(`You have no ${BUILDING_DEFS[kind].name.toLowerCase()}. Select a villager to build one.`);
}

function resourceFor(resource: Resource, units = selectedWorkers()): ResourceNode | undefined {
  const center = units.length ? { x: units.reduce((s, e) => s + e.x, 0) / units.length, z: units.reduce((s, e) => s + e.z, 0) / units.length } : home();
  if (!center) return;
  return world.nodes.filter(n => n.kind === resource && n.amount > 0 && (n.owner === undefined || n.owner === 0) && world.isVisible(0, n.x, n.z))
    .sort((a, b) => distance(a, center) - distance(b, center))[0];
}

function issue(command: Command, message: string, point?: Point) {
  if (!canPlay()) return false;
  const result = world.command(0, command);
  if (!result.ok) { toast(result.reason ?? 'This order is unavailable.'); return false; }
  const tookControl = programming.manualOrder(message);
  el('last-order').textContent = `${clock(world.time)} · ${message}`;
  if (point) view.ping(point, command.type === 'attack' || command.type === 'move' && !!command.attackMove);
  toast(`${message}${tookControl ? ' You have taken control; blue policy is off.' : ''}${paused ? ' Resume to see it happen.' : ''}`);
  renderHUD();
  return true;
}

function gather(resource: Resource, group = selectedWorkers()) {
  const target = resourceFor(resource, group);
  if (!group.length) { toast('Select a blue villager first.'); return; }
  if (!target) { toast(resource === 'food' ? 'Nearby food is exhausted. Build a farm to grow more.' : `No visible ${resource} remains.`); return; }
  issue({ type: 'gather', ids: group.map(e => e.id), targetId: target.id }, `${group.length} villager${group.length === 1 ? '' : 's'} gathering ${resource}. Stock increases when they return home.`, target);
}

function guidedGather(resource: Resource) {
  cancelCommand();
  const all = villagers();
  const group = [...all].sort((a, b) => Number(a.order.type !== 'idle') - Number(b.order.type !== 'idle') || Number(a.order.type === 'build') - Number(b.order.type === 'build') || a.id - b.id).slice(0, 3);
  select(group.map(e => e.id), true);
  gather(resource, group);
}

function build(kind: BuildingKind) {
  cancelCommand();
  const workers = selectedWorkers();
  const group = workers.length ? workers : [...villagers()].sort((a, b) => Number(a.order.type !== 'idle') - Number(b.order.type !== 'idle')).slice(0, 1);
  if (!group.length) { toast('You need a villager to build. Train one at your Town Center.'); return; }
  select(group.map(e => e.id));
  placement = kind;
  placementWorkers = group.map(e => e.id);
  centerHome();
  canvas.style.cursor = 'crosshair';
  el('command-hint').hidden = false;
  el('command-hint-text').textContent = `Place ${BUILDING_DEFS[kind].name.toLowerCase()}: click clear ground. Green means it fits.`;
  updatePreview();
  renderActions();
  canvas.focus({ preventScroll: true });
}

function cancelCommand() {
  placement = null; pointOrder = null; placementWorkers = [];
  view?.preview(null);
  el('command-hint').hidden = true;
  canvas.style.cursor = cameraDragMode ? 'grab' : 'default';
}

function setCameraDragMode(enabled: boolean) {
  cameraDragMode = enabled;
  field.classList.toggle('camera-pan', enabled);
  el('pan-map').setAttribute('aria-pressed', `${enabled}`);
  canvas.style.cursor = placement || pointOrder ? 'crosshair' : enabled ? 'grab' : 'default';
}

function startPointOrder(kind: 'move' | 'attackMove') {
  cancelCommand();
  setCameraDragMode(false);
  pointOrder = kind;
  canvas.style.cursor = 'crosshair';
  el('command-hint').hidden = false;
  el('command-hint-text').textContent = kind === 'move' ? 'Click the ground where these units should go.' : 'Click a destination. Soldiers engage enemies along the way.';
  canvas.focus({ preventScroll: true });
}

function updatePreview() {
  if (!placement || !pointer) { view?.preview(null); return; }
  const point = { x: Math.round(pointer.x), z: Math.round(pointer.z) };
  const result = world.canCommand(0, { type: 'build', kind: placement, ...point, workerIds: placementWorkers });
  view.preview(placement, point, result.ok);
  el('command-hint-text').textContent = result.ok ? `Click to place ${BUILDING_DEFS[placement].name.toLowerCase()}. Your villager will build it.` : result.reason ?? 'Choose clear ground near your town.';
}

function placeOrder(point: Point) {
  if (placement) {
    const kind = placement, site = { x: Math.round(point.x), z: Math.round(point.z) };
    if (issue({ type: 'build', kind, ...site, workerIds: placementWorkers }, `${BUILDING_DEFS[kind].name} paid for. Your villagers are building it.`, site)) cancelCommand();
  } else if (pointOrder) {
    const attackMove = pointOrder === 'attackMove';
    if (issue({ type: 'move', ids: selectedUnits().map(e => e.id), ...point, attackMove }, attackMove ? 'Army advancing. Enemies on the route will be engaged.' : 'Selected units moving to your destination.', point)) cancelCommand();
  }
}

function attackTown(all = false) {
  const target = rivalHome();
  const troops = all ? army() : selectedUnits().filter(e => e.kind !== 'villager');
  if (!troops.length || !target) { toast('Select your soldiers first. Recruit more at the Barracks.'); return; }
  if (all) select(troops.map(e => e.id));
  // An advance fights defenders en route. A direct target order deliberately focuses only that target.
  const approach = missionAttackPoint(world);
  if (!approach) { toast('The approach is blocked. Use Advance & fight to choose another route.'); return; }
  issue({ type: 'move', ids: troops.map(e => e.id), ...approach, attackMove: true }, `${troops.length} soldiers advancing on the red town. They will fight enemies along the way.`, approach);
  focusEntities(troops);
}

function contextualOrder(clientX: number, clientY: number) {
  if (placement || pointOrder) { cancelCommand(); return; }
  const hit = view.pick(clientX, clientY); if (!hit) return;
  const target = hit.entityId === undefined ? undefined : world.entity(hit.entityId);
  const node = hit.nodeId === undefined ? undefined : world.nodes.find(n => n.id === hit.nodeId);
  const units = selectedUnits(), workers = selectedWorkers();
  if (!units.length) { toast('Select blue villagers or soldiers first.'); return; }
  if (target?.owner === 1) issue({ type: 'attack', ids: units.map(e => e.id), targetId: target.id }, `Attacking the enemy ${names(target).toLowerCase()}.`, target);
  else if (target?.owner === 0 && target.construction < 1 && workers.length) issue({ type: 'construct', ids: workers.map(e => e.id), targetId: target.id }, `Villagers resuming construction of your ${names(target).toLowerCase()}.`, target);
  else if ((node || target?.kind === 'farm') && workers.length) issue({ type: 'gather', ids: workers.map(e => e.id), targetId: node?.id ?? target!.id }, `Villagers gathering ${node?.kind ?? 'food'}. Resources arrive after a return trip.`, node ?? target);
  else issue({ type: 'move', ids: units.map(e => e.id), ...hit.point }, 'Selected units moving to your destination.', hit.point);
}

interface Action { id: string; name: string; detail: string; run: () => void; disabled?: boolean; reason?: string; primary?: boolean; minor?: boolean }
function renderActions() {
  const current = selection(), mine = current.filter(e => e.owner === 0), first = current[0];
  const workers = selectedWorkers(), units = selectedUnits();
  const troops = units.filter(e => e.kind !== 'villager');
  const actions: Action[] = [];
  const add = (action: Action) => actions.push(action);
  const train = (building: Entity, kind: UnitKind) => {
    const command: Command = { type: 'train', buildingId: building.id, kind };
    const result = world.canCommand(0, command), def = UNIT_DEFS[kind];
    add({ id: `train-${kind}`, name: `Train ${def.name.toLowerCase()}`, detail: kind === 'spearman' ? '50 food · 25 wood · 15s' : '50 food · 12s', primary: true, disabled: !result.ok, reason: result.reason,
      run: () => issue(command, `${def.name} queued. ${building.queue.length + 1} in training; each takes ${def.time} seconds.`) });
  };
  const building = (kind: BuildingKind, primary = false) => {
    const cost = BUILDING_DEFS[kind].cost;
    const affordable = Object.entries(cost).every(([resource, amount]) => world.players[0].resources[resource as Resource] >= amount);
    add({ id: `build-${kind}`, name: `Build ${BUILDING_DEFS[kind].name.toLowerCase()}`, detail: `${cost.wood ?? 0} wood · ${kind === 'house' ? '+5 population' : kind === 'farm' ? 'grow more food' : 'train soldiers'}`,
      disabled: !affordable || !villagers().length, reason: !affordable ? `Needs ${cost.wood} wood` : !villagers().length ? 'Train a villager first' : undefined, primary, run: () => build(kind) });
  };
  if (workers.length) {
    for (const resource of ['food', 'wood'] as const) {
      const target = resourceFor(resource, workers);
      add({ id: `gather-${resource}`, name: resource === 'food' ? 'Gather food' : 'Chop wood', detail: target ? `${workers.length} selected · collect & return` : resource === 'food' ? 'Build a farm for more food' : 'No wood remains', disabled: !target,
        primary: resource === 'food' ? !workersOn('food').length : workersOn('food').length > 0 && !workersOn('wood').length, run: () => gather(resource) });
    }
    building('house', workersOn('food').length > 0 && workersOn('wood').length > 0 && getMissionProgress(world).housesBuilt === 0);
    building('farm');
    if (!own().some(e => e.kind === 'barracks')) building('barracks');
    add({ id: 'move', name: 'Move', detail: 'Pick a place', minor: true, run: () => startPointOrder('move') });
    add({ id: 'stop', name: 'Stop', detail: 'Hold position', minor: true, run: () => issue({ type: 'stop', ids: units.map(e => e.id) }, 'Selected units stopped. Choose a new job when ready.') });
  } else if (troops.length) {
    add({ id: 'attack-town', name: 'Attack enemy town', detail: `${troops.length} soldiers · fight together`, primary: true, disabled: !rivalHome(), run: () => attackTown() });
    add({ id: 'attack-move', name: 'Advance & fight', detail: 'Choose a destination', run: () => startPointOrder('attackMove') });
    add({ id: 'move', name: 'Move', detail: 'Go to a position', run: () => startPointOrder('move') });
    add({ id: 'defend', name: 'Defend home', detail: 'Return to your town', run: () => { const town = home(); if (town) issue({ type: 'move', ids: troops.map(e => e.id), x: town.x - 1, z: town.z + 3, attackMove: true }, 'Army returning to defend under your Town Center.', town); } });
    add({ id: 'stop', name: 'Stop', detail: 'Hold position', minor: true, run: () => issue({ type: 'stop', ids: troops.map(e => e.id) }, 'Army holding position. Nearby enemies will be engaged.') });
  } else if (mine.length === 1 && first?.category === 'building') {
    if (first.construction < 1) {
      add({ id: 'resume-building', name: 'Resume construction', detail: 'Send a villager', primary: true, disabled: !villagers().length, run: () => {
        const worker = [...villagers()].sort((a, b) => Number(a.order.type !== 'idle') - Number(b.order.type !== 'idle') || distance(a, first) - distance(b, first))[0];
        if (worker) issue({ type: 'construct', ids: [worker.id], targetId: first.id }, `A villager is continuing your ${names(first).toLowerCase()}.`, first);
      } });
    } else {
      if (first.kind === 'towncenter') train(first, 'villager');
      if (first.kind === 'barracks') train(first, 'spearman');
      if (first.kind === 'farm') add({ id: 'work-farm', name: 'Farm food', detail: 'Assign an available villager', primary: true, disabled: !villagers().length, run: () => {
        const worker = villagers().find(e => e.order.type === 'idle') ?? villagers()[0];
        if (worker) { select([worker.id]); issue({ type: 'gather', ids: [worker.id], targetId: first.id }, 'Villager collecting food from your farm.', first); }
      } });
      building('house');
    }
    add({ id: 'villagers', name: 'Select villagers', detail: 'Gather or build', run: selectVillagers });
    if (first.kind === 'barracks') add({ id: 'army', name: 'Select army', detail: `${army().length} soldiers alive`, run: () => selectArmy() });
    else add({ id: 'barracks', name: 'Select Barracks', detail: 'Train soldiers', disabled: !own().some(e => e.kind === 'barracks'), run: () => selectBuilding('barracks') });
  } else if (first?.owner === 1) {
    add({ id: 'attack-with-army', name: 'Attack with army', detail: `${army().length} soldiers available`, primary: true, disabled: !army().length, run: () => {
      if (first.kind === 'towncenter') attackTown(true);
      else { selectArmy(false); issue({ type: 'attack', ids: army().map(e => e.id), targetId: first.id }, `Your army is attacking the enemy ${names(first).toLowerCase()}.`, first); }
    } });
    add({ id: 'home', name: 'Back to my town', detail: 'Manage your settlement', run: () => selectBuilding('towncenter') });
  } else if (selectedNode !== null) {
    const node = world.nodes.find(n => n.id === selectedNode);
    if (node) add({ id: 'assign-resource', name: node.kind === 'wood' ? 'Send woodcutters' : `Gather ${node.kind}`, detail: 'Assign up to 3 villagers', primary: true, disabled: !villagers().length || node.amount <= 0, run: () => {
      const group = [...villagers()].sort((a, b) => Number(a.order.type !== 'idle') - Number(b.order.type !== 'idle')).slice(0, 3);
      select(group.map(e => e.id));
      issue({ type: 'gather', ids: group.map(e => e.id), targetId: node.id }, `${group.length} villagers collecting ${node.kind}.`, node);
    } });
    add({ id: 'home', name: 'Select Town Center', detail: 'Manage your settlement', run: () => selectBuilding('towncenter') });
  } else {
    add({ id: 'villagers', name: 'Select villagers', detail: 'Gather food and wood', primary: true, run: selectVillagers });
    add({ id: 'barracks', name: 'Select Barracks', detail: 'Train your army', run: () => selectBuilding('barracks') });
    add({ id: 'army', name: 'Select army', detail: 'Move and fight', run: () => selectArmy() });
  }
  actionHandlers.clear();
  for (const action of actions) actionHandlers.set(action.id, action.run);
  const signature = JSON.stringify(actions.map(({ run: _run, ...rest }) => rest));
  if (signature !== actionsSignature) {
    const focused = document.activeElement instanceof HTMLElement ? document.activeElement.dataset.action : undefined;
    actionsSignature = signature;
    el('actions').innerHTML = actions.map(a => `<button data-action="${a.id}" ${a.disabled ? 'disabled' : ''} class="${a.primary ? 'primary-action' : ''} ${a.minor ? 'minor-action' : ''}" title="${escape(a.reason ?? a.detail)}"><b>${escape(a.name)}</b><small>${escape(a.disabled && a.reason ? a.reason : a.detail)}</small></button>`).join('');
    if (focused) el('actions').querySelector<HTMLButtonElement>(`button[data-action="${focused}"]:not(:disabled)`)?.focus({ preventScroll: true });
  }
}

function orderText(entity: Entity): string {
  const order = entity.order;
  if (entity.construction < 1) return `Under construction · ${Math.floor(entity.construction * 100)}%`;
  if (entity.category === 'building') return entity.queue.length ? `Training ${UNIT_DEFS[entity.queue[0].kind].name.toLowerCase()}` : entity.kind === 'barracks' ? 'Ready to train soldiers' : entity.kind === 'towncenter' ? 'Trains villagers · protects your base' : entity.kind === 'house' ? 'Provides 5 population spaces' : 'Ready';
  if (order.type === 'gather') return order.returning ? `Returning ${Math.floor(entity.carrying ?? 0)} ${entity.carryResource ?? order.resource} home` : `Gathering ${order.resource ?? 'resources'}`;
  if (order.type === 'build') { const target = world.entity(order.targetId!); return target ? `Building ${names(target).toLowerCase()} · ${Math.floor(target.construction * 100)}%` : 'Looking for construction'; }
  if (order.type === 'attack') { const target = world.entity(order.targetId!); return target ? `Attacking enemy ${names(target).toLowerCase()}` : 'Engaging the enemy'; }
  if (order.type === 'attackMove') return 'Advancing · engaging enemies on the way';
  if (order.type === 'move') return 'Moving to your destination';
  return entity.kind === 'villager' ? 'Idle · waiting for your orders' : 'Holding position · guards nearby ground';
}

function renderSelection() {
  const current = selection(), first = current[0];
  const node = selectedNode === null ? undefined : world.nodes.find(n => n.id === selectedNode);
  el('selection-owner').textContent = first?.owner === 1 ? 'RED OPPONENT · ENEMY' : node ? 'RESOURCE DEPOSIT' : 'YOUR SELECTION · BLUE';
  el('selection-health').style.background = first?.owner === 1 ? '#df9780' : '#9fc5a6';
  if (current.length) {
    const same = current.every(e => e.kind === first.kind);
    el('selection-title').textContent = current.length === 1 ? names(first) : same ? `${current.length} ${first.kind === 'spearman' ? 'spearmen' : `${names(first).toLowerCase()}s`}` : `${current.length} units`;
    const orders = current.map(orderText), counts = new Map<string, number>();
    for (const text of orders) counts.set(text, (counts.get(text) ?? 0) + 1);
    el('selection-description').textContent = counts.size === 1 ? orders[0] : [...counts].slice(0, 2).map(([text, count]) => `${count} ${text.toLowerCase()}`).join(' · ');
    const hp = current.reduce((sum, e) => sum + e.hp, 0), maxHp = current.reduce((sum, e) => sum + e.maxHp, 0);
    el('selection-health').style.width = `${Math.max(0, hp / maxHp * 100)}%`;
    const carrying = current.reduce((sum, e) => sum + (e.carrying ?? 0), 0);
    el('selection-detail').textContent = carrying > 0 ? `${Math.floor(carrying)} resources being carried · not yet in stock` : `${Math.ceil(hp)} / ${maxHp} health${first.owner === 1 ? ' · destroy to defeat' : first.kind === 'villager' ? ' · choose a job →' : ''}`;
  } else if (node) {
    el('selection-title').textContent = node.kind === 'food' ? 'Berry bushes' : node.kind === 'wood' ? 'Forest' : `${node.kind[0].toUpperCase()}${node.kind.slice(1)} deposit`;
    el('selection-description').textContent = `${Math.ceil(node.amount)} ${node.kind} remaining`;
    el('selection-health').style.width = `${node.amount / node.maxAmount * 100}%`;
    el('selection-detail').textContent = 'Villagers must collect and deliver it.';
  } else {
    el('selection-title').textContent = 'Select your people';
    el('selection-description').textContent = 'Click a blue unit or use a shortcut above.';
    el('selection-health').style.width = '0%';
    el('selection-detail').textContent = 'Your units follow your commands.';
  }
  const production = current.length === 1 && first?.owner === 0 && first.category === 'building' ? first : undefined;
  const job = production?.queue[0];
  el('production').hidden = !job && !(production && production.construction < 1);
  if (job) {
    el('production-label').textContent = `${UNIT_DEFS[job.kind].name}: ${Math.ceil(job.remaining)}s remaining · ${production!.queue.length} queued (already paid)`;
    el('production-fill').style.width = `${(1 - job.remaining / job.total) * 100}%`;
  } else if (production && production.construction < 1) {
    const builders = villagers().filter(e => e.order.type === 'build' && e.order.targetId === production.id).length;
    el('production-label').textContent = builders ? `${names(production)}: ${Math.floor(production.construction * 100)}% complete · ${builders} builder${builders === 1 ? '' : 's'}` : 'Construction stopped. Send a villager to finish it.';
    el('production-fill').style.width = `${production.construction * 100}%`;
  }
}

function renderMission() {
  const progress = getMissionProgress(world);
  const goal = progress.goals.find(g => !g.done) ?? progress.goals[3];
  const index = progress.goals.indexOf(goal);
  let title = goal.title, copy = '', status = '', actionLabel = '', action: () => void;
  let fraction = Math.min(1, goal.current / goal.target);
  const unfinishedHouse = own().find(e => e.kind === 'house' && e.construction < 1);
  if (goal.id === 'gather') {
    status = `${Math.floor(progress.gatheredFood)} / ${MISSION_RULES.foodGoal} food · ${Math.floor(progress.gatheredWood)} / ${MISSION_RULES.woodGoal} wood`;
    if (!workersOn('food').length && progress.gatheredFood < MISSION_RULES.foodGoal) {
      copy = 'Start your economy. Send three villagers to collect berries and bring food home.';
      actionLabel = 'Send 3 to food'; action = () => guidedGather('food');
    } else if (!workersOn('wood').length && progress.gatheredWood < MISSION_RULES.woodGoal) {
      copy = 'Food is on the way. Assign three more villagers to chop trees for wood.';
      actionLabel = 'Send 3 to wood'; action = () => guidedGather('wood');
    } else {
      title = 'Your villagers are working';
      copy = 'Resources arrive when they return to town. Build a house while they work.';
      actionLabel = unfinishedHouse ? 'View construction' : progress.housesBuilt ? 'Select Barracks' : 'Build a house';
      action = unfinishedHouse ? () => select([unfinishedHouse.id], true) : progress.housesBuilt ? () => selectBuilding('barracks') : () => { selected.clear(); build('house'); };
    }
  } else if (goal.id === 'house') {
    copy = unfinishedHouse ? 'A villager must finish the foundation. The five extra population spaces appear when it is complete.' : 'Make room for your army. Place a house on clear ground near your settlement.';
    status = unfinishedHouse ? `${Math.floor(unfinishedHouse.construction * 100)}% built · +5 spaces when finished` : '35 wood · +5 population spaces';
    fraction = unfinishedHouse?.construction ?? 0;
    actionLabel = unfinishedHouse ? 'View construction' : 'Place a house';
    action = unfinishedHouse ? () => select([unfinishedHouse.id], true) : () => { selected.clear(); build('house'); };
  } else if (goal.id === 'army') {
    const queued = own().reduce((sum, e) => sum + e.queue.filter(job => job.kind === 'spearman').length, 0);
    copy = `Select your Barracks and train ${MISSION_RULES.trainedGoal} spearmen. Each recruit costs 50 food and 25 wood.`;
    status = `${progress.soldiersTrained} / ${MISSION_RULES.trainedGoal} trained${queued ? ` · ${queued} in queue` : ''} · ${progress.armySize} alive`;
    actionLabel = 'Select Barracks'; action = () => selectBuilding('barracks');
  } else {
    copy = 'Select your army and advance together. Recruit replacements if you lose soldiers.';
    status = `Enemy Town Center · ${Math.ceil(progress.rivalTownCenterHp)} health · ${progress.armySize} soldiers`;
    actionLabel = 'Select army'; action = () => selectArmy();
  }
  const town = home();
  const threatened = town && world.entities.some(e => e.owner === 1 && e.category === 'unit' && e.kind !== 'villager' && e.hp > 0 && distance(e, town) < 11);
  if (threatened && world.status === 'playing') {
    title = 'Defend your Town Center';
    copy = 'Enemy soldiers are at your base. Keep your army together near your Town Center’s defensive fire.';
    status = `Your Town Center · ${Math.ceil(town.hp)} / ${town.maxHp} health`;
    fraction = town.hp / town.maxHp;
    actionLabel = 'Find my army'; action = () => { selectArmy(false); centerHome(); };
  }
  el('objective-title').textContent = title;
  el('objective-copy').textContent = copy;
  el('objective-progress').textContent = status;
  el('objective-fill').style.width = `${fraction * 100}%`;
  el('objective-fill').style.background = threatened ? '#e5a189' : '#dec781';
  el('objective-step').textContent = `${index + 1} / 4`;
  el('objective-action').textContent = `${actionLabel} →`;
  objectiveAction = action;
  for (const element of document.querySelectorAll<HTMLElement>('[data-step]')) {
    const step = progress.goals.find(g => g.id === element.dataset.step)!;
    element.classList.toggle('done', step.done); element.classList.toggle('current', step.id === goal.id && !step.done);
  }
  el('rival-status').textContent = progress.rivalGraceRemaining > 0 ? `Red prepares · ${clock(Math.ceil(progress.rivalGraceRemaining))}` : 'Red opponent · raids possible';
}

function renderLabels() {
  const items: { key: string; point: Point; height: number; label: string; detail: string; team?: string; action: () => void; order?: () => void }[] = [];
  const town = home(), enemy = rivalHome();
  for (const building of own().filter(e => e.kind === 'towncenter' || e.kind === 'barracks' || e.construction < 1)) {
    items.push({ key: `e${building.id}`, point: building, height: view.labelHeight(building) + .25, label: names(building), detail: building.construction < 1 ? `Building · ${Math.floor(building.construction * 100)}%` : building.kind === 'towncenter' ? 'YOUR BASE' : `${building.queue.length ? `${building.queue.length} training` : 'TRAIN SOLDIERS'}`, team: 'mine', action: () => { cancelCommand(); select([building.id]); }, order: () => {
      const workers = selectedWorkers();
      if (building.construction < 1 && workers.length) issue({ type: 'construct', ids: workers.map(e => e.id), targetId: building.id }, `Villagers resuming construction of your ${names(building).toLowerCase()}.`, building);
      else select([building.id]);
    } });
  }
  if (enemy) items.push({ key: 'enemy-town', point: enemy, height: view.labelHeight(enemy) + .25, label: 'Enemy Town Center', detail: `${Math.ceil(enemy.hp)} / ${enemy.maxHp} HP · WIN TARGET`, team: 'enemy', action: () => { cancelCommand(); select([enemy.id]); }, order: () => {
    const units = selectedUnits();
    if (units.length) issue({ type: 'attack', ids: units.map(e => e.id), targetId: enemy.id }, 'Attacking the enemy Town Center.', enemy);
    else toast('Select blue soldiers first, then right-click the enemy target.');
  } });
  if (town) for (const resource of ['food', 'wood'] as const) {
    const node = resourceFor(resource, [town]);
    if (node) items.push({ key: `resource-${resource}`, point: node, height: resource === 'wood' ? 3.2 : 1, label: resource === 'food' ? 'Berries · food' : 'Trees · wood', detail: `${Math.ceil(node.amount)} remaining${selectedWorkers().length ? ' · CLICK TO GATHER' : ''}`, action: () => {
      if (placement || pointOrder) { placeOrder(node); return; }
      if (selectedWorkers().length) issue({ type: 'gather', ids: selectedWorkers().map(e => e.id), targetId: node.id }, `Selected villagers gathering ${resource}. Watch their return trips.`, node);
      else { selected.clear(); selectedNode = node.id; renderHUD(); }
    } });
  }
  const rect = field.getBoundingClientRect();
  const seen = new Set<string>(); labelActions.clear(); labelOrders.clear();
  const occupied = ['.mission-card', '#wsp-field-guide', '.enemy-status', '.map-tools', '#toast', '#command-hint', '#paused-banner']
    .map(selector => field.querySelector<HTMLElement>(selector))
    .filter((element): element is HTMLElement => element !== null && !element.hidden)
    .map(element => element.getBoundingClientRect());
  for (const item of items) {
    seen.add(item.key);
    let label = mapLabels.get(item.key);
    if (!label) { label = document.createElement('button'); label.dataset.label = item.key; el('map-labels').append(label); mapLabels.set(item.key, label); }
    const position = view.screen(item.point.x, item.point.z, item.height);
    label.style.left = `${position.x - rect.left}px`; label.style.top = `${position.y - rect.top}px`;
    label.className = `map-label ${item.team ?? ''}`;
    const html = `${escape(item.label)}<span class="label-detail">${escape(item.detail)}</span>`;
    if (label.innerHTML !== html) label.innerHTML = html;
    label.hidden = false;
    const bounds = label.getBoundingClientRect();
    const clipped = !Number.isFinite(position.x) || !Number.isFinite(position.y) || bounds.left < rect.left + 8 || bounds.right > rect.right - 8 || bounds.top < rect.top + 8 || bounds.bottom > rect.bottom - 8;
    // Keep higher-priority labels readable as the camera zooms out; hidden labels still have selectable map objects.
    const overlaps = occupied.some(other => bounds.left < other.right + 6 && bounds.right > other.left - 6 && bounds.top < other.bottom + 6 && bounds.bottom > other.top - 6);
    label.hidden = clipped || overlaps;
    if (!label.hidden) occupied.push(bounds);
    labelActions.set(item.key, item.action);
    labelOrders.set(item.key, item.order ?? item.action);
  }
  for (const [key, label] of mapLabels) if (!seen.has(key)) { label.remove(); mapLabels.delete(key); }
}

function renderHUD() {
  for (const id of selected) if (!world.entity(id)) selected.delete(id);
  const stock = world.players[0].resources;
  el('food-count').textContent = `${Math.floor(stock.food)}`;
  el('wood-count').textContent = `${Math.floor(stock.wood)}`;
  el('food-workers').textContent = `${workersOn('food').length} working`;
  el('wood-workers').textContent = `${workersOn('wood').length} working`;
  el('population-count').textContent = `${world.population(0)} / ${world.capacity(0)}`;
  el('population-count').style.color = world.population(0) >= world.capacity(0) ? '#f0b395' : '';
  el('match-clock').textContent = clock(world.time);
  const idle = villagers().filter(e => e.order.type === 'idle').length;
  el('villager-shortcut').textContent = idle ? `Idle villagers · ${idle}` : `Villagers · ${villagers().length}`;
  const queued = own().reduce((sum, e) => sum + e.queue.filter(j => j.kind !== 'villager').length, 0);
  el('army-shortcut').textContent = `Army · ${army().length}${queued ? ` +${queued}` : ''}`;
  el('select-villagers').classList.toggle('active', selectedWorkers().length > 0);
  el('select-army').classList.toggle('active', selectedUnits().some(e => e.kind !== 'villager'));
  el('select-town').classList.toggle('active', selection().some(e => e.owner === 0 && e.kind === 'towncenter'));
  el('select-barracks').classList.toggle('active', selection().some(e => e.owner === 0 && e.kind === 'barracks'));
  el('pause').textContent = paused ? 'Resume' : 'Pause';
  el('pause').setAttribute('aria-label', paused ? 'Resume game' : 'Pause game');
  el('paused-banner').hidden = !paused || !!activeModal || ended;
  el('wsp-sim-status').textContent = paused ? `PAUSED · ${clock(world.time)}` : `LIVE · ${speed}× · ${clock(world.time)}`;
  el('wsp-sim-status').dataset.state = paused ? 'paused' : 'running';
  renderMission(); renderSelection(); renderActions();
  if (view) { view.minimap(minimap, selected); renderLabels(); }
}

function renderBrain() {
  programming?.render();
}

function setBrainOpen(open: boolean) {
  cancelCommand(); heldKeys.clear();
  el('brain-panel').hidden = !open;
  el('workspace').classList.toggle('brain-open', open);
  el('brain-toggle').setAttribute('aria-expanded', `${open}`);
  renderBrain();
  if (open) el('brain-close').focus(); else el('brain-toggle').focus();
}

function showModal(id: string | null) {
  activeModal = id;
  for (const name of ['welcome', 'help-modal', 'outcome', 'render-error']) el(name).hidden = name !== id;
  el('game-shell').inert = !!id;
  heldKeys.clear(); drag = null; el('selection-box').hidden = true;
  if (id === 'help-modal') el('close-help').focus();
  else if (id) el(id).querySelector<HTMLElement>('button:not(:disabled)')?.focus();
}

function setLabMode(enabled: boolean) {
  labMode = enabled;
  document.body.classList.toggle('wsp-mode', enabled);
  el('wsp-field-guide').hidden = !enabled;
}

function resumeGame() {
  if (!canPlay()) return;
  paused = false; accumulator = 0; lastFrame = performance.now(); heldKeys.clear();
  renderHUD();
}

function startMatch(mode: 'lab' | 'battle' = 'lab') {
  world = createMissionWorld(42); view.setWorld(world);
  started = true; paused = mode === 'lab'; ended = false; speed = 1; accumulator = 0; lastEvent = world.events.at(-1);
  setLabMode(mode === 'lab');
  el('speed').textContent = '1×'; el('speed').setAttribute('aria-label', 'Game speed, 1 times');
  el('last-order').textContent = 'Your units are waiting for your orders.';
  cancelCommand(); showModal(null); programming.startMatch(mode === 'lab'); setBrainOpen(mode === 'lab');
  select(villagers().slice(0, 3).map(e => e.id)); centerHome();
  if (mode === 'lab') {
    el('toast').hidden = true; toastUntil = 0;
  } else toast('You command blue. Start with “Send 3 to food”. Open Weight-space lab to edit the network or let it play.', 6500);
  canvas.focus({ preventScroll: true }); lastFrame = performance.now();
  renderHUD(); renderBrain();
}

function togglePause() {
  if (!canPlay()) return;
  paused = !paused; accumulator = 0; heldKeys.clear();
  renderHUD(); renderBrain();
}

function openHelp() {
  if (!canPlay()) return;
  helpWasPaused = paused; paused = true; accumulator = 0; cancelCommand();
  showModal('help-modal'); renderHUD();
}

function closeHelp() {
  paused = helpWasPaused; accumulator = 0; showModal(null); renderHUD(); canvas.focus();
}

function finishMatch() {
  if (ended || world.status === 'playing') return;
  ended = true; cancelCommand(); heldKeys.clear(); accumulator = 0;
  el('outcome-title').textContent = world.status === 'won' ? 'Victory. Your town stands.' : world.status === 'lost' ? 'Your Town Center fell.' : 'The battle is over.';
  el('outcome-copy').textContent = world.status === 'won' ? 'Your army destroyed the red Town Center. Your gathering, construction, and recruitment made this victory possible.' : world.status === 'lost' ? 'The red army destroyed your Town Center. Try again: keep villagers gathering, recruit a larger force, and defend together under your town’s protection.' : 'Both sides lost their Town Centers. Build another army and try again.';
  const progress = getMissionProgress(world);
  el('outcome-stats').innerHTML = `<div><b>${clock(world.time)}</b><span>MATCH TIME</span></div><div><b>${progress.soldiersTrained}</b><span>SOLDIERS TRAINED</span></div><div><b>${world.players[0].kills}</b><span>ENEMIES DEFEATED</span></div>`;
  showModal('outcome'); renderHUD(); renderBrain();
}

function showRenderError(message: string) {
  failed = true; paused = true; accumulator = 0; cancelCommand(); heldKeys.clear();
  el('render-error-detail').textContent = `${message} This game needs WebGL and hardware acceleration.`;
  showModal('render-error');
}

canvas.addEventListener('contextmenu', event => event.preventDefault());
canvas.addEventListener('pointerdown', event => {
  if (!canPlay() || event.button > 2) return;
  canvas.focus({ preventScroll: true }); canvas.setPointerCapture(event.pointerId);
  drag = { x: event.clientX, y: event.clientY, lastX: event.clientX, lastY: event.clientY, button: event.button, moved: false, shift: event.shiftKey, pointerId: event.pointerId, pan: event.button === 0 && cameraDragMode && !placement && !pointOrder };
});
canvas.addEventListener('pointermove', event => {
  if (!canPlay()) return;
  pointer = view.ground(event.clientX, event.clientY); updatePreview();
  if (!drag) return;
  if (Math.hypot(event.clientX - drag.x, event.clientY - drag.y) > 6) drag.moved = true;
  if (drag.moved && (drag.pan || drag.button === 1 || drag.button === 2)) view.pan(-(event.clientX - drag.lastX) * view.span / canvas.clientHeight, -(event.clientY - drag.lastY) * view.span / canvas.clientHeight * 1.4);
  else if (drag.moved && drag.button === 0 && !placement && !pointOrder) {
    const rect = field.getBoundingClientRect();
    const box = el('selection-box'); box.hidden = false;
    Object.assign(box.style, { left: `${Math.min(drag.x, event.clientX) - rect.left}px`, top: `${Math.min(drag.y, event.clientY) - rect.top}px`, width: `${Math.abs(event.clientX - drag.x)}px`, height: `${Math.abs(event.clientY - drag.y)}px` });
  }
  drag.lastX = event.clientX; drag.lastY = event.clientY;
});
canvas.addEventListener('pointerup', event => {
  if (!drag) return;
  const state = drag; drag = null; el('selection-box').hidden = true;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  if (!canPlay()) return;
  if (state.button === 2 && !state.moved) { contextualOrder(event.clientX, event.clientY); return; }
  if (state.button !== 0 || state.pan) return;
  const hit = view.pick(event.clientX, event.clientY);
  if (placement || pointOrder) { if (hit) placeOrder(hit.point); return; }
  if (state.moved) {
    const ids = own().filter(e => {
      if (e.category !== 'unit') return false;
      const p = view.screen(e.x, e.z, .8);
      return p.x >= Math.min(state.x, event.clientX) && p.x <= Math.max(state.x, event.clientX) && p.y >= Math.min(state.y, event.clientY) && p.y <= Math.max(state.y, event.clientY);
    }).map(e => e.id);
    select(ids, false, state.shift);
  } else if (event.pointerType === 'touch' && selectedUnits().length && hit && (hit.nodeId !== undefined || hit.entityId === undefined || world.entity(hit.entityId)?.owner === 1)) contextualOrder(event.clientX, event.clientY);
  else if (hit?.entityId !== undefined) select([hit.entityId], false, state.shift);
  else if (hit?.nodeId !== undefined) {
    if (selectedWorkers().length) contextualOrder(event.clientX, event.clientY);
    else { selected.clear(); selectedNode = hit.nodeId; renderHUD(); }
  } else if (!state.shift) select([]);
});
function cancelDrag() { drag = null; el('selection-box').hidden = true; }
canvas.addEventListener('pointercancel', cancelDrag);
canvas.addEventListener('lostpointercapture', cancelDrag);
canvas.addEventListener('wheel', event => { if (!activeModal) { event.preventDefault(); view.zoom(event.deltaY); updatePreview(); renderLabels(); } }, { passive: false });
minimap.addEventListener('pointerdown', event => {
  if (!canPlay()) return;
  const rect = minimap.getBoundingClientRect(); view.focus((event.clientX - rect.left) / rect.width * world.size, (event.clientY - rect.top) / rect.height * world.size); renderLabels();
});
el('actions').addEventListener('click', event => { const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-action]'); if (button && !button.disabled && canPlay()) actionHandlers.get(button.dataset.action!)?.(); });
el('map-labels').addEventListener('click', event => { const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-label]'); if (button && canPlay()) labelActions.get(button.dataset.label!)?.(); });
el('map-labels').addEventListener('contextmenu', event => {
  event.preventDefault();
  if (!canPlay()) return;
  if (placement || pointOrder) { cancelCommand(); return; }
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-label]');
  if (button) labelOrders.get(button.dataset.label!)?.();
});
el('objective-action').addEventListener('click', () => { if (canPlay()) objectiveAction(); });
el('start').addEventListener('click', () => startMatch('lab'));
el('start-battle').addEventListener('click', () => startMatch('battle'));
el('astra-start').addEventListener('click', () => {
  if (!canPlay() || world.status !== 'playing') return;
  // The panel starts the policy; restore the surrounding AI interface in place.
  setLabMode(true); setBrainOpen(true);
  el('toast').hidden = true; toastUntil = 0;
});
el('play-again').addEventListener('click', () => startMatch(labMode ? 'lab' : 'battle'));
el('restart').addEventListener('click', () => startMatch(labMode ? 'lab' : 'battle'));
el('wsp-restart').addEventListener('click', () => startMatch('lab'));
el('wsp-battle').addEventListener('click', () => {
  programming.takeCommand();
  setLabMode(false); setBrainOpen(false); resumeGame();
  toast('You command blue. Astra stopped; your units finish their existing orders. Select units to give them a new job.');
  canvas.focus({ preventScroll: true });
});
el('pause').addEventListener('click', togglePause);
el('resume').addEventListener('click', togglePause);
el('speed').addEventListener('click', () => { speed = speed === 1 ? 2 : 1; el('speed').textContent = `${speed}×`; el('speed').setAttribute('aria-label', `Game speed, ${speed} times`); });
el('help').addEventListener('click', openHelp);
el('close-help').addEventListener('click', closeHelp);
el('reload').addEventListener('click', () => location.reload());
el('select-villagers').addEventListener('click', selectVillagers);
el('select-army').addEventListener('click', () => selectArmy());
el('select-town').addEventListener('click', () => selectBuilding('towncenter'));
el('select-barracks').addEventListener('click', () => selectBuilding('barracks'));
el('home').addEventListener('click', centerHome);
el('pan-map').addEventListener('click', () => {
  if (!canPlay()) return;
  cancelCommand(); setCameraDragMode(!cameraDragMode);
  if (cameraDragMode) toast('Drag the battlefield to move the camera. Turn Pan map off, or select a group below, to give orders.');
  canvas.focus({ preventScroll: true });
});
el('find-enemy').addEventListener('click', () => { const target = rivalHome(); if (target) { view.focus(target.x, target.z); view.ping(target, true); } });
el('zoom-in').addEventListener('click', () => view.zoom(-170));
el('zoom-out').addEventListener('click', () => view.zoom(170));
el('cancel-command').addEventListener('click', cancelCommand);
el('brain-toggle').addEventListener('click', () => setBrainOpen(el('brain-panel').hidden));
el('brain-close').addEventListener('click', () => setBrainOpen(false));

window.addEventListener('keydown', event => {
  if (activeModal) {
    if (event.code === 'Escape' && activeModal === 'help-modal') { event.preventDefault(); closeHelp(); return; }
    if (event.code === 'Tab') {
      const controls = [...el(activeModal).querySelectorAll<HTMLElement>('button:not(:disabled),a[href],input:not(:disabled),[tabindex="0"]')];
      const first = controls[0], last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first || !event.shiftKey && document.activeElement === last || !el(activeModal).contains(document.activeElement)) { event.preventDefault(); (event.shiftKey ? last : first)?.focus(); }
    }
    return;
  }
  if (!started || ended || failed) return;
  if (event.code === 'Escape') {
    event.preventDefault();
    if (placement || pointOrder) cancelCommand();
    else if (!el('brain-panel').hidden) setBrainOpen(false);
    else togglePause();
    return;
  }
  const target = event.target as HTMLElement;
  if (target.closest('input,select,textarea,[contenteditable=true],#brain-panel')) return;
  if (target.closest('button,summary') && ['Space', 'Enter'].includes(event.code)) return;
  if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.code)) event.preventDefault();
  heldKeys.add(event.code);
  if (event.repeat) return;
  if (event.code === 'Space') togglePause();
  if (event.code === 'Digit1') selectVillagers();
  if (event.code === 'Digit2') selectArmy();
  if (event.code === 'KeyH') selectBuilding('towncenter');
  if (event.code === 'KeyB') selectBuilding('barracks');
});
window.addEventListener('keyup', event => heldKeys.delete(event.code));
window.addEventListener('blur', () => { heldKeys.clear(); cancelDrag(); });
document.addEventListener('visibilitychange', () => {
  lastFrame = performance.now(); accumulator = 0; heldKeys.clear();
  if (document.hidden && started && !ended && !paused) { paused = true; renderHUD(); }
});
window.addEventListener('empires-render-error', event => showRenderError((event as CustomEvent<string>).detail));
window.addEventListener('pagehide', event => {
  if (!event.persisted) { programming?.dispose(); view?.dispose(); }
});
window.addEventListener('pageshow', event => {
  if (!event.persisted) return;
  lastFrame = performance.now(); accumulator = 0; heldKeys.clear(); cancelDrag();
  if (started && !ended) paused = true;
  renderHUD(); renderBrain();
});

function frame(now: number) {
  if (failed) return;
  requestAnimationFrame(frame);
  const dt = Math.min(.12, Math.max(0, (now - lastFrame) / 1000)); lastFrame = now;
  if (canPlay()) {
    const horizontal = Number(heldKeys.has('KeyD') || heldKeys.has('ArrowRight')) - Number(heldKeys.has('KeyA') || heldKeys.has('ArrowLeft'));
    const vertical = Number(heldKeys.has('KeyS') || heldKeys.has('ArrowDown')) - Number(heldKeys.has('KeyW') || heldKeys.has('ArrowUp'));
    if (horizontal || vertical) view.pan(horizontal * dt * view.span * .5, vertical * dt * view.span * .5);
  }
  if (started && !paused && !activeModal && world.status === 'playing' && !document.hidden) {
    accumulator = Math.min(accumulator + dt * speed, 1);
    while (accumulator >= DT && world.status === 'playing') {
      world.step(DT); accumulator -= DT;
      programming.tick();
    }
    if (world.events.at(-1) !== lastEvent) {
      const previousIndex = lastEvent ? world.events.indexOf(lastEvent) : -1;
      const fresh = world.events.slice(previousIndex + 1).filter(event => event.team === 0 || event.team === undefined);
      lastEvent = world.events.at(-1);
      if (fresh.length && now > toastUntil && !placement && !pointOrder) toast(fresh.at(-1)!.text, 3000);
    }
    if (world.status !== 'playing') finishMatch();
  }
  view.render(world, selected, dt);
  // Labels follow the camera each frame; counters are throttled to keep the controls responsive.
  renderLabels();
  hudAge += dt; brainAge += dt;
  if (hudAge >= .2) { hudAge = 0; renderHUD(); if (placement) updatePreview(); }
  if (brainAge >= .45) { brainAge = 0; if (!el('brain-panel').hidden) renderBrain(); }
  if (now > toastUntil) el('toast').hidden = true;
}

try {
  el('welcome-grace').textContent = clock(MISSION_RULES.rivalGraceSeconds);
  el('help-recruits').textContent = `${MISSION_RULES.trainedGoal}`;
  view = new EmpiresView(canvas, world);
  programming = new PlayProgramming({
    world: () => world,
    available: canPlay,
    paused: () => paused,
    pause: () => { paused = true; accumulator = 0; heldKeys.clear(); cancelCommand(); renderHUD(); },
    resume: resumeGame,
    watchOrder: (_team, command) => {
      if ('buildingId' in command) select([command.buildingId], true);
      else if ('ids' in command) select(command.ids, true);
      else if (command.type === 'build') {
        const building = world.entities.find(entity => entity.owner === _team && entity.kind === command.kind && Math.hypot(entity.x - command.x, entity.z - command.z) < .1);
        if (building) select([building.id], true);
        else view.focus(command.x, command.z);
      }
    },
    onOrder: (team, command, label, revision) => {
      if (team !== 0) return;
      el('last-order').textContent = `${clock(world.time)} · Policy r${revision} · ${label}`;
      const point = 'x' in command ? command : 'targetId' in command
        ? world.entity(command.targetId) ?? world.nodes.find(node => node.id === command.targetId)
        : 'buildingId' in command ? world.entity(command.buildingId) : undefined;
      if (point) view.ping(point, command.type === 'attack' || command.type === 'move' && !!command.attackMove);
      actionsSignature = ''; renderHUD();
    },
    notify: toast,
  });
  select(villagers().slice(0, 3).map(e => e.id));
  centerHome(); renderHUD(); renderBrain();
  if (new URLSearchParams(location.search).get('wsp') === '1') startMatch('lab');
  else showModal('welcome');
  requestAnimationFrame(frame);
} catch (error) {
  showRenderError(error instanceof Error ? error.message : String(error));
}
