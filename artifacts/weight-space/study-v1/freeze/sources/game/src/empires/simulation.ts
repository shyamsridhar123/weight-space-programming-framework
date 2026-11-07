import { AGE_COSTS, AGE_NAMES, AGE_TIMES, BUILDING_DEFS, DT, RESOURCES, UNIT_DEFS, UPGRADE_DEFS } from './types';
import type { BuildingKind, Command, CommandResult, Entity, EntityKind, Player, Point, ProductionStats, Resource, ResourceNode, Resources, Team, Terrain, UnitKind, WorldOptions } from './types';
export * from './types';

const emptyResources = (): Resources => ({ food: 0, wood: 0, gold: 0, stone: 0 });
const makePlayer = (): Player => ({ age: 1, resources: { food: 340, wood: 420, gold: 160, stone: 120 }, upgrades: [], advancing: 0, gathered: emptyResources(), kills: 0 });
const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.z - b.z);
const good: CommandResult = { ok: true };
const fail = (reason: string): CommandResult => ({ ok: false, reason });
const radius = (e: Entity) => e.category === 'building' ? BUILDING_DEFS[e.kind as BuildingKind].size / 2 : 0.35;

/** Complete continuation state. Unlike snapshot(), this also retains navigation and identity state. */
export interface WorldCheckpoint {
  version: 1;
  size: number;
  seed: number;
  options: WorldOptions;
  time: number;
  tick: number;
  players: [Player, Player];
  production: [ProductionStats, ProductionStats];
  entities: Entity[];
  nodes: ResourceNode[];
  terrain: Terrain[];
  explored: [number[], number[]];
  visibility: [number[], number[]];
  status: 'playing' | 'won' | 'lost' | 'draw';
  events: { time: number; text: string; team?: Team }[];
  nextId: number;
  randomState: number;
  blocked: number[];
  navigationVersion: number;
  routes: [number, { key: string; retry: number; arrival?: Point }][];
}

/** Renderer-independent fixed-step RTS. Commands are the only player/AI input. */
export class World {
  readonly size = 64;
  time = 0;
  tick = 0;
  players: [Player, Player] = [makePlayer(), makePlayer()];
  readonly production: [ProductionStats, ProductionStats] = [
    { trained: {}, constructed: {} }, { trained: {}, constructed: {} },
  ];
  entities: Entity[] = [];
  nodes: ResourceNode[] = [];
  terrain: Terrain[] = [];
  explored = [new Uint8Array(4096), new Uint8Array(4096)];
  visibility = [new Uint8Array(4096), new Uint8Array(4096)];
  status: 'playing' | 'won' | 'lost' | 'draw' = 'playing';
  events: { time: number; text: string; team?: Team }[] = [];
  private nextId = 1;
  private randomState: number;
  private blocked = new Uint8Array(4096);
  private navigationVersion = 0;
  private routes = new Map<number, { key: string; retry: number; arrival?: Point }>();

  constructor(public seed = 42, readonly options: WorldOptions = {}) {
    this.randomState = seed >>> 0 || 1;
    if (options.scenario === 'first-victory') {
      this.createFirstVictory();
      return;
    }
    for (let z = 0; z < this.size; z++) for (let x = 0; x < this.size; x++) {
      this.terrain.push(x >= 30 && x <= 33 ? ((z >= 18 && z <= 21) || (z >= 40 && z <= 43) ? 'bridge' : 'water') : this.random() < 0.055 ? 'hill' : 'grass');
    }
    for (const team of [0, 1] as const) {
      const x = team === 0 ? 16 : 47, z = team === 0 ? 43 : 18, s = team === 0 ? 1 : -1;
      this.spawn(team, 'towncenter', x, z);
      this.spawn(team, 'house', x - 4 * s, z + 4 * s);
      this.spawn(team, 'barracks', x + 5 * s, z + 4 * s);
      for (let i = 0; i < 6; i++) this.spawn(team, 'villager', x + (i - 2.5) * s, z - 4 * s);
      this.spawn(team, 'spearman', x + 6 * s, z - 2 * s);
      this.spawn(team, 'spearman', x + 7 * s, z);
      this.addNode('food', x - 6 * s, z - 6 * s, 800);
      this.addNode('food', x - 7 * s, z - 4 * s, 800);
      this.addNode('gold', x + 7 * s, z - 6 * s, 2400);
      this.addNode('stone', x - 8 * s, z + s, 1800);
      for (let i = 0; i < 24; i++) this.addNode('wood', x + (i % 6 - 2) * 1.1 * s, z + (7 + Math.floor(i / 6) * 1.1) * s, 150);
    }
    for (let i = 0; i < 170; i++) {
      const p = { x: 3 + Math.floor(this.random() * 58) + 0.5, z: 3 + Math.floor(this.random() * 58) + 0.5 };
      if (this.terrainAt(p.x, p.z) === 'water' || this.terrainAt(p.x, p.z) === 'bridge' || this.entities.some(e => distance(e, p) < radius(e) + 4)) continue;
      const kind: Resource = i % 23 === 0 ? 'gold' : i % 29 === 0 ? 'stone' : 'wood';
      this.addNode(kind, p.x, p.z, kind === 'wood' ? 120 : 1600);
    }
    this.rebuildObstacles();
    this.updateVisibility();
    for (const team of [0, 1] as const) {
      const choices: Resource[] = ['food', 'food', 'wood', 'wood', 'gold', 'stone'];
      this.entities.filter(e => e.owner === team && e.kind === 'villager').forEach((e, i) => {
        const node = this.nearestSource(e, choices[i]);
        if (node) e.order = { type: 'gather', targetId: node.id, resource: node.kind };
      });
    }
    this.log('Your charter begins. Grow your settlement and destroy the rival Town Center.');
  }

  private createFirstVictory() {
    // A broad crossing and uncrowded settlements keep first-time orders legible.
    // Both sides use the same units, building health, costs, and gather rates.
    for (let z = 0; z < this.size; z++) for (let x = 0; x < this.size; x++) {
      this.terrain.push(x >= 30 && x <= 33 ? (z >= 27 && z <= 36 ? 'bridge' : 'water')
        : this.random() < 0.035 ? 'hill' : 'grass');
    }
    for (const team of [0, 1] as const) {
      const x = team === 0 ? 18 : 46, z = 32, s = team === 0 ? 1 : -1;
      this.players[team].resources = { food: 500, wood: 300, gold: 0, stone: 0 };
      this.spawn(team, 'towncenter', x, z);
      this.spawn(team, 'barracks', x + 2 * s, z + 6 * s);
      this.spawn(team, 'house', x - 2 * s, z + 7 * s);
      for (let i = 0; i < 6; i++) this.spawn(team, 'villager', x + (i % 3 - 1) * 1.15, z - (4 + Math.floor(i / 3)) * s);
      this.spawn(team, 'spearman', x + 6 * s, z - 1);
      this.spawn(team, 'spearman', x + 6 * s, z + 2);
      this.addNode('food', x - 7 * s, z - 4 * s, 1800);
      this.addNode('food', x - 7 * s, z - 2 * s, 1800);
      for (let i = 0; i < 12; i++) this.addNode('wood', x - (6 + i % 3 * 1.2) * s, z + (3 + Math.floor(i / 3) * 1.2) * s, 200);
    }
    this.rebuildObstacles();
    this.updateVisibility();
    this.log('First victory: gather food and wood, build housing, train troops, and destroy the red Town Center.');
  }

  entity(id: number) { return this.entities.find(e => e.id === id && e.hp > 0); }
  population(team: Team) { return this.entities.filter(e => e.owner === team && e.hp > 0).reduce((n, e) => n + (e.category === 'unit' ? 1 : e.queue.length), 0); }
  capacity(team: Team) { return Math.min(100, this.entities.filter(e => e.owner === team && e.hp > 0 && e.category === 'building' && e.construction >= 1).reduce((n, e) => n + BUILDING_DEFS[e.kind as BuildingKind].population, 0)); }
  isVisible(team: Team, x: number, z: number) { return this.inBounds(x, z) && this.visibility[team][Math.floor(z) * this.size + Math.floor(x)] === 1; }
  isExplored(team: Team, x: number, z: number) { return this.inBounds(x, z) && this.explored[team][Math.floor(z) * this.size + Math.floor(x)] === 1; }

  canCommand(team: Team, cmd: Command): CommandResult {
    if (team !== 0 && team !== 1) return fail('Unknown player.');
    if (this.status !== 'playing') return fail('The match has ended.');
    const player = this.players[team];
    if ('ids' in cmd) {
      if (!cmd.ids.length) return fail('Select units first.');
      if (cmd.ids.some(id => { const e = this.entity(id); return !e || e.owner !== team || e.category !== 'unit'; })) return fail('You can command only your own units.');
    }
    if (cmd.type === 'move') {
      return this.walkable(cmd.x, cmd.z) ? good : fail('Choose open ground or a bridge.');
    }
    if (cmd.type === 'stop') return good;
    if (cmd.type === 'gather') {
      if (cmd.ids.some(id => this.entity(id)?.kind !== 'villager')) return fail('Only villagers gather resources.');
      const source = this.source(cmd.targetId);
      return source && source.amount > 0 && (source.owner === undefined || source.owner === team) && this.isVisible(team, source.x, source.z) ? good : fail('Choose a visible resource or your completed farm.');
    }
    if (cmd.type === 'construct') {
      if (cmd.ids.some(id => this.entity(id)?.kind !== 'villager')) return fail('Only villagers construct buildings.');
      const target = this.entity(cmd.targetId);
      return target && target.owner === team && target.category === 'building' && target.construction < 1
        ? good : fail('Choose one of your unfinished buildings.');
    }
    if (cmd.type === 'attack') {
      const target = this.entity(cmd.targetId);
      return target && target.owner !== team && this.isVisible(team, target.x, target.z) ? good : fail('Choose a visible enemy.');
    }
    if (cmd.type === 'build') {
      const def = BUILDING_DEFS[cmd.kind];
      if (!def) return fail('Unknown building.');
      if (player.age < def.requiredAge) return fail(`Requires ${AGE_NAMES[def.requiredAge]}.`);
      if (!this.affords(team, def.cost)) return fail('Not enough resources.');
      const workers = cmd.workerIds?.length ? cmd.workerIds.map(id => this.entity(id)) : this.entities.filter(e => e.owner === team && e.kind === 'villager' && e.hp > 0);
      if (!workers.length || workers.some(e => !e || e.owner !== team || e.kind !== 'villager')) return fail('A villager must build this.');
      return this.legalSite(team, cmd.kind, cmd.x, cmd.z);
    }
    if (cmd.type === 'train') {
      const building = this.entity(cmd.buildingId), def = UNIT_DEFS[cmd.kind];
      if (!def || !building || building.owner !== team || building.category !== 'building' || building.construction < 1) return fail('Choose a completed production building.');
      if (!BUILDING_DEFS[building.kind as BuildingKind].trainable.includes(cmd.kind)) return fail('This building cannot train that unit.');
      if (player.age < def.requiredAge) return fail(`Requires ${AGE_NAMES[def.requiredAge]}.`);
      if (building.queue.length >= 5) return fail('The training queue is full.');
      if (this.population(team) >= this.capacity(team)) return fail('Population limit. Build a house.');
      return this.affords(team, def.cost) ? good : fail('Not enough resources.');
    }
    if (cmd.type === 'advance') {
      if (player.age >= 4) return fail('Already in the Imperial Age.');
      if (player.advancing > 0) return fail('Age advancement is already underway.');
      if (!this.entities.some(e => e.owner === team && e.kind === 'towncenter' && e.construction === 1 && e.hp > 0)) return fail('Requires a completed Town Center.');
      return this.affords(team, AGE_COSTS[player.age + 1]) ? good : fail('Not enough food and gold to advance.');
    }
    if (cmd.type === 'research') {
      const def = UPGRADE_DEFS[cmd.upgrade];
      if (!def) return fail('Unknown upgrade.');
      if (player.upgrades.includes(cmd.upgrade)) return fail('Already researched.');
      if (player.age < def.requiredAge) return fail(`Requires ${AGE_NAMES[def.requiredAge]}.`);
      if (cmd.upgrade !== 'economy' && !this.entities.some(e => e.owner === team && e.kind === 'forge' && e.construction === 1 && e.hp > 0)) return fail('Build a Forge first.');
      return this.affords(team, def.cost) ? good : fail('Not enough resources.');
    }
    return fail('Unknown command.');
  }

  command(team: Team, cmd: Command): CommandResult {
    const result = this.canCommand(team, cmd);
    if (!result.ok) return result;
    const player = this.players[team];
    if ('ids' in cmd) {
      const units = [...new Set(cmd.ids)].sort((a, b) => a - b).map(id => this.entity(id)!);
      units.forEach((e, i) => {
        this.routes.delete(e.id);
        if (cmd.type === 'stop') e.order = { type: 'idle' };
        if (cmd.type === 'move') {
          const width = Math.ceil(Math.sqrt(units.length));
          const p = { x: cmd.x + (i % width - (width - 1) / 2) * 0.85, z: cmd.z + (Math.floor(i / width) - (Math.ceil(units.length / width) - 1) / 2) * 0.85 };
          e.order = { type: cmd.attackMove ? 'attackMove' : 'move', ...(this.walkable(p.x, p.z) ? p : { x: cmd.x, z: cmd.z }) };
        }
        if (cmd.type === 'attack') { const t = this.entity(cmd.targetId)!; e.order = { type: 'attack', targetId: t.id, x: t.x, z: t.z }; }
        if (cmd.type === 'construct') e.order = { type: 'build', targetId: cmd.targetId };
        if (cmd.type === 'gather') { const n = this.source(cmd.targetId)!; e.order = { type: 'gather', targetId: n.id, resource: n.kind, returning: (e.carrying ?? 0) > 0 && e.carryResource !== n.kind }; }
      });
    } else if (cmd.type === 'build') {
      const def = BUILDING_DEFS[cmd.kind];
      this.pay(team, def.cost);
      const building = this.spawn(team, cmd.kind, cmd.x, cmd.z, 0);
      const workers = cmd.workerIds?.length ? [...new Set(cmd.workerIds)].map(id => this.entity(id)!) : this.entities.filter(e => e.owner === team && e.kind === 'villager' && e.hp > 0).sort((a, b) => distance(a, building) - distance(b, building) || a.id - b.id).slice(0, 1);
      for (const worker of workers) { worker.order = { type: 'build', targetId: building.id }; this.routes.delete(worker.id); }
      this.rebuildObstacles();
      this.log(`${def.name} foundation placed.`, team);
    } else if (cmd.type === 'train') {
      const def = UNIT_DEFS[cmd.kind];
      this.pay(team, def.cost);
      this.entity(cmd.buildingId)!.queue.push({ kind: cmd.kind, remaining: def.time, total: def.time });
    } else if (cmd.type === 'advance') {
      this.pay(team, AGE_COSTS[player.age + 1]);
      player.advancing = AGE_TIMES[player.age + 1];
      this.log(`Advancing to ${AGE_NAMES[player.age + 1]}.`, team);
    } else if (cmd.type === 'research') {
      this.pay(team, UPGRADE_DEFS[cmd.upgrade].cost);
      player.upgrades.push(cmd.upgrade);
      this.log(`${UPGRADE_DEFS[cmd.upgrade].name} researched.`, team);
    }
    return good;
  }

  findBuildSite(team: Team, kind: BuildingKind): Point | undefined {
    const def = BUILDING_DEFS[kind];
    if (!def || this.players[team].age < def.requiredAge || !this.affords(team, def.cost)) return undefined;
    const base = this.entities.find(e => e.owner === team && e.kind === 'towncenter' && e.hp > 0);
    if (!base) return undefined;
    for (let r = 4; r <= 17; r++) for (let dz = -r; dz <= r; dz++) for (let dx = -r; dx <= r; dx++) {
      if (Math.abs(dx) !== r && Math.abs(dz) !== r) continue;
      const p = { x: Math.round(base.x + dx), z: Math.round(base.z + dz) };
      if (this.legalSite(team, kind, p.x, p.z).ok) return p;
    }
    return undefined;
  }

  step(dt = DT) {
    if (!Number.isFinite(dt) || dt <= 0) return;
    for (let left = dt; left > 1e-8 && this.status === 'playing'; left -= DT) this.advanceStep(Math.min(DT, left));
  }

  observe(team: Team): string {
    const rounded = (n: number) => Math.round(n * 10) / 10;
    const compact = (e: Entity) => ({ id: e.id, kind: e.kind, x: rounded(e.x), z: rounded(e.z), hp: rounded(e.hp), construction: rounded(e.construction), ...(e.owner === team ? { order: e.order.type, targetId: e.order.targetId, queue: e.queue.map(q => q.kind), carrying: rounded(e.carrying ?? 0) } : {}) });
    return JSON.stringify({ team, time: rounded(this.time), age: this.players[team].age, advancing: rounded(this.players[team].advancing), resources: this.players[team].resources, upgrades: this.players[team].upgrades, population: this.population(team), capacity: this.capacity(team), own: this.entities.filter(e => e.owner === team && e.hp > 0).map(compact), enemies: this.entities.filter(e => e.owner !== team && e.hp > 0 && this.isVisible(team, e.x, e.z)).map(compact), nodes: this.nodes.filter(n => n.amount > 0 && this.isVisible(team, n.x, n.z) && (n.owner === undefined || n.owner === team)).map(n => ({ id: n.id, kind: n.kind, x: n.x, z: n.z, amount: rounded(n.amount) })) });
  }

  snapshot() {
    return JSON.parse(JSON.stringify({ seed: this.seed, randomState: this.randomState, time: this.time, tick: this.tick, size: this.size, players: this.players, production: this.production, entities: this.entities, nodes: this.nodes, terrain: this.terrain, explored: this.explored.map(v => Array.from(v)), visibility: this.visibility.map(v => Array.from(v)), status: this.status, events: this.events }));
  }

  /** Detached data suitable for a worker message; no simulation state is rounded or rebuilt. */
  checkpoint(): WorldCheckpoint {
    return structuredClone({
      version: 1, size: this.size, seed: this.seed, options: this.options,
      time: this.time, tick: this.tick, players: this.players, production: this.production,
      entities: this.entities, nodes: this.nodes, terrain: this.terrain,
      explored: [Array.from(this.explored[0]), Array.from(this.explored[1])],
      visibility: [Array.from(this.visibility[0]), Array.from(this.visibility[1])],
      status: this.status, events: this.events, nextId: this.nextId, randomState: this.randomState,
      blocked: Array.from(this.blocked), navigationVersion: this.navigationVersion,
      routes: Array.from(this.routes.entries()),
    });
  }

  /** Fork without sharing mutable data with either the source checkpoint or another fork. */
  static fromCheckpoint(checkpoint: WorldCheckpoint): World {
    if (checkpoint.version !== 1 || checkpoint.size !== 64) throw new Error('Unsupported world checkpoint.');
    const state = structuredClone(checkpoint);
    const world = new World(state.seed, state.options);
    world.time = state.time; world.tick = state.tick;
    world.players = state.players;
    world.production[0] = state.production[0]; world.production[1] = state.production[1];
    world.entities = state.entities; world.nodes = state.nodes; world.terrain = state.terrain;
    world.explored = state.explored.map(values => Uint8Array.from(values));
    world.visibility = state.visibility.map(values => Uint8Array.from(values));
    world.status = state.status; world.events = state.events;
    world.nextId = state.nextId; world.randomState = state.randomState;
    world.blocked = Uint8Array.from(state.blocked);
    world.navigationVersion = state.navigationVersion;
    world.routes = new Map(state.routes);
    return world;
  }

  private advanceStep(dt: number) {
    this.time += dt;
    this.tick++;
    for (const team of [0, 1] as const) {
      const player = this.players[team];
      if (player.advancing > 0) {
        player.advancing = Math.max(0, player.advancing - dt);
        if (player.advancing < 1e-7) { player.advancing = 0; player.age++; this.log(`${AGE_NAMES[player.age]} reached.`, team); }
      }
    }
    for (const e of [...this.entities]) {
      if (e.hp <= 0 || e.construction < 1) continue;
      e.cooldown = Math.max(0, e.cooldown - dt);
      if (e.category === 'building') {
        const next = e.queue[0];
        if (next) {
          next.remaining = Math.max(0, next.remaining - dt);
          if (next.remaining < 1e-7) {
            const p = this.spawnPoint(e);
            if (p) {
              const unit = this.spawn(e.owner, next.kind, p.x, p.z);
              e.queue.shift();
              const stats = this.production[e.owner].trained;
              stats[next.kind] = (stats[next.kind] ?? 0) + 1;
              if (unit.kind === 'villager' && this.options.scenario !== 'first-victory') { const node = this.nearestSource(unit, 'food'); if (node) unit.order = { type: 'gather', targetId: node.id, resource: node.kind }; }
            }
          }
        }
        const def = BUILDING_DEFS[e.kind as BuildingKind];
        if (def.damage) { const target = this.nearestEnemy(e, def.range); if (target) this.strike(e, target, def.damage, 1.6); }
      } else this.unitStep(e, dt);
    }
    this.separateUnits();
    const dead = this.entities.filter(e => e.hp <= 0);
    if (dead.length) {
      for (const e of dead) {
        this.routes.delete(e.id);
        if (e.category === 'building') this.log(`${BUILDING_DEFS[e.kind as BuildingKind].name} destroyed.`, e.owner);
        for (const node of this.nodes) if (node.buildingId === e.id) node.amount = 0;
      }
      this.entities = this.entities.filter(e => e.hp > 0);
      if (dead.some(e => e.category === 'building')) this.rebuildObstacles();
    }
    if (this.tick % 2 === 0) this.updateVisibility();
    const alive = ([0, 1] as const).map(team => this.entities.some(e => e.owner === team && e.kind === 'towncenter' && e.hp > 0));
    if (!alive[0] || !alive[1]) {
      this.status = !alive[0] && !alive[1] ? 'draw' : alive[0] ? 'won' : 'lost';
      this.log(this.status === 'won' ? 'Victory! The rival charter has fallen.' : this.status === 'lost' ? 'Defeat. Your Town Center has fallen.' : 'Both charters have fallen.');
    }
  }

  private unitStep(e: Entity, dt: number) {
    const order = e.order, def = UNIT_DEFS[e.kind as UnitKind];
    if (order.type === 'gather') { this.gather(e, dt); return; }
    if (order.type === 'build') {
      const target = this.entity(order.targetId!);
      if (!target || target.owner !== e.owner) { e.order = { type: 'idle' }; return; }
      if (target.construction >= 1) {
        const farm = this.nodes.find(n => n.buildingId === target.id);
        e.order = farm ? { type: 'gather', targetId: farm.id, resource: 'food' } : { type: 'idle' };
      } else if (this.walk(e, target, radius(target) + 1.25, dt)) {
        const amount = Math.min(1 - target.construction, dt / BUILDING_DEFS[target.kind as BuildingKind].time);
        target.construction += amount;
        target.hp = Math.min(target.maxHp, target.hp + amount * target.maxHp * 0.9);
        if (target.construction > 1 - 1e-7) {
          target.construction = 1;
          const stats = this.production[target.owner].constructed, kind = target.kind as BuildingKind;
          stats[kind] = (stats[kind] ?? 0) + 1;
          if (target.kind === 'farm') this.addNode('food', target.x, target.z, 1500, target.owner, target.id);
          this.log(`${BUILDING_DEFS[target.kind as BuildingKind].name} completed.`, target.owner);
        }
      }
      return;
    }
    if (order.type === 'move') { if (this.walk(e, { x: order.x!, z: order.z! }, 0.3, dt, true)) e.order = { type: 'idle' }; return; }
    let target: Entity | undefined;
    if (order.type === 'attack') {
      target = this.entity(order.targetId!);
      if (!target || !this.isVisible(e.owner, target.x, target.z)) {
        if (order.x !== undefined && order.z !== undefined && !this.walk(e, { x: order.x, z: order.z }, 1.2, dt)) return;
        e.order = { type: 'idle' }; return;
      }
      order.x = target.x; order.z = target.z;
    } else if (order.type === 'attackMove' || (order.type === 'idle' && e.kind !== 'villager')) {
      target = this.nearestEnemy(e, Math.max(7, def.range + 2));
    }
    if (target) {
      if (distance(e, target) <= def.range + radius(target) + 0.08 || this.walk(e, target, def.range + radius(target), dt)) this.strike(e, target, def.damage, def.attackRate);
    } else if (order.type === 'attackMove' && this.walk(e, { x: order.x!, z: order.z! }, 0.4, dt, true)) e.order = { type: 'idle' };
  }

  private gather(e: Entity, dt: number) {
    const order = e.order;
    let node = this.source(order.targetId!);
    const carryLimit = this.players[e.owner].upgrades.includes('economy') ? 18 : 10;
    if (!node || node.amount <= 0) {
      if ((e.carrying ?? 0) > 0) order.returning = true;
      else { node = this.nearestSource(e, order.resource ?? 'food'); if (!node) { e.order = { type: 'idle' }; return; } order.targetId = node.id; this.routes.delete(e.id); }
    }
    if ((e.carrying ?? 0) >= carryLimit - 1e-7) order.returning = true;
    if (order.returning) {
      const depot = this.entities.filter(b => b.owner === e.owner && b.kind === 'towncenter' && b.construction === 1 && b.hp > 0).sort((a, b) => distance(e, a) - distance(e, b) || a.id - b.id)[0];
      if (!depot) return;
      if (this.walk(e, depot, radius(depot) + 0.95, dt)) {
        if (e.carryResource) { this.players[e.owner].resources[e.carryResource] += e.carrying ?? 0; this.players[e.owner].gathered[e.carryResource] += e.carrying ?? 0; }
        e.carrying = 0; e.carryResource = undefined; order.returning = false; this.routes.delete(e.id);
      }
    } else if (node && this.walk(e, node, node.buildingId ? radius(this.entity(node.buildingId)!) + 1.25 : 0.85, dt)) {
      const rate = { food: 2.3, wood: 2.1, gold: 1.8, stone: 1.6 }[node.kind] * (this.players[e.owner].upgrades.includes('economy') ? 1.25 : 1);
      const amount = Math.min(rate * dt, node.amount, carryLimit - (e.carrying ?? 0));
      e.carrying = (e.carrying ?? 0) + amount; e.carryResource = node.kind; node.amount = Math.max(0, node.amount - amount);
    }
  }

  private strike(attacker: Entity, target: Entity, base: number, interval: number) {
    if (attacker.cooldown > 0 || target.hp <= 0) return;
    let damage = base;
    if (attacker.kind === 'spearman' && target.kind === 'knight') damage *= 3.5;
    if (attacker.kind === 'archer' && target.kind === 'spearman') damage *= 1.8;
    if (attacker.kind === 'knight' && target.kind === 'archer') damage *= 1.8;
    if (target.category === 'building') damage *= attacker.kind === 'ram' ? 12 : 0.65;
    if (target.kind === 'ram' && (attacker.kind === 'archer' || attacker.category === 'building')) damage *= 0.2;
    if (this.players[attacker.owner].upgrades.includes('weapons')) damage *= 1.25;
    if (this.players[target.owner].upgrades.includes('armor')) damage *= 0.8;
    damage *= 1 + (this.players[attacker.owner].age - 1) * 0.08;
    target.hp -= Math.max(1, damage);
    attacker.cooldown = interval;
    attacker.heading = Math.atan2(target.x - attacker.x, target.z - attacker.z);
    if (target.hp <= 0) this.players[attacker.owner].kills++;
  }

  private nearestEnemy(e: Entity, range: number) {
    let best: Entity | undefined, bestDistance = range;
    for (const other of this.entities) {
      if (other.owner === e.owner || other.hp <= 0 || !this.isVisible(e.owner, other.x, other.z)) continue;
      const d = distance(e, other) - radius(other);
      if (d < bestDistance) { best = other; bestDistance = d; }
    }
    return best;
  }

  private walk(e: Entity, target: Point, stop: number, dt: number, allowApproach = false) {
    if (distance(e, target) <= stop + 0.05) return true;
    const key = `${Math.floor(target.x * 2)},${Math.floor(target.z * 2)},${stop},${allowApproach},${this.navigationVersion}`;
    let route = this.routes.get(e.id);
    if (!route || route.key !== key || (!e.order.path?.length && this.tick >= route.retry)) {
      e.order.path = this.findPath(e, target, stop, allowApproach);
      // Only travel orders may finish at reachable ground when their goal is lost.
      // Combat, gathering, and construction must still reach their target's range.
      route = { key, retry: this.tick + 15, arrival: allowApproach ? e.order.path.at(-1) ?? { x: e.x, z: e.z } : undefined };
      this.routes.set(e.id, route);
    }
    const arrival = route.arrival ?? target;
    let budget = UNIT_DEFS[e.kind as UnitKind].speed * dt;
    const path = e.order.path ?? [];
    while (budget > 1e-8 && path.length) {
      const next = path[0], d = distance(e, next);
      if (!this.walkable(next.x, next.z)) { path.length = 0; this.routes.delete(e.id); break; }
      if (d < 1e-6) { path.shift(); continue; }
      const travel = Math.min(d, budget), x = e.x + (next.x - e.x) / d * travel, z = e.z + (next.z - e.z) / d * travel;
      if (!this.walkable(x, z)) { path.length = 0; this.routes.delete(e.id); break; }
      e.heading = Math.atan2(next.x - e.x, next.z - e.z); e.x = x; e.z = z; budget -= travel;
      if (travel >= d) path.shift();
      if (distance(e, arrival) <= stop + 0.05) break;
    }
    return distance(e, arrival) <= stop + 0.05;
  }

  private findPath(start: Point, target: Point, stop: number, allowApproach = false): Point[] {
    const startIndex = Math.floor(start.z) * this.size + Math.floor(start.x);
    const visited = new Int32Array(4096).fill(-1), queue = new Int32Array(4096);
    let head = 0, tail = 1, end = -1, closest = -1, closestDistance = Infinity;
    queue[0] = startIndex; visited[startIndex] = startIndex;
    while (head < tail) {
      const current = queue[head++], x = current % this.size, z = Math.floor(current / this.size);
      if (!this.blocked[current]) {
        const d = Math.hypot(x + 0.5 - target.x, z + 0.5 - target.z);
        if (d <= Math.max(stop, 0.76)) { end = current; break; }
        if (allowApproach && d < closestDistance) { closest = current; closestDistance = d; }
      }
      for (const next of [x > 0 ? current - 1 : -1, x < 63 ? current + 1 : -1, z > 0 ? current - 64 : -1, z < 63 ? current + 64 : -1]) {
        if (next < 0 || visited[next] !== -1 || this.blocked[next]) continue;
        visited[next] = current; queue[tail++] = next;
      }
    }
    const reachedTarget = end !== -1;
    if (!reachedTarget && allowApproach) end = closest;
    if (end === -1) return [];
    const path: Point[] = [];
    for (let at = end; at !== startIndex; at = visited[at]) path.push({ x: at % this.size + 0.5, z: Math.floor(at / this.size) + 0.5 });
    path.reverse();
    if (!path.length) path.push({ x: end % this.size + 0.5, z: Math.floor(end / this.size) + 0.5 });
    if (reachedTarget && stop < 0.8 && this.walkable(target.x, target.z)) path.push({ x: target.x, z: target.z });
    return path;
  }

  private separateUnits() {
    const units = this.entities.filter(e => e.category === 'unit' && e.hp > 0);
    for (let i = 0; i < units.length; i++) for (let j = i + 1; j < units.length; j++) {
      const a = units[i], b = units[j];
      if (Math.abs(a.x - b.x) > 0.62 || Math.abs(a.z - b.z) > 0.62) continue;
      const dx = b.x - a.x || 0.001, dz = b.z - a.z || 0.001, d = Math.hypot(dx, dz);
      if (d >= 0.62) continue;
      const push = (0.62 - d) * 0.25;
      for (const [unit, sign] of [[a, -1], [b, 1]] as const) {
        const x = unit.x + dx / d * push * sign, z = unit.z + dz / d * push * sign;
        if (this.walkable(x, z)) { unit.x = x; unit.z = z; }
      }
    }
  }

  private legalSite(team: Team, kind: BuildingKind, x: number, z: number): CommandResult {
    const half = BUILDING_DEFS[kind].size / 2;
    if (!Number.isFinite(x) || !Number.isFinite(z) || x - half < 1 || z - half < 1 || x + half >= 63 || z + half >= 63) return fail('Keep the building inside the map.');
    const left = Math.floor(x - half), right = Math.ceil(x + half), top = Math.floor(z - half), bottom = Math.ceil(z + half);
    for (let iz = top; iz < bottom; iz++) for (let ix = left; ix < right; ix++) {
      if (!this.isExplored(team, ix, iz)) return fail('Explore this site first.');
      if (this.terrainAt(ix, iz) === 'water' || this.terrainAt(ix, iz) === 'bridge') return fail('Buildings need dry land.');
      if (this.blocked[iz * this.size + ix]) return fail('A building blocks this site.');
    }
    if (this.entities.some(e => e.hp > 0 && e.category === 'building' && Math.abs(e.x - x) < radius(e) + half + 0.5 && Math.abs(e.z - z) < radius(e) + half + 0.5)) return fail('Leave space between buildings.');
    // Reserve the same cells that navigation will block, including the margin
    // created when an odd-sized building is placed on an integer coordinate.
    if (this.entities.some(e => e.hp > 0 && e.category === 'unit' && e.x > left - 0.35 && e.x < right + 0.35 && e.z > top - 0.35 && e.z < bottom + 0.35)) return fail('Move units clear of the foundation.');
    if (this.nodes.some(n => n.amount > 0 && Math.abs(n.x - x) < half + 0.55 && Math.abs(n.z - z) < half + 0.55)) return fail('A resource deposit blocks this site.');
    return good;
  }

  private spawn(owner: Team, kind: EntityKind, x: number, z: number, construction = 1) {
    const category = kind in UNIT_DEFS ? 'unit' : 'building';
    const def = category === 'unit' ? UNIT_DEFS[kind as UnitKind] : BUILDING_DEFS[kind as BuildingKind];
    const e: Entity = { id: this.nextId++, owner, kind, category, x, z, hp: def.hp * (0.1 + construction * 0.9), maxHp: def.hp, construction, queue: [], order: { type: 'idle' }, cooldown: 0, heading: owner === 0 ? Math.PI / 2 : -Math.PI / 2 };
    this.entities.push(e); return e;
  }
  private spawnPoint(building: Entity): Point | undefined {
    for (let r = Math.ceil(radius(building)) + 1; r < 10; r++) for (let dz = -r; dz <= r; dz++) for (let dx = -r; dx <= r; dx++) {
      if (Math.abs(dx) !== r && Math.abs(dz) !== r) continue;
      const p = { x: Math.floor(building.x) + dx + 0.5, z: Math.floor(building.z) + dz + 0.5 };
      if (this.walkable(p.x, p.z) && !this.entities.some(e => e.category === 'unit' && e.hp > 0 && distance(e, p) < 0.7)) return p;
    }
    return undefined;
  }
  private addNode(kind: Resource, x: number, z: number, amount: number, owner?: Team, buildingId?: number) { const node: ResourceNode = { id: this.nextId++, kind, x, z, amount, maxAmount: amount, ...(owner === undefined ? {} : { owner, buildingId }) }; this.nodes.push(node); return node; }
  private source(id: number) { return this.nodes.find(n => (n.id === id || n.buildingId === id) && (!n.buildingId || this.entity(n.buildingId))); }
  private nearestSource(e: Entity, kind: Resource) { return this.nodes.filter(n => n.kind === kind && n.amount > 0 && (!n.buildingId || this.entity(n.buildingId)) && (n.owner === undefined || n.owner === e.owner) && this.isVisible(e.owner, n.x, n.z)).sort((a, b) => distance(e, a) - distance(e, b) || a.id - b.id)[0]; }
  private affords(team: Team, cost: Partial<Resources>) { return RESOURCES.every(r => this.players[team].resources[r] + 1e-7 >= (cost[r] ?? 0)); }
  private pay(team: Team, cost: Partial<Resources>) { for (const r of RESOURCES) this.players[team].resources[r] -= cost[r] ?? 0; }
  private inBounds(x: number, z: number) { return Number.isFinite(x) && Number.isFinite(z) && x >= 0 && z >= 0 && x < this.size && z < this.size; }
  private terrainAt(x: number, z: number) { return this.terrain[Math.floor(z) * this.size + Math.floor(x)]; }
  private walkable(x: number, z: number) { return this.inBounds(x, z) && !this.blocked[Math.floor(z) * this.size + Math.floor(x)]; }
  private rebuildObstacles() {
    this.blocked = Uint8Array.from(this.terrain, t => t === 'water' ? 1 : 0);
    for (const e of this.entities) if (e.category === 'building' && e.hp > 0) {
      const half = radius(e);
      for (let z = Math.floor(e.z - half); z < Math.ceil(e.z + half); z++) for (let x = Math.floor(e.x - half); x < Math.ceil(e.x + half); x++) if (this.inBounds(x, z)) this.blocked[z * this.size + x] = 1;
    }
    this.navigationVersion++;
  }
  private updateVisibility() {
    // This tutorial is explicitly a surveyed battlefield. There are no hidden
    // enemy reads: the same permanent visibility is used by commands and UI.
    if (this.options.scenario === 'first-victory') {
      for (const team of [0, 1] as const) { this.visibility[team].fill(1); this.explored[team].fill(1); }
      return;
    }
    for (const team of [0, 1] as const) {
      const visible = this.visibility[team]; visible.fill(0);
      for (const e of this.entities) if (e.owner === team && e.hp > 0) {
        const sight = e.category === 'unit' ? UNIT_DEFS[e.kind as UnitKind].vision : e.construction >= 1 ? BUILDING_DEFS[e.kind as BuildingKind].vision : 3;
        for (let z = Math.max(0, Math.floor(e.z - sight)); z <= Math.min(63, Math.ceil(e.z + sight)); z++) for (let x = Math.max(0, Math.floor(e.x - sight)); x <= Math.min(63, Math.ceil(e.x + sight)); x++) if ((x + 0.5 - e.x) ** 2 + (z + 0.5 - e.z) ** 2 <= sight ** 2) { visible[z * this.size + x] = 1; this.explored[team][z * this.size + x] = 1; }
      }
    }
  }
  private random() { this.randomState ^= this.randomState << 13; this.randomState ^= this.randomState >>> 17; this.randomState ^= this.randomState << 5; return (this.randomState >>> 0) / 4294967296; }
  private log(text: string, team?: Team) { this.events.push({ time: this.time, text, ...(team === undefined ? {} : { team }) }); if (this.events.length > 80) this.events.shift(); }
}
