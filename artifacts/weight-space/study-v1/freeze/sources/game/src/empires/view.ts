import * as THREE from 'three';
import { BUILDING_DEFS, UNIT_DEFS, World } from './simulation';
import type { BuildingKind, Entity, Point, ResourceNode, UnitKind } from './types';

const geometry = {
  box: new THREE.BoxGeometry(1, 1, 1),
  cylinder: new THREE.CylinderGeometry(1, 1, 1, 8),
  cone: new THREE.ConeGeometry(1, 1, 7),
  roof: new THREE.ConeGeometry(1, 1, 4),
  rock: new THREE.DodecahedronGeometry(1, 0),
  sphere: new THREE.SphereGeometry(1, 8, 6),
  ring: new THREE.RingGeometry(.82, 1, 36),
  plane: new THREE.PlaneGeometry(1, 1),
};
const materials = new Map<number, THREE.MeshLambertMaterial>();
function material(color: number) {
  if (!materials.has(color)) materials.set(color, new THREE.MeshLambertMaterial({ color }));
  return materials.get(color)!;
}
function mesh(parent: THREE.Object3D, shape: keyof typeof geometry, color: number, scale: [number, number, number], position: [number, number, number], shadow = true) {
  const m = new THREE.Mesh(geometry[shape], material(color));
  m.scale.set(...scale); m.position.set(...position); m.castShadow = shadow; m.receiveShadow = true; parent.add(m); return m;
}
function box(parent: THREE.Object3D, color: number, scale: [number, number, number], position: [number, number, number]) { return mesh(parent, 'box', color, scale, position); }
const hash = (n: number) => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const TEAMS = [0x287fce, 0xc6533f];
const ROOFS = [0x3c7396, 0x9d4e3b];
const UNIT_SCALE = 1.2;
const RESOURCE_COLORS = { food: 0xc97245, wood: 0x89623d, gold: 0xe7be54, stone: 0x9aa9ad };
type Model = {
  root: THREE.Group; body: THREE.Group; ring: THREE.Mesh; health: THREE.Group; fill: THREE.Mesh; scaffold?: THREE.Group;
  orderLine: THREE.Line; destination: THREE.Mesh; damage: THREE.Mesh;
  tick: number; x: number; z: number; hp: number; cooldown: number; carrying: number; construction: number;
  moving: boolean; working: boolean; visible: boolean; attackAt: number; damageAt: number; orderStamp: string;
};
export type Hit = { point: Point; entityId?: number; nodeId?: number };

/** Geometry only. Rendering never advances the simulation or issues an order. */
export class EmpiresView {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.OrthographicCamera(-20, 20, 14, -14, .1, 250);
  readonly focusPoint = new THREE.Vector3(17, 0, 42);
  span = 19;
  private world: World;
  private board = new THREE.Group();
  private models = new Map<number, Model>();
  private resources = new Map<number, THREE.Group>();
  private knownResources = new Map<number, ResourceNode>();
  private lastResourceTick = -1;
  private tiles!: THREE.InstancedMesh;
  private ghost = new THREE.Group();
  private ghostMaterial = new THREE.MeshBasicMaterial({ color: 0xaff6b0, opacity: .27, transparent: true, depthWrite: false });
  private ghostBox = new THREE.Mesh(geometry.box, this.ghostMaterial);
  private orderRing = new THREE.Mesh(geometry.ring, new THREE.MeshBasicMaterial({ color: 0xf5df9c, transparent: true, opacity: 1, depthWrite: false }));
  private pingAge = 10;
  private lastFogTick = -1;
  private raycaster = new THREE.Raycaster();
  private groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  private selectionMat = new THREE.MeshBasicMaterial({ color: 0xffe6a2, transparent: true, opacity: 1, depthWrite: false, side: THREE.DoubleSide });
  private unitMat = new THREE.MeshBasicMaterial({ color: 0x67b9ff, transparent: true, opacity: .55, depthWrite: false, side: THREE.DoubleSide });
  private damageMat = new THREE.MeshBasicMaterial({ color: 0xff654b, transparent: true, opacity: .9, depthWrite: false, side: THREE.DoubleSide });
  private orderMaterials = Object.fromEntries(Object.entries({ move: 0xffe5a1, gather: 0x98dfb0, build: 0xa5d4ff, attack: 0xff9273 }).map(([name, color]) => [name, {
    line: new THREE.LineDashedMaterial({ color, dashSize: .32, gapSize: .2, transparent: true, opacity: .8, depthWrite: false }),
    marker: new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .95, depthWrite: false, side: THREE.DoubleSide }),
  }]));
  private healthMat = [new THREE.MeshBasicMaterial({ color: 0xa6cc9a, depthTest: false }), new THREE.MeshBasicMaterial({ color: 0xdc9a78, depthTest: false })];
  private healthBack = new THREE.MeshBasicMaterial({ color: 0x132b28, transparent: true, opacity: .8, depthTest: false });
  private projection = new THREE.Vector3();
  private settlementSites: Point[] = [];
  private resizeObserver: ResizeObserver;
  private width = 1;
  private height = 1;

  constructor(readonly canvas: HTMLCanvasElement, world: World) {
    this.world = world;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x213d36);
    this.scene.add(new THREE.HemisphereLight(0xfff3d8, 0x556a58, 2.0));
    const sunlight = new THREE.DirectionalLight(0xffe7bd, 2.7);
    sunlight.position.set(-16, 60, 12); sunlight.target.position.set(32, 0, 32);
    sunlight.castShadow = true;
    sunlight.shadow.mapSize.set(2048, 2048);
    sunlight.shadow.camera.left = -50; sunlight.shadow.camera.right = 50;
    sunlight.shadow.camera.top = 50; sunlight.shadow.camera.bottom = -50;
    sunlight.shadow.camera.near = 1; sunlight.shadow.camera.far = 140;
    sunlight.shadow.normalBias = .08; sunlight.shadow.bias = -.0004;
    this.scene.add(sunlight, sunlight.target, this.board, this.ghost, this.orderRing);
    this.ghost.add(this.ghostBox); this.ghost.visible = false;
    this.orderRing.rotation.x = -Math.PI / 2; this.orderRing.visible = false;
    this.createBoard();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas); this.resize(); this.focusHome();
    canvas.addEventListener('webglcontextlost', this.contextLost);
    canvas.addEventListener('webglcontextrestored', this.contextRestored);
  }

  private contextLost = (event: Event) => {
    event.preventDefault();
    window.dispatchEvent(new CustomEvent('empires-render-error', { detail: 'The graphics context was interrupted. Reload the map to resume a new match.' }));
  };
  private contextRestored = () => { this.lastFogTick = -1; };

  setWorld(world: World) {
    this.world = world;
    for (const model of this.models.values()) this.removeModel(model);
    for (const resource of this.resources.values()) this.scene.remove(resource);
    this.models.clear(); this.resources.clear(); this.knownResources.clear(); this.lastResourceTick = -1;
    this.tiles.dispose(); this.board.clear(); this.createBoard();
    this.lastFogTick = -1; this.ghost.visible = false; this.pingAge = 10;
    this.span = 19; this.focusHome(); this.resize();
  }

  private focusHome() {
    const home = this.world.entities.find(e => e.owner === 0 && e.kind === 'towncenter' && e.hp > 0);
    this.focus((home?.x ?? 16) + 1, (home?.z ?? 43) - 1);
  }

  private createBoard() {
    const world = this.world;
    this.settlementSites = world.entities.filter(e => e.kind === 'towncenter').map(e => ({ x: e.x, z: e.z }));
    box(this.board, 0x31463a, [world.size, 2.4, world.size], [world.size / 2, -1.45, world.size / 2]);
    this.tiles = new THREE.InstancedMesh(geometry.box, material(0xffffff), world.size ** 2);
    this.tiles.receiveShadow = true;
    this.tiles.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    const matrix = new THREE.Matrix4(), quaternion = new THREE.Quaternion();
    for (let z = 0; z < world.size; z++) for (let x = 0; x < world.size; x++) {
      const i = z * world.size + x;
      const water = world.terrain[i] === 'water';
      matrix.compose(new THREE.Vector3(x + .5, water ? -.22 : -.11, z + .5), quaternion, new THREE.Vector3(1.006, water ? .09 : .22, 1.006));
      this.tiles.setMatrixAt(i, matrix); this.tiles.setColorAt(i, new THREE.Color(0x253d34));
    }
    this.board.add(this.tiles);
    // Bank posts follow the actual terrain, including the beginner map's wider crossing.
    for (let z = 0; z < world.size; z++) {
      const xs = Array.from({ length: world.size }, (_, x) => x).filter(x => world.terrain[z * world.size + x] === 'bridge');
      if (!xs.length) continue;
      const borders = [];
      if (z === 0 || world.terrain[(z - 1) * world.size + xs[0]] !== 'bridge') borders.push(z + .3);
      if (z === world.size - 1 || world.terrain[(z + 1) * world.size + xs[0]] !== 'bridge') borders.push(z + .7);
      for (const x of [xs[0] - .3, xs.at(-1)! + 1.3]) for (const edge of borders) {
        const p = new THREE.Group(); p.position.set(x, 0, edge); p.userData.bridge = true;
        box(p, 0x877654, [.2, 1.0, .2], [0, .5, 0]);
        mesh(p, 'sphere', 0xd2ae6c, [.17, .13, .17], [0, 1.04, 0]); this.board.add(p);
      }
    }
    this.rememberResources();
    for (const node of this.knownResources.values()) this.addResource(node);
  }

  private rememberResources() {
    if (this.lastResourceTick === this.world.tick) return;
    this.lastResourceTick = this.world.tick;
    // Explored fog keeps the last observed deposit, never its live hidden amount.
    for (const node of this.world.nodes) if (this.world.isVisible(0, node.x, node.z)) this.knownResources.set(node.id, { ...node });
  }

  private addResource(node: ResourceNode) {
    const root = new THREE.Group(); root.position.set(node.x, 0, node.z);
    root.userData.nodeId = node.id; root.rotation.y = hash(node.id) * Math.PI * 2;
    if (node.kind === 'wood') {
      const height = 1.8 + hash(node.id * 4) * 1.7;
      const autumn = hash(node.id + 100) > .78;
      mesh(root, 'cylinder', 0x735741, [.13, height * .75, .13], [0, height * .38, 0]);
      if (hash(node.id * 2) < .52) {
        mesh(root, 'cone', autumn ? 0xbd9451 : 0x4c7252, [.83, height * .82, .83], [0, height * .72, 0]);
        mesh(root, 'cone', autumn ? 0xd3ac62 : 0x608465, [.65, height * .65, .65], [0, height, 0]);
      } else {
        mesh(root, 'rock', autumn ? 0xc7a360 : 0x718957, [.88, height * .44, .82], [0, height * .8, 0]);
        mesh(root, 'rock', autumn ? 0xd6ba75 : 0x85955c, [.59, height * .34, .64], [.15, height * 1.05, -.1]);
      }
    } else if (node.kind === 'food') {
      if (!node.buildingId) for (let i = 0; i < 5; i++) {
        const angle = i * 1.3, x = Math.cos(angle) * .6, z = Math.sin(angle) * .5;
        mesh(root, 'rock', 0x63774e, [.47, .46, .45], [x, .4, z]);
        for (let b = 0; b < 3; b++) mesh(root, 'sphere', 0xb66b53, [.09, .09, .09], [x + (b - 1) * .19, .71, z - .12]);
      }
    } else {
      for (let i = 0; i < 5; i++) {
        const angle = i * 2.4, h = .36 + hash(node.id * 9 + i) * .5;
        const r = mesh(root, 'rock', node.kind === 'gold' ? [0xc0a570, 0x9d9475, 0xd4b871][i % 3] : [0x8b9390, 0xa8aaa0, 0x738681][i % 3], [.45 + h * .3, h, .46], [Math.cos(angle) * .55, h * .65, Math.sin(angle) * .5]);
        r.rotation.set(hash(i) * .3, angle, hash(i + 1) * .4);
      }
    }
    root.visible = node.amount > 0 && !node.buildingId;
    this.resources.set(node.id, root); this.scene.add(root);
  }

  private addEntity(entity: Entity): Model {
    const root = new THREE.Group(), body = new THREE.Group();
    root.userData.entityId = entity.id; root.add(body);
    if (entity.category === 'building') this.building(body, entity.kind as BuildingKind, entity.owner);
    else { this.unit(body, entity.kind as UnitKind, entity.owner); body.scale.setScalar(UNIT_SCALE); }
    const r = entity.category === 'building' ? BUILDING_DEFS[entity.kind as BuildingKind].size * .72 : .73;
    const ring = new THREE.Mesh(geometry.ring, this.unitMat); ring.rotation.x = -Math.PI / 2;
    ring.position.y = .06; ring.scale.setScalar(r); ring.renderOrder = 2; root.add(ring);
    const damage = new THREE.Mesh(geometry.ring, this.damageMat); damage.rotation.x = -Math.PI / 2;
    damage.position.y = .07; damage.visible = false; damage.userData.radius = r; damage.renderOrder = 3; root.add(damage);
    const orderLine = new THREE.Line(new THREE.BufferGeometry(), this.orderMaterials.move.line);
    orderLine.visible = false; orderLine.renderOrder = 1; root.add(orderLine);
    const destination = new THREE.Mesh(geometry.ring, this.orderMaterials.move.marker);
    destination.rotation.x = -Math.PI / 2; destination.scale.setScalar(.52); destination.visible = false; root.add(destination);
    const health = new THREE.Group();
    const healthWidth = entity.category === 'building' ? 2.0 : 1.15;
    const background = new THREE.Mesh(geometry.plane, this.healthBack); background.scale.set(healthWidth + .1, .16, 1); background.renderOrder = 10; health.add(background);
    const fill = new THREE.Mesh(geometry.plane, this.healthMat[entity.owner]); fill.scale.set(healthWidth, .09, 1); fill.position.z = .01; fill.renderOrder = 11; health.add(fill);
    health.userData.width = healthWidth;
    health.position.y = this.labelHeight(entity);
    root.add(health);
    let scaffold: THREE.Group | undefined;
    if (entity.category === 'building') {
      scaffold = new THREE.Group(); root.add(scaffold);
      const size = BUILDING_DEFS[entity.kind as BuildingKind].size;
      for (const x of [-size / 2, size / 2]) for (const z of [-size / 2, size / 2]) box(scaffold, 0x9b875f, [.09, 2.7, .09], [x, 1.35, z]);
      for (const z of [-size / 2, size / 2]) for (const y of [.7, 1.9]) box(scaffold, 0xae9867, [size, .08, .08], [0, y, z]);
      for (const x of [-size / 2, size / 2]) box(scaffold, 0xae9867, [.08, .08, size], [x, 1.9, 0]);
    }
    const model: Model = {
      root, body, ring, health, fill, scaffold, orderLine, destination, damage,
      tick: -1, x: entity.x, z: entity.z, hp: entity.hp, cooldown: entity.cooldown,
      carrying: entity.carrying ?? 0, construction: 0, moving: false, working: false,
      visible: false, attackAt: -10, damageAt: -10, orderStamp: '',
    };
    this.models.set(entity.id, model); this.scene.add(root); return model;
  }

  /** Shared projection height for DOM labels and the renderer's health bars. */
  labelHeight(entity: Entity) {
    if (entity.category === 'unit') return entity.kind === 'spearman' ? 2.7 : entity.kind === 'knight' ? 2.5 : 1.9;
    return entity.kind === 'castle' ? 5.1 : entity.kind === 'tower' ? 4.5 : entity.kind === 'towncenter' ? 4.05 : entity.kind === 'farm' ? .8 : 2.85;
  }

  private flag(root: THREE.Object3D, team: number, x: number, y: number, z: number) {
    box(root, 0x655b48, [.055, .94, .055], [x, y + .42, z]);
    const f = box(root, TEAMS[team], [.55, .32, .035], [x + .27, y + .7, z]); f.userData.flag = true;
    box(root, 0xe5ce8c, [.055, .055, .055], [x, y + .92, z]);
  }

  private building(root: THREE.Group, kind: BuildingKind, team: number) {
    const def = BUILDING_DEFS[kind], size = def.size;
    box(root, 0x9f9875, [size + .15, .12, size + .15], [0, .02, 0]);
    const house = (x: number, z: number, width: number, depth: number, height: number) => {
      box(root, 0xd1c3a0, [width, height, depth], [x, height / 2 + .1, z]);
      for (const dx of [-width * .43, width * .43]) box(root, 0x826d50, [.1, height, depth + .03], [x + dx, height / 2 + .1, z]);
      box(root, 0x756247, [width + .08, .12, depth + .08], [x, .25, z]);
      box(root, 0x8f7c58, [width + .06, .12, depth + .06], [x, height + .04, z]);
      const roof = mesh(root, 'roof', ROOFS[team], [width * .85, .95, depth * .85], [x, height + .53, z]); roof.rotation.y = Math.PI / 4;
      box(root, 0x524b3c, [.38, .65, .04], [x, .45, z + depth / 2 + .03]);
      for (const dx of [-width * .3, width * .3]) box(root, 0x55686a, [.24, .29, .04], [x + dx, height * .65 + .1, z + depth / 2 + .04]);
    };
    const tower = (x: number, z: number, height: number, width: number) => {
      box(root, 0xb4b59e, [width, height, width], [x, height / 2, z]);
      box(root, 0xd2ceb3, [width + .22, .25, width + .22], [x, height -.18, z]);
      for (let i = -1; i <= 1; i++) for (const side of [-1, 1]) {
        box(root, 0xc4c2aa, [.21, .4, .23], [x + i * width * .37, height + .1, z + side * width * .46]);
        if (i === 0) box(root, 0xc4c2aa, [.23, .4, .21], [x + side * width * .46, height + .1, z]);
      }
      box(root, 0x405554, [.13, .48, .03], [x, height * .62, z + width / 2 + .02]);
    };
    if (kind === 'farm') {
      box(root, 0x756140, [2.8, .08, 2.8], [0, .11, 0]);
      for (let z = -1.1; z <= 1.1; z += .38) {
        box(root, 0xceb268, [2.55, .13, .13], [0, .24, z]);
        for (let x = -1; x <= 1; x += .28) mesh(root, 'cone', 0xcfc37b, [.09, .3, .09], [x, .36, z]);
      }
      for (const z of [-1.47, 1.47]) box(root, 0x9f8d5d, [3, .11, .08], [0, .37, z]);
    } else if (kind === 'towncenter') {
      house(0, 0, 2.7, 2.1, 1.6); house(-1.35, .55, 1.0, 1.1, .9);
      box(root, 0xd3c9a3, [.82, 1.2, .82], [.72, 2.2, -.4]);
      const spire = mesh(root, 'roof', ROOFS[team], [.72, .95, .72], [.72, 3.15, -.4]); spire.rotation.y = Math.PI / 4;
      box(root, 0xb69d6a, [2.1, .12, .7], [0, .16, 1.4]);
      for (const x of [-1.2, 1.2]) box(root, 0x856f4d, [.1, 1.18, .1], [x, .7, 1.35]);
      box(root, TEAMS[team], [2.6, .12, .9], [0, 1.35, 1.3]);
      this.flag(root, team, -.65, 2.75, .05);
      for (const x of [-1.2, 1.3]) mesh(root, 'cylinder', 0x8d7c53, [.24, .43, .24], [x, .3, 1.8]);
    } else if (kind === 'house') {
      house(0, 0, 1.65, 1.5, 1.1); box(root, 0xa49c81, [.22, .8, .25], [.5, 1.85, -.25]);
    } else if (kind === 'tower') {
      tower(0, 0, 3.25, 1.35); this.flag(root, team, 0, 3.2, 0);
    } else if (kind === 'castle') {
      box(root, 0xb7b79f, [3.85, 1.9, 3.85], [0, .95, 0]);
      for (const x of [-1.8, 1.8]) for (const z of [-1.8, 1.8]) tower(x, z, 3.2, 1.05);
      house(0, -.1, 1.8, 1.7, 3.2);
      box(root, 0x655e48, [1.05, 1.32, .1], [0, .66, 1.98]);
      for (let x = -.45; x <= .45; x += .15) box(root, 0x403d31, [.055, 1.25, .08], [x, .7, 2.03]);
      this.flag(root, team, 0, 3.85, -.1);
    } else if (kind === 'barracks') {
      house(-.4, -.25, 1.75, 2.0, 1.1);
      for (const z of [-1, 1]) box(root, 0x7c6b49, [.07, .9, .07], [1.05, .45, z]);
      box(root, 0x7c6b49, [.1, .1, 2.2], [1.05, .7, 0]);
      for (const z of [-.7, 0, .7]) {
        const p = box(root, 0xb8a878, [.05, 1.3, .05], [1.1, .7, z]); p.rotation.z = -.14;
        mesh(root, 'cone', 0xc5c8b9, [.09, .23, .09], [1.0, 1.38, z]);
      }
      this.flag(root, team, -.4, 1.88, -.2);
    } else if (kind === 'range') {
      house(-.55, -.4, 1.6, 1.4, 1.15);
      for (const z of [-.4, .7]) {
        box(root, 0x9a8056, [.08, 1.2, .08], [1, .6, z]);
        const target = mesh(root, 'cylinder', 0xe0d4a2, [.37, .09, .37], [1, 1.1, z]); target.rotation.z = Math.PI / 2;
        const center = mesh(root, 'cylinder', TEAMS[team], [.16, .1, .16], [.94, 1.1, z]); center.rotation.z = Math.PI / 2;
      }
      this.flag(root, team, -.5, 1.95, -.4);
    } else if (kind === 'stable') {
      house(-.25, -.35, 2.0, 1.5, 1.15);
      for (const x of [-1.3, 1.3]) for (const z of [.4, 1.3]) box(root, 0xa48e60, [.09, .8, .09], [x, .4, z]);
      for (const z of [.45, 1.3]) box(root, 0xa48e60, [2.7, .1, .08], [0, .5, z]);
      this.flag(root, team, -.6, 2.0, -.4);
    } else if (kind === 'forge') {
      house(-.4, -.15, 1.8, 2, 1.1);
      box(root, 0x827b67, [.63, 2.2, .63], [.9, 1.1, -.9]);
      box(root, 0x625c4b, [.8, .3, .8], [.9, 2.12, -.9]);
      box(root, 0x735439, [.7, .7, .7], [.9, .35, .2]);
      box(root, 0xeda36b, [.4, .25, .025], [.9, .4, .565]);
      box(root, 0x434e4c, [.7, .17, .32], [.7, .64, 1]);
      box(root, 0x434e4c, [.24, .47, .2], [.7, .35, 1]);
    }
  }

  private unit(root: THREE.Group, kind: UnitKind, team: number) {
    const color = TEAMS[team], skin = 0xd8b18a, metal = 0xc3cfce;
    if (kind === 'ram') {
      box(root, 0x836742, [1.12, .7, 1.6], [0, .65, 0]);
      mesh(root, 'roof', color, [.88, .55, 1.26], [0, 1.25, 0]).rotation.y = Math.PI / 4;
      const log = mesh(root, 'cylinder', 0x66553b, [.18, 2.4, .18], [0, .65, .3]); log.rotation.x = Math.PI / 2;
      for (const x of [-.65, .65]) for (const z of [-.55, .55]) { const wheel = mesh(root, 'cylinder', 0x433e31, [.26, .12, .26], [x, .3, z]); wheel.rotation.z = Math.PI / 2; }
      return;
    }
    const rider = kind === 'knight', y = rider ? .65 : 0;
    if (rider) {
      box(root, 0x826247, [.49, .55, 1.07], [0, .74, 0]);
      mesh(root, 'rock', 0x997654, [.25, .45, .32], [0, 1.17, .52]);
      box(root, 0x72503a, [.32, .25, .48], [0, 1.27, .66]);
      for (const x of [-.2, .2]) for (const z of [-.37, .37]) {
        const leg = new THREE.Group(); leg.position.set(x, .55, z); leg.userData.leg = Math.sign(x * z); root.add(leg);
        box(leg, 0x674b36, [.12, .53, .13], [0, -.25, 0]);
      }
      box(root, color, [.58, .12, .7], [0, 1.04, -.1]);
    } else {
      for (const side of [-1, 1]) {
        const leg = new THREE.Group(); leg.position.set(side * .13, .45, 0); leg.userData.leg = side; root.add(leg);
        box(leg, 0x4c4940, [.17, .4, .18], [0, -.2, 0]);
        box(leg, 0x3d3c35, [.18, .1, .24], [0, -.4, .04]);
      }
    }
    mesh(root, 'cylinder', kind === 'villager' ? 0xe0cfaa : color, [.28, .56, .26], [0, .68 + y, 0]);
    // Broad team-colored aprons and shields remain recognizable at tactical zoom.
    box(root, color, [.42, .38, .07], [0, .65 + y, .26]);
    box(root, 0x6a563e, [.51, .08, .46], [0, .5 + y, 0]);
    mesh(root, 'sphere', skin, [.18, .21, .18], [0, 1.03 + y, 0]);
    if (kind === 'villager') {
      mesh(root, 'cylinder', 0xe0c988, [.3, .07, .3], [0, 1.18, 0]);
      mesh(root, 'sphere', 0xc7ab69, [.2, .13, .2], [0, 1.24, 0]);
      const cargo = new THREE.Group(); cargo.name = 'cargo'; cargo.position.set(0, .74, -.35); cargo.visible = false; root.add(cargo);
      box(cargo, 0xb4915d, [.39, .32, .22], [0, 0, 0]);
      const load = mesh(cargo, 'rock', RESOURCE_COLORS.food, [.24, .2, .16], [0, .21, 0]); load.name = 'load';
    } else {
      mesh(root, 'sphere', metal, [.22, .16, .22], [0, 1.16 + y, 0]);
      box(root, color, [.11, .22, .31], [0, 1.31 + y, 0]);
      box(root, metal, [.44, .17, .12], [0, .87 + y, .2]);
    }
    const leftArm = new THREE.Group(), rightArm = new THREE.Group();
    for (const [arm, side] of [[leftArm, -1], [rightArm, 1]] as const) {
      arm.position.set(side * .32, .88 + y, 0); arm.userData.arm = side; root.add(arm);
      box(arm, kind === 'villager' ? 0xe0cfaa : color, [.16, .22, .18], [0, -.07, 0]);
      box(arm, skin, [.13, .2, .14], [0, -.24, .06]);
    }
    rightArm.userData.tool = true;
    if (kind === 'spearman') {
      box(rightArm, 0x806242, [.07, 1.9, .07], [.03, .14, .2]);
      mesh(rightArm, 'cone', metal, [.12, .32, .12], [.03, 1.21, .2]);
      box(leftArm, color, [.43, .58, .12], [-.04, -.14, .2]);
      box(leftArm, 0xe7dcb6, [.09, .5, .04], [-.04, -.14, .28]);
      box(leftArm, metal, [.16, .13, .05], [-.04, -.14, .3]);
    } else if (kind === 'archer') {
      const bow = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(.02, -.56, .23), new THREE.Vector3(.02, -.19, .44), new THREE.Vector3(.02, .19, .23)]), new THREE.LineBasicMaterial({ color: 0xe1bd75 }));
      rightArm.add(bow); bow.userData.disposable = true;
      box(root, 0x7b5a3d, [.2, .66, .18], [0, .75, -.29]);
    } else if (kind === 'knight') {
      box(rightArm, metal, [.08, .9, .15], [.02, .04, .2]);
      box(rightArm, 0xe2bf78, [.27, .07, .1], [.02, -.28, .2]);
      box(leftArm, color, [.15, .62, .43], [-.02, -.18, .06]);
    } else {
      box(rightArm, 0x785b3e, [.06, .85, .06], [.02, -.21, .17]);
      box(rightArm, metal, [.31, .17, .09], [.07, .2, .17]);
    }
  }

  private updateFog() {
    const color = new THREE.Color();
    for (let i = 0; i < this.world.terrain.length; i++) {
      const terrain = this.world.terrain[i], x = i % this.world.size, z = Math.floor(i / this.world.size);
      if (!this.world.explored[0][i]) color.setHex(0x213c35);
      else {
        if (terrain === 'water') color.setHex([0x508d8b, 0x558f8d, 0x5b9690, 0x649c95][(x + z) % 4]);
        else if (terrain === 'bridge') color.setHex(z % 2 ? 0xb3a079 : 0xc1b08b);
        else {
          const nearTown = Math.min(...this.settlementSites.map(site => Math.hypot(x - site.x, z - site.z)));
          const road = nearTown < 4;
          color.setHex(road ? [0xb0ad80, 0xaaa57a, 0xb5ae85][i % 3] : [0x89976d, 0x8e9b70, 0x849369, 0x909a72, 0x8b956d][Math.floor(hash(i) * 5)]);
          if (terrain === 'hill') color.multiplyScalar(.95);
        }
        if (!this.world.visibility[0][i]) color.lerp(new THREE.Color(0x223d36), .62);
      }
      this.tiles.setColorAt(i, color);
    }
    if (this.tiles.instanceColor) this.tiles.instanceColor.needsUpdate = true;
    for (const child of this.board.children) if (child.userData.bridge) child.visible = this.world.isExplored(0, child.position.x, child.position.z);
  }

  private observeAction(entity: Entity, model: Model) {
    if (model.tick === this.world.tick && model.visible) return;
    const target = entity.order.type === 'build' ? this.world.entity(entity.order.targetId!) : undefined;
    const carrying = entity.carrying ?? 0;
    const construction = target?.construction ?? 0;
    if (model.visible) {
      // An order alone is not evidence that a worker is moving or producing anything.
      model.moving = Math.hypot(entity.x - model.x, entity.z - model.z) > .015;
      model.working = entity.order.type === 'gather' && !entity.order.returning && carrying > model.carrying + .001;
      if (entity.order.type === 'build' && target) {
        model.working = construction > model.construction + .00001
          && Math.hypot(entity.x - target.x, entity.z - target.z) <= BUILDING_DEFS[target.kind as BuildingKind].size / 2 + 1.4;
      }
      if (entity.cooldown > model.cooldown + .01) model.attackAt = this.world.time;
      if (entity.hp < model.hp - .01) model.damageAt = this.world.time;
    } else {
      model.moving = false; model.working = false; model.attackAt = -10; model.damageAt = -10;
    }
    model.x = entity.x; model.z = entity.z; model.hp = entity.hp; model.cooldown = entity.cooldown;
    model.carrying = carrying; model.construction = construction; model.tick = this.world.tick;
  }

  private animateUnit(entity: Entity, model: Model) {
    const phase = this.world.time * 11 + entity.id * 1.7;
    const strikeAge = this.world.time - model.attackAt;
    const strike = strikeAge >= 0 && strikeAge < .4 ? Math.sin(strikeAge / .4 * Math.PI) : 0;
    const working = model.working && (entity.order.type === 'gather' || entity.order.type === 'build');
    let heading = entity.heading;
    if (working) {
      const target = entity.order.type === 'build' ? this.world.entity(entity.order.targetId!) : this.knownResources.get(entity.order.targetId!);
      if (target) heading = Math.atan2(target.x - entity.x, target.z - entity.z);
    }
    model.body.rotation.set(working ? .08 : 0, heading, 0);
    model.body.position.set(Math.sin(heading) * strike * .22, model.moving ? Math.abs(Math.sin(phase)) * .045 : 0, Math.cos(heading) * strike * .22);
    for (const part of model.body.children) {
      if (part.userData.leg) part.rotation.x = model.moving ? Math.sin(phase) * .55 * part.userData.leg : 0;
      if (part.userData.arm) part.rotation.x = model.moving ? -Math.sin(phase) * .35 * part.userData.arm : 0;
      if (part.userData.tool) part.rotation.x = strike ? -strike * 1.1 : working ? -.35 - (Math.sin(phase) + 1) * .5 : part.rotation.x;
      if (part.name === 'cargo') {
        part.visible = (entity.carrying ?? 0) > .1;
        const load = part.getObjectByName('load') as THREE.Mesh;
        load.material = material(RESOURCE_COLORS[entity.carryResource ?? 'food']);
      }
    }
  }

  private showOrder(entity: Entity, model: Model, selected: boolean) {
    const order = entity.order;
    const show = selected && entity.owner === 0 && entity.category === 'unit' && order.type !== 'idle';
    model.orderLine.visible = false; model.destination.visible = false;
    if (!show) { model.orderStamp = ''; return; }
    let target: Point | undefined;
    if (order.type === 'gather') {
      target = order.returning ? this.world.entities.filter(e => e.owner === entity.owner && e.kind === 'towncenter' && e.hp > 0 && e.construction === 1)
        .sort((a, b) => Math.hypot(a.x - entity.x, a.z - entity.z) - Math.hypot(b.x - entity.x, b.z - entity.z))[0] : this.knownResources.get(order.targetId!);
    } else if (order.type === 'build') {
      target = this.world.entity(order.targetId!);
    } else if (order.type === 'attack') {
      const enemy = this.world.entity(order.targetId!);
      // Lost enemy vision leaves only the last location in the player's actual order.
      target = enemy && this.world.isVisible(0, enemy.x, enemy.z) ? enemy : order.x === undefined || order.z === undefined ? undefined : { x: order.x, z: order.z };
    } else if (order.x !== undefined && order.z !== undefined) target = { x: order.x, z: order.z };
    if (!target) return;
    const style = this.orderMaterials[order.type === 'attack' || order.type === 'attackMove' ? 'attack' : order.type];
    model.orderLine.material = style.line; model.destination.material = style.marker;
    model.destination.scale.setScalar('category' in target && target.category === 'building' ? BUILDING_DEFS[(target as Entity).kind as BuildingKind].size * .74 : .65);
    const stamp = `${this.world.tick}/${order.type}/${target.x}/${target.z}/${order.returning}`;
    if (model.orderStamp !== stamp) {
      const points = [new THREE.Vector3(0, .09, 0)];
      for (const p of order.path ?? []) points.push(new THREE.Vector3(p.x - entity.x, .09, p.z - entity.z));
      points.push(new THREE.Vector3(target.x - entity.x, .09, target.z - entity.z));
      model.orderLine.geometry.dispose();
      model.orderLine.geometry = new THREE.BufferGeometry().setFromPoints(points);
      model.orderLine.computeLineDistances();
      model.destination.position.set(target.x - entity.x, .1, target.z - entity.z);
      model.orderStamp = stamp;
    }
    model.orderLine.visible = true; model.destination.visible = true;
  }

  render(world: World, selected: Set<number>, dt: number) {
    this.world = world;
    this.pingAge += dt;
    if (this.lastFogTick < 0 || world.tick - this.lastFogTick >= 3) { this.updateFog(); this.lastFogTick = world.tick; }
    const live = new Set<number>();
    for (const e of world.entities) {
      if (e.hp <= 0) continue;
      live.add(e.id); const m = this.models.get(e.id) ?? this.addEntity(e);
      m.root.visible = e.owner === 0 || world.isVisible(0, e.x, e.z);
      if (!m.root.visible) { m.visible = false; continue; }
      this.observeAction(e, m); m.visible = true;
      m.root.position.set(e.x, 0, e.z);
      if (e.category === 'building') m.body.scale.y = .12 + e.construction * .88;
      else this.animateUnit(e, m);
      if (m.scaffold) m.scaffold.visible = e.construction < 1;
      const isSelected = selected.has(e.id);
      m.ring.visible = isSelected || (e.category === 'unit' && e.owner === 0);
      m.ring.material = isSelected ? this.selectionMat : this.unitMat;
      const damageAge = world.time - m.damageAt;
      m.damage.visible = damageAge >= 0 && damageAge < .45;
      if (m.damage.visible) m.damage.scale.setScalar(m.damage.userData.radius * (1 + damageAge * 1.8));
      m.health.visible = isSelected || e.hp < e.maxHp || e.construction < 1;
      m.health.quaternion.copy(this.camera.quaternion);
      const fraction = Math.max(0, Math.min(1, e.hp / e.maxHp)), w = m.health.userData.width;
      m.fill.scale.x = fraction * w; m.fill.position.x = -(1 - fraction) * w / 2;
      this.showOrder(e, m, isSelected);
      for (const part of m.body.children) if (part.userData.flag) part.rotation.z = Math.sin(world.time * 2.2 + e.id) * .045;
    }
    for (const [id, m] of this.models) if (!live.has(id)) {
      this.removeModel(m);
      this.models.delete(id);
    }
    this.rememberResources();
    for (const node of this.knownResources.values()) {
      if (!this.resources.has(node.id)) this.addResource(node);
      const root = this.resources.get(node.id)!;
      root.visible = node.amount > 0 && !node.buildingId;
      if (node.kind === 'wood') root.scale.setScalar(.7 + .3 * Math.max(0, node.amount / node.maxAmount));
    }
    this.orderRing.visible = this.pingAge < .75;
    if (this.orderRing.visible) { this.orderRing.scale.setScalar(.35 + this.pingAge * 1.35); (this.orderRing.material as THREE.MeshBasicMaterial).opacity = 1 - this.pingAge / .75; }
    this.renderer.render(this.scene, this.camera);
  }

  focus(x: number, z: number) { this.focusPoint.set(THREE.MathUtils.clamp(x, 1, this.world.size - 1), 0, THREE.MathUtils.clamp(z, 1, this.world.size - 1)); this.positionCamera(); }
  pan(horizontal: number, vertical: number) { this.focus(this.focusPoint.x + horizontal * .768 + vertical * .64, this.focusPoint.z - horizontal * .64 + vertical * .768); }
  zoom(delta: number) { this.span = THREE.MathUtils.clamp(this.span * Math.exp(delta * .001), 13, 55); this.resize(); }
  private positionCamera() { this.camera.position.copy(this.focusPoint).add(new THREE.Vector3(32, 43, 39)); this.camera.lookAt(this.focusPoint); this.camera.updateMatrixWorld(); }
  resize() {
    this.width = this.canvas.clientWidth || innerWidth; this.height = this.canvas.clientHeight || innerHeight;
    this.renderer.setSize(this.width, this.height, false);
    const aspect = this.width / this.height;
    this.camera.left = -this.span * aspect / 2; this.camera.right = this.span * aspect / 2;
    this.camera.top = this.span / 2; this.camera.bottom = -this.span / 2; this.camera.updateProjectionMatrix();
  }
  ground(clientX: number, clientY: number): Point | null {
    const rect = this.canvas.getBoundingClientRect();
    this.raycaster.setFromCamera(new THREE.Vector2((clientX - rect.left) / rect.width * 2 - 1, -(clientY - rect.top) / rect.height * 2 + 1), this.camera);
    const point = this.raycaster.ray.intersectPlane(this.groundPlane, new THREE.Vector3());
    if (!point || point.x < 0 || point.z < 0 || point.x >= this.world.size || point.z >= this.world.size) return null;
    return { x: point.x, z: point.z };
  }
  pick(clientX: number, clientY: number): Hit | null {
    const point = this.ground(clientX, clientY); if (!point) return null;
    const pickables: THREE.Object3D[] = [...this.models.values()].filter(m => m.root.visible).map(m => m.body);
    pickables.push(...[...this.resources.values()].filter(r => r.visible && this.world.isVisible(0, r.position.x, r.position.z)));
    const hit = this.raycaster.intersectObjects(pickables, true)[0];
    if (hit) {
      let object: THREE.Object3D | null = hit.object;
      while (object) {
        if (object.userData.entityId !== undefined) return { point, entityId: object.userData.entityId };
        if (object.userData.nodeId !== undefined) return { point, nodeId: object.userData.nodeId };
        object = object.parent;
      }
    }
    // A little screen-space generosity makes small moving units easy to select.
    let best: Entity | undefined, bestDistance = 20;
    for (const e of this.world.entities) if (e.hp > 0 && (e.owner === 0 || this.world.isVisible(0, e.x, e.z))) {
      const p = this.screen(e.x, e.z, e.category === 'unit' ? .8 : 1.0);
      const distance = Math.hypot(clientX - p.x, clientY - p.y);
      if (distance < bestDistance) { best = e; bestDistance = distance; }
    }
    return best ? { point, entityId: best.id } : { point };
  }
  screen(x: number, z: number, height = 0) {
    this.projection.set(x, height, z).project(this.camera);
    const rect = this.canvas.getBoundingClientRect();
    return { x: rect.left + (this.projection.x + 1) / 2 * this.width, y: rect.top + (1 - this.projection.y) / 2 * this.height };
  }
  preview(kind: BuildingKind | null, point?: Point | null, valid = false) {
    this.ghost.visible = !!kind && !!point;
    if (!kind || !point) return;
    const size = BUILDING_DEFS[kind].size;
    this.ghost.position.set(point.x, .3, point.z); this.ghostBox.scale.set(size, .6, size);
    this.ghostMaterial.color.setHex(valid ? 0xbbe6a0 : 0xe69c83);
  }
  ping(point: Point, attack = false) { this.pingAge = 0; this.orderRing.position.set(point.x, .12, point.z); (this.orderRing.material as THREE.MeshBasicMaterial).color.setHex(attack ? 0xf2a984 : 0xf3db91); }

  minimap(canvas: HTMLCanvasElement, selected: Set<number>) {
    this.rememberResources();
    const context = canvas.getContext('2d')!, scale = canvas.width / this.world.size;
    context.fillStyle = '#172d2b'; context.fillRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < this.world.terrain.length; i++) {
      if (!this.world.explored[0][i]) continue;
      const terrain = this.world.terrain[i];
      context.fillStyle = terrain === 'water' ? '#649997' : terrain === 'bridge' ? '#c6b18c' : this.world.visibility[0][i] ? '#818e66' : '#465b46';
      context.fillRect(i % this.world.size * scale, Math.floor(i / this.world.size) * scale, scale, scale);
    }
    for (const node of this.knownResources.values()) if (node.amount > 0) {
      context.fillStyle = node.kind === 'gold' ? '#d9bd72' : node.kind === 'stone' ? '#bac0af' : node.kind === 'food' ? '#c59d6e' : '#465f41';
      context.fillRect(node.x * scale - 1.2, node.z * scale - 1.2, 2.4, 2.4);
    }
    for (const e of this.world.entities) if (e.hp > 0 && (e.owner === 0 || this.world.isVisible(0, e.x, e.z))) {
      context.fillStyle = selected.has(e.id) ? '#fff1bb' : e.owner === 0 ? '#8bc9eb' : '#e79372';
      const r = e.category === 'building' ? BUILDING_DEFS[e.kind as BuildingKind].size * scale * .4 : 1.35;
      context.fillRect(e.x * scale - r, e.z * scale - r, r * 2, r * 2);
    }
    context.strokeStyle = '#f6ebbd99'; context.lineWidth = 1;
    context.beginPath(); let n = 0;
    // The full game canvas is the viewport; its offset and HUD dimensions are irrelevant.
    // Permit corners outside the map so the minimap canvas naturally clips the outline.
    for (const [x, y] of [[-1, 1], [1, 1], [1, -1], [-1, -1]]) {
      this.raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera);
      const p = this.raycaster.ray.intersectPlane(this.groundPlane, new THREE.Vector3());
      if (p) { if (n++ === 0) context.moveTo(p.x * scale, p.z * scale); else context.lineTo(p.x * scale, p.z * scale); }
    }
    context.closePath(); context.stroke();
    context.fillStyle = '#f7e4ab'; context.beginPath(); context.arc(this.focusPoint.x * scale, this.focusPoint.z * scale, 1.5, 0, Math.PI * 2); context.fill();
  }

  private removeModel(model: Model) {
    this.scene.remove(model.root);
    model.orderLine.geometry.dispose();
    model.root.traverse(object => {
      if (object.userData.disposable && object instanceof THREE.Line) {
        object.geometry.dispose();
        (object.material as THREE.Material).dispose();
      }
    });
  }

  dispose() {
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener('webglcontextlost', this.contextLost); this.canvas.removeEventListener('webglcontextrestored', this.contextRestored);
    for (const model of this.models.values()) this.removeModel(model);
    this.models.clear(); this.resources.clear(); this.knownResources.clear();
    this.tiles.dispose(); this.ghostMaterial.dispose(); this.selectionMat.dispose(); this.unitMat.dispose(); this.damageMat.dispose(); this.healthBack.dispose();
    for (const entry of Object.values(this.orderMaterials)) { entry.line.dispose(); entry.marker.dispose(); }
    this.healthMat.forEach(m => m.dispose()); (this.orderRing.material as THREE.Material).dispose(); this.renderer.dispose();
  }
}
