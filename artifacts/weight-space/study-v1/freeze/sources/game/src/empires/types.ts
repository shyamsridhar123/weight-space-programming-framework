export type Team = 0 | 1;
export type Resource = 'food' | 'wood' | 'gold' | 'stone';
export type Resources = Record<Resource, number>;
export type UnitKind = 'villager' | 'spearman' | 'archer' | 'knight' | 'ram';
export type BuildingKind = 'towncenter' | 'house' | 'farm' | 'barracks' | 'range' | 'stable' | 'tower' | 'forge' | 'castle';
export type EntityKind = UnitKind | BuildingKind;
export type Upgrade = 'weapons' | 'armor' | 'economy';
export type Point = { x: number; z: number };
export type Terrain = 'grass' | 'water' | 'bridge' | 'hill';
export interface Order {
  type: 'idle' | 'move' | 'gather' | 'build' | 'attack' | 'attackMove';
  x?: number; z?: number; targetId?: number; resource?: Resource; path?: Point[];
  returning?: boolean;
}
export interface Entity extends Point {
  id: number; owner: Team; kind: EntityKind; category: 'unit' | 'building';
  hp: number; maxHp: number; construction: number;
  queue: { kind: UnitKind; remaining: number; total: number }[];
  order: Order; cooldown: number; heading: number; carrying?: number; carryResource?: Resource;
}
export interface Player {
  age: number; resources: Resources; upgrades: string[]; advancing: number;
  gathered: Resources; kills: number;
}
export interface ProductionStats {
  /** Completed production only. Starting entities and queued jobs do not count. */
  trained: Partial<Record<UnitKind, number>>;
  constructed: Partial<Record<BuildingKind, number>>;
}
export interface WorldOptions {
  /** The default skirmish retains its original map, economy, fog, and starting orders. */
  scenario?: 'skirmish' | 'first-victory';
}
export interface ResourceNode extends Point {
  id: number; kind: Resource; amount: number; maxAmount: number; owner?: Team; buildingId?: number;
}
export type Command =
  | { type: 'move'; ids: number[]; x: number; z: number; attackMove?: boolean }
  | { type: 'gather'; ids: number[]; targetId: number }
  | { type: 'construct'; ids: number[]; targetId: number }
  | { type: 'attack'; ids: number[]; targetId: number }
  | { type: 'build'; kind: BuildingKind; x: number; z: number; workerIds?: number[] }
  | { type: 'train'; buildingId: number; kind: UnitKind }
  | { type: 'advance' }
  | { type: 'research'; upgrade: Upgrade }
  | { type: 'stop'; ids: number[] };
export type CommandResult = { ok: boolean; reason?: string };
export interface UnitDefinition {
  name: string; cost: Partial<Resources>; hp: number; time: number; speed: number;
  range: number; damage: number; attackRate: number; vision: number; requiredAge: number; trainAt: BuildingKind;
}
export interface BuildingDefinition {
  name: string; cost: Partial<Resources>; hp: number; time: number; size: number;
  population: number; vision: number; range: number; damage: number; requiredAge: number; trainable: UnitKind[];
}
export const RESOURCES: readonly Resource[] = ['food', 'wood', 'gold', 'stone'];
export const DT = 0.1;
export const AGE_NAMES = ['', 'Frontier Age', 'Feudal Age', 'Citadel Age', 'Imperial Age'];
export const AGE_COSTS: Record<number, Partial<Resources>> = {
  2: { food: 400, gold: 150 }, 3: { food: 700, gold: 350 }, 4: { food: 1000, gold: 650 },
};
export const AGE_TIMES: Record<number, number> = { 2: 35, 3: 50, 4: 65 };
export const UNIT_DEFS: Record<UnitKind, UnitDefinition> = {
  villager: { name: 'Villager', cost: { food: 50 }, hp: 45, time: 12, speed: 2.5, range: 0.9, damage: 3, attackRate: 1.5, vision: 7, requiredAge: 1, trainAt: 'towncenter' },
  spearman: { name: 'Spearman', cost: { food: 50, wood: 25 }, hp: 85, time: 15, speed: 2.5, range: 1.25, damage: 9, attackRate: 1.3, vision: 8, requiredAge: 1, trainAt: 'barracks' },
  archer: { name: 'Archer', cost: { wood: 35, gold: 35 }, hp: 55, time: 18, speed: 2.5, range: 5.5, damage: 7, attackRate: 1.4, vision: 9, requiredAge: 2, trainAt: 'range' },
  knight: { name: 'Knight', cost: { food: 65, gold: 70 }, hp: 155, time: 24, speed: 3.7, range: 1.1, damage: 16, attackRate: 1.4, vision: 9, requiredAge: 3, trainAt: 'stable' },
  ram: { name: 'Siege ram', cost: { wood: 140, gold: 90 }, hp: 250, time: 30, speed: 1.5, range: 1.35, damage: 7, attackRate: 2.2, vision: 6, requiredAge: 3, trainAt: 'forge' },
};
export const BUILDING_DEFS: Record<BuildingKind, BuildingDefinition> = {
  towncenter: { name: 'Town Center', cost: { wood: 250, stone: 100 }, hp: 1800, time: 60, size: 4, population: 10, vision: 12, range: 7.5, damage: 12, requiredAge: 1, trainable: ['villager'] },
  house: { name: 'House', cost: { wood: 35 }, hp: 320, time: 12, size: 2, population: 5, vision: 5, range: 0, damage: 0, requiredAge: 1, trainable: [] },
  farm: { name: 'Farm', cost: { wood: 65 }, hp: 220, time: 15, size: 3, population: 0, vision: 4, range: 0, damage: 0, requiredAge: 1, trainable: [] },
  barracks: { name: 'Barracks', cost: { wood: 130 }, hp: 1000, time: 28, size: 3, population: 0, vision: 7, range: 0, damage: 0, requiredAge: 1, trainable: ['spearman'] },
  range: { name: 'Archery Range', cost: { wood: 150 }, hp: 900, time: 30, size: 3, population: 0, vision: 7, range: 0, damage: 0, requiredAge: 2, trainable: ['archer'] },
  stable: { name: 'Stable', cost: { wood: 170 }, hp: 1100, time: 32, size: 3, population: 0, vision: 7, range: 0, damage: 0, requiredAge: 2, trainable: ['knight'] },
  tower: { name: 'Watchtower', cost: { wood: 100, stone: 100 }, hp: 850, time: 32, size: 2, population: 0, vision: 12, range: 8, damage: 18, requiredAge: 2, trainable: [] },
  forge: { name: 'Forge & Siegeworks', cost: { wood: 180, gold: 50 }, hp: 1100, time: 35, size: 3, population: 0, vision: 7, range: 0, damage: 0, requiredAge: 2, trainable: ['ram'] },
  castle: { name: 'Castle', cost: { wood: 200, stone: 450 }, hp: 2800, time: 70, size: 5, population: 10, vision: 15, range: 10, damage: 34, requiredAge: 3, trainable: ['spearman', 'archer', 'knight', 'ram'] },
};
export const UPGRADE_DEFS: Record<Upgrade, { name: string; cost: Partial<Resources>; requiredAge: number }> = {
  weapons: { name: 'Tempered weapons', cost: { food: 120, gold: 80 }, requiredAge: 2 },
  armor: { name: 'Forged armor', cost: { food: 100, gold: 100 }, requiredAge: 2 },
  economy: { name: 'Wheelbarrow', cost: { food: 150, wood: 100 }, requiredAge: 2 },
};
