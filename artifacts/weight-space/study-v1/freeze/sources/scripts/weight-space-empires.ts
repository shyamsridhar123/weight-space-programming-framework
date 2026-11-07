import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { arch, platform } from 'node:os';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { createMissionWorld, MISSION_RULES } from '../game/src/empires/mission.ts';
import { createDefaultModel, policyFeatures, type PolicyModel } from '../game/src/empires/policy.ts';
import { PLAY_POLICY_TICKS, rankPlayPolicy, shouldPlayPolicyStep } from '../game/src/empires/play-policy.ts';
import {
  runPlayExperiment, type PlayExperimentBranch,
  type PlayExperimentResult, type PlayExperimentSeconds,
} from '../game/src/empires/play-experiment.ts';
import type { WorldCheckpoint } from '../game/src/empires/simulation.ts';
import { DT, RESOURCES, type Command, type Resources, type Team } from '../game/src/empires/types.ts';
import { fingerprintWeightModel, validateWeightModel } from '../game/src/empires/weight-runtime.ts';

export type Split = 'development' | 'validation' | 'test';
export type SplitSelection = Split | 'all';
export interface RtsProtocol {
  development_seeds: number[];
  validation_seeds: number[];
  test_seeds: number[];
  horizon_seconds: PlayExperimentSeconds;
  fixture_seconds: number;
  fixture_interval_seconds: number;
}
export interface WeightSpaceProgram { id: string; model: PolicyModel; source: string; }
export interface PolicyFixture {
  id: string;
  split: Exclude<Split, 'test'>;
  seed: number;
  tick: number;
  time: number;
  team: Team;
  state_id: string;
  candidates: { id: string; label: string; input: number[]; score: number; command: Command | null }[];
}
export interface FixtureSnapshot { state_id: string; checkpoint: WorldCheckpoint; }

/** Scalar candidate scorer; temperature is metadata for RTS argmax, outside this vector ABI. */
export const RTS_PARAMETER_LAYOUT = Object.freeze({
  order: ['w1', 'b1', 'w2', 'b2'],
  flattening: 'row-major', parameters: 1293,
  tensors: {
    w1: { shape: [38, 32], offset: 0, length: 1216 },
    b1: { shape: [38], offset: 1216, length: 38 },
    w2: { shape: [38], offset: 1254, length: 38 },
    b2: { shape: [], offset: 1292, length: 1 },
  },
  temperature: 'separate; unused by rankPlayPolicy',
});

const SPLITS: readonly Split[] = ['development', 'validation', 'test'];
const CONTROL_IDS = new Set(['control_noop', 'control_b2_shift']);
const PROJECT_ROOT = fileURLToPath(new URL('../', import.meta.url));
const json = (value: unknown) => `${JSON.stringify(value, null, 2)}\n`;
export const sha256 = (value: string | Uint8Array) => createHash('sha256').update(value).digest('hex');
const stateId = (checkpoint: WorldCheckpoint) => sha256(JSON.stringify(checkpoint));

function object(value: unknown, label: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)
    || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new Error(`${label} must be a plain object`);
  if (Reflect.ownKeys(value).some(key => typeof key !== 'string'
    || !Object.hasOwn(Object.getOwnPropertyDescriptor(value, key)!, 'value'))) throw new Error(`${label} must contain data fields`);
  return value as Record<string, unknown>;
}

/** Read the RTS section while retaining the complete protocol file hash in every artifact. */
export function validateRtsProtocol(value: unknown): RtsProtocol {
  const raw = object(object(value, 'Protocol').rts, 'Protocol.rts');
  const seen = new Set<number>();
  const seeds = (name: string): number[] => {
    const values = raw[name];
    if (!Array.isArray(values) || !values.length) throw new Error(`${name} must be a nonempty seed array`);
    const result: number[] = [];
    for (const seed of values) {
      if (!Number.isSafeInteger(seed) || seed < 1 || seed > 0xffffffff) throw new Error(`${name} must contain positive uint32 seeds`);
      if (seen.has(seed)) throw new Error(`Seed ${seed} is duplicated within or across protocol splits`);
      seen.add(seed); result.push(seed);
    }
    return result;
  };
  const development_seeds = seeds('development_seeds'), validation_seeds = seeds('validation_seeds'), test_seeds = seeds('test_seeds');
  if (typeof raw.horizon_seconds !== 'number' || ![30, 60, 120, 420].includes(raw.horizon_seconds))
    throw new Error('horizon_seconds must be 30, 60, 120, or 420');
  for (const key of ['fixture_seconds', 'fixture_interval_seconds']) {
    const seconds = raw[key];
    if (typeof seconds !== 'number' || !Number.isSafeInteger(seconds) || seconds <= 0
      || seconds > raw.horizon_seconds || seconds % (DT * PLAY_POLICY_TICKS) !== 0)
      throw new Error(`${key} must be a positive policy-cadence multiple within the horizon`);
  }
  const fixture_seconds = raw.fixture_seconds as number, fixture_interval_seconds = raw.fixture_interval_seconds as number;
  if (fixture_seconds % fixture_interval_seconds !== 0) throw new Error('fixture_seconds must be a multiple of fixture_interval_seconds');
  return { development_seeds, validation_seeds, test_seeds, horizon_seconds: raw.horizon_seconds as PlayExperimentSeconds,
    fixture_seconds, fixture_interval_seconds };
}

export function validatePrograms(value: unknown): WeightSpaceProgram[] {
  const envelope = object(value, 'Programs');
  if (Object.keys(envelope).length !== 1 || !Array.isArray(envelope.candidates))
    throw new Error('Programs must have exactly one candidates array');
  const ids = new Set(CONTROL_IDS);
  return Array.from(envelope.candidates, (entry, index) => {
    const candidate = object(entry, `Candidate ${index}`);
    if (Object.keys(candidate).length !== 3 || !['id', 'model', 'source'].every(key => Object.hasOwn(candidate, key)))
      throw new Error(`Candidate ${index} requires exactly id, model, and source`);
    if (typeof candidate.id !== 'string' || !/^[A-Za-z][A-Za-z0-9._-]{0,79}$/.test(candidate.id) || ids.has(candidate.id))
      throw new Error(`Candidate ${index} needs a unique portable id; control ids are reserved`);
    if (typeof candidate.source !== 'string' || !candidate.source.trim() || candidate.source.length > 2048
      || /[\u0000-\u001f\u007f]/.test(candidate.source)) throw new Error(`Candidate ${index} needs a source description`);
    ids.add(candidate.id);
    return { id: candidate.id, source: candidate.source, model: validateWeightModel(candidate.model) };
  });
}

export function createControlPrograms(value: PolicyModel = createDefaultModel()): WeightSpaceProgram[] {
  const model = validateWeightModel(value), shifted = validateWeightModel(value);
  shifted.b2 += shifted.b2 <= 63 ? 1 : -1;
  return [
    { id: 'control_noop', model, source: 'Exact authored commander-v1 baseline; no parameter edits.' },
    { id: 'control_b2_shift', model: validateWeightModel(shifted), source: 'Only shared output bias shifted by one; behavioral negative control.' },
  ];
}

/** Capture actual legal choices at a decision boundary, before this team issues its order. */
export function collectPolicyFixtures(protocol: RtsProtocol, value: PolicyModel = createDefaultModel()) {
  const model = validateWeightModel(value), fixtures: PolicyFixture[] = [], snapshots: FixtureSnapshot[] = [];
  const intervalTicks = Math.round(protocol.fixture_interval_seconds / DT), lastTick = Math.round(protocol.fixture_seconds / DT);
  for (const split of ['development', 'validation'] as const) for (const seed of protocol[`${split}_seeds`]) {
    const world = createMissionWorld(seed);
    while (world.status === 'playing' && world.tick < lastTick) {
      world.step(DT);
      if (!shouldPlayPolicyStep(world)) continue;
      for (const team of [0, 1] as const) {
        const ranked = rankPlayPolicy(world, team, model);
        if (world.tick % intervalTicks === 0) {
          const checkpoint = world.checkpoint(), state_id = stateId(checkpoint);
          fixtures.push({
            id: `${split}/seed-${seed}/tick-${world.tick}/team-${team}`, split, seed,
            tick: world.tick, time: world.time, team, state_id,
            candidates: ranked.map(({ candidate, score }) => ({ id: candidate.id, label: candidate.label,
              input: policyFeatures(world, team, candidate), score, command: structuredClone(candidate.command) })),
          });
          snapshots.push({ state_id, checkpoint });
        }
        const command = ranked[0]?.candidate.command;
        if (command && !world.command(team, command).ok) throw new Error(`A grounded fixture rollout rejected ${ranked[0].candidate.id}`);
      }
    }
  }
  return { fixtures, snapshots };
}

/** Strict loading is essential: the legacy experiment's validateModel checks fewer invariants. */
export function evaluatePolicyProgram(seed: number, program: WeightSpaceProgram, seconds: PlayExperimentSeconds, team: Team = 0) {
  const baselineModel = validateWeightModel(createDefaultModel()), candidateModel = validateWeightModel(program.model);
  const checkpoint = createMissionWorld(seed).checkpoint();
  const result = runPlayExperiment({ checkpoint, team, baselineModel, candidateModel,
    otherModel: baselineModel, otherControl: 'policy', seconds });
  return { checkpoint, result };
}

function resourcesDifference(final: Resources, initial: Resources): Resources {
  return Object.fromEntries(RESOURCES.map(kind => [kind, final[kind] - initial[kind]])) as Resources;
}

export function summarizeBranch(checkpoint: WorldCheckpoint, result: PlayExperimentResult, branch: PlayExperimentBranch) {
  const team = result.team, otherTeam = (1 - team) as Team;
  const maxHp = (owner: Team) => checkpoint.entities.filter(entity => entity.owner === owner && entity.kind === 'towncenter' && entity.hp > 0)
    .reduce((total, entity) => total + entity.maxHp, 0);
  const town_max_hp = maxHp(team), opponent_town_max_hp = maxHp(otherTeam);
  if (town_max_hp <= 0 || opponent_town_max_hp <= 0) throw new Error('Utility requires an initial Town Center on both sides');
  const own_town_hp_lost = result.initialMetrics.townHealth - branch.metrics.townHealth;
  const opponent_town_hp_lost = result.initialOtherMetrics.townHealth - branch.otherMetrics.townHealth;
  const recruits_completed = branch.metrics.recruitsCompleted - result.initialMetrics.recruitsCompleted;
  const winnerTeam = branch.metrics.status === 'won' ? 0 : branch.metrics.status === 'lost' ? 1 : null;
  const victory = winnerTeam === team, defeat = winnerTeam === otherTeam;
  return {
    outcome: victory ? 'win' : defeat ? 'loss' : branch.metrics.status === 'draw' ? 'draw' : 'unfinished',
    victory, defeat, status: branch.metrics.status, elapsed_seconds: branch.elapsedSeconds,
    utility: opponent_town_hp_lost / opponent_town_max_hp - own_town_hp_lost / town_max_hp + 0.02 * recruits_completed,
    recruits_completed, trained_units: structuredClone(branch.metrics.trained),
    initial_town_hp: result.initialMetrics.townHealth, town_max_hp, final_town_hp: branch.metrics.townHealth,
    initial_opponent_town_hp: result.initialOtherMetrics.townHealth, opponent_town_max_hp, final_opponent_town_hp: branch.otherMetrics.townHealth,
    own_town_hp_lost, opponent_town_hp_lost,
    gathered: resourcesDifference(branch.metrics.gathered, result.initialMetrics.gathered),
    stockpile: { ...branch.metrics.stockpile },
    kills: branch.metrics.kills - result.initialMetrics.kills,
    decisions: branch.decisions, executed_orders: branch.executedOrders, wait_decisions: branch.waitDecisions,
    metrics: branch.metrics, other_metrics: branch.otherMetrics,
  };
}

export function summarizePair(checkpoint: WorldCheckpoint, result: PlayExperimentResult) {
  const baseline = summarizeBranch(checkpoint, result, result.baseline), candidate = summarizeBranch(checkpoint, result, result.candidate);
  return {
    baseline, candidate,
    deltas: {
      utility: candidate.utility - baseline.utility,
      recruits_completed: candidate.recruits_completed - baseline.recruits_completed,
      own_town_hp_lost: candidate.own_town_hp_lost - baseline.own_town_hp_lost,
      opponent_town_hp_lost: candidate.opponent_town_hp_lost - baseline.opponent_town_hp_lost,
      kills: candidate.kills - baseline.kills,
      victory: Number(candidate.victory) - Number(baseline.victory),
      defeat: Number(candidate.defeat) - Number(baseline.defeat),
      gathered: resourcesDifference(candidate.gathered, baseline.gathered),
      stockpile: resourcesDifference(candidate.stockpile, baseline.stockpile),
    },
    behavior_equal: result.firstChangedChoice === null && stateId(result.baseline.finalCheckpoint) === stateId(result.candidate.finalCheckpoint),
    first_changed_choice: result.firstChangedChoice,
    initial_checkpoint_sha256: stateId(checkpoint),
    baseline_checkpoint_sha256: stateId(result.baseline.finalCheckpoint),
    candidate_checkpoint_sha256: stateId(result.candidate.finalCheckpoint),
  };
}

async function save(file: string, content: string): Promise<string> {
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, content, { flag: 'wx' });
  return sha256(content);
}

async function sourceFiles(directory: string, extension: string): Promise<string[]> {
  const entries = await readdir(resolve(PROJECT_ROOT, directory), { withFileTypes: true });
  const files = await Promise.all(entries.map(async entry => {
    const file = `${directory}/${entry.name}`;
    if (entry.isDirectory()) return sourceFiles(file, extension);
    return entry.isFile() && entry.name.endsWith(extension) ? [file] : [];
  }));
  return files.flat();
}

async function provenance(protocolPath: string) {
  const bytes = await readFile(resolve(protocolPath));
  const document: unknown = JSON.parse(bytes.toString('utf8'));
  const files = ['scripts/weight-space-empires.ts', 'package-lock.json',
    ...await sourceFiles('game/src/empires', '.ts'),
    ...await sourceFiles('game/src/empires/weights', '.json')].sort();
  const source_sha256 = Object.fromEntries(await Promise.all(files.map(async file =>
    [file, sha256(await readFile(resolve(PROJECT_ROOT, file)))])));
  return { protocol: validateRtsProtocol(document), document, bytes, receipt: {
    protocol_sha256: sha256(bytes), source_sha256,
    environment: { node: process.version, platform: platform(), arch: arch() },
  } };
}

export async function exportFixtures(protocolPath: string, outputDirectory: string) {
  const context = await provenance(protocolPath), out = resolve(outputDirectory);
  const model = validateWeightModel(createDefaultModel()), modelContent = json(model), model_sha256 = sha256(modelContent);
  const { fixtures, snapshots } = collectPolicyFixtures(context.protocol, model);
  const artifacts: Record<string, string> = {};
  artifacts['protocol.json'] = await save(resolve(out, 'protocol.json'), context.bytes.toString('utf8'));
  artifacts['model.json'] = await save(resolve(out, 'model.json'), modelContent);
  artifacts['fixtures.json'] = await save(resolve(out, 'fixtures.json'), json({
    schema_version: 1, protocol_sha256: context.receipt.protocol_sha256, model_sha256,
    model_fingerprint: fingerprintWeightModel(model), layout: RTS_PARAMETER_LAYOUT,
    semantics: 'First Conquest legal scalar candidate scores, deterministic argmax with lexical ties; blue acts before red.',
    feature_names: model.featureNames, fixtures,
  }));
  artifacts['snapshots.jsonl'] = await save(resolve(out, 'snapshots.jsonl'), snapshots.map(snapshot => JSON.stringify(snapshot)).join('\n') + '\n');
  const receipt = { schema_version: 1, operation: 'export', created_at: new Date().toISOString(), ...context.receipt,
    artifacts, fixture_count: fixtures.length, exported_splits: ['development', 'validation'], test_fixtures_exported: false };
  await save(resolve(out, 'receipt.json'), json(receipt));
  return receipt;
}

type RunSummary = ReturnType<typeof summarizePair> & {
  candidate_id: string; split: Split; seed: number; team: Team; control: boolean;
  source: string; model_sha256: string; model_fingerprint: string;
  raw_result_file: string; raw_result_sha256: string; wall_seconds: number;
};

function aggregate(runs: RunSummary[]) {
  const rows: object[] = [];
  for (const split of SPLITS) for (const candidateId of new Set(runs.filter(run => run.split === split).map(run => run.candidate_id))) {
    const selected = runs.filter(run => run.split === split && run.candidate_id === candidateId);
    const mean = (get: (run: RunSummary) => number) => selected.reduce((total, run) => total + get(run), 0) / selected.length;
    rows.push({ split, candidate_id: candidateId, control: selected[0].control, completed_seeds: selected.length,
      mean_baseline_utility: mean(run => run.baseline.utility), mean_candidate_utility: mean(run => run.candidate.utility),
      mean_utility_delta: mean(run => run.deltas.utility), mean_recruits_delta: mean(run => run.deltas.recruits_completed),
      baseline_loss_rate: mean(run => Number(run.baseline.defeat)), candidate_loss_rate: mean(run => Number(run.candidate.defeat)),
      baseline_win_rate: mean(run => Number(run.baseline.victory)), candidate_win_rate: mean(run => Number(run.candidate.victory)),
      changed_futures: selected.filter(run => !run.behavior_equal).length,
    });
  }
  return rows;
}

export async function evaluatePrograms(options: {
  programsPath: string; protocolPath: string; outputDirectory: string;
  split?: SplitSelection; fullCheckpoints?: boolean;
  onProgress?: (message: string) => void;
}) {
  const context = await provenance(options.protocolPath), out = resolve(options.outputDirectory);
  const bytes = await readFile(resolve(options.programsPath)), supplied = validatePrograms(JSON.parse(bytes.toString('utf8')));
  const split = options.split ?? 'development';
  if (split !== 'all' && !SPLITS.includes(split)) throw new Error('split must be development, validation, test, or all');
  const selectedSplits = split === 'all' ? SPLITS : [split];
  const programs = [...createControlPrograms(), ...supplied], runs: RunSummary[] = [];
  const failures: { candidate_id: string; split: Split; seed: number; reason: string }[] = [];
  const planned_runs = selectedSplits.reduce((count, selected) => count + context.protocol[`${selected}_seeds`].length * programs.length, 0);
  const manifest: Record<string, string> = {};
  manifest['protocol.json'] = await save(resolve(out, 'protocol.json'), context.bytes.toString('utf8'));
  manifest['programs.json'] = await save(resolve(out, 'programs.json'), bytes.toString('utf8'));
  const programMetadata = programs.map(program => ({ id: program.id, source: program.source, control: CONTROL_IDS.has(program.id),
    model_sha256: sha256(json(program.model)), model_fingerprint: fingerprintWeightModel(program.model) }));
  for (const [index, program] of programs.entries()) {
    const path = `models/${program.id}.json`;
    manifest[path] = await save(resolve(out, path), json(program.model));
    if (manifest[path] !== programMetadata[index].model_sha256) throw new Error('Saved model identity mismatch');
  }
  const started = performance.now();
  for (const selected of selectedSplits) for (const seed of context.protocol[`${selected}_seeds`]) for (const [index, program] of programs.entries()) {
    const runStarted = performance.now();
    let captured: ReturnType<typeof evaluatePolicyProgram>;
    try {
      captured = evaluatePolicyProgram(seed, program, context.protocol.horizon_seconds);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      failures.push({ candidate_id: program.id, split: selected, seed, reason });
      options.onProgress?.(`${selected} seed=${seed} candidate=${program.id} failed: ${reason}`);
      continue;
    }
    const { checkpoint, result } = captured;
    const summary = summarizePair(checkpoint, result);
    const prefix = `runs/${selected}/seed-${seed}-${program.id}`;
    const branch = async (name: 'baseline' | 'candidate') => {
      const { finalCheckpoint, ...remaining } = result[name];
      let checkpoint_file: string | null = null;
      if (options.fullCheckpoints) {
        checkpoint_file = `${prefix}-${name}-checkpoint.json`;
        manifest[checkpoint_file] = await save(resolve(out, checkpoint_file), json(finalCheckpoint));
      }
      return { ...remaining, final_checkpoint_sha256: stateId(finalCheckpoint), checkpoint_file };
    };
    const raw_result_file = `${prefix}.json`;
    const raw = { schema_version: 1, protocol_sha256: context.receipt.protocol_sha256, programs_sha256: sha256(bytes),
      split: selected, seed, program: programMetadata[index], ...result,
      baseline: await branch('baseline'), candidate: await branch('candidate') };
    const raw_result_sha256 = await save(resolve(out, raw_result_file), json(raw));
    manifest[raw_result_file] = raw_result_sha256;
    const wall_seconds = (performance.now() - runStarted) / 1000;
    runs.push({ candidate_id: program.id, split: selected, seed, team: result.team, control: CONTROL_IDS.has(program.id),
      source: program.source, model_sha256: programMetadata[index].model_sha256,
      model_fingerprint: programMetadata[index].model_fingerprint,
      ...summary, raw_result_file, raw_result_sha256, wall_seconds });
    options.onProgress?.(`${selected} seed=${seed} candidate=${program.id} utility_delta=${summary.deltas.utility.toFixed(6)} changed=${!summary.behavior_equal} wall=${wall_seconds.toFixed(2)}s`);
  }
  const controls = runs.filter(run => run.control);
  const plannedControls = selectedSplits.reduce((count, selected) => count + context.protocol[`${selected}_seeds`].length * CONTROL_IDS.size, 0);
  const controls_pass = controls.length === plannedControls && controls.every(run => run.behavior_equal && run.deltas.utility === 0);
  const report = {
    schema_version: 1, operation: 'evaluate', created_at: new Date().toISOString(), ...context.receipt,
    programs_sha256: sha256(bytes), split, planned_runs, completed_runs: runs.length, failures, controls_pass,
    methodology: {
      scenario: 'first-victory', candidate_team: 0, opponent: 'authored commander-v1 under the mission rival mask',
      paired_same_checkpoint: true, horizon_seconds: context.protocol.horizon_seconds,
      dt_seconds: DT, policy_interval_seconds: PLAY_POLICY_TICKS * DT, policy_order: [0, 1],
      rival_recruitment_begins_seconds: MISSION_RULES.rivalRecruitStartSeconds, rival_raid_eligibility_seconds: MISSION_RULES.rivalGraceSeconds,
      utility: 'opponent Town Center HP lost / initial opponent maximum HP - own Town Center HP lost / initial own maximum HP + 0.02 * completed recruited soldiers',
      damage_measurement: 'Net Town Center HP loss over the window; total combat damage is not instrumented.',
      unfinished_outcome: 'A playing match at the horizon remains unfinished, with no invented winner.',
      selection: 'No selection occurs in this evaluator. The programs file is captured before simulations; use development/validation to freeze a test candidate.',
      inference_unit: 'world seed; one blue-side paired intervention per candidate and seed',
    },
    programs: programMetadata, summary: aggregate(runs), runs, wall_seconds: (performance.now() - started) / 1000,
  };
  manifest['results.json'] = await save(resolve(out, 'results.json'), json(report));
  const receipt = { schema_version: 1, operation: 'evaluate', created_at: new Date().toISOString(), ...context.receipt,
    programs_sha256: sha256(bytes), artifacts: manifest, planned_runs, completed_runs: runs.length,
    controls_pass, failures: failures.length, complete: failures.length === 0 && runs.length === planned_runs };
  await save(resolve(out, 'receipt.json'), json(receipt));
  return report;
}

export async function main(args = process.argv.slice(2)) {
  const { positionals, values } = parseArgs({ args, allowPositionals: true, strict: true, options: {
    out: { type: 'string' }, protocol: { type: 'string' }, programs: { type: 'string' },
    split: { type: 'string', default: 'development' }, 'full-checkpoints': { type: 'boolean', default: false },
  } });
  if (positionals.length !== 1 || !['export', 'evaluate'].includes(positionals[0]) || !values.out || !values.protocol)
    throw new Error('Usage: weight-space-empires.ts export|evaluate --out <dir> --protocol <json> [--programs <json> --split development|validation|test|all --full-checkpoints]');
  if (positionals[0] === 'export') {
    const receipt = await exportFixtures(values.protocol, values.out);
    process.stdout.write(`Exported ${receipt.fixture_count} grounded fixtures to ${resolve(values.out)}\n`);
  } else {
    if (!values.programs) throw new Error('evaluate requires --programs <json>');
    const report = await evaluatePrograms({ programsPath: values.programs, protocolPath: values.protocol,
      outputDirectory: values.out, split: values.split as SplitSelection, fullCheckpoints: values['full-checkpoints'],
      onProgress: message => process.stdout.write(`${message}\n`) });
    process.stdout.write(`Saved ${report.completed_runs}/${report.planned_runs} runs; controls_pass=${report.controls_pass}\n`);
    if (report.failures.length || !report.controls_pass) process.exitCode = 1;
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main().catch(error => { process.stderr.write(`${error instanceof Error ? error.stack : error}\n`); process.exitCode = 1; });
}
