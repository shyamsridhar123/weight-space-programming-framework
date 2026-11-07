import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { createDefaultModel, forwardPolicy, policyFeatures } from '../game/src/empires/policy.ts';
import { createMissionWorld, MISSION_RULES } from '../game/src/empires/mission.ts';
import { getPlayCandidates, rankPlayPolicy } from '../game/src/empires/play-policy.ts';
import { runPlayExperiment } from '../game/src/empires/play-experiment.ts';
import { World } from '../game/src/empires/simulation.ts';
import { DT } from '../game/src/empires/types.ts';
import {
  RTS_PARAMETER_LAYOUT, collectPolicyFixtures, createControlPrograms, evaluatePolicyProgram,
  evaluatePrograms, exportFixtures, sha256, summarizePair, validatePrograms, validateRtsProtocol,
} from '../scripts/weight-space-empires.ts';

// Dedicated adapter-test seeds: none comes from the frozen scientific protocol.
const protocolDocument = () => ({ rts: {
  development_seeds: [910101], validation_seeds: [910201], test_seeds: [910301],
  horizon_seconds: 30, fixture_seconds: 20, fixture_interval_seconds: 20,
} });

test('grounded export replays actual legal features and exact JavaScript scores, without test fixtures', () => {
  const protocol = validateRtsProtocol(protocolDocument()), model = createDefaultModel();
  const { fixtures, snapshots } = collectPolicyFixtures(protocol, model);
  assert.equal(fixtures.length, 4);
  assert.deepEqual(new Set(fixtures.map(fixture => fixture.split)), new Set(['development', 'validation']));
  assert.ok(fixtures.every(fixture => fixture.seed !== 910301));
  assert.equal(RTS_PARAMETER_LAYOUT.parameters, model.w1.flat().length + model.b1.length + model.w2.length + 1);
  for (let index = 0; index < fixtures.length; index++) {
    const fixture = fixtures[index], snapshot = snapshots[index];
    assert.equal(fixture.state_id, sha256(JSON.stringify(snapshot.checkpoint)));
    assert.equal(fixture.state_id, snapshot.state_id);
    const world = World.fromCheckpoint(snapshot.checkpoint), before = world.checkpoint();
    assert.equal(fixture.tick, 200);
    assert.equal(fixture.tick, world.tick);
    assert.equal(fixture.time, world.time);
    const candidates = getPlayCandidates(world, fixture.team);
    assert.deepEqual(fixture.candidates.map(candidate => candidate.id).sort(), candidates.map(candidate => candidate.id).sort());
    assert.deepEqual(fixture.candidates.map(candidate => ({ id: candidate.id, score: candidate.score })),
      rankPlayPolicy(world, fixture.team, model).map(({ candidate, score }) => ({ id: candidate.id, score })));
    for (const candidate of fixture.candidates) {
      const grounded = candidates.find(value => value.id === candidate.id)!;
      assert.deepEqual(candidate.command, grounded.command);
      assert.deepEqual(candidate.input, policyFeatures(world, fixture.team, grounded));
      assert.equal(candidate.input.length, 32);
      assert.ok(candidate.input.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
      assert.equal(candidate.score, forwardPolicy(candidate.input, model));
      if (candidate.command) assert.equal(world.canCommand(fixture.team, candidate.command).ok, true);
    }
    assert.deepEqual(world.checkpoint(), before, 'Reading exported candidate features must not mutate the state.');
  }
});

test('program loading rejects wrong tensor schemas, reordered features, nonfinite values, and stale schema envelopes', () => {
  const program = () => ({ candidates: [{ id: 'recruit', source: 'unit test', model: createDefaultModel() }] });
  const original = program(), loaded = validatePrograms(original);
  original.candidates[0].model.w2[0] += 1;
  assert.notEqual(loaded[0].model.w2[0], original.candidates[0].model.w2[0], 'Loaded models must be detached.');
  for (const corrupt of [
    (model: ReturnType<typeof createDefaultModel>) => { model.w1[0].pop(); },
    (model: ReturnType<typeof createDefaultModel>) => { model.b1[0] = Infinity; },
    (model: ReturnType<typeof createDefaultModel>) => { model.w2[0] = NaN; },
    (model: ReturnType<typeof createDefaultModel>) => { model.w2[0] = 65; },
    (model: ReturnType<typeof createDefaultModel>) => { model.temperature = 0; },
    (model: ReturnType<typeof createDefaultModel>) => { [model.featureNames[0], model.featureNames[1]] = [model.featureNames[1], model.featureNames[0]]; },
    (model: ReturnType<typeof createDefaultModel>) => { Object.assign(model, { unknown: true }); },
  ]) {
    const value = program(); corrupt(value.candidates[0].model);
    assert.throws(() => validatePrograms(value));
  }
  const overflowJson = JSON.stringify(program()).replace('"b2":0', '"b2":1e999');
  assert.ok(overflowJson.includes('1e999'));
  assert.throws(() => validatePrograms(JSON.parse(overflowJson)), /finite/);
  assert.throws(() => validatePrograms({ candidates: [], extra: true }), /exactly one/);
  assert.throws(() => validatePrograms({ candidates: [program().candidates[0], program().candidates[0]] }), /unique/);
  const reserved = program(); reserved.candidates[0].id = 'control_noop';
  assert.throws(() => validatePrograms(reserved), /reserved/);
});

test('protocol loading enforces seed separation and decision-aligned fixture times', () => {
  const overlap = protocolDocument(); overlap.rts.test_seeds = overlap.rts.development_seeds;
  assert.throws(() => validateRtsProtocol(overlap), /duplicated/);
  const zero = protocolDocument(); zero.rts.development_seeds = [0];
  assert.throws(() => validateRtsProtocol(zero), /uint32/);
  const duration = protocolDocument(); duration.rts.horizon_seconds = 330;
  assert.throws(() => validateRtsProtocol(duration), /420/);
  const interval = protocolDocument(); interval.rts.fixture_interval_seconds = 3;
  assert.throws(() => validateRtsProtocol(interval), /cadence/);
});

test('no-op and common-bias controls preserve complete short futures while raw scores retain the bias shift', () => {
  const controls = createControlPrograms();
  for (const team of [0, 1] as const) {
    const noop = evaluatePolicyProgram(910401, controls[0], 30, team);
    assert.deepEqual(noop.result.baseline, noop.result.candidate);
    const shifted = evaluatePolicyProgram(910401, controls[1], 30, team);
    const summary = summarizePair(shifted.checkpoint, shifted.result);
    assert.equal(summary.behavior_equal, true);
    assert.equal(summary.first_changed_choice, null);
    assert.equal(summary.deltas.utility, 0);
    assert.deepEqual(summary.baseline.metrics, summary.candidate.metrics);
    const ownBaseline = shifted.result.baseline.orders.filter(order => order.team === team);
    const ownCandidate = shifted.result.candidate.orders.filter(order => order.team === team);
    assert.ok(ownBaseline.length > 0);
    assert.equal(ownBaseline.length, ownCandidate.length);
    ownBaseline.forEach((order, index) => {
      assert.deepEqual(order.command, ownCandidate[index].command);
      assert.ok(Math.abs(ownCandidate[index].score - order.score - 1) < 1e-10);
    });
  }
});

test('a real parameter intervention produces legal timed orders that replay the exact advertised checkpoint', () => {
  const model = createDefaultModel();
  model.w2.fill(0);
  model.w2[model.hiddenNames.indexOf('continuation_bias')] = 1;
  const { checkpoint, result } = evaluatePolicyProgram(910501, { id: 'wait', source: 'transparent wait fixture', model }, 30);
  assert.ok(result.firstChangedChoice);
  assert.equal(summarizePair(checkpoint, result).behavior_equal, false);
  for (const branch of [result.baseline, result.candidate]) {
    const world = World.fromCheckpoint(checkpoint);
    let index = 0;
    while (world.tick < branch.finalCheckpoint.tick && world.status === 'playing') {
      world.step(DT);
      while (branch.orders[index]?.tick === world.tick) {
        const order = branch.orders[index++];
        assert.equal(order.time, world.time);
        assert.equal(order.tick % 20, 0);
        assert.equal(world.canCommand(order.team, order.command).ok, true);
        assert.equal(world.command(order.team, order.command).ok, true);
      }
    }
    assert.equal(index, branch.orders.length);
    assert.deepEqual(world.checkpoint(), branch.finalCheckpoint);
  }
});

test('the 420-second horizon honors natural endings and reaches delayed rival phases in controlled no-op futures', () => {
  const { checkpoint, result } = evaluatePolicyProgram(910601, createControlPrograms()[0], 420);
  assert.equal(result.requestedSeconds, 420);
  assert.deepEqual(result.baseline, result.candidate);
  assert.ok(result.baseline.finalCheckpoint.tick <= 4200);
  assert.ok(result.baseline.finalCheckpoint.tick === 4200 || result.baseline.metrics.status !== 'playing');
  const summary = summarizePair(checkpoint, result);
  assert.equal(summary.baseline.utility,
    summary.baseline.opponent_town_hp_lost / summary.baseline.opponent_town_max_hp
    - summary.baseline.own_town_hp_lost / summary.baseline.town_max_hp + 0.02 * summary.baseline.recruits_completed);
  assert.equal(summary.deltas.utility, 0);

  // The authored commander may win before the rival's delay expires. These
  // transparent test policies keep blue idle and let red recruit without assaulting.
  const wait = createDefaultModel();
  wait.w2.fill(0);
  wait.w2[wait.hiddenNames.indexOf('continuation_bias')] = 1;
  const recruit = structuredClone(wait);
  recruit.w2[recruit.hiddenNames.indexOf('army_train_bias')] = 2;
  const full = runPlayExperiment({ checkpoint: createMissionWorld(910601).checkpoint(), team: 0,
    baselineModel: wait, candidateModel: wait, otherModel: recruit, otherControl: 'policy', seconds: 420 });
  assert.deepEqual(full.baseline, full.candidate);
  assert.equal(full.baseline.finalCheckpoint.tick, 4200);
  assert.ok(full.baseline.elapsedSeconds > 419.999 && full.baseline.elapsedSeconds < 420.001);
  const redRecruits = full.baseline.orders.filter(order => order.team === 1 && order.command.type === 'train');
  assert.ok(redRecruits.length > 0, 'The integration test must actually exercise the delayed rival recruitment.');
  assert.ok(redRecruits.every(order => order.time >= MISSION_RULES.rivalRecruitStartSeconds));
  assert.ok(getPlayCandidates(World.fromCheckpoint(full.baseline.finalCheckpoint), 1)
    .some(candidate => candidate.id.startsWith('assault')), 'The rival must reach actual raid eligibility.');
});

test('export and explicit validation evaluation save replay material, raw branches, and verifiable receipts', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'rlcd-weight-space-adapter-'));
  try {
    const protocolPath = join(directory, 'protocol.json'), programsPath = join(directory, 'programs.json');
    await writeFile(protocolPath, JSON.stringify(protocolDocument()));
    await writeFile(programsPath, JSON.stringify({ candidates: [] }));
    const fixtureDirectory = join(directory, 'fixtures');
    const receipt = await exportFixtures(protocolPath, fixtureDirectory);
    assert.equal(receipt.test_fixtures_exported, false);
    assert.equal(receipt.fixture_count, 4);
    const projectRoot = fileURLToPath(new URL('../', import.meta.url));
    const empiresFiles = (await readdir(join(projectRoot, 'game/src/empires'), { recursive: true }))
      .map(file => file.split(sep).join('/'));
    const expectedSources = ['package-lock.json', 'scripts/weight-space-empires.ts', ...empiresFiles
      .filter(file => file.endsWith('.ts') || (file.startsWith('weights/') && file.endsWith('.json')))
      .map(file => `game/src/empires/${file}`)].sort();
    assert.deepEqual(Object.keys(receipt.source_sha256), expectedSources,
      'Source receipts must cover every simulator TypeScript file, weight JSON file, adapter, and lockfile.');
    assert.ok(Object.hasOwn(receipt.source_sha256, 'game/src/empires/laya.ts'));
    for (const [file, hash] of Object.entries(receipt.source_sha256)) {
      assert.ok(!file.includes('\\'));
      assert.equal(sha256(await readFile(join(projectRoot, file))), hash);
    }
    for (const [file, hash] of Object.entries(receipt.artifacts)) assert.equal(sha256(await readFile(join(fixtureDirectory, file))), hash);
    const fixtures = JSON.parse(await readFile(join(fixtureDirectory, 'fixtures.json'), 'utf8'));
    assert.equal(fixtures.model_sha256, sha256(await readFile(join(fixtureDirectory, 'model.json'))));
    const outputDirectory = join(directory, 'validation');
    const report = await evaluatePrograms({ programsPath, protocolPath, outputDirectory, split: 'validation', fullCheckpoints: true });
    assert.equal(report.planned_runs, 2);
    assert.equal(report.completed_runs, 2);
    assert.equal(report.controls_pass, true);
    assert.deepEqual(report.source_sha256, receipt.source_sha256);
    assert.deepEqual(report.failures, []);
    assert.ok(report.runs.every(run => run.split === 'validation' && run.seed === 910201));
    assert.equal(report.programs_sha256, sha256(await readFile(programsPath)));
    const savedReceipt = JSON.parse(await readFile(join(outputDirectory, 'receipt.json'), 'utf8'));
    assert.equal(savedReceipt.complete, true);
    assert.deepEqual(savedReceipt.source_sha256, receipt.source_sha256);
    for (const [file, hash] of Object.entries(savedReceipt.artifacts)) assert.equal(sha256(await readFile(join(outputDirectory, file))), hash);
    const raw = JSON.parse(await readFile(join(outputDirectory, report.runs[0].raw_result_file), 'utf8'));
    assert.equal(raw.requestedSeconds, 30);
    assert.ok(raw.baseline.orders.length > 0);
    assert.ok(raw.baseline.checkpoint_file);
    const finalCheckpoint = JSON.parse(await readFile(join(outputDirectory, raw.baseline.checkpoint_file), 'utf8'));
    assert.equal(sha256(JSON.stringify(finalCheckpoint)), raw.baseline.final_checkpoint_sha256);
    await assert.rejects(exportFixtures(protocolPath, fixtureDirectory), { code: 'EEXIST' });
  } finally {
    assert.equal(resolve(dirnameForCleanup(directory)), resolve(tmpdir()));
    assert.ok(basename(directory).startsWith('rlcd-weight-space-adapter-'));
    await rm(directory, { recursive: true, force: true });
  }
});

function dirnameForCleanup(path: string): string { return resolve(path, '..'); }
