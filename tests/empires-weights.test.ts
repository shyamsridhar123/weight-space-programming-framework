import assert from 'node:assert/strict';
import { test } from 'node:test';
import { World } from '../game/src/empires/simulation.ts';
import {
  chooseNeural, comparePolicies, createDefaultModel, forwardPolicy, layaCandidates,
  policyFeatures, rankActions, tracePolicy, type PolicyModel,
} from '../game/src/empires/policy.ts';
import {
  cloneWeightModel, fingerprintWeightModel, getWeightProposals, presetEdits, STRATEGIC_PRESETS,
  validateWeightModel, WEIGHT_LIMITS, WeightRevisionConflictError, WeightRuntime, WeightValidationError,
  type WeightEdit,
} from '../game/src/empires/weight-runtime.ts';

const context = { source: 'test', reason: 'Exercise a deliberate weight change.', tick: 12 };

test('strict model validation clones tensors and preserves the authored default', () => {
  const authored = createDefaultModel(), canonical = JSON.stringify(authored);
  const clone = validateWeightModel(authored), another = cloneWeightModel(authored);
  assert.deepEqual(clone, authored);
  assert.notEqual(clone.w1[0], authored.w1[0]);
  clone.w1[0][0] = 8; clone.b1[1] = 2; clone.w2[4] = 9;
  clone.hiddenNames[0] = 'Edited label'; another.featureNames[0] = 'Wrong feature';
  assert.equal(JSON.stringify(createDefaultModel()), canonical);
  assert.equal(JSON.stringify(authored), canonical);
});

test('untrusted models reject malformed shapes, feature order, labels and metadata', () => {
  const invalid: [string, (model: PolicyModel) => void][] = [
    ['feature order', model => { [model.featureNames[0], model.featureNames[1]] = [model.featureNames[1], model.featureNames[0]]; }],
    ['feature count', model => { model.featureNames.pop(); }],
    ['hidden count', model => { model.w1.pop(); }],
    ['matrix width', model => { model.w1[2].pop(); }],
    ['bias count', model => { model.b1.pop(); }],
    ['output count', model => { model.w2.push(0); }],
    ['hidden labels', model => { model.hiddenNames[1] = model.hiddenNames[0]; }],
    ['oversized label', model => { model.hiddenNames[1] = 'x'.repeat(121); }],
    ['name metadata', model => { model.name = ''; }],
    ['version metadata', model => { model.version = 'x'.repeat(65); }],
    ['provenance metadata', model => { model.provenance = 'x'.repeat(513); }],
    ['description metadata', model => { model.description = 'x'.repeat(2049); }],
    ['control character', model => { model.name = 'untrusted\0name'; }],
    ['sparse matrix', model => { delete model.w1[0][1]; }],
  ];
  for (const [label, mutate] of invalid) {
    const model = createDefaultModel(); mutate(model);
    assert.throws(() => validateWeightModel(model), WeightValidationError, label);
  }
  assert.throws(() => validateWeightModel({ ...createDefaultModel(), constructor: 'extra' }), /unknown field/);
  let getterRuns = 0;
  const accessor = createDefaultModel();
  Object.defineProperty(accessor, 'b2', { get() { getterRuns++; return 0; } });
  assert.throws(() => validateWeightModel(accessor), /accessors/);
  assert.equal(getterRuns, 0, 'validation must not execute import accessors');
  for (const malformed of [null, [], 'weights', 12, new Date()]) assert.throws(() => validateWeightModel(malformed), WeightValidationError);
});

test('every imported tensor and temperature must be finite and bounded', () => {
  for (const value of [NaN, Infinity, -Infinity, 64.001, -64.001, '1', null]) {
    for (const tensor of ['w1', 'b1', 'w2', 'b2'] as const) {
      const model = createDefaultModel() as unknown as Record<string, any>;
      if (tensor === 'w1') model.w1[4][7] = value;
      else if (tensor === 'b2') model.b2 = value;
      else model[tensor][4] = value;
      assert.throws(() => validateWeightModel(model), WeightValidationError, `${tensor}=${value}`);
    }
  }
  for (const value of [0, -1, 0.0499, 20.0001, NaN, Infinity]) {
    const model = createDefaultModel(); model.temperature = value;
    assert.throws(() => validateWeightModel(model), /temperature/);
  }
  const boundaries = createDefaultModel();
  boundaries.w1[0][0] = -64; boundaries.w1[0][1] = 64; boundaries.temperature = 0.05;
  assert.deepEqual(validateWeightModel(boundaries), boundaries);
  boundaries.temperature = 20;
  assert.equal(validateWeightModel(boundaries).temperature, 20);
});

test('all five parameter surfaces can be edited atomically with immutable revision history', () => {
  const runtime = new WeightRuntime(), initial = runtime.current;
  const edits: WeightEdit[] = [
    { tensor: 'w1', row: 2, column: 5, value: 2 }, { tensor: 'b1', row: 2, value: -0.5 },
    { tensor: 'w2', row: 2, value: 3 }, { tensor: 'b2', value: 0.4 }, { tensor: 'temperature', value: 1.7 },
  ];
  const next = runtime.patch(edits, { ...context, expectedRevision: 0, expectedFingerprint: initial.fingerprint });
  assert.equal(next.revision, 1);
  assert.equal(next.model.w1[2][5], 2); assert.equal(next.model.b1[2], -0.5); assert.equal(next.model.w2[2], 3);
  assert.equal(next.model.b2, 0.4); assert.equal(next.model.temperature, 1.7);
  assert.equal(initial.model.w1[2][5], 0);
  assert.equal(next.provenance.parentRevision, 0); assert.equal(next.provenance.tick, 12);
  assert.equal(next.provenance.source, context.source); assert.equal(next.provenance.reason, context.reason);
  assert.equal(runtime.snapshot, next); assert.equal(runtime.baseline, initial);
  assert.deepEqual(runtime.history.map(snapshot => snapshot.revision), [1, 0]);
  assert.ok(Number.isFinite(Date.parse(next.createdAt)));
  for (const object of [next, next.model, next.model.w1, next.model.w1[0], next.model.b1, next.model.w2, next.model.hiddenNames, next.provenance, runtime.history])
    assert.equal(Object.isFrozen(object), true);
  assert.throws(() => { next.model.w1[0][0] = 2; }, TypeError);
  assert.throws(() => { (runtime.history as unknown[]).pop(); }, TypeError);
  edits[0].value = 60;
  assert.equal(next.model.w1[2][5], 2);
});

test('bad patch indices, ambiguous edits and invalid context never partially apply', () => {
  const runtime = new WeightRuntime(), original = runtime.current, history = runtime.history;
  const invalid: unknown[] = [
    { tensor: 'w1', row: 38, column: 0, value: 1 }, { tensor: 'w1', row: 0, column: 32, value: 1 },
    { tensor: 'w1', row: 1, value: 1 }, { tensor: 'b1', row: -1, value: 1 },
    { tensor: 'w2', row: 1.1, value: 1 }, { tensor: 'w2', row: 1, column: 0, value: 1 },
    { tensor: 'w2', row: 38, value: 1 }, { tensor: 'b2', row: 0, value: 1 },
    { tensor: 'temperature', column: 0, value: 1 }, { tensor: 'temperature', value: 0 },
    { tensor: 'w2', row: 1, value: Infinity }, { tensor: 'other', value: 1 },
    { tensor: 'b2', value: 1, extra: true }, { tensor: 'b2' }, null,
  ];
  for (const edit of invalid) {
    assert.throws(() => runtime.patch([{ tensor: 'b2', value: 3 }, edit as WeightEdit], context), WeightValidationError);
    assert.equal(runtime.current, original); assert.equal(runtime.history, history);
  }
  assert.throws(() => runtime.patch([], context), /1\.\.1294/);
  assert.throws(() => runtime.patch([{ tensor: 'b2', value: 1 }, { tensor: 'b2', value: 2 }], context), /same scalar twice/);
  for (const bad of [{ ...context, source: '' }, { ...context, reason: 'x'.repeat(801) }, { ...context, tick: -1 }, { ...context, tick: 0.1 }])
    assert.throws(() => runtime.apply(createDefaultModel(), bad), WeightValidationError);
  assert.equal(runtime.current, original); assert.equal(runtime.history, history);
});

test('revision and content guards reject stale proposals without rebasing them', () => {
  const runtime = new WeightRuntime(), old = runtime.current;
  runtime.patch([{ tensor: 'w2', row: 36, value: 9 }], context);
  const current = runtime.current, history = runtime.history;
  assert.throws(() => runtime.apply(old.model, { ...context, expectedRevision: 0 }), WeightRevisionConflictError);
  assert.throws(() => runtime.patch([{ tensor: 'b2', value: 1 }], { ...context, expectedFingerprint: old.fingerprint }), WeightRevisionConflictError);
  assert.throws(() => runtime.restore(old.id, { ...context, expectedRevision: 0 }), WeightRevisionConflictError);
  assert.throws(() => runtime.importCheckpoint(runtime.exportCheckpoint(), { ...context, expectedRevision: 0 }), WeightRevisionConflictError);
  assert.equal(runtime.current, current); assert.equal(runtime.history, history);
  assert.equal(runtime.apply(old.model, { ...context, expectedRevision: 1, expectedFingerprint: current.fingerprint }).revision, 2);
});

test('rollback revisions stay monotonic and pinned baselines survive bounded history eviction', () => {
  const runtime = new WeightRuntime(undefined, { historyLimit: 3 }), initial = runtime.baseline;
  const first = runtime.patch([{ tensor: 'b2', value: 1 }], context);
  runtime.patch([{ tensor: 'b2', value: 2 }], context);
  runtime.patch([{ tensor: 'b2', value: 3 }], context);
  assert.deepEqual(runtime.history.map(snapshot => snapshot.revision), [3, 2, 1]);
  const restored = runtime.restore(initial.id, context);
  assert.equal(restored.revision, 4); assert.equal(restored.fingerprint, initial.fingerprint);
  assert.equal(restored.provenance.restoredFrom, initial.id);
  assert.throws(() => runtime.restore(first.id, context), /no longer retained/);
  const pinned = runtime.pinBaseline();
  assert.equal(pinned, restored);
  runtime.patch([{ tensor: 'b2', value: 5 }], context);
  runtime.patch([{ tensor: 'b2', value: 6 }], context);
  runtime.patch([{ tensor: 'b2', value: 7 }], context);
  const reset = runtime.reset(context);
  assert.equal(reset.revision, 8); assert.equal(reset.fingerprint, pinned.fingerprint);
  assert.equal(runtime.baseline, pinned); assert.equal(runtime.history.length, 3);
  assert.equal(new WeightRuntime().historyLimit, WEIGHT_LIMITS.history);
  for (const limit of [0, 1, 201, 2.5, Infinity]) assert.throws(() => new WeightRuntime(undefined, { historyLimit: limit }), WeightValidationError);
});

test('checkpoints roundtrip exact model content into a new local revision with explicit origin', () => {
  const source = new WeightRuntime();
  source.patch([{ tensor: 'w2', row: 30, value: -2.7 }, { tensor: 'temperature', value: 0.9 }], context);
  const serialized = source.exportCheckpoint();
  const receiver = new WeightRuntime();
  receiver.patch([{ tensor: 'b2', value: 2 }], context);
  receiver.patch([{ tensor: 'b2', value: 3 }], context);
  const imported = receiver.importCheckpoint(serialized, { source: 'import', reason: 'Loaded a reviewed local checkpoint.', tick: 200, expectedRevision: 2 });
  assert.equal(imported.revision, 3);
  assert.equal(imported.fingerprint, source.current.fingerprint);
  assert.deepEqual(imported.model, source.current.model);
  assert.equal(imported.provenance.source, 'import'); assert.equal(imported.provenance.tick, 200);
  assert.deepEqual(imported.provenance.importedFrom, {
    fingerprint: source.current.fingerprint, revision: 1, source: context.source, reason: context.reason, createdAt: source.current.createdAt,
  });
  assert.ok(Object.isFrozen(imported.provenance.importedFrom));
  assert.notEqual(imported.model.w1[0], source.current.model.w1[0]);
  const object = JSON.parse(serialized);
  receiver.importCheckpoint(object, context);
  object.snapshot.model.w2[0] = 32;
  assert.notEqual(receiver.current.model.w2[0], 32);
  const nested = new WeightRuntime();
  assert.doesNotThrow(() => nested.importCheckpoint(receiver.exportCheckpoint(), context));
});

test('untrusted checkpoint schema, fingerprint, metadata and oversized JSON fail atomically', () => {
  const runtime = new WeightRuntime(), original = runtime.current;
  const clean = runtime.exportCheckpoint();
  const invalid: ((value: any) => void)[] = [
    value => { value.schemaVersion = 2; }, value => { value.format = 'other'; },
    value => { value.snapshot.model.w2[0] += 1; }, value => { value.snapshot.fingerprint = 'fnv1a64_0000000000000000'; },
    value => { value.snapshot.id = 'r1_wrong'; }, value => { value.snapshot.provenance.source = ''; },
    value => { value.snapshot.provenance.parentRevision = 7; }, value => { value.snapshot.createdAt = 'today'; },
    value => { value.exportedAt = '2026-02-30T00:00:00.000Z'; }, value => { value.snapshot.model.w1[0][1] = null; },
    value => { value.snapshot.revision = -1; }, value => { value.unknown = true; },
  ];
  for (const mutate of invalid) {
    const checkpoint = JSON.parse(clean); mutate(checkpoint);
    assert.throws(() => runtime.importCheckpoint(checkpoint, context), WeightValidationError);
    assert.equal(runtime.current, original);
  }
  assert.throws(() => runtime.importCheckpoint('{not json', context), /valid JSON/);
  assert.throws(() => runtime.importCheckpoint(' '.repeat(250_001), context), /exceeds/);
  assert.throws(() => runtime.importCheckpoint(createDefaultModel(), context), /required fields/);
  assert.equal(runtime.history.length, 1);
});

test('session reload retains exact revision lineage, frozen history and independently pinned baseline', () => {
  const runtime = new WeightRuntime(undefined, { historyLimit: 3 });
  runtime.patch([{ tensor: 'b2', value: 1 }], context);
  const pinned = runtime.pinBaseline();
  for (let value = 2; value <= 6; value++) runtime.patch([{ tensor: 'b2', value }], { ...context, tick: value * 10 });
  assert.equal(runtime.history.some(snapshot => snapshot.id === pinned.id), false);
  const saved = runtime.exportSession(), loaded = WeightRuntime.fromSession(saved);
  assert.deepEqual(loaded.current, runtime.current);
  assert.deepEqual(loaded.history, runtime.history);
  assert.deepEqual(loaded.baseline, pinned);
  assert.equal(loaded.historyLimit, 3);
  assert.notEqual(loaded.current.model.w1[0], runtime.current.model.w1[0]);
  assert.ok(Object.isFrozen(loaded.history)); assert.ok(Object.isFrozen(loaded.baseline.model.w1[0]));
  const previous = loaded.history[1];
  const rollback = loaded.restore(previous.id, { ...context, expectedRevision: 6 });
  assert.equal(rollback.revision, 7); assert.equal(rollback.fingerprint, previous.fingerprint);
  const reset = loaded.reset({ ...context, expectedRevision: 7 });
  assert.equal(reset.revision, 8); assert.equal(reset.fingerprint, pinned.fingerprint);
  assert.equal(runtime.current.revision, 6, 'the reloaded session must not mutate the source session');
  assert.throws(() => loaded.apply(createDefaultModel(), { ...context, expectedRevision: 6 }), WeightRevisionConflictError);
});

test('session roundtrips initial and retained baselines without leaking nested references', () => {
  const initial = new WeightRuntime(), initialLoaded = WeightRuntime.fromSession(initial.exportSession());
  assert.equal(initialLoaded.current, initialLoaded.baseline);
  assert.deepEqual(initialLoaded.current, initial.current);
  initial.patch([{ tensor: 'w2', row: 12, value: 5 }], context);
  initial.pinBaseline();
  initial.patch([{ tensor: 'w2', row: 13, value: 3 }], context);
  const object = JSON.parse(initial.exportSession()), loaded = WeightRuntime.fromSession(object);
  assert.equal(loaded.baseline, loaded.history[1]);
  object.history[0].model.w2[0] = 60;
  object.baseline.provenance.source = 'changed externally';
  assert.notEqual(loaded.current.model.w2[0], 60);
  assert.equal(loaded.baseline.provenance.source, context.source);
});

test('session restoration rejects gaps, duplicates, conflicting baseline claims and corrupt contents', () => {
  const runtime = new WeightRuntime(undefined, { historyLimit: 3 });
  runtime.patch([{ tensor: 'b2', value: 1 }], context);
  runtime.pinBaseline();
  runtime.patch([{ tensor: 'b2', value: 2 }], context);
  const saved = runtime.exportSession();
  const invalid: ((value: any) => void)[] = [
    value => { value.format = 'other'; }, value => { value.schemaVersion = 2; },
    value => { value.history = []; }, value => { value.historyLimit = 1; },
    value => { value.historyLimit = 201; }, value => { value.historyLimit = 2; },
    value => { value.currentId = value.history[1].id; },
    value => { value.history.splice(1, 1); }, value => { value.history.pop(); },
    value => { value.history[1] = value.history[0]; },
    value => { value.history.reverse(); }, value => { value.history[0].model.w1[0][0] = 4; },
    value => { value.baseline.provenance.reason = 'Conflicting origin claim.'; },
    value => { value.baseline.createdAt = 'not a timestamp'; }, value => { value.extra = true; },
  ];
  for (const mutate of invalid) {
    const session = JSON.parse(saved); mutate(session);
    assert.throws(() => WeightRuntime.fromSession(session), WeightValidationError);
  }
  assert.throws(() => WeightRuntime.fromSession(saved.slice(1)), /valid JSON/);
  assert.throws(() => WeightRuntime.fromSession(' '.repeat(WEIGHT_LIMITS.maximumSessionBytes + 1)), /exceeds/);
  assert.equal(runtime.current.revision, 2);
  assert.equal(runtime.history.length, 3);
});

test('fingerprints identify canonical complete model content, independently of object field order', () => {
  const base = createDefaultModel(), fingerprint = fingerprintWeightModel(base);
  assert.match(fingerprint, /^fnv1a64_[0-9a-f]{16}$/);
  assert.equal(fingerprintWeightModel(Object.fromEntries(Object.entries(base).reverse()) as unknown as PolicyModel), fingerprint);
  for (const mutate of [
    (model: PolicyModel) => { model.w1[0][0] += 0.001; }, (model: PolicyModel) => { model.b1[1] += 0.1; },
    (model: PolicyModel) => { model.w2[2] += 0.1; }, (model: PolicyModel) => { model.b2 += 0.1; },
    (model: PolicyModel) => { model.temperature += 0.1; }, (model: PolicyModel) => { model.description += ' Changed.'; },
  ]) {
    const changed = cloneWeightModel(base); mutate(changed);
    assert.notEqual(fingerprintWeightModel(changed), fingerprint);
  }
});

test('explicit live weights change actual decisions without changing frozen or default controllers', () => {
  const world = new World(42), before = world.snapshot(), runtime = new WeightRuntime();
  const defaultRanking = rankActions(world, 0), frozen = runtime.baseline.model;
  assert.notEqual(defaultRanking[0].id, 'wait');
  runtime.patch([{ tensor: 'w2', row: 36, value: 64 }], context);
  assert.equal(chooseNeural(world, 0, runtime.current.model).id, 'wait');
  assert.deepEqual(rankActions(world, 0, frozen), defaultRanking);
  assert.deepEqual(rankActions(world, 0), defaultRanking);
  assert.deepEqual(layaCandidates(world, 0, frozen), layaCandidates(world, 0));
  assert.deepEqual(world.snapshot(), before);
  const altered = cloneWeightModel(frozen); altered.w2[36] = 64;
  const loaded = runtime.apply(altered, context);
  altered.w2[36] = 0;
  assert.equal(loaded.model.w2[36], 64);
});

test('Laya shortlists follow the supplied live weights without changing the frozen default shortlist', () => {
  const world = new World(42), before = world.snapshot(), runtime = new WeightRuntime();
  const original = layaCandidates(world, 0), baseline = runtime.baseline.model;
  assert.notEqual(original[0].id, 'build_house');
  runtime.patch([{ tensor: 'w2', row: baseline.hiddenNames.indexOf('house_bias'), value: 64 }], context);
  const live = layaCandidates(world, 0, runtime.current.model);
  assert.equal(live[0].id, 'build_house');
  assert.notDeepEqual(live.map(candidate => candidate.id), original.map(candidate => candidate.id));
  assert.deepEqual(layaCandidates(world, 0, baseline), original);
  assert.deepEqual(layaCandidates(world, 0), original);
  assert.deepEqual(world.snapshot(), before);
});

test('policy comparison generates one candidate set and trace exactly reconstructs dense inference', () => {
  const world = new World(42), active = createDefaultModel(), baseline = createDefaultModel();
  active.w2[36] = 64;
  let legalityCalls = 0;
  const canCommand = world.canCommand.bind(world);
  world.canCommand = (...args: Parameters<World['canCommand']>) => { legalityCalls++; return canCommand(...args); };
  const baselineRanking = rankActions(world, 0, baseline), onePass = legalityCalls;
  legalityCalls = 0;
  const comparison = comparePolicies(world, 0, active, baseline);
  assert.equal(legalityCalls, onePass, 'comparison must not regenerate candidate legality or features');
  assert.deepEqual(comparison.baseline, baselineRanking);
  assert.deepEqual(comparison.active.map(c => c.id).sort(), comparison.baseline.map(c => c.id).sort());
  assert.deepEqual(comparison.active, rankActions(world, 0, active));
  for (const candidate of comparison.active) {
    const input = policyFeatures(world, 0, candidate), trace = tracePolicy(input, active);
    assert.equal(trace.score, candidate.score); assert.equal(trace.score, forwardPolicy(input, active));
    assert.equal(trace.preActivations.length, 38); assert.equal(trace.hiddenActivations.length, 38);
    assert.equal(trace.contributions.reduce((sum, value) => sum + value, trace.outputBias), trace.score);
    assert.deepEqual(trace.hiddenActivations, trace.preActivations.map(value => Math.max(0, value)));
    assert.deepEqual(trace.contributions, trace.hiddenActivations.map((value, index) => value * active.w2[index]));
    assert.notEqual(trace.input, input);
  }
  const sparse = createDefaultModel();
  sparse.w1.forEach(row => row.fill(0)); sparse.b1.fill(0); sparse.w2.fill(0);
  sparse.w1[2][5] = 2; sparse.b1[2] = -0.5; sparse.w2[2] = 3; sparse.b2 = 0.4;
  const input = Array<number>(32).fill(0); input[5] = 0.75;
  const trace = tracePolicy(input, sparse);
  assert.equal(trace.preActivations[2], 1); assert.equal(trace.contributions[2], 3); assert.equal(trace.score, 3.4);
  assert.throws(() => tracePolicy([1, 2], sparse), /shape/);
  input[4] = Infinity;
  assert.throws(() => tracePolicy(input, sparse), /nonfinite/);
});

test('temperature changes the scored distribution while the deterministic selected action stays stable', () => {
  const world = new World(42), normal = createDefaultModel(), soft = cloneWeightModel(normal);
  soft.temperature = 20;
  const comparison = comparePolicies(world, 0, soft, normal);
  assert.deepEqual(comparison.active.map(candidate => [candidate.id, candidate.score]), comparison.baseline.map(candidate => [candidate.id, candidate.score]));
  assert.ok(comparison.active[0].probability < comparison.baseline[0].probability);
  for (const ranking of [comparison.active, comparison.baseline])
    assert.ok(Math.abs(ranking.reduce((sum, candidate) => sum + candidate.probability, 0) - 1) < 1e-12);
});

test('strategic profiles expose and apply actual named output tensors without mutating their source', () => {
  const base = createDefaultModel(), fingerprint = fingerprintWeightModel(base);
  assert.equal(STRATEGIC_PRESETS.length, 6);
  for (const preset of STRATEGIC_PRESETS) {
    const runtime = new WeightRuntime(base), edits = presetEdits(base, preset.id);
    assert.equal(edits.length, 38); assert.equal(preset.mappings.length, 38);
    const next = runtime.patch(edits, context);
    for (const mapping of preset.mappings) assert.equal(next.model.w2[next.model.hiddenNames.indexOf(mapping.hidden)], mapping.value);
    assert.deepEqual(next.model.w1, base.w1); assert.deepEqual(next.model.b1, base.b1);
    if (preset.id !== 'authored') assert.notEqual(next.fingerprint, fingerprint);
    assert.equal(fingerprintWeightModel(base), fingerprint);
  }
  assert.throws(() => presetEdits(base, 'missing'), /Unknown strategic preset/);
  const renamed = cloneWeightModel(base); renamed.hiddenNames[0] = 'user_neuron_0';
  assert.throws(() => presetEdits(renamed, 'economy'), /requires the authored/);
});

test('Laya proposals are compact bounded numeric edits with an explicit no-op choice', () => {
  const base = createDefaultModel(), before = fingerprintWeightModel(base), proposals = getWeightProposals(base);
  assert.equal(proposals.length, 5); assert.equal(proposals.at(-1)!.id, 'keep'); assert.deepEqual(proposals.at(-1)!.edits, []);
  for (const proposal of proposals) {
    assert.ok(proposal.description.length <= 35); assert.ok(proposal.edits.length <= 2);
    for (const edit of proposal.edits) {
      assert.equal(edit.tensor, 'w2'); assert.equal(edit.column, undefined);
      assert.ok(edit.row! >= 0 && edit.row! < 38); assert.ok(Number.isFinite(edit.value));
      assert.equal(Math.round(edit.value * 10_000) / 10_000, edit.value);
    }
    if (proposal.edits.length) {
      const runtime = new WeightRuntime(base);
      assert.notEqual(runtime.patch(proposal.edits, context).fingerprint, before);
    }
  }
  assert.equal(fingerprintWeightModel(base), before);
  base.w2.fill(64); base.w2[base.hiddenNames.indexOf('home_threat_attack')] = -64;
  assert.deepEqual(getWeightProposals(base).map(proposal => proposal.id), ['keep']);
});
