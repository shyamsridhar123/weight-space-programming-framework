import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LayaCommander } from '../game/src/empires/laya.ts';
import { compactObservation, layaCandidates } from '../game/src/empires/policy.ts';
import { World } from '../game/src/empires/simulation.ts';
import {
  LayaWeightCoach, nativeWeightProposals, type NativeWeightProposal, type NativeWeightProposalFactory, type WeightCoachResponse,
} from '../game/src/empires/weight-coach.ts';
import { WeightRevisionConflictError, WeightRuntime } from '../game/src/empires/weight-runtime.ts';

interface Payload {
  state: string;
  proposals: NativeWeightProposal[];
  match_id: string;
  tick: number;
  base_revision: number;
  base_fingerprint: string;
}

const context = { source: 'test', reason: 'Verify exact live checkpoint application.' };
const mockResponse = (body: unknown, status = 200): Response => ({
  ok: status >= 200 && status < 300, status, json: async () => body,
}) as Response;

function valid(payload: Payload, choice = payload.proposals[0].id): WeightCoachResponse {
  const proposal = payload.proposals.find(proposal => proposal.id === choice)!;
  return {
    choice, probabilities: Object.fromEntries(payload.proposals.map(proposal => [proposal.id, proposal.id === choice ? 1 : 0])),
    confidence: 1, confidence_kind: 'normalized_entropy', inference_ms: 25, queue_ms: 3,
    match_id: payload.match_id, tick: payload.tick,
    model: { weights_sha256: 'a'.repeat(64), model_dir: 'models/laya-game', checkpoint_scope: 'unvalidated-for-rts' },
    kind: 'auxiliary-weight-patch', base_revision: payload.base_revision, base_fingerprint: payload.base_fingerprint,
    patch: proposal.edits.map(edit => ({ ...edit })),
  };
}

function deferredCoach(timeoutMs = 20_000) {
  const calls: { url: RequestInfo | URL; init: RequestInit; payload: Payload; resolve: (response: Response) => void }[] = [];
  const fetcher: typeof fetch = (url, init) => new Promise(resolve => {
    assert.ok(init);
    calls.push({ url, init, payload: JSON.parse(init.body as string) as Payload, resolve });
  });
  return { coach: new LayaWeightCoach(fetcher, timeoutMs), calls };
}

test('the weight coach sends exact revision-bound numerical proposals and returns a patch without applying it', async () => {
  const world = new World(60), runtime = new WeightRuntime(), { coach, calls } = deferredCoach();
  runtime.patch([{ tensor: 'w2', row: 0, value: 4.125 }], context);
  world.step(1);
  const snapshot = runtime.current, proposals = nativeWeightProposals(snapshot);
  const requestedTick = world.tick, requestedTime = world.time, state = compactObservation(world, 1);
  const pending = coach.request(world, 1, 'weight-match', snapshot);
  assert.equal(coach.busy, true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/empires/weight-patch');
  assert.equal(calls[0].init.method, 'POST');
  assert.deepEqual(calls[0].init.headers, { 'Content-Type': 'application/json' });
  assert.deepEqual(calls[0].payload, {
    state, proposals, match_id: 'weight-match', tick: requestedTick,
    base_revision: snapshot.revision, base_fingerprint: snapshot.fingerprint,
  });
  for (const proposal of proposals) for (const edit of proposal.edits) {
    assert.equal(edit.tensor, 'w2');
    assert.equal(edit.before, snapshot.model.w2[edit.row]);
    assert.notEqual(edit.value, edit.before);
  }
  world.step(2);
  const beforeResponse = world.snapshot(), response = valid(calls[0].payload, 'economy');
  calls[0].resolve(mockResponse(response));
  const decision = await pending;
  assert.ok(decision);
  assert.deepEqual(decision.proposal, proposals.find(proposal => proposal.id === 'economy'));
  assert.deepEqual(decision.response, response);
  assert.equal(decision.requestedTick, requestedTick);
  assert.equal(decision.ageSeconds, world.time - requestedTime);
  assert.deepEqual(world.snapshot(), beforeResponse);
  assert.equal(runtime.current, snapshot);
  assert.equal(coach.busy, false);
  response.patch[0].value = 64;
  assert.notEqual(decision.response.patch[0].value, 64, 'the accepted patch must be detached from response objects');
});

test('a native keep choice returns no numerical edits and creates no model revision', async () => {
  const runtime = new WeightRuntime(), snapshot = runtime.current, { coach, calls } = deferredCoach();
  const pending = coach.request(new World(61), 0, 'keep-match', snapshot);
  calls[0].resolve(mockResponse(valid(calls[0].payload, 'keep')));
  const decision = await pending;
  assert.equal(decision?.proposal.id, 'keep');
  assert.deepEqual(decision.proposal.edits, []);
  assert.deepEqual(decision.response.patch, []);
  assert.equal(runtime.current, snapshot);
  assert.equal(runtime.history.length, 1);
  assert.equal(coach.busy, false);
});

test('an optional proposal factory supplies detached alternatives without changing other coach behavior', async () => {
  const world = new World(610), worldBefore = world.checkpoint(), runtime = new WeightRuntime(), snapshot = runtime.current;
  const source = nativeWeightProposals(snapshot).filter(proposal => proposal.id === 'economy' || proposal.id === 'keep');
  source[0].id = 'mission_gather';
  source[0].description = 'Favor the mission resource jobs.';
  const expected = structuredClone(source);
  let payload!: Payload, finish!: (response: Response) => void, factoryCalls = 0;
  const factory: NativeWeightProposalFactory = received => {
    assert.equal(received, snapshot); factoryCalls++;
    return source;
  };
  const coach = new LayaWeightCoach(async (_url, init) => {
    payload = JSON.parse(init!.body as string) as Payload;
    return new Promise(resolve => { finish = resolve; });
  }, 20_000, factory);
  const pending = coach.request(world, 0, 'custom-proposals', snapshot);
  assert.equal(factoryCalls, 1);
  assert.deepEqual(payload.proposals, expected);
  source[0].id = 'changed-after-request';
  source[0].description = 'Changed after dispatch';
  source[0].edits[0].value = 64;
  source.pop();
  finish(mockResponse(valid(payload, 'mission_gather')));
  const decision = await pending;
  assert.ok(decision);
  assert.deepEqual(decision.proposal, expected[0]);
  assert.deepEqual(decision.response.patch, expected[0].edits);
  assert.equal(coach.busy, false);
  assert.equal(runtime.current, snapshot);
  assert.deepEqual(world.checkpoint(), worldBefore);
});

test('invalid custom alternatives are rejected before contacting the native model', async () => {
  const world = new World(611), worldBefore = world.checkpoint(), snapshot = new WeightRuntime().current;
  const cases: [string, (proposals: NativeWeightProposal[]) => void][] = [
    ['too few', proposals => { proposals.pop(); }],
    ['too many', proposals => { proposals.push(...structuredClone(proposals), ...structuredClone(proposals)); }],
    ['duplicate IDs', proposals => { proposals[0].id = 'keep'; }],
    ['unsafe ID', proposals => { proposals[0].id = '<script>'; }],
    ['long ID', proposals => { proposals[0].id = 'a'.repeat(49); }],
    ['blank description', proposals => { proposals[0].description = ' '; }],
    ['long description', proposals => { proposals[0].description = 'a'.repeat(61); }],
    ['control characters', proposals => { proposals[0].description = 'bad\u0000text'; }],
    ['missing keep', proposals => { proposals[1] = { ...structuredClone(proposals[0]), id: 'another' }; }],
    ['empty action', proposals => { proposals[0].edits = []; }],
    ['hidden keep edit', proposals => { proposals[1].edits = structuredClone(proposals[0].edits); }],
    ['three edits', proposals => { proposals[0].edits.push(structuredClone(proposals[0].edits[0])); }],
    ['duplicate row', proposals => { proposals[0].edits[1] = structuredClone(proposals[0].edits[0]); }],
    ['wrong before', proposals => { proposals[0].edits[0].before += 0.1; }],
    ['unknown row', proposals => { proposals[0].edits[0].row = 38; }],
    ['fractional row', proposals => { proposals[0].edits[0].row = 0.5; }],
    ['nonfinite value', proposals => { proposals[0].edits[0].value = Infinity; }],
    ['out of bounds', proposals => { proposals[0].edits[0].value = 65; }],
    ['large change', proposals => { proposals[0].edits[0].value = proposals[0].edits[0].before + 5; }],
    ['no change', proposals => { proposals[0].edits[0].value = proposals[0].edits[0].before; }],
  ];
  let calls = 0;
  for (const [label, mutate] of cases) {
    const proposals = nativeWeightProposals(snapshot).filter(proposal => proposal.id === 'economy' || proposal.id === 'keep');
    mutate(proposals);
    const coach = new LayaWeightCoach(async () => { calls++; throw new Error('must not fetch'); }, 20_000, () => proposals);
    await assert.rejects(coach.request(world, 0, 'invalid-proposals', snapshot), /bounded coach/, label);
    assert.equal(coach.busy, false);
  }
  assert.equal(calls, 0);
  assert.deepEqual(world.checkpoint(), worldBefore);
  assert.throws(() => new LayaWeightCoach(undefined, undefined, null as unknown as NativeWeightProposalFactory), /must be a function/);
});

test('a returned patch cannot overwrite a model revision changed while native inference was pending', async () => {
  const runtime = new WeightRuntime(), requested = runtime.current, { coach, calls } = deferredCoach();
  const pending = coach.request(new World(62), 0, 'edited-match', requested);
  runtime.patch([{ tensor: 'w2', row: 3, value: 8 }], context);
  const current = runtime.current;
  calls[0].resolve(mockResponse(valid(calls[0].payload)));
  const decision = await pending;
  assert.ok(decision);
  const edits = decision.response.patch.map(({ tensor, row, value }) => ({ tensor, row, value }));
  assert.throws(() => runtime.patch(edits, {
    ...context, expectedRevision: decision.response.base_revision, expectedFingerprint: decision.response.base_fingerprint,
  }), WeightRevisionConflictError);
  assert.equal(runtime.current, current);
  const restored = runtime.restore(requested.id, context);
  assert.equal(restored.fingerprint, requested.fingerprint);
  assert.throws(() => runtime.patch(edits, { ...context, expectedRevision: requested.revision }), WeightRevisionConflictError);
});

test('overlapping weight requests do not dispatch or release the active request', async () => {
  const world = new World(63), snapshot = new WeightRuntime().current, { coach, calls } = deferredCoach();
  const first = coach.request(world, 0, 'busy-match', snapshot);
  assert.equal(await coach.request(world, 0, 'busy-match', snapshot), null);
  assert.equal(calls.length, 1);
  assert.equal(coach.busy, true);
  calls[0].resolve(mockResponse(valid(calls[0].payload)));
  assert.ok(await first);
  assert.equal(coach.busy, false);
});

test('weight cancellation settles a transport ignoring abort and cannot release a newer request', async () => {
  const snapshot = new WeightRuntime().current, { coach, calls } = deferredCoach();
  const first = coach.request(new World(64), 0, 'old-match', snapshot);
  coach.cancel();
  assert.equal(coach.busy, false);
  assert.equal(calls[0].init.signal?.aborted, true);
  const second = coach.request(new World(65), 0, 'new-match', snapshot);
  assert.equal(await first, null);
  assert.equal(coach.busy, true);
  calls[0].resolve(mockResponse(valid(calls[0].payload)));
  await Promise.resolve();
  assert.equal(coach.busy, true);
  calls[1].resolve(mockResponse(valid(calls[1].payload)));
  assert.equal((await second)?.response.match_id, 'new-match');
  assert.equal(coach.busy, false);
});

test('weight cancellation settles even when response body parsing ignores abort', async () => {
  const snapshot = new WeightRuntime().current, { coach, calls } = deferredCoach();
  let finishBody!: (body: unknown) => void, beginBody!: () => void;
  const bodyStarted = new Promise<void>(resolve => { beginBody = resolve; });
  const first = coach.request(new World(66), 0, 'old-body', snapshot);
  calls[0].resolve({ ok: true, status: 200, json: () => {
    beginBody();
    return new Promise(resolve => { finishBody = resolve; });
  } } as Response);
  await bodyStarted;
  coach.cancel();
  assert.equal(calls[0].init.signal?.aborted, true);
  const second = coach.request(new World(67), 0, 'new-body', snapshot);
  assert.equal(await first, null);
  finishBody(valid(calls[0].payload));
  await Promise.resolve();
  assert.equal(coach.busy, true);
  calls[1].resolve(mockResponse(valid(calls[1].payload)));
  assert.equal((await second)?.response.match_id, 'new-body');
  assert.equal(coach.busy, false);
});

test('weight request wall deadlines cover stalled headers and stalled response bodies', async () => {
  const snapshot = new WeightRuntime().current, { coach, calls } = deferredCoach(10);
  await assert.rejects(coach.request(new World(68), 0, 'header-timeout', snapshot), /timed out/);
  assert.equal(calls[0].init.signal?.aborted, true);
  assert.equal(coach.busy, false);
  let signal: AbortSignal | null | undefined;
  const bodyCoach = new LayaWeightCoach(async (_url, init) => {
    signal = init?.signal;
    return { ok: true, status: 200, json: () => new Promise(() => {}) } as Response;
  }, 10);
  await assert.rejects(bodyCoach.request(new World(69), 0, 'body-timeout', snapshot), /timed out/);
  assert.equal(signal?.aborted, true);
  assert.equal(bodyCoach.busy, false);
  for (const timeout of [0, -1, Infinity, NaN]) assert.throws(() => new LayaWeightCoach(globalThis.fetch, timeout), /positive number/);
});

test('weight responses from stale, ended or rewound matches are discarded', async () => {
  const snapshot = new WeightRuntime().current;
  for (const change of ['stale', 'won', 'lost', 'draw', 'tick-rewound', 'time-rewound'] as const) {
    const world = new World(70), { coach, calls } = deferredCoach();
    world.step(1);
    const pending = coach.request(world, 0, `world-${change}`, snapshot);
    if (change === 'stale') world.time += 15.1;
    else if (change === 'tick-rewound') world.tick = 0;
    else if (change === 'time-rewound') world.time = 0;
    else world.status = change;
    const before = world.snapshot();
    calls[0].resolve(mockResponse(valid(calls[0].payload)));
    assert.equal(await pending, null, change);
    assert.deepEqual(world.snapshot(), before);
    assert.equal(coach.busy, false);
  }
});

test('a weight patch exactly fifteen simulation seconds old can still be reviewed', async () => {
  const world = new World(71), { coach, calls } = deferredCoach();
  const pending = coach.request(world, 0, 'deadline-boundary', new WeightRuntime().current);
  world.time = 15;
  world.tick = 150;
  calls[0].resolve(mockResponse(valid(calls[0].payload)));
  assert.equal((await pending)?.ageSeconds, 15);
});

test('ended matches and fully saturated proposals never dispatch a weight request', async () => {
  const world = new World(72), runtime = new WeightRuntime(), { coach, calls } = deferredCoach();
  world.status = 'won';
  assert.equal(await coach.request(world, 0, 'ended-match', runtime.current), null);
  world.status = 'playing';
  const saturated = structuredClone(runtime.current.model);
  saturated.w2.fill(64);
  saturated.w2[saturated.hiddenNames.indexOf('home_threat_attack')] = -64;
  runtime.apply(saturated, context);
  await assert.rejects(coach.request(world, 0, 'saturated-match', runtime.current), /no available numerical changes/);
  assert.equal(calls.length, 0);
  assert.equal(coach.busy, false);
});

const invalidResponses: [string, (body: Record<string, any>) => void, RegExp][] = [
  ['match ID', body => { body.match_id = 'another-match'; }, /different match or tick/],
  ['tick', body => { body.tick += 1; }, /different match or tick/],
  ['choice', body => { body.choice = 'invented_patch'; }, /unknown candidate/],
  ['probability evidence', body => { delete body.probabilities.keep; }, /probabilit/],
  ['kind', body => { body.kind = 'laya-model-weights'; }, /different weight revision/],
  ['base revision', body => { body.base_revision += 1; }, /different weight revision/],
  ['base fingerprint', body => { body.base_fingerprint = 'fnv1a64_0000000000000000'; }, /different weight revision/],
  ['missing patch', body => { delete body.patch; }, /changed the proposed numerical patch/],
  ['patch length', body => { body.patch.pop(); }, /changed the proposed numerical patch/],
  ['extra edit', body => { body.patch.push({ tensor: 'w2', row: 0, before: 0, value: 64 }); }, /changed the proposed numerical patch/],
  ['tensor', body => { body.patch[0].tensor = 'w1'; }, /changed the proposed numerical patch/],
  ['row', body => { body.patch[0].row = 38; }, /changed the proposed numerical patch/],
  ['before value', body => { body.patch[0].before += 0.1; }, /changed the proposed numerical patch/],
  ['target value', body => { body.patch[0].value += 0.1; }, /changed the proposed numerical patch/],
  ['nonfinite value', body => { body.patch[0].value = Infinity; }, /changed the proposed numerical patch/],
  ['string value', body => { body.patch[0].value = String(body.patch[0].value); }, /changed the proposed numerical patch/],
  ['extra coordinate', body => { body.patch[0].column = 0; }, /changed the proposed numerical patch/],
  ['null edit', body => { body.patch[0] = null; }, /invalid numerical patch/],
  ['array edit', body => { body.patch[0] = []; }, /invalid numerical patch/],
  ['reordered edits', body => { body.patch.reverse(); }, /changed the proposed numerical patch/],
];

for (const [label, mutate, expectedError] of invalidResponses) {
  test(`a tampered weight response with invalid ${label} fails without model or world mutation`, async () => {
    const world = new World(73), before = world.snapshot(), runtime = new WeightRuntime(), snapshot = runtime.current;
    const coach = new LayaWeightCoach(async (_url, init) => {
      const payload = JSON.parse(init!.body as string) as Payload, response = valid(payload);
      mutate(response);
      return mockResponse(response);
    });
    await assert.rejects(coach.request(world, 0, 'tampered-match', snapshot), expectedError);
    assert.equal(coach.busy, false);
    assert.equal(runtime.current, snapshot);
    assert.deepEqual(world.snapshot(), before);
  });
}

test('a keep response carrying a hidden numerical edit is rejected', async () => {
  const snapshot = new WeightRuntime().current;
  const coach = new LayaWeightCoach(async (_url, init) => {
    const payload = JSON.parse(init!.body as string) as Payload, response = valid(payload, 'keep');
    response.patch = payload.proposals[0].edits;
    return mockResponse(response);
  });
  await assert.rejects(coach.request(new World(74), 0, 'hidden-edit', snapshot), /changed the proposed numerical patch/);
  assert.equal(coach.busy, false);
});

test('native weight errors, invalid JSON and transport failures reach the caller and release the coach', async () => {
  const cases: [typeof fetch, RegExp][] = [
    [async () => mockResponse({ detail: 'Laya cannot preserve this input' }, 422), /cannot preserve this input/],
    [async () => mockResponse({ detail: [{ msg: 'Invalid patch', input: 'private input' }, { msg: 'Invalid tick' }] }, 422), /Invalid patch; Invalid tick/],
    [async () => new Response('<html>not JSON</html>', { status: 200 }), /not valid JSON/],
    [async () => new Response('<html>bad gateway</html>', { status: 502 }), /HTTP 502/],
    [async () => { throw new Error('Network disconnected'); }, /Network disconnected/],
    [async () => { throw 'transport failure'; }, /Laya weight request failed/],
  ];
  for (const [fetcher, expectedError] of cases) {
    const world = new World(75), before = world.snapshot(), snapshot = new WeightRuntime().current;
    const coach = new LayaWeightCoach(fetcher);
    await assert.rejects(coach.request(world, 0, 'failed-match', snapshot), (error: Error) => {
      assert.match(error.message, expectedError);
      assert.doesNotMatch(error.message, /private input|\[object Object\]/);
      return true;
    });
    assert.equal(coach.busy, false);
    assert.deepEqual(world.snapshot(), before);
  }
});

test('Laya action requests use an explicitly supplied live model for their shortlist', async () => {
  const world = new World(76), runtime = new WeightRuntime(), baseline = runtime.current.model;
  const defaultChoices = layaCandidates(world, 0).map(candidate => candidate.id);
  runtime.patch([{ tensor: 'w2', row: baseline.hiddenNames.indexOf('house_bias'), value: 64 }], context);
  const liveChoices = layaCandidates(world, 0, runtime.current.model).map(candidate => candidate.id);
  assert.equal(liveChoices[0], 'build_house');
  assert.notDeepEqual(liveChoices, defaultChoices);
  const sentChoices: string[][] = [];
  const commander = new LayaCommander(async (_url, init) => {
    const payload = JSON.parse(init!.body as string) as { candidates: { id: string }[]; match_id: string; tick: number };
    sentChoices.push(payload.candidates.map(candidate => candidate.id));
    return mockResponse({
      choice: payload.candidates[0].id,
      probabilities: Object.fromEntries(payload.candidates.map((candidate, index) => [candidate.id, index === 0 ? 1 : 0])),
      confidence: 1, confidence_kind: 'normalized_entropy', inference_ms: 25, queue_ms: 3,
      match_id: payload.match_id, tick: payload.tick, model: { weights_sha256: 'a'.repeat(64) },
    });
  });
  assert.equal((await commander.decide(world, 0, 'live-model', runtime.current.model))?.candidate.id, 'build_house');
  await commander.decide(world, 0, 'default-model');
  await commander.decide(world, 0, 'frozen-model', baseline);
  assert.deepEqual(sentChoices, [liveChoices, defaultChoices, defaultChoices]);
});
