import { createDefaultModel, FEATURE_NAMES, type PolicyModel } from './policy';

export const WEIGHT_LIMITS = Object.freeze({
  inputs: 32, hidden: 38, minimum: -64, maximum: 64,
  minimumTemperature: 0.05, maximumTemperature: 20,
  history: 40, maximumHistory: 200, maximumCheckpointBytes: 250_000, maximumSessionBytes: 16_000_000,
});

export class WeightValidationError extends Error {
  constructor(message: string) { super(message); this.name = 'WeightValidationError'; }
}

export class WeightRevisionConflictError extends Error {
  constructor(message: string) { super(message); this.name = 'WeightRevisionConflictError'; }
}

export interface WeightEdit {
  tensor: 'w1' | 'b1' | 'w2' | 'b2' | 'temperature';
  /** Hidden-neuron index, 0..37, for w1, b1 and w2. */
  row?: number;
  /** Input-feature index, 0..31; only w1 uses a column. */
  column?: number;
  value: number;
}

export interface WeightContext {
  source: string;
  reason: string;
  tick?: number;
  expectedRevision?: number;
  expectedFingerprint?: string;
}

export interface ImportedWeightOrigin {
  fingerprint: string;
  revision: number;
  source: string;
  reason: string;
  createdAt: string;
}

export interface WeightProvenance {
  source: string;
  reason: string;
  tick: number | null;
  parentRevision: number | null;
  restoredFrom?: string;
  /** Imported claims are retained as metadata, not verified authorship. */
  importedFrom?: ImportedWeightOrigin;
}

export interface WeightSnapshot {
  readonly id: string;
  readonly revision: number;
  readonly fingerprint: string;
  /** Deeply frozen at runtime; use cloneWeightModel() to obtain an editable copy. */
  readonly model: PolicyModel;
  readonly provenance: Readonly<WeightProvenance>;
  readonly createdAt: string;
}

export interface WeightCheckpoint {
  format: 'crown-charter-weights';
  schemaVersion: 1;
  exportedAt: string;
  snapshot: WeightSnapshot;
}

export interface WeightSession {
  format: 'crown-charter-weight-session';
  schemaVersion: 1;
  exportedAt: string;
  historyLimit: number;
  currentId: string;
  history: readonly WeightSnapshot[];
  baseline: WeightSnapshot;
}

const modelKeys = ['name', 'version', 'provenance', 'description', 'featureNames', 'hiddenNames', 'w1', 'b1', 'w2', 'b2', 'temperature'];
const fail = (message: string): never => { throw new WeightValidationError(message); };

function record(value: unknown, label: string, required: readonly string[], optional: readonly string[] = []): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return fail(`${label} must be an object`);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) return fail(`${label} must be a plain object`);
  const result = value as Record<string, unknown>;
  if (required.some(key => !Object.hasOwn(result, key))) return fail(`${label} is missing required fields`);
  const keys = Reflect.ownKeys(result);
  if (keys.some(key => typeof key !== 'string' || !required.includes(key) && !optional.includes(key)))
    return fail(`${label} contains an unknown field`);
  if (keys.some(key => !Object.hasOwn(Object.getOwnPropertyDescriptor(result, key)!, 'value')))
    return fail(`${label} must contain data fields, not accessors`);
  return result;
}

function textField(value: unknown, label: string, maximum: number): string {
  if (typeof value !== 'string' || value.length > maximum || value.trim().length === 0
    || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value))
    return fail(`${label} must be nonempty text of at most ${maximum} characters without control characters`);
  return value;
}

function numberField(value: unknown, label: string, minimum: number = WEIGHT_LIMITS.minimum, maximum: number = WEIGHT_LIMITS.maximum): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum)
    return fail(`${label} must be finite and between ${minimum} and ${maximum}`);
  return Object.is(value, -0) ? 0 : value;
}

function integerField(value: unknown, label: string, maximum = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value > maximum)
    return fail(`${label} must be an integer between 0 and ${maximum}`);
  return value;
}

function arrayField(value: unknown, length: number, label: string): unknown[] {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype || value.length !== length || Reflect.ownKeys(value).length !== length + 1)
    return fail(`${label} must be a dense array of exactly ${length} entries`);
  for (let index = 0; index < length; index++) {
    if (!Object.hasOwn(value, index)) return fail(`${label} must not contain holes`);
    if (!Object.hasOwn(Object.getOwnPropertyDescriptor(value, index)!, 'value')) return fail(`${label} cannot contain accessors`);
  }
  return value;
}

function numericArray(value: unknown, length: number, label: string): number[] {
  return arrayField(value, length, label).map((entry, index) => numberField(entry, `${label}[${index}]`));
}

/** Validate untrusted data, enforce the executable architecture, and return a detached mutable clone. */
export function validateWeightModel(value: unknown): PolicyModel {
  const candidate = record(value, 'Weight model', modelKeys);
  const featureNames = arrayField(candidate.featureNames, WEIGHT_LIMITS.inputs, 'featureNames').map((entry, index) => {
    if (entry !== FEATURE_NAMES[index]) return fail(`featureNames[${index}] must be ${FEATURE_NAMES[index]}; feature order cannot change`);
    return entry as string;
  });
  const hiddenNames = arrayField(candidate.hiddenNames, WEIGHT_LIMITS.hidden, 'hiddenNames')
    .map((entry, index) => textField(entry, `hiddenNames[${index}]`, 120));
  if (new Set(hiddenNames).size !== WEIGHT_LIMITS.hidden) return fail('Hidden-neuron names must be unique');
  return {
    name: textField(candidate.name, 'name', 120), version: textField(candidate.version, 'version', 64),
    provenance: textField(candidate.provenance, 'provenance', 512), description: textField(candidate.description, 'description', 2048),
    featureNames, hiddenNames,
    w1: arrayField(candidate.w1, WEIGHT_LIMITS.hidden, 'w1').map((row, index) => numericArray(row, WEIGHT_LIMITS.inputs, `w1[${index}]`)),
    b1: numericArray(candidate.b1, WEIGHT_LIMITS.hidden, 'b1'),
    w2: numericArray(candidate.w2, WEIGHT_LIMITS.hidden, 'w2'),
    b2: numberField(candidate.b2, 'b2'),
    temperature: numberField(candidate.temperature, 'temperature', WEIGHT_LIMITS.minimumTemperature, WEIGHT_LIMITS.maximumTemperature),
  };
}

export function cloneWeightModel(value: PolicyModel): PolicyModel { return validateWeightModel(value); }

function fingerprintValidated(model: PolicyModel): string {
  // FNV-1a over canonical JSON encoded as UTF-16LE. This is content identity,
  // not a cryptographic signature or evidence of checkpoint authorship.
  const content = JSON.stringify(model);
  let hash = 0xcbf29ce484222325n;
  for (let index = 0; index < content.length; index++) {
    const code = content.charCodeAt(index);
    hash = BigInt.asUintN(64, (hash ^ BigInt(code & 255)) * 0x100000001b3n);
    hash = BigInt.asUintN(64, (hash ^ BigInt(code >>> 8)) * 0x100000001b3n);
  }
  return `fnv1a64_${hash.toString(16).padStart(16, '0')}`;
}

/** Includes all model tensors, temperature, labels, and model metadata in canonical order. */
export function fingerprintWeightModel(value: PolicyModel): string { return fingerprintValidated(validateWeightModel(value)); }

function freezeModel(model: PolicyModel): PolicyModel {
  for (const row of model.w1) Object.freeze(row);
  for (const array of [model.w1, model.b1, model.w2, model.featureNames, model.hiddenNames]) Object.freeze(array);
  return Object.freeze(model);
}

function dateField(value: unknown, label: string): string {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)
    || !Number.isFinite(Date.parse(value)) || new Date(value).toISOString() !== value)
    return fail(`${label} must be a canonical ISO timestamp`);
  return value;
}

function fingerprintField(value: unknown): string {
  if (typeof value !== 'string' || !/^fnv1a64_[0-9a-f]{16}$/.test(value)) return fail('Invalid weight content fingerprint');
  return value;
}

function contextFields(value: WeightContext): WeightContext {
  const context = record(value, 'Weight context', ['source', 'reason'], ['tick', 'expectedRevision', 'expectedFingerprint']);
  const result: WeightContext = { source: textField(context.source, 'source', 80), reason: textField(context.reason, 'reason', 800) };
  if (context.tick !== undefined) result.tick = integerField(context.tick, 'tick');
  if (context.expectedRevision !== undefined) result.expectedRevision = integerField(context.expectedRevision, 'expectedRevision');
  if (context.expectedFingerprint !== undefined) result.expectedFingerprint = fingerprintField(context.expectedFingerprint);
  return result;
}

function validateProvenance(value: unknown): WeightProvenance {
  const source = record(value, 'Weight provenance', ['source', 'reason', 'tick', 'parentRevision'], ['restoredFrom', 'importedFrom']);
  const result: WeightProvenance = {
    source: textField(source.source, 'source', 80), reason: textField(source.reason, 'reason', 800),
    tick: source.tick === null ? null : integerField(source.tick, 'tick'),
    parentRevision: source.parentRevision === null ? null : integerField(source.parentRevision, 'parentRevision'),
  };
  if (source.restoredFrom !== undefined) result.restoredFrom = textField(source.restoredFrom, 'restoredFrom', 80);
  if (source.importedFrom !== undefined) {
    const origin = record(source.importedFrom, 'Imported provenance', ['fingerprint', 'revision', 'source', 'reason', 'createdAt']);
    result.importedFrom = {
      fingerprint: fingerprintField(origin.fingerprint), revision: integerField(origin.revision, 'Imported revision'),
      source: textField(origin.source, 'Imported source', 80), reason: textField(origin.reason, 'Imported reason', 800),
      createdAt: dateField(origin.createdAt, 'Imported createdAt'),
    };
  }
  return result;
}

function snapshotId(revision: number, fingerprint: string): string { return `r${revision}_${fingerprint}`; }

function makeSnapshot(model: PolicyModel, revision: number, provenance: WeightProvenance): WeightSnapshot {
  const fingerprint = fingerprintValidated(model);
  if (provenance.importedFrom) Object.freeze(provenance.importedFrom);
  return Object.freeze({
    id: snapshotId(revision, fingerprint), revision, fingerprint,
    model: freezeModel(model), provenance: Object.freeze(provenance), createdAt: new Date().toISOString(),
  });
}

function parseDocument(input: unknown, maximumBytes: number): unknown {
  if (typeof input === 'string') {
    // A byte bound also prevents large strings before JSON parsing. UTF-8 uses
    // at most four bytes per code point; TextEncoder gives the exact import size.
    if (input.length > maximumBytes || new TextEncoder().encode(input).byteLength > maximumBytes)
      return fail(`Weight document exceeds ${maximumBytes} bytes`);
    try { return JSON.parse(input); } catch { return fail('Weight document is not valid JSON'); }
  }
  return input;
}

function checkpointSnapshot(input: unknown): WeightSnapshot {
  const value = parseDocument(input, WEIGHT_LIMITS.maximumCheckpointBytes);
  const envelope = record(value, 'Weight checkpoint', ['format', 'schemaVersion', 'exportedAt', 'snapshot']);
  if (envelope.format !== 'crown-charter-weights' || envelope.schemaVersion !== 1) return fail('Unsupported weight checkpoint format or schemaVersion');
  dateField(envelope.exportedAt, 'exportedAt');
  return validatedSnapshot(envelope.snapshot);
}

function validatedSnapshot(input: unknown): WeightSnapshot {
  const candidate = record(input, 'Weight snapshot', ['id', 'revision', 'fingerprint', 'model', 'provenance', 'createdAt']);
  const model = validateWeightModel(candidate.model), revision = integerField(candidate.revision, 'revision');
  const fingerprint = fingerprintField(candidate.fingerprint);
  if (fingerprint !== fingerprintValidated(model)) return fail('Checkpoint fingerprint does not match its model content');
  if (candidate.id !== snapshotId(revision, fingerprint)) return fail('Checkpoint revision id does not match its revision and fingerprint');
  const provenance = validateProvenance(candidate.provenance);
  if (provenance.parentRevision !== (revision === 0 ? null : revision - 1)) return fail('Checkpoint parent revision is inconsistent');
  if (provenance.importedFrom) Object.freeze(provenance.importedFrom);
  return Object.freeze({
    id: candidate.id as string, revision, fingerprint, model: freezeModel(model),
    provenance: Object.freeze(provenance), createdAt: dateField(candidate.createdAt, 'createdAt'),
  });
}

export class WeightRuntime {
  private active: WeightSnapshot;
  private pinned: WeightSnapshot;
  private revisions: readonly WeightSnapshot[];
  readonly historyLimit: number;

  constructor(initialModel: unknown = createDefaultModel(), options: { historyLimit?: number } = {}) {
    const limit = options.historyLimit ?? WEIGHT_LIMITS.history;
    integerField(limit, 'historyLimit', WEIGHT_LIMITS.maximumHistory);
    if (limit < 2) fail('historyLimit must retain at least two revisions');
    this.historyLimit = limit;
    this.active = makeSnapshot(validateWeightModel(initialModel), 0, {
      source: 'initial', reason: 'Initial model; no optimization has run in this session.', tick: null, parentRevision: null,
    });
    this.pinned = this.active;
    this.revisions = Object.freeze([this.active]);
  }

  get current(): WeightSnapshot { return this.active; }
  get snapshot(): WeightSnapshot { return this.active; }
  get baseline(): WeightSnapshot { return this.pinned; }
  /** Newest first; eviction never removes the separately pinned baseline. */
  get history(): readonly WeightSnapshot[] { return this.revisions; }

  private guard(context: WeightContext): WeightContext {
    const validated = contextFields(context);
    if (validated.expectedRevision !== undefined && validated.expectedRevision !== this.active.revision)
      throw new WeightRevisionConflictError(`Expected revision ${validated.expectedRevision}; active revision is ${this.active.revision}`);
    if (validated.expectedFingerprint !== undefined && validated.expectedFingerprint !== this.active.fingerprint)
      throw new WeightRevisionConflictError('The active model changed since this proposal was generated');
    return validated;
  }

  private commit(model: PolicyModel, context: WeightContext, extra: Pick<WeightProvenance, 'restoredFrom' | 'importedFrom'> = {}): WeightSnapshot {
    if (this.active.revision >= Number.MAX_SAFE_INTEGER) return fail('Weight revision counter is exhausted');
    const next = makeSnapshot(model, this.active.revision + 1, {
      source: context.source, reason: context.reason, tick: context.tick ?? null, parentRevision: this.active.revision, ...extra,
    });
    // All data and guards have passed before either reference changes.
    this.revisions = Object.freeze([next, ...this.revisions].slice(0, this.historyLimit));
    this.active = next;
    return next;
  }

  apply(model: unknown, context: WeightContext): WeightSnapshot {
    const validContext = this.guard(context);
    return this.commit(validateWeightModel(model), validContext);
  }

  patch(edits: readonly WeightEdit[], context: WeightContext): WeightSnapshot {
    const validContext = this.guard(context);
    if (!Array.isArray(edits) || edits.length === 0 || edits.length > 1294) return fail('A weight patch must contain 1..1294 scalar edits');
    arrayField(edits, edits.length, 'Weight patch');
    const model = cloneWeightModel(this.active.model), seen = new Set<string>();
    for (const edit of edits) {
      const value = record(edit, 'Weight edit', ['tensor', 'value'], ['row', 'column']);
      const tensor = value.tensor;
      if (tensor !== 'w1' && tensor !== 'b1' && tensor !== 'w2' && tensor !== 'b2' && tensor !== 'temperature') return fail('Unknown weight tensor');
      const scalar = tensor === 'temperature'
        ? numberField(value.value, 'temperature', WEIGHT_LIMITS.minimumTemperature, WEIGHT_LIMITS.maximumTemperature)
        : numberField(value.value, `${tensor} value`);
      let row: number | undefined, column: number | undefined;
      if (tensor === 'w1' || tensor === 'b1' || tensor === 'w2') row = integerField(value.row, `${tensor} row`, WEIGHT_LIMITS.hidden - 1);
      else if (Object.hasOwn(value, 'row')) return fail(`${tensor} does not have a row index`);
      if (tensor === 'w1') column = integerField(value.column, 'w1 column', WEIGHT_LIMITS.inputs - 1);
      else if (Object.hasOwn(value, 'column')) return fail(`${tensor} does not have a column index`);
      const coordinate = `${tensor}:${row ?? ''}:${column ?? ''}`;
      if (seen.has(coordinate)) return fail('A weight patch cannot edit the same scalar twice');
      seen.add(coordinate);
      if (tensor === 'w1') model.w1[row!][column!] = scalar;
      else if (tensor === 'b1' || tensor === 'w2') model[tensor][row!] = scalar;
      else model[tensor] = scalar;
    }
    return this.commit(model, validContext);
  }

  restore(id: string, context: WeightContext): WeightSnapshot {
    const validContext = this.guard(context);
    const target = this.revisions.find(snapshot => snapshot.id === id) ?? (this.pinned.id === id ? this.pinned : undefined);
    if (!target) return fail('That revision is no longer retained; import its exported checkpoint to restore it');
    return this.commit(cloneWeightModel(target.model), validContext, { restoredFrom: target.id });
  }

  pinBaseline(): WeightSnapshot { this.pinned = this.active; return this.pinned; }

  reset(context: WeightContext): WeightSnapshot { return this.restore(this.pinned.id, context); }

  exportCheckpoint(): string {
    const checkpoint: WeightCheckpoint = { format: 'crown-charter-weights', schemaVersion: 1, exportedAt: new Date().toISOString(), snapshot: this.active };
    return JSON.stringify(checkpoint, null, 2);
  }

  importCheckpoint(input: unknown, context: WeightContext): WeightSnapshot {
    const validContext = this.guard(context), imported = checkpointSnapshot(input);
    return this.commit(imported.model, validContext, {
      importedFrom: {
        fingerprint: imported.fingerprint, revision: imported.revision,
        source: imported.provenance.source, reason: imported.provenance.reason, createdAt: imported.createdAt,
      },
    });
  }

  /** Compact serialization retains the live revision counter, bounded history, and pinned baseline. */
  exportSession(): string {
    const session: WeightSession = {
      format: 'crown-charter-weight-session', schemaVersion: 1, exportedAt: new Date().toISOString(),
      historyLimit: this.historyLimit, currentId: this.active.id, history: this.revisions, baseline: this.pinned,
    };
    return JSON.stringify(session);
  }

  /** Strict session restoration; unlike checkpoint import, this preserves the entire local revision lineage. */
  static fromSession(input: unknown): WeightRuntime {
    const session = record(parseDocument(input, WEIGHT_LIMITS.maximumSessionBytes), 'Weight session',
      ['format', 'schemaVersion', 'exportedAt', 'historyLimit', 'currentId', 'history', 'baseline']);
    if (session.format !== 'crown-charter-weight-session' || session.schemaVersion !== 1) return fail('Unsupported weight session format or schemaVersion');
    dateField(session.exportedAt, 'exportedAt');
    const limit = integerField(session.historyLimit, 'historyLimit', WEIGHT_LIMITS.maximumHistory);
    if (limit < 2) return fail('historyLimit must retain at least two revisions');
    if (!Array.isArray(session.history) || session.history.length === 0 || session.history.length > limit)
      return fail('Session history must contain 1..historyLimit snapshots');
    const history = arrayField(session.history, session.history.length, 'Session history').map(validatedSnapshot);
    const current = history[0], baseline = validatedSnapshot(session.baseline);
    if (session.currentId !== current.id) return fail('Session currentId must identify the newest retained snapshot');
    if (history.some((snapshot, index) => snapshot.revision !== current.revision - index))
      return fail('Session history must contain unique consecutive revisions, newest first');
    if (history.length !== Math.min(limit, current.revision + 1)) return fail('Session history is missing retained revisions');
    if (baseline.revision > current.revision) return fail('Session baseline cannot be newer than its current revision');
    const retainedBaseline = history.find(snapshot => snapshot.revision === baseline.revision);
    if (retainedBaseline && JSON.stringify(retainedBaseline) !== JSON.stringify(baseline)) return fail('Session baseline conflicts with its retained revision');
    const runtime = new WeightRuntime(current.model, { historyLimit: limit });
    runtime.active = current;
    runtime.pinned = retainedBaseline ?? baseline;
    runtime.revisions = Object.freeze(history);
    return runtime;
  }
}

export interface StrategicWeightPreset {
  id: string;
  name: string;
  description: string;
  /** Absolute W2 targets for all 38 named hidden neurons. */
  mappings: readonly Readonly<{ hidden: string; value: number }>[];
}

const authored = createDefaultModel();
function preset(id: string, name: string, description: string, overrides: Record<string, number>): StrategicWeightPreset {
  for (const hidden of Object.keys(overrides)) if (!authored.hiddenNames.includes(hidden)) fail(`Unknown preset neuron ${hidden}`);
  return Object.freeze({
    id, name, description,
    mappings: Object.freeze(authored.hiddenNames.map((hidden, index) => Object.freeze({ hidden, value: overrides[hidden] ?? authored.w2[index] }))),
  });
}

/** These are literal tensor profiles, not alternate hand-coded action selectors. */
export const STRATEGIC_PRESETS: readonly StrategicWeightPreset[] = Object.freeze([
  preset('authored', 'Authored balance', 'Restore the authored W2 output profile.', {}),
  preset('economy', 'Economic growth', 'Increase worker, resource, and farm output weights.', {
    worker_deficit_train: 13, worker_train_bias: 1.8, idle_labor_gather: 6, labor_coverage_gap: 4.5,
    food_shortage_gather: 6.5, wood_shortage_gather: 6, farm_coverage_gap: 6, attack_bias: -6,
  }),
  preset('rush', 'Early pressure', 'Increase recruitment and attack output weights.', {
    army_deficit_train: 4.5, army_composition_gap: 3.5, army_train_bias: 1.7,
    ready_army_attack: 8, visible_advantage_attack: 5, attack_bias: -2.5, production_coverage_gap: 5.5,
  }),
  preset('fortress', 'Hold the frontier', 'Increase local defense and counterattack output weights.', {
    visible_threat_defend: 15, ready_army_defend: 3.5, visible_threat_defenses: 11,
    defense_coverage_gap: 3.5, defense_bias: 0.3, home_threat_attack: -10,
  }),
  preset('technology', 'Technology race', 'Increase age advancement and research output weights.', {
    surplus_advance: 8, advance_bias: 5, immature_economy_advance: -2.5,
    surplus_research: 5, research_bias: 1.3, gold_shortage_gather: 4.5,
  }),
  preset('scouting', 'Frontier intelligence', 'Increase scouting and visible-army advantage output weights.', {
    scouting_coverage_gap: 4.5, scout_bias: 0.6, visible_advantage_attack: 5,
    order_disruption_cost: -6, home_threat_attack: -7,
  }),
]);

/** Each profile sets all W2 outputs; previously edited W1, b1, b2, and temperature remain in effect. */
export function presetEdits(model: PolicyModel, id: string): WeightEdit[] {
  const valid = validateWeightModel(model), selected = STRATEGIC_PRESETS.find(candidate => candidate.id === id);
  if (!selected) return fail(`Unknown strategic preset ${id}`);
  return selected.mappings.map(({ hidden, value }) => {
    const row = valid.hiddenNames.indexOf(hidden);
    if (row < 0) return fail(`The preset requires the authored hidden-neuron label ${hidden}`);
    return { tensor: 'w2', row, value };
  });
}

export interface WeightProposal {
  id: string;
  label: string;
  description: string;
  edits: WeightEdit[];
}

/** Small, bounded literal W2 proposals suitable for Laya's compact candidate head. */
export function getWeightProposals(model: PolicyModel): WeightProposal[] {
  const valid = validateWeightModel(model);
  const definitions: { id: string; label: string; description: string; changes: [string, number][] }[] = [
    { id: 'economy', label: 'Favor economy', description: 'Workers and idle labor.', changes: [['worker_deficit_train', 0.75], ['idle_labor_gather', 0.5]] },
    { id: 'pressure', label: 'Favor pressure', description: 'Attack bias and readiness.', changes: [['attack_bias', 0.8], ['ready_army_attack', 0.8]] },
    { id: 'defense', label: 'Favor defense', description: 'Defense under threat.', changes: [['visible_threat_defend', 1], ['home_threat_attack', -0.5]] },
    { id: 'technology', label: 'Favor technology', description: 'Age and research surplus.', changes: [['surplus_advance', 0.7], ['surplus_research', 0.7]] },
  ];
  const proposals: WeightProposal[] = [];
  for (const definition of definitions) {
    const edits: WeightEdit[] = [];
    for (const [hidden, delta] of definition.changes) {
      const row = valid.hiddenNames.indexOf(hidden);
      if (row < 0) return fail(`Weight coaching requires the authored hidden-neuron label ${hidden}`);
      const bounded = Math.max(WEIGHT_LIMITS.minimum, Math.min(WEIGHT_LIMITS.maximum, valid.w2[row] + delta));
      const value = Math.round(bounded * 10_000) / 10_000;
      if (value !== valid.w2[row]) edits.push({ tensor: 'w2', row, value });
    }
    if (edits.length) proposals.push({ id: definition.id, label: definition.label, description: definition.description, edits });
  }
  proposals.push({ id: 'keep', label: 'Keep current weights', description: 'Keep current weights.', edits: [] });
  return proposals;
}
