"""Durable data-role guards for a trusted, cooperative evaluation harness.

Hashes bind finite JSON rows, not native tensor semantics or independently
authenticated provenance. Only declared/observed families and ancestry are
covered. Arbitrary scorers can ignore the supplied data. This module neither
attests evaluator execution nor supplies a durable once-only scoring guarantee.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable, Iterator
import uuid


_SCHEMA = "wsp-data-role-registry/1"
_FLAGS = {"provenance_authenticated": False,
          "semantic_independence_verified": False,
          "scientific_freshness_verified": False,
          "evaluator_execution_attested": False}
_ACTIVE = threading.local()


def _finite(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [_finite(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _finite(item) for key, item in value.items()}
    raise ValueError("invalid_finite_json")


def _bytes(value: Any) -> bytes:
    try:
        return json.dumps(_finite(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
        raise ValueError("invalid_finite_json") from None


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _capture(examples: Any) -> bytes:
    if type(examples) not in (list, tuple) or not examples:
        raise ValueError("nonempty_json_batch_required")
    return _bytes(examples)


def _name(value: Any) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("invalid_identity")
    _bytes(value)
    return value


def _bound(document: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in document.items() if key != "decision_sha256"}
    body["decision_sha256"] = _sha(_bytes(body))
    return json.loads(_bytes(body))


class DataRoleError(ValueError):
    """A controlled error with copied, content-bound evidence and no raw data."""

    def __init__(self, decision: dict[str, Any]):
        super().__init__("Data-role boundary rejected the operation")
        self._decision_bytes = _bytes(_bound(decision))

    @property
    def decision(self) -> dict[str, Any]:
        return json.loads(self._decision_bytes)


@dataclass(frozen=True, slots=True, init=False)
class RegisteredBatch:
    """An immutable JSON capture; a registry must still verify this handle."""

    _payload_bytes: bytes = field(repr=False)
    _manifest_bytes: bytes = field(repr=False)

    @classmethod
    def _from_bytes(cls, payload: bytes, manifest: bytes) -> RegisteredBatch:
        result = object.__new__(cls)
        object.__setattr__(result, "_payload_bytes", payload)
        object.__setattr__(result, "_manifest_bytes", manifest)
        return result

    def materialize(self) -> list[Any]:
        return json.loads(self._payload_bytes)

    def to_dict(self) -> dict[str, Any]:
        result = json.loads(self._manifest_bytes)
        result["manifest_sha256"] = self.manifest_sha256
        return result

    @property
    def registry_id(self) -> str:
        return json.loads(self._manifest_bytes)["registry_id"]

    @property
    def manifest_sha256(self) -> str:
        return _sha(self._manifest_bytes)


class DataRoleRegistry:
    """SQLite-backed monotonic consumption of observed JSON content and lineage.

    Use one shared database under a trusted harness; copies, undeclared
    transformations, external observations and opaque scorer inputs are outside
    this boundary. Failed scoring does not invalidate or reserve an evaluation.
    """

    def __init__(self, path: str | Path):
        self._registry_id: str | None = None
        decision = self._decision("initialize")
        try:
            if str(path) == ":memory:":
                raise ValueError("durable_database_required")
            self._path = str(Path(path).resolve())
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        except (OSError, TypeError, ValueError):
            decision.update(status="rejected", reason="durable_database_required")
            raise DataRoleError(decision) from None
        with self._transaction(decision, initialize=True) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS sources (source_id TEXT PRIMARY KEY, "
                               "manifest_sha256 TEXT NOT NULL, manifest_json BLOB NOT NULL, payload_json BLOB NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS consumed (kind TEXT NOT NULL, key TEXT NOT NULL, "
                               "reason_sha256 TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(kind, key))")
            connection.execute("CREATE TABLE IF NOT EXISTS decisions (decision_sha256 TEXT PRIMARY KEY, "
                               "decision_json BLOB NOT NULL)")
            metadata = dict(connection.execute("SELECT key, value FROM meta"))
            if not metadata:
                metadata = {"schema": _SCHEMA, "registry_id": str(uuid.uuid4()), "revision": "0"}
                connection.executemany("INSERT INTO meta VALUES (?, ?)", metadata.items())
            if metadata.get("schema") != _SCHEMA or not metadata.get("registry_id"):
                self._reject(connection, decision, "registry_invalid")
            self._registry_id = metadata["registry_id"]
            self._snapshot(connection)

    @property
    def registry_id(self) -> str:
        return self._registry_id

    def _decision(self, operation: str) -> dict[str, Any]:
        return {"schema": "wsp-data-role-decision/1", "operation": operation,
                "status": "rejected", "verdict": "unverified", "reason": "not_resolved",
                "registry_id": self._registry_id, "manifest_sha256": None,
                "content_sha256": None, "row_sha256": [], "consumed_matches": [],
                "registry_revision_before": None, "registry_revision_after": None,
                "registry_state_sha256_before": None, "registry_state_sha256_after": None,
                "result_sha256": None, "scoring_started": False, **_FLAGS}

    @contextmanager
    def _transaction(self, decision: dict[str, Any], *, initialize: bool = False) -> Iterator[sqlite3.Connection]:
        paths = getattr(_ACTIVE, "paths", set())
        identities = getattr(_ACTIVE, "identities", set())
        if self._path in paths or self._registry_id in identities:
            raise DataRoleError({**decision, "reason": "callback_reentry"})
        connection = None
        try:
            # ponytail: SQLite serializes writers across the scorer. Use a shared
            # database with finer locking if measured scoring throughput requires it.
            connection = sqlite3.connect(self._path, timeout=2.0, isolation_level=None)
            connection.execute("BEGIN IMMEDIATE")
            if not initialize:
                metadata = dict(connection.execute("SELECT key, value FROM meta"))
                if metadata.get("schema") != _SCHEMA or metadata.get("registry_id") != self._registry_id:
                    self._reject(connection, decision, "registry_identity_changed")
            yield connection
            if connection.in_transaction:
                connection.commit()
        except DataRoleError:
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise
        except (sqlite3.Error, OSError, ValueError):
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise DataRoleError({**decision, "status": "failed" if decision["scoring_started"] else "rejected",
                                 "verdict": "unverified", "reason": "registry_unavailable"}) from None
        finally:
            if connection is not None:
                connection.close()

    def _manifest(self, source_id: str, family_id: str, parents: tuple[str, ...],
                  payload: bytes, records: dict[str, RegisteredBatch]) -> dict[str, Any]:
        ancestors = set(parents)
        for parent in parents:
            if parent not in records:
                raise ValueError("unknown_parent")
            ancestors.update(records[parent].to_dict()["ancestor_source_ids"])
        if source_id in ancestors:
            raise ValueError("cyclic_ancestry")
        return {"schema": "wsp-data-batch/1", "registry_id": self.registry_id,
                "source_id": _name(source_id), "family_id": _name(family_id),
                "parents": list(parents), "ancestor_source_ids": sorted(ancestors),
                "parent_manifest_sha256": {parent: records[parent].manifest_sha256 for parent in parents},
                "content_encoding": "canonical-finite-json/1", "content_sha256": _sha(payload),
                "row_sha256": [_sha(_bytes(row)) for row in json.loads(payload)], **_FLAGS}

    def _snapshot(self, connection: sqlite3.Connection) -> tuple[dict[str, RegisteredBatch], dict[tuple[str, str], tuple[str, int]], int, str]:
        try:
            revision = int(connection.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()[0])
            if revision < 0:
                raise ValueError
            records = {}
            for source_id, digest, manifest, payload in connection.execute("SELECT * FROM sources ORDER BY source_id"):
                record = RegisteredBatch._from_bytes(bytes(payload), bytes(manifest))
                if record.manifest_sha256 != digest or _capture(record.materialize()) != record._payload_bytes:
                    raise ValueError
                records[source_id] = record
            for source_id, record in records.items():
                manifest = record.to_dict()
                parents = tuple(manifest["parents"])
                if parents != tuple(sorted(set(parents))):
                    raise ValueError
                expected = self._manifest(source_id, manifest["family_id"], parents, record._payload_bytes, records)
                if record._manifest_bytes != _bytes(expected):
                    raise ValueError
            consumed = {}
            for kind, key, reason, when in connection.execute("SELECT * FROM consumed ORDER BY kind, key"):
                if kind not in ("content", "family", "source") or not 0 < when <= revision:
                    raise ValueError
                consumed[kind, key] = reason, when
            state = {"schema": "wsp-data-role-state/1", "registry_id": self.registry_id, "revision": revision,
                     "sources": [[key, records[key].manifest_sha256] for key in sorted(records)],
                     "consumed": [[kind, key, *consumed[kind, key]] for kind, key in sorted(consumed)]}
            return records, consumed, revision, _sha(_bytes(state))
        except (ValueError, TypeError, KeyError, IndexError, RecursionError, UnicodeError):
            raise ValueError("registry_invalid") from None

    def _before(self, decision: dict[str, Any], snapshot: tuple) -> None:
        decision.update(registry_id=self.registry_id, registry_revision_before=snapshot[2],
                        registry_revision_after=snapshot[2], registry_state_sha256_before=snapshot[3],
                        registry_state_sha256_after=snapshot[3])

    def _details(self, decision: dict[str, Any], payload: bytes, batch: RegisteredBatch | None = None) -> None:
        decision.update(content_sha256=_sha(payload), row_sha256=[_sha(_bytes(row)) for row in json.loads(payload)],
                        manifest_sha256=batch.manifest_sha256 if batch is not None else None)

    def _known(self, batch: RegisteredBatch, records: dict[str, RegisteredBatch]) -> RegisteredBatch:
        if type(batch) is not RegisteredBatch:
            raise ValueError("registered_batch_required")
        try:
            if batch.registry_id != self.registry_id:
                raise ValueError("foreign_registry")
            stored = records.get(batch.to_dict()["source_id"])
            if stored is None:
                raise ValueError("unknown_registration")
            if batch._manifest_bytes != stored._manifest_bytes or batch._payload_bytes != stored._payload_bytes:
                raise ValueError("registration_mismatch")
            return stored
        except (TypeError, KeyError, AttributeError, json.JSONDecodeError, UnicodeError):
            raise ValueError("invalid_registration") from None

    def _scope(self, batch: RegisteredBatch, records: dict[str, RegisteredBatch]) -> set[tuple[str, str]]:
        manifest = batch.to_dict()
        scope = set()
        for source_id in [manifest["source_id"], *manifest["ancestor_source_ids"]]:
            related = records[source_id].to_dict()
            scope.update({("source", source_id), ("family", related["family_id"])})
            scope.update(("content", row) for row in related["row_sha256"])
        return scope

    def _consume_keys(self, connection: sqlite3.Connection, records: dict[str, RegisteredBatch],
                      consumed: dict[tuple[str, str], tuple[str, int]], seeds: set[tuple[str, str]],
                      reason: str, revision: int, *, registration: bool = False) -> None:
        keys = set(consumed) | seeds
        scopes = [self._scope(record, records) for record in records.values()]
        # ponytail: repeated scans of observed scopes; index the lineage graph
        # when measured registry size makes these conservative scans expensive.
        while True:
            added = set().union(*(scope for scope in scopes if scope & keys)) if scopes else set()
            if added <= keys:
                break
            keys.update(added)
        new = keys - set(consumed)
        if new or registration:
            revision += 1
            connection.execute("UPDATE meta SET value = ? WHERE key = 'revision'", (str(revision),))
            connection.executemany("INSERT INTO consumed VALUES (?, ?, ?, ?)",
                                   ((kind, key, _sha(_bytes(reason)), revision) for kind, key in sorted(new)))

    def _finish(self, connection: sqlite3.Connection, decision: dict[str, Any]) -> dict[str, Any]:
        bound = _bound(decision)
        connection.execute("INSERT OR IGNORE INTO decisions VALUES (?, ?)",
                           (bound["decision_sha256"], _bytes(bound)))
        connection.commit()
        return json.loads(_bytes(bound))

    def _reject(self, connection: sqlite3.Connection, decision: dict[str, Any], reason: str,
                *, verdict: str = "unverified", status: str = "rejected") -> None:
        decision.update(reason=reason, verdict=verdict, status=status)
        raise DataRoleError(self._finish(connection, decision)) from None

    def register_batch(self, examples: Any, *, source_id: str, family_id: str,
                       parents: tuple[str | RegisteredBatch, ...] = ()) -> RegisteredBatch:
        """Capture rows and declare immutable observed lineage under the harness."""
        decision = self._decision("register")
        with self._transaction(decision) as connection:
            try:
                snapshot = self._snapshot(connection)
                records, consumed, revision, _ = snapshot
                self._before(decision, snapshot)
                payload = _capture(examples)
                self._details(decision, payload)
                _name(source_id)
                _name(family_id)
                if type(parents) not in (list, tuple):
                    raise ValueError("invalid_parents")
                parent_ids = []
                for parent in parents:
                    parent_ids.append(self._known(parent, records).to_dict()["source_id"]
                                      if type(parent) is RegisteredBatch else _name(parent))
                manifest = _bytes(self._manifest(source_id, family_id, tuple(sorted(set(parent_ids))), payload, records))
                batch = RegisteredBatch._from_bytes(payload, manifest)
                existing = records.get(source_id)
                if existing is not None:
                    if existing._manifest_bytes != manifest or existing._payload_bytes != payload:
                        raise ValueError("conflicting_registration")
                    return existing
                connection.execute("INSERT INTO sources VALUES (?, ?, ?, ?)",
                                   (source_id, batch.manifest_sha256, manifest, payload))
                records[source_id] = batch
                # Registration can reveal that previously consumed content has
                # another family/ancestor mapping. Propagate before publishing it.
                self._consume_keys(connection, records, consumed, set(), "observed_alias_registration",
                                   revision, registration=True)
                connection.commit()
                return batch
            except ValueError as failure:
                self._reject(connection, decision, str(failure))

    def consume(self, examples: Any, *, reason: str) -> dict[str, Any]:
        """Commit development consumption before any caller exposes these rows."""
        decision = self._decision("consume")
        with self._transaction(decision) as connection:
            try:
                snapshot = self._snapshot(connection)
                records, consumed, revision, _ = snapshot
                self._before(decision, snapshot)
                _name(reason)
                batch = self._known(examples, records) if type(examples) is RegisteredBatch else None
                payload = batch._payload_bytes if batch is not None else _capture(examples)
                self._details(decision, payload, batch)
                seeds = {("content", row) for row in decision["row_sha256"]}
                self._consume_keys(connection, records, consumed, seeds, reason, revision)
                after = self._snapshot(connection)
                decision.update(status="succeeded", verdict="registered_cooperative" if batch is not None else "unverified",
                                reason="development_consumption_committed", registry_revision_after=after[2],
                                registry_state_sha256_after=after[3])
                return self._finish(connection, decision)
            except ValueError as failure:
                self._reject(connection, decision, str(failure))

    def score(self, batch: RegisteredBatch, scorer: Callable[[list[Any]], Any]) -> tuple[Any, dict[str, Any]]:
        """Resolve observed freshness and score while holding the writer lock."""
        decision = self._decision("score")
        with self._transaction(decision) as connection:
            try:
                snapshot = self._snapshot(connection)
                records, consumed, _, _ = snapshot
                self._before(decision, snapshot)
                registered = self._known(batch, records)
                self._details(decision, registered._payload_bytes, registered)
                if not callable(scorer):
                    raise ValueError("scorer_not_callable")
                matches = self._scope(registered, records) & set(consumed)
                decision["consumed_matches"] = [{"kind": kind, "key": key,
                    "reason_sha256": consumed[kind, key][0], "revision": consumed[kind, key][1]}
                    for kind, key in sorted(matches)]
                if matches:
                    self._reject(connection, decision, "observed_development_consumption", verdict="invalidated")
            except ValueError as failure:
                if isinstance(failure, DataRoleError):
                    raise
                self._reject(connection, decision, str(failure))
            paths = getattr(_ACTIVE, "paths", set())
            identities = getattr(_ACTIVE, "identities", set())
            _ACTIVE.paths, _ACTIVE.identities = paths, identities
            paths.add(self._path)
            identities.add(self.registry_id)
            decision["scoring_started"] = True
            try:
                try:
                    result_bytes = _bytes(scorer(registered.materialize()))
                except Exception:
                    self._reject(connection, decision, "scorer_failed", status="failed")
                except BaseException as interrupted:
                    decision.update(status="failed", verdict="unverified", reason="scorer_failed")
                    try:
                        interrupted.data_role_decision = None
                        interrupted.data_role_decision = self._finish(connection, decision)
                    except BaseException:
                        # A failed evidence commit must never replace cancellation.
                        pass
                    raise
                decision.update(status="succeeded", verdict="registered_cooperative", reason="observed_boundary_passed",
                                result_sha256=_sha(result_bytes))
                evidence = self._finish(connection, decision)
                return json.loads(result_bytes), evidence
            finally:
                paths.remove(self._path)
                identities.remove(self.registry_id)
