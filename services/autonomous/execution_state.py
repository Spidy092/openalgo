"""Durable idempotent execution lifecycle for autonomous Analyzer orders.

The sandbox order manager generates a fresh order id for every call, so an
autonomous caller must prevent duplicate calls before entering the sandbox.
This module provides that barrier with a small SQLite state store and a strict
state machine.  It contains no broker or live-order dependency.

The critical boundary is ``SUBMITTING``: it is persisted *before* calling the
Analyzer executor.  If the process dies after that write, the next attempt for
the same candidate is not retried.  It is moved to
``RECONCILIATION_REQUIRED`` so a later reconciliation pass can compare durable
state with the Analyzer orderbook/positions.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path


_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class ExecutionState(StrEnum):
    PROPOSED = "proposed"
    RISK_APPROVED = "risk_approved"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    EXIT_PENDING = "exit_pending"
    CLOSED = "closed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    ERROR = "error"
    RECONCILIATION_REQUIRED = "reconciliation_required"


TERMINAL_EXECUTION_STATES = frozenset(
    {
        ExecutionState.CLOSED,
        ExecutionState.REJECTED,
        ExecutionState.CANCELLED,
        ExecutionState.ERROR,
    }
)


_ALLOWED_TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.PROPOSED: frozenset(
        {
            ExecutionState.RISK_APPROVED,
            ExecutionState.REJECTED,
            ExecutionState.ERROR,
        }
    ),
    ExecutionState.RISK_APPROVED: frozenset(
        {
            ExecutionState.SUBMITTING,
            ExecutionState.REJECTED,
            ExecutionState.ERROR,
        }
    ),
    ExecutionState.SUBMITTING: frozenset(
        {
            ExecutionState.SUBMITTED,
            ExecutionState.RECONCILIATION_REQUIRED,
        }
    ),
    ExecutionState.SUBMITTED: frozenset(
        {
            ExecutionState.ACKNOWLEDGED,
            ExecutionState.CANCELLED,
            ExecutionState.ERROR,
            ExecutionState.RECONCILIATION_REQUIRED,
        }
    ),
    ExecutionState.ACKNOWLEDGED: frozenset(
        {
            ExecutionState.PARTIALLY_FILLED,
            ExecutionState.FILLED,
            ExecutionState.CANCELLED,
            ExecutionState.REJECTED,
            ExecutionState.ERROR,
            ExecutionState.RECONCILIATION_REQUIRED,
        }
    ),
    ExecutionState.PARTIALLY_FILLED: frozenset(
        {
            ExecutionState.FILLED,
            ExecutionState.CANCELLED,
            ExecutionState.EXIT_PENDING,
            ExecutionState.ERROR,
            ExecutionState.RECONCILIATION_REQUIRED,
        }
    ),
    ExecutionState.FILLED: frozenset(
        {
            ExecutionState.EXIT_PENDING,
            ExecutionState.CLOSED,
            ExecutionState.ERROR,
            ExecutionState.RECONCILIATION_REQUIRED,
        }
    ),
    ExecutionState.EXIT_PENDING: frozenset(
        {
            ExecutionState.CLOSED,
            ExecutionState.CANCELLED,
            ExecutionState.ERROR,
            ExecutionState.RECONCILIATION_REQUIRED,
        }
    ),
    ExecutionState.RECONCILIATION_REQUIRED: frozenset(
        {
            ExecutionState.SUBMITTED,
            ExecutionState.ACKNOWLEDGED,
            ExecutionState.PARTIALLY_FILLED,
            ExecutionState.FILLED,
            ExecutionState.EXIT_PENDING,
            ExecutionState.CLOSED,
            ExecutionState.REJECTED,
            ExecutionState.CANCELLED,
            ExecutionState.ERROR,
        }
    ),
    ExecutionState.CLOSED: frozenset(),
    ExecutionState.REJECTED: frozenset(),
    ExecutionState.CANCELLED: frozenset(),
    ExecutionState.ERROR: frozenset(),
}


class ExecutionStateError(RuntimeError):
    """Base error for durable autonomous execution state."""


class ExecutionNotFound(ExecutionStateError):
    pass


class InvalidExecutionTransition(ExecutionStateError):
    def __init__(self, execution_key: str, current: ExecutionState, target: ExecutionState) -> None:
        self.execution_key = execution_key
        self.current = current
        self.target = target
        super().__init__(
            f"invalid execution transition for {execution_key}: {current.value} -> {target.value}"
        )


class CandidateIdentityConflict(ExecutionStateError):
    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        super().__init__(
            f"candidate_id {candidate_id!r} is already bound to a different fingerprint"
        )


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    execution_key: str
    candidate_id: str
    candidate_fingerprint: str
    state: ExecutionState
    analyzer_order_id: str | None
    detail: str | None
    created_at: datetime
    updated_at: datetime
    version: int


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    sequence: int
    execution_key: str
    occurred_at: datetime
    from_state: ExecutionState | None
    to_state: ExecutionState
    analyzer_order_id: str | None
    detail: str | None


class DuplicateExecution(ExecutionStateError):
    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record
        super().__init__(
            f"candidate already has durable execution state {record.state.value} "
            f"({record.execution_key})"
        )


class ExecutionNeedsReconciliation(ExecutionStateError):
    def __init__(self, record: ExecutionRecord) -> None:
        self.record = record
        super().__init__(
            f"execution {record.execution_key} requires reconciliation from "
            f"state {record.state.value}"
        )


def deterministic_execution_key(candidate_fingerprint: str) -> str:
    """Return the stable v1 idempotency key for one research candidate."""

    normalized = str(candidate_fingerprint).strip().lower()
    if not _FINGERPRINT_RE.fullmatch(normalized):
        raise ValueError("candidate_fingerprint must be a 64-character lowercase SHA-256 hex digest")
    return f"openalgo-auto-v1:{normalized}"


def _timestamp(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("execution timestamps must be timezone-aware")
    return current.astimezone(timezone.utc)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stored execution timestamp is timezone-naive")
    return parsed.astimezone(timezone.utc)


class SqliteExecutionStateStore:
    """Atomic, append-audited execution state store using Python stdlib SQLite."""

    def __init__(self, path: str | Path) -> None:
        raw = str(path).strip()
        if not raw:
            raise ValueError("execution state database path is required")
        if raw == ":memory:":
            raise ValueError("in-memory execution state is forbidden for autonomous execution")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=10.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS autonomous_executions (
                    execution_key TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL UNIQUE,
                    candidate_fingerprint TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL,
                    analyzer_order_id TEXT,
                    detail TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 1)
                );

                CREATE TABLE IF NOT EXISTS autonomous_execution_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_key TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    analyzer_order_id TEXT,
                    detail TEXT,
                    UNIQUE (execution_key, sequence),
                    FOREIGN KEY (execution_key)
                        REFERENCES autonomous_executions(execution_key)
                        ON DELETE RESTRICT
                );

                CREATE INDEX IF NOT EXISTS idx_autonomous_execution_events_key
                    ON autonomous_execution_events(execution_key, sequence);
                """
            )
        finally:
            connection.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> ExecutionRecord:
        return ExecutionRecord(
            execution_key=str(row["execution_key"]),
            candidate_id=str(row["candidate_id"]),
            candidate_fingerprint=str(row["candidate_fingerprint"]),
            state=ExecutionState(str(row["state"])),
            analyzer_order_id=(
                None if row["analyzer_order_id"] is None else str(row["analyzer_order_id"])
            ),
            detail=None if row["detail"] is None else str(row["detail"]),
            created_at=_parse_timestamp(str(row["created_at"])),
            updated_at=_parse_timestamp(str(row["updated_at"])),
            version=int(row["version"]),
        )

    @staticmethod
    def _event(row: sqlite3.Row) -> ExecutionEvent:
        return ExecutionEvent(
            sequence=int(row["sequence"]),
            execution_key=str(row["execution_key"]),
            occurred_at=_parse_timestamp(str(row["occurred_at"])),
            from_state=(
                None if row["from_state"] is None else ExecutionState(str(row["from_state"]))
            ),
            to_state=ExecutionState(str(row["to_state"])),
            analyzer_order_id=(
                None if row["analyzer_order_id"] is None else str(row["analyzer_order_id"])
            ),
            detail=None if row["detail"] is None else str(row["detail"]),
        )

    def get(self, execution_key: str) -> ExecutionRecord | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM autonomous_executions WHERE execution_key = ?",
                (execution_key,),
            ).fetchone()
            return None if row is None else self._record(row)
        finally:
            connection.close()

    def get_by_fingerprint(self, candidate_fingerprint: str) -> ExecutionRecord | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM autonomous_executions WHERE candidate_fingerprint = ?",
                (candidate_fingerprint,),
            ).fetchone()
            return None if row is None else self._record(row)
        finally:
            connection.close()

    def events(self, execution_key: str) -> tuple[ExecutionEvent, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT sequence, execution_key, occurred_at, from_state, to_state,
                       analyzer_order_id, detail
                FROM autonomous_execution_events
                WHERE execution_key = ?
                ORDER BY sequence ASC
                """,
                (execution_key,),
            ).fetchall()
            return tuple(self._event(row) for row in rows)
        finally:
            connection.close()

    def create_proposed(
        self,
        *,
        candidate_id: str,
        candidate_fingerprint: str,
        occurred_at: datetime,
    ) -> tuple[ExecutionRecord, bool]:
        candidate_id = str(candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id is required")
        execution_key = deterministic_execution_key(candidate_fingerprint)
        when = _timestamp(occurred_at).isoformat()

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM autonomous_executions WHERE candidate_fingerprint = ?",
                (candidate_fingerprint,),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return self._record(existing), False

            identity = connection.execute(
                "SELECT * FROM autonomous_executions WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if identity is not None:
                connection.rollback()
                raise CandidateIdentityConflict(candidate_id)

            connection.execute(
                """
                INSERT INTO autonomous_executions (
                    execution_key, candidate_id, candidate_fingerprint, state,
                    analyzer_order_id, detail, created_at, updated_at, version
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, 1)
                """,
                (
                    execution_key,
                    candidate_id,
                    candidate_fingerprint,
                    ExecutionState.PROPOSED.value,
                    when,
                    when,
                ),
            )
            connection.execute(
                """
                INSERT INTO autonomous_execution_events (
                    execution_key, sequence, occurred_at, from_state, to_state,
                    analyzer_order_id, detail
                ) VALUES (?, 1, ?, NULL, ?, NULL, NULL)
                """,
                (execution_key, when, ExecutionState.PROPOSED.value),
            )
            row = connection.execute(
                "SELECT * FROM autonomous_executions WHERE execution_key = ?",
                (execution_key,),
            ).fetchone()
            connection.commit()
            assert row is not None
            return self._record(row), True
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def transition(
        self,
        execution_key: str,
        target: ExecutionState,
        *,
        occurred_at: datetime,
        analyzer_order_id: str | None = None,
        detail: str | None = None,
    ) -> ExecutionRecord:
        target = ExecutionState(target)
        clean_order_id = None
        if analyzer_order_id is not None:
            clean_order_id = str(analyzer_order_id).strip()
            if not clean_order_id:
                raise ValueError("analyzer_order_id cannot be empty")
        clean_detail = None if detail is None else str(detail).strip() or None
        when = _timestamp(occurred_at).isoformat()

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM autonomous_executions WHERE execution_key = ?",
                (execution_key,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise ExecutionNotFound(execution_key)

            current = ExecutionState(str(row["state"]))
            if target not in _ALLOWED_TRANSITIONS[current]:
                connection.rollback()
                raise InvalidExecutionTransition(execution_key, current, target)

            existing_order_id = (
                None if row["analyzer_order_id"] is None else str(row["analyzer_order_id"])
            )
            if (
                clean_order_id is not None
                and existing_order_id is not None
                and clean_order_id != existing_order_id
            ):
                connection.rollback()
                raise ExecutionStateError(
                    f"execution {execution_key} is already bound to Analyzer order "
                    f"{existing_order_id!r}"
                )
            effective_order_id = clean_order_id or existing_order_id
            if target is ExecutionState.SUBMITTED and effective_order_id is None:
                connection.rollback()
                raise ValueError("SUBMITTED state requires analyzer_order_id")

            next_version = int(row["version"]) + 1
            connection.execute(
                """
                UPDATE autonomous_executions
                SET state = ?, analyzer_order_id = ?, detail = ?, updated_at = ?, version = ?
                WHERE execution_key = ?
                """,
                (
                    target.value,
                    effective_order_id,
                    clean_detail,
                    when,
                    next_version,
                    execution_key,
                ),
            )
            connection.execute(
                """
                INSERT INTO autonomous_execution_events (
                    execution_key, sequence, occurred_at, from_state, to_state,
                    analyzer_order_id, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_key,
                    next_version,
                    when,
                    current.value,
                    target.value,
                    effective_order_id,
                    clean_detail,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM autonomous_executions WHERE execution_key = ?",
                (execution_key,),
            ).fetchone()
            connection.commit()
            assert updated is not None
            return self._record(updated)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()


Clock = Callable[[], datetime]


class ExecutionStateMachine:
    """Strict lifecycle facade around :class:`SqliteExecutionStateStore`."""

    def __init__(
        self,
        store: SqliteExecutionStateStore,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.store = store
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self) -> datetime:
        return _timestamp(self._clock())

    def propose(self, *, candidate_id: str, candidate_fingerprint: str) -> ExecutionRecord:
        record, created = self.store.create_proposed(
            candidate_id=candidate_id,
            candidate_fingerprint=candidate_fingerprint,
            occurred_at=self._now(),
        )
        if created:
            return record

        if record.candidate_id != str(candidate_id).strip():
            raise CandidateIdentityConflict(candidate_id)

        if record.state is ExecutionState.SUBMITTING:
            record = self.store.transition(
                record.execution_key,
                ExecutionState.RECONCILIATION_REQUIRED,
                occurred_at=self._now(),
                detail="ambiguous_restart_after_submission_started",
            )
            raise ExecutionNeedsReconciliation(record)
        if record.state is ExecutionState.RECONCILIATION_REQUIRED:
            raise ExecutionNeedsReconciliation(record)
        raise DuplicateExecution(record)

    def _transition(
        self,
        execution_key: str,
        target: ExecutionState,
        *,
        analyzer_order_id: str | None = None,
        detail: str | None = None,
    ) -> ExecutionRecord:
        return self.store.transition(
            execution_key,
            target,
            occurred_at=self._now(),
            analyzer_order_id=analyzer_order_id,
            detail=detail,
        )

    def risk_approved(self, execution_key: str) -> ExecutionRecord:
        return self._transition(execution_key, ExecutionState.RISK_APPROVED)

    def reject(self, execution_key: str, *, detail: str | None = None) -> ExecutionRecord:
        return self._transition(execution_key, ExecutionState.REJECTED, detail=detail)

    def begin_submission(self, execution_key: str) -> ExecutionRecord:
        return self._transition(execution_key, ExecutionState.SUBMITTING)

    def submitted(self, execution_key: str, analyzer_order_id: str) -> ExecutionRecord:
        return self._transition(
            execution_key,
            ExecutionState.SUBMITTED,
            analyzer_order_id=analyzer_order_id,
        )

    def acknowledged(self, execution_key: str) -> ExecutionRecord:
        return self._transition(execution_key, ExecutionState.ACKNOWLEDGED)

    def partially_filled(self, execution_key: str, *, detail: str | None = None) -> ExecutionRecord:
        return self._transition(
            execution_key,
            ExecutionState.PARTIALLY_FILLED,
            detail=detail,
        )

    def filled(self, execution_key: str, *, detail: str | None = None) -> ExecutionRecord:
        return self._transition(execution_key, ExecutionState.FILLED, detail=detail)

    def exit_pending(self, execution_key: str, *, detail: str | None = None) -> ExecutionRecord:
        return self._transition(execution_key, ExecutionState.EXIT_PENDING, detail=detail)

    def closed(self, execution_key: str, *, detail: str | None = None) -> ExecutionRecord:
        return self._transition(execution_key, ExecutionState.CLOSED, detail=detail)

    def cancelled(self, execution_key: str, *, detail: str | None = None) -> ExecutionRecord:
        return self._transition(execution_key, ExecutionState.CANCELLED, detail=detail)

    def error(self, execution_key: str, *, detail: str | None = None) -> ExecutionRecord:
        return self._transition(execution_key, ExecutionState.ERROR, detail=detail)

    def reconciliation_required(
        self,
        execution_key: str,
        *,
        detail: str | None = None,
    ) -> ExecutionRecord:
        return self._transition(
            execution_key,
            ExecutionState.RECONCILIATION_REQUIRED,
            detail=detail,
        )
