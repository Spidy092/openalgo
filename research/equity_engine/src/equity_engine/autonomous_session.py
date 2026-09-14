"""Deterministic autonomous session runner for Analyzer/shadow execution.

The LLM/research layer may rank or propose candidates. This module owns the
mechanical loop and cannot be overridden by prompt text. It adds cumulative
session controls, duplicate suppression, an append-only audit journal and a
health gate around :class:`AutonomousOrchestrator`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Callable, Sequence

from equity_engine.autonomous_orchestrator import (
    AutonomousOrchestrator,
    Decision,
    ExecutionRejected,
    OrchestrationResult,
    TradeCandidate,
)


class SessionOutcome(StrEnum):
    EXECUTED = "executed"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SessionLimits:
    max_orders: int
    max_gross_notional: Decimal
    max_orders_per_symbol: int

    def __post_init__(self) -> None:
        if self.max_orders <= 0:
            raise ValueError("max_orders must be positive")
        if self.max_gross_notional <= 0:
            raise ValueError("max_gross_notional must be positive")
        if self.max_orders_per_symbol <= 0:
            raise ValueError("max_orders_per_symbol must be positive")


@dataclass(frozen=True, slots=True)
class JournalEntry:
    timestamp: str
    candidate_id: str
    candidate_fingerprint: str
    symbol: str
    outcome: SessionOutcome
    execution_id: str | None
    reasons: tuple[str, ...]
    notional: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        payload["reasons"] = list(self.reasons)
        return payload


class JsonlExecutionJournal:
    """Append-only JSONL journal with fail-closed parsing and fsync on writes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._by_fingerprint: dict[str, JournalEntry] = {}
        self._by_candidate_id: dict[str, JournalEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    payload = json.loads(raw)
                    entry = JournalEntry(
                        timestamp=str(payload["timestamp"]),
                        candidate_id=str(payload["candidate_id"]),
                        candidate_fingerprint=str(payload["candidate_fingerprint"]),
                        symbol=str(payload["symbol"]),
                        outcome=SessionOutcome(str(payload["outcome"])),
                        execution_id=(
                            None if payload.get("execution_id") is None else str(payload["execution_id"])
                        ),
                        reasons=tuple(str(item) for item in payload.get("reasons", [])),
                        notional=str(payload["notional"]),
                    )
                except Exception as exc:
                    raise ValueError(
                        f"invalid autonomous journal at line {line_number}: {exc}"
                    ) from exc
                self._by_fingerprint[entry.candidate_fingerprint] = entry
                self._by_candidate_id[entry.candidate_id] = entry

    def entry_for_fingerprint(self, fingerprint: str) -> JournalEntry | None:
        return self._by_fingerprint.get(fingerprint)

    def entry_for_candidate_id(self, candidate_id: str) -> JournalEntry | None:
        return self._by_candidate_id.get(candidate_id)

    def append(self, entry: JournalEntry) -> None:
        encoded = json.dumps(entry.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._by_fingerprint[entry.candidate_fingerprint] = entry
        self._by_candidate_id[entry.candidate_id] = entry


HealthCheck = Callable[[], bool]


class AutonomousSession:
    """Run a finite candidate sequence once, never retrying failed execution."""

    def __init__(
        self,
        *,
        orchestrator: AutonomousOrchestrator,
        journal: JsonlExecutionJournal,
        limits: SessionLimits,
        health_check: HealthCheck | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._journal = journal
        self._limits = limits
        self._health_check = health_check or (lambda: True)
        self._orders = 0
        self._gross_notional = Decimal("0")
        self._orders_by_symbol: dict[str, int] = {}

    def _entry(
        self,
        candidate: TradeCandidate,
        outcome: SessionOutcome,
        *,
        execution_id: str | None = None,
        reasons: tuple[str, ...] = (),
    ) -> JournalEntry:
        return JournalEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint,
            symbol=candidate.symbol,
            outcome=outcome,
            execution_id=execution_id,
            reasons=reasons,
            notional=str(candidate.notional),
        )

    def _session_limit_reasons(self, candidate: TradeCandidate) -> tuple[str, ...]:
        reasons: list[str] = []
        if self._orders >= self._limits.max_orders:
            reasons.append("session_max_orders")
        if self._gross_notional + candidate.notional > self._limits.max_gross_notional:
            reasons.append("session_max_gross_notional")
        if self._orders_by_symbol.get(candidate.symbol, 0) >= self._limits.max_orders_per_symbol:
            reasons.append("session_max_orders_per_symbol")
        return tuple(reasons)

    def run(self, candidates: Sequence[TradeCandidate]) -> tuple[JournalEntry, ...]:
        ids = [candidate.candidate_id for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id values must be unique within one session input")

        results: list[JournalEntry] = []
        for candidate in candidates:
            prior = self._journal.entry_for_fingerprint(candidate.fingerprint)
            if prior is not None:
                results.append(
                    self._entry(
                        candidate,
                        SessionOutcome.DUPLICATE,
                        execution_id=prior.execution_id,
                        reasons=("candidate_fingerprint_already_journaled",),
                    )
                )
                continue

            prior_id = self._journal.entry_for_candidate_id(candidate.candidate_id)
            if prior_id is not None and prior_id.candidate_fingerprint != candidate.fingerprint:
                entry = self._entry(
                    candidate,
                    SessionOutcome.STOPPED,
                    reasons=("candidate_id_reused_with_different_fingerprint",),
                )
                self._journal.append(entry)
                results.append(entry)
                break

            if not self._health_check():
                entry = self._entry(
                    candidate,
                    SessionOutcome.STOPPED,
                    reasons=("health_check_failed",),
                )
                self._journal.append(entry)
                results.append(entry)
                break

            limit_reasons = self._session_limit_reasons(candidate)
            if limit_reasons:
                entry = self._entry(candidate, SessionOutcome.REJECTED, reasons=limit_reasons)
                self._journal.append(entry)
                results.append(entry)
                continue

            try:
                result: OrchestrationResult = self._orchestrator.process(candidate)
            except ExecutionRejected as exc:
                entry = self._entry(
                    candidate,
                    SessionOutcome.REJECTED,
                    reasons=exc.reasons,
                )
                self._journal.append(entry)
                results.append(entry)
                continue
            except Exception as exc:
                entry = self._entry(
                    candidate,
                    SessionOutcome.ERROR,
                    reasons=(f"execution_error:{type(exc).__name__}:{exc}",),
                )
                self._journal.append(entry)
                results.append(entry)
                break

            if result.risk.decision is Decision.REJECTED:
                entry = self._entry(
                    candidate,
                    SessionOutcome.REJECTED,
                    reasons=result.risk.reasons,
                )
            else:
                entry = self._entry(
                    candidate,
                    SessionOutcome.EXECUTED,
                    execution_id=result.execution_id,
                )
                self._orders += 1
                self._gross_notional += candidate.notional
                self._orders_by_symbol[candidate.symbol] = (
                    self._orders_by_symbol.get(candidate.symbol, 0) + 1
                )

            self._journal.append(entry)
            results.append(entry)

        return tuple(results)
