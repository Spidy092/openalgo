"""Fail-closed orchestration contract for autonomous research execution.

This module joins research output, deterministic risk approval and an execution
adapter without giving the research package a live-order path.  Version 1 only
permits SHADOW and ANALYZER modes.  A future live adapter must be implemented
outside this package and pass a separate capital-authorization boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Protocol


class ExecutionMode(StrEnum):
    SHADOW = "shadow"
    ANALYZER = "analyzer"
    LIVE = "live"


class Decision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class TradeCandidate:
    candidate_id: str
    symbol: str
    exchange: str
    strategy_id: str
    strategy_version: str
    side: str
    quantity: int
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    expected_edge_bps: Decimal
    confidence: Decimal
    valid_until: datetime
    dataset_fingerprint: str
    research_fingerprint: str

    def validate(self, *, now: datetime | None = None) -> None:
        current = now or datetime.now(timezone.utc)
        if self.valid_until.tzinfo is None:
            raise ValueError("valid_until must be timezone-aware")
        if self.valid_until <= current:
            raise ValueError("candidate is stale")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if min(self.entry_price, self.stop_price, self.target_price) <= 0:
            raise ValueError("prices must be positive")
        if not Decimal("0") <= self.confidence <= Decimal("1"):
            raise ValueError("confidence must be between 0 and 1")
        if self.expected_edge_bps <= 0:
            raise ValueError("expected edge must be positive")
        if not self.dataset_fingerprint or not self.research_fingerprint:
            raise ValueError("research provenance fingerprints are required")

    @property
    def notional(self) -> Decimal:
        return self.entry_price * self.quantity

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["entry_price"] = str(self.entry_price)
        payload["stop_price"] = str(self.stop_price)
        payload["target_price"] = str(self.target_price)
        payload["expected_edge_bps"] = str(self.expected_edge_bps)
        payload["confidence"] = str(self.confidence)
        payload["valid_until"] = self.valid_until.astimezone(timezone.utc).isoformat()
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_order_notional: Decimal
    max_quantity: int
    min_expected_edge_bps: Decimal
    min_confidence: Decimal


@dataclass(frozen=True, slots=True)
class RiskDecision:
    decision: Decision
    reasons: tuple[str, ...]
    candidate_fingerprint: str


class Executor(Protocol):
    """Execution boundary implemented by shadow/analyzer adapters."""

    mode: ExecutionMode

    def submit(self, candidate: TradeCandidate) -> str:
        """Submit an already risk-approved candidate and return an execution id."""


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    candidate_id: str
    candidate_fingerprint: str
    mode: ExecutionMode
    risk: RiskDecision
    execution_id: str | None


class DeterministicRiskGate:
    """Small mandatory gate that an LLM cannot override."""

    def __init__(self, limits: RiskLimits) -> None:
        self._limits = limits

    def evaluate(self, candidate: TradeCandidate, *, now: datetime | None = None) -> RiskDecision:
        reasons: list[str] = []
        try:
            candidate.validate(now=now)
        except ValueError as exc:
            reasons.append(str(exc))

        if candidate.notional > self._limits.max_order_notional:
            reasons.append("max_order_notional")
        if candidate.quantity > self._limits.max_quantity:
            reasons.append("max_quantity")
        if candidate.expected_edge_bps < self._limits.min_expected_edge_bps:
            reasons.append("min_expected_edge_bps")
        if candidate.confidence < self._limits.min_confidence:
            reasons.append("min_confidence")

        return RiskDecision(
            decision=Decision.REJECTED if reasons else Decision.APPROVED,
            reasons=tuple(reasons),
            candidate_fingerprint=candidate.fingerprint,
        )


class AutonomousOrchestrator:
    """Join research decisions to non-live execution with fail-closed semantics."""

    def __init__(self, *, risk_gate: DeterministicRiskGate, executor: Executor) -> None:
        if executor.mode is ExecutionMode.LIVE:
            raise ValueError("live execution is forbidden in equity research orchestration v1")
        if executor.mode not in {ExecutionMode.SHADOW, ExecutionMode.ANALYZER}:
            raise ValueError("unsupported execution mode")
        self._risk_gate = risk_gate
        self._executor = executor

    def process(self, candidate: TradeCandidate, *, now: datetime | None = None) -> OrchestrationResult:
        risk = self._risk_gate.evaluate(candidate, now=now)
        execution_id: str | None = None
        if risk.decision is Decision.APPROVED:
            execution_id = self._executor.submit(candidate)
            if not execution_id:
                raise RuntimeError("executor returned an empty execution id")

        return OrchestrationResult(
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint,
            mode=self._executor.mode,
            risk=risk,
            execution_id=execution_id,
        )
