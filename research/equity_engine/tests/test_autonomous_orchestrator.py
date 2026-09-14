from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from equity_engine.autonomous_orchestrator import (
    AutonomousOrchestrator,
    Decision,
    DeterministicRiskGate,
    ExecutionMode,
    RiskLimits,
    TradeCandidate,
)


class FakeExecutor:
    def __init__(self, mode: ExecutionMode) -> None:
        self.mode = mode
        self.calls = []

    def submit(self, candidate: TradeCandidate) -> str:
        self.calls.append(candidate)
        return f"paper:{candidate.candidate_id}"


def candidate(**overrides):
    values = dict(
        candidate_id="candidate-1",
        symbol="RELIANCE",
        exchange="NSE",
        strategy_id="momentum",
        strategy_version="v1",
        side="BUY",
        quantity=1,
        product="MIS",
        price_type="MARKET",
        entry_price=Decimal("100"),
        stop_price=Decimal("98"),
        target_price=Decimal("105"),
        expected_edge_bps=Decimal("25"),
        confidence=Decimal("0.80"),
        valid_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        dataset_fingerprint="dataset-sha256",
        research_fingerprint="research-sha256",
    )
    values.update(overrides)
    return TradeCandidate(**values)


def gate():
    return DeterministicRiskGate(
        RiskLimits(
            max_order_notional=Decimal("1000"),
            max_quantity=5,
            min_expected_edge_bps=Decimal("10"),
            min_confidence=Decimal("0.70"),
        )
    )


def test_approved_candidate_reaches_analyzer_executor():
    executor = FakeExecutor(ExecutionMode.ANALYZER)
    result = AutonomousOrchestrator(risk_gate=gate(), executor=executor).process(candidate())
    assert result.risk.decision is Decision.APPROVED
    assert result.execution_id == "paper:candidate-1"
    assert len(executor.calls) == 1


def test_rejected_candidate_never_reaches_executor():
    executor = FakeExecutor(ExecutionMode.SHADOW)
    result = AutonomousOrchestrator(risk_gate=gate(), executor=executor).process(
        candidate(quantity=20, confidence=Decimal("0.10"))
    )
    assert result.risk.decision is Decision.REJECTED
    assert "max_quantity" in result.risk.reasons
    assert "min_confidence" in result.risk.reasons
    assert result.execution_id is None
    assert executor.calls == []


def test_stale_candidate_fails_closed():
    executor = FakeExecutor(ExecutionMode.ANALYZER)
    result = AutonomousOrchestrator(risk_gate=gate(), executor=executor).process(
        candidate(valid_until=datetime.now(timezone.utc) - timedelta(seconds=1))
    )
    assert result.risk.decision is Decision.REJECTED
    assert "candidate is stale" in result.risk.reasons
    assert executor.calls == []


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("exchange", "NFO", "only NSE/BSE"),
        ("product", "NRML", "CNC or MIS"),
        ("price_type", "SL", "MARKET or LIMIT"),
    ],
)
def test_unsupported_execution_semantics_fail_closed(field, value, reason):
    executor = FakeExecutor(ExecutionMode.ANALYZER)
    result = AutonomousOrchestrator(risk_gate=gate(), executor=executor).process(
        candidate(**{field: value})
    )
    assert result.risk.decision is Decision.REJECTED
    assert any(reason in item for item in result.risk.reasons)
    assert executor.calls == []


def test_live_executor_is_impossible_in_v1():
    with pytest.raises(ValueError, match="live execution is forbidden"):
        AutonomousOrchestrator(risk_gate=gate(), executor=FakeExecutor(ExecutionMode.LIVE))
