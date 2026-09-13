from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from equity_engine.autonomous_orchestrator import (
    AutonomousOrchestrator,
    DeterministicRiskGate,
    ExecutionMode,
    RiskLimits,
    TradeCandidate,
)
from equity_engine.autonomous_session import (
    AutonomousSession,
    JsonlExecutionJournal,
    SessionLimits,
    SessionOutcome,
)


class FakeExecutor:
    mode = ExecutionMode.ANALYZER

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls = []
        self.fail_on = fail_on

    def submit(self, candidate: TradeCandidate) -> str:
        self.calls.append(candidate.candidate_id)
        if candidate.candidate_id == self.fail_on:
            raise RuntimeError("simulated sandbox failure")
        return f"paper:{candidate.candidate_id}"


def candidate(candidate_id: str, *, symbol="RELIANCE", price="100", quantity=1):
    return TradeCandidate(
        candidate_id=candidate_id,
        symbol=symbol,
        exchange="NSE",
        strategy_id="momentum",
        strategy_version="v1",
        side="BUY",
        quantity=quantity,
        product="MIS",
        price_type="MARKET",
        entry_price=Decimal(price),
        stop_price=Decimal("98"),
        target_price=Decimal("105"),
        expected_edge_bps=Decimal("25"),
        confidence=Decimal("0.80"),
        valid_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        dataset_fingerprint="dataset-sha256",
        research_fingerprint=f"research-{candidate_id}",
    )


def orchestrator(executor):
    return AutonomousOrchestrator(
        risk_gate=DeterministicRiskGate(
            RiskLimits(
                max_order_notional=Decimal("10000"),
                max_quantity=100,
                min_expected_edge_bps=Decimal("10"),
                min_confidence=Decimal("0.70"),
            )
        ),
        executor=executor,
    )


def limits(**overrides):
    values = dict(
        max_orders=3,
        max_gross_notional=Decimal("1000"),
        max_orders_per_symbol=2,
    )
    values.update(overrides)
    return SessionLimits(**values)


def test_journal_suppresses_duplicate_execution(tmp_path):
    executor = FakeExecutor()
    journal = JsonlExecutionJournal(tmp_path / "journal.jsonl")
    session = AutonomousSession(
        orchestrator=orchestrator(executor), journal=journal, limits=limits()
    )
    first = session.run([candidate("a")])
    second = session.run([candidate("a")])
    assert first[0].outcome is SessionOutcome.EXECUTED
    assert second[0].outcome is SessionOutcome.DUPLICATE
    assert executor.calls == ["a"]


def test_session_gross_notional_limit_is_cumulative(tmp_path):
    executor = FakeExecutor()
    session = AutonomousSession(
        orchestrator=orchestrator(executor),
        journal=JsonlExecutionJournal(tmp_path / "journal.jsonl"),
        limits=limits(max_gross_notional=Decimal("150")),
    )
    result = session.run([candidate("a", price="100"), candidate("b", price="100")])
    assert [entry.outcome for entry in result] == [
        SessionOutcome.EXECUTED,
        SessionOutcome.REJECTED,
    ]
    assert "session_max_gross_notional" in result[1].reasons
    assert executor.calls == ["a"]


def test_session_stops_on_health_failure_without_execution(tmp_path):
    executor = FakeExecutor()
    session = AutonomousSession(
        orchestrator=orchestrator(executor),
        journal=JsonlExecutionJournal(tmp_path / "journal.jsonl"),
        limits=limits(),
        health_check=lambda: False,
    )
    result = session.run([candidate("a"), candidate("b")])
    assert len(result) == 1
    assert result[0].outcome is SessionOutcome.STOPPED
    assert result[0].reasons == ("health_check_failed",)
    assert executor.calls == []


def test_execution_error_is_journaled_and_never_retried_in_same_run(tmp_path):
    executor = FakeExecutor(fail_on="b")
    session = AutonomousSession(
        orchestrator=orchestrator(executor),
        journal=JsonlExecutionJournal(tmp_path / "journal.jsonl"),
        limits=limits(),
    )
    result = session.run([candidate("a"), candidate("b"), candidate("c")])
    assert [entry.outcome for entry in result] == [
        SessionOutcome.EXECUTED,
        SessionOutcome.ERROR,
    ]
    assert executor.calls == ["a", "b"]


def test_reused_candidate_id_with_changed_payload_stops(tmp_path):
    executor = FakeExecutor()
    path = tmp_path / "journal.jsonl"
    first = AutonomousSession(
        orchestrator=orchestrator(executor),
        journal=JsonlExecutionJournal(path),
        limits=limits(),
    )
    first.run([candidate("a", price="100")])

    second = AutonomousSession(
        orchestrator=orchestrator(executor),
        journal=JsonlExecutionJournal(path),
        limits=limits(),
    )
    result = second.run([candidate("a", price="101")])
    assert result[0].outcome is SessionOutcome.STOPPED
    assert result[0].reasons == ("candidate_id_reused_with_different_fingerprint",)
    assert executor.calls == ["a"]


def test_corrupt_existing_journal_fails_closed(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid autonomous journal"):
        JsonlExecutionJournal(path)
