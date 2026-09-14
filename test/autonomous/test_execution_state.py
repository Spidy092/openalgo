from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from services.autonomous import (
    AnalyzerPortfolioSnapshotAdapter,
    AnalyzerRiskContext,
    CandidateIdentityConflict,
    DuplicateExecution,
    ExecutionNeedsReconciliation,
    ExecutionState,
    ExecutionStateMachine,
    InvalidExecutionTransition,
    PortfolioAnalyzerBridge,
    SqliteExecutionStateStore,
    deterministic_execution_key,
)
from services.risk import PortfolioLimits

NOW = datetime(2026, 9, 14, 7, 0, tzinfo=timezone.utc)


def fingerprint(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def machine(tmp_path, *, name: str = "execution.sqlite3") -> ExecutionStateMachine:
    return ExecutionStateMachine(
        SqliteExecutionStateStore(tmp_path / name),
        clock=lambda: NOW,
    )


def test_deterministic_execution_key_uses_full_candidate_fingerprint():
    fp = fingerprint("candidate-a")
    assert deterministic_execution_key(fp) == f"openalgo-auto-v1:{fp}"
    assert deterministic_execution_key(fp.upper()) == f"openalgo-auto-v1:{fp}"

    with pytest.raises(ValueError, match="64-character"):
        deterministic_execution_key("not-a-sha")


def test_happy_path_persists_every_submission_boundary(tmp_path):
    state = machine(tmp_path)
    fp = fingerprint("candidate-a")

    record = state.propose(candidate_id="cand-a", candidate_fingerprint=fp)
    assert record.state is ExecutionState.PROPOSED

    record = state.risk_approved(record.execution_key)
    assert record.state is ExecutionState.RISK_APPROVED
    record = state.begin_submission(record.execution_key)
    assert record.state is ExecutionState.SUBMITTING
    record = state.submitted(record.execution_key, "AN-1001")
    assert record.state is ExecutionState.SUBMITTED
    assert record.analyzer_order_id == "AN-1001"
    record = state.acknowledged(record.execution_key)
    assert record.state is ExecutionState.ACKNOWLEDGED

    events = state.store.events(record.execution_key)
    assert [event.to_state for event in events] == [
        ExecutionState.PROPOSED,
        ExecutionState.RISK_APPROVED,
        ExecutionState.SUBMITTING,
        ExecutionState.SUBMITTED,
        ExecutionState.ACKNOWLEDGED,
    ]
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]


def test_store_reopen_preserves_idempotency_barrier(tmp_path):
    path = tmp_path / "execution.sqlite3"
    first = ExecutionStateMachine(SqliteExecutionStateStore(path), clock=lambda: NOW)
    fp = fingerprint("candidate-a")
    first.propose(candidate_id="cand-a", candidate_fingerprint=fp)

    second = ExecutionStateMachine(SqliteExecutionStateStore(path), clock=lambda: NOW)
    with pytest.raises(DuplicateExecution) as excinfo:
        second.propose(candidate_id="cand-a", candidate_fingerprint=fp)

    assert excinfo.value.record.state is ExecutionState.PROPOSED


def test_candidate_id_cannot_be_rebound_to_different_fingerprint(tmp_path):
    state = machine(tmp_path)
    state.propose(candidate_id="cand-a", candidate_fingerprint=fingerprint("one"))

    with pytest.raises(CandidateIdentityConflict):
        state.propose(candidate_id="cand-a", candidate_fingerprint=fingerprint("two"))


def test_restart_during_submitting_becomes_reconciliation_required(tmp_path):
    path = tmp_path / "execution.sqlite3"
    fp = fingerprint("candidate-a")
    first = ExecutionStateMachine(SqliteExecutionStateStore(path), clock=lambda: NOW)
    record = first.propose(candidate_id="cand-a", candidate_fingerprint=fp)
    first.risk_approved(record.execution_key)
    first.begin_submission(record.execution_key)

    restarted = ExecutionStateMachine(SqliteExecutionStateStore(path), clock=lambda: NOW)
    with pytest.raises(ExecutionNeedsReconciliation) as excinfo:
        restarted.propose(candidate_id="cand-a", candidate_fingerprint=fp)

    assert excinfo.value.record.state is ExecutionState.RECONCILIATION_REQUIRED
    assert (
        restarted.store.get(record.execution_key).state
        is ExecutionState.RECONCILIATION_REQUIRED
    )


def test_invalid_transition_fails_without_mutating_record(tmp_path):
    state = machine(tmp_path)
    record = state.propose(
        candidate_id="cand-a",
        candidate_fingerprint=fingerprint("candidate-a"),
    )

    with pytest.raises(InvalidExecutionTransition):
        state.submitted(record.execution_key, "AN-1001")

    stored = state.store.get(record.execution_key)
    assert stored is not None
    assert stored.state is ExecutionState.PROPOSED
    assert stored.version == 1


def test_fill_and_exit_lifecycle_uses_same_execution_record(tmp_path):
    state = machine(tmp_path)
    record = state.propose(
        candidate_id="cand-a",
        candidate_fingerprint=fingerprint("candidate-a"),
    )
    state.risk_approved(record.execution_key)
    state.begin_submission(record.execution_key)
    state.submitted(record.execution_key, "AN-1001")
    state.acknowledged(record.execution_key)
    state.partially_filled(record.execution_key, detail="filled=2/5")
    state.filled(record.execution_key, detail="filled=5/5")
    state.exit_pending(record.execution_key, detail="target_or_stop_exit_requested")
    record = state.closed(record.execution_key, detail="position_flat")

    assert record.state is ExecutionState.CLOSED
    assert record.analyzer_order_id == "AN-1001"
    with pytest.raises(InvalidExecutionTransition):
        state.filled(record.execution_key)


@dataclass
class FakeCandidate:
    candidate_id: str = "cand-bridge"
    symbol: str = "TCS"
    exchange: str = "NSE"
    side: str = "BUY"
    quantity: int = 5
    price_type: str = "MARKET"
    entry_price: Decimal = Decimal("1995")
    valid_until: datetime = datetime(2099, 1, 1, tzinfo=timezone.utc)
    fingerprint: str = fingerprint("bridge-candidate")

    def validate(self, *, now: datetime | None = None) -> None:
        current = now or datetime.now(timezone.utc)
        if self.valid_until <= current:
            raise ValueError("candidate is stale")


class FakeAnalyzerExecutor:
    mode = "analyzer"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def submit(self, candidate: FakeCandidate) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated analyzer transport failure")
        return "AN-2001"


def positions_reader():
    return (
        True,
        {
            "status": "success",
            "mode": "analyze",
            "data": [],
            "total_unrealized_pnl": 0.0,
            "total_today_realized_pnl": 0.0,
        },
        200,
    )


def limits(**overrides) -> PortfolioLimits:
    values = dict(
        max_gross_exposure=Decimal("100000"),
        max_abs_net_exposure=Decimal("100000"),
        max_open_positions=5,
        max_symbol_exposure=Decimal("50000"),
        max_symbol_concentration_pct=Decimal("100"),
        max_daily_loss=Decimal("5000"),
        cooldown_seconds=0,
        max_market_data_age_seconds=30,
    )
    values.update(overrides)
    return PortfolioLimits(**values)


def context() -> AnalyzerRiskContext:
    return AnalyzerRiskContext(
        as_of=NOW,
        market_open=True,
        market_data_timestamp=NOW - timedelta(seconds=1),
        candidate_reference_price=Decimal("2000"),
    )


def bridge(tmp_path, executor: FakeAnalyzerExecutor, **limit_overrides):
    state = machine(tmp_path)
    return (
        PortfolioAnalyzerBridge(
            limits=limits(**limit_overrides),
            snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(
                api_key="test-key",
                positions_reader=positions_reader,
            ),
            analyzer_executor=executor,
            execution_state_machine=state,
        ),
        state,
    )


def test_bridge_persists_acknowledged_before_returning_success(tmp_path):
    executor = FakeAnalyzerExecutor()
    guarded, state = bridge(tmp_path, executor)
    candidate = FakeCandidate()

    result = guarded.process(candidate, context())

    assert result.execution_id == "AN-2001"
    assert result.execution_state is ExecutionState.ACKNOWLEDGED
    assert result.execution_key == deterministic_execution_key(candidate.fingerprint)
    assert executor.calls == 1
    stored = state.store.get(result.execution_key)
    assert stored is not None
    assert stored.state is ExecutionState.ACKNOWLEDGED
    assert stored.analyzer_order_id == "AN-2001"


def test_bridge_duplicate_candidate_never_calls_analyzer_twice(tmp_path):
    executor = FakeAnalyzerExecutor()
    guarded, _ = bridge(tmp_path, executor)
    candidate = FakeCandidate()

    guarded.process(candidate, context())
    with pytest.raises(DuplicateExecution):
        guarded.process(candidate, context())

    assert executor.calls == 1


def test_portfolio_rejection_is_terminal_rejected_without_submit(tmp_path):
    executor = FakeAnalyzerExecutor()
    guarded, state = bridge(
        tmp_path,
        executor,
        max_gross_exposure=Decimal("9000"),
    )
    candidate = FakeCandidate()

    result = guarded.process(candidate, context())

    assert result.execution_id is None
    assert result.execution_state is ExecutionState.REJECTED
    assert executor.calls == 0
    stored = state.store.get(result.execution_key)
    assert stored is not None
    assert stored.state is ExecutionState.REJECTED


def test_analyzer_exception_after_submitting_requires_reconciliation(tmp_path):
    executor = FakeAnalyzerExecutor(fail=True)
    guarded, state = bridge(tmp_path, executor)
    candidate = FakeCandidate()

    with pytest.raises(ExecutionNeedsReconciliation) as excinfo:
        guarded.process(candidate, context())

    assert executor.calls == 1
    assert excinfo.value.record.state is ExecutionState.RECONCILIATION_REQUIRED
    stored = state.store.get(excinfo.value.record.execution_key)
    assert stored is not None
    assert stored.state is ExecutionState.RECONCILIATION_REQUIRED

    with pytest.raises(ExecutionNeedsReconciliation):
        guarded.process(candidate, context())
    assert executor.calls == 1
