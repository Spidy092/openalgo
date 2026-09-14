from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SRC = ROOT / "research" / "equity_engine" / "src"
if str(RESEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SRC))

SCRIPT = ROOT / "research" / "equity_engine" / "scripts" / "autonomous_analyzer_run.py"
spec = importlib.util.spec_from_file_location("autonomous_analyzer_run_script", SCRIPT)
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)

from equity_engine.autonomous_orchestrator import (  # noqa: E402
    AutonomousOrchestrator,
    DeterministicRiskGate,
    ExecutionMode,
    RiskLimits,
    TradeCandidate,
)
from equity_engine.autonomous_session import (  # noqa: E402
    AutonomousSession,
    JsonlExecutionJournal,
    SessionLimits,
    SessionOutcome,
)
from equity_engine.openalgo_analyzer_executor import OpenAlgoAnalyzerExecutor  # noqa: E402
from services.autonomous import (  # noqa: E402
    AnalyzerPortfolioSnapshotAdapter,
    PortfolioAnalyzerBridge,
)
from services.risk import PortfolioLimits  # noqa: E402

NOW = datetime(2026, 9, 14, 6, 30, tzinfo=timezone.utc)


def candidate(**overrides) -> TradeCandidate:
    values = dict(
        candidate_id="cand-001",
        symbol="TCS",
        exchange="NSE",
        strategy_id="momentum",
        strategy_version="1.0.0",
        side="BUY",
        quantity=5,
        product="MIS",
        price_type="MARKET",
        entry_price=Decimal("1995"),
        stop_price=Decimal("1950"),
        target_price=Decimal("2100"),
        expected_edge_bps=Decimal("35"),
        confidence=Decimal("0.80"),
        valid_until=datetime(2099, 1, 1, tzinfo=timezone.utc),
        dataset_fingerprint="dataset-sha",
        research_fingerprint="research-sha",
    )
    values.update(overrides)
    return TradeCandidate(**values)


def portfolio_limits(**overrides) -> PortfolioLimits:
    values = dict(
        max_gross_exposure=Decimal("100000"),
        max_abs_net_exposure=Decimal("100000"),
        max_open_positions=5,
        max_symbol_exposure=Decimal("50000"),
        max_symbol_concentration_pct=Decimal("80"),
        max_daily_loss=Decimal("5000"),
        cooldown_seconds=60,
        max_market_data_age_seconds=30,
    )
    values.update(overrides)
    return PortfolioLimits(**values)


def positions_reader():
    return (
        True,
        {
            "status": "success",
            "mode": "analyze",
            "data": [
                {
                    "symbol": "HDFCBANK",
                    "exchange": "NSE",
                    "product": "MIS",
                    "quantity": 5,
                    "ltp": 1000.0,
                    "lot_size": 1.0,
                }
            ],
            "total_unrealized_pnl": 50.0,
            "total_today_realized_pnl": -100.0,
        },
        200,
    )


def evidence() -> launcher.PortfolioContextEvidence:
    return launcher.PortfolioContextEvidence(
        market_open=True,
        market_data_timestamp=NOW - timedelta(seconds=5),
        kill_switch_engaged=False,
        candidate_reference_prices={"cand-001": Decimal("2000")},
        symbol_activity=(),
    )


def make_raw_executor(calls: list[dict]) -> OpenAlgoAnalyzerExecutor:
    def sandbox(order_data, api_key, original_data):
        calls.append(dict(order_data))
        return True, {"status": "success", "orderid": "AN-2001"}, 200

    return OpenAlgoAnalyzerExecutor(
        api_key="secret",
        analyzer_mode_reader=lambda: True,
        sandbox_place_order=sandbox,
    )


def make_guarded_executor(calls: list[dict], **limit_overrides):
    raw = make_raw_executor(calls)
    bridge = PortfolioAnalyzerBridge(
        limits=portfolio_limits(**limit_overrides),
        snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(
            api_key="secret",
            positions_reader=positions_reader,
        ),
        analyzer_executor=raw,
    )
    ctx = evidence()
    return launcher.PortfolioGuardedAnalyzerExecutor(
        bridge=bridge,
        context_provider=lambda trade: ctx.context_for(trade, now=NOW),
        mode=raw.mode,
    )


def make_orchestrator(executor):
    return AutonomousOrchestrator(
        risk_gate=DeterministicRiskGate(
            RiskLimits(
                max_order_notional=Decimal("50000"),
                max_quantity=100,
                min_expected_edge_bps=Decimal("10"),
                min_confidence=Decimal("0.5"),
            )
        ),
        executor=executor,
    )


def test_guarded_executor_preserves_real_analyzer_mode_and_submits_after_portfolio_approval():
    calls: list[dict] = []
    guarded = make_guarded_executor(calls)

    assert guarded.mode is ExecutionMode.ANALYZER
    result = make_orchestrator(guarded).process(candidate(), now=NOW)

    assert result.execution_id == "AN-2001"
    assert len(calls) == 1


def test_portfolio_rejection_is_journaled_as_rejected_and_never_submitted(tmp_path):
    calls: list[dict] = []
    guarded = make_guarded_executor(calls, max_gross_exposure=Decimal("14999"))
    session = AutonomousSession(
        orchestrator=make_orchestrator(guarded),
        journal=JsonlExecutionJournal(tmp_path / "journal.jsonl"),
        limits=SessionLimits(
            max_orders=10,
            max_gross_notional=Decimal("100000"),
            max_orders_per_symbol=5,
        ),
    )

    entries = session.run((candidate(),))

    assert len(entries) == 1
    assert entries[0].outcome is SessionOutcome.REJECTED
    assert entries[0].execution_id is None
    assert any(reason.startswith("portfolio_max_gross_exposure:") for reason in entries[0].reasons)
    assert calls == []

    stored = json.loads((tmp_path / "journal.jsonl").read_text(encoding="utf-8"))
    assert stored["outcome"] == "rejected"
    assert stored["execution_id"] is None


def test_portfolio_context_loader_requires_exact_candidate_price_coverage(tmp_path):
    path = tmp_path / "context.json"
    path.write_text(
        json.dumps(
            {
                "market_open": True,
                "market_data_timestamp": (NOW - timedelta(seconds=5)).isoformat(),
                "kill_switch_engaged": False,
                "candidate_reference_prices": {"wrong-id": "2000"},
                "symbol_activity": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keys mismatch"):
        launcher._load_portfolio_context(path, (candidate(),))


def test_portfolio_context_loader_parses_canonical_activity_and_uses_runtime_as_of(tmp_path):
    path = tmp_path / "context.json"
    last_increase = NOW - timedelta(minutes=5)
    path.write_text(
        json.dumps(
            {
                "market_open": True,
                "market_data_timestamp": (NOW - timedelta(seconds=5)).isoformat(),
                "kill_switch_engaged": False,
                "candidate_reference_prices": {"cand-001": "2000.50"},
                "symbol_activity": [
                    {
                        "symbol": "nse:tcs",
                        "last_increase_at": last_increase.isoformat(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = launcher._load_portfolio_context(path, (candidate(),))
    later = NOW + timedelta(seconds=10)
    ctx = loaded.context_for(candidate(), now=later)

    assert ctx.as_of == later
    assert ctx.market_data_timestamp == NOW - timedelta(seconds=5)
    assert ctx.candidate_reference_price == Decimal("2000.50")
    assert ctx.symbol_activity[0].symbol == "NSE:TCS"
    assert ctx.symbol_activity[0].last_increase_at == last_increase


def test_old_evidence_expires_against_runtime_clock_before_submit():
    calls: list[dict] = []
    raw = make_raw_executor(calls)
    bridge = PortfolioAnalyzerBridge(
        limits=portfolio_limits(max_market_data_age_seconds=30),
        snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(
            api_key="secret",
            positions_reader=positions_reader,
        ),
        analyzer_executor=raw,
    )
    old = launcher.PortfolioContextEvidence(
        market_open=True,
        market_data_timestamp=NOW - timedelta(minutes=10),
        kill_switch_engaged=False,
        candidate_reference_prices={"cand-001": Decimal("2000")},
        symbol_activity=(),
    )
    guarded = launcher.PortfolioGuardedAnalyzerExecutor(
        bridge=bridge,
        context_provider=lambda trade: old.context_for(trade, now=NOW),
        mode=raw.mode,
    )
    session = AutonomousSession(
        orchestrator=make_orchestrator(guarded),
        journal=JsonlExecutionJournal(Path(pytest.ensuretemp("old-evidence")) / "journal.jsonl")
        if hasattr(pytest, "ensuretemp")
        else JsonlExecutionJournal(Path("/tmp/openalgo-old-evidence-test.jsonl")),
        limits=SessionLimits(
            max_orders=10,
            max_gross_notional=Decimal("100000"),
            max_orders_per_symbol=5,
        ),
    )

    entries = session.run((candidate(),))

    assert entries[0].outcome is SessionOutcome.REJECTED
    assert any(reason.startswith("portfolio_market_data_stale:") for reason in entries[0].reasons)
    assert calls == []
