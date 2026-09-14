from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SRC = ROOT / "research" / "equity_engine" / "src"
if str(RESEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SRC))

from equity_engine.autonomous_orchestrator import ExecutionMode, TradeCandidate  # noqa: E402
from equity_engine.openalgo_analyzer_executor import OpenAlgoAnalyzerExecutor  # noqa: E402
from services.autonomous import (  # noqa: E402
    AnalyzerPortfolioSnapshotAdapter,
    AnalyzerRiskContext,
    AnalyzerSnapshotUnavailable,
    PortfolioAnalyzerBridge,
    TradeCandidatePortfolioIntentAdapter,
)
from services.risk import PortfolioLimits, PortfolioRiskCode  # noqa: E402

NOW = datetime(2026, 9, 14, 6, 10, tzinfo=timezone.utc)


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


def limits(**overrides) -> PortfolioLimits:
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


def analyzer_positions(**overrides):
    response = {
        "status": "success",
        "mode": "analyze",
        "data": [
            {
                "symbol": "HDFCBANK",
                "exchange": "NSE",
                "product": "MIS",
                "quantity": 5,
                "average_price": 990.0,
                "ltp": 1000.0,
                "pnl": 50.0,
                "unrealized_pnl": 50.0,
                "today_realized_pnl": 0.0,
                "total_pnl_today": 50.0,
                "lot_size": 1.0,
            }
        ],
        "total_pnl": -50.0,
        "total_unrealized_pnl": 50.0,
        "total_today_realized_pnl": -100.0,
        "total_pnl_today": -50.0,
    }
    response.update(overrides)
    return True, response, 200


def context(**overrides) -> AnalyzerRiskContext:
    values = dict(
        as_of=NOW,
        market_open=True,
        market_data_timestamp=NOW - timedelta(seconds=5),
        candidate_reference_price=Decimal("2000"),
        kill_switch_engaged=False,
        symbol_activity=(),
    )
    values.update(overrides)
    return AnalyzerRiskContext(**values)


def make_executor(calls: list[dict]) -> OpenAlgoAnalyzerExecutor:
    def sandbox(order_data, api_key, original_data):
        calls.append(
            {
                "order": dict(order_data),
                "api_key": api_key,
                "original": dict(original_data),
            }
        )
        return True, {"status": "success", "orderid": "AN-1001"}, 200

    return OpenAlgoAnalyzerExecutor(
        api_key="secret",
        analyzer_mode_reader=lambda: True,
        sandbox_place_order=sandbox,
    )


def make_bridge(calls: list[dict], **limit_overrides) -> PortfolioAnalyzerBridge:
    return PortfolioAnalyzerBridge(
        limits=limits(**limit_overrides),
        snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(
            api_key="secret",
            positions_reader=analyzer_positions,
        ),
        analyzer_executor=make_executor(calls),
    )


def test_actual_trade_candidate_flows_through_portfolio_gate_to_analyzer():
    calls: list[dict] = []
    trade = candidate()

    result = make_bridge(calls).process(trade, context())

    assert result.submitted is True
    assert result.execution_id == "AN-1001"
    assert result.candidate_id == trade.candidate_id
    assert result.candidate_fingerprint == trade.fingerprint
    assert result.portfolio_decision.allowed is True
    assert result.portfolio_decision.projected_gross_exposure == Decimal("15000.0")
    assert result.portfolio_decision.projected_open_positions == 2
    assert len(calls) == 1
    assert calls[0]["api_key"] == "secret"
    assert calls[0]["order"] == {
        "strategy": "Autonomous:momentum@1.0.0",
        "symbol": "TCS",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 5,
        "pricetype": "MARKET",
        "product": "MIS",
        "price": 0.0,
        "trigger_price": 0.0,
    }
    assert calls[0]["original"] == calls[0]["order"]


def test_portfolio_rejection_never_reaches_analyzer_submit():
    calls: list[dict] = []

    result = make_bridge(calls, max_gross_exposure=Decimal("14999")).process(
        candidate(), context()
    )

    assert result.submitted is False
    assert result.execution_id is None
    assert result.portfolio_decision.allowed is False
    assert PortfolioRiskCode.MAX_GROSS_EXPOSURE in result.portfolio_decision.codes
    assert calls == []


def test_stale_market_evidence_rejects_before_analyzer_submit():
    calls: list[dict] = []

    result = make_bridge(calls).process(
        candidate(),
        context(market_data_timestamp=NOW - timedelta(seconds=31)),
    )

    assert result.submitted is False
    assert result.portfolio_decision.primary_code is PortfolioRiskCode.MARKET_DATA_STALE
    assert calls == []


def test_kill_switch_rejects_new_candidate_before_analyzer_submit():
    calls: list[dict] = []

    result = make_bridge(calls).process(
        candidate(),
        context(kill_switch_engaged=True),
    )

    assert result.submitted is False
    assert PortfolioRiskCode.KILL_SWITCH in result.portfolio_decision.codes
    assert calls == []


def test_market_candidate_requires_verified_reference_price():
    calls: list[dict] = []

    with pytest.raises(ValueError, match="verified candidate_reference_price"):
        make_bridge(calls).process(
            candidate(price_type="MARKET"),
            context(candidate_reference_price=None),
        )

    assert calls == []


def test_limit_candidate_uses_conservative_larger_verified_price():
    trade = candidate(price_type="LIMIT", entry_price=Decimal("1995"))
    intent = TradeCandidatePortfolioIntentAdapter.intent(
        trade,
        context(candidate_reference_price=Decimal("2010")),
    )

    assert intent.symbol == "NSE:TCS"
    assert intent.reference_price == Decimal("2010")


def test_snapshot_source_failure_is_fail_closed():
    calls: list[dict] = []

    def failed_positions():
        return False, {"status": "error", "message": "sandbox database unavailable"}, 503

    bridge = PortfolioAnalyzerBridge(
        limits=limits(),
        snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(
            api_key="secret",
            positions_reader=failed_positions,
        ),
        analyzer_executor=make_executor(calls),
    )

    with pytest.raises(AnalyzerSnapshotUnavailable, match="503"):
        bridge.process(candidate(), context())

    assert calls == []


def test_snapshot_must_prove_analyzer_mode():
    calls: list[dict] = []

    def wrong_mode():
        ok, response, status = analyzer_positions()
        response["mode"] = "live"
        return ok, response, status

    bridge = PortfolioAnalyzerBridge(
        limits=limits(),
        snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(
            api_key="secret",
            positions_reader=wrong_mode,
        ),
        analyzer_executor=make_executor(calls),
    )

    with pytest.raises(AnalyzerSnapshotUnavailable, match="Analyzer mode"):
        bridge.process(candidate(), context())

    assert calls == []


@pytest.mark.parametrize(
    "bad_row",
    [
        {
            "symbol": "NIFTY26SEP",
            "exchange": "NFO",
            "product": "MIS",
            "quantity": 1,
            "ltp": 250.0,
            "lot_size": 50.0,
        },
        {
            "symbol": "INFY",
            "exchange": "NSE",
            "product": "MIS",
            "quantity": 10,
            "ltp": 0.0,
            "lot_size": 1.0,
        },
        {
            "symbol": "INFY",
            "exchange": "NSE",
            "product": "NRML",
            "quantity": 10,
            "ltp": 1500.0,
            "lot_size": 1.0,
        },
    ],
)
def test_unsupported_or_unmarked_open_positions_fail_closed(bad_row):
    calls: list[dict] = []

    def positions():
        return analyzer_positions(data=[bad_row])

    bridge = PortfolioAnalyzerBridge(
        limits=limits(),
        snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(
            api_key="secret",
            positions_reader=positions,
        ),
        analyzer_executor=make_executor(calls),
    )

    with pytest.raises(AnalyzerSnapshotUnavailable):
        bridge.process(candidate(), context())

    assert calls == []


def test_duplicate_open_rows_for_same_risk_symbol_fail_closed_without_netting():
    rows = [
        {
            "symbol": "INFY",
            "exchange": "NSE",
            "product": "CNC",
            "quantity": 10,
            "ltp": 1500.0,
            "lot_size": 1.0,
        },
        {
            "symbol": "INFY",
            "exchange": "NSE",
            "product": "MIS",
            "quantity": -5,
            "ltp": 1500.0,
            "lot_size": 1.0,
        },
    ]
    adapter = AnalyzerPortfolioSnapshotAdapter(
        api_key="secret",
        positions_reader=lambda: analyzer_positions(data=rows),
    )

    with pytest.raises(AnalyzerSnapshotUnavailable, match="multiple open rows for NSE:INFY"):
        adapter.snapshot(context())


def test_reduce_only_is_not_claimed_atomic_by_plain_order_bridge():
    calls: list[dict] = []

    with pytest.raises(ValueError, match="atomic target-position"):
        make_bridge(calls).process(candidate(), context(), reduce_only=True)

    assert calls == []


def test_bridge_rejects_non_analyzer_executor_at_construction():
    class LiveExecutor:
        mode = ExecutionMode.LIVE

        def submit(self, candidate):  # pragma: no cover - construction must reject it
            raise AssertionError("must never be called")

    with pytest.raises(ValueError, match="only an Analyzer executor"):
        PortfolioAnalyzerBridge(
            limits=limits(),
            snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(
                api_key="secret",
                positions_reader=analyzer_positions,
            ),
            analyzer_executor=LiveExecutor(),
        )


def test_stale_candidate_fails_before_snapshot_or_submission():
    position_reads = 0
    calls: list[dict] = []

    def positions():
        nonlocal position_reads
        position_reads += 1
        return analyzer_positions()

    bridge = PortfolioAnalyzerBridge(
        limits=limits(),
        snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(
            api_key="secret",
            positions_reader=positions,
        ),
        analyzer_executor=make_executor(calls),
    )

    with pytest.raises(ValueError, match="candidate is stale"):
        bridge.process(candidate(valid_until=NOW), context())

    assert position_reads == 0
    assert calls == []


def test_context_requires_timezone_aware_snapshot_time():
    with pytest.raises(ValueError, match="as_of must be timezone-aware"):
        context(as_of=datetime(2026, 9, 14, 6, 10))
