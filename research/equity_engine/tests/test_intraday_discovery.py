"""Intraday strategy-discovery pipeline (Agent 2): tests first, no orders.

The pipeline must reuse the existing gauntlet (vectorbt screening,
walk-forward, event-driven simulator, documented costs, gates, shadow
replay) and emit a validated candidate artifact. It must never place a
live order, never touch the network, and stay intraday cash-equity only.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from equity_engine.gates import PromotionThresholds
from equity_engine.intraday_discovery import (
    DISCOVERY_SCHEMA_VERSION,
    DiscoveryConfig,
    build_intraday_candidates,
    run_discovery,
    split_train_validation_test,
)
from equity_engine.models import Exchange
from equity_engine.provenance import MarketDataManifest
from equity_engine.tournament import RankingMetric

IST = "Asia/Kolkata"


def _manifest(start: datetime, end: datetime) -> MarketDataManifest:
    return MarketDataManifest(
        provider="synthetic-discovery-test",
        exchange="NSE",
        instrument_token="NSE_EQ|DISCOVERYTEST",
        symbol="DISCOVERYTEST",
        timezone=IST,
        interval="5m",
        timestamp_semantics="bar_start",
        start=start,
        end=end,
        retrieved_at=datetime(2026, 9, 10, tzinfo=UTC),
        adjustment_policy="unadjusted_splits_not_present_in_fixture",
        universe_rule_version="synthetic-single-liquid-symbol-v1",
        source_reference="synthetic-fixture-no-network",
    )


def _dense_session_frame(*, days: int, drift_per_day: float) -> pd.DataFrame:
    """Build dense 5-minute continuous-session bars (09:15 inclusive, 15:30 exclusive).

    Every day drifts linearly by ``drift_per_day`` so a long intraday rule is
    either clearly profitable (positive drift) or clearly losing (negative
    drift) after documented costs. Volume is spiked post-range so the ORB
    volume filter can trigger deterministically.
    """
    frames: list[pd.DataFrame] = []
    base_date = date(2026, 9, 1)
    # Walk calendar days but keep only weekdays so dates look like trading days.
    trading_dates: list[date] = []
    cursor = base_date
    while len(trading_dates) < days:
        if cursor.weekday() < 5:
            trading_dates.append(cursor)
        cursor = cursor.fromordinal(cursor.toordinal() + 1)
    for day_index, trade_date in enumerate(trading_dates):
        day_start = 100.0 + day_index * 0.5
        index = pd.date_range(
            f"{trade_date.isoformat()} 09:15",
            f"{trade_date.isoformat()} 15:30",
            freq="5min",
            inclusive="left",
            tz=IST,
        )
        assert len(index) == 75, f"expected 75 five-minute bars, got {len(index)}"
        closes: list[float] = []
        opens: list[float] = []
        volumes: list[int] = []
        for bar_index in range(len(index)):
            progress = bar_index / (len(index) - 1)
            price = day_start + drift_per_day * progress
            closes.append(price)
            # Next-bar execution uses bar open; keep open == prior close path
            # except for the first bar of the day.
            if bar_index == 0:
                opens.append(price)
            else:
                opens.append(closes[bar_index - 1])
            # Opening 15-minute range is the first three 5-minute bars;
            # keep their volume low so the post-range spike satisfies the
            # 1.2x ORB volume filter deterministically.
            if bar_index < 3:
                volumes.append(1000)
            else:
                volumes.append(1600)
        frame = pd.DataFrame(
            {
                "open": opens,
                "high": [max(o, c) + 0.08 for o, c in zip(opens, closes)],
                "low": [min(o, c) - 0.08 for o, c in zip(opens, closes)],
                "close": closes,
                "volume": volumes,
            },
            index=index,
        )
        frames.append(frame)
    full = pd.concat(frames).sort_index()
    assert full.index.is_monotonic_increasing
    return full


def _config(
    frame: pd.DataFrame,
    *,
    reconciliation_error: Decimal | None,
) -> DiscoveryConfig:
    trading_dates = sorted(set(frame.index.date))
    assert len(trading_dates) >= 10
    return DiscoveryConfig(
        instrument_token="NSE_EQ|DISCOVERYTEST",
        symbol="DISCOVERYTEST",
        exchange=Exchange.NSE,
        session_open=time(9, 15),
        bar_minutes=5,
        breakout_buffer_bps=Decimal(0),
        screening_cash=Decimal(1000),
        screening_fee_rate=Decimal("0.0005"),
        screening_slippage_rate=Decimal("0.0005"),
        initial_cash=Decimal(1000),
        max_trades_per_day=1,
        base_slippage_bps_per_leg=Decimal(0),
        base_half_spread_bps_per_leg=Decimal(0),
        stress_slippage_bps_per_leg=Decimal(5),
        stress_half_spread_bps_per_leg=Decimal(5),
        train_end_date=trading_dates[4],
        validation_end_date=trading_dates[7],
        walk_forward_train_days=3,
        walk_forward_test_days=2,
        walk_forward_step_days=2,
        walk_forward_embargo_days=0,
        ranking_metric=RankingMetric.NET_RETURN_PCT,
        min_sharpe=Decimal("0.5"),
        promotion_thresholds=PromotionThresholds(
            min_trades=2,
            min_profit_factor=Decimal("1.0"),
            max_drawdown_pct=Decimal(30),
            min_walk_forward_windows=1,
            max_cost_reconciliation_error_inr=Decimal("0.05"),
        ),
        cost_reconciliation_error_inr=reconciliation_error,
        cost_reconciliation_source=(
            "synthetic-test-reconciliation" if reconciliation_error is not None else "not-performed"
        ),
        tick_size_rupees=Decimal("0.05"),
        tick_size_source="synthetic-test",
        exit_buffer_minutes=10,
        cas_eligible=False,
        pricing_date=date(2026, 9, 7),
    )


def test_candidate_templates_cover_orb_baseline_and_reversion() -> None:
    definitions = build_intraday_candidates(
        session_open=time(9, 15),
        bar_minutes=5,
        breakout_buffer_bps=Decimal(0),
    )
    ids = [item.candidate_id for item in definitions]
    assert any(item.startswith("orb:") for item in ids), ids
    assert "baseline:first-bar-hold" in ids, ids
    assert any(item.startswith("meanrev:") for item in ids), ids
    assert len(definitions) >= 3


def test_train_validation_test_split_keeps_test_untouched() -> None:
    frame = _dense_session_frame(days=10, drift_per_day=1.5)
    trading_dates = sorted(set(frame.index.date))
    train, validation, test = split_train_validation_test(
        frame,
        train_end_date=trading_dates[4],
        validation_end_date=trading_dates[7],
    )
    assert max(train.index.date) <= trading_dates[4]
    assert min(validation.index.date) > trading_dates[4]
    assert max(validation.index.date) <= trading_dates[7]
    assert min(test.index.date) > trading_dates[7]
    assert set(train.index.date).isdisjoint(set(test.index.date))
    assert set(validation.index.date).isdisjoint(set(test.index.date))
    assert len(train) > 0 and len(validation) > 0 and len(test) > 0


@pytest.mark.timeout(90)
def test_gauntlet_accepts_known_good_synthetic_trend() -> None:
    frame = _dense_session_frame(days=10, drift_per_day=1.5)
    manifest = _manifest(
        frame.index[0].to_pydatetime(),
        frame.index[-1].to_pydatetime(),
    )
    result = run_discovery(frame, manifest, _config(frame, reconciliation_error=Decimal("0.00")))

    assert result.live_orders_called is False
    assert result.status == "PASS", result.violations
    assert result.gate_decision.passed, result.gate_decision.violations
    assert result.selected_test_metrics.trade_count >= 2
    assert result.selected_test_metrics.net_pnl > 0
    assert result.test_expectancy > 0
    assert result.test_sharpe >= Decimal("0.5")
    assert result.walk_forward_windows >= 1
    assert result.walk_forward_profitable_folds * 2 > result.walk_forward_windows
    assert len(result.walk_forward_folds) == result.walk_forward_windows
    assert result.stress_worst_net_pnl is not None
    assert result.stress_worst_net_pnl > 0
    assert result.shadow_decisions >= 1

    payload = result.to_dict()
    assert payload["schema_version"] == DISCOVERY_SCHEMA_VERSION
    assert payload["status"] == "PASS"
    assert payload["track"] == "intraday"
    assert payload["instrument"]["exchange"] == "NSE"
    assert payload["live_orders_called"] is False
    assert payload["validation_metrics"]["held_out_test_present"] is True
    assert payload["validation_metrics"]["baseline_comparison_present"] is True
    assert payload["validation_metrics"]["slippage_stress_present"] is True
    assert payload["validation_metrics"]["event_driven_validation_present"] is True
    assert Decimal(payload["validation_metrics"]["test_expectancy"]) > 0
    assert Decimal(payload["validation_metrics"]["test_sharpe"]) >= Decimal("0.5")
    raw_pf = payload["validation_metrics"]["test_profit_factor"]
    assert raw_pf is None or Decimal(raw_pf) >= Decimal("1.0")
    assert "walk_forward_folds" in payload["validation_metrics"]
    assert (
        len(payload["validation_metrics"]["walk_forward_folds"])
        == payload["validation_metrics"]["walk_forward_windows"]
    )
    assert payload["validation_metrics"]["walk_forward_majority_profitable"] is True
    # JSON-serializable: Decimal/date encoded as strings.
    json.dumps(payload)


def test_gauntlet_rejects_bad_synthetic_downtrend() -> None:
    frame = _dense_session_frame(days=10, drift_per_day=-1.5)
    manifest = _manifest(
        frame.index[0].to_pydatetime(),
        frame.index[-1].to_pydatetime(),
    )
    result = run_discovery(frame, manifest, _config(frame, reconciliation_error=Decimal("0.00")))

    assert result.live_orders_called is False
    assert result.status == "FAIL"
    assert result.to_dict()["live_orders_called"] is False
    assert result.to_dict()["status"] == "FAIL"


def test_missing_cost_reconciliation_fails_closed() -> None:
    frame = _dense_session_frame(days=10, drift_per_day=1.5)
    manifest = _manifest(
        frame.index[0].to_pydatetime(),
        frame.index[-1].to_pydatetime(),
    )
    result = run_discovery(frame, manifest, _config(frame, reconciliation_error=None))

    assert result.status == "FAIL"
    assert any("reconciliation" in violation.lower() for violation in result.violations), (
        result.violations
    )


def test_unachievable_sharpe_floor_fails_closed() -> None:
    frame = _dense_session_frame(days=10, drift_per_day=1.5)
    manifest = _manifest(
        frame.index[0].to_pydatetime(),
        frame.index[-1].to_pydatetime(),
    )
    base = _config(frame, reconciliation_error=Decimal("0.00"))
    passing = run_discovery(frame, manifest, base)
    assert passing.status == "PASS", passing.violations
    strict = DiscoveryConfig(
        instrument_token=base.instrument_token,
        symbol=base.symbol,
        exchange=base.exchange,
        session_open=base.session_open,
        bar_minutes=base.bar_minutes,
        breakout_buffer_bps=base.breakout_buffer_bps,
        screening_cash=base.screening_cash,
        screening_fee_rate=base.screening_fee_rate,
        screening_slippage_rate=base.screening_slippage_rate,
        initial_cash=base.initial_cash,
        max_trades_per_day=base.max_trades_per_day,
        base_slippage_bps_per_leg=base.base_slippage_bps_per_leg,
        base_half_spread_bps_per_leg=base.base_half_spread_bps_per_leg,
        stress_slippage_bps_per_leg=base.stress_slippage_bps_per_leg,
        stress_half_spread_bps_per_leg=base.stress_half_spread_bps_per_leg,
        train_end_date=base.train_end_date,
        validation_end_date=base.validation_end_date,
        walk_forward_train_days=base.walk_forward_train_days,
        walk_forward_test_days=base.walk_forward_test_days,
        walk_forward_step_days=base.walk_forward_step_days,
        walk_forward_embargo_days=base.walk_forward_embargo_days,
        ranking_metric=base.ranking_metric,
        min_sharpe=passing.test_sharpe + Decimal(1),
        promotion_thresholds=base.promotion_thresholds,
        cost_reconciliation_error_inr=Decimal("0.00"),
        cost_reconciliation_source="synthetic-test-reconciliation",
        tick_size_rupees=base.tick_size_rupees,
        tick_size_source=base.tick_size_source,
        exit_buffer_minutes=base.exit_buffer_minutes,
        cas_eligible=base.cas_eligible,
        pricing_date=base.pricing_date,
    )
    result = run_discovery(frame, manifest, strict)

    assert result.status == "FAIL"
    assert any("sharpe" in violation.lower() for violation in result.violations)
    assert result.to_dict()["status"] == "FAIL"
    assert result.to_dict()["live_orders_called"] is False


def test_discovery_is_deterministic_and_records_fingerprints() -> None:
    frame = _dense_session_frame(days=10, drift_per_day=1.5)
    manifest = _manifest(
        frame.index[0].to_pydatetime(),
        frame.index[-1].to_pydatetime(),
    )
    first = run_discovery(frame, manifest, _config(frame, reconciliation_error=Decimal("0.00")))
    second = run_discovery(frame, manifest, _config(frame, reconciliation_error=Decimal("0.00")))

    assert first.data_fingerprint == second.data_fingerprint
    assert first.selected_candidate_id == second.selected_candidate_id
    assert first.to_dict() == second.to_dict()
    assert len(first.data_fingerprint) == 64
    assert len(first.candidate_fingerprint) == 64


def test_discovery_module_has_no_live_order_path() -> None:
    import equity_engine.intraday_discovery as discovery

    source = Path(discovery.__file__).read_text(encoding="utf-8")
    forbidden = (
        "placeorder",
        "place_order",
        "UpstoxBrokerCostProvider",
        "httpx.post",
        "live_orders_called=True",
        "live_orders_called = True",
        "import broker",
        "from broker",
        "services.place",
    )
    lowered = source.lower()
    for token in forbidden:
        assert token.lower() not in lowered, f"forbidden live-order token present: {token}"

    for module in ("broker", "services.place_order_service", "services.order_router_service"):
        assert module not in sys.modules, f"live module imported: {module}"
