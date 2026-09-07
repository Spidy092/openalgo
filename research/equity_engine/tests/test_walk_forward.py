from datetime import date, time
from decimal import Decimal

import pandas as pd

from equity_engine.candidate_grid import baseline_definition
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import FillAssumptions, IntradaySimulationConfig
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.models import Exchange
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.tournament import RankingMetric
from equity_engine.walk_forward import make_walk_forward_windows, run_walk_forward_selection


def _multi_day_frame(days: int) -> pd.DataFrame:
    timestamps: list[pd.Timestamp] = []
    prices: list[float] = []
    start = pd.Timestamp("2026-08-10", tz="Asia/Kolkata")
    trading_day = start
    created = 0
    while created < days:
        if trading_day.weekday() < 5:
            base = 100.0 + created
            for hhmm, value in (("09:15", base), ("09:20", base + 0.2), ("15:20", base + 0.5)):
                timestamps.append(pd.Timestamp(f"{trading_day.date()} {hhmm}", tz="Asia/Kolkata"))
                prices.append(value)
            created += 1
        trading_day += pd.Timedelta(days=1)

    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": [p + 0.1 for p in prices],
            "volume": [100_000] * len(prices),
        },
        index=pd.DatetimeIndex(timestamps),
    )


def test_walk_forward_windows_are_chronological_and_embargoed() -> None:
    frame = _multi_day_frame(10)
    windows = make_walk_forward_windows(
        frame,
        train_trading_days=4,
        test_trading_days=2,
        step_trading_days=2,
        embargo_trading_days=1,
    )

    assert len(windows) == 2
    for window in windows:
        assert max(window.train_dates) < min(window.test_dates)
        all_dates = sorted(set(frame.index.date))
        train_end_idx = all_dates.index(max(window.train_dates))
        test_start_idx = all_dates.index(min(window.test_dates))
        assert test_start_idx - train_end_idx == 2


def test_walk_forward_selects_only_from_train_and_evaluates_on_test() -> None:
    frame = _multi_day_frame(8)
    windows = make_walk_forward_windows(
        frame,
        train_trading_days=3,
        test_trading_days=2,
        step_trading_days=2,
        embargo_trading_days=0,
    )
    results = run_walk_forward_selection(
        frame=frame,
        windows=windows,
        candidates=[baseline_definition(session_open=time(9, 15))],
        ranking_metric=RankingMetric.NET_RETURN_PCT,
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
        fills=FillAssumptions(
            slippage_bps_per_leg=Decimal("0"),
            half_spread_bps_per_leg=Decimal("0"),
        ),
        session_policy=NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=10),
        tick_size_policy=FixedTickSizePolicy(
            tick_size_rupees=Decimal("0.05"),
            source="synthetic-test",
        ),
        simulation_config=IntradaySimulationConfig(
            initial_cash=Decimal("1000"),
            max_trades_per_day=1,
        ),
    )

    assert results
    assert all(item.selected_candidate_id == "baseline:first-bar-hold" for item in results)
    assert all(item.train_evaluation.metrics.trade_count > 0 for item in results)
    assert all(item.test_evaluation.metrics.trade_count > 0 for item in results)
