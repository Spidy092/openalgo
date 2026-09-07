from datetime import date, time
from decimal import Decimal

import pandas as pd

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import FillAssumptions, IntradaySimulationConfig
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.models import Exchange
from equity_engine.strategies import first_bar_hold_baseline
from equity_engine.stress import FrictionScenario, run_friction_stress
from equity_engine.tick_size import FixedTickSizePolicy


def _frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-01 15:20", tz="Asia/Kolkata"),
        ]
    )
    prices = [100.0, 100.0, 102.0]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": [p + 0.1 for p in prices],
            "volume": [100_000] * 3,
        },
        index=index,
    )


def _tick_policy() -> FixedTickSizePolicy:
    return FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="synthetic-test")


def test_worse_friction_cannot_improve_exact_pnl() -> None:
    frame = _frame()
    signals = first_bar_hold_baseline(frame, session_open=time(9, 15))
    result = run_friction_stress(
        candidate_id="baseline",
        frame=frame,
        signals=signals,
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
        session_policy=NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=10),
        tick_size_policy=_tick_policy(),
        config=IntradaySimulationConfig(initial_cash=Decimal("1000"), max_trades_per_day=1),
        scenarios=[
            FrictionScenario(
                name="zero",
                fills=FillAssumptions(
                    slippage_bps_per_leg=Decimal("0"),
                    half_spread_bps_per_leg=Decimal("0"),
                ),
            ),
            FrictionScenario(
                name="stressed",
                fills=FillAssumptions(
                    slippage_bps_per_leg=Decimal("10"),
                    half_spread_bps_per_leg=Decimal("10"),
                ),
            ),
        ],
    )

    by_name = dict(result.scenarios)
    assert by_name["stressed"].metrics.net_pnl < by_name["zero"].metrics.net_pnl
    assert result.worst_net_pnl == by_name["stressed"].metrics.net_pnl


def test_stress_requires_named_explicit_scenarios() -> None:
    frame = _frame()
    signals = first_bar_hold_baseline(frame, session_open=time(9, 15))
    try:
        run_friction_stress(
            candidate_id="baseline",
            frame=frame,
            signals=signals,
            instrument_token="NSE_EQ|TEST",
            exchange=Exchange.NSE,
            cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
            session_policy=NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=10),
            tick_size_policy=_tick_policy(),
            config=IntradaySimulationConfig(initial_cash=Decimal("1000"), max_trades_per_day=1),
            scenarios=[],
        )
    except ValueError as exc:
        assert "at least one friction scenario" in str(exc)
    else:
        raise AssertionError("empty stress scenario set must fail")
