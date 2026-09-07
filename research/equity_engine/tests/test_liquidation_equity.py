from datetime import date, time
from decimal import Decimal

import pandas as pd

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import FillAssumptions, IntradaySimulationConfig
from equity_engine.liquidation_equity import EquityObservation, liquidation_drawdown_metrics
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.models import Exchange
from equity_engine.strategies import first_bar_hold_baseline
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.tournament import evaluate_candidate_exact


class _Eligible:
    def is_eligible(self, trade_date: date) -> bool:
        return True


def _frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-01 09:25", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-01 15:20", tz="Asia/Kolkata"),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 50.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [100_000] * 4,
        },
        index=index,
    )


def test_ohlc_low_stress_exposes_intratrade_risk_hidden_by_realized_drawdown() -> None:
    frame = _frame()
    signals = first_bar_hold_baseline(frame, session_open=time(9, 15))
    evaluation = evaluate_candidate_exact(
        candidate_id="baseline",
        frame=frame,
        signals=signals,
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
        trading_eligibility_policy=_Eligible(),
        config=IntradaySimulationConfig(
            initial_cash=Decimal("1000"),
            max_trades_per_day=1,
        ),
    )

    metrics = evaluation.metrics
    assert metrics.ohlc_low_liquidation_stress_max_drawdown_pct > Decimal("40")
    assert (
        metrics.ohlc_low_liquidation_stress_max_drawdown_pct
        > metrics.close_liquidation_max_drawdown_pct
    )
    assert (
        metrics.ohlc_low_liquidation_stress_max_drawdown_pct
        > metrics.realized_max_drawdown_pct
    )

    active = [item for item in evaluation.equity_curve if item.position_quantity > 0]
    assert active
    assert min(item.ohlc_low_liquidation_stress_equity for item in active) < Decimal("600")
    assert evaluation.equity_curve[-1].position_quantity == 0
    assert evaluation.equity_curve[-1].close_liquidation_equity == evaluation.simulation.final_cash


def test_close_liquidation_curve_includes_modeled_exit_cost_before_trade_is_closed() -> None:
    frame = _frame()
    signals = first_bar_hold_baseline(frame, session_open=time(9, 15))
    evaluation = evaluate_candidate_exact(
        candidate_id="baseline",
        frame=frame,
        signals=signals,
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
        trading_eligibility_policy=_Eligible(),
        config=IntradaySimulationConfig(
            initial_cash=Decimal("1000"),
            max_trades_per_day=1,
        ),
    )

    entry_mark = next(item for item in evaluation.equity_curve if item.position_quantity > 0)
    assert entry_mark.close_liquidation_equity < Decimal("1000")
    assert evaluation.metrics.close_liquidation_max_drawdown_pct > Decimal("0")


def test_low_stress_uses_previous_close_peak_without_using_same_bar_close_retroactively() -> None:
    observations = (
        EquityObservation(
            timestamp=pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata"),
            cash_on_hand=Decimal("100"),
            position_quantity=1,
            close_liquidation_equity=Decimal("1200"),
            ohlc_low_liquidation_stress_equity=Decimal("1000"),
        ),
        EquityObservation(
            timestamp=pd.Timestamp("2026-09-01 09:25", tz="Asia/Kolkata"),
            cash_on_hand=Decimal("100"),
            position_quantity=1,
            close_liquidation_equity=Decimal("1100"),
            ohlc_low_liquidation_stress_equity=Decimal("900"),
        ),
    )

    metrics = liquidation_drawdown_metrics(observations, initial_equity=Decimal("1000"))

    # First bar low is compared with the initial 1000, not its later 1200 close. The 1200 close
    # then becomes the peak for the next bar, so 900 is a 25% low-stress drawdown.
    assert metrics.ohlc_low_liquidation_stress_max_drawdown_pct == Decimal("25.00")
