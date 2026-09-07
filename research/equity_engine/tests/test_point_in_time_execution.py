from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import FillAssumptions, IntradaySimulationConfig, simulate_long_intraday
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.models import Exchange


class _NeverEligible:
    def is_eligible(self, trade_date: date) -> bool:
        return False


class _AlwaysEligible:
    def is_eligible(self, trade_date: date) -> bool:
        return True


class _MissingTickEvidence:
    def tick_size(self, trade_date: date) -> Decimal:
        raise ValueError(f"no verified tick-size evidence for trade date {trade_date}")


def _frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-01 09:25", tz="Asia/Kolkata"),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10_000, 10_000, 10_000],
        },
        index=index,
    )


def _run(eligibility_policy):
    frame = _frame()
    return simulate_long_intraday(
        frame=frame,
        entries_at_close=pd.Series([True, False, False], index=frame.index),
        exits_at_close=pd.Series([False, False, False], index=frame.index),
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
        fills=FillAssumptions(
            slippage_bps_per_leg=Decimal("0"),
            half_spread_bps_per_leg=Decimal("0"),
        ),
        session_policy=NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=10),
        tick_size_policy=_MissingTickEvidence(),
        trading_eligibility_policy=eligibility_policy,
        config=IntradaySimulationConfig(initial_cash=Decimal("1000"), max_trades_per_day=1),
    )


def test_prelisting_ineligible_day_does_not_demand_nonexistent_tick_evidence() -> None:
    result = _run(_NeverEligible())
    assert result.trades == ()
    assert len(result.rejected_signals) == 1
    assert "exchange not eligible" in result.rejected_signals[0].reason


def test_eligible_entry_still_requires_verified_tick_evidence() -> None:
    with pytest.raises(ValueError, match="no verified tick-size evidence"):
        _run(_AlwaysEligible())
