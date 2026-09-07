from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import (
    ExitReason,
    FillAssumptions,
    IntradaySimulationConfig,
    simulate_long_intraday,
)
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.models import Exchange


def _frame(times: list[str], opens: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(value, tz="Asia/Kolkata") for value in times])
    return pd.DataFrame(
        {
            "open": opens,
            "high": [value + 1 for value in opens],
            "low": [value - 1 for value in opens],
            "close": [value + 0.5 for value in opens],
            "volume": [10_000] * len(opens),
        },
        index=index,
    )


def _provider() -> CurrentTermsNSEIntradayCostProvider:
    return CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))


def _fills() -> FillAssumptions:
    return FillAssumptions(
        slippage_bps_per_leg=Decimal("0"),
        half_spread_bps_per_leg=Decimal("0"),
        tick_size=Decimal("0.05"),
    )


def _session_policy() -> NSEEquitySessionPolicy:
    # Non-CAS stock: continuous session ends at 15:30. Explicit 10-minute research buffer
    # produces a 15:20 exit cutoff for these tests.
    return NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=10)


def _config() -> IntradaySimulationConfig:
    return IntradaySimulationConfig(initial_cash=Decimal("1000"), max_trades_per_day=2)


def test_signal_at_close_executes_at_next_bar_open_with_exact_costs() -> None:
    frame = _frame(
        [
            "2026-09-01 09:15",
            "2026-09-01 09:20",
            "2026-09-01 09:25",
            "2026-09-01 09:30",
        ],
        [100.0, 101.0, 103.0, 103.0],
    )
    entries = pd.Series([True, False, False, False], index=frame.index)
    exits = pd.Series([False, True, False, False], index=frame.index)

    result = simulate_long_intraday(
        frame=frame,
        entries_at_close=entries,
        exits_at_close=exits,
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        cost_provider=_provider(),
        fills=_fills(),
        session_policy=_session_policy(),
        config=_config(),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_timestamp == frame.index[1]
    assert trade.exit_timestamp == frame.index[2]
    assert trade.reference_entry_price == Decimal("101.0")
    assert trade.reference_exit_price == Decimal("103.0")
    assert trade.quantity == 9
    assert trade.exit_reason is ExitReason.SIGNAL
    assert result.final_cash == result.initial_cash + trade.net_pnl
    assert result.net_pnl == trade.net_pnl


def test_cutoff_forces_same_day_exit() -> None:
    frame = _frame(
        [
            "2026-09-01 09:15",
            "2026-09-01 09:20",
            "2026-09-01 15:20",
            "2026-09-01 15:25",
        ],
        [100.0, 100.0, 101.0, 101.0],
    )
    entries = pd.Series([True, False, False, False], index=frame.index)
    exits = pd.Series([False, False, False, False], index=frame.index)

    result = simulate_long_intraday(
        frame=frame,
        entries_at_close=entries,
        exits_at_close=exits,
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        cost_provider=_provider(),
        fills=_fills(),
        session_policy=_session_policy(),
        config=_config(),
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_timestamp == frame.index[2]
    assert result.trades[0].exit_reason is ExitReason.SESSION_CUTOFF


def test_yesterdays_final_signal_is_not_executed_at_next_days_open() -> None:
    frame = _frame(
        [
            "2026-09-01 15:20",
            "2026-09-01 15:25",
            "2026-09-02 09:15",
            "2026-09-02 09:20",
        ],
        [100.0, 100.0, 101.0, 101.0],
    )
    entries = pd.Series([False, True, False, False], index=frame.index)
    exits = pd.Series([False, False, False, False], index=frame.index)

    result = simulate_long_intraday(
        frame=frame,
        entries_at_close=entries,
        exits_at_close=exits,
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        cost_provider=_provider(),
        fills=_fills(),
        session_policy=_session_policy(),
        config=_config(),
    )

    assert result.trades == ()


def test_incomplete_session_fails_instead_of_hiding_overnight_position() -> None:
    frame = _frame(
        [
            "2026-09-01 09:15",
            "2026-09-01 09:20",
            "2026-09-01 09:25",
        ],
        [100.0, 100.0, 100.0],
    )
    entries = pd.Series([True, False, False], index=frame.index)
    exits = pd.Series([False, False, False], index=frame.index)

    with pytest.raises(ValueError, match="dataset ended with an open intraday position"):
        simulate_long_intraday(
            frame=frame,
            entries_at_close=entries,
            exits_at_close=exits,
            instrument_token="NSE_EQ|TEST",
            exchange=Exchange.NSE,
            cost_provider=_provider(),
            fills=_fills(),
            session_policy=_session_policy(),
            config=_config(),
        )
