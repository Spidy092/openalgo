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
from equity_engine.tick_size import FixedTickSizePolicy


class _SyntheticEligibilityPolicy:
    """Explicit unit-test policy; production research uses point-in-time exchange evidence."""

    def __init__(self, *, ineligible_dates: set[date] | None = None) -> None:
        self._ineligible_dates = ineligible_dates or set()

    def is_eligible(self, trade_date: date) -> bool:
        return trade_date not in self._ineligible_dates


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
    )


def _tick_policy() -> FixedTickSizePolicy:
    return FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="synthetic-test")


def _eligibility_policy(
    *, ineligible_dates: set[date] | None = None
) -> _SyntheticEligibilityPolicy:
    return _SyntheticEligibilityPolicy(ineligible_dates=ineligible_dates)


def _session_policy() -> NSEEquitySessionPolicy:
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
        tick_size_policy=_tick_policy(),
        trading_eligibility_policy=_eligibility_policy(),
        config=_config(),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_timestamp == frame.index[1]
    assert trade.exit_timestamp == frame.index[2]
    assert trade.reference_entry_price == Decimal("101.0")
    assert trade.reference_exit_price == Decimal("103.0")
    assert trade.quantity == 9
    assert trade.entry_tick_size_rupees == Decimal("0.05")
    assert trade.exit_tick_size_rupees == Decimal("0.05")
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
        tick_size_policy=_tick_policy(),
        trading_eligibility_policy=_eligibility_policy(),
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
        tick_size_policy=_tick_policy(),
        trading_eligibility_policy=_eligibility_policy(),
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
            tick_size_policy=_tick_policy(),
            trading_eligibility_policy=_eligibility_policy(),
            config=_config(),
        )


def test_ineligible_trade_date_rejects_entry_signal() -> None:
    trade_day = date(2026, 9, 1)
    frame = _frame(
        [
            "2026-09-01 09:15",
            "2026-09-01 09:20",
            "2026-09-01 09:25",
        ],
        [100.0, 101.0, 102.0],
    )
    entries = pd.Series([True, False, False], index=frame.index)
    exits = pd.Series([False, False, False], index=frame.index)

    result = simulate_long_intraday(
        frame=frame,
        entries_at_close=entries,
        exits_at_close=exits,
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        cost_provider=_provider(),
        fills=_fills(),
        session_policy=_session_policy(),
        tick_size_policy=_tick_policy(),
        trading_eligibility_policy=_eligibility_policy(ineligible_dates={trade_day}),
        config=_config(),
    )

    assert result.trades == ()
    assert len(result.rejected_signals) == 1
    assert "exchange not eligible" in result.rejected_signals[0].reason


def test_cas_auxiliary_bar_cannot_be_used_as_simulator_fill() -> None:
    frame = _frame(
        [
            "2026-09-08 09:15",
            "2026-09-08 09:20",
            "2026-09-08 15:10",
            "2026-09-08 15:15",
        ],
        [100.0, 100.0, 101.0, 999.0],
    )
    entries = pd.Series([True, False, False, False], index=frame.index)
    exits = pd.Series([False, False, False, False], index=frame.index)
    cas_policy = NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0)

    with pytest.raises(ValueError, match="non-continuous-session bars"):
        simulate_long_intraday(
            frame=frame,
            entries_at_close=entries,
            exits_at_close=exits,
            instrument_token="NSE_EQ|TEST",
            exchange=Exchange.NSE,
            cost_provider=_provider(),
            fills=_fills(),
            session_policy=cas_policy,
            tick_size_policy=_tick_policy(),
            trading_eligibility_policy=_eligibility_policy(),
            config=_config(),
        )


def test_cas_without_safe_pre_end_bar_fails_closed() -> None:
    frame = _frame(
        ["2026-09-08 09:15", "2026-09-08 09:20", "2026-09-08 15:10"],
        [100.0, 100.0, 101.0],
    )
    entries = pd.Series([True, False, False], index=frame.index)
    exits = pd.Series([False, False, False], index=frame.index)

    with pytest.raises(ValueError, match="no safe continuous-session exit bar"):
        simulate_long_intraday(
            frame=frame,
            entries_at_close=entries,
            exits_at_close=exits,
            instrument_token="NSE_EQ|TEST",
            exchange=Exchange.NSE,
            cost_provider=_provider(),
            fills=_fills(),
            session_policy=NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=0),
            tick_size_policy=_tick_policy(),
            trading_eligibility_policy=_eligibility_policy(),
            config=_config(),
        )


def test_buffered_cas_exit_uses_safe_continuous_bar() -> None:
    frame = _frame(
        [
            "2026-09-08 09:15",
            "2026-09-08 09:20",
            "2026-09-08 15:05",
            "2026-09-08 15:10",
        ],
        [100.0, 100.0, 101.0, 102.0],
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
        session_policy=NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=5),
        tick_size_policy=_tick_policy(),
        trading_eligibility_policy=_eligibility_policy(),
        config=_config(),
    )

    assert result.trades[0].exit_timestamp == frame.index[-1]
    assert result.trades[0].reference_exit_price == Decimal("102.0")
