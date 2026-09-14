import pandas as pd
import pytest
from decimal import Decimal

from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.vectorbt_screening import shift_close_generated_signals


def test_close_generated_signal_moves_to_next_bar() -> None:
    index = pd.date_range("2026-09-01 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    entries = pd.Series([True, False, False, False], index=index)
    exits = pd.Series([False, False, True, False], index=index)

    shifted_entries, shifted_exits = shift_close_generated_signals(
        entries,
        exits,
        lag_bars=1,
    )

    assert shifted_entries.tolist() == [False, True, False, False]
    assert shifted_exits.tolist() == [False, False, False, True]


def test_final_signal_does_not_cross_into_next_trading_day() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-01 15:25", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-02 09:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-02 09:20", tz="Asia/Kolkata"),
        ]
    )
    entries = pd.Series([True, False, False], index=index)
    exits = pd.Series([False, False, False], index=index)

    shifted_entries, _ = shift_close_generated_signals(entries, exits, lag_bars=1)

    assert shifted_entries.tolist() == [False, False, False]


def test_same_bar_close_execution_is_forbidden() -> None:
    index = pd.date_range("2026-09-01 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
    entries = pd.Series([True, False], index=index)
    exits = pd.Series([False, True], index=index)

    with pytest.raises(ValueError):
        shift_close_generated_signals(entries, exits, lag_bars=0)


def test_naive_intraday_timestamps_are_forbidden() -> None:
    index = pd.date_range("2026-09-01 09:15", periods=2, freq="5min")
    entries = pd.Series([True, False], index=index)
    exits = pd.Series([False, True], index=index)

    with pytest.raises(ValueError, match="timezone-aware"):
        shift_close_generated_signals(entries, exits, lag_bars=1)


def test_vectorbt_screening_input_excludes_cas_auxiliary_bars(monkeypatch) -> None:
    import vectorbt as vbt

    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-08 15:10", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-08 15:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-08 15:20", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-08 15:25", tz="Asia/Kolkata"),
        ]
    )
    close = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
    execution = close.copy()
    entries = pd.Series([False, True, False, False], index=index)
    exits = pd.Series([False, False, True, False], index=index)
    captured = {}

    class _Portfolio:
        @staticmethod
        def from_signals(close, **kwargs):
            captured["close_index"] = close.index
            captured["entries_index"] = kwargs["entries"].index
            return _Portfolio()

        def stats(self, settings):
            return {"Total Return [%]": 0.0, "Max Drawdown [%]": 0.0}

    monkeypatch.setattr(vbt, "Portfolio", _Portfolio)
    from equity_engine.vectorbt_screening import screen_long_signals

    screen_long_signals(
        close=close,
        execution_price=execution,
        entries_at_close=entries,
        exits_at_close=exits,
        signal_lag_bars=1,
        screening_cash=Decimal("1000"),
        screening_fee_rate=Decimal("0"),
        screening_slippage_rate=Decimal("0"),
        frequency="5min",
        session_policy=NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=5),
    )

    assert captured["close_index"].tolist() == [index[0]]
    assert captured["entries_index"].tolist() == [index[0]]
