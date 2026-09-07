import pandas as pd
import pytest

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
