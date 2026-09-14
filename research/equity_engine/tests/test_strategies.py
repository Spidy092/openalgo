from datetime import time
from decimal import Decimal

import pandas as pd
import pytest

from equity_engine.strategies import (
    first_bar_hold_baseline,
    mean_reversion_zscore,
    momentum_breakout,
    opening_range_breakout,
    trend_pullback,
)


def _frame(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[int],
    *,
    start: str = "2026-09-01 09:15",
) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(opens), freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )


def test_first_bar_baseline_signals_once_at_session_start() -> None:
    frame = _frame(
        [100, 101, 102, 103],
        [101, 102, 103, 104],
        [99, 100, 101, 102],
        [100.5, 101.5, 102.5, 103.5],
        [100, 100, 100, 100],
    )
    signals = first_bar_hold_baseline(frame, session_open=time(9, 15))
    assert signals.entries_at_close.tolist() == [True, False, False, False]
    assert not signals.exits_at_close.any()


def test_orb_waits_for_complete_range_then_requires_volume() -> None:
    frame = _frame(
        [100, 101, 102, 103, 104, 103],
        [101, 102, 103, 104, 105, 104],
        [99, 100, 101, 102, 103, 102],
        [100.5, 101.5, 102.5, 103.5, 104.5, 102.8],
        [100, 100, 100, 150, 100, 100],
    )
    signals = opening_range_breakout(
        frame,
        session_open=time(9, 15),
        bar_minutes=5,
        opening_range_minutes=15,
        breakout_buffer_bps=Decimal("0"),
        min_volume_ratio=Decimal("1.2"),
    )

    # First three 5-minute bars form 09:15-09:30 range. The 09:30 close breaks the
    # range high with 1.5x the opening-range average volume.
    assert signals.entries_at_close.tolist() == [False, False, False, True, False, False]
    assert signals.exits_at_close.iloc[-1]


def test_orb_refuses_incomplete_opening_range() -> None:
    frame = _frame(
        [100, 101],
        [101, 102],
        [99, 100],
        [100.5, 101.5],
        [100, 100],
    )
    with pytest.raises(ValueError, match="opening range is incomplete"):
        opening_range_breakout(
            frame,
            session_open=time(9, 15),
            bar_minutes=5,
            opening_range_minutes=15,
            breakout_buffer_bps=Decimal("0"),
            min_volume_ratio=Decimal("1.2"),
        )


def test_momentum_uses_only_same_day_prior_bars() -> None:
    frame = _frame(
        [100, 100, 100, 102, 103, 103],
        [101, 101, 101, 103, 104, 104],
        [99, 99, 99, 101, 102, 102],
        [100, 100, 100, 102, 103, 103],
        [100, 100, 100, 150, 100, 100],
    )
    signals = momentum_breakout(
        frame,
        lookback_bars=2,
        entry_return_bps=Decimal("100"),
        exit_return_bps=Decimal("0"),
        volume_lookback_bars=2,
        min_volume_ratio=Decimal("1.2"),
    )
    assert signals.entries_at_close.iloc[3]


def test_trend_pullback_and_mean_reversion_return_index_aligned_signals() -> None:
    frame = _frame(
        [100, 101, 102, 103, 102, 101, 102, 103],
        [101, 102, 103, 104, 103, 102, 103, 104],
        [99, 100, 101, 102, 101, 100, 101, 102],
        [100, 101, 102, 103, 102, 101, 102, 103],
        [100] * 8,
    )
    trend = trend_pullback(
        frame,
        fast_span=2,
        slow_span=4,
        pullback_tolerance_bps=Decimal("25"),
        exit_buffer_bps=Decimal("0"),
    )
    reversion = mean_reversion_zscore(
        frame,
        lookback_bars=3,
        entry_z=Decimal("1"),
        exit_z=Decimal("0"),
    )

    assert trend.entries_at_close.index.equals(frame.index)
    assert trend.exits_at_close.index.equals(frame.index)
    assert reversion.entries_at_close.index.equals(frame.index)
    assert reversion.exits_at_close.index.equals(frame.index)
