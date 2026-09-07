from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from decimal import Decimal
from typing import Mapping

import pandas as pd

from .provenance import validate_ohlcv_frame

_BPS = Decimal("10000")


@dataclass(frozen=True)
class StrategySignals:
    name: str
    entries_at_close: pd.Series
    exits_at_close: pd.Series
    parameters: Mapping[str, str]
    source_refs: tuple[str, ...]


def _validate_frame(frame: pd.DataFrame) -> None:
    violations = validate_ohlcv_frame(frame)
    if violations:
        raise ValueError("invalid OHLCV frame: " + "; ".join(violations))
    if frame.index.tz is None:
        raise ValueError("intraday strategies require timezone-aware timestamps")


def _day_key(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.date, index=index)


def _rising_edge(condition: pd.Series, day: pd.Series) -> pd.Series:
    previous = condition.groupby(day).shift(1, fill_value=False)
    return condition.astype(bool) & ~previous.astype(bool)


def first_bar_hold_baseline(
    frame: pd.DataFrame,
    *,
    session_open: time,
) -> StrategySignals:
    """Signal once per day at the first continuous-session bar close.

    The event simulator executes the signal on the next bar and handles the session exit. This is
    deliberately simple: every more complex candidate must beat it after identical costs.
    """

    _validate_frame(frame)
    times = pd.Series(frame.index.time, index=frame.index)
    day = _day_key(frame.index)
    at_or_after_open = times >= session_open
    ordinal = at_or_after_open.groupby(day).cumsum()
    entries = at_or_after_open & ordinal.eq(1)
    exits = pd.Series(False, index=frame.index, dtype=bool)
    return StrategySignals(
        name="first_bar_hold_baseline",
        entries_at_close=entries.astype(bool),
        exits_at_close=exits,
        parameters={"session_open": session_open.isoformat()},
        source_refs=(),
    )


def opening_range_breakout(
    frame: pd.DataFrame,
    *,
    session_open: time,
    bar_minutes: int,
    opening_range_minutes: int,
    breakout_buffer_bps: Decimal,
    min_volume_ratio: Decimal,
) -> StrategySignals:
    """Long ORB with explicit opening-range and volume definitions.

    Operational definition used here:
    - Opening range: bars whose *start timestamps* are in [session_open, range_end).
    - Breakout: close crosses above opening-range high plus the configured buffer.
    - Volume filter: breakout-bar volume / mean opening-range bar volume.
    - Exit signal: close falls back to/below the unbuffered opening-range high.

    The range lengths and 1.2x/1.5x volume ratios are research candidates, not claims of edge.
    """

    _validate_frame(frame)
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    if opening_range_minutes <= 0 or opening_range_minutes % bar_minutes != 0:
        raise ValueError("opening_range_minutes must be a positive multiple of bar_minutes")
    if breakout_buffer_bps < 0:
        raise ValueError("breakout_buffer_bps cannot be negative")
    if min_volume_ratio <= 0:
        raise ValueError("min_volume_ratio must be positive")

    day = _day_key(frame.index)
    minute_of_day = pd.Series(
        [ts.hour * 60 + ts.minute for ts in frame.index],
        index=frame.index,
    )
    open_minute = session_open.hour * 60 + session_open.minute
    range_end_minute = open_minute + opening_range_minutes
    range_mask = minute_of_day.ge(open_minute) & minute_of_day.lt(range_end_minute)
    post_range = minute_of_day.ge(range_end_minute)

    expected_bars = opening_range_minutes // bar_minutes
    range_counts = range_mask.groupby(day).sum()
    bad_days = range_counts[range_counts != expected_bars]
    if not bad_days.empty:
        raise ValueError(
            "opening range is incomplete for trading days: "
            + ", ".join(f"{d}={int(c)} bars" for d, c in bad_days.items())
        )

    range_high_by_day = frame.loc[range_mask, "high"].groupby(day[range_mask]).max()
    range_volume_by_day = frame.loc[range_mask, "volume"].groupby(day[range_mask]).mean()
    range_high = day.map(range_high_by_day).astype(float)
    opening_volume_mean = day.map(range_volume_by_day).astype(float)

    buffer_multiplier = 1.0 + float(breakout_buffer_bps / _BPS)
    threshold = range_high * buffer_multiplier
    volume_ratio = frame["volume"].astype(float) / opening_volume_mean
    condition = post_range & frame["close"].gt(threshold) & volume_ratio.ge(float(min_volume_ratio))
    entries = _rising_edge(condition, day)
    exits = post_range & frame["close"].le(range_high)

    return StrategySignals(
        name="opening_range_breakout",
        entries_at_close=entries.astype(bool),
        exits_at_close=exits.astype(bool),
        parameters={
            "session_open": session_open.isoformat(),
            "bar_minutes": str(bar_minutes),
            "opening_range_minutes": str(opening_range_minutes),
            "breakout_buffer_bps": str(breakout_buffer_bps),
            "min_volume_ratio": str(min_volume_ratio),
            "volume_definition": "breakout_bar_volume/opening_range_mean_bar_volume",
        },
        source_refs=(
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5198458",
            "https://journals.sagepub.com/doi/10.1177/0972652720930586",
        ),
    )


def momentum_breakout(
    frame: pd.DataFrame,
    *,
    lookback_bars: int,
    entry_return_bps: Decimal,
    exit_return_bps: Decimal,
    volume_lookback_bars: int,
    min_volume_ratio: Decimal,
) -> StrategySignals:
    """Long time-series momentum candidate with a prior-volume baseline."""

    _validate_frame(frame)
    if lookback_bars <= 0 or volume_lookback_bars <= 0:
        raise ValueError("lookback values must be positive")
    if entry_return_bps <= 0:
        raise ValueError("entry_return_bps must be positive")
    if min_volume_ratio <= 0:
        raise ValueError("min_volume_ratio must be positive")

    day = _day_key(frame.index)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)

    prior_close = close.groupby(day).shift(lookback_bars)
    momentum = close / prior_close - 1.0

    def prior_volume_median(group: pd.Series) -> pd.Series:
        return group.shift(1).rolling(
            window=volume_lookback_bars,
            min_periods=volume_lookback_bars,
        ).median()

    volume_base = volume.groupby(day, group_keys=False).apply(prior_volume_median)
    volume_ratio = volume / volume_base

    entry_threshold = float(entry_return_bps / _BPS)
    exit_threshold = float(exit_return_bps / _BPS)
    condition = momentum.ge(entry_threshold) & volume_ratio.ge(float(min_volume_ratio))
    entries = _rising_edge(condition.fillna(False), day)
    exits = momentum.le(exit_threshold).fillna(False)

    return StrategySignals(
        name="momentum_breakout",
        entries_at_close=entries.astype(bool),
        exits_at_close=exits.astype(bool),
        parameters={
            "lookback_bars": str(lookback_bars),
            "entry_return_bps": str(entry_return_bps),
            "exit_return_bps": str(exit_return_bps),
            "volume_lookback_bars": str(volume_lookback_bars),
            "min_volume_ratio": str(min_volume_ratio),
        },
        source_refs=(
            "https://nsearchives.nseindia.com/web/sites/default/files/inline-files/Market%20Pulse%20June%202022.pdf",
        ),
    )


def trend_pullback(
    frame: pd.DataFrame,
    *,
    fast_span: int,
    slow_span: int,
    pullback_tolerance_bps: Decimal,
    exit_buffer_bps: Decimal,
) -> StrategySignals:
    """Long trend-pullback challenger with explicit EMA and exit rules."""

    _validate_frame(frame)
    if fast_span <= 0 or slow_span <= 0 or fast_span >= slow_span:
        raise ValueError("require 0 < fast_span < slow_span")
    if pullback_tolerance_bps < 0 or exit_buffer_bps < 0:
        raise ValueError("buffer values cannot be negative")

    day = _day_key(frame.index)
    close = frame["close"].astype(float)

    fast = close.groupby(day, group_keys=False).apply(
        lambda s: s.ewm(span=fast_span, adjust=False, min_periods=fast_span).mean()
    )
    slow = close.groupby(day, group_keys=False).apply(
        lambda s: s.ewm(span=slow_span, adjust=False, min_periods=slow_span).mean()
    )

    tolerance = float(pullback_tolerance_bps / _BPS)
    exit_buffer = float(exit_buffer_bps / _BPS)
    trend_up = fast.gt(slow)
    in_pullback_zone = close.le(fast * (1.0 + tolerance)) & close.ge(slow)
    condition = trend_up & in_pullback_zone
    entries = _rising_edge(condition.fillna(False), day)
    exits = (fast.le(slow) | close.lt(slow * (1.0 - exit_buffer))).fillna(False)

    return StrategySignals(
        name="trend_pullback",
        entries_at_close=entries.astype(bool),
        exits_at_close=exits.astype(bool),
        parameters={
            "fast_span": str(fast_span),
            "slow_span": str(slow_span),
            "pullback_tolerance_bps": str(pullback_tolerance_bps),
            "exit_buffer_bps": str(exit_buffer_bps),
        },
        source_refs=(),
    )


def mean_reversion_zscore(
    frame: pd.DataFrame,
    *,
    lookback_bars: int,
    entry_z: Decimal,
    exit_z: Decimal,
) -> StrategySignals:
    """Long-only rolling z-score mean-reversion challenger."""

    _validate_frame(frame)
    if lookback_bars < 2:
        raise ValueError("lookback_bars must be at least 2")
    if entry_z <= 0:
        raise ValueError("entry_z must be positive; entry occurs at z <= -entry_z")

    day = _day_key(frame.index)
    close = frame["close"].astype(float)

    def zscore(group: pd.Series) -> pd.Series:
        mean = group.rolling(lookback_bars, min_periods=lookback_bars).mean()
        std = group.rolling(lookback_bars, min_periods=lookback_bars).std(ddof=0)
        return (group - mean) / std.replace(0.0, float("nan"))

    z = close.groupby(day, group_keys=False).apply(zscore)
    condition = z.le(-float(entry_z))
    entries = _rising_edge(condition.fillna(False), day)
    exits = z.ge(float(exit_z)).fillna(False)

    return StrategySignals(
        name="mean_reversion_zscore",
        entries_at_close=entries.astype(bool),
        exits_at_close=exits.astype(bool),
        parameters={
            "lookback_bars": str(lookback_bars),
            "entry_z": str(entry_z),
            "exit_z": str(exit_z),
        },
        source_refs=("https://openalgo.in/quant/intraday-mean-reversion-breakout",),
    )
