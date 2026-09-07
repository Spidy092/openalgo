from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from decimal import Decimal
from functools import partial
from typing import Callable, Iterable

import pandas as pd

from .strategies import (
    StrategySignals,
    first_bar_hold_baseline,
    mean_reversion_zscore,
    momentum_breakout,
    opening_range_breakout,
    trend_pullback,
)


@dataclass(frozen=True)
class StrategyDefinition:
    candidate_id: str
    build_signals: Callable[[pd.DataFrame], StrategySignals]
    research_basis: str
    source_refs: tuple[str, ...]


def baseline_definition(*, session_open: time) -> StrategyDefinition:
    return StrategyDefinition(
        candidate_id="baseline:first-bar-hold",
        build_signals=partial(first_bar_hold_baseline, session_open=session_open),
        research_basis="Deliberately simple same-cost baseline",
        source_refs=(),
    )


def orb_study_grid(
    *,
    session_open: time,
    bar_minutes: int,
    breakout_buffer_bps: Decimal,
) -> list[StrategyDefinition]:
    """Build the initial ORB parameter grid from the 2025 NSE study values.

    The cited study examined 5/15/30-minute opening ranges and 1.2x/1.5x volume thresholds.
    `breakout_buffer_bps` is caller-supplied because the study evidence does not justify silently
    choosing our implementation's breakout buffer.
    """

    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    if breakout_buffer_bps < 0:
        raise ValueError("breakout_buffer_bps cannot be negative")

    ranges = (5, 15, 30)
    volume_ratios = (Decimal("1.2"), Decimal("1.5"))
    definitions: list[StrategyDefinition] = []
    for opening_range_minutes in ranges:
        if opening_range_minutes % bar_minutes != 0:
            continue
        for volume_ratio in volume_ratios:
            candidate_id = (
                f"orb:{opening_range_minutes}m:vol{volume_ratio}:buf{breakout_buffer_bps}bps"
            )
            definitions.append(
                StrategyDefinition(
                    candidate_id=candidate_id,
                    build_signals=partial(
                        opening_range_breakout,
                        session_open=session_open,
                        bar_minutes=bar_minutes,
                        opening_range_minutes=opening_range_minutes,
                        breakout_buffer_bps=breakout_buffer_bps,
                        min_volume_ratio=volume_ratio,
                    ),
                    research_basis=(
                        "NSE ORB study range/volume grid; our volume denominator and breakout "
                        "buffer are explicitly documented implementation choices"
                    ),
                    source_refs=(
                        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5198458",
                    ),
                )
            )
    if not definitions:
        raise ValueError("bar_minutes is incompatible with all evidence-seeded ORB ranges")
    return definitions


def momentum_grid(
    *,
    lookback_bars: Iterable[int],
    entry_return_bps: Iterable[Decimal],
    exit_return_bps: Iterable[Decimal],
    volume_lookback_bars: Iterable[int],
    min_volume_ratios: Iterable[Decimal],
) -> list[StrategyDefinition]:
    """Caller-defined momentum grid; no literature parameters are silently invented."""

    definitions: list[StrategyDefinition] = []
    for lb in lookback_bars:
        for entry_bps in entry_return_bps:
            for exit_bps in exit_return_bps:
                for vol_lb in volume_lookback_bars:
                    for vol_ratio in min_volume_ratios:
                        candidate_id = (
                            f"mom:lb{lb}:entry{entry_bps}:exit{exit_bps}:"
                            f"vollb{vol_lb}:vol{vol_ratio}"
                        )
                        definitions.append(
                            StrategyDefinition(
                                candidate_id=candidate_id,
                                build_signals=partial(
                                    momentum_breakout,
                                    lookback_bars=lb,
                                    entry_return_bps=entry_bps,
                                    exit_return_bps=exit_bps,
                                    volume_lookback_bars=vol_lb,
                                    min_volume_ratio=vol_ratio,
                                ),
                                research_basis=(
                                    "Intraday momentum challenger; parameter values supplied "
                                    "explicitly by the research experiment"
                                ),
                                source_refs=(
                                    "https://nsearchives.nseindia.com/web/sites/default/files/inline-files/Market%20Pulse%20June%202022.pdf",
                                ),
                            )
                        )
    return definitions


def trend_pullback_grid(
    *,
    fast_spans: Iterable[int],
    slow_spans: Iterable[int],
    pullback_tolerance_bps: Iterable[Decimal],
    exit_buffer_bps: Iterable[Decimal],
) -> list[StrategyDefinition]:
    definitions: list[StrategyDefinition] = []
    for fast in fast_spans:
        for slow in slow_spans:
            if fast >= slow:
                continue
            for tolerance in pullback_tolerance_bps:
                for exit_buffer in exit_buffer_bps:
                    candidate_id = (
                        f"trend:ema{fast}-{slow}:pull{tolerance}:exit{exit_buffer}"
                    )
                    definitions.append(
                        StrategyDefinition(
                            candidate_id=candidate_id,
                            build_signals=partial(
                                trend_pullback,
                                fast_span=fast,
                                slow_span=slow,
                                pullback_tolerance_bps=tolerance,
                                exit_buffer_bps=exit_buffer,
                            ),
                            research_basis="Caller-defined trend-pullback challenger",
                            source_refs=(),
                        )
                    )
    return definitions


def mean_reversion_grid(
    *,
    lookback_bars: Iterable[int],
    entry_z_values: Iterable[Decimal],
    exit_z_values: Iterable[Decimal],
) -> list[StrategyDefinition]:
    definitions: list[StrategyDefinition] = []
    for lb in lookback_bars:
        for entry_z in entry_z_values:
            for exit_z in exit_z_values:
                candidate_id = f"meanrev:lb{lb}:entry{entry_z}:exit{exit_z}"
                definitions.append(
                    StrategyDefinition(
                        candidate_id=candidate_id,
                        build_signals=partial(
                            mean_reversion_zscore,
                            lookback_bars=lb,
                            entry_z=entry_z,
                            exit_z=exit_z,
                        ),
                        research_basis="Caller-defined rolling-z-score mean-reversion challenger",
                        source_refs=(
                            "https://openalgo.in/quant/intraday-mean-reversion-breakout",
                        ),
                    )
                )
    return definitions
