from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pandas as pd

from .candidate_grid import StrategyDefinition
from .costs import CostProvider
from .event_simulator import (
    FillAssumptions,
    IntradaySimulationConfig,
    SessionExitResolver,
    TickSizeResolver,
    TradingEligibilityResolver,
)
from .models import Exchange
from .tournament import CandidateEvaluation, RankingMetric, evaluate_candidate_exact, rank_candidates


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    train_dates: tuple[date, ...]
    test_dates: tuple[date, ...]


@dataclass(frozen=True)
class WalkForwardResult:
    window: WalkForwardWindow
    selected_candidate_id: str
    train_evaluation: CandidateEvaluation
    test_evaluation: CandidateEvaluation


def make_walk_forward_windows(
    frame: pd.DataFrame,
    *,
    train_trading_days: int,
    test_trading_days: int,
    step_trading_days: int,
    embargo_trading_days: int,
) -> list[WalkForwardWindow]:
    """Create chronological train/test windows from actual trading dates in the dataset."""

    for name, value in (
        ("train_trading_days", train_trading_days),
        ("test_trading_days", test_trading_days),
        ("step_trading_days", step_trading_days),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if embargo_trading_days < 0:
        raise ValueError("embargo_trading_days cannot be negative")
    if frame.empty:
        raise ValueError("frame cannot be empty")
    if frame.index.tz is None:
        raise ValueError("walk-forward frame requires timezone-aware timestamps")

    trading_dates = tuple(sorted(set(frame.index.date)))
    windows: list[WalkForwardWindow] = []
    start = 0
    window_id = 1
    while True:
        train_end = start + train_trading_days
        test_start = train_end + embargo_trading_days
        test_end = test_start + test_trading_days
        if test_end > len(trading_dates):
            break
        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                train_dates=trading_dates[start:train_end],
                test_dates=trading_dates[test_start:test_end],
            )
        )
        start += step_trading_days
        window_id += 1

    if not windows:
        raise ValueError("dataset is too short for requested walk-forward schedule")
    return windows


def _slice_dates(frame: pd.DataFrame, dates: tuple[date, ...]) -> pd.DataFrame:
    allowed = set(dates)
    return frame[[d in allowed for d in frame.index.date]].copy()


def run_walk_forward_selection(
    *,
    frame: pd.DataFrame,
    windows: list[WalkForwardWindow],
    candidates: list[StrategyDefinition],
    ranking_metric: RankingMetric,
    instrument_token: str,
    exchange: Exchange,
    cost_provider: CostProvider,
    fills: FillAssumptions,
    session_policy: SessionExitResolver,
    tick_size_policy: TickSizeResolver,
    trading_eligibility_policy: TradingEligibilityResolver,
    simulation_config: IntradaySimulationConfig,
) -> list[WalkForwardResult]:
    """Select only on train data, freeze candidate, then evaluate untouched test data.

    The same point-in-time trading-eligibility resolver is used for both train and test windows;
    the simulator queries it for each actual entry date and fails if evidence is missing.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")

    results: list[WalkForwardResult] = []
    for window in windows:
        train_frame = _slice_dates(frame, window.train_dates)
        test_frame = _slice_dates(frame, window.test_dates)

        train_evaluations: list[CandidateEvaluation] = []
        definition_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        for candidate in candidates:
            signals = candidate.build_signals(train_frame)
            train_evaluations.append(
                evaluate_candidate_exact(
                    candidate_id=candidate.candidate_id,
                    frame=train_frame,
                    signals=signals,
                    instrument_token=instrument_token,
                    exchange=exchange,
                    cost_provider=cost_provider,
                    fills=fills,
                    session_policy=session_policy,
                    tick_size_policy=tick_size_policy,
                    trading_eligibility_policy=trading_eligibility_policy,
                    config=simulation_config,
                )
            )

        ranked = rank_candidates(train_evaluations, metric=ranking_metric)
        selected_train = ranked[0]
        selected_definition = definition_by_id[selected_train.candidate_id]
        test_signals = selected_definition.build_signals(test_frame)
        selected_test = evaluate_candidate_exact(
            candidate_id=selected_definition.candidate_id,
            frame=test_frame,
            signals=test_signals,
            instrument_token=instrument_token,
            exchange=exchange,
            cost_provider=cost_provider,
            fills=fills,
            session_policy=session_policy,
            tick_size_policy=tick_size_policy,
            trading_eligibility_policy=trading_eligibility_policy,
            config=simulation_config,
        )
        results.append(
            WalkForwardResult(
                window=window,
                selected_candidate_id=selected_definition.candidate_id,
                train_evaluation=selected_train,
                test_evaluation=selected_test,
            )
        )

    return results


def aggregate_test_net_return_pct(results: list[WalkForwardResult]) -> Decimal:
    if not results:
        raise ValueError("results cannot be empty")
    return sum((item.test_evaluation.metrics.net_return_pct for item in results), Decimal("0"))
