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
from .market_sessions import filter_to_continuous_session
from .models import Exchange
from .tournament import (
    CandidateEvaluation,
    RankingMetric,
    evaluate_candidate_exact,
    rank_candidates,
)
from .wfo_schedule import plan_wfo_date_windows


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
    """Create chronological train/test windows from actual trading dates in the dataset.

    Delegates fold arithmetic to :mod:`wfo_schedule` so plan-only compilation and
    frame-based execution share one scheduler.
    """

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
    try:
        folds = plan_wfo_date_windows(
            trading_dates,
            train_trading_days=train_trading_days,
            test_trading_days=test_trading_days,
            step_trading_days=step_trading_days,
            embargo_trading_days=embargo_trading_days,
        )
    except ValueError as exc:
        raise ValueError("dataset is too short for requested walk-forward schedule") from exc
    return [
        WalkForwardWindow(
            window_id=fold.window_id,
            train_dates=fold.train_dates,
            test_dates=fold.test_dates,
        )
        for fold in folds
    ]


def _slice_dates(frame: pd.DataFrame, dates: tuple[date, ...]) -> pd.DataFrame:
    allowed = set(dates)
    return frame[[d in allowed for d in frame.index.date]].copy()


def _filter_to_trading_eligible_dates(
    frame: pd.DataFrame,
    *,
    trading_eligibility_policy: TradingEligibilityResolver,
    window_id: int,
    phase: str,
) -> pd.DataFrame:
    """Remove exchange-ineligible dates before any strategy can observe their bars."""

    if phase not in {"train", "test"}:
        raise ValueError("phase must be train or test")
    dates = tuple(sorted(set(frame.index.date)))
    eligible_dates = {
        trade_date for trade_date in dates if trading_eligibility_policy.is_eligible(trade_date)
    }
    filtered = frame[[d in eligible_dates for d in frame.index.date]].copy()
    if filtered.empty:
        raise ValueError(
            f"walk-forward window {window_id} {phase} contains no exchange-eligible rows"
        )
    return filtered


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
    """Select on eligible train data, freeze candidate, then evaluate eligible test data.

    Point-in-time exchange eligibility is applied before signal generation as well as during
    execution. This prevents suspended, not-yet-listed, or otherwise ineligible bars from
    influencing indicators on later eligible dates.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")

    results: list[WalkForwardResult] = []
    for window in windows:
        train_frame = _filter_to_trading_eligible_dates(
            _slice_dates(frame, window.train_dates),
            trading_eligibility_policy=trading_eligibility_policy,
            window_id=window.window_id,
            phase="train",
        )
        test_frame = _filter_to_trading_eligible_dates(
            _slice_dates(frame, window.test_dates),
            trading_eligibility_policy=trading_eligibility_policy,
            window_id=window.window_id,
            phase="test",
        )
        train_frame = filter_to_continuous_session(train_frame, session_policy)
        test_frame = filter_to_continuous_session(test_frame, session_policy)
        if train_frame.empty or test_frame.empty:
            raise ValueError(
                f"walk-forward window {window.window_id} has no continuous-session rows"
            )

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
