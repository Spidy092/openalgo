from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

import pandas as pd

from .costs import CostProvider
from .event_simulator import (
    FillAssumptions,
    IntradaySimulationConfig,
    IntradaySimulationResult,
    SessionExitResolver,
    TickSizeResolver,
    TradingEligibilityResolver,
    simulate_long_intraday,
)
from .liquidation_equity import (
    EquityObservation,
    build_liquidation_equity_curve,
    liquidation_drawdown_metrics,
)
from .models import Exchange
from .strategies import StrategySignals


class RankingMetric(StrEnum):
    NET_RETURN_PCT = "net_return_pct"
    PROFIT_FACTOR = "profit_factor"
    REALIZED_MAX_DRAWDOWN_PCT = "realized_max_drawdown_pct"
    CLOSE_LIQUIDATION_MAX_DRAWDOWN_PCT = "close_liquidation_max_drawdown_pct"
    OHLC_LOW_LIQUIDATION_STRESS_MAX_DRAWDOWN_PCT = (
        "ohlc_low_liquidation_stress_max_drawdown_pct"
    )


@dataclass(frozen=True)
class ExactMetrics:
    trade_count: int
    wins: int
    losses: int
    flat_trades: int
    win_rate_pct: Decimal | None
    profit_factor: Decimal | None
    net_pnl: Decimal
    net_return_pct: Decimal
    realized_max_drawdown_pct: Decimal
    close_liquidation_max_drawdown_pct: Decimal
    ohlc_low_liquidation_stress_max_drawdown_pct: Decimal
    transaction_costs: Decimal
    execution_friction: Decimal


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    strategy_name: str
    parameters: dict[str, str]
    source_refs: tuple[str, ...]
    metrics: ExactMetrics
    simulation: IntradaySimulationResult
    equity_curve: tuple[EquityObservation, ...]


def _exact_metrics(
    result: IntradaySimulationResult,
    *,
    equity_curve: tuple[EquityObservation, ...],
) -> ExactMetrics:
    trades = result.trades
    wins = sum(1 for trade in trades if trade.net_pnl > 0)
    losses = sum(1 for trade in trades if trade.net_pnl < 0)
    flat = len(trades) - wins - losses

    gross_profit = sum((trade.net_pnl for trade in trades if trade.net_pnl > 0), Decimal("0"))
    gross_loss = -sum((trade.net_pnl for trade in trades if trade.net_pnl < 0), Decimal("0"))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    win_rate = Decimal(wins) / Decimal(len(trades)) * Decimal("100") if trades else None

    equity = result.initial_cash
    peak = equity
    realized_max_drawdown = Decimal("0")
    for trade in trades:
        equity += trade.net_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            drawdown = (peak - equity) / peak * Decimal("100")
            if drawdown > realized_max_drawdown:
                realized_max_drawdown = drawdown

    liquidation = liquidation_drawdown_metrics(
        equity_curve,
        initial_equity=result.initial_cash,
    )
    net_return = result.net_pnl / result.initial_cash * Decimal("100")
    return ExactMetrics(
        trade_count=len(trades),
        wins=wins,
        losses=losses,
        flat_trades=flat,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        net_pnl=result.net_pnl,
        net_return_pct=net_return,
        realized_max_drawdown_pct=realized_max_drawdown,
        close_liquidation_max_drawdown_pct=(
            liquidation.close_liquidation_max_drawdown_pct
        ),
        ohlc_low_liquidation_stress_max_drawdown_pct=(
            liquidation.ohlc_low_liquidation_stress_max_drawdown_pct
        ),
        transaction_costs=result.total_transaction_costs,
        execution_friction=result.total_execution_friction,
    )


def evaluate_candidate_exact(
    *,
    candidate_id: str,
    frame: pd.DataFrame,
    signals: StrategySignals,
    instrument_token: str,
    exchange: Exchange,
    cost_provider: CostProvider,
    fills: FillAssumptions,
    session_policy: SessionExitResolver,
    tick_size_policy: TickSizeResolver,
    trading_eligibility_policy: TradingEligibilityResolver,
    config: IntradaySimulationConfig,
) -> CandidateEvaluation:
    """Run one strategy candidate through the point-in-time Decimal simulator."""

    if not candidate_id:
        raise ValueError("candidate_id is required")

    simulation = simulate_long_intraday(
        frame=frame,
        entries_at_close=signals.entries_at_close,
        exits_at_close=signals.exits_at_close,
        instrument_token=instrument_token,
        exchange=exchange,
        cost_provider=cost_provider,
        fills=fills,
        session_policy=session_policy,
        tick_size_policy=tick_size_policy,
        trading_eligibility_policy=trading_eligibility_policy,
        config=config,
    )
    equity_curve = build_liquidation_equity_curve(
        frame=frame,
        simulation=simulation,
        instrument_token=instrument_token,
        exchange=exchange,
        cost_provider=cost_provider,
        fills=fills,
    )
    return CandidateEvaluation(
        candidate_id=candidate_id,
        strategy_name=signals.name,
        parameters=dict(signals.parameters),
        source_refs=signals.source_refs,
        metrics=_exact_metrics(simulation, equity_curve=equity_curve),
        simulation=simulation,
        equity_curve=equity_curve,
    )


def rank_candidates(
    evaluations: list[CandidateEvaluation],
    *,
    metric: RankingMetric,
) -> list[CandidateEvaluation]:
    """Rank by one caller-selected metric; there is deliberately no hidden composite score."""

    if metric is RankingMetric.NET_RETURN_PCT:
        return sorted(evaluations, key=lambda item: item.metrics.net_return_pct, reverse=True)
    if metric is RankingMetric.PROFIT_FACTOR:
        return sorted(
            evaluations,
            key=lambda item: (
                item.metrics.profit_factor is not None,
                item.metrics.profit_factor or Decimal("0"),
            ),
            reverse=True,
        )
    if metric is RankingMetric.REALIZED_MAX_DRAWDOWN_PCT:
        return sorted(evaluations, key=lambda item: item.metrics.realized_max_drawdown_pct)
    if metric is RankingMetric.CLOSE_LIQUIDATION_MAX_DRAWDOWN_PCT:
        return sorted(
            evaluations,
            key=lambda item: item.metrics.close_liquidation_max_drawdown_pct,
        )
    if metric is RankingMetric.OHLC_LOW_LIQUIDATION_STRESS_MAX_DRAWDOWN_PCT:
        return sorted(
            evaluations,
            key=lambda item: item.metrics.ohlc_low_liquidation_stress_max_drawdown_pct,
        )
    raise ValueError(f"unsupported ranking metric: {metric}")
