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
    simulate_long_intraday,
)
from .models import Exchange
from .strategies import StrategySignals


class RankingMetric(StrEnum):
    NET_RETURN_PCT = "net_return_pct"
    PROFIT_FACTOR = "profit_factor"
    REALIZED_MAX_DRAWDOWN_PCT = "realized_max_drawdown_pct"


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


def _exact_metrics(result: IntradaySimulationResult) -> ExactMetrics:
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
    max_drawdown = Decimal("0")
    for trade in trades:
        equity += trade.net_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            drawdown = (peak - equity) / peak * Decimal("100")
            if drawdown > max_drawdown:
                max_drawdown = drawdown

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
        realized_max_drawdown_pct=max_drawdown,
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
    config: IntradaySimulationConfig,
) -> CandidateEvaluation:
    """Run one strategy candidate through the Decimal event-driven simulator."""

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
        config=config,
    )
    return CandidateEvaluation(
        candidate_id=candidate_id,
        strategy_name=signals.name,
        parameters=dict(signals.parameters),
        source_refs=signals.source_refs,
        metrics=_exact_metrics(simulation),
        simulation=simulation,
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
        # Undefined PF means there were no realized losing trades. Do not silently treat that as
        # infinity; place it after candidates with a defined PF and inspect sample size separately.
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
    raise ValueError(f"unsupported ranking metric: {metric}")
