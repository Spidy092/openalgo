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
)
from .historical_membership import (
    HistoricalMembershipAssessment,
    HistoricalTradingEligibilityPolicy,
    filter_frame_to_eligible_dates,
)
from .models import Exchange
from .market_sessions import filter_to_continuous_session
from .tournament import CandidateEvaluation, RankingMetric, evaluate_candidate_exact
from .universe_builder import ResearchUniverseAudit, ResearchUniverseBuildResult


@dataclass(frozen=True)
class CrossSectionalInstrumentInput:
    """One universe-approved instrument with the exact evidence used for simulation."""

    instrument_key: str
    symbol: str
    frame: pd.DataFrame
    dataset_fingerprint: str
    historical_membership: HistoricalMembershipAssessment
    tick_size_policy: TickSizeResolver
    session_policy: SessionExitResolver


@dataclass(frozen=True)
class CrossSectionalEvaluation:
    instrument_key: str
    symbol: str
    dataset_fingerprint: str
    candidate: CandidateEvaluation


@dataclass(frozen=True)
class CrossSectionalExclusion:
    instrument_key: str
    candidate_id: str
    reason: str


@dataclass(frozen=True)
class CrossSectionalTournamentResult:
    """Independent equal-capital strategy/instrument comparisons.

    P&Ls across rows must not be added together as a portfolio. Each evaluation starts with the
    same `capital_per_evaluation` solely to make candidate comparisons like-for-like.
    """

    selection_cutoff: date
    ranking_metric: RankingMetric
    min_trade_count: int
    capital_per_evaluation: Decimal
    evaluations: tuple[CrossSectionalEvaluation, ...]
    ranked: tuple[CrossSectionalEvaluation, ...]
    exclusions: tuple[CrossSectionalExclusion, ...]

    @property
    def winner(self) -> CrossSectionalEvaluation | None:
        return self.ranked[0] if self.ranked else None


def _rank_value(item: CrossSectionalEvaluation, metric: RankingMetric) -> Decimal:
    metrics = item.candidate.metrics
    if metric is RankingMetric.NET_RETURN_PCT:
        return metrics.net_return_pct
    if metric is RankingMetric.REALIZED_MAX_DRAWDOWN_PCT:
        return -metrics.realized_max_drawdown_pct
    if metric is RankingMetric.CLOSE_LIQUIDATION_MAX_DRAWDOWN_PCT:
        return -metrics.close_liquidation_max_drawdown_pct
    if metric is RankingMetric.OHLC_LOW_LIQUIDATION_STRESS_MAX_DRAWDOWN_PCT:
        return -metrics.ohlc_low_liquidation_stress_max_drawdown_pct
    if metric is RankingMetric.PROFIT_FACTOR:
        # None is deliberately not treated as infinity: it covers both no-loss and undefined
        # cases. Callers wanting a no-loss tie-break must define another explicit metric.
        return metrics.profit_factor if metrics.profit_factor is not None else Decimal("-1")
    raise ValueError(f"unsupported ranking metric: {metric}")


def _audit_by_instrument(
    universe: ResearchUniverseBuildResult,
) -> dict[str, ResearchUniverseAudit]:
    audits = {audit.instrument_key: audit for audit in universe.audits}
    if len(audits) != len(universe.audits):
        raise ValueError("universe result contains duplicate instrument audits")
    return audits


def run_cross_sectional_tournament(
    *,
    universe: ResearchUniverseBuildResult,
    instruments: list[CrossSectionalInstrumentInput],
    candidates: list[StrategyDefinition],
    ranking_metric: RankingMetric,
    min_trade_count: int,
    cost_provider: CostProvider,
    fills: FillAssumptions,
    simulation_config: IntradaySimulationConfig,
) -> CrossSectionalTournamentResult:
    """Evaluate every frozen-universe stock × strategy pair with exact modeled costs.

    The supplied instrument set must equal the universe builder's eligible set exactly. Strategy
    signal generation is performed only on dates that the frozen point-in-time membership marks
    exchange-eligible, so suspended/not-yet-listed days cannot influence later indicators.
    """

    if min_trade_count <= 0:
        raise ValueError("min_trade_count must be positive")
    if not candidates:
        raise ValueError("at least one strategy candidate is required")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("strategy candidate IDs must be unique")

    instrument_keys = [item.instrument_key for item in instruments]
    if len(set(instrument_keys)) != len(instrument_keys):
        raise ValueError("cross-sectional instrument keys must be unique")

    expected_keys = set(universe.eligible_instrument_keys)
    supplied_keys = set(instrument_keys)
    if supplied_keys != expected_keys:
        missing = sorted(expected_keys - supplied_keys)
        unexpected = sorted(supplied_keys - expected_keys)
        details: list[str] = []
        if missing:
            details.append("missing eligible instruments: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected/ineligible instruments: " + ", ".join(unexpected))
        raise ValueError(
            "instrument set does not match frozen research universe; " + "; ".join(details)
        )

    audits = _audit_by_instrument(universe)
    evaluations: list[CrossSectionalEvaluation] = []
    exclusions: list[CrossSectionalExclusion] = []

    for instrument in sorted(instruments, key=lambda item: item.instrument_key):
        if not instrument.symbol.strip():
            raise ValueError(f"symbol is required for {instrument.instrument_key}")
        audit = audits[instrument.instrument_key]
        if not audit.decision.eligible:
            raise ValueError(f"instrument {instrument.instrument_key} is not universe-eligible")
        if instrument.dataset_fingerprint != audit.dataset_fingerprint:
            raise ValueError(
                f"dataset fingerprint changed after universe selection for {instrument.instrument_key}"
            )
        if instrument.historical_membership.instrument_key != instrument.instrument_key:
            raise ValueError(
                f"historical membership belongs to another instrument for {instrument.instrument_key}"
            )
        if not instrument.historical_membership.complete:
            raise ValueError(f"historical membership is incomplete for {instrument.instrument_key}")
        if instrument.frame.empty:
            raise ValueError(f"tournament frame is empty for {instrument.instrument_key}")
        if instrument.frame.index.tz is None:
            raise ValueError(f"tournament frame is timezone-naive for {instrument.instrument_key}")
        if instrument.frame.index.max().date() > universe.selection_cutoff:
            raise ValueError(
                f"tournament lookahead detected for {instrument.instrument_key}: "
                "data extends beyond selection cutoff"
            )

        eligible_frame = filter_frame_to_eligible_dates(
            instrument.frame,
            instrument.historical_membership,
        )
        eligible_frame = filter_to_continuous_session(eligible_frame, instrument.session_policy)
        if eligible_frame.empty:
            raise ValueError(
                f"frozen universe instrument {instrument.instrument_key} has no eligible simulation rows"
            )
        trading_policy = HistoricalTradingEligibilityPolicy(instrument.historical_membership)

        for definition in sorted(candidates, key=lambda item: item.candidate_id):
            signals = definition.build_signals(eligible_frame)
            candidate = evaluate_candidate_exact(
                candidate_id=definition.candidate_id,
                frame=eligible_frame,
                signals=signals,
                instrument_token=instrument.instrument_key,
                exchange=Exchange.NSE,
                cost_provider=cost_provider,
                fills=fills,
                session_policy=instrument.session_policy,
                tick_size_policy=instrument.tick_size_policy,
                trading_eligibility_policy=trading_policy,
                config=simulation_config,
            )
            wrapped = CrossSectionalEvaluation(
                instrument_key=instrument.instrument_key,
                symbol=instrument.symbol,
                dataset_fingerprint=instrument.dataset_fingerprint,
                candidate=candidate,
            )
            evaluations.append(wrapped)
            if candidate.metrics.trade_count < min_trade_count:
                exclusions.append(
                    CrossSectionalExclusion(
                        instrument_key=instrument.instrument_key,
                        candidate_id=definition.candidate_id,
                        reason=(
                            f"trade_count {candidate.metrics.trade_count} is below explicit minimum "
                            f"{min_trade_count}"
                        ),
                    )
                )

    excluded_pairs = {(item.instrument_key, item.candidate_id) for item in exclusions}
    rankable = [
        item
        for item in evaluations
        if (item.instrument_key, item.candidate.candidate_id) not in excluded_pairs
    ]
    # Lexical identity is used only as a deterministic tie-break when the selected metric is
    # exactly equal; it is not a second performance score.
    rankable.sort(key=lambda item: (item.instrument_key, item.candidate.candidate_id))
    rankable.sort(key=lambda item: _rank_value(item, ranking_metric), reverse=True)

    return CrossSectionalTournamentResult(
        selection_cutoff=universe.selection_cutoff,
        ranking_metric=ranking_metric,
        min_trade_count=min_trade_count,
        capital_per_evaluation=simulation_config.initial_cash,
        evaluations=tuple(evaluations),
        ranked=tuple(rankable),
        exclusions=tuple(exclusions),
    )
