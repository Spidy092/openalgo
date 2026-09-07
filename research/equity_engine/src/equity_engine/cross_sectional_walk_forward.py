from __future__ import annotations

from dataclasses import dataclass

from .candidate_grid import StrategyDefinition
from .costs import CostProvider
from .cross_sectional_tournament import (
    CrossSectionalInstrumentInput,
    CrossSectionalTournamentResult,
    run_cross_sectional_tournament,
)
from .event_simulator import FillAssumptions, IntradaySimulationConfig
from .historical_membership import (
    HistoricalTradingEligibilityPolicy,
    filter_frame_to_eligible_dates,
)
from .models import Exchange
from .tournament import CandidateEvaluation, RankingMetric, evaluate_candidate_exact
from .universe_builder import ResearchUniverseBuildResult
from .walk_forward import WalkForwardWindow


@dataclass(frozen=True)
class CrossSectionalWalkForwardResult:
    """One train-select / future-test result with the stock and strategy frozen together."""

    window: WalkForwardWindow
    selected_instrument_key: str
    selected_symbol: str
    selected_candidate_id: str
    train_tournament: CrossSectionalTournamentResult
    train_evaluation: CandidateEvaluation
    test_evaluation: CandidateEvaluation
    test_dataset_fingerprint: str


def _require_exact_instrument_set(
    *,
    label: str,
    instruments: list[CrossSectionalInstrumentInput],
    expected_keys: set[str],
) -> dict[str, CrossSectionalInstrumentInput]:
    keys = [item.instrument_key for item in instruments]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} instrument keys must be unique")
    supplied = set(keys)
    if supplied != expected_keys:
        missing = sorted(expected_keys - supplied)
        unexpected = sorted(supplied - expected_keys)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError(
            f"{label} instrument set must equal frozen train universe; " + "; ".join(details)
        )
    return {item.instrument_key: item for item in instruments}


def _validate_period_input(
    *,
    instrument: CrossSectionalInstrumentInput,
    allowed_dates: tuple,
    label: str,
) -> None:
    if not instrument.dataset_fingerprint.strip():
        raise ValueError(f"{label} dataset fingerprint is missing for {instrument.instrument_key}")
    if instrument.frame.empty:
        raise ValueError(f"{label} frame is empty for {instrument.instrument_key}")
    if instrument.frame.index.tz is None:
        raise ValueError(f"{label} frame is timezone-naive for {instrument.instrument_key}")
    if instrument.historical_membership.instrument_key != instrument.instrument_key:
        raise ValueError(
            f"{label} membership belongs to another instrument for {instrument.instrument_key}"
        )
    if not instrument.historical_membership.complete:
        raise ValueError(f"{label} membership is incomplete for {instrument.instrument_key}")

    allowed = set(allowed_dates)
    frame_dates = set(instrument.frame.index.date)
    requested = set(instrument.historical_membership.requested_dates)
    if not allowed.issubset(requested):
        raise ValueError(
            f"{label} membership does not cover every walk-forward date for {instrument.instrument_key}"
        )
    if not frame_dates.issubset(allowed):
        raise ValueError(
            f"{label} frame contains dates outside the walk-forward {label} window for "
            f"{instrument.instrument_key}"
        )

    eligible_required = set(instrument.historical_membership.eligible_dates) & allowed
    missing_market_data = sorted(eligible_required - frame_dates)
    if missing_market_data:
        raise ValueError(
            f"{label} frame is missing OHLCV for exchange-eligible dates on "
            f"{instrument.instrument_key}: "
            + ", ".join(day.isoformat() for day in missing_market_data)
        )


def run_cross_sectional_walk_forward_window(
    *,
    window: WalkForwardWindow,
    train_universe: ResearchUniverseBuildResult,
    train_instruments: list[CrossSectionalInstrumentInput],
    test_instruments: list[CrossSectionalInstrumentInput],
    candidates: list[StrategyDefinition],
    ranking_metric: RankingMetric,
    min_train_trade_count: int,
    cost_provider: CostProvider,
    fills: FillAssumptions,
    simulation_config: IntradaySimulationConfig,
) -> CrossSectionalWalkForwardResult:
    """Select stock+strategy on train only, then freeze and evaluate on untouched future dates.

    Test data is supplied for every stock in the frozen train universe before the winner is known.
    The test period never re-ranks stocks or strategies: only the train winner is evaluated.
    """

    if not window.train_dates or not window.test_dates:
        raise ValueError("walk-forward train and test dates must both be non-empty")
    if max(window.train_dates) >= min(window.test_dates):
        raise ValueError("walk-forward test dates must be strictly after train dates")
    if train_universe.selection_cutoff != max(window.train_dates):
        raise ValueError(
            "frozen train universe cutoff must equal the final train date for this walk-forward window"
        )

    expected_keys = set(train_universe.eligible_instrument_keys)
    if not expected_keys:
        raise ValueError("frozen train universe contains no eligible instruments")
    train_by_key = _require_exact_instrument_set(
        label="train",
        instruments=train_instruments,
        expected_keys=expected_keys,
    )
    test_by_key = _require_exact_instrument_set(
        label="test",
        instruments=test_instruments,
        expected_keys=expected_keys,
    )

    for instrument in train_by_key.values():
        _validate_period_input(
            instrument=instrument,
            allowed_dates=window.train_dates,
            label="train",
        )
    for instrument in test_by_key.values():
        _validate_period_input(
            instrument=instrument,
            allowed_dates=window.test_dates,
            label="test",
        )

    train_tournament = run_cross_sectional_tournament(
        universe=train_universe,
        instruments=train_instruments,
        candidates=candidates,
        ranking_metric=ranking_metric,
        min_trade_count=min_train_trade_count,
        cost_provider=cost_provider,
        fills=fills,
        simulation_config=simulation_config,
    )
    winner = train_tournament.winner
    if winner is None:
        raise ValueError("no train stock-strategy pair satisfies the explicit trade-count gate")

    definitions = {candidate.candidate_id: candidate for candidate in candidates}
    selected_definition = definitions[winner.candidate.candidate_id]
    selected_test_input = test_by_key[winner.instrument_key]
    eligible_test_frame = filter_frame_to_eligible_dates(
        selected_test_input.frame,
        selected_test_input.historical_membership,
    )
    if eligible_test_frame.empty:
        raise ValueError(
            f"selected train winner {winner.instrument_key} has no exchange-eligible test rows"
        )

    # No re-selection here: the train winner's exact strategy definition is frozen before these
    # signals are generated from future data.
    test_signals = selected_definition.build_signals(eligible_test_frame)
    test_evaluation = evaluate_candidate_exact(
        candidate_id=selected_definition.candidate_id,
        frame=eligible_test_frame,
        signals=test_signals,
        instrument_token=selected_test_input.instrument_key,
        exchange=Exchange.NSE,
        cost_provider=cost_provider,
        fills=fills,
        session_policy=selected_test_input.session_policy,
        tick_size_policy=selected_test_input.tick_size_policy,
        trading_eligibility_policy=HistoricalTradingEligibilityPolicy(
            selected_test_input.historical_membership
        ),
        config=simulation_config,
    )

    return CrossSectionalWalkForwardResult(
        window=window,
        selected_instrument_key=winner.instrument_key,
        selected_symbol=winner.symbol,
        selected_candidate_id=winner.candidate.candidate_id,
        train_tournament=train_tournament,
        train_evaluation=winner.candidate,
        test_evaluation=test_evaluation,
        test_dataset_fingerprint=selected_test_input.dataset_fingerprint,
    )
