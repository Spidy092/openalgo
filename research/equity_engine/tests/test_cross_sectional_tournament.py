from datetime import date, time
from decimal import Decimal

import pandas as pd
import pytest

from equity_engine.candidate_grid import StrategyDefinition, baseline_definition
from equity_engine.cross_sectional_tournament import (
    CrossSectionalInstrumentInput,
    run_cross_sectional_tournament,
)
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import FillAssumptions, IntradaySimulationConfig
from equity_engine.historical_membership import (
    HistoricalTradingStatus,
    assess_historical_membership,
)
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.strategies import first_bar_hold_baseline
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.tournament import RankingMetric
from equity_engine.universe import UniverseDecision
from equity_engine.universe_builder import ResearchUniverseAudit, ResearchUniverseBuildResult


def _frame(days: list[tuple[str, float, float]]) -> pd.DataFrame:
    timestamps: list[pd.Timestamp] = []
    prices: list[float] = []
    for day, entry_open, cutoff_open in days:
        for hhmm, price in (
            ("09:15", entry_open),
            ("09:20", entry_open),
            ("15:20", cutoff_open),
        ):
            timestamps.append(pd.Timestamp(f"{day} {hhmm}", tz="Asia/Kolkata"))
            prices.append(price)
    return pd.DataFrame(
        {
            "open": prices,
            "high": [price + 0.5 for price in prices],
            "low": [price - 0.5 for price in prices],
            "close": [price + 0.1 for price in prices],
            "volume": [100_000] * len(prices),
        },
        index=pd.DatetimeIndex(timestamps),
    )


def _membership(
    instrument_key: str,
    frame: pd.DataFrame,
    *,
    ineligible_dates: set[date] | None = None,
):
    ineligible = ineligible_dates or set()
    days = tuple(sorted(set(frame.index.date)))
    statuses = [
        HistoricalTradingStatus(
            trade_date=day,
            instrument_key=instrument_key,
            listed_on_nse=True,
            normal_equity=True,
            tradeable_in_normal_market=day not in ineligible,
            source=f"synthetic-membership:{day}",
        )
        for day in days
    ]
    return assess_historical_membership(
        instrument_key=instrument_key,
        trading_dates=days,
        statuses=statuses,
    )


def _audit(instrument_key: str, fingerprint: str, frame: pd.DataFrame) -> ResearchUniverseAudit:
    return ResearchUniverseAudit(
        instrument_key=instrument_key,
        decision=UniverseDecision(
            instrument_key=instrument_key,
            eligible=True,
            violations=(),
        ),
        dataset_fingerprint=fingerprint,
        data_start=frame.index[0],
        data_end=frame.index[-1],
        eligible_trading_days=len(set(frame.index.date)),
        affordable_quantity=1,
        last_price_rupees=Decimal(str(frame.iloc[-1]["close"])),
    )


def _instrument(
    instrument_key: str,
    symbol: str,
    frame: pd.DataFrame,
    fingerprint: str,
    *,
    ineligible_dates: set[date] | None = None,
) -> CrossSectionalInstrumentInput:
    return CrossSectionalInstrumentInput(
        instrument_key=instrument_key,
        symbol=symbol,
        frame=frame,
        dataset_fingerprint=fingerprint,
        historical_membership=_membership(
            instrument_key,
            frame,
            ineligible_dates=ineligible_dates,
        ),
        tick_size_policy=FixedTickSizePolicy(
            tick_size_rupees=Decimal("0.05"),
            source="synthetic-test",
        ),
        session_policy=NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=10),
    )


def _costs() -> CurrentTermsNSEIntradayCostProvider:
    return CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))


def _fills() -> FillAssumptions:
    return FillAssumptions(
        slippage_bps_per_leg=Decimal("0"),
        half_spread_bps_per_leg=Decimal("0"),
    )


def _config() -> IntradaySimulationConfig:
    return IntradaySimulationConfig(
        initial_cash=Decimal("1000"),
        max_trades_per_day=1,
    )


def test_cross_sectional_tournament_ranks_stock_strategy_pairs_after_exact_costs() -> None:
    strong = _frame([("2026-09-01", 100.0, 110.0)])
    weak = _frame([("2026-09-01", 100.0, 101.0)])
    universe = ResearchUniverseBuildResult(
        selection_cutoff=date(2026, 9, 1),
        audits=(
            _audit("NSE_EQ|AAA", "fp-aaa", strong),
            _audit("NSE_EQ|BBB", "fp-bbb", weak),
        ),
    )

    result = run_cross_sectional_tournament(
        universe=universe,
        instruments=[
            _instrument("NSE_EQ|AAA", "AAA", strong, "fp-aaa"),
            _instrument("NSE_EQ|BBB", "BBB", weak, "fp-bbb"),
        ],
        candidates=[baseline_definition(session_open=time(9, 15))],
        ranking_metric=RankingMetric.NET_RETURN_PCT,
        min_trade_count=1,
        cost_provider=_costs(),
        fills=_fills(),
        simulation_config=_config(),
    )

    assert result.winner is not None
    assert result.winner.instrument_key == "NSE_EQ|AAA"
    assert result.capital_per_evaluation == Decimal("1000")
    assert len(result.evaluations) == 2
    assert all(
        item.candidate.simulation.initial_cash == Decimal("1000")
        for item in result.evaluations
    )
    assert result.ranked[0].candidate.metrics.net_pnl > result.ranked[1].candidate.metrics.net_pnl


def test_tournament_requires_every_frozen_eligible_instrument_no_cherry_picking() -> None:
    frame = _frame([("2026-09-01", 100.0, 101.0)])
    universe = ResearchUniverseBuildResult(
        selection_cutoff=date(2026, 9, 1),
        audits=(
            _audit("NSE_EQ|AAA", "fp-aaa", frame),
            _audit("NSE_EQ|BBB", "fp-bbb", frame),
        ),
    )

    with pytest.raises(ValueError, match="missing eligible instruments: NSE_EQ\\|BBB"):
        run_cross_sectional_tournament(
            universe=universe,
            instruments=[_instrument("NSE_EQ|AAA", "AAA", frame, "fp-aaa")],
            candidates=[baseline_definition(session_open=time(9, 15))],
            ranking_metric=RankingMetric.NET_RETURN_PCT,
            min_trade_count=1,
            cost_provider=_costs(),
            fills=_fills(),
            simulation_config=_config(),
        )


def test_dataset_fingerprint_cannot_change_after_universe_selection() -> None:
    frame = _frame([("2026-09-01", 100.0, 101.0)])
    universe = ResearchUniverseBuildResult(
        selection_cutoff=date(2026, 9, 1),
        audits=(_audit("NSE_EQ|AAA", "frozen-fingerprint", frame),),
    )

    with pytest.raises(ValueError, match="dataset fingerprint changed"):
        run_cross_sectional_tournament(
            universe=universe,
            instruments=[_instrument("NSE_EQ|AAA", "AAA", frame, "different-fingerprint")],
            candidates=[baseline_definition(session_open=time(9, 15))],
            ranking_metric=RankingMetric.NET_RETURN_PCT,
            min_trade_count=1,
            cost_provider=_costs(),
            fills=_fills(),
            simulation_config=_config(),
        )


def test_strategy_never_observes_exchange_ineligible_dates() -> None:
    first_day = date(2026, 9, 1)
    second_day = date(2026, 9, 2)
    frame = _frame(
        [
            (first_day.isoformat(), 100.0, 500.0),
            (second_day.isoformat(), 100.0, 101.0),
        ]
    )
    universe = ResearchUniverseBuildResult(
        selection_cutoff=second_day,
        audits=(_audit("NSE_EQ|AAA", "fp-aaa", frame),),
    )

    def build_only_after_filter(filtered: pd.DataFrame):
        assert set(filtered.index.date) == {second_day}
        return first_bar_hold_baseline(filtered, session_open=time(9, 15))

    definition = StrategyDefinition(
        candidate_id="test:eligible-only",
        build_signals=build_only_after_filter,
        research_basis="synthetic anti-leakage test",
        source_refs=(),
    )
    result = run_cross_sectional_tournament(
        universe=universe,
        instruments=[
            _instrument(
                "NSE_EQ|AAA",
                "AAA",
                frame,
                "fp-aaa",
                ineligible_dates={first_day},
            )
        ],
        candidates=[definition],
        ranking_metric=RankingMetric.NET_RETURN_PCT,
        min_trade_count=1,
        cost_provider=_costs(),
        fills=_fills(),
        simulation_config=_config(),
    )

    assert len(result.evaluations) == 1
    assert result.evaluations[0].candidate.metrics.trade_count == 1


def test_explicit_minimum_trade_count_prevents_zero_or_tiny_sample_winner() -> None:
    frame = _frame([("2026-09-01", 100.0, 110.0)])
    universe = ResearchUniverseBuildResult(
        selection_cutoff=date(2026, 9, 1),
        audits=(_audit("NSE_EQ|AAA", "fp-aaa", frame),),
    )

    result = run_cross_sectional_tournament(
        universe=universe,
        instruments=[_instrument("NSE_EQ|AAA", "AAA", frame, "fp-aaa")],
        candidates=[baseline_definition(session_open=time(9, 15))],
        ranking_metric=RankingMetric.NET_RETURN_PCT,
        min_trade_count=2,
        cost_provider=_costs(),
        fills=_fills(),
        simulation_config=_config(),
    )

    assert result.winner is None
    assert result.ranked == ()
    assert len(result.exclusions) == 1
    assert "below explicit minimum 2" in result.exclusions[0].reason
