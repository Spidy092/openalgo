from datetime import date, time
from decimal import Decimal

import pandas as pd
import pytest

from equity_engine.candidate_grid import baseline_definition
from equity_engine.cross_sectional_tournament import CrossSectionalInstrumentInput
from equity_engine.cross_sectional_walk_forward import (
    CrossSectionalTestInstrumentInput,
    DatedCorporateActionEvidence,
    run_cross_sectional_walk_forward_window,
)
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import FillAssumptions, IntradaySimulationConfig
from equity_engine.historical_membership import HistoricalTradingStatus, assess_historical_membership
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.tournament import RankingMetric
from equity_engine.universe import CorporateActionAssessment, UniverseDecision
from equity_engine.universe_builder import ResearchUniverseAudit, ResearchUniverseBuildResult
from equity_engine.walk_forward import WalkForwardWindow


def _frame(day: date, entry_open: float, cutoff_open: float) -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{day} 09:15", tz="Asia/Kolkata"),
            pd.Timestamp(f"{day} 09:20", tz="Asia/Kolkata"),
            pd.Timestamp(f"{day} 15:20", tz="Asia/Kolkata"),
        ]
    )
    prices = [entry_open, entry_open, cutoff_open]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": [p + 0.1 for p in prices],
            "volume": [100_000] * 3,
        },
        index=timestamps,
    )


def _membership(instrument_key: str, days: tuple[date, ...]):
    return assess_historical_membership(
        instrument_key=instrument_key,
        trading_dates=days,
        statuses=[
            HistoricalTradingStatus(
                trade_date=day,
                instrument_key=instrument_key,
                listed_on_nse=True,
                normal_equity=True,
                tradeable_in_normal_market=True,
                source=f"synthetic:{day}",
            )
            for day in days
        ],
    )


def _train_input(
    instrument_key: str,
    symbol: str,
    frame: pd.DataFrame,
    fingerprint: str,
    membership_days: tuple[date, ...],
) -> CrossSectionalInstrumentInput:
    return CrossSectionalInstrumentInput(
        instrument_key=instrument_key,
        symbol=symbol,
        frame=frame,
        dataset_fingerprint=fingerprint,
        historical_membership=_membership(instrument_key, membership_days),
        tick_size_policy=FixedTickSizePolicy(
            tick_size_rupees=Decimal("0.05"),
            source="synthetic-test",
        ),
        session_policy=NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=10),
    )


def _test_input(
    instrument_key: str,
    symbol: str,
    frame: pd.DataFrame,
    fingerprint: str,
    membership_days: tuple[date, ...],
    *,
    tick_policy=None,
    corporate_actions_complete: bool = True,
    blocking_events: tuple[str, ...] = (),
    corporate_action_window: tuple[date, date] | None = None,
) -> CrossSectionalTestInstrumentInput:
    first_day = min(membership_days)
    last_day = max(membership_days)
    ca_start, ca_end = corporate_action_window or (first_day, last_day)
    return CrossSectionalTestInstrumentInput(
        instrument_key=instrument_key,
        symbol=symbol,
        frame=frame,
        dataset_fingerprint=fingerprint,
        historical_membership=_membership(instrument_key, membership_days),
        tick_size_policy=tick_policy
        or FixedTickSizePolicy(
            tick_size_rupees=Decimal("0.05"),
            source="synthetic-test",
        ),
        session_policy=NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=10),
        corporate_actions=DatedCorporateActionEvidence(
            window_start=ca_start,
            window_end=ca_end,
            assessment=CorporateActionAssessment(
                complete=corporate_actions_complete,
                blocking_events=blocking_events,
            ),
            source="synthetic-corporate-actions",
        ),
    )


def _audit(
    instrument_key: str,
    fingerprint: str,
    frame: pd.DataFrame,
) -> ResearchUniverseAudit:
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
        eligible_trading_days=1,
        affordable_quantity=1,
        last_price_rupees=Decimal(str(frame.iloc[-1]["close"])),
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


def _fixture():
    train_day = date(2026, 9, 1)
    test_day = date(2026, 9, 2)
    window = WalkForwardWindow(
        window_id=1,
        train_dates=(train_day,),
        test_dates=(test_day,),
    )
    # AAA wins train; BBB would win test. Correct walk-forward must still test AAA.
    aaa_train = _frame(train_day, 100.0, 110.0)
    bbb_train = _frame(train_day, 100.0, 101.0)
    aaa_test = _frame(test_day, 100.0, 90.0)
    bbb_test = _frame(test_day, 100.0, 120.0)
    universe = ResearchUniverseBuildResult(
        selection_cutoff=train_day,
        audits=(
            _audit("NSE_EQ|AAA", "aaa-train-fp", aaa_train),
            _audit("NSE_EQ|BBB", "bbb-train-fp", bbb_train),
        ),
    )
    train_inputs = [
        _train_input("NSE_EQ|AAA", "AAA", aaa_train, "aaa-train-fp", (train_day,)),
        _train_input("NSE_EQ|BBB", "BBB", bbb_train, "bbb-train-fp", (train_day,)),
    ]
    test_inputs = [
        _test_input("NSE_EQ|AAA", "AAA", aaa_test, "aaa-test-fp", (test_day,)),
        _test_input("NSE_EQ|BBB", "BBB", bbb_test, "bbb-test-fp", (test_day,)),
    ]
    return window, universe, train_inputs, test_inputs


def _run(window, universe, train_inputs, test_inputs):
    return run_cross_sectional_walk_forward_window(
        window=window,
        train_universe=universe,
        train_instruments=train_inputs,
        test_instruments=test_inputs,
        candidates=[baseline_definition(session_open=time(9, 15))],
        ranking_metric=RankingMetric.NET_RETURN_PCT,
        min_train_trade_count=1,
        cost_provider=_costs(),
        fills=_fills(),
        simulation_config=_config(),
    )


def test_walk_forward_freezes_train_stock_and_strategy_even_when_future_winner_differs() -> None:
    window, universe, train_inputs, test_inputs = _fixture()
    result = _run(window, universe, train_inputs, test_inputs)

    assert result.selected_instrument_key == "NSE_EQ|AAA"
    assert result.selected_candidate_id == "baseline:first-bar-hold"
    assert result.train_evaluation.metrics.net_pnl > 0
    assert result.test_evaluation.metrics.net_pnl < 0
    assert result.test_dataset_fingerprint == "aaa-test-fp"
    # BBB's excellent future move cannot cause re-selection because test is never ranked.
    assert result.test_evaluation.simulation.trades[0].entry_timestamp.date() == window.test_dates[0]


def test_walk_forward_requires_test_data_for_entire_frozen_train_universe_before_selection() -> None:
    window, universe, train_inputs, test_inputs = _fixture()
    with pytest.raises(
        ValueError,
        match="test instrument set must equal frozen train universe; missing: NSE_EQ\\|BBB",
    ):
        _run(window, universe, train_inputs, [test_inputs[0]])


def test_walk_forward_train_universe_cutoff_must_be_final_train_date() -> None:
    window, universe, train_inputs, test_inputs = _fixture()
    stale_universe = ResearchUniverseBuildResult(
        selection_cutoff=date(2026, 8, 31),
        audits=universe.audits,
    )
    with pytest.raises(ValueError, match="cutoff must equal the final train date"):
        _run(window, stale_universe, train_inputs, test_inputs)


def test_walk_forward_rejects_test_frame_that_contains_train_date() -> None:
    window, universe, train_inputs, test_inputs = _fixture()
    contaminated = list(test_inputs)
    contaminated[0] = _test_input(
        "NSE_EQ|AAA",
        "AAA",
        train_inputs[0].frame,
        "contaminated-test-fp",
        window.test_dates,
    )
    with pytest.raises(
        ValueError,
        match="test frame contains dates outside the walk-forward test window",
    ):
        _run(window, universe, train_inputs, contaminated)


def test_walk_forward_requires_corporate_action_evidence_for_exact_test_window() -> None:
    window, universe, train_inputs, test_inputs = _fixture()
    wrong_window = list(test_inputs)
    wrong_window[0] = _test_input(
        "NSE_EQ|AAA",
        "AAA",
        test_inputs[0].frame,
        "aaa-test-fp",
        window.test_dates,
        corporate_action_window=(date(2026, 9, 1), date(2026, 9, 2)),
    )
    with pytest.raises(ValueError, match="corporate-action evidence window does not match"):
        _run(window, universe, train_inputs, wrong_window)


def test_selected_winner_structural_action_blocks_future_score() -> None:
    window, universe, train_inputs, test_inputs = _fixture()
    blocked = list(test_inputs)
    blocked[0] = _test_input(
        "NSE_EQ|AAA",
        "AAA",
        test_inputs[0].frame,
        "aaa-test-fp",
        window.test_dates,
        blocking_events=("Split@2026-09-02 ratio=1:2",),
    )
    with pytest.raises(ValueError, match="blocking corporate actions in test window"):
        _run(window, universe, train_inputs, blocked)


def test_selected_winner_requires_tick_evidence_for_every_eligible_test_date() -> None:
    window, universe, train_inputs, test_inputs = _fixture()

    class _MissingTickPolicy:
        def tick_size(self, trade_date: date) -> Decimal:
            raise ValueError(f"missing tick for {trade_date}")

    missing_tick = list(test_inputs)
    missing_tick[0] = _test_input(
        "NSE_EQ|AAA",
        "AAA",
        test_inputs[0].frame,
        "aaa-test-fp",
        window.test_dates,
        tick_policy=_MissingTickPolicy(),
    )
    with pytest.raises(ValueError, match="lacks verified tick evidence for test dates"):
        _run(window, universe, train_inputs, missing_tick)
