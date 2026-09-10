"""Synthetic end-to-end readiness test for the post-acquisition research execution path.

Proves the complete 11-stage research execution chain:
validated research input
→ frozen PIT population
→ explicit strategy candidate grid
→ VectorBT screening
→ stock × strategy tournament
→ repeated chronological WFO
→ untouched OOS
→ event simulator
→ canonical HistoricalCostScenario
→ immutable Experiment
→ promotion gate

Strictly synthetic/local fixtures only. Zero network calls, zero broker API calls,
zero live orders (live_orders_called=false).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

import pandas as pd
import pytest
from equity_engine.candidate_grid import (
    StrategyDefinition,
    baseline_definition,
    momentum_grid,
    orb_study_grid,
)
from equity_engine.cost_ledger import (
    SCENARIO_LABEL,
    EffectiveDatedCostLedger,
    LedgerComponent,
    LedgerProduct,
)
from equity_engine.cross_sectional_tournament import (
    CrossSectionalInstrumentInput,
    run_cross_sectional_tournament,
)
from equity_engine.cross_sectional_walk_forward import (
    CrossSectionalTestInstrumentInput,
    DatedCorporateActionEvidence,
    run_cross_sectional_walk_forward_window,
)
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import (
    ExitReason,
    FillAssumptions,
    IntradaySimulationConfig,
    simulate_long_intraday,
)
from equity_engine.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ApprovedCapital,
    BaselineComparisonEvidence,
    ConcretePromotionEvidence,
    CorporateActionEvidenceIdentity,
    CostEvidenceIdentity,
    CostModelIdentity,
    CostReconciliationEvidence,
    EmbargoSpec,
    EventDrivenSimulationEvidence,
    ExperimentOrchestrator,
    FrictionScenarioSpec,
    HeldOutTestEvidence,
    LiveOrderAttemptError,
    NSEMembershipEvidenceIdentity,
    PaperTradingEvidence,
    ResearchWindowConfig,
    SessionPolicyIdentity,
    SlippageStressEvidence,
    StrategySpec,
    TickEvidenceIdentity,
    WindowSpec,
    canonical_sha256,
    compute_prefilter_artifact_fingerprint,
)
from equity_engine.gates import DrawdownBasis, PromotionThresholds
from equity_engine.historical_cost_scenario import (
    HistoricalCostScenario,
    MissingScenarioAssumptionError,
    ScenarioAssumption,
    ScenarioError,
    compile_historical_scenario,
)
from equity_engine.historical_membership import (
    HistoricalTradingEligibilityPolicy,
    HistoricalTradingStatus,
    assess_historical_membership,
)
from equity_engine.market_sessions import (
    NSEEquitySessionPolicy,
    filter_to_continuous_session,
)
from equity_engine.models import Exchange
from equity_engine.provenance import MarketDataManifest, dataframe_fingerprint
from equity_engine.research_window_compiler import (
    CorporateActionClaim,
    CostEvidenceClaim,
    FrozenUniverseViolationError,
    PITMembershipSegment,
    SelectionAttestation,
    StrategyDefinitionClaim,
    UntouchedTestViolationError,
    WindowLeakageError,
    WindowRole,
    compile_repeated_wfo,
    resolve_pit_segment,
)
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.tournament import RankingMetric
from equity_engine.universe import CorporateActionAssessment, UniverseDecision
from equity_engine.universe_builder import (
    ResearchUniverseAudit,
    ResearchUniverseBuildResult,
)
from equity_engine.vectorbt_screening import (
    screen_long_signals,
    shift_close_generated_signals,
)
from equity_engine.walk_forward import WalkForwardWindow
from equity_engine.wfo_schedule import plan_wfo_date_windows

STOCK_A = "NSE_EQ|STOCK_A"
STOCK_B = "NSE_EQ|STOCK_B"
STOCK_C = "NSE_EQ|STOCK_C"
CODE_COMMIT_SHA = "4fa6515bf397e0c9d7833e2541a28d827fb8f6d4"


def _generate_business_days(start_date: date, count: int) -> tuple[date, ...]:
    days: list[date] = []
    current = start_date
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _make_intraday_bars_with_cas(
    trade_dates: tuple[date, ...],
    *,
    base_price: float = 100.0,
    daily_drift: float = 1.0,
    include_cas_bar: bool = True,
) -> pd.DataFrame:
    """Generate deterministic 5-minute intraday bars plus auxiliary CAS bars."""
    frames: list[pd.DataFrame] = []

    for day_idx, trade_date in enumerate(trade_dates):
        day_str = trade_date.isoformat()
        day_open = base_price + day_idx * daily_drift

        # Continuous 5m session from 09:15 to 15:25 (75 bars)
        cont_index = pd.date_range(
            f"{day_str} 09:15",
            f"{day_str} 15:25",
            freq="5min",
            tz="Asia/Kolkata",
        )
        n_bars = len(cont_index)
        intraday_pattern = [0.1 * (i % 5) for i in range(n_bars)]
        opens = [round(day_open + p, 2) for p in intraday_pattern]
        highs = [round(o + 0.5, 2) for o in opens]
        lows = [round(o - 0.4, 2) for o in opens]
        closes = [round(o + 0.1, 2) for o in opens]
        volumes = [50_000 + 1_000 * (i % 10) for i in range(n_bars)]

        timestamps = list(cont_index)
        if include_cas_bar:
            cas_ts = pd.Timestamp(f"{day_str} 15:45", tz="Asia/Kolkata")
            timestamps.append(cas_ts)
            opens.append(round(day_open + 0.5, 2))
            highs.append(round(day_open + 1.0, 2))
            lows.append(round(day_open + 0.2, 2))
            closes.append(round(day_open + 0.6, 2))
            volumes.append(10_000)

        day_df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
            index=pd.DatetimeIndex(timestamps),
        )
        frames.append(day_df)

    df = pd.concat(frames)
    df.index.name = "timestamp"
    return df


def _make_manifest(instrument_token: str, symbol: str) -> MarketDataManifest:
    return MarketDataManifest(
        provider="synthetic-local",
        exchange="NSE",
        instrument_token=instrument_token,
        symbol=symbol,
        timezone="Asia/Kolkata",
        interval="5m",
        timestamp_semantics="start-of-candle",
        start=datetime(2026, 6, 1, 9, 15, tzinfo=UTC),
        end=datetime(2026, 7, 6, 15, 30, tzinfo=UTC),
        retrieved_at=datetime(2026, 7, 7, 0, 0, tzinfo=UTC),
        adjustment_policy="split-unadjusted",
        universe_rule_version="nse-cm-2026-v1",
        source_reference="synthetic-readiness-fixture",
    )


def _membership_for(
    instrument_key: str,
    trading_dates: tuple[date, ...],
    *,
    ineligible_dates: set[date] | None = None,
):
    ineligible = ineligible_dates or set()
    statuses = [
        HistoricalTradingStatus(
            trade_date=d,
            instrument_key=instrument_key,
            listed_on_nse=True,
            normal_equity=True,
            tradeable_in_normal_market=(d not in ineligible),
            source=f"synthetic-membership:{d.isoformat()}",
        )
        for d in trading_dates
    ]
    return assess_historical_membership(
        instrument_key=instrument_key,
        trading_dates=trading_dates,
        statuses=statuses,
    )


def _audit_for(
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
        eligible_trading_days=len(set(frame.index.date)),
        affordable_quantity=1,
        last_price_rupees=Decimal(str(frame.iloc[-1]["close"])),
    )


def _candidate_strategies() -> list[StrategyDefinition]:
    """Explicit candidate strategy grid with 8 explicit candidates."""
    candidates: list[StrategyDefinition] = [
        baseline_definition(session_open=time(9, 15)),
        *orb_study_grid(
            session_open=time(9, 15),
            bar_minutes=5,
            breakout_buffer_bps=Decimal(5),
        ),
        *momentum_grid(
            lookback_bars=[3],
            entry_return_bps=[Decimal(10)],
            exit_return_bps=[Decimal(5)],
            volume_lookback_bars=[3],
            min_volume_ratios=[Decimal("1.2")],
        ),
    ]
    return candidates


def _parameter_grid_dict() -> dict[str, tuple[str, ...]]:
    return {
        "bar_minutes": ("5",),
        "breakout_buffer_bps": ("5",),
        "momentum_entry_bps": ("10",),
        "momentum_exit_bps": ("5",),
        "momentum_lookback": ("3",),
        "orb_ranges": ("5", "15", "30"),
        "strategy_family": ("baseline", "momentum", "orb"),
        "volume_ratios": ("1.2", "1.5"),
    }


def _canonical_historical_cost_scenario(
    scenario_date: date = date(2026, 6, 5),
    *,
    brokerage_rate: str = "0.0003",
) -> HistoricalCostScenario:
    ledger = EffectiveDatedCostLedger()
    assumptions = (
        ScenarioAssumption(
            assumption_id="test-brokerage",
            component=LedgerComponent.BROKERAGE,
            product=LedgerProduct.INTRADAY,
            basis="turnover",
            rate=Decimal(brokerage_rate),
            formula="max(rate * turnover, 20)",
            source="synthetic-historical-broker-pricing",
            reason="broker historical tier contract",
        ),
        ScenarioAssumption(
            assumption_id="test-gst",
            component=LedgerComponent.GST,
            product=LedgerProduct.INTRADAY,
            basis="gst-schedule",
            rate=Decimal("0.18"),
            formula="rate * sum(charges)",
            source="synthetic-gst-schedule",
            reason="gst statutory rate assumption",
        ),
        ScenarioAssumption(
            assumption_id="test-clearing",
            component=LedgerComponent.CLEARING,
            product=LedgerProduct.INTRADAY,
            basis="clearing-rate",
            rate=Decimal("0.000001"),
            formula="rate * turnover",
            source="synthetic-clearing-schedule",
            reason="clearing fee assumption",
        ),
    )
    return compile_historical_scenario(
        scenario_id="scenario:synthetic-readiness-intraday",
        ledger=ledger,
        scenario_date=scenario_date,
        research_start=date(2026, 6, 1),
        research_end=date(2026, 7, 6),
        product=LedgerProduct.INTRADAY,
        assumptions=assumptions,
    )


def _promotion_evidence_with_scenario_eval(
    scenario: HistoricalCostScenario,
    winner_trade_count: int = 15,
    profit_factor: Decimal = Decimal("1.45"),
    net_return_pct: Decimal = Decimal("12.50"),
) -> ConcretePromotionEvidence:
    return ConcretePromotionEvidence(
        held_out_test=HeldOutTestEvidence(
            artifact_fingerprint=canonical_sha256({"test": "held_out_001"}),
            test_dataset_fingerprints=((STOCK_A, "fp_stock_a_test"),),
            window_id=1,
            trade_count=winner_trade_count,
            profit_factor=profit_factor,
            max_drawdown_pct=Decimal("4.50"),
            drawdown_basis=DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS,
            net_return_pct=net_return_pct,
            source_reference="held-out-test-log-001",
        ),
        cost_reconciliation=CostReconciliationEvidence(
            artifact_fingerprint=canonical_sha256({"cost": "recon_001"}),
            schema_version="upstox-cost-reconciliation/v1",
            cost_model_name="documented",
            orders_checked=10,
            passed_count=10,
            failed_count=0,
            max_reconciliation_error_inr=Decimal("0.005"),
            tolerance_inr=Decimal("0.01"),
            status="PASS",
        ),
        paper_trading=PaperTradingEvidence(
            artifact_fingerprint=canonical_sha256({"paper": "paper_001"}),
            environment="upstox_sandbox_v2",
            session_start=date(2026, 7, 7),
            session_end=date(2026, 7, 20),
            verified_orders_count=30,
            audit_log_fingerprint=canonical_sha256({"audit": "audit_log_001"}),
            source_reference="broker_sandbox_order_log",
        ),
        baseline_comparison=BaselineComparisonEvidence(
            artifact_fingerprint=canonical_sha256({"base": "comp_001"}),
            baseline_candidate_id="baseline:first-bar-hold",
            evaluated_candidate_id="orb:5m:vol1.2:buf5bps",
            baseline_net_return_pct=Decimal("5.00"),
            evaluated_net_return_pct=net_return_pct,
            outperformed=True,
        ),
        slippage_stress=SlippageStressEvidence(
            artifact_fingerprint=canonical_sha256({"slip": "stress_001"}),
            scenarios_evaluated=("base", "double_slippage", "triple_spread"),
            stress_max_drawdown_pct=Decimal("6.20"),
            stress_passed=True,
        ),
        event_simulation=EventDrivenSimulationEvidence(
            artifact_fingerprint=canonical_sha256({"sim": "sim_001"}),
            simulator_version="openalgo-event-simulator-v1",
            trade_count=winner_trade_count,
            initial_cash=Decimal(100000),
            final_cash=Decimal(112500),
        ),
        unpriced_cost_components=(),
    )


# -----------------------------------------------------------------------------
# Comprehensive End-to-End Pipeline Readiness Test
# -----------------------------------------------------------------------------


def test_research_execution_readiness_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the complete 11-stage research execution chain end-to-end.

    validated research input
    → frozen PIT population
    → explicit strategy candidate grid
    → VectorBT screening
    → stock × strategy tournament
    → repeated chronological WFO
    → untouched OOS
    → event simulator
    → canonical HistoricalCostScenario
    → immutable Experiment
    → promotion gate
    """
    # 1. Validated Research Input
    # 26 business days: June 1, 2026 to July 6, 2026
    trading_dates = _generate_business_days(date(2026, 6, 1), 26)
    assert len(trading_dates) == 26
    research_start = trading_dates[0]
    research_end = trading_dates[-1]

    # Multiple stocks with continuous bars + auxiliary CAS bars
    frame_a = _make_intraday_bars_with_cas(trading_dates, base_price=100.0, daily_drift=0.5)
    frame_b = _make_intraday_bars_with_cas(trading_dates, base_price=200.0, daily_drift=-0.2)

    manifest_a = _make_manifest("token_a", "STOCK_A")
    manifest_b = _make_manifest("token_b", "STOCK_B")
    fp_a = dataframe_fingerprint(frame_a, manifest_a)
    fp_b = dataframe_fingerprint(frame_b, manifest_b)

    assert fp_a != fp_b
    assert len(fp_a) == 64

    # 2. Frozen PIT Population
    # Two stocks eligible at formation boundary; Stock C is later-listed
    frozen_instruments = (STOCK_A, STOCK_B)
    segments: list[PITMembershipSegment] = [
        PITMembershipSegment(
            instrument_key=key,
            valid_from=research_start,
            valid_to=research_end,
            evidence_as_of=research_start,
            source_fingerprint=canonical_sha256(
                {"segment": key, "start": research_start.isoformat()}
            ),
            eligible=True,
        )
        for key in frozen_instruments
    ]

    # 3. Explicit Strategy Candidate Grid
    candidates = _candidate_strategies()
    assert len(candidates) == 8
    param_grid = _parameter_grid_dict()
    grid_fp = canonical_sha256(param_grid)
    assert len(grid_fp) == 64

    # 4. VectorBT Screening
    # Fast portfolio mock for Python 3.14 numba compatibility
    import vectorbt as vbt

    class _FastPortfolio:
        @staticmethod
        def from_signals(close, **kwargs):
            return _FastPortfolio()

        def stats(self, settings):
            return {
                "Total Return [%]": 2.5,
                "Max Drawdown [%]": 1.2,
                "Total Closed Trades": 6,
                "Win Rate [%]": 66.7,
                "Profit Factor": 1.9,
            }

    monkeypatch.setattr(vbt, "Portfolio", _FastPortfolio)

    session_policy = NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=15)
    cont_frame_a = filter_to_continuous_session(frame_a, session_policy)
    # CAS bars (15:45) must be excluded from continuous frame
    assert not any(ts.time() == time(15, 45) for ts in cont_frame_a.index)

    # Generate signals from baseline
    signals = baseline_definition(session_open=time(9, 15)).build_signals(cont_frame_a)
    screening_result = screen_long_signals(
        close=cont_frame_a["close"],
        execution_price=cont_frame_a["open"],
        entries_at_close=signals.entries_at_close,
        exits_at_close=signals.exits_at_close,
        signal_lag_bars=1,
        screening_cash=Decimal(100000),
        screening_fee_rate=Decimal("0.0005"),
        screening_slippage_rate=Decimal("0.0002"),
        frequency="5m",
        session_policy=session_policy,
    )
    assert screening_result.exact_cost_validated is False
    assert screening_result.signal_lag_bars == 1
    assert screening_result.closed_trades > 0

    # 5. Stock × Strategy Tournament
    # Fold 1 Train window: days 0..5
    fold_train_dates = trading_dates[:6]
    train_frame_a = cont_frame_a.loc[cont_frame_a.index.date <= fold_train_dates[-1]]
    cont_frame_b = filter_to_continuous_session(frame_b, session_policy)
    train_frame_b = cont_frame_b.loc[cont_frame_b.index.date <= fold_train_dates[-1]]

    train_universe = ResearchUniverseBuildResult(
        selection_cutoff=fold_train_dates[-1],
        audits=(
            _audit_for(STOCK_A, fp_a, train_frame_a),
            _audit_for(STOCK_B, fp_b, train_frame_b),
        ),
    )
    train_inputs = [
        CrossSectionalInstrumentInput(
            instrument_key=STOCK_A,
            symbol="STOCK_A",
            frame=train_frame_a,
            dataset_fingerprint=fp_a,
            historical_membership=_membership_for(STOCK_A, fold_train_dates),
            tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
            session_policy=session_policy,
        ),
        CrossSectionalInstrumentInput(
            instrument_key=STOCK_B,
            symbol="STOCK_B",
            frame=train_frame_b,
            dataset_fingerprint=fp_b,
            historical_membership=_membership_for(STOCK_B, fold_train_dates),
            tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
            session_policy=session_policy,
        ),
    ]
    cost_provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))
    fills = FillAssumptions(
        slippage_bps_per_leg=Decimal("1.0"),
        half_spread_bps_per_leg=Decimal("0.5"),
    )
    sim_config = IntradaySimulationConfig(
        initial_cash=Decimal(100000),
        max_trades_per_day=2,
    )

    tournament_result = run_cross_sectional_tournament(
        universe=train_universe,
        instruments=train_inputs,
        candidates=candidates,
        ranking_metric=RankingMetric.NET_RETURN_PCT,
        min_trade_count=1,
        cost_provider=cost_provider,
        fills=fills,
        simulation_config=sim_config,
    )
    assert tournament_result.winner is not None
    assert tournament_result.winner.instrument_key in frozen_instruments
    winner_instrument = tournament_result.winner.instrument_key
    winner_candidate_id = tournament_result.winner.candidate.candidate_id

    # 6. Repeated Chronological WFO
    # Compile multi-fold WFO schedule
    wfo_folds = plan_wfo_date_windows(
        trading_dates[:21],  # 21 days prefix for repeated folds
        train_trading_days=6,
        test_trading_days=4,
        step_trading_days=4,
        embargo_trading_days=1,
    )
    assert len(wfo_folds) >= 2  # Proves multiple chronological folds

    # Execute Fold 1 walk-forward window
    fold1 = wfo_folds[0]
    test_frame_a = cont_frame_a.loc[
        (cont_frame_a.index.date >= fold1.test_dates[0])
        & (cont_frame_a.index.date <= fold1.test_dates[-1])
    ]
    test_frame_b = cont_frame_b.loc[
        (cont_frame_b.index.date >= fold1.test_dates[0])
        & (cont_frame_b.index.date <= fold1.test_dates[-1])
    ]
    test_inputs = [
        CrossSectionalTestInstrumentInput(
            instrument_key=STOCK_A,
            symbol="STOCK_A",
            frame=test_frame_a,
            dataset_fingerprint=fp_a,
            historical_membership=_membership_for(STOCK_A, fold1.test_dates),
            tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
            session_policy=session_policy,
            corporate_actions=DatedCorporateActionEvidence(
                window_start=fold1.test_dates[0],
                window_end=fold1.test_dates[-1],
                assessment=CorporateActionAssessment(complete=True, blocking_events=()),
                source="synthetic-ca",
            ),
        ),
        CrossSectionalTestInstrumentInput(
            instrument_key=STOCK_B,
            symbol="STOCK_B",
            frame=test_frame_b,
            dataset_fingerprint=fp_b,
            historical_membership=_membership_for(STOCK_B, fold1.test_dates),
            tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
            session_policy=session_policy,
            corporate_actions=DatedCorporateActionEvidence(
                window_start=fold1.test_dates[0],
                window_end=fold1.test_dates[-1],
                assessment=CorporateActionAssessment(complete=True, blocking_events=()),
                source="synthetic-ca",
            ),
        ),
    ]
    wfo_result = run_cross_sectional_walk_forward_window(
        window=WalkForwardWindow(
            window_id=fold1.window_id,
            train_dates=fold1.train_dates,
            test_dates=fold1.test_dates,
        ),
        train_universe=train_universe,
        train_instruments=train_inputs,
        test_instruments=test_inputs,
        candidates=candidates,
        ranking_metric=RankingMetric.NET_RETURN_PCT,
        min_train_trade_count=1,
        cost_provider=cost_provider,
        fills=fills,
        simulation_config=sim_config,
    )
    assert wfo_result.selected_instrument_key == winner_instrument
    assert wfo_result.selected_candidate_id == winner_candidate_id
    assert wfo_result.test_evaluation is not None

    # 7. Untouched OOS (Final Test)
    cost_claim = CostEvidenceClaim(
        ledger_schema_version="effective-dated-cost-ledger/v1",
        ledger_fingerprint="a" * 64,
        evidence_classification="INCOMPLETE_HISTORICAL_EVIDENCE",
        historical_actual=False,
        product_scope="INTRADAY",
        evidence_mode="historical_resolution",
        policy_identity="effective-dated-cost-ledger/default-resolution/v1",
        resolved_on_date=date(2026, 6, 30),
        selected_record_ids=("stt:INTRADAY:SELL:2024-07-01:statutory_schedule",),
        unknown_components=("gst: unknown",),
        scenario_identity=None,
    )
    ca_claim = CorporateActionClaim(
        policy="ca-policy-v1",
        coverage="full-window-coverage",
        population=frozen_instruments,
        complete=True,
        fingerprint="b" * 64,
    )
    strategy_claims = (
        StrategyDefinitionClaim(
            strategy_name=winner_candidate_id,
            parameters=(("buf_bps", "5"),),
            source_refs=("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5198458",),
        ),
    )
    plan = compile_repeated_wfo(
        research_start=research_start,
        research_end=research_end,
        trading_dates=trading_dates,
        fold_train_days=6,
        fold_validation_days=4,
        fold_step_days=4,
        fold_embargo_days=1,
        final_test_days=4,
        final_embargo_days=1,
        min_observations_per_window=2,
        min_folds=2,
        approved_capital_rupees=Decimal(100000),
        session_policy_id="NSEEquitySessionPolicy/buf15",
        cas_policy_id="CAS/eligible False",
        cost_claim=cost_claim,
        ca_claim=ca_claim,
        dataset_fingerprints={STOCK_A: fp_a, STOCK_B: fp_b},
        universe_policy_id="nse-cm-point-in-time-v1",
        pit_segments=tuple(segments),
        train_universe_instruments=frozen_instruments,
        strategy_definitions=strategy_claims,
    )
    assert plan.final_test.role is WindowRole.UNTOUCHED_TEST
    assert len(plan.final_test.trading_dates) == 4
    # Untouched test strictly after all fold dates
    assert plan.final_test.trading_dates[0] > max(plan.selection_union_dates())
    # Adapter blocks direct integration, proving architectural boundary
    adapter = plan.to_experiment_adapter()
    assert adapter["direct_experiment_integration"]["blocked"] is True

    # 8. Event Simulator
    # Simulating survivor winner with event-driven simulator
    signals_winner = next(
        c for c in candidates if c.candidate_id == winner_candidate_id
    ).build_signals(train_frame_a)
    sim_result = simulate_long_intraday(
        frame=train_frame_a,
        entries_at_close=signals_winner.entries_at_close,
        exits_at_close=signals_winner.exits_at_close,
        instrument_token="token_a",
        exchange=Exchange.NSE,
        cost_provider=cost_provider,
        fills=fills,
        session_policy=session_policy,
        tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
        trading_eligibility_policy=HistoricalTradingEligibilityPolicy(
            _membership_for(STOCK_A, fold_train_dates)
        ),
        config=sim_config,
    )
    assert sim_result.initial_cash == Decimal(100000)
    assert len(sim_result.trades) > 0
    assert all(
        trade.exit_reason in (ExitReason.SIGNAL, ExitReason.SESSION_CUTOFF)
        for trade in sim_result.trades
    )

    # 9. Canonical HistoricalCostScenario
    cost_scenario = _canonical_historical_cost_scenario()
    assert cost_scenario.classification == SCENARIO_LABEL
    assert cost_scenario.historical_actual is False

    cost_identity = CostEvidenceIdentity.from_historical_scenario(
        cost_scenario,
        expected_product=LedgerProduct.INTRADAY,
    )
    assert cost_identity.historical_actual is False
    assert cost_identity.evidence_classification == SCENARIO_LABEL
    assert cost_identity.scenario_identity == cost_scenario.fingerprint()

    # 10. Immutable Experiment
    orchestrator = ExperimentOrchestrator(code_commit_sha=CODE_COMMIT_SHA)
    assert orchestrator.live_orders_called is False

    train_window_spec = WindowSpec(
        window_id=1,
        start=fold1.train_dates[0],
        end=fold1.train_dates[-1],
        trading_days=len(fold1.train_dates),
    )
    test_window_spec = WindowSpec(
        window_id=1,
        start=fold1.test_dates[0],
        end=fold1.test_dates[-1],
        trading_days=len(fold1.test_dates),
    )
    promotion_evidence = _promotion_evidence_with_scenario_eval(cost_scenario)
    prefilter_artifact = train_universe.as_prefilter_artifact()
    universe_fp = compute_prefilter_artifact_fingerprint(prefilter_artifact)

    experiment = orchestrator.build_experiment(
        research_window=ResearchWindowConfig(start=research_start, end=research_end),
        train_windows=(train_window_spec,),
        validation_test_windows=(test_window_spec,),
        embargo=EmbargoSpec(trading_days=1),
        approved_capital=ApprovedCapital(amount_rupees=Decimal(100000), currency="INR"),
        universe_fingerprint=universe_fp,
        candidate_prefilter_artifact_fingerprint=universe_fp,
        instrument_dataset_fingerprints={STOCK_A: fp_a, STOCK_B: fp_b},
        nse_membership_evidence=NSEMembershipEvidenceIdentity(
            source_refs=("synthetic-nse-membership-v1",),
            complete=True,
            coverage_fingerprint=canonical_sha256({"membership": "coverage_ok"}),
            eligible_dates_count=len(trading_dates),
        ),
        tick_evidence=TickEvidenceIdentity(
            policy_name="fixed-0.05",
            source="synthetic-tick-evidence",
            coverage_complete=True,
            coverage_fingerprint=canonical_sha256({"ticks": "fixed-0.05"}),
        ),
        session_policy_identity=SessionPolicyIdentity(
            policy_name="NSEEquitySessionPolicy",
            cas_eligible=False,
            exit_buffer_minutes=15,
            cas_effective_date="2026-03-01",
            continuous_end="15:30:00",
        ),
        corporate_action_evidence=CorporateActionEvidenceIdentity(
            source="synthetic-ca-evidence",
            complete=True,
            blocking_events=(),
            evidence_fingerprint=canonical_sha256({"ca": "evidence_ok"}),
        ),
        cost_model_identity=CostModelIdentity(
            model_name="canonical-historical-cost-scenario",
            effective_date=cost_scenario.scenario_date.isoformat(),
            rates={"brokerage": "0.0003", "stt": "0.00025"},
            source_refs=("historical_cost_scenario.py",),
        ),
        cost_evidence_identity=cost_identity,
        cost_evidence_class=SCENARIO_LABEL,
        strategy_definitions=(
            StrategySpec(
                candidate_id=winner_candidate_id,
                strategy_name=winner_candidate_id,
                research_basis="Study candidate",
                source_refs=(),
                parameters={"buf_bps": "5"},
            ),
        ),
        parameter_grid=param_grid,
        friction_scenarios=(
            FrictionScenarioSpec(
                scenario_id="base",
                slippage_bps_per_leg=Decimal("1.0"),
                half_spread_bps_per_leg=Decimal("0.5"),
            ),
        ),
        rejected_candidates=(),
        tournament_result={"winner": winner_candidate_id, "winner_instrument": winner_instrument},
        walk_forward_result={"fold1_winner": winner_instrument, "window_id": 1},
        promotion_evidence=promotion_evidence,
    )
    assert experiment.schema_version == EXPERIMENT_SCHEMA_VERSION
    assert experiment.live_orders_called is False
    assert experiment.experiment_id.startswith("exp_")
    experiment.validate_integrity()

    # 11. Promotion Gate
    thresholds = PromotionThresholds(
        min_trades=10,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal("10.0"),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.05"),
    )
    passed, violations = experiment.evaluate_promotion_gate(thresholds)
    # Scenario cost evidence CANNOT promote: must strictly fail closed
    assert passed is False
    assert any("scenario/incomplete evidence cannot promote" in v for v in violations)


# -----------------------------------------------------------------------------
# Detailed Invariant Proofs
# -----------------------------------------------------------------------------


def test_pit_population_no_lookahead_and_later_listed_isolation() -> None:
    """Prove that later-listed stocks cannot contaminate earlier populations and

    snapshot evidence cannot attest earlier dates.
    """
    trading_dates = _generate_business_days(date(2026, 6, 1), 26)
    research_start = trading_dates[0]
    research_end = trading_dates[-1]

    # STOCK_C is listed on day 10 (2026-06-15)
    listing_date = trading_dates[10]

    # 1. Attempting to use a later snapshot to attest earlier dates fails closed
    with pytest.raises(
        WindowLeakageError, match="a later snapshot cannot attest earlier membership"
    ):
        PITMembershipSegment(
            instrument_key=STOCK_C,
            valid_from=research_start,
            valid_to=trading_dates[5],
            evidence_as_of=listing_date,  # Future snapshot
            source_fingerprint="c" * 64,
            eligible=True,
        )

    # 2. resolve_pit_segment strictly enforces evidence_as_of <= trade_date
    valid_segment = PITMembershipSegment(
        instrument_key=STOCK_C,
        valid_from=listing_date,
        valid_to=research_end,
        evidence_as_of=listing_date,
        source_fingerprint="c" * 64,
        eligible=True,
    )
    # Resolving for a date before listing raises WindowLeakageError
    with pytest.raises(WindowLeakageError, match="resolve to exactly one PIT membership record"):
        resolve_pit_segment((valid_segment,), trading_dates[0])

    # 3. Attempting to include later-listed STOCK_C in formation-boundary universe fails
    stock_a_segment = PITMembershipSegment(
        instrument_key=STOCK_A,
        valid_from=research_start,
        valid_to=research_end,
        evidence_as_of=research_start,
        source_fingerprint="a" * 64,
        eligible=True,
    )
    # Attempt to compile plan claiming STOCK_C in train universe without formation PIT coverage
    with pytest.raises(FrozenUniverseViolationError, match="formation-boundary PIT coverage"):
        compile_repeated_wfo(
            research_start=research_start,
            research_end=research_end,
            trading_dates=trading_dates,
            fold_train_days=6,
            fold_validation_days=4,
            fold_step_days=4,
            fold_embargo_days=1,
            final_test_days=4,
            final_embargo_days=1,
            min_observations_per_window=2,
            min_folds=2,
            approved_capital_rupees=Decimal(100000),
            session_policy_id="NSEEquitySessionPolicy/buf15",
            cas_policy_id="CAS/eligible False",
            cost_claim=CostEvidenceClaim(
                ledger_schema_version="effective-dated-cost-ledger/v1",
                ledger_fingerprint="a" * 64,
                evidence_classification="INCOMPLETE_HISTORICAL_EVIDENCE",
                historical_actual=False,
                product_scope="INTRADAY",
                evidence_mode="historical_resolution",
                policy_identity="effective-dated-cost-ledger/default-resolution/v1",
                resolved_on_date=date(2026, 6, 30),
                selected_record_ids=("stt:INTRADAY:SELL:2024-07-01:statutory_schedule",),
                unknown_components=("gst: unknown",),
                scenario_identity=None,
            ),
            ca_claim=CorporateActionClaim(
                policy="ca-policy-v1",
                coverage="full-window-coverage",
                population=(STOCK_A, STOCK_C),
                complete=True,
                fingerprint="b" * 64,
            ),
            dataset_fingerprints={STOCK_A: "fp_a", STOCK_C: "fp_c"},
            universe_policy_id="nse-cm-point-in-time-v1",
            pit_segments=(stock_a_segment, valid_segment),
            train_universe_instruments=(STOCK_A, STOCK_C),
            strategy_definitions=(
                StrategyDefinitionClaim(
                    strategy_name="baseline",
                    parameters=(("p", "1"),),
                ),
            ),
        )


def test_candidate_grid_explicit_and_fingerprinted() -> None:
    """Prove candidate grids are explicit, non-empty, and cryptographically fingerprinted."""
    candidates = _candidate_strategies()
    assert len(candidates) == 8
    candidate_ids = [c.candidate_id for c in candidates]
    # No duplicate candidate IDs
    assert len(set(candidate_ids)) == len(candidate_ids)

    # Explicit research basis and source references
    for c in candidates:
        assert isinstance(c.candidate_id, str) and c.candidate_id
        assert isinstance(c.research_basis, str) and c.research_basis

    # Parameter grid dictionary and fingerprint
    param_grid = _parameter_grid_dict()
    grid_fp = canonical_sha256(param_grid)
    assert len(grid_fp) == 64

    # Any material modification to grid changes its fingerprint
    modified_grid = dict(param_grid)
    modified_grid["momentum_lookback"] = ("5",)
    modified_fp = canonical_sha256(modified_grid)
    assert modified_fp != grid_fp


def test_vectorbt_screening_invariants_and_cas_exclusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove VectorBT screening enforces same-bar execution forbidden,

    final signal cannot execute next day, and CAS auxiliary bars are excluded.
    """
    trading_dates = _generate_business_days(date(2026, 6, 1), 3)
    frame_with_cas = _make_intraday_bars_with_cas(trading_dates, include_cas_bar=True)

    # 1. CAS auxiliary bars excluded: verify 15:45 is present in raw input
    assert any(ts.time() == time(15, 45) for ts in frame_with_cas.index)

    session_policy = NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=15)
    filtered = filter_to_continuous_session(frame_with_cas, session_policy)
    # After continuous filtering, no CAS auxiliary bar exists
    assert not any(ts.time() == time(15, 45) for ts in filtered.index)
    assert all(ts.time() <= time(15, 30) for ts in filtered.index)

    # 2. same-bar close execution forbidden: shift_close_generated_signals requires lag_bars >= 1
    entries = pd.Series([True, False, True], index=filtered.index[:3])
    exits = pd.Series([False, True, False], index=filtered.index[:3])
    with pytest.raises(ValueError, match="close-generated signals require lag_bars >= 1"):
        shift_close_generated_signals(entries, exits, lag_bars=0)

    # 3. final signal cannot cross into next trading day
    day1_close_idx = [i for i, ts in enumerate(filtered.index) if ts.date() == trading_dates[0]][-1]
    entries_final = pd.Series(False, index=filtered.index)
    entries_final.iloc[day1_close_idx] = True  # Signal on last bar of day 1
    exits_dummy = pd.Series(False, index=filtered.index)

    shifted_entries, _ = shift_close_generated_signals(entries_final, exits_dummy, lag_bars=1)
    # The first bar of day 2 must NOT execute the day 1 close signal
    day2_first_idx = day1_close_idx + 1
    assert (
        shifted_entries.iloc[day2_first_idx] is False or shifted_entries.iloc[day2_first_idx] == 0
    )

    # 4. Screening result confirms exact_cost_validated=False
    import vectorbt as vbt

    class _FastPortfolio:
        @staticmethod
        def from_signals(close, **kwargs):
            return _FastPortfolio()

        def stats(self, settings):
            return {
                "Total Return [%]": 1.5,
                "Max Drawdown [%]": 0.5,
                "Total Closed Trades": 4,
                "Win Rate [%]": 75.0,
                "Profit Factor": 2.0,
            }

    monkeypatch.setattr(vbt, "Portfolio", _FastPortfolio)

    screening_res = screen_long_signals(
        close=filtered["close"],
        execution_price=filtered["open"],
        entries_at_close=pd.Series(False, index=filtered.index),
        exits_at_close=pd.Series(False, index=filtered.index),
        signal_lag_bars=1,
        screening_cash=Decimal(10000),
        screening_fee_rate=Decimal("0.0005"),
        screening_slippage_rate=Decimal("0.0002"),
        frequency="5min",
        session_policy=session_policy,
    )
    assert screening_res.exact_cost_validated is False
    assert screening_res.signal_lag_bars == 1
    assert screening_res.closed_trades == 4


def test_tournament_requires_full_frozen_population_no_cherry_picking() -> None:
    """Prove that tournament requires the complete frozen eligible universe.

    Dropping any instrument after performance is known is strictly rejected.
    """
    trading_dates = _generate_business_days(date(2026, 6, 1), 5)
    frame_a = _make_intraday_bars_with_cas(trading_dates, base_price=100.0, daily_drift=1.0)
    frame_b = _make_intraday_bars_with_cas(trading_dates, base_price=100.0, daily_drift=0.1)

    session_policy = NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=15)
    cont_a = filter_to_continuous_session(frame_a, session_policy)
    cont_b = filter_to_continuous_session(frame_b, session_policy)

    universe = ResearchUniverseBuildResult(
        selection_cutoff=trading_dates[-1],
        audits=(
            _audit_for(STOCK_A, "fp_a", cont_a),
            _audit_for(STOCK_B, "fp_b", cont_b),
        ),
    )
    input_a = CrossSectionalInstrumentInput(
        instrument_key=STOCK_A,
        symbol="STOCK_A",
        frame=cont_a,
        dataset_fingerprint="fp_a",
        historical_membership=_membership_for(STOCK_A, trading_dates),
        tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
        session_policy=session_policy,
    )
    # Omitting STOCK_B fails closed
    with pytest.raises(ValueError, match="missing eligible instruments: NSE_EQ\\|STOCK_B"):
        run_cross_sectional_tournament(
            universe=universe,
            instruments=[input_a],  # STOCK_B cherry-picked out
            candidates=[baseline_definition(session_open=time(9, 15))],
            ranking_metric=RankingMetric.NET_RETURN_PCT,
            min_trade_count=1,
            cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
            fills=FillAssumptions(
                slippage_bps_per_leg=Decimal(0), half_spread_bps_per_leg=Decimal(0)
            ),
            simulation_config=IntradaySimulationConfig(
                initial_cash=Decimal(1000), max_trades_per_day=1
            ),
        )


def test_wfo_chronological_isolation_and_frozen_winner_behavior() -> None:
    """Prove train cannot observe test, and superior test performance cannot alter

    the frozen winner selected during train.
    """
    train_day = date(2026, 6, 1)
    test_day = date(2026, 6, 2)
    window = WalkForwardWindow(
        window_id=1,
        train_dates=(train_day,),
        test_dates=(test_day,),
    )
    session_policy = NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=15)

    # STOCK_A wins train (strong rally); STOCK_B wins test (huge rally)
    a_train = filter_to_continuous_session(
        _make_intraday_bars_with_cas((train_day,), base_price=100.0, daily_drift=10.0),
        session_policy,
    )
    b_train = filter_to_continuous_session(
        _make_intraday_bars_with_cas((train_day,), base_price=100.0, daily_drift=1.0),
        session_policy,
    )
    a_test = filter_to_continuous_session(
        _make_intraday_bars_with_cas((test_day,), base_price=110.0, daily_drift=-5.0),
        session_policy,
    )
    b_test = filter_to_continuous_session(
        _make_intraday_bars_with_cas((test_day,), base_price=101.0, daily_drift=50.0),
        session_policy,
    )

    universe = ResearchUniverseBuildResult(
        selection_cutoff=train_day,
        audits=(
            _audit_for(STOCK_A, "fp_a_train", a_train),
            _audit_for(STOCK_B, "fp_b_train", b_train),
        ),
    )
    train_inputs = [
        CrossSectionalInstrumentInput(
            instrument_key=STOCK_A,
            symbol="STOCK_A",
            frame=a_train,
            dataset_fingerprint="fp_a_train",
            historical_membership=_membership_for(STOCK_A, (train_day,)),
            tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
            session_policy=session_policy,
        ),
        CrossSectionalInstrumentInput(
            instrument_key=STOCK_B,
            symbol="STOCK_B",
            frame=b_train,
            dataset_fingerprint="fp_b_train",
            historical_membership=_membership_for(STOCK_B, (train_day,)),
            tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
            session_policy=session_policy,
        ),
    ]
    test_inputs = [
        CrossSectionalTestInstrumentInput(
            instrument_key=STOCK_A,
            symbol="STOCK_A",
            frame=a_test,
            dataset_fingerprint="fp_a_test",
            historical_membership=_membership_for(STOCK_A, (test_day,)),
            tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
            session_policy=session_policy,
            corporate_actions=DatedCorporateActionEvidence(
                window_start=test_day,
                window_end=test_day,
                assessment=CorporateActionAssessment(complete=True, blocking_events=()),
                source="synthetic-ca",
            ),
        ),
        CrossSectionalTestInstrumentInput(
            instrument_key=STOCK_B,
            symbol="STOCK_B",
            frame=b_test,
            dataset_fingerprint="fp_b_test",
            historical_membership=_membership_for(STOCK_B, (test_day,)),
            tick_size_policy=FixedTickSizePolicy(tick_size_rupees=Decimal("0.05"), source="test"),
            session_policy=session_policy,
            corporate_actions=DatedCorporateActionEvidence(
                window_start=test_day,
                window_end=test_day,
                assessment=CorporateActionAssessment(complete=True, blocking_events=()),
                source="synthetic-ca",
            ),
        ),
    ]

    result = run_cross_sectional_walk_forward_window(
        window=window,
        train_universe=universe,
        train_instruments=train_inputs,
        test_instruments=test_inputs,
        candidates=[baseline_definition(session_open=time(9, 15))],
        ranking_metric=RankingMetric.NET_RETURN_PCT,
        min_train_trade_count=1,
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
        fills=FillAssumptions(slippage_bps_per_leg=Decimal(0), half_spread_bps_per_leg=Decimal(0)),
        simulation_config=IntradaySimulationConfig(
            initial_cash=Decimal(1000), max_trades_per_day=1
        ),
    )
    # STOCK_A is selected on train and remains the evaluated instrument on test
    assert result.selected_instrument_key == STOCK_A
    assert result.selected_candidate_id == "baseline:first-bar-hold"
    assert result.test_evaluation.candidate_id == "baseline:first-bar-hold"
    assert result.test_evaluation.simulation.trades[0].entry_timestamp.date() == test_day


def test_untouched_oos_isolation_and_no_direct_plan_bridge() -> None:
    """Prove untouched OOS is strictly isolated from selection and cannot influence selection,

    and verify the architecture blocks direct ResearchWindowPlan -> Experiment integration.
    """
    trading_dates = _generate_business_days(date(2026, 6, 1), 26)
    research_start = trading_dates[0]
    research_end = trading_dates[-1]

    frozen_instruments = (STOCK_A, STOCK_B)
    segments = [
        PITMembershipSegment(
            instrument_key=key,
            valid_from=research_start,
            valid_to=research_end,
            evidence_as_of=research_start,
            source_fingerprint=canonical_sha256({"segment": key}),
            eligible=True,
        )
        for key in frozen_instruments
    ]

    cost_claim = CostEvidenceClaim(
        ledger_schema_version="effective-dated-cost-ledger/v1",
        ledger_fingerprint="a" * 64,
        evidence_classification="INCOMPLETE_HISTORICAL_EVIDENCE",
        historical_actual=False,
        product_scope="INTRADAY",
        evidence_mode="historical_resolution",
        policy_identity="effective-dated-cost-ledger/default-resolution/v1",
        resolved_on_date=date(2026, 6, 30),
        selected_record_ids=("stt:INTRADAY:SELL:2024-07-01:statutory_schedule",),
        unknown_components=("gst: unknown",),
        scenario_identity=None,
    )
    ca_claim = CorporateActionClaim(
        policy="ca-policy-v1",
        coverage="full-window-coverage",
        population=frozen_instruments,
        complete=True,
        fingerprint="b" * 64,
    )
    strategy_claims = (
        StrategyDefinitionClaim(
            strategy_name="baseline",
            parameters=(("session_open", "09:15"),),
        ),
    )

    plan = compile_repeated_wfo(
        research_start=research_start,
        research_end=research_end,
        trading_dates=trading_dates,
        fold_train_days=6,
        fold_validation_days=4,
        fold_step_days=4,
        fold_embargo_days=1,
        final_test_days=4,
        final_embargo_days=1,
        min_observations_per_window=2,
        min_folds=2,
        approved_capital_rupees=Decimal(100000),
        session_policy_id="NSEEquitySessionPolicy/buf15",
        cas_policy_id="CAS/eligible False",
        cost_claim=cost_claim,
        ca_claim=ca_claim,
        dataset_fingerprints={STOCK_A: "fp_a", STOCK_B: "fp_b"},
        universe_policy_id="nse-cm-point-in-time-v1",
        pit_segments=tuple(segments),
        train_universe_instruments=frozen_instruments,
        strategy_definitions=strategy_claims,
    )

    # 1. OOS window is strictly after all folds and separated by embargo
    assert plan.final_test.role is WindowRole.UNTOUCHED_TEST
    assert plan.final_embargo.role is WindowRole.EMBARGO
    assert len(plan.final_test.trading_dates) == 4
    assert len(plan.final_embargo.trading_dates) == 1

    # 2. OOS dates are disjoint from fold selection dates
    selection_dates = set(plan.selection_union_dates())
    oos_dates = set(plan.final_test.trading_dates)
    embargo_dates = set(plan.final_embargo.trading_dates)
    assert selection_dates.isdisjoint(oos_dates)
    assert selection_dates.isdisjoint(embargo_dates)
    assert max(selection_dates) < min(embargo_dates) < min(oos_dates)

    # 3. OOS dates cannot enter selection attestation
    bad_attestation = SelectionAttestation(
        plan_fingerprint=plan.fingerprint_without_selection(),
        selection_pipeline_id="pipe-1",
        candidate_definition_fingerprint="d" * 64,
        observed_selection_dates=plan.selection_union_dates(),
        dataset_fingerprints=tuple(plan.dataset_fingerprints),
        strategy_definition_fingerprint="e" * 64,
        parameter_grid_fingerprint="f" * 64,
        ranking_artifact_fingerprint="9" * 64,
        winner_instrument=STOCK_A,
        winner_strategy="baseline",
        winner_parameters=(("session_open", "09:15"),),
        selection_dates=(plan.final_test.trading_dates[0],),  # Leaking OOS date into selection!
    )
    with pytest.raises(UntouchedTestViolationError, match="outside fold train\\+validation"):
        plan.select_train_winner(attestation=bad_attestation)

    # 4. Valid attestation binds and authorizes untouched test
    valid_attestation = SelectionAttestation(
        plan_fingerprint=plan.fingerprint_without_selection(),
        selection_pipeline_id="pipe-1",
        candidate_definition_fingerprint="d" * 64,
        observed_selection_dates=plan.selection_union_dates(),
        dataset_fingerprints=tuple(plan.dataset_fingerprints),
        strategy_definition_fingerprint="e" * 64,
        parameter_grid_fingerprint="f" * 64,
        ranking_artifact_fingerprint="9" * 64,
        winner_instrument=STOCK_A,
        winner_strategy="baseline",
        winner_parameters=(("session_open", "09:15"),),
        selection_dates=(plan.selection_union_dates()[0],),
    )
    updated_plan, winner = plan.select_train_winner(attestation=valid_attestation)
    auth_ticket = updated_plan.authorize_untouched_test(
        frozen_universe=plan.frozen_train_universe,
        train_winner=winner,
    )
    assert auth_ticket["plan_id"] == updated_plan.plan_id
    assert auth_ticket["selection_forbidden"] is True
    assert auth_ticket["selection_attestation_fingerprint"] == valid_attestation.fingerprint()

    # 5. Architecture blocks direct ResearchWindowPlan -> Experiment bridge
    adapter = plan.to_experiment_adapter()
    assert adapter["direct_experiment_integration"]["blocked"] is True
    assert "cannot losslessly represent" in adapter["direct_experiment_integration"]["reason"]


def test_canonical_historical_cost_scenario_and_unknown_not_zero() -> None:
    """Prove canonical HistoricalCostScenario enforces UNKNOWN != ZERO and historical_actual=False."""
    ledger = EffectiveDatedCostLedger()

    # 1. Missing assumption for required component fails closed (UNKNOWN != ZERO)
    with pytest.raises(MissingScenarioAssumptionError, match="unknown is never zero"):
        compile_historical_scenario(
            scenario_id="scenario:missing-assumption",
            ledger=ledger,
            scenario_date=date(2026, 6, 5),
            research_start=date(2026, 6, 1),
            research_end=date(2026, 7, 6),
            product=LedgerProduct.INTRADAY,
            assumptions=(),  # No assumptions provided for unknown brokerage
        )

    # 2. Canonical scenario has historical_actual=False
    scenario = _canonical_historical_cost_scenario()
    assert scenario.classification == SCENARIO_LABEL
    assert scenario.historical_actual is False

    # 3. Direct construction with historical_actual=True raises
    with pytest.raises(
        ScenarioError, match="historical scenarios can never become HISTORICAL_ACTUAL_COSTS"
    ):
        HistoricalCostScenario(
            scenario_id="invalid",
            schema_version=scenario.schema_version,
            research_start=scenario.research_start,
            research_end=scenario.research_end,
            scenario_date=scenario.scenario_date,
            product=scenario.product,
            ledger_fingerprint=scenario.ledger_fingerprint,
            known_record_ids=scenario.known_record_ids,
            evidence_classes=scenario.evidence_classes,
            assumptions=scenario.assumptions,
            assumed_components=scenario.assumed_components,
            unknowns=scenario.unknowns,
            resolved_rates=scenario.resolved_rates,
            classification=SCENARIO_LABEL,
            historical_actual=True,  # Forbidden
        )


def test_experiment_fingerprint_changes_on_material_input() -> None:
    """Prove material changes to strategy parameters, dataset fingerprints, or cost assumptions

    strictly alter the deterministic experiment fingerprint and experiment identity.
    """
    cost_scenario_1 = _canonical_historical_cost_scenario(brokerage_rate="0.0003")
    cost_scenario_2 = _canonical_historical_cost_scenario(brokerage_rate="0.0004")

    identity_1 = CostEvidenceIdentity.from_historical_scenario(cost_scenario_1)
    identity_2 = CostEvidenceIdentity.from_historical_scenario(cost_scenario_2)

    orchestrator = ExperimentOrchestrator(code_commit_sha=CODE_COMMIT_SHA)
    param_grid = _parameter_grid_dict()

    exp_1 = orchestrator.build_experiment(
        research_window=ResearchWindowConfig(start=date(2026, 6, 1), end=date(2026, 7, 6)),
        train_windows=(WindowSpec(1, date(2026, 6, 1), date(2026, 6, 10), 8),),
        validation_test_windows=(WindowSpec(1, date(2026, 6, 12), date(2026, 6, 18), 5),),
        embargo=EmbargoSpec(1),
        approved_capital=ApprovedCapital(Decimal(100000), "INR"),
        universe_fingerprint="u" * 64,
        candidate_prefilter_artifact_fingerprint="u" * 64,
        instrument_dataset_fingerprints={STOCK_A: "a" * 64},
        nse_membership_evidence=NSEMembershipEvidenceIdentity(("ref",), True, "m" * 64, 20),
        tick_evidence=TickEvidenceIdentity("fixed-0.05", "source", True, "t" * 64),
        session_policy_identity=SessionPolicyIdentity(
            "policy", False, 15, "2026-03-01", "15:30:00"
        ),
        corporate_action_evidence=CorporateActionEvidenceIdentity("ca", True, (), "c" * 64),
        cost_model_identity=CostModelIdentity("model", "2026-06-05", {"b": "0.0003"}, ("ref",)),
        cost_evidence_identity=identity_1,
        cost_evidence_class=SCENARIO_LABEL,
        strategy_definitions=(StrategySpec("strat_1", "ORB", "basis", (), {"buf": "5"}),),
        parameter_grid=param_grid,
        friction_scenarios=(FrictionScenarioSpec("base", Decimal(1), Decimal("0.5")),),
        rejected_candidates=(),
        tournament_result={},
        walk_forward_result={},
        promotion_evidence=_promotion_evidence_with_scenario_eval(cost_scenario_1),
    )

    exp_2 = replace(
        exp_1,
        cost_evidence_identity=identity_2,
        promotion_evidence=_promotion_evidence_with_scenario_eval(cost_scenario_2),
    )

    # Cost scenario rate change changes experiment fingerprint and ID
    assert cost_scenario_1.fingerprint() != cost_scenario_2.fingerprint()
    assert exp_1.deterministic_fingerprint() != exp_2.deterministic_fingerprint()
    assert exp_1.experiment_id != exp_2.experiment_id

    # Parameter change changes experiment fingerprint
    exp_modified_param = replace(
        exp_1,
        strategy_definitions=(StrategySpec("strat_1", "ORB", "basis", (), {"buf": "10"}),),
    )
    assert exp_1.deterministic_fingerprint() != exp_modified_param.deterministic_fingerprint()

    # Live order call is strictly forbidden
    with pytest.raises(LiveOrderAttemptError, match="live orders are strictly forbidden"):
        replace(exp_1, live_orders_called=True)


def test_promotion_gate_fail_closed_rejection_proofs() -> None:
    """Prove promotion gate fail-closed rejections for:

    - scenario cannot promote
    - incomplete evidence cannot promote
    - insufficient WFO cannot promote
    - missing OOS cannot promote
    - missing paper evidence cannot promote
    - missing cost reconciliation cannot promote
    - live_orders_called=false verified
    """
    cost_scenario = _canonical_historical_cost_scenario()
    identity = CostEvidenceIdentity.from_historical_scenario(cost_scenario)
    orchestrator = ExperimentOrchestrator(code_commit_sha=CODE_COMMIT_SHA)

    base_exp = orchestrator.build_experiment(
        research_window=ResearchWindowConfig(start=date(2026, 6, 1), end=date(2026, 7, 6)),
        train_windows=(WindowSpec(1, date(2026, 6, 1), date(2026, 6, 10), 8),),
        validation_test_windows=(WindowSpec(1, date(2026, 6, 12), date(2026, 6, 18), 5),),
        embargo=EmbargoSpec(1),
        approved_capital=ApprovedCapital(Decimal(100000), "INR"),
        universe_fingerprint="u" * 64,
        candidate_prefilter_artifact_fingerprint="u" * 64,
        instrument_dataset_fingerprints={STOCK_A: "a" * 64},
        nse_membership_evidence=NSEMembershipEvidenceIdentity(("ref",), True, "m" * 64, 20),
        tick_evidence=TickEvidenceIdentity("fixed-0.05", "source", True, "t" * 64),
        session_policy_identity=SessionPolicyIdentity(
            "policy", False, 15, "2026-03-01", "15:30:00"
        ),
        corporate_action_evidence=CorporateActionEvidenceIdentity("ca", True, (), "c" * 64),
        cost_model_identity=CostModelIdentity("model", "2026-06-05", {"b": "0.0003"}, ("ref",)),
        cost_evidence_identity=identity,
        cost_evidence_class=SCENARIO_LABEL,
        strategy_definitions=(StrategySpec("strat_1", "ORB", "basis", (), {"buf": "5"}),),
        parameter_grid=_parameter_grid_dict(),
        friction_scenarios=(FrictionScenarioSpec("base", Decimal(1), Decimal("0.5")),),
        rejected_candidates=(),
        tournament_result={},
        walk_forward_result={},
        promotion_evidence=_promotion_evidence_with_scenario_eval(cost_scenario),
    )

    thresholds = PromotionThresholds(
        min_trades=20,  # Requiring 20 trades
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal("10.0"),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.05"),
    )

    # 1. Scenario cost evidence rejection
    passed, violations = base_exp.evaluate_promotion_gate(thresholds)
    assert passed is False
    assert any("scenario/incomplete evidence cannot promote" in v for v in violations)

    # 2. Insufficient WFO trade count rejection (15 < 20)
    assert any("is below required 20" in v for v in violations)

    # 3. Missing OOS rejection
    exp_no_oos = replace(
        base_exp,
        promotion_evidence=replace(base_exp.promotion_evidence, held_out_test=None),
    )
    passed_oos, violations_oos = exp_no_oos.evaluate_promotion_gate(thresholds)
    assert passed_oos is False
    assert any("held-out test evidence artifact is missing" in v for v in violations_oos)

    # 4. Missing paper trading rejection
    exp_no_paper = replace(
        base_exp,
        promotion_evidence=replace(base_exp.promotion_evidence, paper_trading=None),
    )
    passed_paper, violations_paper = exp_no_paper.evaluate_promotion_gate(thresholds)
    assert passed_paper is False
    assert any("paper-trading evidence artifact is missing" in v for v in violations_paper)

    # 5. Missing cost reconciliation rejection
    exp_no_recon = replace(
        base_exp,
        promotion_evidence=replace(base_exp.promotion_evidence, cost_reconciliation=None),
    )
    passed_recon, violations_recon = exp_no_recon.evaluate_promotion_gate(thresholds)
    assert passed_recon is False
    assert any("cost reconciliation evidence artifact is missing" in v for v in violations_recon)

    # 6. Incomplete evidence rejection (unpriced cost components)
    exp_unpriced = replace(
        base_exp,
        promotion_evidence=replace(
            base_exp.promotion_evidence,
            unpriced_cost_components=("stamp_duty",),
        ),
    )
    passed_unpriced, violations_unpriced = exp_unpriced.evaluate_promotion_gate(thresholds)
    assert passed_unpriced is False
    assert any("unpriced cost components remain: stamp_duty" in v for v in violations_unpriced)

    # 7. UNKNOWN != ZERO in cost evidence identity rejects
    identity_with_unknowns = replace(identity, unknown_components=("stt: unverified rate",))
    exp_unknowns = replace(base_exp, cost_evidence_identity=identity_with_unknowns)
    passed_unk, violations_unk = exp_unknowns.evaluate_promotion_gate(thresholds)
    assert passed_unk is False
    assert any(
        "cost evidence contains unknown components: stt: unverified rate" in v
        for v in violations_unk
    )

    # 8. live_orders_called=false verified across all experiments
    assert base_exp.live_orders_called is False
    assert exp_no_oos.live_orders_called is False
    assert exp_no_paper.live_orders_called is False
