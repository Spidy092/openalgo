"""Tests for the canonical immutable research experiment orchestration and provenance layer.

Covers:
- dataset fingerprint mismatch
- universe mismatch
- missing test evidence
- missing cost evidence
- changed strategy params changes identity
- changed capital changes identity
- train/test boundary changes identity
- simulator version changes identity
- created_at does not change identity
- future/test leakage rejected
- no live-order capability
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from equity_engine.cost_ledger import (
    INCOMPLETE_LABEL,
    SCENARIO_LABEL,
    EffectiveDatedCostLedger,
    LedgerComponent,
    LedgerProduct,
    UnsupportedResearchDate,
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
    CurrentCalibrationReference,
    DataLeakageError,
    DatasetFingerprintMismatchError,
    EmbargoSpec,
    EventDrivenSimulationEvidence,
    ExperimentArtifact,
    ExperimentOrchestrator,
    FrictionScenarioSpec,
    HeldOutTestEvidence,
    LiveOrderAttemptError,
    MissingEvidenceError,
    NSEMembershipEvidenceIdentity,
    PaperTradingEvidence,
    RejectedCandidateSpec,
    ResearchWindowConfig,
    SessionPolicyIdentity,
    SlippageStressEvidence,
    StrategySpec,
    TickEvidenceIdentity,
    UniverseFingerprintMismatchError,
    WindowSpec,
    compute_prefilter_artifact_fingerprint,
)
from equity_engine.gates import DrawdownBasis, PromotionThresholds
from equity_engine.historical_cost_scenario import ScenarioAssumption, compile_historical_scenario
from equity_engine.provenance import MarketDataManifest, dataframe_fingerprint


def _make_frame(dates: list[str]) -> pd.DataFrame:
    timestamps: list[pd.Timestamp] = []
    for d in dates:
        for hhmm in ("09:15", "11:00", "15:25"):
            timestamps.append(pd.Timestamp(f"{d} {hhmm}", tz="Asia/Kolkata"))
    return pd.DataFrame(
        {
            "open": [100.0] * len(timestamps),
            "high": [102.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [101.0] * len(timestamps),
            "volume": [50000] * len(timestamps),
        },
        index=pd.DatetimeIndex(timestamps),
    )


def _make_manifest(instrument_token: str, symbol: str) -> MarketDataManifest:
    return MarketDataManifest(
        provider="synthetic-test",
        exchange="NSE",
        instrument_token=instrument_token,
        symbol=symbol,
        timezone="Asia/Kolkata",
        interval="5m",
        timestamp_semantics="start-of-candle",
        start=datetime(2026, 1, 1, 9, 15, tzinfo=UTC),
        end=datetime(2026, 6, 30, 15, 30, tzinfo=UTC),
        retrieved_at=datetime(2026, 7, 1, 0, 0, tzinfo=UTC),
        adjustment_policy="split-unadjusted",
        universe_rule_version="nse-cm-2026-v1",
        source_reference="test-dataset-ref",
    )


def _sample_prefilter_artifact() -> dict[str, object]:
    return {
        "schema_version": "research-universe-prefilter-v1",
        "selection_cutoff": "2026-03-31",
        "selection_as_of": "2026-03-31T15:30:00+05:30",
        "approved_capital_rupees": "100000",
        "reference_price_policy": "latest continuous-session close",
        "candidates": [
            {
                "instrument_key": "NSE_EQ|INE002A01018",
                "eligible": True,
                "reason_codes": [],
                "dataset_fingerprint": "mock_fingerprint_reliance",
            }
        ],
        "live_orders_called": False,
    }


_OMIT = object()


def _sample_promotion_evidence(
    *,
    held_out_test: Any = _OMIT,
    cost_reconciliation: Any = _OMIT,
    paper_trading: Any = _OMIT,
    baseline_comparison: Any = _OMIT,
    slippage_stress: Any = _OMIT,
    event_simulation: Any = _OMIT,
) -> ConcretePromotionEvidence:
    return ConcretePromotionEvidence(
        held_out_test=(
            HeldOutTestEvidence(
                artifact_fingerprint="sha256_test_eval_artifact_001",
                test_dataset_fingerprints=(("NSE_EQ|INE002A01018", "fp_test_rel"),),
                window_id=1,
                trade_count=120,
                profit_factor=Decimal("1.45"),
                max_drawdown_pct=Decimal("6.50"),
                drawdown_basis=DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS,
                net_return_pct=Decimal("14.20"),
                source_reference="test_window_eval_log_001",
            )
            if held_out_test is _OMIT
            else held_out_test
        ),
        cost_reconciliation=(
            CostReconciliationEvidence(
                artifact_fingerprint="sha256_cost_recon_artifact_001",
                schema_version="upstox-cost-reconciliation/v1",
                cost_model_name="documented",
                orders_checked=10,
                passed_count=10,
                failed_count=0,
                max_reconciliation_error_inr=Decimal("0.005"),
                tolerance_inr=Decimal("0.01"),
                status="PASS",
            )
            if cost_reconciliation is _OMIT
            else cost_reconciliation
        ),
        paper_trading=(
            PaperTradingEvidence(
                artifact_fingerprint="sha256_paper_trading_artifact_001",
                environment="upstox_sandbox_v2",
                session_start=date(2026, 7, 1),
                session_end=date(2026, 7, 31),
                verified_orders_count=45,
                audit_log_fingerprint="sha256_paper_audit_log_001",
                source_reference="broker_sandbox_order_log",
            )
            if paper_trading is _OMIT
            else paper_trading
        ),
        baseline_comparison=(
            BaselineComparisonEvidence(
                artifact_fingerprint="sha256_baseline_artifact_001",
                baseline_candidate_id="baseline:first-bar-hold",
                evaluated_candidate_id="orb:15m:vol1.5:buf5bps",
                baseline_net_return_pct=Decimal("2.10"),
                evaluated_net_return_pct=Decimal("14.20"),
                outperformed=True,
            )
            if baseline_comparison is _OMIT
            else baseline_comparison
        ),
        slippage_stress=(
            SlippageStressEvidence(
                artifact_fingerprint="sha256_slippage_stress_artifact_001",
                scenarios_evaluated=("base_2bps", "stress_5bps", "severe_10bps"),
                stress_max_drawdown_pct=Decimal("8.20"),
                stress_passed=True,
            )
            if slippage_stress is _OMIT
            else slippage_stress
        ),
        event_simulation=(
            EventDrivenSimulationEvidence(
                artifact_fingerprint="sha256_event_sim_artifact_001",
                simulator_version="openalgo-event-simulator-v1",
                trade_count=120,
                initial_cash=Decimal(100000),
                final_cash=Decimal(114200),
            )
            if event_simulation is _OMIT
            else event_simulation
        ),
        unpriced_cost_components=(),
    )


@pytest.fixture
def baseline_experiment() -> ExperimentArtifact:
    prefilter = _sample_prefilter_artifact()
    prefilter_fp = compute_prefilter_artifact_fingerprint(prefilter)

    orchestrator = ExperimentOrchestrator(
        code_commit_sha="c4322d43956de1b43a764a7849b7437b38a1c932",
        vectorbt_version="1.1.0",
        simulator_version="openalgo-event-simulator-v1",
    )
    cost_evidence_identity = CostEvidenceIdentity.from_ledger(
        EffectiveDatedCostLedger(),
        on_date=date(2026, 6, 30),
        product=LedgerProduct.INTRADAY,
    )

    return orchestrator.build_experiment(
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        train_windows=(
            WindowSpec(window_id=1, start=date(2026, 1, 1), end=date(2026, 3, 31), trading_days=60),
        ),
        validation_test_windows=(
            WindowSpec(window_id=1, start=date(2026, 4, 5), end=date(2026, 6, 30), trading_days=60),
        ),
        embargo=EmbargoSpec(trading_days=2),
        approved_capital=ApprovedCapital(amount_rupees=Decimal(100000)),
        universe_fingerprint=prefilter_fp,
        candidate_prefilter_artifact_fingerprint=prefilter_fp,
        instrument_dataset_fingerprints={"NSE_EQ|INE002A01018": "dataset_hash_reliance_123"},
        nse_membership_evidence=NSEMembershipEvidenceIdentity(
            source_refs=("https://nsearchives.nseindia.com/content/historical_members.csv",),
            complete=True,
            coverage_fingerprint="sha256_nse_membership_coverage_001",
            eligible_dates_count=120,
        ),
        tick_evidence=TickEvidenceIdentity(
            policy_name="tiered_tick",
            source="https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf",
            coverage_complete=True,
            coverage_fingerprint="sha256_tick_coverage_001",
        ),
        session_policy_identity=SessionPolicyIdentity(
            policy_name="NSEEquitySessionPolicy",
            cas_eligible=True,
            exit_buffer_minutes=15,
            cas_effective_date="2026-03-01",
            continuous_end="15:30:00",
        ),
        corporate_action_evidence=CorporateActionEvidenceIdentity(
            source="https://upstox.com/developer/api-documentation/get-corporate-actions/",
            complete=True,
            blocking_events=(),
            evidence_fingerprint="sha256_ca_evidence_001",
        ),
        cost_model_identity=CostModelIdentity(
            model_name="documented",
            effective_date="2026-03-01",
            rates={"brokerage": "0.001", "brokerage_cap": "20", "gst": "0.18"},
            source_refs=("https://upstox.com/brokerage-charges/",),
        ),
        cost_evidence_identity=cost_evidence_identity,
        cost_evidence_class=INCOMPLETE_LABEL,
        strategy_definitions=(
            StrategySpec(
                candidate_id="orb:15m:vol1.5:buf5bps",
                strategy_name="opening_range_breakout",
                research_basis="NSE ORB empirical study SSRN-5198458",
                source_refs=("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5198458",),
                parameters={"range_minutes": "15", "volume_ratio": "1.5", "buffer_bps": "5"},
            ),
        ),
        parameter_grid={"range_minutes": ("15",), "volume_ratio": ("1.5",), "buffer_bps": ("5",)},
        friction_scenarios=(
            FrictionScenarioSpec(
                scenario_id="base",
                slippage_bps_per_leg=Decimal("2.0"),
                half_spread_bps_per_leg=Decimal("1.0"),
            ),
        ),
        rejected_candidates=(
            RejectedCandidateSpec(
                candidate_id="orb:30m:vol1.2:buf5bps",
                instrument_key="NSE_EQ|INE002A01018",
                stage="tournament_trade_count",
                reasons=("trade_count 12 is below explicit minimum 30",),
            ),
        ),
        tournament_result={"winner_candidate_id": "orb:15m:vol1.5:buf5bps", "trade_count": 85},
        walk_forward_result={"window_1_winner": "orb:15m:vol1.5:buf5bps", "test_trades": 35},
        promotion_evidence=_sample_promotion_evidence(),
        random_seeds={"simulator": 42},
        created_at="2026-09-09T16:00:00Z",
    )


# Schema version verification


def test_experiment_schema_version(baseline_experiment: ExperimentArtifact) -> None:
    assert baseline_experiment.schema_version == EXPERIMENT_SCHEMA_VERSION


# 1. Dataset fingerprint mismatch fails closed


def test_dataset_fingerprint_mismatch_fails_closed(baseline_experiment: ExperimentArtifact) -> None:
    frame = _make_frame(["2026-01-02", "2026-01-05"])
    manifest = _make_manifest("NSE_EQ|INE002A01018", "RELIANCE")
    actual_fp = dataframe_fingerprint(frame, manifest)

    # Valid check passes
    valid_exp = replace(
        baseline_experiment,
        instrument_dataset_fingerprints={"NSE_EQ|INE002A01018": actual_fp},
    )
    valid_exp.validate_integrity(
        dataset_frames={"NSE_EQ|INE002A01018": frame},
        dataset_manifests={"NSE_EQ|INE002A01018": manifest},
    )

    # Mismatched fingerprint raises DatasetFingerprintMismatchError
    mismatched_exp = replace(
        baseline_experiment,
        instrument_dataset_fingerprints={"NSE_EQ|INE002A01018": "tampered_or_stale_fingerprint"},
    )
    with pytest.raises(DatasetFingerprintMismatchError, match="dataset fingerprint mismatch"):
        mismatched_exp.validate_integrity(
            dataset_frames={"NSE_EQ|INE002A01018": frame},
            dataset_manifests={"NSE_EQ|INE002A01018": manifest},
        )


# 2. Universe mismatch fails closed


def test_universe_mismatch_fails_closed(baseline_experiment: ExperimentArtifact) -> None:
    prefilter = _sample_prefilter_artifact()
    actual_fp = compute_prefilter_artifact_fingerprint(prefilter)

    # Correct universe fingerprint matches prefilter artifact
    valid_exp = replace(
        baseline_experiment,
        universe_fingerprint=actual_fp,
        candidate_prefilter_artifact_fingerprint=actual_fp,
    )
    valid_exp.validate_integrity(prefilter_artifact=prefilter)

    # Mismatched universe prefilter fingerprint fails
    mismatched_prefilter_exp = replace(
        baseline_experiment,
        universe_fingerprint=actual_fp,
        candidate_prefilter_artifact_fingerprint="tampered_prefilter_hash_999",
    )
    with pytest.raises(UniverseFingerprintMismatchError, match="prefilter fingerprint mismatch"):
        mismatched_prefilter_exp.validate_integrity(prefilter_artifact=prefilter)

    # Internal inconsistency between universe_fingerprint and candidate_prefilter fails
    inconsistent_universe_exp = replace(
        baseline_experiment,
        universe_fingerprint="universe_hash_a",
        candidate_prefilter_artifact_fingerprint=actual_fp,
    )
    with pytest.raises(UniverseFingerprintMismatchError, match="universe fingerprint"):
        inconsistent_universe_exp.validate_integrity(prefilter_artifact=prefilter)


# 3. Missing test evidence fails closed (arbitrary booleans rejected)


def test_missing_test_evidence_fails_closed(baseline_experiment: ExperimentArtifact) -> None:
    # A boolean flag is not accepted; missing held_out_test artifact must fail
    missing_test_evidence = _sample_promotion_evidence(held_out_test=None)
    exp = replace(baseline_experiment, promotion_evidence=missing_test_evidence)

    with pytest.raises(MissingEvidenceError, match="held-out test evidence artifact is missing"):
        exp.validate_integrity()


# 4. Missing cost evidence fails closed


def test_missing_cost_evidence_fails_closed(baseline_experiment: ExperimentArtifact) -> None:
    missing_cost_evidence = _sample_promotion_evidence(cost_reconciliation=None)
    exp = replace(baseline_experiment, promotion_evidence=missing_cost_evidence)

    with pytest.raises(MissingEvidenceError, match="broker cost reconciliation evidence"):
        exp.validate_integrity()


# 5. Changed strategy params changes identity


def test_changed_strategy_params_changes_identity(
    baseline_experiment: ExperimentArtifact,
) -> None:
    initial_fp = baseline_experiment.deterministic_fingerprint()
    initial_id = baseline_experiment.experiment_id

    # Modify a strategy parameter (e.g. buffer_bps from 5 to 10)
    modified_strategy = StrategySpec(
        candidate_id="orb:15m:vol1.5:buf10bps",
        strategy_name="opening_range_breakout",
        research_basis="NSE ORB empirical study SSRN-5198458",
        source_refs=("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5198458",),
        parameters={"range_minutes": "15", "volume_ratio": "1.5", "buffer_bps": "10"},
    )
    changed_exp = replace(baseline_experiment, strategy_definitions=(modified_strategy,))

    assert changed_exp.deterministic_fingerprint() != initial_fp
    assert changed_exp.experiment_id != initial_id


# 6. Changed capital changes identity


def test_changed_capital_changes_identity(baseline_experiment: ExperimentArtifact) -> None:
    initial_fp = baseline_experiment.deterministic_fingerprint()
    initial_id = baseline_experiment.experiment_id

    changed_capital_exp = replace(
        baseline_experiment,
        approved_capital=ApprovedCapital(amount_rupees=Decimal(250000)),
    )

    assert changed_capital_exp.deterministic_fingerprint() != initial_fp
    assert changed_capital_exp.experiment_id != initial_id


# 7. Train/Test boundary changes identity


def test_train_test_boundary_changes_identity(baseline_experiment: ExperimentArtifact) -> None:
    initial_fp = baseline_experiment.deterministic_fingerprint()
    initial_id = baseline_experiment.experiment_id

    # Change train window end from 2026-03-31 to 2026-03-15
    changed_train = (
        WindowSpec(window_id=1, start=date(2026, 1, 1), end=date(2026, 3, 15), trading_days=50),
    )
    changed_boundary_exp = replace(baseline_experiment, train_windows=changed_train)

    assert changed_boundary_exp.deterministic_fingerprint() != initial_fp
    assert changed_boundary_exp.experiment_id != initial_id


# 8. Simulator version changes identity


def test_simulator_version_changes_identity(baseline_experiment: ExperimentArtifact) -> None:
    initial_fp = baseline_experiment.deterministic_fingerprint()
    initial_id = baseline_experiment.experiment_id

    changed_sim_exp = replace(
        baseline_experiment,
        simulator_version="openalgo-event-simulator-v2-experimental",
    )

    assert changed_sim_exp.deterministic_fingerprint() != initial_fp
    assert changed_sim_exp.experiment_id != initial_id


# 9. Created_at does not change identity


def test_created_at_does_not_change_identity(baseline_experiment: ExperimentArtifact) -> None:
    initial_fp = baseline_experiment.deterministic_fingerprint()
    initial_id = baseline_experiment.experiment_id

    # Change created_at timestamp to different times
    time_variant_1 = replace(baseline_experiment, created_at="2026-01-01T00:00:00Z")
    time_variant_2 = replace(baseline_experiment, created_at="2026-09-09T16:55:00+05:30")

    assert time_variant_1.deterministic_fingerprint() == initial_fp
    assert time_variant_2.deterministic_fingerprint() == initial_fp
    assert time_variant_1.experiment_id == initial_id
    assert time_variant_2.experiment_id == initial_id


# 10. Future/test leakage rejected


def test_future_test_leakage_rejected(baseline_experiment: ExperimentArtifact) -> None:
    # Attempting to overlap train window end with test window start must be rejected
    leaky_train = (
        WindowSpec(window_id=1, start=date(2026, 1, 1), end=date(2026, 4, 10), trading_days=70),
    )
    test_windows = (
        WindowSpec(window_id=1, start=date(2026, 4, 5), end=date(2026, 6, 30), trading_days=60),
    )

    orchestrator = ExperimentOrchestrator(
        code_commit_sha="c4322d43956de1b43a764a7849b7437b38a1c932",
    )

    with pytest.raises(DataLeakageError, match="overlaps or touches test start"):
        orchestrator.build_experiment(
            research_window=baseline_experiment.research_window,
            train_windows=leaky_train,
            validation_test_windows=test_windows,
            embargo=baseline_experiment.embargo,
            approved_capital=baseline_experiment.approved_capital,
            universe_fingerprint=baseline_experiment.universe_fingerprint,
            candidate_prefilter_artifact_fingerprint=(
                baseline_experiment.candidate_prefilter_artifact_fingerprint
            ),
            instrument_dataset_fingerprints=baseline_experiment.instrument_dataset_fingerprints,
            nse_membership_evidence=baseline_experiment.nse_membership_evidence,
            tick_evidence=baseline_experiment.tick_evidence,
            session_policy_identity=baseline_experiment.session_policy_identity,
            corporate_action_evidence=baseline_experiment.corporate_action_evidence,
            cost_model_identity=baseline_experiment.cost_model_identity,
            cost_evidence_identity=baseline_experiment.cost_evidence_identity,
            cost_evidence_class=baseline_experiment.cost_evidence_class,
            strategy_definitions=baseline_experiment.strategy_definitions,
            parameter_grid=baseline_experiment.parameter_grid,
            friction_scenarios=baseline_experiment.friction_scenarios,
            rejected_candidates=baseline_experiment.rejected_candidates,
            tournament_result=baseline_experiment.tournament_result,
            walk_forward_result=baseline_experiment.walk_forward_result,
            promotion_evidence=baseline_experiment.promotion_evidence,
        )


# 11. No live-order capability


def test_no_live_order_capability(baseline_experiment: ExperimentArtifact) -> None:
    # Experiment artifact must declare live_orders_called as False
    assert baseline_experiment.live_orders_called is False

    # Attempting to construct or load an artifact with live_orders_called=True must fail
    with pytest.raises(LiveOrderAttemptError, match="live orders are strictly forbidden"):
        replace(baseline_experiment, live_orders_called=True)

    # Orchestrator property confirms no live order capability
    orchestrator = ExperimentOrchestrator(
        code_commit_sha="c4322d43956de1b43a764a7849b7437b38a1c932"
    )
    assert orchestrator.live_orders_called is False


# 12. Evaluation of concrete promotion thresholds


def test_promotion_gate_threshold_evaluation(baseline_experiment: ExperimentArtifact) -> None:
    thresholds = PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.20"),
        max_drawdown_pct=Decimal("10.0"),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )

    # Strong numerical evidence cannot promote an experiment whose ledger is incomplete.
    with pytest.raises(MissingEvidenceError, match="not verified HISTORICAL_ACTUAL_COSTS"):
        baseline_experiment.validate_integrity(promotion_thresholds=thresholds)

    # High drawdown fails gate
    bad_drawdown = replace(
        baseline_experiment.promotion_evidence.held_out_test,
        max_drawdown_pct=Decimal("15.50"),
    )
    failing_drawdown_evidence = replace(
        baseline_experiment.promotion_evidence, held_out_test=bad_drawdown
    )
    failing_exp = replace(baseline_experiment, promotion_evidence=failing_drawdown_evidence)

    with pytest.raises(MissingEvidenceError, match="max drawdown 15.50% exceeds allowed 10.0%"):
        failing_exp.validate_integrity(promotion_thresholds=thresholds)


def test_historical_actual_identity_cannot_be_constructed_from_a_digest() -> None:
    fields = {
        "ledger_schema_version": "effective-dated-cost-ledger/v1",
        "ledger_fingerprint": "a" * 64,
        "evidence_classification": "HISTORICAL_ACTUAL_COSTS",
        "historical_actual": True,
        "product_scope": LedgerProduct.INTRADAY.value,
        "evidence_mode": "historical_resolution",
        "policy_identity": "test-policy",
        "resolved_on_date": date(2026, 6, 30),
        "selected_record_ids": (),
        "unknown_components": (),
    }
    with pytest.raises(ValueError, match="concrete ledger"):
        CostEvidenceIdentity(**fields)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        CostEvidenceIdentity(**fields, _verified_ledger_fingerprint="a" * 64)


def test_cost_identity_uses_real_product_aware_ledger_assessment() -> None:
    ledger = EffectiveDatedCostLedger()
    identity = CostEvidenceIdentity.from_ledger(
        ledger,
        on_date=date(2026, 6, 30),
        product=LedgerProduct.DELIVERY,
    )
    assessment = ledger.describe(date(2026, 6, 30), LedgerProduct.DELIVERY)
    assert identity.ledger_fingerprint == ledger.fingerprint()
    assert identity.product_scope == LedgerProduct.DELIVERY.value
    assert identity.historical_actual is assessment.historical_actual is False
    assert identity.unknown_components == tuple(sorted(assessment.unknowns))

    with pytest.raises(TypeError, match="EffectiveDatedCostLedger"):
        CostEvidenceIdentity.from_ledger(
            object(), on_date=date(2026, 6, 30), product=LedgerProduct.INTRADAY
        )
    with pytest.raises(TypeError, match="LedgerProduct"):
        CostEvidenceIdentity.from_ledger(
            ledger,
            on_date=date(2026, 6, 30),
            product="INTRADAY",  # type: ignore[arg-type]
        )
    with pytest.raises(UnsupportedResearchDate):
        CostEvidenceIdentity.from_ledger(
            ledger, on_date=date(2024, 6, 30), product=LedgerProduct.INTRADAY
        )


def test_public_scenario_binds_ledger_fingerprint_and_cannot_promote() -> None:
    ledger = EffectiveDatedCostLedger()
    identity = CostEvidenceIdentity.from_public_scenario(
        ledger,
        on_date=date(2026, 6, 30),
        product=LedgerProduct.INTRADAY,
        scenario_identity="documented-public-terms",
    )
    assert identity.ledger_fingerprint == ledger.fingerprint()
    assert identity.evidence_classification == SCENARIO_LABEL
    assert identity.historical_actual is False
    assert identity.scenario_identity == "documented-public-terms"


def _canonical_historical_scenario(
    *,
    product: LedgerProduct = LedgerProduct.INTRADAY,
    brokerage_rate: str = "0.0006",
):
    return compile_historical_scenario(
        scenario_id="integration-scenario",
        ledger=EffectiveDatedCostLedger(),
        scenario_date=date(2026, 6, 30),
        research_start=date(2026, 1, 1),
        research_end=date(2026, 6, 30),
        product=product,
        assumptions=(
            ScenarioAssumption(
                assumption_id="integration-brokerage",
                component=LedgerComponent.BROKERAGE,
                product=product,
                basis="integration-test",
                rate=Decimal(brokerage_rate),
                formula="turnover-rate",
                source="integration-test",
                reason="scenario-only assumption",
            ),
            ScenarioAssumption(
                assumption_id="integration-gst",
                component=LedgerComponent.GST,
                product=product,
                basis="integration-test",
                rate=Decimal("0.18"),
                formula="gst-on-known-charges",
                source="integration-test",
                reason="scenario-only assumption",
            ),
            ScenarioAssumption(
                assumption_id="integration-clearing",
                component=LedgerComponent.CLEARING,
                product=product,
                basis="integration-test",
                rate=Decimal("0.000001"),
                formula="turnover-rate",
                source="integration-test",
                reason="scenario-only assumption",
            ),
        ),
    )


def test_canonical_historical_scenario_binds_experiment_v3_identity(
    baseline_experiment: ExperimentArtifact,
) -> None:
    scenario = _canonical_historical_scenario()
    identity = CostEvidenceIdentity.from_historical_scenario(
        scenario,
        expected_product=LedgerProduct.INTRADAY,
    )
    scenario_experiment = replace(
        baseline_experiment,
        cost_evidence_identity=identity,
        cost_evidence_class=SCENARIO_LABEL,
    )

    assert identity.scenario_identity == scenario.fingerprint()
    assert identity.ledger_fingerprint == scenario.ledger_fingerprint
    assert identity.product_scope == scenario.product.value
    assert identity.evidence_classification == SCENARIO_LABEL
    assert identity.historical_actual is False
    scenario_experiment.validate_integrity()

    thresholds = PromotionThresholds(
        min_trades=10,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal(10),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal(1),
    )
    passed, violations = scenario_experiment.evaluate_promotion_gate(thresholds)
    assert passed is False
    assert any("scenario/incomplete evidence cannot promote" in item for item in violations)


def test_scenario_assumption_changes_bind_new_experiment_identity(
    baseline_experiment: ExperimentArtifact,
) -> None:
    first = _canonical_historical_scenario(brokerage_rate="0.0006")
    second = _canonical_historical_scenario(brokerage_rate="0.0007")
    first_identity = CostEvidenceIdentity.from_historical_scenario(first)
    second_identity = CostEvidenceIdentity.from_historical_scenario(second)

    first_experiment = replace(
        baseline_experiment,
        cost_evidence_identity=first_identity,
        cost_evidence_class=SCENARIO_LABEL,
    )
    second_experiment = replace(
        baseline_experiment,
        cost_evidence_identity=second_identity,
        cost_evidence_class=SCENARIO_LABEL,
    )

    assert first.fingerprint() != second.fingerprint()
    assert (
        first_experiment.deterministic_fingerprint()
        != second_experiment.deterministic_fingerprint()
    )
    assert first_identity.ledger_fingerprint == second_identity.ledger_fingerprint


def test_scenario_product_mismatch_fails_closed() -> None:
    scenario = _canonical_historical_scenario(product=LedgerProduct.INTRADAY)

    with pytest.raises(ValueError, match="does not match expected experiment product"):
        CostEvidenceIdentity.from_historical_scenario(
            scenario,
            expected_product=LedgerProduct.DELIVERY,
        )


def test_current_calibration_cannot_collide_with_canonical_scenario_evidence(
    baseline_experiment: ExperimentArtifact,
) -> None:
    scenario = _canonical_historical_scenario()
    identity = CostEvidenceIdentity.from_historical_scenario(scenario)
    scenario_experiment = replace(
        baseline_experiment,
        cost_evidence_identity=identity,
        cost_evidence_class=SCENARIO_LABEL,
    )

    with pytest.raises(DataLeakageError, match="historical PIT or cost evidence"):
        replace(
            scenario_experiment,
            current_calibration_reference=CurrentCalibrationReference(
                snapshot_fingerprint=identity.ledger_fingerprint,
                snapshot_as_of="2026-09-10T09:15:00Z",
            ),
        )


def test_current_calibration_is_not_historical_eligibility(
    baseline_experiment: ExperimentArtifact,
) -> None:
    calibration = CurrentCalibrationReference(
        snapshot_fingerprint="c" * 64,
        snapshot_as_of="2026-09-10T09:15:00Z",
    )
    with_calibration = replace(baseline_experiment, current_calibration_reference=calibration)
    with_calibration.validate_integrity()
    assert with_calibration.current_calibration_reference is not None
    assert (
        with_calibration.current_calibration_reference.current_snapshot_not_historical_eligibility
        is True
    )
    assert (
        with_calibration.deterministic_fingerprint()
        != baseline_experiment.deterministic_fingerprint()
    )

    with pytest.raises(ValueError, match="current_snapshot_not_historical_eligibility=True"):
        CurrentCalibrationReference(
            snapshot_fingerprint="c" * 64,
            snapshot_as_of="2026-09-10T09:15:00Z",
            current_snapshot_not_historical_eligibility=False,
        )

    with pytest.raises(DataLeakageError, match="historical PIT"):
        replace(
            baseline_experiment,
            current_calibration_reference=CurrentCalibrationReference(
                snapshot_fingerprint=baseline_experiment.universe_fingerprint,
                snapshot_as_of="2026-09-10T09:15:00Z",
            ),
        )


def test_promotion_rejects_scenario_cost_evidence(
    baseline_experiment: ExperimentArtifact,
) -> None:
    ledger = EffectiveDatedCostLedger()
    scenario_identity = CostEvidenceIdentity.from_public_scenario(
        ledger,
        on_date=date(2026, 6, 30),
        product=LedgerProduct.INTRADAY,
        scenario_identity="public-broker-model",
    )
    scenario_exp = replace(
        baseline_experiment,
        cost_evidence_identity=scenario_identity,
        cost_evidence_class=SCENARIO_LABEL,
    )
    # Research integrity passes with scenario evidence
    scenario_exp.validate_integrity()

    # Promotion gate strictly rejects scenario evidence
    thresholds = PromotionThresholds(
        min_trades=10,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal("10.0"),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("1.0"),
    )
    passed, violations = scenario_exp.evaluate_promotion_gate(thresholds)
    assert passed is False
    assert any("scenario/incomplete evidence cannot promote" in v for v in violations)

    with pytest.raises(MissingEvidenceError, match="scenario/incomplete evidence cannot promote"):
        scenario_exp.validate_integrity(promotion_thresholds=thresholds)


def test_promotion_rejects_incomplete_cost_evidence_with_unknowns(
    baseline_experiment: ExperimentArtifact,
) -> None:
    ledger = EffectiveDatedCostLedger()
    delivery_identity = CostEvidenceIdentity.from_ledger(
        ledger,
        on_date=date(2026, 6, 30),
        product=LedgerProduct.DELIVERY,
    )
    assert delivery_identity.unknown_components
    delivery_exp = replace(
        baseline_experiment,
        cost_evidence_identity=delivery_identity,
        cost_evidence_class=delivery_identity.evidence_classification,
    )
    # Research integrity passes
    delivery_exp.validate_integrity()

    # Promotion gate rejects unknown components
    thresholds = PromotionThresholds(
        min_trades=10,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal("10.0"),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("1.0"),
    )
    passed, violations = delivery_exp.evaluate_promotion_gate(thresholds)
    assert passed is False
    assert any("contains unknown components" in v for v in violations)

    with pytest.raises(MissingEvidenceError, match="contains unknown components"):
        delivery_exp.validate_integrity(promotion_thresholds=thresholds)


def test_current_calibration_cannot_substitute_for_historical_cost_evidence(
    baseline_experiment: ExperimentArtifact,
) -> None:
    calib = CurrentCalibrationReference(
        snapshot_fingerprint=baseline_experiment.cost_evidence_identity.ledger_fingerprint,
        snapshot_as_of="2026-09-10T09:15:00Z",
    )
    with pytest.raises(DataLeakageError, match="historical PIT or cost evidence"):
        replace(baseline_experiment, current_calibration_reference=calib)
