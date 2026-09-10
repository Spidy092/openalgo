from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from equity_engine.corporate_actions import (
    CoverageScope,
    PointInTimeCorporateActionLedger,
)
from equity_engine.cost_ledger import (
    ACCOUNT_SNAPSHOT_DATE,
    GST_RATE,
    HISTORICAL_ACTUAL_LABEL,
    INCOMPLETE_LABEL,
    SCENARIO_LABEL,
    EffectiveDatedCostLedger,
    EvidenceClass,
    LedgerComponent,
    LedgerProduct,
    LedgerSide,
    UnknownCostEvidence,
    UnsupportedResearchDate,
)
from equity_engine.experiment import (
    ApprovedCapital,
    BaselineComparisonEvidence,
    ConcretePromotionEvidence,
    CorporateActionEvidenceIdentity,
    CostEvidenceIdentity,
    CostModelIdentity,
    CostReconciliationEvidence,
    DataLeakageError,
    EmbargoSpec,
    EventDrivenSimulationEvidence,
    ExperimentArtifact,
    ExperimentOrchestrator,
    FrictionScenarioSpec,
    HeldOutTestEvidence,
    MissingEvidenceError,
    NSEMembershipEvidenceIdentity,
    PaperTradingEvidence,
    RejectedCandidateSpec,
    ResearchWindowConfig,
    SessionPolicyIdentity,
    SlippageStressEvidence,
    StrategySpec,
    TickEvidenceIdentity,
    WindowSpec,
)
from equity_engine.gates import DrawdownBasis, PromotionThresholds


def _cost_identity(
    *,
    on_date: date = date(2026, 9, 8),
    product: LedgerProduct = LedgerProduct.INTRADAY,
) -> CostEvidenceIdentity:
    return CostEvidenceIdentity.from_ledger(
        EffectiveDatedCostLedger(),
        on_date=on_date,
        product=product,
    )


def _sample_ca_ledger() -> PointInTimeCorporateActionLedger:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 9, 8),
            source="CANONICAL_PIT_CORPORATE_ACTION_LEDGER",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    return ledger


def _experiment(
    *,
    cost_evidence_identity: CostEvidenceIdentity | None = None,
    created_at: str = "2026-09-10T10:00:00+05:30",
    ca_ledger: PointInTimeCorporateActionLedger | None = None,
) -> ExperimentArtifact:
    identity = cost_evidence_identity or _cost_identity()
    orchestrator = ExperimentOrchestrator(code_commit_sha="baa4e10aa")
    effective_ca_ledger = ca_ledger if ca_ledger is not None else _sample_ca_ledger()
    ca_evidence = CorporateActionEvidenceIdentity.from_ledger(
        effective_ca_ledger,
        research_window=ResearchWindowConfig(
            start=date(2026, 1, 1),
            end=date(2026, 9, 8),
        ),
        instruments=("NSE_EQ|INE002A01018",),
    )
    return orchestrator.build_experiment(
        research_window=ResearchWindowConfig(
            start=date(2026, 1, 1),
            end=date(2026, 9, 8),
        ),
        train_windows=(
            WindowSpec(
                window_id=1,
                start=date(2026, 1, 1),
                end=date(2026, 6, 30),
                trading_days=120,
            ),
        ),
        validation_test_windows=(
            WindowSpec(
                window_id=1,
                start=date(2026, 7, 1),
                end=date(2026, 9, 8),
                trading_days=45,
            ),
        ),
        embargo=EmbargoSpec(trading_days=2),
        approved_capital=ApprovedCapital(amount_rupees=Decimal(1000)),
        universe_fingerprint="universe-fingerprint",
        candidate_prefilter_artifact_fingerprint="prefilter-fingerprint",
        instrument_dataset_fingerprints={"NSE_EQ|INE002A01018": "dataset-fingerprint"},
        nse_membership_evidence=NSEMembershipEvidenceIdentity(
            source_refs=("nse-membership-source",),
            complete=True,
            coverage_fingerprint="membership-fingerprint",
            eligible_dates_count=45,
        ),
        tick_evidence=TickEvidenceIdentity(
            policy_name="tiered_tick",
            source="nse-tick-source",
            coverage_complete=True,
            coverage_fingerprint="tick-fingerprint",
        ),
        session_policy_identity=SessionPolicyIdentity(
            policy_name="NSEEquitySessionPolicy",
            cas_eligible=True,
            exit_buffer_minutes=15,
            cas_effective_date="2026-03-01",
            continuous_end="15:30:00",
        ),
        corporate_action_evidence=ca_evidence,
        cost_model_identity=CostModelIdentity(
            model_name="documented",
            effective_date="2026-03-01",
            rates={"brokerage": "0.001", "gst": "0.18"},
            source_refs=("public-pricing-source",),
        ),
        cost_evidence_identity=identity,
        cost_evidence_class=identity.evidence_classification,
        strategy_definitions=(
            StrategySpec(
                candidate_id="orb:15m",
                strategy_name="opening_range_breakout",
                research_basis="research basis",
                source_refs=("strategy-source",),
                parameters={"range_minutes": "15"},
            ),
        ),
        parameter_grid={"range_minutes": ("15",)},
        friction_scenarios=(
            FrictionScenarioSpec(
                scenario_id="base",
                slippage_bps_per_leg=Decimal(2),
                half_spread_bps_per_leg=Decimal(1),
            ),
        ),
        rejected_candidates=(
            RejectedCandidateSpec(
                candidate_id="orb:30m",
                instrument_key="NSE_EQ|INE002A01018",
                stage="screening",
                reasons=("insufficient trades",),
            ),
        ),
        tournament_result={"winner": "orb:15m"},
        walk_forward_result={"windows": 1},
        promotion_evidence=ConcretePromotionEvidence(
            held_out_test=HeldOutTestEvidence(
                artifact_fingerprint="test-artifact",
                test_dataset_fingerprints=(("NSE_EQ|INE002A01018", "test-dataset"),),
                window_id=1,
                trade_count=120,
                profit_factor=Decimal("1.45"),
                max_drawdown_pct=Decimal(6),
                drawdown_basis=DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS,
                net_return_pct=Decimal(14),
                source_reference="test-source",
            ),
            cost_reconciliation=CostReconciliationEvidence(
                artifact_fingerprint="cost-artifact",
                schema_version="cost-reconciliation-v1",
                cost_model_name="documented",
                orders_checked=10,
                passed_count=10,
                failed_count=0,
                max_reconciliation_error_inr=Decimal(0),
                tolerance_inr=Decimal("0.01"),
                status="PASS",
            ),
            paper_trading=PaperTradingEvidence(
                artifact_fingerprint="paper-artifact",
                environment="offline-test",
                session_start=date(2026, 7, 1),
                session_end=date(2026, 7, 31),
                verified_orders_count=10,
                audit_log_fingerprint="paper-log",
                source_reference="paper-source",
            ),
            baseline_comparison=BaselineComparisonEvidence(
                artifact_fingerprint="baseline-artifact",
                baseline_candidate_id="baseline",
                evaluated_candidate_id="orb:15m",
                baseline_net_return_pct=Decimal(1),
                evaluated_net_return_pct=Decimal(14),
                outperformed=True,
            ),
            slippage_stress=SlippageStressEvidence(
                artifact_fingerprint="stress-artifact",
                scenarios_evaluated=("base", "stress"),
                stress_max_drawdown_pct=Decimal(8),
                stress_passed=True,
            ),
            event_simulation=EventDrivenSimulationEvidence(
                artifact_fingerprint="simulation-artifact",
                simulator_version="sim-v1",
                trade_count=120,
                initial_cash=Decimal(1000),
                final_cash=Decimal(1140),
            ),
        ),
        created_at=created_at,
    )


def test_cost_ledger_fingerprint_is_deterministic() -> None:
    first = EffectiveDatedCostLedger()
    second = EffectiveDatedCostLedger()

    assert first.to_json() == second.to_json()
    assert first.fingerprint() == second.fingerprint()
    assert json.loads(first.to_json())["schema_version"] == "effective-dated-cost-ledger/v1"


def test_experiment_identity_binds_cost_ledger_fingerprint() -> None:
    ledger = EffectiveDatedCostLedger()
    identity = CostEvidenceIdentity.from_ledger(
        ledger,
        on_date=date(2026, 9, 8),
        product=LedgerProduct.INTRADAY,
    )
    experiment = _experiment(cost_evidence_identity=identity)

    assert identity.ledger_fingerprint == ledger.fingerprint()
    assert (
        experiment.deterministic_payload()["cost_evidence_identity"]["ledger_fingerprint"]
        == ledger.fingerprint()
    )


def test_modifying_ledger_fingerprint_changes_experiment_fingerprint() -> None:
    experiment = _experiment()
    changed_identity = replace(
        experiment.cost_evidence_identity,
        ledger_fingerprint="b" * 64,
    )
    changed = replace(experiment, cost_evidence_identity=changed_identity)

    assert changed.deterministic_fingerprint() != experiment.deterministic_fingerprint()


def test_changing_cost_evidence_policy_changes_experiment_fingerprint() -> None:
    experiment = _experiment()
    changed_identity = replace(
        experiment.cost_evidence_identity,
        policy_identity="effective-dated-cost-ledger/alternate-policy/v1",
    )
    changed = replace(experiment, cost_evidence_identity=changed_identity)

    assert changed.deterministic_fingerprint() != experiment.deterministic_fingerprint()


def test_free_form_rates_cannot_claim_historical_actual_costs() -> None:
    with pytest.raises(ValueError, match="verified ledger evidence"):
        CostEvidenceIdentity(
            ledger_schema_version="effective-dated-cost-ledger/v1",
            ledger_fingerprint="a" * 64,
            evidence_classification=HISTORICAL_ACTUAL_LABEL,
            historical_actual=True,
            product_scope="INTRADAY",
            evidence_mode="free_form_rates",
            policy_identity="unverified",
            resolved_on_date=date(2026, 9, 8),
            selected_record_ids=(),
            unknown_components=(),
        )


def test_unknown_gst_stays_unknown_through_experiment_provenance() -> None:
    ledger = EffectiveDatedCostLedger()
    with pytest.raises(UnknownCostEvidence):
        ledger.resolve(
            LedgerComponent.GST,
            date(2026, 9, 8),
            LedgerProduct.INTRADAY,
            LedgerSide.BUY,
        )

    identity = _cost_identity()
    assert identity.evidence_classification == INCOMPLETE_LABEL
    assert identity.historical_actual is False
    assert any("GST" in component.upper() for component in identity.unknown_components)
    assert _experiment(cost_evidence_identity=identity).cost_evidence_class == INCOMPLETE_LABEL


def test_unknown_is_serialized_as_null_not_zero() -> None:
    payload = json.loads(EffectiveDatedCostLedger().to_json())
    gst_unknown = [
        record
        for record in payload["records"]
        if record["component"] == LedgerComponent.GST.value
        and record["evidence_class"] == EvidenceClass.UNKNOWN.value
    ]

    assert gst_unknown
    assert all(record["rate"] is None for record in gst_unknown)
    assert all(record["rate"] != "0" for record in gst_unknown)


def test_sep_8_experiment_cannot_use_sep_9_account_snapshot() -> None:
    ledger = EffectiveDatedCostLedger()
    with pytest.raises(UnknownCostEvidence):
        ledger.brokerage_record(date(2026, 9, 8), LedgerProduct.INTRADAY)
    snapshot = ledger.brokerage_record(ACCOUNT_SNAPSHOT_DATE, LedgerProduct.INTRADAY)
    assert snapshot.evidence_class is EvidenceClass.ACCOUNT_SNAPSHOT

    historical_identity = CostEvidenceIdentity.from_ledger(
        ledger,
        on_date=date(2026, 9, 8),
        product=LedgerProduct.INTRADAY,
    )
    assert historical_identity.resolved_on_date == date(2026, 9, 8)
    assert not any("account_snapshot" in record for record in historical_identity.selected_record_ids)
    assert historical_identity.historical_actual is False


def test_experiment_builder_rejects_sep_9_cost_evidence_for_sep_8_window() -> None:
    sep_9_identity = _cost_identity(on_date=ACCOUNT_SNAPSHOT_DATE)

    with pytest.raises(DataLeakageError, match="after research window end"):
        _experiment(cost_evidence_identity=sep_9_identity)


def test_sep_9_snapshot_is_explicitly_nonhistorical() -> None:
    identity = _cost_identity(on_date=ACCOUNT_SNAPSHOT_DATE)

    assert identity.evidence_classification == INCOMPLETE_LABEL
    assert identity.historical_actual is False
    assert any("account_snapshot" in record for record in identity.selected_record_ids)


def test_public_gst_scenario_is_explicit_and_not_default_historical_evidence() -> None:
    ledger = EffectiveDatedCostLedger()
    with pytest.raises(UnknownCostEvidence):
        ledger.resolve(
            LedgerComponent.GST,
            ACCOUNT_SNAPSHOT_DATE,
            LedgerProduct.INTRADAY,
            LedgerSide.BOTH,
        )
    public_gst = ledger.resolve(
        LedgerComponent.GST,
        ACCOUNT_SNAPSHOT_DATE,
        LedgerProduct.INTRADAY,
        LedgerSide.BOTH,
        evidence_classes=(EvidenceClass.BROKER_PUBLIC_SCENARIO,),
    )
    assert public_gst.rate == GST_RATE
    assert public_gst.historical_actual is False
    assert public_gst.evidence_class is EvidenceClass.BROKER_PUBLIC_SCENARIO

    scenario_identity = CostEvidenceIdentity.from_public_scenario(
        ledger,
        on_date=ACCOUNT_SNAPSHOT_DATE,
        product=LedgerProduct.INTRADAY,
        scenario_identity="upstox-public-terms",
    )
    assert scenario_identity.evidence_classification == SCENARIO_LABEL
    assert scenario_identity.evidence_mode == "public_scenario"
    assert scenario_identity.historical_actual is False


def test_incomplete_historical_evidence_cannot_pass_promotion_gate() -> None:
    ca_ledger = _sample_ca_ledger()
    experiment = _experiment(ca_ledger=ca_ledger)
    thresholds = PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal(10),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )

    passed, violations = experiment.evaluate_promotion_gate(
        thresholds, trusted_corporate_action_ledger=ca_ledger
    )

    assert passed is False
    assert any("not verified HISTORICAL_ACTUAL_COSTS" in violation for violation in violations)
    with pytest.raises(MissingEvidenceError, match="not verified HISTORICAL_ACTUAL_COSTS"):
        experiment.validate_integrity(
            promotion_thresholds=thresholds,
            trusted_corporate_action_ledger=ca_ledger,
        )


def test_pre_boundary_and_mii_transition_semantics_remain_exact() -> None:
    ledger = EffectiveDatedCostLedger()
    with pytest.raises(UnsupportedResearchDate):
        ledger.describe(date(2024, 6, 30), LedgerProduct.INTRADAY)
    with pytest.raises(UnknownCostEvidence):
        ledger.mii_transaction_record(date(2024, 9, 30))
    assert ledger.mii_transaction_record(date(2024, 10, 1)).rate == Decimal(297) / Decimal(
        10000000
    )
    assert ledger.mii_transaction_record(date(2026, 2, 28)).rate == Decimal(297) / Decimal(
        10000000
    )
    assert ledger.mii_transaction_record(date(2026, 3, 1)).rate == Decimal("306.99") / Decimal(
        10000000
    )
    with pytest.raises(UnknownCostEvidence):
        ledger.brokerage_record(date(2026, 9, 10), LedgerProduct.INTRADAY)


def test_volatile_created_at_does_not_change_immutable_identity() -> None:
    first = _experiment(created_at="2026-09-10T10:00:00+05:30")
    second = _experiment(created_at="2026-09-10T11:00:00+05:30")

    assert first.deterministic_fingerprint() == second.deterministic_fingerprint()
    assert first.experiment_id == second.experiment_id
    assert first.as_dict()["created_at"] != second.as_dict()["created_at"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("universe_fingerprint", "changed-universe"),
        ("instrument_dataset_fingerprints", {"NSE_EQ|INE002A01018": "changed-dataset"}),
        ("approved_capital", ApprovedCapital(amount_rupees=Decimal(10000))),
        (
            "strategy_definitions",
            (
                StrategySpec(
                    candidate_id="orb:30m",
                    strategy_name="opening_range_breakout",
                    research_basis="research basis",
                    source_refs=("strategy-source",),
                    parameters={"range_minutes": "30"},
                ),
            ),
        ),
    ),
)
def test_core_experiment_inputs_change_identity(field: str, value: object) -> None:
    experiment = _experiment()

    changed = replace(experiment, **{field: value})

    assert changed.deterministic_fingerprint() != experiment.deterministic_fingerprint()


def test_root_package_import_is_lazy_and_does_not_load_pandas() -> None:
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_root, env.get("PYTHONPATH", "")) if path
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import equity_engine; print('equity_engine.experiment' in sys.modules); print('pandas' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.stdout.splitlines() == ["False", "False"]
