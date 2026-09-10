from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

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
    canonical_sha256,
    ConcretePromotionEvidence,
    CorporateActionEvidenceIdentity,
    CostEvidenceIdentity,
    CostEvidenceMismatchError,
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


def _experiment(
    *,
    cost_evidence_identity: CostEvidenceIdentity | None = None,
    created_at: str = "2026-09-10T10:00:00+05:30",
) -> ExperimentArtifact:
    identity = cost_evidence_identity or _cost_identity()
    orchestrator = ExperimentOrchestrator(code_commit_sha="baa4e10aa")
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
        corporate_action_evidence=CorporateActionEvidenceIdentity(
            source="corporate-action-source",
            complete=True,
            blocking_events=(),
            evidence_fingerprint="corporate-action-fingerprint",
        ),
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


def test_verification_fields_are_not_public_constructor_inputs() -> None:
    with pytest.raises(TypeError, match="_verified_ledger_fingerprint"):
        CostEvidenceIdentity(
            ledger_schema_version="effective-dated-cost-ledger/v1",
            ledger_fingerprint="a" * 64,
            evidence_classification=INCOMPLETE_LABEL,
            historical_actual=False,
            product_scope=LedgerProduct.INTRADAY.value,
            evidence_mode="historical_resolution",
            policy_identity="untrusted",
            resolved_on_date=date(2026, 9, 8),
            selected_record_ids=(),
            unknown_components=(),
            _verified_ledger_fingerprint="a" * 64,
        )


def test_kiro_forged_historical_claim_cannot_pass_promotion_or_integrity() -> None:
    forged = CostEvidenceIdentity(
        ledger_schema_version="effective-dated-cost-ledger/v1",
        ledger_fingerprint="f" * 64,
        evidence_classification=HISTORICAL_ACTUAL_LABEL,
        historical_actual=True,
        product_scope=LedgerProduct.INTRADAY.value,
        evidence_mode="historical_resolution",
        policy_identity="effective-dated-cost-ledger/default-resolution/v1",
        resolved_on_date=date(2026, 9, 8),
        selected_record_ids=("0" * 64,),
        unknown_components=(),
    )
    experiment = _experiment(cost_evidence_identity=forged)
    ledger = EffectiveDatedCostLedger()
    thresholds = PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal(10),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )

    passed, violations = experiment.evaluate_promotion_gate(thresholds)
    assert passed is False
    assert any("trusted cost evidence ledger is required" in item for item in violations)
    with pytest.raises(MissingEvidenceError, match="trusted cost evidence ledger is required"):
        experiment.validate_integrity(promotion_thresholds=thresholds)
    with pytest.raises(MissingEvidenceError, match="cost evidence integrity mismatch"):
        experiment.validate_integrity(
            promotion_thresholds=thresholds,
            trusted_cost_ledger=ledger,
        )


def test_forged_ledger_fingerprint_is_rejected_against_trusted_ledger() -> None:
    ledger = EffectiveDatedCostLedger()
    identity = _cost_identity()
    forged = replace(identity, ledger_fingerprint="b" * 64)

    with pytest.raises(CostEvidenceMismatchError, match="ledger_fingerprint"):
        forged.validate_against_trusted_ledger(ledger)


def test_unrelated_genuine_ledger_is_rejected() -> None:
    ledger = EffectiveDatedCostLedger()
    identity = _cost_identity()
    unrelated = replace(ledger.records[-1], source_refs=("unrelated-genuine-source",))
    changed_ledger = EffectiveDatedCostLedger(records=ledger.records[:-1] + (unrelated,))

    with pytest.raises(CostEvidenceMismatchError, match="ledger_fingerprint"):
        identity.validate_against_trusted_ledger(changed_ledger)


@pytest.mark.parametrize("change", ("rate", "source", "unknowns"))
def test_complete_record_changes_are_rejected(change: str) -> None:
    ledger = EffectiveDatedCostLedger()
    identity = _cost_identity()
    records = list(ledger.records)
    if change == "rate":
        index = next(i for i, record in enumerate(records) if record.rate is not None)
        records[index] = replace(records[index], rate=records[index].rate + Decimal("0.000001"))
    elif change == "source":
        index = 0
        records[index] = replace(records[index], source_refs=("changed-record-source",))
    else:
        index = next(i for i, record in enumerate(records) if record.unknowns)
        records[index] = replace(records[index], unknowns=records[index].unknowns + ("tampered",))

    changed_ledger = EffectiveDatedCostLedger(records=tuple(records))
    with pytest.raises(CostEvidenceMismatchError, match="ledger_fingerprint|selected_record_ids"):
        identity.validate_against_trusted_ledger(changed_ledger)


def test_selected_record_id_is_complete_canonical_sha256() -> None:
    ledger = EffectiveDatedCostLedger()
    record = ledger.records[0]
    identity = _cost_identity()
    expected = canonical_sha256(record.to_dict())

    assert expected in identity.selected_record_ids
    assert len(expected) == 64
    changed = replace(record, formula=(record.formula or "") + "+tampered")
    assert canonical_sha256(changed.to_dict()) != expected


def test_sep_9_record_cannot_be_claimed_for_sep_8() -> None:
    ledger = EffectiveDatedCostLedger()
    sep_9 = CostEvidenceIdentity.from_ledger(
        ledger,
        on_date=date(2026, 9, 9),
        product=LedgerProduct.INTRADAY,
    )
    forged_sep_8 = replace(sep_9, resolved_on_date=date(2026, 9, 8))

    with pytest.raises(CostEvidenceMismatchError, match="selected_record_ids"):
        forged_sep_8.validate_against_trusted_ledger(ledger)


def test_public_scenario_cannot_be_claimed_as_historical_actual() -> None:
    ledger = EffectiveDatedCostLedger()
    scenario = CostEvidenceIdentity.from_public_scenario(
        ledger,
        on_date=date(2026, 9, 9),
        product=LedgerProduct.INTRADAY,
        scenario_identity="upstox-public-terms",
    )
    forged = replace(
        scenario,
        evidence_classification=HISTORICAL_ACTUAL_LABEL,
        historical_actual=True,
    )

    with pytest.raises(CostEvidenceMismatchError, match="evidence_classification"):
        forged.validate_against_trusted_ledger(
            ledger,
            trusted_scenario_identity="upstox-public-terms",
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
    claim = CostEvidenceIdentity(
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

    # Claims remain loadable as artifacts, but a free-form configuration is
    # rejected at the trusted-evidence integrity boundary.
    with pytest.raises(ValueError, match="unsupported cost-evidence mode"):
        claim.validate_against_trusted_ledger(EffectiveDatedCostLedger())


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
    ledger = EffectiveDatedCostLedger()
    identity = _cost_identity(on_date=ACCOUNT_SNAPSHOT_DATE)

    assert identity.evidence_classification == INCOMPLETE_LABEL
    assert identity.historical_actual is False
    snapshot_ids = {
        identity_value
        for record in ledger.records
        if record.evidence_class is EvidenceClass.ACCOUNT_SNAPSHOT
        for identity_value in (canonical_sha256(record.to_dict()),)
    }
    assert snapshot_ids.intersection(identity.selected_record_ids)


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
    experiment = _experiment()
    thresholds = PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal(10),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )

    passed, violations = experiment.evaluate_promotion_gate(thresholds)

    assert passed is False
    assert any("not verified HISTORICAL_ACTUAL_COSTS" in violation for violation in violations)
    with pytest.raises(MissingEvidenceError, match="not verified HISTORICAL_ACTUAL_COSTS"):
        experiment.validate_integrity(promotion_thresholds=thresholds)


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
