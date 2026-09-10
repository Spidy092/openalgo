from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from equity_engine.corporate_actions import (
    AdjustmentFactorRecord,
    CorporateActionConfidence,
    CorporateActionCoverageError,
    CorporateActionDataLeakageError,
    CorporateActionEvaluationMode,
    CorporateActionEvent,
    CorporateActionEventType,
    CorporateActionPolicy,
    CorporateActionRecord,
    CoverageScope,
    DividendPolicy,
    PointInTimeCorporateActionLedger,
    assess_corporate_actions,
    parse_corporate_action_rows,
    parse_ratio,
)
from equity_engine.experiment import (
    ApprovedCapital,
    BaselineComparisonEvidence,
    ConcretePromotionEvidence,
    CorporateActionEvidenceIdentity,
    CorporateActionMismatchError,
    CostEvidenceIdentity,
    CostModelIdentity,
    CostReconciliationEvidence,
    DrawdownBasis,
    EmbargoSpec,
    EventDrivenSimulationEvidence,
    ExperimentArtifact,
    ExperimentOrchestrator,
    HeldOutTestEvidence,
    MissingEvidenceError,
    NSEMembershipEvidenceIdentity,
    PaperTradingEvidence,
    ResearchWindowConfig,
    SessionPolicyIdentity,
    SlippageStressEvidence,
    StrategySpec,
    TickEvidenceIdentity,
    WindowSpec,
)
from equity_engine.gates import PromotionThresholds


def _sample_experiment(
    *,
    corporate_action_evidence: CorporateActionEvidenceIdentity,
    research_start: date = date(2026, 1, 1),
    research_end: date = date(2026, 6, 30),
    instrument_dataset_fingerprints: dict[str, str] | None = None,
) -> ExperimentArtifact:
    orchestrator = ExperimentOrchestrator(
        code_commit_sha="c4322d43956de1b43a764a7849b7437b38a1c932",
        vectorbt_version="1.1.0",
        simulator_version="openalgo-event-simulator-v1",
    )
    return orchestrator.build_experiment(
        research_window=ResearchWindowConfig(start=research_start, end=research_end),
        train_windows=(
            WindowSpec(window_id=1, start=research_start, end=date(2026, 3, 31), trading_days=60),
        ),
        validation_test_windows=(
            WindowSpec(window_id=1, start=date(2026, 4, 5), end=research_end, trading_days=60),
        ),
        embargo=EmbargoSpec(trading_days=2),
        approved_capital=ApprovedCapital(amount_rupees=Decimal(100000)),
        universe_fingerprint="fp_univ_001",
        candidate_prefilter_artifact_fingerprint="fp_univ_001",
        instrument_dataset_fingerprints=(
            instrument_dataset_fingerprints
            if instrument_dataset_fingerprints is not None
            else {"NSE_EQ|INE002A01018": "fp_ds_001"}
        ),
        nse_membership_evidence=NSEMembershipEvidenceIdentity(
            source_refs=("https://nsearchives.nseindia.com/members.csv",),
            complete=True,
            coverage_fingerprint="fp_mem_001",
            eligible_dates_count=120,
        ),
        tick_evidence=TickEvidenceIdentity(
            policy_name="tiered_tick",
            source="https://nsearchives.nseindia.com/tick.pdf",
            coverage_complete=True,
            coverage_fingerprint="fp_tick_001",
        ),
        session_policy_identity=SessionPolicyIdentity(
            policy_name="NSEEquitySessionPolicy",
            cas_eligible=True,
            exit_buffer_minutes=15,
            cas_effective_date="2026-03-01",
            continuous_end="15:30:00",
        ),
        corporate_action_evidence=corporate_action_evidence,
        cost_model_identity=CostModelIdentity(
            model_name="documented",
            effective_date="2026-03-01",
            rates={"brokerage": "0.001", "gst": "0.18"},
            source_refs=("source",),
        ),
        cost_evidence_identity=CostEvidenceIdentity(
            ledger_schema_version="effective-dated-cost-ledger/v1",
            ledger_fingerprint="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            evidence_classification="HISTORICAL_ACTUAL_COSTS",
            historical_actual=True,
            product_scope="INTRADAY",
            evidence_mode="historical_resolution",
            policy_identity="policy",
            resolved_on_date=date(2026, 6, 30),
            selected_record_ids=("rec1",),
            unknown_components=(),
            _verified_ledger_fingerprint="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            _verified_historical_actual=True,
        ),
        cost_evidence_class="HISTORICAL_ACTUAL_COSTS",
        strategy_definitions=(
            StrategySpec(
                candidate_id="c1",
                strategy_name="s1",
                research_basis="basis",
                source_refs=("ref",),
                parameters={"p": "1"},
            ),
        ),
        parameter_grid={"p": ("1",)},
        friction_scenarios=(),
        rejected_candidates=(),
        tournament_result={"winner": "s1"},
        walk_forward_result={"status": "pass"},
        promotion_evidence=ConcretePromotionEvidence(
            held_out_test=HeldOutTestEvidence(
                artifact_fingerprint="sha256_test_eval_artifact_001",
                test_dataset_fingerprints=(("NSE_EQ|INE002A01018", "fp_test_rel"),),
                window_id=1,
                trade_count=120,
                profit_factor=Decimal("1.45"),
                max_drawdown_pct=Decimal("6.50"),
                drawdown_basis=DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS,
                net_return_pct=Decimal("14.20"),
                source_reference="test_window_eval_log_001",
            ),
            cost_reconciliation=CostReconciliationEvidence(
                artifact_fingerprint="sha256_cost_recon_artifact_001",
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
                artifact_fingerprint="sha256_paper_trading_artifact_001",
                environment="upstox_sandbox_v2",
                session_start=date(2026, 7, 1),
                session_end=date(2026, 7, 31),
                verified_orders_count=45,
                audit_log_fingerprint="sha256_paper_audit_log_001",
                source_reference="broker_sandbox_order_log",
            ),
            baseline_comparison=BaselineComparisonEvidence(
                artifact_fingerprint="sha256_baseline_artifact_001",
                baseline_candidate_id="baseline:first-bar-hold",
                evaluated_candidate_id="orb:15m:vol1.5:buf5bps",
                baseline_net_return_pct=Decimal("2.10"),
                evaluated_net_return_pct=Decimal("14.20"),
                outperformed=True,
            ),
            slippage_stress=SlippageStressEvidence(
                artifact_fingerprint="sha256_slippage_stress_artifact_001",
                scenarios_evaluated=("base_2bps", "stress_5bps", "severe_10bps"),
                stress_max_drawdown_pct=Decimal("8.20"),
                stress_passed=True,
            ),
            event_simulation=EventDrivenSimulationEvidence(
                artifact_fingerprint="sha256_event_sim_artifact_001",
                simulator_version="openalgo-event-simulator-v1",
                trade_count=120,
                initial_cash=Decimal(100000),
                final_cash=Decimal(114200),
            ),
            unpriced_cost_components=(),
        ),
    )


def test_parse_and_block_structural_action_in_research_window() -> None:
    events = parse_corporate_action_rows(
        [
            {
                "name": "Split",
                "expiry_date": "14 Aug 2025",
                "amount": None,
                "ratio": "1:2",
                "event_details": [],
            },
            {
                "name": "Dividend",
                "expiry_date": "15 Sep 2025",
                "amount": 5.5,
                "ratio": None,
                "event_details": [],
            },
        ]
    )
    assessment = assess_corporate_actions(
        events=events,
        research_start=date(2025, 1, 1),
        research_end=date(2025, 12, 31),
        blocked_event_names=frozenset({"Split", "Bonus", "Rights"}),
    )

    assert assessment.complete is True
    assert assessment.blocking_events == ("Split@2025-08-14 ratio=1:2",)
    assert events[1].amount == Decimal("5.5")


def test_out_of_window_action_does_not_block() -> None:
    events = parse_corporate_action_rows(
        [{"name": "Bonus", "expiry_date": "01 Jan 2024", "amount": None, "ratio": "1:1"}]
    )
    assessment = assess_corporate_actions(
        events=events,
        research_start=date(2025, 1, 1),
        research_end=date(2026, 1, 1),
        blocked_event_names=frozenset({"Split", "Bonus", "Rights"}),
    )
    assert assessment.blocking_events == ()


def test_corporate_action_record_deterministic_fingerprint() -> None:
    ts = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
    r1 = CorporateActionRecord(
        instrument_key="NSE_EQ|INE002A01018",
        isin="INE002A01018",
        event_type=CorporateActionEventType.SPLIT,
        effective_date=date(2025, 8, 14),
        announcement_date=date(2025, 7, 1),
        ex_date=date(2025, 8, 14),
        source="NSE_CIRCULAR_CMTR12345",
        retrieval_timestamp=ts,
        confidence=CorporateActionConfidence.CONFIRMED,
        raw_candles_comparable=False,
        adjustment_required=True,
        blocking=True,
        ratio="1:2",
    )
    r2 = CorporateActionRecord(
        instrument_key="NSE_EQ|INE002A01018",
        isin="INE002A01018",
        event_type=CorporateActionEventType.SPLIT,
        effective_date=date(2025, 8, 14),
        announcement_date=date(2025, 7, 1),
        ex_date=date(2025, 8, 14),
        source="NSE_CIRCULAR_CMTR12345",
        retrieval_timestamp=ts,
        confidence=CorporateActionConfidence.CONFIRMED,
        raw_candles_comparable=False,
        adjustment_required=True,
        blocking=True,
        ratio="1:2",
    )
    r_changed = CorporateActionRecord(
        instrument_key="NSE_EQ|INE002A01018",
        isin="INE002A01018",
        event_type=CorporateActionEventType.SPLIT,
        effective_date=date(2025, 8, 14),
        announcement_date=date(2025, 7, 2),  # changed
        ex_date=date(2025, 8, 14),
        source="NSE_CIRCULAR_CMTR12345",
        retrieval_timestamp=ts,
        confidence=CorporateActionConfidence.CONFIRMED,
        raw_candles_comparable=False,
        adjustment_required=True,
        blocking=True,
        ratio="1:2",
    )

    assert r1.fingerprint() == r2.fingerprint()
    assert len(r1.fingerprint()) == 64
    assert r1.fingerprint() != r_changed.fingerprint()
    assert r1.knowledge_date == date(2025, 7, 1)


def test_unknown_event_coverage_is_not_treated_as_no_event() -> None:
    ledger = PointInTimeCorporateActionLedger()
    # Ledger has NO coverage recorded for this instrument
    assessment = ledger.assess_window(
        "NSE_EQ|INE002A01018",
        date(2025, 1, 1),
        date(2025, 12, 31),
    )
    # UNKNOWN coverage MUST fail closed: complete is False, never True
    assert assessment.complete is False
    assert len(assessment.blocking_events) == 1
    assert "UNCOVERED_WINDOW" in assessment.blocking_events[0]


def test_date_scoped_coverage_exact_boundaries() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)

    # Coverage for [2025-01-01, 2025-06-30]
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )

    # Exact coverage matches
    assert ledger.is_covered("NSE_EQ|INE002A01018", date(2025, 1, 1), date(2025, 6, 30)) is True
    # Sub-window is covered
    assert ledger.is_covered("NSE_EQ|INE002A01018", date(2025, 2, 1), date(2025, 5, 31)) is True
    # Window extending even 1 day beyond coverage fails closed
    assert ledger.is_covered("NSE_EQ|INE002A01018", date(2025, 1, 1), date(2025, 7, 1)) is False
    # Window starting 1 day before coverage fails closed
    assert ledger.is_covered("NSE_EQ|INE002A01018", date(2024, 12, 31), date(2025, 6, 30)) is False

    # Add contiguous second coverage scope
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2025, 7, 1),
            end_date=date(2025, 12, 31),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    # Merged contiguous span covers entire year
    assert ledger.is_covered("NSE_EQ|INE002A01018", date(2025, 1, 1), date(2025, 12, 31)) is True

    # Assessment on covered window succeeds
    assessment = ledger.assess_window(
        "NSE_EQ|INE002A01018",
        date(2025, 1, 1),
        date(2025, 12, 31),
    )
    assert assessment.complete is True
    assert assessment.blocking_events == ()


def test_split_boundary_blocks_unadjusted_research_window() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            source="NSE_FEED",
            retrieval_timestamp=ts,
        )
    )
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            event_type=CorporateActionEventType.SPLIT,
            effective_date=date(2025, 8, 14),
            announcement_date=date(2025, 7, 1),
            ex_date=date(2025, 8, 14),
            source="NSE_CIRCULAR",
            retrieval_timestamp=ts,
            raw_candles_comparable=False,
            adjustment_required=True,
            blocking=True,
            ratio="1:2",
        )
    )

    # Unadjusted default policy blocks
    assessment = ledger.assess_window(
        "NSE_EQ|INE002A01018",
        date(2025, 1, 1),
        date(2025, 12, 31),
    )
    assert assessment.complete is True
    assert assessment.blocking_events == ("SPLIT@2025-08-14 ratio=1:2",)

    # When explicit policy allows ex-post normalized splits, it does not block
    adjusted_policy = CorporateActionPolicy(allow_ex_post_adjusted_splits=True)
    assessment_adj = ledger.assess_window(
        "NSE_EQ|INE002A01018",
        date(2025, 1, 1),
        date(2025, 12, 31),
        policy=adjusted_policy,
    )
    assert assessment_adj.complete is True
    assert assessment_adj.blocking_events == ()


def test_bonus_boundary_blocks_unadjusted_research_window() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            source="NSE_FEED",
            retrieval_timestamp=ts,
        )
    )
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            event_type=CorporateActionEventType.BONUS,
            effective_date=date(2025, 5, 20),
            announcement_date=date(2025, 4, 15),
            ex_date=date(2025, 5, 20),
            source="NSE_CIRCULAR",
            retrieval_timestamp=ts,
            raw_candles_comparable=False,
            adjustment_required=True,
            blocking=True,
            ratio="1:1",
        )
    )

    assessment = ledger.assess_window(
        "NSE_EQ|INE002A01018",
        date(2025, 1, 1),
        date(2025, 12, 31),
    )
    assert assessment.blocking_events == ("BONUS@2025-05-20 ratio=1:1",)

    # Ex-post bonus policy unblocks
    policy = CorporateActionPolicy(allow_ex_post_adjusted_bonuses=True)
    assert (
        ledger.assess_window(
            "NSE_EQ|INE002A01018",
            date(2025, 1, 1),
            date(2025, 12, 31),
            policy=policy,
        ).blocking_events
        == ()
    )


def test_ex_post_adjustment_factor_computation() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)

    # 1:2 split
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|STOCK1",
            isin="INE001A01011",
            event_type=CorporateActionEventType.SPLIT,
            effective_date=date(2025, 4, 1),
            source="NSE",
            retrieval_timestamp=ts,
            ratio="1:2",
        )
    )
    # 2:1 bonus (2 bonus shares for 1 held => 1 becomes 3 shares)
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|STOCK1",
            isin="INE001A01011",
            event_type=CorporateActionEventType.BONUS,
            effective_date=date(2025, 8, 1),
            source="NSE",
            retrieval_timestamp=ts,
            ratio="2:1",
        )
    )
    # ₹10 dividend
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|STOCK1",
            isin="INE001A01011",
            event_type=CorporateActionEventType.DIVIDEND,
            effective_date=date(2025, 10, 1),
            source="NSE",
            retrieval_timestamp=ts,
            amount=Decimal("10.00"),
        )
    )

    factors = ledger.compute_adjustment_factors("NSE_EQ|STOCK1", date(2025, 1, 1), date(2025, 12, 31))
    assert len(factors) == 3

    # Split 1:2: old price halved, volume doubled
    assert factors[0].effective_date == date(2025, 4, 1)
    assert factors[0].price_multiplier == Decimal("0.5")
    assert factors[0].volume_multiplier == Decimal("2.0")

    # Bonus 2:1: 1 share becomes 3 shares: price / 3, volume * 3
    assert factors[1].effective_date == date(2025, 8, 1)
    assert factors[1].price_multiplier == Decimal(1) / Decimal(3)
    assert factors[1].volume_multiplier == Decimal(3)

    # Dividend ₹10: cash adjustment
    assert factors[2].effective_date == date(2025, 10, 1)
    assert factors[2].cash_adjustment == Decimal("10.00")


def test_future_announcement_leakage_is_strictly_prevented() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|LEAK_TEST",
            isin="INE999A01099",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            source="NSE",
            retrieval_timestamp=ts,
        )
    )
    # Split taking effect 2025-09-01, announced 2025-08-01
    split_record = CorporateActionRecord(
        instrument_key="NSE_EQ|LEAK_TEST",
        isin="INE999A01099",
        event_type=CorporateActionEventType.SPLIT,
        effective_date=date(2025, 9, 1),
        announcement_date=date(2025, 8, 1),
        source="NSE",
        retrieval_timestamp=ts,
        ratio="1:2",
    )
    ledger.add_record(split_record)

    # 1. Trading decision on 2025-07-15:
    # At trading time 2025-07-15, the split announcement did not exist yet!
    tradable_july = ledger.get_tradable_events("NSE_EQ|LEAK_TEST", as_of_date=date(2025, 7, 15))
    assert tradable_july == ()

    # 2. Trading decision on 2025-08-05:
    # Now the split has been announced, so it is known at trading time
    tradable_aug = ledger.get_tradable_events("NSE_EQ|LEAK_TEST", as_of_date=date(2025, 8, 5))
    assert len(tradable_aug) == 1
    assert tradable_aug[0].event_type == CorporateActionEventType.SPLIT

    # 3. Attempting to verify no future leakage raises error if an event was announced in the future
    with pytest.raises(CorporateActionDataLeakageError, match="announced on 2025-08-01, which is after"):
        ledger.verify_no_future_leakage((split_record,), as_of_date=date(2025, 7, 15))


def test_ex_post_normalization_mode_explicitly_allows_historical_adjustment() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|EX_POST",
            isin="INE888A01088",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            source="NSE",
            retrieval_timestamp=ts,
        )
    )
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|EX_POST",
            isin="INE888A01088",
            event_type=CorporateActionEventType.SPLIT,
            effective_date=date(2025, 6, 15),
            announcement_date=date(2025, 5, 1),
            source="NSE",
            retrieval_timestamp=ts,
            ratio="1:5",
        )
    )

    # In EX_POST_NORMALIZATION mode, events across the historical window are retrieved for offline adjustment
    events = ledger.get_ex_post_events("NSE_EQ|EX_POST", date(2025, 1, 1), date(2025, 12, 31))
    assert len(events) == 1
    assert events[0].ratio == "1:5"


def test_symbol_change_point_in_time_resolution() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|INE010B01027",
            isin="INE010B01027",
            event_type=CorporateActionEventType.SYMBOL_CHANGE,
            effective_date=date(2022, 5, 23),
            announcement_date=date(2022, 5, 10),
            source="NSE_CIRCULAR",
            retrieval_timestamp=ts,
            from_symbol="CADILAHC",
            to_symbol="ZYDUSLIFE",
            raw_candles_comparable=True,
            adjustment_required=False,
            blocking=False,
        )
    )

    # Before effective date: CADILAHC
    assert ledger.resolve_symbol_at("INE010B01027", date(2022, 5, 22)) == "CADILAHC"
    # On and after effective date: ZYDUSLIFE
    assert ledger.resolve_symbol_at("INE010B01027", date(2022, 5, 23)) == "ZYDUSLIFE"
    assert ledger.resolve_symbol_at("INE010B01027", date(2026, 1, 1)) == "ZYDUSLIFE"


def test_isin_change_blocks_when_raw_candles_incomparable() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE111A01010",
            isin="INE111A01010",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            source="NSE",
            retrieval_timestamp=ts,
        )
    )
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|INE111A01010",
            isin="INE111A01010",
            event_type=CorporateActionEventType.ISIN_CHANGE,
            effective_date=date(2025, 6, 1),
            source="NSE",
            retrieval_timestamp=ts,
            old_isin="INE111A01010",
            new_isin="INE111A01028",
            raw_candles_comparable=False,  # Capital reorganization
            adjustment_required=True,
            blocking=True,
        )
    )

    assessment = ledger.assess_window("NSE_EQ|INE111A01010", date(2025, 1, 1), date(2025, 12, 31))
    assert assessment.complete is True
    assert assessment.blocking_events == ("ISIN_CHANGE@2025-06-01",)


def test_merger_and_demerger_block_window() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|MERGER_STOCK",
            isin="INE222B01022",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            source="NSE",
            retrieval_timestamp=ts,
        )
    )
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|MERGER_STOCK",
            isin="INE222B01022",
            event_type=CorporateActionEventType.MERGER,
            effective_date=date(2025, 7, 1),
            source="NSE",
            retrieval_timestamp=ts,
            raw_candles_comparable=False,
            adjustment_required=True,
            blocking=True,
        )
    )

    assessment = ledger.assess_window("NSE_EQ|MERGER_STOCK", date(2025, 1, 1), date(2025, 12, 31))
    assert assessment.blocking_events == ("MERGER@2025-07-01",)


def test_delisting_and_relisting_boundaries() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|DELIST_STOCK",
            isin="INE333C01033",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            source="NSE",
            retrieval_timestamp=ts,
        )
    )
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|DELIST_STOCK",
            isin="INE333C01033",
            event_type=CorporateActionEventType.DELISTING,
            effective_date=date(2025, 8, 31),
            source="NSE",
            retrieval_timestamp=ts,
            raw_candles_comparable=False,
            adjustment_required=True,
            blocking=True,
        )
    )

    # Window prior to delisting is not blocked
    assert (
        ledger.assess_window(
            "NSE_EQ|DELIST_STOCK",
            date(2025, 1, 1),
            date(2025, 8, 30),
        ).blocking_events
        == ()
    )

    # Window containing delisting is blocked
    assert (
        ledger.assess_window(
            "NSE_EQ|DELIST_STOCK",
            date(2025, 1, 1),
            date(2025, 9, 30),
        ).blocking_events
        == ("DELISTING@2025-08-31",)
    )


def test_dividend_policy_intraday_ignore_vs_extraordinary_blocking() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|DIV_STOCK",
            isin="INE444D01044",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            source="NSE",
            retrieval_timestamp=ts,
        )
    )
    # Small ordinary dividend: ₹2 on a ₹500 stock = 0.4%
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|DIV_STOCK",
            isin="INE444D01044",
            event_type=CorporateActionEventType.DIVIDEND,
            effective_date=date(2025, 4, 15),
            source="NSE",
            retrieval_timestamp=ts,
            amount=Decimal("2.00"),
            raw_candles_comparable=True,
            adjustment_required=False,
            blocking=False,
        )
    )
    # Extraordinary special dividend: ₹25 on a ₹500 stock = 5.0% (> 2% threshold)
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|DIV_STOCK",
            isin="INE444D01044",
            event_type=CorporateActionEventType.DIVIDEND,
            effective_date=date(2025, 9, 15),
            source="NSE",
            retrieval_timestamp=ts,
            amount=Decimal("25.00"),
            raw_candles_comparable=False,
            adjustment_required=True,
            blocking=True,
        )
    )

    policy = CorporateActionPolicy(
        dividend_policy=DividendPolicy.IGNORE_BELOW_THRESHOLD,
        dividend_threshold_percent=Decimal("2.0"),
        reference_price_for_dividend=Decimal("500.00"),
    )

    # Window covering only ordinary dividend passes
    assessment_ord = ledger.assess_window(
        "NSE_EQ|DIV_STOCK",
        date(2025, 1, 1),
        date(2025, 6, 30),
        policy=policy,
    )
    assert assessment_ord.blocking_events == ()

    # Window covering extraordinary dividend is blocked
    assessment_ext = ledger.assess_window(
        "NSE_EQ|DIV_STOCK",
        date(2025, 7, 1),
        date(2025, 12, 31),
        policy=policy,
    )
    assert len(assessment_ext.blocking_events) == 1
    assert "DIVIDEND@2025-09-15" in assessment_ext.blocking_events[0]
    assert "dividend_yield_pct=5.00%>=cap" in assessment_ext.blocking_events[0]


def test_experiment_cannot_claim_corporate_action_complete_without_exact_window_coverage() -> None:
    # 1. Uncovered window fails closed
    uncovered_evidence = CorporateActionEvidenceIdentity(
        source="NSE",
        complete=True,
        blocking_events=(),
        evidence_fingerprint="fp_ca_001",
        coverage_start=date(2026, 1, 1),
        coverage_end=date(2026, 5, 31),  # Ends early; research window is until 2026-06-30
        covered_instruments=("NSE_EQ|INE002A01018",),
        authoritative=False,
    )
    exp = _sample_experiment(corporate_action_evidence=uncovered_evidence)
    with pytest.raises(MissingEvidenceError, match="cannot claim corporate-action-complete unless evidence covers"):
        exp.validate_integrity()

    # 2. Incomplete evidence fails closed
    with pytest.raises(ValueError, match="corporate-action evidence must be complete"):
        CorporateActionEvidenceIdentity(
            source="NSE",
            complete=False,
            blocking_events=(),
            evidence_fingerprint="fp_ca_001",
            covered_instruments=("NSE_EQ|INE002A01018",),
            authoritative=False,
        )

    # 3. Missing date-scoped coverage fails closed
    no_dates_evidence = CorporateActionEvidenceIdentity(
        source="NSE",
        complete=True,
        blocking_events=(),
        evidence_fingerprint="fp_ca_001",
        coverage_start=None,
        coverage_end=None,
        covered_instruments=("NSE_EQ|INE002A01018",),
        authoritative=False,
    )
    exp_no_dates = _sample_experiment(corporate_action_evidence=no_dates_evidence)
    with pytest.raises(MissingEvidenceError, match=r"got coverage \[None, None\]"):
        exp_no_dates.validate_integrity()


def test_corporate_action_evidence_identity_from_ledger_integration() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )

    ca_identity = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE002A01018",),
    )

    assert ca_identity.complete is True
    assert ca_identity.coverage_start == date(2026, 1, 1)
    assert ca_identity.coverage_end == date(2026, 6, 30)
    assert ca_identity.covered_instruments == ("NSE_EQ|INE002A01018",)
    assert ca_identity.blocking_events == ()
    assert ca_identity.evidence_fingerprint == ledger.fingerprint()
    assert ca_identity.authoritative is True

    # Authoritative claim derived from ledger requires trusted ledger to validate integrity
    exp = _sample_experiment(corporate_action_evidence=ca_identity)
    with pytest.raises(MissingEvidenceError, match="trusted corporate-action evidence ledger is required"):
        exp.validate_integrity()

    # Revalidation against trusted ledger passes cleanly
    exp.validate_integrity(trusted_corporate_action_ledger=ledger)
    assert exp.deterministic_fingerprint() is not None


def test_fix1_caller_supplied_fingerprints_rejected_when_spoofed() -> None:
    # 1. CorporateActionRecord rejects spoofed fingerprint
    with pytest.raises(ValueError, match="invalid caller-supplied evidence_fingerprint"):
        CorporateActionRecord(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            event_type=CorporateActionEventType.SPLIT,
            effective_date=date(2026, 3, 1),
            source="NSE_FEED",
            retrieval_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
            evidence_fingerprint="spoofed_record_fp",
        )

    # Valid record derives canonical fingerprint automatically
    rec = CorporateActionRecord(
        instrument_key="NSE_EQ|INE002A01018",
        isin="INE002A01018",
        event_type=CorporateActionEventType.SPLIT,
        effective_date=date(2026, 3, 1),
        source="NSE_FEED",
        retrieval_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert rec.evidence_fingerprint == rec.fingerprint()

    # Re-supplying the exact canonical fingerprint is allowed
    rec_exact = CorporateActionRecord(
        instrument_key="NSE_EQ|INE002A01018",
        isin="INE002A01018",
        event_type=CorporateActionEventType.SPLIT,
        effective_date=date(2026, 3, 1),
        source="NSE_FEED",
        retrieval_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        evidence_fingerprint=rec.fingerprint(),
    )
    assert rec_exact.evidence_fingerprint == rec.evidence_fingerprint

    # Deserializing tampered payload via from_dict fails closed
    record_dict = rec.as_dict()
    record_dict["evidence_fingerprint"] = "tampered_dict_fp"
    with pytest.raises(ValueError, match="invalid caller-supplied evidence_fingerprint"):
        CorporateActionRecord.from_dict(record_dict)

    # 2. CoverageScope rejects spoofed fingerprint
    with pytest.raises(ValueError, match="invalid caller-supplied coverage_fingerprint"):
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            coverage_fingerprint="spoofed_coverage_fp",
        )

    cov = CoverageScope(
        instrument_key="NSE_EQ|INE002A01018",
        isin="INE002A01018",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 6, 30),
        source="NSE_FEED",
        retrieval_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert cov.coverage_fingerprint == cov.fingerprint()

    cov_dict = cov.as_dict()
    cov_dict["coverage_fingerprint"] = "tampered_cov_dict_fp"
    with pytest.raises(ValueError, match="invalid caller-supplied coverage_fingerprint"):
        CoverageScope.from_dict(cov_dict)


def test_fix2_empty_instrument_population_fails_closed() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )

    # Empty instruments requested on ledger must fail closed with CorporateActionCoverageError
    with pytest.raises(CorporateActionCoverageError, match="requested instrument population is empty"):
        ledger.to_evidence_identity(
            research_start=date(2026, 1, 1),
            research_end=date(2026, 6, 30),
            instruments=(),
        )

    # Empty instruments in CorporateActionEvidenceIdentity must fail closed
    with pytest.raises(ValueError, match="covered_instruments cannot be empty"):
        CorporateActionEvidenceIdentity(
            source="NSE",
            complete=True,
            blocking_events=(),
            evidence_fingerprint="fp_ca_empty",
            coverage_start=date(2026, 1, 1),
            coverage_end=date(2026, 6, 30),
            covered_instruments=(),
        )


def test_exploit_private_parameters_removed_and_forged_identity_rejected() -> None:
    # 1. Private verification parameters must NOT exist on constructor
    with pytest.raises(TypeError, match="unexpected keyword argument '_verified_ledger_fingerprint'"):
        CorporateActionEvidenceIdentity(
            source="CANONICAL_PIT_CORPORATE_ACTION_LEDGER",
            complete=True,
            blocking_events=(),
            evidence_fingerprint="f" * 64,
            coverage_start=date(2026, 1, 1),
            coverage_end=date(2026, 6, 30),
            covered_instruments=("NSE_EQ|INE002A01018",),
            _verified_ledger_fingerprint="f" * 64,  # type: ignore[call-arg]
        )

    with pytest.raises(TypeError, match="unexpected keyword argument '_verified_covered_instruments'"):
        CorporateActionEvidenceIdentity(
            source="CANONICAL_PIT_CORPORATE_ACTION_LEDGER",
            complete=True,
            blocking_events=(),
            evidence_fingerprint="f" * 64,
            coverage_start=date(2026, 1, 1),
            coverage_end=date(2026, 6, 30),
            covered_instruments=("NSE_EQ|INE002A01018",),
            _verified_covered_instruments=("NSE_EQ|INE002A01018",),  # type: ignore[call-arg]
        )

    # 2. Construct forged identity manually with complete=True, fake fingerprint, correct coverage,
    # covered instruments, no blocking events, claiming authoritative=True
    forged = CorporateActionEvidenceIdentity(
        source="CANONICAL_PIT_CORPORATE_ACTION_LEDGER",
        complete=True,
        blocking_events=(),
        evidence_fingerprint="f" * 64,
        coverage_start=date(2026, 1, 1),
        coverage_end=date(2026, 6, 30),
        covered_instruments=("NSE_EQ|INE002A01018",),
        events_count=0,
        policy_identity="DEFAULT",
        authoritative=True,
    )

    exp_forged = _sample_experiment(corporate_action_evidence=forged)

    thresholds = PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal(10),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )

    # Cannot pass promotion gate without trusted ledger
    passed, violations = exp_forged.evaluate_promotion_gate(thresholds)
    assert passed is False
    assert any("trusted corporate-action evidence ledger is required for promotion" in v for v in violations)

    # Cannot pass promotion integrity check without trusted ledger
    with pytest.raises(MissingEvidenceError, match="trusted corporate-action evidence ledger is required"):
        exp_forged.validate_integrity(promotion_thresholds=thresholds)

    # Cannot claim authoritative evidence without trusted ledger
    with pytest.raises(MissingEvidenceError, match="trusted corporate-action evidence ledger is required"):
        exp_forged.validate_integrity()

    # Real ledger rejects forged claim
    real_ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    real_ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    with pytest.raises(MissingEvidenceError, match="corporate-action evidence integrity mismatch"):
        exp_forged.validate_integrity(trusted_corporate_action_ledger=real_ledger)


def test_genuine_ledger_derived_claim_plus_same_ledger_passes_revalidation() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    claim = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE002A01018",),
    )

    # Direct revalidation against same ledger
    claim.validate_against_trusted_ledger(ledger)

    # Experiment integrity validation with same ledger passes
    exp = _sample_experiment(corporate_action_evidence=claim)
    exp.validate_integrity(trusted_corporate_action_ledger=ledger)

    # Promotion evaluation passes corporate action boundary with trusted ledger
    thresholds = PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal(10),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )
    passed, violations = exp.evaluate_promotion_gate(
        thresholds,
        trusted_corporate_action_ledger=ledger,
    )
    assert not any("corporate-action" in v for v in violations)


def test_unrelated_ledger_rejected() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    claim = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE002A01018",),
    )

    unrelated_ledger = PointInTimeCorporateActionLedger()
    unrelated_ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="UNRELATED_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )

    with pytest.raises(CorporateActionMismatchError, match="evidence_fingerprint"):
        claim.validate_against_trusted_ledger(unrelated_ledger)

    exp = _sample_experiment(corporate_action_evidence=claim)
    with pytest.raises(MissingEvidenceError, match="corporate-action evidence integrity mismatch"):
        exp.validate_integrity(trusted_corporate_action_ledger=unrelated_ledger)


def test_modified_event_rejected() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    claim = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE002A01018",),
    )

    # Modified ledger with added event
    modified_ledger = PointInTimeCorporateActionLedger()
    modified_ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    modified_ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            event_type=CorporateActionEventType.DIVIDEND,
            effective_date=date(2026, 3, 1),
            announcement_date=date(2026, 2, 1),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            amount=Decimal("5.0"),
        )
    )

    with pytest.raises(CorporateActionMismatchError, match="evidence_fingerprint"):
        claim.validate_against_trusted_ledger(modified_ledger)


def test_modified_coverage_rejected() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    claim = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE002A01018",),
    )

    forged_start = replace(claim, coverage_start=date(2026, 1, 2))
    with pytest.raises(CorporateActionMismatchError, match="coverage_start"):
        forged_start.validate_against_trusted_ledger(ledger, research_start=date(2026, 1, 1))

    forged_end = replace(claim, coverage_end=date(2026, 6, 29))
    with pytest.raises(CorporateActionMismatchError, match="coverage_end"):
        forged_end.validate_against_trusted_ledger(ledger, research_end=date(2026, 6, 30))


def test_changed_policy_rejected() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    claim = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE002A01018",),
    )

    forged = replace(claim, policy_identity="TAMPERED_POLICY")
    with pytest.raises(CorporateActionMismatchError, match="policy_identity"):
        forged.validate_against_trusted_ledger(ledger)


def test_partial_instruments_rejected() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    for inst, isin in [("NSE_EQ|INE001A01010", "INE001A01010"), ("NSE_EQ|INE002A01018", "INE002A01018")]:
        ledger.add_coverage(
            CoverageScope(
                instrument_key=inst,
                isin=isin,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 6, 30),
                source="NSE_FEED",
                retrieval_timestamp=ts,
                is_complete=True,
            )
        )

    claim_partial = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE002A01018",),
    )
    with pytest.raises(CorporateActionMismatchError, match="covered_instruments"):
        claim_partial.validate_against_trusted_ledger(
            ledger,
            canonical_instruments=("NSE_EQ|INE001A01010", "NSE_EQ|INE002A01018"),
        )


def test_extra_instruments_rejected() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    for inst, isin in [("NSE_EQ|INE001A01010", "INE001A01010"), ("NSE_EQ|INE002A01018", "INE002A01018")]:
        ledger.add_coverage(
            CoverageScope(
                instrument_key=inst,
                isin=isin,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 6, 30),
                source="NSE_FEED",
                retrieval_timestamp=ts,
                is_complete=True,
            )
        )

    claim_both = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE001A01010", "NSE_EQ|INE002A01018"),
    )
    with pytest.raises(CorporateActionMismatchError, match="covered_instruments"):
        claim_both.validate_against_trusted_ledger(
            ledger,
            canonical_instruments=("NSE_EQ|INE002A01018",),
        )


def test_missing_trusted_ledger_blocks_promotion() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    claim = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE002A01018",),
    )
    exp = _sample_experiment(corporate_action_evidence=claim)

    thresholds = PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal(10),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )
    passed, violations = exp.evaluate_promotion_gate(thresholds, trusted_corporate_action_ledger=None)
    assert passed is False
    assert any("trusted corporate-action evidence ledger is required for promotion" in v for v in violations)


def test_non_authoritative_claim_allowed_in_research_artifact_but_blocks_promotion() -> None:
    non_auth_claim = CorporateActionEvidenceIdentity(
        source="CANONICAL_PIT_CORPORATE_ACTION_LEDGER",
        complete=True,
        blocking_events=(),
        evidence_fingerprint="fp_claim_001",
        coverage_start=date(2026, 1, 1),
        coverage_end=date(2026, 6, 30),
        covered_instruments=("NSE_EQ|INE002A01018",),
        events_count=0,
        policy_identity="DEFAULT",
        authoritative=False,
    )
    exp = _sample_experiment(corporate_action_evidence=non_auth_claim)

    # Research artifact carries non-authoritative claim and passes integrity validation
    exp.validate_integrity()

    # But cannot pass promotion
    thresholds = PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal(10),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )
    passed, violations = exp.evaluate_promotion_gate(thresholds)
    assert passed is False
    assert any("trusted corporate-action evidence ledger is required for promotion" in v for v in violations)
    assert any("corporate-action evidence is non-authoritative claim" in v for v in violations)

    # And cannot pass promotion integrity check
    with pytest.raises(MissingEvidenceError, match="experiment failed promotion gate criteria"):
        exp.validate_integrity(promotion_thresholds=thresholds)


def test_fix4_exact_instrument_coverage_enforcement() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    for inst, isin in [("NSE_EQ|INE001A01010", "INE001A01010"), ("NSE_EQ|INE002A01018", "INE002A01018")]:
        ledger.add_coverage(
            CoverageScope(
                instrument_key=inst,
                isin=isin,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 6, 30),
                source="NSE_FEED",
                retrieval_timestamp=ts,
                is_complete=True,
            )
        )

    # Partial coverage: only covers INE002
    partial_identity = CorporateActionEvidenceIdentity.from_ledger(
        ledger,
        research_window=ResearchWindowConfig(start=date(2026, 1, 1), end=date(2026, 6, 30)),
        instruments=("NSE_EQ|INE002A01018",),
    )

    exp_partial = _sample_experiment(
        corporate_action_evidence=partial_identity,
        instrument_dataset_fingerprints={
            "NSE_EQ|INE001A01010": "fp_ds_001",
            "NSE_EQ|INE002A01018": "fp_ds_002",
        },
    )

    # Validate integrity fails closed on partial instrument coverage
    with pytest.raises(MissingEvidenceError, match="does not exactly match canonical experiment instrument population"):
        exp_partial.validate_integrity()

    # Promotion gate also fails closed
    thresholds = PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.2"),
        max_drawdown_pct=Decimal(10),
        min_walk_forward_windows=1,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )
    passed, violations = exp_partial.evaluate_promotion_gate(thresholds)
    assert passed is False
    assert any("does not cover exact experiment instrument population" in v for v in violations)


def test_fix5_remove_hidden_2_percent_dividend_threshold() -> None:
    # Default policy has BLOCK_ALL and None threshold (no implicit 2%!)
    default_policy = CorporateActionPolicy()
    assert default_policy.dividend_policy == DividendPolicy.BLOCK_ALL
    assert default_policy.dividend_threshold_percent is None
    assert default_policy.reference_price_for_dividend is None
    assert default_policy.policy_identity == "DEFAULT_RESEARCH_POLICY:BLOCK_ALL"

    # IGNORE_BELOW_THRESHOLD requires explicit positive threshold and positive reference price
    with pytest.raises(ValueError, match="requires an explicit positive dividend_threshold_percent"):
        CorporateActionPolicy(
            dividend_policy=DividendPolicy.IGNORE_BELOW_THRESHOLD,
            dividend_threshold_percent=None,
            reference_price_for_dividend=Decimal(100),
        )

    with pytest.raises(ValueError, match="requires an explicit positive dividend_threshold_percent"):
        CorporateActionPolicy(
            dividend_policy=DividendPolicy.IGNORE_BELOW_THRESHOLD,
            dividend_threshold_percent=Decimal(0),
            reference_price_for_dividend=Decimal(100),
        )

    with pytest.raises(ValueError, match="requires an explicit positive reference_price_for_dividend"):
        CorporateActionPolicy(
            dividend_policy=DividendPolicy.IGNORE_BELOW_THRESHOLD,
            dividend_threshold_percent=Decimal("2.0"),
            reference_price_for_dividend=None,
        )

    with pytest.raises(ValueError, match="requires an explicit positive reference_price_for_dividend"):
        CorporateActionPolicy(
            dividend_policy=DividendPolicy.IGNORE_BELOW_THRESHOLD,
            dividend_threshold_percent=Decimal("2.0"),
            reference_price_for_dividend=Decimal(0),
        )

    # Valid IGNORE_BELOW_THRESHOLD policy binds threshold to identity
    valid_threshold_policy = CorporateActionPolicy(
        dividend_policy=DividendPolicy.IGNORE_BELOW_THRESHOLD,
        dividend_threshold_percent=Decimal("2.5"),
        reference_price_for_dividend=Decimal(1000),
        policy_name="default",
    )
    assert valid_threshold_policy.policy_identity == "default:IGNORE_BELOW_THRESHOLD:2.5%"

    # Under default BLOCK_ALL, even a small dividend (e.g. 0.5%) produces a blocking event in window assessment
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )
    ledger.add_record(
        CorporateActionRecord(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            event_type=CorporateActionEventType.DIVIDEND,
            effective_date=date(2026, 3, 1),
            announcement_date=date(2026, 2, 1),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            amount=Decimal("5.0"),
        )
    )

    # Default policy blocks window
    assess_default = ledger.assess_window(
        instrument_key="NSE_EQ|INE002A01018",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 6, 30),
        policy=default_policy,
    )
    assert assess_default.complete is True
    assert len(assess_default.blocking_events) >= 1
    assert any("DIVIDEND@2026-03-01" in b for b in assess_default.blocking_events)

    # Valid IGNORE_BELOW_THRESHOLD (2.5%) ignores 0.5% dividend (5 rupees / 1000)
    assess_threshold = ledger.assess_window(
        instrument_key="NSE_EQ|INE002A01018",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 6, 30),
        policy=valid_threshold_policy,
    )
    assert assess_threshold.complete is True
    assert len(assess_threshold.blocking_events) == 0


def test_fix6_unknown_announcement_date_fails_closed_in_tradable_information() -> None:
    ledger = PointInTimeCorporateActionLedger()
    ts = datetime(2026, 2, 1, tzinfo=UTC)
    ledger.add_coverage(
        CoverageScope(
            instrument_key="NSE_EQ|INE002A01018",
            isin="INE002A01018",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 6, 30),
            source="NSE_FEED",
            retrieval_timestamp=ts,
            is_complete=True,
        )
    )

    # Add a record with missing announcement_date
    rec = CorporateActionRecord(
        instrument_key="NSE_EQ|INE002A01018",
        isin="INE002A01018",
        event_type=CorporateActionEventType.SPLIT,
        effective_date=date(2026, 3, 1),
        announcement_date=None,  # Missing!
        source="NSE_FEED",
        retrieval_timestamp=ts,
    )
    ledger.add_record(rec)

    # 1. get_tradable_events with fail_on_unknown=True raises CorporateActionCoverageError
    with pytest.raises(CorporateActionCoverageError, match="has unknown announcement date"):
        ledger.get_tradable_events("NSE_EQ|INE002A01018", as_of_date=date(2026, 3, 15))

    # 2. verify_no_future_leakage raises CorporateActionCoverageError
    with pytest.raises(CorporateActionCoverageError, match="has unknown announcement date"):
        ledger.verify_no_future_leakage([rec], as_of_date=date(2026, 3, 15))

    # 3. assess_window in TRADABLE_INFORMATION mode fails closed with UNKNOWN_ANNOUNCEMENT_DATE blocking event
    assessment = ledger.assess_window(
        instrument_key="NSE_EQ|INE002A01018",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 6, 30),
        evaluation_mode=CorporateActionEvaluationMode.TRADABLE_INFORMATION,
        as_of_date=date(2026, 6, 30),
    )
    assert assessment.complete is False
    assert len(assessment.blocking_events) >= 1
    assert any("UNKNOWN_ANNOUNCEMENT_DATE" in b for b in assessment.blocking_events)

    # 4. EX_POST_NORMALIZATION mode uses effective_date and does not fail for unknown announcement date
    ex_post_assessment = ledger.assess_window(
        instrument_key="NSE_EQ|INE002A01018",
        window_start=date(2026, 1, 1),
        window_end=date(2026, 6, 30),
        evaluation_mode=CorporateActionEvaluationMode.EX_POST_NORMALIZATION,
    )
    # The split itself is an unhandled corporate action blocking event under default policy,
    # but NOT UNKNOWN_ANNOUNCEMENT_DATE
    assert not any("UNKNOWN_ANNOUNCEMENT_DATE" in b for b in ex_post_assessment.blocking_events)



