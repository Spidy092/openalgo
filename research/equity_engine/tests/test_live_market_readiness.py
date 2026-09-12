"""Monday live-market readiness gate tests (synthetic, no network, no orders).

Covers the three readiness scopes: pure plumbing shadow (INFRA) must not
require prior paper evidence, research shadow must not require it either, and
only live-order review requires accumulated paper evidence. No hidden
defaults; no order capability.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from equity_engine.experiment import (
    ApprovedCapital,
    CostReconciliationEvidence,
    LiveOrderAttemptError,
    PaperTradingEvidence,
)
from equity_engine.historical_validation import (
    DailyIntradayValidation,
    HistoricalDatasetValidation,
)
from equity_engine.instrument_master import EquityInstrument
from equity_engine.live_market_readiness import (
    BROKER_BALANCE_UNKNOWN,
    BROKER_CONNECTIVITY_FAILED,
    CAPITAL_NOT_APPROVED,
    CAS_POLICY_UNKNOWN,
    CLOCK_INVALID,
    COST_RECONCILIATION_FAILED,
    COST_RECONCILIATION_MISSING,
    FEED_GAP_DETECTED,
    FEED_STATUS_UNKNOWN,
    FEED_UNAVAILABLE,
    HISTORICAL_EVIDENCE_MISSING,
    INSTRUMENT_NOT_TRADABLE,
    INSUFFICIENT_BALANCE,
    KILL_SWITCH_ENGAGED,
    LIVE_TRADABILITY_UNPROVEN,
    PAPER_EVIDENCE_MISSING,
    PIT_EVIDENCE_INCOMPLETE,
    QUOTE_MISSING,
    QUOTE_STALE,
    QUOTE_TIMESTAMP_INVALID,
    SESSION_BOUNDARY_VIOLATED,
    SESSION_CLOSED,
    STATIC_IP_MISSING,
    STRATEGY_EVIDENCE_MISSING,
    SUSPENSION_ACTIVE,
    SUSPENSION_CONFLICT,
    TICK_SIZE_UNVERIFIED,
    TOKEN_MISSING,
    FeedHealthEvidence,
    LiveMarketReadinessInputs,
    ReadinessClassification,
    evaluate_live_market_readiness,
)
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.nse_calendar import CalendarEvidence
from equity_engine.suspension_identity import NO_SUSPENSION_RECORD
from equity_engine.tick_size import TickSizeVerification
from equity_engine.upstox_market_context import QuoteBatchResult
from equity_engine.upstox_readiness import ReadinessCheck, UpstoxReadinessSnapshot

IST = ZoneInfo("Asia/Kolkata")
TRADE_DATE = date(2026, 9, 7)  # Monday, not an NSE CM holiday
NOW_IST = datetime(2026, 9, 7, 10, 0, 30, tzinfo=IST)
QUOTE_TS = datetime(2026, 9, 7, 10, 0, 0, tzinfo=IST)
INSTRUMENT_KEY = "NSE_EQ|INE002A01018"


def _snapshot(*, static_ip: bool = True) -> UpstoxReadinessSnapshot:
    checks = (
        ReadinessCheck(name="profile_api", passed=True, detail="ok"),
        ReadinessCheck(name="nse_enabled", passed=True, detail="ok"),
        ReadinessCheck(name="intraday_product_enabled", passed=True, detail="ok"),
        ReadinessCheck(name="funds_api", passed=True, detail="ok"),
        ReadinessCheck(name="minimum_test_capital_present", passed=True, detail="ok"),
        ReadinessCheck(
            name="primary_static_ip",
            passed=static_ip,
            detail="registered" if static_ip else "not registered",
        ),
    )
    return UpstoxReadinessSnapshot(
        checks=checks,
        available_to_trade=Decimal(15000),
        exchanges=("NSE", "BSE"),
        products=("D", "I"),
        primary_static_ip_configured=static_ip,
        secondary_static_ip_configured=False,
    )


def _calendar() -> CalendarEvidence:
    return CalendarEvidence(
        trading_dates=(TRADE_DATE,),
        holiday_dates=(),
        excluded_special_session_dates=(),
        source_urls=("https://nsearchives.nseindia.com/content/circulars/CMTR71775.pdf",),
    )


def _policy() -> NSEEquitySessionPolicy:
    return NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=15)


def _quotes(*, timestamp: datetime = QUOTE_TS) -> QuoteBatchResult:
    return QuoteBatchResult(
        requested_instrument_keys=(INSTRUMENT_KEY,),
        quotes={
            INSTRUMENT_KEY: {
                "instrument_token": INSTRUMENT_KEY,
                "last_price": "100",
                "timestamp": timestamp.isoformat(),
            }
        },
        failures={},
        request_count=1,
    )


def _instrument() -> EquityInstrument:
    return EquityInstrument(
        instrument_key=INSTRUMENT_KEY,
        isin="INE002A01018",
        trading_symbol="RELIANCE",
        name="RELIANCE INDUSTRIES",
        segment="NSE_EQ",
        exchange="NSE",
        instrument_type="EQ",
        security_type="NORMAL",
        lot_size=1,
        freeze_quantity=Decimal(100000),
        tick_size_raw=Decimal(5),
        tick_size_rupees=Decimal("0.05"),
        cas_eligible=False,
        mis_eligible=True,
        suspended=False,
        exchange_token="2885",
        suspension_status=NO_SUSPENSION_RECORD,
        suspension_variant_row_count=0,
        exact_token_match_count=0,
        live_tradability_proven=True,
    )


def _tick() -> TickSizeVerification:
    return TickSizeVerification(
        passed=True,
        expected_rupees=Decimal("0.05"),
        observed_rupees=Decimal("0.05"),
        source="https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf",
    )


def _cost() -> CostReconciliationEvidence:
    return CostReconciliationEvidence(
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


def _historical() -> HistoricalDatasetValidation:
    per_day = DailyIntradayValidation(
        trade_date=TRADE_DATE,
        row_count=2,
        first_timestamp="2026-09-07 09:15:00+05:30",
        last_timestamp="2026-09-07 09:20:00+05:30",
        duplicate_count=0,
        continuous_session_rows=2,
        cas_auxiliary_rows=0,
        cas_auxiliary_timestamps=(),
        missing_expected_slots=(),
        unexpected_timestamps=(),
        timezone="Asia/Kolkata",
        ohlcv_violations=(),
        session_rule={"rule_id": "synthetic-normal-session"},
    )
    return HistoricalDatasetValidation(
        rows=2,
        trading_dates=(TRADE_DATE,),
        per_day=(per_day,),
        structural_violations=(),
        deterministic_data_fingerprint="f" * 64,
        manifest_fingerprint_reference="f" * 64,
        fingerprint_schema="equity-market-data-v2",
        manifest_reference="synthetic-manifest",
        timezone="Asia/Kolkata",
        calendar_evidence=None,
    )


def _paper() -> PaperTradingEvidence:
    return PaperTradingEvidence(
        artifact_fingerprint="sha256_paper_trading_artifact_001",
        environment="upstox_sandbox_v2",
        session_start=date(2026, 8, 1),
        session_end=date(2026, 8, 29),
        verified_orders_count=45,
        audit_log_fingerprint="sha256_paper_audit_log_001",
        source_reference="broker_sandbox_order_log",
    )


def _ready_inputs() -> LiveMarketReadinessInputs:
    return LiveMarketReadinessInputs(
        token_present=True,
        readiness_snapshot=_snapshot(static_ip=True),
        now_ist=NOW_IST,
        timezone_name="Asia/Kolkata",
        trade_date=TRADE_DATE,
        is_trading_day=True,
        session_open=True,
        session_policy=_policy(),
        calendar_evidence=_calendar(),
        quotes=_quotes(),
        expected_instrument_keys=(INSTRUMENT_KEY,),
        max_quote_age_seconds=60.0,
        feed=FeedHealthEvidence(available=True, gap_detected=False, last_heartbeat_ist=NOW_IST),
        instrument=_instrument(),
        tick_verification=_tick(),
        approved_capital=ApprovedCapital(amount_rupees=Decimal(10000)),
        broker_available_to_trade=Decimal(15000),
        cost_evidence=_cost(),
        cost_tolerance_inr=Decimal("0.01"),
        pit_complete=True,
        historical_validation=_historical(),
        strategy_evidence_present=True,
        paper_evidence=_paper(),
        kill_switch_engaged=False,
        live_orders_called=False,
    )


def test_fully_synthetic_ready_for_live_order_review() -> None:
    report = evaluate_live_market_readiness(_ready_inputs())

    assert report.classification is ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW
    assert report.reason_codes == ()
    assert report.live_orders_called is False
    assert report.infra_ready is True
    assert report.research_shadow_ready is True
    assert report.shadow_ready is True
    assert report.live_review_ready is True
    # Broker balance cannot increase approved capital.
    assert report.approved_capital_rupees == Decimal(10000)
    assert report.broker_available_to_trade == Decimal(15000)
    assert report.effective_capital_rupees == Decimal(10000)


def test_first_ever_shadow_session_reaches_infra_without_paper_evidence() -> None:
    # The first live-market shadow session creates paper evidence, so prior
    # paper evidence — and research evidence — must not block plumbing.
    inputs = replace(
        _ready_inputs(),
        paper_evidence=None,
        pit_complete=False,
        historical_validation=None,
        strategy_evidence_present=False,
        cost_evidence=None,
    )

    report = evaluate_live_market_readiness(inputs)

    assert report.classification is ReadinessClassification.READY_FOR_SHADOW_INFRA
    assert PAPER_EVIDENCE_MISSING in report.reason_codes
    assert report.infra_ready is True
    assert report.research_shadow_ready is False
    assert report.live_review_ready is False
    assert report.live_orders_called is False


def test_research_evidence_reaches_research_shadow_without_paper_evidence() -> None:
    inputs = replace(_ready_inputs(), paper_evidence=None)

    report = evaluate_live_market_readiness(inputs)

    assert report.classification is ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    assert PAPER_EVIDENCE_MISSING in report.reason_codes
    assert report.infra_ready is True
    assert report.research_shadow_ready is True
    assert report.live_review_ready is False


def test_paper_evidence_missing_blocks_live_order_review() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), paper_evidence=None))

    assert report.classification is not ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW
    assert report.classification is ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    assert PAPER_EVIDENCE_MISSING in report.reason_codes
    assert report.live_review_ready is False


def test_static_ip_missing_blocks_live_review_but_not_research_shadow() -> None:
    inputs = replace(_ready_inputs(), readiness_snapshot=_snapshot(static_ip=False))

    report = evaluate_live_market_readiness(inputs)

    assert report.classification is ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    assert STATIC_IP_MISSING in report.reason_codes
    assert report.infra_ready is True
    assert report.research_shadow_ready is True
    assert report.live_review_ready is False


def test_missing_token_fails_closed() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), token_present=False))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert TOKEN_MISSING in report.reason_codes


def test_broker_connectivity_failure_fails_closed() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), readiness_snapshot=None))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert BROKER_CONNECTIVITY_FAILED in report.reason_codes


def test_closed_market_fails_closed() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), session_open=False))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert SESSION_CLOSED in report.reason_codes


def test_non_trading_day_fails_closed() -> None:
    report = evaluate_live_market_readiness(
        replace(_ready_inputs(), is_trading_day=False, session_open=False)
    )

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert SESSION_CLOSED in report.reason_codes


def test_clock_invalid_fails_closed() -> None:
    naive = NOW_IST.replace(tzinfo=None)
    report = evaluate_live_market_readiness(replace(_ready_inputs(), now_ist=naive))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert CLOCK_INVALID in report.reason_codes


def test_stale_quote_fails_closed() -> None:
    stale_ts = datetime(2026, 9, 7, 9, 0, 0, tzinfo=IST)
    report = evaluate_live_market_readiness(
        replace(_ready_inputs(), quotes=_quotes(timestamp=stale_ts))
    )

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert QUOTE_STALE in report.reason_codes


def test_stale_quote_blocks_every_shadow_mode() -> None:
    stale_ts = datetime(2026, 9, 7, 9, 0, 0, tzinfo=IST)
    # Even a research-level session (paper missing, all else ready) is still
    # blocked at infra scope by a stale quote.
    inputs = replace(_ready_inputs(), paper_evidence=None, quotes=_quotes(timestamp=stale_ts))

    report = evaluate_live_market_readiness(inputs)

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert QUOTE_STALE in report.reason_codes
    assert report.infra_ready is False


def test_missing_quote_fails_closed() -> None:
    empty = QuoteBatchResult(
        requested_instrument_keys=(INSTRUMENT_KEY,),
        quotes={},
        failures={INSTRUMENT_KEY: "missing_quote"},
        request_count=1,
    )
    report = evaluate_live_market_readiness(replace(_ready_inputs(), quotes=empty))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert QUOTE_MISSING in report.reason_codes


def test_invalid_quote_timestamp_fails_closed() -> None:
    bad = QuoteBatchResult(
        requested_instrument_keys=(INSTRUMENT_KEY,),
        quotes={INSTRUMENT_KEY: {"instrument_token": INSTRUMENT_KEY, "last_price": "100"}},
        failures={},
        request_count=1,
    )
    report = evaluate_live_market_readiness(replace(_ready_inputs(), quotes=bad))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert QUOTE_TIMESTAMP_INVALID in report.reason_codes


def test_feed_unknown_fails_closed() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), feed=None))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert FEED_STATUS_UNKNOWN in report.reason_codes


def test_feed_unavailable_fails_closed() -> None:
    feed = FeedHealthEvidence(available=False, gap_detected=False, last_heartbeat_ist=NOW_IST)
    report = evaluate_live_market_readiness(replace(_ready_inputs(), feed=feed))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert FEED_UNAVAILABLE in report.reason_codes


def test_feed_gap_fails_closed() -> None:
    feed = FeedHealthEvidence(available=True, gap_detected=True, last_heartbeat_ist=NOW_IST)
    report = evaluate_live_market_readiness(replace(_ready_inputs(), feed=feed))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert FEED_GAP_DETECTED in report.reason_codes


def test_cas_unknown_fails_closed() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), session_policy=None))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert CAS_POLICY_UNKNOWN in report.reason_codes


def test_cas_unknown_blocks_every_shadow_mode() -> None:
    # A research-level session is still blocked at infra scope without CAS policy.
    inputs = replace(_ready_inputs(), paper_evidence=None, session_policy=None)

    report = evaluate_live_market_readiness(inputs)

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert CAS_POLICY_UNKNOWN in report.reason_codes
    assert report.infra_ready is False


def test_cas_mismatch_fails_closed() -> None:
    mismatched_policy = NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=15)
    report = evaluate_live_market_readiness(
        replace(_ready_inputs(), session_policy=mismatched_policy)
    )

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert CAS_POLICY_UNKNOWN in report.reason_codes


def test_session_boundary_violated_fails_closed() -> None:
    late = datetime(2026, 9, 7, 15, 20, 0, tzinfo=IST)
    late_quote = _quotes(timestamp=datetime(2026, 9, 7, 15, 19, 30, tzinfo=IST))
    inputs = replace(_ready_inputs(), now_ist=late, quotes=late_quote)
    report = evaluate_live_market_readiness(inputs)

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert SESSION_BOUNDARY_VIOLATED in report.reason_codes


def test_suspended_instrument_fails_closed() -> None:
    suspended = replace(_instrument(), suspended=True, suspension_status="SUSPENDED_EXACT")
    report = evaluate_live_market_readiness(replace(_ready_inputs(), instrument=suspended))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert SUSPENSION_ACTIVE in report.reason_codes


def test_suspension_conflict_fails_closed() -> None:
    conflicted = replace(_instrument(), suspended=False, suspension_status="SUSPENSION_CONFLICT")
    report = evaluate_live_market_readiness(replace(_ready_inputs(), instrument=conflicted))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert SUSPENSION_CONFLICT in report.reason_codes


def test_non_mis_instrument_fails_closed() -> None:
    non_mis = replace(_instrument(), mis_eligible=False, live_tradability_proven=False)
    report = evaluate_live_market_readiness(replace(_ready_inputs(), instrument=non_mis))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert INSTRUMENT_NOT_TRADABLE in report.reason_codes


def test_unproven_live_tradability_blocks_live_review_only() -> None:
    unproven = replace(_instrument(), live_tradability_proven=False)
    report = evaluate_live_market_readiness(replace(_ready_inputs(), instrument=unproven))

    assert report.classification is ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    assert LIVE_TRADABILITY_UNPROVEN in report.reason_codes
    assert report.research_shadow_ready is True
    assert report.live_review_ready is False


def test_tick_unverified_fails_closed() -> None:
    bad_tick = TickSizeVerification(
        passed=False,
        expected_rupees=Decimal("0.05"),
        observed_rupees=Decimal("0.01"),
        source="https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf",
    )
    report = evaluate_live_market_readiness(replace(_ready_inputs(), tick_verification=bad_tick))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert TICK_SIZE_UNVERIFIED in report.reason_codes


def test_capital_absent_fails_closed() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), approved_capital=None))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert CAPITAL_NOT_APPROVED in report.reason_codes


def test_broker_balance_unknown_blocks_live_review_only() -> None:
    report = evaluate_live_market_readiness(
        replace(_ready_inputs(), broker_available_to_trade=None)
    )

    assert report.classification is ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    assert BROKER_BALANCE_UNKNOWN in report.reason_codes
    assert report.research_shadow_ready is True
    assert report.live_review_ready is False


def test_broker_balance_cannot_increase_approved_capital() -> None:
    # Balance above approved: effective is capped at approved, live review allowed.
    high = evaluate_live_market_readiness(_ready_inputs())
    assert high.effective_capital_rupees == Decimal(10000)

    # Balance below approved: effective follows the broker balance and live
    # review is blocked, while the cap invariant still holds.
    low = evaluate_live_market_readiness(
        replace(_ready_inputs(), broker_available_to_trade=Decimal(4000))
    )
    assert low.effective_capital_rupees == Decimal(4000)
    assert low.effective_capital_rupees <= Decimal(10000)
    assert low.classification is ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    assert INSUFFICIENT_BALANCE in low.reason_codes


def test_missing_cost_evidence_blocks_research_but_not_infra() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), cost_evidence=None))

    assert report.classification is ReadinessClassification.READY_FOR_SHADOW_INFRA
    assert COST_RECONCILIATION_MISSING in report.reason_codes
    assert report.infra_ready is True
    assert report.research_shadow_ready is False


def test_failed_cost_reconciliation_blocks_research_but_not_infra() -> None:
    # CostReconciliationEvidence itself fails closed at construction for
    # non-PASS status, so a FAIL artifact can never be built. Exercise the
    # gate's FAILED branch with a duck-typed artifact carrying FAIL status.
    cost = _cost()

    class FailedCost:
        status = "FAIL"
        max_reconciliation_error_inr = cost.max_reconciliation_error_inr

    report = evaluate_live_market_readiness(
        replace(_ready_inputs(), cost_evidence=FailedCost())  # type: ignore[arg-type]
    )

    assert report.classification is ReadinessClassification.READY_FOR_SHADOW_INFRA
    assert COST_RECONCILIATION_FAILED in report.reason_codes


def test_non_pass_cost_evidence_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="must be PASS"):
        replace(_cost(), status="FAIL")


def test_cost_tolerance_exceeded_blocks_research_but_not_infra() -> None:
    report = evaluate_live_market_readiness(
        replace(_ready_inputs(), cost_tolerance_inr=Decimal("0.001"))
    )

    assert report.classification is ReadinessClassification.READY_FOR_SHADOW_INFRA
    assert COST_RECONCILIATION_FAILED in report.reason_codes


def test_pit_incomplete_blocks_research_but_not_infra() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), pit_complete=False))

    assert report.classification is ReadinessClassification.READY_FOR_SHADOW_INFRA
    assert PIT_EVIDENCE_INCOMPLETE in report.reason_codes
    assert report.infra_ready is True
    assert report.research_shadow_ready is False


def test_historical_missing_blocks_research_but_not_infra() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), historical_validation=None))

    assert report.classification is ReadinessClassification.READY_FOR_SHADOW_INFRA
    assert HISTORICAL_EVIDENCE_MISSING in report.reason_codes
    assert report.infra_ready is True
    assert report.research_shadow_ready is False


def test_strategy_evidence_missing_blocks_research_but_not_infra() -> None:
    report = evaluate_live_market_readiness(
        replace(_ready_inputs(), strategy_evidence_present=False)
    )

    assert report.classification is ReadinessClassification.READY_FOR_SHADOW_INFRA
    assert STRATEGY_EVIDENCE_MISSING in report.reason_codes
    assert report.infra_ready is True
    assert report.research_shadow_ready is False


def test_missing_paper_evidence_allows_research_shadow() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), paper_evidence=None))

    assert report.classification is ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    assert PAPER_EVIDENCE_MISSING in report.reason_codes


def test_kill_switch_fails_closed() -> None:
    report = evaluate_live_market_readiness(replace(_ready_inputs(), kill_switch_engaged=True))

    assert report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW
    assert KILL_SWITCH_ENGAGED in report.reason_codes


def test_attempted_live_orders_called_is_rejected() -> None:
    with pytest.raises(LiveOrderAttemptError, match="strictly forbidden"):
        evaluate_live_market_readiness(replace(_ready_inputs(), live_orders_called=True))


def test_report_is_credential_free_and_machine_readable() -> None:
    report = evaluate_live_market_readiness(_ready_inputs())

    payload = report.to_dict()
    assert payload["live_orders_called"] is False
    assert payload["classification"] == "READY_FOR_LIVE_ORDER_REVIEW"
    serialized = report.to_json()
    assert "PRESENT" not in serialized or "token" not in serialized.lower().replace(
        "instrument_token", ""
    )
    # PRESENT/ABSENT only: no token material can appear because inputs cannot carry it.
    assert "secret-token" not in serialized
    assert "Bearer" not in serialized
    assert payload["reason_codes"] == []
    assert payload["infra_ready"] is True
    assert payload["research_shadow_ready"] is True
    # No hidden defaults: thresholds are explicit caller inputs.
    inputs = _ready_inputs()
    assert inputs.max_quote_age_seconds == 60.0
    assert inputs.cost_tolerance_inr == Decimal("0.01")


def test_readiness_module_has_no_order_api_dependency() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "equity_engine" / "live_market_readiness.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    for forbidden in (
        "place_order",
        "modify_order",
        "cancel_order",
        "placeorder",
        "cancelorder",
        "access_token",
        "bearer",
        "api_key",
        "apikey",
    ):
        assert forbidden not in lowered, f"forbidden dependency {forbidden!r} in readiness module"
    assert "import httpx" not in source
    assert "from broker" not in source
    assert "services.place_order_service" not in source
