from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from equity_engine.historical_acquisition_plan import (
    CapitalScenarioAnalysis,
    CorporateActionEvidence,
    EvidenceSourceIdentity,
    HistoricalAcquisitionEvidenceError,
    PITMembershipCoverage,
    PITMembershipEvidence,
    SessionEvidence,
    build_example_historical_acquisition_plan,
    build_historical_acquisition_plan,
)
from equity_engine.nse_calendar import CalendarEvidence

KEY_A = "NSE_EQ|INE001A01010"
KEY_B = "NSE_EQ|INE002A01018"


def _digest(seed: str) -> str:
    return (seed.encode().hex() + "0" * 64)[:64]


def _sources() -> tuple[EvidenceSourceIdentity, ...]:
    return tuple(
        EvidenceSourceIdentity(source_id=source_id, evidence_type=source_id, fingerprint=_digest(source_id))
        for source_id in (
            "calendar",
            "membership",
            "identity",
            "status",
            "tick",
            "sessions",
            "corporate-actions",
        )
    )


def _record(
    trade_date: date,
    instrument_key: str = KEY_A,
    *,
    tradeable: bool = True,
    symbol: str = "ALPHA",
    evidence_as_of: date | None = None,
) -> PITMembershipEvidence:
    isin = instrument_key.split("|", 1)[1]
    return PITMembershipEvidence(
        trade_date=trade_date,
        instrument_key=instrument_key,
        isin=isin,
        symbol=symbol,
        name=f"{symbol} LIMITED",
        series="EQ",
        listed_on_nse=True,
        normal_equity=True,
        tradeable_in_normal_market=tradeable,
        tick_size_rupees=Decimal("0.05"),
        evidence_as_of=evidence_as_of or trade_date,
        membership_source_id="membership",
        identity_source_id="identity",
        status_source_id="status",
        tick_source_id="tick",
    )


def _inputs(
    records: tuple[PITMembershipEvidence, ...],
    days: tuple[date, ...],
    *,
    special_days: tuple[date, ...] = (),
    source_overrides: tuple[EvidenceSourceIdentity, ...] = (),
):
    calendar = CalendarEvidence(
        trading_dates=days,
        holiday_dates=(),
        excluded_special_session_dates=special_days,
        source_urls=("https://example.invalid/calendar",),
    )
    counts = {day: sum(record.trade_date == day for record in records) for day in days}
    coverage = tuple(
        PITMembershipCoverage(
            trade_date=day,
            source_id="membership",
            record_count=counts[day],
        )
        for day in days
    )
    sessions = [
        SessionEvidence(
            trade_date=day,
            session_kind="NORMAL",
            source_id="sessions",
            timezone="Asia/Kolkata",
            start_time="09:15",
            end_time_exclusive="09:25",
            expected_rows_by_interval=((5, 2),),
        )
        for day in days
    ]
    sessions.extend(
        SessionEvidence(
            trade_date=day,
            session_kind="SPECIAL",
            source_id="sessions",
            timezone="Asia/Kolkata",
            start_time="00:00",
            end_time_exclusive="00:01",
            acquisition_allowed=False,
            exclusion_reason="special session requires explicit rules",
        )
        for day in special_days
    )
    eligible_keys = tuple(
        sorted(
            {
                record.instrument_key
                for record in records
                if record.trade_date in set(days) and record.eligible
            }
        )
    )
    ca = CorporateActionEvidence(
        source_id="corporate-actions",
        fingerprint=_digest("ca"),
        coverage_start=days[0],
        coverage_end=days[-1],
        covered_instruments=eligible_keys,
        complete=True,
        policy_identity="raw-unadjusted-block-structural-actions/v1",
    )
    return {
        "research_start": days[0],
        "research_end": days[-1],
        "calendar": calendar,
        "calendar_source_id": "calendar",
        "evidence_sources": source_overrides or _sources(),
        "membership_coverage": coverage,
        "pit_membership_evidence": records,
        "session_evidence": tuple(sessions),
        "corporate_action_evidence": ca,
        "requested_interval_minutes": (5,),
        "estimated_bytes_per_row": 100,
    }


def _build(records: tuple[PITMembershipEvidence, ...], days: tuple[date, ...]):
    return build_historical_acquisition_plan(**_inputs(records, days))


def test_survivorship_is_closed_by_union_of_historical_membership() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3))
    plan = _build((_record(days[0], KEY_A), _record(days[-1], KEY_B, symbol="BETA")), days)

    assert tuple(item.instrument_key for item in plan.instrument_identity_union) == (KEY_A, KEY_B)
    assert {item.instrument_key for item in plan.requested_intervals} == {KEY_A, KEY_B}
    assert plan.instrument_identity_union[0].last_required_date == days[0]


def test_future_membership_does_not_leak_into_research_window() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2))
    future_record = _record(date(2024, 10, 3), KEY_B, symbol="FUTURE")
    plan = _build((_record(days[0], KEY_A), future_record), days)

    assert tuple(item.instrument_key for item in plan.instrument_identity_union) == (KEY_A,)
    assert all(item.instrument_key != KEY_B for item in plan.requested_intervals)
    assert future_record.trade_date not in {item.trade_date for item in plan.pit_membership_evidence}


def test_current_suspension_cannot_back_project_over_historical_eligibility() -> None:
    historical_day = date(2024, 10, 1)
    current_snapshot_day = date(2024, 10, 2)
    plan = _build(
        (
            _record(historical_day, KEY_A, tradeable=True),
            _record(current_snapshot_day, KEY_A, tradeable=False),
        ),
        (historical_day, current_snapshot_day),
    )

    assert plan.instrument_identity_union[0].required_trade_dates == (historical_day,)
    assert plan.requested_intervals[0].trade_dates == (historical_day,)
    assert any(
        item.reason_code == "DATED_MEMBERSHIP_NOT_ELIGIBLE"
        and item.trade_date == current_snapshot_day
        for item in plan.acquisition_exclusions
    )


def test_changing_memberships_are_unioned_without_current_population_filter() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3))
    records = (
        _record(days[0], KEY_A),
        _record(days[1], KEY_B, symbol="BETA"),
        _record(days[2], KEY_A),
    )
    plan = _build(records, days)

    assert {item.instrument_key for item in plan.instrument_identity_union} == {KEY_A, KEY_B}
    assert [item.trade_dates for item in plan.requested_intervals if item.instrument_key == KEY_A] == [
        (days[0],),
        (days[2],),
    ]
    assert [item.trade_dates for item in plan.requested_intervals if item.instrument_key == KEY_B] == [
        (days[1],)
    ]


def test_plan_identity_is_deterministic_and_binds_evidence() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2))
    records = (_record(days[0], KEY_A), _record(days[1], KEY_B, symbol="BETA"))
    first = _build(records, days)
    second = _build(tuple(reversed(records)), days)

    assert first.deterministic_fingerprint() == second.deterministic_fingerprint()
    assert first.to_json() == second.to_json()

    changed_sources = list(_sources())
    changed_sources[1] = replace(changed_sources[1], fingerprint=_digest("changed-membership"))
    changed = build_historical_acquisition_plan(
        **_inputs(records, days, source_overrides=tuple(changed_sources))
    )
    assert changed.deterministic_fingerprint() != first.deterministic_fingerprint()


def test_unknown_evidence_fails_closed() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2))
    inputs = _inputs((_record(days[0], KEY_A),), days)
    inputs["membership_coverage"] = inputs["membership_coverage"][:1]

    with pytest.raises(HistoricalAcquisitionEvidenceError, match="missing membership coverage"):
        build_historical_acquisition_plan(**inputs)


def test_capital_scenarios_are_analysis_only_and_do_not_filter_union() -> None:
    days = (date(2024, 10, 1),)
    inputs = _inputs((_record(days[0], KEY_A), _record(days[0], KEY_B, symbol="BETA")), days)
    inputs["capital_scenarios"] = (
        CapitalScenarioAnalysis(
            scenario_id="inr-1000",
            capital_rupees=Decimal(1000),
            note="analysis only",
        ),
    )
    plan = build_historical_acquisition_plan(**inputs)

    assert {item.instrument_key for item in plan.instrument_identity_union} == {KEY_A, KEY_B}
    assert plan.capital_scenarios[0].excluded_instruments == ()


def test_special_session_is_explicitly_excluded() -> None:
    day = date(2024, 10, 1)
    special = date(2024, 10, 2)
    inputs = _inputs((_record(day, KEY_A),), (day,), special_days=(special,))
    inputs["research_end"] = special
    inputs["corporate_action_evidence"] = replace(
        inputs["corporate_action_evidence"], coverage_end=special
    )
    plan = build_historical_acquisition_plan(**inputs)

    assert special in plan.excluded_special_session_dates
    assert any(item.reason_code == "SPECIAL_SESSION_EXCLUDED" for item in plan.acquisition_exclusions)


def test_example_plan_is_dry_run_and_contains_requested_window() -> None:
    plan = build_example_historical_acquisition_plan()

    assert plan.mode == "DRY_RUN"
    assert plan.research_start == date(2024, 10, 1)
    assert plan.research_end == date(2026, 9, 8)
    assert plan.live_orders_called is False
    assert {item.instrument_key for item in plan.instrument_identity_union} == {
        "NSE_EQ|INE000A01010",
        "NSE_EQ|INE001A01010",
        "NSE_EQ|INE002A01018",
    }
    assert plan.missing_unknown_evidence == ()
