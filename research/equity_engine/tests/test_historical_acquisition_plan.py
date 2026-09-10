from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from equity_engine.historical_acquisition_plan import (
    HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION,
    SAFE_CHUNK_CALENDAR_DAYS,
    CapitalScenarioAnalysis,
    CorporateActionEvidence,
    EvidenceSourceIdentity,
    FormationBoundaryAdapter,
    HistoricalAcquisitionEvidenceError,
    PITMembershipSegment,
    SessionEvidence,
    build_example_historical_acquisition_plan,
    build_historical_acquisition_plan,
)
from equity_engine.nse_calendar import CalendarEvidence
from equity_engine.research_window_compiler import (
    FrozenTrainUniverse,
    derive_population_fingerprint,
)

KEY_A = "NSE_EQ|INE001A01010"
KEY_B = "NSE_EQ|INE002A01018"


def _digest(label: str) -> str:
    return (label.encode().hex() + "0" * 64)[:64]


def _sources() -> tuple[EvidenceSourceIdentity, ...]:
    return tuple(
        EvidenceSourceIdentity(
            source_id=source_id, evidence_type=source_id, fingerprint=_digest(source_id)
        )
        for source_id in ("calendar", "segments", "sessions", "corporate-actions")
    )


def _adapter(
    *,
    start: date,
    frozen_segments: tuple[PITMembershipSegment, ...],
    frozen_keys: tuple[str, ...] = (KEY_A,),
) -> FormationBoundaryAdapter:
    policy = "test:formation-frozen-v1"
    frozen = FrozenTrainUniverse(
        instruments=frozen_keys,
        universe_policy_id=policy,
        population_fingerprint=derive_population_fingerprint(
            instruments=frozen_keys,
            universe_policy_id=policy,
            pit_segments=frozen_segments,
        ),
        frozen_as_of=start,
    )
    return FormationBoundaryAdapter(
        formation_boundary=start,
        frozen_train_universe=frozen,
        research_window_plan_fingerprint=_digest("research-window-plan"),
    )


def _segment(
    instrument_key: str,
    start: date,
    end: date,
    *,
    eligible: bool = True,
    evidence_as_of: date | None = None,
) -> PITMembershipSegment:
    return PITMembershipSegment(
        instrument_key=instrument_key,
        valid_from=start,
        valid_to=end,
        evidence_as_of=evidence_as_of or start,
        source_fingerprint=_digest("segments"),
        eligible=eligible,
    )


def _inputs(
    segments: tuple[PITMembershipSegment, ...],
    days: tuple[date, ...],
    *,
    special_days: tuple[date, ...] = (),
    frozen_segments: tuple[PITMembershipSegment, ...] | None = None,
    frozen_keys: tuple[str, ...] = (KEY_A,),
):
    start = days[0]
    end = max(days[-1], special_days[-1] if special_days else days[-1])
    calendar = CalendarEvidence(
        trading_dates=days,
        holiday_dates=(),
        excluded_special_session_dates=special_days,
        source_urls=("calendar-source",),
    )
    sessions = tuple(
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
    ) + tuple(
        SessionEvidence(
            trade_date=day,
            session_kind="SPECIAL",
            source_id="sessions",
            timezone="Asia/Kolkata",
            start_time="00:00",
            end_time_exclusive="00:01",
            acquisition_allowed=False,
            exclusion_reason="special session requires a dedicated rule",
        )
        for day in special_days
    )
    eligible_keys = tuple(
        sorted(
            {
                segment.instrument_key
                for segment in segments
                if segment.eligible and any(segment.covers(day) for day in days)
            }
        )
    )
    ca = CorporateActionEvidence(
        source_id="corporate-actions",
        fingerprint=_digest("corporate-actions"),
        coverage_start=start,
        coverage_end=end,
        covered_instruments=eligible_keys,
        complete=True,
        policy_identity="test:raw-unadjusted-block-structural-actions-v1",
    )
    frozen_segments = frozen_segments or tuple(
        segment for segment in segments if segment.instrument_key in set(frozen_keys)
    )
    return {
        "research_start": start,
        "research_end": end,
        "calendar": calendar,
        "calendar_source_id": "calendar",
        "evidence_sources": _sources(),
        "pit_segments": segments,
        "formation_boundary_adapter": _adapter(
            start=start,
            frozen_segments=frozen_segments,
            frozen_keys=frozen_keys,
        ),
        "session_evidence": sessions,
        "corporate_action_evidence": ca,
        "requested_interval_minutes": (5,),
        "estimated_bytes_per_row": 100,
    }


def _build(segments: tuple[PITMembershipSegment, ...], days: tuple[date, ...]):
    return build_historical_acquisition_plan(**_inputs(segments, days))


def test_later_listed_instrument_is_acquired_but_not_frozen_for_wfo() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3))
    segments = (
        _segment(KEY_A, days[0], days[-1]),
        _segment(KEY_B, days[1], days[-1]),
    )
    plan = _build(segments, days)

    assert plan.historical_acquisition_superset == (KEY_A, KEY_B)
    assert plan.frozen_wfo_population == (KEY_A,)
    assert KEY_B in {item.instrument_key for item in plan.requested_intervals}
    assert KEY_B not in plan.formation_boundary_adapter.frozen_train_universe.instruments


def test_current_ineligibility_does_not_back_project_or_delete_history() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3))
    segments = (
        _segment(KEY_A, days[0], days[0]),
        _segment(KEY_A, days[1], days[1], eligible=False),
        _segment(KEY_A, days[2], days[2]),
    )
    plan = _build(segments, days)

    assert plan.historical_acquisition_superset == (KEY_A,)
    assert [item.trade_dates for item in plan.requested_intervals] == [
        (days[0],),
        (days[2],),
    ]
    assert any(
        item.reason_code == "DATED_MEMBERSHIP_NOT_ELIGIBLE" and item.trade_date == days[1]
        for item in plan.acquisition_exclusions
    )


def test_segment_future_evidence_fails_closed() -> None:
    with pytest.raises(ValueError, match="after segment start"):
        _segment(
            KEY_A,
            date(2024, 10, 1),
            date(2024, 10, 3),
            evidence_as_of=date(2024, 10, 2),
        )


def test_overlapping_segment_timeline_fails_closed() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3))
    segments = (
        _segment(KEY_A, days[0], days[1]),
        _segment(KEY_A, days[1], days[-1]),
    )
    with pytest.raises(HistoricalAcquisitionEvidenceError, match="overlapping PIT segments"):
        _build(segments, days)


def test_plan_identity_is_deterministic_and_binds_segment_evidence() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2))
    segments = (_segment(KEY_A, days[0], days[-1]),)
    first = _build(segments, days)
    second = _build(tuple(reversed(segments)), days)

    assert first.schema_version == HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION
    assert first.deterministic_fingerprint() == second.deterministic_fingerprint()
    assert first.to_json() == second.to_json()

    changed_segments = (replace(segments[0], source_fingerprint=_digest("changed-segments")),)
    changed_sources = list(_sources())
    changed_sources[1] = replace(changed_sources[1], fingerprint=_digest("changed-segments"))
    changed_inputs = _inputs(changed_segments, days)
    changed_inputs["evidence_sources"] = tuple(changed_sources)
    changed = build_historical_acquisition_plan(**changed_inputs)
    assert changed.deterministic_fingerprint() != first.deterministic_fingerprint()


def test_missing_segment_source_fails_closed() -> None:
    days = (date(2024, 10, 1), date(2024, 10, 2))
    unknown_source = _segment(KEY_A, days[0], days[-1], evidence_as_of=days[0])
    unknown_source = replace(unknown_source, source_fingerprint=_digest("not-supplied"))
    with pytest.raises(HistoricalAcquisitionEvidenceError, match="missing PIT segment source"):
        _build((unknown_source,), days)


def test_capital_scenarios_are_analysis_only_and_do_not_filter_history() -> None:
    days = (date(2024, 10, 1),)
    segments = (
        _segment(KEY_A, days[0], days[0]),
        _segment(KEY_B, days[0], days[0]),
    )
    inputs = _inputs(segments, days)
    inputs["capital_scenarios"] = (
        CapitalScenarioAnalysis(
            scenario_id="inr-1000",
            capital_rupees=Decimal(1000),
            note="analysis only",
        ),
    )
    plan = build_historical_acquisition_plan(**inputs)

    assert plan.historical_acquisition_superset == (KEY_A, KEY_B)
    assert plan.capital_scenarios[0].excluded_instruments == ()


def test_special_session_is_explicitly_excluded() -> None:
    normal = date(2024, 10, 1)
    special = date(2024, 10, 2)
    segments = (_segment(KEY_A, normal, special),)
    plan = build_historical_acquisition_plan(
        **_inputs(segments, (normal,), special_days=(special,))
    )

    assert special in plan.excluded_special_session_dates
    assert any(
        item.reason_code == "SPECIAL_SESSION_EXCLUDED" for item in plan.acquisition_exclusions
    )


def test_formation_adapter_requires_canonical_plan_type() -> None:
    with pytest.raises(TypeError, match="ResearchWindowPlan"):
        FormationBoundaryAdapter.from_research_window_plan(object())  # type: ignore[arg-type]


def test_example_plan_is_compact_dry_run_and_has_two_populations() -> None:
    plan = build_example_historical_acquisition_plan()
    second = build_example_historical_acquisition_plan()
    serialized = plan.to_json()

    assert plan.mode == "DRY_RUN"
    assert plan.research_start == date(2024, 10, 1)
    assert plan.research_end == date(2024, 10, 30)
    assert plan.live_orders_called is False
    assert plan.historical_acquisition_superset == (
        "NSE_EQ|INE000A01010",
        "NSE_EQ|INE001A01010",
    )
    assert plan.frozen_wfo_population == ("NSE_EQ|INE000A01010",)
    assert plan.historical_acquisition_superset != plan.frozen_wfo_population
    assert plan.safe_chunk_calendar_days == SAFE_CHUNK_CALENDAR_DAYS
    assert plan.missing_unknown_evidence == ()
    assert len(plan.pit_segments) == 4
    assert date(2024, 10, 2) in plan.excluded_special_session_dates
    assert any(item.request_count == 2 for item in plan.requested_intervals)
    assert any(
        item.instrument_key == "NSE_EQ|INE001A01010" and item.start == date(2024, 10, 3)
        for item in plan.requested_intervals
    )
    assert any(
        item.reason_code == "DATED_MEMBERSHIP_NOT_ELIGIBLE"
        and item.trade_date == date(2024, 10, 15)
        for item in plan.acquisition_exclusions
    )
    assert plan.pit_segments[1].evidence_as_of == date(2024, 10, 3)
    assert len(serialized.encode("utf-8")) < 20_000
    assert serialized == second.to_json()
    assert "PITMembershipEvidence" not in serialized
