"""Deterministic DRY-RUN planning for historical market-data acquisition.

This module is downstream of the research-window compiler. It consumes the
compiler's canonical :class:`PITMembershipSegment` records and never defines a
second point-in-time membership truth model. It plans a historical acquisition
*superset* without changing the compiler's formation-boundary frozen WFO
population. No broker/API calls, downloads, live orders, or data writes occur.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from itertools import pairwise
from typing import Any

from .nse_calendar import CalendarEvidence
from .research_window_compiler import (
    FrozenTrainUniverse,
    PITMembershipSegment,
    ResearchWindowPlan,
    resolve_pit_segment,
)

HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION = "historical-acquisition-plan/v2"
SAFE_CHUNK_CALENDAR_DAYS = 28
_DIGEST_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _date_tuple(values: Iterable[date], *, name: str) -> tuple[date, ...]:
    result = tuple(values)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be sorted unique dates")
    return result


class HistoricalAcquisitionEvidenceError(ValueError):
    """Raised when a dry-run cannot prove its evidence inputs completely."""

    def __init__(self, reasons: Iterable[str]) -> None:
        self.reasons = tuple(str(reason) for reason in reasons if str(reason).strip())
        detail = "; ".join(self.reasons) if self.reasons else "unknown evidence"
        super().__init__(f"historical acquisition evidence is incomplete: {detail}")


@dataclass(frozen=True)
class EvidenceSourceIdentity:
    """Identity for evidence already supplied to the planner."""

    source_id: str
    evidence_type: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.evidence_type.strip():
            raise ValueError("evidence source identity and type are required")
        _require_digest("evidence source fingerprint", self.fingerprint)

    def as_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "evidence_type": self.evidence_type,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class FormationBoundaryAdapter:
    """Lossless adapter from the compiler's formation boundary to acquisition planning."""

    formation_boundary: date
    frozen_train_universe: FrozenTrainUniverse
    research_window_plan_fingerprint: str

    def __post_init__(self) -> None:
        if self.frozen_train_universe.frozen_as_of != self.formation_boundary:
            raise HistoricalAcquisitionEvidenceError(
                ["frozen WFO population is not formed at the acquisition boundary"]
            )
        _require_digest("research-window plan fingerprint", self.research_window_plan_fingerprint)

    @classmethod
    def from_research_window_plan(cls, plan: ResearchWindowPlan) -> FormationBoundaryAdapter:
        """Adapt a canonical compiled plan without copying or recomputing its population."""

        if not isinstance(plan, ResearchWindowPlan):
            raise TypeError("plan must be a ResearchWindowPlan")
        return cls(
            formation_boundary=plan.research_start,
            frozen_train_universe=plan.frozen_train_universe,
            research_window_plan_fingerprint=plan.fingerprint(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "formation_boundary": self.formation_boundary.isoformat(),
            "frozen_train_universe": self.frozen_train_universe.as_dict(),
            "research_window_plan_fingerprint": self.research_window_plan_fingerprint,
        }


@dataclass(frozen=True)
class SessionEvidence:
    """Explicit normal or special-session evidence for one calendar date."""

    trade_date: date
    session_kind: str
    source_id: str
    timezone: str
    start_time: str
    end_time_exclusive: str
    expected_rows_by_interval: tuple[tuple[int, int], ...] = ()
    acquisition_allowed: bool = True
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if self.session_kind not in {"NORMAL", "SPECIAL"}:
            raise ValueError("session_kind must be NORMAL or SPECIAL")
        for name, value in (
            ("source_id", self.source_id),
            ("timezone", self.timezone),
            ("start_time", self.start_time),
            ("end_time_exclusive", self.end_time_exclusive),
        ):
            if not value.strip():
                raise ValueError(f"session evidence {name} is required")
        if self.start_time >= self.end_time_exclusive:
            raise ValueError("session evidence start_time must precede end_time_exclusive")
        if not isinstance(self.acquisition_allowed, bool):
            raise TypeError("session evidence acquisition_allowed must be boolean")
        pairs = tuple(self.expected_rows_by_interval)
        if tuple(sorted(pairs)) != pairs or len({interval for interval, _ in pairs}) != len(pairs):
            raise ValueError("expected_rows_by_interval must be sorted unique intervals")
        for interval, rows in pairs:
            if (
                not isinstance(interval, int)
                or isinstance(interval, bool)
                or not 1 <= interval <= 15
            ):
                raise ValueError("session interval must be an integer from 1 through 15")
            if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
                raise ValueError("expected session rows must be a positive integer")
        if self.acquisition_allowed and self.exclusion_reason is not None:
            raise ValueError("allowed session cannot carry an exclusion reason")
        if not self.acquisition_allowed and not (self.exclusion_reason or "").strip():
            raise ValueError("excluded session requires an exclusion reason")

    def expected_rows(self, interval_minutes: int) -> int:
        for interval, rows in self.expected_rows_by_interval:
            if interval == interval_minutes:
                return rows
        raise HistoricalAcquisitionEvidenceError(
            [
                (
                    f"session {self.trade_date.isoformat()} has no expected row evidence for "
                    f"{interval_minutes}m"
                )
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "session_kind": self.session_kind,
            "source_id": self.source_id,
            "timezone": self.timezone,
            "start_time": self.start_time,
            "end_time_exclusive": self.end_time_exclusive,
            "expected_rows_by_interval": [list(item) for item in self.expected_rows_by_interval],
            "acquisition_allowed": self.acquisition_allowed,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class CorporateActionEvidence:
    """Already-resolved corporate-action evidence for the acquisition superset."""

    source_id: str
    fingerprint: str
    coverage_start: date
    coverage_end: date
    covered_instruments: tuple[str, ...]
    complete: bool
    unknown_instruments: tuple[str, ...] = ()
    blocking_events: tuple[str, ...] = ()
    policy_identity: str = ""
    events_count: int = 0

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.policy_identity.strip():
            raise ValueError("corporate-action source and policy identity are required")
        _require_digest("corporate-action evidence fingerprint", self.fingerprint)
        if self.coverage_start > self.coverage_end:
            raise ValueError("corporate-action coverage dates are invalid")
        if tuple(sorted(set(self.covered_instruments))) != self.covered_instruments:
            raise ValueError("corporate-action covered_instruments must be sorted unique")
        if tuple(sorted(set(self.unknown_instruments))) != self.unknown_instruments:
            raise ValueError("corporate-action unknown_instruments must be sorted unique")
        if not isinstance(self.complete, bool):
            raise TypeError("corporate-action completeness must be boolean")
        if self.events_count < 0:
            raise ValueError("corporate-action events_count cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "fingerprint": self.fingerprint,
            "coverage_start": self.coverage_start.isoformat(),
            "coverage_end": self.coverage_end.isoformat(),
            "covered_instruments": list(self.covered_instruments),
            "complete": self.complete,
            "unknown_instruments": list(self.unknown_instruments),
            "blocking_events": list(self.blocking_events),
            "policy_identity": self.policy_identity,
            "events_count": self.events_count,
        }


@dataclass(frozen=True)
class InstrumentAcquisitionUnion:
    """Compact summary of dates acquired for one member of the historical superset."""

    instrument_key: str
    eligible_trade_dates: tuple[date, ...]
    source_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.instrument_key.strip():
            raise ValueError("instrument_key is required")
        if not self.eligible_trade_dates:
            raise ValueError("eligible_trade_dates must not be empty")
        if tuple(sorted(set(self.eligible_trade_dates))) != self.eligible_trade_dates:
            raise ValueError("eligible_trade_dates must be sorted unique")
        if tuple(sorted(set(self.source_fingerprints))) != self.source_fingerprints:
            raise ValueError("source_fingerprints must be sorted unique")
        for fingerprint in self.source_fingerprints:
            _require_digest("segment source fingerprint", fingerprint)

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument_key": self.instrument_key,
            "eligible_trade_dates": [item.isoformat() for item in self.eligible_trade_dates],
            "source_fingerprints": list(self.source_fingerprints),
        }


@dataclass(frozen=True)
class RequestedAcquisitionInterval:
    instrument_key: str
    interval_minutes: int
    start: date
    end: date
    trade_dates: tuple[date, ...]
    expected_rows: int
    estimated_storage_bytes: int
    request_count: int

    def __post_init__(self) -> None:
        if not self.instrument_key.strip():
            raise ValueError("requested interval instrument_key is required")
        if not isinstance(self.interval_minutes, int) or not 1 <= self.interval_minutes <= 15:
            raise ValueError("requested interval interval_minutes must be 1 through 15")
        if self.start > self.end or not self.trade_dates:
            raise ValueError("requested interval dates are invalid")
        if tuple(sorted(set(self.trade_dates))) != self.trade_dates:
            raise ValueError("requested interval trade_dates must be sorted unique")
        if self.trade_dates[0] != self.start or self.trade_dates[-1] != self.end:
            raise ValueError("requested interval bounds must match trade_dates")
        if self.expected_rows <= 0 or self.estimated_storage_bytes <= 0 or self.request_count <= 0:
            raise ValueError("requested interval estimates must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument_key": self.instrument_key,
            "interval_minutes": self.interval_minutes,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "trade_dates": [item.isoformat() for item in self.trade_dates],
            "expected_rows": self.expected_rows,
            "estimated_storage_bytes": self.estimated_storage_bytes,
            "request_count": self.request_count,
        }


@dataclass(frozen=True)
class AcquisitionExclusion:
    trade_date: date | None
    instrument_key: str | None
    reason_code: str
    detail: str

    def __post_init__(self) -> None:
        if not self.reason_code.strip() or not self.detail.strip():
            raise ValueError("acquisition exclusion reason and detail are required")
        if self.instrument_key is not None and not self.instrument_key.strip():
            raise ValueError("acquisition exclusion instrument_key cannot be blank")

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date.isoformat() if self.trade_date else None,
            "instrument_key": self.instrument_key,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CapitalScenarioAnalysis:
    """Capital commentary that is forbidden from filtering historical acquisition."""

    scenario_id: str
    capital_rupees: Decimal
    note: str
    analysis_only: bool = True
    excluded_instruments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.note.strip():
            raise ValueError("capital scenario identity and note are required")
        if self.capital_rupees <= 0:
            raise ValueError("capital scenario capital must be positive")
        if self.analysis_only is not True:
            raise ValueError("capital scenarios must remain analysis_only")
        if self.excluded_instruments:
            raise ValueError("capital scenarios cannot exclude historical instruments")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "capital_rupees": str(self.capital_rupees),
            "note": self.note,
            "analysis_only": True,
            "excluded_instruments": list(self.excluded_instruments),
        }


@dataclass(frozen=True)
class HistoricalAcquisitionPlan:
    """Immutable DRY-RUN artifact with separate superset and frozen WFO populations."""

    schema_version: str
    mode: str
    research_start: date
    research_end: date
    calendar_source_id: str
    normal_trading_dates: tuple[date, ...]
    holiday_dates: tuple[date, ...]
    excluded_special_session_dates: tuple[date, ...]
    evidence_sources: tuple[EvidenceSourceIdentity, ...]
    pit_segments: tuple[PITMembershipSegment, ...]
    formation_boundary_adapter: FormationBoundaryAdapter
    historical_acquisition_superset: tuple[str, ...]
    frozen_wfo_population: tuple[str, ...]
    session_evidence: tuple[SessionEvidence, ...]
    corporate_action_evidence: CorporateActionEvidence
    instrument_acquisition_union: tuple[InstrumentAcquisitionUnion, ...]
    requested_intervals: tuple[RequestedAcquisitionInterval, ...]
    requested_interval_minutes: tuple[int, ...]
    safe_chunk_calendar_days: int
    expected_request_count: int
    estimated_rows: int
    estimated_storage_bytes: int
    estimated_bytes_per_row: int
    missing_unknown_evidence: tuple[str, ...] = ()
    acquisition_exclusions: tuple[AcquisitionExclusion, ...] = ()
    capital_scenarios: tuple[CapitalScenarioAnalysis, ...] = ()
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported historical acquisition plan schema {self.schema_version!r}"
            )
        if self.mode != "DRY_RUN":
            raise ValueError("historical acquisition plan mode must be DRY_RUN")
        if self.research_start > self.research_end:
            raise ValueError("research window dates are invalid")
        if not self.calendar_source_id.strip():
            raise ValueError("calendar_source_id is required")
        if self.live_orders_called:
            raise ValueError("live orders are forbidden in historical acquisition planning")
        if self.safe_chunk_calendar_days != SAFE_CHUNK_CALENDAR_DAYS:
            raise ValueError("safe_chunk_calendar_days must remain the conservative 28-day value")
        _date_tuple(self.normal_trading_dates, name="normal_trading_dates")
        _date_tuple(self.holiday_dates, name="holiday_dates")
        _date_tuple(
            self.excluded_special_session_dates,
            name="excluded_special_session_dates",
        )
        for name, population in (
            ("historical_acquisition_superset", self.historical_acquisition_superset),
            ("frozen_wfo_population", self.frozen_wfo_population),
        ):
            if not population or tuple(sorted(set(population))) != population:
                raise ValueError(f"{name} must be sorted unique and non-empty")
        if self.formation_boundary_adapter.formation_boundary != self.research_start:
            raise HistoricalAcquisitionEvidenceError(
                ["formation boundary must equal research_start"]
            )
        if tuple(self.formation_boundary_adapter.frozen_train_universe.instruments) != (
            self.frozen_wfo_population
        ):
            raise HistoricalAcquisitionEvidenceError(
                ["frozen_wfo_population must come from the formation-boundary adapter"]
            )
        if self.missing_unknown_evidence:
            raise HistoricalAcquisitionEvidenceError(self.missing_unknown_evidence)
        source_ids = tuple(item.source_id for item in self.evidence_sources)
        if tuple(sorted(set(source_ids))) != source_ids:
            raise ValueError("evidence_sources must be sorted unique by source_id")
        if self.calendar_source_id not in set(source_ids):
            raise ValueError("calendar source is missing from evidence_sources")
        segment_keys = tuple(sorted({item.instrument_key for item in self.pit_segments}))
        if not segment_keys:
            raise ValueError("pit_segments must not be empty")
        if tuple(item.instrument_key for item in self.instrument_acquisition_union) != (
            self.historical_acquisition_superset
        ):
            raise ValueError(
                "instrument acquisition union must equal historical acquisition superset"
            )
        interval_keys = tuple(
            (item.instrument_key, item.interval_minutes, item.start, item.end)
            for item in self.requested_intervals
        )
        if tuple(sorted(interval_keys)) != interval_keys:
            raise ValueError("requested_intervals must be in canonical order")
        if not self.requested_interval_minutes:
            raise ValueError("at least one requested interval is required")
        if self.expected_request_count <= 0 or self.estimated_rows <= 0:
            raise ValueError("plan estimates must be positive")
        if self.estimated_storage_bytes <= 0 or self.estimated_bytes_per_row <= 0:
            raise ValueError("storage estimates must be positive")

    def deterministic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "research_window": {
                "start": self.research_start.isoformat(),
                "end": self.research_end.isoformat(),
            },
            "calendar_source_id": self.calendar_source_id,
            "normal_trading_dates": [item.isoformat() for item in self.normal_trading_dates],
            "holiday_dates": [item.isoformat() for item in self.holiday_dates],
            "excluded_special_session_dates": [
                item.isoformat() for item in self.excluded_special_session_dates
            ],
            "evidence_sources": [item.as_dict() for item in self.evidence_sources],
            "pit_segments": [item.as_dict() for item in self.pit_segments],
            "formation_boundary_adapter": self.formation_boundary_adapter.as_dict(),
            "historical_acquisition_superset": list(self.historical_acquisition_superset),
            "frozen_wfo_population": list(self.frozen_wfo_population),
            "session_evidence": [item.as_dict() for item in self.session_evidence],
            "corporate_action_evidence": self.corporate_action_evidence.as_dict(),
            "instrument_acquisition_union": [
                item.as_dict() for item in self.instrument_acquisition_union
            ],
            "requested_intervals": [item.as_dict() for item in self.requested_intervals],
            "requested_interval_minutes": list(self.requested_interval_minutes),
            "safe_chunk_calendar_days": self.safe_chunk_calendar_days,
            "expected_request_count": self.expected_request_count,
            "estimated_rows": self.estimated_rows,
            "estimated_storage_bytes": self.estimated_storage_bytes,
            "estimated_bytes_per_row": self.estimated_bytes_per_row,
            "acquisition_exclusions": [item.as_dict() for item in self.acquisition_exclusions],
            "capital_scenarios": [item.as_dict() for item in self.capital_scenarios],
            "live_orders_called": False,
        }

    def deterministic_fingerprint(self) -> str:
        return _canonical_sha256(self.deterministic_payload())

    @property
    def plan_id(self) -> str:
        return f"hap_{self.deterministic_fingerprint()[:16]}"

    def to_dict(self) -> dict[str, Any]:
        payload = self.deterministic_payload()
        payload["plan_id"] = self.plan_id
        payload["deterministic_fingerprint"] = self.deterministic_fingerprint()
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"


def _source_map(sources: Iterable[EvidenceSourceIdentity]) -> dict[str, EvidenceSourceIdentity]:
    result: dict[str, EvidenceSourceIdentity] = {}
    for source in sorted(sources, key=lambda item: item.source_id):
        if source.source_id in result:
            raise ValueError(f"duplicate evidence source_id {source.source_id!r}")
        result[source.source_id] = source
    if not result:
        raise HistoricalAcquisitionEvidenceError(["no evidence source identities supplied"])
    return result


def _validate_window_dates(
    *, start: date, end: date, dates: Iterable[date], name: str
) -> tuple[date, ...]:
    ordered = _date_tuple(dates, name=name)
    if any(item < start or item > end for item in ordered):
        raise HistoricalAcquisitionEvidenceError(
            [f"{name} contains a date outside research window {start}..{end}"]
        )
    return ordered


def _historical_request_count(*, start: date, end: date) -> int:
    if start > end:
        raise ValueError("start must be on or before end")
    span_days = (end - start).days + 1
    return (span_days + SAFE_CHUNK_CALENDAR_DAYS - 1) // SAFE_CHUNK_CALENDAR_DAYS


def _contiguous_runs(
    *, normal_dates: tuple[date, ...], eligible_dates: tuple[date, ...]
) -> tuple[tuple[date, ...], ...]:
    positions = {item: index for index, item in enumerate(normal_dates)}
    runs: list[list[date]] = []
    for day in eligible_dates:
        if day not in positions:
            raise HistoricalAcquisitionEvidenceError(
                [f"eligible acquisition date {day.isoformat()} is not a normal session date"]
            )
        if not runs or positions[day] != positions[runs[-1][-1]] + 1:
            runs.append([day])
        else:
            runs[-1].append(day)
    return tuple(tuple(run) for run in runs)


def _validate_segment_timelines(
    *,
    segments_by_key: Mapping[str, tuple[PITMembershipSegment, ...]],
    research_end: date,
) -> tuple[str, ...]:
    unknown: list[str] = []
    for instrument_key, segments in sorted(segments_by_key.items()):
        ordered = tuple(sorted(segments, key=lambda item: item.valid_from))
        for first, second in pairwise(ordered):
            if first.valid_to >= second.valid_from:
                unknown.append(f"overlapping PIT segments for {instrument_key}")
            elif first.valid_to + timedelta(days=1) < second.valid_from:
                unknown.append(f"gap in PIT segment evidence for {instrument_key}")
        if ordered[-1].valid_to < research_end:
            unknown.append(
                f"PIT segment evidence for {instrument_key} ends before research_end "
                f"{research_end.isoformat()}"
            )
    return tuple(unknown)


def build_historical_acquisition_plan(
    *,
    research_start: date,
    research_end: date,
    calendar: CalendarEvidence,
    calendar_source_id: str,
    evidence_sources: Iterable[EvidenceSourceIdentity],
    pit_segments: Iterable[PITMembershipSegment],
    formation_boundary_adapter: FormationBoundaryAdapter,
    session_evidence: Iterable[SessionEvidence],
    corporate_action_evidence: CorporateActionEvidence,
    requested_interval_minutes: Iterable[int] = (5,),
    estimated_bytes_per_row: int = 160,
    capital_scenarios: Iterable[CapitalScenarioAnalysis] = (),
) -> HistoricalAcquisitionPlan:
    """Build a no-download acquisition plan from canonical date-scoped PIT segments.

    The formation adapter is authoritative for the frozen WFO candidate population.
    Segment keys outside that population are allowed in the acquisition superset when
    they become eligible later, but they are never inserted into the frozen universe.
    """

    if research_start > research_end:
        raise ValueError("research_start must be on or before research_end")
    if not isinstance(estimated_bytes_per_row, int) or isinstance(estimated_bytes_per_row, bool):
        raise TypeError("estimated_bytes_per_row must be an integer")
    if estimated_bytes_per_row <= 0:
        raise ValueError("estimated_bytes_per_row must be positive")
    if not isinstance(formation_boundary_adapter, FormationBoundaryAdapter):
        raise TypeError("formation_boundary_adapter must be a FormationBoundaryAdapter")

    sources = _source_map(evidence_sources)
    unknown: list[str] = []
    if formation_boundary_adapter.formation_boundary != research_start:
        unknown.append("formation boundary must equal research_start")
    source_by_fingerprint = {item.fingerprint for item in sources.values()}
    if calendar_source_id not in sources:
        unknown.append(f"missing calendar evidence source {calendar_source_id!r}")

    normal_dates = _validate_window_dates(
        start=research_start,
        end=research_end,
        dates=calendar.trading_dates,
        name="calendar.trading_dates",
    )
    holidays = _validate_window_dates(
        start=research_start,
        end=research_end,
        dates=calendar.holiday_dates,
        name="calendar.holiday_dates",
    )
    special_dates = _validate_window_dates(
        start=research_start,
        end=research_end,
        dates=calendar.excluded_special_session_dates,
        name="calendar.excluded_special_session_dates",
    )
    normal_set = set(normal_dates)
    if normal_set & set(special_dates):
        unknown.append("calendar marks a date as both normal and excluded special session")
    if normal_set & set(holidays):
        unknown.append("calendar marks a date as both normal trading and holiday")
    if not normal_dates:
        unknown.append("calendar has no normal trading dates in research window")

    raw_segments = tuple(pit_segments)
    if any(not isinstance(item, PITMembershipSegment) for item in raw_segments):
        unknown.append("all PIT evidence must be canonical PITMembershipSegment records")
    segments = tuple(
        sorted(
            (item for item in raw_segments if isinstance(item, PITMembershipSegment)),
            key=lambda item: (item.instrument_key, item.valid_from, item.valid_to),
        )
    )
    if not segments:
        unknown.append("no canonical PIT membership segments supplied")
    segments_by_key: dict[str, tuple[PITMembershipSegment, ...]] = {}
    for segment in segments:
        segments_by_key.setdefault(segment.instrument_key, ())
        segments_by_key[segment.instrument_key] += (segment,)
        if segment.source_fingerprint not in source_by_fingerprint:
            unknown.append(
                f"missing PIT segment source fingerprint {segment.source_fingerprint!r} "
                f"for {segment.instrument_key}"
            )
    if segments:
        unknown.extend(
            _validate_segment_timelines(
                segments_by_key=segments_by_key,
                research_end=research_end,
            )
        )

    frozen = formation_boundary_adapter.frozen_train_universe
    frozen_keys = set(frozen.instruments)
    for instrument_key in frozen.instruments:
        formation_segments = tuple(
            segment
            for segment in segments_by_key.get(instrument_key, ())
            if segment.valid_from <= research_start <= segment.valid_to
        )
        if len(formation_segments) != 1:
            unknown.append(
                f"frozen instrument {instrument_key} lacks exactly one formation-boundary PIT segment"
            )

    sessions = tuple(sorted(session_evidence, key=lambda item: item.trade_date))
    sessions_by_date: dict[date, SessionEvidence] = {}
    for item in sessions:
        if item.trade_date < research_start or item.trade_date > research_end:
            unknown.append(f"session evidence is outside research window: {item.trade_date}")
        if item.trade_date in sessions_by_date:
            unknown.append(f"duplicate session evidence for {item.trade_date.isoformat()}")
        sessions_by_date[item.trade_date] = item
        if item.source_id not in sources:
            unknown.append(f"missing session evidence source {item.source_id!r}")
    for day in normal_dates:
        session = sessions_by_date.get(day)
        if session is None:
            unknown.append(f"missing normal session evidence for {day.isoformat()}")
        elif session.acquisition_allowed is not True or session.session_kind != "NORMAL":
            unknown.append(
                f"normal session unexpectedly excluded or misclassified for {day.isoformat()}"
            )
    for day in special_dates:
        session = sessions_by_date.get(day)
        if session is None:
            unknown.append(f"missing special-session evidence for {day.isoformat()}")
        elif session.acquisition_allowed or session.session_kind != "SPECIAL":
            unknown.append(f"special session is not explicitly excluded for {day.isoformat()}")

    ca = corporate_action_evidence
    if ca.source_id not in sources:
        unknown.append(f"missing corporate-action evidence source {ca.source_id!r}")
    if not ca.complete:
        unknown.append("corporate-action evidence is incomplete")
    if ca.unknown_instruments:
        unknown.append(
            "corporate-action evidence has unknown instruments: "
            + ", ".join(ca.unknown_instruments)
        )
    if ca.coverage_start > research_start or ca.coverage_end < research_end:
        unknown.append("corporate-action evidence does not cover the full research window")

    raw_intervals = tuple(requested_interval_minutes)
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 1 or item > 15
        for item in raw_intervals
    ):
        raise ValueError("requested_interval_minutes must contain integers from 1 through 15")
    intervals = tuple(sorted(set(raw_intervals)))
    if not intervals:
        raise ValueError("requested_interval_minutes must contain at least one interval")

    eligible_by_key: dict[str, tuple[date, ...]] = {}
    segment_sources_by_key: dict[str, set[str]] = {}
    exclusions: list[AcquisitionExclusion] = []
    for instrument_key, instrument_segments in sorted(segments_by_key.items()):
        eligible_dates: list[date] = []
        segment_sources_by_key[instrument_key] = set()
        first_segment = min(instrument_segments, key=lambda item: item.valid_from)
        for day in normal_dates:
            if day < first_segment.valid_from:
                continue
            covering = [item for item in instrument_segments if item.covers(day)]
            if len(covering) != 1:
                unknown.append(
                    f"PIT membership for {instrument_key} on {day.isoformat()} "
                    f"must resolve exactly once, found {len(covering)}"
                )
                continue
            resolved = resolve_pit_segment(tuple(covering), day)
            segment_sources_by_key[instrument_key].add(resolved.source_fingerprint)
            if resolved.eligible:
                eligible_dates.append(day)
            else:
                exclusions.append(
                    AcquisitionExclusion(
                        trade_date=day,
                        instrument_key=instrument_key,
                        reason_code="DATED_MEMBERSHIP_NOT_ELIGIBLE",
                        detail="canonical PIT segment marks the instrument ineligible on this date",
                    )
                )
        if eligible_dates:
            eligible_by_key[instrument_key] = tuple(eligible_dates)

    historical_superset = tuple(sorted(eligible_by_key))
    if not historical_superset:
        unknown.append("no historically eligible instrument dates")
    if tuple(ca.covered_instruments) != historical_superset:
        unknown.append(
            "corporate-action evidence population must equal historical acquisition superset: "
            f"expected {historical_superset}, got {ca.covered_instruments}"
        )
    if unknown:
        raise HistoricalAcquisitionEvidenceError(tuple(sorted(set(unknown))))

    exclusions.extend(
        AcquisitionExclusion(
            trade_date=day,
            instrument_key=None,
            reason_code="NSE_HOLIDAY",
            detail="normal-session acquisition excluded by dated NSE calendar holiday evidence",
        )
        for day in holidays
    )
    exclusions.extend(
        AcquisitionExclusion(
            trade_date=day,
            instrument_key=None,
            reason_code="SPECIAL_SESSION_EXCLUDED",
            detail="special session excluded until a dedicated acquisition rule is supplied",
        )
        for day in special_dates
    )
    exclusions.sort(
        key=lambda item: (
            item.trade_date or date.min,
            item.instrument_key or "",
            item.reason_code,
        )
    )

    unions = tuple(
        InstrumentAcquisitionUnion(
            instrument_key=instrument_key,
            eligible_trade_dates=eligible_by_key[instrument_key],
            source_fingerprints=tuple(sorted(segment_sources_by_key[instrument_key])),
        )
        for instrument_key in historical_superset
    )
    requested: list[RequestedAcquisitionInterval] = []
    for union in unions:
        for run in _contiguous_runs(
            normal_dates=normal_dates,
            eligible_dates=union.eligible_trade_dates,
        ):
            for interval_minutes in intervals:
                expected_rows = sum(
                    sessions_by_date[day].expected_rows(interval_minutes) for day in run
                )
                requested.append(
                    RequestedAcquisitionInterval(
                        instrument_key=union.instrument_key,
                        interval_minutes=interval_minutes,
                        start=run[0],
                        end=run[-1],
                        trade_dates=run,
                        expected_rows=expected_rows,
                        estimated_storage_bytes=expected_rows * estimated_bytes_per_row,
                        request_count=_historical_request_count(start=run[0], end=run[-1]),
                    )
                )
    requested_tuple = tuple(
        sorted(
            requested,
            key=lambda item: (item.instrument_key, item.interval_minutes, item.start, item.end),
        )
    )
    if not requested_tuple:
        raise HistoricalAcquisitionEvidenceError(["no historically eligible acquisition intervals"])

    return HistoricalAcquisitionPlan(
        schema_version=HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION,
        mode="DRY_RUN",
        research_start=research_start,
        research_end=research_end,
        calendar_source_id=calendar_source_id,
        normal_trading_dates=normal_dates,
        holiday_dates=holidays,
        excluded_special_session_dates=special_dates,
        evidence_sources=tuple(sorted(sources.values(), key=lambda item: item.source_id)),
        pit_segments=segments,
        formation_boundary_adapter=formation_boundary_adapter,
        historical_acquisition_superset=historical_superset,
        frozen_wfo_population=tuple(sorted(frozen_keys)),
        session_evidence=sessions,
        corporate_action_evidence=ca,
        instrument_acquisition_union=unions,
        requested_intervals=requested_tuple,
        requested_interval_minutes=intervals,
        safe_chunk_calendar_days=SAFE_CHUNK_CALENDAR_DAYS,
        expected_request_count=sum(item.request_count for item in requested_tuple),
        estimated_rows=sum(item.expected_rows for item in requested_tuple),
        estimated_storage_bytes=sum(item.estimated_storage_bytes for item in requested_tuple),
        estimated_bytes_per_row=estimated_bytes_per_row,
        acquisition_exclusions=tuple(exclusions),
        capital_scenarios=tuple(capital_scenarios),
        live_orders_called=False,
    )


def _example_digest(label: str) -> str:
    return hashlib.sha256(f"historical-acquisition-example-v2::{label}".encode()).hexdigest()


def build_example_historical_acquisition_plan() -> HistoricalAcquisitionPlan:
    """Build a compact deterministic range-segment example for the first design window."""

    start = date(2024, 10, 1)
    end = date(2024, 10, 30)
    special_session = date(2024, 10, 2)
    normal_dates = tuple(
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
        if (start + timedelta(days=offset)).weekday() < 5
        and start + timedelta(days=offset) != special_session
    )
    calendar = CalendarEvidence(
        trading_dates=normal_dates,
        holiday_dates=(),
        excluded_special_session_dates=(special_session,),
        source_urls=("example-v2:synthetic-calendar",),
    )
    source_ids = {
        "calendar": "example-v2:synthetic-calendar",
        "segments": "example-v2:canonical-pit-segments",
        "sessions": "example-v2:nse-normal-sessions",
        "corporate_actions": "example-v2:corporate-actions",
    }
    sources = tuple(
        EvidenceSourceIdentity(
            source_id=source_id,
            evidence_type=evidence_type,
            fingerprint=_example_digest(source_id),
        )
        for source_id, evidence_type in (
            (source_ids["calendar"], "nse-calendar"),
            (source_ids["segments"], "canonical-pit-membership-segments"),
            (source_ids["sessions"], "nse-sessions"),
            (source_ids["corporate_actions"], "corporate-actions"),
        )
    )
    segment_source = _example_digest(source_ids["segments"])
    alpha = "NSE_EQ|INE000A01010"
    beta = "NSE_EQ|INE001A01010"
    segments = (
        PITMembershipSegment(alpha, start, end, start, segment_source, True),
        PITMembershipSegment(
            beta,
            date(2024, 10, 3),
            date(2024, 10, 14),
            date(2024, 10, 3),
            segment_source,
            True,
        ),
        PITMembershipSegment(
            beta,
            date(2024, 10, 15),
            date(2024, 10, 15),
            date(2024, 10, 15),
            segment_source,
            False,
        ),
        PITMembershipSegment(
            beta,
            date(2024, 10, 16),
            end,
            date(2024, 10, 16),
            segment_source,
            True,
        ),
    )
    policy = "example-v2:formation-frozen-universe"
    frozen_segments = (segments[0],)
    from .research_window_compiler import derive_population_fingerprint

    frozen = FrozenTrainUniverse(
        instruments=(alpha,),
        universe_policy_id=policy,
        population_fingerprint=derive_population_fingerprint(
            instruments=(alpha,),
            universe_policy_id=policy,
            pit_segments=frozen_segments,
        ),
        frozen_as_of=start,
    )
    adapter = FormationBoundaryAdapter(
        formation_boundary=start,
        frozen_train_universe=frozen,
        research_window_plan_fingerprint=_example_digest("formation-plan"),
    )
    sessions = tuple(
        SessionEvidence(
            trade_date=day,
            session_kind="NORMAL",
            source_id=source_ids["sessions"],
            timezone="Asia/Kolkata",
            start_time="09:15",
            end_time_exclusive="15:30",
            expected_rows_by_interval=((5, 75), (15, 25)),
        )
        for day in calendar.trading_dates
    ) + tuple(
        SessionEvidence(
            trade_date=day,
            session_kind="SPECIAL",
            source_id=source_ids["sessions"],
            timezone="Asia/Kolkata",
            start_time="00:00",
            end_time_exclusive="00:01",
            acquisition_allowed=False,
            exclusion_reason="special session requires a dedicated rule",
        )
        for day in calendar.excluded_special_session_dates
    )
    ca = CorporateActionEvidence(
        source_id=source_ids["corporate_actions"],
        fingerprint=_example_digest(source_ids["corporate_actions"]),
        coverage_start=start,
        coverage_end=end,
        covered_instruments=(alpha, beta),
        complete=True,
        policy_identity="example-v2:raw-unadjusted-block-structural-actions",
    )
    return build_historical_acquisition_plan(
        research_start=start,
        research_end=end,
        calendar=calendar,
        calendar_source_id=source_ids["calendar"],
        evidence_sources=sources,
        pit_segments=segments,
        formation_boundary_adapter=adapter,
        session_evidence=sessions,
        corporate_action_evidence=ca,
        requested_interval_minutes=(5, 15),
        estimated_bytes_per_row=160,
        capital_scenarios=(
            CapitalScenarioAnalysis(
                scenario_id="capital-1000-inr",
                capital_rupees=Decimal(1000),
                note="Analysis only; does not remove any historically required instrument.",
            ),
        ),
    )


__all__ = [
    "HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION",
    "SAFE_CHUNK_CALENDAR_DAYS",
    "AcquisitionExclusion",
    "CapitalScenarioAnalysis",
    "CorporateActionEvidence",
    "EvidenceSourceIdentity",
    "FormationBoundaryAdapter",
    "HistoricalAcquisitionEvidenceError",
    "HistoricalAcquisitionPlan",
    "InstrumentAcquisitionUnion",
    "PITMembershipSegment",
    "RequestedAcquisitionInterval",
    "SessionEvidence",
    "build_example_historical_acquisition_plan",
    "build_historical_acquisition_plan",
]
