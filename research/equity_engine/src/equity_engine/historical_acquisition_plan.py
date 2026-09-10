"""Deterministic, point-in-time historical acquisition planning.

This module is intentionally a DRY-RUN boundary. It consumes dated evidence that has
already been acquired or supplied by a caller; it never downloads historical candles,
reads current MIS/suspension state, places orders, or mutates research data.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from .nse_calendar import CalendarEvidence

HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION = "historical-acquisition-plan/v1"
_DIGEST_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")
_UPSTOX_MINUTE_MAX_CALENDAR_DAYS = 28


def _historical_request_count(*, start: date, end: date, interval_minutes: int) -> int:
    if start > end:
        raise ValueError("start must be on or before end")
    if interval_minutes < 1 or interval_minutes > 15:
        raise ValueError("interval_minutes must be between 1 and 15")
    span_days = (end - start).days + 1
    return (span_days + _UPSTOX_MINUTE_MAX_CALENDAR_DAYS - 1) // _UPSTOX_MINUTE_MAX_CALENDAR_DAYS


class HistoricalAcquisitionEvidenceError(ValueError):
    """Raised when a dry-run cannot prove its historical evidence inputs completely."""

    def __init__(self, reasons: Iterable[str]) -> None:
        self.reasons = tuple(str(reason) for reason in reasons if str(reason).strip())
        detail = "; ".join(self.reasons) if self.reasons else "unknown evidence"
        super().__init__(f"historical acquisition evidence is incomplete: {detail}")


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


@dataclass(frozen=True)
class EvidenceSourceIdentity:
    """Stable identity for one already-sourced evidence input."""

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
class PITMembershipCoverage:
    """Completeness certificate for one dated membership snapshot."""

    trade_date: date
    source_id: str
    record_count: int
    complete: bool = True

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("PIT membership coverage source_id is required")
        if not isinstance(self.record_count, int) or isinstance(self.record_count, bool):
            raise TypeError("PIT membership coverage record_count must be an integer")
        if self.record_count < 0:
            raise ValueError("PIT membership coverage record_count cannot be negative")
        if not isinstance(self.complete, bool):
            raise TypeError("PIT membership coverage completeness must be boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "source_id": self.source_id,
            "record_count": self.record_count,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class PITMembershipEvidence:
    """One dated NSE membership, identity, status, and tick observation.

    ``evidence_as_of`` is explicit so a record cannot silently use a later snapshot to
    establish an earlier historical status. The four source IDs may point to the same
    dated MII payload when that payload supplies all fields, or to separately attested
    identity/status/tick contracts.
    """

    trade_date: date
    instrument_key: str
    isin: str
    symbol: str
    name: str
    series: str
    listed_on_nse: bool
    normal_equity: bool
    tradeable_in_normal_market: bool
    tick_size_rupees: Decimal
    evidence_as_of: date
    membership_source_id: str
    identity_source_id: str
    status_source_id: str
    tick_source_id: str

    def __post_init__(self) -> None:
        if not self.instrument_key.strip() or not self.isin.strip():
            raise ValueError("PIT membership instrument identity is required")
        if self.instrument_key != f"NSE_EQ|{self.isin}":
            raise ValueError("instrument_key must be NSE_EQ|<isin>")
        for name, value in (
            ("symbol", self.symbol),
            ("series", self.series),
            ("membership_source_id", self.membership_source_id),
            ("identity_source_id", self.identity_source_id),
            ("status_source_id", self.status_source_id),
            ("tick_source_id", self.tick_source_id),
        ):
            if not value.strip():
                raise ValueError(f"PIT membership {name} is required")
        for name, value in (
            ("listed_on_nse", self.listed_on_nse),
            ("normal_equity", self.normal_equity),
            ("tradeable_in_normal_market", self.tradeable_in_normal_market),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"PIT membership {name} must be boolean")
        if self.tick_size_rupees <= 0:
            raise ValueError("PIT membership tick_size_rupees must be positive")
        if self.evidence_as_of > self.trade_date:
            raise HistoricalAcquisitionEvidenceError(
                [
                    (
                        f"future PIT evidence for {self.instrument_key} on "
                        f"{self.trade_date.isoformat()}"
                    )
                ]
            )

    @property
    def eligible(self) -> bool:
        return self.listed_on_nse and self.normal_equity and self.tradeable_in_normal_market

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "instrument_key": self.instrument_key,
            "isin": self.isin,
            "symbol": self.symbol,
            "name": self.name,
            "series": self.series,
            "listed_on_nse": self.listed_on_nse,
            "normal_equity": self.normal_equity,
            "tradeable_in_normal_market": self.tradeable_in_normal_market,
            "eligible": self.eligible,
            "tick_size_rupees": str(self.tick_size_rupees),
            "evidence_as_of": self.evidence_as_of.isoformat(),
            "membership_source_id": self.membership_source_id,
            "identity_source_id": self.identity_source_id,
            "status_source_id": self.status_source_id,
            "tick_source_id": self.tick_source_id,
        }


@dataclass(frozen=True)
class SessionEvidence:
    """Explicit session/special-session evidence for one calendar date."""

    trade_date: date
    session_kind: str
    source_id: str
    timezone: str
    start_time: str
    end_time_exclusive: str
    expected_rows_by_interval: tuple[tuple[int, int], ...] = ()
    acquisition_allowed: bool = True
    exclusion_reason: str | None = None
    auxiliary_windows: tuple[str, ...] = ()

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
            if not isinstance(interval, int) or isinstance(interval, bool) or not 1 <= interval <= 15:
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
            "auxiliary_windows": list(self.auxiliary_windows),
        }


@dataclass(frozen=True)
class CorporateActionEvidence:
    """Already-resolved corporate-action evidence required by the plan boundary."""

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
class HistoricalIdentityPoint:
    trade_date: date
    symbol: str
    name: str
    source_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "symbol": self.symbol,
            "name": self.name,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class HistoricalTickPoint:
    trade_date: date
    tick_size_rupees: Decimal
    source_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "tick_size_rupees": str(self.tick_size_rupees),
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class InstrumentIdentityUnion:
    """Union of all historically required dates and identity/tick observations."""

    instrument_key: str
    isin: str
    first_required_date: date
    last_required_date: date
    required_trade_dates: tuple[date, ...]
    identity_points: tuple[HistoricalIdentityPoint, ...]
    tick_points: tuple[HistoricalTickPoint, ...]

    def __post_init__(self) -> None:
        if self.instrument_key != f"NSE_EQ|{self.isin}":
            raise ValueError("instrument identity union key does not match ISIN")
        if not self.required_trade_dates:
            raise ValueError("instrument identity union requires trade dates")
        if tuple(sorted(set(self.required_trade_dates))) != self.required_trade_dates:
            raise ValueError("required_trade_dates must be sorted unique")
        if self.first_required_date != self.required_trade_dates[0]:
            raise ValueError("first_required_date does not match required dates")
        if self.last_required_date != self.required_trade_dates[-1]:
            raise ValueError("last_required_date does not match required dates")
        identity_dates = tuple(item.trade_date for item in self.identity_points)
        tick_dates = tuple(item.trade_date for item in self.tick_points)
        if tuple(sorted(set(identity_dates))) != identity_dates:
            raise ValueError("identity_points must be sorted unique by date")
        if tuple(sorted(set(tick_dates))) != tick_dates:
            raise ValueError("tick_points must be sorted unique by date")

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument_key": self.instrument_key,
            "isin": self.isin,
            "first_required_date": self.first_required_date.isoformat(),
            "last_required_date": self.last_required_date.isoformat(),
            "required_trade_dates": [item.isoformat() for item in self.required_trade_dates],
            "identity_points": [item.as_dict() for item in self.identity_points],
            "tick_points": [item.as_dict() for item in self.tick_points],
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
        if self.interval_minutes < 1 or self.interval_minutes > 15:
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
    """Optional capital commentary that is explicitly forbidden from filtering the plan."""

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
        if not self.analysis_only:
            raise ValueError("capital scenarios must remain analysis_only")
        if self.excluded_instruments:
            raise ValueError("capital scenarios cannot exclude historical instruments")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "capital_rupees": str(self.capital_rupees),
            "note": self.note,
            "analysis_only": self.analysis_only,
            "excluded_instruments": list(self.excluded_instruments),
        }


@dataclass(frozen=True)
class HistoricalAcquisitionPlan:
    """Immutable dry-run artifact for a PIT historical acquisition union."""

    schema_version: str
    mode: str
    research_start: date
    research_end: date
    calendar_source_id: str
    normal_trading_dates: tuple[date, ...]
    holiday_dates: tuple[date, ...]
    excluded_special_session_dates: tuple[date, ...]
    evidence_sources: tuple[EvidenceSourceIdentity, ...]
    membership_coverage: tuple[PITMembershipCoverage, ...]
    pit_membership_evidence: tuple[PITMembershipEvidence, ...]
    session_evidence: tuple[SessionEvidence, ...]
    corporate_action_evidence: CorporateActionEvidence
    instrument_identity_union: tuple[InstrumentIdentityUnion, ...]
    requested_intervals: tuple[RequestedAcquisitionInterval, ...]
    requested_interval_minutes: tuple[int, ...]
    expected_request_count: int
    estimated_rows: int
    estimated_storage_bytes: int
    estimated_bytes_per_row: int
    missing_unknown_evidence: tuple[str, ...]
    acquisition_exclusions: tuple[AcquisitionExclusion, ...]
    capital_scenarios: tuple[CapitalScenarioAnalysis, ...] = ()
    universe_policy_id: str = "dated-nse-pit-membership-v1"
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported historical acquisition plan schema {self.schema_version!r}")
        if self.mode != "DRY_RUN":
            raise ValueError("historical acquisition plan mode must be DRY_RUN")
        if self.research_start > self.research_end:
            raise ValueError("research window dates are invalid")
        if self.calendar_source_id.strip() == "":
            raise ValueError("calendar_source_id is required")
        if self.live_orders_called:
            raise ValueError("live orders are forbidden in historical acquisition planning")
        if not self.universe_policy_id.strip():
            raise ValueError("universe_policy_id is required")
        _date_tuple(self.normal_trading_dates, name="normal_trading_dates")
        _date_tuple(self.holiday_dates, name="holiday_dates")
        _date_tuple(self.excluded_special_session_dates, name="excluded_special_session_dates")
        if not self.requested_interval_minutes:
            raise ValueError("at least one requested interval is required")
        if tuple(sorted(set(self.requested_interval_minutes))) != self.requested_interval_minutes:
            raise ValueError("requested_interval_minutes must be sorted unique")
        if self.expected_request_count <= 0 or self.estimated_rows <= 0:
            raise ValueError("plan estimates must be positive")
        if self.estimated_storage_bytes <= 0 or self.estimated_bytes_per_row <= 0:
            raise ValueError("storage estimates must be positive")
        if self.missing_unknown_evidence:
            raise HistoricalAcquisitionEvidenceError(self.missing_unknown_evidence)
        source_ids = tuple(item.source_id for item in self.evidence_sources)
        if tuple(sorted(set(source_ids))) != source_ids:
            raise ValueError("evidence_sources must be sorted unique by source_id")
        if self.calendar_source_id not in set(source_ids):
            raise ValueError("calendar source is missing from evidence_sources")
        coverage_dates = tuple(item.trade_date for item in self.membership_coverage)
        if tuple(sorted(set(coverage_dates))) != coverage_dates:
            raise ValueError("membership_coverage must be sorted unique by trade_date")
        membership_keys = tuple(
            (item.trade_date, item.instrument_key) for item in self.pit_membership_evidence
        )
        if tuple(sorted(set(membership_keys))) != membership_keys:
            raise ValueError("pit_membership_evidence must be sorted unique by date and instrument")
        session_dates = tuple(item.trade_date for item in self.session_evidence)
        if tuple(sorted(set(session_dates))) != session_dates:
            raise ValueError("session_evidence must be sorted unique by trade_date")
        union_keys = tuple(item.instrument_key for item in self.instrument_identity_union)
        if tuple(sorted(set(union_keys))) != union_keys:
            raise ValueError("instrument_identity_union must be sorted unique")
        interval_keys = tuple(
            (item.instrument_key, item.interval_minutes, item.start, item.end)
            for item in self.requested_intervals
        )
        if tuple(sorted(interval_keys)) != interval_keys:
            raise ValueError("requested_intervals must be in canonical order")

    def deterministic_payload(self) -> dict[str, Any]:
        """Return every non-volatile field used for plan identity."""

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
            "membership_coverage": [item.as_dict() for item in self.membership_coverage],
            "pit_membership_evidence": [item.as_dict() for item in self.pit_membership_evidence],
            "session_evidence": [item.as_dict() for item in self.session_evidence],
            "corporate_action_evidence": self.corporate_action_evidence.as_dict(),
            "instrument_identity_union": [
                item.as_dict() for item in self.instrument_identity_union
            ],
            "requested_intervals": [item.as_dict() for item in self.requested_intervals],
            "requested_interval_minutes": list(self.requested_interval_minutes),
            "expected_request_count": self.expected_request_count,
            "estimated_rows": self.estimated_rows,
            "estimated_storage_bytes": self.estimated_storage_bytes,
            "estimated_bytes_per_row": self.estimated_bytes_per_row,
            "missing_unknown_evidence": list(self.missing_unknown_evidence),
            "acquisition_exclusions": [item.as_dict() for item in self.acquisition_exclusions],
            "capital_scenarios": [item.as_dict() for item in self.capital_scenarios],
            "universe_policy_id": self.universe_policy_id,
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

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"


def _validate_window_dates(
    *, start: date, end: date, dates: Iterable[date], name: str
) -> tuple[date, ...]:
    ordered = _date_tuple(dates, name=name)
    if any(item < start or item > end for item in ordered):
        raise HistoricalAcquisitionEvidenceError(
            [f"{name} contains a date outside research window {start}..{end}"]
        )
    return ordered


def _source_map(
    sources: Iterable[EvidenceSourceIdentity],
) -> dict[str, EvidenceSourceIdentity]:
    ordered = tuple(sorted(sources, key=lambda item: item.source_id))
    result: dict[str, EvidenceSourceIdentity] = {}
    for source in ordered:
        if source.source_id in result:
            raise ValueError(f"duplicate evidence source_id {source.source_id!r}")
        result[source.source_id] = source
    if not result:
        raise HistoricalAcquisitionEvidenceError(["no evidence source identities supplied"])
    return result


def _identity_union(
    *,
    records: tuple[PITMembershipEvidence, ...],
) -> tuple[InstrumentIdentityUnion, ...]:
    by_key: dict[str, list[PITMembershipEvidence]] = {}
    for record in records:
        if record.eligible:
            by_key.setdefault(record.instrument_key, []).append(record)
    unions: list[InstrumentIdentityUnion] = []
    for instrument_key in sorted(by_key):
        observations = sorted(by_key[instrument_key], key=lambda item: item.trade_date)
        required_dates = tuple(item.trade_date for item in observations)
        identity_points = tuple(
            HistoricalIdentityPoint(
                trade_date=item.trade_date,
                symbol=item.symbol,
                name=item.name,
                source_id=item.identity_source_id,
            )
            for item in observations
        )
        tick_points = tuple(
            HistoricalTickPoint(
                trade_date=item.trade_date,
                tick_size_rupees=item.tick_size_rupees,
                source_id=item.tick_source_id,
            )
            for item in observations
        )
        unions.append(
            InstrumentIdentityUnion(
                instrument_key=instrument_key,
                isin=observations[0].isin,
                first_required_date=required_dates[0],
                last_required_date=required_dates[-1],
                required_trade_dates=required_dates,
                identity_points=identity_points,
                tick_points=tick_points,
            )
        )
    return tuple(unions)


def _contiguous_runs(
    *, normal_dates: tuple[date, ...], eligible_dates: tuple[date, ...]
) -> tuple[tuple[date, ...], ...]:
    positions = {item: index for index, item in enumerate(normal_dates)}
    runs: list[list[date]] = []
    for day in eligible_dates:
        if not runs or positions[day] != positions[runs[-1][-1]] + 1:
            runs.append([day])
        else:
            runs[-1].append(day)
    return tuple(tuple(run) for run in runs)


def build_historical_acquisition_plan(
    *,
    research_start: date,
    research_end: date,
    calendar: CalendarEvidence,
    calendar_source_id: str,
    evidence_sources: Iterable[EvidenceSourceIdentity],
    membership_coverage: Iterable[PITMembershipCoverage],
    pit_membership_evidence: Iterable[PITMembershipEvidence],
    session_evidence: Iterable[SessionEvidence],
    corporate_action_evidence: CorporateActionEvidence,
    requested_interval_minutes: Iterable[int] = (5,),
    estimated_bytes_per_row: int = 160,
    capital_scenarios: Iterable[CapitalScenarioAnalysis] = (),
    universe_policy_id: str = "dated-nse-pit-membership-v1",
) -> HistoricalAcquisitionPlan:
    """Build a deterministic historical union without network or current-state inputs.

    Membership is resolved only from dated evidence on or before each trade date. A complete
    dated snapshot coverage certificate is required for every normal session. Records outside
    the requested window are ignored, so post-window/current membership cannot back-project into
    history. No capital scenario or current MIS/suspension state is accepted as a filter.
    """

    if research_start > research_end:
        raise ValueError("research_start must be on or before research_end")
    if not isinstance(estimated_bytes_per_row, int) or isinstance(estimated_bytes_per_row, bool):
        raise TypeError("estimated_bytes_per_row must be an integer")
    if estimated_bytes_per_row <= 0:
        raise ValueError("estimated_bytes_per_row must be positive")

    sources = _source_map(evidence_sources)
    unknown: list[str] = []
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
    special_set = set(special_dates)
    if normal_set & special_set:
        unknown.append("calendar marks a date as both normal and excluded special session")
    if not normal_dates:
        unknown.append("calendar has no normal trading dates in research window")

    coverage = tuple(sorted(membership_coverage, key=lambda item: item.trade_date))
    coverage_by_date: dict[date, PITMembershipCoverage] = {}
    for item in coverage:
        if item.trade_date in coverage_by_date:
            unknown.append(f"duplicate membership coverage for {item.trade_date.isoformat()}")
        coverage_by_date[item.trade_date] = item
        if item.source_id not in sources:
            unknown.append(
                f"missing membership coverage source {item.source_id!r} for "
                f"{item.trade_date.isoformat()}"
            )
        if not item.complete:
            unknown.append(f"incomplete membership coverage for {item.trade_date.isoformat()}")
    for day in normal_dates:
        if day not in coverage_by_date:
            unknown.append(f"missing membership coverage for {day.isoformat()}")

    all_records = tuple(
        sorted(
            pit_membership_evidence,
            key=lambda item: (item.trade_date, item.instrument_key),
        )
    )
    record_keys: set[tuple[date, str]] = set()
    records: list[PITMembershipEvidence] = []
    for record in all_records:
        # Deliberately ignore records outside the requested window. A current/future snapshot
        # cannot add a historical instrument or status to this plan.
        if not (research_start <= record.trade_date <= research_end):
            continue
        key = (record.trade_date, record.instrument_key)
        if key in record_keys:
            unknown.append(
                f"duplicate PIT membership record for {record.instrument_key} on "
                f"{record.trade_date.isoformat()}"
            )
        record_keys.add(key)
        records.append(record)
        for source_id in (
            record.membership_source_id,
            record.identity_source_id,
            record.status_source_id,
            record.tick_source_id,
        ):
            if source_id not in sources:
                unknown.append(
                    f"missing PIT membership source {source_id!r} for "
                    f"{record.instrument_key} on {record.trade_date.isoformat()}"
                )
    records_tuple = tuple(records)
    records_by_date: dict[date, list[PITMembershipEvidence]] = {}
    for record in records_tuple:
        records_by_date.setdefault(record.trade_date, []).append(record)
    for day in normal_dates:
        expected_count = coverage_by_date.get(day).record_count if day in coverage_by_date else None
        actual_count = len(records_by_date.get(day, ()))
        if expected_count is not None and expected_count != actual_count:
            unknown.append(
                f"membership coverage count mismatch for {day.isoformat()}: "
                f"expected {expected_count}, got {actual_count}"
            )

    sessions = tuple(sorted(session_evidence, key=lambda item: item.trade_date))
    sessions_by_date: dict[date, SessionEvidence] = {}
    for item in sessions:
        if item.trade_date in sessions_by_date:
            unknown.append(f"duplicate session evidence for {item.trade_date.isoformat()}")
        sessions_by_date[item.trade_date] = item
        if item.source_id not in sources:
            unknown.append(f"missing session evidence source {item.source_id!r}")
    for day in normal_dates:
        session = sessions_by_date.get(day)
        if session is None:
            unknown.append(f"missing normal session evidence for {day.isoformat()}")
        elif not session.acquisition_allowed:
            unknown.append(f"normal session unexpectedly excluded for {day.isoformat()}")
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

    intervals = tuple(sorted(set(requested_interval_minutes)))
    if not intervals or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 1 or item > 15
        for item in intervals
    ):
        raise ValueError("requested_interval_minutes must contain integers from 1 through 15")

    union = _identity_union(records=records_tuple)
    union_keys = tuple(item.instrument_key for item in union)
    if tuple(ca.covered_instruments) != union_keys:
        unknown.append(
            "corporate-action evidence population does not match historical instrument union: "
            f"expected {union_keys}, got {ca.covered_instruments}"
        )

    if unknown:
        raise HistoricalAcquisitionEvidenceError(tuple(sorted(set(unknown))))

    exclusions: list[AcquisitionExclusion] = [
        AcquisitionExclusion(
            trade_date=day,
            instrument_key=None,
            reason_code="NSE_HOLIDAY",
            detail="normal-session acquisition excluded by dated NSE calendar holiday evidence",
        )
        for day in holidays
    ]
    exclusions.extend(
        AcquisitionExclusion(
            trade_date=day,
            instrument_key=None,
            reason_code="SPECIAL_SESSION_EXCLUDED",
            detail=(
                "special session excluded because no normal-session acquisition rule is inferred; "
                "explicit special-session rules are required"
            ),
        )
        for day in special_dates
    )
    exclusions.extend(
        AcquisitionExclusion(
            trade_date=record.trade_date,
            instrument_key=record.instrument_key,
            reason_code="DATED_MEMBERSHIP_NOT_ELIGIBLE",
            detail=(
                "historical record is retained as evidence but excluded from acquisition on its "
                "dated status: listed_on_nse="
                f"{record.listed_on_nse}, normal_equity={record.normal_equity}, "
                f"tradeable_in_normal_market={record.tradeable_in_normal_market}"
            ),
        )
        for record in records_tuple
        if not record.eligible
    )
    exclusions.sort(
        key=lambda item: (
            item.trade_date or date.min,
            item.instrument_key or "",
            item.reason_code,
        )
    )

    requested: list[RequestedAcquisitionInterval] = []
    records_by_key: dict[str, tuple[date, ...]] = {}
    for item in union:
        records_by_key[item.instrument_key] = item.required_trade_dates
    for instrument_key in sorted(records_by_key):
        for run in _contiguous_runs(
            normal_dates=normal_dates,
            eligible_dates=records_by_key[instrument_key],
        ):
            for interval_minutes in intervals:
                expected_rows = sum(
                    sessions_by_date[day].expected_rows(interval_minutes) for day in run
                )
                requested.append(
                    RequestedAcquisitionInterval(
                        instrument_key=instrument_key,
                        interval_minutes=interval_minutes,
                        start=run[0],
                        end=run[-1],
                        trade_dates=run,
                        expected_rows=expected_rows,
                        estimated_storage_bytes=expected_rows * estimated_bytes_per_row,
                        request_count=_historical_request_count(
                            start=run[0],
                            end=run[-1],
                            interval_minutes=interval_minutes,
                        ),
                    )
                )
    requested_tuple = tuple(
        sorted(
            requested,
            key=lambda item: (item.instrument_key, item.interval_minutes, item.start, item.end),
        )
    )
    if not requested_tuple:
        raise HistoricalAcquisitionEvidenceError(["no historically eligible instrument dates"])

    scenario_tuple = tuple(capital_scenarios)
    plan = HistoricalAcquisitionPlan(
        schema_version=HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION,
        mode="DRY_RUN",
        research_start=research_start,
        research_end=research_end,
        calendar_source_id=calendar_source_id,
        normal_trading_dates=normal_dates,
        holiday_dates=holidays,
        excluded_special_session_dates=special_dates,
        evidence_sources=tuple(sorted(sources.values(), key=lambda item: item.source_id)),
        membership_coverage=coverage,
        pit_membership_evidence=records_tuple,
        session_evidence=sessions,
        corporate_action_evidence=ca,
        instrument_identity_union=union,
        requested_intervals=requested_tuple,
        requested_interval_minutes=intervals,
        expected_request_count=sum(item.request_count for item in requested_tuple),
        estimated_rows=sum(item.expected_rows for item in requested_tuple),
        estimated_storage_bytes=sum(item.estimated_storage_bytes for item in requested_tuple),
        estimated_bytes_per_row=estimated_bytes_per_row,
        missing_unknown_evidence=(),
        acquisition_exclusions=tuple(exclusions),
        capital_scenarios=scenario_tuple,
        universe_policy_id=universe_policy_id,
        live_orders_called=False,
    )
    return plan


def _example_digest(label: str) -> str:
    return hashlib.sha256(f"historical-acquisition-example::{label}".encode()).hexdigest()


def build_example_historical_acquisition_plan() -> HistoricalAcquisitionPlan:
    """Return a deterministic synthetic example for the requested first design window."""

    from .nse_calendar import nse_cm_normal_session_calendar

    start = date(2024, 10, 1)
    end = date(2026, 9, 8)
    calendar = nse_cm_normal_session_calendar(start=start, end=end)
    source_ids = {
        "calendar": "example:nse-cm-calendar:2024-2026",
        "membership": "example:nse-mii:dated-snapshots",
        "semantics": "example:nse-cm-semantics:v15",
        "sessions": "example:nse-session-policy:normal-and-special",
        "corporate_actions": "example:corporate-actions:full-window",
    }
    sources = tuple(
        EvidenceSourceIdentity(
            source_id=source_id,
            evidence_type=evidence_type,
            fingerprint=_example_digest(source_id),
        )
        for source_id, evidence_type in (
            (source_ids["calendar"], "nse-calendar"),
            (source_ids["membership"], "dated-nse-membership"),
            (source_ids["semantics"], "nse-mii-semantics"),
            (source_ids["sessions"], "nse-sessions"),
            (source_ids["corporate_actions"], "corporate-actions"),
        )
    )
    alpha = "NSE_EQ|INE000A01010"
    beta = "NSE_EQ|INE001A01010"
    gamma = "NSE_EQ|INE002A01018"
    records: list[PITMembershipEvidence] = []
    for day in calendar.trading_dates:
        candidates: list[tuple[str, str, str, bool]] = []
        if day <= date(2025, 12, 31):
            candidates.append((alpha, "ALPHA", "ALPHA INDUSTRIES", True))
        if day >= date(2025, 1, 2):
            candidates.append((beta, "BETA", "BETA INDUSTRIES", day != date(2025, 3, 3)))
        if day >= date(2026, 8, 3):
            candidates.append((gamma, "GAMMA", "GAMMA INDUSTRIES", True))
        for instrument_key, symbol, name, tradeable in candidates:
            isin = instrument_key.split("|", 1)[1]
            records.append(
                PITMembershipEvidence(
                    trade_date=day,
                    instrument_key=instrument_key,
                    isin=isin,
                    symbol=symbol,
                    name=name,
                    series="EQ",
                    listed_on_nse=True,
                    normal_equity=True,
                    tradeable_in_normal_market=tradeable,
                    tick_size_rupees=Decimal("0.05"),
                    evidence_as_of=day,
                    membership_source_id=source_ids["membership"],
                    identity_source_id=source_ids["membership"],
                    status_source_id=source_ids["semantics"],
                    tick_source_id=source_ids["membership"],
                )
            )
    records = sorted(records, key=lambda item: (item.trade_date, item.instrument_key))
    counts: dict[date, int] = {}
    for record in records:
        counts[record.trade_date] = counts.get(record.trade_date, 0) + 1
    coverage = tuple(
        PITMembershipCoverage(
            trade_date=day,
            source_id=source_ids["membership"],
            record_count=counts.get(day, 0),
        )
        for day in calendar.trading_dates
    )
    sessions = [
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
    ]
    sessions.extend(
        SessionEvidence(
            trade_date=day,
            session_kind="SPECIAL",
            source_id=source_ids["sessions"],
            timezone="Asia/Kolkata",
            start_time="00:00",
            end_time_exclusive="00:01",
            expected_rows_by_interval=(),
            acquisition_allowed=False,
            exclusion_reason="special session requires an explicit separate acquisition rule",
        )
        for day in calendar.excluded_special_session_dates
    )
    ca = CorporateActionEvidence(
        source_id=source_ids["corporate_actions"],
        fingerprint=_example_digest(source_ids["corporate_actions"]),
        coverage_start=start,
        coverage_end=end,
        covered_instruments=(alpha, beta, gamma),
        complete=True,
        policy_identity="example:raw-unadjusted-block-structural-actions:v1",
        events_count=0,
    )
    return build_historical_acquisition_plan(
        research_start=start,
        research_end=end,
        calendar=calendar,
        calendar_source_id=source_ids["calendar"],
        evidence_sources=sources,
        membership_coverage=coverage,
        pit_membership_evidence=records,
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
            CapitalScenarioAnalysis(
                scenario_id="capital-10000-inr",
                capital_rupees=Decimal(10000),
                note="Analysis only; does not remove any historically required instrument.",
            ),
        ),
    )


__all__ = [
    "HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION",
    "AcquisitionExclusion",
    "CapitalScenarioAnalysis",
    "CorporateActionEvidence",
    "EvidenceSourceIdentity",
    "HistoricalAcquisitionEvidenceError",
    "HistoricalAcquisitionPlan",
    "HistoricalIdentityPoint",
    "HistoricalTickPoint",
    "InstrumentIdentityUnion",
    "PITMembershipCoverage",
    "PITMembershipEvidence",
    "RequestedAcquisitionInterval",
    "SessionEvidence",
    "build_example_historical_acquisition_plan",
    "build_historical_acquisition_plan",
]
