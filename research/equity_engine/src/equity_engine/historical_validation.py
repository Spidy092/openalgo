from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Mapping

import pandas as pd

from .market_sessions import (
    NSE_CAS_EFFECTIVE_DATE,
    NSE_NORMAL_CONTINUOUS_START,
    NSEEquitySessionPolicy,
)
from .nse_calendar import CalendarEvidence
from .provenance import MarketDataManifest, dataframe_fingerprint, validate_ohlcv_frame


@dataclass(frozen=True)
class IntradaySessionRule:
    """Explicit expected candle schedule for one trading date.

    `end_time` is exclusive, matching exchange session semantics: a normal NSE rule ending at
    15:30 expects a final 5-minute candle starting at 15:25. Dates without a rule are not
    inferred to be normal trading days.
    """

    rule_id: str
    timezone: str
    start_time: time
    end_time: time
    interval_minutes: int
    source_reference: str

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.timezone.strip():
            raise ValueError("session rule identity and timezone are required")
        if self.start_time >= self.end_time:
            raise ValueError("session rule start_time must be before end_time")
        if self.interval_minutes < 1:
            raise ValueError("session rule interval_minutes must be positive")
        if not self.source_reference.strip():
            raise ValueError("session rule source_reference is required")

    def expected_timestamps(self, trade_date: date) -> pd.DatetimeIndex:
        start = pd.Timestamp(
            datetime.combine(trade_date, self.start_time),
            tz=self.timezone,
        )
        end = pd.Timestamp(
            datetime.combine(trade_date, self.end_time),
            tz=self.timezone,
        )
        return pd.date_range(
            start=start,
            end=end,
            freq=f"{self.interval_minutes}min",
            inclusive="left",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "timezone": self.timezone,
            "start_time": self.start_time.isoformat(),
            "end_time_exclusive": self.end_time.isoformat(),
            "interval_minutes": self.interval_minutes,
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True)
class DailyIntradayValidation:
    trade_date: date
    row_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    duplicate_count: int
    missing_expected_slots: tuple[str, ...]
    unexpected_timestamps: tuple[str, ...]
    timezone: str | None
    ohlcv_violations: tuple[str, ...]
    session_rule: dict[str, object] | None

    @property
    def passed(self) -> bool:
        return not (
            self.duplicate_count
            or self.missing_expected_slots
            or self.unexpected_timestamps
            or self.ohlcv_violations
            or self.session_rule is None
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "date": self.trade_date.isoformat(),
            "row_count": self.row_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "duplicate_count": self.duplicate_count,
            "missing_expected_slots": list(self.missing_expected_slots),
            "missing_expected_5_minute_slots": list(self.missing_expected_slots),
            "unexpected_timestamps": list(self.unexpected_timestamps),
            "timezone": self.timezone,
            "ohlcv_violations": list(self.ohlcv_violations),
            "session_rule": self.session_rule,
            "status": "PASS" if self.passed else "FAIL",
        }


@dataclass(frozen=True)
class HistoricalDatasetValidation:
    rows: int
    trading_dates: tuple[date, ...]
    per_day: tuple[DailyIntradayValidation, ...]
    structural_violations: tuple[str, ...]
    deterministic_data_fingerprint: str
    manifest_fingerprint_reference: str | None
    manifest_reference: str | None
    timezone: str | None
    calendar_evidence: dict[str, object] | None

    @property
    def passed(self) -> bool:
        return not self.structural_violations and all(item.passed for item in self.per_day)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "PASS" if self.passed else "FAIL",
            "rows": self.rows,
            "trading_dates": [item.isoformat() for item in self.trading_dates],
            "per_day": [item.as_dict() for item in self.per_day],
            "structural_violations": list(self.structural_violations),
            "deterministic_data_fingerprint": self.deterministic_data_fingerprint,
            "manifest_fingerprint_reference": self.manifest_fingerprint_reference,
            "manifest_reference": self.manifest_reference,
            "timezone": self.timezone,
            "calendar_evidence": self.calendar_evidence,
            "strategy_ready": False,
            "live_orders_called": False,
        }


def nse_session_rules_for_calendar(
    calendar: CalendarEvidence,
    *,
    timezone: str,
    interval_minutes: int,
    cas_eligible: bool,
    special_session_rules: Mapping[date, IntradaySessionRule] | None = None,
) -> dict[date, IntradaySessionRule]:
    """Build explicit rules from the sourced NSE calendar and effective-dated CAS policy.

    Normal-calendar holidays and excluded special sessions deliberately produce no rule. If data
    contains one of those dates, validation reports it as an undeclared/unexpected date instead of
    treating it as a missing normal session.
    """

    policy = NSEEquitySessionPolicy(cas_eligible=cas_eligible, exit_buffer_minutes=0)
    rules: dict[date, IntradaySessionRule] = {}
    for trade_date in calendar.trading_dates:
        end_time = policy.continuous_end(trade_date)
        rule_id = (
            "nse-cm-cas-continuous-session"
            if cas_eligible and trade_date >= NSE_CAS_EFFECTIVE_DATE
            else "nse-cm-normal-continuous-session"
        )
        rules[trade_date] = IntradaySessionRule(
            rule_id=rule_id,
            timezone=timezone,
            start_time=NSE_NORMAL_CONTINUOUS_START,
            end_time=end_time,
            interval_minutes=interval_minutes,
            source_reference="nse_cm_normal_session_calendar + NSEEquitySessionPolicy",
        )
    if special_session_rules:
        rules.update(special_session_rules)
    return rules


def _timestamp_text(value: object) -> str:
    return str(pd.Timestamp(value))


def validate_intraday_dataset(
    frame: pd.DataFrame,
    manifest: MarketDataManifest,
    *,
    session_rules: Mapping[date, IntradaySessionRule],
    manifest_fingerprint_reference: str | None = None,
    manifest_reference: str | None = None,
    calendar_evidence: CalendarEvidence | None = None,
) -> HistoricalDatasetValidation:
    """Validate a downloaded intraday dataset without network or broker access."""

    structural: list[str] = []
    if not isinstance(frame.index, pd.DatetimeIndex):
        structural.append("dataset index is not a DatetimeIndex")
        frame_tz: str | None = None
        frame_dates: set[date] = set()
    else:
        frame_tz = str(frame.index.tz) if frame.index.tz is not None else None
        frame_dates = set(frame.index.date)
        if frame.index.tz is None:
            structural.append("dataset timestamps are timezone-naive")
        if frame_tz != manifest.timezone:
            structural.append(
                f"dataset timezone {frame_tz!r} does not match manifest timezone {manifest.timezone!r}"
            )
        structural.extend(validate_ohlcv_frame(frame))

    declared_dates = set(session_rules)
    undeclared_dates = sorted(frame_dates.difference(declared_dates))
    if undeclared_dates:
        structural.append(
            "data contains dates without an explicit session rule: "
            + ", ".join(item.isoformat() for item in undeclared_dates)
        )

    report_dates = sorted(frame_dates.union(declared_dates))
    daily_reports: list[DailyIntradayValidation] = []
    for trade_date in report_dates:
        if isinstance(frame.index, pd.DatetimeIndex):
            date_mask = frame.index.date == trade_date
            day_frame = frame.loc[date_mask]
        else:
            day_frame = frame.iloc[0:0]

        rule = session_rules.get(trade_date)
        actual_index = (
            pd.DatetimeIndex(day_frame.index) if rule is not None else pd.DatetimeIndex([])
        )
        expected_index = (
            rule.expected_timestamps(trade_date) if rule is not None else pd.DatetimeIndex([])
        )
        actual_unique = actual_index.unique()
        missing = expected_index.difference(actual_unique)
        unexpected = actual_unique.difference(expected_index)
        first = _timestamp_text(day_frame.index[0]) if len(day_frame) else None
        last = _timestamp_text(day_frame.index[-1]) if len(day_frame) else None
        timezone = (
            str(day_frame.index.tz)
            if isinstance(day_frame.index, pd.DatetimeIndex) and day_frame.index.tz is not None
            else None
        )
        ohlcv = (
            validate_ohlcv_frame(day_frame) if len(day_frame) else ["no candles for declared date"]
        )
        daily_reports.append(
            DailyIntradayValidation(
                trade_date=trade_date,
                row_count=len(day_frame),
                first_timestamp=first,
                last_timestamp=last,
                duplicate_count=int(day_frame.index.duplicated(keep="first").sum()),
                missing_expected_slots=tuple(_timestamp_text(item) for item in missing),
                unexpected_timestamps=tuple(_timestamp_text(item) for item in unexpected),
                timezone=timezone,
                ohlcv_violations=tuple(ohlcv),
                session_rule=rule.as_dict() if rule is not None else None,
            )
        )

    try:
        fingerprint = dataframe_fingerprint(frame, manifest)
    except ValueError as exc:
        fingerprint = ""
        structural.append(f"cannot compute deterministic data fingerprint: {exc}")

    if manifest_fingerprint_reference is not None and fingerprint:
        if fingerprint != manifest_fingerprint_reference:
            structural.append(
                "manifest fingerprint does not match the deterministic data fingerprint; "
                "artifact was produced by a different fingerprint schema or was changed"
            )
    elif manifest_fingerprint_reference is None:
        structural.append("manifest has no fingerprint_sha256 reference")

    calendar_payload = None
    if calendar_evidence is not None:
        calendar_payload = {
            "holiday_dates": [item.isoformat() for item in calendar_evidence.holiday_dates],
            "excluded_special_session_dates": [
                item.isoformat() for item in calendar_evidence.excluded_special_session_dates
            ],
            "source_urls": list(calendar_evidence.source_urls),
        }

    return HistoricalDatasetValidation(
        rows=len(frame),
        trading_dates=tuple(report_dates),
        per_day=tuple(daily_reports),
        structural_violations=tuple(structural),
        deterministic_data_fingerprint=fingerprint,
        manifest_fingerprint_reference=manifest_fingerprint_reference,
        manifest_reference=manifest_reference,
        timezone=frame_tz,
        calendar_evidence=calendar_payload,
    )
