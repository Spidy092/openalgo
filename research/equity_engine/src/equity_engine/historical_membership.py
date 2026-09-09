from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class HistoricalTradingStatus:
    """Point-in-time exchange eligibility evidence for one security and trade date."""

    trade_date: date
    instrument_key: str
    listed_on_nse: bool
    normal_equity: bool
    tradeable_in_normal_market: bool
    source: str

    @property
    def eligible(self) -> bool:
        return self.listed_on_nse and self.normal_equity and self.tradeable_in_normal_market


@dataclass(frozen=True)
class HistoricalMembershipAssessment:
    instrument_key: str
    requested_dates: tuple[date, ...]
    eligible_dates: tuple[date, ...]
    ineligible_dates: tuple[date, ...]
    missing_dates: tuple[date, ...]
    source_refs: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_dates


class HistoricalTradingEligibilityPolicy:
    """Per-day entry permission derived only from point-in-time exchange evidence."""

    def __init__(self, assessment: HistoricalMembershipAssessment) -> None:
        if not assessment.complete:
            raise ValueError(
                "cannot build trading eligibility policy from incomplete membership evidence"
            )
        self._instrument_key = assessment.instrument_key
        self._requested = frozenset(assessment.requested_dates)
        self._eligible = frozenset(assessment.eligible_dates)

    @property
    def instrument_key(self) -> str:
        return self._instrument_key

    def is_eligible(self, trade_date: date) -> bool:
        if trade_date not in self._requested:
            raise ValueError(
                f"no point-in-time trading eligibility evidence for {trade_date}"
            )
        return trade_date in self._eligible


def assess_historical_membership(
    *,
    instrument_key: str,
    trading_dates: Iterable[date],
    statuses: Iterable[HistoricalTradingStatus],
) -> HistoricalMembershipAssessment:
    """Require point-in-time evidence for every requested trading date."""

    if not instrument_key:
        raise ValueError("instrument_key is required")
    requested = tuple(sorted(set(trading_dates)))
    if not requested:
        raise ValueError("at least one trading date is required")

    by_date: dict[date, HistoricalTradingStatus] = {}
    sources: set[str] = set()
    for status in statuses:
        if status.instrument_key != instrument_key:
            raise ValueError("historical status belongs to a different instrument")
        if status.trade_date in by_date:
            raise ValueError(f"duplicate point-in-time status for {status.trade_date}")
        if not status.source.strip():
            raise ValueError("historical status source is required")
        by_date[status.trade_date] = status
        sources.add(status.source)

    missing = tuple(day for day in requested if day not in by_date)
    eligible = tuple(day for day in requested if day in by_date and by_date[day].eligible)
    ineligible = tuple(day for day in requested if day in by_date and not by_date[day].eligible)
    return HistoricalMembershipAssessment(
        instrument_key=instrument_key,
        requested_dates=requested,
        eligible_dates=eligible,
        ineligible_dates=ineligible,
        missing_dates=missing,
        source_refs=tuple(sorted(sources)),
    )


def filter_frame_to_eligible_dates(
    frame: pd.DataFrame,
    assessment: HistoricalMembershipAssessment,
) -> pd.DataFrame:
    """Return only dates known eligible; incomplete point-in-time evidence fails closed."""

    if not assessment.complete:
        raise ValueError(
            "historical membership evidence is incomplete for dates: "
            + ", ".join(day.isoformat() for day in assessment.missing_dates)
        )
    if frame.index.tz is None:
        raise ValueError("historical frame requires timezone-aware timestamps")
    frame_dates = set(frame.index.date)
    if not frame_dates.issubset(set(assessment.requested_dates)):
        raise ValueError("frame contains dates outside membership assessment")
    eligible = set(assessment.eligible_dates)
    return frame[[day in eligible for day in frame.index.date]].copy()
