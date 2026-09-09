from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Mapping, Protocol

import pandas as pd


NSE_CAS_EFFECTIVE_DATE = date(2026, 8, 3)
NSE_NORMAL_CONTINUOUS_START = time(9, 15)
NSE_NORMAL_CONTINUOUS_END = time(15, 30)
NSE_CAS_CONTINUOUS_END = time(15, 15)

NSE_CAS_SOURCE = "https://www.nseindia.com/static/products-services/closing-auction-session"


class ContinuousSessionPolicy(Protocol):
    def continuous_start(self, trade_date: date) -> time:
        """Return the effective continuous-session start for an instrument/date."""

    def continuous_end(self, trade_date: date) -> time:
        """Return the effective continuous-session end for an instrument/date."""


@dataclass(frozen=True)
class NSEEquitySessionPolicy:
    """Resolve the last allowed entry/holding time without guessing one universal close.

    Phase-1 CAS applies to NSE cash stocks with derivative contracts from 2026-08-03. This
    research policy deliberately avoids participating in CAS: positions are scheduled to leave
    during continuous trading before 15:15 for CAS-eligible stocks, or before 15:30 otherwise.

    `exit_buffer_minutes` is mandatory because execution safety margin is a research parameter,
    not an exchange rule. Special-session overrides must be supplied explicitly when required.
    """

    cas_eligible: bool
    exit_buffer_minutes: int
    special_session_continuous_end: Mapping[date, time] = field(default_factory=dict)
    special_session_continuous_start: Mapping[date, time] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.exit_buffer_minutes < 0:
            raise ValueError("exit_buffer_minutes cannot be negative")
        if self.exit_buffer_minutes >= 60:
            raise ValueError("exit_buffer_minutes must be below 60 for the current NSE policy")

    def continuous_start(self, trade_date: date) -> time:
        return self.special_session_continuous_start.get(trade_date, NSE_NORMAL_CONTINUOUS_START)

    def continuous_end(self, trade_date: date) -> time:
        override = self.special_session_continuous_end.get(trade_date)
        if override is not None:
            return override
        if trade_date >= NSE_CAS_EFFECTIVE_DATE and self.cas_eligible:
            return NSE_CAS_CONTINUOUS_END
        return NSE_NORMAL_CONTINUOUS_END

    def exit_time(self, trade_date: date) -> time:
        end = self.continuous_end(trade_date)
        anchor = datetime.combine(trade_date, end)
        return (anchor - timedelta(minutes=self.exit_buffer_minutes)).time()


def filter_to_continuous_session(
    frame: pd.DataFrame,
    session_policy: ContinuousSessionPolicy,
    *,
    session_start: time | None = None,
) -> pd.DataFrame:
    """Return only bars inside the effective continuous session for each date.

    This is the single market-data boundary used before strategy signals, screening, and exact
    execution simulation. It intentionally preserves the input order and all columns while
    excluding CAS/transition bars and other out-of-session timestamps.
    """

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("continuous-session filtering requires a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("continuous-session filtering requires timezone-aware timestamps")

    keep = [
        (
            session_start
            if session_start is not None
            else session_policy.continuous_start(timestamp.date())
        )
        <= timestamp.time()
        < session_policy.continuous_end(timestamp.date())
        for timestamp in frame.index
    ]
    return frame.loc[keep].copy()
