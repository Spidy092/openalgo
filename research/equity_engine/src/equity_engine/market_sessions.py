from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Mapping


NSE_CAS_EFFECTIVE_DATE = date(2026, 8, 3)
NSE_NORMAL_CONTINUOUS_END = time(15, 30)
NSE_CAS_CONTINUOUS_END = time(15, 15)

NSE_CAS_SOURCE = "https://www.nseindia.com/static/products-services/closing-auction-session"


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

    def __post_init__(self) -> None:
        if self.exit_buffer_minutes < 0:
            raise ValueError("exit_buffer_minutes cannot be negative")
        if self.exit_buffer_minutes >= 60:
            raise ValueError("exit_buffer_minutes must be below 60 for the current NSE policy")

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
