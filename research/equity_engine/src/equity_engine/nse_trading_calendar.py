from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

NSE_RESEARCH_START = date(2024, 7, 1)
NSE_INITIAL_RESEARCH_END = date(2026, 7, 31)

# These are the NSE Capital Market full-day closures.  Diwali Laxmi Pujan/Muhurat dates are
# deliberately not included: they are special trading sessions, not missing market days.
NSE_HOLIDAY_SOURCES = {
    2024: "https://nsearchives.nseindia.com/content/circulars/CMTR59722.pdf",
    2025: "https://nsearchives.nseindia.com/content/circulars/CMTR65587.pdf",
    2026: "https://nsearchives.nseindia.com/content/circulars/CMTR71775.pdf",
}

NSE_CAPITAL_MARKET_CLOSED_DATES = frozenset(
    {
        # 2024
        date(2024, 7, 17),
        date(2024, 8, 15),
        date(2024, 10, 2),
        date(2024, 11, 15),
        date(2024, 12, 25),
        # 2025
        date(2025, 2, 26),
        date(2025, 3, 14),
        date(2025, 3, 31),
        date(2025, 4, 10),
        date(2025, 4, 14),
        date(2025, 4, 18),
        date(2025, 5, 1),
        date(2025, 8, 15),
        date(2025, 8, 27),
        date(2025, 10, 2),
        date(2025, 10, 22),
        date(2025, 11, 5),
        date(2025, 12, 25),
        # 2026, through the initial research boundary and beyond for safe reuse.
        date(2026, 1, 26),
        date(2026, 3, 3),
        date(2026, 3, 26),
        date(2026, 3, 31),
        date(2026, 4, 3),
        date(2026, 4, 14),
        date(2026, 5, 1),
        date(2026, 5, 28),
        date(2026, 6, 26),
        date(2026, 9, 14),
        date(2026, 10, 2),
        date(2026, 10, 20),
        date(2026, 11, 10),
        date(2026, 11, 24),
        date(2026, 12, 25),
    }
)


@dataclass(frozen=True)
class NseTradingCalendar:
    """Official NSE cash-market date policy for point-in-time acquisition.

    Saturdays and Sundays are excluded by default.  The holiday list is explicit and sourced
    from NSE's yearly capital-market circulars, so an unavailable archive on a weekday cannot be
    silently reclassified as a holiday.
    """

    closed_dates: frozenset[date] = NSE_CAPITAL_MARKET_CLOSED_DATES
    source_urls: tuple[str, ...] = tuple(NSE_HOLIDAY_SOURCES[year] for year in (2024, 2025, 2026))
    additional_trading_dates: frozenset[date] = frozenset()

    def __post_init__(self) -> None:
        if self.closed_dates.intersection(self.additional_trading_dates):
            raise ValueError("a date cannot be both closed and an additional trading date")
        if any(not isinstance(item, date) for item in self.closed_dates):
            raise TypeError("closed_dates must contain date values")
        if any(not isinstance(item, date) for item in self.additional_trading_dates):
            raise TypeError("additional_trading_dates must contain date values")
        if not self.source_urls or any(not item.strip() for item in self.source_urls):
            raise ValueError("official holiday source URLs are required")

    def is_trading_date(self, value: date) -> bool:
        if value in self.additional_trading_dates:
            return True
        return value.weekday() < 5 and value not in self.closed_dates

    def trading_dates(self, start: date, end: date) -> tuple[date, ...]:
        if start > end:
            raise ValueError("start must be on or before end")
        dates: list[date] = []
        current = start
        while current <= end:
            if self.is_trading_date(current):
                dates.append(current)
            current += timedelta(days=1)
        if not dates:
            raise ValueError("date range contains no NSE cash-market trading dates")
        return tuple(dates)

    def describe(self) -> dict[str, object]:
        return {
            "kind": "nse_capital_market",
            "source_urls": list(self.source_urls),
            "closed_dates": sorted(item.isoformat() for item in self.closed_dates),
            "additional_trading_dates": sorted(
                item.isoformat() for item in self.additional_trading_dates
            ),
        }


def validate_initial_research_boundary(start: date, end: date) -> None:
    if start < NSE_RESEARCH_START:
        raise ValueError(
            f"historical NSE semantics are not verified before {NSE_RESEARCH_START.isoformat()}"
        )
    if end > NSE_INITIAL_RESEARCH_END:
        raise ValueError(
            "initial research acquisition ends at "
            f"{NSE_INITIAL_RESEARCH_END.isoformat()}; August 2026 CAS semantics are a separate run"
        )


def normalize_trading_dates(values: Iterable[date]) -> tuple[date, ...]:
    result = tuple(sorted(set(values)))
    if not result:
        raise ValueError("at least one trading date is required")
    return result
