from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class CalendarEvidence:
    trading_dates: tuple[date, ...]
    holiday_dates: tuple[date, ...]
    excluded_special_session_dates: tuple[date, ...]
    source_urls: tuple[str, ...]


# Primary NSE Capital Market circulars. The 2026 list has a later amendment for
# the January 15 Maharashtra municipal-election holiday.
NSE_CM_HOLIDAY_SOURCES = {
    2024: (
        "https://nsearchives.nseindia.com/content/circulars/CMTR59722.pdf",
        "https://nsearchives.nseindia.com/content/circulars/CMTR61518.pdf",
    ),
    2025: (
        "https://nsearchives.nseindia.com/content/circulars/CMTR65587.pdf",
    ),
    2026: (
        "https://nsearchives.nseindia.com/content/circulars/CMTR71775.pdf",
        "https://nsearchives.nseindia.com/content/circulars/CMTR72260.pdf",
    ),
}

# Capital Market trading holidays. These values are intentionally effective-dated
# evidence, not a generic Indian holiday calendar.
_NSE_CM_HOLIDAYS = {
    2024: frozenset(
        {
            date(2024, 1, 26),
            date(2024, 3, 8),
            date(2024, 3, 25),
            date(2024, 3, 29),
            date(2024, 4, 11),
            date(2024, 4, 17),
            date(2024, 5, 1),
            date(2024, 5, 20),
            date(2024, 6, 17),
            date(2024, 7, 17),
            date(2024, 8, 15),
            date(2024, 10, 2),
            date(2024, 11, 1),
            date(2024, 11, 15),
            date(2024, 12, 25),
        }
    ),
    2025: frozenset(
        {
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
            date(2025, 10, 21),
            date(2025, 10, 22),
            date(2025, 11, 5),
            date(2025, 12, 25),
        }
    ),
    2026: frozenset(
        {
            date(2026, 1, 15),
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
    ),
}

# These dates are exchange holidays with a separately announced Muhurat/special trading
# session. The first normal-session research pass excludes them rather than pretending their
# shortened session is a standard day. They can be added later only with explicit session rules.
_NSE_CM_EXCLUDED_SPECIAL_SESSIONS = frozenset(
    {
        date(2024, 11, 1),
        date(2025, 10, 21),
        date(2026, 11, 8),
    }
)


def nse_cm_normal_session_calendar(*, start: date, end: date) -> CalendarEvidence:
    """Return sourced NSE CM normal-session dates for the supported research years.

    This is deliberately not a perpetual calendar. Unknown years fail closed so a future run
    cannot silently inherit stale holiday assumptions.
    """

    if start > end:
        raise ValueError("start must be on or before end")
    years = set(range(start.year, end.year + 1))
    unsupported = sorted(year for year in years if year not in _NSE_CM_HOLIDAYS)
    if unsupported:
        raise ValueError(
            "no verified NSE Capital Market holiday calendar for years: "
            + ", ".join(str(year) for year in unsupported)
        )

    trading: list[date] = []
    holidays: list[date] = []
    special: list[date] = []
    current = start
    while current <= end:
        if current in _NSE_CM_EXCLUDED_SPECIAL_SESSIONS:
            special.append(current)
        elif current.weekday() < 5:
            if current in _NSE_CM_HOLIDAYS[current.year]:
                holidays.append(current)
            else:
                trading.append(current)
        current += timedelta(days=1)

    sources = tuple(
        source
        for year in sorted(years)
        for source in NSE_CM_HOLIDAY_SOURCES[year]
    )
    return CalendarEvidence(
        trading_dates=tuple(trading),
        holiday_dates=tuple(holidays),
        excluded_special_session_dates=tuple(special),
        source_urls=sources,
    )
