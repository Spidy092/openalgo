from datetime import date

import pytest

from equity_engine.nse_trading_calendar import (
    NseTradingCalendar,
    validate_initial_research_boundary,
)


def test_calendar_excludes_weekends_and_full_day_holidays_but_keeps_muhurat_date() -> None:
    calendar = NseTradingCalendar()
    dates = calendar.trading_dates(date(2024, 7, 13), date(2024, 7, 19))

    assert dates == (date(2024, 7, 15), date(2024, 7, 16), date(2024, 7, 18), date(2024, 7, 19))
    assert calendar.is_trading_date(date(2024, 11, 1)) is True
    assert calendar.is_trading_date(date(2024, 11, 15)) is False


def test_calendar_can_represent_an_off_weekday_special_session() -> None:
    calendar = NseTradingCalendar(
        closed_dates=frozenset(),
        source_urls=("https://example.test/nse-holidays",),
        additional_trading_dates=frozenset({date(2024, 7, 13)}),
    )
    assert calendar.trading_dates(date(2024, 7, 13), date(2024, 7, 14)) == (date(2024, 7, 13),)


def test_initial_research_boundary_rejects_pre_v15_and_cas_regime() -> None:
    with pytest.raises(ValueError, match="not verified"):
        validate_initial_research_boundary(date(2024, 6, 28), date(2024, 7, 1))
    with pytest.raises(ValueError, match="August 2026 CAS"):
        validate_initial_research_boundary(date(2024, 7, 1), date(2026, 8, 3))
