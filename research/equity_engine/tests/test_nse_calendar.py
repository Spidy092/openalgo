from datetime import date

import pytest

from equity_engine.nse_calendar import nse_cm_normal_session_calendar


def test_calendar_excludes_sourced_holiday_without_guessing_weekends() -> None:
    result = nse_cm_normal_session_calendar(
        start=date(2024, 7, 15),
        end=date(2024, 7, 19),
    )
    assert result.trading_dates == (
        date(2024, 7, 15),
        date(2024, 7, 16),
        date(2024, 7, 18),
        date(2024, 7, 19),
    )
    assert result.holiday_dates == (date(2024, 7, 17),)


def test_calendar_excludes_muhurat_from_normal_session_research() -> None:
    result = nse_cm_normal_session_calendar(
        start=date(2024, 11, 1),
        end=date(2024, 11, 1),
    )
    assert result.trading_dates == ()
    assert result.excluded_special_session_dates == (date(2024, 11, 1),)


def test_calendar_detects_weekend_special_session_before_weekday_filter() -> None:
    result = nse_cm_normal_session_calendar(
        start=date(2026, 11, 8),
        end=date(2026, 11, 8),
    )
    assert result.trading_dates == ()
    assert result.excluded_special_session_dates == (date(2026, 11, 8),)


def test_calendar_fails_closed_for_unsupported_year() -> None:
    with pytest.raises(ValueError, match="no verified NSE Capital Market holiday calendar"):
        nse_cm_normal_session_calendar(
            start=date(2027, 1, 1),
            end=date(2027, 1, 10),
        )
