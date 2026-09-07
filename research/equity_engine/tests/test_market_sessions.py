from datetime import date, time

from equity_engine.market_sessions import NSEEquitySessionPolicy


def test_pre_cas_stock_keeps_normal_continuous_close() -> None:
    policy = NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=5)
    assert policy.exit_time(date(2026, 8, 2)) == time(15, 25)


def test_cas_eligible_stock_uses_1515_continuous_end_after_effective_date() -> None:
    policy = NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=5)
    assert policy.exit_time(date(2026, 8, 3)) == time(15, 10)


def test_non_cas_stock_remains_on_1530_continuous_session() -> None:
    policy = NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=5)
    assert policy.exit_time(date(2026, 9, 7)) == time(15, 25)


def test_special_session_override_wins() -> None:
    special = date(2026, 10, 20)
    policy = NSEEquitySessionPolicy(
        cas_eligible=False,
        exit_buffer_minutes=5,
        special_session_continuous_end={special: time(13, 0)},
    )
    assert policy.exit_time(special) == time(12, 55)
