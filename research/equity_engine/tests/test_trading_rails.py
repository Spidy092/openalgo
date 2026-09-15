"""Tests for the safety-rails library (Agent 1, Phase 2).

Every rail is pure: no broker calls, no network, no order placement. Each check
returns an explicit allow/deny verdict and denies on unknown inputs
(fail-closed). Boundary cases are pinned: exactly-at-cap passes, one paisa over
fails.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.trading_rails import (
    IdempotencyRegistry,
    RateLimiter,
    TokenProbeResult,
    check_daily_loss_limit,
    check_instrument_allowlist,
    check_kill_switch,
    check_per_order_notional,
    check_session_gate,
    check_token_validity,
    check_total_capital,
    evaluate_all_rails,
    generate_idempotency_key,
)

IST = ZoneInfo("Asia/Kolkata")


def _policy(buffer_minutes: int = 15) -> NSEEquitySessionPolicy:
    return NSEEquitySessionPolicy(cas_eligible=False, exit_buffer_minutes=buffer_minutes)


def test_total_capital_exactly_at_cap_is_allowed() -> None:
    verdict = check_total_capital(
        deployed_rupees=Decimal(1000),
        approved_capital_rupees=Decimal(1000),
    )
    assert verdict.allowed
    assert verdict.code == "ok"


def test_total_capital_one_paisa_over_is_denied() -> None:
    verdict = check_total_capital(
        deployed_rupees=Decimal("1000.01"),
        approved_capital_rupees=Decimal(1000),
    )
    assert not verdict.allowed
    assert verdict.code == "capital_cap_exceeded"


def test_total_capital_over_approved_but_under_hard_cap_is_denied() -> None:
    verdict = check_total_capital(
        deployed_rupees=Decimal(600),
        approved_capital_rupees=Decimal(500),
    )
    assert not verdict.allowed


@pytest.mark.parametrize("unknown", [None, {"amount": 100}, float("nan"), True, Decimal("NaN")])
def test_total_capital_unknown_input_is_denied(unknown) -> None:
    assert not check_total_capital(unknown, Decimal(1000)).allowed
    assert not check_total_capital(Decimal(100), unknown).allowed


def test_total_capital_accepts_numeric_strings() -> None:
    assert check_total_capital("1000", "1000").allowed


def test_total_capital_negative_deployed_is_denied() -> None:
    assert not check_total_capital(Decimal(-1), Decimal(1000)).allowed


def test_per_order_notional_exactly_at_cap_is_allowed() -> None:
    assert check_per_order_notional(Decimal(1000), Decimal(1000)).allowed


def test_per_order_notional_one_paisa_over_is_denied() -> None:
    verdict = check_per_order_notional(Decimal("1000.01"), Decimal(1000))
    assert not verdict.allowed
    assert verdict.code == "per_order_cap_exceeded"


@pytest.mark.parametrize("bad", [None, Decimal(0), Decimal(-5), True, float("nan")])
def test_per_order_notional_invalid_is_denied(bad) -> None:
    assert not check_per_order_notional(bad, Decimal(1000)).allowed


def test_per_order_notional_accepts_numeric_strings() -> None:
    assert check_per_order_notional("500", "1000").allowed


def test_daily_loss_exactly_at_limit_trips() -> None:
    verdict = check_daily_loss_limit(
        realized_pnl_rupees=Decimal(-150),
        unrealized_pnl_rupees=Decimal(-50),
        daily_loss_limit_rupees=Decimal(200),
    )
    assert not verdict.allowed
    assert verdict.code == "daily_loss_limit_breached"


def test_daily_loss_one_paisa_inside_is_allowed() -> None:
    verdict = check_daily_loss_limit(
        realized_pnl_rupees=Decimal(-150),
        unrealized_pnl_rupees=Decimal("-49.99"),
        daily_loss_limit_rupees=Decimal(200),
    )
    assert verdict.allowed


def test_daily_profit_is_allowed() -> None:
    assert check_daily_loss_limit(Decimal(100), Decimal(50), Decimal(200)).allowed


def test_daily_loss_unknown_input_is_denied() -> None:
    assert not check_daily_loss_limit(None, Decimal(0), Decimal(200)).allowed
    assert not check_daily_loss_limit(Decimal(0), Decimal(0), None).allowed


def test_rate_limiter_allows_up_to_max_then_denies() -> None:
    start = datetime(2026, 9, 15, 9, 30, tzinfo=UTC)
    limiter = RateLimiter(
        max_orders=2,
        window_seconds=60,
        min_interval_seconds=0,
        clock=lambda: start,
    )
    assert limiter.check_and_record().allowed
    assert limiter.check_and_record().allowed
    third = limiter.check_and_record()
    assert not third.allowed
    assert third.code == "rate_limit_exceeded"


def test_rate_limiter_cooldown_blocks_immediate_reorder() -> None:
    ticks = [datetime(2026, 9, 15, 9, 30, tzinfo=UTC)]
    limiter = RateLimiter(
        max_orders=10,
        window_seconds=3600,
        min_interval_seconds=60,
        clock=lambda: ticks[0],
    )
    assert limiter.check_and_record().allowed
    ticks[0] = ticks[0] + timedelta(seconds=1)
    verdict = limiter.check_and_record()
    assert not verdict.allowed
    assert verdict.code == "cooldown_active"


def test_rate_limiter_allows_after_cooldown() -> None:
    ticks = [datetime(2026, 9, 15, 9, 30, tzinfo=UTC)]
    limiter = RateLimiter(
        max_orders=10,
        window_seconds=3600,
        min_interval_seconds=60,
        clock=lambda: ticks[0],
    )
    assert limiter.check_and_record().allowed
    ticks[0] = ticks[0] + timedelta(seconds=60)
    assert limiter.check_and_record().allowed


def test_rate_limiter_window_slides() -> None:
    ticks = [datetime(2026, 9, 15, 9, 30, tzinfo=UTC)]
    limiter = RateLimiter(
        max_orders=1,
        window_seconds=60,
        min_interval_seconds=0,
        clock=lambda: ticks[0],
    )
    assert limiter.check_and_record().allowed
    ticks[0] = ticks[0] + timedelta(seconds=61)
    assert limiter.check_and_record().allowed


def test_idempotency_key_is_deterministic() -> None:
    fields = {"strategy_id": "orb-v1", "symbol": "RELIANCE", "qty": "1"}
    assert generate_idempotency_key(fields) == generate_idempotency_key(fields)
    assert len(generate_idempotency_key(fields)) == 64


def test_idempotency_key_differs_for_different_orders() -> None:
    first = generate_idempotency_key({"strategy_id": "orb-v1", "qty": "1"})
    second = generate_idempotency_key({"strategy_id": "orb-v1", "qty": "2"})
    assert first != second


def test_idempotency_key_rejects_empty_payload() -> None:
    with pytest.raises(ValueError):
        generate_idempotency_key({})


def test_idempotency_registry_denies_duplicate_key() -> None:
    registry = IdempotencyRegistry()
    key = generate_idempotency_key({"strategy_id": "orb-v1", "qty": "1"})
    assert registry.check_and_claim(key).allowed
    duplicate = registry.check_and_claim(key)
    assert not duplicate.allowed
    assert duplicate.code == "duplicate_order"


def test_idempotency_registry_denies_blank_key() -> None:
    registry = IdempotencyRegistry()
    assert not registry.check_and_claim("").allowed
    assert not registry.check_and_claim(None).allowed  # type: ignore[arg-type]


def test_allowlist_permits_listed_symbol() -> None:
    verdict = check_instrument_allowlist("NSE_EQ|RELIANCE", {"NSE_EQ|RELIANCE"})
    assert verdict.allowed


def test_allowlist_denies_unlisted_symbol() -> None:
    verdict = check_instrument_allowlist("NSE_EQ|INFY", {"NSE_EQ|RELIANCE"})
    assert not verdict.allowed
    assert verdict.code == "instrument_not_allowed"


def test_allowlist_denies_everything_when_empty() -> None:
    assert not check_instrument_allowlist("NSE_EQ|RELIANCE", set()).allowed


def test_allowlist_denies_blank_symbol() -> None:
    assert not check_instrument_allowlist("", {"NSE_EQ|RELIANCE"}).allowed
    assert not check_instrument_allowlist(None, {"NSE_EQ|RELIANCE"}).allowed  # type: ignore[arg-type]


def test_session_gate_allows_mid_session() -> None:
    now = datetime(2026, 9, 15, 10, 0, tzinfo=IST)
    assert check_session_gate(now, _policy()).allowed


def test_session_gate_denies_before_open() -> None:
    now = datetime(2026, 9, 15, 9, 0, tzinfo=IST)
    verdict = check_session_gate(now, _policy())
    assert not verdict.allowed
    assert verdict.code == "outside_session"


def test_session_gate_denies_inside_close_buffer() -> None:
    # Continuous end 15:30 with a 15-minute buffer means no new orders at/after 15:15.
    verdict = check_session_gate(datetime(2026, 9, 15, 15, 15, tzinfo=IST), _policy())
    assert not verdict.allowed
    assert verdict.code == "close_buffer"


def test_session_gate_allows_one_second_before_cutoff() -> None:
    now = datetime(2026, 9, 15, 15, 14, 59, tzinfo=IST)
    assert check_session_gate(now, _policy()).allowed


def test_session_gate_denies_naive_datetime() -> None:
    # Intentionally naive: the rail must refuse a timestamp it cannot place.
    verdict = check_session_gate(datetime(2026, 9, 15, 10, 0), _policy())  # noqa: DTZ001
    assert not verdict.allowed


def test_session_gate_is_cas_aware() -> None:
    cas_policy = NSEEquitySessionPolicy(cas_eligible=True, exit_buffer_minutes=15)
    # From 2026-08-03 a CAS-eligible stock ends continuous trading at 15:15,
    # so with a 15-minute buffer the cutoff is 15:00.
    assert check_session_gate(datetime(2026, 9, 15, 14, 59, 59, tzinfo=IST), cas_policy).allowed
    verdict = check_session_gate(datetime(2026, 9, 15, 15, 0, tzinfo=IST), cas_policy)
    assert not verdict.allowed


def test_token_probe_valid_is_allowed() -> None:
    probe = TokenProbeResult(token_valid=True, status_code=200)
    assert check_token_validity(probe).allowed


def test_token_probe_unauthorized_is_denied() -> None:
    probe = TokenProbeResult(token_valid=False, status_code=401)
    verdict = check_token_validity(probe)
    assert not verdict.allowed
    assert verdict.code == "token_invalid"


def test_token_probe_unknown_is_denied() -> None:
    assert not check_token_validity(None).allowed  # type: ignore[arg-type]
    assert not check_token_validity("ok").allowed  # type: ignore[arg-type]
    probe = TokenProbeResult(token_valid=True, status_code=None)
    assert not check_token_validity(probe).allowed


def test_kill_switch_clear_is_allowed(tmp_path: Path) -> None:
    missing = tmp_path / "kill-switch"
    assert check_kill_switch(False, missing).allowed


def test_kill_switch_flag_is_denied(tmp_path: Path) -> None:
    verdict = check_kill_switch(True, tmp_path / "kill-switch")
    assert not verdict.allowed
    assert verdict.code == "kill_switch_engaged"


def test_kill_switch_file_is_denied(tmp_path: Path) -> None:
    switch = tmp_path / "kill-switch"
    switch.write_text("stop")
    verdict = check_kill_switch(False, switch)
    assert not verdict.allowed
    assert verdict.code == "kill_switch_engaged"


def test_evaluate_all_rails_passes_when_every_rail_passes(tmp_path: Path) -> None:
    verdict = evaluate_all_rails(
        deployed_rupees=Decimal(500),
        approved_capital_rupees=Decimal(1000),
        order_notional_rupees=Decimal(500),
        per_order_cap_rupees=Decimal(1000),
        realized_pnl_rupees=Decimal(10),
        unrealized_pnl_rupees=Decimal(5),
        daily_loss_limit_rupees=Decimal(200),
        instrument_key="NSE_EQ|RELIANCE",
        instrument_allowlist={"NSE_EQ|RELIANCE"},
        now_ist=datetime(2026, 9, 15, 10, 0, tzinfo=IST),
        session_policy=_policy(),
        token_probe=TokenProbeResult(token_valid=True, status_code=200),
        kill_switch_engaged=False,
        kill_switch_path=tmp_path / "kill-switch",
        idempotency_registry=IdempotencyRegistry(),
        idempotency_key=generate_idempotency_key({"order": "1"}),
        rate_limiter=RateLimiter(max_orders=5, window_seconds=60, min_interval_seconds=0),
        trade_date=date(2026, 9, 15),
    )
    assert verdict.allowed


def test_evaluate_all_rails_denies_on_first_failing_rail(tmp_path: Path) -> None:
    verdict = evaluate_all_rails(
        deployed_rupees=Decimal(500),
        approved_capital_rupees=Decimal(1000),
        order_notional_rupees=Decimal(500),
        per_order_cap_rupees=Decimal(1000),
        realized_pnl_rupees=Decimal(10),
        unrealized_pnl_rupees=Decimal(5),
        daily_loss_limit_rupees=Decimal(200),
        instrument_key="NSE_EQ|INFY",
        instrument_allowlist={"NSE_EQ|RELIANCE"},
        now_ist=datetime(2026, 9, 15, 10, 0, tzinfo=IST),
        session_policy=_policy(),
        token_probe=TokenProbeResult(token_valid=True, status_code=200),
        kill_switch_engaged=True,
        kill_switch_path=tmp_path / "kill-switch",
        idempotency_registry=IdempotencyRegistry(),
        idempotency_key=generate_idempotency_key({"order": "2"}),
        rate_limiter=RateLimiter(max_orders=5, window_seconds=60, min_interval_seconds=0),
        trade_date=date(2026, 9, 15),
    )
    assert not verdict.allowed
    # Kill-switch is evaluated before the allowlist, so it must win.
    assert verdict.code == "kill_switch_engaged"


def test_verdict_dict_carries_live_orders_false() -> None:
    verdict = check_total_capital(Decimal(100), Decimal(1000))
    assert verdict.as_dict()["live_orders_called"] is False


def test_modules_have_no_order_or_network_imports() -> None:
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "equity_engine"
    for name in ("eligibility_decision.py", "trading_rails.py"):
        source = (root / name).read_text()
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any(
            module.split(".")[0] in {"broker", "services", "httpx", "requests"}
            for module in imported
        )
        assert "place_order" not in source
