"""Phase 3 autonomous bridge tests — sandbox-only, deny-by-default.

These tests prove the bridge:
- refuses a live dispatch target outright (no broker path exists),
- refuses on invalid/expired/mismatched eligibility,
- refuses on any failing safety rail,
- refuses full_auto unless explicitly permitted,
- dispatches ONLY through services.sandbox_service.sandbox_place_order,
- never sets live_orders_called True.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

# Make the equity_engine package importable (src layout) without installing it.
_ENGINE_SRC = Path(__file__).resolve().parents[1] / "research" / "equity_engine" / "src"
if str(_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(_ENGINE_SRC))

from equity_engine.eligibility_decision import (  # noqa: E402
    AutonomyMode,
    Track,
    build_eligibility_decision,
    sign_eligibility_decision,
)
from equity_engine.trading_rails import (  # noqa: E402
    IdempotencyRegistry,
    RateLimiter,
    TokenProbeResult,
)

from services.autonomous_bridge.bridge import (  # noqa: E402
    DispatchTarget,
    OrderIntent,
    autonomous_dispatch,
)

INSTRUMENT = "NSE_EQ|INE745G01043"
IST_OFFSET = timedelta(hours=5, minutes=30)


class _StubSessionPolicy:
    """Minimal session policy: open 09:15, close 15:30, exit buffer to 15:20."""

    def continuous_start(self, _day):
        from datetime import time

        return time(9, 15)

    def continuous_end(self, _day):
        from datetime import time

        return time(15, 30)

    def exit_time(self, _day):
        from datetime import time

        return time(15, 20)


def _ist(hour: int, minute: int) -> datetime:
    from datetime import timezone

    return datetime(2026, 9, 7, hour, minute, tzinfo=timezone(IST_OFFSET))


def _valid_probe() -> TokenProbeResult:
    return TokenProbeResult(token_valid=True, status_code=200, checked_at=_ist(10, 0))


def _signed_decision(*, autonomy: AutonomyMode = AutonomyMode.SUPERVISED_AUTO) -> dict:
    decision = build_eligibility_decision(
        strategy_id="orb-v1",
        instrument_key=INSTRUMENT,
        track=Track.INTRADAY,
        approved_capital_rupees=Decimal(1000),
        per_order_notional_cap_rupees=Decimal(500),
        daily_loss_limit_rupees=Decimal(200),
        valid_from=_ist(9, 0),
        valid_until=_ist(15, 30),
        autonomy_mode=autonomy,
    )
    return sign_eligibility_decision(decision)


def _intent() -> OrderIntent:
    return OrderIntent(
        symbol="MCX",
        exchange="NSE",
        action="BUY",
        quantity=1,
        product="MIS",
        instrument_key=INSTRUMENT,
        price_type="LIMIT",
        price=Decimal(300),
    )


def _kwargs(tmp_path: Path, **overrides):
    base = {
        "session_id": "sess-1",
        "api_key": "test-openalgo-key",
        "eligibility_payload": _signed_decision(),
        "intent": _intent(),
        "now": _ist(10, 0),
        "session_policy": _StubSessionPolicy(),
        "token_probe": _valid_probe(),
        "deployed_rupees": Decimal(0),
        "realized_pnl_rupees": Decimal(0),
        "unrealized_pnl_rupees": Decimal(0),
        "instrument_allowlist": {INSTRUMENT},
        "rate_limiter": RateLimiter(max_orders=5, window_seconds=60, clock=lambda: _ist(10, 0)),
        "idempotency_registry": IdempotencyRegistry(),
        "audit_path": tmp_path / "audit.jsonl",
    }
    base.update(overrides)
    return base


@pytest.fixture()
def sandbox_spy(monkeypatch):
    """Replace sandbox_place_order with a spy that records calls and accepts."""
    calls: list[dict] = []

    def _fake(order_data, api_key, original_data, prefetched_quote=None):
        calls.append({"order_data": dict(order_data), "api_key": api_key})
        return True, {"status": "success", "orderid": "SBX-1", "mode": "analyze"}, 200

    import services.sandbox_service as sandbox_service

    monkeypatch.setattr(sandbox_service, "sandbox_place_order", _fake)
    return calls


def test_happy_path_dispatches_to_sandbox(tmp_path, sandbox_spy):
    result = autonomous_dispatch(**_kwargs(tmp_path))
    assert result.dispatched is True
    assert result.target == "sandbox"
    assert result.live_orders_called is False
    assert len(sandbox_spy) == 1
    # The order carried no credential into the sandbox payload itself.
    assert "apikey" not in sandbox_spy[0]["order_data"]


def test_live_target_is_refused_before_any_dispatch(tmp_path, sandbox_spy):
    result = autonomous_dispatch(**_kwargs(tmp_path, target=DispatchTarget.LIVE))
    assert result.dispatched is False
    assert result.code == "live_target_blocked"
    assert sandbox_spy == []  # nothing dispatched


def test_expired_decision_is_refused(tmp_path, sandbox_spy):
    result = autonomous_dispatch(**_kwargs(tmp_path, now=_ist(16, 0)))
    assert result.dispatched is False
    assert result.stage == "eligibility"
    assert sandbox_spy == []


def test_instrument_mismatch_is_refused(tmp_path, sandbox_spy):
    other = OrderIntent(
        symbol="INFY",
        exchange="NSE",
        action="BUY",
        quantity=1,
        product="MIS",
        instrument_key="NSE_EQ|INE009A01021",
        price_type="LIMIT",
        price=Decimal(100),
    )
    result = autonomous_dispatch(**_kwargs(tmp_path, intent=other))
    assert result.dispatched is False
    # Either eligibility instrument mismatch or allowlist denial — both refuse.
    assert sandbox_spy == []


def test_kill_switch_refuses(tmp_path, sandbox_spy):
    result = autonomous_dispatch(**_kwargs(tmp_path, kill_switch_engaged=True))
    assert result.dispatched is False
    assert result.stage == "rails"
    assert sandbox_spy == []


def test_outside_session_refuses(tmp_path, sandbox_spy):
    result = autonomous_dispatch(**_kwargs(tmp_path, now=_ist(8, 0)))
    assert result.dispatched is False
    assert sandbox_spy == []


def test_invalid_token_refuses(tmp_path, sandbox_spy):
    bad = TokenProbeResult(token_valid=False, status_code=401)
    result = autonomous_dispatch(**_kwargs(tmp_path, token_probe=bad))
    assert result.dispatched is False
    assert result.stage == "rails"
    assert sandbox_spy == []


def test_per_order_cap_refuses(tmp_path, sandbox_spy):
    big = OrderIntent(
        symbol="MCX",
        exchange="NSE",
        action="BUY",
        quantity=100,
        product="MIS",
        instrument_key=INSTRUMENT,
        price_type="LIMIT",
        price=Decimal(300),
    )
    result = autonomous_dispatch(**_kwargs(tmp_path, intent=big))
    assert result.dispatched is False
    assert result.stage == "rails"
    assert sandbox_spy == []


def test_daily_loss_limit_refuses(tmp_path, sandbox_spy):
    result = autonomous_dispatch(
        **_kwargs(tmp_path, realized_pnl_rupees=Decimal(-250))
    )
    assert result.dispatched is False
    assert result.stage == "rails"
    assert sandbox_spy == []


def test_duplicate_order_refused_on_second_call(tmp_path, sandbox_spy):
    shared_registry = IdempotencyRegistry()
    shared_limiter = RateLimiter(max_orders=5, window_seconds=60, clock=lambda: _ist(10, 0))
    first = autonomous_dispatch(
        **_kwargs(tmp_path, idempotency_registry=shared_registry, rate_limiter=shared_limiter)
    )
    assert first.dispatched is True
    second = autonomous_dispatch(
        **_kwargs(tmp_path, idempotency_registry=shared_registry, rate_limiter=shared_limiter)
    )
    assert second.dispatched is False
    assert second.code == "duplicate_order"
    assert len(sandbox_spy) == 1


def test_full_auto_requires_explicit_opt_in(tmp_path, sandbox_spy):
    payload = _signed_decision(autonomy=AutonomyMode.FULL_AUTO)
    refused = autonomous_dispatch(**_kwargs(tmp_path, eligibility_payload=payload))
    assert refused.dispatched is False
    assert refused.code == "full_auto_not_permitted"
    assert sandbox_spy == []

    allowed = autonomous_dispatch(
        **_kwargs(tmp_path, eligibility_payload=payload, allow_full_auto=True)
    )
    assert allowed.dispatched is True
    assert allowed.live_orders_called is False
    assert len(sandbox_spy) == 1


def test_tampered_eligibility_fingerprint_refused(tmp_path, sandbox_spy):
    payload = _signed_decision()
    payload = dict(payload)
    payload["approved_capital_rupees"] = "999"  # break the signed fingerprint
    result = autonomous_dispatch(**_kwargs(tmp_path, eligibility_payload=payload))
    assert result.dispatched is False
    assert result.stage == "eligibility"
    assert sandbox_spy == []


def test_audit_trail_written(tmp_path, sandbox_spy):
    audit_path = tmp_path / "audit.jsonl"
    autonomous_dispatch(**_kwargs(tmp_path, audit_path=audit_path))
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    phases = [__import__("json").loads(line)["phase"] for line in lines]
    assert "attempt" in phases
    assert "decision" in phases
    assert "result" in phases
    # No credential ever written.
    assert "test-openalgo-key" not in audit_path.read_text(encoding="utf-8")
