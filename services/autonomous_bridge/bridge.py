"""Core autonomous bridge: eligibility + rails -> audited sandbox dispatch.

Phase 3 contract (enforced here, not merely documented):

1. The caller supplies a signed ``EligibilityDecision`` payload and an
   ``OrderIntent``. The decision is validated (fail-closed) against the clock,
   the intent's instrument, and the hard capital cap.
2. Every safety rail from ``equity_engine.trading_rails.evaluate_all_rails``
   must pass (deny-by-default). Any denial refuses the dispatch.
3. Dispatch target in Phase 3 is ALWAYS the sandbox. ``DispatchTarget.LIVE``
   exists as a named constant for later phases but is refused here with a clear
   message; there is no code path in this module that reaches a broker.
4. Full-auto requires BOTH the decision's ``autonomy_mode == full_auto`` AND the
   caller passing ``allow_full_auto=True``. Otherwise supervised-auto is assumed
   and a dispatch still proceeds only through the sandbox.
5. Every step is audited: attempt, rails decision, and result.

The bridge returns a uniform ``BridgeDispatchResult`` and never raises for a
mere refusal; it raises only for programmer error (e.g. wrong argument types).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from equity_engine.eligibility_decision import (
    AutonomyMode,
    validate_eligibility_decision,
)
from equity_engine.trading_rails import (
    IdempotencyRegistry,
    RateLimiter,
    TokenProbeResult,
    evaluate_all_rails,
    generate_idempotency_key,
)

from services.autonomous_bridge.audit_trail import AuditTrail
from utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_AUDIT_PATH = "logs/autonomous_bridge/audit.jsonl"


class BridgeMode(StrEnum):
    """The only dispatch pipes the bridge understands."""

    SANDBOX = "sandbox"
    LIVE = "live"


class DispatchTarget(StrEnum):
    """Where an approved order is sent. Phase 3 permits SANDBOX only."""

    SANDBOX = "sandbox"
    LIVE = "live"


@dataclass(frozen=True)
class OrderIntent:
    """One intended order in OpenAlgo order-service shape (no credentials)."""

    symbol: str
    exchange: str
    action: str
    quantity: int
    product: str
    instrument_key: str
    price_type: str = "MARKET"
    price: Decimal = Decimal(0)
    trigger_price: Decimal = Decimal(0)
    strategy: str = "Autonomous Bridge"

    def order_data(self) -> dict[str, Any]:
        """Render the OpenAlgo order-service payload (no api key, no token)."""
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "action": self.action,
            "quantity": str(self.quantity),
            "product": self.product,
            "pricetype": self.price_type,
            "price": str(self.price),
            "trigger_price": str(self.trigger_price),
        }

    def rail_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "action": self.action,
            "quantity": self.quantity,
            "product": self.product,
            "instrument_key": self.instrument_key,
            "price_type": self.price_type,
            "price": str(self.price),
            "trigger_price": str(self.trigger_price),
        }


@dataclass(frozen=True)
class BridgeDispatchResult:
    """Uniform bridge outcome. ``dispatched`` is the only field to branch on."""

    dispatched: bool
    stage: str
    target: str
    code: str
    reason: str
    order_response: dict[str, Any] = field(default_factory=dict)
    live_orders_called: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "dispatched": self.dispatched,
            "stage": self.stage,
            "target": self.target,
            "code": self.code,
            "reason": self.reason,
            "order_response": self.order_response,
            "live_orders_called": False,
        }


def _refused(stage: str, code: str, reason: str, target: str) -> BridgeDispatchResult:
    return BridgeDispatchResult(
        dispatched=False, stage=stage, target=target, code=code, reason=reason
    )


def autonomous_dispatch(
    *,
    session_id: str,
    api_key: str,
    eligibility_payload: Mapping[str, Any],
    intent: OrderIntent,
    now: datetime,
    session_policy: Any,
    token_probe: TokenProbeResult,
    deployed_rupees: Decimal,
    realized_pnl_rupees: Decimal,
    unrealized_pnl_rupees: Decimal,
    instrument_allowlist: Any,
    rate_limiter: RateLimiter,
    idempotency_registry: IdempotencyRegistry,
    target: DispatchTarget = DispatchTarget.SANDBOX,
    allow_full_auto: bool = False,
    kill_switch_engaged: bool = False,
    kill_switch_path: str | Path | None = None,
    trade_date: Any = None,
    audit_path: str | Path = _DEFAULT_AUDIT_PATH,
) -> BridgeDispatchResult:
    """Validate eligibility, run every rail, and dispatch to the SANDBOX only.

    Returns a refusal (never raises) for any gate failure. Real orders are
    impossible here: a non-sandbox target is refused before any dispatch.
    """
    audit = AuditTrail(audit_path)
    target_str = str(target.value if isinstance(target, DispatchTarget) else target)

    audit.attempt(
        session_id,
        {
            "api_key": api_key,  # redacted by the audit layer
            "intent": intent.rail_payload(),
            "target": target_str,
            "allow_full_auto": allow_full_auto,
        },
    )

    # Gate 0 — Phase 3 permits the sandbox pipe only. This is the structural
    # guarantee that no real order can be placed from this module.
    if target != DispatchTarget.SANDBOX:
        result = _refused(
            "target_gate",
            "live_target_blocked",
            "Phase 3 bridge dispatches to the sandbox only; the live pipe is not wired here",
            target_str,
        )
        audit.result(session_id, result.as_dict())
        return result

    # Gate 1 — eligibility decision must validate against clock, instrument, cap.
    validation = validate_eligibility_decision(
        eligibility_payload,
        now=now,
        expected_instrument_key=intent.instrument_key,
    )
    audit.decision(
        session_id,
        {"gate": "eligibility", "valid": validation.valid, "violations": list(validation.violations)},
    )
    if not validation.valid:
        result = _refused(
            "eligibility",
            "eligibility_invalid",
            "; ".join(validation.violations) or "eligibility decision refused",
            target_str,
        )
        audit.result(session_id, result.as_dict())
        return result

    # Full-auto requires both the signed decision's mode and an explicit caller opt-in.
    autonomy = str(eligibility_payload.get("autonomy_mode", "")).strip().lower()
    if autonomy == AutonomyMode.FULL_AUTO.value and not allow_full_auto:
        result = _refused(
            "autonomy_gate",
            "full_auto_not_permitted",
            "decision is full_auto but the caller did not pass allow_full_auto=True",
            target_str,
        )
        audit.result(session_id, result.as_dict())
        return result

    approved_capital = Decimal(str(eligibility_payload["approved_capital_rupees"]))
    per_order_cap = Decimal(str(eligibility_payload["per_order_notional_cap_rupees"]))
    daily_loss_limit = Decimal(str(eligibility_payload["daily_loss_limit_rupees"]))

    # A conservative notional for the rail: quantity * limit/trigger/price when
    # available, else deployed+per-order boundary is enforced by the caps below.
    reference_price = intent.price if intent.price > 0 else intent.trigger_price
    order_notional = (
        Decimal(intent.quantity) * reference_price if reference_price > 0 else per_order_cap
    )

    idempotency_key = generate_idempotency_key(
        {
            "session_id": session_id,
            "instrument_key": intent.instrument_key,
            **intent.rail_payload(),
        }
    )

    # Gate 2 — every safety rail (deny-by-default).
    verdict = evaluate_all_rails(
        deployed_rupees=deployed_rupees,
        approved_capital_rupees=approved_capital,
        order_notional_rupees=order_notional,
        per_order_cap_rupees=per_order_cap,
        realized_pnl_rupees=realized_pnl_rupees,
        unrealized_pnl_rupees=unrealized_pnl_rupees,
        daily_loss_limit_rupees=daily_loss_limit,
        instrument_key=intent.instrument_key,
        instrument_allowlist=instrument_allowlist,
        now_ist=now,
        session_policy=session_policy,
        trade_date=trade_date,
        token_probe=token_probe,
        kill_switch_engaged=kill_switch_engaged,
        kill_switch_path=kill_switch_path,
        idempotency_registry=idempotency_registry,
        idempotency_key=idempotency_key,
        rate_limiter=rate_limiter,
    )
    audit.decision(session_id, {"gate": "rails", **verdict.as_dict()})
    if not verdict.allowed:
        result = _refused("rails", verdict.code, verdict.reason, target_str)
        audit.result(session_id, result.as_dict())
        return result

    # Dispatch — SANDBOX ONLY. Imported function-locally per services convention.
    try:
        from services.sandbox_service import sandbox_place_order

        order_data = intent.order_data()
        original_data = dict(order_data)
        original_data["apikey"] = api_key
        success, response, status_code = sandbox_place_order(
            order_data, api_key, original_data
        )
    except Exception as exc:  # noqa: BLE001 - a dispatch failure is reported, never raised out
        logger.exception("autonomous bridge sandbox dispatch failed")
        result = _refused(
            "dispatch",
            "sandbox_dispatch_error",
            f"{type(exc).__name__}: {exc}",
            target_str,
        )
        audit.result(session_id, result.as_dict())
        return result

    result = BridgeDispatchResult(
        dispatched=bool(success),
        stage="dispatched" if success else "dispatch_rejected",
        target=target_str,
        code="ok" if success else "sandbox_rejected",
        reason="sandbox order accepted" if success else str(response),
        order_response=response if isinstance(response, dict) else {"raw": response},
    )
    audit.result(session_id, {**result.as_dict(), "status_code": status_code})
    return result
