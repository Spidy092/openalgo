from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from .models import CostQuote, ExecutionFriction, OrderSpec, RoundTripResult, Side


_BPS = Decimal("10000")


class CostProvider(Protocol):
    def quote(self, order: OrderSpec) -> CostQuote:
        """Return a complete charge quote for one order."""


@dataclass(frozen=True)
class ReconciliationResult:
    estimated_total: Decimal
    broker_total: Decimal
    absolute_error: Decimal
    tolerance: Decimal

    @property
    def passed(self) -> bool:
        return self.absolute_error <= self.tolerance


def reconcile_costs(
    estimated: CostQuote,
    authoritative: CostQuote,
    *,
    tolerance: Decimal,
) -> ReconciliationResult:
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    if not authoritative.authoritative:
        raise ValueError("authoritative quote must come from the broker")
    if estimated.order != authoritative.order:
        raise ValueError("cannot reconcile quotes for different orders")

    estimated_total = estimated.charges.total
    broker_total = (
        authoritative.broker_reported_total
        if authoritative.broker_reported_total is not None
        else authoritative.charges.total
    )
    absolute_error = abs(estimated_total - broker_total)
    return ReconciliationResult(
        estimated_total=estimated_total,
        broker_total=broker_total,
        absolute_error=absolute_error,
        tolerance=tolerance,
    )


def round_trip_result(
    *,
    entry_quote: CostQuote,
    exit_quote: CostQuote,
    friction: ExecutionFriction,
) -> RoundTripResult:
    entry = entry_quote.order
    exit_order = exit_quote.order

    if entry.instrument_token != exit_order.instrument_token:
        raise ValueError("entry and exit must reference the same instrument")
    if entry.exchange != exit_order.exchange:
        raise ValueError("entry and exit must use the same exchange")
    if entry.product != exit_order.product:
        raise ValueError("entry and exit must use the same product")
    if entry.quantity != exit_order.quantity:
        raise ValueError("partial-position round trips must be modeled as separate lots")
    if entry.side is not Side.BUY or exit_order.side is not Side.SELL:
        raise ValueError("initial equity engine supports long BUY -> SELL round trips only")

    friction_bps = friction.slippage_bps_per_leg + friction.half_spread_bps_per_leg
    modeled_execution_friction = (
        entry.notional * friction_bps / _BPS
        + exit_order.notional * friction_bps / _BPS
    )
    gross_pnl = (exit_order.price - entry.price) * entry.quantity

    return RoundTripResult(
        entry=entry,
        exit=exit_order,
        entry_charges=entry_quote.charges,
        exit_charges=exit_quote.charges,
        modeled_execution_friction=modeled_execution_friction,
        gross_pnl=gross_pnl,
    )


def minimum_exit_price_for_nonnegative_net_pnl(
    *,
    entry_quote: CostQuote,
    exit_cost_at_candidate_price: CostProvider,
    friction: ExecutionFriction,
    price_step: Decimal,
    max_steps: int = 100_000,
) -> Decimal:
    """Find the first candidate exit price whose modeled net P&L is non-negative.

    This intentionally queries the configured cost provider at every candidate price instead
    of assuming that all charges scale linearly. It is for research/sizing, not an order loop.
    """

    if price_step <= 0:
        raise ValueError("price_step must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    entry = entry_quote.order
    price = entry.price
    for _ in range(max_steps):
        price += price_step
        exit_order = OrderSpec(
            instrument_token=entry.instrument_token,
            exchange=entry.exchange,
            side=Side.SELL,
            product=entry.product,
            quantity=entry.quantity,
            price=price,
        )
        exit_quote = exit_cost_at_candidate_price.quote(exit_order)
        result = round_trip_result(
            entry_quote=entry_quote,
            exit_quote=exit_quote,
            friction=friction,
        )
        if result.net_pnl >= 0:
            return price

    raise RuntimeError("no non-negative exit price found within max_steps")
