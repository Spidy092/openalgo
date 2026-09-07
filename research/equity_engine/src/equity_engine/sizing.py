from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

from .costs import CostProvider
from .models import Exchange, OrderSpec, Product, Side


@dataclass(frozen=True)
class PositionSize:
    quantity: int
    notional: Decimal
    entry_charges: Decimal
    cash_required: Decimal
    cash_remaining: Decimal


def max_affordable_buy_quantity(
    *,
    instrument_token: str,
    exchange: Exchange,
    product: Product,
    price: Decimal,
    cash_limit: Decimal,
    cost_provider: CostProvider,
) -> PositionSize:
    """Return the largest integer quantity whose buy notional + quoted charges fits cash.

    The search calls the configured cost provider instead of assuming linear fees, so the same
    function works with capped brokerage and with the authoritative broker quote provider.
    """

    if price <= 0:
        raise ValueError("price must be positive")
    if cash_limit <= 0:
        raise ValueError("cash_limit must be positive")

    upper = int((cash_limit / price).to_integral_value(rounding=ROUND_FLOOR))
    if upper <= 0:
        return PositionSize(
            quantity=0,
            notional=Decimal("0"),
            entry_charges=Decimal("0"),
            cash_required=Decimal("0"),
            cash_remaining=cash_limit,
        )

    low = 0
    high = upper
    best: PositionSize | None = None

    while low <= high:
        quantity = (low + high) // 2
        if quantity == 0:
            low = 1
            continue

        order = OrderSpec(
            instrument_token=instrument_token,
            exchange=exchange,
            side=Side.BUY,
            product=product,
            quantity=quantity,
            price=price,
        )
        quote = cost_provider.quote(order)
        required = order.notional + quote.charges.total

        if required <= cash_limit:
            best = PositionSize(
                quantity=quantity,
                notional=order.notional,
                entry_charges=quote.charges.total,
                cash_required=required,
                cash_remaining=cash_limit - required,
            )
            low = quantity + 1
        else:
            high = quantity - 1

    if best is None:
        return PositionSize(
            quantity=0,
            notional=Decimal("0"),
            entry_charges=Decimal("0"),
            cash_required=Decimal("0"),
            cash_remaining=cash_limit,
        )
    return best
