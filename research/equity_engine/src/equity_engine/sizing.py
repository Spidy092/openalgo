from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

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
    minimum_tradable_quantity: int = 1,
) -> PositionSize:
    """Return the largest tradable quantity whose buy cash requirement fits cash.

    The search calls the configured cost provider instead of assuming linear fees, so the same
    function works with capped brokerage and with the authoritative broker quote provider.
    ``cash_limit`` is explicit approved capital; it is never read from a broker balance or
    inferred from account readiness.
    """

    if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
        raise ValueError("price must be positive")
    if not isinstance(cash_limit, Decimal) or not cash_limit.is_finite() or cash_limit <= 0:
        raise ValueError("cash_limit must be positive")
    if (
        isinstance(minimum_tradable_quantity, bool)
        or not isinstance(minimum_tradable_quantity, int)
        or minimum_tradable_quantity <= 0
    ):
        raise ValueError("minimum_tradable_quantity must be a positive integer")

    upper = int((cash_limit / price).to_integral_value(rounding=ROUND_FLOOR))
    upper_units = upper // minimum_tradable_quantity
    if upper_units <= 0:
        return PositionSize(
            quantity=0,
            notional=Decimal("0"),
            entry_charges=Decimal("0"),
            cash_required=Decimal("0"),
            cash_remaining=cash_limit,
        )

    low = 0
    high = upper_units
    best: PositionSize | None = None

    while low <= high:
        units = (low + high) // 2
        quantity = units * minimum_tradable_quantity
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
        required = order.notional + quote.total

        if required <= cash_limit:
            best = PositionSize(
                quantity=quantity,
                notional=order.notional,
                entry_charges=quote.total,
                cash_required=required,
                cash_remaining=cash_limit - required,
            )
            low = units + 1
        else:
            high = units - 1

    if best is None:
        return PositionSize(
            quantity=0,
            notional=Decimal("0"),
            entry_charges=Decimal("0"),
            cash_required=Decimal("0"),
            cash_remaining=cash_limit,
        )
    return best
