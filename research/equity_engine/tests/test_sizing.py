from datetime import date
from decimal import Decimal

import pytest

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.models import Exchange, Product
from equity_engine.sizing import max_affordable_buy_quantity


def test_1000_cash_does_not_overallocate_notional() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))

    result = max_affordable_buy_quantity(
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        product=Product.INTRADAY,
        price=Decimal("100"),
        cash_limit=Decimal("1000"),
        cost_provider=provider,
    )

    assert result.quantity == 9
    assert result.notional == Decimal("900")
    assert result.cash_required <= Decimal("1000")
    assert result.cash_remaining >= 0


def test_zero_quantity_when_one_share_plus_charges_exceeds_cash() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))

    result = max_affordable_buy_quantity(
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        product=Product.INTRADAY,
        price=Decimal("1000"),
        cash_limit=Decimal("1000"),
        cost_provider=provider,
    )

    assert result.quantity == 0
    assert result.cash_remaining == Decimal("1000")


def test_minimum_tradable_quantity_is_a_quantity_multiple() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))

    result = max_affordable_buy_quantity(
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        product=Product.INTRADAY,
        price=Decimal("100"),
        cash_limit=Decimal("2500"),
        cost_provider=provider,
        minimum_tradable_quantity=10,
    )

    assert result.quantity > 0
    assert result.quantity % 10 == 0
    assert result.cash_required <= Decimal("2500")


@pytest.mark.parametrize(
    ("price", "cash_limit", "minimum_tradable_quantity"),
    [
        (Decimal("NaN"), Decimal("1000"), 1),
        (Decimal("0"), Decimal("1000"), 1),
        (Decimal("-1"), Decimal("1000"), 1),
        (Decimal("100"), Decimal("Infinity"), 1),
        (Decimal("100"), Decimal("1000"), 0),
        (Decimal("100"), Decimal("1000"), -1),
    ],
)
def test_invalid_affordability_inputs_fail_closed(
    price: Decimal,
    cash_limit: Decimal,
    minimum_tradable_quantity: int,
) -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))

    with pytest.raises(ValueError):
        max_affordable_buy_quantity(
            instrument_token="NSE_EQ|TEST",
            exchange=Exchange.NSE,
            product=Product.INTRADAY,
            price=price,
            cash_limit=cash_limit,
            cost_provider=provider,
            minimum_tradable_quantity=minimum_tradable_quantity,
        )
