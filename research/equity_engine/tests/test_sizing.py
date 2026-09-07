from datetime import date
from decimal import Decimal

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
