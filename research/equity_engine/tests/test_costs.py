from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from equity_engine.costs import reconcile_costs, round_trip_result
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.models import (
    ChargeBreakdown,
    CostQuote,
    CostSource,
    Exchange,
    ExecutionFriction,
    OrderSpec,
    Product,
    Side,
)


def _order(side: Side, price: str) -> OrderSpec:
    return OrderSpec(
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        side=side,
        product=Product.INTRADAY,
        quantity=1,
        price=Decimal(price),
    )


def test_current_terms_1000_rupee_buy_cost_is_decimal_exact() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))
    quote = provider.quote(_order(Side.BUY, "1000"))

    assert quote.charges.brokerage == Decimal("1.000")
    assert quote.charges.transaction == Decimal("0.030699000")
    assert quote.charges.ipft == Decimal("0.000001000")
    assert quote.charges.sebi_turnover == Decimal("0.001000")
    assert quote.charges.stamp_duty == Decimal("0.03000")
    assert quote.charges.gst == Decimal("0.18552600000")
    assert quote.charges.total == Decimal("1.24722600000")


def test_round_trip_reports_net_after_all_modeled_costs() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))
    entry = provider.quote(_order(Side.BUY, "1000"))
    exit_quote = provider.quote(_order(Side.SELL, "1005"))

    result = round_trip_result(
        entry_quote=entry,
        exit_quote=exit_quote,
        friction=ExecutionFriction(
            slippage_bps_per_leg=Decimal("0"),
            half_spread_bps_per_leg=Decimal("0"),
        ),
    )

    assert result.gross_pnl == Decimal("5")
    assert result.transaction_costs == Decimal("2.72178813000")
    assert result.net_pnl == Decimal("2.27821187000")


def test_slippage_and_half_spread_are_charged_on_both_legs() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))
    entry = provider.quote(_order(Side.BUY, "1000"))
    exit_quote = provider.quote(_order(Side.SELL, "1005"))

    result = round_trip_result(
        entry_quote=entry,
        exit_quote=exit_quote,
        friction=ExecutionFriction(
            slippage_bps_per_leg=Decimal("2"),
            half_spread_bps_per_leg=Decimal("1"),
        ),
    )

    expected_friction = (Decimal("1000") + Decimal("1005")) * Decimal("3") / Decimal(
        "10000"
    )
    assert result.modeled_execution_friction == expected_friction
    assert result.net_pnl == Decimal("5") - Decimal("2.72178813000") - expected_friction


def test_cost_reconciliation_requires_authoritative_broker_quote() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))
    estimate = provider.quote(_order(Side.BUY, "1000"))

    broker = CostQuote(
        order=estimate.order,
        charges=ChargeBreakdown(
            brokerage=Decimal("1"),
            gst=Decimal("0.18"),
            stt=Decimal("0"),
            stamp_duty=Decimal("0.03"),
            transaction=Decimal("0.03"),
            ipft=Decimal("0"),
            sebi_turnover=Decimal("0.001"),
        ),
        source=CostSource.BROKER_QUOTE,
        retrieved_at=datetime.now(timezone.utc),
        source_refs=("broker-test",),
        broker_reported_total=Decimal("1.25"),
    )

    result = reconcile_costs(estimate, broker, tolerance=Decimal("0.01"))
    assert result.absolute_error == abs(Decimal("1.24722600000") - Decimal("1.25"))
    assert result.passed


def test_documented_provider_rejects_old_pricing_date() -> None:
    with pytest.raises(ValueError):
        CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 2, 28))
