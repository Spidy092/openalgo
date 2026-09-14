from datetime import date
from decimal import Decimal

import pytest

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.models import CostSource, Exchange, OrderSpec, Product, Side
from equity_engine.observed_costs import (
    PAISA,
    ObservedUpstoxNSEIntradayCostProvider,
)
from equity_engine.sizing import max_affordable_buy_quantity


OBSERVED_AT = date(2026, 9, 9)


def _order(*, side: Side, notional: str) -> OrderSpec:
    return OrderSpec(
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        side=side,
        product=Product.INTRADAY,
        quantity=1,
        price=Decimal(notional),
    )


@pytest.mark.parametrize(
    ("notional", "side", "expected"),
    [
        (
            "250",
            Side.BUY,
            {
                "brokerage": "0.15",
                "gst": "0.03",
                "stamp_duty": "0.01",
                "stt": "0",
                "transaction": "0.01",
                "total": "0.20",
            },
        ),
        (
            "250",
            Side.SELL,
            {
                "brokerage": "0.15",
                "gst": "0.03",
                "stamp_duty": "0",
                "stt": "0.06",
                "transaction": "0.01",
                "total": "0.25",
            },
        ),
        (
            "500",
            Side.BUY,
            {
                "brokerage": "0.30",
                "gst": "0.06",
                "stamp_duty": "0.02",
                "stt": "0",
                "transaction": "0.02",
                "total": "0.40",
            },
        ),
        (
            "500",
            Side.SELL,
            {
                "brokerage": "0.30",
                "gst": "0.06",
                "stamp_duty": "0",
                "stt": "0.13",
                "transaction": "0.02",
                "total": "0.51",
            },
        ),
        (
            "750",
            Side.BUY,
            {
                "brokerage": "0.45",
                "gst": "0.08",
                "stamp_duty": "0.02",
                "stt": "0",
                "transaction": "0.02",
                "total": "0.57",
            },
        ),
        (
            "750",
            Side.SELL,
            {
                "brokerage": "0.45",
                "gst": "0.08",
                "stamp_duty": "0",
                "stt": "0.19",
                "transaction": "0.02",
                "total": "0.74",
            },
        ),
    ],
)
def test_authenticated_upstox_fixtures_match_exactly(
    notional: str,
    side: Side,
    expected: dict[str, str],
) -> None:
    provider = ObservedUpstoxNSEIntradayCostProvider(
        pricing_date=OBSERVED_AT,
        effective_date=OBSERVED_AT,
        observed_at=OBSERVED_AT,
    )
    quote = provider.quote(_order(side=side, notional=notional))
    charges = quote.charges

    assert quote.source is CostSource.OBSERVED_SNAPSHOT
    assert charges.brokerage == Decimal(expected["brokerage"])
    assert charges.gst == Decimal(expected["gst"])
    assert charges.stamp_duty == Decimal(expected["stamp_duty"])
    assert charges.stt == Decimal(expected["stt"])
    assert charges.transaction == Decimal(expected["transaction"])
    assert charges.clearing == Decimal("0")
    assert charges.demat_transaction == Decimal("0")
    assert charges.ipft == Decimal("0")
    assert charges.sebi_turnover == Decimal("0")
    assert charges.other == Decimal("0")
    assert charges.total == Decimal(expected["total"])


def test_documented_model_remains_at_point_one_percent() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=OBSERVED_AT)
    quote = provider.quote(_order(side=Side.BUY, notional="250"))

    assert provider.BROKERAGE_RATE == Decimal("0.001")
    assert quote.charges.brokerage == Decimal("0.250")
    assert provider.provenance["brokerage_rate"] == "0.001"


def test_observed_model_is_decimal_and_has_explicit_provenance() -> None:
    provider = ObservedUpstoxNSEIntradayCostProvider(
        pricing_date=OBSERVED_AT,
        effective_date=OBSERVED_AT,
        observed_at=OBSERVED_AT,
    )
    quote = provider.quote(_order(side=Side.BUY, notional="750"))

    assert provider.BROKERAGE_RATE == Decimal("0.0006")
    assert isinstance(PAISA, Decimal)
    assert all(isinstance(value, Decimal) for value in vars(quote.charges).values())
    assert provider.provenance == {
        "source": "Upstox authenticated Brokerage Details API",
        "observed_at": "2026-09-09",
        "effective_date": "2026-09-09",
        "brokerage_rate": "0.0006",
        "scope": "NSE equity intraday / this authenticated account snapshot",
    }
    assert "access_token" not in str(provider.provenance)


def test_gst_uses_rounded_brokerage_transaction_and_ipft() -> None:
    provider = ObservedUpstoxNSEIntradayCostProvider(pricing_date=OBSERVED_AT)
    quote = provider.quote(_order(side=Side.BUY, notional="750"))

    rounded_basis = quote.charges.brokerage + quote.charges.transaction + quote.charges.ipft
    assert rounded_basis == Decimal("0.47")
    assert (rounded_basis * Decimal("0.18")).quantize(PAISA) == Decimal("0.08")
    assert quote.charges.gst == Decimal("0.08")


def test_observed_snapshot_requires_effective_date() -> None:
    with pytest.raises(ValueError, match="effective only on or after"):
        ObservedUpstoxNSEIntradayCostProvider(
            pricing_date=date(2026, 9, 8),
            effective_date=OBSERVED_AT,
        )


def test_charge_aware_sizing_works_with_observed_snapshot() -> None:
    provider = ObservedUpstoxNSEIntradayCostProvider(pricing_date=OBSERVED_AT)
    result = max_affordable_buy_quantity(
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        product=Product.INTRADAY,
        price=Decimal("100"),
        cash_limit=Decimal("1000"),
        cost_provider=provider,
    )

    assert result.quantity == 9
    assert result.cash_required <= Decimal("1000")
