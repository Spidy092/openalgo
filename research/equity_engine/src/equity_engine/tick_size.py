from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .instrument_master import EquityInstrument


NSE_TIERED_TICK_EFFECTIVE_DATE = date(2025, 4, 15)
NSE_TICK_SOURCE = "https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf"


@dataclass(frozen=True)
class TickSizeVerification:
    passed: bool
    expected_rupees: Decimal
    observed_rupees: Decimal
    source: str


def expected_nse_cm_tick_size_rupees(
    *,
    effective_trade_date: date,
    exchange_reference_price_rupees: Decimal,
) -> Decimal:
    """Return the CM tick tier from NSE/CMTR/67133 for its applicable period.

    The caller must supply the exchange reference price used for the monthly review. Current LTP
    is not silently substituted because a security can cross a tier boundary during the month.
    """

    if effective_trade_date < NSE_TIERED_TICK_EFFECTIVE_DATE:
        raise ValueError("tiered tick verifier only supports 2025-04-15 onward")
    if exchange_reference_price_rupees <= 0:
        raise ValueError("exchange reference price must be positive")

    price = exchange_reference_price_rupees
    if price < Decimal("250"):
        return Decimal("0.01")
    if price <= Decimal("1000"):
        return Decimal("0.05")
    if price <= Decimal("5000"):
        return Decimal("0.10")
    if price <= Decimal("10000"):
        return Decimal("0.50")
    if price <= Decimal("20000"):
        return Decimal("1.00")
    return Decimal("5.00")


def verify_instrument_tick_size(
    *,
    instrument: EquityInstrument,
    effective_trade_date: date,
    exchange_reference_price_rupees: Decimal,
) -> TickSizeVerification:
    expected = expected_nse_cm_tick_size_rupees(
        effective_trade_date=effective_trade_date,
        exchange_reference_price_rupees=exchange_reference_price_rupees,
    )
    return TickSizeVerification(
        passed=instrument.tick_size_rupees == expected,
        expected_rupees=expected,
        observed_rupees=instrument.tick_size_rupees,
        source=NSE_TICK_SOURCE,
    )
