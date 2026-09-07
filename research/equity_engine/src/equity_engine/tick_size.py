from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable

from .instrument_master import EquityInstrument


NSE_TIERED_TICK_EFFECTIVE_DATE = date(2025, 4, 15)
NSE_TICK_SOURCE = "https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf"


@dataclass(frozen=True)
class TickSizeVerification:
    passed: bool
    expected_rupees: Decimal
    observed_rupees: Decimal
    source: str


@dataclass(frozen=True)
class TickSizePoint:
    effective_from: date
    tick_size_rupees: Decimal
    source: str

    def __post_init__(self) -> None:
        if self.tick_size_rupees <= 0:
            raise ValueError("tick_size_rupees must be positive")
        if not self.source.strip():
            raise ValueError("tick-size source is required")


@dataclass(frozen=True)
class FixedTickSizePolicy:
    """Explicit fixed tick policy for synthetic/unit-test datasets only."""

    tick_size_rupees: Decimal
    source: str

    def __post_init__(self) -> None:
        if self.tick_size_rupees <= 0:
            raise ValueError("tick_size_rupees must be positive")
        if not self.source.strip():
            raise ValueError("fixed tick-size source is required")

    def tick_size(self, trade_date: date) -> Decimal:
        return self.tick_size_rupees


class EffectiveDatedTickSizePolicy:
    """Resolve tick size from explicit dated security-master evidence.

    The policy does not infer monthly effective dates. Each point must come from an external
    security master/circular snapshot. For a trade date, the latest point not after that date is
    used. Dates before the earliest point fail closed.
    """

    def __init__(self, points: Iterable[TickSizePoint]) -> None:
        ordered = tuple(sorted(points, key=lambda item: item.effective_from))
        if not ordered:
            raise ValueError("at least one effective-dated tick point is required")
        if len({item.effective_from for item in ordered}) != len(ordered):
            raise ValueError("duplicate effective_from dates in tick policy")
        self._points = ordered

    @property
    def points(self) -> tuple[TickSizePoint, ...]:
        return self._points

    def tick_size(self, trade_date: date) -> Decimal:
        applicable: TickSizePoint | None = None
        for point in self._points:
            if point.effective_from <= trade_date:
                applicable = point
            else:
                break
        if applicable is None:
            raise ValueError(f"no verified tick-size evidence for trade date {trade_date}")
        return applicable.tick_size_rupees


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
