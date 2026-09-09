from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Tuple


class CostSource(StrEnum):
    BROKER_QUOTE = "broker_quote"
    DOCUMENTED_SNAPSHOT = "documented_snapshot"
    OBSERVED_SNAPSHOT = "observed_snapshot"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class Product(StrEnum):
    INTRADAY = "I"
    DELIVERY = "D"


class Exchange(StrEnum):
    NSE = "NSE"
    BSE = "BSE"


@dataclass(frozen=True)
class OrderSpec:
    instrument_token: str
    exchange: Exchange
    side: Side
    product: Product
    quantity: int
    price: Decimal

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if not self.instrument_token:
            raise ValueError("instrument_token is required")

    @property
    def notional(self) -> Decimal:
        return self.price * self.quantity


@dataclass(frozen=True)
class ChargeBreakdown:
    brokerage: Decimal
    gst: Decimal
    stt: Decimal
    stamp_duty: Decimal
    transaction: Decimal
    clearing: Decimal = Decimal("0")
    ipft: Decimal = Decimal("0")
    sebi_turnover: Decimal = Decimal("0")
    demat_transaction: Decimal = Decimal("0")
    other: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def total(self) -> Decimal:
        return sum(self.__dict__.values(), start=Decimal("0"))


@dataclass(frozen=True)
class CostQuote:
    order: OrderSpec
    charges: ChargeBreakdown
    source: CostSource
    retrieved_at: datetime
    source_refs: Tuple[str, ...]
    effective_date: date | None = None
    broker_reported_total: Decimal | None = None

    @property
    def authoritative(self) -> bool:
        return self.source is CostSource.BROKER_QUOTE

    @property
    def total(self) -> Decimal:
        """Cash-impacting cost total.

        For an authoritative broker quote, the broker's reported total wins over the local sum
        of components because the broker may apply component-level rounding. Components remain
        available for audit and reconciliation.
        """

        if self.authoritative and self.broker_reported_total is not None:
            return self.broker_reported_total
        return self.charges.total

    @property
    def reconciliation_error(self) -> Decimal | None:
        if self.broker_reported_total is None:
            return None
        return abs(self.charges.total - self.broker_reported_total)


@dataclass(frozen=True)
class ExecutionFriction:
    slippage_bps_per_leg: Decimal
    half_spread_bps_per_leg: Decimal

    def __post_init__(self) -> None:
        if self.slippage_bps_per_leg < 0:
            raise ValueError("slippage_bps_per_leg cannot be negative")
        if self.half_spread_bps_per_leg < 0:
            raise ValueError("half_spread_bps_per_leg cannot be negative")


@dataclass(frozen=True)
class RoundTripResult:
    entry: OrderSpec
    exit: OrderSpec
    entry_charges: ChargeBreakdown
    exit_charges: ChargeBreakdown
    entry_cost_total: Decimal
    exit_cost_total: Decimal
    modeled_execution_friction: Decimal
    gross_pnl: Decimal

    @property
    def transaction_costs(self) -> Decimal:
        return self.entry_cost_total + self.exit_cost_total

    @property
    def net_pnl(self) -> Decimal:
        return self.gross_pnl - self.transaction_costs - self.modeled_execution_friction

    @property
    def capital_return_pct(self) -> Decimal:
        if self.entry.notional == 0:
            raise ZeroDivisionError("entry notional cannot be zero")
        return self.net_pnl / self.entry.notional * Decimal("100")
