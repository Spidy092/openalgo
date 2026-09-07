from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from enum import StrEnum
from typing import Protocol

import pandas as pd

from .costs import CostProvider
from .models import Exchange, OrderSpec, Product, Side
from .provenance import validate_ohlcv_frame
from .sizing import max_affordable_buy_quantity

_BPS = Decimal("10000")


class SessionExitResolver(Protocol):
    def exit_time(self, trade_date: date) -> time:
        """Return the explicit same-day exit cutoff for this instrument/date."""


class TickSizeResolver(Protocol):
    def tick_size(self, trade_date: date) -> Decimal:
        """Return verified rupee tick size applicable on this trade date."""


class TradingEligibilityResolver(Protocol):
    def is_eligible(self, trade_date: date) -> bool:
        """Return whether a new position may be opened on this date."""


class ExitReason(StrEnum):
    SIGNAL = "signal"
    SESSION_CUTOFF = "session_cutoff"


@dataclass(frozen=True)
class FillAssumptions:
    """Explicit execution-friction assumptions; tick size is resolved separately by date."""

    slippage_bps_per_leg: Decimal
    half_spread_bps_per_leg: Decimal

    def __post_init__(self) -> None:
        if self.slippage_bps_per_leg < 0:
            raise ValueError("slippage_bps_per_leg cannot be negative")
        if self.half_spread_bps_per_leg < 0:
            raise ValueError("half_spread_bps_per_leg cannot be negative")


@dataclass(frozen=True)
class IntradaySimulationConfig:
    initial_cash: Decimal
    max_trades_per_day: int

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.max_trades_per_day <= 0:
            raise ValueError("max_trades_per_day must be positive")


@dataclass(frozen=True)
class TradeRecord:
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    quantity: int
    reference_entry_price: Decimal
    reference_exit_price: Decimal
    fill_entry_price: Decimal
    fill_exit_price: Decimal
    entry_tick_size_rupees: Decimal
    exit_tick_size_rupees: Decimal
    entry_cost: Decimal
    exit_cost: Decimal
    gross_reference_pnl: Decimal
    execution_friction_cost: Decimal
    net_pnl: Decimal
    exit_reason: ExitReason


@dataclass(frozen=True)
class RejectedSignal:
    timestamp: pd.Timestamp
    reason: str


@dataclass(frozen=True)
class IntradaySimulationResult:
    initial_cash: Decimal
    final_cash: Decimal
    trades: tuple[TradeRecord, ...]
    rejected_signals: tuple[RejectedSignal, ...]

    @property
    def net_pnl(self) -> Decimal:
        return self.final_cash - self.initial_cash

    @property
    def total_transaction_costs(self) -> Decimal:
        return sum((trade.entry_cost + trade.exit_cost for trade in self.trades), Decimal("0"))

    @property
    def total_execution_friction(self) -> Decimal:
        return sum((trade.execution_friction_cost for trade in self.trades), Decimal("0"))


def _as_decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _round_to_tick_adverse(price: Decimal, *, tick_size: Decimal, side: Side) -> Decimal:
    if tick_size <= 0:
        raise ValueError("resolved tick size must be positive")
    units = price / tick_size
    rounding = ROUND_CEILING if side is Side.BUY else ROUND_FLOOR
    return units.to_integral_value(rounding=rounding) * tick_size


def _modeled_fill_price(
    reference_price: Decimal,
    *,
    side: Side,
    assumptions: FillAssumptions,
    tick_size: Decimal,
) -> Decimal:
    friction_bps = assumptions.slippage_bps_per_leg + assumptions.half_spread_bps_per_leg
    multiplier = Decimal("1") + (
        friction_bps / _BPS if side is Side.BUY else -friction_bps / _BPS
    )
    raw = reference_price * multiplier
    return _round_to_tick_adverse(raw, tick_size=tick_size, side=side)


def simulate_long_intraday(
    *,
    frame: pd.DataFrame,
    entries_at_close: pd.Series,
    exits_at_close: pd.Series,
    instrument_token: str,
    exchange: Exchange,
    cost_provider: CostProvider,
    fills: FillAssumptions,
    session_policy: SessionExitResolver,
    tick_size_policy: TickSizeResolver,
    trading_eligibility_policy: TradingEligibilityResolver,
    config: IntradaySimulationConfig,
) -> IntradaySimulationResult:
    """Simulate long-only intraday trades with point-in-time market structure.

    Tick evidence is resolved lazily only when an exchange-eligible entry is actually evaluated.
    This avoids demanding tick evidence on pre-listing/ineligible dates. An intraday position uses
    its entry-date tick for its same-session exit; tick size cannot change within that position's
    trading date in this daily reference model.
    """

    violations = validate_ohlcv_frame(frame)
    if violations:
        raise ValueError("invalid OHLCV frame: " + "; ".join(violations))
    if frame.index.tz is None:
        raise ValueError("intraday frame must use timezone-aware timestamps")
    if not frame.index.equals(entries_at_close.index) or not frame.index.equals(exits_at_close.index):
        raise ValueError("frame, entries and exits must share the same index")

    entries = entries_at_close.astype(bool)
    exits = exits_at_close.astype(bool)
    cash = config.initial_cash
    trades: list[TradeRecord] = []
    rejected: list[RejectedSignal] = []
    trades_by_day: dict[object, int] = {}
    position: dict[str, object] | None = None

    for i in range(1, len(frame)):
        previous_ts = frame.index[i - 1]
        current_ts = frame.index[i]
        previous_day = previous_ts.date()
        current_day = current_ts.date()
        session_exit_time = session_policy.exit_time(current_day)
        current_open = _as_decimal(frame.iloc[i]["open"])

        if position is not None and position["entry_timestamp"].date() != current_day:
            raise ValueError("simulation would carry an intraday position overnight; verify session data/cutoff")

        if position is not None:
            should_exit_cutoff = current_ts.time() >= session_exit_time
            should_exit_signal = previous_day == current_day and bool(exits.iloc[i - 1])
            if should_exit_cutoff or should_exit_signal:
                reference_exit = current_open
                exit_tick_size = position["entry_tick_size_rupees"]
                fill_exit = _modeled_fill_price(
                    reference_exit,
                    side=Side.SELL,
                    assumptions=fills,
                    tick_size=exit_tick_size,
                )
                quantity = int(position["quantity"])
                exit_order = OrderSpec(
                    instrument_token=instrument_token,
                    exchange=exchange,
                    side=Side.SELL,
                    product=Product.INTRADAY,
                    quantity=quantity,
                    price=fill_exit,
                )
                exit_quote = cost_provider.quote(exit_order)
                cash += exit_order.notional - exit_quote.total

                reference_entry = position["reference_entry_price"]
                fill_entry = position["fill_entry_price"]
                entry_cost = position["entry_cost"]
                gross_reference_pnl = (reference_exit - reference_entry) * quantity
                fill_price_pnl = (fill_exit - fill_entry) * quantity
                execution_friction_cost = gross_reference_pnl - fill_price_pnl
                net_pnl = fill_price_pnl - entry_cost - exit_quote.total

                trades.append(
                    TradeRecord(
                        entry_timestamp=position["entry_timestamp"],
                        exit_timestamp=current_ts,
                        quantity=quantity,
                        reference_entry_price=reference_entry,
                        reference_exit_price=reference_exit,
                        fill_entry_price=fill_entry,
                        fill_exit_price=fill_exit,
                        entry_tick_size_rupees=position["entry_tick_size_rupees"],
                        exit_tick_size_rupees=exit_tick_size,
                        entry_cost=entry_cost,
                        exit_cost=exit_quote.total,
                        gross_reference_pnl=gross_reference_pnl,
                        execution_friction_cost=execution_friction_cost,
                        net_pnl=net_pnl,
                        exit_reason=(ExitReason.SESSION_CUTOFF if should_exit_cutoff else ExitReason.SIGNAL),
                    )
                )
                position = None

        if position is not None:
            continue
        if previous_day != current_day:
            continue
        if current_ts.time() >= session_exit_time:
            if bool(entries.iloc[i - 1]):
                rejected.append(RejectedSignal(current_ts, "entry at/after session cutoff"))
            continue
        if not bool(entries.iloc[i - 1]):
            continue

        if not trading_eligibility_policy.is_eligible(current_day):
            rejected.append(
                RejectedSignal(current_ts, "exchange not eligible for new entry on trade date")
            )
            continue

        day_trade_count = trades_by_day.get(current_day, 0)
        if day_trade_count >= config.max_trades_per_day:
            rejected.append(RejectedSignal(current_ts, "daily trade limit reached"))
            continue

        current_tick_size = tick_size_policy.tick_size(current_day)
        if current_tick_size <= 0:
            raise ValueError(f"non-positive tick size for {current_day}")

        reference_entry = current_open
        fill_entry = _modeled_fill_price(
            reference_entry,
            side=Side.BUY,
            assumptions=fills,
            tick_size=current_tick_size,
        )
        size = max_affordable_buy_quantity(
            instrument_token=instrument_token,
            exchange=exchange,
            product=Product.INTRADAY,
            price=fill_entry,
            cash_limit=cash,
            cost_provider=cost_provider,
        )
        if size.quantity <= 0:
            rejected.append(RejectedSignal(current_ts, "insufficient cash after modeled entry charges"))
            continue

        entry_order = OrderSpec(
            instrument_token=instrument_token,
            exchange=exchange,
            side=Side.BUY,
            product=Product.INTRADAY,
            quantity=size.quantity,
            price=fill_entry,
        )
        entry_quote = cost_provider.quote(entry_order)
        required_cash = entry_order.notional + entry_quote.total
        if required_cash > cash:
            rejected.append(RejectedSignal(current_ts, "entry quote changed beyond available cash"))
            continue

        cash -= required_cash
        position = {
            "entry_timestamp": current_ts,
            "quantity": size.quantity,
            "reference_entry_price": reference_entry,
            "fill_entry_price": fill_entry,
            "entry_tick_size_rupees": current_tick_size,
            "entry_cost": entry_quote.total,
        }
        trades_by_day[current_day] = day_trade_count + 1

    if position is not None:
        raise ValueError("dataset ended with an open intraday position; include bars through the session cutoff")

    return IntradaySimulationResult(
        initial_cash=config.initial_cash,
        final_cash=cash,
        trades=tuple(trades),
        rejected_signals=tuple(rejected),
    )
