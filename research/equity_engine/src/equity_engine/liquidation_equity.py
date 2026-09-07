from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from .costs import CostProvider
from .event_simulator import FillAssumptions, IntradaySimulationResult, _modeled_fill_price
from .models import Exchange, OrderSpec, Product, Side


@dataclass(frozen=True)
class EquityObservation:
    timestamp: pd.Timestamp
    realized_cash: Decimal
    position_quantity: int
    close_liquidation_equity: Decimal
    ohlc_low_liquidation_stress_equity: Decimal


@dataclass(frozen=True)
class LiquidationDrawdownMetrics:
    close_liquidation_max_drawdown_pct: Decimal
    ohlc_low_liquidation_stress_max_drawdown_pct: Decimal


def _liquidation_equity(
    *,
    cash_after_entry: Decimal,
    quantity: int,
    reference_price: Decimal,
    tick_size: Decimal,
    instrument_token: str,
    exchange: Exchange,
    cost_provider: CostProvider,
    fills: FillAssumptions,
) -> Decimal:
    fill = _modeled_fill_price(
        reference_price,
        side=Side.SELL,
        assumptions=fills,
        tick_size=tick_size,
    )
    order = OrderSpec(
        instrument_token=instrument_token,
        exchange=exchange,
        side=Side.SELL,
        product=Product.INTRADAY,
        quantity=quantity,
        price=fill,
    )
    quote = cost_provider.quote(order)
    return cash_after_entry + order.notional - quote.total


def build_liquidation_equity_curve(
    *,
    frame: pd.DataFrame,
    simulation: IntradaySimulationResult,
    instrument_token: str,
    exchange: Exchange,
    cost_provider: CostProvider,
    fills: FillAssumptions,
) -> tuple[EquityObservation, ...]:
    """Reconstruct liquidation-value equity at each observed bar.

    `close_liquidation_equity` assumes the open long were liquidated using the observed bar close,
    then applies the same adverse spread/slippage/tick model and exit charges as the simulator.
    `ohlc_low_liquidation_stress_equity` does the same from the bar low. The latter is a stress
    measure, not a claim about the unknown order of intrabar prices.
    """

    if frame.empty:
        raise ValueError("cannot build liquidation equity curve from an empty frame")
    if frame.index.tz is None:
        raise ValueError("liquidation equity frame requires timezone-aware timestamps")

    trades = tuple(sorted(simulation.trades, key=lambda item: item.entry_timestamp))
    if tuple(simulation.trades) != trades:
        raise ValueError("simulation trades must be chronological")

    observations: list[EquityObservation] = []
    realized_cash = simulation.initial_cash
    trade_index = 0

    for timestamp, row in frame.iterrows():
        while trade_index < len(trades) and trades[trade_index].exit_timestamp <= timestamp:
            realized_cash += trades[trade_index].net_pnl
            trade_index += 1

        active = None
        if trade_index < len(trades):
            candidate = trades[trade_index]
            if candidate.entry_timestamp <= timestamp < candidate.exit_timestamp:
                active = candidate

        if active is None:
            close_equity = realized_cash
            low_equity = realized_cash
            quantity = 0
        else:
            quantity = active.quantity
            cash_after_entry = (
                realized_cash
                - active.fill_entry_price * quantity
                - active.entry_cost
            )
            close_equity = _liquidation_equity(
                cash_after_entry=cash_after_entry,
                quantity=quantity,
                reference_price=Decimal(str(row["close"])),
                tick_size=active.entry_tick_size_rupees,
                instrument_token=instrument_token,
                exchange=exchange,
                cost_provider=cost_provider,
                fills=fills,
            )
            low_equity = _liquidation_equity(
                cash_after_entry=cash_after_entry,
                quantity=quantity,
                reference_price=Decimal(str(row["low"])),
                tick_size=active.entry_tick_size_rupees,
                instrument_token=instrument_token,
                exchange=exchange,
                cost_provider=cost_provider,
                fills=fills,
            )

        observations.append(
            EquityObservation(
                timestamp=timestamp,
                realized_cash=realized_cash,
                position_quantity=quantity,
                close_liquidation_equity=close_equity,
                ohlc_low_liquidation_stress_equity=low_equity,
            )
        )

    if trade_index != len(trades):
        raise ValueError("equity curve ended before all simulated trades were closed")
    if observations[-1].close_liquidation_equity != simulation.final_cash:
        raise ValueError("final liquidation equity does not reconcile to simulation final cash")
    return tuple(observations)


def _max_drawdown_pct(values: tuple[Decimal, ...], *, initial_equity: Decimal) -> Decimal:
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    peak = initial_equity
    maximum = Decimal("0")
    for value in values:
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak * Decimal("100")
            if drawdown > maximum:
                maximum = drawdown
    return maximum


def liquidation_drawdown_metrics(
    observations: tuple[EquityObservation, ...],
    *,
    initial_equity: Decimal,
) -> LiquidationDrawdownMetrics:
    if not observations:
        raise ValueError("equity observations cannot be empty")
    close_values = tuple(item.close_liquidation_equity for item in observations)
    low_values = tuple(item.ohlc_low_liquidation_stress_equity for item in observations)
    return LiquidationDrawdownMetrics(
        close_liquidation_max_drawdown_pct=_max_drawdown_pct(
            close_values,
            initial_equity=initial_equity,
        ),
        ohlc_low_liquidation_stress_max_drawdown_pct=_max_drawdown_pct(
            low_values,
            initial_equity=initial_equity,
        ),
    )
