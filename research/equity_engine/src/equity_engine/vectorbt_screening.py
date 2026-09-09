from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math

import pandas as pd

from .market_sessions import ContinuousSessionPolicy, filter_to_continuous_session


@dataclass(frozen=True)
class VectorBTScreeningResult:
    total_return_pct: Decimal
    max_drawdown_pct: Decimal
    closed_trades: int
    win_rate_pct: Decimal | None
    profit_factor: Decimal | None
    signal_lag_bars: int
    exact_cost_validated: bool = False


def shift_close_generated_signals(
    entries: pd.Series,
    exits: pd.Series,
    *,
    lag_bars: int,
) -> tuple[pd.Series, pd.Series]:
    """Move close-generated signals to later bars without carrying across sessions.

    Signals are assumed to be generated from information known only at each bar close. They must
    therefore execute on a later bar. For intraday research we additionally forbid a signal from
    the final bar of one trading day from becoming an order on the next trading day.
    """

    if lag_bars < 1:
        raise ValueError("close-generated signals require lag_bars >= 1")
    if not entries.index.equals(exits.index):
        raise ValueError("entry and exit signal indexes must match")
    if entries.index.tz is None:
        raise ValueError("intraday screening requires timezone-aware timestamps")

    dates = pd.Series(entries.index.date, index=entries.index)
    same_session = dates.eq(dates.shift(lag_bars))
    shifted_entries = entries.astype(bool).shift(lag_bars, fill_value=False) & same_session
    shifted_exits = exits.astype(bool).shift(lag_bars, fill_value=False) & same_session
    return shifted_entries, shifted_exits


def _decimal_or_none(value: object) -> Decimal | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return Decimal(str(numeric))


def screen_long_signals(
    *,
    close: pd.Series,
    execution_price: pd.Series,
    entries_at_close: pd.Series,
    exits_at_close: pd.Series,
    signal_lag_bars: int,
    screening_cash: Decimal,
    screening_fee_rate: Decimal,
    screening_slippage_rate: Decimal,
    frequency: str,
    session_policy: ContinuousSessionPolicy,
) -> VectorBTScreeningResult:
    """Fast candidate screening only; not an exact brokerage/P&L validator.

    `close` is the mark-to-market series. `execution_price` is a separate, explicit order-price
    series (normally bar open for next-bar execution). Fees/slippage are mandatory caller inputs.
    Surviving strategies must still be re-priced by the Decimal event-driven simulator and broker
    cost reconciliation before any promotion decision.
    """

    if screening_cash <= 0:
        raise ValueError("screening_cash must be positive")
    if screening_fee_rate < 0:
        raise ValueError("screening_fee_rate cannot be negative")
    if screening_slippage_rate < 0:
        raise ValueError("screening_slippage_rate cannot be negative")
    if close.empty:
        raise ValueError("close series cannot be empty")
    if not close.index.equals(execution_price.index):
        raise ValueError("close and execution_price must share the same index")
    if not close.index.equals(entries_at_close.index) or not close.index.equals(
        exits_at_close.index
    ):
        raise ValueError("prices, entries and exits must share the same index")
    session_index = filter_to_continuous_session(
        pd.DataFrame(index=close.index),
        session_policy,
    ).index
    close = close.loc[session_index]
    execution_price = execution_price.loc[session_index]
    entries_at_close = entries_at_close.loc[session_index]
    exits_at_close = exits_at_close.loc[session_index]
    if close.empty:
        raise ValueError("continuous-session price series cannot be empty")
    if close.isna().any() or execution_price.isna().any():
        raise ValueError("price series contain missing values")
    if (close <= 0).any() or (execution_price <= 0).any():
        raise ValueError("price series must be positive")

    shifted_entries, shifted_exits = shift_close_generated_signals(
        entries_at_close,
        exits_at_close,
        lag_bars=signal_lag_bars,
    )

    # Lazy import keeps the production OpenAlgo process independent of VectorBT.
    import vectorbt as vbt

    portfolio = vbt.Portfolio.from_signals(
        close.astype(float),
        entries=shifted_entries,
        exits=shifted_exits,
        price=execution_price.astype(float),
        init_cash=float(screening_cash),
        direction="longonly",
        fees=float(screening_fee_rate),
        slippage=float(screening_slippage_rate),
        freq=frequency,
    )
    stats = portfolio.stats(settings=dict(incl_open=False))

    total_return = _decimal_or_none(stats.get("Total Return [%]"))
    max_drawdown = _decimal_or_none(stats.get("Max Drawdown [%]"))
    if total_return is None or max_drawdown is None:
        raise RuntimeError("VectorBT did not produce finite return/drawdown metrics")

    closed_trades_value = stats.get("Total Closed Trades", 0)
    try:
        closed_trades = int(closed_trades_value)
    except (TypeError, ValueError):
        closed_trades = 0

    return VectorBTScreeningResult(
        total_return_pct=total_return,
        max_drawdown_pct=max_drawdown,
        closed_trades=closed_trades,
        win_rate_pct=_decimal_or_none(stats.get("Win Rate [%]")),
        profit_factor=_decimal_or_none(stats.get("Profit Factor")),
        signal_lag_bars=signal_lag_bars,
        exact_cost_validated=False,
    )
