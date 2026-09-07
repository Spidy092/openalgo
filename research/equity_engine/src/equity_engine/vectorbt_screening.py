from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math

import pandas as pd


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
    """Move close-generated signals to a later executable bar.

    VectorBT documentation explicitly warns that signals generated with the close price must be
    shifted forward so execution uses a price that comes after the signal. We require at least
    one bar of lag rather than making same-bar close execution an option.
    """

    if lag_bars < 1:
        raise ValueError("close-generated signals require lag_bars >= 1")
    if not entries.index.equals(exits.index):
        raise ValueError("entry and exit signal indexes must match")

    return (
        entries.astype(bool).shift(lag_bars, fill_value=False),
        exits.astype(bool).shift(lag_bars, fill_value=False),
    )


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
    entries_at_close: pd.Series,
    exits_at_close: pd.Series,
    signal_lag_bars: int,
    screening_cash: Decimal,
    screening_fee_rate: Decimal,
    screening_slippage_rate: Decimal,
    frequency: str,
) -> VectorBTScreeningResult:
    """Fast candidate screening only; not an exact brokerage/P&L validator.

    Fee/slippage rates are mandatory caller inputs: there are deliberately no silent defaults.
    Strategies surviving this stage must be re-priced by the Decimal event-driven simulator and
    broker cost reconciliation before any promotion decision.
    """

    if screening_cash <= 0:
        raise ValueError("screening_cash must be positive")
    if screening_fee_rate < 0:
        raise ValueError("screening_fee_rate cannot be negative")
    if screening_slippage_rate < 0:
        raise ValueError("screening_slippage_rate cannot be negative")
    if close.empty:
        raise ValueError("close series cannot be empty")
    if not close.index.equals(entries_at_close.index) or not close.index.equals(
        exits_at_close.index
    ):
        raise ValueError("close, entries and exits must share the same index")
    if close.isna().any():
        raise ValueError("close series contains missing values")

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
