from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from statistics import median

import pandas as pd

from .provenance import validate_ohlcv_frame
from .universe import HistoricalLiquidityEvidence, LiveSpreadEvidence

_BPS = Decimal("10000")


def _d(value: object) -> Decimal:
    return Decimal(str(value))


def summarize_historical_liquidity(
    *,
    frame: pd.DataFrame,
    affordable_quantity_after_entry_costs: int,
    approved_capital_rupees: Decimal | None = None,
) -> HistoricalLiquidityEvidence:
    """Summarize OHLCV liquidity without mislabeling estimated notional as exchange turnover.

    Daily notional proxy = sum(bar close * bar volume). It is a screening proxy because OHLCV does
    not contain every execution price. The definition is fixed and auditable, but it must not be
    reported as exact exchange turnover.
    """

    violations = validate_ohlcv_frame(frame)
    if violations:
        raise ValueError("invalid OHLCV frame: " + "; ".join(violations))
    if frame.index.tz is None:
        raise ValueError("liquidity frame requires timezone-aware timestamps")
    if affordable_quantity_after_entry_costs < 0:
        raise ValueError("affordable quantity cannot be negative")

    volume_by_day: dict[object, Decimal] = {}
    notional_by_day: dict[object, Decimal] = {}
    for timestamp, row in frame.iterrows():
        day = timestamp.date()
        volume = _d(row["volume"])
        close = _d(row["close"])
        volume_by_day[day] = volume_by_day.get(day, Decimal("0")) + volume
        notional_by_day[day] = notional_by_day.get(day, Decimal("0")) + close * volume

    if not volume_by_day:
        raise ValueError("no trading days in liquidity frame")

    return HistoricalLiquidityEvidence(
        last_price_rupees=_d(frame.iloc[-1]["close"]),
        median_daily_notional_proxy_rupees=median(notional_by_day.values()),
        median_daily_volume_shares=median(volume_by_day.values()),
        observed_trading_days=len(volume_by_day),
        affordable_quantity_after_entry_costs=affordable_quantity_after_entry_costs,
        source_complete=True,
        approved_capital_rupees=approved_capital_rupees,
    )


@dataclass(frozen=True)
class SpreadObservation:
    bid_rupees: Decimal
    ask_rupees: Decimal

    def spread_bps(self) -> Decimal:
        if self.bid_rupees <= 0 or self.ask_rupees <= 0:
            raise ValueError("bid/ask must be positive")
        if self.ask_rupees < self.bid_rupees:
            raise ValueError("crossed market observation is invalid for spread evidence")
        mid = (self.bid_rupees + self.ask_rupees) / Decimal("2")
        return (self.ask_rupees - self.bid_rupees) / mid * _BPS


def summarize_live_spreads(observations: Iterable[SpreadObservation]) -> LiveSpreadEvidence:
    items = list(observations)
    if not items:
        raise ValueError("at least one spread observation is required")
    values = [item.spread_bps() for item in items]
    return LiveSpreadEvidence(
        median_spread_bps=median(values),
        spread_observations=len(values),
        source_complete=True,
    )
