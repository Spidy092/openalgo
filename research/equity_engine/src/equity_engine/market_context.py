from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

_BPS = Decimal("10000")
_HUNDRED = Decimal("100")


@dataclass(frozen=True)
class MarketSnapshot:
    captured_at: datetime
    nifty_open: Decimal
    nifty_last: Decimal
    nifty_previous_close: Decimal
    india_vix_last: Decimal
    india_vix_previous_close: Decimal
    advancers: int
    decliners: int
    unchanged: int

    def __post_init__(self) -> None:
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        for name in (
            "nifty_open",
            "nifty_last",
            "nifty_previous_close",
            "india_vix_last",
            "india_vix_previous_close",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if min(self.advancers, self.decliners, self.unchanged) < 0:
            raise ValueError("breadth counts cannot be negative")
        if self.advancers + self.decliners + self.unchanged == 0:
            raise ValueError("breadth universe cannot be empty")


@dataclass(frozen=True)
class MarketContextThresholds:
    """Caller-supplied thresholds; there are intentionally no trading defaults."""

    large_gap_bps: Decimal
    strong_index_move_bps: Decimal
    high_vix_level: Decimal
    vix_jump_pct: Decimal
    strong_breadth_pct: Decimal
    weak_breadth_pct: Decimal

    def __post_init__(self) -> None:
        if self.large_gap_bps < 0 or self.strong_index_move_bps < 0:
            raise ValueError("gap/move thresholds cannot be negative")
        if self.high_vix_level <= 0 or self.vix_jump_pct < 0:
            raise ValueError("VIX thresholds are invalid")
        if not (Decimal("0") <= self.weak_breadth_pct <= _HUNDRED):
            raise ValueError("weak_breadth_pct must be between 0 and 100")
        if not (Decimal("0") <= self.strong_breadth_pct <= _HUNDRED):
            raise ValueError("strong_breadth_pct must be between 0 and 100")
        if self.weak_breadth_pct >= self.strong_breadth_pct:
            raise ValueError("weak_breadth_pct must be below strong_breadth_pct")


@dataclass(frozen=True)
class MarketContext:
    nifty_gap_bps: Decimal
    nifty_change_bps: Decimal
    india_vix_change_pct: Decimal
    breadth_advancers_pct: Decimal
    large_gap: bool
    high_volatility: bool
    index_strong_up: bool
    index_strong_down: bool
    breadth_strong: bool
    breadth_weak: bool


def build_market_context(
    snapshot: MarketSnapshot,
    *,
    thresholds: MarketContextThresholds,
) -> MarketContext:
    """Convert an observed market snapshot into transparent measurements and flags.

    This function does not decide BUY/SELL and does not choose a strategy. Strategy-routing rules
    must consume these flags explicitly so they can be separately backtested and audited.
    """

    gap_bps = (snapshot.nifty_open / snapshot.nifty_previous_close - Decimal("1")) * _BPS
    change_bps = (snapshot.nifty_last / snapshot.nifty_previous_close - Decimal("1")) * _BPS
    vix_change_pct = (
        snapshot.india_vix_last / snapshot.india_vix_previous_close - Decimal("1")
    ) * _HUNDRED
    breadth_total = snapshot.advancers + snapshot.decliners + snapshot.unchanged
    breadth_pct = Decimal(snapshot.advancers) / Decimal(breadth_total) * _HUNDRED

    return MarketContext(
        nifty_gap_bps=gap_bps,
        nifty_change_bps=change_bps,
        india_vix_change_pct=vix_change_pct,
        breadth_advancers_pct=breadth_pct,
        large_gap=abs(gap_bps) >= thresholds.large_gap_bps,
        high_volatility=(
            snapshot.india_vix_last >= thresholds.high_vix_level
            or vix_change_pct >= thresholds.vix_jump_pct
        ),
        index_strong_up=change_bps >= thresholds.strong_index_move_bps,
        index_strong_down=change_bps <= -thresholds.strong_index_move_bps,
        breadth_strong=breadth_pct >= thresholds.strong_breadth_pct,
        breadth_weak=breadth_pct <= thresholds.weak_breadth_pct,
    )
