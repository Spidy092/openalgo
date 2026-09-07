from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .instrument_master import EquityInstrument


@dataclass(frozen=True)
class UniverseThresholds:
    """All research-universe thresholds are caller supplied; there are no silent defaults."""

    max_last_price_rupees: Decimal
    min_median_daily_turnover_rupees: Decimal
    min_median_daily_volume_shares: Decimal
    max_median_spread_bps: Decimal
    min_observed_trading_days: int
    min_affordable_quantity: int

    def __post_init__(self) -> None:
        if self.max_last_price_rupees <= 0:
            raise ValueError("max_last_price_rupees must be positive")
        if self.min_median_daily_turnover_rupees < 0:
            raise ValueError("min_median_daily_turnover_rupees cannot be negative")
        if self.min_median_daily_volume_shares < 0:
            raise ValueError("min_median_daily_volume_shares cannot be negative")
        if self.max_median_spread_bps < 0:
            raise ValueError("max_median_spread_bps cannot be negative")
        if self.min_observed_trading_days <= 0:
            raise ValueError("min_observed_trading_days must be positive")
        if self.min_affordable_quantity <= 0:
            raise ValueError("min_affordable_quantity must be positive")


@dataclass(frozen=True)
class LiquidityEvidence:
    """Measured evidence for one instrument over an explicitly chosen observation window."""

    last_price_rupees: Decimal
    median_daily_turnover_rupees: Decimal
    median_daily_volume_shares: Decimal
    median_spread_bps: Decimal
    observed_trading_days: int
    affordable_quantity_after_entry_costs: int
    spread_observations: int
    source_complete: bool

    def __post_init__(self) -> None:
        if self.last_price_rupees <= 0:
            raise ValueError("last_price_rupees must be positive")
        if self.median_daily_turnover_rupees < 0:
            raise ValueError("median_daily_turnover_rupees cannot be negative")
        if self.median_daily_volume_shares < 0:
            raise ValueError("median_daily_volume_shares cannot be negative")
        if self.median_spread_bps < 0:
            raise ValueError("median_spread_bps cannot be negative")
        if self.observed_trading_days < 0:
            raise ValueError("observed_trading_days cannot be negative")
        if self.affordable_quantity_after_entry_costs < 0:
            raise ValueError("affordable quantity cannot be negative")
        if self.spread_observations < 0:
            raise ValueError("spread_observations cannot be negative")


@dataclass(frozen=True)
class CorporateActionAssessment:
    complete: bool
    blocking_events: tuple[str, ...]


@dataclass(frozen=True)
class UniverseDecision:
    eligible: bool
    violations: tuple[str, ...]


def evaluate_universe_candidate(
    *,
    instrument: EquityInstrument,
    liquidity: LiquidityEvidence,
    corporate_actions: CorporateActionAssessment,
    thresholds: UniverseThresholds,
) -> UniverseDecision:
    """Fail closed when broker eligibility, liquidity, affordability or data evidence is incomplete."""

    violations: list[str] = []

    if instrument.exchange != "NSE" or instrument.segment != "NSE_EQ":
        violations.append("instrument is not NSE cash equity")
    if instrument.instrument_type != "EQ":
        violations.append(f"instrument_type {instrument.instrument_type!r} is not EQ")
    if instrument.security_type != "NORMAL":
        violations.append(f"security_type {instrument.security_type!r} is not NORMAL")
    if not instrument.mis_eligible:
        violations.append("instrument is not present in Upstox NSE MIS list")
    if instrument.suspended:
        violations.append("instrument is present in Upstox suspended list")
    if instrument.tick_size_rupees <= 0:
        violations.append("resolved tick size is unavailable")

    if not liquidity.source_complete:
        violations.append("liquidity evidence is incomplete")
    if liquidity.last_price_rupees > thresholds.max_last_price_rupees:
        violations.append(
            f"last price ₹{liquidity.last_price_rupees} exceeds research cap "
            f"₹{thresholds.max_last_price_rupees}"
        )
    if liquidity.median_daily_turnover_rupees < thresholds.min_median_daily_turnover_rupees:
        violations.append("median daily turnover is below required threshold")
    if liquidity.median_daily_volume_shares < thresholds.min_median_daily_volume_shares:
        violations.append("median daily volume is below required threshold")
    if liquidity.median_spread_bps > thresholds.max_median_spread_bps:
        violations.append("median bid/ask spread exceeds allowed threshold")
    if liquidity.observed_trading_days < thresholds.min_observed_trading_days:
        violations.append("insufficient observed trading-history days")
    if liquidity.affordable_quantity_after_entry_costs < thresholds.min_affordable_quantity:
        violations.append("₹1,000 account cannot afford required quantity after entry costs")
    if liquidity.spread_observations <= 0:
        violations.append("no measured spread observations")

    if not corporate_actions.complete:
        violations.append("corporate-action evidence is incomplete")
    if corporate_actions.blocking_events:
        violations.append(
            "blocking corporate actions in research window: "
            + ", ".join(corporate_actions.blocking_events)
        )

    return UniverseDecision(eligible=not violations, violations=tuple(violations))
