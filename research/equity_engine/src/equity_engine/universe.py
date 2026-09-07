from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .instrument_master import EquityInstrument


@dataclass(frozen=True)
class ResearchUniverseThresholds:
    """Historical-research thresholds. Every value is explicit; there are no defaults."""

    max_last_price_rupees: Decimal
    min_median_daily_notional_proxy_rupees: Decimal
    min_median_daily_volume_shares: Decimal
    min_observed_trading_days: int
    min_affordable_quantity: int

    def __post_init__(self) -> None:
        if self.max_last_price_rupees <= 0:
            raise ValueError("max_last_price_rupees must be positive")
        if self.min_median_daily_notional_proxy_rupees < 0:
            raise ValueError("minimum notional proxy cannot be negative")
        if self.min_median_daily_volume_shares < 0:
            raise ValueError("minimum daily volume cannot be negative")
        if self.min_observed_trading_days <= 0:
            raise ValueError("min_observed_trading_days must be positive")
        if self.min_affordable_quantity <= 0:
            raise ValueError("min_affordable_quantity must be positive")


@dataclass(frozen=True)
class LiveUniverseThresholds:
    """Additional thresholds required before paper/live eligibility."""

    max_median_spread_bps: Decimal
    min_spread_observations: int

    def __post_init__(self) -> None:
        if self.max_median_spread_bps < 0:
            raise ValueError("max_median_spread_bps cannot be negative")
        if self.min_spread_observations <= 0:
            raise ValueError("min_spread_observations must be positive")


@dataclass(frozen=True)
class HistoricalLiquidityEvidence:
    """OHLCV-derived evidence; notional is explicitly a proxy, not exchange turnover."""

    last_price_rupees: Decimal
    median_daily_notional_proxy_rupees: Decimal
    median_daily_volume_shares: Decimal
    observed_trading_days: int
    affordable_quantity_after_entry_costs: int
    source_complete: bool

    def __post_init__(self) -> None:
        if self.last_price_rupees <= 0:
            raise ValueError("last_price_rupees must be positive")
        if self.median_daily_notional_proxy_rupees < 0:
            raise ValueError("notional proxy cannot be negative")
        if self.median_daily_volume_shares < 0:
            raise ValueError("daily volume cannot be negative")
        if self.observed_trading_days < 0:
            raise ValueError("observed_trading_days cannot be negative")
        if self.affordable_quantity_after_entry_costs < 0:
            raise ValueError("affordable quantity cannot be negative")


@dataclass(frozen=True)
class LiveSpreadEvidence:
    median_spread_bps: Decimal
    spread_observations: int
    source_complete: bool

    def __post_init__(self) -> None:
        if self.median_spread_bps < 0:
            raise ValueError("median_spread_bps cannot be negative")
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


def _static_violations(instrument: EquityInstrument) -> list[str]:
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
    return violations


def evaluate_research_universe_candidate(
    *,
    instrument: EquityInstrument,
    liquidity: HistoricalLiquidityEvidence,
    corporate_actions: CorporateActionAssessment,
    thresholds: ResearchUniverseThresholds,
) -> UniverseDecision:
    """Eligibility for historical strategy research; does not pretend OHLC contains spread."""

    violations = _static_violations(instrument)

    if not liquidity.source_complete:
        violations.append("historical liquidity evidence is incomplete")
    if liquidity.last_price_rupees > thresholds.max_last_price_rupees:
        violations.append(
            f"last price ₹{liquidity.last_price_rupees} exceeds research cap "
            f"₹{thresholds.max_last_price_rupees}"
        )
    if (
        liquidity.median_daily_notional_proxy_rupees
        < thresholds.min_median_daily_notional_proxy_rupees
    ):
        violations.append("median daily notional proxy is below required threshold")
    if liquidity.median_daily_volume_shares < thresholds.min_median_daily_volume_shares:
        violations.append("median daily volume is below required threshold")
    if liquidity.observed_trading_days < thresholds.min_observed_trading_days:
        violations.append("insufficient observed trading-history days")
    if liquidity.affordable_quantity_after_entry_costs < thresholds.min_affordable_quantity:
        violations.append("₹1,000 account cannot afford required quantity after entry costs")

    if not corporate_actions.complete:
        violations.append("corporate-action evidence is incomplete")
    if corporate_actions.blocking_events:
        violations.append(
            "blocking corporate actions in research window: "
            + ", ".join(corporate_actions.blocking_events)
        )

    return UniverseDecision(eligible=not violations, violations=tuple(violations))


def evaluate_live_universe_candidate(
    *,
    research_decision: UniverseDecision,
    spread: LiveSpreadEvidence,
    thresholds: LiveUniverseThresholds,
) -> UniverseDecision:
    """Add measured live bid/ask evidence; historical OHLC is never used as a spread substitute."""

    violations = list(research_decision.violations)
    if not research_decision.eligible and not violations:
        violations.append("research universe decision did not pass")
    if not spread.source_complete:
        violations.append("live spread evidence is incomplete")
    if spread.spread_observations < thresholds.min_spread_observations:
        violations.append("insufficient live spread observations")
    if spread.median_spread_bps > thresholds.max_median_spread_bps:
        violations.append("median bid/ask spread exceeds allowed threshold")
    return UniverseDecision(eligible=not violations, violations=tuple(violations))
