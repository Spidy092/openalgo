from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .historical_membership import HistoricalMembershipAssessment
from .instrument_master import EquityInstrument
from .tick_size import TickCoverageAssessment, TickSizeVerification


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
    instrument_key: str
    eligible: bool
    violations: tuple[str, ...]


def evaluate_research_universe_candidate(
    *,
    instrument_key: str,
    liquidity: HistoricalLiquidityEvidence,
    corporate_actions: CorporateActionAssessment,
    historical_membership: HistoricalMembershipAssessment,
    tick_coverage: TickCoverageAssessment,
    thresholds: ResearchUniverseThresholds,
) -> UniverseDecision:
    """Historical eligibility uses only point-in-time exchange evidence, never today's broker list."""

    if not instrument_key:
        raise ValueError("instrument_key is required")
    violations: list[str] = []

    if historical_membership.instrument_key != instrument_key:
        violations.append("historical membership belongs to a different instrument")
    if not historical_membership.complete:
        violations.append(
            "point-in-time exchange evidence is missing for: "
            + ", ".join(day.isoformat() for day in historical_membership.missing_dates)
        )
    if not historical_membership.eligible_dates:
        violations.append("instrument has no eligible historical trading dates in the research window")

    eligible_dates = set(historical_membership.eligible_dates)
    if set(tick_coverage.requested_dates) != eligible_dates:
        violations.append("tick-size coverage was not evaluated on exactly the eligible historical dates")
    if not tick_coverage.complete:
        violations.append(
            "verified historical tick size is missing for: "
            + ", ".join(day.isoformat() for day in tick_coverage.missing_dates)
        )

    if not liquidity.source_complete:
        violations.append("historical liquidity evidence is incomplete")
    if liquidity.last_price_rupees > thresholds.max_last_price_rupees:
        violations.append(
            f"last price ₹{liquidity.last_price_rupees} exceeds research cap "
            f"₹{thresholds.max_last_price_rupees}"
        )
    if liquidity.median_daily_notional_proxy_rupees < thresholds.min_median_daily_notional_proxy_rupees:
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

    return UniverseDecision(
        instrument_key=instrument_key,
        eligible=not violations,
        violations=tuple(violations),
    )


def evaluate_live_universe_candidate(
    *,
    research_decision: UniverseDecision,
    current_instrument: EquityInstrument,
    current_tick_verification: TickSizeVerification,
    spread: LiveSpreadEvidence,
    thresholds: LiveUniverseThresholds,
) -> UniverseDecision:
    """Add today's broker eligibility and measured spread only after historical research passes."""

    violations = list(research_decision.violations)
    if current_instrument.instrument_key != research_decision.instrument_key:
        violations.append("current broker instrument does not match research instrument")
    if current_instrument.exchange != "NSE" or current_instrument.segment != "NSE_EQ":
        violations.append("current instrument is not NSE cash equity")
    if current_instrument.instrument_type != "EQ":
        violations.append(f"current instrument_type {current_instrument.instrument_type!r} is not EQ")
    if current_instrument.security_type != "NORMAL":
        violations.append(f"current security_type {current_instrument.security_type!r} is not NORMAL")
    if not current_instrument.mis_eligible:
        violations.append("instrument is not present in current Upstox NSE MIS list")
    if current_instrument.suspended:
        violations.append("instrument is present in current Upstox suspended list")
    if not current_tick_verification.passed:
        violations.append(
            "current tick-size verification failed: "
            f"observed ₹{current_tick_verification.observed_rupees} vs "
            f"expected ₹{current_tick_verification.expected_rupees}"
        )
    if not spread.source_complete:
        violations.append("live spread evidence is incomplete")
    if spread.spread_observations < thresholds.min_spread_observations:
        violations.append("insufficient live spread observations")
    if spread.median_spread_bps > thresholds.max_median_spread_bps:
        violations.append("median bid/ask spread exceeds allowed threshold")

    return UniverseDecision(
        instrument_key=research_decision.instrument_key,
        eligible=not violations,
        violations=tuple(violations),
    )
