from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class DrawdownBasis(StrEnum):
    REALIZED_CLOSED_TRADES = "realized_closed_trades"
    CLOSE_LIQUIDATION = "close_liquidation"
    OHLC_LOW_LIQUIDATION_STRESS = "ohlc_low_liquidation_stress"


@dataclass(frozen=True)
class PromotionThresholds:
    """Thresholds are intentionally caller-supplied: there are no hidden defaults."""

    min_trades: int
    min_profit_factor: Decimal
    max_drawdown_pct: Decimal
    min_walk_forward_windows: int
    max_cost_reconciliation_error_inr: Decimal

    def __post_init__(self) -> None:
        if self.min_trades <= 0:
            raise ValueError("min_trades must be positive")
        if self.min_profit_factor <= 0:
            raise ValueError("min_profit_factor must be positive")
        if self.max_drawdown_pct < 0:
            raise ValueError("max_drawdown_pct cannot be negative")
        if self.min_walk_forward_windows <= 0:
            raise ValueError("min_walk_forward_windows must be positive")
        if self.max_cost_reconciliation_error_inr < 0:
            raise ValueError("cost reconciliation tolerance cannot be negative")


@dataclass(frozen=True)
class ResearchEvidence:
    trade_count: int
    profit_factor: Decimal
    max_drawdown_pct: Decimal
    drawdown_basis: DrawdownBasis
    walk_forward_windows: int
    max_cost_reconciliation_error_inr: Decimal | None
    held_out_test_present: bool
    baseline_comparison_present: bool
    slippage_stress_present: bool
    event_driven_validation_present: bool
    paper_trading_present: bool
    data_provenance_complete: bool
    unpriced_cost_components: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    violations: tuple[str, ...]


def evaluate_promotion_gate(
    evidence: ResearchEvidence,
    thresholds: PromotionThresholds,
) -> GateDecision:
    violations: list[str] = []

    if not evidence.data_provenance_complete:
        violations.append("market-data provenance is incomplete")
    if not evidence.held_out_test_present:
        violations.append("held-out test evidence is missing")
    if not evidence.baseline_comparison_present:
        violations.append("baseline comparison is missing")
    if not evidence.slippage_stress_present:
        violations.append("slippage/spread stress test is missing")
    if not evidence.event_driven_validation_present:
        violations.append("event-driven validation is missing")
    if not evidence.paper_trading_present:
        violations.append("paper-trading evidence is missing")
    if evidence.unpriced_cost_components:
        violations.append(
            "unpriced cost components remain: " + ", ".join(evidence.unpriced_cost_components)
        )

    if evidence.drawdown_basis is not DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS:
        violations.append(
            "promotion drawdown must use OHLC-low liquidation stress; "
            f"received {evidence.drawdown_basis.value}"
        )
    if evidence.trade_count < thresholds.min_trades:
        violations.append(
            f"trade count {evidence.trade_count} is below required {thresholds.min_trades}"
        )
    if evidence.profit_factor < thresholds.min_profit_factor:
        violations.append(
            f"profit factor {evidence.profit_factor} is below required {thresholds.min_profit_factor}"
        )
    if evidence.max_drawdown_pct > thresholds.max_drawdown_pct:
        violations.append(
            f"max drawdown {evidence.max_drawdown_pct}% exceeds allowed "
            f"{thresholds.max_drawdown_pct}%"
        )
    if evidence.walk_forward_windows < thresholds.min_walk_forward_windows:
        violations.append(
            f"walk-forward windows {evidence.walk_forward_windows} is below required "
            f"{thresholds.min_walk_forward_windows}"
        )

    if evidence.max_cost_reconciliation_error_inr is None:
        violations.append("broker cost reconciliation has not been performed")
    elif (
        evidence.max_cost_reconciliation_error_inr
        > thresholds.max_cost_reconciliation_error_inr
    ):
        violations.append(
            f"cost reconciliation error ₹{evidence.max_cost_reconciliation_error_inr} exceeds "
            f"₹{thresholds.max_cost_reconciliation_error_inr}"
        )

    return GateDecision(passed=not violations, violations=tuple(violations))
