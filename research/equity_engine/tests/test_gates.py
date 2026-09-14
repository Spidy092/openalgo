from decimal import Decimal

from equity_engine.gates import (
    DrawdownBasis,
    PromotionThresholds,
    ResearchEvidence,
    evaluate_promotion_gate,
)


def _thresholds() -> PromotionThresholds:
    return PromotionThresholds(
        min_trades=100,
        min_profit_factor=Decimal("1.20"),
        max_drawdown_pct=Decimal("10"),
        min_walk_forward_windows=6,
        max_cost_reconciliation_error_inr=Decimal("0.01"),
    )


def test_incomplete_research_is_rejected_even_when_profit_factor_is_high() -> None:
    evidence = ResearchEvidence(
        trade_count=500,
        profit_factor=Decimal("3.00"),
        max_drawdown_pct=Decimal("2"),
        drawdown_basis=DrawdownBasis.REALIZED_CLOSED_TRADES,
        walk_forward_windows=12,
        max_cost_reconciliation_error_inr=None,
        held_out_test_present=False,
        baseline_comparison_present=False,
        slippage_stress_present=False,
        event_driven_validation_present=False,
        paper_trading_present=False,
        data_provenance_complete=False,
        unpriced_cost_components=("unknown_charge",),
    )

    decision = evaluate_promotion_gate(evidence, _thresholds())

    assert not decision.passed
    assert "broker cost reconciliation has not been performed" in decision.violations
    assert any("unpriced cost components" in item for item in decision.violations)
    assert any("held-out" in item for item in decision.violations)
    assert any("OHLC-low liquidation stress" in item for item in decision.violations)


def test_complete_evidence_passes_only_with_ohlc_low_liquidation_drawdown() -> None:
    evidence = ResearchEvidence(
        trade_count=150,
        profit_factor=Decimal("1.25"),
        max_drawdown_pct=Decimal("8"),
        drawdown_basis=DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS,
        walk_forward_windows=8,
        max_cost_reconciliation_error_inr=Decimal("0.005"),
        held_out_test_present=True,
        baseline_comparison_present=True,
        slippage_stress_present=True,
        event_driven_validation_present=True,
        paper_trading_present=True,
        data_provenance_complete=True,
    )

    decision = evaluate_promotion_gate(evidence, _thresholds())

    assert decision.passed
    assert decision.violations == ()


def test_close_only_drawdown_cannot_pass_promotion_even_if_number_is_small() -> None:
    evidence = ResearchEvidence(
        trade_count=150,
        profit_factor=Decimal("1.25"),
        max_drawdown_pct=Decimal("1"),
        drawdown_basis=DrawdownBasis.CLOSE_LIQUIDATION,
        walk_forward_windows=8,
        max_cost_reconciliation_error_inr=Decimal("0.005"),
        held_out_test_present=True,
        baseline_comparison_present=True,
        slippage_stress_present=True,
        event_driven_validation_present=True,
        paper_trading_present=True,
        data_provenance_complete=True,
    )

    decision = evaluate_promotion_gate(evidence, _thresholds())

    assert not decision.passed
    assert any("OHLC-low liquidation stress" in item for item in decision.violations)


def test_drawdown_failure_blocks_promotion_despite_other_evidence() -> None:
    evidence = ResearchEvidence(
        trade_count=150,
        profit_factor=Decimal("1.50"),
        max_drawdown_pct=Decimal("10.01"),
        drawdown_basis=DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS,
        walk_forward_windows=8,
        max_cost_reconciliation_error_inr=Decimal("0.001"),
        held_out_test_present=True,
        baseline_comparison_present=True,
        slippage_stress_present=True,
        event_driven_validation_present=True,
        paper_trading_present=True,
        data_provenance_complete=True,
    )

    decision = evaluate_promotion_gate(evidence, _thresholds())

    assert not decision.passed
    assert any("drawdown" in item for item in decision.violations)
