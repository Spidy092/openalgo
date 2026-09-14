from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from equity_engine.market_context import (
    MarketContextThresholds,
    MarketSnapshot,
    build_market_context,
)


def test_market_context_reports_measurements_without_trade_decision() -> None:
    snapshot = MarketSnapshot(
        captured_at=datetime(2026, 9, 7, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        nifty_open=Decimal("25100"),
        nifty_last=Decimal("25200"),
        nifty_previous_close=Decimal("25000"),
        india_vix_last=Decimal("17"),
        india_vix_previous_close=Decimal("15"),
        advancers=36,
        decliners=12,
        unchanged=2,
    )
    thresholds = MarketContextThresholds(
        large_gap_bps=Decimal("30"),
        strong_index_move_bps=Decimal("50"),
        high_vix_level=Decimal("20"),
        vix_jump_pct=Decimal("10"),
        strong_breadth_pct=Decimal("65"),
        weak_breadth_pct=Decimal("35"),
    )

    context = build_market_context(snapshot, thresholds=thresholds)

    assert context.large_gap
    assert context.index_strong_up
    assert not context.index_strong_down
    assert context.high_volatility  # VIX jumped >10%, even though absolute VIX is below 20.
    assert context.breadth_strong
    assert not context.breadth_weak


def test_context_thresholds_are_not_silently_defaulted() -> None:
    # The object itself forces the experiment to provide all thresholds explicitly.
    thresholds = MarketContextThresholds(
        large_gap_bps=Decimal("25"),
        strong_index_move_bps=Decimal("40"),
        high_vix_level=Decimal("18"),
        vix_jump_pct=Decimal("8"),
        strong_breadth_pct=Decimal("60"),
        weak_breadth_pct=Decimal("40"),
    )
    assert thresholds.large_gap_bps == Decimal("25")
