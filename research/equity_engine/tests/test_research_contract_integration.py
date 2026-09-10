"""Cross-contract checks for the approved research integration boundary."""

from __future__ import annotations

from pathlib import Path

from equity_engine.historical_acquisition_plan import (
    build_example_historical_acquisition_plan,
)
from equity_engine.research_window_compiler import PITMembershipSegment


def test_acquisition_superset_is_not_the_frozen_wfo_population() -> None:
    plan = build_example_historical_acquisition_plan()

    assert plan.mode == "DRY_RUN"
    assert plan.live_orders_called is False
    assert all(isinstance(segment, PITMembershipSegment) for segment in plan.pit_segments)
    assert plan.historical_acquisition_superset != plan.frozen_wfo_population
    assert "NSE_EQ|INE001A01010" in plan.historical_acquisition_superset
    assert "NSE_EQ|INE001A01010" not in plan.frozen_wfo_population
    assert "PITMembershipEvidence" not in plan.to_json()


def test_compact_acquisition_fixture_is_deterministic_and_below_size_limit() -> None:
    fixture = Path(__file__).parents[1] / "examples" / "historical_acquisition_plan_example.json"
    expected = build_example_historical_acquisition_plan().to_json(indent=None)
    actual = fixture.read_text(encoding="utf-8")

    assert actual == expected
    assert len(actual.encode("utf-8")) < 20_000
