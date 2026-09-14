from datetime import date
from decimal import Decimal

from equity_engine.corporate_actions import assess_corporate_actions, parse_corporate_action_rows


def test_parse_and_block_structural_action_in_research_window() -> None:
    events = parse_corporate_action_rows(
        [
            {
                "name": "Split",
                "expiry_date": "14 Aug 2025",
                "amount": None,
                "ratio": "1:2",
                "event_details": [],
            },
            {
                "name": "Dividend",
                "expiry_date": "15 Sep 2025",
                "amount": 5.5,
                "ratio": None,
                "event_details": [],
            },
        ]
    )
    assessment = assess_corporate_actions(
        events=events,
        research_start=date(2025, 1, 1),
        research_end=date(2025, 12, 31),
        blocked_event_names=frozenset({"Split", "Bonus", "Rights"}),
    )

    assert assessment.complete is True
    assert assessment.blocking_events == ("Split@2025-08-14 ratio=1:2",)
    assert events[1].amount == Decimal("5.5")


def test_out_of_window_action_does_not_block() -> None:
    events = parse_corporate_action_rows(
        [{"name": "Bonus", "expiry_date": "01 Jan 2024", "amount": None, "ratio": "1:1"}]
    )
    assessment = assess_corporate_actions(
        events=events,
        research_start=date(2025, 1, 1),
        research_end=date(2026, 1, 1),
        blocked_event_names=frozenset({"Split", "Bonus", "Rights"}),
    )
    assert assessment.blocking_events == ()
