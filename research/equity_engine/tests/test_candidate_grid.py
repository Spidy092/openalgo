from datetime import time
from decimal import Decimal

from equity_engine.candidate_grid import baseline_definition, orb_study_grid


def test_orb_study_grid_uses_published_range_and_volume_values() -> None:
    definitions = orb_study_grid(
        session_open=time(9, 15),
        bar_minutes=5,
        breakout_buffer_bps=Decimal("0"),
    )

    assert len(definitions) == 6
    ids = {definition.candidate_id for definition in definitions}
    assert "orb:5m:vol1.2:buf0bps" in ids
    assert "orb:15m:vol1.5:buf0bps" in ids
    assert "orb:30m:vol1.2:buf0bps" in ids


def test_orb_grid_does_not_invent_incompatible_ranges() -> None:
    definitions = orb_study_grid(
        session_open=time(9, 15),
        bar_minutes=15,
        breakout_buffer_bps=Decimal("0"),
    )
    ids = {definition.candidate_id for definition in definitions}
    assert all(not candidate_id.startswith("orb:5m") for candidate_id in ids)
    assert len(definitions) == 4


def test_baseline_is_explicitly_available() -> None:
    definition = baseline_definition(session_open=time(9, 15))
    assert definition.candidate_id == "baseline:first-bar-hold"
