from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from equity_engine.autonomous_contract import (
    SCHEMA_VERSION,
    candidate_from_dict,
    candidate_to_dict,
)
from equity_engine.autonomous_orchestrator import TradeCandidate


def candidate():
    return TradeCandidate(
        candidate_id="candidate-1",
        symbol="RELIANCE",
        exchange="NSE",
        strategy_id="momentum",
        strategy_version="v1",
        side="BUY",
        quantity=1,
        product="MIS",
        price_type="MARKET",
        entry_price=Decimal("100"),
        stop_price=Decimal("98"),
        target_price=Decimal("105"),
        expected_edge_bps=Decimal("25"),
        confidence=Decimal("0.8"),
        valid_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        dataset_fingerprint="dataset-sha256",
        research_fingerprint="research-sha256",
    )


def test_candidate_round_trip_is_exact():
    original = candidate()
    payload = candidate_to_dict(original)
    assert payload["schema_version"] == SCHEMA_VERSION
    decoded = candidate_from_dict(payload)
    assert decoded == original
    assert decoded.fingerprint == original.fingerprint


def test_unknown_field_fails_closed():
    payload = candidate_to_dict(candidate())
    payload["leverage"] = 10
    with pytest.raises(ValueError, match="unknown trade-candidate fields"):
        candidate_from_dict(payload)


def test_missing_field_fails_closed():
    payload = candidate_to_dict(candidate())
    del payload["product"]
    with pytest.raises(ValueError, match="missing trade-candidate fields"):
        candidate_from_dict(payload)


def test_wrong_schema_fails_closed():
    payload = candidate_to_dict(candidate())
    payload["schema_version"] = "equity-trade-candidate/v2"
    with pytest.raises(ValueError, match="unsupported trade-candidate schema"):
        candidate_from_dict(payload)
