from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from equity_engine.autonomous_orchestrator import TradeCandidate
from equity_engine.openalgo_analyzer_executor import OpenAlgoAnalyzerExecutor


def candidate(**overrides):
    values = dict(
        candidate_id="candidate-1",
        symbol="RELIANCE",
        exchange="NSE",
        strategy_id="momentum",
        strategy_version="v1",
        side="BUY",
        quantity=2,
        product="MIS",
        price_type="LIMIT",
        entry_price=Decimal("100.50"),
        stop_price=Decimal("98"),
        target_price=Decimal("105"),
        expected_edge_bps=Decimal("25"),
        confidence=Decimal("0.80"),
        valid_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        dataset_fingerprint="dataset-sha256",
        research_fingerprint="research-sha256",
    )
    values.update(overrides)
    return TradeCandidate(**values)


def test_executor_submits_directly_to_sandbox():
    calls = []

    def sandbox(order_data, api_key, original_data):
        calls.append((order_data, api_key, original_data))
        return True, {"status": "success", "orderid": "paper-123"}, 200

    executor = OpenAlgoAnalyzerExecutor(
        api_key="secret",
        analyzer_mode_reader=lambda: True,
        sandbox_place_order=sandbox,
    )
    order_id = executor.submit(candidate())

    assert order_id == "paper-123"
    assert len(calls) == 1
    order, api_key, original = calls[0]
    assert api_key == "secret"
    assert order == original
    assert order["strategy"] == "Autonomous:momentum@v1"
    assert order["pricetype"] == "LIMIT"
    assert order["price"] == 100.5


def test_executor_stops_when_operator_turns_analyzer_off():
    called = False

    def sandbox(*args):
        nonlocal called
        called = True
        return True, {"orderid": "should-not-run"}, 200

    executor = OpenAlgoAnalyzerExecutor(
        api_key="secret",
        analyzer_mode_reader=lambda: False,
        sandbox_place_order=sandbox,
    )
    with pytest.raises(RuntimeError, match="analyzer mode is not enabled"):
        executor.submit(candidate())
    assert called is False


def test_market_order_never_uses_research_entry_as_limit_price():
    captured = {}

    def sandbox(order_data, api_key, original_data):
        captured.update(order_data)
        return True, {"orderid": "paper-456"}, 200

    executor = OpenAlgoAnalyzerExecutor(
        api_key="secret",
        analyzer_mode_reader=lambda: True,
        sandbox_place_order=sandbox,
    )
    executor.submit(candidate(price_type="MARKET"))
    assert captured["price"] == 0.0


def test_sandbox_failure_is_not_retried_or_hidden():
    calls = 0

    def sandbox(*args):
        nonlocal calls
        calls += 1
        return False, {"status": "error", "message": "rejected"}, 400

    executor = OpenAlgoAnalyzerExecutor(
        api_key="secret",
        analyzer_mode_reader=lambda: True,
        sandbox_place_order=sandbox,
    )
    with pytest.raises(RuntimeError, match="rejected"):
        executor.submit(candidate())
    assert calls == 1
