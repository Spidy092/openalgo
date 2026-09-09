import importlib.util
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.models import CostQuote, CostSource, Exchange, OrderSpec, Product, Side
from equity_engine.observed_costs import ObservedUpstoxNSEIntradayCostProvider
from equity_engine.upstox_cost_reconciliation import (
    COST_MODEL_BROKER_OBSERVED,
    COST_MODEL_DOCUMENTED,
    EXIT_BROKER_API_ERROR,
    EXIT_CONFIGURATION_ERROR,
    EXIT_PASS,
    ReconciliationReport,
    build_orders,
    cost_provider_for_model,
    exit_code_for,
    reconcile_orders,
    report_as_dict,
    report_as_json,
)
from equity_engine.upstox_costs import UpstoxBrokerCostProvider


TOKEN = "secret-token-that-must-not-appear"


def _broker_quote(order: OrderSpec, total: Decimal) -> CostQuote:
    local = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)).quote(order)
    return CostQuote(
        order=order,
        charges=local.charges,
        source=CostSource.BROKER_QUOTE,
        retrieved_at=datetime.now(timezone.utc),
        source_refs=("test-broker",),
        broker_reported_total=total,
    )


class _FixedBroker:
    def __init__(self, delta: Decimal = Decimal("0")) -> None:
        self.delta = delta

    def quote(self, order: OrderSpec) -> CostQuote:
        local = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)).quote(order)
        return _broker_quote(order, local.total + self.delta)


class _ErrorBroker:
    def quote(self, order: OrderSpec) -> CostQuote:
        raise httpx.ReadTimeout("simulated transient Upstox failure")


class _MatchingBroker:
    def __init__(self, provider) -> None:
        self.provider = provider

    def quote(self, order: OrderSpec) -> CostQuote:
        estimate = self.provider.quote(order)
        return CostQuote(
            order=order,
            charges=estimate.charges,
            source=CostSource.BROKER_QUOTE,
            retrieved_at=datetime.now(timezone.utc),
            source_refs=("test-broker",),
            broker_reported_total=estimate.total,
        )


def _run(*, delta: Decimal = Decimal("0"), tolerance: Decimal = Decimal("0.01")):
    return reconcile_orders(
        access_token=TOKEN,
        instrument_token="NSE_EQ|INE001A01036",
        symbol="TEST",
        price=Decimal("100"),
        capital=Decimal("1000"),
        pricing_date=date(2026, 9, 7),
        tolerance=tolerance,
        target_notionals=(Decimal("250"),),
        broker_provider=_FixedBroker(delta),
    )


def test_raw_notional_that_exceeds_cash_after_entry_charges_is_reduced() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))
    orders = build_orders(
        instrument_token="NSE_EQ|TEST",
        price=Decimal("100"),
        capital=Decimal("1000"),
        target_notionals=(Decimal("1000"),),
        cost_provider=provider,
    )
    buy = next(order for order in orders if order.side is Side.BUY)
    quote = provider.quote(buy)
    assert buy.quantity == 9
    assert buy.notional == Decimal("900")
    assert buy.notional + quote.total <= Decimal("1000")


def test_generated_buy_cash_requirement_never_exceeds_capital() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))
    capital = Decimal("1000")
    orders = build_orders(
        instrument_token="NSE_EQ|TEST",
        price=Decimal("100"),
        capital=capital,
        target_notionals=(
            Decimal("100"),
            Decimal("250"),
            Decimal("500"),
            Decimal("750"),
            Decimal("950"),
        ),
        cost_provider=provider,
    )
    buys = [order for order in orders if order.side is Side.BUY]
    assert buys
    assert all(order.notional + provider.quote(order).total <= capital for order in buys)


def test_generated_sell_uses_corresponding_buy_quantity() -> None:
    provider = CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))
    orders = build_orders(
        instrument_token="NSE_EQ|TEST",
        price=Decimal("100"),
        capital=Decimal("1000"),
        target_notionals=(Decimal("100"), Decimal("250"), Decimal("500")),
        cost_provider=provider,
    )
    buy_quantities = [order.quantity for order in orders if order.side is Side.BUY]
    sell_quantities = [order.quantity for order in orders if order.side is Side.SELL]
    assert buy_quantities == sell_quantities
    assert len(buy_quantities) == len(set(buy_quantities))


def test_missing_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="UPSTOX_ACCESS_TOKEN"):
        reconcile_orders(
            access_token="",
            instrument_token="NSE_EQ|TEST",
            symbol="TEST",
            price=Decimal("100"),
            capital=Decimal("1000"),
            pricing_date=date(2026, 9, 7),
            tolerance=Decimal("0.05"),
            target_notionals=(Decimal("100"),),
        )


def test_buy_comparison_passes() -> None:
    report = _run()
    buy = next(item for item in report.items if item.side == Side.BUY.value)
    assert buy.status == "PASS"
    assert buy.absolute_difference == Decimal("0")


def test_sell_comparison_passes() -> None:
    report = _run()
    sell = next(item for item in report.items if item.side == Side.SELL.value)
    assert sell.status == "PASS"


def test_documented_model_is_the_explicit_safe_default() -> None:
    report = _run()

    assert report.cost_model == COST_MODEL_DOCUMENTED
    assert report.model_provenance["brokerage_rate"] == "0.001"


def test_runner_can_explicitly_select_broker_observed_model() -> None:
    provider = cost_provider_for_model(
        cost_model=COST_MODEL_BROKER_OBSERVED,
        pricing_date=date(2026, 9, 9),
    )
    assert isinstance(provider, ObservedUpstoxNSEIntradayCostProvider)
    report = reconcile_orders(
        access_token=TOKEN,
        instrument_token="NSE_EQ|INE001A01036",
        symbol="TEST",
        price=Decimal("100"),
        capital=Decimal("1000"),
        pricing_date=date(2026, 9, 9),
        tolerance=Decimal("0.01"),
        target_notionals=(Decimal("250"),),
        local_provider=provider,
        broker_provider=_MatchingBroker(provider),
        cost_model=COST_MODEL_BROKER_OBSERVED,
    )

    assert report.overall == "PASS"
    assert report.cost_model == COST_MODEL_BROKER_OBSERVED
    assert report.model_provenance["brokerage_rate"] == "0.0006"


def test_tolerance_boundary_is_inclusive() -> None:
    report = _run(delta=Decimal("0.05"), tolerance=Decimal("0.05"))
    assert report.overall == "PASS"
    assert exit_code_for(report) == EXIT_PASS


def test_comparison_failure_returns_reconciliation_failure() -> None:
    report = _run(delta=Decimal("0.051"), tolerance=Decimal("0.05"))
    assert report.overall == "FAIL"
    assert exit_code_for(report) == 1
    assert report.failed_count == 2


def test_broker_api_error_is_not_a_false_pass() -> None:
    report = reconcile_orders(
        access_token=TOKEN,
        instrument_token="NSE_EQ|INE001A01036",
        symbol="TEST",
        price=Decimal("100"),
        capital=Decimal("1000"),
        pricing_date=date(2026, 9, 7),
        tolerance=Decimal("0.05"),
        target_notionals=(Decimal("250"),),
        broker_provider=_ErrorBroker(),
    )
    assert report.overall == "ERROR"
    assert exit_code_for(report) == EXIT_BROKER_API_ERROR
    assert all(item.status == "ERROR" for item in report.items)


def test_malformed_broker_response_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "success", "data": {"charges": {}}},
            request=request,
        )

    order = OrderSpec(
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        side=Side.BUY,
        product=Product.INTRADAY,
        quantity=1,
        price=Decimal("100"),
    )
    with pytest.raises(ValueError, match="missing total"):
        UpstoxBrokerCostProvider(
            access_token=TOKEN,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ).quote(order)


def test_broker_reported_total_is_authoritative() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/charges/brokerage"
        assert request.url.params["instrument_token"] == "NSE_EQ|TEST"
        assert request.url.params["quantity"] == "1"
        assert request.url.params["product"] == "I"
        assert request.url.params["transaction_type"] == "BUY"
        assert request.url.params["price"] == "100"
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "charges": {
                        "brokerage": "1.00",
                        "taxes": {"gst": "0.18", "stt": "0", "stamp_duty": "0.03"},
                        "other_charges": {"transaction": "0.03"},
                        "total": "1.25",
                    }
                },
            },
            request=request,
        )

    order = OrderSpec(
        instrument_token="NSE_EQ|TEST",
        exchange=Exchange.NSE,
        side=Side.BUY,
        product=Product.INTRADAY,
        quantity=1,
        price=Decimal("100"),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        quote = UpstoxBrokerCostProvider(access_token=TOKEN, client=client).quote(order)
    assert quote.broker_reported_total == Decimal("1.25")
    assert quote.total == Decimal("1.25")


def test_decimal_safe_serialization_contains_no_token() -> None:
    report = _run()
    payload = report_as_dict(report)
    encoded = report_as_json(report)
    assert TOKEN not in encoded
    assert isinstance(payload["capital"], str)
    assert isinstance(payload["items"][0]["local_modeled_total"], str)
    assert json.loads(encoded)["tolerance"] == "0.01"


def test_evidence_artifact_contains_audit_fields(tmp_path) -> None:
    from equity_engine.upstox_cost_reconciliation import write_evidence

    report = _run()
    evidence_path = tmp_path / "reconciliation.json"
    write_evidence(evidence_path, report)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "upstox-cost-reconciliation/v1"
    assert evidence["generated_at_utc"].endswith("+00:00")
    assert evidence["instrument_token"] == "NSE_EQ|INE001A01036"
    assert evidence["summary"]["orders_checked"] == 2
    assert evidence["live_orders_called"] is False


def test_configuration_exit_status_and_json_are_safe(capsys, monkeypatch) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "upstox_cost_reconciliation.py"
    spec = importlib.util.spec_from_file_location("upstox_cost_reconciliation_cli", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    status = cli.main(
        [
            "--instrument-token",
            "NSE_EQ|TEST",
            "--price",
            "100",
            "--tolerance",
            "0.05",
            "--json",
        ]
    )
    assert status == EXIT_CONFIGURATION_ERROR
    output = capsys.readouterr()
    assert "UPSTOX_ACCESS_TOKEN is not set" in output.out
    assert TOKEN not in output.out


def test_report_type_is_explicit() -> None:
    report = _run()
    assert isinstance(report, ReconciliationReport)
