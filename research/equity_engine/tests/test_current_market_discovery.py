import json
from datetime import datetime
from decimal import Decimal

from equity_engine.current_market_discovery import (
    APPROVED_CAPITALS,
    measure_current_market,
    quote_request_keys,
)
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.upstox_instruments import InstrumentFilePayload
from equity_engine.upstox_market_context import QuoteBatchResult


def _row(
    key: str,
    *,
    symbol: str,
    price: str = "100",
    lot_size: int = 1,
    cas_eligible: bool = False,
    security_type: str = "NORMAL",
) -> dict[str, object]:
    return {
        "segment": "NSE_EQ",
        "name": f"{symbol} LIMITED",
        "exchange": "NSE",
        "isin": key.split("|", 1)[1],
        "instrument_type": "EQ",
        "instrument_key": key,
        "lot_size": lot_size,
        "freeze_quantity": 100000,
        "tick_size": 5,
        "trading_symbol": symbol,
        "series": "EQ",
        "security_type": security_type,
        "cas_eligible": cas_eligible,
        "test_price": price,
    }


def _payload(url: str, rows: list[dict[str, object]], sha: str) -> InstrumentFilePayload:
    return InstrumentFilePayload(
        url=url,
        rows=tuple(rows),
        sha256=sha,
        etag=None,
        last_modified=None,
    )


def _artifact(
    *,
    rows: list[dict[str, object]],
    quotes: dict[str, dict[str, object]],
    quote_failures: dict[str, str] | None = None,
    max_last_price_rupees: Decimal | None = None,
):
    keys = quote_request_keys(rows)
    return measure_current_market(
        snapshot_as_of=datetime.fromisoformat("2026-09-09T10:00:00+05:30"),
        bod=_payload("https://example.test/NSE.json.gz", rows, "bod-sha"),
        mis=_payload(
            "https://example.test/NSE_MIS.json.gz",
            [{"instrument_key": row["instrument_key"]} for row in rows],
            "mis-sha",
        ),
        suspended=_payload("https://example.test/suspended.json.gz", [], "suspended-sha"),
        quotes=QuoteBatchResult(
            requested_instrument_keys=keys,
            quotes=quotes,
            failures=quote_failures or {},
            request_count=1,
        ),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=datetime(2026, 9, 9).date()),
        approved_capitals=APPROVED_CAPITALS,
        max_last_price_rupees=max_last_price_rupees,
    )


def _quote(key: str, *, price: str = "100", cas_eligible: bool = False) -> dict[str, object]:
    return {
        "instrument_token": key,
        "last_price": price,
        "reference_price": "9999",
        "indicative_equilibrium_price": "8888",
        "timestamp": "2026-09-09T10:00:00+05:30",
        "cas_eligible": cas_eligible,
    }


def test_quote_request_set_is_structural_and_deterministic() -> None:
    good = _row("NSE_EQ|INE000000001", symbol="GOOD")
    malformed = _row("NSE_EQ|INE000000002", symbol="BAD")
    malformed["isin"] = "DUMMYSAN005"
    duplicate = _row("NSE_EQ|INE000000003", symbol="DUP")

    assert quote_request_keys([duplicate, good, duplicate, malformed]) == (good["instrument_key"],)


def test_measurement_uses_last_price_not_cas_reference_fields() -> None:
    key = "NSE_EQ|INE000000001"
    artifact = _artifact(rows=[_row(key, symbol="GOOD")], quotes={key: _quote(key)})
    item = artifact.instruments[0]

    assert item.reference_price == Decimal("100")
    assert item.price_source == "last_price"
    assert item.quote_status == "success"
    assert item.capital_measurements[0].raw_minimum_capital == Decimal("100")


def test_capital_scenarios_use_same_charge_aware_path_for_1000_and_10000() -> None:
    key = "NSE_EQ|INE000000001"
    artifact = _artifact(rows=[_row(key, symbol="GOOD", lot_size=10)], quotes={key: _quote(key)})
    item = artifact.instruments[0]
    measurements = {measurement.capital: measurement for measurement in item.capital_measurements}

    assert set(measurements) == {Decimal("1000.00"), Decimal("10000.00")}
    assert measurements[Decimal("1000.00")].minimum_entry_cash_required > Decimal("1000")
    assert measurements[Decimal("1000.00")].affordable is False
    assert measurements[Decimal("10000.00")].affordable is True
    for measurement in measurements.values():
        assert measurement.cash_required <= measurement.capital


def test_missing_quote_and_invalid_price_are_audited_without_fallback() -> None:
    missing = "NSE_EQ|INE000000001"
    invalid = "NSE_EQ|INE000000002"
    zero = "NSE_EQ|INE000000003"
    negative = "NSE_EQ|INE000000004"
    artifact = _artifact(
        rows=[
            _row(missing, symbol="MISSING"),
            _row(invalid, symbol="INVALID"),
            _row(zero, symbol="ZERO"),
            _row(negative, symbol="NEGATIVE"),
        ],
        quotes={
            invalid: _quote(invalid, price="NaN"),
            zero: _quote(zero, price="0"),
            negative: _quote(negative, price="-1"),
        },
        quote_failures={missing: "missing_quote"},
    )

    by_key = {item.instrument_key: item for item in artifact.instruments}
    assert by_key[missing].quote_status == "failure"
    assert "missing_quote" in by_key[missing].exclusion_reasons
    assert by_key[invalid].reference_price is None
    assert "invalid_last_price" in by_key[invalid].exclusion_reasons
    assert by_key[zero].reference_price is None
    assert by_key[negative].reference_price is None
    assert artifact.quote_failure_count == 4


def test_max_price_filter_is_reported_separately_from_affordability() -> None:
    cheap = "NSE_EQ|INE000000001"
    expensive = "NSE_EQ|INE000000002"
    artifact = _artifact(
        rows=[_row(cheap, symbol="CHEAP"), _row(expensive, symbol="EXPENSIVE")],
        quotes={cheap: _quote(cheap, price="100"), expensive: _quote(expensive, price="200")},
        max_last_price_rupees=Decimal("150"),
    )

    assert artifact.max_price_analysis["relationship"] == "independent_nominal_price_filter"
    assert artifact.max_price_analysis["excluded_by_filter"] == 1
    assert artifact.max_price_analysis["required_for_affordability"] is False


def test_calibration_artifact_is_deterministic_and_does_not_serialize_credentials() -> None:
    key = "NSE_EQ|INE000000001"
    first = _artifact(rows=[_row(key, symbol="GOOD")], quotes={key: _quote(key)})
    second = _artifact(rows=[_row(key, symbol="GOOD")], quotes={key: _quote(key)})

    assert first.fingerprint == second.fingerprint
    serialized = json.dumps(first.to_dict(), sort_keys=True)
    assert "secret-token" not in serialized
    assert "Authorization" not in serialized
    assert first.to_dict() == second.to_dict()
    assert first.live_orders_called is False


def test_exact_bod_duplicate_is_excluded_fail_closed() -> None:
    key = "NSE_EQ|INE000000001"
    first = _row(key, symbol="FIRST")
    second = _row(key, symbol="SECOND")
    artifact = _artifact(rows=[first, second], quotes={})

    assert artifact.quote_request_count == 0
    assert artifact.gate_counts["raw_upstox_instruments"] == 2
    assert artifact.exclusion_reason_counts["duplicate_instrument_key"] == 2
