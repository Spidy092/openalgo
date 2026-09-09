import json
from datetime import datetime
from decimal import Decimal

import equity_engine.current_market_discovery as current_market_discovery
import equity_engine.instrument_master as instrument_master
from equity_engine.current_market_discovery import (
    APPROVED_CAPITALS,
    measure_current_market,
    quote_request_keys,
)
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.instrument_master import build_nse_equity_master
from equity_engine.suspension_identity import (
    AMBIGUOUS_EXACT,
    NO_SUSPENSION_RECORD,
    SUSPENDED_EXACT,
    SUSPENSION_CONFLICT,
    build_suspension_index,
    resolve_suspension,
)
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
        "exchange_token": 2885,
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
    mis_keys: set[str] | None = None,
    suspended_rows: list[dict[str, object]] | None = None,
):
    keys = quote_request_keys(rows)
    eligible_mis_keys = (
        {str(row["instrument_key"]) for row in rows} if mis_keys is None else mis_keys
    )
    return measure_current_market(
        snapshot_as_of=datetime.fromisoformat("2026-09-09T10:00:00+05:30"),
        bod=_payload("https://example.test/NSE.json.gz", rows, "bod-sha"),
        mis=_payload(
            "https://example.test/NSE_MIS.json.gz",
            [{"instrument_key": key} for key in sorted(eligible_mis_keys)],
            "mis-sha",
        ),
        suspended=_payload(
            "https://example.test/suspended.json.gz",
            suspended_rows or [],
            "suspended-sha",
        ),
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


def _suspended_variant(
    row: dict[str, object],
    *,
    instrument_type: str,
    symbol: str | None = None,
    exchange_token: int = 1,
) -> dict[str, object]:
    suspended = dict(row)
    suspended["instrument_type"] = instrument_type
    suspended["exchange_token"] = exchange_token
    if symbol is not None:
        suspended["trading_symbol"] = symbol
    suspended.pop("security_type", None)
    return suspended


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


def test_suspended_key_only_variant_does_not_false_positive() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    variants = [
        _suspended_variant(bod, instrument_type=kind, exchange_token=index)
        for index, kind in enumerate(("AF", "BE", "BL", "IQ", "RL", "TL"), start=1)
    ]

    artifact = _artifact(
        rows=[bod],
        quotes={key: _quote(key)},
        suspended_rows=variants,
    )
    item = artifact.instruments[0]

    assert item.suspended is False
    assert item.suspension_status == "SUSPENSION_CONFLICT"
    assert item.candidate is True
    assert item.research_candidate is True
    assert item.live_tradability_proven is False
    assert item.suspension_match_count == 0
    assert item.exact_token_match_count == 0
    assert item.suspension_variant_row_count == 6
    assert artifact.suspension_match_count == 0
    assert artifact.suspension_key_variant_count == 1
    assert artifact.conflict_count == 1
    assert artifact.live_orders_called is False


def test_exact_suspended_eq_identity_is_rejected() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    exact = _suspended_variant(bod, instrument_type="EQ", symbol="RELIANCE", exchange_token=2885)

    artifact = _artifact(rows=[bod], quotes={key: _quote(key)}, suspended_rows=[exact])
    item = artifact.instruments[0]

    assert item.suspended is True
    assert item.suspension_status == "SUSPENDED_EXACT"
    assert item.candidate is False
    assert item.research_candidate is False
    assert item.live_tradability_proven is False
    assert item.suspension_match_count == 1
    assert item.exact_token_match_count == 1
    assert item.suspension_ambiguous is False
    assert "suspended_exact_identity" in item.exclusion_reasons
    assert artifact.suspension_match_count == 1
    assert artifact.exact_suspended_count == 1


def test_duplicate_exact_suspended_identity_fails_closed() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    duplicate_a = _suspended_variant(bod, instrument_type="EQ", exchange_token=2885)
    duplicate_b = _suspended_variant(bod, instrument_type="EQ", exchange_token=2885)
    duplicate_b["trading_symbol"] = "RELIANCE-ALT"

    artifact = _artifact(
        rows=[bod],
        quotes={key: _quote(key)},
        suspended_rows=[duplicate_a, duplicate_b],
    )
    item = artifact.instruments[0]

    assert item.suspended is True
    assert item.suspension_status == "AMBIGUOUS_EXACT"
    assert item.suspension_ambiguous is True
    assert item.suspension_match_count == 2
    assert item.exact_token_match_count == 2
    assert item.candidate is False
    assert item.live_tradability_proven is False
    assert "ambiguous_suspended_identity" in item.exclusion_reasons
    assert artifact.suspension_ambiguous_count == 1
    assert artifact.ambiguous_exact_count == 1


def test_reliance_shape_and_mis_intersection_are_resolved_by_identity() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    variants = [
        _suspended_variant(bod, instrument_type=kind, exchange_token=index)
        for index, kind in enumerate(("AF", "BE", "BL", "IQ", "RL", "TL"), start=1)
    ]
    artifact = _artifact(rows=[bod], quotes={key: _quote(key)}, suspended_rows=variants)

    assert artifact.gate_counts["current_mis_eligible"] == 1
    assert artifact.gate_counts["current_suspended"] == 0
    assert artifact.gate_counts["suspension_conflicts"] == 1
    assert artifact.instruments[0].candidate is True
    assert artifact.instruments[0].suspension_status == "SUSPENSION_CONFLICT"
    assert artifact.instruments[0].live_tradability_proven is False
    assert artifact.instruments[0].current_exchange_token == "2885"
    assert artifact.instruments[0].exact_token_match_count == 0


def test_suspension_evidence_is_deterministic() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    variants = [
        _suspended_variant(bod, instrument_type=kind, exchange_token=index)
        for index, kind in enumerate(("AF", "BE", "BL"), start=1)
    ]
    first = _artifact(rows=[bod], quotes={key: _quote(key)}, suspended_rows=variants)
    second = _artifact(rows=[bod], quotes={key: _quote(key)}, suspended_rows=variants)

    assert first.fingerprint == second.fingerprint
    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["suspension_identity_policy"] == (
        "segment+instrument_key+exchange_token;"
        "exchange+instrument_type consistency guards"
    )
    assert first.to_dict()["suspended_source_hash"] == "suspended-sha"


def test_no_suspended_segment_key_record_is_explicit() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    unrelated_segment = _suspended_variant(bod, instrument_type="EQ", exchange_token=2885)
    unrelated_segment["segment"] = "BSE_EQ"

    artifact = _artifact(rows=[bod], quotes={key: _quote(key)}, suspended_rows=[unrelated_segment])
    item = artifact.instruments[0]

    assert item.suspension_status == "NO_SUSPENSION_RECORD"
    assert item.same_key_variant_count == 0
    assert item.live_tradability_proven is True
    assert artifact.no_record_count == 1


def test_exact_token_instrument_type_mismatch_is_a_conflict() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    mismatched = _suspended_variant(bod, instrument_type="AF", exchange_token=2885)

    artifact = _artifact(rows=[bod], quotes={key: _quote(key)}, suspended_rows=[mismatched])
    item = artifact.instruments[0]

    assert item.suspension_status == "SUSPENSION_CONFLICT"
    assert item.exact_token_match_count == 1
    assert item.suspension_match_count == 0
    assert item.live_tradability_proven is False


def test_exchange_token_is_scoped_by_segment_and_instrument_key() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    token_only = {"exchange_token": 2885, "instrument_type": "EQ"}

    artifact = _artifact(rows=[bod], quotes={key: _quote(key)}, suspended_rows=[token_only])

    assert artifact.instruments[0].suspension_status == "NO_SUSPENSION_RECORD"


def test_suspension_index_is_sorted_once_and_input_order_independent(monkeypatch) -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    variants = [
        _suspended_variant(bod, instrument_type=kind, exchange_token=token)
        for kind, token in (("TL", 757288), ("AF", 11139), ("BE", 4615))
    ]
    sort_calls = 0

    import equity_engine.suspension_identity as suspension_identity

    original = suspension_identity.row_sort_key

    def counting_sort_key(row):
        nonlocal sort_calls
        sort_calls += 1
        return original(row)

    monkeypatch.setattr(suspension_identity, "row_sort_key", counting_sort_key)
    first = build_suspension_index(variants)
    second = build_suspension_index(list(reversed(variants)))

    assert sort_calls == len(variants) * 2
    assert first.rows_by_segment_key == second.rows_by_segment_key
    assert first.rows_by_segment_key_token == second.rows_by_segment_key_token
    assert resolve_suspension(bod, first).status == SUSPENSION_CONFLICT


def test_index_resolution_does_not_reiterate_source_for_each_instrument() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    variants = [_suspended_variant(bod, instrument_type="AF", exchange_token=11139)]

    class OnePassRows:
        def __init__(self, rows):
            self.rows = rows
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("suspended source was rescanned")
            return iter(self.rows)

    source = OnePassRows(variants)
    index = build_suspension_index(source)

    class CountingIndex:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.scope_lookups = 0
            self.token_lookups = 0

        def rows_for(self, segment, instrument_key):
            self.scope_lookups += 1
            return self.wrapped.rows_for(segment, instrument_key)

        def exact_token_rows_for(self, segment, instrument_key, exchange_token):
            self.token_lookups += 1
            return self.wrapped.exact_token_rows_for(segment, instrument_key, exchange_token)

    counting_index = CountingIndex(index)
    for _ in range(100):
        assert resolve_suspension(bod, counting_index).status == SUSPENSION_CONFLICT

    assert source.iterations == 1
    assert counting_index.scope_lookups == 100
    assert counting_index.token_lookups == 100


def test_realistic_scale_uses_constant_index_lookups() -> None:
    suspended_rows = [
        _suspended_variant(
            _row(f"NSE_EQ|INE{i:09d}", symbol=f"S{i}"),
            instrument_type="AF",
            exchange_token=100000 + i,
        )
        for i in range(34429)
    ]
    index = build_suspension_index(suspended_rows)

    class CountingIndex:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.scope_lookups = 0
            self.token_lookups = 0

        def rows_for(self, segment, instrument_key):
            self.scope_lookups += 1
            return self.wrapped.rows_for(segment, instrument_key)

        def exact_token_rows_for(self, segment, instrument_key, exchange_token):
            self.token_lookups += 1
            return self.wrapped.exact_token_rows_for(segment, instrument_key, exchange_token)

    counting_index = CountingIndex(index)
    resolved = 0
    for i in range(77068):
        row = _row(f"NSE_EQ|INE{i:09d}", symbol=f"B{i}")
        resolve_suspension(row, counting_index)
        resolved += 1

    assert resolved == 77068
    assert counting_index.scope_lookups == 77068
    assert counting_index.token_lookups == 77068
    assert len(index.rows_by_segment_key) == 34429
    assert index.rows_for("NSE_EQ", "NSE_EQ|INE000000000")[0] is suspended_rows[0]


def test_current_market_and_instrument_master_each_build_one_shared_index(monkeypatch) -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    suspended_rows = [_suspended_variant(bod, instrument_type="AF", exchange_token=11139)]

    current_calls = 0
    current_builder = current_market_discovery.build_suspension_index

    def count_current(rows):
        nonlocal current_calls
        current_calls += 1
        return current_builder(rows)

    monkeypatch.setattr(current_market_discovery, "build_suspension_index", count_current)
    _artifact(rows=[bod], quotes={key: _quote(key)}, suspended_rows=suspended_rows)
    assert current_calls == 1

    master_calls = 0
    master_builder = instrument_master.build_suspension_index

    def count_master(rows):
        nonlocal master_calls
        master_calls += 1
        return master_builder(rows)

    monkeypatch.setattr(instrument_master, "build_suspension_index", count_master)
    snapshot = build_nse_equity_master(
        as_of_date=datetime(2026, 9, 9).date(),
        bod_rows=[bod],
        mis_rows=[{"instrument_key": key}],
        suspended_rows=suspended_rows,
        tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
    )
    assert master_calls == 1
    assert snapshot.instruments[0].suspension_status == SUSPENSION_CONFLICT


def test_index_preserves_all_approved_statuses_and_master_matches_resolver() -> None:
    key = "NSE_EQ|INE002A01018"
    bod = _row(key, symbol="RELIANCE")
    exact = _suspended_variant(bod, instrument_type="EQ", exchange_token=2885)
    duplicate = _suspended_variant(bod, instrument_type="EQ", exchange_token=2885)
    conflict = _suspended_variant(bod, instrument_type="AF", exchange_token=11139)
    unrelated = _suspended_variant(bod, instrument_type="EQ", exchange_token=2885)
    unrelated["segment"] = "BSE_EQ"

    assert resolve_suspension(bod, build_suspension_index([exact])).status == SUSPENDED_EXACT
    assert (
        resolve_suspension(bod, build_suspension_index([exact, duplicate])).status
        == AMBIGUOUS_EXACT
    )
    assert resolve_suspension(bod, build_suspension_index([conflict])).status == SUSPENSION_CONFLICT
    assert (
        resolve_suspension(bod, build_suspension_index([unrelated])).status
        == NO_SUSPENSION_RECORD
    )

    snapshot = build_nse_equity_master(
        as_of_date=datetime(2026, 9, 9).date(),
        bod_rows=[bod],
        mis_rows=[{"instrument_key": key}],
        suspended_rows=[conflict],
        tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
    )
    assert snapshot.instruments[0].suspension_status == SUSPENSION_CONFLICT


def test_no_mis_membership_is_not_an_intraday_candidate() -> None:
    key = "NSE_EQ|INE002A01018"
    artifact = _artifact(
        rows=[_row(key, symbol="RELIANCE")],
        quotes={key: _quote(key)},
        mis_keys=set(),
    )

    assert artifact.instruments[0].mis_eligible is False
    assert artifact.instruments[0].candidate is False
