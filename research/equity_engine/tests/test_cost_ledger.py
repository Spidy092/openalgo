"""Tests for the effective-dated cost evidence ledger.

UNKNOWN != ZERO is the load-bearing invariant: every unknown is an explicit
``rate=None`` record with an ``unknowns`` explanation, never a silent zero.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest

from equity_engine.cost_ledger import (
    ACCOUNT_SNAPSHOT_DATE,
    BROKERAGE_SNAPSHOT_RATE,
    HISTORICAL_ACTUAL_LABEL,
    INCOMPLETE_LABEL,
    SUPPORTED_RESEARCH_START,
    EffectiveDatedCostLedger,
    EvidenceClass,
    InsufficientHistoricalCostEvidence,
    LedgerComponent,
    LedgerProduct,
    LedgerSide,
    UnknownCostEvidence,
    UnsupportedResearchDate,
)


_CRORE = Decimal("10000000")


def _ledger() -> EffectiveDatedCostLedger:
    return EffectiveDatedCostLedger()


def test_supported_boundary_starts_2024_07_01() -> None:
    assert SUPPORTED_RESEARCH_START == date(2024, 7, 1)


def test_before_supported_boundary_does_not_infer() -> None:
    ledger = _ledger()
    with pytest.raises(UnsupportedResearchDate):
        ledger.mii_transaction_record(date(2024, 6, 30))
    with pytest.raises(UnsupportedResearchDate):
        ledger.resolve(
            LedgerComponent.STT,
            date(2024, 6, 30),
            LedgerProduct.INTRADAY,
            LedgerSide.SELL,
        )


def test_2024_09_30_does_not_use_2024_10_01_rate() -> None:
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence) as first:
        ledger.mii_transaction_record(date(2024, 9, 30))
    with pytest.raises(UnknownCostEvidence) as second:
        ledger.mii_ipft_record(date(2024, 9, 30))
    assert "2024-10-01" in " ".join(first.value.unknowns) or first.value.unknowns
    assert second.value.unknowns
    with pytest.raises(UnknownCostEvidence):
        ledger.total_mii_rate(date(2024, 9, 30))


def test_2024_10_01_resolves_297_plus_10() -> None:
    ledger = _ledger()
    transaction = ledger.mii_transaction_record(date(2024, 10, 1))
    ipft = ledger.mii_ipft_record(date(2024, 10, 1))

    assert transaction.rate == Decimal("297") / _CRORE
    assert ipft.rate == Decimal("10") / _CRORE
    assert transaction.evidence_class is EvidenceClass.MII_SCHEDULE
    assert ipft.evidence_class is EvidenceClass.MII_SCHEDULE
    assert transaction.effective_from == date(2024, 10, 1)
    assert transaction.effective_to == date(2026, 2, 28)


def test_2026_02_28_uses_old_split() -> None:
    ledger = _ledger()
    transaction = ledger.mii_transaction_record(date(2026, 2, 28))
    ipft = ledger.mii_ipft_record(date(2026, 2, 28))

    assert transaction.rate == Decimal("297") / _CRORE
    assert ipft.rate == Decimal("10") / _CRORE


def test_2026_03_01_uses_306_99_plus_0_01() -> None:
    ledger = _ledger()
    transaction = ledger.mii_transaction_record(date(2026, 3, 1))
    ipft = ledger.mii_ipft_record(date(2026, 3, 1))

    assert transaction.rate == Decimal("306.99") / _CRORE
    assert ipft.rate == Decimal("0.01") / _CRORE
    assert transaction.evidence_class is EvidenceClass.MII_SCHEDULE
    assert transaction.source_effective_date == date(2026, 3, 1)


def test_total_mii_outflow_is_307_per_crore_in_both_periods() -> None:
    ledger = _ledger()
    expected = Decimal("307") / _CRORE

    assert ledger.total_mii_rate(date(2024, 10, 1)) == expected
    assert ledger.total_mii_rate(date(2026, 2, 28)) == expected
    assert ledger.total_mii_rate(date(2026, 3, 1)) == expected
    assert ledger.total_mii_rate(date(2026, 9, 9)) == expected


def test_snapshot_0_06_percent_cannot_resolve_2026_09_08() -> None:
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence) as exc:
        ledger.brokerage_record(date(2026, 9, 8), LedgerProduct.INTRADAY)

    assert exc.value.unknowns
    assert any("0.06%" in item or "0.06" in item for item in exc.value.unknowns)

    snapshot = ledger.brokerage_record(ACCOUNT_SNAPSHOT_DATE, LedgerProduct.INTRADAY)
    assert snapshot.rate == BROKERAGE_SNAPSHOT_RATE == Decimal("0.0006")
    assert snapshot.evidence_class is EvidenceClass.ACCOUNT_SNAPSHOT
    assert snapshot.observed_at == date(2026, 9, 9)
    assert snapshot.historical_actual is False


def test_snapshot_is_not_projected_backward_for_history() -> None:
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence):
        ledger.brokerage_record(date(2024, 10, 1), LedgerProduct.INTRADAY)
    with pytest.raises(UnknownCostEvidence):
        ledger.brokerage_record(date(2026, 3, 1), LedgerProduct.INTRADAY)


def test_unknown_clearing_is_not_zero() -> None:
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence) as exc:
        ledger.resolve(
            LedgerComponent.CLEARING,
            date(2025, 1, 15),
            LedgerProduct.INTRADAY,
            LedgerSide.BOTH,
        )

    assert exc.value.unknowns
    # The ledger must not carry a zero clearing rate anywhere.
    for record in ledger.records:
        if record.component is LedgerComponent.CLEARING:
            assert record.rate is None
            assert record.evidence_class is EvidenceClass.UNKNOWN
            assert record.unknowns


def test_unknown_delivery_dp_is_not_zero() -> None:
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence) as exc:
        ledger.resolve(
            LedgerComponent.DP_DEMAT,
            date(2025, 1, 15),
            LedgerProduct.DELIVERY,
            LedgerSide.BOTH,
        )

    assert exc.value.unknowns
    for record in ledger.records:
        if record.component is LedgerComponent.DP_DEMAT:
            assert record.rate is None
            assert record.evidence_class is EvidenceClass.UNKNOWN
            assert record.unknowns


def test_intraday_stt_is_sell_side_only() -> None:
    ledger = _ledger()
    buy = ledger.resolve(
        LedgerComponent.STT, date(2025, 6, 1), LedgerProduct.INTRADAY, LedgerSide.BUY
    )
    sell = ledger.resolve(
        LedgerComponent.STT, date(2025, 6, 1), LedgerProduct.INTRADAY, LedgerSide.SELL
    )

    assert buy.rate == Decimal("0")
    assert sell.rate == Decimal("0.00025")
    assert buy.side is LedgerSide.BUY
    assert sell.side is LedgerSide.SELL


def test_delivery_stt_is_both_sides() -> None:
    ledger = _ledger()
    buy = ledger.resolve(
        LedgerComponent.STT, date(2025, 6, 1), LedgerProduct.DELIVERY, LedgerSide.BUY
    )
    sell = ledger.resolve(
        LedgerComponent.STT, date(2025, 6, 1), LedgerProduct.DELIVERY, LedgerSide.SELL
    )

    assert buy.rate == Decimal("0.001")
    assert sell.rate == Decimal("0.001")
    assert buy.side is LedgerSide.BOTH
    assert sell.side is LedgerSide.BOTH


def test_stamp_duty_side_and_product_differences() -> None:
    ledger = _ledger()
    intraday_buy = ledger.resolve(
        LedgerComponent.STAMP_DUTY,
        date(2025, 6, 1),
        LedgerProduct.INTRADAY,
        LedgerSide.BUY,
    )
    delivery_buy = ledger.resolve(
        LedgerComponent.STAMP_DUTY,
        date(2025, 6, 1),
        LedgerProduct.DELIVERY,
        LedgerSide.BUY,
    )
    intraday_sell = ledger.resolve(
        LedgerComponent.STAMP_DUTY,
        date(2025, 6, 1),
        LedgerProduct.INTRADAY,
        LedgerSide.SELL,
    )
    delivery_sell = ledger.resolve(
        LedgerComponent.STAMP_DUTY,
        date(2025, 6, 1),
        LedgerProduct.DELIVERY,
        LedgerSide.SELL,
    )

    assert intraday_buy.rate == Decimal("0.00003")
    assert delivery_buy.rate == Decimal("0.00015")
    assert intraday_buy.rate != delivery_buy.rate
    assert intraday_sell.rate == Decimal("0")
    assert delivery_sell.rate == Decimal("0")


def test_statutory_sebi_and_gst_are_known() -> None:
    ledger = _ledger()
    sebi = ledger.resolve(
        LedgerComponent.SEBI_TURNOVER,
        date(2025, 6, 1),
        LedgerProduct.INTRADAY,
        LedgerSide.BOTH,
    )
    gst = ledger.resolve(
        LedgerComponent.GST, date(2025, 6, 1), LedgerProduct.DELIVERY, LedgerSide.BOTH
    )

    assert sebi.rate == Decimal("10") / _CRORE
    assert sebi.evidence_class is EvidenceClass.STATUTORY_SCHEDULE
    assert gst.rate == Decimal("0.18")
    assert gst.evidence_class is EvidenceClass.STATUTORY_SCHEDULE


def test_evidence_classes_cover_required_taxonomy() -> None:
    values = {item.value for item in EvidenceClass}
    assert values == {
        "statutory_schedule",
        "mii_schedule",
        "account_snapshot",
        "broker_public_scenario",
        "scenario",
        "unknown",
    }


def test_schema_preserves_required_fields() -> None:
    ledger = _ledger()
    record = ledger.mii_transaction_record(date(2024, 10, 1))
    payload = record.to_dict()

    for field in (
        "effective_from",
        "effective_to",
        "component",
        "product",
        "side",
        "basis",
        "rate",
        "formula",
        "rounding",
        "minimum",
        "cap",
        "gst_taxable",
        "evidence_class",
        "historical_actual",
        "confidence",
        "source_refs",
        "source_publication_date",
        "source_effective_date",
        "observed_at",
        "unknowns",
    ):
        assert field in payload


def test_deterministic_serialization() -> None:
    first = _ledger().to_json()
    second = _ledger().to_json()

    assert first == second
    assert json.loads(first) == json.loads(second)
    assert _ledger().fingerprint() == _ledger().fingerprint()
    # Canonical form uses sorted keys and compact separators.
    assert first.endswith("\n")
    assert _ledger().to_dict() == json.loads(json.dumps(_ledger().to_dict()))


def test_unsupported_full_historical_actual_fails_closed() -> None:
    ledger = _ledger()
    for query_date in (
        date(2024, 9, 30),
        date(2024, 10, 1),
        date(2026, 2, 28),
        date(2026, 3, 1),
        date(2026, 9, 9),
    ):
        for product in (LedgerProduct.INTRADAY, LedgerProduct.DELIVERY):
            with pytest.raises(InsufficientHistoricalCostEvidence) as exc:
                ledger.quote_historical_actual(on_date=query_date, product=product)
            assert exc.value.unknowns


def test_provider_never_labels_historical_actual_costs() -> None:
    ledger = _ledger()
    for query_date in (
        date(2024, 7, 1),
        date(2024, 9, 30),
        date(2024, 10, 1),
        date(2026, 2, 28),
        date(2026, 3, 1),
        date(2026, 9, 9),
    ):
        for product in (LedgerProduct.INTRADAY, LedgerProduct.DELIVERY):
            label = ledger.cost_label_for(query_date, product)
            assessment = ledger.describe(query_date, product)
            assert label != HISTORICAL_ACTUAL_LABEL
            assert assessment.historical_actual is False
            assert assessment.classification == INCOMPLETE_LABEL
            assert assessment.unknowns


def test_public_scenario_does_not_leak_into_default_resolution() -> None:
    ledger = _ledger()
    # The documented 0.1% public terms exist only as an explicit scenario.
    scenario = ledger.resolve(
        LedgerComponent.BROKERAGE,
        date(2026, 9, 9),
        LedgerProduct.INTRADAY,
        LedgerSide.BOTH,
        evidence_classes=(EvidenceClass.BROKER_PUBLIC_SCENARIO,),
    )
    assert scenario.rate == Decimal("0.001")
    assert scenario.historical_actual is False

    # Default brokerage resolution on the snapshot date is the account snapshot.
    default = ledger.resolve(
        LedgerComponent.BROKERAGE,
        date(2026, 9, 9),
        LedgerProduct.INTRADAY,
        LedgerSide.BOTH,
    )
    assert default.evidence_class is EvidenceClass.ACCOUNT_SNAPSHOT
