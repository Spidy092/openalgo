"""Tests for the historical cost evidence matrix and canonical assumptions.

Proves the market-realism contract for 2024-10-01 through 2026-09-08: statutory
schedules are graded as documented evidence, account-specific history stays
unknown, the Sep-9 snapshot is never projected, and unproven components reach
backtests only as named fingerprinted assumptions through the single canonical
scenario implementation.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from equity_engine.cost_ledger import (
    ACCOUNT_SNAPSHOT_DATE,
    EffectiveDatedCostLedger,
    LedgerComponent,
    LedgerProduct,
    LedgerSide,
    UnknownCostEvidence,
)
from equity_engine.historical_cost_evidence import (
    EVIDENCE_WINDOW_END,
    EVIDENCE_WINDOW_START,
    EvidenceGrade,
    build_evidence_matrix,
    canonical_assumptions_for,
    compile_evidence_scenario,
    delivery_cost_assumptions,
    friction_evidence_fingerprint,
    intraday_cost_assumptions,
    named_friction_scenarios,
)
from equity_engine.historical_cost_scenario import ScenarioAssumption, compile_historical_scenario


def _ledger() -> EffectiveDatedCostLedger:
    return EffectiveDatedCostLedger()


def test_evidence_window_defaults() -> None:
    assert EVIDENCE_WINDOW_START == date(2024, 10, 1)
    assert EVIDENCE_WINDOW_END == date(2026, 9, 8)


def test_statutory_schedules_graded_documented() -> None:
    matrix = build_evidence_matrix(_ledger())
    for query_date in (date(2024, 10, 1), date(2025, 6, 1), date(2026, 2, 28), date(2026, 9, 8)):
        for component, product, side in (
            (LedgerComponent.STT, LedgerProduct.INTRADAY, LedgerSide.SELL),
            (LedgerComponent.STT, LedgerProduct.DELIVERY, LedgerSide.BUY),
            (LedgerComponent.STAMP_DUTY, LedgerProduct.INTRADAY, LedgerSide.BUY),
            (LedgerComponent.STAMP_DUTY, LedgerProduct.DELIVERY, LedgerSide.BUY),
            (LedgerComponent.SEBI_TURNOVER, LedgerProduct.INTRADAY, LedgerSide.BOTH),
        ):
            covering = [
                cell
                for cell in matrix.cells_for(component, product, side)
                if cell.date_from <= query_date <= cell.date_to
            ]
            assert covering, (component, product, side, query_date)
            assert all(cell.grade is EvidenceGrade.DOCUMENTED_STATUTORY for cell in covering)
            assert all(cell.rate_known for cell in covering)


def test_mii_transition_appears_as_distinct_cells() -> None:
    matrix = build_evidence_matrix(_ledger())
    before = [
        cell
        for cell in matrix.cells_for(
            LedgerComponent.TRANSACTION, LedgerProduct.INTRADAY, LedgerSide.BOTH
        )
        if cell.date_from <= date(2026, 2, 28) <= cell.date_to
    ]
    after = [
        cell
        for cell in matrix.cells_for(
            LedgerComponent.TRANSACTION, LedgerProduct.INTRADAY, LedgerSide.BOTH
        )
        if cell.date_from <= date(2026, 3, 1) <= cell.date_to
    ]
    assert before and after
    assert all(cell.grade is EvidenceGrade.DOCUMENTED_STATUTORY for cell in before + after)
    assert before[0].record_identities != after[0].record_identities
    assert before[0].date_to == date(2026, 2, 28)
    assert after[0].date_from == date(2026, 3, 1)


def test_brokerage_history_unknown_across_window() -> None:
    matrix = build_evidence_matrix(_ledger())
    for product in (LedgerProduct.INTRADAY, LedgerProduct.DELIVERY):
        for side in (LedgerSide.BUY, LedgerSide.SELL):
            cells = matrix.cells_for(LedgerComponent.BROKERAGE, product, side)
            assert cells
            assert all(cell.grade is EvidenceGrade.UNKNOWN for cell in cells)
            assert all(not cell.rate_known for cell in cells)


def test_snapshot_never_projected_backward_or_forward() -> None:
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence):
        ledger.brokerage_record(date(2026, 9, 8), LedgerProduct.INTRADAY)
    snapshot = ledger.brokerage_record(ACCOUNT_SNAPSHOT_DATE, LedgerProduct.INTRADAY)
    assert snapshot.effective_from == snapshot.effective_to == ACCOUNT_SNAPSHOT_DATE
    with pytest.raises(UnknownCostEvidence):
        ledger.brokerage_record(date(2026, 9, 10), LedgerProduct.INTRADAY)
    matrix = build_evidence_matrix(ledger, start=date(2026, 9, 8), end=date(2026, 9, 10))
    cells = matrix.cells_for(LedgerComponent.BROKERAGE, LedgerProduct.INTRADAY, LedgerSide.BOTH)
    by_date = {cell.date_from: cell.grade for cell in cells}
    assert by_date == {
        date(2026, 9, 8): EvidenceGrade.UNKNOWN,
        date(2026, 9, 9): EvidenceGrade.ACCOUNT_OBSERVED_CURRENT,
        date(2026, 9, 10): EvidenceGrade.UNKNOWN,
    }


def test_gst_unknown_and_public_scenario_isolated() -> None:
    matrix = build_evidence_matrix(_ledger())
    for product in (LedgerProduct.INTRADAY, LedgerProduct.DELIVERY):
        cells = matrix.cells_for(LedgerComponent.GST, product, LedgerSide.BOTH)
        assert cells
        assert all(cell.grade is EvidenceGrade.UNKNOWN for cell in cells)
        assert all(not cell.rate_known for cell in cells)
    assert EvidenceGrade.BROKER_PUBLIC_SCENARIO.value not in matrix.grades_present()


def test_delivery_dp_and_clearing_unknown_never_zero() -> None:
    matrix = build_evidence_matrix(_ledger())
    dp_cells = matrix.cells_for(LedgerComponent.DP_DEMAT, LedgerProduct.DELIVERY, LedgerSide.BOTH)
    assert dp_cells
    assert all(cell.grade is EvidenceGrade.UNKNOWN for cell in dp_cells)
    clearing = matrix.cells_for(LedgerComponent.CLEARING, LedgerProduct.INTRADAY, LedgerSide.BOTH)
    assert clearing
    assert all(cell.grade is EvidenceGrade.UNKNOWN for cell in clearing)
    for record in _ledger().records:
        if record.component in (LedgerComponent.CLEARING, LedgerComponent.DP_DEMAT):
            assert record.rate is None


def test_no_historical_actual_cells_with_current_evidence() -> None:
    matrix = build_evidence_matrix(_ledger())
    assert EvidenceGrade.HISTORICAL_ACTUAL.value not in matrix.grades_present()


def test_unknown_cells_carry_no_rates() -> None:
    matrix = build_evidence_matrix(_ledger())
    assert matrix.unknown_cells()
    assert all(not cell.rate_known for cell in matrix.unknown_cells())


def test_canonical_assumption_sets_are_product_locked() -> None:
    intraday = intraday_cost_assumptions()
    delivery = delivery_cost_assumptions()
    assert {item.component for item in intraday} == {
        LedgerComponent.BROKERAGE,
        LedgerComponent.GST,
        LedgerComponent.CLEARING,
    }
    assert {item.component for item in delivery} == {
        LedgerComponent.BROKERAGE,
        LedgerComponent.GST,
        LedgerComponent.CLEARING,
        LedgerComponent.DP_DEMAT,
    }
    assert all(item.product is LedgerProduct.INTRADAY for item in intraday)
    assert all(item.product is LedgerProduct.DELIVERY for item in delivery)
    assert not {item.assumption_id for item in intraday} & {item.assumption_id for item in delivery}
    assert canonical_assumptions_for(LedgerProduct.INTRADAY) == intraday
    assert canonical_assumptions_for(LedgerProduct.DELIVERY) == delivery
    with pytest.raises(ValueError, match="INTRADAY or DELIVERY"):
        canonical_assumptions_for(LedgerProduct.ALL)


def test_evidence_scenario_builds_and_stays_non_actual() -> None:
    for product in (LedgerProduct.INTRADAY, LedgerProduct.DELIVERY):
        scenario = compile_evidence_scenario(
            scenario_id=f"evidence-{product.value.lower()}",
            ledger=_ledger(),
            scenario_date=date(2025, 6, 1),
            research_start=date(2025, 1, 1),
            research_end=date(2025, 12, 31),
            product=product,
        )
        assert scenario.historical_actual is False
        assert scenario.ledger_fingerprint == _ledger().fingerprint()
        assert scenario.to_dict()["historical_actual"] is False


def test_evidence_scenario_fingerprint_deterministic_and_sensitive() -> None:
    kwargs: dict[str, object] = {
        "scenario_id": "evidence-intraday",
        "ledger": _ledger(),
        "scenario_date": date(2025, 6, 1),
        "research_start": date(2025, 1, 1),
        "research_end": date(2025, 12, 31),
        "product": LedgerProduct.INTRADAY,
    }
    first = compile_evidence_scenario(**kwargs)  # type: ignore[arg-type]
    second = compile_evidence_scenario(**kwargs)  # type: ignore[arg-type]
    assert first.fingerprint() == second.fingerprint()
    assert first.to_json() == second.to_json()
    changed_assumptions = tuple(
        ScenarioAssumption(
            assumption_id=item.assumption_id,
            component=item.component,
            product=item.product,
            basis=item.basis,
            rate=Decimal("0.002") if item.component is LedgerComponent.BROKERAGE else item.rate,
            formula=item.formula,
            source=item.source,
            reason=item.reason,
        )
        for item in intraday_cost_assumptions()
    )
    changed = compile_historical_scenario(
        scenario_id="evidence-intraday",
        ledger=_ledger(),
        scenario_date=date(2025, 6, 1),
        research_start=date(2025, 1, 1),
        research_end=date(2025, 12, 31),
        product=LedgerProduct.INTRADAY,
        assumptions=changed_assumptions,
    )
    assert changed.fingerprint() != first.fingerprint()


def test_friction_scenarios_are_named_non_observed_stress() -> None:
    scenarios = named_friction_scenarios()
    assert len(scenarios) == 3
    assert all(not item.observed for item in scenarios)
    assert len({item.scenario_id for item in scenarios}) == 3
    first = friction_evidence_fingerprint()
    assert first == friction_evidence_fingerprint()
    assert len(first) == 64


def test_matrix_serialization_deterministic() -> None:
    first = build_evidence_matrix(_ledger()).to_json()
    second = build_evidence_matrix(_ledger()).to_json()
    assert first == second
    assert json.loads(first) == json.loads(second)
    assert first.endswith("\n")
    assert (
        build_evidence_matrix(_ledger()).fingerprint()
        == build_evidence_matrix(_ledger()).fingerprint()
    )


def test_module_has_no_live_order_or_network_capability() -> None:
    source = Path(__file__).parents[1] / "src" / "equity_engine" / "historical_cost_evidence.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("import httpx", "import requests", "place_order", "import socket"):
        assert forbidden not in text
