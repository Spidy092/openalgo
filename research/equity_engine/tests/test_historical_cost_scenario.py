"""Tests for the research-only historical cost scenario contract.

Load-bearing invariants: scenarios bind ledger evidence plus explicit named
assumptions, never synthesize historical actuals, never project the single-day
account snapshot, and fail closed instead of turning UNKNOWN into zero.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from equity_engine.cost_ledger import (
    ACCOUNT_SNAPSHOT_DATE,
    BROKERAGE_SNAPSHOT_RATE,
    HISTORICAL_ACTUAL_LABEL,
    EffectiveDatedCostLedger,
    EvidenceClass,
    LedgerComponent,
    LedgerProduct,
    LedgerSide,
    UnknownCostEvidence,
)
from equity_engine.historical_cost_scenario import (
    SCHEMA_VERSION,
    HistoricalCostScenario,
    MissingScenarioAssumptionError,
    ScenarioAssumption,
    ScenarioError,
    compile_historical_scenario,
)


def _ledger() -> EffectiveDatedCostLedger:
    return EffectiveDatedCostLedger()


def _assumption(
    component: LedgerComponent,
    product: LedgerProduct,
    rate: str,
    assumption_id: str | None = None,
) -> ScenarioAssumption:
    return ScenarioAssumption(
        assumption_id=assumption_id or f"assume-{component.value}-test",
        component=component,
        product=product,
        basis="test-basis",
        rate=Decimal(rate),
        formula=f"test-formula-{rate}",
        source="test-source",
        reason="test-reason",
    )


def _intraday_assumptions(
    product: LedgerProduct = LedgerProduct.INTRADAY,
) -> tuple[ScenarioAssumption, ...]:
    return (
        _assumption(LedgerComponent.BROKERAGE, product, "0.0006"),
        _assumption(LedgerComponent.GST, product, "0.18"),
        _assumption(LedgerComponent.CLEARING, product, "0.000001"),
    )


def _scenario(
    product: LedgerProduct = LedgerProduct.INTRADAY,
    scenario_date: date = date(2025, 6, 1),
    assumptions: tuple[ScenarioAssumption, ...] | None = None,
    **overrides: object,
) -> HistoricalCostScenario:
    params: dict[str, object] = {
        "scenario_id": "test-scenario",
        "ledger": _ledger(),
        "scenario_date": scenario_date,
        "research_start": date(2025, 1, 1),
        "research_end": date(2025, 12, 31),
        "product": product,
        "assumptions": assumptions if assumptions is not None else _intraday_assumptions(product),
    }
    params.update(overrides)
    return compile_historical_scenario(**params)  # type: ignore[arg-type]


def test_sep_8_brokerage_actual_is_unknown() -> None:
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence) as exc:
        ledger.brokerage_record(date(2026, 9, 8), LedgerProduct.INTRADAY)
    assert exc.value.unknowns


def test_sep_9_snapshot_exists_only_for_sep_9() -> None:
    ledger = _ledger()
    snapshot = ledger.brokerage_record(ACCOUNT_SNAPSHOT_DATE, LedgerProduct.INTRADAY)
    assert snapshot.rate == BROKERAGE_SNAPSHOT_RATE
    assert snapshot.evidence_class is EvidenceClass.ACCOUNT_SNAPSHOT
    assert snapshot.historical_actual is False
    assert snapshot.effective_from == snapshot.effective_to == ACCOUNT_SNAPSHOT_DATE
    with pytest.raises(UnknownCostEvidence):
        ledger.brokerage_record(date(2026, 9, 8), LedgerProduct.INTRADAY)
    with pytest.raises(UnknownCostEvidence):
        ledger.brokerage_record(date(2026, 9, 10), LedgerProduct.INTRADAY)


def test_sep_10_brokerage_actual_is_unknown() -> None:
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence) as exc:
        ledger.brokerage_record(date(2026, 9, 10), LedgerProduct.INTRADAY)
    assert exc.value.unknowns
    assert any("forward" in item or "2026-09-09" in item for item in exc.value.unknowns)


def test_scenario_uses_explicit_assumption_without_relabelling_actual() -> None:
    scenario = _scenario(
        scenario_date=ACCOUNT_SNAPSHOT_DATE,
        research_start=date(2026, 9, 1),
        research_end=date(2026, 9, 30),
    )
    assert scenario.historical_actual is False
    assert scenario.classification != HISTORICAL_ACTUAL_LABEL
    rate, provenance = scenario.rate_for(LedgerComponent.BROKERAGE, LedgerSide.BUY)
    assert rate == Decimal("0.0006")
    assert provenance == "assumed"
    # The snapshot record itself never enters the known identities, even on Sep 9.
    assert not any("account_snapshot" in identity for identity in scenario.known_record_ids)
    assert LedgerComponent.BROKERAGE.value in scenario.assumed_components


def test_assumption_change_changes_fingerprint() -> None:
    first = _scenario()
    changed_assumptions = (
        _assumption(LedgerComponent.BROKERAGE, LedgerProduct.INTRADAY, "0.001"),
        _assumption(LedgerComponent.GST, LedgerProduct.INTRADAY, "0.18"),
        _assumption(LedgerComponent.CLEARING, LedgerProduct.INTRADAY, "0.000001"),
    )
    second = _scenario(assumptions=changed_assumptions)
    assert second.fingerprint() != first.fingerprint()
    rate, _ = second.rate_for(LedgerComponent.BROKERAGE, LedgerSide.SELL)
    assert rate == Decimal("0.001")


def test_unknown_never_becomes_zero() -> None:
    with pytest.raises(MissingScenarioAssumptionError):
        _scenario(assumptions=())
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence):
        ledger.resolve(
            LedgerComponent.CLEARING,
            date(2025, 6, 1),
            LedgerProduct.INTRADAY,
            LedgerSide.BOTH,
        )
    scenario = _scenario()
    for record in ledger.records:
        if record.component is LedgerComponent.CLEARING:
            assert record.rate is None
    rate, provenance = scenario.rate_for(LedgerComponent.CLEARING, LedgerSide.BOTH)
    assert rate == Decimal("0.000001")
    assert provenance == "assumed"


def test_gst_public_scenario_cannot_leak_into_historical_actual() -> None:
    ledger = _ledger()
    with pytest.raises(UnknownCostEvidence):
        ledger.resolve(
            LedgerComponent.GST,
            date(2025, 6, 1),
            LedgerProduct.INTRADAY,
            LedgerSide.BOTH,
        )
    scenario = _scenario()
    assert not any(
        EvidenceClass.BROKER_PUBLIC_SCENARIO.value in identity
        for identity in scenario.known_record_ids
    )
    with pytest.raises(MissingScenarioAssumptionError):
        _scenario(
            assumptions=(
                _assumption(LedgerComponent.BROKERAGE, LedgerProduct.INTRADAY, "0.0006"),
                _assumption(LedgerComponent.CLEARING, LedgerProduct.INTRADAY, "0.000001"),
            )
        )


def test_delivery_does_not_inherit_intraday_gst() -> None:
    intraday_gst = _assumption(LedgerComponent.GST, LedgerProduct.INTRADAY, "0.18")
    with pytest.raises(ScenarioError, match="does not match scenario product"):
        _scenario(product=LedgerProduct.DELIVERY, assumptions=(intraday_gst,))
    delivery_assumptions = (
        _assumption(LedgerComponent.BROKERAGE, LedgerProduct.DELIVERY, "0.001"),
        _assumption(LedgerComponent.GST, LedgerProduct.DELIVERY, "0.12"),
        _assumption(LedgerComponent.CLEARING, LedgerProduct.DELIVERY, "0.000001"),
        _assumption(LedgerComponent.DP_DEMAT, LedgerProduct.DELIVERY, "15.0"),
    )
    scenario = _scenario(product=LedgerProduct.DELIVERY, assumptions=delivery_assumptions)
    assert scenario.historical_actual is False
    rate, provenance = scenario.rate_for(LedgerComponent.GST, LedgerSide.BOTH)
    assert rate == Decimal("0.12")
    assert provenance == "assumed"


def test_scenario_always_historical_actual_false() -> None:
    scenario = _scenario()
    assert scenario.historical_actual is False
    assert scenario.to_dict()["historical_actual"] is False
    assert "HISTORICAL_ACTUAL" not in json.dumps(scenario.to_dict())
    with pytest.raises(ScenarioError, match="never become HISTORICAL_ACTUAL"):
        HistoricalCostScenario(
            scenario_id="forged",
            schema_version=SCHEMA_VERSION,
            research_start=date(2025, 1, 1),
            research_end=date(2025, 12, 31),
            scenario_date=date(2025, 6, 1),
            product=LedgerProduct.INTRADAY,
            ledger_fingerprint="f" * 64,
            known_record_ids=(),
            evidence_classes=(),
            assumptions=(),
            assumed_components=(),
            unknowns=(),
            resolved_rates=(),
            classification="SCENARIO",
            historical_actual=True,
        )


def test_deterministic_serialization() -> None:
    first = _scenario().to_json()
    second = _scenario().to_json()
    assert first == second
    assert json.loads(first) == json.loads(second)
    assert first.endswith("\n")
    assert _scenario().fingerprint() == _scenario().fingerprint()
    assert _scenario().to_dict() == json.loads(json.dumps(_scenario().to_dict()))


def test_scenario_date_must_lie_inside_research_window() -> None:
    with pytest.raises(ScenarioError, match="inside the research window"):
        _scenario(scenario_date=date(2024, 1, 1))
    with pytest.raises(ScenarioError, match="inside the research window"):
        _scenario(
            scenario_date=date(2025, 6, 1),
            research_start=date(2025, 7, 1),
            research_end=date(2025, 12, 31),
        )


def test_unsupported_scenario_date_fails_closed() -> None:
    with pytest.raises(ScenarioError, match="supported research boundary"):
        _scenario(
            scenario_date=date(2024, 6, 30),
            research_start=date(2024, 6, 1),
            research_end=date(2024, 6, 30),
        )


def test_assumption_cannot_override_evidence() -> None:
    with pytest.raises(ScenarioError, match="must not override evidence"):
        _scenario(
            assumptions=_intraday_assumptions()
            + (_assumption(LedgerComponent.STT, LedgerProduct.INTRADAY, "0.999"),)
        )


def test_product_all_rejected_everywhere() -> None:
    with pytest.raises(ScenarioError, match="must be INTRADAY or DELIVERY"):
        _scenario(product=LedgerProduct.ALL)
    with pytest.raises(ScenarioError, match="never.*ALL|must be INTRADAY"):
        _assumption(LedgerComponent.GST, LedgerProduct.ALL, "0.18")


def test_side_divergent_component_requires_per_side_query() -> None:
    scenario = _scenario()
    buy_rate, buy_provenance = scenario.rate_for(LedgerComponent.STT, LedgerSide.BUY)
    sell_rate, sell_provenance = scenario.rate_for(LedgerComponent.STT, LedgerSide.SELL)
    assert (buy_rate, buy_provenance) == (Decimal(0), "ledger")
    assert (sell_rate, sell_provenance) == (Decimal("0.00025"), "ledger")
    with pytest.raises(ScenarioError, match="no scenario rate"):
        scenario.rate_for(LedgerComponent.STT, LedgerSide.BOTH)


def test_module_has_no_live_order_or_network_capability() -> None:
    source = Path(__file__).parents[1] / "src" / "equity_engine" / "historical_cost_scenario.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("import httpx", "import requests", "place_order", "import socket"):
        assert forbidden not in text
    assert "live" not in text.lower() or "never" in text.lower()
