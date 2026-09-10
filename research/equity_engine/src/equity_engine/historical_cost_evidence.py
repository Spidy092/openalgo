"""Historical transaction-cost evidence matrix and canonical scenario assumptions.

Covers the research window 2024-10-01 through 2026-09-08 across every ledger
component, product (INTRADAY vs DELIVERY), and side (BUY vs SELL).

Evidence grades (market-realism taxonomy):

- ``HISTORICAL_ACTUAL``: account-specific proven all-in costs. Empty with
  current evidence by design: nothing account-specific is proven, and the
  single-day 2026-09-09 snapshot is never projected backward or forward.
- ``DOCUMENTED_STATUTORY``: exchange, clearing-corporation, SEBI, or statutory
  schedules evidenced by public circulars and kept at their effective dates:
  STT (equity cash legs unchanged through every in-window F&O-only revision),
  uniform stamp duty since 2020-07-01, flat SEBI turnover fee Rs 10/crore, and
  the NSE true-to-label uniform MII schedule (Rs 297 + Rs 10 through
  2026-02-28, Rs 306.99 + Rs 0.01 from 2026-03-01 per NSE/FA/73061).
- ``BROKER_PUBLIC_SCENARIO``: current public broker pricing, isolated from
  default historical resolution and never historical-account evidence.
- ``ACCOUNT_OBSERVED_CURRENT``: the single-account 2026-09-09 observation,
  applicable only to that date.
- ``UNKNOWN``: no sufficient evidence. ``UNKNOWN != ZERO`` everywhere.

Where history cannot be proven, backtesting uses explicit deterministic
:data:`ScenarioAssumption` sets (one per product, never shared) compiled
through the single canonical :class:`HistoricalCostScenario` implementation.
Assumption values are illustrative, named, sourced, reasoned, and
fingerprinted; they are never tuned for profitability and never relabelled as
actuals. Spread/slippage have no observations at all, so friction evidence is
named stress levels only.

This module never calls broker APIs, places orders, downloads data, or wires
into live order execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any

from .cost_ledger import (
    _DEFAULT_RESOLVE_CLASSES,
    BROKERAGE_CAP,
    BROKERAGE_PUBLIC_RATE,
    GST_RATE,
    CostEvidenceRecord,
    EffectiveDatedCostLedger,
    EvidenceClass,
    LedgerComponent,
    LedgerProduct,
    LedgerSide,
)
from .historical_cost_scenario import (
    HistoricalCostScenario,
    ScenarioAssumption,
    compile_historical_scenario,
)

EVIDENCE_WINDOW_START = date(2024, 10, 1)
EVIDENCE_WINDOW_END = date(2026, 9, 8)

SCHEMA_VERSION = "historical-cost-evidence/v1"
FRICTION_SCHEMA_VERSION = "historical-friction-evidence/v1"


class EvidenceGrade(StrEnum):
    """Market-realism grade for one evidence cell."""

    HISTORICAL_ACTUAL = "HISTORICAL_ACTUAL"
    DOCUMENTED_STATUTORY = "DOCUMENTED_STATUTORY"
    BROKER_PUBLIC_SCENARIO = "BROKER_PUBLIC_SCENARIO"
    ACCOUNT_OBSERVED_CURRENT = "ACCOUNT_OBSERVED_CURRENT"
    UNKNOWN = "UNKNOWN"


_EVIDENCE_CLASS_GRADES = {
    EvidenceClass.STATUTORY_SCHEDULE: EvidenceGrade.DOCUMENTED_STATUTORY,
    EvidenceClass.MII_SCHEDULE: EvidenceGrade.DOCUMENTED_STATUTORY,
    EvidenceClass.ACCOUNT_SNAPSHOT: EvidenceGrade.ACCOUNT_OBSERVED_CURRENT,
    EvidenceClass.BROKER_PUBLIC_SCENARIO: EvidenceGrade.BROKER_PUBLIC_SCENARIO,
    EvidenceClass.SCENARIO: EvidenceGrade.BROKER_PUBLIC_SCENARIO,
    EvidenceClass.UNKNOWN: EvidenceGrade.UNKNOWN,
}


@dataclass(frozen=True)
class EvidenceCell:
    """One graded (component, date-range, product, side) fact."""

    component: LedgerComponent
    product: LedgerProduct
    side: LedgerSide
    date_from: date
    date_to: date
    grade: EvidenceGrade
    record_identities: tuple[str, ...]
    rate_known: bool
    note: str

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe mapping."""
        return {
            "component": self.component.value,
            "date_from": self.date_from.isoformat(),
            "date_to": self.date_to.isoformat(),
            "grade": self.grade.value,
            "note": self.note,
            "product": self.product.value,
            "rate_known": self.rate_known,
            "record_identities": list(self.record_identities),
            "side": self.side.value,
        }


def _record_identity(record: CostEvidenceRecord) -> str:
    return ":".join(
        (
            record.component.value,
            record.product.value,
            record.side.value,
            record.effective_from.isoformat(),
            record.effective_to.isoformat() if record.effective_to else "",
            record.evidence_class.value,
        )
    )


def build_evidence_matrix(
    ledger: EffectiveDatedCostLedger,
    start: date = EVIDENCE_WINDOW_START,
    end: date = EVIDENCE_WINDOW_END,
    products: tuple[LedgerProduct, ...] = (LedgerProduct.INTRADAY, LedgerProduct.DELIVERY),
) -> EvidenceMatrix:
    """Grade every required component over the window from default resolution.

    One cell is emitted per ledger record clipped to the window, so effective-date
    transitions (for example the 2026-03-01 MII revision) appear as distinct
    cells. Public-scenario records never enter: only default resolution classes
    are considered.
    """
    if start > end:
        raise ValueError("evidence window start must be on or before end")
    cells: list[EvidenceCell] = []
    for product in products:
        if product is LedgerProduct.ALL:
            raise ValueError("matrix products must be INTRADAY or DELIVERY")
        for component in ledger.required_components(product):
            for side in (LedgerSide.BUY, LedgerSide.SELL):
                covering = sorted(
                    (
                        record
                        for record in ledger.records
                        if record.component is component
                        and record.applies_to(product, side)
                        and record.evidence_class in _DEFAULT_RESOLVE_CLASSES
                        and record.effective_from <= end
                        and (record.effective_to is None or record.effective_to >= start)
                    ),
                    key=lambda record: record.effective_from,
                )
                if not covering:
                    cells.append(
                        EvidenceCell(
                            component=component,
                            product=product,
                            side=side,
                            date_from=start,
                            date_to=end,
                            grade=EvidenceGrade.UNKNOWN,
                            record_identities=(),
                            rate_known=False,
                            note="no default-resolution record covers the window",
                        )
                    )
                    continue
                for record in covering:
                    clipped_from = max(record.effective_from, start)
                    clipped_to = (
                        min(record.effective_to, end) if record.effective_to is not None else end
                    )
                    grade = _EVIDENCE_CLASS_GRADES[record.evidence_class]
                    cells.append(
                        EvidenceCell(
                            component=component,
                            product=record.product,
                            side=record.side,
                            date_from=clipped_from,
                            date_to=clipped_to,
                            grade=grade,
                            record_identities=(_record_identity(record),),
                            rate_known=record.rate is not None and record.historical_actual,
                            note="; ".join(record.unknowns)
                            if record.rate is None or not record.historical_actual
                            else "",
                        )
                    )
    cells.sort(
        key=lambda cell: (
            cell.component.value,
            cell.product.value,
            cell.side.value,
            cell.date_from.isoformat(),
        )
    )
    deduplicated: list[EvidenceCell] = []
    seen: set[tuple[str, ...]] = set()
    for cell in cells:
        key = (
            cell.component.value,
            cell.product.value,
            cell.side.value,
            cell.date_from.isoformat(),
            cell.date_to.isoformat(),
            *cell.record_identities,
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(cell)
    return EvidenceMatrix(
        window_start=start,
        window_end=end,
        ledger_fingerprint=ledger.fingerprint(),
        cells=tuple(deduplicated),
    )


@dataclass(frozen=True)
class EvidenceMatrix:
    """Fingerprinted evidence grades for a research window."""

    window_start: date
    window_end: date
    ledger_fingerprint: str
    cells: tuple[EvidenceCell, ...]

    def grades_present(self) -> tuple[str, ...]:
        """Return the sorted unique grades appearing in the matrix."""
        return tuple(sorted({cell.grade.value for cell in self.cells}))

    def cells_for(
        self,
        component: LedgerComponent,
        product: LedgerProduct,
        side: LedgerSide,
    ) -> tuple[EvidenceCell, ...]:
        """Return cells for one component/product/side in date order.

        ``ALL``-product and ``BOTH``-side records apply to specific queries,
        mirroring ledger resolution scope.
        """
        return tuple(
            cell
            for cell in self.cells
            if cell.component is component
            and (cell.product is product or cell.product is LedgerProduct.ALL)
            and (cell.side is side or cell.side is LedgerSide.BOTH)
        )

    def unknown_cells(self) -> tuple[EvidenceCell, ...]:
        """Return cells graded UNKNOWN (explicit, never zero)."""
        return tuple(cell for cell in self.cells if cell.grade is EvidenceGrade.UNKNOWN)

    def deterministic_payload(self) -> dict[str, Any]:
        """Return the canonical identity payload."""
        return {
            "cells": [cell.as_dict() for cell in self.cells],
            "ledger_fingerprint": self.ledger_fingerprint,
            "schema_version": SCHEMA_VERSION,
            "window_end": self.window_end.isoformat(),
            "window_start": self.window_start.isoformat(),
        }

    def fingerprint(self) -> str:
        """Return the deterministic matrix fingerprint."""
        payload = json.dumps(self.deterministic_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe mapping including identity."""
        payload = self.deterministic_payload()
        payload["fingerprint"] = self.fingerprint()
        return payload

    def to_json(self) -> str:
        """Return canonical JSON with sorted keys and stable separators."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"


def _assumption(
    assumption_id: str,
    component: LedgerComponent,
    product: LedgerProduct,
    basis: str,
    rate: Decimal,
    formula: str,
    source: str,
    reason: str,
) -> ScenarioAssumption:
    return ScenarioAssumption(
        assumption_id=assumption_id,
        component=component,
        product=product,
        basis=basis,
        rate=rate,
        formula=formula,
        source=source,
        reason=reason,
    )


def intraday_cost_assumptions() -> tuple[ScenarioAssumption, ...]:
    """Return the canonical named assumptions for UNKNOWN intraday components.

    Brokerage follows the shape of current public pricing (0.1% capped at Rs 20)
    as an illustrative scenario only; it is not account evidence and must never
    be read as the historical account rate. GST uses the public intraday taxable
    base; broker sheets vary (some also tax SEBI fees), so the base is a named
    claim, not a proven composition. Clearing and DP lines are explicitly zero
    with handling reasons, never silent zeroes.
    """
    return (
        _assumption(
            "intraday-brokerage-public-illustrative-v1",
            LedgerComponent.BROKERAGE,
            LedgerProduct.INTRADAY,
            "turnover",
            BROKERAGE_PUBLIC_RATE,
            f"min(turnover * {BROKERAGE_PUBLIC_RATE}, {BROKERAGE_CAP})",
            "https://upstox.com/brokerage-charges/",
            "illustrative current public pricing shape only; no historical "
            "account brokerage is evidenced and the 2026-09-09 snapshot is not projected",
        ),
        _assumption(
            "intraday-gst-public-base-illustrative-v1",
            LedgerComponent.GST,
            LedgerProduct.INTRADAY,
            "taxable_base:brokerage+transaction+ipft",
            GST_RATE,
            f"(brokerage + transaction + ipft) * {GST_RATE}",
            "https://upstox.com/brokerage-charges/",
            "named public intraday base only; broker sheets vary and the "
            "account-specific historical base is unknown",
        ),
        _assumption(
            "intraday-clearing-no-separate-line-v1",
            LedgerComponent.CLEARING,
            LedgerProduct.INTRADAY,
            "turnover",
            Decimal(0),
            "0",
            "retail contract-note convention",
            "retail NSE cash contract notes typically carry no separate clearing line; "
            "explicit zero for the turnover model, not a proven charge",
        ),
    )


def delivery_cost_assumptions() -> tuple[ScenarioAssumption, ...]:
    """Return the canonical named assumptions for UNKNOWN delivery components.

    Declared separately from intraday: delivery never inherits intraday
    formulas. DP debit charges are flat per-ISIN settlement debits, not
    turnover-proportional, so the turnover model carries an explicit zero with
    a handle-separately reason instead of a fabricated rate.
    """
    return (
        _assumption(
            "delivery-brokerage-public-illustrative-v1",
            LedgerComponent.BROKERAGE,
            LedgerProduct.DELIVERY,
            "turnover",
            BROKERAGE_PUBLIC_RATE,
            f"min(turnover * {BROKERAGE_PUBLIC_RATE}, {BROKERAGE_CAP})",
            "https://upstox.com/brokerage-charges/",
            "illustrative current public pricing shape only; no historical "
            "account delivery brokerage is evidenced",
        ),
        _assumption(
            "delivery-gst-separate-base-illustrative-v1",
            LedgerComponent.GST,
            LedgerProduct.DELIVERY,
            "taxable_base:brokerage+transaction+sebi",
            GST_RATE,
            f"(brokerage + transaction + sebi_turnover) * {GST_RATE}",
            "broker charge-sheet convention",
            "separately declared delivery base; not shared with intraday and not proven",
        ),
        _assumption(
            "delivery-clearing-no-separate-line-v1",
            LedgerComponent.CLEARING,
            LedgerProduct.DELIVERY,
            "turnover",
            Decimal(0),
            "0",
            "retail contract-note convention",
            "retail NSE cash contract notes typically carry no separate clearing line; "
            "explicit zero for the turnover model, not a proven charge",
        ),
        _assumption(
            "delivery-dp-flat-debit-excluded-v1",
            LedgerComponent.DP_DEMAT,
            LedgerProduct.DELIVERY,
            "per-settlement-debit",
            Decimal(0),
            "0",
            "depository tariff convention",
            "DP debits are flat per-ISIN settlement charges, not turnover-proportional; "
            "excluded from the turnover model and must be handled separately",
        ),
    )


def canonical_assumptions_for(product: LedgerProduct) -> tuple[ScenarioAssumption, ...]:
    """Return the canonical assumption set for one product."""
    if product is LedgerProduct.INTRADAY:
        return intraday_cost_assumptions()
    if product is LedgerProduct.DELIVERY:
        return delivery_cost_assumptions()
    raise ValueError("canonical assumptions exist only for INTRADAY or DELIVERY")


@dataclass(frozen=True)
class FrictionScenarioEvidence:
    """One named, fingerprinted friction stress level. Never an observation."""

    scenario_id: str
    slippage_bps_per_leg: Decimal
    half_spread_bps_per_leg: Decimal
    source: str
    reason: str
    observed: bool = False

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("friction scenario_id is required")
        for name in ("slippage_bps_per_leg", "half_spread_bps_per_leg"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite non-negative Decimal")
        if not self.source.strip() or not self.reason.strip():
            raise ValueError("friction source and reason are required")
        if self.observed:
            raise ValueError("friction scenarios are stress levels, never observations")

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe mapping."""
        return {
            "half_spread_bps_per_leg": format(self.half_spread_bps_per_leg, "f"),
            "observed": self.observed,
            "reason": self.reason,
            "scenario_id": self.scenario_id,
            "slippage_bps_per_leg": format(self.slippage_bps_per_leg, "f"),
            "source": self.source,
        }


def named_friction_scenarios() -> tuple[FrictionScenarioEvidence, ...]:
    """Return illustrative friction stress levels (round, untuned, non-observed).

    No spread or slippage observations exist, so nothing here is calibrated to
    market data. Levels are round stress rungs for sensitivity analysis only.
    """
    return (
        FrictionScenarioEvidence(
            scenario_id="friction-frictionless-baseline-v1",
            slippage_bps_per_leg=Decimal(0),
            half_spread_bps_per_leg=Decimal(0),
            source="research-methodology",
            reason="frictionless baseline isolating transaction-cost effects; not a market claim",
        ),
        FrictionScenarioEvidence(
            scenario_id="friction-moderate-stress-v1",
            slippage_bps_per_leg=Decimal(2),
            half_spread_bps_per_leg=Decimal(1),
            source="research-methodology",
            reason="moderate illustrative stress rung; not calibrated to observations",
        ),
        FrictionScenarioEvidence(
            scenario_id="friction-severe-stress-v1",
            slippage_bps_per_leg=Decimal(5),
            half_spread_bps_per_leg=Decimal(5),
            source="research-methodology",
            reason="severe illustrative stress rung; not calibrated to observations",
        ),
    )


def friction_evidence_fingerprint(
    scenarios: tuple[FrictionScenarioEvidence, ...] = named_friction_scenarios(),
) -> str:
    """Return the deterministic fingerprint of a friction scenario set."""
    payload = json.dumps(
        {
            "schema_version": FRICTION_SCHEMA_VERSION,
            "scenarios": [item.as_dict() for item in scenarios],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compile_evidence_scenario(
    *,
    scenario_id: str,
    ledger: EffectiveDatedCostLedger,
    scenario_date: date,
    research_start: date,
    research_end: date,
    product: LedgerProduct,
) -> HistoricalCostScenario:
    """Compile a scenario using the canonical assumption set for the product.

    Uses the single canonical :class:`HistoricalCostScenario` implementation.
    Fail-closed behavior (missing assumptions, cross-product leakage) is
    inherited unchanged.
    """
    return compile_historical_scenario(
        scenario_id=scenario_id,
        ledger=ledger,
        scenario_date=scenario_date,
        research_start=research_start,
        research_end=research_end,
        product=product,
        assumptions=canonical_assumptions_for(product),
    )
