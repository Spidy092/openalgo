"""Research-only historical cost scenarios over the effective-dated ledger.

We cannot currently claim full account-specific historical actual all-in
costs, but backtesting still needs explicitly labelled cost scenarios. This
module binds one scenario to ledger evidence plus explicit, named, fingerprinted
assumptions for the components the ledger leaves unknown.

Safety contract (load-bearing):

- Known components resolve from :class:`EffectiveDatedCostLedger` default
  resolution, and only when the record is historical actual. A resolved but
  non-actual record (for example the single-day 2026-09-09 account snapshot,
  even on 2026-09-09 itself) never counts as evidence and is never projected
  backward or forward.
- Unknown components stay explicit. ``UNKNOWN != ZERO``: a required component
  with neither evidence nor an explicit assumption fails closed instead of
  defaulting to zero.
- A scenario can never become ``HISTORICAL_ACTUAL_COSTS``: ``historical_actual``
  is always ``False`` and construction with ``True`` raises.
- Current public broker pricing never leaks into historical resolution: the
  builder uses default ledger resolution only, which excludes
  ``BROKER_PUBLIC_SCENARIO`` records.
- Scenarios are product-specific. ``INTRADAY`` and ``DELIVERY`` never share an
  assumption or formula unless it is declared separately per product.

This module never calls broker APIs, places orders, downloads data, or wires
into live order execution. It produces labelled quote artifacts only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from .cost_ledger import (
    SCENARIO_LABEL,
    SUPPORTED_RESEARCH_START,
    EffectiveDatedCostLedger,
    EvidenceClass,
    LedgerComponent,
    LedgerProduct,
    LedgerSide,
    UnknownCostEvidence,
)

SCHEMA_VERSION = "historical-cost-scenario/v1"


class ScenarioError(ValueError):
    """Base error for invalid historical cost scenarios."""


class MissingScenarioAssumptionError(ScenarioError):
    """Raised when a required component has neither evidence nor an assumption."""


def _record_identity(
    *,
    component: LedgerComponent,
    product: LedgerProduct,
    side: LedgerSide,
    effective_from: date,
    effective_to: date | None,
    evidence_class: EvidenceClass,
) -> str:
    """Return a stable identity for one ledger record used by a scenario."""
    return ":".join(
        (
            component.value,
            product.value,
            side.value,
            effective_from.isoformat(),
            effective_to.isoformat() if effective_to else "",
            evidence_class.value,
        )
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        {str(key): _canonical_value(value) for key, value in sorted(payload.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class ScenarioAssumption:
    """One explicit, named assumption for a ledger-unknown component.

    The assumption is scoped to a single product (``INTRADAY`` or ``DELIVERY``,
    never ``ALL``): products must not share an unsupported formula. An explicit
    zero rate is the caller's stated claim, recorded with its source and reason;
    it is never produced implicitly for an unknown.
    """

    assumption_id: str
    component: LedgerComponent
    product: LedgerProduct
    basis: str
    rate: Decimal
    formula: str
    source: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("assumption_id", "basis", "formula", "source", "reason"):
            if not str(getattr(self, name)).strip():
                raise ScenarioError(f"assumption {name} is required")
        if self.product is LedgerProduct.ALL:
            raise ScenarioError(
                "assumption product must be INTRADAY or DELIVERY; "
                "products must not share an unsupported formula"
            )
        if not isinstance(self.rate, Decimal) or not self.rate.is_finite() or self.rate < 0:
            raise ScenarioError("assumption rate must be a finite non-negative Decimal")

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe mapping."""
        return {
            "assumption_id": self.assumption_id,
            "basis": self.basis,
            "component": self.component.value,
            "formula": self.formula,
            "product": self.product.value,
            "rate": format(self.rate, "f"),
            "reason": self.reason,
            "source": self.source,
        }


@dataclass(frozen=True)
class HistoricalCostScenario:
    """One fingerprinted historical cost scenario for a research window.

    ``historical_actual`` is always ``False``: scenarios describe labelled
    assumptions over partial evidence and can never become historical actuals.
    """

    scenario_id: str
    schema_version: str
    research_start: date
    research_end: date
    scenario_date: date
    product: LedgerProduct
    ledger_fingerprint: str
    known_record_ids: tuple[str, ...]
    evidence_classes: tuple[str, ...]
    assumptions: tuple[ScenarioAssumption, ...]
    assumed_components: tuple[str, ...]
    unknowns: tuple[str, ...]
    resolved_rates: tuple[tuple[str, str, str, str, str | None, str | None], ...]
    classification: str
    historical_actual: bool

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ScenarioError("scenario_id is required")
        if self.schema_version != SCHEMA_VERSION:
            raise ScenarioError(f"unsupported schema {self.schema_version!r}")
        if self.research_start > self.research_end:
            raise ScenarioError("research_start must be on or before research_end")
        if not self.research_start <= self.scenario_date <= self.research_end:
            raise ScenarioError("scenario_date must lie inside the research window")
        if self.product is LedgerProduct.ALL:
            raise ScenarioError(
                "scenario product must be INTRADAY or DELIVERY; "
                "products must not share an unsupported formula"
            )
        if self.historical_actual:
            raise ScenarioError("historical scenarios can never become HISTORICAL_ACTUAL_COSTS")
        if self.classification != SCENARIO_LABEL:
            raise ScenarioError("historical scenarios are always labelled SCENARIO")
        if not self.ledger_fingerprint.strip():
            raise ScenarioError("ledger_fingerprint is required")
        for assumption in self.assumptions:
            if assumption.product is not self.product:
                raise ScenarioError(
                    f"assumption {assumption.assumption_id} product "
                    f"{assumption.product.value} does not match scenario product "
                    f"{self.product.value}"
                )
        assumption_components = [item.component for item in self.assumptions]
        if len(set(assumption_components)) != len(assumption_components):
            raise ScenarioError("assumptions contain duplicate components")
        if tuple(sorted(item.value for item in assumption_components)) != tuple(
            self.assumed_components
        ):
            raise ScenarioError("assumed_components must list exactly the assumed components")

    def deterministic_payload(self) -> dict[str, Any]:
        """Return the canonical identity payload (includes historical_actual=false)."""
        return {
            "assumed_components": list(self.assumed_components),
            "assumptions": [item.as_dict() for item in self.assumptions],
            "classification": self.classification,
            "evidence_classes": list(self.evidence_classes),
            "historical_actual": self.historical_actual,
            "known_record_ids": list(self.known_record_ids),
            "ledger_fingerprint": self.ledger_fingerprint,
            "product": self.product.value,
            "research_end": self.research_end.isoformat(),
            "research_start": self.research_start.isoformat(),
            "resolved_rates": [list(item) for item in self.resolved_rates],
            "scenario_date": self.scenario_date.isoformat(),
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "unknowns": list(self.unknowns),
        }

    def fingerprint(self) -> str:
        """Return the deterministic scenario fingerprint."""
        return hashlib.sha256(_canonical_json_bytes(self.deterministic_payload())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe mapping including identity."""
        payload = self.deterministic_payload()
        payload["fingerprint"] = self.fingerprint()
        return payload

    def to_json(self) -> str:
        """Return canonical JSON with sorted keys and stable separators."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"

    def rate_for(self, component: LedgerComponent, side: LedgerSide) -> tuple[Decimal, str]:
        """Return the scenario (rate, provenance) for one component and side.

        Provenance is ``"ledger"`` for evidenced records and ``"assumed"`` for
        explicit assumptions. Unknowns never appear here: the builder fails
        closed instead of recording them. A ``BOTH`` query succeeds only when
        the recorded ``BUY`` and ``SELL`` entries agree on rate and provenance;
        side-divergent components (for example intraday STT) must be queried
        per side.
        """
        for entry_component, entry_side, entry_rate, provenance, _, _ in self.resolved_rates:
            if entry_component == component.value and entry_side == side.value:
                return Decimal(entry_rate), provenance
        if side is LedgerSide.BOTH:
            agreed = {
                (entry_rate, provenance)
                for entry_component, entry_side, entry_rate, provenance, _, _ in self.resolved_rates
                if entry_component == component.value
            }
            if len(agreed) == 1:
                entry_rate, provenance = agreed.pop()
                return Decimal(entry_rate), provenance
        raise ScenarioError(
            f"no scenario rate for {component.value} {side.value}; "
            "scenarios never synthesize rates for uncovered components"
        )


def compile_historical_scenario(
    *,
    scenario_id: str,
    ledger: EffectiveDatedCostLedger,
    scenario_date: date,
    research_start: date,
    research_end: date,
    product: LedgerProduct,
    assumptions: tuple[ScenarioAssumption, ...] = (),
) -> HistoricalCostScenario:
    """Compile a labelled historical scenario over ledger evidence plus assumptions.

    Known components resolve through default ledger resolution and count only
    when the record is historical actual; anything else (including the
    single-day account snapshot, on any date) requires an explicit assumption.
    A required component with neither evidence nor an assumption fails closed.
    An assumption alongside full evidence for the same component also fails
    closed, so explicit claims can never silently override evidence.
    """
    if not scenario_id.strip():
        raise ScenarioError("scenario_id is required")
    if product is LedgerProduct.ALL:
        raise ScenarioError(
            "scenario product must be INTRADAY or DELIVERY; "
            "products must not share an unsupported formula"
        )
    if research_start > research_end:
        raise ScenarioError("research_start must be on or before research_end")
    if not research_start <= scenario_date <= research_end:
        raise ScenarioError("scenario_date must lie inside the research window")
    if scenario_date < SUPPORTED_RESEARCH_START:
        raise ScenarioError(
            f"scenario_date {scenario_date.isoformat()} precedes the supported "
            f"research boundary {SUPPORTED_RESEARCH_START.isoformat()}"
        )
    for assumption in assumptions:
        if assumption.product is not product:
            raise ScenarioError(
                f"assumption {assumption.assumption_id} product "
                f"{assumption.product.value} does not match scenario product "
                f"{product.value}; delivery does not inherit intraday formulas"
            )
    assumption_components = [item.component for item in assumptions]
    if len(set(assumption_components)) != len(assumption_components):
        raise ScenarioError("assumptions contain duplicate components")
    assumption_by_component = {item.component: item for item in assumptions}

    known_ids: list[str] = []
    evidence_classes: list[str] = []
    assumed: list[str] = []
    unknowns: list[str] = []
    resolved: list[tuple[str, str, str, str, str | None, str | None]] = []

    for component in ledger.required_components(product):
        sides_known = 0
        for side in (LedgerSide.BUY, LedgerSide.SELL):
            try:
                record = ledger.resolve(component, scenario_date, product, side)
            except UnknownCostEvidence as exc:
                if component not in assumption_by_component:
                    raise MissingScenarioAssumptionError(
                        f"{component.value} has neither ledger evidence nor an explicit "
                        f"scenario assumption for {scenario_date.isoformat()} "
                        f"{product.value}; unknown is never zero"
                    ) from exc
                unknowns.extend(exc.unknowns)
                continue
            if record.rate is None or not record.historical_actual:
                if component not in assumption_by_component:
                    raise MissingScenarioAssumptionError(
                        f"{component.value} has neither sufficient ledger evidence nor an "
                        f"explicit scenario assumption for {scenario_date.isoformat()} "
                        f"{product.value}; unknown is never zero"
                    )
                unknowns.append(
                    f"{component.value}: {record.evidence_class.value} "
                    "is not sufficient historical-account evidence"
                )
                unknowns.extend(record.unknowns)
                continue
            known_ids.append(
                _record_identity(
                    component=record.component,
                    product=record.product,
                    side=record.side,
                    effective_from=record.effective_from,
                    effective_to=record.effective_to,
                    evidence_class=record.evidence_class,
                )
            )
            evidence_classes.append(record.evidence_class.value)
            resolved.append(
                (
                    component.value,
                    side.value,
                    format(record.rate, "f"),
                    "ledger",
                    known_ids[-1],
                    None,
                )
            )
            sides_known += 1
        if sides_known == 2 and component in assumption_by_component:
            raise ScenarioError(
                f"assumption {assumption_by_component[component].assumption_id} is unnecessary: "
                f"{component.value} is fully evidenced and assumptions must not override evidence"
            )
        if sides_known < 2:
            assumption = assumption_by_component[component]
            assumed.append(component.value)
            for side in (LedgerSide.BUY, LedgerSide.SELL):
                if not any(
                    entry[0] == component.value and entry[1] == side.value for entry in resolved
                ):
                    resolved.append(
                        (
                            component.value,
                            side.value,
                            format(assumption.rate, "f"),
                            "assumed",
                            None,
                            assumption.assumption_id,
                        )
                    )

    resolved.sort(key=lambda entry: (entry[0], entry[1]))
    return HistoricalCostScenario(
        scenario_id=scenario_id,
        schema_version=SCHEMA_VERSION,
        research_start=research_start,
        research_end=research_end,
        scenario_date=scenario_date,
        product=product,
        ledger_fingerprint=ledger.fingerprint(),
        known_record_ids=tuple(sorted(set(known_ids))),
        evidence_classes=tuple(sorted(set(evidence_classes))),
        assumptions=tuple(sorted(assumptions, key=lambda item: item.assumption_id)),
        assumed_components=tuple(sorted(set(assumed))),
        unknowns=tuple(dict.fromkeys(unknowns)),
        resolved_rates=tuple(resolved),
        classification=SCENARIO_LABEL,
        historical_actual=False,
    )
