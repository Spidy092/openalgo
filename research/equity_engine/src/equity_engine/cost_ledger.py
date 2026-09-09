"""Effective-dated cost evidence ledger.

Machine-readable provenance layer for historical equity costs. It answers which
rate applied on a given date, with what evidence, and -- critically -- what is
still unknown.

Invariant: UNKNOWN != ZERO. A missing component is represented with
``rate=None`` and an explicit ``unknowns`` explanation. It is never silently
treated as zero.

This module never calls broker APIs and never places orders. It is additive
provenance that can later compose with the simulator; it does not replace the
current-terms providers in ``documented_costs`` or ``observed_costs``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum


SUPPORTED_RESEARCH_START = date(2024, 7, 1)

MII_A_START = date(2024, 10, 1)
MII_A_END = date(2026, 2, 28)
MII_B_START = date(2026, 3, 1)

ACCOUNT_SNAPSHOT_DATE = date(2026, 9, 9)
BROKERAGE_SNAPSHOT_RATE = Decimal("0.0006")
BROKERAGE_PUBLIC_RATE = Decimal("0.001")
BROKERAGE_CAP = Decimal("20")

_CRORE = Decimal("10000000")

MII_A_TRANSACTION_RATE = Decimal("297") / _CRORE
MII_A_IPFT_RATE = Decimal("10") / _CRORE
MII_B_TRANSACTION_RATE = Decimal("306.99") / _CRORE
MII_B_IPFT_RATE = Decimal("0.01") / _CRORE
MII_COMBINED_RATE = Decimal("307") / _CRORE

STT_INTRADAY_SELL_RATE = Decimal("0.00025")
STT_DELIVERY_RATE = Decimal("0.001")
STAMP_INTRADAY_BUY_RATE = Decimal("0.00003")
STAMP_DELIVERY_BUY_RATE = Decimal("0.00015")
SEBI_TURNOVER_RATE = Decimal("10") / _CRORE
GST_RATE = Decimal("0.18")

HISTORICAL_ACTUAL_LABEL = "HISTORICAL_ACTUAL_COSTS"
INCOMPLETE_LABEL = "INCOMPLETE_HISTORICAL_EVIDENCE"
SCENARIO_LABEL = "SCENARIO"
UNKNOWN_LABEL = "UNKNOWN"

SCHEMA_VERSION = "effective-dated-cost-ledger/v1"

UPSTOX_PRICING_SOURCE = "https://upstox.com/brokerage-charges/"
NSE_TRANSACTION_SOURCE = "https://nsearchives.nseindia.com/content/circulars/FA73061.pdf"
NSE_STT_SOURCE = (
    "https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies"
)
UPSTOX_BROKERAGE_URL = "https://api.upstox.com/v2/charges/brokerage"
MII_A_AUDIT_REF = (
    "audit: NSE MII cash-market schedule 2024-10-01 through 2026-02-28"
    " (transaction Rs 297/crore, IPFT Rs 10/crore)"
)


class EvidenceClass(StrEnum):
    """Machine-readable evidence quality for one cost record."""

    STATUTORY_SCHEDULE = "statutory_schedule"
    MII_SCHEDULE = "mii_schedule"
    ACCOUNT_SNAPSHOT = "account_snapshot"
    BROKER_PUBLIC_SCENARIO = "broker_public_scenario"
    SCENARIO = "scenario"
    UNKNOWN = "unknown"


class LedgerComponent(StrEnum):
    """Cost component names used by the ledger."""

    BROKERAGE = "brokerage"
    STT = "stt"
    STAMP_DUTY = "stamp_duty"
    TRANSACTION = "transaction"
    IPFT = "ipft"
    SEBI_TURNOVER = "sebi_turnover"
    GST = "gst"
    CLEARING = "clearing"
    DP_DEMAT = "dp_demat"


class LedgerProduct(StrEnum):
    """Product scope for one record. ALL matches any product query."""

    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"
    ALL = "ALL"


class LedgerSide(StrEnum):
    """Side scope for one record. BOTH matches BUY and SELL queries."""

    BUY = "BUY"
    SELL = "SELL"
    BOTH = "BOTH"


class UnsupportedResearchDate(ValueError):
    """Raised when a date precedes the supported evidence boundary."""


class UnknownCostEvidence(LookupError):
    """Raised when a component has no known rate for the requested date."""

    def __init__(self, message: str, *, unknowns: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.unknowns = unknowns


class InsufficientHistoricalCostEvidence(RuntimeError):
    """Raised when a full historical-actual cost cannot be evidenced."""

    def __init__(self, message: str, *, unknowns: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.unknowns = unknowns


def _ensure_supported(on_date: date) -> None:
    if on_date < SUPPORTED_RESEARCH_START:
        raise UnsupportedResearchDate(
            f"research boundary starts at {SUPPORTED_RESEARCH_START.isoformat()}; "
            f"no inference is made for {on_date.isoformat()}"
        )


def _date_text(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _decimal_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


@dataclass(frozen=True)
class CostEvidenceRecord:
    """One effective-dated cost fact, or one explicit unknown.

    A genuine zero charge uses ``rate=Decimal("0")``. An unknown uses
    ``rate=None`` plus a non-empty ``unknowns`` explanation. Callers must not
    conflate the two.
    """

    effective_from: date
    effective_to: date | None
    component: LedgerComponent
    product: LedgerProduct
    side: LedgerSide
    basis: str
    rate: Decimal | None
    formula: str | None
    rounding: str | None
    minimum: Decimal | None
    cap: Decimal | None
    gst_taxable: bool | None
    evidence_class: EvidenceClass
    historical_actual: bool
    confidence: str
    source_refs: tuple[str, ...]
    source_publication_date: date | None
    source_effective_date: date | None
    observed_at: date | None
    unknowns: tuple[str, ...]

    def covers(self, on_date: date) -> bool:
        """Return True when this record's effective window contains the date."""
        if on_date < self.effective_from:
            return False
        if self.effective_to is not None and on_date > self.effective_to:
            return False
        return True

    def applies_to(self, product: LedgerProduct, side: LedgerSide) -> bool:
        """Return True when product/side scope matches a query."""
        product_ok = self.product is LedgerProduct.ALL or self.product is product
        side_ok = self.side is LedgerSide.BOTH or self.side is side
        return product_ok and side_ok

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-safe mapping."""
        return {
            "basis": self.basis,
            "cap": _decimal_text(self.cap),
            "component": self.component.value,
            "confidence": self.confidence,
            "effective_from": self.effective_from.isoformat(),
            "effective_to": _date_text(self.effective_to),
            "evidence_class": self.evidence_class.value,
            "formula": self.formula,
            "gst_taxable": self.gst_taxable,
            "historical_actual": self.historical_actual,
            "minimum": _decimal_text(self.minimum),
            "observed_at": _date_text(self.observed_at),
            "product": self.product.value,
            "rate": _decimal_text(self.rate),
            "rounding": self.rounding,
            "side": self.side.value,
            "source_effective_date": _date_text(self.source_effective_date),
            "source_publication_date": _date_text(self.source_publication_date),
            "source_refs": list(self.source_refs),
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class LedgerAssessment:
    """Structured incomplete result for one date/product query."""

    on_date: date
    product: LedgerProduct
    classification: str
    historical_actual: bool
    records: tuple[CostEvidenceRecord, ...]
    unknowns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-safe mapping."""
        return {
            "classification": self.classification,
            "historical_actual": self.historical_actual,
            "on_date": self.on_date.isoformat(),
            "product": self.product.value,
            "records": [record.to_dict() for record in self.records],
            "unknowns": list(self.unknowns),
        }


def _statutory(
    *,
    component: LedgerComponent,
    product: LedgerProduct,
    side: LedgerSide,
    basis: str,
    rate: Decimal,
    formula: str,
    gst_taxable: bool | None,
    source_refs: tuple[str, ...],
    confidence: str = "high",
) -> CostEvidenceRecord:
    return CostEvidenceRecord(
        effective_from=SUPPORTED_RESEARCH_START,
        effective_to=None,
        component=component,
        product=product,
        side=side,
        basis=basis,
        rate=rate,
        formula=formula,
        rounding="none_preserve_decimal",
        minimum=None,
        cap=None,
        gst_taxable=gst_taxable,
        evidence_class=EvidenceClass.STATUTORY_SCHEDULE,
        historical_actual=True,
        confidence=confidence,
        source_refs=source_refs,
        source_publication_date=None,
        source_effective_date=SUPPORTED_RESEARCH_START,
        observed_at=None,
        unknowns=(),
    )


def build_default_records() -> tuple[CostEvidenceRecord, ...]:
    """Build the audit-supplied effective-dated record set.

    No Upstox API is called. No order is placed. Unknowns are explicit records
    with ``rate=None``; they are never populated with an assumed universal rate.
    """
    records: list[CostEvidenceRecord] = [
        # STT: intraday sell-only, delivery both sides.
        _statutory(
            component=LedgerComponent.STT,
            product=LedgerProduct.INTRADAY,
            side=LedgerSide.SELL,
            basis="turnover",
            rate=STT_INTRADAY_SELL_RATE,
            formula="turnover * 0.00025",
            gst_taxable=False,
            source_refs=(NSE_STT_SOURCE,),
        ),
        _statutory(
            component=LedgerComponent.STT,
            product=LedgerProduct.INTRADAY,
            side=LedgerSide.BUY,
            basis="turnover",
            rate=Decimal("0"),
            formula="0",
            gst_taxable=False,
            source_refs=(NSE_STT_SOURCE,),
        ),
        _statutory(
            component=LedgerComponent.STT,
            product=LedgerProduct.DELIVERY,
            side=LedgerSide.BOTH,
            basis="turnover",
            rate=STT_DELIVERY_RATE,
            formula="turnover * 0.001",
            gst_taxable=False,
            source_refs=(NSE_STT_SOURCE, UPSTOX_PRICING_SOURCE),
        ),
        # Stamp duty: buy-only, with intraday/delivery product differences.
        _statutory(
            component=LedgerComponent.STAMP_DUTY,
            product=LedgerProduct.INTRADAY,
            side=LedgerSide.BUY,
            basis="turnover",
            rate=STAMP_INTRADAY_BUY_RATE,
            formula="turnover * 0.00003",
            gst_taxable=False,
            source_refs=(UPSTOX_PRICING_SOURCE, NSE_STT_SOURCE),
        ),
        _statutory(
            component=LedgerComponent.STAMP_DUTY,
            product=LedgerProduct.INTRADAY,
            side=LedgerSide.SELL,
            basis="turnover",
            rate=Decimal("0"),
            formula="0",
            gst_taxable=False,
            source_refs=(UPSTOX_PRICING_SOURCE, NSE_STT_SOURCE),
        ),
        _statutory(
            component=LedgerComponent.STAMP_DUTY,
            product=LedgerProduct.DELIVERY,
            side=LedgerSide.BUY,
            basis="turnover",
            rate=STAMP_DELIVERY_BUY_RATE,
            formula="turnover * 0.00015",
            gst_taxable=False,
            source_refs=(UPSTOX_PRICING_SOURCE, NSE_STT_SOURCE),
        ),
        _statutory(
            component=LedgerComponent.STAMP_DUTY,
            product=LedgerProduct.DELIVERY,
            side=LedgerSide.SELL,
            basis="turnover",
            rate=Decimal("0"),
            formula="0",
            gst_taxable=False,
            source_refs=(UPSTOX_PRICING_SOURCE, NSE_STT_SOURCE),
        ),
        # SEBI turnover fee and GST rate apply across the supported boundary.
        _statutory(
            component=LedgerComponent.SEBI_TURNOVER,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="turnover",
            rate=SEBI_TURNOVER_RATE,
            formula="turnover * 10 / 10000000",
            gst_taxable=False,
            source_refs=(NSE_STT_SOURCE,),
        ),
        _statutory(
            component=LedgerComponent.GST,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="taxable_base:brokerage+transaction+ipft",
            rate=GST_RATE,
            formula="(brokerage + transaction + ipft) * 0.18",
            gst_taxable=False,
            source_refs=(UPSTOX_PRICING_SOURCE, NSE_STT_SOURCE),
        ),
        # NSE MII period A: 2024-10-01 through 2026-02-28.
        CostEvidenceRecord(
            effective_from=MII_A_START,
            effective_to=MII_A_END,
            component=LedgerComponent.TRANSACTION,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="turnover",
            rate=MII_A_TRANSACTION_RATE,
            formula="turnover * 297 / 10000000",
            rounding="none_preserve_decimal",
            minimum=None,
            cap=None,
            gst_taxable=True,
            evidence_class=EvidenceClass.MII_SCHEDULE,
            historical_actual=True,
            confidence="medium",
            source_refs=(MII_A_AUDIT_REF,),
            source_publication_date=None,
            source_effective_date=MII_A_START,
            observed_at=None,
            unknowns=(),
        ),
        CostEvidenceRecord(
            effective_from=MII_A_START,
            effective_to=MII_A_END,
            component=LedgerComponent.IPFT,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="turnover",
            rate=MII_A_IPFT_RATE,
            formula="turnover * 10 / 10000000",
            rounding="none_preserve_decimal",
            minimum=None,
            cap=None,
            gst_taxable=True,
            evidence_class=EvidenceClass.MII_SCHEDULE,
            historical_actual=True,
            confidence="medium",
            source_refs=(MII_A_AUDIT_REF,),
            source_publication_date=None,
            source_effective_date=MII_A_START,
            observed_at=None,
            unknowns=(),
        ),
        # NSE MII period B: 2026-03-01 onward.
        CostEvidenceRecord(
            effective_from=MII_B_START,
            effective_to=None,
            component=LedgerComponent.TRANSACTION,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="turnover",
            rate=MII_B_TRANSACTION_RATE,
            formula="turnover * 306.99 / 10000000",
            rounding="none_preserve_decimal",
            minimum=None,
            cap=None,
            gst_taxable=True,
            evidence_class=EvidenceClass.MII_SCHEDULE,
            historical_actual=True,
            confidence="high",
            source_refs=(NSE_TRANSACTION_SOURCE,),
            source_publication_date=None,
            source_effective_date=MII_B_START,
            observed_at=None,
            unknowns=(),
        ),
        CostEvidenceRecord(
            effective_from=MII_B_START,
            effective_to=None,
            component=LedgerComponent.IPFT,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="turnover",
            rate=MII_B_IPFT_RATE,
            formula="turnover * 0.01 / 10000000",
            rounding="none_preserve_decimal",
            minimum=None,
            cap=None,
            gst_taxable=True,
            evidence_class=EvidenceClass.MII_SCHEDULE,
            historical_actual=True,
            confidence="high",
            source_refs=(NSE_TRANSACTION_SOURCE,),
            source_publication_date=None,
            source_effective_date=MII_B_START,
            observed_at=None,
            unknowns=(),
        ),
        # Explicit MII gap: never populate Jul-Sep 2024 with the Oct rate.
        CostEvidenceRecord(
            effective_from=SUPPORTED_RESEARCH_START,
            effective_to=date(2024, 9, 30),
            component=LedgerComponent.TRANSACTION,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="unknown",
            rate=None,
            formula=None,
            rounding=None,
            minimum=None,
            cap=None,
            gst_taxable=None,
            evidence_class=EvidenceClass.UNKNOWN,
            historical_actual=False,
            confidence="none",
            source_refs=(),
            source_publication_date=None,
            source_effective_date=None,
            observed_at=None,
            unknowns=(
                "no universal MII transaction rate evidenced for 2024-07-01"
                " through 2024-09-30; do not use the 2024-10-01 rate",
            ),
        ),
        CostEvidenceRecord(
            effective_from=SUPPORTED_RESEARCH_START,
            effective_to=date(2024, 9, 30),
            component=LedgerComponent.IPFT,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="unknown",
            rate=None,
            formula=None,
            rounding=None,
            minimum=None,
            cap=None,
            gst_taxable=None,
            evidence_class=EvidenceClass.UNKNOWN,
            historical_actual=False,
            confidence="none",
            source_refs=(),
            source_publication_date=None,
            source_effective_date=None,
            observed_at=None,
            unknowns=(
                "no universal MII IPFT rate evidenced for 2024-07-01"
                " through 2024-09-30; do not use the 2024-10-01 rate",
            ),
        ),
        # Account snapshot: 0.06% observed 2026-09-09 only. Never projected back.
        CostEvidenceRecord(
            effective_from=ACCOUNT_SNAPSHOT_DATE,
            effective_to=None,
            component=LedgerComponent.BROKERAGE,
            product=LedgerProduct.INTRADAY,
            side=LedgerSide.BOTH,
            basis="turnover",
            rate=BROKERAGE_SNAPSHOT_RATE,
            formula="min(turnover * 0.0006, 20)",
            rounding="paisa_half_up",
            minimum=None,
            cap=BROKERAGE_CAP,
            gst_taxable=True,
            evidence_class=EvidenceClass.ACCOUNT_SNAPSHOT,
            historical_actual=False,
            confidence="account_specific_single_observation",
            source_refs=(UPSTOX_BROKERAGE_URL,),
            source_publication_date=None,
            source_effective_date=ACCOUNT_SNAPSHOT_DATE,
            observed_at=ACCOUNT_SNAPSHOT_DATE,
            unknowns=(
                "single-account snapshot observed 2026-09-09; "
                "historical applicability unknown; do not project backward",
            ),
        ),
        # Historical brokerage unknown: explicit gap before the snapshot.
        CostEvidenceRecord(
            effective_from=SUPPORTED_RESEARCH_START,
            effective_to=date(2026, 9, 8),
            component=LedgerComponent.BROKERAGE,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="unknown",
            rate=None,
            formula=None,
            rounding=None,
            minimum=None,
            cap=None,
            gst_taxable=None,
            evidence_class=EvidenceClass.UNKNOWN,
            historical_actual=False,
            confidence="none",
            source_refs=(),
            source_publication_date=None,
            source_effective_date=None,
            observed_at=None,
            unknowns=(
                "no historical account brokerage evidence; do not project "
                "the 2026-09-09 observed 0.06% model backward",
            ),
        ),
        # Delivery brokerage remains unknown even on/after the snapshot date:
        # the 2026-09-09 observation is intraday-only.
        CostEvidenceRecord(
            effective_from=ACCOUNT_SNAPSHOT_DATE,
            effective_to=None,
            component=LedgerComponent.BROKERAGE,
            product=LedgerProduct.DELIVERY,
            side=LedgerSide.BOTH,
            basis="unknown",
            rate=None,
            formula=None,
            rounding=None,
            minimum=None,
            cap=None,
            gst_taxable=None,
            evidence_class=EvidenceClass.UNKNOWN,
            historical_actual=False,
            confidence="none",
            source_refs=(),
            source_publication_date=None,
            source_effective_date=None,
            observed_at=None,
            unknowns=(
                "no delivery account brokerage evidence; the 2026-09-09 "
                "0.06% snapshot is intraday-only",
            ),
        ),
        # Public scenario: documented 0.1% terms are a scenario, not actual.
        CostEvidenceRecord(
            effective_from=MII_B_START,
            effective_to=None,
            component=LedgerComponent.BROKERAGE,
            product=LedgerProduct.INTRADAY,
            side=LedgerSide.BOTH,
            basis="turnover",
            rate=BROKERAGE_PUBLIC_RATE,
            formula="min(turnover * 0.001, 20)",
            rounding="none_preserve_decimal",
            minimum=None,
            cap=BROKERAGE_CAP,
            gst_taxable=True,
            evidence_class=EvidenceClass.BROKER_PUBLIC_SCENARIO,
            historical_actual=False,
            confidence="low",
            source_refs=(UPSTOX_PRICING_SOURCE,),
            source_publication_date=None,
            source_effective_date=MII_B_START,
            observed_at=None,
            unknowns=(
                "public documented terms scenario; not account-specific historical evidence",
            ),
        ),
        # Clearing remains unknown where unsupported: never zero.
        CostEvidenceRecord(
            effective_from=SUPPORTED_RESEARCH_START,
            effective_to=None,
            component=LedgerComponent.CLEARING,
            product=LedgerProduct.ALL,
            side=LedgerSide.BOTH,
            basis="unknown",
            rate=None,
            formula=None,
            rounding=None,
            minimum=None,
            cap=None,
            gst_taxable=None,
            evidence_class=EvidenceClass.UNKNOWN,
            historical_actual=False,
            confidence="none",
            source_refs=(),
            source_publication_date=None,
            source_effective_date=None,
            observed_at=None,
            unknowns=("clearing evidence unsupported; unknown must not be treated as zero",),
        ),
        # Delivery DP/demat history remains unknown: never zero.
        CostEvidenceRecord(
            effective_from=SUPPORTED_RESEARCH_START,
            effective_to=None,
            component=LedgerComponent.DP_DEMAT,
            product=LedgerProduct.DELIVERY,
            side=LedgerSide.BOTH,
            basis="unknown",
            rate=None,
            formula=None,
            rounding=None,
            minimum=None,
            cap=None,
            gst_taxable=None,
            evidence_class=EvidenceClass.UNKNOWN,
            historical_actual=False,
            confidence="none",
            source_refs=(),
            source_publication_date=None,
            source_effective_date=None,
            observed_at=None,
            unknowns=("delivery DP/demat account history unknown; do not substitute zero",),
        ),
    ]
    return tuple(records)


def _record_sort_key(record: CostEvidenceRecord) -> tuple[str, str, str, str, str]:
    return (
        record.component.value,
        record.product.value,
        record.side.value,
        record.effective_from.isoformat(),
        record.evidence_class.value,
    )


_DEFAULT_RESOLVE_CLASSES = (
    EvidenceClass.STATUTORY_SCHEDULE,
    EvidenceClass.MII_SCHEDULE,
    EvidenceClass.ACCOUNT_SNAPSHOT,
    EvidenceClass.UNKNOWN,
)


class EffectiveDatedCostLedger:
    """Provenance-first resolver over effective-dated cost evidence.

    The ledger never synthesizes a total when any required component is
    unknown. Full historical actuals therefore always fail closed with the
    current evidence because account brokerage history, clearing, and delivery
    DP history are unknown.
    """

    def __init__(self, records: tuple[CostEvidenceRecord, ...] | None = None) -> None:
        self._records = tuple(records) if records is not None else build_default_records()

    @property
    def records(self) -> tuple[CostEvidenceRecord, ...]:
        """Return ledger records in stored order."""
        return self._records

    def resolve(
        self,
        component: LedgerComponent,
        on_date: date,
        product: LedgerProduct,
        side: LedgerSide,
        *,
        evidence_classes: tuple[EvidenceClass, ...] | None = None,
    ) -> CostEvidenceRecord:
        """Resolve one component for a date/product/side.

        By default only historical-capable evidence plus explicit unknowns is
        considered. Public scenarios and hypothetical scenarios never leak into
        a default resolution; request them explicitly via ``evidence_classes``.

        Raises:
            UnsupportedResearchDate: when the date precedes 2024-07-01.
            UnknownCostEvidence: when the matched record is unknown or absent.
        """
        _ensure_supported(on_date)
        allowed = evidence_classes if evidence_classes is not None else _DEFAULT_RESOLVE_CLASSES
        candidates = [
            record
            for record in self._records
            if record.component is component
            and record.covers(on_date)
            and record.applies_to(product, side)
            and record.evidence_class in allowed
        ]
        if not candidates:
            raise UnknownCostEvidence(
                f"no {component.value} evidence for {on_date.isoformat()} "
                f"{product.value} {side.value}",
                unknowns=(f"no {component.value} record covers {on_date.isoformat()}",),
            )
        candidates.sort(key=_record_sort_key)
        # When several records overlap (for example a zero-side record and a
        # BOTH record), prefer an exact side match first for determinism.
        exact = [item for item in candidates if item.side is side]
        selected = exact[0] if exact else candidates[0]
        if selected.rate is None:
            raise UnknownCostEvidence(
                f"{component.value} is unknown for {on_date.isoformat()} "
                f"{product.value} {side.value}",
                unknowns=selected.unknowns,
            )
        return selected

    def mii_transaction_record(self, on_date: date) -> CostEvidenceRecord:
        """Return the MII transaction record for a date or fail closed."""
        _ensure_supported(on_date)
        try:
            return self.resolve(
                LedgerComponent.TRANSACTION,
                on_date,
                LedgerProduct.ALL,
                LedgerSide.BOTH,
                evidence_classes=(EvidenceClass.MII_SCHEDULE,),
            )
        except UnknownCostEvidence as exc:
            # Surface the explicit gap unknowns instead of a generic miss so
            # callers see why the 2024-10-01 rate must not be reused.
            gap_unknowns: list[str] = list(exc.unknowns)
            for record in self._records:
                if (
                    record.component is LedgerComponent.TRANSACTION
                    and record.covers(on_date)
                    and record.rate is None
                ):
                    gap_unknowns.extend(record.unknowns)
            raise UnknownCostEvidence(
                f"transaction is unknown for {on_date.isoformat()}",
                unknowns=tuple(dict.fromkeys(gap_unknowns)),
            ) from None

    def mii_ipft_record(self, on_date: date) -> CostEvidenceRecord:
        """Return the MII IPFT record for a date or fail closed."""
        _ensure_supported(on_date)
        try:
            return self.resolve(
                LedgerComponent.IPFT,
                on_date,
                LedgerProduct.ALL,
                LedgerSide.BOTH,
                evidence_classes=(EvidenceClass.MII_SCHEDULE,),
            )
        except UnknownCostEvidence as exc:
            gap_unknowns = list(exc.unknowns)
            for record in self._records:
                if (
                    record.component is LedgerComponent.IPFT
                    and record.covers(on_date)
                    and record.rate is None
                ):
                    gap_unknowns.extend(record.unknowns)
            raise UnknownCostEvidence(
                f"ipft is unknown for {on_date.isoformat()}",
                unknowns=tuple(dict.fromkeys(gap_unknowns)),
            ) from None

    def total_mii_rate(self, on_date: date) -> Decimal:
        """Return combined MII outflow per rupee of turnover.

        Both supported MII periods total Rs 307/crore. Unknown windows raise.
        """
        transaction = self.mii_transaction_record(on_date)
        ipft = self.mii_ipft_record(on_date)
        assert transaction.rate is not None and ipft.rate is not None
        return transaction.rate + ipft.rate

    def brokerage_record(self, on_date: date, product: LedgerProduct) -> CostEvidenceRecord:
        """Return account-snapshot brokerage or fail closed for history."""
        _ensure_supported(on_date)
        try:
            return self.resolve(
                LedgerComponent.BROKERAGE,
                on_date,
                product,
                LedgerSide.BOTH,
                evidence_classes=(EvidenceClass.ACCOUNT_SNAPSHOT,),
            )
        except UnknownCostEvidence as exc:
            gap_unknowns = list(exc.unknowns)
            for record in self._records:
                if (
                    record.component is LedgerComponent.BROKERAGE
                    and record.covers(on_date)
                    and (record.product is LedgerProduct.ALL or record.product is product)
                    and record.rate is None
                ):
                    gap_unknowns.extend(record.unknowns)
            raise UnknownCostEvidence(
                f"brokerage is unknown for {on_date.isoformat()} {product.value}",
                unknowns=tuple(dict.fromkeys(gap_unknowns)),
            ) from None

    def required_components(self, product: LedgerProduct) -> tuple[LedgerComponent, ...]:
        """Return components required for a full historical actual."""
        base = (
            LedgerComponent.BROKERAGE,
            LedgerComponent.STT,
            LedgerComponent.STAMP_DUTY,
            LedgerComponent.TRANSACTION,
            LedgerComponent.IPFT,
            LedgerComponent.SEBI_TURNOVER,
            LedgerComponent.GST,
            LedgerComponent.CLEARING,
        )
        if product is LedgerProduct.DELIVERY:
            return base + (LedgerComponent.DP_DEMAT,)
        return base

    def describe(self, on_date: date, product: LedgerProduct) -> LedgerAssessment:
        """Describe evidence completeness without ever faking a total."""
        _ensure_supported(on_date)
        unknowns: list[str] = []
        matched: list[CostEvidenceRecord] = []
        for component in self.required_components(product):
            covering = [
                record
                for record in self._records
                if record.component is component
                and record.covers(on_date)
                and (
                    record.product is LedgerProduct.ALL
                    or product is LedgerProduct.ALL
                    or record.product is product
                )
                and record.evidence_class in _DEFAULT_RESOLVE_CLASSES
            ]
            matched.extend(covering)
            if not covering:
                unknowns.append(
                    f"{component.value}: no record covers {on_date.isoformat()} for {product.value}"
                )
                continue
            unknown_parts = [record for record in covering if record.rate is None]
            if unknown_parts:
                for record in unknown_parts:
                    unknowns.extend(record.unknowns)
                continue
            non_actual = [record for record in covering if not record.historical_actual]
            if non_actual:
                unknowns.append(
                    f"{component.value}: {non_actual[0].evidence_class.value} "
                    "is not sufficient historical-account evidence"
                )
                for record in non_actual:
                    unknowns.extend(record.unknowns)
                continue
        # Clearing and DP unknowns are load-bearing: they keep every full
        # historical actual incomplete with the current evidence.
        matched.sort(key=_record_sort_key)
        return LedgerAssessment(
            on_date=on_date,
            product=product,
            classification=INCOMPLETE_LABEL,
            historical_actual=False,
            records=tuple(matched),
            unknowns=tuple(unknowns),
        )

    def cost_label_for(self, on_date: date, product: LedgerProduct) -> str:
        """Return a cost classification that never fakes HISTORICAL_ACTUAL_COSTS."""
        assessment = self.describe(on_date, product)
        if assessment.historical_actual and not assessment.unknowns:
            return HISTORICAL_ACTUAL_LABEL
        if assessment.classification == SCENARIO_LABEL:
            return SCENARIO_LABEL
        if not assessment.unknowns:
            return SCENARIO_LABEL
        return INCOMPLETE_LABEL

    def quote_historical_actual(self, *, on_date: date, product: LedgerProduct) -> LedgerAssessment:
        """Fail closed: full historical actuals are unsupported with current evidence.

        Raises:
            UnsupportedResearchDate: when the date precedes 2024-07-01.
            InsufficientHistoricalCostEvidence: always, with current evidence,
                because brokerage history, clearing, and (for delivery) DP
                history are unknown.
        """
        _ensure_supported(on_date)
        assessment = self.describe(on_date, product)
        raise InsufficientHistoricalCostEvidence(
            f"full {HISTORICAL_ACTUAL_LABEL} unsupported for {on_date.isoformat()} "
            f"{product.value}; unknowns remain: {'; '.join(assessment.unknowns)}",
            unknowns=assessment.unknowns,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-safe ledger snapshot."""
        ordered = sorted(self._records, key=_record_sort_key)
        return {
            "account_snapshot_date": ACCOUNT_SNAPSHOT_DATE.isoformat(),
            "historical_actual_label": HISTORICAL_ACTUAL_LABEL,
            "records": [record.to_dict() for record in ordered],
            "schema_version": SCHEMA_VERSION,
            "supported_start": SUPPORTED_RESEARCH_START.isoformat(),
        }

    def to_json(self) -> str:
        """Return canonical JSON with sorted keys and stable separators."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 over the canonical JSON snapshot."""
        payload = self.to_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
