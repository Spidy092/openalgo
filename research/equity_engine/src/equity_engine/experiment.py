"""Canonical immutable research experiment orchestration and provenance layer.

This module permanently binds research experiments to their input datasets, point-in-time
universe, exact strategy parameter grids, simulator version, cost evidence, and concrete
verification artifacts. Arbitrary booleans are strictly forbidden: promotion and integrity
decisions require verified evidence fingerprints. Volatile timestamps are metadata and do
not alter deterministic research identity. Live broker and order APIs are never called.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from .cost_ledger import (
    HISTORICAL_ACTUAL_LABEL,
    SCENARIO_LABEL,
    SUPPORTED_RESEARCH_START,
    EffectiveDatedCostLedger,
    EvidenceClass,
    LedgerComponent,
    LedgerProduct,
    LedgerSide,
    UnsupportedResearchDate,
)
from .gates import DrawdownBasis, PromotionThresholds
from .provenance import MarketDataManifest, dataframe_fingerprint

EXPERIMENT_SCHEMA_VERSION = "openalgo-equity-experiment-v2"
VECTORBT_RESEARCH_VERSION = "1.1.0"
SIMULATOR_RESEARCH_VERSION = "openalgo-event-simulator-v1"
DEFAULT_COST_EVIDENCE_POLICY = "effective-dated-cost-ledger/default-resolution/v1"


class ExperimentValidationError(ValueError):
    """Base exception for experiment provenance or integrity failure."""


class DatasetFingerprintMismatchError(ExperimentValidationError):
    """Raised when an instrument dataset fingerprint does not match verified market data."""


class UniverseFingerprintMismatchError(ExperimentValidationError):
    """Raised when the universe prefilter fingerprint does not match the canonical prefilter."""


class MissingEvidenceError(ExperimentValidationError):
    """Raised when required concrete evidence is missing (arbitrary booleans are forbidden)."""


class CorporateActionMismatchError(MissingEvidenceError):
    """Raised when a corporate-action evidence claim disagrees with trusted evidence."""


class DataLeakageError(ExperimentValidationError):
    """Raised when future/test dates or artifacts leak into train selection."""


class LiveOrderAttemptError(RuntimeError):
    """Raised if any live-order execution is requested or simulated."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _canonical_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(v) for v in value]
    if hasattr(value, "as_dict"):
        return _canonical_value(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _canonical_value(asdict(value))
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return canonical, sorted, compact JSON UTF-8 bytes for cryptographic hashing."""
    canonical_dict = {str(k): _canonical_value(v) for k, v in sorted(payload.items())}
    return json.dumps(
        canonical_dict,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Return hexadecimal SHA-256 digest of canonical JSON payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_prefilter_artifact_fingerprint(prefilter_dict: Mapping[str, Any]) -> str:
    """Return deterministic SHA-256 fingerprint for a research universe prefilter artifact."""
    if not prefilter_dict:
        raise ValueError("prefilter artifact cannot be empty")
    return canonical_sha256(prefilter_dict)


@dataclass(frozen=True)
class ResearchWindowConfig:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("research window start must be on or before end")

    def as_dict(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


@dataclass(frozen=True)
class WindowSpec:
    window_id: int
    start: date
    end: date
    trading_days: int

    def __post_init__(self) -> None:
        if self.window_id < 1:
            raise ValueError("window_id must be positive")
        if self.start > self.end:
            raise ValueError("window start must be on or before end")
        if self.trading_days <= 0:
            raise ValueError("trading_days must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "trading_days": self.trading_days,
        }


@dataclass(frozen=True)
class EmbargoSpec:
    trading_days: int

    def __post_init__(self) -> None:
        if self.trading_days < 0:
            raise ValueError("embargo trading_days cannot be negative")

    def as_dict(self) -> dict[str, int]:
        return {"trading_days": self.trading_days}


@dataclass(frozen=True)
class ApprovedCapital:
    amount_rupees: Decimal
    currency: str = "INR"

    def __post_init__(self) -> None:
        if self.amount_rupees <= Decimal(0):
            raise ValueError("approved capital must be positive")
        if not self.currency.strip():
            raise ValueError("currency is required")

    def as_dict(self) -> dict[str, str]:
        return {
            "amount_rupees": str(self.amount_rupees),
            "currency": self.currency,
        }


@dataclass(frozen=True)
class SessionPolicyIdentity:
    policy_name: str
    cas_eligible: bool
    exit_buffer_minutes: int
    cas_effective_date: str
    continuous_end: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "cas_eligible": self.cas_eligible,
            "exit_buffer_minutes": self.exit_buffer_minutes,
            "cas_effective_date": self.cas_effective_date,
            "continuous_end": self.continuous_end,
        }


@dataclass(frozen=True)
class TickEvidenceIdentity:
    policy_name: str
    source: str
    coverage_complete: bool
    coverage_fingerprint: str

    def __post_init__(self) -> None:
        if not self.coverage_complete:
            raise ValueError("tick coverage must be complete for research promotion")
        if not self.coverage_fingerprint.strip():
            raise ValueError("tick coverage fingerprint is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "source": self.source,
            "coverage_complete": self.coverage_complete,
            "coverage_fingerprint": self.coverage_fingerprint,
        }


@dataclass(frozen=True)
class NSEMembershipEvidenceIdentity:
    source_refs: tuple[str, ...]
    complete: bool
    coverage_fingerprint: str
    eligible_dates_count: int

    def __post_init__(self) -> None:
        if not self.complete:
            raise ValueError("NSE membership evidence must be complete")
        if not self.coverage_fingerprint.strip():
            raise ValueError("NSE membership coverage fingerprint is required")
        if self.eligible_dates_count < 0:
            raise ValueError("eligible_dates_count cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_refs": list(self.source_refs),
            "complete": self.complete,
            "coverage_fingerprint": self.coverage_fingerprint,
            "eligible_dates_count": self.eligible_dates_count,
        }


@dataclass(frozen=True)
class CorporateActionEvidenceIdentity:
    source: str
    complete: bool
    blocking_events: tuple[str, ...]
    evidence_fingerprint: str
    coverage_start: date | None = None
    coverage_end: date | None = None
    covered_instruments: tuple[str, ...] = ()
    events_count: int = 0
    policy_identity: str = "DEFAULT"
    authoritative: bool = False

    def __post_init__(self) -> None:
        if not self.complete:
            raise ValueError("corporate-action evidence must be complete")
        if not self.evidence_fingerprint.strip():
            raise ValueError("corporate-action evidence fingerprint is required")
        if not self.covered_instruments:
            raise ValueError("covered_instruments cannot be empty for corporate-action evidence")
        if self.coverage_start is not None and self.coverage_end is not None:
            if self.coverage_start > self.coverage_end:
                raise ValueError("coverage_start must be on or before coverage_end")

    @property
    def is_authoritative(self) -> bool:
        """Return True only if this claim asserts authoritative status.

        NOTE: This is a caller-supplied or serialized CLAIM, never proof.
        Authoritative status must always be revalidated against a trusted
        PointInTimeCorporateActionLedger at integrity or promotion time.
        """
        return self.authoritative

    def covers_window(self, start: date, end: date) -> bool:
        if self.coverage_start is None or self.coverage_end is None:
            return False
        return self.coverage_start <= start and self.coverage_end >= end

    def _derive_from_trusted_ledger(
        self,
        ledger: Any,
        *,
        research_start: date | None = None,
        research_end: date | None = None,
        canonical_instruments: Iterable[str] | None = None,
        policy: Any = None,
    ) -> CorporateActionEvidenceIdentity:
        start = research_start if research_start is not None else self.coverage_start
        end = research_end if research_end is not None else self.coverage_end
        instruments = (
            canonical_instruments
            if canonical_instruments is not None
            else self.covered_instruments
        )
        if start is None or end is None:
            raise CorporateActionMismatchError(
                "coverage start and end dates are required to validate against trusted ledger"
            )
        if not instruments:
            raise CorporateActionMismatchError(
                "canonical instruments are required to validate against trusted ledger"
            )

        try:
            expected = ledger.to_evidence_identity(
                research_start=start,
                research_end=end,
                instruments=instruments,
                policy=policy,
                source=self.source,
            )
        except Exception as exc:
            raise CorporateActionMismatchError(
                f"trusted corporate-action ledger failed revalidation: {exc}"
            ) from exc

        return expected

    def validate_against_trusted_ledger(
        self,
        ledger: Any,
        *,
        research_start: date | None = None,
        research_end: date | None = None,
        canonical_instruments: Iterable[str] | None = None,
        policy: Any = None,
    ) -> None:
        """Fail closed unless this claim exactly matches trusted ledger evidence.

        Recomputes from trusted ledger:
        - complete ledger fingerprint
        - exact research_start
        - exact research_end
        - exact instrument population
        - corporate-action policy identity
        - blocking events
        - event count
        - completeness
        - coverage
        - source/evidence mode where applicable
        """
        expected = self._derive_from_trusted_ledger(
            ledger,
            research_start=research_start,
            research_end=research_end,
            canonical_instruments=canonical_instruments,
            policy=policy,
        )

        mismatches: list[str] = []

        # 1. Complete ledger fingerprint
        if self.evidence_fingerprint != expected.evidence_fingerprint:
            mismatches.append(
                f"evidence_fingerprint (expected {expected.evidence_fingerprint!r}, "
                f"got {self.evidence_fingerprint!r})"
            )

        # 2. Exact research_start
        if self.coverage_start != expected.coverage_start:
            mismatches.append(
                f"coverage_start (expected {expected.coverage_start}, got {self.coverage_start})"
            )
        if research_start is not None and self.coverage_start != research_start:
            mismatches.append(
                f"research_start mismatch with coverage_start (expected {research_start}, got {self.coverage_start})"
            )

        # 3. Exact research_end
        if self.coverage_end != expected.coverage_end:
            mismatches.append(
                f"coverage_end (expected {expected.coverage_end}, got {self.coverage_end})"
            )
        if research_end is not None and self.coverage_end != research_end:
            mismatches.append(
                f"research_end mismatch with coverage_end (expected {research_end}, got {self.coverage_end})"
            )

        # 4. Exact instrument population
        if tuple(sorted(self.covered_instruments)) != tuple(sorted(expected.covered_instruments)):
            mismatches.append(
                f"covered_instruments (expected {expected.covered_instruments}, got {self.covered_instruments})"
            )
        if canonical_instruments is not None:
            expected_canon = tuple(sorted(canonical_instruments))
            if tuple(sorted(self.covered_instruments)) != expected_canon:
                mismatches.append(
                    f"covered_instruments vs canonical experiment instruments "
                    f"(expected {expected_canon}, got {tuple(sorted(self.covered_instruments))})"
                )

        # 5. Corporate-action policy identity
        if self.policy_identity != expected.policy_identity:
            mismatches.append(
                f"policy_identity (expected {expected.policy_identity!r}, got {self.policy_identity!r})"
            )

        # 6. Blocking events
        if self.blocking_events != expected.blocking_events:
            mismatches.append(
                f"blocking_events (expected {expected.blocking_events}, got {self.blocking_events})"
            )

        # 7. Event count
        if self.events_count != expected.events_count:
            mismatches.append(
                f"events_count (expected {expected.events_count}, got {self.events_count})"
            )

        # 8. Completeness
        if self.complete != expected.complete:
            mismatches.append(
                f"complete (expected {expected.complete}, got {self.complete})"
            )

        # 9. Coverage window
        if (
            self.coverage_start is None
            or self.coverage_end is None
            or expected.coverage_start is None
            or expected.coverage_end is None
            or not self.covers_window(expected.coverage_start, expected.coverage_end)
        ):
            mismatches.append("coverage window does not cover required window")

        # 10. Source / evidence mode
        if self.source != expected.source:
            mismatches.append(
                f"source (expected {expected.source!r}, got {self.source!r})"
            )

        if not self.authoritative:
            mismatches.append("authoritative (claim is not marked authoritative)")

        if mismatches:
            raise CorporateActionMismatchError(
                "corporate-action evidence claim does not match trusted ledger; mismatches: "
                + "; ".join(mismatches)
            )

    @classmethod
    def from_ledger(
        cls,
        ledger: Any,
        *,
        research_window: ResearchWindowConfig | Any,
        instruments: Iterable[str],
        policy: Any = None,
        source: str = "CANONICAL_PIT_CORPORATE_ACTION_LEDGER",
    ) -> CorporateActionEvidenceIdentity:
        start = (
            research_window.start
            if hasattr(research_window, "start")
            else research_window[0]
        )
        end = (
            research_window.end
            if hasattr(research_window, "end")
            else research_window[1]
        )
        return ledger.to_evidence_identity(
            research_start=start,
            research_end=end,
            instruments=instruments,
            policy=policy,
            source=source,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "complete": self.complete,
            "blocking_events": list(self.blocking_events),
            "evidence_fingerprint": self.evidence_fingerprint,
            "authoritative": self.authoritative,
        }
        if self.coverage_start is not None:
            payload["coverage_start"] = self.coverage_start.isoformat()
        if self.coverage_end is not None:
            payload["coverage_end"] = self.coverage_end.isoformat()
        if self.covered_instruments:
            payload["covered_instruments"] = list(self.covered_instruments)
        if self.events_count:
            payload["events_count"] = self.events_count
        if self.policy_identity != "DEFAULT":
            payload["policy_identity"] = self.policy_identity
        return payload


@dataclass(frozen=True)
class CostModelIdentity:
    """Scenario/configuration metadata, never authoritative cost evidence.

    ``rates`` is retained for compatibility with the original experiment
    schema. It cannot establish historical cost evidence; that role belongs
    exclusively to :class:`CostEvidenceIdentity`.
    """

    model_name: str
    effective_date: str
    rates: dict[str, str]
    source_refs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "effective_date": self.effective_date,
            "role": "scenario_configuration_only",
            "rates": dict(sorted(self.rates.items())),
            "source_refs": list(self.source_refs),
        }


def _record_identity(record: Any) -> str:
    """Return a stable identity for one ledger record selected by a policy."""

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


@dataclass(frozen=True)
class CostEvidenceIdentity:
    """Cryptographic identity of the exact ledger and policy used by research.

    This is deliberately separate from ``CostModelIdentity``. A free-form
    rates mapping can describe a scenario, but it cannot claim verified
    historical-account costs. Use ``from_ledger`` so the ledger digest and
    selected effective-dated records are created from the canonical ledger.
    """

    ledger_schema_version: str
    ledger_fingerprint: str
    evidence_classification: str
    historical_actual: bool
    product_scope: str
    evidence_mode: str
    policy_identity: str
    resolved_on_date: date
    selected_record_ids: tuple[str, ...]
    unknown_components: tuple[str, ...]
    scenario_identity: str | None = None
    _verified_ledger_fingerprint: str | None = field(default=None, repr=False, compare=False)
    _verified_historical_actual: bool | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("ledger_schema_version", self.ledger_schema_version),
            ("ledger_fingerprint", self.ledger_fingerprint),
            ("evidence_classification", self.evidence_classification),
            ("product_scope", self.product_scope),
            ("evidence_mode", self.evidence_mode),
            ("policy_identity", self.policy_identity),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")
        if len(self.ledger_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.ledger_fingerprint
        ):
            raise ValueError("ledger_fingerprint must be a hexadecimal digest")
        if self.evidence_classification == HISTORICAL_ACTUAL_LABEL:
            if not self.historical_actual:
                raise ValueError("historical actual classification requires historical_actual=True")
            if (
                self._verified_ledger_fingerprint != self.ledger_fingerprint
                or self._verified_historical_actual is not True
            ):
                raise ValueError(
                    "historical actual cost evidence must be created from verified ledger evidence"
                )
        elif self.historical_actual:
            raise ValueError(
                "historical_actual=True requires HISTORICAL_ACTUAL_COSTS classification"
            )
        if self.evidence_classification == SCENARIO_LABEL and self.historical_actual:
            raise ValueError("scenario cost evidence cannot be historical actual")
        if self.resolved_on_date < SUPPORTED_RESEARCH_START:
            raise UnsupportedResearchDate(
                "cost evidence identity cannot precede the supported research boundary"
            )

    @classmethod
    def from_ledger(
        cls,
        ledger: EffectiveDatedCostLedger,
        *,
        on_date: date,
        product: LedgerProduct,
        evidence_mode: str = "historical_resolution",
        policy_identity: str = DEFAULT_COST_EVIDENCE_POLICY,
        scenario_identity: str | None = None,
    ) -> CostEvidenceIdentity:
        """Derive identity from the ledger's date-scoped assessment."""

        assessment = ledger.describe(on_date, product)
        fingerprint = ledger.fingerprint()
        return cls(
            ledger_schema_version=str(ledger.to_dict()["schema_version"]),
            ledger_fingerprint=fingerprint,
            evidence_classification=assessment.classification,
            historical_actual=assessment.historical_actual,
            product_scope=product.value,
            evidence_mode=evidence_mode,
            policy_identity=policy_identity,
            resolved_on_date=on_date,
            selected_record_ids=tuple(
                sorted({_record_identity(record) for record in assessment.records})
            ),
            unknown_components=tuple(sorted(set(assessment.unknowns))),
            scenario_identity=scenario_identity,
            _verified_ledger_fingerprint=fingerprint,
            _verified_historical_actual=assessment.historical_actual,
        )

    @classmethod
    def from_public_scenario(
        cls,
        ledger: EffectiveDatedCostLedger,
        *,
        on_date: date,
        product: LedgerProduct,
        scenario_identity: str,
        policy_identity: str = "effective-dated-cost-ledger/public-scenario/v1",
    ) -> CostEvidenceIdentity:
        """Derive an explicitly labelled public broker scenario identity."""

        if not scenario_identity.strip():
            raise ValueError("scenario_identity is required")
        record = ledger.resolve(
            LedgerComponent.BROKERAGE,
            on_date,
            product,
            LedgerSide.BOTH,
            evidence_classes=(EvidenceClass.BROKER_PUBLIC_SCENARIO,),
        )
        fingerprint = ledger.fingerprint()
        return cls(
            ledger_schema_version=str(ledger.to_dict()["schema_version"]),
            ledger_fingerprint=fingerprint,
            evidence_classification=SCENARIO_LABEL,
            historical_actual=False,
            product_scope=product.value,
            evidence_mode="public_scenario",
            policy_identity=policy_identity,
            resolved_on_date=on_date,
            selected_record_ids=(_record_identity(record),),
            unknown_components=(
                "public broker scenario is not account-specific historical evidence",
            ),
            scenario_identity=scenario_identity,
            _verified_ledger_fingerprint=fingerprint,
            _verified_historical_actual=False,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_classification": self.evidence_classification,
            "evidence_mode": self.evidence_mode,
            "historical_actual": self.historical_actual,
            "ledger_fingerprint": self.ledger_fingerprint,
            "ledger_schema_version": self.ledger_schema_version,
            "policy_identity": self.policy_identity,
            "product_scope": self.product_scope,
            "resolved_on_date": self.resolved_on_date.isoformat(),
            "scenario_identity": self.scenario_identity,
            "selected_record_ids": list(self.selected_record_ids),
            "unknown_components": list(self.unknown_components),
        }


@dataclass(frozen=True)
class StrategySpec:
    candidate_id: str
    strategy_name: str
    research_basis: str
    source_refs: tuple[str, ...]
    parameters: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_name": self.strategy_name,
            "research_basis": self.research_basis,
            "source_refs": list(self.source_refs),
            "parameters": dict(sorted(self.parameters.items())),
        }


@dataclass(frozen=True)
class FrictionScenarioSpec:
    scenario_id: str
    slippage_bps_per_leg: Decimal
    half_spread_bps_per_leg: Decimal

    def __post_init__(self) -> None:
        if self.slippage_bps_per_leg < Decimal(0):
            raise ValueError("slippage cannot be negative")
        if self.half_spread_bps_per_leg < Decimal(0):
            raise ValueError("half_spread cannot be negative")

    def as_dict(self) -> dict[str, str]:
        return {
            "scenario_id": self.scenario_id,
            "slippage_bps_per_leg": str(self.slippage_bps_per_leg),
            "half_spread_bps_per_leg": str(self.half_spread_bps_per_leg),
        }


@dataclass(frozen=True)
class RejectedCandidateSpec:
    candidate_id: str
    instrument_key: str
    stage: str
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "instrument_key": self.instrument_key,
            "stage": self.stage,
            "reasons": list(self.reasons),
        }


# Concrete Evidence Artifact Types: arbitrary booleans are strictly forbidden.


@dataclass(frozen=True)
class HeldOutTestEvidence:
    artifact_fingerprint: str
    test_dataset_fingerprints: tuple[tuple[str, str], ...]
    window_id: int
    trade_count: int
    profit_factor: Decimal
    max_drawdown_pct: Decimal
    drawdown_basis: DrawdownBasis
    net_return_pct: Decimal
    source_reference: str

    def __post_init__(self) -> None:
        if not self.artifact_fingerprint.strip():
            raise ValueError("held-out test artifact fingerprint is required")
        if self.trade_count <= 0:
            raise ValueError("held-out test trade_count must be positive")
        if self.profit_factor <= Decimal(0):
            raise ValueError("held-out test profit_factor must be positive")
        if self.max_drawdown_pct < Decimal(0):
            raise ValueError("held-out test max_drawdown_pct cannot be negative")
        if not self.source_reference.strip():
            raise ValueError("held-out test source reference is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_fingerprint": self.artifact_fingerprint,
            "test_dataset_fingerprints": [
                {"instrument_key": key, "dataset_fingerprint": fp}
                for key, fp in sorted(self.test_dataset_fingerprints)
            ],
            "window_id": self.window_id,
            "trade_count": self.trade_count,
            "profit_factor": str(self.profit_factor),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "drawdown_basis": self.drawdown_basis.value,
            "net_return_pct": str(self.net_return_pct),
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True)
class CostReconciliationEvidence:
    artifact_fingerprint: str
    schema_version: str
    cost_model_name: str
    orders_checked: int
    passed_count: int
    failed_count: int
    max_reconciliation_error_inr: Decimal
    tolerance_inr: Decimal
    status: str

    def __post_init__(self) -> None:
        if not self.artifact_fingerprint.strip():
            raise ValueError("cost reconciliation artifact fingerprint is required")
        if self.orders_checked <= 0:
            raise ValueError("orders_checked must be positive")
        if self.max_reconciliation_error_inr < Decimal(0):
            raise ValueError("max_reconciliation_error_inr cannot be negative")
        if self.status != "PASS":
            raise ValueError(f"cost reconciliation status must be PASS, got {self.status!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_fingerprint": self.artifact_fingerprint,
            "schema_version": self.schema_version,
            "cost_model_name": self.cost_model_name,
            "orders_checked": self.orders_checked,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "max_reconciliation_error_inr": str(self.max_reconciliation_error_inr),
            "tolerance_inr": str(self.tolerance_inr),
            "status": self.status,
        }


@dataclass(frozen=True)
class PaperTradingEvidence:
    artifact_fingerprint: str
    environment: str
    session_start: date
    session_end: date
    verified_orders_count: int
    audit_log_fingerprint: str
    source_reference: str

    def __post_init__(self) -> None:
        if not self.artifact_fingerprint.strip():
            raise ValueError("paper-trading artifact fingerprint is required")
        if not self.audit_log_fingerprint.strip():
            raise ValueError("paper-trading audit log fingerprint is required")
        if self.verified_orders_count <= 0:
            raise ValueError("verified_orders_count must be positive")
        if self.session_start > self.session_end:
            raise ValueError("session_start must be on or before session_end")
        if not self.environment.strip():
            raise ValueError("paper-trading environment is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_fingerprint": self.artifact_fingerprint,
            "environment": self.environment,
            "session_start": self.session_start.isoformat(),
            "session_end": self.session_end.isoformat(),
            "verified_orders_count": self.verified_orders_count,
            "audit_log_fingerprint": self.audit_log_fingerprint,
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True)
class BaselineComparisonEvidence:
    artifact_fingerprint: str
    baseline_candidate_id: str
    evaluated_candidate_id: str
    baseline_net_return_pct: Decimal
    evaluated_net_return_pct: Decimal
    outperformed: bool

    def __post_init__(self) -> None:
        if not self.artifact_fingerprint.strip():
            raise ValueError("baseline comparison artifact fingerprint is required")
        if not self.baseline_candidate_id.strip():
            raise ValueError("baseline_candidate_id is required")
        if not self.evaluated_candidate_id.strip():
            raise ValueError("evaluated_candidate_id is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_fingerprint": self.artifact_fingerprint,
            "baseline_candidate_id": self.baseline_candidate_id,
            "evaluated_candidate_id": self.evaluated_candidate_id,
            "baseline_net_return_pct": str(self.baseline_net_return_pct),
            "evaluated_net_return_pct": str(self.evaluated_net_return_pct),
            "outperformed": self.outperformed,
        }


@dataclass(frozen=True)
class SlippageStressEvidence:
    artifact_fingerprint: str
    scenarios_evaluated: tuple[str, ...]
    stress_max_drawdown_pct: Decimal
    stress_passed: bool

    def __post_init__(self) -> None:
        if not self.artifact_fingerprint.strip():
            raise ValueError("slippage stress artifact fingerprint is required")
        if not self.scenarios_evaluated:
            raise ValueError("at least one stress scenario must be evaluated")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_fingerprint": self.artifact_fingerprint,
            "scenarios_evaluated": list(self.scenarios_evaluated),
            "stress_max_drawdown_pct": str(self.stress_max_drawdown_pct),
            "stress_passed": self.stress_passed,
        }


@dataclass(frozen=True)
class EventDrivenSimulationEvidence:
    artifact_fingerprint: str
    simulator_version: str
    trade_count: int
    initial_cash: Decimal
    final_cash: Decimal

    def __post_init__(self) -> None:
        if not self.artifact_fingerprint.strip():
            raise ValueError("event simulation artifact fingerprint is required")
        if not self.simulator_version.strip():
            raise ValueError("simulator_version is required")
        if self.trade_count <= 0:
            raise ValueError("simulation trade_count must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_fingerprint": self.artifact_fingerprint,
            "simulator_version": self.simulator_version,
            "trade_count": self.trade_count,
            "initial_cash": str(self.initial_cash),
            "final_cash": str(self.final_cash),
        }


@dataclass(frozen=True)
class ConcretePromotionEvidence:
    """Rigorous gate decision grounded strictly in concrete evidence artifacts.

    Arbitrary booleans like ``paper_trading_present=True`` are strictly forbidden:
    every claim must link to a verified artifact fingerprint.
    """

    held_out_test: HeldOutTestEvidence | None
    cost_reconciliation: CostReconciliationEvidence | None
    paper_trading: PaperTradingEvidence | None
    baseline_comparison: BaselineComparisonEvidence | None
    slippage_stress: SlippageStressEvidence | None
    event_simulation: EventDrivenSimulationEvidence | None
    unpriced_cost_components: tuple[str, ...] = ()

    def evaluate_gate(self, thresholds: PromotionThresholds) -> tuple[bool, tuple[str, ...]]:
        violations: list[str] = []

        if self.held_out_test is None:
            violations.append("held-out test evidence artifact is missing")
        else:
            if (
                self.held_out_test.drawdown_basis
                is not DrawdownBasis.OHLC_LOW_LIQUIDATION_STRESS
            ):
                violations.append(
                    "promotion drawdown must use OHLC-low liquidation stress; "
                    f"received {self.held_out_test.drawdown_basis.value}"
                )
            if self.held_out_test.trade_count < thresholds.min_trades:
                violations.append(
                    f"trade count {self.held_out_test.trade_count} is below required "
                    f"{thresholds.min_trades}"
                )
            if self.held_out_test.profit_factor < thresholds.min_profit_factor:
                violations.append(
                    f"profit factor {self.held_out_test.profit_factor} is below required "
                    f"{thresholds.min_profit_factor}"
                )
            if self.held_out_test.max_drawdown_pct > thresholds.max_drawdown_pct:
                violations.append(
                    f"max drawdown {self.held_out_test.max_drawdown_pct}% exceeds allowed "
                    f"{thresholds.max_drawdown_pct}%"
                )

        if self.cost_reconciliation is None:
            violations.append("broker cost reconciliation evidence artifact is missing")
        else:
            if (
                self.cost_reconciliation.max_reconciliation_error_inr
                > thresholds.max_cost_reconciliation_error_inr
            ):
                violations.append(
                    f"cost reconciliation error ₹{self.cost_reconciliation.max_reconciliation_error_inr} "
                    f"exceeds ₹{thresholds.max_cost_reconciliation_error_inr}"
                )

        if self.paper_trading is None:
            violations.append("paper-trading evidence artifact is missing")

        if self.baseline_comparison is None:
            violations.append("baseline comparison evidence artifact is missing")
        elif not self.baseline_comparison.outperformed:
            violations.append("candidate did not outperform baseline same-cost definition")

        if self.slippage_stress is None:
            violations.append("slippage/spread stress test evidence artifact is missing")
        elif not self.slippage_stress.stress_passed:
            violations.append("slippage/spread stress test failed acceptable thresholds")

        if self.event_simulation is None:
            violations.append("event-driven simulation evidence artifact is missing")

        if self.unpriced_cost_components:
            violations.append(
                "unpriced cost components remain: " + ", ".join(self.unpriced_cost_components)
            )

        return (len(violations) == 0, tuple(violations))

    def as_dict(self) -> dict[str, Any]:
        return {
            "held_out_test": self.held_out_test.as_dict() if self.held_out_test else None,
            "cost_reconciliation": (
                self.cost_reconciliation.as_dict() if self.cost_reconciliation else None
            ),
            "paper_trading": self.paper_trading.as_dict() if self.paper_trading else None,
            "baseline_comparison": (
                self.baseline_comparison.as_dict() if self.baseline_comparison else None
            ),
            "slippage_stress": (
                self.slippage_stress.as_dict() if self.slippage_stress else None
            ),
            "event_simulation": (
                self.event_simulation.as_dict() if self.event_simulation else None
            ),
            "unpriced_cost_components": list(self.unpriced_cost_components),
        }


@dataclass(frozen=True)
class ExperimentArtifact:
    """Canonical immutable experiment record tying research together.

    Permanently identifies research window, dataset fingerprints, point-in-time universe,
    strategy parameters, simulator version, cost evidence, walk-forward windows, friction
    scenarios, concrete promotion evidence, and code commit SHA.

    Determinism Guarantee:
    Given identical data, config, code identity, strategies, cost model, and window definitions,
    the deterministic fingerprint and experiment identity are strictly identical. Volatile
    generation timestamps (``created_at``) are recorded in metadata but excluded from research
    identity.
    """

    schema_version: str
    created_at: str
    research_window: ResearchWindowConfig
    train_windows: tuple[WindowSpec, ...]
    validation_test_windows: tuple[WindowSpec, ...]
    embargo: EmbargoSpec
    approved_capital: ApprovedCapital
    universe_fingerprint: str
    candidate_prefilter_artifact_fingerprint: str
    instrument_dataset_fingerprints: dict[str, str]
    nse_membership_evidence: NSEMembershipEvidenceIdentity
    tick_evidence: TickEvidenceIdentity
    session_policy_identity: SessionPolicyIdentity
    corporate_action_evidence: CorporateActionEvidenceIdentity
    cost_model_identity: CostModelIdentity
    cost_evidence_identity: CostEvidenceIdentity
    cost_evidence_class: str
    strategy_definitions: tuple[StrategySpec, ...]
    parameter_grid: dict[str, tuple[str, ...]]
    vectorbt_version: str
    simulator_version: str
    random_seeds: dict[str, int] | None
    friction_scenarios: tuple[FrictionScenarioSpec, ...]
    rejected_candidates: tuple[RejectedCandidateSpec, ...]
    tournament_result: dict[str, Any]
    walk_forward_result: dict[str, Any]
    promotion_evidence: ConcretePromotionEvidence
    code_commit_sha: str
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.live_orders_called:
            raise LiveOrderAttemptError("live orders are strictly forbidden in research")
        if not self.code_commit_sha.strip():
            raise ValueError("code_commit_sha is required")
        if not self.universe_fingerprint.strip():
            raise ValueError("universe_fingerprint is required")
        if not self.candidate_prefilter_artifact_fingerprint.strip():
            raise ValueError("candidate_prefilter_artifact_fingerprint is required")
        if not self.instrument_dataset_fingerprints:
            raise ValueError("instrument_dataset_fingerprints cannot be empty")
        if not self.strategy_definitions:
            raise ValueError("strategy_definitions cannot be empty")
        if not self.train_windows:
            raise ValueError("train_windows cannot be empty")
        if not self.validation_test_windows:
            raise ValueError("validation_test_windows cannot be empty")
        if self.cost_evidence_class != self.cost_evidence_identity.evidence_classification:
            raise ValueError(
                "cost_evidence_class must match the ledger-derived cost evidence classification"
            )
        if self.cost_evidence_identity.resolved_on_date > self.research_window.end:
            raise DataLeakageError(
                "cost evidence date "
                f"{self.cost_evidence_identity.resolved_on_date} is after research window end "
                f"{self.research_window.end}"
            )

    def deterministic_payload(self) -> dict[str, Any]:
        """Return canonical dictionary of all deterministic research identity fields.

        Volatile metadata (e.g. ``created_at``) is strictly excluded here to preserve
        repeatable research identity across machines and execution runs.
        """
        return {
            "schema_version": self.schema_version,
            "research_window": self.research_window.as_dict(),
            "train_windows": [w.as_dict() for w in self.train_windows],
            "validation_test_windows": [w.as_dict() for w in self.validation_test_windows],
            "embargo": self.embargo.as_dict(),
            "approved_capital": self.approved_capital.as_dict(),
            "universe_fingerprint": self.universe_fingerprint,
            "candidate_prefilter_artifact_fingerprint": (
                self.candidate_prefilter_artifact_fingerprint
            ),
            "instrument_dataset_fingerprints": dict(
                sorted(self.instrument_dataset_fingerprints.items())
            ),
            "nse_membership_evidence": self.nse_membership_evidence.as_dict(),
            "tick_evidence": self.tick_evidence.as_dict(),
            "session_policy_identity": self.session_policy_identity.as_dict(),
            "corporate_action_evidence": self.corporate_action_evidence.as_dict(),
            "cost_model_identity": self.cost_model_identity.as_dict(),
            "cost_evidence_identity": self.cost_evidence_identity.as_dict(),
            "cost_evidence_class": self.cost_evidence_class,
            "strategy_definitions": [s.as_dict() for s in self.strategy_definitions],
            "parameter_grid": {
                k: list(v) for k, v in sorted(self.parameter_grid.items())
            },
            "vectorbt_version": self.vectorbt_version,
            "simulator_version": self.simulator_version,
            "random_seeds": (
                dict(sorted(self.random_seeds.items()))
                if self.random_seeds is not None
                else None
            ),
            "friction_scenarios": [s.as_dict() for s in self.friction_scenarios],
            "rejected_candidates": [r.as_dict() for r in self.rejected_candidates],
            "tournament_result": _canonical_value(self.tournament_result),
            "walk_forward_result": _canonical_value(self.walk_forward_result),
            "promotion_evidence": self.promotion_evidence.as_dict(),
            "code_commit_sha": self.code_commit_sha,
            "live_orders_called": False,
        }

    def deterministic_fingerprint(self) -> str:
        """SHA-256 fingerprint of canonical deterministic research identity."""
        return canonical_sha256(self.deterministic_payload())

    @property
    def experiment_id(self) -> str:
        """Canonical immutable experiment identity derived from deterministic fingerprint."""
        return f"exp_{self.deterministic_fingerprint()[:16]}"

    def as_dict(self) -> dict[str, Any]:
        """Full representation including volatile metadata and deterministic identity."""
        payload = self.deterministic_payload()
        payload["experiment_id"] = self.experiment_id
        payload["deterministic_fingerprint"] = self.deterministic_fingerprint()
        payload["created_at"] = self.created_at
        return payload

    def as_json(self, *, indent: int = 2) -> str:
        """JSON-serialized experiment artifact."""
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True) + "\n"

    def evaluate_promotion_gate(
        self,
        thresholds: PromotionThresholds,
        *,
        trusted_corporate_action_ledger: Any = None,
        trusted_corporate_action_policy: Any = None,
        corporate_action_ledger: Any = None,
        corporate_action_policy: Any = None,
        **kwargs: Any,
    ) -> tuple[bool, tuple[str, ...]]:
        """Evaluate promotion evidence after revalidating corporate-action evidence.

        A promotion decision always requires ``trusted_corporate_action_ledger``. The
        serialized ``CorporateActionEvidenceIdentity`` on the artifact is only a claim;
        it cannot establish authoritative corporate-action evidence by itself.
        """

        passed, violations = self.promotion_evidence.evaluate_gate(thresholds)
        gate_violations = list(violations)
        if (
            self.cost_evidence_identity.evidence_classification != HISTORICAL_ACTUAL_LABEL
            or not self.cost_evidence_identity.historical_actual
        ):
            gate_violations.append(
                "cost evidence is not verified HISTORICAL_ACTUAL_COSTS; "
                "scenario/incomplete evidence cannot promote"
            )

        ca_ledger = (
            trusted_corporate_action_ledger
            if trusted_corporate_action_ledger is not None
            else corporate_action_ledger
        )
        ca_policy = (
            trusted_corporate_action_policy
            if trusted_corporate_action_policy is not None
            else corporate_action_policy
        )

        # Corporate-action trusted ledger revalidation at promotion boundary
        if ca_ledger is None:
            gate_violations.append(
                "trusted corporate-action evidence ledger is required for promotion; "
                "serialized corporate-action evidence is a claim, not proof"
            )
        else:
            try:
                self.corporate_action_evidence.validate_against_trusted_ledger(
                    ca_ledger,
                    research_start=self.research_window.start,
                    research_end=self.research_window.end,
                    canonical_instruments=sorted(self.instrument_dataset_fingerprints.keys()),
                    policy=ca_policy,
                )
            except (CorporateActionMismatchError, LookupError, ValueError) as exc:
                gate_violations.append(f"corporate-action evidence integrity mismatch: {exc}")

        if not self.corporate_action_evidence.authoritative:
            gate_violations.append(
                "corporate-action evidence is non-authoritative claim; cannot promote without trusted ledger revalidation"
            )
        if not self.corporate_action_evidence.complete:
            gate_violations.append("corporate-action evidence is incomplete; cannot promote")
        if self.corporate_action_evidence.blocking_events:
            gate_violations.append(
                "corporate-action evidence has unresolved blocking events: "
                + ", ".join(self.corporate_action_evidence.blocking_events)
            )
        if set(self.corporate_action_evidence.covered_instruments) != set(
            self.instrument_dataset_fingerprints.keys()
        ):
            gate_violations.append(
                "corporate-action evidence does not cover exact experiment instrument population"
            )

        return (passed and not gate_violations, tuple(gate_violations))

    def validate_integrity(
        self,
        *,
        dataset_frames: Mapping[str, pd.DataFrame] | None = None,
        dataset_manifests: Mapping[str, MarketDataManifest] | None = None,
        prefilter_artifact: Mapping[str, Any] | None = None,
        promotion_thresholds: PromotionThresholds | None = None,
        corporate_action_ledger: Any = None,
        corporate_action_policy: Any = None,
        trusted_corporate_action_ledger: Any = None,
        trusted_corporate_action_policy: Any = None,
        **kwargs: Any,
    ) -> None:
        """Enforce strict fail-closed validation of all fingerprints and evidence.

        Raises:
            DatasetFingerprintMismatchError: If an instrument dataset fingerprint does not match.
            UniverseFingerprintMismatchError: If the universe prefilter fingerprint does not match.
            MissingEvidenceError: If concrete evidence is missing or invalid.
            DataLeakageError: If test/validation windows overlap or contaminate train.
            LiveOrderAttemptError: If live order execution was marked true or invoked.
        """
        if self.live_orders_called:
            raise LiveOrderAttemptError("live-order execution is strictly forbidden in research")

        # 1. Dataset fingerprint verification
        if dataset_frames is not None:
            if dataset_manifests is None:
                raise ValueError("dataset_manifests required when dataset_frames provided")
            for key, frame in dataset_frames.items():
                if key not in self.instrument_dataset_fingerprints:
                    raise DatasetFingerprintMismatchError(
                        f"instrument {key} not found in experiment dataset fingerprints"
                    )
                expected_fp = self.instrument_dataset_fingerprints[key]
                manifest = dataset_manifests[key]
                actual_fp = dataframe_fingerprint(frame, manifest)
                if actual_fp != expected_fp:
                    raise DatasetFingerprintMismatchError(
                        f"dataset fingerprint mismatch for {key}: expected {expected_fp}, got {actual_fp}"
                    )

        # 2. Universe prefilter fingerprint verification
        if prefilter_artifact is not None:
            actual_prefilter_fp = compute_prefilter_artifact_fingerprint(prefilter_artifact)
            if actual_prefilter_fp != self.candidate_prefilter_artifact_fingerprint:
                raise UniverseFingerprintMismatchError(
                    f"universe prefilter fingerprint mismatch: expected "
                    f"{self.candidate_prefilter_artifact_fingerprint}, got {actual_prefilter_fp}"
                )
            if self.universe_fingerprint != actual_prefilter_fp:
                raise UniverseFingerprintMismatchError(
                    f"universe fingerprint {self.universe_fingerprint} does not match "
                    f"prefilter artifact fingerprint {actual_prefilter_fp}"
                )

        # 3. Train/Test date leakage verification
        for train_w in self.train_windows:
            for test_w in self.validation_test_windows:
                if train_w.window_id == test_w.window_id and train_w.end >= test_w.start:
                    raise DataLeakageError(
                        f"window {train_w.window_id} train end {train_w.end} must be strictly "
                        f"before test start {test_w.start}"
                    )

        # 4. Fail-closed concrete evidence checks
        if self.promotion_evidence.held_out_test is None:
            raise MissingEvidenceError(
                "held-out test evidence artifact is missing; arbitrary booleans are forbidden"
            )
        if self.promotion_evidence.cost_reconciliation is None:
            raise MissingEvidenceError(
                "broker cost reconciliation evidence artifact is missing"
            )
        if self.promotion_evidence.paper_trading is None:
            raise MissingEvidenceError(
                "paper-trading evidence artifact is missing"
            )
        if self.promotion_evidence.baseline_comparison is None:
            raise MissingEvidenceError(
                "baseline comparison evidence artifact is missing"
            )
        if self.promotion_evidence.slippage_stress is None:
            raise MissingEvidenceError(
                "slippage/spread stress evidence artifact is missing"
            )
        if self.promotion_evidence.event_simulation is None:
            raise MissingEvidenceError(
                "event-driven simulation evidence artifact is missing"
            )

        # 5. Fail-closed corporate-action evidence verification
        if not self.corporate_action_evidence.complete:
            raise MissingEvidenceError("corporate-action evidence is incomplete")
        if (
            self.corporate_action_evidence.coverage_start is None
            or self.corporate_action_evidence.coverage_end is None
            or not self.corporate_action_evidence.covers_window(
                self.research_window.start, self.research_window.end
            )
        ):
            cov_start = (
                self.corporate_action_evidence.coverage_start.isoformat()
                if self.corporate_action_evidence.coverage_start
                else "None"
            )
            cov_end = (
                self.corporate_action_evidence.coverage_end.isoformat()
                if self.corporate_action_evidence.coverage_end
                else "None"
            )
            raise MissingEvidenceError(
                "experiment cannot claim corporate-action-complete unless evidence covers "
                f"exact research window [{self.research_window.start.isoformat()}, {self.research_window.end.isoformat()}]; "
                f"got coverage [{cov_start}, {cov_end}]"
            )
        if self.corporate_action_evidence.blocking_events:
            raise MissingEvidenceError(
                "experiment has unresolved blocking corporate actions: "
                + ", ".join(self.corporate_action_evidence.blocking_events)
            )

        # Exact instrument coverage
        canonical_experiment_instruments = set(self.instrument_dataset_fingerprints.keys())
        ca_covered_instruments = set(self.corporate_action_evidence.covered_instruments)
        if ca_covered_instruments != canonical_experiment_instruments:
            missing_ca = canonical_experiment_instruments - ca_covered_instruments
            extra_ca = ca_covered_instruments - canonical_experiment_instruments
            raise MissingEvidenceError(
                f"corporate-action evidence population does not exactly match canonical experiment "
                f"instrument population: missing={sorted(missing_ca)}, extra={sorted(extra_ca)}"
            )

        ca_ledger = (
            trusted_corporate_action_ledger
            if trusted_corporate_action_ledger is not None
            else corporate_action_ledger
        )
        ca_policy = (
            trusted_corporate_action_policy
            if trusted_corporate_action_policy is not None
            else corporate_action_policy
        )

        # Revalidate against trusted PointInTimeCorporateActionLedger
        if ca_ledger is not None:
            try:
                self.corporate_action_evidence.validate_against_trusted_ledger(
                    ca_ledger,
                    research_start=self.research_window.start,
                    research_end=self.research_window.end,
                    canonical_instruments=sorted(canonical_experiment_instruments),
                    policy=ca_policy,
                )
            except (CorporateActionMismatchError, LookupError, ValueError) as exc:
                raise MissingEvidenceError(
                    f"corporate-action evidence integrity mismatch: {exc}"
                ) from exc
        elif self.corporate_action_evidence.authoritative:
            raise MissingEvidenceError(
                "trusted corporate-action evidence ledger is required to validate authoritative "
                "corporate-action evidence; serialized corporate-action evidence is a claim, not proof"
            )

        # 6. Threshold evaluation if requested
        if promotion_thresholds is not None:
            passed, violations = self.evaluate_promotion_gate(
                promotion_thresholds,
                trusted_corporate_action_ledger=ca_ledger,
                trusted_corporate_action_policy=ca_policy,
                **kwargs,
            )
            if not passed:
                raise MissingEvidenceError(
                    "experiment failed promotion gate criteria: " + "; ".join(violations)
                )


class ExperimentOrchestrator:
    """Orchestrates creation and deterministic binding of canonical experiment artifacts.

    Guarantees:
    - Never calls live broker or order APIs.
    - Deterministic research identity is independent of generation timestamp.
    - Full fail-closed validation on missing evidence or fingerprint mismatch.
    """

    def __init__(
        self,
        *,
        code_commit_sha: str,
        vectorbt_version: str = VECTORBT_RESEARCH_VERSION,
        simulator_version: str = SIMULATOR_RESEARCH_VERSION,
    ) -> None:
        if not code_commit_sha.strip():
            raise ValueError("code_commit_sha is required")
        self._code_commit_sha = code_commit_sha
        self._vectorbt_version = vectorbt_version
        self._simulator_version = simulator_version

    @property
    def code_commit_sha(self) -> str:
        return self._code_commit_sha

    @property
    def live_orders_called(self) -> bool:
        return False

    def build_experiment(
        self,
        *,
        research_window: ResearchWindowConfig,
        train_windows: tuple[WindowSpec, ...],
        validation_test_windows: tuple[WindowSpec, ...],
        embargo: EmbargoSpec,
        approved_capital: ApprovedCapital,
        universe_fingerprint: str,
        candidate_prefilter_artifact_fingerprint: str,
        instrument_dataset_fingerprints: dict[str, str],
        nse_membership_evidence: NSEMembershipEvidenceIdentity,
        tick_evidence: TickEvidenceIdentity,
        session_policy_identity: SessionPolicyIdentity,
        corporate_action_evidence: CorporateActionEvidenceIdentity,
        cost_model_identity: CostModelIdentity,
        cost_evidence_identity: CostEvidenceIdentity,
        cost_evidence_class: str,
        strategy_definitions: tuple[StrategySpec, ...],
        parameter_grid: dict[str, tuple[str, ...]],
        friction_scenarios: tuple[FrictionScenarioSpec, ...],
        rejected_candidates: tuple[RejectedCandidateSpec, ...],
        tournament_result: dict[str, Any],
        walk_forward_result: dict[str, Any],
        promotion_evidence: ConcretePromotionEvidence,
        random_seeds: dict[str, int] | None = None,
        created_at: str | None = None,
    ) -> ExperimentArtifact:
        """Create and return an immutable ExperimentArtifact with strict fail-closed validation."""
        timestamp = (
            created_at
            if created_at is not None
            else datetime.now(UTC).isoformat()
        )

        artifact = ExperimentArtifact(
            schema_version=EXPERIMENT_SCHEMA_VERSION,
            created_at=timestamp,
            research_window=research_window,
            train_windows=train_windows,
            validation_test_windows=validation_test_windows,
            embargo=embargo,
            approved_capital=approved_capital,
            universe_fingerprint=universe_fingerprint,
            candidate_prefilter_artifact_fingerprint=candidate_prefilter_artifact_fingerprint,
            instrument_dataset_fingerprints=instrument_dataset_fingerprints,
            nse_membership_evidence=nse_membership_evidence,
            tick_evidence=tick_evidence,
            session_policy_identity=session_policy_identity,
            corporate_action_evidence=corporate_action_evidence,
            cost_model_identity=cost_model_identity,
            cost_evidence_identity=cost_evidence_identity,
            cost_evidence_class=cost_evidence_class,
            strategy_definitions=strategy_definitions,
            parameter_grid=parameter_grid,
            vectorbt_version=self._vectorbt_version,
            simulator_version=self._simulator_version,
            random_seeds=random_seeds,
            friction_scenarios=friction_scenarios,
            rejected_candidates=rejected_candidates,
            tournament_result=tournament_result,
            walk_forward_result=walk_forward_result,
            promotion_evidence=promotion_evidence,
            code_commit_sha=self._code_commit_sha,
            live_orders_called=False,
        )

        # Immediate fail-closed validation of window leakage
        for tw in train_windows:
            for vtw in validation_test_windows:
                if tw.window_id == vtw.window_id and tw.end >= vtw.start:
                    raise DataLeakageError(
                        f"train window {tw.window_id} end {tw.end} overlaps or touches "
                        f"test start {vtw.start}"
                    )

        return artifact
