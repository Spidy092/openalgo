"""Staged, point-in-time historical acquisition planning.

This module plans the acquisition boundary; it does not contact NSE or Upstox.  The existing
``NseBatchUniverseBuilder`` and ``UpstoxHistoricalBatchDownloader`` perform the actual resumable
I/O when an operator explicitly starts a later run.  Stage A is daily/coarse data plus dated NSE
membership.  Stage B is 5-minute data only for a completed Stage-A prefilter.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from .costs import CostProvider
from .historical_membership import HistoricalTradingStatus, assess_historical_membership
from .liquidity import summarize_historical_liquidity
from .models import Exchange, Product
from .nse_semantics import NSE_MASTER_DATA_V15_EFFECTIVE_EVIDENCE_DATE
from .provenance import validate_ohlcv_frame
from .sizing import max_affordable_buy_quantity
from .tick_size import TickPolicy, assess_tick_policy_coverage
from .universe import (
    CorporateActionAssessment,
    ResearchUniverseThresholds,
    evaluate_research_universe_candidate,
)
from .upstox_batch_history import (
    HistoricalBatchCandidate,
    historical_request_count,
    plan_historical_batch,
)
from .upstox_history import UPSTOX_DAILY_HISTORY_START

PIT_ACQUISITION_SCHEMA = "openalgo-pit-historical-acquisition/v1"
STAGE_A_SCHEMA = "openalgo-pit-historical-acquisition/stage-a/v1"
STAGE_B_SCHEMA = "openalgo-pit-historical-acquisition/stage-b/v1"
RAW_ACQUISITION_ONLY = "RAW_ACQUISITION_ONLY"
SUPPORTED_RESEARCH_START = NSE_MASTER_DATA_V15_EFFECTIVE_EVIDENCE_DATE
SUPPORTED_PRICE_DATA_START = UPSTOX_DAILY_HISTORY_START


class PITAcquisitionError(ValueError):
    """Base error for incomplete or inconsistent acquisition evidence."""


class IncompleteUniverseManifestError(PITAcquisitionError):
    """Raised when a source NSE universe manifest is not complete and auditable."""


class IncompletePrefilterError(PITAcquisitionError):
    """Raised when Stage B is attempted without a complete Stage-A prefilter."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if hasattr(value, "as_dict"):
        return _canonical_value(value.as_dict())
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _canonical_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _decimal_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _require_content_fingerprint(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PITAcquisitionError(f"{name} is required")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PITAcquisitionError(f"{name} must be a 64-character lowercase hex digest")
    return value


def _thresholds_dict(thresholds: ResearchUniverseThresholds) -> dict[str, object]:
    return {
        "max_last_price_rupees": _decimal_text(thresholds.max_last_price_rupees),
        "min_affordable_quantity": thresholds.min_affordable_quantity,
        "min_median_daily_notional_proxy_rupees": _decimal_text(
            thresholds.min_median_daily_notional_proxy_rupees
        ),
        "min_median_daily_volume_shares": _decimal_text(thresholds.min_median_daily_volume_shares),
        "min_observed_trading_days": thresholds.min_observed_trading_days,
    }


def prior_completed_trading_session(
    selection_cutoff: date, trading_dates: tuple[date, ...]
) -> date:
    """Return the exact previous sourced normal trading session strictly before cutoff.

    Never infers from ``timestamp <= midnight`` alone. Daily candles timestamped
    at midnight for date T contain T's completed close, so T must never supply a
    decision made on T. Monday resolves to Friday (or the previous actual trading
    session); a holiday-following session resolves to the previous actual trading
    session. Fails closed when no prior session exists (for example a new listing
    whose first eligible date is the cutoff itself).
    """
    prior: date | None = None
    for trading_date in trading_dates:
        if trading_date < selection_cutoff and (prior is None or trading_date > prior):
            prior = trading_date
    if prior is None:
        raise PITAcquisitionError(
            f"no prior completed trading session exists before {selection_cutoff.isoformat()}; "
            "reference price is unknown and must not use the current day close"
        )
    return prior


def _prior_session_close(
    frame: pd.DataFrame, prior_session: date, *, instrument_key: str
) -> tuple[Decimal, pd.Timestamp]:
    """Return the close of the exact prior session date, failing closed if absent.

    Selects rows by trading ``date == prior_session`` only. A daily candle
    timestamped ``T 00:00 Asia/Kolkata`` belongs to date T and must never satisfy
    a decision with ``selection_cutoff == T``.
    """
    session_rows = frame.loc[[ts.date() == prior_session for ts in frame.index]]
    if session_rows.empty:
        raise PITAcquisitionError(
            f"no Stage-A daily observation for prior completed session "
            f"{prior_session.isoformat()} for {instrument_key}; fail closed"
        )
    reference_timestamp = session_rows.index[-1]
    try:
        reference_price = Decimal(str(session_rows.iloc[-1]["close"]))
    except (InvalidOperation, ValueError, KeyError) as exc:
        raise PITAcquisitionError(
            f"reference price is invalid for {instrument_key} on {prior_session.isoformat()}"
        ) from exc
    return reference_price, reference_timestamp


@dataclass(frozen=True)
class PITResearchBoundary:
    """Caller-supplied research range; no experiment dates are hidden in code."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("research boundary start must be on or before end")
        if self.start < SUPPORTED_RESEARCH_START:
            raise PITAcquisitionError(
                "historical NSE semantics are unsupported before the verified 2024-07-01 boundary"
            )

    def as_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True)
class PITFormationPolicy:
    """Explicit separation of universe formation, signal, and execution timing."""

    policy_id: str
    timezone: str
    decision_time: time
    price_reference_policy: str
    signal_time_policy: str
    execution_time_policy: str

    def __post_init__(self) -> None:
        for name, value in (
            ("policy_id", self.policy_id),
            ("timezone", self.timezone),
            ("price_reference_policy", self.price_reference_policy),
            ("signal_time_policy", self.signal_time_policy),
            ("execution_time_policy", self.execution_time_policy),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")
        if self.price_reference_policy != "prior_completed_session_close":
            raise ValueError(
                "daily Stage-A affordability requires the verified prior completed-session close policy"
            )

    def price_cutoff(self, decision_date: date) -> pd.Timestamp:
        if self.price_reference_policy == "prior_completed_session_close":
            naive = datetime.combine(decision_date, time.min)
        return pd.Timestamp(naive, tz=self.timezone)

    def as_dict(self) -> dict[str, str]:
        return {
            "decision_time": self.decision_time.isoformat(),
            "execution_time_policy": self.execution_time_policy,
            "policy_id": self.policy_id,
            "price_reference_policy": self.price_reference_policy,
            "signal_time_policy": self.signal_time_policy,
            "timezone": self.timezone,
        }


@dataclass(frozen=True)
class AcquisitionRateLimit:
    """Pacing/retry values supplied from an evidence source, never guessed in code."""

    policy_id: str
    minimum_interval_seconds: Decimal
    max_attempts: int
    backoff_seconds: Decimal
    source_reference: str

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.source_reference.strip():
            raise ValueError("rate-limit policy and source are required")
        for name, value in (
            ("minimum_interval_seconds", self.minimum_interval_seconds),
            ("backoff_seconds", self.backoff_seconds),
        ):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite non-negative Decimal")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")

    def as_dict(self) -> dict[str, object]:
        return {
            "backoff_seconds": _decimal_text(self.backoff_seconds),
            "max_attempts": self.max_attempts,
            "minimum_interval_seconds": _decimal_text(self.minimum_interval_seconds),
            "policy_id": self.policy_id,
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True)
class CorporateActionEvidenceClaim:
    """Date-bounded corporate-action evidence CLAIM for one instrument and cutoff.

    This branch binds the claim into plan identity but does NOT authenticate it;
    convergence with the trusted corporate-action ledger verifies content. A claim
    whose ``assessment_as_of`` is after the decision cutoff is rejected so
    future or current evidence is never silently reused for earlier cutoffs.
    """

    instrument_key: str
    assessment_as_of: date
    coverage_start: date
    coverage_end: date
    source_fingerprint: str
    policy_identity: str
    blocking_events: tuple[str, ...]
    complete: bool

    def __post_init__(self) -> None:
        if not self.instrument_key.strip():
            raise PITAcquisitionError("corporate-action claim instrument_key is required")
        if self.coverage_start > self.coverage_end:
            raise PITAcquisitionError("corporate-action claim coverage range is invalid")
        _require_content_fingerprint(
            "corporate-action claim source_fingerprint", self.source_fingerprint
        )
        if not self.policy_identity.strip():
            raise PITAcquisitionError("corporate-action claim policy_identity is required")
        for event in self.blocking_events:
            if not str(event).strip():
                raise PITAcquisitionError("corporate-action blocking event identity is required")
        if type(self.complete) is not bool:
            raise PITAcquisitionError("corporate-action claim completeness must be boolean")

    def validate_for_cutoff(self, selection_cutoff: date, expected_dates: tuple[date, ...]) -> None:
        """Fail closed when this claim cannot serve the given decision cutoff."""
        if self.assessment_as_of > selection_cutoff:
            raise PITAcquisitionError(
                f"corporate-action evidence for {self.instrument_key} is assessed as of "
                f"{self.assessment_as_of.isoformat()}, after decision cutoff "
                f"{selection_cutoff.isoformat()}; future evidence is rejected"
            )
        if expected_dates and (
            self.coverage_start > min(expected_dates) or self.coverage_end < max(expected_dates)
        ):
            raise PITAcquisitionError(
                f"corporate-action evidence for {self.instrument_key} covers "
                f"{self.coverage_start.isoformat()} through {self.coverage_end.isoformat()}, "
                "which does not cover the decision trading dates"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "assessment_as_of": self.assessment_as_of.isoformat(),
            "blocking_events": list(self.blocking_events),
            "complete": self.complete,
            "coverage_end": self.coverage_end.isoformat(),
            "coverage_start": self.coverage_start.isoformat(),
            "instrument_key": self.instrument_key,
            "policy_identity": self.policy_identity,
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True)
class PITUniverseMembership:
    trade_date: date
    instrument_key: str
    symbol: str
    isin: str
    series: str
    tick_size_rupees: str
    eligible: bool
    snapshot_sha256: str
    source_row_number: int
    source_url: str

    def __post_init__(self) -> None:
        for name, value in (
            ("instrument_key", self.instrument_key),
            ("symbol", self.symbol),
            ("isin", self.isin),
            ("series", self.series),
            ("tick_size_rupees", self.tick_size_rupees),
            ("snapshot_sha256", self.snapshot_sha256),
            ("source_url", self.source_url),
        ):
            if not value.strip():
                raise ValueError(f"membership {name} is required")
        if type(self.eligible) is not bool or type(self.source_row_number) is not int:
            raise ValueError("membership eligible/source_row_number types are invalid")
        if self.source_row_number < 1:
            raise ValueError("membership source_row_number must be positive")
        try:
            tick_size = Decimal(self.tick_size_rupees)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("membership tick_size_rupees must be a Decimal") from exc
        if not tick_size.is_finite() or tick_size <= 0:
            raise ValueError("membership tick_size_rupees must be positive and finite")

    def as_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "instrument_key": self.instrument_key,
            "isin": self.isin,
            "series": self.series,
            "snapshot_sha256": self.snapshot_sha256,
            "source_row_number": self.source_row_number,
            "source_url": self.source_url,
            "symbol": self.symbol,
            "tick_size_rupees": self.tick_size_rupees,
            "trade_date": self.trade_date.isoformat(),
        }


@dataclass(frozen=True)
class PITUniverseSource:
    manifest_path: str
    manifest_sha256: str
    schema_version: int
    trading_dates: tuple[date, ...]
    memberships: tuple[PITUniverseMembership, ...]

    @property
    def eligible_memberships(self) -> tuple[PITUniverseMembership, ...]:
        return tuple(item for item in self.memberships if item.eligible)

    @property
    def eligible_instrument_keys(self) -> tuple[str, ...]:
        return tuple(sorted({item.instrument_key for item in self.eligible_memberships}))


def _resolve_artifact_path(manifest_path: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def load_pit_universe_source(manifest_path: Path) -> PITUniverseSource:
    """Load and verify an existing completed NSE batch manifest without network access."""

    raw_bytes = manifest_path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise IncompleteUniverseManifestError("NSE universe manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise IncompleteUniverseManifestError("NSE universe manifest must be an object")
    summary = payload.get("summary")
    days = payload.get("days")
    failures = payload.get("failures")
    calendar = payload.get("calendar")
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "success"
        or not isinstance(days, list)
        or not isinstance(failures, list)
        or failures
        or not isinstance(calendar, dict)
    ):
        raise IncompleteUniverseManifestError(
            "NSE universe manifest is incomplete; Stage A cannot use partial source data"
        )

    try:
        trading_dates = tuple(
            date.fromisoformat(str(item)) for item in calendar["normal_trading_dates"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IncompleteUniverseManifestError(
            "NSE universe manifest has no valid normal trading-date list"
        ) from exc
    if not trading_dates or len(set(trading_dates)) != len(trading_dates):
        raise IncompleteUniverseManifestError("normal trading dates must be unique and non-empty")
    if any(item.weekday() >= 5 for item in trading_dates):
        raise IncompleteUniverseManifestError("normal trading dates cannot contain weekends")

    day_by_date: dict[date, dict[str, object]] = {}
    for raw_day in days:
        if not isinstance(raw_day, dict):
            raise IncompleteUniverseManifestError("NSE universe day entry must be an object")
        try:
            report_date = date.fromisoformat(str(raw_day["report_date"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise IncompleteUniverseManifestError(
                "NSE universe day has invalid report_date"
            ) from exc
        if report_date in day_by_date:
            raise IncompleteUniverseManifestError(f"duplicate NSE universe date {report_date}")
        day_by_date[report_date] = raw_day
    if set(day_by_date) != set(trading_dates):
        raise IncompleteUniverseManifestError(
            "NSE universe manifest must contain exactly every sourced normal trading date; "
            "holidays/special sessions are excluded by the calendar and are not missing data"
        )

    memberships: list[PITUniverseMembership] = []
    for trade_date in trading_dates:
        day = day_by_date[trade_date]
        source_url = str(day.get("source_url", "")).strip()
        if not source_url:
            raise IncompleteUniverseManifestError(
                f"NSE universe day has no source URL for {trade_date}"
            )
        expected_snapshot_hash = str(day.get("snapshot_sha256", ""))
        raw_gzip_value = day.get("raw_gzip")
        if not raw_gzip_value:
            raise IncompleteUniverseManifestError(
                f"NSE universe day has no raw snapshot path for {trade_date}"
            )
        raw_gzip = _resolve_artifact_path(manifest_path, raw_gzip_value)
        if not raw_gzip.exists():
            raise IncompleteUniverseManifestError(
                f"missing immutable NSE raw snapshot for {trade_date}: {raw_gzip}"
            )
        actual_snapshot_hash = _sha256_file(raw_gzip)
        if actual_snapshot_hash != expected_snapshot_hash:
            raise IncompleteUniverseManifestError(
                f"NSE raw snapshot hash mismatch for {trade_date}"
            )
        hash_sidecar = raw_gzip.with_suffix(raw_gzip.suffix + ".sha256")
        sidecar_hash = (
            hash_sidecar.read_text(encoding="utf-8").strip() if hash_sidecar.exists() else None
        )
        if sidecar_hash != actual_snapshot_hash:
            raise IncompleteUniverseManifestError(
                f"NSE raw snapshot SHA-256 sidecar is missing or mismatched for {trade_date}"
            )
        parquet_path = _resolve_artifact_path(manifest_path, day.get("universe_parquet"))
        if not parquet_path.exists():
            raise IncompleteUniverseManifestError(
                f"missing NSE daily universe artifact for {trade_date}: {parquet_path}"
            )
        required_columns = [
            "report_date",
            "instrument_key",
            "symbol",
            "isin",
            "series",
            "tick_size_rupees",
            "eligible",
            "snapshot_sha256",
            "source_row_number",
            "source_url",
        ]
        try:
            frame = pd.read_parquet(parquet_path, columns=required_columns)
        except Exception as exc:  # noqa: BLE001 - source artifact must fail closed
            raise IncompleteUniverseManifestError(
                f"cannot read NSE daily universe artifact for {trade_date}: {exc}"
            ) from exc
        try:
            expected_records = int(day["records"])
            expected_eligible_records = int(day["eligible_records"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IncompleteUniverseManifestError(
                f"NSE daily record counts are invalid for {trade_date}"
            ) from exc
        if len(frame) != expected_records:
            raise IncompleteUniverseManifestError(
                f"NSE daily record count mismatch for {trade_date}"
            )
        if int(frame["eligible"].sum()) != expected_eligible_records:
            raise IncompleteUniverseManifestError(
                f"NSE daily eligible record count mismatch for {trade_date}"
            )
        seen: set[str] = set()
        for row in frame.itertuples(index=False):
            row_date = date.fromisoformat(str(row.report_date))
            if row_date != trade_date:
                raise IncompleteUniverseManifestError(
                    f"NSE daily artifact date mismatch for {trade_date}"
                )
            key = str(row.instrument_key)
            if key in seen:
                raise IncompleteUniverseManifestError(
                    f"duplicate instrument identity in daily universe {trade_date}: {key}"
                )
            seen.add(key)
            if str(row.snapshot_sha256) != expected_snapshot_hash:
                raise IncompleteUniverseManifestError(
                    f"daily row snapshot hash mismatch for {trade_date}: {key}"
                )
            if str(row.source_url) != source_url:
                raise IncompleteUniverseManifestError(
                    f"daily row source URL mismatch for {trade_date}: {key}"
                )
            memberships.append(
                PITUniverseMembership(
                    trade_date=trade_date,
                    instrument_key=key,
                    symbol=str(row.symbol),
                    isin=str(row.isin),
                    series=str(row.series),
                    tick_size_rupees=str(row.tick_size_rupees),
                    eligible=bool(row.eligible),
                    snapshot_sha256=str(row.snapshot_sha256),
                    source_row_number=int(row.source_row_number),
                    source_url=str(row.source_url),
                )
            )
    try:
        schema_version = int(payload["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IncompleteUniverseManifestError("NSE universe schema version is invalid") from exc
    if schema_version < 1:
        raise IncompleteUniverseManifestError("NSE universe schema version is unsupported")
    return PITUniverseSource(
        manifest_path=str(manifest_path.resolve()),
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        schema_version=schema_version,
        trading_dates=tuple(sorted(trading_dates)),
        memberships=tuple(
            sorted(memberships, key=lambda item: (item.trade_date, item.instrument_key))
        ),
    )


@dataclass(frozen=True)
class PITHistoricalRequest:
    instrument_key: str
    symbol: str
    start: date
    end: date
    eligible_dates: tuple[date, ...]
    symbol_by_date: tuple[tuple[date, str], ...]

    def __post_init__(self) -> None:
        if not self.instrument_key.strip() or not self.symbol.strip():
            raise ValueError("historical request instrument identity is required")
        if self.start > self.end or not self.eligible_dates:
            raise ValueError("historical request range/membership is invalid")
        if tuple(sorted(set(self.eligible_dates))) != self.eligible_dates:
            raise ValueError("eligible dates must be unique and sorted")
        if any(not symbol.strip() for _, symbol in self.symbol_by_date):
            raise ValueError("date-scoped symbols cannot be blank")

    def as_dict(self) -> dict[str, object]:
        return {
            "eligible_dates": [item.isoformat() for item in self.eligible_dates],
            "end": self.end.isoformat(),
            "instrument_key": self.instrument_key,
            "start": self.start.isoformat(),
            "symbol": self.symbol,
            "symbol_by_date": [
                {"date": day.isoformat(), "symbol": symbol} for day, symbol in self.symbol_by_date
            ],
        }


def _validate_capital(capital: Decimal) -> None:
    if not isinstance(capital, Decimal) or not capital.is_finite() or capital <= 0:
        raise ValueError("approved capital must be a positive finite Decimal")


@dataclass(frozen=True)
class StageAPlan:
    boundary: PITResearchBoundary
    source: PITUniverseSource
    candidates: tuple[PITHistoricalRequest, ...]
    approved_capital_rupees: Decimal
    thresholds: ResearchUniverseThresholds
    formation_policy: PITFormationPolicy
    rate_limit: AcquisitionRateLimit
    universe_rule_version: str
    adjustment_policy: str
    lookback_trading_sessions: int
    estimated_rows_per_trading_day: int
    estimated_bytes_per_row: int
    estimated_requests: int
    estimated_rows: int
    estimated_storage_bytes: int
    cost_model_identity: str

    def __post_init__(self) -> None:
        _validate_capital(self.approved_capital_rupees)
        if not self.candidates:
            raise ValueError("Stage A requires at least one PIT candidate")
        if self.lookback_trading_sessions < 1:
            raise ValueError("lookback_trading_sessions must be positive and explicit")
        if self.estimated_rows_per_trading_day < 1 or self.estimated_bytes_per_row < 1:
            raise ValueError("Stage A storage estimates must be positive and explicit")
        for name, value in (
            ("universe_rule_version", self.universe_rule_version),
            ("adjustment_policy", self.adjustment_policy),
            ("cost_model_identity", self.cost_model_identity),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")

    def deterministic_payload(self) -> dict[str, object]:
        return {
            "adjustment_policy": self.adjustment_policy,
            "approved_capital_rupees": self.approved_capital_rupees,
            "boundary": self.boundary.as_dict(),
            "candidates": [item.as_dict() for item in self.candidates],
            "cost_model_identity": self.cost_model_identity,
            "estimated_bytes_per_row": self.estimated_bytes_per_row,
            "estimated_requests": self.estimated_requests,
            "estimated_rows": self.estimated_rows,
            "estimated_rows_per_trading_day": self.estimated_rows_per_trading_day,
            "formation_policy": self.formation_policy.as_dict(),
            "lookback_trading_sessions": self.lookback_trading_sessions,
            "rate_limit": self.rate_limit.as_dict(),
            "source_manifest_sha256": self.source.manifest_sha256,
            "source_manifest_schema_version": self.source.schema_version,
            "thresholds": _thresholds_dict(self.thresholds),
            "trading_dates": [item.isoformat() for item in self.source.trading_dates],
            "universe_rule_version": self.universe_rule_version,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.deterministic_payload())

    def as_dict(self) -> dict[str, object]:
        return {
            "deterministic_fingerprint": self.fingerprint,
            "live_orders_called": False,
            "schema_version": STAGE_A_SCHEMA,
            "source_manifest_path": self.source.manifest_path,
            "source_manifest_sha256": self.source.manifest_sha256,
            "stage": "A",
            "summary": {
                "candidate_count": len(self.candidates),
                "estimated_requests": self.estimated_requests,
                "estimated_rows": self.estimated_rows,
                "estimated_storage_bytes": self.estimated_storage_bytes,
                "live_orders_called": False,
            },
            **self.deterministic_payload(),
        }


def build_stage_a_plan(
    *,
    universe_manifest_path: Path,
    boundary: PITResearchBoundary,
    approved_capital_rupees: Decimal,
    thresholds: ResearchUniverseThresholds,
    formation_policy: PITFormationPolicy,
    rate_limit: AcquisitionRateLimit,
    universe_rule_version: str,
    adjustment_policy: str,
    lookback_trading_sessions: int,
    estimated_rows_per_trading_day: int,
    estimated_bytes_per_row: int,
    cost_model_identity: str,
) -> StageAPlan:
    source = load_pit_universe_source(universe_manifest_path)
    _validate_capital(approved_capital_rupees)
    eligible_by_key: dict[str, list[PITUniverseMembership]] = {}
    for membership in source.eligible_memberships:
        if boundary.start <= membership.trade_date <= boundary.end:
            eligible_by_key.setdefault(membership.instrument_key, []).append(membership)
    candidates: list[PITHistoricalRequest] = []
    for key, memberships in sorted(eligible_by_key.items()):
        memberships.sort(key=lambda item: item.trade_date)
        first_date = memberships[0].trade_date
        # Locate the download start from the actual sourced normal trading
        # sessions, never from naive calendar-day subtraction. A Monday first
        # date with a 1-session lookback must reach Friday, not Sunday. When the
        # full requested lookback is not sourced, fail closed instead of
        # silently acquiring a shorter history.
        try:
            first_index = source.trading_dates.index(first_date)
        except ValueError as exc:
            raise PITAcquisitionError(
                f"first eligible date {first_date.isoformat()} for {key} is not a "
                "sourced normal trading session"
            ) from exc
        if first_index < lookback_trading_sessions:
            raise PITAcquisitionError(
                f"Stage A price lookback for {key} requires "
                f"{lookback_trading_sessions} sourced trading sessions before "
                f"{first_date.isoformat()}, which are not all available; fail closed"
            )
        request_start: date = source.trading_dates[first_index - lookback_trading_sessions]
        if request_start < SUPPORTED_PRICE_DATA_START:
            raise PITAcquisitionError(
                f"Stage A price lookback for {key} precedes supported Upstox history"
            )
        candidates.append(
            PITHistoricalRequest(
                instrument_key=key,
                symbol=memberships[0].symbol,
                start=request_start,
                end=memberships[-1].trade_date,
                eligible_dates=tuple(item.trade_date for item in memberships),
                symbol_by_date=tuple((item.trade_date, item.symbol) for item in memberships),
            )
        )
    if not candidates:
        raise PITAcquisitionError(
            "source universe has no eligible candidates in requested boundary"
        )
    estimated_requests = sum(
        historical_request_count(start=item.start, end=item.end, interval="daily")
        for item in candidates
    )
    estimated_rows = (
        sum(len(item.eligible_dates) for item in candidates) * estimated_rows_per_trading_day
    )
    return StageAPlan(
        boundary=boundary,
        source=source,
        candidates=tuple(candidates),
        approved_capital_rupees=approved_capital_rupees,
        thresholds=thresholds,
        formation_policy=formation_policy,
        rate_limit=rate_limit,
        universe_rule_version=universe_rule_version,
        adjustment_policy=adjustment_policy,
        lookback_trading_sessions=lookback_trading_sessions,
        estimated_rows_per_trading_day=estimated_rows_per_trading_day,
        estimated_bytes_per_row=estimated_bytes_per_row,
        estimated_requests=estimated_requests,
        estimated_rows=estimated_rows,
        estimated_storage_bytes=estimated_rows * estimated_bytes_per_row,
        cost_model_identity=cost_model_identity,
    )


def stage_a_batch_candidates(plan: StageAPlan) -> tuple[HistoricalBatchCandidate, ...]:
    """Adapt the immutable Stage-A plan to the existing daily batch downloader."""

    return tuple(
        HistoricalBatchCandidate(
            instrument_key=item.instrument_key,
            symbol=item.symbol,
            start=item.start,
            end=item.end,
        )
        for item in plan.candidates
    )


@dataclass(frozen=True)
class StageAPrefilterDecision:
    instrument_key: str
    symbol: str
    eligible: bool
    reasons: tuple[str, ...]
    daily_dataset_fingerprint: str | None
    reference_price_rupees: Decimal | None
    reference_price_timestamp: pd.Timestamp | None
    observed_trading_days: int
    affordable_quantity: int | None
    cash_required_rupees: Decimal | None
    entry_charges_rupees: Decimal | None
    median_daily_notional_proxy_rupees: Decimal | None
    median_daily_volume_shares: Decimal | None
    evidence_as_of: pd.Timestamp | None = None
    prior_completed_session: date | None = None
    ca_evidence_as_of: date | None = None
    ca_source_fingerprint: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "affordable_quantity": self.affordable_quantity,
            "ca_evidence_as_of": (
                self.ca_evidence_as_of.isoformat() if self.ca_evidence_as_of is not None else None
            ),
            "ca_source_fingerprint": self.ca_source_fingerprint,
            "cash_required_rupees": _decimal_text(self.cash_required_rupees),
            "daily_dataset_fingerprint": self.daily_dataset_fingerprint,
            "eligible": self.eligible,
            "entry_charges_rupees": _decimal_text(self.entry_charges_rupees),
            "evidence_as_of": (
                self.evidence_as_of.isoformat() if self.evidence_as_of is not None else None
            ),
            "instrument_key": self.instrument_key,
            "median_daily_notional_proxy_rupees": _decimal_text(
                self.median_daily_notional_proxy_rupees
            ),
            "median_daily_volume_shares": _decimal_text(self.median_daily_volume_shares),
            "observed_trading_days": self.observed_trading_days,
            "prior_completed_session": (
                self.prior_completed_session.isoformat()
                if self.prior_completed_session is not None
                else None
            ),
            "reasons": list(self.reasons),
            "reference_price_rupees": _decimal_text(self.reference_price_rupees),
            "reference_price_timestamp": (
                self.reference_price_timestamp.isoformat()
                if self.reference_price_timestamp is not None
                else None
            ),
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class StageAPrefilterResult:
    stage_a_plan_fingerprint: str
    source_manifest_sha256: str
    boundary: PITResearchBoundary
    selection_cutoff: date
    selection_as_of: pd.Timestamp
    formation_policy: PITFormationPolicy
    approved_capital_rupees: Decimal
    thresholds: ResearchUniverseThresholds
    cost_model_identity: str
    decisions: tuple[StageAPrefilterDecision, ...]
    failures: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.failures and bool(self.decisions)

    @property
    def candidate_count(self) -> int:
        return sum(item.eligible for item in self.decisions)

    @property
    def affordability_candidate_count(self) -> int:
        return sum(
            item.affordable_quantity is not None
            and item.affordable_quantity >= self.thresholds.min_affordable_quantity
            for item in self.decisions
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.deterministic_payload())

    def deterministic_payload(self) -> dict[str, object]:
        return {
            "approved_capital_rupees": self.approved_capital_rupees,
            "affordability_candidate_count": self.affordability_candidate_count,
            "boundary": self.boundary.as_dict(),
            "cost_model_identity": self.cost_model_identity,
            "decisions": [item.as_dict() for item in self.decisions],
            "failures": list(self.failures),
            "formation_policy": self.formation_policy.as_dict(),
            "selection_as_of": self.selection_as_of,
            "selection_cutoff": self.selection_cutoff,
            "source_manifest_sha256": self.source_manifest_sha256,
            "stage_a_plan_fingerprint": self.stage_a_plan_fingerprint,
            "thresholds": _thresholds_dict(self.thresholds),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "approved_capital_rupees": _decimal_text(self.approved_capital_rupees),
            "affordability_candidate_count": self.affordability_candidate_count,
            "boundary": self.boundary.as_dict(),
            "complete": self.complete,
            "cost_model_identity": self.cost_model_identity,
            "decisions": [item.as_dict() for item in self.decisions],
            "deterministic_fingerprint": self.fingerprint,
            "failures": list(self.failures),
            "formation_policy": self.formation_policy.as_dict(),
            "schema_version": "openalgo-pit-historical-acquisition/stage-a-prefilter/v1",
            "selection_as_of": self.selection_as_of.isoformat(),
            "selection_cutoff": self.selection_cutoff.isoformat(),
            "source_manifest_sha256": self.source_manifest_sha256,
            "stage_a_plan_fingerprint": self.stage_a_plan_fingerprint,
            "thresholds": _thresholds_dict(self.thresholds),
            "live_orders_called": False,
        }


def _prefilter_rejection(
    *,
    key: str,
    symbol: str,
    fingerprint: str | None,
    reasons: list[str],
    reference_price: Decimal | None = None,
    reference_timestamp: pd.Timestamp | None = None,
    observed_days: int = 0,
    affordable_quantity: int | None = None,
    cash_required: Decimal | None = None,
    entry_charges: Decimal | None = None,
    notional_proxy: Decimal | None = None,
    volume: Decimal | None = None,
    evidence_as_of: pd.Timestamp | None = None,
    prior_completed_session: date | None = None,
    ca_evidence_as_of: date | None = None,
    ca_source_fingerprint: str | None = None,
) -> StageAPrefilterDecision:
    return StageAPrefilterDecision(
        instrument_key=key,
        symbol=symbol,
        eligible=False,
        reasons=tuple(reasons),
        daily_dataset_fingerprint=fingerprint,
        reference_price_rupees=reference_price,
        reference_price_timestamp=reference_timestamp,
        observed_trading_days=observed_days,
        affordable_quantity=affordable_quantity,
        cash_required_rupees=cash_required,
        entry_charges_rupees=entry_charges,
        median_daily_notional_proxy_rupees=notional_proxy,
        median_daily_volume_shares=volume,
        evidence_as_of=evidence_as_of,
        prior_completed_session=prior_completed_session,
        ca_evidence_as_of=ca_evidence_as_of,
        ca_source_fingerprint=ca_source_fingerprint,
    )


def build_stage_a_prefilter(
    *,
    stage_a_plan: StageAPlan,
    daily_frames: Mapping[str, pd.DataFrame],
    daily_dataset_fingerprints: Mapping[str, str],
    tick_policies: Mapping[str, TickPolicy],
    corporate_actions: Mapping[str, CorporateActionAssessment],
    minimum_tradable_quantities: Mapping[str, int],
    cost_provider: CostProvider,
    selection_cutoff: date,
    selection_as_of: pd.Timestamp | None = None,
    corporate_action_claims: Mapping[str, CorporateActionEvidenceClaim] | None = None,
) -> StageAPrefilterResult:
    """Evaluate Stage-A daily data without using prices after the formation decision.

    When ``corporate_action_claims`` is supplied, each instrument uses its
    date-bounded claim (``assessment_as_of`` must not be after the decision
    cutoff); the opaque ``corporate_actions`` mapping is then ignored for covered
    instruments. The claim is bound into the decision but never authenticated
    here.
    """

    if not stage_a_plan.boundary.start <= selection_cutoff <= stage_a_plan.boundary.end:
        raise ValueError("selection_cutoff must be inside the acquisition boundary")
    cutoff_timestamp = stage_a_plan.formation_policy.price_cutoff(selection_cutoff)
    if selection_as_of is None:
        selection_as_of = cutoff_timestamp
    else:
        selection_as_of = pd.Timestamp(selection_as_of)
        if selection_as_of.tzinfo is None or selection_as_of > cutoff_timestamp:
            raise ValueError(
                "selection_as_of must be timezone-aware and not after the price cutoff"
            )

    try:
        calendar_prior_session: date | None = prior_completed_trading_session(
            selection_cutoff, stage_a_plan.source.trading_dates
        )
    except PITAcquisitionError:
        calendar_prior_session = None

    decisions: list[StageAPrefilterDecision] = []
    failures: list[str] = []
    for candidate in stage_a_plan.candidates:
        key = candidate.instrument_key
        fingerprint = daily_dataset_fingerprints.get(key)
        frame = daily_frames.get(key)
        symbol_by_date = dict(candidate.symbol_by_date)
        if not fingerprint or frame is None:
            reason = f"missing Stage-A daily data or fingerprint for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                )
            )
            continue
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            reason = f"Stage-A daily timestamps must be timezone-aware for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                )
            )
            continue
        violations = validate_ohlcv_frame(frame)
        if violations:
            reason = f"invalid Stage-A daily OHLCV for {key}: {'; '.join(violations)}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                )
            )
            continue

        # Strict PIT history: only trading dates strictly before the formation
        # cutoff. A daily candle timestamped T 00:00 belongs to date T and must
        # never inform a decision with selection_cutoff == T.
        history_frame = frame.loc[[ts.date() < selection_cutoff for ts in frame.index]]
        if stage_a_plan.formation_policy.price_reference_policy == "prior_completed_session_close":
            expected_dates = tuple(
                day for day in candidate.eligible_dates if day < selection_cutoff
            )
        else:
            expected_dates = tuple(
                day for day in candidate.eligible_dates if day <= selection_cutoff
            )
        observed_dates = set(history_frame.index.date)
        missing_dates = tuple(day for day in expected_dates if day not in observed_dates)
        if missing_dates:
            reason = f"missing Stage-A daily observations for {key}: " + ", ".join(
                day.isoformat() for day in missing_dates
            )
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    observed_days=len(observed_dates.intersection(expected_dates)),
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                )
            )
            continue
        if not expected_dates:
            reason = f"no point-in-time eligible membership date available by {selection_cutoff} for {key}"
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                )
            )
            continue

        if calendar_prior_session is None:
            reason = (
                f"no prior completed trading session exists before {selection_cutoff} for {key}; "
                "reference price is unknown"
            )
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    evidence_as_of=selection_as_of,
                    prior_completed_session=None,
                )
            )
            continue
        if calendar_prior_session not in set(expected_dates):
            reason = (
                f"no PIT eligible prior completed session {calendar_prior_session.isoformat()} "
                f"for {key} by {selection_cutoff}; fail closed without using the current close"
            )
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=symbol_by_date.get(calendar_prior_session, candidate.symbol),
                    fingerprint=fingerprint,
                    reasons=[reason],
                    observed_days=len(expected_dates),
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                )
            )
            continue

        try:
            reference_price, reference_timestamp = _prior_session_close(
                frame, calendar_prior_session, instrument_key=key
            )
        except PITAcquisitionError as exc:
            reason = str(exc)
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=symbol_by_date.get(calendar_prior_session, candidate.symbol),
                    fingerprint=fingerprint,
                    reasons=[reason],
                    observed_days=len(expected_dates),
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                )
            )
            continue

        if history_frame.empty:
            reason = f"no Stage-A daily price is available by the formation cutoff for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                )
            )
            continue

        expected_date_set = set(expected_dates)
        eligible_frame = history_frame.loc[
            [timestamp.date() in expected_date_set for timestamp in history_frame.index]
        ]
        if not reference_price.is_finite() or reference_price <= 0:
            reason = f"reference price is invalid for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=symbol_by_date.get(calendar_prior_session, candidate.symbol),
                    fingerprint=fingerprint,
                    reasons=[reason],
                    reference_price=reference_price,
                    reference_timestamp=reference_timestamp,
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                )
            )
            continue
        minimum_quantity = minimum_tradable_quantities.get(key)
        tick_policy = tick_policies.get(key)
        ca_evidence_as_of: date | None = None
        ca_source_fingerprint: str | None = None
        corporate_action: CorporateActionAssessment | None = None
        if corporate_action_claims is not None:
            ca_claim = corporate_action_claims.get(key)
            if ca_claim is None:
                reason = f"missing Stage-A tick/lot/corporate-action evidence for {key}"
                failures.append(reason)
                decisions.append(
                    _prefilter_rejection(
                        key=key,
                        symbol=symbol_by_date.get(calendar_prior_session, candidate.symbol),
                        fingerprint=fingerprint,
                        reasons=[reason],
                        reference_price=reference_price,
                        reference_timestamp=reference_timestamp,
                        observed_days=len(expected_dates),
                        evidence_as_of=selection_as_of,
                        prior_completed_session=calendar_prior_session,
                    )
                )
                continue
            try:
                ca_claim.validate_for_cutoff(selection_cutoff, expected_dates)
            except PITAcquisitionError as exc:
                reason = str(exc)
                failures.append(reason)
                decisions.append(
                    _prefilter_rejection(
                        key=key,
                        symbol=symbol_by_date.get(calendar_prior_session, candidate.symbol),
                        fingerprint=fingerprint,
                        reasons=[reason],
                        reference_price=reference_price,
                        reference_timestamp=reference_timestamp,
                        observed_days=len(expected_dates),
                        evidence_as_of=selection_as_of,
                        prior_completed_session=calendar_prior_session,
                        ca_evidence_as_of=ca_claim.assessment_as_of,
                        ca_source_fingerprint=ca_claim.source_fingerprint,
                    )
                )
                continue
            corporate_action = CorporateActionAssessment(
                complete=ca_claim.complete,
                blocking_events=tuple(ca_claim.blocking_events),
            )
            ca_evidence_as_of = ca_claim.assessment_as_of
            ca_source_fingerprint = ca_claim.source_fingerprint
        else:
            corporate_action = corporate_actions.get(key)
        if minimum_quantity is None or tick_policy is None or corporate_action is None:
            reason = f"missing Stage-A tick/lot/corporate-action evidence for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=symbol_by_date.get(calendar_prior_session, candidate.symbol),
                    fingerprint=fingerprint,
                    reasons=[reason],
                    reference_price=reference_price,
                    reference_timestamp=reference_timestamp,
                    observed_days=len(expected_dates),
                    evidence_as_of=selection_as_of,
                    prior_completed_session=calendar_prior_session,
                    ca_evidence_as_of=ca_evidence_as_of,
                    ca_source_fingerprint=ca_source_fingerprint,
                )
            )
            continue

        membership = assess_historical_membership(
            instrument_key=key,
            trading_dates=expected_dates,
            statuses=(
                HistoricalTradingStatus(
                    trade_date=day,
                    instrument_key=key,
                    listed_on_nse=True,
                    normal_equity=True,
                    tradeable_in_normal_market=True,
                    source=stage_a_plan.source.manifest_sha256,
                )
                for day in expected_dates
            ),
        )
        tick_coverage = assess_tick_policy_coverage(
            policy=tick_policy,
            trading_dates=expected_dates,
        )
        affordability = max_affordable_buy_quantity(
            instrument_token=key,
            exchange=Exchange.NSE,
            product=Product.INTRADAY,
            price=reference_price,
            cash_limit=stage_a_plan.approved_capital_rupees,
            cost_provider=cost_provider,
            minimum_tradable_quantity=minimum_quantity,
        )
        liquidity = summarize_historical_liquidity(
            frame=eligible_frame,
            affordable_quantity_after_entry_costs=affordability.quantity,
            approved_capital_rupees=stage_a_plan.approved_capital_rupees,
        )
        decision = evaluate_research_universe_candidate(
            instrument_key=key,
            liquidity=liquidity,
            corporate_actions=corporate_action,
            historical_membership=membership,
            tick_coverage=tick_coverage,
            thresholds=stage_a_plan.thresholds,
        )
        decisions.append(
            StageAPrefilterDecision(
                instrument_key=key,
                symbol=symbol_by_date.get(calendar_prior_session, candidate.symbol),
                eligible=decision.eligible,
                reasons=decision.violations,
                daily_dataset_fingerprint=fingerprint,
                reference_price_rupees=reference_price,
                reference_price_timestamp=reference_timestamp,
                observed_trading_days=liquidity.observed_trading_days,
                affordable_quantity=affordability.quantity,
                cash_required_rupees=affordability.cash_required,
                entry_charges_rupees=affordability.entry_charges,
                median_daily_notional_proxy_rupees=liquidity.median_daily_notional_proxy_rupees,
                median_daily_volume_shares=liquidity.median_daily_volume_shares,
                evidence_as_of=selection_as_of,
                prior_completed_session=calendar_prior_session,
                ca_evidence_as_of=ca_evidence_as_of,
                ca_source_fingerprint=ca_source_fingerprint,
            )
        )
    return StageAPrefilterResult(
        stage_a_plan_fingerprint=stage_a_plan.fingerprint,
        source_manifest_sha256=stage_a_plan.source.manifest_sha256,
        boundary=stage_a_plan.boundary,
        selection_cutoff=selection_cutoff,
        selection_as_of=selection_as_of,
        formation_policy=stage_a_plan.formation_policy,
        approved_capital_rupees=stage_a_plan.approved_capital_rupees,
        thresholds=stage_a_plan.thresholds,
        cost_model_identity=stage_a_plan.cost_model_identity,
        decisions=tuple(sorted(decisions, key=lambda item: item.instrument_key)),
        failures=tuple(sorted(failures)),
    )


def build_stage_a_prefilter_timeline(
    *,
    stage_a_plan: StageAPlan,
    daily_frames: Mapping[str, pd.DataFrame],
    daily_dataset_fingerprints: Mapping[str, str],
    tick_policies: Mapping[str, TickPolicy],
    corporate_actions: Mapping[str, CorporateActionAssessment],
    minimum_tradable_quantities: Mapping[str, int],
    cost_provider: CostProvider,
    selection_cutoffs: tuple[date, ...],
    corporate_action_claims_by_cutoff: Mapping[date, Mapping[str, CorporateActionEvidenceClaim]]
    | None = None,
) -> tuple[StageAPrefilterResult, ...]:
    """Build deterministic window-scoped prefilters, each strictly point-in-time.

    Every timeline entry may use only evidence available at or before its own
    formation cutoff. The Stage-B union must be derived from this timeline, never
    from a single final-day prefilter alone.

    When more than one cutoff is present, date-bounded corporate-action claims
    are required per cutoff so future evidence cannot leak into earlier windows;
    a shared opaque mapping is then refused.
    """
    if not selection_cutoffs:
        raise ValueError("selection_cutoffs must contain at least one window cutoff")
    if tuple(sorted(set(selection_cutoffs))) != tuple(selection_cutoffs):
        raise ValueError("selection_cutoffs must be sorted unique dates")
    for cutoff in selection_cutoffs:
        if not stage_a_plan.boundary.start <= cutoff <= stage_a_plan.boundary.end:
            raise ValueError("every timeline cutoff must be inside the acquisition boundary")
    if corporate_action_claims_by_cutoff is None:
        if len(selection_cutoffs) > 1:
            raise ValueError(
                "multi-cutoff timelines require date-bounded corporate-action claims "
                "per cutoff; reusing one opaque mapping across cutoffs is refused"
            )
        return tuple(
            build_stage_a_prefilter(
                stage_a_plan=stage_a_plan,
                daily_frames=daily_frames,
                daily_dataset_fingerprints=daily_dataset_fingerprints,
                tick_policies=tick_policies,
                corporate_actions=corporate_actions,
                minimum_tradable_quantities=minimum_tradable_quantities,
                cost_provider=cost_provider,
                selection_cutoff=cutoff,
            )
            for cutoff in selection_cutoffs
        )
    if set(corporate_action_claims_by_cutoff) != set(selection_cutoffs):
        raise ValueError("corporate-action claims must be supplied for exactly every cutoff")
    return tuple(
        build_stage_a_prefilter(
            stage_a_plan=stage_a_plan,
            daily_frames=daily_frames,
            daily_dataset_fingerprints=daily_dataset_fingerprints,
            tick_policies=tick_policies,
            corporate_actions=corporate_actions,
            minimum_tradable_quantities=minimum_tradable_quantities,
            cost_provider=cost_provider,
            selection_cutoff=cutoff,
            corporate_action_claims=corporate_action_claims_by_cutoff[cutoff],
        )
        for cutoff in selection_cutoffs
    )


def canonical_eligible_intervals(
    eligible_dates: tuple[date, ...],
) -> tuple[tuple[date, date], ...]:
    """Collapse sorted eligible dates into canonical contiguous [start, end] intervals."""
    if not eligible_dates:
        raise PITAcquisitionError("eligible dates must not be empty")
    if tuple(sorted(set(eligible_dates))) != eligible_dates:
        raise PITAcquisitionError("eligible dates must be sorted unique")
    intervals: list[tuple[date, date]] = []
    start = previous = eligible_dates[0]
    for day in eligible_dates[1:]:
        if (day - previous).days == 1:
            previous = day
            continue
        intervals.append((start, previous))
        start = previous = day
    intervals.append((start, previous))
    return tuple(intervals)


def symbol_ranges_for(
    symbol_by_date: tuple[tuple[date, str], ...],
) -> tuple[tuple[date, date, str], ...]:
    """Collapse date-scoped symbols into canonical contiguous ranges."""
    if not symbol_by_date:
        raise PITAcquisitionError("symbol lineage must not be empty")
    ordered = tuple(sorted(symbol_by_date))
    if len({day for day, _ in ordered}) != len(ordered):
        raise PITAcquisitionError("symbol lineage dates must be unique")
    for _, symbol in ordered:
        if not str(symbol).strip():
            raise PITAcquisitionError("symbol lineage symbols cannot be blank")
    ranges: list[tuple[date, date, str]] = []
    range_start, current_symbol = ordered[0][0], ordered[0][1]
    range_end = ordered[0][0]
    for day, symbol in ordered[1:]:
        if symbol == current_symbol and (day - range_end).days == 1:
            range_end = day
            continue
        ranges.append((range_start, range_end, current_symbol))
        range_start, range_end, current_symbol = day, day, symbol
    ranges.append((range_start, range_end, current_symbol))
    return tuple(ranges)


def stage_b_eligibility_mask_fingerprint(
    instrument_key: str, eligible_dates: tuple[date, ...], source_manifest_sha256: str
) -> str:
    """Fingerprint the exact eligibility mask for one Stage-B instrument."""
    return _fingerprint(
        {
            "instrument_key": instrument_key,
            "eligible_dates": [day.isoformat() for day in eligible_dates],
            "source_manifest_sha256": source_manifest_sha256,
        }
    )


@dataclass(frozen=True)
class StageBInstrumentAcquisition:
    """Per-instrument Stage-B provenance: broad RAW range plus exact eligible mask.

    The downloader may fetch the continuous ``raw_start`` through ``raw_end``
    range, so gap bars can exist physically. The resulting raw Parquet is labelled
    :data:`RAW_ACQUISITION_ONLY` and must never enter research directly: only
    :func:`consume_research_bars` with this mask yields research bars.
    Date-scoped ``symbol_lineage`` (not just the latest symbol) is preserved.
    """

    instrument_key: str
    data_class: str
    raw_start: date
    raw_end: date
    eligible_dates: tuple[date, ...]
    eligible_intervals: tuple[tuple[date, date], ...]
    eligibility_mask_fingerprint: str
    membership_fingerprint: str
    authorizing_cutoffs: tuple[date, ...]
    authorizing_prefilter_fingerprints: tuple[str, ...]
    symbol_lineage: tuple[tuple[date, str], ...]
    symbol_ranges: tuple[tuple[date, date, str], ...]
    download_symbol: str

    def __post_init__(self) -> None:
        if not self.instrument_key.strip():
            raise PITAcquisitionError("Stage-B instrument_key is required")
        if self.data_class != RAW_ACQUISITION_ONLY:
            raise PITAcquisitionError("Stage-B raw data must be labelled RAW_ACQUISITION_ONLY")
        if self.raw_start > self.raw_end:
            raise PITAcquisitionError("Stage-B raw range is invalid")
        if (
            not self.eligible_dates
            or tuple(sorted(set(self.eligible_dates))) != self.eligible_dates
        ):
            raise PITAcquisitionError("Stage-B eligible dates must be sorted unique")
        if self.raw_start > min(self.eligible_dates) or self.raw_end < max(self.eligible_dates):
            raise PITAcquisitionError("Stage-B raw range must cover every eligible date")
        if canonical_eligible_intervals(self.eligible_dates) != self.eligible_intervals:
            raise PITAcquisitionError("Stage-B eligible intervals are not canonical")
        if not self.membership_fingerprint.strip():
            raise PITAcquisitionError("Stage-B membership fingerprint is required")
        expected_mask = stage_b_eligibility_mask_fingerprint(
            self.instrument_key, self.eligible_dates, self.membership_fingerprint
        )
        if self.eligibility_mask_fingerprint != expected_mask:
            raise PITAcquisitionError(
                f"Stage-B eligibility mask fingerprint mismatch for {self.instrument_key}; "
                "the mask must be recomputed canonically, never hand-supplied"
            )
        if not self.authorizing_cutoffs or tuple(sorted(set(self.authorizing_cutoffs))) != tuple(
            self.authorizing_cutoffs
        ):
            raise PITAcquisitionError("Stage-B authorizing cutoffs must be sorted unique")
        if not self.authorizing_prefilter_fingerprints:
            raise PITAcquisitionError("Stage-B authorizing prefilter fingerprints are required")
        if not self.symbol_lineage:
            raise PITAcquisitionError("Stage-B symbol lineage must not be empty")
        if tuple(day for day, _ in self.symbol_lineage) != self.eligible_dates:
            raise PITAcquisitionError(
                "Stage-B symbol lineage must cover exactly the eligible dates"
            )
        if symbol_ranges_for(self.symbol_lineage) != self.symbol_ranges:
            raise PITAcquisitionError("Stage-B symbol ranges are not canonical")
        if not self.download_symbol.strip():
            raise PITAcquisitionError("Stage-B download symbol is required")
        if self.download_symbol != dict(self.symbol_lineage)[max(self.eligible_dates)]:
            raise PITAcquisitionError(
                f"Stage-B download symbol for {self.instrument_key} must be the canonical "
                "date-scoped lineage symbol at the latest eligible date, never a stale symbol"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "authorizing_cutoffs": [day.isoformat() for day in self.authorizing_cutoffs],
            "authorizing_prefilter_fingerprints": list(self.authorizing_prefilter_fingerprints),
            "data_class": self.data_class,
            "download_symbol": self.download_symbol,
            "eligibility_mask_fingerprint": self.eligibility_mask_fingerprint,
            "eligible_dates": [day.isoformat() for day in self.eligible_dates],
            "eligible_intervals": [
                {"end": end.isoformat(), "start": start.isoformat()}
                for start, end in self.eligible_intervals
            ],
            "instrument_key": self.instrument_key,
            "membership_fingerprint": self.membership_fingerprint,
            "raw_end": self.raw_end.isoformat(),
            "raw_start": self.raw_start.isoformat(),
            "symbol_lineage": [
                {"date": day.isoformat(), "symbol": symbol} for day, symbol in self.symbol_lineage
            ],
            "symbol_ranges": [
                {"end": end.isoformat(), "start": start.isoformat(), "symbol": symbol}
                for start, end, symbol in self.symbol_ranges
            ],
        }


def filter_frame_to_eligible_bars(
    frame: pd.DataFrame, eligible_dates: tuple[date, ...]
) -> pd.DataFrame:
    """Return only bars on eligible dates; gap bars from a broad RAW range are removed.

    Fails closed when any eligible date has no observation. Extra physical dates
    (for example D3/D4 inside a raw D1-D5 range) are dropped and never enter research.
    """
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise PITAcquisitionError("research frame timestamps must be timezone-aware")
    if not eligible_dates:
        raise PITAcquisitionError("eligible dates must not be empty")
    eligible_set = set(eligible_dates)
    observed = set(frame.index.date)
    missing = [day for day in eligible_dates if day not in observed]
    if missing:
        raise PITAcquisitionError(
            "raw acquisition is missing eligible observations: "
            + ", ".join(day.isoformat() for day in missing)
        )
    return frame.loc[[ts.date() in eligible_set for ts in frame.index]].copy()


def _filtered_bars_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint filtered research bars: columns, timestamps, and values."""
    payload = {
        "columns": [str(column) for column in frame.columns.tolist()],
        "index": [ts.isoformat() for ts in frame.index.tolist()],
        "values": [[str(value) for value in row] for row in frame.itertuples(index=False)],
    }
    return _fingerprint(payload)


@dataclass(frozen=True)
class VerifiedStageBResearchSlice:
    """Research-ready result produced only by trusted Stage-B verification.

    Distinct from :data:`RAW_ACQUISITION_ONLY` frames: an ordinary raw
    DataFrame can never assume this type. ``research_bars_fingerprint`` covers
    the FILTERED bars, never the raw broad-range frame. There is no
    caller-controlled ``verified`` flag; instances only arise from
    :func:`consume_research_bars` after trusted plan revalidation.
    """

    instrument_key: str
    plan_fingerprint: str
    eligibility_mask_fingerprint: str
    eligible_dates: tuple[date, ...]
    research_bars_fingerprint: str
    data_class: str = "RESEARCH_READY"

    def __post_init__(self) -> None:
        if not self.instrument_key.strip():
            raise PITAcquisitionError("verified slice instrument_key is required")
        _require_content_fingerprint("verified slice plan fingerprint", self.plan_fingerprint)
        _require_content_fingerprint(
            "verified slice eligibility mask fingerprint", self.eligibility_mask_fingerprint
        )
        _require_content_fingerprint(
            "verified slice research bars fingerprint", self.research_bars_fingerprint
        )
        if (
            not self.eligible_dates
            or tuple(sorted(set(self.eligible_dates))) != self.eligible_dates
        ):
            raise PITAcquisitionError("verified slice eligible dates must be sorted unique")
        if self.data_class != "RESEARCH_READY":
            raise PITAcquisitionError("verified slices must be labelled RESEARCH_READY")

    def as_dict(self) -> dict[str, object]:
        return {
            "data_class": self.data_class,
            "eligibility_mask_fingerprint": self.eligibility_mask_fingerprint,
            "eligible_dates": [day.isoformat() for day in self.eligible_dates],
            "instrument_key": self.instrument_key,
            "plan_fingerprint": self.plan_fingerprint,
            "research_bars_fingerprint": self.research_bars_fingerprint,
        }


def consume_research_bars(
    frame: pd.DataFrame,
    detail: StageBInstrumentAcquisition,
    *,
    plan: StageBPlan,
    stage_a_plan: StageAPlan,
    trusted_prefilters: tuple[StageAPrefilterResult, ...],
) -> tuple[pd.DataFrame, VerifiedStageBResearchSlice]:
    """Produce research-ready bars only after trusted Stage-B revalidation.

    The caller-supplied ``detail`` is never trusted directly: the plan is
    revalidated against the trusted prefilter artifacts and the caller detail
    must canonically equal the verified plan detail for its instrument. A
    self-consistent forgery therefore fails closed. Returns the FILTERED
    research frame plus its verified slice; gap bars can never enter research.
    """
    verified_details = plan.validate_against_prefilter_timeline(
        stage_a_plan=stage_a_plan, trusted_prefilters=trusted_prefilters
    )
    verified_by_key = {item.instrument_key: item for item in verified_details}
    verified = verified_by_key.get(detail.instrument_key)
    if verified is None or verified.as_dict() != detail.as_dict():
        raise PITAcquisitionError(
            f"unverified Stage-B detail for {detail.instrument_key}; "
            "research consumption requires trusted plan revalidation"
        )
    research_frame = filter_frame_to_eligible_bars(frame, verified.eligible_dates)
    research_fingerprint = _filtered_bars_fingerprint(research_frame)
    return research_frame, VerifiedStageBResearchSlice(
        instrument_key=verified.instrument_key,
        plan_fingerprint=plan.fingerprint,
        eligibility_mask_fingerprint=verified.eligibility_mask_fingerprint,
        eligible_dates=verified.eligible_dates,
        research_bars_fingerprint=research_fingerprint,
    )


@dataclass(frozen=True)
class StageBPlan:
    boundary: PITResearchBoundary
    source_manifest_sha256: str
    stage_a_plan_fingerprint: str
    prefilter_fingerprint: str
    candidates: tuple[HistoricalBatchCandidate, ...]
    details: tuple[StageBInstrumentAcquisition, ...]
    interval_minutes: int
    expected_rows_per_trading_day: int
    estimated_bytes_per_row: int
    approved_capital_rupees: Decimal
    thresholds: ResearchUniverseThresholds
    formation_policy: PITFormationPolicy
    rate_limit: AcquisitionRateLimit
    universe_rule_version: str
    adjustment_policy: str
    estimated_requests: int
    estimated_rows: int
    estimated_storage_bytes: int
    timeline_cutoffs: tuple[date, ...] = ()
    timeline_prefilter_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_capital(self.approved_capital_rupees)
        if not self.candidates:
            raise ValueError("Stage B requires at least one prefiltered candidate")
        if self.interval_minutes < 1 or self.interval_minutes > 15:
            raise ValueError("Stage B interval must be between 1 and 15 minutes")
        if self.interval_minutes != 5:
            raise ValueError("Stage B initially supports only the explicit 5-minute resolution")
        if self.expected_rows_per_trading_day < 1 or self.estimated_bytes_per_row < 1:
            raise ValueError("Stage B storage estimates must be positive and explicit")
        candidate_keys = tuple(item.instrument_key for item in self.candidates)
        detail_keys = tuple(item.instrument_key for item in self.details)
        if tuple(sorted(candidate_keys)) != tuple(sorted(detail_keys)):
            raise ValueError(
                "Stage-B candidates and details must correspond exactly: "
                "no candidate without detail, no detail without candidate"
            )
        if len(set(detail_keys)) != len(detail_keys):
            raise ValueError("Stage-B details contain duplicate instruments")
        if not self.timeline_cutoffs or tuple(sorted(set(self.timeline_cutoffs))) != tuple(
            self.timeline_cutoffs
        ):
            raise ValueError("Stage-B timeline cutoffs must be sorted unique dates")
        if len(self.timeline_cutoffs) != len(self.timeline_prefilter_fingerprints):
            raise ValueError("Stage-B timeline cutoffs and prefilter fingerprints must align")
        cutoff_to_fingerprint = dict(
            zip(self.timeline_cutoffs, self.timeline_prefilter_fingerprints, strict=True)
        )
        authorized: set[date] = set()
        by_key = {item.instrument_key: item for item in self.details}
        for candidate in self.candidates:
            detail = by_key[candidate.instrument_key]
            if (
                candidate.start != detail.raw_start
                or candidate.end != detail.raw_end
                or candidate.symbol != detail.download_symbol
            ):
                raise ValueError(
                    f"Stage-B candidate for {candidate.instrument_key} diverges from its detail"
                )
            if len(detail.authorizing_cutoffs) != len(detail.authorizing_prefilter_fingerprints):
                raise ValueError(
                    f"Stage-B authorizing cutoffs and fingerprints must align for {detail.instrument_key}"
                )
            for cutoff, fingerprint in zip(
                detail.authorizing_cutoffs,
                detail.authorizing_prefilter_fingerprints,
                strict=True,
            ):
                if cutoff_to_fingerprint.get(cutoff) != fingerprint:
                    raise ValueError(
                        f"Stage-B authorizing prefilter fingerprint mismatch for "
                        f"{detail.instrument_key} at {cutoff.isoformat()}"
                    )
                authorized.add(cutoff)
            if detail.membership_fingerprint != self.source_manifest_sha256:
                raise ValueError(
                    f"Stage-B membership fingerprint mismatch for {detail.instrument_key}"
                )
            for day in (detail.raw_start, detail.raw_end, *detail.eligible_dates):
                if not self.boundary.start <= day <= self.boundary.end:
                    raise ValueError(
                        f"Stage-B range for {detail.instrument_key} escapes the plan boundary"
                    )
        if authorized != set(self.timeline_cutoffs):
            raise ValueError("Stage-B timeline cutoffs must all authorize at least one detail")

    def deterministic_payload(self) -> dict[str, object]:
        return {
            "adjustment_policy": self.adjustment_policy,
            "approved_capital_rupees": self.approved_capital_rupees,
            "boundary": self.boundary.as_dict(),
            "candidates": [
                {
                    "end": item.end,
                    "instrument_key": item.instrument_key,
                    "start": item.start,
                    "symbol": item.symbol,
                }
                for item in self.candidates
            ],
            "details": [item.as_dict() for item in self.details],
            "estimated_bytes_per_row": self.estimated_bytes_per_row,
            "estimated_requests": self.estimated_requests,
            "estimated_rows": self.estimated_rows,
            "expected_rows_per_trading_day": self.expected_rows_per_trading_day,
            "formation_policy": self.formation_policy.as_dict(),
            "interval_minutes": self.interval_minutes,
            "prefilter_fingerprint": self.prefilter_fingerprint,
            "rate_limit": self.rate_limit.as_dict(),
            "resolution": "minutes",
            "source_manifest_sha256": self.source_manifest_sha256,
            "stage_a_plan_fingerprint": self.stage_a_plan_fingerprint,
            "thresholds": _thresholds_dict(self.thresholds),
            "timeline_cutoffs": [item.isoformat() for item in self.timeline_cutoffs],
            "timeline_prefilter_fingerprints": list(self.timeline_prefilter_fingerprints),
            "universe_rule_version": self.universe_rule_version,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.deterministic_payload())

    def validate_against_prefilter_timeline(
        self,
        *,
        stage_a_plan: StageAPlan,
        trusted_prefilters: tuple[StageAPrefilterResult, ...],
    ) -> tuple[StageBInstrumentAcquisition, ...]:
        """Revalidate this plan against trusted Stage-A prefilter artifacts.

        Rebuilds the expected Stage-B authorization from the trusted prefilters
        and compares canonical payloads covering instrument population, eligible
        dates and intervals, membership fingerprint, authorizing cutoffs and
        prefilter fingerprints, symbol lineage, raw ranges, mask fingerprints,
        Stage-A plan fingerprint, capital, thresholds, and formation policy.
        Any mismatch fails closed. Returns the verified expected details.
        """
        if not trusted_prefilters:
            raise PITAcquisitionError("trusted Stage-B revalidation requires prefilters")
        cutoffs = tuple(item.selection_cutoff for item in trusted_prefilters)
        if tuple(sorted(set(cutoffs))) != cutoffs:
            raise PITAcquisitionError("trusted prefilter cutoffs must be sorted unique dates")
        for prefilter in trusted_prefilters:
            if not prefilter.complete:
                raise PITAcquisitionError("trusted prefilter is not complete")
            if prefilter.stage_a_plan_fingerprint != stage_a_plan.fingerprint:
                raise PITAcquisitionError("trusted prefilter is not bound to Stage-A plan")
            if prefilter.source_manifest_sha256 != stage_a_plan.source.manifest_sha256:
                raise PITAcquisitionError("trusted prefilter source fingerprint mismatch")
            if prefilter.approved_capital_rupees != stage_a_plan.approved_capital_rupees:
                raise PITAcquisitionError("trusted prefilter approved capital mismatch")
        if cutoffs != self.timeline_cutoffs:
            raise PITAcquisitionError(
                "trusted Stage-B revalidation cutoff mismatch: "
                f"plan covers {[day.isoformat() for day in self.timeline_cutoffs]}"
            )
        trusted_fingerprints = tuple(item.fingerprint for item in trusted_prefilters)
        if trusted_fingerprints != self.timeline_prefilter_fingerprints:
            raise PITAcquisitionError("trusted Stage-B revalidation prefilter fingerprint mismatch")
        if self.boundary != stage_a_plan.boundary:
            raise PITAcquisitionError("Stage-B boundary diverges from trusted Stage-A plan")
        if self.source_manifest_sha256 != stage_a_plan.source.manifest_sha256:
            raise PITAcquisitionError(
                "Stage-B source fingerprint diverges from trusted Stage-A plan"
            )
        if self.stage_a_plan_fingerprint != stage_a_plan.fingerprint:
            raise PITAcquisitionError("Stage-B Stage-A fingerprint diverges from trusted plan")
        if self.approved_capital_rupees != stage_a_plan.approved_capital_rupees:
            raise PITAcquisitionError("Stage-B capital diverges from trusted Stage-A plan")
        if self.thresholds != stage_a_plan.thresholds:
            raise PITAcquisitionError("Stage-B thresholds diverge from trusted Stage-A plan")
        if self.formation_policy != stage_a_plan.formation_policy:
            raise PITAcquisitionError("Stage-B formation policy diverges from trusted Stage-A plan")
        if self.universe_rule_version != stage_a_plan.universe_rule_version:
            raise PITAcquisitionError("Stage-B universe rule diverges from trusted Stage-A plan")
        if self.adjustment_policy != stage_a_plan.adjustment_policy:
            raise PITAcquisitionError(
                "Stage-B adjustment policy diverges from trusted Stage-A plan"
            )
        expected = _stage_b_details_for_covering(
            stage_a_plan, _covering_from_prefilters(trusted_prefilters)
        )
        if [item.as_dict() for item in expected] != [item.as_dict() for item in self.details]:
            raise PITAcquisitionError(
                "trusted Stage-B revalidation mismatch: plan details diverge from the "
                "authorization rebuilt from trusted prefilter artifacts"
            )
        expected_combined = (
            trusted_fingerprints[0]
            if len(trusted_fingerprints) == 1
            else _fingerprint({"timeline_prefilter_fingerprints": list(trusted_fingerprints)})
        )
        if expected_combined != self.prefilter_fingerprint:
            raise PITAcquisitionError("Stage-B combined prefilter fingerprint mismatch")
        expected_candidates = _stage_b_candidates_from_details(expected)
        if [(c.instrument_key, c.symbol, c.start, c.end) for c in expected_candidates] != [
            (c.instrument_key, c.symbol, c.start, c.end) for c in self.candidates
        ]:
            raise PITAcquisitionError("Stage-B candidates diverge from verified details")
        return expected

    def as_dict(self) -> dict[str, object]:
        return {
            "deterministic_fingerprint": self.fingerprint,
            "schema_version": STAGE_B_SCHEMA,
            "stage": "B",
            "source_manifest_sha256": self.source_manifest_sha256,
            "stage_a_plan_fingerprint": self.stage_a_plan_fingerprint,
            "prefilter_fingerprint": self.prefilter_fingerprint,
            "candidates": [
                {
                    "end": item.end.isoformat(),
                    "instrument_key": item.instrument_key,
                    "start": item.start.isoformat(),
                    "symbol": item.symbol,
                }
                for item in self.candidates
            ],
            "summary": {
                "candidate_count": len(self.candidates),
                "estimated_requests": self.estimated_requests,
                "estimated_rows": self.estimated_rows,
                "estimated_storage_bytes": self.estimated_storage_bytes,
                "live_orders_called": False,
            },
            **self.deterministic_payload(),
        }


def _covering_from_prefilters(
    prefilters: tuple[StageAPrefilterResult, ...],
) -> dict[str, list[StageAPrefilterResult]]:
    """Map each eligible instrument to the prefilters authorizing it, in order."""
    covering: dict[str, list[StageAPrefilterResult]] = {}
    for prefilter in prefilters:
        for decision in prefilter.decisions:
            if decision.eligible:
                if prefilter not in covering.setdefault(decision.instrument_key, []):
                    covering[decision.instrument_key].append(prefilter)
    for key in covering:
        covering[key] = sorted(covering[key], key=lambda item: item.selection_cutoff)
    return covering


def _stage_b_details_for_covering(
    stage_a_plan: StageAPlan,
    covering: dict[str, list[StageAPrefilterResult]],
) -> tuple[StageBInstrumentAcquisition, ...]:
    """Build canonical per-instrument details; shared by builders and revalidation.

    Only eligible decisions carrying date-bounded corporate-action evidence can
    authorize Stage-B eligibility. Opaque unscoped assessments authorize
    exploratory single-cutoff evaluation but never research acquisition.
    """
    date_by_key = {item.instrument_key: item for item in stage_a_plan.candidates}
    details: list[StageBInstrumentAcquisition] = []
    for key in sorted(covering):
        for prefilter in covering[key]:
            decision = next(
                item for item in prefilter.decisions if item.instrument_key == key and item.eligible
            )
            if decision.ca_source_fingerprint is None:
                raise IncompletePrefilterError(
                    f"opaque corporate-action evidence cannot authorize Stage-B eligibility "
                    f"for {key}; supply date-bounded CorporateActionEvidenceClaim"
                )
        source_candidate = date_by_key.get(key)
        if source_candidate is None:
            raise IncompletePrefilterError(
                f"prefilter contains an instrument absent from Stage A: {key}"
            )
        max_covered_cutoff = max(item.selection_cutoff for item in covering[key])
        union_dates = tuple(
            day for day in source_candidate.eligible_dates if day <= max_covered_cutoff
        )
        if not union_dates:
            raise IncompletePrefilterError(f"no eligible dates covered for {key}")
        authorizing = covering[key]
        lineage = tuple(
            (day, symbol)
            for day, symbol in source_candidate.symbol_by_date
            if day <= max_covered_cutoff
        )
        # Download label uses the canonical lineage symbol at the latest eligible
        # date, never a stale prior-session symbol from a cutoff-day rename.
        symbol = dict(lineage)[max(union_dates)]
        details.append(
            StageBInstrumentAcquisition(
                instrument_key=key,
                data_class=RAW_ACQUISITION_ONLY,
                raw_start=min(union_dates),
                raw_end=max(union_dates),
                eligible_dates=union_dates,
                eligible_intervals=canonical_eligible_intervals(union_dates),
                eligibility_mask_fingerprint=stage_b_eligibility_mask_fingerprint(
                    key, union_dates, stage_a_plan.source.manifest_sha256
                ),
                membership_fingerprint=stage_a_plan.source.manifest_sha256,
                authorizing_cutoffs=tuple(item.selection_cutoff for item in authorizing),
                authorizing_prefilter_fingerprints=tuple(item.fingerprint for item in authorizing),
                symbol_lineage=lineage,
                symbol_ranges=symbol_ranges_for(lineage),
                download_symbol=symbol,
            )
        )
    return tuple(details)


def _stage_b_candidates_from_details(
    details: tuple[StageBInstrumentAcquisition, ...],
) -> tuple[HistoricalBatchCandidate, ...]:
    """Derive downloader candidates from details so the two can never diverge."""
    return tuple(
        HistoricalBatchCandidate(
            instrument_key=item.instrument_key,
            symbol=item.download_symbol,
            start=item.raw_start,
            end=item.raw_end,
        )
        for item in sorted(details, key=lambda item: item.instrument_key)
    )


def build_stage_b_plan(
    *,
    stage_a_plan: StageAPlan,
    prefilter: StageAPrefilterResult,
    interval_minutes: int,
    expected_rows_per_trading_day: int,
    estimated_bytes_per_row: int,
    rate_limit: AcquisitionRateLimit,
    window_cutoffs: tuple[date, ...],
) -> StageBPlan:
    """Create Stage B only from a complete, cryptographically bound Stage-A result.

    ``window_cutoffs`` declares every research cutoff/fold for this acquisition.
    More than one cutoff requires :func:`build_stage_b_plan_from_timeline`; the
    legacy single-final-prefilter path is refused for multi-window research so a
    final-day snapshot can never silently decide multi-year history.
    """

    if not prefilter.complete:
        raise IncompletePrefilterError("Stage B is blocked until Stage-A prefilter is complete")
    if prefilter.stage_a_plan_fingerprint != stage_a_plan.fingerprint:
        raise IncompletePrefilterError("prefilter is not bound to the supplied Stage-A plan")
    if prefilter.source_manifest_sha256 != stage_a_plan.source.manifest_sha256:
        raise IncompletePrefilterError(
            "prefilter source universe fingerprint does not match Stage A"
        )
    if prefilter.approved_capital_rupees != stage_a_plan.approved_capital_rupees:
        raise IncompletePrefilterError("prefilter approved capital differs from Stage A")
    if not stage_a_plan.boundary.start <= prefilter.selection_cutoff <= stage_a_plan.boundary.end:
        raise IncompletePrefilterError(
            "Stage B prefilter cutoff must lie inside the requested acquisition boundary"
        )
    if not window_cutoffs or tuple(sorted(set(window_cutoffs))) != tuple(window_cutoffs):
        raise IncompletePrefilterError("window_cutoffs must be sorted unique dates")
    if any(
        not stage_a_plan.boundary.start <= cutoff <= stage_a_plan.boundary.end
        for cutoff in window_cutoffs
    ):
        raise IncompletePrefilterError("every window cutoff must lie inside the boundary")
    if len(window_cutoffs) > 1:
        raise IncompletePrefilterError(
            "multi-window acquisition requires build_stage_b_plan_from_timeline; "
            "the single-prefilter path cannot represent more than one research cutoff"
        )
    if prefilter.selection_cutoff != window_cutoffs[0]:
        raise IncompletePrefilterError(
            "single-window prefilter cutoff must equal the declared window cutoff"
        )
    if interval_minutes != 5:
        raise ValueError("Stage B initially supports only the explicit 5-minute resolution")

    details = _stage_b_details_for_covering(stage_a_plan, _covering_from_prefilters((prefilter,)))
    candidates = _stage_b_candidates_from_details(details)
    trading_day_counts = {item.instrument_key: len(item.eligible_dates) for item in details}
    if not candidates:
        raise IncompletePrefilterError("complete Stage-A prefilter produced no Stage-B candidates")
    plan = plan_historical_batch(
        candidates=candidates,
        interval_minutes=interval_minutes,
        resolution="minutes",
        expected_rows_per_trading_day=expected_rows_per_trading_day,
        estimated_bytes_per_row=estimated_bytes_per_row,
        trading_day_counts=trading_day_counts,
        affordability_prefilter_applied=True,
    )
    return StageBPlan(
        boundary=stage_a_plan.boundary,
        source_manifest_sha256=stage_a_plan.source.manifest_sha256,
        stage_a_plan_fingerprint=stage_a_plan.fingerprint,
        prefilter_fingerprint=prefilter.fingerprint,
        candidates=tuple(candidates),
        details=tuple(details),
        interval_minutes=interval_minutes,
        expected_rows_per_trading_day=expected_rows_per_trading_day,
        estimated_bytes_per_row=estimated_bytes_per_row,
        approved_capital_rupees=stage_a_plan.approved_capital_rupees,
        thresholds=stage_a_plan.thresholds,
        formation_policy=stage_a_plan.formation_policy,
        rate_limit=rate_limit,
        universe_rule_version=stage_a_plan.universe_rule_version,
        adjustment_policy=stage_a_plan.adjustment_policy,
        estimated_requests=plan.estimated_requests,
        estimated_rows=plan.estimated_rows or 0,
        estimated_storage_bytes=plan.estimated_storage_bytes or 0,
        timeline_cutoffs=(prefilter.selection_cutoff,),
        timeline_prefilter_fingerprints=(prefilter.fingerprint,),
    )


def build_stage_b_plan_from_timeline(
    *,
    stage_a_plan: StageAPlan,
    prefilters: tuple[StageAPrefilterResult, ...],
    interval_minutes: int,
    expected_rows_per_trading_day: int,
    estimated_bytes_per_row: int,
    rate_limit: AcquisitionRateLimit,
) -> StageBPlan:
    """Create Stage B as the union of point-in-time eligible stocks/ranges.

    Each timeline prefilter covers one immutable research window cutoff using only
    evidence at or before that cutoff. The acquisition population is the union
    across all cutoffs: a stock eligible in an early window remains acquired for
    that early window even when it is later delisted, illiquid, or renamed. This
    must not degenerate to stocks passing on the final day alone.
    """
    if not prefilters:
        raise IncompletePrefilterError("Stage B timeline requires at least one prefilter")
    cutoffs = tuple(item.selection_cutoff for item in prefilters)
    if tuple(sorted(set(cutoffs))) != cutoffs:
        raise IncompletePrefilterError("timeline cutoffs must be sorted unique dates")
    for prefilter in prefilters:
        if not prefilter.complete:
            raise IncompletePrefilterError(
                "Stage B timeline is blocked until every window prefilter is complete"
            )
        if prefilter.stage_a_plan_fingerprint != stage_a_plan.fingerprint:
            raise IncompletePrefilterError("timeline prefilter is not bound to Stage-A plan")
        if prefilter.source_manifest_sha256 != stage_a_plan.source.manifest_sha256:
            raise IncompletePrefilterError("timeline prefilter source fingerprint mismatch")
        if prefilter.approved_capital_rupees != stage_a_plan.approved_capital_rupees:
            raise IncompletePrefilterError("timeline prefilter approved capital mismatch")
        if (
            not stage_a_plan.boundary.start
            <= prefilter.selection_cutoff
            <= stage_a_plan.boundary.end
        ):
            raise IncompletePrefilterError("timeline cutoff must lie inside the boundary")
    if interval_minutes != 5:
        raise ValueError("Stage B initially supports only the explicit 5-minute resolution")

    covering = _covering_from_prefilters(prefilters)
    if not covering:
        raise IncompletePrefilterError("timeline prefilters produced no eligible candidates")
    details = _stage_b_details_for_covering(stage_a_plan, covering)
    candidates = _stage_b_candidates_from_details(details)
    trading_day_counts = {item.instrument_key: len(item.eligible_dates) for item in details}
    plan = plan_historical_batch(
        candidates=candidates,
        interval_minutes=interval_minutes,
        resolution="minutes",
        expected_rows_per_trading_day=expected_rows_per_trading_day,
        estimated_bytes_per_row=estimated_bytes_per_row,
        trading_day_counts=trading_day_counts,
        affordability_prefilter_applied=True,
    )
    timeline_fingerprints = tuple(item.fingerprint for item in prefilters)
    if len(timeline_fingerprints) == 1:
        combined_fingerprint = timeline_fingerprints[0]
    else:
        combined_fingerprint = _fingerprint(
            {"timeline_prefilter_fingerprints": list(timeline_fingerprints)}
        )
    return StageBPlan(
        boundary=stage_a_plan.boundary,
        source_manifest_sha256=stage_a_plan.source.manifest_sha256,
        stage_a_plan_fingerprint=stage_a_plan.fingerprint,
        prefilter_fingerprint=combined_fingerprint,
        candidates=tuple(candidates),
        details=tuple(details),
        interval_minutes=interval_minutes,
        expected_rows_per_trading_day=expected_rows_per_trading_day,
        estimated_bytes_per_row=estimated_bytes_per_row,
        approved_capital_rupees=stage_a_plan.approved_capital_rupees,
        thresholds=stage_a_plan.thresholds,
        formation_policy=stage_a_plan.formation_policy,
        rate_limit=rate_limit,
        universe_rule_version=stage_a_plan.universe_rule_version,
        adjustment_policy=stage_a_plan.adjustment_policy,
        estimated_requests=plan.estimated_requests,
        estimated_rows=plan.estimated_rows or 0,
        estimated_storage_bytes=plan.estimated_storage_bytes or 0,
        timeline_cutoffs=cutoffs,
        timeline_prefilter_fingerprints=timeline_fingerprints,
    )


@dataclass(frozen=True)
class PITHistoricalAcquisitionPlan:
    stage_a: StageAPlan
    stage_b: StageBPlan | None = None

    @property
    def fingerprint(self) -> str:
        payload = {
            "stage_a": self.stage_a.deterministic_payload(),
            "stage_b": self.stage_b.deterministic_payload() if self.stage_b else None,
        }
        return _fingerprint(payload)

    def as_dict(self) -> dict[str, object]:
        return {
            "deterministic_fingerprint": self.fingerprint,
            "schema_version": PIT_ACQUISITION_SCHEMA,
            "stage_a": self.stage_a.as_dict(),
            "stage_b": self.stage_b.as_dict() if self.stage_b else None,
            "live_orders_called": False,
        }


def write_acquisition_plan(plan: PITHistoricalAcquisitionPlan, path: Path) -> None:
    """Persist a plan atomically; this function performs no market-data I/O."""

    _atomic_write_text(path, json.dumps(plan.as_dict(), indent=2, default=str) + "\n")


def write_stage_a_prefilter(prefilter: StageAPrefilterResult, path: Path) -> None:
    """Persist Stage-A decisions, including incomplete evidence, without network access."""

    _atomic_write_text(path, json.dumps(prefilter.as_dict(), indent=2, default=str) + "\n")
