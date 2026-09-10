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


def _thresholds_dict(thresholds: ResearchUniverseThresholds) -> dict[str, object]:
    return {
        "max_last_price_rupees": _decimal_text(thresholds.max_last_price_rupees),
        "min_affordable_quantity": thresholds.min_affordable_quantity,
        "min_median_daily_notional_proxy_rupees": _decimal_text(
            thresholds.min_median_daily_notional_proxy_rupees
        ),
        "min_median_daily_volume_shares": _decimal_text(
            thresholds.min_median_daily_volume_shares
        ),
        "min_observed_trading_days": thresholds.min_observed_trading_days,
    }


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
        trading_dates = tuple(date.fromisoformat(str(item)) for item in calendar["normal_trading_dates"])
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
            raise IncompleteUniverseManifestError("NSE universe day has invalid report_date") from exc
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
                {"date": day.isoformat(), "symbol": symbol}
                for day, symbol in self.symbol_by_date
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
    lookback_calendar_days: int
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
        if self.lookback_calendar_days < 1:
            raise ValueError("lookback_calendar_days must be positive and explicit")
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
            "lookback_calendar_days": self.lookback_calendar_days,
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
    lookback_calendar_days: int,
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
        request_start = first_date - pd.Timedelta(days=lookback_calendar_days)
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
        raise PITAcquisitionError("source universe has no eligible candidates in requested boundary")
    estimated_requests = sum(
        historical_request_count(start=item.start, end=item.end, interval="daily")
        for item in candidates
    )
    estimated_rows = sum(len(item.eligible_dates) for item in candidates) * estimated_rows_per_trading_day
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
        lookback_calendar_days=lookback_calendar_days,
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

    def as_dict(self) -> dict[str, object]:
        return {
            "affordable_quantity": self.affordable_quantity,
            "cash_required_rupees": _decimal_text(self.cash_required_rupees),
            "daily_dataset_fingerprint": self.daily_dataset_fingerprint,
            "eligible": self.eligible,
            "entry_charges_rupees": _decimal_text(self.entry_charges_rupees),
            "instrument_key": self.instrument_key,
            "median_daily_notional_proxy_rupees": _decimal_text(
                self.median_daily_notional_proxy_rupees
            ),
            "median_daily_volume_shares": _decimal_text(self.median_daily_volume_shares),
            "observed_trading_days": self.observed_trading_days,
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
) -> StageAPrefilterResult:
    """Evaluate Stage-A daily data without using prices after the formation decision."""

    if not stage_a_plan.boundary.start <= selection_cutoff <= stage_a_plan.boundary.end:
        raise ValueError("selection_cutoff must be inside the acquisition boundary")
    cutoff_timestamp = stage_a_plan.formation_policy.price_cutoff(selection_cutoff)
    if selection_as_of is None:
        selection_as_of = cutoff_timestamp
    else:
        selection_as_of = pd.Timestamp(selection_as_of)
        if selection_as_of.tzinfo is None or selection_as_of > cutoff_timestamp:
            raise ValueError("selection_as_of must be timezone-aware and not after the price cutoff")
    effective_cutoff = selection_as_of

    decisions: list[StageAPrefilterDecision] = []
    failures: list[str] = []
    for candidate in stage_a_plan.candidates:
        key = candidate.instrument_key
        fingerprint = daily_dataset_fingerprints.get(key)
        frame = daily_frames.get(key)
        if not fingerprint or frame is None:
            reason = f"missing Stage-A daily data or fingerprint for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                )
            )
            continue
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            reason = f"Stage-A daily timestamps must be timezone-aware for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key, symbol=candidate.symbol, fingerprint=fingerprint, reasons=[reason]
                )
            )
            continue
        violations = validate_ohlcv_frame(frame)
        if violations:
            reason = f"invalid Stage-A daily OHLCV for {key}: {'; '.join(violations)}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key, symbol=candidate.symbol, fingerprint=fingerprint, reasons=[reason]
                )
            )
            continue

        before_cutoff = frame.loc[frame.index <= effective_cutoff]
        if stage_a_plan.formation_policy.price_reference_policy == "prior_completed_session_close":
            expected_dates = tuple(day for day in candidate.eligible_dates if day < selection_cutoff)
        else:
            expected_dates = tuple(day for day in candidate.eligible_dates if day <= selection_cutoff)
        observed_dates = set(before_cutoff.index.date)
        missing_dates = tuple(day for day in expected_dates if day not in observed_dates)
        if missing_dates:
            reason = (
                f"missing Stage-A daily observations for {key}: "
                + ", ".join(day.isoformat() for day in missing_dates)
            )
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    observed_days=len(observed_dates.intersection(expected_dates)),
                )
            )
            continue
        if not expected_dates:
            reason = f"no point-in-time eligible membership date available by {selection_cutoff} for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key, symbol=candidate.symbol, fingerprint=fingerprint, reasons=[reason]
                )
            )
            continue

        reference_frame = before_cutoff
        if reference_frame.empty:
            reason = f"no Stage-A daily price is available by the formation cutoff for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key, symbol=candidate.symbol, fingerprint=fingerprint, reasons=[reason]
                )
            )
            continue

        expected_date_set = set(expected_dates)
        eligible_frame = before_cutoff.loc[
            [timestamp.date() in expected_date_set for timestamp in before_cutoff.index]
        ]
        reference_timestamp = reference_frame.index[-1]
        reference_price = Decimal(str(reference_frame.iloc[-1]["close"]))
        if not reference_price.is_finite() or reference_price <= 0:
            reason = f"reference price is invalid for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    reference_price=reference_price,
                    reference_timestamp=reference_timestamp,
                )
            )
            continue
        minimum_quantity = minimum_tradable_quantities.get(key)
        tick_policy = tick_policies.get(key)
        corporate_action = corporate_actions.get(key)
        if minimum_quantity is None or tick_policy is None or corporate_action is None:
            reason = f"missing Stage-A tick/lot/corporate-action evidence for {key}"
            failures.append(reason)
            decisions.append(
                _prefilter_rejection(
                    key=key,
                    symbol=candidate.symbol,
                    fingerprint=fingerprint,
                    reasons=[reason],
                    reference_price=reference_price,
                    reference_timestamp=reference_timestamp,
                    observed_days=len(expected_dates),
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
                symbol=candidate.symbol,
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


@dataclass(frozen=True)
class StageBPlan:
    boundary: PITResearchBoundary
    source_manifest_sha256: str
    stage_a_plan_fingerprint: str
    prefilter_fingerprint: str
    candidates: tuple[HistoricalBatchCandidate, ...]
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
            "universe_rule_version": self.universe_rule_version,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.deterministic_payload())

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


def build_stage_b_plan(
    *,
    stage_a_plan: StageAPlan,
    prefilter: StageAPrefilterResult,
    interval_minutes: int,
    expected_rows_per_trading_day: int,
    estimated_bytes_per_row: int,
    rate_limit: AcquisitionRateLimit,
) -> StageBPlan:
    """Create Stage B only from a complete, cryptographically bound Stage-A result."""

    if not prefilter.complete:
        raise IncompletePrefilterError("Stage B is blocked until Stage-A prefilter is complete")
    if prefilter.stage_a_plan_fingerprint != stage_a_plan.fingerprint:
        raise IncompletePrefilterError("prefilter is not bound to the supplied Stage-A plan")
    if prefilter.source_manifest_sha256 != stage_a_plan.source.manifest_sha256:
        raise IncompletePrefilterError("prefilter source universe fingerprint does not match Stage A")
    if prefilter.approved_capital_rupees != stage_a_plan.approved_capital_rupees:
        raise IncompletePrefilterError("prefilter approved capital differs from Stage A")
    if prefilter.selection_cutoff != stage_a_plan.boundary.end:
        raise IncompletePrefilterError(
            "Stage B requires a prefilter covering the complete requested acquisition boundary"
        )
    if interval_minutes != 5:
        raise ValueError("Stage B initially supports only the explicit 5-minute resolution")

    date_by_key = {item.instrument_key: item for item in stage_a_plan.candidates}
    candidates: list[HistoricalBatchCandidate] = []
    trading_day_counts: dict[str, int] = {}
    for decision in prefilter.decisions:
        if not decision.eligible:
            continue
        source_candidate = date_by_key.get(decision.instrument_key)
        if source_candidate is None:
            raise IncompletePrefilterError(
                f"prefilter contains an instrument absent from Stage A: {decision.instrument_key}"
            )
        dates = tuple(day for day in source_candidate.eligible_dates if day <= prefilter.selection_cutoff)
        candidates.append(
            HistoricalBatchCandidate(
                instrument_key=decision.instrument_key,
                symbol=decision.symbol,
                start=min(dates),
                end=max(dates),
            )
        )
        trading_day_counts[decision.instrument_key] = len(dates)
    candidates.sort(key=lambda item: item.instrument_key)
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
