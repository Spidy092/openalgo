from __future__ import annotations

import io
import json
import os
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import httpx
import pandas as pd

from .historical_acquisition_plan import HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION
from .historical_validation import IntradaySessionRule, validate_intraday_dataset
from .provenance import (
    FINGERPRINT_SCHEMA,
    MarketDataManifest,
    bytes_sha256,
    canonical_sha256,
    dataframe_fingerprint,
)
from .upstox_history import (
    UPSTOX_HISTORY_BASE,
    HistoricalChunk,
    UpstoxHistoricalDataProvider,
    candles_from_raw_payload,
    frame_from_candles,
)

_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
UPSTOX_MINUTE_MAX_CALENDAR_DAYS = 28
UPSTOX_DAILY_MAX_CALENDAR_DAYS = 3650
_BATCH_SCHEMA_VERSION = 2
_CHUNK_SCHEMA_VERSION = 1
_STATE_VALUES = frozenset({"COMPLETE", "PARTIAL", "FAILED"})
_DIGEST_LENGTH = 64


class _HttpGetter(Protocol):
    def get(self, url: str, **kwargs: object) -> httpx.Response: ...


class ArtifactCorruptionError(ValueError):
    """Raised when an existing immutable artifact cannot be trusted."""


class ArtifactValidationError(ValueError):
    """Raised when captured data does not satisfy the canonical validation contract."""


class RateLimitedRetryClient:
    """GET-only wrapper for market-data APIs with pacing and bounded transient retries."""

    def __init__(
        self,
        *,
        inner: _HttpGetter,
        min_interval_seconds: float,
        max_attempts: int,
        backoff_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_interval_seconds < 0 or backoff_seconds < 0:
            raise ValueError("interval/backoff values cannot be negative")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.inner = inner
        self.min_interval_seconds = min_interval_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        self.monotonic = monotonic
        self._last_request_at: float | None = None
        self.request_count = 0
        self.retry_count = 0

    def _pace(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self.monotonic() - self._last_request_at
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            self.sleep(remaining)

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._pace()
            self.request_count += 1
            try:
                response = self.inner.get(url, **kwargs)
                self._last_request_at = self.monotonic()
                if response.status_code not in _TRANSIENT_STATUS:
                    return response
                last_error = RuntimeError(f"transient historical-data HTTP {response.status_code}")
            except httpx.TransportError as exc:
                self._last_request_at = self.monotonic()
                last_error = exc
            if attempt < self.max_attempts:
                self.retry_count += 1
                self.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(f"historical-data request failed after retries: {last_error}")


@dataclass(frozen=True)
class HistoricalAcquisitionEvidence:
    """Evidence identities and explicit session rules required for COMPLETE artifacts."""

    pit_fingerprint: str
    corporate_action_fingerprint: str
    acquisition_plan_fingerprint: str
    session_policy_identity: str
    expected_trade_dates: tuple[date, ...]
    session_rules: Mapping[date, IntradaySessionRule]

    def __post_init__(self) -> None:
        for name in (
            "pit_fingerprint",
            "corporate_action_fingerprint",
            "acquisition_plan_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != _DIGEST_LENGTH
                or any(char not in "0123456789abcdefABCDEF" for char in value)
            ):
                raise ValueError(f"{name} must be a 64-character hexadecimal digest")
        if not self.session_policy_identity.strip():
            raise ValueError("session_policy_identity is required")
        dates = tuple(sorted(set(self.expected_trade_dates)))
        if dates != self.expected_trade_dates or not dates:
            raise ValueError("expected_trade_dates must be sorted unique and non-empty")
        rules = dict(self.session_rules)
        if tuple(sorted(rules)) != tuple(rules):
            raise ValueError("session_rules must be ordered by date")
        if set(dates) != set(rules):
            raise ValueError("session rules must cover exactly expected_trade_dates")
        for trade_date, rule in rules.items():
            if not isinstance(trade_date, date) or not isinstance(rule, IntradaySessionRule):
                raise TypeError("session_rules must map dates to IntradaySessionRule records")
        object.__setattr__(self, "session_rules", rules)

    def as_dict(self) -> dict[str, object]:
        return {
            "pit_evidence_fingerprint": self.pit_fingerprint,
            "corporate_action_evidence_fingerprint": self.corporate_action_fingerprint,
            "acquisition_plan_fingerprint": self.acquisition_plan_fingerprint,
            "session_policy_identity": self.session_policy_identity,
            "expected_trade_dates": [item.isoformat() for item in self.expected_trade_dates],
            "session_rules": [
                {"trade_date": trade_date.isoformat(), "rule": rule.as_dict()}
                for trade_date, rule in self.session_rules.items()
            ],
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class HistoricalBatchCandidate:
    instrument_key: str
    symbol: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if not self.instrument_key.strip() or not self.symbol.strip():
            raise ValueError("candidate instrument_key and symbol are required")
        if self.start > self.end:
            raise ValueError("candidate start must be on or before end")


@dataclass(frozen=True)
class HistoricalBatchPlan:
    candidates: tuple[HistoricalBatchCandidate, ...]
    interval_minutes: int
    estimated_requests: int
    estimated_rows: int | None
    estimated_storage_bytes: int | None
    affordability_prefilter_applied: bool
    note: str


@dataclass(frozen=True)
class HistoricalBatchItemResult:
    instrument_key: str
    symbol: str
    start: date
    end: date
    rows: int
    fingerprint: str
    parquet: str
    manifest: str
    retrieval: str
    state: str = "COMPLETE"
    raw_sha256: str = ""
    covered_dates: tuple[str, ...] = ()
    continuous_session_rows: int = 0
    cas_auxiliary_rows: int = 0
    request_count: int = 0
    manifest_fingerprint: str = ""


@dataclass(frozen=True)
class HistoricalBatchRunResult:
    items: tuple[HistoricalBatchItemResult, ...]
    failures: tuple[str, ...]
    manifest_path: str
    requests: int = 0
    retries: int = 0
    raw_bytes: int = 0
    manifest_bytes: int = 0

    @property
    def passed(self) -> bool:
        return not self.failures and bool(self.items)


def historical_request_limit_days(interval: str) -> int:
    """Return the documented V3 maximum calendar span for an interval family."""

    normalized = interval.strip().lower()
    if normalized in {"1m", "5m", "15m", "minutes"}:
        return UPSTOX_MINUTE_MAX_CALENDAR_DAYS
    if normalized in {"daily", "1d", "days"}:
        return UPSTOX_DAILY_MAX_CALENDAR_DAYS
    raise ValueError(f"unsupported historical interval family: {interval}")


def historical_chunk_ranges(
    *, start: date, end: date, interval: str
) -> tuple[tuple[date, date], ...]:
    if start > end:
        raise ValueError("start must be on or before end")
    limit = historical_request_limit_days(interval)
    ranges: list[tuple[date, date]] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=limit - 1), end)
        ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return tuple(ranges)


def historical_request_count(*, start: date, end: date, interval: str) -> int:
    return len(historical_chunk_ranges(start=start, end=end, interval=interval))


def _chunk_count(start: date, end: date) -> int:
    return historical_request_count(start=start, end=end, interval="minutes")


def plan_historical_batch(
    *,
    candidates: Iterable[HistoricalBatchCandidate],
    interval_minutes: int = 5,
    expected_rows_per_trading_day: int | None = None,
    estimated_bytes_per_row: int | None = None,
    trading_day_counts: dict[str, int] | None = None,
    affordability_prefilter_applied: bool,
) -> HistoricalBatchPlan:
    if interval_minutes < 1 or interval_minutes > 15:
        raise ValueError("interval_minutes must be between 1 and 15")
    candidate_list = tuple(candidates)
    keys = [candidate.instrument_key for candidate in candidate_list]
    if len(keys) != len(set(keys)):
        raise ValueError("historical batch candidates must have unique instrument keys")

    estimated_requests = sum(_chunk_count(item.start, item.end) for item in candidate_list)
    estimated_rows: int | None = None
    estimated_storage: int | None = None
    if expected_rows_per_trading_day is not None:
        if expected_rows_per_trading_day <= 0:
            raise ValueError("expected_rows_per_trading_day must be positive")
        if trading_day_counts is None:
            raise ValueError("trading_day_counts are required for row estimation")
        estimated_rows = sum(
            trading_day_counts.get(item.instrument_key, 0) * expected_rows_per_trading_day
            for item in candidate_list
        )
        if estimated_bytes_per_row is not None:
            if estimated_bytes_per_row <= 0:
                raise ValueError("estimated_bytes_per_row must be positive")
            estimated_storage = estimated_rows * estimated_bytes_per_row

    note = (
        "candidate set was explicitly prefiltered before 5-minute acquisition"
        if affordability_prefilter_applied
        else (
            "NSE reference masters contain no historical market price; approved-capital affordability "
            "cannot be inferred safely here. Full acquisition is planning-only until an explicit "
            "prefilter candidate file is supplied."
        )
    )
    return HistoricalBatchPlan(
        candidates=candidate_list,
        interval_minutes=interval_minutes,
        estimated_requests=estimated_requests,
        estimated_rows=estimated_rows,
        estimated_storage_bytes=estimated_storage,
        affordability_prefilter_applied=affordability_prefilter_applied,
        note=note,
    )


def load_candidate_file(path: Path) -> tuple[HistoricalBatchCandidate, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("candidate file must contain a JSON list")
    candidates = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise TypeError("each candidate entry must be an object")
        candidates.append(
            HistoricalBatchCandidate(
                instrument_key=str(raw["instrument_key"]),
                symbol=str(raw["symbol"]),
                start=date.fromisoformat(str(raw["start"])),
                end=date.fromisoformat(str(raw["end"])),
            )
        )
    return tuple(candidates)


def load_acquisition_evidence(path: Path) -> HistoricalAcquisitionEvidence:
    """Load explicit PIT/CA/plan/session identities for an executable batch."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("acquisition evidence file must contain a JSON object")
    raw_rules = payload.get("session_rules")
    if not isinstance(raw_rules, list):
        raise TypeError("acquisition evidence session_rules must be a JSON list")
    rules: dict[date, IntradaySessionRule] = {}
    for raw in raw_rules:
        if not isinstance(raw, dict) or not isinstance(raw.get("rule"), dict):
            raise TypeError("each session rule must contain trade_date and rule")
        rule_payload = dict(raw["rule"])
        trade_date = date.fromisoformat(str(raw["trade_date"]))
        for serialized_name, constructor_name in (
            ("end_time_exclusive", "end_time"),
            ("auxiliary_end_time_exclusive", "auxiliary_end_time"),
        ):
            if serialized_name in rule_payload:
                rule_payload[constructor_name] = rule_payload.pop(serialized_name)
        for field in (
            "start_time",
            "end_time",
            "auxiliary_start_time",
            "auxiliary_end_time",
        ):
            if rule_payload.get(field) is not None:
                rule_payload[field] = dt_time.fromisoformat(str(rule_payload[field]))
        rule_payload["source_reference"] = str(rule_payload["source_reference"])
        rule_payload["rule_id"] = str(rule_payload["rule_id"])
        rule_payload["timezone"] = str(rule_payload["timezone"])
        rules[trade_date] = IntradaySessionRule(**rule_payload)
    return HistoricalAcquisitionEvidence(
        pit_fingerprint=str(payload["pit_evidence_fingerprint"]),
        corporate_action_fingerprint=str(payload["corporate_action_evidence_fingerprint"]),
        acquisition_plan_fingerprint=str(payload["acquisition_plan_fingerprint"]),
        session_policy_identity=str(payload["session_policy_identity"]),
        expected_trade_dates=tuple(
            date.fromisoformat(str(item)) for item in payload["expected_trade_dates"]
        ),
        session_rules=rules,
    )


def candidates_from_universe_manifest(
    manifest_path: Path,
) -> tuple[tuple[HistoricalBatchCandidate, ...], dict[str, int]]:
    """Create a planning-only full-universe candidate set from dated universe parquet files."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    days = manifest.get("days")
    if not isinstance(days, list) or not days:
        raise ValueError("NSE universe manifest contains no completed days")

    by_key: dict[str, dict[str, object]] = {}
    for day in days:
        parquet = Path(str(day["universe_parquet"]))
        if not parquet.exists() and not parquet.is_absolute():
            candidate_path = manifest_path.parent / parquet
            if candidate_path.exists():
                parquet = candidate_path
        frame = pd.read_parquet(
            parquet,
            columns=["instrument_key", "symbol", "report_date", "eligible"],
        )
        frame = frame[frame["eligible"]]
        for row in frame.itertuples(index=False):
            key = str(row.instrument_key)
            current = by_key.setdefault(
                key,
                {
                    "symbol": str(row.symbol),
                    "dates": [],
                },
            )
            current["dates"].append(date.fromisoformat(str(row.report_date)))

    candidates: list[HistoricalBatchCandidate] = []
    counts: dict[str, int] = {}
    for key, item in by_key.items():
        dates = sorted(set(item["dates"]))
        if not dates:
            continue
        candidates.append(
            HistoricalBatchCandidate(
                instrument_key=key,
                symbol=str(item["symbol"]),
                start=dates[0],
                end=dates[-1],
            )
        )
        counts[key] = len(dates)
    candidates.sort(key=lambda item: item.instrument_key)
    return tuple(candidates), counts


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime, dt_time)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_once(path: Path, payload: bytes) -> None:
    """Write an immutable artifact once, refusing any conflicting duplicate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise ArtifactCorruptionError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ArtifactCorruptionError(f"immutable artifact was concurrently changed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise ArtifactCorruptionError(f"{name} is not a SHA-256 digest")
    return value


@dataclass(frozen=True)
class AcquisitionPlanBinding:
    """Validated execution identity extracted from a canonical dry-run plan."""

    deterministic_fingerprint: str
    historical_acquisition_superset: tuple[str, ...]
    frozen_wfo_population: tuple[str, ...]
    requested_intervals: tuple[tuple[str, int, date, date, tuple[date, ...]], ...]
    requested_interval_minutes: tuple[int, ...]
    pit_source_fingerprints: tuple[str, ...]
    corporate_action_fingerprint: str
    session_policy_identities: tuple[str, ...]

    def validate_execution(
        self,
        *,
        candidates: tuple[HistoricalBatchCandidate, ...],
        interval_minutes: int,
        evidence: HistoricalAcquisitionEvidence,
    ) -> None:
        if interval_minutes not in self.requested_interval_minutes:
            raise ArtifactValidationError(
                f"acquisition plan does not authorize {interval_minutes}-minute requests"
            )
        if evidence.acquisition_plan_fingerprint != self.deterministic_fingerprint:
            raise ArtifactValidationError(
                "acquisition evidence fingerprint does not match the canonical acquisition plan"
            )
        if evidence.pit_fingerprint not in self.pit_source_fingerprints:
            raise ArtifactValidationError(
                "acquisition evidence PIT fingerprint is not bound to the canonical plan"
            )
        if evidence.corporate_action_fingerprint != self.corporate_action_fingerprint:
            raise ArtifactValidationError(
                "acquisition evidence corporate-action fingerprint does not match the plan"
            )
        if self.session_policy_identities and (
            evidence.session_policy_identity not in self.session_policy_identities
        ):
            raise ArtifactValidationError(
                "acquisition evidence session policy is not bound to the canonical plan"
            )

        planned = {
            (instrument_key, start, end): trade_dates
            for instrument_key, planned_interval, start, end, trade_dates in self.requested_intervals
            if planned_interval == interval_minutes
        }
        candidate_keys = {candidate.instrument_key for candidate in candidates}
        if candidate_keys != set(self.historical_acquisition_superset):
            raise ArtifactValidationError(
                "candidate population does not exactly match historical_acquisition_superset"
            )
        if not set(self.frozen_wfo_population).issubset(candidate_keys):
            raise ArtifactValidationError(
                "frozen_wfo_population is not represented in the candidate population"
            )
        actual = {}
        for candidate in candidates:
            key = (candidate.instrument_key, candidate.start, candidate.end)
            if key in actual:
                raise ArtifactValidationError("candidate population contains duplicate intervals")
            actual[key] = tuple(
                trade_date
                for trade_date in evidence.expected_trade_dates
                if candidate.start <= trade_date <= candidate.end
            )
        if set(actual) != set(planned):
            raise ArtifactValidationError(
                "candidate intervals do not exactly match canonical requested_intervals"
            )
        for key, trade_dates in actual.items():
            if trade_dates != planned[key]:
                raise ArtifactValidationError(
                    f"candidate dates do not match canonical requested_intervals for {key[0]}"
                )


def load_acquisition_plan_binding(path: Path) -> AcquisitionPlanBinding:
    """Load and verify a canonical HistoricalAcquisitionPlan JSON artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("acquisition plan must contain a JSON object")
    if payload.get("schema_version") != HISTORICAL_ACQUISITION_PLAN_SCHEMA_VERSION:
        raise ArtifactCorruptionError("unsupported canonical acquisition plan schema")
    fingerprint = _require_digest(
        "acquisition plan deterministic_fingerprint", payload.get("deterministic_fingerprint")
    )
    canonical_payload = dict(payload)
    canonical_payload.pop("plan_id", None)
    canonical_payload.pop("deterministic_fingerprint", None)
    if canonical_sha256(canonical_payload) != fingerprint:
        raise ArtifactCorruptionError("canonical acquisition plan fingerprint mismatch")
    if payload.get("plan_id") != f"hap_{fingerprint[:16]}":
        raise ArtifactCorruptionError("canonical acquisition plan id mismatch")
    if payload.get("mode") != "DRY_RUN" or payload.get("live_orders_called") is not False:
        raise ArtifactCorruptionError("canonical acquisition plan is not a dry-run artifact")

    def ordered_strings(name: str) -> tuple[str, ...]:
        value = payload.get(name)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) for item in value)
        ):
            raise ArtifactCorruptionError(f"canonical plan {name} is invalid")
        result = tuple(value)
        if tuple(sorted(set(result))) != result:
            raise ArtifactCorruptionError(f"canonical plan {name} is not sorted and unique")
        return result

    superset = ordered_strings("historical_acquisition_superset")
    frozen = ordered_strings("frozen_wfo_population")
    if not set(frozen).issubset(superset):
        raise ArtifactCorruptionError(
            "canonical plan frozen_wfo_population is outside its superset"
        )

    interval_minutes_payload = payload.get("requested_interval_minutes")
    if (
        not isinstance(interval_minutes_payload, list)
        or not interval_minutes_payload
        or any(
            not isinstance(item, int) or isinstance(item, bool) for item in interval_minutes_payload
        )
    ):
        raise ArtifactCorruptionError("canonical plan requested_interval_minutes is invalid")
    requested_interval_minutes = tuple(interval_minutes_payload)
    if tuple(sorted(set(requested_interval_minutes))) != requested_interval_minutes:
        raise ArtifactCorruptionError("canonical plan requested_interval_minutes is not canonical")

    raw_intervals = payload.get("requested_intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise ArtifactCorruptionError("canonical plan requested_intervals is invalid")
    intervals: list[tuple[str, int, date, date, tuple[date, ...]]] = []
    for raw in raw_intervals:
        if not isinstance(raw, dict):
            raise ArtifactCorruptionError("canonical plan requested interval is not an object")
        try:
            instrument_key = str(raw["instrument_key"])
            interval = int(raw["interval_minutes"])
            start = date.fromisoformat(str(raw["start"]))
            end = date.fromisoformat(str(raw["end"]))
            trade_dates_raw = raw["trade_dates"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactCorruptionError("canonical plan requested interval is malformed") from exc
        if not isinstance(trade_dates_raw, list):
            raise ArtifactCorruptionError("canonical plan interval trade_dates is invalid")
        trade_dates = tuple(date.fromisoformat(str(item)) for item in trade_dates_raw)
        if not instrument_key or start > end or not trade_dates:
            raise ArtifactCorruptionError("canonical plan requested interval has invalid bounds")
        if (
            tuple(sorted(set(trade_dates))) != trade_dates
            or trade_dates[0] != start
            or trade_dates[-1] != end
        ):
            raise ArtifactCorruptionError(
                "canonical plan requested interval dates are not canonical"
            )
        intervals.append((instrument_key, interval, start, end, trade_dates))
    interval_tuple = tuple(intervals)
    if tuple(sorted(interval_tuple)) != interval_tuple or len(set(interval_tuple)) != len(
        interval_tuple
    ):
        raise ArtifactCorruptionError("canonical plan requested_intervals are not canonical")
    if {item[0] for item in interval_tuple} != set(superset):
        raise ArtifactCorruptionError(
            "canonical plan intervals do not cover its acquisition superset"
        )

    sources = payload.get("evidence_sources")
    if not isinstance(sources, list):
        raise ArtifactCorruptionError("canonical plan evidence_sources is invalid")
    pit_fingerprints = tuple(
        sorted(
            {
                str(source["fingerprint"])
                for source in sources
                if isinstance(source, dict)
                and "pit" in str(source.get("evidence_type", "")).lower()
            }
        )
    )
    for item in pit_fingerprints:
        _require_digest("canonical PIT source fingerprint", item)
    ca_payload = payload.get("corporate_action_evidence")
    if not isinstance(ca_payload, dict):
        raise ArtifactCorruptionError("canonical plan corporate-action evidence is invalid")
    corporate_action_fingerprint = _require_digest(
        "canonical corporate-action fingerprint", ca_payload.get("fingerprint")
    )
    sessions = payload.get("session_evidence")
    if not isinstance(sessions, list) or not sessions:
        raise ArtifactCorruptionError("canonical plan session_evidence is invalid")
    session_policy_identities = tuple(
        sorted(
            {
                str(item["source_id"])
                for item in sessions
                if isinstance(item, dict) and str(item.get("source_id", "")).strip()
            }
        )
    )
    return AcquisitionPlanBinding(
        deterministic_fingerprint=fingerprint,
        historical_acquisition_superset=superset,
        frozen_wfo_population=frozen,
        requested_intervals=interval_tuple,
        requested_interval_minutes=requested_interval_minutes,
        pit_source_fingerprints=pit_fingerprints,
        corporate_action_fingerprint=corporate_action_fingerprint,
        session_policy_identities=session_policy_identities,
    )


def _safe_key(instrument_key: str) -> str:
    return instrument_key.replace("|", "_").replace("/", "_")


def _datetime_from_payload(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ArtifactCorruptionError("persisted datetime is timezone-naive")
    return parsed


def _market_manifest_from_payload(payload: Mapping[str, object]) -> MarketDataManifest:
    required = {
        "provider",
        "exchange",
        "instrument_token",
        "symbol",
        "timezone",
        "interval",
        "timestamp_semantics",
        "start",
        "end",
        "retrieved_at",
        "adjustment_policy",
        "universe_rule_version",
        "source_reference",
    }
    if not required.issubset(payload):
        raise ArtifactCorruptionError("persisted market-data manifest is incomplete")
    values = dict(payload)
    values["start"] = _datetime_from_payload(values["start"])
    values["end"] = _datetime_from_payload(values["end"])
    values["retrieved_at"] = _datetime_from_payload(values["retrieved_at"])
    return MarketDataManifest(**values)  # type: ignore[arg-type]


class UpstoxHistoricalBatchDownloader:
    """Download an explicit candidate set with immutable raw chunks and strict resume."""

    def __init__(
        self,
        *,
        access_token: str,
        output_dir: Path,
        interval_minutes: int = 5,
        min_request_interval_seconds: float = 0.15,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        client: _HttpGetter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not access_token.strip():
            raise ValueError("access_token is required")
        if interval_minutes < 1 or interval_minutes > 15:
            raise ValueError("interval_minutes must be between 1 and 15")
        self.output_dir = output_dir.resolve()
        self.interval_minutes = interval_minutes
        self._owned_client: httpx.Client | None = None
        inner: _HttpGetter
        if client is None:
            self._owned_client = httpx.Client()
            inner = self._owned_client
        else:
            inner = client
        self._resilient_client = RateLimitedRetryClient(
            inner=inner,
            min_interval_seconds=min_request_interval_seconds,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
        self.provider = UpstoxHistoricalDataProvider(
            access_token=access_token,
            client=self._resilient_client,  # type: ignore[arg-type]
        )

    def close(self) -> None:
        if self._owned_client is not None:
            self._owned_client.close()

    def _paths(self, candidate: HistoricalBatchCandidate) -> tuple[Path, Path]:
        safe_key = _safe_key(candidate.instrument_key)
        root = self.output_dir / safe_key
        base = f"{candidate.start.isoformat()}_{candidate.end.isoformat()}_{self.interval_minutes}m"
        return root / f"{base}.parquet", root / f"{base}.manifest.json"

    def _state_path(self, candidate: HistoricalBatchCandidate) -> Path:
        _, manifest_path = self._paths(candidate)
        return manifest_path.with_suffix(".state.json")

    def _chunk_paths(
        self,
        candidate: HistoricalBatchCandidate,
        chunk_start: date,
        chunk_end: date,
    ) -> tuple[Path, Path, Path, Path]:
        root = self.output_dir / _safe_key(candidate.instrument_key) / "chunks"
        base = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}_{self.interval_minutes}m"
        return (
            root / f"{base}.raw.json",
            root / f"{base}.raw.sha256",
            root / f"{base}.raw.manifest.json",
            root / f"{base}.state.json",
        )

    def _candidate_request(self, candidate: HistoricalBatchCandidate) -> dict[str, object]:
        return {
            "instrument_key": candidate.instrument_key,
            "symbol": candidate.symbol,
            "start": candidate.start.isoformat(),
            "end": candidate.end.isoformat(),
            "interval_minutes": self.interval_minutes,
        }

    def _candidate_rules(
        self,
        candidate: HistoricalBatchCandidate,
        evidence: HistoricalAcquisitionEvidence,
    ) -> dict[date, IntradaySessionRule]:
        rules = {
            trade_date: rule
            for trade_date, rule in evidence.session_rules.items()
            if candidate.start <= trade_date <= candidate.end
        }
        if not rules:
            raise ArtifactValidationError(
                f"no explicit session evidence covers {candidate.instrument_key}"
            )
        if set(rules) != {
            day for day in evidence.expected_trade_dates if candidate.start <= day <= candidate.end
        }:
            raise ArtifactValidationError(
                f"session evidence does not cover the requested interval for {candidate.instrument_key}"
            )
        for rule in rules.values():
            if rule.interval_minutes != self.interval_minutes:
                raise ArtifactValidationError(
                    "session evidence interval differs from acquisition interval"
                )
        return rules

    def _evidence_payload(self, evidence: HistoricalAcquisitionEvidence) -> dict[str, object]:
        return {
            "pit_evidence_fingerprint": evidence.pit_fingerprint,
            "corporate_action_evidence_fingerprint": evidence.corporate_action_fingerprint,
            "acquisition_plan_fingerprint": evidence.acquisition_plan_fingerprint,
            "session_policy_identity": evidence.session_policy_identity,
            "evidence_fingerprint": evidence.fingerprint(),
        }

    def _dataset_manifest(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        frame: pd.DataFrame,
        adjustment_policy: str,
        universe_rule_version: str = "",
        retrieved_at: datetime | None = None,
    ) -> MarketDataManifest:
        if frame.empty:
            raise ArtifactValidationError("cannot build a manifest for an empty dataset")
        return MarketDataManifest(
            provider="upstox_v3",
            exchange="NSE",
            instrument_token=candidate.instrument_key,
            symbol=candidate.symbol,
            timezone=str(frame.index.tz),
            interval=f"{self.interval_minutes}m",
            timestamp_semantics="candle_start",
            start=frame.index[0].to_pydatetime(),
            end=frame.index[-1].to_pydatetime(),
            retrieved_at=retrieved_at or datetime.now(UTC),
            adjustment_policy=adjustment_policy,
            universe_rule_version=universe_rule_version,
            source_reference="https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/",
        )

    def _validate_frame(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        frame: pd.DataFrame,
        evidence: HistoricalAcquisitionEvidence,
        adjustment_policy: str,
        universe_rule_version: str,
        reference_fingerprint: str | None = None,
        manifest: MarketDataManifest | None = None,
        session_rules: Mapping[date, IntradaySessionRule] | None = None,
    ) -> tuple[MarketDataManifest, object]:
        rules = (
            dict(session_rules)
            if session_rules is not None
            else self._candidate_rules(candidate, evidence)
        )
        if not rules:
            raise ArtifactValidationError("session validation requires at least one explicit rule")
        for rule in rules.values():
            if rule.interval_minutes != self.interval_minutes:
                raise ArtifactValidationError(
                    "session evidence interval differs from acquisition interval"
                )
        effective_manifest = manifest or self._dataset_manifest(
            candidate=candidate,
            frame=frame,
            adjustment_policy=adjustment_policy,
        )
        if effective_manifest.universe_rule_version != universe_rule_version:
            effective_manifest = MarketDataManifest(
                **{
                    **asdict(effective_manifest),
                    "universe_rule_version": universe_rule_version,
                }
            )
        fingerprint = reference_fingerprint or dataframe_fingerprint(frame, effective_manifest)
        report = validate_intraday_dataset(
            frame,
            effective_manifest,
            session_rules=rules,
            manifest_fingerprint_reference=fingerprint,
            fingerprint_schema=FINGERPRINT_SCHEMA,
        )
        if not report.passed:
            raise ArtifactValidationError(
                "historical data failed canonical validation: "
                + "; ".join(report.structural_violations)
                + "; "
                + "; ".join(
                    f"{item.trade_date}: missing={len(item.missing_expected_slots)} "
                    f"unexpected={len(item.unexpected_timestamps)}"
                    for item in report.per_day
                    if not item.passed
                )
            )
        expected_dates = tuple(sorted(rules))
        if tuple(report.trading_dates) != expected_dates:
            raise ArtifactValidationError("covered dates do not exactly match session evidence")
        return effective_manifest, report

    def _chunk_identity_payload(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        chunk_start: date,
        chunk_end: date,
        raw_sha: str,
        frame: pd.DataFrame,
        adjustment_policy: str,
        universe_rule_version: str,
        request_url: str,
        evidence: HistoricalAcquisitionEvidence,
    ) -> dict[str, object]:
        rows = len(frame)
        covered_dates = tuple(sorted({item.date() for item in frame.index})) if rows else ()
        return {
            "schema_version": _CHUNK_SCHEMA_VERSION,
            "artifact_type": "upstox_v3_historical_raw_response",
            "request": {
                **self._candidate_request(candidate),
                "chunk_start": chunk_start.isoformat(),
                "chunk_end": chunk_end.isoformat(),
            },
            "covered_dates": [item.isoformat() for item in covered_dates],
            "raw_sha256": raw_sha,
            "rows": rows,
            "timezone": str(frame.index.tz) if isinstance(frame.index, pd.DatetimeIndex) else None,
            "adjustment_policy": adjustment_policy,
            **self._evidence_payload(evidence),
            "source_api": "upstox_v3_historical_candle",
            "request_url": request_url,
            "universe_rule_version": universe_rule_version,
        }

    def _expected_chunk_request_url(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        chunk_start: date,
        chunk_end: date,
    ) -> str:
        return (
            f"{UPSTOX_HISTORY_BASE}/{quote(candidate.instrument_key, safe='')}/minutes/"
            f"{self.interval_minutes}/{chunk_end.isoformat()}/{chunk_start.isoformat()}"
        )

    def _chunk_manifest_payload(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        chunk_start: date,
        chunk_end: date,
        raw_path: Path,
        raw_sha: str,
        raw_payload: bytes,
        frame: pd.DataFrame,
        status: str,
        evidence: HistoricalAcquisitionEvidence,
        adjustment_policy: str,
        universe_rule_version: str,
        request_url: str,
        retrieved_at: datetime,
        validation: object | None,
    ) -> dict[str, object]:
        if status not in _STATE_VALUES:
            raise ValueError(f"unsupported artifact state {status}")
        rows = len(frame)
        raw_payload_for_identity = self._chunk_identity_payload(
            candidate=candidate,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            raw_sha=raw_sha,
            frame=frame,
            adjustment_policy=adjustment_policy,
            universe_rule_version=universe_rule_version,
            request_url=request_url,
            evidence=evidence,
        )
        return {
            **raw_payload_for_identity,
            "retrieval_timestamp": retrieved_at.isoformat(),
            "raw_path": str(raw_path),
            "raw_bytes": len(raw_payload),
            "min_timestamp": str(frame.index[0]) if rows else None,
            "max_timestamp": str(frame.index[-1]) if rows else None,
            "session_policy_identity": evidence.session_policy_identity,
            "universe_rule_version": universe_rule_version,
            "fingerprint_schema": FINGERPRINT_SCHEMA,
            "data_fingerprint": (
                dataframe_fingerprint(
                    frame,
                    self._dataset_manifest(
                        candidate=candidate,
                        frame=frame,
                        adjustment_policy=adjustment_policy,
                        universe_rule_version=universe_rule_version,
                        retrieved_at=retrieved_at,
                    ),
                )
                if rows
                else None
            ),
            "validation": validation.as_dict() if validation is not None else None,
            "status": status,
            "live_orders_called": False,
            "manifest_fingerprint": canonical_sha256(raw_payload_for_identity),
        }

    def _write_raw_once(
        self,
        *,
        raw_path: Path,
        sha_path: Path,
        raw_payload: bytes,
        raw_sha: str,
    ) -> None:
        _write_once(raw_path, raw_payload)
        _write_once(sha_path, f"{raw_sha}\n".encode("ascii"))

    def _validate_cached_chunk_metadata(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        chunk_start: date,
        chunk_end: date,
        evidence: HistoricalAcquisitionEvidence,
        adjustment_policy: str,
        universe_rule_version: str,
        raw_path: Path,
        raw_payload: bytes,
        raw_sha: str,
        payload: Mapping[str, object],
        state: Mapping[str, object],
        frame: pd.DataFrame,
    ) -> tuple[datetime, MarketDataManifest]:
        expected_request = {
            **self._candidate_request(candidate),
            "chunk_start": chunk_start.isoformat(),
            "chunk_end": chunk_end.isoformat(),
        }
        expected_url = self._expected_chunk_request_url(
            candidate=candidate,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        status = payload.get("status")
        if status not in _STATE_VALUES or state.get("status") != status:
            raise ArtifactCorruptionError("chunk manifest/state status mismatch")
        if payload.get("schema_version") != _CHUNK_SCHEMA_VERSION:
            raise ArtifactCorruptionError("unsupported chunk manifest schema")
        if state.get("schema_version") != _CHUNK_SCHEMA_VERSION:
            raise ArtifactCorruptionError("unsupported chunk state schema")
        if payload.get("artifact_type") != "upstox_v3_historical_raw_response":
            raise ArtifactCorruptionError("unexpected chunk artifact type")
        if payload.get("request") != expected_request or state.get("request") != expected_request:
            raise ArtifactCorruptionError("chunk request metadata mismatch")
        if payload.get("raw_path") != str(raw_path):
            raise ArtifactCorruptionError("chunk raw path metadata mismatch")
        if payload.get("raw_sha256") != raw_sha or state.get("raw_sha256") != raw_sha:
            raise ArtifactCorruptionError("chunk raw SHA-256 metadata mismatch")
        _require_digest("chunk raw SHA-256", raw_sha)
        if payload.get("request_url") != expected_url:
            raise ArtifactCorruptionError("chunk request URL metadata mismatch")
        if payload.get("source_api") != "upstox_v3_historical_candle":
            raise ArtifactCorruptionError("unexpected chunk source API")
        for key, expected in self._evidence_payload(evidence).items():
            if payload.get(key) != expected:
                raise ArtifactCorruptionError(f"chunk evidence mismatch for {key}")
        for key, expected in (
            ("adjustment_policy", adjustment_policy),
            ("universe_rule_version", universe_rule_version),
            ("fingerprint_schema", FINGERPRINT_SCHEMA),
            ("live_orders_called", False),
        ):
            if payload.get(key) != expected:
                raise ArtifactCorruptionError(f"chunk metadata mismatch for {key}")
            if key in {"adjustment_policy", "universe_rule_version"} and state.get(key) != expected:
                raise ArtifactCorruptionError(f"chunk state metadata mismatch for {key}")
        if state.get("live_orders_called") is not False:
            raise ArtifactCorruptionError("chunk state has live-order activity")

        rows = len(frame)
        covered_dates = sorted({timestamp.date().isoformat() for timestamp in frame.index})
        timezone = str(frame.index.tz) if isinstance(frame.index, pd.DatetimeIndex) else None
        if payload.get("rows") != rows or state.get("rows") != rows:
            raise ArtifactCorruptionError("chunk row-count metadata mismatch")
        if (
            payload.get("covered_dates") != covered_dates
            or state.get("covered_dates") != covered_dates
        ):
            raise ArtifactCorruptionError("chunk coverage metadata mismatch")
        if payload.get("timezone") != timezone:
            raise ArtifactCorruptionError("chunk timezone metadata mismatch")
        if payload.get("raw_bytes") != len(raw_payload):
            raise ArtifactCorruptionError("chunk raw-byte metadata mismatch")
        expected_minimum = str(frame.index[0]) if rows else None
        expected_maximum = str(frame.index[-1]) if rows else None
        if (
            payload.get("min_timestamp") != expected_minimum
            or payload.get("max_timestamp") != expected_maximum
        ):
            raise ArtifactCorruptionError("chunk timestamp bounds metadata mismatch")

        retrieved_at = _datetime_from_payload(payload.get("retrieval_timestamp"))
        market_manifest = (
            self._dataset_manifest(
                candidate=candidate,
                frame=frame,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
                retrieved_at=retrieved_at,
            )
            if rows
            else None
        )
        expected_data_fingerprint = (
            dataframe_fingerprint(frame, market_manifest) if market_manifest is not None else None
        )
        if payload.get("data_fingerprint") != expected_data_fingerprint:
            raise ArtifactCorruptionError("chunk data fingerprint mismatch")
        expected_identity = self._chunk_identity_payload(
            candidate=candidate,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            raw_sha=raw_sha,
            frame=frame,
            adjustment_policy=adjustment_policy,
            universe_rule_version=universe_rule_version,
            request_url=expected_url,
            evidence=evidence,
        )
        if payload.get("manifest_fingerprint") != canonical_sha256(expected_identity):
            raise ArtifactCorruptionError("chunk manifest fingerprint mismatch")
        _require_digest("chunk manifest fingerprint", payload.get("manifest_fingerprint"))
        if expected_data_fingerprint is not None:
            _require_digest("chunk data fingerprint", payload.get("data_fingerprint"))
        if market_manifest is None:
            raise ArtifactCorruptionError("empty chunk has no market-data manifest")
        return retrieved_at, market_manifest

    def _load_chunk(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        chunk_start: date,
        chunk_end: date,
        evidence: HistoricalAcquisitionEvidence,
        adjustment_policy: str,
        universe_rule_version: str,
    ) -> tuple[HistoricalChunk, dict[str, object]] | None:
        raw_path, sha_path, manifest_path, state_path = self._chunk_paths(
            candidate, chunk_start, chunk_end
        )
        any_existing = any(
            path.exists() for path in (raw_path, sha_path, manifest_path, state_path)
        )
        if not any_existing:
            return None
        if not all(path.exists() for path in (raw_path, sha_path, manifest_path, state_path)):
            raise ArtifactCorruptionError(f"incomplete immutable chunk artifact: {raw_path}")
        raw_payload = raw_path.read_bytes()
        raw_sha = bytes_sha256(raw_payload)
        recorded_sha = sha_path.read_text(encoding="utf-8").strip()
        if recorded_sha != raw_sha:
            raise ArtifactCorruptionError(f"raw response SHA-256 mismatch: {raw_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(state, dict):
            raise ArtifactCorruptionError(f"chunk metadata is not a JSON object: {manifest_path}")
        if payload.get("raw_sha256") != raw_sha:
            raise ArtifactCorruptionError(f"raw manifest SHA-256 mismatch: {manifest_path}")
        expected_request = {
            **self._candidate_request(candidate),
            "chunk_start": chunk_start.isoformat(),
            "chunk_end": chunk_end.isoformat(),
        }
        if payload.get("request") != expected_request:
            raise ArtifactCorruptionError(f"chunk request metadata mismatch: {manifest_path}")
        for key, expected in self._evidence_payload(evidence).items():
            if payload.get(key) != expected:
                raise ArtifactCorruptionError(f"chunk evidence mismatch for {key}: {manifest_path}")
        frame = frame_from_candles(candles_from_raw_payload(raw_payload))
        chunk = HistoricalChunk(
            start=chunk_start,
            end=chunk_end,
            frame=frame,
            raw_payload=raw_payload,
            request_url=str(payload["request_url"]),
        )
        _, manifest = self._validate_cached_chunk_metadata(
            candidate=candidate,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            evidence=evidence,
            adjustment_policy=adjustment_policy,
            universe_rule_version=universe_rule_version,
            raw_path=raw_path,
            raw_payload=raw_payload,
            raw_sha=raw_sha,
            payload=payload,
            state=state,
            frame=frame,
        )
        try:
            self._validate_frame(
                candidate=candidate,
                frame=frame,
                evidence=evidence,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
                manifest=manifest,
                reference_fingerprint=str(payload.get("data_fingerprint")),
                session_rules={
                    trade_date: rule
                    for trade_date, rule in evidence.session_rules.items()
                    if chunk_start <= trade_date <= chunk_end
                },
            )
            status = "COMPLETE"
        except Exception as exc:
            status = "PARTIAL" if len(frame) else "FAILED"
            state = {
                "schema_version": _CHUNK_SCHEMA_VERSION,
                "status": status,
                "request": expected_request,
                "raw_sha256": raw_sha,
                "rows": len(frame),
                "covered_dates": sorted({item.date().isoformat() for item in frame.index}),
                "adjustment_policy": adjustment_policy,
                "universe_rule_version": universe_rule_version,
                "error": f"{type(exc).__name__}: {exc}",
                "live_orders_called": False,
            }
            _write_json(state_path, state)
            payload["status"] = status
            payload["error"] = state["error"]
            _write_json(manifest_path, payload)
            raise ArtifactValidationError(state["error"]) from exc
        state = {
            "schema_version": _CHUNK_SCHEMA_VERSION,
            "status": status,
            "request": expected_request,
            "raw_sha256": raw_sha,
            "rows": len(frame),
            "covered_dates": sorted({item.date().isoformat() for item in frame.index}),
            "adjustment_policy": adjustment_policy,
            "universe_rule_version": universe_rule_version,
            "live_orders_called": False,
        }
        _write_json(state_path, state)
        payload["status"] = status
        _write_json(manifest_path, payload)
        return chunk, payload

    def _download_chunk(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        chunk_start: date,
        chunk_end: date,
        evidence: HistoricalAcquisitionEvidence,
        adjustment_policy: str,
        universe_rule_version: str,
    ) -> tuple[HistoricalChunk, dict[str, object]]:
        raw_path, sha_path, manifest_path, state_path = self._chunk_paths(
            candidate, chunk_start, chunk_end
        )
        chunk = self.provider.fetch_minute_chunk(
            instrument_token=candidate.instrument_key,
            start=chunk_start,
            end=chunk_end,
            interval_minutes=self.interval_minutes,
        )
        raw_sha = chunk.raw_sha256
        self._write_raw_once(
            raw_path=raw_path,
            sha_path=sha_path,
            raw_payload=chunk.raw_payload,
            raw_sha=raw_sha,
        )
        retrieved_at = datetime.now(UTC)
        try:
            manifest, validation = self._validate_frame(
                candidate=candidate,
                frame=chunk.frame,
                evidence=evidence,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
                session_rules={
                    trade_date: rule
                    for trade_date, rule in evidence.session_rules.items()
                    if chunk_start <= trade_date <= chunk_end
                },
            )
            del manifest
            status = "COMPLETE"
        except Exception as exc:
            status = "PARTIAL" if len(chunk.frame) else "FAILED"
            state = {
                "schema_version": _CHUNK_SCHEMA_VERSION,
                "status": status,
                "request": {
                    **self._candidate_request(candidate),
                    "chunk_start": chunk_start.isoformat(),
                    "chunk_end": chunk_end.isoformat(),
                },
                "raw_sha256": raw_sha,
                "rows": len(chunk.frame),
                "covered_dates": sorted({item.date().isoformat() for item in chunk.frame.index}),
                "adjustment_policy": adjustment_policy,
                "universe_rule_version": universe_rule_version,
                "error": f"{type(exc).__name__}: {exc}",
                "live_orders_called": False,
            }
            _write_json(state_path, state)
            payload = self._chunk_manifest_payload(
                candidate=candidate,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                raw_path=raw_path,
                raw_sha=raw_sha,
                raw_payload=chunk.raw_payload,
                frame=chunk.frame,
                status=status,
                evidence=evidence,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
                request_url=chunk.request_url,
                retrieved_at=retrieved_at,
                validation=None,
            )
            payload["error"] = state["error"]
            _write_json(manifest_path, payload)
            raise ArtifactValidationError(state["error"]) from exc

        payload = self._chunk_manifest_payload(
            candidate=candidate,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            raw_path=raw_path,
            raw_sha=raw_sha,
            raw_payload=chunk.raw_payload,
            frame=chunk.frame,
            status=status,
            evidence=evidence,
            adjustment_policy=adjustment_policy,
            universe_rule_version=universe_rule_version,
            request_url=chunk.request_url,
            retrieved_at=retrieved_at,
            validation=validation,
        )
        _write_json(manifest_path, payload)
        _write_json(
            state_path,
            {
                "schema_version": _CHUNK_SCHEMA_VERSION,
                "status": status,
                "request": payload["request"],
                "raw_sha256": raw_sha,
                "rows": len(chunk.frame),
                "covered_dates": [
                    item.isoformat() for item in sorted({ts.date() for ts in chunk.frame.index})
                ],
                "adjustment_policy": adjustment_policy,
                "universe_rule_version": universe_rule_version,
                "live_orders_called": False,
            },
        )
        return chunk, payload

    def _candidate_manifest_base(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        evidence: HistoricalAcquisitionEvidence,
        adjustment_policy: str,
        universe_rule_version: str,
        status: str,
        retrieved_at: datetime,
        chunks: list[dict[str, object]],
        failure: str | None = None,
    ) -> dict[str, object]:
        raw_hashes = [str(item["raw_sha256"]) for item in chunks]
        immutable_chunk_provenance = [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "retrieval_timestamp",
                    "validation",
                    "error",
                    "status",
                    "manifest_fingerprint",
                    "raw_path",
                }
            }
            for item in chunks
        ]
        identity = {
            "schema_version": _BATCH_SCHEMA_VERSION,
            "artifact_type": "upstox_v3_historical_dataset",
            "request": self._candidate_request(candidate),
            **self._evidence_payload(evidence),
            "adjustment_policy": adjustment_policy,
            "universe_rule_version": universe_rule_version,
            "raw_sha256": canonical_sha256({"chunks": raw_hashes}),
            "raw_artifacts_identity": immutable_chunk_provenance,
            "status": status,
        }
        payload: dict[str, object] = {
            **identity,
            "retrieval_timestamp": retrieved_at.isoformat(),
            "requested_start": candidate.start.isoformat(),
            "requested_end": candidate.end.isoformat(),
            "covered_dates": sorted(
                {
                    date.fromisoformat(item).isoformat()
                    for chunk in chunks
                    for item in chunk["covered_dates"]
                }
            ),
            "raw_bytes": sum(int(item["raw_bytes"]) for item in chunks),
            "rows": sum(int(item["rows"]) for item in chunks),
            "timezone": next((item["timezone"] for item in chunks if item["timezone"]), None),
            "min_timestamp": min(
                (str(item["min_timestamp"]) for item in chunks if item["min_timestamp"]),
                default=None,
            ),
            "max_timestamp": max(
                (str(item["max_timestamp"]) for item in chunks if item["max_timestamp"]),
                default=None,
            ),
            "pit_evidence_fingerprint": evidence.pit_fingerprint,
            "corporate_action_evidence_fingerprint": evidence.corporate_action_fingerprint,
            "acquisition_plan_fingerprint": evidence.acquisition_plan_fingerprint,
            "session_policy_identity": evidence.session_policy_identity,
            "fingerprint_schema": FINGERPRINT_SCHEMA,
            "failure": failure,
            "raw_artifacts": chunks,
            "manifest_fingerprint": canonical_sha256(identity),
            "live_orders_called": False,
        }
        return payload

    def _write_candidate_state(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        evidence: HistoricalAcquisitionEvidence,
        status: str,
        completed_chunks: list[dict[str, object]],
        adjustment_policy: str,
        universe_rule_version: str,
        failure: str | None = None,
    ) -> None:
        _write_json(
            self._state_path(candidate),
            {
                "schema_version": _BATCH_SCHEMA_VERSION,
                "status": status,
                "request": self._candidate_request(candidate),
                **self._evidence_payload(evidence),
                "adjustment_policy": adjustment_policy,
                "universe_rule_version": universe_rule_version,
                "completed_chunks": [
                    {
                        "start": item["request"]["chunk_start"],
                        "end": item["request"]["chunk_end"],
                        "raw_sha256": item["raw_sha256"],
                        "status": item["status"],
                    }
                    for item in completed_chunks
                ],
                "failure": failure,
                "live_orders_called": False,
            },
        )

    def _item_from_complete(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        parquet_path: Path,
        manifest_path: Path,
        payload: Mapping[str, object],
        frame: pd.DataFrame,
        report: object,
        retrieval: str,
        request_count: int,
    ) -> HistoricalBatchItemResult:
        return HistoricalBatchItemResult(
            instrument_key=candidate.instrument_key,
            symbol=candidate.symbol,
            start=candidate.start,
            end=candidate.end,
            rows=len(frame),
            fingerprint=str(payload["data_fingerprint"]),
            parquet=str(parquet_path),
            manifest=str(manifest_path),
            retrieval=retrieval,
            state="COMPLETE",
            raw_sha256=str(payload["raw_sha256"]),
            covered_dates=tuple(str(item) for item in payload["covered_dates"]),
            continuous_session_rows=sum(item.continuous_session_rows for item in report.per_day),
            cas_auxiliary_rows=sum(item.cas_auxiliary_rows for item in report.per_day),
            request_count=request_count,
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
        )

    def _reuse_complete(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        parquet_path: Path,
        manifest_path: Path,
        evidence: HistoricalAcquisitionEvidence,
        adjustment_policy: str,
        universe_rule_version: str,
    ) -> HistoricalBatchItemResult | None:
        if not parquet_path.exists() and not manifest_path.exists():
            return None
        if not manifest_path.exists():
            raise ArtifactCorruptionError(
                f"complete artifact is missing its manifest: {manifest_path}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ArtifactCorruptionError(
                f"complete artifact manifest is not an object: {manifest_path}"
            )
        if payload.get("status") != "COMPLETE":
            return None
        if not parquet_path.exists():
            raise ArtifactCorruptionError(
                f"complete artifact is missing its Parquet file: {parquet_path}"
            )
        state_path = self._state_path(candidate)
        if not state_path.exists():
            raise ArtifactCorruptionError(f"complete artifact is missing its state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ArtifactCorruptionError(f"complete artifact state is not an object: {state_path}")
        if state.get("schema_version") != _BATCH_SCHEMA_VERSION:
            raise ArtifactCorruptionError("unsupported complete artifact state schema")
        if state.get("status") != "COMPLETE" or state.get("failure") is not None:
            raise ArtifactCorruptionError("complete artifact state is not COMPLETE")
        if state.get("request") != self._candidate_request(candidate):
            raise ArtifactCorruptionError(f"complete artifact state request mismatch: {state_path}")
        for key, expected in self._evidence_payload(evidence).items():
            if payload.get(key) != expected or state.get(key) != expected:
                raise ArtifactCorruptionError(f"complete artifact evidence mismatch for {key}")
        for key, expected in (
            ("adjustment_policy", adjustment_policy),
            ("universe_rule_version", universe_rule_version),
            ("fingerprint_schema", FINGERPRINT_SCHEMA),
            ("live_orders_called", False),
        ):
            if payload.get(key) != expected or (
                key in {"adjustment_policy", "universe_rule_version"} and state.get(key) != expected
            ):
                raise ArtifactCorruptionError(f"complete artifact metadata mismatch for {key}")
        if state.get("live_orders_called") is not False:
            raise ArtifactCorruptionError("complete artifact state has live-order activity")
        if payload.get("request") != self._candidate_request(candidate):
            raise ArtifactCorruptionError(f"complete artifact request mismatch: {manifest_path}")
        if payload.get("failure") is not None:
            raise ArtifactCorruptionError("complete artifact carries a failure")

        parquet_bytes = parquet_path.read_bytes()
        recorded_parquet_sha = _require_digest("Parquet SHA-256", payload.get("parquet_sha256"))
        if recorded_parquet_sha != bytes_sha256(parquet_bytes):
            raise ArtifactCorruptionError(f"Parquet SHA-256 mismatch: {parquet_path}")
        frame = pd.read_parquet(io.BytesIO(parquet_bytes))
        retrieved_at = _datetime_from_payload(payload.get("retrieval_timestamp"))
        persisted_market_manifest = payload.get("market_data_manifest")
        if not isinstance(persisted_market_manifest, dict):
            raise ArtifactCorruptionError("complete artifact market-data manifest is invalid")
        market_manifest = _market_manifest_from_payload(persisted_market_manifest)
        expected_market_manifest = self._dataset_manifest(
            candidate=candidate,
            frame=frame,
            adjustment_policy=adjustment_policy,
            universe_rule_version=universe_rule_version,
            retrieved_at=retrieved_at,
        )
        if asdict(market_manifest) != asdict(expected_market_manifest):
            raise ArtifactCorruptionError("complete artifact market-data manifest mismatch")
        data_fingerprint = dataframe_fingerprint(frame, expected_market_manifest)
        if (
            _require_digest("dataset fingerprint", payload.get("data_fingerprint"))
            != data_fingerprint
        ):
            raise ArtifactCorruptionError(f"dataset fingerprint mismatch: {parquet_path}")
        frame_covered_dates = sorted({timestamp.date().isoformat() for timestamp in frame.index})
        frame_timezone = str(frame.index.tz) if isinstance(frame.index, pd.DatetimeIndex) else None
        if payload.get("rows") != len(frame):
            raise ArtifactCorruptionError("complete artifact row-count metadata mismatch")
        if payload.get("covered_dates") != frame_covered_dates:
            raise ArtifactCorruptionError("complete artifact coverage metadata mismatch")
        if payload.get("timezone") != frame_timezone:
            raise ArtifactCorruptionError("complete artifact timezone metadata mismatch")
        if payload.get("min_timestamp") != str(frame.index[0]) or payload.get(
            "max_timestamp"
        ) != str(frame.index[-1]):
            raise ArtifactCorruptionError("complete artifact timestamp bounds metadata mismatch")

        raw_artifacts = payload.get("raw_artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ArtifactCorruptionError("complete artifact has no raw chunk provenance")
        validated_chunk_payloads: list[dict[str, object]] = []
        expected_raw_hashes: list[str] = []
        for chunk_start, chunk_end in historical_chunk_ranges(
            start=candidate.start,
            end=candidate.end,
            interval="minutes",
        ):
            cached_chunk = self._load_chunk(
                candidate=candidate,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                evidence=evidence,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
            )
            if cached_chunk is None:
                raise ArtifactCorruptionError(
                    f"complete artifact is missing a cached chunk: {candidate.instrument_key} "
                    f"{chunk_start}:{chunk_end}"
                )
            _, chunk_payload = cached_chunk
            validated_chunk_payloads.append(chunk_payload)
            expected_raw_hashes.append(
                _require_digest("chunk raw SHA-256", chunk_payload.get("raw_sha256"))
            )
        if canonical_sha256({"chunks": expected_raw_hashes}) != payload.get("raw_sha256"):
            raise ArtifactCorruptionError("complete artifact aggregate raw SHA-256 mismatch")

        expected_identity = self._candidate_manifest_base(
            candidate=candidate,
            evidence=evidence,
            adjustment_policy=adjustment_policy,
            universe_rule_version=universe_rule_version,
            status="COMPLETE",
            retrieved_at=retrieved_at,
            chunks=validated_chunk_payloads,
        )
        for key, expected in expected_identity.items():
            if payload.get(key) != expected:
                raise ArtifactCorruptionError(f"complete artifact metadata mismatch for {key}")
        expected_completed_chunks = [
            {
                "start": item["request"]["chunk_start"],
                "end": item["request"]["chunk_end"],
                "raw_sha256": item["raw_sha256"],
                "status": "COMPLETE",
            }
            for item in validated_chunk_payloads
        ]
        if state.get("completed_chunks") != expected_completed_chunks:
            raise ArtifactCorruptionError("complete artifact state/chunk metadata mismatch")
        if payload.get("parquet_path") != str(parquet_path) or payload.get("manifest_path") != str(
            manifest_path
        ):
            raise ArtifactCorruptionError("complete artifact path metadata mismatch")

        _, report = self._validate_frame(
            candidate=candidate,
            frame=frame,
            evidence=evidence,
            adjustment_policy=adjustment_policy,
            universe_rule_version=universe_rule_version,
            reference_fingerprint=data_fingerprint,
            manifest=expected_market_manifest,
        )
        expected_continuous = sum(item.continuous_session_rows for item in report.per_day)
        expected_auxiliary = sum(item.cas_auxiliary_rows for item in report.per_day)
        if payload.get("continuous_session_rows") != expected_continuous:
            raise ArtifactCorruptionError(
                "complete artifact continuous-session row metadata mismatch"
            )
        if payload.get("cas_auxiliary_rows") != expected_auxiliary:
            raise ArtifactCorruptionError("complete artifact auxiliary-row metadata mismatch")
        return self._item_from_complete(
            candidate=candidate,
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            payload=payload,
            frame=frame,
            report=report,
            retrieval="cached",
            request_count=0,
        )

    def _run_candidate(
        self,
        *,
        candidate: HistoricalBatchCandidate,
        evidence: HistoricalAcquisitionEvidence,
        adjustment_policy: str,
        universe_rule_version: str,
    ) -> HistoricalBatchItemResult:
        parquet_path, manifest_path = self._paths(candidate)
        chunks: list[HistoricalChunk] = []
        chunk_payloads: list[dict[str, object]] = []
        try:
            cached = self._reuse_complete(
                candidate=candidate,
                parquet_path=parquet_path,
                manifest_path=manifest_path,
                evidence=evidence,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
            )
            if cached is not None:
                return cached

            self._write_candidate_state(
                candidate=candidate,
                evidence=evidence,
                status="PARTIAL",
                completed_chunks=[],
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
            )
            for chunk_start, chunk_end in historical_chunk_ranges(
                start=candidate.start,
                end=candidate.end,
                interval="minutes",
            ):
                cached_chunk = self._load_chunk(
                    candidate=candidate,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    evidence=evidence,
                    adjustment_policy=adjustment_policy,
                    universe_rule_version=universe_rule_version,
                )
                if cached_chunk is None:
                    chunk, payload = self._download_chunk(
                        candidate=candidate,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        evidence=evidence,
                        adjustment_policy=adjustment_policy,
                        universe_rule_version=universe_rule_version,
                    )
                else:
                    chunk, payload = cached_chunk
                if payload.get("status") != "COMPLETE":
                    raise ArtifactValidationError("chunk is not COMPLETE after validation")
                chunks.append(chunk)
                chunk_payloads.append(payload)
                self._write_candidate_state(
                    candidate=candidate,
                    evidence=evidence,
                    status="PARTIAL",
                    completed_chunks=chunk_payloads,
                    adjustment_policy=adjustment_policy,
                    universe_rule_version=universe_rule_version,
                )

            frame = pd.concat([item.frame for item in chunks]).sort_index()
            market_manifest, report = self._validate_frame(
                candidate=candidate,
                frame=frame,
                evidence=evidence,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
            )
            data_fingerprint = dataframe_fingerprint(frame, market_manifest)
            buffer = io.BytesIO()
            frame.to_parquet(buffer)
            parquet_bytes = buffer.getvalue()
            _write_once(parquet_path, parquet_bytes)
            retrieved_at = market_manifest.retrieved_at
            identity = self._candidate_manifest_base(
                candidate=candidate,
                evidence=evidence,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
                status="COMPLETE",
                retrieved_at=retrieved_at,
                chunks=chunk_payloads,
            )
            payload = {
                **identity,
                "data_fingerprint": data_fingerprint,
                "parquet_sha256": bytes_sha256(parquet_bytes),
                "parquet_path": str(parquet_path),
                "manifest_path": str(manifest_path),
                "market_data_manifest": asdict(market_manifest),
                "validation": report.as_dict(),
                "continuous_session_rows": sum(
                    item.continuous_session_rows for item in report.per_day
                ),
                "cas_auxiliary_rows": sum(item.cas_auxiliary_rows for item in report.per_day),
            }
            _write_json(manifest_path, payload)
            self._write_candidate_state(
                candidate=candidate,
                evidence=evidence,
                status="COMPLETE",
                completed_chunks=chunk_payloads,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
            )
            return self._item_from_complete(
                candidate=candidate,
                parquet_path=parquet_path,
                manifest_path=manifest_path,
                payload=payload,
                frame=frame,
                report=report,
                retrieval="downloaded",
                request_count=len(chunk_payloads),
            )
        except Exception as exc:
            has_raw_capture = bool(chunk_payloads) or any(
                self.output_dir.joinpath(_safe_key(candidate.instrument_key), "chunks").glob(
                    "*.raw.json"
                )
            )
            status = (
                "FAILED"
                if isinstance(exc, ArtifactCorruptionError)
                else ("PARTIAL" if has_raw_capture else "FAILED")
            )
            failure = f"{type(exc).__name__}: {exc}"
            self._write_candidate_state(
                candidate=candidate,
                evidence=evidence,
                status=status,
                completed_chunks=chunk_payloads,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
                failure=failure,
            )
            failure_payload = self._candidate_manifest_base(
                candidate=candidate,
                evidence=evidence,
                adjustment_policy=adjustment_policy,
                universe_rule_version=universe_rule_version,
                status=status,
                retrieved_at=datetime.now(UTC),
                chunks=chunk_payloads,
                failure=failure,
            )
            _write_json(manifest_path, failure_payload)
            raise

    def run(
        self,
        *,
        candidates: Iterable[HistoricalBatchCandidate],
        universe_rule_version: str,
        adjustment_policy: str,
        evidence: HistoricalAcquisitionEvidence,
    ) -> HistoricalBatchRunResult:
        candidate_list = tuple(candidates)
        if not candidate_list:
            raise ValueError("at least one explicitly prefiltered candidate is required")
        keys = [candidate.instrument_key for candidate in candidate_list]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate instrument keys must be unique")
        if not isinstance(evidence, HistoricalAcquisitionEvidence):
            raise TypeError("evidence must be HistoricalAcquisitionEvidence")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        items: list[HistoricalBatchItemResult] = []
        failures: list[str] = []
        try:
            for candidate in candidate_list:
                try:
                    items.append(
                        self._run_candidate(
                            candidate=candidate,
                            evidence=evidence,
                            adjustment_policy=adjustment_policy,
                            universe_rule_version=universe_rule_version,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - every candidate is persisted fail-closed
                    failures.append(f"{candidate.instrument_key}: {type(exc).__name__}: {exc}")
        finally:
            self.close()

        batch_manifest = self.output_dir / "upstox_history_batch_manifest.json"
        persisted_raw_bytes = sum(
            int(json.loads(Path(item.manifest).read_text(encoding="utf-8")).get("raw_bytes", 0))
            for item in items
        )
        batch_payload: dict[str, object] = {
            "schema_version": _BATCH_SCHEMA_VERSION,
            "fingerprint_schema": FINGERPRINT_SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "interval_minutes": self.interval_minutes,
            "items": [asdict(item) for item in items],
            "failures": failures,
            "summary": {
                "requested": len(candidate_list),
                "completed": len(items),
                "failed": len(failures),
                "requests": self._resilient_client.request_count,
                "retries": self._resilient_client.retry_count,
                "raw_bytes": persisted_raw_bytes,
                "manifest_bytes": 0,
                "live_orders_called": False,
            },
            "live_orders_called": False,
        }
        _write_json(batch_manifest, batch_payload)
        batch_payload["summary"] = {
            **batch_payload["summary"],
            "manifest_bytes": batch_manifest.stat().st_size,
        }
        _write_json(batch_manifest, batch_payload)
        return HistoricalBatchRunResult(
            items=tuple(items),
            failures=tuple(failures),
            manifest_path=str(batch_manifest),
            requests=self._resilient_client.request_count,
            retries=self._resilient_client.retry_count,
            raw_bytes=persisted_raw_bytes,
            manifest_bytes=batch_manifest.stat().st_size,
        )
