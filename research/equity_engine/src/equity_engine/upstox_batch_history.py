from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import httpx
import pandas as pd

from .provenance import FINGERPRINT_SCHEMA
from .upstox_history import UpstoxHistoricalDataProvider

_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
UPSTOX_MINUTE_MAX_CALENDAR_DAYS = 28
UPSTOX_DAILY_MAX_CALENDAR_DAYS = 3650


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


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class _HttpGetter(Protocol):
    def get(self, url: str, **kwargs: object) -> httpx.Response: ...


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
                self.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(f"historical-data request failed after retries: {last_error}")


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
    resolution: str = "minutes"


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


@dataclass(frozen=True)
class HistoricalBatchRunResult:
    items: tuple[HistoricalBatchItemResult, ...]
    failures: tuple[str, ...]
    manifest_path: str

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


def historical_request_count(*, start: date, end: date, interval: str) -> int:
    if start > end:
        raise ValueError("start must be on or before end")
    return math.ceil(
        ((end - start).days + 1) / historical_request_limit_days(interval)
    )


def _chunk_count(start: date, end: date, *, resolution: str) -> int:
    return historical_request_count(start=start, end=end, interval=resolution)


def plan_historical_batch(
    *,
    candidates: Iterable[HistoricalBatchCandidate],
    interval_minutes: int = 5,
    expected_rows_per_trading_day: int | None = None,
    estimated_bytes_per_row: int | None = None,
    trading_day_counts: dict[str, int] | None = None,
    affordability_prefilter_applied: bool,
    resolution: str = "minutes",
) -> HistoricalBatchPlan:
    if interval_minutes < 1 or interval_minutes > 15:
        raise ValueError("interval_minutes must be between 1 and 15")
    if resolution not in {"minutes", "daily"}:
        raise ValueError("resolution must be 'minutes' or 'daily'")
    if resolution == "daily" and interval_minutes != 1:
        raise ValueError("daily resolution requires interval_minutes=1")
    candidate_list = tuple(candidates)
    keys = [candidate.instrument_key for candidate in candidate_list]
    if len(keys) != len(set(keys)):
        raise ValueError("historical batch candidates must have unique instrument keys")

    estimated_requests = sum(
        _chunk_count(item.start, item.end, resolution=resolution) for item in candidate_list
    )
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
        resolution=resolution,
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


class UpstoxHistoricalBatchDownloader:
    """Download only an explicit prefiltered candidate set; never discovers trade candidates itself."""

    def __init__(
        self,
        *,
        access_token: str,
        output_dir: Path,
        interval_minutes: int = 5,
        resolution: str = "minutes",
        min_request_interval_seconds: float,
        max_attempts: int,
        backoff_seconds: float,
        client: _HttpGetter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not access_token.strip():
            raise ValueError("access_token is required")
        if interval_minutes < 1 or interval_minutes > 15:
            raise ValueError("interval_minutes must be between 1 and 15")
        if resolution not in {"minutes", "daily"}:
            raise ValueError("resolution must be 'minutes' or 'daily'")
        if resolution == "daily" and interval_minutes != 1:
            raise ValueError("daily resolution requires interval_minutes=1")
        self.output_dir = output_dir.resolve()
        self.interval_minutes = interval_minutes
        self.resolution = resolution
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._owned_client: httpx.Client | None = None
        inner: _HttpGetter
        if client is None:
            self._owned_client = httpx.Client()
            inner = self._owned_client
        else:
            inner = client
        resilient = RateLimitedRetryClient(
            inner=inner,
            min_interval_seconds=min_request_interval_seconds,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
        self.provider = UpstoxHistoricalDataProvider(
            access_token=access_token,
            client=resilient,  # type: ignore[arg-type]
        )

    def close(self) -> None:
        if self._owned_client is not None:
            self._owned_client.close()

    def _paths(self, candidate: HistoricalBatchCandidate) -> tuple[Path, Path]:
        safe_key = candidate.instrument_key.replace("|", "_").replace("/", "_")
        root = self.output_dir / safe_key
        suffix = "daily" if self.resolution == "daily" else f"{self.interval_minutes}m"
        base = f"{candidate.start.isoformat()}_{candidate.end.isoformat()}_{suffix}"
        return root / f"{base}.parquet", root / f"{base}.manifest.json"

    def _existing_item(
        self,
        candidate: HistoricalBatchCandidate,
        parquet_path: Path,
        manifest_path: Path,
    ) -> HistoricalBatchItemResult | None:
        if parquet_path.exists() != manifest_path.exists():
            raise ValueError(
                f"partial historical artifact for {candidate.instrument_key}; "
                "resume requires both Parquet and manifest"
            )
        if not parquet_path.exists():
            return None
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        request = payload.get("request", {})
        expected = {
            "instrument_key": candidate.instrument_key,
            "symbol": candidate.symbol,
            "start": candidate.start.isoformat(),
            "end": candidate.end.isoformat(),
            "interval_minutes": self.interval_minutes,
        }
        if request.get("resolution", "minutes") != self.resolution:
            raise ValueError(
                f"existing historical artifact resolution differs for {candidate.instrument_key}"
            )
        if {key: request.get(key) for key in expected} != expected:
            raise ValueError(
                f"existing historical artifact request metadata differs for {candidate.instrument_key}"
            )
        frame = pd.read_parquet(parquet_path)
        expected_rows = payload.get("rows")
        if not isinstance(expected_rows, int) or expected_rows != len(frame):
            raise ValueError("existing historical artifact row count is not verified")
        expected_parquet_sha256 = payload.get("parquet_sha256")
        if expected_parquet_sha256 is not None:
            actual_parquet_sha256 = _sha256_file(parquet_path)
            if expected_parquet_sha256 != actual_parquet_sha256:
                raise ValueError(
                    f"existing historical artifact bytes changed for {candidate.instrument_key}"
                )
        fingerprint = payload.get("fingerprint_sha256")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("existing historical artifact fingerprint is missing")
        return HistoricalBatchItemResult(
            instrument_key=candidate.instrument_key,
            symbol=candidate.symbol,
            start=candidate.start,
            end=candidate.end,
            rows=len(frame),
            fingerprint=fingerprint,
            parquet=str(parquet_path),
            manifest=str(manifest_path),
            retrieval="cached",
        )

    def run(
        self,
        *,
        candidates: Iterable[HistoricalBatchCandidate],
        universe_rule_version: str,
        adjustment_policy: str,
    ) -> HistoricalBatchRunResult:
        candidate_list = tuple(candidates)
        if not candidate_list:
            raise ValueError("at least one explicitly prefiltered candidate is required")
        keys = [candidate.instrument_key for candidate in candidate_list]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate instrument keys must be unique")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        items: list[HistoricalBatchItemResult] = []
        failures: list[str] = []
        try:
            for candidate in candidate_list:
                parquet_path, manifest_path = self._paths(candidate)
                try:
                    existing = self._existing_item(candidate, parquet_path, manifest_path)
                    if existing is not None:
                        items.append(existing)
                        continue

                    fetch_kwargs = {
                        "instrument_token": candidate.instrument_key,
                        "symbol": candidate.symbol,
                        "exchange": "NSE",
                        "start": candidate.start,
                        "end": candidate.end,
                        "universe_rule_version": universe_rule_version,
                        "adjustment_policy": adjustment_policy,
                    }
                    if self.resolution == "daily":
                        dataset = self.provider.fetch_daily(**fetch_kwargs)
                    else:
                        dataset = self.provider.fetch_minutes(
                            interval_minutes=self.interval_minutes,
                            **fetch_kwargs,
                        )
                    parquet_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_write_parquet(parquet_path, dataset.frame)
                    request = {
                        "instrument_key": candidate.instrument_key,
                        "symbol": candidate.symbol,
                        "start": candidate.start.isoformat(),
                        "end": candidate.end.isoformat(),
                        "interval_minutes": self.interval_minutes,
                        "resolution": self.resolution,
                    }
                    payload = {
                        "schema_version": 1,
                        "request": request,
                        "market_data_manifest": asdict(dataset.manifest),
                        "fingerprint_sha256": dataset.fingerprint,
                        "fingerprint_schema": FINGERPRINT_SCHEMA,
                        "rows": len(dataset.frame),
                        "parquet_sha256": _sha256_file(parquet_path),
                        "rate_limit": {
                            "backoff_seconds": self.backoff_seconds,
                            "max_attempts": self.max_attempts,
                            "minimum_interval_seconds": self.min_request_interval_seconds,
                        },
                        "live_orders_called": False,
                    }
                    _atomic_write_text(
                        manifest_path,
                        json.dumps(payload, indent=2, default=str),
                    )
                    items.append(
                        HistoricalBatchItemResult(
                            instrument_key=candidate.instrument_key,
                            symbol=candidate.symbol,
                            start=candidate.start,
                            end=candidate.end,
                            rows=len(dataset.frame),
                            fingerprint=dataset.fingerprint,
                            parquet=str(parquet_path),
                            manifest=str(manifest_path),
                            retrieval="downloaded",
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - each instrument is recorded fail-closed
                    failures.append(f"{candidate.instrument_key}: {type(exc).__name__}: {exc}")
        finally:
            self.close()

        batch_manifest = self.output_dir / "upstox_history_batch_manifest.json"
        batch_payload = {
            "schema_version": 1,
            "fingerprint_schema": FINGERPRINT_SCHEMA,
            "generated_at": datetime.now(UTC).isoformat(),
            "interval_minutes": self.interval_minutes,
            "resolution": self.resolution,
            "rate_limit": {
                "backoff_seconds": self.backoff_seconds,
                "max_attempts": self.max_attempts,
                "minimum_interval_seconds": self.min_request_interval_seconds,
            },
            "items": [asdict(item) for item in items],
            "failures": failures,
            "summary": {
                "requested": len(candidate_list),
                "completed": len(items),
                "failed": len(failures),
                "live_orders_called": False,
            },
        }
        _atomic_write_text(batch_manifest, json.dumps(batch_payload, indent=2, default=str))
        return HistoricalBatchRunResult(
            items=tuple(items),
            failures=tuple(failures),
            manifest_path=str(batch_manifest),
        )
