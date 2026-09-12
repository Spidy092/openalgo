from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from urllib.parse import quote

import httpx
import pandas as pd

from .provenance import MarketDataManifest, dataframe_fingerprint, validate_ohlcv_frame

UPSTOX_HISTORY_DOC = "https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/"
UPSTOX_HISTORY_BASE = "https://api.upstox.com/v3/historical-candle"
_HISTORY_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
]


@dataclass(frozen=True)
class HistoricalChunk:
    """One raw Upstox response and its normalized candle frame.

    ``raw_payload`` is the exact response body received from the provider. The batch layer
    persists it write-once and uses ``raw_sha256`` as the immutable chunk identity.
    """

    start: date
    end: date
    frame: pd.DataFrame
    raw_payload: bytes
    request_url: str

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.raw_payload).hexdigest()


@dataclass(frozen=True)
class HistoricalDataset:
    frame: pd.DataFrame
    manifest: MarketDataManifest
    fingerprint: str
    chunks: tuple[HistoricalChunk, ...] = ()


def candles_from_raw_payload(raw_payload: bytes) -> list[list[object]]:
    """Decode one captured response without making a network call."""

    if not isinstance(raw_payload, bytes) or not raw_payload:
        raise ValueError("raw historical response must be non-empty bytes")
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError("raw historical response is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise ValueError("raw historical response does not have successful Upstox status")
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("candles"), list):
        raise TypeError("raw historical response did not contain data.candles")
    return data["candles"]


def frame_from_candles(candles: Iterable[object]) -> pd.DataFrame:
    """Normalize captured candle rows into the provider's canonical frame shape."""

    rows = list(candles)
    frame = pd.DataFrame(rows, columns=_HISTORY_COLUMNS)
    if frame.empty:
        frame = pd.DataFrame(columns=_HISTORY_COLUMNS).set_index("timestamp")
        frame.index = pd.DatetimeIndex([], name="timestamp")
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
    return frame.set_index("timestamp").sort_index()


class UpstoxHistoricalDataProvider:
    """Fetch Upstox V3 minute candles without hiding data-quality problems."""

    def __init__(
        self,
        *,
        access_token: str,
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
        min_request_interval_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not access_token:
            raise ValueError("access_token is required")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds cannot be negative")
        self._access_token = access_token
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._min_request_interval_seconds = min_request_interval_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None

    def _validate_request(
        self,
        *,
        start: date,
        end: date,
        interval_minutes: int,
    ) -> None:
        if interval_minutes < 1 or interval_minutes > 15:
            raise ValueError("this loader currently supports 1-15 minute V3 intervals")
        if start > end:
            raise ValueError("start must be on or before end")
        if start < date(2022, 1, 1):
            raise ValueError("Upstox documents minute history availability from January 2022")

    def fetch_minute_chunk(
        self,
        *,
        instrument_token: str,
        start: date,
        end: date,
        interval_minutes: int,
    ) -> HistoricalChunk:
        """Fetch one bounded raw chunk for the resumable batch layer."""

        self._validate_request(start=start, end=end, interval_minutes=interval_minutes)
        candles, raw_payload, request_url = self._fetch_chunk(
            instrument_token=instrument_token,
            start=start,
            end=end,
            interval_minutes=interval_minutes,
        )
        frame = frame_from_candles(candles)
        violations = validate_ohlcv_frame(frame)
        if violations and not frame.empty:
            raise ValueError("invalid Upstox historical dataset: " + "; ".join(violations))
        return HistoricalChunk(
            start=start,
            end=end,
            frame=frame,
            raw_payload=raw_payload,
            request_url=request_url,
        )

    def fetch_minutes(
        self,
        *,
        instrument_token: str,
        symbol: str,
        exchange: str,
        start: date,
        end: date,
        interval_minutes: int,
        universe_rule_version: str,
        adjustment_policy: str,
    ) -> HistoricalDataset:
        self._validate_request(start=start, end=end, interval_minutes=interval_minutes)

        chunks: list[HistoricalChunk] = []
        chunk_start = start
        # Upstox caps 1-15 minute retrieval at one month. 28-day inclusive windows remain
        # safely inside that documented maximum without making calendar-month assumptions.
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=27), end)
            chunks.append(
                self.fetch_minute_chunk(
                    instrument_token=instrument_token,
                    start=chunk_start,
                    end=chunk_end,
                    interval_minutes=interval_minutes,
                )
            )
            chunk_start = chunk_end + timedelta(days=1)

        non_empty = [chunk.frame for chunk in chunks if not chunk.frame.empty]
        if not non_empty:
            raise ValueError("Upstox returned no candles for requested range")
        frame = pd.concat(non_empty).sort_index()

        violations = validate_ohlcv_frame(frame)
        if violations:
            raise ValueError("invalid Upstox historical dataset: " + "; ".join(violations))

        manifest = MarketDataManifest(
            provider="upstox_v3",
            exchange=exchange,
            instrument_token=instrument_token,
            symbol=symbol,
            timezone=str(frame.index.tz),
            interval=f"{interval_minutes}m",
            timestamp_semantics="candle_start",
            start=frame.index[0].to_pydatetime(),
            end=frame.index[-1].to_pydatetime(),
            retrieved_at=datetime.now(UTC),
            adjustment_policy=adjustment_policy,
            universe_rule_version=universe_rule_version,
            source_reference=UPSTOX_HISTORY_DOC,
        )
        fingerprint = dataframe_fingerprint(frame, manifest)
        return HistoricalDataset(
            frame=frame,
            manifest=manifest,
            fingerprint=fingerprint,
            chunks=tuple(chunks),
        )

    def _fetch_chunk(
        self,
        *,
        instrument_token: str,
        start: date,
        end: date,
        interval_minutes: int,
    ) -> tuple[list[list[object]], bytes, str]:
        encoded_instrument = quote(instrument_token, safe="")
        url = (
            f"{UPSTOX_HISTORY_BASE}/{encoded_instrument}/minutes/{interval_minutes}/"
            f"{end.isoformat()}/{start.isoformat()}"
        )
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._access_token}",
        }

        response = self._request_with_retry(url, headers=headers)
        raw_payload = response.content
        candles = candles_from_raw_payload(raw_payload)
        return candles, raw_payload, url

    def _request_with_retry(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            self._wait_for_rate_limit()
            try:
                if self._client is not None:
                    response = self._client.get(url, headers=headers, timeout=self._timeout_seconds)
                else:
                    response = httpx.get(url, headers=headers, timeout=self._timeout_seconds)
            except httpx.RequestError:
                if attempt >= self._max_retries:
                    raise
                self._sleep(self._retry_backoff_seconds * (2**attempt))
                continue

            if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
                if attempt >= self._max_retries:
                    response.raise_for_status()
                self._sleep(self._retry_after(response, attempt))
                continue
            response.raise_for_status()
            return response
        raise AssertionError("unreachable retry loop")

    def _wait_for_rate_limit(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            remaining = self._min_request_interval_seconds - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._monotonic()

    def _retry_after(self, response: httpx.Response, attempt: int) -> float:
        raw = response.headers.get("Retry-After")
        try:
            if raw is not None:
                return max(float(raw), 0.0)
        except ValueError:
            pass
        return self._retry_backoff_seconds * (2**attempt)
