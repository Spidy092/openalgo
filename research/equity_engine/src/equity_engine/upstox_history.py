from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from urllib.parse import quote

import httpx
import pandas as pd

from .provenance import MarketDataManifest, dataframe_fingerprint, validate_ohlcv_frame

UPSTOX_HISTORY_DOC = "https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/"
UPSTOX_HISTORY_BASE = "https://api.upstox.com/v3/historical-candle"


@dataclass(frozen=True)
class HistoricalDataset:
    frame: pd.DataFrame
    manifest: MarketDataManifest
    fingerprint: str


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
        if interval_minutes < 1 or interval_minutes > 15:
            raise ValueError("this loader currently supports 1-15 minute V3 intervals")
        if start > end:
            raise ValueError("start must be on or before end")
        if start < date(2022, 1, 1):
            raise ValueError("Upstox documents minute history availability from January 2022")

        all_rows: list[list[object]] = []
        chunk_start = start
        # Upstox caps 1-15 minute retrieval at one month. 28-day inclusive windows remain
        # safely inside that documented maximum without making calendar-month assumptions.
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=27), end)
            all_rows.extend(
                self._fetch_chunk(
                    instrument_token=instrument_token,
                    start=chunk_start,
                    end=chunk_end,
                    interval_minutes=interval_minutes,
                )
            )
            chunk_start = chunk_end + timedelta(days=1)

        if not all_rows:
            raise ValueError("Upstox returned no candles for requested range")

        frame = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
        frame = frame.set_index("timestamp").sort_index()

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
        return HistoricalDataset(frame=frame, manifest=manifest, fingerprint=fingerprint)

    def _fetch_chunk(
        self,
        *,
        instrument_token: str,
        start: date,
        end: date,
        interval_minutes: int,
    ) -> list[list[object]]:
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
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"unexpected Upstox history response: {payload!r}")

        candles = payload.get("data", {}).get("candles")
        if candles is None:
            raise RuntimeError("Upstox history response did not contain data.candles")
        return candles

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
