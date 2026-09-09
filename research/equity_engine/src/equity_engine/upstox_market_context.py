from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import httpx

from .market_context import MarketSnapshot

UPSTOX_FULL_QUOTE_V3_URL = "https://api.upstox.com/v3/market-quote/quotes"
UPSTOX_FULL_QUOTE_V3_DOC = (
    "https://upstox.com/developer/api-documentation/get-full-market-quote-v3/"
)
MAX_INSTRUMENTS_PER_REQUEST = 500


@dataclass(frozen=True)
class QuoteBatchResult:
    """Sanitized result of a read-only quote batch.

    ``failures`` contains stable category names only.  Response bodies, headers and
    authentication material are intentionally not retained in the result.
    """

    requested_instrument_keys: tuple[str, ...]
    quotes: dict[str, dict[str, object]]
    failures: dict[str, str]
    request_count: int

    def __post_init__(self) -> None:
        requested = set(self.requested_instrument_keys)
        if len(requested) != len(self.requested_instrument_keys):
            raise ValueError("requested_instrument_keys contain duplicates")
        if set(self.quotes) - requested:
            raise ValueError("quote result contains an unrequested instrument")
        if set(self.failures) - requested:
            raise ValueError("quote failures contain an unrequested instrument")
        if set(self.quotes) & set(self.failures):
            raise ValueError("an instrument cannot be both a quote success and failure")
        if self.request_count < 0:
            raise ValueError("request_count cannot be negative")


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive parser boundary
        raise ValueError(f"invalid numeric field {field}: {value!r}") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


class UpstoxFullQuoteV3Client:
    """Read-only client for exchange snapshots used by the daily context engine."""

    def __init__(
        self,
        *,
        access_token: str,
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("access_token is required")
        self._access_token = access_token
        self._timeout_seconds = timeout_seconds
        self._client = client

    def fetch_by_instrument_token(self, instrument_keys: list[str]) -> dict[str, dict[str, object]]:
        if not instrument_keys:
            raise ValueError("instrument_keys cannot be empty")
        if len(set(instrument_keys)) != len(instrument_keys):
            raise ValueError("instrument_keys contain duplicates")

        by_token: dict[str, dict[str, object]] = {}
        for chunk in _chunks(instrument_keys, MAX_INSTRUMENTS_PER_REQUEST):
            params = {"instrument_key": ",".join(chunk)}
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self._access_token}",
            }
            if self._client is not None:
                response = self._client.get(
                    UPSTOX_FULL_QUOTE_V3_URL,
                    params=params,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            else:
                response = httpx.get(
                    UPSTOX_FULL_QUOTE_V3_URL,
                    params=params,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "success":
                raise RuntimeError(f"unexpected Upstox quote response: {payload!r}")
            data = payload.get("data")
            if not isinstance(data, dict):
                raise RuntimeError("Upstox quote response did not contain a data object")
            for quote in data.values():
                if not isinstance(quote, dict):
                    raise RuntimeError("Upstox quote entry is not an object")
                token = quote.get("instrument_token")
                if not isinstance(token, str) or not token:
                    raise RuntimeError("Upstox quote entry is missing instrument_token")
                if token in by_token:
                    raise RuntimeError(f"duplicate quote returned for {token}")
                by_token[token] = quote

        missing = [key for key in instrument_keys if key not in by_token]
        if missing:
            raise RuntimeError("Upstox did not return requested instruments: " + ", ".join(missing))
        return by_token

    def fetch_partial_by_instrument_token(
        self, instrument_keys: list[str]
    ) -> QuoteBatchResult:
        """Fetch deterministic quote batches while retaining per-key failure evidence.

        This method is for current-market measurement only.  It makes no trading calls and
        never stores the bearer token.  The pre-existing ``fetch_by_instrument_token`` remains
        strict for market-context construction.
        """

        if not instrument_keys:
            raise ValueError("instrument_keys cannot be empty")
        if len(set(instrument_keys)) != len(instrument_keys):
            raise ValueError("instrument_keys contain duplicates")

        requested = tuple(sorted(instrument_keys))
        by_token: dict[str, dict[str, object]] = {}
        failures: dict[str, str] = {}
        request_count = 0

        for chunk in _chunks(list(requested), MAX_INSTRUMENTS_PER_REQUEST):
            request_count += 1
            params = {"instrument_key": ",".join(chunk)}
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self._access_token}",
            }
            try:
                if self._client is not None:
                    response = self._client.get(
                        UPSTOX_FULL_QUOTE_V3_URL,
                        params=params,
                        headers=headers,
                        timeout=self._timeout_seconds,
                    )
                else:
                    response = httpx.get(
                        UPSTOX_FULL_QUOTE_V3_URL,
                        params=params,
                        headers=headers,
                        timeout=self._timeout_seconds,
                    )
                status_code = response.status_code
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPStatusError:
                reason = f"http_status_{status_code}"
                failures.update({key: reason for key in chunk})
                continue
            except (httpx.RequestError, ValueError, TypeError):
                failures.update({key: "transport_or_payload_error" for key in chunk})
                continue

            if not isinstance(payload, dict) or payload.get("status") != "success":
                failures.update({key: "unsuccessful_response" for key in chunk})
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                failures.update({key: "missing_data_object" for key in chunk})
                continue

            returned_in_chunk: set[str] = set()
            malformed = False
            for quote in data.values():
                if not isinstance(quote, dict):
                    malformed = True
                    continue
                token = quote.get("instrument_token")
                if not isinstance(token, str) or not token:
                    malformed = True
                    continue
                if token not in chunk:
                    continue
                if token in by_token or token in returned_in_chunk:
                    failures[token] = "duplicate_quote"
                    by_token.pop(token, None)
                    continue
                returned_in_chunk.add(token)
                by_token[token] = quote

            if malformed:
                for key in chunk:
                    if key not in by_token:
                        failures.setdefault(key, "malformed_quote")
            for key in chunk:
                if key not in by_token:
                    failures.setdefault(key, "missing_quote")

        for key in requested:
            if key not in by_token:
                failures.setdefault(key, "missing_quote")

        return QuoteBatchResult(
            requested_instrument_keys=requested,
            quotes=dict(sorted(by_token.items())),
            failures=dict(sorted(failures.items())),
            request_count=request_count,
        )

    def build_market_snapshot(
        self,
        *,
        nifty_instrument_key: str,
        vix_instrument_key: str,
        breadth_instrument_keys: list[str],
    ) -> MarketSnapshot:
        if not breadth_instrument_keys:
            raise ValueError("breadth_instrument_keys cannot be empty")
        requested = list(
            dict.fromkeys([nifty_instrument_key, vix_instrument_key, *breadth_instrument_keys])
        )
        quotes = self.fetch_by_instrument_token(requested)
        nifty = quotes[nifty_instrument_key]
        vix = quotes[vix_instrument_key]

        nifty_ohlc = nifty.get("ohlc")
        if not isinstance(nifty_ohlc, dict):
            raise RuntimeError("NIFTY quote is missing ohlc")

        timestamps: list[datetime] = []
        for token in requested:
            raw = quotes[token].get("timestamp")
            if not isinstance(raw, str):
                raise RuntimeError(f"quote {token} is missing timestamp")
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                raise RuntimeError(f"quote {token} timestamp is timezone-naive")
            timestamps.append(parsed)

        advancers = 0
        decliners = 0
        unchanged = 0
        for token in breadth_instrument_keys:
            quote = quotes[token]
            last_price = _decimal(quote.get("last_price"), field=f"{token}.last_price")
            previous_close = _decimal(
                quote.get("prev_close_price"), field=f"{token}.prev_close_price"
            )
            if last_price > previous_close:
                advancers += 1
            elif last_price < previous_close:
                decliners += 1
            else:
                unchanged += 1

        return MarketSnapshot(
            captured_at=max(timestamps),
            nifty_open=_decimal(nifty_ohlc.get("open"), field="nifty.ohlc.open"),
            nifty_last=_decimal(nifty.get("last_price"), field="nifty.last_price"),
            nifty_previous_close=_decimal(
                nifty.get("prev_close_price"), field="nifty.prev_close_price"
            ),
            india_vix_last=_decimal(vix.get("last_price"), field="vix.last_price"),
            india_vix_previous_close=_decimal(
                vix.get("prev_close_price"), field="vix.prev_close_price"
            ),
            advancers=advancers,
            decliners=decliners,
            unchanged=unchanged,
        )
