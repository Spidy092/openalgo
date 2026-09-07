from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import gzip
import json
from typing import Mapping

import httpx

from .instrument_master import InstrumentMasterSnapshot, build_nse_equity_master


UPSTOX_NSE_BOD_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
UPSTOX_NSE_MIS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE_MIS.json.gz"
UPSTOX_SUSPENDED_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/suspended-instrument.json.gz"
)


@dataclass(frozen=True)
class InstrumentFilePayload:
    url: str
    rows: tuple[Mapping[str, object], ...]
    etag: str | None
    last_modified: str | None


class UpstoxPublicInstrumentFiles:
    """Fetch the public BOD/MIS/Suspended gzip JSON files documented by Upstox."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._client = client

    def fetch(self, url: str) -> InstrumentFilePayload:
        if self._client is None:
            response = httpx.get(url, timeout=self._timeout_seconds)
        else:
            response = self._client.get(url, timeout=self._timeout_seconds)
        response.raise_for_status()
        try:
            decoded = gzip.decompress(response.content)
            payload = json.loads(decoded.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Upstox gzip JSON payload from {url}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Upstox instrument payload from {url} is not a JSON array")
        rows: list[Mapping[str, object]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"non-object instrument row from {url}")
            rows.append(item)
        return InstrumentFilePayload(
            url=url,
            rows=tuple(rows),
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
        )

    def build_current_nse_equity_master(
        self,
        *,
        as_of_date: date,
        tick_size_scale_rupees_per_raw_unit: Decimal,
    ) -> InstrumentMasterSnapshot:
        bod = self.fetch(UPSTOX_NSE_BOD_URL)
        mis = self.fetch(UPSTOX_NSE_MIS_URL)
        suspended = self.fetch(UPSTOX_SUSPENDED_URL)
        return build_nse_equity_master(
            as_of_date=as_of_date,
            bod_rows=bod.rows,
            mis_rows=mis.rows,
            suspended_rows=suspended.rows,
            tick_size_scale_rupees_per_raw_unit=tick_size_scale_rupees_per_raw_unit,
        )
