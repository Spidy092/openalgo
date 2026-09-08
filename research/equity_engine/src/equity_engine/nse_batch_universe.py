from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Callable

import httpx
import pandas as pd

from .nse_calendar import CalendarEvidence
from .nse_daily_universe import NseDailyEquityUniverse, materialize_nse_daily_equity_universe
from .nse_mii_security import NseMiiSecurityMasterParser
from .nse_semantics import EffectiveDatedNseCmSemanticsPolicy


_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class NseBatchDaySummary:
    report_date: date
    source_url: str
    snapshot_sha256: str
    records: int
    eligible_records: int
    ineligible_records: int
    rejected_duplicate_rows: int
    raw_gzip: str
    universe_parquet: str
    retrieval: str


@dataclass(frozen=True)
class NseBatchFailure:
    report_date: date
    source_url: str
    error: str


@dataclass(frozen=True)
class NseBatchUniverseResult:
    start: date
    end: date
    normal_trading_dates: tuple[date, ...]
    holiday_dates: tuple[date, ...]
    excluded_special_session_dates: tuple[date, ...]
    days: tuple[NseBatchDaySummary, ...]
    failures: tuple[NseBatchFailure, ...]
    unique_instruments: int
    unique_eligible_instruments: int
    manifest_path: str

    @property
    def passed(self) -> bool:
        return not self.failures and len(self.days) == len(self.normal_trading_dates)


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _universe_frame(universe: NseDailyEquityUniverse) -> pd.DataFrame:
    rows = []
    for record in universe.records:
        rows.append(
            {
                "report_date": record.report_date.isoformat(),
                "instrument_key": record.instrument_key,
                "isin": record.isin,
                "symbol": record.symbol,
                "series": record.series,
                "name": record.name,
                "board_lot_quantity": record.board_lot_quantity,
                "tick_size_rupees": str(record.tick_size_rupees),
                "eligible": record.eligible,
                "listed_on_nse": record.trading_status.listed_on_nse,
                "normal_equity": record.trading_status.normal_equity,
                "tradeable_in_normal_market": record.trading_status.tradeable_in_normal_market,
                "raw_security_type_flag": record.raw_security_type_flag,
                "raw_permitted_to_trade": record.raw_permitted_to_trade,
                "raw_normal_market_status": record.raw_normal_market_status,
                "raw_normal_market_eligibility": record.raw_normal_market_eligibility,
                "source_url": record.source_url,
                "snapshot_sha256": record.snapshot_sha256,
                "source_row_number": record.source_row_number,
            }
        )
    return pd.DataFrame(rows)


class NseBatchUniverseBuilder:
    """Cache and materialize dated NSE CM universes with fail-closed evidence handling."""

    def __init__(
        self,
        *,
        output_dir: Path,
        semantics_policy: EffectiveDatedNseCmSemanticsPolicy,
        parser: NseMiiSecurityMasterParser | None = None,
        client: httpx.Client | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if timeout_seconds <= 0 or backoff_seconds < 0:
            raise ValueError("timeout/backoff values are invalid")
        self.output_dir = output_dir.resolve()
        self.semantics_policy = semantics_policy
        self.parser = parser or NseMiiSecurityMasterParser()
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep

    def _get(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.client is not None:
                    response = self.client.get(
                        url,
                        headers={"User-Agent": "openalgo-equity-research/0.1"},
                        timeout=self.timeout_seconds,
                        follow_redirects=True,
                    )
                else:
                    response = httpx.get(
                        url,
                        headers={"User-Agent": "openalgo-equity-research/0.1"},
                        timeout=self.timeout_seconds,
                        follow_redirects=True,
                    )
                if response.status_code == 404:
                    raise FileNotFoundError(
                        f"declared NSE trading-date snapshot returned HTTP 404: {url}"
                    )
                if response.status_code not in _TRANSIENT_STATUS:
                    response.raise_for_status()
                    return response.content
                last_error = RuntimeError(
                    f"transient NSE HTTP {response.status_code} for {url}"
                )
            except FileNotFoundError:
                raise
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
            if attempt < self.max_attempts:
                self.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(f"NSE snapshot download failed after retries: {last_error}")

    def _load_snapshot_bytes(
        self,
        report_date: date,
        *,
        refresh_existing: bool,
    ) -> tuple[bytes, str]:
        filename = f"NSE_CM_security_{report_date.strftime('%d%m%Y')}.csv.gz"
        raw_path = self.output_dir / "raw" / str(report_date.year) / filename
        hash_path = raw_path.with_suffix(raw_path.suffix + ".sha256")
        source_url = self.parser.source_url(report_date)
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        if raw_path.exists():
            cached = raw_path.read_bytes()
            cached_hash = sha256(cached).hexdigest()
            if hash_path.exists():
                recorded = hash_path.read_text(encoding="utf-8").strip()
                if recorded != cached_hash:
                    raise ValueError(
                        f"cached NSE payload hash mismatch for {report_date}: "
                        f"recorded={recorded} actual={cached_hash}"
                    )
            else:
                hash_path.write_text(cached_hash + "\n", encoding="utf-8")

            if not refresh_existing:
                return cached, "cached"

            downloaded = self._get(source_url)
            downloaded_hash = sha256(downloaded).hexdigest()
            if downloaded_hash != cached_hash:
                raise ValueError(
                    f"NSE archive payload changed for cached date {report_date}; "
                    "refusing to overwrite evidence"
                )
            return cached, "verified-cache"

        payload = self._get(source_url)
        digest = sha256(payload).hexdigest()
        raw_path.write_bytes(payload)
        hash_path.write_text(digest + "\n", encoding="utf-8")
        return payload, "downloaded"

    def _process_day(
        self,
        report_date: date,
        *,
        refresh_existing: bool,
    ) -> tuple[NseBatchDaySummary, set[str], set[str]]:
        filename = f"NSE_CM_security_{report_date.strftime('%d%m%Y')}.csv.gz"
        payload, retrieval = self._load_snapshot_bytes(
            report_date,
            refresh_existing=refresh_existing,
        )
        snapshot = self.parser.parse_bytes(payload, filename=filename)
        if snapshot.report_date != report_date:
            raise ValueError("NSE snapshot date does not match requested report date")
        universe = materialize_nse_daily_equity_universe(
            snapshot=snapshot,
            semantics_policy=self.semantics_policy,
        )

        day_dir = self.output_dir / "daily" / str(report_date.year)
        day_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = day_dir / f"NSE_CM_universe_{report_date.isoformat()}.parquet"
        _universe_frame(universe).to_parquet(parquet_path, index=False)

        raw_path = self.output_dir / "raw" / str(report_date.year) / filename
        summary = NseBatchDaySummary(
            report_date=report_date,
            source_url=universe.source_url,
            snapshot_sha256=universe.snapshot_sha256,
            records=len(universe.records),
            eligible_records=len(universe.eligible_records),
            ineligible_records=len(universe.ineligible_records),
            rejected_duplicate_rows=len(universe.rejected_duplicate_rows),
            raw_gzip=str(raw_path),
            universe_parquet=str(parquet_path),
            retrieval=retrieval,
        )
        all_keys = {record.instrument_key for record in universe.records}
        eligible_keys = {record.instrument_key for record in universe.eligible_records}
        return summary, all_keys, eligible_keys

    def build(
        self,
        *,
        calendar: CalendarEvidence,
        start: date,
        end: date,
        refresh_existing: bool = False,
    ) -> NseBatchUniverseResult:
        if start > end:
            raise ValueError("start must be on or before end")
        dates = tuple(day for day in calendar.trading_dates if start <= day <= end)
        if not dates:
            raise ValueError("calendar supplied no normal trading dates for requested range")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        day_summaries: list[NseBatchDaySummary] = []
        failures: list[NseBatchFailure] = []
        all_keys: set[str] = set()
        eligible_keys: set[str] = set()

        for report_date in dates:
            source_url = self.parser.source_url(report_date)
            try:
                summary, day_all, day_eligible = self._process_day(
                    report_date,
                    refresh_existing=refresh_existing,
                )
            except Exception as exc:
                failures.append(
                    NseBatchFailure(
                        report_date=report_date,
                        source_url=source_url,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            day_summaries.append(summary)
            all_keys.update(day_all)
            eligible_keys.update(day_eligible)

        manifest_path = self.output_dir / "nse_universe_manifest.json"
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "research_boundary": {"start": start.isoformat(), "end": end.isoformat()},
            "calendar": {
                "normal_trading_dates": [day.isoformat() for day in dates],
                "holiday_dates": [
                    day.isoformat() for day in calendar.holiday_dates if start <= day <= end
                ],
                "excluded_special_session_dates": [
                    day.isoformat()
                    for day in calendar.excluded_special_session_dates
                    if start <= day <= end
                ],
                "source_urls": list(calendar.source_urls),
            },
            "summary": {
                "status": (
                    "success"
                    if not failures and len(day_summaries) == len(dates)
                    else "failed"
                ),
                "trading_dates": len(dates),
                "completed_dates": len(day_summaries),
                "failed_dates": len(failures),
                "unique_instruments": len(all_keys),
                "unique_eligible_instruments": len(eligible_keys),
                "live_orders_called": False,
            },
            "days": [asdict(item) for item in day_summaries],
            "failures": [asdict(item) for item in failures],
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )

        return NseBatchUniverseResult(
            start=start,
            end=end,
            normal_trading_dates=dates,
            holiday_dates=tuple(
                day for day in calendar.holiday_dates if start <= day <= end
            ),
            excluded_special_session_dates=tuple(
                day
                for day in calendar.excluded_special_session_dates
                if start <= day <= end
            ),
            days=tuple(day_summaries),
            failures=tuple(failures),
            unique_instruments=len(all_keys),
            unique_eligible_instruments=len(eligible_keys),
            manifest_path=str(manifest_path),
        )
