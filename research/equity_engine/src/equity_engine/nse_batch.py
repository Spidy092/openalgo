from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx

from .documented_costs import CurrentTermsNSEIntradayCostProvider
from .models import Exchange, Product
from .nse_daily_universe import (
    NseDailyEquityUniverse,
    NseDailyEquityUniverseRecord,
    materialize_nse_daily_equity_universe,
)
from .nse_mii_security import NseMiiSecurityMasterParser, NseMiiSecurityRow
from .nse_semantics import (
    EffectiveDatedNseCmSemanticsPolicy,
    interpret_nse_mii_equity_row,
    nse_cm_master_data_v15_semantics,
    tick_point_from_nse_mii_price_field,
)
from .nse_trading_calendar import (
    NSE_INITIAL_RESEARCH_END,
    NSE_RESEARCH_START,
    NseTradingCalendar,
    validate_initial_research_boundary,
)
from .sizing import max_affordable_buy_quantity

NSE_BATCH_SCHEMA_VERSION = 1
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class NseBatchAcquisitionError(RuntimeError):
    """Raised after the resumable manifest has recorded every failed date."""

    def __init__(self, message: str, *, manifest_path: Path) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path


class NseCachePayloadConflictError(RuntimeError):
    """A cache path already contains bytes different from the newly fetched payload."""


@dataclass(frozen=True)
class NseSnapshotFetch:
    report_date: date
    source_url: str
    status: str
    raw_path: Path | None
    snapshot: object | None
    snapshot_sha256: str | None
    attempts: int
    error: str | None = None


class NseMiiSnapshotDownloader:
    """Fetch and validate official NSE MII snapshots with content-addressed cache safety."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        parser: NseMiiSecurityMasterParser | None = None,
        client: httpx.Client | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
        retry_backoff_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        self.cache_dir = Path(cache_dir)
        self.parser = parser or NseMiiSecurityMasterParser()
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.sleep = sleep

    def acquire(self, report_date: date, *, refresh: bool = False) -> NseSnapshotFetch:
        filename = f"NSE_CM_security_{report_date.strftime('%d%m%Y')}.csv.gz"
        source_url = self.parser.source_url(report_date)
        raw_path = self.cache_dir / filename
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if raw_path.exists() and not refresh:
            return self._validate_cached(report_date, source_url, raw_path)

        payload, attempts, status, error = self._download(source_url)
        if payload is None:
            return NseSnapshotFetch(
                report_date=report_date,
                source_url=source_url,
                status=status,
                raw_path=raw_path if raw_path.exists() else None,
                snapshot=None,
                snapshot_sha256=None,
                attempts=attempts,
                error=error,
            )

        snapshot = self.parser.parse_bytes(payload, filename=filename)
        if snapshot.report_date != report_date:
            raise ValueError("NSE MII payload date does not match requested report date")
        payload_hash = hashlib.sha256(payload).hexdigest()

        if raw_path.exists():
            existing = raw_path.read_bytes()
            if existing != payload:
                raise NseCachePayloadConflictError(
                    f"refusing to overwrite differing NSE snapshot cache payload: {raw_path}"
                )
            return NseSnapshotFetch(
                report_date=report_date,
                source_url=source_url,
                status="cached_verified",
                raw_path=raw_path,
                snapshot=snapshot,
                snapshot_sha256=payload_hash,
                attempts=attempts,
            )

        self._write_once(raw_path, payload)
        return NseSnapshotFetch(
            report_date=report_date,
            source_url=source_url,
            status="downloaded",
            raw_path=raw_path,
            snapshot=snapshot,
            snapshot_sha256=payload_hash,
            attempts=attempts,
        )

    def _validate_cached(
        self, report_date: date, source_url: str, raw_path: Path
    ) -> NseSnapshotFetch:
        payload = raw_path.read_bytes()
        filename = raw_path.name
        snapshot = self.parser.parse_bytes(payload, filename=filename)
        if snapshot.report_date != report_date:
            raise ValueError(f"cached NSE snapshot filename/date mismatch: {raw_path}")
        return NseSnapshotFetch(
            report_date=report_date,
            source_url=source_url,
            status="cached",
            raw_path=raw_path,
            snapshot=snapshot,
            snapshot_sha256=hashlib.sha256(payload).hexdigest(),
            attempts=0,
        )

    def _download(self, url: str) -> tuple[bytes | None, int, str, str | None]:
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                if self.client is None:
                    response = httpx.get(
                        url,
                        headers={"User-Agent": "openalgo-equity-research/0.1"},
                        timeout=self.timeout_seconds,
                        follow_redirects=True,
                    )
                else:
                    response = self.client.get(
                        url,
                        headers={"User-Agent": "openalgo-equity-research/0.1"},
                        timeout=self.timeout_seconds,
                        follow_redirects=True,
                    )
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    return None, attempts, "error", type(exc).__name__
                self.sleep(self.retry_backoff_seconds * (2**attempt))
                continue

            if response.status_code == 404:
                return None, attempts, "missing", "official archive returned HTTP 404"
            if response.status_code in _TRANSIENT_HTTP_STATUSES:
                if attempt >= self.max_retries:
                    return (
                        None,
                        attempts,
                        "error",
                        f"official archive returned HTTP {response.status_code}",
                    )
                delay = self._retry_after(response, attempt)
                self.sleep(delay)
                continue
            if response.is_error:
                return (
                    None,
                    attempts,
                    "error",
                    f"official archive returned HTTP {response.status_code}",
                )
            return response.content, attempts, "downloaded", None
        raise AssertionError("unreachable retry loop")

    def _retry_after(self, response: httpx.Response, attempt: int) -> float:
        raw = response.headers.get("Retry-After")
        try:
            if raw is not None:
                return max(float(raw), 0.0)
        except ValueError:
            pass
        return self.retry_backoff_seconds * (2**attempt)

    @staticmethod
    def _write_once(path: Path, payload: bytes) -> None:
        temp = path.with_name(f".{path.name}.{hashlib.sha256(payload).hexdigest()}.part")
        try:
            with temp.open("xb") as handle:
                handle.write(payload)
            if path.exists():
                existing = path.read_bytes()
                if existing != payload:
                    raise NseCachePayloadConflictError(
                        f"refusing to overwrite differing NSE snapshot cache payload: {path}"
                    )
            else:
                os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()


@dataclass(frozen=True)
class NseBatchRunResult:
    manifest_path: Path
    manifest: Mapping[str, object]


def _record_to_dict(record: NseDailyEquityUniverseRecord) -> dict[str, object]:
    return {
        "report_date": record.report_date.isoformat(),
        "instrument_key": record.instrument_key,
        "isin": record.isin,
        "symbol": record.symbol,
        "series": record.series,
        "name": record.name,
        "board_lot_quantity": record.board_lot_quantity,
        "tick_size_rupees": str(record.tick_size_rupees),
        "raw_security_type_flag": record.raw_security_type_flag,
        "raw_permitted_to_trade": record.raw_permitted_to_trade,
        "raw_normal_market_status": record.raw_normal_market_status,
        "raw_normal_market_eligibility": record.raw_normal_market_eligibility,
        "eligibility": record.eligible,
        "status": {
            "listed_on_nse": record.trading_status.listed_on_nse,
            "normal_equity": record.trading_status.normal_equity,
            "tradeable_in_normal_market": record.trading_status.tradeable_in_normal_market,
            "source": record.trading_status.source,
        },
        "source_url": record.source_url,
        "snapshot_sha256": record.snapshot_sha256,
        "source_row_number": record.source_row_number,
    }


def _rejected_row_to_dict(
    row: NseMiiSecurityRow,
    *,
    semantics: object,
) -> dict[str, object]:
    status = interpret_nse_mii_equity_row(row, semantics=semantics)
    tick_point = tick_point_from_nse_mii_price_field(row, semantics=semantics)
    return {
        "instrument_key": row.instrument_key,
        "financial_instrument_id": row.financial_instrument_id,
        "symbol": row.symbol,
        "isin": row.isin,
        "raw_isin": row.raw_isin,
        "series": row.series,
        "tick_size_rupees": str(tick_point.tick_size_rupees),
        "tick_size_raw": str(row.bid_interval_raw),
        "eligibility": status.eligible,
        "status": {
            "listed_on_nse": status.listed_on_nse,
            "normal_equity": status.normal_equity,
            "tradeable_in_normal_market": status.tradeable_in_normal_market,
            "source": status.source,
        },
        "raw_permitted_to_trade": row.permitted_to_trade_raw,
        "raw_normal_market_status": row.normal_market_status_raw,
        "raw_normal_market_eligibility": row.normal_market_eligibility_raw,
        "source_url": row.source_url,
        "source_row_number": row.source_row_number,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def load_affordability_prices(path: Path) -> tuple[dict[str, Decimal], str]:
    """Load a caller-supplied point-in-time/reference price map without guessing prices."""

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    raw = json.loads(payload.decode("utf-8"))
    if isinstance(raw, dict) and "prices" in raw:
        raw = raw["prices"]
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((item.get("instrument_key"), item.get("price_rupees")) for item in raw)
    else:
        raise TypeError("affordability price file must be a JSON object or list")

    prices: dict[str, Decimal] = {}
    for instrument_key, value in items:
        if not isinstance(instrument_key, str) or not instrument_key.strip():
            raise ValueError("affordability price file contains a blank instrument_key")
        try:
            price = Decimal(str(value))
        except Exception as exc:
            raise ValueError(f"invalid affordability price for {instrument_key!r}") from exc
        if not price.is_finite() or price <= 0:
            raise ValueError(f"affordability price must be positive for {instrument_key!r}")
        if instrument_key in prices:
            raise ValueError(f"duplicate affordability price for {instrument_key}")
        prices[instrument_key] = price
    return prices, digest


def _affordability_summary(
    *,
    universes: list[NseDailyEquityUniverse],
    prices: Mapping[str, Decimal] | None,
    cash_limit: Decimal,
    pricing_date: date,
) -> dict[str, object]:
    eligible_dates: dict[str, list[date]] = {}
    for universe in universes:
        for record in universe.eligible_records:
            eligible_dates.setdefault(record.instrument_key, []).append(universe.report_date)
    unique_keys = sorted(eligible_dates)
    summary: dict[str, object] = {
        "unique_eligible_instruments": len(unique_keys),
        "affordability_cash_rupees": str(cash_limit),
        "affordability_minimum_quantity": 1,
        "affordability_pricing_date": pricing_date.isoformat(),
        "affordability_method": "current_documented_intraday_terms",
        # No price evidence means no instrument is safe to schedule.  Keep the count numeric for
        # machine-readable dry-run reports, and expose the unevaluable population separately.
        "candidate_count_after_affordability_filter": 0,
        "candidate_instrument_keys": [],
        "candidate_count_unknown_due_to_missing_price": len(unique_keys),
        "estimated_api_request_count_5m": 0,
        "estimated_rows_5m": 0,
        "estimated_storage_bytes": 0,
        "affordability_status": "price evidence not supplied",
    }
    if prices is None:
        return summary

    cost_provider = CurrentTermsNSEIntradayCostProvider(pricing_date=pricing_date)
    candidate_keys: list[str] = []
    for key in unique_keys:
        price = prices.get(key)
        if price is None:
            continue
        size = max_affordable_buy_quantity(
            instrument_token=key,
            exchange=Exchange.NSE,
            product=Product.INTRADAY,
            price=price,
            cash_limit=cash_limit,
            cost_provider=cost_provider,
        )
        if size.quantity >= 1:
            candidate_keys.append(key)

    estimated_requests = 0
    estimated_rows = 0
    for key in candidate_keys:
        dates = sorted(set(eligible_dates[key]))
        span_days = (dates[-1] - dates[0]).days + 1
        estimated_requests += (span_days + 27) // 28
        estimated_rows += len(dates) * 75
    summary.update(
        {
            "candidate_count_after_affordability_filter": len(candidate_keys),
            "candidate_instrument_keys": candidate_keys,
            "candidate_count_unknown_due_to_missing_price": len(
                set(unique_keys).difference(prices)
            ),
            "estimated_api_request_count_5m": estimated_requests,
            "estimated_rows_5m": estimated_rows,
            "estimated_storage_bytes": estimated_rows * 150,
            "affordability_status": "screened with supplied prices",
        }
    )
    return summary


class NseHistoricalUniverseBatch:
    """Resumable official NSE snapshot acquisition and daily universe materialization."""

    def __init__(
        self,
        *,
        output_dir: Path,
        downloader: NseMiiSnapshotDownloader | None = None,
        calendar: NseTradingCalendar | None = None,
        semantics_policy: EffectiveDatedNseCmSemanticsPolicy | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.audit_dir = self.output_dir / "daily"
        self.manifest_path = self.output_dir / "nse_universe_manifest.json"
        self.calendar = calendar or NseTradingCalendar()
        self.downloader = downloader or NseMiiSnapshotDownloader(cache_dir=self.raw_dir)
        self.semantics_policy = semantics_policy or EffectiveDatedNseCmSemanticsPolicy(
            [nse_cm_master_data_v15_semantics()]
        )

    def run(
        self,
        *,
        start: date = NSE_RESEARCH_START,
        end: date = NSE_INITIAL_RESEARCH_END,
        resume: bool = True,
        refresh: bool = False,
        affordability_prices: Mapping[str, Decimal] | None = None,
        affordability_price_file_sha256: str | None = None,
        cash_limit: Decimal = Decimal(1000),
        affordability_pricing_date: date = date(2026, 9, 7),
    ) -> NseBatchRunResult:
        validate_initial_research_boundary(start, end)
        if cash_limit <= 0:
            raise ValueError("cash_limit must be positive")
        trading_dates = self.calendar.trading_dates(start, end)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = self._load_existing_manifest() if resume else {}
        if existing and (
            existing.get("start_date") != start.isoformat()
            or existing.get("end_date") != end.isoformat()
        ):
            raise ValueError(
                "existing NSE batch manifest covers a different date range; use a new output "
                "directory or explicitly remove the old run"
            )
        date_entries: dict[str, dict[str, object]] = dict(existing.get("dates", {}))
        universes: list[NseDailyEquityUniverse] = []
        failures: list[dict[str, object]] = []

        for trade_date in trading_dates:
            key = trade_date.isoformat()
            try:
                acquired = self.downloader.acquire(trade_date, refresh=refresh)
                if acquired.snapshot is None:
                    failures.append(
                        {
                            "report_date": key,
                            "status": acquired.status,
                            "source_url": acquired.source_url,
                            "error": acquired.error,
                        }
                    )
                    date_entries[key] = {
                        "status": acquired.status,
                        "source_url": acquired.source_url,
                        "attempts": acquired.attempts,
                        "error": acquired.error,
                    }
                    self._write_manifest(
                        start=start,
                        end=end,
                        trading_dates=trading_dates,
                        date_entries=date_entries,
                        universes=universes,
                        failures=failures,
                        affordability_prices=affordability_prices,
                        affordability_price_file_sha256=affordability_price_file_sha256,
                        cash_limit=cash_limit,
                        affordability_pricing_date=affordability_pricing_date,
                    )
                    continue

                universe = materialize_nse_daily_equity_universe(
                    snapshot=acquired.snapshot,
                    semantics_policy=self.semantics_policy,
                )
                audit_path = self.audit_dir / f"{key}.json"
                semantics = self.semantics_policy.resolve(trade_date)
                _write_json(audit_path, self._daily_audit(universe, semantics=semantics))
                universes.append(universe)
                date_entries[key] = {
                    "status": "complete",
                    "source_url": universe.source_url,
                    "snapshot_sha256": universe.snapshot_sha256,
                    "raw_snapshot": str(acquired.raw_path.relative_to(self.output_dir)),
                    "audit_file": str(audit_path.relative_to(self.output_dir)),
                    "records": len(universe.records),
                    "eligible_records": len(universe.eligible_records),
                    "ineligible_records": len(universe.ineligible_records),
                    "rejected_duplicate_rows": len(universe.rejected_duplicate_rows),
                    "attempts": acquired.attempts,
                }
            except Exception as exc:  # noqa: BLE001 - every date must be recorded fail-closed
                failures.append(
                    {
                        "report_date": key,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                date_entries[key] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self._write_manifest(
                start=start,
                end=end,
                trading_dates=trading_dates,
                date_entries=date_entries,
                universes=universes,
                failures=failures,
                affordability_prices=affordability_prices,
                affordability_price_file_sha256=affordability_price_file_sha256,
                cash_limit=cash_limit,
                affordability_pricing_date=affordability_pricing_date,
            )

        manifest = self._write_manifest(
            start=start,
            end=end,
            trading_dates=trading_dates,
            date_entries=date_entries,
            universes=universes,
            failures=failures,
            affordability_prices=affordability_prices,
            affordability_price_file_sha256=affordability_price_file_sha256,
            cash_limit=cash_limit,
            affordability_pricing_date=affordability_pricing_date,
        )
        if failures:
            raise NseBatchAcquisitionError(
                f"NSE historical acquisition failed for {len(failures)} date(s); see {self.manifest_path}",
                manifest_path=self.manifest_path,
            )
        return NseBatchRunResult(manifest_path=self.manifest_path, manifest=manifest)

    def _daily_audit(
        self, universe: NseDailyEquityUniverse, *, semantics: object
    ) -> dict[str, object]:
        return {
            "schema_version": NSE_BATCH_SCHEMA_VERSION,
            "report_date": universe.report_date.isoformat(),
            "source_url": universe.source_url,
            "snapshot_sha256": universe.snapshot_sha256,
            "semantics_source": universe.semantics_source,
            "records": [_record_to_dict(record) for record in universe.records],
            "rejected_duplicate_rows": [
                _rejected_row_to_dict(row, semantics=semantics)
                for row in universe.rejected_duplicate_rows
            ],
            "live_orders_called": False,
        }

    def _write_manifest(
        self,
        *,
        start: date,
        end: date,
        trading_dates: tuple[date, ...],
        date_entries: Mapping[str, Mapping[str, object]],
        universes: list[NseDailyEquityUniverse],
        failures: list[Mapping[str, object]],
        affordability_prices: Mapping[str, Decimal] | None,
        affordability_price_file_sha256: str | None,
        cash_limit: Decimal,
        affordability_pricing_date: date,
    ) -> dict[str, object]:
        summary = _affordability_summary(
            universes=universes,
            prices=affordability_prices,
            cash_limit=cash_limit,
            pricing_date=affordability_pricing_date,
        )
        manifest: dict[str, object] = {
            "schema_version": NSE_BATCH_SCHEMA_VERSION,
            "pipeline": "nse_historical_universe",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "verified_semantics_boundary": NSE_RESEARCH_START.isoformat(),
            "initial_research_end": NSE_INITIAL_RESEARCH_END.isoformat(),
            "cas_regime": "excluded; August 2026 requires a separate effective-dated validation",
            "calendar": self.calendar.describe(),
            "semantics_sources": [contract.source for contract in self.semantics_policy.contracts],
            "trading_dates": [item.isoformat() for item in trading_dates],
            "dates": {key: date_entries[key] for key in sorted(date_entries)},
            "completed_dates": sum(
                1 for item in date_entries.values() if item.get("status") == "complete"
            ),
            "failed_dates": len(failures),
            "failures": list(failures),
            "price_evidence_sha256": affordability_price_file_sha256,
            "dry_run_screen": summary,
            "live_orders_called": False,
        }
        _write_json(self.manifest_path, manifest)
        return manifest

    def _load_existing_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {}
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid existing NSE batch manifest: {self.manifest_path}") from exc
        if raw.get("pipeline") != "nse_historical_universe":
            raise ValueError("existing output manifest belongs to another pipeline")
        return raw
