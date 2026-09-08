from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from .documented_costs import CurrentTermsNSEIntradayCostProvider
from .models import Exchange, Product
from .nse_batch import load_affordability_prices
from .provenance import dataframe_fingerprint
from .sizing import max_affordable_buy_quantity
from .upstox_history import HistoricalDataset, UpstoxHistoricalDataProvider

UPSTOX_BATCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class UpstoxBatchTask:
    instrument_key: str
    symbol: str
    isin: str
    start: date
    end: date
    eligible_dates: tuple[date, ...]
    reference_price_rupees: Decimal
    affordable_quantity: int

    @property
    def estimated_request_count(self) -> int:
        span_days = (self.end - self.start).days + 1
        return (span_days + 27) // 28

    @property
    def estimated_rows(self) -> int:
        return len(self.eligible_dates) * 75


@dataclass(frozen=True)
class UpstoxBatchPlan:
    tasks: tuple[UpstoxBatchTask, ...]
    skipped: tuple[dict[str, object], ...]
    universe_manifest_sha256: str
    price_file_sha256: str
    estimated_request_count: int
    estimated_rows: int


class UpstoxHistoricalBatchAcquirer:
    """Acquire only affordability-screened, point-in-time eligible NSE candles."""

    def __init__(
        self,
        *,
        provider: UpstoxHistoricalDataProvider,
        output_dir: Path,
        cash_limit: Decimal = Decimal(1000),
        minimum_affordable_quantity: int = 1,
        pricing_date: date = date(2026, 9, 7),
    ) -> None:
        if cash_limit <= 0:
            raise ValueError("cash_limit must be positive")
        if minimum_affordable_quantity <= 0:
            raise ValueError("minimum_affordable_quantity must be positive")
        self.provider = provider
        self.output_dir = Path(output_dir)
        self.cash_limit = cash_limit
        self.minimum_affordable_quantity = minimum_affordable_quantity
        self.pricing_date = pricing_date

    def plan_from_manifest(
        self,
        *,
        universe_manifest_path: Path,
        price_file: Path,
    ) -> UpstoxBatchPlan:
        manifest_payload = universe_manifest_path.read_bytes()
        universe_manifest = json.loads(manifest_payload.decode("utf-8"))
        if universe_manifest.get("pipeline") != "nse_historical_universe":
            raise ValueError("universe manifest is not an NSE historical-universe manifest")
        if universe_manifest.get("live_orders_called") is not False:
            raise ValueError("universe manifest has an invalid live-order safety marker")
        if universe_manifest.get("failed_dates", 0):
            raise ValueError(
                "cannot acquire candles from an incomplete NSE universe manifest; "
                "resolve every missing/schema-error date first"
            )
        prices, price_digest = load_affordability_prices(price_file)
        provider = CurrentTermsNSEIntradayCostProvider(pricing_date=self.pricing_date)

        rows_by_key: dict[str, list[dict[str, object]]] = {}
        skipped: list[dict[str, object]] = []
        for report_date, entry in sorted(universe_manifest.get("dates", {}).items()):
            if entry.get("status") != "complete":
                raise ValueError(f"NSE date {report_date} is not complete")
            audit_path = universe_manifest_path.parent / str(entry["audit_file"])
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            for record in audit.get("records", []):
                if not record.get("eligibility"):
                    continue
                rows_by_key.setdefault(str(record["instrument_key"]), []).append(
                    {
                        "report_date": date.fromisoformat(report_date),
                        "symbol": record["symbol"],
                        "isin": record["isin"],
                    }
                )

        tasks: list[UpstoxBatchTask] = []
        for instrument_key, rows in sorted(rows_by_key.items()):
            price = prices.get(instrument_key)
            if price is None:
                skipped.append(
                    {
                        "instrument_key": instrument_key,
                        "reason": "no affordability price evidence",
                    }
                )
                continue
            affordability = max_affordable_buy_quantity(
                instrument_token=instrument_key,
                exchange=Exchange.NSE,
                product=Product.INTRADAY,
                price=price,
                cash_limit=self.cash_limit,
                cost_provider=provider,
            )
            if affordability.quantity < self.minimum_affordable_quantity:
                skipped.append(
                    {
                        "instrument_key": instrument_key,
                        "reason": "fails affordability filter",
                        "price_rupees": str(price),
                        "affordable_quantity": affordability.quantity,
                    }
                )
                continue
            eligible_dates = tuple(sorted({row["report_date"] for row in rows}))
            tasks.append(
                UpstoxBatchTask(
                    instrument_key=instrument_key,
                    symbol=str(rows[0]["symbol"]),
                    isin=str(rows[0]["isin"]),
                    start=eligible_dates[0],
                    end=eligible_dates[-1],
                    eligible_dates=eligible_dates,
                    reference_price_rupees=price,
                    affordable_quantity=affordability.quantity,
                )
            )

        return UpstoxBatchPlan(
            tasks=tuple(tasks),
            skipped=tuple(skipped),
            universe_manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
            price_file_sha256=price_digest,
            estimated_request_count=sum(task.estimated_request_count for task in tasks),
            estimated_rows=sum(task.estimated_rows for task in tasks),
        )

    def run(
        self,
        *,
        universe_manifest_path: Path,
        price_file: Path,
        dry_run: bool = False,
        resume: bool = True,
    ) -> dict[str, object]:
        plan = self.plan_from_manifest(
            universe_manifest_path=universe_manifest_path,
            price_file=price_file,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "upstox_history_manifest.json"
        previous = self._load_manifest(manifest_path) if resume else {}
        datasets: dict[str, dict[str, object]] = dict(previous.get("datasets", {}))
        failures: list[dict[str, object]] = []

        if not dry_run:
            for task in plan.tasks:
                try:
                    existing = datasets.get(task.instrument_key)
                    if resume and existing and existing.get("status") == "complete":
                        continue
                    dataset = self.provider.fetch_minutes(
                        instrument_token=task.instrument_key,
                        symbol=task.symbol,
                        exchange="NSE",
                        start=task.start,
                        end=task.end,
                        interval_minutes=5,
                        universe_rule_version="nse_mii_v15_point_in_time",
                        adjustment_policy="upstox_adjustment_semantics_unverified_no_fill",
                    )
                    filtered = self._filter_to_eligible_dates(dataset, task)
                    if filtered.frame.empty:
                        raise ValueError("no candles remain on point-in-time eligible dates")
                    saved = self._save_dataset(task, filtered)
                    datasets[task.instrument_key] = saved
                except Exception as exc:  # noqa: BLE001 - each instrument must fail closed
                    failure = {
                        "instrument_key": task.instrument_key,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    datasets[task.instrument_key] = failure

        result: dict[str, object] = {
            "schema_version": UPSTOX_BATCH_SCHEMA_VERSION,
            "pipeline": "upstox_historical_candles",
            "mode": "dry_run" if dry_run else "acquire",
            "universe_manifest": str(universe_manifest_path),
            "universe_manifest_sha256": plan.universe_manifest_sha256,
            "price_file_sha256": plan.price_file_sha256,
            "interval": "5m",
            "cash_limit_rupees": str(self.cash_limit),
            "candidate_count": len(plan.tasks),
            "skipped_count": len(plan.skipped),
            "skipped": list(plan.skipped),
            "estimated_api_request_count": plan.estimated_request_count,
            "estimated_rows": plan.estimated_rows,
            "estimated_storage_bytes": plan.estimated_rows * 150,
            "datasets": datasets,
            "failures": failures,
            "live_orders_called": False,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        if failures:
            raise RuntimeError(
                f"Upstox historical acquisition failed for {len(failures)} instrument(s); "
                f"see {manifest_path}"
            )
        return result

    @staticmethod
    def _filter_to_eligible_dates(
        dataset: HistoricalDataset, task: UpstoxBatchTask
    ) -> HistoricalDataset:
        eligible = set(task.eligible_dates)
        observed = set(dataset.frame.index.date)
        missing_dates = sorted(eligible - observed)
        if missing_dates:
            raise ValueError(
                "Upstox returned no candles for eligible date(s): "
                + ", ".join(item.isoformat() for item in missing_dates)
            )
        frame = dataset.frame[[item in eligible for item in dataset.frame.index.date]].copy()
        manifest = replace(
            dataset.manifest,
            start=frame.index[0].to_pydatetime(),
            end=frame.index[-1].to_pydatetime(),
        )
        return HistoricalDataset(
            frame=frame,
            manifest=manifest,
            fingerprint=dataframe_fingerprint(frame, manifest),
        )

    def _save_dataset(self, task: UpstoxBatchTask, dataset: HistoricalDataset) -> dict[str, object]:
        stem = task.instrument_key.replace("|", "_")
        parquet_path = self.output_dir / f"{stem}_5m.parquet"
        manifest_path = self.output_dir / f"{stem}_5m.manifest.json"
        dataset.frame.to_parquet(parquet_path, index=True)
        payload = {
            "provider": dataset.manifest.provider,
            "exchange": dataset.manifest.exchange,
            "instrument_token": dataset.manifest.instrument_token,
            "symbol": dataset.manifest.symbol,
            "timezone": dataset.manifest.timezone,
            "interval": dataset.manifest.interval,
            "timestamp_semantics": dataset.manifest.timestamp_semantics,
            "start": dataset.manifest.start.isoformat(),
            "end": dataset.manifest.end.isoformat(),
            "retrieved_at": dataset.manifest.retrieved_at.isoformat(),
            "adjustment_policy": dataset.manifest.adjustment_policy,
            "universe_rule_version": dataset.manifest.universe_rule_version,
            "source_reference": dataset.manifest.source_reference,
            "fingerprint_sha256": dataset.fingerprint,
            "rows": len(dataset.frame),
            "eligible_dates": [item.isoformat() for item in task.eligible_dates],
            "reference_price_rupees": str(task.reference_price_rupees),
            "affordable_quantity": task.affordable_quantity,
            "parquet": str(parquet_path),
            "live_orders_called": False,
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "status": "complete",
            "instrument_key": task.instrument_key,
            "symbol": task.symbol,
            "rows": len(dataset.frame),
            "fingerprint_sha256": dataset.fingerprint,
            "parquet": str(parquet_path),
            "manifest": str(manifest_path),
            "request_count": task.estimated_request_count,
        }

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("pipeline") != "upstox_historical_candles":
            raise ValueError("existing output manifest belongs to another pipeline")
        return raw


def price_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
