from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from equity_engine.nse_batch import (
    NseBatchAcquisitionError,
    NseHistoricalUniverseBatch,
    load_affordability_prices,
)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire official NSE CM MII snapshots and materialize a resumable point-in-time "
            "NSE_EQ universe."
        )
    )
    parser.add_argument("--start", type=_parse_date, default=date(2024, 7, 1))
    parser.add_argument("--end", type=_parse_date, default=date(2026, 7, 31))
    parser.add_argument("--output-dir", default="data/nse_universe")
    parser.add_argument("--affordability-price-file", type=Path)
    parser.add_argument("--cash-limit", type=Decimal, default=Decimal(1000))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the date/screen/request report; no broker or order endpoint is called",
    )
    args = parser.parse_args()

    prices = None
    price_digest = None
    if args.affordability_price_file is not None:
        prices, price_digest = load_affordability_prices(args.affordability_price_file)

    batch = NseHistoricalUniverseBatch(output_dir=Path(args.output_dir))
    try:
        result = batch.run(
            start=args.start,
            end=args.end,
            resume=args.resume,
            refresh=args.refresh,
            affordability_prices=prices,
            affordability_price_file_sha256=price_digest,
            cash_limit=args.cash_limit,
        )
    except NseBatchAcquisitionError as exc:
        print(json.dumps({"status": "failed", "manifest": str(exc.manifest_path)}, indent=2))
        return 2

    report = dict(result.manifest)
    report["status"] = "dry_run_complete" if args.dry_run else "complete"
    report["report"] = {
        "trading_dates_found": len(report["trading_dates"]),
        "snapshots_available": report["completed_dates"],
        "unique_instruments": report["dry_run_screen"]["unique_eligible_instruments"],
        "eligible_counts_by_date": {
            key: value.get("eligible_records", 0)
            for key, value in report["dates"].items()
            if value.get("status") == "complete"
        },
        "candidate_count_after_1000_rupee_affordability_filter": report["dry_run_screen"][
            "candidate_count_after_affordability_filter"
        ],
        "estimated_api_request_count": report["dry_run_screen"]["estimated_api_request_count_5m"],
        "estimated_storage_bytes": report["dry_run_screen"]["estimated_storage_bytes"],
        "live_orders_called": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
