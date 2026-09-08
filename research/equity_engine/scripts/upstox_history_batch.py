from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal
from getpass import getpass
from pathlib import Path

from equity_engine.upstox_batch import UpstoxHistoricalBatchAcquirer
from equity_engine.upstox_history import UpstoxHistoricalDataProvider


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or acquire 5-minute Upstox candles only for affordability-screened, "
            "point-in-time eligible NSE instruments."
        )
    )
    parser.add_argument("--universe-manifest", required=True, type=Path)
    parser.add_argument("--price-file", required=True, type=Path)
    parser.add_argument("--output-dir", default="data/upstox_history")
    parser.add_argument("--cash-limit", type=Decimal, default=Decimal(1000))
    parser.add_argument("--min-affordable-quantity", type=int, default=1)
    parser.add_argument("--access-token", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rate-limit-seconds", type=float, default=0.25)
    args = parser.parse_args()

    token = args.access_token or os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not args.dry_run and not token:
        token = getpass("Paste Upstox access token: ")
    if token is None:
        token = "dry-run-token-not-used"

    provider = UpstoxHistoricalDataProvider(
        access_token=token,
        max_retries=args.max_retries,
        min_request_interval_seconds=args.rate_limit_seconds,
    )
    acquirer = UpstoxHistoricalBatchAcquirer(
        provider=provider,
        output_dir=Path(args.output_dir),
        cash_limit=args.cash_limit,
        minimum_affordable_quantity=args.min_affordable_quantity,
    )
    report = acquirer.run(
        universe_manifest_path=args.universe_manifest,
        price_file=args.price_file,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    report["status"] = "dry_run_complete" if args.dry_run else "complete"
    report["zero_live_orders"] = report["live_orders_called"] is False
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
