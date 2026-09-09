from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from equity_engine.current_market_discovery import (
    APPROVED_CAPITALS,
    DEFAULT_ESTIMATED_BYTES_PER_ROW,
    DEFAULT_TICK_SIZE_SCALE_RUPEES_PER_RAW_UNIT,
    fingerprint_payload,
    measure_current_market,
    quote_request_keys,
)
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.observed_costs import ObservedUpstoxNSEIntradayCostProvider
from equity_engine.upstox_instruments import (
    UPSTOX_NSE_BOD_URL,
    UPSTOX_NSE_MIS_URL,
    UPSTOX_SUSPENDED_URL,
    UpstoxPublicInstrumentFiles,
)
from equity_engine.upstox_market_context import UpstoxFullQuoteV3Client


def _decimal(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("must be a Decimal") from exc
    if not result.is_finite() or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def _datetime(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO-8601 datetime with timezone") from exc
    if result.tzinfo is None:
        raise argparse.ArgumentTypeError("must include a timezone")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only current NSE market discovery and capital calibration. "
            "This command does not download historical candles or call order APIs."
        )
    )
    parser.add_argument("--snapshot-as-of", required=True, type=_datetime)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--capital",
        action="append",
        type=_decimal,
        help="approved capital scenario; repeat to replace the defaults",
    )
    parser.add_argument("--max-last-price", type=_decimal)
    parser.add_argument(
        "--cost-model",
        choices=("documented", "broker-observed"),
        default="documented",
    )
    parser.add_argument(
        "--tick-size-scale",
        type=_decimal,
        default=DEFAULT_TICK_SIZE_SCALE_RUPEES_PER_RAW_UNIT,
    )
    parser.add_argument(
        "--estimated-bytes-per-row",
        type=int,
        default=DEFAULT_ESTIMATED_BYTES_PER_ROW,
    )
    args = parser.parse_args()

    capitals = tuple(args.capital) if args.capital else APPROVED_CAPITALS
    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if not token:
        parser.error("UPSTOX_ACCESS_TOKEN is not set")

    files = UpstoxPublicInstrumentFiles()
    bod = files.fetch(UPSTOX_NSE_BOD_URL)
    mis = files.fetch(UPSTOX_NSE_MIS_URL)
    suspended = files.fetch(UPSTOX_SUSPENDED_URL)
    request_keys = quote_request_keys(bod.rows)
    quote_client = UpstoxFullQuoteV3Client(access_token=token)
    quotes = quote_client.fetch_partial_by_instrument_token(list(request_keys))

    if args.cost_model == "documented":
        cost_provider = CurrentTermsNSEIntradayCostProvider(pricing_date=args.snapshot_as_of.date())
    else:
        cost_provider = ObservedUpstoxNSEIntradayCostProvider(
            pricing_date=args.snapshot_as_of.date()
        )
    artifact = measure_current_market(
        snapshot_as_of=args.snapshot_as_of,
        bod=bod,
        mis=mis,
        suspended=suspended,
        quotes=quotes,
        cost_provider=cost_provider,
        approved_capitals=capitals,
        max_last_price_rupees=args.max_last_price,
        tick_size_scale_rupees_per_raw_unit=args.tick_size_scale,
        estimated_bytes_per_row=args.estimated_bytes_per_row,
    )
    output = artifact.to_dict()
    artifact_fingerprint = fingerprint_payload(output)
    output["artifact_fingerprint"] = artifact_fingerprint
    serialized_output = json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized_output, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "success",
                "output": str(args.output),
                "artifact_fingerprint": artifact_fingerprint,
                "raw_upstox_instruments": artifact.gate_counts["raw_upstox_instruments"],
                "quote_requests": artifact.quote_request_count,
                "quote_successes": artifact.quote_success_count,
                "quote_failures": artifact.quote_failure_count,
                "live_orders_called": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
