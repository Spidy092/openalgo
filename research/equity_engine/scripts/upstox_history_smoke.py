from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import httpx

from equity_engine.provenance import FINGERPRINT_SCHEMA
from equity_engine.upstox_history import UpstoxHistoricalDataProvider


SEARCH_URL = "https://api.upstox.com/v2/instruments/search"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Upstox historical-data smoke test. Searches one exact NSE equity symbol, "
            "fetches historical candles, validates them, and writes Parquet + manifest files."
        )
    )
    parser.add_argument("--symbol", required=True, help="Exact NSE trading symbol, e.g. RELIANCE")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--interval", required=True, type=int, help="Minute interval, 1-15")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _find_exact_nse_equity(*, token: str, symbol: str) -> dict[str, object]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    params = {
        "query": symbol,
        "exchanges": "NSE",
        "segments": "EQ",
        "page_number": 1,
        "records": 30,
    }
    response = httpx.get(SEARCH_URL, headers=headers, params=params, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"instrument search failed: {payload!r}")

    wanted = symbol.strip().upper()
    exact = [
        item
        for item in payload.get("data", [])
        if str(item.get("trading_symbol", "")).upper() == wanted
        and item.get("segment") == "NSE_EQ"
        and item.get("instrument_type") == "EQ"
    ]
    if len(exact) != 1:
        raise RuntimeError(
            f"expected exactly one NSE_EQ/EQ match for {wanted!r}, found {len(exact)}"
        )
    return exact[0]


def main() -> int:
    args = _parse_args()
    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if not token:
        print("UPSTOX_ACCESS_TOKEN is not set", file=sys.stderr)
        return 2
    if args.start > args.end:
        print("--start must be on or before --end", file=sys.stderr)
        return 2

    instrument = _find_exact_nse_equity(token=token, symbol=args.symbol)
    instrument_key = str(instrument["instrument_key"])
    trading_symbol = str(instrument["trading_symbol"])

    dataset = UpstoxHistoricalDataProvider(access_token=token).fetch_minutes(
        instrument_token=instrument_key,
        symbol=trading_symbol,
        exchange="NSE",
        start=args.start,
        end=args.end,
        interval_minutes=args.interval,
        universe_rule_version="connectivity-smoke-only",
        adjustment_policy="none-unverified",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{trading_symbol}_{args.start}_{args.end}_{args.interval}m"
    parquet_path = args.output_dir / f"{stem}.parquet"
    manifest_path = args.output_dir / f"{stem}.manifest.json"

    dataset.frame.to_parquet(parquet_path)
    manifest = {
        "purpose": "authenticated_historical_data_connectivity_smoke_only",
        "not_for_strategy_selection": True,
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "instrument": {
            "instrument_key": instrument_key,
            "trading_symbol": trading_symbol,
            "name": instrument.get("name"),
            "isin": instrument.get("isin"),
            "tick_size_raw": instrument.get("tick_size"),
            "cas_eligible": instrument.get("cas_eligible"),
        },
        "dataset_manifest": asdict(dataset.manifest),
        "fingerprint_sha256": dataset.fingerprint,
        "rows": len(dataset.frame),
        "parquet_file": parquet_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "success",
                "symbol": trading_symbol,
                "instrument_key": instrument_key,
                "rows": len(dataset.frame),
                "first_timestamp": str(dataset.frame.index[0]),
                "last_timestamp": str(dataset.frame.index[-1]),
                "fingerprint_sha256": dataset.fingerprint,
                "fingerprint_schema": FINGERPRINT_SCHEMA,
                "parquet": str(parquet_path),
                "manifest": str(manifest_path),
                "live_orders_called": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
