from __future__ import annotations

import argparse
import json
from dataclasses import fields
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from equity_engine.historical_validation import (
    nse_session_rules_for_calendar,
    validate_intraday_dataset,
)
from equity_engine.market_sessions import NSE_CAS_EFFECTIVE_DATE
from equity_engine.nse_calendar import nse_cm_normal_session_calendar
from equity_engine.provenance import MarketDataManifest


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _load_manifest(path: Path) -> tuple[dict[str, object], MarketDataManifest]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain a JSON object")
    raw = (
        payload.get("dataset_manifest")
        or payload.get("market_data_manifest")
        or payload.get("manifest")
    )
    if not isinstance(raw, dict):
        raise ValueError("manifest does not contain dataset_manifest or market_data_manifest")
    required = {item.name for item in fields(MarketDataManifest)}
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError("manifest is missing fields: " + ", ".join(missing))
    values = dict(raw)
    values["start"] = _parse_datetime(values["start"])
    values["end"] = _parse_datetime(values["end"])
    values["retrieved_at"] = _parse_datetime(values["retrieved_at"])
    return payload, MarketDataManifest(**values)


def _parse_interval(interval: str) -> int:
    if not interval.endswith("m"):
        raise ValueError(f"unsupported manifest interval: {interval!r}")
    minutes = int(interval[:-1])
    if minutes < 1 or minutes > 15:
        raise ValueError("manifest interval must be between 1m and 15m")
    return minutes


def _resolve_date_bounds(
    payload: dict[str, object], manifest: MarketDataManifest
) -> tuple[date, date]:
    request = payload.get("request")
    if isinstance(request, dict) and request.get("start") and request.get("end"):
        return date.fromisoformat(str(request["start"])), date.fromisoformat(str(request["end"]))
    return manifest.start.date(), manifest.end.date()


def _resolve_cas_eligibility(
    args: argparse.Namespace, payload: dict[str, object], start: date
) -> bool:
    if args.cas_eligible is not None:
        return args.cas_eligible
    instrument = payload.get("instrument")
    if isinstance(instrument, dict) and instrument.get("cas_eligible") is not None:
        return bool(instrument["cas_eligible"])
    if start >= NSE_CAS_EFFECTIVE_DATE:
        raise ValueError(
            "CAS eligibility is not explicit for a post-2026-08-03 dataset; "
            "pass --cas-eligible or --non-cas-eligible"
        )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline/read-only validation of a historical Parquet plus manifest artifact."
    )
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    cas = parser.add_mutually_exclusive_group()
    cas.add_argument("--cas-eligible", dest="cas_eligible", action="store_true")
    cas.add_argument("--non-cas-eligible", dest="cas_eligible", action="store_false")
    parser.set_defaults(cas_eligible=None)
    args = parser.parse_args(argv)

    try:
        payload, manifest = _load_manifest(args.manifest)
        frame = pd.read_parquet(args.parquet)
        start, end = _resolve_date_bounds(payload, manifest)
        if start > end:
            raise ValueError("manifest date range is reversed")
        calendar = nse_cm_normal_session_calendar(start=start, end=end)
        cas_eligible = _resolve_cas_eligibility(args, payload, start)
        rules = nse_session_rules_for_calendar(
            calendar,
            timezone=manifest.timezone,
            interval_minutes=_parse_interval(manifest.interval),
            cas_eligible=cas_eligible,
        )
        manifest_fingerprint_reference = payload.get("fingerprint_sha256")
        if manifest_fingerprint_reference is None:
            manifest_fingerprint_reference = payload.get("data_fingerprint")
        report = validate_intraday_dataset(
            frame,
            manifest,
            session_rules=rules,
            manifest_fingerprint_reference=(
                str(manifest_fingerprint_reference)
                if manifest_fingerprint_reference is not None
                else None
            ),
            fingerprint_schema=(
                str(payload["fingerprint_schema"])
                if payload.get("fingerprint_schema") is not None
                else None
            ),
            manifest_reference=str(args.manifest),
            calendar_evidence=calendar,
        )
    except Exception as exc:  # noqa: BLE001 - CLI must emit a machine-readable fail-closed result
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "structural_violations": [f"validator error: {type(exc).__name__}: {exc}"],
                    "strategy_ready": False,
                    "live_orders_called": False,
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(report.as_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
