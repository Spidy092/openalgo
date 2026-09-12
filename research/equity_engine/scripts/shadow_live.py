"""Read-only live shadow CLI for Monday testing. No order mode exists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from equity_engine.live_market_readiness import (
    LiveMarketReadinessReport,
    build_synthetic_readiness_report,
)
from equity_engine.shadow_live_runner import (
    BLOCKED_TOKEN_MISSING,
    TOKEN_ENV_VAR,
    FeedMode,
    ReadinessGateError,
    RunnerMode,
    ShadowLiveConfig,
    ShadowLiveRunner,
    SyntheticQuoteSource,
    UpstoxPollingQuoteSource,
)
from equity_engine.upstox_market_context import UpstoxFullQuoteV3Client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only shadow runner. DRY_RUN uses a synthetic fixture; "
            "LIVE_READ_ONLY performs a minimal authenticated quote smoke test. "
            "No live-order mode exists."
        )
    )
    parser.add_argument("--mode", required=True, choices=["DRY_RUN", "LIVE_READ_ONLY"])
    parser.add_argument("--instrument-keys", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-polls", type=int, default=3)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.0)
    parser.add_argument("--approved-capital", default="100000")
    parser.add_argument("--exit-buffer-minutes", type=int, default=15)
    parser.add_argument(
        "--strategy", default="always", choices=["always", "never", "exit-second-bar"]
    )
    parser.add_argument("--cas-eligible", action="store_true")
    parser.add_argument("--quote-freshness-seconds", type=float, default=60.0)
    parser.add_argument("--expected-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--session-id", default="monday-shadow-live-v1")
    parser.add_argument(
        "--trade-date",
        type=date.fromisoformat,
        default=date(2026, 9, 7),
        help="ISO exchange trade date bound to the readiness report (default: 2026-09-07)",
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        help="Canonical LiveMarketReadinessReport JSON; required for LIVE_READ_ONLY",
    )
    return parser.parse_args()


def _synthetic_batches(keys: list[str], trade_date: date) -> list[dict[str, dict[str, object]]]:
    batches: list[dict[str, dict[str, object]]] = []
    base = datetime.fromisoformat(f"{trade_date.isoformat()}T09:15:00+05:30")
    for poll in range(3):
        ts = base.fromtimestamp(base.timestamp() + poll * 300, tz=ZoneInfo("Asia/Kolkata"))
        price = 100 + poll
        batches.append(
            {
                key: {
                    "instrument_token": key,
                    "timestamp": ts.isoformat(),
                    "last_price": price + 0.1,
                    "prev_close_price": 100,
                    "ohlc": {
                        "open": price,
                        "high": price + 0.15,
                        "low": price - 0.05,
                        "close": price + 0.1,
                    },
                }
                for key in keys
            }
        )
    return batches


def main() -> int:
    args = _parse_args()
    keys = [item.strip() for item in args.instrument_keys.split(",") if item.strip()]
    if not keys:
        print("instrument-keys cannot be empty", file=sys.stderr)
        return 2
    mode = RunnerMode(args.mode)
    trade_date = args.trade_date
    config = ShadowLiveConfig(
        session_id=args.session_id,
        instrument_keys=tuple(keys),
        cas_eligible_by_key=tuple((key, bool(args.cas_eligible)) for key in keys),
        tick_size_by_key=tuple((key, "0.05") for key in keys),
        feed_mode=FeedMode.POLL,
        poll_interval_seconds=args.poll_interval_seconds,
        max_polls=args.max_polls,
        quote_freshness_threshold_seconds=args.quote_freshness_seconds,
        expected_cadence_seconds=args.expected_cadence_seconds,
        approved_capital_rupees=args.approved_capital,
        exit_buffer_minutes=args.exit_buffer_minutes,
        strategy_name=args.strategy,
        output_dir=str(args.output_dir),
        mode=mode,
    )
    if mode is RunnerMode.DRY_RUN:
        synthetic_start = datetime.fromisoformat(f"{trade_date.isoformat()}T09:15:05+05:30")
        synthetic_report = build_synthetic_readiness_report(
            checked_at_ist=synthetic_start,
            trade_date=trade_date,
            instrument_keys=config.instrument_keys,
            cas_eligible_by_key=config.cas_eligible_by_key,
            tick_size_by_key=config.tick_size_by_key,
            exit_buffer_minutes=config.exit_buffer_minutes,
            approved_capital=config.approved_capital(),
            quote_freshness_threshold_seconds=config.quote_freshness_threshold_seconds,
        )
        synthetic_times = iter(
            synthetic_start + timedelta(minutes=5 * index) for index in range(config.max_polls)
        )
        runner = ShadowLiveRunner(
            config=config,
            source=SyntheticQuoteSource(_synthetic_batches(keys, trade_date)),
            now=lambda: next(synthetic_times),
            session_day=trade_date,
            readiness_report=synthetic_report,
        )
        try:
            runner.run()
            summary = runner.persist(args.output_dir)
        except ReadinessGateError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        print(BLOCKED_TOKEN_MISSING, file=sys.stderr)
        return 2
    if args.readiness_report is None:
        print("READINESS_REPORT_INVALID: --readiness-report is required", file=sys.stderr)
        return 2
    try:
        readiness_payload = json.loads(args.readiness_report.read_text(encoding="utf-8"))
        if not isinstance(readiness_payload, dict):
            raise TypeError("report root must be an object")
        readiness_report = LiveMarketReadinessReport.from_dict(readiness_payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        print("READINESS_REPORT_INVALID", file=sys.stderr)
        return 2
    # Minimal authenticated READ-ONLY smoke: one bounded quote batch, no session.
    client = UpstoxFullQuoteV3Client(access_token=token)
    source = UpstoxPollingQuoteSource(client=client, instrument_keys=tuple(keys))
    runner = ShadowLiveRunner(
        config=ShadowLiveConfig(
            session_id=config.session_id,
            instrument_keys=config.instrument_keys,
            cas_eligible_by_key=config.cas_eligible_by_key,
            tick_size_by_key=config.tick_size_by_key,
            feed_mode=config.feed_mode,
            poll_interval_seconds=0.0,
            max_polls=1,
            quote_freshness_threshold_seconds=config.quote_freshness_threshold_seconds,
            expected_cadence_seconds=config.expected_cadence_seconds,
            approved_capital_rupees=config.approved_capital_rupees,
            exit_buffer_minutes=config.exit_buffer_minutes,
            strategy_name=config.strategy_name,
            output_dir=config.output_dir,
            mode=config.mode,
            readiness_report=readiness_report,
        ),
        source=source,
        session_day=trade_date,
        readiness_report=readiness_report,
    )
    try:
        runner.run()
        summary = runner.persist(args.output_dir)
    except ReadinessGateError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # Smoke result carries counts and fingerprints only; never credentials.
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
