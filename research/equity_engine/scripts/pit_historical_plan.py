from __future__ import annotations

import argparse
import json
from datetime import date, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from equity_engine.pit_historical_acquisition import (
    AcquisitionRateLimit,
    PITFormationPolicy,
    PITHistoricalAcquisitionPlan,
    PITResearchBoundary,
    build_stage_a_plan,
    write_acquisition_plan,
)
from equity_engine.universe import ResearchUniverseThresholds


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def _parse_time(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("time must be HH:MM[:SS]") from exc


def _parse_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("value must be a Decimal") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dry-run, point-in-time historical acquisition plan. "
            "This command reads an existing NSE universe manifest and never downloads data."
        )
    )
    parser.add_argument("--universe-manifest", required=True, type=Path)
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument("--approved-capital", required=True, type=_parse_decimal)
    parser.add_argument("--max-last-price", required=True, type=_parse_decimal)
    parser.add_argument("--min-median-daily-notional", required=True, type=_parse_decimal)
    parser.add_argument("--min-median-daily-volume", required=True, type=_parse_decimal)
    parser.add_argument("--min-observed-trading-days", required=True, type=int)
    parser.add_argument("--min-affordable-quantity", required=True, type=int)
    parser.add_argument("--timezone", required=True)
    parser.add_argument("--formation-policy-id", required=True)
    parser.add_argument("--decision-time", required=True, type=_parse_time)
    parser.add_argument(
        "--price-reference-policy",
        required=True,
        choices=("prior_completed_session_close",),
    )
    parser.add_argument("--signal-time-policy", required=True)
    parser.add_argument("--execution-time-policy", required=True)
    parser.add_argument("--rate-limit-policy-id", required=True)
    parser.add_argument("--rate-limit-source", required=True)
    parser.add_argument("--min-request-interval", required=True, type=_parse_decimal)
    parser.add_argument("--max-attempts", required=True, type=int)
    parser.add_argument("--backoff-seconds", required=True, type=_parse_decimal)
    parser.add_argument("--universe-rule-version", required=True)
    parser.add_argument("--adjustment-policy", required=True)
    parser.add_argument("--lookback-calendar-days", required=True, type=int)
    parser.add_argument("--estimated-rows-per-trading-day", required=True, type=int)
    parser.add_argument("--estimated-bytes-per-daily-row", required=True, type=int)
    parser.add_argument("--cost-model-identity", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pit_historical_acquisition_plan.json"),
    )
    args = parser.parse_args()

    boundary = PITResearchBoundary(start=args.start, end=args.end)
    thresholds = ResearchUniverseThresholds(
        max_last_price_rupees=args.max_last_price,
        min_median_daily_notional_proxy_rupees=args.min_median_daily_notional,
        min_median_daily_volume_shares=args.min_median_daily_volume,
        min_observed_trading_days=args.min_observed_trading_days,
        min_affordable_quantity=args.min_affordable_quantity,
    )
    formation_policy = PITFormationPolicy(
        policy_id=args.formation_policy_id,
        timezone=args.timezone,
        decision_time=args.decision_time,
        price_reference_policy=args.price_reference_policy,
        signal_time_policy=args.signal_time_policy,
        execution_time_policy=args.execution_time_policy,
    )
    rate_limit = AcquisitionRateLimit(
        policy_id=args.rate_limit_policy_id,
        minimum_interval_seconds=args.min_request_interval,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff_seconds,
        source_reference=args.rate_limit_source,
    )
    stage_a = build_stage_a_plan(
        universe_manifest_path=args.universe_manifest,
        boundary=boundary,
        approved_capital_rupees=args.approved_capital,
        thresholds=thresholds,
        formation_policy=formation_policy,
        rate_limit=rate_limit,
        universe_rule_version=args.universe_rule_version,
        adjustment_policy=args.adjustment_policy,
        lookback_calendar_days=args.lookback_calendar_days,
        estimated_rows_per_trading_day=args.estimated_rows_per_trading_day,
        estimated_bytes_per_row=args.estimated_bytes_per_daily_row,
        cost_model_identity=args.cost_model_identity,
    )
    plan = PITHistoricalAcquisitionPlan(stage_a=stage_a)
    write_acquisition_plan(plan, args.output)

    source_dates = tuple(
        item for item in stage_a.source.trading_dates if boundary.start <= item <= boundary.end
    )
    membership_rows = sum(len(item.eligible_dates) for item in stage_a.candidates)
    output = {
        "status": "planned",
        "mode": "dry-run",
        "boundary": boundary.as_dict(),
        "trading_dates_found": len(source_dates),
        "snapshots_available": len(source_dates),
        "unique_pit_instruments": len(stage_a.candidates),
        "eligible_membership_rows": membership_rows,
        "candidate_count_after_affordability_filter": None,
        "candidate_count_after_affordability_filter_status": (
            "pending Stage-A daily acquisition and point-in-time prefilter"
        ),
        "stage_a_estimated_api_requests": stage_a.estimated_requests,
        "stage_a_estimated_rows": stage_a.estimated_rows,
        "stage_a_estimated_storage_bytes": stage_a.estimated_storage_bytes,
        "stage_b_estimated_api_requests": None,
        "stage_b_estimated_storage_bytes": None,
        "plan": str(args.output),
        "plan_fingerprint": plan.fingerprint,
        "live_orders_called": False,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
