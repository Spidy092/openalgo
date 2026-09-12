from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from equity_engine.upstox_batch_history import (
    UpstoxHistoricalBatchDownloader,
    candidates_from_universe_manifest,
    load_acquisition_evidence,
    load_acquisition_plan_binding,
    load_candidate_file,
    plan_historical_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute resumable Upstox 5-minute historical acquisition."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--universe-manifest")
    source.add_argument("--candidate-file")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "verify candidate, canonical acquisition plan, acquisition evidence, fingerprints, "
            "and interval bindings without reading a token or making network requests"
        ),
    )
    parser.add_argument("--prefilter-evidence")
    parser.add_argument(
        "--acquisition-plan",
        help="canonical HistoricalAcquisitionPlan JSON required for execution",
    )
    parser.add_argument(
        "--acquisition-evidence",
        help="JSON file binding PIT, corporate-action, acquisition-plan, and session evidence",
    )
    parser.add_argument("--output-dir", default="data/upstox_history_batch")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--expected-rows-per-trading-day", type=int, default=75)
    parser.add_argument("--estimated-bytes-per-row", type=int, default=80)
    parser.add_argument("--min-request-interval", type=float, default=0.15)
    parser.add_argument("--universe-rule-version", default="nse-cm-v15-point-in-time")
    parser.add_argument("--adjustment-policy", default="raw-unadjusted-block-structural-actions")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute cannot be combined")

    trading_day_counts = None
    if args.universe_manifest:
        candidates, trading_day_counts = candidates_from_universe_manifest(
            Path(args.universe_manifest)
        )
        affordability_prefilter = False
    else:
        candidates = load_candidate_file(Path(args.candidate_file))
        affordability_prefilter = bool(args.prefilter_evidence)

    plan = plan_historical_batch(
        candidates=candidates,
        interval_minutes=args.interval,
        expected_rows_per_trading_day=(
            args.expected_rows_per_trading_day if trading_day_counts is not None else None
        ),
        estimated_bytes_per_row=(
            args.estimated_bytes_per_row if trading_day_counts is not None else None
        ),
        trading_day_counts=trading_day_counts,
        affordability_prefilter_applied=affordability_prefilter,
    )

    plan_output = {
        "mode": "execute" if args.execute else "dry-run",
        "candidate_count": len(plan.candidates),
        "interval_minutes": plan.interval_minutes,
        "estimated_upstox_requests": plan.estimated_requests,
        "estimated_rows": plan.estimated_rows,
        "estimated_storage_bytes": plan.estimated_storage_bytes,
        "affordability_prefilter_applied": plan.affordability_prefilter_applied,
        "note": plan.note,
        "live_orders_called": False,
    }

    if args.dry_run:
        if args.universe_manifest:
            parser.error("--dry-run requires --candidate-file, not the full universe manifest")
        if not args.prefilter_evidence:
            parser.error("--dry-run requires --prefilter-evidence describing candidate selection")
        if not args.acquisition_plan:
            parser.error("--dry-run requires --acquisition-plan from the canonical planner")
        if not args.acquisition_evidence:
            parser.error("--dry-run requires --acquisition-evidence binding historical evidence")

        evidence = load_acquisition_evidence(Path(args.acquisition_evidence))
        acquisition_plan = load_acquisition_plan_binding(Path(args.acquisition_plan))
        acquisition_plan.validate_execution(
            candidates=plan.candidates,
            interval_minutes=args.interval,
            evidence=evidence,
        )
        canonical_plan_payload = json.loads(Path(args.acquisition_plan).read_text(encoding="utf-8"))
        dry_run_output = {
            **plan_output,
            "mode": "dry-run-verified",
            "status": "verified",
            "acquisition_plan_fingerprint": acquisition_plan.deterministic_fingerprint,
            "acquisition_evidence_fingerprint": evidence.fingerprint(),
            "canonical_estimated_request_count": canonical_plan_payload["expected_request_count"],
            "canonical_estimated_rows": canonical_plan_payload["estimated_rows"],
            "canonical_estimated_storage_bytes": canonical_plan_payload["estimated_storage_bytes"],
            "candidate_intervals": [asdict(item) for item in plan.candidates],
            "network_requests": 0,
            "token_required": False,
            "downloader_constructed": False,
            "live_orders_called": False,
        }
        print(json.dumps(dry_run_output, indent=2, default=str))
        return 0

    if not args.execute:
        print(json.dumps(plan_output, indent=2))
        return 0

    if args.universe_manifest:
        parser.error(
            "execution from the full universe manifest is blocked; supply --candidate-file after "
            "the explicit historical affordability/liquidity prefilter"
        )
    if not args.prefilter_evidence:
        parser.error("--execute requires --prefilter-evidence describing candidate selection")
    if not args.acquisition_plan:
        parser.error("--execute requires --acquisition-plan from the canonical planner")
    if not args.acquisition_evidence:
        parser.error("--execute requires --acquisition-evidence binding historical evidence")

    evidence = load_acquisition_evidence(Path(args.acquisition_evidence))
    acquisition_plan = load_acquisition_plan_binding(Path(args.acquisition_plan))
    acquisition_plan.validate_execution(
        candidates=plan.candidates,
        interval_minutes=args.interval,
        evidence=evidence,
    )

    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if not token:
        parser.error("UPSTOX_ACCESS_TOKEN is not set")

    downloader = UpstoxHistoricalBatchDownloader(
        access_token=token,
        output_dir=Path(args.output_dir),
        interval_minutes=args.interval,
        min_request_interval_seconds=args.min_request_interval,
    )
    result = downloader.run(
        candidates=plan.candidates,
        universe_rule_version=args.universe_rule_version,
        adjustment_policy=args.adjustment_policy,
        evidence=evidence,
    )
    output = {
        **plan_output,
        "status": "success" if result.passed else "failed",
        "completed": len(result.items),
        "failed": len(result.failures),
        "batch_manifest": result.manifest_path,
        "failures": list(result.failures),
        "items": [asdict(item) for item in result.items],
        "live_orders_called": False,
    }
    print(json.dumps(output, indent=2, default=str))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
