from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from equity_engine.nse_batch_universe import NseBatchUniverseBuilder
from equity_engine.nse_calendar import nse_cm_normal_session_calendar
from equity_engine.nse_semantics import (
    EffectiveDatedNseCmSemanticsPolicy,
    NSE_MASTER_DATA_V15_EFFECTIVE_EVIDENCE_DATE,
    nse_cm_master_data_v15_semantics,
)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cached point-in-time NSE CM equity universe across sourced normal-session dates."
        )
    )
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--output-dir", default="data/nse_universe_batch")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="re-download cached dates only to verify the archive payload has not changed",
    )
    args = parser.parse_args()

    if args.start < NSE_MASTER_DATA_V15_EFFECTIVE_EVIDENCE_DATE:
        parser.error("exact NSE CM semantics are intentionally bounded to 2024-07-01 or later")
    calendar = nse_cm_normal_session_calendar(start=args.start, end=args.end)
    policy = EffectiveDatedNseCmSemanticsPolicy([nse_cm_master_data_v15_semantics()])
    result = NseBatchUniverseBuilder(
        output_dir=Path(args.output_dir),
        semantics_policy=policy,
    ).build(
        calendar=calendar,
        start=args.start,
        end=args.end,
        refresh_existing=args.refresh_existing,
    )

    output = {
        "status": "success" if result.passed else "failed",
        "start": result.start.isoformat(),
        "end": result.end.isoformat(),
        "trading_dates_found": len(result.normal_trading_dates),
        "completed_snapshots": len(result.days),
        "failed_snapshots": len(result.failures),
        "holiday_dates_excluded": len(result.holiday_dates),
        "special_session_dates_excluded": [
            day.isoformat() for day in result.excluded_special_session_dates
        ],
        "unique_instruments": result.unique_instruments,
        "unique_eligible_instruments": result.unique_eligible_instruments,
        "manifest": result.manifest_path,
        "failures": [asdict(item) for item in result.failures],
        "next_gate": (
            "plan 5-minute Upstox acquisition; full execution still requires an explicit "
            "historical affordability/liquidity prefilter candidate file"
        ),
        "live_orders_called": False,
    }
    print(json.dumps(output, indent=2, default=str))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
