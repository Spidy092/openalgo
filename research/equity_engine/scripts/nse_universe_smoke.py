from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import httpx

from equity_engine.nse_daily_universe import materialize_nse_daily_equity_universe
from equity_engine.nse_mii_security import NseMiiSecurityMasterParser
from equity_engine.nse_semantics import (
    EffectiveDatedNseCmSemanticsPolicy,
    nse_cm_master_data_v15_semantics,
)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and validate one official NSE MII security master snapshot."
    )
    parser.add_argument("--date", required=True, type=_parse_date)
    parser.add_argument("--output-dir", default="data/nse_universe_smoke")
    args = parser.parse_args()

    mii = NseMiiSecurityMasterParser()
    filename = f"NSE_CM_security_{args.date.strftime('%d%m%Y')}.csv.gz"
    source_url = mii.source_url(args.date)

    response = httpx.get(
        source_url,
        headers={"User-Agent": "openalgo-equity-research/0.1"},
        timeout=30.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    snapshot = mii.parse_bytes(response.content, filename=filename)

    policy = EffectiveDatedNseCmSemanticsPolicy([nse_cm_master_data_v15_semantics()])
    universe = materialize_nse_daily_equity_universe(
        snapshot=snapshot,
        semantics_policy=policy,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gz_path = output_dir / filename
    gz_path.write_bytes(response.content)

    summary = {
        "status": "success",
        "report_date": universe.report_date.isoformat(),
        "source_url": universe.source_url,
        "snapshot_sha256": universe.snapshot_sha256,
        "semantics_source": universe.semantics_source,
        "records": len(universe.records),
        "eligible_records": len(universe.eligible_records),
        "ineligible_records": len(universe.ineligible_records),
        "sample_eligible": [
            {
                "symbol": record.symbol,
                "isin": record.isin,
                "instrument_key": record.instrument_key,
                "tick_size_rupees": str(record.tick_size_rupees),
            }
            for record in universe.eligible_records[:10]
        ],
        "saved_gzip": str(gz_path),
        "live_orders_called": False,
    }
    summary_path = output_dir / f"NSE_CM_universe_{args.date.isoformat()}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
