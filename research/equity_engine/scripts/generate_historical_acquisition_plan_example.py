"""Generate the synthetic HistoricalAcquisitionPlan DRY-RUN example."""

from __future__ import annotations

import argparse
from pathlib import Path

from equity_engine.historical_acquisition_plan import build_example_historical_acquisition_plan

_DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "examples" / (
    "historical_acquisition_plan_2024-10-01_2026-09-08.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = build_example_historical_acquisition_plan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(plan.to_json(indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"plan_id={plan.plan_id}")
    print(f"deterministic_fingerprint={plan.deterministic_fingerprint()}")
    print(f"expected_request_count={plan.expected_request_count}")
    print(f"estimated_rows={plan.estimated_rows}")
    print(f"estimated_storage_bytes={plan.estimated_storage_bytes}")


if __name__ == "__main__":
    main()
