"""Read-only shadow session health status for Monday. No orders, no network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from equity_engine.shadow_session_health import (
    HealthStatus,
    HealthThresholds,
    check_persisted_session,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only health status over persisted shadow evidence."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--freshness-seconds", type=float, default=60.0)
    parser.add_argument("--now", default=None, help="ISO timestamp override for determinism")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = check_persisted_session(
        args.output_dir,
        now_iso=args.now,
        thresholds=HealthThresholds(freshness_seconds=args.freshness_seconds),
    )
    payload = report.as_dict()
    payload["fingerprint"] = report.fingerprint()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if report.status is HealthStatus.HEALTHY:
        return 0
    if report.status is HealthStatus.DEGRADED_NO_TRADING:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
