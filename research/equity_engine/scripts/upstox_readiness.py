from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from decimal import Decimal

from equity_engine.upstox_readiness import UpstoxReadinessProbe


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def main() -> int:
    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if not token:
        print("UPSTOX_ACCESS_TOKEN is not set", file=sys.stderr)
        return 2

    snapshot = UpstoxReadinessProbe(access_token=token).run()
    # Snapshot contains only readiness facts; no email/mobile/user-id/token is emitted.
    print(json.dumps(asdict(snapshot), indent=2, default=_json_default))
    return 0 if snapshot.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
