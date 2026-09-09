from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from equity_engine.upstox_cost_reconciliation import (
    DEFAULT_NOTIONALS,
    EXIT_BROKER_API_ERROR,
    EXIT_CONFIGURATION_ERROR,
    reconcile_orders,
    report_as_json,
    render_terminal,
    write_evidence,
    exit_code_for,
)


def _decimal_arg(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid Decimal: {value!r}") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("Decimal must be finite")
    return parsed


def _notionals_arg(value: str) -> tuple[Decimal, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--notionals must contain at least one amount")
    try:
        notionals = tuple(Decimal(part) for part in parts)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError("--notionals must be comma-separated Decimals") from exc
    if any(notional <= 0 or notional.is_finite() for notional in notionals):
        raise argparse.ArgumentTypeError("--notionals must contain positive finite amounts")
    return notionals


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Upstox Brokerage Details reconciliation for representative NSE "
            "intraday orders."
        )
    )
    parser.add_argument("--instrument-token", required=True, help="Supplied Upstox NSE_EQ token")
    parser.add_argument(
        "--symbol", default=None, help="Optional display symbol; never used as a token"
    )
    parser.add_argument("--price", required=True, type=_decimal_arg, help="Price in INR")
    parser.add_argument("--capital", type=_decimal_arg, default=Decimal("1000"))
    parser.add_argument("--pricing-date", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--notionals", type=_notionals_arg, default=tuple(Decimal(x) for x in DEFAULT_NOTIONALS)
    )
    parser.add_argument("--tolerance", required=True, type=_decimal_arg, help="INR tolerance")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()
    if not token:
        message = {"status": "ERROR", "error": "UPSTOX_ACCESS_TOKEN is not set"}
        if args.json:
            print(json.dumps(message, indent=2) + "\n")
        else:
            print(message["error"], file=sys.stderr)
        return EXIT_CONFIGURATION_ERROR

    try:
        report = reconcile_orders(
            access_token=token,
            instrument_token=args.instrument_token,
            symbol=args.symbol,
            price=args.price,
            capital=args.capital,
            pricing_date=args.pricing_date,
            tolerance=args.tolerance,
            target_notionals=args.notionals,
        )
    except (ValueError, NotImplementedError) as exc:
        if args.json:
            print(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2) + "\n")
        else:
            print(str(exc), file=sys.stderr)
        return EXIT_CONFIGURATION_ERROR
    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "ERROR", "error": type(exc).__name__}, indent=2) + "\n")
        else:
            print(
                f"{type(exc).__name__}: broker reconciliation could not complete", file=sys.stderr
            )
        return EXIT_BROKER_API_ERROR

    if args.output is not None:
        write_evidence(args.output, report)
    if args.json:
        print(report_as_json(report), end="")
    else:
        print(render_terminal(report), end="")
    return exit_code_for(report)


if __name__ == "__main__":
    raise SystemExit(main())
