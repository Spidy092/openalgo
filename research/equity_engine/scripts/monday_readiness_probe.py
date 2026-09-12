"""Monday readiness probe CLI: DRY_RUN synthetic preflight or LIVE_READ_ONLY preflight.

Read-only. There is no order mode: this command never places, modifies, or
cancels an order. The credential value is read from the environment and passed
only to read-only clients (profile/funds/static-IP, full quote, public
instrument files); it is never printed, persisted, or fingerprinted. The
persisted readiness report carries token presence as a boolean only.

Exit codes: 0 live-order review, 1 research shadow, 2 shadow infra,
3 not ready, 4 blocked token missing, 5 probe/CLI error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from equity_engine.monday_readiness_probe import (
    EXIT_BLOCKED_TOKEN,
    EXIT_ERROR,
    TOKEN_ENV_VAR,
    FeedHealthEvidence,
    MondayProbeConfig,
    ProbeMode,
    blocked_token_result,
    current_ist_now,
    exit_code_for_result,
    parse_cost_evidence_file,
    parse_historical_validation_file,
    parse_paper_evidence_file,
    result_summary,
    run_dry_run,
    run_live_read_only,
    token_present_from_env,
    write_probe_report,
)


def _decimal_arg(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid Decimal: {value!r}") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("Decimal must be finite")
    return parsed


def _positive_decimal_arg(value: str) -> Decimal:
    parsed = _decimal_arg(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_decimal_arg(value: str) -> Decimal:
    parsed = _decimal_arg(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}") from exc
    if not parsed > 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _buffer_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc
    if parsed < 0 or parsed >= 60:
        raise argparse.ArgumentTypeError("must be an integer in [0, 60)")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monday live-market readiness preflight (read-only, never places orders). "
            "DRY_RUN is a synthetic self-check; LIVE_READ_ONLY collects live "
            "read-only evidence plus caller-supplied research evidence."
        )
    )
    parser.add_argument("--mode", required=True, choices=("dry-run", "live-read-only"))
    parser.add_argument("--instrument-key", required=True)
    parser.add_argument("--approved-capital", required=True, type=_positive_decimal_arg)
    parser.add_argument("--cost-tolerance", required=True, type=_non_negative_decimal_arg)
    parser.add_argument("--max-quote-age-seconds", required=True, type=_positive_float_arg)
    parser.add_argument("--exit-buffer-minutes", required=True, type=_buffer_arg)
    parser.add_argument("--tick-size-scale", required=True, type=_positive_decimal_arg)
    parser.add_argument("--tick-reference-price", required=True, type=_positive_decimal_arg)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--feed-status", choices=("missing", "available", "unavailable"), default="missing"
    )
    parser.add_argument("--feed-gap", action="store_true")
    parser.add_argument("--pit-complete", action="store_true")
    parser.add_argument("--strategy-evidence-present", action="store_true")
    parser.add_argument("--historical-json", type=Path, default=None)
    parser.add_argument("--cost-json", type=Path, default=None)
    parser.add_argument("--paper-json", type=Path, default=None)
    parser.add_argument("--kill-switch-engaged", action="store_true")
    parser.add_argument("--token-env-var", default=TOKEN_ENV_VAR)
    return parser.parse_args(argv)


def _redacted(message: str, token: str) -> str:
    if token:
        return message.replace(token, "[REDACTED]")
    return message


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = ProbeMode.DRY_RUN if args.mode == "dry-run" else ProbeMode.LIVE_READ_ONLY
    token_present = token_present_from_env(env_var=args.token_env_var)
    now_ist = current_ist_now()

    if mode is ProbeMode.DRY_RUN:
        # Synthetic plumbing preflight. Research evidence is missing by
        # construction (never fabricated), so the ceiling is SHADOW_INFRA;
        # token presence stays real (PRESENT/ABSENT only).
        config = MondayProbeConfig(
            mode=mode,
            instrument_key=args.instrument_key,
            approved_capital_rupees=args.approved_capital,
            cost_tolerance_inr=args.cost_tolerance,
            max_quote_age_seconds=args.max_quote_age_seconds,
            exit_buffer_minutes=args.exit_buffer_minutes,
            tick_size_scale_rupees_per_raw_unit=args.tick_size_scale,
            tick_reference_price_rupees=args.tick_reference_price,
            pit_complete=True,
            historical_validation=None,
            strategy_evidence_present=True,
            cost_evidence=None,
            paper_evidence=None,
            feed=None,
            kill_switch_engaged=args.kill_switch_engaged,
        )
        try:
            result = run_dry_run(config, now_ist=now_ist, token_present=token_present)
        except ValueError as exc:
            print(f"probe configuration error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        write_probe_report(args.output, result)
        print(json.dumps(result_summary(result), indent=2, sort_keys=True))
        return exit_code_for_result(result)

    # LIVE_READ_ONLY: token first. Absent token blocks before any client exists.
    if not token_present:
        blocked = blocked_token_result()
        write_probe_report(args.output, blocked)
        print(json.dumps(result_summary(blocked), indent=2, sort_keys=True))
        return EXIT_BLOCKED_TOKEN

    def load_evidence(label: str, path: Path | None, parser: object) -> object:
        if path is None:
            return None
        parsed = parser(path)  # type: ignore[operator]
        if parsed is None:
            print(f"warning: {label} evidence file could not be verified; treated as missing")
        return parsed

    historical = load_evidence("historical", args.historical_json, parse_historical_validation_file)
    cost = load_evidence("cost", args.cost_json, parse_cost_evidence_file)
    paper = load_evidence("paper", args.paper_json, parse_paper_evidence_file)

    token = os.environ.get(args.token_env_var, "")
    try:
        import httpx

        from equity_engine.upstox_instruments import (
            UPSTOX_NSE_BOD_URL,
            UPSTOX_NSE_MIS_URL,
            UPSTOX_SUSPENDED_URL,
            UpstoxPublicInstrumentFiles,
        )
        from equity_engine.upstox_market_context import UpstoxFullQuoteV3Client
        from equity_engine.upstox_readiness import UpstoxReadinessProbe

        _fetch_errors = (httpx.HTTPError, ValueError, RuntimeError, OSError)
        try:
            snapshot = UpstoxReadinessProbe(access_token=token).run()
        except _fetch_errors as exc:
            print(f"warning: broker readiness fetch failed: {_redacted(str(exc), token)}")
            snapshot = None
        try:
            quotes = UpstoxFullQuoteV3Client(access_token=token).fetch_partial_by_instrument_token(
                [args.instrument_key]
            )
        except _fetch_errors as exc:
            print(f"warning: quote fetch failed: {_redacted(str(exc), token)}")
            quotes = None
        try:
            files = UpstoxPublicInstrumentFiles()
            bod_rows = tuple(files.fetch(UPSTOX_NSE_BOD_URL).rows)
            mis_rows = tuple(files.fetch(UPSTOX_NSE_MIS_URL).rows)
            suspended_rows = tuple(files.fetch(UPSTOX_SUSPENDED_URL).rows)
        except _fetch_errors as exc:
            print(f"warning: instrument file fetch failed: {_redacted(str(exc), token)}")
            bod_rows, mis_rows, suspended_rows = (), (), ()
    finally:
        del token

    if args.feed_status == "missing":
        feed = None
    else:
        feed = FeedHealthEvidence(
            available=args.feed_status == "available",
            gap_detected=args.feed_gap,
            last_heartbeat_ist=now_ist if args.feed_status == "available" else None,
        )

    config = MondayProbeConfig(
        mode=mode,
        instrument_key=args.instrument_key,
        approved_capital_rupees=args.approved_capital,
        cost_tolerance_inr=args.cost_tolerance,
        max_quote_age_seconds=args.max_quote_age_seconds,
        exit_buffer_minutes=args.exit_buffer_minutes,
        tick_size_scale_rupees_per_raw_unit=args.tick_size_scale,
        tick_reference_price_rupees=args.tick_reference_price,
        pit_complete=args.pit_complete,
        historical_validation=historical,  # type: ignore[arg-type]
        strategy_evidence_present=args.strategy_evidence_present,
        cost_evidence=cost,  # type: ignore[arg-type]
        paper_evidence=paper,  # type: ignore[arg-type]
        feed=feed,
        kill_switch_engaged=args.kill_switch_engaged,
    )
    try:
        result = run_live_read_only(
            config,
            token_present=True,
            now_ist=now_ist,
            readiness_snapshot=snapshot,
            quotes=quotes,
            bod_rows=bod_rows,
            mis_rows=mis_rows,
            suspended_rows=suspended_rows,
        )
    except ValueError as exc:
        print(f"probe configuration error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    write_probe_report(args.output, result)
    print(json.dumps(result_summary(result), indent=2, sort_keys=True))
    return exit_code_for_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
