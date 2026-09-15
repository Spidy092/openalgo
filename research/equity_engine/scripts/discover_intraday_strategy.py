"""Discover one intraday cash-equity candidate through the existing gauntlet.

Offline and research-only: reads an already-acquired Parquet dataset plus its
manifest, runs screening -> train/validation/test with untouched test ->
walk-forward -> event-driven simulation -> documented cost model -> friction
stress -> baseline comparison -> offline paper replay -> promotion gates, and
writes ``strategy_candidate.json``. The artifact records the live flag as
false and is the input that an eligibility decision consumes. No execution
path exists here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from equity_engine.gates import PromotionThresholds
from equity_engine.intraday_discovery import DiscoveryConfig, run_discovery
from equity_engine.models import Exchange
from equity_engine.provenance import MarketDataManifest
from equity_engine.tournament import RankingMetric


def _parse_decimal(value: str, *, name: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal number: {value!r}") from exc
    return parsed


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _load_manifest(path: Path) -> MarketDataManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain a JSON object")
    raw = (
        payload.get("dataset_manifest")
        or payload.get("market_data_manifest")
        or payload.get("manifest")
    )
    if not isinstance(raw, dict):
        raise ValueError("manifest does not contain dataset_manifest or market_data_manifest")
    required = {item.name for item in fields(MarketDataManifest)}
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError("manifest is missing fields: " + ", ".join(missing))
    values = dict(raw)
    values["start"] = _parse_datetime(values["start"])
    values["end"] = _parse_datetime(values["end"])
    values["retrieved_at"] = _parse_datetime(values["retrieved_at"])
    return MarketDataManifest(**values)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline intraday discovery: validate one NSE cash-equity candidate "
            "through the existing gauntlet and emit strategy_candidate.json. "
            "No network and no execution path."
        )
    )
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--instrument-key", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-end", required=True, type=date.fromisoformat)
    parser.add_argument("--validation-end", required=True, type=date.fromisoformat)
    parser.add_argument("--session-open", default="09:15")
    parser.add_argument("--bar-minutes", default=5, type=int)
    parser.add_argument("--breakout-buffer-bps", default="0")
    parser.add_argument("--screening-cash", default="1000")
    parser.add_argument("--screening-fee-rate", default="0.0005")
    parser.add_argument("--screening-slippage-rate", default="0.0005")
    parser.add_argument("--initial-cash", default="1000")
    parser.add_argument("--max-trades-per-day", default=1, type=int)
    parser.add_argument("--base-slippage-bps", default="0")
    parser.add_argument("--base-half-spread-bps", default="0")
    parser.add_argument("--stress-slippage-bps", default="5")
    parser.add_argument("--stress-half-spread-bps", default="5")
    parser.add_argument("--wf-train-days", default=3, type=int)
    parser.add_argument("--wf-test-days", default=2, type=int)
    parser.add_argument("--wf-step-days", default=2, type=int)
    parser.add_argument("--wf-embargo-days", default=0, type=int)
    parser.add_argument("--ranking-metric", default="net_return_pct")
    parser.add_argument("--min-trades", default=20, type=int)
    parser.add_argument("--min-profit-factor", default="1.2")
    parser.add_argument("--min-sharpe", default="0.5")
    parser.add_argument("--max-drawdown-pct", default="10")
    parser.add_argument("--min-wf-windows", default=2, type=int)
    parser.add_argument("--max-reconciliation-error-inr", default="0.01")
    parser.add_argument("--cost-reconciliation-error-inr", default=None)
    parser.add_argument("--cost-reconciliation-source", default="not-performed")
    parser.add_argument("--tick-size-rupees", default="0.05")
    parser.add_argument("--tick-size-source", default="cli-explicit")
    parser.add_argument("--exit-buffer-minutes", default=10, type=int)
    parser.add_argument("--pricing-date", default="2026-09-07", type=date.fromisoformat)
    cas = parser.add_mutually_exclusive_group()
    cas.add_argument("--cas-eligible", dest="cas_eligible", action="store_true")
    cas.add_argument("--non-cas-eligible", dest="cas_eligible", action="store_false")
    parser.set_defaults(cas_eligible=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = _load_manifest(args.manifest)
        frame = pd.read_parquet(args.parquet)
        session_open = time.fromisoformat(args.session_open)
        ranking = RankingMetric(args.ranking_metric)
        reconciliation_error = (
            _parse_decimal(args.cost_reconciliation_error_inr, name="cost-reconciliation-error")
            if args.cost_reconciliation_error_inr is not None
            else None
        )
        config = DiscoveryConfig(
            instrument_token=args.instrument_key,
            symbol=args.symbol,
            exchange=Exchange.NSE,
            session_open=session_open,
            bar_minutes=args.bar_minutes,
            breakout_buffer_bps=_parse_decimal(args.breakout_buffer_bps, name="breakout-buffer"),
            screening_cash=_parse_decimal(args.screening_cash, name="screening-cash"),
            screening_fee_rate=_parse_decimal(args.screening_fee_rate, name="screening-fee"),
            screening_slippage_rate=_parse_decimal(
                args.screening_slippage_rate, name="screening-slippage"
            ),
            initial_cash=_parse_decimal(args.initial_cash, name="initial-cash"),
            max_trades_per_day=args.max_trades_per_day,
            base_slippage_bps_per_leg=_parse_decimal(args.base_slippage_bps, name="base-slippage"),
            base_half_spread_bps_per_leg=_parse_decimal(
                args.base_half_spread_bps, name="base-half-spread"
            ),
            stress_slippage_bps_per_leg=_parse_decimal(
                args.stress_slippage_bps, name="stress-slippage"
            ),
            stress_half_spread_bps_per_leg=_parse_decimal(
                args.stress_half_spread_bps, name="stress-half-spread"
            ),
            train_end_date=args.train_end,
            validation_end_date=args.validation_end,
            walk_forward_train_days=args.wf_train_days,
            walk_forward_test_days=args.wf_test_days,
            walk_forward_step_days=args.wf_step_days,
            walk_forward_embargo_days=args.wf_embargo_days,
            ranking_metric=ranking,
            min_sharpe=_parse_decimal(args.min_sharpe, name="min-sharpe"),
            promotion_thresholds=PromotionThresholds(
                min_trades=args.min_trades,
                min_profit_factor=_parse_decimal(args.min_profit_factor, name="min-profit-factor"),
                max_drawdown_pct=_parse_decimal(args.max_drawdown_pct, name="max-drawdown"),
                min_walk_forward_windows=args.min_wf_windows,
                max_cost_reconciliation_error_inr=_parse_decimal(
                    args.max_reconciliation_error_inr, name="max-reconciliation-error"
                ),
            ),
            cost_reconciliation_error_inr=reconciliation_error,
            cost_reconciliation_source=args.cost_reconciliation_source,
            tick_size_rupees=_parse_decimal(args.tick_size_rupees, name="tick-size"),
            tick_size_source=args.tick_size_source,
            exit_buffer_minutes=args.exit_buffer_minutes,
            cas_eligible=bool(args.cas_eligible),
            pricing_date=args.pricing_date,
        )
        result = run_discovery(frame, manifest, config)
        payload = result.to_dict()
    except Exception as exc:  # noqa: BLE001 - CLI must emit a machine-readable fail-closed result
        payload = {
            "schema_version": "intraday-discovery/v1",
            "track": "intraday",
            "status": "FAIL",
            "discovery_violations": [f"discovery error: {type(exc).__name__}: {exc}"],
            "live_orders_called": False,
        }
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            pass
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
