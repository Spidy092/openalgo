#!/usr/bin/env python3
"""Run a finite autonomous candidate batch in OpenAlgo Analyzer only.

This command never enables Analyzer mode and never accepts a live-execution
flag. The operator must enable Analyzer separately. Every candidate that passes
the research/order gate is then projected against the current Analyzer portfolio
before the hard-wired sandbox executor can be called.

``--portfolio-context`` is required evidence, not a convenience default. Its
``market_data_timestamp`` must be the oldest verified timestamp covering the
candidate reference prices and portfolio marks represented by the run. The
launcher uses the current UTC clock as ``as_of`` on every candidate so an old
context file expires instead of making old data appear fresh.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

# This script is the deliberate platform-integration surface. Keep the reusable
# equity_engine package independent of root OpenAlgo services, but make the CLI
# able to import the canonical platform risk implementation when launched from
# research/equity_engine or from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from equity_engine.autonomous_contract import candidate_from_dict
from equity_engine.autonomous_orchestrator import (
    AutonomousOrchestrator,
    DeterministicRiskGate,
    ExecutionRejected,
    RiskLimits,
    TradeCandidate,
)
from equity_engine.autonomous_session import (
    AutonomousSession,
    JsonlExecutionJournal,
    SessionLimits,
    SessionOutcome,
)
from equity_engine.openalgo_analyzer_executor import OpenAlgoAnalyzerExecutor
from equity_engine.shadow_session_health import (
    HealthStatus,
    HealthThresholds,
    check_persisted_session,
)
from services.autonomous import (
    AnalyzerPortfolioSnapshotAdapter,
    AnalyzerRiskContext,
    PortfolioAnalyzerBridge,
)
from services.risk import PortfolioLimits, SymbolActivity


def _positive_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite decimal")
    return parsed


def _confidence(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError("confidence must be a decimal") from exc
    if not parsed.is_finite() or parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1")
    return parsed


def _percentage(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError("percentage must be a decimal") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > 100:
        raise argparse.ArgumentTypeError("percentage must be in (0, 100]")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _aware_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed


def _evidence_decimal(value: object, *, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} must be a positive finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{field} must be a positive finite decimal")
    return parsed


def _load_candidates(path: Path) -> tuple[TradeCandidate, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if set(payload) != {"candidates"}:
            raise ValueError("candidate document object must contain only 'candidates'")
        rows = payload["candidates"]
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate document must contain a non-empty list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every candidate must be a JSON object")
    return tuple(candidate_from_dict(row) for row in rows)


@dataclass(frozen=True, slots=True)
class PortfolioContextEvidence:
    market_open: bool
    market_data_timestamp: datetime
    kill_switch_engaged: bool
    candidate_reference_prices: dict[str, Decimal]
    symbol_activity: tuple[SymbolActivity, ...]

    def context_for(
        self,
        candidate: TradeCandidate,
        *,
        now: datetime | None = None,
    ) -> AnalyzerRiskContext:
        current = now or datetime.now(timezone.utc)
        price = self.candidate_reference_prices.get(candidate.candidate_id)
        if price is None:
            # The loader requires exact candidate coverage; keep this guard for
            # programmatic callers that construct evidence directly.
            raise ExecutionRejected("portfolio_context_missing_candidate_reference_price")
        return AnalyzerRiskContext(
            as_of=current,
            market_open=self.market_open,
            market_data_timestamp=self.market_data_timestamp,
            candidate_reference_price=price,
            kill_switch_engaged=self.kill_switch_engaged,
            symbol_activity=self.symbol_activity,
        )


def _load_portfolio_context(
    path: Path,
    candidates: tuple[TradeCandidate, ...],
) -> PortfolioContextEvidence:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("portfolio context must be a JSON object")

    required = {
        "market_open",
        "market_data_timestamp",
        "kill_switch_engaged",
        "candidate_reference_prices",
        "symbol_activity",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(
            f"portfolio context keys mismatch; missing={missing}, extra={extra}"
        )

    market_open = payload["market_open"]
    if not isinstance(market_open, bool):
        raise ValueError("market_open must be boolean")
    kill_switch = payload["kill_switch_engaged"]
    if not isinstance(kill_switch, bool):
        raise ValueError("kill_switch_engaged must be boolean")
    market_timestamp = _aware_datetime(
        payload["market_data_timestamp"],
        field="market_data_timestamp",
    )

    references = payload["candidate_reference_prices"]
    if not isinstance(references, dict):
        raise ValueError("candidate_reference_prices must be an object")
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    reference_ids = {str(key) for key in references}
    if reference_ids != candidate_ids:
        missing = sorted(candidate_ids - reference_ids)
        extra = sorted(reference_ids - candidate_ids)
        raise ValueError(
            f"candidate_reference_prices keys mismatch; missing={missing}, extra={extra}"
        )
    parsed_references = {
        candidate_id: _evidence_decimal(
            references[candidate_id],
            field=f"candidate_reference_prices.{candidate_id}",
        )
        for candidate_id in sorted(candidate_ids)
    }

    activity_rows = payload["symbol_activity"]
    if not isinstance(activity_rows, list):
        raise ValueError("symbol_activity must be a list")
    activity: list[SymbolActivity] = []
    seen_symbols: set[str] = set()
    for index, row in enumerate(activity_rows):
        if not isinstance(row, dict) or set(row) != {"symbol", "last_increase_at"}:
            raise ValueError(
                f"symbol_activity[{index}] must contain only symbol and last_increase_at"
            )
        symbol = str(row["symbol"]).strip().upper()
        if not symbol or ":" not in symbol:
            raise ValueError(
                f"symbol_activity[{index}].symbol must use EXCHANGE:SYMBOL form"
            )
        exchange, trading_symbol = symbol.split(":", 1)
        if exchange not in {"NSE", "BSE"} or not trading_symbol:
            raise ValueError(
                f"symbol_activity[{index}].symbol must be NSE:SYMBOL or BSE:SYMBOL"
            )
        if symbol in seen_symbols:
            raise ValueError(f"symbol_activity contains duplicate symbol {symbol}")
        seen_symbols.add(symbol)
        activity.append(
            SymbolActivity(
                symbol=symbol,
                last_increase_at=_aware_datetime(
                    row["last_increase_at"],
                    field=f"symbol_activity[{index}].last_increase_at",
                ),
            )
        )

    return PortfolioContextEvidence(
        market_open=market_open,
        market_data_timestamp=market_timestamp,
        kill_switch_engaged=kill_switch,
        candidate_reference_prices=parsed_references,
        symbol_activity=tuple(activity),
    )


RiskContextProvider = Callable[[TradeCandidate], AnalyzerRiskContext]


class PortfolioGuardedAnalyzerExecutor:
    """Executor facade that makes portfolio approval mandatory before Analyzer."""

    def __init__(
        self,
        *,
        bridge: PortfolioAnalyzerBridge,
        context_provider: RiskContextProvider,
        mode: object,
    ) -> None:
        self._bridge = bridge
        self._context_provider = context_provider
        self.mode = mode

    def submit(self, candidate: TradeCandidate) -> str:
        context = self._context_provider(candidate)
        result = self._bridge.process(candidate, context)
        if not result.portfolio_decision.allowed:
            reasons = tuple(
                f"portfolio_{code.value}:{reason}"
                for code, reason in zip(
                    result.portfolio_decision.codes,
                    result.portfolio_decision.reasons,
                    strict=True,
                )
            )
            raise ExecutionRejected(*reasons)
        if result.execution_id is None or not result.execution_id.strip():
            raise RuntimeError("portfolio-approved Analyzer submission returned no execution id")
        return result.execution_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument(
        "--api-key-env",
        required=True,
        help="Name of the environment variable containing the OpenAlgo API key; the value is never printed.",
    )
    parser.add_argument("--max-order-notional", type=_positive_decimal, required=True)
    parser.add_argument("--max-quantity", type=_positive_int, required=True)
    parser.add_argument("--min-edge-bps", type=_positive_decimal, required=True)
    parser.add_argument("--min-confidence", type=_confidence, required=True)
    parser.add_argument("--max-orders", type=_positive_int, required=True)
    parser.add_argument("--max-gross-notional", type=_positive_decimal, required=True)
    parser.add_argument("--max-orders-per-symbol", type=_positive_int, required=True)

    portfolio = parser.add_argument_group("mandatory portfolio risk")
    portfolio.add_argument(
        "--portfolio-context",
        type=Path,
        required=True,
        help=(
            "Strict JSON market/risk evidence. market_data_timestamp must be the oldest "
            "verified timestamp covering candidate reference prices and Analyzer portfolio marks."
        ),
    )
    portfolio.add_argument(
        "--max-portfolio-gross-exposure", type=_positive_decimal, required=True
    )
    portfolio.add_argument(
        "--max-portfolio-net-exposure", type=_positive_decimal, required=True
    )
    portfolio.add_argument("--max-open-positions", type=_positive_int, required=True)
    portfolio.add_argument("--max-symbol-exposure", type=_positive_decimal, required=True)
    portfolio.add_argument(
        "--max-symbol-concentration-pct", type=_percentage, required=True
    )
    portfolio.add_argument("--max-daily-loss", type=_positive_decimal, required=True)
    portfolio.add_argument("--cooldown-seconds", type=_nonnegative_int, required=True)
    portfolio.add_argument(
        "--max-market-data-age-seconds", type=_nonnegative_int, required=True
    )

    parser.add_argument(
        "--shadow-health-dir",
        type=Path,
        default=None,
        help="Optional persisted shadow session directory. When supplied, every candidate requires HEALTHY status.",
    )
    parser.add_argument(
        "--health-freshness-seconds",
        type=float,
        default=60.0,
        help="Freshness threshold used only with --shadow-health-dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"environment variable {args.api_key_env!r} is missing or empty")

    candidates = _load_candidates(args.candidates)
    portfolio_evidence = _load_portfolio_context(args.portfolio_context, candidates)

    health_check = None
    if args.shadow_health_dir is not None:
        if args.health_freshness_seconds <= 0:
            raise SystemExit("--health-freshness-seconds must be positive")
        thresholds = HealthThresholds(freshness_seconds=args.health_freshness_seconds)

        def health_check() -> bool:
            report = check_persisted_session(args.shadow_health_dir, thresholds=thresholds)
            return (
                report.status is HealthStatus.HEALTHY
                and report.allow_new_theoretical_trades
                and not report.live_orders_called
            )

    raw_analyzer_executor = OpenAlgoAnalyzerExecutor(api_key=api_key)
    portfolio_bridge = PortfolioAnalyzerBridge(
        limits=PortfolioLimits(
            max_gross_exposure=args.max_portfolio_gross_exposure,
            max_abs_net_exposure=args.max_portfolio_net_exposure,
            max_open_positions=args.max_open_positions,
            max_symbol_exposure=args.max_symbol_exposure,
            max_symbol_concentration_pct=args.max_symbol_concentration_pct,
            max_daily_loss=args.max_daily_loss,
            cooldown_seconds=args.cooldown_seconds,
            max_market_data_age_seconds=args.max_market_data_age_seconds,
        ),
        snapshot_adapter=AnalyzerPortfolioSnapshotAdapter(api_key=api_key),
        analyzer_executor=raw_analyzer_executor,
    )
    guarded_executor = PortfolioGuardedAnalyzerExecutor(
        bridge=portfolio_bridge,
        context_provider=portfolio_evidence.context_for,
        mode=raw_analyzer_executor.mode,
    )

    orchestrator = AutonomousOrchestrator(
        risk_gate=DeterministicRiskGate(
            RiskLimits(
                max_order_notional=args.max_order_notional,
                max_quantity=args.max_quantity,
                min_expected_edge_bps=args.min_edge_bps,
                min_confidence=args.min_confidence,
            )
        ),
        executor=guarded_executor,
    )
    session = AutonomousSession(
        orchestrator=orchestrator,
        journal=JsonlExecutionJournal(args.journal),
        limits=SessionLimits(
            max_orders=args.max_orders,
            max_gross_notional=args.max_gross_notional,
            max_orders_per_symbol=args.max_orders_per_symbol,
        ),
        health_check=health_check,
    )
    entries = session.run(candidates)
    summary: dict[str, Any] = {
        "processed": len(entries),
        "executed": sum(entry.outcome is SessionOutcome.EXECUTED for entry in entries),
        "rejected": sum(entry.outcome is SessionOutcome.REJECTED for entry in entries),
        "duplicates": sum(entry.outcome is SessionOutcome.DUPLICATE for entry in entries),
        "stopped": sum(entry.outcome is SessionOutcome.STOPPED for entry in entries),
        "errors": sum(entry.outcome is SessionOutcome.ERROR for entry in entries),
        "journal": str(args.journal),
        "outcomes": [entry.as_dict() for entry in entries],
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 1 if summary["errors"] or summary["stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
