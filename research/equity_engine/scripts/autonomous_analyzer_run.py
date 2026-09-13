#!/usr/bin/env python3
"""Run a finite autonomous candidate batch in OpenAlgo Analyzer only.

This command never enables Analyzer mode and never accepts a live-execution
flag. The operator must enable Analyzer separately. The executor itself is
hard-wired to ``sandbox_place_order``.
"""

from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from equity_engine.autonomous_contract import candidate_from_dict
from equity_engine.autonomous_orchestrator import (
    AutonomousOrchestrator,
    DeterministicRiskGate,
    RiskLimits,
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


def _positive_decimal(value: str) -> Decimal:
    parsed = Decimal(value)
    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite decimal")
    return parsed


def _confidence(value: str) -> Decimal:
    parsed = Decimal(value)
    if not parsed.is_finite() or parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _load_candidates(path: Path):
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

    executor = OpenAlgoAnalyzerExecutor(api_key=api_key)
    orchestrator = AutonomousOrchestrator(
        risk_gate=DeterministicRiskGate(
            RiskLimits(
                max_order_notional=args.max_order_notional,
                max_quantity=args.max_quantity,
                min_expected_edge_bps=args.min_edge_bps,
                min_confidence=args.min_confidence,
            )
        ),
        executor=executor,
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
