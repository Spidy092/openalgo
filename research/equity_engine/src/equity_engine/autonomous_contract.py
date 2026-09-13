"""Strict JSON serialization for autonomous trade candidates.

Schema drift fails closed. Unknown fields are rejected so a producer cannot add
execution semantics that the consumer silently ignores.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping

from equity_engine.autonomous_orchestrator import TradeCandidate

SCHEMA_VERSION = "equity-trade-candidate/v1"

_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "symbol",
        "exchange",
        "strategy_id",
        "strategy_version",
        "side",
        "quantity",
        "product",
        "price_type",
        "entry_price",
        "stop_price",
        "target_price",
        "expected_edge_bps",
        "confidence",
        "valid_until",
        "dataset_fingerprint",
        "research_fingerprint",
    }
)


def candidate_to_dict(candidate: TradeCandidate) -> dict[str, Any]:
    candidate.validate()
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "symbol": candidate.symbol,
        "exchange": candidate.exchange,
        "strategy_id": candidate.strategy_id,
        "strategy_version": candidate.strategy_version,
        "side": candidate.side,
        "quantity": candidate.quantity,
        "product": candidate.product,
        "price_type": candidate.price_type,
        "entry_price": str(candidate.entry_price),
        "stop_price": str(candidate.stop_price),
        "target_price": str(candidate.target_price),
        "expected_edge_bps": str(candidate.expected_edge_bps),
        "confidence": str(candidate.confidence),
        "valid_until": candidate.valid_until.isoformat(),
        "dataset_fingerprint": candidate.dataset_fingerprint,
        "research_fingerprint": candidate.research_fingerprint,
    }


def candidate_from_dict(payload: Mapping[str, Any]) -> TradeCandidate:
    keys = frozenset(str(key) for key in payload)
    unknown = sorted(keys - _FIELDS)
    missing = sorted(_FIELDS - keys)
    if unknown:
        raise ValueError(f"unknown trade-candidate fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing trade-candidate fields: {', '.join(missing)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported trade-candidate schema: {payload.get('schema_version')!r}")

    try:
        candidate = TradeCandidate(
            candidate_id=str(payload["candidate_id"]),
            symbol=str(payload["symbol"]),
            exchange=str(payload["exchange"]),
            strategy_id=str(payload["strategy_id"]),
            strategy_version=str(payload["strategy_version"]),
            side=str(payload["side"]),
            quantity=int(payload["quantity"]),
            product=str(payload["product"]),
            price_type=str(payload["price_type"]),
            entry_price=Decimal(str(payload["entry_price"])),
            stop_price=Decimal(str(payload["stop_price"])),
            target_price=Decimal(str(payload["target_price"])),
            expected_edge_bps=Decimal(str(payload["expected_edge_bps"])),
            confidence=Decimal(str(payload["confidence"])),
            valid_until=datetime.fromisoformat(str(payload["valid_until"]).replace("Z", "+00:00")),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            research_fingerprint=str(payload["research_fingerprint"]),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid trade candidate: {exc}") from exc
    candidate.validate()
    return candidate


__all__ = ["SCHEMA_VERSION", "candidate_from_dict", "candidate_to_dict"]
