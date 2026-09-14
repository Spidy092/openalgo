"""Convert a research trade-candidate document into a stopped strategy config.

The bridge is deliberately configuration-only.  It cannot start a strategy,
enable live mode, or dispatch an order.  The HTTP layer persists the returned
configuration through the Strategy Module's normal validator and store.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

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


class ResearchCandidateError(ValueError):
    """The research document cannot safely become a Strategy Module config."""


def _text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ResearchCandidateError(f"{field} must be non-empty text")
    return value.strip()


def _decimal(payload: Mapping[str, Any], field: str) -> Decimal:
    value = payload.get(field)
    if value is None or isinstance(value, bool):
        raise ResearchCandidateError(f"{field} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ResearchCandidateError(f"{field} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ResearchCandidateError(f"{field} must be a finite decimal")
    return parsed


def _fingerprint(payload: Mapping[str, Any], field: str) -> str:
    value = _text(payload, field).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ResearchCandidateError(f"{field} must be a SHA-256 hex fingerprint")
    return value


def candidate_to_strategy_config(
    payload: Mapping[str, Any], *, now: datetime | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate one v1 candidate and return config plus audit provenance.

    LIMIT candidates fail closed because the Strategy Module currently has no
    field in which to preserve their limit price.  Silently changing one to a
    market strategy would change the researched execution semantics.
    """

    if not isinstance(payload, Mapping):
        raise ResearchCandidateError("candidate must be a JSON object")
    keys = frozenset(str(key) for key in payload)
    unknown = sorted(keys - _FIELDS)
    missing = sorted(_FIELDS - keys)
    if unknown:
        raise ResearchCandidateError(f"unknown candidate fields: {', '.join(unknown)}")
    if missing:
        raise ResearchCandidateError(f"missing candidate fields: {', '.join(missing)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ResearchCandidateError("unsupported candidate schema_version")

    candidate_id = _text(payload, "candidate_id")
    strategy_id = _text(payload, "strategy_id")
    strategy_version = _text(payload, "strategy_version")
    symbol = _text(payload, "symbol").upper()
    exchange = _text(payload, "exchange").upper()
    side = _text(payload, "side").upper()
    product = _text(payload, "product").upper()
    price_type = _text(payload, "price_type").upper()

    if exchange not in {"NSE", "BSE"}:
        raise ResearchCandidateError("only NSE/BSE cash-equity candidates are supported")
    if side not in {"BUY", "SELL"}:
        raise ResearchCandidateError("side must be BUY or SELL")
    if product not in {"CNC", "MIS"}:
        raise ResearchCandidateError("product must be CNC or MIS")
    if price_type != "MARKET":
        raise ResearchCandidateError(
            "only MARKET candidates can be imported; LIMIT price semantics cannot be preserved"
        )
    if side == "SELL" and product != "MIS":
        raise ResearchCandidateError("a short cash-equity strategy must use MIS")

    quantity = payload.get("quantity")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ResearchCandidateError("quantity must be a positive whole number")

    entry = _decimal(payload, "entry_price")
    stop = _decimal(payload, "stop_price")
    target = _decimal(payload, "target_price")
    edge = _decimal(payload, "expected_edge_bps")
    confidence = _decimal(payload, "confidence")
    if min(entry, stop, target) <= 0:
        raise ResearchCandidateError("entry, stop and target prices must be positive")
    if edge <= 0:
        raise ResearchCandidateError("expected_edge_bps must be positive")
    if confidence < 0 or confidence > 1:
        raise ResearchCandidateError("confidence must be between 0 and 1")
    if side == "BUY" and not (stop < entry < target):
        raise ResearchCandidateError("BUY candidate requires stop < entry < target")
    if side == "SELL" and not (target < entry < stop):
        raise ResearchCandidateError("SELL candidate requires target < entry < stop")

    try:
        valid_until = datetime.fromisoformat(
            _text(payload, "valid_until").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ResearchCandidateError("valid_until must be an ISO-8601 timestamp") from exc
    if valid_until.tzinfo is None or valid_until.utcoffset() is None:
        raise ResearchCandidateError("valid_until must be timezone-aware")
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if valid_until <= current:
        raise ResearchCandidateError("candidate is stale")

    dataset_fingerprint = _fingerprint(payload, "dataset_fingerprint")
    research_fingerprint = _fingerprint(payload, "research_fingerprint")
    stop_distance = abs(entry - stop)
    target_distance = abs(target - entry)

    # Saved imports are inert: stopped, live-disabled (store defaults), and
    # scheduler-disabled with sandbox as the only configured default mode.
    config = {
        "name": f"Research {strategy_id} {candidate_id}"[:200],
        "strategy_kind": "signal",
        "direction": "long_only" if side == "BUY" else "short_only",
        "universe_tab": "stocks_fno",
        "underlying": symbol,
        "underlying_exchange": exchange,
        "strategy_type": "intraday",
        "entry_time": "09:15",
        "exit_time": "15:20",
        "product": product,
        "pricetype": "MARKET",
        "legs": [
            {
                "id": 1,
                "symbol": symbol,
                "exchange": exchange,
                "side": "long" if side == "BUY" else "short",
                "segment": "cash",
                "qty_mode": "units",
                "qty": quantity,
                "risk_unit": "points",
                "sl_pts": float(stop_distance),
                "target_pts": float(target_distance),
            }
        ],
        "scheduler": {
            "enabled": False,
            "days": [],
            "start_time": None,
            "auto_stop_time": None,
            "default_mode": "sandbox",
        },
    }
    provenance = {
        "candidate_id": candidate_id,
        "strategy_id": strategy_id,
        "strategy_version": strategy_version,
        "dataset_fingerprint": dataset_fingerprint,
        "research_fingerprint": research_fingerprint,
    }
    return config, provenance


__all__ = [
    "ResearchCandidateError",
    "SCHEMA_VERSION",
    "candidate_to_strategy_config",
]
