"""Append-only audit trail for the autonomous bridge.

Two rules, mirrored from ``services/agent/safety/audit.py``:

- Nothing secret is ever written. Order intents and rail inputs carry no
  credential here (the bridge resolves auth downstream), but this module still
  refuses to serialize obvious secret-shaped keys as a defence in depth.
- A write failure never blocks or unblocks a dispatch. The audit is evidence,
  not a control. Every entry point swallows its own exception after logging.

Each dispatch writes at least two rows: ``attempt`` before the rails/dispatch
and ``result`` after. A ``decision`` row records the rails verdict. An attempt
with no matching result is a dispatch that hung or a worker that died mid-call.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

PHASE_ATTEMPT = "attempt"
PHASE_DECISION = "decision"
PHASE_RESULT = "result"

REDACTED = "[redacted]"

_SECRET_KEY_MARKERS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "api-key",
    "secret",
    "token",
    "password",
    "authorization",
    "auth_token",
    "credential",
    "cookie",
    "pepper",
    "private",
    "passphrase",
    "totp",
)

_SAFE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "instrument_token",
        "exchange_token",
        "symbol_token",
        "token",
        "tokens",
    }
)

_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
    re.compile(r"^[0-9a-f]{48,}$", re.IGNORECASE),
)

_MAX_STRING = 2000


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            value = value[:_MAX_STRING] + "...[truncated]"
        for pattern in _SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                return REDACTED
        return value
    if isinstance(value, Mapping):
        return redact(value)
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def redact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip credential-shaped keys and values from a mapping (defence in depth)."""
    out: dict[str, Any] = {}
    for key, value in payload.items():
        lowered = str(key).lower()
        if lowered not in _SAFE_KEY_NAMES and any(
            marker in lowered for marker in _SECRET_KEY_MARKERS
        ):
            out[str(key)] = REDACTED
            continue
        out[str(key)] = _redact_value(value)
    return out


class AuditTrail:
    """Append-only JSONL audit trail. All writes are best-effort and non-blocking."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _append(self, phase: str, session_id: str, payload: Mapping[str, Any]) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "phase": phase,
            "session_id": str(session_id),
            "live_orders_called": False,
            **redact(payload),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        except Exception:  # noqa: BLE001 - audit must never block a decision
            logger.exception("autonomous bridge audit write failed (phase=%s)", phase)

    def attempt(self, session_id: str, payload: Mapping[str, Any]) -> None:
        self._append(PHASE_ATTEMPT, session_id, payload)

    def decision(self, session_id: str, payload: Mapping[str, Any]) -> None:
        self._append(PHASE_DECISION, session_id, payload)

    def result(self, session_id: str, payload: Mapping[str, Any]) -> None:
        self._append(PHASE_RESULT, session_id, payload)
