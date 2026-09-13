"""OpenAlgo Analyzer executor for autonomous equity v1.

This adapter is intentionally hard-wired to ``sandbox_place_order`` rather than
the generic order facade. Even if the platform analyzer toggle changes between
checks, this module has no code path to a live broker order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from equity_engine.autonomous_orchestrator import ExecutionMode, TradeCandidate


AnalyzerModeReader = Callable[[], bool]
SandboxPlaceOrder = Callable[
    [dict[str, Any], str, dict[str, Any]], tuple[bool, dict[str, Any], int]
]


class OpenAlgoAnalyzerExecutor:
    """Submit approved candidates only to OpenAlgo's sandbox/analyzer backend."""

    mode = ExecutionMode.ANALYZER

    def __init__(
        self,
        *,
        api_key: str,
        analyzer_mode_reader: AnalyzerModeReader | None = None,
        sandbox_place_order: SandboxPlaceOrder | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._analyzer_mode_reader = analyzer_mode_reader or self._default_analyzer_mode_reader
        self._sandbox_place_order = sandbox_place_order or self._default_sandbox_place_order

    @staticmethod
    def _default_analyzer_mode_reader() -> bool:
        from database.settings_db import get_analyze_mode

        return bool(get_analyze_mode())

    @staticmethod
    def _default_sandbox_place_order(
        order_data: dict[str, Any], api_key: str, original_data: dict[str, Any]
    ) -> tuple[bool, dict[str, Any], int]:
        from services.sandbox_service import sandbox_place_order

        return sandbox_place_order(order_data, api_key, original_data)

    @staticmethod
    def _order_payload(candidate: TradeCandidate) -> dict[str, Any]:
        candidate.validate()
        price = float(candidate.entry_price) if candidate.price_type == "LIMIT" else 0.0
        return {
            "strategy": f"Autonomous:{candidate.strategy_id}@{candidate.strategy_version}",
            "symbol": candidate.symbol,
            "exchange": candidate.exchange,
            "action": candidate.side,
            "quantity": candidate.quantity,
            "pricetype": candidate.price_type,
            "product": candidate.product,
            "price": price,
            "trigger_price": 0.0,
        }

    def submit(self, candidate: TradeCandidate) -> str:
        # This policy check is intentionally separate from the hard-wired
        # sandbox call. It stops a session when the operator turned Analyzer off,
        # while the direct sandbox dependency makes a race incapable of becoming
        # a live order.
        if not self._analyzer_mode_reader():
            raise RuntimeError("analyzer mode is not enabled; autonomous session stopped")

        order_data = self._order_payload(candidate)
        original_data = dict(order_data)
        success, response, status_code = self._sandbox_place_order(
            order_data, self._api_key, original_data
        )
        if not success:
            message = response.get("message", "sandbox order failed") if isinstance(response, dict) else "sandbox order failed"
            raise RuntimeError(f"analyzer order failed ({status_code}): {message}")

        order_id = response.get("orderid") if isinstance(response, dict) else None
        if order_id is None or not str(order_id).strip():
            raise RuntimeError("analyzer order succeeded without an orderid")
        return str(order_id)
