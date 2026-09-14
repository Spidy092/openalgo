"""TradeCandidate -> portfolio risk -> Analyzer-only submission bridge.

This module deliberately lives outside ``research/equity_engine``: research may
produce a candidate, but platform services own portfolio state, deterministic
risk and execution.  The bridge has no live-order dependency.  Its executor is
required to advertise Analyzer mode, and the production snapshot adapter reads
positions directly from the sandbox service.

Market-data freshness is explicit.  Sandbox position MTM refresh can retain an
older mark when a quote refresh fails, so this adapter never fabricates a quote
timestamp.  The autonomous session must provide a verified market-data
``market_data_timestamp`` in :class:`AnalyzerRiskContext`; missing/stale values
are rejected by ``services.risk.evaluate_portfolio_order``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from services.risk import (
    PortfolioDecision,
    PortfolioIntent,
    PortfolioLimits,
    PortfolioPosition,
    PortfolioSnapshot,
    SymbolActivity,
    evaluate_portfolio_order,
)


class TradeCandidateLike(Protocol):
    candidate_id: str
    symbol: str
    exchange: str
    side: str
    quantity: int
    price_type: str
    entry_price: Decimal
    valid_until: datetime

    @property
    def fingerprint(self) -> str: ...

    def validate(self, *, now: datetime | None = None) -> None: ...


class AnalyzerExecutorLike(Protocol):
    mode: Any

    def submit(self, candidate: TradeCandidateLike) -> str: ...


AnalyzerPositionsReader = Callable[[], tuple[bool, dict[str, Any], int]]


class AnalyzerSnapshotUnavailable(RuntimeError):
    """Analyzer portfolio state could not be proven safe enough to evaluate."""


@dataclass(frozen=True, slots=True)
class AnalyzerRiskContext:
    """Point-in-time market/session evidence supplied by the autonomous loop.

    ``candidate_reference_price`` is mandatory for MARKET candidates.  LIMIT
    candidates can use their limit price, but when a verified current reference
    price is supplied the adapter uses the larger value for a conservative
    notional projection.
    """

    as_of: datetime
    market_open: bool | None
    market_data_timestamp: datetime | None
    candidate_reference_price: Decimal | None = None
    kill_switch_engaged: bool = False
    symbol_activity: tuple[SymbolActivity, ...] = ()


def _decimal(value: object, *, field: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise AnalyzerSnapshotUnavailable(f"{field} is missing or invalid")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise AnalyzerSnapshotUnavailable(f"{field} is not a finite number") from exc
    if not parsed.is_finite():
        raise AnalyzerSnapshotUnavailable(f"{field} is not a finite number")
    return parsed


def _portfolio_symbol(exchange: str, symbol: str) -> str:
    """Create the canonical portfolio-risk identifier for one cash equity."""

    return f"{exchange.strip().upper()}:{symbol.strip().upper()}"


class AnalyzerPortfolioSnapshotAdapter:
    """Build the canonical risk snapshot from Analyzer/sandbox positions only."""

    def __init__(
        self,
        *,
        api_key: str,
        positions_reader: AnalyzerPositionsReader | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._positions_reader = positions_reader

    def _read_positions(self) -> tuple[bool, dict[str, Any], int]:
        if self._positions_reader is not None:
            return self._positions_reader()

        # Direct sandbox dependency is intentional.  Do not replace this with
        # get_positionbook(auth_token=..., broker=...), whose internal-call
        # branch can read a live broker position book.
        from services.sandbox_service import sandbox_get_positions

        original_data = {"apikey": self._api_key}
        return sandbox_get_positions(self._api_key, original_data)

    def snapshot(self, context: AnalyzerRiskContext) -> PortfolioSnapshot:
        try:
            success, response, status_code = self._read_positions()
        except Exception as exc:  # snapshot acquisition is a hard safety boundary
            raise AnalyzerSnapshotUnavailable(
                f"analyzer position snapshot raised {type(exc).__name__}: {exc}"
            ) from exc

        if not success:
            message = response.get("message", "unknown error") if isinstance(response, dict) else "unknown error"
            raise AnalyzerSnapshotUnavailable(
                f"analyzer position snapshot failed ({status_code}): {message}"
            )
        if not isinstance(response, dict):
            raise AnalyzerSnapshotUnavailable("analyzer position snapshot response is not an object")
        if response.get("status") != "success":
            raise AnalyzerSnapshotUnavailable("analyzer position snapshot did not report success")
        if response.get("mode") != "analyze":
            raise AnalyzerSnapshotUnavailable("position snapshot is not proven to be Analyzer mode")

        rows = response.get("data")
        if not isinstance(rows, list):
            raise AnalyzerSnapshotUnavailable("analyzer position snapshot data must be a list")

        positions: list[PortfolioPosition] = []
        seen_keys: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise AnalyzerSnapshotUnavailable(f"position row {index} is not an object")

            quantity = row.get("quantity")
            if isinstance(quantity, bool) or not isinstance(quantity, int):
                raise AnalyzerSnapshotUnavailable(
                    f"position row {index} quantity must be a whole number"
                )
            if quantity == 0:
                # Closed rows still contribute to the aggregate realized PnL
                # fields below, but have no current exposure.
                continue

            symbol = str(row.get("symbol") or "").strip().upper()
            exchange = str(row.get("exchange") or "").strip().upper()
            product = str(row.get("product") or "").strip().upper()
            if not symbol:
                raise AnalyzerSnapshotUnavailable(f"position row {index} symbol is missing")
            if exchange not in {"NSE", "BSE"}:
                raise AnalyzerSnapshotUnavailable(
                    f"open position {symbol} is outside cash-equity NSE/BSE scope"
                )
            if product not in {"CNC", "MIS"}:
                raise AnalyzerSnapshotUnavailable(
                    f"open position {exchange}:{symbol} has unsupported product {product or '<missing>'}"
                )

            # Sandbox exposes contract_value as lot_size.  Autonomous equity v1
            # is intentionally unit-notional cash equity only; refusing any
            # other multiplier avoids understating exposure for derivatives or
            # crypto contracts.
            if "lot_size" not in row:
                raise AnalyzerSnapshotUnavailable(
                    f"open position {exchange}:{symbol} is missing lot_size evidence"
                )
            lot_size = _decimal(row.get("lot_size"), field=f"{exchange}:{symbol}.lot_size")
            if lot_size != Decimal("1"):
                raise AnalyzerSnapshotUnavailable(
                    f"open position {exchange}:{symbol} has unsupported lot_size {lot_size}"
                )

            mark = _decimal(row.get("ltp"), field=f"{exchange}:{symbol}.ltp")
            if mark <= 0:
                raise AnalyzerSnapshotUnavailable(
                    f"open position {exchange}:{symbol} has no positive Analyzer mark"
                )

            key = _portfolio_symbol(exchange, symbol)
            if key in seen_keys:
                # The current canonical PortfolioSnapshot is one row per risk
                # symbol.  Do not silently net CNC/MIS rows because doing so can
                # hide gross exposure.
                raise AnalyzerSnapshotUnavailable(
                    f"analyzer snapshot contains multiple open rows for {key}"
                )
            seen_keys.add(key)
            positions.append(PortfolioPosition(key, quantity, mark))

        realized = _decimal(
            response.get("total_today_realized_pnl"),
            field="total_today_realized_pnl",
        )
        unrealized = _decimal(
            response.get("total_unrealized_pnl"),
            field="total_unrealized_pnl",
        )

        return PortfolioSnapshot(
            as_of=context.as_of,
            positions=tuple(positions),
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            market_open=context.market_open,
            market_data_timestamp=context.market_data_timestamp,
            kill_switch_engaged=context.kill_switch_engaged,
            symbol_activity=context.symbol_activity,
        )


class TradeCandidatePortfolioIntentAdapter:
    """Translate a validated research candidate into canonical portfolio intent."""

    @staticmethod
    def intent(
        candidate: TradeCandidateLike,
        context: AnalyzerRiskContext,
        *,
        reduce_only: bool = False,
    ) -> PortfolioIntent:
        try:
            entry = Decimal(str(candidate.entry_price))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("candidate entry_price is not a finite number") from exc
        if not entry.is_finite() or entry <= 0:
            raise ValueError("candidate entry_price must be a positive finite number")

        current: Decimal | None = None
        if context.candidate_reference_price is not None:
            try:
                current = Decimal(str(context.candidate_reference_price))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ValueError("candidate_reference_price is not a finite number") from exc
            if not current.is_finite() or current <= 0:
                raise ValueError("candidate_reference_price must be a positive finite number")

        price_type = str(candidate.price_type).strip().upper()
        if price_type == "MARKET":
            if current is None:
                raise ValueError(
                    "MARKET candidate requires a verified candidate_reference_price"
                )
            reference_price = current
        elif price_type == "LIMIT":
            # A verified current price above the limit is conservatively useful
            # for risk projection (especially for short exposure).  Otherwise
            # the executable limit itself is the maximum known order price.
            reference_price = max(entry, current) if current is not None else entry
        else:
            raise ValueError("candidate price_type must be MARKET or LIMIT")

        return PortfolioIntent(
            symbol=_portfolio_symbol(candidate.exchange, candidate.symbol),
            side=candidate.side,
            quantity=candidate.quantity,
            reference_price=reference_price,
            reduce_only=reduce_only,
        )


@dataclass(frozen=True, slots=True)
class PortfolioAnalyzerResult:
    candidate_id: str
    candidate_fingerprint: str
    portfolio_decision: PortfolioDecision
    execution_id: str | None

    @property
    def submitted(self) -> bool:
        return self.execution_id is not None


class PortfolioAnalyzerBridge:
    """Non-bypassable portfolio gate immediately in front of Analyzer submit."""

    def __init__(
        self,
        *,
        limits: PortfolioLimits,
        snapshot_adapter: AnalyzerPortfolioSnapshotAdapter,
        analyzer_executor: AnalyzerExecutorLike,
        intent_adapter: TradeCandidatePortfolioIntentAdapter | None = None,
    ) -> None:
        mode = getattr(analyzer_executor, "mode", None)
        mode_value = getattr(mode, "value", mode)
        if str(mode_value).strip().lower() != "analyzer":
            raise ValueError("portfolio bridge accepts only an Analyzer executor")
        self._limits = limits
        self._snapshot_adapter = snapshot_adapter
        self._analyzer_executor = analyzer_executor
        self._intent_adapter = intent_adapter or TradeCandidatePortfolioIntentAdapter()

    def process(
        self,
        candidate: TradeCandidateLike,
        context: AnalyzerRiskContext,
        *,
        reduce_only: bool = False,
    ) -> PortfolioAnalyzerResult:
        # Validate against the exact snapshot time rather than a second implicit
        # clock.  The Analyzer executor validates again at submission, closing
        # the candidate-expiry race between this decision and sandbox submit.
        candidate.validate(now=context.as_of)

        snapshot = self._snapshot_adapter.snapshot(context)
        intent = self._intent_adapter.intent(
            candidate,
            context,
            reduce_only=reduce_only,
        )
        decision = evaluate_portfolio_order(self._limits, snapshot, intent)

        if not decision.allowed:
            return PortfolioAnalyzerResult(
                candidate_id=candidate.candidate_id,
                candidate_fingerprint=candidate.fingerprint,
                portfolio_decision=decision,
                execution_id=None,
            )

        execution_id = self._analyzer_executor.submit(candidate)
        if execution_id is None or not str(execution_id).strip():
            raise RuntimeError("Analyzer executor returned an empty execution id")

        return PortfolioAnalyzerResult(
            candidate_id=candidate.candidate_id,
            candidate_fingerprint=candidate.fingerprint,
            portfolio_decision=decision,
            execution_id=str(execution_id),
        )
