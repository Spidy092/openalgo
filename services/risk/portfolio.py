"""Pure deterministic portfolio-level pre-trade risk evaluation.

This module performs no I/O. Callers provide the proposed order, a point-in-time
portfolio snapshot and immutable limits. The engine projects the portfolio after
the order and fails closed when any required market or portfolio state is
missing or invalid.

Per-order syntactic and broker-facing checks remain the responsibility of the
agent/order guard. This layer owns portfolio-wide exposure policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum

HUNDRED = Decimal("100")


class PortfolioRiskCode(StrEnum):
    OK = "ok"
    INVALID_INTENT = "invalid_intent"
    INVALID_SNAPSHOT = "invalid_snapshot"
    MARKET_STATE_UNKNOWN = "market_state_unknown"
    MARKET_CLOSED = "market_closed"
    MARKET_DATA_MISSING = "market_data_missing"
    MARKET_DATA_STALE = "market_data_stale"
    MARKET_DATA_FROM_FUTURE = "market_data_from_future"
    KILL_SWITCH = "kill_switch"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    COOLDOWN = "cooldown"
    MAX_OPEN_POSITIONS = "max_open_positions"
    MAX_GROSS_EXPOSURE = "max_gross_exposure"
    MAX_NET_EXPOSURE = "max_net_exposure"
    MAX_SYMBOL_EXPOSURE = "max_symbol_exposure"
    MAX_SYMBOL_CONCENTRATION = "max_symbol_concentration"
    REDUCE_ONLY_VIOLATION = "reduce_only_violation"


def _decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _aware(value: datetime | None) -> bool:
    return value is not None and value.tzinfo is not None and value.utcoffset() is not None


@dataclass(frozen=True, slots=True)
class PortfolioLimits:
    max_gross_exposure: Decimal
    max_open_positions: int
    max_symbol_exposure: Decimal
    max_symbol_concentration_pct: Decimal
    max_daily_loss: Decimal
    cooldown_seconds: int
    max_market_data_age_seconds: int
    max_abs_net_exposure: Decimal | None = None

    def __post_init__(self) -> None:
        positive_money = (
            ("max_gross_exposure", self.max_gross_exposure),
            ("max_symbol_exposure", self.max_symbol_exposure),
            ("max_daily_loss", self.max_daily_loss),
        )
        for name, raw in positive_money:
            value = _decimal(raw)
            if value is None or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, value)
        concentration = _decimal(self.max_symbol_concentration_pct)
        if concentration is None or not Decimal("0") < concentration <= HUNDRED:
            raise ValueError("max_symbol_concentration_pct must be in (0, 100]")
        object.__setattr__(self, "max_symbol_concentration_pct", concentration)
        if (
            isinstance(self.max_open_positions, bool)
            or not isinstance(self.max_open_positions, int)
            or self.max_open_positions <= 0
        ):
            raise ValueError("max_open_positions must be a positive whole number")
        if (
            isinstance(self.cooldown_seconds, bool)
            or not isinstance(self.cooldown_seconds, int)
            or self.cooldown_seconds < 0
        ):
            raise ValueError("cooldown_seconds must be a non-negative whole number")
        if (
            isinstance(self.max_market_data_age_seconds, bool)
            or not isinstance(self.max_market_data_age_seconds, int)
            or self.max_market_data_age_seconds < 0
        ):
            raise ValueError("max_market_data_age_seconds must be a non-negative whole number")
        if self.max_abs_net_exposure is not None:
            net = _decimal(self.max_abs_net_exposure)
            if net is None or net <= 0:
                raise ValueError("max_abs_net_exposure must be positive when configured")
            object.__setattr__(self, "max_abs_net_exposure", net)


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    symbol: str
    quantity: int
    mark_price: Decimal

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.strip().upper()

    @property
    def gross_notional(self) -> Decimal:
        return abs(Decimal(self.quantity) * self.mark_price)

    @property
    def signed_notional(self) -> Decimal:
        return Decimal(self.quantity) * self.mark_price


@dataclass(frozen=True, slots=True)
class SymbolActivity:
    symbol: str
    last_increase_at: datetime

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.strip().upper()


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    as_of: datetime
    positions: tuple[PortfolioPosition, ...]
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    market_open: bool | None
    market_data_timestamp: datetime | None
    kill_switch_engaged: bool = False
    symbol_activity: tuple[SymbolActivity, ...] = ()

    @property
    def total_pnl(self) -> Decimal:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def gross_exposure(self) -> Decimal:
        return sum((item.gross_notional for item in self.positions), Decimal("0"))

    @property
    def net_exposure(self) -> Decimal:
        return sum((item.signed_notional for item in self.positions), Decimal("0"))

    @property
    def open_positions(self) -> int:
        return sum(1 for item in self.positions if item.quantity != 0)


@dataclass(frozen=True, slots=True)
class PortfolioIntent:
    symbol: str
    side: str
    quantity: int
    reference_price: Decimal
    reduce_only: bool = False

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.strip().upper()

    @property
    def signed_quantity(self) -> int:
        return self.quantity if self.side.strip().upper() == "BUY" else -self.quantity


@dataclass(frozen=True, slots=True)
class PortfolioDecision:
    allowed: bool
    codes: tuple[PortfolioRiskCode, ...]
    reasons: tuple[str, ...]
    risk_reducing: bool
    current_gross_exposure: Decimal
    projected_gross_exposure: Decimal
    projected_net_exposure: Decimal
    projected_symbol_exposure: Decimal
    projected_symbol_concentration_pct: Decimal
    projected_open_positions: int

    @property
    def primary_code(self) -> PortfolioRiskCode:
        return self.codes[0] if self.codes else PortfolioRiskCode.OK


def _invalid_decision(
    code: PortfolioRiskCode,
    reason: str,
    snapshot: PortfolioSnapshot | None = None,
) -> PortfolioDecision:
    current_gross = snapshot.gross_exposure if snapshot is not None else Decimal("0")
    current_net = snapshot.net_exposure if snapshot is not None else Decimal("0")
    current_positions = snapshot.open_positions if snapshot is not None else 0
    return PortfolioDecision(
        allowed=False,
        codes=(code,),
        reasons=(reason,),
        risk_reducing=False,
        current_gross_exposure=current_gross,
        projected_gross_exposure=current_gross,
        projected_net_exposure=current_net,
        projected_symbol_exposure=Decimal("0"),
        projected_symbol_concentration_pct=Decimal("0"),
        projected_open_positions=current_positions,
    )


def _validate_intent(intent: PortfolioIntent) -> str | None:
    if not intent.normalized_symbol:
        return "symbol is required"
    if intent.side.strip().upper() not in {"BUY", "SELL"}:
        return "side must be BUY or SELL"
    if isinstance(intent.quantity, bool) or not isinstance(intent.quantity, int) or intent.quantity <= 0:
        return "quantity must be a positive whole number"
    price = _decimal(intent.reference_price)
    if price is None or price <= 0:
        return "reference_price must be a positive finite number"
    return None


def _validate_snapshot(snapshot: PortfolioSnapshot) -> str | None:
    if not _aware(snapshot.as_of):
        return "snapshot.as_of must be timezone-aware"
    if _decimal(snapshot.realized_pnl) is None or _decimal(snapshot.unrealized_pnl) is None:
        return "portfolio PnL values must be finite"

    symbols: set[str] = set()
    for position in snapshot.positions:
        symbol = position.normalized_symbol
        if not symbol:
            return "portfolio position symbol is required"
        if symbol in symbols:
            return f"portfolio snapshot contains duplicate symbol {symbol}"
        symbols.add(symbol)
        if isinstance(position.quantity, bool) or not isinstance(position.quantity, int):
            return f"position quantity for {symbol} must be a whole number"
        mark = _decimal(position.mark_price)
        if mark is None or mark <= 0:
            return f"mark price for {symbol} must be a positive finite number"

    activity_symbols: set[str] = set()
    for activity in snapshot.symbol_activity:
        symbol = activity.normalized_symbol
        if not symbol:
            return "symbol activity requires a symbol"
        if symbol in activity_symbols:
            return f"portfolio snapshot contains duplicate activity for {symbol}"
        activity_symbols.add(symbol)
        if not _aware(activity.last_increase_at):
            return f"last_increase_at for {symbol} must be timezone-aware"
        if activity.last_increase_at > snapshot.as_of:
            return f"last_increase_at for {symbol} is in the future"
    return None


def evaluate_portfolio_order(
    limits: PortfolioLimits,
    snapshot: PortfolioSnapshot,
    intent: PortfolioIntent,
) -> PortfolioDecision:
    """Project one order against portfolio-wide limits.

    Verified ``reduce_only`` intents are permitted through the kill switch and
    daily-loss/cooldown halts because they can only shrink an existing position.
    They still require known-open market state and fresh market data, and they
    cannot flip a position through zero.
    """
    intent_error = _validate_intent(intent)
    if intent_error:
        return _invalid_decision(PortfolioRiskCode.INVALID_INTENT, intent_error, snapshot)
    snapshot_error = _validate_snapshot(snapshot)
    if snapshot_error:
        return _invalid_decision(PortfolioRiskCode.INVALID_SNAPSHOT, snapshot_error, snapshot)

    if snapshot.market_open is None:
        return _invalid_decision(
            PortfolioRiskCode.MARKET_STATE_UNKNOWN,
            "market-open state is unknown",
            snapshot,
        )
    if snapshot.market_open is False:
        return _invalid_decision(PortfolioRiskCode.MARKET_CLOSED, "market is closed", snapshot)
    if snapshot.market_data_timestamp is None:
        return _invalid_decision(
            PortfolioRiskCode.MARKET_DATA_MISSING,
            "market-data timestamp is required",
            snapshot,
        )
    if not _aware(snapshot.market_data_timestamp):
        return _invalid_decision(
            PortfolioRiskCode.INVALID_SNAPSHOT,
            "market_data_timestamp must be timezone-aware",
            snapshot,
        )
    age = snapshot.as_of - snapshot.market_data_timestamp
    if age.total_seconds() < 0:
        return _invalid_decision(
            PortfolioRiskCode.MARKET_DATA_FROM_FUTURE,
            "market-data timestamp is later than snapshot.as_of",
            snapshot,
        )
    if age.total_seconds() > limits.max_market_data_age_seconds:
        return _invalid_decision(
            PortfolioRiskCode.MARKET_DATA_STALE,
            f"market data age {age.total_seconds():.3f}s exceeds {limits.max_market_data_age_seconds}s",
            snapshot,
        )

    symbol = intent.normalized_symbol
    side = intent.side.strip().upper()
    reference_price = _decimal(intent.reference_price)
    assert reference_price is not None

    by_symbol = {item.normalized_symbol: item for item in snapshot.positions}
    current = by_symbol.get(symbol)
    current_quantity = current.quantity if current is not None else 0
    projected_quantity = current_quantity + (intent.quantity if side == "BUY" else -intent.quantity)

    risk_reducing = False
    if intent.reduce_only:
        opposite = (current_quantity > 0 and side == "SELL") or (current_quantity < 0 and side == "BUY")
        no_flip = (
            projected_quantity == 0
            or (current_quantity > 0 and projected_quantity > 0)
            or (current_quantity < 0 and projected_quantity < 0)
        )
        if current_quantity == 0 or not opposite or not no_flip or abs(projected_quantity) > abs(current_quantity):
            return _invalid_decision(
                PortfolioRiskCode.REDUCE_ONLY_VIOLATION,
                "reduce_only intent must shrink an existing position without flipping its side",
                snapshot,
            )
        risk_reducing = True

    other_gross = sum(
        (item.gross_notional for item in snapshot.positions if item.normalized_symbol != symbol),
        Decimal("0"),
    )
    other_net = sum(
        (item.signed_notional for item in snapshot.positions if item.normalized_symbol != symbol),
        Decimal("0"),
    )
    projected_symbol = abs(Decimal(projected_quantity) * reference_price)
    projected_gross = other_gross + projected_symbol
    projected_net = other_net + (Decimal(projected_quantity) * reference_price)
    projected_open = sum(
        1 for item in snapshot.positions if item.normalized_symbol != symbol and item.quantity != 0
    ) + (1 if projected_quantity != 0 else 0)
    concentration = (
        (projected_symbol / projected_gross) * HUNDRED
        if projected_gross > 0
        else Decimal("0")
    )

    codes: list[PortfolioRiskCode] = []
    reasons: list[str] = []

    def block(code: PortfolioRiskCode, reason: str) -> None:
        codes.append(code)
        reasons.append(reason)

    if snapshot.kill_switch_engaged and not risk_reducing:
        block(PortfolioRiskCode.KILL_SWITCH, "portfolio kill switch is engaged")

    if snapshot.total_pnl <= -limits.max_daily_loss and not risk_reducing:
        block(
            PortfolioRiskCode.DAILY_LOSS_LIMIT,
            f"daily PnL {snapshot.total_pnl} is at or below loss limit {-limits.max_daily_loss}",
        )

    if not risk_reducing and limits.cooldown_seconds > 0:
        last_by_symbol = {
            item.normalized_symbol: item.last_increase_at for item in snapshot.symbol_activity
        }
        last = last_by_symbol.get(symbol)
        if last is not None:
            elapsed = (snapshot.as_of - last).total_seconds()
            if elapsed < limits.cooldown_seconds:
                block(
                    PortfolioRiskCode.COOLDOWN,
                    f"{symbol} cooldown has {limits.cooldown_seconds - elapsed:.3f}s remaining",
                )

    current_gross = snapshot.gross_exposure
    current_net_abs = abs(snapshot.net_exposure)
    current_symbol = current.gross_notional if current is not None else Decimal("0")
    current_open = snapshot.open_positions
    current_concentration = (
        (current_symbol / current_gross) * HUNDRED if current_gross > 0 else Decimal("0")
    )

    if projected_open > limits.max_open_positions and (
        not risk_reducing or projected_open > current_open
    ):
        block(
            PortfolioRiskCode.MAX_OPEN_POSITIONS,
            f"projected open positions {projected_open} exceeds limit {limits.max_open_positions}",
        )
    if projected_gross > limits.max_gross_exposure and (
        not risk_reducing or projected_gross > current_gross
    ):
        block(
            PortfolioRiskCode.MAX_GROSS_EXPOSURE,
            f"projected gross exposure {projected_gross} exceeds limit {limits.max_gross_exposure}",
        )
    if limits.max_abs_net_exposure is not None:
        projected_net_abs = abs(projected_net)
        if projected_net_abs > limits.max_abs_net_exposure and (
            not risk_reducing or projected_net_abs > current_net_abs
        ):
            block(
                PortfolioRiskCode.MAX_NET_EXPOSURE,
                f"projected absolute net exposure {projected_net_abs} exceeds limit {limits.max_abs_net_exposure}",
            )
    if projected_symbol > limits.max_symbol_exposure and (
        not risk_reducing or projected_symbol > current_symbol
    ):
        block(
            PortfolioRiskCode.MAX_SYMBOL_EXPOSURE,
            f"projected {symbol} exposure {projected_symbol} exceeds limit {limits.max_symbol_exposure}",
        )
    if concentration > limits.max_symbol_concentration_pct and (
        not risk_reducing or concentration > current_concentration
    ):
        block(
            PortfolioRiskCode.MAX_SYMBOL_CONCENTRATION,
            f"projected {symbol} concentration {concentration}% exceeds limit {limits.max_symbol_concentration_pct}%",
        )

    return PortfolioDecision(
        allowed=not codes,
        codes=tuple(codes) if codes else (PortfolioRiskCode.OK,),
        reasons=tuple(reasons),
        risk_reducing=risk_reducing,
        current_gross_exposure=current_gross,
        projected_gross_exposure=projected_gross,
        projected_net_exposure=projected_net,
        projected_symbol_exposure=projected_symbol,
        projected_symbol_concentration_pct=concentration,
        projected_open_positions=projected_open,
    )


__all__ = [
    "PortfolioDecision",
    "PortfolioIntent",
    "PortfolioLimits",
    "PortfolioPosition",
    "PortfolioRiskCode",
    "PortfolioSnapshot",
    "SymbolActivity",
    "evaluate_portfolio_order",
]
