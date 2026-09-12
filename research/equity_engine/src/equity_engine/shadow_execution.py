"""Live shadow execution V1 (read-only, never sends an order).

Monday live-market shadow path:

read-only market event
-> validated snapshot
-> session/CAS filter
-> strategy signal (next-bar execution, no same-bar lookahead)
-> risk/capital check
-> ShadowOrderIntent (theoretical, never broker-confirmed)
-> theoretical fill model (reuses event-simulator fill)
-> ShadowTrade (theoretical)
-> P&L / drawdown evidence
-> immutable shadow-session report

Safety contract (load-bearing):

- This module has NO order method. It never calls a broker, never mutates a
  real position, and never reads a real funds balance. Approved capital is an
  explicit caller-supplied value; it is never inferred.
- The engine constructor accepts only read-only primitives (market events,
  session policy, tick policy, cost quoting provider, cost scenario for
  fingerprinting, strategy). There is deliberately no parameter into which a
  live broker client could be passed.
- Theoretical fills are always labelled ``theoretical`` and never
  broker-confirmed.
- UNKNOWN != ZERO: a missing observation stays ``None`` and fails closed to
  NO TRADE. It is never fabricated or defaulted to zero.
- CAS auxiliary bars never create a signal or a fill.
- A final entry signal never rolls to the next session.
- Stale quotes, feed gaps, disconnects, duplicates, out-of-order events,
  unknown instruments, unknown sessions, and missing capital all fail closed
  to NO TRADE.

Reuses (does not duplicate):

- :mod:`equity_engine.market_sessions` for session/CAS boundaries
- :mod:`equity_engine.event_simulator` for the fill model
  (``FillAssumptions`` plus the modelled fill helper)
- :mod:`equity_engine.historical_cost_scenario` for the cost-scenario
  fingerprint (``HistoricalCostScenario``)
- :mod:`equity_engine.sizing` for the approved-capital quantity check
- :mod:`equity_engine.experiment` for the approved-capital identity
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from .event_simulator import FillAssumptions
from .event_simulator import _modeled_fill_price as _simulator_fill_price
from .experiment import ApprovedCapital
from .historical_cost_scenario import HistoricalCostScenario
from .market_sessions import NSEEquitySessionPolicy
from .models import Exchange, OrderSpec, Product, Side
from .sizing import max_affordable_buy_quantity

SCHEMA_VERSION = "shadow-execution/v1"
SESSION_TIMEZONE = "Asia/Kolkata"


class ShadowRejectReason:
    NO_SIGNAL = "no_signal"
    ENTRY_SIGNAL_PENDING = "entry_signal_pending_next_bar"
    THEORETICAL_ENTRY = "theoretical_entry"
    THEORETICAL_EXIT_SIGNAL = "theoretical_exit_signal"
    THEORETICAL_EXIT_CUTOFF = "theoretical_exit_session_cutoff"
    HOLDING_NO_EXIT = "holding_no_exit_signal"
    STALE_QUOTE = "stale_quote_no_trade"
    FEED_GAP = "feed_gap_no_trade"
    FEED_DISCONNECTED = "feed_disconnected_no_trade"
    DUPLICATE_EVENT = "duplicate_event_no_trade"
    OUT_OF_ORDER_EVENT = "out_of_order_event_no_trade"
    CAS_AUXILIARY_EXCLUDED = "cas_auxiliary_excluded_no_trade"
    OUTSIDE_CONTINUOUS_SESSION = "outside_continuous_session_no_trade"
    UNKNOWN_INSTRUMENT = "unknown_instrument_no_trade"
    UNKNOWN_SESSION = "unknown_session_no_trade"
    NO_APPROVED_CAPITAL = "no_approved_capital_no_trade"
    INSUFFICIENT_CAPITAL = "insufficient_capital_no_trade"
    SIGNAL_EXPIRED_NEXT_SESSION = "signal_expired_next_session_no_trade"
    ENTRY_AFTER_CUTOFF = "entry_after_cutoff_no_trade"
    DAILY_LIMIT = "daily_trade_limit_no_trade"
    EXIT_AFTER_GAP_RECOVERY = "theoretical_exit_gap_recovery"


class ShadowStrategy(Protocol):
    """Read-only signal source. Implementations must use only the current bar."""

    @property
    def strategy_id(self) -> str: ...

    def entry_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        """Return True when the closed bar generates an entry signal for next bar."""
        ...

    def exit_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        """Return True when the closed bar generates an exit signal for next bar."""
        ...


@dataclass(frozen=True)
class ShadowMarketEvent:
    """One read-only live market observation (one bar). No order capability."""

    event_id: str
    seq: int
    instrument_key: str
    exchange: Exchange
    bar_timestamp: datetime
    bar_open: Decimal
    bar_high: Decimal
    bar_low: Decimal
    bar_close: Decimal
    bar_volume: int
    received_at: datetime
    is_cas_auxiliary: bool
    feed_connected: bool

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id is required")
        if self.seq < 0:
            raise ValueError("seq must be non-negative")
        if not self.instrument_key.strip():
            raise ValueError("instrument_key is required")
        if self.bar_timestamp.tzinfo is None:
            raise ValueError("bar_timestamp must be timezone-aware")
        if self.received_at.tzinfo is None:
            raise ValueError("received_at must be timezone-aware")
        if self.received_at < self.bar_timestamp:
            raise ValueError("received_at cannot precede bar_timestamp")
        for name in ("bar_open", "bar_high", "bar_low", "bar_close"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"{name} must be a positive finite Decimal")
        if self.bar_volume < 0:
            raise ValueError("bar_volume cannot be negative")
        if self.bar_high < max(self.bar_open, self.bar_close, self.bar_low):
            raise ValueError("bar_high violates OHLC invariant")
        if self.bar_low > min(self.bar_open, self.bar_close, self.bar_high):
            raise ValueError("bar_low violates OHLC invariant")

    def quote_age_seconds(self) -> float:
        return (self.received_at - self.bar_timestamp).total_seconds()

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "seq": self.seq,
            "instrument_key": self.instrument_key,
            "exchange": self.exchange.value,
            "bar_timestamp": self.bar_timestamp.isoformat(),
            "bar_open": format(self.bar_open, "f"),
            "bar_high": format(self.bar_high, "f"),
            "bar_low": format(self.bar_low, "f"),
            "bar_close": format(self.bar_close, "f"),
            "bar_volume": self.bar_volume,
            "received_at": self.received_at.isoformat(),
            "is_cas_auxiliary": self.is_cas_auxiliary,
            "feed_connected": self.feed_connected,
        }


def shadow_event_fingerprint(event: ShadowMarketEvent) -> str:
    payload = json.dumps(event.canonical_payload(), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ShadowInstrumentIdentity:
    """Known-instrument registry entry. Unknown keys fail closed."""

    instrument_key: str
    exchange: Exchange
    cas_eligible: bool
    tick_size_rupees: Decimal
    source: str

    def __post_init__(self) -> None:
        if not self.instrument_key.strip():
            raise ValueError("instrument_key is required")
        if self.tick_size_rupees <= 0:
            raise ValueError("tick_size_rupees must be positive")
        if not self.source.strip():
            raise ValueError("instrument source is required")


@dataclass(frozen=True)
class ShadowOrderIntent:
    """Theoretical intent. Never sent, never broker-confirmed."""

    intent_id: str
    instrument_key: str
    side: Side
    intended_quantity: int
    theoretical_price: Decimal
    signal_timestamp: str
    decision_timestamp: str
    label: str = "theoretical_never_broker_confirmed"

    def __post_init__(self) -> None:
        if self.label != "theoretical_never_broker_confirmed":
            raise ValueError("shadow intents must stay theoretical")
        if self.intended_quantity <= 0:
            raise ValueError("intended quantity must be positive")


@dataclass(frozen=True)
class ShadowTrade:
    """One completed theoretical round trip."""

    trade_id: str
    instrument_key: str
    quantity: int
    theoretical_entry: Decimal
    theoretical_exit: Decimal
    entry_cost_total: Decimal
    exit_cost_total: Decimal
    gross_reference_pnl: Decimal
    net_theoretical_pnl: Decimal
    entry_timestamp: str
    exit_timestamp: str
    exit_reason: str
    label: str = "theoretical_never_broker_confirmed"

    def __post_init__(self) -> None:
        if self.label != "theoretical_never_broker_confirmed":
            raise ValueError("shadow trades must stay theoretical")


@dataclass(frozen=True)
class ShadowDecision:
    """Per-event evidence record. Missing observations stay None (UNKNOWN != ZERO)."""

    timestamp: str
    instrument: str
    input_market_data_fingerprint: str
    quote_age_seconds: float
    session_identity: str | None
    cas_state: str
    strategy_identity: str
    signal_timestamp: str | None
    decision_timestamp: str
    intended_side: str | None
    intended_quantity: int
    approved_capital_identity: str | None
    theoretical_entry: str | None
    theoretical_exit: str | None
    spread_classification: str
    slippage_classification: str
    cost_scenario_fingerprint: str
    reason: str
    realized_theoretical_pnl: str | None
    drawdown_rupees: str
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.live_orders_called:
            raise ValueError("shadow decisions never call live orders")

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "instrument": self.instrument,
            "input_market_data_fingerprint": self.input_market_data_fingerprint,
            "quote_age_seconds": self.quote_age_seconds,
            "session_identity": self.session_identity,
            "cas_state": self.cas_state,
            "strategy_identity": self.strategy_identity,
            "signal_timestamp": self.signal_timestamp,
            "decision_timestamp": self.decision_timestamp,
            "intended_side": self.intended_side,
            "intended_quantity": self.intended_quantity,
            "approved_capital_identity": self.approved_capital_identity,
            "theoretical_entry": self.theoretical_entry,
            "theoretical_exit": self.theoretical_exit,
            "spread_classification": self.spread_classification,
            "slippage_classification": self.slippage_classification,
            "cost_scenario_fingerprint": self.cost_scenario_fingerprint,
            "reason": self.reason,
            "realized_theoretical_pnl": self.realized_theoretical_pnl,
            "drawdown_rupees": self.drawdown_rupees,
            "live_orders_called": False,
        }


@dataclass(frozen=True)
class ShadowSessionReport:
    """Immutable shadow-session evidence."""

    schema_version: str
    session_id: str
    instrument_key: str
    strategy_identity: str
    session_policy_identity: str
    approved_capital_identity: str | None
    cost_scenario_fingerprint: str
    starting_theoretical_cash: str
    ending_theoretical_cash: str
    max_drawdown_rupees: str
    decisions: tuple[ShadowDecision, ...]
    trades: tuple[ShadowTrade, ...]
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {self.schema_version!r}")
        if self.live_orders_called:
            raise ValueError("shadow sessions never call live orders")

    def deterministic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "instrument_key": self.instrument_key,
            "strategy_identity": self.strategy_identity,
            "session_policy_identity": self.session_policy_identity,
            "approved_capital_identity": self.approved_capital_identity,
            "cost_scenario_fingerprint": self.cost_scenario_fingerprint,
            "starting_theoretical_cash": self.starting_theoretical_cash,
            "ending_theoretical_cash": self.ending_theoretical_cash,
            "max_drawdown_rupees": self.max_drawdown_rupees,
            "decisions": [item.as_dict() for item in self.decisions],
            "trades": [
                {
                    "trade_id": item.trade_id,
                    "instrument_key": item.instrument_key,
                    "quantity": item.quantity,
                    "theoretical_entry": format(item.theoretical_entry, "f"),
                    "theoretical_exit": format(item.theoretical_exit, "f"),
                    "net_theoretical_pnl": format(item.net_theoretical_pnl, "f"),
                    "entry_timestamp": item.entry_timestamp,
                    "exit_timestamp": item.exit_timestamp,
                    "exit_reason": item.exit_reason,
                    "label": item.label,
                }
                for item in self.trades
            ],
            "live_orders_called": False,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.deterministic_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self.deterministic_payload()
        payload["fingerprint"] = self.fingerprint()
        return payload


@dataclass(frozen=True)
class ShadowEngineConfig:
    session_id: str
    exit_buffer_minutes: int
    max_quote_age_seconds: float
    bar_interval_seconds: int
    max_gap_multiplier: float
    max_trades_per_session: int

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id is required")
        if self.exit_buffer_minutes < 0 or self.exit_buffer_minutes >= 60:
            raise ValueError("exit_buffer_minutes must be in [0, 60)")
        if self.max_quote_age_seconds <= 0:
            raise ValueError("max_quote_age_seconds must be positive")
        if self.bar_interval_seconds <= 0:
            raise ValueError("bar_interval_seconds must be positive")
        if self.max_gap_multiplier < 1:
            raise ValueError("max_gap_multiplier must be >= 1")
        if self.max_trades_per_session <= 0:
            raise ValueError("max_trades_per_session must be positive")


def _capital_identity(capital: ApprovedCapital | None) -> str | None:
    if capital is None:
        return None
    payload = json.dumps(
        {"amount_rupees": str(capital.amount_rupees), "currency": capital.currency},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _session_identity(policy: NSEEquitySessionPolicy, trade_date: Any) -> str:
    return (
        f"NSEEquitySessionPolicy/cas_eligible={policy.cas_eligible}/"
        f"exit_buffer={policy.exit_buffer_minutes}/"
        f"continuous={policy.continuous_start(trade_date).isoformat()}-"
        f"{policy.continuous_end(trade_date).isoformat()}/"
        f"exit={policy.exit_time(trade_date).isoformat()}"
    )


class ShadowSessionEngine:
    """Single-instrument live-shadow engine. Read-only; has no order method."""

    def __init__(
        self,
        *,
        instrument: ShadowInstrumentIdentity,
        strategy: ShadowStrategy,
        approved_capital: ApprovedCapital | None,
        cost_scenario: HistoricalCostScenario,
        cost_provider: Any,
        fills: FillAssumptions,
        config: ShadowEngineConfig,
    ) -> None:
        if not instrument.instrument_key.strip():
            raise ValueError("instrument is required")
        if not strategy.strategy_id.strip():
            raise ValueError("strategy identity is required")
        if cost_scenario is None:
            raise ValueError("cost scenario is required")
        if cost_provider is None or not hasattr(cost_provider, "quote"):
            raise ValueError("read-only cost quoting provider is required")
        self._instrument = instrument
        self._strategy = strategy
        self._approved_capital = approved_capital
        self._cost_scenario = cost_scenario
        self._cost_provider = cost_provider
        self._fills = fills
        self._config = config
        self._policy = NSEEquitySessionPolicy(
            cas_eligible=instrument.cas_eligible,
            exit_buffer_minutes=config.exit_buffer_minutes,
        )
        start_cash = approved_capital.amount_rupees if approved_capital is not None else Decimal(0)
        self._cash = start_cash
        self._start_cash = start_cash
        self._peak_equity = start_cash
        self._max_drawdown = Decimal(0)
        self._decisions: list[ShadowDecision] = []
        self._trades: list[ShadowTrade] = []
        self._seen_ids: set[str] = set()
        self._last_seq: int | None = None
        self._last_bar_ts: datetime | None = None
        self._pending_entry_signal_ts: datetime | None = None
        self._pending_exit_signal_ts: datetime | None = None
        self._open_position: dict[str, Any] | None = None
        self._trades_today: dict[str, int] = {}
        self._trade_counter = 0

    @property
    def instrument_key(self) -> str:
        return self._instrument.instrument_key

    @property
    def decisions(self) -> tuple[ShadowDecision, ...]:
        return tuple(self._decisions)

    @property
    def trades(self) -> tuple[ShadowTrade, ...]:
        return tuple(self._trades)

    def _spread_slippage_labels(self) -> tuple[str, str]:
        spread = f"assumed:{format(self._fills.half_spread_bps_per_leg, 'f')}bps"
        slippage = f"assumed:{format(self._fills.slippage_bps_per_leg, 'f')}bps"
        return spread, slippage

    def _record_drawdown(self, equity: Decimal) -> Decimal:
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = self._peak_equity - equity
        if drawdown < 0:
            drawdown = Decimal(0)
        self._max_drawdown = max(self._max_drawdown, drawdown)
        return drawdown

    def _base_decision(
        self,
        *,
        event: ShadowMarketEvent,
        fingerprint: str,
        quote_age: float,
        session_identity: str | None,
        cas_state: str,
        signal_ts: datetime | None,
        reason: str,
        side: str | None = None,
        quantity: int = 0,
        entry: Decimal | None = None,
        exit_: Decimal | None = None,
        realized: Decimal | None = None,
    ) -> ShadowDecision:
        spread, slippage = self._spread_slippage_labels()
        equity = self._cash
        if self._open_position is not None:
            equity = self._cash + Decimal(self._open_position["quantity"]) * event.bar_close
        drawdown = self._record_drawdown(equity)
        return ShadowDecision(
            timestamp=event.bar_timestamp.isoformat(),
            instrument=event.instrument_key,
            input_market_data_fingerprint=fingerprint,
            quote_age_seconds=quote_age,
            session_identity=session_identity,
            cas_state=cas_state,
            strategy_identity=self._strategy.strategy_id,
            signal_timestamp=signal_ts.isoformat() if signal_ts is not None else None,
            decision_timestamp=event.received_at.isoformat(),
            intended_side=side,
            intended_quantity=quantity,
            approved_capital_identity=_capital_identity(self._approved_capital),
            theoretical_entry=format(entry, "f") if entry is not None else None,
            theoretical_exit=format(exit_, "f") if exit_ is not None else None,
            spread_classification=spread,
            slippage_classification=slippage,
            cost_scenario_fingerprint=self._cost_scenario.fingerprint(),
            reason=reason,
            realized_theoretical_pnl=format(realized, "f") if realized is not None else None,
            drawdown_rupees=format(drawdown, "f"),
            live_orders_called=False,
        )

    def process(self, event: ShadowMarketEvent) -> ShadowDecision:
        fingerprint = shadow_event_fingerprint(event)
        quote_age = event.quote_age_seconds()

        # Unknown instrument (also covers unknown session/CAS identity).
        if event.instrument_key != self._instrument.instrument_key:
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=None,
                cas_state="unknown_instrument",
                signal_ts=None,
                reason=ShadowRejectReason.UNKNOWN_INSTRUMENT,
            )
            self._decisions.append(decision)
            return decision

        trade_date = event.bar_timestamp.date()
        try:
            session_id_str = _session_identity(self._policy, trade_date)
            continuous_end = self._policy.continuous_end(trade_date)
            continuous_start = self._policy.continuous_start(trade_date)
            exit_time = self._policy.exit_time(trade_date)
        except ValueError:
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=None,
                cas_state="unknown_session",
                signal_ts=None,
                reason=ShadowRejectReason.UNKNOWN_SESSION,
            )
            self._decisions.append(decision)
            return decision

        if event.is_cas_auxiliary:
            # CAS auxiliary data never creates a signal or a fill.
            self._pending_entry_signal_ts = None
            self._pending_exit_signal_ts = None
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state="cas_auxiliary",
                signal_ts=None,
                reason=ShadowRejectReason.CAS_AUXILIARY_EXCLUDED,
            )
            self._decisions.append(decision)
            return decision

        cas_state = "continuous"

        # Duplicate / out-of-order guards come before any state mutation.
        if event.event_id in self._seen_ids:
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=None,
                reason=ShadowRejectReason.DUPLICATE_EVENT,
            )
            self._decisions.append(decision)
            return decision
        if self._last_bar_ts is not None and event.bar_timestamp <= self._last_bar_ts:
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=None,
                reason=ShadowRejectReason.OUT_OF_ORDER_EVENT,
            )
            self._decisions.append(decision)
            return decision
        if self._last_seq is not None and event.seq <= self._last_seq:
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=None,
                reason=ShadowRejectReason.OUT_OF_ORDER_EVENT,
            )
            self._decisions.append(decision)
            return decision

        # Feed gap: missing sequence or a timestamp jump beyond tolerance.
        if self._last_seq is not None:
            expected_seq = self._last_seq + 1
            max_gap = self._config.bar_interval_seconds * self._config.max_gap_multiplier
            time_gap = (event.bar_timestamp - self._last_bar_ts).total_seconds()  # type: ignore[operator]
            if event.seq != expected_seq or time_gap > max_gap:
                self._pending_entry_signal_ts = None
                self._pending_exit_signal_ts = None
                self._seen_ids.add(event.event_id)
                self._last_seq = event.seq
                self._last_bar_ts = event.bar_timestamp
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=None,
                    reason=ShadowRejectReason.FEED_GAP,
                )
                self._decisions.append(decision)
                return decision

        # Disconnected feed fails closed and clears pending signals.
        if not event.feed_connected:
            self._pending_entry_signal_ts = None
            self._pending_exit_signal_ts = None
            self._seen_ids.add(event.event_id)
            self._last_seq = event.seq
            self._last_bar_ts = event.bar_timestamp
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=None,
                reason=ShadowRejectReason.FEED_DISCONNECTED,
            )
            self._decisions.append(decision)
            return decision

        # Stale quotes fail closed and clear pending signals.
        if quote_age > self._config.max_quote_age_seconds:
            self._pending_entry_signal_ts = None
            self._pending_exit_signal_ts = None
            self._seen_ids.add(event.event_id)
            self._last_seq = event.seq
            self._last_bar_ts = event.bar_timestamp
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=None,
                reason=ShadowRejectReason.STALE_QUOTE,
            )
            self._decisions.append(decision)
            return decision

        # Continuous-session filter (single boundary before signals).
        bar_time = event.bar_timestamp.time()
        if not (continuous_start <= bar_time < continuous_end):
            self._pending_entry_signal_ts = None
            self._pending_exit_signal_ts = None
            self._seen_ids.add(event.event_id)
            self._last_seq = event.seq
            self._last_bar_ts = event.bar_timestamp
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state="outside_continuous",
                signal_ts=None,
                reason=ShadowRejectReason.OUTSIDE_CONTINUOUS_SESSION,
            )
            self._decisions.append(decision)
            return decision

        # No approved capital fails closed.
        if self._approved_capital is None:
            self._seen_ids.add(event.event_id)
            self._last_seq = event.seq
            self._last_bar_ts = event.bar_timestamp
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=None,
                reason=ShadowRejectReason.NO_APPROVED_CAPITAL,
            )
            self._decisions.append(decision)
            return decision

        # Final entry signals never roll to the next session.
        if (
            self._pending_entry_signal_ts is not None
            and self._pending_entry_signal_ts.date() != trade_date
        ):
            self._pending_entry_signal_ts = None
            self._seen_ids.add(event.event_id)
            self._last_seq = event.seq
            self._last_bar_ts = event.bar_timestamp
            # Evaluate the current bar for a fresh signal below? No: this bar is
            # the first bar of a new session and its own signal (if any) belongs
            # to the next bar. Record the expiry and also check fresh signal.
            fresh = self._strategy.entry_signal_at_close(event)
            if fresh:
                self._pending_entry_signal_ts = event.bar_timestamp
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=event.bar_timestamp,
                    reason=ShadowRejectReason.ENTRY_SIGNAL_PENDING,
                )
            else:
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=None,
                    reason=ShadowRejectReason.SIGNAL_EXPIRED_NEXT_SESSION,
                )
            self._decisions.append(decision)
            return decision
        if (
            self._pending_exit_signal_ts is not None
            and self._pending_exit_signal_ts.date() != trade_date
            and self._open_position is not None
            and self._open_position["entry_timestamp"].date() != trade_date
        ):
            # Carried theoretical position: allow a gap-recovery exit below but
            # never treat the stale exit signal as live.
            self._pending_exit_signal_ts = None

        self._seen_ids.add(event.event_id)
        self._last_seq = event.seq
        self._last_bar_ts = event.bar_timestamp

        # --- Exit path first (a position blocks new entries) ---
        if self._open_position is not None:
            entry_date = self._open_position["entry_timestamp"].date()
            should_cutoff = bar_time >= exit_time
            pending_exit_due = (
                self._pending_exit_signal_ts is not None
                and self._pending_exit_signal_ts.date() == trade_date
            )
            carried_overnight = entry_date != trade_date
            if should_cutoff or pending_exit_due or carried_overnight:
                if bar_time >= continuous_end:
                    decision = self._base_decision(
                        event=event,
                        fingerprint=fingerprint,
                        quote_age=quote_age,
                        session_identity=session_id_str,
                        cas_state="outside_continuous",
                        signal_ts=self._pending_exit_signal_ts,
                        reason=ShadowRejectReason.OUTSIDE_CONTINUOUS_SESSION,
                    )
                    self._decisions.append(decision)
                    return decision
                reference_exit = event.bar_open
                exit_tick = self._open_position["tick_size"]
                fill_exit = _simulator_fill_price(
                    reference_exit,
                    side=Side.SELL,
                    assumptions=self._fills,
                    tick_size=exit_tick,
                )
                quantity = int(self._open_position["quantity"])
                exit_order = OrderSpec(
                    instrument_token=self._instrument.instrument_key,
                    exchange=self._instrument.exchange,
                    side=Side.SELL,
                    product=Product.INTRADAY,
                    quantity=quantity,
                    price=fill_exit,
                )
                exit_quote = self._cost_provider.quote(exit_order)
                self._cash += exit_order.notional - exit_quote.total
                entry_price = self._open_position["fill_entry"]
                reference_entry = self._open_position["reference_entry"]
                entry_cost = self._open_position["entry_cost"]
                gross = (reference_exit - reference_entry) * quantity
                net_pnl = (fill_exit - entry_price) * quantity - entry_cost - exit_quote.total
                if carried_overnight and not should_cutoff and not pending_exit_due:
                    exit_reason = ShadowRejectReason.EXIT_AFTER_GAP_RECOVERY
                elif should_cutoff:
                    exit_reason = ShadowRejectReason.THEORETICAL_EXIT_CUTOFF
                else:
                    exit_reason = ShadowRejectReason.THEORETICAL_EXIT_SIGNAL
                self._trade_counter += 1
                trade = ShadowTrade(
                    trade_id=f"shadow-trade-{self._trade_counter}",
                    instrument_key=self._instrument.instrument_key,
                    quantity=quantity,
                    theoretical_entry=entry_price,
                    theoretical_exit=fill_exit,
                    entry_cost_total=entry_cost,
                    exit_cost_total=exit_quote.total,
                    gross_reference_pnl=gross,
                    net_theoretical_pnl=net_pnl,
                    entry_timestamp=self._open_position["entry_timestamp"].isoformat(),
                    exit_timestamp=event.bar_timestamp.isoformat(),
                    exit_reason=exit_reason,
                )
                self._trades.append(trade)
                day_key = trade_date.isoformat()
                self._trades_today[day_key] = self._trades_today.get(day_key, 0)
                self._open_position = None
                self._pending_exit_signal_ts = None
                self._pending_entry_signal_ts = None
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=self._pending_exit_signal_ts,
                    reason=exit_reason,
                    side=Side.SELL.value,
                    quantity=quantity,
                    entry=entry_price,
                    exit_=fill_exit,
                    realized=net_pnl,
                )
                self._decisions.append(decision)
                return decision
            # No exit due: evaluate current bar for a future exit signal.
            exit_now_signal = self._strategy.exit_signal_at_close(event)
            if exit_now_signal:
                self._pending_exit_signal_ts = event.bar_timestamp
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=event.bar_timestamp if exit_now_signal else None,
                reason=ShadowRejectReason.HOLDING_NO_EXIT,
                side=None,
                quantity=int(self._open_position["quantity"]),
                entry=self._open_position["fill_entry"],
                exit_=None,
                realized=None,
            )
            self._decisions.append(decision)
            return decision

        # --- Entry path (flat) ---
        if self._pending_entry_signal_ts is not None:
            pending_ts = self._pending_entry_signal_ts
            self._pending_entry_signal_ts = None
            if bar_time >= exit_time:
                # Pending entry expires at the cutoff; record and evaluate fresh
                # signal handling below through a holding decision.
                fresh = self._strategy.entry_signal_at_close(event)
                if fresh and bar_time < exit_time:
                    self._pending_entry_signal_ts = event.bar_timestamp
                    decision = self._base_decision(
                        event=event,
                        fingerprint=fingerprint,
                        quote_age=quote_age,
                        session_identity=session_id_str,
                        cas_state=cas_state,
                        signal_ts=event.bar_timestamp,
                        reason=ShadowRejectReason.ENTRY_SIGNAL_PENDING,
                    )
                else:
                    decision = self._base_decision(
                        event=event,
                        fingerprint=fingerprint,
                        quote_age=quote_age,
                        session_identity=session_id_str,
                        cas_state=cas_state,
                        signal_ts=pending_ts,
                        reason=ShadowRejectReason.ENTRY_AFTER_CUTOFF,
                    )
                self._decisions.append(decision)
                return decision
            day_key = trade_date.isoformat()
            if self._trades_today.get(day_key, 0) >= self._config.max_trades_per_session:
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=pending_ts,
                    reason=ShadowRejectReason.DAILY_LIMIT,
                )
                self._decisions.append(decision)
                return decision
            reference_entry = event.bar_open
            tick = self._instrument.tick_size_rupees
            if tick <= 0:
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=pending_ts,
                    reason=ShadowRejectReason.UNKNOWN_SESSION,
                )
                self._decisions.append(decision)
                return decision
            fill_entry = _simulator_fill_price(
                reference_entry,
                side=Side.BUY,
                assumptions=self._fills,
                tick_size=tick,
            )
            size = max_affordable_buy_quantity(
                instrument_token=self._instrument.instrument_key,
                exchange=self._instrument.exchange,
                product=Product.INTRADAY,
                price=fill_entry,
                cash_limit=self._cash,
                cost_provider=self._cost_provider,
            )
            if size.quantity <= 0:
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=pending_ts,
                    reason=ShadowRejectReason.INSUFFICIENT_CAPITAL,
                )
                self._decisions.append(decision)
                return decision
            entry_order = OrderSpec(
                instrument_token=self._instrument.instrument_key,
                exchange=self._instrument.exchange,
                side=Side.BUY,
                product=Product.INTRADAY,
                quantity=size.quantity,
                price=fill_entry,
            )
            entry_quote = self._cost_provider.quote(entry_order)
            required = entry_order.notional + entry_quote.total
            if required > self._cash:
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=pending_ts,
                    reason=ShadowRejectReason.INSUFFICIENT_CAPITAL,
                )
                self._decisions.append(decision)
                return decision
            self._cash -= required
            self._open_position = {
                "quantity": size.quantity,
                "reference_entry": reference_entry,
                "fill_entry": fill_entry,
                "tick_size": tick,
                "entry_cost": entry_quote.total,
                "entry_timestamp": event.bar_timestamp,
            }
            self._trades_today[day_key] = self._trades_today.get(day_key, 0) + 1
            intent = ShadowOrderIntent(
                intent_id=f"shadow-intent-{len(self._decisions) + 1}",
                instrument_key=self._instrument.instrument_key,
                side=Side.BUY,
                intended_quantity=size.quantity,
                theoretical_price=fill_entry,
                signal_timestamp=pending_ts.isoformat(),
                decision_timestamp=event.received_at.isoformat(),
            )
            _ = intent
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=pending_ts,
                reason=ShadowRejectReason.THEORETICAL_ENTRY,
                side=Side.BUY.value,
                quantity=size.quantity,
                entry=fill_entry,
                exit_=None,
                realized=None,
            )
            self._decisions.append(decision)
            return decision

        # No pending entry: evaluate the closed bar for the next bar.
        if bar_time >= exit_time:
            fresh = self._strategy.entry_signal_at_close(event)
            if fresh:
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=event.bar_timestamp,
                    reason=ShadowRejectReason.ENTRY_AFTER_CUTOFF,
                )
            else:
                decision = self._base_decision(
                    event=event,
                    fingerprint=fingerprint,
                    quote_age=quote_age,
                    session_identity=session_id_str,
                    cas_state=cas_state,
                    signal_ts=None,
                    reason=ShadowRejectReason.NO_SIGNAL,
                )
            self._decisions.append(decision)
            return decision
        fresh = self._strategy.entry_signal_at_close(event)
        if fresh:
            self._pending_entry_signal_ts = event.bar_timestamp
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=event.bar_timestamp,
                reason=ShadowRejectReason.ENTRY_SIGNAL_PENDING,
            )
        else:
            decision = self._base_decision(
                event=event,
                fingerprint=fingerprint,
                quote_age=quote_age,
                session_identity=session_id_str,
                cas_state=cas_state,
                signal_ts=None,
                reason=ShadowRejectReason.NO_SIGNAL,
            )
        self._decisions.append(decision)
        return decision

    def _mark_trade_day(self, trade_date: Any) -> None:
        key = trade_date.isoformat()
        self._trades_today[key] = self._trades_today.get(key, 0) + 1

    def report(self) -> ShadowSessionReport:
        policy_identity = (
            f"NSEEquitySessionPolicy/cas_eligible={self._instrument.cas_eligible}/"
            f"exit_buffer={self._config.exit_buffer_minutes}"
        )
        return ShadowSessionReport(
            schema_version=SCHEMA_VERSION,
            session_id=self._config.session_id,
            instrument_key=self._instrument.instrument_key,
            strategy_identity=self._strategy.strategy_id,
            session_policy_identity=policy_identity,
            approved_capital_identity=_capital_identity(self._approved_capital),
            cost_scenario_fingerprint=self._cost_scenario.fingerprint(),
            starting_theoretical_cash=format(self._start_cash, "f"),
            ending_theoretical_cash=format(self._cash, "f"),
            max_drawdown_rupees=format(self._max_drawdown, "f"),
            decisions=tuple(self._decisions),
            trades=tuple(self._trades),
            live_orders_called=False,
        )


def replay_shadow_session(
    events: list[ShadowMarketEvent],
    *,
    instrument: ShadowInstrumentIdentity,
    strategy: ShadowStrategy,
    approved_capital: ApprovedCapital | None,
    cost_scenario: HistoricalCostScenario,
    cost_provider: Any,
    fills: FillAssumptions,
    config: ShadowEngineConfig,
) -> ShadowSessionReport:
    """Deterministically replay recorded shadow market events."""
    engine = ShadowSessionEngine(
        instrument=instrument,
        strategy=strategy,
        approved_capital=approved_capital,
        cost_scenario=cost_scenario,
        cost_provider=cost_provider,
        fills=fills,
        config=config,
    )
    for event in events:
        engine.process(event)
    return engine.report()


class AlwaysSignalShadowStrategy:
    """Synthetic strategy that signals entry once and exits only at cutoff."""

    def __init__(self, strategy_id: str = "synthetic-always") -> None:
        self._strategy_id = strategy_id
        self._fired = False

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def entry_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        if not self._fired:
            self._fired = True
            return True
        return False

    def exit_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        return False


class NeverSignalShadowStrategy:
    """Synthetic strategy that never signals."""

    def __init__(self, strategy_id: str = "synthetic-never") -> None:
        self._strategy_id = strategy_id

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def entry_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        return False

    def exit_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        return False


class ExitOnSecondBarStrategy:
    """Synthetic strategy: one entry, then an exit signal on the next closed bar."""

    def __init__(self, strategy_id: str = "synthetic-exit-second-bar") -> None:
        self._strategy_id = strategy_id
        self._bars_seen = 0

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def entry_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        self._bars_seen += 1
        return self._bars_seen == 1

    def exit_signal_at_close(self, event: ShadowMarketEvent) -> bool:
        self._bars_seen += 1
        return self._bars_seen >= 3


__all__ = [
    "SCHEMA_VERSION",
    "AlwaysSignalShadowStrategy",
    "ApprovedCapital",
    "ExitOnSecondBarStrategy",
    "FillAssumptions",
    "HistoricalCostScenario",
    "NeverSignalShadowStrategy",
    "ShadowDecision",
    "ShadowEngineConfig",
    "ShadowInstrumentIdentity",
    "ShadowMarketEvent",
    "ShadowOrderIntent",
    "ShadowRejectReason",
    "ShadowSessionEngine",
    "ShadowSessionReport",
    "ShadowStrategy",
    "ShadowTrade",
    "replay_shadow_session",
    "shadow_event_fingerprint",
]
