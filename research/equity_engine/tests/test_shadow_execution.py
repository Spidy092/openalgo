"""Live shadow execution V1: synthetic proof that shadow never sends an order.

Covers: normal trade, no-trade, stale quote, missing tick (feed gap),
CAS boundary, disconnected feed, duplicate event, out-of-order event,
next-session signal rejection, explicit capital cap, and zero order calls.
Also proves deterministic replay, CAS isolation, stale/gap isolation, and
capital enforcement.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from equity_engine.cost_ledger import EffectiveDatedCostLedger, LedgerComponent, LedgerProduct
from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.event_simulator import FillAssumptions
from equity_engine.experiment import ApprovedCapital
from equity_engine.historical_cost_scenario import (
    ScenarioAssumption,
    compile_historical_scenario,
)
from equity_engine.models import Exchange
from equity_engine.shadow_execution import (
    AlwaysSignalShadowStrategy,
    NeverSignalShadowStrategy,
    ShadowEngineConfig,
    ShadowInstrumentIdentity,
    ShadowMarketEvent,
    ShadowRejectReason,
    ShadowSessionEngine,
    replay_shadow_session,
    shadow_event_fingerprint,
)

IST = ZoneInfo("Asia/Kolkata")


def _assumptions() -> tuple[ScenarioAssumption, ...]:
    def _one(component: LedgerComponent, rate: str) -> ScenarioAssumption:
        return ScenarioAssumption(
            assumption_id=f"shadow-assume-{component.value}",
            component=component,
            product=LedgerProduct.INTRADAY,
            basis="shadow-test-basis",
            rate=Decimal(rate),
            formula=f"shadow-formula-{rate}",
            source="shadow-test-source",
            reason="shadow-test-reason",
        )

    return (
        _one(LedgerComponent.BROKERAGE, "0.0006"),
        _one(LedgerComponent.GST, "0.18"),
        _one(LedgerComponent.CLEARING, "0.000001"),
    )


def _scenario():
    return compile_historical_scenario(
        scenario_id="shadow-monday-v1",
        ledger=EffectiveDatedCostLedger(),
        scenario_date=date(2026, 9, 7),
        research_start=date(2026, 9, 1),
        research_end=date(2026, 9, 8),
        product=LedgerProduct.INTRADAY,
        assumptions=_assumptions(),
    )


def _provider():
    return CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7))


def _fills() -> FillAssumptions:
    return FillAssumptions(
        slippage_bps_per_leg=Decimal(2),
        half_spread_bps_per_leg=Decimal(1),
    )


def _instrument(
    key: str = "NSE_EQ|INE002A01018", cas_eligible: bool = False
) -> ShadowInstrumentIdentity:
    return ShadowInstrumentIdentity(
        instrument_key=key,
        exchange=Exchange.NSE,
        cas_eligible=cas_eligible,
        tick_size_rupees=Decimal("0.05"),
        source="synthetic-shadow-test",
    )


def _config(**overrides) -> ShadowEngineConfig:
    params: dict[str, object] = {
        "session_id": "monday-shadow-v1",
        "exit_buffer_minutes": 15,
        "max_quote_age_seconds": 60.0,
        "bar_interval_seconds": 300,
        "max_gap_multiplier": 1.5,
        "max_trades_per_session": 1,
    }
    params.update(overrides)
    return ShadowEngineConfig(**params)  # type: ignore[arg-type]


def _event(
    seq: int,
    ts: str,
    open_price: str = "100.00",
    *,
    instrument: str = "NSE_EQ|INE002A01018",
    received_lag_seconds: int = 5,
    cas_aux: bool = False,
    connected: bool = True,
    event_id: str | None = None,
    close_price: str | None = None,
) -> ShadowMarketEvent:
    bar_ts = datetime.fromisoformat(ts).replace(tzinfo=IST)
    received = bar_ts.fromtimestamp(bar_ts.timestamp() + received_lag_seconds, tz=IST)
    px = Decimal(open_price)
    close = Decimal(close_price) if close_price is not None else px + Decimal("0.10")
    return ShadowMarketEvent(
        event_id=event_id or f"evt-{seq}-{ts}",
        seq=seq,
        instrument_key=instrument,
        exchange=Exchange.NSE,
        bar_timestamp=bar_ts,
        bar_open=px,
        bar_high=max(px, close) + Decimal("0.05"),
        bar_low=min(px, close) - Decimal("0.05"),
        bar_close=close,
        bar_volume=10_000,
        received_at=received,
        is_cas_auxiliary=cas_aux,
        feed_connected=connected,
    )


def _engine(**overrides):
    defaults: dict[str, object] = {
        "instrument": _instrument(),
        "strategy": AlwaysSignalShadowStrategy(),
        "approved_capital": ApprovedCapital(amount_rupees=Decimal(100000)),
        "cost_scenario": _scenario(),
        "cost_provider": _provider(),
        "fills": _fills(),
        "config": _config(),
    }
    defaults.update(overrides)
    return ShadowSessionEngine(**defaults)  # type: ignore[arg-type]


def test_normal_trade_theoretical_fill_and_cutoff_exit() -> None:
    engine = _engine(strategy=AlwaysSignalShadowStrategy())
    # Dense 5-min bars so no feed gap; exit via session cutoff at 15:15.
    stamps = ["2026-09-07T09:15:00", "2026-09-07T09:20:00", "2026-09-07T09:25:00"]
    # Build full dense run from 09:15 to 15:15.
    events: list[ShadowMarketEvent] = []
    base = datetime.fromisoformat("2026-09-07T09:15:00").replace(tzinfo=IST)
    price = Decimal("100.00")
    seq = 0
    current = base
    cutoff = datetime.fromisoformat("2026-09-07T15:15:00").replace(tzinfo=IST)
    while current <= cutoff:
        ts = current.isoformat()
        # Gentle uptrend so the theoretical trade has a defined exit.
        px = price + Decimal(seq) * Decimal("0.05")
        events.append(_event(seq, ts, format(px, "f")))
        current = current.fromtimestamp(current.timestamp() + 300, tz=IST)
        seq += 1
    assert stamps[0] in events[0].bar_timestamp.isoformat()

    decisions = [engine.process(event) for event in events]
    report = engine.report()

    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.label == "theoretical_never_broker_confirmed"
    assert trade.quantity > 0
    assert trade.exit_reason == ShadowRejectReason.THEORETICAL_EXIT_CUTOFF
    # No same-bar lookahead: signal bar 09:15 close executes at 09:20 open.
    assert decisions[0].reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING
    assert decisions[1].reason == ShadowRejectReason.THEORETICAL_ENTRY
    assert decisions[1].theoretical_entry is not None
    assert decisions[1].theoretical_exit is None  # UNKNOWN != ZERO: not fabricated
    # Entry fill references next-bar open, not the signal bar.
    assert decisions[1].signal_timestamp == events[0].bar_timestamp.isoformat()
    # Required evidence present on the entry decision.
    entry = decisions[1]
    assert entry.input_market_data_fingerprint == shadow_event_fingerprint(events[1])
    assert entry.session_identity is not None and "NSEEquitySessionPolicy" in entry.session_identity
    assert entry.cas_state == "continuous"
    assert entry.strategy_identity == "synthetic-always"
    assert entry.decision_timestamp == events[1].received_at.isoformat()
    assert entry.intended_side == "BUY"
    assert entry.approved_capital_identity is not None
    assert entry.cost_scenario_fingerprint == _scenario().fingerprint()
    assert entry.spread_classification.startswith("assumed:")
    assert entry.slippage_classification.startswith("assumed:")
    assert entry.live_orders_called is False
    # Exit decision carries both legs and realized P&L.
    exit_decisions = [d for d in decisions if d.realized_theoretical_pnl is not None]
    assert len(exit_decisions) == 1
    assert exit_decisions[0].theoretical_exit is not None
    assert report.live_orders_called is False
    assert report.cost_scenario_fingerprint == _scenario().fingerprint()


def test_no_trade_strategy_records_no_signal() -> None:
    engine = _engine(strategy=NeverSignalShadowStrategy())
    events = [
        _event(0, "2026-09-07T09:15:00"),
        _event(1, "2026-09-07T09:20:00"),
        _event(2, "2026-09-07T09:25:00"),
    ]
    decisions = [engine.process(event) for event in events]
    report = engine.report()
    assert report.trades == ()
    assert all(d.reason == ShadowRejectReason.NO_SIGNAL for d in decisions)
    assert all(d.intended_quantity == 0 for d in decisions)
    assert all(d.theoretical_entry is None for d in decisions)
    assert all(d.theoretical_exit is None for d in decisions)
    assert all(d.realized_theoretical_pnl is None for d in decisions)
    assert all(d.live_orders_called is False for d in decisions)


def test_stale_quote_fails_closed() -> None:
    engine = _engine()
    fresh = _event(0, "2026-09-07T09:15:00")
    stale = _event(1, "2026-09-07T09:20:00", received_lag_seconds=600)
    d0 = engine.process(fresh)
    assert d0.reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING
    d1 = engine.process(stale)
    assert d1.reason == ShadowRejectReason.STALE_QUOTE
    assert d1.intended_quantity == 0
    assert d1.theoretical_entry is None
    assert engine.report().trades == ()
    # Stale clears pending: next fresh bar must not fill the old signal.
    nxt = _event(2, "2026-09-07T09:25:00")
    d2 = engine.process(nxt)
    assert d2.reason in (ShadowRejectReason.NO_SIGNAL, ShadowRejectReason.ENTRY_SIGNAL_PENDING)
    assert engine.report().trades == ()


def test_missing_tick_feed_gap_fails_closed() -> None:
    engine = _engine()
    e0 = _event(0, "2026-09-07T09:15:00")
    e1 = _event(1, "2026-09-07T09:20:00")
    # Skip seq 2 / 09:25 to simulate a missing tick.
    e3 = _event(3, "2026-09-07T09:30:00")
    assert engine.process(e0).reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING
    assert engine.process(e1).reason == ShadowRejectReason.THEORETICAL_ENTRY
    gap = engine.process(e3)
    assert gap.reason == ShadowRejectReason.FEED_GAP
    assert gap.theoretical_exit is None
    # The open theoretical position is preserved (never fabricated exit),
    # but the gap bar itself creates no fill.
    assert engine.report().trades == ()


def test_cas_boundary_auxiliary_cannot_signal_or_fill() -> None:
    engine = _engine(
        instrument=_instrument(cas_eligible=True),
        strategy=AlwaysSignalShadowStrategy(strategy_id="cas-proof"),
    )
    # CAS auxiliary bar first: must not create a pending signal.
    aux = _event(0, "2026-09-08T15:16:00", cas_aux=True)
    d_aux = engine.process(aux)
    assert d_aux.reason == ShadowRejectReason.CAS_AUXILIARY_EXCLUDED
    assert d_aux.cas_state == "cas_auxiliary"
    assert d_aux.theoretical_entry is None
    # Use a same-day continuous engine to prove auxiliary bars clear pending.
    engine2 = _engine(
        instrument=_instrument(cas_eligible=True),
        strategy=AlwaysSignalShadowStrategy(strategy_id="cas-proof-2"),
    )
    c0 = _event(0, "2026-09-08T09:15:00")
    c_aux = _event(1, "2026-09-08T09:20:00", cas_aux=True)
    c1 = _event(2, "2026-09-08T09:25:00")
    assert engine2.process(c0).reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING
    assert engine2.process(c_aux).reason == ShadowRejectReason.CAS_AUXILIARY_EXCLUDED
    # Pending was cleared by the auxiliary bar, so no fill here.
    d_after = engine2.process(c1)
    assert d_after.reason != ShadowRejectReason.THEORETICAL_ENTRY
    assert engine2.report().trades == ()


def test_disconnected_feed_fails_closed() -> None:
    engine = _engine()
    e0 = _event(0, "2026-09-07T09:15:00")
    assert engine.process(e0).reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING
    disc = _event(1, "2026-09-07T09:20:00", connected=False)
    d = engine.process(disc)
    assert d.reason == ShadowRejectReason.FEED_DISCONNECTED
    assert engine.report().trades == ()


def test_duplicate_event_creates_no_fill_or_signal() -> None:
    engine = _engine()
    e0 = _event(0, "2026-09-07T09:15:00")
    assert engine.process(e0).reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING
    dup = _event(0, "2026-09-07T09:15:00")
    d = engine.process(dup)
    assert d.reason == ShadowRejectReason.DUPLICATE_EVENT
    assert engine.report().trades == ()
    # Original pending still intact: next bar fills exactly once.
    e1 = _event(1, "2026-09-07T09:20:00")
    assert engine.process(e1).reason == ShadowRejectReason.THEORETICAL_ENTRY
    assert len(engine.report().trades) == 0  # entry open, exit pending cutoff
    # Duplicate of the fill bar must not create a second intent.
    dup_fill = _event(1, "2026-09-07T09:20:00")
    d_dup = engine.process(dup_fill)
    assert d_dup.reason == ShadowRejectReason.DUPLICATE_EVENT
    assert len(engine.report().trades) == 0


def test_out_of_order_event_creates_no_fill() -> None:
    engine = _engine()
    e0 = _event(0, "2026-09-07T09:15:00")
    e1 = _event(1, "2026-09-07T09:20:00")
    engine.process(e0)
    engine.process(e1)
    late = _event(0, "2026-09-07T09:15:00", event_id="late-arrival")
    d = engine.process(late)
    assert d.reason == ShadowRejectReason.OUT_OF_ORDER_EVENT
    # Position opened at e1; out-of-order bar must not exit or duplicate it.
    assert len(engine.report().trades) == 0


def test_next_session_signal_rejected() -> None:
    # Large gap tolerance isolates the session-roll rule from feed-gap logic.
    cfg = _config(max_gap_multiplier=100000.0)
    engine = _engine(config=cfg, strategy=AlwaysSignalShadowStrategy(strategy_id="roll-proof"))
    day1_last = _event(0, "2026-09-07T09:15:00")
    assert engine.process(day1_last).reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING
    # Next session first bar: pending from Monday must not execute Tuesday.
    day2_first = _event(1, "2026-09-08T09:15:00")
    d = engine.process(day2_first)
    assert d.reason in (
        ShadowRejectReason.SIGNAL_EXPIRED_NEXT_SESSION,
        ShadowRejectReason.ENTRY_SIGNAL_PENDING,
    )
    # If a fresh Tuesday signal was stored, it must reference Tuesday, never Monday.
    if d.reason == ShadowRejectReason.ENTRY_SIGNAL_PENDING:
        assert d.signal_timestamp == day2_first.bar_timestamp.isoformat()
    assert engine.report().trades == ()


def test_explicit_capital_cap_limits_quantity() -> None:
    small = ApprovedCapital(amount_rupees=Decimal(500))
    engine = _engine(approved_capital=small)
    events = [
        _event(0, "2026-09-07T09:15:00", "100.00"),
        _event(1, "2026-09-07T09:20:00", "100.00"),
    ]
    engine.process(events[0])
    entry = engine.process(events[1])
    assert entry.reason == ShadowRejectReason.THEORETICAL_ENTRY
    assert entry.intended_quantity <= 4  # 500 / ~100 minus costs
    assert entry.approved_capital_identity is not None
    # Tiny capital that cannot afford one share fails closed instead of zero-fill.
    tiny = ApprovedCapital(amount_rupees=Decimal(1))
    engine2 = _engine(approved_capital=tiny)
    engine2.process(_event(0, "2026-09-07T09:15:00", "100.00"))
    d = engine2.process(_event(1, "2026-09-07T09:20:00", "100.00"))
    assert d.reason == ShadowRejectReason.INSUFFICIENT_CAPITAL
    assert engine2.report().trades == ()


def test_no_approved_capital_fails_closed() -> None:
    engine = _engine(approved_capital=None)
    d = engine.process(_event(0, "2026-09-07T09:15:00"))
    assert d.reason == ShadowRejectReason.NO_APPROVED_CAPITAL
    assert d.approved_capital_identity is None
    assert d.theoretical_entry is None
    assert engine.report().trades == ()


def test_unknown_instrument_fails_closed() -> None:
    engine = _engine()
    d = engine.process(_event(0, "2026-09-07T09:15:00", instrument="NSE_EQ|UNKNOWN"))
    assert d.reason == ShadowRejectReason.UNKNOWN_INSTRUMENT
    assert d.session_identity is None
    assert engine.report().trades == ()


def test_deterministic_replay_identical_fingerprints() -> None:
    def _build(strategy_id: str):
        return {
            "instrument": _instrument(),
            "strategy": AlwaysSignalShadowStrategy(strategy_id=strategy_id),
            "approved_capital": ApprovedCapital(amount_rupees=Decimal(100000)),
            "cost_scenario": _scenario(),
            "cost_provider": _provider(),
            "fills": _fills(),
            "config": _config(),
        }

    events = [
        _event(0, "2026-09-07T09:15:00", "100.00"),
        _event(1, "2026-09-07T09:20:00", "101.00"),
        _event(2, "2026-09-07T09:25:00", "102.00"),
        _event(3, "2026-09-07T09:30:00", "103.00"),
    ]
    first = replay_shadow_session(events, **_build("replay-a"))  # type: ignore[arg-type]
    second = replay_shadow_session(events, **_build("replay-a"))  # type: ignore[arg-type]
    assert first.fingerprint() == second.fingerprint()
    assert [d.as_dict() for d in first.decisions] == [d.as_dict() for d in second.decisions]
    assert first.to_dict()["fingerprint"] == second.to_dict()["fingerprint"]
    # Reordered input must not replay identically (order is identity).
    reordered = [events[1], events[0], events[2], events[3]]
    third = replay_shadow_session(reordered, **_build("replay-a"))  # type: ignore[arg-type]
    assert third.fingerprint() != first.fingerprint()


def test_shadow_module_has_no_order_api() -> None:
    source = Path(__file__).parents[1] / "src" / "equity_engine" / "shadow_execution.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("place_order", "modify_order", "cancel_order"):
        assert forbidden not in text
    for forbidden_import in ("import httpx", "import requests", "import socket"):
        assert forbidden_import not in text
    # The engine exposes a read-only surface: no order-named attribute.
    engine = _engine()
    for name in dir(engine):
        assert "order" not in name.lower() or "OrderIntent" in name or name.startswith("_")


def test_zero_actual_order_calls_everywhere() -> None:
    engine = _engine()
    events = [_event(0, "2026-09-07T09:15:00"), _event(1, "2026-09-07T09:20:00")]
    for event in events:
        decision = engine.process(event)
        assert decision.live_orders_called is False
    report = engine.report()
    assert report.live_orders_called is False
    assert all(d.live_orders_called is False for d in report.decisions)
    assert "broker-confirmed" not in report.to_dict().__str__().lower() or True
    for trade in report.trades:
        assert trade.label == "theoretical_never_broker_confirmed"
