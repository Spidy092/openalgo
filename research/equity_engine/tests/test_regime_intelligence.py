"""Tests for regime_intelligence: gates, classification, cost math, learning loop."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from equity_engine.market_context import (
    MarketContext,
    MarketContextThresholds,
    MarketSnapshot,
    build_market_context,
)
from equity_engine.regime_intelligence import (
    Action,
    Gate,
    LearningLedger,
    Regime,
    RegimeThresholds,
    assert_no_lookahead,
    breakeven_win_rate,
    classify_symbol,
    cost_as_fraction_of_risk,
    emit_paper_decision,
)

IST = timezone(timedelta(hours=5, minutes=30))


def _bars(n=60, start_price=100.0, drift=0.0, seed=1, spike_at=None, noise=0.3):
    import random

    rng = random.Random(seed)
    rows = []
    px = start_price
    base = datetime(2026, 9, 25, 9, 15, tzinfo=IST)
    for i in range(n):
        o = px
        c = px + drift + rng.uniform(-noise, noise)
        h = max(o, c) + rng.uniform(0, 0.2)
        low = min(o, c) - rng.uniform(0, 0.2)
        v = 10000 + rng.randint(-2000, 2000)
        if spike_at is not None and spike_at[0] <= i <= spike_at[1]:
            v *= 8
            h = c + 3.0
            low = c - 3.0
        rows.append((base + timedelta(minutes=5 * i), o, h, low, c, v))
        px = c
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [r[5] for r in rows]},
        index=idx,
    )


def _ctx(vix=Decimal("13")):
    snap = MarketSnapshot(
        captured_at=datetime(2026, 9, 25, 11, 0, tzinfo=IST),
        nifty_open=Decimal("23000"), nifty_last=Decimal("23050"),
        nifty_previous_close=Decimal("22980"),
        india_vix_last=vix, india_vix_previous_close=Decimal("12.8"),
        advancers=1200, decliners=600, unchanged=100,
    )
    return build_market_context(snap, thresholds=MarketContextThresholds(
        large_gap_bps=Decimal("100"), strong_index_move_bps=Decimal("75"),
        high_vix_level=Decimal("20"), vix_jump_pct=Decimal("10"),
        strong_breadth_pct=Decimal("65"), weak_breadth_pct=Decimal("35"),
    ))


def _noon():
    return datetime(2026, 9, 25, 11, 0, tzinfo=IST)


def _at_end(bars):
    return (bars.index[-1] + timedelta(minutes=1)).to_pydatetime()


def _bars_pullback(seed=7, pb_bars=4):
    """Uptrend then a shallow pullback that tags EMA9: the entry setup."""
    import random

    rng = random.Random(seed)
    rows = []
    px = 100.0
    base = datetime(2026, 9, 25, 9, 15, tzinfo=IST)
    for i in range(56 + pb_bars):
        drift = 0.35 if i < 56 else -0.35
        o = px
        c = px + drift + rng.uniform(-0.25, 0.25)
        h = max(o, c) + rng.uniform(0, 0.15)
        low = min(o, c) - rng.uniform(0, 0.15)
        v = 10000 + rng.randint(-2000, 2000)
        rows.append((base + timedelta(minutes=5 * i), o, h, low, c, v))
        px = c
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [r[5] for r in rows]},
        index=idx,
    )



# ------------------------------------------------------------------ gates


def test_vix_kill_gate_blocks():
    bars = _bars()
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars), vix_level=Decimal("21"))
    assert c.gate == Gate.VIX_KILL and c.action == Action.SIT_OUT


def test_vix_below_kill_passes_gate():
    bars = _bars()
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars), vix_level=Decimal("13"))
    assert c.gate is None


def test_event_day_blocks():
    bars = _bars()
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars),
                        event_days=frozenset({_at_end(bars).date()}))
    assert c.gate == Gate.EVENT_DAY and c.action == Action.SIT_OUT


def test_time_gate_blocks_early_and_late():
    early_bars = _bars(n=3)  # 9:15, 9:20, 9:25
    early = classify_symbol("X", early_bars, _ctx(),
                            now=datetime(2026, 9, 25, 9, 30, tzinfo=IST))
    late_bars = _bars(n=68)  # ends 14:50
    late = classify_symbol("X", late_bars, _ctx(),
                           now=datetime(2026, 9, 25, 14, 55, tzinfo=IST))
    assert early.gate == Gate.TIME_BLOCKED
    assert late.gate == Gate.TIME_BLOCKED


# ------------------------------------------------------- classification


def test_trend_up_detected():
    bars = _bars(n=60, drift=0.35, seed=7)
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars))
    assert c.regime == Regime.TREND_UP, c.reasons
    assert c.action == Action.TRADE_LONG


def test_trend_down_without_proven_shorts_sits_out():
    bars = _bars(n=60, drift=-0.35, seed=7)
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars), shorts_proven=False)
    assert c.regime == Regime.TREND_DOWN
    assert c.action == Action.SIT_OUT
    assert any("unproven" in r for r in c.reasons)


def test_trend_down_with_proven_shorts_trades():
    bars = _bars(n=60, drift=-0.35, seed=7)
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars), shorts_proven=True)
    assert c.action == Action.TRADE_SHORT


def test_range_detected_on_flat_data():
    bars = _bars(n=60, drift=0.0, seed=3, noise=0.05)
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars))
    assert c.regime in (Regime.RANGE, Regime.DEAD)
    assert c.action == Action.SIT_OUT


def test_news_volatile_on_volume_spike():
    bars = _bars(n=60, drift=0.1, seed=5, spike_at=(57, 59))
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars))
    assert c.regime == Regime.NEWS_VOLATILE
    assert c.action == Action.SIT_OUT


def test_dont_chase_converts_extended_trend_to_sit_out():
    # strong smooth trend that never pulled back to EMA9 -> must not chase
    bars = _bars(n=60, drift=0.6, seed=11)
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars))
    assert c.action == Action.TRADE_LONG
    d = emit_paper_decision(c, bars, now=_at_end(bars))
    assert d.action == Action.SIT_OUT
    assert "never tagged" in d.rationale


def test_pullback_entry_and_sizing_respects_risk_and_cap():
    bars = _bars_pullback()
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars))
    assert c.regime == Regime.TREND_UP
    d = emit_paper_decision(c, bars, now=_at_end(bars))
    assert d.action == Action.TRADE_LONG, d.rationale
    assert d.risk_rupees <= Decimal("20") + Decimal("5")  # rounding slack
    assert d.entry * d.quantity <= Decimal("950")
    assert d.cost_fraction_of_risk is not None
    assert d.breakeven_win_rate == Decimal("0.333")


def test_sit_out_decision_has_no_prices():
    bars = _bars()
    c = classify_symbol("X", bars, _ctx(), now=_at_end(bars), vix_level=Decimal("25"))
    d = emit_paper_decision(c, bars, now=_at_end(bars))
    assert d.action == Action.SIT_OUT and d.entry is None and d.quantity == 0


# ---------------------------------------------------------------- cost math


def test_cost_fraction_of_risk():
    assert cost_as_fraction_of_risk(Decimal("2.60"), Decimal("20")) == Decimal("0.13")


def test_breakeven_win_rate():
    assert breakeven_win_rate(2) == Decimal(1) / Decimal(3)
    assert breakeven_win_rate(1) == Decimal("0.5")


def test_cost_math_rejects_bad_inputs():
    with pytest.raises(ValueError):
        cost_as_fraction_of_risk(1, 0)
    with pytest.raises(ValueError):
        breakeven_win_rate(0)


# --------------------------------------------------------------- lookahead


def test_assert_no_lookahead_raises_on_future_bars():
    bars = _bars()
    with pytest.raises(ValueError, match="lookahead"):
        assert_no_lookahead(bars, datetime(2026, 9, 25, 9, 20, tzinfo=IST))


def test_classify_rejects_future_bars():
    bars = _bars()
    with pytest.raises(ValueError, match="lookahead"):
        classify_symbol("X", bars, _ctx(),
                        now=datetime(2026, 9, 25, 9, 20, tzinfo=IST))


# ------------------------------------------------------------ learning loop


def test_learning_ledger_scoreboard_and_starving(tmp_path):
    from datetime import timezone as tz

    led = LearningLedger(tmp_path / "learn.jsonl")
    assert led.scoreboard() == {}
    bars = _bars_pullback()
    t = _at_end(bars)
    c = classify_symbol("X", bars, _ctx(), now=t)
    d = emit_paper_decision(c, bars, now=t)
    assert d.action == Action.TRADE_LONG, d.rationale
    led.log_decision(d)
    # close as a loser 6 times -> regime should be starved
    for i in range(6):
        dd = emit_paper_decision(c, bars, now=t)
        led.log_decision(dd)
        led.record_outcome("X", dd.decided_at.isoformat(), "90", "-22.50")
    sb = led.scoreboard()
    assert sb["trend_up"]["trades"] == 6
    assert sb["trend_up"]["win_rate"] == 0
    assert Decimal(sb["trend_up"]["net_pnl"]) < 0
    assert "trend_up" in led.starved_regimes(min_trades=5)
    assert led.starved_regimes(min_trades=7) == []
