"""Dry shadow-scan demo: runs the regime pipeline over a synthetic trading day.

No network, no broker, no orders. Simulates 6 symbols x 5-min bars, classifies
each symbol at 11:00 IST, emits paper decisions, then closes them against the
next bars' actual outcomes to exercise the learning loop end to end.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from equity_engine.market_context import (  # noqa: E402
    MarketContextThresholds,
    MarketSnapshot,
    build_market_context,
)
from equity_engine.regime_intelligence import (  # noqa: E402
    Action,
    LearningLedger,
    RegimeThresholds,
    classify_symbol,
    emit_paper_decision,
)

IST = timezone(timedelta(hours=5, minutes=30))
DAY = datetime(2026, 9, 25, tzinfo=IST)


def synth(symbol: str, drift: float, seed: int, n: int = 78,
          pb_mult: float = 1.2) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    px = 100.0 + seed
    for i in range(n):
        # regime shifts mid-day for realism: trend -> pullback -> chop
        d = drift if i < 56 else (-drift * pb_mult if i < 62 else 0.0)
        o = px
        c = px + d + rng.uniform(-0.25, 0.25)
        h = max(o, c) + rng.uniform(0, 0.15)
        low = min(o, c) - rng.uniform(0, 0.15)
        v = 10000 + rng.randint(-2000, 2000)
        rows.append((DAY + timedelta(hours=9, minutes=15) + timedelta(minutes=5 * i),
                     o, h, low, c, v))
        px = c
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [r[5] for r in rows]},
        index=idx,
    )


def main() -> None:
    symbols = {
        "TRENDA": (0.35, 11, 1.2), "TRENDB": (0.30, 22, 1.2),
        "PULLBACK": (0.35, 77, 0.9),  # gentle pullback: tags EMA9 -> trade
        "FALLER": (-0.32, 33, 1.2),
        "FLAT1": (0.0, 44, 1.2), "FLAT2": (0.02, 55, 1.2),
        "CHOPPY": (0.05, 66, 1.2),
    }
    snap = MarketSnapshot(
        captured_at=DAY + timedelta(hours=11),
        nifty_open=Decimal("23000"), nifty_last=Decimal("23060"),
        nifty_previous_close=Decimal("22980"),
        india_vix_last=Decimal("13.4"), india_vix_previous_close=Decimal("13.1"),
        advancers=1150, decliners=650, unchanged=100,
    )
    ctx = build_market_context(
        snap,
        thresholds=MarketContextThresholds(
            large_gap_bps=Decimal("100"), strong_index_move_bps=Decimal("75"),
            high_vix_level=Decimal("20"), vix_jump_pct=Decimal("10"),
            strong_breadth_pct=Decimal("65"), weak_breadth_pct=Decimal("35")),
    )
    ledger = LearningLedger("/tmp/regime_demo/learn.jsonl")
    if Path("/tmp/regime_demo/learn.jsonl").exists():
        Path("/tmp/regime_demo/learn.jsonl").unlink()

    scan_at = DAY + timedelta(hours=14, minutes=11)  # after the 14:00 bar
    print(f"shadow scan at {scan_at.isoformat()} | VIX 13.4 (below kill level)\n")
    for sym, (drift, seed, pb) in symbols.items():
        bars = synth(sym, drift, seed, pb_mult=pb)
        asof = bars.iloc[:60]  # only bars up to 14:00 visible at scan time
        c = classify_symbol(sym, asof, ctx, now=scan_at, vix_level=Decimal("13.4"))
        d = emit_paper_decision(c, asof, now=scan_at)
        ledger.log_decision(d)
        # close against reality: exit at 14:40 bar close, costs deducted
        exit_px = Decimal(str(bars["close"].iloc[68]))
        if d.action == Action.TRADE_LONG:
            gross = (exit_px - d.entry) * d.quantity
        elif d.action == Action.TRADE_SHORT:
            gross = (d.entry - exit_px) * d.quantity
        else:
            gross = None
        if gross is not None:
            net = gross - Decimal("2.60")
            ledger.record_outcome(sym, d.decided_at.isoformat(), exit_px, net,
                                  exit_at=bars.index[68].to_pydatetime())
            print(f"{sym:7s} {c.regime.value:13s} -> {d.action.value:11s} "
                  f"entry={d.entry} stop={d.stop} qty={d.quantity} "
                  f"net={net:+.2f}")
        else:
            print(f"{sym:7s} {c.regime.value:13s} -> SIT_OUT  ({d.rationale[:70]})")

    print("\n--- per-regime scoreboard (learning loop) ---")
    for regime, s in ledger.scoreboard().items():
        print(f"{regime:13s} trades={s['trades']} win_rate={s['win_rate']} "
              f"net={s['net_pnl']} PF={s['profit_factor']}")
    print("starved regimes:", ledger.starved_regimes(min_trades=1))


if __name__ == "__main__":
    main()
