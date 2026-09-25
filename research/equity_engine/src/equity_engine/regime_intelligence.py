"""Regime-aware paper-trading decision layer (intelligent-trader-v1).

This is the piece no branch had: instead of fixed generic strategies, every
scan classifies the *current market situation* and only acts when the
situation favors it. Nothing here places orders — it emits paper decisions
for the shadow/paper ledger and logs outcomes for the learning loop.

Pipeline (all explicit, auditable):
  VIX kill-gate (A1) -> event calendar (A2) -> time-of-day gate (A2)
  -> per-symbol regime classification -> decision + rationale
  -> outcome logging -> per-regime scoreboard (learning loop)

Regimes: TREND_UP, TREND_DOWN, RANGE, NEWS_VOLATILE, DEAD.
Gates (refusals, not regimes): VIX_KILL, EVENT_DAY, TIME_BLOCKED.

Cost math (A3): cost_as_fraction_of_risk(), breakeven_win_rate().
OOS discipline (A4): assert_no_lookahead() — a decision timestamped T may only
use bars with timestamps <= T.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .market_context import MarketContext
from .provenance import validate_ohlcv_frame

_BPS = Decimal("10000")
_HUNDRED = Decimal("100")


class Regime(StrEnum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    NEWS_VOLATILE = "news_volatile"
    DEAD = "dead"


class Gate(StrEnum):
    """Hard refusals. A gate hit means SIT_OUT regardless of regime."""

    VIX_KILL = "vix_kill"  # India VIX above kill level: fear regime, whipsaws
    EVENT_DAY = "event_day"  # MPC / Budget / election / result-day: observe only
    TIME_BLOCKED = "time_blocked"  # outside the tradable window


class Action(StrEnum):
    TRADE_LONG = "trade_long"
    TRADE_SHORT = "trade_short"
    SIT_OUT = "sit_out"


@dataclass(frozen=True)
class RegimeThresholds:
    vix_kill_level: Decimal = Decimal("20")  # A1: VIX > 20 -> no trading
    vix_halve_level: Decimal = Decimal("17")  # 17-20 -> halve size (flag only)
    adx_trend: Decimal = Decimal("20")  # ADX >= 20 counts as trending
    ema_fast: int = 9
    ema_slow: int = 21
    adx_period: int = 14
    atr_period: int = 14
    vol_mult_trend: Decimal = Decimal("1.2")  # volume vs 20-bar mean for trend
    vol_mult_news: Decimal = Decimal("3")  # volume spike -> news volatile
    atr_expansion_news: Decimal = Decimal("1.5")
    range_atr_band: Decimal = Decimal("1")  # within +/-1 ATR of VWAP = range
    no_entry_before: time = time(9, 45)  # A2: opening chop
    no_entry_after: time = time(14, 30)  # A2: avoid close liquidity traps
    hard_exit_by: time = time(15, 0)  # flat well before 15:30 close
    risk_rupees: Decimal = Decimal("20")  # per-trade planned risk
    max_position_rupees: Decimal = Decimal("950")


@dataclass(frozen=True)
class RegimeClassification:
    symbol: str
    regime: Regime
    gate: Gate | None
    action: Action
    reasons: tuple[str, ...]
    indicators: Mapping[str, str]


@dataclass(frozen=True)
class PaperDecision:
    symbol: str
    regime: Regime
    action: Action
    entry: Decimal | None
    stop: Decimal | None
    target: Decimal | None
    quantity: int
    risk_rupees: Decimal
    rationale: str
    decided_at: datetime
    cost_fraction_of_risk: Decimal | None = None
    breakeven_win_rate: Decimal | None = None


# ---------------------------------------------------------------- indicators


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_s = tr.ewm(span=n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=n, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=n, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return dx.ewm(span=n, adjust=False).mean().fillna(0)


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    day = df.index.date
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, float("nan"))
    return (pv / vv).ffill()


# ------------------------------------------------------------------- cost math (A3)


def cost_as_fraction_of_risk(round_trip_cost_rupees: Any, risk_rupees: Any) -> Decimal:
    """What share of one risk unit does friction eat? At Rs 1000 capital this
    is the binding constraint of the whole program."""
    cost = Decimal(str(round_trip_cost_rupees))
    risk = Decimal(str(risk_rupees))
    if risk <= 0:
        raise ValueError("risk_rupees must be positive")
    return cost / risk


def breakeven_win_rate(avg_win_r: Any, avg_loss_r: Any = 1) -> Decimal:
    """Win rate needed for zero expectancy given average win/loss in R
    multiples. A 2R strategy needs 1/(1+2) = 33.3% before costs."""
    win = Decimal(str(avg_win_r))
    loss = Decimal(str(avg_loss_r))
    if win <= 0 or loss <= 0:
        raise ValueError("avg_win_r and avg_loss_r must be positive")
    return loss / (win + loss)


# --------------------------------------------------------------- lookahead (A4)


def assert_no_lookahead(bars: pd.DataFrame, decided_at: datetime) -> None:
    """A decision timestamped T may only use bars with timestamps <= T."""
    if decided_at.tzinfo is None:
        raise ValueError("decided_at must be timezone-aware")
    future = bars.index[bars.index > decided_at]
    if len(future):
        raise ValueError(
            f"lookahead detected: {len(future)} bars after decision time {decided_at}"
        )


# ------------------------------------------------------------------ classifier


def classify_symbol(
    symbol: str,
    bars: pd.DataFrame,
    ctx: MarketContext,
    *,
    now: datetime,
    event_days: frozenset[date] = frozenset(),
    thresholds: RegimeThresholds = RegimeThresholds(),
    vix_level: Decimal | None = None,
    shorts_proven: bool = False,
) -> RegimeClassification:
    """Classify one symbol's current situation. Every branch cites its inputs."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    violations = validate_ohlcv_frame(bars)
    if violations:
        raise ValueError("invalid OHLCV frame: " + "; ".join(violations))
    assert_no_lookahead(bars, now)

    reasons: list[str] = []
    ind: dict[str, str] = {}

    # Gate 1: VIX kill (A1)
    if vix_level is not None and vix_level > thresholds.vix_kill_level:
        return RegimeClassification(
            symbol, Regime.DEAD, Gate.VIX_KILL, Action.SIT_OUT,
            (f"India VIX {vix_level} > kill level {thresholds.vix_kill_level}: fear regime",),
            {"vix": str(vix_level)},
        )

    # Gate 2: event day (A2)
    if now.date() in event_days:
        return RegimeClassification(
            symbol, Regime.DEAD, Gate.EVENT_DAY, Action.SIT_OUT,
            (f"{now.date()} is an event day (MPC/Budget/election/results): observe only",),
            {},
        )

    # Gate 3: time of day (A2)
    t = now.time()
    if t < thresholds.no_entry_before or t >= thresholds.no_entry_after:
        return RegimeClassification(
            symbol, Regime.DEAD, Gate.TIME_BLOCKED, Action.SIT_OUT,
            (f"time {t} outside entry window "
             f"{thresholds.no_entry_before}-{thresholds.no_entry_after}",),
            {},
        )

    if len(bars) < max(thresholds.ema_slow, thresholds.adx_period) + 5:
        return RegimeClassification(
            symbol, Regime.DEAD, None, Action.SIT_OUT,
            ("insufficient bars for indicators",), {}
        )

    close = bars["close"]
    ema_f = _ema(close, thresholds.ema_fast).iloc[-1]
    ema_s = _ema(close, thresholds.ema_slow).iloc[-1]
    adx = _adx(bars, thresholds.adx_period).iloc[-1]
    atr = _atr(bars, thresholds.atr_period).iloc[-1]
    vwap = _session_vwap(bars).iloc[-1]
    last = close.iloc[-1]
    vol_mean = bars["volume"].iloc[-20:].mean()
    vol_mult = (bars["volume"].iloc[-1] / vol_mean) if vol_mean > 0 else 0.0
    atr_now = atr
    atr_mean = _atr(bars, thresholds.atr_period).iloc[-20:].mean()
    atr_exp = (atr_now / atr_mean) if atr_mean > 0 else 1.0
    # structure: the 10-bar extreme must be recent (within last 5 bars), which
    # tolerates a shallow pullback but rejects a broken trend
    lookback = close.iloc[-10:]
    hh = lookback.idxmax() >= lookback.index[-5]
    ll = lookback.idxmin() >= lookback.index[-5]

    ind = {
        "close": f"{last:.2f}", "vwap": f"{vwap:.2f}",
        "ema9": f"{ema_f:.2f}", "ema21": f"{ema_s:.2f}",
        "adx14": f"{adx:.1f}", "atr": f"{atr_now:.2f}",
        "vol_mult": f"{vol_mult:.2f}", "atr_expansion": f"{atr_exp:.2f}",
    }

    d = lambda x: Decimal(str(x))  # noqa: E731

    # NEWS_VOLATILE: abnormal volume + ATR expansion (unusual activity)
    if d(vol_mult) >= thresholds.vol_mult_news and d(atr_exp) >= thresholds.atr_expansion_news:
        reasons.append(
            f"volume {vol_mult:.1f}x 20-bar mean with ATR expansion {atr_exp:.1f}x: "
            "news-driven whipsaw risk"
        )
        return RegimeClassification(symbol, Regime.NEWS_VOLATILE, None, Action.SIT_OUT,
                                    tuple(reasons), ind)

    trending = d(adx) >= thresholds.adx_trend
    # meaningful excursion: the 10-bar range must exceed 2 ATR, otherwise ADX
    # is just reading noise as trend (common on flat random-walk data)
    excursion_atr = (lookback.max() - lookback.min()) / atr_now if atr_now > 0 else 0.0
    structured = excursion_atr >= 2.0
    vol_ok = d(vol_mult) >= thresholds.vol_mult_trend
    if trending and structured and last > vwap and ema_f > ema_s and hh:
        reasons.append(
            f"ADX {adx:.0f}>=20, close {last:.2f}>VWAP {vwap:.2f}, EMA stack up, 10-bar high recent"
            + (f", vol {vol_mult:.1f}x confirms" if vol_ok else ", vol unconfirmed")
        )
        return RegimeClassification(symbol, Regime.TREND_UP, None, Action.TRADE_LONG,
                                    tuple(reasons), ind)
    if trending and structured and last < vwap and ema_f < ema_s and ll:
        reasons.append(
            f"ADX {adx:.0f}>=20, close {last:.2f}<VWAP {vwap:.2f}, EMA stack down, 10-bar low recent"
            + (f", vol {vol_mult:.1f}x confirms" if vol_ok else ", vol unconfirmed")
        )
        action = Action.TRADE_SHORT if shorts_proven else Action.SIT_OUT
        if not shorts_proven:
            reasons.append("shorts unproven in paper: sit out until proven")
        return RegimeClassification(symbol, Regime.TREND_DOWN, None, action,
                                    tuple(reasons), ind)

    if abs(last - vwap) <= float(thresholds.range_atr_band) * atr_now:
        reasons.append(f"ADX {adx:.0f}<20 and |close-VWAP| within 1 ATR: range")
        return RegimeClassification(symbol, Regime.RANGE, None, Action.SIT_OUT,
                                    tuple(reasons), ind)

    reasons.append("no trend / range / news signature: dead")
    return RegimeClassification(symbol, Regime.DEAD, None, Action.SIT_OUT,
                                tuple(reasons), ind)


def emit_paper_decision(
    classification: RegimeClassification,
    bars: pd.DataFrame,
    *,
    now: datetime,
    round_trip_cost_rupees: Any = Decimal("2.60"),
    target_r: Decimal = Decimal("2"),
    thresholds: RegimeThresholds = RegimeThresholds(),
) -> PaperDecision:
    """Turn a classification into a concrete paper decision. Never chases:
    longs only enter on a pullback within 0.5 ATR of the VWAP/EMA9 reference."""
    c = classification
    if c.action == Action.SIT_OUT:
        return PaperDecision(
            symbol=c.symbol, regime=c.regime, action=Action.SIT_OUT,
            entry=None, stop=None, target=None, quantity=0,
            risk_rupees=Decimal("0"),
            rationale="; ".join(c.reasons), decided_at=now,
        )

    last = Decimal(str(bars["close"].iloc[-1]))
    atr = _atr(bars, thresholds.atr_period).iloc[-1]
    atr_d = Decimal(str(atr))
    vwap = Decimal(str(_session_vwap(bars).iloc[-1]))
    ema9 = Decimal(str(_ema(bars["close"], thresholds.ema_fast).iloc[-1]))

    if c.action == Action.TRADE_LONG:
        ref = ema9  # pullback reference: the trend-riding dynamic support
        bar_low = Decimal(str(bars["low"].iloc[-1]))
        tagged = bar_low <= ref  # price touched support intraday: the pullback
        held = last >= ref - Decimal("0.5") * atr_d  # but did not break down
        if not tagged:
            return PaperDecision(
                symbol=c.symbol, regime=c.regime, action=Action.SIT_OUT,
                entry=None, stop=None, target=None, quantity=0,
                risk_rupees=Decimal("0"),
                rationale=(f"don't chase: price {last} never tagged EMA9 support "
                           f"{ref:.2f}; wait for the pullback"),
                decided_at=now,
            )
        if not held:
            return PaperDecision(
                symbol=c.symbol, regime=c.regime, action=Action.SIT_OUT,
                entry=None, stop=None, target=None, quantity=0,
                risk_rupees=Decimal("0"),
                rationale=(f"support lost: close {last} broke EMA9 {ref:.2f} "
                           "by > 0.5 ATR; trend may be failing"),
                decided_at=now,
            )
        entry = ref
        stop = ref - atr_d  # 1 ATR stop below the pullback reference
    else:  # TRADE_SHORT
        ref = ema9
        bar_high = Decimal(str(bars["high"].iloc[-1]))
        tagged = bar_high >= ref
        held = last <= ref + Decimal("0.5") * atr_d
        if not tagged:
            return PaperDecision(
                symbol=c.symbol, regime=c.regime, action=Action.SIT_OUT,
                entry=None, stop=None, target=None, quantity=0,
                risk_rupees=Decimal("0"),
                rationale=(f"don't chase: price {last} never tagged EMA9 resistance "
                           f"{ref:.2f}; wait for the pullback"),
                decided_at=now,
            )
        if not held:
            return PaperDecision(
                symbol=c.symbol, regime=c.regime, action=Action.SIT_OUT,
                entry=None, stop=None, target=None, quantity=0,
                risk_rupees=Decimal("0"),
                rationale=(f"resistance lost: close {last} broke EMA9 {ref:.2f} "
                           "by > 0.5 ATR; trend may be failing"),
                decided_at=now,
            )
        entry = ref
        stop = ref + atr_d

    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        raise ValueError("degenerate stop: entry == stop")
    qty = int(thresholds.risk_rupees / risk_per_share)
    # cap by max position value
    max_qty = int(thresholds.max_position_rupees / entry) if entry > 0 else 0
    qty = max(0, min(qty, max_qty))
    if qty <= 0:
        return PaperDecision(
            symbol=c.symbol, regime=c.regime, action=Action.SIT_OUT,
            entry=None, stop=None, target=None, quantity=0,
            risk_rupees=Decimal("0"),
            rationale=f"position sizing gives 0 shares (risk {thresholds.risk_rupees} "
                      f"vs stop distance {risk_per_share:.2f})",
            decided_at=now,
        )
    risk = risk_per_share * qty
    target = entry + target_r * (entry - stop) if c.action == Action.TRADE_LONG \
        else entry - target_r * (stop - entry)
    cost = Decimal(str(round_trip_cost_rupees))

    rationale = (
        f"{c.regime.value}: entry {entry:.2f} at pullback ref, stop {stop:.2f} "
        f"({risk_per_share:.2f}/share), target {target:.2f} ({target_r}R), "
        f"qty {qty} (risk Rs {risk:.2f}). " + "; ".join(c.reasons)
    )
    return PaperDecision(
        symbol=c.symbol, regime=c.regime, action=c.action,
        entry=entry.quantize(Decimal("0.01")), stop=stop.quantize(Decimal("0.01")),
        target=target.quantize(Decimal("0.01")), quantity=qty,
        risk_rupees=risk.quantize(Decimal("0.01")),
        rationale=rationale, decided_at=now,
        cost_fraction_of_risk=(cost / risk).quantize(Decimal("0.001")),
        breakeven_win_rate=breakeven_win_rate(target_r).quantize(Decimal("0.001")),
    )


# ------------------------------------------------------------- learning loop


@dataclass
class OutcomeRecord:
    decided_at: str
    symbol: str
    regime: str
    action: str
    entry: str
    stop: str
    target: str
    quantity: int
    rationale: str
    exit_price: str | None = None
    exit_at: str | None = None
    net_pnl_rupees: str | None = None
    closed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class LearningLedger:
    """Append-only JSONL log of paper decisions + outcomes, with a per-regime
    scoreboard. Losing regimes are starved, never hidden."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(self, decision: PaperDecision) -> None:
        rec = OutcomeRecord(
            decided_at=decision.decided_at.isoformat(), symbol=decision.symbol,
            regime=decision.regime.value, action=decision.action.value,
            entry=str(decision.entry), stop=str(decision.stop),
            target=str(decision.target), quantity=decision.quantity,
            rationale=decision.rationale,
        )
        with self.path.open("a") as f:
            f.write(json.dumps(rec.as_dict()) + "\n")

    def record_outcome(self, symbol: str, decided_at: str, exit_price: Any,
                       net_pnl_rupees: Any, exit_at: datetime | None = None) -> bool:
        """Close the most recent open record for (symbol, decided_at)."""
        if not self.path.exists():
            return False
        lines = self.path.read_text().splitlines()
        updated = False
        for i in range(len(lines) - 1, -1, -1):
            rec = json.loads(lines[i])
            if (rec["symbol"] == symbol and rec["decided_at"] == decided_at
                    and not rec["closed"]):
                rec["closed"] = True
                rec["exit_price"] = str(exit_price)
                rec["net_pnl_rupees"] = str(net_pnl_rupees)
                rec["exit_at"] = (exit_at or datetime.now(timezone.utc)).isoformat()
                lines[i] = json.dumps(rec)
                updated = True
                break
        if updated:
            self.path.write_text("\n".join(lines) + "\n")
        return updated

    def scoreboard(self) -> dict[str, dict[str, Any]]:
        """Per-regime: trades, wins, net P&L, profit factor. Open records excluded."""
        if not self.path.exists():
            return {}
        agg: dict[str, dict[str, Any]] = {}
        for line in self.path.read_text().splitlines():
            rec = json.loads(line)
            if not rec.get("closed") or rec.get("action") == Action.SIT_OUT.value:
                continue
            r = rec["regime"]
            s = agg.setdefault(r, {"trades": 0, "wins": 0, "net": Decimal("0"),
                                   "gross_win": Decimal("0"), "gross_loss": Decimal("0")})
            pnl = Decimal(str(rec["net_pnl_rupees"]))
            s["trades"] += 1
            s["net"] += pnl
            if pnl > 0:
                s["wins"] += 1
                s["gross_win"] += pnl
            else:
                s["gross_loss"] += -pnl
        out = {}
        for regime, s in agg.items():
            pf = (s["gross_win"] / s["gross_loss"]) if s["gross_loss"] > 0 else None
            out[regime] = {
                "trades": s["trades"],
                "win_rate": round(s["wins"] / s["trades"], 3) if s["trades"] else 0,
                "net_pnl": str(s["net"].quantize(Decimal("0.01"))),
                "profit_factor": str(pf.quantize(Decimal("0.01"))) if pf else None,
            }
        return out

    def starved_regimes(self, min_trades: int = 5) -> list[str]:
        """Regimes with >= min_trades closed and negative net: stop trading them."""
        return [r for r, s in self.scoreboard().items()
                if s["trades"] >= min_trades and Decimal(s["net_pnl"]) < 0]
