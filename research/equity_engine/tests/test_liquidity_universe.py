from datetime import date
from decimal import Decimal

import pandas as pd

from equity_engine.instrument_master import build_nse_equity_master
from equity_engine.liquidity import SpreadObservation, summarize_historical_liquidity, summarize_live_spreads
from equity_engine.universe import (
    CorporateActionAssessment,
    LiveUniverseThresholds,
    ResearchUniverseThresholds,
    evaluate_live_universe_candidate,
    evaluate_research_universe_candidate,
)


def _instrument(*, mis: bool = True, suspended: bool = False):
    key = "NSE_EQ|INE000000001"
    row = {
        "segment": "NSE_EQ",
        "name": "TEST LTD",
        "exchange": "NSE",
        "isin": "INE000000001",
        "instrument_type": "EQ",
        "instrument_key": key,
        "lot_size": 1,
        "freeze_quantity": 100000,
        "exchange_token": "123",
        "tick_size": 5,
        "trading_symbol": "TEST",
        "short_name": "Test",
        "security_type": "NORMAL",
        "cas_eligible": False,
    }
    snapshot = build_nse_equity_master(
        as_of_date=date(2026, 9, 7),
        bod_rows=[row],
        mis_rows=[{"instrument_key": key}] if mis else [],
        suspended_rows=[{"instrument_key": key}] if suspended else [],
        tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
    )
    return snapshot.instruments[0]


def _frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-01 09:20", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-02 09:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-09-02 09:20", tz="Asia/Kolkata"),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100, 101, 102, 103],
            "volume": [1000, 2000, 3000, 4000],
        },
        index=index,
    )


def test_historical_liquidity_uses_labeled_notional_proxy() -> None:
    evidence = summarize_historical_liquidity(
        frame=_frame(),
        affordable_quantity_after_entry_costs=9,
    )
    assert evidence.observed_trading_days == 2
    assert evidence.median_daily_volume_shares == Decimal("5000")
    assert evidence.median_daily_notional_proxy_rupees == Decimal("510000")
    assert evidence.last_price_rupees == Decimal("103")


def test_live_spread_is_computed_from_actual_bid_ask_observations() -> None:
    spread = summarize_live_spreads(
        [
            SpreadObservation(Decimal("100.00"), Decimal("100.05")),
            SpreadObservation(Decimal("100.10"), Decimal("100.15")),
            SpreadObservation(Decimal("100.20"), Decimal("100.25")),
        ]
    )
    assert spread.spread_observations == 3
    assert spread.median_spread_bps > 0


def test_research_and_live_eligibility_are_separate() -> None:
    instrument = _instrument()
    liquidity = summarize_historical_liquidity(
        frame=_frame(),
        affordable_quantity_after_entry_costs=9,
    )
    research = evaluate_research_universe_candidate(
        instrument=instrument,
        liquidity=liquidity,
        corporate_actions=CorporateActionAssessment(complete=True, blocking_events=()),
        thresholds=ResearchUniverseThresholds(
            max_last_price_rupees=Decimal("500"),
            min_median_daily_notional_proxy_rupees=Decimal("100000"),
            min_median_daily_volume_shares=Decimal("1000"),
            min_observed_trading_days=2,
            min_affordable_quantity=1,
        ),
    )
    assert research.eligible is True

    live = evaluate_live_universe_candidate(
        research_decision=research,
        spread=summarize_live_spreads(
            [SpreadObservation(Decimal("100"), Decimal("100.05"))]
        ),
        thresholds=LiveUniverseThresholds(
            max_median_spread_bps=Decimal("10"),
            min_spread_observations=5,
        ),
    )
    assert live.eligible is False
    assert "insufficient live spread observations" in live.violations


def test_not_mis_and_suspended_are_hard_rejections() -> None:
    instrument = _instrument(mis=False, suspended=True)
    liquidity = summarize_historical_liquidity(frame=_frame(), affordable_quantity_after_entry_costs=9)
    decision = evaluate_research_universe_candidate(
        instrument=instrument,
        liquidity=liquidity,
        corporate_actions=CorporateActionAssessment(complete=True, blocking_events=()),
        thresholds=ResearchUniverseThresholds(
            max_last_price_rupees=Decimal("500"),
            min_median_daily_notional_proxy_rupees=Decimal("0"),
            min_median_daily_volume_shares=Decimal("0"),
            min_observed_trading_days=1,
            min_affordable_quantity=1,
        ),
    )
    assert decision.eligible is False
    assert any("MIS" in violation for violation in decision.violations)
    assert any("suspended" in violation for violation in decision.violations)
