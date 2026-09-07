from datetime import date
from decimal import Decimal

import pandas as pd

from equity_engine.historical_membership import HistoricalTradingStatus, assess_historical_membership
from equity_engine.instrument_master import build_nse_equity_master
from equity_engine.liquidity import SpreadObservation, summarize_historical_liquidity, summarize_live_spreads
from equity_engine.tick_size import (
    FixedTickSizePolicy,
    TickSizeVerification,
    assess_tick_policy_coverage,
)
from equity_engine.universe import (
    CorporateActionAssessment,
    LiveUniverseThresholds,
    ResearchUniverseThresholds,
    evaluate_live_universe_candidate,
    evaluate_research_universe_candidate,
)

KEY = "NSE_EQ|INE000000001"


def _instrument(*, mis: bool = True, suspended: bool = False):
    row = {
        "segment": "NSE_EQ",
        "name": "TEST LTD",
        "exchange": "NSE",
        "isin": "INE000000001",
        "instrument_type": "EQ",
        "instrument_key": KEY,
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
        mis_rows=[{"instrument_key": KEY}] if mis else [],
        suspended_rows=[{"instrument_key": KEY}] if suspended else [],
        tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
    )
    return snapshot.instruments[0]


def _tick_ok() -> TickSizeVerification:
    return TickSizeVerification(
        passed=True,
        expected_rupees=Decimal("0.05"),
        observed_rupees=Decimal("0.05"),
        source="synthetic-current-test",
    )


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


def _membership(*, missing_second_day: bool = False):
    statuses = [
        HistoricalTradingStatus(
            trade_date=date(2026, 9, 1),
            instrument_key=KEY,
            listed_on_nse=True,
            normal_equity=True,
            tradeable_in_normal_market=True,
            source="nse-security-master-2026-09-01",
        )
    ]
    if not missing_second_day:
        statuses.append(
            HistoricalTradingStatus(
                trade_date=date(2026, 9, 2),
                instrument_key=KEY,
                listed_on_nse=True,
                normal_equity=True,
                tradeable_in_normal_market=True,
                source="nse-security-master-2026-09-02",
            )
        )
    return assess_historical_membership(
        instrument_key=KEY,
        trading_dates=[date(2026, 9, 1), date(2026, 9, 2)],
        statuses=statuses,
    )


def _tick_coverage(membership=None):
    membership = membership or _membership()
    return assess_tick_policy_coverage(
        policy=FixedTickSizePolicy(Decimal("0.05"), "synthetic-historical-test"),
        trading_dates=membership.eligible_dates,
    )


def _research_decision(membership=None):
    membership = membership or _membership()
    liquidity = summarize_historical_liquidity(frame=_frame(), affordable_quantity_after_entry_costs=9)
    return evaluate_research_universe_candidate(
        instrument_key=KEY,
        liquidity=liquidity,
        corporate_actions=CorporateActionAssessment(complete=True, blocking_events=()),
        historical_membership=membership,
        tick_coverage=_tick_coverage(membership),
        thresholds=ResearchUniverseThresholds(
            max_last_price_rupees=Decimal("500"),
            min_median_daily_notional_proxy_rupees=Decimal("100000"),
            min_median_daily_volume_shares=Decimal("1000"),
            min_observed_trading_days=2,
            min_affordable_quantity=1,
        ),
    )


def test_historical_liquidity_uses_labeled_notional_proxy() -> None:
    evidence = summarize_historical_liquidity(frame=_frame(), affordable_quantity_after_entry_costs=9)
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


def test_research_uses_point_in_time_exchange_evidence_not_current_mis() -> None:
    research = _research_decision()
    assert research.eligible is True

    # Current broker state is intentionally evaluated only in the live gate.
    live = evaluate_live_universe_candidate(
        research_decision=research,
        current_instrument=_instrument(mis=False, suspended=False),
        current_tick_verification=_tick_ok(),
        spread=summarize_live_spreads(
            [SpreadObservation(Decimal("100"), Decimal("100.05")) for _ in range(5)]
        ),
        thresholds=LiveUniverseThresholds(
            max_median_spread_bps=Decimal("10"),
            min_spread_observations=5,
        ),
    )
    assert live.eligible is False
    assert any("current Upstox NSE MIS" in violation for violation in live.violations)


def test_live_gate_rejects_current_suspension_and_insufficient_spread_samples() -> None:
    research = _research_decision()
    live = evaluate_live_universe_candidate(
        research_decision=research,
        current_instrument=_instrument(mis=True, suspended=True),
        current_tick_verification=_tick_ok(),
        spread=summarize_live_spreads([SpreadObservation(Decimal("100"), Decimal("100.05"))]),
        thresholds=LiveUniverseThresholds(
            max_median_spread_bps=Decimal("10"),
            min_spread_observations=5,
        ),
    )
    assert live.eligible is False
    assert any("current Upstox suspended" in violation for violation in live.violations)
    assert "insufficient live spread observations" in live.violations


def test_missing_point_in_time_date_blocks_historical_research() -> None:
    membership = _membership(missing_second_day=True)
    decision = _research_decision(membership)
    assert decision.eligible is False
    assert any("point-in-time exchange evidence is missing" in violation for violation in decision.violations)


def test_failed_current_tick_verification_blocks_live_not_historical_research() -> None:
    research = _research_decision()
    assert research.eligible is True

    live = evaluate_live_universe_candidate(
        research_decision=research,
        current_instrument=_instrument(),
        current_tick_verification=TickSizeVerification(
            passed=False,
            expected_rupees=Decimal("0.10"),
            observed_rupees=Decimal("0.05"),
            source="synthetic-current-test",
        ),
        spread=summarize_live_spreads(
            [SpreadObservation(Decimal("100"), Decimal("100.05")) for _ in range(5)]
        ),
        thresholds=LiveUniverseThresholds(
            max_median_spread_bps=Decimal("10"),
            min_spread_observations=5,
        ),
    )
    assert live.eligible is False
    assert any("current tick-size verification failed" in violation for violation in live.violations)
