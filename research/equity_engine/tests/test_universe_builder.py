from datetime import date
from decimal import Decimal

import pandas as pd

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.historical_membership import HistoricalTradingStatus, assess_historical_membership
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.universe import CorporateActionAssessment, ResearchUniverseThresholds
from equity_engine.universe_builder import ResearchUniverseCandidateData, build_research_universe


def _frame(day1: str = "2026-08-31", day2: str = "2026-09-01") -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{day1} 09:15", tz="Asia/Kolkata"),
            pd.Timestamp(f"{day1} 15:20", tz="Asia/Kolkata"),
            pd.Timestamp(f"{day2} 09:15", tz="Asia/Kolkata"),
            pd.Timestamp(f"{day2} 15:20", tz="Asia/Kolkata"),
        ]
    )
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100, 101, 102, 103],
            "volume": [100_000, 100_000, 120_000, 120_000],
        },
        index=index,
    )


def _membership(key: str, frame: pd.DataFrame, *, omit_last: bool = False):
    dates = tuple(sorted(set(frame.index.date)))
    statuses = [
        HistoricalTradingStatus(
            trade_date=trade_date,
            instrument_key=key,
            listed_on_nse=True,
            normal_equity=True,
            tradeable_in_normal_market=True,
            source=f"nse-security-master-{trade_date.isoformat()}",
        )
        for trade_date in (dates[:-1] if omit_last else dates)
    ]
    return assess_historical_membership(
        instrument_key=key,
        trading_dates=dates,
        statuses=statuses,
    )


def _candidate(key: str, frame: pd.DataFrame, *, omit_membership_last: bool = False):
    return ResearchUniverseCandidateData(
        instrument_key=key,
        frame=frame,
        dataset_fingerprint=f"fingerprint-{key}",
        historical_membership=_membership(key, frame, omit_last=omit_membership_last),
        tick_size_policy=FixedTickSizePolicy(
            tick_size_rupees=Decimal("0.05"),
            source="synthetic-historical-test",
        ),
        corporate_actions=CorporateActionAssessment(complete=True, blocking_events=()),
    )


def _thresholds() -> ResearchUniverseThresholds:
    return ResearchUniverseThresholds(
        max_last_price_rupees=Decimal("500"),
        min_median_daily_notional_proxy_rupees=Decimal("1000000"),
        min_median_daily_volume_shares=Decimal("100000"),
        min_observed_trading_days=2,
        min_affordable_quantity=1,
    )


def test_builder_returns_auditable_eligible_candidate() -> None:
    candidate = _candidate("NSE_EQ|INE000000001", _frame())
    result = build_research_universe(
        candidates=[candidate],
        selection_cutoff=date(2026, 9, 1),
        capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    )

    assert result.eligible_instrument_keys == ("NSE_EQ|INE000000001",)
    audit = result.audits[0]
    assert audit.decision.eligible is True
    assert audit.affordable_quantity == 9
    assert audit.dataset_fingerprint == "fingerprint-NSE_EQ|INE000000001"
    assert audit.data_end.date() == date(2026, 9, 1)


def test_builder_rejects_future_data_in_universe_selection() -> None:
    candidate = _candidate(
        "NSE_EQ|INE000000002",
        _frame(day1="2026-09-01", day2="2026-09-02"),
    )
    result = build_research_universe(
        candidates=[candidate],
        selection_cutoff=date(2026, 9, 1),
        capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    )

    assert result.eligible_instrument_keys == ()
    assert any("lookahead" in item for item in result.audits[0].decision.violations)


def test_builder_rejects_incomplete_point_in_time_membership() -> None:
    candidate = _candidate(
        "NSE_EQ|INE000000003",
        _frame(),
        omit_membership_last=True,
    )
    result = build_research_universe(
        candidates=[candidate],
        selection_cutoff=date(2026, 9, 1),
        capital_rupees=Decimal("1000"),
        thresholds=_thresholds(),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    )

    assert result.rejected_instrument_keys == ("NSE_EQ|INE000000003",)
    assert any(
        "point-in-time exchange evidence is incomplete" in item
        for item in result.audits[0].decision.violations
    )
