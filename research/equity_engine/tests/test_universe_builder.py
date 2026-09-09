from datetime import date, datetime
from decimal import Decimal

import pandas as pd
import pytest

from equity_engine.documented_costs import CurrentTermsNSEIntradayCostProvider
from equity_engine.historical_membership import (
    HistoricalTradingStatus,
    assess_historical_membership,
)
from equity_engine.market_sessions import NSEEquitySessionPolicy
from equity_engine.models import ChargeBreakdown, CostQuote, CostSource, OrderSpec
from equity_engine.tick_size import FixedTickSizePolicy
from equity_engine.universe import CorporateActionAssessment, ResearchUniverseThresholds
from equity_engine.universe_builder import ResearchUniverseCandidateData, build_research_universe


class _ZeroChargeProvider:
    def quote(self, order: OrderSpec) -> CostQuote:
        return CostQuote(
            order=order,
            charges=ChargeBreakdown(
                brokerage=Decimal("0"),
                gst=Decimal("0"),
                stt=Decimal("0"),
                stamp_duty=Decimal("0"),
                transaction=Decimal("0"),
            ),
            source=CostSource.DOCUMENTED_SNAPSHOT,
            retrieved_at=datetime(2026, 9, 7),
            source_refs=("synthetic-zero-charge-test",),
        )


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


def _candidate(
    key: str,
    frame: pd.DataFrame,
    *,
    omit_membership_last: bool = False,
    cas_eligible: bool = False,
    minimum_tradable_quantity: int = 1,
):
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
        session_policy=NSEEquitySessionPolicy(cas_eligible=cas_eligible, exit_buffer_minutes=0),
        minimum_tradable_quantity=minimum_tradable_quantity,
        minimum_tradable_quantity_source="synthetic-NSE-board-lot",
    )


def _thresholds(
    *,
    min_observed_trading_days: int = 2,
    max_last_price_rupees: Decimal = Decimal("500"),
) -> ResearchUniverseThresholds:
    return ResearchUniverseThresholds(
        max_last_price_rupees=max_last_price_rupees,
        min_median_daily_notional_proxy_rupees=Decimal("1000000"),
        min_median_daily_volume_shares=Decimal("100000"),
        min_observed_trading_days=min_observed_trading_days,
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


def test_same_builder_path_scales_from_one_thousand_to_ten_thousand() -> None:
    frame = _frame(day1="2026-09-01", day2="2026-09-02").iloc[:2].copy()
    frame["open"] = 1001
    frame["high"] = 1002
    frame["low"] = 1000
    frame["close"] = 1001
    candidate = _candidate("NSE_EQ|INE000000004", frame)
    kwargs = {
        "candidates": [candidate],
        "selection_cutoff": date(2026, 9, 1),
        "thresholds": _thresholds(
            min_observed_trading_days=1,
            max_last_price_rupees=Decimal("5000"),
        ),
        "cost_provider": CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    }

    one_thousand = build_research_universe(capital_rupees=Decimal("1000.00"), **kwargs)
    ten_thousand = build_research_universe(capital_rupees=Decimal("10000.00"), **kwargs)

    assert one_thousand.rejected_instrument_keys == ("NSE_EQ|INE000000004",)
    assert ten_thousand.eligible_instrument_keys == ("NSE_EQ|INE000000004",)
    assert one_thousand.approved_capital_rupees == Decimal("1000.00")
    assert ten_thousand.approved_capital_rupees == Decimal("10000.00")
    assert ten_thousand.audits[0].minimum_tradable_quantity == 1
    assert ten_thousand.audits[0].minimum_tradable_quantity_source == "synthetic-NSE-board-lot"
    artifact = ten_thousand.as_prefilter_artifact()
    assert artifact["approved_capital_rupees"] == "10000.00"
    assert artifact["candidates"][0]["dataset_fingerprint"] == "fingerprint-NSE_EQ|INE000000004"
    assert artifact["candidates"][0]["cas_eligible"] is False
    assert artifact["candidates"][0]["liquidity_metric"] == (
        "median daily close-volume notional proxy"
    )
    assert artifact["candidates"][0]["liquidity_threshold_rupees"] == "1000000"
    assert artifact["candidates"][0]["minimum_observed_trading_days"] == 1
    assert artifact["live_orders_called"] is False


def test_minimum_tradable_quantity_is_respected_by_universe_affordability() -> None:
    candidate = _candidate(
        "NSE_EQ|INE000000005",
        _frame(day1="2026-09-01", day2="2026-09-02")
        .iloc[:2]
        .assign(
            open=100,
            high=101,
            low=99,
            close=100,
        ),
        minimum_tradable_quantity=10,
    )
    result = build_research_universe(
        candidates=[candidate],
        selection_cutoff=date(2026, 9, 1),
        capital_rupees=Decimal("2500"),
        thresholds=_thresholds(min_observed_trading_days=1),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    )

    assert result.audits[0].affordable_quantity % 10 == 0
    assert result.audits[0].required_capital_rupees == Decimal("1000")


@pytest.mark.parametrize(
    ("reference_price", "expected_eligible"),
    [
        (Decimal("999.99"), True),
        (Decimal("1000.00"), True),
        (Decimal("1000.01"), False),
    ],
)
def test_required_capital_boundary_is_explicit(
    reference_price: Decimal,
    expected_eligible: bool,
) -> None:
    frame = _frame(day1="2026-09-01", day2="2026-09-02").iloc[:2].copy()
    frame[["open", "high", "low", "close"]] = reference_price
    result = build_research_universe(
        candidates=[_candidate("NSE_EQ|INE000000010", frame)],
        selection_cutoff=date(2026, 9, 1),
        capital_rupees=Decimal("1000.00"),
        thresholds=_thresholds(min_observed_trading_days=1, max_last_price_rupees=Decimal("2000")),
        cost_provider=_ZeroChargeProvider(),
    )

    audit = result.audits[0]
    assert audit.required_capital_rupees == reference_price
    assert audit.decision.eligible is expected_eligible


def test_broker_balance_does_not_change_explicit_approved_capital_result() -> None:
    frame = _frame(day1="2026-09-01", day2="2026-09-02").iloc[:2].copy()
    kwargs = {
        "candidates": [_candidate("NSE_EQ|INE000000011", frame)],
        "selection_cutoff": date(2026, 9, 1),
        "capital_rupees": Decimal("1000.00"),
        "thresholds": _thresholds(min_observed_trading_days=1),
        "cost_provider": CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    }

    results = []
    for observed_broker_balance in (Decimal("1000.00"), Decimal("50000.00")):
        assert observed_broker_balance > 0
        results.append(build_research_universe(**kwargs))

    assert results[0] == results[1]
    assert results[0].approved_capital_rupees == Decimal("1000.00")


def test_future_intraday_price_cannot_change_earlier_reference_snapshot() -> None:
    candidate = _candidate(
        "NSE_EQ|INE000000006",
        _frame(day1="2026-09-01", day2="2026-09-02").iloc[:2],
    )
    as_of = pd.Timestamp("2026-09-01 10:00", tz="Asia/Kolkata")
    earlier = build_research_universe(
        candidates=[candidate],
        selection_cutoff=date(2026, 9, 1),
        selection_as_of=as_of,
        capital_rupees=Decimal("1000"),
        thresholds=_thresholds(min_observed_trading_days=1),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    )

    truncated = _candidate(
        "NSE_EQ|INE000000006",
        candidate.frame.loc[candidate.frame.index <= as_of],
    )
    same_data = build_research_universe(
        candidates=[truncated],
        selection_cutoff=date(2026, 9, 1),
        selection_as_of=as_of,
        capital_rupees=Decimal("1000"),
        thresholds=_thresholds(min_observed_trading_days=1),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    )

    assert earlier.audits[0].last_price_rupees == Decimal("100")
    assert earlier.audits[0] == same_data.audits[0]


def test_cas_auxiliary_price_and_volume_are_excluded_from_universe_evidence() -> None:
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-08-03 09:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-08-03 15:10", tz="Asia/Kolkata"),
            pd.Timestamp("2026-08-03 15:15", tz="Asia/Kolkata"),
            pd.Timestamp("2026-08-03 15:20", tz="Asia/Kolkata"),
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [100, 100, 10_000, 10_000],
            "high": [101, 101, 10_001, 10_001],
            "low": [99, 99, 9_999, 9_999],
            "close": [100, 100, 10_000, 10_000],
            "volume": [100_000, 100_000, 1_000_000, 1_000_000],
        },
        index=timestamps,
    )
    result = build_research_universe(
        candidates=[
            _candidate(
                "NSE_EQ|INE000000007",
                frame,
                cas_eligible=True,
            )
        ],
        selection_cutoff=date(2026, 8, 3),
        capital_rupees=Decimal("1000"),
        thresholds=_thresholds(min_observed_trading_days=1),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    )

    audit = result.audits[0]
    assert audit.reference_price_timestamp == pd.Timestamp("2026-08-03 15:10", tz="Asia/Kolkata")
    assert audit.last_price_rupees == Decimal("100")
    assert audit.liquidity_lookback_days == 1
    assert audit.decision.eligible


@pytest.mark.parametrize("invalid_close", [float("nan"), 0, -1])
def test_invalid_reference_price_fails_closed(invalid_close: float | int) -> None:
    frame = _frame(day1="2026-09-01", day2="2026-09-02").iloc[:2].copy()
    frame["close"] = invalid_close
    result = build_research_universe(
        candidates=[_candidate("NSE_EQ|INE000000008", frame)],
        selection_cutoff=date(2026, 9, 1),
        capital_rupees=Decimal("1000"),
        thresholds=_thresholds(min_observed_trading_days=1),
        cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=date(2026, 9, 7)),
    )

    assert not result.audits[0].decision.eligible
    assert any("invalid historical OHLCV" in item for item in result.audits[0].decision.violations)


def test_invalid_minimum_tradable_quantity_fails_closed() -> None:
    with pytest.raises(ValueError, match="minimum_tradable_quantity"):
        _candidate(
            "NSE_EQ|INE000000009",
            _frame(),
            minimum_tradable_quantity=0,
        )
