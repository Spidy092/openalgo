from datetime import date
from decimal import Decimal

import pytest

from equity_engine.instrument_master import build_nse_equity_master
from equity_engine.tick_size import (
    EffectiveDatedTickSizePolicy,
    TickSizePoint,
    expected_nse_cm_tick_size_rupees,
    verify_instrument_tick_size,
)


def _instrument(raw_tick: str):
    key = "NSE_EQ|INE000000001"
    snapshot = build_nse_equity_master(
        as_of_date=date(2026, 9, 7),
        bod_rows=[
            {
                "segment": "NSE_EQ",
                "name": "TEST LTD",
                "exchange": "NSE",
                "isin": "INE000000001",
                "instrument_type": "EQ",
                "instrument_key": key,
                "lot_size": 1,
                "freeze_quantity": 100000,
                "exchange_token": "123",
                "tick_size": raw_tick,
                "trading_symbol": "TEST",
                "security_type": "NORMAL",
                "cas_eligible": False,
            }
        ],
        mis_rows=[{"instrument_key": key}],
        suspended_rows=[],
        tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
    )
    return snapshot.instruments[0]


def test_nse_tick_tiers_match_exchange_circular() -> None:
    d = date(2026, 9, 7)
    assert expected_nse_cm_tick_size_rupees(effective_trade_date=d, exchange_reference_price_rupees=Decimal("249.99")) == Decimal("0.01")
    assert expected_nse_cm_tick_size_rupees(effective_trade_date=d, exchange_reference_price_rupees=Decimal("250")) == Decimal("0.05")
    assert expected_nse_cm_tick_size_rupees(effective_trade_date=d, exchange_reference_price_rupees=Decimal("1000.01")) == Decimal("0.10")
    assert expected_nse_cm_tick_size_rupees(effective_trade_date=d, exchange_reference_price_rupees=Decimal("5000.01")) == Decimal("0.50")
    assert expected_nse_cm_tick_size_rupees(effective_trade_date=d, exchange_reference_price_rupees=Decimal("10000.01")) == Decimal("1.00")
    assert expected_nse_cm_tick_size_rupees(effective_trade_date=d, exchange_reference_price_rupees=Decimal("20000.01")) == Decimal("5.00")


def test_converted_upstox_tick_must_match_nse_reference_tier() -> None:
    instrument = _instrument("5")
    result = verify_instrument_tick_size(
        instrument=instrument,
        effective_trade_date=date(2026, 9, 7),
        exchange_reference_price_rupees=Decimal("700"),
    )
    assert result.passed is True
    assert result.observed_rupees == Decimal("0.05")

    mismatch = verify_instrument_tick_size(
        instrument=instrument,
        effective_trade_date=date(2026, 9, 7),
        exchange_reference_price_rupees=Decimal("1200"),
    )
    assert mismatch.passed is False
    assert mismatch.expected_rupees == Decimal("0.10")


def test_effective_dated_policy_uses_latest_verified_point_only() -> None:
    policy = EffectiveDatedTickSizePolicy(
        [
            TickSizePoint(
                effective_from=date(2025, 4, 15),
                tick_size_rupees=Decimal("0.05"),
                source="security-master-2025-04",
            ),
            TickSizePoint(
                effective_from=date(2026, 9, 1),
                tick_size_rupees=Decimal("0.10"),
                source="security-master-2026-09",
            ),
        ]
    )

    assert policy.tick_size(date(2025, 4, 15)) == Decimal("0.05")
    assert policy.tick_size(date(2026, 8, 31)) == Decimal("0.05")
    assert policy.tick_size(date(2026, 9, 1)) == Decimal("0.10")
    assert policy.tick_size(date(2026, 9, 7)) == Decimal("0.10")


def test_effective_dated_policy_fails_before_earliest_verified_point() -> None:
    policy = EffectiveDatedTickSizePolicy(
        [
            TickSizePoint(
                effective_from=date(2025, 4, 15),
                tick_size_rupees=Decimal("0.05"),
                source="security-master-2025-04",
            )
        ]
    )

    with pytest.raises(ValueError, match="no verified tick-size evidence"):
        policy.tick_size(date(2025, 4, 14))
