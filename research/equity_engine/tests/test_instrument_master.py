from datetime import date
from decimal import Decimal

import pytest

from equity_engine.instrument_master import build_nse_equity_master


def _row(key: str, *, instrument_type: str = "EQ", security_type: str = "NORMAL") -> dict[str, object]:
    return {
        "segment": "NSE_EQ",
        "name": "TEST LTD",
        "exchange": "NSE",
        "isin": key.split("|")[-1],
        "instrument_type": instrument_type,
        "instrument_key": key,
        "lot_size": 1,
        "freeze_quantity": 100000,
        "exchange_token": "123",
        "tick_size": 5.0,
        "trading_symbol": "TEST",
        "short_name": "Test",
        "security_type": security_type,
        "cas_eligible": True,
    }


def test_master_joins_bod_mis_and_suspended_by_instrument_key() -> None:
    normal = "NSE_EQ|INE000000001"
    suspended = "NSE_EQ|INE000000002"
    snapshot = build_nse_equity_master(
        as_of_date=date(2026, 9, 7),
        bod_rows=[_row(normal), _row(suspended)],
        mis_rows=[{"instrument_key": normal}, {"instrument_key": suspended}],
        suspended_rows=[{"instrument_key": suspended}],
        tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
    )

    by_key = snapshot.by_key()
    assert by_key[normal].mis_eligible is True
    assert by_key[normal].suspended is False
    assert by_key[suspended].suspended is True
    assert by_key[normal].tick_size_raw == Decimal("5.0")
    assert by_key[normal].tick_size_rupees == Decimal("0.050")
    assert snapshot.source_digest


def test_tick_size_conversion_scale_is_never_implicit() -> None:
    with pytest.raises(ValueError, match="tick-size scale"):
        build_nse_equity_master(
            as_of_date=date(2026, 9, 7),
            bod_rows=[_row("NSE_EQ|INE000000001")],
            mis_rows=[],
            suspended_rows=[],
            tick_size_scale_rupees_per_raw_unit=Decimal("0"),
        )


def test_missing_cas_metadata_fails_closed() -> None:
    row = _row("NSE_EQ|INE000000001")
    row.pop("cas_eligible")
    with pytest.raises(ValueError, match="missing cas_eligible"):
        build_nse_equity_master(
            as_of_date=date(2026, 9, 7),
            bod_rows=[row],
            mis_rows=[],
            suspended_rows=[],
            tick_size_scale_rupees_per_raw_unit=Decimal("0.01"),
        )
