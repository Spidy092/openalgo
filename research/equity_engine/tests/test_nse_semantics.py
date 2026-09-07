from datetime import date
from decimal import Decimal

import pytest

from equity_engine.nse_mii_security import NseMiiSecurityRow
from equity_engine.nse_semantics import (
    EffectiveDatedNseCmSemanticsPolicy,
    NSE_MASTER_DATA_V15_EFFECTIVE_EVIDENCE_DATE,
    interpret_nse_mii_equity_row,
    nse_cm_master_data_v15_semantics,
    tick_point_from_nse_mii_price_field,
)


def _row(*, permitted: str = "1", eligibility: str = "1", status: str = "2") -> NseMiiSecurityRow:
    return NseMiiSecurityRow(
        report_date=date(2024, 7, 1),
        financial_instrument_id="1594",
        symbol="TEST",
        series="EQ",
        name="TEST LIMITED",
        raw_isin="INE001A01036",
        isin="INE001A01036",
        board_lot_quantity=1,
        security_type_flag="0",
        bid_interval_raw=Decimal("5"),
        call_auction_indicator="0",
        permitted_to_trade_raw=permitted,
        normal_market_status_raw=status,
        normal_market_eligibility_raw=eligibility,
        source_url="https://nsearchives.nseindia.com/content/cm/NSE_CM_security_01072024.csv.gz",
        source_row_number=2,
    )


def test_primary_v15_contract_starts_on_documented_evidence_date() -> None:
    semantics = nse_cm_master_data_v15_semantics()
    assert NSE_MASTER_DATA_V15_EFFECTIVE_EVIDENCE_DATE == date(2024, 7, 1)
    assert semantics.cm_price_scale_rupees_per_raw_unit == Decimal("0.01")
    assert semantics.known_permitted_to_trade_values == frozenset({"0", "1"})
    assert semantics.permitted_to_trade_values == frozenset({"1"})
    assert semantics.known_normal_market_status_values == frozenset({"1", "2", "3", "4", "5", "6"})
    assert "3" not in semantics.normal_market_tradeable_status_values


def test_listed_but_not_permitted_is_listed_and_not_tradeable() -> None:
    status = interpret_nse_mii_equity_row(
        _row(permitted="0"), semantics=nse_cm_master_data_v15_semantics()
    )
    assert status.listed_on_nse is True
    assert status.tradeable_in_normal_market is False
    assert status.eligible is False


def test_permission_eligibility_and_non_suspended_status_are_all_required() -> None:
    semantics = nse_cm_master_data_v15_semantics()
    for active_status in ("1", "2", "4", "5", "6"):
        result = interpret_nse_mii_equity_row(
            _row(status=active_status), semantics=semantics
        )
        assert result.eligible is True

    suspended = interpret_nse_mii_equity_row(_row(status="3"), semantics=semantics)
    assert suspended.eligible is False

    not_market_eligible = interpret_nse_mii_equity_row(
        _row(eligibility="0"), semantics=semantics
    )
    assert not_market_eligible.eligible is False


def test_unknown_or_future_permitted_code_fails_under_v15_contract() -> None:
    with pytest.raises(ValueError, match="unknown PrtdToTrad value"):
        interpret_nse_mii_equity_row(
            _row(permitted="2"), semantics=nse_cm_master_data_v15_semantics()
        )


def test_v15_contract_is_not_projected_before_july_2024() -> None:
    row = _row()
    row = NseMiiSecurityRow(**{**row.__dict__, "report_date": date(2024, 6, 28)})
    with pytest.raises(ValueError, match="do not apply"):
        interpret_nse_mii_equity_row(row, semantics=nse_cm_master_data_v15_semantics())

    policy = EffectiveDatedNseCmSemanticsPolicy([nse_cm_master_data_v15_semantics()])
    with pytest.raises(ValueError, match="no unique verified NSE CM semantics"):
        policy.resolve(date(2024, 6, 28))
    assert policy.resolve(date(2024, 7, 1)).effective_from == date(2024, 7, 1)


def test_bid_interval_uses_primary_cm_paise_scale() -> None:
    point = tick_point_from_nse_mii_price_field(
        _row(), semantics=nse_cm_master_data_v15_semantics()
    )
    assert point.tick_size_rupees == Decimal("0.05")
