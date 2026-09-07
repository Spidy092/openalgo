from datetime import date
from decimal import Decimal
import csv
import gzip
import io

import pytest

from equity_engine.historical_membership import HistoricalTradingEligibilityPolicy
from equity_engine.nse_mii_security import NseMiiEligibilitySemantics, NseMiiSecurityMasterParser
from equity_engine.nse_reference_materializer import materialize_nse_historical_reference
from equity_engine.tick_size import EffectiveDatedTickSizePolicy


_HEADER = [
    "FinInstrmId",
    "TckrSymb",
    "SctySrs",
    "FinInstrmNm",
    "ISIN",
    "NewBrdLotQty",
    "SctyTpFlg",
    "BidIntrvl",
    "CallAuctnInd",
    "PrtdToTrad",
    "SctyStsNrmlMkt",
    "ElgbltyNrmlMkt",
]


def _semantics() -> NseMiiEligibilitySemantics:
    return NseMiiEligibilitySemantics(
        normal_equity_series=frozenset({"EQ"}),
        permitted_to_trade_values=frozenset({"0", "1"}),
        normal_market_eligible_values=frozenset({"1"}),
        normal_market_tradeable_status_values=frozenset({"2"}),
        known_permitted_to_trade_values=frozenset({"0", "1", "2"}),
        known_normal_market_eligibility_values=frozenset({"0", "1"}),
        known_normal_market_status_values=frozenset({"1", "2", "3", "4", "5", "6"}),
        source="synthetic-test-semantics",
    )


def _snapshot(day: date, *, include_target: bool = True, status: str = "2", tick: str = "5"):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_HEADER, lineterminator="\n")
    writer.writeheader()
    if include_target:
        writer.writerow(
            {
                "FinInstrmId": "1594",
                "TckrSymb": "TEST",
                "SctySrs": "EQ",
                "FinInstrmNm": "TEST LIMITED",
                "ISIN": "INE001A01036",
                "NewBrdLotQty": "1",
                "SctyTpFlg": "0",
                "BidIntrvl": tick,
                "CallAuctnInd": "1",
                "PrtdToTrad": "0",
                "SctyStsNrmlMkt": status,
                "ElgbltyNrmlMkt": "1",
            }
        )
    else:
        writer.writerow(
            {
                "FinInstrmId": "9999",
                "TckrSymb": "OTHER",
                "SctySrs": "EQ",
                "FinInstrmNm": "OTHER LIMITED",
                "ISIN": "INE002A01034",
                "NewBrdLotQty": "1",
                "SctyTpFlg": "0",
                "BidIntrvl": "5",
                "CallAuctnInd": "1",
                "PrtdToTrad": "0",
                "SctyStsNrmlMkt": "2",
                "ElgbltyNrmlMkt": "1",
            }
        )
    payload = gzip.compress(output.getvalue().encode("utf-8"))
    return NseMiiSecurityMasterParser().parse_bytes(
        payload,
        filename=f"NSE_CM_security_{day.strftime('%d%m%Y')}.csv.gz",
    )


def test_existing_snapshot_with_absent_isin_is_explicitly_ineligible_not_missing() -> None:
    d1 = date(2026, 9, 1)
    d2 = date(2026, 9, 2)
    d3 = date(2026, 9, 3)
    evidence = materialize_nse_historical_reference(
        isin="INE001A01036",
        trading_dates=[d1, d2, d3],
        snapshots=[
            _snapshot(d1),
            _snapshot(d2, include_target=False),
            _snapshot(d3, tick="10"),
        ],
        semantics=_semantics(),
        bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
        scale_source="synthetic-test-scale",
    )

    assert evidence.complete is True
    assert evidence.membership.eligible_dates == (d1, d3)
    assert evidence.membership.ineligible_dates == (d2,)
    assert evidence.membership.missing_dates == ()
    assert [point.tick_size_rupees for point in evidence.tick_points] == [
        Decimal("0.05"),
        Decimal("0.10"),
    ]
    policy = HistoricalTradingEligibilityPolicy(evidence.membership)
    assert policy.is_eligible(d1) is True
    assert policy.is_eligible(d2) is False
    assert policy.is_eligible(d3) is True


def test_missing_daily_master_remains_missing_evidence_and_blocks_policy() -> None:
    d1 = date(2026, 9, 1)
    d2 = date(2026, 9, 2)
    evidence = materialize_nse_historical_reference(
        isin="INE001A01036",
        trading_dates=[d1, d2],
        snapshots=[_snapshot(d1)],
        semantics=_semantics(),
        bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
        scale_source="synthetic-test-scale",
    )

    assert evidence.complete is False
    assert evidence.membership.missing_dates == (d2,)
    with pytest.raises(ValueError, match="incomplete membership evidence"):
        HistoricalTradingEligibilityPolicy(evidence.membership)


def test_dated_tick_policy_uses_only_listed_day_observations() -> None:
    d1 = date(2026, 9, 1)
    d2 = date(2026, 9, 2)
    d3 = date(2026, 9, 3)
    evidence = materialize_nse_historical_reference(
        isin="INE001A01036",
        trading_dates=[d1, d2, d3],
        snapshots=[
            _snapshot(d1, include_target=False),
            _snapshot(d2, tick="5"),
            _snapshot(d3, tick="10"),
        ],
        semantics=_semantics(),
        bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
        scale_source="synthetic-test-scale",
    )
    policy = EffectiveDatedTickSizePolicy(evidence.tick_points)
    with pytest.raises(ValueError, match="no verified tick-size evidence"):
        policy.tick_size(d1)
    assert policy.tick_size(d2) == Decimal("0.05")
    assert policy.tick_size(d3) == Decimal("0.10")


def test_unknown_status_on_present_target_row_fails_materialization() -> None:
    d1 = date(2026, 9, 1)
    with pytest.raises(ValueError, match="unknown SctyStsNrmlMkt value"):
        materialize_nse_historical_reference(
            isin="INE001A01036",
            trading_dates=[d1],
            snapshots=[_snapshot(d1, status="99")],
            semantics=_semantics(),
            bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
            scale_source="synthetic-test-scale",
        )
