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


def _target_row(
    *,
    symbol: str = "TEST",
    isin: str = "INE001A01036",
    financial_instrument_id: str = "1594",
    permitted: str = "0",
    status: str = "2",
    eligibility: str = "1",
    tick: str = "5",
) -> dict[str, str]:
    return {
        "FinInstrmId": financial_instrument_id,
        "TckrSymb": symbol,
        "SctySrs": "EQ",
        "FinInstrmNm": f"{symbol} LIMITED",
        "ISIN": isin,
        "NewBrdLotQty": "1",
        "SctyTpFlg": "0",
        "BidIntrvl": tick,
        "CallAuctnInd": "1",
        "PrtdToTrad": permitted,
        "SctyStsNrmlMkt": status,
        "ElgbltyNrmlMkt": eligibility,
    }


def _snapshot_rows(day: date, rows: list[dict[str, str]]):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = gzip.compress(output.getvalue().encode("utf-8"))
    return NseMiiSecurityMasterParser().parse_bytes(
        payload,
        filename=f"NSE_CM_security_{day.strftime('%d%m%Y')}.csv.gz",
    )


def _snapshot(day: date, *, include_target: bool = True, status: str = "2", tick: str = "5"):
    rows = (
        [_target_row(status=status, tick=tick)]
        if include_target
        else [_target_row(symbol="OTHER", isin="INE002A01034")]
    )
    return _snapshot_rows(day, rows)


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


def test_reference_accepts_two_ineligible_duplicate_eq_rows_and_preserves_audit() -> None:
    day = date(2026, 9, 7)
    evidence = materialize_nse_historical_reference(
        isin="INE287A01015",
        trading_dates=[day],
        snapshots=[
            _snapshot_rows(
                day,
                [
                    _target_row(
                        symbol="BECREL",
                        isin="INE287A01015",
                        financial_instrument_id="19384",
                        permitted="1",
                        status="3",
                        eligibility="0",
                    ),
                    _target_row(
                        symbol="BESTCROMP",
                        isin="INE287A01015",
                        financial_instrument_id="410",
                        permitted="1",
                        status="1",
                        eligibility="0",
                    ),
                ],
            )
        ],
        semantics=_semantics(),
        bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
        scale_source="synthetic-test-scale",
    )

    assert evidence.membership.ineligible_dates == (day,)
    assert evidence.membership.eligible_dates == ()
    assert [row.symbol for row in evidence.rejected_duplicate_rows] == ["BESTCROMP"]
    assert evidence.statuses[0].instrument_key == "NSE_EQ|INE287A01015"


def test_reference_selects_unique_eligible_duplicate_row() -> None:
    day = date(2026, 9, 7)
    evidence = materialize_nse_historical_reference(
        isin="INE001A01036",
        trading_dates=[day],
        snapshots=[
            _snapshot_rows(
                day,
                [
                    _target_row(symbol="BLOCKED", eligibility="0"),
                    _target_row(symbol="OPEN", financial_instrument_id="410"),
                ],
            )
        ],
        semantics=_semantics(),
        bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
        scale_source="synthetic-test-scale",
    )

    assert evidence.membership.eligible_dates == (day,)
    assert evidence.rejected_duplicate_rows[0].symbol == "BLOCKED"


def test_reference_still_rejects_two_eligible_duplicate_rows() -> None:
    day = date(2026, 9, 7)
    with pytest.raises(ValueError, match="duplicate normal-equity instrument"):
        materialize_nse_historical_reference(
            isin="INE001A01036",
            trading_dates=[day],
            snapshots=[
                _snapshot_rows(
                    day,
                    [
                        _target_row(symbol="ONE"),
                        _target_row(symbol="TWO", financial_instrument_id="410"),
                    ],
                )
            ],
            semantics=_semantics(),
            bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
            scale_source="synthetic-test-scale",
        )
