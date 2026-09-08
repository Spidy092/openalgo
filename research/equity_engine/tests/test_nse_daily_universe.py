from datetime import date
import csv
import gzip
import io

import pytest

from equity_engine.nse_daily_universe import materialize_nse_daily_equity_universe
from equity_engine.nse_mii_security import NseMiiSecurityMasterParser
from equity_engine.nse_semantics import (
    EffectiveDatedNseCmSemanticsPolicy,
    nse_cm_master_data_v15_semantics,
)


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


def _row(
    *,
    symbol: str,
    isin: str,
    series: str = "EQ",
    permitted: str = "1",
    status: str = "2",
    eligibility: str = "1",
    tick: str = "5",
) -> dict[str, str]:
    return {
        "FinInstrmId": symbol,
        "TckrSymb": symbol,
        "SctySrs": series,
        "FinInstrmNm": f"{symbol} LIMITED",
        "ISIN": isin,
        "NewBrdLotQty": "1",
        "SctyTpFlg": "0",
        "BidIntrvl": tick,
        "CallAuctnInd": "0",
        "PrtdToTrad": permitted,
        "SctyStsNrmlMkt": status,
        "ElgbltyNrmlMkt": eligibility,
    }


def _snapshot(rows: list[dict[str, str]], day: date = date(2024, 7, 1)):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return NseMiiSecurityMasterParser().parse_bytes(
        gzip.compress(output.getvalue().encode("utf-8")),
        filename=f"NSE_CM_security_{day.strftime('%d%m%Y')}.csv.gz",
    )


def _policy() -> EffectiveDatedNseCmSemanticsPolicy:
    return EffectiveDatedNseCmSemanticsPolicy([nse_cm_master_data_v15_semantics()])


def test_daily_universe_keeps_ineligible_rows_for_audit_and_filters_candidates() -> None:
    snapshot = _snapshot(
        [
            _row(symbol="OPEN", isin="INE001A01036", tick="5"),
            _row(symbol="SUSP", isin="INE002A01034", status="3", tick="10"),
            _row(symbol="NOPERM", isin="INE003A01032", permitted="0"),
            _row(symbol="BEONLY", isin="INE004A01030", series="BE"),
        ]
    )
    universe = materialize_nse_daily_equity_universe(
        snapshot=snapshot,
        semantics_policy=_policy(),
    )

    assert [record.symbol for record in universe.records] == ["NOPERM", "OPEN", "SUSP"]
    assert [record.symbol for record in universe.eligible_records] == ["OPEN"]
    assert {record.symbol for record in universe.ineligible_records} == {"NOPERM", "SUSP"}
    by_symbol = {record.symbol: record for record in universe.records}
    assert str(by_symbol["OPEN"].tick_size_rupees) == "0.05"
    assert str(by_symbol["SUSP"].tick_size_rupees) == "0.10"
    assert by_symbol["NOPERM"].trading_status.listed_on_nse is True
    assert by_symbol["NOPERM"].eligible is False


def test_daily_universe_skips_nse_dummy_placeholder_before_equity_validation() -> None:
    snapshot = _snapshot(
        [
            _row(symbol="011NSETEST", isin="DUMMYSAN005", permitted="0", status="6"),
            _row(symbol="AB10BKINAV", isin="DUMMY0000339", permitted="0", status="1"),
            _row(symbol="OPEN", isin="INE001A01036"),
        ]
    )

    universe = materialize_nse_daily_equity_universe(
        snapshot=snapshot,
        semantics_policy=_policy(),
    )

    assert [record.symbol for record in universe.records] == ["OPEN"]
    assert snapshot.rows[0].is_placeholder is True
    assert snapshot.rows[1].is_placeholder is True


def test_daily_universe_still_rejects_nonplaceholder_invalid_equity_isin() -> None:
    snapshot = _snapshot([_row(symbol="BROKEN", isin="NOT-AN-ISIN")])

    with pytest.raises(ValueError, match="invalid/blank ISIN"):
        materialize_nse_daily_equity_universe(
            snapshot=snapshot,
            semantics_policy=_policy(),
        )


def test_daily_universe_accepts_two_ineligible_duplicate_eq_rows_and_preserves_audit() -> None:
    snapshot = _snapshot(
        [
            _row(
                symbol="BECREL",
                isin="INE287A01015",
                status="3",
                eligibility="0",
            ),
            _row(
                symbol="BESTCROMP",
                isin="INE287A01015",
                status="1",
                eligibility="0",
            ),
        ],
        day=date(2026, 9, 7),
    )

    universe = materialize_nse_daily_equity_universe(
        snapshot=snapshot,
        semantics_policy=_policy(),
    )

    assert [record.symbol for record in universe.eligible_records] == []
    assert [record.symbol for record in universe.ineligible_records] == ["BECREL"]
    assert [row.symbol for row in universe.rejected_duplicate_rows] == ["BESTCROMP"]
    assert universe.records[0].instrument_key == "NSE_EQ|INE287A01015"


def test_daily_universe_selects_unique_eligible_duplicate_row() -> None:
    snapshot = _snapshot(
        [
            _row(symbol="BLOCKED", isin="INE001A01036", eligibility="0"),
            _row(symbol="OPEN", isin="INE001A01036"),
        ]
    )

    universe = materialize_nse_daily_equity_universe(
        snapshot=snapshot,
        semantics_policy=_policy(),
    )

    assert [record.symbol for record in universe.eligible_records] == ["OPEN"]
    assert [record.symbol for record in universe.records] == ["OPEN"]
    assert universe.records[0].instrument_key == f"NSE_EQ|{universe.records[0].isin}"
    assert [row.symbol for row in universe.rejected_duplicate_rows] == ["BLOCKED"]


def test_daily_universe_refuses_semantics_before_verified_boundary() -> None:
    snapshot = _snapshot(
        [_row(symbol="TEST", isin="INE001A01036")],
        day=date(2024, 6, 28),
    )
    with pytest.raises(ValueError, match="no unique verified NSE CM semantics"):
        materialize_nse_daily_equity_universe(
            snapshot=snapshot,
            semantics_policy=_policy(),
        )


def test_unknown_code_blocks_entire_daily_universe_instead_of_silently_dropping_stock() -> None:
    snapshot = _snapshot(
        [_row(symbol="TEST", isin="INE001A01036", status="9")]
    )
    with pytest.raises(ValueError, match="unknown SctyStsNrmlMkt value"):
        materialize_nse_daily_equity_universe(
            snapshot=snapshot,
            semantics_policy=_policy(),
        )


def test_daily_universe_still_rejects_two_eligible_duplicate_eq_rows() -> None:
    snapshot = _snapshot(
        [
            _row(symbol="ONE", isin="INE001A01036"),
            _row(symbol="TWO", isin="INE001A01036"),
        ]
    )
    with pytest.raises(ValueError, match="duplicate normal-equity instrument"):
        materialize_nse_daily_equity_universe(
            snapshot=snapshot,
            semantics_policy=_policy(),
        )
