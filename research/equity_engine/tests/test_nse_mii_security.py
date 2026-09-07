from datetime import date
from decimal import Decimal
import csv
import gzip
import io

import pytest

from equity_engine.nse_mii_security import (
    NSE_MII_WEBSITE_AVAILABLE_FROM,
    NseMiiEligibilitySemantics,
    NseMiiSecurityMasterParser,
    equity_candidate_rows,
    to_historical_trading_status,
    to_tick_size_point,
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
    "ExtraFutureField",
]


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "FinInstrmId": "1594",
        "TckrSymb": "TEST",
        "SctySrs": "EQ",
        "FinInstrmNm": "TEST LIMITED",
        "ISIN": "INE001A01036",
        "NewBrdLotQty": "1",
        "SctyTpFlg": "0",
        "BidIntrvl": "5",
        "CallAuctnInd": "1",
        "PrtdToTrad": "0",
        "SctyStsNrmlMkt": "2",
        "ElgbltyNrmlMkt": "1",
        "ExtraFutureField": "ignored-but-preserved-in-header",
    }
    row.update(overrides)
    return row


def _gzip_csv(rows: list[dict[str, str]], *, header: list[str] | None = None) -> bytes:
    output = io.StringIO(newline="")
    fieldnames = header or _HEADER
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return gzip.compress(output.getvalue().encode("utf-8"))


def _semantics() -> NseMiiEligibilitySemantics:
    # Synthetic contract for unit tests only. Production values must be effective-dated and
    # sourced from NSE documentation; the class deliberately provides no defaults.
    return NseMiiEligibilitySemantics(
        normal_equity_series=frozenset({"EQ"}),
        permitted_to_trade_values=frozenset({"0", "1"}),
        normal_market_eligible_values=frozenset({"1"}),
        normal_market_tradeable_status_values=frozenset({"1", "2", "4", "5"}),
        known_permitted_to_trade_values=frozenset({"0", "1"}),
        known_normal_market_eligibility_values=frozenset({"0", "1"}),
        known_normal_market_status_values=frozenset({"1", "2", "3", "4", "5"}),
        source="synthetic-test-semantics",
    )


def test_parser_reads_iso_tags_by_header_and_keeps_raw_market_codes() -> None:
    payload = _gzip_csv([_row()])
    snapshot = NseMiiSecurityMasterParser().parse_bytes(
        payload,
        filename="NSE_CM_security_07092026.csv.gz",
    )

    assert snapshot.report_date == date(2026, 9, 7)
    assert snapshot.source_url.endswith("NSE_CM_security_07092026.csv.gz")
    assert len(snapshot.payload_sha256) == 64
    assert "ExtraFutureField" in snapshot.header
    parsed = snapshot.rows[0]
    assert parsed.instrument_key == "NSE_EQ|INE001A01036"
    assert parsed.bid_interval_raw == Decimal("5")
    assert parsed.normal_market_status_raw == "2"
    assert parsed.normal_market_eligibility_raw == "1"


def test_missing_required_iso_tag_fails_closed() -> None:
    header = [name for name in _HEADER if name != "ElgbltyNrmlMkt"]
    payload = _gzip_csv([_row()], header=header)
    with pytest.raises(ValueError, match="missing required ISO tags: ElgbltyNrmlMkt"):
        NseMiiSecurityMasterParser().parse_bytes(
            payload,
            filename="NSE_CM_security_07092026.csv.gz",
        )


def test_verified_website_source_does_not_claim_pre_2024_history() -> None:
    assert NSE_MII_WEBSITE_AVAILABLE_FROM == date(2024, 2, 5)
    with pytest.raises(ValueError, match="only verified from 2024-02-05"):
        NseMiiSecurityMasterParser.report_date_from_filename(
            "NSE_CM_security_04022024.csv.gz"
        )


def test_non_equity_row_can_be_audited_without_valid_equity_identity() -> None:
    payload = _gzip_csv(
        [
            _row(
                TckrSymb="OTHER",
                SctySrs="BE",
                ISIN="",
                NewBrdLotQty="",
                BidIntrvl="",
            ),
            _row(),
        ]
    )
    snapshot = NseMiiSecurityMasterParser().parse_bytes(
        payload, filename="NSE_CM_security_07092026.csv.gz"
    )
    assert len(snapshot.rows) == 2
    assert snapshot.rows[0].isin is None
    assert snapshot.rows[0].board_lot_quantity is None
    assert snapshot.rows[0].bid_interval_raw is None

    candidates = equity_candidate_rows(snapshot, semantics=_semantics())
    assert len(candidates) == 1
    assert candidates[0].symbol == "TEST"


def test_in_scope_equity_with_bad_identity_fails_at_promotion_boundary() -> None:
    snapshot = NseMiiSecurityMasterParser().parse_bytes(
        _gzip_csv([_row(ISIN="")]),
        filename="NSE_CM_security_07092026.csv.gz",
    )
    with pytest.raises(ValueError, match="invalid/blank ISIN"):
        equity_candidate_rows(snapshot, semantics=_semantics())


def test_malformed_nonblank_bid_interval_is_source_corruption() -> None:
    with pytest.raises(ValueError, match="invalid BidIntrvl"):
        NseMiiSecurityMasterParser().parse_bytes(
            _gzip_csv([_row(SctySrs="BE", BidIntrvl="not-a-number")]),
            filename="NSE_CM_security_07092026.csv.gz",
        )


def test_eligibility_requires_explicit_semantics_and_rejects_suspended_status() -> None:
    parser = NseMiiSecurityMasterParser()
    normal = parser.parse_bytes(
        _gzip_csv([_row()]), filename="NSE_CM_security_07092026.csv.gz"
    ).rows[0]
    status = to_historical_trading_status(normal, semantics=_semantics())
    assert status.eligible is True

    suspended = parser.parse_bytes(
        _gzip_csv([_row(SctyStsNrmlMkt="3")]),
        filename="NSE_CM_security_07092026.csv.gz",
    ).rows[0]
    suspended_status = to_historical_trading_status(suspended, semantics=_semantics())
    assert suspended_status.tradeable_in_normal_market is False
    assert suspended_status.eligible is False


def test_unknown_market_code_fails_instead_of_becoming_false_or_true() -> None:
    row = NseMiiSecurityMasterParser().parse_bytes(
        _gzip_csv([_row(SctyStsNrmlMkt="99")]),
        filename="NSE_CM_security_07092026.csv.gz",
    ).rows[0]
    with pytest.raises(ValueError, match="unknown SctyStsNrmlMkt value"):
        to_historical_trading_status(row, semantics=_semantics())


def test_bid_interval_conversion_requires_explicit_sourced_scale() -> None:
    row = NseMiiSecurityMasterParser().parse_bytes(
        _gzip_csv([_row(BidIntrvl="5")]),
        filename="NSE_CM_security_07092026.csv.gz",
    ).rows[0]

    point = to_tick_size_point(
        row,
        bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
        scale_source="synthetic-test-scale",
    )
    assert point.tick_size_rupees == Decimal("0.05")
    assert point.effective_from == date(2026, 9, 7)

    with pytest.raises(ValueError, match="scale source is required"):
        to_tick_size_point(
            row,
            bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
            scale_source="",
        )


def test_blank_bid_interval_cannot_be_promoted_to_tick_evidence() -> None:
    row = NseMiiSecurityMasterParser().parse_bytes(
        _gzip_csv([_row(SctySrs="BE", BidIntrvl="")]),
        filename="NSE_CM_security_07092026.csv.gz",
    ).rows[0]
    with pytest.raises(ValueError, match="blank BidIntrvl"):
        to_tick_size_point(
            row,
            bid_interval_scale_rupees_per_raw_unit=Decimal("0.01"),
            scale_source="synthetic-test-scale",
        )
