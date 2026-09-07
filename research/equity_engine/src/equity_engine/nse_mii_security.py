from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import csv
import gzip
import hashlib
import io
import re
from typing import Iterable

from .historical_membership import HistoricalTradingStatus
from .tick_size import TickSizePoint


NSE_MII_WEBSITE_AVAILABLE_FROM = date(2024, 2, 5)
NSE_MII_SECURITY_SOURCE_CIRCULAR = (
    "https://nsearchives.nseindia.com/content/circulars/MSD60315.pdf"
)
NSE_MII_SECURITY_ARCHIVE_PATTERN = (
    "https://nsearchives.nseindia.com/content/cm/NSE_CM_security_{ddmmyyyy}.csv.gz"
)

# Fields used by this research engine. The parser is header-driven: these exact ISO tags must
# be present. Additional columns are preserved as an upstream schema extension and do not affect
# field lookup. Missing required columns fail closed rather than falling back to column numbers.
_REQUIRED_FIELDS = (
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
)

_FILENAME = re.compile(r"NSE_CM_security_(\d{8})\.csv\.gz\Z")
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]\Z")


@dataclass(frozen=True)
class NseMiiSecurityRow:
    report_date: date
    financial_instrument_id: str
    symbol: str
    series: str
    name: str
    isin: str
    board_lot_quantity: int
    security_type_flag: str
    bid_interval_raw: Decimal
    call_auction_indicator: str
    permitted_to_trade_raw: str
    normal_market_status_raw: str
    normal_market_eligibility_raw: str
    source_url: str
    source_row_number: int

    @property
    def instrument_key(self) -> str:
        """Stable internal NSE-equity identity keyed by ISIN, compatible with our Upstox join."""

        return f"NSE_EQ|{self.isin}"


@dataclass(frozen=True)
class NseMiiSecuritySnapshot:
    report_date: date
    source_url: str
    payload_sha256: str
    header: tuple[str, ...]
    rows: tuple[NseMiiSecurityRow, ...]


@dataclass(frozen=True)
class NseMiiEligibilitySemantics:
    """Explicit interpretation contract for raw NSE MII codes.

    There are deliberately no defaults. The experiment must populate these values from an
    effective-dated NSE specification/circular. Unknown raw values fail instead of being treated
    as eligible or ineligible by guesswork.
    """

    normal_equity_series: frozenset[str]
    permitted_to_trade_values: frozenset[str]
    normal_market_eligible_values: frozenset[str]
    normal_market_tradeable_status_values: frozenset[str]
    known_permitted_to_trade_values: frozenset[str]
    known_normal_market_eligibility_values: frozenset[str]
    known_normal_market_status_values: frozenset[str]
    source: str

    def __post_init__(self) -> None:
        named_sets = {
            "normal_equity_series": self.normal_equity_series,
            "permitted_to_trade_values": self.permitted_to_trade_values,
            "normal_market_eligible_values": self.normal_market_eligible_values,
            "normal_market_tradeable_status_values": self.normal_market_tradeable_status_values,
            "known_permitted_to_trade_values": self.known_permitted_to_trade_values,
            "known_normal_market_eligibility_values": self.known_normal_market_eligibility_values,
            "known_normal_market_status_values": self.known_normal_market_status_values,
        }
        for name, values in named_sets.items():
            if not values:
                raise ValueError(f"{name} cannot be empty")
        if not self.source.strip():
            raise ValueError("eligibility-semantics source is required")
        if not self.permitted_to_trade_values.issubset(self.known_permitted_to_trade_values):
            raise ValueError("permitted values must be contained in known permitted values")
        if not self.normal_market_eligible_values.issubset(
            self.known_normal_market_eligibility_values
        ):
            raise ValueError("eligible values must be contained in known eligibility values")
        if not self.normal_market_tradeable_status_values.issubset(
            self.known_normal_market_status_values
        ):
            raise ValueError("tradeable statuses must be contained in known status values")


class NseMiiSecurityMasterParser:
    """Strict parser for a dated NSE CM MII security-master gzip CSV.

    This class validates transport/file structure and parses raw fields. It intentionally does not
    decide what NSE status codes mean and does not assume the unit of `BidIntrvl`; those are
    separate, explicit, effective-dated research decisions.
    """

    def __init__(
        self,
        *,
        maximum_compressed_bytes: int = 32 * 1024 * 1024,
        maximum_uncompressed_bytes: int = 128 * 1024 * 1024,
        maximum_rows: int = 500_000,
    ) -> None:
        for name, value in (
            ("maximum_compressed_bytes", maximum_compressed_bytes),
            ("maximum_uncompressed_bytes", maximum_uncompressed_bytes),
            ("maximum_rows", maximum_rows),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.maximum_compressed_bytes = maximum_compressed_bytes
        self.maximum_uncompressed_bytes = maximum_uncompressed_bytes
        self.maximum_rows = maximum_rows

    @staticmethod
    def report_date_from_filename(filename: str) -> date:
        match = _FILENAME.fullmatch(filename)
        if match is None:
            raise ValueError("expected NSE_CM_security_DDMMYYYY.csv.gz")
        try:
            report_date = datetime.strptime(match.group(1), "%d%m%Y").date()
        except ValueError as exc:
            raise ValueError("NSE MII security filename contains an invalid date") from exc
        if report_date < NSE_MII_WEBSITE_AVAILABLE_FROM:
            raise ValueError(
                "NSE website MII security-master source is only verified from 2024-02-05"
            )
        return report_date

    @staticmethod
    def source_url(report_date: date) -> str:
        if report_date < NSE_MII_WEBSITE_AVAILABLE_FROM:
            raise ValueError(
                "NSE website MII security-master source is only verified from 2024-02-05"
            )
        return NSE_MII_SECURITY_ARCHIVE_PATTERN.format(
            ddmmyyyy=report_date.strftime("%d%m%Y")
        )

    def parse_bytes(self, payload: bytes, *, filename: str) -> NseMiiSecuritySnapshot:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("NSE MII security payload must be non-empty bytes")
        if len(payload) > self.maximum_compressed_bytes:
            raise ValueError("compressed NSE MII security payload exceeds configured limit")
        if not payload.startswith(b"\x1f\x8b"):
            raise ValueError("NSE MII security payload is not gzip")

        report_date = self.report_date_from_filename(filename)
        source_url = self.source_url(report_date)
        try:
            uncompressed = gzip.decompress(payload)
        except OSError as exc:
            raise ValueError("invalid NSE MII security gzip payload") from exc
        if len(uncompressed) > self.maximum_uncompressed_bytes:
            raise ValueError("uncompressed NSE MII security payload exceeds configured limit")
        try:
            text = uncompressed.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("NSE MII security payload is not strict UTF-8") from exc
        if "\x00" in text:
            raise ValueError("NSE MII security payload contains a NUL byte")

        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if reader.fieldnames is None:
            raise ValueError("NSE MII security payload has no CSV header")
        header = tuple(reader.fieldnames)
        if len(header) != len(set(header)):
            raise ValueError("NSE MII security header contains duplicate field names")
        missing = tuple(field for field in _REQUIRED_FIELDS if field not in header)
        if missing:
            raise ValueError(
                "NSE MII security header is missing required ISO tags: " + ", ".join(missing)
            )

        rows: list[NseMiiSecurityRow] = []
        seen_identity: set[tuple[str, str]] = set()
        try:
            for source_row_number, raw in enumerate(reader, start=2):
                if len(rows) >= self.maximum_rows:
                    raise ValueError("NSE MII security payload exceeds configured row limit")
                if None in raw:
                    raise ValueError(
                        f"row {source_row_number} contains more fields than the declared header"
                    )
                values = {field: (raw.get(field) or "").strip() for field in _REQUIRED_FIELDS}

                isin = values["ISIN"].upper()
                if _ISIN.fullmatch(isin) is None:
                    raise ValueError(f"row {source_row_number} has invalid/blank ISIN")
                series = values["SctySrs"].upper()
                symbol = values["TckrSymb"].upper()
                if not symbol or not series:
                    raise ValueError(f"row {source_row_number} has blank symbol/series")

                try:
                    board_lot = int(values["NewBrdLotQty"])
                except ValueError as exc:
                    raise ValueError(
                        f"row {source_row_number} has invalid NewBrdLotQty"
                    ) from exc
                if board_lot <= 0:
                    raise ValueError(f"row {source_row_number} has non-positive board lot")

                try:
                    bid_interval_raw = Decimal(values["BidIntrvl"])
                except InvalidOperation as exc:
                    raise ValueError(
                        f"row {source_row_number} has invalid BidIntrvl"
                    ) from exc
                if not bid_interval_raw.is_finite() or bid_interval_raw <= 0:
                    raise ValueError(f"row {source_row_number} has non-positive BidIntrvl")

                identity = (isin, series)
                if identity in seen_identity:
                    raise ValueError(
                        f"duplicate NSE MII security identity {isin}/{series} in one snapshot"
                    )
                seen_identity.add(identity)

                rows.append(
                    NseMiiSecurityRow(
                        report_date=report_date,
                        financial_instrument_id=values["FinInstrmId"],
                        symbol=symbol,
                        series=series,
                        name=values["FinInstrmNm"],
                        isin=isin,
                        board_lot_quantity=board_lot,
                        security_type_flag=values["SctyTpFlg"],
                        bid_interval_raw=bid_interval_raw,
                        call_auction_indicator=values["CallAuctnInd"],
                        permitted_to_trade_raw=values["PrtdToTrad"],
                        normal_market_status_raw=values["SctyStsNrmlMkt"],
                        normal_market_eligibility_raw=values["ElgbltyNrmlMkt"],
                        source_url=source_url,
                        source_row_number=source_row_number,
                    )
                )
        except csv.Error as exc:
            raise ValueError("invalid NSE MII security CSV structure") from exc

        if not rows:
            raise ValueError("NSE MII security payload contains no data rows")
        return NseMiiSecuritySnapshot(
            report_date=report_date,
            source_url=source_url,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            header=header,
            rows=tuple(rows),
        )


def to_historical_trading_status(
    row: NseMiiSecurityRow,
    *,
    semantics: NseMiiEligibilitySemantics,
) -> HistoricalTradingStatus:
    """Interpret one raw NSE row only under an explicit, sourced semantics contract."""

    if row.permitted_to_trade_raw not in semantics.known_permitted_to_trade_values:
        raise ValueError(
            f"unknown PrtdToTrad value {row.permitted_to_trade_raw!r} on {row.report_date}"
        )
    if row.normal_market_eligibility_raw not in semantics.known_normal_market_eligibility_values:
        raise ValueError(
            "unknown ElgbltyNrmlMkt value "
            f"{row.normal_market_eligibility_raw!r} on {row.report_date}"
        )
    if row.normal_market_status_raw not in semantics.known_normal_market_status_values:
        raise ValueError(
            "unknown SctyStsNrmlMkt value "
            f"{row.normal_market_status_raw!r} on {row.report_date}"
        )

    normal_equity = row.series in semantics.normal_equity_series
    listed_or_permitted = row.permitted_to_trade_raw in semantics.permitted_to_trade_values
    normal_market_eligible = (
        row.normal_market_eligibility_raw in semantics.normal_market_eligible_values
    )
    normal_status_tradeable = (
        row.normal_market_status_raw in semantics.normal_market_tradeable_status_values
    )
    return HistoricalTradingStatus(
        trade_date=row.report_date,
        instrument_key=row.instrument_key,
        listed_on_nse=listed_or_permitted,
        normal_equity=normal_equity,
        tradeable_in_normal_market=normal_market_eligible and normal_status_tradeable,
        source=f"{row.source_url} | semantics={semantics.source}",
    )


def to_tick_size_point(
    row: NseMiiSecurityRow,
    *,
    bid_interval_scale_rupees_per_raw_unit: Decimal,
    scale_source: str,
) -> TickSizePoint:
    """Convert `BidIntrvl` only with an explicit sourced scale; no implicit paise assumption."""

    if bid_interval_scale_rupees_per_raw_unit <= 0:
        raise ValueError("BidIntrvl scale must be positive")
    if not scale_source.strip():
        raise ValueError("BidIntrvl scale source is required")
    tick = row.bid_interval_raw * bid_interval_scale_rupees_per_raw_unit
    if tick <= 0:
        raise ValueError("resolved tick size must be positive")
    return TickSizePoint(
        effective_from=row.report_date,
        tick_size_rupees=tick,
        source=(
            f"{row.source_url} | BidIntrvl={row.bid_interval_raw} | "
            f"scale={bid_interval_scale_rupees_per_raw_unit} | {scale_source}"
        ),
    )


def rows_for_isin(
    snapshots: Iterable[NseMiiSecuritySnapshot], *, isin: str
) -> tuple[NseMiiSecurityRow, ...]:
    normalized = isin.strip().upper()
    if _ISIN.fullmatch(normalized) is None:
        raise ValueError("invalid ISIN")
    selected = [row for snapshot in snapshots for row in snapshot.rows if row.isin == normalized]
    return tuple(sorted(selected, key=lambda row: (row.report_date, row.series)))
