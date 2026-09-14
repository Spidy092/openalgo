from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import re
from typing import Iterable

from .historical_membership import (
    HistoricalMembershipAssessment,
    HistoricalTradingStatus,
    assess_historical_membership,
)
from .nse_mii_security import (
    NseMiiEligibilitySemantics,
    NseMiiSecurityRow,
    NseMiiSecuritySnapshot,
    resolve_nse_equity_rows,
    to_historical_trading_status,
    to_tick_size_point,
)
from .tick_size import TickSizePoint

_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]\Z")


@dataclass(frozen=True)
class NseHistoricalReferenceEvidence:
    """Point-in-time NSE reference evidence for one equity identity across requested dates."""

    instrument_key: str
    isin: str
    requested_dates: tuple[date, ...]
    membership: HistoricalMembershipAssessment
    statuses: tuple[HistoricalTradingStatus, ...]
    tick_points: tuple[TickSizePoint, ...]
    snapshot_hashes: tuple[tuple[date, str], ...]
    rejected_duplicate_rows: tuple[NseMiiSecurityRow, ...] = ()

    @property
    def complete(self) -> bool:
        return self.membership.complete


def _matching_equity_rows(
    snapshot: NseMiiSecuritySnapshot,
    *,
    isin: str,
    semantics: NseMiiEligibilitySemantics,
) -> tuple[NseMiiSecurityRow, ...]:
    return tuple(
        row
        for row in snapshot.rows
        if (
            row.isin == isin
            and row.series in semantics.normal_equity_series
            and not row.is_placeholder
        )
    )


def materialize_nse_historical_reference(
    *,
    isin: str,
    trading_dates: Iterable[date],
    snapshots: Iterable[NseMiiSecuritySnapshot],
    semantics: NseMiiEligibilitySemantics,
    bid_interval_scale_rupees_per_raw_unit: Decimal,
    scale_source: str,
) -> NseHistoricalReferenceEvidence:
    """Build daily membership and tick evidence without using today's broker universe.

    Missing *snapshot dates* remain missing evidence. By contrast, when a complete dated snapshot
    exists but the requested ISIN/normal-equity series is absent, that date is explicitly recorded
    as not listed/eligible. This distinction prevents an unavailable file from being mistaken for
    a delisting.
    """

    normalized_isin = isin.strip().upper()
    if _ISIN.fullmatch(normalized_isin) is None:
        raise ValueError("invalid ISIN")
    requested = tuple(sorted(set(trading_dates)))
    if not requested:
        raise ValueError("at least one trading date is required")
    if bid_interval_scale_rupees_per_raw_unit <= 0:
        raise ValueError("BidIntrvl scale must be positive")
    if not scale_source.strip():
        raise ValueError("BidIntrvl scale source is required")

    by_date: dict[date, NseMiiSecuritySnapshot] = {}
    for snapshot in snapshots:
        if snapshot.report_date in by_date:
            raise ValueError(f"duplicate NSE MII snapshot for {snapshot.report_date}")
        by_date[snapshot.report_date] = snapshot

    instrument_key = f"NSE_EQ|{normalized_isin}"
    statuses: list[HistoricalTradingStatus] = []
    tick_points: list[TickSizePoint] = []
    hashes: list[tuple[date, str]] = []
    rejected_duplicate_rows: list[NseMiiSecurityRow] = []

    for trade_date in requested:
        snapshot = by_date.get(trade_date)
        if snapshot is None:
            continue
        hashes.append((trade_date, snapshot.payload_sha256))
        matches = _matching_equity_rows(
            snapshot,
            isin=normalized_isin,
            semantics=semantics,
        )
        if not matches:
            statuses.append(
                HistoricalTradingStatus(
                    trade_date=trade_date,
                    instrument_key=instrument_key,
                    listed_on_nse=False,
                    normal_equity=True,
                    tradeable_in_normal_market=False,
                    source=(
                        f"{snapshot.source_url} | ISIN absent from configured normal-equity series | "
                        f"semantics={semantics.source}"
                    ),
                )
            )
            continue

        for row in matches:
            if row.isin is None or row.board_lot_quantity is None or row.bid_interval_raw is None:
                raise ValueError(
                    f"in-scope NSE equity row is structurally incomplete on {trade_date} "
                    f"(source row {row.source_row_number})"
                )

        resolution = resolve_nse_equity_rows(
            matches,
            status_resolver=lambda row: to_historical_trading_status(row, semantics=semantics),
        )
        rejected_duplicate_rows.extend(resolution.rejected_duplicate_rows)
        row = resolution.selected_row
        statuses.append(resolution.selected_status)
        tick_points.append(
            to_tick_size_point(
                row,
                bid_interval_scale_rupees_per_raw_unit=bid_interval_scale_rupees_per_raw_unit,
                scale_source=scale_source,
            )
        )

    membership = assess_historical_membership(
        instrument_key=instrument_key,
        trading_dates=requested,
        statuses=statuses,
    )
    return NseHistoricalReferenceEvidence(
        instrument_key=instrument_key,
        isin=normalized_isin,
        requested_dates=requested,
        membership=membership,
        statuses=tuple(statuses),
        tick_points=tuple(tick_points),
        snapshot_hashes=tuple(hashes),
        rejected_duplicate_rows=tuple(rejected_duplicate_rows),
    )
