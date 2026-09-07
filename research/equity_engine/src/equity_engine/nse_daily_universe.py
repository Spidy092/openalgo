from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .historical_membership import HistoricalTradingStatus
from .nse_mii_security import NseMiiSecurityRow, NseMiiSecuritySnapshot
from .nse_semantics import (
    EffectiveDatedNseCmSemanticsPolicy,
    interpret_nse_mii_equity_row,
    tick_point_from_nse_mii_price_field,
)


@dataclass(frozen=True)
class NseDailyEquityUniverseRecord:
    report_date: date
    instrument_key: str
    isin: str
    symbol: str
    series: str
    name: str
    board_lot_quantity: int
    tick_size_rupees: Decimal
    raw_security_type_flag: str
    raw_permitted_to_trade: str
    raw_normal_market_status: str
    raw_normal_market_eligibility: str
    trading_status: HistoricalTradingStatus
    source_url: str
    snapshot_sha256: str
    source_row_number: int

    @property
    def eligible(self) -> bool:
        return self.trading_status.eligible


@dataclass(frozen=True)
class NseDailyEquityUniverse:
    report_date: date
    source_url: str
    snapshot_sha256: str
    semantics_source: str
    records: tuple[NseDailyEquityUniverseRecord, ...]

    @property
    def eligible_records(self) -> tuple[NseDailyEquityUniverseRecord, ...]:
        return tuple(record for record in self.records if record.eligible)

    @property
    def ineligible_records(self) -> tuple[NseDailyEquityUniverseRecord, ...]:
        return tuple(record for record in self.records if not record.eligible)


def _validate_equity_row(row: NseMiiSecurityRow) -> None:
    if not row.symbol:
        raise ValueError(f"equity row {row.source_row_number} has blank symbol")
    if row.isin is None:
        raise ValueError(
            f"equity row {row.source_row_number} has invalid/blank ISIN {row.raw_isin!r}"
        )
    if row.board_lot_quantity is None:
        raise ValueError(f"equity row {row.source_row_number} has blank board lot")
    if row.bid_interval_raw is None:
        raise ValueError(f"equity row {row.source_row_number} has blank BidIntrvl")


def materialize_nse_daily_equity_universe(
    *,
    snapshot: NseMiiSecuritySnapshot,
    semantics_policy: EffectiveDatedNseCmSemanticsPolicy,
) -> NseDailyEquityUniverse:
    """Materialize one point-in-time EQ universe from one dated NSE master snapshot.

    This function does not apply liquidity or ₹1,000 affordability filters. Those use only market
    data known by the later selection cutoff. Here we answer the narrower exchange-reference
    question: which configured normal-equity rows existed, what was their dated tick, and were
    they permitted/eligible/not-suspended according to the sourced semantics for this date?
    """

    semantics = semantics_policy.resolve(snapshot.report_date)
    records: list[NseDailyEquityUniverseRecord] = []
    seen_instruments: set[str] = set()

    for row in snapshot.rows:
        if row.series not in semantics.normal_equity_series:
            continue
        _validate_equity_row(row)
        status = interpret_nse_mii_equity_row(row, semantics=semantics)
        tick_point = tick_point_from_nse_mii_price_field(row, semantics=semantics)
        instrument_key = row.instrument_key
        if instrument_key in seen_instruments:
            raise ValueError(
                f"duplicate normal-equity instrument {instrument_key} in snapshot {snapshot.report_date}"
            )
        seen_instruments.add(instrument_key)
        records.append(
            NseDailyEquityUniverseRecord(
                report_date=snapshot.report_date,
                instrument_key=instrument_key,
                isin=row.isin,
                symbol=row.symbol,
                series=row.series,
                name=row.name,
                board_lot_quantity=row.board_lot_quantity,
                tick_size_rupees=tick_point.tick_size_rupees,
                raw_security_type_flag=row.security_type_flag,
                raw_permitted_to_trade=row.permitted_to_trade_raw,
                raw_normal_market_status=row.normal_market_status_raw,
                raw_normal_market_eligibility=row.normal_market_eligibility_raw,
                trading_status=status,
                source_url=row.source_url,
                snapshot_sha256=snapshot.payload_sha256,
                source_row_number=row.source_row_number,
            )
        )

    if not records:
        raise ValueError(
            f"NSE snapshot {snapshot.report_date} produced no configured normal-equity records"
        )
    records.sort(key=lambda item: (item.symbol, item.isin))
    return NseDailyEquityUniverse(
        report_date=snapshot.report_date,
        source_url=snapshot.source_url,
        snapshot_sha256=snapshot.payload_sha256,
        semantics_source=semantics.source,
        records=tuple(records),
    )
