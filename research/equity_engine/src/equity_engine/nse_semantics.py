from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable

from .historical_membership import HistoricalTradingStatus
from .nse_mii_security import NseMiiSecurityRow
from .tick_size import TickSizePoint


NSE_MASTER_DATA_V15_EFFECTIVE_EVIDENCE_DATE = date(2024, 7, 1)
NSE_MASTER_DATA_V15_SOURCE = (
    "https://nsearchives.nseindia.com/web/sites/default/files/inline-files/"
    "NSE_MasterData_Technical_Specifications.pdf"
)


@dataclass(frozen=True)
class NseCmSemantics:
    """Effective-dated interpretation of the NSE CM raw master fields we consume.

    Listing, permission, per-market eligibility and market status are intentionally separate.
    A security can be listed but not permitted to trade. Unknown codes always fail closed.
    """

    effective_from: date
    effective_to: date | None
    normal_equity_series: frozenset[str]
    listed_on_nse_values: frozenset[str]
    permitted_to_trade_values: frozenset[str]
    known_permitted_to_trade_values: frozenset[str]
    normal_market_eligible_values: frozenset[str]
    known_normal_market_eligibility_values: frozenset[str]
    normal_market_tradeable_status_values: frozenset[str]
    known_normal_market_status_values: frozenset[str]
    cm_price_scale_rupees_per_raw_unit: Decimal
    source: str

    def __post_init__(self) -> None:
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot precede effective_from")
        for name, values in (
            ("normal_equity_series", self.normal_equity_series),
            ("listed_on_nse_values", self.listed_on_nse_values),
            ("permitted_to_trade_values", self.permitted_to_trade_values),
            ("known_permitted_to_trade_values", self.known_permitted_to_trade_values),
            ("normal_market_eligible_values", self.normal_market_eligible_values),
            ("known_normal_market_eligibility_values", self.known_normal_market_eligibility_values),
            ("normal_market_tradeable_status_values", self.normal_market_tradeable_status_values),
            ("known_normal_market_status_values", self.known_normal_market_status_values),
        ):
            if not values:
                raise ValueError(f"{name} cannot be empty")
        if not self.listed_on_nse_values.issubset(self.known_permitted_to_trade_values):
            raise ValueError("listed values must be contained in known permitted-to-trade codes")
        if not self.permitted_to_trade_values.issubset(self.known_permitted_to_trade_values):
            raise ValueError("permitted values must be contained in known permitted-to-trade codes")
        if not self.normal_market_eligible_values.issubset(
            self.known_normal_market_eligibility_values
        ):
            raise ValueError("eligible values must be contained in known eligibility codes")
        if not self.normal_market_tradeable_status_values.issubset(
            self.known_normal_market_status_values
        ):
            raise ValueError("tradeable statuses must be contained in known status codes")
        if self.cm_price_scale_rupees_per_raw_unit <= 0:
            raise ValueError("CM price scale must be positive")
        if not self.source.strip():
            raise ValueError("NSE semantics source is required")

    def applies_to(self, trade_date: date) -> bool:
        return self.effective_from <= trade_date and (
            self.effective_to is None or trade_date <= self.effective_to
        )


class EffectiveDatedNseCmSemanticsPolicy:
    """Resolve a non-overlapping sourced NSE CM semantics contract for each trade date."""

    def __init__(self, contracts: Iterable[NseCmSemantics]) -> None:
        ordered = tuple(sorted(contracts, key=lambda item: item.effective_from))
        if not ordered:
            raise ValueError("at least one NSE CM semantics contract is required")
        for index, contract in enumerate(ordered[:-1]):
            following = ordered[index + 1]
            if contract.effective_to is None:
                raise ValueError("open-ended semantics contract cannot precede another contract")
            if following.effective_from <= contract.effective_to:
                raise ValueError("NSE CM semantics contracts overlap")
        self._contracts = ordered

    @property
    def contracts(self) -> tuple[NseCmSemantics, ...]:
        return self._contracts

    def resolve(self, trade_date: date) -> NseCmSemantics:
        matches = tuple(contract for contract in self._contracts if contract.applies_to(trade_date))
        if len(matches) != 1:
            raise ValueError(f"no unique verified NSE CM semantics for trade date {trade_date}")
        return matches[0]


def nse_cm_master_data_v15_semantics() -> NseCmSemantics:
    """Primary NSE v1.5 CM interpretation, conservatively anchored at the document date.

    NSE v1.5 states:
    - CM price fields are in paise and divide by 100 for rupees.
    - PermittedToTrade 0 = listed but not permitted, 1 = permitted.
    - normal-market Eligibility 1 = allowed, 0 = not allowed.
    - Status 1 preopen, 2 open, 3 suspended, 4 preopen extended,
      5 stock-open-with-market, 6 price discovery.

    For a day-level tradability gate, every documented state except explicit suspension (3) can
    represent an active security state. Permission and Eligibility must still both pass. Actual
    OHLCV presence is validated separately by the market-data layer.
    """

    return NseCmSemantics(
        effective_from=NSE_MASTER_DATA_V15_EFFECTIVE_EVIDENCE_DATE,
        effective_to=None,
        normal_equity_series=frozenset({"EQ"}),
        listed_on_nse_values=frozenset({"0", "1"}),
        permitted_to_trade_values=frozenset({"1"}),
        known_permitted_to_trade_values=frozenset({"0", "1"}),
        normal_market_eligible_values=frozenset({"1"}),
        known_normal_market_eligibility_values=frozenset({"0", "1"}),
        normal_market_tradeable_status_values=frozenset({"1", "2", "4", "5", "6"}),
        known_normal_market_status_values=frozenset({"1", "2", "3", "4", "5", "6"}),
        cm_price_scale_rupees_per_raw_unit=Decimal("0.01"),
        source=NSE_MASTER_DATA_V15_SOURCE,
    )


def interpret_nse_mii_equity_row(
    row: NseMiiSecurityRow,
    *,
    semantics: NseCmSemantics,
) -> HistoricalTradingStatus:
    """Convert one MII row into day-level NSE membership/tradability under dated semantics."""

    if not semantics.applies_to(row.report_date):
        raise ValueError(
            f"NSE CM semantics dated {semantics.effective_from} do not apply to {row.report_date}"
        )
    if row.series not in semantics.normal_equity_series:
        raise ValueError(f"row series {row.series!r} is outside configured normal equity series")
    if row.isin is None:
        raise ValueError("equity row does not have a valid ISIN")
    if row.permitted_to_trade_raw not in semantics.known_permitted_to_trade_values:
        raise ValueError(
            f"unknown PrtdToTrad value {row.permitted_to_trade_raw!r} on {row.report_date}"
        )
    if row.normal_market_eligibility_raw not in semantics.known_normal_market_eligibility_values:
        raise ValueError(
            f"unknown ElgbltyNrmlMkt value {row.normal_market_eligibility_raw!r} "
            f"on {row.report_date}"
        )
    if row.normal_market_status_raw not in semantics.known_normal_market_status_values:
        raise ValueError(
            f"unknown SctyStsNrmlMkt value {row.normal_market_status_raw!r} on {row.report_date}"
        )

    listed = row.permitted_to_trade_raw in semantics.listed_on_nse_values
    permitted = row.permitted_to_trade_raw in semantics.permitted_to_trade_values
    market_eligible = row.normal_market_eligibility_raw in semantics.normal_market_eligible_values
    active_status = row.normal_market_status_raw in semantics.normal_market_tradeable_status_values
    return HistoricalTradingStatus(
        trade_date=row.report_date,
        instrument_key=row.instrument_key,
        listed_on_nse=listed,
        normal_equity=True,
        tradeable_in_normal_market=permitted and market_eligible and active_status,
        source=f"{row.source_url} | semantics={semantics.source}",
    )


def tick_point_from_nse_mii_price_field(
    row: NseMiiSecurityRow,
    *,
    semantics: NseCmSemantics,
) -> TickSizePoint:
    """Resolve MII BidIntrvl to rupees using the dated primary CM price-unit contract."""

    if not semantics.applies_to(row.report_date):
        raise ValueError(
            f"NSE CM price semantics dated {semantics.effective_from} do not apply to {row.report_date}"
        )
    if row.bid_interval_raw is None:
        raise ValueError("cannot resolve tick size from blank BidIntrvl")
    tick = row.bid_interval_raw * semantics.cm_price_scale_rupees_per_raw_unit
    if tick <= 0:
        raise ValueError("resolved tick size must be positive")
    return TickSizePoint(
        effective_from=row.report_date,
        tick_size_rupees=tick,
        source=(
            f"{row.source_url} | BidIntrvl={row.bid_interval_raw} | "
            f"CM price scale={semantics.cm_price_scale_rupees_per_raw_unit} | "
            f"{semantics.source}"
        ),
    )
