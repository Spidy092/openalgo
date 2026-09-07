from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import hashlib
import json
from typing import Iterable, Mapping


UPSTOX_INSTRUMENTS_DOC = "https://upstox.com/developer/api-documentation/instruments/"


@dataclass(frozen=True)
class EquityInstrument:
    instrument_key: str
    isin: str
    trading_symbol: str
    name: str
    segment: str
    exchange: str
    instrument_type: str
    security_type: str | None
    lot_size: int
    freeze_quantity: Decimal
    tick_size_raw: Decimal
    tick_size_rupees: Decimal
    cas_eligible: bool
    mis_eligible: bool
    suspended: bool


@dataclass(frozen=True)
class InstrumentMasterSnapshot:
    as_of_date: date
    instruments: tuple[EquityInstrument, ...]
    source_digest: str
    tick_size_scale_rupees_per_raw_unit: Decimal

    def by_key(self) -> dict[str, EquityInstrument]:
        return {item.instrument_key: item for item in self.instruments}


def _required_text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"instrument row missing non-empty {field}")
    return value.strip()


def _as_decimal(row: Mapping[str, object], field: str) -> Decimal:
    value = row.get(field)
    if value is None:
        raise ValueError(f"instrument row missing {field}")
    try:
        result = Decimal(str(value))
    except Exception as exc:  # Decimal raises several subclasses depending on input
        raise ValueError(f"invalid numeric {field}: {value!r}") from exc
    return result


def _instrument_keys(rows: Iterable[Mapping[str, object]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        key = _required_text(row, "instrument_key")
        keys.add(key)
    return keys


def build_nse_equity_master(
    *,
    as_of_date: date,
    bod_rows: Iterable[Mapping[str, object]],
    mis_rows: Iterable[Mapping[str, object]],
    suspended_rows: Iterable[Mapping[str, object]],
    tick_size_scale_rupees_per_raw_unit: Decimal,
) -> InstrumentMasterSnapshot:
    """Build an auditable NSE-equity master from Upstox's official daily files.

    Upstox publishes BOD, MIS and Suspended files separately. The JSON documentation currently
    describes `tick_size` as a minimum price movement but does not explicitly state the unit of
    its numeric JSON representation. Therefore the conversion scale is mandatory caller input;
    there is deliberately no hidden `/100` conversion in this package.
    """

    if tick_size_scale_rupees_per_raw_unit <= 0:
        raise ValueError("tick-size scale must be positive and explicit")

    bod = list(bod_rows)
    mis = list(mis_rows)
    suspended = list(suspended_rows)
    mis_keys = _instrument_keys(mis)
    suspended_keys = _instrument_keys(suspended)

    seen: set[str] = set()
    instruments: list[EquityInstrument] = []
    for row in bod:
        if row.get("segment") != "NSE_EQ":
            continue

        key = _required_text(row, "instrument_key")
        if key in seen:
            raise ValueError(f"duplicate NSE_EQ instrument_key in BOD master: {key}")
        seen.add(key)

        lot_size_decimal = _as_decimal(row, "lot_size")
        if lot_size_decimal != lot_size_decimal.to_integral_value() or lot_size_decimal <= 0:
            raise ValueError(f"invalid lot_size for {key}: {lot_size_decimal}")

        raw_tick = _as_decimal(row, "tick_size")
        if raw_tick <= 0:
            raise ValueError(f"non-positive tick_size for {key}")

        freeze_quantity = _as_decimal(row, "freeze_quantity")
        if freeze_quantity <= 0:
            raise ValueError(f"non-positive freeze_quantity for {key}")

        security_type = row.get("security_type")
        if security_type is not None and not isinstance(security_type, str):
            raise ValueError(f"invalid security_type for {key}")

        cas_value = row.get("cas_eligible")
        if cas_value is None:
            # CAS metadata is money/execution relevant after 2026-08-03. Missing is not False.
            raise ValueError(f"missing cas_eligible for {key}")
        if not isinstance(cas_value, bool):
            raise ValueError(f"invalid cas_eligible for {key}")

        instruments.append(
            EquityInstrument(
                instrument_key=key,
                isin=_required_text(row, "isin"),
                trading_symbol=_required_text(row, "trading_symbol"),
                name=_required_text(row, "name"),
                segment="NSE_EQ",
                exchange=_required_text(row, "exchange"),
                instrument_type=_required_text(row, "instrument_type"),
                security_type=security_type,
                lot_size=int(lot_size_decimal),
                freeze_quantity=freeze_quantity,
                tick_size_raw=raw_tick,
                tick_size_rupees=raw_tick * tick_size_scale_rupees_per_raw_unit,
                cas_eligible=cas_value,
                mis_eligible=key in mis_keys,
                suspended=key in suspended_keys,
            )
        )

    if not instruments:
        raise ValueError("BOD master contained no NSE_EQ instruments")

    canonical = {
        "as_of_date": as_of_date.isoformat(),
        "tick_size_scale": str(tick_size_scale_rupees_per_raw_unit),
        "bod": bod,
        "mis": mis,
        "suspended": suspended,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return InstrumentMasterSnapshot(
        as_of_date=as_of_date,
        instruments=tuple(sorted(instruments, key=lambda item: item.instrument_key)),
        source_digest=digest,
        tick_size_scale_rupees_per_raw_unit=tick_size_scale_rupees_per_raw_unit,
    )
