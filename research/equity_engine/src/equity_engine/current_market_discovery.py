from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

from .costs import CostProvider
from .instrument_master import build_nse_equity_master
from .models import Exchange, OrderSpec, Product, Side
from .sizing import max_affordable_buy_quantity
from .suspension_identity import (
    AMBIGUOUS_EXACT,
    NO_SUSPENSION_RECORD,
    SUSPENDED_EXACT,
    SUSPENSION_CONFLICT,
    SUSPENSION_IDENTITY_POLICY,
    field_text,
    resolve_suspension,
    row_evidence,
)
from .upstox_batch_history import historical_request_limit_days
from .upstox_instruments import InstrumentFilePayload
from .upstox_market_context import (
    MAX_INSTRUMENTS_PER_REQUEST,
    UPSTOX_FULL_QUOTE_V3_DOC,
    UPSTOX_FULL_QUOTE_V3_URL,
    QuoteBatchResult,
)


CURRENT_MARKET_SCHEMA_VERSION = "current-market-calibration-v1"
DEFAULT_TICK_SIZE_SCALE_RUPEES_PER_RAW_UNIT = Decimal("0.01")
DEFAULT_ESTIMATED_BYTES_PER_ROW = 256
APPROVED_CAPITALS = (Decimal("1000.00"), Decimal("10000.00"))
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]\Z")
SUSPENSION_MATCH_POLICY = SUSPENSION_IDENTITY_POLICY


@dataclass(frozen=True)
class CapitalAffordabilityMeasurement:
    capital: Decimal
    raw_minimum_capital: Decimal
    minimum_entry_charges: Decimal
    minimum_entry_cash_required: Decimal
    max_affordable_quantity: int
    max_affordable_notional: Decimal
    max_affordable_entry_charges: Decimal
    cash_required: Decimal
    cash_remaining: Decimal
    affordable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "capital": str(self.capital),
            "raw_minimum_capital": str(self.raw_minimum_capital),
            "minimum_entry_charges": str(self.minimum_entry_charges),
            "minimum_entry_cash_required": str(self.minimum_entry_cash_required),
            "max_affordable_quantity": self.max_affordable_quantity,
            "max_affordable_notional": str(self.max_affordable_notional),
            "max_affordable_entry_charges": str(self.max_affordable_entry_charges),
            "cash_required": str(self.cash_required),
            "cash_remaining": str(self.cash_remaining),
            "affordable": self.affordable,
        }


@dataclass(frozen=True)
class CurrentInstrumentMeasurement:
    instrument_key: str
    symbol: str
    isin: str
    series: str
    exchange: str
    instrument_type: str
    security_type: str
    tick_size_raw: Decimal | None
    tick_size_rupees: Decimal | None
    minimum_tradable_quantity: int | None
    minimum_tradable_quantity_source: str | None
    mis_eligible: bool
    suspended: bool
    current_exchange_token: str | None
    suspension_status: str
    suspension_match_count: int
    suspension_ambiguous: bool
    suspension_variant_row_count: int
    same_key_variant_count: int
    exact_token_match_count: int
    suspension_evidence: tuple[dict[str, object], ...]
    research_candidate: bool
    live_tradability_proven: bool
    cas_eligible: bool | None
    quote_status: str
    quote_failure_reason: str | None
    reference_price: Decimal | None
    quote_timestamp: str | None
    price_source: str | None
    candidate: bool
    candidate_after_max_last_price: bool
    exclusion_reasons: tuple[str, ...]
    capital_measurements: tuple[CapitalAffordabilityMeasurement, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument_key": self.instrument_key,
            "symbol": self.symbol,
            "isin": self.isin,
            "series": self.series,
            "exchange": self.exchange,
            "instrument_type": self.instrument_type,
            "security_type": self.security_type,
            "tick_size_raw": _decimal_or_none(self.tick_size_raw),
            "tick_size_rupees": _decimal_or_none(self.tick_size_rupees),
            "minimum_tradable_quantity": self.minimum_tradable_quantity,
            "minimum_tradable_quantity_source": self.minimum_tradable_quantity_source,
            "mis_eligible": self.mis_eligible,
            "suspended": self.suspended,
            "current_exchange_token": self.current_exchange_token,
            "suspension_status": self.suspension_status,
            "suspension_match_count": self.suspension_match_count,
            "suspension_ambiguous": self.suspension_ambiguous,
            "suspension_variant_row_count": self.suspension_variant_row_count,
            "same_key_variant_count": self.same_key_variant_count,
            "exact_token_match_count": self.exact_token_match_count,
            "suspension_evidence": list(self.suspension_evidence),
            "cas_eligible": self.cas_eligible,
            "quote_status": self.quote_status,
            "quote_failure_reason": self.quote_failure_reason,
            "reference_price": _decimal_or_none(self.reference_price),
            "quote_timestamp": self.quote_timestamp,
            "price_source": self.price_source,
            "candidate": self.candidate,
            "research_candidate": self.research_candidate,
            "live_tradability_proven": self.live_tradability_proven,
            "candidate_after_max_last_price": self.candidate_after_max_last_price,
            "exclusion_reasons": list(self.exclusion_reasons),
            "capital_measurements": [item.to_dict() for item in self.capital_measurements],
        }


@dataclass(frozen=True)
class CurrentMarketCalibrationArtifact:
    snapshot_as_of: datetime
    source_files: tuple[dict[str, object], ...]
    suspended_source_hash: str
    suspended_row_count: int
    suspended_unique_key_count: int
    suspended_duplicate_key_count: int
    suspended_duplicate_isin_count: int
    suspension_match_policy: str
    suspension_match_count: int
    suspension_ambiguous_count: int
    suspension_key_variant_count: int
    exact_suspended_count: int
    conflict_count: int
    no_record_count: int
    ambiguous_exact_count: int
    gate_counts: dict[str, int]
    quote_request_count: int
    quote_success_count: int
    quote_failure_count: int
    quote_http_request_count: int
    quote_source: dict[str, object]
    cas_distribution: dict[str, int]
    minimum_quantity_distribution: dict[str, object]
    price_distribution: dict[str, object]
    capital_scenarios: tuple[dict[str, object], ...]
    max_price_analysis: dict[str, object]
    exclusion_reason_counts: dict[str, int]
    historical_acquisition_estimates: tuple[dict[str, object], ...]
    instruments: tuple[CurrentInstrumentMeasurement, ...]
    instrument_master_digest: str | None
    cost_model: dict[str, object]
    live_orders_called: bool = False
    schema_version: str = CURRENT_MARKET_SCHEMA_VERSION

    @property
    def suspension_identity_policy(self) -> str:
        """Explicit name for the policy; the older match name is retained for compatibility."""

        return self.suspension_match_policy

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "snapshot_as_of": self.snapshot_as_of.isoformat(),
            "source_files": list(self.source_files),
            "suspended_source_hash": self.suspended_source_hash,
            "suspended_row_count": self.suspended_row_count,
            "suspended_unique_key_count": self.suspended_unique_key_count,
            "suspended_duplicate_key_count": self.suspended_duplicate_key_count,
            "suspended_duplicate_isin_count": self.suspended_duplicate_isin_count,
            "suspension_match_policy": self.suspension_match_policy,
            "suspension_identity_policy": self.suspension_identity_policy,
            "suspension_match_count": self.suspension_match_count,
            "suspension_ambiguous_count": self.suspension_ambiguous_count,
            "suspension_key_variant_count": self.suspension_key_variant_count,
            "exact_suspended_count": self.exact_suspended_count,
            "conflict_count": self.conflict_count,
            "no_record_count": self.no_record_count,
            "ambiguous_exact_count": self.ambiguous_exact_count,
            "gate_counts": dict(sorted(self.gate_counts.items())),
            "quote_requests": {
                "instrument_count": self.quote_request_count,
                "successful_quotes": self.quote_success_count,
                "failed_quotes": self.quote_failure_count,
                "http_request_count": self.quote_http_request_count,
            },
            "quote_source": self.quote_source,
            "cas_distribution": dict(sorted(self.cas_distribution.items())),
            "minimum_quantity_distribution": self.minimum_quantity_distribution,
            "price_distribution": self.price_distribution,
            "capital_scenarios": list(self.capital_scenarios),
            "max_price_analysis": self.max_price_analysis,
            "exclusion_reason_counts": dict(sorted(self.exclusion_reason_counts.items())),
            "historical_acquisition_estimates": list(self.historical_acquisition_estimates),
            "instruments": [item.to_dict() for item in self.instruments],
            "instrument_master_digest": self.instrument_master_digest,
            "cost_model": self.cost_model,
            "live_orders_called": self.live_orders_called,
        }

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _decimal_or_none(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _positive_decimal(value: object) -> Decimal | None:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not result.is_finite() or result <= 0:
        return None
    return result


def _text(row: Mapping[str, object], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _valid_isin(value: object) -> bool:
    return isinstance(value, str) and bool(_ISIN.fullmatch(value.strip())) and not value.startswith("DUMMY")


def _positive_integral(value: object) -> int | None:
    parsed = _positive_decimal(value)
    if parsed is None or parsed != parsed.to_integral_value():
        return None
    return int(parsed)


def _row_gate_reasons(row: Mapping[str, object]) -> list[str]:
    reasons: list[str] = []
    if row.get("exchange") != "NSE":
        reasons.append("not_nse")
    if row.get("segment") != "NSE_EQ":
        reasons.append("not_nse_eq")
    if row.get("instrument_type") != "EQ":
        reasons.append("not_equity_instrument_type")
    if row.get("security_type") != "NORMAL":
        reasons.append("not_normal_security")
    key = _text(row, "instrument_key")
    if not key.startswith("NSE_EQ|"):
        reasons.append("invalid_instrument_key")
    if not _valid_isin(row.get("isin")):
        reasons.append("invalid_isin")
    if not _text(row, "trading_symbol", "symbol"):
        reasons.append("missing_symbol")
    if not _text(row, "name"):
        reasons.append("missing_name")
    if _positive_decimal(row.get("tick_size")) is None:
        reasons.append("invalid_tick_size")
    if _positive_integral(row.get("lot_size")) is None:
        reasons.append("invalid_minimum_tradable_quantity")
    if _positive_decimal(row.get("freeze_quantity")) is None:
        reasons.append("invalid_freeze_quantity")
    return reasons


def quote_request_keys(bod_rows: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    """Return the deterministic, structurally valid quote request set."""

    rows = list(bod_rows)
    counts = Counter(
        _text(row, "instrument_key")
        for row in rows
        if not _row_gate_reasons(row) and _text(row, "instrument_key")
    )
    return tuple(
        sorted(
            key
            for row in rows
            if not _row_gate_reasons(row)
            for key in [_text(row, "instrument_key")]
            if counts[key] == 1
        )
    )


def _source_file(payload: InstrumentFilePayload) -> dict[str, object]:
    return {
        "url": payload.url,
        "sha256": payload.sha256,
        "etag": payload.etag,
        "last_modified": payload.last_modified,
        "row_count": len(payload.rows),
    }


def _distribution(values: Sequence[Decimal | int]) -> dict[str, object]:
    if not values:
        return {"count": 0, "distinct": 0, "min": None, "max": None, "counts": {}}
    counts = Counter(str(value) for value in values)
    return {
        "count": len(values),
        "distinct": len(counts),
        "min": str(min(values)),
        "max": str(max(values)),
        "counts": dict(sorted(counts.items(), key=lambda item: item[0])),
    }


def _parse_quote_timestamp(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.isoformat()


def _quote_cas_value(quote: Mapping[str, object]) -> tuple[bool | None, str | None]:
    if "cas_eligible" not in quote:
        return None, None
    value = quote.get("cas_eligible")
    if isinstance(value, bool):
        return value, None
    return None, "invalid_cas_eligibility"


def _measure_capital(
    *,
    instrument_key: str,
    price: Decimal,
    minimum_quantity: int,
    capital: Decimal,
    cost_provider: CostProvider,
) -> CapitalAffordabilityMeasurement:
    minimum_order = OrderSpec(
        instrument_token=instrument_key,
        exchange=Exchange.NSE,
        side=Side.BUY,
        product=Product.INTRADAY,
        quantity=minimum_quantity,
        price=price,
    )
    minimum_quote = cost_provider.quote(minimum_order)
    minimum_cash = minimum_order.notional + minimum_quote.total
    position = max_affordable_buy_quantity(
        instrument_token=instrument_key,
        exchange=Exchange.NSE,
        product=Product.INTRADAY,
        price=price,
        cash_limit=capital,
        cost_provider=cost_provider,
        minimum_tradable_quantity=minimum_quantity,
    )
    return CapitalAffordabilityMeasurement(
        capital=capital,
        raw_minimum_capital=minimum_order.notional,
        minimum_entry_charges=minimum_quote.total,
        minimum_entry_cash_required=minimum_cash,
        max_affordable_quantity=position.quantity,
        max_affordable_notional=position.notional,
        max_affordable_entry_charges=position.entry_charges,
        cash_required=position.cash_required,
        cash_remaining=position.cash_remaining,
        affordable=position.quantity >= minimum_quantity,
    )


def _estimate_historical_acquisition(
    *,
    candidate_counts: dict[str, int],
    estimated_bytes_per_row: int,
) -> tuple[dict[str, object], ...]:
    if estimated_bytes_per_row <= 0:
        raise ValueError("estimated_bytes_per_row must be positive")

    durations = (
        ("1_trading_day", 1, 1),
        ("5_trading_sessions", 5, 7),
        ("20_trading_sessions", 20, 28),
        ("3_months", 63, 92),
        ("1_year", 252, 366),
    )
    intervals = (("1m", 1, 375), ("5m", 5, 75), ("15m", 15, 25), ("daily", 1, 1))
    output: list[dict[str, object]] = []
    for capital_key, candidate_count in sorted(candidate_counts.items()):
        for duration, sessions, calendar_days in durations:
            for interval, interval_minutes, rows_per_session in intervals:
                limit_days = historical_request_limit_days(interval)
                requests_per_instrument = (calendar_days + limit_days - 1) // limit_days
                rows = candidate_count * sessions * rows_per_session
                output.append(
                    {
                        "capital": capital_key,
                        "duration": duration,
                        "interval": interval,
                        "candidate_count": candidate_count,
                        "trading_sessions_assumed": sessions,
                        "calendar_days_assumed": calendar_days,
                        "provider_max_calendar_days_per_request": limit_days,
                        "requests_per_instrument": requests_per_instrument,
                        "estimated_api_requests": candidate_count * requests_per_instrument,
                        "estimated_rows": rows,
                        "estimated_storage_bytes": rows * estimated_bytes_per_row,
                        "estimate_only": True,
                    }
                )
    return tuple(output)


def _cost_model_metadata(cost_provider: CostProvider) -> dict[str, object]:
    provenance = getattr(cost_provider, "provenance", {})
    return {
        "name": getattr(cost_provider, "MODEL_NAME", cost_provider.__class__.__name__),
        "provenance": dict(provenance) if isinstance(provenance, Mapping) else {},
    }


def measure_current_market(
    *,
    snapshot_as_of: datetime,
    bod: InstrumentFilePayload,
    mis: InstrumentFilePayload,
    suspended: InstrumentFilePayload,
    quotes: QuoteBatchResult,
    cost_provider: CostProvider,
    approved_capitals: Sequence[Decimal] = APPROVED_CAPITALS,
    max_last_price_rupees: Decimal | None = None,
    tick_size_scale_rupees_per_raw_unit: Decimal = DEFAULT_TICK_SIZE_SCALE_RUPEES_PER_RAW_UNIT,
    estimated_bytes_per_row: int = DEFAULT_ESTIMATED_BYTES_PER_ROW,
) -> CurrentMarketCalibrationArtifact:
    """Build a deterministic, read-only current-market calibration artifact."""

    if snapshot_as_of.tzinfo is None:
        raise ValueError("snapshot_as_of must be timezone-aware")
    capitals = tuple(approved_capitals)
    if not capitals or any(not value.is_finite() or value <= 0 for value in capitals):
        raise ValueError("approved_capitals must contain positive finite Decimals")
    if len(set(capitals)) != len(capitals):
        raise ValueError("approved_capitals must be unique")
    if max_last_price_rupees is not None and (
        not max_last_price_rupees.is_finite() or max_last_price_rupees <= 0
    ):
        raise ValueError("max_last_price_rupees must be positive and finite")

    bod_rows = list(bod.rows)
    mis_keys = {
        _text(row, "instrument_key") for row in mis.rows if _text(row, "instrument_key")
    }
    suspended_rows = tuple(suspended.rows)
    suspended_key_counts = Counter(
        _text(row, "instrument_key")
        for row in suspended_rows
        if _text(row, "instrument_key")
    )
    suspended_isin_counts = Counter(
        _text(row, "isin") for row in suspended_rows if _text(row, "isin")
    )
    valid_key_counts = Counter(
        _text(row, "instrument_key") for row in bod_rows if _text(row, "instrument_key")
    )
    instruments: list[CurrentInstrumentMeasurement] = []
    strict_master_rows: list[dict[str, object]] = []
    quote_successes = 0
    quote_failures = 0
    prices: list[Decimal] = []
    minimum_quantities: list[int] = []
    cas_counts = Counter({"cas_eligible": 0, "non_cas": 0, "unknown": 0})
    exclusion_counts: Counter[str] = Counter()
    suspension_status_keys: dict[str, set[str]] = {
        SUSPENDED_EXACT: set(),
        SUSPENSION_CONFLICT: set(),
        NO_SUSPENSION_RECORD: set(),
        AMBIGUOUS_EXACT: set(),
    }

    for row in bod_rows:
        reasons = _row_gate_reasons(row)
        key = _text(row, "instrument_key")
        symbol = _text(row, "trading_symbol", "symbol")
        isin = str(row.get("isin", "")).strip() if row.get("isin") is not None else ""
        duplicate = bool(key) and valid_key_counts[key] > 1
        if duplicate and not reasons:
            reasons.append("duplicate_instrument_key")

        mis_eligible = key in mis_keys
        suspension = resolve_suspension(row, suspended_rows)
        suspended_value = suspension.status in {SUSPENDED_EXACT, AMBIGUOUS_EXACT}
        strict_row = not _row_gate_reasons(row)
        if strict_row and key:
            suspension_status_keys[suspension.status].add(key)
        tick_raw = _positive_decimal(row.get("tick_size"))
        tick_rupees = (
            tick_raw * tick_size_scale_rupees_per_raw_unit if tick_raw is not None else None
        )
        minimum_quantity = _positive_integral(row.get("lot_size"))
        quote = quotes.quotes.get(key)
        quote_failure_reason = quotes.failures.get(key)
        cas_value: bool | None = None
        reference_price: Decimal | None = None
        quote_timestamp: str | None = None
        price_source: str | None = None

        if not reasons and quote is None:
            reasons.append(quote_failure_reason or "missing_quote")
        if quote is not None:
            raw_price = _positive_decimal(quote.get("last_price"))
            if raw_price is None:
                reasons.append("invalid_last_price")
            else:
                reference_price = raw_price
                price_source = "last_price"
            quote_timestamp = _parse_quote_timestamp(quote.get("timestamp"))
            if quote_timestamp is None:
                reasons.append("invalid_quote_timestamp")
            quote_cas, cas_error = _quote_cas_value(quote)
            if cas_error is not None:
                reasons.append(cas_error)
            if quote_cas is not None:
                cas_value = quote_cas
            elif isinstance(row.get("cas_eligible"), bool):
                cas_value = row["cas_eligible"]
            else:
                reasons.append("missing_cas_eligibility")

        measurement_ready = (
            not reasons
            and quote is not None
            and reference_price is not None
            and quote_timestamp is not None
            and cas_value is not None
            and minimum_quantity is not None
        )
        capital_measurements: list[CapitalAffordabilityMeasurement] = []
        if measurement_ready:
            quote_successes += 1
            prices.append(reference_price)
            minimum_quantities.append(minimum_quantity)
            cas_counts["cas_eligible" if cas_value else "non_cas"] += 1
            master_row = dict(row)
            master_row["cas_eligible"] = cas_value
            strict_master_rows.append(master_row)
            for capital in sorted(capitals):
                capital_measurements.append(
                    _measure_capital(
                        instrument_key=key,
                        price=reference_price,
                        minimum_quantity=minimum_quantity,
                        capital=capital,
                        cost_provider=cost_provider,
                    )
                )
        else:
            if key in quotes.requested_instrument_keys and not reasons:
                reasons.append(quote_failure_reason or "quote_not_usable")
            if key in quotes.requested_instrument_keys:
                quote_failures += 1
            cas_counts["unknown"] += 1

        if suspension.status == AMBIGUOUS_EXACT:
            reasons.append("ambiguous_suspended_identity")
        elif suspension.status == SUSPENDED_EXACT:
            reasons.append("suspended_exact_identity")
        elif suspension.status == SUSPENSION_CONFLICT:
            reasons.append("suspension_identity_conflict")
        reasons = tuple(sorted(set(reasons)))
        for reason in reasons:
            exclusion_counts[reason] += 1
        research_candidate = (
            measurement_ready
            and mis_eligible
            and suspension.status not in {SUSPENDED_EXACT, AMBIGUOUS_EXACT}
        )
        candidate = research_candidate
        candidate_after_max = research_candidate and (
            max_last_price_rupees is None or reference_price <= max_last_price_rupees
        )
        instruments.append(
            CurrentInstrumentMeasurement(
                instrument_key=key,
                symbol=symbol,
                isin=isin,
                series=_text(row, "series", "instrument_type"),
                exchange=_text(row, "exchange"),
                instrument_type=_text(row, "instrument_type"),
                security_type=_text(row, "security_type"),
                tick_size_raw=tick_raw,
                tick_size_rupees=tick_rupees,
                minimum_tradable_quantity=minimum_quantity,
                minimum_tradable_quantity_source=(
                    "upstox_nse_bod.lot_size" if minimum_quantity is not None else None
                ),
                mis_eligible=mis_eligible,
                suspended=suspended_value,
                current_exchange_token=(field_text(row, "exchange_token") or None),
                suspension_status=suspension.status,
                suspension_match_count=len(suspension.exact_identity_rows),
                suspension_ambiguous=suspension.status == AMBIGUOUS_EXACT,
                suspension_variant_row_count=len(suspension.same_segment_key_rows),
                same_key_variant_count=len(suspension.same_segment_key_rows),
                exact_token_match_count=len(suspension.exact_token_rows),
                suspension_evidence=tuple(
                    row_evidence(suspended_row)
                    for suspended_row in suspension.same_segment_key_rows
                ),
                cas_eligible=cas_value,
                quote_status="success" if measurement_ready else "failure",
                quote_failure_reason=(None if measurement_ready else (reasons[0] if reasons else "unknown")),
                reference_price=reference_price,
                quote_timestamp=quote_timestamp,
                price_source=price_source,
                candidate=candidate,
                candidate_after_max_last_price=candidate_after_max,
                research_candidate=research_candidate,
                live_tradability_proven=(
                    measurement_ready
                    and mis_eligible
                    and suspension.status == NO_SUSPENSION_RECORD
                ),
                exclusion_reasons=reasons,
                capital_measurements=tuple(capital_measurements),
            )
        )

    master_digest: str | None = None
    if strict_master_rows:
        master = build_nse_equity_master(
            as_of_date=snapshot_as_of.date(),
            bod_rows=strict_master_rows,
            mis_rows=mis.rows,
            suspended_rows=suspended.rows,
            tick_size_scale_rupees_per_raw_unit=tick_size_scale_rupees_per_raw_unit,
        )
        master_digest = master.source_digest

    instruments.sort(key=lambda item: (item.instrument_key, item.symbol, item.isin))
    gate_counts = {
        "raw_upstox_instruments": len(bod_rows),
        "nse": sum(row.get("exchange") == "NSE" for row in bod_rows),
        "nse_eq": sum(row.get("segment") == "NSE_EQ" for row in bod_rows),
        "instrument_type_eq": sum(row.get("instrument_type") == "EQ" for row in bod_rows),
        "security_type_normal": sum(row.get("security_type") == "NORMAL" for row in bod_rows),
        "valid_isin_and_instrument_key": sum(
            _text(row, "instrument_key").startswith("NSE_EQ|") and _valid_isin(row.get("isin"))
            for row in bod_rows
        ),
        "valid_tick_metadata": sum(_positive_decimal(row.get("tick_size")) is not None for row in bod_rows),
        "valid_minimum_tradable_quantity": sum(
            _positive_integral(row.get("lot_size")) is not None for row in bod_rows
        ),
        "current_mis_eligible": sum(
            not _row_gate_reasons(row) and _text(row, "instrument_key") in mis_keys for row in bod_rows
        ),
        "current_suspended": len(
            suspension_status_keys[SUSPENDED_EXACT] | suspension_status_keys[AMBIGUOUS_EXACT]
        ),
        "current_suspended_exact": len(suspension_status_keys[SUSPENDED_EXACT]),
        "current_suspended_ambiguous": len(suspension_status_keys[AMBIGUOUS_EXACT]),
        "suspension_conflicts": len(suspension_status_keys[SUSPENSION_CONFLICT]),
        "suspension_no_record": len(suspension_status_keys[NO_SUSPENSION_RECORD]),
        "quote_request_candidates": len(quotes.requested_instrument_keys),
        "quote_usable": quote_successes,
        "quote_unusable": quote_failures,
        "research_candidates": sum(item.research_candidate for item in instruments),
        "live_tradability_proven": sum(
            item.live_tradability_proven for item in instruments
        ),
        # Retained as a compatibility alias.  It now means research candidates
        # and is not a live-tradability assertion.
        "candidate_current_mis_not_suspended": sum(
            item.research_candidate for item in instruments
        ),
        "candidate_after_nominal_price_filter": sum(
            item.candidate_after_max_last_price for item in instruments
        ),
    }

    scenario_summaries: list[dict[str, object]] = []
    candidate_counts: dict[str, int] = {}
    for capital in sorted(capitals):
        capital_key = str(capital)
        measured_candidates = [item for item in instruments if item.research_candidate]
        measured_after_max = [
            item for item in instruments if item.candidate_after_max_last_price
        ]
        by_key = {
            item.instrument_key: item.capital_measurements
            for item in instruments
            if item.capital_measurements
        }
        affordable = sum(
            any(measurement.capital == capital and measurement.affordable for measurement in by_key[key])
            for key in (item.instrument_key for item in measured_candidates)
        )
        affordable_after_max = sum(
            any(measurement.capital == capital and measurement.affordable for measurement in by_key[key])
            for key in (item.instrument_key for item in measured_after_max)
        )
        candidate_counts[capital_key] = affordable_after_max
        scenario_summaries.append(
            {
                "capital": capital_key,
                "candidate_count_before_affordability": len(measured_candidates),
                "affordable_count_without_nominal_price_filter": affordable,
                "candidate_count_after_nominal_price_filter": len(measured_after_max),
                "affordable_count_after_nominal_price_filter": affordable_after_max,
                "nominal_price_filter_applied": max_last_price_rupees is not None,
                "affordability_rule": "canonical_charge_aware_max_affordable_buy_quantity",
            }
        )

    max_price_analysis = {
        "max_last_price_rupees": _decimal_or_none(max_last_price_rupees),
        "applied": max_last_price_rupees is not None,
        "independent_of_charge_aware_affordability": True,
        "required_for_affordability": False,
        "relationship": "independent_nominal_price_filter",
        "candidate_count_before": sum(item.research_candidate for item in instruments),
        "candidate_count_after": sum(item.candidate_after_max_last_price for item in instruments),
        "excluded_by_filter": sum(
            item.research_candidate and not item.candidate_after_max_last_price
            for item in instruments
        ),
    }
    return CurrentMarketCalibrationArtifact(
        snapshot_as_of=snapshot_as_of,
        source_files=tuple(
            sorted(
                (_source_file(payload) for payload in (bod, mis, suspended)),
                key=lambda item: str(item["url"]),
            )
        ),
        suspended_source_hash=suspended.sha256,
        suspended_row_count=len(suspended_rows),
        suspended_unique_key_count=len(suspended_key_counts),
        suspended_duplicate_key_count=sum(count > 1 for count in suspended_key_counts.values()),
        suspended_duplicate_isin_count=sum(count > 1 for count in suspended_isin_counts.values()),
        suspension_match_policy=SUSPENSION_MATCH_POLICY,
        suspension_match_count=len(suspension_status_keys[SUSPENDED_EXACT]),
        suspension_ambiguous_count=len(suspension_status_keys[AMBIGUOUS_EXACT]),
        suspension_key_variant_count=len(
            suspension_status_keys[SUSPENSION_CONFLICT]
            | suspension_status_keys[AMBIGUOUS_EXACT]
        ),
        exact_suspended_count=len(suspension_status_keys[SUSPENDED_EXACT]),
        conflict_count=len(suspension_status_keys[SUSPENSION_CONFLICT]),
        no_record_count=len(suspension_status_keys[NO_SUSPENSION_RECORD]),
        ambiguous_exact_count=len(suspension_status_keys[AMBIGUOUS_EXACT]),
        gate_counts=gate_counts,
        quote_request_count=len(quotes.requested_instrument_keys),
        quote_success_count=quote_successes,
        quote_failure_count=quote_failures,
        quote_http_request_count=quotes.request_count,
        quote_source={
            "url": UPSTOX_FULL_QUOTE_V3_URL,
            "documentation": UPSTOX_FULL_QUOTE_V3_DOC,
            "max_instrument_keys_per_request": MAX_INSTRUMENTS_PER_REQUEST,
            "batch_order": "lexicographic_instrument_key",
        },
        cas_distribution={key: cas_counts[key] for key in ("cas_eligible", "non_cas", "unknown")},
        minimum_quantity_distribution=_distribution(minimum_quantities),
        price_distribution=_distribution(prices),
        capital_scenarios=tuple(scenario_summaries),
        max_price_analysis=max_price_analysis,
        exclusion_reason_counts=dict(exclusion_counts),
        historical_acquisition_estimates=_estimate_historical_acquisition(
            candidate_counts=candidate_counts,
            estimated_bytes_per_row=estimated_bytes_per_row,
        ),
        instruments=tuple(instruments),
        instrument_master_digest=master_digest,
        cost_model=_cost_model_metadata(cost_provider),
    )
