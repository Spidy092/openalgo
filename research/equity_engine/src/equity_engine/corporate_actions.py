from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from .universe import CorporateActionAssessment

UPSTOX_CORPORATE_ACTIONS_DOC = (
    "https://upstox.com/developer/api-documentation/get-corporate-actions/"
)
UPSTOX_CORPORATE_ACTIONS_BASE = "https://api.upstox.com/v2/fundamentals"


class CorporateActionEventType(StrEnum):
    """Canonical corporate action event classifications."""

    SPLIT = "SPLIT"
    BONUS = "BONUS"
    MERGER = "MERGER"
    DEMERGER = "DEMERGER"
    SYMBOL_CHANGE = "SYMBOL_CHANGE"
    ISIN_CHANGE = "ISIN_CHANGE"
    DELISTING = "DELISTING"
    RELISTING = "RELISTING"
    RIGHTS = "RIGHTS"
    DIVIDEND = "DIVIDEND"
    CAPITAL_REDUCTION = "CAPITAL_REDUCTION"
    OTHER_STRUCTURAL = "OTHER_STRUCTURAL"


class CorporateActionConfidence(StrEnum):
    """Provenance and confidence level of corporate action evidence."""

    EXCHANGE_NOTICE = "EXCHANGE_NOTICE"
    CONFIRMED = "CONFIRMED"
    PROVISIONAL = "PROVISIONAL"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


class CorporateActionEvaluationMode(StrEnum):
    """Operational mode distinguishing tradable point-in-time state from ex-post normalization."""

    TRADABLE_INFORMATION = "TRADABLE_INFORMATION"
    EX_POST_NORMALIZATION = "EX_POST_NORMALIZATION"


class DividendPolicy(StrEnum):
    """Explicit experiment policy governing dividend price interpretation."""

    IGNORE_BELOW_THRESHOLD = "IGNORE_BELOW_THRESHOLD"
    BLOCK_ALL = "BLOCK_ALL"
    CASH_ADJUST = "CASH_ADJUST"


class CorporateActionDataLeakageError(ValueError):
    """Raised when an earlier trading decision attempts to observe a future announcement."""


class CorporateActionCoverageError(ValueError):
    """Raised when corporate-action evidence does not cover the requested window."""


def parse_ratio(ratio_str: str) -> tuple[Decimal, Decimal]:
    """Parse ratio 'A:B' into (A, B)."""
    parts = ratio_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid ratio format: {ratio_str!r}, expected 'A:B'")
    try:
        a = Decimal(parts[0].strip())
        b = Decimal(parts[1].strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid numbers in ratio: {ratio_str!r}") from exc
    if a <= 0 or b <= 0:
        raise ValueError(f"ratio values must be positive: {ratio_str!r}")
    return a, b


def _canonical_timestamp(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)
    return ts.isoformat()


@dataclass(frozen=True)
class CorporateActionRecord:
    """Canonical point-in-time corporate action evidence record."""

    instrument_key: str
    isin: str
    event_type: CorporateActionEventType
    effective_date: date
    source: str
    retrieval_timestamp: datetime
    announcement_date: date | None = None
    ex_date: date | None = None
    confidence: CorporateActionConfidence = CorporateActionConfidence.CONFIRMED
    raw_candles_comparable: bool = False
    adjustment_required: bool = True
    blocking: bool = True
    ratio: str | None = None
    amount: Decimal | None = None
    split_factor: Decimal | None = None
    from_symbol: str | None = None
    to_symbol: str | None = None
    old_isin: str | None = None
    new_isin: str | None = None
    details: tuple[tuple[str, str], ...] = ()
    evidence_fingerprint: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.instrument_key.strip():
            raise ValueError("instrument_key is required")
        if not self.isin.strip():
            raise ValueError("isin is required")
        if not self.source.strip():
            raise ValueError("source is required")
        if self.announcement_date and self.announcement_date > self.effective_date:
            raise ValueError("announcement_date cannot be after effective_date")
        if self.ex_date and self.ex_date > self.effective_date:
            raise ValueError("ex_date cannot be after effective_date")

        aware_ts = (
            self.retrieval_timestamp.replace(tzinfo=UTC)
            if self.retrieval_timestamp.tzinfo is None
            else self.retrieval_timestamp.astimezone(UTC)
        )
        if aware_ts != self.retrieval_timestamp:
            object.__setattr__(self, "retrieval_timestamp", aware_ts)

        computed_fp = self._compute_fingerprint()
        if self.evidence_fingerprint is not None and self.evidence_fingerprint != computed_fp:
            raise ValueError(
                f"invalid caller-supplied evidence_fingerprint: "
                f"expected {computed_fp}, got {self.evidence_fingerprint}"
            )
        object.__setattr__(self, "evidence_fingerprint", computed_fp)

    @property
    def knowledge_date(self) -> date | None:
        """The date on which this corporate action was announced/known, or None if unknown."""
        return self.announcement_date

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "instrument_key": self.instrument_key,
            "isin": self.isin,
            "event_type": self.event_type.value,
            "announcement_date": (
                self.announcement_date.isoformat() if self.announcement_date else None
            ),
            "ex_date": self.ex_date.isoformat() if self.ex_date else None,
            "effective_date": self.effective_date.isoformat(),
            "source": self.source,
            "retrieval_timestamp": _canonical_timestamp(self.retrieval_timestamp),
            "confidence": self.confidence.value,
            "raw_candles_comparable": self.raw_candles_comparable,
            "adjustment_required": self.adjustment_required,
            "blocking": self.blocking,
            "ratio": self.ratio,
            "amount": str(self.amount) if self.amount is not None else None,
            "split_factor": str(self.split_factor) if self.split_factor is not None else None,
            "from_symbol": self.from_symbol,
            "to_symbol": self.to_symbol,
            "old_isin": self.old_isin,
            "new_isin": self.new_isin,
            "details": sorted(self.details),
        }

    def _compute_fingerprint(self) -> str:
        payload = self.canonical_payload()
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        return self.evidence_fingerprint or self._compute_fingerprint()

    def as_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["evidence_fingerprint"] = self.fingerprint()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CorporateActionRecord:
        """Construct record from serialized dictionary, verifying fingerprint integrity."""
        raw_fp = data.get("evidence_fingerprint")
        ann_date = (
            date.fromisoformat(data["announcement_date"])
            if data.get("announcement_date")
            else None
        )
        ex_date = (
            date.fromisoformat(data["ex_date"]) if data.get("ex_date") else None
        )
        eff_date = date.fromisoformat(data["effective_date"])
        ret_ts = datetime.fromisoformat(data["retrieval_timestamp"])
        amt = Decimal(str(data["amount"])) if data.get("amount") is not None else None
        split_f = (
            Decimal(str(data["split_factor"]))
            if data.get("split_factor") is not None
            else None
        )
        raw_details = data.get("details", ())
        details = tuple(
            tuple(item) if isinstance(item, (list, tuple)) else item
            for item in raw_details
        )
        return cls(
            instrument_key=data["instrument_key"],
            isin=data["isin"],
            event_type=CorporateActionEventType(data["event_type"]),
            effective_date=eff_date,
            source=data["source"],
            retrieval_timestamp=ret_ts,
            announcement_date=ann_date,
            ex_date=ex_date,
            confidence=CorporateActionConfidence(data.get("confidence", "CONFIRMED")),
            raw_candles_comparable=bool(data.get("raw_candles_comparable", False)),
            adjustment_required=bool(data.get("adjustment_required", True)),
            blocking=bool(data.get("blocking", True)),
            ratio=data.get("ratio"),
            amount=amt,
            split_factor=split_f,
            from_symbol=data.get("from_symbol"),
            to_symbol=data.get("to_symbol"),
            old_isin=data.get("old_isin"),
            new_isin=data.get("new_isin"),
            details=details,
            evidence_fingerprint=raw_fp,
        )


@dataclass(frozen=True)
class CoverageScope:
    """Date-scoped evidence coverage certificate for an instrument."""

    instrument_key: str
    isin: str
    start_date: date
    end_date: date
    source: str
    retrieval_timestamp: datetime
    is_complete: bool = True
    notes: str = ""
    coverage_fingerprint: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        if not self.instrument_key.strip():
            raise ValueError("instrument_key is required")
        if not self.isin.strip():
            raise ValueError("isin is required")
        if not self.source.strip():
            raise ValueError("source is required")

        aware_ts = (
            self.retrieval_timestamp.replace(tzinfo=UTC)
            if self.retrieval_timestamp.tzinfo is None
            else self.retrieval_timestamp.astimezone(UTC)
        )
        if aware_ts != self.retrieval_timestamp:
            object.__setattr__(self, "retrieval_timestamp", aware_ts)

        computed_fp = self._compute_fingerprint()
        if self.coverage_fingerprint is not None and self.coverage_fingerprint != computed_fp:
            raise ValueError(
                f"invalid caller-supplied coverage_fingerprint: "
                f"expected {computed_fp}, got {self.coverage_fingerprint}"
            )
        object.__setattr__(self, "coverage_fingerprint", computed_fp)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "instrument_key": self.instrument_key,
            "isin": self.isin,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "source": self.source,
            "retrieval_timestamp": _canonical_timestamp(self.retrieval_timestamp),
            "is_complete": self.is_complete,
            "notes": self.notes,
        }

    def _compute_fingerprint(self) -> str:
        payload = self.canonical_payload()
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        return self.coverage_fingerprint or self._compute_fingerprint()

    def as_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["coverage_fingerprint"] = self.fingerprint()
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CoverageScope:
        """Construct scope from serialized dictionary, verifying fingerprint integrity."""
        raw_fp = data.get("coverage_fingerprint")
        return cls(
            instrument_key=data["instrument_key"],
            isin=data["isin"],
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            source=data["source"],
            retrieval_timestamp=datetime.fromisoformat(data["retrieval_timestamp"]),
            is_complete=bool(data.get("is_complete", True)),
            notes=str(data.get("notes", "")),
            coverage_fingerprint=raw_fp,
        )


@dataclass(frozen=True)
class AdjustmentFactorRecord:
    """Ex-post mechanical adjustment factors for historical candles."""

    effective_date: date
    event_type: CorporateActionEventType
    price_multiplier: Decimal
    volume_multiplier: Decimal
    ratio: str | None = None
    cash_adjustment: Decimal | None = None


@dataclass(frozen=True)
class CorporateActionPolicy:
    """Explicit experiment policy governing corporate action blocking and interpretation."""

    blocked_event_types: frozenset[CorporateActionEventType] = frozenset(
        {
            CorporateActionEventType.SPLIT,
            CorporateActionEventType.BONUS,
            CorporateActionEventType.MERGER,
            CorporateActionEventType.DEMERGER,
            CorporateActionEventType.DELISTING,
            CorporateActionEventType.RIGHTS,
            CorporateActionEventType.CAPITAL_REDUCTION,
            CorporateActionEventType.OTHER_STRUCTURAL,
        }
    )
    allow_ex_post_adjusted_splits: bool = False
    allow_ex_post_adjusted_bonuses: bool = False
    dividend_policy: DividendPolicy = DividendPolicy.BLOCK_ALL
    dividend_threshold_percent: Decimal | None = None
    reference_price_for_dividend: Decimal | None = None
    policy_name: str = "DEFAULT_RESEARCH_POLICY"

    def __post_init__(self) -> None:
        if self.dividend_policy == DividendPolicy.IGNORE_BELOW_THRESHOLD:
            if (
                self.dividend_threshold_percent is None
                or self.dividend_threshold_percent <= 0
            ):
                raise ValueError(
                    "dividend policy IGNORE_BELOW_THRESHOLD requires an explicit positive dividend_threshold_percent; "
                    "no implicit threshold is permitted"
                )
            if (
                self.reference_price_for_dividend is None
                or self.reference_price_for_dividend <= 0
            ):
                raise ValueError(
                    "dividend policy IGNORE_BELOW_THRESHOLD requires an explicit positive reference_price_for_dividend; "
                    "cannot evaluate threshold without a reference price"
                )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "blocked_event_types": sorted(
                e.value if isinstance(e, CorporateActionEventType) else str(e)
                for e in self.blocked_event_types
            ),
            "allow_ex_post_adjusted_splits": self.allow_ex_post_adjusted_splits,
            "allow_ex_post_adjusted_bonuses": self.allow_ex_post_adjusted_bonuses,
            "dividend_policy": (
                self.dividend_policy.value
                if isinstance(self.dividend_policy, DividendPolicy)
                else str(self.dividend_policy)
            ),
            "dividend_threshold_percent": (
                str(self.dividend_threshold_percent)
                if self.dividend_threshold_percent is not None
                else None
            ),
            "reference_price_for_dividend": (
                str(self.reference_price_for_dividend)
                if self.reference_price_for_dividend is not None
                else None
            ),
        }

    def policy_fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint over complete canonical policy payload."""
        payload = self.canonical_payload()
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @property
    def policy_identity(self) -> str:
        """Full cryptographic policy identity binding all behavior-changing fields."""
        return f"{self.policy_name}:{self.policy_fingerprint()}"



@dataclass(frozen=True)
class CorporateActionEvent:
    """Legacy-compatible corporate action event presentation."""

    name: str
    effective_date: date
    amount: Decimal | None
    ratio: str | None

    def to_record(
        self,
        *,
        instrument_key: str,
        isin: str,
        source: str = "UPSTOX_API",
        retrieval_timestamp: datetime | None = None,
        announcement_date: date | None = None,
    ) -> CorporateActionRecord:
        """Convert to canonical CorporateActionRecord."""
        name_upper = self.name.strip().upper()
        if "SPLIT" in name_upper:
            event_type = CorporateActionEventType.SPLIT
            raw_comp = False
            adj_req = True
            blocking = True
        elif "BONUS" in name_upper:
            event_type = CorporateActionEventType.BONUS
            raw_comp = False
            adj_req = True
            blocking = True
        elif "DIVIDEND" in name_upper:
            event_type = CorporateActionEventType.DIVIDEND
            raw_comp = True
            adj_req = False
            blocking = False
        elif "RIGHT" in name_upper:
            event_type = CorporateActionEventType.RIGHTS
            raw_comp = False
            adj_req = True
            blocking = True
        elif "DEMERGER" in name_upper:
            event_type = CorporateActionEventType.DEMERGER
            raw_comp = False
            adj_req = True
            blocking = True
        elif "MERGER" in name_upper:
            event_type = CorporateActionEventType.MERGER
            raw_comp = False
            adj_req = True
            blocking = True
        elif "SYMBOL" in name_upper:
            event_type = CorporateActionEventType.SYMBOL_CHANGE
            raw_comp = True
            adj_req = False
            blocking = False
        elif "DELIST" in name_upper:
            event_type = CorporateActionEventType.DELISTING
            raw_comp = False
            adj_req = True
            blocking = True
        else:
            event_type = CorporateActionEventType.OTHER_STRUCTURAL
            raw_comp = False
            adj_req = True
            blocking = True

        return CorporateActionRecord(
            instrument_key=instrument_key,
            isin=isin,
            event_type=event_type,
            effective_date=self.effective_date,
            announcement_date=announcement_date,
            source=source,
            retrieval_timestamp=retrieval_timestamp or datetime.now(UTC),
            raw_candles_comparable=raw_comp,
            adjustment_required=adj_req,
            blocking=blocking,
            ratio=self.ratio,
            amount=self.amount,
        )


class UpstoxCorporateActionProvider:
    """Read corporate actions by ISIN; no price adjustment is inferred here."""

    def __init__(
        self,
        *,
        access_token: str,
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("access_token is required")
        self._access_token = access_token
        self._timeout_seconds = timeout_seconds
        self._client = client

    def fetch(self, isin: str) -> tuple[CorporateActionEvent, ...]:
        if not isin:
            raise ValueError("isin is required")
        url = f"{UPSTOX_CORPORATE_ACTIONS_BASE}/{isin}/corporate-actions"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._access_token}",
        }
        if self._client is None:
            if httpx is None:
                raise RuntimeError("httpx is required to fetch Upstox corporate actions")
            response = httpx.get(url, headers=headers, timeout=self._timeout_seconds)
        else:
            response = self._client.get(url, headers=headers, timeout=self._timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"unexpected Upstox corporate-actions response: {payload!r}")
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise RuntimeError("Upstox corporate-actions response did not contain data array")
        return parse_corporate_action_rows(rows)


def _parse_effective_date(value: object) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("corporate action missing expiry_date")
    try:
        return datetime.strptime(value.strip(), "%d %b %Y").date()
    except ValueError as exc:
        raise ValueError(f"unsupported corporate-action date format: {value!r}") from exc


def parse_corporate_action_rows(rows: Iterable[object]) -> tuple[CorporateActionEvent, ...]:
    events: list[CorporateActionEvent] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("corporate action row must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("corporate action row missing name")
        amount_raw = raw.get("amount")
        amount = Decimal(str(amount_raw)) if amount_raw is not None else None
        ratio_raw = raw.get("ratio")
        if ratio_raw is not None and not isinstance(ratio_raw, str):
            raise ValueError("corporate action ratio must be text or null")
        events.append(
            CorporateActionEvent(
                name=name.strip(),
                effective_date=_parse_effective_date(raw.get("expiry_date")),
                amount=amount,
                ratio=ratio_raw,
            )
        )
    return tuple(sorted(events, key=lambda item: (item.effective_date, item.name)))


def assess_corporate_actions(
    *,
    events: Iterable[CorporateActionEvent | CorporateActionRecord],
    research_start: date,
    research_end: date,
    blocked_event_names: frozenset[str],
) -> CorporateActionAssessment:
    """Mark structural events as blockers using an explicit caller-supplied policy.

    Upstox documents `expiry_date` as the ex/effective date. This function does not assume which
    event types need adjustment: `blocked_event_names` is mandatory experiment policy. A typical
    research policy may block Split/Bonus/Rights until a normalization method is verified.
    """

    if research_start > research_end:
        raise ValueError("research_start must be on or before research_end")
    if not blocked_event_names:
        raise ValueError("blocked_event_names must be explicitly non-empty")

    blocking: list[str] = []
    for item in events:
        if isinstance(item, CorporateActionRecord):
            name = item.event_type.value
            effective_date = item.effective_date
            ratio = item.ratio
        else:
            name = item.name
            effective_date = item.effective_date
            ratio = item.ratio

        if research_start <= effective_date <= research_end and name in blocked_event_names:
            detail = f"{name}@{effective_date.isoformat()}"
            if ratio:
                detail += f" ratio={ratio}"
            blocking.append(detail)

    return CorporateActionAssessment(complete=True, blocking_events=tuple(blocking))


class PointInTimeCorporateActionLedger:
    """Canonical Point-in-Time Corporate-Action Evidence Ledger.

    Enforces:
    1. Explicit date-scoped coverage tracking (UNKNOWN coverage != no-event).
    2. Zero future announcement leakage into earlier trading decisions.
    3. Strict separation of ex-post mechanical adjustment from tradable knowledge.
    4. Deterministic cryptographic evidence fingerprinting.
    """

    def __init__(self, source: str = "CANONICAL_PIT_CORPORATE_ACTION_LEDGER") -> None:
        self.source = source
        self._records: dict[str, list[CorporateActionRecord]] = {}
        self._coverage: dict[str, list[CoverageScope]] = {}
        self._symbol_history: list[tuple[date, str, str]] = []  # effective_date, isin, symbol


    def add_coverage(self, coverage: CoverageScope) -> None:
        """Register a verified coverage scope for an instrument."""
        scopes = self._coverage.setdefault(coverage.instrument_key, [])
        scopes.append(coverage)

    def add_record(self, record: CorporateActionRecord) -> None:
        """Register a point-in-time corporate action record."""
        records = self._records.setdefault(record.instrument_key, [])
        records.append(record)
        if record.event_type == CorporateActionEventType.SYMBOL_CHANGE:
            if record.from_symbol:
                self._symbol_history.append(
                    (record.effective_date - timedelta(days=1), record.isin, record.from_symbol)
                )
            if record.to_symbol:
                self._symbol_history.append(
                    (record.effective_date, record.isin, record.to_symbol)
                )

    def is_covered(self, instrument_key: str, start: date, end: date) -> bool:
        """Verify that [start, end] is contiguously covered by complete evidence.

        UNKNOWN event coverage must NOT be treated as no-event.
        """
        if start > end:
            raise ValueError("start must be on or before end")
        scopes = [s for s in self._coverage.get(instrument_key, []) if s.is_complete]
        if not scopes:
            return False

        # Sort scopes by start_date
        sorted_scopes = sorted(scopes, key=lambda s: (s.start_date, s.end_date))

        # Merge contiguous or overlapping intervals
        merged: list[tuple[date, date]] = []
        for s in sorted_scopes:
            if not merged:
                merged.append((s.start_date, s.end_date))
                continue
            cur_start, cur_end = merged[-1]
            if s.start_date <= cur_end + timedelta(days=1):
                merged[-1] = (cur_start, max(cur_end, s.end_date))
            else:
                merged.append((s.start_date, s.end_date))

        # Check if [start, end] is fully enclosed in any merged span
        return any(m_start <= start and m_end >= end for m_start, m_end in merged)

    def get_coverage_range(self, instrument_key: str) -> tuple[date, date] | None:
        """Return the overall minimum start and maximum end date of coverage for an instrument."""
        scopes = [s for s in self._coverage.get(instrument_key, []) if s.is_complete]
        if not scopes:
            return None
        min_start = min(s.start_date for s in scopes)
        max_end = max(s.end_date for s in scopes)
        return min_start, max_end

    def get_records(self, instrument_key: str) -> tuple[CorporateActionRecord, ...]:
        """Return all registered corporate action records for an instrument, sorted."""
        return tuple(
            sorted(
                self._records.get(instrument_key, []),
                key=lambda r: (r.effective_date, r.event_type.value),
            )
        )

    def get_tradable_events(
        self,
        instrument_key: str,
        as_of_date: date,
        *,
        fail_on_unknown: bool = True,
    ) -> tuple[CorporateActionRecord, ...]:
        """Return events that were known to the market on or before as_of_date.

        If fail_on_unknown is True and any record has an unknown announcement date,
        raises CorporateActionCoverageError.
        Events announced after as_of_date are strictly filtered out to prevent data leakage.
        """
        known_events: list[CorporateActionRecord] = []
        for record in self.get_records(instrument_key):
            if record.announcement_date is None:
                if fail_on_unknown:
                    raise CorporateActionCoverageError(
                        f"corporate action {record.event_type.value}@{record.effective_date.isoformat()} "
                        f"has unknown announcement date; cannot determine tradable knowledge as of {as_of_date.isoformat()}"
                    )
                continue
            if record.announcement_date <= as_of_date:
                known_events.append(record)
        return tuple(known_events)

    def verify_no_future_leakage(
        self, events: Iterable[CorporateActionRecord], as_of_date: date
    ) -> None:
        """Verify that no event in the collection has an announcement after as_of_date."""
        for record in events:
            if record.announcement_date is None:
                raise CorporateActionCoverageError(
                    f"corporate action {record.event_type.value}@{record.effective_date.isoformat()} "
                    f"has unknown announcement date; cannot verify leakage against as_of_date {as_of_date.isoformat()}"
                )
            if record.announcement_date > as_of_date:
                raise CorporateActionDataLeakageError(
                    f"corporate action {record.event_type.value}@{record.effective_date.isoformat()} "
                    f"was announced on {record.announcement_date.isoformat()}, which is after "
                    f"trading decision as_of_date {as_of_date.isoformat()}"
                )

    def get_ex_post_events(
        self, instrument_key: str, window_start: date, window_end: date
    ) -> tuple[CorporateActionRecord, ...]:
        """Return events taking effect within [window_start, window_end] for ex-post normalization."""
        if window_start > window_end:
            raise ValueError("window_start must be on or before window_end")
        return tuple(
            r
            for r in self.get_records(instrument_key)
            if window_start <= r.effective_date <= window_end
        )

    def assess_window(
        self,
        instrument_key: str,
        window_start: date,
        window_end: date,
        *,
        policy: CorporateActionPolicy | None = None,
        as_of_date: date | None = None,
        evaluation_mode: CorporateActionEvaluationMode | None = None,
    ) -> CorporateActionAssessment:
        """Assess corporate actions for an instrument over an exact research window.

        Enforces:
        - UNKNOWN coverage != no-event (incomplete coverage returns complete=False).
        - TRADABLE_INFORMATION filters out unannounced future events and fails closed on unknown announcement dates.
        - EX_POST_NORMALIZATION evaluates mechanical adjustments and candle comparability.
        """
        if window_start > window_end:
            raise ValueError("window_start must be on or before window_end")

        effective_policy = policy if policy is not None else CorporateActionPolicy()
        mode = (
            evaluation_mode
            if evaluation_mode is not None
            else (
                CorporateActionEvaluationMode.TRADABLE_INFORMATION
                if as_of_date is not None
                else CorporateActionEvaluationMode.EX_POST_NORMALIZATION
            )
        )

        # Step 1: Coverage verification (fail closed on unknown / missing coverage)
        if not self.is_covered(instrument_key, window_start, window_end):
            return CorporateActionAssessment(
                complete=False,
                blocking_events=(
                    f"UNCOVERED_WINDOW: corporate-action coverage missing for "
                    f"{instrument_key} in [{window_start.isoformat()}, {window_end.isoformat()}]",
                ),
            )

        # Step 2: Determine events within the window
        candidate_records = self.get_ex_post_events(instrument_key, window_start, window_end)

        # Step 3: Enforce mode-specific leakage checks
        if mode == CorporateActionEvaluationMode.TRADABLE_INFORMATION:
            unknown_announcement_records = [
                r for r in candidate_records if r.announcement_date is None
            ]
            if unknown_announcement_records:
                blocking_unknowns = [
                    f"UNKNOWN_ANNOUNCEMENT_DATE: {r.event_type.value}@{r.effective_date.isoformat()} "
                    f"lacks verified announcement_date; cannot establish tradable knowledge for {instrument_key}"
                    for r in unknown_announcement_records
                ]
                return CorporateActionAssessment(
                    complete=False,
                    blocking_events=tuple(blocking_unknowns),
                )
            eval_date = as_of_date if as_of_date is not None else window_end
            if as_of_date is not None:
                # If caller explicitly asks what was tradable as of as_of_date,
                # any event taking effect in the window but announced after as_of_date
                # was NOT known yet.
                candidate_records = tuple(
                    r
                    for r in candidate_records
                    if r.announcement_date is not None and r.announcement_date <= eval_date
                )

        # Step 4: Policy-driven blocking evaluation
        blocking: list[str] = []
        for r in candidate_records:
            is_blocked = False
            detail = f"{r.event_type.value}@{r.effective_date.isoformat()}"
            if r.ratio:
                detail += f" ratio={r.ratio}"
            if r.amount is not None:
                detail += f" amount={r.amount}"

            if r.event_type == CorporateActionEventType.DIVIDEND:
                if effective_policy.dividend_policy == DividendPolicy.BLOCK_ALL:
                    is_blocked = True
                elif (
                    effective_policy.dividend_policy
                    == DividendPolicy.IGNORE_BELOW_THRESHOLD
                ):
                    if (
                        r.amount is not None
                        and effective_policy.reference_price_for_dividend is not None
                        and effective_policy.reference_price_for_dividend > 0
                    ):
                        div_pct = (
                            r.amount / effective_policy.reference_price_for_dividend
                        ) * Decimal(100)
                        if div_pct >= effective_policy.dividend_threshold_percent:
                            is_blocked = True
                            detail += f" dividend_yield_pct={div_pct:.2f}%>=cap"
                        else:
                            is_blocked = False
                    else:
                        is_blocked = r.blocking
                elif effective_policy.dividend_policy == DividendPolicy.CASH_ADJUST:
                    is_blocked = False
            elif r.event_type in effective_policy.blocked_event_types:
                if (
                    r.event_type == CorporateActionEventType.SPLIT
                    and effective_policy.allow_ex_post_adjusted_splits
                ) or (
                    r.event_type == CorporateActionEventType.BONUS
                    and effective_policy.allow_ex_post_adjusted_bonuses
                ):
                    is_blocked = False
                elif r.event_type == CorporateActionEventType.SYMBOL_CHANGE:
                    # Symbol changes do not block raw candles if share terms are identical
                    is_blocked = not r.raw_candles_comparable
                else:
                    is_blocked = True
            elif r.blocking or not r.raw_candles_comparable:
                is_blocked = True

            if is_blocked:
                blocking.append(detail)

        return CorporateActionAssessment(complete=True, blocking_events=tuple(blocking))

    def resolve_symbol_at(
        self, isin: str, as_of_date: date, *, default_symbol: str = ""
    ) -> str:
        """Resolve the active trading symbol for an ISIN at an exact historical date."""
        history = [
            (eff_date, s)
            for eff_date, i, s in self._symbol_history
            if i == isin and eff_date <= as_of_date
        ]
        if not history:
            return default_symbol
        # Most recent symbol on or before as_of_date
        history.sort(key=lambda item: item[0])
        return history[-1][1]

    def compute_adjustment_factors(
        self, instrument_key: str, window_start: date, window_end: date
    ) -> tuple[AdjustmentFactorRecord, ...]:
        """Compute ex-post mechanical adjustment factors for historical candles."""
        records = self.get_ex_post_events(instrument_key, window_start, window_end)
        factors: list[AdjustmentFactorRecord] = []
        for r in records:
            if r.event_type == CorporateActionEventType.SPLIT:
                if r.ratio:
                    a, b = parse_ratio(r.ratio)
                    # A old shares become B new shares (e.g. 1:2 split -> price factor 0.5, volume factor 2.0)
                    price_mult = a / b
                    vol_mult = b / a
                elif r.split_factor:
                    price_mult = Decimal(1) / r.split_factor
                    vol_mult = r.split_factor
                else:
                    raise ValueError(f"split event missing ratio or split_factor: {r}")
                factors.append(
                    AdjustmentFactorRecord(
                        effective_date=r.effective_date,
                        event_type=r.event_type,
                        price_multiplier=price_mult,
                        volume_multiplier=vol_mult,
                        ratio=r.ratio,
                    )
                )
            elif r.event_type == CorporateActionEventType.BONUS:
                if not r.ratio:
                    raise ValueError(f"bonus event missing ratio: {r}")
                a, b = parse_ratio(r.ratio)
                # A bonus shares for B held: B shares become B + A shares
                # e.g. 1:1 bonus -> 1 becomes 2 -> price factor 1/2 = 0.5, vol factor 2/1 = 2.0
                price_mult = b / (b + a)
                vol_mult = (b + a) / b
                factors.append(
                    AdjustmentFactorRecord(
                        effective_date=r.effective_date,
                        event_type=r.event_type,
                        price_multiplier=price_mult,
                        volume_multiplier=vol_mult,
                        ratio=r.ratio,
                    )
                )
            elif r.event_type == CorporateActionEventType.DIVIDEND and r.amount is not None:
                factors.append(
                    AdjustmentFactorRecord(
                        effective_date=r.effective_date,
                        event_type=r.event_type,
                        price_multiplier=Decimal(1),
                        volume_multiplier=Decimal(1),
                        cash_adjustment=r.amount,
                    )
                )
        return tuple(factors)

    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint of the entire ledger's records and coverage."""
        all_scopes: list[dict[str, Any]] = []
        for scopes in self._coverage.values():
            for s in scopes:
                all_scopes.append(s.canonical_payload())
        all_scopes.sort(key=lambda s: (s["instrument_key"], s["start_date"], s["end_date"]))

        all_records: list[dict[str, Any]] = []
        for records in self._records.values():
            for r in records:
                all_records.append(r.canonical_payload())
        all_records.sort(
            key=lambda r: (r["instrument_key"], r["effective_date"], r["event_type"])
        )

        ledger_payload = {
            "scopes": all_scopes,
            "records": all_records,
        }
        canonical_json = json.dumps(ledger_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def to_evidence_identity(
        self,
        *,
        research_start: date,
        research_end: date,
        instruments: Iterable[str],
        policy: CorporateActionPolicy | None = None,
        evaluation_mode: CorporateActionEvaluationMode | str = CorporateActionEvaluationMode.TRADABLE_INFORMATION,
        source: str | None = None,
    ) -> Any:
        """Derive canonical CorporateActionEvidenceIdentity for an experiment."""
        from .experiment import CorporateActionEvidenceIdentity

        effective_policy = policy if policy is not None else CorporateActionPolicy()
        raw_instruments = tuple(instruments)
        if not raw_instruments:
            raise CorporateActionCoverageError(
                "cannot derive corporate-action evidence identity: requested instrument population is empty"
            )
        if any(not str(k).strip() for k in raw_instruments):
            raise CorporateActionCoverageError(
                "cannot derive corporate-action evidence identity: instrument keys must be non-empty"
            )
        if len(set(raw_instruments)) != len(raw_instruments):
            raise CorporateActionCoverageError(
                "cannot derive corporate-action evidence identity: instrument population contains duplicates"
            )
        instrument_list = tuple(sorted(raw_instruments))
        if instrument_list != raw_instruments:
            raise CorporateActionCoverageError(
                "cannot derive corporate-action evidence identity: instrument population must be sorted unique"
            )

        mode = (
            evaluation_mode
            if isinstance(evaluation_mode, CorporateActionEvaluationMode)
            else CorporateActionEvaluationMode(str(evaluation_mode))
        )
        effective_source = (
            source
            if source is not None
            else getattr(self, "source", "CANONICAL_PIT_CORPORATE_ACTION_LEDGER")
        )

        # Check coverage across all instruments
        all_complete = True
        all_blocking: list[str] = []
        events_count = 0

        for key in instrument_list:
            assessment = self.assess_window(
                key,
                research_start,
                research_end,
                policy=effective_policy,
                evaluation_mode=mode,
            )
            if not assessment.complete:
                all_complete = False
            for b in assessment.blocking_events:
                all_blocking.append(f"{key}:{b}")
            events_count += len(self.get_ex_post_events(key, research_start, research_end))

        if not all_complete:
            raise CorporateActionCoverageError(
                f"corporate-action evidence is incomplete for requested instruments: {', '.join(all_blocking)}"
            )

        return CorporateActionEvidenceIdentity(
            source=effective_source,
            complete=all_complete,
            blocking_events=tuple(all_blocking),
            evidence_fingerprint=self.fingerprint(),
            coverage_start=research_start,
            coverage_end=research_end,
            covered_instruments=instrument_list,
            events_count=events_count,
            policy_identity=effective_policy.policy_identity,
            authoritative=True,
            evaluation_mode=mode,
        )
