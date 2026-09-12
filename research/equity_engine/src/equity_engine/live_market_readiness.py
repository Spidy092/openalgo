"""Monday live-market readiness gate (shadow testing only).

Strict machine-readable aggregate that fails closed unless all required
evidence is available for the requested scope. This module is research/read-only:

- It never places, modifies, or cancels an order.
- It never accepts a broker token value: auth is PRESENT/ABSENT only.
- It reuses existing readiness, broker, market-context, session, cost and
  safety components instead of duplicating them.
- ``READY_FOR_LIVE_ORDER_REVIEW`` is NOT authorization to trade. Human/Product
  Owner approval remains separately required.

Readiness scopes (ascending):

- ``READY_FOR_SHADOW_INFRA``: live-market plumbing with theoretical decisions
  only. Proves token, read-only connectivity, session/calendar, clock, fresh
  quotes/feed without gaps, instrument identity with suspension/tradability
  evidence for consuming live prices, tick evidence, known CAS policy, valid
  session boundary, explicit approved shadow capital, kill switch off.
  Deliberately does NOT require prior paper evidence, strategy evidence,
  PIT/historical research evidence, or cost reconciliation: the first shadow
  session is what creates paper evidence, so requiring it here would be
  circular.
- ``READY_FOR_RESEARCH_SHADOW``: all infra requirements PLUS complete PIT
  evidence, complete historical dataset validation, valid selected research
  strategy/experiment evidence, and an explicit scenario cost
  identity/reconciliation policy. Still does NOT require prior paper/shadow
  evidence. This is the state required to start collecting real paper/shadow
  evidence.
- ``READY_FOR_LIVE_ORDER_REVIEW``: all research-shadow requirements PLUS
  sufficient accumulated paper/shadow evidence, passing broker cost
  reconciliation, static-IP/live prerequisites, broker balance at or above
  approved capital, and proven live tradability. Still NOT authorization to
  trade.

Reused components (imported, not reimplemented):

- :class:`UpstoxReadinessSnapshot` (read-only broker connectivity + static IP)
- :class:`QuoteBatchResult` (missing-quote detection)
- :class:`EquityInstrument` (identity, tradability, suspension, tick, CAS)
- :class:`TickSizeVerification` (tick-size evidence)
- :class:`NSEEquitySessionPolicy` (CAS eligibility + session boundary)
- :class:`CalendarEvidence` (exchange trading-session state)
- :class:`ApprovedCapital`, :class:`CostReconciliationEvidence`,
  :class:`PaperTradingEvidence` (capital, cost, paper evidence)
- :class:`HistoricalDatasetValidation` (historical evidence readiness)
- :class:`LiveOrderAttemptError` (live-order guard)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from .experiment import (
    ApprovedCapital,
    CostReconciliationEvidence,
    LiveOrderAttemptError,
    PaperTradingEvidence,
)
from .historical_validation import HistoricalDatasetValidation
from .instrument_master import EquityInstrument
from .market_sessions import NSEEquitySessionPolicy
from .nse_calendar import CalendarEvidence
from .provenance import canonical_sha256
from .suspension_identity import (
    AMBIGUOUS_EXACT,
    NO_SUSPENSION_RECORD,
    SUSPENDED_EXACT,
)
from .tick_size import TickSizeVerification
from .upstox_market_context import QuoteBatchResult
from .upstox_readiness import UpstoxReadinessSnapshot

SCHEMA_VERSION = "live-market-readiness/v2"
REQUIRED_TIMEZONE = "Asia/Kolkata"
DEFAULT_READINESS_MAX_AGE_SECONDS = 1800.0


def _safe_identity_value(value: Any) -> Any:
    """Convert evidence state to credential-free, JSON-stable identity values."""

    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return _safe_identity_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _safe_identity_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_safe_identity_value(item) for item in value]
    return value


def _identity_fingerprint(payload: Any) -> str:
    return canonical_sha256(_safe_identity_value(payload))


def _capital_identity(capital: ApprovedCapital | None) -> str | None:
    if capital is None:
        return None
    return _identity_fingerprint(
        {"amount_rupees": capital.amount_rupees, "currency": capital.currency}
    )


class ReadinessClassification(StrEnum):
    """Explicit final gate classification."""

    NOT_READY_FOR_SHADOW = "NOT_READY_FOR_SHADOW"
    READY_FOR_SHADOW_INFRA = "READY_FOR_SHADOW_INFRA"
    READY_FOR_RESEARCH_SHADOW = "READY_FOR_RESEARCH_SHADOW"
    READY_FOR_LIVE_ORDER_REVIEW = "READY_FOR_LIVE_ORDER_REVIEW"


@dataclass(frozen=True)
class LiveMarketReadinessContext:
    """Deterministic binding between a readiness report and one shadow run."""

    trade_date: date
    timezone_name: str
    instrument_keys: tuple[str, ...]
    instrument_context_fingerprint: str
    session_context_fingerprint: str | None
    capital_identity: str | None
    quote_feed_fingerprint: str
    quote_freshness_threshold_seconds: float
    readiness_max_age_seconds: float = DEFAULT_READINESS_MAX_AGE_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.trade_date, date):
            raise TypeError("context trade_date must be a date")
        if not self.timezone_name.strip():
            raise ValueError("context timezone_name is required")
        if not self.instrument_keys or len(set(self.instrument_keys)) != len(self.instrument_keys):
            raise ValueError("context instrument_keys must be non-empty and unique")
        if not self.instrument_context_fingerprint:
            raise ValueError("context instrument fingerprint is required")
        if not self.quote_feed_fingerprint:
            raise ValueError("context quote/feed fingerprint is required")
        if self.quote_freshness_threshold_seconds <= 0:
            raise ValueError("context quote freshness threshold must be positive")
        if self.readiness_max_age_seconds <= 0:
            raise ValueError("context readiness max age must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "timezone_name": self.timezone_name,
            "instrument_keys": list(self.instrument_keys),
            "instrument_context_fingerprint": self.instrument_context_fingerprint,
            "session_context_fingerprint": self.session_context_fingerprint,
            "capital_identity": self.capital_identity,
            "quote_feed_fingerprint": self.quote_feed_fingerprint,
            "quote_freshness_threshold_seconds": self.quote_freshness_threshold_seconds,
            "readiness_max_age_seconds": self.readiness_max_age_seconds,
        }

    def fingerprint(self) -> str:
        return _identity_fingerprint(self.as_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LiveMarketReadinessContext:
        try:
            return cls(
                trade_date=date.fromisoformat(str(payload["trade_date"])),
                timezone_name=str(payload["timezone_name"]),
                instrument_keys=tuple(str(item) for item in payload["instrument_keys"]),
                instrument_context_fingerprint=str(payload["instrument_context_fingerprint"]),
                session_context_fingerprint=(
                    str(payload["session_context_fingerprint"])
                    if payload.get("session_context_fingerprint") is not None
                    else None
                ),
                capital_identity=(
                    str(payload["capital_identity"])
                    if payload.get("capital_identity") is not None
                    else None
                ),
                quote_feed_fingerprint=str(payload["quote_feed_fingerprint"]),
                quote_freshness_threshold_seconds=float(
                    payload["quote_freshness_threshold_seconds"]
                ),
                readiness_max_age_seconds=float(
                    payload.get("readiness_max_age_seconds", DEFAULT_READINESS_MAX_AGE_SECONDS)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid readiness context") from exc


def _instrument_context_fingerprint(
    *, expected_instrument_keys: tuple[str, ...], instrument: EquityInstrument | None
) -> str:
    instrument_state: object = None
    if instrument is not None:
        instrument_state = {
            "instrument_key": instrument.instrument_key,
            "exchange": instrument.exchange,
            "segment": instrument.segment,
            "instrument_type": instrument.instrument_type,
            "cas_eligible": instrument.cas_eligible,
            "tick_size_rupees": instrument.tick_size_rupees,
        }
    return _identity_fingerprint(
        {
            "expected_instrument_keys": expected_instrument_keys,
            "instrument": instrument_state,
        }
    )


def _session_context_fingerprint(policy: NSEEquitySessionPolicy | None) -> str | None:
    if policy is None:
        return None
    return _identity_fingerprint(
        {
            "class": f"{type(policy).__module__}.{type(policy).__qualname__}",
            "state": policy,
        }
    )


def _quote_feed_fingerprint(
    *, quotes: QuoteBatchResult | None, feed: FeedHealthEvidence | None, threshold: float
) -> str:
    return _identity_fingerprint(
        {
            "quotes": quotes,
            "feed": feed,
            "quote_freshness_threshold_seconds": threshold,
        }
    )


# Reason codes: stable, human-readable, machine-checkable.
TOKEN_MISSING = "TOKEN_MISSING"
BROKER_CONNECTIVITY_FAILED = "BROKER_CONNECTIVITY_FAILED"
STATIC_IP_MISSING = "STATIC_IP_MISSING"
SESSION_CLOSED = "SESSION_CLOSED"
CALENDAR_EVIDENCE_MISSING = "CALENDAR_EVIDENCE_MISSING"
CLOCK_INVALID = "CLOCK_INVALID"
QUOTE_MISSING = "QUOTE_MISSING"
QUOTE_STALE = "QUOTE_STALE"
QUOTE_TIMESTAMP_INVALID = "QUOTE_TIMESTAMP_INVALID"
FEED_STATUS_UNKNOWN = "FEED_STATUS_UNKNOWN"
FEED_UNAVAILABLE = "FEED_UNAVAILABLE"
FEED_GAP_DETECTED = "FEED_GAP_DETECTED"
INSTRUMENT_IDENTITY_INCOMPLETE = "INSTRUMENT_IDENTITY_INCOMPLETE"
INSTRUMENT_NOT_TRADABLE = "INSTRUMENT_NOT_TRADABLE"
SUSPENSION_ACTIVE = "SUSPENSION_ACTIVE"
SUSPENSION_CONFLICT = "SUSPENSION_CONFLICT"
TICK_SIZE_UNVERIFIED = "TICK_SIZE_UNVERIFIED"
CAS_POLICY_UNKNOWN = "CAS_POLICY_UNKNOWN"
SESSION_BOUNDARY_VIOLATED = "SESSION_BOUNDARY_VIOLATED"
CAPITAL_NOT_APPROVED = "CAPITAL_NOT_APPROVED"
BROKER_BALANCE_UNKNOWN = "BROKER_BALANCE_UNKNOWN"
INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
COST_RECONCILIATION_MISSING = "COST_RECONCILIATION_MISSING"
COST_RECONCILIATION_FAILED = "COST_RECONCILIATION_FAILED"
PIT_EVIDENCE_INCOMPLETE = "PIT_EVIDENCE_INCOMPLETE"
HISTORICAL_EVIDENCE_MISSING = "HISTORICAL_EVIDENCE_MISSING"
STRATEGY_EVIDENCE_MISSING = "STRATEGY_EVIDENCE_MISSING"
PAPER_EVIDENCE_MISSING = "PAPER_EVIDENCE_MISSING"
KILL_SWITCH_ENGAGED = "KILL_SWITCH_ENGAGED"
LIVE_ORDERS_CALLED = "LIVE_ORDERS_CALLED"
LIVE_TRADABILITY_UNPROVEN = "LIVE_TRADABILITY_UNPROVEN"

REASON_MESSAGES: dict[str, str] = {
    TOKEN_MISSING: "broker credential is absent (PRESENT/ABSENT only; value never inspected)",
    BROKER_CONNECTIVITY_FAILED: "read-only broker connectivity check did not pass",
    STATIC_IP_MISSING: "primary static IP is not registered; live API orders remain blocked",
    SESSION_CLOSED: "exchange session is not open on a verified trading day",
    CALENDAR_EVIDENCE_MISSING: "sourced NSE calendar evidence is missing or inconsistent",
    CLOCK_INVALID: "market clock is not a timezone-aware Asia/Kolkata timestamp for the trade date",
    QUOTE_MISSING: "required quote is missing from the read-only quote batch",
    QUOTE_STALE: "quote age exceeds the caller-supplied freshness bound",
    QUOTE_TIMESTAMP_INVALID: "quote timestamp is missing, naive, or unparsable",
    FEED_STATUS_UNKNOWN: "websocket/data-feed availability was not supplied as explicit evidence",
    FEED_UNAVAILABLE: "websocket/data-feed reports unavailable",
    FEED_GAP_DETECTED: "data-feed gap or reconnect gap was detected",
    INSTRUMENT_IDENTITY_INCOMPLETE: "instrument identity fields are incomplete or inconsistent",
    INSTRUMENT_NOT_TRADABLE: "instrument is not currently tradable for intraday shadow",
    SUSPENSION_ACTIVE: "suspension evidence shows an exact or ambiguous active suspension",
    SUSPENSION_CONFLICT: "suspension evidence is contradictory and fails closed",
    TICK_SIZE_UNVERIFIED: "tick-size evidence is missing or did not verify",
    CAS_POLICY_UNKNOWN: "CAS eligibility or session-boundary policy is unknown or mismatched",
    SESSION_BOUNDARY_VIOLATED: "market time is outside the effective continuous session",
    CAPITAL_NOT_APPROVED: "approved capital is not explicitly configured",
    BROKER_BALANCE_UNKNOWN: "broker available-to-trade balance is unknown",
    INSUFFICIENT_BALANCE: "broker balance is below approved capital",
    COST_RECONCILIATION_MISSING: "broker cost-reconciliation evidence artifact is missing",
    COST_RECONCILIATION_FAILED: "cost reconciliation did not pass within tolerance",
    PIT_EVIDENCE_INCOMPLETE: "point-in-time membership evidence is incomplete",
    HISTORICAL_EVIDENCE_MISSING: "historical dataset validation evidence is missing or failed",
    STRATEGY_EVIDENCE_MISSING: "selected research strategy/experiment evidence is missing",
    PAPER_EVIDENCE_MISSING: "paper/shadow trading evidence artifact is missing",
    KILL_SWITCH_ENGAGED: "kill switch is engaged; gate fails closed",
    LIVE_ORDERS_CALLED: "live order execution was requested; strictly forbidden in readiness",
    LIVE_TRADABILITY_UNPROVEN: "live tradability is not proven for live-order review",
}

# Codes that block research-shadow and live review but NOT pure plumbing shadow.
# PIT/historical/strategy/cost evidence is required to run a research strategy,
# never to prove the live feed works.
RESEARCH_ONLY_CODES = frozenset(
    {
        PIT_EVIDENCE_INCOMPLETE,
        HISTORICAL_EVIDENCE_MISSING,
        STRATEGY_EVIDENCE_MISSING,
        COST_RECONCILIATION_MISSING,
        COST_RECONCILIATION_FAILED,
    }
)

# Codes that block live-order review only. Prior paper evidence, static IP,
# broker-balance knowledge/sufficiency and proven live tradability are
# live-review concerns; none is genuinely required for read-only feed access
# or theoretical shadow decisions.
LIVE_ONLY_CODES = frozenset(
    {
        PAPER_EVIDENCE_MISSING,
        STATIC_IP_MISSING,
        BROKER_BALANCE_UNKNOWN,
        INSUFFICIENT_BALANCE,
        LIVE_TRADABILITY_UNPROVEN,
    }
)

ALL_REASON_CODES = frozenset(REASON_MESSAGES)


@dataclass(frozen=True)
class FeedHealthEvidence:
    """Explicit websocket/data-feed health.

    ``available=None`` means unknown and fails closed. There is no implicit
    "feed is fine" default.
    """

    available: bool | None
    gap_detected: bool
    last_heartbeat_ist: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.gap_detected, bool):
            raise TypeError("gap_detected must be boolean")
        if self.last_heartbeat_ist is not None and self.last_heartbeat_ist.tzinfo is None:
            raise ValueError("last_heartbeat_ist must be timezone-aware when supplied")


@dataclass(frozen=True)
class LiveMarketReadinessInputs:
    """All evidence required by the gate. Every field is mandatory input.

    Nullable fields mean "explicitly missing evidence" and fail closed at the
    scope that requires them; they are not hidden defaults. Thresholds
    (``max_quote_age_seconds``, ``cost_tolerance_inr``) have no defaults and
    must be caller-supplied. Auth is a boolean only: this type cannot carry a
    token value, prefix, or length.
    """

    token_present: bool
    readiness_snapshot: UpstoxReadinessSnapshot | None
    now_ist: datetime
    timezone_name: str
    trade_date: date
    is_trading_day: bool
    session_open: bool
    session_policy: NSEEquitySessionPolicy | None
    calendar_evidence: CalendarEvidence | None
    quotes: QuoteBatchResult | None
    expected_instrument_keys: tuple[str, ...]
    max_quote_age_seconds: float
    feed: FeedHealthEvidence | None
    instrument: EquityInstrument | None
    tick_verification: TickSizeVerification | None
    approved_capital: ApprovedCapital | None
    broker_available_to_trade: Decimal | None
    cost_evidence: CostReconciliationEvidence | None
    cost_tolerance_inr: Decimal
    pit_complete: bool
    historical_validation: HistoricalDatasetValidation | None
    strategy_evidence_present: bool
    paper_evidence: PaperTradingEvidence | None
    kill_switch_engaged: bool
    live_orders_called: bool

    def __post_init__(self) -> None:
        if not isinstance(self.token_present, bool):
            raise TypeError("token_present must be boolean")
        if not isinstance(self.now_ist, datetime):
            raise TypeError("now_ist must be a datetime")
        if not isinstance(self.timezone_name, str) or not self.timezone_name.strip():
            raise ValueError("timezone_name is required")
        if not isinstance(self.trade_date, date):
            raise TypeError("trade_date must be a date")
        if not isinstance(self.is_trading_day, bool):
            raise TypeError("is_trading_day must be boolean")
        if not isinstance(self.session_open, bool):
            raise TypeError("session_open must be boolean")
        if not self.expected_instrument_keys:
            raise ValueError("expected_instrument_keys must be explicitly non-empty")
        if len(set(self.expected_instrument_keys)) != len(self.expected_instrument_keys):
            raise ValueError("expected_instrument_keys contain duplicates")
        if not isinstance(self.max_quote_age_seconds, (int, float)):
            raise TypeError("max_quote_age_seconds must be numeric")
        if not self.max_quote_age_seconds > 0:
            raise ValueError("max_quote_age_seconds must be positive")
        if not isinstance(self.cost_tolerance_inr, Decimal):
            raise TypeError("cost_tolerance_inr must be a Decimal")
        if not self.cost_tolerance_inr.is_finite() or self.cost_tolerance_inr < 0:
            raise ValueError("cost_tolerance_inr must be a non-negative finite Decimal")
        if not isinstance(self.pit_complete, bool):
            raise TypeError("pit_complete must be boolean")
        if not isinstance(self.strategy_evidence_present, bool):
            raise TypeError("strategy_evidence_present must be boolean")
        if not isinstance(self.kill_switch_engaged, bool):
            raise TypeError("kill_switch_engaged must be boolean")
        if not isinstance(self.live_orders_called, bool):
            raise TypeError("live_orders_called must be boolean")
        if self.broker_available_to_trade is not None and (
            not isinstance(self.broker_available_to_trade, Decimal)
            or not self.broker_available_to_trade.is_finite()
        ):
            raise ValueError("broker_available_to_trade must be a finite Decimal when supplied")

    def readiness_context(self) -> LiveMarketReadinessContext:
        """Return the canonical credential-free context bound to this evidence set."""

        return LiveMarketReadinessContext(
            trade_date=self.trade_date,
            timezone_name=self.timezone_name,
            instrument_keys=self.expected_instrument_keys,
            instrument_context_fingerprint=_instrument_context_fingerprint(
                expected_instrument_keys=self.expected_instrument_keys,
                instrument=self.instrument,
            ),
            session_context_fingerprint=_session_context_fingerprint(self.session_policy),
            capital_identity=_capital_identity(self.approved_capital),
            quote_feed_fingerprint=_quote_feed_fingerprint(
                quotes=self.quotes,
                feed=self.feed,
                threshold=self.max_quote_age_seconds,
            ),
            quote_freshness_threshold_seconds=self.max_quote_age_seconds,
        )


def build_runner_readiness_context(
    *,
    trade_date: date,
    timezone_name: str,
    instrument_keys: tuple[str, ...],
    cas_eligible_by_key: tuple[tuple[str, bool], ...],
    tick_size_by_key: tuple[tuple[str, str], ...],
    exit_buffer_minutes: int,
    approved_capital: ApprovedCapital | None,
    quote_freshness_threshold_seconds: float,
    readiness_max_age_seconds: float = DEFAULT_READINESS_MAX_AGE_SECONDS,
) -> LiveMarketReadinessContext:
    """Build the structural context a runner must match to a readiness report."""

    cas_by_key = dict(cas_eligible_by_key)
    tick_by_key = dict(tick_size_by_key)
    if len(instrument_keys) == 1:
        key = instrument_keys[0]
        instrument_state: object = {
            "instrument_key": key,
            "exchange": "NSE",
            "segment": "NSE_EQ",
            "instrument_type": "EQ",
            "cas_eligible": cas_by_key[key],
            "tick_size_rupees": tick_by_key[key],
        }
        policy_state: object = NSEEquitySessionPolicy(
            cas_eligible=cas_by_key[key],
            exit_buffer_minutes=exit_buffer_minutes,
        )
    else:
        instrument_state = [
            {
                "instrument_key": key,
                "exchange": "NSE",
                "segment": "NSE_EQ",
                "instrument_type": "EQ",
                "cas_eligible": cas_by_key[key],
                "tick_size_rupees": tick_by_key[key],
            }
            for key in instrument_keys
        ]
        policy_state = [
            NSEEquitySessionPolicy(
                cas_eligible=cas_by_key[key],
                exit_buffer_minutes=exit_buffer_minutes,
            )
            for key in instrument_keys
        ]
    return LiveMarketReadinessContext(
        trade_date=trade_date,
        timezone_name=timezone_name,
        instrument_keys=instrument_keys,
        instrument_context_fingerprint=_identity_fingerprint(
            {"expected_instrument_keys": instrument_keys, "instrument": instrument_state}
        ),
        session_context_fingerprint=_identity_fingerprint(
            {
                "class": f"{type(policy_state).__module__}.{type(policy_state).__qualname__}",
                "state": policy_state,
            }
        ),
        capital_identity=_capital_identity(approved_capital),
        quote_feed_fingerprint=_identity_fingerprint(
            {
                "source": "runner-declared-readiness-context",
                "instrument_keys": instrument_keys,
                "quote_freshness_threshold_seconds": quote_freshness_threshold_seconds,
            }
        ),
        quote_freshness_threshold_seconds=quote_freshness_threshold_seconds,
        readiness_max_age_seconds=readiness_max_age_seconds,
    )


def build_synthetic_readiness_report(
    *,
    checked_at_ist: datetime,
    trade_date: date,
    instrument_keys: tuple[str, ...],
    cas_eligible_by_key: tuple[tuple[str, bool], ...],
    tick_size_by_key: tuple[tuple[str, str], ...],
    exit_buffer_minutes: int,
    approved_capital: ApprovedCapital,
    quote_freshness_threshold_seconds: float,
    classification: ReadinessClassification = ReadinessClassification.READY_FOR_RESEARCH_SHADOW,
) -> LiveMarketReadinessReport:
    """Construct an explicit synthetic report for offline DRY_RUN rehearsal only."""

    context = build_runner_readiness_context(
        trade_date=trade_date,
        timezone_name=REQUIRED_TIMEZONE,
        instrument_keys=instrument_keys,
        cas_eligible_by_key=cas_eligible_by_key,
        tick_size_by_key=tick_size_by_key,
        exit_buffer_minutes=exit_buffer_minutes,
        approved_capital=approved_capital,
        quote_freshness_threshold_seconds=quote_freshness_threshold_seconds,
    )
    return LiveMarketReadinessReport(
        schema_version=SCHEMA_VERSION,
        classification=classification,
        reason_codes=(),
        reasons=(),
        checked_at_ist=checked_at_ist,
        trade_date=trade_date,
        context=context,
        approved_capital_rupees=approved_capital.amount_rupees,
        broker_available_to_trade=None,
        effective_capital_rupees=approved_capital.amount_rupees,
        live_orders_called=False,
    )


@dataclass(frozen=True)
class ReasonDetail:
    code: str
    message: str
    blocks_infra: bool
    blocks_research_shadow: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "blocks_infra": self.blocks_infra,
            "blocks_research_shadow": self.blocks_research_shadow,
        }


@dataclass(frozen=True)
class LiveMarketReadinessReport:
    """Aggregate gate result. Fails closed; credential-free."""

    schema_version: str
    classification: ReadinessClassification
    reason_codes: tuple[str, ...]
    reasons: tuple[ReasonDetail, ...]
    checked_at_ist: datetime
    trade_date: date
    context: LiveMarketReadinessContext
    approved_capital_rupees: Decimal | None
    broker_available_to_trade: Decimal | None
    effective_capital_rupees: Decimal | None
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {self.schema_version!r}")
        if not isinstance(self.classification, ReadinessClassification):
            raise TypeError("classification must be a ReadinessClassification")
        if self.live_orders_called:
            raise LiveOrderAttemptError("live orders are strictly forbidden in readiness")
        if tuple(sorted(set(self.reason_codes))) != tuple(sorted(self.reason_codes)):
            raise ValueError("reason_codes must be unique")
        for code in self.reason_codes:
            if code not in ALL_REASON_CODES:
                raise ValueError(f"unknown reason code {code!r}")
        reason_codes_from_details = tuple(item.code for item in self.reasons)
        if tuple(sorted(reason_codes_from_details)) != tuple(sorted(self.reason_codes)):
            raise ValueError("reasons must match reason_codes exactly")
        if not isinstance(self.checked_at_ist, datetime):
            raise TypeError("checked_at_ist must be a datetime")
        if self.context.trade_date != self.trade_date:
            raise ValueError("readiness context trade_date does not match report")
        if self.approved_capital_rupees is not None and self.context.capital_identity is None:
            raise ValueError("readiness context lacks approved-capital identity")

    @property
    def infra_ready(self) -> bool:
        return self.classification in (
            ReadinessClassification.READY_FOR_SHADOW_INFRA,
            ReadinessClassification.READY_FOR_RESEARCH_SHADOW,
            ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW,
        )

    @property
    def research_shadow_ready(self) -> bool:
        return self.classification in (
            ReadinessClassification.READY_FOR_RESEARCH_SHADOW,
            ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW,
        )

    @property
    def shadow_ready(self) -> bool:
        """Whether strategy shadow execution is permitted, not just infrastructure."""

        return self.research_shadow_ready

    @property
    def live_review_ready(self) -> bool:
        return self.classification is ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "classification": self.classification.value,
            "reason_codes": list(self.reason_codes),
            "reasons": [item.as_dict() for item in self.reasons],
            "checked_at_ist": self.checked_at_ist.isoformat(),
            "trade_date": self.trade_date.isoformat(),
            "context": self.context.as_dict(),
            "context_fingerprint": self.context.fingerprint(),
            "approved_capital_rupees": (
                format(self.approved_capital_rupees, "f")
                if self.approved_capital_rupees is not None
                else None
            ),
            "broker_available_to_trade": (
                format(self.broker_available_to_trade, "f")
                if self.broker_available_to_trade is not None
                else None
            ),
            "effective_capital_rupees": (
                format(self.effective_capital_rupees, "f")
                if self.effective_capital_rupees is not None
                else None
            ),
            "infra_ready": self.infra_ready,
            "research_shadow_ready": self.research_shadow_ready,
            "shadow_ready": self.shadow_ready,
            "live_review_ready": self.live_review_ready,
            "live_orders_called": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LiveMarketReadinessReport:
        """Load a persisted report while ignoring derived convenience booleans."""

        if payload.get("live_orders_called") is not False:
            raise ValueError("persisted readiness report has live-order activity")
        context_payload = payload.get("context")
        if not isinstance(context_payload, dict):
            raise TypeError("persisted readiness report lacks context")
        context = LiveMarketReadinessContext.from_dict(context_payload)
        if payload.get("context_fingerprint") != context.fingerprint():
            raise ValueError("persisted readiness context fingerprint mismatch")
        raw_reasons = payload.get("reasons")
        if not isinstance(raw_reasons, list):
            raise TypeError("persisted readiness report reasons are invalid")
        reasons = tuple(
            ReasonDetail(
                code=str(item["code"]),
                message=str(item["message"]),
                blocks_infra=bool(item["blocks_infra"]),
                blocks_research_shadow=bool(item["blocks_research_shadow"]),
            )
            for item in raw_reasons
            if isinstance(item, dict)
        )
        try:
            return cls(
                schema_version=str(payload["schema_version"]),
                classification=ReadinessClassification(str(payload["classification"])),
                reason_codes=tuple(str(item) for item in payload["reason_codes"]),
                reasons=reasons,
                checked_at_ist=datetime.fromisoformat(str(payload["checked_at_ist"])),
                trade_date=date.fromisoformat(str(payload["trade_date"])),
                context=context,
                approved_capital_rupees=(
                    Decimal(str(payload["approved_capital_rupees"]))
                    if payload.get("approved_capital_rupees") is not None
                    else None
                ),
                broker_available_to_trade=(
                    Decimal(str(payload["broker_available_to_trade"]))
                    if payload.get("broker_available_to_trade") is not None
                    else None
                ),
                effective_capital_rupees=(
                    Decimal(str(payload["effective_capital_rupees"]))
                    if payload.get("effective_capital_rupees") is not None
                    else None
                ),
                live_orders_called=False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid persisted readiness report") from exc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"


def _quote_age_seconds(*, quote_timestamp: datetime, now_ist: datetime) -> float:
    return (now_ist - quote_timestamp).total_seconds()


def _parse_quote_timestamp(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def evaluate_live_market_readiness(inputs: LiveMarketReadinessInputs) -> LiveMarketReadinessReport:
    """Evaluate the Monday live-market shadow readiness gate across three scopes.

    Fails closed per scope: infra-blocking evidence yields
    ``NOT_READY_FOR_SHADOW``; research-only gaps yield
    ``READY_FOR_SHADOW_INFRA``; live-only gaps yield
    ``READY_FOR_RESEARCH_SHADOW``. A fully evidenced run yields
    ``READY_FOR_LIVE_ORDER_REVIEW``, which is still NOT authorization to
    trade.

    Raises:
        LiveOrderAttemptError: if ``live_orders_called`` is True.
    """

    if inputs.live_orders_called:
        raise LiveOrderAttemptError("live orders are strictly forbidden in readiness")

    infra_blocking: list[str] = []
    research_blocking: list[str] = []
    live_blocking: list[str] = []

    def add(code: str) -> None:
        if code not in ALL_REASON_CODES:
            raise ValueError(f"unknown reason code {code!r}")
        if code in LIVE_ONLY_CODES:
            target = live_blocking
        elif code in RESEARCH_ONLY_CODES:
            target = research_blocking
        else:
            target = infra_blocking
        if (
            code not in infra_blocking
            and code not in research_blocking
            and code not in live_blocking
        ):
            target.append(code)

    # Auth: PRESENT/ABSENT only. The token value is never accepted here.
    if not inputs.token_present:
        add(TOKEN_MISSING)

    # Read-only broker connectivity + static IP (reuses UpstoxReadinessSnapshot).
    snapshot = inputs.readiness_snapshot
    if snapshot is None:
        add(BROKER_CONNECTIVITY_FAILED)
    else:
        non_ip_checks = [item for item in snapshot.checks if item.name != "primary_static_ip"]
        non_ip_failed = any(not item.passed for item in non_ip_checks)
        if non_ip_failed or not snapshot.passed and snapshot.primary_static_ip_configured:
            add(BROKER_CONNECTIVITY_FAILED)
        # Static IP is live-only: shadow may proceed without it.
        if not snapshot.primary_static_ip_configured:
            add(STATIC_IP_MISSING)

    # Market clock / timezone correctness.
    clock_ok = (
        inputs.now_ist.tzinfo is not None
        and inputs.timezone_name == REQUIRED_TIMEZONE
        and inputs.now_ist.date() == inputs.trade_date
    )
    if not clock_ok:
        add(CLOCK_INVALID)

    # Exchange / trading-session state (reuses CalendarEvidence).
    calendar = inputs.calendar_evidence
    if calendar is None:
        add(CALENDAR_EVIDENCE_MISSING)
    else:
        trading = set(calendar.trading_dates)
        holidays = set(calendar.holiday_dates)
        special = set(calendar.excluded_special_session_dates)
        if inputs.is_trading_day and inputs.trade_date not in trading:
            add(CALENDAR_EVIDENCE_MISSING)
        if (
            not inputs.is_trading_day
            or inputs.trade_date in holidays
            or inputs.trade_date in special
        ):
            add(SESSION_CLOSED)
    if not inputs.is_trading_day or not inputs.session_open:
        add(SESSION_CLOSED)

    # CAS eligibility / session boundary (reuses NSEEquitySessionPolicy).
    policy = inputs.session_policy
    instrument = inputs.instrument
    if policy is None:
        add(CAS_POLICY_UNKNOWN)
    if instrument is not None and instrument.cas_eligible is None:
        add(CAS_POLICY_UNKNOWN)
    if (
        policy is not None
        and instrument is not None
        and instrument.cas_eligible is not None
        and policy.cas_eligible != instrument.cas_eligible
    ):
        add(CAS_POLICY_UNKNOWN)
    if policy is not None and clock_ok:
        try:
            start = policy.continuous_start(inputs.trade_date)
            end = policy.continuous_end(inputs.trade_date)
            exit_time = policy.exit_time(inputs.trade_date)
            now_time = inputs.now_ist.time()
            # Continuous session is [start, end); shadow must still be inside
            # the manageable window [start, exit_time].
            if not (start <= now_time <= exit_time):
                add(SESSION_BOUNDARY_VIOLATED)
            # Defensive: a policy whose end precedes start is misconfigured.
            if end <= start:
                add(CAS_POLICY_UNKNOWN)
        except ValueError:
            add(CAS_POLICY_UNKNOWN)

    # Quote freshness + missing/stale detection (reuses QuoteBatchResult).
    quotes = inputs.quotes
    if quotes is None:
        add(QUOTE_MISSING)
    else:
        for key in inputs.expected_instrument_keys:
            if key not in quotes.quotes:
                add(QUOTE_MISSING)
                continue
            raw_quote = quotes.quotes[key]
            if not isinstance(raw_quote, dict):
                add(QUOTE_TIMESTAMP_INVALID)
                continue
            parsed = _parse_quote_timestamp(raw_quote.get("timestamp"))
            if parsed is None:
                add(QUOTE_TIMESTAMP_INVALID)
                continue
            try:
                age = _quote_age_seconds(quote_timestamp=parsed, now_ist=inputs.now_ist)
            except TypeError:
                # Naive market clock cannot be compared to an aware quote
                # timestamp; CLOCK_INVALID already fails closed above.
                add(QUOTE_TIMESTAMP_INVALID)
                continue
            if age < 0 or age > inputs.max_quote_age_seconds:
                add(QUOTE_STALE)

    # Websocket / data-feed availability + reconnect/gap detection.
    feed = inputs.feed
    if feed is None:
        add(FEED_STATUS_UNKNOWN)
    else:
        if feed.available is None:
            add(FEED_STATUS_UNKNOWN)
        elif feed.available is False:
            add(FEED_UNAVAILABLE)
        if feed.gap_detected:
            add(FEED_GAP_DETECTED)

    # Instrument identity, tradability, suspension (reuses EquityInstrument).
    if instrument is None:
        add(INSTRUMENT_IDENTITY_INCOMPLETE)
    else:
        identity_ok = bool(
            instrument.instrument_key.strip()
            and instrument.isin.strip()
            and instrument.trading_symbol.strip()
            and instrument.segment == "NSE_EQ"
            and instrument.exchange == "NSE"
            and instrument.instrument_type == "EQ"
            and instrument.tick_size_rupees > 0
            and instrument.lot_size > 0
        )
        if not identity_ok:
            add(INSTRUMENT_IDENTITY_INCOMPLETE)
        status = instrument.suspension_status
        if status in (SUSPENDED_EXACT, AMBIGUOUS_EXACT):
            add(SUSPENSION_ACTIVE)
        elif status == NO_SUSPENSION_RECORD:
            pass
        elif status == "SUSPENSION_CONFLICT":
            add(SUSPENSION_CONFLICT)
        else:
            # Unknown status strings fail closed as a conflict.
            add(SUSPENSION_CONFLICT)
        if status == NO_SUSPENSION_RECORD and instrument.suspended:
            add(SUSPENSION_ACTIVE)
        # Shadow requires a currently non-suspended MIS-eligible instrument.
        if status != NO_SUSPENSION_RECORD or not instrument.mis_eligible:
            add(INSTRUMENT_NOT_TRADABLE)
        # Live review additionally requires proven tradability.
        if status != NO_SUSPENSION_RECORD or not instrument.live_tradability_proven:
            add(LIVE_TRADABILITY_UNPROVEN)

    # Tick-size evidence (reuses TickSizeVerification).
    if inputs.tick_verification is None or not inputs.tick_verification.passed:
        add(TICK_SIZE_UNVERIFIED)

    # Approved capital is explicit; broker balance can never raise it.
    # Balance knowledge/sufficiency is a live-review concern only: infra and
    # research shadow run theoretical decisions on approved capital.
    approved = (
        inputs.approved_capital.amount_rupees if inputs.approved_capital is not None else None
    )
    if inputs.approved_capital is None:
        add(CAPITAL_NOT_APPROVED)
    broker_balance = inputs.broker_available_to_trade
    if broker_balance is None:
        add(BROKER_BALANCE_UNKNOWN)
    effective: Decimal | None = None
    if approved is not None and broker_balance is not None:
        effective = min(approved, broker_balance)
        # Invariant: broker balance cannot increase approved capital.
        effective = min(effective, approved)
        if broker_balance < approved:
            add(INSUFFICIENT_BALANCE)
    elif approved is not None:
        effective = approved

    # Scenario cost identity/reconciliation policy (reuses
    # CostReconciliationEvidence). Required for research shadow and live
    # review; pure plumbing shadow has no cost-policy requirement.
    cost = inputs.cost_evidence
    if cost is None:
        add(COST_RECONCILIATION_MISSING)
    else:
        if cost.status != "PASS" or cost.max_reconciliation_error_inr > inputs.cost_tolerance_inr:
            add(COST_RECONCILIATION_FAILED)

    # Historical research evidence: PIT completeness plus dataset validation.
    # Required for research shadow and live review, never for pure infra.
    if not inputs.pit_complete:
        add(PIT_EVIDENCE_INCOMPLETE)
    validation = inputs.historical_validation
    if validation is None or not validation.passed:
        add(HISTORICAL_EVIDENCE_MISSING)

    # Selected research strategy/experiment evidence. Required for research
    # shadow and live review, never for pure infra plumbing.
    if not inputs.strategy_evidence_present:
        add(STRATEGY_EVIDENCE_MISSING)

    # Paper/shadow evidence readiness (reuses PaperTradingEvidence). Required
    # for live-order review only: the first shadow sessions are what create
    # this evidence, so requiring it earlier would be circular.
    if inputs.paper_evidence is None:
        add(PAPER_EVIDENCE_MISSING)

    # Kill / fail-closed state.
    if inputs.kill_switch_engaged:
        add(KILL_SWITCH_ENGAGED)

    if infra_blocking:
        classification = ReadinessClassification.NOT_READY_FOR_SHADOW
    elif research_blocking:
        classification = ReadinessClassification.READY_FOR_SHADOW_INFRA
    elif live_blocking:
        classification = ReadinessClassification.READY_FOR_RESEARCH_SHADOW
    else:
        classification = ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW

    ordered_codes = tuple(sorted(set(infra_blocking) | set(research_blocking) | set(live_blocking)))
    reasons = tuple(
        ReasonDetail(
            code=code,
            message=REASON_MESSAGES[code],
            blocks_infra=code not in RESEARCH_ONLY_CODES and code not in LIVE_ONLY_CODES,
            blocks_research_shadow=code not in LIVE_ONLY_CODES,
        )
        for code in ordered_codes
    )

    return LiveMarketReadinessReport(
        schema_version=SCHEMA_VERSION,
        classification=classification,
        reason_codes=ordered_codes,
        reasons=reasons,
        checked_at_ist=inputs.now_ist,
        trade_date=inputs.trade_date,
        context=inputs.readiness_context(),
        approved_capital_rupees=approved,
        broker_available_to_trade=broker_balance,
        effective_capital_rupees=effective,
        live_orders_called=False,
    )


def report_as_dict(report: LiveMarketReadinessReport) -> dict[str, Any]:
    """Return a credential-free JSON-safe mapping of the report."""

    return report.to_dict()


def report_as_json(report: LiveMarketReadinessReport) -> str:
    """Return canonical JSON for the report (credential-free)."""

    return report.to_json()


__all__ = [
    "ALL_REASON_CODES",
    "BROKER_BALANCE_UNKNOWN",
    "BROKER_CONNECTIVITY_FAILED",
    "CALENDAR_EVIDENCE_MISSING",
    "CAPITAL_NOT_APPROVED",
    "CAS_POLICY_UNKNOWN",
    "CLOCK_INVALID",
    "COST_RECONCILIATION_FAILED",
    "COST_RECONCILIATION_MISSING",
    "DEFAULT_READINESS_MAX_AGE_SECONDS",
    "FEED_GAP_DETECTED",
    "FEED_STATUS_UNKNOWN",
    "FEED_UNAVAILABLE",
    "HISTORICAL_EVIDENCE_MISSING",
    "INSTRUMENT_IDENTITY_INCOMPLETE",
    "INSTRUMENT_NOT_TRADABLE",
    "INSUFFICIENT_BALANCE",
    "KILL_SWITCH_ENGAGED",
    "LIVE_ONLY_CODES",
    "LIVE_ORDERS_CALLED",
    "LIVE_TRADABILITY_UNPROVEN",
    "PAPER_EVIDENCE_MISSING",
    "PIT_EVIDENCE_INCOMPLETE",
    "QUOTE_MISSING",
    "QUOTE_STALE",
    "QUOTE_TIMESTAMP_INVALID",
    "REASON_MESSAGES",
    "REQUIRED_TIMEZONE",
    "RESEARCH_ONLY_CODES",
    "SCHEMA_VERSION",
    "SESSION_BOUNDARY_VIOLATED",
    "SESSION_CLOSED",
    "STATIC_IP_MISSING",
    "STRATEGY_EVIDENCE_MISSING",
    "SUSPENSION_ACTIVE",
    "SUSPENSION_CONFLICT",
    "TICK_SIZE_UNVERIFIED",
    "TOKEN_MISSING",
    "FeedHealthEvidence",
    "LiveMarketReadinessContext",
    "LiveMarketReadinessInputs",
    "LiveMarketReadinessReport",
    "ReadinessClassification",
    "ReasonDetail",
    "build_runner_readiness_context",
    "build_synthetic_readiness_report",
    "evaluate_live_market_readiness",
    "report_as_dict",
    "report_as_json",
]
