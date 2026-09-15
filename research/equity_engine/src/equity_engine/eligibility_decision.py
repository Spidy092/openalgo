"""Signed eligibility decision artifact for the autonomous trading bridge.

A decision says "strategy X is approved for live, capital N, instrument Y,
valid until date Z". It is the input the bridge consumes before any safety rail
runs, and it never places an order: this module has no order method, performs
no I/O, and never touches a broker, a database, or the network.

Safety contract (load-bearing):

- Every artifact carries ``live_orders_called=False``. Construction with any
  other value raises :class:`LiveOrderAttemptError`, and validation fails
  closed on any payload whose flag ``is not False`` (``True``, ``1``,
  ``"false"``, ``None`` and a missing key are all refusals, not defaults).
- Validation fails closed on: expired decisions, not-yet-valid decisions,
  wrong instrument, missing fields, capital over the hard cap, tampered
  fingerprints, naive timestamps, and non-mapping payloads.
- The fingerprint is deterministic: :func:`provenance.canonical_sha256` over a
  canonical payload, so re-signing an unchanged decision reproduces it byte
  for byte and any mutation breaks it.
- Money is :class:`Decimal`. Floats are refused rather than rounded, because a
  silently rounded capital cap is a wider permission than the approver signed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from .provenance import canonical_sha256

SCHEMA_VERSION = "openalgo-eligibility-decision/v1"

# Real-money testing cap from the bridge design (Phase 4: one symbol, Rs 1000).
# Raised later only by an explicit, reviewed change to this constant.
HARD_CAPITAL_CAP_RUPEES = Decimal(1000)

FINGERPRINT_KEY = "fingerprint"

_REQUIRED_FIELDS = (
    "strategy_id",
    "instrument_key",
    "track",
    "approved_capital_rupees",
    "per_order_notional_cap_rupees",
    "daily_loss_limit_rupees",
    "valid_from",
    "valid_until",
    "autonomy_mode",
)


class EligibilityError(ValueError):
    """Raised when an eligibility decision cannot be built or parsed."""


class LiveOrderAttemptError(RuntimeError):
    """Raised if any live-order execution is requested or simulated."""


class Track(StrEnum):
    INTRADAY = "intraday"
    SWING = "swing"


class AutonomyMode(StrEnum):
    SUPERVISED_AUTO = "supervised_auto"
    FULL_AUTO = "full_auto"


def _to_decimal(value: Any, field: str) -> Decimal:
    """Coerce an amount to Decimal, refusing bools, floats and non-finite values."""
    if isinstance(value, bool):
        raise EligibilityError(f"{field} must be a decimal amount, not {value!r}")
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, int):
        parsed = Decimal(value)
    elif isinstance(value, str):
        try:
            parsed = Decimal(value.strip())
        except (InvalidOperation, ValueError):
            raise EligibilityError(f"{field} must be a decimal amount, not {value!r}") from None
    else:
        raise EligibilityError(f"{field} must be a decimal amount, not {value!r}")
    if not parsed.is_finite():
        raise EligibilityError(f"{field} must be a finite amount, not {value!r}")
    return parsed


def _to_datetime(value: Any, field: str) -> datetime:
    """Coerce to an aware datetime; naive timestamps are refused, not assumed."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            raise EligibilityError(f"{field} must be an ISO-8601 datetime, not {value!r}") from None
    else:
        raise EligibilityError(f"{field} must be a datetime, not {value!r}")
    if parsed.tzinfo is None:
        raise EligibilityError(f"{field} must carry a timezone; naive timestamps are refused")
    return parsed


@dataclass(frozen=True)
class EligibilityDecision:
    """Immutable approval for one strategy on one instrument over one window."""

    strategy_id: str
    instrument_key: str
    track: Track
    approved_capital_rupees: Decimal
    per_order_notional_cap_rupees: Decimal
    daily_loss_limit_rupees: Decimal
    valid_from: datetime
    valid_until: datetime
    autonomy_mode: AutonomyMode
    schema_version: str = SCHEMA_VERSION
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.live_orders_called is not False:
            raise LiveOrderAttemptError(
                "eligibility decisions must carry live_orders_called=False; "
                "live orders are strictly forbidden in research"
            )
        strategy_id = self.strategy_id
        if not isinstance(strategy_id, str) or not strategy_id.strip():
            raise EligibilityError("strategy_id is required and must be non-blank")
        object.__setattr__(self, "strategy_id", strategy_id.strip())
        instrument_key = self.instrument_key
        if not isinstance(instrument_key, str) or not instrument_key.strip():
            raise EligibilityError("instrument_key is required and must be non-blank")
        object.__setattr__(self, "instrument_key", instrument_key.strip())

        try:
            track = (
                self.track
                if isinstance(self.track, Track)
                else Track(str(self.track).strip().lower())
            )
        except ValueError:
            raise EligibilityError(
                f"unknown track {self.track!r}; expected one of: "
                + ", ".join(item.value for item in Track)
            ) from None
        object.__setattr__(self, "track", track)

        try:
            autonomy = (
                self.autonomy_mode
                if isinstance(self.autonomy_mode, AutonomyMode)
                else AutonomyMode(str(self.autonomy_mode).strip().lower())
            )
        except ValueError:
            raise EligibilityError(
                f"unknown autonomy_mode {self.autonomy_mode!r}; expected one of: "
                + ", ".join(item.value for item in AutonomyMode)
            ) from None
        object.__setattr__(self, "autonomy_mode", autonomy)

        approved = _to_decimal(self.approved_capital_rupees, "approved_capital_rupees")
        if approved <= 0:
            raise EligibilityError("approved_capital_rupees must be positive")
        if approved > HARD_CAPITAL_CAP_RUPEES:
            raise EligibilityError(
                f"approved_capital_rupees {approved} exceeds the hard cap of "
                f"{HARD_CAPITAL_CAP_RUPEES}"
            )
        object.__setattr__(self, "approved_capital_rupees", approved)

        per_order = _to_decimal(self.per_order_notional_cap_rupees, "per_order_notional_cap_rupees")
        if per_order <= 0:
            raise EligibilityError("per_order_notional_cap_rupees must be positive")
        if per_order > approved:
            raise EligibilityError(
                f"per-order cap {per_order} exceeds approved capital {approved}; "
                "a single order must never be allowed to deploy more than approved"
            )
        object.__setattr__(self, "per_order_notional_cap_rupees", per_order)

        loss_limit = _to_decimal(self.daily_loss_limit_rupees, "daily_loss_limit_rupees")
        if loss_limit <= 0:
            raise EligibilityError("daily loss limit must be positive")
        if loss_limit > approved:
            raise EligibilityError(
                f"daily loss limit {loss_limit} exceeds approved capital {approved}"
            )
        object.__setattr__(self, "daily_loss_limit_rupees", loss_limit)

        valid_from = _to_datetime(self.valid_from, "valid_from")
        valid_until = _to_datetime(self.valid_until, "valid_until")
        if valid_from > valid_until:
            raise EligibilityError("valid_from must be on or before valid_until")
        object.__setattr__(self, "valid_from", valid_from)
        object.__setattr__(self, "valid_until", valid_until)

        if self.schema_version != SCHEMA_VERSION:
            raise EligibilityError(f"unsupported eligibility schema {self.schema_version!r}")

    def deterministic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "instrument_key": self.instrument_key,
            "track": self.track.value,
            "approved_capital_rupees": str(self.approved_capital_rupees),
            "per_order_notional_cap_rupees": str(self.per_order_notional_cap_rupees),
            "daily_loss_limit_rupees": str(self.daily_loss_limit_rupees),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "autonomy_mode": self.autonomy_mode.value,
            "live_orders_called": False,
        }

    def deterministic_fingerprint(self) -> str:
        return canonical_sha256(self.deterministic_payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self.deterministic_payload()
        payload[FINGERPRINT_KEY] = self.deterministic_fingerprint()
        return payload


@dataclass(frozen=True)
class EligibilityValidationResult:
    """Outcome of validating a signed decision payload. Fail-closed by default."""

    valid: bool
    violations: tuple[str, ...]
    fingerprint: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": list(self.violations),
            "fingerprint": self.fingerprint,
            "live_orders_called": False,
        }


def build_eligibility_decision(
    *,
    strategy_id: str,
    instrument_key: str,
    track: Track | str,
    approved_capital_rupees: Decimal | int | str,
    per_order_notional_cap_rupees: Decimal | int | str,
    daily_loss_limit_rupees: Decimal | int | str,
    valid_from: datetime | str,
    valid_until: datetime | str,
    autonomy_mode: AutonomyMode | str,
) -> EligibilityDecision:
    """Build a decision; every invariant violation raises :class:`EligibilityError`."""
    return EligibilityDecision(
        strategy_id=strategy_id,
        instrument_key=instrument_key,
        track=track,  # type: ignore[arg-type]
        approved_capital_rupees=approved_capital_rupees,  # type: ignore[arg-type]
        per_order_notional_cap_rupees=per_order_notional_cap_rupees,  # type: ignore[arg-type]
        daily_loss_limit_rupees=daily_loss_limit_rupees,  # type: ignore[arg-type]
        valid_from=valid_from,  # type: ignore[arg-type]
        valid_until=valid_until,  # type: ignore[arg-type]
        autonomy_mode=autonomy_mode,  # type: ignore[arg-type]
    )


def sign_eligibility_decision(decision: EligibilityDecision) -> dict[str, Any]:
    """Render a decision as a signed payload carrying its deterministic fingerprint."""
    if not isinstance(decision, EligibilityDecision):
        raise EligibilityError("only an EligibilityDecision can be signed")
    return decision.as_dict()


def eligibility_from_dict(payload: Mapping[str, Any]) -> EligibilityDecision:
    """Parse a signed payload back into a decision, failing closed on any anomaly."""
    if not isinstance(payload, Mapping):
        raise EligibilityError("eligibility payload must be a mapping")
    if "live_orders_called" not in payload:
        raise EligibilityError("live_orders_called is required and must be literal false")
    if payload["live_orders_called"] is not False:
        raise LiveOrderAttemptError(
            "eligibility payload reports live_orders_called="
            f"{payload['live_orders_called']!r}; live orders are strictly forbidden"
        )
    missing = [field for field in _REQUIRED_FIELDS if field not in payload]
    if missing:
        raise EligibilityError(
            "eligibility payload is missing required field(s): " + ", ".join(missing)
        )
    return EligibilityDecision(
        strategy_id=payload["strategy_id"],
        instrument_key=payload["instrument_key"],
        track=payload["track"],
        approved_capital_rupees=payload["approved_capital_rupees"],
        per_order_notional_cap_rupees=payload["per_order_notional_cap_rupees"],
        daily_loss_limit_rupees=payload["daily_loss_limit_rupees"],
        valid_from=payload["valid_from"],
        valid_until=payload["valid_until"],
        autonomy_mode=payload["autonomy_mode"],
    )


def validate_eligibility_decision(
    payload: Mapping[str, Any],
    *,
    now: datetime,
    expected_instrument_key: str | None = None,
    max_capital_rupees: Decimal | int | str = HARD_CAPITAL_CAP_RUPEES,
) -> EligibilityValidationResult:
    """Validate a signed payload against the clock, instrument and capital cap.

    Every refusal path returns ``valid=False`` with a human-readable violation;
    nothing here raises for a merely invalid payload.
    """
    violations: list[str] = []

    def invalid() -> EligibilityValidationResult:
        return EligibilityValidationResult(
            valid=False, violations=tuple(violations), fingerprint=None
        )

    if not isinstance(payload, Mapping):
        violations.append("eligibility payload must be a mapping")
        return invalid()
    if payload.get("live_orders_called") is not False:
        violations.append("live order guard: live_orders_called must be literal false; refusing")
        return invalid()

    try:
        decision = eligibility_from_dict(payload)
    except LiveOrderAttemptError as exc:
        violations.append(str(exc))
        return invalid()
    except EligibilityError as exc:
        violations.append(str(exc))
        return invalid()

    try:
        cap = _to_decimal(max_capital_rupees, "max_capital_rupees")
    except EligibilityError as exc:
        violations.append(str(exc))
        return invalid()
    if decision.approved_capital_rupees > cap:
        violations.append(
            f"approved capital {decision.approved_capital_rupees} exceeds the hard cap of {cap}"
        )

    if not isinstance(now, datetime) or now.tzinfo is None:
        violations.append("evaluation time must be a timezone-aware datetime")
        return EligibilityValidationResult(
            valid=False,
            violations=tuple(violations),
            fingerprint=decision.deterministic_fingerprint(),
        )
    if now < decision.valid_from:
        violations.append(
            f"decision is not yet valid; valid_from is {decision.valid_from.isoformat()}"
        )
    if now > decision.valid_until:
        violations.append(f"decision expired at {decision.valid_until.isoformat()}; refusing")

    if expected_instrument_key is not None and (
        not isinstance(expected_instrument_key, str)
        or expected_instrument_key.strip() != decision.instrument_key
    ):
        violations.append(
            f"instrument mismatch: decision approves {decision.instrument_key}, "
            f"but {expected_instrument_key!r} was requested"
        )

    expected_fingerprint = decision.deterministic_fingerprint()
    provided = payload.get(FINGERPRINT_KEY)
    if provided is None:
        violations.append("fingerprint is missing; refusing an unprovable decision")
    elif provided != expected_fingerprint:
        violations.append("fingerprint mismatch: the payload was altered after signing; refusing")

    return EligibilityValidationResult(
        valid=not violations,
        violations=tuple(violations),
        fingerprint=expected_fingerprint,
    )
