"""Pure safety-rails library for the autonomous trading bridge (Phase 2).

Each rail answers one question — may this order proceed? — with an explicit
allow/deny verdict carrying a machine-readable reason code. The rails are pure
functions of their inputs (plus explicit clock/registry arguments): no broker
calls, no network, no order placement, no database. The only filesystem touch
is the kill-switch existence check, which is the half of the switch that works
when the operator's shell is the only thing left.

Fail-closed throughout: any input a rail cannot evaluate (``None``, a bool
where money belongs, a non-finite number, a naive timestamp, an unknown probe
shape) is a denial, never a pass. UNKNOWN != ZERO: a missing observation stays
missing and refuses; it is never defaulted to zero.

Central-bank style money discipline: amounts are :class:`Decimal`. Floats are
refused rather than rounded, so a rounded cap cannot silently widen permission.

Rail order in :func:`evaluate_all_rails` is fixed and load-bearing: the
kill-switch and token precheck run before anything that claims shared state,
and the stateful rails (duplicate suppression, rate limit) run last so a
refused order never consumes another order's budget or poisons the duplicate
registry.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from .eligibility_decision import HARD_CAPITAL_CAP_RUPEES
from .provenance import canonical_sha256

if TYPE_CHECKING:
    from .market_sessions import NSEEquitySessionPolicy

IST = ZoneInfo("Asia/Kolkata")

# Bound for stateful registries in a worker that never restarts.
_MAX_TRACKED_KEYS = 512


class RailCode:
    OK = "ok"
    CAPITAL_CAP_EXCEEDED = "capital_cap_exceeded"
    PER_ORDER_CAP_EXCEEDED = "per_order_cap_exceeded"
    DAILY_LOSS_LIMIT_BREACHED = "daily_loss_limit_breached"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    COOLDOWN_ACTIVE = "cooldown_active"
    DUPLICATE_ORDER = "duplicate_order"
    INSTRUMENT_NOT_ALLOWED = "instrument_not_allowed"
    OUTSIDE_SESSION = "outside_session"
    CLOSE_BUFFER = "close_buffer"
    TOKEN_INVALID = "token_invalid"
    KILL_SWITCH_ENGAGED = "kill_switch_engaged"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class RailVerdict:
    """One rail's answer. ``allowed`` is the only field the bridge branches on."""

    allowed: bool
    code: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "reason": self.reason,
            "live_orders_called": False,
        }


def _allow(reason: str) -> RailVerdict:
    return RailVerdict(allowed=True, code=RailCode.OK, reason=reason)


def _deny(code: str, reason: str) -> RailVerdict:
    return RailVerdict(allowed=False, code=code, reason=reason)


def _refuse_unknown(reason: str) -> RailVerdict:
    return _deny(RailCode.INVALID_INPUT, reason)


def _as_decimal(value: Any) -> Decimal | None:
    """Coerce to Decimal, returning None for anything unusable (fail-closed)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        try:
            parsed = Decimal(value.strip())
        except (InvalidOperation, ValueError):
            return None
        return parsed if parsed.is_finite() else None
    return None


def check_total_capital(
    deployed_rupees: Any,
    approved_capital_rupees: Any,
    *,
    hard_cap_rupees: Any = HARD_CAPITAL_CAP_RUPEES,
) -> RailVerdict:
    """Deny when deployed capital exceeds the approval or the hard cap."""
    deployed = _as_decimal(deployed_rupees)
    approved = _as_decimal(approved_capital_rupees)
    hard = _as_decimal(hard_cap_rupees)
    if deployed is None or approved is None or hard is None:
        return _refuse_unknown("deployed, approved and hard-cap capital must all be known")
    if deployed < 0:
        return _refuse_unknown(f"deployed capital {deployed} is negative; refusing")
    if approved <= 0 or hard <= 0:
        return _refuse_unknown("approved and hard-cap capital must be positive")
    if approved > hard:
        return _deny(
            RailCode.CAPITAL_CAP_EXCEEDED,
            f"approved capital {approved} exceeds the hard cap of {hard}",
        )
    if deployed > approved:
        return _deny(
            RailCode.CAPITAL_CAP_EXCEEDED,
            f"deployed {deployed} exceeds approved capital {approved}",
        )
    if deployed > hard:
        return _deny(
            RailCode.CAPITAL_CAP_EXCEEDED,
            f"deployed {deployed} exceeds the hard cap of {hard}",
        )
    return _allow(f"deployed {deployed} is within approved capital {approved}")


def check_per_order_notional(notional_rupees: Any, per_order_cap_rupees: Any) -> RailVerdict:
    """Deny when one order's notional exceeds the per-order cap."""
    notional = _as_decimal(notional_rupees)
    cap = _as_decimal(per_order_cap_rupees)
    if notional is None or cap is None:
        return _refuse_unknown("order notional and per-order cap must both be known")
    if cap <= 0:
        return _refuse_unknown("per-order cap must be positive")
    if notional <= 0:
        return _refuse_unknown(f"order notional {notional} is not positive; refusing")
    if notional > cap:
        return _deny(
            RailCode.PER_ORDER_CAP_EXCEEDED,
            f"order notional {notional} exceeds the per-order cap of {cap}",
        )
    return _allow(f"order notional {notional} is within the per-order cap of {cap}")


def check_daily_loss_limit(
    realized_pnl_rupees: Any,
    unrealized_pnl_rupees: Any,
    daily_loss_limit_rupees: Any,
) -> RailVerdict:
    """Deny when the session loss reaches (not just exceeds) the daily limit."""
    realized = _as_decimal(realized_pnl_rupees)
    unrealized = _as_decimal(unrealized_pnl_rupees)
    limit = _as_decimal(daily_loss_limit_rupees)
    if realized is None or unrealized is None or limit is None:
        return _refuse_unknown("realized, unrealized P&L and the loss limit must all be known")
    if limit <= 0:
        return _refuse_unknown("daily loss limit must be positive")
    total = realized + unrealized
    if total <= -limit:
        return _deny(
            RailCode.DAILY_LOSS_LIMIT_BREACHED,
            f"session P&L {total} has reached the daily loss limit of {limit}",
        )
    return _allow(f"session P&L {total} is inside the daily loss limit of {limit}")


class RateLimiter:
    """Sliding-window order cap plus a minimum interval between orders.

    Stateful but side-effect free beyond its own deque: no clock reads except
    the injected one, no I/O. ``check_and_record`` both decides and claims, so
    a racing second approval cannot slip through the same window.
    """

    def __init__(
        self,
        *,
        max_orders: int,
        window_seconds: int,
        min_interval_seconds: int = 0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        for name, value in (
            ("max_orders", max_orders),
            ("window_seconds", window_seconds),
            ("min_interval_seconds", min_interval_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if max_orders < 1:
            raise ValueError("max_orders must be at least 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be at least 1")
        self._max_orders = max_orders
        self._window_seconds = window_seconds
        self._min_interval_seconds = min_interval_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._events: deque[datetime] = deque()
        self._last: datetime | None = None

    def _prune(self, now: datetime) -> None:
        cutoff = self._window_seconds
        while self._events and (now - self._events[0]).total_seconds() >= cutoff:
            self._events.popleft()

    def check_and_record(self) -> RailVerdict:
        """Allow and record one order, or deny without recording anything."""
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            return _refuse_unknown("rate-limiter clock must return an aware datetime")
        self._prune(now)
        if self._last is not None and now < self._last:
            return _refuse_unknown("clock moved backwards; refusing")
        if len(self._events) >= self._max_orders:
            return _deny(
                RailCode.RATE_LIMIT_EXCEEDED,
                f"{len(self._events)} orders already claimed inside "
                f"{self._window_seconds}s (max {self._max_orders})",
            )
        if self._last is not None:
            age = (now - self._last).total_seconds()
            if age < self._min_interval_seconds:
                return _deny(
                    RailCode.COOLDOWN_ACTIVE,
                    f"last order was {age:.1f}s ago; "
                    f"minimum interval is {self._min_interval_seconds}s",
                )
        self._events.append(now)
        self._last = now
        return _allow("within the order rate limit")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def generate_idempotency_key(payload: Mapping[str, Any]) -> str:
    """Return a deterministic key for one intended order.

    The same order fields always produce the same key, so a retry or a loop
    cannot double-fire; different fields produce a different key. Raises on
    empty or non-mapping payloads rather than minting a key for "nothing".
    """
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("idempotency payload must be a non-empty mapping")
    return canonical_sha256(_jsonable(dict(payload)))


class IdempotencyRegistry:
    """Duplicate detector: the first claim wins, every repeat is denied."""

    def __init__(self, *, max_tracked: int = _MAX_TRACKED_KEYS) -> None:
        # ValueError (not TypeError) matches this package's input-validation contract.
        if isinstance(max_tracked, bool) or not isinstance(max_tracked, int):
            raise ValueError("max_tracked must be an integer")  # noqa: TRY004
        if max_tracked < 1:
            raise ValueError("max_tracked must be at least 1")
        self._max_tracked = max_tracked
        self._seen: dict[str, None] = {}

    def is_seen(self, key: Any) -> bool:
        """Peek without claiming; unknown key shapes read as seen (fail-closed)."""
        if not isinstance(key, str) or not key.strip():
            return True
        return key in self._seen

    def check_and_claim(self, key: Any) -> RailVerdict:
        """Allow and record an unseen key, or deny a duplicate without recording."""
        if not isinstance(key, str) or not key.strip():
            return _refuse_unknown("idempotency key must be a non-blank string")
        if key in self._seen:
            return _deny(
                RailCode.DUPLICATE_ORDER,
                "this order was already claimed under the same idempotency key",
            )
        self._seen[key] = None
        while len(self._seen) > self._max_tracked:
            self._seen.pop(next(iter(self._seen)))
        return _allow("idempotency key claimed")


def check_instrument_allowlist(instrument_key: Any, allowlist: Any) -> RailVerdict:
    """Deny anything not on the pre-approved instrument allowlist."""
    if not isinstance(instrument_key, str) or not instrument_key.strip():
        return _refuse_unknown("instrument key must be a non-blank string")
    if allowlist is None or isinstance(allowlist, (str, bytes)):
        return _refuse_unknown("instrument allowlist must be a known collection")
    try:
        allowed = {str(item).strip().upper() for item in allowlist}
    except TypeError:
        return _refuse_unknown("instrument allowlist must be a known collection")
    allowed.discard("")
    if instrument_key.strip().upper() not in allowed:
        return _deny(
            RailCode.INSTRUMENT_NOT_ALLOWED,
            f"{instrument_key.strip()} is not on the approved instrument allowlist",
        )
    return _allow(f"{instrument_key.strip()} is on the approved instrument allowlist")


def check_session_gate(
    now: Any,
    session_policy: NSEEquitySessionPolicy,
    *,
    trade_date: date | None = None,
) -> RailVerdict:
    """Deny outside the continuous session and inside the close buffer (CAS-aware).

    ``now`` may carry any timezone; it is converted to IST before comparison
    because the policy's session times are IST wall times. Naive timestamps are
    denied rather than assumed to be IST.
    """
    if not isinstance(now, datetime) or now.tzinfo is None:
        return _refuse_unknown("session evaluation needs a timezone-aware timestamp")
    try:
        ist_now = now.astimezone(IST)
    except Exception:  # noqa: BLE001 -- any conversion failure must deny, not propagate
        return _refuse_unknown("session evaluation could not resolve IST; refusing")
    day = trade_date if trade_date is not None else ist_now.date()
    if not isinstance(day, date):
        return _refuse_unknown("trade date must be a date")
    try:
        start = session_policy.continuous_start(day)
        end = session_policy.continuous_end(day)
        cutoff = session_policy.exit_time(day)
    except Exception:  # noqa: BLE001 -- an unreadable policy denies, never passes
        return _refuse_unknown("session policy could not be evaluated; refusing")
    moment = ist_now.time()
    if not (start <= moment < end):
        return _deny(
            RailCode.OUTSIDE_SESSION,
            f"{moment.isoformat(timespec='seconds')} IST is outside the continuous "
            f"session {start.isoformat()}–{end.isoformat()} IST",
        )
    if moment >= cutoff:
        return _deny(
            RailCode.CLOSE_BUFFER,
            f"{moment.isoformat(timespec='seconds')} IST is inside the close buffer; "
            f"no new orders at or after {cutoff.isoformat()} IST",
        )
    return _allow(f"{moment.isoformat(timespec='seconds')} IST is inside the trading session")


@dataclass(frozen=True)
class TokenProbeResult:
    """Read-only token probe outcome, handed in by the caller.

    The rail never performs the probe itself: it judges a result the bridge
    obtained from a read-only endpoint. A probe the rail cannot read is a
    denial, because an unverified token must never carry an order.
    """

    token_valid: bool
    status_code: int | None
    checked_at: datetime | None = None
    detail: str = ""


def check_token_validity(probe: Any) -> RailVerdict:
    """Deny unless a well-formed probe positively confirms a working token."""
    if not isinstance(probe, TokenProbeResult):
        return _refuse_unknown("token probe result is missing or of unknown shape")
    if probe.token_valid is not True:
        return _deny(RailCode.TOKEN_INVALID, "token probe did not confirm a valid token")
    if (
        probe.status_code is None
        or isinstance(probe.status_code, bool)
        or not isinstance(probe.status_code, int)
        or not 200 <= probe.status_code <= 299
    ):
        return _deny(
            RailCode.TOKEN_INVALID,
            f"token probe status {probe.status_code!r} does not confirm a working token",
        )
    return _allow("token probe confirms a working token")


def check_kill_switch(engaged: Any, kill_switch_path: Any = None) -> RailVerdict:
    """Deny when the flag is set or the kill-switch file exists.

    An unreadable path denies rather than passes: a switch that cannot be read
    is a switch that cannot be proven clear.
    """
    if engaged is True:
        return _deny(
            RailCode.KILL_SWITCH_ENGAGED,
            "the kill switch is engaged; all automatic orders are blocked",
        )
    if engaged is not False:
        return _refuse_unknown("kill-switch flag state is unknown; refusing")
    if kill_switch_path is None:
        return _allow("kill switch is clear")
    try:
        present = Path(kill_switch_path).exists()
    except Exception:  # noqa: BLE001 -- an unreadable switch denies, never passes
        return _refuse_unknown(f"kill-switch path {kill_switch_path!r} could not be read; refusing")
    if present:
        return _deny(
            RailCode.KILL_SWITCH_ENGAGED,
            f"kill-switch file {kill_switch_path} exists; all automatic orders are blocked",
        )
    return _allow("kill switch is clear")


@dataclass
class RailsEvaluation:
    """Inputs for one combined pre-order evaluation (all money as Decimal)."""

    deployed_rupees: Any = None
    approved_capital_rupees: Any = None
    order_notional_rupees: Any = None
    per_order_cap_rupees: Any = None
    realized_pnl_rupees: Any = None
    unrealized_pnl_rupees: Any = None
    daily_loss_limit_rupees: Any = None
    instrument_key: Any = None
    instrument_allowlist: Any = field(default_factory=set)
    now_ist: Any = None
    session_policy: Any = None
    trade_date: date | None = None
    token_probe: Any = None
    kill_switch_engaged: Any = None
    kill_switch_path: Any = None
    idempotency_registry: IdempotencyRegistry | None = None
    idempotency_key: str | None = None
    rate_limiter: RateLimiter | None = None


def evaluate_all_rails(
    *,
    deployed_rupees: Any = None,
    approved_capital_rupees: Any = None,
    order_notional_rupees: Any = None,
    per_order_cap_rupees: Any = None,
    realized_pnl_rupees: Any = None,
    unrealized_pnl_rupees: Any = None,
    daily_loss_limit_rupees: Any = None,
    instrument_key: Any = None,
    instrument_allowlist: Any = None,
    now_ist: Any = None,
    session_policy: Any = None,
    trade_date: date | None = None,
    token_probe: Any = None,
    kill_switch_engaged: Any = None,
    kill_switch_path: Any = None,
    idempotency_registry: IdempotencyRegistry | None = None,
    idempotency_key: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> RailVerdict:
    """Run every rail in fixed order and return the first denial, else allow.

    Stateful rails claim last: the duplicate peek runs before the rate limiter,
    the rate limiter records, and the idempotency key is claimed only after
    every stateless rail has passed — so a refused order consumes nothing and
    poisons nothing. A supplied key without a registry (or vice versa) is a
    denial: dedupe that cannot be recorded is dedupe that does not exist.
    """
    verdict = check_kill_switch(kill_switch_engaged, kill_switch_path)
    if not verdict.allowed:
        return verdict
    verdict = check_token_validity(token_probe)
    if not verdict.allowed:
        return verdict
    verdict = check_session_gate(now_ist, session_policy, trade_date=trade_date)
    if not verdict.allowed:
        return verdict
    verdict = check_instrument_allowlist(instrument_key, instrument_allowlist)
    if not verdict.allowed:
        return verdict
    verdict = check_daily_loss_limit(
        realized_pnl_rupees, unrealized_pnl_rupees, daily_loss_limit_rupees
    )
    if not verdict.allowed:
        return verdict
    verdict = check_per_order_notional(order_notional_rupees, per_order_cap_rupees)
    if not verdict.allowed:
        return verdict
    verdict = check_total_capital(deployed_rupees, approved_capital_rupees)
    if not verdict.allowed:
        return verdict

    registry_given = idempotency_registry is not None
    key_given = idempotency_key is not None
    if registry_given != key_given:
        return _refuse_unknown(
            "idempotency needs both a registry and a key; refusing a half-wired dedupe"
        )
    if registry_given and key_given:
        assert idempotency_registry is not None and idempotency_key is not None
        if idempotency_registry.is_seen(idempotency_key):
            return _deny(
                RailCode.DUPLICATE_ORDER,
                "this order was already claimed under the same idempotency key",
            )
    if rate_limiter is not None:
        verdict = rate_limiter.check_and_record()
        if not verdict.allowed:
            return verdict
    if registry_given and key_given:
        assert idempotency_registry is not None and idempotency_key is not None
        verdict = idempotency_registry.check_and_claim(idempotency_key)
        if not verdict.allowed:
            return verdict
    return _allow("every safety rail passed")
