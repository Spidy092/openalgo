"""Deterministic research-window plan compiler.

Transforms an overall research range into immutable chronological windows:

- train
- validation
- untouched test
- embargo gaps between them

PLAN artifacts only. No strategy execution, no profitability computation, no
performance thresholds, no optimal window inference, no broker calls, no data
downloads. Window durations, embargo length, and minimum observations are
explicit caller inputs; there are no hidden defaults.

Ordering guarantees (enforced by construction and by plan guards):

- Chronological only; no random split exists in this module.
- Train, validation, and untouched test never overlap; embargo gaps separate
  them so no overlap occurs through the embargo.
- The full train universe is frozen at compile time, before any future test
  loading or selection step can run.
- The final untouched test cannot enter strategy or stock selection:
  :meth:`ResearchWindowPlan.select_train_winner` accepts only train and
  validation dates and rejects any test or embargo date.
- A train winner always includes stock (instrument key), strategy name, and
  parameters.
- The untouched test authorization (:meth:`ResearchWindowPlan.authorize_untouched_test`)
  receives the full frozen train universe and fails closed otherwise.
- Point-in-time membership is determined per date:
  :meth:`ResearchWindowPlan.check_pit_membership` rejects evidence dated after
  the trade date.

Identity guarantees:

- Cost evidence fingerprint, approved capital, session policy, CAS policy,
  corporate-action evidence, dataset fingerprints, universe fingerprint, PIT
  membership fingerprint, train universe instruments, and every window boundary
  are bound into the deterministic fingerprint. Changing any of them changes
  the plan identity.
- Volatile ``created_at`` metadata is excluded from identity.
- Invalid or too-short windows fail closed with :class:`ResearchWindowError`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "openalgo-research-window-plan/v1"


class ResearchWindowError(ValueError):
    """Base failure for invalid research-window plans."""


class WindowTooShortError(ResearchWindowError):
    """Raised when the supplied calendar cannot satisfy requested durations."""


class WindowLeakageError(ResearchWindowError):
    """Raised when windows overlap or embargo separation is violated."""


class UntouchedTestViolationError(ResearchWindowError):
    """Raised when the untouched test enters selection or is bypassed."""


class FrozenUniverseViolationError(ResearchWindowError):
    """Raised when the frozen train universe is missing, partial, or replaced."""


class WindowRole(StrEnum):
    """Immutable role for one compiled window."""

    TRAIN = "train"
    VALIDATION = "validation"
    EMBARGO = "embargo"
    UNTOUCHED_TEST = "untouched_test"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _canonical_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(v) for v in value]
    if hasattr(value, "as_dict"):
        return _canonical_value(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict as _asdict

        return _canonical_value(_asdict(value))
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return canonical sorted compact JSON bytes for hashing."""
    canonical = {str(k): _canonical_value(v) for k, v in sorted(payload.items())}
    return json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Return hex SHA-256 over the canonical JSON payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_nonempty_str(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchWindowError(f"{name} is required and must be a non-empty string")
    return value


def _require_hex_digest(name: str, value: str) -> str:
    _require_nonempty_str(name, value)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ResearchWindowError(f"{name} must be a 64-character lowercase hex digest")
    return value


@dataclass(frozen=True)
class CompiledWindow:
    """One immutable chronological window slice."""

    role: WindowRole
    window_id: str
    start: date | None
    end: date | None
    trading_dates: tuple[date, ...]

    def __post_init__(self) -> None:
        if not self.window_id.strip():
            raise ResearchWindowError("window_id is required")
        if self.role is WindowRole.EMBARGO:
            if not self.trading_dates and (self.start is not None or self.end is not None):
                raise ResearchWindowError("empty embargo window must have null bounds")
            if self.trading_dates and (
                self.start != self.trading_dates[0] or self.end != self.trading_dates[-1]
            ):
                raise ResearchWindowError("embargo bounds must match its trading dates")
        else:
            if not self.trading_dates:
                raise ResearchWindowError(f"{self.role.value} window must not be empty")
            if self.start != self.trading_dates[0] or self.end != self.trading_dates[-1]:
                raise ResearchWindowError("window bounds must match its trading dates")

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "window_id": self.window_id,
            "start": self.start.isoformat() if self.start is not None else None,
            "end": self.end.isoformat() if self.end is not None else None,
            "trading_dates": [d.isoformat() for d in self.trading_dates],
            "trading_day_count": len(self.trading_dates),
        }


@dataclass(frozen=True)
class FrozenTrainUniverse:
    """Full train universe frozen before any future test access."""

    instruments: tuple[str, ...]
    universe_fingerprint: str
    pit_membership_fingerprint: str
    frozen_as_of: date

    def __post_init__(self) -> None:
        if not self.instruments:
            raise FrozenUniverseViolationError("frozen train universe must not be empty")
        if tuple(sorted(set(self.instruments))) != self.instruments:
            raise FrozenUniverseViolationError(
                "frozen train universe must be sorted unique instrument keys"
            )
        for key in self.instruments:
            if not key.strip():
                raise FrozenUniverseViolationError("instrument key must be non-empty")
        _require_nonempty_str("universe_fingerprint", self.universe_fingerprint)
        _require_nonempty_str("pit_membership_fingerprint", self.pit_membership_fingerprint)

    def as_dict(self) -> dict[str, Any]:
        return {
            "instruments": list(self.instruments),
            "universe_fingerprint": self.universe_fingerprint,
            "pit_membership_fingerprint": self.pit_membership_fingerprint,
            "frozen_as_of": self.frozen_as_of.isoformat(),
        }


@dataclass(frozen=True)
class TrainWinner:
    """Plan-level train winner. Always stock + strategy + params. No metrics."""

    instrument_key: str
    strategy_name: str
    parameters: dict[str, str]

    def __post_init__(self) -> None:
        if not self.instrument_key.strip():
            raise ResearchWindowError("winner instrument_key (stock) is required")
        if not self.strategy_name.strip():
            raise ResearchWindowError("winner strategy_name is required")
        if not self.parameters:
            raise ResearchWindowError("winner parameters must not be empty")
        for key, value in self.parameters.items():
            if not str(key).strip() or not str(value).strip():
                raise ResearchWindowError("winner parameters must be non-empty strings")

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument_key": self.instrument_key,
            "strategy_name": self.strategy_name,
            "parameters": dict(sorted(self.parameters.items())),
        }


@dataclass(frozen=True)
class ResearchWindowPlan:
    """Immutable research-window plan. No execution capability."""

    schema_version: str
    research_start: date
    research_end: date
    train: CompiledWindow
    validation: CompiledWindow
    embargo_after_train: CompiledWindow
    embargo_after_validation: CompiledWindow
    untouched_test: CompiledWindow
    train_trading_days: int
    validation_trading_days: int
    test_trading_days: int
    embargo_trading_days: int
    min_observations_per_window: int
    cost_evidence_fingerprint: str
    approved_capital_rupees: Decimal
    session_policy_id: str
    cas_policy_id: str
    corporate_action_evidence_fingerprint: str
    dataset_fingerprints: dict[str, str]
    universe_fingerprint: str
    pit_membership_fingerprint: str
    frozen_train_universe: FrozenTrainUniverse
    created_at: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ResearchWindowError(f"unsupported schema {self.schema_version!r}")
        if self.research_start > self.research_end:
            raise ResearchWindowError("research_start must be on or before research_end")
        for window, expected in (
            (self.train, WindowRole.TRAIN),
            (self.validation, WindowRole.VALIDATION),
            (self.embargo_after_train, WindowRole.EMBARGO),
            (self.embargo_after_validation, WindowRole.EMBARGO),
            (self.untouched_test, WindowRole.UNTOUCHED_TEST),
        ):
            if window.role is not expected:
                raise ResearchWindowError(f"expected {expected.value} window")
        ordered = [
            self.train,
            self.embargo_after_train,
            self.validation,
            self.embargo_after_validation,
            self.untouched_test,
        ]
        nonempty = [w for w in ordered if w.trading_dates]
        for first, second in zip(nonempty, nonempty[1:], strict=False):
            assert first.end is not None and second.start is not None
            if first.end >= second.start:
                raise WindowLeakageError(
                    f"{first.window_id} ending {first.end} must be strictly before "
                    f"{second.window_id} starting {second.start}"
                )
        test_dates = set(self.untouched_test.trading_dates)
        for window in (self.train, self.validation):
            if test_dates & set(window.trading_dates):
                raise WindowLeakageError("untouched test dates must never appear in train")
        if self.approved_capital_rupees <= Decimal(0):
            raise ResearchWindowError("approved_capital_rupees must be positive")

    def deterministic_payload(self) -> dict[str, Any]:
        """Canonical identity payload. Volatile created_at is excluded."""
        return {
            "schema_version": self.schema_version,
            "research_start": self.research_start.isoformat(),
            "research_end": self.research_end.isoformat(),
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "embargo_after_train": self.embargo_after_train.as_dict(),
            "embargo_after_validation": self.embargo_after_validation.as_dict(),
            "untouched_test": self.untouched_test.as_dict(),
            "train_trading_days": self.train_trading_days,
            "validation_trading_days": self.validation_trading_days,
            "test_trading_days": self.test_trading_days,
            "embargo_trading_days": self.embargo_trading_days,
            "min_observations_per_window": self.min_observations_per_window,
            "cost_evidence_fingerprint": self.cost_evidence_fingerprint,
            "approved_capital_rupees": str(self.approved_capital_rupees),
            "session_policy_id": self.session_policy_id,
            "cas_policy_id": self.cas_policy_id,
            "corporate_action_evidence_fingerprint": (self.corporate_action_evidence_fingerprint),
            "dataset_fingerprints": dict(sorted(self.dataset_fingerprints.items())),
            "universe_fingerprint": self.universe_fingerprint,
            "pit_membership_fingerprint": self.pit_membership_fingerprint,
            "frozen_train_universe": self.frozen_train_universe.as_dict(),
        }

    def fingerprint(self) -> str:
        """Deterministic SHA-256 over the canonical plan payload."""
        return canonical_sha256(self.deterministic_payload())

    @property
    def plan_id(self) -> str:
        """Deterministic plan identity derived from the fingerprint."""
        return f"rwp_{self.fingerprint()[:16]}"

    def to_dict(self) -> dict[str, Any]:
        """Full representation including identity and volatile metadata."""
        payload = self.deterministic_payload()
        payload["plan_id"] = self.plan_id
        payload["fingerprint"] = self.fingerprint()
        payload["created_at"] = self.created_at
        return payload

    def to_json(self) -> str:
        """Canonical JSON with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"

    def to_experiment_inputs(self) -> dict[str, Any]:
        """Map compiled windows onto the immutable experiment window schema.

        Returns plain ``ResearchWindowConfig`` / ``WindowSpec`` / ``EmbargoSpec``
        compatible mappings so callers can build an :class:`ExperimentArtifact`
        without re-deriving boundaries. No data is loaded here.
        """
        return {
            "research_window": {
                "start": self.research_start.isoformat(),
                "end": self.research_end.isoformat(),
            },
            "train_windows": [
                {
                    "window_id": 1,
                    "start": self.train.start.isoformat() if self.train.start else None,
                    "end": self.train.end.isoformat() if self.train.end else None,
                    "trading_days": len(self.train.trading_dates),
                }
            ],
            "validation_test_windows": [
                {
                    "window_id": 1,
                    "start": self.validation.start.isoformat() if self.validation.start else None,
                    "end": self.validation.end.isoformat() if self.validation.end else None,
                    "trading_days": len(self.validation.trading_dates),
                },
                {
                    "window_id": 2,
                    "start": self.untouched_test.start.isoformat()
                    if self.untouched_test.start
                    else None,
                    "end": self.untouched_test.end.isoformat() if self.untouched_test.end else None,
                    "trading_days": len(self.untouched_test.trading_dates),
                },
            ],
            "embargo": {"trading_days": self.embargo_trading_days},
        }

    def selection_universe(self) -> FrozenTrainUniverse:
        """Return the frozen train universe that selection must use."""
        return self.frozen_train_universe

    def select_train_winner(
        self,
        *,
        instrument_key: str,
        strategy_name: str,
        parameters: Mapping[str, str],
        selection_dates: tuple[date, ...],
    ) -> TrainWinner:
        """Record a train winner without executing anything.

        Fails closed when the winner is incomplete, when the stock is outside
        the frozen universe, or when any selection date touches embargo or the
        untouched test window.
        """
        if not instrument_key.strip():
            raise ResearchWindowError("instrument_key (stock) is required")
        if instrument_key not in set(self.frozen_train_universe.instruments):
            raise FrozenUniverseViolationError(
                f"{instrument_key!r} is not in the frozen train universe"
            )
        forbidden = set(self.embargo_after_train.trading_dates) | set(
            self.embargo_after_validation.trading_dates
        )
        forbidden |= set(self.untouched_test.trading_dates)
        leaked = [d for d in selection_dates if d in forbidden]
        if leaked:
            raise UntouchedTestViolationError(
                "selection dates must never include embargo or untouched test dates; "
                f"leaked {sorted(d.isoformat() for d in leaked)}"
            )
        allowed = set(self.train.trading_dates) | set(self.validation.trading_dates)
        outside = [d for d in selection_dates if d not in allowed]
        if outside:
            raise WindowLeakageError(
                "selection dates must lie within train and validation windows; "
                f"outside {sorted(d.isoformat() for d in outside)}"
            )
        return TrainWinner(
            instrument_key=instrument_key,
            strategy_name=strategy_name,
            parameters={str(k): str(v) for k, v in dict(parameters).items()},
        )

    def authorize_untouched_test(
        self,
        *,
        frozen_universe: FrozenTrainUniverse,
        train_winner: TrainWinner,
    ) -> dict[str, Any]:
        """Authorize untouched-test evaluation against the frozen universe.

        The caller receives the full frozen train universe plus the frozen
        winner. No selection may occur here; use :meth:`select_train_winner`
        strictly before this step.
        """
        if frozen_universe != self.frozen_train_universe:
            raise FrozenUniverseViolationError(
                "untouched test must receive the full frozen train universe unchanged"
            )
        if not train_winner.instrument_key.strip() or not train_winner.strategy_name.strip():
            raise ResearchWindowError("train winner must include stock, strategy, and params")
        if not train_winner.parameters:
            raise ResearchWindowError("train winner must include params")
        return {
            "plan_id": self.plan_id,
            "test_window": self.untouched_test.as_dict(),
            "frozen_train_universe": self.frozen_train_universe.as_dict(),
            "train_winner": train_winner.as_dict(),
            "selection_forbidden": True,
        }

    def check_pit_membership(self, *, trade_date: date, evidence_as_of: date) -> None:
        """Enforce per-date point-in-time membership. Future evidence fails closed."""
        if evidence_as_of > trade_date:
            raise WindowLeakageError(
                f"PIT membership for {trade_date.isoformat()} cannot use evidence "
                f"as of {evidence_as_of.isoformat()}"
            )
        universe_dates = (
            set(self.train.trading_dates)
            | set(self.validation.trading_dates)
            | set(self.untouched_test.trading_dates)
        )
        if trade_date not in universe_dates:
            raise ResearchWindowError(f"{trade_date.isoformat()} is outside the compiled plan")


def compile_research_windows(
    *,
    research_start: date,
    research_end: date,
    trading_dates: tuple[date, ...],
    train_trading_days: int,
    validation_trading_days: int,
    test_trading_days: int,
    embargo_trading_days: int,
    min_observations_per_window: int,
    cost_evidence_fingerprint: str,
    approved_capital_rupees: Decimal,
    session_policy_id: str,
    cas_policy_id: str,
    corporate_action_evidence_fingerprint: str,
    dataset_fingerprints: Mapping[str, str],
    universe_fingerprint: str,
    pit_membership_fingerprint: str,
    train_universe_instruments: tuple[str, ...],
    created_at: str | None = None,
) -> ResearchWindowPlan:
    """Compile an overall range into immutable train/validation/test/embargo windows.

    Every duration and every provenance binding is an explicit caller input.
    There are no hidden defaults and no inferred optimal lengths. Windows are
    chronological slices of ``trading_dates``; randomness is never used. The
    compiled range must exactly consume ``trading_dates`` with no leftover and
    no overlap; otherwise compilation fails closed.
    """
    if research_start > research_end:
        raise ResearchWindowError("research_start must be on or before research_end")
    for name, value in (
        ("train_trading_days", train_trading_days),
        ("validation_trading_days", validation_trading_days),
        ("test_trading_days", test_trading_days),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ResearchWindowError(f"{name} must be a positive integer")
    if (
        not isinstance(embargo_trading_days, int)
        or isinstance(embargo_trading_days, bool)
        or embargo_trading_days < 0
    ):
        raise ResearchWindowError("embargo_trading_days must be a non-negative integer")
    if (
        not isinstance(min_observations_per_window, int)
        or isinstance(min_observations_per_window, bool)
        or min_observations_per_window <= 0
    ):
        raise ResearchWindowError("min_observations_per_window must be a positive integer")
    for name, value in (
        ("train_trading_days", train_trading_days),
        ("validation_trading_days", validation_trading_days),
        ("test_trading_days", test_trading_days),
    ):
        if value < min_observations_per_window:
            raise WindowTooShortError(
                f"{name}={value} is below min_observations_per_window={min_observations_per_window}"
            )
    if not trading_dates:
        raise WindowTooShortError("trading_dates must not be empty")
    ordered = tuple(trading_dates)
    if tuple(sorted(set(ordered))) != ordered:
        raise ResearchWindowError("trading_dates must be sorted unique dates")
    if ordered[0] != research_start or ordered[-1] != research_end:
        raise ResearchWindowError(
            "trading_dates must exactly span research_start through research_end"
        )
    if any(d < research_start or d > research_end for d in ordered):
        raise ResearchWindowError("trading_dates must lie within the research range")

    required = (
        train_trading_days
        + validation_trading_days
        + test_trading_days
        + (2 * embargo_trading_days)
    )
    if len(ordered) != required:
        raise WindowTooShortError(
            f"trading_dates has {len(ordered)} days but requested windows require "
            f"{required} (train {train_trading_days} + validation "
            f"{validation_trading_days} + test {test_trading_days} + 2x embargo "
            f"{embargo_trading_days}); no truncation or padding is applied"
        )

    _require_hex_digest("cost_evidence_fingerprint", cost_evidence_fingerprint)
    if not isinstance(approved_capital_rupees, Decimal) or (approved_capital_rupees <= Decimal(0)):
        raise ResearchWindowError("approved_capital_rupees must be a positive Decimal")
    _require_nonempty_str("session_policy_id", session_policy_id)
    _require_nonempty_str("cas_policy_id", cas_policy_id)
    _require_nonempty_str(
        "corporate_action_evidence_fingerprint", corporate_action_evidence_fingerprint
    )
    _require_nonempty_str("universe_fingerprint", universe_fingerprint)
    _require_nonempty_str("pit_membership_fingerprint", pit_membership_fingerprint)
    if not isinstance(dataset_fingerprints, Mapping) or not dataset_fingerprints:
        raise ResearchWindowError("dataset_fingerprints must be a non-empty mapping")
    for key, value in dataset_fingerprints.items():
        _require_nonempty_str("dataset instrument key", str(key))
        _require_nonempty_str(f"dataset fingerprint for {key}", str(value))
    if not train_universe_instruments:
        raise FrozenUniverseViolationError("train_universe_instruments must not be empty")
    frozen_instruments = tuple(sorted(set(train_universe_instruments)))
    for key in frozen_instruments:
        if not str(key).strip():
            raise FrozenUniverseViolationError("train universe instrument key must be non-empty")

    cursor = 0
    train_dates = ordered[cursor : cursor + train_trading_days]
    cursor += train_trading_days
    embargo1_dates = ordered[cursor : cursor + embargo_trading_days]
    cursor += embargo_trading_days
    validation_dates = ordered[cursor : cursor + validation_trading_days]
    cursor += validation_trading_days
    embargo2_dates = ordered[cursor : cursor + embargo_trading_days]
    cursor += embargo_trading_days
    test_dates = ordered[cursor : cursor + test_trading_days]
    cursor += test_trading_days
    assert cursor == len(ordered)

    def _window(role: WindowRole, window_id: str, dates: tuple[date, ...]) -> CompiledWindow:
        if role is WindowRole.EMBARGO and not dates:
            return CompiledWindow(
                role=role, window_id=window_id, start=None, end=None, trading_dates=()
            )
        assert dates
        return CompiledWindow(
            role=role,
            window_id=window_id,
            start=dates[0],
            end=dates[-1],
            trading_dates=dates,
        )

    train = _window(WindowRole.TRAIN, "train_01", train_dates)
    embargo_after_train = _window(WindowRole.EMBARGO, "embargo_train_validation_01", embargo1_dates)
    validation = _window(WindowRole.VALIDATION, "validation_01", validation_dates)
    embargo_after_validation = _window(
        WindowRole.EMBARGO, "embargo_validation_test_01", embargo2_dates
    )
    untouched_test = _window(WindowRole.UNTOUCHED_TEST, "test_untouched_01", test_dates)

    frozen_universe = FrozenTrainUniverse(
        instruments=frozen_instruments,
        universe_fingerprint=universe_fingerprint,
        pit_membership_fingerprint=pit_membership_fingerprint,
        frozen_as_of=train_dates[-1],
    )

    return ResearchWindowPlan(
        schema_version=SCHEMA_VERSION,
        research_start=research_start,
        research_end=research_end,
        train=train,
        validation=validation,
        embargo_after_train=embargo_after_train,
        embargo_after_validation=embargo_after_validation,
        untouched_test=untouched_test,
        train_trading_days=train_trading_days,
        validation_trading_days=validation_trading_days,
        test_trading_days=test_trading_days,
        embargo_trading_days=embargo_trading_days,
        min_observations_per_window=min_observations_per_window,
        cost_evidence_fingerprint=cost_evidence_fingerprint,
        approved_capital_rupees=approved_capital_rupees,
        session_policy_id=session_policy_id,
        cas_policy_id=cas_policy_id,
        corporate_action_evidence_fingerprint=corporate_action_evidence_fingerprint,
        dataset_fingerprints=dict(dataset_fingerprints),
        universe_fingerprint=universe_fingerprint,
        pit_membership_fingerprint=pit_membership_fingerprint,
        frozen_train_universe=frozen_universe,
        created_at=created_at,
    )
