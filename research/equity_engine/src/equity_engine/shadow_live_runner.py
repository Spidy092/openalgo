"""Read-only live shadow runner V1 (polling, never sends an order).

Wires the safest already-existing read-only market-data interface
(:class:`UpstoxFullQuoteV3Client.fetch_partial_by_instrument_token`) into the
existing :class:`ShadowSessionEngine` for Monday testing.

Path:

read-only live quote source (polling GET only)
-> normalized live market event (tz-checked, OHLC/LTP, fingerprint)
-> freshness / sequence / gap validation (runner level, then engine level)
-> ShadowMarketEvent
-> existing ShadowSessionEngine
-> append-only shadow evidence + immutable shadow report

Safety contract (load-bearing):

- No order capability. This module never imports a broker object exposing
  order methods and never touches an order submission endpoint. It never
  mutates a portfolio, position, or funds balance.
- Only two modes exist: ``DRY_RUN`` (synthetic fixture) and
  ``LIVE_READ_ONLY`` (real market-data GET only). There is no live-order
  mode and no flag can enable one.
- Authentication material comes only from the ``UPSTOX_ACCESS_TOKEN``
  environment variable at runtime. It is never printed (not even a prefix
  or length), never persisted, never written under ``/tmp``, and never
  included in exceptions, decisions, or reports.
- Missing token in ``LIVE_READ_ONLY`` fails closed with
  ``BLOCKED_TOKEN_MISSING`` and does not fail offline tests.
- Missing observations are never fabricated (UNKNOWN != ZERO). Naive
  timestamps, non-positive prices, and OHLC violations fail closed to
  NO TRADE. Unobserved volume is recorded as unobserved; the engine
  placeholder is never presented as an observed quantity.
- Websocket streaming is deliberately not introduced here. V1 uses bounded
  read-only quote polling, which is the proven read-only GET path already
  present in this repository.

Reused canonical interfaces (not duplicated):

- ``upstox_market_context.UpstoxFullQuoteV3Client``
  (``fetch_partial_by_instrument_token`` only)
- ``shadow_execution.ShadowSessionEngine`` and deterministic replay
- ``historical_cost_scenario`` / ``cost_ledger`` for the cost fingerprint
- ``market_sessions.NSEEquitySessionPolicy`` for session/CAS boundaries
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from .cost_ledger import EffectiveDatedCostLedger, LedgerComponent, LedgerProduct
from .documented_costs import CurrentTermsNSEIntradayCostProvider
from .event_simulator import FillAssumptions
from .experiment import ApprovedCapital
from .historical_cost_scenario import ScenarioAssumption, compile_historical_scenario
from .live_market_readiness import (
    REQUIRED_TIMEZONE,
    LiveMarketReadinessReport,
    ReadinessClassification,
    build_runner_readiness_context,
)
from .market_sessions import NSEEquitySessionPolicy
from .models import Exchange
from .shadow_execution import (
    AlwaysSignalShadowStrategy,
    ExitOnSecondBarStrategy,
    NeverSignalShadowStrategy,
    ShadowEngineConfig,
    ShadowInstrumentIdentity,
    ShadowMarketEvent,
    ShadowSessionEngine,
    ShadowSessionReport,
    replay_shadow_session,
    shadow_event_fingerprint,
)
from .shadow_session_health import (
    HealthStatus,
    HealthThresholds,
    ShadowSessionHealthReport,
    check_persisted_session,
    evaluate_session_health,
)
from .upstox_market_context import UpstoxFullQuoteV3Client

SCHEMA_VERSION = "shadow-live-runner/v1"
RUNNER_TIMEZONE = "Asia/Kolkata"
TOKEN_ENV_VAR = "UPSTOX_ACCESS_TOKEN"
BLOCKED_TOKEN_MISSING = "BLOCKED_TOKEN_MISSING"
READINESS_REPORT_MISSING = "READINESS_REPORT_MISSING"
READINESS_NOT_READY = "READINESS_NOT_READY"
READINESS_STALE = "READINESS_STALE"
READINESS_CONTEXT_MISMATCH = "READINESS_CONTEXT_MISMATCH"


class RunnerMode(StrEnum):
    DRY_RUN = "DRY_RUN"
    LIVE_READ_ONLY = "LIVE_READ_ONLY"


class FeedMode(StrEnum):
    POLL = "poll"


@dataclass(frozen=True)
class ShadowLiveConfig:
    session_id: str
    instrument_keys: tuple[str, ...]
    cas_eligible_by_key: tuple[tuple[str, bool], ...]
    tick_size_by_key: tuple[tuple[str, str], ...]
    feed_mode: FeedMode
    poll_interval_seconds: float
    max_polls: int
    quote_freshness_threshold_seconds: float
    expected_cadence_seconds: float
    approved_capital_rupees: str
    exit_buffer_minutes: int
    strategy_name: str
    output_dir: str
    mode: RunnerMode
    readiness_report: LiveMarketReadinessReport | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id is required")
        if not self.instrument_keys:
            raise ValueError("instrument_keys cannot be empty")
        if len(set(self.instrument_keys)) != len(self.instrument_keys):
            raise ValueError("instrument_keys contain duplicates")
        cas_keys = {key for key, _ in self.cas_eligible_by_key}
        tick_keys = {key for key, _ in self.tick_size_by_key}
        if cas_keys != set(self.instrument_keys):
            raise ValueError("cas_eligible_by_key must cover exactly instrument_keys")
        if tick_keys != set(self.instrument_keys):
            raise ValueError("tick_size_by_key must cover exactly instrument_keys")
        if self.feed_mode is not FeedMode.POLL:
            raise ValueError("V1 supports bounded read-only polling only")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds cannot be negative")
        if self.max_polls <= 0:
            raise ValueError("max_polls must be positive")
        if self.quote_freshness_threshold_seconds <= 0:
            raise ValueError("quote_freshness_threshold_seconds must be positive")
        if self.expected_cadence_seconds <= 0:
            raise ValueError("expected_cadence_seconds must be positive")
        try:
            capital = Decimal(self.approved_capital_rupees)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("approved_capital_rupees must be a Decimal string") from exc
        if capital <= 0:
            raise ValueError("approved_capital_rupees must be positive")
        if self.exit_buffer_minutes < 0 or self.exit_buffer_minutes >= 60:
            raise ValueError("exit_buffer_minutes must be in [0, 60)")
        if self.strategy_name not in ("always", "never", "exit-second-bar"):
            raise ValueError("strategy_name must be always|never|exit-second-bar")
        if not self.output_dir.strip():
            raise ValueError("output_dir is required")

    def cas_eligible(self, key: str) -> bool:
        return dict(self.cas_eligible_by_key)[key]

    def tick_size(self, key: str) -> Decimal:
        return Decimal(dict(self.tick_size_by_key)[key])

    def approved_capital(self) -> ApprovedCapital:
        return ApprovedCapital(amount_rupees=Decimal(self.approved_capital_rupees))

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "instrument_keys": list(self.instrument_keys),
            "cas_eligible_by_key": [
                {"instrument_key": key, "cas_eligible": value}
                for key, value in sorted(self.cas_eligible_by_key)
            ],
            "tick_size_by_key": [
                {"instrument_key": key, "tick_size_rupees": value}
                for key, value in sorted(self.tick_size_by_key)
            ],
            "feed_mode": self.feed_mode.value,
            "poll_interval_seconds": self.poll_interval_seconds,
            "max_polls": self.max_polls,
            "quote_freshness_threshold_seconds": self.quote_freshness_threshold_seconds,
            "expected_cadence_seconds": self.expected_cadence_seconds,
            "approved_capital_rupees": self.approved_capital_rupees,
            "exit_buffer_minutes": self.exit_buffer_minutes,
            "strategy_name": self.strategy_name,
            "output_dir": self.output_dir,
            "mode": self.mode.value,
            "live_orders_called": False,
        }


class ReadOnlyQuoteSource(Protocol):
    def fetch_quotes(self) -> dict[str, dict[str, object]]:
        """Return raw quote objects keyed by instrument key. Read-only GET."""
        ...


class SyntheticQuoteSource:
    """Deterministic offline fixture. No network, no token."""

    def __init__(self, batches: list[dict[str, dict[str, object]]]) -> None:
        self._batches = list(batches)
        self._index = 0

    def fetch_quotes(self) -> dict[str, dict[str, object]]:
        if self._index >= len(self._batches):
            return {}
        batch = self._batches[self._index]
        self._index += 1
        return batch


class UpstoxPollingQuoteSource:
    """Bounded read-only polling over the canonical full-quote V3 client."""

    def __init__(
        self,
        *,
        client: UpstoxFullQuoteV3Client,
        instrument_keys: tuple[str, ...],
    ) -> None:
        self._client = client
        self._keys = tuple(instrument_keys)

    def fetch_quotes(self) -> dict[str, dict[str, object]]:
        result = self._client.fetch_partial_by_instrument_token(list(self._keys))
        return dict(result.quotes)


def _read_token_redacted() -> str:
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        raise MissingTokenError(BLOCKED_TOKEN_MISSING)
    return token


class MissingTokenError(RuntimeError):
    pass


class ReadinessGateError(RuntimeError):
    """Raised when a runner has no current, matching readiness report."""


def _parse_decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"invalid numeric field {field}") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _parse_source_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is missing")
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} is timezone-naive; clock/timezone mismatch")
    return parsed


@dataclass(frozen=True)
class NormalizedLiveEvent:
    event: ShadowMarketEvent | None
    instrument_key: str
    source_timestamp: str | None
    received_timestamp: str
    seq: int
    quote_age_seconds: float | None
    data_fingerprint: str | None
    feed_status: str
    volume_observed: bool
    reconnect_boundary: bool
    reason: str


def build_strategy(name: str):  # type: ignore[no-untyped-def]
    if name == "always":
        return AlwaysSignalShadowStrategy(strategy_id="live-always")
    if name == "never":
        return NeverSignalShadowStrategy(strategy_id="live-never")
    return ExitOnSecondBarStrategy(strategy_id="live-exit-second-bar")


def build_default_cost_scenario(session_day: date):  # type: ignore[no-untyped-def]
    def _one(component: LedgerComponent, rate: str) -> ScenarioAssumption:
        return ScenarioAssumption(
            assumption_id=f"live-shadow-assume-{component.value}",
            component=component,
            product=LedgerProduct.INTRADAY,
            basis="live-shadow-test-basis",
            rate=Decimal(rate),
            formula=f"live-shadow-formula-{rate}",
            source="live-shadow-test-source",
            reason="live-shadow-test-reason",
        )

    return compile_historical_scenario(
        scenario_id="live-shadow-monday-v1",
        ledger=EffectiveDatedCostLedger(),
        scenario_date=session_day,
        research_start=date(2026, 9, 1),
        research_end=date(2026, 9, 8),
        product=LedgerProduct.INTRADAY,
        assumptions=(
            _one(LedgerComponent.BROKERAGE, "0.0006"),
            _one(LedgerComponent.GST, "0.18"),
            _one(LedgerComponent.CLEARING, "0.000001"),
        ),
    )


def normalize_quote(
    *,
    instrument_key: str,
    quote: dict[str, object],
    seq: int,
    received_at: datetime,
    cas_eligible: bool,
) -> NormalizedLiveEvent:
    received_iso = received_at.isoformat()
    try:
        source_ts = _parse_source_timestamp(quote.get("timestamp"), field="quote.timestamp")
    except ValueError as exc:
        return NormalizedLiveEvent(
            event=None,
            instrument_key=instrument_key,
            source_timestamp=None,
            received_timestamp=received_iso,
            seq=seq,
            quote_age_seconds=None,
            data_fingerprint=None,
            feed_status="clock_timezone_mismatch",
            volume_observed=False,
            reconnect_boundary=False,
            reason=f"clock_timezone_mismatch: {exc}",
        )
    try:
        last_price = _parse_decimal(quote.get("last_price"), field="quote.last_price")
        ohlc = quote.get("ohlc")
        if not isinstance(ohlc, dict):
            raise TypeError("quote.ohlc is missing")
        bar_open = _parse_decimal(ohlc.get("open"), field="quote.ohlc.open")
        bar_high = _parse_decimal(ohlc.get("high"), field="quote.ohlc.high")
        bar_low = _parse_decimal(ohlc.get("low"), field="quote.ohlc.low")
    except ValueError as exc:
        return NormalizedLiveEvent(
            event=None,
            instrument_key=instrument_key,
            source_timestamp=source_ts.isoformat(),
            received_timestamp=received_iso,
            seq=seq,
            quote_age_seconds=(received_at - source_ts).total_seconds(),
            data_fingerprint=None,
            feed_status="malformed_quote",
            volume_observed=False,
            reconnect_boundary=False,
            reason=f"malformed_quote: {exc}",
        )
    bar_close = last_price
    if bar_high < max(bar_open, bar_close, bar_low) or bar_low > min(bar_open, bar_close, bar_high):
        return NormalizedLiveEvent(
            event=None,
            instrument_key=instrument_key,
            source_timestamp=source_ts.isoformat(),
            received_timestamp=received_iso,
            seq=seq,
            quote_age_seconds=(received_at - source_ts).total_seconds(),
            data_fingerprint=None,
            feed_status="ohlc_violation",
            volume_observed=False,
            reconnect_boundary=False,
            reason="ohlc_violation_no_trade",
        )
    volume_raw = quote.get("volume")
    volume_observed = isinstance(volume_raw, (int, float)) and int(volume_raw) >= 0
    volume = int(volume_raw) if volume_observed else 0
    policy = NSEEquitySessionPolicy(cas_eligible=cas_eligible, exit_buffer_minutes=0)
    try:
        continuous_end = policy.continuous_end(source_ts.date())
        is_aux = source_ts.time() >= continuous_end
    except ValueError:
        is_aux = False
    event_id = f"live-{instrument_key}-{seq}-{source_ts.isoformat()}"
    try:
        event = ShadowMarketEvent(
            event_id=event_id,
            seq=seq,
            instrument_key=instrument_key,
            exchange=Exchange.NSE,
            bar_timestamp=source_ts,
            bar_open=bar_open,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_close=bar_close,
            bar_volume=volume,
            received_at=received_at,
            is_cas_auxiliary=is_aux,
            feed_connected=True,
        )
    except ValueError as exc:
        return NormalizedLiveEvent(
            event=None,
            instrument_key=instrument_key,
            source_timestamp=source_ts.isoformat(),
            received_timestamp=received_iso,
            seq=seq,
            quote_age_seconds=(received_at - source_ts).total_seconds(),
            data_fingerprint=None,
            feed_status="invalid_event",
            volume_observed=volume_observed,
            reconnect_boundary=False,
            reason=f"invalid_event: {exc}",
        )
    return NormalizedLiveEvent(
        event=event,
        instrument_key=instrument_key,
        source_timestamp=source_ts.isoformat(),
        received_timestamp=received_iso,
        seq=seq,
        quote_age_seconds=(received_at - source_ts).total_seconds(),
        data_fingerprint=shadow_event_fingerprint(event),
        feed_status="cas_auxiliary" if is_aux else "ok",
        volume_observed=volume_observed,
        reconnect_boundary=False,
        reason="cas_auxiliary" if is_aux else "ok",
    )


class ShadowLiveRunner:
    """Bounded polling runner. Read-only; exposes no order method."""

    def __init__(
        self,
        *,
        config: ShadowLiveConfig,
        source: ReadOnlyQuoteSource,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        session_day: date | None = None,
        readiness_report: LiveMarketReadinessReport | None = None,
    ) -> None:
        self._config = config
        self._source = source
        self._now = now or (lambda: datetime.now(ZoneInfo(RUNNER_TIMEZONE)))
        self._sleep = sleep or (lambda _: None)
        self._session_day = session_day or date(2026, 9, 7)
        self._readiness_report = readiness_report or config.readiness_report
        self._engines: dict[str, ShadowSessionEngine] = {}
        self._seq_by_key: dict[str, int] = {key: 0 for key in config.instrument_keys}
        self._normalized: list[NormalizedLiveEvent] = []
        self._saw_failure = False
        self._shadow_strategy_enabled = False
        self._health_report: ShadowSessionHealthReport | None = None
        self._recovery_observation_reasons: list[str] = []

    def _validate_readiness(self, now: datetime | None = None) -> LiveMarketReadinessReport:
        report = self._readiness_report
        if not isinstance(report, LiveMarketReadinessReport):
            raise ReadinessGateError(READINESS_REPORT_MISSING)
        if report.live_orders_called is not False:
            raise ReadinessGateError("readiness report contains live-order activity")
        if report.trade_date != self._session_day:
            raise ReadinessGateError(READINESS_CONTEXT_MISMATCH + ": trade_date")
        if report.context.timezone_name != REQUIRED_TIMEZONE:
            raise ReadinessGateError(READINESS_CONTEXT_MISMATCH + ": timezone")
        expected = build_runner_readiness_context(
            trade_date=self._session_day,
            timezone_name=REQUIRED_TIMEZONE,
            instrument_keys=self._config.instrument_keys,
            cas_eligible_by_key=self._config.cas_eligible_by_key,
            tick_size_by_key=self._config.tick_size_by_key,
            exit_buffer_minutes=self._config.exit_buffer_minutes,
            approved_capital=self._config.approved_capital(),
            quote_freshness_threshold_seconds=self._config.quote_freshness_threshold_seconds,
        )
        if report.context.instrument_keys != expected.instrument_keys:
            raise ReadinessGateError(READINESS_CONTEXT_MISMATCH + ": instruments")
        if (
            report.context.instrument_context_fingerprint != expected.instrument_context_fingerprint
            or report.context.session_context_fingerprint != expected.session_context_fingerprint
            or report.context.capital_identity != expected.capital_identity
            or report.context.quote_freshness_threshold_seconds
            != expected.quote_freshness_threshold_seconds
            or report.context.readiness_max_age_seconds != expected.readiness_max_age_seconds
        ):
            raise ReadinessGateError(READINESS_CONTEXT_MISMATCH)
        if report.approved_capital_rupees != self._config.approved_capital().amount_rupees:
            raise ReadinessGateError(READINESS_CONTEXT_MISMATCH + ": capital")
        now = now or self._now()
        if now.tzinfo is None:
            raise ReadinessGateError(READINESS_STALE + ": runner clock")
        now_ist = now.astimezone(ZoneInfo(REQUIRED_TIMEZONE))
        checked_at = report.checked_at_ist
        if checked_at.tzinfo is None:
            raise ReadinessGateError(READINESS_STALE + ": report clock")
        age = (now_ist - checked_at.astimezone(ZoneInfo(REQUIRED_TIMEZONE))).total_seconds()
        if (
            now_ist.date() != report.trade_date
            or age < 0
            or age > report.context.readiness_max_age_seconds
        ):
            raise ReadinessGateError(READINESS_STALE)
        if report.classification is ReadinessClassification.NOT_READY_FOR_SHADOW:
            raise ReadinessGateError(READINESS_NOT_READY)
        if not report.infra_ready:
            raise ReadinessGateError(READINESS_NOT_READY)
        return report

    def _build_shadow_engines(self) -> None:
        for key in self._config.instrument_keys:
            instrument = ShadowInstrumentIdentity(
                instrument_key=key,
                exchange=Exchange.NSE,
                cas_eligible=self._config.cas_eligible(key),
                tick_size_rupees=self._config.tick_size(key),
                source="live-runner-config",
            )
            engine_config = ShadowEngineConfig(
                session_id=f"{self._config.session_id}-{key}",
                exit_buffer_minutes=self._config.exit_buffer_minutes,
                max_quote_age_seconds=self._config.quote_freshness_threshold_seconds,
                bar_interval_seconds=max(1, int(self._config.expected_cadence_seconds)),
                max_gap_multiplier=1.5,
                max_trades_per_session=1,
            )
            self._engines[key] = ShadowSessionEngine(
                instrument=instrument,
                strategy=build_strategy(self._config.strategy_name),
                approved_capital=self._config.approved_capital(),
                cost_scenario=build_default_cost_scenario(self._session_day),
                cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=self._session_day),
                fills=FillAssumptions(
                    slippage_bps_per_leg=Decimal(2),
                    half_spread_bps_per_leg=Decimal(1),
                ),
                config=engine_config,
            )

    @property
    def normalized(self) -> tuple[NormalizedLiveEvent, ...]:
        return tuple(self._normalized)

    @property
    def health_report(self) -> ShadowSessionHealthReport | None:
        """Latest read-only health gate for this bounded run."""

        return self._health_report

    def _decision_reasons(self) -> tuple[str, ...]:
        return tuple(
            decision.reason
            for engine in self._engines.values()
            for decision in engine.report().decisions
        )

    def _theoretical_trade_count(self) -> int:
        return sum(len(engine.report().trades) for engine in self._engines.values())

    def _update_health(
        self,
        *,
        now: datetime,
        last_quote_age_seconds: float | None = None,
    ) -> ShadowSessionHealthReport:
        report = evaluate_session_health(
            session_id=self._config.session_id,
            runner_started=True,
            runner_stopped=False,
            killed=False,
            disk_error=False,
            persistence_ok=True,
            checksums_ok=True,
            reports_ok=True,
            normalized_feed_statuses=tuple(item.feed_status for item in self._normalized),
            normalized_received_ats=tuple(item.received_timestamp for item in self._normalized),
            normalized_reconnects=tuple(item.reconnect_boundary for item in self._normalized),
            decision_reasons=self._decision_reasons() + tuple(self._recovery_observation_reasons),
            decisions_count=sum(
                len(engine.report().decisions) for engine in self._engines.values()
            ),
            theoretical_trades_count=self._theoretical_trade_count(),
            expected_cadence_seconds=self._config.expected_cadence_seconds,
            last_quote_age_seconds=last_quote_age_seconds,
            now_iso=now.isoformat(),
            thresholds=HealthThresholds(
                freshness_seconds=self._config.quote_freshness_threshold_seconds,
            ),
        )
        self._health_report = report
        return report

    def run(self) -> dict[str, ShadowSessionReport]:
        first_received = self._now()
        report = self._validate_readiness(first_received)
        self._shadow_strategy_enabled = report.research_shadow_ready
        if self._shadow_strategy_enabled and not self._engines:
            self._build_shadow_engines()
        for poll_number in range(self._config.max_polls):
            received = first_received if poll_number == 0 else self._now()
            if poll_number > 0:
                # A readiness report is a point-in-time authorization for this
                # read-only shadow run, not a session-long bypass. Recheck its
                # bound before every subsequent poll and fail closed if it
                # becomes stale or the context no longer matches.
                self._validate_readiness(received)
            try:
                batch = self._source.fetch_quotes()
            except Exception as exc:  # noqa: BLE001 - any poll failure must fail closed
                self._saw_failure = True
                for key in self._config.instrument_keys:
                    self._normalized.append(
                        NormalizedLiveEvent(
                            event=None,
                            instrument_key=key,
                            source_timestamp=None,
                            received_timestamp=received.isoformat(),
                            seq=self._seq_by_key[key],
                            quote_age_seconds=None,
                            data_fingerprint=None,
                            feed_status="poll_failure",
                            volume_observed=False,
                            reconnect_boundary=True,
                            reason=f"poll_failure_no_trade: {type(exc).__name__}",
                        )
                    )
                self._update_health(now=received)
                self._sleep(self._config.poll_interval_seconds)
                continue
            reconnect = self._saw_failure
            self._saw_failure = False
            for key in self._config.instrument_keys:
                seq = self._seq_by_key[key]
                self._seq_by_key[key] += 1
                quote = batch.get(key)
                if quote is None:
                    self._normalized.append(
                        NormalizedLiveEvent(
                            event=None,
                            instrument_key=key,
                            source_timestamp=None,
                            received_timestamp=received.isoformat(),
                            seq=seq,
                            quote_age_seconds=None,
                            data_fingerprint=None,
                            feed_status="missing_instrument",
                            volume_observed=False,
                            reconnect_boundary=reconnect,
                            reason="missing_instrument_no_trade",
                        )
                    )
                    self._update_health(now=received)
                    continue
                normalized = normalize_quote(
                    instrument_key=key,
                    quote=quote,
                    seq=seq,
                    received_at=received,
                    cas_eligible=self._config.cas_eligible(key),
                )
                if reconnect:
                    normalized = NormalizedLiveEvent(
                        event=normalized.event,
                        instrument_key=normalized.instrument_key,
                        source_timestamp=normalized.source_timestamp,
                        received_timestamp=normalized.received_timestamp,
                        seq=normalized.seq,
                        quote_age_seconds=normalized.quote_age_seconds,
                        data_fingerprint=normalized.data_fingerprint,
                        feed_status=normalized.feed_status,
                        volume_observed=normalized.volume_observed,
                        reconnect_boundary=True,
                        reason=normalized.reason,
                    )
                self._normalized.append(normalized)
                health = self._update_health(
                    now=received,
                    last_quote_age_seconds=normalized.quote_age_seconds,
                )
                safe_evidence = normalized.feed_status == "cas_auxiliary" or (
                    normalized.quote_age_seconds is not None
                    and normalized.quote_age_seconds
                    > self._config.quote_freshness_threshold_seconds
                )
                if (
                    normalized.event is not None
                    and self._shadow_strategy_enabled
                    and (health.status is HealthStatus.HEALTHY or safe_evidence)
                    and not normalized.reconnect_boundary
                ):
                    self._engines[key].process(normalized.event)
                elif (
                    normalized.event is not None
                    and self._shadow_strategy_enabled
                    and health.status is not HealthStatus.HEALTHY
                    and normalized.feed_status == "ok"
                    and not normalized.reconnect_boundary
                    and normalized.quote_age_seconds is not None
                    and normalized.quote_age_seconds
                    <= self._config.quote_freshness_threshold_seconds
                ):
                    self._recovery_observation_reasons.append("recovery_observation_ok")
                self._update_health(
                    now=received,
                    last_quote_age_seconds=normalized.quote_age_seconds,
                )
            self._sleep(self._config.poll_interval_seconds)
        return {key: engine.report() for key, engine in self._engines.items()}

    def persist(self, output_dir: Path) -> dict[str, Any]:
        if not isinstance(self._readiness_report, LiveMarketReadinessReport):
            raise ReadinessGateError(READINESS_REPORT_MISSING)
        output_dir.mkdir(parents=True, exist_ok=True)
        reports = {key: engine.report() for key, engine in self._engines.items()}
        (output_dir / "readiness-report.json").write_text(
            self._readiness_report.to_json(), encoding="utf-8"
        )
        config_path = output_dir / "config.json"
        config_path.write_text(
            json.dumps(self._config.as_safe_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        events_path = output_dir / "market_events.jsonl"
        with events_path.open("w", encoding="utf-8") as handle:
            for item in self._normalized:
                payload = {
                    "instrument_key": item.instrument_key,
                    "source_timestamp": item.source_timestamp,
                    "received_timestamp": item.received_timestamp,
                    "seq": item.seq,
                    "quote_age_seconds": item.quote_age_seconds,
                    "data_fingerprint": item.data_fingerprint,
                    "feed_status": item.feed_status,
                    "volume_observed": item.volume_observed,
                    "reconnect_boundary": item.reconnect_boundary,
                    "reason": item.reason,
                    "event": item.event.canonical_payload() if item.event is not None else None,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        decisions_path = output_dir / "decisions.jsonl"
        with decisions_path.open("w", encoding="utf-8") as handle:
            for key, report in sorted(reports.items()):
                for decision in report.decisions:
                    handle.write(
                        json.dumps({"instrument_key": key, **decision.as_dict()}, sort_keys=True)
                        + "\n"
                    )
        trades_path = output_dir / "trades.jsonl"
        with trades_path.open("w", encoding="utf-8") as handle:
            for key, report in sorted(reports.items()):
                for trade in report.trades:
                    handle.write(
                        json.dumps(
                            {
                                "instrument_key": key,
                                "trade_id": trade.trade_id,
                                "quantity": trade.quantity,
                                "theoretical_entry": format(trade.theoretical_entry, "f"),
                                "theoretical_exit": format(trade.theoretical_exit, "f"),
                                "net_theoretical_pnl": format(trade.net_theoretical_pnl, "f"),
                                "entry_timestamp": trade.entry_timestamp,
                                "exit_timestamp": trade.exit_timestamp,
                                "exit_reason": trade.exit_reason,
                                "label": trade.label,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        for key, report in sorted(reports.items()):
            safe_key = key.replace("|", "_").replace(":", "_")
            (output_dir / f"report-{safe_key}.json").write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self._config.session_id,
            "mode": self._config.mode.value,
            "readiness_classification": self._readiness_report.classification.value,
            "readiness_context_fingerprint": self._readiness_report.context.fingerprint(),
            "shadow_strategy_enabled": self._shadow_strategy_enabled,
            "instruments": sorted(reports),
            "report_fingerprints": {
                key: report.fingerprint() for key, report in sorted(reports.items())
            },
            "live_orders_called": False,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        def _write_checksums() -> None:
            checksums: dict[str, str] = {}
            for path in sorted(output_dir.iterdir()):
                if path.is_file() and path.name != "CHECKSUMS.sha256":
                    checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (output_dir / "CHECKSUMS.sha256").write_text(
                "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
                encoding="utf-8",
            )

        _write_checksums()
        health_now = (
            self._normalized[-1].received_timestamp
            if self._normalized
            else self._readiness_report.checked_at_ist.isoformat()
        )
        self._health_report = check_persisted_session(output_dir, now_iso=health_now)
        health_payload = self._health_report.as_dict()
        health_payload["fingerprint"] = self._health_report.fingerprint()
        (output_dir / "health-report.json").write_text(
            json.dumps(health_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary["health_status"] = self._health_report.status.value
        summary["health_report_fingerprint"] = self._health_report.fingerprint()
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_checksums()
        return summary


def verify_persisted_replay(
    output_dir: Path,
    *,
    config: ShadowLiveConfig,
    session_day: date | None = None,
) -> dict[str, Any]:
    """Reload persisted events and verify deterministic replay matches."""
    from .shadow_execution import ShadowMarketEvent as _Event

    events_by_key: dict[str, list[_Event]] = {key: [] for key in config.instrument_keys}
    events_path = output_dir / "market_events.jsonl"
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        payload = row.get("event")
        if payload is None:
            continue
        key = str(row["instrument_key"])
        event = _Event(
            event_id=str(payload["event_id"]),
            seq=int(payload["seq"]),
            instrument_key=key,
            exchange=Exchange.NSE,
            bar_timestamp=datetime.fromisoformat(str(payload["bar_timestamp"])),
            bar_open=Decimal(str(payload["bar_open"])),
            bar_high=Decimal(str(payload["bar_high"])),
            bar_low=Decimal(str(payload["bar_low"])),
            bar_close=Decimal(str(payload["bar_close"])),
            bar_volume=int(payload["bar_volume"]),
            received_at=datetime.fromisoformat(str(payload["received_at"])),
            is_cas_auxiliary=bool(payload["is_cas_auxiliary"]),
            feed_connected=bool(payload["feed_connected"]),
        )
        events_by_key[key].append(event)
    day = session_day or date(2026, 9, 7)
    matched: dict[str, bool] = {}
    fingerprints: dict[str, str] = {}
    for key in config.instrument_keys:
        instrument = ShadowInstrumentIdentity(
            instrument_key=key,
            exchange=Exchange.NSE,
            cas_eligible=config.cas_eligible(key),
            tick_size_rupees=config.tick_size(key),
            source="live-runner-replay",
        )
        engine_config = ShadowEngineConfig(
            session_id=f"{config.session_id}-{key}",
            exit_buffer_minutes=config.exit_buffer_minutes,
            max_quote_age_seconds=config.quote_freshness_threshold_seconds,
            bar_interval_seconds=max(1, int(config.expected_cadence_seconds)),
            max_gap_multiplier=1.5,
            max_trades_per_session=1,
        )
        replayed = replay_shadow_session(
            events_by_key[key],
            instrument=instrument,
            strategy=build_strategy(config.strategy_name),
            approved_capital=config.approved_capital(),
            cost_scenario=build_default_cost_scenario(day),
            cost_provider=CurrentTermsNSEIntradayCostProvider(pricing_date=day),
            fills=FillAssumptions(
                slippage_bps_per_leg=Decimal(2),
                half_spread_bps_per_leg=Decimal(1),
            ),
            config=engine_config,
        )
        safe_key = key.replace("|", "_").replace(":", "_")
        persisted = json.loads((output_dir / f"report-{safe_key}.json").read_text())
        matched[key] = persisted["fingerprint"] == replayed.fingerprint()
        fingerprints[key] = replayed.fingerprint()
    return {"matched": matched, "fingerprints": fingerprints}


def scan_output_for_credentials(output_dir: Path, token: str | None = None) -> list[str]:
    """Return persisted files containing credential material (empty means safe)."""
    candidates = token.strip() if token and token.strip() else None
    violations: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lowered = text.lower()
        if "upstox_access_token" in lowered and path.name != "config.json":
            violations.append(str(path))
        if candidates and candidates in text:
            violations.append(str(path))
    config_text = (output_dir / "config.json").read_text(encoding="utf-8")
    if candidates and candidates in config_text:
        violations.append(str(output_dir / "config.json"))
    return sorted(set(violations))


__all__ = [
    "BLOCKED_TOKEN_MISSING",
    "READINESS_CONTEXT_MISMATCH",
    "READINESS_NOT_READY",
    "READINESS_REPORT_MISSING",
    "READINESS_STALE",
    "SCHEMA_VERSION",
    "FeedMode",
    "NormalizedLiveEvent",
    "ReadOnlyQuoteSource",
    "ReadinessGateError",
    "RunnerMode",
    "ShadowLiveConfig",
    "ShadowLiveRunner",
    "SyntheticQuoteSource",
    "UpstoxPollingQuoteSource",
    "build_default_cost_scenario",
    "build_strategy",
    "normalize_quote",
    "scan_output_for_credentials",
    "verify_persisted_replay",
]
