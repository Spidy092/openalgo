"""Monday readiness probe V1: safe runtime adapter over the readiness gate.

This module populates the canonical :class:`LiveMarketReadinessInputs` using
only existing read-only repository interfaces and caller-supplied evidence.
It reuses :class:`LiveMarketReadinessReport` and never duplicates it.

Safety contract (load-bearing):

- No order mode exists. This module has no order method, imports no order or
  broker-trading service, and never calls place/modify/cancel APIs.
- Token handling is PRESENT/ABSENT only. This module never accepts, returns,
  logs, or persists a token value, prefix, or length. The credential value is
  held transiently by the CLI process (same pattern as the existing
  read-only scripts) and passed only to read-only clients
  (:class:`UpstoxReadinessProbe`, :class:`UpstoxFullQuoteV3Client`); the
  adapter functions below receive ``token_present: bool`` and pre-fetched
  evidence, never the value.
- Persisted artifacts are credential-free and carry a deterministic
  SHA-256 fingerprint over the canonical payload (fingerprint field excluded
  before hashing, mirroring :func:`fingerprint_payload`).

Modes:

- ``DRY_RUN``: synthetic preflight. No network, no token needed. Proves gate
  logic plus real token presence (presence is still read from the
  environment); every other evidence item is synthetic and the payload is
  labelled with the mode so it can never be mistaken for market state.
- ``LIVE_READ_ONLY``: live preflight. Collects token presence, read-only
  broker connectivity, calendar/session evidence, the current IST clock,
  quote freshness, caller-attested feed health, instrument identity with
  suspension/tradability evidence, tick evidence, CAS policy, and explicit
  approved shadow capital. PIT/historical/strategy/cost/paper evidence is
  supplied by caller/config, never synthesized. When the token is absent the
  probe stops before any client is touched and returns
  ``BLOCKED_TOKEN_MISSING``.

There is deliberately no live feed interface in the research engine, so feed
health is caller-attested even in ``LIVE_READ_ONLY``: pass an explicit
:class:`FeedHealthEvidence`, or ``None`` to fail closed with
``FEED_STATUS_UNKNOWN``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

from .current_market_discovery import fingerprint_payload
from .experiment import (
    ApprovedCapital,
    CostReconciliationEvidence,
    LiveOrderAttemptError,
    PaperTradingEvidence,
)
from .historical_validation import (
    DailyIntradayValidation,
    HistoricalDatasetValidation,
)
from .instrument_master import EquityInstrument, build_nse_equity_master
from .live_market_readiness import (
    REQUIRED_TIMEZONE,
    FeedHealthEvidence,
    LiveMarketReadinessInputs,
    LiveMarketReadinessReport,
    ReadinessClassification,
    evaluate_live_market_readiness,
)
from .market_sessions import NSEEquitySessionPolicy
from .nse_calendar import CalendarEvidence, nse_cm_normal_session_calendar
from .tick_size import TickSizeVerification, verify_instrument_tick_size
from .upstox_market_context import QuoteBatchResult
from .upstox_readiness import ReadinessCheck, UpstoxReadinessSnapshot

PROBE_SCHEMA_VERSION = "monday-readiness-probe/v1"
TOKEN_ENV_VAR = "UPSTOX_ACCESS_TOKEN"
BLOCKED_TOKEN_MISSING = "BLOCKED_TOKEN_MISSING"

EXIT_LIVE_REVIEW = 0
EXIT_RESEARCH_SHADOW = 1
EXIT_SHADOW_INFRA = 2
EXIT_NOT_READY = 3
EXIT_BLOCKED_TOKEN = 4
EXIT_ERROR = 5


class ProbeMode(StrEnum):
    """Explicit probe mode. There is no order mode."""

    DRY_RUN = "DRY_RUN"
    LIVE_READ_ONLY = "LIVE_READ_ONLY"


def token_present_from_env(*, env_var: str = TOKEN_ENV_VAR) -> bool:
    """Return broker-credential presence only; the value never leaves this call."""

    return bool(os.environ.get(env_var, "").strip())


def current_ist_now() -> datetime:
    """Return the current timezone-aware IST clock reading."""

    return datetime.now(ZoneInfo(REQUIRED_TIMEZONE))


@dataclass(frozen=True)
class MondayProbeConfig:
    """Explicit caller/config inputs. Every field is mandatory; no hidden defaults.

    Research evidence (PIT/historical/strategy/cost/paper) and feed health
    are caller-supplied: ``None``/``False`` means explicitly missing and fails
    closed at the scope that requires it.
    """

    mode: ProbeMode
    instrument_key: str
    approved_capital_rupees: Decimal
    cost_tolerance_inr: Decimal
    max_quote_age_seconds: float
    exit_buffer_minutes: int
    tick_size_scale_rupees_per_raw_unit: Decimal
    tick_reference_price_rupees: Decimal
    pit_complete: bool
    historical_validation: HistoricalDatasetValidation | None
    strategy_evidence_present: bool
    cost_evidence: CostReconciliationEvidence | None
    paper_evidence: PaperTradingEvidence | None
    feed: FeedHealthEvidence | None
    kill_switch_engaged: bool

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ProbeMode):
            raise TypeError("mode must be a ProbeMode")
        if not self.instrument_key.strip():
            raise ValueError("instrument_key is required")
        for name in ("approved_capital_rupees", "cost_tolerance_inr"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise TypeError(f"{name} must be a finite Decimal")
        if self.approved_capital_rupees <= 0:
            raise ValueError("approved_capital_rupees must be positive")
        if self.cost_tolerance_inr < 0:
            raise ValueError("cost_tolerance_inr cannot be negative")
        if not isinstance(self.max_quote_age_seconds, (int, float)):
            raise TypeError("max_quote_age_seconds must be numeric")
        if not self.max_quote_age_seconds > 0:
            raise ValueError("max_quote_age_seconds must be positive")
        if not isinstance(self.exit_buffer_minutes, int) or isinstance(
            self.exit_buffer_minutes, bool
        ):
            raise TypeError("exit_buffer_minutes must be an integer")
        if self.exit_buffer_minutes < 0 or self.exit_buffer_minutes >= 60:
            raise ValueError("exit_buffer_minutes must be in [0, 60)")
        if not isinstance(self.tick_size_scale_rupees_per_raw_unit, Decimal):
            raise TypeError("tick_size_scale_rupees_per_raw_unit must be a Decimal")
        if self.tick_size_scale_rupees_per_raw_unit <= 0:
            raise ValueError("tick_size_scale_rupees_per_raw_unit must be positive")
        if not isinstance(self.tick_reference_price_rupees, Decimal):
            raise TypeError("tick_reference_price_rupees must be a Decimal")
        if self.tick_reference_price_rupees <= 0:
            raise ValueError("tick_reference_price_rupees must be positive")
        for name in ("pit_complete", "strategy_evidence_present", "kill_switch_engaged"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")


@dataclass(frozen=True)
class MondayProbeResult:
    """Credential-free probe outcome with deterministic fingerprint."""

    schema_version: str
    mode: ProbeMode
    blocked_code: str | None
    token_present: bool
    report: LiveMarketReadinessReport | None
    artifact_fingerprint: str
    live_orders_called: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != PROBE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {self.schema_version!r}")
        if not isinstance(self.mode, ProbeMode):
            raise TypeError("mode must be a ProbeMode")
        if self.blocked_code is not None and self.blocked_code != BLOCKED_TOKEN_MISSING:
            raise ValueError(f"unknown blocked code {self.blocked_code!r}")
        if not isinstance(self.token_present, bool):
            raise TypeError("token_present must be boolean")
        if self.live_orders_called:
            raise LiveOrderAttemptError("live orders are strictly forbidden in the probe")
        if len(self.artifact_fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in self.artifact_fingerprint
        ):
            raise ValueError("artifact_fingerprint must be a lowercase hex digest")
        if self.blocked_code is not None and self.report is not None:
            raise ValueError("a blocked probe carries no readiness report")
        if self.blocked_code is None and self.report is None:
            raise ValueError("an unblocked probe must carry a readiness report")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "blocked_code": self.blocked_code,
            "token_present": self.token_present,
            "report": self.report.to_dict() if self.report is not None else None,
            "live_orders_called": False,
            "artifact_fingerprint": self.artifact_fingerprint,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"


def _payload_without_fingerprint(result: MondayProbeResult) -> dict[str, object]:
    payload = result.to_dict()
    payload.pop("artifact_fingerprint")
    return payload


def verify_probe_fingerprint(result: MondayProbeResult) -> bool:
    """Recompute the fingerprint over the canonical payload and compare."""

    return fingerprint_payload(_payload_without_fingerprint(result)) == result.artifact_fingerprint


def _finalize(
    *,
    mode: ProbeMode,
    blocked_code: str | None,
    token_present: bool,
    report: LiveMarketReadinessReport | None,
) -> MondayProbeResult:
    """Attach a deterministic fingerprint to a credential-free probe outcome."""

    if blocked_code is not None and report is not None:
        raise ValueError("a blocked probe carries no readiness report")
    placeholder = MondayProbeResult(
        schema_version=PROBE_SCHEMA_VERSION,
        mode=mode,
        blocked_code=blocked_code,
        token_present=token_present,
        report=report,
        artifact_fingerprint="0" * 64,
    )
    fingerprint = fingerprint_payload(_payload_without_fingerprint(placeholder))
    return MondayProbeResult(
        schema_version=PROBE_SCHEMA_VERSION,
        mode=mode,
        blocked_code=blocked_code,
        token_present=token_present,
        report=report,
        artifact_fingerprint=fingerprint,
    )


def blocked_token_result() -> MondayProbeResult:
    """Return the credential-free blocked outcome used when the token is absent.

    No client is touched and no gate evaluation runs; the caller must stop
    before any broker fetch.
    """

    return _finalize(
        mode=ProbeMode.LIVE_READ_ONLY,
        blocked_code=BLOCKED_TOKEN_MISSING,
        token_present=False,
        report=None,
    )


def write_probe_report(path: Path, result: MondayProbeResult) -> Path:
    """Persist a credential-free fingerprinted probe report."""

    if not verify_probe_fingerprint(result):
        raise ValueError("probe result fingerprint does not verify; refusing to persist")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json(), encoding="utf-8")
    return path


def exit_code_for_result(result: MondayProbeResult) -> int:
    """Map a probe outcome to a documented process exit code."""

    if result.blocked_code == BLOCKED_TOKEN_MISSING:
        return EXIT_BLOCKED_TOKEN
    if result.report is None:
        return EXIT_ERROR
    match result.report.classification:
        case ReadinessClassification.READY_FOR_LIVE_ORDER_REVIEW:
            return EXIT_LIVE_REVIEW
        case ReadinessClassification.READY_FOR_RESEARCH_SHADOW:
            return EXIT_RESEARCH_SHADOW
        case ReadinessClassification.READY_FOR_SHADOW_INFRA:
            return EXIT_SHADOW_INFRA
        case _:
            return EXIT_NOT_READY


def derive_session_state(
    *,
    calendar: CalendarEvidence | None,
    session_policy: NSEEquitySessionPolicy | None,
    trade_date: date,
    now_ist: datetime,
) -> tuple[bool, bool]:
    """Derive (is_trading_day, session_open) without network or broker access."""

    if calendar is None:
        return False, False
    is_trading_day = trade_date in set(calendar.trading_dates)
    if not is_trading_day or session_policy is None:
        return is_trading_day, False
    try:
        start = session_policy.continuous_start(trade_date)
        end = session_policy.continuous_end(trade_date)
    except ValueError:
        return is_trading_day, False
    now_time = now_ist.time()
    return is_trading_day, bool(start <= now_time < end)


def select_instrument(
    *,
    bod_rows: tuple[Mapping[str, object], ...],
    mis_rows: tuple[Mapping[str, object], ...],
    suspended_rows: tuple[Mapping[str, object], ...],
    as_of_date: date,
    tick_size_scale_rupees_per_raw_unit: Decimal,
    instrument_key: str,
) -> EquityInstrument | None:
    """Resolve one instrument from public master files; None fails closed downstream."""

    try:
        master = build_nse_equity_master(
            as_of_date=as_of_date,
            bod_rows=bod_rows,
            mis_rows=mis_rows,
            suspended_rows=suspended_rows,
            tick_size_scale_rupees_per_raw_unit=tick_size_scale_rupees_per_raw_unit,
        )
    except ValueError:
        return None
    return master.by_key().get(instrument_key)


def verify_tick(
    *,
    instrument: EquityInstrument | None,
    trade_date: date,
    reference_price_rupees: Decimal,
) -> TickSizeVerification | None:
    """Verify tick size against the dated NSE tier; None fails closed downstream."""

    if instrument is None:
        return None
    try:
        return verify_instrument_tick_size(
            instrument=instrument,
            effective_trade_date=trade_date,
            exchange_reference_price_rupees=reference_price_rupees,
        )
    except ValueError:
        return None


def assemble_inputs(
    *,
    config: MondayProbeConfig,
    token_present: bool,
    readiness_snapshot: UpstoxReadinessSnapshot | None,
    now_ist: datetime,
    trade_date: date,
    is_trading_day: bool,
    session_open: bool,
    session_policy: NSEEquitySessionPolicy | None,
    calendar_evidence: CalendarEvidence | None,
    quotes: QuoteBatchResult | None,
    feed: FeedHealthEvidence | None,
    instrument: EquityInstrument | None,
    tick_verification: TickSizeVerification | None,
) -> LiveMarketReadinessInputs:
    """Assemble canonical gate inputs from probe evidence. Pure: no I/O, no token value."""

    if not isinstance(token_present, bool):
        raise TypeError("token_present must be boolean")
    return LiveMarketReadinessInputs(
        token_present=token_present,
        readiness_snapshot=readiness_snapshot,
        now_ist=now_ist,
        timezone_name=REQUIRED_TIMEZONE,
        trade_date=trade_date,
        is_trading_day=is_trading_day,
        session_open=session_open,
        session_policy=session_policy,
        calendar_evidence=calendar_evidence,
        quotes=quotes,
        expected_instrument_keys=(config.instrument_key,),
        max_quote_age_seconds=config.max_quote_age_seconds,
        feed=feed,
        instrument=instrument,
        tick_verification=tick_verification,
        approved_capital=ApprovedCapital(amount_rupees=config.approved_capital_rupees),
        broker_available_to_trade=(
            readiness_snapshot.available_to_trade if readiness_snapshot is not None else None
        ),
        cost_evidence=config.cost_evidence,
        cost_tolerance_inr=config.cost_tolerance_inr,
        pit_complete=config.pit_complete,
        historical_validation=config.historical_validation,
        strategy_evidence_present=config.strategy_evidence_present,
        paper_evidence=config.paper_evidence,
        kill_switch_engaged=config.kill_switch_engaged,
        live_orders_called=False,
    )


def _synthetic_snapshot(*, static_ip: bool = True) -> UpstoxReadinessSnapshot:
    return UpstoxReadinessSnapshot(
        checks=(
            ReadinessCheck(name="profile_api", passed=True, detail="synthetic preflight"),
            ReadinessCheck(name="nse_enabled", passed=True, detail="synthetic preflight"),
            ReadinessCheck(
                name="intraday_product_enabled", passed=True, detail="synthetic preflight"
            ),
            ReadinessCheck(name="funds_api", passed=True, detail="synthetic preflight"),
            ReadinessCheck(
                name="minimum_test_capital_present", passed=True, detail="synthetic preflight"
            ),
            ReadinessCheck(
                name="primary_static_ip",
                passed=static_ip,
                detail="synthetic preflight",
            ),
        ),
        available_to_trade=Decimal(15000),
        exchanges=("NSE", "BSE"),
        products=("D", "I"),
        primary_static_ip_configured=static_ip,
        secondary_static_ip_configured=False,
    )


def _synthetic_instrument() -> EquityInstrument:
    return EquityInstrument(
        instrument_key="SYNTHETIC_DRY_RUN",
        isin="INE000A01010",
        trading_symbol="DRYRUN",
        name="DRY RUN SYNTHETIC",
        segment="NSE_EQ",
        exchange="NSE",
        instrument_type="EQ",
        security_type="NORMAL",
        lot_size=1,
        freeze_quantity=Decimal(100000),
        tick_size_raw=Decimal(5),
        tick_size_rupees=Decimal("0.05"),
        cas_eligible=False,
        mis_eligible=True,
        suspended=False,
        exchange_token="1",
        suspension_status="NO_SUSPENSION_RECORD",
        suspension_variant_row_count=0,
        exact_token_match_count=0,
        live_tradability_proven=True,
    )


def run_dry_run(
    config: MondayProbeConfig, *, now_ist: datetime, token_present: bool
) -> MondayProbeResult:
    """Run the synthetic preflight: no network, token presence only, plumbing synthetic.

    Research evidence comes from the config (missing by default, never
    fabricated), so the ceiling is ``READY_FOR_SHADOW_INFRA``.
    """

    if config.mode is not ProbeMode.DRY_RUN:
        raise ValueError("run_dry_run requires a DRY_RUN config")
    trade_date = now_ist.date()
    quotes = QuoteBatchResult(
        requested_instrument_keys=(config.instrument_key,),
        quotes={
            config.instrument_key: {
                "instrument_token": config.instrument_key,
                "last_price": "100",
                "timestamp": now_ist.isoformat(),
            }
        },
        failures={},
        request_count=1,
    )
    inputs = LiveMarketReadinessInputs(
        token_present=token_present,
        readiness_snapshot=_synthetic_snapshot(),
        now_ist=now_ist,
        timezone_name=REQUIRED_TIMEZONE,
        trade_date=trade_date,
        is_trading_day=True,
        session_open=True,
        session_policy=NSEEquitySessionPolicy(
            cas_eligible=False, exit_buffer_minutes=config.exit_buffer_minutes
        ),
        calendar_evidence=CalendarEvidence(
            trading_dates=(trade_date,),
            holiday_dates=(),
            excluded_special_session_dates=(),
            source_urls=("synthetic-dry-run-preflight",),
        ),
        quotes=quotes,
        expected_instrument_keys=(config.instrument_key,),
        max_quote_age_seconds=config.max_quote_age_seconds,
        feed=FeedHealthEvidence(available=True, gap_detected=False, last_heartbeat_ist=now_ist),
        instrument=_synthetic_instrument(),
        tick_verification=TickSizeVerification(
            passed=True,
            expected_rupees=Decimal("0.05"),
            observed_rupees=Decimal("0.05"),
            source="synthetic-dry-run-preflight",
        ),
        approved_capital=ApprovedCapital(amount_rupees=config.approved_capital_rupees),
        broker_available_to_trade=Decimal(15000),
        cost_evidence=config.cost_evidence,
        cost_tolerance_inr=config.cost_tolerance_inr,
        pit_complete=config.pit_complete,
        historical_validation=config.historical_validation,
        strategy_evidence_present=config.strategy_evidence_present,
        paper_evidence=config.paper_evidence,
        kill_switch_engaged=config.kill_switch_engaged,
        live_orders_called=False,
    )
    report = evaluate_live_market_readiness(inputs)
    return _finalize(
        mode=ProbeMode.DRY_RUN,
        blocked_code=None,
        token_present=token_present,
        report=report,
    )


def run_live_read_only(
    config: MondayProbeConfig,
    *,
    token_present: bool,
    now_ist: datetime,
    readiness_snapshot: UpstoxReadinessSnapshot | None,
    quotes: QuoteBatchResult | None,
    bod_rows: tuple[Mapping[str, object], ...],
    mis_rows: tuple[Mapping[str, object], ...],
    suspended_rows: tuple[Mapping[str, object], ...],
) -> MondayProbeResult:
    """Assemble live read-only evidence into a canonical report.

    Takes pre-fetched evidence only, so the token value never enters this
    module. When the token is absent the probe stops here with
    ``BLOCKED_TOKEN_MISSING`` and evaluates nothing.
    """

    if config.mode is not ProbeMode.LIVE_READ_ONLY:
        raise ValueError("run_live_read_only requires a LIVE_READ_ONLY config")
    if not token_present:
        return _finalize(
            mode=ProbeMode.LIVE_READ_ONLY,
            blocked_code=BLOCKED_TOKEN_MISSING,
            token_present=False,
            report=None,
        )
    trade_date = now_ist.date()
    try:
        calendar: CalendarEvidence | None = nse_cm_normal_session_calendar(
            start=trade_date, end=trade_date
        )
    except ValueError:
        calendar = None
    instrument = select_instrument(
        bod_rows=bod_rows,
        mis_rows=mis_rows,
        suspended_rows=suspended_rows,
        as_of_date=trade_date,
        tick_size_scale_rupees_per_raw_unit=config.tick_size_scale_rupees_per_raw_unit,
        instrument_key=config.instrument_key,
    )
    if instrument is None or instrument.cas_eligible is None:
        session_policy: NSEEquitySessionPolicy | None = None
    else:
        session_policy = NSEEquitySessionPolicy(
            cas_eligible=instrument.cas_eligible,
            exit_buffer_minutes=config.exit_buffer_minutes,
        )
    is_trading_day, session_open = derive_session_state(
        calendar=calendar,
        session_policy=session_policy,
        trade_date=trade_date,
        now_ist=now_ist,
    )
    tick = verify_tick(
        instrument=instrument,
        trade_date=trade_date,
        reference_price_rupees=config.tick_reference_price_rupees,
    )
    inputs = assemble_inputs(
        config=config,
        token_present=True,
        readiness_snapshot=readiness_snapshot,
        now_ist=now_ist,
        trade_date=trade_date,
        is_trading_day=is_trading_day,
        session_open=session_open,
        session_policy=session_policy,
        calendar_evidence=calendar,
        quotes=quotes,
        feed=config.feed,
        instrument=instrument,
        tick_verification=tick,
    )
    report = evaluate_live_market_readiness(inputs)
    return _finalize(
        mode=ProbeMode.LIVE_READ_ONLY,
        blocked_code=None,
        token_present=True,
        report=report,
    )


def _require_text(node: Mapping[str, object], field: str) -> str:
    value = node.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"evidence file is missing non-empty {field!r}")
    return value.strip()


def _require_decimal(node: Mapping[str, object], field: str) -> Decimal:
    value = node.get(field)
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"evidence file has invalid decimal {field!r}") from exc
    if not result.is_finite():
        raise ValueError(f"evidence file has non-finite decimal {field!r}")
    return result


def parse_cost_evidence_file(path: Path) -> CostReconciliationEvidence | None:
    """Strictly parse a cost-reconciliation evidence file; None fails closed downstream."""

    try:
        node = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(node, dict):
        return None
    try:
        orders_checked = node.get("orders_checked")
        passed_count = node.get("passed_count")
        failed_count = node.get("failed_count")
        if (
            not isinstance(orders_checked, int)
            or isinstance(orders_checked, bool)
            or orders_checked <= 0
        ):
            raise ValueError("orders_checked must be a positive integer")
        for label, count in (("passed_count", passed_count), ("failed_count", failed_count)):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        evidence = CostReconciliationEvidence(
            artifact_fingerprint=_require_text(node, "artifact_fingerprint"),
            schema_version=_require_text(node, "schema_version"),
            cost_model_name=_require_text(node, "cost_model_name"),
            orders_checked=orders_checked,
            passed_count=passed_count,
            failed_count=failed_count,
            max_reconciliation_error_inr=_require_decimal(node, "max_reconciliation_error_inr"),
            tolerance_inr=_require_decimal(node, "tolerance_inr"),
            status=_require_text(node, "status"),
        )
    except (ValueError, TypeError):
        return None
    if evidence.max_reconciliation_error_inr < 0 or evidence.tolerance_inr < 0:
        return None
    return evidence


def parse_paper_evidence_file(path: Path) -> PaperTradingEvidence | None:
    """Strictly parse a paper-trading evidence file; None fails closed downstream."""

    try:
        node = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(node, dict):
        return None
    try:
        session_start = date.fromisoformat(_require_text(node, "session_start"))
        session_end = date.fromisoformat(_require_text(node, "session_end"))
        verified = node.get("verified_orders_count")
        if not isinstance(verified, int) or isinstance(verified, bool):
            raise TypeError("verified_orders_count must be an integer")
        evidence = PaperTradingEvidence(
            artifact_fingerprint=_require_text(node, "artifact_fingerprint"),
            environment=_require_text(node, "environment"),
            session_start=session_start,
            session_end=session_end,
            verified_orders_count=verified,
            audit_log_fingerprint=_require_text(node, "audit_log_fingerprint"),
            source_reference=_require_text(node, "source_reference"),
        )
    except (ValueError, TypeError):
        return None
    return evidence


def _parse_daily_validation(node: Mapping[str, object]) -> DailyIntradayValidation:
    def text_or_none(field: str) -> str | None:
        value = node.get(field)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"daily validation field {field!r} must be text or null")
        return value

    def string_tuple(field: str) -> tuple[str, ...]:
        value = node.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"daily validation field {field!r} must be a string array")
        return tuple(value)

    row_count = node.get("row_count")
    duplicate_count = node.get("duplicate_count")
    continuous_rows = node.get("continuous_session_rows")
    cas_rows = node.get("cas_auxiliary_rows")
    for label, count in (
        ("row_count", row_count),
        ("duplicate_count", duplicate_count),
        ("continuous_session_rows", continuous_rows),
        ("cas_auxiliary_rows", cas_rows),
    ):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"daily validation field {label!r} must be a non-negative integer")
    session_rule = node.get("session_rule")
    if session_rule is not None and not isinstance(session_rule, dict):
        raise ValueError("daily validation field 'session_rule' must be an object or null")
    timezone = node.get("timezone")
    if timezone is not None and not isinstance(timezone, str):
        raise ValueError("daily validation field 'timezone' must be text or null")
    return DailyIntradayValidation(
        trade_date=date.fromisoformat(_require_text(node, "date")),
        row_count=row_count,
        first_timestamp=text_or_none("first_timestamp"),
        last_timestamp=text_or_none("last_timestamp"),
        duplicate_count=duplicate_count,
        continuous_session_rows=continuous_rows,
        cas_auxiliary_rows=cas_rows,
        cas_auxiliary_timestamps=string_tuple("cas_auxiliary_timestamps"),
        missing_expected_slots=string_tuple("missing_expected_slots"),
        unexpected_timestamps=string_tuple("unexpected_timestamps"),
        timezone=timezone,
        ohlcv_violations=string_tuple("ohlcv_violations"),
        session_rule=dict(session_rule) if session_rule is not None else None,
    )


def parse_historical_validation_file(path: Path) -> HistoricalDatasetValidation | None:
    """Strictly parse a historical-validation artifact; None fails closed downstream."""

    try:
        node = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(node, dict):
        return None
    try:
        rows = node.get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool):
            raise TypeError("historical validation field 'rows' must be an integer")
        raw_dates = node.get("trading_dates")
        if not isinstance(raw_dates, list) or any(not isinstance(item, str) for item in raw_dates):
            raise ValueError("historical validation field 'trading_dates' must be a string array")
        raw_days = node.get("per_day")
        if not isinstance(raw_days, list) or any(not isinstance(item, dict) for item in raw_days):
            raise ValueError("historical validation field 'per_day' must be an object array")
        raw_violations = node.get("structural_violations")
        if not isinstance(raw_violations, list) or any(
            not isinstance(item, str) for item in raw_violations
        ):
            raise ValueError("historical validation field 'structural_violations' must be text")
        manifest_ref = node.get("manifest_reference")
        if manifest_ref is not None and not isinstance(manifest_ref, str):
            raise ValueError("historical validation field 'manifest_reference' must be text")
        fingerprint_ref = node.get("manifest_fingerprint_reference")
        if fingerprint_ref is not None and not isinstance(fingerprint_ref, str):
            raise ValueError("manifest_fingerprint_reference must be text or null")
        fingerprint_schema = node.get("fingerprint_schema")
        if not isinstance(fingerprint_schema, str):
            raise TypeError("historical validation field 'fingerprint_schema' must be text")
        timezone = node.get("timezone")
        if timezone is not None and not isinstance(timezone, str):
            raise ValueError("historical validation field 'timezone' must be text or null")
        calendar_evidence = node.get("calendar_evidence")
        if calendar_evidence is not None and not isinstance(calendar_evidence, dict):
            raise ValueError("historical validation field 'calendar_evidence' must be an object")
        evidence = HistoricalDatasetValidation(
            rows=rows,
            trading_dates=tuple(date.fromisoformat(item) for item in raw_dates),
            per_day=tuple(_parse_daily_validation(item) for item in raw_days),
            structural_violations=tuple(raw_violations),
            deterministic_data_fingerprint=_require_text(node, "deterministic_data_fingerprint"),
            manifest_fingerprint_reference=fingerprint_ref,
            fingerprint_schema=fingerprint_schema,
            manifest_reference=manifest_ref,
            timezone=timezone,
            calendar_evidence=(dict(calendar_evidence) if calendar_evidence is not None else None),
        )
    except (ValueError, TypeError):
        return None
    return evidence


def result_summary(result: MondayProbeResult) -> dict[str, object]:
    """Return a credential-free one-line-style summary mapping (token boolean only)."""

    return {
        "schema_version": result.schema_version,
        "mode": result.mode.value,
        "blocked_code": result.blocked_code,
        "token_present": result.token_present,
        "classification": result.report.classification.value if result.report else None,
        "reason_codes": list(result.report.reason_codes) if result.report else [],
        "artifact_fingerprint": result.artifact_fingerprint,
        "live_orders_called": False,
    }


__all__ = [
    "BLOCKED_TOKEN_MISSING",
    "EXIT_BLOCKED_TOKEN",
    "EXIT_ERROR",
    "EXIT_LIVE_REVIEW",
    "EXIT_NOT_READY",
    "EXIT_RESEARCH_SHADOW",
    "EXIT_SHADOW_INFRA",
    "PROBE_SCHEMA_VERSION",
    "TOKEN_ENV_VAR",
    "MondayProbeConfig",
    "MondayProbeResult",
    "ProbeMode",
    "assemble_inputs",
    "blocked_token_result",
    "current_ist_now",
    "derive_session_state",
    "exit_code_for_result",
    "parse_cost_evidence_file",
    "parse_historical_validation_file",
    "parse_paper_evidence_file",
    "result_summary",
    "run_dry_run",
    "run_live_read_only",
    "select_instrument",
    "token_present_from_env",
    "verify_probe_fingerprint",
    "verify_tick",
    "write_probe_report",
]
